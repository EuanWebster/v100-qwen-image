"""Prompt-rewriting contract: task profiles, message building, answer parsing.

Pure stdlib and no I/O beyond reading the operator-supplied system-prompt file
(design ``docs/prompt-rewriter-plan.md`` §4.2, implementation split §5 R1).
Nothing here opens a socket, imports a model library or writes a file.

Upstream (``QwenLM/Qwen-Image-2.1`` ``prompt_rewrite/pe_core.py``) is the
source of the frozen sampling numbers and of the answer contract. Three
deliberate differences, each because upstream is a batch tool and this is a UI:

* **Exclusivity is enforced.** ``pe_core.parse_answer`` returns ``wh_ratio``
  *and* ``ratio_follow`` when the model emits both (it only clears
  ``ratio_follow`` for tasks without that field), so a contradictory answer
  survives into the record. A UI cannot show two competing canvas hints, so
  both are dropped and the conflict is flagged.
* **No raw-answer fallback into the prompt.** Upstream sets
  ``positive_prompt = raw answer`` on a parse failure, which for a UI would
  offer a thinking block as a prompt. Here the parse failure keeps
  ``prompt == ""`` and puts bounded raw text in ``unparsed_answer`` — displayed
  without an Apply button, never applied.
* **No ``json_repair``.** It rewrites malformed JSON, so a truncated 400-word
  paragraph would be silently "repaired" into different prose. The balanced
  brace scan handles every well-formed answer; anything else is reported as
  unparsed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from qwen_engine.errors import EngineError

#: The two rewriter tasks. Distinct fine-tunes with distinct system prompts;
#: upstream states they are not interchangeable.
TASKS: tuple[str, ...] = ("t2i", "edit")

#: Longest draft the UI may submit, in characters. Nothing in the tree capped
#: prompt length before this module (``index.html:48-50`` has no ``maxlength``
#: and ``config.py:251`` only rejects an empty string); the web body cap of
#: 40 KiB is derived from this value at 4 bytes/char.
PROMPT_MAX_CHARS = 8000

#: ``unparsed_answer`` is diagnostic text shown next to the draft, never
#: applied; 8 KiB of UTF-8 is enough to see what went wrong and is bounded so
#: the response stays inside the client's ``MAX_RESPONSE_BYTES``.
UNPARSED_ANSWER_MAX_BYTES = 8 * 1024

#: md5 of the upstream ``prompts/system_prompt_{t2i,edit}.txt``, which are
#: byte-identical to the ``system_prompt.txt`` inside the HF checkpoints. The
#: files themselves are Qwen-research-licensed and stay OUT of this repo, so
#: the pin is what proves the operator handed us the prompt the weights were
#: trained on. Identity pin, not a security primitive.
SYSTEM_PROMPT_MD5: Mapping[str, str] = {
    "t2i": "a5e1efc49aad097648079b734969045f",
    "edit": "493264f3621269df0f9ff727671fba28",
}

#: The frozen upstream ``PROFILES`` numbers, per task. ``presence_penalty`` is
#: the load-bearing difference (1.5 t2i / 0.0 edit): upstream's README warns a
#: wrong value "quietly changes the distribution" rather than failing.
_SAMPLING: Mapping[str, Mapping[str, Any]] = {
    "t2i": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_new_tokens": 16256,
        "image_max_pixels": None,
    },
    "edit": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "max_new_tokens": 24000,
        "image_max_pixels": 1024 * 1024,
    },
}

#: Which profile values are transmitted, and under which wire name. The full
#: profile travels: ``presence_penalty`` is the load-bearing difference between
#: the tasks (1.5 t2i / 0.0 edit) and upstream's README warns that a wrong value
#: changes the sampled distribution without failing loudly, so sending only the
#: OpenAI-standard fields silently ran t2i at penalty 0 -- a real deviation from
#: the trained profile, not a conservative default.
#:
#: ``top_k``/``min_p`` are vLLM extensions (not OpenAI fields) and go out at the
#: top level, which is what ``client.py`` puts in ``extra_body`` upstream. A
#: strict OpenAI-only server would refuse them; this client is built for the
#: vLLM-backed backend the profile came from, and :meth:`RewriteClient.complete`
#: surfaces the refusal as a typed error rather than silently dropping fields.
#:
#: ``image_max_pixels`` is deliberately absent: it is client-side preprocessing
#: (upstream downscales before sending), not a wire field.
SAMPLING_WIRE_FIELDS: Mapping[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "presence_penalty": "presence_penalty",
    "max_new_tokens": "max_tokens",
}

_WH_RATIO_RE = re.compile(r"[0-9]{1,4}:[0-9]{1,4}\Z")
_RATIO_FOLLOW_RE = re.compile(r"<image([1-9][0-9]{0,2})>\Z")
_DATA_IMAGE_PREFIX = "data:image/"

#: The chat template's thinking markers, assembled from pieces so this source
#: carries no literal markup tag. `split_thinking` is the only consumer.
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"


class RewriteError(EngineError):
    """Typed refusal of the rewriting contract (bad task/prompt/prompt file)."""


class ParsedAnswer(NamedTuple):
    """One parsed answer. Field order matches the plan's ``(prompt, ratio,
    parse_ok, conflict)`` plus the diagnostic raw text."""

    prompt: str
    ratio: dict[str, Any] | None
    parse_ok: bool
    ratio_conflict: bool
    unparsed_answer: str | None


def check_task(task: Any) -> str:
    """Return `task` if it is one of :data:`TASKS`, else raise."""
    if not isinstance(task, str) or task not in TASKS:
        raise RewriteError(
            f"unknown rewrite task {task!r}; choose one of {list(TASKS)}")
    return task


def sampling(task: str) -> dict[str, Any]:
    """The frozen sampling profile for `task`, as a fresh dict."""
    return dict(_SAMPLING[check_task(task)])


def check_prompt(prompt: Any) -> str:
    """Return `prompt` if it is a non-empty draft within the character cap."""
    if not isinstance(prompt, str):
        raise RewriteError(f"prompt must be a string, got {type(prompt).__name__}")
    if not prompt.strip():
        raise RewriteError("prompt must be non-empty")
    if len(prompt) > PROMPT_MAX_CHARS:
        raise RewriteError(
            f"prompt is {len(prompt)} characters; the cap is {PROMPT_MAX_CHARS}")
    return prompt


def _digest(raw: bytes, expected: str) -> str:
    """Hex digest of `raw` using the algorithm `expected` is a digest of
    (32 hex chars = md5, 64 = sha256). Anything else is a caller error."""
    if len(expected) == 32:
        return hashlib.md5(raw, usedforsecurity=False).hexdigest()
    if len(expected) == 64:
        return hashlib.sha256(raw).hexdigest()
    raise RewriteError(
        "pinned digest must be 32 hex chars (md5) or 64 (sha256), "
        f"got {len(expected)}")


def load_system_prompt(path: str | Path, task: str,
                       expected: str | None = None) -> str:
    """Read the operator-supplied system prompt for `task` and verify its pin.

    `expected` is a pinned digest (md5 or sha256, by length); it defaults to the
    upstream :data:`SYSTEM_PROMPT_MD5` value. A mismatch is a refusal, never a
    warning: a prompt that drifted from the weights fails fluently and silently,
    which is the one failure mode this pin exists to prevent.

    The error message names the file, so it is for the startup log only — do
    not forward it into an HTTP response (status reasons carry no paths).
    """
    check_task(task)
    prompt_path = Path(path)
    try:
        raw = prompt_path.read_bytes()
    except OSError as exc:
        raise RewriteError(
            f"system prompt for {task!r} is unreadable: {prompt_path} "
            f"({exc.strerror or type(exc).__name__})") from None
    want = expected if expected is not None else SYSTEM_PROMPT_MD5[task]
    got = _digest(raw, want)
    if got != want.lower():
        raise RewriteError(
            f"system prompt for {task!r} does not match its pinned digest "
            f"(got {got}, want {want.lower()}): {prompt_path}. Refusing to "
            "rewrite with a prompt that may not be the one the weights were "
            "trained on.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise RewriteError(
            f"system prompt for {task!r} is not valid UTF-8: {prompt_path}") from None
    text = text.strip()
    if not text:
        raise RewriteError(f"system prompt for {task!r} is empty: {prompt_path}")
    return text


def build_messages(system_prompt: str, user_prompt: str,
                   images: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Build the two-turn conversation for the rewriter.

    Text-only (t2i, and any edit rewrite with no references) produces plain
    string content — the shape the probe against the shared service verified.
    With `images` the user turn becomes a parts list, image parts first and in
    order, because the system prompt addresses them as ``<imageN>``: reordering
    silently re-points every reference in the rewrite. Each entry must be a
    ``data:image/...`` URI, so a draft can never make the backend fetch a URL.

    The parts shape is the one Root verified against the shared service on
    2026-10-26: a user turn of ``image_url`` parts followed by one ``text``
    part returned 200 and a correctly described image (0.6 s for a PNG
    downscaled to 512 px). Callers pass ``data:image/png;base64,`` URIs built
    from the reference store by ``web.rewrite_service``; this function does no
    encoding of its own and never touches the filesystem.
    """
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise RewriteError("system prompt must be a non-empty string")
    check_prompt(user_prompt)
    parts: list[dict[str, Any]] = []
    for i, image in enumerate(images or ()):
        if not isinstance(image, str) or not image.startswith(_DATA_IMAGE_PREFIX):
            raise RewriteError(
                f"image {i + 1} must be a data:image/... URI, got "
                f"{type(image).__name__ if not isinstance(image, str) else 'another URI scheme'}")
        parts.append({"type": "image_url", "image_url": {"url": image}})
    if parts:
        parts.append({"type": "text", "text": user_prompt})
        user_content: Any = parts
    else:
        user_content = user_prompt
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def split_thinking(text: str) -> tuple[str, str]:
    """Split a decoded generation into ``(thinking, answer)``.

    The chat template pre-fills the thinking marker before generation, so the
    answer section follows the first closing marker. An unterminated thinking
    block means the generation hit its token budget: there is no answer section.
    """
    if _THINK_CLOSE in text:
        think, _, answer = text.partition(_THINK_CLOSE)
        if _THINK_OPEN in think:
            think = think.partition(_THINK_OPEN)[2]
        return think.strip(), answer.strip()
    if _THINK_OPEN in text:
        return text.partition(_THINK_OPEN)[2].strip(), ""
    return "", text.strip()


def _balanced_spans(answer: str) -> list[str]:
    """Every balanced top-level ``{...}`` span in `answer`, in order.

    Braces inside JSON string literals are skipped, so a rewrite containing
    ``{`` does not end the span early; a brace in prose after the object does
    not stretch it past its real end (which a greedy match would).
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(answer):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(answer[start:i + 1])
    return spans


def _cap_utf8(text: str, limit: int) -> str:
    """Truncate `text` to at most `limit` UTF-8 bytes on a character boundary."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", "ignore")


def _parse_ratio(obj: Mapping[str, Any], task: str
                 ) -> tuple[dict[str, Any] | None, bool]:
    """``(ratio, conflict)`` from one answer object.

    A conflict is both fields populated (whatever their value): that is the
    model being ambiguous, and neither hint may then be shown. A populated but
    unusable value (``"banana"``, ``"<img1>"``) is dropped silently — it is a
    display hint we cannot render, not content, so the prompt is unaffected.
    """
    wh = obj.get("wh_ratio")
    wh_text = wh.strip() if isinstance(wh, str) else ""
    follow = obj.get("ratio_follow") if task == "edit" else None
    follow_text = follow.strip() if isinstance(follow, str) else ""

    if wh_text and follow_text:
        return None, True
    if wh_text and _WH_RATIO_RE.fullmatch(wh_text):
        return {"kind": "wh_ratio", "value": wh_text}, False
    match = _RATIO_FOLLOW_RE.fullmatch(follow_text) if follow_text else None
    if match is not None:
        return {"kind": "ratio_follow", "image": int(match.group(1))}, False
    return None, False


def parse_answer(answer: str, task: str) -> ParsedAnswer:
    """Parse the model's `answer` text into the task's declared fields.

    `answer` is the assistant message content, possibly still carrying a
    ``split_thinking`` preamble (the transformers path decodes it; the shared
    service normally reports it separately). Candidates are scanned last-first
    because the answer object is emitted at the end of the prose.
    """
    check_task(task)
    if not isinstance(answer, str):
        raise RewriteError(
            f"answer must be a string, got {type(answer).__name__}")
    raw_text = answer.strip()
    thinking, body = split_thinking(raw_text)
    candidates = list(reversed(_balanced_spans(body)))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # Some training runs mis-typed the key; accept both spellings.
        rewritten = obj.get("rewritten_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            rewritten = obj.get("rewrited_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        ratio, conflict = _parse_ratio(obj, task)
        return ParsedAnswer(prompt=rewritten.strip(), ratio=ratio, parse_ok=True,
                            ratio_conflict=conflict, unparsed_answer=None)
    # No usable answer object: keep the model's text for diagnosis only. The
    # answer section is preferred; with nothing after the closing marker
    # (budget hit) the thinking text is what there is to show.
    return ParsedAnswer(prompt="", ratio=None, parse_ok=False, ratio_conflict=False,
                        unparsed_answer=_cap_utf8(body or thinking or raw_text,
                                                  UNPARSED_ANSWER_MAX_BYTES) or None)
