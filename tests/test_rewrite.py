"""Rewrite contract (R1): profiles, prompt pinning, messages, answer parsing.

No sockets, no network, no torch: every input here is a string or a file in
``tmp_path``. The system-prompt files are tiny synthetics with their own
digests — the real (research-licensed) prompts are never copied into the repo.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from qwen_engine import rewrite as rw
from qwen_engine.errors import EngineError


def write_prompt(tmp_path, text: str, name: str = "prompt.txt"):
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def md5_of(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# Task profiles and caps
# ---------------------------------------------------------------------------


def test_tasks_are_the_two_upstream_rewriters():
    assert rw.TASKS == ("t2i", "edit")


@pytest.mark.parametrize("task", rw.TASKS)
def test_sampling_matches_the_frozen_upstream_profile(task):
    profile = rw.sampling(task)
    assert profile["temperature"] == 1.0
    assert profile["top_p"] == 0.95
    assert profile["top_k"] == 20
    assert profile["min_p"] == 0.0


def test_presence_penalty_and_budget_differ_per_task():
    """The upstream README calls the penalty difference load-bearing."""
    assert rw.sampling("t2i")["presence_penalty"] == 1.5
    assert rw.sampling("edit")["presence_penalty"] == 0.0
    assert rw.sampling("t2i")["max_new_tokens"] == 16256
    assert rw.sampling("edit")["max_new_tokens"] == 24000


def test_sampling_returns_a_copy():
    profile = rw.sampling("t2i")
    profile["temperature"] = 99.0
    assert rw.sampling("t2i")["temperature"] == 1.0


def test_wire_fields_are_a_subset_of_the_profile():
    for task in rw.TASKS:
        profile = rw.sampling(task)
        for name, wire in rw.SAMPLING_WIRE_FIELDS.items():
            assert name in profile, f"{name} is not a profile field"
            assert isinstance(wire, str) and wire
    # The three values that were actually exercised against the service.
    assert set(rw.SAMPLING_WIRE_FIELDS) == {"temperature", "top_p", "max_new_tokens"}


def test_unknown_task_is_a_typed_refusal():
    for bad in ("", "editing", "T2I", None, 3):
        with pytest.raises(rw.RewriteError):
            rw.check_task(bad)


def test_prompt_cap_boundary():
    assert rw.check_prompt("x" * rw.PROMPT_MAX_CHARS)
    with pytest.raises(rw.RewriteError, match="cap"):
        rw.check_prompt("x" * (rw.PROMPT_MAX_CHARS + 1))


@pytest.mark.parametrize("bad", ["", "   ", None, b"bytes", 7])
def test_prompt_must_be_a_non_empty_string(bad):
    with pytest.raises(rw.RewriteError):
        rw.check_prompt(bad)


# ---------------------------------------------------------------------------
# System prompt pinning
# ---------------------------------------------------------------------------


def test_pinned_md5_loads_and_strips(tmp_path):
    text = "You rewrite prompts. Contract: JSON only.\n"
    path = write_prompt(tmp_path, text)
    loaded = rw.load_system_prompt(path, "t2i", expected=md5_of(text))
    assert loaded == text.strip()


def test_pinned_sha256_also_accepted(tmp_path):
    text = "edit contract"
    path = write_prompt(tmp_path, text)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert rw.load_system_prompt(path, "edit", expected=sha) == text


def test_default_pin_rejects_a_foreign_prompt(tmp_path):
    """No explicit digest means the upstream one must match."""
    path = write_prompt(tmp_path, "some other prompt")
    with pytest.raises(rw.RewriteError, match="pinned digest"):
        rw.load_system_prompt(path, "t2i")


def test_digest_mismatch_names_the_file_not_the_content(tmp_path):
    text = "REFUSED-CONTENT-MARKER"
    path = write_prompt(tmp_path, text)
    with pytest.raises(rw.RewriteError) as info:
        rw.load_system_prompt(path, "t2i", expected=md5_of("different"))
    message = str(info.value)
    assert str(path) in message  # the operator needs the path in the startup log
    assert text not in message


def test_missing_prompt_file_is_refused(tmp_path):
    with pytest.raises(rw.RewriteError, match="unreadable"):
        rw.load_system_prompt(tmp_path / "absent.txt", "t2i")


def test_empty_and_non_utf8_prompt_files_are_refused(tmp_path):
    empty = write_prompt(tmp_path, "   \n", name="empty.txt")
    with pytest.raises(rw.RewriteError, match="empty"):
        rw.load_system_prompt(empty, "t2i", expected=md5_of("   \n"))

    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00prompt")
    with pytest.raises(rw.RewriteError, match="UTF-8"):
        rw.load_system_prompt(
            binary, "t2i",
            expected=hashlib.sha256(binary.read_bytes()).hexdigest())


def test_pinned_digest_must_be_a_digest(tmp_path):
    path = write_prompt(tmp_path, "x")
    with pytest.raises(rw.RewriteError, match="32 hex"):
        rw.load_system_prompt(path, "t2i", expected="deadbeef")


def test_unknown_task_refused_before_reading(tmp_path):
    with pytest.raises(rw.RewriteError):
        rw.load_system_prompt(tmp_path / "absent.txt", "video")


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def test_text_only_messages_use_plain_string_content():
    messages = rw.build_messages("system", "a cat on a stool")
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a cat on a stool"},
    ]


def test_images_come_first_and_in_order():
    messages = rw.build_messages(
        "system", "make it night", images=["data:image/png;base64,AAA",
                                           "data:image/png;base64,BBB"])
    parts = messages[1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].endswith("AAA")
    assert parts[1]["image_url"]["url"].endswith("BBB")
    assert parts[2]["text"] == "make it night"


def test_non_data_image_uris_are_refused():
    """A draft must never make the backend fetch a URL."""
    for bad in ("http://example.com/x.png", "file:/tmp/x.png", b"raw", 3):
        with pytest.raises(rw.RewriteError, match="data:image"):
            rw.build_messages("system", "edit this", images=[bad])


def test_message_building_enforces_the_prompt_contract():
    with pytest.raises(rw.RewriteError):
        rw.build_messages("system", "")
    with pytest.raises(rw.RewriteError, match="cap"):
        rw.build_messages("system", "x" * (rw.PROMPT_MAX_CHARS + 1))
    with pytest.raises(rw.RewriteError):
        rw.build_messages("   ", "a prompt")


def test_empty_image_sequence_is_text_only():
    assert rw.build_messages("s", "p", images=()) == rw.build_messages("s", "p")


# ---------------------------------------------------------------------------
# Thinking split
# ---------------------------------------------------------------------------

# Built by concatenation so this test file never contains a literal tag.
OPEN_THINK = "<" + "think>"
CLOSE_THINK = "</" + "think>"


def test_split_without_markers_returns_the_whole_text():
    assert rw.split_thinking("  plain answer  ") == ("", "plain answer")


def test_split_terminated_thinking():
    think, answer = rw.split_thinking(
        f"{OPEN_THINK}\nweighing options\n{CLOSE_THINK}\n\n{{'a': 1}}  ")
    assert think == "weighing options"
    assert answer == "{'a': 1}"


def test_split_unterminated_thinking_has_no_answer():
    think, answer = rw.split_thinking(f"{OPEN_THINK}\nstill going")
    assert think == "still going"
    assert answer == ""


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def test_clean_t2i_answer():
    parsed = rw.parse_answer(
        '{"rewritten_prompt": "A wide shot of a cat.", "wh_ratio": "16:9"}', "t2i")
    assert parsed.prompt == "A wide shot of a cat."
    assert parsed.ratio == {"kind": "wh_ratio", "value": "16:9"}
    assert parsed.parse_ok is True
    assert parsed.ratio_conflict is False
    assert parsed.unparsed_answer is None


def test_fenced_json_is_parsed():
    parsed = rw.parse_answer(
        'Here you go:\n```json\n{"rewritten_prompt": "Fenced.", "wh_ratio": "2:3"}'
        '\n```\nThanks!', "t2i")
    assert parsed.parse_ok and parsed.prompt == "Fenced."
    assert parsed.ratio == {"kind": "wh_ratio", "value": "2:3"}


def test_last_object_wins_when_prose_contains_braces():
    answer = ('The scene {roughly} is described. '
              '{"rewritten_prompt": "First."} '
              '{"rewritten_prompt": "Final.", "wh_ratio": "1:1"}')
    parsed = rw.parse_answer(answer, "t2i")
    assert parsed.prompt == "Final."


def test_braces_inside_the_string_do_not_end_the_span():
    parsed = rw.parse_answer(
        '{"rewritten_prompt": "A sign reading {OPEN} hangs", "wh_ratio": "3:4"}',
        "t2i")
    assert parsed.parse_ok
    assert parsed.prompt == "A sign reading {OPEN} hangs"


def test_unclosed_brace_after_the_object_does_not_swallow_it():
    parsed = rw.parse_answer('{"rewritten_prompt": "Kept."}\n\n{unclosed', "t2i")
    assert parsed.prompt == "Kept."


def test_misspelled_key_is_accepted():
    parsed = rw.parse_answer('{"rewrited_prompt": "Typo tolerated."}', "t2i")
    assert parsed.parse_ok and parsed.prompt == "Typo tolerated."


def test_empty_object_falls_back_to_an_earlier_valid_one():
    parsed = rw.parse_answer(
        '{"rewritten_prompt": "Good one."} {"rewritten_prompt": "   "}', "t2i")
    assert parsed.parse_ok and parsed.prompt == "Good one."


def test_both_ratios_on_edit_is_a_flagged_conflict():
    answer = json.dumps({"rewritten_prompt": "Edit it.",
                         "wh_ratio": "16:9", "ratio_follow": "<image1>"})
    parsed = rw.parse_answer(answer, "edit")
    assert parsed.parse_ok is True
    assert parsed.prompt == "Edit it."
    assert parsed.ratio is None
    assert parsed.ratio_conflict is True


def test_ratio_follow_is_read_for_edit():
    parsed = rw.parse_answer(
        '{"rewritten_prompt": "Crop in.", "wh_ratio": "", "ratio_follow": "<image2>"}',
        "edit")
    assert parsed.ratio == {"kind": "ratio_follow", "image": 2}
    assert parsed.ratio_conflict is False


def test_ratio_follow_is_ignored_for_t2i():
    parsed = rw.parse_answer(
        '{"rewritten_prompt": "Wide.", "ratio_follow": "<image1>"}', "t2i")
    assert parsed.parse_ok
    assert parsed.ratio is None
    assert parsed.ratio_conflict is False


@pytest.mark.parametrize("ratio", ["banana", "16-9", "16:9:", "<image0>", "", "16:9 extra"])
def test_unusable_ratio_keeps_the_prompt(ratio):
    parsed = rw.parse_answer(
        json.dumps({"rewritten_prompt": "Kept.", "wh_ratio": ratio}), "t2i")
    assert parsed.parse_ok is True
    assert parsed.prompt == "Kept."
    assert parsed.ratio is None


def test_non_string_ratio_is_dropped_not_raised():
    parsed = rw.parse_answer(
        '{"rewritten_prompt": "Kept.", "wh_ratio": 16.9}', "t2i")
    assert parsed.parse_ok and parsed.ratio is None


def test_prose_only_answer_is_unparsed_and_never_a_prompt():
    parsed = rw.parse_answer("I could not produce JSON, here is the scene instead.",
                             "t2i")
    assert parsed.parse_ok is False
    assert parsed.prompt == ""
    assert parsed.ratio is None
    assert parsed.unparsed_answer == "I could not produce JSON, here is the scene instead."


def test_unterminated_thinking_keeps_raw_text_for_diagnosis():
    parsed = rw.parse_answer(f"{OPEN_THINK}\nrun out of budget mid sentence", "t2i")
    assert parsed.parse_ok is False
    assert parsed.prompt == ""
    assert parsed.unparsed_answer == "run out of budget mid sentence"


def test_empty_answer_has_no_unparsed_text():
    for answer in ("", "   \n\t "):
        parsed = rw.parse_answer(answer, "t2i")
        assert parsed.parse_ok is False
        assert parsed.unparsed_answer is None


def test_object_nested_in_an_array_is_still_read():
    """The scan finds the object itself, whatever wraps it."""
    parsed = rw.parse_answer('[{"rewritten_prompt": "Nope."}]', "t2i")
    assert parsed.parse_ok is True and parsed.prompt == "Nope."


def test_answer_without_any_object_is_unparsed():
    parsed = rw.parse_answer("[1, 2, 3]", "t2i")
    assert parsed.parse_ok is False
    assert parsed.unparsed_answer == "[1, 2, 3]"


def test_unparsed_text_is_bounded_by_bytes_not_characters():
    answer = "ü" * 20000  # 2 bytes each in UTF-8
    parsed = rw.parse_answer(answer, "t2i")
    assert parsed.unparsed_answer is not None
    assert len(parsed.unparsed_answer.encode("utf-8")) <= rw.UNPARSED_ANSWER_MAX_BYTES
    assert parsed.unparsed_answer.startswith("ü")
    parsed.unparsed_answer.encode("utf-8")  # truncation never splits a character


def test_answer_must_be_a_string():
    for bad in (None, 3, b"bytes"):
        with pytest.raises(rw.RewriteError):
            rw.parse_answer(bad, "t2i")


def test_parse_errors_are_engine_errors():
    """The web layer catches EngineError; a stray ValueError would 500."""
    assert issubclass(rw.RewriteError, EngineError)
