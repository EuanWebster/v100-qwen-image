# Prompt rewriting (t2i + edit) — research and implementation plan

Status: research + design only, **no `src/` changes**. 2026-10-25, branch
`feature/v100-features` (D3). §5 is meant to be split by root. This is the design half of
§5 ("Prompt rewriting") in `docs/next-features-plan-2026-10-25.md`, and adopts its exit
criteria verbatim: useful rewriting, measured latency, preserved intent, no silent
changes, explicit acceptance of any proposed dimension, edit rewriting gated on
reference support.

Marks: *(verified)* = read in the tree, in an upstream file, or measured this session.
*(derived)* = arithmetic over verified inputs. *(infer)* = judgement, no measurement.
*(measure)* = must be measured before any number is cited or shipped.

## 0. What upstream ships *(verified: cloned `prompt_rewrite/` at /tmp/pe_probe; HF API)*

`QwenLM/Qwen-Image-2.1` → `prompt_rewrite/`: `pe_core.py` (378 L: profiles, message
building, answer parsing, record format), `client.py` (OpenAI-compatible client),
`run_vllm.py`, `run_transformers.py`, `serve.sh` (vLLM 0.19.1, transformers 5.4.0,
torch 2.10.0), `prompts/system_prompt_{t2i,edit}.txt` (10045 B/192 L, 18344 B/205 L),
`data/{t2i,edit}_example.jsonl` (3 text-only rows; 4 rows of 1, 1, 1 and 2 images).

Both prompt files are **byte-identical** to the `system_prompt.txt` inside the HF
checkpoints — md5 `a5e1efc49aad097648079b734969045f` (T2I), `493264f3621269df0f9ff727671fba28` (I2I).

**Two separate models, not one.** `Qwen/Qwen-Image-2.1-PE-T2I` and `-PE-I2I` are
distinct fine-tunes with distinct system prompts. The upstream README states they are
not interchangeable, that the answer contract "is part of what each model was trained
on", and that pointing at stock Qwen3.5-VL 9B "does not reliably emit the answer JSON —
expect `parse_ok: false` on most rows". A shared `pe_core.py` does not make them
swappable.

### 0.1 The rewriters *(verified: HF API, `config.json`, GGUF headers)*

- **9,409,813,744 parameters**, both checkpoints, 4 shards, 18.82 GB = **17.55 GiB** bf16.
- `model_type: qwen3_5` → `Qwen3_5ForConditionalGeneration`; `tie_word_embeddings false`.
  text: 32 layers, hidden 4096, 16 q / 4 kv heads, head_dim 256, vocab 248320, ctx
  262144, `layer_types` = **8 full_attention + 24 linear_attention** (hybrid, interval 4),
  ff 12288. vision: depth 27, hidden 1152, ff 4304, patch 16, out_hidden 4096.
  GGUF adds the SSM block: conv_kernel 4, state_size 128, group_count 16,
  time_step_rank 32, inner_size 4096.
- **License: Qwen RESEARCH LICENSE** — non-commercial; redistribution only if the
  agreement + Notice travel with the copies. A product constraint, not a footnote (§6).
- GGUF mirrors (`prithivMLmods/Qwen-Image-2.1-PE-{T2I,I2I}-GGUF`, GiB): BF16 16.69,
  Q5_K_M 6.02, Q5_K_S 5.87, Q4_K_M 5.24, Q4_K_S 4.98, Q3_K_L 4.59, Q3_K_M 4.31; I2I also
  ships `mmproj-bf16.gguf` 0.86 GiB (`qwen3vl_merger`, 456M).

### 0.2 Answer contract *(verified: both system prompts + `pe_core.parse_answer`)*

- **t2i**: brief in any language → **one English paragraph**, "about twenty sentences
  and four to five hundred words", ratio **only** in `wh_ratio`, never in the prose.
  `{"rewritten_prompt": "...", "wh_ratio": "3:2"}`, single line.
- **edit**: instruction + 1..N images → an editing directive. `wh_ratio` and
  `ratio_follow` (`"<imageN>"`, edit only) are mutually exclusive; one continuous
  paragraph; text to be *rendered in the image* goes in double quotes; two independent
  language decisions (prose language follows the instruction; rendered text has its own
  priority order). `{"rewritten_prompt": ..., "wh_ratio": ..., "ratio_follow": ...}`.
- **Sampling** (`pe_core.PROFILES`): temp 1.0, top_p 0.95, top_k 20, min_p 0.0,
  presence_penalty **1.5 (t2i) / 0.0 (edit)**, max_new_tokens **16256 / 24000**,
  `image_max_pixels` 1024×1024 **pixels**, thinking ON (`enable_thinking=True`). The
  README is explicit that the penalty difference is load-bearing and that a wrong value
  "quietly changes the distribution"; transformers has no native presence_penalty, so
  upstream reimplements vLLM semantics as a `LogitsProcessor`.
- **Parsing is lenient**: `split_thinking` on `</think>` (unterminated → empty answer),
  reverse-ordered balanced-brace scan (braces inside JSON strings skipped), optional
  `json_repair`, and it accepts the misspelling `rewrited_prompt`. On total failure it
  returns `positive_prompt = raw answer` with `parse_ok: false`.
- `serve.sh` defaults: `MAX_LEN 24576`, `MEM_UTIL 0.90`, `--reasoning-parser qwen3`.
  "~20 GB bf16", one GPU enough; a 24 GB card wants `--max-model-len 12000`.

## 1. This machine *(verified this session)*

- Shared LLM, OpenAI-compatible: `GET http://127.0.0.1:8079/v1/models` → `id "Qwen3.8"`,
  `max_model_len 262144`, `capabilities {chat, reasoning, image}`.
- **Three probes with the official system prompts** (upstream-style example briefs, no
  user drafts; the brief allowed at most three):

  | probe | wall | prompt tok | completion (reasoning) | parse |
  | --- | --- | --- | --- | --- |
  | t2i, Chinese brief | 38.2 s | 2487 | 4065 (3420) | JSON OK, `2:3` |
  | t2i, English brief | 29.2 s | ~2000 (2048 cached) | 2251 | JSON OK, 553 words, `16:9` |
  | edit + 1 image | 25.1 s | 5619 | 1175 | JSON OK, 3 fields, `ratio_follow "<image1>"` |

- Host: 122 GiB RAM, 16 threads (7800X3D 8C/16T), V100 idle at probe time. CPU measured:
  8-thread `sgemv` on a 640 MiB matrix (beyond L3, so RAM-streamed) **44.7 GiB/s /
  24 GFLOP/s**; 172 MiB matrix 85.5 GiB/s (partly cached); page-cached file read
  2.01 GiB/s single-thread.
- No llama.cpp here (no `llama-cli`/`llama-server`/`llama-cpp-python`); `ollama` 0.40.5
  with only `qwen3:14b` + `nomic-embed-text`. The venv has transformers **5.17.0** with
  `models/qwen3_5/` incl. `Qwen3_5ForConditionalGeneration`, so the HF path loads these
  checkpoints with no new packages.
- llama.cpp `master` knows `LLM_ARCH_QWEN35` (+`qwen35moe`, SSM tensors in the gguf-py
  map) but its mtmd vision enum has **no** qwen35 entry — the **I2I image path through
  llama.cpp is unverified**; t2i is text-only. The Q4_K_M mirror's GGUF header parses
  cleanly (`general.architecture = qwen35`, 427 tensors).
- Image generation peaks at **15,373–15,379 MiB** device VRAM on 1K presets
  (`docs/quality-suite-2026-10-24.md:39-44`); 2K "can still use almost all of the V100's
  memory" (`index.html:206`).

## 2. Local execution options

None is chosen here. Each has a real cost; the cheapest to *measure* should be measured
before any of them is built into the product.

### 2.a Official rewriter on CPU

Fit *(derived)*: 5.24–6.02 GiB of weights, plus KV for the 8 full-attention layers only —
8 × 2 × 4 × 256 × 2 B = **32 KiB/token** → 128 MiB at 4k tokens, ~600 MiB at the full
16k+2.5k budget. The 24 linear layers carry a constant SSM state (≈1 MiB/layer). Resident
≈ **6.5–7.5 GiB**, comfortable in 122 GiB.

Latency *(derived, wide bar)*: decode streams ~6 GiB/token. At the measured 44.7 GiB/s,
6.02/44.7 = 135 ms/token = **7.4 tok/s ceiling**; Q4 dequantisation costs 1.5–2× more
*(infer)* → **4–5 tok/s**. For the observed completion lengths (2251, 4065 — mostly
thinking) that is **7–17 min per rewrite**. Prefill 2·9.41e9·2487 ≈ 47 TFLOP → ≈79 s at
the 595 GFLOP/s `sgemm` peak *(derived)*, realistically 150–250 s *(infer)*. **CPU
inference is a background job, not a one-minute action** — the t2i/edit quality may
nonetheless survive.

**Must be measured** *(measure)*: a `llama.cpp` build on this box; confirm `qwen35` text
generation runs (SSM kernels, not silence); tok/s prefill+decode; RSS; cold-cache load
time; and **answer quality** on the upstream example rows against the recorded
local-service output. Quantisation damage to a 400–500-word prose contract is the real
risk, not speed.

> **Q4 policy.** The numbers above use Q4_K_M, and this project removed Q4 for the *image
> model* at the user's request. That decision was about image quality on a checkpoint
> where Q4 visibly degraded output; it does not automatically extend to a separate text
> rewriter, and it does not automatically forbid one either. **Root must rule.** If the
> "no Q4" rule is read as project-wide, the ceiling becomes Q5_K_M (6.02 GiB) — the
> latency numbers scale by 6.02/5.24 = 1.15× and every conclusion above is unchanged.
> Do not download any quant before that ruling.

### 2.b Official rewriter on the V100 between image jobs

bf16 is **17.55 GiB > 16 GiB**, and the V100 has neither BF16 nor FP8 *(project
constraint)*, so only a GGUF fits, fp16 compute *(derived)*. At ~900 GB/s HBM2,
5.63 GB/token ⇒ 6.3 ms/token ceiling; 50–90 tok/s realistic *(infer)* ⇒ **25–80 s** for
the observed lengths — the only interactive option with the official weights.

Cost: the card has **no** headroom for a co-resident rewriter (§1), so it would be
loaded per rewrite and released before the next generation job, putting load time into
every rewrite. It also violates the rule that GPU execution belongs to root: it must be a
**separate supervised process**, never something the web server starts.

**Must be measured** *(measure)*: CUDA build with `qwen35` (SSM layers may fall back to
CPU and destroy the claim), load time, VRAM residency, tok/s, and a hard release check.

### 2.c Official system prompt on the shared Qwen3.8 service

Measured and working (§1): 25–38 s worst case, strict JSON 3/3, no new weights, no GPU
time here. Caveats, all real:

- **Not the trained model.** It emits JSON because the prompt says so. 3/3 is not a
  quality result; *content* fidelity (counts, colours, rendered text, edit only the named
  attribute) is **unverified** against the PE checkpoints and can only be graded on the
  upstream example rows.
- **Privacy**: the draft prompt (and, for edit, instruction + images) leaves this
  application for a service other agents share, with logs we do not control. Hence
  **off by default, loopback-only, explicitly opted into**, and the UI states which
  backend answered.
- **Shared capacity**: tensor-parallel on two AMD GPUs that coding agents also drive; a
  38 s generation competes with them. Single-flight and cancellable on our side, or it
  becomes a queue nobody owns.
- `max_model_len 262144` is far above the 2487/5619-token prompts we send, so length is
  not a risk — but the model can change under us. The **response contract**, not the
  model name, is what the code must trust.

### 2.d Recommendation

Build the **plumbing** with a backend seam and a fake backend in tests (§4/§5), ship
**2.c** as the only initially selectable backend (it is the only one measured here), and
leave 2.a/2.b as measurements root runs on idle hardware. Endpoint, model and prompt
paths stay server-side configuration — never hard-wired.

## 3. UX contract (non-negotiable)

- **Explicit action only.** A `Rewrite` button in the prompt panel (`index.html:46-57`).
  Nothing on typing, blur, or load; never before Generate.
- **Original vs proposed.** The proposal appears in its own bounded block **below** the
  textarea (never replacing it in place), with the original shown above. While a rewrite
  is in flight the draft stays editable — editing does not cancel the request, it makes
  the answer stale.
- **Edit / Apply / Keep original.** *Edit* copies the proposal into the textarea and
  focuses it; *Apply* replaces the textarea content and records that a rewrite was applied
  (state only); *Keep original* discards it. Never automatic, never on a timer.
- **Stale replies are discarded.** The client sends a monotonic `draft_revision` and the
  reply must echo it *and* its `original_prompt` must equal the current textarea value —
  two independent conditions, because a counter alone misses a programmatic change. A
  mismatch drops the reply with a one-line `aria-live` note.
- **Failures leave the draft intact**: timeout, 502, 503, malformed JSON, contract
  violation → textarea untouched, proposal hidden, typed reason in the existing
  `#prompt-error` region. **No fallback**: no local "improver", no retry elsewhere, no
  partial application.
- **Ratio is opt-in.** A `wh_ratio` suggestion renders as text plus an "Apply ratio"
  button that selects the matching `PRESETS` entry (`app.js:138-223`; 1:1, 4:3, 3:4, 3:2,
  2:3, 16:9, 9:16, each in 1K and 2K). An unknown ratio is text with **no** button.
  `ratio_follow` is text only — this UI has no reference-image support yet, so it must
  not pretend to.
- **Backend honesty.** The block names the backend and the measured wall time. When
  rewriting is unconfigured the button is **absent** (not disabled-and-explained).
- **The help text must be re-checked.** `index.html:52-53` currently promises "Your words
  are used exactly as typed. Nothing is generated or auto-filled for you." That stays true
  only while rewriting is user-pressed; R4 must extend it to name the new action.
- Accessibility: labelled `role="group"`, announced once via `aria-live="polite"`, focus
  moves to the block on arrival, Escape returns focus to the textarea unchanged.

## 4. Engine and API design

### 4.1 Wire contract

New `POST /api/v1/rewrite` (`resolve_route`, `server.py:248-296`), added to the existing
`known` set so a wrong method still yields a typed 405 with `Allow`. Reuses
`header_policy` (`server.py:178-244`): same-origin, `X-Qwen-UI: 1`, no
`Transfer-Encoding`, no `Expect`, `application/json`, exactly one `Content-Length`.

Body cap **40 KiB**, enforced on this route in addition to the global 64 KiB header-stage
check (`server.py:242`, unchanged). Derived, not chosen: the new
`PROMPT_MAX_CHARS = 8000` (introduced by R1 — nothing in the tree caps prompt length
today, `index.html:48-50` has no `maxlength` and `config.py:251` only requires a non-empty
string) can be 4 bytes/char in UTF-8 = 32,000 B, plus ≤~1 KiB envelope. The client applies
the same formula locally, so the server cap is a backstop.

```json
{"schema": "qwen_engine.web_rewrite_request.v1", "task": "t2i",
 "prompt": "<draft>", "draft_revision": 7, "image_ids": []}
```

`draft_revision` is echoed verbatim; `image_ids` must be `[]` until edit support lands.

```json
{"schema": "qwen_engine.web_rewrite.v1", "task": "t2i", "draft_revision": 7,
 "original_prompt": "<exactly what was sent>", "rewritten_prompt": "<one paragraph>",
 "parse_ok": true, "unparsed_answer": null,
 "ratio": {"kind": "wh_ratio", "value": "16:9"}, "ratio_conflict": false,
 "backend": "local-llm", "elapsed_ms": 24210}
```

- `ratio` is `null` when none, `{"kind": "ratio_follow", "image": 1}` for edit, and
  `ratio_conflict: true` when the model set both. Exclusivity is **stricter than
  upstream**: `pe_core.parse_answer` returns `wh_ratio` *and* `ratio_follow` when both
  are emitted (it only clears `ratio_follow` for tasks without that field), so a
  contradictory answer silently survives into the batch record. A UI cannot show two
  competing hints, so here the conflict is flagged, the prompt kept, both ratios dropped.
- **The server does not map ratio → preset.** `PRESETS` in `app.js` is the single source
  of width/height; the response carries the ratio string and R4 matches it exactly.
- **Unparsed answers do not become suggestions.** Upstream's fallback
  (`positive_prompt = raw answer`, `parse_ok: false`) is right for a batch and wrong for a
  UI, where it would offer a thinking block as a prompt. Here the raw text goes in
  `unparsed_answer` (capped **8 KiB**, only when `parse_ok` is false),
  `rewritten_prompt` is empty, and the UI shows it with **no Apply button**. This is a
  **200**: the round trip succeeded. `502 rewrite_invalid_answer` is reserved for a
  transport violation — not a chat completion, no `choices`, non-string `content`.
- No `json_repair` dependency: the balanced-brace scan handles well-formed answers, and a
  repair library "fixing" a truncated 400-word paragraph is silent content change.
- **Size is bounded by construction and the client cap binds first.** Worst honest case is
  `original_prompt` 32 KiB + a 24000-token `rewritten_prompt` (≈96 KB), and
  `unparsed_answer` is present only when `rewritten_prompt` is empty — so ≤ ~128 KiB,
  inside the client's existing `MAX_RESPONSE_BYTES = 262144` (`app.js:509`). The **1 MiB
  server cap is a backstop against a misbehaving backend**, not a size the UI must handle;
  R4 must not raise `MAX_RESPONSE_BYTES` for this route.

Typed failures (existing `ApiError`, messages `_safe`-bounded at 200 chars, **never
containing prompt text**): `503 rewrite_disabled`, `503 rewrite_busy`,
`504 rewrite_timeout`, `502 rewrite_unavailable`, `502 rewrite_invalid_answer`,
`422 prompt_too_long`, `422 task_unsupported`, `400 invalid_envelope`.

`GET /api/v1/status` gains one **additive, optional** block:

```json
"rewrite": {"state": "available|unavailable", "code": "rewrite_available|rewrite_disabled|rewrite_unconfigured",
            "backend": "local-llm|official-cpu|official-gpu|null", "reason": "<one line, no host:port, no path>"}
```

**`GATE_STATES` does not change**, and `WEB_STATUS_SCHEMA` is **not bumped**.
`GATE_STATES` (`app.js:59-63`) is the closed set of *generation* gates and `parseStatus`
rejects any document whose gate-key count differs (`app.js:659`); rewriting is an
optional capability that must never block generating, so it cannot join that predicate.
R4 gets its own strict parser for the block (exact keys, closed state/code pairs,
otherwise `null` → the action is not offered). No bump keeps R3 and R4 disjoint: a bump
would force `api.py:40`, `app.js:51` and `tests/test_web_api.py:52` to change together.

### 4.2 Modules

- **NEW `src/qwen_engine/rewrite.py`** — pure, stdlib-only, no sockets/ML/writes.
  `TASKS`, `PROMPT_MAX_CHARS = 8000`, `SYSTEM_PROMPT_MD5`,
  `load_system_prompt(path, task)` refusing on hash mismatch (project style: explicit
  refusal over silent drift), `build_messages`, `parse_answer` (thinking split, reverse
  brace scan, `rewrited_prompt`, exclusivity → `(prompt, ratio, parse_ok, conflict)`),
  `sampling(task)` (the frozen `PROFILES` numbers).
- **NEW `src/qwen_engine/rewrite_client.py`** — the only outbound HTTP in the codebase.
  Stdlib `http.client`, no redirects, proxy env ignored, fixed connect+read timeout,
  streaming read with a hard **1 MiB** cap, and a **loopback-only** target check
  (`127.0.0.0/8` / `::1`; anything else is a startup error, never a runtime retry).
  Transport is injected (`send(bytes) -> bytes`) so tests never open a socket — the CPU
  guard denies socket construction outright (`tests/cpu_guard/sitecustomize.py:16-20`).
- **`web/api.py`** — `REWRITE_REQUEST_SCHEMA`, `WEB_REWRITE_SCHEMA`,
  `build_rewrite_request(raw_body)` (strict JSON, exact keys, non-negative int revision,
  non-empty prompt ≤ cap, `image_ids == []`), `rewrite_response(...)`, and the status
  block. **`EDITABLE_FIELDS` untouched** — a suggestion is not config and must never
  enter the plan/export envelope.
- **`web/server.py`** — route + service construction from flags:
  `--rewrite-backend {off,local-llm}` (default `off`), `--rewrite-endpoint`,
  `--rewrite-model`, `--rewrite-prompt-{t2i,edit}`, `--rewrite-timeout` (default 180 s).
  Served only when configured; otherwise `503 rewrite_disabled`.
- **NEW `web/rewrite_service.py`** — one slot, one worker: `submit()` under a
  `threading.Lock` (second caller → `503 rewrite_busy`, never a growing queue), a cancel
  `Event`, monotonic `elapsed_ms`, and the client call run off the HTTP worker so the 4
  handler threads (`MAX_CONCURRENT_HANDLERS`, `server.py:85`) are released. It never
  touches `JobController` or the executor lease: a rewrite must not queue behind or stall
  an image job.

### 4.3 Timeouts, caps, logging

- **Timeout** 180 s default (worst measured 38 s). A CPU or V100-GGUF backend needs it
  raised explicitly — a flag decision, not an accident. Expiry → `504`, draft untouched.
- **Caps**: 40 KiB body, 8000 chars prompt, 1 MiB response, one in-flight rewrite,
  `max_new_tokens` from `sampling(task)` — the rewriter's own budget.
- **No prompt logging.** The server already silences request/error lines
  (`server.py:378`). The rewrite path must also never pass the prompt to `PhaseLogger`,
  `performance.json` or an exception message; log only
  `{task, backend, elapsed_ms, prompt_chars, parse_ok}`. The one copy of the prompt is
  `original_prompt` in the response, bounded by the cap, so the UI can prove staleness.
- **Non-blocking.** `CONNECTION_TIMEOUT_SECONDS = 30` (`server.py:79`) is a **per-socket
  idle timeout** (`server.py:374`), not a total request deadline, so a long outbound call
  does not race it; the inbound body is fully read before dispatch.

## 5. Implementation split (disjoint ownership)

§4 shapes are frozen; R3/R4 can start against them with a fake client.

- **R1 contract** — NEW `src/qwen_engine/rewrite.py` + `tests/test_rewrite.py`. Prompt
  loading and md5 refusal; message building; `parse_answer` over clean JSON, fenced JSON,
  trailing prose, inline `think`, misspelled key, both ratios, unterminated thinking
  (`parse_ok False` + raw text); `sampling` values; the char cap. No I/O, no sockets.
- **R2 transport** — NEW `src/qwen_engine/rewrite_client.py` + `tests/test_rewrite_client.py`.
  Injected transport: non-loopback target refused at construction, oversize response
  refused, short read and deadline expiry typed, proxy env not consulted, request bytes
  exactly the JSON body.
- **R3 web** — `web/api.py`, `web/server.py`, NEW `web/rewrite_service.py` +
  `tests/test_web_rewrite.py`. Route and header policy, 405/404, caps, single-flight
  `503 rewrite_busy`, `504` on timeout, `rewrite_disabled` when off, status block, and
  leak guards (prompt absent from error payloads and logs; `EDITABLE_FIELDS` unchanged).
- **R4 UI** — `web_static/{index.html,app.js,app.css}` + additions to
  `tests/test_web_static.py`. Button, proposal block, revision counter, stale discard,
  Edit/Apply/Keep original, ratio→`PRESETS` lookup, a **separate** rewrite fetch timeout
  (shared `FETCH_TIMEOUT_MS` stays 15 s; rewrite needs ≥240 s), failure text, help-copy
  re-check. Two existing pins must be *extended*, never weakened: the exact
  `(constant, route)` list at `tests/test_web_static.py:133-137` (add
  `("API_REWRITE", "/api/v1/rewrite")`, nothing else) and the "no `capability` string"
  assertion (`:127`). `GATE_STATES` is not extended; the block gets its own parser+test.

Order: R1 → (R2 ∥ R4-mock) → R3 → R4. Whoever lands last runs the guarded CPU suite once.
**No test may open a socket or import torch**; the fake backend is a callable and the real
endpoint is exercised only by root.

## 6. Open questions for root

1. **Which backend ships on?** Only 2.c is measured; 2.a/2.b need a llama.cpp build and
   (2.b) exclusive V100 time. This plan assumes 2.c stays **opt-in** for the privacy and
   capacity reasons in §2.c — confirm, or accept the shared-service trade.
2. **Prompt files and weights stay out of the repo.** The two `system_prompt.txt` files
   (28 KB total) are Qwen-research-licensed, so vendoring them means shipping the licence
   + Notice; otherwise they are operator-supplied and hash-pinned at startup (the design
   above). The 17.55 GiB bf16 / 5.24 GiB Q4_K_M weights are never vendored either —
   non-commercial licence, downloads stay root's.
3. **Edit rewriting** depends on reference-image support
   (`docs/image-edit-design.md` §7 T1-T4). Until then `task: "edit"` with non-empty
   `image_ids` is `422 task_unsupported`; route and parser are built for it now so no
   later schema change is needed.
4. **Ratio must never touch width/height automatically** — explicit "Apply ratio" only.
   Confirm.
