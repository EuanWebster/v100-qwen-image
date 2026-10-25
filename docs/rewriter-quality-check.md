# Rewriter quality check (round 3, d3)

A small, real check of the prompt rewriter through **this repository's own
modules** against the shared local model. Seven backend requests, sequential,
against `Qwen3.8` at `http://127.0.0.1:8079`.

This is a **quality sanity check, not an evaluation**: n = 1 per prompt, at
`temperature = 1.0`, on a service shared with other agents. Every number below
is a single sample.

> **Historical sample.** These seven answers were produced before the sampling
> fix described at the end (they ran at `presence_penalty = 0`; production now
> sends `1.5` for t2i). The measurements below are accurate for what was sent at
> the time and have **not** been re-run since. Treat this as a record of the
> pre-fix behaviour, not as a description of the current client.

## What was exercised

Nothing in the run re-implements the wire. The path is the production path:

| Stage | Module |
|---|---|
| Load + verify the system prompt | `qwen_engine.rewrite.load_system_prompt` (digest-pinned) |
| Build the conversation | `qwen_engine.rewrite.build_messages` |
| Send the request | `qwen_engine.rewrite_client.RewriteClient.complete` |
| Parse the answer | `qwen_engine.rewrite.parse_answer` |
| Orchestrate (what `/api/v1/rewrite` runs) | `qwen_engine.web.rewrite_service.RewriteService.submit` |

Runner: `output/gpu/rewriter_quality_d3.py` (`--dry-run` loads prompts and cases
and sends nothing).

### Inputs

- System prompt: `/tmp/pe_probe/prompts_system_prompt_t2i.txt`, md5
  `a5e1efc49aad097648079b734969045f` — **matches** `rewrite.SYSTEM_PROMPT_MD5["t2i"]`,
  so the run used the exact prompt the pin describes. Read in place; **not copied
  into this repository.**
- Examples: the three upstream t2i rows from
  `/tmp/pe_probe/data_t2i_example.jsonl`, read at runtime (not transcribed).
- Plus three prompts written for this check: exact rendered text, an exact count,
  and a non-English (Japanese) brief.

### Budget

7 of the 8 permitted requests were used; the eighth was left unspent.

## Results

| # | Case | Prompt (abbrev.) | Latency | `parse_ok` | Ratio | Fidelity |
|---|---|---|---|---|---|---|
| 1 | `t2i_1` (upstream) | 一只在雨中弹吉他的柯基 | 29.9 s | ✅ | `3:2` | PASS |
| 2 | `t2i_2` (upstream) | cyberpunk noodle stall at night, wide shot | 25.4 s | ✅ | `16:9` | PASS |
| 3 | `t2i_3` (upstream) | minimalist poster for a jazz festival | **190.9 s** | ✅ | `2:3` | PASS |
| 4 | `own_text` (own) | chalkboard reading "FRESH BREAD DAILY" | 21.6 s | ✅ | `2:3` | PASS |
| 5 | `own_count` (own) | three orange cats sleeping on a blue sofa | 19.0 s | ✅ | `3:2` | PASS |
| 6 | `own_japanese` (own) | 雨の夜、ネオンが反射する路地に座る黒猫 | 22.0 s | ✅ | `2:3` | PASS |
| 7 | `presence_penalty` probe | case 4 again, `presence_penalty=1.5` | 29.5 s | ✅ | `2:3` | PASS |

Latency: min 19.0 s, max 190.9 s, mean 48.3 s. **Case 3 is an outlier** — a
single 191 s stall on a shared service; it says nothing about the rewriter in
general but does mean the 240 s UI budget was only ~50 s from expiring.

`parse_ok` was **7/7**. No answer needed the unparsed-answer fallback, and every
ratio returned was a `wh_ratio` value drawn from the set the system prompt itself
enumerates (`1:1, 3:2, 2:3, 16:9, …`). There was no `ratio_follow` and no
`ratio_conflict` in this sample — that path stays unexercised by real answers.

### Fidelity, case by case (checked against the actual output text)

**`t2i_1` — Chinese brief → English description. PASS.** `corgi` ×6, `guitar` ×11,
`rain` ×9. Opens "a wide photorealistic photograph of a corgi playing a guitar in
rain". All three fixed facts from 一只在雨中弹吉他的柯基 (one corgi, in rain,
playing guitar) survived the language change, and no Chinese characters leaked
into the description.

**`t2i_2` — wide shot. PASS.** `16:9` was chosen, consistent with "wide shot" in
the brief; the description calls it "dense and cinematic".

**`t2i_3` — jazz poster. PASS.** `2:3`, a poster treatment, and the headline text
"JAZZ FESTIVAL" invented in keeping with the brief. Ratio and subject both fine.
This is the case that took 191 s.

**`own_text` — exact rendered text. PASS.** The description contains
`a bold white chalk headline reads "FRESH BREAD DAILY"` — the quoted string
survived **character for character, in capitals**, which is the whole point of
that check. Colours preserved: white chalk, black board, and the sign is described
as hanging by a door.

**`own_count` — exact count. PASS.** "three orange cats sleeping on a blue sofa"
appears near-verbatim in the opening. `orange` ×3, `blue` ×5. The three cats are
then enumerated as "the first cat… the second cat… the third cat", which is
enumeration of a set of three, not a contradiction — see the checker caveat below.

**`own_japanese` — non-English brief. PASS.** From Japanese alone the description
correctly extracted the subject: "a black cat sitting in a narrow rainy alley,
with wet asphalt reflecting magenta and cyan neon". This brief is outside the
upstream example distribution (those are Chinese and English), so it is mild
evidence of generality rather than a test of the prompt file.

**`presence_penalty` probe — accepted. PASS (acceptance only).** Sending
`presence_penalty = 1.5` through `RewriteClient.complete(..., extra_sampling=…)`
was accepted and produced a valid answer. **This proves wire acceptance, nothing
about quality**: one sample at temperature 1.0 cannot detect a distribution shift,
and upstream's own README warns a wrong value changes the distribution quietly
rather than failing.

### Contract compliance

The system prompt asks for ~400–500 words in ~20 sentences, present tense, third
person, no quality boosters, no instructions to a renderer, and exactly one
closing `The overall composition …` sentence.

| Check | Result |
|---|---|
| Word count | 404–562; only 3/7 land inside 400–500, four run 511–562 |
| Sentence count | 21–25 |
| Quality boosters (`masterpiece`, `8K`, `award-winning`, …) | none, 0/7 |
| Second person `you` | 0/7 |
| Instruction verbs | 2 hits, both false positives ("shapes … to create a mood", "raindrops create circles") |
| Exactly one closing sentence | yes, 7/7, all starting `The overall composition …` |
| Ratio/resolution restated in prose | none, 0/7 |

## Caveat: the fidelity checker had four bugs, all found by self-testing

These matter for how much weight the PASS column carries. Each was found by
feeding the checker hand-written answers before spending any request budget:

1. **`"13"` satisfied `"3"`** — digit matching was a bare substring. Fixed to
   require standalone digits.
2. **`"fourth kitten"` was missed** — `\bfour\b` does not match `fourth`, so an
   extra object went undetected. Ordinals added.
3. **`"one front paw"`, `"two of them"` flagged as wrong counts** — descriptive
   prose is full of incidental numbers. Fixed by requiring the number to sit
   adjacent to the counted noun ("three orange cats").
4. **`"the first cat"`, `"the second cat"` flagged as wrong counts** — ordinals
   here *enumerate* a set of three rather than add a fourth. Fixed by treating
   cardinal counts and `"a fourth cat"`-style indefinite introductions as the only
   count statements.

After fixes, all seven real answers pass and the checker still rejects synthetic
contradictions ("Two orange cats…", "Four orange cats…", "A fourth cat joins…",
"13 orange cats…"). The checker is a screening tool: it catches missing facts,
lost quoted text, and invented counts. It cannot judge composition quality, which
is why the PASS entries above name what was actually found rather than asserting
"good".

## Deviations and limitations to state wherever this is reported

- **These seven answers were produced *before* the sampling fix — they ran at
  `presence_penalty = 0`, not the trained 1.5.** At the time, the service
  transmitted only `temperature`, `top_p`, `max_tokens`; upstream's t2i profile
  specifies `presence_penalty = 1.5`, and its README warns a wrong value quietly
  changes the distribution. So this sample describes a *deviation from the
  trained profile*, and every PASS above is a result under that deviation.
  **This is now fixed on the production path**: `SAMPLING_WIRE_FIELDS` carries the
  whole profile (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
  `max_tokens`) and `tests/test_web_rewrite_service.py` pins the exact
  end-to-end request body for both tasks. The results here were **not** re-run
  after the fix, so their latencies, ratios and fidelity verdicts do not describe
  the current sampling. Expect a re-run to differ.
- **The probe above proves acceptance, not effect.** It showed the shared service
  *accepts* `presence_penalty = 1.5`; a separate check showed it *honours*
  vLLM extensions (`min_p = 1.0` produced byte-identical greedy output across two
  runs at `temperature = 1.0`). Neither is a quality measurement.
- **n = 1 per prompt**, `temperature = 1.0`. No answer here is reproducible; a
  different sample could lose a fact that this one kept.
- **Shared service.** Other agents were using the same backend, so latency
  reflects contention, and the 191 s outlier is unexplained.
- **Ratio coverage is thin.** All seven answers were `wh_ratio`; `ratio_follow`
  needs reference images, and reference-image rewriting is still gated at
  `422 task_unsupported`.
- **No image was generated.** This checks the text contract, not whether the
  resulting image matches the description.
- The upstream prompt and examples are read in place from `/tmp/pe_probe`; if that
  directory is cleaned, the script refuses at the digest check rather than
  proceeding with a drifted prompt.
