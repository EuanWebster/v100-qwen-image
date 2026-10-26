# Transparent backgrounds

Opt-in support for generating images with a transparent background and saving
them with a real alpha channel. It is **prompt-level guidance plus an
alpha-capable container** — not a guaranteed cutout. No claim is made here
about how reliably the pinned model produces transparency; that is checked on
the GPU (see [Acceptance](#acceptance-still-root-owned)).

Verified on CPU only: `tests/test_transparency.py`, `tests/test_web_api.py`,
`tests/test_web_static.py` and `tests/test_image_output.py` in
`run_cpu_tests.sh`.

## Contract

`GenerationConfig` (frozen) gains one strict-bool field:

* `transparent_background: bool = False` — non-bool values are a `ConfigError`
  on field `transparent_background`. Old JSON configs without the key default
  to `False`; `to_dict`/`to_json`/`from_dict`/`from_json` round-trip the bool.

Derived, never stored:

* `effective_prompt` — with the flag `True`, exactly
  `TRANSPARENCY_PROMPT_PREFIX + prompt + TRANSPARENCY_PROMPT_SUFFIX`
  (`"This is an RGBA image with transparency. "` … `" The image has alpha
  channel and the background is transparent."`); otherwise `prompt` verbatim.
  Pure function of frozen fields: repeated calls never mutate, accumulate, or
  normalize whitespace. `config.prompt` is always preserved unchanged.
* `TRANSPARENCY_WRAPPER_REVISION = "qwen-engine.transparent-wrapper.v1"` —
  named identity of the exact wrapper text, reserved for metadata and audit
  records. It changes no behaviour.

Encoder binding:

* `encoder_identity.project_encoder_request` sends `config.effective_prompt`
  as the positive `EncoderRequest.prompt`, so the encoder cache key and the
  key-document evidence bind the actual wrapper text. Negative prompt,
  `cfg_active`, dtype, model identity and the encoder used are unchanged.
  Transparent and non-transparent requests therefore get different cache keys;
  identical configs get identical keys.

Format policy:

* Alpha cannot live in JPEG. `transparent_background=True` requires
  `output_format` in `ALPHA_OUTPUT_FORMATS` (`png`, `webp`); `jpg` (including
  the default) is refused with a `ConfigError` on `output_format`. The format
  is never silently rewritten by the engine, the API or the CLI.
* `image_output.save_output_image` already encodes PNG as-is and WebP
  lossless, so RGBA survives byte-exactly; the opaque RGB path is unchanged
  (pinned by a byte-identity test). JPEG flattens onto white, unchanged.

## CLI

* `--transparent / --no-transparent` (`BooleanOptionalAction`, default `None`
  → the field is only mapped when the flag is given).
* `--transparent` with **no explicit** `--output-format` auto-selects `png`.
  An explicit `--output-format jpg` (or a config-file `jpg`) is **refused**,
  never silently rewritten.
* `--no-transparent` overrides a config-file `true` and keeps the current
  format.

## Web API

`transparent_background` is part of the editable config envelope
(`EDITABLE_FIELDS`), so it flows through `/api/v1/plan`, `/api/v1/config/export`
and `/api/v1/config/import` with the same strict validation:

* Non-bool → `422 invalid_config` with a field error on
  `transparent_background`.
* `true` with `output_format: jpg` → `422 invalid_config` with the field error
  on `output_format` (the message names the alpha-capable formats).
* An envelope or export file without the key imports as `false`.

## Web UI

* A **Transparent background** checkbox with plain-language help sits above
  the seed row; its state, explanation and the exact model-visible prompt are
  exposed to assistive tech through `aria-describedby`.
* Ticking it while the format is `jpg` switches the format to `png` and keeps a
  visible explanation of why. Unticking never rewrites the user's format.
* "Text the model reads (read-only)" shows `TRANSPARENT_PREFIX + prompt +
  TRANSPARENT_SUFFIX`; the prompt field itself is never modified.
* The server plan panel gained a `Transparent background` row echoing the
  validated value and the saving format.
* Config import/export round-trips it. An imported (server-validated) plan is
  applied verbatim — import never triggers the local auto-switch.
* Transparent previews and history thumbnails render on a checkerboard so
  alpha reads as alpha. Preview only: the saved file is never altered.

## Acceptance (still root-owned)

1. A real GPU run producing a PNG/WebP with a genuinely varying alpha channel
   (pixel-level check).
2. Upstream wrapper quality on the pinned model — no quality claim is made
   from CPU tests.
3. Stamping `transparent_background` and `TRANSPARENCY_WRAPPER_REVISION` into
   per-job metadata (`jobs.py` / `web/generation.py` own that shape).
