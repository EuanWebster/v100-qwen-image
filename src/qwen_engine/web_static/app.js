/* Lightseed Studio — W3 front-end (local same-origin engine jobs).
 *
 * Honest by construction:
 *   * Network calls use the local status/plan/config/export and job API. Opened over
 *     file:// (or without the qwen-web server) every fetch fails and the UI
 *     says so: "Not connected" — failures are never swallowed as success.
 *   * Generation requires explicit server enablement and per-job engine checks.
 *   * The server is authoritative for GenerationConfig validation. The
 *     lightweight form checks here exist only to give accessible inline
 *     feedback; they never replace the server's verdict.
 *   * Plan/status seeds are decimal STRINGS (BigInt-handled, full uint64,
 *     never coerced through JS Number). The server NEVER sends a numeric
 *     seed in plan/status; a plan response with a numeric seed is rejected
 *     as an unrecognized schema rather than rendered.
 *   * Export downloads RAW response bytes as a blob — never JSON.parsed
 *     and reserialized — and only after the server has validated the
 *     config authoritatively (runtime paths are excluded server-side).
 *   * No fake results, jobs, timings or thumbnails. No draft autosave,
 *     no storage APIs. User text and server values are only ever inserted
 *     via textContent — markup is never injected as parsed HTML.
 *   * Every request is bounded END TO END: one AbortController timer covers
 *     headers AND body consumption (a slow or truncated body can never hang
 *     the page), and bodies are read with a fixed size cap — never
 *     unbounded. A malformed/truncated-but-complete JSON body is a schema
 *     REJECTION (chip: "Status rejected"); an unreachable or timed-out
 *     request is a network failure (chip: "Not connected") — never mixed.
 *   * A visible plan result is invalidated BEFORE the next attempt is
 *     validated, and becomes clearly STALE as soon as the draft no longer
 *     matches the validated snapshot (including preset/random/change-only
 *     edits). An out-of-order answer never wins and never clears newer
 *     validation errors.
 */
'use strict';

(function () {
  var SEED_MIN = 0n;
  var SEED_MAX = 18446744073709551615n; // 2^64 - 1
  var SEED_MAX_DIGITS = SEED_MAX.toString().length; // 20

  var GENERATION_UNAVAILABLE_REASON =
    'Connect to a qwen-web server started with --enable-generation to generate images.';

  /* ---- Same-origin API contract (relative paths only: never a host,
   *      never an absolute URL, never a CDN) ---- */
  var API_STATUS = '/api/v1/status';
  var API_PLAN = '/api/v1/plan';
  var API_EXPORT = '/api/v1/config/export';
  var API_IMPORT = '/api/v1/config/import';
  var API_JOBS = '/api/v1/jobs';
  var API_REFERENCES = '/api/v1/references';
  var API_REWRITE = '/api/v1/rewrite';
  // The rewriter choice is a SERVER setting (Settings page): one document in
  // the server's cache dir, shared by every browser. The page never learns —
  // and never sends — a filesystem path.
  var API_REWRITE_SETTINGS = '/api/v1/settings/rewrite';
  var API_REWRITE_SETTINGS_TEST = API_REWRITE_SETTINGS + '/test';
  var PLAN_REQUEST_SCHEMA = 'qwen_engine.web_plan_request.v1';
  var STATUS_SCHEMA = 'qwen_engine.web_status.v1';
  var PLAN_SCHEMA = 'qwen_engine.web_plan.v1';
  var ERROR_SCHEMA = 'qwen_engine.web_error.v1';
  var REFERENCE_SCHEMA = 'qwen_engine.web_reference.v1';
  var REWRITE_REQUEST_SCHEMA = 'qwen_engine.web_rewrite_request.v1';
  var REWRITE_SCHEMA = 'qwen_engine.web_rewrite.v1';
  var REWRITE_SETTINGS_SCHEMA = 'qwen_engine.web_rewrite_settings.v1';
  var REWRITE_TEST_REQUEST_SCHEMA = 'qwen_engine.web_rewrite_test_request.v1';
  var REWRITE_TEST_SCHEMA = 'qwen_engine.web_rewrite_test.v1';
  var UI_HEADER = 'X-Qwen-UI';

  // Reference images (design §5). The caps MIRROR the server contract
  // (config.py REF_MAX = 10, reference_store.MAX_REFERENCE_BYTES = 32 MiB,
  // MAX_REFERENCE_DIM = 4096) so the page refuses locally with the same
  // reason the server would give — the server stays authoritative and its
  // typed error is surfaced verbatim when it refuses anyway.
  var MAX_REFERENCES = 10;
  var MAX_REFERENCE_BYTES = 32 * 1024 * 1024;
  var MAX_REFERENCE_DIM = 4096;
  var REFERENCE_ID_RE = /^[0-9a-f]{64}$/;
  var REFERENCE_ACCEPT = 'image/png,image/jpeg,image/webp';

  // Closed gate contract: each gate has exactly one truthful state and one
  // stable code (qwen_engine.web/api.py). An arbitrary bounded string is a
  // schema rejection, never a trusted state.
  var GATE_STATES = {
    adapter:     { states: ['available'],       code: 'adapter_available' },
    hardware:    { states: ['unknown'],         code: 'hardware_not_probed' },
    assets:      { states: ['not_verified'],    code: 'assets_unverified' },
    calibration: { states: ['unknown'],         code: 'calibration_unknown' }
  };

  // CLOSED blocker-code list — exactly what the real server can emit.
  var KNOWN_BLOCKER_CODES = [
    'generation_disabled', 'runtime_unconfigured', 'manifest_unconfigured', 'manifest_file_missing',
    'manifest_invalid', 'manifest_unreadable'
  ];

  /* Rewrite availability (plan §4.1 plus the two-backend amendment). CLOSED
     state/code pairs: the code is the key, the state is the value it must pair
     with. Rewriting is optional and NEVER a generation gate, so it is
     deliberately absent from GATE_STATES. */
  var REWRITE_STATES = {
    rewrite_available: 'available',
    rewrite_disabled: 'unavailable'
  };
  // The closed key set of the status block: the four core keys, plus the ones
  // the per-task server adds. A key outside this list is an unread contract,
  // not a thing to ignore.
  var REWRITE_BLOCK_KEYS = ['state', 'code', 'backend', 'reason', 'backends',
                            'settings', 'active_by_task'];
  var REWRITE_BACKEND_ROW_KEYS = ['id', 'label', 'tasks', 'uses_gpu'];
  var REWRITE_BACKEND_ID_RE = /^[a-z0-9][a-z0-9._-]{0,63}$/;
  // The Settings document's own closed sets: the state, one saved choice and
  // one per-task group.
  var REWRITE_SETTINGS_KEYS = ['schema', 'enabled', 'choices', 'saved', 'local',
                               'backends', 'tasks', 'off', 'persistent', 'note'];
  // The saved choice is PER TASK plus one on/off switch; the block describing
  // the shared endpoint selects nothing, so it carries no choice keys. Two
  // documents, two closed key sets.
  var REWRITE_CHOICE_KEYS = ['t2i', 'edit', 'enabled', 'endpoint', 'model',
                             'timeout_s'];
  var REWRITE_LOCAL_KEYS = ['endpoint', 'model', 'timeout_s'];
  var REWRITE_SETTINGS_ROW_KEYS = ['id', 'label', 'tasks', 'uses_gpu',
                                   'available', 'reason'];
  // One radio group per draft kind: `{task, label, choices[]}`, each choice
  // `id/label/available/uses_gpu/reason` — the server has already decided what
  // can be chosen here and why not, so the page never guesses.
  var REWRITE_TASK_ROW_KEYS = ['task', 'label', 'choices'];
  var REWRITE_TASK_CHOICE_KEYS = ['id', 'label', 'available', 'uses_gpu',
                                  'reason'];
  var REWRITE_SETTINGS_BACKENDS = ['pe-llamacpp', 'local-llm', 'off'];
  // The two rewriters that can be CHOSEN for a draft kind. `off` is not one of
  // them: it is the switch below the groups, not a rewriter.
  var REWRITE_BACKEND_IDS = ['pe-llamacpp', 'local-llm'];
  // Every label the rewriter block carries is pinned here, exactly as the
  // server publishes it. A block whose labels have drifted is a block this page
  // was not written for: rendering it would put words on the screen about a
  // rewriter nobody chose, which is how a settings page starts lying.
  var REWRITE_TASK_LABELS = { t2i: 'Prompt enhance (text-to-image)',
                              edit: 'Edit prompt enhance' };
  var REWRITE_BACKEND_LABELS = {
    'pe-llamacpp': 'Official enhancer (Q8, uses GPU)',
    'local-llm': 'OpenAI-compatible endpoint',
    'off': 'Off'
  };
  var REWRITE_BACKEND_USES_GPU = { 'pe-llamacpp': true, 'local-llm': false,
                                   'off': false };
  // Sentinel meaning "present but malformed" — distinct from null
  // ("absent or legitimately unavailable"), so a bad block rejects the
  // whole status document instead of silently hiding a feature.
  var INVALID = { invalid: true };

  var REWRITE_TASKS = ['t2i', 'edit'];
  // The server's rewrite prompt cap (rewrite.PROMPT_MAX_CHARS). Refused
  // locally so an over-long draft is never sent; the server re-checks.
  var PROMPT_MAX_CHARS = 8000;
  // Local generation of a full paragraph is slow; the short general-purpose
  // deadline would abort a working rewrite. Still finite.
  var REWRITE_TIMEOUT_MS = 240000;
  // Manifest failures always block submission. Hardware/assets/calibration
  // evidence stays separate: enabling submission is not a memory-fit claim.
  var MANIFEST_REASON_CODES = [
    'manifest_unconfigured', 'manifest_file_missing', 'manifest_invalid',
    'manifest_unreadable'
  ];

  // Public string bound, ALIGNED with the server's 64 KiB request cap: a
  // UTF-8 body of at most 65536 bytes can never encode a JSON string longer
  // than 65536 UTF-16 code units, so no server-valid echo is ever rejected
  // here. This mirrors the transport contract — it is deliberately NOT an
  // invented generation limit (config.py imposes no prompt-length limit).
  var MAX_CONFIG_STRING_CHARS = 65536;

  // Import reads the SELECTED FILE locally before uploading it: the same
  // 64 KiB bound the server imposes on request bodies. The file is read as
  // raw bytes and POSTed verbatim — the browser NEVER JSON.parses it (the
  // export carries an exact uint64 numeric seed that JSON.parse would
  // round through Number; the server parses it in Python exactly).
  var MAX_IMPORT_FILE_BYTES = 65536;

  // Output formats — the exact OutputFormat enum values of config.py.
  var OUTPUT_FORMATS = ['jpg', 'png', 'webp'];

  // The 15 browser-editable GenerationConfig fields, in wire order (the
  // server's EDITABLE_FIELDS, web/api.py). Plan and job echoes must carry
  // EXACTLY this set: the explicit legacy branches in parsePlan/parseJob
  // fill in the keys OLDER servers never sent, and anything else (an unknown
  // key, an alias) is a schema rejection — never "any key count".
  var EDITABLE_KEYS = ['prompt', 'negative_prompt', 'width', 'height', 'steps',
                       'recipe', 'true_cfg_scale', 'seed', 'quant', 'offload',
                       'attention_backend', 'vae_tiled_decode', 'output_format',
                       'transparent_background', 'references'];

  // Recipe enum values (config.py Recipe) and the step count Turbo owns
  // (config.py TURBO_STEPS). A test pins both against the Python values:
  // Turbo's shipped sigmas make num_inference_steps inert, so the recipe
  // owns the step count and the control is LOCKED at exactly this number.
  var RECIPES = ['base', 'turbo'];
  var TURBO_STEPS = 8;

  // Formats that can carry an alpha channel (config.py ALPHA_OUTPUT_FORMATS).
  // JPEG cannot store transparency, so a transparent background REQUIRES one
  // of these; the UI switches jpg to png and says so rather than silently
  // flattening the image.
  var ALPHA_FORMATS = ['png', 'webp'];

  // The EXACT feedback wrapper config.py appends to the model-visible prompt
  // when transparent_background is true (TRANSPARENCY_PROMPT_PREFIX/SUFFIX,
  // TRANSPARENCY_WRAPPER_REVISION 'qwen-engine.transparent-wrapper.v1').
  // Mirrored here only to DISPLAY the text the model reads — the draft's
  // prompt field itself is never rewritten. tests/test_web_static.py pins
  // these literals against the Python constants.
  var TRANSPARENT_PREFIX = 'This is an RGBA image with transparency. ';
  var TRANSPARENT_SUFFIX =
    ' The image has alpha channel and the background is transparent.';

  // True when the draft asks for transparency but the chosen format cannot
  // store it. The change handlers keep this from ever being the submitted
  // state; it exists so a mismatch is REFUSED locally, never silently saved.
  function transparencyFormatMismatch() {
    return $('transparent').checked &&
           ALPHA_FORMATS.indexOf($('output-format').value) === -1;
  }

  // Closed theme set (the only values ever written to documentElement's
  // data-theme; the ONLY localStorage key the app ever touches).
  var THEMES = ['original', 'neon', 'paper'];
  var THEME_STORAGE_KEY = 'qwen-studio-theme';

  // CPython str.strip() whitespace set EXACTLY (JS trim() differs: it drops
  // U+FEFF, and a plain character class without \x20 would KEEP the most
  // mainstream whitespace of all — plain SPACE U+0020). parsePlan mirrors
  // Python exactly so a server-valid cfg_active is never falsely rejected
  // by JS whitespace rules ('   ' + cfg>1 is cfg_active FALSE server-side).
  function pyStrip(s) {
    // EDGES ONLY — Python str.strip() removes LEADING/TRAILING whitespace
    // and NEVER interior whitespace: '4 2' must stay '4 2' (rejected by the
    // seed grammar), not silently become '42'.
    return s.replace(/^[\x20\x09\x0A\x0B\x0C\x0D\x1C\x1D\x1E\x1F\x85\xA0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]+|[\x20\x09\x0A\x0B\x0C\x0D\x1C\x1D\x1E\x1F\x85\xA0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]+$/g, '');
  }

  // BEGIN GENERATED PRESETS (scripts/sync_presets.py)
  var PRESETS = {
    "square": {
      "width": 1024,
      "height": 1024,
      "ratio": "1:1",
      "tier": "1K"
    },
    "square2k": {
      "width": 2048,
      "height": 2048,
      "ratio": "1:1",
      "tier": "2K"
    },
    "fourthree": {
      "width": 1152,
      "height": 864,
      "ratio": "4:3",
      "tier": "1K"
    },
    "fourthree2k": {
      "width": 2400,
      "height": 1792,
      "ratio": "4:3",
      "tier": "2K"
    },
    "threefour": {
      "width": 864,
      "height": 1152,
      "ratio": "3:4",
      "tier": "1K"
    },
    "threefour2k": {
      "width": 1792,
      "height": 2400,
      "ratio": "3:4",
      "tier": "2K"
    },
    "threetwo": {
      "width": 1248,
      "height": 832,
      "ratio": "3:2",
      "tier": "1K"
    },
    "threetwo2k": {
      "width": 2528,
      "height": 1696,
      "ratio": "3:2",
      "tier": "2K"
    },
    "twothree": {
      "width": 832,
      "height": 1248,
      "ratio": "2:3",
      "tier": "1K"
    },
    "twothree2k": {
      "width": 1696,
      "height": 2528,
      "ratio": "2:3",
      "tier": "2K"
    },
    "landscape": {
      "width": 1344,
      "height": 768,
      "ratio": "16:9",
      "tier": "1K"
    },
    "landscape2k": {
      "width": 2752,
      "height": 1536,
      "ratio": "16:9",
      "tier": "2K"
    },
    "portrait": {
      "width": 768,
      "height": 1344,
      "ratio": "9:16",
      "tier": "1K"
    },
    "portrait2k": {
      "width": 1536,
      "height": 2752,
      "ratio": "9:16",
      "tier": "2K"
    }
  };
  // END GENERATED PRESETS

  var VIEWS = { create: 'view-create', history: 'view-history', settings: 'view-settings' };
  var VIEW_HEADINGS = { create: 'create-heading', history: 'history-heading', settings: 'settings-heading' };

  function $(id) { return document.getElementById(id); }

  function setText(id, text) {
    var node = $(id);
    if (node) node.textContent = text;
  }

  /* ---------------- Seed handling (BigInt decimal strings) ---------------- */

  // Accepted grammar for the TYPED value (browser-side; the wire itself
  // NEVER strips — it carries only canonical digits; deliberately strict,
  // checked BEFORE any BigInt):
  //   * surrounding whitespace of the user's input trimmed at the edges;
  //   * at most SEED_MAX_DIGITS (20) characters AFTER trimming — length is
  //     bounded first so a longer string is rejected as-is even if leading
  //     zeros would put its value in range (documented policy; the HTML has
  //     deliberately NO maxlength, because maxlength silently TRUNCATES a
  //     pasted 21-digit value into a DIFFERENT valid seed — losing user
  //     input; parseSeed rejects instead);
  //   * ASCII decimal digits only (\d is ASCII-only without /u);
  //   * the parsed value must be within [0, 2^64 - 1].
  // The full uint64 maximum stays representable as an exact decimal STRING,
  // and it stays a STRING on the wire, in the draft and in every render —
  // never a rounded JS Number.
  function parseSeed(str) {
    // BROWSER-SIDE user input only: the EDGES of what the user typed are
    // trimmed (Python whitespace set) before the canonical digit STRING
    // goes on the wire. The API wire NEVER strips — the server accepts
    // only canonical digits and refuses anything else, never editing it.
    var s = pyStrip(String(str));
    if (s.length > SEED_MAX_DIGITS) return null;
    if (!/^\d+$/.test(s)) return null;
    var value;
    try { value = BigInt(s); } catch (err) { return null; }
    if (value < SEED_MIN || value > SEED_MAX) return null;
    return value;
  }

  function seedWireString(str) {
    var value = parseSeed(str);
    return value === null ? null : value.toString();
  }

  function randomSeedString() {
    // 8 cryptographic random bytes -> exactly 64 bits -> full uint64 range,
    // assembled without any floating-point step.
    var bytes = new Uint8Array(8);
    crypto.getRandomValues(bytes);
    var hex = '';
    for (var i = 0; i < bytes.length; i++) {
      hex += bytes[i].toString(16).padStart(2, '0');
    }
    return BigInt('0x' + hex).toString();
  }

  /* ---------------- Lightweight form validation ----------------
   * The server validates GenerationConfig authoritatively; these checks
   * mirror config.py truthfully for accessible inline feedback only.
   */

  function isValidDimension(n) {
    return Number.isInteger(n) && n >= 64 && n <= 4096 && n % 32 === 0;
  }

  // A typed numeric read: empty strings are EMPTY, not 0; non-finite
  // results (JSON.stringify would silently turn Infinity into null) and
  // non-integers for integer fields are refused BEFORE serialization.
  function numberField(id, wantInt) {
    // BROWSER-SIDE user input only: edges of the typed value are trimmed
    // (Python whitespace set) before parsing — a UI affordance. The API
    // wire NEVER strips: the server receives already-canonical JSON.
    var raw = pyStrip(String($(id).value));
    if (raw === '') return { error: 'A value is required (empty fields are not sent as 0).' };
    var n = Number(raw);
    if (!Number.isFinite(n)) return { error: 'Must be a finite number.' };
    if (wantInt && !Number.isInteger(n)) return { error: 'Must be a whole number.' };
    return { value: n };
  }

  function validateForm() {
    var errors = {};
    // GenerationConfig (config.py) requires a non-empty trimmed prompt, so
    // the form mirrors that — surfaced only after the user has touched the
    // prompt or tried to submit (no intrusive alert on first paint, and we
    // never auto-fill anything).
    if (!pyStrip($('prompt').value)) {
      errors.prompt = 'Prompt must not be empty — enter the text you want ' +
                      'used exactly as typed.';
    }
    // Under Turbo these controls are locked (applyRecipeMode) at values the
    // server accepts, so the ordinary checks stay as the honest mirror: a
    // restored OUT-OF-RANGE value from before Turbo still surfaces its error.
    var steps = numberField('steps', true);
    if (steps.error) { errors.steps = steps.error; }
    else if (steps.value < 1 || steps.value > 1000) {
      errors.steps = 'Steps must be a whole number from 1 to 1000.';
    }
    var seedStr = seedWireString($('seed').value);
    if (seedStr === null) {
      var seedRaw = pyStrip(String($('seed').value));
      if (/^\d+$/.test(seedRaw) && seedRaw.length > SEED_MAX_DIGITS) {
        errors.seed = 'Seed is too long: at most ' + SEED_MAX_DIGITS +
                      ' decimal digits. The text was kept exactly as entered ' +
                      '(nothing was truncated).';
      } else {
        errors.seed = 'Seed must be a whole-number string of at most ' +
                      SEED_MAX_DIGITS + ' digits, from 0 to ' + SEED_MAX + '.';
      }
    }
    var cfg = numberField('cfg', false);
    if (cfg.error) { errors.cfg = cfg.error; }
    else if (cfg.value < 1) {
      errors.cfg = 'Guidance must be a finite number of 1 or more.';
    }
    var width = numberField('width', true);
    var height = numberField('height', true);
    if (width.error || height.error) {
      errors.dims = width.error || height.error;
    } else if (!isValidDimension(width.value) || !isValidDimension(height.value)) {
      errors.dims = 'Width and height must be multiples of 32 between 64 ' +
                    'and 4096.';
    }
    // config.py refuses transparent_background=True with a format that has
    // no alpha channel. The change handlers auto-switch jpg -> png so this
    // is unreachable in normal use; it stays as the honest local mirror of
    // the server rule (rejected, never silently flattened).
    if (transparencyFormatMismatch()) {
      errors.format = 'Transparent background needs PNG or WebP — JPEG ' +
                      'cannot store transparency. Choose png or webp, or ' +
                      'untick Transparent background.';
    }
    return { ok: Object.keys(errors).length === 0, errors: errors };
  }

  function setAriaInvalid(id, invalid) {
    var el = $(id);
    if (el) el.setAttribute('aria-invalid', invalid ? 'true' : 'false');
  }

  var promptTouched = false;

  function syncOptionHelp() {
    ['quant', 'offload', 'attention'].forEach(function (id) {
      var control = $(id);
      var option = control.options[control.selectedIndex];
      var description = option ? option.title : 'Choose an option to see what it does.';
      control.title = description;
      if ($(id + '-help').textContent !== description) setText(id + '-help', description);
    });
  }

  function renderValidation(errors) {
    syncOptionHelp();
    setText('steps-error', errors.steps || '');
    setText('seed-error', errors.seed || '');
    setText('cfg-error', errors.cfg || '');
    setText('dims-error', errors.dims || '');
    setText('format-error', errors.format || '');
    // Server-side policy refusals for these four fields are rendered inline
    // by applyServerFieldErrors; any draft edit clears them (they described
    // the configuration that was sent, not the current one).
    setText('quant-error', '');
    setText('recipe-error', '');
    setAriaInvalid('quant', false);
    syncTransparency();
    // The prompt error only appears once the user has edited the prompt or
    // attempted to submit — never as a first-load alert.
    var showPrompt = promptTouched && !!errors.prompt;
    setText('prompt-error', showPrompt ? errors.prompt : '');
    setAriaInvalid('steps', !!errors.steps);
    setAriaInvalid('seed', !!errors.seed);
    setAriaInvalid('cfg', !!errors.cfg);
    setAriaInvalid('prompt', showPrompt);
    setAriaInvalid('output-format', !!errors.format);
    // The dimension error message is shared, so it marks BOTH controls.
    setAriaInvalid('width', !!errors.dims);
    setAriaInvalid('height', !!errors.dims);
  }

  // Server refusals that describe THIS recipe combination are mapped onto the
  // controls the user can act on, so "the server said no" becomes "this
  // control is wrong, and here is why". Every other field stays in the
  // bounded api-message text (renderApiError). Returns the number applied.
  var SERVER_FIELD_SLOTS = {
    steps: 'steps-error',
    true_cfg_scale: 'cfg-error',
    quant: 'quant-error',
    recipe: 'recipe-error'
  };
  var SERVER_FIELD_CONTROLS = {
    steps: 'steps', true_cfg_scale: 'cfg', quant: 'quant', recipe: 'recipe-field'
  };

  function applyServerFieldErrors(payload) {
    if (!payload || !Array.isArray(payload.fields)) return 0;
    var applied = 0;
    for (var i = 0; i < payload.fields.length; i++) {
      var f = payload.fields[i];
      if (!f || typeof f.field !== 'string' ||
          !Object.prototype.hasOwnProperty.call(SERVER_FIELD_SLOTS, f.field) ||
          typeof f.message !== 'string') continue;
      var slot = SERVER_FIELD_SLOTS[f.field];
      setText(slot, f.message.slice(0, 200));
      var control = $(SERVER_FIELD_CONTROLS[f.field]);
      if (control) control.setAttribute('aria-invalid', 'true');
      applied++;
    }
    return applied;
  }

  // Set only by the jpg -> png auto-switch, so the explanation survives the
  // validation re-renders that follow (typing in the prompt, etc.) instead
  // of flashing once. Cleared by an explicit format or checkbox change.
  var transparencySwitchNote = '';

  // The transparency row's derived state: WHY the format changed (only ever
  // a visible, explained switch), and the exact text the model will read.
  // Never rewrites the prompt field itself — the wrapper is display only.
  function syncTransparency() {
    var on = $('transparent').checked;
    var format = $('output-format').value;
    var why = '';
    if (on && ALPHA_FORMATS.indexOf(format) === -1) {
      why = 'JPEG cannot keep transparency — choose PNG or WebP, or untick this option.';
    } else if (on) {
      why = transparencySwitchNote ||
            ('Saving as ' + format.toUpperCase() + ', which keeps the ' +
             'transparency. The result is not guaranteed to be transparent ' +
             '— check the image.');
    }
    setText('transparency-note', why);
    $('transparency-note').hidden = why === '';
    $('effective-prompt-field').hidden = !on;
    if (on) {
      setText('effective-prompt',
        TRANSPARENT_PREFIX + $('prompt').value + TRANSPARENT_SUFFIX);
    } else {
      setText('effective-prompt', '');
    }
  }

  // Ticking transparency with a non-alpha format switches jpg -> png and
  // SAYS so (transparency-note). Unticking never rewrites the user's format:
  // PNG stays PNG, because that is the format the user last had selected.
  function onTransparencyChange() {
    transparencySwitchNote = '';
    if ($('transparent').checked &&
        ALPHA_FORMATS.indexOf($('output-format').value) === -1) {
      $('output-format').value = 'png';
      transparencySwitchNote =
        'Output format switched from jpg to png: JPEG cannot store a ' +
        'transparent background. PNG keeps it.';
    }
    renderValidation(validateForm().errors);
    syncPlanStaleness();
  }

  /* ---------------- Recipe (base / Turbo) ----------------
   * The server's config.py gates Turbo on three things: exactly
   * TURBO_STEPS sampling steps, no true-CFG (no negative prompt AND no
   * scale above 1.0), and FP16 weights. The UI therefore LOCKS those
   * controls while Turbo is selected and says why — it never silently
   * submits a combination the server would refuse. Nothing the user typed
   * is deleted: only a locked, restored-on-switch-back value is replaced,
   * and the negative prompt TEXT is kept (it is legal at scale 1.0).
   */

  // Values captured when Turbo is entered, restored when Quality is chosen
  // again. null means "not in a locked Turbo draft".
  var recipeSaved = null;

  function currentRecipe() {
    var el = document.querySelector('input[name="recipe"]:checked');
    return el && RECIPES.indexOf(el.value) !== -1 ? el.value : 'base';
  }

  function setRecipeControl(value) {
    var el = document.getElementById('recipe-' + value);
    if (el) el.checked = true;
  }

  // Brings the form in line with the selected recipe: locks the controls the
  // server will not accept under Turbo, and restores the user's own values
  // when switching back. Safe to call repeatedly.
  //
  // `keepValues` marks an EXTERNALLY supplied Turbo draft (an imported plan):
  // its steps/guidance/quant already are the recipe's own values, so there is
  // nothing of the user's to capture — capturing them would later "restore"
  // numbers the user never chose while in Quality.
  function applyRecipeMode(keepValues) {
    var turbo = currentRecipe() === 'turbo';
    var steps = $('steps'), cfg = $('cfg'), quant = $('quant'), negative = $('negative');
    if (turbo && recipeSaved === null && !keepValues) {
      recipeSaved = { steps: steps.value, cfg: cfg.value, quant: quant.value };
      steps.value = String(TURBO_STEPS);
      cfg.value = '1.0';
      quant.value = 'fp16';
    } else if (!turbo && recipeSaved !== null) {
      steps.value = recipeSaved.steps;
      cfg.value = recipeSaved.cfg;
      quant.value = recipeSaved.quant;
      recipeSaved = null;
    }
    steps.disabled = turbo;
    cfg.disabled = turbo;
    negative.disabled = turbo;
    quant.disabled = turbo;
    setText('recipe-note', turbo
      ? 'Turbo is selected: steps are fixed at ' + TURBO_STEPS + ', guidance ' +
        'is fixed at 1.0 (Turbo has no negative prompt), and quantization ' +
        'is FP16. These controls are locked until you switch back to Quality.'
      : '');
    $('recipe-note').hidden = !turbo;
  }

  /* ---------------- UI wire snapshot and request envelope ----------------
   * formSnapshot() is the UI wire format (kind marks it; seed is a STRING).
   * buildEnvelope() turns the CURRENT draft into the exact server envelope:
   * {schema, config} — the kind marker is excluded, there are no aliases
   * and no other remapping. It never carries runtime paths.
   */

  function formSnapshot() {
    return {
      kind: 'ui-wire/w2',
      prompt: $('prompt').value,
      negative_prompt: $('negative').value,
      width: Number($('width').value),
      height: Number($('height').value),
      steps: Number($('steps').value),
      recipe: currentRecipe(),                 // base | turbo (exact config.py values)
      seed: seedWireString($('seed').value),   // decimal STRING or null (UI-only)
      true_cfg_scale: Number($('cfg').value),
      quant: $('quant').value,                 // fp16 | q8 (exact config.py values)
      offload: $('offload').value,             // resident | streamed | offload
      attention_backend: $('attention').value, // auto | sdpa | efficient | eager
      vae_tiled_decode: $('vae-tiling').checked,
      output_format: $('output-format').value, // jpg | png | webp (exact config.py enum)
      transparent_background: $('transparent').checked,
      // Reference ids in LIST order — the displayed order IS the
      // conditioning order ("image 1" in the prompt). Ids only: the bytes
      // live in the server store, keyed by these hashes.
      references: activeReferenceIds()
    };
  }

  // Returns { ok: true, envelope } or { ok: false, errors }. Numeric fields
  // are re-checked FINITE before serialization: an Infinity sneaking into
  // JSON.stringify would become null and hide the real error.
  function buildEnvelope() {
    var check = validateForm();
    if (!check.ok) return { ok: false, errors: check.errors };
    var snap = formSnapshot();
    return {
      ok: true,
      envelope: {
        schema: PLAN_REQUEST_SCHEMA,
        config: {
          prompt: snap.prompt,
          negative_prompt: snap.negative_prompt,
          width: snap.width,
          height: snap.height,
          steps: snap.steps,
          recipe: snap.recipe,
          true_cfg_scale: snap.true_cfg_scale,
          seed: snap.seed,                     // decimal STRING or null — never a Number
          quant: snap.quant,
          offload: snap.offload,
          attention_backend: snap.attention_backend,
          vae_tiled_decode: snap.vae_tiled_decode,
          output_format: snap.output_format,
          transparent_background: snap.transparent_background,
          references: snap.references
        }
      }
    };
  }

  /* ---------------- Connection / status ---------------- */

  var FETCH_TIMEOUT_MS = 15000;  // finite: every fetch aborts; no polling
  // Fixed response cap. Every legitimate answer to a <=64 KiB request fits
  // comfortably (static assets are small, the export echo of the editable
  // fields is bounded by the request itself); an over-cap body is refused,
  // never buffered. No unbounded bodies are ever read.
  var MAX_RESPONSE_BYTES = 262144;
  // The ONE response this UI has to read whole is a finished job's own image,
  // which "Use as reference" feeds straight back into the reference upload: a
  // large PNG is nowhere near the JSON cap. The override is bound to the
  // store's own upload limit, so an over-cap body is still refused rather than
  // buffered, and it is passed PER CALL — every other route keeps
  // MAX_RESPONSE_BYTES.
  var MAX_JOB_IMAGE_BYTES = MAX_REFERENCE_BYTES;
  var apiConnected = false;      // a RECOGNIZED status document was seen
  var generationConnected = false;
  var submittingJob = false;
  var jobCache = Object.create(null);
  var selectedJob = null;
  var refreshingJobs = false;
  var jobTimer = null;
  var jobSeq = 0;
  // EVERY status completion branch is guarded by this sequence: a newer
  // check (success OR failure) always wins over an older late arrival —
  // an old success must not re-enable planning after a newer disconnect,
  // and an old failure must not disable a newer valid connection.
  var statusSeq = 0;

  var BODY_TOO_LARGE = { bodyTooLarge: true };
  // Fail-closed marker: when the readable-stream API is unavailable the
  // body is NOT read through an unbounded fallback — it is refused as a
  // typed body error (schema rejection), never silently buffered whole.
  var UNSUPPORTED_BODY = { unsupportedBody: true };

  // Complete body bytes, capped. The abort signal of the surrounding
  // request also errors an in-flight body stream, so a stalled or
  // truncated body cannot hang the page. `cap` is the per-call override
  // computed by apiRequest; anything else (absent, zero, negative, NaN or
  // larger than the ONE legitimately large response) fails closed to the
  // default cap — an unbounded read is never reachable.
  function readBodyLimited(res, cap) {
    if (typeof cap !== 'number' || !(cap > 0) || cap > MAX_JOB_IMAGE_BYTES) {
      cap = MAX_RESPONSE_BYTES;
    }
    if (!res.body || typeof res.body.getReader !== 'function') {
      // No bounded streaming available: FAIL CLOSED. An arrayBuffer-style
      // fallback would read the whole body before the cap could be
      // checked — refusing is the honest bounded behaviour.
      return Promise.reject(UNSUPPORTED_BODY);
    }
    var reader = res.body.getReader();
    var chunks = [];
    var total = 0;
    function pump() {
      return reader.read().then(function (r) {
        if (r.done) {
          var out = new Uint8Array(total);
          var off = 0;
          for (var i = 0; i < chunks.length; i++) {
            out.set(chunks[i], off);
            off += chunks[i].byteLength;
          }
          return out;
        }
        total += r.value.byteLength;
        if (total > cap) {
          // Stop at once: an over-cap body is cancelled, never buffered.
          return reader.cancel().then(function () { throw BODY_TOO_LARGE; });
        }
        chunks.push(r.value);
        return pump();
      });
    }
    return pump();
  }

  // One bounded same-origin exchange. The AbortController timer covers the
  // WHOLE operation — fetch headers AND body consumption — and the timer is
  // cleared in settle(), after the body has been consumed (or failed).
  // Resolves (never rejects) with exactly one of:
  //   { network: true, timedOut }      — unreachable, aborted or timed out
  //                                    (headers OR body stage): an honest
  //                                    network outcome, never a schema verdict
  //   { res, json }                    — body decoded UTF-8, JSON-parsed
  //                                    (json may be null — a parser value the
  //                                    strict parsers reject, not a crash)
  //   { res, bodyError: 'json' }       — complete body, malformed/undecodable
  //                                    JSON: a schema rejection, NOT a
  //                                    disconnect
  //   { res, bodyError: 'too_large' }  — body exceeded the applicable cap
  //                                    (the default, or opts.maxBytes)
  //   { res, bytes }                   — mode 'raw': the COMPLETE raw bytes,
  //                                    untouched (export downloads these)
  function apiRequest(url, opts, mode) {
    var controller = new AbortController();
    var timedOut = false;
    // Per-call deadline override. Only the rewriter uses it: a local LLM
    // generating a paragraph legitimately takes minutes, while every other
    // call must stay bounded by the short default.
    var budget = (opts && typeof opts.timeoutMs === 'number' &&
                  opts.timeoutMs > 0) ? opts.timeoutMs : FETCH_TIMEOUT_MS;
    // Per-call response cap. Only the finished-image download raises it, and
    // only up to MAX_JOB_IMAGE_BYTES (the reference store's own upload limit);
    // an absent, non-positive or oversized value keeps the default.
    var cap = (opts && typeof opts.maxBytes === 'number' &&
               opts.maxBytes > 0 && opts.maxBytes <= MAX_JOB_IMAGE_BYTES)
      ? opts.maxBytes : MAX_RESPONSE_BYTES;
    var timer = setTimeout(function () {
      timedOut = true;
      controller.abort();
    }, budget);
    var requestOpts = {};
    if (opts) {
      for (var k in opts) {
        // timeoutMs/maxBytes are this core's own knobs, never fetch options.
        if (k !== 'timeoutMs' && k !== 'maxBytes' &&
            Object.prototype.hasOwnProperty.call(opts, k)) requestOpts[k] = opts[k];
      }
    }
    requestOpts.signal = controller.signal;
    // No HTTP-cache path for API calls: every status/plan/export request
    // goes to the server (a cached status is NEVER an authoritative one,
    // and two checks must be two REAL requests — verifiable in the wire).
    requestOpts.cache = 'no-store';
    function settle(value) { clearTimeout(timer); return value; }
    return fetch(url, requestOpts).then(function (res) {
      return readBodyLimited(res, cap).then(function (bytes) {
        if (mode === 'raw') return settle({ res: res, bytes: bytes });
        var value;
        try {
          value = JSON.parse(new TextDecoder('utf-8', { fatal: true })
            .decode(bytes));
        } catch (err) {
          return settle({ res: res, bodyError: 'json' });
        }
        return settle({ res: res, json: value });
      }, function (err) {
        // The body stage failed: the whole-operation timeout aborted the
        // stream, the connection died, or the fixed cap was exceeded.
        if (err === BODY_TOO_LARGE) {
          return settle({ res: res, bodyError: 'too_large' });
        }
        if (err === UNSUPPORTED_BODY) {
          return settle({ res: res, bodyError: 'unsupported' });
        }
        return settle({ network: true, timedOut: timedOut });
      });
    }, function () {
      // Never got past the headers stage: an honest network outcome.
      return settle({ network: true, timedOut: timedOut });
    });
  }

  function renderChip(text, backendText) {
    setText('connection-chip', text);
    setText('stat-backend', backendText);
  }

  // Plan/Export are only offered once a RECOGNIZED status has been seen;
  // any disconnect or schema rejection takes them and generation away again.
  function setApiAvailable(available) {
    apiConnected = available;
    $('btn-plan').disabled = !available;
    $('btn-export').disabled = !available;
    $('btn-import').disabled = !available;
    if (!available) generationConnected = false;
    if (!available) setRewriterState(null);   // rewriter availability is status-gated
    updateReferenceControls();   // uploads are writes: status-gated too
    updateGenerate();
  }

  // The status document is NOT trusted by shape alone: schema tag, strict
  // literal booleans (generation_available must match the blockers), the
  // exact known gates with their CLOSED states AND codes, blocker codes
  // from the closed list only, and the exact manifest verification — ALL
  // must match, or the whole document is rejected. A JSON null parses to
  // the value null and is rejected here by the schema, never by a property
  // access on null.
  function parseStatus(s) {
    if (!s || typeof s !== 'object') return null;   // null-safe by design
    if (s.schema !== STATUS_SCHEMA) return null;
    if (s.planning_available !== true) return null;
    if (typeof s.generation_available !== 'boolean') return null;
    if (!s.gates || typeof s.gates !== 'object') return null;
    var gates = Object.keys(GATE_STATES);
    if (Object.keys(s.gates).length !== gates.length) return null;  // exact
    for (var i = 0; i < gates.length; i++) {
      var gate = s.gates[gates[i]];
      var spec = GATE_STATES[gates[i]];
      if (!gate || typeof gate.state !== 'string' || gate.state.length > 64 ||
          gate.code !== spec.code ||
          spec.states.indexOf(gate.state) === -1) return null;
    }
    // Closed blockers and mandatory manifest reason, when applicable.
    var manifest = parseManifestCore(s.manifest);
    if (!manifest) return null;   // status ALWAYS carries the manifest block
    if (!blockersExact(s.generation_blockers, manifest.blocker)) return null;
    if (s.generation_available !== (s.generation_blockers.length === 0)) return null;
    // The rewrite block is OPTIONAL and additive: an older server omits it,
    // which simply means "no rewriter". Once present it must be EXACTLY the
    // four closed keys — a partial or widened block is rejected, and with it
    // the rewriter (never a guessed backend).
    var rewrite = parseRewriteBlock(s.rewrite);
    if (rewrite === INVALID) return null;
    return { recognized: true, planning: true, generation: s.generation_available,
             rewrite: rewrite };
  }

  /* The rewrite availability, strictly. `undefined` (absent) and the disabled
     pair both mean unavailable; only the exact available pair yields the
     backend. The four core keys are ALWAYS present and `backends`, `settings`
     and `active_by_task` are the only additive ones, each parsed in full: an
     unknown key, a missing key, an unknown state/code pair or a non-string
     backend is INVALID and rejects the whole status document, exactly like every
     other closed-set parse here. The page therefore never guesses at a contract
     it has not read. */
  function parseRewriteBlock(block) {
    if (block === undefined) return null;
    if (!block || typeof block !== 'object' || Array.isArray(block)) return INVALID;
    var keys = Object.keys(block);
    if (keys.length < 4) return INVALID;
    for (var i = 0; i < keys.length; i++) {
      if (REWRITE_BLOCK_KEYS.indexOf(keys[i]) === -1) return INVALID;
    }
    if (typeof block.state !== 'string' || typeof block.code !== 'string' ||
        typeof block.reason !== 'string') return INVALID;
    if (block.backend !== null && typeof block.backend !== 'string') return INVALID;
    var pair = REWRITE_STATES[block.code];
    if (!pair || pair !== block.state) return INVALID;
    var rows = null;
    if (block.backends !== undefined) {
      rows = parseRewriteBackends(block.backends);
      if (rows === INVALID) return INVALID;
    }
    var pointer = null;
    if (block.settings !== undefined) {
      pointer = parseRewriteSettingsPointer(block.settings);
      if (pointer === INVALID) return INVALID;
    }
    var byTask = null;
    if (block.active_by_task !== undefined) {
      byTask = parseRewriteActiveByTask(block.active_by_task, rows,
                                       block.state === 'available');
      if (byTask === INVALID) return INVALID;
    }
    if (block.state !== 'available') {
      return { backend: null, reason: block.reason, backends: rows,
               settings: pointer, byTask: null };   // disabled / unavailable
    }
    if (typeof block.backend !== 'string' || !block.backend) return INVALID;
    if (block.backend.length > 64) return INVALID;
    if (block.reason !== '') return INVALID;        // available carries no reason
    if (rows !== null && !containsId(rows, block.backend)) return INVALID;
    if (byTask !== null && !hasAnyValue(byTask)) return INVALID;
    return { backend: block.backend, reason: '', backends: rows,
             settings: pointer, byTask: byTask };
  }

  // Which rewriter answers EACH draft kind. One entry per known task, each one
  // null or a rewriter the same block advertises: a block that says the text
  // enhancer answers an edit draft while its own rows say otherwise is a
  // contradiction, and the page shows the truth it can read instead.
  function parseRewriteActiveByTask(active, rows, available) {
    if (!active || typeof active !== 'object' || Array.isArray(active)) return INVALID;
    var keys = Object.keys(active);
    if (keys.length !== REWRITE_TASKS.length) return INVALID;
    var out = {};
    var answered = false;
    for (var i = 0; i < REWRITE_TASKS.length; i++) {
      var task = REWRITE_TASKS[i];
      if (keys.indexOf(task) === -1) return INVALID;
      var backend = active[task];
      if (backend !== null) {
        if (typeof backend !== 'string' || !REWRITE_BACKEND_ID_RE.test(backend)) return INVALID;
        if (rows !== null && !containsId(rows, backend)) return INVALID;
        answered = true;
      }
      out[task] = backend;
    }
    if (available !== answered) return INVALID;
    return out;
  }


  // The additive `backends` rows: which rewriters exist here and what each can
  // do. Labels are display text the server owns; a path, address or model name
  // in one is a contract violation, so it rejects the document.
  function parseRewriteBackends(rows) {
    if (!Array.isArray(rows) || !rows.length || rows.length > 4) return INVALID;
    var out = [];
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      if (!row || typeof row !== 'object' || Array.isArray(row)) return INVALID;
      var keys = Object.keys(row);
      if (keys.length !== 4) return INVALID;
      for (var k = 0; k < keys.length; k++) {
        if (REWRITE_BACKEND_ROW_KEYS.indexOf(keys[k]) === -1) return INVALID;
      }
      if (typeof row.id !== 'string' || !REWRITE_BACKEND_ID_RE.test(row.id)) return INVALID;
      if (typeof row.label !== 'string' || !row.label || row.label.length > 120) return INVALID;
      if (/[\/\\]/.test(row.label)) return INVALID;   // a label is never a path
      if (typeof row.uses_gpu !== 'boolean') return INVALID;
      if (!Array.isArray(row.tasks) || !row.tasks.length ||
          row.tasks.length > REWRITE_TASKS.length) return INVALID;
      for (var t = 0; t < row.tasks.length; t++) {
        if (REWRITE_TASKS.indexOf(row.tasks[t]) === -1) return INVALID;
      }
      if (containsId(out, row.id)) return INVALID;   // one row per backend
      out.push({ id: row.id, label: row.label, tasks: row.tasks.slice(),
                 uses_gpu: row.uses_gpu });
    }
    return out;
  }

  // The additive `settings` pointer: the ROUTE the choice is written to, plus
  // whether a saved document is on disk. A route is a path on THIS origin and
  // nothing else — never a filesystem location.
  function parseRewriteSettingsPointer(pointer) {
    if (!pointer || typeof pointer !== 'object' || Array.isArray(pointer)) return INVALID;
    var keys = Object.keys(pointer);
    if (keys.length !== 2) return INVALID;
    for (var i = 0; i < keys.length; i++) {
      if (['href', 'saved'].indexOf(keys[i]) === -1) return INVALID;
    }
    if (typeof pointer.href !== 'string' ||
        pointer.href.indexOf(API_REWRITE_SETTINGS) !== 0) return INVALID;
    if (typeof pointer.saved !== 'boolean') return INVALID;
    return { href: pointer.href, saved: pointer.saved };
  }

  function containsId(rows, id) {
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].id === id) return true;
    }
    return false;
  }

  // True when at least one entry of an object is non-null — "something answers".
  function hasAnyValue(map) {
    var keys = Object.keys(map);
    for (var i = 0; i < keys.length; i++) {
      if (map[keys[i]] !== null) return true;
    }
    return false;
  }

  function statusUnrecognizedText(reason) {
    return 'Connected, but the status document did not match the expected ' +
           'schema' + (reason ? ' (' + reason + ')' : '') +
           ' — the whole document was rejected and nothing from it is ' +
           'trusted.';
  }

  // The EXACT manifest core the server projects (api._manifest_core, shared
  // by status and plan): metadata only — never filenames/hashes/paths, and
  // verification is ALWAYS 'not_performed'. When valid: typed counts, no
  // reason. When invalid/unconfigured: a CLOSED reason code. Returns null
  // (reject) or { configured, valid, blocker } where blocker is the code
  // the server adds to generation_blockers (null when it adds none).
  function parseManifestCore(m) {
    if (!m || typeof m !== 'object' ||
        typeof m.configured !== 'boolean' || typeof m.valid !== 'boolean' ||
        m.verification !== 'not_performed') return null;
    if (m.valid && !m.configured) return null;      // valid implies configured
    var blocker = null;
    if (!m.configured) {
      blocker = 'manifest_unconfigured';
    } else if (!m.valid) {
      if (MANIFEST_REASON_CODES.indexOf(m.reason) === -1 ||
          m.reason === 'manifest_unconfigured') return null;
      blocker = m.reason;
    } else {
      if (m.reason !== undefined) return null;      // reason only when invalid
      if (m.schema_version !== 1) return null;
      if (!Number.isInteger(m.artifact_count) || m.artifact_count < 1 ||
          m.artifact_count > 1000000) return null;
    }
    return { configured: m.configured, valid: m.valid, blocker: blocker };
  }

  // Only known static blockers, no duplicates, and exactly the manifest
  // failure indicated by the metadata. Submission availability must agree.
  function blockersExact(list, manifestBlocker) {
    if (!Array.isArray(list)) return false;
    if (list.length > 3 || new Set(list).size !== list.length) return false;
    for (var j = 0; j < list.length; j++) {
      if (KNOWN_BLOCKER_CODES.indexOf(list[j]) === -1) return false;
      if (MANIFEST_REASON_CODES.indexOf(list[j]) !== -1 && list[j] !== manifestBlocker) return false;
    }
    if (manifestBlocker && list.indexOf(manifestBlocker) === -1) return false;
    return true;
  }

  function checkStatus() {
    var seq = ++statusSeq;
    return apiRequest(API_STATUS, { headers: { 'Accept': 'application/json' } },
                      'json')
      .then(function (r) {
        if (seq !== statusSeq) return null;   // superseded: never land late
        if (r.network) {
          // No server (file://, qwen-web not running) or the WHOLE request
          // (headers and body) timed out — an honest disconnect, never a
          // schema verdict and never an error storm.
          setApiAvailable(false);
          renderChip('Not connected',
            r.timedOut ? 'Not connected — the request timed out'
                       : 'Not connected — no local engine server');
          setText('settings-backend',
            'No local engine server was reachable from this page (or the ' +
            'request timed out). Opened over file:// or without qwen-web, ' +
            'the workspace is disconnected: planning and authoritative ' +
            'validation are unavailable.');
          setText('settings-generation',
            'No authoritative status — generation availability unknown; ' +
            'Generate stays disabled.');
          return null;
        }
        if (!r.res.ok) {
          setApiAvailable(false);
          renderChip('Server error', 'Server error (status ' + r.res.status + ')');
          setText('settings-backend',
            'The local engine server answered the status request with an ' +
            'error; no authoritative information is available.');
          setText('settings-generation',
            'No authoritative status — generation availability unknown; ' +
            'Generate stays disabled.');
          return null;
        }
        if (r.bodyError) {
          // A COMPLETE body that is not usable JSON (or over the fixed cap)
          // is a REJECTED schema — explicitly not a disconnect.
          setApiAvailable(false);
          var why = r.bodyError === 'too_large' ? 'body over the size cap'
                    : r.bodyError === 'unsupported'
                      ? 'body could not be read bounded by this engine'
                      : 'not decodable JSON';
          renderChip('Status rejected', 'Status rejected — ' + why);
          setText('settings-backend', statusUnrecognizedText(
            r.bodyError === 'too_large' ? 'body over the size cap'
              : r.bodyError === 'unsupported' ? 'body reading unsupported'
                                              : 'body was not valid JSON'));
          return null;
        }
        var status = parseStatus(r.json);   // JSON null rejects safely here
        if (!status) {
          setApiAvailable(false);
          renderChip('Status rejected', 'Status rejected — not trusted');
          setText('settings-backend', statusUnrecognizedText(''));
          return null;
        }
        setApiAvailable(true);
        generationConnected = status.generation;
        updateGenerate();
        setRewriterState(status.rewrite);
        renderChip(status.generation ? 'Connected — engine jobs' : 'Connected — planning only',
                   status.generation ? 'Connected — engine jobs' : 'Connected — planning only');
        setText('settings-backend',
          'Connected to the local qwen-web server. Generation ' +
          (status.generation ? 'submission is enabled.' : 'is disabled or unconfigured.'));
        setText('settings-gates',
          'adapter: available · hardware: checked during each job · ' +
          'assets: not verified · calibration: not verified for this ' +
          'server (unknown)');
        setText('settings-generation',
          'FP16 resident generation uses the same engine as the CLI. ' +
          'Submission availability does not establish memory fitness.');
        if (status.generation) refreshJobs();
        return status;
      });
  }

  /* ---------------- Typed API errors ---------------- */

  function renderApiError(res, payload) {
    // Bounded rendering only: fixed-length code pattern, truncated strings,
    // at most 8 field lines — server error text is never a rendering risk.
    var message = 'The server refused the request (HTTP ' + res.status + ').';
    if (payload && payload.schema === ERROR_SCHEMA &&
        typeof payload.code === 'string' && /^[a-z0-9_]{1,64}$/.test(payload.code)) {
      message = payload.code + ': ' +
                (typeof payload.message === 'string'
                  ? payload.message.slice(0, 200) : 'no message');
      if (Array.isArray(payload.fields)) {
        var fieldTexts = [];
        for (var i = 0; i < payload.fields.length && fieldTexts.length < 8; i++) {
          var f = payload.fields[i];
          if (f && typeof f.field === 'string' && f.field.length <= 32 &&
              typeof f.message === 'string') {
            fieldTexts.push(f.field.slice(0, 32) + ': ' + f.message.slice(0, 160));
          }
        }
        if (fieldTexts.length) message += ' — ' + fieldTexts.join('; ');
      }
    }
    setText('api-message', message.slice(0, 600));
  }

  /* ---------------- Prompt rewriting (plan §3/§4) ----------------
   * A SUGGESTION, never an edit: the server returns a proposed prompt; the
   * draft only changes when the user presses Apply. Everything here is
   * gated on a strictly-parsed `available` rewriter block, and every
   * reply is checked against the draft it was requested for — a response
   * for older text is discarded, never applied to newer text.
   */

  var rewriteBackend = null;      // non-null ONLY when status said available
  var rewriteByTask = null;       // {t2i: id|null, edit: id|null} or null
  var rewriteBackends = null;     // the status `backends` rows, or null
  var rewriteSettingsHref = API_REWRITE_SETTINGS;
  var rewriteInFlight = false;
  // Monotonic PROMPT revision: bumped by every change to the prompt TEXT,
  // however it was made (typing, Apply, import). Deliberately NOT the plan
  // module's draftRevision: that one also moves on width/steps changes, and a
  // resolution tweak must not throw away a perfectly good prompt suggestion.
  // The server echoes the value it was given; a reply is only ever accepted
  // while the prompt has not moved since.
  var promptRevision = 0;
  var rewriteSeq = 0;
  var proposal = null;            // { prompt, original, revision }

  // Which rewriter answers ONE draft kind. A per-task server says so directly;
  // against an older one the single `backend` is all the page may claim, and the
  // server still decides which draft kinds it can actually take.
  function rewriteActiveFor(task) {
    if (rewriteByTask) return rewriteByTask[task] || null;
    return rewriteBackend;
  }

  // Whether Rewrite can be pressed right now. An edit rewrite names its
  // references by number and the server refuses one without them, so the
  // button is unavailable until at least one image is attached — the task is
  // derived from the references, so this can never ask for an impossible edit.
  // The rewriter for THAT draft kind is what gates it: a server with the text
  // enhancer only still rewrites text, and an edit draft says so by staying dark.
  function rewriteAvailable() {
    if (rewriteInFlight) return false;
    var task = rewriteTask();
    if (!rewriteActiveFor(task)) return false;
    return task === 't2i' || activeReferenceIds().length > 0;
  }

  function refreshRewriteButton() {
    var button = $('rewrite-btn');
    if (!button) return;
    button.hidden = !rewritingSomewhere();
    button.disabled = !rewriteAvailable();
  }

  // Any draft kind answered at all? That is what shows the Rewrite button;
  // `rewriteAvailable` decides whether THIS draft can use it.
  function rewritingSomewhere() {
    if (rewriteByTask) return hasAnyValue(rewriteByTask);
    return !!rewriteBackend;
  }

  function setRewriterState(rewriter) {
    rewriteBackend = rewriter ? rewriter.backend : null;
    rewriteByTask = rewriter && rewriter.byTask ? rewriter.byTask : null;
    rewriteBackends = rewriter && rewriter.backends ? rewriter.backends : null;
    if (rewriter && rewriter.settings) rewriteSettingsHref = rewriter.settings.href;
    var button = $('rewrite-btn');
    if (button) {
      button.hidden = !rewritingSomewhere();
      button.disabled = !rewriteAvailable();
    }
    var hint = $('rewrite-hint');
    if (hint) hint.hidden = rewritingSomewhere();
    // Which rewriter is answering is stated, not implied: the two behave
    // differently (one takes the graphics card, one does not), they are chosen
    // separately per draft kind, and the user just chose between them.
    var active = $('rewrite-active');
    if (active) {
      var sentence = rewriterSentence();
      active.hidden = !sentence;
      setText('rewrite-active-text', sentence);
    }
    if (!rewritingSomewhere()) hideProposal();
  }

  // The one honest sentence about who is rewriting what, built from the labels
  // the server sent. When the two kinds use different rewriters the page says
  // both rather than naming one and implying the other.
  function rewriterSentence() {
    var text = rewriteActiveFor('t2i');
    var edit = rewriteActiveFor('edit');
    if (text && edit && text === edit) {
      return 'Rewriter in use: ' + rewriteLabel(text) + '.';
    }
    var parts = [];
    if (text) parts.push('text drafts by ' + rewriteLabel(text));
    if (edit) parts.push('edit drafts by ' + rewriteLabel(edit));
    if (!parts.length) return '';
    var line = 'Rewriter in use: ' + parts.join(', ') + '.';
    if (!text || !edit) {
      line += ' ' + (!text ? 'Text' : 'Edit') + ' drafts have no rewriter here.';
    }
    return line;
  }

  // The friendly name for a backend id, taken from the status rows when the
  // server sent them. Never a guessed sentence, never an address or a path.
  function rewriteLabel(backend) {
    if (rewriteBackends) {
      for (var i = 0; i < rewriteBackends.length; i++) {
        if (rewriteBackends[i].id === backend) return rewriteBackends[i].label;
      }
    }
    return backend === 'pe-llamacpp' ? 'Official enhancer (Q8, uses GPU)'
      : backend === 'local-llm' ? 'OpenAI-compatible endpoint' : backend;
  }

  function bumpPromptRevision() {
    promptRevision++;
    // A newer draft invalidates any visible suggestion: it was derived from
    // text that is no longer on screen.
    if (proposal) hideProposal();
    var status = $('rewrite-status');
    if (status) status.textContent = '';
  }

  function hideProposal() {
    proposal = null;
    var block = $('rewrite-proposal');
    if (block) block.hidden = true;
    setText('rewrite-original', '');
    setText('rewrite-suggested', '');
    setText('rewrite-ratio', '');
    setText('rewrite-source', '');
    var applyRatio = $('rewrite-apply-ratio');
    if (applyRatio) applyRatio.hidden = true;
    setText('rewrite-error', '');
  }

  function rewriteTask() {
    // Reference images imply editing; the request must never claim a task the
    // draft's own references contradict.
    return activeReferenceIds().length ? 'edit' : 't2i';
  }

  function requestRewrite() {
    // The rewriter for THIS draft kind, not "some rewriter": an edit draft on a
    // text-only server must not fire a request the server will refuse.
    if (rewriteInFlight || !rewriteActiveFor(rewriteTask())) return;
    var draft = $('prompt').value;
    if (!draft.trim()) {
      setText('rewrite-error',
        'Write a draft first — the rewriter improves text, it does not invent a prompt.');
      return;
    }
    if (draft.length > PROMPT_MAX_CHARS) {
      // The server cap, refused locally: nothing is sent.
      setText('rewrite-error', 'The draft is ' + draft.length +
        ' characters; the rewriter accepts at most ' + PROMPT_MAX_CHARS + ' — nothing was sent.');
      return;
    }
    var task = rewriteTask();
    // The displayed order IS the conditioning order: reference N is <imageN>
    // in the edit system prompt, so the ids go out exactly as rendered.
    var imageIds = task === 'edit' ? activeReferenceIds() : [];
    if (task === 'edit' && !imageIds.length) {
      setText('rewrite-error', 'An edit rewrite needs at least one reference image.');
      return;
    }
    var revision = promptRevision;
    var seq = ++rewriteSeq;
    rewriteInFlight = true;
    $('rewrite-btn').disabled = true;
    $('rewrite-btn').setAttribute('aria-busy', 'true');
    setText('rewrite-status', 'Rewriting…');
    setText('rewrite-error', '');
    apiRequest(API_REWRITE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json',
                 'X-Qwen-UI': '1' },
      // The one call that legitimately needs minutes; still finite.
      timeoutMs: REWRITE_TIMEOUT_MS,
      body: JSON.stringify({
        schema: REWRITE_REQUEST_SCHEMA,
        task: task,
        prompt: draft,
        draft_revision: revision,
        image_ids: imageIds
      })
    }, 'json').then(function (r) {
      if (seq !== rewriteSeq) return;        // a newer attempt superseded this
      rewriteInFlight = false;
      var button = $('rewrite-btn');
      if (button) button.removeAttribute('aria-busy');
      refreshRewriteButton();
      setText('rewrite-status', '');
      if (r.network) {
        // The draft is untouched: this is a failure to CONSULT, never an edit.
        setText('rewrite-error', r.timedOut
          ? 'The rewriter did not answer in time; your draft is unchanged.'
          : 'The local engine server is unreachable; your draft is unchanged.');
        return;
      }
      if (r.bodyError) {
        setText('rewrite-error', 'The rewriter reply could not be read (' +
          (r.bodyError === 'too_large' ? 'response over the size cap' : 'not JSON') +
          '); your draft is unchanged.');
        return;
      }
      if (!r.res.ok) {
        renderRewriteError(r.res.status, r.json);
        return;
      }
      var answer = parseRewriteAnswer(r.json);
      if (!answer) {
        setText('rewrite-error', 'The rewriter reply did not match the expected ' +
          'schema; your draft is unchanged.');
        return;
      }
      // Stale discard, twice over: the revision must still be the current
      // one AND the text the server rewrote must still be the textarea's
      // content. Either mismatch means the answer is for text the user moved
      // away from — dropped silently, because it is not an error.
      if (answer.draft_revision !== promptRevision ||
          answer.original_prompt !== $('prompt').value) return;
      showProposal(answer);
    });
  }

  function renderRewriteError(status, payload) {
    var message = 'The rewriter refused the request (HTTP ' + status + ').';
    if (payload && payload.schema === ERROR_SCHEMA &&
        typeof payload.code === 'string' && /^[a-z0-9_]{1,64}$/.test(payload.code)) {
      message = payload.code + ': ' +
        (typeof payload.message === 'string' ? payload.message.slice(0, 200)
                                            : 'no message');
    }
    setText('rewrite-error', message.slice(0, 400) + ' Your draft is unchanged.');
  }

  // The answer document, strictly: exact schema, an echoed revision, the
  // original prompt, and a NON-EMPTY suggestion. A model that produced no
  // parsable prompt yields parse_ok false with an empty suggestion — that is
  // a usable diagnostic, rendered as text, never as an applyable proposal.
  function parseRewriteAnswer(s) {
    if (!s || typeof s !== 'object') return null;
    if (s.schema !== REWRITE_SCHEMA) return null;
    if (REWRITE_TASKS.indexOf(s.task) === -1) return null;
    if (!Number.isInteger(s.draft_revision) || s.draft_revision < 0) return null;
    if (typeof s.original_prompt !== 'string') return null;
    if (typeof s.rewritten_prompt !== 'string') return null;
    if (typeof s.parse_ok !== 'boolean') return null;
    if (typeof s.backend !== 'string' || !s.backend) return null;
    if (!Number.isInteger(s.elapsed_ms) || s.elapsed_ms < 0) return null;
    var ratio = s.ratio;
    if (ratio !== null) {
      if (!ratio || typeof ratio !== 'object') return null;
      if (ratio.kind === 'wh_ratio') {
        if (typeof ratio.value !== 'string' || !/^[0-9]{1,4}:[0-9]{1,4}$/.test(ratio.value)) return null;
      } else if (ratio.kind === 'ratio_follow') {
        if (!Number.isInteger(ratio.image) || ratio.image < 1 || ratio.image > 10) return null;
      } else return null;
    }
    if (typeof s.ratio_conflict !== 'boolean') return null;
    if (s.unparsed_answer !== null && typeof s.unparsed_answer !== 'string') return null;
    if (s.parse_ok && !s.rewritten_prompt.trim()) return null;  // parser invariant
    return s;
  }

  function showProposal(answer) {
    // Only a USABLE answer becomes pending state. An unparsable answer is
    // diagnostic text rendered in the block; keeping it as `proposal` would
    // let a forced apply write an empty draft.
    var usable = answer.parse_ok && !!answer.rewritten_prompt.trim();
    proposal = usable
      ? { prompt: answer.rewritten_prompt, original: answer.original_prompt,
          revision: answer.draft_revision }
      : null;
    var block = $('rewrite-proposal');
    if (!block) return;
    setText('rewrite-original', answer.original_prompt);
    setText('rewrite-suggested', answer.rewritten_prompt);
    // Which rewriter answered is stated, not hidden: the two differ in cost
    // (one takes the graphics card) and in what they can do, and for an edit
    // draft with the enhancer selected a DIFFERENT one answers than for a t2i
    // draft — the user is entitled to know which produced this text.
    setText('rewrite-source', 'Rewritten by ' + rewriteLabel(answer.backend) +
      ' in ' + answer.elapsed_ms + ' ms.');
    setText('rewrite-error', '');
    var apply = $('rewrite-apply');
    var edit = $('rewrite-edit');
    var keep = $('rewrite-keep');
    if (apply) apply.hidden = !usable;
    if (edit) edit.hidden = !usable;
    if (keep) keep.hidden = false;
    if (!usable) {
      setText('rewrite-error',
        'The rewriter did not return a usable prompt' +
        (answer.unparsed_answer
          ? ' — its answer was kept below as text, not as a proposal.'
          : '. Your draft is unchanged.'));
      setText('rewrite-suggested', answer.unparsed_answer || '');
    }
    // Ratio is a HINT: text always; a button ONLY when the value matches a
    // preset ratio exactly, so the control can never send a ratio the preset
    // table cannot represent.
    var ratioText = '';
    var ratioButton = $('rewrite-apply-ratio');
    var presetName = null;
    if (answer.ratio_conflict) {
      ratioText = 'The rewriter suggested both a ratio and an image to follow; ' +
                  'neither is offered.';
    } else if (answer.ratio && answer.ratio.kind === 'wh_ratio') {
      presetName = presetForRatio(answer.ratio.value);
      ratioText = 'Suggested aspect ratio: ' + answer.ratio.value +
        (presetName ? ' (available as a preset)' : ' (no preset matches exactly, so no button)');
    } else if (answer.ratio && answer.ratio.kind === 'ratio_follow') {
      ratioText = 'The rewriter suggests following reference image ' +
                  answer.ratio.image + '; pick a preset yourself.';
    }
    setText('rewrite-ratio', ratioText);
    if (ratioButton) {
      ratioButton.hidden = !presetName;
      if (presetName) ratioButton.dataset.preset = presetName;
    }
    block.hidden = false;
    block.focus();
  }

  // Exact ratio -> preset name, never a nearest match. Prefers the tier the
  // draft currently uses, so applying a hint does not silently quadruple the
  // resolution; falls back to 1K.
  function presetForRatio(ratio) {
    var currentTier = currentDraftTier();
    var fallback = null;
    for (var name in PRESETS) {
      if (!Object.prototype.hasOwnProperty.call(PRESETS, name)) continue;
      if (PRESETS[name].ratio !== ratio) continue;
      if (PRESETS[name].tier === currentTier) return name;
      if (!fallback) fallback = name;
    }
    return fallback;
  }

  function currentDraftTier() {
    var w = Number($('width').value);
    var h = Number($('height').value);
    for (var name in PRESETS) {
      if (!Object.prototype.hasOwnProperty.call(PRESETS, name)) continue;
      var p = PRESETS[name];
      if (p.width === w && p.height === h) return p.tier;
    }
    return '1K';
  }

  function applyProposal() {
    // `proposal` is only ever assigned a real, non-empty suggestion (the one
    // assignment site guards it), so a null here means "nothing pending".
    if (!proposal) return;
    $('prompt').value = proposal.prompt;
    hideProposal();
    bumpPromptRevision();
    promptTouched = true;          // a programmatic edit is a real edit
    renderValidation(validateForm().errors);
    syncPlanStaleness();
    $('prompt').focus();
  }

  function editProposal() {
    // Edit moves the suggestion INTO the textarea but keeps the user in
    // control: same as apply, then the caret is theirs.
    if (!proposal) return;
    $('prompt').value = proposal.prompt;
    hideProposal();
    bumpPromptRevision();
    promptTouched = true;
    renderValidation(validateForm().errors);
    syncPlanStaleness();
    var area = $('prompt');
    area.focus();
    area.setSelectionRange(area.value.length, area.value.length);
  }

  function keepOriginal() {
    hideProposal();
    $('prompt').focus();
  }

  // Observable state for the browser checks (no server, no model).
  function rewriteState() {
    return { backend: rewriteBackend, byTask: rewriteByTask,
             revision: promptRevision, inFlight: rewriteInFlight,
             hasProposal: !!proposal,
             proposal: proposal ? proposal.prompt : null };
  }

  function apiErrorPayloadFromBytes(bytes) {
    // Error-path only: a server error body is parsed defensively for
    // bounded rendering. It never feeds a success render.
    if (!bytes) return null;
    try {
      return JSON.parse(new TextDecoder('utf-8', { fatal: true })
        .decode(bytes));
    } catch (err) {
      return null;
    }
  }

  /* ---------------- Prompt rewriter settings ----------------
   * The Settings page changes ONE thing that belongs to the server rather
   * than to this browser: which rewriter answers, and how the local service
   * is reached. It is stored by the server in its own cache directory and
   * applies to every browser using it, so NOTHING here is kept in the browser
   * — no localStorage key, no draft — and no field names a file: the
   * llama-server binary, its GGUF and the system prompts stay launch flags
   * the operator owns, and the server refuses any path-shaped key.
   */

  var rewriteSettings = null;        // the last state the server sent
  var rewriteSettingsBusy = false;
  var rewriteTestBusy = false;

  // What the FORM shows, which is deliberately NOT what the server holds: a
  // radio or switch the user moved is a proposal until Save comes back. The
  // renderer needs both, because it redraws labels, field muting and button
  // states on every click — and redrawing from the server document alone would
  // snap the user's click straight back.
  var rewriteForm = null;   // {t2i, edit, enabled, endpoint, model, timeout_s} or null

  function rewriteFormState() {
    if (rewriteForm) return rewriteForm;
    if (!rewriteSettings) return null;
    return { t2i: rewriteSettings.choices.t2i, edit: rewriteSettings.choices.edit,
             enabled: rewriteSettings.enabled,
             endpoint: rewriteSettings.local.endpoint,
             model: rewriteSettings.local.model,
             timeout_s: rewriteSettings.local.timeout_s };
  }

  function noteRewriteChoice(task, id) {
    rewriteForm = rewriteFormState() || { t2i: null, edit: null, enabled: true,
                                          endpoint: null, model: null,
                                          timeout_s: null };
    rewriteForm[task] = id;
  }

  function noteRewriteEnabled(enabled) {
    rewriteForm = rewriteFormState() || { t2i: null, edit: null, enabled: true,
                                          endpoint: null, model: null,
                                          timeout_s: null };
    rewriteForm.enabled = !!enabled;
  }

  // The three endpoint boxes are part of the same pending form: an unrelated
  // click (a radio, the Off switch) redraws the page, and a redraw that refills
  // them from the server document would delete an address the user was typing.
  function noteRewriteFields(fields) {
    rewriteForm = rewriteFormState() || { t2i: null, edit: null, enabled: true,
                                          endpoint: null, model: null,
                                          timeout_s: null };
    rewriteForm.endpoint = fields.endpoint;
    rewriteForm.model = fields.model;
    rewriteForm.timeout_s = fields.timeout_s;
  }

  // The settings document, strictly — the same closed shape the server
  // validates before it sends it. Anything unexpected rejects the document
  // rather than rendering a partial guess about which rewriter is live.
  function parseRewriteSettings(payload) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null;
    var keys = Object.keys(payload);
    for (var i = 0; i < keys.length; i++) {
      if (REWRITE_SETTINGS_KEYS.indexOf(keys[i]) === -1) return null;
    }
    if (payload.schema !== REWRITE_SETTINGS_SCHEMA) return null;
    if (typeof payload.enabled !== 'boolean') return null;
    if (typeof payload.persistent !== 'boolean') return null;
    var choices = parseRewriteChoices(payload.choices);
    var saved = parseRewriteChoice(payload.saved);
    var local = parseRewriteLocal(payload.local);
    if (choices === null || saved === null || local === null) return null;
    var rows = parseRewriteSettingsRows(payload.backends);
    if (rows === null) return null;
    var groups = parseRewriteTaskRows(payload.tasks);
    if (groups === null) return null;
    var off = payload.off;
    if (!off || off.id !== 'off' || typeof off.label !== 'string') return null;
    if (payload.note !== undefined && typeof payload.note !== 'string') return null;
    return {
      enabled: payload.enabled, choices: choices, saved: saved, local: local,
      backends: rows, tasks: groups,
      off: { id: 'off', label: off.label }, persistent: payload.persistent,
      note: typeof payload.note === 'string' ? payload.note : ''
    };
  }

  // The choice IN FORCE per draft kind: one entry per known task, each a
  // rewriter id. Never null — the server resolves its default, so the page
  // always has a radio to tick rather than an empty group.
  function parseRewriteChoices(choices) {
    if (!choices || typeof choices !== 'object' || Array.isArray(choices)) return null;
    if (Object.keys(choices).length !== REWRITE_TASKS.length) return null;
    var out = {};
    for (var i = 0; i < REWRITE_TASKS.length; i++) {
      var task = REWRITE_TASKS[i];
      if (REWRITE_BACKEND_IDS.indexOf(choices[task]) === -1) return null;
      out[task] = choices[task];
    }
    return out;
  }

  // The three local-service values, shared by the saved choice and by the block
  // that describes the local service: each is null (nothing typed there, so the
  // launch flags stay in force) or a bounded scalar. The bounds are the server's
  // own; re-checking them is what makes a hand-edited or corrupted document
  // render nothing instead of a partial guess.
  function parseRewriteLocalFields(values) {
    var out = {};
    if (values.endpoint !== null) {
      if (typeof values.endpoint !== 'string' || values.endpoint.length > 512) return null;
      out.endpoint = values.endpoint;
    } else { out.endpoint = null; }
    if (values.model !== null) {
      if (typeof values.model !== 'string' || values.model.length > 256) return null;
      out.model = values.model;
    } else { out.model = null; }
    if (values.timeout_s !== null) {
      if (typeof values.timeout_s !== 'number' || !Number.isFinite(values.timeout_s) ||
          values.timeout_s <= 0 || values.timeout_s > 3600) return null;
      out.timeout_s = values.timeout_s;
    } else { out.timeout_s = null; }
    return out;
  }

  function hasOnlyKeys(object, keys) {
    var names = Object.keys(object);
    for (var i = 0; i < names.length; i++) {
      if (keys.indexOf(names[i]) === -1) return false;
    }
    return true;
  }

  // The SAVED choice: the per-task rewriters (each null when the user never
  // picked one), the on/off switch, and the three shared endpoint values. Null
  // means "not saved", which is a different fact from "off".
  function parseRewriteChoice(choice) {
    if (!choice || typeof choice !== 'object' || Array.isArray(choice)) return null;
    if (!hasOnlyKeys(choice, REWRITE_CHOICE_KEYS)) return null;
    var out = {};
    for (var i = 0; i < REWRITE_TASKS.length; i++) {
      var task = REWRITE_TASKS[i];
      var value = choice[task];
      if (value === null || value === undefined) {
        out[task] = null;
      } else if (REWRITE_BACKEND_IDS.indexOf(value) !== -1) {
        out[task] = value;
      } else {
        return null;
      }
    }
    if (choice.enabled === null || choice.enabled === undefined) {
      out.enabled = null;
    } else if (typeof choice.enabled === 'boolean') {
      out.enabled = choice.enabled;
    } else {
      return null;
    }
    var fields = parseRewriteLocalFields(choice);
    if (fields === null) return null;
    out.endpoint = fields.endpoint;
    out.model = fields.model;
    out.timeout_s = fields.timeout_s;
    return out;
  }

  // The block that describes the LOCAL service as this server runs it: the same
  // three values and no backend, because the page does not choose a backend by
  // editing it.
  function parseRewriteLocal(local) {
    if (!local || typeof local !== 'object' || Array.isArray(local)) return null;
    if (!hasOnlyKeys(local, REWRITE_LOCAL_KEYS)) return null;
    return parseRewriteLocalFields(local);
  }

  // One radio group per draft kind. The server has already resolved whether each
  // rewriter can be chosen for THAT kind (PE-I2I missing is not the page's
  // business) and why not, so a greyed-out radio arrives with its reason and the
  // page renders it verbatim instead of inventing one.
  function parseRewriteTaskRows(rows) {
    if (!Array.isArray(rows) || rows.length !== REWRITE_TASKS.length) return null;
    var out = [];
    for (var i = 0; i < REWRITE_TASKS.length; i++) {
      var row = rows[i];
      if (!row || typeof row !== 'object' || Array.isArray(row)) return null;
      if (Object.keys(row).length !== REWRITE_TASK_ROW_KEYS.length ||
          !hasOnlyKeys(row, REWRITE_TASK_ROW_KEYS)) return null;
      if (row.task !== REWRITE_TASKS[i]) return null;   // the server's order
      // The label is not free text: it names the draft kind this group is for.
      if (REWRITE_TASK_LABELS[row.task] !== row.label) return null;
      var choices = row.choices;
      if (!Array.isArray(choices) || choices.length !== REWRITE_BACKEND_IDS.length) return null;
      var picked = [];
      for (var c = 0; c < choices.length; c++) {
        var choice = choices[c];
        if (!choice || typeof choice !== 'object' || Array.isArray(choice)) return null;
        if (Object.keys(choice).length !== REWRITE_TASK_CHOICE_KEYS.length ||
            !hasOnlyKeys(choice, REWRITE_TASK_CHOICE_KEYS)) return null;
        if (REWRITE_BACKEND_IDS.indexOf(choice.id) === -1) return null;
        if (REWRITE_BACKEND_LABELS[choice.id] !== choice.label ||
            REWRITE_BACKEND_USES_GPU[choice.id] !== choice.uses_gpu) return null;
        if (typeof choice.uses_gpu !== 'boolean' ||
            typeof choice.available !== 'boolean' ||
            typeof choice.reason !== 'string') return null;
        // Greyed out means the server owes the user a sentence, and an
        // available option must not carry one: a reason beside a radio the user
        // CAN click reads as a warning about something they are about to do.
        if (choice.available !== (choice.reason === '')) return null;
        if (containsId(picked, choice.id)) return null;  // one row per rewriter
        picked.push({ id: choice.id, label: choice.label,
                      uses_gpu: choice.uses_gpu, available: choice.available,
                      reason: choice.reason });
      }
      out.push({ task: row.task, label: row.label, choices: picked });
    }
    return out;
  }

  // The settings form's rows: every backend that exists on this server, with
  // the reason one of them is greyed out. `available` is what the LAUNCH FLAGS
  // make possible; no value typed here can change it.
  function parseRewriteSettingsRows(rows) {
    if (!Array.isArray(rows) || rows.length !== 2) return null;
    var out = [];
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      if (!row || typeof row !== 'object' || Array.isArray(row)) return null;
      var keys = Object.keys(row);
      for (var k = 0; k < keys.length; k++) {
        if (REWRITE_SETTINGS_ROW_KEYS.indexOf(keys[k]) === -1) return null;
      }
      if (REWRITE_SETTINGS_BACKENDS.indexOf(row.id) === -1) return null;
      if (containsId(out, row.id)) return null;   // one row per rewriter
      if (REWRITE_BACKEND_LABELS[row.id] !== row.label ||
          REWRITE_BACKEND_USES_GPU[row.id] !== row.uses_gpu) return null;
      if (typeof row.available !== 'boolean') return null;
      if (typeof row.reason !== 'string') return null;
      // The same rule as a task choice: unavailable comes with a sentence.
      if (row.available !== (row.reason === '')) return null;
      // `tasks` may be EMPTY here: a checkpoint that is installed but cannot
      // answer either draft kind is a real server shape, and its row is how the
      // page says so. What it may never be is a list of draft kinds this page
      // has no group for.
      if (!Array.isArray(row.tasks)) return null;
      for (var t = 0; t < row.tasks.length; t++) {
        if (REWRITE_TASKS.indexOf(row.tasks[t]) === -1) return null;
      }
      out.push({ id: row.id, label: row.label, tasks: row.tasks.slice(),
                 uses_gpu: row.uses_gpu, available: row.available,
                 reason: row.reason });
    }
    return out;
  }

  function rewriteRow(id) {
    if (!rewriteSettings) return null;
    for (var i = 0; i < rewriteSettings.backends.length; i++) {
      if (rewriteSettings.backends[i].id === id) return rewriteSettings.backends[i];
    }
    return null;
  }

  // One draft kind's group, as the server described it.
  function rewriteTaskRow(task) {
    if (!rewriteSettings) return null;
    for (var i = 0; i < rewriteSettings.tasks.length; i++) {
      if (rewriteSettings.tasks[i].task === task) return rewriteSettings.tasks[i];
    }
    return null;
  }

  // One rewriter's row FOR one draft kind: the label, whether it can be chosen
  // for that kind here, and the server's own sentence if it cannot.
  function rewriteChoiceRow(task, id) {
    var row = rewriteTaskRow(task);
    if (!row) return null;
    for (var i = 0; i < row.choices.length; i++) {
      if (row.choices[i].id === id) return row.choices[i];
    }
    return null;
  }

  function selectedRewriteChoice(task) {
    var radios = document.querySelectorAll('input[name="rewrite-' + task + '"]');
    for (var i = 0; i < radios.length; i++) {
      if (radios[i].checked) return radios[i].value;
    }
    return null;
  }

  // The single Off switch. Ticking the box means OFF, so the document's
  // `enabled` is the negation — one sentence on screen, one boolean on the wire.
  function rewriteEnabled() {
    var box = $('rewrite-off');
    return box ? !box.checked : true;
  }

  // The three endpoint inputs. An empty box is null — "keep what the server was
  // started with" — never an empty string, which the server would refuse as a
  // malformed address or name.
  function rewriteEndpointFields() {
    var timeoutRaw = pyStrip(String($('rewrite-timeout').value));
    var timeout = null;
    if (timeoutRaw !== '') timeout = Number(timeoutRaw);
    return {
      endpoint: pyStrip(String($('rewrite-endpoint').value)) || null,
      model: pyStrip(String($('rewrite-model').value)) || null,
      timeout_s: timeout
    };
  }

  function rewriteSettingsEnvelope() {
    var fields = rewriteEndpointFields();
    fields.schema = REWRITE_SETTINGS_SCHEMA;
    fields.t2i = selectedRewriteChoice('t2i');
    fields.edit = selectedRewriteChoice('edit');
    fields.enabled = rewriteEnabled();
    return fields;
  }

  // The probe addresses the shared endpoint ALONE: no choice keys, and its own
  // schema, because testing an address is not a change of rewriter.
  function rewriteTestEnvelope() {
    var fields = rewriteEndpointFields();
    fields.schema = REWRITE_TEST_REQUEST_SCHEMA;
    return fields;
  }

  function loadRewriteSettings() {
    setText('rewrite-settings-error', '');
    return apiRequest(API_REWRITE_SETTINGS,
      { headers: { 'Accept': 'application/json', 'X-Qwen-UI': '1' } }, 'json')
      .then(function (r) {
        if (r.network) {
          setText('rewrite-settings-error',
            'The local engine server is unreachable, so the rewriter settings ' +
            'could not be read.');
          return null;
        }
        var payload = r.json;
        var state = parseRewriteSettings(payload);
        if (!state) {
          // A refusal is information the user asked for; an unreadable
          // document is not a setting, so nothing is pre-filled from it.
          var err = r.res && r.res.ok ? null : payload;
          setText('rewrite-settings-error', err && typeof err.message === 'string'
            ? err.code + ': ' + err.message.slice(0, 200)
            : 'The rewriter settings did not match the expected schema, so ' +
              'nothing on this page was taken from them.');
          rewriteSettings = null;
          rewriteForm = null;      // nothing on screen came from the server
        } else {
          rewriteSettings = state;
          rewriteForm = null;      // the form now shows exactly this document
        }
        renderRewriteSettings();
        return state;
      });
  }

  // The short name the page's own element ids use for a rewriter id.
  function rewriteShortName(id) {
    return id === 'pe-llamacpp' ? 'pe' : id === 'local-llm' ? 'local' : id;
  }

  function renderRewriteSettings() {
    var loaded = !!rewriteSettings;
    var form = rewriteFormState();   // what the user sees, not what is saved
    // One group per draft kind. Labels and reasons are the server's words, so a
    // renamed rewriter or a new availability sentence shows up without a page
    // change; a group the document does not describe renders nothing rather
    // than a guess.
    for (var g = 0; g < REWRITE_TASKS.length; g++) {
      var task = REWRITE_TASKS[g];
      var group = rewriteTaskRow(task);
      var legend = $('rewrite-group-' + task);
      if (legend && group) setText(legend.id, group.label);
      var radios = document.querySelectorAll('input[name="rewrite-' + task + '"]');
      for (var i = 0; i < radios.length; i++) {
        var radio = radios[i];
        var choice = rewriteChoiceRow(task, radio.value);
        var chosen = loaded && !!form && form[task] === radio.value;
        var available = !!(choice && choice.available);
        // An unavailable rewriter stays VISIBLE with its reason, and a choice
        // the user already made stays ticked even when this server cannot honour
        // it: the page shows what is saved, so putting the missing file in place
        // fixes it without choosing twice.
        radio.checked = chosen;
        radio.disabled = !loaded || (!available && !chosen);
        var suffix = '-' + task + '-' + rewriteShortName(radio.value);
        if (choice) setText('rewrite' + suffix + '-label', choice.label);
        setText('rewrite' + suffix + '-reason', !loaded || available ? ''
          : (choice && choice.reason ? choice.reason
             : 'this server cannot run that rewriter for this kind of draft'));
      }
    }
    var off = $('rewrite-off');
    if (off) {
      off.disabled = !loaded;
      off.checked = loaded && !!form ? !form.enabled : false;
    }
    setText('rewrite-off-reason', loaded && !!form && !form.enabled
      ? 'rewriting is switched off: both kinds of draft are used exactly as ' +
        'you wrote them' : '');
    // The three boxes show what the USER typed, or — with nothing typed yet —
    // what the server sent. Refilling them from the document on every redraw
    // would erase an address mid-word the moment another control is clicked.
    var local = form || { endpoint: null, model: null, timeout_s: null };
    $('rewrite-endpoint').value = local.endpoint == null ? '' : String(local.endpoint);
    $('rewrite-model').value = local.model == null ? '' : String(local.model);
    $('rewrite-timeout').value = local.timeout_s == null ? '' : String(local.timeout_s);
    // The fields describe the SHARED endpoint, so they belong to whichever group
    // chooses it — and stay readable when neither does, because a greyed-out
    // official rewriter still leaves this as the fallback the user must be able
    // to see and fix.
    var usesEndpoint = loaded && (selectedRewriteChoice('t2i') === 'local-llm' ||
                                  selectedRewriteChoice('edit') === 'local-llm');
    ['rewrite-endpoint', 'rewrite-model', 'rewrite-timeout'].forEach(function (id) {
      var input = $(id);
      if (input) input.disabled = !loaded;
      var box = input ? input.closest('.field') : null;
      if (box) box.classList.toggle('field-muted', loaded && !usesEndpoint);
    });
    var test = $('rewrite-test');
    if (test) {
      test.disabled = !loaded || rewriteTestBusy ||
        !pyStrip(String($('rewrite-endpoint').value));
    }
    var save = $('rewrite-save');
    if (save) {
      save.disabled = !loaded || rewriteSettingsBusy ||
        !(selectedRewriteChoice('t2i') && selectedRewriteChoice('edit'));
    }
    setText('rewrite-settings-note', loaded ? rewriteSettings.note : '');
    if (!loaded) {
      setText('rewrite-save-status', '');
      setText('rewrite-test-status', '');
    }
  }

  function saveRewriteSettings() {
    if (rewriteSettingsBusy) return;
    var envelope = rewriteSettingsEnvelope();
    if (!envelope.t2i || !envelope.edit) {
      // Both groups must carry a choice; the server refuses a partial document
      // rather than guess what the other kind of draft should use.
      setText('rewrite-settings-error',
        'Choose a rewriter for both kinds of draft first.');
      return;
    }
    rewriteSettingsBusy = true;
    $('rewrite-save').disabled = true;
    $('rewrite-save').setAttribute('aria-busy', 'true');
    setText('rewrite-save-status', 'Saving…');
    setText('rewrite-settings-error', '');
    apiRequest(API_REWRITE_SETTINGS, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json',
                 'X-Qwen-UI': '1' },
      body: JSON.stringify(envelope)
    }, 'json').then(function (r) {
      rewriteSettingsBusy = false;
      var button = $('rewrite-save');
      if (button) button.removeAttribute('aria-busy');
      if (r.network) {
        setText('rewrite-save-status', '');
        setText('rewrite-settings-error',
          'The local engine server is unreachable; nothing was changed.');
        renderRewriteSettings();
        return;
      }
      var state = parseRewriteSettings(r.json);
      if (!r.res.ok || !state) {
        var payload = r.json;
        setText('rewrite-save-status', '');
        setText('rewrite-settings-error',
          payload && typeof payload.message === 'string'
            ? (payload.code || 'refused') + ': ' + payload.message.slice(0, 200)
            : 'The server refused the change; the settings are unchanged.');
        // The server is authoritative: re-read rather than keep a form the
        // server did not accept.
        loadRewriteSettings();
        return;
      }
      rewriteSettings = state;   // the server answered: the proposal IS the state
      rewriteForm = null;
      // Say what the NEXT rewrite will actually do, per draft kind, in the same
      // words the groups use — a saved "Official" that had to fall back would
      // otherwise read as a promise the server never made.
      var sentence;
      if (!state.enabled) {
        sentence = 'Saved. Rewriting is switched off.';
      } else if (state.choices.t2i === state.choices.edit) {
        sentence = 'Saved. The next rewrite uses ' +
          rewriteLabel(state.choices.t2i) + '.';
      } else {
        sentence = 'Saved. Text drafts will use ' +
          rewriteLabel(state.choices.t2i) + ', edit drafts ' +
          rewriteLabel(state.choices.edit) + '.';
      }
      setText('rewrite-save-status',
        sentence + (state.persistent ? ''
          : ' — but this server has no cache directory, so the change lasts ' +
            'only until it restarts.'));
      renderRewriteSettings();
      checkStatus();   // the Create view names the rewriter it will use
    });
  }

  function testRewriteConnection() {
    if (rewriteTestBusy) return;
    var envelope = rewriteTestEnvelope();   // the endpoint alone, never a choice
    rewriteTestBusy = true;
    $('rewrite-test').disabled = true;
    $('rewrite-test').setAttribute('aria-busy', 'true');
    setText('rewrite-test-status', 'Testing…');
    apiRequest(API_REWRITE_SETTINGS_TEST, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json',
                 'X-Qwen-UI': '1' },
      body: JSON.stringify(envelope),
      // A probe is bounded by the server; this only stops the page waiting on
      // a socket that never answers.
      timeoutMs: REWRITE_TIMEOUT_MS
    }, 'json').then(function (r) {
      rewriteTestBusy = false;
      var button = $('rewrite-test');
      if (button) button.removeAttribute('aria-busy');
      renderRewriteSettings();
      if (r.network) {
        setText('rewrite-test-status',
          'The local engine server is unreachable, so nothing could be tested.');
        return;
      }
      var result = r.json;
      if (!result || result.schema !== REWRITE_TEST_SCHEMA ||
          typeof result.ok !== 'boolean' || typeof result.detail !== 'string') {
        setText('rewrite-test-status',
          'The test reply did not match the expected schema.');
        return;
      }
      setText('rewrite-test-status',
        (result.ok ? 'Works: ' : 'Failed: ') + result.detail.slice(0, 200) +
        ' (' + result.elapsed_ms + ' ms)');
    });
  }

  /* ---------------- Plan flow ----------------
   * Both "Check configuration" and "Plan" submit the snapshot; the server
   * verdict is authoritative. On ANY failure the draft stays exactly as
   * typed — nothing is auto-filled or overwritten from a failed response.
   * Previous plan results never masquerade as current: the panel is hidden
   * while a new attempt is in flight, out-of-order answers are ignored and
   * a draft that changed meanwhile is labelled as a submitted snapshot.
   */

  var planSeq = 0;
  var planNoteBase = '';          // the note of the currently RENDERED result
  var planShownSignature = null;  // draft signature that panel validated
                                  // (null: no rendered result to keep fresh)
  // Monotonic DRAFT revision: bumped synchronously by syncPlanStaleness,
  // which every existing input/preset/random/change handler already calls.
  // An in-flight import captures it BEFORE the request and re-checks it
  // before applying, so a draft edit (or a newer import) that happened
  // while the request was in flight can never be silently overwritten.
  var draftRevision = 0;
  var importSeq = 0;              // newest import wins; late answers die

  function envelopeSignature(built) {
    return built.ok ? JSON.stringify(built.envelope) : null;
  }

  // A visible validated result must NEVER keep looking current after the
  // draft changed — including preset, random and change-only fields that
  // produce no inline errors. The panel stays visible but is clearly
  // labelled stale; nothing is silently re-trusted.
  function syncPlanStaleness() {
    draftRevision++;   // any observable draft change bumps the revision
    var panel = $('plan-panel');
    if (panel.hidden || planShownSignature === null) return;
    var stale = envelopeSignature(buildEnvelope()) !== planShownSignature;
    if (!stale) {
      panel.classList.remove('stale');
      setText('plan-panel-note', planNoteBase);
      return;
    }
    panel.classList.add('stale');
    setText('plan-panel-note', planNoteBase + ' The draft changed after this ' +
      'plan was computed — this result is STALE and does not validate the ' +
      'current draft. Run Plan again.');
  }

  function runPlan(showFull) {
    setText('api-message', '');
    // Invalidate prior attempts BEFORE any validation: a newer attempt —
    // even one that fails local validation — must cancel an in-flight
    // older one, so it can never clear the new errors or render an old
    // success over them.
    planSeq++;
    var seq = planSeq;
    var built = buildEnvelope();
    if (!built.ok) {
      promptTouched = true;
      renderValidation(built.errors);
      syncPlanStaleness();   // an invalid draft stales any visible result
      setText('api-message', 'The draft has local form errors — nothing was sent. ' +
                             'The server verdict would be authoritative anyway.');
      return Promise.resolve(false);
    }
    var sentSignature = JSON.stringify(built.envelope);
    planShownSignature = null;
    $('plan-panel').hidden = true;   // previous results are not current now
    return apiRequest(API_PLAN, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Qwen-UI': '1' },
      body: sentSignature
    }, 'json').then(function (r) {
      if (seq !== planSeq) return false;      // a newer attempt superseded this
      if (r.network) {
        setApiAvailable(false);
        renderChip('Not connected',
          r.timedOut ? 'Not connected — the request timed out'
                     : 'Not connected — request failed');
        setText('api-message', 'The request failed (no reachable local server, ' +
                               'or it timed out). No planning result received; ' +
                               'your draft is unchanged; nothing was generated.');
        return false;
      }
      if (!r.res.ok) {
        renderApiError(r.res, r.json === undefined ? null : r.json);
        applyServerFieldErrors(r.json);
        setText('api-message', $('api-message').textContent +
          ' No planning result received; your draft is unchanged; nothing ' +
          'was generated.');
        return false;
      }
      if (r.bodyError) {
        // A rejected PLAN document invalidates the recognized status too:
        // the server contract is broken, so planning must not stay enabled
        // on a stale 'Connected'. This is a typed schema rejection — an
        // explicit re-check is required to trust the connection again.
        setApiAvailable(false);
        renderChip('Status rejected',
          'Status rejected — server contract broken');
        setText('api-message', 'The plan body was not usable (' +
                               (r.bodyError === 'too_large'
                                 ? 'over the fixed response cap'
                                 : r.bodyError === 'unsupported'
                                   ? 'unreadable bounded body'
                                   : 'not decodable JSON') +
                               ') — a schema rejection, not a disconnect; ' +
                               'nothing is rendered or trusted. Use ' +
                               '"Check configuration" to re-establish a ' +
                               'status before planning again.');
        return false;
      }
      var plan = parsePlan(r.json);
      // A coherent plan for DIFFERENT settings is not validation of the
      // submitted snapshot. Compare the complete editable echo, independent
      // of JSON key order, before granting any current/stale plan label.
      if (plan && Object.keys(built.envelope.config).some(function (key) {
        return plan.config[key] !== built.envelope.config[key];
      })) plan = null;
      if (!plan) {
        setApiAvailable(false);
        renderChip('Status rejected',
          'Status rejected — server contract broken');
        setText('api-message', 'The plan document did not match the expected ' +
                               'schema or submitted configuration — nothing is rendered or trusted; ' +
                               'previous results are hidden and your draft is ' +
                               'unchanged. Use "Check configuration" to ' +
                               're-establish a recognized status before ' +
                               'planning again.');
        return false;
      }
      promptTouched = true;
      // Render the CURRENT draft's inline validation — an answer arriving
      // for a draft that changed meanwhile must NOT clear newer errors.
      renderValidation(validateForm().errors);
      planRenderFields(plan);
      planNoteBase = showFull
        ? 'Server plan (authoritative GenerationConfig validation) of the ' +
          'submitted snapshot. Planning only — nothing was generated.'
        : 'The server accepted this configuration (authoritative ' +
          'validation) of the submitted snapshot. Nothing was generated.';
      planShownSignature = sentSignature;
      setText('plan-panel-note', planNoteBase);
      $('plan-panel').classList.remove('stale');
      $('plan-panel').hidden = false;
      syncPlanStaleness();   // edited while awaiting the answer -> stale
      if (!apiConnected) { checkStatus(); }  // server proved itself; refresh
      return true;
    }).catch(function () {
      // Defensive: the network layer never rejects; anything reaching here
      // is a client-side bug, reported honestly, never as a fake success.
      if (seq !== planSeq) return false;
      setText('api-message', 'An unexpected client-side error occurred while ' +
                             'rendering the plan; no result is shown and ' +
                             'your draft is unchanged.');
      return false;
    });
  }

  function planRenderFields(plan) {
    setText('plan-seed', plan.config.seed === null ? 'null (random at run time)' : plan.config.seed);
    setText('plan-dims', plan.config.width + ' × ' + plan.config.height);
    setText('plan-steps', String(plan.config.steps));
    setText('plan-recipe', plan.config.recipe === 'turbo'
      ? 'Turbo — Turbo image model with the same text encoder and decoder ' +
        'as Quality, fixed ' + TURBO_STEPS + ' sampling steps'
      : 'Quality — reference model at ' + plan.config.steps + ' steps');
    setText('plan-cfg', plan.cfg_active === true ? 'active (scale ' + plan.config.true_cfg_scale + ')'
                                                 : 'inactive (scale ' + plan.config.true_cfg_scale + ')');
    setText('plan-tokens', String(plan.latent_tokens));
    setText('plan-quant', plan.config.quant + ' · ' + plan.config.offload + ' · ' +
                          plan.config.attention_backend + ' · VAE tiling ' +
                          (plan.config.vae_tiled_decode ? 'on' : 'off'));
    setText('plan-transparency',
      plan.config.transparent_background
        ? 'on — prompts the model for a transparent background; saved as ' +
          plan.config.output_format + ', which can keep it (not guaranteed)'
        : 'off — saved as ' + plan.config.output_format);
    setText('plan-references', plan.config.references.length === 0
      ? 'none — the prompt is used on its own'
      : plan.config.references.length + ' image' +
        (plan.config.references.length === 1 ? '' : 's') +
        ' (image 1' + (plan.config.references.length > 1
          ? '…image ' + plan.config.references.length : '') +
        ') — cited by number in the prompt');
    setText('plan-manifest',
      plan.manifest.ready === false && plan.manifest.verification === 'not_performed'
        ? 'metadata only — verification not performed, fit not evaluated, not ready'
        : 'manifest state unrecognized — not trusted');
    setText('plan-hardware', 'unknown — this server has not probed hardware');
    setText('plan-calibration', 'unknown — not verified for this server');
    var list = $('plan-blockers');
    while (list.firstChild) list.removeChild(list.firstChild);
    var blockers = Array.isArray(plan.generation_blockers) ? plan.generation_blockers : [];
    for (var i = 0; i < blockers.length; i++) {
      var li = document.createElement('li');
      li.textContent = String(blockers[i]);
      list.appendChild(li);
    }
  }

  var PLAN_ENUMS = {
    quant: ['fp16', 'q8'],
    offload: ['resident', 'streamed', 'offload'],
    attention_backend: ['auto', 'sdpa', 'efficient', 'eager'],
    output_format: OUTPUT_FORMATS,
    recipe: RECIPES
  };

  // The recipe gates config.py enforces, mirrored so an echo that could not
  // have come from a validated config is refused rather than rendered:
  // Turbo owns exactly TURBO_STEPS steps, runs at CFG 1.0 (no usable
  // negative prompt), and has no reviewed Q8 artifact.
  function recipeConsistent(c) {
    if (PLAN_ENUMS.recipe.indexOf(c.recipe) === -1) return false;
    if (c.recipe !== 'turbo') return true;
    return c.steps === TURBO_STEPS && c.quant === 'fp16' &&
           !(pyStrip(c.negative_prompt).length > 0 && c.true_cfg_scale > 1);
  }

  // The WHOLE public plan response is validated before anything renders:
  // exact editable keys, string lengths ALIGNED with the server's 64 KiB
  // transport cap (no invented generation limits), integer ranges,
  // dimensions multiples of 32, finite CFG, enums, boolean VAE, seed as
  // decimal STRING/null in uint64 range (a numeric seed is refused — never
  // rendered through Number), cfg_active and latent_tokens CONSISTENT with
  // the echoed config (the server's own derivations recomputed exactly),
  // the manifest EXACTLY verification not_performed / fit not_evaluated /
  // ready false, unknown hardware/calibration, and blocker codes from the
  // closed list only.
  function parsePlan(raw) {
    if (!raw || typeof raw !== 'object' || raw.schema !== PLAN_SCHEMA) return null;
    if (raw.planning_available !== true || typeof raw.generation_available !== 'boolean') return null;
    if (typeof raw.cfg_active !== 'boolean') return null;
    if (!Number.isInteger(raw.latent_tokens) || raw.latent_tokens < 0) return null;
    var c = raw.config;
    if (!c || typeof c !== 'object') return null;
    var want = EDITABLE_KEYS;
    var have = Object.keys(c);
    var hasKey = function (key) {
      return Object.prototype.hasOwnProperty.call(c, key);
    };
    // Backward compatibility is EXPLICIT, never "any key count": an older
    // server that predates the recipe sent 14 keys, the one before
    // references 13, the one before transparency 12, the one before
    // output_format 11. Each branch fills in exactly what that server
    // actually did — a pre-recipe server could only run the reference
    // model, so recipe is "base"; 10 keys, 16 keys or any unknown key is a
    // schema rejection, never a guess.
    var legacyNoRecipe = have.length === 14 && !hasKey('recipe');
    var legacyNoReferences = have.length === 13 &&
      !hasKey('references') && !hasKey('recipe');
    var legacyNoTransparency = have.length === 12 &&
      !hasKey('references') && !hasKey('recipe') &&
      !hasKey('transparent_background');
    var legacyNoFormat = have.length === 11 &&
      !hasKey('references') && !hasKey('recipe') &&
      !hasKey('transparent_background') &&
      !hasKey('output_format');
    var legacyAny = legacyNoRecipe || legacyNoReferences ||
                    legacyNoTransparency || legacyNoFormat;
    if (!(have.length === want.length || legacyAny)) return null;
    var i;
    // Reference ids: the content-addressed digests the server validates
    // against (config.py REF_MAX = 10) with one filled-in-per-era rule.
    if (legacyAny) c.recipe = 'base';
    if (legacyNoFormat) c.output_format = 'png';
    if (legacyNoFormat || legacyNoTransparency) c.transparent_background = false;
    if (legacyNoFormat || legacyNoTransparency || legacyNoReferences) c.references = [];
    for (i = 0; i < want.length; i++) {
      if (!hasKey(want[i])) return null;
    }
    if (Object.keys(c).length !== want.length) return null;  // no extra runtime fields
    // Reference ids: the content-addressed digests the server validates
    // against (config.py REF_MAX = 10; duplicates are LEGAL — the same
    // image may be cited twice). Any non-hex id or an over-cap list is a
    // broken echo.
    if (!Array.isArray(c.references) || c.references.length > MAX_REFERENCES) return null;
    for (i = 0; i < c.references.length; i++) {
      if (typeof c.references[i] !== 'string' ||
          !REFERENCE_ID_RE.test(c.references[i])) return null;
    }
    if (typeof c.prompt !== 'string' || c.prompt.length > MAX_CONFIG_STRING_CHARS) return null;
    // EXACTLY shared GenerationConfig (config.py: `not self.prompt.strip()`):
    // an echo whose prompt is empty or Python-whitespace-only is a broken
    // contract — the validated server never echoes one. JS trim differs from
    // Python strip, so use the corrected pyStrip predicate.
    if (pyStrip(c.prompt).length === 0) return null;
    if (typeof c.negative_prompt !== 'string' ||
        c.negative_prompt.length > MAX_CONFIG_STRING_CHARS) return null;
    // config.py validates dimensions as multiples of DIM_MULTIPLE (32) in
    // [64, 4096]; a validated echo must honour the same contract.
    if (!Number.isInteger(c.width) || c.width < 64 || c.width > 4096 ||
        c.width % 32 !== 0) return null;
    if (!Number.isInteger(c.height) || c.height < 64 || c.height > 4096 ||
        c.height % 32 !== 0) return null;
    if (!Number.isInteger(c.steps) || c.steps < 1 || c.steps > 1000) return null;
    if (typeof c.true_cfg_scale !== 'number' || !isFinite(c.true_cfg_scale) ||
        c.true_cfg_scale < 1) return null;
    // Public seeds are decimal STRING/null — a numeric seed would mean the
    // browser is being handed a rounded JS Number. Refuse, never render it.
    if (!(c.seed === null || (typeof c.seed === 'string' && /^\d{1,20}$/.test(c.seed)))) return null;
    if (c.seed !== null &&
        BigInt(c.seed) > BigInt('18446744073709551615')) return null;
    if (PLAN_ENUMS.quant.indexOf(c.quant) === -1) return null;
    if (PLAN_ENUMS.offload.indexOf(c.offload) === -1) return null;
    if (PLAN_ENUMS.attention_backend.indexOf(c.attention_backend) === -1) return null;
    if (typeof c.vae_tiled_decode !== 'boolean') return null;
    if (PLAN_ENUMS.output_format.indexOf(c.output_format) === -1) return null;
    if (typeof c.transparent_background !== 'boolean') return null;
    // The server refuses transparency without an alpha-capable format; an
    // echo claiming both is a broken contract, never a value to render.
    if (c.transparent_background &&
        ALPHA_FORMATS.indexOf(c.output_format) === -1) return null;
    // cfg_active must EQUAL the server's single source of truth
    // (config.py): a non-blank negative prompt under PYTHON strip rules
    // AND a scale strictly above 1.0.
    var cfgActive = pyStrip(c.negative_prompt).length > 0 && c.true_cfg_scale > 1;
    if (raw.cfg_active !== cfgActive) return null;
    // Recipe contract: enum value plus the exact Turbo gates (config.py).
    if (!recipeConsistent(c)) return null;
    // latent_tokens must equal the /16 compression of the echoed
    // dimensions (config.py C6); multiples-of-32 make this division exact.
    if (raw.latent_tokens !== (c.width / 16) * (c.height / 16)) return null;
    var m = raw.manifest;
    var manifest = parseManifestCore(m);
    if (!manifest) return null;
    if (m.ready !== false || m.verification !== 'not_performed' ||
        m.fit !== 'not_evaluated') return null;
    if (!raw.hardware || raw.hardware.status !== 'unknown') return null;
    if (!raw.calibration || raw.calibration.status !== 'unknown') return null;
    // Closed static blockers; submission availability must agree.
    if (!blockersExact(raw.generation_blockers, manifest.blocker)) return null;
    if (raw.generation_available !== (raw.generation_blockers.length === 0)) return null;
    return raw;
  }

  /* ---------------- Export ----------------
   * Raw attachment bytes only: the response is downloaded as a blob and
   * NEVER JSON.parsed/reserialized (the export carries the exact uint64
   * numeric seed). Runtime paths are excluded by the server. The draft is
   * validated again (locally here + authoritatively server-side) — never a
   * silent export of an invalid or stale snapshot.
   */

  function exportConfig() {
    setText('api-message', '');
    var built = buildEnvelope();
    if (!built.ok) {
      promptTouched = true;
      renderValidation(built.errors);
      setText('api-message', 'Export refused: the draft has local form errors. ' +
                             'Nothing was downloaded.');
      return Promise.resolve(false);
    }
    return apiRequest(API_EXPORT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Qwen-UI': '1' },
      body: JSON.stringify(built.envelope)
    }, 'raw').then(function (r) {
      if (r.network) {
        setApiAvailable(false);
        renderChip('Not connected',
          r.timedOut ? 'Not connected — the request timed out'
                     : 'Not connected — request failed');
        setText('api-message', 'Export failed (no reachable local server, or ' +
                               'it timed out). Nothing was downloaded; your ' +
                               'draft is unchanged.');
        return false;
      }
      // ONE unified bodyError branch: it comes FIRST, before both the
      // HTTP-error path and the raw-bytes download path, and it covers
      // EVERY decoding/readability outcome — 'too_large' (the fixed cap
      // was exceeded), 'unsupported' (no readable bounded stream) and any
      // unexpected body error. The OLD early too_large return that
      // bypassed this and left 'Connected' plus enabled buttons standing
      // is gone: an over-cap body is exactly as much a broken contract as
      // an unreadable one, so it invalidates availability too. Blob is
      // never reached, so Blob([undefined]) can never download the
      // literal text 'undefined' under a success message.
      if (r.bodyError) {
        // A typed schema rejection, NOT a disconnect: the trusted
        // availability is invalidated (explicit re-check required) with a
        // DISTINCT rejected chip — never a fabricated disconnect.
        setApiAvailable(false);
        renderChip('Status rejected',
          'Status rejected — server contract broken');
        setText('api-message', 'Export refused: the response body was unusable (' +
                               (r.bodyError === 'too_large'
                                 ? 'over the fixed response cap'
                                 : r.bodyError === 'unsupported'
                                   ? 'unreadable bounded body'
                                   : 'unexpected body decoding error') +
                               ') — a schema rejection, not a disconnect; ' +
                               'nothing was downloaded and your draft is ' +
                               'unchanged. Use "Check configuration" to ' +
                               're-establish a recognized status before ' +
                               'exporting again.');
        return false;
      }
      if (!r.res.ok) {
        // Reachable only with a FULLY READABLE body: an HTTP error whose
        // body failed to read bounded was already refused above as the
        // schema rejection it is.
        renderApiError(r.res, apiErrorPayloadFromBytes(r.bytes));
        setText('api-message', $('api-message').textContent +
          ' Nothing was downloaded; your draft is unchanged.');
        return false;
      }
      // RAW bytes only — never parsed, never reserialized: the exact
      // uint64 seed must survive byte-for-byte into the download, and a
      // download is produced ONLY from actual Uint8Array bytes. A success
      // render whose bytes are missing or wrong-typed is a client contract
      // bug: refuse honestly, never fabricate a Blob from 'undefined'.
      if (!(r.bytes instanceof Uint8Array)) {
        setApiAvailable(false);
        renderChip('Status rejected',
          'Status rejected — server contract broken');
        setText('api-message', 'Export refused: the raw success response ' +
                               'carried no readable bytes — nothing was ' +
                               'downloaded and your draft is unchanged.');
        return false;
      }
      var blob = new Blob([r.bytes]);
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'qwen-editable-config.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      if (!apiConnected) { checkStatus(); }  // server proved itself; refresh
      setText('api-message', 'Downloaded the editable configuration (raw ' +
                             'bytes, no reserialization). Server runtime paths ' +
                             'are excluded by design; import is a later ' +
                             'milestone.');
      return true;
    }).catch(function () {
      // Defensive: the network layer never rejects; anything reaching here
      // is a client-side bug, reported honestly, never as a fake success.
      setApiAvailable(false);
      renderChip('Not connected', 'Not connected — request failed');
      setText('api-message', 'Export failed unexpectedly. Nothing was ' +
                             'downloaded; your draft is unchanged.');
      return false;
    });
  }

  /* ---------------- Theme (the ONLY persisted preference) ----------------
   * Closed value set; documentElement.dataset.theme is the single live
   * switch. localStorage is touched ONLY under 'qwen-studio-theme', every
   * access is wrapped (private-mode/denied storage degrades to 'original'
   * silently — an honest fallback, never a crash), and no other storage
   * API is ever used.
   */

  function applyTheme(name) {
    var theme = THEMES.indexOf(name) !== -1 ? name : 'original';
    document.documentElement.dataset.theme = theme;
    var select = $('theme-select');
    if (select && THEMES.indexOf(select.value) === -1) select.value = theme;
    return theme;
  }

  function readStoredTheme() {
    try {
      var stored = window.localStorage.getItem(THEME_STORAGE_KEY);
      return THEMES.indexOf(stored) !== -1 ? stored : 'original';
    } catch (err) {
      return 'original';   // storage denied/unavailable: honest fallback
    }
  }

  function storeTheme(theme) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (err) {
      // Denied storage: the live theme still applies for this session;
      // nothing else is claimed and no other storage is attempted.
    }
  }

  function initTheme() {
    var select = $('theme-select');
    var theme = applyTheme(readStoredTheme());
    if (select) {
      select.value = theme;
      select.addEventListener('change', function () {
        var applied = applyTheme(select.value);
        if (THEMES.indexOf(select.value) === -1) select.value = applied;
        storeTheme(applied);
      });
    }
    return theme;
  }

  /* ---------------- Config import ----------------
   * The selected file is read as RAW BYTES (never JSON.parsed in the
   * browser: the export carries an exact uint64 numeric seed that
   * JSON.parse would round through Number — the server parses it in
   * Python exactly) and POSTed verbatim to /api/v1/config/import, which
   * returns a full web_plan.v1 payload. The response goes through the
   * SAME strict parsePlan as the plan flow before ANY field is applied.
   * Concurrency is honest: importSeq makes only the newest import land,
   * and draftRevision (bumped by every existing input/preset/random/
   * change handler via syncPlanStaleness) means an in-flight import can
   * NEVER overwrite a draft edit made after the file was picked. On any
   * failure the draft stays exactly as typed and nothing is generated.
   */

  // One bounded read path for every local file (config import and reference
  // uploads): the SAME size/staleness checks, so neither flow can grow a
  // second, subtly different reading convention.
  function readFileLimited(file) {
    return readFileBytes(file, MAX_IMPORT_FILE_BYTES);
  }

  function applyImportedPlan(plan) {
    var c = plan.config;
    $('prompt').value = c.prompt;
    // The draft text changed wholesale: any visible rewrite proposal is for
    // text that no longer exists.
    bumpPromptRevision();
    $('negative').value = c.negative_prompt;
    $('width').value = String(c.width);
    $('height').value = String(c.height);
    // The recipe is applied FIRST (and its lock re-derived LAST) so the
    // imported steps/cfg/quant land as the file's values instead of being
    // overwritten by a leftover Turbo lock or restored from a stale capture.
    recipeSaved = null;
    $('steps').value = String(c.steps);
    $('cfg').value = String(c.true_cfg_scale);
    // Seed stays a decimal STRING end to end — never a Number.
    $('seed').value = c.seed === null ? '' : c.seed;
    $('quant').value = c.quant;
    $('offload').value = c.offload;
    $('attention').value = c.attention_backend;
    $('vae-tiling').checked = c.vae_tiled_decode;
    $('output-format').value = c.output_format;
    // Import is an explicit server-validated application: an imported
    // transparency setting never triggers the jpg -> png auto-switch, so a
    // stale switch explanation must not survive it either.
    transparencySwitchNote = '';
    $('transparent').checked = c.transparent_background;
    setRecipeControl(c.recipe);
    applyRecipeMode(true);   // values came from the file: never capture them
    // Round-trip: the imported ids become the draft's reference list in the
    // file's order. The export carries ids only, so the display metadata
    // (dimensions/alpha/bytes) is UNKNOWN until the thumbnails load — shown
    // honestly as "stored on the server" rather than invented numbers.
    references = c.references.map(function (id) {
      return { id: id, width: null, height: null, has_alpha: null, bytes: null };
    });
    renderReferences();
    // An imported draft with references is an edit; one without is not.
    $('edit-mode').checked = c.references.length > 0;
    applyEditMode();
    setReferenceStatus(c.references.length
      ? 'Imported ' + c.references.length + ' reference image' +
        (c.references.length === 1 ? '' : 's') + ' in the file\'s order. ' +
        'Their thumbnails appear if the server still holds them.'
      : '');
  }

  function importConfigFile(file) {
    setText('api-message', '');
    if (!apiConnected) {
      setText('api-message', 'Import is unavailable without a recognized ' +
                             'server status. Use "Check configuration" first.');
      return Promise.resolve(false);
    }
    var seq = ++importSeq;
    var revisionAtPick = draftRevision;   // captured BEFORE any await
    return readFileLimited(file).then(function (bytes) {
      if (seq !== importSeq) return false;   // a newer import superseded this
      return apiRequest(API_IMPORT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Qwen-UI': '1' },
        body: bytes   // raw export bytes, verbatim — never reserialized
      }, 'json').then(function (r) {
        if (seq !== importSeq) return false;   // newer import wins; never land
        if (draftRevision !== revisionAtPick) {
          setText('api-message', 'Import discarded: the draft was edited ' +
                                 'while the file was being imported. Your ' +
                                 'current values were kept untouched; ' +
                                 'nothing was generated.');
          return false;
        }
        if (r.network) {
          setApiAvailable(false);
          renderChip('Not connected',
            r.timedOut ? 'Not connected — the request timed out'
                       : 'Not connected — request failed');
          setText('api-message', 'Import failed (no reachable local server, ' +
                                 'or it timed out). Your draft is unchanged; ' +
                                 'nothing was generated.');
          return false;
        }
        if (!r.res.ok) {
          renderApiError(r.res, r.json === undefined ? null : r.json);
          setText('api-message', $('api-message').textContent +
            ' Nothing was imported; your draft is unchanged.');
          return false;
        }
        if (r.bodyError) {
          // A rejected import payload is a broken server contract, exactly
          // like a rejected plan: availability is invalidated, nothing is
          // applied, the draft stays as typed.
          setApiAvailable(false);
          renderChip('Status rejected',
            'Status rejected — server contract broken');
          setText('api-message', 'The import response body was not usable (' +
                                 (r.bodyError === 'too_large'
                                   ? 'over the fixed response cap'
                                   : r.bodyError === 'unsupported'
                                     ? 'unreadable bounded body'
                                     : 'not decodable JSON') +
                                ') — a schema rejection, not a disconnect; ' +
                                'nothing was applied and your draft is ' +
                                'unchanged.');
          return false;
        }
        var plan = parsePlan(r.json);   // full strict validation FIRST
        if (!plan || plan.planning_available !== true) {
          setApiAvailable(false);
          renderChip('Status rejected',
            'Status rejected — server contract broken');
          setText('api-message', 'The import payload did not match the ' +
                                 'expected plan schema — nothing was ' +
                                 'applied and your draft is unchanged.');
          return false;
        }
        // Final synchronous guard immediately before touching the DOM:
        // nothing between this check and the assignment can interleave.
        if (draftRevision !== revisionAtPick) {
          setText('api-message', 'Import discarded: the draft was edited ' +
                                 'while the file was being imported. Your ' +
                                 'current values were kept untouched; ' +
                                 'nothing was generated.');
          return false;
        }
        applyImportedPlan(plan);
        planSeq++;                 // any in-flight PLAN answer is now stale
        planShownSignature = null;
        $('plan-panel').hidden = true;   // imported draft is not yet validated
        promptTouched = true;
        syncPresetButtons();
        syncPreviewShape();
        renderValidation(validateForm().errors);
        syncPlanStaleness();       // keeps staleness bookkeeping consistent
        setText('api-message', 'Imported configuration into the draft ' +
                               '(validated server-side; the file itself was ' +
                               'never parsed in the browser). The draft is ' +
                               'not yet planned — run "Check configuration" ' +
                               'or Plan. Nothing was generated.');
        return true;
      });
    }).catch(function (err) {
      if (seq !== importSeq) return false;
      var why = (err && err.message === 'too_large')
        ? 'the file exceeds the ' + MAX_IMPORT_FILE_BYTES +
          '-byte configuration bound'
        : 'the file could not be read';
      setText('api-message', 'Import refused: ' + why +
                             '. Your draft is unchanged; nothing was sent ' +
                             'or generated.');
      return false;
    });
  }

  /* ---------------- Reference images ----------------
   * Design §5 / §2.4. Uploads carry the RAW file bytes to
   * POST /api/v1/references with Content-Type: application/octet-stream —
   * never base64 inside JSON: the JSON routes cap bodies at 64 KiB while one
   * reference may be 32 MiB. The browser only ever holds ids plus display
   * metadata; the bytes and the store live on the server. The displayed
   * order IS the conditioning order, so the list is renumbered and the
   * envelope rebuilt in the same turn as any change.
   */

  var references = [];            // [{id, width, height, has_alpha, bytes}]
  var uploadingReference = false;

  function referenceIds() {
    return references.map(function (ref) { return ref.id; });
  }

  // The ids a job, plan or rewrite actually uses: the list only counts while
  // edit mode is ticked. Unticking keeps the list (ticking again restores it)
  // but sends nothing, so a hidden reference can never condition a job.
  function activeReferenceIds() {
    return $('edit-mode').checked ? referenceIds() : [];
  }

  // Show or hide the reference controls to match the edit-mode box. Toggling
  // changes which images the draft uses, so a visible rewrite proposal and the
  // current plan go stale exactly as they do when a reference is removed.
  function applyEditMode() {
    $('reference-field').hidden = !$('edit-mode').checked;
    if (references.length) bumpPromptRevision();
    refreshRewriteButton();
    syncPlanStaleness();
  }

  function setReferenceStatus(text) { setText('reference-status', text); }

  function formatReferenceBytes(n) {
    if (typeof n !== 'number' || !isFinite(n) || n < 0) return 'unknown size';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KiB';
    return (n / (1024 * 1024)).toFixed(1) + ' MiB';
  }

  // Add is offered only when a RECOGNIZED status has been seen (uploads are
  // writes), no upload is in flight and the hard cap is not reached. The
  // disabled state is a courtesy: the handler refuses on the same terms.
  function updateReferenceControls() {
    var full = references.length >= MAX_REFERENCES;
    $('add-reference').disabled = !apiConnected || uploadingReference || full;
    setText('reference-limit', String(MAX_REFERENCES));
    setText('reference-count', references.length === 0
      ? 'No references'
      : references.length + ' of ' + MAX_REFERENCES + ' references');
    $('reference-empty').hidden = references.length > 0;
    // The cap gates the finished-image shortcut too: a full list must not
    // offer another image, whether it comes from disk or from a job.
    updateUseAsReferenceControl();
  }

  function referenceButton(id, action) {
    return $('reference-list').querySelector(
      '[data-reference-id="' + id + '"] button[data-action="' + action + '"]');
  }

  // Focus restore after a re-render. The wanted button may be disabled (an
  // "up" that reached the top) or gone (a removed card), so fall back to the
  // nearest ENABLED control rather than dropping focus to <body>.
  function focusReferenceButton(id, action, index) {
    var want = referenceButton(id, action);
    if (!want || want.disabled) {
      // The acted-on card may be gone (remove): keep focus in the list.
      var cards = $('reference-list').children;
      var nearby = cards[index < cards.length ? index : cards.length - 1];
      want = null;
      if (nearby) {
        var buttons = nearby.querySelectorAll('button');
        for (var i = 0; i < buttons.length && !want; i++) {
          if (!buttons[i].disabled) want = buttons[i];
        }
      }
    }
    // No card left with an enabled action: the Add button is the next step.
    if (!want || want.disabled) want = $('add-reference');
    if (want) want.focus();
  }

  // One card. It is its OWN function for a reason: `var` is function-scoped,
  // so building every card inside renderReferences' loop left all the error
  // handlers sharing the LAST iteration's item/img/meta — a failed thumbnail
  // would mark the wrong card (and, with the loop variable, the wrong
  // ordinal). One call per card gives each handler its own bindings.
  function referenceCard(ref, index) {
    var item = document.createElement('li');
    item.className = 'reference-card';
    item.dataset.referenceId = ref.id;   // validated 64-hex: safe selector
    var ordinal = document.createElement('span');
    ordinal.className = 'reference-ordinal';
    ordinal.textContent = 'Image ' + (index + 1);
    var img = document.createElement('img');
    img.alt = 'Reference image ' + (index + 1);
    img.loading = 'lazy';
    // Server-owned thumbnail of stored bytes; the id is a checked hash, so
    // the URL is built from a validated value only (never a raw string).
    img.src = API_REFERENCES + '/' + ref.id + '/thumb';
    var meta = document.createElement('span');
    meta.className = 'reference-meta';
    var facts = [];
    if (Number.isInteger(ref.width) && Number.isInteger(ref.height)) {
      facts.push(ref.width + '×' + ref.height);
    }
    if (typeof ref.bytes === 'number') facts.push(formatReferenceBytes(ref.bytes));
    if (ref.has_alpha === true) facts.push('alpha');
    meta.textContent = facts.length ? facts.join(' · ') : 'stored on the server';
    // A pruned id (an imported plan naming bytes the server no longer holds)
    // would otherwise render a silent broken-image icon. The card stays in
    // place — the envelope must keep showing the order the user will submit
    // — but the missing thumbnail is stated.
    img.addEventListener('error', function () {
      // The list is rebuilt on every change, so a discarded card's later
      // error must not narrate for the CURRENT list.
      if (!item.isConnected) return;
      // The ordinal comes from the LIVE DOM position, not a captured value,
      // so the message always matches the numbering on screen.
      var pos = Array.prototype.indexOf.call(item.parentNode.children, item);
      if (pos === -1) return;
      // A failed thumbnail request does NOT prove the bytes are gone (the
      // server may be restarting, or the page opened from file://), so the
      // copy says what was observed. Only a submit-time 422 claims pruning.
      item.classList.add('reference-missing');
      meta.textContent = 'thumbnail unavailable — the preview could not be ' +
                         'loaded';
      setReferenceStatus('The preview for reference image ' + (pos + 1) +
        ' could not be loaded. The image may be missing from the server ' +
        'store — remove it or upload it again before submitting.');
    });
    var actions = document.createElement('span');
    actions.className = 'reference-actions';
    var buttons = [['up', 'Up', 'Move reference image ' + (index + 1) + ' up'],
                   ['down', 'Down', 'Move reference image ' + (index + 1) + ' down'],
                   ['remove', 'Remove', 'Remove reference image ' + (index + 1)]];
    for (var b = 0; b < buttons.length; b++) {
      var button = document.createElement('button');
      button.type = 'button';
      button.dataset.action = buttons[b][0];
      button.textContent = buttons[b][1];
      button.setAttribute('aria-label', buttons[b][2]);
      // Keyboard-only reordering: no drag, and the ends are honestly
      // unavailable instead of silently doing nothing.
      if (buttons[b][0] === 'up') button.disabled = index === 0;
      if (buttons[b][0] === 'down') button.disabled = index === references.length - 1;
      if (buttons[b][0] === 'remove') button.disabled = uploadingReference;
      actions.appendChild(button);
    }
    item.appendChild(ordinal);
    item.appendChild(img);
    item.appendChild(meta);
    item.appendChild(actions);
    return item;
  }

  // Numbering is derived from position on EVERY render — never stored — so a
  // remove or reorder renumbers immediately and the envelope cannot carry an
  // order the list does not show.
  function renderReferences() {
    var list = $('reference-list');
    while (list.firstChild) list.removeChild(list.firstChild);
    for (var i = 0; i < references.length; i++) {
      list.appendChild(referenceCard(references[i], i));
    }
    updateReferenceControls();
  }

  // Closed shape: exactly the six documented fields, id a lowercase sha256
  // hex digest, dimensions and byte count inside the server's own caps.
  function parseReferenceResponse(raw) {
    if (!raw || typeof raw !== 'object' || raw.schema !== REFERENCE_SCHEMA) return null;
    if (typeof raw.id !== 'string' || !REFERENCE_ID_RE.test(raw.id)) return null;
    if (!Number.isInteger(raw.width) || raw.width < 1 || raw.width > MAX_REFERENCE_DIM) return null;
    if (!Number.isInteger(raw.height) || raw.height < 1 || raw.height > MAX_REFERENCE_DIM) return null;
    if (typeof raw.has_alpha !== 'boolean') return null;
    if (!Number.isInteger(raw.bytes) || raw.bytes < 1 || raw.bytes > MAX_REFERENCE_BYTES) return null;
    if (Object.keys(raw).length !== 6) return null;
    return { id: raw.id, width: raw.width, height: raw.height,
             has_alpha: raw.has_alpha, bytes: raw.bytes };
  }

  function readFileBytes(file, cap) {
    return new Promise(function (resolve, reject) {
      if (!file || typeof file.size !== 'number' || file.size > cap) {
        reject(new Error('too_large'));
        return;
      }
      var reader = new FileReader();
      reader.onerror = function () { reject(new Error('unreadable')); };
      reader.onload = function () {
        var bytes = reader.result;
        if (!(bytes instanceof ArrayBuffer) || bytes.byteLength > cap) {
          reject(new Error('too_large'));
          return;
        }
        // Re-read the live file size: a file changed on disk between
        // selection and read must not slip past the bound.
        if (typeof file.size === 'number' && file.size > cap) {
          reject(new Error('too_large'));
          return;
        }
        resolve(new Uint8Array(bytes));
      };
      reader.readAsArrayBuffer(file);
    });
  }

  // One reference upload. Resolves true only when the server returned a
  // schema-valid reference document; every refusal is stated in the
  // aria-live status line with the server's own typed code, never silently
  // dropped. The server is authoritative: local checks only mirror its caps.
  function uploadReference(file) {
    return readFileBytes(file, MAX_REFERENCE_BYTES).then(function (bytes) {
      return apiRequest(API_REFERENCES, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream', 'X-Qwen-UI': '1' },
        body: bytes   // raw bytes, verbatim — never base64, never JSON
      }, 'json').then(function (r) {
        if (r.network) {
          setApiAvailable(false);
          renderChip('Not connected',
            r.timedOut ? 'Not connected — the request timed out'
                       : 'Not connected — request failed');
          setReferenceStatus('Upload failed: the local server is not ' +
                             'reachable (or it timed out). The image was not ' +
                             'added and your draft is otherwise unchanged.');
          return false;
        }
        if (r.bodyError) {
          // An unusable body (including one over the response cap) is a
          // broken server contract, exactly like a rejected plan.
          setApiAvailable(false);
          renderChip('Status rejected',
            'Status rejected — server contract broken');
          setReferenceStatus('Upload response was unusable (' +
            (r.bodyError === 'too_large' ? 'over the fixed response cap'
              : r.bodyError === 'unsupported' ? 'unreadable bounded body'
                                              : 'not decodable JSON') +
            ') — a schema rejection, not a disconnect. The image was not ' +
            'added.');
          return false;
        }
        if (!r.res.ok) {
          var payload = r.json && r.json.schema === ERROR_SCHEMA ? r.json : null;
          var code = payload && typeof payload.code === 'string' &&
                     /^[a-z0-9_]{1,64}$/.test(payload.code)
            ? payload.code : 'HTTP ' + r.res.status;
          var detail = payload && typeof payload.message === 'string'
            ? payload.message.slice(0, 200) : '';
          setReferenceStatus('Upload refused — ' + code +
            (detail ? ': ' + detail : '') + '. The image was not added.');
          return false;
        }
        var ref = parseReferenceResponse(r.json);
        if (!ref) {
          setApiAvailable(false);
          renderChip('Status rejected',
            'Status rejected — server contract broken');
          setReferenceStatus('The upload response did not match the ' +
                             'expected reference schema — nothing was added.');
          return false;
        }
        // The store is content-addressed: the same bytes are the same id.
        var existing = referenceIds().indexOf(ref.id);
        if (existing !== -1) {
          setReferenceStatus('That image is already reference image ' +
                             (existing + 1) + ' — it was not added twice.');
          return true;
        }
        references.push(ref);
        renderReferences();
        // The reference list is an input to the rewrite too: image N is what
        // the rewrite calls <imageN>, so adding one changes the question.
        bumpPromptRevision();
        refreshRewriteButton();
        syncPlanStaleness();   // the draft now conditions on more images
        setReferenceStatus('Added reference image ' + references.length +
          ' (' + ref.width + '×' + ref.height +
          (ref.has_alpha ? ', transparency kept' : '') +
          '). Refer to it as image ' + references.length + ' in your prompt.');
        return true;
      });
    }).catch(function (err) {
      var why = (err && err.message === 'too_large')
        ? 'the file exceeds the ' + formatReferenceBytes(MAX_REFERENCE_BYTES) +
          ' upload limit'
        : 'the file could not be read';
      setReferenceStatus('Upload refused: ' + why +
                         '. The image was not added.');
      return false;
    });
  }

  // Sequential uploads: one request in flight at a time (each is up to
  // 32 MiB), room for the remainder of the hard cap is computed BEFORE any
  // bytes are sent, and the cap is a hard stop with a stated reason.
  function addReferenceFiles(fileList) {
    var files = Array.prototype.slice.call(fileList || []);
    if (!files.length) return Promise.resolve(0);
    if (!apiConnected) {
      setReferenceStatus('Uploads need a recognized server status — use ' +
                         '"Check configuration" first. Nothing was uploaded.');
      return Promise.resolve(0);
    }
    if (uploadingReference) return Promise.resolve(0);
    var room = MAX_REFERENCES - references.length;
    if (room <= 0) {
      setReferenceStatus('At most ' + MAX_REFERENCES + ' reference images are ' +
                         'allowed. Remove an image before adding another — ' +
                         'nothing was uploaded.');
      return Promise.resolve(0);
    }
    var accepted = files.slice(0, room);
    var refused = files.length - accepted.length;
    uploadingReference = true;
    updateReferenceControls();
    var added = 0;
    var index = 0;
    function step() {
      if (index >= accepted.length) return Promise.resolve(added);
      var file = accepted[index];
      setReferenceStatus('Uploading ' + (file && file.name ? file.name : 'image') +
        ' — ' + (index + 1) + ' of ' + accepted.length + ' queued…');
      return uploadReference(file).then(function (ok) {
        if (ok) added++;
        index++;
        return step();
      });
    }
    // The batch end RE-RENDERS the cards, not just the toolbar: each render
    // during the batch saw uploadingReference === true, so the Remove
    // buttons were built disabled and would stay that way.
    return step().then(function (count) {
      uploadingReference = false;
      renderReferences();
      if (refused > 0) {
        setReferenceStatus('At most ' + MAX_REFERENCES + ' reference images are ' +
          'allowed — ' + refused + ' more file' + (refused === 1 ? ' was' : 's were') +
          ' not uploaded. Nothing was silently dropped.');
      } else if (count === 0 && references.length === 0) {
        setReferenceStatus('No reference images were added.');
      }
      return count;
    }, function () {
      uploadingReference = false;
      renderReferences();   // same reason: re-enable the per-card actions
      setReferenceStatus('The upload stopped unexpectedly; nothing further ' +
                         'was added. Check the server and try again.');
      return added;
    });
  }

  function removeReference(index) {
    if (!(index >= 0 && index < references.length)) return;
    references.splice(index, 1);
    renderReferences();
    bumpPromptRevision();   // every remaining <imageN> was renumbered
    refreshRewriteButton();
    syncPlanStaleness();
    setReferenceStatus('Removed reference image ' + (index + 1) + '. ' +
      (references.length
        ? 'The remaining images were renumbered — check that your prompt ' +
          'still names the right image numbers.'
        : 'No references remain; the prompt is used on its own.'));
  }

  function moveReference(index, delta) {
    var target = index + delta;
    if (index < 0 || index >= references.length ||
        target < 0 || target >= references.length) return;
    var moved = references[index];
    references[index] = references[target];
    references[target] = moved;
    renderReferences();
    // Same set, different order: the rewrite's <imageN> mapping changed, so a
    // visible proposal is stale even though the draft text did not move.
    bumpPromptRevision();
    syncPlanStaleness();
    setReferenceStatus('Reference image ' + (index + 1) + ' is now image ' +
      (target + 1) + '. The prompt refers to images by number, so check your ' +
      'wording.');
  }

  /* ---------------- Presets & preview ---------------- */

  function presetNameFor(width, height) {
    var names = Object.keys(PRESETS);
    for (var i = 0; i < names.length; i++) {
      var p = PRESETS[names[i]];
      if (p.width === width && p.height === height) return names[i];
    }
    return null;
  }

  // Recompute exact preset match from the CURRENT dimension values, so manual
  // edits never leave a stale pressed preset. Non-preset dimensions clear all.
  function syncPresetButtons() {
    var name = presetNameFor(Number($('width').value), Number($('height').value));
    var buttons = document.querySelectorAll('.preset-btn');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute('aria-pressed',
        String(buttons[i].dataset.preset === name));
    }
  }

  function applyPreset(name) {
    var preset = PRESETS[name];
    if (!preset) return;
    $('width').value = String(preset.width);
    $('height').value = String(preset.height);
    // Larger presets need tiled decode; 1K presets restore clean untiled decode.
    $('vae-tiling').checked = (preset.width / 16) * (preset.height / 16) > 4096;
    syncPresetButtons();
    syncPreviewShape();
    renderValidation(validateForm().errors);
    syncPlanStaleness();   // preset edits stale any visible result too
  }

  // Size of the finished image on screen ({width, height}), or null when the
  // frame shows no image. While an image is shown the frame keeps ITS shape:
  // job polling re-renders it, so following the draft would flip the frame to
  // the new preset and straight back. The draft's size shows as a label.
  var shownImageSize = null;

  function sizeLabel(width, height) {
    var name = presetNameFor(width, height);
    return (name ? PRESETS[name].ratio + ' \u00b7 ' : '') + width + '\u00d7' + height;
  }

  function syncPreviewShape() {
    var width = Number($('width').value), height = Number($('height').value);
    var note = $('preview-next-size');
    if (!shownImageSize) {
      setPreviewAspect(width, height);
      note.hidden = true;
      return;
    }
    setPreviewAspect(shownImageSize.width, shownImageSize.height);
    var differs = width !== shownImageSize.width || height !== shownImageSize.height;
    note.hidden = !differs || !isValidDimension(width) || !isValidDimension(height);
    note.textContent = note.hidden ? '' : 'Next image: ' + sizeLabel(width, height);
  }

  function setPreviewAspect(width, height) {
    var frame = $('preview-frame');
    if (isValidDimension(width) && isValidDimension(height) && height > 0) {
      frame.style.setProperty('--ar-w', String(width));
      frame.style.setProperty('--ar-h', String(height));
    }
  }

  /* ---------------- View routing ----------------
   * Hash-driven views only. An unknown hash (e.g. the skip link's #main)
   * must NOT silently reroute a History/Settings view back to Create — the
   * current view is kept. Headings are focused only on user-driven hash
   * changes (nav clicks, back/forward), never on initial load.
   */

  var activeView = null;

  function renderView(active) {
    Object.keys(VIEWS).forEach(function (name) {
      $(VIEWS[name]).hidden = name !== active;
    });
    ['create', 'history', 'settings', 'help'].forEach(function (name) {
      var link = $('nav-' + name);
      if (name === active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    activeView = active;
    // The rewriter choice lives on the server, so opening the page RE-reads
    // it: another browser (or another tab) may have changed it since, and a
    // form that shows a stale choice would then be a form that lies.
    if (active === 'settings') loadRewriteSettings();
  }

  function routeOnHashchange() {
    var name = (location.hash || '').replace(/^#/, '');
    if (name === '') {
      // Empty hash is Create. Without this, navigating History from an
      // initially-hash-less page and pressing browser Back would keep
      // History instead of restoring Create.
      if (activeView !== 'create') {
        renderView('create');
        $('create-heading').focus();   // user-driven back/forward only
      }
      return;
    }
    if (!Object.prototype.hasOwnProperty.call(VIEWS, name)) {
      return;   // unknown non-empty hash (#main, #whatever): keep current view
    }
    if (name === activeView) return;
    renderView(name);
    var heading = $(VIEW_HEADINGS[name]);
    if (heading) heading.focus();   // tabindex="-1" headings; user-driven only
  }

  function initialRoute() {
    var name = (location.hash || '').replace(/^#/, '');
    // Own-property check only: inherited names like #constructor or
    // #toString must NOT select a view (unknown hash falls back to Create).
    if (!Object.prototype.hasOwnProperty.call(VIEWS, name)) {
      name = 'create';
    }
    renderView(name);   // no focus on load
  }

  /* ---------------- Real engine jobs ---------------- */

  var JOB_STATES = ['queued', 'running', 'cancelling', 'cancelled', 'succeeded', 'failed'];
  var JOB_PHASES = ['load', 'encode', 'denoise', 'decode', 'save'];

  function jobTerminal(job) {
    return ['cancelled', 'succeeded', 'failed'].indexOf(job.state) !== -1;
  }

  function jobUrl(id) { return API_JOBS + '/' + id; }

  function parseJob(job) {
    if (!job || job.schema !== 'qwen_engine.web_job.v1' ||
        typeof job.job_id !== 'string' || !/^[0-9a-f]{32}$/.test(job.job_id) ||
        JOB_STATES.indexOf(job.state) === -1 ||
        !(job.phase === null || JOB_PHASES.indexOf(job.phase) !== -1) ||
        typeof job.cancel_requested !== 'boolean') return null;
    var c = job.config;
    if (!c || typeof c !== 'object') return null;
    // Current servers echo all 15 editable fields. FOUR older shapes are
    // accepted ONLY via explicit backward-compat branches and default the
    // missing keys to what those servers actually did: 14 keys without
    // recipe (always the reference model, so "base"), 13 keys without
    // references either (no conditioning images), 12 keys without
    // transparent_background either (never transparent), and 11 keys
    // without output_format either (saved PNG). Anything else (10 keys,
    // 16 keys, extra runtime fields) is a schema rejection, never a guess.
    var ckeys = Object.keys(c);
    var hasKey = function (key) {
      return Object.prototype.hasOwnProperty.call(c, key);
    };
    var legacyNoRecipe = ckeys.length === 14 && !hasKey('recipe');
    var legacyNoReferences = ckeys.length === 13 && !hasKey('references') &&
      !hasKey('recipe');
    var legacyNoTransparency = ckeys.length === 12 &&
      !hasKey('transparent_background') && hasKey('output_format') &&
      !hasKey('references') && !hasKey('recipe');
    var legacyNoFormat = ckeys.length === 11 && !hasKey('output_format') &&
      !hasKey('transparent_background') && !hasKey('references') &&
      !hasKey('recipe');
    var legacyAny = legacyNoRecipe || legacyNoReferences ||
                    legacyNoTransparency || legacyNoFormat;
    if (!(ckeys.length === 15 || legacyAny)) return null;
    if (legacyAny) c.recipe = 'base';
    if (legacyNoFormat) c.output_format = 'png';
    if (legacyNoFormat || legacyNoTransparency) c.transparent_background = false;
    if (legacyNoFormat || legacyNoTransparency || legacyNoReferences) c.references = [];
    // Same id contract as parsePlan: an ordered list of at most REF_MAX
    // content digests, in conditioning order.
    if (!Array.isArray(c.references) || c.references.length > MAX_REFERENCES) return null;
    for (var r = 0; r < c.references.length; r++) {
      if (typeof c.references[r] !== 'string' ||
          !REFERENCE_ID_RE.test(c.references[r])) return null;
    }
    if (typeof c.prompt !== 'string' ||
        !pyStrip(c.prompt) || c.prompt.length > MAX_CONFIG_STRING_CHARS ||
        typeof c.negative_prompt !== 'string' || c.negative_prompt.length > MAX_CONFIG_STRING_CHARS ||
        typeof c.seed !== 'string' || !/^\d{1,20}$/.test(c.seed) ||
        BigInt(c.seed) > SEED_MAX || !Number.isInteger(c.steps) || c.steps < 1 || c.steps > 1000 ||
        typeof c.true_cfg_scale !== 'number' || !isFinite(c.true_cfg_scale) || c.true_cfg_scale < 1 ||
        typeof c.vae_tiled_decode !== 'boolean' ||
        PLAN_ENUMS.quant.indexOf(c.quant) === -1 || PLAN_ENUMS.offload.indexOf(c.offload) === -1 ||
        PLAN_ENUMS.attention_backend.indexOf(c.attention_backend) === -1 ||
        PLAN_ENUMS.output_format.indexOf(c.output_format) === -1 ||
        typeof c.transparent_background !== 'boolean') return null;
    // A job config asking for transparency in a format that cannot store it
    // cannot have come from a validated server config.
    if (c.transparent_background &&
        ALPHA_FORMATS.indexOf(c.output_format) === -1) return null;
    // Same recipe contract as parsePlan: a job config the server would have
    // refused is a broken echo, not something to display or resubmit.
    if (!recipeConsistent(c)) return null;
    if (['width', 'height'].some(function (key) {
      return !Number.isInteger(c[key]) || c[key] < 64 || c[key] > 4096 || c[key] % 32 !== 0;
    })) return null;
    if (!(job.elapsed_ms === null || (typeof job.elapsed_ms === 'number' &&
          isFinite(job.elapsed_ms) && job.elapsed_ms >= 0))) return null;
    if (!job.phase_ms || typeof job.phase_ms !== 'object' || Array.isArray(job.phase_ms)) return null;
    if (Object.keys(job.phase_ms).some(function (phase) {
      var value = job.phase_ms[phase];
      return JOB_PHASES.indexOf(phase) === -1 || typeof value !== 'number' || !isFinite(value) || value < 0;
    })) return null;
    if ([null, 'execution_failed', 'worker_stopped', 'invalid_cancellation_acknowledgment',
         'out_of_gpu_memory'].indexOf(job.error_code) === -1) return null;
    if (job.image_url !== (job.state === 'succeeded' ? jobUrl(job.job_id) + '/image' : null)) return null;
    // Older servers have no step observations; keep their bar indeterminate.
    if (job.progress !== undefined && (!job.progress ||
        !Number.isInteger(job.progress.completed_steps) ||
        job.progress.total_steps !== c.steps || job.progress.completed_steps < 0 ||
        job.progress.completed_steps > c.steps)) return null;
    return job;
  }

  // Safe download name ONLY: a server-supplied filename must match a strict
  // conservative pattern (no separators, no control chars, bounded length)
  // or we fall back to a locally built name. Never a raw server string.
  var SAFE_FILENAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/;

  function jobDownloadName(job) {
    if (typeof job.image_filename === 'string' &&
        SAFE_FILENAME_RE.test(job.image_filename)) {
      return job.image_filename;
    }
    var fmt = OUTPUT_FORMATS.indexOf(job.config.output_format) !== -1
      ? job.config.output_format : 'png';
    return 'qi21-' + job.config.seed + '.' + fmt;
  }

  function updateGenerate() {
    $('generate-btn').disabled = !apiConnected || !generationConnected || submittingJob;
    var active = Object.keys(jobCache).some(function (id) { return !jobTerminal(jobCache[id]); });
    $('generate-btn').textContent = submittingJob ? 'Submitting…' : active ? 'Queue another image' : 'Generate';
    setText('generate-reason', generationConnected
      ? (submittingJob ? 'Submitting…' : 'Jobs run one at a time. Model and resource checks run for each job.')
      : GENERATION_UNAVAILABLE_REASON);
  }

  function renderSelectedJob(job) {
    selectedJob = job.job_id;
    setText('job-status', job.state + (job.phase ? ' · ' + job.phase :
      job.state === 'running' ? ' · verifying model assets' : '') +
      (job.error_code ? ' · ' + job.error_code + ' (see server terminal)' : ''));
    var failureMessages = {
      // Plain-language, actionable and prompt-free; unknown codes keep the
      // generic fixed sentence below (closed list, no server text).
      out_of_gpu_memory: 'The GPU ran out of memory for this job. Try Q8, ' +
        'a smaller size, fewer reference images, or Turbo.'
    };
    var descriptions = {
      queued: 'Queued — waiting for the current generation to finish.',
      running: 'Generating — ' + ({load: 'loading model weights', encode: 'encoding your prompt',
        denoise: 'drawing the image', decode: 'decoding the image', save: 'saving the image'}[job.phase] || 'verifying model files') + '.',
      cancelling: 'Cancellation requested — waiting for the engine to release resources.',
      cancelled: 'Generation cancelled.',
      succeeded: 'Image ready. Use View completed image below.',
      failed: failureMessages[job.error_code] ||
        ('Generation failed — ' + job.error_code + '. See the server terminal for details.')
    };
    var feedback = descriptions[job.state];
    if (job.elapsed_ms !== null) feedback += ' Elapsed: ' + (job.elapsed_ms / 1000).toFixed(0) + ' s.';
    setText('job-feedback', feedback);
    $('job-feedback').dataset.state = job.state;
    $('job-progress-panel').hidden = false;
    var progress = $('job-progress');
    progress.max = job.config.steps;
    if (job.state === 'succeeded') {
      progress.value = job.config.steps;
      setText('job-progress-label', 'Complete — image saved');
    } else if (job.phase === 'denoise' && job.progress && !jobTerminal(job)) {
      progress.value = job.progress.completed_steps;
      setText('job-progress-label', 'Drawing: step ' + job.progress.completed_steps + ' of ' +
        job.progress.total_steps + ' (' + Math.floor(100 * job.progress.completed_steps / job.progress.total_steps) + '% of drawing)');
    } else {
      progress.removeAttribute('value');
      progress.hidden = jobTerminal(job);
      setText('job-progress-label', descriptions[job.state]);
    }
    if (!jobTerminal(job) || job.state === 'succeeded') progress.hidden = false;
    $('view-result').hidden = job.state !== 'succeeded';
    setText('preview-title', job.state === 'running' ? 'Generating your image…' :
      job.state === 'queued' ? 'Generation queued' : job.state === 'failed' ? 'Generation failed' : 'No image yet');
    setText('preview-message', feedback);
    updateGenerate();
    setText('job-timings', Object.keys(job.phase_ms).map(function (phase) {
      return phase + ': ' + (job.phase_ms[phase] / 1000).toFixed(2) + ' s';
    }).join(' · '));
    setText('stat-elapsed', job.elapsed_ms === null ? '—' : (job.elapsed_ms / 1000).toFixed(1) + ' s');
    setText('stat-steps', job.state === 'succeeded' ? String(job.config.steps) : '—');
    setText('stat-result', job.state);
    $('cancel-job').disabled = jobTerminal(job) || job.cancel_requested;
    var complete = job.state === 'succeeded';
    ['action-download', 'action-open', 'action-again'].forEach(function (id) {
      $(id).disabled = !complete;
      $(id).title = complete ? '' : 'No completed result selected';
    });
    // Its own helper: the gate is the cap and the connection too, not just
    // "a result exists", and the reason belongs in the title.
    updateUseAsReferenceControl();
    var image = $('preview-image');
    if (complete) {
      if (image.getAttribute('src') !== job.image_url) image.src = job.image_url;
      image.alt = job.config.prompt;
      shownImageSize = { width: job.config.width, height: job.config.height };
    } else {
      image.removeAttribute('src');
      shownImageSize = null;
    }
    syncPreviewShape();
    image.hidden = !complete;
    $('preview-empty').hidden = complete;
  }

  function renderHistory(ids) {
    var grid = $('history-grid');
    grid.replaceChildren();
    $('history-empty').hidden = ids.length > 0;
    var note = $('history-missing-note');
    note.hidden = true;
    note.textContent = '';
    ids.slice().reverse().forEach(function (id) {
      var job = jobCache[id];
      if (!job) return;
      var card = document.createElement('article');
      card.className = 'history-card';
      if (job.state === 'succeeded') {
        var image = document.createElement('img');
        image.src = job.image_url;
        image.alt = job.config.prompt;
        image.loading = 'lazy';
        image.addEventListener('error', function () {
          // The grid is rebuilt on every refresh, so a discarded card's later
          // error must not narrate for the CURRENT grid.
          if (!card.isConnected) return;
          // A failed request does not prove the file was tampered with (the
          // server may be restarting), so state only what was observed.
          card.classList.add('history-missing');
          setText('history-missing-note', 'One or more saved images could not ' +
            'be loaded. Their history entries are kept; open a job to see its ' +
            'status, or use its "View job" details.');
          $('history-missing-note').hidden = false;
        });
        card.appendChild(image);
      }
      var label = document.createElement('p');
      label.textContent = job.state +
        (job.config.recipe === 'turbo' ? ' · Turbo' : '') +
        ' · seed ' + job.config.seed + ' · ' +
        job.config.prompt.slice(0, 160);
      card.appendChild(label);
      var view = document.createElement('button');
      view.type = 'button';
      view.className = 'btn-secondary';
      view.textContent = 'View job';
      view.addEventListener('click', function () { renderSelectedJob(jobCache[id]); location.hash = '#create'; });
      card.appendChild(view);
      // Only entries that HAVE an image get the shortcut: a queued, cancelled
      // or failed job has nothing to feed back in.
      if (job.state === 'succeeded') {
        var reuse = document.createElement('button');
        reuse.type = 'button';
        reuse.className = 'btn-secondary';
        reuse.textContent = 'Use as reference';
        var blocked = useAsReferenceBlock(job);
        reuse.disabled = blocked !== '';
        reuse.title = blocked || 'Add this image as reference image ' +
          (references.length + 1) + ' of ' + MAX_REFERENCES;
        reuse.addEventListener('click', function () {
          if (reuse.disabled) return;
          // The list and its status line live on the Create view, so switch
          // there first: the added card must be visible, not narrated to a
          // hidden view. The selected job is deliberately left alone.
          location.hash = '#create';
          useJobImageAsReference(id);
        });
        card.appendChild(reuse);
      }
      if (jobTerminal(job)) {
        var forget = document.createElement('button');
        forget.type = 'button';
        forget.className = 'btn-secondary';
        forget.textContent = 'Forget entry';
        forget.title = 'Remove this saved history entry (job.json); the image file stays on disk';
        forget.addEventListener('click', function () { jobAction(id, 'forget'); });
        card.appendChild(forget);
      }
      grid.appendChild(card);
    });
  }

  function jobFailure(message) {
    setText('job-status', message);
    setText('job-feedback', message);
    $('job-feedback').dataset.state = 'error';
    $('job-progress').removeAttribute('value');
    setText('job-progress-label', 'Progress unavailable. ' + message);
    // No automatic POST retry: a lost submit response may still own a real job.
  }

  function refreshJobs() {
    if (refreshingJobs || !generationConnected) return Promise.resolve(false);
    if (jobTimer) clearTimeout(jobTimer);
    refreshingJobs = true;
    var seq = jobSeq;
    return apiRequest(API_JOBS, {headers: {'Accept': 'application/json'}}, 'json').then(function (r) {
      var list = r.json;
      if (!r.res || !r.res.ok || r.bodyError || !list || list.schema !== 'qwen_engine.web_jobs.v1' ||
          !Array.isArray(list.jobs) || list.jobs.length > 32 || new Set(list.jobs).size !== list.jobs.length ||
          list.jobs.some(function (id) { return typeof id !== 'string' || !/^[0-9a-f]{32}$/.test(id); })) {
        throw new Error('jobs unavailable');
      }
      return Promise.all(list.jobs.map(function (id) {
        if (jobCache[id] && jobTerminal(jobCache[id])) return jobCache[id];
        return apiRequest(jobUrl(id), {headers: {'Accept': 'application/json'}}, 'json').then(function (answer) {
          var job = answer.res && answer.res.ok && !answer.bodyError && parseJob(answer.json);
          if (!job || job.job_id !== id) throw new Error('job rejected');
          jobCache[id] = job;
          return job;
        });
      })).then(function () {
        if (seq !== jobSeq) return true;
        Object.keys(jobCache).forEach(function (id) { if (list.jobs.indexOf(id) === -1) delete jobCache[id]; });
        renderHistory(list.jobs);
        if (!selectedJob || !jobCache[selectedJob]) selectedJob = list.jobs[list.jobs.length - 1] || null;
        if (selectedJob) renderSelectedJob(jobCache[selectedJob]);
        else {
          $('preview-image').hidden = true;
          $('preview-image').removeAttribute('src');
          $('preview-empty').hidden = false;
          shownImageSize = null;
          syncPreviewShape();
          ['cancel-job', 'action-download', 'action-open', 'action-again'].forEach(function (id) { $(id).disabled = true; });
          updateUseAsReferenceControl();   // same block reason: no result
          setText('job-status', 'No job selected.');
          setText('job-feedback', 'Ready to submit a generation.');
          $('job-feedback').removeAttribute('data-state');
          $('view-result').hidden = true;
          $('job-progress-panel').hidden = true;
          setText('job-timings', '');
          ['stat-result', 'stat-steps', 'stat-elapsed'].forEach(function (id) { setText(id, '—'); });
        }
        return true;
      });
    }).catch(function () {
      setApiAvailable(false);
      jobFailure('Job refresh failed; work may still be running. Use Check configuration to reconnect.');
      return false;
    }).then(function (ok) {
      refreshingJobs = false;
      // Poll serially after completion of this request; never overlap requests.
      if (generationConnected) jobTimer = setTimeout(refreshJobs, 2000);
      return ok;
    });
  }

  // The server's message is untrusted text: extract at most the ordinal
  // ("reference 3 is not available…") and fall back to a neutral phrase.
  function referenceNumberFromError(message) {
    if (typeof message === 'string') {
      var m = /^reference (\d{1,2}) is not available/.exec(message);
      if (m && Number(m[1]) >= 1 && Number(m[1]) <= MAX_REFERENCES) return m[1];
    }
    return 'listed in the plan';
  }

  function submitGeneration(config) {
    if (submittingJob) return;
    if (!config) {
      var validation = validateForm();
      promptTouched = true;
      renderValidation(validation.errors);
      if (Object.keys(validation.errors).length) return;
    }
    if (!generationConnected || !apiConnected) {
      jobFailure('Nothing was submitted: connect to a generation-enabled server first.');
      return;
    }
    var envelope = config ? {schema: PLAN_REQUEST_SCHEMA, config: config} : buildEnvelope().envelope;
    submittingJob = true;
    setText('job-feedback', 'Submitting your generation…');
    $('job-feedback').dataset.state = 'submitting';
    $('job-progress-panel').hidden = false;
    $('job-progress').hidden = false;
    $('job-progress').removeAttribute('value');
    setText('job-progress-label', 'Submitting generation…');
    updateGenerate();
    apiRequest(API_JOBS, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Qwen-UI': '1'},
      body: JSON.stringify(envelope)}, 'json').then(function (r) {
      var job = r.res && r.res.ok && !r.bodyError && parseJob(r.json);
      if (!job) {
        if (r.res && !r.res.ok && r.json && r.json.schema === ERROR_SCHEMA) {
          renderApiError(r.res, r.json);
          // Recipe-combination refusals land on the controls they describe.
          applyServerFieldErrors(r.json);
          // The server names the ORDINAL, not the id: mirror it in the live
          // status line so the user knows which thumbnail to re-upload.
          if (r.json.code === 'reference_unavailable') {
            setReferenceStatus('The server no longer holds reference image ' +
              'number ' + referenceNumberFromError(r.json.message) +
              ' — it was probably pruned. Remove it or upload it again. ' +
              'Nothing was submitted.');
          }
          jobFailure('Submission refused. See the configuration message.');
        } else {
          jobFailure('Submission response was lost or rejected. Check History before retrying; the job may have been accepted.');
        }
        return;
      }
      jobCache[job.job_id] = job;
      jobSeq++;
      renderSelectedJob(job);
    }).catch(function () {
      jobFailure('Submission could not be confirmed. Check History before retrying.');
    }).then(function () {
      submittingJob = false;
      updateGenerate();
      refreshJobs();
    });
  }

  function jobAction(id, action) {
    apiRequest(jobUrl(id) + '/' + action,
      {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Qwen-UI': '1'}, body: '{}'},
      'json').then(function (r) {
      if (!r.res || !r.res.ok || r.bodyError) {
        jobFailure('Job action could not be confirmed. Refresh before retrying.');
      } else if (action === 'cancel') {
        var job = parseJob(r.json);
        if (job) { jobCache[id] = job; renderSelectedJob(job); }
      } else if (r.json && r.json.schema === 'qwen_engine.web_jobs.v1' && r.json.forgotten === id) {
        delete jobCache[id];
        jobSeq++;
      }
      refreshJobs();
    });
  }

  /* ---------------- Use as reference ----------------
   * A finished image goes back in as a reference image. The bytes come from
   * the job's own image endpoint and leave through the EXISTING reference
   * upload, so no route is added and no extra trust is granted: the server
   * re-identifies the bytes by magic and re-hashes what it just served.
   */

  // MIME type for the carrier object only. The store sniffs the magic and
  // ignores headers; this mirrors what the endpoint served for this job's
  // output_format (generation.py image_content_type), so name and type agree.
  var IMAGE_CONTENT_TYPES = { jpg: 'image/jpeg', png: 'image/png',
                              webp: 'image/webp' };

  // Why the action is unavailable for THIS job, or '' when it is available.
  // The result-panel button and every History card share it, so an image that
  // cannot be added never looks addable, and the same sentence is the status
  // line if a stale enabled card is clicked anyway.
  function useAsReferenceBlock(job) {
    if (!job || job.state !== 'succeeded' || !job.image_url) {
      return 'That job has no finished image to add as a reference';
    }
    if (!apiConnected) {
      return 'Adding a reference needs a recognized server status — use ' +
             '"Check configuration" first';
    }
    if (uploadingReference) {
      return 'A reference upload is already in flight — wait for it to finish';
    }
    if (references.length >= MAX_REFERENCES) {
      return 'At most ' + MAX_REFERENCES + ' reference images are allowed — ' +
             'remove an image before adding another';
    }
    return '';
  }

  function updateUseAsReferenceControl() {
    var reason = useAsReferenceBlock(jobCache[selectedJob]);
    // "Last image" is the one in the preview: the newest result unless an
    // older one was opened from History.
    $('use-last-image').disabled = reason !== '';
    $('use-last-image').title = reason || 'Add the image shown in the preview ' +
      'as reference image ' + (references.length + 1) + ' of ' + MAX_REFERENCES;
  }

  // Bytes plus a display name for the upload path, which reads file.size and
  // file.name: File where the constructor exists (the status line then names
  // the real filename), Blob otherwise. Neither wrapper reaches the server.
  function carrierFileFromBytes(bytes, name, type) {
    var parts = [bytes];
    if (typeof File === 'function') {
      try {
        return new File(parts, name, { type: type });
      } catch (err) {
        /* this build rejects the File constructor: the Blob below carries it */
      }
    }
    return new Blob(parts, { type: type });
  }

  function useJobImageAsReference(id) {
    var job = jobCache[id];
    var block = useAsReferenceBlock(job);
    if (block) {
      setReferenceStatus(block + '. Nothing was added.');
      return Promise.resolve(false);
    }
    // Reveal the reference controls BEFORE the first request: progress,
    // success and every refusal are narrated on the reference status line,
    // and a hidden fieldset would leave the user with silence. Ticking the
    // box is what this action is for — no other setting is written here.
    $('edit-mode').checked = true;
    applyEditMode();
    setReferenceStatus('Loading the saved image so it can be added as a ' +
                       'reference…');
    return apiRequest(job.image_url, {
      headers: {'Accept': 'image/png, image/jpeg, image/webp'},
      maxBytes: MAX_JOB_IMAGE_BYTES
    }, 'raw').then(function (r) {
      if (r.network) {
        setApiAvailable(false);
        renderChip('Not connected', 'Not connected — the image request ' +
          (r.timedOut ? 'timed out' : 'failed'));
        setReferenceStatus('Could not load the saved image: the local server ' +
          'is not reachable (or it timed out). Nothing was added.');
        return false;
      }
      if (r.bodyError) {
        // A usable response whose BODY failed is a broken contract, exactly
        // like a rejected upload: a typed rejection, never a fake disconnect.
        setApiAvailable(false);
        renderChip('Status rejected',
          'Status rejected — server contract broken');
        setReferenceStatus('The saved image could not be read (' +
          (r.bodyError === 'too_large'
            ? 'over the ' + formatReferenceBytes(MAX_JOB_IMAGE_BYTES) + ' limit'
            : 'unreadable bounded body') +
          ') — a broken response, not a disconnect. Nothing was added.');
        return false;
      }
      if (!r.res.ok) {
        // The endpoint answers with the typed error document; only a
        // schema-tagged, code-shaped value is ever quoted back.
        var payload = apiErrorPayloadFromBytes(r.bytes);
        var code = payload && payload.schema === ERROR_SCHEMA &&
                   typeof payload.code === 'string' &&
                   /^[a-z0-9_]{1,64}$/.test(payload.code)
          ? payload.code : 'HTTP ' + r.res.status;
        setReferenceStatus('The saved image could not be loaded — ' + code +
          '. It may be missing from the server or have changed; nothing was ' +
          'added.');
        return false;
      }
      if (!(r.bytes instanceof Uint8Array) || r.bytes.length === 0) {
        setReferenceStatus('The saved image arrived with no bytes to add. ' +
                           'Nothing was added.');
        return false;
      }
      // The existing upload path keeps ownership of the caps, the sequential
      // upload, the dedupe by content id and the "Added reference image N"
      // message; only the origin of the bytes differs here.
      return addReferenceFiles([carrierFileFromBytes(
        r.bytes, jobDownloadName(job),
        IMAGE_CONTENT_TYPES[job.config.output_format] || 'image/png'
      )]).then(function (count) { return count > 0; });
    }, function () {
      setReferenceStatus('Loading the saved image stopped unexpectedly; ' +
                         'nothing was added.');
      return false;
    });
  }

  /* ---------------- Wiring ---------------- */

  function init() {
    Object.keys(PRESETS).forEach(function (name) {
      var preset = PRESETS[name];
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'preset-btn';
      button.dataset.preset = name;
      button.setAttribute('aria-pressed', String(name === 'square'));
      button.textContent = preset.ratio + ' · ' + preset.width + '×' + preset.height;
      $('presets-' + preset.tier.toLowerCase()).appendChild(button);
    });
    $('create-form').addEventListener('submit', function (ev) {
      ev.preventDefault();
      promptTouched = true;
      renderValidation(validateForm().errors);
      submitGeneration();
    });

    $('create-form').addEventListener('keydown', function (ev) {
      if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
        ev.preventDefault();
        promptTouched = true;
        renderValidation(validateForm().errors);
        submitGeneration();
      }
    });

    // 'Check configuration' doubles as the reconnect/refresh action: it
    // re-reads the authoritative status first, then validates the draft.
    $('btn-check-config').addEventListener('click', function () {
      checkStatus().then(function (status) {
        return status ? runPlan(false) : false;
      });
    });
    $('btn-plan').addEventListener('click', function () { runPlan(true); });
    $('btn-export').addEventListener('click', function () { exportConfig(); });
    $('btn-import').addEventListener('click', function () {
      if ($('btn-import').disabled) return;
      $('import-file').click();   // opens the native file picker
    });
    $('import-file').addEventListener('change', function (ev) {
      var input = ev.currentTarget;
      var file = input.files && input.files[0];
      input.value = '';   // reset FIRST: the same file can be re-picked
      if (file) importConfigFile(file);
    });

    // Reference images: the visible Add button opens the hidden multi-file
    // input, and the input resets BEFORE the uploads start so picking the
    // same file again still fires 'change'. Uploads are sequential and
    // capacity-checked inside addReferenceFiles.
    $('add-reference').addEventListener('click', function () {
      if ($('add-reference').disabled) return;
      $('reference-file').click();
    });
    $('reference-file').addEventListener('change', function (ev) {
      var input = ev.currentTarget;
      var files = input.files ? Array.prototype.slice.call(input.files) : [];
      input.value = '';   // reset FIRST: re-picking the same file must work
      addReferenceFiles(files);
    });
    // One delegated handler for the per-card buttons: the list is rebuilt on
    // every render, so per-button listeners would need re-attaching.
    $('reference-list').addEventListener('click', function (ev) {
      var button = ev.target && ev.target.closest
        ? ev.target.closest('button[data-action]') : null;
      if (!button || button.disabled) return;
      var card = button.closest('.reference-card');
      if (!card) return;
      var index = Array.prototype.indexOf.call($('reference-list').children, card);
      var action = button.dataset.action;
      if (action === 'remove') {
        removeReference(index);
        focusReferenceButton(card.dataset.referenceId, 'remove', index);
      } else if (action === 'up' || action === 'down') {
        var delta = action === 'up' ? -1 : 1;
        moveReference(index, delta);
        // Focus follows the MOVED card to its new position, so repeated
        // presses keep moving the same image instead of the neighbour.
        focusReferenceButton(card.dataset.referenceId, action, index + delta);
      }
    });
    // Drag is never the ONLY way: it is not offered at all, so keyboard and
    // pointer users get the same buttons.
    renderReferences();
    $('view-result').addEventListener('click', function () {
      $('preview-frame').scrollIntoView({behavior: 'smooth', block: 'center'});
    });
    $('cancel-job').addEventListener('click', function () {
      if (selectedJob) jobAction(selectedJob, 'cancel');
    });
    $('action-download').addEventListener('click', function () {
      var job = jobCache[selectedJob];
      if (!job || job.state !== 'succeeded') return;
      var link = document.createElement('a');
      link.href = job.image_url;
      link.download = jobDownloadName(job);
      link.click();
    });
    $('action-open').addEventListener('click', function () {
      var job = jobCache[selectedJob];
      if (job && job.state === 'succeeded') window.open(job.image_url, '_blank', 'noopener');
    });
    $('action-again').addEventListener('click', function () {
      var job = jobCache[selectedJob];
      if (job && job.state === 'succeeded') submitGeneration(job.config);
    });
    // Same refusal discipline as its siblings: the disabled state is a
    // courtesy, the function re-checks the block and states the reason.
    $('use-last-image').addEventListener('click', function () {
      useJobImageAsReference(selectedJob);
    });
    $('preview-image').addEventListener('error', function () {
      jobFailure('The saved image could not be loaded; it may have moved or changed.');
    });

    var presetButtons = document.querySelectorAll('.preset-btn');
    for (var i = 0; i < presetButtons.length; i++) {
      presetButtons[i].addEventListener('click', function (ev) {
        applyPreset(ev.currentTarget.dataset.preset);
      });
    }

    $('random-seed').addEventListener('click', function () {
      $('seed').value = randomSeedString();
      renderValidation(validateForm().errors);
      syncPlanStaleness();   // a random seed stale any visible result
    });

    // Prompt rewriting: a suggestion flow, never an automatic edit. The
    // proposal block is focusable so keyboard and screen-reader users land on
    // it when it appears; Escape abandons it and returns to the prompt.
    $('rewrite-btn').addEventListener('click', requestRewrite);
    $('rewrite-apply').addEventListener('click', applyProposal);
    $('rewrite-edit').addEventListener('click', editProposal);
    $('rewrite-keep').addEventListener('click', keepOriginal);
    $('rewrite-apply-ratio').addEventListener('click', function (ev) {
      var name = ev.currentTarget.dataset.preset;
      if (name) applyPreset(name);
    });
    $('rewrite-proposal').addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') {
        ev.preventDefault();
        keepOriginal();
      }
    });

    // The Settings page. Every control here writes the SERVER's choice, so a
    // change is only real once the server has answered — but a radio the user
    // clicked must survive the redraw that follows, so the form keeps its own
    // pending selection until a load or a save replaces it.
    ['rewrite-t2i', 'rewrite-edit'].forEach(function (name) {
      Array.prototype.forEach.call(
        document.querySelectorAll('input[name="' + name + '"]'),
        function (radio) {
          radio.addEventListener('change', function () {
            noteRewriteChoice(radio.name.slice('rewrite-'.length), radio.value);
            setText('rewrite-save-status', '');
            setText('rewrite-settings-error', '');
            renderRewriteSettings();
          });
        });
    });
    $('rewrite-off').addEventListener('change', function () {
      noteRewriteEnabled(!$('rewrite-off').checked);
      setText('rewrite-save-status', '');
      setText('rewrite-settings-error', '');
      renderRewriteSettings();
    });
    ['rewrite-endpoint', 'rewrite-model', 'rewrite-timeout'].forEach(function (id) {
      $(id).addEventListener('input', function () {
        noteRewriteFields(rewriteEndpointFields());
        setText('rewrite-save-status', '');
        setText('rewrite-test-status', '');
        renderRewriteSettings();
      });
    });
    $('rewrite-save').addEventListener('click', saveRewriteSettings);
    $('rewrite-test').addEventListener('click', testRewriteConnection);

    ['steps', 'seed', 'cfg', 'width', 'height', 'prompt', 'negative'].forEach(function (id) {
      $(id).addEventListener('input', function () {
        if (id === 'width' || id === 'height') {
          syncPreviewShape();
          syncPresetButtons();
        }
        if (id === 'prompt') { promptTouched = true; bumpPromptRevision(); }
        renderValidation(validateForm().errors);
        syncPlanStaleness();
      });
    });

    // Change-only fields (they never carry inline errors) must ALSO make a
    // visible result clearly stale, not leave it looking validated.
    ['quant', 'offload', 'attention', 'vae-tiling', 'output-format'].forEach(function (id) {
      $(id).addEventListener('change', function () {
        // An explicit format choice replaces any auto-switch explanation.
        if (id === 'output-format') transparencySwitchNote = '';
        renderValidation(validateForm().errors);
        syncPlanStaleness();
      });
    });

    // Transparency owns the jpg -> png switch, so it is wired separately
    // from the change-only group above.
    $('transparent').addEventListener('change', onTransparencyChange);
    $('edit-mode').addEventListener('change', applyEditMode);

    // Recipe: every radio in the group shares one handler, so the lock state
    // is derived from whichever control the user (or an import) selected.
    Array.prototype.forEach.call(
      document.querySelectorAll('input[name="recipe"]'),
      function (radio) {
        radio.addEventListener('change', function () {
          applyRecipeMode();
          renderValidation(validateForm().errors);
          syncPlanStaleness();
        });
      });

    window.addEventListener('hashchange', routeOnHashchange);
    initTheme();   // persisted theme applied before first paint of content
    initialRoute();
    syncPresetButtons();
    applyRecipeMode();   // lock state matches the checked recipe on load
    renderValidation(validateForm().errors);
    // The Settings form starts unread: nothing is selectable until the server
    // has said which rewriters it can actually run.
    renderRewriteSettings();

    var generate = $('generate-btn');
    generate.disabled = true;   // enabled only by recognized server status
    setText('generate-reason', GENERATION_UNAVAILABLE_REASON);
    setText('settings-generation',
      'Planning and authoritative config validation come from the local ' +
      'qwen-web server; generation requires --enable-generation and configured runtime paths.');

    checkStatus();   // same-origin only; failures render honestly
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Small explicit surface for manual inspection and the real-Chrome
  // checks. snapshot() is the UI wire format (not GenerationConfig JSON);
  // buildEnvelope() is what the server receives. The parsers are exposed so
  // the browser checks can assert schema rejection without a live server;
  // setFetchTimeoutForChecks is a TEST-ONLY knob (bounded to sane values)
  // that shortens the whole-operation timeout so a stalled response BODY
  // can be observed quickly — production code never calls it.
  window.QwenStudio = {
    version: 'w7-rewriter-settings',
    formSnapshot: formSnapshot,
    buildEnvelope: buildEnvelope,
    validateForm: validateForm,
    parseSeed: parseSeed,
    parseStatus: parseStatus,
    parsePlan: parsePlan,
    parseJob: parseJob,
    randomSeedString: randomSeedString,
    checkStatus: checkStatus,
    importConfigFile: importConfigFile,
    applyTheme: applyTheme,
    readStoredTheme: readStoredTheme,
    jobDownloadName: jobDownloadName,
    // Reference surface: the checks (and a curious operator) can assert the
    // upload path, the ordering contract and the hard cap without a server.
    referenceIds: referenceIds,
    addReferenceFiles: addReferenceFiles,
    removeReference: removeReference,
    moveReference: moveReference,
    parseReferenceResponse: parseReferenceResponse,
    renderReferences: renderReferences,
    MAX_REFERENCES: MAX_REFERENCES,
    // Finished-image shortcut: the checks assert the block reasons and drive
    // the fetch-and-upload round trip without a server.
    useJobImageAsReference: useJobImageAsReference,
    useAsReferenceBlock: useAsReferenceBlock,
    EDITABLE_KEYS: EDITABLE_KEYS,
    // Recipe surface: the real-Chrome check drives the lock behaviour and
    // reads what the draft would submit without a server.
    RECIPES: RECIPES,
    TURBO_STEPS: TURBO_STEPS,
    currentRecipe: currentRecipe,
    applyRecipeMode: applyRecipeMode,
    // TEST-ONLY: opens the write gate without a live server so the offline
    // checks can drive the upload path (production only ever opens it from a
    // RECOGNIZED status document).
    setApiAvailableForChecks: function (available) {
      if (typeof available === 'boolean') setApiAvailable(available);
    },
    setFetchTimeoutForChecks: function (ms) {
      if (Number.isInteger(ms) && ms > 0 && ms <= 60000) FETCH_TIMEOUT_MS = ms;
    },
    // Rewrite surface: the checks drive the suggestion flow offline (the
    // rewriter gate, the strict answer parser, stale discard and the
    // ratio->preset mapping) without a server or a model.
    parseRewriteBlock: parseRewriteBlock,
    parseRewriteAnswer: parseRewriteAnswer,
    setRewriterState: setRewriterState,
    presetForRatio: presetForRatio,
    requestRewrite: requestRewrite,
    applyProposal: applyProposal,
    editProposal: editProposal,
    keepOriginal: keepOriginal,
    rewriteState: rewriteState,
    // Renders a proposal the caller already validated with parseRewriteAnswer;
    // the offline render check uses it so the proposal DOM is exercised for real.
    showProposalForChecks: function (answer) {
      var parsed = parseRewriteAnswer(answer);
      if (parsed) showProposal(parsed);
      return !!parsed;
    },
    // The RAW renderer, bypassing the parser: the offline check must be able
    // to feed showProposal an answer the wire parser would reject, proving the
    // renderer itself refuses to become an applyable suggestion.
    showProposalRawForChecks: showProposal,
    bumpPromptRevisionForChecks: function () { bumpPromptRevision(); },
    PROMPT_MAX_CHARS: PROMPT_MAX_CHARS,
    REWRITE_TIMEOUT_MS: REWRITE_TIMEOUT_MS,
    // Rewriter-settings surface for the real-Chrome checks: the closed
    // parsers, the envelopes the page sends (a choice per draft kind; the probe
    // carries the endpoint alone), and the two calls the Settings page makes.
    parseRewriteSettings: parseRewriteSettings,
    rewriteSettingsEnvelope: rewriteSettingsEnvelope,
    rewriteTestEnvelope: rewriteTestEnvelope,
    renderRewriteSettings: renderRewriteSettings,
    rewriteLabel: rewriteLabel,
    loadRewriteSettings: loadRewriteSettings,
    saveRewriteSettings: saveRewriteSettings,
    testRewriteConnection: testRewriteConnection,
    API_REWRITE_SETTINGS: API_REWRITE_SETTINGS
  };
})();
