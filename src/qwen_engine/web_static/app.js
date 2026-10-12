/* Qwen Image Studio — W2 front-end (local same-origin status/plan API).
 *
 * Honest by construction:
 *   * The ONLY network calls are same-origin relative fetches to
 *     /api/v1/status, /api/v1/plan and /api/v1/config/export. Opened over
 *     file:// (or without the qwen-web server) every fetch fails and the UI
 *     says so: "Not connected" — failures are never swallowed as success.
 *   * Generate remains ALWAYS disabled: generation is W3 and not
 *     implemented anywhere. The submit handlers refuse unconditionally;
 *     the disabled button is a visual state, not the gate.
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
    'Generate is disabled: no generation adapter or job API is implemented ' +
    'yet (that is W3). Planning and configuration validation are available ' +
    'from the local server when this page is served by qwen-web.';

  /* ---- Same-origin API contract (relative paths only: never a host,
   *      never an absolute URL, never a CDN) ---- */
  var API_STATUS = '/api/v1/status';
  var API_PLAN = '/api/v1/plan';
  var API_EXPORT = '/api/v1/config/export';
  var PLAN_REQUEST_SCHEMA = 'qwen_engine.web_plan_request.v1';
  var STATUS_SCHEMA = 'qwen_engine.web_status.v1';
  var PLAN_SCHEMA = 'qwen_engine.web_plan.v1';
  var ERROR_SCHEMA = 'qwen_engine.web_error.v1';
  var UI_HEADER = 'X-Qwen-UI';

  // Closed gate contract: each gate has exactly one truthful state and one
  // stable code (qwen_engine.web/api.py). An arbitrary bounded string is a
  // schema rejection, never a trusted state.
  var GATE_STATES = {
    adapter:     { states: ['not_implemented'], code: 'adapter_not_implemented' },
    hardware:    { states: ['unknown'],         code: 'hardware_not_probed' },
    assets:      { states: ['not_verified'],    code: 'assets_unverified' },
    calibration: { states: ['unknown'],         code: 'calibration_unknown' }
  };

  // CLOSED blocker-code list — exactly what the real server can emit.
  var KNOWN_BLOCKER_CODES = [
    'adapter_not_implemented', 'hardware_not_probed', 'assets_unverified',
    'calibration_unknown', 'manifest_unconfigured', 'manifest_file_missing',
    'manifest_invalid', 'manifest_unreadable'
  ];

  // The FOUR gate codes are ALWAYS blockers; the manifest contributes its
  // closed reason code ONLY when configured=false or valid=false (exact
  // server contract, api._blockers). A document carrying any other mix —
  // even a recognizable-looking subset — is refused, so a malformed server
  // can never hide a required blocker behind a plausible shape.
  var REQUIRED_BLOCKER_CODES = [
    'adapter_not_implemented', 'hardware_not_probed', 'assets_unverified',
    'calibration_unknown'
  ];
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

  var PRESETS = {
    square:    { width: 1024, height: 1024 },
    landscape: { width: 1344, height: 768 },
    portrait:  { width: 768,  height: 1344 }
  };

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
    return { ok: Object.keys(errors).length === 0, errors: errors };
  }

  function setAriaInvalid(id, invalid) {
    var el = $(id);
    if (el) el.setAttribute('aria-invalid', invalid ? 'true' : 'false');
  }

  var promptTouched = false;

  function renderValidation(errors) {
    setText('steps-error', errors.steps || '');
    setText('seed-error', errors.seed || '');
    setText('cfg-error', errors.cfg || '');
    setText('dims-error', errors.dims || '');
    // The prompt error only appears once the user has edited the prompt or
    // attempted to submit — never as a first-load alert.
    var showPrompt = promptTouched && !!errors.prompt;
    setText('prompt-error', showPrompt ? errors.prompt : '');
    setAriaInvalid('steps', !!errors.steps);
    setAriaInvalid('seed', !!errors.seed);
    setAriaInvalid('cfg', !!errors.cfg);
    setAriaInvalid('prompt', showPrompt);
    // The dimension error message is shared, so it marks BOTH controls.
    setAriaInvalid('width', !!errors.dims);
    setAriaInvalid('height', !!errors.dims);
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
      seed: seedWireString($('seed').value),   // decimal STRING or null (UI-only)
      true_cfg_scale: Number($('cfg').value),
      quant: $('quant').value,                 // fp16 | q8 | q4 (exact config.py values)
      offload: $('offload').value,             // resident | streamed | offload
      attention_backend: $('attention').value, // auto | sdpa | efficient | eager
      vae_tiled_decode: $('vae-tiling').checked
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
          true_cfg_scale: snap.true_cfg_scale,
          seed: snap.seed,                     // decimal STRING or null — never a Number
          quant: snap.quant,
          offload: snap.offload,
          attention_backend: snap.attention_backend,
          vae_tiled_decode: snap.vae_tiled_decode
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
  var apiConnected = false;      // a RECOGNIZED status document was seen
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
  // truncated body cannot hang the page.
  function readBodyLimited(res) {
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
        if (total > MAX_RESPONSE_BYTES) {
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
  //   { res, bodyError: 'too_large' }  — body exceeded the fixed cap
  //   { res, bytes }                   — mode 'raw': the COMPLETE raw bytes,
  //                                    untouched (export downloads these)
  function apiRequest(url, opts, mode) {
    var controller = new AbortController();
    var timedOut = false;
    var timer = setTimeout(function () {
      timedOut = true;
      controller.abort();
    }, FETCH_TIMEOUT_MS);
    var requestOpts = {};
    if (opts) {
      for (var k in opts) {
        if (Object.prototype.hasOwnProperty.call(opts, k)) requestOpts[k] = opts[k];
      }
    }
    requestOpts.signal = controller.signal;
    // No HTTP-cache path for API calls: every status/plan/export request
    // goes to the server (a cached status is NEVER an authoritative one,
    // and two checks must be two REAL requests — verifiable in the wire).
    requestOpts.cache = 'no-store';
    function settle(value) { clearTimeout(timer); return value; }
    return fetch(url, requestOpts).then(function (res) {
      return readBodyLimited(res).then(function (bytes) {
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
  // any disconnect or schema rejection takes them away again. Generate is
  // NEVER touched here — it is and stays disabled (generation is W3).
  function setApiAvailable(available) {
    apiConnected = available;
    $('btn-plan').disabled = !available;
    $('btn-export').disabled = !available;
  }

  // The status document is NOT trusted by shape alone: schema tag, strict
  // literal booleans (generation_available must be the literal false), the
  // exact known gates with their CLOSED states AND codes, blocker codes
  // from the closed list only, and the exact manifest verification — ALL
  // must match, or the whole document is rejected. A JSON null parses to
  // the value null and is rejected here by the schema, never by a property
  // access on null.
  function parseStatus(s) {
    if (!s || typeof s !== 'object') return null;   // null-safe by design
    if (s.schema !== STATUS_SCHEMA) return null;
    if (s.planning_available !== true) return null;
    if (s.generation_available !== false) return null;  // literal false only
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
    // EXACT blocker set (four required + manifest reason when applicable).
    var manifest = parseManifestCore(s.manifest);
    if (!manifest) return null;   // status ALWAYS carries the manifest block
    if (!blockersExact(s.generation_blockers, manifest.blocker)) return null;
    return { recognized: true, planning: true, generation: false };
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
      if (!Number.isInteger(m.schema_version) || m.schema_version < 1 ||
          m.schema_version > 64) return null;
      if (!Number.isInteger(m.artifact_count) || m.artifact_count < 0 ||
          m.artifact_count > 1000000) return null;
    }
    return { configured: m.configured, valid: m.valid, blocker: blocker };
  }

  // The EXACT blocker set: the four REQUIRED gate codes plus the manifest
  // blocker (when one applies), each exactly once. Missing, duplicated or
  // extra codes all reject — a malformed server can never hide a required
  // blocker behind an otherwise recognizable document.
  function blockersExact(list, manifestBlocker) {
    if (!Array.isArray(list)) return false;
    var expected = REQUIRED_BLOCKER_CODES.slice();
    if (manifestBlocker) expected.push(manifestBlocker);
    if (list.length !== expected.length) return false;
    for (var i = 0; i < expected.length; i++) {
      if (list.indexOf(expected[i]) === -1) return false;
    }
    for (var j = 0; j < list.length; j++) {
      if (expected.indexOf(list[j]) === -1) return false;
    }
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
        renderChip('Connected — planning only', 'Connected — planning only');
        setText('settings-backend',
          'Connected to the local qwen-web server (planning only). ' +
          'Generation is not implemented; every gate below explains why.');
        setText('settings-gates',
          'adapter: not implemented · hardware: not probed (unknown) · ' +
          'assets: not verified · calibration: not verified for this ' +
          'server (unknown)');
        setText('settings-generation',
          'Planning is available against the configured manifest. ' +
          'Generation is NOT implemented — Generate stays disabled.');
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

  function envelopeSignature(built) {
    return built.ok ? JSON.stringify(built.envelope) : null;
  }

  // A visible validated result must NEVER keep looking current after the
  // draft changed — including preset, random and change-only fields that
  // produce no inline errors. The panel stays visible but is clearly
  // labelled stale; nothing is silently re-trusted.
  function syncPlanStaleness() {
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
      if (!plan) {
        setApiAvailable(false);
        renderChip('Status rejected',
          'Status rejected — server contract broken');
        setText('api-message', 'The plan document did not match the expected ' +
                               'schema — nothing is rendered or trusted; ' +
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
    setText('plan-cfg', plan.cfg_active === true ? 'active (scale ' + plan.config.true_cfg_scale + ')'
                                                 : 'inactive (scale ' + plan.config.true_cfg_scale + ')');
    setText('plan-tokens', String(plan.latent_tokens));
    setText('plan-quant', plan.config.quant + ' · ' + plan.config.offload + ' · ' +
                          plan.config.attention_backend + ' · VAE tiling ' +
                          (plan.config.vae_tiled_decode ? 'on' : 'off'));
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
    quant: ['fp16', 'q8', 'q4'],
    offload: ['resident', 'streamed', 'offload'],
    attention_backend: ['auto', 'sdpa', 'efficient', 'eager']
  };

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
    if (raw.planning_available !== true || raw.generation_available !== false) return null;
    if (typeof raw.cfg_active !== 'boolean') return null;
    if (!Number.isInteger(raw.latent_tokens) || raw.latent_tokens < 0) return null;
    var c = raw.config;
    if (!c || typeof c !== 'object') return null;
    var want = ['prompt', 'negative_prompt', 'width', 'height', 'steps',
                'true_cfg_scale', 'seed', 'quant', 'offload',
                'attention_backend', 'vae_tiled_decode'];
    var have = Object.keys(c);
    if (have.length !== want.length) return null;   // no extra runtime fields
    for (var i = 0; i < want.length; i++) {
      if (!Object.prototype.hasOwnProperty.call(c, want[i])) return null;
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
    // cfg_active must EQUAL the server's single source of truth
    // (config.py): a non-blank negative prompt under PYTHON strip rules
    // AND a scale strictly above 1.0.
    var cfgActive = pyStrip(c.negative_prompt).length > 0 && c.true_cfg_scale > 1;
    if (raw.cfg_active !== cfgActive) return null;
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
    // EXACT blocker set (four required + manifest reason when applicable).
    if (!blockersExact(raw.generation_blockers, manifest.blocker)) return null;
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
    syncPresetButtons();
    setPreviewAspect(preset.width, preset.height);
    renderValidation(validateForm().errors);
    syncPlanStaleness();   // preset edits stale any visible result too
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
    ['create', 'history', 'settings'].forEach(function (name) {
      var link = $('nav-' + name);
      if (name === active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    activeView = active;
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

  /* ---------------- Submission (generation refuses) ----------------
   * Generation is not implemented (W3). The handlers enforce that directly:
   * they never pretend a submission happened, regardless of how they were
   * reached. The disabled button is a visual state, not the gate.
   */

  function refuseSubmit() {
    setText('generate-reason',
      'Nothing was submitted: generation is not implemented in this build. ' +
      'No generation adapter or job API exists, so there is no submit path. ' +
      'Use Plan to validate this configuration with the local server.');
  }

  /* ---------------- Wiring ---------------- */

  function init() {
    $('create-form').addEventListener('submit', function (ev) {
      ev.preventDefault();
      promptTouched = true;
      renderValidation(validateForm().errors);
      refuseSubmit();
    });

    $('create-form').addEventListener('keydown', function (ev) {
      if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
        ev.preventDefault();
        promptTouched = true;
        renderValidation(validateForm().errors);
        refuseSubmit();
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

    ['steps', 'seed', 'cfg', 'width', 'height', 'prompt', 'negative'].forEach(function (id) {
      $(id).addEventListener('input', function () {
        if (id === 'width' || id === 'height') {
          setPreviewAspect(Number($('width').value), Number($('height').value));
          syncPresetButtons();
        }
        if (id === 'prompt') promptTouched = true;
        renderValidation(validateForm().errors);
        syncPlanStaleness();
      });
    });

    // Change-only fields (they never carry inline errors) must ALSO make a
    // visible result clearly stale, not leave it looking validated.
    ['quant', 'offload', 'attention', 'vae-tiling'].forEach(function (id) {
      $(id).addEventListener('change', function () {
        renderValidation(validateForm().errors);
        syncPlanStaleness();
      });
    });

    window.addEventListener('hashchange', routeOnHashchange);
    initialRoute();
    syncPresetButtons();
    renderValidation(validateForm().errors);

    var generate = $('generate-btn');
    generate.disabled = true;   // unconditional in W2: generation is W3
    setText('generate-reason', GENERATION_UNAVAILABLE_REASON);
    setText('settings-generation',
      'Planning and authoritative config validation come from the local ' +
      'qwen-web server; generation needs a pipeline adapter and job API that ' +
      'do not exist yet (W3).');

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
    version: 'w2-api',
    formSnapshot: formSnapshot,
    buildEnvelope: buildEnvelope,
    validateForm: validateForm,
    parseSeed: parseSeed,
    parseStatus: parseStatus,
    parsePlan: parsePlan,
    randomSeedString: randomSeedString,
    checkStatus: checkStatus,
    setFetchTimeoutForChecks: function (ms) {
      if (Number.isInteger(ms) && ms > 0 && ms <= 60000) FETCH_TIMEOUT_MS = ms;
    }
  };
})();
