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

  var GATE_STATES = {
    adapter: ['not_implemented'],
    hardware: ['unknown'],
    assets: ['not_verified'],
    calibration: ['unknown']
  };

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

  // Accepted grammar (deliberately strict, checked BEFORE any BigInt):
  //   * surrounding whitespace trimmed;
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
    var s = String(str).trim();
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
    var raw = String($(id).value).trim();
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
    if (!$('prompt').value.trim()) {
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
      var seedRaw = String($('seed').value).trim();
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

  var lastApiOk = false;   // did the last server call succeed?

  function renderChip(text, backendText) {
    setText('connection-chip', text);
    setText('stat-backend', backendText);
  }

  // The status document is NOT trusted by shape alone: schema tag, strict
  // booleans and the known bounded gate states must all match, or we treat
  // the server as unrecognized — never as ready.
  function parseStatus(s) {
    if (!s || typeof s !== 'object') return null;
    if (s.schema !== STATUS_SCHEMA) return null;
    if (s.planning_available !== true) return null;
    if (!s.gates || typeof s.gates !== 'object') return null;
    var gates = Object.keys(GATE_STATES);
    for (var i = 0; i < gates.length; i++) {
      var gate = s.gates[gates[i]];
      if (!gate || typeof gate.state !== 'string' ||
          GATE_STATES[gates[i]].indexOf(gate.state) === -1) return null;
    }
    if (!Array.isArray(s.generation_blockers)) return null;
    for (var j = 0; j < s.generation_blockers.length; j++) {
      if (typeof s.generation_blockers[j] !== 'string') return null;
    }
    var gen = s.generation_available;
    return { recognized: true, planning: true, generation: gen === false ? false : 'unexpected' };
  }

  function statusUnrecognizedText() {
    return 'Connected, but the status document did not match the expected ' +
           'schema — treated as unrecognized, nothing is trusted.';
  }

  function checkStatus() {
    return fetch(API_STATUS, { headers: { 'Accept': 'application/json' } })
      .then(function (res) {
        if (!res.ok) {
          renderChip('Server error', 'Server error (status ' + res.status + ')');
          return null;
        }
        return res.json();
      })
      .catch(function () {
        // No server (file://, qwen-web not running) — honest, not an error storm.
        renderChip('Not connected', 'Not connected — no local engine server');
        setText('settings-backend',
          'No local engine server was reachable from this page. Opened ' +
          'over file:// or without qwen-web, the workspace is disconnected: ' +
          'planning and authoritative validation are unavailable.');
        return null;
      })
      .then(function (raw) {
        if (raw === null) { lastApiOk = false; return null; }
        var status = parseStatus(raw);
        if (!status) {
          lastApiOk = false;
          renderChip('Status unrecognized', 'Status unrecognized — not trusted');
          setText('settings-backend', statusUnrecognizedText());
          return null;
        }
        lastApiOk = true;
        renderChip('Connected — planning only', 'Connected — planning only');
        setText('settings-backend',
          'Connected to the local qwen-web server (planning only). ' +
          'Generation is not implemented; every gate below explains why.');
        setText('settings-gates',
          'adapter: not implemented · hardware: not probed (unknown) · ' +
          'assets: not verified · calibration: unknown');
        if (status.generation === 'unexpected') {
          // A server claiming generation_available would contradict this
          // build; refuse to trust it and keep Generate disabled.
          renderChip('Connected — status unexpected', 'Status unexpected — not trusted');
          setText('api-message',
            'The server reported an unexpected generation state; Generate ' +
            'stays disabled because generation is not implemented here.');
        }
        return status;
      });
  }

  /* ---------------- Typed API errors ---------------- */

  function renderApiError(res, payload) {
    lastApiOk = false;
    var message = 'The server refused the request (HTTP ' + res.status + ').';
    if (payload && payload.schema === ERROR_SCHEMA && typeof payload.code === 'string') {
      message = payload.code + ': ' +
                (typeof payload.message === 'string' ? payload.message : 'no message');
      if (Array.isArray(payload.fields)) {
        var fieldTexts = [];
        for (var i = 0; i < payload.fields.length; i++) {
          var f = payload.fields[i];
          if (f && typeof f.field === 'string' && typeof f.message === 'string') {
            fieldTexts.push(f.field + ': ' + f.message);
          }
        }
        if (fieldTexts.length) message += ' — ' + fieldTexts.join('; ');
      }
    }
    setText('api-message', message);
  }

  function apiPost(url, envelope) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Qwen-UI': '1' },
      body: JSON.stringify(envelope)
    }).then(function (res) {
      return res.json()
        .catch(function () { return null; })
        .then(function (payload) { return { res: res, payload: payload }; });
    });
  }

  /* ---------------- Plan flow ----------------
   * Both "Check configuration" and "Plan" submit the snapshot; the server
   * verdict is authoritative. On ANY failure the draft stays exactly as
   * typed — nothing is auto-filled or overwritten from a failed response.
   */

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
    setText('plan-hardware', plan.hardware && plan.hardware.status === 'unknown'
      ? 'unknown — not probed' : 'unknown — unrecognized status field');
    setText('plan-calibration', plan.calibration && plan.calibration.status === 'unknown'
      ? 'unknown — no calibration records' : 'unknown — unrecognized status field');
    var list = $('plan-blockers');
    while (list.firstChild) list.removeChild(list.firstChild);
    var blockers = Array.isArray(plan.generation_blockers) ? plan.generation_blockers : [];
    for (var i = 0; i < blockers.length; i++) {
      var li = document.createElement('li');
      li.textContent = String(blockers[i]);
      list.appendChild(li);
    }
  }

  function parsePlan(raw) {
    if (!raw || typeof raw !== 'object' || raw.schema !== PLAN_SCHEMA) return null;
    if (typeof raw.cfg_active !== 'boolean') return null;
    if (!Number.isInteger(raw.latent_tokens) || raw.latent_tokens < 0) return null;
    var c = raw.config;
    if (!c || typeof c !== 'object') return null;
    // Public seeds are decimal STRING/null — a numeric seed would mean the
    // browser is being handed a rounded JS Number. Refuse, never render it.
    if (!(c.seed === null || (typeof c.seed === 'string' && /^\d{1,20}$/.test(c.seed)))) return null;
    if (raw.planning_available !== true || raw.generation_available !== false) return null;
    return raw;
  }

  function runPlan(showFull) {
    setText('api-message', '');
    var built = buildEnvelope();
    if (!built.ok) {
      promptTouched = true;
      renderValidation(built.errors);
      setText('api-message', 'The draft has local form errors — nothing was sent. ' +
                             'The server verdict would be authoritative anyway.');
      return Promise.resolve(false);
    }
    return apiPost(API_PLAN, built.envelope).then(function (r) {
      if (!r.res.ok) {
        renderApiError(r.res, r.payload);
        setText('plan-panel-note', 'Configuration rejected by the server. ' +
                                   'Your draft was preserved exactly as typed.');
        $('plan-panel').hidden = false;
        return false;
      }
      var plan = parsePlan(r.payload);
      if (!plan) {
        lastApiOk = false;
        setText('api-message', 'The plan document did not match the expected ' +
                               'schema — not rendered, nothing is trusted.');
        return false;
      }
      lastApiOk = true;
      promptTouched = true;
      renderValidation({});
      planRenderFields(plan);
      setText('plan-panel-note', showFull
        ? 'Server plan (authoritative GenerationConfig validation). ' +
          'Planning only — nothing was generated.'
        : 'The server accepted this configuration (authoritative validation). ' +
          'Nothing was generated.');
      $('plan-panel').hidden = false;
      return true;
    }).catch(function () {
      renderChip('Not connected', 'Not connected — request failed');
      setText('api-message', 'The request failed (no reachable local server). ' +
                             'Nothing was sent or changed.');
      return false;
    });
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
    return fetch(API_EXPORT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Qwen-UI': '1' },
      body: JSON.stringify(built.envelope)
    }).then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return null; })
          .then(function (payload) { renderApiError(res, payload); return false; });
      }
      return res.blob().then(function (blob) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'qwen-editable-config.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        lastApiOk = true;
        setText('api-message', 'Downloaded the editable configuration (raw ' +
                               'bytes, no reserialization). Server runtime paths ' +
                               'are excluded by design; import is a later ' +
                               'milestone.');
        return true;
      });
    }).catch(function () {
      renderChip('Not connected', 'Not connected — request failed');
      setText('api-message', 'Export failed (no reachable local server). ' +
                             'Nothing was downloaded.');
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
    renderView(VIEWS[name] ? name : 'create');   // no focus on load
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

    $('btn-check-config').addEventListener('click', function () { runPlan(false); });
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
    });

    ['steps', 'seed', 'cfg', 'width', 'height', 'prompt'].forEach(function (id) {
      $(id).addEventListener('input', function () {
        if (id === 'width' || id === 'height') {
          setPreviewAspect(Number($('width').value), Number($('height').value));
          syncPresetButtons();
        }
        if (id === 'prompt') promptTouched = true;
        renderValidation(validateForm().errors);
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
  // buildEnvelope() is what the server receives.
  window.QwenStudio = {
    version: 'w2-api',
    formSnapshot: formSnapshot,
    buildEnvelope: buildEnvelope,
    validateForm: validateForm,
    parseSeed: parseSeed,
    randomSeedString: randomSeedString,
    checkStatus: checkStatus
  };
})();
