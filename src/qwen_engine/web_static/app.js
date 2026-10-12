/* Qwen Image Studio — W1 static workspace front-end.
 *
 * Honest by construction:
 *   * W1 ships NO server and makes NO network requests at all: no fetch and
 *     no speculative status probe of any kind. Whether this page is opened
 *     over file:// or
 *     served by static HTTP, nothing connects to an engine — both are
 *     equally disconnected.
 *   * Generate is therefore ALWAYS disabled, with a visible reason: no
 *     generation adapter or job API is implemented. The disabled button is
 *     not the gate — the submit handlers themselves refuse unconditionally.
 *     A reviewed status/plan contract may arrive with W2; generation is W3
 *     and is never simulated here.
 *   * No fake results, jobs, timings or thumbnails. No draft autosave.
 *   * Seeds are decimal STRINGS handled with BigInt (full uint64 range) —
 *     never coerced through JS Number.
 *   * User-supplied text is only ever inserted via textContent.
 *   * Browser-side checks are lightweight FORM validation only; a future
 *     server would remain the source of truth for GenerationConfig
 *     validation.
 *
 * This snapshot is a UI wire format, NOT the engine's GenerationConfig JSON
 * (whose seed semantics the server would own). Do not treat it as valid
 * GenerationConfig output.
 */
'use strict';

(function () {
  var SEED_MIN = 0n;
  var SEED_MAX = 18446744073709551615n; // 2^64 - 1
  var SEED_MAX_DIGITS = SEED_MAX.toString().length; // 20

  var GENERATION_UNAVAILABLE_REASON =
    'Generate is disabled: no generation adapter or job API is implemented ' +
    'yet. This is a static page — nothing connects to an engine server, ' +
    'whether it is opened over file:// or served statically.';

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
  // The full uint64 maximum stays representable as an exact decimal STRING.
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

  /* ---------------- Lightweight form validation ---------------- */

  function isValidDimension(n) {
    return Number.isInteger(n) && n >= 64 && n <= 4096 && n % 32 === 0;
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
    var steps = Number($('steps').value);
    if (!Number.isInteger(steps) || steps < 1 || steps > 1000) {
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
    var cfg = Number($('cfg').value);
    if (!Number.isFinite(cfg) || cfg < 1) {
      errors.cfg = 'Guidance must be a finite number of 1 or more.';
    }
    var width = Number($('width').value);
    var height = Number($('height').value);
    if (!isValidDimension(width) || !isValidDimension(height)) {
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

  /* ---------------- UI wire snapshot ----------------
   * Keys mirror the engine's GenerationConfig fields (negative_prompt,
   * true_cfg_scale, attention_backend, vae_tiled_decode, ...) with enum
   * values taken verbatim from config.py; the ONE deliberate difference is
   * the UI-only decimal-STRING seed. This snapshot is NOT an exported
   * GenerationConfig JSON document and nothing consumes it in W1 — there is
   * no endpoint to send it to.
   */

  function formSnapshot() {
    return {
      kind: 'ui-wire/w1',
      prompt: $('prompt').value,
      negative_prompt: $('negative').value,
      width: Number($('width').value),
      height: Number($('height').value),
      steps: Number($('steps').value),
      seed: seedWireString($('seed').value),   // decimal STRING or null (UI-only)
      true_cfg_scale: Number($('cfg').value),
      quant: $('quant').value,                 // fp16 | q8 | q4
      offload: $('offload').value,             // resident | streamed | offload
      attention_backend: $('attention').value, // auto | sdpa | efficient | eager
      vae_tiled_decode: $('vae-tiling').checked
    };
  }

  /* ---------------- Connection state (no probe: nothing to probe) --------- */

  function renderConnection() {
    var status = 'Generation unavailable';
    setText('connection-chip', status);
    setText('stat-backend', status);
    setText('settings-backend',
      'No engine server exists yet in this build. This page never contacts ' +
      'one: it is static, so both file:// and plain static HTTP hosting are ' +
      'disconnected. A reviewed status/plan contract is a W2 concern.');

    var generate = $('generate-btn');
    generate.disabled = true;   // unconditional in W1
    setText('generate-reason', GENERATION_UNAVAILABLE_REASON);
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

  /* ---------------- Submission ----------------
   * Generation is not implemented in W1. The handlers enforce that directly:
   * they never pretend a submission happened, regardless of how they were
   * reached. The disabled button is a visual state, not the gate.
   */

  function refuseSubmit() {
    // Honest, unconditional refusal — reachable via the form submit event or
    // Ctrl/Meta+Enter, never reports success and never touches the network.
    setText('generate-reason',
      'Nothing was submitted: generation is not implemented in this build. ' +
      'No generation adapter or job API exists, so there is no submit path.');
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
    renderConnection();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Small explicit surface for manual inspection; snapshot() is UI wire
  // format, not GenerationConfig, and nothing consumes it in W1.
  window.QwenStudio = {
    version: 'w1-static',
    formSnapshot: formSnapshot,
    validateForm: validateForm,
    parseSeed: parseSeed,
    randomSeedString: randomSeedString
  };
})();
