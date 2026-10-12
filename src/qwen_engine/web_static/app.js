/* Qwen Image Studio — W1 static workspace front-end.
 *
 * Honest by construction:
 *   * Starts "Not connected". Generate stays disabled with a visible reason
 *     until a future same-origin capability endpoint reports ready.
 *   * No fake results, jobs, timings or thumbnails. No draft autosave.
 *   * Seeds are decimal STRINGS handled with BigInt (full uint64 range) —
 *     never coerced through JS Number.
 *   * User-supplied text is only ever inserted via textContent.
 *   * Browser-side checks are lightweight FORM validation only; the engine
 *     server remains the source of truth for GenerationConfig validation.
 *
 * This snapshot is a UI wire format, NOT the engine's GenerationConfig JSON
 * (whose seed semantics the server will own). Do not treat it as valid
 * GenerationConfig output.
 */
'use strict';

(function () {
  // The single same-origin endpoint this page may ever contact. Probed only
  // when served over http/https by a future W2 server — never under file://,
  // and never with anything derived from user input.
  var CAPABILITY_PATH = '/api/v1/capability';

  var SEED_MIN = 0n;
  var SEED_MAX = 18446744073709551615n; // 2^64 - 1

  var PRESETS = {
    square:    { width: 1024, height: 1024 },
    landscape: { width: 1344, height: 768 },
    portrait:  { width: 768,  height: 1344 }
  };

  var VIEWS = { create: 'view-create', history: 'view-history', settings: 'view-settings' };

  function $(id) { return document.getElementById(id); }

  function setText(id, text) {
    var node = $(id);
    if (node) node.textContent = text;
  }

  /* ---------------- Seed handling (BigInt decimal strings) ---------------- */

  function parseSeed(str) {
    var s = String(str).trim();
    if (!/^\d+$/.test(s)) return null;           // decimal digits only
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
    var steps = Number($('steps').value);
    if (!Number.isInteger(steps) || steps < 1 || steps > 1000) {
      errors.steps = 'Steps must be a whole number from 1 to 1000.';
    }
    var seedStr = seedWireString($('seed').value);
    if (seedStr === null) {
      errors.seed = 'Seed must be a whole-number string from 0 to ' +
                    '18446744073709551615.';
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

  function renderValidation(errors) {
    setText('steps-error', errors.steps || '');
    setText('seed-error', errors.seed || '');
    setText('cfg-error', errors.cfg || '');
    setText('dims-error', errors.dims || '');
  }

  /* ---------------- UI wire snapshot ----------------
   * Keys mirror the engine's GenerationConfig fields (negative_prompt,
   * true_cfg_scale, attention_backend, vae_tiled_decode, ...) with enum
   * values taken verbatim from config.py; the ONE deliberate difference is
   * the UI-only decimal-STRING seed, kept exact until the W2 API converts a
   * validated decimal to a Python int. This snapshot is NOT an exported
   * GenerationConfig JSON document.
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

  /* ---------------- Capability / connection state ---------------- */

  var connection = {
    connected: false,
    ready: false,
    reason: 'This page has no engine server to talk to yet. Open it through ' +
            'the local studio server (a later milestone) to connect.'
  };

  function renderConnection() {
    var chip = $('connection-chip');
    var status = connection.ready ? 'Engine ready' : 'Not connected';
    chip.textContent = status;
    chip.classList.toggle('is-ready', connection.ready);
    setText('stat-backend', status);
    setText('settings-backend', connection.connected
      ? 'Connected, but not reporting ready yet.'
      : 'Not connected. This build of the page has no engine server to talk ' +
        'to yet.');

    var generate = $('generate-btn');
    var reason = $('generate-reason');
    var formOk = validateForm().ok;
    var promptOk = $('prompt').value.trim().length > 0;

    if (!connection.ready) {
      generate.disabled = true;
      reason.textContent = 'Generate is disabled: ' + connection.reason +
        ' Generation not available.';
    } else if (!promptOk) {
      generate.disabled = true;
      reason.textContent = 'Generate is disabled: enter a prompt first.';
    } else if (!formOk) {
      generate.disabled = true;
      reason.textContent = 'Generate is disabled: fix the highlighted ' +
        'form fields (lightweight check only — the server validates ' +
        'authoritatively).';
    } else {
      generate.disabled = false;
      reason.textContent = '';
    }
  }

  function probeCapability() {
    // Only ever when served over http(s); the path is fixed, never derived
    // from user input. file:// pages simply stay honestly disconnected.
    var proto = location.protocol;
    if (proto !== 'http:' && proto !== 'https:') {
      renderConnection();
      return;
    }
    fetch(CAPABILITY_PATH, { headers: { 'accept': 'application/json' } })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (cap) {
        connection.connected = true;
        connection.ready = cap && cap.ready === true;
        connection.reason = connection.ready
          ? ''
          : 'The engine server is connected but has not reported ready.';
        renderConnection();
      })
      .catch(function () {
        connection.connected = false;
        connection.ready = false;
        connection.reason = 'The local engine server did not answer.';
        renderConnection();
      });
  }

  /* ---------------- Presets & preview ---------------- */

  function applyPreset(name) {
    var preset = PRESETS[name];
    if (!preset) return;
    $('width').value = String(preset.width);
    $('height').value = String(preset.height);
    var buttons = document.querySelectorAll('.preset-btn');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute('aria-pressed',
        String(buttons[i].dataset.preset === name));
    }
    setPreviewAspect(preset.width, preset.height);
    renderConnection();
  }

  function setPreviewAspect(width, height) {
    var frame = $('preview-frame');
    if (isValidDimension(width) && isValidDimension(height) && height > 0) {
      frame.style.setProperty('--ar-w', String(width));
      frame.style.setProperty('--ar-h', String(height));
    }
  }

  /* ---------------- View routing ---------------- */

  function currentView() {
    var name = (location.hash || '').replace(/^#/, '');
    return VIEWS[name] ? name : 'create';
  }

  function route() {
    var active = currentView();
    Object.keys(VIEWS).forEach(function (name) {
      $(VIEWS[name]).hidden = name !== active;
    });
    ['create', 'history', 'settings'].forEach(function (name) {
      var link = $('nav-' + name);
      if (name === active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
  }

  /* ---------------- Submission ---------------- */

  function trySubmit() {
    // Called only from an explicit user action. If Generate is genuinely
    // enabled we will hand the UI snapshot to the W2/W3 API; nothing exists
    // yet, so this can never fire while disconnected.
    if ($('generate-btn').disabled) return;
    // W3 will wire formSnapshot() to the real job endpoint here.
  }

  /* ---------------- Wiring ---------------- */

  function init() {
    $('create-form').addEventListener('submit', function (ev) {
      ev.preventDefault();
      renderValidation(validateForm().errors);
      trySubmit();
    });

    $('create-form').addEventListener('keydown', function (ev) {
      if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
        ev.preventDefault();
        renderValidation(validateForm().errors);
        trySubmit();
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
      renderConnection();
    });

    ['steps', 'seed', 'cfg', 'width', 'height', 'prompt'].forEach(function (id) {
      $(id).addEventListener('input', function () {
        if (id === 'width' || id === 'height') {
          setPreviewAspect(Number($('width').value), Number($('height').value));
        }
        renderValidation(validateForm().errors);
        renderConnection();
      });
    });

    window.addEventListener('hashchange', route);
    route();
    renderValidation(validateForm().errors);
    renderConnection();
    probeCapability();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Small explicit surface for the future server integration and for
  // manual inspection; snapshot() is UI wire format, not GenerationConfig.
  window.QwenStudio = {
    version: 'w1-static',
    formSnapshot: formSnapshot,
    validateForm: validateForm,
    parseSeed: parseSeed,
    randomSeedString: randomSeedString
  };
})();
