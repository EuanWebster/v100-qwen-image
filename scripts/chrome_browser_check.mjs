#!/usr/bin/env node
/* Real-Chrome behavioural check for the static web workspace (W1 assets,
 * W2-aware expectations).
 *
 * NOT part of the guarded main suite (`run_cpu_tests.sh` stays static-only,
 * ML-free and socket-free). Run it directly, and only when a Chrome binary
 * is available and the machine has free task slots:
 *
 *     CHROME_PATH=/path/to/chrome node scripts/chrome_browser_check.mjs
 *
 * Dependencies: Node built-ins/global WebSocket and the existing Python
 * interpreter for a private Linux lineage owner (shared lifecycle helper).
 * It launches its own Chrome lineage
 * with a throwaway profile under the system temp dir and always cleans
 * that lineage up (Browser.close, then owner-PID TERM — never a broad kill).
 * Chrome's own sandbox stays enabled unless QWEN_CHROME_NO_SANDBOX=1 is set
 * explicitly. Every wait is bounded: DevTools announce, websocket open,
 * EACH CDP send and the whole check; an early child exit rejects startup.
 * A kernel-unreapable child retains its owner/profile instead of falsely
 * claiming cleanup or being orphaned by an explicit parent process.exit.
 * No screenshots are written or committed.
 *
 * Covers: seed edges/length/non-ASCII, snapshot keys, presets and manual
 * dimension matching, preview ratio, hash nav + heading focus (and the
 * #main skip-link case), aria-invalid wiring, Generate ALWAYS disabled with
 * no remote probes (W2: the only fetch is the same-origin /api/v1/status
 * check, which must fail honestly to "Not connected" on a file:// page),
 * and horizontal overflow at 1440/375/320. The served-UI (real status/plan)
 * browser checks live in scripts/web_integration_check.mjs.
 *
 * Historical status: not run to green inside the 2026-10-12 review container (the
 * cgroup pids controller was saturated by processes from earlier sessions;
 * that cause is REPORTED, not verified). On 2026-10-13 this check passed
 * 61/61 on the clean sandbox; details in docs/web-review.md. Every new run
 * still performs its own capacity preflight and reports actual outcomes.
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  cgroupCapacity, launchChrome, makeCdp, openCdpSocket, shutdownChrome, watchdog,
} from './chrome_lifecycle.mjs';

const WHOLE_CHECK_MS = 180000;

const CHROME = process.env.CHROME_PATH;
if (!CHROME) {
  console.error('chrome_browser_check: CHROME_PATH is not set — this script ' +
                'requires an existing Chrome binary and is NOT part of the ' +
                'guarded main suite.');
  process.exit(2);
}

const here = dirname(fileURLToPath(import.meta.url));
const pageUrl = 'file://' + resolve(here, '..', 'src', 'qwen_engine',
                                    'web_static', 'index.html');

// Honest preflight on cgroup-limited machines: report the free capacity and
// refuse EARLY instead of hanging. The CAUSE is not claimed — no procstat
// diagnostics run here, so the message stays free-capacity-only.
{
  const cap = cgroupCapacity();
  if (cap && !cap.ok) {
    console.error(`chrome_browser_check: only ${cap.free} free task slots ` +
                  `(pids ${cap.detail} in the cgroup pids controller) — ` +
                  'Chrome cannot start reliably. Cause not verified from ' +
                  'here; run this script on a clean sandbox. This is an ' +
                  'environment problem, not a product failure.');
    process.exit(3);
  }
}

const profileDir = mkdtempSync(join(tmpdir(), 'qwen-chrome-check-'));

let results = [];
let chrome = null;
let ws = null;
let cdp = null;
let failed = 0;

function check(name, ok, detail = '') {
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  [' + detail + ']' : ''}`);
  if (!ok) failed++;
}

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const launched = launchChrome({ chromePath: CHROME, profileDir });
  chrome = launched.chrome;
  const url = await launched.wsUrl;   // early child exit rejects here
  const sock = openCdpSocket(url);
  ws = sock.ws;
  await sock.ready;
  cdp = makeCdp(ws);

  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  const s = (method, params) => cdp.send(method, params || {}, sessionId);

  // fetch spy installed BEFORE any page script runs
  await s('Page.enable');
  await s('Runtime.enable');
  await s('Network.enable');
  const netRequests = [];
  cdp.onEvent = (msg) => {
    if (msg.method === 'Network.requestWillBeSent') {
      // the resource type lives on params.type — NOT params.request.type
      netRequests.push(`${msg.params.type} ${msg.params.request.url}`);
    }
  };
  await s('Page.addScriptToEvaluateOnNewDocument', { source:
    'window.__fetchCalls = []; var __of = window.fetch;' +
    'window.fetch = function () { window.__fetchCalls.push(String(arguments[0]));' +
    'return __of ? __of.apply(this, arguments) : Promise.reject(new Error("no fetch")); };' });

  const evaluate = async (expression) => {
    const r = await s('Runtime.evaluate',
      { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval exception');
    return r.result.value;
  };

  await s('Emulation.setDeviceMetricsOverride',
    { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });

  await s('Page.navigate', { url: pageUrl });
  await delay(600); // local file:// load; settle deterministically

  /* ---- 1. no remote probes: over file:// the only fetch attempt is the
   * SAME-ORIGIN status call (which must fail honestly to "Not connected") ---- */
  const remoteProbed = netRequests.filter((r) => /https?:/.test(r));
  check('zero network-origin (http/https) probes from a file:// page',
        remoteProbed.length === 0, remoteProbed.join('; '));
  const fetchCalls = await evaluate('window.__fetchCalls.join(",")');
  check('only the same-origin status endpoint is ever fetched',
        fetchCalls === '' || fetchCalls === '/api/v1/status', fetchCalls);
  const chip = await evaluate('document.getElementById("connection-chip").textContent');
  check('file:// lands honestly on Not connected',
        /Not connected/.test(chip), chip);

  /* ---- 2. app surface + initial honest state ---- */
  check('QwenStudio exposes a versioned test surface',
        /^w[0-9]+-[a-z0-9-]+$/.test(
          await evaluate('window.QwenStudio && window.QwenStudio.version')));
  check('Generate starts disabled',
        await evaluate('document.getElementById("generate-btn").disabled') === true);
  check('no autofocus on load (activeElement is body)',
        await evaluate('document.activeElement.tagName') === 'BODY');
  const reason = await evaluate('document.getElementById("generate-reason").textContent');
  check('disabled reason names the missing generation switch',
        /--enable-generation/.test(reason), reason);

  /* ---- 3. seed grammar edges ---- */
  const seedCases = [
    ['18446744073709551615', '18446744073709551615'], // full uint64 max
    ['18446744073709551616', 'null'],                 // out of range
    ['9'.repeat(21), 'null'],                         // 21 digits: too long
    ['0'.repeat(21), 'null'],                         // leading zeros > 20: too long
    ['0'.repeat(20), '0'],                            // 20 chars still fine
    [' 42 ', '42'],                                   // trim ok
    ['', 'null'], ['-1', 'null'], ['1e3', 'null'], ['4a', 'null'],
    ['١٢٣', 'null'],                                  // Arabic-Indic digits
    ['1_000', 'null'], ['0x10', 'null'],
  ];
  for (const [input, want] of seedCases) {
    const got = await evaluate(`String(window.QwenStudio.parseSeed(${JSON.stringify(input)}))`);
    check(`parseSeed(${JSON.stringify(input)}) -> ${want}`, got === want, `got ${got}`);
  }

  /* ---- 4. Random button: valid exact strings ---- */
  const seeds = [];
  for (let i = 0; i < 5; i++) {
    await evaluate('document.getElementById("random-seed").click()');
    seeds.push(await evaluate('document.getElementById("seed").value'));
  }
  check('random seeds are <=20-digit decimals and parse',
        seeds.every((s) => /^\d{1,20}$/.test(s)), seeds.join(','));
  check('random seeds are not all identical', new Set(seeds).size > 1);
  await evaluate('document.getElementById("seed").value = "0"');
  await evaluate(`document.getElementById('seed').dispatchEvent(new Event('input'))`);

  /* ---- 5. snapshot keys mirror GenerationConfig, seed is a STRING ---- */
  const snapInfo = await evaluate(`(function () {
    var s = window.QwenStudio.formSnapshot();
    return { keys: Object.keys(s).sort().join(','), seedType: typeof s.seed,
             seed: String(s.seed) };
  })()`);
  // Full GenerationConfig surface the UI owns: the four W3/W4 fields
  // (output_format, transparent_background, references) were previously
  // missing from this list, which is why the check reported a stale
  // snapshot mismatch rather than a real regression.
  const wantKeys = ['attention_backend', 'height', 'kind', 'negative_prompt',
                    'offload', 'output_format', 'prompt', 'quant', 'references',
                    'seed', 'steps', 'transparent_background', 'true_cfg_scale',
                    'vae_tiled_decode', 'width'].join(',');
  check('snapshot keys mirror GenerationConfig fields + kind (snake_case, no aliases)',
        snapInfo.keys === wantKeys, snapInfo.keys);
  check('snapshot seed is a decimal STRING',
        snapInfo.seedType === 'string' && snapInfo.seed === '0', snapInfo.seedType);

  /* ---- 5b. reference images (W4): envelope order contract, cap, strict
   * response parsing and the status-gated Add button — all without a
   * server, so this is the offline half of the upload flow ---- */
  check('reference cap mirrors config.REF_MAX (10)',
        await evaluate('window.QwenStudio.MAX_REFERENCES') === 10);
  const emptyEnv = await evaluate(`(function () {
    document.getElementById('prompt').value = 'a lighthouse';
    document.getElementById('prompt').dispatchEvent(new Event('input'));
    var b = window.QwenStudio.buildEnvelope();
    return b.ok ? JSON.stringify({ refs: b.envelope.config.references,
                                   ids: window.QwenStudio.referenceIds(),
                                   keys: Object.keys(b.envelope.config).sort() })
                : 'invalid';
  })()`);
  check('empty draft envelope carries references: [] in order',
        JSON.parse(emptyEnv).refs.length === 0 &&
        JSON.parse(emptyEnv).ids.length === 0, emptyEnv);
  check('envelope config carries all 14 editable keys',
        JSON.parse(emptyEnv).keys.length === 14, emptyEnv);
  check('Add reference is disabled while disconnected (writes need status)',
        await evaluate('document.getElementById("add-reference").disabled') === true);
  check('reference empty-state is visible with no references',
        await evaluate('document.getElementById("reference-empty").hidden') === false);
  check('reference list starts empty',
        await evaluate('document.getElementById("reference-list").children.length') === 0);
  const goodRef = { schema: 'qwen_engine.web_reference.v1', id: 'a'.repeat(64),
                    width: 512, height: 768, has_alpha: true, bytes: 1234 };
  const refCases = [
    ['valid document accepted', goodRef, true],
    ['wrong schema rejected', { ...goodRef, schema: 'web_reference.v1' }, false],
    ['uppercase id rejected', { ...goodRef, id: 'A'.repeat(64) }, false],
    ['short id rejected', { ...goodRef, id: 'a'.repeat(63) }, false],
    ['non-integer width rejected', { ...goodRef, width: 512.5 }, false],
    ['over-cap width rejected', { ...goodRef, width: 4097 }, false],
    ['zero bytes rejected', { ...goodRef, bytes: 0 }, false],
    ['over-cap bytes rejected', { ...goodRef, bytes: 32 * 1024 * 1024 + 1 }, false],
    ['non-boolean has_alpha rejected', { ...goodRef, has_alpha: 1 }, false],
    ['extra key rejected', { ...goodRef, path: '/tmp/x' }, false],
    ['null rejected', null, false],
  ];
  for (const [label, doc, want] of refCases) {
    const got = await evaluate(
      `window.QwenStudio.parseReferenceResponse(${JSON.stringify(doc)}) !== null`);
    check(`parseReferenceResponse: ${label}`, got === want, `got ${got}`);
  }
  /* ---- 5c. the ordering contract end to end: two mocked uploads, then
   * reorder/remove, with the envelope and the visible numbering following
   * the DISPLAYED order (the ids are 64-hex, so they never collide) ---- */
  const uploadInfo = await evaluate(`(async function () {
    var seen = [];
    var realFetch = window.fetch;   // the spying wrapper: RESTORED below
    window.fetch = function (url, opts) {
      seen.push({ url: String(url), method: opts.method,
                  ctype: opts.headers['Content-Type'],
                  ui: opts.headers['X-Qwen-UI'],
                  isBytes: opts.body instanceof Uint8Array,
                  isString: typeof opts.body === 'string',
                  hasBtoa: typeof window.btoa === 'function' });
      var id = 'c'.repeat(63) + (seen.length === 1 ? '1' : '2');
      var doc = { schema: 'qwen_engine.web_reference.v1', id: id,
                  width: 640, height: 480, has_alpha: false, bytes: 40 };
      return Promise.resolve(new Response(JSON.stringify(doc),
        { status: 201, headers: { 'Content-Type': 'application/json' } }));
    };
    // Uploads are status-gated; the checks have no server, so the gate is
    // opened through the test-only knob (production only ever opens it from
    // a RECOGNIZED status document).
    window.QwenStudio.setApiAvailableForChecks(true);
    var files = [new File([new Uint8Array([1, 2, 3, 4])], 'a.png', { type: 'image/png' }),
                 new File([new Uint8Array([5, 6, 7, 8])], 'b.png', { type: 'image/png' })];
    var added = await window.QwenStudio.addReferenceFiles(files);
    var orderAfterAdd = window.QwenStudio.referenceIds().join(',');
    var cards = document.getElementById('reference-list').children;
    var ordinals = [];
    for (var i = 0; i < cards.length; i++) {
      ordinals.push(cards[i].querySelector('.reference-ordinal').textContent);
      ordinals.push(cards[i].querySelector('img').getAttribute('src'));
    }
    var envAfterAdd = window.QwenStudio.buildEnvelope().envelope.config.references.join(',');
    window.QwenStudio.moveReference(1, -1);           // swap 1 and 2
    var orderAfterMove = window.QwenStudio.referenceIds().join(',');
    var envAfterMove = window.QwenStudio.buildEnvelope().envelope.config.references.join(',');
    window.QwenStudio.removeReference(0);             // drop the first
    window.fetch = realFetch;      // restore the spy for the later checks
    window.QwenStudio.setApiAvailableForChecks(false);   // back to disconnected
    return JSON.stringify({ added: added, seen: seen, firstOrder: orderAfterAdd,
                            ordinals: ordinals, envAfterAdd: envAfterAdd,
                            movedOrder: orderAfterMove, envAfterMove: envAfterMove,
                            afterRemove: window.QwenStudio.referenceIds().join(','),
                            envAfterRemove:
                              window.QwenStudio.buildEnvelope().envelope.config.references.join(','),
                            countLabel: document.getElementById('reference-count').textContent });
  })()`);
  const up = JSON.parse(uploadInfo);
  check('two mocked uploads both land', up.added === 2 && up.seen.length === 2, uploadInfo);
  check('uploads go to /api/v1/references as raw octet-stream bytes',
        up.seen.every((r) => r.url === '/api/v1/references' && r.method === 'POST' &&
          r.ctype === 'application/octet-stream' && r.ui === '1' && r.isBytes && !r.isString),
        JSON.stringify(up.seen));
  check('thumbnails are numbered 1..N with server-owned thumb URLs',
        up.ordinals.join('|') === 'Image 1|/api/v1/references/' + 'c'.repeat(63) + '1/thumb' +
        '|Image 2|/api/v1/references/' + 'c'.repeat(63) + '2/thumb', up.ordinals.join('|'));
  check('envelope order matches the displayed order',
        up.envAfterAdd === up.firstOrder && up.envAfterAdd.split(',').length === 2, up.envAfterAdd);
  check('move up reorders BOTH the list and the envelope',
        up.movedOrder === up.firstOrder.split(',').reverse().join(',') &&
        up.envAfterMove === up.movedOrder, up.movedOrder);
  check('remove drops the id from the envelope and renumbers the rest',
        up.afterRemove === up.movedOrder.split(',')[1] && up.envAfterRemove === up.afterRemove,
        up.envAfterRemove);
  check('the visible count follows the hard cap',
        up.countLabel === '1 of 10 references', up.countLabel);


  /* ---- 6. presets and manual dimension matching ----
   * The preset table gained 2K tiers since these assertions were written,
   * so the expected value is ONE pressed button with a KNOWN name rather
   * than a hardcoded three-button string (which was stale, not a bug). */
  const pressed = () => evaluate(`(function () {
    var all = Array.prototype.slice.call(document.querySelectorAll('.preset-btn'));
    var on = all.filter(function (b) { return b.getAttribute('aria-pressed') === 'true'; });
    return on.map(function (b) { return b.dataset.preset; }).join(',') +
           '|' + all.length;
  })()`);
  const pressedIs = async (name) => {
    const got = await pressed();
    const [names, total] = got.split('|');
    return names === name && Number(total) >= 3;
  };
  await evaluate('document.querySelector(\'[data-preset="landscape"]\').click()');
  check('preset click sets 1344x768',
        await evaluate('document.getElementById("width").value + "x" + document.getElementById("height").value') === '1344x768');
  check('preset click presses ONLY landscape', await pressedIs('landscape'), await pressed());
  const setDims = async (w, h) => {
    await evaluate(`(function () {
      var wid = document.getElementById('width'), hei = document.getElementById('height');
      wid.value = '${w}'; hei.value = '${h}';
      wid.dispatchEvent(new Event('input')); hei.dispatchEvent(new Event('input'));
    })()`);
  };
  await setDims('1344', '768');
  check('manual exact 1344x768 re-presses landscape', await pressedIs('landscape'),
        await pressed());
  await setDims('1024', '1024');
  check('manual exact 1024x1024 re-presses square', await pressedIs('square'),
        await pressed());
  await setDims('1056', '608');
  check('non-preset dimensions clear ALL presets', await pressedIs(''), await pressed());

  /* ---- 7. preview true ratio (with the height-cap max-width rule) ---- */
  await evaluate('document.querySelector(\'[data-preset="landscape"]\').click()');
  const ratio = await evaluate(`(function () {
    var f = document.getElementById('preview-frame');
    return [f.style.getPropertyValue('--ar-w'), f.style.getPropertyValue('--ar-h'),
            getComputedStyle(f).aspectRatio,
            getComputedStyle(f).maxWidth].join('|');
  })()`);
  const [arW, arH, computedAR] = ratio.split('|');
  check('preview frame carries exact 1344/768 ratio vars', arW === '1344' && arH === '768', ratio);
  check('computed aspect-ratio is 1344 / 768', computedAR.replace(/\s/g, '') === '1344/768', computedAR);
  check('max-width ratio cap present in computed style',
        ratio.split('|')[3] !== 'none', ratio);

  /* ---- 8. hash nav + heading focus + #main + back/forward ---- */
  await evaluate('document.getElementById("nav-history").click()');
  await delay(150);
  check('History view shown after nav click',
        await evaluate('document.getElementById("view-history").hidden') === false &&
        await evaluate('document.getElementById("view-create").hidden') === true);
  check('History heading focused on user nav',
        await evaluate('document.activeElement.id') === 'history-heading');
  check('aria-current follows the view',
        await evaluate('document.getElementById("nav-history").getAttribute("aria-current")') === 'page');
  await evaluate('document.querySelector(".skip-link").click()');
  await delay(150);
  check('skip-link (#main) does NOT reroute History back to Create',
        await evaluate('document.getElementById("view-history").hidden') === false &&
        await evaluate('location.hash') === '#main');
  await evaluate('history.back()');
  await delay(250);
  check('back restores the History view',
        await evaluate('location.hash') === '#history' &&
        await evaluate('document.getElementById("view-history").hidden') === false);
  await evaluate('document.getElementById("nav-create").click()');
  await delay(150);
  check('Create heading focused on user nav',
        await evaluate('document.activeElement.id') === 'create-heading' &&
        await evaluate('document.getElementById("view-create").hidden') === false);

  /* ---- 8b. back from History on a hash-less start restores Create ---- */
  await s('Page.navigate', { url: pageUrl });   // fresh, no hash
  await delay(500);
  check('fresh load shows Create without focusing anything',
        await evaluate('document.getElementById("view-create").hidden') === false &&
        await evaluate('document.activeElement.tagName') === 'BODY');
  await evaluate('document.getElementById("nav-history").click()');
  await delay(150);
  await evaluate('history.back()');
  await delay(250);
  check('back to hash-less URL restores Create (empty hash routes to Create)',
        await evaluate('location.hash') === '' &&
        await evaluate('document.getElementById("view-create").hidden') === false &&
        await evaluate('document.getElementById("view-history").hidden') === true);

  /* ---- 9. aria-invalid wiring ---- */
  const setVal = async (id, v) => evaluate(`(function () {
    var el = document.getElementById('${id}'); el.value = '${v}';
    el.dispatchEvent(new Event('input')); return true; })()`);
  await setVal('steps', '0');
  check('steps=0 -> aria-invalid true + error text',
        await evaluate('document.getElementById("steps").getAttribute("aria-invalid")') === 'true' &&
        (await evaluate('document.getElementById("steps-error").textContent')).length > 0);
  await setVal('steps', '40');
  check('steps restored -> aria-invalid false',
        await evaluate('document.getElementById("steps").getAttribute("aria-invalid")') === 'false');
  await setVal('width', '33');
  check('bad width marks BOTH dimension controls aria-invalid',
        await evaluate('document.getElementById("width").getAttribute("aria-invalid")') === 'true' &&
        await evaluate('document.getElementById("height").getAttribute("aria-invalid")') === 'true');
  await setVal('width', '1056');
  await setVal('height', '608');
  check('valid dims clear aria-invalid on both',
        await evaluate('document.getElementById("width").getAttribute("aria-invalid")') === 'false' &&
        await evaluate('document.getElementById("height").getAttribute("aria-invalid")') === 'false');
  await setVal('seed', '123456789012345678901');
  check('21-digit seed -> aria-invalid true',
        await evaluate('document.getElementById("seed").getAttribute("aria-invalid")') === 'true');
  await setVal('seed', '0');

  /* ---- 9b. over-length seed entry: NO silent truncation ----
   * Input.insertText works WITHOUT an enable call — the Input domain has no
   * Input.enable method (requesting it would fail the whole session). */
  await evaluate(`(function () {
    var el = document.getElementById('seed');
    el.value = ''; el.focus(); return true; })()`);
  await s('Input.insertText', { text: '100000000000000000000' }); // 21 digits, paste-like
  const longSeed = await evaluate(`[
      document.getElementById('seed').value,
      document.getElementById('seed').getAttribute('aria-invalid'),
      document.getElementById('seed-error').textContent
    ].join('||')`);
  const [rawVal, rawInvalid, rawErr] = longSeed.split('||');
  check('21-digit entry keeps the ORIGINAL text (no maxlength truncation)',
        rawVal === '100000000000000000000', `got ${rawVal}`);
  check('21-digit entry is marked invalid with explanation',
        rawInvalid === 'true' && /too long|at most 20/i.test(rawErr), rawErr);
  await setVal('seed', '0');

  /* ---- 9c. prompt required (mirrors config.py) but never alerts on load ---- */
  check('no prompt error on fresh load (no first-paint alert)',
        (await evaluate('document.getElementById("prompt-error").textContent')) === '' &&
        await evaluate('document.getElementById("prompt").getAttribute("aria-invalid")') === 'false');
  await setVal('prompt', '   ');
  check('whitespace-only prompt -> accessible error after editing',
        (await evaluate('document.getElementById("prompt-error").textContent')).length > 0 &&
        await evaluate('document.getElementById("prompt").getAttribute("aria-invalid")') === 'true');
  await setVal('prompt', 'a red fox');
  check('non-empty prompt clears the prompt error',
        (await evaluate('document.getElementById("prompt-error").textContent')) === '' &&
        await evaluate('document.getElementById("prompt").getAttribute("aria-invalid")') === 'false');

  /* ---- 10. always disabled: submit refuses, no pretend success ---- */
  await setVal('prompt', 'a red fox');
  const fetchesBefore = await evaluate('window.__fetchCalls.length');
  await evaluate('document.getElementById("generate-btn").click()');
  await evaluate(`document.getElementById('create-form').dispatchEvent(new Event('submit', {cancelable:true}))`);
  await delay(120);
  const afterSubmit = await evaluate(`[
      document.getElementById('generate-btn').disabled,
      document.getElementById('generate-reason').textContent,
      document.getElementById('job-feedback').textContent,
      window.__fetchCalls.join(",")
    ].join('||')`);
  const [stillDisabled, submitMsg, submitFeedback, spyCalls] = afterSubmit.split('||');
  check('Generate still disabled after submit attempt', stillDisabled === 'true');
  // W3: the static reason names the missing switch; the REFUSAL of an
  // actual submit attempt is reported on the live job-feedback line.
  check('disabled reason names the missing generation switch',
        /--enable-generation/.test(submitMsg), submitMsg);
  check('submit handler refuses honestly (Nothing was submitted)',
        /Nothing was submitted/.test(submitFeedback), submitFeedback);
  check('submit attempt triggers no new fetches (status check only, ever)',
        spyCalls === '' || spyCalls === '/api/v1/status',
        `calls=[${spyCalls}] before=${fetchesBefore}`);

  /* ---- 10b. the hard cap at 10: every card rendered, Add disabled, the
   * count honest, and the 11th file refused WITHOUT a request ---- */
  const capped = await evaluate(`(async function () {
    var realFetch = window.fetch;
    var calls = 0;
    window.fetch = function () {
      calls++;
      var id = 'd'.repeat(63) + (calls % 10);
      return Promise.resolve(new Response(JSON.stringify({
        schema: 'qwen_engine.web_reference.v1', id: id,
        width: 800, height: 600, has_alpha: true, bytes: 2048 }),
        { status: 201, headers: { 'Content-Type': 'application/json' } }));
    };
    window.QwenStudio.setApiAvailableForChecks(true);
    var files = [];
    for (var i = 0; i < 10; i++) {
      files.push(new File([new Uint8Array([i, i, i])], 'r' + i + '.png',
                          { type: 'image/png' }));
    }
    await window.QwenStudio.addReferenceFiles(files);
    var afterTen = window.QwenStudio.referenceIds().length;
    var addDisabled = document.getElementById('add-reference').disabled;
    var countText = document.getElementById('reference-count').textContent;
    var lastOrdinal = document.getElementById('reference-list')
      .lastElementChild.querySelector('.reference-ordinal').textContent;
    var before = calls;
    // The 11th is refused up front: capacity is checked BEFORE bytes.
    await window.QwenStudio.addReferenceFiles(
      [new File([new Uint8Array([9])], 'extra.png', { type: 'image/png' })]);
    var refusedCalls = calls - before;
    var capMsg = document.getElementById('reference-status').textContent;
    var envLen = window.QwenStudio.buildEnvelope().envelope.config.references.length;
    // Every per-card action must be ENABLED once the batch finishes: the
    // cards are rendered while the upload flag is set, so a batch end that
    // only refreshed the toolbar left Remove permanently disabled (found by
    // screenshot).
    var cards = Array.prototype.slice.call(
      document.querySelectorAll('#reference-list > li'));
    var disabledAfter = 0;
    var disabledTotal = 0;
    var removeEnabled = [];
    var cardCount = cards.length;
    cards.forEach(function (card, idx) {
      Array.prototype.slice.call(card.querySelectorAll('button')).forEach(function (b) {
        var action = b.dataset.action;
        if (action === 'remove') removeEnabled.push(!b.disabled);
        if (!b.disabled) return;
        disabledTotal++;
        // Only the honest END-of-list disables are allowed.
        var atTop = action === 'up' && idx === 0;
        var atBottom = action === 'down' && idx === cardCount - 1;
        if (!atTop && !atBottom) disabledAfter++;
      });
    });
    window.fetch = realFetch;
    window.QwenStudio.setApiAvailableForChecks(false);
    return JSON.stringify({ afterTen: afterTen, addDisabled: addDisabled,
                            countText: countText, lastOrdinal: lastOrdinal,
                            refusedCalls: refusedCalls, capMsg: capMsg, envLen: envLen,
                            disabledAfter: disabledAfter,
                            disabledTotal: disabledTotal,
                            removeEnabled: removeEnabled.join(','),
                            cardCount: cardCount });
  })()`);
  const cap = JSON.parse(capped);
  check('ten uploads fill the list to the hard cap',
        cap.afterTen === 10 && cap.envLen === 10, capped);
  check('Add is disabled at the cap', cap.addDisabled === true);
  check('the count reads 10 of 10', cap.countText === '10 of 10 references',
        cap.countText);
  check('the last card is numbered Image 10', cap.lastOrdinal === 'Image 10',
        cap.lastOrdinal);
  check('an 11th file is refused with NO request sent', cap.refusedCalls === 0,
        String(cap.refusedCalls));
  check('the cap refusal states the limit instead of dropping the file',
        /At most 10 reference images/.test(cap.capMsg), cap.capMsg);
  check('Remove is ENABLED on every card after the batch finishes',
        cap.removeEnabled === Array(10).fill('true').join(','),
        `${cap.cardCount} cards: ${cap.removeEnabled}`);
  // Exactly two disabled buttons exist across 10 cards: Up on the first and
  // Down on the last. Anything else is a stuck control.
  check('exactly the two list-end buttons are disabled',
        cap.disabledAfter === 0 && cap.disabledTotal === 2,
        `dishonest=${cap.disabledAfter} total=${cap.disabledTotal}`);

  /* ---- 10c/10d. a failed thumbnail: stated per card, honest about what
   * was actually observed, and never leaking the render loop's `var` ----
   * This page is file://, so thumbnail requests cannot succeed at all; the
   * point of these checks is that EACH card narrates for ITSELF. With the
   * shared-`var` bug every write named the last card's ordinal. ---- */
  const thumbs = JSON.parse(await evaluate(`(async function () {
    while (window.QwenStudio.referenceIds().length) window.QwenStudio.removeReference(0);
    var writes = [];
    var status = document.getElementById('reference-status');
    // setText replaces textContent, so the ADDED TEXT NODE carries that
    // write's own value. Reading status.textContent in the callback would
    // give the FINAL text for all three (batching), hiding a scope leak.
    var seen = new MutationObserver(function (records) {
      records.forEach(function (m) {
        m.addedNodes.forEach(function (node) {
          writes.push(node.textContent || '');
        });
      });
    });
    seen.observe(status, { childList: true, characterData: true, subtree: true });
    var realFetch = window.fetch;
    var n = 0;
    window.fetch = function () {
      var i = n++;
      return Promise.resolve(new Response(JSON.stringify({
        schema: 'qwen_engine.web_reference.v1',
        id: String(i + 3).padStart(64, '0'),
        width: 512, height: 512, has_alpha: false, bytes: 1024 }),
        { status: 201, headers: { 'Content-Type': 'application/json' } }));
    };
    window.QwenStudio.setApiAvailableForChecks(true);
    var files = [];
    for (var j = 0; j < 3; j++) {
      files.push(new File([new Uint8Array([j])], 't' + j + '.png',
                          { type: 'image/png' }));
    }
    await window.QwenStudio.addReferenceFiles(files);
    window.fetch = realFetch;
    await new Promise(function (r) { setTimeout(r, 500); });
    seen.disconnect();
    var cards = Array.prototype.slice.call(
      document.querySelectorAll('#reference-list > li'));
    return JSON.stringify({
      total: cards.length,
      ids: window.QwenStudio.referenceIds().length,
      marked: cards.map(function (c) {
        return c.classList.contains('reference-missing');
      }),
      ordinals: cards.map(function (c) {
        return c.querySelector('.reference-ordinal').textContent;
      }),
      metaHonest: cards.every(function (c) {
        return /preview could not be loaded/.test(
          c.querySelector('.reference-meta').textContent);
      }),
      noUnprovenClaim: cards.every(function (c) {
        return !/no longer holds this image/.test(
          c.querySelector('.reference-meta').textContent);
      }),
      // Token 5 is the ordinal in "The preview for reference image 6 …"
      // (no regex escaping inside this template literal).
      named: writes.filter(function (t) {
        return /preview for reference image/.test(t);
      }).map(function (t) {
        return t.split(' ')[5];
      })
    });
  })()`));
  check('all three cards stay in the list with their order',
        thumbs.total === 3 && thumbs.ids === 3 &&
        JSON.stringify(thumbs.ordinals) ===
          JSON.stringify(['Image 1', 'Image 2', 'Image 3']),
        JSON.stringify(thumbs));
  check('every card whose thumbnail failed is marked',
        JSON.stringify(thumbs.marked) === '[true,true,true]', JSON.stringify(thumbs));
  check('EACH card narrates its OWN ordinal (no shared-var scope leak)',
        JSON.stringify(thumbs.named.slice().sort()) === JSON.stringify(['1', '2', '3']),
        JSON.stringify(thumbs.named));
  check('a failed thumbnail does not claim the store lost the image',
        thumbs.noUnprovenClaim === true && thumbs.metaHonest === true,
        JSON.stringify(thumbs));

  /* ---- 10b. rewrite flow: gate, strict answer, stale discard ---- */
  // Without a status document the button stays hidden AND disabled: an
  // unavailable rewriter must not look like a broken one.
  check('rewrite button hidden and disabled with no rewriter',
        await evaluate(`(function(){var b=document.getElementById('rewrite-btn');
          return b.hidden===true && b.disabled===true;})()`) === true);
  check('rewrite hint explains the rewriter is unavailable',
        await evaluate(`(function(){var h=document.getElementById('rewrite-hint');
          return h.hidden===false && /unavailable/.test(h.textContent);})()`) === true);
  await setVal('prompt', 'a red fox in the snow');
  const revBefore = await evaluate('window.QwenStudio.rewriteState().revision');
  await setVal('prompt', 'a red fox in deep snow');
  // A width change must NOT invalidate a prompt suggestion: the prompt text
  // is what the revision tracks.
  const revAfterWidth = await evaluate(`(function(){
    var before = window.QwenStudio.rewriteState().revision;
    var el = document.getElementById('width');
    el.value = '1152'; el.dispatchEvent(new Event('input'));
    return window.QwenStudio.rewriteState().revision === before;})()`);
  check('a resolution change leaves the prompt revision alone',
        revAfterWidth === true);
  const revAfter = await evaluate('window.QwenStudio.rewriteState().revision');
  check('typing bumps the draft revision exactly once',
        revAfter === revBefore + 1, `before=${revBefore} after=${revAfter}`);

  // Enable the rewriter the way production does — from a parsed status block.
  // Exactly the four server keys, exactly the available pair.
  const parsed = await evaluate(`(function(){
    var block = {state:'available', code:'rewrite_available',
                 backend:'local-llm', reason:''};
    return JSON.stringify(window.QwenStudio.parseRewriteBlock(block));})()`);
  check('status block parses to an available rewriter',
        /local-llm/.test(parsed), parsed);
  await evaluate(`window.QwenStudio.setRewriterState(${parsed})`);
  check('rewrite button becomes visible and enabled',
        await evaluate(`(function(){var b=document.getElementById('rewrite-btn');
          return b.hidden===false && b.disabled===false;})()`) === true);
  check('rewrite hint hides once available',
        await evaluate(`document.getElementById('rewrite-hint').hidden`) === true);

  // Refusing to rewrite an empty draft sends nothing (checked while the
  // rewriter IS available, so the refusal is the empty-draft guard and not
  // the missing-rewriter early return).
  const draftBefore = await evaluate(`document.getElementById('prompt').value`);
  await setVal('prompt', '');
  const refuse = await evaluate(`(function(){
    window.QwenStudio.requestRewrite();
    return [document.getElementById('rewrite-error').textContent,
            window.QwenStudio.rewriteState().inFlight];})()`);
  check('empty draft is refused locally and nothing is sent',
        /Write a draft first/.test(refuse[0]) && refuse[1] === false,
        JSON.stringify(refuse));
  await setVal('prompt', draftBefore);

  // Strictness of the answer document: each mutation must be REJECTED.
  const base = {schema: 'qwen_engine.web_rewrite.v1', task: 't2i', draft_revision: 3,
    original_prompt: 'a', rewritten_prompt: 'b', parse_ok: true,
    backend: 'local-llm', elapsed_ms: 12, ratio: null,
    ratio_conflict: false, unparsed_answer: null};
  const mutations = [
    ['schema', 'x'], ['task', 'video'], ['draft_revision', -1],
    ['original_prompt', 5], ['rewritten_prompt', null],
    ['parse_ok', 'yes'], ['backend', ''], ['elapsed_ms', 1.5],
    ['ratio', {kind: 'wh_ratio', value: '16x9'}],
    ['ratio', {kind: 'ratio_follow', image: 0}],
    ['ratio', {kind: 'ratio_follow', image: 99}],
    ['ratio_conflict', 'no'], ['unparsed_answer', 7],
    ['parse_ok', true], ['rewritten_prompt', '   '],
  ];
  const rejected = await evaluate(`(function(){
    var base=${JSON.stringify(base)}; var bad=0;
    ${JSON.stringify(mutations)}.forEach(function(m){
      var doc=JSON.parse(JSON.stringify(base));
      if (m[0]==='parse_ok' && m[1]===true) { doc.parse_ok=true; doc.rewritten_prompt='   '; }
      else { doc[m[0]]=m[1]; }
      if (window.QwenStudio.parseRewriteAnswer(doc)!==null) bad++;
    });
    return bad;})()`);
  check('every malformed answer document is rejected', rejected === 0,
        `${rejected} accepted`);
  // An UNPARSABLE model answer (parse_ok false) is diagnostic text, never a
  // pending suggestion: a forced apply must not blank the draft.
  const unparsable = await evaluate(`(function(){
    var before = document.getElementById('prompt').value;
    window.QwenStudio.showProposalRawForChecks({schema:'qwen_engine.web_rewrite.v1',
      task:'t2i', draft_revision: 0, original_prompt: before,
      rewritten_prompt:'', parse_ok:false, backend:'local-llm',
      elapsed_ms: 1, ratio:null, ratio_conflict:false,
      unparsed_answer:'{"not":"a proposal"}'});
    var stateAfterShow = window.QwenStudio.rewriteState().hasProposal;
    window.QwenStudio.applyProposal();
    window.QwenStudio.editProposal();
    return {before: before,
            after: document.getElementById('prompt').value,
            stateAfterShow: stateAfterShow,
            applyHidden: document.getElementById('rewrite-apply').hidden,
            editHidden: document.getElementById('rewrite-edit').hidden,
            err: document.getElementById('rewrite-error').textContent,
            shown: document.getElementById('rewrite-suggested').textContent};})()`);
  check('an unparsable answer is never a pending proposal',
        unparsable.stateAfterShow === false, JSON.stringify(unparsable.stateAfterShow));
  check('an unparsable answer withholds Apply and Edit',
        unparsable.applyHidden === true && unparsable.editHidden === true,
        JSON.stringify([unparsable.applyHidden, unparsable.editHidden]));
  check('a forced apply of an unparsable answer cannot blank the draft',
        unparsable.after === unparsable.before && unparsable.after !== '',
        JSON.stringify([unparsable.before, unparsable.after]));
  check('the unparsable answer is kept as diagnostic text',
        /not.*a proposal/.test(unparsable.shown) && /usable prompt/.test(unparsable.err),
        JSON.stringify(unparsable.shown));

  check('absent rewrite block parses to null, malformed to a refusal',
        await evaluate(`(function(){
          var a = window.QwenStudio.parseRewriteBlock(undefined) === null;
          var b = window.QwenStudio.parseRewriteBlock(
            {state:'unavailable', code:'rewrite_disabled', backend:null,
             reason:'off'});
          var c = window.QwenStudio.parseRewriteBlock(
            {state:'available', code:'rewrite_available', backend:'',
             reason:''});
          var d = window.QwenStudio.parseRewriteBlock(
            {state:'available', code:'rewrite_available', backend:'x',
             reason:'', extra:1});
          // b: legitimately unavailable -> null. c/d: malformed -> a refusal
          // object, never null, so the caller rejects the whole status.
          return a && b === null && c !== null && d !== null;})()`) === true);

  // Render a real proposal and drive Apply.
  const shown = await evaluate(`(function(){
    var doc=${JSON.stringify(base)};
    doc.draft_revision = window.QwenStudio.rewriteState().revision;
    doc.original_prompt = document.getElementById('prompt').value;
    doc.rewritten_prompt = 'a red fox, deep snow, low sun';
    doc.ratio = {kind:'wh_ratio', value:'16:9'};
    var ok = window.QwenStudio.showProposalForChecks(doc);
    var p = document.getElementById('rewrite-proposal');
    return [ok, p.hidden, document.getElementById('rewrite-suggested').textContent,
            document.getElementById('rewrite-ratio').textContent,
            document.getElementById('rewrite-apply-ratio').hidden,
            document.getElementById('rewrite-original').textContent];
  })()`);
  check('a valid answer renders a visible proposal with both texts',
        shown[0] === true && shown[1] === false &&
        shown[2] === 'a red fox, deep snow, low sun' &&
        shown[5] === 'a red fox in deep snow', JSON.stringify(shown));
  check('suggestion is NOT applied while the proposal is shown',
        await evaluate(`document.getElementById('prompt').value`) === 'a red fox in deep snow');
  check('16:9 maps to an exact preset, so the ratio button is offered',
        shown[4] === false && /16:9/.test(shown[3]), JSON.stringify(shown[3]));
  // The button names a REAL preset, and one whose ratio is exactly 16:9:
  // never an invented name, never a nearest match.
  const ratioPreset = await evaluate(`(function(){
    var name = document.getElementById('rewrite-apply-ratio').dataset.preset;
    var btn = document.querySelector('.preset-btn[data-preset="' + name + '"]');
    if (!name || !btn) return 'missing:' + name;
    btn.click();
    return [name, document.getElementById('width').value + 'x' +
            document.getElementById('height').value].join(' ');})()`);
  check('the ratio button names a real 16:9 preset and applies it',
        ratioPreset === 'landscape 1344x768' || ratioPreset === 'landscape2k 2752x1536',
        ratioPreset);
  const applied = await evaluate(`(function(){
    var before = window.QwenStudio.rewriteState().revision;
    document.getElementById('rewrite-apply').click();
    return [document.getElementById('prompt').value,
            window.QwenStudio.rewriteState().revision > before,
            document.getElementById('rewrite-proposal').hidden,
            document.activeElement.id];})()`);
  check('Apply replaces the draft, bumps the revision, hides the block, refocuses',
        applied[0] === 'a red fox, deep snow, low sun' && applied[1] === true &&
        applied[2] === true && applied[3] === 'prompt', JSON.stringify(applied));

  // Escape abandons a proposal without touching the draft.
  const escaped = await evaluate(`(function(){
    var doc=${JSON.stringify(base)};
    doc.draft_revision = window.QwenStudio.rewriteState().revision;
    doc.original_prompt = document.getElementById('prompt').value;
    doc.rewritten_prompt = 'something else entirely';
    window.QwenStudio.showProposalForChecks(doc);
    var p = document.getElementById('rewrite-proposal');
    p.dispatchEvent(new KeyboardEvent('keydown', {key:'Escape', bubbles:true}));
    return [p.hidden, document.getElementById('prompt').value,
            window.QwenStudio.rewriteState().hasProposal];})()`);
  check('Escape closes the proposal and keeps the draft intact',
        escaped[0] === true && escaped[1] === 'a red fox, deep snow, low sun' &&
        escaped[2] === false, JSON.stringify(escaped));

  // A draft edit stales any visible proposal.
  const staled = await evaluate(`(function(){
    var doc=${JSON.stringify(base)};
    doc.draft_revision = window.QwenStudio.rewriteState().revision;
    doc.original_prompt = document.getElementById('prompt').value;
    doc.rewritten_prompt = 'x';
    window.QwenStudio.showProposalForChecks(doc);
    var el=document.getElementById('prompt');
    el.value='typed over it'; el.dispatchEvent(new Event('input'));
    return document.getElementById('rewrite-proposal').hidden;})()`);
  check('editing the draft while a proposal is shown closes it',
        staled === true);
  await evaluate(`window.QwenStudio.setRewriterState(null)`);
  check('losing the rewriter hides the button again',
        await evaluate(`document.getElementById('rewrite-btn').hidden`) === true);
  await setVal('prompt', 'a red fox in deep snow');

  /* ---- 11. overflow at 1440 / 375 / 320 ---- */
  for (const width of [1440, 375, 320]) {
    await s('Emulation.setDeviceMetricsOverride',
      { width, height: 850, deviceScaleFactor: 1, mobile: width < 500 });
    await delay(120);
    const res = await evaluate(`(function () {
      var over = [];
      var els = document.querySelectorAll('body *');
      for (var i = 0; i < els.length; i++) {
        var el = els[i];
        if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') continue;
        var r = el.getBoundingClientRect();
        if (r.width > 0 && r.right > window.innerWidth + 1) over.push(el.id || el.className || el.tagName);
      }
      var seed = document.getElementById('seed');
      return [document.documentElement.scrollWidth, window.innerWidth,
              over.slice(0, 3).join(','), seed.clientWidth].join('|');
    })()`);
    const [scrollW, innerW, over, seedW] = res.split('|');
    // innerWidth is itself the symptom when the layout viewport widens:
    // a 320px emulation that reports inner=352 means real overflow.
    check(`no horizontal overflow at ${width}px`,
          Number(scrollW) <= Number(innerW) + 1 && over === '' &&
          Number(innerW) <= width + 1,
          `scrollW=${scrollW} inner=${innerW} over=[${over}]`);
    check(`seed input keeps usable width at ${width}px`, Number(seedW) >= 90,
          `seed input ${seedW}px`);
  }
  // full 20-digit inspection at the narrow widths
  await s('Emulation.setDeviceMetricsOverride', { width: 320, height: 850, deviceScaleFactor: 1, mobile: true });
  await delay(120);
  await setVal('seed', '18446744073709551615');
  check('full 20-digit max seed parses at 320px layout',
        await evaluate('String(window.QwenStudio.parseSeed(document.getElementById("seed").value))') === '18446744073709551615');
  check('seed row shows no horizontal overflow with max seed at 320px',
        await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1') === true);
}

const guard = watchdog(WHOLE_CHECK_MS, 'whole browser check');
try {
  await Promise.race([main(), guard]);
} catch (err) {
  check('script completed without error', false, String(err && err.stack || err));
} finally {
  guard.cancel();
  console.log(results.join('\n'));
  console.log(`\n${results.length - failed}/${results.length} browser checks passed`);
  const cleaned = await shutdownChrome(chrome, ws, cdp);
  if (cleaned === false) {
    failed++;
    console.error('browser cleanup pending: retained owner and profile; no false successful exit');
  } else {
    try { rmSync(profileDir, { recursive: true, force: true, maxRetries: 5 }); } catch {}
  }
}
process.exitCode = failed === 0 ? 0 : 1; // retain any pending owner rather than orphaning it
