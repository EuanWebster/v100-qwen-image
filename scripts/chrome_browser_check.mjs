#!/usr/bin/env node
/* Real-Chrome behavioural check for the W1 static web workspace.
 *
 * NOT part of the guarded main suite (`run_cpu_tests.sh` stays static-only,
 * ML-free and socket-free). Run it directly, and only when a Chrome binary
 * is available:
 *
 *     CHROME_PATH=/path/to/chrome node scripts/chrome_browser_check.mjs
 *
 * Dependencies: none. Uses the existing Chrome binary plus Node built-ins
 * only (node:child_process, node:fs, node:os, node:path, and the global
 * WebSocket for the DevTools protocol). It launches its own Chrome child
 * with a throwaway profile under the system temp dir and always cleans
 * both up. No screenshots are written or committed.
 *
 * Covers: seed edges/length/non-ASCII, snapshot keys, presets and manual
 * dimension matching, preview ratio, hash nav + heading focus (and the
 * #main skip-link case), aria-invalid wiring, Generate ALWAYS disabled with
 * zero HTTP probes, and horizontal overflow at 1440/375/320.
 */
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

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

// Honest preflight on cgroup-limited machines: spawning Chrome needs a batch
// of free task slots (cgroup pids controller). Refuse EARLY with a clear
// message instead of hanging on pthread_create EAGAIN.
try {
  const { readFileSync } = await import('node:fs');
  const max = Number(readFileSync('/sys/fs/cgroup/pids.max', 'utf8').trim());
  const cur = Number(readFileSync('/sys/fs/cgroup/pids.current', 'utf8').trim());
  if (Number.isFinite(max) && Number.isFinite(cur) && max - cur < 64) {
    console.error(`chrome_browser_check: only ${max - cur} free task slots ` +
                  `(${cur}/${max} in the cgroup pids controller) — Chrome ` +
                  'cannot start. Unreaped zombie processes are leaking the ' +
                  'limit; this is an environment problem, not a product ' +
                  'failure. Clean the sandbox (or run on a fresh box).');
    process.exit(3);
  }
} catch { /* not a cgroup v2 box we can read — proceed and let Chrome speak */ }

const profileDir = mkdtempSync(join(tmpdir(), 'qwen-chrome-check-'));

let results = [];
let chrome = null;
let ws = null;
let failed = 0;

function check(name, ok, detail = '') {
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  [' + detail + ']' : ''}`);
  if (!ok) failed++;
}

function delay(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function main() {
  chrome = spawn(CHROME, [
    '--remote-debugging-port=0',
    `--user-data-dir=${profileDir}`,
    '--no-first-run', '--no-default-browser-check', '--no-sandbox',
    '--disable-gpu', '--disable-dev-shm-usage', '--disable-crash-reporter',
    '--disable-breakpad', '--disable-background-networking', '--disable-sync',
    '--no-zygote', '--renderer-process-limit=1', '--num-raster-threads=1',
    '--disable-features=AudioServiceOutOfProcess', '--mute-audio',
    'about:blank',
  ], { stdio: ['ignore', 'pipe', 'pipe'], detached: true });
  const wsUrl = await new Promise((res, rej) => {
    const t = setTimeout(() => rej(new Error('chrome did not announce a DevTools port within 45 s')), 45000);
    const onData = (buf, sink) => {
      const m = /DevTools listening on (ws:\/\/\S+)/.exec(sink.toString());
      if (m) { clearTimeout(t); res(m[1]); }
    };
    let errBuf = '';
    chrome.stderr.on('data', (b) => { errBuf += b; onData(0, Buffer.from(errBuf)); });
    chrome.on('error', (e) => { clearTimeout(t); rej(e); });
  });

  ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  let nextId = 1;
  const pending = new Map();
  const netRequests = [];
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve: r, reject: j } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? j(new Error(JSON.stringify(msg.error))) : r(msg.result);
    } else if (msg.method === 'Network.requestWillBeSent') {
      netRequests.push(msg.params.request.type + ' ' + msg.params.request.url);
    }
  };
  const send = (method, params = {}, sessionId) => new Promise((resolveP, rejectP) => {
    const id = nextId++;
    pending.set(id, { resolve: resolveP, reject: rejectP });
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });

  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  const s = (method, params) => send(method, params || {}, sessionId);

  // fetch spy installed BEFORE any page script runs
  await s('Page.enable');
  await s('Runtime.enable');
  await s('Network.enable');
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

  /* ---- 1. no network probes at all ---- */
  const probed = netRequests.filter((r) => /Fetch|XHR|WebSocket/.test(r));
  check('zero HTTP/XHR/Fetch probes from the page', probed.length === 0, probed.join('; '));
  check('fetch spy saw no calls', await evaluate('window.__fetchCalls.length') === 0);

  /* ---- 2. app surface + initial honest state ---- */
  check('QwenStudio exposes w1-static',
        await evaluate('window.QwenStudio && window.QwenStudio.version') === 'w1-static');
  check('Generate starts disabled',
        await evaluate('document.getElementById("generate-btn").disabled') === true);
  check('no autofocus on load (activeElement is body)',
        await evaluate('document.activeElement.tagName') === 'BODY');
  const reason = await evaluate('document.getElementById("generate-reason").textContent');
  check('disabled reason names the missing adapter/job API',
        /no generation adapter or job API/i.test(reason), reason);

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
  const wantKeys = ['attention_backend', 'height', 'kind', 'negative_prompt',
                    'offload', 'prompt', 'quant', 'seed', 'steps', 'true_cfg_scale',
                    'vae_tiled_decode', 'width'].join(',');
  check('snapshot keys mirror GenerationConfig fields + kind (snake_case, no aliases)',
        snapInfo.keys === wantKeys, snapInfo.keys);
  check('snapshot seed is a decimal STRING',
        snapInfo.seedType === 'string' && snapInfo.seed === '0', snapInfo.seedType);

  /* ---- 6. presets and manual dimension matching ---- */
  const pressed = () => evaluate(`Array.prototype.map.call(
      document.querySelectorAll('.preset-btn'),
      function (b) { return b.dataset.preset + '=' + b.getAttribute('aria-pressed'); }
    ).join(',')`);
  await evaluate('document.querySelector(\'[data-preset="landscape"]\').click()');
  check('preset click sets 1344x768',
        await evaluate('document.getElementById("width").value + "x" + document.getElementById("height").value') === '1344x768');
  check('preset click presses ONLY landscape',
        await pressed() === 'square=false,landscape=true,portrait=false');
  const setDims = async (w, h) => {
    await evaluate(`(function () {
      var wid = document.getElementById('width'), hei = document.getElementById('height');
      wid.value = '${w}'; hei.value = '${h}';
      wid.dispatchEvent(new Event('input')); hei.dispatchEvent(new Event('input'));
    })()`);
  };
  await setDims('1344', '768');
  check('manual exact 1344x768 re-presses landscape',
        await pressed() === 'square=false,landscape=true,portrait=false');
  await setDims('1024', '1024');
  check('manual exact 1024x1024 re-presses square',
        await pressed() === 'square=true,landscape=false,portrait=false');
  await setDims('1056', '608');
  check('non-preset dimensions clear ALL presets',
        await pressed() === 'square=false,landscape=false,portrait=false');

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

  /* ---- 9b. over-length seed entry: NO silent truncation ---- */
  await s('Input.enable');
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
  await evaluate('document.getElementById("generate-btn").click()');
  await evaluate(`document.getElementById('create-form').dispatchEvent(new Event('submit', {cancelable:true}))`);
  const afterSubmit = await evaluate(`[
      document.getElementById('generate-btn').disabled,
      document.getElementById('generate-reason').textContent,
      window.__fetchCalls.length
    ].join('||')`);
  const [stillDisabled, submitMsg, spyCount] = afterSubmit.split('||');
  check('Generate still disabled after submit attempt', stillDisabled === 'true');
  check('submit handler refuses honestly (Nothing was submitted)',
        /Nothing was submitted/.test(submitMsg), submitMsg);
  check('still zero fetch calls after submit attempt', spyCount === '0');

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
    check(`no horizontal overflow at ${width}px`,
          Number(scrollW) <= Number(innerW) + 1 && over === '',
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

try {
  await main();
} catch (err) {
  check('script completed without error', false, String(err && err.stack || err));
} finally {
  console.log(results.join('\n'));
  console.log(`\n${results.length - failed}/${results.length} browser checks passed`);
  if (ws) { try { ws.close(); } catch {} }
  if (chrome) {
    // Prefer a graceful CDP close so Chrome reaps its own children; fall back
    // to killing the whole process group.
    try { chrome.kill('SIGTERM', true); } catch {}
    await delay(1500);
    try { chrome.kill('SIGKILL', true); } catch {}
    try { chrome.unref(); } catch {}
  }
  try { rmSync(profileDir, { recursive: true, force: true, maxRetries: 5 }); } catch {}
}
process.exit(failed === 0 ? 0 : 1);
