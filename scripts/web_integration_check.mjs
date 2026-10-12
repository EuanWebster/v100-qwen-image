#!/usr/bin/env node
/* W2 local HTTP integration check — SEPARATE from the guarded main suite.
 *
 * What it does:
 *   1. builds a tiny temp manifest + empty models dir under the system temp
 *      directory (NO real assets, no downloads, no repo writes);
 *   2. spawns the app server (python -m qwen_engine.web) as OUR OWN child on
 *      an ephemeral 127.0.0.1 port (server prints its actual origin);
 *   3. runs raw-socket checks: static + content types, traversal/encoded/
 *      absolute-target refusal, single-exact Host validation, external
 *      Origin/cross-site denial with NO CORS headers ever, X-Qwen-UI write
 *      gate, Content-Length/Transfer-Encoding/Expect limits, full 64-bit
 *      seed round-trip (string in, exact uint64 number in the export bytes),
 *      hostile-JSON typed rejections, defensive headers;
 *   4. REAL CHROME (opt-in only): set QWEN_RUN_CHROME=1 *and* CHROME_PATH.
 *      The leg opens the served page (desktop + mobile widths), exercises
 *      status/plan/export through the real UI, and signals only its OWN
 *      spawned process group on cleanup. Inside a capacity-limited sandbox
 *      it reports NOT RUN honestly (free cgroup slots are reported, the
 *      cause is not claimed). It never launches Chrome unless opted in.
 *
 * Cleanup only ever touches the processes THIS script spawned (server and,
 * when opted in, Chrome) via their process groups — never a broad kill.
 * Exit: 0 all executed checks passed; 1 a check failed; 2 environment
 * problem (server could not start); 3 cgroup capacity preflight refused.
 */
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import net from 'node:net';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  cgroupCapacity, launchChrome, makeCdp, openCdpSocket, shutdownChrome, watchdog,
} from './chrome_lifecycle.mjs';

const WHOLE_CHECK_MS = 240000;
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');
const PYTHON = process.env.QWEN_PYTHON || 'python3';
const WITH_CHROME = process.env.QWEN_RUN_CHROME === '1';

let results = [];
let failed = 0;
function check(name, ok, detail = '') {
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  [' + detail + ']' : ''}`);
  if (!ok) failed++;
}
const delay = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- server
const tmpRoot = mkdtempSync(join(tmpdir(), 'qwen-web-integration-'));
const manifestPath = join(tmpRoot, 'manifest.json');
const modelsDir = join(tmpRoot, 'models');
writeFileSync(manifestPath, JSON.stringify({
  schema_version: 1,
  artifacts: [{
    role: 'transformer', repo: 'Qwen/Qwen-Image', revision: '0'.repeat(40),
    filename: 'tiny-placeholder.safetensors', sha256: 'a'.repeat(64), bytes: 1,
  }],
}));

let server = null;
try {
  server = await startServer();
} catch (err) {
  console.error('integration: server could not start —', err.message);
  try { rmSync(tmpRoot, { recursive: true, force: true }); } catch {}
  process.exit(2);
}
const { port, child } = server;
const ORIGIN = `http://127.0.0.1:${port}`;
console.log(`server origin: ${ORIGIN}`);

function startServer() {
  const child = spawn(PYTHON,
    ['-B', '-m', 'qwen_engine.web', '--host', '127.0.0.1', '--port', '0',
     '--manifest', manifestPath, '--models-dir', modelsDir],
    { cwd: repoRoot, env: { ...process.env, PYTHONPATH: join(repoRoot, 'src') },
      stdio: ['ignore', 'pipe', 'pipe'], detached: true });
  const announced = new Promise((res, rej) => {
    let buf = '';
    let errBuf = '';
    const t = setTimeout(() => rej(new Error('server did not announce an origin within 30s')), 30000);
    child.stdout.on('data', (b) => {
      buf += String(b);
      const m = /listening on http:\/\/127\.0\.0\.1:(\d+)/.exec(buf);
      if (m) { clearTimeout(t); res({ port: Number(m[1]), child }); }
    });
    child.stderr.on('data', (b) => { errBuf += String(b); });
    child.on('error', (e) => { clearTimeout(t); rej(e); });
    child.on('exit', (code, sig) =>
      rej(new Error(`server exited early (code=${code} signal=${sig}): ${errBuf.slice(-400)}`)));
  });
  return announced;
}

// ---------------------------------------------------------------- raw HTTP
function rawRequest(text, expectStatusFirst = true, ms = 8000) {
  return new Promise((res, rej) => {
    const sock = net.connect(port, '127.0.0.1');
    let buf = '';
    const timer = setTimeout(() => { sock.destroy(); rej(new Error('raw request timed out')); }, ms);
    const done = () => { clearTimeout(timer); sock.destroy(); res(buf); };
    const wantBody = () => {
      const m = /\r\ncontent-length: *(\d+)/i.exec(buf.slice(0, buf.indexOf('\r\n\r\n') + 4));
      return m ? Number(m[1]) : 0;
    };
    const settled = () => {
      if (!buf.includes('\r\n\r\n')) return false;
      // skip any interim 1xx responses (each ends with its own blank line)
      while (/^HTTP\/1\.\d 1\d\d\b/.test(buf)) {
        buf = buf.replace(/^HTTP\/1\.\d 1\d\d[^\r\n]*\r\n\r\n/, '');
        if (!buf.includes('\r\n\r\n')) return false;
      }
      const headEnd = buf.indexOf('\r\n\r\n') + 4;
      return buf.length - headEnd >= wantBody();
    };
    sock.on('error', (e) => { clearTimeout(timer); rej(e); });
    sock.on('data', (b) => { buf += String(b); if (settled()) done(); });
    sock.on('close', () => { if (buf.length && !timer.fired) { clearTimeout(timer); res(buf); } });
    sock.on('connect', () => sock.write(text));
  });
}
const head = (r) => r.split('\r\n\r\n')[0];
const body = (r) => r.slice(r.indexOf('\r\n\r\n') + 4);
const status = (r) => Number(/^HTTP\/1\.[01] (\d{3})/.exec(r)[1]);
const hdr = (r, name) => head(r).split('\r\n').filter((l) => l.toLowerCase().startsWith(name.toLowerCase() + ':')).map((l) => l.split(': ')[1]);
const jsonOf = (r) => { try { return JSON.parse(body(r)); } catch { return null; } };

function post(target, payload, extraHeaders = '') {
  const bytes = Buffer.byteLength(payload);
  return rawRequest(
    `POST ${target} HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n` +
    `Content-Type: application/json\r\nContent-Length: ${bytes}\r\n` +
    `X-Qwen-UI: 1\r\nOrigin: ${ORIGIN}\r\n${extraHeaders}\r\n${payload}`);
}
const envelope = (config) => JSON.stringify({ schema: 'qwen_engine.web_plan_request.v1', config });

async function httpChecks() {
  // static
  const index = await rawRequest(`GET / HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('GET / serves the app over text/html',
        status(index) === 200 && /text\/html/.test(head(index)) &&
        head(index).includes('charset=utf-8'));
  check('defensive headers on static',
        head(index).includes('nosniff') && head(index).includes('no-store') &&
        head(index).includes('frame-ancestors') && head(index).includes('no-referrer'));
  const appjs = await rawRequest(`GET /app.js HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('GET /app.js serves text/javascript',
        status(appjs) === 200 && /text\/javascript/.test(head(appjs)));
  const bad = await rawRequest(`GET /../pyproject.toml HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('path traversal is refused (not served, not translated)',
        status(bad) === 404 && body(bad).includes('not_found') && !body(bad).includes('[project]'));
  for (const [target, expect] of [['/app%2Ejs', 400], ['/app.js?v=1', 400],
       ['/%2e%2e/app.js', 400], [`http://127.0.0.1:${port}/app.js`, 400],
       // '//x' is folded to '/x' by Python's own request-line parser before
       // the app sees it; either way it must be REFUSED, never served.
       ['//etc/hosts', 404], ['/../../etc/hosts', 404]]) {
    const r = await rawRequest(`GET ${target} HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
    check(`target refused: ${target}`,
          status(r) === expect && !body(r).includes('root:') && !body(r).includes('[project]'),
          head(r).split('\r\n')[0]);
  }

  // Host header
  const dupHost = await rawRequest(`GET / HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nHost: localhost:${port}\r\n\r\n`);
  check('duplicate Host refused', status(dupHost) === 400);
  const evilHost = await rawRequest(`GET / HTTP/1.1\r\nHost: evil.example:${port}\r\n\r\n`);
  check('foreign Host refused (DNS-rebinding shape)', status(evilHost) === 400);
  const wrongPort = await rawRequest(`GET / HTTP/1.1\r\nHost: 127.0.0.1:1\r\n\r\n`);
  check('Host with wrong port refused', status(wrongPort) === 400);
  const lh = await rawRequest(`GET / HTTP/1.1\r\nHost: localhost:${port}\r\n\r\n`);
  check('localhost alias at the bound port accepted', status(lh) === 200);

  // OPTIONS / CORS
  const opt = await rawRequest(`OPTIONS /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nOrigin: http://evil.example\r\nAccess-Control-Request-Method: POST\r\n\r\n`);
  check('OPTIONS is 405 and never grants CORS',
        status(opt) === 405 && !/access-control-allow/i.test(head(opt)));

  // foreign origin / cross-site on every route
  const foreignOriginGet = await rawRequest(`GET /api/v1/status HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nOrigin: http://evil.example\r\n\r\n`);
  check('foreign Origin denied on GET status', status(foreignOriginGet) === 403);
  const crossSite = await post('/api/v1/plan', envelope({ prompt: 'x' }), 'Sec-Fetch-Site: cross-site\r\n');
  check('cross-site Sec-Fetch-Site denied on plan', status(crossSite) === 403);
  const sameSite = await post('/api/v1/plan', envelope({ prompt: 'x' }), 'Sec-Fetch-Site: same-site\r\n');
  check('same-site also denied (stricter than required)', status(sameSite) === 403);

  // status
  const st = jsonOf(await rawRequest(`GET /api/v1/status HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n\r\n`));
  check('status schema + honest booleans',
        st && st.schema === 'qwen_engine.web_status.v1' &&
        st.planning_available === true && st.generation_available === false &&
        st.gates.adapter.code === 'adapter_not_implemented' &&
        st.gates.hardware.state === 'unknown' &&
        st.generation_blockers.includes('assets_unverified') &&
        st.generation_blockers.includes('calibration_unknown'),
        st && st.generation_blockers.join(','));
  const stBody = JSON.stringify(st);
  check('status leaks no manifest filename', !stBody.includes('tiny-placeholder') && !stBody.includes('safetensors'));

  // write gate
  const noUi = await rawRequest(`POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Length: 10\r\nOrigin: ${ORIGIN}\r\n\r\n{"config":{}`);
  check('POST without exact X-Qwen-UI refused', status(noUi) === 403);

  // full 64-bit seed round-trip
  const fullSeed = '18446744073709551615';
  const planResp = await post('/api/v1/plan', envelope({ prompt: 'a fox', seed: fullSeed, width: 1024, height: 1024 }));
  const plan = jsonOf(planResp);
  check('plan accepts full uint64 seed STRING and echoes it exactly as a string',
        status(planResp) === 200 && plan && plan.schema === 'qwen_engine.web_plan.v1' &&
        plan.config.seed === fullSeed && typeof plan.config.seed === 'string' &&
        plan.latent_tokens === 4096 && plan.manifest.ready === false &&
        plan.manifest.verification === 'not_performed' && plan.manifest.fit === 'not_evaluated');
  check('plan leaks no runtime path',
        !JSON.stringify(plan).includes(tmpRoot));
  const planManifest = JSON.stringify(plan.manifest);
  check('plan manifest is metadata-only (no filename/hash leaked)',
        !planManifest.includes('safetensors') && plan.manifest.configured === true &&
        plan.manifest.valid === true, planManifest);
  check('status manifest is metadata-only (counts, never filenames/hashes)',
        st.manifest.artifact_count === 1 && st.manifest.valid === true &&
        st.manifest.verification === 'not_performed' &&
        !JSON.stringify(st.manifest).includes('tiny-placeholder') &&
        !JSON.stringify(st.manifest).includes('a'.repeat(64)));

  const exp = await post('/api/v1/config/export', envelope({ prompt: 'a fox', seed: fullSeed }));
  check('export is a raw attachment',
        status(exp) === 200 && /attachment; filename="qwen-editable-config.json"/.test(head(exp)));
  const expBody = body(exp);
  check('export carries the exact uint64 NUMBER and excludes runtime paths',
        expBody.includes('"seed": 18446744073709551615') &&
        !expBody.includes('output_dir') && !expBody.includes('manifest_path') &&
        !expBody.includes(tmpRoot));

  // hostile JSON typed
  const dupKeys = await post('/api/v1/plan', '{"schema":"a","schema":"b"}');
  check('duplicate JSON keys refused', status(dupKeys) === 400 && jsonOf(dupKeys).code === 'duplicate_json_key');
  const nanBody = await post('/api/v1/plan', '{"schema":"qwen_engine.web_plan_request.v1","config":{"width":NaN}}');
  check('NaN literal refused', status(nanBody) === 400);
  const numericSeed = await post('/api/v1/plan', '{"schema":"qwen_engine.web_plan_request.v1","config":{"seed":5}}');
  check('numeric JSON seed refused (string wire only)',
        status(numericSeed) === 400 && jsonOf(numericSeed).code === 'invalid_seed');
  const bigSeed = await post('/api/v1/plan', envelope({ prompt: 'x', seed: '1'.repeat(21) }));
  check('21-digit seed refused as typed error', status(bigSeed) === 400 && jsonOf(bigSeed).code === 'invalid_seed');
  const chunked = rawChunked();
  check('chunked Transfer-Encoding refused', (await chunked) === 400);
  const hugeLen = await rawRequest(`POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Length: 70000\r\nX-Qwen-UI: 1\r\nOrigin: ${ORIGIN}\r\n\r\nx`);
  check('Content-Length over 64KiB refused BEFORE reading', status(hugeLen) === 413);
  const expectHdr = await rawRequest(`POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Length: 2\r\nExpect: unknown-ext\r\nX-Qwen-UI: 1\r\n\r\n{}`);
  check('unsupported Expect refused (417)', status(expectHdr) === 417);
  const continueHdr = await rawRequest(`POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Length: ${Buffer.byteLength(envelope({ prompt: 'x' }))}\r\nExpect: 100-continue\r\nX-Qwen-UI: 1\r\nOrigin: ${ORIGIN}\r\n\r\n${envelope({ prompt: 'x' })}`);
  check('100-continue flows to a normal 200', status(continueHdr) === 200);
}

function rawChunked() {
  return new Promise((res) => {
    const sock = net.connect(port, '127.0.0.1');
    let buf = '';
    const timer = setTimeout(() => { sock.destroy(); res(-1); }, 8000);
    sock.on('data', (b) => {
      buf += String(b);
      if (buf.includes('\r\n\r\n')) { clearTimeout(timer); sock.destroy(); res(status(buf)); }
    });
    sock.on('error', () => { clearTimeout(timer); res(-1); });
    sock.on('connect', () => sock.write(
      `POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nTransfer-Encoding: chunked\r\nX-Qwen-UI: 1\r\nOrigin: ${ORIGIN}\r\n\r\n2\r\n{}\r\n0\r\n\r\n`));
  });
}

// ---------------------------------------------------------------- Chrome leg
async function chromeChecks() {
  if (!WITH_CHROME) {
    results.push('NOT RUN  real Chrome leg — set QWEN_RUN_CHROME=1 (with CHROME_PATH) on a clean sandbox');
    return;
  }
  if (!process.env.CHROME_PATH) {
    results.push('NOT RUN  real Chrome leg — CHROME_PATH is not set');
    return;
  }
  const cap = cgroupCapacity();
  if (cap && !cap.ok) {
    results.push(`NOT RUN  real Chrome leg — only ${cap.free} free task slots (pids ${cap.detail}); cause not verified, run on a clean sandbox`);
    return;
  }
  const profileDir = mkdtempSync(join(tmpdir(), 'qwen-web-chrome-'));
  let chrome = null, ws = null, cdp = null;
  try {
    const launched = launchChrome({ chromePath: process.env.CHROME_PATH, profileDir });
    chrome = launched.chrome;
    const sock = openCdpSocket(await launched.wsUrl);
    ws = sock.ws;
    await sock.ready;
    cdp = makeCdp(ws);
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    const s = (m, p) => cdp.send(m, p || {}, sessionId);
    await s('Page.enable'); await s('Runtime.enable');
    await s('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: tmpRoot, eventsEnabled: true });
    const downloads = [];
    cdp.onEvent = (msg) => { if (msg.method === 'Browser.downloadWillBegin') downloads.push(msg.params.url); };
    const evaluate = async (expression) => {
      const r = await s('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval exception');
      return r.result.value;
    };
    for (const [w, h, mobile] of [[1440, 900, false], [375, 800, true]]) {
      await s('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile });
      await s('Page.navigate', { url: ORIGIN + '/' });
      await delay(600);
      const chip = await evaluate('document.getElementById("connection-chip").textContent');
      check(`status chip shows planning-connected at ${w}px`, /Connected .*planning only/.test(chip), chip);
      check(`Generate stays disabled at ${w}px`,
            await evaluate('document.getElementById("generate-btn").disabled') === true);
      await evaluate(`(function(){
        document.getElementById('prompt').value='integration fox';
        document.getElementById('prompt').dispatchEvent(new Event('input'));
        document.getElementById('seed').value='${'9'.repeat(20)}';
        document.getElementById('seed').dispatchEvent(new Event('input'));
      })()`);
      await evaluate('document.getElementById("btn-plan").click()');
      await delay(500);
      check(`plan renders the full-range seed string at ${w}px`,
            await evaluate('document.getElementById("plan-seed").textContent') === '9'.repeat(20));
      const tokenText = await evaluate('document.getElementById("plan-tokens").textContent');
      check(`plan renders latent tokens at ${w}px`, tokenText === '4096', tokenText);
      const manifestText = await evaluate('document.getElementById("plan-manifest").textContent');
      check(`plan manifest text is verification-not-performed at ${w}px`,
            /verification not performed/.test(manifestText), manifestText);
      await evaluate('document.getElementById("btn-export").click()');
      await delay(600);
      check(`export downloaded raw bytes at ${w}px`, downloads.length >= 1, downloads.join(','));
      const overflow = await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1');
      check(`no horizontal overflow at ${w}px`, overflow === true);
    }
  } finally {
    await shutdownChrome(chrome, ws, cdp);
    try { rmSync(profileDir, { recursive: true, force: true, maxRetries: 5 }); } catch {}
  }
}

// ---------------------------------------------------------------- run
const guard = watchdog(WHOLE_CHECK_MS, 'whole integration check');
try {
  await Promise.race([
    (async () => { await httpChecks(); await chromeChecks(); })(),
    guard,
  ]);
} catch (err) {
  check('integration script completed without error', false, String(err && err.stack || err));
} finally {
  guard.cancel();
  console.log(results.join('\n'));
  console.log(`\n${results.length - failed}/${results.length} integration checks passed`);
  // cleanup ONLY our own spawned server process group (bounded):
  try {
    const exited = new Promise((res) => child.once('exit', () => res(true)));
    let got = await Promise.race([exited, new Promise((r) => setTimeout(() => r(false), 100))]);
    if (!got) {
      try { process.kill(-child.pid, 'SIGTERM'); } catch {}
      got = await Promise.race([exited, new Promise((r) => setTimeout(() => r(false), 3000))]);
      if (!got) { try { process.kill(-child.pid, 'SIGKILL'); } catch {} }
    }
  } catch {}
  try { rmSync(tmpRoot, { recursive: true, force: true, maxRetries: 5 }); } catch {}
}
process.exit(failed === 0 ? 0 : 1);
