#!/usr/bin/env node
/* W2 local HTTP integration check — SEPARATE from the guarded main suite.
 *
 * What it does:
 *   1. builds a tiny temp manifest + empty models dir under the system temp
 *      directory (NO real assets, no downloads, no repo writes);
 *   2. spawns the app server (python -m qwen_engine.web) as OUR OWN child on
 *      an ephemeral 127.0.0.1 port (server prints its actual origin) — the
 *      STARTUP itself is covered by the whole-check watchdog, so a hung
 *      server can never outlive this script;
 *   3. runs raw-socket checks through scripts/raw_http.mjs (Buffer framing:
 *      interim-1xx sliced by CRLF boundary, bounded response bytes, latin1
 *      headers / UTF-8 bodies): static + content types, traversal/encoded/
 *      absolute-target refusal, single-exact Host validation, external
 *      Origin/cross-site denial with NO CORS headers ever, X-Qwen-UI write
 *      gate, Content-Length/Transfer-Encoding/Expect limits, unsupported
 *      methods obeying the SAME Host/Origin policy (typed 405/404 JSON,
 *      never the stock 501 page), full 64-bit seed round-trip (string in,
 *      exact uint64 number in the export bytes), Unicode + long (5000-char)
 *      prompt round-trips, hostile-JSON typed rejections, defensive headers;
 *   4. REAL CHROME (opt-in only): set QWEN_RUN_CHROME=1 *and* CHROME_PATH.
 *      The leg opens the served page at 1440/375/320 px (sandbox ENABLED —
 *      never --no-sandbox), exercises status/plan/export through the real
 *      UI including the browser-regression scenarios from the review:
 *      an invalid draft during an in-flight valid request (stale response
 *      must never land), a stalled response BODY past a test-reduced
 *      deadline (timeout, not hang), schema-rejected status JSON (typed
 *      rejection, not a disconnect), Unicode/long prompts in the form and
 *      a raw uint64-max download verified from the NEW file on disk, an
 *      export success envelope WITHOUT a readable body (controlled-backend
 *      fixture: refused as a typed schema rejection, never a download of
 *      'undefined') and a parsePlan blank/Python-whitespace-only echoed
 *      prompt rejection against the real served plan echo.
 *      Cleanup lives at the OUTERMOST scope: however the run ends, the
 *      chrome lifecycle state (outer-scoped) asks its dedicated subreaper to
 *      finish before the server group is killed. Kernel-unreapable children
 *      retain the owner/fixture tree, not false cleanup success.
 *      Inside a capacity-limited sandbox it reports NOT RUN
 *      honestly (free cgroup slots reported, cause not claimed). NOT RUN
 *      checks are counted SEPARATELY — never inside passed/total.
 *
 * Cleanup only ever touches the processes THIS script spawned (server and,
 * when opted in, Chrome) — server group and browser owner PID, never a broad kill.
 * Exit: 0 all EXECUTED checks passed; 1 a check failed or errored;
 * 2 environment problem (server could not start).
 */
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import net from 'node:net';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  cgroupCapacity, launchChrome, makeCdp, openCdpSocket, shutdownChrome, watchdog,
} from './chrome_lifecycle.mjs';
import {
  MAX_RESPONSE_BYTES, inspectResponse, responseHead, responseBody,
  responseStatus, headerValues,
} from './raw_http.mjs';

const WHOLE_CHECK_MS = 240000;
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');
const PYTHON = process.env.QWEN_PYTHON || 'python3';
const WITH_CHROME = process.env.QWEN_RUN_CHROME === '1';

// Three honest counters: NOT RUN never dilutes passed/total.
const results = [];
let passed = 0, failed = 0, notRun = 0;
let stopped = false;            // frozen once the watchdog fires
function check(name, ok, detail = '') {
  if (stopped) return;          // results are final once the deadline hit
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  [' + detail + ']' : ''}`);
  if (ok) passed++; else failed++;
}
function notRunCheck(name, detail = '') {
  if (stopped) return;
  results.push(`NOT RUN  ${name}${detail ? '  [' + detail + ']' : ''}`);
  notRun++;
}
// An aborted run must NEVER report success: these recorders bypass the
// freeze (late results from other legs stay frozen, but the ABORT itself is
// always recorded, so 'N/N passed' can never follow a timeout).
function forceFail(name, detail = '') {
  results.push(`FAIL  ${name}${detail ? '  [' + detail + ']' : ''}`);
  failed++;
}
function forceNotRun(name, detail = '') {
  results.push(`NOT RUN  ${name}${detail ? '  [' + detail + ']' : ''}`);
  notRun++;
}
const delay = (ms) => new Promise((r) => setTimeout(r, ms));

/** Bounded cleanup of OUR OWN spawned child group (never a broad kill). */
async function killGroup(kid) {
  if (!kid || !Number.isInteger(kid.pid)) return;
  const exited = new Promise((res) => {
    if (kid.exitCode !== null || kid.signalCode !== null) res(true);
    else kid.once('exit', () => res(true));
  });
  try { kid.unref(); } catch {}
  if (await Promise.race([exited, new Promise((r) => setTimeout(() => r(false), 100))])) return;
  for (const sig of ['SIGTERM', 'SIGKILL']) {
    try { process.kill(-kid.pid, sig); } catch { /* already gone */ }
    if (await Promise.race([exited, new Promise((r) => setTimeout(() => r(false), 3000))])) return;
  }
}

// ---------------------------------------------------------------- server
const tmpRoot = mkdtempSync(join(tmpdir(), 'qwen-web-integration-'));
const manifestPath = join(tmpRoot, 'manifest.json');
const modelsDir = join(tmpRoot, 'models');
mkdirSync(modelsDir, { recursive: true });   // real empty dir (config validates existence)
writeFileSync(manifestPath, JSON.stringify({
  schema_version: 1,
  artifacts: [{
    role: 'transformer', repo: 'Qwen/Qwen-Image', revision: '0'.repeat(40),
    filename: 'tiny-placeholder.safetensors', sha256: 'a'.repeat(64), bytes: 1,
  }],
}));

// OUTERMOST lifecycle state: the watchdog/finally can clean up no matter
// where in the run it is interrupted — nothing lives only inside a promise.
let child = null;               // our own server child
let chromeState = null;         // { chrome, ws, cdp, profileDir } once acquired
let port = 0;
let ORIGIN = '';

async function cleanupChrome() {
  // Idempotent, bounded, OUR OWN Chrome only — called from the outermost
  // finally, so even a watchdog abort mid-leg cannot orphan the browser.
  const st = chromeState;
  if (!st) return;
  const cleaned = await shutdownChrome(st.chrome, st.ws, st.cdp);
  if (cleaned === false) return false;
  chromeState = null;
  try { rmSync(st.profileDir, { recursive: true, force: true, maxRetries: 5 }); } catch {}
}

function startServer() {
  child = spawn(PYTHON,
    ['-B', '-m', 'qwen_engine.web', '--host', '127.0.0.1', '--port', '0',
     '--manifest', manifestPath, '--models-dir', modelsDir],
    { cwd: repoRoot, env: { ...process.env, PYTHONPATH: join(repoRoot, 'src') },
      stdio: ['ignore', 'pipe', 'pipe'], detached: true });
  const announced = new Promise((res, rej) => {
    let buf = '';
    let errBuf = '';
    let done = false;
    const fail = (err) => { if (!done) { done = true; clearTimeout(t); rej(err); } };
    const t = setTimeout(() => fail(new Error('server did not announce an origin within 30s')), 30000);
    child.stdout.on('data', (b) => {
      buf = (buf + String(b)).slice(0, 16384);           // bounded head
      const m = /listening on http:\/\/127\.0\.0\.1:(\d+)/.exec(buf);
      if (m && !done) { done = true; clearTimeout(t); res({ port: Number(m[1]), child }); }
    });
    child.stdout.on('end', () => child.stdout.resume?.());  // drain, never block
    child.stderr.on('data', (b) => { errBuf = (errBuf + String(b)).slice(-16384); });
    child.stderr.resume?.();
    child.on('error', (e) => fail(e));
    child.on('exit', (code, sig) => fail(new Error(
      `server exited early (code=${code} signal=${sig}): ${errBuf.slice(-400)}`)));
  });
  return announced;
}

// ---------------------------------------------------------------- raw HTTP
/**
 * One raw request, response framed by scripts/raw_http.mjs (pure,
 * self-checked): interim 1xx sliced at their CRLF boundary, Content-Length
 * completion on the FINAL block only, a fixed byte cap, latin1 head bytes
 * and UTF-8 body decode. Resolves a Buffer.
 */
function rawRequest(text, ms = 8000) {
  return new Promise((res, rej) => {
    const sock = net.connect(port, '127.0.0.1');
    const chunks = [];
    let len = 0;
    const noBody = /^HEAD\s/.test(text);   // HEAD: CL declared, no body to await
    const timer = setTimeout(() => { sock.destroy(); rej(new Error('raw request timed out')); }, ms);
    const settledBuf = () => {
      const buf = Buffer.concat(chunks);
      const s = inspectResponse(buf);
      return (s.complete && (noBody || s.bodyReady)) ? buf : null;
    };
    const done = () => {
      const buf = settledBuf();
      if (buf === null) return false;
      clearTimeout(timer); sock.destroy(); res(buf); return true;
    };
    sock.on('error', (e) => {
      // A post-response EPIPE/close race is fine once we hold a complete
      // response; a truncated one must reject, never resolve as 'pass'.
      if (!done()) { clearTimeout(timer); rej(e); }
    });
    sock.on('data', (b) => {
      len += b.length;
      if (len > MAX_RESPONSE_BYTES) {
        clearTimeout(timer); sock.destroy();
        rej(new Error(`response exceeded the ${MAX_RESPONSE_BYTES} byte cap`)); return;
      }
      chunks.push(b);
      done();
    });
    sock.on('close', () => {
      if (!done()) {
        clearTimeout(timer);
        rej(new Error('connection closed with an incomplete response (' + len + ' bytes)'));
      }
    });
    sock.on('connect', () => sock.write(text));
  });
}
const head = (r) => responseHead(r);
const body = (r) => responseBody(r);        // UTF-8 (non-ASCII prompt echoes)
const status = (r) => responseStatus(r);
const hdr = (r, name) => headerValues(r, name);
const jsonOf = (r) => { try { return JSON.parse(responseBody(r)); } catch { return null; } };

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
       // Routing uses the RAW wire target (the stdlib folds '//' in
       // self.path before the app sees it), so '//x' is now a typed 400;
       // '/../../x' folds to a nonexistent origin-form path and 404s.
       // Either way: REFUSED, never served.
       ['//etc/hosts', 400], ['/../../etc/hosts', 404]]) {
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

  // OPTIONS / CORS — every method (OPTIONS included) passes the Host and
  // Origin/site policy FIRST; OPTIONS then lands on a typed 405 with Allow
  // and NEVER an Access-Control-* header.
  const opt = await rawRequest(`OPTIONS /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nAccess-Control-Request-Method: POST\r\n\r\n`);
  check('OPTIONS is 405 with Allow and never grants CORS',
        status(opt) === 405 && !/access-control-allow/i.test(head(opt)) &&
        /allow: POST/i.test(head(opt)),
        head(opt).split('\r\n')[0]);
  const optForeign = await rawRequest(`OPTIONS /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nOrigin: http://evil.example\r\nAccess-Control-Request-Method: POST\r\n\r\n`);
  check('OPTIONS with foreign Origin denied by policy first', status(optForeign) === 403);
  const putReq = await rawRequest(`PUT /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nOrigin: http://evil.example\r\n\r\n`);
  check('PUT bypass attempt denied by policy (403), not a policy-free 405', status(putReq) === 403);
  const headStatic = await rawRequest(`HEAD / HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('HEAD routed through policy to 405 with Allow',
        status(headStatic) === 405 && /allow: GET/i.test(head(headStatic)));

  // Methods with NO dispatcher at all (the stock stdlib would answer an
  // unguarded 501 HTML page): they must obey the SAME Host/Origin policy
  // and land on the typed JSON 405/404 path.
  const weirdForeign = await rawRequest(`WEIRD /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nOrigin: http://evil.example\r\n\r\n`);
  check('unknown method + foreign Origin: policy denies BEFORE method typing',
        status(weirdForeign) === 403 && !/html/i.test(head(weirdForeign)),
        head(weirdForeign).split('\r\n')[0]);
  const weird = await rawRequest(`WEIRD /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nOrigin: ${ORIGIN}\r\n\r\n`);
  check('unknown method on a known route is typed 405 JSON with Allow (never stock 501 HTML)',
        status(weird) === 405 && /application\/json/.test(head(weird)) &&
        /method_not_allowed/.test(body(weird)) && /allow: POST/i.test(head(weird)),
        head(weird).split('\r\n')[0]);
  const weirdUnknown = await rawRequest(`WEIRD /nope HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nOrigin: ${ORIGIN}\r\n\r\n`);
  check('unknown method on an unknown route is typed 404 JSON',
        status(weirdUnknown) === 404 && /not_found/.test(body(weirdUnknown)));

  // Request-line abuse (root review): an oversized request line must get a
  // TYPED 414 JSON with the defensive headers — never a silent close — and
  // a TAB-separated request line must not be served through the stdlib's
  // whitespace folding (the old normalized self.path turned '\t//app.js'
  // into a 200 of //app.js).
  const longLine = await rawRequest(
    `GET /${'a'.repeat(70000)} HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('request line over 65536 bytes gets typed 414 JSON, never a silent close',
        status(longLine) === 414 && /malformed_request/.test(body(longLine)) &&
        /x-content-type-options: nosniff/i.test(head(longLine)),
        head(longLine).split('\r\n')[0]);
  const tabLine = await rawRequest(`GET\t//app.js HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('TAB-separated request line with a bad target form is typed 400, never served 200',
        status(tabLine) === 400 && /malformed|target/i.test(body(tabLine)),
        head(tabLine).split('\r\n')[0] + ' | ' + body(tabLine).slice(0, 60));
  const weirdHost = await rawRequest(`WEIRD / HTTP/1.1\r\nHost: evil.example:${port}\r\n\r\n`);
  check('unknown method + foreign Host: host policy refuses BEFORE method typing (typed 400)',
        status(weirdHost) === 400 && /invalid_host/.test(body(weirdHost)),
        head(weirdHost).split('\r\n')[0]);

  // cfg_active over REAL HTTP with the most mainstream blank of all: an
  // all-space negative prompt + cfg>1 is FALSE (config.py .strip() strips
  // plain space). A client mixing JS trim semantics would reject this plan.
  const spaceNeg = await post('/api/v1/plan',
    envelope({ prompt: 'cfg probe', negative_prompt: '   ', true_cfg_scale: 2.0 }));
  check('all-space negative prompt: server cfg_active is false (config.py strip)',
        status(spaceNeg) === 200 && jsonOf(spaceNeg).cfg_active === false,
        responseBody(spaceNeg).slice(0, 100));

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

  // prompt bounds aligned with the SERVER contract: UTF-8 round-trip and a
  // prompt over the old browser-side 4000 limit but well under 64 KiB.
  const uniPrompt = 'fox ünïcode 日本語 \u{1F98A}   continuation';
  const uniResp = await post('/api/v1/plan', envelope({ prompt: uniPrompt }));
  check('unicode prompt echoes byte-exact through UTF-8 JSON (framing decodes body as UTF-8)',
        status(uniResp) === 200 && jsonOf(uniResp)?.config?.prompt === uniPrompt,
        head(uniResp).split('\r\n')[0]);
  const longPrompt = 'x'.repeat(5000);       // > old client 4000 cap, < 64 KiB cap
  const longResp = await post('/api/v1/plan', envelope({ prompt: longPrompt }));
  check('5000-char prompt plans fine (server contract has no 4000 limit)',
        status(longResp) === 200 && jsonOf(longResp)?.config?.prompt?.length === 5000,
        head(longResp).split('\r\n')[0]);

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
  const hugeLen = await rawRequest(`POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Type: application/json\r\nContent-Length: 70000\r\nX-Qwen-UI: 1\r\nOrigin: ${ORIGIN}\r\n\r\nx`);
  check('Content-Length over 64KiB refused BEFORE reading', status(hugeLen) === 413);
  const wrongType = await rawRequest(`POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: 5\r\nX-Qwen-UI: 1\r\nOrigin: ${ORIGIN}\r\n\r\na=b=1`);
  check('form content type refused 415 before body', status(wrongType) === 415);
  const longPort = await rawRequest(`GET / HTTP/1.1\r\nHost: 127.0.0.1:${'9'.repeat(10000)}\r\n\r\n`);
  check('arbitrary-length digit port typed 400 (never a 500)', status(longPort) === 400);
  const expectHdr = await rawRequest(`POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Length: 2\r\nExpect: unknown-ext\r\nX-Qwen-UI: 1\r\n\r\n{}`);
  check('unsupported Expect refused (417)', status(expectHdr) === 417);
  const continueHdr = await rawRequest(`POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Length: ${Buffer.byteLength(envelope({ prompt: 'x' }))}\r\nExpect: 100-continue\r\nX-Qwen-UI: 1\r\nOrigin: ${ORIGIN}\r\n\r\n${envelope({ prompt: 'x' })}`);
  // Closed posture: NO early 100 is ever sent before validation — every
  // Expect extension, including 100-continue, is refused 417. (The framing
  // under test slices a 1xx WITH headers by its CRLF boundary — pure
  // protocol cases are covered by scripts/raw_http_selfcheck.mjs.)
  check('100-continue is refused 417 (no early 100)', status(continueHdr) === 417);

  // ---- legacy/early-error HTTP framing (root 16:48 review) -------------
  // The SAME four wire legs as root's memory-socket probe, now against the
  // REAL served socket. The inherited stdlib framing suppresses ALL output
  // while request_version is the HTTP/0.9 default: the versionless GET and
  // explicit HTTP/0.9 were SERVED as bare legacy bodies, and the broken
  // request line / HTTP/2.0 refusals arrived as bare JSON with NO status
  // line. All four must now come back FRAMED: real HTTP/1.x status line,
  // the shared defensive header set, typed JSON error — never a bare body.
  const defensive = (r) => ['x-content-type-options', 'referrer-policy',
    'cross-origin-resource-policy', 'x-frame-options',
    'content-security-policy', 'cache-control', 'connection']
    .every((n) => head(r).toLowerCase().includes(n + ':'));
  const legacyNoVersion = await rawRequest(`GET /app.js\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('versionless request refused FRAMED 400 with defensive headers (never a bare body)',
        status(legacyNoVersion) === 400 && defensive(legacyNoVersion) &&
        jsonOf(legacyNoVersion)?.code === 'unsupported_request_version' &&
        !body(legacyNoVersion).includes('console.log'),
        head(legacyNoVersion).split('\r\n')[0]);
  const legacy09 = await rawRequest(`GET /app.js HTTP/0.9\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('explicit HTTP/0.9 refused FRAMED 400 with defensive headers (never served)',
        status(legacy09) === 400 && defensive(legacy09) &&
        jsonOf(legacy09)?.code === 'unsupported_request_version',
        head(legacy09).split('\r\n')[0]);
  const brokenLine = await rawRequest('BROKEN\r\n\r\n');
  check('broken request line FRAMED typed 400 with defensive headers',
        status(brokenLine) === 400 && defensive(brokenLine) &&
        jsonOf(brokenLine)?.code === 'malformed_request',
        head(brokenLine).split('\r\n')[0]);
  const http20 = await rawRequest(`GET /api/v1/status HTTP/2.0\r\nHost: 127.0.0.1:${port}\r\n\r\n`);
  check('HTTP/2.0 refused FRAMED 505 with defensive headers',
        status(http20) === 505 && defensive(http20),
        head(http20).split('\r\n')[0]);
  // Malformed HEAD legs: framed refusals with an honest Content-Length but
  // NEVER a payload (the HEAD framing contract holds even pre-dispatch).
  for (const wire of [`HEAD /app.js\r\nHost: 127.0.0.1:${port}\r\n\r\n`,
                      `HEAD /app.js HTTP/0.9\r\nHost: 127.0.0.1:${port}\r\n\r\n`]) {
    const r = await rawRequest(wire);
    check(`malformed HEAD leg framed with NO payload (${wire.split('\r\n')[0]})`,
          status(r) === 400 && defensive(r) && body(r).length === 0 &&
          /content-length: \d+/i.test(head(r)),
          head(r).split('\r\n')[0]);
  }
}

function rawChunked() {
  return new Promise((res) => {
    const sock = net.connect(port, '127.0.0.1');
    const chunks = [];
    let len = 0;
    const timer = setTimeout(() => { sock.destroy(); res(-1); }, 8000);
    sock.on('data', (b) => {
      len += b.length;
      if (len > MAX_RESPONSE_BYTES) { clearTimeout(timer); sock.destroy(); res(-1); return; }
      chunks.push(b);
      const buf = Buffer.concat(chunks);
      if (inspectResponse(buf).complete) { clearTimeout(timer); sock.destroy(); res(responseStatus(buf)); }
    });
    sock.on('error', () => { clearTimeout(timer); res(-1); });
    sock.on('connect', () => sock.write(
      `POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nTransfer-Encoding: chunked\r\nX-Qwen-UI: 1\r\nOrigin: ${ORIGIN}\r\n\r\n2\r\n{}\r\n0\r\n\r\n`));
  });
}

// ---------------------------------------------------------------- Chrome leg
async function chromeChecks() {
  if (!WITH_CHROME) {
    notRunCheck('real Chrome leg — set QWEN_RUN_CHROME=1 (with CHROME_PATH) on a clean sandbox');
    return;
  }
  if (!process.env.CHROME_PATH) {
    notRunCheck('real Chrome leg — CHROME_PATH is not set');
    return;
  }
  const cap = cgroupCapacity();
  if (cap && !cap.ok) {
    notRunCheck('real Chrome leg — only ' + cap.free + ' free task slots (pids ' + cap.detail +
                '); cause not verified, run on a clean sandbox');
    return;
  }
  // ACQUISITION GUARD: once the whole-run deadline has fired, do NOT
  // create a profile or spawn a browser we might fail to supervise.
  if (stopped) {
    forceNotRun('real Chrome leg — deadline already hit before acquisition ' +
                '(no profile created, no browser spawned)');
    return;
  }
  const profileDir = mkdtempSync(join(tmpdir(), 'qwen-web-chrome-'));
  // Lifecycle state is registered OUTWARD as it is acquired (chrome first,
  // then ws, then cdp); cleanup happens in the script's outermost finally,
  // never in a leg-local finally a watchdog abort could skip.
  const launched = launchChrome({ chromePath: process.env.CHROME_PATH, profileDir });
  chromeState = { chrome: launched.chrome, ws: null, cdp: null, profileDir };
  const sock = openCdpSocket(await launched.wsUrl);
  chromeState.ws = sock.ws;
  await sock.ready;
  chromeState.cdp = makeCdp(sock.ws);
  const cdp = chromeState.cdp;
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  const s = (m, p) => cdp.send(m, p || {}, sessionId);
  await s('Page.enable'); await s('Runtime.enable'); await s('Network.enable');
  // Browser-scoped CDP command: Browser.* goes on the BROWSER connection,
  // not the page session.
  await cdp.send('Browser.setDownloadBehavior',
    { behavior: 'allow', downloadPath: tmpRoot, eventsEnabled: true });
  const downloads = [];        // {guid, url, done}
  const pageExceptions = [];
  const netUrls = [];
  cdp.onEvent = (msg) => {
    if (msg.method === 'Browser.downloadWillBegin')
      downloads.push({ guid: msg.params.guid, url: msg.params.url, done: false });
    else if (msg.method === 'Browser.downloadProgress') {
      const d = downloads.find((x) => x.guid === msg.params.guid);
      if (d && msg.params.state === 'completed') d.done = true;
    } else if (msg.method === 'Runtime.exceptionThrown') {
      pageExceptions.push(msg.params.exceptionDetails?.exception?.description || 'page exception');
    } else if (msg.method === 'Network.requestWillBeSent') {
      netUrls.push({ url: msg.params.request.url, type: msg.params.type });
    }
  };
  const evaluate = async (expression) => {
    const r = await s('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval exception');
    return r.result.value;
  };
  async function waitUntil(fn, ms, label) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      // A poll against a still-navigating (about:blank) page throws — that
      // is 'not yet', not a failure: keep polling until the bound.
      try { if (await fn()) return true; } catch { /* not ready yet */ }
      await delay(120);
    }
    throw new Error(`timed out waiting for ${label}`);
  }
  const maxSeed = '18446744073709551615';   // real uint64 max ('9'*20 would EXCEED it)
  for (const [w, h, mobile] of [[1440, 900, false], [375, 800, true], [320, 640, true]]) {
    await s('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile });
    const beforeDownloads = downloads.length;
    const filesBefore = new Set(readdirSync(tmpRoot));
    await s('Page.navigate', { url: ORIGIN + '/' });
    await waitUntil(async () =>
      /Connected .*planning only/.test(await evaluate('document.getElementById("connection-chip").textContent')),
      15000, 'status chip at ' + w);
    const chip = await evaluate('document.getElementById("connection-chip").textContent');
    check(`status chip shows planning-connected at ${w}px`, /Connected .*planning only/.test(chip), chip);
    check(`Generate stays disabled at ${w}px`,
          await evaluate('document.getElementById("generate-btn").disabled') === true);
    check(`plan/export gated until status recognized (enabled now) at ${w}px`,
          await evaluate('!document.getElementById("btn-plan").disabled && !document.getElementById("btn-export").disabled') === true);
    await evaluate(`(function(){
      document.getElementById('prompt').value='integration fox';
      document.getElementById('prompt').dispatchEvent(new Event('input'));
      document.getElementById('seed').value='${maxSeed}';
      document.getElementById('seed').dispatchEvent(new Event('input'));
    })()`);
    await evaluate('document.getElementById("btn-plan").click()');
    await waitUntil(async () =>
      !await evaluate('document.getElementById("plan-panel").hidden'), 15000, 'plan panel at ' + w);
    check(`plan renders the full-range seed string at ${w}px`,
          await evaluate('document.getElementById("plan-seed").textContent') === maxSeed);
    const tokenText = await evaluate('document.getElementById("plan-tokens").textContent');
    check(`plan renders latent tokens at ${w}px`, tokenText === '4096', tokenText);
    const manifestText = await evaluate('document.getElementById("plan-manifest").textContent');
    check(`plan manifest text shows not-ready/verification-not-performed at ${w}px`,
          /verification not performed/.test(manifestText) && /not ready/.test(manifestText), manifestText);
    await evaluate('document.getElementById("btn-export").click()');
    await waitUntil(async () => downloads.slice(beforeDownloads)
                                      .some((d) => d.done), 15000, 'completed download at ' + w);
    const newDownloads = downloads.slice(beforeDownloads);
    check(`export triggered a NEW completed download at ${w}px`, newDownloads.length >= 1);
    // Verify the RAW bytes of a file NEWLY CREATED in this click's window
    // (never a leftover from an earlier geometry leg).
    const newFiles = readdirSync(tmpRoot).filter((f) => !filesBefore.has(f) &&
                                                       f.startsWith('qwen-editable-config'));
    let rawOk = false, rawDetail = 'no NEW downloaded file found';
    if (newFiles.length) {
      const bytes = readFileSync(join(tmpRoot, newFiles[0]), 'utf8');
      rawOk = bytes.includes('"seed": 18446744073709551615') &&
              !bytes.includes('output_dir') && !bytes.includes(tmpRoot);
      rawDetail = newFiles[0];
    }
    check(`downloaded export bytes carry the exact uint64 seed and no paths at ${w}px`,
          rawOk, rawDetail);
    for (const f of newFiles) { try { rmSync(join(tmpRoot, f), { force: true }); } catch {} }
    const overflow = await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1');
    check(`no horizontal overflow at ${w}px`, overflow === true);
  }

  // ------------------------------------------------ browser regressions
  // (real-Chrome scenarios from the review; all at desktop metrics)
  await s('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });

  // R1: an invalid draft entered WHILE a valid request is still consuming
  // its (deliberately delayed) body must win immediately — the late, now
  // STALE response must never land over the fresh errors.
  await evaluate(`(() => {
    window.__plainFetch = window.fetch;
    window.fetch = function (u, o) {
      const p = window.__plainFetch(u, o);
      if (String(u).indexOf('/api/v1/plan') < 0) return p;
      return p.then(function (res) {
        return new Response(new ReadableStream({
          start: function (c) {
            const r = res.body.getReader();
            (async function () {
              const first = await r.read();
              await new Promise(function (k) { setTimeout(k, 1500); });
              if (first.value) c.enqueue(first.value);
              for (;;) { const n = await r.read(); if (n.done) break; c.enqueue(n.value); }
              c.close();
            })().catch(function (e) { try { c.error(e); } catch {} });
            if (o && o.signal) o.signal.addEventListener('abort', function () {
              try { c.error(new DOMException('aborted', 'AbortError')); } catch {}
            });
          },
        }), { status: res.status, headers: res.headers });
      });
    };
  })()`);
  await evaluate(`(function(){
    var p = document.getElementById('prompt');
    p.value = 'delayed fox'; p.dispatchEvent(new Event('input'));
    var sd = document.getElementById('seed');
    sd.value = '42'; sd.dispatchEvent(new Event('input'));
    document.getElementById('btn-plan').click();            // attempt A: slow body
    sd.value = '${'9'.repeat(21)}'; sd.dispatchEvent(new Event('input'));
    document.getElementById('btn-plan').click();            // attempt B: invalid now
  })()`);
  const overlap = await evaluate(`({
    hidden: document.getElementById('plan-panel').hidden,
    seedErr: document.getElementById('seed-error').textContent,
    msg: document.getElementById('api-message').textContent })`);
  check('invalid draft during an in-flight valid request shows its errors immediately (controlled-backend fixture: delayed fetch override, not a real stalled network)',
        overlap.hidden === true && overlap.seedErr !== '' &&
        /local form errors/.test(overlap.msg), overlap.msg);
  await delay(2500);      // attempt A's body lands ~1.5 s in: seq is stale
  const lateLanding = await evaluate(`({
    hidden: document.getElementById('plan-panel').hidden,
    seedErr: document.getElementById('seed-error').textContent,
    msg: document.getElementById('api-message').textContent })`);
  check('the stale delayed response never clears the newer errors or renders (same fixture)',
        lateLanding.hidden === true && lateLanding.seedErr !== '',
        lateLanding.msg);
  await evaluate('window.fetch = window.__plainFetch;');
  await evaluate(`(function(){
    var sd = document.getElementById('seed');
    sd.value = '42'; sd.dispatchEvent(new Event('input'));
    document.getElementById('btn-plan').click();
  })()`);
  await waitUntil(async () => !await evaluate('document.getElementById("plan-panel").hidden'),
                  15000, 'plan panel after restoring plain fetch');
  check('UI fully functional again after the delayed-response scenario', true);

  // R2: a response whose BODY stalls past the deadline must surface as an
  // honest timeout — with a test-only REDUCED deadline so the whole check
  // stays bounded. Production FETCH_TIMEOUT_MS is never relied on here.
  await evaluate('window.QwenStudio.setFetchTimeoutForChecks(700);');
  await evaluate(`(() => {
    window.__plainFetch2 = window.fetch;
    window.fetch = function (u, o) {
      const p = window.__plainFetch2(u, o);
      if (String(u).indexOf('/api/v1/plan') < 0) return p;
      return p.then(function (res) {
        return new Response(new ReadableStream({
          start: function (c) {
            const r = res.body.getReader();
            r.then ? 0 : 0;
            (async function () { const f = await r.read(); if (f.value) c.enqueue(f.value); })()
              .catch(function (e) { try { c.error(e); } catch {} });
            if (o && o.signal) o.signal.addEventListener('abort', function () {
              try { c.error(new DOMException('aborted', 'AbortError')); } catch {}
            });
          },
        }), { status: res.status, headers: res.headers });
      });
    };
  })()`);
  await evaluate('document.getElementById("btn-plan").click()');
  await waitUntil(async () =>
    /timed out/i.test(await evaluate('document.getElementById("api-message").textContent')),
    8000, 'body-stall timeout surfaced');
  const stalled = await evaluate(`({
    chip: document.getElementById('connection-chip').textContent,
    msg: document.getElementById('api-message').textContent,
    hidden: document.getElementById('plan-panel').hidden })`);
  check('stalled response BODY past the (test-reduced) deadline = timeout, never a hang (controlled-backend fixture: stalled fetch override + test-only 700 ms deadline)',
        /timed out/i.test(stalled.msg) && stalled.hidden === true, stalled.msg);
  check('body timeout surfaces as an honest disconnect chip, not a success (same fixture)',
        /Not connected/.test(stalled.chip), stalled.chip);
  await evaluate('window.fetch = window.__plainFetch2; window.QwenStudio.setFetchTimeoutForChecks(15000);');
  await evaluate('document.getElementById("btn-check-config").click()');
  await waitUntil(async () =>
    /Connected .*planning only/.test(await evaluate('document.getElementById("connection-chip").textContent')),
    15000, 'reconnect after timeout');

  // R3: schema-rejected status JSON (valid HTTP, unusable document) is a
  // typed rejection — distinct from a disconnect — and keeps write gating.
  await evaluate(`(() => {
    window.__plainFetch3 = window.fetch;
    window.fetch = function (u, o) {
      if (String(u).indexOf('/api/v1/status') >= 0)
        return Promise.resolve(new Response('null',
          { status: 200, headers: { 'Content-Type': 'application/json' } }));
      return window.__plainFetch3(u, o);
    };
  })()`);
  await evaluate('window.QwenStudio.checkStatus()');
  const rejected = await evaluate(`({
    chip: document.getElementById('connection-chip').textContent,
    planDisabled: document.getElementById('btn-plan').disabled })`);
  check('schema-rejected status JSON is "Status rejected" (typed), not a disconnect',
        /Status rejected/.test(rejected.chip) && !/Not connected/.test(rejected.chip),
        rejected.chip);
  check('write actions stay gated after a schema rejection', rejected.planDisabled === true);
  await evaluate(`(() => {
    window.fetch = function (u, o) {
      if (String(u).indexOf('/api/v1/status') >= 0)
        return Promise.resolve(new Response('this is { not JSON',
          { status: 200, headers: { 'Content-Type': 'application/json' } }));
      return window.__plainFetch3(u, o);
    };
  })()`);
  await evaluate('window.QwenStudio.checkStatus()');
  const badJson = await evaluate(`({
    chip: document.getElementById('connection-chip').textContent,
    detail: document.getElementById('settings-backend').textContent })`);
  check('unparseable status body is also a typed rejection, detailed honestly',
        /Status rejected/.test(badJson.chip) && /not valid JSON|rejected/i.test(badJson.detail),
        badJson.chip + ' | ' + badJson.detail.slice(0, 80));
  await evaluate('window.fetch = window.__plainFetch3;');
  await evaluate('document.getElementById("btn-check-config").click()');
  await waitUntil(async () =>
    /Connected .*planning only/.test(await evaluate('document.getElementById("connection-chip").textContent')),
    15000, 'reconnect after schema rejection');

  // R4: Unicode and long (5000-char) prompts through the REAL form (client
  // bounds must match the server contract: no artificial 4000 limit).
  const uniPrompt = 'vulpes ü 🦊 日本語 ✨ fox';
  await evaluate(`(function(){
    var p = document.getElementById('prompt');
    p.value = ${JSON.stringify(uniPrompt)}; p.dispatchEvent(new Event('input'));
    document.getElementById('btn-plan').click();
  })()`);
  await waitUntil(async () => !await evaluate('document.getElementById("plan-panel").hidden'),
                  15000, 'unicode prompt plan');
  check('unicode prompt plans through the real UI (no client-side charset surprise)', true);
  const longPrompt = 'x'.repeat(5000);
  await evaluate(`(function(){
    var p = document.getElementById('prompt');
    p.value = ${JSON.stringify(longPrompt)}; p.dispatchEvent(new Event('input'));
    document.getElementById('btn-plan').click();
  })()`);
  await waitUntil(async () => !await evaluate('document.getElementById("plan-panel").hidden'),
                  15000, '5000-char prompt plan');
  check('5000-char prompt plans through the real UI (bounds match the server contract)', true);

  // The '   ' negative-prompt case through the REAL form: pyStrip must
  // mirror Python str.strip() (space included), so a plan must succeed and
  // CFG must render INACTIVE — a pyStrip without U+0020 would falsely
  // reject this server-valid document client-side.
  await evaluate(`(function(){
    var p = document.getElementById('prompt');
    p.value = 'space negative probe'; p.dispatchEvent(new Event('input'));
    var n = document.getElementById('negative');
    n.value = '   '; n.dispatchEvent(new Event('input'));
    var c = document.getElementById('cfg');
    c.value = '2'; c.dispatchEvent(new Event('input'));
    document.getElementById('btn-plan').click();
  })()`);
  await waitUntil(async () => !await evaluate('document.getElementById("plan-panel").hidden'),
                  15000, 'space-negative plan');
  const spacePlan = await evaluate(`({
    cfg: document.getElementById('plan-cfg').textContent,
    msg: document.getElementById('api-message').textContent })`);
  check('all-space negative + cfg 2 plans with CFG inactive (Python strip parity in the browser)',
        /inactive/.test(spacePlan.cfg) && spacePlan.msg === '',
        spacePlan.cfg + ' | ' + spacePlan.msg.slice(0, 80));

  // R5: an export response that is HTTP-successful but has NO readable
  // body must never download anything. Controlled-backend fixture: the
  // fetch override resolves a Response-like success envelope whose body
  // stream is absent — the exact shape that previously produced
  // Blob([undefined]) → a file containing the literal 'undefined' under a
  // success message. (Fixture, not a real stalled-network proof.)
  const beforeRejectedDownload = downloads.length;
  await evaluate(`(() => {
    window.__plainFetch4 = window.fetch;
    window.fetch = function (u, o) {
      if (String(u).indexOf('/api/v1/config/export') >= 0)
        return Promise.resolve({
          ok: true, status: 200, url: String(u),
          headers: new Headers({ 'Content-Type': 'application/json' }),
          body: null,   // NO readable stream: the typed 'unsupported' case
        });
      return window.__plainFetch4(u, o);
    };
  })()`);
  const draftPromptBefore = await evaluate('document.getElementById("prompt").value');
  await evaluate('document.getElementById("btn-export").click()');
  await waitUntil(async () =>
    /Export refused/.test(await evaluate('document.getElementById("api-message").textContent')),
    8000, 'no-body export refused');
  const rejectedExport = await evaluate(`({
    chip: document.getElementById('connection-chip').textContent,
    msg: document.getElementById('api-message').textContent,
    prompt: document.getElementById('prompt').value })`);
  check('no-body export success envelope is refused, never downloaded (controlled-backend fixture: fetch override without readable body)',
        /Export refused/.test(rejectedExport.msg) &&
        !/Downloaded/.test(rejectedExport.msg), rejectedExport.msg.slice(0, 90));
  check('no-body export is a typed schema rejection — distinct chip, not a fabricated disconnect (same fixture)',
        /Status rejected/.test(rejectedExport.chip) && !/Not connected/.test(rejectedExport.chip) &&
        /not a disconnect/.test(rejectedExport.msg), rejectedExport.chip);
  await delay(800);   // any spurious download would have begun by now
  check('no download was triggered by the refused export (same fixture)',
        downloads.length === beforeRejectedDownload,
        `new downloads: ${downloads.length - beforeRejectedDownload}`);
  check('draft is unchanged by the refused export (same fixture)',
        rejectedExport.prompt === draftPromptBefore, rejectedExport.prompt.slice(0, 40));
  await evaluate('window.fetch = window.__plainFetch4;');
  await evaluate('document.getElementById("btn-check-config").click()');
  await waitUntil(async () =>
    /Connected .*planning only/.test(await evaluate('document.getElementById("connection-chip").textContent')),
    15000, 'reconnect after refused export');
  const beforeRestoredDownload = downloads.length;
  await evaluate('document.getElementById("btn-export").click()');
  await waitUntil(async () => downloads.slice(beforeRestoredDownload)
                                    .some((d) => d.done), 15000, 'export after reconnect');
  check('export works again after the refused-response scenario (real server bytes)',
        downloads.slice(beforeRestoredDownload).length >= 1,
        `new downloads: ${downloads.length - beforeRestoredDownload}`);

  // R6: parsePlan must refuse an echoed config.prompt that is empty or
  // Python-whitespace-only — EXACTLY shared GenerationConfig
  // (config.py: `not prompt.strip()`). Runs against the REAL served plan
  // echo in the same pass: no fetch stub, no DOM stub, no extra browser.
  const blankPrompt = await evaluate(`(async function () {
    var build = window.QwenStudio.buildEnvelope();
    var res = await window.fetch('/api/v1/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Qwen-UI': '1' },
      cache: 'no-store',
      body: JSON.stringify(build.envelope)
    });
    var doc = await res.json();
    var baseline = window.QwenStudio.parsePlan(doc) !== null;
    var blanks = ['', ' ', '\\t\\n\\u000b\\u000c\\r', ' \\u00a0 ',
                  '\\u2028\\u2029', '\\u3000', '\\u0085\\u001c\\u001f'];
    var accepted = [];
    for (var i = 0; i < blanks.length; i++) {
      var d = JSON.parse(JSON.stringify(doc));
      d.config.prompt = blanks[i];
      if (window.QwenStudio.parsePlan(d) !== null) accepted.push(JSON.stringify(blanks[i]));
    }
    var d2 = JSON.parse(JSON.stringify(doc));
    d2.config.prompt = '\\u200b';    // zero-width space: NOT Python whitespace
    var zwspAccepted = window.QwenStudio.parsePlan(d2) !== null;
    var d3 = JSON.parse(JSON.stringify(doc));
    d3.config.prompt = 'a\\tb';      // interior whitespace must be preserved
    var interiorAccepted = window.QwenStudio.parsePlan(d3) !== null;
    return { baseline: baseline, accepted: accepted,
             zwspAccepted: zwspAccepted, interiorAccepted: interiorAccepted };
  })()`);
  check('parsePlan rejects empty/Python-whitespace-only echoed prompt (real served plan echo, corrected strip predicate)',
        blankPrompt.baseline === true && blankPrompt.accepted.length === 0,
        'accepted blanks: ' + blankPrompt.accepted.join(' '));
  check('parsePlan keeps ZWSP and interior-whitespace prompts (exact Python strip parity, never over-strips)',
        blankPrompt.zwspAccepted === true && blankPrompt.interiorAccepted === true,
        JSON.stringify(blankPrompt));

  // Reviewed metadata policy is exactly schema-v1 with a nonempty artifact
  // set. Use real served status/plan documents, then mutate only metadata.
  const metadataPolicy = await evaluate(`(async function () {
    var status = await (await fetch('/api/v1/status', { cache: 'no-store' })).json();
    var build = window.QwenStudio.buildEnvelope();
    var plan = await (await fetch('/api/v1/plan', {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Qwen-UI': '1' },
      body: JSON.stringify(build.envelope), cache: 'no-store'
    })).json();
    var accepted = [];
    [status, plan].forEach(function (original, index) {
      ['version', 'empty'].forEach(function (fault) {
        var doc = JSON.parse(JSON.stringify(original));
        if (fault === 'version') doc.manifest.schema_version = 2;
        else doc.manifest.artifact_count = 0;
        var parser = index === 0 ? window.QwenStudio.parseStatus : window.QwenStudio.parsePlan;
        if (parser(doc) !== null) accepted.push(index + ':' + fault);
      });
    });
    return { baseline: window.QwenStudio.parseStatus(status) !== null &&
                       window.QwenStudio.parsePlan(plan) !== null, accepted: accepted };
  })()`);
  check('status/plan refuse unreviewed manifest schema and empty valid artifact set',
        metadataPolicy.baseline && metadataPolicy.accepted.length === 0, JSON.stringify(metadataPolicy));

  // A semantically coherent echo for OTHER settings must not be labelled
  // authoritative validation of our submitted snapshot. This is a controlled
  // backend response mutation in real Chrome, not a server generation run.
  await evaluate(`(function () {
    window.__echoPlainFetch = window.fetch;
    window.fetch = function (url, options) {
      return window.__echoPlainFetch(url, options).then(function (res) {
        if (String(url) !== '/api/v1/plan' || !res.ok) return res;
        return res.json().then(function (doc) {
          doc.config.prompt = 'different but valid prompt';
          window.__coherentDifferentPlan = window.QwenStudio.parsePlan(doc) !== null;
          return new Response(JSON.stringify(doc), { status: 200,
            headers: { 'Content-Type': 'application/json' } });
        });
      });
    };
    document.getElementById('btn-plan').click();
  })()`);
  await waitUntil(async () => /submitted configuration/.test(await evaluate(
    'document.getElementById("api-message").textContent')), 15000, 'mismatched plan echo refusal');
  const mismatched = await evaluate(`({ coherent: window.__coherentDifferentPlan,
    hidden: document.getElementById('plan-panel').hidden,
    chip: document.getElementById('connection-chip').textContent,
    disabled: document.getElementById('btn-plan').disabled,
    prompt: document.getElementById('prompt').value })`);
  check('coherent plan for a different request is rejected without modifying the draft (controlled response mutation)',
        mismatched.coherent === true && mismatched.hidden && mismatched.disabled &&
        mismatched.chip === 'Status rejected' && mismatched.prompt !== 'different but valid prompt',
        JSON.stringify(mismatched));
  await evaluate('window.fetch = window.__echoPlainFetch; window.QwenStudio.checkStatus();');
  await waitUntil(async () => !await evaluate('document.getElementById("btn-plan").disabled'),
                  15000, 'reconnect after mismatched echo');
  await evaluate('document.getElementById("btn-plan").click();');
  await waitUntil(async () => !await evaluate('document.getElementById("plan-panel").hidden'),
                  15000, 'matching real plan after restoring backend');
  check('matching real-server plan works again after request-echo refusal', true);

  // every network-origin request the page made was same-origin (CSP
  // connect-src 'self' holds in practice; data:/blob: URLs are local)
  const foreign = netUrls.filter((n) => /^https?:/i.test(n.url) && !n.url.startsWith(ORIGIN + '/'));
  check('no third-party network requests were made', foreign.length === 0,
        foreign.map((f) => f.url).slice(0, 3).join(' '));
  check('no uncaught page exceptions', pageExceptions.length === 0,
        pageExceptions.slice(0, 2).join(' | '));
}

// ---------------------------------------------------------------- run
let exitCode = 1;
let startupDone = false;
let timedOut = false;
const guard = watchdog(WHOLE_CHECK_MS, 'whole integration check');
guard.catch((err) => {
  stopped = true;                          // freeze late results
  timedOut = true;
  forceFail('whole-run watchdog aborted the integration check',
            String((err && err.message) || err).slice(0, 160));
});
try {
  // STARTUP is covered by the whole-check deadline too (previously it was
  // awaited BEFORE any watchdog existed — a hung start orphaned the child).
  const srv = await Promise.race([startServer(), guard]);
  port = srv.port;
  ORIGIN = `http://127.0.0.1:${port}`;
  startupDone = true;
  console.log(`server origin: ${ORIGIN}`);
  const runner = (async () => { await httpChecks(); await chromeChecks(); })();
  runner.catch(() => { /* after a watchdog abort, late failures are expected */ });
  await Promise.race([runner, guard]);
  // timedOut runs are failed BY DEFINITION (the forceFail above guarantees
  // failed > 0); never print a clean N/N after an abort.
  exitCode = (failed === 0 && !timedOut) ? 0 : 1;
} catch (err) {
  if (!startupDone && !/exceeded its/.test(String(err && err.message))) {
    // never announced an origin — environment problem (child group, if any,
    // is still cleaned in finally)
    console.error('integration: server could not start —', err.message);
    exitCode = 2;
  } else {
    forceFail('integration script completed without error',
              String((err && err.stack) || err).slice(0, 160));
    exitCode = 1;
  }
} finally {
  guard.cancel();
  // OUTERMOST cleanup, in order: OUR Chrome (bounded, awaited — an aborted
  // chromeChecks can never leave it running), THEN the server group, THEN
  // the temp tree. Nothing here touches any other process.
  let chromeCleaned = true;
  try {
    if (await cleanupChrome() === false) {
      chromeCleaned = false;
      exitCode = 1;
      forceFail('Chrome owner cleanup pending; retained owner and fixture tree', 'no owner KILL/false cleanup success');
    }
  } catch (e) {
    chromeCleaned = false;
    exitCode = 1;
    forceFail('Chrome owner cleanup failed; retained fixture tree', e.message);
  }
  console.log(results.join('\n'));
  console.log(`\n${passed}/${passed + failed} executed integration checks passed` +
              (notRun ? ` — ${notRun} NOT RUN (see above, not counted in the total)` : ''));
  await killGroup(child);
  if (chromeCleaned) {
    try { rmSync(tmpRoot, { recursive: true, force: true, maxRetries: 5 }); } catch {}
  }
}
process.exitCode = exitCode; // never orphan an unfinished ownership wrapper
