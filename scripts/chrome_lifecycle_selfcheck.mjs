#!/usr/bin/env node
/* Pure protocol self-check for scripts/chrome_lifecycle.mjs.
 *
 * NO Chrome is launched and NO DOM is faked: it exercises only the pure
 * promise/timeout protocol of the shared helpers with small fixtures, plus
 * a documented source check for the one upstream fact (zygote + sandbox is
 * an explicitly-exiting combination, so --no-zygote must not appear).
 * Real-browser acceptance is a separate command; this check never claims it.
 *
 * Usage: node scripts/chrome_lifecycle_selfcheck.mjs   (exits 0/1)
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { DEFAULTS, makeCdp, shutdownChrome, launchChrome } from './chrome_lifecycle.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const results = [];
function check(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => results.push(['PASS', name]))
    .catch((err) => results.push(['FAIL', name + ': ' + (err && err.message)]));
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }

// Tighten the protocol timeouts so the pure checks run fast (no sleeps).
DEFAULTS.sendMs = 100;
DEFAULTS.browserCloseMs = 60;
DEFAULTS.childExitMs = 60;

class FakeSocket {
  constructor() { this.listeners = {}; this.sent = []; this.throwOnSend = false; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  emit(type, ev) {
    if (type === 'message' && typeof this.onmessage === 'function') {
      this.onmessage(ev);
      return;
    }
    for (const fn of this.listeners[type] || []) fn(ev || {});
  }
  send(text) {
    if (this.throwOnSend) throw new Error('socket not writable');
    this.sent.push(JSON.parse(text));
  }
}

await check('send timeout rejects and marks dead (no dangling pending)', async () => {
  const ws = new FakeSocket();
  const cdp = makeCdp(ws);
  let rejected = false;
  await cdp.send('Never.answered').catch(() => { rejected = true; });
  assert(rejected, 'timed-out send must reject');
  assert(cdp.dead === true, 'timeout must mark the socket dead');
  await cdp.send('Anything').catch(() => {});   // dead socket: rejects fast
});

await check('websocket close rejects every pending immediately', async () => {
  const ws = new FakeSocket();
  const cdp = makeCdp(ws);
  const inflight = cdp.send('Long.running');
  let rejected = null;
  const witness = inflight.catch((err) => { rejected = err.message; });
  ws.emit('close');
  await witness;
  assert(cdp.dead === true, 'close must mark the socket dead');
  assert(/closed/.test(rejected || ''), 'pending must reject with the close cause');
});

await check('synchronous send failure clears the pending entry', async () => {
  const ws = new FakeSocket();
  ws.throwOnSend = true;
  const cdp = makeCdp(ws);
  let rejected = null;
  await cdp.send('Sync.fail').catch((err) => { rejected = err.message; });
  assert(/synchronously/.test(rejected || ''), 'typed synchronous rejection');
  assert(cdp.dead === false, 'a send failure is not a socket death');
  ws.throwOnSend = false;
  const p = cdp.send('Page.enable');
  const sent = ws.sent[ws.sent.length - 1];
  assert(sent.method === 'Page.enable', 'next send proceeds normally');
  ws.emit('message', { data: JSON.stringify({ id: sent.id, result: {} }) });
  await p;
});

await check('shutdownChrome signals only its own spawned group, bounded', async () => {
  const calls = [];
  const originalKill = process.kill;
  process.kill = (pid, sig) => {
    calls.push([pid, sig]);
    if (sig === 'SIGTERM') setTimeout(() => chrome.emit('exit'), 5);
  };
  const listeners = {};
  const chrome = {
    pid: 424242, exitCode: null, signalCode: null,
    once(ev, fn) { (listeners[ev] ||= []).push(fn); },
    unref() {},
    emit(ev) { for (const fn of listeners[ev] || []) fn(); chrome.exitCode = 0; },
  };
  try {
    await shutdownChrome(chrome, null, null);   // no ws: straight to signals
    assert(calls.length >= 1, 'group signal expected');
    assert(calls[0][0] === -424242, 'must signal the process GROUP (-pid)');
    assert(!calls.some(([p]) => p === 424242), 'never the bare child pid');
  } finally {
    process.kill = originalKill;
  }
});

await check('early child exit / spawn error rejects startup (bounded)', async () => {
  const { wsUrl } = launchChrome({ chromePath: 'qwen-no-such-binary-xyz',
                                   profileDir: '/tmp/qwen-selfcheck-profile' });
  let rejected = false;
  await wsUrl.catch(() => { rejected = true; });
  assert(rejected, 'spawn failure must reject the wsUrl promise');
});

await check('source check: no --no-zygote with sandbox defaults (upstream exits)', () => {
  const src = readFileSync(path.join(here, 'chrome_lifecycle.mjs'), 'utf8');
  assert(!src.includes("'--no-zygote'") && !src.includes('"--no-zygote"'),
         '--no-zygote must not be part of the defaults');
  assert(/content_main_runner_impl\.cc/.test(src),
         'the upstream fact stays documented at the argument list');
});

let failed = 0;
for (const [status, line] of results) {
  if (status === 'FAIL') failed++;
  console.log(`${status} ${line}`);
}
console.log('NOTE: PURE protocol checks only — run chrome_browser_check.mjs ' +
            'and web_integration_check.mjs separately for real-Chrome evidence.');
process.exit(failed ? 1 : 0);
