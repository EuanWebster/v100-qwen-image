/* Shared, bounded real-Chrome lifecycle helpers for the standalone browser
 * check scripts (chrome_browser_check.mjs, web_integration_check.mjs).
 *
 * NOT part of the guarded main suite; Node built-ins only, no dependencies.
 *
 * Contract (2026-10-12 root review):
 *   * the script must NEVER hang: the WebSocket open, EVERY CDP send and the
 *     whole check each carry a finite timeout;
 *   * an early child exit rejects startup instead of waiting on a DevTools
 *     port that will never be announced;
 *   * cleanup is scoped to OUR OWN spawned child only: Browser.close over the
 *     still-open WebSocket first, then waiting for the child to exit; the
 *     bounded fallback signals our spawned process GROUP with
 *     process.kill(-child.pid, sig) (child.spawned detached: true).
 *     child.kill(sig, true) does NOT signal a group and must not be used;
 *   * Chrome runs with its own sandbox ENABLED by default — --no-sandbox is
 *     added only when the environment explicitly asks (QWEN_CHROME_NO_SANDBOX=1);
 *   * capacity preflight only REPORTS free cgroup pids slots — it does not
 *     claim a verified cause (no procstat diagnostics here).
 */
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';

export const DEFAULTS = {
  devtoolsAnnounceMs: 45000,
  wsOpenMs: 15000,
  sendMs: 30000,
  browserCloseMs: 8000,
  childExitMs: 10000,
};

/** Read the cgroup v2 pids controller, if readable. Never throws. */
export function cgroupCapacity() {
  try {
    const max = Number(readFileSync('/sys/fs/cgroup/pids.max', 'utf8').trim());
    const cur = Number(readFileSync('/sys/fs/cgroup/pids.current', 'utf8').trim());
    if (Number.isFinite(max) && Number.isFinite(cur)) {
      return { ok: max - cur >= 64, free: max - cur, detail: `${cur}/${max}` };
    }
  } catch { /* not a readable cgroup v2 box — Chrome itself will speak */ }
  return null;
}

/**
 * Launch our own Chrome child and resolve its DevTools WebSocket URL.
 * Rejects on spawn error, on an early child exit (stderr tail attached), or
 * on the bounded announce timeout. The child is detached so cleanup can
 * signal the whole group; we only ever signal OUR OWN spawned pid group.
 */
export function launchChrome({ chromePath, profileDir, extraArgs = [] }) {
  const args = [
    '--remote-debugging-port=0',
    `--user-data-dir=${profileDir}`,
    '--no-first-run', '--no-default-browser-check',
    '--disable-gpu', '--disable-dev-shm-usage', '--disable-crash-reporter',
    '--disable-breakpad', '--disable-background-networking', '--disable-sync',
    // NO --no-zygote: per upstream content_main_runner_impl.cc the
    // zygote-disabled + sandbox-enabled combination EXPLICITLY exits —
    // normal Chrome defaults stay as they are.
    '--renderer-process-limit=1', '--num-raster-threads=1',
    '--disable-features=AudioServiceOutOfProcess', '--mute-audio',
    // Sandbox stays ENABLED by default; only an explicit environment
    // request disables it (never by default, per policy).
    ...(process.env.QWEN_CHROME_NO_SANDBOX === '1' ? ['--no-sandbox'] : []),
    ...extraArgs,
    'about:blank',
  ];
  // stdout is IGNORED (the child can never block writing it; we parse only
  // stderr), stderr is capped to a rolling 16 KiB tail (bounded memory).
  const chrome = spawn(chromePath, args,
    { stdio: ['ignore', 'ignore', 'pipe'], detached: true });
  const wsUrl = new Promise((res, rej) => {
    let settled = false;
    const fail = (err) => { if (!settled) { settled = true; clearTimeout(t); rej(err); } };
    const t = setTimeout(
      () => fail(new Error(`chrome did not announce a DevTools port within ${DEFAULTS.devtoolsAnnounceMs} ms`)),
      DEFAULTS.devtoolsAnnounceMs);
    let errBuf = '';
    chrome.stderr?.on('data', (b) => {
      errBuf = (errBuf + String(b)).slice(-16384);   // bounded tail only
      const m = /DevTools listening on (ws:\/\/\S+)/.exec(errBuf);
      if (m && !settled) { settled = true; clearTimeout(t); res(m[1]); }
    });
    chrome.on('error', (e) => fail(e));
    // EARLY child exit rejects startup (bounded: never waits for the port).
    chrome.on('exit', (code, signal) =>
      fail(new Error(`chrome exited early (code=${code} signal=${signal}); stderr tail: ${errBuf.slice(-500)}`)));
  });
  return { chrome, wsUrl };
}

/** Open the CDP WebSocket with a finite timeout (ws.onerror also rejects). */
export function openCdpSocket(wsUrl) {
  const ws = new WebSocket(wsUrl);
  const ready = new Promise((res, rej) => {
    const t = setTimeout(() => { try { ws.close(); } catch {}
      rej(new Error(`websocket open timed out after ${DEFAULTS.wsOpenMs} ms`)); },
      DEFAULTS.wsOpenMs);
    ws.onopen = () => { clearTimeout(t); res(); };
    ws.onerror = (ev) => { clearTimeout(t); rej(new Error(`websocket error: ${ev?.type || 'error'}`)); };
  });
  return { ws, ready };
}

/** Minimal CDP client: EVERY send has a finite timeout; events are routed to
 *  an optional listener. A timed-out send rejects and marks the socket dead. */
export function makeCdp(ws) {
  let nextId = 1;
  const pending = new Map();
  const killAll = (why) => {
    // A closed/errored socket must reject ALL pending and clear their
    // timers immediately - no stale entries left dangling.
    cdp.dead = true;
    for (const { rej, timer } of pending.values()) {
      clearTimeout(timer);
      rej(new Error(`cdp socket ${why}`));
    }
    pending.clear();
  };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.id && pending.has(msg.id)) {
      const { res, rej, timer } = pending.get(msg.id);
      pending.delete(msg.id);
      clearTimeout(timer);
      msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
    } else if (msg.method && cdp.onEvent) {
      cdp.onEvent(msg);
    }
  };
  const cdp = {
    onEvent: null,
    dead: false,
    send(method, params = {}, sessionId) {
      if (cdp.dead) return Promise.reject(new Error('cdp socket is dead (previous timeout)'));
      return new Promise((res, rej) => {
        const id = nextId++;
        const timer = setTimeout(() => {
          pending.delete(id);
          cdp.dead = true;   // never leave a dangling request: the run is over
          rej(new Error(`CDP send ${method} timed out after ${DEFAULTS.sendMs} ms`));
        }, DEFAULTS.sendMs);
        pending.set(id, { res, rej, timer });
        try {
          ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
        } catch (err) {
          // A SYNCHRONOUS send failure clears the pending entry and its
          // timer right away, not leaving a stale timeout behind.
          pending.delete(id);
          clearTimeout(timer);
          rej(new Error(`CDP send ${method} failed synchronously: ${err?.message || err}`));
        }
      });
    },
  };
  ws.addEventListener('close', () => killAll('closed'));
  ws.addEventListener('error', () => killAll('errored'));
  return cdp;
}

/** Whole-check watchdog: rejects after the bound (callers must cleanup). */
export function watchdog(ms, label = 'whole check') {
  let timer;
  const promise = new Promise((_, rej) => {
    timer = setTimeout(() => rej(new Error(`${label} exceeded its ${ms} ms bound`)), ms);
  });
  promise.cancel = () => clearTimeout(timer);
  return promise;
}

/**
 * Cleanup, scoped to OUR OWN spawned child only (never a broad kill):
 *  1. Browser.close over the live WebSocket (Chrome reaps its own children);
 *  2. wait (bounded) for the child to exit;
 *  3. bounded fallback: signal our spawned process GROUP — process.kill(-pid)
 *     with the child spawned detached. child.kill(sig, true) does NOT do
 *     this and is deliberately not used.
 */
export async function shutdownChrome(chrome, ws, cdp) {
  if (chrome && ws && ws.readyState === 1 && cdp && !cdp.dead) {
    try {
      await Promise.race([
        cdp.send('Browser.close'),
        new Promise((_, rej) => setTimeout(() => rej(new Error('Browser.close timed out')), DEFAULTS.browserCloseMs)),
      ]);
    } catch { /* fall through to the signal path */ }
  }
  try { ws?.close(); } catch {}
  if (!chrome) return;
  const exited = new Promise((res) => {
    if (chrome.exitCode !== null || chrome.signalCode !== null) res(true);
    else chrome.once('exit', () => res(true));
  });
  try { chrome.unref(); } catch {}
  const gotExit = await Promise.race([
    exited,
    new Promise((res) => setTimeout(() => res(false), DEFAULTS.childExitMs)),
  ]);
  if (!gotExit && Number.isInteger(chrome.pid)) {
    for (const sig of ['SIGTERM', 'SIGKILL']) {
      try { process.kill(-chrome.pid, sig); } catch { /* already gone */ }
      const gone = await Promise.race([
        exited,
        new Promise((res) => setTimeout(() => res(false), 3000)),
      ]);
      if (gone) break;
    }
  }
}
