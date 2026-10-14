/* Shared, bounded real-Chrome lifecycle helpers for the standalone browser
 * check scripts (chrome_browser_check.mjs, web_integration_check.mjs).
 *
 * Real Chrome is separate from the guarded main suite. Node built-ins plus
 * the existing Python interpreter run a dedicated Linux ownership wrapper.
 *
 * Contract (2026-10-12 root review):
 *   * protocol waits are bounded: the WebSocket open, EVERY CDP send and the
 *     whole check each carry a finite timeout;
 *   * an early child exit rejects startup instead of waiting on a DevTools
 *     port that will never be announced;
 *   * cleanup signals only OUR dedicated ownership wrapper. It adopts and
 *     reaps Chrome grandchildren; never KILL/unref this owner first. A
 *     kernel-unreapable child keeps ownership pending, not false success;
 *   * Chrome runs with its own sandbox ENABLED by default — --no-sandbox is
 *     added only when the environment explicitly asks (QWEN_CHROME_NO_SANDBOX=1);
 *   * capacity preflight only REPORTS free cgroup pids slots — it does not
 *     claim a verified cause (no procstat diagnostics here).
 */
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const DEFAULTS = {
  devtoolsAnnounceMs: 45000,
  wsOpenMs: 15000,
  sendMs: 30000,
  browserCloseMs: 8000,
  childExitMs: 10000,
  ownerExitMs: 8000,
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
 * on the bounded announce timeout. The detached child is a private ownership
 * wrapper, not Chrome itself; signal only its PID to request scoped cleanup.
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
  // A dedicated Linux subreaper owns/adopts/waits Chrome grandchildren.
  // Never rely on the container's PID1 to reap orphaned browser descendants.
  const owner = fileURLToPath(new URL('./process_lineage_owner.py', import.meta.url));
  const python = process.env.QWEN_PYTHON || process.env.PYTHON || 'python3';
  const chrome = spawn(python, ['-B', owner, '--', chromePath, ...args],
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
    abort() { killAll('aborted'); },
    send(method, params = {}, sessionId) {
      if (cdp.dead) return Promise.reject(new Error('cdp socket is dead (previous timeout)'));
      return new Promise((res, rej) => {
        const id = nextId++;
        const timeoutMs = DEFAULTS.sendMs;
        const timer = setTimeout(() => {
          // One timed-out exchange makes the whole connection unusable.
          // Reject/clear every other pending exchange immediately too.
          killAll(`CDP send ${method} timed out after ${timeoutMs} ms`);
        }, timeoutMs);
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

async function boundedWait(promise, ms, timeoutValue) {
  let timer;
  try {
    return await Promise.race([
      promise, new Promise((res) => { timer = setTimeout(() => res(timeoutValue), ms); }),
    ]);
  } finally {
    clearTimeout(timer); // a completed child must not leave a live deadline behind
  }
}

/**
 * Cleanup, scoped to OUR OWN spawned child only (never a broad kill):
 *  1. Browser.close, then wait for our wrapper to reap the whole lineage;
 *  2. bounded fallback: TERM only the wrapper PID; it owns descendant cleanup;
 *  3. pending ownership returns false, retains refs/profile and forbids KILL
 *     or explicit parent process.exit before the owner is finished.
 */
export async function shutdownChrome(chrome, ws, cdp) {
  if (chrome && ws && ws.readyState === 1 && cdp && !cdp.dead) {
    try {
      await boundedWait(cdp.send('Browser.close'), DEFAULTS.browserCloseMs, false);
    } catch { /* fall through to the signal path */ }
  }
  cdp?.abort?.(); // terminate any remaining own protocol waits even without a close event
  try { ws?.close(); } catch {}
  if (!chrome) return;
  const exited = new Promise((res) => {
    if (chrome.exitCode !== null || chrome.signalCode !== null) res(true);
    else chrome.once('exit', () => res(true));
  });
  const gotExit = await boundedWait(exited, DEFAULTS.childExitMs, false);
  if (!gotExit && Number.isInteger(chrome.pid)) {
    // Signal ONLY the owner; it performs scoped TERM/KILL on its children
    // and stays alive until they are reaped. Killing the owner/group would
    // recreate the PID1 zombie leak this wrapper prevents.
    try { process.kill(chrome.pid, 'SIGTERM'); } catch { /* already gone */ }
    const gone = await boundedWait(exited, DEFAULTS.ownerExitMs, false);
    if (!gone) {
      chrome.cleanupPending = true;
      return false; // retain owner refs, no SIGKILL/unref/false cleanup success
    }
  }
  return true;
}
