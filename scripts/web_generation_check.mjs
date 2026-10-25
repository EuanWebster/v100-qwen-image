#!/usr/bin/env node
/* Real Chrome W3 check. Default: explicit CPU fixture; --real: actual local
 * FP16 engine, V100 and model bundle. Owns only its temporary server/browser.
 * Artifacts remain under the printed /tmp directory for inspection.
 */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdirSync, mkdtempSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {launchChrome, openCdpSocket, makeCdp, shutdownChrome} from './chrome_lifecycle.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.QWEN_PYTHON || join(root, '.venv/bin/python');
const real = process.argv.includes('--real');
const artifactDir = mkdtempSync(join(tmpdir(), 'qwen-w3-'));
const output = join(artifactDir, 'output');
mkdirSync(output);
console.log(`Mode: ${real ? 'REAL V100 / Qwen weights' : 'CPU fixture (synthetic test images)'}`);
console.log(`Artifacts: ${artifactDir}`);
const args = real ? ['-m', 'qwen_engine.web', '--port', '0', '--enable-generation',
  '--attest-unbounded-host', '--manifest', join(root, 'models/manifest.json'),
  '--models-dir', join(root, 'models/assets'), '--output-dir', output, '--cache-dir', join(root, 'cache')]
  : [join(root, 'tests/fixtures/web_generation_server.py'), output];
const server = spawn(python, args, {cwd: root, stdio: ['ignore', 'pipe', 'pipe']});
let log = '', chrome, ws, cdp;
let failed = false;
let aborted = false;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
let fatal;
const deadline = new Promise((_, reject) => { fatal = setTimeout(() => reject(new Error('whole W3 check timed out')), real ? 360000 : 90000); });
async function check() {
  const origin = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('server start timed out')), 10000);
    server.stdout.on('data', chunk => {
      log = (log + chunk).slice(-65536);
      const match = /listening on (http:\/\/127\.0\.0\.1:\d+)/.exec(log);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
    server.on('error', reject);
    server.on('exit', code => { clearTimeout(timer); reject(new Error(`server exited ${code}: ${log}`)); });
  });
  server.stderr.on('data', chunk => { log = (log + chunk).slice(-65536); });
  console.log(origin);
  const launched = launchChrome({chromePath: process.env.CHROME_PATH || '/opt/google/chrome/chrome',
    profileDir: join(artifactDir, 'chrome'), extraArgs: ['--headless=new']});
  chrome = launched.chrome;
  const wsUrl = await launched.wsUrl;
  if (aborted) throw new Error('check aborted');
  const socket = openCdpSocket(wsUrl);
  ws = socket.ws;
  await socket.ready;
  cdp = makeCdp(ws);
  const {targetId} = await cdp.send('Target.createTarget', {url: 'about:blank'});
  const {sessionId} = await cdp.send('Target.attachToTarget', {targetId, flatten: true});
  const send = (method, params = {}) => cdp.send(method, params, sessionId);
  await send('Runtime.enable');
  await send('Page.enable');
  const exceptions = [];
  cdp.onEvent = event => { if (event.method === 'Runtime.exceptionThrown') exceptions.push(event.params); };
  async function evaluate(expression) {
    if (aborted) throw new Error('check aborted');
    const result = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || 'page exception');
    return result.result.value;
  }
  async function until(expression, timeout = 15000) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      if (aborted) throw new Error('check aborted');
      try { if (await evaluate(expression)) return; } catch { /* navigation not ready */ }
      await delay(200);
    }
    throw new Error(`timed out: ${expression}\n${log}`);
  }
  async function api(path, config) {
    return evaluate(`fetch(${JSON.stringify(path)}, ${config === undefined ? '{}' : JSON.stringify({
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Qwen-UI': '1'}, body: JSON.stringify(config)
    })}).then(async r => ({status:r.status, body:await r.json()}))`);
  }
  const jobIds = async () => (await api('/api/v1/jobs')).body.jobs;
  await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  await send('Page.navigate', {url: origin + '/'});
  await until(`document.getElementById('generate-btn') && !document.getElementById('generate-btn').disabled`);
  console.log('PASS generation-enabled status recognized');
  for (const [id, values] of [['quant',['fp16','q8']], ['offload',['resident','streamed','offload']], ['attention',['auto','sdpa','efficient','eager']]]) {
    for (const value of values) {
      const help = await evaluate(`(() => {
        const c=document.getElementById('${id}'); c.value='${value}'; c.dispatchEvent(new Event('change'));
        return {title:c.title,help:document.getElementById('${id}-help').textContent,description:c.getAttribute('aria-describedby')};
      })()`);
      assert(help.title.length > 30);
      assert.equal(help.help,help.title);
      assert.equal(help.description,id+'-help');
    }
  }
  await evaluate(`for (const [id,value] of [['quant','fp16'],['offload','resident'],['attention','auto']]) {const c=document.getElementById(id);c.value=value;c.dispatchEvent(new Event('change'));}`);
  console.log('PASS option explanations match each selection and are accessible without hover');
  assert.equal(await evaluate(`document.getElementById('output-format').value`), 'jpg');
  const imported = await evaluate(`QwenStudio.importConfigFile(new File([
    '{"prompt":"imported cube","seed":18446744073709551615,"output_format":"webp","quant":"q8"}'
  ], 'config.json', {type:'application/json'}))`);
  assert.equal(imported, true);
  assert.equal(await evaluate(`document.getElementById('seed').value`), '18446744073709551615');
  assert.equal(await evaluate(`document.getElementById('output-format').value`), 'webp');
  assert(await evaluate(`document.getElementById('quant-help').textContent.includes('Compresses the model')`));
  assert.equal((await jobIds()).length, 0);
  assert.equal(await evaluate(`(async () => {
    const pending=QwenStudio.importConfigFile(new File(['{"prompt":"late import"}'],'config.json'));
    document.getElementById('prompt').value='newer draft';
    document.getElementById('prompt').dispatchEvent(new Event('input'));
    return await pending;
  })()`), false);
  assert.equal(await evaluate(`document.getElementById('prompt').value`), 'newer draft');
  await evaluate(`document.getElementById('output-format').value='jpg'`);
  for (const theme of ['original', 'neon', 'paper']) {
    await evaluate(`document.getElementById('theme-select').value='${theme}'; document.getElementById('theme-select').dispatchEvent(new Event('change'))`);
    assert.equal(await evaluate(`document.documentElement.dataset.theme`), theme);
    const shot = await send('Page.captureScreenshot', {format:'png'});
    writeFileSync(join(artifactDir, 'theme-' + theme + '.png'), Buffer.from(shot.data,'base64'));
  }
  await send('Page.reload');
  await until(`document.getElementById('generate-btn') && !document.getElementById('generate-btn').disabled`);
  assert.equal(await evaluate(`document.documentElement.dataset.theme`), 'paper');
  console.log('PASS raw configuration import preserves uint64 seed, does not generate; three themes persist');
  const catalog = [
    ['square',1024,1024], ['fourthree',1152,864], ['threefour',864,1152],
    ['threetwo',1248,832], ['twothree',832,1248], ['landscape',1344,768], ['portrait',768,1344],
    ['square2k',2048,2048], ['fourthree2k',2400,1792], ['threefour2k',1792,2400],
    ['threetwo2k',2528,1696], ['twothree2k',1696,2528],
    ['landscape2k',2752,1536], ['portrait2k',1536,2752]
  ];
  for (const [preset, width, height] of catalog) {
    const result = await evaluate(`(() => {
      document.querySelector('[data-preset="${preset}"]').click();
      return {config:QwenStudio.formSnapshot(),pressed:document.querySelector('[data-preset="${preset}"]').getAttribute('aria-pressed')};
    })()`);
    assert.equal(result.config.width,width);
    assert.equal(result.config.height,height);
    assert.equal(result.config.vae_tiled_decode,preset.endsWith('2k'));
    assert.equal(result.pressed,'true');
  }
  await evaluate(`document.querySelector('[data-preset="square"]').click()`);
  assert.equal(await evaluate(`document.getElementById('vae-tiling').checked`),false);
  console.log('PASS all fourteen presets set exact dimensions/tiling/pressed state');
  const bad = await evaluate(`fetch('/api/v1/jobs', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(r=>r.status)`);
  assert.equal(bad, 403);
  console.log('PASS job write requires exact UI header');
  await evaluate(`document.getElementById('prompt').value='a red cube on a white table, studio photo, sharp focus, soft daylight';
    document.getElementById('steps').value='20'; document.getElementById('seed').value='${real ? '42' : '18446744073709551615'}';
    document.getElementById('create-form').requestSubmit();`);
  await until(`['Generating', 'Queued'].some(text => document.getElementById('job-feedback').textContent.startsWith(text))`);
  assert.equal(await evaluate(`document.getElementById('generate-btn').textContent`), 'Queue another image');
  assert(await evaluate(`document.getElementById('preview-message').textContent === document.getElementById('job-feedback').textContent`));
  console.log('PASS active generation is visible beside Generate and inside the empty preview');
  await until(`document.getElementById('job-progress').hasAttribute('value') && document.getElementById('job-progress').value > 0 && document.getElementById('job-progress').value < 20`, real ? 180000 : 15000);
  assert(await evaluate(`document.getElementById('job-progress-label').textContent.includes('% of drawing')`));
  console.log('PASS progress bar shows observed sampler steps during denoising');
  const progressShot = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  writeFileSync(join(artifactDir, 'progress-active.png'), Buffer.from(progressShot.data, 'base64'));
  await until(`document.getElementById('job-status').textContent.startsWith('succeeded')`, real ? 240000 : 15000);
  await until(`document.getElementById('preview-image').naturalWidth > 0`);
  assert(await evaluate(`!document.getElementById('view-result').hidden && document.getElementById('job-feedback').textContent.startsWith('Image ready')`));
  assert(await evaluate(`document.getElementById('job-progress').value === document.getElementById('job-progress').max`));
  let ids = await jobIds();
  assert.equal(ids.length, 1);
  const id = ids[0];
  let job = (await api('/api/v1/jobs/' + id)).body;
  assert.equal(job.config.seed, real ? '42' : '18446744073709551615');
  assert.equal(job.state, 'succeeded');
  assert(job.elapsed_ms > 0);
  if (real) assert.deepEqual(Object.keys(job.phase_ms).sort(), ['decode', 'denoise', 'encode', 'load', 'save']);
  assert(await evaluate(`document.querySelectorAll('.history-card').length === 1 && !document.getElementById('action-download').disabled`));
  console.log(`PASS real UI submit → successful job → loaded Preview + History (${(job.elapsed_ms/1000).toFixed(1)} s)`);
  writeFileSync(join(artifactDir, 'job.json'), JSON.stringify(job, null, 2));
  const image = await fetch(origin + job.image_url);
  assert.equal(image.headers.get('content-type'), 'image/jpeg');
  assert.equal(image.headers.get('x-content-type-options'), 'nosniff');
  const imageBytes = Buffer.from(await image.arrayBuffer());
  assert.equal(imageBytes.subarray(0, 3).toString('hex'), 'ffd8ff');
  assert.match(job.image_filename, /^qi21-\d{8}-\d+\.jpg$/);
  writeFileSync(join(artifactDir, 'result.jpg'), imageBytes);
  await send('Page.reload');
  await until(`document.getElementById('preview-image').naturalWidth > 0 && document.querySelectorAll('.history-card').length === 1`);
  console.log('PASS reload recovers server history and completed image');
  const snapshot = await evaluate(`JSON.stringify(QwenStudio.formSnapshot())`);
  const malformed = await evaluate(`(() => { var j=${JSON.stringify(job)}; j.image_url='https://example.invalid/x'; return QwenStudio.parseJob(j); })()`);
  assert.equal(malformed, null);
  assert.equal(await evaluate(`JSON.stringify(QwenStudio.formSnapshot())`), snapshot);
  for (const width of [1440, 375, 320]) {
    await send('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: width < 500});
    await delay(150);
    assert(await evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'));
    await evaluate(`document.getElementById('advanced').open=true; window.scrollTo(0,500)`);
    if (width > 940) {
      const positions = await evaluate(`(() => {
        const panel=document.querySelector('.prompt-panel');
        const preview=document.getElementById('preview-frame');
        const settings=document.querySelector('.settings-scroll');
        const before=[panel.getBoundingClientRect().top,preview.getBoundingClientRect().top];
        settings.scrollTop=500;
        return {before, after:[panel.getBoundingClientRect().top,preview.getBoundingClientRect().top],
          scroll:settings.scrollTop, page:window.scrollY};
      })()`);
      assert.deepEqual(positions.before, positions.after);
      assert(positions.scroll > 0);
      assert.equal(positions.page, 0);
      await evaluate(`document.getElementById('quant').scrollIntoView({block:'center'})`);
      const helpShot = await send('Page.captureScreenshot', {format:'png'});
      writeFileSync(join(artifactDir, 'option-help-1440.png'), Buffer.from(helpShot.data,'base64'));
    }
    assert(await evaluate(`(() => {const r=document.getElementById('generate-btn').getBoundingClientRect(); return r.top>=0 && r.bottom<window.innerHeight;})()`));
    const shot = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
    writeFileSync(join(artifactDir, `studio-${width}.png`), Buffer.from(shot.data, 'base64'));
  }
  console.log('PASS desktop/mobile layouts at 1440/375/320 px');
  if (!real) {
    await evaluate(`document.getElementById('prompt').value='cancel me'; document.getElementById('create-form').requestSubmit()`);
    await until(`['queued','running'].some(s => document.getElementById('job-status').textContent.startsWith(s))`);
    await evaluate(`document.getElementById('cancel-job').click()`);
    await until(`document.getElementById('job-status').textContent.startsWith('cancelled')`);
    console.log('PASS UI cancellation acknowledged without invented result');
    ids = await jobIds();
    await api('/api/v1/jobs/' + ids[1] + '/forget', {});
    assert.equal((await api('/api/v1/jobs')).body.jobs.length, 1);
    console.log('PASS terminal entry can be forgotten');
  }
  assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
  console.log('PASS no uncaught browser exceptions');
}

try { await Promise.race([check(), deadline]); }
catch (error) { aborted = true; failed = true; console.error(error.stack); }
finally {
  clearTimeout(fatal);
  writeFileSync(join(artifactDir, 'server.log'), log);
  if (chrome) {
    const result = await shutdownChrome(chrome, ws, cdp);
    if (!result) { failed = true; console.error('Chrome cleanup pending'); }
  }
  server.kill('SIGTERM');
  let stopTimer;
  const ended = await Promise.race([
    new Promise(resolve => {
      if (server.exitCode !== null || server.signalCode !== null) resolve(true);
      else server.once('exit', () => resolve(true));
    }), new Promise(resolve => { stopTimer = setTimeout(() => resolve(false), 60000); }),
  ]);
  clearTimeout(stopTimer);
  if (!ended) { failed = true; console.error('Owned test server did not stop after SIGTERM'); }
  if (server.exitCode !== 0) { failed = true; console.error('Test server exited abnormally', server.exitCode, server.signalCode); }
}
process.exitCode = failed ? 1 : 0;
