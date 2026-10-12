#!/usr/bin/env node
/* Focused protocol checks for scripts/raw_http.mjs — NO server, NO socket,
 * NO Chrome, NO fake DOM: pure Buffer fixtures drive the framing logic that
 * the integration check relies on (interim-100 slicing by CRLF boundary,
 * header/body decoding, content-length completion, cap).
 *
 * Usage: node scripts/raw_http_selfcheck.mjs   (exits 0/1)
 */
import { MAX_RESPONSE_BYTES, peelInterimResponses, inspectResponse,
         responseHead, responseBody, responseStatus, headerValues,
         responseBodyBytes } from './raw_http.mjs';

const results = [];
function check(name, fn) {
  try { fn(); results.push(['PASS', name]); }
  catch (err) { results.push(['FAIL', name + ': ' + err.message]); }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
const B = (s) => Buffer.from(s, 'binary');   // byte-true fixture helper

check('headerless interim 100 is peeled by its CRLF boundary', () => {
  const buf = B('HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi');
  const s = inspectResponse(buf);
  assert(s.complete && s.interim === 1, 'one interim peeled, final complete');
  assert(responseStatus(buf) === 200, 'final status, not the interim one');
  assert(responseBody(buf) === 'hi', 'body after the FINAL header block');
});

check('interim 100 WITH headers is peeled too (the regex-framing bug)', () => {
  // This is the case the old regex could not strip — it used to loop
  // forever and defeat the whole-check watchdog.
  const buf = B('HTTP/1.1 100 Continue\r\nX-Interim: yes\r\n\r\n'
              + 'HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc');
  const s = inspectResponse(buf);
  assert(s.complete && s.interim === 1, 'interim with headers must peel');
  assert(responseBody(buf) === 'abc', 'final body only');
  assert(!responseHead(buf).includes('X-Interim'), 'interim headers not reported');
});

check('multiple interim blocks all peel; incomplete stream stays pending', () => {
  let buf = B('HTTP/1.1 100 Continue\r\n\r\nHTTP/1.0 101 Switching Protocols\r\n\r\n'
            + 'HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n');
  let s = inspectResponse(buf);
  assert(s.complete && s.interim === 2 && responseStatus(buf) === 204, 'two interims');
  buf = B('HTTP/1.1 100 Continue\r\nX-H: 1\r\n');          // interim INCOMPLETE
  s = inspectResponse(buf);
  assert(!s.complete && s.headEnd === -1, 'incomplete interim must await more bytes');
  buf = B('HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nab');   // body truncated
  s = inspectResponse(buf);
  assert(s.complete && !s.bodyReady, 'truncated body is not ready');
});

check('content-length parsed from the FINAL header block only', () => {
  const buf = B('HTTP/1.1 100 Continue\r\nContent-Length: 999\r\n\r\n'
              + 'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok');
  const s = inspectResponse(buf);
  assert(s.contentLength === 2, 'interim CL must not win');
  assert(s.bodyReady, 'body complete at 2 bytes');
});

check('headers decode latin1; body decodes UTF-8 (non-ASCII prompt echo)', () => {
  const prompt = 'fox ünïcode 日本語 🦊';
  const json = JSON.stringify({ prompt });
  const body = Buffer.from(json, 'utf8');
  const buf = Buffer.concat([
    Buffer.from('HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n'
              + `Content-Length: ${body.length}\r\nX-Latin1-Test: café\r\n\r\n`, 'latin1'),
    body,
  ]);
  assert(responseHead(buf).includes('café'), 'latin1 head keeps é as byte 0xE9');
  assert(JSON.parse(responseBody(buf)).prompt === prompt,
         'UTF-8 body must decode exactly — latin1 framing would corrupt it');
  assert(headerValues(buf, 'content-type').length === 1, 'header values extracted');
});

check('response cap constant is fixed and generous but bounded', () => {
  assert(MAX_RESPONSE_BYTES === 256 * 1024, '256 KiB fixed response cap');
});

check('responseBodyBytes keeps raw bytes (no reserialization path)', () => {
  const raw = Buffer.from('{"seed": 18446744073709551615}\n', 'utf8');
  const buf = Buffer.concat([
    Buffer.from(`HTTP/1.1 200 OK\r\nContent-Length: ${raw.length}\r\n\r\n`, 'ascii'),
    raw]);
  assert(responseBodyBytes(buf).equals(raw), 'uint64 export bytes survive byte-for-byte');
});

let failed = 0;
for (const [status, line] of results) {
  if (status === 'FAIL') failed++;
  console.log(`${status} ${line}`);
}
console.log('NOTE: pure framing checks only — exercised against the live ' +
            'server by scripts/web_integration_check.mjs.');
process.exit(failed ? 1 : 0);
