/* Raw HTTP response framing for the integration check — BYTE-true and
 * dependency-free (Buffer only; no fetch, no fake sockets).
 *
 * Why this exists as its own module (2026-10-12 review):
 *   * interim 1xx (e.g. '100 Continue') blocks must be sliced by their CRLF
 *     boundary — a regex that only recognises a headerless interim block
 *     never strips a WITH-headers one and then loops forever, defeating the
 *     watchdog;
 *   * response bytes are bounded by a fixed cap (never unbounded);
 *   * headers are decoded latin1 (byte-true per RFC 7230) while the body is
 *     decoded UTF-8 — a latin1 body would corrupt a non-ASCII prompt echo.
 *
 * Every function here is PURE over a Buffer and unit-tested by
 * scripts/raw_http_selfcheck.mjs (no server, no sockets, no Chrome).
 */
export const MAX_RESPONSE_BYTES = 256 * 1024;

/**
 * Peel COMPLETE leading 1xx interim response blocks, delimited strictly by
 * their CRLFCRLF boundary. Returns { interim, finalHeadEnd } with absolute
 * offsets into the ORIGINAL buffer; finalHeadEnd is -1 while any block —
 * interim or final — is still incomplete (more bytes must be awaited).
 */
export function peelInterimResponses(buf) {
  let offset = 0;
  let interim = 0;
  for (;;) {
    // indexOf with a string searches the exact CR LF CR LF bytes.
    const end = buf.indexOf('\r\n\r\n', offset);
    if (end < 0) return { interim, finalHeadEnd: -1 };
    const head = buf.subarray(offset, end).toString('latin1');
    if (/^HTTP\/1\.[01] 1\d\d\b/.test(head)) {
      offset = end + 4;
      interim++;
      continue;
    }
    return { interim, finalHeadEnd: end };
  }
}

/** { interim, complete, headEnd, contentLength, bodyReady, head } for a Buffer. */
export function inspectResponse(buf) {
  const peeled = peelInterimResponses(buf);
  if (peeled.finalHeadEnd < 0) {
    return { interim: peeled.interim, complete: false, headEnd: -1,
             contentLength: 0, bodyReady: false, head: '' };
  }
  const head = buf.subarray(0, peeled.finalHeadEnd).toString('latin1')
    // strip any peeled interim blocks from the reported head string
    .split('\r\n\r\n').pop();
  const m = /^content-length: *(0|[1-9]\d*)[ \t]*$/im.exec(head);
  const contentLength = m ? Number(m[1]) : 0;
  const bodyBytes = buf.length - (peeled.finalHeadEnd + 4);
  return { interim: peeled.interim, complete: true,
           headEnd: peeled.finalHeadEnd, contentLength,
           bodyReady: bodyBytes >= contentLength, head };
}

/** Everything after the FINAL header block, as a Buffer (or null). */
export function responseBodyBytes(buf) {
  const s = inspectResponse(buf);
  return s.complete ? buf.subarray(s.headEnd + 4) : null;
}

/** Headers of the FINAL response, latin1 (byte-true) as a string. */
export function responseHead(buf) {
  return inspectResponse(buf).head;
}

/** Body of the FINAL response decoded UTF-8 (correct for non-ASCII). */
export function responseBody(buf) {
  const bytes = responseBodyBytes(buf);
  return bytes === null ? '' : bytes.toString('utf8');
}

export function responseStatus(buf) {
  const m = /^HTTP\/1\.[01] (\d{3})/.exec(responseHead(buf));
  return m ? Number(m[1]) : NaN;
}

/** All values of a header (wire order) of the FINAL response. */
export function headerValues(buf, name) {
  const want = name.toLowerCase() + ':';
  return responseHead(buf).split('\r\n')
    .filter((l) => l.toLowerCase().startsWith(want))
    .map((l) => l.slice(l.indexOf(':') + 1).trim());
}
