// Two origins, like a real deployment: the app, and an object store holding
// the archive. The "store" is the part under test, so it is written to behave
// the way S3 and RustFS do for ranged reads: 206 with Content-Range, 416 for an
// unsatisfiable range, a strong ETag, and CORS headers that expose the ones a
// browser has to read. Two extra paths misbehave on purpose, so a spec can
// show it would notice.
//
//   :4173  the built app (dist/) and the glyph files
//   :4174  /golden.pmtiles          well behaved, with CORS
//          /nocors/golden.pmtiles   Range works, no CORS headers
//          /__log, /__reset         what the store was asked
import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, '..', 'dist');
const fixtures = resolve(here, 'fixtures');
const log = [];

const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.pbf': 'application/x-protobuf',
  '.map': 'application/json',
};

function serveStatic(root) {
  return (req, res) => {
    const url = new URL(req.url, 'http://x');
    let rel = normalize(decodeURIComponent(url.pathname));
    if (rel.includes('..')) return void res.writeHead(400).end();
    if (rel === '/' || rel === '') rel = '/index.html';
    const file = join(root, rel);
    if (!file.startsWith(root) || !existsSync(file) || !statSync(file).isFile()) {
      return void res.writeHead(404).end('not found');
    }
    res.writeHead(200, {
      'content-type': types[extname(file)] ?? 'application/octet-stream',
      'cache-control': 'no-store',
    });
    createReadStream(file).pipe(res);
  };
}

function parseRange(header, size) {
  const m = /^bytes=(\d*)-(\d*)$/.exec(header ?? '');
  if (!m) return null;
  let [, a, b] = m;
  if (a === '' && b === '') return null;
  let start;
  let end;
  if (a === '') {
    const n = Number(b);
    start = Math.max(0, size - n);
    end = size - 1;
  } else {
    start = Number(a);
    end = b === '' ? size - 1 : Math.min(Number(b), size - 1);
  }
  if (start > end || start >= size) return 'unsatisfiable';
  return { start, end };
}

function store({ cors }) {
  return (req, res) => {
    const url = new URL(req.url, 'http://x');
    const name = url.pathname.split('/').pop();
    const file = join(fixtures, name ?? '');
    const headers = {};
    if (cors) {
      headers['access-control-allow-origin'] = '*';
      headers['access-control-expose-headers'] = 'Content-Range, Content-Length, Accept-Ranges, ETag';
      headers['access-control-allow-headers'] = 'Range, If-Match, If-None-Match';
      headers['access-control-allow-methods'] = 'GET, HEAD, OPTIONS';
      headers['access-control-max-age'] = '600';
    }
    if (req.method === 'OPTIONS') {
      log.push({ method: 'OPTIONS', path: url.pathname, status: 204 });
      return void res.writeHead(204, headers).end();
    }
    if (!existsSync(file) || !statSync(file).isFile()) {
      log.push({ method: req.method, path: url.pathname, status: 404 });
      return void res.writeHead(404, headers).end();
    }
    const size = statSync(file).size;
    const etag = '"' + createHash('sha256').update(readFileSync(file)).digest('hex').slice(0, 32) + '"';
    headers['accept-ranges'] = 'bytes';
    headers['etag'] = etag;
    headers['cache-control'] = 'no-store';
    headers['content-type'] = 'application/octet-stream';
    const rangeHeader = req.headers['range'];
    const range = rangeHeader ? parseRange(rangeHeader, size) : null;
    if (range === 'unsatisfiable') {
      headers['content-range'] = `bytes */${size}`;
      log.push({ method: req.method, path: url.pathname, range: rangeHeader, status: 416 });
      return void res.writeHead(416, headers).end();
    }
    if (range) {
      const length = range.end - range.start + 1;
      headers['content-range'] = `bytes ${range.start}-${range.end}/${size}`;
      headers['content-length'] = String(length);
      log.push({ method: req.method, path: url.pathname, range: rangeHeader, status: 206, bytes: length });
      res.writeHead(206, headers);
      if (req.method === 'HEAD') return void res.end();
      return void createReadStream(file, { start: range.start, end: range.end }).pipe(res);
    }
    headers['content-length'] = String(size);
    log.push({ method: req.method, path: url.pathname, range: null, status: 200, bytes: size });
    res.writeHead(200, headers);
    if (req.method === 'HEAD') return void res.end();
    createReadStream(file).pipe(res);
  };
}

const app = serveStatic(dist);
createServer(app).listen(4173, '127.0.0.1');

const good = store({ cors: true });
const bare = store({ cors: false });
createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname === '/__log') {
    res.writeHead(200, { 'content-type': 'application/json', 'access-control-allow-origin': '*' });
    return void res.end(JSON.stringify(log));
  }
  if (url.pathname === '/__reset') {
    log.length = 0;
    res.writeHead(204, { 'access-control-allow-origin': '*' });
    return void res.end();
  }
  if (url.pathname.startsWith('/nocors/')) return bare(req, res);
  return good(req, res);
}).listen(4174, '127.0.0.1');

console.log('viewer on :4173, store on :4174');
