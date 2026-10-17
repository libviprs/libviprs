import { expect, test } from '@playwright/test';
import { decode, openViewerAt } from './helpers';

// The same archive served by a real S3-compatible object store (RustFS), not by
// the fixture server. Skipped unless OBJECT_STORE_URL names the archive; the
// deployment script e2e/objectstore.sh starts the store, uploads the archive,
// sets the bucket's CORS rules and runs this file. LIBVIPRS_REQUIRE_OBJECT_STORE=1
// turns a missing store into a failure, because a skipped lane looks like a
// passing one.
const url = process.env['OBJECT_STORE_URL'];

test.beforeEach(() => {
  if (!url && process.env['LIBVIPRS_REQUIRE_OBJECT_STORE'] === '1') {
    throw new Error('LIBVIPRS_REQUIRE_OBJECT_STORE=1 and OBJECT_STORE_URL is not set');
  }
  test.skip(!url, 'no object store: set OBJECT_STORE_URL');
});

test.describe('a real object store', () => {
  test('serves byte ranges the browser can read, headers included', async ({ page }) => {
    await page.goto('/');
    const r = await page.evaluate(async (u) => {
      const res = await fetch(u, { headers: { Range: 'bytes=0-127' } });
      const body = new Uint8Array(await res.arrayBuffer());
      return {
        status: res.status,
        contentRange: res.headers.get('content-range'),
        contentLength: res.headers.get('content-length'),
        acceptRanges: res.headers.get('accept-ranges'),
        etag: res.headers.get('etag'),
        magic: new TextDecoder().decode(body.slice(0, 7)),
        got: body.length,
      };
    }, url!);
    expect(r.status).toBe(206);
    expect(r.got).toBe(128);
    expect(r.magic).toBe('PMTiles');
    // These three are only readable by script when the store lists them in
    // Access-Control-Expose-Headers, so reading them proves the CORS rule.
    expect(r.contentRange).toMatch(/^bytes 0-127\/\d+$/);
    expect(r.contentLength).toBe('128');
    expect(r.acceptRanges).toBe('bytes');
    expect(r.etag).toBeTruthy();
  });

  test('an unsatisfiable range is a 416, not a body', async ({ page }) => {
    await page.goto('/');
    const status = await page.evaluate(async (u) => {
      const res = await fetch(u, { headers: { Range: 'bytes=999999999-999999999' } });
      return res.status;
    }, url!);
    expect(status).toBe(416);
  });

  test('the viewer draws the drawing from it with ranged reads only', async ({ page }) => {
    const reads: { status: number; range: string | undefined }[] = [];
    page.on('response', (r) => {
      if (r.url() === url) reads.push({ status: r.status(), range: r.request().headers()['range'] });
    });
    await openViewerAt(page, url!);
    await expect(page.getByRole('alert')).toHaveCount(0);
    expect(reads.length).toBeGreaterThan(1);
    for (const e of reads) {
      expect(e.range).toMatch(/^bytes=\d+-\d+$/);
      expect(e.status).toBe(206);
    }
    const shot = decode(await page.getByTestId('map').screenshot());
    let dark = 0;
    for (const v of shot.grey) if (v < 64) dark++;
    expect(dark / shot.grey.length).toBeGreaterThan(0.002);
  });
});
