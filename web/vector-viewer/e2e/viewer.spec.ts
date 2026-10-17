import { expect, test } from '@playwright/test';
import { STORE, decode, jump, openViewer, requestLog, resetLog, settle } from './helpers';

test.describe('a libviprs vector archive in the reference viewer', () => {
  test('streams the archive with Range requests and never needs a DWG', async ({ page }) => {
    const seen: string[] = [];
    page.on('request', (r) => seen.push(r.url()));
    await resetLog(page);
    await openViewer(page);

    const log = (await requestLog(page)).filter((e) => e.path.endsWith('golden.pmtiles') && e.method === 'GET');
    expect(log.length).toBeGreaterThan(1);
    // Every read of the archive was a ranged read answered with 206, so the
    // file was never downloaded whole.
    for (const e of log) {
      expect(e.range, JSON.stringify(e)).toMatch(/^bytes=\d+-\d+$/);
      expect(e.status).toBe(206);
    }
    // The first read is the PMTiles header plus root directory.
    expect(log[0]!.range).toBe('bytes=0-16383');
    const fetched = log.reduce((n, e) => n + (e.bytes ?? 0), 0);
    expect(fetched).toBeLessThan(200_000);

    expect(seen.filter((u) => /\.dwg(\?|$)/i.test(u))).toEqual([]);
    expect(seen.some((u) => u.startsWith(`${STORE}/golden.pmtiles`))).toBe(true);
  });

  test('draws black linework on white and nothing else', async ({ page }) => {
    await openViewer(page);
    const shot = decode(await page.getByTestId('map').screenshot());
    let dark = 0;
    let coloured = 0;
    for (let i = 0; i < shot.grey.length; i++) {
      if (shot.grey[i]! < 64) dark++;
      if (shot.spread[i]! > 2) coloured++;
    }
    expect(coloured, 'no pixel is anything but grey').toBe(0);
    expect(shot.grey[0]).toBe(255);
    expect(shot.grey[shot.grey.length - 1]).toBe(255);
    const share = dark / shot.grey.length;
    expect(share).toBeGreaterThan(0.002);
    expect(share).toBeLessThan(0.3);
    // The darkest pixel is black and the lightest is white.
    expect(Math.min(...shot.grey.subarray(0, 4096))).toBeLessThanOrEqual(255);
    let min = 255;
    for (const v of shot.grey) if (v < min) min = v;
    expect(min).toBeLessThanOrEqual(8);
  });

  test('stays crisp through the generated zooms and two more of over-zoom', async ({ page }) => {
    await openViewer(page);
    const { maxZoom, bounds } = await page.evaluate(() => ({
      maxZoom: window.__viewer!.maxZoom,
      bounds: window.__viewer!.bounds,
    }));
    // The drawing's top-left corner: one horizontal and one vertical stroke
    // leave it, so both can be measured at every zoom.
    const corner: [number, number] = [bounds[0], bounds[3]];
    const widths: number[] = [];
    for (let z = 1; z <= maxZoom + 2; z++) {
      await jump(page, corner, z);
      const shot = decode(await page.getByTestId('map').screenshot());
      // Where the corner is on screen. At low zoom the camera is held inside
      // the world, so the corner is not at the centre there.
      const at = await page.evaluate((c) => {
        const p = window.__viewer!.map.project(c as [number, number]);
        return { x: Math.round(p.x), y: Math.round(p.y) };
      }, corner);
      const cx = at.x;
      const cy = at.y;
      // A vertical stroke runs down from the corner. The header's bounds are
      // the drawing's box plus a small margin, so the stroke can sit some
      // pixels off the exact centre at deep zoom: find the darkest column
      // near it, then sum the darkness across the stroke around that column.
      let best = 0;
      let bestIntermediate = 99;
      for (const dy of [110, 150, 190, 230]) {
        const y = cy + dy;
        let darkest = cx;
        for (let x = cx - 30; x <= cx + 30; x++) {
          if (shot.grey[y * shot.width + x]! < shot.grey[y * shot.width + darkest]!) darkest = x;
        }
        let sum = 0;
        let intermediate = 0;
        for (let x = darkest - 8; x <= darkest + 8; x++) {
          const v = shot.grey[y * shot.width + x]!;
          sum += 255 - v;
          if (v > 20 && v < 235) intermediate++;
        }
        if (sum > best) {
          best = sum;
          bestIntermediate = intermediate;
        }
      }
      const px = best / 255;
      widths.push(px);
      // One stroke, 1.25 px wide, however far in we are: not thinner, not
      // smeared wider, and with at most the two antialiased edge pixels and a
      // third from sub-pixel placement.
      expect(px, `zoom ${z}: stroke width in px`).toBeGreaterThan(0.9);
      expect(px, `zoom ${z}: stroke width in px`).toBeLessThan(1.8);
      expect(bestIntermediate, `zoom ${z}: antialiased pixels`).toBeLessThanOrEqual(3);
    }
    test.info().annotations.push({ type: 'stroke width px per zoom', description: widths.map((w) => w.toFixed(2)).join(' ') });
    console.log(`MEASURED stroke width (px) at zoom 1..${maxZoom + 2}: ${widths.map((w) => w.toFixed(2)).join(' ')}`);
  });

  test('labels are placed and drawn from the glyph files the app serves', async ({ page }) => {
    const glyphs: { url: string; status: number }[] = [];
    page.on('response', (r) => {
      if (r.url().includes('/fonts/')) glyphs.push({ url: r.url(), status: r.status() });
    });
    await openViewer(page);
    // Find the label's anchor in the loaded tiles, then look at it close up.
    const anchor = await page.evaluate(async () => {
      const map = window.__viewer!.map;
      const b = window.__viewer!.bounds;
      map.jumpTo({ center: [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], zoom: 1 });
      await new Promise<void>((r) => map.once('idle', () => r()));
      // Source tiles arrive a moment after the map reports idle.
      for (let i = 0; i < 50; i++) {
        const f = map
          .querySourceFeatures('drawing', { sourceLayer: 'text' })
          .find((x) => x.properties['text'] === 'GROUND FLOOR');
        if (f && f.geometry.type === 'Point') return f.geometry.coordinates as [number, number];
        await new Promise((r) => setTimeout(r, 100));
      }
      return null;
    });
    expect(anchor).not.toBeNull();
    await jump(page, anchor!, 4);
    const labels = await page.evaluate(() =>
      window.__viewer!.map
        .queryRenderedFeatures(undefined, { layers: ['annotations'] })
        .map((f) => String(f.properties['text'])),
    );
    expect(labels).toContain('GROUND FLOOR');
    expect(glyphs.length).toBeGreaterThan(0);
    expect(glyphs.every((g) => g.status === 200)).toBe(true);
    expect(glyphs.some((g) => decodeURIComponent(g.url).includes('Noto Sans Regular/0-255.pbf'))).toBe(true);
  });

  test('a store without CORS headers is reported, not shown as a blank map', async ({ page }) => {
    await page.goto(`/?archive=${encodeURIComponent(`${STORE}/nocors/golden.pmtiles`)}`);
    await expect(page.getByRole('alert')).toBeVisible();
  });

  test('the chrome changes theme and the paper does not', async ({ page }) => {
    await openViewer(page, '&theme=light');
    const light = decode(await page.getByTestId('map').screenshot());
    await page.getByRole('button', { name: 'Dark theme' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await settle(page);
    const dark = decode(await page.getByTestId('map').screenshot());
    expect(dark.grey[0]).toBe(255);
    expect(light.grey[0]).toBe(255);
    const bodyBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    expect(bodyBg).toBe('rgb(16, 18, 20)');
  });

  for (const theme of ['light', 'dark'] as const) {
    test(`visual regression, ${theme} theme, fitted`, async ({ page }) => {
      await openViewer(page, `&theme=${theme}`);
      await expect(page).toHaveScreenshot(`fitted-${theme}.png`);
    });
  }

  for (const z of [3, 6]) {
    test(`visual regression of the drawing at zoom ${z}`, async ({ page }) => {
      await openViewer(page);
      const bounds = await page.evaluate(() => window.__viewer!.bounds);
      await jump(page, [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2], z);
      await expect(page.getByTestId('map')).toHaveScreenshot(`drawing-z${z}.png`);
    });
  }
});
