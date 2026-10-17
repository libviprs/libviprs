import type { Page } from '@playwright/test';
import { PNG } from 'pngjs';
import type {} from '../src/App';

export const STORE = 'http://127.0.0.1:4174';

export async function openViewer(page: Page, query = ''): Promise<void> {
  await openViewerAt(page, `${STORE}/golden.pmtiles`, query);
}

export async function openViewerAt(page: Page, archive: string, query = ''): Promise<void> {
  await page.goto(`/?archive=${encodeURIComponent(archive)}${query}`);
  await page.waitForFunction(() => window.__viewer !== undefined);
  await settle(page);
}

/** Waits until the map has drawn everything it was asked to. */
export async function settle(page: Page): Promise<void> {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        const map = window.__viewer!.map;
        const done = () => requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
        if (map.loaded()) {
          map.triggerRepaint();
          map.once('idle', done);
        } else {
          map.once('idle', done);
        }
      }),
  );
}

export async function jump(page: Page, center: [number, number], zoom: number): Promise<void> {
  await page.evaluate(
    ([c, z]) => window.__viewer!.map.jumpTo({ center: c as [number, number], zoom: z as number }),
    [center, zoom] as const,
  );
  await settle(page);
}

export interface Raster {
  width: number;
  height: number;
  /** Grey level of each pixel, and the colour spread (max channel - min). */
  grey: Uint8Array;
  spread: Uint8Array;
}

export function decode(png: Buffer): Raster {
  const img = PNG.sync.read(png);
  const grey = new Uint8Array(img.width * img.height);
  const spread = new Uint8Array(img.width * img.height);
  for (let i = 0; i < grey.length; i++) {
    const r = img.data[i * 4]!;
    const g = img.data[i * 4 + 1]!;
    const b = img.data[i * 4 + 2]!;
    grey[i] = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
    spread[i] = Math.max(r, g, b) - Math.min(r, g, b);
  }
  return { width: img.width, height: img.height, grey, spread };
}

export async function requestLog(page: Page): Promise<
  { method: string; path: string; range?: string | null; status: number; bytes?: number }[]
> {
  return (await (await page.request.get(`${STORE}/__log`)).json()) as never;
}

export async function resetLog(page: Page): Promise<void> {
  await page.request.get(`${STORE}/__reset`);
}

// ---- WCAG 2.x contrast ------------------------------------------------------

export function parseRgb(css: string): [number, number, number] {
  const m = /rgba?\(\s*(\d+)[ ,]+(\d+)[ ,]+(\d+)/.exec(css);
  if (!m) throw new Error(`not an rgb colour: ${css}`);
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

function channel(v: number): number {
  const s = v / 255;
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
}

export function luminance([r, g, b]: [number, number, number]): number {
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

export function contrast(a: [number, number, number], b: [number, number, number]): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}
