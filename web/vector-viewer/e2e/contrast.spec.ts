import { expect, test } from '@playwright/test';
import { contrast, decode, openViewer, parseRgb } from './helpers';

// WCAG 2.2 AA: 4.5:1 for normal text, 3:1 for large text and for the
// boundaries of user interface components. Measured from the computed styles
// of the rendered page in both themes, not read off the CSS, and printed so
// the numbers can be quoted.

interface Pair {
  name: string;
  fg: string;
  bg: string;
  need: number;
}

for (const theme of ['light', 'dark'] as const) {
  test(`contrast, ${theme} theme`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: theme });
    await openViewer(page);
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);

    const pairs: Pair[] = await page.evaluate(() => {
      const color = (el: Element, prop: 'color' | 'backgroundColor' | 'borderTopColor' | 'outlineColor') =>
        getComputedStyle(el)[prop];
      const body = document.body;
      const h1 = document.querySelector('h1')!;
      const archive = document.querySelector('[data-testid=archive]')!;
      const button = document.querySelector('button')!;
      const zoom = document.querySelector('[data-testid=zoom]')!;
      const frame = document.querySelector('.map-frame')!;
      (button as HTMLElement).focus();
      const focus = color(button, 'outlineColor');
      return [
        { name: 'title text on page', fg: color(h1, 'color'), bg: color(body, 'backgroundColor'), need: 4.5 },
        { name: 'archive name (muted) on page', fg: color(archive, 'color'), bg: color(body, 'backgroundColor'), need: 4.5 },
        { name: 'zoom readout (muted) on page', fg: color(zoom, 'color'), bg: color(body, 'backgroundColor'), need: 4.5 },
        { name: 'button text on button', fg: color(button, 'color'), bg: color(button, 'backgroundColor'), need: 4.5 },
        { name: 'button border on page', fg: color(button, 'borderTopColor'), bg: color(body, 'backgroundColor'), need: 3 },
        { name: 'focus ring on page', fg: focus, bg: color(body, 'backgroundColor'), need: 3 },
        { name: 'map frame border on page', fg: color(frame, 'borderTopColor'), bg: color(body, 'backgroundColor'), need: 3 },
      ];
    });

    // The drawing: black linework and text on white paper, from the pixels.
    const shot = decode(await page.getByTestId('map').screenshot());
    let min = 255;
    for (const v of shot.grey) if (v < min) min = v;
    const ink = min;
    const paper = shot.grey[0]!;
    pairs.push({
      name: 'linework on paper (measured pixels)',
      fg: `rgb(${ink}, ${ink}, ${ink})`,
      bg: `rgb(${paper}, ${paper}, ${paper})`,
      need: 4.5,
    });
    pairs.push({ name: 'linework on paper (style colours)', fg: 'rgb(0, 0, 0)', bg: 'rgb(255, 255, 255)', need: 4.5 });

    const rows: string[] = [];
    for (const p of pairs) {
      const ratio = contrast(parseRgb(p.fg), parseRgb(p.bg));
      rows.push(`${theme.padEnd(5)} ${p.name.padEnd(40)} ${ratio.toFixed(2).padStart(6)}:1  (needs ${p.need}:1)`);
      expect(ratio, `${theme}: ${p.name} is ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(p.need);
    }
    console.log(`CONTRAST\n${rows.join('\n')}`);
  });
}
