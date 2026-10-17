import { useEffect, useRef, useState } from 'react';
import {
  Map as MapLibreMap,
  addProtocol,
  removeProtocol,
  setWorkerUrl,
  type StyleSpecification,
} from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { PMTiles, Protocol } from 'pmtiles';
import template from '../../../src/vector/black_on_white.style.json';

type Theme = 'light' | 'dark';

/** What the e2e specs read. Not an API. */
export interface ViewerHooks {
  map: MapLibreMap;
  maxZoom: number;
  bounds: [number, number, number, number];
}

declare global {
  interface Window {
    __viewer?: ViewerHooks;
  }
}

/** Fills the two placeholders in the shared style, as the Rust helper does. */
function styleFor(archiveUrl: string): StyleSpecification {
  const style = structuredClone(template) as unknown as StyleSpecification;
  const source = style.sources['drawing'];
  if (!source || source.type !== 'vector') throw new Error('style has no vector source');
  (source as { url: string }).url = `pmtiles://${archiveUrl}`;
  style.glyphs = `${window.location.origin}/fonts/{fontstack}/{range}.pbf`;
  return style;
}

function initialTheme(): Theme {
  const asked = new URLSearchParams(window.location.search).get('theme');
  if (asked === 'light' || asked === 'dark') return asked;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function App() {
  const archive =
    new URLSearchParams(window.location.search).get('archive') ??
    `${window.location.protocol}//${window.location.hostname}:4174/golden.pmtiles`;
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [zoom, setZoom] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.documentElement.dataset['theme'] = theme;
  }, [theme]);

  useEffect(() => {
    const el = container.current;
    if (!el) return;
    setWorkerUrl(`${window.location.origin}/maplibre/maplibre-gl-worker.mjs`);
    const protocol = new Protocol();
    addProtocol('pmtiles', protocol.tile);
    const source = new PMTiles(archive);
    protocol.add(source);
    let map: MapLibreMap | undefined;
    let cancelled = false;

    source
      .getHeader()
      .then((h) => {
        if (cancelled) return;
        const bounds: [number, number, number, number] = [h.minLon, h.minLat, h.maxLon, h.maxLat];
        map = new MapLibreMap({
          container: el,
          style: styleFor(archive),
          bounds,
          fitBoundsOptions: { padding: 24 },
          minZoom: 0,
          renderWorldCopies: false,
          // The drawing is crisp through the generated zooms and about two
          // more of over-zoom; past that the tile grid shows.
          maxZoom: h.maxZoom + 2,
          fadeDuration: 0,
          attributionControl: false,
          dragRotate: false,
          pitchWithRotate: false,
        });
        map.on('zoom', () => setZoom(map?.getZoom() ?? null));
        map.on('load', () => setZoom(map?.getZoom() ?? null));
        map.on('error', (e) => {
          const err = (e as unknown as { error?: Error }).error;
          setError(err?.message ?? 'The map reported an error.');
        });
        window.__viewer = { map, maxZoom: h.maxZoom, bounds };
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });

    return () => {
      cancelled = true;
      map?.remove();
      delete window.__viewer;
      removeProtocol('pmtiles');
    };
  }, [archive]);

  return (
    <div className="viewer">
      <header>
        <h1>Drawing viewer</h1>
        <span className="muted" data-testid="archive">
          {archive.split('/').pop()}
        </span>
        <button
          type="button"
          aria-pressed={theme === 'dark'}
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          Dark theme
        </button>
      </header>
      <main className="map-frame">
        <div ref={container} className="map" data-testid="map" aria-label="Drawing" role="application" />
      </main>
      <footer>
        {error ? <p role="alert">{error}</p> : <span className="muted">Scroll to zoom, drag to pan.</span>}
        <output aria-live="polite" data-testid="zoom" className="muted">
          {zoom === null ? 'zoom --' : `zoom ${zoom.toFixed(2)}`}
        </output>
      </footer>
    </div>
  );
}
