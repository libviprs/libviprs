import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';

const require = createRequire(import.meta.url);
const WORKER_FILES = ['maplibre-gl-worker.mjs', 'maplibre-gl-shared.mjs'];

/**
 * MapLibre GL JS 6 runs its tile worker from a separate module file that
 * imports a shared one beside it. A bundler cannot see that, so both files are
 * served from /maplibre/ and App.tsx points `setWorkerUrl` there.
 */
function maplibreWorker(): Plugin {
  const dist = (name: string) => require.resolve(`maplibre-gl/dist/${name}`);
  return {
    name: 'maplibre-worker-files',
    configureServer(server) {
      server.middlewares.use('/maplibre', (req, res, next) => {
        const name = (req.url ?? '').replace(/^\//, '').split('?')[0] ?? '';
        if (!WORKER_FILES.includes(name)) return next();
        res.setHeader('content-type', 'text/javascript');
        res.end(readFileSync(dist(name)));
      });
    },
    generateBundle() {
      for (const name of WORKER_FILES) {
        this.emitFile({ type: 'asset', fileName: `maplibre/${name}`, source: readFileSync(dist(name)) });
      }
    },
  };
}

// The style lives in the Rust crate (src/vector/black_on_white.style.json) so
// the web harness, the Rust tests and the mobile documentation read one file.
export default defineConfig({
  plugins: [react(), maplibreWorker()],
  server: { fs: { allow: ['../..'] } },
  build: { sourcemap: false },
});
