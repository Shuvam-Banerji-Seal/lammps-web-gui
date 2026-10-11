import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * Chunk names of engine code (src/engine, and the GPL-2.0 ports in src/engine-gpl): in the engine worker build, the lazily loaded style registries
 * (src/engine/lazyStyles.ts) and their shared modules are engine-*.js; in the page build, chunks made only of
 * engine code belong to the main-thread fallback copy of the engine and are host-*.js.
 * scripts/check-bundle-size.mjs budgets them by these prefixes.
 */
const engineOnly = (chunk: { moduleIds: readonly string[] }): boolean =>
  chunk.moduleIds.length > 0 && chunk.moduleIds.every((id) => id.includes('/src/engine/') || id.includes('/src/engine-gpl/'));
const workerChunkName = (chunk: { moduleIds: readonly string[] }): string =>
  engineOnly(chunk) ? 'assets/engine-[name]-[hash].js' : 'assets/[name]-[hash].js';
const pageChunkName = (chunk: { moduleIds: readonly string[] }): string =>
  engineOnly(chunk) ? 'assets/host-[name]-[hash].js' : 'assets/[name]-[hash].js';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: process.env.NODE_ENV === 'production'
    ? '/lammps-web-gui/'   // GitHub Pages project path
    : '/',                         // local dev
  // Module workers (every worker is created with { type: 'module' }) so the engine worker can load its
  // rarely used style registries on demand (src/engine/lazyStyles.ts) instead of inlining them.
  worker: {
    format: 'es',
    rollupOptions: { output: { chunkFileNames: workerChunkName } },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
        chunkFileNames: pageChunkName,
        // Split the heavy 3D stack so app code updates don't bust the three.js cache.
        // Function form required by Vite 8 / Rolldown. Order matters:
        // @react-three must match before generic react.
        manualChunks(id: string) {
          if (id.includes('node_modules/@react-three')) return 'r3f';
          if (id.includes('node_modules/three')) return 'three';
          if (
            id.includes('node_modules/react-dom') ||
            id.includes('node_modules/react/') ||
            id.includes('node_modules/scheduler')
          ) {
            return 'react';
          }
          return undefined;
        },
      },
    },
  },
});
