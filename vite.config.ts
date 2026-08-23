import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: process.env.NODE_ENV === 'production'
    ? '/molecule3d/'   // GitHub Pages project path
    : '/',                         // local dev
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
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
