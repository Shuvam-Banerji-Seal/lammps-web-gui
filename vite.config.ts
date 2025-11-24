import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // Use relative paths for assets to ensure it works on GitHub Pages and Flask
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false
  }
});
