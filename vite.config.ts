import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: process.env.NODE_ENV === 'production'
    ? '/lammps_data_web_viewer/'   // GitHub Pages
    : '/',                 // local dev
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false
  }
});
