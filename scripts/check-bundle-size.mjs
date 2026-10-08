#!/usr/bin/env node
/**
 * Bundle-size budget gate.
 * Fails (exit 1) when gzipped JS output exceeds the budget, preventing
 * silent performance regressions from reaching main.
 *
 * Budgets (gzipped):
 *   initial JS <= 420 KB  (what index.html loads: entry + modulepreloads;
 *                          three.js dominates)
 *   lazy JS    <= 120 KB  (code-split modules and small workers, e.g. the
 *                          MD notebook UI, fetched on use)
 *   engine     <= 400 KB  (the notebook's MD engine: its Web Worker, and the
 *                          same code split out as the main-thread fallback,
 *                          counted once; it loads only when the notebook
 *                          runs and implements the LAMMPS input language)
 *   any chunk  <= 300 KB  (currently the r3f/three vendor chunk; the engine
 *                          worker has its own limit above)
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';

const DIST = new URL('../dist', import.meta.url).pathname;
const ASSETS = join(DIST, 'assets');

const INITIAL_BUDGET = 420 * 1024;
const LAZY_BUDGET = 120 * 1024;
const CHUNK_BUDGET = 300 * 1024;
const ENGINE_BUDGET = 400 * 1024;
// the engine worker and its main-thread fallback chunk (host-*.js) hold the same code
const isEngine = (name) => /^engine\.worker-/.test(name);
const isEngineFallback = (name) => /^host-/.test(name);

// Chunks index.html pulls in up front: the module entry and its preloads.
const html = readFileSync(join(DIST, 'index.html'), 'utf8');
const initialNames = new Set(
  [...html.matchAll(/<(?:script[^>]*\bsrc|link[^>]*\brel="modulepreload"[^>]*\bhref)="([^"]+\.js)"/g)]
    .map((m) => m[1].split('/').pop()),
);
if (initialNames.size === 0) {
  console.error('✗ found no initial JS in dist/index.html — is this a production build?');
  process.exit(1);
}

let initial = 0;
let lazy = 0;
let engine = 0;
let failed = false;
const rows = [];

for (const name of readdirSync(ASSETS)) {
  if (!name.endsWith('.js')) continue;
  const size = gzipSync(readFileSync(join(ASSETS, name))).length;
  const isInitial = initialNames.has(name);
  const kind = isInitial ? 'initial' : isEngine(name) ? 'engine' : isEngineFallback(name) ? 'engine (fallback copy)' : 'lazy';
  if (kind === 'initial') initial += size; else if (kind === 'engine') engine += size; else if (kind === 'lazy') lazy += size;
  rows.push({ chunk: name, load: kind, 'gzip KB': (size / 1024).toFixed(1) });
  if (size > CHUNK_BUDGET && kind !== 'engine' && kind !== 'engine (fallback copy)') {
    failed = true;
    console.error(`✗ chunk over budget: ${name} = ${(size / 1024).toFixed(1)} KB gzipped (limit ${CHUNK_BUDGET / 1024} KB)`);
  }
}

console.table(rows);
console.log(`initial gzip: ${(initial / 1024).toFixed(1)} KB / budget ${INITIAL_BUDGET / 1024} KB`);
console.log(`lazy gzip:    ${(lazy / 1024).toFixed(1)} KB / budget ${LAZY_BUDGET / 1024} KB`);
console.log(`engine gzip:  ${(engine / 1024).toFixed(1)} KB / budget ${ENGINE_BUDGET / 1024} KB`);

if (initial > INITIAL_BUDGET) {
  console.error('✗ INITIAL JS BUDGET EXCEEDED');
  failed = true;
}
if (engine > ENGINE_BUDGET) {
  console.error('✗ ENGINE JS BUDGET EXCEEDED');
  failed = true;
}
if (lazy > LAZY_BUDGET) {
  console.error('✗ LAZY JS BUDGET EXCEEDED');
  failed = true;
}

process.exit(failed ? 1 : 0);
