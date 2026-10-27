#!/usr/bin/env node
/**
 * Bundle-size budget gate.
 * Fails (exit 1) when gzipped JS output exceeds the budget, preventing
 * silent performance regressions from reaching main.
 *
 * Budgets (gzipped):
 *   initial JS <= 420 KB  (what index.html loads: entry + modulepreloads;
 *                          three.js dominates)
 *   lazy JS    <= 140 KB  (code-split modules and small workers, e.g. the
 *                          MD notebook UI with its structure import and example
 *                          notebooks and its run files (DCD and XTC frames
 *                          counted, saved as bytes), the browser-storage panel,
 *                          and the pair worker with the threaded neighbor
 *                          search and the pair, EAM and three-body kernels it
 *                          runs on the cores, fetched only for a threaded run)
 *   engine     <= 400 KB  (the notebook's MD engine: its Web Worker with the
 *                          engine-* chunks it imports statically, and the
 *                          same code split out as the main-thread fallback,
 *                          counted once; it loads only when the notebook
 *                          runs and implements the LAMMPS input language)
 *   engine lazy chunk <= 60 KB each (engine-*.js: style registries the
 *                          worker loads on first use, src/engine/lazyStyles.ts;
 *                          their total is reported, not capped, because a
 *                          script loads only the groups it names)
 *   any chunk  <= 300 KB  (currently the r3f/three vendor chunk; the engine
 *                          worker has its own limit above)
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';

const DIST = new URL('../dist', import.meta.url).pathname;
const ASSETS = join(DIST, 'assets');

const INITIAL_BUDGET = 420 * 1024;
const LAZY_BUDGET = 140 * 1024;
const CHUNK_BUDGET = 300 * 1024;
const ENGINE_BUDGET = 400 * 1024;
const ENGINE_LAZY_CHUNK_BUDGET = 60 * 1024;
// the engine worker and its main-thread fallback chunk (host-*.js) hold the same code
const isEngine = (name) => /^engine\.worker-/.test(name);
const isEngineFallback = (name) => /^host-/.test(name);
// lazily loaded style registries of the engine worker (vite.config.ts names them engine-*)
const isEngineLazy = (name) => /^engine-/.test(name);

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
let engineLazy = 0;
let failed = false;
const rows = [];

// A host-* chunk (engine code in the page build) with no engine-* twin in the worker build is shared with the
// notebook UI (e.g. atoms.ts through engine/view), so the page loads it: it counts as lazy, not as the fallback.
// The worker entry statically imports the engine chunks it shares with the lazy groups (engine-atoms, ...):
// those load at start-up with it, so they count as engine, not as lazy styles.
const staticImports = (name, seen = new Set()) => {
  if (seen.has(name)) return seen;
  seen.add(name);
  const text = readFileSync(join(ASSETS, name), 'utf8');
  for (const m of text.matchAll(/(?:from|import)\s*"\.\/(engine-[^"]+\.js)"/g)) staticImports(m[1], seen);
  return seen;
};
const workerEntry = readdirSync(ASSETS).find((n) => isEngine(n));
const engineStartup = workerEntry ? staticImports(workerEntry) : new Set();
const base = (name) => name.replace(/^(engine|host)-/, '').replace(/-[A-Za-z0-9_-]{8}\.js$/, '');
const workerLazy = new Set(readdirSync(ASSETS).filter((n) => /^engine-/.test(n)).map(base));
const isFallbackCopy = (name) => isEngineFallback(name) && (/^host-host-/.test(name) || workerLazy.has(base(name)));

for (const name of readdirSync(ASSETS)) {
  if (!name.endsWith('.js')) continue;
  const size = gzipSync(readFileSync(join(ASSETS, name))).length;
  const isInitial = initialNames.has(name);
  const kind = isInitial ? 'initial' : isEngine(name) || engineStartup.has(name) ? 'engine' : isFallbackCopy(name) ? 'engine (fallback copy)'
    : isEngineLazy(name) ? 'engine (lazy styles)' : 'lazy';
  if (kind === 'initial') initial += size; else if (kind === 'engine') engine += size; else if (kind === 'lazy') lazy += size;
  else if (kind === 'engine (lazy styles)') {
    engineLazy += size;
    if (size > ENGINE_LAZY_CHUNK_BUDGET) {
      failed = true;
      console.error(`✗ engine lazy chunk over budget: ${name} = ${(size / 1024).toFixed(1)} KB gzipped (limit ${ENGINE_LAZY_CHUNK_BUDGET / 1024} KB)`);
    }
  }
  rows.push({ chunk: name, load: kind, 'gzip KB': (size / 1024).toFixed(1) });
  if (size > CHUNK_BUDGET && kind !== 'engine' && kind !== 'engine (fallback copy)' && kind !== 'engine (lazy styles)') {
    failed = true;
    console.error(`✗ chunk over budget: ${name} = ${(size / 1024).toFixed(1)} KB gzipped (limit ${CHUNK_BUDGET / 1024} KB)`);
  }
}

console.table(rows);
console.log(`initial gzip: ${(initial / 1024).toFixed(1)} KB / budget ${INITIAL_BUDGET / 1024} KB`);
console.log(`lazy gzip:    ${(lazy / 1024).toFixed(1)} KB / budget ${LAZY_BUDGET / 1024} KB`);
console.log(`engine gzip:  ${(engine / 1024).toFixed(1)} KB / budget ${ENGINE_BUDGET / 1024} KB`);
console.log(`engine lazy styles gzip: ${(engineLazy / 1024).toFixed(1)} KB in total (each chunk <= ${ENGINE_LAZY_CHUNK_BUDGET / 1024} KB)`);

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
