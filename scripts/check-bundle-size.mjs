#!/usr/bin/env node
/**
 * Bundle-size budget gate.
 * Fails (exit 1) when gzipped JS output exceeds the budget, preventing
 * silent performance regressions from reaching main.
 *
 * Budgets (gzipped):
 *   initial JS <= 420 KB  (what index.html loads: entry + modulepreloads;
 *                          three.js dominates)
 *   lazy JS    <= 120 KB  (code-split modules and workers, e.g. the MD
 *                          notebook and its engine worker, fetched on use)
 *   any chunk  <= 300 KB  (currently the r3f/three vendor chunk)
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';

const DIST = new URL('../dist', import.meta.url).pathname;
const ASSETS = join(DIST, 'assets');

const INITIAL_BUDGET = 420 * 1024;
const LAZY_BUDGET = 120 * 1024;
const CHUNK_BUDGET = 300 * 1024;

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
let failed = false;
const rows = [];

for (const name of readdirSync(ASSETS)) {
  if (!name.endsWith('.js')) continue;
  const size = gzipSync(readFileSync(join(ASSETS, name))).length;
  const isInitial = initialNames.has(name);
  if (isInitial) initial += size; else lazy += size;
  rows.push({ chunk: name, load: isInitial ? 'initial' : 'lazy', 'gzip KB': (size / 1024).toFixed(1) });
  if (size > CHUNK_BUDGET) {
    failed = true;
    console.error(`✗ chunk over budget: ${name} = ${(size / 1024).toFixed(1)} KB gzipped (limit ${CHUNK_BUDGET / 1024} KB)`);
  }
}

console.table(rows);
console.log(`initial gzip: ${(initial / 1024).toFixed(1)} KB / budget ${INITIAL_BUDGET / 1024} KB`);
console.log(`lazy gzip:    ${(lazy / 1024).toFixed(1)} KB / budget ${LAZY_BUDGET / 1024} KB`);

if (initial > INITIAL_BUDGET) {
  console.error('✗ INITIAL JS BUDGET EXCEEDED');
  failed = true;
}
if (lazy > LAZY_BUDGET) {
  console.error('✗ LAZY JS BUDGET EXCEEDED');
  failed = true;
}

process.exit(failed ? 1 : 0);
