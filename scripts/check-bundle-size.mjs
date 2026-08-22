#!/usr/bin/env node
/**
 * Bundle-size budget gate.
 * Fails (exit 1) when gzipped JS output exceeds the budget, preventing
 * silent performance regressions from reaching main.
 *
 * Budgets (gzipped):
 *   total JS  <= 420 KB   (three.js dominates; app code ~30 KB)
 *   any chunk <= 300 KB   (currently the r3f/three vendor chunk)
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';

const DIST = new URL('../dist/assets', import.meta.url).pathname;

const TOTAL_BUDGET = 420 * 1024;
const CHUNK_BUDGET = 300 * 1024;

let total = 0;
let failed = false;
const rows = [];

for (const name of readdirSync(DIST)) {
  if (!name.endsWith('.js')) continue;
  const path = join(DIST, name);
  const size = gzipSync(readFileSync(path)).length;
  total += size;
  rows.push({ name, kb: (size / 1024).toFixed(1) });
  if (size > CHUNK_BUDGET) {
    failed = true;
    console.error(`✗ chunk over budget: ${name} = ${(size / 1024).toFixed(1)} KB gzipped (limit ${CHUNK_BUDGET / 1024} KB)`);
  }
}

console.table(rows.map(r => ({ chunk: r.name, 'gzip KB': r.kb })));
console.log(`total gzip: ${(total / 1024).toFixed(1)} KB / budget ${TOTAL_BUDGET / 1024} KB`);

if (total > TOTAL_BUDGET) {
  console.error('✗ TOTAL JS BUDGET EXCEEDED');
  failed = true;
}

process.exit(failed ? 1 : 0);
