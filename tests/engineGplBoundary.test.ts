import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';

// AGENTS.md rule 7: code ported from LAMMPS lives only in src/engine-gpl/ (GPL-2.0). The rest of src/ reaches it
// only through the lazy style registries of src/engine/lazyStyles.ts, and every ported file names its source.
const ROOT = join(__dirname, '..');
const SRC = join(ROOT, 'src');
const GPL = join(SRC, 'engine-gpl');

const walk = (dir: string): string[] =>
  readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : /\.(ts|tsx|js|mjs)$/.test(n) ? [p] : [];
  });

/** Import specifiers of a source file (static and dynamic). */
const importsOf = (text: string): string[] =>
  [...text.matchAll(/(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)['"]([^'"]+)['"]/g)].map((m) => m[1]);

describe('the GPL-2.0 port directory (src/engine-gpl)', () => {
  it('has its own licence and README', () => {
    expect(existsSync(join(GPL, 'LICENSE'))).toBe(true);
    expect(readFileSync(join(GPL, 'LICENSE'), 'utf8')).toMatch(/GNU GENERAL PUBLIC LICENSE\s+Version 2/);
    expect(existsSync(join(GPL, 'README.md'))).toBe(true);
  });

  it('is imported only by src/engine/lazyStyles.ts', () => {
    const offenders: string[] = [];
    for (const f of walk(SRC)) {
      if (f.startsWith(GPL + '/') || relative(SRC, f) === join('engine', 'lazyStyles.ts')) continue;
      if (importsOf(readFileSync(f, 'utf8')).some((s) => s.includes('engine-gpl'))) offenders.push(relative(ROOT, f));
    }
    expect(offenders).toEqual([]);
  });

  it('names the LAMMPS source and commit at the top of every ported file', () => {
    const missing: string[] = [];
    for (const f of walk(GPL)) {
      const head = readFileSync(f, 'utf8').slice(0, 1500);
      if (!/Ported from LAMMPS/i.test(head) || !/b332f28/.test(head) || !/GPL-2\.0/.test(head)) missing.push(relative(ROOT, f));
    }
    expect(missing).toEqual([]);
  });
});
