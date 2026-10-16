import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

/*
 * The engine runs in the browser, where `process` does not exist (Vite defines only process.env.NODE_ENV), so a
 * `process.env.X` read in engine code throws a ReferenceError in the app even though it works under Node in tests.
 * A temporary debugging knob of this kind once reached the tempering swap (temper.ts); this test keeps them out.
 */
const filesUnder = (dir: string): string[] =>
  readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? filesUnder(p) : /\.(ts|tsx)$/.test(n) ? [p] : [];
  });

describe('engine code has no process.env reads', () => {
  it('src/engine and src/engine-gpl never read process.env', () => {
    const offenders = ['src/engine', 'src/engine-gpl']
      .flatMap((d) => filesUnder(d))
      .filter((f) => /\bprocess\.env\b/.test(readFileSync(f, 'utf8')));
    expect(offenders).toEqual([]);
  });
});
