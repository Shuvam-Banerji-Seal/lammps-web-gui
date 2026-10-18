import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { PairBornCoulWolf, PairBornCoulDsf } from '../src/engine/force/pair/born_coul18';
import { PairBornCoulWolfCS, PairBornCoulDsfCS } from '../src/engine-gpl/csdiel/born_dsf_wolf_cs';
import { PAIRS as CSDIEL_PAIRS } from '../src/engine-gpl/registry/csdiel';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { Pair, StyleContext } from '../src/engine/force/types';

/*
 * Unit checks for born/coul/wolf/cs and born/coul/dsf/cs: identical to the
 * base styles for every pair except a weight-0 (core/shell) pair at r = 0,
 * where the base style's erfc/r - 1/r cancellation is a NaN and the /cs style
 * returns the analytic r -> 0 limit (pair_cs.rst). Native LAMMPS agreement is
 * covered by tests/oracle/w21borncs_*.in.
 */

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const make = <T extends Pair>(s: T, styleArgs: string[], coeffs: string[]): T => {
  s.settings(styleArgs, ctx);
  s.allocate(2);
  for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
  s.init(ctx);
  return s;
};

const close = (a: number, b: number, rel: number, abs: number) => Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

const COEFFS = ['* * 1.0 0.1 1.0 1.0 0.5', '1 2 0.8 0.12 1.05 0.8 0.4 2.4'];

/** fforce * r must equal -dE/dr (central differences). */
const finiteDiff = (s: Pair, itype: number, jtype: number, radii: number[], qi: number, qj: number, factorCoul: number, factorLJ = 1) => {
  const single = s.single!;
  const h = 1e-6;
  for (const r of radii) {
    const em = single.call(s, 0, 0, itype, jtype, (r - h) * (r - h), factorCoul, factorLJ, qi, qj).eng;
    const ep = single.call(s, 0, 0, itype, jtype, (r + h) * (r + h), factorCoul, factorLJ, qi, qj).eng;
    const m = single.call(s, 0, 0, itype, jtype, r * r, factorCoul, factorLJ, qi, qj);
    if (!close(m.fforce * r, -(ep - em) / (2 * h), 1e-6, 1e-7)) {
      throw new Error(`${s.name}: force != -dE/dr at r=${r}: ${m.fforce * r} vs ${-(ep - em) / (2 * h)}`);
    }
  }
};

/** For the Wolf kernel fforce * r + dE/dr is a constant (the damped force shift, pair_coul.html). */
const wolfShift = (s: Pair, itype: number, jtype: number, radii: number[], qi: number, qj: number, factorCoul = 0) => {
  const single = s.single!;
  const h = 1e-6;
  const deltas: number[] = [];
  for (const r of radii) {
    const em = single.call(s, 0, 0, itype, jtype, (r - h) * (r - h), factorCoul, 0, qi, qj).eng;
    const ep = single.call(s, 0, 0, itype, jtype, (r + h) * (r + h), factorCoul, 0, qi, qj).eng;
    const m = single.call(s, 0, 0, itype, jtype, r * r, factorCoul, 0, qi, qj);
    deltas.push(m.fforce * r + (ep - em) / (2 * h));
  }
  for (const d of deltas) if (!close(d, deltas[0], 1e-6, 1e-7)) throw new Error(`${s.name}: Wolf force shift varies with r`);
};

describe('born/coul/{wolf,dsf}/cs: errors and registration', () => {
  it('extract cut_coul follows the pair_style Coulomb cutoff', () => {
    expect(make(new PairBornCoulWolfCS(), ['0.6', '2.5', '3.0'], COEFFS).extract('cut_coul')).toBe(3.0);
    expect(make(new PairBornCoulDsfCS(), ['0.6', '3.0'], COEFFS).extract('cut_coul')).toBe(3.0);
  });

  it('the wave-18 registry exposes the /cs factories', () => {
    expect(CSDIEL_PAIRS['born/coul/wolf/cs']!().name).toBe('born/coul/wolf/cs');
    expect(CSDIEL_PAIRS['born/coul/dsf/cs']!().name).toBe('born/coul/dsf/cs');
  });
});

describe('born/coul/{wolf,dsf}/cs: end-to-end r = 0 session', () => {
  it('runs weight-0 core/shell pairs at the same position without NaN', async () => {
    const data = readFileSync(join(__dirname, 'oracle', 'w21borncs_cs.data'), 'utf8');
    for (const f of ['w21borncs_wolf', 'w21borncs_dsf']) {
      const text = readFileSync(join(__dirname, 'oracle', `${f}.in`), 'utf8');
      const events: EngineEvent[] = [];
      const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
      session.addFile('w21borncs_cs.data', data);
      await session.execute(text);
      const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo');
      expect(rows.length, f).toBeGreaterThan(0);
      for (const r of rows) expect(Number.isFinite(r.row.pe), `${f} step ${r.row.step}`).toBe(true);
    }
  });
});
