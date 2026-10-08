import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  PairBornCoulWolf, PairBornCoulDsf, PairBornCoulWolfCS, PairBornCoulDsfCS,
} from '../src/engine/force/pair/born_coul18';
import { PAIRS as W18_MISC_PAIRS } from '../src/engine/registry/pair_misc18';
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

describe('born/coul/{wolf,dsf}/cs: weight-0 core/shell pair at r = 0', () => {
  it('born/coul/wolf/cs returns a finite r -> 0 limit, the base style is NaN', () => {
    const s = make(new PairBornCoulWolfCS(), ['0.6', '3.0'], COEFFS);
    const zero = s.single!(0, 0, 1, 2, 0, 0, 0, 1.0, -1.0);
    expect(Number.isFinite(zero.eng)).toBe(true);
    expect(zero.fforce).toBe(0);
    const base = make(new PairBornCoulWolf(), ['0.6', '3.0'], COEFFS);
    expect(Number.isNaN(base.single!(0, 0, 1, 2, 0, 0, 0, 1.0, -1.0).eng)).toBe(true);
  });

  it('born/coul/dsf/cs returns a finite r -> 0 limit, the base style is NaN', () => {
    const s = make(new PairBornCoulDsfCS(), ['0.6', '3.0'], COEFFS);
    const zero = s.single!(0, 0, 1, 2, 0, 0, 0, 1.0, -1.0);
    expect(Number.isFinite(zero.eng)).toBe(true);
    expect(zero.fforce).toBe(0);
    const base = make(new PairBornCoulDsf(), ['0.6', '3.0'], COEFFS);
    expect(Number.isNaN(base.single!(0, 0, 1, 2, 0, 0, 0, 1.0, -1.0).eng)).toBe(true);
  });

  it('the r = 0 value is the limit approached from r > 0', () => {
    for (const cs of [new PairBornCoulWolfCS(), new PairBornCoulDsfCS()] as Pair[]) {
      make(cs, ['0.6', '3.0'], COEFFS);
      const zero = cs.single!(0, 0, 1, 2, 0, 0, 0, 1.0, -1.0).eng;
      const tiny = cs.single!(0, 0, 1, 2, 1e-10 * 1e-10, 0, 0, 1.0, -1.0).eng;
      expect(close(zero, tiny, 1e-6, 1e-9), `${cs.name}`).toBe(true);
    }
  });

  it('agrees with the base style for a weight-0 pair away from r = 0', () => {
    for (const [cs, base] of [[new PairBornCoulWolfCS(), new PairBornCoulWolf()], [new PairBornCoulDsfCS(), new PairBornCoulDsf()]] as [Pair, Pair][]) {
      make(cs, ['0.6', '3.0'], COEFFS);
      make(base, ['0.6', '3.0'], COEFFS);
      for (const r of [0.5, 1.4, 2.5]) {
        const a = cs.single!(0, 0, 1, 2, r * r, 0, 1, 1.0, -1.0);
        const b = base.single!(0, 0, 1, 2, r * r, 0, 1, 1.0, -1.0);
        expect(close(a.eng, b.eng, 1e-12, 1e-12), `${cs.name} e r=${r}`).toBe(true);
        expect(close(a.fforce, b.fforce, 1e-12, 1e-12), `${cs.name} f r=${r}`).toBe(true);
      }
    }
  });

  it('the weight-0 /cs force obeys the base Wolf/dsf conventions', () => {
    wolfShift(make(new PairBornCoulWolfCS(), ['0.6', '3.0'], COEFFS), 1, 2, [0.4, 0.9, 2.2], 1.0, -1.0);
    finiteDiff(make(new PairBornCoulDsfCS(), ['0.6', '3.0'], COEFFS), 1, 2, [0.4, 0.9, 2.2], 1.0, -1.0, 0);
  });

  it('a weight-1 pair keeps the base damped Coulomb and the Born term', () => {
    const cs = make(new PairBornCoulWolfCS(), ['0.6', '3.0'], COEFFS);
    const base = make(new PairBornCoulWolf(), ['0.6', '3.0'], COEFFS);
    const r = 1.1;
    const a = cs.single!(0, 0, 1, 2, r * r, 1, 1, 0.5, -0.5);
    const b = base.single!(0, 0, 1, 2, r * r, 1, 1, 0.5, -0.5);
    expect(close(a.eng, b.eng, 1e-12, 1e-12)).toBe(true);
  });
});

describe('born/coul/{wolf,dsf}/cs: errors and registration', () => {
  it('bad pair_style arguments throw a StyleError naming the /cs style', () => {
    expect(() => new PairBornCoulWolfCS().settings([], ctx)).toThrow(/usage: pair_style born\/coul\/wolf\/cs/);
    expect(() => new PairBornCoulWolfCS().settings(['0.6'], ctx)).toThrow(/usage: pair_style born\/coul\/wolf\/cs/);
    expect(() => new PairBornCoulWolfCS().settings(['0.6', '3.0', '4.0', '5.0'], ctx)).toThrow(/usage: pair_style born\/coul\/wolf\/cs/);
    expect(() => new PairBornCoulDsfCS().settings([], ctx)).toThrow(/usage: pair_style born\/coul\/dsf\/cs/);
    expect(() => new PairBornCoulDsfCS().settings(['0.6', '3.0', '4.0', '5.0'], ctx)).toThrow(/usage: pair_style born\/coul\/dsf\/cs/);
  });

  it('no per-pair Coulomb cutoff is accepted, so a 9th coeff throws', () => {
    const s = new PairBornCoulWolfCS();
    s.settings(['0.6', '3.0'], ctx);
    s.allocate(2);
    expect(() => s.coeff(['1', '1', '1.0', '0.1', '1.0', '1.0', '0.5', '2.4', '2.8'], ctx)).toThrow(/usage: pair_coeff/);
  });

  it('extract cut_coul follows the pair_style Coulomb cutoff', () => {
    expect(make(new PairBornCoulWolfCS(), ['0.6', '2.5', '3.0'], COEFFS).extract('cut_coul')).toBe(3.0);
    expect(make(new PairBornCoulDsfCS(), ['0.6', '3.0'], COEFFS).extract('cut_coul')).toBe(3.0);
  });

  it('the wave-18 registry exposes the /cs factories', () => {
    expect(W18_MISC_PAIRS['born/coul/wolf/cs']!().name).toBe('born/coul/wolf/cs');
    expect(W18_MISC_PAIRS['born/coul/dsf/cs']!().name).toBe('born/coul/dsf/cs');
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
