import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { PairParams, type StyleContext } from '../src/engine/force/types';
import { PairBeck } from '../src/engine-gpl/xpair1/pair_beck';
import { PairBornGauss } from '../src/engine-gpl/xpair1/pair_born_gauss';
import { PairBuckMDF, PairLennardMDF, PairLJMDF } from '../src/engine-gpl/xpair1/pair_mdf';
import { PairCosineSquared } from '../src/engine-gpl/xpair1/pair_cosine_squared';
import { PairCoulCutGlobal, PairCoulExclude, PairCoulSlaterCut } from '../src/engine-gpl/xpair1/pair_coul_family';
import { PairUFM } from '../src/engine-gpl/xpair1/pair_ufm';
import { PairWFCut } from '../src/engine-gpl/xpair1/pair_wf_cut';
import { mixDistance, mixEnergy, powint } from '../src/engine-gpl/xpair1/common';

// Errors and coefficient logic of the EXTRA-PAIR ports (src/engine-gpl/xpair1). Energies and forces are checked against
// native LAMMPS by the tests/oracle/w219xp_* cases.

const ctx = (logs: string[] = []): StyleContext => ({ s: null, readFile: () => '', log: (t) => logs.push(t) });

/** Runs a script on a fresh session and returns the error messages it emitted. */
const errorsOf = async (script: string): Promise<string[]> => {
  const errors: string[] = [];
  const s = new Session({
    emit: (e: { kind: string; message?: string }) => { if (e.kind === 'error') errors.push(e.message ?? ''); },
    writeFile: () => {},
  } as never);
  try {
    await s.execute(`units lj
lattice fcc 0.8442
region b block 0 3 0 3 0 3
create_box 2 b
create_atoms 1 box
mass * 1.0
${script}
`);
  } catch {
    // the failing command is also emitted as an error line; the throw only ends the script
  }
  return errors;
};

const expectError = async (script: string, text: string): Promise<void> => {
  const errors = await errorsOf(script);
  expect(errors.some((m) => m.includes(text)), `expected "${text}" in ${JSON.stringify(errors)}`).toBe(true);
};

describe('argument errors of the xpair1 styles (native messages)', () => {
  it('beck, born/gauss, buck/mdf, lennard/mdf, lj/mdf, cosine/squared, ufm, wf/cut, coul/cut/global, coul/exclude, coul/slater/cut', async () => {
    await expectError('pair_style beck', 'Illegal pair_style command');
    await expectError('pair_style beck 2.5\npair_coeff 1 1 1 1 1 1', 'Incorrect args for pair coefficients');
    await expectError('pair_style born/gauss 2.5 3.0', 'Pair style born/gauss must have exactly one argument');
    await expectError('pair_style buck/mdf 2.0 2.5\npair_coeff 1 1 1.0 0.0 1.0', 'Incorrect args for pair coefficients');
    await expectError('pair_style lj/mdf 3.0 2.5', 'Illegal pair_style command');
    await expectError('pair_style lennard/mdf 2.0 2.5\npair_coeff 1 1 1 1 3.0 2.5', 'Illegal pair_coeff command');
    await expectError('pair_style cosine/squared 2.5\npair_coeff 1 1 1.0 1.0 1.0', 'cutoff = sigma w/o wca');
    await expectError('pair_style cosine/squared 2.5\npair_coeff 1 1 1.0 1.0 2.0 nowca', 'unknown option');
    await expectError('pair_style cosine/squared 2.5\npair_modify mix arithmetic', 'pair_modify mix not supported for pair_style cosine/squared');
    // both I,I pairs set (an unset one stops first at Pair::init): the I,J pair cannot be mixed
    await expectError('pair_style cosine/squared 2.5\npair_coeff 1 1 1.0 1.0\npair_coeff 2 2 1.0 1.0\nrun 0', 'Mixing not supported in pair_style cosine/squared');
    await expectError('pair_style ufm 2.5\npair_coeff 1 1 1.0', 'Incorrect args for pair coefficients');
    await expectError('pair_style wf/cut 2.5\npair_coeff 1 1 1.0 1.0 1.5 2', "Expected integer parameter instead of '1.5' in input script or data file");
    await expectError('pair_style coul/cut/global 2.5\npair_coeff 1 1 2.0', 'Incorrect args for pair coefficients');
    await expectError('pair_style coul/exclude 2.5\npair_coeff 1 1 extra', 'Incorrect args for pair coefficients');
    await expectError('pair_style coul/slater/cut 0.7\npair_coeff * *', 'Illegal pair_style command');
    await expectError('pair_style coul/slater/cut 0.7 2.5 1.0', 'Illegal pair_style command');
  });
});

describe('cosine/squared cutoff = sigma', () => {
  it('is a warning with wca (WCA only) and an error without it', () => {
    const logs: string[] = [];
    const c = new PairCosineSquared();
    c.allocate(1);
    c.settings(['2.5']);
    c.coeff(['1', '1', '1.0', '1.0', '1.0', 'wca'], ctx(logs));
    expect(logs).toEqual(['WARNING: Cosine/squared set to WCA only (cutoff = sigma)']);
  });
});

describe('mixing and coefficient stores', () => {
  it('lj/mdf mixes epsilon, sigma and the cutoffs with the pair_modify mix rule', () => {
    const p = new PairLJMDF();
    p.allocate(2);
    p.settings(['2.0', '2.5']);
    p.coeff(['1', '1', '1.0', '1.0']);
    p.coeff(['2', '2', '4.0', '2.0', '1.8', '2.2']);
    p.init(ctx());
    const nt = 3;
    // geometric (default): sqrt(1*4), sqrt(1*2), sqrt(2.0*1.8), sqrt(2.5*2.2)
    expect(p.extract('epsilon')).toBeInstanceOf(Float64Array);
    const eps = p.extract('epsilon') as Float64Array, sig = p.extract('sigma') as Float64Array;
    expect(eps[1 * nt + 2]).toBeCloseTo(2, 14);
    expect(sig[1 * nt + 2]).toBeCloseTo(Math.sqrt(2), 14);
    expect(p.cut[1 * nt + 2]).toBeCloseTo(Math.sqrt(2.5 * 2.2), 14);
    expect(p.cut[2 * nt + 1]).toBe(p.cut[1 * nt + 2]);
  });

  it('sixth-power mixing uses the native formula (mix_energy, mix_distance)', () => {
    const e = mixEnergy('sixthpower', 1.0, 4.0, 1.0, 2.0);
    expect(e).toBeCloseTo(2 * Math.sqrt(4) * 1 * 8 / (1 + 64), 14);
    expect(mixDistance('sixthpower', 1.0, 1.0)).toBe(1);
    expect(mixDistance('arithmetic', 1.0, 3.0)).toBe(2);
    expect(powint(2, -2)).toBe(0.25);
    expect(powint(0, 3)).toBe(0);
  });

  it('born/gauss and buck/mdf keep their own extract() keys, cosine/squared has none', () => {
    const b = new PairBornGauss();
    b.allocate(1);
    b.settings(['2.5']);
    b.coeff(['1', '1', '1', '1', '1', '1', '1']);
    b.init(ctx());
    expect(b.extract('biga0')).toBeInstanceOf(Float64Array);
    expect(b.extract('epsilon')).toBeUndefined();

    const m = new PairBuckMDF();
    m.allocate(1);
    m.settings(['2.0', '2.5']);
    m.coeff(['1', '1', '3.0', '0.3', '0.5']);
    m.init(ctx());
    expect((m.extract('a') as Float64Array)[1 * 2 + 1]).toBe(3);
    expect((m.extract('c') as Float64Array)[1 * 2 + 1]).toBe(0.5);

    const l = new PairLennardMDF();
    l.allocate(1);
    l.settings(['2.0', '2.5']);
    l.coeff(['1', '1', '4.0', '2.0']);
    l.init(ctx());
    expect((l.extract('a') as Float64Array)[1 * 2 + 1]).toBe(4);
    expect((l.extract('b') as Float64Array)[1 * 2 + 1]).toBe(2);

    const c = new PairCosineSquared();
    c.allocate(1);
    c.settings(['2.5']);
    c.coeff(['1', '1', '1.0', '1.0'], ctx());
    c.init(ctx());
    expect(c.extract('epsilon')).toBeUndefined();
  });

  it('ufm extract keys include scale and coul/cut/global reports the global cutoff', () => {
    const u = new PairUFM();
    u.allocate(1);
    u.settings(['2.5']);
    u.coeff(['1', '1', '0.5', '1.0']);
    u.init(ctx());
    expect((u.extract('scale') as Float64Array)[3]).toBe(1);
    const cg = new PairCoulCutGlobal();
    cg.allocate(2);
    cg.settings(['3.0']);
    cg.coeff(['*', '*']);
    cg.init(ctx());
    expect(cg.extract('cut_coul')).toBe(3);
    expect(new PairCoulExclude().extract('cut_coul')).toBe(0);
  });
});

describe('write_data coefficient lines (native pair_<style>.cpp write_data / write_data_all)', () => {
  it('lj/mdf and ufm write epsilon sigma (and inner/outer cutoffs in the IJ form)', () => {
    const p = new PairLJMDF();
    p.allocate(2);
    p.settings(['2.0', '2.5']);
    p.coeff(['1', '1', '1.0', '1.0']);
    p.coeff(['2', '2', '0.5', '1.2', '1.9', '2.4']);
    p.init(ctx());
    expect(p.dataCoeffs()).toEqual(['1 1 1', '2 0.5 1.2']);
    // pair 1 2 mixed: epsilon sqrt(1 * 0.5), sigma sqrt(1 * 1.2), cut_inner sqrt(2.0 * 1.9), cut sqrt(2.5 * 2.4)
    expect(p.dataCoeffsIJ()).toEqual(['1 1 1 1 2 2.5', '1 2 0.707107 1.09545 1.94936 2.44949', '2 2 0.5 1.2 1.9 2.4']);
  });

  it('cosine/squared writes the wca word, with a trailing space when the pair has none', () => {
    const c = new PairCosineSquared();
    c.allocate(2);
    c.settings(['2.5']);
    c.coeff(['1', '1', '1.0', '1.0', '2.0'], ctx());
    c.coeff(['2', '2', '1', '1', '1.8', 'wca'], ctx());
    c.coeff(['1', '2', '1', '1', '2.0'], ctx());
    c.init(ctx());
    expect(c.dataCoeffs()).toEqual(['1 1 1 2 ', '2 1 1 1.8 wca']);
  });

  it('coul/exclude writes the type numbers only; wf/cut writes nu and mu as integers', () => {
    const e = new PairCoulExclude();
    e.allocate(2);
    e.settings(['2.5']);
    e.coeff(['*', '*']);
    expect(e.dataCoeffs()).toEqual(['1', '2']);
    expect(e.dataCoeffsIJ()).toEqual(['1 1', '1 2', '2 2']);

    const w = new PairWFCut();
    w.allocate(1);
    w.settings(['2.5']);
    w.coeff(['1', '1', '1.0', '1.0', '1', '2']);
    w.init(ctx());
    expect(w.dataCoeffs()).toEqual(['1 1 1 1 2']);
  });

  it('beck, buck/mdf and lennard/mdf write no coefficient section (native writedata is 0)', () => {
    const b = new PairBeck();
    b.allocate(1);
    expect(b.dataCoeffs()).toBeNull();
    expect(new PairBuckMDF().dataCoeffs()).toBeNull();
    expect(new PairLennardMDF().dataCoeffs()).toBeNull();
  });

  it('coul/slater/cut keeps the coul/cut layout (bare types, then I J cutoff)', () => {
    const s = new PairCoulSlaterCut();
    s.allocate(2);
    s.settings(['0.7', '2.5']);
    s.coeff(['*', '*']);
    s.init(ctx());
    expect(s.dataCoeffs()).toEqual(['1', '2']);
    expect(s.dataCoeffsIJ()).toEqual(['1 1 2.5', '1 2 2.5', '2 2 2.5']);
  });
});

describe('a coefficient set once is not overwritten by a later pair_coeff of another range', () => {
  it('PairParams keeps the explicit flags of each I <= J pair', () => {
    const p = new PairParams(2, ['a']);
    p.set(1, 1, [1]);
    expect(p.isSet(1, 1)).toBe(true);
    expect(p.isSet(1, 2)).toBe(false);
  });
});
