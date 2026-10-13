import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { ctipSelf, parseCtipFile } from '../src/engine-gpl/ctip/pair_coul_ctip';
import { mdfTaper } from '../src/engine/force/pair/coul_streitz';

/* pair coul/ctip (hybrid/overlay with eam/fs) and fix qeq/ctip (src/engine-gpl/ctip, src/engine-gpl/qeq): the parse,
 * the refusals and the taper of the dsf Slater part. Bit-level agreement with native LAMMPS is in the w140ct_* oracle cases. */

const ORACLE = join(__dirname, 'oracle');
const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');

const session = (events: EngineEvent[]): Session => {
  const s = new Session({ emit: (e) => events.push(e) });
  s.addFile('w140ct_small.data', readFileSync(join(ORACLE, 'w140ct_small.data'), 'utf8'));
  s.addFile('NiO.ctip', readFileSync(join(POT, 'NiO.ctip'), 'utf8'));
  s.addFile('NiO.eam.fs', readFileSync(join(POT, 'NiO.eam.fs'), 'utf8'));
  return s;
};

const head = 'units metal\natom_style charge\nboundary p p p\nread_data w140ct_small.data\n';
const ctipPair = 'pair_style hybrid/overlay eam/fs coul/ctip 0.30 12.0\npair_coeff * * eam/fs NiO.eam.fs Ni O\npair_coeff * * coul/ctip NiO.ctip Ni O\n';

describe('pair coul/ctip and fix qeq/ctip', () => {
  it('coul/ctip parses under hybrid/overlay with eam/fs, and fix qeq/ctip reports its CG count as f_ID', async () => {
    const events: EngineEvent[] = [];
    await session(events).execute(`${head}${ctipPair}fix 1 all qeq/ctip 1 12.0 1.0e-8 100 coul/ctip cdamp 0.30 maxrepeat 10\nthermo_style custom step pe f_1\nthermo 0\nrun 0\n`);
    const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')[0].row;
    expect(Number.isFinite(row.pe as number)).toBe(true);
    expect(Number.isInteger(row.f_1 as number)).toBe(true);
    expect(row.f_1 as number).toBeGreaterThan(0);
  });

  it('refuses a qfile other than coul/ctip by name', async () => {
    await expect(session([]).execute(`${head}${ctipPair}fix 1 all qeq/ctip 1 12.0 1.0e-8 100 NiO.ctip\n`)).rejects.toThrow(/qfile 'NiO.ctip' is not supported/);
  });

  it('refuses an unknown keyword by name', async () => {
    await expect(session([]).execute(`${head}${ctipPair}fix 1 all qeq/ctip 1 12.0 1.0e-8 100 coul/ctip bogus 1\n`)).rejects.toThrow(/Unknown fix qeq\/ctip keyword: bogus/);
  });

  it('needs a pair style coul/ctip', async () => {
    const pair = 'pair_style eam/fs\npair_coeff * * NiO.eam.fs Ni O\n';
    await expect(session([]).execute(`${head}${pair}fix 1 all qeq/ctip 1 12.0 1.0e-8 100 coul/ctip\n`)).rejects.toThrow(/No pair style coul\/ctip/);
  });

  it('the potential-file reader enforces the sanity checks of pair_coul_ctip.cpp', () => {
    expect(() => parseCtipFile('Ni 1.0 -1.0 0.5 0.0 0.0 0.0 1.0 1.0\n', 'bad.ctip')).toThrow(/Illegal coul\/ctip parameter/);
    expect(() => parseCtipFile('Ni 1.0 1.0 0.5 0.0 0.0 2.0 1.0 1.0\n', 'bad.ctip')).toThrow(/Illegal coul\/ctip parameter/);
    const p = parseCtipFile('# comment\nNi -5.84 14.15739 0.54929 0 0 0 2 100\n', 'ok.ctip').get('Ni');
    expect(p?.chi).toBeCloseTo(-5.84, 12);
    expect(p?.qmax).toBe(2);
  });

  it('the self energy is the charge-regime polynomial (native self(): qmin and qmax branches continue)', () => {
    const p = { chi: 1.0, eta: 2.0, gamma: 1, zeta: 0, zcore: 0, qmin: -1, qmax: 1, omega: 0.5 };
    expect(ctipSelf(p, 0)).toBe(0);
    // at qmax from below and from above the value is continuous
    const below = ctipSelf(p, 1 - 1e-9), above = ctipSelf(p, 1 + 1e-9);
    expect(Math.abs(below - above)).toBeLessThan(1e-7);
  });
});

describe('dsf taper of the Slater part (native MathSpecial::mdftaper)', () => {
  it('is 1 inside rc - w, 0 at rc, and its derivative matches a finite difference', () => {
    const out = { f: 0, df: 0 };
    mdfTaper(5, 12, 2, out);
    expect(out.f).toBe(1);
    mdfTaper(12, 12, 2, out);
    expect(out.f).toBe(0);
    const h = 1e-6;
    for (const r of [10.5, 11, 11.6]) {
      mdfTaper(r, 12, 2, out);
      const d = out.df;
      mdfTaper(r + h, 12, 2, out);
      const fp = out.f;
      mdfTaper(r - h, 12, 2, out);
      const fm = out.f;
      expect(d).toBeCloseTo((fp - fm) / (2 * h), 6);
    }
  });

  it('a width of zero is the step at the cutoff', () => {
    const out = { f: 0, df: 0 };
    mdfTaper(11.9, 12, 0, out);
    expect(out.f).toBe(1);
    expect(out.df).toBe(0);
  });
});
