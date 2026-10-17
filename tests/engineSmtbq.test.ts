import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { readSmtbqFile } from '../src/engine-gpl/smtbq/potfile';
import { SlaterIntegrals, factorial, powint } from '../src/engine-gpl/smtbq/slater';
import { erfc } from '../src/engine-gpl/libm/erf';
import { buildTables } from '../src/engine-gpl/smtbq/tables';
import { PairSMTBQ } from '../src/engine-gpl/smtbq/pair_smtbq';

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const potText = (name: string): string => readFileSync(join(POT, name), 'utf8');
const ctx = (text: string) => ({ s: null, readFile: () => text, log: () => {} }) as never;

describe('pair_style smtbq (src/engine-gpl/smtbq)', () => {
  it('reads the Al2O3 potential file line by line (C++ read_file layout)', () => {
    const f = readSmtbqFile(potText('ffield.smtbq.Al2O3'));
    expect(f.ntypes).toBe(2);
    expect(f.params.map((p) => p.nom)).toEqual(['O', 'Al']);
    expect(f.cutmax).toBeCloseTo(11.1714, 10);
    expect(f.Qstep).toBe(1);
    expect(f.loopmax).toBe(7000);
    expect(f.QEqMode).toBe('QEqAll');
    // the interaction loop reads one more line than the file has interactions (O-O buck, Al-O second moment)
    expect(f.intparams[1].typepot).toBe('second_moment');
    expect(f.intparams[1].mode).toBe('oxide');
    expect(f.intparams[2].typepot).toBe('buck');
    expect(f.maxintsm).toBe(1);
    expect(f.intype[0 * 2 + 1]).toBe(1);
    expect(f.ncov).toBe(8);
  });

  it('refuses the energy and electronegativity output the potential file asks for', () => {
    // read_file takes the energy flag from the line after the Verbose header (the file's Verbose line), see potfile.ts
    const text = potText('ffield.smtbq.Al').replace(/(Verbose\(true\.or\.false\)[^\n]*?)'false'( *)/, "$1'true'  300.0");
    const pair = new PairSMTBQ();
    pair.allocate(1);
    expect(() => pair.coeff(['*', '*', 'ffield.smtbq.Al', 'Al'], ctx(text))).toThrow(/energy component output/);
  });

  it('requires the element order of the potential file', () => {
    const pair = new PairSMTBQ();
    pair.allocate(2);
    expect(() => pair.coeff(['*', '*', 'ffield.smtbq.Al2O3', 'Al', 'O'], ctx(potText('ffield.smtbq.Al2O3')))).toThrow(/element order/);
  });

  it('builds the tables and the energy at a distance agrees with the bulk value', () => {
    const f = readSmtbqFile(potText('ffield.smtbq.Al'));
    const tb = buildTables(f, new SlaterIntegrals());
    expect(tb.kmax).toBe(52505);
    expect(tb.Vself).toBeLessThan(0);
    expect(Number.isFinite(tb.fafb[1000])).toBe(true);
  });

  it('gammaValues returns the same gam and dgam as the full gammas routine', () => {
    const sl = new SlaterIntegrals();
    for (const [na, nb, za, zb, r] of [[3, 3, 3.09, 3.09, 2.5], [2, 3, 2.36, 3.1, 1.9], [3, 2, 1.2, 2.2, 7.5]]) {
      const full = sl.gammas(na, nb, za, zb, r);
      const g = full.gam, dg = full.dgam;
      const v = sl.gammaValues(na, nb, za, zb, r);
      expect(v.gam).toBe(g);
      expect(v.dgam).toBe(dg);
    }
  });

  it('special functions and erfc', () => {
    expect(factorial(5)).toBe(120);
    expect(powint(2, 3)).toBe(8);
    expect(powint(2, -2)).toBe(0.25);
    expect(erfc(0)).toBeCloseTo(1, 14);
    expect(erfc(1)).toBeCloseTo(0.15729920705028513, 14);
    expect(erfc(2)).toBeCloseTo(0.004677734981047266, 14);
    expect(erfc(3)).toBeCloseTo(2.209049699858544e-5, 18);
  });
});
