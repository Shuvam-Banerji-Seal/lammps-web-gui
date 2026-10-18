import { describe, expect, it } from 'vitest';
import { meamPe, oracleFiles, relErr, runMeam, worstForceError, type MeamSetup } from './meamHarness';

/*
 * pair_style meam, multi-element, meam29 scope: erose_form = 2 with per-pair attrac/repuls, and the diamond
 * (zincblende) alloy reference lattce(I,J) = dia, through the engine. The entries are the synthetic files of
 * tests/oracle (w29meama_erose2_*, w29meama_dia_*, w33meamtrip_*). All energies are native LAMMPS values
 * (black box) for the listed geometry; the periodic crystal parity is in tests/engineOracle*.test.ts
 * (w29meama_erose2, w29meama_dia). Types 1 and 2 are elements A and B; a [0,1] pair of the tables is types [1,2].
 */

const EROSE2 = { files: oracleFiles('w29meama_erose2_lib.meam', 'w29meama_erose2.meam'), coeff: 'w29meama_erose2_lib.meam A B w29meama_erose2.meam A B' };
const DIA = { files: oracleFiles('w29meama_dia_lib.meam', 'w29meama_dia.meam'), coeff: 'w29meama_dia_lib.meam A B w29meama_dia.meam A B' };

/** A setup of the given element types and flat coordinates in one of the parameter sets. */
const setup = (base: { files: Record<string, string>; coeff: string }, types: number[], x: number[]): MeamSetup => ({ ...base, types, x, ntypes: 2 });
const dimer = (base: { files: Record<string, string>; coeff: string }, types: number[], r: number) => setup(base, types, [0, 0, 0, r, 0, 0]);

const MIXED_X = [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2];
const MIXED_T = [1, 2, 1, 2];

/** Native pe of A-A (types [1,1]), A-B ([1,2]) and B-B ([2,2]) dimers at each distance, eV. */
const EROSE2_DIMERS: Array<[number[], number, number]> = [
  [[1, 2], 1.8, 7.085753163255], [[1, 2], 2.0, -0.055971213385], [[1, 2], 2.4, -3.210705990202], [[1, 2], 3.0, -2.711249732667], [[1, 2], 3.5, -2.291520174189],
  [[1, 1], 1.8, 3.656328263965], [[1, 1], 2.0, -1.402657276419], [[1, 1], 2.4, -2.846419350895], [[1, 1], 3.0, -2.168008695256], [[1, 1], 3.5, -1.767502316916],
  [[2, 2], 1.8, -0.914612347745], [[2, 2], 2.0, -2.917303688195], [[2, 2], 2.4, -3.533576507627], [[2, 2], 3.0, -2.752103176795], [[2, 2], 3.5, -2.257816382071],
];
const DIA_DIMERS: Array<[number[], number, number]> = [
  [[1, 2], 1.8, -5.058657065583], [[1, 2], 2.0, -5.046022752973], [[1, 2], 2.4, -4.292944624419], [[1, 2], 3.0, -3.058933085371], [[1, 2], 3.5, -2.328643009059],
  [[1, 1], 1.8, -3.411611730081], [[1, 1], 2.0, -4.635928335455], [[1, 1], 2.4, -4.806102367449], [[1, 1], 3.0, -3.564721609794], [[1, 1], 3.5, -2.627769469148],
  [[2, 2], 1.8, -4.383094642469], [[2, 2], 2.0, -3.760535863967], [[2, 2], 2.4, -2.974789726846], [[2, 2], 3.0, -2.370346499956], [[2, 2], 3.5, -2.008359258368],
];

describe('pair_style meam: erose_form = 2 with per-pair attrac/repuls (meam29)', () => {
  for (const [types, r, pe] of EROSE2_DIMERS) {
    it(`dimer ${types[0]}-${types[1]} at ${r} A matches native pe to 1e-9 relative`, async () => {
      expect(relErr(await meamPe(dimer(EROSE2, types, r)), pe)).toBeLessThan(1e-9);
    });
  }

  it('mixed trimer and cluster match native pe to 1e-9 relative', async () => {
    // Measured with native LAMMPS (black box): A-B-A at (0,0,0),(2.4,0,0),(1.2,2.0,0.3) is -6.080252753316 eV;
    // A-B-A-A at (0,0,0),(2.4,0.1,0),(1.2,2.0,0.3),(1.0,-0.2,2.2) is -8.342851416058 eV.
    expect(relErr(await meamPe(setup(EROSE2, [1, 2, 1], [0, 0, 0, 2.4, 0, 0, 1.2, 2.0, 0.3])), -6.080252753316)).toBeLessThan(1e-9);
    expect(relErr(await meamPe(setup(EROSE2, [1, 2, 1, 1], MIXED_X)), -8.342851416058)).toBeLessThan(1e-9);
  });

  it('forces equal -grad E to 1e-6 relative (mixed cluster)', async () => {
    expect(await worstForceError(setup(EROSE2, MIXED_T, MIXED_X), 1e-5, 1e-2)).toBeLessThan(1e-6);
  });

  it('attrac and repuls are read with the default erose_form (every form of native erose() uses them)', async () => {
    const par = EROSE2.files['w29meama_erose2.meam'];
    const form0 = { ...EROSE2, files: { ...EROSE2.files, 'w29meama_erose2.meam': par.replace('erose_form = 2', '') } };
    const noAttrac = { ...form0, files: { ...form0.files, 'w29meama_erose2.meam': form0.files['w29meama_erose2.meam'].replace(/attrac\(\d,\d\) = [0-9.]+\n|repuls\(\d,\d\) = [0-9.]+\n/g, '') } };
    const e0 = await meamPe(setup(form0, MIXED_T, MIXED_X));
    const eNo = await meamPe(setup(noAttrac, MIXED_T, MIXED_X));
    expect(Number.isFinite(e0)).toBe(true);
    expect(e0).not.toBe(eNo);
  });

  it('erose_form = 3 gives the form-0 energies (native erose(): forms other than 1 and 2 take the else branch)', async () => {
    const par = EROSE2.files['w29meama_erose2.meam'];
    const withForm = (f: number) => ({ ...EROSE2, files: { ...EROSE2.files, 'w29meama_erose2.meam': par.replace('erose_form = 2', `erose_form = ${f}`) } });
    expect(await meamPe(setup(withForm(3), MIXED_T, MIXED_X))).toBe(await meamPe(setup(withForm(0), MIXED_T, MIXED_X)));
  });
});

describe('pair_style meam: diamond (zincblende) alloy, lattce(I,J) = dia (meam29)', () => {
  for (const [types, r, pe] of DIA_DIMERS) {
    it(`dimer ${types[0]}-${types[1]} at ${r} A matches native pe to 1e-9 relative`, async () => {
      expect(relErr(await meamPe(dimer(DIA, types, r)), pe)).toBeLessThan(1e-9);
    });
  }

  it('mixed trimer and cluster match native pe to 1e-9 relative', async () => {
    // Measured with native LAMMPS (black box): A-B-A at (0,0,0),(2.4,0,0),(1.2,2.0,0.3) is -7.758690128429 eV;
    // A-B-A-A at (0,0,0),(2.4,0.1,0),(1.2,2.0,0.3),(1.0,-0.2,2.2) is -9.910487748140 eV.
    expect(relErr(await meamPe(setup(DIA, [1, 2, 1], [0, 0, 0, 2.4, 0, 0, 1.2, 2.0, 0.3])), -7.758690128429)).toBeLessThan(1e-9);
    expect(relErr(await meamPe(setup(DIA, [1, 2, 1, 1], MIXED_X)), -9.910487748140)).toBeLessThan(1e-9);
  });

  it('forces equal -grad E to 1e-6 relative (mixed cluster)', async () => {
    expect(await worstForceError(setup(DIA, MIXED_T, MIXED_X), 1e-5, 1e-2)).toBeLessThan(1e-6);
  });
});

describe('pair_style meam: per-triplet Cmin(I,J,K) / Cmax(I,J,K) (w33meamtrip)', () => {
  it('the rock-salt second-neighbour A-A dimer at 3.677 A matches native pe (w33meamtrip_dim)', async () => {
    // Measured with native LAMMPS (black box, tests/oracle/w33meamtrip_dim.in): pe -1.35564625506306 eV.
    const base = { files: oracleFiles('w33meamtrip_b1_lib.meam', 'w33meamtrip_b1.meam'), coeff: 'w33meamtrip_b1_lib.meam A B w33meamtrip_b1.meam A B' };
    expect(relErr(await meamPe(dimer(base, [1, 1], 3.677)), -1.35564625506306)).toBeLessThan(1e-9);
  });
});

describe('pair_style meam: parameter-file errors in a multi-element potential', () => {
  it('erose_form and the screening keywords are read, an unknown erose_form value is not an error (native)', async () => {
    const files = { ...EROSE2.files, 'w29meama_erose2.meam': EROSE2.files['w29meama_erose2.meam'].replace('erose_form = 2', 'erose_form = 7') };
    expect(Number.isFinite(await meamPe(setup({ ...EROSE2, files }, MIXED_T, MIXED_X)))).toBe(true);
  });

  it('an element index outside the potential is an error (Ec(3,3) in a two-element file)', async () => {
    const files = { ...EROSE2.files, 'w29meama_erose2.meam': EROSE2.files['w29meama_erose2.meam'] + 'Ec(3,3) = 1\n' };
    await expect(runMeam(setup({ ...EROSE2, files }, MIXED_T, MIXED_X))).rejects.toThrow(/out of range element index/);
  });
});
