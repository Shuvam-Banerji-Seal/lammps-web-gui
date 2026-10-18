import { describe, expect, it } from 'vitest';
import { meamPe, oracleFiles, relErr, runMeam, worstForceError, cellAtoms, perturbed, type MeamSetup } from './meamHarness';

/*
 * pair_style meam with non-fcc reference structures (bcc, dia) for one element and multi-element B1 alloys
 * (meam15 scope), through the engine. The entries are the synthetic files of tests/oracle (w15meam_bcc_*,
 * w15meam_dia_*, w15meam_alloy_*). Native reference values were measured with LAMMPS as a black box on the same
 * entries; they are numbers, not LAMMPS text. The one-element bcc crystal is 2x2x2 cells (16 atoms).
 */

const BCC_BASIS = [[0, 0, 0], [0.5, 0.5, 0.5]];
const DIA_BASIS = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]];

const BCC = oracleFiles('w15meam_bcc_lib.meam', 'w15meam_bcc.meam');
const BCC_CMD = 'w15meam_bcc_lib.meam W w15meam_bcc.meam W';
const DIA = oracleFiles('w15meam_dia_lib.meam', 'w15meam_dia.meam');
const DIA_CMD = 'w15meam_dia_lib.meam W w15meam_dia.meam W';
const ALLOY = oracleFiles('w15meam_alloy_lib.meam', 'w15meam_alloy_b1.meam');
const ALLOY_CMD = 'w15meam_alloy_lib.meam A B w15meam_alloy_b1.meam A B';

/** Setup of a one-element structure from flat coordinates, with a periodic cubic box of side L (no offset). */
const single = (files: Record<string, string>, coeff: string, x: number[], L: number): MeamSetup => ({
  files,
  coeff,
  types: Array.from({ length: x.length / 3 }, () => 1),
  x,
  box: L,
  offset: 0,
});
const bccSingle = (x: number[], L: number) => single(BCC, BCC_CMD, x, L);
const diaSingle = (x: number[], L: number) => single(DIA, DIA_CMD, x, L);

/** Two-element setup (types 1 = A, 2 = B) of flat coordinates; box and offset as in the single-element setups. */
const alloy = (types: number[], x: number[], box = 60, offset = 30, files: Record<string, string> = ALLOY, coeff = ALLOY_CMD): MeamSetup => ({
  files,
  coeff,
  types,
  x,
  box,
  offset,
  ntypes: 2,
});

/** Dimer of two atoms along x. */
const dimerOf = (r: number): number[] => [0, 0, 0, r, 0, 0];

describe('pair_style meam: bcc reference (single element)', () => {
  const A_BCC = 3.16;
  it('perfect crystal energy per atom equals -Ec at the nearest-neighbour distance re', async () => {
    const a = (2 * 2.73664) / Math.sqrt(3);
    const { pe } = await runMeam(bccSingle(cellAtoms(BCC_BASIS, a, 2), 2 * a));
    expect(relErr(pe / 16, -8.9)).toBeLessThan(1e-9);
  });

  it('perfect crystals match native pe at three lattice constants (1e-8 relative)', async () => {
    // Measured with native LAMMPS (black box): 2x2x2 bcc cells at lattice constants 2.8868, 3.16 and 3.3466 A.
    const native: Array<[number, number]> = [
      [2.8868, -124.560876810954],
      [3.16, -142.400000000091],
      [3.3466, -137.289643831942],
    ];
    for (const [a, pe] of native) {
      expect(relErr(await meamPe(bccSingle(cellAtoms(BCC_BASIS, a, 2), 2 * a)), pe)).toBeLessThan(1e-8);
    }
  });

  it('dimers match native pe (angular terms included), 1e-8 relative', async () => {
    const native: Array<[number, number]> = [
      [2.2, -7.92102130999959],
      [2.5, -8.89713603446008],
      [2.7, -8.61185420158656],
      [2.9, -8.00882116592403],
      [3.2, -6.91191704779047],
    ];
    for (const [r, pe] of native) {
      expect(relErr(await meamPe(bccSingle(dimerOf(r), 40)), pe)).toBeLessThan(1e-8);
    }
  });

  it('forces equal -grad E to 1e-6 relative for a perturbed bcc cell', async () => {
    const x = perturbed(cellAtoms(BCC_BASIS, A_BCC, 2), 0.08, 7);
    expect(await worstForceError(bccSingle(x, 2 * A_BCC), 1e-5, 1e-2)).toBeLessThan(1e-6);
  }, 120000);

  it('forces for a bcc dimer and trimer equal -grad E', async () => {
    expect(await worstForceError(bccSingle([0, 0, 0, 2.6, 0.2, 0], 40), 1e-5, 1e-2)).toBeLessThan(1e-6);
    expect(await worstForceError(bccSingle([0, 0, 0, 2.5, 0, 0, 1.4, 2.2, 0.3], 40), 1e-5, 1e-2)).toBeLessThan(1e-6);
  });
});

describe('pair_style meam: diamond reference (single element)', () => {
  it('perfect diamond crystals match native pe, 1e-8 relative', async () => {
    // Measured with native LAMMPS (black box): 2x2x2 diamond cells (64 atoms) at lattice constants 5.0 and 5.3 A.
    const native: Array<[number, number]> = [
      [5.0, -294.400000000636],
      [5.3, -283.526028000241],
    ];
    for (const [a, pe] of native) {
      expect(relErr(await meamPe(diaSingle(cellAtoms(DIA_BASIS, a, 2), 2 * a)), pe)).toBeLessThan(1e-8);
    }
  });

  it('diamond-reference dimers and non-periodic clusters match native pe to 1e-10 relative', async () => {
    // Measured with native LAMMPS (black box, lib W dia with the same synthetic parameters):
    // dimers at 2.2, 2.4, 2.6 A: -5.56337126219287, -5.09095676003276, -4.50106696309083 eV;
    // tetrahedral cluster (centre + 4 neighbours at re): -15.8322440471797 eV; the same with a second
    // neighbour at (2.5, 2.5, 0) A: -19.3138040281089 eV; with that atom displaced by (0.04, 0.02, -0.02) A
    // (position (2.54, 2.52, -0.02)): -19.2618591247003 eV.
    const dimers: Array<[number, number]> = [[2.2, -5.56337126219287], [2.4, -5.09095676003276], [2.6, -4.50106696309083]];
    for (const [r, pe] of dimers) {
      expect(relErr(await meamPe(diaSingle(dimerOf(r), 40)), pe)).toBeLessThan(1e-10);
    }
    const t = 2.165064 / Math.sqrt(3);
    const tetra = [0, 0, 0, t, t, t, t, -t, -t, -t, t, -t, -t, -t, t];
    expect(relErr(await meamPe(diaSingle(tetra, 60)), -15.8322440471797)).toBeLessThan(1e-10);
    expect(relErr(await meamPe(diaSingle([...tetra, 2.5, 2.5, 0], 60)), -19.3138040281089)).toBeLessThan(1e-10);
    expect(relErr(await meamPe(diaSingle([...tetra, 2.54, 2.52, -0.02], 60)), -19.2618591247003)).toBeLessThan(1e-10);
  });

  it('forces equal -grad E to 1e-6 relative for a perturbed diamond cell', async () => {
    // one 8-atom cell with periodic images (ceil(rc/L) = 1 image shell)
    const x = perturbed(cellAtoms(DIA_BASIS, 5.0, 1), 0.06, 13);
    expect(await worstForceError(diaSingle(x, 5.0), 1e-5, 1e-2)).toBeLessThan(1e-6);
  }, 120000);
});

describe('pair_style meam: reference lattices in the parameter and library files', () => {
  const LIB_BCC = "'W' 'bcc' 8 74 183.84 5.0 2.0 1.5 2.5 4.0 3.16 8.9 1.0 1.0 2.0 1.0 1.0 1.0 0\n";
  const PAR_BCC = 'rc = 4.0\ndelr = 0.1\nEc(1,1) = 8.9\nre(1,1) = 2.73664\nalpha(1,1) = 5.0\nlattce(1,1) = bcc\nzbl(1,1) = 0\n';
  const bccWith = (lib: string, par: string) => ({ files: { 'lib.meam': lib, 'par.meam': par }, coeff: 'lib.meam W par.meam W', types: [1, 1], x: dimerOf(2.6), box: 40, offset: 0 });

  it('a bcc library entry with a matching parameter file runs', async () => {
    expect(Number.isFinite(await meamPe(bccWith(LIB_BCC, PAR_BCC)))).toBe(true);
  });

  it('a parameter-file lattce(1,1) overrides the library lattice (native does not compare them)', async () => {
    // Measured with native LAMMPS (black box): bcc library entry, lattce(1,1) = fcc in the parameter file, dimer at 2.6 A: -7.30926725240879 eV.
    const par = PAR_BCC.replace('bcc', 'fcc');
    expect(relErr(await meamPe(bccWith(LIB_BCC, par)), -7.30926725240879)).toBeLessThan(1e-9);
  });

  it('lattce(1,1) = b1 on a single element runs (native accepts the name in the user file)', async () => {
    // Measured with native LAMMPS (black box): bcc entry with lattce(1,1) = b1, dimer at 2.7 A: -9.58948699873987 eV.
    const par = PAR_BCC.replace('bcc', 'b1');
    expect(relErr(await meamPe({ ...bccWith(LIB_BCC, par), x: dimerOf(2.7) }), -9.58948699873987)).toBeLessThan(1e-9);
  });

  it('a library entry whose z does not match its lattice is an error naming the mismatch', async () => {
    const lib = LIB_BCC.replace("'bcc' 8", "'bcc' 12");
    await expect(runMeam(bccWith(lib, PAR_BCC))).rejects.toThrow(/Mismatched parameter in MEAM library file: z != lat/);
  });

  it('a lattice name the engine port does not implement for an element is an error naming it', async () => {
    await expect(runMeam(bccWith(LIB_BCC, PAR_BCC.replace('bcc', 'lin')))).rejects.toThrow(/lin/);
  });
});

describe('pair_style meam: multi-element B1 alloy (fcc elements)', () => {
  it('dimers (A-B, A-A, B-B) match native pe to 1e-10 relative', async () => {
    const cases: Array<[number[], number, number]> = [
      [[1, 2], 2.2, -3.09974618373519],
      [[1, 2], 2.6, -3.15152249979223],
      [[1, 2], 3.0, -2.64815172340702],
      [[1, 1], 2.2, -2.96854645415178],
      [[1, 1], 3.0, -2.08217015898284],
      [[2, 2], 2.2, -3.82225743844039],
      [[2, 2], 3.0, -2.74459781879424],
    ];
    for (const [types, r, pe] of cases) {
      expect(relErr(await meamPe(alloy(types, dimerOf(r), 40, 0)), pe)).toBeLessThan(1e-10);
    }
  });

  it('A-B dimers inside the smoothing window [3.9, 4.0] A match native pe to 1e-10 relative', async () => {
    // Measured with native LAMMPS (black box, w15meam_alloy_* entries): A-B dimers at 3.92, 3.95, 3.97, 3.99 A.
    const cases: Array<[number, number]> = [
      [3.92, -1.56249847692229],
      [3.95, -1.41962888480406],
      [3.97, -1.06207996048462],
      [3.99, -0.321820764013811],
    ];
    for (const [r, pe] of cases) {
      expect(relErr(await meamPe(alloy([1, 2], dimerOf(r), 40, 0)), pe)).toBeLessThan(1e-10);
    }
  });

  it('mixed clusters (t-averaging weighted by the partial densities) match native to 1e-9 relative', async () => {
    const cases: Array<[number[], number[], number]> = [
      [[1, 2, 1], [0, 0, 0, 2.4, 0, 0, 1.2, 2.0, 0.3], -6.23612331013911],
      [[2, 1, 1, 2], [0, 0, 0, 2.5, 0.2, 0, 1.3, 2.2, 0.4, 1.1, 0.7, 2.5], -9.05192035780966],
      [[1, 1, 2, 2], [0, 0, 0, 2.6, 0, 0, 1.3, 2.3, 0, 1.0, 0.8, 2.8], -8.28828893530625],
    ];
    for (const [types, x, pe] of cases) {
      expect(relErr(await meamPe(alloy(types, x)), pe)).toBeLessThan(1e-9);
    }
  });

  it('B1 crystal (8 atoms, a = 5.2 A = 2 re(1,2)) has pe -30.400000000089 eV (periodic)', async () => {
    const { x, types } = b1Cell(5.2);
    expect(relErr(await meamPe(alloy(types, x, 5.2, 0)), -30.400000000089)).toBeLessThan(1e-9);
  });

  it('forces equal -grad E to 1e-6 relative (mixed cluster and perturbed B1 cell)', async () => {
    expect(await worstForceError(alloy([1, 2, 1, 2], [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2]), 1e-5, 1e-2)).toBeLessThan(1e-6);
    const cell = b1Cell(5.2);
    const x = perturbed(cell.x, 0.05, 4);
    expect(await worstForceError(alloy(cell.types, x, 5.2, 0), 1e-5, 1e-2)).toBeLessThan(1e-6);
  }, 120000);
});

/** B1 cell: A (type 1) on fcc sites, B (type 2) on the octahedral sites, 8 atoms in a cubic cell of side a. */
const b1Cell = (a: number): { x: number[]; types: number[] } => {
  const fcc = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]];
  const oct = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]];
  return {
    x: [...fcc.flat().map((c) => c * a), ...oct.flat().map((c) => c * a)],
    types: [...fcc.map(() => 1), ...oct.map(() => 2)],
  };
};

describe('pair_style meam: pair_coeff and parameter-file handling (multi-element)', () => {
  const LIB2 = ALLOY['w15meam_alloy_lib.meam'];
  const PAR2 = ALLOY['w15meam_alloy_b1.meam'];
  /** A-B dimer at 2.6 A with the given parameter file, pair_coeff arguments and library. */
  const dimerSetup = (par: string, args = 'lib.meam A B par.meam A B', lib = LIB2): MeamSetup => ({
    files: { 'lib.meam': lib, 'par.meam': par },
    coeff: args,
    types: [1, 2],
    x: dimerOf(2.6),
    box: 60,
    offset: 30,
    ntypes: 2,
  });

  it('a two-element B1 alloy with fcc elements is accepted, and a third type may map to an element again', async () => {
    expect(Number.isFinite(await meamPe(dimerSetup(PAR2)))).toBe(true);
    const three: MeamSetup = { ...dimerSetup(PAR2, 'lib.meam A B par.meam A B A'), types: [1, 2, 3], x: [0, 0, 0, 2.6, 0, 0, 1.3, 2.0, 0.3], ntypes: 3 };
    expect(Number.isFinite(await meamPe(three))).toBe(true);
  });

  it('a lattce(1,2) the engine port does not implement is an error naming it', async () => {
    await expect(runMeam(dimerSetup(PAR2.replace('b1', 'lin')))).rejects.toThrow(/lattce\(1,2\) = lin is not supported/);
  });

  it('a missing pair parameter is not an error (native uses the defaults): the dimer runs and matches native', async () => {
    // Measured with native LAMMPS (black box): the same dimer at 2.6 A without Ec(1,2): -3.13485583312551 eV.
    expect(relErr(await meamPe(dimerSetup(PAR2.replace('Ec(1,2) = 3.8\n', ''))), -3.13485583312551)).toBeLessThan(1e-9);
  });

  it('an element that the library does not contain is an error naming it', async () => {
    await expect(runMeam(dimerSetup(PAR2, 'lib.meam A C par.meam A C'))).rejects.toThrow(/missing: C/);
  });

  it('delta(1,2) is read as a native keyword and the dimer runs', async () => {
    expect(Number.isFinite(await meamPe(dimerSetup(PAR2 + 'delta(1,2) = 0.1\n')))).toBe(true);
  });

  it('per-triplet screening Cmax(1,1,2) and nn2(1,2) = 1 (b1) are accepted', async () => {
    expect(Number.isFinite(await meamPe(dimerSetup(PAR2 + 'Cmax(1,1,2) = 3.0\n')))).toBe(true);
    expect(Number.isFinite(await meamPe(dimerSetup(PAR2 + 'nn2(1,2) = 1\n')))).toBe(true);
  });

  it('a pair keyword with I > J is read as native does: the entry is not used (Ec(2,1) changes nothing)', async () => {
    // Measured with native LAMMPS (black box): the A-B dimer at 2.6 A is -3.15152249979223 eV with or without Ec(2,1) = 3.0.
    expect(relErr(await meamPe(dimerSetup(PAR2 + 'Ec(2,1) = 3.0\n')), -3.15152249979223)).toBeLessThan(1e-10);
  });

  it('an sc library entry with z = 12 is a mismatch error (z != lat), as native', async () => {
    const lib = LIB2.replace("'A' 'fcc'", "'A' 'sc'");
    await expect(runMeam(dimerSetup(PAR2, 'lib.meam A B par.meam A B', lib))).rejects.toThrow(/Mismatched parameter in MEAM library file: z != lat/);
  });

  it('a pair index outside a one-element potential is an error (lattce(1,2) with one element)', async () => {
    const one = { files: { 'lib.meam': BCC['w15meam_bcc_lib.meam'], 'par.meam': PAR2 + 'lattce(1,2) = b1\n' }, coeff: 'lib.meam W par.meam W', types: [1, 1], x: dimerOf(2.6), box: 40, offset: 0 };
    await expect(runMeam(one)).rejects.toThrow(/out of range element index/);
  });
});
