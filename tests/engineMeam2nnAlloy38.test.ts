import { describe, expect, it } from 'vitest';
import { meamPe, oracleFiles, perturbed, relErr, runMeam, cellAtoms, type MeamSetup } from './meamHarness';

/*
 * pair_style meam with nn2 = 1 in a two-element alloy (meam38 scope), through the engine. The entries are the
 * synthetic w38m2nn files of tests/oracle (library w38m2nn_lib.meam; parameter files w38m2nn_b2.meam and
 * w38m2nn_b1.meam with nn2 = 1 on every pair, Cmin(1,1,1) = Cmin(2,2,2) = 0.7). docs.lammps.org/pair_meam.html:
 * "nn2(I,J) = turn on second-nearest neighbor MEAM formulation for". For a B2 (CsCl) reference crystal the energy per
 * atom must equal -Ec(1,2) at its nearest-neighbour distance re(1,2) (the Rose minimum of the I-J pair). The
 * crystal parity with native LAMMPS is in tests/oracle/w38m2nn_*.in (engineOracle*.test.ts).
 */

const LIB = oracleFiles('w38m2nn_lib.meam');
const B2_PAR = oracleFiles('w38m2nn_b2.meam')['w38m2nn_b2.meam'];
const B2_CMD = 'w38m2nn_lib.meam A B w38m2nn_b2.meam A B';

/** B2 (CsCl) crystal: n x n x n simple-cubic cells of side a, A on the corners and B at the body centres. */
const b2Crystal = (re: number, n: number, par = B2_PAR): MeamSetup => {
  const a = (2 * re) / Math.sqrt(3);
  const x = cellAtoms([[0, 0, 0], [0.5, 0.5, 0.5]], a, n);
  return {
    files: { 'lib.meam': LIB['w38m2nn_lib.meam'], 'par.meam': par },
    coeff: 'lib.meam A B par.meam A B',
    types: Array.from({ length: x.length / 3 }, (_, i) => (i % 2 === 0 ? 1 : 2)),
    x,
    box: n * a,
    offset: 0,
    ntypes: 2,
  };
};

describe('pair_style meam: nn2 = 1 reference crystals (B2)', () => {
  it('B2 reference crystal: energy per atom equals -Ec(1,2) at re(1,2) = 2.6 A (3x3x3 cells, 54 atoms)', async () => {
    const { pe } = await runMeam(b2Crystal(2.6, 3));
    expect(relErr(pe / 54, -3.8)).toBeLessThan(1e-9);
  }, 120000);

  it('nn2 = 1 changes the energy of a displaced B2 crystal relative to nn2 = 0 (the second shell contributes)', async () => {
    // Measured with native LAMMPS (black box, the displaced 54-atom crystal of tests/oracle/w38m2nn_b2_crystal.in, run 0):
    // -204.756630227768 eV with nn2 and -204.756634725853 eV without, so the second shell matters off the symmetric lattice.
    const noNn2 = B2_PAR.replace(/^nn2\(\d,\d\) = 1\n/gm, '');
    const displaced = (par: string): MeamSetup => {
      const s = b2Crystal(2.6, 2, par);
      return { ...s, x: perturbed(s.x, 0.04, 9) };
    };
    const e2 = await meamPe(displaced(B2_PAR));
    const e0 = await meamPe(displaced(noNn2));
    expect(Math.abs(e2 - e0)).toBeGreaterThan(1e-7);
  }, 120000);

  it('nn2 = 1 with the l12 reference runs (the native port takes the second shell of the L12 pair)', async () => {
    const l12 = B2_PAR.replace('lattce(1,2) = b2', 'lattce(1,2) = l12');
    const dimer: MeamSetup = { files: { 'lib.meam': LIB['w38m2nn_lib.meam'], 'par.meam': l12 }, coeff: 'lib.meam A B par.meam A B', types: [1, 2], x: [0, 0, 0, 2.6, 0, 0], box: 60, offset: 30, ntypes: 2 };
    expect(Number.isFinite(await meamPe(dimer))).toBe(true);
  });
});

describe('pair_style meam: nn2 = 1 parameter handling', () => {
  it('a B2 pair with nn2 = 1 is accepted (the dimer runs)', async () => {
    const dimer: MeamSetup = { files: { 'lib.meam': LIB['w38m2nn_lib.meam'], 'par.meam': B2_PAR }, coeff: 'lib.meam A B par.meam A B', types: [1, 2], x: [0, 0, 0, 2.6, 0, 0], box: 60, offset: 30, ntypes: 2 };
    expect(Number.isFinite(await meamPe(dimer))).toBe(true);
  });

  it('an element index outside the potential is an error (nn2(3,3) in a two-element file)', async () => {
    const dimer: MeamSetup = { files: { 'lib.meam': LIB['w38m2nn_lib.meam'], 'par.meam': B2_PAR + 'nn2(3,3) = 1\n' }, coeff: 'lib.meam A B par.meam A B', types: [1, 2], x: [0, 0, 0, 2.6, 0, 0], box: 60, offset: 30, ntypes: 2 };
    await expect(runMeam(dimer)).rejects.toThrow(/out of range element index/);
  });
});
