import { describe, expect, it } from 'vitest';
import { meamPe, oracleFiles, relErr, runMeam, worstForceError, type MeamSetup } from './meamHarness';

/*
 * pair_style meam, meam37 scope: the B2 (CsCl, lattce(I,J) = b2) and L12 (Cu3Au, lattce(I,J) = l12) reference pairs
 * with nn2 = 0, through the engine. The entries are the synthetic w37meam_* files of tests/oracle (A = fcc, B = fcc,
 * Ec(1,2) = 3.8, re(1,2) = 2.6, alpha(1,2) = 4.5); the crystal parity over 30 nve steps is in tests/engineOracle*.test.ts
 * (w37meam_b2_crystal, w37meam_l12_crystal). Measured with native LAMMPS (black box): A-B dimer pe (eV), A (type 1) at
 * the origin and B (type 2) at r along x, for both lattices.
 */

const LIB = oracleFiles('w37meam_lib.meam');
const LATTICE = { b2: oracleFiles('w37meam_b2.meam'), l12: oracleFiles('w37meam_l12.meam') };

const dimer = (lat: 'b2' | 'l12', r: number, types = [1, 2]): MeamSetup => ({
  files: { 'lib.meam': LIB['w37meam_lib.meam'], 'par.meam': LATTICE[lat][`w37meam_${lat}.meam`] },
  coeff: 'lib.meam A B par.meam A B',
  types,
  x: [0, 0, 0, r, 0, 0],
});

// Measured with native LAMMPS (black box): A-B dimer pe in eV at the given separation.
const NATIVE: Record<'b2' | 'l12', Array<[number, number]>> = {
  b2: [
    [2.2, -3.17318584742675],
    [2.4, -3.18355780887972],
    [2.6, -3.01714514983461],
    [2.8, -2.77210888910646],
    [3.0, -2.50437984484646],
    [3.3, -2.12139726506609],
  ],
  l12: [
    [2.2, -3.30605723516086],
    [2.4, -3.21007025722725],
    [2.6, -3.01417184001872],
    [2.8, -2.77244316678356],
    [3.0, -2.51862414289175],
    [3.3, -2.15581526217788],
  ],
};

describe('pair_style meam: lattce(I,J) = b2 and l12 (nn2 = 0)', () => {
  for (const lat of ['b2', 'l12'] as const) {
    for (const [r, pe] of NATIVE[lat]) {
      it(`${lat} A-B dimer at ${r} A matches native pe to 1e-10 relative`, async () => {
        expect(relErr(await meamPe(dimer(lat, r)), pe)).toBeLessThan(1e-10);
      });
    }
    it(`${lat} dimer energy does not depend on which element sits at the origin`, async () => {
      for (const r of [2.2, 2.6, 3.3]) {
        expect(await meamPe(dimer(lat, r, [2, 1]))).toBe(await meamPe(dimer(lat, r, [1, 2])));
      }
    });
    it(`${lat} forces equal -grad E (the pair derivative is the derivative of the pair energy)`, async () => {
      for (const r of [2.2, 2.6, 3.0, 3.5]) {
        expect(await worstForceError(dimer(lat, r), 1e-6, 1)).toBeLessThan(1e-6);
      }
    });
  }
});

describe('pair_style meam: b2 and l12 parser', () => {
  const PAR_BASE = 'Ec(1,1) = 3.5\nre(1,1) = 2.5\nalpha(1,1) = 5.0\nEc(2,2) = 4.0\nre(2,2) = 2.7\nalpha(2,2) = 4.0\nzbl(1,1) = 0\nzbl(2,2) = 0\nzbl(1,2) = 0\n';
  const PAR = (lat: string) => `rc = 4\ndelr = 0.1\nEc(1,2) = 3.8\nre(1,2) = 2.6\nalpha(1,2) = 4.5\nlattce(1,2) = ${lat}\n`;
  const withLattice = (par: string) => ({
    files: { 'lib.meam': LIB['w37meam_lib.meam'], 'par.meam': PAR_BASE + par },
    coeff: 'lib.meam A B par.meam A B',
    types: [1, 2],
    x: [0, 0, 0, 2.6, 0, 0],
  });

  it('lattce(1,2) = b2 and l12 are accepted for an I-J pair', async () => {
    for (const lat of ['b2', 'l12']) {
      expect(Number.isFinite(await meamPe(withLattice(PAR(lat))))).toBe(true);
    }
  });

  it('an I-J lattice the engine port does not support is refused, naming it', async () => {
    await expect(runMeam(withLattice(PAR('lin')))).rejects.toThrow(/lattce\(1,2\) = lin is not supported/);
  });
});
