import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { meamPe, relErr, worstForceError, type MeamSetup } from './meamHarness';

/*
 * pair_style meam, nn2 = 1 with an hcp reference (meam47 scope), through the engine. The library is the Fe-Ti library
 * of third_party/lammps/potentials; the parameters are the Ti set of tests/oracle/w47mn_ti.meam with the Cmin of the
 * second-shell screening varied. docs.lammps.org/pair_meam.html: "nn2(I,J) = turn on second-nearest neighbor MEAM
 * formulation for". Native values (black box, dimer at 2.92 A, measured with the same input) are checked here;
 * the crystal parity is in tests/oracle/w47mn_ti_crystal.in (engineOracle*.test.ts).
 */

const LIB = readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', 'FeTiClibrary.meam'), 'utf8');

/** Ti hcp parameter set with nn2 = 1 and the given Cmin of the second-shell screening. */
const TI_PARAMS = (cmin: number) => `rc = 4.8
delr = 0.1
augt1 = 0
erose_form = 2
ialloy = 2
zbl(1,1) = 0
nn2(1,1) = 1
attrac(1,1) = 0
repuls(1,1) = 0
Cmin(1,1,1) = ${cmin}
Cmax(1,1,1) = 1.44
Ec(1,1) = 4.87
re(1,1) = 2.92
`;

/** A Ti dimer at 2.92 A (the first-neighbour distance of the parameters) with the given Cmin. */
const dimer = (cmin: number): MeamSetup => ({
  files: { 'FeTiClibrary.meam': LIB, 'ti.meam': TI_PARAMS(cmin) },
  coeff: 'FeTiClibrary.meam Ti ti.meam Ti',
  types: [1, 1],
  x: [0, 0, 0, 2.92, 0, 0],
  box: 80,
});

describe('pair_style meam: nn2 = 1 with the hcp reference (meam47)', () => {
  it('a fully screened Ti hcp dimer matches native pe to 1e-9 relative (-2.38541715171305 eV, tests/oracle/w47mn_ti_dimer)', async () => {
    expect(relErr(await meamPe(dimer(1)), -2.38541715171305)).toBeLessThan(1e-9);
  });

  it('a partly screened hcp second shell (Cmin = 0.36) matches native pe to 1e-9 relative (-2.22034693937246 eV)', async () => {
    // Measured with native LAMMPS (black box, the same dimer and parameters with Cmin = 0.36): -2.22034693937246 eV.
    expect(relErr(await meamPe(dimer(0.36)), -2.22034693937246)).toBeLessThan(1e-9);
  });

  it('the partly screened hcp dimer: forces equal -grad E to 1e-6 relative', async () => {
    expect(await worstForceError(dimer(0.36), 1e-6, 1)).toBeLessThan(1e-6);
  });
});
