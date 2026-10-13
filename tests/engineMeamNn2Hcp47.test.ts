import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { DEFAULT_MEAM_OPTIONS, nn2RefData } from '../src/engine/force/pair/meam';
import { referenceVectors, PAIR_SHELL_RATIO } from '../src/engine/force/pair/meam_lattice';

/*
 * 2NN (nn2 = 1) MEAM for an hcp reference (meam47 scope). Reference geometry: the ideal hcp site has 12 first
 * neighbours at a and 6 second neighbours at sqrt2 a (the 2NN shell used by the recurrence; c/a ideal).
 * docs.lammps.org/pair_meam.html: "nn2(I,J)    = turn on second-nearest neighbor MEAM formulation for". Native
 * values (black box) are in tests/oracle/w47mn_*.in (engineOracle.test.ts): fully screened Ti hcp (Cmin = 1, z2 = 0)
 * dimers and a 108-atom crystal. The partly screened hcp second shell (z2 > 0) is not modelled by the core: the core
 * hands it to the native port (w136meam_hcp2nn, oracle parity).
 */

const LIB = readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', 'FeTiClibrary.meam'), 'utf8');
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

/** Runs a one-dimer input with the Ti params of the given Cmin; resolves to the thermo rows or rejects. */
const runTi = async (cmin: number, r: number): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  session.addFile('FeTiClibrary.meam', LIB);
  session.addFile('ti.meam', TI_PARAMS(cmin));
  await session.execute(`units metal
atom_style atomic
boundary f f f
region box block -40 40 -40 40 -40 40
create_box 1 box
create_atoms 1 single 0 0 0 units box
create_atoms 1 single ${r} 0 0 units box
mass 1 47.88
pair_style meam
pair_coeff * * FeTiClibrary.meam Ti ti.meam Ti
thermo_style custom step pe
thermo_modify format float %.15g
run 0
`);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

/** Fraction of a second-shell screening atom pair: fs(x) = [1-(1-x)^4]^2 with x = (C - Cmin)/(Cmax - Cmin). */
const fs = (x: number) => (1 - (1 - x) ** 4) ** 2;

describe('MEAM nn2 = 1 with the hcp reference', () => {
  it('ideal hcp shells: 12 first neighbours at a and 6 second neighbours at sqrt2 a', () => {
    const v = referenceVectors('hcp', 2.5, 2.5 * PAIR_SHELL_RATIO.hcp * (1 + 1e-6));
    expect(v.length).toBe(18);
    const first = v.filter((p) => Math.abs(p.r - 2.5) < 1e-9);
    const second = v.filter((p) => Math.abs(p.r - 2.5 * Math.SQRT2) < 1e-9);
    expect(first.length).toBe(12);
    expect(second.length).toBe(6);
    expect(PAIR_SHELL_RATIO.hcp).toBeCloseTo(Math.SQRT2, 12);
  });

  it('z1 = 12, a = sqrt2 and z2 = 6 S with S = fs(C)^4 for the four screeners at C = 1 (fully screened: z2 = 0)', () => {
    const o = { ...DEFAULT_MEAM_OPTIONS, rc: 4.8, delr: 0.1, Cmax: 1.44, Cmin: 1 };
    const full = nn2RefData('hcp', o);
    expect(full.a).toBeCloseTo(Math.SQRT2, 12);
    expect(full.z1).toBeCloseTo(12, 12);
    expect(full.z2).toBeLessThan(1e-12);
    const oPart = { ...DEFAULT_MEAM_OPTIONS, rc: 4.8, delr: 0.1, Cmax: 2.8, Cmin: 0.36 };
    const part = nn2RefData('hcp', oPart);
    const S = fs((1 - 0.36) / (2.8 - 0.36)) ** 4;
    expect(part.z2).toBeCloseTo(6 * S, 9);
    expect(part.z2).toBeCloseTo(0.361350189, 6);
  });

  it('a fully screened Ti hcp dimer runs (native value -2.38541715171305 eV at 2.92 A, tests/oracle/w47mn_ti_dimer)', async () => {
    const rows = await runTi(1, 2.92);
    expect(rows.length).toBe(1);
    expect(rows[0].pe).toBeCloseTo(-2.38541715171305, 8);
  });

  it('a partly screened hcp second shell (Cmin = 0.36) is handed to the standard MEAM port (parity: w136meam_hcp2nn)', async () => {
    // the core model refuses this set; the pair runs through the native port of src/engine-gpl/meam (pair_meam.cpp, msmeamflag 0)
    const rows = await runTi(0.36, 2.92);
    expect(rows.length).toBe(1);
    expect(Number.isFinite(rows[0].pe)).toBe(true);
  });
});
