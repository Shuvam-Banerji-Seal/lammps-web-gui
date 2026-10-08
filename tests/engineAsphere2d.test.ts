import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * 2d aspherical particles: compute temp/asphere (2d degrees of freedom), fix
 * enforce2d (zeroing of the x and y angular momentum of ellipsoids), and the
 * refusals. Native parity for the 2d dynamics is in tests/oracle/w23asph2d_*.in.
 * The checks here are the degree-of-freedom counts and the relations that
 * follow from them, measured with native LAMMPS (black box) on the same
 * 9-particle layout.
 */

/** Nine unit spheres on a 3 x 3 square lattice in a 2d box (z periodic, as 2d requires). */
const SQUARES = `units lj
dimension 2
atom_style ellipsoid
lattice sq 0.3
region box block 0 3 0 3 -0.5 0.5
create_box 1 box
create_atoms 1 box
set group all mass 1.0
set group all shape 1 1 1
set group all quat 0 0 1 30
pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0
`;

const runThermo = async (script: string): Promise<{ rows: ThermoRow[]; errors: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  try {
    await session.execute(script);
  } catch {
    // the error is also emitted as an event
  }
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const errors = events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error').map((e) => e.message);
  return { rows, errors };
};

const N = 9;

describe('compute temp/asphere in 2d', () => {
  it('is accepted for a 2d system (no refusal)', async () => {
    const { rows, errors } = await runThermo(`${SQUARES}velocity all create 1.0 4928 dist gaussian
compute tmp all temp/asphere
thermo_style custom step c_tmp
thermo_modify norm no
run 0
`);
    expect(errors).toEqual([]);
    expect(rows.length).toBe(1);
  });

  it('counts 3N - 2 degrees of freedom for dof all and N for dof rotate (measured with native LAMMPS, black box)', async () => {
    // Zero angular momentum: the rotational part is 0, so c_tmp = (sum m v^2) / dof with sum m v^2 = (2N - 2) at T = 1.
    const { rows, errors } = await runThermo(`${SQUARES}velocity all create 1.0 4928 dist gaussian
set group all angmom 0 0 0
compute tmp all temp/asphere
thermo_style custom step c_tmp
thermo_modify norm no
run 0
`);
    expect(errors).toEqual([]);
    expect(rows[0].c_tmp).toBeCloseTo((2 * N - 2) / (3 * N - 2), 12);
  });

  it('dof rotate is N in 2d: c_trot * N = sum L.w = 2 c_erot', async () => {
    const { rows, errors } = await runThermo(`${SQUARES}velocity all create 1.0 4928 dist gaussian
set group all angmom 0 0 0.3
compute trot all temp/asphere dof rotate
compute erot all erotate/asphere
thermo_style custom step c_trot c_erot
thermo_modify norm no
run 0
`);
    expect(errors).toEqual([]);
    const r = rows[0];
    expect(r['c_trot'] * N).toBeCloseTo(2 * r['c_erot'], 10);
    expect(r['c_erot']).toBeGreaterThan(0);
  });
});

describe('fix enforce2d and aspherical angular momentum in 2d', () => {
  it('zeroes the x and y angular momentum of ellipsoids and keeps z (measured with native LAMMPS, black box)', async () => {
    const script = (fix: boolean) => `${SQUARES}velocity all create 1.0 4928 dist gaussian
set group all angmom 0.2 0.3 0.4
compute p all property/atom angmomx angmomy angmomz
compute r all reduce sum c_p[1] c_p[2] c_p[3]
thermo_style custom step c_r[1] c_r[2] c_r[3]
thermo_modify norm no
${fix ? 'fix 2 all enforce2d\n' : ''}run 0
`;
    const withFix = await runThermo(script(true));
    expect(withFix.errors).toEqual([]);
    expect(withFix.rows[0]['c_r[1]']).toBeCloseTo(0, 12);
    expect(withFix.rows[0]['c_r[2]']).toBeCloseTo(0, 12);
    expect(withFix.rows[0]['c_r[3]']).toBeCloseTo(0.4 * N, 10);

    const noFix = await runThermo(script(false));
    expect(noFix.rows[0]['c_r[1]']).toBeCloseTo(0.2 * N, 10);
    expect(noFix.rows[0]['c_r[2]']).toBeCloseTo(0.3 * N, 10);
  });

  it('still refuses enforce2d in a 3d system', async () => {
    const { errors } = await runThermo(`units lj
atom_style ellipsoid
lattice sc 0.3
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
fix 2 all enforce2d
run 0
`);
    expect(errors.join(' ')).toMatch(/enforce2d/);
  });
});
