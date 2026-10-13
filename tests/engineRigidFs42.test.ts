import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix rigid / rigid/small with finite-size particles (src/engine/fix/rigid.ts): spheres are integrated as
 * finite-size bodies in 2d and 3d; ellipsoids and point dipoles are refused with an error that names them.
 * The 2d sphere bodies are checked against native LAMMPS by the oracle cases w92rig2d_*. The sphere-spin dynamics are checked against native LAMMPS by the oracle case w42rfs_sphere_spin.
 * fix_rigid.rst: "Finite-size particles contribute differently to the moment of inertia of a rigid body than
 * do point particles."
 */

/** Runs the input; returns the thermo rows and the engine error (null when the run succeeds). */
type ErrEvent = Extract<EngineEvent, { kind: 'error' }>;
const run = async (text: string): Promise<{ rows: ThermoRow[]; error: { message: string } | null }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let thrown: unknown = null;
  try { await session.execute(text); } catch (e) { thrown = e; }
  const ev = events.find((e): e is ErrEvent => e.kind === 'error');
  const err = ev ? { message: ev.message } : thrown ? { message: String(thrown) } : null;
  return { rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row), error: err };
};

/** Two spheres of a rigid body with spin 0.3 about z, plus six free spheres at rest (temperature dof stay positive). */
const DIMER = `units lj
atom_style sphere
region box block 0 20 0 20 0 20 units box
create_box 1 box
create_atoms 1 single 8 10 10
create_atoms 1 single 9 10 10
create_atoms 1 single 3 3 3
create_atoms 1 single 4 3 3
create_atoms 1 single 5 3 3
create_atoms 1 single 6 3 3
create_atoms 1 single 3 4 3
create_atoms 1 single 3 5 3
set type 1 diameter 1.0 density 1.0
pair_style zero 2.5
pair_coeff * *
velocity all set 0 0 0 units box
set atom 1 omega 0 0 0.3
set atom 2 omega 0 0 0.3
set atom 1 vx 0.1 vy 0 vz 0
timestep 0.001
group rb id 1:2
fix 1 rb rigid single
thermo_style custom step temp ke
thermo_modify format float %.15g
thermo 1
`;

describe('fix rigid with finite-size particles', () => {
  it('two-sphere body: spin enters the body angular momentum with the sphere inertia 2/5 m r^2', async () => {
    const { rows, error } = await run(`${DIMER}run 2\n`);
    expect(error).toBeNull();
    // Measured with native LAMMPS (black box): omega_z = 0.0857142857142857 = L / I with I including 2/5 m r^2 per sphere
    // (the two sphere spins 0.015708 each add to the orbital angular momentum before the rigid setup)
    // Measured with native LAMMPS (black box), thermo output of the same input (2 steps): temp 0.000302761196774527,
    // ke 0.000283838621976119 per atom (8 atoms; body dof 3*8 - 3 - 6 = 15)
    expect(rows[0].temp).toBeCloseTo(0.000302761196774527, 12);
    expect(rows[0].ke).toBeCloseTo(0.000283838621976119, 12);
  });

  it('refuses ellipsoidal particles in a rigid body, naming them', async () => {
    const { error } = await run(`units lj
atom_style ellipsoid
region box block 0 10 0 10 0 10 units box
create_box 1 box
create_atoms 1 single 5 5 5
create_atoms 1 single 6 5 5
set type 1 diameter 1.0 density 1.0
set atom 1 shape 0.5 0.5 0.5
set atom 2 shape 0.5 0.5 0.5
pair_style zero 2.5
pair_coeff * *
fix 1 all rigid single
run 0
`);
    expect(error?.message ?? '').toMatch(/ellipsoid/);
  });

  it('refuses point dipoles in a rigid body, naming them', async () => {
    const { error } = await run(`units lj
atom_style hybrid sphere dipole
region box block 0 10 0 10 0 10 units box
create_box 1 box
create_atoms 1 single 5 5 5
create_atoms 1 single 6 5 5
set type 1 diameter 1.0 density 1.0
set atom 1 dipole 0 0 1
set atom 2 dipole 0 0 1
pair_style zero 2.5
pair_coeff * *
fix 1 all rigid single
run 0
`);
    expect(error?.message ?? '').toMatch(/dipole/);
  });

  it('accepts finite-size spheres in 2d bodies (oracle cases w92rig2d_*)', async () => {
    const { error } = await run(`dimension 2
units lj
atom_style sphere
region box block 0 10 0 10 -0.5 0.5 units box
create_box 1 box
create_atoms 1 single 5 5 0
create_atoms 1 single 6 5 0
set type 1 diameter 1.0 density 1.0
pair_style zero 2.5
pair_coeff * *
fix 1 all rigid single
run 0
`);
    expect(error).toBeNull();
  });
});
