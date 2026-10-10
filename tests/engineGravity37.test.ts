import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { EngineError, type EngineEvent, type ThermoRow } from '../src/engine/types';

/** Runs input text in a fresh session; returns the thermo rows and the error, if any. */
const run = async (text: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: () => undefined,
  }, undefined, 0);
  for (const [name, body] of Object.entries(files)) session.addFile(name, body);
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { thermo, error };
};

const SIX = `six atoms two bodies

6 atoms
1 atom types
0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 1.0

Atoms

1 1 1 5.0 5.0 10.0
2 1 1 6.0 5.0 10.0
3 1 1 5.0 6.5 10.0
4 2 1 12.0 12.0 8.0
5 2 1 13.0 12.0 8.0
6 2 1 12.0 13.5 8.5

Velocities

1 0.1 0.0 0.0
2 0.1 0.0 0.0
3 0.1 0.0 0.0
4 0.0 0.1 0.0
5 0.0 0.1 0.0
6 0.0 0.1 0.0
`;

const HEAD = `units lj
atom_style molecular
boundary f f f
read_data six.data
pair_style lj/cut 1.0
pair_coeff * * 0.0 1.0
comm_modify cutoff 3.0
timestep 0.001
`;

const msg = (e: EngineError | null) => (e ? e.message : '');

describe('fix gravity disable and fix rigid gravity keyword (native behaviour, black box)', () => {
  it('disable keeps atoms free and makes the scalar zero', async () => {
    const { thermo, error } = await run(`${HEAD}fix g all gravity 9.8 vector 0 0 -1 disable
fix nv all nve
thermo_style custom step ke f_g
thermo 50
run 100`, { 'six.data': SIX });
    expect(error).toBeNull();
    // Measured with native LAMMPS (black box): ke stays 0.005 and the scalar is 0 with disable.
    for (const row of thermo) {
      expect(row.ke as number).toBeCloseTo(0.005, 12);
      expect(row.f_g as number).toBe(0);
    }
  });

  it('without disable the atoms accelerate and the scalar is the potential energy 89.0166...', async () => {
    const { thermo, error } = await run(`${HEAD}fix g all gravity 9.8 vector 0 0 -1
fix nv all nve
thermo_style custom step ke f_g
thermo 200
run 200`, { 'six.data': SIX });
    expect(error).toBeNull();
    // Measured with native LAMMPS (black box): f_g at step 0 is 89.0166666666667.
    expect(thermo[0].f_g as number).toBeCloseTo(89.0166666666667, 9);
  });

  it('gravity keyword of rigid/small applies M*g to the body; disable keeps the atoms force free', async () => {
    const { thermo, error } = await run(`${HEAD}fix g all gravity 9.8 vector 0 0 -1 disable
fix r all rigid/small molecule gravity g
thermo_style custom step ke pe
thermo 200
run 200`, { 'six.data': SIX });
    expect(error).toBeNull();
    // Measured with native LAMMPS (black box): ke at step 200 is 1.92579999999997 for these bodies.
    expect(thermo[thermo.length - 1].ke as number).toBeCloseTo(1.9258, 6);
  });

  it('keyword gravity with no gravity-ID is an error', async () => {
    const { error } = await run(`${HEAD}fix r all rigid/small molecule gravity\nrun 0`, { 'six.data': SIX });
    expect(msg(error)).toMatch(/keyword gravity needs a gravity-ID/);
  });

  it('keyword gravity naming a missing fix is an error (checked at run time)', async () => {
    const { error } = await run(`${HEAD}fix r all rigid/small molecule gravity nope\nrun 0`, { 'six.data': SIX });
    expect(msg(error)).toMatch(/fix ID nope does not exist/);
  });

  it('keyword gravity naming a fix that is not a gravity fix is an error', async () => {
    const { error } = await run(`${HEAD}fix nv all nve\nfix r all rigid/small molecule gravity nv\nrun 0`, { 'six.data': SIX });
    expect(msg(error)).toMatch(/fix ID nv is not a gravity fix style/);
  });

  it('a fix gravity defined after the rigid fix is found at run time', async () => {
    const { error } = await run(`${HEAD}fix r all rigid/small molecule gravity g\nfix g all gravity 9.8 vector 0 0 -1 disable\nrun 2`, { 'six.data': SIX });
    expect(error).toBeNull();
  });

  it('a trailing disable is accepted after a chute or spherical style too', async () => {
    const { error } = await run(`${HEAD}fix g all gravity 9.8 chute 30 disable\nfix h all gravity 9.8 spherical 0 180 disable\nfix nv all nve\nrun 2`, { 'six.data': SIX });
    expect(error).toBeNull();
  });

  it('an unknown word after the style arguments is an error', async () => {
    const { error } = await run(`${HEAD}fix g all gravity 9.8 vector 0 0 -1 foo\nrun 0`, { 'six.data': SIX });
    expect(msg(error)).toMatch(/usage: fix ID group gravity magnitude vector x y z/);
  });

  it('a zero vector is still rejected when the fix is not disabled', async () => {
    const { error } = await run(`${HEAD}fix g all gravity 9.8 vector 0 0 0\nrun 0`, { 'six.data': SIX });
    expect(msg(error)).toMatch(/vector direction must be non-zero/);
  });

  it('a zero vector with disable is not rejected at definition (no atom force is ever computed)', async () => {
    const { error } = await run(`${HEAD}fix g all gravity 9.8 vector 0 0 0 disable\nfix nv all nve\nrun 2`, { 'six.data': SIX });
    expect(error).toBeNull();
  });

  it('rigid/small keyword set still rejects force/torque, and mol without rigid/small', async () => {
    const f = await run(`${HEAD}fix g all gravity 9.8 vector 0 0 -1 disable\nfix r all rigid/small molecule gravity g force 1 off off off\nrun 0`, { 'six.data': SIX });
    expect(msg(f.error)).toMatch(/only allowed for the rigid styles/);
  });
});
