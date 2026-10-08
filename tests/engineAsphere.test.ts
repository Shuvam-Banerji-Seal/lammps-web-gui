import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix nve/asphere, fix nve/asphere/noforce, compute erotate/asphere and
 * compute temp/asphere (src/engine/fix/nve_asphere.ts, src/engine/compute/asphere.ts).
 * Native parity (asymmetric top, a mixed ellipsoid and point-particle fluid,
 * the noforce variant) is in tests/oracle/w19asphere_*.in. The checks here are
 * the torque-free physics (energy and angular momentum of a free asymmetric
 * top, a closed-form rotation about a principal axis), the degrees of freedom
 * and the bias and tensor definitions measured with native LAMMPS (black box),
 * and the refusals.
 */

const newSession = () => {
  const files = new Map<string, string>();
  const session = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
  return { session, files };
};

const BOX = `units lj
atom_style ellipsoid
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5.0 5.0 5.0 units box
`;

/** One asymmetric ellipsoid with half-axes 1, 1.5, 2 and mass 2 (diameters in set shape). */
const TOP = `${BOX}set atom 1 shape 2.0 3.0 4.0
set atom 1 mass 2.0
`;

const quatNorm = (q: Float64Array, i: number): number => Math.hypot(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);

/** Rotational energy 1/2 sum_k Lb_k^2 / I_k of ellipsoid 0 (independent of the engine's helpers). */
const rotEnergy = (q: Float64Array, L: Float64Array, shape: [number, number, number], m: number): number => {
  const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
  const R = [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
  const [a, b, c] = shape;
  const I = [(m / 5) * (b * b + c * c), (m / 5) * (a * a + c * c), (m / 5) * (a * a + b * b)];
  let e = 0;
  for (let k = 0; k < 3; k++) {
    const lb = R[k] * L[0] + R[3 + k] * L[1] + R[6 + k] * L[2];
    e += (lb * lb) / I[k];
  }
  return 0.5 * e;
};

describe('fix nve/asphere: free rotation of an asymmetric top', () => {
  it('conserves the rotational energy and the unit quaternion over 500 steps', async () => {
    const { session } = newSession();
    await session.execute(`${TOP}set atom 1 quat 0.3 1.0 0.5 47
set atom 1 angmom 0.5 -0.8 1.3
velocity all set 0.1 -0.05 0.2
pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0
timestep 0.002
fix 1 all nve/asphere
compute erot all erotate/asphere
run 0
`);
    const s = session.sys.state;
    const e0 = session.sys.compute('erot').scalarValue();
    const shape: [number, number, number] = [1, 1.5, 2];
    expect(e0).toBeCloseTo(rotEnergy(s.quat!, s.angmom!, shape, 2), 12);
    await session.execute('run 500\n');
    const e1 = session.sys.compute('erot').scalarValue();
    expect(Math.abs(e1 - e0) / e0).toBeLessThan(1e-8);
    expect(Math.abs(quatNorm(s.quat!, 0) - 1)).toBeLessThan(1e-12);
  });

  it('rotates about a principal axis with the closed-form angle (omega = L / I)', async () => {
    const { session } = newSession();
    // L along body x (identity orientation): I_x = m/5 (b^2+c^2) = 2.5, omega_x = 1 / 2.5 = 0.4
    await session.execute(`${TOP}set atom 1 angmom 1 0 0
timestep 0.001
fix 1 all nve/asphere
run 1000
`);
    const q = session.sys.state.quat!;
    const theta = 0.4 * 1.0;
    expect(Math.abs(q[0] - Math.cos(theta / 2))).toBeLessThan(1e-5);
    expect(Math.abs(q[1] - Math.sin(theta / 2))).toBeLessThan(1e-5);
    expect(Math.abs(q[2])).toBeLessThan(1e-12);
    expect(Math.abs(q[3])).toBeLessThan(1e-12);
  });

  it('keeps the space-frame angular momentum fixed without torque', async () => {
    const { session } = newSession();
    await session.execute(`${TOP}set atom 1 quat 0.3 1.0 0.5 47
set atom 1 angmom 0.5 -0.8 1.3
timestep 0.002
fix 1 all nve/asphere
run 200
`);
    const L = session.sys.state.angmom!;
    expect([L[0], L[1], L[2]]).toEqual([0.5, -0.8, 1.3]);
  });

  it('noforce updates the orientation and positions but not the velocity or angular momentum', async () => {
    const { session } = newSession();
    await session.execute(`${TOP}set atom 1 quat 0.3 1.0 0.5 47
set atom 1 angmom 0.5 -0.8 1.3
velocity all set 0.1 -0.05 0.2
timestep 0.002
fix 1 all nve/asphere/noforce
run 50
`);
    const s = session.sys.state;
    expect([s.angmom![0], s.angmom![1], s.angmom![2]]).toEqual([0.5, -0.8, 1.3]);
    expect([s.v[0], s.v[1], s.v[2]]).toEqual([0.1, -0.05, 0.2]);
    expect(s.x[0]).toBeCloseTo(5 + 50 * 0.002 * 0.1, 12);
    expect(Math.abs(quatNorm(s.quat!, 0) - 1)).toBeLessThan(1e-12);
  });

  it('refuses point particles in the group and unknown keywords', async () => {
    const { session } = newSession();
    await expect(session.execute(`${TOP}create_atoms 1 single 2.0 2.0 2.0 units box
fix 1 all nve/asphere
run 0
`)).rejects.toThrow(/requires extended particles/);
    const second = newSession();
    await expect(second.session.execute(`${TOP}fix 1 all nve/asphere foo
`)).rejects.toThrow(/Illegal fix nve\/asphere keyword foo/);
  });

  it('needs an atom style with shape, quaternion and angular momentum', async () => {
    const { session } = newSession();
    await expect(session.execute(`units lj
atom_style sphere
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5.0 5.0 5.0 units box
fix 1 all nve/asphere
`)).rejects.toThrow(/atom style ellipsoid/);
  });
});

describe('compute erotate/asphere', () => {
  it('equals 1/2 I omega^2 with I = m/5 (b^2+c^2) etc. (identity orientation, L = (1,1,0))', async () => {
    const { session } = newSession();
    // identity orientation: the body frame is the space frame; omega = (0.4, 0.5, 0); E = 1/2 (1*0.4 + 1*0.5)
    await session.execute(`${TOP}set atom 1 quat 1 0 0 0
set atom 1 angmom 1 1 0
compute erot all erotate/asphere
run 0
`);
    expect(session.sys.compute('erot').scalarValue()).toBeCloseTo(0.45, 12);
  });

  it('refuses arguments and point particles', async () => {
    const { session } = newSession();
    await expect(session.execute(`${TOP}compute e all erotate/asphere foo\n`)).rejects.toThrow(/Illegal compute erotate\/asphere keyword foo/);
    const second = newSession();
    await expect(second.session.execute(`${TOP}create_atoms 1 single 2.0 2.0 2.0 units box
compute e all erotate/asphere
run 0
`)).rejects.toThrow(/requires extended particles/);
  });
});

describe('compute temp/asphere', () => {
  it('uses 6N - 3 degrees of freedom for dof all and 3N for dof rotate (measured with native LAMMPS)', async () => {
    const { session } = newSession();
    await session.execute(`${TOP}create_atoms 1 single 2.0 2.0 2.0 units box
set atom 2 shape 2.0 2.0 2.0
set atom 2 mass 2.0
set atom 2 angmom 0.5 -0.8 1.3
compute tmp all temp/asphere
compute tmr all temp/asphere dof rotate
run 0
`);
    // measured: dof all 9 (6*2-3), temperature 0.35833333 (2 E / 9 with E = 1.6125); dof rotate 6, temperature 0.5375
    expect(session.sys.compute('tmp').scalarValue()).toBeCloseTo(0.358333333333333, 12);
    expect(session.sys.compute('tmr').scalarValue()).toBeCloseTo(0.5375, 12);
  });

  it('gives the bias-removed temperature of the measured example (temp/com bias, dof all)', async () => {
    const { session } = newSession();
    await session.execute(`${TOP}create_atoms 1 single 2.0 2.0 2.0 units box
set atom 2 shape 2.0 2.0 2.0
set atom 2 mass 1.0
set atom 2 angmom 0.5 -0.8 1.3
velocity all set 1 0 0
compute tc all temp/com
compute tmp all temp/asphere
compute tmb all temp/asphere bias tc
run 0
`);
    // measured with native LAMMPS: 1.05 without bias, 0.71666667 with the temp/com bias
    expect(session.sys.compute('tmp').scalarValue()).toBeCloseTo(1.05, 12);
    expect(session.sys.compute('tmb').scalarValue()).toBeCloseTo(0.716666666666667, 12);
  });

  it('refuses unknown keywords, a bad dof value, a missing bias compute and point particles', async () => {
    await expect(newSession().session.execute(`${TOP}compute t all temp/asphere foo bar\n`)).rejects.toThrow(/Illegal compute temp\/asphere keyword foo/);
    await expect(newSession().session.execute(`${TOP}compute t all temp/asphere dof trans\n`)).rejects.toThrow(/dof keyword trans/);
    await expect(newSession().session.execute(`${TOP}compute t all temp/asphere bias nobias\n`)).rejects.toThrow(/nobias/);
    const second = newSession();
    await expect(second.session.execute(`${TOP}create_atoms 1 single 2.0 2.0 2.0 units box
compute t all temp/asphere
run 0
`)).rejects.toThrow(/requires all extended particles/);
  });

  it('requires an ellipsoid atom style', async () => {
    const { session } = newSession();
    await expect(session.execute(`units lj
atom_style sphere
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5.0 5.0 5.0 units box
compute t all temp/asphere
`)).rejects.toThrow(/atom style ellipsoid/);
  });

  it('gives the rotational tensor in the body frame, with trace 2 E (measured with native LAMMPS)', async () => {
    const { session } = newSession();
    // identity orientation, L = (1,1,0): measured tensor xx 0.4, yy 0.5, zz 0, xy 0.5, xz 0, yz 0
    await session.execute(`${TOP}set atom 1 quat 1 0 0 0
set atom 1 angmom 1 1 0
compute t all temp/asphere dof rotate
compute erot all erotate/asphere
run 0
`);
    const t = session.sys.compute('t').vectorValues();
    expect(t[0]).toBeCloseTo(0.4, 12);
    expect(t[1]).toBeCloseTo(0.5, 12);
    expect(t[2]).toBeCloseTo(0, 12);
    expect(t[3]).toBeCloseTo(0.5, 12);
    expect(t[4]).toBeCloseTo(0, 12);
    expect(t[5]).toBeCloseTo(0, 12);
    expect(t[0] + t[1] + t[2]).toBeCloseTo(2 * session.sys.compute('erot').scalarValue(), 12);
  });
});
