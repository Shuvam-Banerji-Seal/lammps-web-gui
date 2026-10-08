import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * atom_style ellipsoid core (atoms.ts, output/data.ts, commands/setup.ts, compute/peratom.ts, output/dump.ts,
 * output/restart.ts). Values and error messages were measured with native LAMMPS (black box); the full
 * read_data / set / displace_atoms / replicate / write_data path is the oracle case tests/oracle/w19ellipsoid.in.
 */

const newSession = () => {
  const files = new Map<string, string>();
  const session = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
  return { session, files };
};

const BOX = `units lj
atom_style ellipsoid
lattice sc 1.0
region box block 0 3 0 1 0 1
create_box 1 box
create_atoms 1 box
`;

describe('atom_style ellipsoid', () => {
  it('creates point particles of mass 1 with the identity quaternion', async () => {
    const { session } = newSession();
    await session.execute(BOX);
    const s = session.sys.state;
    expect(Array.from(s.rmass!)).toEqual([1, 1, 1]);
    expect(Array.from(s.shape!)).toEqual(new Array(9).fill(0));
    expect(Array.from(s.quat!.subarray(0, 4))).toEqual([1, 0, 0, 0]);
  });

  it('refuses the per-type mass command', async () => {
    const { session } = newSession();
    await expect(session.execute(`${BOX}mass 1 2.0\n`)).rejects.toThrow('Cannot set per-type atom mass for atom style ellipsoid');
  });

  it('set shape takes diameters, keeps the mass, and refuses one zero diameter', async () => {
    const { session } = newSession();
    await session.execute(`${BOX}set atom 1 shape 1 2 3\nset atom 2 shape 2 2 2\nset atom 2 shape 0 0 0\n`);
    const s = session.sys.state;
    expect(Array.from(s.shape!.subarray(0, 6))).toEqual([0.5, 1, 1.5, 0, 0, 0]);
    expect(s.rmass![0]).toBe(1);
    await expect(session.execute('set atom 3 shape 1 0 1\n')).rejects.toThrow('Invalid shape in set command');
  });

  it('set density gives density times volume to an ellipsoid and density to a point particle', async () => {
    const { session } = newSession();
    await session.execute(`${BOX}set atom 1 shape 1 2 3\nset atom 1 density 2.0\nset atom 2 density 2.5\n`);
    const s = session.sys.state;
    // measured: 6.28318530717959 and 2.5
    expect(s.rmass![0]).toBeCloseTo(2 * Math.PI, 13);
    expect(s.rmass![1]).toBe(2.5);
  });

  it('set quat normalizes the 4-vector and refuses point particles', async () => {
    const { session } = newSession();
    await session.execute(`${BOX}set atom 1 shape 1 1 1\nset atom 1 quat 1 1 0 90\n`);
    const q = Array.from(session.sys.state.quat!.subarray(0, 4));
    // measured: 0.577350269189626 0.577350269189626 0.577350269189626 0
    for (let d = 0; d < 3; d++) expect(q[d]).toBeCloseTo(1 / Math.sqrt(3), 14);
    expect(q[3]).toBe(0);
    await expect(session.execute('set atom 2 quat 0 0 1 30\n')).rejects.toThrow('Cannot set quaternion for atom that has none');
  });

  it('set quat/random gives unit quaternions (engine seeding) and refuses point particles; dipole/random sets Dlen', async () => {
    const { session } = newSession();
    await session.execute(`${BOX}set atom 1*2 shape 1 2 3\nset atom 1*2 quat/random 4321\n`);
    const q = session.sys.state.quat!;
    for (const i of [0, 1]) expect(Math.hypot(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3])).toBeCloseTo(1, 14);
    expect(Array.from(q.subarray(0, 4))).not.toEqual(Array.from(q.subarray(4, 8)));
    await expect(session.execute('set atom 3 quat/random 4321\n')).rejects.toThrow('Cannot set quaternion for atom that has none');
    const d = newSession();
    await d.session.execute(`units lj
atom_style hybrid sphere dipole
lattice sc 1.0
region box block 0 3 0 1 0 1
create_box 1 box
create_atoms 1 box
set group all dipole/random 77 1.5
`);
    const mu = d.session.sys.state.mu!;
    for (let i = 0; i < 3; i++) {
      expect(Math.hypot(mu[4 * i], mu[4 * i + 1], mu[4 * i + 2])).toBeCloseTo(1.5, 13);
      expect(mu[4 * i + 3]).toBe(1.5);
    }
  });

  it('refuses a quaternion with xy components in 2d', async () => {
    const { session } = newSession();
    const box2d = `units lj
dimension 2
atom_style ellipsoid
lattice sq 1.0
region box block 0 2 0 1 -0.5 0.5
create_box 1 box
create_atoms 1 box
set atom 1 shape 1 2 1
`;
    await expect(session.execute(`${box2d}set atom 1 quat 1 0 0 90\n`)).rejects.toThrow('Cannot set quaternion with xy components for 2d system');
    const { session: ok } = newSession();
    await ok.execute(`${box2d}set atom 1 quat 0 0 1 60\n`);
    const q = Array.from(ok.sys.state.quat!.subarray(0, 4));
    expect(q[0]).toBeCloseTo(Math.cos(Math.PI / 6), 14);
    expect(q[3]).toBeCloseTo(0.5, 14);
  });

  it('property/atom reports diameters and quaternions, with 1 1 1 and 1 0 0 0 for point particles', async () => {
    const { session, files } = newSession();
    await session.execute(`${BOX}set atom 3 shape 2 4 6
set atom 3 quat 0 0 1 60
set atom 2 angmom 0.1 0.2 0.3
compute p all property/atom shapex shapey shapez quatw quati quatj quatk angmomx angmomy angmomz
run 0
write_dump all custom p.dump id c_p[*] angmomz modify sort id format float %.15g
`);
    const rows = files.get('p.dump')!.trim().split('\n').slice(9).map((l) => l.split(' ').map(Number));
    expect(rows[0]).toEqual([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
    expect(rows[1].slice(8)).toEqual([0.1, 0.2, 0.3, 0.3]);
    expect(rows[2].slice(0, 4)).toEqual([3, 2, 4, 6]);
    expect(rows[2][4]).toBeCloseTo(0.866025403784439, 14);
    expect(rows[2][7]).toBeCloseTo(0.5, 14);
  });

  it('dump custom angmom* needs atom_style ellipsoid', async () => {
    const { session } = newSession();
    await expect(session.execute(`units lj
lattice sc 1.0
region box block 0 2 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 1.0
write_dump all custom p.dump id angmomx
`)).rejects.toThrow(/angmomx needs atom_style ellipsoid/);
  });

  it('write_restart / read_restart keeps shape, quaternion, angular momentum and per-atom mass', async () => {
    const { session } = newSession();
    await session.execute(`${BOX}set atom 1 shape 1 2 3
set atom 1 quat 1 2 3 40
set atom 1 density 1.7
set atom 2 angmom 0.1 0.2 0.3
write_restart e.restart
`);
    const s0 = session.sys.state;
    const before = { rmass: Array.from(s0.rmass!), shape: Array.from(s0.shape!), quat: Array.from(s0.quat!), angmom: Array.from(s0.angmom!) };
    await session.execute('clear\nread_restart e.restart\n');
    const s1 = session.sys.state;
    expect({ rmass: Array.from(s1.rmass!), shape: Array.from(s1.shape!), quat: Array.from(s1.quat!), angmom: Array.from(s1.angmom!) }).toEqual(before);
  });
});
