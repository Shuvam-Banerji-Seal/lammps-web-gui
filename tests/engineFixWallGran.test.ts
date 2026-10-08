import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Direct checks for fix wall/gran (docs.lammps.org/fix_wall_gran.html) and
 * fix freeze (docs.lammps.org/fix_freeze.html). A single sphere of diameter 1
 * (density 1, so m = 4/3 pi 0.125 = 0.5236) touches a zplane wall at 0.0 from
 * z = 0.4 (overlap delta = 0.1). The force on the sphere is read through a
 * fix freeze probe placed after the wall: its global vector is the force on
 * the group before zeroing, which for a wall contact equals the wall force
 * (measured with native LAMMPS).
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  let error: Error | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as Error;
  }
  const rows = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { rows, error };
};

const BASE = `
units           lj
boundary        f f f
atom_style      sphere
atom_modify     map array
comm_modify     vel yes
region          box block -5 5 -5 5 -5 5
create_box      1 box
create_atoms    1 single 0 0 0.4 units box
set atom 1 diameter 1.0 density 1.0
pair_style      none
timestep        0.0001
thermo_modify   norm no
`;

/** Force on the sphere from the wall: a freeze probe after the wall fix (run 0 = setup). */
const probeForce = async (wall: string, setVel = '0 0 0', extra = '') => {
  const text = `${BASE}
set atom 1 vx ${setVel.split(' ')[0]} vy ${setVel.split(' ')[1]} vz ${setVel.split(' ')[2]}
${extra}
fix             w all wall/gran ${wall}
group           probe id 1
fix             p probe freeze
thermo_style    custom step f_p[1] f_p[2] f_p[3]
run             0
`;
  const { rows, error } = await runScript(text);
  expect(error).toBeNull();
  const r = rows[0] as unknown as Record<string, number>;
  return [r['f_p[1]'], r['f_p[2]'], r['f_p[3]']];
};

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-9 + 1e-9 * Math.abs(b);

describe('fix wall/gran argument errors', () => {
  const cases: [string, string, RegExp][] = [
    ['unknown fstyle', 'foo 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL', /foo/],
    ['zcylinder wallstyle', 'hooke 2000.0 NULL 50.0 NULL 0.5 0 zcylinder 1.0', /zcylinder/],
    ['contacts keyword', 'hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL contacts', /contacts/],
    ['temperature keyword', 'hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL temperature 1.0', /temperature/],
    ['both wiggle and shear', 'hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL wiggle z 0.1 2.0 shear x 0.1', /Cannot wiggle and shear/],
    ['shear along the wall normal', 'hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL shear z 0.1', /Invalid shear direction/],
    ['both planes NULL', 'hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane NULL NULL', /NULL/],
    ['dampflag not 0 or 1', 'hooke 2000.0 NULL 50.0 NULL 0.5 2 zplane 0.0 NULL', /dampflag/],
    ['wiggle period not positive', 'hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL wiggle z 0.1 0', /period/],
    ['unknown keyword', 'hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL foo', /foo/],
  ];
  for (const [name, wall, re] of cases) {
    it(`rejects ${name}`, async () => {
      const { error } = await runScript(`${BASE}\nfix w all wall/gran ${wall}\nrun 0\n`);
      expect(error?.message ?? '').toMatch(re);
    });
  }

  it('rejects a periodic dimension for a wall', async () => {
    const text = `units lj
boundary p p f
atom_style sphere
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5 5 2
set atom 1 diameter 1.0 density 1.0
pair_style none
fix w all wall/gran hooke 2000.0 NULL 50.0 NULL 0.5 0 xplane 0.0 NULL
run 0
`;
    const { error } = await runScript(text);
    expect(error?.message ?? '').toMatch(/periodic/);
  });

  it('rejects fix wall/gran/region', async () => {
    const { error } = await runScript(`${BASE}\nfix w all wall/gran/region box hooke 2000.0 NULL 50.0 NULL 0.5 0\nrun 0\n`);
    expect(error?.message ?? '').toMatch(/wall\/gran\/region/);
  });

  it('rejects a second freeze fix and freeze arguments', async () => {
    const twice = await runScript(`${BASE}\ngroup g id 1\nfix a g freeze\nfix b g freeze\nrun 0\n`);
    expect(twice.error?.message ?? '').toMatch(/single freeze/);
    const args = await runScript(`${BASE}\ngroup g id 1\nfix a g freeze foo\nrun 0\n`);
    expect(args.error?.message ?? '').toMatch(/freeze/);
  });
});

describe('fix wall/gran single-contact forces (zplane lo at 0, sphere at z = 0.4, delta = 0.1)', () => {
  it('hooke: F_z = Kn delta = 200 at rest', async () => {
    const [fx, fy, fz] = await probeForce('hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL');
    expect(close(fx, 0)).toBe(true);
    expect(close(fy, 0)).toBe(true);
    expect(close(fz, 200)).toBe(true);
  });

  it('hooke: normal damping adds m gamma_n |v_n| (vz = -0.3): F_z = 200 + 0.5236*50*0.3', async () => {
    const m = (4 / 3) * Math.PI * 0.125;
    const [, , fz] = await probeForce('hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 NULL', '0 0 -0.3');
    expect(close(fz, 200 + m * 50 * 0.3)).toBe(true);
  });

  it('hertz/history: F_z = Kn delta sqrt(delta R) = 2000 * 0.1 * sqrt(0.05)', async () => {
    const [, , fz] = await probeForce('hertz/history 2000.0 NULL 0.0 NULL 0.5 0 zplane 0.0 NULL');
    expect(close(fz, 2000 * 0.1 * Math.sqrt(0.1 * 0.5))).toBe(true);
  });

  it('tangential damping with vx = 0.2: F_x = -m gamma_t v_t', async () => {
    const m = (4 / 3) * Math.PI * 0.125;
    const [fx] = await probeForce('hooke 2000.0 NULL 50.0 30.0 0.5 1 zplane 0.0 NULL', '0.2 0 0');
    expect(close(fx, -m * 30 * 0.2)).toBe(true);
  });

  it('Coulomb cap: |F_t| = xmu |F_n| (vx = 2, xmu = 0.1 gives F_x = -20)', async () => {
    const [fx] = await probeForce('hooke 2000.0 2000.0 50.0 30.0 0.1 1 zplane 0.0 NULL', '2 0 0');
    expect(close(fx, -20)).toBe(true);
  });

  it('limit_damping zeroes the contact when F_n < 0 (gn = 1000, vz = 1)', async () => {
    const [fx, fy, fz] = await probeForce('hooke 2000.0 NULL 1000.0 NULL 0.5 0 limit_damping zplane 0.0 NULL', '0 0 1.0');
    expect(close(fz, 0) && close(fx, 0) && close(fy, 0)).toBe(true);
  });

  it('wiggle z 0.1 2.0 at t = 0.1 moves the wall to 0.1 - 0.1 cos(0.1 pi)', async () => {
    const text = `${BASE}
fix             w all wall/gran hooke 2000.0 NULL 0.0 NULL 0.5 0 zplane 0.0 NULL wiggle z 0.1 2.0
group           probe id 1
fix             p probe freeze
thermo_style    custom step f_p[3]
run             0
run             1000
`;
    const { rows, error } = await runScript(text);
    expect(error).toBeNull();
    const last = rows[rows.length - 1] as unknown as Record<string, number>;
    const wall = 0.1 - 0.1 * Math.cos(0.1 * Math.PI);
    expect(close(last['f_p[3]'], 2000 * (0.5 - (0.4 - wall)))).toBe(true);
  });
});
