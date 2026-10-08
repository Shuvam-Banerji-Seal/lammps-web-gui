import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Direct checks of fix move (docs.lammps.org/fix_move.html) on tiny systems:
 * the linear and wiggle formulas (position and velocity), the NULL component
 * integrated like fix nve, the units box|lattice scaling, the time origin
 * (the timestep the fix is specified, captured X0) and the argument errors.
 * The measured native values behind the expectations are quoted in the
 * source of src/engine/fix/move.ts; the oracle cases
 * tests/oracle/w17move_linear.in and w17move_wiggle.in cover larger systems.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  return { session, files, error, events };
};

/** First frame of a written custom dump, keyed by atom id. */
const dumpAtoms = (files: Map<string, string>, name: string): Map<number, Record<string, number>> => {
  const lines = (files.get(name) ?? '').trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  const out = new Map<number, Record<string, number>>();
  for (const line of lines.slice(k + 1)) {
    if (!line.trim()) continue;
    const w = line.trim().split(/\s+/).map(Number);
    const a: Record<string, number> = {};
    cols.forEach((c, i) => { a[c] = w[i]; });
    out.set(a.id, a);
    break;
  }
  return out;
};

const errText = (e: unknown): string => String((e as Error)?.message ?? e);

const box = (ntypes: number, size = 20): string => `
units           lj
atom_style      atomic
region          box block 0 ${size} 0 ${size} 0 ${size}
create_box      ${ntypes} box
`;

const VDUMP = 'write_dump all custom v.dump id x y z xu yu zu vx vy vz modify format float %.15g sort id\n';

describe('fix move linear (fix_move.html)', () => {
  it('moves at constant velocity and sets the velocity (units box)', async () => {
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5 units box
mass            1 1.0
velocity        all set 0.0 0.0 0.0
fix             1 all move linear 0.5 0.0 0.0 units box
run             4
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump').get(1)!;
    // X(t) = X0 + V*delta with delta = 4*0.005
    expect(a.x).toBeCloseTo(5 + 0.5 * 4 * 0.005, 12);
    expect(a.vx).toBeCloseTo(0.5, 12);
    expect(a.vy).toBeCloseTo(0, 12);
  });

  it('uses delta since the fix was specified, not since the run started', async () => {
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5 units box
mass            1 1.0
velocity        all set 0.0 0.0 0.0
fix             1 all nve
run             2
unfix           1
fix             2 all move linear 0.5 0.0 0.0 units box
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump').get(1)!;
    // fix specified at step 2: on step 3 delta = 1*dt, not 3*dt
    expect(a.x).toBeCloseTo(5 + 0.5 * 0.005, 12);
  });

  it('captures X0 at the fix command, not at run start', async () => {
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5 units box
mass            1 1.0
velocity        all set 0.0 0.0 0.0
fix             1 all move linear 0.5 0.0 0.0 units box
displace_atoms  all move 1.0 0.0 0.0 units box
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump').get(1)!;
    // X0 = 5 (at the fix command), so x = 5 + V*dt even after the displacement
    expect(a.x).toBeCloseTo(5 + 0.5 * 0.005, 12);
  });

  it('integrates a NULL component exactly like fix nve', async () => {
    const common = `
${box(1)}
create_atoms    1 single 5 5 5 units box
mass            1 1.0
velocity        all set 0.0 0.0 0.0
fix             2 all addforce 1.0 0.0 0.0
`;
    const ref = await runScript(`${common}fix 1 all nve\nrun 5\n${VDUMP}`);
    const nul = await runScript(`${common}fix 1 all move linear NULL 0.0 0.0 units box\nrun 5\n${VDUMP}`);
    const r = dumpAtoms(ref.files, 'v.dump').get(1)!;
    const n = dumpAtoms(nul.files, 'v.dump').get(1)!;
    expect(n.x).toBeCloseTo(r.x, 12);
    expect(n.vx).toBeCloseTo(r.vx, 12);
    // the non-NULL components are held at their X0 and V
    expect(n.y).toBeCloseTo(5, 12);
    expect(n.vy).toBeCloseTo(0, 12);
  });

  it('scales velocity by the lattice spacing for units lattice (the default)', async () => {
    const spacing = Math.pow(0.5, 1 / 3); // lattice sc 2.0 in lj units
    const { files } = await runScript(`
units           lj
atom_style      atomic
lattice         sc 2.0
region          box block 0 20 0 20 0 20
create_box      1 box
create_atoms    1 single 5 5 5 units box
mass            1 1.0
velocity        all set 0.0 0.0 0.0
fix             1 all move linear 1.0 0.0 0.0
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump').get(1)!;
    expect(a.vx).toBeCloseTo(spacing, 12);
    expect(a.x).toBeCloseTo(5 + spacing * 0.005, 12);
  });

  it('wraps moved atoms through periodic boundaries, keeping xu continuous', async () => {
    const { files } = await runScript(`
${box(1, 10)}
create_atoms    1 single 9.8 5 5 units box
mass            1 1.0
velocity        all set 0.0 0.0 0.0
fix             1 all move linear 10.0 0.0 0.0 units box
run             3
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump').get(1)!;
    expect(a.xu).toBeCloseTo(9.8 + 10 * 3 * 0.005, 12);
    expect(a.x).toBeCloseTo(9.95, 12);
  });
});

describe('fix move wiggle (fix_move.html)', () => {
  it('oscillates X = X0 + A sin(omega*delta) with the derivative as velocity', async () => {
    const A = 0.2, T = 1.0, dt = 0.005;
    const omega = (2 * Math.PI) / T;
    const delta = 2 * dt;
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5 units box
mass            1 1.0
velocity        all set 0.0 0.0 0.0
fix             1 all move wiggle 0.2 0.0 0.0 ${T} units box
run             2
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump').get(1)!;
    expect(a.x).toBeCloseTo(5 + A * Math.sin(omega * delta), 12);
    expect(a.vx).toBeCloseTo(A * omega * Math.cos(omega * delta), 12);
  });

  it('leaves a NULL amplitude component to fix nve', async () => {
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5 units box
mass            1 1.0
velocity        all set 0.0 0.0 0.0
fix             2 all addforce 0.0 0.0 1.0
fix             1 all move wiggle 0.1 0.0 NULL 2.0 units box
run             3
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump').get(1)!;
    // z is nve-integrated under fz = 1: after 3 steps v = 3*dt*ftm2v, x += ...
    expect(a.vz).toBeCloseTo(3 * 0.005, 12);
    expect(a.z).toBeGreaterThan(5);
    // x oscillates with the wiggle; the non-NULL y = 0.0 holds y at X0
    const omega = Math.PI;
    expect(a.x).toBeCloseTo(5 + 0.1 * Math.sin(omega * 3 * 0.005), 12);
    expect(a.y).toBeCloseTo(5, 12);
  });
});

describe('fix move errors (fix_move.html)', () => {
  const base = `${box(1)}\ncreate_atoms 1 single 5 5 5 units box\nmass 1 1.0\nvelocity all set 0 0 0\n`;
  it('throws StyleError naming variable; transrot refuses NULL translation components', async () => {
    expect(errText((await runScript(`${base}fix 1 all move variable v_x NULL NULL NULL NULL NULL\nrun 1\n`)).error)).toMatch(/variable/);
    expect(errText((await runScript(`${base}fix 1 all move transrot NULL 0 0 0 0 0 0 0 1 5\nrun 1\n`)).error)).toMatch(/transrot: velocity components must be a number/);
    expect(errText((await runScript(`${base}fix 1 all move rotate 0 0 0 0 0 0 5\nrun 1\n`)).error)).toMatch(/zero length rotation vector/);
  });

  it('rejects bad arguments and keywords', async () => {
    expect(errText((await runScript(`${base}fix 1 all move linear 1 0\nrun 1\n`)).error)).toMatch(/usage/);
    expect(errText((await runScript(`${base}fix 1 all move linear 1 bogus 0\nrun 1\n`)).error)).toMatch(/velocity components must be a number/);
    expect(errText((await runScript(`${base}fix 1 all move linear 1 0 0 units parsecs\nrun 1\n`)).error)).toMatch(/box or lattice/);
    expect(errText((await runScript(`${base}fix 1 all move linear 1 0 0 bogus\nrun 1\n`)).error)).toMatch(/unknown keyword/);
    expect(errText((await runScript(`${base}fix 1 all move linear 1 0 0 units box update dipole\nrun 1\n`)).error)).toMatch(/rotate or transrot/);
    expect(errText((await runScript(`${base}fix 1 all move wiggle 1 0 0 -2\nrun 1\n`)).error)).toMatch(/period/);
  });
});
