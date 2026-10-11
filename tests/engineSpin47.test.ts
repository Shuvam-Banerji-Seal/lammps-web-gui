import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * SPIN package core (atom_style spin, pair spin/exchange, compute spin, fix nve/spin lattice frozen):
 * parsing, refusals and energy conservation. Parity with native LAMMPS is in tests/oracle/w47spin_*.in.
 */

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    error = err && 'message' in err ? String(err.message) : String(e);
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { error, thermo };
};

const header = `units metal
atom_style spin
boundary p p p
atom_modify map array
lattice sc 2.5
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 55.845
`;

describe('SPIN core', () => {
  it('parses atom_style spin and set spin/atom (direction normalized, magnitude kept)', async () => {
    const r = await run(`${header}set atom 1 spin/atom 2.5 3 0 4\npair_style spin/exchange 4.0\npair_coeff * * exchange 4.0 0.02726 0.2171 1.841 offset no\ncompute s all spin\nthermo_style custom step c_s[4]\nrun 0\n`);
    expect(r.error).toBeNull();
    expect(r.thermo.length).toBe(1);
  });

  it('refuses a zero spin magnitude and a zero direction', async () => {
    const z = await run(`${header}set atom 1 spin/atom 0 1 0 0\n`);
    expect(z.error).toContain('Invalid spin magnitude 0 in set spin/atom command');
    const d = await run(`${header}set atom 1 spin/atom 1 0 0 0\n`);
    expect(d.error).toContain('At least one spin vector component must be non-zero');
  });

  it('refuses a bad lattice value; stt and zero directions', async () => {
    const m = await run(`${header}pair_style spin/exchange 4.0\npair_coeff * * exchange 4.0 0.02726 0.2171 1.841\nfix 1 all nve/spin lattice wobbly\nrun 0\n`);
    expect(m.error).toContain('lattice must be moving or frozen');
    // native LAMMPS runs nve/spin without a spin pair style (a zeeman-only spin rotates)
    const p = await run(`${header}fix 1 all nve/spin lattice frozen\nrun 0\n`);
    expect(p.error).toBeNull();
    const t = await run(`${header}fix 2 all precession/spin stt 1.0 0 0 1\n`);
    expect(t.error).toContain('stt is not implemented');
    const z = await run(`${header}fix 2 all precession/spin zeeman 1.0 0 0 0\n`);
    expect(z.error).toContain('must be non-zero');
  });

  it('refuses an unknown spin pair style with its name', async () => {
    const r = await run(`${header}pair_style spin/dmi 4.0\n`);
    expect(r.error).toMatch(/spin\/dmi/);
  });

  it('runs nve/spin lattice moving (spin and lattice energy conserved over a short run)', async () => {
    const r = await run(`${header}set group all spin/atom 2.0 0.3 -0.5 0.8
pair_style spin/exchange 4.0
pair_coeff * * exchange 4.0 0.02726 0.2171 1.841 offset yes
fix 1 all nve/spin lattice moving
timestep 0.0001
thermo_style custom step pe etotal
thermo 50
run 100
`);
    expect(r.error).toBeNull();
    const e0 = r.thermo[0].etotal as number, e1 = r.thermo[r.thermo.length - 1].etotal as number;
    expect(Math.abs(e1 - e0)).toBeLessThan(1e-6 * Math.max(1, Math.abs(e0)));
  });

  it('zeeman precession rotates a lone spin and adds its energy with fix_modify energy yes', async () => {
    const r = await run(`units metal
atom_style spin
boundary f f f
atom_modify map array
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0 0 0
mass 1 55.845
set atom 1 spin/atom 2.2 0.6 0.48 0.64
fix 2 all precession/spin zeeman 10.0 0 0 1
fix_modify 2 energy yes
fix 1 all nve/spin lattice frozen
timestep 0.0001
thermo_style custom step pe
thermo 1
run 1
`);
    expect(r.error).toBeNull();
    // measured with native LAMMPS (black box): -0.001630185216 at step 0 for mu = 2.2, B = 10 T along z, s_z = 0.64
    expect(r.thermo[0].pe as number).toBeCloseTo(-0.001630185216, 9);
  });

  it('conserves the magnetic energy of nve/spin lattice frozen over a short run', async () => {
    const r = await run(`${header}set group all spin/atom 2.0 0.3 -0.5 0.8
set atom 1 spin/atom 1.0 1 0 0
set atom 2 spin/atom 2.0 0 1 0.2
pair_style spin/exchange 4.0
pair_coeff * * exchange 4.0 0.02726 0.2171 1.841 offset no
fix 1 all nve/spin lattice frozen
timestep 0.0005
thermo_style custom step pe
thermo 50
run 200
`);
    expect(r.error).toBeNull();
    const e0 = r.thermo[0].pe as number, e1 = r.thermo[r.thermo.length - 1].pe as number;
    expect(Math.abs(e1 - e0)).toBeLessThan(1e-4 * Math.max(1, Math.abs(e0)));
  });
});
