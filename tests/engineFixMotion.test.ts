import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Direct checks of the motion fixes on tiny systems:
 *   fix momentum   (docs.lammps.org/fix_momentum.html)
 *   fix recenter   (docs.lammps.org/fix_recenter.html)
 *   fix nve/limit  (docs.lammps.org/fix_nve_limit.html)
 *   fix nve/noforce (docs.lammps.org/fix_nve_noforce.html)
 * documented formulas (linear momentum, angular-momentum removal, rescale,
 * COM shift, displacement limit) and the argument errors. thermo_modify norm
 * no keeps the raw (unnormalized) fix values.
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
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { session, thermo, error, files, events };
};

/** Per-atom columns of a written custom dump, keyed by atom id. */
const dumpAtoms = (files: Map<string, string>, name: string): Map<number, Record<string, number>> => {
  const lines = (files.get(name) ?? '').trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  const out = new Map<number, Record<string, number>>();
  for (const line of lines.slice(k + 1)) {
    const w = line.trim().split(/\s+/).map(Number);
    const a: Record<string, number> = {};
    cols.forEach((c, i) => { a[c] = w[i]; });
    out.set(a.id, a);
  }
  return out;
};

const errText = (e: unknown): string => String((e as Error)?.message ?? e);

const box = (ntypes: number, size = 10): string => `
units           lj
atom_style      atomic
region          box block 0 ${size} 0 ${size} 0 ${size}
create_box      ${ntypes} box
`;

const VDUMP = 'write_dump all custom v.dump id x y z vx vy vz modify format float %.15g sort id\n';

describe('fix momentum (fix_momentum.html)', () => {
  it('linear subtracts the center-of-mass velocity in flagged dimensions', async () => {
    // two unit masses, v = (2,0,0) and (0,0,0): vcom = (1,0,0)
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 1 5 5
create_atoms    1 single 3 5 5
mass            1 1.0
group           g1 id 1
group           g2 id 2
velocity        g1 set 2.0 0.0 0.0 units box
velocity        g2 set 0.0 0.0 0.0 units box
fix             1 all momentum 1 linear 1 0 0
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump');
    expect(a.get(1)!.vx).toBeCloseTo(1, 12);
    expect(a.get(1)!.vy).toBeCloseTo(0, 12);
    expect(a.get(2)!.vx).toBeCloseTo(-1, 12);
    expect(a.get(2)!.vz).toBeCloseTo(0, 12);
  });

  it('linear 1 1 1 removes all three components', async () => {
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 1 5 5
create_atoms    1 single 3 5 5
mass            1 1.0
group           g1 id 1
group           g2 id 2
velocity        g1 set 2.0 1.0 0.5 units box
velocity        g2 set 0.0 0.0 0.0 units box
fix             1 all momentum 1 linear 1 1 1
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump');
    // vcom = (1, 0.5, 0.25)
    expect(a.get(1)!.vx).toBeCloseTo(1, 12);
    expect(a.get(1)!.vy).toBeCloseTo(0.5, 12);
    expect(a.get(1)!.vz).toBeCloseTo(0.25, 12);
    expect(a.get(2)!.vx).toBeCloseTo(-1, 12);
    expect(a.get(2)!.vy).toBeCloseTo(-0.5, 12);
    expect(a.get(2)!.vz).toBeCloseTo(-0.25, 12);
  });

  it('angular removes exactly the rotational part omega = I^-1 L', async () => {
    // three atoms at (-1,0,0), (0,1,1), (1,2,-1) + (5,5,5); expected
    // velocities measured on native LAMMPS with the same input (zero forces)
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 4 5 5
create_atoms    1 single 5 6 6
create_atoms    1 single 6 7 4
mass            1 1.0
group           g1 id 1
group           g2 id 2
group           g3 id 3
velocity        g1 set 0.3 0.1 0.0 units box
velocity        g2 set -0.2 0.4 0.2 units box
velocity        g3 set 0.1 -0.3 -0.1 units box
fix             1 all momentum 1 angular
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump');
    expect(a.get(1)!.vx).toBeCloseTo(0.2, 12);
    expect(a.get(1)!.vy).toBeCloseTo(0.2, 12);
    expect(a.get(1)!.vz).toBeCloseTo(-0.1, 12);
    expect(a.get(2)!.vx).toBeCloseTo(0.05, 12);
    expect(a.get(2)!.vy).toBeCloseTo(0.05, 12);
    expect(a.get(2)!.vz).toBeCloseTo(0.2, 12);
    expect(a.get(3)!.vx).toBeCloseTo(-0.05, 12);
    expect(a.get(3)!.vy).toBeCloseTo(-0.05, 12);
    expect(a.get(3)!.vz).toBeCloseTo(0, 12);
  });

  it('linear 1 1 1 angular rescale conserves the raw kinetic energy', async () => {
    // same three-atom configuration; expected velocities measured on native
    // LAMMPS with the same input (ke 0.225 before and after, norm no)
    const { files, thermo } = await runScript(`
${box(1)}
create_atoms    1 single 4 5 5
create_atoms    1 single 5 6 6
create_atoms    1 single 6 7 4
mass            1 1.0
group           g1 id 1
group           g2 id 2
group           g3 id 3
velocity        g1 set 0.3 0.1 0.0 units box
velocity        g2 set -0.2 0.4 0.2 units box
velocity        g3 set 0.1 -0.3 -0.1 units box
fix             1 all momentum 1 linear 1 1 1 angular rescale
thermo_style    custom step ke
thermo_modify   norm no format float %.15g
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump');
    expect(a.get(1)!.vx).toBeCloseTo(0.269679944985297, 12);
    expect(a.get(1)!.vy).toBeCloseTo(0.269679944985297, 12);
    expect(a.get(1)!.vz).toBeCloseTo(-0.269679944985297, 12);
    expect(a.get(2)!.vx).toBeCloseTo(-0.0337099931231621, 12);
    expect(a.get(2)!.vy).toBeCloseTo(-0.0337099931231621, 12);
    expect(a.get(2)!.vz).toBeCloseTo(0.337099931231621, 12);
    expect(a.get(3)!.vx).toBeCloseTo(-0.235969951862135, 12);
    expect(a.get(3)!.vy).toBeCloseTo(-0.235969951862135, 12);
    expect(a.get(3)!.vz).toBeCloseTo(-0.0674199862463242, 12);
    expect(thermo[thermo.length - 1].ke).toBeCloseTo(0.225, 12);
  });

  it('rescale restores the raw kinetic energy 1/2 sum m v^2 after removal', async () => {
    // vcom = (1,0,0); raw KE 2 before, 1 after the linear removal;
    // rescale multiplies all velocities by sqrt(2)
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 1 5 5
create_atoms    1 single 3 5 5
mass            1 1.0
group           g1 id 1
group           g2 id 2
velocity        g1 set 2.0 0.0 0.0 units box
velocity        g2 set 0.0 0.0 0.0 units box
fix             1 all momentum 1 linear 1 1 1 rescale
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump');
    expect(a.get(1)!.vx).toBeCloseTo(Math.SQRT2, 9);
    expect(a.get(2)!.vx).toBeCloseTo(-Math.SQRT2, 9);
    const ke = 0.5 * (a.get(1)!.vx ** 2 + a.get(2)!.vx ** 2);
    expect(ke).toBeCloseTo(2, 12);
  });

  it('rejects bad arguments', async () => {
    const base = `${box(1)}
create_atoms    1 single 1 5 5
mass            1 1.0
`;
    expect(errText((await runScript(`${base}\nfix 1 all momentum 1\nrun 1\n`)).error)).toMatch(/linear/);
    expect(errText((await runScript(`${base}\nfix 1 all momentum 0 linear 1 1 1\nrun 1\n`)).error)).toMatch(/N must be a positive integer/);
    expect(errText((await runScript(`${base}\nfix 1 all momentum 1 linear 1 2 1\nrun 1\n`)).error)).toMatch(/0\/1/);
    expect(errText((await runScript(`${base}\nfix 1 all momentum 1 linear 1 1\nrun 1\n`)).error)).toMatch(/0\/1/);
    expect(errText((await runScript(`${base}\nfix 1 all momentum 1 bogus\nrun 1\n`)).error)).toMatch(/unknown keyword/);
  });
});

describe('fix recenter (fix_recenter.html)', () => {
  it('shifts the group to the target and reports scalar = |shift|, vector = shift', async () => {
    // two unit masses at x = 2 and 4 (COM = (3,0,0)); target (5, 0.5, 0) box
    const { thermo, files } = await runScript(`
${box(1)}
create_atoms    1 single 2 0 0
create_atoms    1 single 4 0 0
mass            1 1.0
fix             1 all recenter 5.0 0.5 0.0 units box
thermo_style    custom step f_1 f_1[1] f_1[2] f_1[3]
thermo_modify   norm no format float %.15g
run             1
${VDUMP}
`);
    const r = thermo[thermo.length - 1];
    expect(r['f_1[1]']).toBeCloseTo(2, 12);
    expect(r['f_1[2]']).toBeCloseTo(0.5, 12);
    expect(r['f_1[3]']).toBeCloseTo(0, 12);
    expect(r['f_1']).toBeCloseTo(Math.sqrt(2 * 2 + 0.5 * 0.5), 12);
    const a = dumpAtoms(files, 'v.dump');
    expect(a.get(1)!.x).toBeCloseTo(4, 12);
    expect(a.get(1)!.y).toBeCloseTo(0.5, 12);
    expect(a.get(2)!.x).toBeCloseTo(6, 12);
    expect(a.get(2)!.y).toBeCloseTo(0.5, 12);
  });

  it('INIT pins the COM to its value at the beginning of the run, NULL excludes', async () => {
    // atoms at x = 2, 4 moving with v = (1,0,0); recenter INIT NULL 0.0 box
    const { thermo, files } = await runScript(`
${box(1)}
timestep        0.001
create_atoms    1 single 2 5 5
create_atoms    1 single 4 5 5
mass            1 1.0
group           g1 id 1
group           g2 id 2
velocity        g1 set 1.0 0.0 0.0 units box
velocity        g2 set 1.0 0.0 0.0 units box
fix             1 all nve
fix             2 all recenter INIT NULL 0.0 units box
thermo_style    custom step f_2 f_2[1] f_2[2] f_2[3]
thermo_modify   norm no format float %.15g
run             2
${VDUMP}
`);
    const r = thermo[thermo.length - 1];
    // the COM moves +0.001 per step and is pulled back each step
    expect(r['f_2[1]']).toBeCloseTo(-0.001, 12);
    expect(r['f_2[2]']).toBeCloseTo(0, 12);
    expect(r['f_2[3]']).toBeCloseTo(0, 12);
    const a = dumpAtoms(files, 'v.dump');
    expect(a.get(1)!.x).toBeCloseTo(2, 12);
    expect(a.get(2)!.x).toBeCloseTo(4, 12);
  });

  it('units lattice scales the target by the lattice spacing', async () => {
    const { thermo } = await runScript(`
lattice         sc 2.0
${box(1)}
create_atoms    1 single 2 0 0
create_atoms    1 single 4 0 0
mass            1 1.0
fix             1 all recenter 0.0 1.0 0.0
thermo_style    custom step f_1[2]
thermo_modify   norm no format float %.15g
run             1
`);
    const r = thermo[thermo.length - 1];
    // lj units: the scale is the reduced density (factor^dim = rho/rho*),
    // so 1.0 lattice spacing = (1/2)^(1/3) distance units
    expect(r['f_1[2]']).toBeCloseTo(Math.pow(0.5, 1 / 3), 12);
  });

  it('units fraction measures the target between the lo/hi box boundaries', async () => {
    const { thermo } = await runScript(`
${box(1)}
create_atoms    1 single 2 0 0
create_atoms    1 single 4 0 0
mass            1 1.0
fix             1 all recenter 0.0 0.25 0.0 units fraction
thermo_style    custom step f_1[2]
thermo_modify   norm no format float %.15g
run             1
`);
    const r = thermo[thermo.length - 1];
    // target y = lo + 0.25*(hi-lo) = 2.5
    expect(r['f_1[2]']).toBeCloseTo(2.5, 12);
  });

  it('shift keyword moves a different group than the one the COM is computed on', async () => {
    // COM of group one (atom 1 at x=2) targets 0; the shift applies to all
    const { files } = await runScript(`
${box(1)}
create_atoms    1 single 2 5 5
create_atoms    1 single 4 5 5
mass            1 1.0
group           one id 1
fix             1 one recenter 0.0 0.0 0.0 units box shift all
run             1
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump');
    expect(a.get(1)!.x).toBeCloseTo(0, 12);
    expect(a.get(2)!.x).toBeCloseTo(2, 12);
  });

  it('warns when defined before a time integration fix', async () => {
    // "LAMMPS will warn you if your fixes are not ordered this way"
    const { events } = await runScript(`
${box(1)}
timestep        0.001
create_atoms    1 single 2 5 5
mass            1 1.0
fix             1 all recenter 0.0 0.0 0.0 units box
fix             2 all nve
run             1
`);
    const warnings = events
      .filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log')
      .map((e) => e.text)
      .filter((t) => t.startsWith('WARNING:'));
    expect(warnings.some((t) => t.includes('recenter'))).toBe(true);
  });

  it('rejects bad arguments', async () => {
    const base = `${box(1)}
create_atoms    1 single 2 5 5
mass            1 1.0
`;
    expect(errText((await runScript(`${base}\nfix 1 all recenter 0.0 5.0\nrun 1\n`)).error)).toMatch(/usage/);
    expect(errText((await runScript(`${base}\nfix 1 all recenter 0.0 bogus 0.0 units box\nrun 1\n`)).error)).toMatch(/NULL or INIT/);
    expect(errText((await runScript(`${base}\nfix 1 all recenter 0.0 0.0 0.0 units parsecs\nrun 1\n`)).error)).toMatch(/box or lattice or fraction/);
    expect(errText((await runScript(`${base}\nfix 1 all recenter 0.0 0.0 0.0 shift\nrun 1\n`)).error)).toMatch(/shift needs a group-ID/);
  });
});

describe('fix nve/limit (fix_nve_limit.html)', () => {
  it('limits the displacement to xmax per timestep and counts the clamps', async () => {
    // dt = 0.002, xmax = 0.001 -> vmax = 0.5; v = (10,0,0) is clamped to 0.5
    // at the initial half-step of the first timestep (count 1); no forces,
    // so the later half-steps sit exactly at the limit and do not count
    const { thermo, files } = await runScript(`
${box(1)}
timestep        0.002
create_atoms    1 single 5 5 5
mass            1 1.0
velocity        all set 10.0 0.0 0.0 units box
fix             1 all nve/limit 0.001
thermo_style    custom step f_1
thermo_modify   norm no format float %.15g
run             2
${VDUMP}
`);
    const rows = thermo;
    expect(rows[rows.length - 1]['f_1']).toBe(1);
    const a = dumpAtoms(files, 'v.dump');
    expect(a.get(1)!.vx).toBeCloseTo(0.5, 12);
    // two steps of displacement 0.002*0.5 = 0.001 each
    expect(a.get(1)!.x).toBeCloseTo(5 + 0.002, 12);
  });

  it('does not limit velocities below the criterion and resets the count per run', async () => {
    const { thermo } = await runScript(`
${box(1)}
timestep        0.002
create_atoms    1 single 5 5 5
mass            1 1.0
velocity        all set 0.1 0.0 0.0 units box
fix             1 all nve/limit 0.001
thermo_style    custom step f_1
thermo_modify   norm no format float %.15g
run             3
run             2
`);
    expect(thermo[2]['f_1']).toBe(0);
    // second run starts from step 3 and re-initializes the count
    expect(thermo[thermo.length - 1]['f_1']).toBe(0);
  });

  it('rejects bad arguments', async () => {
    const base = `${box(1)}
create_atoms    1 single 5 5 5
mass            1 1.0
`;
    expect(errText((await runScript(`${base}\nfix 1 all nve/limit\nrun 1\n`)).error)).toMatch(/usage/);
    expect(errText((await runScript(`${base}\nfix 1 all nve/limit -0.5\nrun 1\n`)).error)).toMatch(/positive distance/);
    expect(errText((await runScript(`${base}\nfix 1 all nve/limit 0.5 0.2\nrun 1\n`)).error)).toMatch(/usage/);
  });
});

describe('fix nve/noforce (fix_nve_noforce.html)', () => {
  it('moves atoms with their velocity but ignores forces', async () => {
    // two atoms 0.9 apart: the LJ force acts on both; the noforce atom keeps
    // its velocity exactly while the nve atom's velocity changes
    const { files } = await runScript(`
${box(1)}
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0
timestep        0.002
create_atoms    1 single 4 5 5
create_atoms    1 single 4.9 5 5
mass            1 1.0
group           wall id 1
group           free id 2
velocity        wall set 0.0 0.0 0.0 units box
velocity        free set 0.0 0.0 0.0 units box
fix             1 free nve
fix             2 wall nve/noforce
velocity        wall set 1.0 0.0 0.0 units box
run             5
${VDUMP}
`);
    const a = dumpAtoms(files, 'v.dump');
    // the wall atom moved with v = (1,0,0) throughout: x = 4 + 5*0.002
    expect(a.get(1)!.x).toBeCloseTo(4 + 0.01, 12);
    expect(a.get(1)!.vx).toBeCloseTo(1, 12);
    expect(a.get(1)!.vy).toBeCloseTo(0, 12);
    // the free atom was pushed by the (repulsive at 0.9) pair force
    expect(a.get(2)!.vx).not.toBeCloseTo(0, 3);
  });

  it('takes no arguments', async () => {
    const base = `${box(1)}
create_atoms    1 single 5 5 5
mass            1 1.0
`;
    expect(errText((await runScript(`${base}\nfix 1 all nve/noforce 0.5\nrun 1\n`)).error)).toMatch(/no arguments/);
  });
});

describe('momentum/recenter thermo normalization', () => {
  it('recenter outputs are extensive and normalize like LAMMPS with norm yes', async () => {
    // lj units default norm yes: the displacement divides by the atom count
    const { thermo } = await runScript(`
${box(1)}
create_atoms    1 single 2 0 0
create_atoms    1 single 4 0 0
mass            1 1.0
fix             1 all recenter 0.0 0.5 0.0 units box
thermo_style    custom step f_1 f_1[1]
run             1
`);
    const r = thermo[thermo.length - 1];
    expect(r['f_1[1]']).toBeCloseTo(-3 / 2, 12);
    expect(r['f_1']).toBeCloseTo(Math.sqrt(3 * 3 + 0.5 * 0.5) / 2, 12);
  });
});
