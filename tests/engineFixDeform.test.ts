import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Direct checks of fix deform (docs.lammps.org/fix_deform.html) and compute
 * temp/deform (docs.lammps.org/compute_temp_deform.html) on tiny systems:
 * the documented length formulas L(t) = L0(1+erate*dt), L0 exp(trate*dt),
 * L0 + A sin(2*pi t/Tp), the linear final/delta/vel ramps, the volume style,
 * the tilt styles, remap x/v/none and the argument errors.
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

const errText = (e: unknown): string => String((e as Error)?.message ?? e);

const box = (prism = false): string => `
units           lj
atom_style      atomic
region          box ${prism ? 'prism' : 'block'} 0 10 0 10 0 10${prism ? ' 0.0 0.0 0.0' : ''}
create_box      1 box
mass            1 1.0
`;

const DUMP = 'write_dump all custom d.dump id x y z vx vy vz ix iy iz modify format float %.15g sort id\n';

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

const thermoOf = (rows: ThermoRow[], step: number): ThermoRow => rows.find((r) => r.step === step)!;

describe('fix deform length styles', () => {
  it('erate: L(t) = L0 (1 + erate*dt) around the mid point', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 x erate 0.1 units box
      thermo_style custom step lx xlo xhi
      thermo 10
      run 50
    `);
    const t = 50 * 0.001;
    const lx = 10 * (1 + 0.1 * t);
    expect(thermoOf(thermo, 50).lx).toBeCloseTo(lx, 9);
    // expanded or compressed around its mid point (xlo = 5 - L/2, xhi = 5 + L/2)
    expect(thermoOf(thermo, 50).xlo).toBeCloseTo(5 - lx / 2, 9);
    expect(thermoOf(thermo, 50).xhi).toBeCloseTo(5 + lx / 2, 9);
    expect(thermoOf(thermo, 30).lx).toBeCloseTo(10 * (1 + 0.1 * 30 * 0.001), 9);
  });

  it('trate: L(t) = L0 exp(trate*dt)', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 x trate 0.02 units box
      thermo_style custom step lx
      thermo 50
      run 50
    `);
    expect(thermoOf(thermo, 50).lx).toBeCloseTo(10 * Math.exp(0.02 * 0.05), 9);
  });

  it('vel: L(t) = L0 + V*dt', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 x vel 0.5 units box
      thermo_style custom step lx
      thermo 50
      run 50
    `);
    expect(thermoOf(thermo, 50).lx).toBeCloseTo(10 + 0.5 * 0.05, 9);
  });

  it('final and delta ramp both boundaries linearly over the run', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 x final 0.0 11.0 y delta -0.2 0.3 units box
      thermo_style custom step lx xlo xhi ly ylo yhi
      thermo 10
      run 50
    `);
    const f = (50 * 0.001) / 0.05;
    // final: hi ramps from its initial 10 to the target 11 (lo ramps 0 -> 0)
    expect(thermoOf(thermo, 50).xlo).toBeCloseTo(0, 9);
    expect(thermoOf(thermo, 50).xhi).toBeCloseTo(10 + 1 * f, 9);
    expect(thermoOf(thermo, 50).ylo).toBeCloseTo(-0.2 * f, 9);
    expect(thermoOf(thermo, 50).yhi).toBeCloseTo(10 + 0.3 * f, 9);
    const f10 = (10 * 0.001) / 0.05;
    expect(thermoOf(thermo, 10).xhi).toBeCloseTo(10 + 1 * f10, 9);
  });

  it('scale: L = factor * L0 at the end of the run', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 x scale 1.05 units box
      thermo_style custom step lx xlo xhi
      thermo 50
      run 50
    `);
    expect(thermoOf(thermo, 50).lx).toBeCloseTo(10.5, 9);
    // expanded around the mid point: xlo = 5 - 5.25, xhi = 5 + 5.25
    expect(thermoOf(thermo, 50).xlo).toBeCloseTo(5 - 5.25, 9);
    expect(thermoOf(thermo, 50).xhi).toBeCloseTo(5 + 5.25, 9);
  });

  it('volume keeps the box volume constant and splits it evenly', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 x erate 0.1 y volume z volume units box
      thermo_style custom step lx ly lz vol
      thermo 10
      run 50
    `);
    const lx = 10 * (1 + 0.1 * 0.05);
    const ly = Math.sqrt(1000 / lx);
    expect(thermoOf(thermo, 50).lx).toBeCloseTo(lx, 9);
    expect(thermoOf(thermo, 50).ly).toBeCloseTo(ly, 9);
    expect(thermoOf(thermo, 50).lz).toBeCloseTo(ly, 9);
    expect(thermoOf(thermo, 50).vol).toBeCloseTo(1000, 8);
    expect(thermoOf(thermo, 20).vol).toBeCloseTo(1000, 8);
  });

  it('wiggle: L(t) = L0 + A sin(2 pi t / Tp)', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 x wiggle 0.4 0.2 units box
      thermo_style custom step lx
      thermo 50
      run 50
    `);
    // t = Tp/4: the sine is at its maximum
    expect(thermoOf(thermo, 50).lx).toBeCloseTo(10 + 0.4 * Math.sin(2 * Math.PI * 0.05 / 0.2), 9);
  });

  it('variable style takes the length change from an equal-style variable in box units', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      variable dl equal 0.1*elapsed*dt
      variable dv equal 0.0
      fix 2 all deform 1 z variable v_dl v_dv
      thermo_style custom step lz
      thermo 50
      run 50
    `);
    expect(thermoOf(thermo, 50).lz).toBeCloseTo(10 + 0.1 * 50 * 0.001, 9);
  });

  it('deforms only every N steps', async () => {
    const { thermo } = await runScript(`
      ${box()}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 10 x vel 1.0 units box
      thermo_style custom step lx
      thermo 1
      run 25
    `);
    // deformed at steps 10 and 20 with the elapsed time of those steps
    expect(thermoOf(thermo, 9).lx).toBeCloseTo(10, 9);
    expect(thermoOf(thermo, 10).lx).toBeCloseTo(10 + 1.0 * 0.01, 9);
    expect(thermoOf(thermo, 20).lx).toBeCloseTo(10 + 1.0 * 0.02, 9);
    expect(thermoOf(thermo, 25).lx).toBeCloseTo(10 + 1.0 * 0.02, 9);
  });
});

describe('fix deform tilt styles', () => {
  it('xy erate: T(t) = T0 + L0*erate*dt with L0 the perpendicular length', async () => {
    const { thermo } = await runScript(`
      ${box(true)}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 xy erate 0.1 units box
      thermo_style custom step xy
      thermo 50
      run 50
    `);
    expect(thermoOf(thermo, 50).xy).toBeCloseTo(10 * 0.1 * 0.05, 9);
  });

  it('xy final, xz vel and yz delta', async () => {
    const { thermo } = await runScript(`
      ${box(true)}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 xy final 1.0 xz vel 0.4 yz delta 0.3 units box
      thermo_style custom step xy xz yz
      thermo 10
      run 50
    `);
    const f = 1; // (50*0.001)/0.05
    expect(thermoOf(thermo, 50).xy).toBeCloseTo(0 + (1.0 - 0) * f, 9);
    expect(thermoOf(thermo, 50).xz).toBeCloseTo(0.4 * 0.05, 9);
    expect(thermoOf(thermo, 50).yz).toBeCloseTo(0.3 * f, 9);
    expect(thermoOf(thermo, 10).xy).toBeCloseTo(1.0 * 0.2, 9);
  });

  it('yz trate grows the initial tilt exponentially and errors on a zero tilt', async () => {
    const { thermo, error } = await runScript(`
      ${box(true)}
      timestep 0.001
      fix 1 all nve
      fix 2 all deform 1 yz delta 0.3 units box
      run 1
      unfix 2
      fix 2 all deform 1 yz trate 0.5 units box
      thermo_style custom step yz
      thermo 50
      run 50
    `);
    expect(error).toBeNull();
    // after one run the tilt is 0.3; trate restarts from the box at run start.
    // The second run starts at step 1, so at step 50 elapsed = (50-1)*dt.
    // Measured with native LAMMPS (black box, same script): step 50 yz = 0.30744078
    // which equals 0.3*exp(0.5*0.049) = 0.30744077733...
    expect(thermoOf(thermo, 50).yz).toBeCloseTo(0.3 * Math.exp(0.5 * 0.049), 9);
    const zero = await runScript(`
      ${box(true)}
      fix 1 all nve
      fix 2 all deform 1 xz trate 0.5 units box
      run 1
    `);
    expect(errText(zero.error)).toContain('trate');
  });

  it('flip yes reports a tilt factor beyond half the parallel box length', async () => {
    const { error } = await runScript(`
      ${box(true)}
      timestep 0.1
      fix 1 all nve
      fix 2 all deform 1 xy erate 2.0 units box
      run 20
    `);
    expect(errText(error)).toContain('flip');
  });
});

describe('fix deform remap', () => {
  it('remap x keeps fractional coordinates and does not touch velocities', async () => {
    const { files } = await runScript(`
      ${box()}
      create_atoms 1 single 8.0 5.0 5.0
      mass 1 1.0
      timestep 0.01
      fix 1 all nve
      fix 2 all deform 1 x erate 0.5 units box remap x
      run 20
      ${DUMP}
    `);
    // L = 10*(1+0.5*0.2) = 11 around the mid point; initial lamda = 0.8;
    // new lo = 5 - 5.5 = -0.5 -> x = -0.5 + 0.8*11 = 8.3, velocity untouched
    const a = dumpAtoms(files, 'd.dump').get(1)!;
    expect(a.x).toBeCloseTo(-0.5 + 0.8 * 11, 8);
    expect(a.vx).toBe(0);
  });

  it('remap none keeps cartesian coordinates', async () => {
    const { files } = await runScript(`
      ${box()}
      create_atoms 1 single 8.0 5.0 5.0
      mass 1 1.0
      timestep 0.01
      fix 1 all nve
      fix 2 all deform 1 x erate 0.5 units box remap none
      run 20
      ${DUMP}
    `);
    const a = dumpAtoms(files, 'd.dump').get(1)!;
    expect(a.x).toBeCloseTo(8.0, 8);
  });

  it('remap v adds the boundary velocity difference when the atom is wrapped', async () => {
    const { files } = await runScript(`
      ${box(true)}
      create_atoms 1 single 5.0 9.9 5.0
      mass 1 1.0
      timestep 0.1
      velocity all set 0.0 2.0 0.0 units box
      fix 1 all nve
      fix 2 all deform 1 xy erate 1.0 units box remap v
      run 4
      ${DUMP}
    `);
    // vy = 2 gives a 0.2 step: the atom crosses the y hi boundary (10) during
    // step 1 (9.9 -> 10.1); the remap step wraps it and subtracts
    // xydot = L0*erate = 10 from vx. With vy = 0.2 (0.02 per step) it would not
    // cross in 4 steps.
    // Measured with native LAMMPS (black box, this script with vy = 2.0):
    // id 1 x = 2, y = 0.7, vx = -10, vy = 2, ix = 0, iy = 1.
    const a = dumpAtoms(files, 'd.dump').get(1)!;
    expect(a.iy).toBe(1);
    expect(a.vx).toBeCloseTo(-10.0, 8);
    expect(a.x).toBeCloseTo(2.0, 8);
  });
});

describe('compute temp/deform', () => {
  it('removes the streaming velocity of the deforming box', async () => {
    const { thermo } = await runScript(`
      ${box(true)}
      create_atoms 1 single 5.0 2.5 5.0
      create_atoms 1 single 5.0 7.5 5.0
      mass 1 1.0
      variable vxA atom 0.5*y/10.0
      velocity all set v_vxA 0.0 0.0 units box
      fix 1 all nve
      fix 2 all deform 1 xy erate 0.05 remap v
      compute td all temp/deform
      thermo_style custom step temp c_td
      thermo 1
      run 0
    `);
    // both atoms stream exactly with the box (vx = xydot*lamda_y), so the
    // thermal temperature is 0 while the plain temperature is not
    expect(thermoOf(thermo, 0).c_td).toBeCloseTo(0, 8);
    expect(thermoOf(thermo, 0).temp).toBeGreaterThan(0.04);
  });

  it('needs a fix deform and takes no arguments', async () => {
    const noFix = await runScript(`
      ${box()}
      create_atoms 1 single 5.0 5.0 5.0
      mass 1 1.0
      compute td all temp/deform
      run 0
    `);
    expect(errText(noFix.error)).toContain('fix deform');
    const args = await runScript(`
      ${box()}
      create_atoms 1 single 5.0 5.0 5.0
      mass 1 1.0
      fix 1 all nve
      fix 2 all deform 1 x erate 0.1 units box
      compute td all temp/deform extra 1
      run 0
    `);
    expect(errText(args.error)).toContain('no arguments');
  });
});

describe('fix deform argument errors', () => {
  it('rejects a non-positive interval, unknown parameters and bad values', async () => {
    for (const cmd of [
      'fix 2 all deform 0 x erate 0.1 units box',
      'fix 2 all deform -1 x erate 0.1 units box',
      'fix 2 all deform 1 q erate 0.1 units box',
      'fix 2 all deform 1 x erate units box',
      'fix 2 all deform 1 x erate abc units box',
      'fix 2 all deform 1 x wiggle 0.1 units box',
      'fix 2 all deform 1 x wiggle 0.1 -1.0 units box',
      'fix 2 all deform 1 x final 1.0 units box',
      'fix 2 all deform 1',
      'fix 2 all deform 1 x erate 0.1 remap fast units box',
      'fix 2 all deform 1 x erate 0.1 units lattice box',
      'fix 2 all deform 1 x erate 0.1 bogus yes units box',
    ]) {
      const { error } = await runScript(`${box()}\ntimestep 0.001\n${cmd}\n`);
      expect(error, cmd).not.toBeNull();
    }
  });

  it('rejects tilt-only misuse and fix deform/pressure-only styles and keywords', async () => {
    for (const cmd of [
      'fix 2 all deform 1 xy scale 1.1 units box',
      'fix 2 all deform 1 xy volume units box',
      'fix 2 all deform 1 x pressure 1.0 units box',
      'fix 2 all deform 1 x pressure/mean 1.0 units box',
      'fix 2 all deform 1 x erate 0.1 couple xyz units box',
      'fix 2 all deform 1 x erate 0.1 max/rate 1.0 units box',
    ]) {
      const { error } = await runScript(`${box(true)}\ntimestep 0.001\n${cmd}\n`);
      expect(error, cmd).not.toBeNull();
    }
    const pressure = await runScript(`${box()}\nfix 2 all deform 1 x pressure 1.0 units box\n`);
    expect(errText(pressure.error)).toContain('deform/pressure');
  });

  it('rejects a second deformation of the same parameter', async () => {
    const { error } = await runScript(`${box()}\nfix 2 all deform 1 x erate 0.1 x vel 1.0 units box\n`);
    expect(errText(error)).toContain('twice');
  });
});
