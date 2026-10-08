import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Direct unit checks for the wave-18 fix styles (addtorque, drag, oneway):
 * the documented formulas on tiny systems, the global outputs, and argument
 * errors. Oracle parity with native LAMMPS for these styles lives in
 * tests/engineOracle.test.ts (w18fix_* cases).
 */

interface Run {
  thermo: ThermoRow[];
  files: Map<string, string>;
  error: EngineError | null;
}

const runScript = async (text: string): Promise<Run> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { thermo, files, error };
};

/** Per-atom columns of a write_dump custom file, keyed by atom id. */
const dumpAtoms = (r: Run, name: string): Map<number, Record<string, number>> => {
  const text = r.files.get(name) ?? '';
  const lines = text.trim().split('\n');
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

const errText = (r: Run): string => {
  expect(r.error, 'expected the script to fail').not.toBeNull();
  return r.error?.message ?? '';
};

const DUMP = 'write_dump all custom ext.dump id x fx fy fz vx vy vz modify format float %.17g sort id';

describe('fix addtorque (docs.lammps.org/fix_addtorque.html)', () => {
  it('adds m*(alpha x r) with alpha = I^-1 T (collinear group, singular axis)', async () => {
    // two atoms on x: I = diag(0,2,2), so T = +z gives alpha_z = 0.5 and
    // F = m alpha x r = (0, +/-0.5, 0)
    const r = await runScript(`
units lj
atom_style atomic
boundary f f f
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single 1.0 0.0 0.0
create_atoms 1 single -1.0 0.0 0.0
mass 1 1.0
pair_style zero 5.0
pair_coeff * *
fix t all addtorque 0.0 0.0 1.0
thermo 1
run 0
${DUMP}
`);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump');
    expect(a.get(1)!.fx).toBeCloseTo(0, 15);
    expect(a.get(1)!.fy).toBeCloseTo(0.5, 15);
    expect(a.get(2)!.fy).toBeCloseTo(-0.5, 15);
  });

  it('applies the rigid-body centripetal force and reports the energy sum m|omega x r|^2', async () => {
    // square in xy at radius 1, rigid rotation omega = 0.1 about z:
    // U = sum m |omega x r|^2 = 4*0.01 = 0.04, F_i = -m omega^2 r_i
    const r = await runScript(`
units lj
atom_style atomic
boundary f f f
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single 1.0 0.0 0.0
create_atoms 1 single -1.0 0.0 0.0
create_atoms 1 single 0.0 1.0 0.0
create_atoms 1 single 0.0 -1.0 0.0
mass 1 1.0
velocity all set 0.0 0.0 0.0 units box
set atom 1 vx 0.0 vy 0.1
set atom 2 vx 0.0 vy -0.1
set atom 3 vx -0.1 vy 0.0
set atom 4 vx 0.1 vy 0.0
pair_style zero 5.0
pair_coeff * *
fix t all addtorque 0.0 0.0 0.0
fix_modify t energy yes
thermo_style custom step pe f_t
thermo_modify format float %.15g norm no
thermo 1
run 0
${DUMP}
`);
    expect(r.error).toBeNull();
    expect(r.thermo[0].pe).toBeCloseTo(0.04, 12);
    expect(r.thermo[0].f_t).toBeCloseTo(0.04, 12);
    const a = dumpAtoms(r, 'ext.dump');
    expect(a.get(1)!.fx).toBeCloseTo(-0.01, 15);
    expect(a.get(3)!.fy).toBeCloseTo(-0.01, 15);
  });

  it('the energy keyword is off by default; fix_modify energy yes adds it to pe', async () => {
    const base = `
units lj
atom_style atomic
boundary f f f
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single 1.0 0.0 0.0
create_atoms 1 single -1.0 0.0 0.0
create_atoms 1 single 0.0 1.0 0.0
create_atoms 1 single 0.0 -1.0 0.0
mass 1 1.0
velocity all set 0.0 0.0 0.0 units box
set atom 1 vx 0.0 vy 0.1
set atom 2 vx 0.0 vy -0.1
set atom 3 vx -0.1 vy 0.0
set atom 4 vx 0.1 vy 0.0
pair_style zero 5.0
pair_coeff * *
fix t all addtorque 0.0 0.0 0.0
thermo_style custom step pe f_t
thermo_modify format float %.15g norm no
thermo 1
run 0
`;
    const off = await runScript(base);
    expect(off.error).toBeNull();
    expect(off.thermo[0].pe).toBeCloseTo(0, 15); // energy no: not in pe
    expect(off.thermo[0].f_t).toBeCloseTo(4 * 0.01, 12); // but the scalar reports it
    const on = await runScript(base.replace('run 0', 'fix_modify t energy yes\nrun 0'));
    expect(on.thermo[0].pe).toBeCloseTo(4 * 0.01, 12);
  });

  it('accepts an equal-style variable torque', async () => {
    const r = await runScript(`
units lj
atom_style atomic
boundary f f f
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single 1.0 0.0 0.0
create_atoms 1 single -1.0 0.0 0.0
mass 1 1.0
variable tq equal 2.0
pair_style zero 5.0
pair_coeff * *
fix t all addtorque 0.0 0.0 v_tq
thermo 1
run 0
${DUMP}
`);
    expect(r.error).toBeNull();
    expect(dumpAtoms(r, 'ext.dump').get(1)!.fy).toBeCloseTo(1.0, 14);
  });

  it('rejects bad arguments', async () => {
    expect(errText(await runScript(`units lj
atom_style atomic
boundary f f f
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single 1.0 0.0 0.0
mass 1 1.0
fix t all addtorque 0.0 0.0
`))).toContain('addtorque Tx Ty Tz');
    expect(errText(await runScript(`units lj
atom_style atomic
boundary f f f
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single 1.0 0.0 0.0
mass 1 1.0
fix t all addtorque 0.0 0.0 v_missing
thermo 1
run 0
`))).toContain('missing');
  });
});

describe('fix drag (docs.lammps.org/fix_drag.html)', () => {
  const box = `
units lj
atom_style atomic
boundary f f f
region box block -20 20 -20 20 -20 20
create_box 1 box
create_atoms 1 single 3.0 4.0 0.0
mass 1 1.0
pair_style zero 5.0
pair_coeff * *
`;

  it('pulls each atom toward the target with magnitude fmag', async () => {
    const r = await runScript(box + `fix d all drag 0.0 0.0 0.0 2.0 1.0
thermo_style custom step f_d[1] f_d[2] f_d[3]
thermo_modify format float %.15g norm no
thermo 1
run 0
${DUMP}
`);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBeCloseTo(-1.2, 14);
    expect(a.fy).toBeCloseTo(-1.6, 14);
    // the vector is "the total force on the group of atoms by the drag force"
    expect(r.thermo[0]['f_d[1]']).toBeCloseTo(-1.2, 14);
    expect(r.thermo[0]['f_d[2]']).toBeCloseTo(-1.6, 14);
    expect(r.thermo[0]['f_d[3]']).toBe(0);
  });

  it('does not apply the force at or inside distance delta', async () => {
    const r = await runScript(`
units lj
atom_style atomic
boundary f f f
region box block -20 20 -20 20 -20 20
create_box 1 box
create_atoms 1 single 0.0 1.0 0.0
create_atoms 1 single 0.0 1.0000001 0.0
mass 1 1.0
pair_style zero 5.0
pair_coeff * *
fix d all drag NULL 0.0 NULL 2.0 1.0
thermo 1
run 0
${DUMP}
`);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump');
    expect(a.get(1)!.fy).toBe(0); // exactly at delta: no force
    expect(a.get(2)!.fy).toBeCloseTo(-2.0, 6); // just outside
    expect(a.get(2)!.fx).toBe(0); // NULL dimension untouched
  });

  it('rejects bad arguments', async () => {
    expect(errText(await runScript(box + 'fix d all drag 0 0 0 1\n'))).toContain('x y z fmag delta');
    expect(errText(await runScript(box + 'fix d all drag v_x 0 0 1 1\n'))).toContain('number or NULL');
    expect(errText(await runScript(box + 'fix d all drag 0 0 0 abc 1\n'))).toContain('fmag');
  });
});

describe('fix oneway (docs.lammps.org/fix_oneway.html)', () => {
  const box = (fix: string) => `
units lj
atom_style atomic
boundary f f f
region box block -10 10 -10 10 -10 10
region all block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0
mass 1 1.0
velocity all set -0.1 0.0 0.0 units box
pair_style zero 5.0
pair_coeff * *
fix o ${fix}
fix n all nve
thermo 1
run 1
${DUMP}
`;

  it('reverses the wrong-sign component after the step (end_of_step)', async () => {
    const r = await runScript(box('all oneway 1 all x'));
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.vx).toBeCloseTo(0.1, 15);
    expect(a.x).toBeCloseTo(-0.0005, 15); // drift used the pre-reversal velocity
  });

  it('applies every N steps only', async () => {
    const r = await runScript(box('all oneway 2 all x'));
    expect(r.error).toBeNull();
    // step 1 is not a multiple of 2: no reversal yet
    expect(dumpAtoms(r, 'ext.dump').get(1)!.vx).toBeCloseTo(-0.1, 15);
    const r2 = await runScript(box('all oneway 2 all x').replace('run 1', 'run 2'));
    expect(dumpAtoms(r2, 'ext.dump').get(1)!.vx).toBeCloseTo(0.1, 15);
  });

  it('direction -x reverses a positive velocity', async () => {
    const r = await runScript(box('all oneway 1 all -x').replace('velocity all set -0.1 0.0 0.0 units box', 'velocity all set 0.1 0.0 0.0 units box'));
    expect(r.error).toBeNull();
    expect(dumpAtoms(r, 'ext.dump').get(1)!.vx).toBeCloseTo(-0.1, 15);
  });

  it('leaves atoms outside the region untouched', async () => {
    const r = await runScript(`
units lj
atom_style atomic
boundary f f f
region box block -10 10 -10 10 -10 10
region right block 0 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single -2.0 0.0 0.0
create_atoms 1 single 0.5 0.0 0.0
mass 1 1.0
velocity all set -0.1 0.0 0.0 units box
pair_style zero 5.0
pair_coeff * *
fix o all oneway 1 right x
fix n all nve
thermo 1
run 1
${DUMP}
`);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump');
    expect(a.get(1)!.vx).toBeCloseTo(-0.1, 15); // outside right region
    expect(a.get(2)!.vx).toBeCloseTo(0.1, 15); // inside, reversed
  });

  it('rejects bad arguments', async () => {
    expect(errText(await runScript(box('all oneway 0 all x')))).toContain('positive integer');
    expect(errText(await runScript(box('all oneway 1 nosuch x')))).toContain('nosuch');
    expect(errText(await runScript(box('all oneway 1 all q')))).toContain('direction');
  });
});
