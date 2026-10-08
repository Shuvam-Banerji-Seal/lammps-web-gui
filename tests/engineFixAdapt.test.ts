import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix adapt (docs.lammps.org/fix_adapt.html): argument errors, and that the
 * energy follows an adapted pair parameter at fixed positions (run 0 keeps the
 * coordinates). Oracle parity with native LAMMPS is in tests/oracle/w9adapt_*.in.
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
  return { thermo, error, files, message: error instanceof Error ? error.message : String(error ?? '') };
};

const SYS1 = `
units           lj
atom_style      atomic
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
thermo_style    custom step evdwl pe
thermo_modify   format float %.15g
`;

const SYS2 = SYS1.replace('create_box      1 box', 'create_box      2 box')
  .replace('mass            1 1.0', 'mass * 1.0\nset group all type 1\ngroup odd id 1:8:2\nset group odd type 2');

describe('fix adapt: energy follows the adapted parameter', () => {
  it('lj/cut epsilon doubled doubles evdwl at fixed positions', async () => {
    const r = await runScript(`${SYS1}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut epsilon 1 1 v_e
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo.length).toBe(2);
    const e1 = r.thermo[0].evdwl, e2 = r.thermo[1].evdwl;
    expect(Math.abs(e1)).toBeGreaterThan(0);
    expect(e2).toBeCloseTo(2 * e1, 12);
  });

  it('lj/cut sigma changes the energy (sigma 0.9 vs 1.0 at the same positions)', async () => {
    const r = await runScript(`${SYS1}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
variable s equal 0.9
fix 1 all adapt 0 pair lj/cut sigma 1 1 v_s
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].evdwl).not.toBeCloseTo(r.thermo[0].evdwl, 6);
  });

  it('soft a doubled doubles the energy (soft is linear in A)', async () => {
    const r = await runScript(`${SYS1}
pair_style soft 2.5
pair_coeff * * 10.0
run 0
variable a equal 20.0
fix 1 all adapt 0 pair soft a * * v_a
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].evdwl).toBeCloseTo(2 * r.thermo[0].evdwl, 10);
  });

  it('morse D0 doubled doubles the energy', async () => {
    const r = await runScript(`${SYS1}
pair_style morse 2.5
pair_coeff * * 1.0 1.5 1.0
run 0
variable d equal 2.0
fix 1 all adapt 0 pair morse D0 * * v_d
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].evdwl).toBeCloseTo(2 * r.thermo[0].evdwl, 10);
  });

  it('scale yes multiplies the value the run started with', async () => {
    const r = await runScript(`${SYS1}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
variable e equal 3.0
fix 1 all adapt 0 pair lj/cut epsilon 1 1 v_e scale yes
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].evdwl).toBeCloseTo(3 * r.thermo[0].evdwl, 12);
  });

  it('reset yes restores the original value at the end of the run; reset no keeps it', async () => {
    const body = (reset: string) => `${SYS1}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut epsilon 1 1 v_e reset ${reset}
run 0
unfix 1
run 0
`;
    const yes = await runScript(body('yes'));
    const no = await runScript(body('no'));
    expect(yes.error).toBeNull();
    expect(no.error).toBeNull();
    const e1 = yes.thermo[0].evdwl;
    expect(yes.thermo[2].evdwl).toBeCloseTo(e1, 12);
    expect(no.thermo[2].evdwl).toBeCloseTo(2 * e1, 12);
  });

  it('a mixed (mixing-derived) epsilon(1,2) follows the diagonal epsilon(1,1)', async () => {
    // pair_coeff 1 1 2.0 with 1 2 mixed (geometric) equals the adapted run
    const adapted = await runScript(`${SYS2}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut epsilon 1 1 v_e
run 0
`);
    const direct = await runScript(`${SYS2}
pair_style lj/cut 2.5
pair_coeff 1 1 2.0 1.0
pair_coeff 2 2 0.5 1.1
run 0
`);
    expect(adapted.error).toBeNull();
    expect(direct.error).toBeNull();
    expect(adapted.thermo[0].evdwl).toBeCloseTo(direct.thermo[0].evdwl, 12);
  });

  it('atom charge scaled by a variable changes ecoul as q squared', async () => {
    const r = await runScript(`
units           lj
atom_style      charge
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
set             group all charge 1.0
pair_style      coul/cut 2.5
pair_coeff      * *
thermo_style    custom step ecoul pe
thermo_modify   format float %.15g
run 0
variable q equal 2.0
fix 1 all adapt 0 atom charge v_q
run 0
`);
    expect(r.error).toBeNull();
    expect(r.thermo[1].ecoul).toBeCloseTo(4 * r.thermo[0].ecoul, 10);
  });

  it('atom diameter with mass yes: mass scales with d^3 (sphere style, dynamic radii)', async () => {
    const r = await runScript(`
units           lj
atom_style      sphere 1
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
set             group all diameter 1.0 density 1.0
variable d equal 2.0
fix 1 all adapt 0 atom diameter v_d
run 0
write_dump all custom dmp.txt id radius mass modify format float %.12g sort id
`);
    expect(r.error).toBeNull();
    const lines = (r.files.get('dmp.txt') ?? '').trim().split('\n');
    const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    const rows = lines.slice(k + 1).map((l) => l.trim().split(/\s+/).map(Number));
    expect(rows.length).toBe(8);
    for (const [, radius, mass] of rows) {
      expect(radius).toBeCloseTo(1.0, 10);
      // density 1.0 (per-atom) and volume 4/3 pi r^3: mass = 4/3 pi (d/2)^3 at d = 2
      expect(mass).toBeCloseTo((4 / 3) * Math.PI, 8);
    }
  });

  it('atom diameter with mass no keeps the mass', async () => {
    const r = await runScript(`
units           lj
atom_style      sphere 1
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
set             group all diameter 1.0 density 1.0
variable d equal 2.0
fix 1 all adapt 0 atom diameter v_d mass no
run 0
write_dump all custom dmp.txt id radius mass modify format float %.12g sort id
`);
    expect(r.error).toBeNull();
    const lines = (r.files.get('dmp.txt') ?? '').trim().split('\n');
    const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    const first = lines[k + 1].trim().split(/\s+/).map(Number);
    expect(first[1]).toBeCloseTo(1.0, 10);
    expect(first[2]).toBeCloseTo((4 / 3) * Math.PI * 0.125, 8);
  });
});

describe('fix adapt: argument and option errors', () => {
  const pairSetup = `${SYS2}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
variable e equal 2.0
`;

  it('N must be a non-negative integer', async () => {
    const r = await runScript(`${SYS1}\nvariable e equal 2.0\nfix 1 all adapt -1 pair lj/cut epsilon 1 1 v_e\n`);
    expect(r.message).toMatch(/N must be a non-negative integer/);
  });

  it('at least one attribute is required', async () => {
    const r = await runScript(`${SYS1}\nfix 1 all adapt 1 scale no\n`);
    expect(r.message).toMatch(/no attribute given/);
  });

  it('unknown attribute or keyword is named', async () => {
    const r = await runScript(`${SYS1}\nfix 1 all adapt 1 frob v_x\n`);
    expect(r.message).toMatch(/unknown attribute or keyword 'frob'/);
  });

  it.each([
    ['bond', 'bond harmonic k 1 v_e'],
    ['angle', 'angle harmonic k 1 v_e'],
    ['dihedral', 'dihedral quadratic k 1 v_e'],
    ['improper', 'improper harmonic k 1 v_e'],
    ['kspace', 'kspace v_e'],
  ])('%s attribute is an unsupported option (StyleError naming it)', async (name, attr) => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 ${attr}\n`);
    expect(r.message).toMatch(new RegExp(`attribute ${name} is not supported`));
  });

  it('the variable must exist', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 v_nope\n`);
    expect(r.message).toMatch(/variable nope does not exist/);
  });

  it('the value must be v_name', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 2.0\n`);
    expect(r.message).toMatch(/expected v_name/);
  });

  it('atom-style variables are rejected (equal-style only)', async () => {
    const r = await runScript(`${pairSetup}\nvariable av atom x/10\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 v_av\n`);
    expect(r.message).toMatch(/must be equal-style/);
  });

  it('cutoff parameters cannot be adapted (the neighbor list would not follow)', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut cut 1 1 v_e\n`);
    expect(r.message).toMatch(/parameter 'cut' cannot be adapted/);
  });

  it('a pair style that is not supported is named', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair eam epsilon 1 1 v_e\n`);
    expect(r.message).toMatch(/pair style 'eam' is not supported by fix adapt/);
  });

  it('hybrid sub-style selection (style:N) is refused', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut:1 epsilon 1 1 v_e\n`);
    expect(r.message).toMatch(/requires hybrid/);
  });

  it('the pair style must be the defined one', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair soft a 1 1 v_e\nrun 0\n`);
    expect(r.message).toMatch(/is not the defined pair style/);
  });

  it('a type pair set only by mixing is refused (mixed values cannot be adapted)', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut epsilon 1 2 v_e\nrun 0\n`);
    expect(r.message).toMatch(/was not set by pair_coeff/);
  });

  it('atom diameter needs atom_style sphere', async () => {
    const r = await runScript(`${SYS1}\nvariable d equal 2.0\nfix 1 all adapt 0 atom diameter v_d\nrun 0\n`);
    expect(r.message).toMatch(/atom diameter needs atom_style sphere/);
  });

  it('atom charge needs atom_style charge or full', async () => {
    const r = await runScript(`${SYS1}\nvariable q equal 2.0\nfix 1 all adapt 0 atom charge v_q\nrun 0\n`);
    expect(r.message).toMatch(/atom charge needs atom_style charge or full/);
  });

  it('diameter/disc is refused', async () => {
    const r = await runScript(`${SYS1}\nvariable d equal 2.0\nfix 1 all adapt 0 atom diameter/disc v_d\n`);
    expect(r.message).toMatch(/diameter\/disc is not supported/);
  });

  it('scale and reset take yes or no', async () => {
    const r = await runScript(`${pairSetup}\nfix 1 all adapt 1 pair lj/cut epsilon 1 1 v_e reset maybe\n`);
    expect(r.message).toMatch(/keyword reset must be yes or no/);
  });
});
