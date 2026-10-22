import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { argBounds, select2 } from '../src/engine-gpl/xcompute/common';

/*
 * Computes ported from LAMMPS (commit b332f28) in src/engine-gpl/xcompute: ackland/atom, ave/sphere/atom, composition/atom,
 * efield/wolf/atom, pressure/cylinder (the stress/cylinder class under its second name), rattlers/atom, temp/rotate and ti
 * (src/EXTRA-COMPUTE). Native parity is in tests/oracle/w227xc_*.in (engineOracle.test.ts); this file checks argument
 * errors, the refusals, and closed-form values of the computes.
 */

const SETUP = `units lj
atom_style atomic
lattice fcc 1.0
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
`;
const PAIR = 'pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\n';

/** Runs a script; returns the error messages and the thermo rows (each a record of the printed keywords). */
const run = async (script: string): Promise<{ errors: string[]; rows: Record<string, number>[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  try {
    await session.execute(script);
  } catch (e) {
    events.push({ kind: 'error', message: String(e instanceof Error ? e.message : e) } as unknown as EngineEvent);
  }
  return {
    errors: events.filter((e) => e.kind === 'error').map((e) => ('message' in e ? String(e.message) : '')),
    rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as Record<string, number>),
  };
};

describe('select2 and the bounds of compute ti (ported helpers)', () => {
  it('moves the k smallest values to the front and permutes the index array with them', () => {
    const vals = [5.5, 1.25, 9.0, 0.5, 3.75, 2.0, 7.0, 0.75];
    const idx = vals.map((_, i) => 100 + i);
    const a = Float64Array.from(vals), ia = Int32Array.from(idx);
    select2(3, vals.length, a, ia);
    expect([...a.subarray(0, 3)].sort((x, y) => x - y)).toEqual([0.5, 0.75, 1.25]);
    for (let k = 0; k < vals.length; k++) expect(vals[ia[k] - 100]).toBe(a[k]);
    expect([...a].sort((x, y) => x - y)).toEqual([...vals].sort((x, y) => x - y));
  });

  it('reads the type ranges of compute ti as utils::bounds does', () => {
    expect(argBounds('2', 3, 'compute ti')).toEqual([2, 2]);
    expect(argBounds('*', 3, 'compute ti')).toEqual([1, 3]);
    expect(argBounds('*2', 3, 'compute ti')).toEqual([1, 2]);
    expect(argBounds('2*', 3, 'compute ti')).toEqual([2, 3]);
    expect(argBounds('1*2', 3, 'compute ti')).toEqual([1, 2]);
    expect(() => argBounds('4', 3, 'compute ti')).toThrow(/out of bounds/);
  });
});

describe('argument errors (native texts)', () => {
  it('ackland/atom takes at most a legacy keyword', async () => {
    expect((await run(`${SETUP}${PAIR}compute a all ackland/atom foo\nrun 0\n`)).errors[0]).toMatch(/Illegal compute ackland\/atom command/);
    expect((await run(`${SETUP}${PAIR}compute a all ackland/atom legacy\nrun 0\n`)).errors[0]).toMatch(/Invalid compute ackland\/atom command/);
  });

  it('ave/sphere/atom needs a cutoff or a pair style, and a positive cutoff', async () => {
    expect((await run(`${SETUP}compute a all ave/sphere/atom\nrun 0\n`)).errors[0]).toMatch(/requires a cutoff be specified or a pair style be defined/);
    expect((await run(`${SETUP}${PAIR}compute a all ave/sphere/atom cutoff 0\nrun 0\n`)).errors[0]).toMatch(/Illegal compute ave\/sphere\/atom command/);
  });

  it('composition/atom names an unknown keyword', async () => {
    expect((await run(`${SETUP}${PAIR}compute a all composition/atom bogus 1\nrun 0\n`)).errors[0]).toMatch(/Illegal compute composition\/atom command/);
  });

  it('efield/wolf/atom needs charges, a positive alpha and an existing group', async () => {
    expect((await run(`${SETUP}${PAIR}compute e all efield/wolf/atom 0.2\nrun 0\n`)).errors[0]).toMatch(/requires atom attribute q/);
    expect((await run(`${SETUP}${PAIR}compute e all efield/wolf/atom 0\nrun 0\n`)).errors[0]).toMatch(/alpha value 0 is invalid/);
    expect((await run(`${SETUP}${PAIR}compute e all efield/wolf/atom 0.2 group nope\nrun 0\n`)).errors[0]).toMatch(/group nope does not exist/);
  });

  it('rattlers/atom needs its three arguments and the radius attribute for the radius style', async () => {
    expect((await run(`${SETUP}${PAIR}compute r all rattlers/atom size 3 10\nrun 0\n`)).errors[0]).toMatch(/Illegal compute rattlers\/atom command/);
    expect((await run(`${SETUP}${PAIR}compute r all rattlers/atom radius 3 10\nrun 0\n`)).errors[0]).toMatch(/radius style requires atom attribute radius/);
  });

  it('temp/rotate takes no arguments', async () => {
    expect((await run(`${SETUP}${PAIR}compute t all temp/rotate 1\nrun 0\n`)).errors[0]).toMatch(/Illegal compute temp\/rotate command/);
  });

  it('ti takes quadruplets of keyword, types and two v_ variables', async () => {
    expect((await run(`${SETUP}${PAIR}compute t all ti lj/cut 1 v_a\nrun 0\n`)).errors[0]).toMatch(/Illegal compute ti command/);
    expect((await run(`${SETUP}${PAIR}compute t all ti lj/cut 1 a b\nrun 0\n`)).errors[0]).toMatch(/Illegal compute ti command/);
    expect((await run(`${SETUP}${PAIR}compute t all ti lj/cut 1 v_nope v_nope\nrun 0\n`)).errors[0]).toMatch(/Variable name for compute ti does not exist/);
    expect((await run(`${SETUP}${PAIR}variable a equal 1.0\nvariable b atom 1.0\ncompute t all ti lj/cut 1 v_a v_b\nrun 0\n`)).errors[0]).toMatch(/Variable for compute ti is invalid style/);
    expect((await run(`${SETUP}${PAIR}variable a equal 1.0\nvariable b equal 1.0\ncompute t all ti coul/cut 1 v_a v_b\nrun 0\n`)).errors[0]).toMatch(/Compute ti pair style does not exist/);
    expect((await run(`${SETUP}${PAIR}variable a equal 1.0\nvariable b equal 1.0\ncompute t all ti kspace 1 v_a v_b\nrun 0\n`)).errors[0]).toMatch(/Compute ti kspace style does not exist/);
    expect((await run(`${SETUP}${PAIR}variable a equal 1.0\nvariable b equal 1.0\ncompute t all ti tail 1 v_a v_b\nrun 0\n`)).errors[0]).toMatch(/Compute ti tail when pair style does not compute tail corrections/);
    // Measured with native LAMMPS: without charges the kspace style's init (force->init) stops first, with
    // Kspace style requires atom attribute q
    expect((await run(`${SETUP}${PAIR}kspace_style ewald 1.0e-4\nvariable a equal 1.0\nvariable b equal 1.0\ncompute t all ti kspace 1 v_a v_b\nrun 0\n`)).errors[0]).toMatch(/Kspace style requires atom attribute q/);
  });
});

describe('closed-form values', () => {
  it('ackland/atom: a perfect fcc crystal is fcc (2), a perfect bcc crystal is bcc (1)', async () => {
    const fcc = await run(`${SETUP}${PAIR}compute a all ackland/atom\ncompute s all reduce sum c_a\nthermo_style custom step c_s\nthermo_modify norm no\nrun 0\n`);
    expect(fcc.errors).toEqual([]);
    expect(fcc.rows[0].c_s).toBe(2 * 32);
    const bcc = await run(`units lj
atom_style atomic
lattice bcc 1.0
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
${PAIR}compute a all ackland/atom\ncompute s all reduce sum c_a\nthermo_style custom step c_s\nthermo_modify norm no\nrun 0\n`);
    expect(bcc.errors).toEqual([]);
    expect(bcc.rows[0].c_s).toBe(16);
  });

  it('composition/atom: in a single-type fcc crystal the cutoff sphere holds 13 atoms, all of type 1', async () => {
    const r = await run(`${SETUP}${PAIR}compute c all composition/atom cutoff 1.2\ncompute s all reduce sum c_c[1] c_c[2]\nthermo_style custom step c_s[1] c_s[2]\nthermo_modify norm no\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(r.rows[0]['c_s[1]']).toBe(13 * 32);
    expect(r.rows[0]['c_s[2]']).toBeCloseTo(32, 10);
  });

  it('ave/sphere/atom: at rest, the density is the mass in the sphere over its volume (mv2d = 1 in lj)', async () => {
    const r = await run(`${SETUP}${PAIR}compute a all ave/sphere/atom cutoff 1.2\ncompute s all reduce sum c_a[1] c_a[2]\nthermo_style custom step c_s[1] c_s[2]\nthermo_modify norm no\nrun 0\n`);
    expect(r.errors).toEqual([]);
    const volume = (4.0 / 3.0) * Math.PI * 1.2 ** 3;
    expect(r.rows[0]['c_s[1]']).toBeCloseTo(32 * (13 / volume), 9);
    expect(r.rows[0]['c_s[2]']).toBe(0);
  });

  it('efield/wolf/atom: two like charges push each other apart (the field at atom 1 points away from atom 2)', async () => {
    const r = await run(`units lj
atom_style charge
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5.0 5.0 5.0 units box
create_atoms 1 single 6.5 5.0 5.0 units box
mass 1 1.0
set atom 1 charge 1.0
set atom 2 charge 1.0
pair_style coul/cut 4.0
pair_coeff * *
compute e all efield/wolf/atom 0.25
compute lo all reduce min c_e[1]
compute hi all reduce max c_e[1]
compute ys all reduce max c_e[2]
thermo_style custom step c_lo c_hi c_ys
thermo_modify norm no
run 0
`);
    expect(r.errors).toEqual([]);
    // atom 1 (at x = 5) feels a field along -x, atom 2 (at x = 6.5) along +x; the y components vanish
    expect(r.rows[0].c_lo).toBeLessThan(-0.1);
    expect(r.rows[0].c_hi).toBeGreaterThan(0.1);
    expect(Math.abs(r.rows[0]['c_ys'])).toBeLessThan(1e-12);
  });

  it('rattlers/atom: with a threshold above every contact count all atoms are rattlers; with 0 none is', async () => {
    const all = await run(`${SETUP}${PAIR}compute r all rattlers/atom type 1000 10\ncompute s all reduce sum c_r\nthermo_style custom step c_s\nthermo_modify norm no\nrun 0\n`);
    expect(all.errors).toEqual([]);
    expect(all.rows[0].c_s).toBe(32);
    const none = await run(`${SETUP}${PAIR}compute r all rattlers/atom type 0 10\ncompute s all reduce sum c_r\nthermo_style custom step c_s\nthermo_modify norm no\nrun 0\n`);
    expect(none.rows[0].c_s).toBe(0);
  });

  it('temp/rotate: a pure rigid rotation of a symmetric cluster has no thermal temperature', async () => {
    const r = await run(`units lj
atom_style atomic
boundary f f f
lattice fcc 1.0
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
variable cx equal 2*xlat/2
variable vx atom -0.2*(y-1.5*xlat/1.0*1.0)
variable vy atom 0.2*(x-1.5*xlat/1.0*1.0)
velocity all set v_vx v_vy 0.0 units box
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
compute rot all temp/rotate
compute t all temp
thermo_style custom step c_t c_rot
run 0
`);
    expect(r.errors).toEqual([]);
    expect(r.rows[0].c_t).toBeGreaterThan(1e-3);
    expect(Math.abs(r.rows[0].c_rot)).toBeLessThan(1e-12);
  });

  it('ti: with f = 1 and df = 1 the derivative is the pair energy itself', async () => {
    const r = await run(`${SETUP}${PAIR}variable f equal 1.0\nvariable df equal 1.0\ncompute ti all ti lj/cut 1 v_f v_df\nthermo_style custom step pe c_ti\nthermo_modify format float %.15g\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(Math.abs(r.rows[0].c_ti - r.rows[0].pe)).toBeLessThan(1e-12 * Math.max(1, Math.abs(r.rows[0].pe)));
  });

  it('ti: a zero derivative gives a zero dU/dlambda, and a zero scale gives no term', async () => {
    const r = await run(`${SETUP}${PAIR}variable f equal 0.5\nvariable df equal 0.0\nvariable g equal 0.0\nvariable dg equal 1.0\ncompute ti all ti lj/cut 1 v_f v_df lj/cut 1 v_g v_dg\nthermo_style custom step c_ti\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(r.rows[0].c_ti).toBe(0);
  });
});

describe('registration', () => {
  it('pressure/cylinder is the stress/cylinder class: its argument errors use that class', async () => {
    const r = await run(`${SETUP}${PAIR}compute p all pressure/cylinder 1 2\nrun 0\n`);
    expect(r.errors[0]).toMatch(/stress\/cylinder/);
  });
});
