import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Checks of fix vector (docs.lammps.org/fix_vector.html), fix wall/region
 * (docs.lammps.org/fix_wall_region.html) and fix wall/reflect/stochastic
 * (docs.lammps.org/fix_wall_reflect_stochastic.html): argument errors, one
 * analytic value per fix, and statistics of the stochastic wall models. Exact
 * parity with native LAMMPS is in tests/oracle/w7misc_*.in.
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
  return { thermo, error, files };
};

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Atom rows of a dump file written with write_dump (sorted by id). */
const parseDump = (text: string) => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  return lines.slice(k + 1).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    const a: Record<string, number> = {};
    cols.forEach((c, i) => { a[c] = w[i]; });
    return a;
  });
};

const HEAD = `units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10 units box
create_box 1 box
mass 1 1.0
`;

describe('fix vector', () => {
  const base = `${HEAD}pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
create_atoms 1 single 5 5 5 units box
fix 1 all nve
variable t equal 2*step+1
`;

  it('stores the Nevery-sampled values: sum over the growing vector (analytic)', async () => {
    // samples at steps 0, 5, 10 hold 2*step+1 = 1, 11, 21
    const r = await runScript(`${base}fix 3 all vector 5 v_t
variable s equal sum(f_3)
variable a equal ave(f_3)
thermo_style custom step v_s v_a
thermo 5
run 10`);
    expect(r.error).toBeNull();
    expect(r.thermo.map((t) => [t.step, t['v_s'], t['v_a']])).toEqual([
      [0, 1, 1],
      [5, 12, 6],
      [10, 33, 11],
    ]);
  });

  it('direct thermo columns f_ID[I] and f_ID[I][J] read the stored entries (analytic)', async () => {
    const r = await runScript(`${base}variable u equal step
fix 3 all vector 5 v_t
fix 4 all vector 5 v_t v_u
thermo_style custom step f_3[1] f_4[1][1] f_4[1][2]
thermo 5
run 10`);
    expect(r.error).toBeNull();
    // only the first row of each vector is referenced: LAMMPS itself errors on a direct
    // f_ID[I] at step 0 (the vector is still empty in the thermo header of that run); this
    // engine samples at setup, so the step-0 row exists here too (a deviation, not tested)
    const at = (step: number) => r.thermo.find((t) => t.step === step);
    expect(at(5)?.['f_3[1]']).toBe(1);
    expect(at(10)?.['f_3[1]']).toBe(1);
    expect(at(10)?.['f_4[1][1]']).toBe(1);
    expect(at(10)?.['f_4[1][2]']).toBe(0);
  });

  it('a 2-input fix stores an array: rows (2*step+1, step) at steps 0, 5, 10 (analytic)', async () => {
    const r = await runScript(`${base}variable u equal step
fix 4 all vector 5 v_t v_u
variable c equal sum(f_4[2])
variable a equal ave(f_4[1])
thermo_style custom step v_c v_a
thermo 5
run 10`);
    expect(r.error).toBeNull();
    expect(r.thermo.map((t) => [t.step, t['v_c'], t['v_a']])).toEqual([
      [0, 0, 1],
      [5, 5, 6],
      [10, 15, 11],
    ]);
  });

  it('nmax keeps only the newest entries (ring)', async () => {
    const r = await runScript(`${base}fix 3 all vector 1 v_t nmax 2
variable s equal sum(f_3)
thermo_style custom step v_s
thermo 4
run 4`);
    expect(r.error).toBeNull();
    // steps 0..4 store 1, 3, 5, 7, 9; the last two are kept: 7 + 9
    expect(r.thermo[1]['v_s']).toBe(16);
  });

  it('referencing the fix at a step that is not a multiple of Nevery errors', async () => {
    const r = await runScript(`${base}fix 3 all vector 5 v_t
variable e equal f_3[1]
thermo_style custom step v_e
thermo 1
run 2`);
    expect(message(r.error)).toMatch(/compatible time/);
  });

  it('argument errors', async () => {
    const cases: [string, RegExp][] = [
      ['fix 3 all vector 0 v_t', /Nevery must be a positive integer/],
      ['fix 3 all vector 2', /usage: fix ID group-ID vector/],
      ['fix 3 all vector 2 v_t nmax 0', /nmax must be a positive integer/],
      ['fix 3 all vector 2 v_t nmax 2 v_t', /after nmax/],
      ['fix 3 all vector 2 v_missing', /variable missing for fix vector does not exist/],
      ['fix 3 all vector 2 c_missing', /compute/],
      ['fix 3 all vector 2 f_missing', /fix/],
      ['compute ke all ke\nfix 3 all vector 2 c_ke v_t', /cannot set output array intensive\/extensive/],
    ];
    for (const [line, re] of cases) {
      const r = await runScript(`${base}${line}\nrun 0`);
      expect(message(r.error), line).toMatch(re);
    }
  });
});

describe('fix wall/region', () => {
  // one atom in a sphere of radius 2.2 (side in), lj93 with epsilon 1, sigma 1, cutoff 2.5
  const lj93 = (r: number, eps = 1, sig = 1) => eps * ((2 / 15) * (sig / r) ** 9 - (sig / r) ** 3);
  const rc = 2.5;

  it('sphere lj93, side in: energy and the reaction force on the wall (analytic)', async () => {
    const p = [0.5, 0.3, 0.2];
    const R = 2.2;
    const r = R - Math.hypot(...p);
    const r2 = await runScript(`${HEAD}pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
create_atoms 1 single ${p.join(' ')} units box
region sph sphere 0 0 0 ${R} side in units box
fix 2 all wall/region sph lj93 1.0 1.0 ${rc}
fix_modify 2 energy yes
thermo_style custom step pe f_2 f_2[1] f_2[2] f_2[3]
thermo_modify format float %.15g
run 0`);
    expect(r2.error).toBeNull();
    const row = r2.thermo[0];
    expect(row['pe']).toBeCloseTo(lj93(r) - lj93(rc), 10);
    // the atom is pulled by -dE/dd along the outward radial direction (E is attractive at r = 1.58):
    // the wall gets the opposite force
    const h = 1e-6;
    const dEdd = (lj93(r + h) - lj93(r - h)) / (2 * h);
    const pn = Math.hypot(...p);
    for (let k = 0; k < 3; k++) {
      expect(row[`f_2[${k + 1}]`]).toBeCloseTo(-dEdd * p[k] / pn, 6);
    }
  });

  it('block harmonic at a corner: one contribution per face within the cutoff (analytic)', async () => {
    const r2 = await runScript(`${HEAD}pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
create_atoms 1 single 1.5 1.5 0.2 units box
region blk block -2 2 -2 2 -2 2 side in units box
fix 2 all wall/region blk harmonic 1.0 1.0 ${rc}
fix_modify 2 energy yes
thermo_style custom step pe f_2 f_2[1] f_2[2] f_2[3]
thermo_modify format float %.15g
run 0`);
    expect(r2.error).toBeNull();
    // faces at distance 0.5 (x), 0.5 (y), 1.8 and 2.2 (z): sum of (rc - d)^2
    const e = (rc - 0.5) ** 2 * 2 + (rc - 1.8) ** 2 + (rc - 2.2) ** 2;
    expect(r2.thermo[0]['pe']).toBeCloseTo(e, 10);
    expect(r2.thermo[0]['f_2[1]']).toBeCloseTo(2 * (rc - 0.5), 10);
    expect(r2.thermo[0]['f_2[3]']).toBeCloseTo(2 * (rc - 1.8) - 2 * (rc - 2.2), 10);
  });

  it('side out: the nearest point of the solid is the only contribution', async () => {
    const r2 = await runScript(`${HEAD}pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
create_atoms 1 single 2.5 2.5 0.2 units box
region blk block -2 2 -2 2 -2 2 side out units box
fix 2 all wall/region blk harmonic 1.0 1.0 ${rc}
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.15g
run 0`);
    expect(r2.error).toBeNull();
    expect(r2.thermo[0]['pe']).toBeCloseTo((rc - Math.SQRT1_2) ** 2, 10);
  });

  it('argument and region errors', async () => {
    const atom = `${HEAD}pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
create_atoms 1 single 0.5 0.3 0.2 units box
region sph sphere 0 0 0 2.2 side in units box
region sph_out sphere 0 0 0 2.2 side out units box
region uni union 2 sph sph_out units box
`;
    const cases: [string, RegExp][] = [
      ['fix 2 all wall/region sph foo 1 1 2.5', /unknown style 'foo'/],
      ['fix 2 all wall/region sph lj93 1 2.5', /takes 2 parameters/],
      ['fix 2 all wall/region sph lj93 1 1', /takes 2 parameters and a cutoff/],
      ['fix 2 all wall/region sph lj93 1 1 0', /cutoff must be a number > 0/],
      ['fix 2 all wall/region sph morse 1 1 2.5', /takes 3 parameters/],
      ['fix 2 all wall/region sph colloid 1 1 2.5', /colloid is not supported/],
      ['fix 2 all wall/region nowhere lj93 1 1 2.5', /region ID 'nowhere' does not exist/],
      ['fix 2 all wall/region uni lj93 1 1 2.5', /region style union is not supported/],
      ['fix 2 all wall/region sph lj93 v_x 1 2.5', /must be a number/],
    ];
    for (const [line, re] of cases) {
      const r = await runScript(`${atom}${line}\nrun 0`);
      expect(message(r.error), line).toMatch(re);
    }
  });

  it('a particle outside a side-in region is an error', async () => {
    const r = await runScript(`${HEAD}pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
create_atoms 1 single 3 3 3 units box
region sph sphere 0 0 0 2.2 side in units box
fix 2 all wall/region sph harmonic 1.0 1.0 2.5
run 0`);
    expect(message(r.error)).toMatch(/outside surface of region/);
  });
});

describe('fix wall/reflect/stochastic', () => {
  /** n atoms at z = 0.3 (x spread), velocity (0, 0, -2): all hit the zlo wall at 0. */
  const atoms = (n: number, vz = -2) => {
    const lines: string[] = [];
    // a grid over the 10 x 10 box (x, y in [0.1, 9.9]); z = 0.3 puts every atom 0.3 above the zlo wall
    const side = Math.ceil(Math.sqrt(n));
    const step = 9.8 / side;
    for (let k = 0; k < n; k++) {
      lines.push(`create_atoms 1 single ${(0.1 + (k % side) * step).toFixed(4)} ${(0.1 + Math.floor(k / side) * step).toFixed(4)} 0.3 units box`);
    }
    // a wider box in x and y: the emitted atoms drift tangentially during the run
    return `${HEAD.replace('region box block 0 10 0 10 0 10', 'region box block -5 15 -5 15 0 10')}${lines.join('\n')}\npair_style lj/cut 2.5\npair_coeff 1 1 0.0 1.0 2.5\nvelocity all set 0 0 ${vz} units box\nfix 1 all nve\n`;
  };

  it('maxwell with accom 0 is specular: the normal velocity is reversed, tangential kept', async () => {
    const r = await runScript(`${atoms(1)}fix 2 all wall/reflect/stochastic maxwell 99 zlo 0.0 1.0 0 0 0 0.0 units box
timestep 0.005
dump 1 all custom 40 d.txt id vx vy vz
dump_modify 1 format float %.15g
run 40`);
    expect(r.error).toBeNull();
    const last = parseDump(r.files.get('d.txt') ?? '').at(-1);
    expect(last?.vz).toBeCloseTo(2, 12);
    expect(last?.vx).toBeCloseTo(0, 12);
    expect(last?.vy).toBeCloseTo(0, 12);
  });

  it('diffusive: <vz^2> = 2 kT/m and <vx^2> = kT/m for the emitted atoms (analytic, statistical)', async () => {
    const n = 2000;
    const r = await runScript(`${atoms(n)}fix 2 all wall/reflect/stochastic diffusive 4242 zlo 0.0 1.0 0 0 0 units box
timestep 0.005
run 40
write_dump all custom stoch.dump id vx vy vz modify format float %.15g sort id`);
    expect(r.error).toBeNull();
    const rows = parseDump(r.files.get('stoch.dump') ?? '');
    expect(rows.length).toBe(n);
    const m2z = rows.reduce((s, a) => s + a.vz ** 2, 0) / n;
    const m2x = rows.reduce((s, a) => s + a.vx ** 2, 0) / n;
    expect(Math.abs(m2z - 2) / 2).toBeLessThan(0.08);
    expect(Math.abs(m2x - 1) / 1).toBeLessThan(0.08);
    // the emitted normal velocity points away from the wall
    expect(rows.every((a) => a.vz > 0)).toBe(true);
  });

  it('ccl: <vz^2> = (1 - alpha_n) v_in^2 + 2 alpha_n kT/m, <vx^2> = alpha_x (2 - alpha_x) kT/m (analytic, statistical)', async () => {
    const n = 2000;
    const r = await runScript(`${atoms(n)}fix 2 all wall/reflect/stochastic ccl 2024 zlo 0.0 1.0 0 0 0 0.5 0.6 0.7 units box
timestep 0.005
run 40
write_dump all custom stoch.dump id vx vy vz modify format float %.15g sort id`);
    expect(r.error).toBeNull();
    const rows = parseDump(r.files.get('stoch.dump') ?? '');
    const m2z = rows.reduce((s, a) => s + a.vz ** 2, 0) / n;
    const m2x = rows.reduce((s, a) => s + a.vx ** 2, 0) / n;
    const expZ = (1 - 0.7) * 4 + 2 * 0.7;
    expect(Math.abs(m2z - expZ) / expZ).toBeLessThan(0.08);
    expect(Math.abs(m2x - 0.5 * 1.5) / 0.75).toBeLessThan(0.08);
  });

  it('argument and wall errors', async () => {
    const cases: [string, RegExp][] = [
      ['fix 2 all wall/reflect/stochastic foo 1 zlo 0 1 0 0 0', /rstyle must be diffusive, maxwell or ccl/],
      ['fix 2 all wall/reflect/stochastic diffusive 0 zlo 0 1 0 0 0', /seed must be a positive integer/],
      ['fix 2 all wall/reflect/stochastic maxwell 5 zlo 0 1 0 0 0', /needs 6 arguments after face zlo/],
      ['fix 2 all wall/reflect/stochastic ccl 5 zlo 0 1 0 0 0 0.5 0.5', /needs 8 arguments/],
      ['fix 2 all wall/reflect/stochastic maxwell 5 zlo 0 1 0 0 0 1.5', /between 0 and 1/],
      ['fix 2 all wall/reflect/stochastic diffusive 5 zlo 0 1 0 0 1', /wall velocity must lie in the plane/],
      ['fix 2 all wall/reflect/stochastic diffusive 5 zlo 0 1 0 0 0 units furlong', /units must be lattice or box/],
      ['fix 2 all wall/reflect/stochastic diffusive 5 qlo 0 1 0 0 0', /unknown argument 'qlo'/],
      ['fix 2 all wall/reflect/stochastic diffusive 5 zlo v_x 1 0 0 0', /must be a number/],
    ];
    for (const [line, re] of cases) {
      const r = await runScript(`${atoms(1)}${line}\nrun 0`);
      expect(message(r.error), line).toMatch(re);
    }
    const periodic = await runScript(`${atoms(1).replace('boundary f f f', 'boundary f f p')}fix 2 all wall/reflect/stochastic diffusive 5 zlo 0 1 0 0 0 units box\nrun 0`);
    expect(message(periodic.error)).toMatch(/non-periodic dimension/);
  });
});
