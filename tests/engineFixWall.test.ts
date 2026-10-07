import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Direct checks for the fix wall styles (docs.lammps.org/fix_wall.html) on a
 * tiny one-atom system with no pair style, so every documented formula can be
 * computed by hand. All cases use units lj (ftm2v = mvv2e = 1) and
 * thermo_modify norm no, so thermo values are raw.
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
  return { session, rows, error, events };
};

const BASE = `
units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 2
mass            1 1.0
`;

/** Documented 9-3 potential, unshifted and shifted to 0 at the cutoff. */
const E93 = (r: number, eps = 1, sig = 1, rc = 2.5) => {
  const u = sig / r;
  const e = eps * ((2 / 15) * u ** 9 - u ** 3);
  const uc = sig / rc;
  return e - eps * ((2 / 15) * uc ** 9 - uc ** 3);
};
/** F = -dE/dr for the 9-3 wall. */
const F93 = (r: number, eps = 1, sig = 1) => {
  const u = sig / r;
  return eps * (sig / (r * r)) * ((6 / 5) * u ** 8 - 3 * u ** 2);
};
const E126 = (r: number, eps = 1, sig = 1, rc = 2.5) => {
  const u = sig / r;
  return 4 * eps * (u ** 12 - u ** 6) - 4 * eps * ((sig / rc) ** 12 - (sig / rc) ** 6);
};
const F126 = (r: number, eps = 1, sig = 1) => {
  const u = sig / r;
  return 4 * eps * (sig / (r * r)) * (12 * u ** 11 - 6 * u ** 5);
};
const E1043 = (r: number, eps = 1, sig = 1, rc = 2.5) => {
  const a = 0.61 / Math.SQRT2;
  const f = (x: number) => 2 * Math.PI * eps * ((2 / 5) * (sig / x) ** 10 - (sig / x) ** 4 - (Math.SQRT2 * sig ** 3) / (3 * (x + a * sig) ** 3));
  return f(r) - f(rc);
};
const F1043 = (r: number, eps = 1, sig = 1) => {
  const a = 0.61 / Math.SQRT2;
  const u = sig / r;
  return 2 * Math.PI * eps * ((4 * u ** 10 - 4 * u ** 4) / r - (Math.SQRT2 * sig ** 3) / (r + a * sig) ** 4);
};
const Emorse = (r: number, d0 = 1, al = 2, r0 = 1, rc = 2.5) => {
  const f = (x: number) => d0 * (Math.exp(-2 * al * (x - r0)) - 2 * Math.exp(-al * (x - r0)));
  return f(r) - f(rc);
};
const Fmorse = (r: number, d0 = 1, al = 2, r0 = 1) => {
  const a = Math.exp(-al * (r - r0));
  return 2 * al * d0 * (a * a - a);
};

describe('fix wall/lj93', () => {
  it('applies the documented 9-3 force and shifted energy at r < rc', async () => {
    const { rows, error } = await runScript(`${BASE}
fix 1 all wall/lj93 zlo 1.0 1.0 1.0 2.5 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    // atom at z=2, wall at z=1: r = 1, inside the 2.5 cutoff
    const r = rows[rows.length - 1];
    expect(r.pe).toBeCloseTo(E93(1), 10);
    expect(r.f_1).toBeCloseTo(E93(1), 10);
    // attraction at r=1 (past the 9-3 minimum): atom pushed toward the wall,
    // so the force ON the lo wall points +z (not outward) and is positive
    expect(r['f_1[1]']).toBeCloseTo(-F93(1), 10);
    const { session } = await runScript(`${BASE}
fix 1 all wall/lj93 zlo 1.0 1.0 1.0 2.5 units box
run 0`);
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, F93(1)]);
  });

  it('is repulsive for r below the 9-3 minimum and the wall force is outward-negative', async () => {
    const { rows, error, session } = await runScript(`units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 1.8
mass            1 1.0
fix 1 all wall/lj93 zlo 1.0 1.0 1.0 2.5 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    // r = 0.8 is inside the 9-3 minimum (r_min = 0.828 sigma): repulsion
    const r = rows[rows.length - 1];
    expect(r.pe).toBeCloseTo(E93(0.8), 10);
    expect(r['f_1[1]']).toBeCloseTo(-F93(0.8), 10);
    expect(F93(0.8)).toBeGreaterThan(0);
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, F93(0.8)]);
  });

  it('gives zero interaction beyond the cutoff', async () => {
    const { rows, error, session } = await runScript(`${BASE}
fix 1 all wall/lj93 zlo 1.0 1.0 1.0 0.5 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    expect(r.pe).toBe(0);
    expect(r.f_1).toBe(0);
    expect(r['f_1[1]']).toBe(0);
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, 0]);
  });
});

describe('fix wall/lj126', () => {
  it('applies the documented 12-6 force and shifted energy', async () => {
    const { rows, error, session } = await runScript(`${BASE}
fix 1 all wall/lj126 zlo 1.0 1.0 1.0 2.5 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    expect(r.pe).toBeCloseTo(E126(1), 10);
    expect(r['f_1[1]']).toBeCloseTo(-F126(1), 10);
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, F126(1)]);
  });
});

describe('fix wall/lj1043', () => {
  it('applies the documented 10-4-3 force and shifted energy', async () => {
    const { rows, error, session } = await runScript(`${BASE}
fix 1 all wall/lj1043 zlo 1.0 1.0 1.0 2.5 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    expect(r.pe).toBeCloseTo(E1043(1), 10);
    expect(r['f_1[1]']).toBeCloseTo(-F1043(1), 10);
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, F1043(1)]);
  });
});

describe('fix wall/harmonic', () => {
  it('applies E = eps (r - rc)^2 and ignores sigma', async () => {
    const { rows, error, session } = await runScript(`${BASE}
fix 1 all wall/harmonic zlo 1.0 5.0 7.0 2.0 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    // r = 1, rc = 2: E = 5*(1-2)^2 = 5, F = 2*5*(2-1) = 10 (repulsive)
    expect(r.pe).toBeCloseTo(5 * (1 - 2) ** 2, 12);
    expect(r['f_1[1]']).toBeCloseTo(-10, 12);
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, 10]);
  });
});

describe('fix wall/morse', () => {
  it('applies the documented Morse force and shifted energy', async () => {
    const { rows, error, session } = await runScript(`${BASE}
fix 1 all wall/morse zlo 1.0 1.0 2.0 1.0 2.5 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    // r = 1 = r0: potential minimum, force zero after the shift
    expect(r.pe).toBeCloseTo(Emorse(1), 10);
    expect(r['f_1[1]']).toBeCloseTo(-Fmorse(1), 10);
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, Fmorse(1)]);
    // r = 0.8 < r0 = 1: repulsive branch pushes the atom away from the wall
    const near = await runScript(`units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 1.8
mass            1 1.0
fix 1 all wall/morse zlo 1.0 1.0 2.0 1.0 2.5 units box
thermo_style custom step f_1[1]
thermo_modify norm no
run 0`);
    expect(near.error?.message ?? '').toBe('');
    expect(Fmorse(0.8)).toBeGreaterThan(0);
    expect(near.rows[near.rows.length - 1]['f_1[1]']).toBeCloseTo(-Fmorse(0.8), 10);
  });
});

describe('fix wall faces, groups and coordinates', () => {
  it('flips the force for a hi wall and tallies one vector entry per wall', async () => {
    const { rows, error, session } = await runScript(`${BASE}
fix 1 all wall/harmonic zlo 1.0 5.0 1.0 2.0 zhi 3.0 5.0 1.0 2.0 units box
thermo_style custom step f_1 f_1[1] f_1[2]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    // rlo = 2-1 = 1 (repulsive, atom pushed +z), rhi = 3-2 = 1 (pushed -z): cancel
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, 10 - 10]);
    expect(r['f_1[1]']).toBeCloseTo(-10, 12);   // outward on the lo wall: negative
    expect(r['f_1[2]']).toBeCloseTo(10, 12);    // outward on the hi wall: positive
  });

  it('only acts on atoms in the fix group', async () => {
    const { error, session } = await runScript(`${BASE}
create_atoms    1 single 5 5 1.0
group           far id 1
fix 1 far wall/harmonic zlo 1.0 5.0 1.0 2.0 units box
run 0`);
    expect(error?.message ?? '').toBe('');
    // atom 2 (id 2, z=1, ON the wall) is not in group far: no error, no force;
    // atom 1 (id 1, z=2, r=1) is in the group and gets the full wall force
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, 10]);
    expect([...session!.system!.f.slice(3, 6)]).toEqual([0, 0, 0]);
  });

  it('EDGE follows the box edge; numeric coords scale with the lattice for units lattice', async () => {
    const edge = await runScript(`${BASE}
fix 1 all wall/harmonic zlo EDGE 5.0 1.0 1.5 units box
thermo_style custom step f_1[1]
thermo_modify norm no
run 0`);
    expect(edge.error?.message ?? '').toBe('');
    // zlo EDGE = 0, atom z=2: r = 2 > rc = 1.5: no force
    expect(edge.rows[edge.rows.length - 1]['f_1[1]']).toBe(0);

    const lat = await runScript(`${BASE}
lattice none 2.0
fix 1 all wall/harmonic zlo 0.5 5.0 1.0 2.0 units lattice
thermo_style custom step f_1[1]
thermo_modify norm no
run 0`);
    expect(lat.error?.message ?? '').toBe('');
    // lattice spacing 2: wall at 0.5*2 = 1.0, r = 1
    expect(lat.rows[lat.rows.length - 1]['f_1[1]']).toBeCloseTo(-10, 12);

    const box = await runScript(`${BASE}
lattice none 2.0
fix 1 all wall/harmonic zlo 0.5 5.0 1.0 2.0 units box
thermo_style custom step f_1[1]
thermo_modify norm no
run 0`);
    expect(box.error?.message ?? '').toBe('');
    // units box: wall at 0.5, r = 1.5 < rc = 2.0: F = 2*5*(2-1.5) = 5
    expect(box.rows[box.rows.length - 1]['f_1[1]']).toBeCloseTo(-5, 12);
  });

  it('evaluates a variable wall position', async () => {
    const { rows, error } = await runScript(`${BASE}
variable zw equal 1.0
fix 1 all wall/harmonic zlo v_zw 5.0 1.0 2.0 units box
thermo_style custom step f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    expect(rows[rows.length - 1]['f_1[1]']).toBeCloseTo(-10, 12);
  });

  it('supports fix_modify virial yes: the wall virial is r.F along the wall axis', async () => {
    const withV = await runScript(`${BASE}
fix 1 all wall/harmonic zlo 1.0 5.0 1.0 2.0 units box
fix_modify 1 virial yes
thermo_style custom step pzz
thermo_modify norm no
run 0`);
    expect(withV.error?.message ?? '').toBe('');
    // temp = 0 (no velocities): pzz = W_zz/V (compute_pressure vector form),
    // W_zz = r*F = 1*10, V = 1000
    expect(withV.rows[withV.rows.length - 1].pzz).toBeCloseTo(10 / 1000, 12);
    const withoutV = await runScript(`${BASE}
fix 1 all wall/harmonic zlo 1.0 5.0 1.0 2.0 units box
thermo_style custom step pzz
thermo_modify norm no
run 0`);
    expect(withoutV.rows[withoutV.rows.length - 1].pzz).toBe(0);
  });

  it('normalizes the extensive scalar and vector with thermo norm yes', async () => {
    const { rows, error } = await runScript(`${BASE}
fix 1 all wall/harmonic zlo 1.0 5.0 1.0 2.0 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
run 0`);
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    expect(r.pe).toBeCloseTo(5, 12);      // one atom: norm divides by 1
    expect(r.f_1).toBeCloseTo(5, 12);
    expect(r['f_1[1]']).toBeCloseTo(-10, 12);
  });
});

describe('fix wall errors', () => {
  it('errors for a particle on or behind the wall (measured native message)', async () => {
    const on = await runScript(`${BASE}
fix 1 all wall/lj93 zlo 2.0 1.0 1.0 2.5 units box
run 0`);
    expect(on.error?.message).toMatch(/Particle on or inside fix 1 wall\/lj93 surface/);
    const behind = await runScript(`units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 3.5
mass            1 1.0
fix 1 all wall/lj93 zhi 3.0 1.0 1.0 2.5 units box
run 0`);
    expect(behind.error?.message).toMatch(/Particle on or inside fix 1 wall\/lj93 surface/);
  });

  it('rejects bad arguments', async () => {
    const bad = async (line: string) => (await runScript(`${BASE}\n${line}\nrun 0`)).error?.message ?? '';
    expect(await bad('fix 1 all wall/lj93')).toMatch(/usage: fix ID group-ID wall\/lj93/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE 1.0 1.0')).toMatch(/face zlo needs 4 arguments/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE 1.0 1.0 2.5 foo bar')).toMatch(/unknown argument 'foo'/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE 1.0 1.0 2.5 units parsec')).toMatch(/units must be lattice or box/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE 1.0 1.0 2.5 pbc maybe')).toMatch(/pbc value must be yes or no/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE 1.0 1.0 2.5 fld 1')).toMatch(/fld value must be yes or no/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE 1.0 1.0 v_cut')).toMatch(/cutoff must be a number/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE 1.0 1.0 0.0')).toMatch(/cutoff must be > 0/);
    expect(await bad('fix 1 all wall/lj93 1.0 1.0 1.0 2.5')).toMatch(/unknown argument '1.0'/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE 1.0 1.0 2.5 zlo EDGE 1.0 1.0 2.5')).toMatch(/face zlo is specified more than once/);
    expect(await bad('fix 1 all wall/morse zlo EDGE 1.0 1.0 2.5')).toMatch(/face zlo needs 5 arguments/);
    expect(await bad('fix 1 all wall/lj93 zlo EDGE v_nope 1.0 2.5')).toMatch(/variable nope does not exist/);
    expect(await bad('fix 1 all wall/lj93 zlo v_nope 1.0 1.0 2.5')).toMatch(/variable nope does not exist/);
    expect(await bad('fix 1 all wall/lj93 zlo 1e-3junk 1.0 1.0 2.5')).toMatch(/expected a number/);
  });

  it('rejects walls in periodic dimensions unless pbc yes, and z walls in 2d', async () => {
    const per = await runScript(`units lj
atom_style atomic
boundary p p p
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5 5 2
mass 1 1.0
fix 1 all wall/lj93 zlo 1.0 1.0 1.0 2.5 units box
run 0`);
    expect(per.error?.message).toMatch(/periodic dimension requires pbc yes/);
    const ok = await runScript(`units lj
atom_style atomic
boundary p p p
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5 5 2
mass 1 1.0
fix 1 all wall/lj93 zlo 1.0 1.0 1.0 2.5 units box pbc yes
run 0`);
    expect(ok.error?.message ?? '').toBe('');
    const two = await runScript(`units lj
atom_style atomic
dimension 2
boundary p p p
region box block 0 10 0 10 -1 1
create_box 1 box
create_atoms 1 single 5 5 0
mass 1 1.0
fix 1 all wall/lj93 zlo 1.0 1.0 1.0 2.5 units box
run 0`);
    expect(two.error?.message).toMatch(/z wall .*2d simulation/);
  });
});

