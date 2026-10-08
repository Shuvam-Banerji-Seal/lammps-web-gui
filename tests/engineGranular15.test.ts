import { describe, expect, it } from 'vitest';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { parseGranularSpec } from '../src/engine/force/pair/granular';
import { mdrDampCoeff, mdrElastic, mdrYieldDisplacement } from '../src/engine/force/pair/granular_mdr';

/*
 * pair_style granular, normal model mdr (docs.lammps.org/pair_granular.html). The closed forms are in
 * src/engine/force/pair/granular_mdr.ts; the parity runs are tests/oracle/w15gran_*.in (engineOracle.test.ts).
 * Values marked "measured with native LAMMPS" come from black-box runs of two equal spheres (diameter 1,
 * density 1, dt 0.001 unless stated).
 */

/** Runs a script; returns its thermo rows and the error message when the engine stops. */
async function run(script: string): Promise<{ rows: ThermoRow[]; error: string | null }> {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const ev = events.find((x) => x.kind === 'error');
  if (!error && ev && 'message' in ev) error = String(ev.message);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, error };
}

interface TwoSphere {
  /** Positions of the two spheres (x, y, z) and their diameters. */
  p1?: [number, number, number];
  p2?: [number, number, number];
  d1?: number;
  d2?: number;
  v1?: [number, number, number];
  v2?: [number, number, number];
  pairCoeff: string;
  runs: string;
  extra?: string;
}

/** A two-sphere script with thermo ke after every step. */
const twoSphere = (o: TwoSphere): string => `
units           lj
atom_style      sphere 1
atom_modify     map array
comm_modify     vel yes
newton          off
boundary        f f f
region          b block -10 10 -10 10 -10 10
create_box      1 b
create_atoms    1 single ${(o.p1 ?? [0, 0, 0]).join(' ')}
create_atoms    1 single ${(o.p2 ?? [0.9, 0, 0]).join(' ')}
set atom 1 diameter ${o.d1 ?? 1} density 1.0
set atom 2 diameter ${o.d2 ?? 1} density 1.0
group           one id 1
group           two id 2
velocity        one set ${(o.v1 ?? [0, 0, 0]).join(' ')} units box
velocity        two set ${(o.v2 ?? [0, 0, 0]).join(' ')} units box
pair_style granular
pair_coeff * * ${o.pairCoeff}
timestep 0.001
fix 1 all nve/sphere
thermo_style custom step ke
thermo_modify format float %.15g
thermo 1
${o.extra ?? ''}
${o.runs}
`;

/** Translational kinetic energy of two spheres of density 1 and diameter 1 (mass pi/6 each). */
const MASS = (4 / 3) * Math.PI * 0.125;

describe('pair_style granular mdr: parsing', () => {
  it('accepts the elastic mdr model with damping mdr 1 or 2 and the linear tangential models', () => {
    expect(() => parseGranularSpec(['mdr', '1e6', '0.3', '1e3', '0', '0.5', '0.2', 'damping', 'mdr', '1', 'tangential', 'linear_history', '940', '1.0', '0.7'])).not.toThrow();
    expect(() => parseGranularSpec(['mdr', '1e6', '0.3', '1e3', '0', '0.5', '0.2', 'damping', 'mdr', '2', 'tangential', 'linear_nohistory', '1.0', '0.7'])).not.toThrow();
  });

  it('names each unsupported part: adhesion, the damping class, tangential and twisting models', () => {
    const base = ['mdr', '1', '0.3', '1', '0', '0', '0.2'];
    expect(() => parseGranularSpec(['mdr', '1', '0.3', '1', '0.5', '0', '0.2', 'damping', 'mdr', '1', 'tangential', 'linear_nohistory', '0', '0'])).toThrow(/'mdr' is not implemented/);
    expect(() => parseGranularSpec([...base, 'tangential', 'linear_nohistory', '0', '0'])).toThrow(/damping mdr/);
    expect(() => parseGranularSpec([...base, 'damping', 'mdr', '3', 'tangential', 'linear_nohistory', '0', '0'])).toThrow(/d_type must be 1 or 2/);
    expect(() => parseGranularSpec([...base, 'damping', 'mdr', '1', 'tangential', 'mindlin', 'NULL', '1', '0.4'])).toThrow(/tangential mindlin with normal model 'mdr'/);
    expect(() => parseGranularSpec([...base, 'damping', 'mdr', '1', 'tangential', 'linear_nohistory', '0', '0', 'twisting', 'marshall'])).toThrow(/twisting marshall/);
    expect(() => parseGranularSpec(['hooke', '1', '0', 'tangential', 'linear_nohistory', '0', '0', 'damping', 'mdr', '1'])).toThrow(/needs the normal model 'mdr'/);
  });

  it('keeps synchronized_verlet as a StyleError', () => {
    expect(() => parseGranularSpec(['hooke', '1', '0', 'tangential', 'linear_history', '300', '0.5', '0.4', 'damping', 'velocity', 'synchronized_verlet'])).toThrow(/synchronized_verlet/);
  });
});

describe('pair_style granular mdr: closed forms (measured with native LAMMPS)', () => {
  // E' = E/(1 - nu^2); two spheres of radius R at overlap delta carry 2 R^2 E' br(delta/(4R)), br(x) = arccos(1-x) - (1-x) sqrt(2x-x^2)
  it('elastic force of two spheres of radius 0.5 (nu = 0; native F = 0.0293629534388009 at overlap 0.2)', () => {
    expect(mdrElastic(0.2, 0.5, 1, 1000).fne).toBeCloseTo(0.0293629534388009, 12);
    expect(mdrElastic(0.4, 0.5, 1, 1000).fne).toBeCloseTo(0.0817505543966421, 12);
  });

  it('elastic force scales with the radius (diameter 2, nu = 0, overlap 0.4: native 0.117452)', () => {
    expect(mdrElastic(0.4, 1.0, 1, 1000).fne).toBeCloseTo(0.117452, 5);
  });

  it('elastic force with nu = 0.3 uses E/(1-nu^2) (native 0.0322669818 at overlap 0.2)', () => {
    expect(mdrElastic(0.2, 0.5, 1 / (1 - 0.09), 1000).fne).toBeCloseTo(0.0322669818, 8);
  });

  // damping mdr 1: eta_n = eta_n0 sqrt(m_eff k_mdr), k_mdr = 2 E' a, a^2 = 4 R d - d^2 with d = delta/2
  it('damping mdr 1 coefficient (native eta_n = 0.4777357 for eta_n0 = 1, overlap 0.2, nu = 0)', () => {
    expect(mdrDampCoeff(1, 1, MASS / 2, 0.2, 0.5, 1)).toBeCloseTo(0.4777357, 6);
  });

  it('damping mdr 2 is the constant eta_n0 (native eta_n = 1.000000 for eta_n0 = 1)', () => {
    expect(mdrDampCoeff(2, 0.7, 0.3, 0.2, 0.5, 1)).toBe(0.7);
  });

  // yield: the first deviation of the native force from the elastic law was at a per-sphere overlap of 0.089531;
  // the root of paper eq. 13 with the exponent 4.4 (eq. 5) is 0.0895301
  it('yield displacement of the per-sphere overlap (Y = 0.1, nu = 0)', () => {
    expect(mdrYieldDisplacement(1, 0.5, 0.1)).toBeCloseTo(0.0895301, 6);
  });
});

describe('pair_style granular mdr: engine runs against measured native behaviour', () => {
  const H = {
    p2: [0.9, 0, 0] as [number, number, number],
    v1: [0.15, 0.09, 0] as [number, number, number],
    v2: [-0.12, 0.09, 0] as [number, number, number],
    pairCoeff: 'mdr 1.0 0.3 1000.0 0.0 0.0 0.2 damping mdr 2 tangential linear_nohistory 0.0 0.0',
  };

  it('moving contact, first step: kinetic energy matches native (0.00694557534801406)', async () => {
    const { rows, error } = await run(twoSphere({ ...H, runs: 'run 1' }));
    expect(error).toBeNull();
    expect(rows[1].ke).toBeCloseTo(0.00694557534801406, 12);
  });

  it('moving contact, six steps: kinetic energy matches native (0.00690146993155778)', async () => {
    const { rows, error } = await run(twoSphere({ ...H, runs: 'run 6' }));
    expect(error).toBeNull();
    expect(rows[6].ke).toBeCloseTo(0.00690146993155778, 12);
  });

  it('split run: the second run has the mdr damping in its setup (native 0.00690147160679585)', async () => {
    const { rows, error } = await run(twoSphere({ ...H, runs: 'run 1\nrun 5' }));
    expect(error).toBeNull();
    expect(rows[rows.length - 1].ke).toBeCloseTo(0.00690147160679585, 12);
  });

  it('two spheres with tangential linear_history and tangential sliding match native (step 6)', async () => {
    // measured with native LAMMPS: kinetic energy after step 6 is 0.00523348004551992 (tangential sliding, mu = 0.4)
    const { rows, error } = await run(twoSphere({
      p2: [0.9, 0, 0], v1: [0, 0.2, 0], v2: [0, 0, 0],
      pairCoeff: 'mdr 1.0 0.3 1000.0 0.0 0.0 0.2 damping mdr 2 tangential linear_history 300.0 0.5 0.4',
      runs: 'run 6',
    }));
    expect(error).toBeNull();
    expect(rows[6].ke).toBeCloseTo(0.00523348004551992, 12);
  });

  it('a contact closing a triangle of touching spheres is a StyleError (native topological penalty not implemented)', async () => {
    const { error } = await run(`
units           lj
atom_style      sphere 1
atom_modify     map array
comm_modify     vel yes
newton          off
boundary        f f f
region          b block -10 10 -10 10 -10 10
create_box      1 b
create_atoms    1 single 0.0 0.0 0.0
create_atoms    1 single 0.9 0.0 0.0
create_atoms    1 single 0.45 0.7794228634 0.0
set type 1 diameter 1.0 density 1.0
velocity all set 0.1 0.05 0.0 units box
pair_style granular
pair_coeff * * mdr 1.0 0.3 1000.0 0.0 0.0 0.2 damping mdr 2 tangential linear_nohistory 0.0 0.0
timestep 0.001
fix 1 all nve/sphere
run 3
`);
    expect(error).toMatch(/close a triangle/);
  });

  it('particles of unequal radii are a StyleError', async () => {
    const { error } = await run(twoSphere({
      p2: [1.2, 0, 0], d1: 1.0, d2: 2.0,
      pairCoeff: 'mdr 1.0 0.3 1000.0 0.0 0.0 0.2 damping mdr 2 tangential linear_nohistory 0.0 0.0',
      runs: 'run 1',
    }));
    expect(error).toMatch(/unequal radii/);
  });

  it('a contact reaching the yield displacement is a StyleError (plastic mdr not implemented)', async () => {
    const { error } = await run(twoSphere({
      p2: [0.9, 0, 0], v1: [0.5, 0, 0], v2: [0, 0, 0],
      pairCoeff: 'mdr 1.0 0.3 0.001 0.0 0.0 0.2 damping mdr 2 tangential linear_nohistory 0.0 0.0',
      runs: 'run 1',
    }));
    expect(error).toMatch(/yield displacement/);
  });

  it('fix wall/gran with the mdr model is a StyleError', async () => {
    const { error } = await run(`
units lj
atom_style sphere 1
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single 0.0 0.0 0.0
set type 1 diameter 1.0 density 1.0
pair_style granular
pair_coeff * * hooke 1 0 tangential linear_nohistory 0 0
fix w all wall/gran granular mdr 1.0 0.3 1000.0 0.0 0.0 0.2 damping mdr 2 tangential linear_nohistory 0 0 zplane -5 5
run 0
`);
    expect(error).toMatch(/fix wall\/gran with normal model 'mdr'/);
  });
});

describe('pair_style granular mdr: committed oracle cases', () => {
  const oracle = join(__dirname, 'oracle');
  const fixtures = join(__dirname, 'fixtures', 'oracle');
  it('the mdr oracle cases and their native fixtures are committed', () => {
    for (const name of ['w15gran_mdr_pair', 'w15gran_mdr_many']) {
      expect(existsSync(join(oracle, `${name}.in`))).toBe(true);
      expect(existsSync(join(fixtures, `${name}.json`))).toBe(true);
    }
  });
});
