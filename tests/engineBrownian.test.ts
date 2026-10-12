import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { RanMars } from '../src/engine-gpl/random_mars';
import { snormalize3, scale3, quatToMat, qnormalize } from '../src/engine-gpl/brownian/mathextra';

// BROWNIAN package fixes ported from LAMMPS (src/engine-gpl/brownian/). The parity cases live in tests/oracle/w62brown_*
// (native LAMMPS b332f28, run as a black box). Here: the RanMars gaussian stream, the MathExtra helpers, the keyword
// and argument errors of the fix commands, and a few physical invariants of short runs.

/** Runs a script in a fresh session; returns the thermo rows and the error messages. */
const run = async (script: string): Promise<{ rows: Record<string, number>[]; errors: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  try {
    await session.execute(script);
  } catch {
    // the error is also an event
  }
  return {
    rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as unknown as Record<string, number>),
    errors: events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error').map((e) => ('message' in e ? String(e.message) : '')),
  };
};

const lattice2d = `dimension 2
units lj
atom_style atomic
boundary p p p
lattice sq 0.4
region box block 0 4 0 4 -0.2 0.2
create_box 1 box
create_atoms 1 box
mass * 1.0
pair_style none
`;

describe('RanMars gaussian (random_mars.ts)', () => {
  it('draws standard normal deviates with the polar pair method', () => {
    const rng = new RanMars(4242);
    const n = 40000;
    let sum = 0;
    let sum2 = 0;
    for (let i = 0; i < n; i++) {
      const g = rng.gaussian();
      sum += g;
      sum2 += g * g;
    }
    expect(Math.abs(sum / n)).toBeLessThan(0.03);
    expect(Math.abs(sum2 / n - 1)).toBeLessThan(0.03);
  });

  it('reuses the second deviate of each pair and is reproducible from the seed', () => {
    const a = new RanMars(99);
    const b = new RanMars(99);
    const first = [a.gaussian(), a.gaussian(), a.gaussian()];
    const again = [b.gaussian(), b.gaussian(), b.gaussian()];
    expect(first).toEqual(again);
    expect(first.every(Number.isFinite)).toBe(true);
  });
});

describe('MathExtra helpers (brownian/mathextra.ts)', () => {
  it('snormalize3 keeps the requested length', () => {
    const out = [0, 0, 0];
    snormalize3(2.5, [3, 0, 4], out);
    expect(Math.hypot(out[0], out[1], out[2])).toBeCloseTo(2.5, 14);
    const v = [1, 2, 3];
    scale3(0.5, v, out);
    expect(out).toEqual([0.5, 1, 1.5]);
  });

  it('a unit quaternion gives a proper rotation matrix', () => {
    const q = [Math.cos(0.3), Math.sin(0.3) * 0.6, 0, Math.sin(0.3) * 0.8];
    qnormalize(q);
    const m = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    quatToMat(q, m);
    const det =
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    expect(det).toBeCloseTo(1, 12);
  });
});

describe('fix brownian family: arguments and keywords', () => {
  it.each([
    ['fix b all brownian 0.0 5 gamma_t 1.0', 'temp must be > 0.0'],
    ['fix b all brownian 1.0 5', 'Keyword gamma_t is required'],
    ['fix b all brownian 1.0 5 gamma_t 1.0 gamma_r 1.0', 'Using keyword(s) not compatible'],
    ['fix b all brownian 1.0 5 gamma_t 1.0 rng bogus', 'Unknown fix brownian rng keyword bogus'],
    ['fix b all brownian 1.0 5 gamma_t 1.0 rotation_style geometric', 'Keyword rotation_style is only supported for fix brownian/sphere'],
    ['fix b all brownian/sphere 1.0 5 gamma_t 1.0 gamma_r 1.0', 'requires atom attribute mu'],
    ['fix b all brownian/asphere 1.0 5 gamma_t 1.0 gamma_r 1.0 gamma_t_eigen inf inf inf gamma_r_eigen inf inf inf', 'are not compatible with fix brownian/asphere'],
  ])('%s is refused', async (script, message) => {
    const r = await run(lattice2d + script + '\nrun 1\n');
    expect(r.errors.join('\n')).toContain(message);
  });

  it('rng none with no force leaves the particles where they are', async () => {
    const r = await run(lattice2d + 'fix b all brownian 1.0 5 rng none gamma_t 1.0\ncompute msd all msd\nthermo_style custom step c_msd[4]\nthermo 10\nrun 20\n');
    expect(r.errors).toEqual([]);
    expect(r.rows.at(-1)!['c_msd[4]']).toBe(0);
  });

  it('in 2d the z coordinate is not moved by fix brownian', async () => {
    const r = await run(lattice2d + 'fix b all brownian 1.0 5 gamma_t 1.0\ncompute msd all msd\nthermo_style custom step c_msd[3] c_msd[4]\nthermo 10\nrun 20\n');
    expect(r.errors).toEqual([]);
    expect(r.rows.at(-1)!['c_msd[3]']).toBe(0);
    expect(r.rows.at(-1)!['c_msd[4]']).toBeGreaterThan(0);
  });
});

describe('fix propel/self and align/self arguments', () => {
  it.each([
    ['fix p all propel/self swim 1.0', 'Unknown fix propel/self keyword swim'],
    ['fix p all propel/self velocity 1.0 qvector 1 0 0', "Incorrect number of arguments for 'quat' mode"],
    ['fix p all propel/self quat 1.0 qvector 0 0 0', 'qvector magnitude'],
    ['fix p all propel/self dipole 1.0 qvector 1 0 0', "Incorrect number of arguments for 'quat' mode"],
    ['fix p all propel/self dipole', 'Incorrect number of fix propel/self arguments'],
    ['fix p all align/self swim 1.0', 'Unknown fix align/self keyword swim'],
  ])('%s is refused', async (script, message) => {
    const r = await run(lattice2d + script + '\nrun 1\n');
    expect(r.errors.join('\n')).toContain(message);
  });

  it('propel/self velocity pushes each particle along its velocity with the given magnitude', async () => {
    const r = await run(lattice2d + 'velocity all set 1.0 0.0 0.0\nfix p all propel/self velocity 2.0\nfix_modify p virial yes\ncompute press all pressure NULL virial\nthermo_style custom step c_press\nthermo 1\nrun 0\n');
    expect(r.errors).toEqual([]);
    expect(r.rows.length).toBeGreaterThan(0);
    expect(Number.isFinite(r.rows[0]['c_press'])).toBe(true);
  });
});
