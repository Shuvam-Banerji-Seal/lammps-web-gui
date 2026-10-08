import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { PAIRS } from '../src/engine/registry/pair_misc17';
import { PairHarmonicCut } from '../src/engine/force/pair/harmonic_cut';

/*
 * pair_style harmonic/cut (docs.lammps.org/pair_harmonic_cut.html,
 * src/engine/force/pair/harmonic_cut.ts): E = k (r_c - r)^2 for r < r_c, whose
 * force is 2 k (r_c - r) outward. Two atoms on the x axis in a periodic box;
 * the force on atom 2 must equal -dE/dr (central finite differences of the
 * total potential energy, norm no). Native parity lives in
 * tests/engineOracle.test.ts (w17harmcut).
 */

const BOX = 30;
const X1 = 5;

interface Run { pe: number; f2x: number; }

/** Runs one configuration through the interpreter: pe (total) and the x force on atom 2. */
const evaluate = async (style: string[], r: number, types: [number, number] = [1, 1]): Promise<Run> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  const lines = [
    'units lj', 'atom_style atomic', 'boundary p p p',
    `region box block 0 ${BOX} 0 ${BOX} 0 ${BOX}`, 'create_box 2 box',
    `create_atoms ${types[0]} single ${X1} ${X1} ${X1}`,
    `create_atoms ${types[1]} single ${X1 + r} ${X1} ${X1}`, 'mass * 1.0',
    ...style, 'thermo_style custom step pe', 'thermo_modify format float %.17g norm no', 'run 0',
    'write_dump all custom fdump.txt id fx fy fz modify format float %.17g sort id',
  ];
  await session.execute(lines.join('\n') + '\n');
  const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').pop();
  if (!row) throw new Error('no thermo row');
  const dump = (files.get('fdump.txt') ?? '').trim().split('\n');
  const k0 = dump.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const f2 = dump[k0 + 2].trim().split(/\s+/).map(Number);
  return { pe: row.row.pe as number, f2x: f2[1] };
};

/** Force on atom 2 equals -dE/dr from central finite differences, at each separation. */
const checkForces = async (style: string[], radii: number[], types: [number, number] = [1, 1]): Promise<void> => {
  const h = 1e-6;
  for (const r of radii) {
    const p = await evaluate(style, r + h, types);
    const m = await evaluate(style, r - h, types);
    const fd = -(p.pe - m.pe) / (2 * h);
    const f = (await evaluate(style, r, types)).f2x;
    expect(Math.abs(f - fd), `r=${r} force ${f} vs -dE/dr ${fd}`).toBeLessThan(1e-6 * (1 + Math.abs(fd)));
  }
};

const HARM = (k1 = 1.0, rc1 = 2.0, k2 = 1.0, rc2 = 2.0): string[] => [
  'pair_style harmonic/cut', `pair_coeff 1 1 ${k1} ${rc1}`, `pair_coeff 2 2 ${k2} ${rc2}`,
];

describe('pair_style harmonic/cut: forces are -dE/dr', () => {
  it('is registered under "harmonic/cut"', () => {
    expect(PAIRS['harmonic/cut']).toBeDefined();
    expect(PAIRS['harmonic/cut']()).toBeInstanceOf(PairHarmonicCut);
  });

  it('explicit coefficients: force = 2 k (r_c - r) at several separations', async () => {
    await checkForces(HARM(), [0.5, 0.8, 1.0, 1.3, 1.8]);
  });

  it('cross coefficients (types 1 and 2) are -dE/dr too', async () => {
    await checkForces(['pair_style harmonic/cut', 'pair_coeff 1 1 5.0 0.9', 'pair_coeff 2 2 2.0 1.1'], [0.4, 0.7, 1.0], [1, 2]);
  });

  it('is zero at and beyond r_c: no energy, no force', async () => {
    const out = await evaluate(HARM(1.0, 1.0), 1.3);
    expect(out.pe).toBe(0);
    expect(out.f2x).toBe(0);
    const at = await evaluate(HARM(1.0, 1.0), 1.0);
    expect(at.pe).toBe(0);
    expect(at.f2x).toBe(0);
  });
});

describe('pair_style harmonic/cut: mixing (measured native behaviour)', () => {
  const R = 1.3;

  it('geometric (default): k = sqrt(k1 k2), r_c = sqrt(rc1 rc2)', async () => {
    const out = await evaluate(HARM(2.0, 3.0, 8.0, 5.0), R, [1, 2]);
    const k = Math.sqrt(2 * 8), rc = Math.sqrt(3 * 5);
    expect(out.pe).toBeCloseTo(k * (rc - R) ** 2, 12);
  });

  it('explicit cross term equal to the mixed values is identical', async () => {
    const mixed = await evaluate(HARM(2.0, 3.0, 8.0, 5.0), R, [1, 2]);
    const explicit = await evaluate(
      ['pair_style harmonic/cut', 'pair_coeff 1 1 2.0 3.0', 'pair_coeff 2 2 8.0 5.0', `pair_coeff 1 2 ${Math.sqrt(16)} ${Math.sqrt(15)}`], R, [1, 2],
    );
    expect(explicit.pe).toBeCloseTo(mixed.pe, 12);
  });

  it('mix arithmetic: k is still geometric, r_c is the arithmetic mean', async () => {
    const out = await evaluate([...HARM(2.0, 3.0, 8.0, 5.0), 'pair_modify mix arithmetic'], R, [1, 2]);
    const k = Math.sqrt(2 * 8), rc = 0.5 * (3 + 5);
    expect(out.pe).toBeCloseTo(k * (rc - R) ** 2, 12);
  });

  it('mix sixthpower: the sixthpower energy and distance rules', async () => {
    const out = await evaluate([...HARM(2.0, 3.0, 8.0, 5.0), 'pair_modify mix sixthpower'], R, [1, 2]);
    const k = 2 * Math.sqrt(2 * 8) * 3 ** 3 * 5 ** 3 / (3 ** 6 + 5 ** 6);
    const rc = Math.pow(0.5 * (3 ** 6 + 5 ** 6), 1 / 6);
    expect(out.pe).toBeCloseTo(k * (rc - R) ** 2, 12);
  });
});

describe('pair_style harmonic/cut: argument checking and pair_modify', () => {
  it('pair_style takes no arguments', async () => {
    await expect(evaluate(['pair_style harmonic/cut 2.5', 'pair_coeff 1 1 1.0 2.0'], 1.3)).rejects.toThrow(/pair_style harmonic\/cut/);
  });

  it('pair_coeff needs exactly I J k r_c', async () => {
    await expect(evaluate(['pair_style harmonic/cut', 'pair_coeff 1 1 1.0'], 1.3)).rejects.toThrow(/pair_coeff/);
    await expect(evaluate(['pair_style harmonic/cut', 'pair_coeff 1 1 1.0 2.0 3.0'], 1.3)).rejects.toThrow(/pair_coeff/);
  });

  it('an unset type pair with no mixable diagonals is a StyleError', async () => {
    await expect(evaluate(['pair_style harmonic/cut', 'pair_coeff 1 1 1.0 2.0'], 1.3, [1, 2]))
      .rejects.toThrow(/not set/);
  });

  it('shift and tail are accepted and ignored (the potential is zero at r_c)', async () => {
    const base = await evaluate(HARM(), 1.3);
    const shift = await evaluate([...HARM(), 'pair_modify shift yes'], 1.3);
    const tail = await evaluate([...HARM(), 'pair_modify tail yes'], 1.3);
    expect(shift.pe).toBeCloseTo(base.pe, 15);
    expect(tail.pe).toBeCloseTo(base.pe, 15);
  });

  it('shift yes with tail yes is a StyleError (native pair_modify rule)', async () => {
    await expect(evaluate([...HARM(), 'pair_modify shift yes', 'pair_modify tail yes'], 1.3)).rejects.toThrow(/shift and tail/);
  });
});
