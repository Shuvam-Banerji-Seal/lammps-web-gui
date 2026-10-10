import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { parseComChunkWrap } from '../src/engine/compute/chunk';

/*
 * Wave-39 finite-size chunk support: the com/chunk wrap keyword
 * (docs.lammps.org/compute_com_chunk.html) and the spin and own-inertia terms
 * of angmom/chunk and inertia/chunk (src/engine/compute/finiteSpin.ts). The
 * expected numbers are closed-form (worked from the definitions) or were
 * measured with native LAMMPS (black box); the oracle cases w39cfs_* hold the
 * full native comparisons.
 */

/** Molecule 1: atom 1 at x = 9.5 and atom 2 at x = 0.5 with image x = 1, so the unwrapped centre is x = 10. */
const MOLECULAR = `straddling molecule
2 atoms
1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # molecular

1 1 1 9.5 5.0 5.0 0 0 0
2 1 1 0.5 5.0 5.0 1 0 0
`;

/** Two identical solid spheres (diameter 2, density 0.5, so m = 0.5 * 4/3 pi) at x = 5 and 6, spinning about z. */
const SPHERES = `two spheres
2 atoms
1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Atoms # sphere

1 1 2.0 0.5 5.0 5.0 5.0 0 0 0
2 1 2.0 0.5 6.0 5.0 5.0 0 0 0

Velocities

1 0 0 0 0 0 2
2 0 0 0 0 0 2
`;

/** Runs a script on the given data file; returns thermo rows, or the first error message. */
const run = async (text: string, data: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  session.addFile('case.data', data);
  try {
    await session.execute(text);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return { rows: [] as ThermoRow[], error: err && 'message' in err ? String(err.message) : String(e) };
  }
  const err = events.find((ev) => ev.kind === 'error');
  if (err && 'message' in err) return { rows: [] as ThermoRow[], error: String(err.message) };
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, error: null as string | null };
};

const close = (got: number, want: number, abs = 1e-9) => Math.abs(got - want) <= abs + 1e-9 * Math.abs(want);

describe('com/chunk wrap keyword parsing', () => {
  it('accepts the logical words of the page, lower case only', () => {
    for (const w of ['yes', 'true', 'on', '1']) expect(parseComChunkWrap('c', ['wrap', w])).toBe(true);
    for (const w of ['no', 'false', 'off', '0']) expect(parseComChunkWrap('c', ['wrap', w])).toBe(false);
    expect(parseComChunkWrap('c', [])).toBe(false);
  });

  it('refuses other words, a missing value, unknown keywords and upper-case words', () => {
    for (const w of ['maybe', 'YES', 'Yes', 't', '2', '.true.']) {
      expect(() => parseComChunkWrap('c', ['wrap', w])).toThrow(/expected a boolean value for wrap/);
    }
    expect(() => parseComChunkWrap('c', ['wrap'])).toThrow(/wrap needs a logical value/);
    expect(() => parseComChunkWrap('c', ['wrap', 'yes', 'extra'])).toThrow(/unknown keyword 'extra'/);
  });

  it('a repeated wrap keeps the last value', () => {
    expect(parseComChunkWrap('c', ['wrap', 'yes', 'wrap', 'no'])).toBe(false);
    expect(parseComChunkWrap('c', ['wrap', 'no', 'wrap', 'yes'])).toBe(true);
  });

  it('wrap on the script maps the centre back into the periodic cell', async () => {
    const script = (kw: string) => `units lj
atom_style molecular
read_data case.data
compute cc all chunk/atom molecule compress no
compute c all com/chunk cc ${kw}
thermo_style custom step c_c[1][1] c_c[1][2] c_c[1][3]
thermo_modify format float %.15g norm no
run 0
`;
    const wrapped = await run(script('wrap yes'), MOLECULAR);
    expect(wrapped.error).toBeNull();
    // Measured with native LAMMPS (black box): unwrapped centre x = 10 becomes 0 with wrap yes.
    expect(close(wrapped.rows[0]['c_c[1][1]'] as number, 0)).toBe(true);
    expect(close(wrapped.rows[0]['c_c[1][2]'] as number, 5)).toBe(true);
    const plain = await run(script('wrap no'), MOLECULAR);
    expect(close(plain.rows[0]['c_c[1][1]'] as number, 10)).toBe(true);
    const bad = await run(script('wrap maybe'), MOLECULAR);
    expect(bad.error).toMatch(/expected a boolean value for wrap/);
  });
});

describe('finite-size spin in angmom/chunk and inertia/chunk', () => {
  it('two solid spheres in one chunk: spin angular momentum and own inertia', async () => {
    const { rows, error } = await run(`units lj
atom_style sphere
read_data case.data
compute cc all chunk/atom type
compute L all angmom/chunk cc
compute I all inertia/chunk cc
compute W all omega/chunk cc
thermo_style custom step c_L[1][1] c_L[1][2] c_L[1][3] c_I[1][1] c_I[1][2] c_I[1][3] c_I[1][4] c_I[1][5] c_I[1][6] c_W[1][1] c_W[1][2] c_W[1][3]
thermo_modify format float %.15g norm no
run 0
`, SPHERES);
    expect(error).toBeNull();
    const r = rows[0];
    // m = 0.5 * 4/3 pi 1^3; each sphere has spin I_s = 2/5 m r^2 about its own centre with omega = 2 z.
    const m = 0.5 * (4 / 3) * Math.PI;
    const Is = 0.4 * m;
    // The centre of mass is x = 5.5. Spin adds 2 I_s omega along z only.
    expect(close(r['c_L[1][1]'] as number, 0)).toBe(true);
    expect(close(r['c_L[1][2]'] as number, 0)).toBe(true);
    expect(close(r['c_L[1][3]'] as number, 2 * Is * 2)).toBe(true);
    // Own tensor is isotropic per sphere; the parallel-axis part is m (d.d 1 - d d) with d = (+-0.5, 0, 0).
    const Ixx = 2 * Is;
    const Iyy = 2 * Is + 2 * m * 0.25;
    expect(close(r['c_I[1][1]'] as number, Ixx)).toBe(true);
    expect(close(r['c_I[1][2]'] as number, Iyy)).toBe(true);
    expect(close(r['c_I[1][3]'] as number, Iyy)).toBe(true);
    expect(close(r['c_I[1][4]'] as number, 0)).toBe(true);
    expect(close(r['c_I[1][5]'] as number, 0)).toBe(true);
    expect(close(r['c_I[1][6]'] as number, 0)).toBe(true);
    // omega/chunk keeps the point-mass values: no velocities, so zero.
    expect(close(r['c_W[1][1]'] as number, 0)).toBe(true);
    expect(close(r['c_W[1][3]'] as number, 0)).toBe(true);
  });
});
