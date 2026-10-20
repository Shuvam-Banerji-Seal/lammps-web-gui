import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { geodesicDistance } from '../src/engine-gpl/spin/fix_neb_spin';
import { initialRotation, parseSpinFile } from '../src/engine-gpl/spin/neb_spin';

/*
 * neb/spin and fix neb/spin (docs.lammps.org/neb_spin.html, fix_neb_spin.html; GPL port in src/engine-gpl/spin/):
 * refusals, the table and the pure helpers. Numeric parity with native LAMMPS is in tests/oracle/w194gneb_*.in.
 */

const DATA = `spin chain (atom_style spin data columns: ID type x y z spx spy spz sp)
3 atoms
1 atom types
0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 55.845

Atoms # spin

1 1 1.0 5 5 0 0 1 2.2 0 0 0
2 1 3.5 5 5 0 0 1 2.2 0 0 0
3 1 6.0 5 5 0 0 1 2.2 0 0 0
`;
/** The same chain without spins (atom_style atomic: ID type x y z). */
const DATA_ATOMIC = DATA.replace(/^(\d+) 1 (\S+) 5 5 .*$/gm, '$1 1 $2 5 5');
const FINAL = '3\n1 2.2 1.0 5 5 1 0 0\n2 2.2 3.5 5 5 0.6 0 0.8\n3 2.2 6.0 5 5 0 1 0\n';

const setup = (opts: { style?: string; minStyle?: string; fix?: string } = {}): string => `units metal
boundary p p p
atom_style ${opts.style ?? 'spin'}
atom_modify map array
read_data chain.data
mass 1 55.845
${opts.style && opts.style !== 'spin' ? 'pair_style zero 3.0\npair_coeff * *' : 'pair_style spin/exchange 3.0\npair_coeff * * exchange 3.0 0.02726 0.2171 1.841'}
fix 1 all precession/spin zeeman 0.1 0.0 0.0 1.0
fix 3 all ${opts.fix ?? 'neb/spin 1.0'}
min_style ${opts.minStyle ?? 'spin'}
`;

const run = async (extra: string, partitions = 3, text = setup()) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, partitions);
  session.addFile('chain.data', text.includes('atom_style atomic') ? DATA_ATOMIC : DATA);
  session.addFile('chain.final', FINAL);
  let error: Error | null = null;
  try {
    await session.execute(text + extra);
  } catch (e) {
    error = e as Error;
  }
  return { events, error };
};

describe('neb/spin refusals', () => {
  it('needs two or more partitions', async () => {
    const { error } = await run('neb/spin 0.0 0.0 2 2 1 final chain.final\n', 1);
    expect(error?.message).toMatch(/Cannot use NEBSpin with a single replica/);
  });

  it('needs atom_style spin', async () => {
    const text = setup({ style: 'atomic' }).replace(/fix 1 all precession.*\n/, '');
    const { error } = await run('neb/spin 0.0 0.0 2 2 1 none\n', 3, text);
    expect(error?.message).toMatch(/requires atom\/spin style/);
  });

  it('runs with min_style spin only', async () => {
    expect((await run('neb/spin 0.0 0.0 2 2 1 final chain.final\n', 3, setup({ minStyle: 'spin/cg' }))).error?.message).toMatch(/min_style spin\/cg is not supported by the browser engine yet/);
    expect((await run('neb/spin 0.0 0.0 2 2 1 final chain.final\n', 3, setup({ minStyle: 'cg' }))).error?.message).toMatch(/requires a spin minimizer/);
  });

  it('read_data refuses an atom on the upper face of a fixed dimension (native err0016)', async () => {
    // read_data.cpp: "Did not assign all atoms correctly" when an atom is not owned by the box (lo <= x < hi, Domain::ownatom);
    // the chain's atoms sit at z = 5, so a z box 0..5 with fixed faces leaves them outside
    const data = DATA.replace('0 10 zlo zhi', '0 5 zlo zhi');
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, 1);
    session.addFile('chain.data', data);
    await expect(session.execute('units metal\nboundary p p f\natom_style spin\natom_modify map array\nread_data chain.data\n')).rejects.toThrow(/Did not assign all atoms correctly/);
  });

  it('checks its arguments', async () => {
    expect((await run('neb/spin 0.0 0.0 2 2\n')).error?.message).toMatch(/usage: neb\/spin/);
    expect((await run('neb/spin 0.0 0.0 4 4 3 none\n')).error?.message).toMatch(/multiples of Nevery/);
    expect((await run('neb/spin 0.0 0.0 2 2 1 sideways\n')).error?.message).toMatch(/file-style must be final, each or none/);
    expect((await run('neb/spin 0.0 0.0 2 2 1 final chain.final loud\n')).error?.message).toMatch(/illegal neb\/spin keyword 'loud'/);
  });

  it('refuses the fix neb/spin keywords and a minimize with the fix', async () => {
    // As native LAMMPS does (src/SPIN/fix_neb_spin.cpp at b332f28): parallel and perp are errors, end and lattice are read
    // past and ignored, a Kspring <= 0 is "Illegal fix neb command"
    expect((await run('', 3, setup({ fix: 'neb/spin 1.0 parallel neigh' }))).error?.message).toMatch(/Illegal fix neb\/spin command/);
    expect((await run('', 3, setup({ fix: 'neb/spin 1.0 perp 1.0' }))).error?.message).toMatch(/Illegal fix neb\/spin command/);
    expect((await run('', 3, setup({ fix: 'neb/spin 1.0 lattice yes end first 1.0' }))).error).toBeNull();
    expect((await run('', 3, setup({ fix: 'neb/spin 1.0 sideways 2' }))).error?.message).toMatch(/Illegal fix neb\/spin command/);
    expect((await run('', 3, setup({ fix: 'neb/spin 0.0' }))).error?.message).toMatch(/Illegal fix neb command/);
    expect((await run('minimize 0.0 0.0 5 5\n', 1)).error?.message).toMatch(/acts only inside a neb\/spin command/);
  });
});

describe('neb/spin table', () => {
  it('prints the regular and climbing stages with the neb/spin columns', async () => {
    const { events, error } = await run('neb/spin 0.0 0.0 2 2 1 final chain.final verbose\n');
    expect(error).toBeNull();
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo' && 'RD1' in e.row).map((e) => e.row);
    // setup row and 2 iterations per stage, twice
    expect(rows.map((r) => r.Step)).toEqual([0, 1, 2, 2, 3, 4]);
    for (const r of rows) {
      expect(r.RD1).toBe(0);
      expect(r.RD3).toBeCloseTo(1, 12);
      expect(r.MaxReplicaTorque).toBeGreaterThan(0);
      expect(Number.isNaN(r.GradV2dottan)).toBe(true);
      expect(r.EBF).toBeCloseTo(Math.max(r.PE1, r.PE2, r.PE3) - r.PE1, 12);
    }
    expect(events.some((e) => e.kind === 'log' && /^Climbing replica = \d$/.test(e.text))).toBe(true);
  });
});

describe('neb/spin helpers', () => {
  it('geodesic distance is the angle between two unit spins', () => {
    expect(geodesicDistance([1, 0, 0], [0, 1, 0])).toBeCloseTo(Math.PI / 2, 15);
    expect(geodesicDistance([0, 0, 1], [0, 0, -1])).toBeCloseTo(Math.PI, 15);
    expect(() => geodesicDistance([0, 0, 0], [0, 0, 1])).toThrow(/geodesic_distance/);
  });

  it('rotates a spin part of the way along the great circle (Rodrigues)', () => {
    const half = initialRotation([0, 0, 1], [1, 0, 0], 0.5);
    expect(half.rot).toBe(false);
    expect(half.sp[0]).toBeCloseTo(Math.SQRT1_2, 14);
    expect(half.sp[1]).toBeCloseTo(0, 14);
    expect(half.sp[2]).toBeCloseTo(Math.SQRT1_2, 14);
    // antiparallel spins: an arbitrary axis (spi x [100]) is chosen and flagged
    const anti = initialRotation([0, 0, 1], [0, 0, -1], 0.5);
    expect(anti.rot).toBe(true);
    expect(Math.hypot(...anti.sp)).toBeCloseTo(1, 14);
    expect(anti.sp[2]).toBeCloseTo(0, 14);
  });

  it('reads the neb/spin file layout', () => {
    expect(parseSpinFile(FINAL)[1]).toEqual({ id: 2, musp: 2.2, x: [3.5, 5, 5], sp: [0.6, 0, 0.8] });
    expect(() => parseSpinFile('1\n1 2.2 0 0 0 1 0\n')).toThrow(/Incorrect atom format/);
  });
});
