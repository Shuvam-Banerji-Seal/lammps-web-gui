import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix qeq/slater with pair coul/streitz (fix_qeq.html, qeq/slater section; the model and the solver are
 * measured with native LAMMPS, see the header comment of src/engine/fix/qeq.ts). Parsing and refusals,
 * the stationarity of the coul/streitz energy at the solved charges (dE/dq_i equal for all atoms, by
 * central differences of the energy at fixed charges), and charge conservation.
 */

const POT = readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', 'AlO.streitz'), 'utf8');

const cluster = (pairLine: string, fixLine: string): string => `units metal
atom_style charge
boundary p p p
region box block -50 50 -50 50 -50 50
create_box 2 box
create_atoms 1 single 0 0 0
create_atoms 2 single 2.0 0.3 -0.2
create_atoms 2 single 0.5 2.2 1.0
create_atoms 1 single -1.5 1.0 0.5
mass * 1.0
${pairLine}
pair_coeff * * AlO.streitz Al O
${fixLine}
run 0
`;

const WOLF = 'pair_style coul/streitz 12.0 wolf 0.2';
// The pair style's wolf shifts (erfc(alpha rc)/rc on the pairs and the self term, about 6e-4 at 12 A) are not part of
// the fix's energy (measured with native LAMMPS, see fix/qeq.ts). With a 40 A pair cutoff and a 40 A fix cutoff the
// shifts vanish, so the energy of the pe is the energy the fix minimises.
const WOLF_LONG = 'pair_style coul/streitz 40.0 wolf 0.2';
const FIX_LONG = 'fix 1 all qeq/slater 1 40.0 1.0e-12 500 coul/streitz';
const FIX = 'fix 1 all qeq/slater 1 12.0 1.0e-12 500 coul/streitz';

const newSession = (): Session => {
  const s = new Session({ emit: () => {} });
  s.addFile('AlO.streitz', POT);
  return s;
};

const chargesOf = async (input: string): Promise<number[]> => {
  const s = newSession();
  await s.execute(input);
  const q = (s as unknown as { sys: { state: { q: Float64Array } } }).sys.state.q;
  return Array.from(q.subarray(0, 4));
};

/** pe at the given charges (no fix): the coul/streitz energy as a function of q. */
const energyAt = async (q: number[]): Promise<number> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e) });
  s.addFile('AlO.streitz', POT);
  const sets = q.map((v, i) => `set atom ${i + 1} charge ${v}`).join('\n');
  await s.execute(cluster(WOLF_LONG, '').replace('run 0', `${sets}\nthermo_style custom step pe\nthermo_modify format float %.15g\nrun 0\n`));
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return rows[rows.length - 1].pe as number;
};

describe('fix qeq/slater: solution and stationarity', () => {
  it('conserves the total charge of the group', async () => {
    const q = await chargesOf(cluster(WOLF, FIX));
    expect(q.reduce((a, b) => a + b, 0)).toBeCloseTo(0, 12);
  });

  it('makes the coul/streitz energy stationary under charge-conserving variations', async () => {
    const q = await chargesOf(cluster(WOLF_LONG, FIX_LONG));
    const h = 1e-5;
    const pairs: [number, number][] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    for (const [i, j] of pairs) {
      const plus = [...q], minus = [...q];
      plus[i] += h; plus[j] -= h;
      minus[i] -= h; minus[j] += h;
      const slope = (await energyAt(plus) - (await energyAt(minus))) / (2 * h);
      // dE/dq_i - dE/dq_j = 0 at the solution (the Lagrange multiplier is common to all atoms)
      expect(Math.abs(slope)).toBeLessThan(1e-5);
    }
  });

  it('the default alpha is 0.2 (as the explicit keyword)', async () => {
    const a = await chargesOf(cluster(WOLF, FIX));
    const b = await chargesOf(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-12 500 coul/streitz alpha 0.2'));
    for (let i = 0; i < 4; i++) expect(a[i]).toBeCloseTo(b[i], 12);
  });

  it('alpha changes the solution (it is the damping, not the Slater exponent)', async () => {
    const a = await chargesOf(cluster(WOLF, FIX));
    const b = await chargesOf(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-12 500 coul/streitz alpha 0.31'));
    expect(Math.abs(a[0] - b[0])).toBeGreaterThan(1e-3);
  });

  it('dsf 0.0 is accepted and changes the solution', async () => {
    const a = await chargesOf(cluster(WOLF, FIX));
    const b = await chargesOf(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-12 500 coul/streitz alpha 0.2 dsf 0.0'));
    expect(Number.isFinite(b[0])).toBe(true);
    expect(b.length).toBe(4);
    expect(a[0]).not.toBe(b[0]);
  });
});

describe('fix qeq/slater: parsing and refusals', () => {
  it('refuses the wolf keyword by name', async () => {
    await expect(newSession().execute(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-6 100 coul/streitz alpha 0.2 wolf 0.0'))).rejects.toThrow(/wolf/);
  });

  it('accepts dsf with a taper width (the wolf keyword stays refused)', async () => {
    await newSession().execute(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-6 100 coul/streitz alpha 0.2 dsf 2.0'));
  });

  it('refuses dsf at the top level (only alpha takes it as a sub-keyword)', async () => {
    await expect(newSession().execute(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-6 100 coul/streitz dsf 0.0'))).rejects.toThrow(/dsf/);
  });

  it('refuses a non-positive alpha', async () => {
    await expect(newSession().execute(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-6 100 coul/streitz alpha 0'))).rejects.toThrow(/alpha/);
  });

  it('qfile coul/streitz needs an active coul/streitz pair style', async () => {
    const input = cluster('pair_style coul/cut 20.0', 'fix 1 all qeq/slater 1 12.0 1.0e-6 100 coul/streitz').replace('pair_coeff * * AlO.streitz Al O', 'pair_coeff * *');
    await expect(newSession().execute(input)).rejects.toThrow(/coul\/streitz/);
  });

  it('refuses the reaxff and coul/ctip qfile words', async () => {
    await expect(newSession().execute(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-6 100 reaxff'))).rejects.toThrow(/reaxff/);
    await expect(newSession().execute(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-6 100 coul/ctip'))).rejects.toThrow(/coul\/ctip/);
  });

  it('an eta of zero is refused (the preconditioner is 1/eta)', async () => {
    const s = newSession();
    s.addFile('zeroeta.params', '1 0.0 0.0 0.0 0.968438 0.763905\n2 5.484763 14.035715 0.0 2.143957 0.0\n');
    await expect(s.execute(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-6 100 zeroeta.params'))).rejects.toThrow(/eta/);
  });

  it('reads a qfile with the eta column (the fix uses eta on its diagonal, not the pair J)', async () => {
    const s = newSession();
    s.addFile('eta.params', '1 0.0 12.5 0.0 0.968438 0.763905\n2 5.484763 14.035715 0.0 2.143957 0.0\n');
    await s.execute(cluster(WOLF, 'fix 1 all qeq/slater 1 12.0 1.0e-12 500 eta.params'));
    const q = Array.from((s as unknown as { sys: { state: { q: Float64Array } } }).sys.state.q.subarray(0, 4));
    const ref = await chargesOf(cluster(WOLF, FIX));
    expect(Math.abs(q[0] - ref[0])).toBeGreaterThan(1e-3);
  });
});
