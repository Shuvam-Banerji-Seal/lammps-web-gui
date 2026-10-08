import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * velocity create/scale with "bias yes temp <compute>" — measured with native
 * LAMMPS as a black box (tests/oracle/w26velbias_*.in reproduce the parity):
 * `create` rescales the freshly generated ensemble to the target with the
 * internal plain compute (the *temp* keyword does not enter the scale factor),
 * then applies the temperature compute's bias to the rescaled velocities
 * without restoring it — temp/cs and temp/partial remove it, temp/ramp adds
 * the ramp back, temp/com and temp/region leave the velocities unchanged.
 * `scale` keeps the documented remove/scale/restore sequence.
 */

const DATA = `w26 velocity bias unit test (4 core/shell pairs + 2 ions)

10 atoms
4 bonds
0 angles
0 dihedrals

3 atom types
1 bond types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 20.0
2 2.0
3 25.0

Atoms

1 1 1 1.0 2.0 2.0 2.0
2 1 2 -1.0 2.1 2.0 2.0
3 2 1 1.0 6.0 2.0 2.0
4 2 2 -1.0 6.1 2.0 2.0
5 3 1 1.0 2.0 6.0 2.0
6 3 2 -1.0 2.1 6.0 2.0
7 4 1 1.0 2.0 2.0 6.0
8 4 2 -1.0 2.1 2.0 6.0
9 5 3 0.0 14.0 14.0 14.0
10 5 3 0.0 14.0 14.0 15.0

Bonds

1 1 1 2
2 1 3 4
3 1 5 6
4 1 7 8
`;

const HEAD = `
units metal
atom_style full
boundary p p p
read_data w26.data
comm_modify vel yes
group cores type 1
group shells type 2
pair_style zero 10.0
pair_coeff * *
bond_style harmonic
bond_coeff 1 100.0 0.1
compute CStemp all temp/cs cores shells
compute Txy all temp/partial 1 1 0
compute Tz all temp/partial 0 0 1
region left block 0 4 0 20 0 20
compute Treg all temp/region left
compute Tramp all temp/ramp vx 0.0 2.0 x 0.0 20.0 units box
compute Tcom all temp/com
thermo_style custom step temp c_CStemp c_Tz c_Tramp
thermo_modify format float %.15g
thermo 1
`;

interface Atom { id: number; x: number; vx: number; vy: number; vz: number }

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => files.set(n, t) });
  session.addFile('w26.data', DATA);
  await session.execute(text);
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { thermo, files };
};

const velocities = (files: Map<string, string>): Atom[] => {
  const lines = (files.get('out.dump') ?? '').trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  return lines.slice(k + 1).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    const a: Record<string, number> = {};
    cols.forEach((c, i) => (a[c] = w[i]));
    return a as unknown as Atom;
  }).sort((a, b) => a.id - b.id);
};

const MASS: Record<number, number> = { 1: 20, 2: 2, 3: 25 };
const PAIRS: [number, number][] = [[1, 2], [3, 4], [5, 6], [7, 8]];
const X = [2, 2.1, 6, 6.1, 2, 2.1, 2, 2.1, 14, 14];

const create = (mom: 'yes' | 'no', bias?: string) =>
  `velocity all set 0.0 0.0 0.0 units box
velocity all create 1427.0 134 dist gaussian mom ${mom} rot no${bias ? ` bias yes temp ${bias}` : ''}
run 0
write_dump all custom out.dump id x vx vy vz modify format float %.17g sort id
`;

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-10 + 1e-9 * Math.max(Math.abs(a), Math.abs(b));

describe('velocity create/scale bias handling', () => {
  it('create bias yes temp/cs equals a plain create with the pair-COM velocity', async () => {
    const plain = await runScript(HEAD + create('yes'));
    const bias = await runScript(HEAD + create('yes', 'CStemp'));
    const p = velocities(plain.files), b = velocities(bias.files);
    for (const [ia, ib] of PAIRS) {
      const a = p.find((t) => t.id === ia)!, c = p.find((t) => t.id === ib)!;
      const u = (d: 'vx' | 'vy' | 'vz') => (MASS[1] * a[d] + MASS[2] * c[d]) / (MASS[1] + MASS[2]);
      const x = b.find((t) => t.id === ia)!, y = b.find((t) => t.id === ib)!;
      for (const d of ['vx', 'vy', 'vz'] as const) {
        expect(close(x[d], u(d)), `atom ${ia}.${d}`).toBe(true);
        expect(close(y[d], u(d)), `atom ${ib}.${d}`).toBe(true);
      }
    }
    // the plain create rescales to 1427; with bias yes the *temp* keyword does
    // not drive the scale, so temp/cs keeps the plain pair-COM value (not 1427)
    expect(plain.thermo[0].temp).toBeCloseTo(1427, 8);
    expect(bias.thermo[0].c_CStemp).toBeCloseTo(plain.thermo[0].c_CStemp as number, 8);
    expect(Math.abs((bias.thermo[0].temp as number) - 1427)).toBeGreaterThan(1);
  });

  it('create bias yes temp/partial zeroes the excluded components', async () => {
    const plain = await runScript(HEAD + create('yes'));
    const bias = await runScript(HEAD + create('yes', 'Tz'));
    const p = velocities(plain.files), b = velocities(bias.files);
    for (let i = 0; i < p.length; i++) {
      expect(b[i].vx).toBeCloseTo(0, 12);
      expect(b[i].vy).toBeCloseTo(0, 12);
      expect(close(b[i].vz, p[i].vz)).toBe(true);
    }
    // without bias the *temp* keyword does drive the scale (z-only field -> 1427)
    const noBias = await runScript(HEAD + create('yes', 'Tz').replace(' bias yes temp Tz', ' temp Tz'));
    const t = noBias.thermo[0] as Record<string, number>;
    expect(t.c_Tz).toBeCloseTo(1427, 8);
  });

  it('create bias yes temp/ramp adds the ramp to the plain field', async () => {
    const plain = await runScript(HEAD + create('yes'));
    const bias = await runScript(HEAD + create('yes', 'Tramp'));
    const p = velocities(plain.files), b = velocities(bias.files);
    for (let i = 0; i < p.length; i++) {
      const ramp = X[i] / 10;
      expect(close(b[i].vx, p[i].vx + ramp), `atom ${i + 1} vx`).toBe(true);
      expect(close(b[i].vy, p[i].vy)).toBe(true);
      expect(close(b[i].vz, p[i].vz)).toBe(true);
    }
  });

  it('create bias yes temp/com and temp/region leave the plain field unchanged', async () => {
    const plain = await runScript(HEAD + create('no'));
    for (const c of ['Tcom', 'Treg']) {
      const bias = await runScript(HEAD + create('no', c));
      const p = velocities(plain.files), b = velocities(bias.files);
      for (let i = 0; i < p.length; i++) {
        expect(close(b[i].vx, p[i].vx), `${c} atom ${i + 1} vx`).toBe(true);
        expect(close(b[i].vy, p[i].vy), `${c} atom ${i + 1} vy`).toBe(true);
        expect(close(b[i].vz, p[i].vz), `${c} atom ${i + 1} vz`).toBe(true);
      }
    }
  });

  it('scale bias yes temp/cs rescales the pair-COM frame to the target', async () => {
    const { thermo } = await runScript(`${HEAD}
velocity all set 0.0 0.0 0.0 units box
velocity cores set 1.0 0.0 0.0 units box
velocity shells set -1.0 0.0 0.0 units box
velocity all scale 10.0 bias yes temp CStemp
run 0
`);
    expect(thermo[0].c_CStemp).toBeCloseTo(10, 8);
  });
});
