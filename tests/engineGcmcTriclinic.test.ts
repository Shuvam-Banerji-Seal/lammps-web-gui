import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { RanPark } from '../src/engine/rng';
import { boxInsertionPoint } from '../src/engine/fix/widom';
import type { System } from '../src/engine/system';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix gcmc and fix widom in a triclinic box (src/engine/fix/widom.ts boxInsertionPoint, used by gcmc.ts drawPosition).
 * Native fix_gcmc.cpp attempt_atomic_insertion (b332f28, lines 1015-1032) and fix_widom.cpp (lines 556-573) draw three
 * fractional coordinates lamda = uniform() for a triclinic box with no region, map a draw of 1.0 to 0.0 and convert with
 * domain->lamda2x. The insertion point is then inside the tilted cell. The oracle cases w191gt_tri_* check the
 * numbers against native; this file checks the draw order and the cell membership directly.
 */

const runScript = async (text: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  return { session, error };
};

const DIMER = `# dimer
2 atoms
0 bonds

Coords

1 0.0 0.0 0.0
2 1.2 0.0 0.0

Types

1 1
2 1
`;

/** A System-shaped stand-in with only the box that boxInsertionPoint reads. */
const fakeSys = (box: { lo: number[]; hi: number[]; tilt: number[]; triclinic: boolean }): System =>
  ({ state: { box } }) as unknown as System;

describe('boxInsertionPoint (fix gcmc / fix widom, no region)', () => {
  it('draws lamda in order and maps it with lamda2x for a triclinic box', () => {
    const box = { lo: [-1, 2, 0.5], hi: [3, 7, 4.5], tilt: [1.25, -0.75, 0.5], triclinic: true };
    const seed = 4321;
    const p = boxInsertionPoint(fakeSys(box), new RanPark(seed));
    // the same three draws by hand, in native's order
    const ref = new RanPark(seed);
    const l = [ref.uniform(), ref.uniform(), ref.uniform()];
    const lx = 4, ly = 5, lz = 4;
    const [xy, xz, yz] = box.tilt;
    const expected = [
      lx * l[0] + xy * l[1] + xz * l[2] + box.lo[0],
      ly * l[1] + yz * l[2] + box.lo[1],
      lz * l[2] + box.lo[2],
    ];
    expect(p).toEqual(expected);
  });

  it('keeps the point inside the tilted cell (fractional coordinates in [0, 1))', () => {
    const box = { lo: [0, 0, 0], hi: [6, 6, 6], tilt: [2.5, 1.5, -1], triclinic: true };
    const rng = new RanPark(7);
    const sys = fakeSys(box);
    // lamda = h^-1 (p - lo) for h = [[lx, xy, xz], [0, ly, yz], [0, 0, lz]], lx = ly = lz = 6, xy 2.5, xz 1.5, yz -1
    const inv = (p: number[]) => {
      const l2 = p[2] / 6;
      const l1 = (p[1] - (-1) * l2) / 6;
      const l0 = (p[0] - 2.5 * l1 - 1.5 * l2) / 6;
      return [l0, l1, l2];
    };
    for (let k = 0; k < 200; k++) {
      const p = boxInsertionPoint(sys, rng);
      const l = inv(p);
      for (const c of l) {
        expect(c).toBeGreaterThanOrEqual(-1e-12);
        expect(c).toBeLessThan(1 + 1e-12);
      }
    }
  });

  it('draws lo + u (hi - lo) per dimension for an orthogonal box (unchanged)', () => {
    const box = { lo: [0, -2, 1], hi: [5, 3, 4], tilt: [0, 0, 0], triclinic: false };
    const seed = 99;
    const p = boxInsertionPoint(fakeSys(box), new RanPark(seed));
    const ref = new RanPark(seed);
    expect(p).toEqual([
      0 + ref.uniform() * 5,
      -2 + ref.uniform() * 5,
      1 + ref.uniform() * 3,
    ]);
  });
});

describe('fix gcmc triclinic insertion', () => {
  it('inserts every atom inside the tilted prism cell', async () => {
    const { session, error } = await runScript(`units lj
atom_style atomic
region box prism 0 6 0 6 0 6 2.0 1.0 -1.5 units box
create_box 1 box
create_atoms 1 single 0.5 0.5 0.5 units box
mass 1 1.0
pair_style lj/cut 2.0
pair_coeff * * 1.0 1.0
fix g all gcmc 1 20 0 1 4242 1.0 5.0 0.0
thermo_style custom step atoms
thermo 1
run 1
`);
    expect(error).toBeNull();
    const s = session.sys.state;
    expect(s.n).toBeGreaterThan(1);
    const l = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      session.sys.geom.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], l);
      for (const c of l) {
        expect(c).toBeGreaterThanOrEqual(-1e-9);
        expect(c).toBeLessThan(1 + 1e-9);
      }
    }
  });
});

describe('fix gcmc mcmoves real proportions', () => {
  it('accepts fractional proportions on a molecule template and counts the attempts', async () => {
    const { session, error } = await runScript(`units lj
atom_style molecular
pair_style lj/cut 3.0
region box block 0 6 0 6 0 6
create_box 1 box
pair_coeff * * 1.0 1.0
mass 1 1.0
molecule dimer dimer.mol
create_atoms 0 single 1.0 1.0 1.0 mol dimer 12345 units box
fix g all gcmc 1 0 40 0 29494 1.0 -1.0 0.3 mol dimer mcmoves 0 0.3 0.7
run 1
`, { 'dimer.mol': DIMER });
    expect(error).toBeNull();
    const f = session.sys.fixes.find((x) => x.id === 'g');
    expect(f).toBeDefined();
    const v = (i: number) => (f as unknown as { computeVector(k: number): number }).computeVector(i);
    // every cycle is a translation (index 0) or a rotation (index 6); no exchanges with X = 0
    expect(v(0) + v(6)).toBe(40);
    expect(v(2)).toBe(0);
    expect(v(4)).toBe(0);
  });

  it('refuses a negative proportion and an all-zero set, and parses a real one', async () => {
    const base = (mc: string) => `units lj
atom_style atomic
region box block 0 5 0 5 0 5
create_box 1 box
pair_style lj/cut 3.0
pair_coeff * * 1.0 1.0
mass 1 1.0
create_atoms 1 single 1.0 1.0 1.0 units box
fix g all gcmc 1 0 1 1 29494 2.0 -1.0 0.0 mcmoves ${mc}
run 0
`;
    const neg = await runScript(base('-0.5 1 0'));
    expect(neg.error).toMatch(/Illegal fix gcmc mcmoves parameters/);
    const zero = await runScript(base('0 0 0'));
    expect(zero.error).toMatch(/at least one proportion must be non-zero/);
    // 0.5 0.25 0.25 parses; the refusal that follows is the mol-only rule for molecule moves (not a parse error)
    const frac = await runScript(base('0.5 0.25 0.25'));
    expect(frac.error).toMatch(/molecule translation\/rotation moves need the mol keyword/);
    const atomsOnly = await runScript(base('0.5 0 0'));
    expect(atomsOnly.error).toBeNull();
  });
});
