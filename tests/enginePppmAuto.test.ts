import { describe, expect, it } from 'vitest';
import { selectPppmAuto } from '../src/engine/force/kspace/pppm';

/*
 * Automatic PPPM G-ewald and mesh (selectPppmAuto). Every row is a native
 * LAMMPS setup measured as a black box: the log line of run 0 printed the G
 * vector (8 significant digits, so rows match to about 1e-7 relative) and the
 * grid. Inputs: natoms atoms, half of them q = +-q with q2 = qqrd2e * sum q^2,
 * accE = accuracy * qqrd2e (absolute force of two unit charges 1 Angstrom
 * apart), order P, cutoff rc. Charges are random positions in a periodic box.
 * Measured with native LAMMPS (black box): see the rows below.
 */
interface Row { name: string; box: number[]; natoms: number; q2: number; rc: number; accE: number; order: number; g: number; grid: number[] }

const CUBIC: Row[] = [
  { name: 'c_a0.001_L8_N50', box: [8, 8, 8], natoms: 100, q2: 100, rc: 2.5, accE: 0.001, order: 5, g: 1.0437345, grid: [16, 16, 16] },
  { name: 'c_a0.001_L8_N200', box: [8, 8, 8], natoms: 400, q2: 400, rc: 2.5, accE: 0.001, order: 5, g: 1.0835943, grid: [18, 18, 18] },
  { name: 'c_a0.001_L10_N50', box: [10, 10, 10], natoms: 100, q2: 100, rc: 2.5, accE: 0.001, order: 5, g: 1.0071255, grid: [18, 18, 18] },
  { name: 'c_a0.001_L10_N200', box: [10, 10, 10], natoms: 400, q2: 400, rc: 2.5, accE: 0.001, order: 5, g: 1.1048821, grid: [24, 24, 24] },
  { name: 'c_a0.001_L14_N50', box: [14, 14, 14], natoms: 100, q2: 100, rc: 2.5, accE: 0.001, order: 5, g: 0.98969436, grid: [24, 24, 24] },
  { name: 'c_a0.001_L14_N200', box: [14, 14, 14], natoms: 400, q2: 400, rc: 2.5, accE: 0.001, order: 5, g: 1.0312026, grid: [27, 27, 27] },
  { name: 'c_a0.001_L20_N50', box: [20, 20, 20], natoms: 100, q2: 100, rc: 2.5, accE: 0.001, order: 5, g: 0.941466, grid: [30, 30, 30] },
  { name: 'c_a0.001_L20_N200', box: [20, 20, 20], natoms: 400, q2: 400, rc: 2.5, accE: 0.001, order: 5, g: 0.96459446, grid: [32, 32, 32] },
  { name: 'c_a0.0001_L8_N50', box: [8, 8, 8], natoms: 100, q2: 100, rc: 2.5, accE: 0.0001, order: 5, g: 1.1872725, grid: [25, 25, 25] },
  { name: 'c_a0.0001_L8_N200', box: [8, 8, 8], natoms: 400, q2: 400, rc: 2.5, accE: 0.0001, order: 5, g: 1.2398674, grid: [30, 30, 30] },
  { name: 'c_a0.0001_L10_N50', box: [10, 10, 10], natoms: 100, q2: 100, rc: 2.5, accE: 0.0001, order: 5, g: 1.1731428, grid: [30, 30, 30] },
  { name: 'c_a0.0001_L10_N200', box: [10, 10, 10], natoms: 400, q2: 400, rc: 2.5, accE: 0.0001, order: 5, g: 1.2255254, grid: [36, 36, 36] },
  { name: 'c_a0.0001_L14_N50', box: [14, 14, 14], natoms: 100, q2: 100, rc: 2.5, accE: 0.0001, order: 5, g: 1.1267675, grid: [36, 36, 36] },
  { name: 'c_a0.0001_L14_N200', box: [14, 14, 14], natoms: 400, q2: 400, rc: 2.5, accE: 0.0001, order: 5, g: 1.1920279, grid: [45, 45, 45] },
  { name: 'c_a0.0001_L20_N50', box: [20, 20, 20], natoms: 100, q2: 100, rc: 2.5, accE: 0.0001, order: 5, g: 1.1020539, grid: [48, 48, 48] },
  { name: 'c_a0.0001_L20_N200', box: [20, 20, 20], natoms: 400, q2: 400, rc: 2.5, accE: 0.0001, order: 5, g: 1.1418331, grid: [54, 54, 54] },
  { name: 'c_a1e-05_L8_N50', box: [8, 8, 8], natoms: 100, q2: 100, rc: 2.5, accE: 0.00001, order: 5, g: 1.3478416, grid: [45, 45, 45] },
  { name: 'c_a1e-05_L8_N200', box: [8, 8, 8], natoms: 400, q2: 400, rc: 2.5, accE: 0.00001, order: 5, g: 1.3696985, grid: [48, 48, 48] },
  { name: 'c_a1e-05_L10_N50', box: [10, 10, 10], natoms: 100, q2: 100, rc: 2.5, accE: 0.00001, order: 5, g: 1.3105675, grid: [48, 48, 48] },
  { name: 'c_a1e-05_L10_N200', box: [10, 10, 10], natoms: 400, q2: 400, rc: 2.5, accE: 0.00001, order: 5, g: 1.365991, grid: [60, 60, 60] },
  { name: 'c_a1e-05_L14_N50', box: [14, 14, 14], natoms: 100, q2: 100, rc: 2.5, accE: 0.00001, order: 5, g: 1.2794175, grid: [60, 60, 60] },
  { name: 'c_a1e-05_L14_N200', box: [14, 14, 14], natoms: 400, q2: 400, rc: 2.5, accE: 0.00001, order: 5, g: 1.328115, grid: [72, 72, 72] },
  { name: 'c_a1e-05_L20_N50', box: [20, 20, 20], natoms: 100, q2: 100, rc: 2.5, accE: 0.00001, order: 5, g: 1.2418385, grid: [75, 75, 75] },
  { name: 'c_a1e-05_L20_N200', box: [20, 20, 20], natoms: 400, q2: 400, rc: 2.5, accE: 0.00001, order: 5, g: 1.2925015, grid: [90, 90, 90] },
  { name: 'c_a1e-06_L8_N50', box: [8, 8, 8], natoms: 100, q2: 100, rc: 2.5, accE: 0.000001, order: 5, g: 1.4685811, grid: [72, 72, 72] },
  { name: 'c_a1e-06_L8_N200', box: [8, 8, 8], natoms: 400, q2: 400, rc: 2.5, accE: 0.000001, order: 5, g: 1.4976097, grid: [81, 81, 81] },
  { name: 'c_a1e-06_L10_N50', box: [10, 10, 10], natoms: 100, q2: 100, rc: 2.5, accE: 0.000001, order: 5, g: 1.4416205, grid: [80, 80, 80] },
  { name: 'c_a1e-06_L10_N200', box: [10, 10, 10], natoms: 400, q2: 400, rc: 2.5, accE: 0.000001, order: 5, g: 1.4845536, grid: [96, 96, 96] },
  { name: 'c_a1e-06_L14_N50', box: [14, 14, 14], natoms: 100, q2: 100, rc: 2.5, accE: 0.000001, order: 5, g: 1.4137718, grid: [100, 100, 100] },
  { name: 'c_a1e-06_L14_N200', box: [14, 14, 14], natoms: 400, q2: 400, rc: 2.5, accE: 0.000001, order: 5, g: 1.4575745, grid: [120, 120, 120] },
  { name: 'c_a1e-06_L20_N50', box: [20, 20, 20], natoms: 100, q2: 100, rc: 2.5, accE: 0.000001, order: 5, g: 1.3803375, grid: [125, 125, 125] },
  { name: 'c_a1e-06_L20_N200', box: [20, 20, 20], natoms: 400, q2: 400, rc: 2.5, accE: 0.000001, order: 5, g: 1.4255259, grid: [150, 150, 150] },
  { name: 'c_rc2', box: [12, 12, 12], natoms: 300, q2: 300, rc: 2, accE: 0.0001, order: 5, g: 1.4924372, grid: [48, 48, 48] },
  { name: 'c_rc3.5', box: [12, 12, 12], natoms: 300, q2: 300, rc: 3.5, accE: 0.0001, order: 5, g: 0.84723314, grid: [27, 27, 27] },
  { name: 'c_rc5', box: [12, 12, 12], natoms: 300, q2: 300, rc: 5, accE: 0.0001, order: 5, g: 0.58601024, grid: [18, 18, 18] },
  { name: 'c_q2', box: [10, 10, 10], natoms: 600, q2: 54, rc: 2.5, accE: 0.0001, order: 5, g: 1.1040547, grid: [24, 24, 24] },
  { name: 'c_q3', box: [10, 10, 10], natoms: 200, q2: 800, rc: 2.5, accE: 0.0001, order: 5, g: 1.2895783, grid: [45, 45, 45] },
  { name: 'c_order4', box: [10, 10, 10], natoms: 400, q2: 400, rc: 2.5, accE: 0.0001, order: 4, g: 1.2106659, grid: [48, 48, 48] },
  { name: 'c_order3', box: [10, 10, 10], natoms: 400, q2: 400, rc: 2.5, accE: 0.0001, order: 3, g: 1.2047502, grid: [96, 96, 96] },
  { name: 'c_order7', box: [10, 10, 10], natoms: 400, q2: 400, rc: 2.5, accE: 0.0001, order: 7, g: 1.228972, grid: [27, 27, 27] },
  { name: 'c_real', box: [30, 30, 30], natoms: 400, q2: 33206.371, rc: 10, accE: 0.033206371, order: 5, g: 0.24126287, grid: [12, 12, 12] },
  { name: 'c_metal', box: [30, 30, 30], natoms: 400, q2: 3686.30912, rc: 8, accE: 0.0014399645, order: 5, g: 0.32178055, grid: [18, 18, 18] },
  { name: 'c_coul', box: [10, 10, 10], natoms: 400, q2: 400, rc: 2.5, accE: 0.0001, order: 5, g: 1.2255254, grid: [36, 36, 36] },
];

// orthorhombic, non-cubic edges: the mesh rule is matched; the G rule is not (see pppm.ts)
const ORTHO: Row[] = [
  { name: 'c_elong', box: [8, 8, 30], natoms: 300, q2: 300, rc: 2.5, accE: 0.0001, order: 5, g: 1.1792915, grid: [27, 27, 75] },
  { name: 'c_elong2', box: [6, 12, 30], natoms: 400, q2: 196, rc: 2.5, accE: 0.00001, order: 5, g: 1.2934107, grid: [36, 54, 108] },
  { name: 'c_real2', box: [40, 30, 25], natoms: 600, q2: 97626.73074, rc: 12, accE: 0.0033206371, order: 5, g: 0.24800349, grid: [24, 20, 18] },
];

const choose = (r: Pick<Row, 'box' | 'natoms' | 'q2' | 'rc' | 'accE' | 'order'>, gewald = 0, mesh: [number, number, number] | null = null) =>
  selectPppmAuto({ lx: r.box[0], ly: r.box[1], lz: r.box[2], natoms: r.natoms, q2: r.q2, rc: r.rc, accE: r.accE, order: r.order, gUser: gewald, meshUser: mesh });

describe('PPPM automatic G-ewald and mesh (native-measured)', () => {
  it('has at least 20 measured cubic systems', () => {
    expect(CUBIC.length).toBeGreaterThanOrEqual(20);
  });

  for (const r of CUBIC) {
    it(`matches native grid and G for ${r.name}`, () => {
      const got = choose(r);
      expect(got.mesh).toEqual(r.grid);
      expect(Math.abs(got.g / r.g - 1)).toBeLessThan(1e-7);
    });
  }

  for (const r of ORTHO) {
    it(`matches native grid for orthorhombic ${r.name}`, () => {
      const got = choose(r);
      expect(got.mesh).toEqual(r.grid);
      expect(Math.abs(got.g / r.g - 1)).toBeLessThan(3e-2);
    });
  }

  it('keeps the first branch of the initial G (accuracy >= real-space prefactor)', () => {
    // Measured with native LAMMPS (black box): run 0 of 20 charges in a 10 box with accuracy 0.5 gives G 0.59236144 and grid 6 6 6.
    const got = selectPppmAuto({ lx: 10, ly: 10, lz: 10, natoms: 20, q2: 20, rc: 2.5, accE: 0.5, order: 5, gUser: 0, meshUser: null });
    expect(got.mesh).toEqual([6, 6, 6]);
    expect(Math.abs(got.g / 0.59236144 - 1)).toBeLessThan(1e-7);
  });

  it('chooses the mesh from a pinned G', () => {
    // Measured with native LAMMPS (black box): kspace_modify gewald 0.8 with accuracy 1e-4 in a 10 box of 200 atoms gives grid 24 24 24.
    const got = choose({ box: [10, 10, 10], natoms: 200, q2: 200, rc: 2.5, accE: 1e-4, order: 5 }, 0.8);
    expect(got.g).toBe(0.8);
    expect(got.mesh).toEqual([24, 24, 24]);
  });

  it('solves G for a pinned mesh', () => {
    // Measured with native LAMMPS (black box): kspace_modify mesh 24 24 24 with the same system gives G 1.1045955.
    const got = choose({ box: [10, 10, 10], natoms: 200, q2: 200, rc: 2.5, accE: 1e-4, order: 5 }, 0, [24, 24, 24]);
    expect(got.mesh).toEqual([24, 24, 24]);
    expect(Math.abs(got.g / 1.1045955 - 1)).toBeLessThan(1e-7);
  });

  it('uses the order in the error estimate (real units, order 3)', () => {
    // Measured with native LAMMPS (black box): real units, 20 box, 200 atoms with charge 0.6, cutoff 8, accuracy 1e-4, order 3 gives G 0.31380752 and grid 20 20 20.
    const got = choose({ box: [20, 20, 20], natoms: 200, q2: 332.06371 * 200 * 0.36, rc: 8, accE: 1e-4 * 332.06371, order: 3 });
    expect(got.mesh).toEqual([20, 20, 20]);
    expect(Math.abs(got.g / 0.31380752 - 1)).toBeLessThan(1e-7);
  });
});
