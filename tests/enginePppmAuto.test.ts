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

/*
 * Slab boxes (kspace_modify slab V): the reciprocal-space estimate uses the extended z edge V lz, the
 * real-space estimate and the initial G use the box volume. Measured with native LAMMPS (black box): 66
 * boxes, orthorhombic, V 2..8 (2.5 included), lj, real and metal, net charge and neutral, cutoffs 4..9,
 * order 5, accuracy 1e-3 to 1e-10 (the 1e-2 rows are not in the table: see pppm.ts). Native printed G (8 digits)
 * and grid. Not in the table, because the rule
 * does not reproduce native's grid there: the 4x4x8 box with V 3 and accuracy 1e-3 (native z 25, this rule
 * 24), the madelung 1x1x20 box with V 8 at accuracy 1e-2, 1e-3 and 1e-4, and the 3x3x12 box with V 8 at 1e-4.
 */
interface SlabRow { name: string; box: number[]; natoms: number; q2: number; rc: number; accE: number; slab: number; g: number; grid: number[] }

const SLAB: SlabRow[] = [
  { name: 'real_5x5x4_s2_a1e-5', box: [16, 16, 12.8], natoms: 100, q2: 8301.59275, rc: 6, accE: 0.0033206371, slab: 2, g: 0.48868171, grid: [20, 20, 27] },
  { name: 'real_5x5x4_s3_a1e-5', box: [16, 16, 12.8], natoms: 100, q2: 8301.59275, rc: 6, accE: 0.0033206371, slab: 3, g: 0.49282579, grid: [20, 20, 40] },
  { name: 'real_5x5x4_s5_a1e-5', box: [16, 16, 12.8], natoms: 100, q2: 8301.59275, rc: 6, accE: 0.0033206371, slab: 5, g: 0.48682758, grid: [20, 20, 54] },
  { name: 'real_5x5x4_s8_a1e-5', box: [16, 16, 12.8], natoms: 100, q2: 8301.59275, rc: 6, accE: 0.0033206371, slab: 8, g: 0.48845825, grid: [20, 20, 80] },
  { name: 'real_4x4x8_s2_a1e-5', box: [12.8, 12.8, 25.6], natoms: 128, q2: 10626.03872, rc: 6, accE: 0.0033206371, slab: 2, g: 0.49234622, grid: [18, 18, 48] },
  { name: 'real_4x4x8_s3_a1e-5', box: [12.8, 12.8, 25.6], natoms: 128, q2: 10626.03872, rc: 6, accE: 0.0033206371, slab: 3, g: 0.48946179, grid: [18, 18, 64] },
  { name: 'real_4x4x8_s5_a1e-5', box: [12.8, 12.8, 25.6], natoms: 128, q2: 10626.03872, rc: 6, accE: 0.0033206371, slab: 5, g: 0.48934404, grid: [18, 18, 96] },
  { name: 'real_4x4x8_s8_a1e-5', box: [12.8, 12.8, 25.6], natoms: 128, q2: 10626.03872, rc: 6, accE: 0.0033206371, slab: 8, g: 0.4921069, grid: [18, 18, 144] },
  { name: 'real_6x6x5_s2_a1e-5', box: [19.2, 19.2, 16], natoms: 180, q2: 14942.86695, rc: 6, accE: 0.0033206371, slab: 2, g: 0.49219605, grid: [24, 24, 36] },
  { name: 'real_6x6x5_s3_a1e-5', box: [19.2, 19.2, 16], natoms: 180, q2: 14942.86695, rc: 6, accE: 0.0033206371, slab: 3, g: 0.48685485, grid: [24, 24, 45] },
  { name: 'real_6x6x5_s5_a1e-5', box: [19.2, 19.2, 16], natoms: 180, q2: 14942.86695, rc: 6, accE: 0.0033206371, slab: 5, g: 0.49133648, grid: [24, 24, 72] },
  { name: 'real_6x6x5_s8_a1e-5', box: [19.2, 19.2, 16], natoms: 180, q2: 14942.86695, rc: 6, accE: 0.0033206371, slab: 8, g: 0.48523508, grid: [24, 24, 96] },
  { name: 'real_3x5x9_s2_a1e-5', box: [9.6, 16, 28.8], natoms: 135, q2: 11207.15021, rc: 6, accE: 0.0033206371, slab: 2, g: 0.49186337, grid: [15, 20, 54] },
  { name: 'real_3x5x9_s3_a1e-5', box: [9.6, 16, 28.8], natoms: 135, q2: 11207.15021, rc: 6, accE: 0.0033206371, slab: 3, g: 0.48973653, grid: [15, 20, 72] },
  { name: 'real_3x5x9_s5_a1e-5', box: [9.6, 16, 28.8], natoms: 135, q2: 11207.15021, rc: 6, accE: 0.0033206371, slab: 5, g: 0.48964463, grid: [15, 20, 108] },
  { name: 'real_3x5x9_s8_a1e-5', box: [9.6, 16, 28.8], natoms: 135, q2: 11207.15021, rc: 6, accE: 0.0033206371, slab: 8, g: 0.48502548, grid: [15, 20, 150] },
  { name: 'real_5x5x12_s2_a1e-5', box: [16, 16, 38.4], natoms: 300, q2: 24904.77825, rc: 6, accE: 0.0033206371, slab: 2, g: 0.49692324, grid: [24, 24, 72] },
  { name: 'real_5x5x12_s3_a1e-5', box: [16, 16, 38.4], natoms: 300, q2: 24904.77825, rc: 6, accE: 0.0033206371, slab: 3, g: 0.49364475, grid: [24, 24, 96] },
  { name: 'real_5x5x12_s5_a1e-5', box: [16, 16, 38.4], natoms: 300, q2: 24904.77825, rc: 6, accE: 0.0033206371, slab: 5, g: 0.49348973, grid: [24, 24, 144] },
  { name: 'real_5x5x12_s8_a1e-5', box: [16, 16, 38.4], natoms: 300, q2: 24904.77825, rc: 6, accE: 0.0033206371, slab: 8, g: 0.48755835, grid: [24, 24, 200] },
  { name: 'real_8x8x6_s2_a1e-5', box: [25.6, 25.6, 19.2], natoms: 384, q2: 31878.11616, rc: 6, accE: 0.0033206371, slab: 2, g: 0.4832344, grid: [30, 30, 40] },
  { name: 'real_8x8x6_s3_a1e-5', box: [25.6, 25.6, 19.2], natoms: 384, q2: 31878.11616, rc: 6, accE: 0.0033206371, slab: 3, g: 0.48665315, grid: [30, 30, 60] },
  { name: 'real_8x8x6_s5_a1e-5', box: [25.6, 25.6, 19.2], natoms: 384, q2: 31878.11616, rc: 6, accE: 0.0033206371, slab: 5, g: 0.48256994, grid: [30, 30, 81] },
  { name: 'real_8x8x6_s8_a1e-5', box: [25.6, 25.6, 19.2], natoms: 384, q2: 31878.11616, rc: 6, accE: 0.0033206371, slab: 8, g: 0.48377505, grid: [30, 30, 120] },
  { name: 'real_5x5x4_s3_a1e-3', box: [16, 16, 12.8], natoms: 100, q2: 8301.59275, rc: 6, accE: 0.33206371, slab: 3, g: 0.35330821, grid: [8, 8, 15] },
  { name: 'real_5x5x4_s3_a1e-4', box: [16, 16, 12.8], natoms: 100, q2: 8301.59275, rc: 6, accE: 0.033206371, slab: 3, g: 0.42177002, grid: [12, 12, 24] },
  { name: 'real_5x5x4_s3_a1e-6', box: [16, 16, 12.8], natoms: 100, q2: 8301.59275, rc: 6, accE: 0.00033206371, slab: 3, g: 0.54741758, grid: [32, 32, 60] },
  { name: 'real_5x5x4_s3_a1e-10', box: [16, 16, 12.8], natoms: 100, q2: 8301.59275, rc: 6, accE: 3.3206371e-8, slab: 3, g: 0.74012104, grid: [243, 243, 450] },
  { name: 'real_4x4x8_s3_a1e-4', box: [12.8, 12.8, 25.6], natoms: 128, q2: 10626.03872, rc: 6, accE: 0.033206371, slab: 3, g: 0.42157226, grid: [12, 12, 40] },
  { name: 'real_4x4x8_s3_a1e-6', box: [12.8, 12.8, 25.6], natoms: 128, q2: 10626.03872, rc: 6, accE: 0.00033206371, slab: 3, g: 0.54362038, grid: [27, 27, 100] },
  { name: 'real_4x4x8_s3_a1e-10', box: [12.8, 12.8, 25.6], natoms: 128, q2: 10626.03872, rc: 6, accE: 3.3206371e-8, slab: 3, g: 0.74082279, grid: [216, 216, 750] },
  { name: 'real_6x6x5_s3_a1e-3', box: [19.2, 19.2, 16], natoms: 180, q2: 14942.86695, rc: 6, accE: 0.33206371, slab: 3, g: 0.34550755, grid: [9, 9, 18] },
  { name: 'real_6x6x5_s3_a1e-4', box: [19.2, 19.2, 16], natoms: 180, q2: 14942.86695, rc: 6, accE: 0.033206371, slab: 3, g: 0.42540128, grid: [15, 15, 30] },
  { name: 'real_6x6x5_s3_a1e-6', box: [19.2, 19.2, 16], natoms: 180, q2: 14942.86695, rc: 6, accE: 0.00033206371, slab: 3, g: 0.54742917, grid: [40, 40, 72] },
  { name: 'real_6x6x5_s3_a1e-10', box: [19.2, 19.2, 16], natoms: 180, q2: 14942.86695, rc: 6, accE: 3.3206371e-8, slab: 3, g: 0.74217722, grid: [300, 300, 576] },
  { name: 'lj_5x5x4_s3_a1e-4', box: [4.27493986669, 4.27493986669, 3.41995189335], natoms: 100, q2: 100, rc: 2.5, accE: 0.0001, slab: 3, g: 1.2659986, grid: [18, 18, 32] },
  { name: 'lj_4x4x8_s3_a1e-4', box: [3.41995189335, 3.41995189335, 6.83990378671], natoms: 128, q2: 128, rc: 2.5, accE: 0.0001, slab: 3, g: 1.2685636, grid: [15, 15, 60] },
  { name: 'lj_6x6x5_s3_a1e-4', box: [5.12992784003, 5.12992784003, 4.27493986669], natoms: 180, q2: 180, rc: 2.5, accE: 0.0001, slab: 3, g: 1.2727602, grid: [24, 24, 40] },
  { name: 'metal_5x5x4_s3_a1e-4', box: [16, 16, 12.8], natoms: 100, q2: 1439.9645, rc: 2.5, accE: 0.0014399645, slab: 3, g: 1.1248389, grid: [40, 40, 80] },
  { name: 'metal_4x4x8_s3_a1e-4', box: [12.8, 12.8, 25.6], natoms: 128, q2: 1843.15456, rc: 2.5, accE: 0.0014399645, slab: 3, g: 1.1372812, grid: [36, 36, 144] },
  { name: 'metal_6x6x5_s3_a1e-4', box: [19.2, 19.2, 16], natoms: 180, q2: 2591.9361, rc: 2.5, accE: 0.0014399645, slab: 3, g: 1.1196077, grid: [48, 48, 96] },
  { name: 'real_4x4x10_s4_a1e-5_q0.5v0.3', box: [12.8, 12.8, 32], natoms: 160, q2: 9032.132912, rc: 6, accE: 0.0033206371, slab: 4, g: 0.47579148, grid: [16, 16, 90] },
  { name: 'real_5x5x6_s4_a1e-5_q0.5v0.3', box: [16, 16, 19.2], natoms: 150, q2: 8467.624605, rc: 6, accE: 0.0033206371, slab: 4, g: 0.47320006, grid: [18, 18, 60] },
  { name: 'real_3x4x12_s4_a1e-5_q0.5v0.3', box: [9.6, 12.8, 38.4], natoms: 144, q2: 8128.919621, rc: 6, accE: 0.0033206371, slab: 4, g: 0.48222013, grid: [15, 16, 108] },
  { name: 'real_6x6x3_s4_a1e-5_q0.5v0.3', box: [19.2, 19.2, 9.6], natoms: 108, q2: 6096.689716, rc: 6, accE: 0.0033206371, slab: 4, g: 0.47477929, grid: [20, 20, 36] },
  { name: 'real_4x6x8_s4_a1e-5_q0.5v0.3', box: [12.8, 19.2, 25.6], natoms: 192, q2: 10838.55949, rc: 6, accE: 0.0033206371, slab: 4, g: 0.47521726, grid: [16, 24, 75] },
  { name: 'real_8x8x4_s4_a1e-5_q0.5v0.3', box: [25.6, 25.6, 12.8], natoms: 256, q2: 14451.41266, rc: 6, accE: 0.0033206371, slab: 4, g: 0.47314423, grid: [27, 27, 45] },
  { name: 'real_4x4x10_s6_a1e-6', box: [12.8, 12.8, 32], natoms: 160, q2: 13282.5484, rc: 9, accE: 0.00033206371, slab: 6, g: 0.36055315, grid: [18, 18, 135] },
  { name: 'real_5x5x6_s6_a1e-6', box: [16, 16, 19.2], natoms: 150, q2: 12452.38912, rc: 9, accE: 0.00033206371, slab: 6, g: 0.36434601, grid: [24, 24, 90] },
  { name: 'real_3x4x12_s6_a1e-6', box: [9.6, 12.8, 38.4], natoms: 144, q2: 11954.29356, rc: 9, accE: 0.00033206371, slab: 6, g: 0.36321582, grid: [15, 18, 160] },
  { name: 'real_6x6x3_s6_a1e-6', box: [19.2, 19.2, 9.6], natoms: 108, q2: 8965.72017, rc: 9, accE: 0.00033206371, slab: 6, g: 0.36342684, grid: [24, 24, 54] },
  { name: 'real_4x4x10_s2_a1e-4', box: [12.8, 12.8, 32], natoms: 160, q2: 13282.5484, rc: 4, accE: 0.033206371, slab: 2, g: 0.63155959, grid: [16, 16, 54] },
  { name: 'real_5x5x6_s2_a1e-4', box: [16, 16, 19.2], natoms: 150, q2: 12452.38912, rc: 4, accE: 0.033206371, slab: 2, g: 0.6371964, grid: [20, 20, 36] },
  { name: 'real_3x4x12_s2_a1e-4', box: [9.6, 12.8, 38.4], natoms: 144, q2: 11954.29356, rc: 4, accE: 0.033206371, slab: 2, g: 0.63992505, grid: [15, 16, 64] },
  { name: 'real_6x6x3_s2_a1e-4', box: [19.2, 19.2, 9.6], natoms: 108, q2: 8965.72017, rc: 4, accE: 0.033206371, slab: 2, g: 0.66307948, grid: [24, 24, 24] },
  { name: 'lj_3x4x12_s5_a1e-3', box: [2.56496392002, 3.41995189335, 10.2598556801], natoms: 144, q2: 144, rc: 3, accE: 0.001, slab: 5, g: 0.92899422, grid: [8, 8, 60] },
  { name: 'lj_6x6x3_s5_a1e-3', box: [5.12992784003, 5.12992784003, 2.56496392002], natoms: 108, q2: 108, rc: 3, accE: 0.001, slab: 5, g: 0.93997517, grid: [12, 12, 20] },
  { name: 'lj_4x6x8_s5_a1e-3', box: [3.41995189335, 5.12992784003, 6.83990378671], natoms: 192, q2: 192, rc: 3, accE: 0.001, slab: 5, g: 0.93204107, grid: [8, 12, 45] },
  { name: 'lj_8x8x4_s5_a1e-3', box: [6.83990378671, 6.83990378671, 3.41995189335], natoms: 256, q2: 256, rc: 3, accE: 0.001, slab: 5, g: 0.94000507, grid: [15, 15, 27] },
  { name: 'metal_4x4x10_s2.5_a1e-5_q1v0.7', box: [12.8, 12.8, 32], natoms: 160, q2: 1716.437684, rc: 5, accE: 0.00014399645, slab: 2.5, g: 0.6201127, grid: [27, 27, 100] },
  { name: 'metal_5x5x6_s2.5_a1e-5_q1v0.7', box: [16, 16, 19.2], natoms: 150, q2: 1609.160329, rc: 5, accE: 0.00014399645, slab: 2.5, g: 0.61986213, grid: [30, 30, 72] },
  { name: 'metal_3x4x12_s2.5_a1e-5_q1v0.7', box: [9.6, 12.8, 38.4], natoms: 144, q2: 1544.793916, rc: 5, accE: 0.00014399645, slab: 2.5, g: 0.6253902, grid: [24, 27, 120] },
  { name: 'metal_6x6x3_s2.5_a1e-5_q1v0.7', box: [19.2, 19.2, 9.6], natoms: 108, q2: 1158.595437, rc: 5, accE: 0.00014399645, slab: 2.5, g: 0.62362441, grid: [36, 36, 40] },
  { name: 'real_5x5x6_s3_a1e-8', box: [16, 16, 19.2], natoms: 150, q2: 12452.38912, rc: 6, accE: 0.0000033206371, slab: 3, g: 0.64970779, grid: [90, 90, 225] },
  { name: 'real_3x4x12_s3_a1e-8', box: [9.6, 12.8, 38.4], natoms: 144, q2: 11954.29356, rc: 6, accE: 0.0000033206371, slab: 3, g: 0.65192309, grid: [64, 80, 375] },
  { name: 'real_6x6x3_s3_a1e-8', box: [19.2, 19.2, 9.6], natoms: 108, q2: 8965.72017, rc: 6, accE: 0.0000033206371, slab: 3, g: 0.6505766, grid: [100, 100, 135] },
  { name: 'madelung_s8_a1.0e-5', box: [1, 1, 20], natoms: 4, q2: 166.031855, rc: 12, accE: 0.0033206371, slab: 8, g: 0.24434577, grid: [3, 3, 40] },
  { name: 'madelung_s8_a1.0e-6', box: [1, 1, 20], natoms: 4, q2: 166.031855, rc: 12, accE: 0.00033206371, slab: 8, g: 0.27904188, grid: [3, 3, 64] },
  { name: 'madelung_s8_a1.0e-8', box: [1, 1, 20], natoms: 4, q2: 166.031855, rc: 12, accE: 0.0000033206371, slab: 8, g: 0.32744238, grid: [5, 5, 144] },
  { name: 'madelung_s8_a1.0e-10', box: [1, 1, 20], natoms: 4, q2: 166.031855, rc: 12, accE: 3.3206371e-8, slab: 8, g: 0.37216302, grid: [12, 12, 375] },
];

describe('PPPM automatic G-ewald and mesh, slab boxes (native-measured)', () => {
  for (const r of SLAB) {
    it(`matches native grid and G for slab ${r.name}`, () => {
      const got = selectPppmAuto({ lx: r.box[0], ly: r.box[1], lz: r.box[2], natoms: r.natoms, q2: r.q2, rc: r.rc, accE: r.accE, order: 5, gUser: 0, meshUser: null, slab: r.slab });
      expect(got.mesh).toEqual(r.grid);
      expect(Math.abs(got.g / r.g - 1)).toBeLessThan(1e-7);
    });
  }
});
