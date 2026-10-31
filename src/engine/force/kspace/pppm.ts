import { StyleError, type KSpaceCompute, type StyleContext } from '../types';
import type { SimState } from '../../types';
import type { Geometry } from '../../domain';
import { inumeric, parseNum } from '../util';
import { KSpaceBase, invertUpper, ignoredKeyword } from './ewald';
import { fft3d, isFFTSize, nextFFTSize } from '../fft';
import { pppmGplHook } from './pppmHook';
import { libm } from '../../libm';
import { pppmField, pppmWeights, weightScratch, wrap, type PppmGrid } from './pppmKernel';

/*
 * kspace_style pppm accuracy — docs.lammps.org/kspace_style.html: "The pppm
 * style invokes a particle-particle particle-mesh solver (Hockney) which maps
 * atom charge to a 3d mesh, uses 3d FFTs to solve Poisson's equation on the
 * mesh, then interpolates electric fields on the mesh points back to the
 * atoms." kspace_modify.html: "The ik approach is the default for PPPM and is
 * the original formulation used in (Hockney). It performs differentiation in
 * Kspace, and uses 3 FFTs to transfer each component of the computed fields
 * back to real space"; "The mesh keyword sets the grid size ... each
 * dimension must be factorizable into powers of 2, 3, and 5"; "The order
 * keyword determines how many grid spacings an atom's charge extends when it
 * is mapped to the grid ... The default for this parameter is 5 for PPPM ...
 * For PPPM, the minimum allowed setting is 2 and the maximum allowed setting
 * is 7."
 *
 * Written from Hockney & Eastwood (1988) and Deserno & Holm, J Chem Phys 109,
 * 7678 (1998): charges are assigned to the mesh in fractional coordinates
 * with the cardinal B-spline of order P (the Hockney-Eastwood assignment
 * function); the mesh charge is Fourier transformed; with k = 2 pi H^-T n
 * and aliases k_m = 2 pi H^-T (n + m N), U(k) the Fourier transform of the
 * assignment function and the reference kernel 4 pi / k^2 exp(-k^2/(4 g^2)),
 * the optimal influence function for ik differentiation is
 *   G(k) = sum_m U^2(k_m) (k . k_m) / k_m^2 4 pi exp(-k_m^2/(4 g^2))
 *          / (k^2 (sum_m U^2(k_m))^2),
 *   E    = C / (2 V) sum_k G(k) |rho(k)|^2 + self + background (as Ewald),
 *   E_a  = inverse FFT of -i k_a G rho / V, interpolated back with the same
 *          weights; F_j = C q_j E(r_j),
 *   W_ab = C / (2 V) sum_k G |rho|^2 (delta_ab - 2 k_a k_b (1/k^2 + 1/(4g^2))).
 * Automatic g_ewald, mesh and order: see selectPppmAuto below (orthorhombic
 * boxes, with or without slab). Triclinic boxes (and kspace_modify diff ad)
 * take the native choice from the lazily loaded GPL helper (PppmGplHook, see
 * src/engine-gpl/pppm), reached through styles.ts pppmGplHook().
 */

/*
 * Alias images in the numerator of G(k): per mesh dimension d,
 *   n_d = floor(g h_d / pi * ALIAS_C),  h_d = (L1 norm of box edge d) / N_d,
 * with ALIAS_C = (ln 1e7)^(1/4), i.e. images m with (pi m / (g h))^4 <= ln 1e7.
 * Not documented; measured against native LAMMPS (2 Sep 2026) by scanning
 * g_ewald on a coarse mesh: the switch points bracket ALIAS_C in
 * (2.0030, 2.0044] (orthogonal, both the 0->1 and 1->2 switch), and in
 * triclinic boxes the switch points of all three dimensions follow the L1
 * norms lx, |xy| + ly, |xz| + |yz| + lz of the edge vectors (two tilt sets,
 * predicted switches confirmed to 0.005 in g). Oracle case pppm_slab is the
 * one that depends on it (it uses no images; using one changes pzz by 5e-7).
 */
const ALIAS_C = Math.pow(7 * Math.LN10, 0.25);

/**
 * The numerator 4 pi / k^2 of the ik influence function. As native LAMMPS does (src/KSPACE/pppm.cpp
 * compute_gf_ik and compute_gf_ik_triclinic at b332f28: "numerator = 12.5663706/sqk;"), it is 4 pi rounded
 * to 8 digits; the analytic-differentiation Green function (compute_gf_ad) keeps MY_4PI. Measured with
 * native LAMMPS (black box, 12 charged atoms, real units, pppm 1e-4 with mesh 8 8 8 and G 0.16093073):
 * with the exact 4 pi the step-0 pe is 7.8e-9 relative off and the pressure 2.4e-9; with this constant
 * the pe agrees to 1.5e-14 and the pressure to 5e-14.
 */
const NATIVE_IK_FOUR_PI = 12.5663706;

/** Cardinal B-spline M_p(x), support [0, p]. */
const bspline = (p: number, x: number): number => {
  if (p === 1) return x >= 0 && x < 1 ? 1 : 0;
  if (x <= 0 || x >= p) return 0;
  return (x * bspline(p - 1, x) + (p - x) * bspline(p - 1, x - 1)) / (p - 1);
};

// the weights (bsplineWeights) and the per-atom loops are in pppmKernel.ts, which the pair threads run too

/*
 * Automatic g_ewald, mesh (orthorhombic boxes, with or without slab).
 *
 * docs.lammps.org/kspace_modify.html: "this setting, LAMMPS chooses the
 * parameter automatically as a function" (the gewald keyword); "must be
 * factorizable into powers of 2, 3, and 5." (the mesh keyword); "solver
 * chooses its own grid size, consistent with the" (the mesh keyword, unset).
 * kspace_modify.html on force: "The accuracy determines the RMS error in
 * per-atom forces" (absolute accuracy in force units, accE).
 *
 * Error estimates, from Deserno and Holm, J Chem Phys 109, 7694 (1998)
 * (arXiv cond-mat/9807100): real space, eq. (23),
 *   dF_r = 2 Q^2 / sqrt(N r_max V) exp(-alpha^2 r_max^2);
 * reciprocal space for ik differentiation, eq. (38),
 *   dF_k = (Q^2 / L^2) (h alpha)^P sqrt(alpha L / N sqrt(2 pi) sum_m a_m (h alpha)^2m),
 * with the coefficients a_m^(P) of their Table 2 (IK_COEF below). Q^2 is the
 * sum of q_i^2 times qqrd2e (the Coulomb constant of the units).
 *
 * Measured with native LAMMPS (black box), 42 orthorhombic cubic-edge systems
 * (accuracy 1e-3..1e-6, cutoffs 2..12, lj, real and metal units, orders 3, 4,
 * 5 and 7, charges 0.3..2, coul/long and lj/cut/coul/long) and the three
 * orthorhombic non-cubic systems (edges 8x8x30, 6x12x30, 40x30x25):
 *  - the initial g: g0 = acc/pre with pre = 2 Q^2 / sqrt(N rc V); g0 = sqrt(-ln g0)/rc,
 *    or (1.35 - 0.15 ln acc)/rc when acc/pre >= 1 (the Kolafa-Perram form; the
 *    second branch was measured with acc = 0.5 at N = 20, L = 10).
 *  - the mesh: per edge L_d, the smallest n >= 2 with dF_k(g0, L_d/n) <= acc,
 *    then the next size of the form 2^a 3^b 5^c after n + 1 (the +1 was
 *    measured: the smallest n alone gives one grid point too few in 6 of 38
 *    cubic cases). Matched in all 38 cubic and all 9 orthorhombic edges.
 *  - g: a Newton-Raphson root of dF_r(g) - dF_k(g, mesh) started at g0, with
 *    a finite-difference derivative (step 1e-6) and the stop |f| < 1e-5 in
 *    absolute force units (the stop is measured: the native g is off the exact
 *    root by up to 4e-3 when the stop is loose). Matched to the 8 digits native
 *    prints in all cubic cases.
 *  - non-cubic edges: dF_k is the root mean square of the three per-edge
 *    values, sqrt((dF_x^2 + dF_y^2 + dF_z^2) / 3), each per-edge value with its
 *    own L_d and h_d = L_d / n_d. Measured with native LAMMPS (black box) on
 *    the orthorhombic boxes 6x8x10, 12x6x6 and 10x12x14 (lj units, 2 atoms per
 *    cell, meshes 32x40x45, 60x36x36, 16x18x24): g = 1.2504352, 0.98484711 and
 *    0.60009333, reproduced to 2e-9 relative. The largest per-edge value (the
 *    earlier rule), the sum, the mean and the volume-based forms differ from
 *    these by 1e-3 to 5e-2.
 *  - kspace_modify gewald set: the mesh is chosen with that g; mesh set: g is
 *    solved with that mesh (both measured on a cubic 10-box).
 *  - kspace_modify slab V (slab box): the reciprocal-space estimate uses the
 *    extended z edge L_z = V lz (and its mesh spacing L_z / n_z), while the
 *    real-space estimate and the initial G keep the volume of the simulation
 *    domain V = lx ly lz. The docs say: kspace_modify.html, slab: "The volfactor
 *    value sets the ratio of the extended dimension in z divided by the actual
 *    dimension in z."; kspace_style.html: "The specified accuracy determines the
 *    relative RMS error in per-atom forces calculated by the long-range solver."
 *    Measured with native LAMMPS (black box): the printed "estimated absolute RMS
 *    force accuracy" equals sqrt(dF_r^2 + dF_k^2) with that rule (dF_k the root
 *    mean square over the three edges) to 0.1 percent on the slab boxes with
 *    accuracy 1e-2 to 1e-6 (madelung 1x1x20 box and 3x3x12 to 8x8x6 cell boxes; lj,
 *    real and metal; net charge and neutral). The boxes at accuracy 1e-8 and 1e-10
 *    print about 1e-5 whatever the grid, which this formula does not give; their G
 *    and grid still match.
 *    G and grid match native (8 digits) on 129 of 170 measured slab boxes (slab
 *    factors 2, 2.5, 3, 4, 5, 6, 8): all 58 at accuracy 1e-5 and all at 1e-6, 1e-8
 *    and 1e-10; at 1e-4 36 of 38 (misses: the madelung box and the 3x3x12 box with slab 8); at 1e-3 15 of 32; at 1e-2 3 of 25.
 *    At 1e-2 and 1e-3 the native z (and x, y) grid is not reproduced by any rule
 *    tested: factorable(ceil(n*)+1) with n* from the initial G, from the native G,
 *    the smallest factorable size, a fixed point of G and mesh, a slab-scaled error,
 *    and the box length or spacing with lz or lz*slab swapped (the initial-G window
 *    that gives each native grid is not common to the three edges). The native
 *    grid there looks path dependent (not a function of G alone).
 *    Measured on the 400 Angstrom z box of the SDS monolayer (non-neutral, order 3,
 *    cutoff 15, no slab): native G 0.015838529 on mesh 2 1 3, which this rule does
 *    not produce (engine 0.0195 on 3 3 3); that case stays pinned.
 */
const IK_COEF: Record<number, number[]> = {
  1: [2 / 3],
  2: [1 / 50, 5 / 294],
  3: [1 / 588, 7 / 1440, 21 / 3872],
  4: [1 / 4320, 3 / 1936, 7601 / 2271360, 143 / 28800],
  5: [1 / 23232, 7601 / 13628160, 143 / 69120, 517231 / 106536960, 106640677 / 11737571328],
  6: [691 / 68140800, 13 / 57600, 47021 / 35512320, 9694607 / 2095994880, 733191589 / 59609088000, 326190917 / 11700633600],
  7: [1 / 345600, 3617 / 35512320, 745739 / 838397952, 56399353 / 12773376000, 25091609 / 1560084480, 1755948832039 / 36229939200000, 4887769399 / 37838389248],
};

/** Real-space rms force error, Deserno and Holm eq. (23). */
export const pppmRealError = (g: number, rc: number, q2: number, n: number, volume: number): number =>
  (2 * q2 / Math.sqrt(n * rc * volume)) * libm.exp(-g * g * rc * rc);

/** Reciprocal-space rms force error for ik differentiation, Deserno and Holm eq. (38), one edge L, mesh spacing h. */
export const pppmKspaceError = (g: number, h: number, P: number, q2: number, n: number, L: number): number => {
  const coef = IK_COEF[P];
  if (!coef) throw new StyleError(`kspace_modify order ${P} has no PPPM error estimate`);
  const ha = h * g;
  let s = 0;
  // the C library's pow, as native LAMMPS calls it (src/KSPACE/pppm.cpp at b332f28, estimate_ik_error): the automatic
  // G-ewald is a Newton root of this error, so its last bits follow pow's rounding (Math.pow's differs between Node 22 and 24)
  for (let m = 0; m < P; m++) s += coef[m] * libm.pow(ha, 2 * m);
  return (q2 / (L * L)) * libm.pow(ha, P) * Math.sqrt(((g * L) / n) * Math.sqrt(2 * Math.PI) * s);
};

/** The inputs of the native G-ewald and mesh choice for a triclinic box (or diff ad), see PppmGplHook. */
export interface PppmGridRequest {
  lx: number; ly: number; lz: number;
  /** tilt factors (domain h[5], h[4], h[3]: xy, xz, yz) */
  xy: number; xz: number; yz: number;
  natoms: number;
  /** qqrd2e times the sum of q_i^2 */
  q2: number;
  /** absolute force accuracy */
  accuracy: number;
  /** real-space cutoff */
  cutoff: number;
  order: number;
  /** kspace_modify gewald (0 when automatic) */
  gUser: number;
  /** kspace_modify mesh (null when automatic) */
  meshUser: [number, number, number] | null;
  /** kspace_modify slab volfactor (1 for no slab) */
  slab: number;
  /** kspace_modify diff ad (orthogonal boxes only) */
  ad: boolean;
}

export interface PppmGridResult {
  g: number;
  mesh: [number, number, number];
}

/**
 * The lazily loaded GPL helper (src/engine-gpl/registry/pppm.ts, HOOKS 'pppm'): native's triclinic and diff ad grid
 * choice, and the analytic-differentiation solver. Core code reaches it only through styles.ts pppmGplHook().
 */
export interface PppmGplHook {
  grid(request: PppmGridRequest): PppmGridResult;
  /** kspace_modify diff ad: the analytic-differentiation style built from a kspace_style pppm instance (null when not applicable). */
  promoteAd(base: KSpacePPPM): KSpacePPPM | null;
}

/** Smallest size >= n that factors into 2, 3 and 5. */
const factorableAtLeast = (n: number): number => {
  let m = Math.max(2, n);
  while (!isFFTSize(m)) m++;
  return m;
};

export interface PppmAutoInput {
  lx: number; ly: number; lz: number;
  natoms: number;
  /** qqrd2e times the sum of q_i^2 */
  q2: number;
  rc: number;
  /** absolute force accuracy */
  accE: number;
  order: number;
  gUser: number;
  meshUser: [number, number, number] | null;
  /** kspace_modify slab volfactor (1 for no slab): the z edge of the reciprocal-space estimate is lz times this */
  slab?: number;
}

export interface PppmAutoResult {
  g: number;
  mesh: [number, number, number];
  g0: number;
}

/** Native-matched automatic choice of g_ewald and mesh (see the comment above). */
export function selectPppmAuto(inp: PppmAutoInput): PppmAutoResult {
  const slab = inp.slab ?? 1;
  const L = [inp.lx, inp.ly, inp.lz * slab];
  const V = inp.lx * inp.ly * inp.lz;
  const N = Math.max(1, inp.natoms);
  const P = inp.order;
  const rc = inp.rc;
  const pre = (2 * inp.q2) / Math.sqrt(N * rc * V);
  const ratio = inp.accE / pre;
  const g0 = ratio >= 1 ? (1.35 - 0.15 * libm.log(inp.accE)) / rc : Math.sqrt(-libm.log(ratio)) / rc;
  const gGrid = inp.gUser > 0 ? inp.gUser : g0;
  let mesh: [number, number, number];
  // As native LAMMPS does (src/KSPACE/pppm.cpp set_grid_global at b332f28): "boost grid size until it is factorable",
  // a mesh given by kspace_modify mesh too (7 7 7 runs as 8 8 8; factorable(1) is true, so 1 stays 1)
  if (inp.meshUser) mesh = [nextFFTSize(inp.meshUser[0]), nextFFTSize(inp.meshUser[1]), nextFFTSize(inp.meshUser[2])];
  else {
    // Measured with native LAMMPS (black box): the mesh starts at h = 1/G with n = int(L/h) + 1 and gains a point
    // while the error of the previous spacing is above the target (polystyrene box 16.5147 A, 320 atoms: 6 6 6 with
    // G 0.25561147; the start at n = 2 gave 5 and G 0.23512). The point is compared before it is added.
    const m = L.map((len) => {
      let h = 1 / gGrid;
      let n = Math.trunc(len / h) + 1;
      let err = pppmKspaceError(gGrid, h, P, inp.q2, N, len);
      while (err > inp.accE) {
        err = pppmKspaceError(gGrid, h, P, inp.q2, N, len);
        n++;
        h = len / n;
        if (n > 100000) throw new StyleError('kspace_style pppm: no mesh meets the accuracy');
      }
      return factorableAtLeast(n);
    });
    mesh = [m[0], m[1], m[2]];
  }
  if (inp.gUser > 0) return { g: inp.gUser, mesh, g0 };
  const kspace = (g: number) => {
    // root mean square over the edges (see the comment above; equal to the single-edge value for a cube)
    let sum = 0;
    for (let d = 0; d < 3; d++) {
      const e = pppmKspaceError(g, L[d] / mesh[d], P, inp.q2, N, L[d]);
      sum += e * e;
    }
    return Math.sqrt(sum / 3);
  };
  const f = (g: number) => pppmRealError(g, rc, inp.q2, N, V) - kspace(g);
  let g = g0;
  for (let it = 0; it < 10000; it++) {
    const f1 = f(g);
    const h = 1e-6;
    const df = (f(g + h) - f1) / h;
    g -= f1 / df;
    if (Math.abs(f(g)) < 1e-5) return { g, mesh, g0 };
  }
  throw new StyleError('kspace_style pppm: the G-ewald Newton iteration did not converge');
}

export class KSpacePPPM extends KSpaceBase {
  protected override readonly kspaceLabel: string = 'PPPM';

  /**
   * As native LAMMPS does (src/KSPACE/pppm.cpp init at b332f28): after the boundary checks, the stencil order must lie
   * in 2..7 (MAXORDER). kspace_modify minorder: the order may not be reduced below M, so M above the order is refused
   * (measured: order 5 with minorder 6 stops with PPPM order < minimum allowed order; order 7 with minorder 7 runs).
   */
  protected checkOrder(): void {
    if (this.order < 2 || this.order > 7) throw new StyleError(`${this.kspaceLabel} order cannot be < 2 or > 7`);
    if (this.minorder > this.order) throw new StyleError(`PPPM order < minimum allowed order (kspace_modify minorder ${this.minorder} > order ${this.order})`);
  }

  readonly name: string = 'pppm';
  override groupGroupEnable = true;
  meshUser: [number, number, number] | null = null;
  order = 5;
  /** Visibility: the electrode subclass (src/engine-gpl) reads the mesh, the influence function and the weights. */
  protected mesh: [number, number, number] = [0, 0, 0];
  protected green = new Float64Array(0);
  protected greenBox = '';

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'mesh': {
        // As native LAMMPS does (src/kspace.cpp modify_params at b332f28): all zero (automatic) or all positive
        const m = [inumeric(values[0]), inumeric(values[1]), inumeric(values[2])];
        if (m.every((v) => v === 0)) { this.meshUser = null; return 3; }
        if (m.some((v) => v <= 0)) throw new StyleError('Kspace_modify mesh parameters must be all zero or all positive');
        this.meshUser = m as [number, number, number];
        return 3;
      }
      // any integer; init checks the range, as native's PPPM::init does
      case 'order': this.order = inumeric(values[0]); return 1;
      // diff ad is built by the kspace_modify handler through pppmGplHook; the handler checks the flag
      case 'diff':
        if (values[0] !== 'ik') throw new StyleError(`kspace_modify diff ${values[0]} is not supported by kspace_style ${this.name}`);
        return 1;
      case 'wire': return this.wireNoCorrection(values);
      default: return super.modify(key, values) || ignoredKeyword(key, values, 'pppm');
    }
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    this.cutCoul = cutCoul;
    this.setupCharges(s, geom, ctx);
    this.checkOrder();
    const ortho = !s.box.triclinic && geom.xy === 0 && geom.xz === 0 && geom.yz === 0;
    if (ortho && this.qsqsum > 0 && this.natoms > 0) {
      const r = selectPppmAuto({
        lx: geom.lx, ly: geom.ly, lz: geom.lz, natoms: this.natoms, q2: qqrd2e * this.qsqsum, rc: cutCoul,
        accE: this.absAccuracy(s, qqrd2e), order: this.order, gUser: this.gewaldUser, meshUser: this.meshUser, slab: this.slab,
      });
      this.gEwald = r.g;
      this.mesh = r.mesh;
      // see the slab paragraph of the header: at accuracy 1e-5 and finer every measured slab box matches native; at
      // coarser accuracy some do not (36 of 38 at 1e-4, 15 of 32 at 1e-3, 3 of 25 at 1e-2)
      if (this.slab > 1 && (this.gewaldUser <= 0 || !this.meshUser) && this.accuracy > 1e-5) {
        ctx.log(`WARNING: kspace_style pppm: with kspace_modify slab, the automatic G-ewald and mesh match native LAMMPS only at accuracy 1e-5 or finer; set kspace_modify gewald and mesh to native's values for an exact match`);
      }
    } else {
      // triclinic box: the GPL helper reproduces native G-ewald and mesh choice, see src/engine-gpl/pppm
      const r = pppmGplHook().grid({
        lx: geom.lx, ly: geom.ly, lz: geom.lz, xy: geom.xy, xz: geom.xz, yz: geom.yz, natoms: this.natoms,
        q2: qqrd2e * this.qsqsum, accuracy: this.absAccuracy(s, qqrd2e), cutoff: cutCoul, order: this.order,
        gUser: this.gewaldUser, meshUser: this.meshUser, slab: this.slab, ad: false,
      });
      this.gEwald = r.g;
      this.mesh = r.mesh;
    }
    this.greenBox = '';
    ctx.log(`  G vector (1/distance) = ${this.gEwald}\n  grid = ${this.mesh.join(' ')}\n  stencil order = ${this.order}`);
  }

  /** Optimal influence function for the current box. */
  protected buildGreen(geom: Geometry): void {
    const [nx, ny, nz] = this.mesh;
    const lz = geom.lz * this.slab;
    const key = [geom.lx, geom.ly, lz, geom.xy, geom.xz, geom.yz, nx, ny, nz, this.order, this.gEwald].join(',');
    if (key === this.greenBox) return;
    this.greenBox = key;
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, lz);
    const g = this.gEwald;
    const P = this.order;
    const N = [nx, ny, nz];
    const green = new Float64Array(nx * ny * nz);
    const kvec = (a: number, b: number, c: number, out: number[]) => {
      out[0] = 2 * Math.PI * (a * inv[0][0] + b * inv[1][0] + c * inv[2][0]);
      out[1] = 2 * Math.PI * (a * inv[0][1] + b * inv[1][1] + c * inv[2][1]);
      out[2] = 2 * Math.PI * (a * inv[0][2] + b * inv[1][2] + c * inv[2][2]);
    };
    const sinc = (x: number) => (x === 0 ? 1 : libm.sinAlone(x) / x);
    const u2 = (n: number, nn: number) => Math.pow(sinc((Math.PI * n) / nn), 2 * P);
    // the numerator's alias images per dimension (measured, see ALIAS_C)
    const edge = [Math.abs(geom.lx), Math.abs(geom.xy) + Math.abs(geom.ly), Math.abs(geom.xz) + Math.abs(geom.yz) + Math.abs(lz)];
    const nb = [0, 1, 2].map((d) => Math.floor(((g * edge[d]) / (Math.PI * N[d])) * ALIAS_C));
    // the denominator sum_m U^2(k_m) factorizes per dimension; by Poisson
    // summation each factor is the finite cosine series of the autocorrelation
    // of the order-P B-spline (the order-2P B-spline at the integers):
    //   sum_m U^2(x + m) = sum_{|j| < P} M_2P(P + j) cos(2 pi j x),  x = n / N
    // the coefficients M_2P(P + j) once per setup: bspline's recursion costs 2^(2P) calls per value, and a changing box
    // (fix npt, nph) rebuilds the Green's function every step (the same values, so the same sums)
    const m2p = new Float64Array(2 * P - 1);
    for (let j = 1 - P; j < P; j++) m2p[j + P - 1] = bspline(2 * P, P + j);
    const den1 = N.map((nn) => {
      const out = new Float64Array(nn);
      for (let i = 0; i < nn; i++) {
        for (let j = 1 - P; j < P; j++) out[i] += m2p[j + P - 1] * libm.cos((2 * Math.PI * j * i) / nn);
      }
      return out;
    });
    const k = [0, 0, 0], km = [0, 0, 0];
    for (let iz = 0; iz < nz; iz++) {
      const mz = iz < nz / 2 ? iz : iz - nz;
      for (let iy = 0; iy < ny; iy++) {
        const my = iy < ny / 2 ? iy : iy - ny;
        for (let ix = 0; ix < nx; ix++) {
          const mx = ix < nx / 2 ? ix : ix - nx;
          const idx = (iz * ny + iy) * nx + ix;
          if (mx === 0 && my === 0 && mz === 0) { green[idx] = 0; continue; }
          kvec(mx, my, mz, k);
          const k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
          let num = 0;
          for (let ax = -nb[0]; ax <= nb[0]; ax++) {
            const nxm = mx + ax * nx;
            const ux = u2(nxm, nx);
            for (let ay = -nb[1]; ay <= nb[1]; ay++) {
              const nym = my + ay * ny;
              const uy = u2(nym, ny);
              for (let az = -nb[2]; az <= nb[2]; az++) {
                const nzm = mz + az * nz;
                kvec(nxm, nym, nzm, km);
                const km2 = km[0] * km[0] + km[1] * km[1] + km[2] * km[2];
                num += ux * uy * u2(nzm, nz) * ((k[0] * km[0] + k[1] * km[1] + k[2] * km[2]) / km2) * NATIVE_IK_FOUR_PI * libm.exp(-km2 / (4 * g * g));
              }
            }
          }
          const den = den1[0][ix] * den1[1][iy] * den1[2][iz];
          green[idx] = num / (k2 * den * den);
        }
      }
    }
    this.green = green;
  }

  /**
   * Charge assignment: the grid index of the first stencil point and the P B-spline weights per dimension for every
   * atom (the weights are shared by the charge spreading and the field interpolation).
   */
  protected assignWeights(s: SimState, geom: Geometry): { first: Int32Array; wts: Float64Array } {
    const P = this.order;
    const n = s.n;
    const first = new Int32Array(3 * n);
    const wts = new Float64Array(3 * n * P);
    pppmWeights(this.grid(geom), s.x, 0, n, first, wts, weightScratch(P));
    return { first, wts };
  }

  /** The box and mesh the per-atom loops of pppmKernel.ts need. */
  protected grid(geom: Geometry): PppmGrid {
    const [i00, i01, i02, i11, i12, i22] = geom.hInv;
    const [nx, ny, nz] = this.mesh;
    return { lo0: geom.lo[0], lo1: geom.lo[1], lo2: geom.lo[2], i00, i01, i02, i11, i12, i22, slab: this.slab, nx, ny, nz, order: this.order };
  }

  /** The P mesh indices of atom i in x, y and z (from first, wrapped into the mesh), in a reused array. */
  private meshIdx = new Int32Array(0);
  protected meshIndices(i: number, first: Int32Array): Int32Array {
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    if (this.meshIdx.length !== 3 * P) this.meshIdx = new Int32Array(3 * P);
    const xs = this.meshIdx;
    const fx = first[3 * i], fy = first[3 * i + 1], fz = first[3 * i + 2];
    for (let k = 0; k < P; k++) {
      xs[k] = wrap(fx + k, nx);
      xs[P + k] = wrap(fy + k, ny);
      xs[2 * P + k] = wrap(fz + k, nz);
    }
    return xs;
  }

  /** Adds charge q of atom i to the real parts of the complex mesh rho with its assignment weights. */
  protected spread(rho: Float64Array, i: number, q: number, first: Int32Array, wts: Float64Array): void {
    const [nx, ny] = this.mesh;
    const P = this.order;
    const xs = this.meshIndices(i, first);
    for (let c = 0; c < P; c++) {
      const iz = xs[2 * P + c];
      const wz = q * wts[(3 * i + 2) * P + c];
      for (let b = 0; b < P; b++) {
        const iy = xs[P + b];
        const wyz = wz * wts[(3 * i + 1) * P + b];
        const row = (iz * ny + iy) * nx;
        for (let a = 0; a < P; a++) {
          rho[2 * (row + xs[a])] += wyz * wts[(3 * i) * P + a];
        }
      }
    }
  }

  /**
   * The mesh charges of two atom subsets in k-space (the forward transform of compute(), re and im interleaved), with
   * the influence function, the wave vector of every mesh point and the slab-extended volume: the long-range group-group
   * term of compute group/group (docs.lammps.org/compute_group_group.html: "the interaction energy will include a
   * Kspace component which is the long-range Coulombic energy between all the atoms in the first group and all the
   * atoms in the second group"). In compute()'s normalization the energy of the mesh is C / (2 V) sum_k G |rho(k)|^2.
   */
  groupDensities(s: SimState, geom: Geometry, inA: (i: number) => boolean, inB: (i: number) => boolean): {
    rhoA: Float64Array; rhoB: Float64Array; green: Float64Array; kx: Float64Array; ky: Float64Array; kz: Float64Array; volume: number;
  } {
    const [nx, ny, nz] = this.mesh;
    const ng = nx * ny * nz;
    this.buildGreen(geom);
    const { first, wts } = this.assignWeights(s, geom);
    const qg = this.gridCharges(s);
    const rhoA = new Float64Array(2 * ng), rhoB = new Float64Array(2 * ng);
    for (let i = 0; i < s.n; i++) {
      if (qg[i] === 0) continue;
      if (inA(i)) this.spread(rhoA, i, qg[i], first, wts);
      if (inB(i)) this.spread(rhoB, i, qg[i], first, wts);
    }
    fft3d(rhoA, nx, ny, nz, -1);
    fft3d(rhoB, nx, ny, nz, -1);
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, geom.lz * this.slab);
    const kx = new Float64Array(ng), ky = new Float64Array(ng), kz = new Float64Array(ng);
    for (let iz = 0; iz < nz; iz++) {
      const mz = iz < nz / 2 ? iz : iz - nz;
      for (let iy = 0; iy < ny; iy++) {
        const my = iy < ny / 2 ? iy : iy - ny;
        for (let ix = 0; ix < nx; ix++) {
          const mx = ix < nx / 2 ? ix : ix - nx;
          const idx = (iz * ny + iy) * nx + ix;
          kx[idx] = 2 * Math.PI * (mx * inv[0][0] + my * inv[1][0] + mz * inv[2][0]);
          ky[idx] = 2 * Math.PI * (mx * inv[0][1] + my * inv[1][1] + mz * inv[2][1]);
          kz[idx] = 2 * Math.PI * (mx * inv[0][2] + my * inv[1][2] + mz * inv[2][2]);
        }
      }
    }
    return { rhoA, rhoB, green: this.green, kx, ky, kz, volume: geom.volume(3) * this.slab };
  }

  /** Charges that enter the mesh (and so the mesh forces and energy). pppm: every charge. */
  protected gridCharges(s: SimState): ArrayLike<number> {
    return s.q;
  }

  compute(kc: KSpaceCompute): void {
    const s = kc.s;
    this.refreshChargeSums(s);
    const geom = kc.geom;
    const n = s.n;
    const C = kc.qqrd2e;
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    const g = this.gEwald;
    this.volume = geom.volume(3) * this.slab;
    const V = this.volume;
    this.buildGreen(geom);
    const ng = nx * ny * nz;
    // the per-atom loops (weights, field interpolation) on the pair threads when there are any (pppmKernel.ts: the same
    // code on ranges of atoms, so the result does not depend on the threads); a style that assigns its own weights
    // keeps them on the engine thread
    const thr = kc.threads && typeof kc.threads.pppmWeights === 'function' && n >= kc.threads.minAtoms ? kc.threads : null;
    const grid = this.grid(geom);
    const tw = thr && this.assignWeights === KSpacePPPM.prototype.assignWeights ? thr.pppmWeights(grid, s.x, n) : null;
    const { first, wts } = tw ?? this.assignWeights(s, geom);
    const rho = new Float64Array(2 * ng);
    const qg = this.gridCharges(s);
    for (let i = 0; i < n; i++) if (qg[i] !== 0) this.spread(rho, i, qg[i], first, wts);
    fft3d(rho, nx, ny, nz, -1);
    // energy, virial and the three field components in k-space
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, geom.lz * this.slab);
    const meshes = tw ? thr!.pppmMeshes(2 * ng) : null;
    const [ex, ey, ez] = meshes ?? [new Float64Array(2 * ng), new Float64Array(2 * ng), new Float64Array(2 * ng)];
    // per-atom energy and virial (compute pe/atom, stress/atom): the potential G rho and the six G rho vg_j on the mesh,
    // as native PPPM::poisson_peratom builds u_brick and v0_brick .. v5_brick
    const peratom = kc.eatom !== null || kc.vatom !== null;
    const ub = kc.eatom ? new Float64Array(2 * ng) : null;
    const vb = kc.vatom ? Array.from({ length: 6 }, () => new Float64Array(2 * ng)) : null;
    let e = 0;
    const w = [0, 0, 0, 0, 0, 0];
    for (let iz = 0; iz < nz; iz++) {
      const mz = iz < nz / 2 ? iz : iz - nz;
      for (let iy = 0; iy < ny; iy++) {
        const my = iy < ny / 2 ? iy : iy - ny;
        for (let ix = 0; ix < nx; ix++) {
          const mx = ix < nx / 2 ? ix : ix - nx;
          const idx = (iz * ny + iy) * nx + ix;
          const G = this.green[idx];
          if (G === 0) continue;
          const re = rho[2 * idx], im = rho[2 * idx + 1];
          const r2 = re * re + im * im;
          const kx = 2 * Math.PI * (mx * inv[0][0] + my * inv[1][0] + mz * inv[2][0]);
          const ky = 2 * Math.PI * (mx * inv[0][1] + my * inv[1][1] + mz * inv[2][1]);
          const kz = 2 * Math.PI * (mx * inv[0][2] + my * inv[1][2] + mz * inv[2][2]);
          const k2 = kx * kx + ky * ky + kz * kz;
          e += G * r2;
          const b = 2 * (1 / k2 + 1 / (4 * g * g));
          const t = G * r2;
          w[0] += t * (1 - b * kx * kx); w[1] += t * (1 - b * ky * ky); w[2] += t * (1 - b * kz * kz);
          w[3] += t * (-b * kx * ky); w[4] += t * (-b * kx * kz); w[5] += t * (-b * ky * kz);
          // E_a(k) = -i k_a G rho  ->  (re, im) * (-i k_a G) = (k_a G im, -k_a G re)
          ex[2 * idx] = kx * G * im; ex[2 * idx + 1] = -kx * G * re;
          ey[2 * idx] = ky * G * im; ey[2 * idx + 1] = -ky * G * re;
          ez[2 * idx] = kz * G * im; ez[2 * idx + 1] = -kz * G * re;
          if (ub) { ub[2 * idx] = G * re; ub[2 * idx + 1] = G * im; }
          if (vb) {
            const vg = [1 - b * kx * kx, 1 - b * ky * ky, 1 - b * kz * kz, -b * kx * ky, -b * kx * kz, -b * ky * kz];
            for (let c = 0; c < 6; c++) { vb[c][2 * idx] = G * re * vg[c]; vb[c][2 * idx + 1] = G * im * vg[c]; }
          }
        }
      }
    }
    fft3d(ex, nx, ny, nz, 1);
    fft3d(ey, nx, ny, nz, 1);
    fft3d(ez, nx, ny, nz, 1);
    if (ub) fft3d(ub, nx, ny, nz, 1);
    if (vb) for (const v of vb) fft3d(v, nx, ny, nz, 1);
    // interpolate the field back and apply F = C q E / V (pppmKernel.ts pppmField, on the threads when the weights were)
    let fe = meshes ? thr!.pppmField(grid, n, qg, first, wts, ex, ey, ez, C, V, this.zForce) : null;
    if (!fe) {
      fe = new Float64Array(3 * n);
      pppmField(grid, 0, n, qg, first, wts, ex, ey, ez, C, V, this.zForce, fe, new Int32Array(3 * P));
    }
    for (let i = 0; i < n; i++) {
      if (qg[i] === 0) continue;
      kc.f[3 * i] += fe[3 * i]; kc.f[3 * i + 1] += fe[3 * i + 1];
      if (this.zForce) kc.f[3 * i + 2] += fe[3 * i + 2];
    }
    const pref = C / (2 * V);
    if (peratom) this.peratomMesh(kc, qg, first, wts, ub, vb, pref);
    kc.acc.elong += pref * e + this.constantEnergy(C);
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += pref * w[c];
    this.slabCorrection(kc);
  }

  /**
   * Per-atom energy and virial of the mesh part. As native LAMMPS does (src/KSPACE/pppm.cpp fieldforce_peratom and
   * compute at b332f28): u and v_j interpolated at each charged atom with the assignment weights, eatom_i =
   * qscale (0.5 q_i u_i - g q_i^2 / sqrt(pi) - pi q_i qsum / (2 g^2 V)), vatom_i,j = 0.5 qscale q_i v_j(x_i). In the
   * engine's mesh normalization 0.5 qscale u_native = C / (2 V) u (pref here). They sum to the global energy and virial.
   */
  protected peratomMesh(kc: KSpaceCompute, qg: ArrayLike<number>, first: Int32Array, wts: Float64Array,
    ub: Float64Array | null, vb: Float64Array[] | null, pref: number): void {
    const project = this.peratomProject;
    const s = kc.s;
    const [nx, ny] = this.mesh;
    const P = this.order;
    const vsum = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      const q = qg[i];
      if (q === 0) continue;
      const xs = this.meshIndices(i, first);
      let u = 0;
      vsum.fill(0);
      for (let c = 0; c < P; c++) {
        const iz = xs[2 * P + c];
        const wz = wts[(3 * i + 2) * P + c];
        for (let b = 0; b < P; b++) {
          const iy = xs[P + b];
          const wyz = wz * wts[(3 * i + 1) * P + b];
          const row = (iz * ny + iy) * nx;
          for (let a = 0; a < P; a++) {
            const ww = wyz * wts[(3 * i) * P + a];
            const idx = 2 * (row + xs[a]);
            if (ub) u += ww * ub[idx];
            if (vb) for (let k = 0; k < 6; k++) vsum[k] += ww * vb[k][idx];
          }
        }
      }
      if (project) {
        for (let k = 0; k < 6; k++) vsum[k] *= pref * q;
        project(i, kc.eatom && ub ? pref * q * u : 0, kc.vatom && vb ? vsum : null);
        continue;
      }
      if (kc.eatom && ub) kc.eatom[i] += pref * q * u;
      if (kc.vatom && vb) for (let k = 0; k < 6; k++) kc.vatom[6 * i + k] += pref * q * vsum[k];
    }
    if (kc.eatom) this.constantEnergyPeratom(kc, qg);
  }

  /**
   * Where the mesh part of a charge site's per-atom energy and virial goes (pppm/tip4p: an O's M site to O and its
   * two H). Null: to the atom itself. The self terms are applied afterwards to every atom's own entry.
   */
  peratomProject: ((i: number, e: number, v: number[] | null) => void) | null = null;
}

/*
 * kspace_style pppm/cg accuracy (smallq) — docs.lammps.org/kspace_style.html: "The *pppm/cg* style is identical to the
 * *pppm* style except that it has an optimization for systems where most particles are uncharged." "The optional
 * *smallq* argument defines the cutoff for the absolute charge value which determines whether a particle is considered
 * charged or not. Its default value is 1.0e-5."
 *
 * Measured with native LAMMPS (black box), 64-atom simple-cubic charge test (pos/neg charges +-0.4 on 2/6 of the atoms,
 * the rest 0, displaced; kspace_modify gewald 0.4 mesh 16 16 16 order 5):
 *  - with every nonzero charge above smallq, pppm/cg equals pppm bitwise (energy and all forces);
 *  - atoms with |q| <= smallq are left out of the mesh: no mesh force on them and no mesh energy from them. Tested with
 *    charge 3e-6 (default smallq 1e-5) and 0.05 (smallq 0.1); the boundary is strict: |q| = smallq (0.1 with smallq 0.1)
 *    is excluded as well, and 0.10000001 with smallq 0.10000001 excludes it too;
 *  - the self-energy and the neutralising term still use all atoms: with 10 atoms of charge 0.05 excluded the energy
 *    shift relative to the same run with those charges set to 0 was -2.262094 (= -(g qqrd2e/sqrt(pi)) sum q^2
 *    - (pi qqrd2e/(2 V g^2)) Q^2 with Q the net charge of all atoms), the forces were identical;
 *  - the automatic G-ewald and mesh estimate use the charges of all atoms (identical G and mesh when every charge is
 *    excluded by a large smallq);
 *  - with every charge excluded the energy is the self-energy term of all atoms (-297.748580 with the automatic G 0.45150291
 *    for the +-0.4 charges), pressure 0.
 * The engine: the mesh charges are the charges with |q| > smallq (others zero); energy constants are those of pppm.
 * kspace_modify slab is refused for pppm/cg (the slab correction was not measured with the optimisation).
 */
export class KSpacePPPMCG extends KSpacePPPM {
  override readonly name: string = 'pppm/cg';
  smallq = 1e-5;

  settings(args: string[]): void {
    // As native LAMMPS does (src/KSPACE/pppm_cg.cpp PPPMCG::settings at b332f28): one or two arguments.
    if (args.length < 1 || args.length > 2) throw new StyleError(`Illegal kspace_style ${this.styleShown} command`);
    super.settings([args[0]]);
    if (args.length === 2) {
      this.smallq = parseNum(args[1], 'smallq');
      if (!(this.smallq >= 0)) throw new StyleError('kspace_style pppm/cg smallq must be >= 0');
    }
  }

  modify(key: string, values: string[]): number {
    if (key === 'slab') throw new StyleError('kspace_modify slab is not supported by kspace_style pppm/cg (the slab correction was not measured with it)');
    return super.modify(key, values);
  }

  /** Mesh charges: atoms with |q| > smallq keep their charge; the rest contribute nothing to the mesh. */
  protected override gridCharges(s: SimState): ArrayLike<number> {
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) {
      const q = s.q[i];
      out[i] = Math.abs(q) > this.smallq ? q : 0;
    }
    return out;
  }
}
