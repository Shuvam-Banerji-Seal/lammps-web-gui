import { NEIGHMASK } from '../../neighbor';
import {
  PAIR_SHELL_RATIO,
  referenceVectors,
  SUPPORTED_REFERENCE_LATTICES,
  type ReferenceLattice,
  type ReferenceVector,
} from './meam_lattice';
import {
  gOfIbar,
  gPrimeOfIbar,
  PHI_INTERVALS,
  PHI_LO,
  PHI_SPAN,
  PhiTable,
  alloyAtomEnergyGrad,
  makeAlloyModel,
  type AlloyElement,
  type AlloyModel,
  type AlloyNeighbor,
  type AlloyOptions,
  type AlloyPair,
} from './meam_alloy';
import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';

/*
 * MEAM, single element: energy, analytic forces and pair_style meam (see the sections below).
 * Documented model: docs.lammps.org/pair_meam.html (plans/lammps-docs/pair_meam.rst):
 *   "E = \sum_i \left\{ F_i(\bar{\rho}_i) + \frac{1}{2} \sum_{i \neq j} \phi_{ij} (r_{ij}) \right\}"
 *   "astar = alpha \* (r/re - 1.d0)"
 *   "if erose_form = 0: erose = -Ec\*(1+astar+a3\*(astar\*\*3)/(r/re))\*exp(-astar)"  (a3 = 0 here)
 * Documented defaults: rc = 4.0, delr = 0.1, Cmax = 2.8, Cmin = 2.0, augt1 = 1, zbl = 1.
 * "delr = length of smoothing distance for cutoff function" (no cutoff form given in the doc).
 *
 * Measured with native LAMMPS (black box) and reproduced here:
 *  - pre-filter: a screener k of bond ij is used only if X_ik and X_jk are both <= Cmax^2/(4(Cmax-1))
 *    (X = (r/r_ij)^2). Without it the fcc first-neighbour S is 0 and rho_ref vanishes.
 *  - screening S_ij = prod_k fs((C - Cmin)/(Cmax - Cmin)), fs(x) = [1-(1-x)^4]^2 for 0<x<1.
 *  - pair term screened: E_pair = 1/2 sum S_ij fc(r) phi(r).
 *  - t1 augmented by 0.6 t3 (augt1 = 1).
 *  - radial weight fc(r) = [1-(1-x)^4]^2 with x = (rc - r)/delr, inside the window [rc - delr, rc]. Measured
 *    with native LAMMPS (black box): the pair term of a bond inside the window is exactly fc(r) times phi(r)
 *    (A = 0 probe, 1e-12), and the partial densities of that bond carry the same fc(r) (dimers 3.9 to 3.999 A
 *    agree to 1e-12 eV). The reference structure of phi carries no fc (see refRhoBarPrime). The screening
 *    factor S uses the C function only, with no fc of the screening atom (trimers with an in-window screener
 *    agree to 1e-12 eV).
 *  - ZBL blend (zbl = 1) is not implemented; an energy offset of -2.3e-5 eV appears for a pair at 2.0 A.
 */

export interface MeamElement {
  z: number;
  re: number;
  alpha: number;
  Ec: number;
  A: number;
  beta: [number, number, number, number];
  t: [number, number, number, number];
  ibar: number;
  /** reference lattice of the element (default fcc); see meam_lattice.ts */
  lat?: ReferenceLattice;
  /** Rose energy form (erose_form, attrac, repuls); default form 0 with attrac = repuls = 0 */
  erose?: EroseSettings;
  /** 2NN (nn2 = 1) data: the second-neighbour pair recurrence of Lee-Baskes (see nn2Phi) */
  nn2?: Nn2Ref;
}

/**
 * 2NN pair recurrence data of an element: a = r2/r1 of the reference lattice, z1/z2 = the
 * screening-weighted coordination of its first/second neighbour shell.
 */
export interface Nn2Ref {
  a: number;
  z1: number;
  z2: number;
  /** unit directions of the first two shells (3 per neighbour) */
  dir: Float64Array;
  /** |vector| of each neighbour of the unit-scale reference structure */
  ratio: Float64Array;
  /** screening weight of each neighbour */
  s: Float64Array;
}

/** erose_form, attrac and repuls of an element (see eroseE). */
export interface EroseSettings {
  form: number;
  attrac: number;
  repuls: number;
}
const NO_EROSE: EroseSettings = { form: 0, attrac: 0, repuls: 0 };

export interface MeamOptions {
  rc: number;
  delr: number;
  Cmin: number;
  Cmax: number;
  augt1: boolean;
}

export const DEFAULT_MEAM_OPTIONS: MeamOptions = { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8, augt1: true };

/** [1-(1-x)^4]^2 for 0 < x < 1; 0 below, 1 above. */
const poly = (x: number): number => {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const q = (1 - x) ** 4;
  return (1 - q) * (1 - q);
};

export const radialWeight = (r: number, o: MeamOptions): number => poly((o.rc - r) / o.delr);
const screenWeight = (C: number, o: MeamOptions): number => (C >= o.Cmax ? 1 : poly((C - o.Cmin) / (o.Cmax - o.Cmin)));
const ebound = (o: MeamOptions): number => (o.Cmax * o.Cmax) / (4 * (o.Cmax - 1));

export interface MeamNeighbor {
  j: number;
  dx: number;
  dy: number;
  dz: number;
  r: number;
}

/** Screening weights S for one atom's neighbour list (pre-filter + product of C-based weights). */
export function screening(nb: MeamNeighbor[], o: MeamOptions): Float64Array {
  const n = nb.length;
  const S = new Float64Array(n);
  const eb = ebound(o);
  for (let a = 0; a < n; a++) {
    const p = nb[a];
    const rij2 = p.r * p.r;
    let s = 1;
    for (let b = 0; b < n && s !== 0; b++) {
      if (b === a) continue;
      const q = nb[b];
      const Xik = (q.r * q.r) / rij2;
      const Xjk = ((q.dx - p.dx) ** 2 + (q.dy - p.dy) ** 2 + (q.dz - p.dz) ** 2) / rij2;
      if (Xik > eb || Xjk > eb) continue;
      const d = Xik - Xjk;
      const C = (2 * (Xik + Xjk) - d * d - 1) / (1 - d * d);
      s *= screenWeight(C, o);
    }
    S[a] = s;
  }
  return S;
}

/**
 * ZBL blend guard. The default zbl = 1 blends the pair term with the ZBL potential at short range, which is NOT
 * implemented. Measured with native LAMMPS (black box) on a Cu dimer (library Cu, Cu.meam with and without
 * zbl(1,1) = 0): the blended pair term differs from the unblended one below about 2.0 A, and the blend weight is
 * 1 up to about 1.05 A, falls to about 0.1 at 1.6 A and is about 0 at 2.0 A (Cu re = 2.55 A, so 0.78 re). Pairs
 * are refused below ZBL_GUARD * re (0.8 re = 2.04 A for Cu), which keeps every pair where the blend is measurably
 * nonzero out of the engine's energy. The blend profile of other elements is not measured.
 */
export const ZBL_GUARD = 0.8;

/** ibar values with a G(Gamma) form (gOfIbar in meam_alloy.ts; docs: ibar 2 is not implemented, 4 needs gsmooth_factor). */
export const SUPPORTED_IBAR: readonly number[] = [0, 1, 3];
/** G(Gamma) forms (the docs quotes are in meam_alloy.ts). */
export const gOf = gOfIbar;

/** Reference-structure t parameters (augt1 applied to t1); t0 = 1 (library entries normalised to t0 = 1). */
const refTuple = (el: MeamElement, o: MeamOptions): [number, number, number, number] => [
  1,
  o.augt1 ? el.t[1] + 0.6 * el.t[3] : el.t[1],
  el.t[2],
  el.t[3],
];

/*
 * Background density of the reference structure at nearest-neighbour distance r:
 * rho_bar = rho0 * G(Gamma), evaluated for one atom of the lattice el.lat (fcc, bcc or dia) with the
 * screening S and the angular moments of the atom energy, but WITHOUT the radial cutoff weight fc.
 * Measured with native LAMMPS (black box): for a bcc dimer inside the window [rc - delr, rc] the native
 * energy agrees with the engine to 1e-12 eV only if the reference shell carries no fc(r), while the
 * dimer's own partial densities and pair weight carry fc(r) (the pair term's weight is exactly fc(r)).
 * The derivative with respect to r is analytic: S depends only on distance ratios (fixed by the scaled
 * lattice), so each neighbour contributes d/dr [S_m a_n(r_m) u...] with dr_m/dr = r_m/r.
 * Measured with native LAMMPS (black box): bcc and dia crystal energies (dia has Gamma != 0 in the
 * reference) agree with this background to about 1e-11 relative (see tests/enginePairMeam15.test.ts).
 * Without explicit vectors (nn2 = 0) the reference holds the first neighbour shell only. Measured with
 * native LAMMPS (black box): with Cmin(1,1,1) = 0.70 the fcc second shell (four screening atoms at C = 1)
 * is partly screened, and native's energies, per-atom energies and forces (w33meamref_trimer,
 * w33meamref_fcc) match to 1e-14 only when that shell is left out of the reference; with the default
 * Cmin = 2 the second shell is fully screened, so both forms agree there.
 */
const refRhoBarPrime = (
  el: MeamElement,
  o: MeamOptions,
  r: number,
  vectors?: ReferenceVector[],
): { rho: number; drho: number; rho0: number } => {
  const t = refTuple(el, o);
  const nb: MeamNeighbor[] = (vectors ?? referenceVectors(el.lat ?? 'fcc', r, r * (1 + 1e-6))).map((v) => ({ j: -1, ...v }));
  const S = screening(nb, o);
  let rho0 = 0, drho0 = 0, s2 = 0, ds2 = 0;
  const v1 = [0, 0, 0], dv1 = [0, 0, 0], v3 = [0, 0, 0], dv3 = [0, 0, 0];
  const V2 = new Float64Array(9), dV2 = new Float64Array(9), V3 = new Float64Array(27), dV3 = new Float64Array(27);
  for (let m = 0; m < nb.length; m++) {
    const p = nb[m];
    const rm = p.r, sm = rm / r;
    // Reference structure: the radial cutoff is not applied to its shells (see the header of this function).
    const wt = S[m], dwt = 0;
    const u = [p.dx / rm, p.dy / rm, p.dz / rm];
    const a = [0, 1, 2, 3].map((n) => Math.exp(-el.beta[n] * (rm / el.re - 1)));
    // weights W_n = wt a_n and their r-derivatives dW_n = dwt a_n + wt (-beta_n/re) a_n s_m
    const W = [0, 1, 2, 3].map((n) => wt * a[n]);
    const dW = [0, 1, 2, 3].map((n) => dwt * a[n] + wt * (-el.beta[n] / el.re) * a[n] * sm);
    rho0 += W[0]; drho0 += dW[0];
    s2 += W[2]; ds2 += dW[2];
    for (let c = 0; c < 3; c++) {
      v1[c] += W[1] * u[c]; dv1[c] += dW[1] * u[c];
      v3[c] += W[3] * u[c]; dv3[c] += dW[3] * u[c];
    }
    for (let q = 0; q < 9; q++) {
      const uu = u[Math.floor(q / 3)] * u[q % 3];
      V2[q] += W[2] * uu; dV2[q] += dW[2] * uu;
    }
    for (let q = 0; q < 27; q++) {
      const uuu = u[Math.floor(q / 9)] * u[Math.floor(q / 3) % 3] * u[q % 3];
      V3[q] += W[3] * uuu; dV3[q] += dW[3] * uuu;
    }
  }
  if (rho0 <= 0) return { rho: 0, drho: 0, rho0: 0 };
  // rho1^2 = |v1|^2, rho2^2 = sum V2^2 - s2^2/3, rho3^2 = sum V3^2 - (3/5)|v3|^2
  const rho1sq = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
  const drho1sq = 2 * (v1[0] * dv1[0] + v1[1] * dv1[1] + v1[2] * dv1[2]);
  let V2sq = 0, dV2sq = 0;
  for (let q = 0; q < 9; q++) { V2sq += V2[q] * V2[q]; dV2sq += V2[q] * dV2[q]; }
  const rho2sq = V2sq - (s2 * s2) / 3;
  const drho2sq = 2 * dV2sq - (2 * s2 * ds2) / 3;
  let V3sq = 0, dV3sq = 0;
  for (let q = 0; q < 27; q++) { V3sq += V3[q] * V3[q]; dV3sq += V3[q] * dV3[q]; }
  const v3sq = v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2];
  const dv3sq = 2 * (v3[0] * dv3[0] + v3[1] * dv3[1] + v3[2] * dv3[2]);
  const rho3sq = V3sq - (3 / 5) * v3sq;
  const drho3sq = 2 * dV3sq - (3 / 5) * dv3sq;
  const Q = t[1] * rho1sq + t[2] * rho2sq + t[3] * rho3sq;
  const dQ = t[1] * drho1sq + t[2] * drho2sq + t[3] * drho3sq;
  const gam = Q / (rho0 * rho0);
  const dgam = dQ / (rho0 * rho0) - (2 * Q * drho0) / (rho0 * rho0 * rho0);
  const G = gOf(el.ibar, gam), Gp = gPrimeOf(el.ibar, gam);
  return { rho: rho0 * G, drho: drho0 * G + rho0 * Gp * dgam, rho0 };
};

/** Background density of the reference structure at nearest-neighbour distance r (rho0 G(Gamma)). */
const refBackground = (el: MeamElement, o: MeamOptions, r: number): number => refRhoBarPrime(el, o, r).rho;

/*
 * Embedding normalisation rho_ref of the element: the reference rho0 at re, WITHOUT the G(Gamma) factor.
 * Measured with native LAMMPS (black box): for the diamond reference (Gamma != 0 in the reference) the A-atom dimers
 * at 2.2, 2.4 and 2.6 A (tests/enginePairMeam15.test.ts) agree to 1e-10 eV only with this normalisation; the
 * pair term keeps the full background rho0 G(Gamma) of the reference at distance r.
 */
export const referenceBackground = (el: MeamElement, o: MeamOptions): number =>
  refRhoBarPrime(el, o, el.re, el.nn2 ? nn2Vectors(el.lat ?? 'fcc', el.re) : undefined).rho0;

/*
 * 2NN (nn2 = 1) pair term of Lee-Baskes (Phys. Rev. B 62, 8564 (2000); docs.lammps.org/pair_meam.html
 * "nn2(I,J)    = turn on second-nearest neighbor MEAM formulation for I-J pair"). The reference
 * structure of the pair term holds exactly its first two neighbour shells; with the screening
 * weights S of that structure the reference energy per atom is
 *   E_u(r) = F(rho_bar(r)) + (z1/2) phi(r) + (z2/2) phi(a r),
 * z1 = sum S over the first shell and z2 = sum S over the second shell (both independent of r
 * because S depends only on distance ratios). Solved for phi:
 *   phi(r) = (2/z1) sum_{n>=0} (-(z2/z1))^n [E_rose(a^n r) - F(rho_bar(a^n r))].
 * Measured with native LAMMPS (black box): for the WL library entry (bcc, ibar = 3) and a
 * synthetic sc entry this sum reproduces the native dimer pair term to the native table
 * precision; the reference energy erose(r) is exactly the ideal reference-crystal energy.
 */
const nn2Vectors = (lat: ReferenceLattice, r: number): ReferenceVector[] =>
  referenceVectors(lat, r, PAIR_SHELL_RATIO[lat] * r * (1 + 1e-6));

/** a, z1, z2 of the reference lattice (screening weights of the first two shells). */
export const nn2RefData = (lat: ReferenceLattice, o: MeamOptions): Nn2Ref => {
  const nb: MeamNeighbor[] = nn2Vectors(lat, 1).map((v) => ({ j: -1, ...v }));
  const S = screening(nb, o);
  let r1 = Infinity, r2 = 0;
  for (const p of nb) {
    if (p.r < r1) r1 = p.r;
    if (p.r > r2) r2 = p.r;
  }
  const n = nb.length;
  const dir = new Float64Array(3 * n), ratio = new Float64Array(n);
  let z1 = 0, z2 = 0;
  for (let m = 0; m < n; m++) {
    const p = nb[m];
    ratio[m] = p.r;
    dir[3 * m] = p.dx / p.r; dir[3 * m + 1] = p.dy / p.r; dir[3 * m + 2] = p.dz / p.r;
    if (Math.abs(p.r - r1) < 1e-9) z1 += S[m];
    else z2 += S[m];
  }
  return { a: r2 / r1, z1, z2, dir, ratio, s: S };
};

/*
 * Background density rho0 G(Gamma) of the two-shell reference structure at nearest-neighbour
 * distance ar and its r-derivative, from the precomputed unit-scale shells. Mirror of
 * refRhoBarPrime with the screening constant (S depends only on distance ratios, so the
 * reference's screening weight does not depend on r; d r_m/d r = ratio_m).
 */
const nn2RhoBarPrime = (
  el: MeamElement,
  o: MeamOptions,
  ref: Nn2Ref,
  r: number,
): { rho: number; drho: number; rho0: number } => {
  const t = refTuple(el, o);
  const n = ref.ratio.length;
  let rho0 = 0, drho0 = 0, s2 = 0, ds2 = 0;
  const v1 = [0, 0, 0], dv1 = [0, 0, 0], v3 = [0, 0, 0], dv3 = [0, 0, 0];
  const V2 = new Float64Array(9), dV2 = new Float64Array(9), V3 = new Float64Array(27), dV3 = new Float64Array(27);
  for (let m = 0; m < n; m++) {
    const sm = ref.ratio[m];
    const rm = r * sm;
    const wt = ref.s[m];
    const u = [ref.dir[3 * m], ref.dir[3 * m + 1], ref.dir[3 * m + 2]];
    const a = [0, 1, 2, 3].map((q) => Math.exp(-el.beta[q] * (rm / el.re - 1)));
    const W = [0, 1, 2, 3].map((q) => wt * a[q]);
    const dW = [0, 1, 2, 3].map((q) => wt * (-el.beta[q] / el.re) * a[q] * sm);
    rho0 += W[0]; drho0 += dW[0];
    s2 += W[2]; ds2 += dW[2];
    for (let c = 0; c < 3; c++) {
      v1[c] += W[1] * u[c]; dv1[c] += dW[1] * u[c];
      v3[c] += W[3] * u[c]; dv3[c] += dW[3] * u[c];
    }
    for (let q = 0; q < 9; q++) {
      const uu = u[Math.floor(q / 3)] * u[q % 3];
      V2[q] += W[2] * uu; dV2[q] += dW[2] * uu;
    }
    for (let q = 0; q < 27; q++) {
      const uuu = u[Math.floor(q / 9)] * u[Math.floor(q / 3) % 3] * u[q % 3];
      V3[q] += W[3] * uuu; dV3[q] += dW[3] * uuu;
    }
  }
  if (rho0 <= 0) return { rho: 0, drho: 0, rho0: 0 };
  const rho1sq = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
  const drho1sq = 2 * (v1[0] * dv1[0] + v1[1] * dv1[1] + v1[2] * dv1[2]);
  let V2sq = 0, dV2sq = 0;
  for (let q = 0; q < 9; q++) { V2sq += V2[q] * V2[q]; dV2sq += V2[q] * dV2[q]; }
  const rho2sq = V2sq - (s2 * s2) / 3;
  const drho2sq = 2 * dV2sq - (2 * s2 * ds2) / 3;
  let V3sq = 0, dV3sq = 0;
  for (let q = 0; q < 27; q++) { V3sq += V3[q] * V3[q]; dV3sq += V3[q] * dV3[q]; }
  const v3sq = v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2];
  const dv3sq = 2 * (v3[0] * dv3[0] + v3[1] * dv3[1] + v3[2] * dv3[2]);
  const rho3sq = V3sq - (3 / 5) * v3sq;
  const drho3sq = 2 * dV3sq - (3 / 5) * dv3sq;
  const Q = t[1] * rho1sq + t[2] * rho2sq + t[3] * rho3sq;
  const dQ = t[1] * drho1sq + t[2] * drho2sq + t[3] * drho3sq;
  const gam = Q / (rho0 * rho0);
  const dgam = dQ / (rho0 * rho0) - (2 * Q * drho0) / (rho0 * rho0 * rho0);
  const G = gOf(el.ibar, gam), Gp = gPrimeOf(el.ibar, gam);
  return { rho: rho0 * G, drho: drho0 * G + rho0 * Gp * dgam, rho0 };
};

/** F(rho_bar) of the two-shell reference structure at nearest-neighbour distance r. */
const nn2RefEmbedding = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number =>
  embedding(el, rhoRef, nn2RhoBarPrime(el, o, el.nn2!, r).rho);

/** 2NN pair term phi(r) (see the recurrence note above). */
export const nn2Phi = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number => {
  const { a, z1, z2 } = el.nn2!;
  const q = z2 / z1;
  let s = 0, ar = r, sign = 1;
  for (let n = 0; n < 500; n++) {
    s += sign * (2 / z1) * (eroseE(el, ar) - nn2RefEmbedding(el, o, rhoRef, ar));
    sign *= -q;
    ar *= a;
    if (ar > 60) break;
  }
  return s;
};

/** d phi / dr of the 2NN pair term (term-by-term derivative of nn2Phi). */
export const nn2PhiPrime = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number => {
  const { a, z1, z2 } = el.nn2!;
  const q = z2 / z1;
  let s = 0, ar = r, sign = 1, scale = 1;
  for (let n = 0; n < 500; n++) {
    const { rho, drho } = nn2RhoBarPrime(el, o, el.nn2!, ar);
    const fp = rho > 0 ? (el.A * el.Ec * (Math.log(rho / rhoRef) + 1)) / rhoRef : 0;
    s += sign * (2 / z1) * scale * (eroseDeriv(el, ar) - fp * drho);
    sign *= -q;
    ar *= a;
    scale *= a;
    if (ar > 60) break;
  }
  return s;
};

/** F(rhobar) = A Ec (rhobar/rhoRef) ln(rhobar/rhoRef). */
export const embedding = (el: MeamElement, rhoRef: number, rhoBar: number): number => {
  if (rhoBar <= 0) return 0;
  const x = rhoBar / rhoRef;
  return el.A * el.Ec * x * Math.log(x);
};

/*
 * Rose energy erose(r) (docs, pair_meam.rst):
 *   "astar = alpha \* (r/re - 1.d0)"
 *   "if erose_form = 0: erose = -Ec\*(1+astar+a3\*(astar\*\*3)/(r/re))\*exp(-astar)"
 *   "if erose_form = 1: erose = -Ec\*(1+astar+(-attrac+repuls/r)\*(astar\*\*3))\*exp(-astar)"
 *   "if erose_form = 2: erose = -Ec\*(1 +astar + a3\*(astar\*\*3))\*exp(-astar)"
 *   "a3 = repuls, astar < 0"
 *   "a3 = attrac, astar >= 0"
 * Derivative with respect to r: d/dr[-Ec(1+s+T)e^{-s}] = -Ec e^{-s}[T' - (s+T) s'], s' = alpha/re.
 */
const eroseA3 = (e: EroseSettings, astar: number): number => (astar < 0 ? e.repuls : e.attrac);
export const eroseE = (el: MeamElement, r: number): number => {
  const e = el.erose ?? NO_EROSE;
  const q = r / el.re;
  const s = el.alpha * (q - 1);
  let T: number;
  if (e.form === 0) T = (eroseA3(e, s) * s ** 3) / q;
  else if (e.form === 1) T = (-e.attrac + e.repuls / r) * s ** 3;
  else T = eroseA3(e, s) * s ** 3;
  return -el.Ec * (1 + s + T) * Math.exp(-s);
};
export const eroseDeriv = (el: MeamElement, r: number): number => {
  const e = el.erose ?? NO_EROSE;
  const q = r / el.re;
  const s = el.alpha * (q - 1);
  const sp = el.alpha / el.re;
  const a3 = eroseA3(e, s);
  let T: number, Tp: number;
  if (e.form === 0) {
    T = (a3 * s ** 3) / q;
    Tp = a3 * (3 * s * s * sp / q - (s ** 3) / (el.re * q * q));
  } else if (e.form === 1) {
    T = (-e.attrac + e.repuls / r) * s ** 3;
    Tp = (-e.repuls / (r * r)) * s ** 3 + (-e.attrac + e.repuls / r) * 3 * s * s * sp;
  } else {
    T = a3 * s ** 3;
    Tp = a3 * 3 * s * s * sp;
  }
  return -el.Ec * Math.exp(-s) * (Tp - (s + T) * sp);
};

/** Pair term phi(r) from the reference structure (Rose energy erose and the reference background). */
export const pairPhi = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number =>
  (2 / el.z) * (eroseE(el, r) - embedding(el, rhoRef, refBackground(el, o, r)));

/*
 * Tabulated pair term (production path and the helper). Measured with native LAMMPS (black box): the pair
 * term is read from a table of 1000 uniform intervals over [0, 1.1 rc], with nodes at k * dr, dr = 1.1 rc / 1000,
 * and the piecewise cubic of eam.ts (Hermite form with five-point finite-difference slopes) gives the energy and
 * the force of a bcc dimer inside the window to 1e-13 eV and 2e-13 eV/A (1500 dimer distances, 2.3 to 2.75 A);
 * the analytic pair term leaves a force residual of 6e-8 eV/A. dr = 1.1 rc / 999 does not fit (1e-7 eV/A).
 * Nodes below 1 A are not tabulated (see PHI_LO); pairs below that distance use the analytic pair term.
 */


const phiTables = new WeakMap<MeamElement, { rc: number; delr: number; Cmin: number; Cmax: number; augt1: boolean; rhoRef: number; tab: PhiTable }>();

/** The pair-term table of an element (cached per element and options). */
const phiTableOf = (el: MeamElement, o: MeamOptions, rhoRef: number): PhiTable => {
  // Cache keyed by the element object (its parameters do not change) and the numbers the table depends on.
  const hit = phiTables.get(el);
  if (hit && hit.rc === o.rc && hit.delr === o.delr && hit.Cmin === o.Cmin && hit.Cmax === o.Cmax && hit.augt1 === o.augt1 && hit.rhoRef === rhoRef) return hit.tab;
  const dr = (PHI_SPAN * o.rc) / PHI_INTERVALS;
  const kLo = Math.ceil(PHI_LO / dr);
  const y = new Float64Array(PHI_INTERVALS + 1);
  const oT = phiOpts(o);
  // The 2NN recurrence is defined by the reference structure (not by rc), so it needs no rc extension.
  for (let k = kLo; k <= PHI_INTERVALS; k++) y[k] = el.nn2 ? nn2Phi(el, o, rhoRef, k * dr) : pairPhi(el, oT, rhoRef, k * dr);
  for (let k = 0; k < kLo; k++) y[k] = y[kLo];
  const tab = new PhiTable(y, dr);
  phiTables.set(el, { rc: o.rc, delr: o.delr, Cmin: o.Cmin, Cmax: o.Cmax, augt1: o.augt1, rhoRef, tab });
  return tab;
};

/** Tabulated pair term and its derivative (the values used by the energy and the forces). */
/**
 * Options of the pair term: the reference shells of phi are summed up to PHI_SPAN rc (not rc), so phi is continuous
 * across rc; a pair is only ever evaluated inside rc. Measured with native LAMMPS (black box): with shells cut at rc
 * the table is off by 1e-4 eV and 0.5 eV/A on the bcc dimers at 3.99 to 3.9996 A (the reference's first shell
 * switches off at rc inside the last table intervals), while with the cut at 1.1 rc it agrees to 1e-14.
 */
const phiOpts = (o: MeamOptions): MeamOptions => ({ ...o, rc: PHI_SPAN * o.rc });

const tabulated = (o: MeamOptions, r: number): boolean => r >= (Math.ceil(PHI_LO / ((PHI_SPAN * o.rc) / PHI_INTERVALS)) + 2) * ((PHI_SPAN * o.rc) / PHI_INTERVALS);
export const pairPhiTab = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number =>
  tabulated(o, r) ? phiTableOf(el, o, rhoRef).eval(r) : el.nn2 ? nn2Phi(el, o, rhoRef, r) : pairPhi(el, phiOpts(o), rhoRef, r);
export const pairPhiTabPrime = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number =>
  tabulated(o, r) ? phiTableOf(el, o, rhoRef).deriv(r) : el.nn2 ? nn2PhiPrime(el, o, rhoRef, r) : pairPhiPrime(el, phiOpts(o), rhoRef, r);

/** Total energy of a periodic orthorhombic configuration (brute-force neighbour images). */
export function meamEnergy(el: MeamElement, o: MeamOptions, x: Float64Array, L: [number, number, number]): number {
  const n = x.length / 3;
  const t1 = o.augt1 ? el.t[1] + 0.6 * el.t[3] : el.t[1];
  const t = [1, t1, el.t[2], el.t[3]];
  const rhoRef = referenceBackground(el, o);
  const nimg = [Math.ceil(o.rc / L[0]), Math.ceil(o.rc / L[1]), Math.ceil(o.rc / L[2])];
  let E = 0;
  for (let i = 0; i < n; i++) {
    const nb: MeamNeighbor[] = [];
    for (let j = 0; j < n; j++)
      for (let ia = -nimg[0]; ia <= nimg[0]; ia++)
        for (let ib = -nimg[1]; ib <= nimg[1]; ib++)
          for (let ic = -nimg[2]; ic <= nimg[2]; ic++) {
            if (i === j && ia === 0 && ib === 0 && ic === 0) continue;
            const dx = x[3 * j] - x[3 * i] + ia * L[0];
            const dy = x[3 * j + 1] - x[3 * i + 1] + ib * L[1];
            const dz = x[3 * j + 2] - x[3 * i + 2] + ic * L[2];
            const r = Math.hypot(dx, dy, dz);
            if (r >= o.rc) continue;
            nb.push({ j, dx, dy, dz, r });
          }
    const S = screening(nb, o);
    let rho0 = 0, v1x = 0, v1y = 0, v1z = 0, rho2s = 0, rho3s = 0, v3vx = 0, v3vy = 0, v3vz = 0;
    const v2 = new Float64Array(9), v3 = new Float64Array(27);
    for (let m = 0; m < nb.length; m++) {
      const p = nb[m];
      const w = radialWeight(p.r, o) * S[m];
      const r = p.r;
      const u = [p.dx / r, p.dy / r, p.dz / r];
      const a0 = Math.exp(-el.beta[0] * (r / el.re - 1));
      const a1 = Math.exp(-el.beta[1] * (r / el.re - 1));
      const a2 = Math.exp(-el.beta[2] * (r / el.re - 1));
      const a3 = Math.exp(-el.beta[3] * (r / el.re - 1));
      rho0 += w * a0;
      v1x += w * a1 * u[0]; v1y += w * a1 * u[1]; v1z += w * a1 * u[2];
      rho2s += w * a2;
      for (let q = 0; q < 9; q++) v2[q] += w * a2 * u[Math.floor(q / 3)] * u[q % 3];
      rho3s += w * a3;
      for (let q = 0; q < 27; q++) v3[q] += w * a3 * u[Math.floor(q / 9)] * u[Math.floor(q / 3) % 3] * u[q % 3];
      v3vx += w * a3 * u[0]; v3vy += w * a3 * u[1]; v3vz += w * a3 * u[2];
    }
    if (rho0 <= 0) continue;
    const rho1sq = v1x * v1x + v1y * v1y + v1z * v1z;
    let rho2sq = 0;
    for (let q = 0; q < 9; q++) rho2sq += v2[q] * v2[q];
    rho2sq -= (rho2s * rho2s) / 3;
    let rho3sq = 0;
    for (let q = 0; q < 27; q++) rho3sq += v3[q] * v3[q];
    rho3sq -= (3 / 5) * (v3vx * v3vx + v3vy * v3vy + v3vz * v3vz);
    const gamma = (t[1] * rho1sq + t[2] * rho2sq + t[3] * rho3sq) / (rho0 * rho0);
    E += embedding(el, rhoRef, rho0 * gOf(el.ibar, gamma));
  }
  for (let i = 0; i < n; i++) {
    const nb: MeamNeighbor[] = [];
    for (let j = 0; j < n; j++)
      for (let ia = -nimg[0]; ia <= nimg[0]; ia++)
        for (let ib = -nimg[1]; ib <= nimg[1]; ib++)
          for (let ic = -nimg[2]; ic <= nimg[2]; ic++) {
            if (i === j && ia === 0 && ib === 0 && ic === 0) continue;
            const dx = x[3 * j] - x[3 * i] + ia * L[0];
            const dy = x[3 * j + 1] - x[3 * i + 1] + ib * L[1];
            const dz = x[3 * j + 2] - x[3 * i + 2] + ic * L[2];
            const r = Math.hypot(dx, dy, dz);
            if (r >= o.rc) continue;
            nb.push({ j, dx, dy, dz, r });
          }
    const S = screening(nb, o);
    for (let m = 0; m < nb.length; m++) E += 0.5 * S[m] * radialWeight(nb[m].r, o) * pairPhiTab(el, o, rhoRef, nb[m].r);
  }
  return E;
}

/*
 * Analytic forces (task 1). Same model as meamEnergy; the derivation (textbook chain rule,
 * no LAMMPS source):
 *   W_m = fc(r_m) S_m, rho0 = sum W a0, v1 = sum W a1 u, V2 = sum W a2 uu, s2 = sum W a2,
 *   V3 = sum W a3 uuu, v3 = sum W a3 u, with u = d/r and a_n = exp(-beta_n (r/re - 1)).
 *   E_i = F(rho0 G(Gamma)) + 1/2 sum_m W_m phi(r_m).
 * Partial derivatives are taken with respect to the per-neighbour vector d_k; S_m depends on d_k
 * through the screening pair factors f_mk(A = r_k^2/r_m^2, B = |d_k-d_m|^2/r_m^2), with the
 * product over the other screeners obtained by a prefix/suffix product (no division by zero).
 */

const polyPrime = (x: number): number => {
  if (x <= 0 || x >= 1) return 0;
  const q = (1 - x) ** 4;
  return 8 * (1 - x) ** 3 * (1 - q);
};
const radialWeightPrime = (r: number, o: MeamOptions): number => polyPrime((o.rc - r) / o.delr) * (-1 / o.delr);
const screenWeightPrime = (C: number, o: MeamOptions): number =>
  C >= o.Cmax || C <= o.Cmin ? 0 : polyPrime((C - o.Cmin) / (o.Cmax - o.Cmin)) / (o.Cmax - o.Cmin);

export const gPrimeOf = gPrimeOfIbar;

/** rho_ref(r) of the reference structure and its r-derivative (see refRhoBarPrime). */
export const refBackgroundPrime = (el: MeamElement, o: MeamOptions, r: number): { rho: number; drho: number } =>
  refRhoBarPrime(el, o, r);

/** d phi / dr for the fcc-reference pair term (same form as pairPhi). */
export const pairPhiPrime = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number => {
  const dEu = eroseDeriv(el, r);
  const { rho, drho } = refBackgroundPrime(el, o, r);
  const fp = rho > 0 ? (el.A * el.Ec * (Math.log(rho / rhoRef) + 1)) / rhoRef : 0;
  return (2 / el.z) * (dEu - fp * drho);
};

/**
 * Energy of one atom i and dE_i/d(d_m) for each neighbour vector d_m (written to g, length 3N).
 * Includes this atom's pair term 1/2 sum_m W_m phi(r_m), which is evaluated even when rho0 <= 0.
 */
export function meamAtomEnergyGrad(
  el: MeamElement,
  o: MeamOptions,
  rhoRef: number,
  t: [number, number, number, number],
  nb: MeamNeighbor[],
  g: Float64Array,
): number {
  const N = nb.length;
  g.fill(0, 0, 3 * N);
  if (N === 0) return 0;
  const eb = ebound(o);
  const r = new Float64Array(N), D = new Float64Array(3 * N), U = new Float64Array(3 * N);
  for (let m = 0; m < N; m++) {
    r[m] = nb[m].r;
    D[3 * m] = nb[m].dx; D[3 * m + 1] = nb[m].dy; D[3 * m + 2] = nb[m].dz;
    for (let c = 0; c < 3; c++) U[3 * m + c] = D[3 * m + c] / r[m];
  }
  // Screening pair factors f_mk and the derivatives of C with respect to A and B (times sw').
  const f = new Float64Array(N * N).fill(1);
  const cA = new Float64Array(N * N), cB = new Float64Array(N * N);
  for (let m = 0; m < N; m++) {
    const rm2 = r[m] * r[m];
    for (let k = 0; k < N; k++) {
      if (k === m) continue;
      const A = (r[k] * r[k]) / rm2;
      const ex = D[3 * k] - D[3 * m], ey = D[3 * k + 1] - D[3 * m + 1], ez = D[3 * k + 2] - D[3 * m + 2];
      const B = (ex * ex + ey * ey + ez * ez) / rm2;
      if (A > eb || B > eb) continue;
      const dd = A - B;
      const Dn = 1 - dd * dd;
      const Nn = 2 * (A + B) - dd * dd - 1;
      const C = Nn / Dn;
      const sp = screenWeightPrime(C, o);
      f[m * N + k] = screenWeight(C, o);
      cA[m * N + k] = (sp * ((2 - 2 * dd) * Dn + 2 * dd * Nn)) / (Dn * Dn);
      cB[m * N + k] = (sp * ((2 + 2 * dd) * Dn - 2 * dd * Nn)) / (Dn * Dn);
    }
  }
  // S_m (product over screeners) and R_mk = product over screeners other than k (prefix/suffix).
  const S = new Float64Array(N), R = new Float64Array(N * N), pre = new Float64Array(N);
  for (let m = 0; m < N; m++) {
    let p = 1;
    for (let k = 0; k < N; k++) {
      pre[k] = p;
      if (k !== m) p *= f[m * N + k];
    }
    S[m] = p;
    let s = 1;
    for (let k = N - 1; k >= 0; k--) {
      R[m * N + k] = pre[k] * s;
      if (k !== m) s *= f[m * N + k];
    }
  }
  const fc = new Float64Array(N), fcp = new Float64Array(N), W = new Float64Array(N);
  const phi = new Float64Array(N), phip = new Float64Array(N);
  const a = [new Float64Array(N), new Float64Array(N), new Float64Array(N), new Float64Array(N)];
  const ap = [new Float64Array(N), new Float64Array(N), new Float64Array(N), new Float64Array(N)];
  for (let m = 0; m < N; m++) {
    fc[m] = radialWeight(r[m], o);
    fcp[m] = radialWeightPrime(r[m], o);
    W[m] = fc[m] * S[m];
    phi[m] = pairPhiTab(el, o, rhoRef, r[m]);
    phip[m] = pairPhiTabPrime(el, o, rhoRef, r[m]);
    for (let n = 0; n < 4; n++) {
      a[n][m] = Math.exp(-el.beta[n] * (r[m] / el.re - 1));
      ap[n][m] = (-el.beta[n] / el.re) * a[n][m];
    }
  }
  let Eden = 0;
  const gm = new Float64Array(N);
  const hv = new Float64Array(3 * N);
  let pairE = 0;
  for (let m = 0; m < N; m++) pairE += 0.5 * W[m] * phi[m];

  // Density sums.
  let rho0 = 0, s2 = 0;
  const v1 = [0, 0, 0], v3 = [0, 0, 0], V2 = new Float64Array(9), V3 = new Float64Array(27);
  for (let m = 0; m < N; m++) {
    const w = W[m], u0 = U[3 * m], u1 = U[3 * m + 1], u2 = U[3 * m + 2];
    const uu = [u0, u1, u2];
    rho0 += w * a[0][m];
    for (let c = 0; c < 3; c++) {
      v1[c] += w * a[1][m] * uu[c];
      v3[c] += w * a[3][m] * uu[c];
    }
    s2 += w * a[2][m];
    for (let p = 0; p < 3; p++)
      for (let q = 0; q < 3; q++) V2[3 * p + q] += w * a[2][m] * uu[p] * uu[q];
    for (let p = 0; p < 3; p++)
      for (let q = 0; q < 3; q++)
        for (let s = 0; s < 3; s++) V3[9 * p + 3 * q + s] += w * a[3][m] * uu[p] * uu[q] * uu[s];
  }
  let G0 = 0, G1 = 0, G2 = 0, G3 = 0;
  let Phi1 = [0, 0, 0], Phi2 = new Float64Array(9), Phi2s = 0, Phi3 = new Float64Array(27), Phiv = [0, 0, 0];
  if (rho0 > 0) {
    const rho1sq = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    let V2sq = 0;
    for (let q = 0; q < 9; q++) V2sq += V2[q] * V2[q];
    const rho2sq = V2sq - (s2 * s2) / 3;
    let V3sq = 0;
    for (let q = 0; q < 27; q++) V3sq += V3[q] * V3[q];
    const v3sq = v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2];
    const rho3sq = V3sq - (3 / 5) * v3sq;
    const gam = (t[1] * rho1sq + t[2] * rho2sq + t[3] * rho3sq) / (rho0 * rho0);
    const Gg = gOf(el.ibar, gam), Gp = gPrimeOf(el.ibar, gam);
    const rb = rho0 * Gg;
    if (rb > 0) {
      const x = rb / rhoRef;
      Eden = el.A * el.Ec * x * Math.log(x);
      const Fp = (el.A * el.Ec * (Math.log(x) + 1)) / rhoRef;
      G0 = Fp * (Gg - 2 * gam * Gp);
      G1 = (Fp * Gp * t[1]) / rho0;
      G2 = (Fp * Gp * t[2]) / rho0;
      G3 = (Fp * Gp * t[3]) / rho0;
      Phi1 = [2 * G1 * v1[0], 2 * G1 * v1[1], 2 * G1 * v1[2]];
      for (let q = 0; q < 9; q++) Phi2[q] = 2 * G2 * V2[q];
      Phi2s = -(2 / 3) * G2 * s2;
      for (let q = 0; q < 27; q++) Phi3[q] = 2 * G3 * V3[q];
      Phiv = [-(6 / 5) * G3 * v3[0], -(6 / 5) * G3 * v3[1], -(6 / 5) * G3 * v3[2]];
    }
  }
  const hasDen = G0 !== 0 || G1 !== 0 || G2 !== 0 || G3 !== 0;
  for (let m = 0; m < N; m++) {
    const u = [U[3 * m], U[3 * m + 1], U[3 * m + 2]];
    let gden = 0, sc = 0;
    let vperp = [0, 0, 0];
    if (hasDen) {
      const Phi1u = Phi1[0] * u[0] + Phi1[1] * u[1] + Phi1[2] * u[2];
      let uPhi2u = 0;
      const Phi2u = [0, 0, 0];
      for (let p = 0; p < 3; p++)
        for (let q = 0; q < 3; q++) {
          Phi2u[p] += Phi2[3 * p + q] * u[q];
          uPhi2u += u[p] * Phi2[3 * p + q] * u[q];
        }
      let Phi3uuu = 0;
      const psi = [0, 0, 0];
      for (let p = 0; p < 3; p++)
        for (let q = 0; q < 3; q++)
          for (let s = 0; s < 3; s++) {
            psi[p] += Phi3[9 * p + 3 * q + s] * u[q] * u[s];
          }
      for (let p = 0; p < 3; p++) Phi3uuu += u[p] * psi[p];
      const Phivu = Phiv[0] * u[0] + Phiv[1] * u[1] + Phiv[2] * u[2];
      const cs2 = uPhi2u + Phi2s;
      gden = G0 * a[0][m] + a[1][m] * Phi1u + a[2][m] * cs2 + a[3][m] * (Phi3uuu + Phivu);
      sc = G0 * ap[0][m] + ap[1][m] * Phi1u + ap[2][m] * cs2 + ap[3][m] * (Phi3uuu + Phivu);
      const vp = [0, 0, 0];
      for (let c = 0; c < 3; c++)
        vp[c] = a[1][m] * Phi1[c] + 2 * a[2][m] * Phi2u[c] + 3 * a[3][m] * psi[c] + a[3][m] * Phiv[c];
      const uvp = u[0] * vp[0] + u[1] * vp[1] + u[2] * vp[2];
      vperp = [vp[0] - u[0] * uvp, vp[1] - u[1] * uvp, vp[2] - u[2] * uvp];
    }
    gm[m] = gden + 0.5 * phi[m];
    const radial = W[m] * (sc + 0.5 * phip[m]);
    for (let c = 0; c < 3; c++) hv[3 * m + c] = radial * u[c] + (W[m] * vperp[c]) / r[m];
  }
  // Chain rule through W_m = fc(r_m) S_m: explicit fc' term, and the screening terms.
  for (let k = 0; k < N; k++) {
    const pref = gm[k] * fcp[k] * S[k];
    for (let c = 0; c < 3; c++) g[3 * k + c] = hv[3 * k + c] + pref * U[3 * k + c];
  }
  for (let m = 0; m < N; m++) {
    const rm2 = r[m] * r[m];
    const cm = gm[m] * fc[m];
    if (cm === 0) continue;
    for (let k = 0; k < N; k++) {
      if (k === m) continue;
      const idx = m * N + k;
      if (cA[idx] === 0 && cB[idx] === 0) continue;
      const cw = cm * R[idx];
      const A = (r[k] * r[k]) / rm2;
      const ex = D[3 * k] - D[3 * m], ey = D[3 * k + 1] - D[3 * m + 1], ez = D[3 * k + 2] - D[3 * m + 2];
      const B = (ex * ex + ey * ey + ez * ez) / rm2;
      for (let c = 0; c < 3; c++) {
        const dAk = (2 * D[3 * k + c]) / rm2;
        const dBk = (2 * (D[3 * k + c] - D[3 * m + c])) / rm2;
        const dAm = (-2 * A * D[3 * m + c]) / rm2;
        const dBm = (-2 * (D[3 * k + c] - D[3 * m + c])) / rm2 - (2 * B * D[3 * m + c]) / rm2;
        g[3 * k + c] += cw * (cA[idx] * dAk + cB[idx] * dBk);
        g[3 * m + c] += cw * (cA[idx] * dAm + cB[idx] * dBm);
      }
    }
  }
  return Eden + pairE;
}

/** Total energy and forces (F = -grad E) for a periodic orthorhombic configuration. */
export function meamEnergyForces(
  el: MeamElement,
  o: MeamOptions,
  x: Float64Array,
  L: [number, number, number],
): { E: number; F: Float64Array } {
  const n = x.length / 3;
  const t1 = o.augt1 ? el.t[1] + 0.6 * el.t[3] : el.t[1];
  const t: [number, number, number, number] = [1, t1, el.t[2], el.t[3]];
  const rhoRef = referenceBackground(el, o);
  const nimg = [Math.ceil(o.rc / L[0]), Math.ceil(o.rc / L[1]), Math.ceil(o.rc / L[2])];
  const F = new Float64Array(x.length);
  let E = 0;
  for (let i = 0; i < n; i++) {
    const nb: MeamNeighbor[] = [];
    for (let j = 0; j < n; j++)
      for (let ia = -nimg[0]; ia <= nimg[0]; ia++)
        for (let ib = -nimg[1]; ib <= nimg[1]; ib++)
          for (let ic = -nimg[2]; ic <= nimg[2]; ic++) {
            if (i === j && ia === 0 && ib === 0 && ic === 0) continue;
            const dx = x[3 * j] - x[3 * i] + ia * L[0];
            const dy = x[3 * j + 1] - x[3 * i + 1] + ib * L[1];
            const dz = x[3 * j + 2] - x[3 * i + 2] + ic * L[2];
            const r = Math.hypot(dx, dy, dz);
            if (r >= o.rc) continue;
            nb.push({ j, dx, dy, dz, r });
          }
    const g = new Float64Array(3 * nb.length);
    E += meamAtomEnergyGrad(el, o, rhoRef, t, nb, g);
    for (let m = 0; m < nb.length; m++) {
      const j = nb[m].j;
      for (let c = 0; c < 3; c++) {
        F[3 * j + c] -= g[3 * m + c];
        F[3 * i + c] += g[3 * m + c];
      }
    }
  }
  return { E, F };
}

/*
 * pair_style meam: parameter files (one or more elements; the alloy part is in meam_alloy.ts). Documented file formats (docs.lammps.org/pair_meam.html,
 * plans/lammps-docs/pair_meam.rst):
 *   "The first 2 arguments must be \* \* so as to span all LAMMPS atom types."
 *   "formatted as a series of entries, each of which" (then "has 19 parameters and can span multiple lines:")
 *   "elt, lat, z, ielement, atwt, alpha, b0, b1, b2, b3, alat, esub, asub,"  (then t0 ... ibar on the next line)
 *   "finds and ignores the rest." (after "LAMMPS reads the first matching entry it")
 *   "normalized to *t0 = 1.0* are supported."  (after "Note that only parameters")
 *   "typically 1.0 for single-element systems."  (rozero)
 *   "rc          = cutoff radius for cutoff function; default = 4.0"
 *   "delr        = length of smoothing distance for cutoff function; default = 0.1"
 *   "by K (I<=J); default = 2.8" (Cmax), "by K (I<=J); default = 2.0" (Cmin)
 *   "augt1           = integer flag for whether to augment t1 parameter by"  (default = 1)
 *   "zbl(I,J)    = blend the MEAM I-J pair potential with the ZBL potential for small"  (default = 1)
 *   "rho0(I)     = relative density for element I (overwrites value"
 * Single-element Ec and re when the parameter file omits them (docs: "esub     = energy per atom (eV) in the reference structure at equilibrium"
 * in the library entry, and "alat     = lattice constant of reference structure"). Measured with
 * native LAMMPS (black box): with Cu.meam, which sets neither, the energy equals the one with Ec = esub = 3.54 and
 * re = alat / sqrt(2) = 3.62 / sqrt(2) to all printed digits; so Ec defaults to esub and, for fcc, re to alat / sqrt(2).
 * The other reference lattices keep the requirement that the parameter file sets re(1,1) (not measured).
 * The NULL parameter file is still rejected.
 * Masses: measured with native LAMMPS (black box), with pair_style meam the type masses are the library atomic weights
 * (Cu: 63.54 from library.meam, whatever a mass command before pair_coeff sets), as the kinetic energy of Cu.meam with
 * mass 100 equals the one with mass 63.54 and not the one of mass 100. The engine sets the same masses in pair_coeff.
 * ibar: the docs list "0 => G = sqrt(1+Gamma)" and "1 => G = exp(Gamma/2)"; only ibar = 0 is measured here.
 */

/** Library entry fields used by this style (the 19-field entry of the docs). */
interface LibraryEntry {
  lat: string;
  z: number;
  alpha: number;
  b: [number, number, number, number];
  /** atomic weight (library column 5); native LAMMPS sets the type mass from it (measured). */
  atwt: number;
  /** lattice constant of the reference structure (library column 11). */
  alat: number;
  /** energy per atom of the reference structure (library column 12). */
  esub: number;
  asub: number;
  t: [number, number, number, number];
  rozero: number;
  ibar: number;
}

const LIB_FIELDS = 19;

/** Reads the first entry whose elt matches (docs: "LAMMPS reads the first matching entry"). */
export const parseMeamLibrary = (text: string, elt: string, name: string): LibraryEntry => {
  const toks: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    const h = line.indexOf('#');
    const body = (h >= 0 ? line.slice(0, h) : line).trim();
    for (const w of body.split(/[\s,]+/)) if (w) toks.push(w.replace(/^'|'$/g, ''));
  }
  for (let p = 0; p + LIB_FIELDS <= toks.length; p += LIB_FIELDS) {
    if (toks[p] !== elt) continue;
    const v = toks.slice(p, p + LIB_FIELDS);
    const num = (k: number): number => {
      const x = Number(v[k]);
      if (!Number.isFinite(x)) throw new StyleError(`MEAM library ${name}: field ${k + 1} of '${elt}' is not a number ('${v[k]}')`);
      return x;
    };
    return {
      lat: v[1],
      z: num(2),
      alpha: num(5),
      b: [num(6), num(7), num(8), num(9)],
      atwt: num(4),
      alat: num(10),
      esub: num(11),
      asub: num(12),
      t: [num(13), num(14), num(15), num(16)],
      rozero: num(17),
      ibar: num(18),
    };
  }
  throw new StyleError(`element '${elt}' is not in MEAM library file ${name}`);
};

/** Settings from the parameter file, after the checks of the supported subset. */
interface PairParams {
  Ec?: number;
  re?: number;
  alpha?: number;
  lattce?: string;
  /** nn2(I,J): 1 selects the 2NN pair recurrence (docs: "1 = second-nearest neighbor formulation on") */
  nn2?: number;
  /** attrac(I,J) and repuls(I,J): cubic Rose-energy terms (only erose_form 1 or 2 in a multi-element potential) */
  attrac?: number;
  repuls?: number;
}

export interface MeamParams {
  /** indexed settings keyed "i,j" with i <= j (an element's own settings have i = j) */
  pair: Map<string, PairParams>;
  /** pairs with an explicit zbl(I,J) = 0 ("i,j" keys); every other pair keeps the default zbl = 1 (see ZBL_GUARD) */
  zblOff: Set<string>;
  /** erose_form, attrac(1,1), repuls(1,1) of a single element */
  erose: EroseSettings;
  /** single-element convenience: the (1,1) settings */
  Ec?: number;
  re?: number;
  alpha?: number;
  lattce?: string;
  /**
   * ialloy: t-averaging rule for the alloy density (docs pair_meam.rst: "0 = standard averaging",
   * "1 = alternative averaging", "2 = no averaging of t (use single-element values)"; default 0).
   * Only used by the multi-element path (meam_alloy.ts); a single element has one t vector.
   */
  ialloy: number;
  opts: MeamOptions;
  /**
   * Per-triplet Cmin/Cmax overrides of a multi-element potential, keyed i,j,k exactly as written
   * (docs.lammps.org/pair_meam.html "Cmin(I,J,K) ... (I<=J)"). Unused by the single-element style.
   */
  cmin3: Map<string, number>;
  cmax3: Map<string, number>;
}

const NUMERIC_DEFAULT_ZERO = ['emb_lin_neg', 'bkgd_dyn', 'mixture_ref_t'];
const PAIR_KEYS = ['Ec', 're', 'alpha', 'lattce', 'nn2', 'attrac', 'repuls', 'zbl'];

/**
 * Parses "keyword = value" and "keyword(I,J) = value" lines for nelem elements; rejects everything outside the
 * verified subset (see meam_alloy.ts for the multi-element part, and the header of this file for the rest).
 */
export const parseMeamParams = (text: string, name: string, nelem = 1): MeamParams => {
  const out: MeamParams = { pair: new Map(), zblOff: new Set(), opts: { ...DEFAULT_MEAM_OPTIONS }, erose: { form: 0, attrac: 0, repuls: 0 }, ialloy: 0, cmin3: new Map(), cmax3: new Map() };
  const pairOf = (i: number, j: number): PairParams => {
    const k = `${i},${j}`;
    let p = out.pair.get(k);
    if (!p) out.pair.set(k, (p = {}));
    return p;
  };
  for (const line of text.split(/\r?\n/)) {
    const h = line.indexOf('#');
    const body = (h >= 0 ? line.slice(0, h) : line).trim();
    if (!body) continue;
    const m = /^([A-Za-z_][A-Za-z_0-9]*)\s*(?:\(([^)]*)\))?\s*=\s*(\S+)$/.exec(body);
    if (!m) throw new StyleError(`cannot parse MEAM parameter line '${body}' in ${name}`);
    const key = m[1];
    const idx = m[2] ? m[2].split(',').map((s) => Number(s.trim())) : [];
    const val = m[3];
    const label = (_n: number) => (nelem === 1 ? '(1,1)' : `with I<=J<=${nelem}`);
    /** Validates the index tuple of a keyword with `arity` indices; returns the indices (1-based). */
    const index = (arity: number): number[] => {
      if (idx.length !== arity || idx.some((v) => !Number.isInteger(v) || v < 1 || v > nelem)) {
        if (nelem === 1 && arity === 2) throw new StyleError(`MEAM parameter ${key} in ${name} must be indexed (1,1) for a single element`);
        if (nelem === 1 && arity === 3) throw new StyleError(`MEAM parameter ${key} in ${name} must be indexed (1,1,1) for a single element`);
        if (nelem === 1 && arity === 1) throw new StyleError(`MEAM parameter ${key} in ${name} must be indexed (1) for a single element`);
        if (nelem === 1 && arity === 0) throw new StyleError(`MEAM parameter ${key} in ${name} takes no index`);
        throw new StyleError(`MEAM parameter ${key} in ${name} must be indexed ${label(nelem)} (${arity} indices)`);
      }
      return idx;
    };
    const pairIndex = (): [number, number] => {
      const [i, j] = index(2);
      if (i > j) throw new StyleError(`MEAM parameter ${key}(${i},${j}) in ${name}: I<=J is required`);
      return [i, j];
    };
    const num = (): number => {
      const x = Number(val);
      if (!Number.isFinite(x)) throw new StyleError(`MEAM parameter ${key} in ${name}: '${val}' is not a number`);
      return x;
    };
    if (PAIR_KEYS.includes(key) && idx.length === 0 && key !== 'zbl') {
      if (key === 'lattce' || key === 'Ec' || key === 're' || key === 'alpha') {
        throw new StyleError(`MEAM parameter ${key} in ${name} must be indexed by element numbers`);
      }
    }
    switch (key) {
      case 'rc':
        if (idx.length) throw new StyleError(`MEAM parameter rc in ${name} takes no index`);
        out.opts.rc = num();
        break;
      case 'delr':
        if (idx.length) throw new StyleError(`MEAM parameter delr in ${name} takes no index`);
        out.opts.delr = num();
        break;
      case 'erose_form': {
        if (idx.length) throw new StyleError(`MEAM parameter erose_form in ${name} takes no index`);
        const v = num();
        if (!Number.isInteger(v) || v < 0 || v > 2) throw new StyleError(`MEAM erose_form = ${val} is not supported (only 0, 1, 2; ${name})`);
        out.erose.form = v;
        break;
      }
      case 'attrac':
      case 'repuls': {
        const [i, j] = pairIndex();
        const v = num();
        // Multi-element attrac/repuls enter the Rose energy only through the a3 term of erose_form 1 and 2 (see
        // pairErose in meam_alloy.ts); erose_form = 0 with nonzero attrac/repuls is refused at the end below.
        if (nelem === 1) {
          if (key === 'attrac') out.erose.attrac = v;
          else out.erose.repuls = v;
        } else {
          const p = pairOf(i, j);
          if (key === 'attrac') p.attrac = v;
          else p.repuls = v;
        }
        break;
      }
      case 'Ec':
      case 're':
      case 'alpha': {
        const [i, j] = pairIndex();
        const v = num();
        const p = pairOf(i, j);
        if (key === 'Ec') p.Ec = v;
        else if (key === 're') p.re = v;
        else p.alpha = v;
        if (nelem === 1) {
          out.Ec = p.Ec;
          out.re = p.re;
          out.alpha = p.alpha;
        }
        break;
      }
      case 'Cmin':
      case 'Cmax': {
        const [i, j, k] = index(3);
        const v = num();
        if (nelem === 1) {
          // Single-element style: the one triplet Cmin(1,1,1)/Cmax(1,1,1) is the uniform limit.
          if (key === 'Cmin') out.opts.Cmin = v;
          else out.opts.Cmax = v;
        } else {
          // Multi-element: kept exactly as written; the alloy screening looks the pair up sorted (see meam_alloy.ts).
          (key === 'Cmin' ? out.cmin3 : out.cmax3).set(`${i},${j},${k}`, v);
        }
        break;
      }
      case 'augt1': {
        if (idx.length) throw new StyleError(`MEAM parameter augt1 in ${name} takes no index`);
        const x = num();
        if (x !== 0 && x !== 1) throw new StyleError(`augt1 must be 0 or 1 (${name})`);
        out.opts.augt1 = x === 1;
        break;
      }
      case 'ialloy': {
        if (idx.length) throw new StyleError(`MEAM parameter ialloy in ${name} takes no index`);
        const x = num();
        // docs.lammps.org/pair_meam.html: "0 = standard averaging", "1 = alternative averaging",
        // "2 = no averaging of t (use single-element values)"; default 0.
        if (x !== 0 && x !== 1 && x !== 2) throw new StyleError(`MEAM ialloy = ${val} is not supported (only 0, 1, 2; ${name})`);
        out.ialloy = x;
        break;
      }
      case 'nn2': {
        const [i, j] = pairIndex();
        const v = num();
        if (v !== 0 && v !== 1) throw new StyleError(`MEAM nn2(${i},${j}) = ${val} must be 0 or 1 (${name})`);
        pairOf(i, j).nn2 = v;
        break;
      }
      case 'lattce': {
        const [i, j] = pairIndex();
        if (i === j) {
          if (!SUPPORTED_REFERENCE_LATTICES.includes(val as ReferenceLattice)) {
            throw new StyleError(`MEAM lattce(${i},${j}) = ${val} is not supported (only fcc, bcc, dia, hcp, sc; ${name})`);
          }
          pairOf(i, j).lattce = val;
          if (nelem === 1) out.lattce = val;
        } else {
          if (val !== 'b1' && val !== 'dia') {
            throw new StyleError(`MEAM lattce(${i},${j}) = ${val} is not supported (only b1 and dia for an I-J pair; ${name})`);
          }
          pairOf(i, j).lattce = val;
        }
        break;
      }
      case 'zbl': {
        pairIndex();
        if (num() !== 0) throw new StyleError(`MEAM zbl(I,J) = ${val} is not supported; set zbl(I,J) = 0 (${name})`);
        out.zblOff.add(`${pairIndex()[0]},${pairIndex()[1]}`);
        break;
      }
      case 'rho0': {
        index(1);
        if (num() !== 1) throw new StyleError(`MEAM rho0 = ${val} is not supported; only 1 (${name})`);
        break;
      }
      default:
        if (NUMERIC_DEFAULT_ZERO.includes(key)) {
          const k = idx.length ? `${key}(${idx.join(',')})` : key;
          if (idx.length) pairIndex();
          if (num() !== 0) throw new StyleError(`MEAM keyword ${k} = ${val} is not supported (only the default 0; ${name})`);
          break;
        }
        throw new StyleError(`MEAM parameter keyword '${key}' is not supported (${name})`);
    }
  }
  if (nelem > 1 && out.erose.form === 0) {
    for (const [k, p] of out.pair) {
      if ((p.attrac ?? 0) !== 0 || (p.repuls ?? 0) !== 0) {
        throw new StyleError(
          `MEAM attrac(I,J) or repuls(I,J) of pair ${k} in a multi-element potential is not supported with erose_form = 0 (set erose_form = 1 or 2; ${name})`,
        );
      }
    }
  }
  if (out.opts.delr <= 0 || out.opts.rc <= out.opts.delr) throw new StyleError(`MEAM rc/delr must satisfy 0 < delr < rc (${name})`);
  return out;
};

/** pair_style meam: energy and analytic forces, full neighbour list; one element or several (meam_alloy.ts). */
export class PairMeam extends Pair {
  readonly name = 'meam';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private el: MeamElement | null = null;
  private alloy: AlloyModel | null = null;
  /** LAMMPS type (1..ntypes) -> element index of the alloy model */
  private typeElem: number[] = [];
  /** per type pair (ti * nt + tj): pairs closer than this distance are inside the unimplemented ZBL blend region (0 = no guard) */
  private zblGuard: Float64Array = new Float64Array(0);
  private opts: MeamOptions = { ...DEFAULT_MEAM_OPTIONS };
  private rhoRef = 0;
  private t: [number, number, number, number] = [1, 1, 1, 1];

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('usage: pair_style meam (no arguments)');
  }

  override modify(key: string, _values: string[]): number {
    throw new StyleError(`pair_modify ${key} is not supported for pair style meam`);
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args[0] !== '*' || args[1] !== '*') {
      throw new StyleError('the first 2 arguments of pair_coeff for style meam must be * *');
    }
    const nelem = args.length - 4 - this.ntypes;
    if (nelem < 1) {
      throw new StyleError(`pair_coeff for style meam needs the element list and one element name per atom type (${this.ntypes})`);
    }
    const elems = args.slice(3, 3 + nelem);
    const paramFile = args[3 + nelem];
    const maps = args.slice(4 + nelem);
    for (const m of maps) {
      if (m === 'NULL') throw new StyleError('NULL type mappings are not supported for pair_style meam');
      if (!elems.includes(m)) throw new StyleError(`element '${m}' is not one of the MEAM elements ${elems.join(' ')}`);
    }
    if (paramFile === 'NULL') {
      throw new StyleError('pair_coeff for style meam needs a parameter file: the NULL parameter file (zbl = 1 default, Ec and re without verified defaults) is not supported');
    }
    if (nelem > 1) {
      this.coeffAlloy(args, elems, maps, ctx);
      return;
    }
    const elem = elems[0];
    const lib = parseMeamLibrary(ctx.readFile(args[2]), elem, args[2]);
    const par = parseMeamParams(ctx.readFile(paramFile), paramFile);
    if (!SUPPORTED_REFERENCE_LATTICES.includes(lib.lat as ReferenceLattice)) {
      throw new StyleError(`MEAM reference lattice '${lib.lat}' is not supported (only fcc, bcc, dia, hcp, sc)`);
    }
    const nn2 = (par.pair.get('1,1')?.nn2 ?? 0) === 1;
    if (lib.t[0] !== 1) throw new StyleError('only MEAM parameters normalized to t0 = 1.0 are supported');
    if (lib.rozero !== 1) throw new StyleError(`MEAM rozero = ${lib.rozero} is not supported (only 1)`);
    if (!SUPPORTED_IBAR.includes(lib.ibar) && !(nn2 && lib.ibar === -5)) {
      throw new StyleError(`MEAM ibar = ${lib.ibar} is not supported (only ibar = 0, 1, 3, and -5 with nn2 = 1; ${paramFile})`);
    }
    if (lib.ibar !== 0 && !nn2 && lib.lat !== 'fcc' && lib.lat !== 'bcc') {
      throw new StyleError(`MEAM ibar = ${lib.ibar} with the ${lib.lat} reference of ${elem} is not supported (measured to mismatch at step 0 for dia; only fcc and bcc verified)`);
    }
    if (nn2 && lib.lat === 'hcp') {
      throw new StyleError(`MEAM nn2 = 1 with the hcp reference of ${elem} is not supported (no c/a parameter for the second-neighbour shell)`);
    }
    if (par.lattce !== undefined && par.lattce !== lib.lat) {
      throw new StyleError(`MEAM lattce(1,1) = ${par.lattce} differs from the library lattice '${lib.lat}' of ${elem}; not supported`);
    }
    // Ec and re default as measured below (see the defaults note above the coeff method)
    // The masses of the types mapped to this element come from the library (see the mass note above the coeff method).
    if (ctx.s && !ctx.s.rmass) {
      for (let t = 1; t <= this.ntypes; t++) if (maps[t - 1] === elem) ctx.s.massByType[t] = lib.atwt;
    }
    const Ec = par.Ec ?? lib.esub;
    let re = par.re;
    if (re === undefined) {
      // nearest-neighbour distance of the reference structure from the conventional lattice constant
      const lat = lib.lat as ReferenceLattice;
      if (lat === 'fcc') re = lib.alat / Math.SQRT2;
      else if (lat === 'bcc') re = (lib.alat * Math.sqrt(3)) / 2;
      else if (lat === 'dia') re = (lib.alat * Math.sqrt(3)) / 4;
      else if (lat === 'sc' || lat === 'hcp') re = lib.alat;
      else throw new StyleError(`MEAM parameter file ${paramFile} must set re(1,1) for the ${lib.lat} reference (${elem})`);
    }
    this.el = {
      z: lib.z,
      re,
      alpha: par.alpha ?? lib.alpha,
      Ec,
      A: lib.asub,
      beta: lib.b,
      t: lib.t,
      ibar: lib.ibar,
      lat: lib.lat as ReferenceLattice,
      erose: par.erose,
      nn2: nn2 ? nn2RefData(lib.lat as ReferenceLattice, par.opts) : undefined,
    };
    this.opts = par.opts;
    this.zblGuard = new Float64Array((this.ntypes + 1) * (this.ntypes + 1)).fill(par.zblOff.has('1,1') ? 0 : ZBL_GUARD * re);
    this.alloy = null;
    this.typeElem = [];
  }

  /** Multi-element pair_coeff: fcc/dia elements, b1/dia pairs, default screening, erose_form 0/1/2 (see meam_alloy.ts). */
  private coeffAlloy(args: string[], elems: string[], maps: string[], ctx: StyleContext): void {
    const libFile = args[2], paramFile = args[3 + elems.length];
    const libs = elems.map((elt) => {
      try {
        return parseMeamLibrary(ctx.readFile(libFile), elt, libFile);
      } catch (e) {
        throw new StyleError(`multi-element MEAM: ${(e as Error).message}`);
      }
    });
    const par = parseMeamParams(ctx.readFile(paramFile), paramFile, elems.length);
    for (const [, p] of par.pair) if (p.nn2 === 1) throw new StyleError(`multi-element MEAM: nn2 = 1 is not supported (${paramFile})`);
    const n = elems.length;
    // masses of the mapped types from the library atomic weights (same rule as the single-element style)
    if (ctx.s && !ctx.s.rmass) {
      for (let t = 1; t <= this.ntypes; t++) {
        const c = elems.indexOf(maps[t - 1]);
        if (c >= 0) ctx.s.massByType[t] = libs[c].atwt;
      }
    }
    const elements: AlloyElement[] = libs.map((lib, c) => {
      const elt = elems[c];
      if (lib.lat !== 'fcc' && lib.lat !== 'dia') {
        throw new StyleError(`multi-element MEAM: reference lattice '${lib.lat}' of ${elt} is not supported (only fcc and dia)`);
      }
      if (lib.t[0] !== 1) throw new StyleError('only MEAM parameters normalized to t0 = 1.0 are supported');
      if (lib.rozero !== 1) throw new StyleError(`multi-element MEAM: rozero = ${lib.rozero} is not supported (only 1)`);
      if (lib.ibar !== 0) throw new StyleError(`multi-element MEAM: ibar = ${lib.ibar} is not supported (only ibar = 0; the alloy density has no verified G-function selection)`);
      const own = par.pair.get(`${c + 1},${c + 1}`) ?? {};
      if (own.lattce !== undefined && own.lattce !== lib.lat) throw new StyleError(`multi-element MEAM: lattce(${c + 1},${c + 1}) = ${own.lattce} for ${elt} differs from the library lattice '${lib.lat}'`);
      // Ec and re of the element default as in the single-element style (measured: the same energies as the explicit values)
      return {
        z: lib.z,
        lat: lib.lat as ReferenceLattice,
        re: own.re ?? (lib.lat === 'dia' ? (lib.alat * Math.sqrt(3)) / 4 : lib.alat / Math.SQRT2),
        alpha: own.alpha ?? lib.alpha,
        Ec: own.Ec ?? lib.esub,
        A: lib.asub,
        beta: lib.b,
        t: lib.t,
        ibar: lib.ibar,
      };
    });
    const pairs: AlloyPair[][] = [];
    for (let i = 0; i < n; i++) {
      pairs.push([]);
      for (let j = 0; j < n; j++) {
        if (i === j) {
          const own = par.pair.get(`${i + 1},${i + 1}`) ?? {};
          pairs[i].push({ Ec: elements[i].Ec, re: elements[i].re, alpha: elements[i].alpha, lat: 'self', attrac: own.attrac, repuls: own.repuls });
          continue;
        }
        const a = Math.min(i, j), b = Math.max(i, j);
        const p = par.pair.get(`${a + 1},${b + 1}`) ?? {};
        if (p.Ec === undefined || p.re === undefined || p.alpha === undefined) {
          throw new StyleError(`multi-element MEAM: parameter file ${paramFile} must set Ec(${a + 1},${b + 1}), re(${a + 1},${b + 1}) and alpha(${a + 1},${b + 1})`);
        }
        if (p.lattce !== 'b1' && p.lattce !== 'dia') {
          throw new StyleError(`multi-element MEAM: lattce(${a + 1},${b + 1}) must be set to b1 or dia (${p.lattce ?? 'not set'} is not supported)`);
        }
        pairs[i].push({ Ec: p.Ec, re: p.re, alpha: p.alpha, lat: p.lattce, attrac: p.attrac, repuls: p.repuls });
      }
    }
    // Per-triplet screening limits: key "min(c,j),max(c,j),k" (1-based), the pair sorted as measured
    // with native LAMMPS (black box); undefined entries keep the uniform Cmin/Cmax defaults.
    const CminOf = (c: number, j: number, k: number): number => {
      const a = Math.min(c, j) + 1, b = Math.max(c, j) + 1;
      return par.cmin3.get(`${a},${b},${k + 1}`) ?? par.opts.Cmin;
    };
    const CmaxOf = (c: number, j: number, k: number): number => {
      const a = Math.min(c, j) + 1, b = Math.max(c, j) + 1;
      return par.cmax3.get(`${a},${b},${k + 1}`) ?? par.opts.Cmax;
    };
    const alloyOpts: AlloyOptions = { rc: par.opts.rc, delr: par.opts.delr, Cmin: par.opts.Cmin, Cmax: par.opts.Cmax, eroseForm: par.erose.form, ialloy: par.ialloy, CminOf, CmaxOf };
    this.alloy = makeAlloyModel(elements, pairs, alloyOpts, par.opts.augt1);
    this.typeElem = [-1, ...maps.map((m) => elems.indexOf(m))];
    this.el = null;
    this.opts = par.opts;
    const nt0 = this.ntypes + 1;
    this.zblGuard = new Float64Array(nt0 * nt0);
    for (let a = 1; a <= this.ntypes; a++) {
      for (let b = 1; b <= this.ntypes; b++) {
        const ea = this.typeElem[a], eb = this.typeElem[b];
        const key = `${Math.min(ea, eb) + 1},${Math.max(ea, eb) + 1}`;
        this.zblGuard[a * nt0 + b] = par.zblOff.has(key) ? 0 : ZBL_GUARD * pairs[ea][eb].re;
      }
    }
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.el && !this.alloy) throw new StyleError('pair_coeff for style meam must be set before the run');
    if (this.shift || this.tail) throw new StyleError('pair_style meam does not support the pair_modify shift and tail options');
    if (this.table !== 12) throw new StyleError('pair_style meam does not support the pair_modify table option');
    if (this.alloy) return;
    this.rhoRef = referenceBackground(this.el!, this.opts);
    const t1 = this.opts.augt1 ? this.el!.t[1] + 0.6 * this.el!.t[3] : this.el!.t[1];
    this.t = [1, t1, this.el!.t[2], this.el!.t[3]];
  }

  override initOne(_i: number, _j: number): number {
    if (!this.el && !this.alloy) throw new StyleError('pair_coeff for style meam must be set before the run');
    return this.opts.rc;
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style meam needs a full neighbor list');
    if (!this.el && !this.alloy) throw new StyleError('pair_coeff for style meam must be set before the run');
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const eatom = pc.eatom;
    let evdwl = 0;
    for (let i = 0; i < pc.nlocal; i++) {
      const ti = type[i] * nt;
      const nb: MeamNeighbor[] = [];
      const anb: AlloyNeighbor[] = [];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti + type[j]]) continue;
        const r = Math.sqrt(rsq);
        const zg = this.zblGuard[ti + type[j]];
        if (r < zg) {
          throw new StyleError(`MEAM pair at ${r} A is inside the ZBL blend region (zbl = 1 is the default; blending is not implemented): the blend is not applied below 0.9 re = ${zg} A; set zbl(I,J) = 0 in the parameter file, or keep the atoms farther apart`);
        }
        if (this.alloy) anb.push({ e: this.typeElem[type[j]], j, dx, dy, dz, r });
        else nb.push({ j, dx, dy, dz, r });
      }
      const count = this.alloy ? anb.length : nb.length;
      if (count === 0) continue;
      const g = new Float64Array(3 * count);
      let e: number;
      if (this.alloy) e = alloyAtomEnergyGrad(this.alloy, this.typeElem[type[i]], anb, g);
      else e = meamAtomEnergyGrad(this.el!, this.opts, this.rhoRef, this.t, nb, g);
      const list2 = this.alloy ? anb : nb;
      evdwl += e;
      if (eatom) eatom[i] += e;
      for (let m = 0; m < list2.length; m++) {
        const j = list2[m].j;
        for (let c = 0; c < 3; c++) {
          f[3 * j + c] -= g[3 * m + c];
          f[3 * i + c] += g[3 * m + c];
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
