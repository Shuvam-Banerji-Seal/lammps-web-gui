import { Pair, PairParams, StyleError, typeBounds, type PairCompute, type PairSep, type StyleContext } from '../types';
import type { SimState } from '../../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style resquared — docs.lammps.org/pair_resquared.html (live copy: plans/lammps-docs-live/pair_resquared.rst).
 *   pair_style resquared cutoff
 *   pair_coeff I J A12 sigma epsilon_ia epsilon_ib epsilon_ic epsilon_ja epsilon_jb epsilon_jc [cutoff]
 * The page: "Style *resquared* computes the RE-squared anisotropic interaction" between ellipsoidal and/or spherical
 * Lennard-Jones particles. The ellipsoid formulas are those of Everaers and Ejtehadi (Phys Rev E 67, 041710, 2003;
 * arXiv cond-mat/0306096, eqs. 41, 42, 11-12, 66-67) and Babadi, Ejtehadi and Everaers (J Comp Phys 219, 770, 2006;
 * arXiv cond-mat/0605193, eqs. 2a, 2b, 4-9). With S_i the half radii (shape parameters are diameters, as in
 * atom_style ellipsoid), R_i the body-to-space rotation, M_i = R S_i^2 R^T, K_i = R S_i^-2 R^T, N_i = R diag(e_i) R^T:
 *   G = M1 + M2,  P = r^T G^-1 r / 2,  sigma12 = |r| / sqrt(P),  h = |r| - sigma12   (Gay-Berne distance)
 *   sigma_i = (rhat^T K_i rhat)^-1/2,  H = M1 / sigma1 + M2 / sigma2
 *   eta = [ detS1 / sigma1^2 + detS2 / sigma2^2 ] / [ det H / (sigma1 + sigma2) ]^1/2
 *   B = N1 + N2,  chi = 2 rhat^T B^-1 rhat
 *   F = -(A12/36) (1 + 3 eta chi sigma/h) prod_{i,e} S_ie/(S_ie + h/2)
 *     + (A12/2025) (sigma/h)^6 (1 + (45/56) eta chi sigma/h) prod_{i,e} S_ie/(S_ie + h/60^(1/3))
 * Measured with native LAMMPS (black box, total energy with thermo_modify norm no; lj units normalize pe per atom by
 * default): the pair energy is F, with sigma_c = sigma (the pair_coeff sigma), S = shape/2, and chi from B = sum of
 * R diag(e) R^T with the epsilon values themselves (a larger epsilon gives a smaller chi). The forces and torques are
 * the derivatives of F.
 * Probed with identity and rotated quaternions, several shapes, sigma and epsilon values; see tests/oracle/w43rsq_*.
 * Quaternions are normalized before they become rotation matrices (as in pair_gayberne).
 *
 * LJ spheres: "If all 3 shape parameters = 0.0, then the particle is treated as an LJ sphere." LJ-sphere pairs use the
 * plain Lennard-Jones formula in r with A12 as epsilon: "For LJ sphere/LJ sphere interactions, the interaction is
 * computed using the standard Lennard-Jones formula". Measured with native LAMMPS (black box): two point particles
 * follow that formula with epsilon = A12.
 *
 * NOT reproduced: an ellipsoid paired with an LJ sphere. The page says the interaction "is also computed by the
 * formulas in the supplementary document referenced above", which is not available here; the expression above with a
 * sphere of radius sigma/2 does not reproduce the native energies (measured, black box). Such pairs are refused with a
 * StyleError rather than approximated.
 *
 * Torques: tau = -dF/dphi with phi a space-frame rotation of each body. A space rotation changes M, N and K by the
 * commutator [W, X] with W = [omega]x, so dF/domega is the axial vector of X F_X - F_X X summed over the matrix
 * derivatives F_X = dF/dX (X = M, N, K). Derivatives are written out at the code. tests/enginePairResquared43.test.ts
 * checks forces and torques against central finite differences of the engine's own energy.
 *
 * Refusals (StyleError): an atom style without ellipsoids; pair_modify tail (the page says this option is not
 * supported); a special neighbour with a factor other than 1 (no special-bond factor is described for this style);
 * a type without epsilon a,b,c (measured: native init fails for LJ-sphere types too); an unset I J pair (mixing is not
 * reproduced); an ellipsoid-LJ sphere pair; a singular B (measured: native Bad matrix inversion in mldivide3 when an
 * epsilon component is zero along an aligned pair).
 */

/** Row-major 3x3 helpers (9 entries). */
const det3 = (m: ArrayLike<number>): number =>
  m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);

/** Inverse of a 3x3 matrix into out; returns the determinant. */
const inv3 = (m: ArrayLike<number>, out: Float64Array): number => {
  const d = det3(m);
  const id = 1 / d;
  out[0] = (m[4] * m[8] - m[5] * m[7]) * id;
  out[1] = (m[2] * m[7] - m[1] * m[8]) * id;
  out[2] = (m[1] * m[5] - m[2] * m[4]) * id;
  out[3] = (m[5] * m[6] - m[3] * m[8]) * id;
  out[4] = (m[0] * m[8] - m[2] * m[6]) * id;
  out[5] = (m[2] * m[3] - m[0] * m[5]) * id;
  out[6] = (m[3] * m[7] - m[4] * m[6]) * id;
  out[7] = (m[1] * m[6] - m[0] * m[7]) * id;
  out[8] = (m[0] * m[4] - m[1] * m[3]) * id;
  return d;
};

/** m v into o (3 entries). */
const mv = (m: ArrayLike<number>, v: ArrayLike<number>, o: number[]): number[] => {
  o[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  o[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  o[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
  return o;
};

/** Rotation matrix (body to space) of a quaternion (w, i, j, k), normalized first. */
const quatToMat = (w: number, x: number, y: number, z: number, out: Float64Array): void => {
  const n = 1 / Math.sqrt(w * w + x * x + y * y + z * z);
  w *= n; x *= n; y *= n; z *= n;
  out[0] = 1 - 2 * (y * y + z * z); out[1] = 2 * (x * y - w * z); out[2] = 2 * (x * z + w * y);
  out[3] = 2 * (x * y + w * z); out[4] = 1 - 2 * (x * x + z * z); out[5] = 2 * (y * z - w * x);
  out[6] = 2 * (x * z - w * y); out[7] = 2 * (y * z + w * x); out[8] = 1 - 2 * (x * x + y * y);
};

/** R diag(d) R^T. */
const rdrt = (R: ArrayLike<number>, d: ArrayLike<number>, out: Float64Array): void => {
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += R[3 * a + k] * d[k] * R[3 * b + k];
      out[3 * a + b] = s;
    }
  }
};

/** Matrix product a b into out (row-major). */
const mm = (a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array): void => {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) out[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
  }
};

/** Axial vector of the antisymmetric part of C: (C12 - C21, C20 - C02, C01 - C10), 0-based row-major indices. */
const addAxial = (C: ArrayLike<number>, o: number[]): void => {
  o[0] += C[5] - C[7];
  o[1] += C[6] - C[2];
  o[2] += C[1] - C[3];
};

/** Per owned atom (flat, 9 entries per atom for M, K and N): M = R S^2 R^T, K = R S^-2 R^T, N = R diag(e) R^T, the half radii, detS and the ellipsoid flag. */
interface RsqSites { M: Float64Array; K: Float64Array; N: Float64Array; Sh: Float64Array; detS: Float64Array; ell: Uint8Array }

const newRsqSites = (n: number): RsqSites => ({
  M: new Float64Array(9 * n), K: new Float64Array(9 * n), N: new Float64Array(9 * n),
  Sh: new Float64Array(3 * n), detS: new Float64Array(n), ell: new Uint8Array(n),
});

/** Fills the site data of site o from atom a (shape, quaternion, epsilon a,b,c of type t); an LJ sphere (shape 0) leaves ell 0. */
const rsqSite = (sites: RsqSites, o: number, a: number, shape: Float64Array, quat: Float64Array, t: number, eT: Float64Array): void => {
  if (!(shape[3 * a] > 0)) return;
  sites.ell[o] = 1;
  const R = new Float64Array(9), D2 = new Float64Array(3), Dm2 = new Float64Array(3), De = new Float64Array(3);
  const tM = new Float64Array(9), tK = new Float64Array(9), tN = new Float64Array(9);
  quatToMat(quat[4 * a], quat[4 * a + 1], quat[4 * a + 2], quat[4 * a + 3], R);
  for (let d = 0; d < 3; d++) {
    const sd = shape[3 * a + d];
    sites.Sh[3 * o + d] = sd;
    D2[d] = sd * sd;
    Dm2[d] = 1 / (sd * sd);
    De[d] = eT[3 * t + d];
  }
  sites.detS[o] = sites.Sh[3 * o] * sites.Sh[3 * o + 1] * sites.Sh[3 * o + 2];
  rdrt(R, D2, tM);
  rdrt(R, Dm2, tK);
  rdrt(R, De, tN);
  sites.M.set(tM, 9 * o);
  sites.K.set(tK, 9 * o);
  sites.N.set(tN, 9 * o);
};

/** Scratch arrays of one ellipsoid pair (reused between pairs). */
interface RsqScratch {
  G: Float64Array; Gi: Float64Array; B: Float64Array; Bi: Float64Array; H: Float64Array; Hi: Float64Array;
  Gmat: Float64Array; t1: Float64Array; t2: Float64Array; C: Float64Array;
  FM: Float64Array[]; FN: Float64Array[]; FK: Float64Array[]; Mb: Float64Array[]; Kb: Float64Array[]; Nb: Float64Array[];
  v: number[]; w: number[]; tmpv: number[]; rv: number[]; rh: number[]; gF: number[];
  sg: number[]; u: number[]; lam: number[]; dEtaDsg: number[]; acc: number[];
}

const newRsqScratch = (): RsqScratch => ({
  G: new Float64Array(9), Gi: new Float64Array(9), B: new Float64Array(9), Bi: new Float64Array(9),
  H: new Float64Array(9), Hi: new Float64Array(9), Gmat: new Float64Array(9), t1: new Float64Array(9),
  t2: new Float64Array(9), C: new Float64Array(9),
  FM: [new Float64Array(9), new Float64Array(9)], FN: [new Float64Array(9), new Float64Array(9)], FK: [new Float64Array(9), new Float64Array(9)],
  Mb: [new Float64Array(9), new Float64Array(9)], Kb: [new Float64Array(9), new Float64Array(9)], Nb: [new Float64Array(9), new Float64Array(9)],
  v: [0, 0, 0], w: [0, 0, 0], tmpv: [0, 0, 0], rv: [0, 0, 0], rh: [0, 0, 0], gF: [0, 0, 0],
  sg: [0, 0], u: [0, 0], lam: [0, 0], dEtaDsg: [0, 0], acc: [0, 0, 0],
});

/** The force and torques of one ellipsoid pair: fi = force on i (dF/dr), ti and tj = torques on i and j. */
interface RsqPairOut { fi: Float64Array; ti: Float64Array; tj: Float64Array }

const newRsqPairOut = (): RsqPairOut => ({ fi: new Float64Array(3), ti: new Float64Array(3), tj: new Float64Array(3) });

/**
 * The RE-squared energy and forces of one ellipsoid pair of owned atoms a (= i) and b (= j); dxi = x_i - x_j. Returns the
 * energy F and fills out. An ellipsoid with an LJ sphere is refused (see the header); the LJ-sphere pairs are the callers'.
 */
const rsqPair = (sites: RsqSites, sc: RsqScratch, a: number, b: number, dxi: number, dyi: number, dzi: number, rsq: number, eps: number, sig: number, out: RsqPairOut): number => {
  if (sites.ell[a] !== sites.ell[b]) throw new StyleError('pair_style resquared: an ellipsoid with an LJ sphere is not supported');
  const { G, Gi, B, Bi, H, Hi, Gmat, t1, t2, C, FM, FN, FK, Mb, Kb, Nb, v, w, tmpv, rv, rh, gF, sg, u, lam, dEtaDsg, acc } = sc;
  const { Sh } = sites;

  // Ellipsoid pair. rv = x_j - x_i, rh = rv / |rv|.
  const rx = -dxi, ry = -dyi, rz = -dzi;
  rv[0] = rx; rv[1] = ry; rv[2] = rz;
  const rr = Math.sqrt(rsq);
  rh[0] = rx / rr; rh[1] = ry / rr; rh[2] = rz / rr;
  const Mi = sites.M.subarray(9 * a, 9 * a + 9), Mj = sites.M.subarray(9 * b, 9 * b + 9);
  const Ki = sites.K.subarray(9 * a, 9 * a + 9), Kj = sites.K.subarray(9 * b, 9 * b + 9);
  const Ni = sites.N.subarray(9 * a, 9 * a + 9), Nj = sites.N.subarray(9 * b, 9 * b + 9);
  Mb[0] = Mi; Mb[1] = Mj; Kb[0] = Ki; Kb[1] = Kj; Nb[0] = Ni; Nb[1] = Nj;
  for (let q = 0; q < 9; q++) { G[q] = Mi[q] + Mj[q]; B[q] = Ni[q] + Nj[q]; }
  inv3(G, Gi);
  const detB = inv3(B, Bi);
  if (!Number.isFinite(detB) || Math.abs(detB) <= 1e-14 * Math.abs(B[0] + B[4] + B[8]) ** 3) {
    throw new StyleError('Bad matrix inversion in mldivide3');
  }
  // Gay-Berne distance: P = r^T G^-1 r / 2, sigma12 = |r| / sqrt(P), h = |r| - sigma12.
  mv(Gi, rv, v);
  const P = 0.5 * (rv[0] * v[0] + rv[1] * v[1] + rv[2] * v[2]);
  const sP = Math.sqrt(P);
  const h = rr - rr / sP;
  // chi = 2 rhat^T B^-1 rhat = 2 r^T B^-1 r / r^2 ; w = B^-1 r.
  mv(Bi, rv, w);
  const chi = 2 * (rv[0] * w[0] + rv[1] * w[1] + rv[2] * w[2]) / rsq;
  // Projected radii sigma_b = (rhat^T K_b rhat)^-1/2 and H = M1/sigma1 + M2/sigma2.
  for (let b = 0; b < 2; b++) {
    const Kx = Kb[b];
    let uu = 0;
    for (let a = 0; a < 3; a++) for (let c = 0; c < 3; c++) uu += rh[a] * Kx[3 * a + c] * rh[c];
    u[b] = uu;
    sg[b] = 1 / Math.sqrt(uu);
  }
  for (let q = 0; q < 9; q++) H[q] = Mi[q] / sg[0] + Mj[q] / sg[1];
  const DH = inv3(H, Hi);
  const detS = [sites.detS[a], sites.detS[b]];
  const n = detS[0] * u[0] + detS[1] * u[1];
  const SS = sg[0] + sg[1];
  const d = Math.sqrt(DH / SS);
  const eta = n / d;
  const Q = eta * chi;
  const sigc = sig;
  const x0 = sigc / h;
  const x6 = x0 ** 6;
  const a56 = 45 / 56;
  // Prod over the six half radii: P_c = prod 1/(1 + h/(c S)), with d ln P_c / dh = -sum 1/(c S + h).
  const c60 = Math.cbrt(60);
  let P2 = 1, P60 = 1, sum2 = 0, sum60 = 0;
  for (let d3 = 0; d3 < 3; d3++) {
    for (const off of [3 * a, 3 * b]) {
      const Sd = Sh[off + d3];
      const a2 = 2 * Sd, a60 = c60 * Sd;
      P2 /= 1 + h / a2;
      P60 /= 1 + h / a60;
      sum2 += 1 / (a2 + h);
      sum60 += 1 / (a60 + h);
    }
  }
  // F = T1 + T2 with T1 = -(A/36)(1 + 3 Q x) P2 and T2 = (A/2025) x^6 (1 + a Q x) P60, x = sigma/h.
  const F = -(eps / 36) * (1 + 3 * Q * x0) * P2 + (eps / 2025) * x6 * (1 + a56 * Q * x0) * P60;
  const Fh = -(eps / 36) * (-(3 * Q * x0 / h) * P2 - (1 + 3 * Q * x0) * P2 * sum2)
    + (eps / 2025) * (-(6 * x6 / h) * (1 + a56 * Q * x0) * P60 - a56 * Q * x6 * (x0 / h) * P60 - x6 * (1 + a56 * Q * x0) * P60 * sum60);
  const FQ = -(eps / 36) * 3 * x0 * P2 + (eps / 2025) * x6 * a56 * x0 * P60;
  const Fchi = FQ * eta; // dF/dchi (Q = eta chi)
  const Feta = FQ * chi; // dF/deta

  // eta depends on sigma_b through H and sigma1 + sigma2: d eta / d sigma_b = eta/(2 SS) + eta/(2 sigma_b^2) tr(H^-1 M_b).
  // Through u_b (direct in n and via sigma_b): lam_b = detS_b/d + d eta/d sigma_b * (-sigma_b^3 / 2).
  for (let b = 0; b < 2; b++) {
    const Mx = Mb[b];
    let trHM = 0;
    for (let a = 0; a < 3; a++) for (let c = 0; c < 3; c++) trHM += Hi[3 * a + c] * Mx[3 * c + a];
    dEtaDsg[b] = eta / (2 * SS) + (eta / (2 * sg[b] * sg[b])) * trHM;
    lam[b] = detS[b] / d + dEtaDsg[b] * (-0.5 * sg[b] ** 3);
  }

  // Matrix derivatives. dh/dG = -(r/4) P^-3/2 v v^T ; dF/dH = Feta (-eta/2) H^-1 ; dchi/dB = -2 w' w'^T with
  // w' = B^-1 rhat ; dF/dK_b = Feta lam_b rhat rhat^T.
  const kappa = -(rr / 4) * Math.pow(P, -1.5);
  for (let a = 0; a < 3; a++) for (let c = 0; c < 3; c++) Gmat[3 * a + c] = Fh * kappa * v[a] * v[c];
  mv(Bi, rh, tmpv); // w' = B^-1 rhat
  for (let b = 0; b < 2; b++) {
    const sgb = sg[b];
    const Fm = FM[b], Fn = FN[b], Fk = FK[b];
    for (let q = 0; q < 9; q++) Fm[q] = Gmat[q] + (Feta * (-eta / 2)) * Hi[q] / sgb;
    for (let a = 0; a < 3; a++) for (let c = 0; c < 3; c++) {
      Fn[3 * a + c] = Fchi * (-2) * tmpv[a] * tmpv[c];
      Fk[3 * a + c] = Feta * lam[b] * rh[a] * rh[c];
    }
  }

  // dF/drv: through h, chi, and u_b = rhat^T K_b rhat (du_b/drv = 2 (K_b rhat - u_b rhat) / r).
  for (let a = 0; a < 3; a++) {
    const dhdr = rh[a] * (1 - 1 / sP) + 0.5 * rr * Math.pow(P, -1.5) * v[a];
    const dchidr = 4 * w[a] / rsq - 2 * chi * rv[a] / rsq;
    gF[a] = Fh * dhdr + Fchi * dchidr;
  }
  for (let b = 0; b < 2; b++) {
    mv(Kb[b], rh, tmpv);
    for (let a = 0; a < 3; a++) gF[a] += Feta * lam[b] * (2 / rr) * (tmpv[a] - u[b] * rh[a]);
  }
  // Force on i is +dF/drv, on j is -dF/drv. Measured with native LAMMPS (black box): the force is dF/dr.
  out.fi.set(gF);

  // Torques: tau_b = -dF/domega_b = -sum over X of axial(X F_X - F_X X); the torque on body b is -acc_b.
  for (let b = 0; b < 2; b++) {
    acc[0] = 0; acc[1] = 0; acc[2] = 0;
    const pairsX: [Float64Array, Float64Array][] = [[Mb[b], FM[b]], [Nb[b], FN[b]], [Kb[b], FK[b]]];
    for (const [X, Fx] of pairsX) {
      mm(X, Fx, t1);
      mm(Fx, X, t2);
      for (let q = 0; q < 9; q++) C[q] = t1[q] - t2[q];
      addAxial(C, acc);
    }
    const tb = b === 0 ? out.ti : out.tj;
    tb[0] = -acc[0]; tb[1] = -acc[1]; tb[2] = -acc[2];
  }
  return F;
};

export class PairResquared extends Pair {
  readonly name = 'resquared';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  /** Per type: epsilon a,b,c (index 3 t + d), and whether any were set. */
  eT = new Float64Array(0);
  eSet = new Uint8Array(0);
  offset = new Float64Array(0);
  /** The system of the last init (single() reads the shapes and quaternions of its two atoms). */
  private state: SimState | null = null;
  /** Native svector of the last single() call: force and torque on i (pN 1..6). */
  private readonly svec = new Float64Array(6);
  private readonly sc = newRsqScratch();
  private readonly pout = newRsqPairOut();

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style resquared cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut']);
    this.eT = new Float64Array(3 * (ntypes + 1));
    this.eSet = new Uint8Array(ntypes + 1);
  }

  coeff(args: string[]): void {
    if (args.length < 10 || args.length > 11) throw new StyleError('Incorrect args for pair coefficients');
    const A = parseNum(args[2], 'A12');
    const sig = parseNum(args[3], 'sigma');
    const ei = [4, 5, 6].map((k) => parseNum(args[k], 'epsilon_i'));
    const ej = [7, 8, 9].map((k) => parseNum(args[k], 'epsilon_j'));
    const cut = args.length === 11 ? parseNum(args[10], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [A, sig, cut]);
    // Epsilon a,b,c are per atom type: a non-zero triple of the I values goes to type I, then one of the J values to type J.
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const assign = (t: number, e: number[]) => {
      if (!e.some((v) => v !== 0)) return;
      for (let d = 0; d < 3; d++) this.eT[3 * t + d] = e[d];
      this.eSet[t] = 1;
    };
    for (let i = ilo; i <= ihi; i++) for (let j = jlo; j <= jhi; j++) {
      assign(i, ei);
      assign(j, ej);
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    // Measured with native LAMMPS (black box): init fails unless every type has epsilon a,b,c set (LJ spheres too).
    for (const t of [i, j]) {
      if (!this.eSet[t]) throw new StyleError('Pair resquared epsilon a,b,c coeffs are not all set');
    }
    // Mixing is not reproduced: an unset I J pair is refused (the page says sphere pairs can be mixed; the engine does
    // not know which atoms are spheres when it sets the pair table, so it does not mix).
    if (!p.isSet(i, j)) throw new StyleError(`pair resquared: pair ${i} ${j} is not set (mixing is not supported)`);
    const nt = this.ntypes + 1;
    if (this.offset.length !== nt * nt) this.offset = new Float64Array(nt * nt);
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    // pair_modify shift: "but only for sphere-sphere interactions", so the offset is used by LJ-sphere pairs only.
    if (this.shift && cut > 0) {
      const r = sig / cut;
      this.offset[k1] = this.offset[k2] = 4 * eps * (r ** 12 - r ** 6);
    } else this.offset[k1] = this.offset[k2] = 0;
    return cut;
  }

  init(ctx: StyleContext): void {
    // The page describes no tail correction for this style, so tail yes is refused rather than ignored.
    this.offset = new Float64Array(0);
    this.state = ctx.s;
    super.init(ctx);
  }

  /** Native single_extra (pair_resquared.cpp: single_extra = 6): the force and torque on i (pN 1..6). */
  singleExtra(): number { return 6; }

  singleNeedsSeparation(): boolean { return true; }

  /**
   * Native single() (pair_resquared.cpp): fforce is the force on i projected on the centre axis, the svector holds the full
   * force on i (1-3) and the torque on i (4-6), both times factor_lj, and the energy is factor_lj times the pair energy.
   * i and j are owned atoms; del = x_i - x_j.
   */
  single(i: number, j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, del?: PairSep): { eng: number; fforce: number; svector: Float64Array } {
    const s = this.state;
    if (!s || !s.shape || !s.quat || !s.torque) throw new StyleError('Pair resquared requires atom style ellipsoid');
    if (!del) throw new StyleError('pair resquared single() needs the separation vector x_i - x_j (compute pair/local passes it)');
    const nt = this.ntypes + 1;
    const eps = this.p.get('epsilon', itype, jtype), sig = this.p.get('sigma', itype, jtype);
    const sites = newRsqSites(2);
    rsqSite(sites, 0, i, s.shape, s.quat, itype, this.eT);
    rsqSite(sites, 1, j, s.shape, s.quat, jtype, this.eT);
    const svector = this.svec;
    svector.fill(0);
    if (!sites.ell[0] && !sites.ell[1]) {
      // LJ sphere pair: plain Lennard-Jones in r with A12 as epsilon, shifted by the pair_modify shift offset
      const r2inv = 1 / rsq;
      const sr2 = sig * sig * r2inv, sr6 = sr2 * sr2 * sr2, sr12 = sr6 * sr6;
      const eLJ = 4 * eps * (sr12 - sr6) - this.offset[itype * nt + jtype];
      const fljr = (48 * eps * sr12 - 24 * eps * sr6) * r2inv;
      svector[0] = factorLJ * fljr * del.dx; svector[1] = factorLJ * fljr * del.dy; svector[2] = factorLJ * fljr * del.dz;
      return { eng: factorLJ * eLJ, fforce: factorLJ * fljr, svector };
    }
    const out = this.pout;
    const F = rsqPair(sites, this.sc, 0, 1, del.dx, del.dy, del.dz, rsq, eps, sig, out);
    for (let c = 0; c < 3; c++) { svector[c] = factorLJ * out.fi[c]; svector[3 + c] = factorLJ * out.ti[c]; }
    const radial = (out.fi[0] * del.dx + out.fi[1] * del.dy + out.fi[2] * del.dz) / rsq;
    return { eng: factorLJ * F, fforce: factorLJ * radial, svector };
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.shape || !s.quat || !s.torque) throw new StyleError('Pair resquared requires atom style ellipsoid');
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const tq = new Float64Array(3 * pc.nall);
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const cutsq = this.cutsq;
    const shape = s.shape, quat = s.quat;

    // Per owned atom: half radii S, detS, the ellipsoid flag, and the space matrices M = R S^2 R^T, K = R S^-2 R^T,
    // N = R diag(e) R^T (e: the atom type's epsilon a,b,c).
    const sites = newRsqSites(nlocal);
    for (let o = 0; o < nlocal; o++) rsqSite(sites, o, o, shape, quat, type[o], this.eT);

    const sc = this.sc, out = this.pout;
    let evdwl = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const dxi = xi - x[3 * j], dyi = yi - x[3 * j + 1], dzi = zi - x[3 * j + 2];
        const rsq = dxi * dxi + dyi * dyi + dzi * dzi;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const sLj = sL[jj >>> SBBITS], sCj = sC[jj >>> SBBITS];
        if (jj >>> SBBITS !== 0 && (sLj !== 1 || sCj !== 1)) {
          throw new StyleError('pair_style resquared does not support special_bonds factors other than 1');
        }
        const ti_ = type[i], tj_ = type[j];
        const eps = this.p.get('epsilon', ti_, tj_), sig = this.p.get('sigma', ti_, tj_);
        if (!sites.ell[oi] && !sites.ell[oj]) {
          // LJ sphere pair: plain Lennard-Jones in r with A12 as epsilon, shifted by the pair_modify shift offset.
          const r2inv = 1 / rsq;
          const sr2 = sig * sig * r2inv, sr6 = sr2 * sr2 * sr2, sr12 = sr6 * sr6;
          const eLJ = 4 * eps * (sr12 - sr6) - this.offset[t];
          evdwl += eLJ;
          const fljr = (48 * eps * sr12 - 24 * eps * sr6) * r2inv;
          f[3 * i] += fljr * dxi; f[3 * i + 1] += fljr * dyi; f[3 * i + 2] += fljr * dzi;
          f[3 * j] -= fljr * dxi; f[3 * j + 1] -= fljr * dyi; f[3 * j + 2] -= fljr * dzi;
          if (pc.eatom) { pc.eatom[i] += 0.5 * eLJ; pc.eatom[j] += 0.5 * eLJ; }
          if (pc.vatom) {
            const va = pc.vatom;
            const ww = [dxi * fljr * dxi, dyi * fljr * dyi, dzi * fljr * dzi, dxi * fljr * dyi, dxi * fljr * dzi, dyi * fljr * dzi];
            for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * ww[c]; va[6 * j + c] += 0.5 * ww[c]; }
          }
          continue;
        }
        const F = rsqPair(sites, sc, oi, oj, dxi, dyi, dzi, rsq, eps, sig, out);
        evdwl += F;
        const g = out.fi;
        f[3 * i] += g[0]; f[3 * i + 1] += g[1]; f[3 * i + 2] += g[2];
        f[3 * j] -= g[0]; f[3 * j + 1] -= g[1]; f[3 * j + 2] -= g[2];
        for (let c = 0; c < 3; c++) { tq[3 * i + c] += out.ti[c]; tq[3 * j + c] += out.tj[c]; }
        if (pc.eatom) { pc.eatom[i] += 0.5 * F; pc.eatom[j] += 0.5 * F; }
        if (pc.vatom) {
          const va = pc.vatom;
          const fx = g[0], fy = g[1], fz = g[2];
          const wv = [dxi * fx, dyi * fy, dzi * fz, dxi * fy, dxi * fz, dyi * fz];
          for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * wv[c]; va[6 * j + c] += 0.5 * wv[c]; }
        }
      }
    }
    nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
