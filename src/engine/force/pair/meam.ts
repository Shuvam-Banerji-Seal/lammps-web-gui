import { NEIGHMASK } from '../../neighbor';
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
 *  - radial weight: [1-(1-x)^4]^2 with x = (rc - r)/delr. Dimer fits give weights within ~1e-3 of
 *    this form, residual 1e-7 eV; cases with neighbours in [rc-delr, rc] are NOT covered by tests.
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
}

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

export const gOf = (ibar: number, gamma: number): number => {
  if (ibar === 0 || ibar === 4) return Math.sqrt(1 + gamma);
  if (ibar === 1) return Math.exp(gamma / 2);
  if (ibar === 3) return 2 / (1 + Math.exp(-gamma));
  if (ibar === -5) return (gamma >= 0 ? 1 : -1) * Math.sqrt(Math.abs(1 + gamma));
  throw new Error(`MEAM ibar ${ibar} is not supported`);
};

/** Background density of the fcc reference at nearest-neighbour distance r (G = 1 for cubic symmetry). */
const refBackground = (el: MeamElement, o: MeamOptions, r: number): number => {
  const a = r * Math.SQRT2;
  const nb: MeamNeighbor[] = [];
  const m = Math.ceil(o.rc / a) + 1;
  for (let i = -m; i <= m; i++)
    for (let j = -m; j <= m; j++)
      for (let k = -m; k <= m; k++) {
        if (((i + j + k) & 1) !== 0 || (i === 0 && j === 0 && k === 0)) continue;
        const dx = (i * a) / 2, dy = (j * a) / 2, dz = (k * a) / 2;
        const rr = Math.hypot(dx, dy, dz);
        if (rr >= o.rc) continue;
        nb.push({ j: -1, dx, dy, dz, r: rr });
      }
  const S = screening(nb, o);
  let rho0 = 0;
  for (let n = 0; n < nb.length; n++) rho0 += radialWeight(nb[n].r, o) * S[n] * Math.exp(-el.beta[0] * (nb[n].r / el.re - 1));
  return rho0;
};

/** Reference background at equilibrium; F is normalised by it. */
export const referenceBackground = (el: MeamElement, o: MeamOptions): number => refBackground(el, o, el.re);

/** F(rhobar) = A Ec (rhobar/rhoRef) ln(rhobar/rhoRef). */
export const embedding = (el: MeamElement, rhoRef: number, rhoBar: number): number => {
  if (rhoBar <= 0) return 0;
  const x = rhoBar / rhoRef;
  return el.A * el.Ec * x * Math.log(x);
};

/** Pair term phi(r) from the fcc reference structure (Rose universal form, erose_form = 0, a3 = 0). */
export const pairPhi = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number => {
  const rs = r / el.re;
  const astar = el.alpha * (rs - 1);
  const Eu = -el.Ec * (1 + astar) * Math.exp(-astar);
  return (2 / el.z) * (Eu - embedding(el, rhoRef, refBackground(el, o, r)));
};

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
    for (let m = 0; m < nb.length; m++) E += 0.5 * S[m] * radialWeight(nb[m].r, o) * pairPhi(el, o, rhoRef, nb[m].r);
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

/** dG/dGamma for the same ibar forms as gOf. */
export const gPrimeOf = (ibar: number, gamma: number): number => {
  if (ibar === 0 || ibar === 4) return 1 / (2 * Math.sqrt(1 + gamma));
  if (ibar === 1) return Math.exp(gamma / 2) / 2;
  if (ibar === 3) {
    const e = Math.exp(-gamma);
    return (2 * e) / ((1 + e) * (1 + e));
  }
  if (ibar === -5) {
    const s = gamma >= 0 ? 1 : -1;
    const s2 = 1 + gamma >= 0 ? 1 : -1;
    return (s * s2) / (2 * Math.sqrt(Math.abs(1 + gamma)));
  }
  throw new Error(`MEAM ibar ${ibar} is not supported`);
};

/** rho_ref(r) of the fcc reference and its r-derivative (same neighbour set as refBackground). */
export const refBackgroundPrime = (el: MeamElement, o: MeamOptions, r: number): { rho: number; drho: number } => {
  const a = r * Math.SQRT2;
  const nb: MeamNeighbor[] = [];
  const m = Math.ceil(o.rc / a) + 1;
  for (let i = -m; i <= m; i++)
    for (let j = -m; j <= m; j++)
      for (let k = -m; k <= m; k++) {
        if (((i + j + k) & 1) !== 0 || (i === 0 && j === 0 && k === 0)) continue;
        const dx = (i * a) / 2, dy = (j * a) / 2, dz = (k * a) / 2;
        const rr = Math.hypot(dx, dy, dz);
        if (rr >= o.rc) continue;
        nb.push({ j: -1, dx, dy, dz, r: rr });
      }
  const S = screening(nb, o);
  let rho = 0, drho = 0;
  for (let n = 0; n < nb.length; n++) {
    const rr = nb[n].r, s = rr / r;
    const e = Math.exp(-el.beta[0] * (rr / el.re - 1));
    const f = radialWeight(rr, o), fp = radialWeightPrime(rr, o);
    rho += f * S[n] * e;
    drho += S[n] * s * (fp * e + f * ((-el.beta[0] / el.re) * e));
  }
  return { rho, drho };
};

/** d phi / dr for the fcc-reference pair term (same form as pairPhi). */
export const pairPhiPrime = (el: MeamElement, o: MeamOptions, rhoRef: number, r: number): number => {
  const as = el.alpha * (r / el.re - 1);
  const dEu = (el.Ec * as * Math.exp(-as) * el.alpha) / el.re;
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
    phi[m] = pairPhi(el, o, rhoRef, r[m]);
    phip[m] = pairPhiPrime(el, o, rhoRef, r[m]);
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
 * pair_style meam (task 2). Single element only. Documented file formats (docs.lammps.org/pair_meam.html,
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
 * The parameter-file keywords Ec and re have no documented default for a single element, so the parameter file
 * must set Ec(1,1) and re(1,1). The NULL parameter file is therefore rejected.
 * ibar: the docs list "0 => G = sqrt(1+Gamma)" and "1 => G = exp(Gamma/2)"; only ibar = 0 is measured here.
 */

/** Library entry fields used by this style (the 19-field entry of the docs). */
interface LibraryEntry {
  lat: string;
  z: number;
  alpha: number;
  b: [number, number, number, number];
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
      asub: num(12),
      t: [num(13), num(14), num(15), num(16)],
      rozero: num(17),
      ibar: num(18),
    };
  }
  throw new StyleError(`element '${elt}' is not in MEAM library file ${name}`);
};

/** Settings from the parameter file, after the checks of the supported subset. */
interface MeamParams {
  Ec?: number;
  re?: number;
  alpha?: number;
  opts: MeamOptions;
}

const NUMERIC_DEFAULT_ZERO = ['nn2', 'attrac', 'repuls', 'erose_form', 'emb_lin_neg', 'bkgd_dyn', 'ialloy', 'mixture_ref_t'];

/** Parses "keyword = value" and "keyword(I,J) = value" lines; rejects everything outside the verified subset. */
export const parseMeamParams = (text: string, name: string): MeamParams => {
  const out: MeamParams = { opts: { ...DEFAULT_MEAM_OPTIONS } };
  for (const line of text.split(/\r?\n/)) {
    const h = line.indexOf('#');
    const body = (h >= 0 ? line.slice(0, h) : line).trim();
    if (!body) continue;
    const m = /^([A-Za-z_][A-Za-z_0-9]*)\s*(?:\(([^)]*)\))?\s*=\s*(\S+)$/.exec(body);
    if (!m) throw new StyleError(`cannot parse MEAM parameter line '${body}' in ${name}`);
    const key = m[1];
    const idx = m[2] ? m[2].split(',').map((s) => Number(s.trim())) : [];
    const val = m[3];
    const want = (n: number[], label: string) => {
      if (idx.length !== n.length || idx.some((v, k) => v !== n[k])) {
        throw new StyleError(`MEAM parameter ${key} in ${name} must be indexed ${label} for a single element`);
      }
    };
    const num = (): number => {
      const x = Number(val);
      if (!Number.isFinite(x)) throw new StyleError(`MEAM parameter ${key} in ${name}: '${val}' is not a number`);
      return x;
    };
    switch (key) {
      case 'rc': want([], ''); out.opts.rc = num(); break;
      case 'delr': want([], ''); out.opts.delr = num(); break;
      case 'Ec': want([1, 1], '(1,1)'); out.Ec = num(); break;
      case 're': want([1, 1], '(1,1)'); out.re = num(); break;
      case 'alpha': want([1, 1], '(1,1)'); out.alpha = num(); break;
      case 'Cmin': want([1, 1, 1], '(1,1,1)'); out.opts.Cmin = num(); break;
      case 'Cmax': want([1, 1, 1], '(1,1,1)'); out.opts.Cmax = num(); break;
      case 'augt1': {
        want([], '');
        const x = num();
        if (x !== 0 && x !== 1) throw new StyleError(`augt1 must be 0 or 1 (${name})`);
        out.opts.augt1 = x === 1;
        break;
      }
      case 'lattce':
        want([1, 1], '(1,1)');
        if (val !== 'fcc') throw new StyleError(`MEAM lattce '${val}' is not supported (only fcc, ${name})`);
        break;
      case 'zbl':
        want([1, 1], '(1,1)');
        if (num() !== 0) throw new StyleError(`MEAM zbl(1,1) = ${val} is not supported; set zbl(1,1) = 0 (${name})`);
        break;
      case 'rho0':
        want([1], '(1)');
        if (num() !== 1) throw new StyleError(`MEAM rho0 = ${val} is not supported; only 1 (${name})`);
        break;
      default:
        if (NUMERIC_DEFAULT_ZERO.includes(key)) {
          const k = idx.length ? `${key}(${idx.join(',')})` : key;
          if (num() !== 0) throw new StyleError(`MEAM keyword ${k} = ${val} is not supported (only the default 0; ${name})`);
          break;
        }
        throw new StyleError(`MEAM parameter keyword '${key}' is not supported (${name})`);
    }
  }
  if (out.opts.delr <= 0 || out.opts.rc <= out.opts.delr) throw new StyleError(`MEAM rc/delr must satisfy 0 < delr < rc (${name})`);
  return out;
};

/** pair_style meam: single element, energy and analytic forces, full neighbour list. */
export class PairMeam extends Pair {
  readonly name = 'meam';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private el: MeamElement | null = null;
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
    if (nelem !== 1) throw new StyleError('multi-element MEAM potentials are not supported (single element only)');
    const elem = args[3];
    const paramFile = args[3 + nelem];
    const maps = args.slice(4 + nelem);
    for (const m of maps) {
      if (m === 'NULL') throw new StyleError('NULL type mappings are not supported for pair_style meam');
      if (m !== elem) throw new StyleError(`element '${m}' is not the MEAM element '${elem}'`);
    }
    if (paramFile === 'NULL') {
      throw new StyleError('pair_coeff for style meam needs a parameter file: the NULL parameter file (zbl = 1 default, Ec and re without verified defaults) is not supported');
    }
    const lib = parseMeamLibrary(ctx.readFile(args[2]), elem, args[2]);
    const par = parseMeamParams(ctx.readFile(paramFile), paramFile);
    if (lib.lat !== 'fcc') throw new StyleError(`MEAM reference lattice '${lib.lat}' is not supported (only fcc)`);
    if (lib.t[0] !== 1) throw new StyleError('only MEAM parameters normalized to t0 = 1.0 are supported');
    if (lib.rozero !== 1) throw new StyleError(`MEAM rozero = ${lib.rozero} is not supported (only 1)`);
    if (lib.ibar !== 0) throw new StyleError(`MEAM ibar = ${lib.ibar} is not supported (only ibar = 0)`);
    if (par.Ec === undefined || par.re === undefined) {
      throw new StyleError(`MEAM parameter file ${paramFile} must set Ec(1,1) and re(1,1)`);
    }
    this.el = {
      z: lib.z,
      re: par.re,
      alpha: par.alpha ?? lib.alpha,
      Ec: par.Ec,
      A: lib.asub,
      beta: lib.b,
      t: lib.t,
      ibar: lib.ibar,
    };
    this.opts = par.opts;
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.el) throw new StyleError('pair_coeff for style meam must be set before the run');
    if (this.shift || this.tail) throw new StyleError('pair_style meam does not support the pair_modify shift and tail options');
    if (this.table !== 12) throw new StyleError('pair_style meam does not support the pair_modify table option');
    this.rhoRef = referenceBackground(this.el, this.opts);
    const t1 = this.opts.augt1 ? this.el.t[1] + 0.6 * this.el.t[3] : this.el.t[1];
    this.t = [1, t1, this.el.t[2], this.el.t[3]];
  }

  override initOne(_i: number, _j: number): number {
    if (!this.el) throw new StyleError('pair_coeff for style meam must be set before the run');
    return this.opts.rc;
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style meam needs a full neighbor list');
    if (!this.el) throw new StyleError('pair_coeff for style meam must be set before the run');
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const eatom = pc.eatom;
    let evdwl = 0;
    for (let i = 0; i < pc.nlocal; i++) {
      const ti = type[i] * nt;
      const nb: MeamNeighbor[] = [];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti + type[j]]) continue;
        nb.push({ j, dx, dy, dz, r: Math.sqrt(rsq) });
      }
      if (nb.length === 0) continue;
      const g = new Float64Array(3 * nb.length);
      const e = meamAtomEnergyGrad(this.el, this.opts, this.rhoRef, this.t, nb, g);
      evdwl += e;
      if (eatom) eatom[i] += e;
      for (let m = 0; m < nb.length; m++) {
        const j = nb[m].j;
        for (let c = 0; c < 3; c++) {
          f[3 * j + c] -= g[3 * m + c];
          f[3 * i + c] += g[3 * m + c];
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
