/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/amoeba_hal.cpp (the buffered 14-7 van der Waals term, adapted from Tinker ehal1c()) at commit
 * b332f28. AMOEBA package authors: the LAMMPS developers (Sandia Corporation, GNU General Public License). This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The buffered 14-7 term on a list of pairs. Positions are the reduced hydrogen positions xred: a hydrogen (class with
 * kred > 0, bonded to one partner) sits at kred between its partner and its own position, and its force is shared
 * with the partner in the same proportions (amoeba_hal.cpp). Pairs are the half list; the caller passes the special
 * code of each pair (0 none, 1 1-2, 2 1-3, 3 1-4, 4 1-5) and the pairs with a zero special factor are dropped by the
 * caller (factor_hal == 0), as native does.
 */

export interface HalTables {
  /** Cutoff and taper distances (vdwcut, vdwtaper). */
  off: number;
  cut: number;
  /** Halgren constants (delta-halgren, gamma-halgren). */
  dhal: number;
  ghal: number;
  /** special_hal[0..4]: the factor per special code (index 0 = no special relation). */
  specialHal: readonly number[];
  /** Per class (1-based): kred, radmin, epsilon, radmin4, epsilon4 as (n+1)*(n+1) arrays. */
  kred: Float64Array;
  nclass: number;
  radmin: Float64Array;
  epsilon: Float64Array;
  radmin4: Float64Array;
  epsilon4: Float64Array;
}

export interface HalInput {
  /** Reduced positions of owned and ghost atoms (3 per copy; xred of pair_amoeba.ts). */
  xred: Float64Array;
  /** Atom classes (1-based) of owned and ghost copies. */
  cls: Int32Array;
  /** Index of the copy whose position the atom is reduced toward (itself when it is not a hydrogen). */
  red2local: Int32Array;
  /** Pair list: i, j, and the special code of the pair. */
  pi: Int32Array;
  pj: Int32Array;
  pspecial: Int8Array;
  npairs: number;
}

export interface HalResult {
  e: number;
  /** Virial in LAMMPS order xx yy zz xy xz yz. */
  virial: Float64Array;
}

/** MathSpecial::powint: integer power by repeated squaring (the same rounding as native). */
export const powint = (x: number, n: number): number => {
  let yy = n < 0 ? 1.0 / x : x;
  let ww = 1.0;
  for (let e = Math.abs(n); e !== 0; e >>= 1, yy *= yy) if (e & 1) ww *= yy;
  return ww;
};

/** Quintic taper coefficients c0..c5 for the cutoff range [cut, off] (pair_amoeba.cpp choose()). */
export const taperCoeffs = (off: number, cut: number): number[] => {
  const off2 = off * off, cut2 = cut * cut;
  const denom = powint(off - cut, 5);
  return [
    off * off2 * (off2 - 5.0 * off * cut + 10.0 * cut2) / denom,
    -30.0 * off2 * cut2 / denom,
    30.0 * (off2 * cut + off * cut2) / denom,
    -10.0 * (off2 + 4.0 * off * cut + cut2) / denom,
    15.0 * (off + cut) / denom,
    -6.0 / denom,
  ];
};

/**
 * Energy, forces (added to f, owned atoms) and virial of the buffered 14-7 term. The forces are added to
 * f[3*atom + k]; native sign: f[i] -= dedx for the pair (i, j) with dedx = dE/dr along x_ij / r.
 */
export const halTerm = (t: HalTables, inp: HalInput, f: Float64Array): HalResult => {
  const xred = inp.xred;
  const off2 = t.off * t.off, cut2 = t.cut * t.cut;
  const c = taperCoeffs(t.off, t.cut);
  const [c0, c1, c2, c3, c4, c5] = c;
  const nt = t.nclass + 1;
  const vir = new Float64Array(6);
  let ehal = 0;
  for (let p = 0; p < inp.npairs; p++) {
    const i = inp.pi[p], j = inp.pj[p];
    const sw = inp.pspecial[p];
    const factorHal = t.specialHal[sw];
    if (factorHal === 0.0) continue;
    const xr = xred[3 * i] - xred[3 * j];
    const yr = xred[3 * i + 1] - xred[3 * j + 1];
    const zr = xred[3 * i + 2] - xred[3 * j + 2];
    const rik2 = xr * xr + yr * yr + zr * zr;
    if (rik2 > off2) continue;
    const iclass = inp.cls[i], jclass = inp.cls[j];
    const k = iclass * nt + jclass;
    let rv = t.radmin[k];
    let eps = t.epsilon[k];
    if (sw === 3) {
      rv = t.radmin4[k];
      eps = t.epsilon4[k];
    }
    eps *= factorHal;
    const rik = Math.sqrt(rik2);
    const rv7 = powint(rv, 7);
    const rik6 = rik2 * rik2 * rik2;
    const rik7 = rik6 * rik;
    const rho = rik7 + t.ghal * rv7;
    const tau = (t.dhal + 1.0) / (rik + t.dhal * rv);
    const tau7 = powint(tau, 7);
    const dtau = tau / (t.dhal + 1.0);
    const q = rv7 / rho;
    const gtau = eps * tau7 * rik6 * (t.ghal + 1.0) * (q * q);
    let e = eps * tau7 * rv7 * ((t.ghal + 1.0) * rv7 / rho - 2.0);
    let de = -7.0 * (dtau * e + gtau);
    // energy switching near the cutoff (taper)
    if (rik2 > cut2) {
      const rik3 = rik2 * rik, rik4 = rik2 * rik2, rik5 = rik2 * rik3;
      const taper = c5 * rik5 + c4 * rik4 + c3 * rik3 + c2 * rik2 + c1 * rik + c0;
      const dtaper = 5.0 * c5 * rik4 + 4.0 * c4 * rik3 + 3.0 * c3 * rik2 + 2.0 * c2 * rik + c1;
      de = e * dtaper + de * taper;
      e *= taper;
    }
    ehal += e;
    de = de / rik;
    const dedx = de * xr, dedy = de * yr, dedz = de * zr;
    // forces: a reduced hydrogen gives kred of its force to itself and 1-kred to its partner
    const iv = inp.red2local[i], jv = inp.red2local[j];
    if (i === iv) {
      f[3 * i] -= dedx; f[3 * i + 1] -= dedy; f[3 * i + 2] -= dedz;
    } else {
      const redi = t.kred[iclass], rediv = 1.0 - redi;
      f[3 * i] -= dedx * redi; f[3 * i + 1] -= dedy * redi; f[3 * i + 2] -= dedz * redi;
      f[3 * iv] -= dedx * rediv; f[3 * iv + 1] -= dedy * rediv; f[3 * iv + 2] -= dedz * rediv;
    }
    if (j === jv) {
      f[3 * j] += dedx; f[3 * j + 1] += dedy; f[3 * j + 2] += dedz;
    } else {
      const redj = t.kred[jclass], redjv = 1.0 - redj;
      f[3 * j] += dedx * redj; f[3 * j + 1] += dedy * redj; f[3 * j + 2] += dedz * redj;
      f[3 * jv] += dedx * redjv; f[3 * jv + 1] += dedy * redjv; f[3 * jv + 2] += dedz * redjv;
    }
    // virial as amoeba_hal.cpp: virhal -= (xr*dedx, ...) with the reduced separation
    vir[0] -= xr * dedx;
    vir[1] -= yr * dedy;
    vir[2] -= zr * dedz;
    vir[3] -= yr * dedx;
    vir[4] -= zr * dedx;
    vir[5] -= zr * dedy;
  }
  return { e: ehal, virial: vir };
};
