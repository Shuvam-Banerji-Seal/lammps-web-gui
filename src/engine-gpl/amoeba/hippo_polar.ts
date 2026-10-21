/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/amoeba_induce.cpp (udirect2b and dampdir, umutual2b, uscale0b, dampmut, the !amoeba branches) and
 * src/AMOEBA/amoeba_polar.cpp (polar_real, the !amoeba branches: the charge-penetration damped direct field, the
 * damppole factors, the mutual forces) at commit b332f28, adapted from Tinker epolar1d() and the HIPPO induced
 * dipole code. AMOEBA package authors: the LAMMPS developers (Sandia Corporation, GNU General Public License). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The HIPPO induced-dipole polarization (pair hippo, poltyp MUTUAL, no Ewald): the direct field of the permanent core
 * and valence charges with the Gordon charge-penetration damping (dampdir), the mutual dipole-dipole tensors and the
 * preconditioner with the induce-1n-scale factors (dampmut), and the polarization forces of polar_real. The
 * conjugate-gradient solver, the energy and the torque conversion are the AMOEBA ones (polar.ts). Scale factors follow
 * native's HIPPO branch: within a polarization group the direct and p scalings are the polar-1n-intra factors, between
 * groups they are the polar-1n-scale factors; the mutual scaling is induce-1n-scale for every pair.
 */
import { damppole } from './hippo_mpole';
import { ELECTRIC } from './multipole';
import type { PolarAtoms, PolarSettings, InducedDipoles, PolarResult } from './polar';
import type { MpoleSites } from './multipole';
import { exp as libmExp } from '../libm/exp';
import { erfc as libmErfc } from '../libm/erf';

/** sqrt(pi) (amoeba_polar.cpp MY_PIS). */
const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)

/** Per-copy HIPPO inputs of the polarization: pcore and palpha of the class, pval of the owner (see pair_amoeba.ts). */
export interface HippoPolarAtoms {
  core: Float64Array;
  alphaP: Float64Array;
  val: Float64Array;
}

/**
 * dampdir (amoeba_induce.cpp): the core-valence direct field damping factors dmpi[2], [4], [6] of alphai and dmpk[2], [4],
 * [6] of alphak (the same when the exponents agree within 0.001).
 */
export const dampdir = (r: number, alphai: number, alphak: number, dmpi: Float64Array, dmpk: Float64Array): void => {
  const eps = 0.001;
  const diff = Math.abs(alphai - alphak);
  const dampi = alphai * r, dampk = alphak * r;
  const expi = libmExp(-dampi), expk = libmExp(-dampk);
  const dampi2 = dampi * dampi, dampi3 = dampi * dampi2, dampi4 = dampi2 * dampi2;
  dmpi[2] = 1.0 - (1.0 + dampi + 0.5 * dampi2) * expi;
  dmpi[4] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0) * expi;
  dmpi[6] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + dampi4 / 30.0) * expi;
  if (diff < eps) {
    dmpk[2] = dmpi[2];
    dmpk[4] = dmpi[4];
    dmpk[6] = dmpi[6];
  } else {
    const dampk2 = dampk * dampk, dampk3 = dampk * dampk2, dampk4 = dampk2 * dampk2;
    dmpk[2] = 1.0 - (1.0 + dampk + 0.5 * dampk2) * expk;
    dmpk[4] = 1.0 - (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0) * expk;
    dmpk[6] = 1.0 - (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0 + dampk4 / 30.0) * expk;
  }
};

/**
 * dampmut (amoeba_induce.cpp): the valence-valence mutual damping factors dmpik[2] and dmpik[4] for the exponents alphai
 * and alphak (the equal-exponent form when they agree within 0.001).
 */
export const dampmut = (r: number, alphai: number, alphak: number, dmpik: Float64Array): void => {
  const eps = 0.001;
  const diff = Math.abs(alphai - alphak);
  const dampi = alphai * r, dampk = alphak * r;
  const expi = libmExp(-dampi), expk = libmExp(-dampk);
  const dampi2 = dampi * dampi, dampi3 = dampi * dampi2;
  if (diff < eps) {
    const dampi4 = dampi2 * dampi2, dampi5 = dampi2 * dampi3;
    dmpik[2] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + 7.0 * dampi3 / 48.0 + dampi4 / 48.0) * expi;
    dmpik[4] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + dampi4 / 24.0 + dampi5 / 144.0) * expi;
  } else {
    const dampk2 = dampk * dampk, dampk3 = dampk * dampk2;
    const alphai2 = alphai * alphai, alphak2 = alphak * alphak;
    const termi = alphak2 / (alphak2 - alphai2);
    const termk = alphai2 / (alphai2 - alphak2);
    const termi2 = termi * termi, termk2 = termk * termk;
    dmpik[2] = 1.0 - termi2 * (1.0 + dampi + 0.5 * dampi2) * expi - termk2 * (1.0 + dampk + 0.5 * dampk2) * expk -
      2.0 * termi2 * termk * (1.0 + dampi) * expi - 2.0 * termk2 * termi * (1.0 + dampk) * expk;
    dmpik[4] = 1.0 - termi2 * (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0) * expi -
      termk2 * (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0) * expk -
      2.0 * termi2 * termk * (1.0 + dampi + dampi2 / 3.0) * expi -
      2.0 * termk2 * termi * (1.0 + dampk + dampk2 / 3.0) * expk;
  }
};

/**
 * The HIPPO scale factors of a pair (udirect2b, polar_real, uscale0b): within one polarization group the direct and p
 * factors are the polar-1n-intra values, between groups the polar-1n-scale values (no direct-11 factor).
 */
const hippoPScale = (a: PolarAtoms, s: PolarSettings, sb: number, I: number, J: number): number =>
  a.group[I] === a.group[J] ? s.piscale[sb] : s.pscale[sb];

/**
 * udirect2b (HIPPO branch): the direct field (dscale) and the p field (pscale) of the core-valence charges and the
 * dipole-dipole tensors of the mutual field (wscale) for each pair in range. Without Ewald the bn[m] are the reciprocal
 * distance terms rr(2m+1) (aewald = 0 is required; pair_amoeba.ts refuses HIPPO with Ewald).
 */
export const hippoDirectField = (a: PolarAtoms, h: HippoPolarAtoms, s: PolarSettings, pairs: ArrayLike<number>, off2: number) => {
  const n = a.n;
  const field = new Float64Array(3 * n), fieldp = new Float64Array(3 * n);
  // dfield0c: the reciprocal permanent field starts both fields (udirect1), then the real-space pairs add to them
  if (s.recipDirect) {
    field.set(s.recipDirect);
    fieldp.set(s.recipDirect);
  }
  const tdip = new Float64Array(6 * pairs.length);
  const inRange: boolean[] = new Array<boolean>(pairs.length).fill(false);
  const x = a.x, rp = a.rpole;
  const dmpi = new Float64Array(7), dmpk = new Float64Array(7), dmpik = new Float64Array(5);
  for (let q = 0; q < pairs.length; q++) {
    const p = pairs[q];
    const i = a.pi[p], j = a.pj[p];
    const I = a.own[i], J = a.own[j];
    const xr = x[3 * j] - x[3 * i], yr = x[3 * j + 1] - x[3 * i + 1], zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    inRange[q] = true;
    const sb = a.ps[p];
    const dscale = hippoPScale(a, s, sb, I, J);
    const pscale = dscale;
    const wscale = s.wscale[sb];
    const r = Math.sqrt(r2);
    const rr1 = 1.0 / r;
    const rr2 = rr1 * rr1;
    const rr3 = rr2 * rr1;
    const rr5 = 3.0 * rr2 * rr3;
    const rr7 = 5.0 * rr2 * rr5;
    // Ewald real-space terms of udirect2b: bn[0] = erfc(aewald r)/r and bn[m] = ((2m-1) bn[m-1] + aesq2n exp) / r^2
    const bnE = ewaldTerms(s.aewald, r, rr1, rr2, 3);
    const bn1 = bnE[1], bn2 = bnE[2], bn3 = bnE[3];
    const oi = 13 * I, ok = 13 * J;
    const dix = rp[oi + 1], diy = rp[oi + 2], diz = rp[oi + 3];
    const qixx = rp[oi + 4], qixy = rp[oi + 5], qixz = rp[oi + 6], qiyy = rp[oi + 8], qiyz = rp[oi + 9], qizz = rp[oi + 12];
    const dkx = rp[ok + 1], dky = rp[ok + 2], dkz = rp[ok + 3];
    const qkxx = rp[ok + 4], qkxy = rp[ok + 5], qkxz = rp[ok + 6], qkyy = rp[ok + 8], qkyz = rp[ok + 9], qkzz = rp[ok + 12];
    const corei = h.core[i], alphai = h.alphaP[i], vali = h.val[i];
    const corek = h.core[j], alphak = h.alphaP[j], valk = h.val[j];
    const dir = dix * xr + diy * yr + diz * zr;
    const qix = qixx * xr + qixy * yr + qixz * zr;
    const qiy = qixy * xr + qiyy * yr + qiyz * zr;
    const qiz = qixz * xr + qiyz * yr + qizz * zr;
    const qir = qix * xr + qiy * yr + qiz * zr;
    const dkr = dkx * xr + dky * yr + dkz * zr;
    const qkx = qkxx * xr + qkxy * yr + qkxz * zr;
    const qky = qkxy * xr + qkyy * yr + qkyz * zr;
    const qkz = qkxz * xr + qkyz * yr + qkzz * zr;
    const qkr = qkx * xr + qky * yr + qkz * zr;
    dampdir(r, alphai, alphak, dmpi, dmpk);
    // direct field with the dscale factors (the undamped core term rr3s uses the same scale)
    let scalek = dscale;
    const rr3i = bn1 - (1.0 - scalek * dmpi[2]) * rr3;
    const rr5i = bn2 - (1.0 - scalek * dmpi[4]) * rr5;
    const rr7i = bn3 - (1.0 - scalek * dmpi[6]) * rr7;
    const rr3k = bn1 - (1.0 - scalek * dmpk[2]) * rr3;
    const rr5k = bn2 - (1.0 - scalek * dmpk[4]) * rr5;
    const rr7k = bn3 - (1.0 - scalek * dmpk[6]) * rr7;
    const rr3s = bn1 - (1.0 - scalek) * rr3;
    const fid0 = -xr * (rr3s * corek + rr3k * valk - rr5k * dkr + rr7k * qkr) - rr3k * dkx + 2.0 * rr5k * qkx;
    const fid1 = -yr * (rr3s * corek + rr3k * valk - rr5k * dkr + rr7k * qkr) - rr3k * dky + 2.0 * rr5k * qky;
    const fid2 = -zr * (rr3s * corek + rr3k * valk - rr5k * dkr + rr7k * qkr) - rr3k * dkz + 2.0 * rr5k * qkz;
    const fkd0 = xr * (rr3s * corei + rr3i * vali + rr5i * dir + rr7i * qir) - rr3i * dix - 2.0 * rr5i * qix;
    const fkd1 = yr * (rr3s * corei + rr3i * vali + rr5i * dir + rr7i * qir) - rr3i * diy - 2.0 * rr5i * qiy;
    const fkd2 = zr * (rr3s * corei + rr3i * vali + rr5i * dir + rr7i * qir) - rr3i * diz - 2.0 * rr5i * qiz;
    // p field with the pscale factors
    scalek = pscale;
    const pr3i = bn1 - (1.0 - scalek * dmpi[2]) * rr3;
    const pr5i = bn2 - (1.0 - scalek * dmpi[4]) * rr5;
    const pr7i = bn3 - (1.0 - scalek * dmpi[6]) * rr7;
    const pr3k = bn1 - (1.0 - scalek * dmpk[2]) * rr3;
    const pr5k = bn2 - (1.0 - scalek * dmpk[4]) * rr5;
    const pr7k = bn3 - (1.0 - scalek * dmpk[6]) * rr7;
    const pr3s = bn1 - (1.0 - scalek) * rr3;
    const fip0 = -xr * (pr3s * corek + pr3k * valk - pr5k * dkr + pr7k * qkr) - pr3k * dkx + 2.0 * pr5k * qkx;
    const fip1 = -yr * (pr3s * corek + pr3k * valk - pr5k * dkr + pr7k * qkr) - pr3k * dky + 2.0 * pr5k * qky;
    const fip2 = -zr * (pr3s * corek + pr3k * valk - pr5k * dkr + pr7k * qkr) - pr3k * dkz + 2.0 * pr5k * qkz;
    const fkp0 = xr * (pr3s * corei + pr3i * vali + pr5i * dir + pr7i * qir) - pr3i * dix - 2.0 * pr5i * qix;
    const fkp1 = yr * (pr3s * corei + pr3i * vali + pr5i * dir + pr7i * qir) - pr3i * diy - 2.0 * pr5i * qiy;
    const fkp2 = zr * (pr3s * corei + pr3i * vali + pr5i * dir + pr7i * qir) - pr3i * diz - 2.0 * pr5i * qiz;
    // mutual dipole-dipole tensor with the wscale factors (dampmut)
    dampmut(r, alphai, alphak, dmpik);
    const rr3ik = bn1 - (1.0 - wscale * dmpik[2]) * rr3;
    const rr5ik = bn2 - (1.0 - wscale * dmpik[4]) * rr5;
    tdip[6 * q] = -rr3ik + rr5ik * xr * xr;
    tdip[6 * q + 1] = rr5ik * xr * yr;
    tdip[6 * q + 2] = rr5ik * xr * zr;
    tdip[6 * q + 3] = -rr3ik + rr5ik * yr * yr;
    tdip[6 * q + 4] = rr5ik * yr * zr;
    tdip[6 * q + 5] = -rr3ik + rr5ik * zr * zr;
    field[3 * I] += fid0;
    field[3 * I + 1] += fid1;
    field[3 * I + 2] += fid2;
    field[3 * J] += fkd0;
    field[3 * J + 1] += fkd1;
    field[3 * J + 2] += fkd2;
    fieldp[3 * I] += fip0;
    fieldp[3 * I + 1] += fip1;
    fieldp[3 * I + 2] += fip2;
    fieldp[3 * J] += fkp0;
    fieldp[3 * J + 1] += fkp1;
    fieldp[3 * J + 2] += fkp2;
  }
  // the self-energy portion of the permanent field (dfield0c): the dipole times the Ewald self term
  const term = (4.0 / 3.0) * s.aewald * s.aewald * s.aewald / MY_PIS;
  for (let i = 0; i < n; i++) {
    for (let m = 0; m < 3; m++) {
      field[3 * i + m] += term * rp[13 * i + 1 + m];
      fieldp[3 * i + m] += term * rp[13 * i + 1 + m];
    }
  }
  return { field, fieldp, tdip, inRange };
};

/**
 * The Ewald real-space terms of the induced-dipole sums (amoeba_induce.cpp udirect2b, with rr1 = 1/r and rr2 = 1/r^2):
 * bn[m] for m = 0..mmax (bn[0] = erfc(aewald r)/r, bn[m] = ((2m-1) bn[m-1] + aesq2n exp(-(aewald r)^2)) / r^2). With
 * aewald = 0 these are the reciprocal distance terms rr(2m+1).
 */
const ewaldTerms = (aewald: number, r: number, rr1: number, rr2: number, mmax: number): number[] => {
  const ralpha = aewald * r;
  const bn = [libmErfc(ralpha) * rr1];
  const aesq2 = 2.0 * aewald * aewald;
  let aesq2n = aewald > 0.0 ? 1.0 / (MY_PIS * aewald) : 0.0;
  const exp2a = libmExp(-ralpha * ralpha);
  for (let m = 1; m <= mmax; m++) {
    aesq2n = aesq2 * aesq2n;
    bn.push(((2 * m - 1) * bn[m - 1] + aesq2n * exp2a) * rr2);
  }
  return bn;
};

/**
 * uscale0b BUILD (HIPPO branch): the off-diagonal preconditioner tensor of each pair with the wscale factor and the
 * dampmut factors of the pair's exponents (the pair's special code kept in pc.ps).
 */
export const hippoPrecondTensors = (
  a: PolarAtoms, h: HippoPolarAtoms, pc: { pi: Int32Array; pj: Int32Array; ps: Int8Array }, s: PolarSettings,
): Float64Array => {
  const out = new Float64Array(6 * pc.pi.length);
  const x = a.x;
  const dmpik = new Float64Array(5);
  for (let q = 0; q < pc.pi.length; q++) {
    const i = pc.pi[q], j = pc.pj[q];
    const I = a.own[i], J = a.own[j];
    const xr = x[3 * j] - x[3 * i], yr = x[3 * j + 1] - x[3 * i + 1], zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    const r = Math.sqrt(r2);
    const poli = a.alpha[I];
    dampmut(r, h.alphaP[i], h.alphaP[j], dmpik);
    const wscale = s.wscale[pc.ps[q]];
    const scale3 = wscale * dmpik[2], scale5 = wscale * dmpik[4];
    const polik = poli * a.alpha[J];
    const rr3 = (scale3 * polik) / (r * r2);
    const rr5 = (3.0 * scale5 * polik) / (r * r2 * r2);
    out[6 * q] = rr5 * xr * xr - rr3;
    out[6 * q + 1] = rr5 * xr * yr;
    out[6 * q + 2] = rr5 * xr * zr;
    out[6 * q + 3] = rr5 * yr * yr - rr3;
    out[6 * q + 4] = rr5 * yr * zr;
    out[6 * q + 5] = rr5 * zr * zr - rr3;
  }
  return out;
};

/**
 * polar_real (HIPPO branch, poltyp MUTUAL, no Ewald): the polarization forces between the core-valence charges and the
 * induced dipoles (the direct term with the dscale damping, -2 times the derivative) and the mutual term (wscale), the
 * induced field gradients for the torques, and the torque conversion of every site (the AMOEBA tail, polarSiteTorques).
 */
export const hippoPolarReal = (
  a: PolarAtoms, h: HippoPolarAtoms, s: PolarSettings, sites: MpoleSites, ind: InducedDipoles, f: Float64Array,
  pairs: ArrayLike<number>, polarSiteTorques: (a: PolarAtoms, s: PolarSettings, sites: MpoleSites, ind: InducedDipoles,
    ufld: Float64Array, dufld: Float64Array, f: Float64Array, vir: Float64Array) => void,
): PolarResult => {
  const n = a.n;
  const felec = (0.5 * ELECTRIC) / s.dielectric;
  const off2 = s.ewaldcut * s.ewaldcut;
  const x = a.x, rp = a.rpole, uind = ind.uind, uinp = ind.uinp;
  const ufld = new Float64Array(3 * n);
  const dufld = new Float64Array(6 * n);
  const vir = new Float64Array(6);
  const dmpi = new Float64Array(9), dmpk = new Float64Array(9), dmpik = new Float64Array(9);
  for (let q = 0; q < pairs.length; q++) {
    const p = pairs[q];
    const i = a.pi[p], j = a.pj[p];
    const I = a.own[i], J = a.own[j];
    const xr = x[3 * j] - x[3 * i], yr = x[3 * j + 1] - x[3 * i + 1], zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    const sb = a.ps[p];
    const dscale = hippoPScale(a, s, sb, I, J);
    const wscale = s.wscale[sb];
    const r = Math.sqrt(r2);
    const oi = 13 * I, ok = 13 * J;
    const dix = rp[oi + 1], diy = rp[oi + 2], diz = rp[oi + 3];
    const qixx = rp[oi + 4], qixy = rp[oi + 5], qixz = rp[oi + 6], qiyy = rp[oi + 8], qiyz = rp[oi + 9], qizz = rp[oi + 12];
    const dkx = rp[ok + 1], dky = rp[ok + 2], dkz = rp[ok + 3];
    const qkxx = rp[ok + 4], qkxy = rp[ok + 5], qkxz = rp[ok + 6], qkyy = rp[ok + 8], qkyz = rp[ok + 9], qkzz = rp[ok + 12];
    const uix = uind[3 * I], uiy = uind[3 * I + 1], uiz = uind[3 * I + 2];
    const ukx = uind[3 * J], uky = uind[3 * J + 1], ukz = uind[3 * J + 2];
    const dir = dix * xr + diy * yr + diz * zr;
    const qix = qixx * xr + qixy * yr + qixz * zr;
    const qiy = qixy * xr + qiyy * yr + qiyz * zr;
    const qiz = qixz * xr + qiyz * yr + qizz * zr;
    const qir = qix * xr + qiy * yr + qiz * zr;
    const dkr = dkx * xr + dky * yr + dkz * zr;
    const qkx = qkxx * xr + qkxy * yr + qkxz * zr;
    const qky = qkxy * xr + qkyy * yr + qkyz * zr;
    const qkz = qkxz * xr + qkyz * yr + qkzz * zr;
    const qkr = qkx * xr + qky * yr + qkz * zr;
    const uir = uix * xr + uiy * yr + uiz * zr;
    const ukr = ukx * xr + uky * yr + ukz * zr;
    // reciprocal distance terms with the felec factor (aewald = 0: bn[m] = rr(2m+1))
    const rr1 = felec / r;
    const rr3 = rr1 / r2;
    const rr5 = (3.0 * rr3) / r2;
    const rr7 = (5.0 * rr5) / r2;
    const rr9 = (7.0 * rr7) / r2;
    // Ewald terms of polar_real: bn[m] = ((2m-1) bn[m-1] + alsq2n exp) / r^2, times felec (bn[0] = erfc(aewald r)/r)
    const bnE = ewaldTerms(s.aewald, r, 1.0 / r, 1.0 / r2, 4);
    const bn1 = bnE[1] * felec, bn2 = bnE[2] * felec, bn3 = bnE[3] * felec, bn4 = bnE[4] * felec;
    const corei = h.core[i], alphai = h.alphaP[i], vali = h.val[i];
    const corek = h.core[j], alphak = h.alphaP[j], valk = h.val[j];
    damppole(r, 9, alphai, alphak, dmpi, dmpk, dmpik);
    const rr3core = bn1 - (1.0 - dscale) * rr3;
    const rr5core = bn2 - (1.0 - dscale) * rr5;
    const rr3i = bn1 - (1.0 - dscale * dmpi[2]) * rr3;
    const rr5i = bn2 - (1.0 - dscale * dmpi[4]) * rr5;
    const rr7i = bn3 - (1.0 - dscale * dmpi[6]) * rr7;
    const rr9i = bn4 - (1.0 - dscale * dmpi[8]) * rr9;
    const rr3k = bn1 - (1.0 - dscale * dmpk[2]) * rr3;
    const rr5k = bn2 - (1.0 - dscale * dmpk[4]) * rr5;
    const rr7k = bn3 - (1.0 - dscale * dmpk[6]) * rr7;
    const rr9k = bn4 - (1.0 - dscale * dmpk[8]) * rr9;
    const rr5ik = bn2 - (1.0 - wscale * dmpik[4]) * rr5;
    const rr7ik = bn3 - (1.0 - wscale * dmpik[6]) * rr7;
    // induced field of the dipoles for the torques (tix3, tix5 ... with the direct-scaled damping)
    const tix3 = 2.0 * rr3i * ukx, tiy3 = 2.0 * rr3i * uky, tiz3 = 2.0 * rr3i * ukz;
    const tkx3 = 2.0 * rr3k * uix, tky3 = 2.0 * rr3k * uiy, tkz3 = 2.0 * rr3k * uiz;
    let tuir = -2.0 * rr5i * ukr;
    let tukr = -2.0 * rr5k * uir;
    ufld[3 * I] += tix3 + xr * tuir;
    ufld[3 * I + 1] += tiy3 + yr * tuir;
    ufld[3 * I + 2] += tiz3 + zr * tuir;
    ufld[3 * J] += tkx3 + xr * tukr;
    ufld[3 * J + 1] += tky3 + yr * tukr;
    ufld[3 * J + 2] += tkz3 + zr * tukr;
    const tix5 = 4.0 * (rr5i * ukx), tiy5 = 4.0 * (rr5i * uky), tiz5 = 4.0 * (rr5i * ukz);
    const tkx5 = 4.0 * (rr5k * uix), tky5 = 4.0 * (rr5k * uiy), tkz5 = 4.0 * (rr5k * uiz);
    tuir = -2.0 * rr7i * ukr;
    tukr = -2.0 * rr7k * uir;
    dufld[6 * I] += xr * tix5 + xr * xr * tuir;
    dufld[6 * I + 1] += xr * tiy5 + yr * tix5 + 2.0 * xr * yr * tuir;
    dufld[6 * I + 2] += yr * tiy5 + yr * yr * tuir;
    dufld[6 * I + 3] += xr * tiz5 + zr * tix5 + 2.0 * xr * zr * tuir;
    dufld[6 * I + 4] += yr * tiz5 + zr * tiy5 + 2.0 * yr * zr * tuir;
    dufld[6 * I + 5] += zr * tiz5 + zr * zr * tuir;
    dufld[6 * J] -= xr * tkx5 + xr * xr * tukr;
    dufld[6 * J + 1] -= xr * tky5 + yr * tkx5 + 2.0 * xr * yr * tukr;
    dufld[6 * J + 2] -= yr * tky5 + yr * yr * tukr;
    dufld[6 * J + 3] -= xr * tkz5 + zr * tkx5 + 2.0 * xr * zr * tukr;
    dufld[6 * J + 4] -= yr * tkz5 + zr * tky5 + 2.0 * yr * zr * tukr;
    dufld[6 * J + 5] -= zr * tkz5 + zr * zr * tukr;
    // direct polarization force (dEd/dR), HIPPO branch
    const dd = directForceHippo(
      { xr, yr, zr, corei, vali, corek, valk, dix, diy, diz, dkx, dky, dkz, dir, dkr, qixx, qixy, qixz, qiyy, qiyz, qizz,
        qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, qix, qiy, qiz, qir, qkx, qky, qkz, qkr, uix, uiy, uiz, ukx, uky, ukz },
      { rr3core, rr5core, rr3i, rr5i, rr7i, rr9i, rr3k, rr5k, rr7k, rr9k },
    );
    let frcx = -2.0 * dd[0], frcy = -2.0 * dd[1], frcz = -2.0 * dd[2];
    // mutual force (poltyp MUTUAL, !amoeba branch)
    const mm = mutualForceHippo(
      { xr, yr, zr, rr5ik, rr7ik, uir, ukr, uix, uiy, uiz, ukx, uky, ukz,
        uixp: uinp[3 * I], uiyp: uinp[3 * I + 1], uizp: uinp[3 * I + 2], ukxp: uinp[3 * J], ukyp: uinp[3 * J + 1], ukzp: uinp[3 * J + 2] },
    );
    frcx -= mm[0];
    frcy -= mm[1];
    frcz -= mm[2];
    f[3 * i] += frcx;
    f[3 * i + 1] += frcy;
    f[3 * i + 2] += frcz;
    f[3 * j] -= frcx;
    f[3 * j + 1] -= frcy;
    f[3 * j + 2] -= frcz;
    vir[0] -= xr * frcx;
    vir[1] -= yr * frcy;
    vir[2] -= zr * frcz;
    vir[3] -= 0.5 * (yr * frcx + xr * frcy);
    vir[4] -= 0.5 * (zr * frcx + xr * frcz);
    vir[5] -= 0.5 * (zr * frcy + yr * frcz);
  }
  polarSiteTorques(a, s, sites, ind, ufld, dufld, f, vir);
  return { virial: vir };
};

/** The direct HIPPO force terms (polar_real, the !amoeba block): the derivative vector depx, depy, depz (before -2). */
const directForceHippo = (
  m: {
    xr: number; yr: number; zr: number; corei: number; vali: number; corek: number; valk: number;
    dix: number; diy: number; diz: number; dkx: number; dky: number; dkz: number;
    dir: number; dkr: number;
    qixx: number; qixy: number; qixz: number; qiyy: number; qiyz: number; qizz: number;
    qkxx: number; qkxy: number; qkxz: number; qkyy: number; qkyz: number; qkzz: number;
    qix: number; qiy: number; qiz: number; qir: number; qkx: number; qky: number; qkz: number; qkr: number;
    uix: number; uiy: number; uiz: number; ukx: number; uky: number; ukz: number;
  },
  d: { rr3core: number; rr5core: number; rr3i: number; rr5i: number; rr7i: number; rr9i: number; rr3k: number; rr5k: number; rr7k: number; rr9k: number },
): [number, number, number] => {
  const { xr, yr, zr, corei, vali, corek, valk, dix, diy, diz, dkx, dky, dkz, dir, dkr } = m;
  const { qixx, qixy, qixz, qiyy, qiyz, qizz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz } = m;
  const { qix, qiy, qiz, qir, qkx, qky, qkz, qkr, uix, uiy, uiz, ukx, uky, ukz } = m;
  const { rr3core, rr5core, rr3i, rr5i, rr7i, rr9i, rr3k, rr5k, rr7k, rr9k } = d;
  // xx
  let term1i = rr3i - rr5i * xr * xr, term1core = rr3core - rr5core * xr * xr;
  let term2i = 2.0 * rr5i * xr, term3i = rr7i * xr * xr - rr5i, term4i = 2.0 * rr5i, term5i = 5.0 * rr7i * xr, term6i = rr9i * xr * xr;
  let term1k = rr3k - rr5k * xr * xr, term2k = 2.0 * rr5k * xr, term3k = rr7k * xr * xr - rr5k, term4k = 2.0 * rr5k;
  let term5k = 5.0 * rr7k * xr, term6k = rr9k * xr * xr;
  let term7i = 0.0, term8i = 0.0, term7k = 0.0, term8k = 0.0;
  const tixx = vali * term1i + corei * term1core + dix * term2i - dir * term3i - qixx * term4i + qix * term5i - qir * term6i +
    (qiy * yr + qiz * zr) * rr7i;
  const tkxx = valk * term1k + corek * term1core - dkx * term2k + dkr * term3k - qkxx * term4k + qkx * term5k - qkr * term6k +
    (qky * yr + qkz * zr) * rr7k;
  // yy
  term1i = rr3i - rr5i * yr * yr; term1core = rr3core - rr5core * yr * yr;
  term2i = 2.0 * rr5i * yr; term3i = rr7i * yr * yr - rr5i; term4i = 2.0 * rr5i; term5i = 5.0 * rr7i * yr; term6i = rr9i * yr * yr;
  term1k = rr3k - rr5k * yr * yr; term2k = 2.0 * rr5k * yr; term3k = rr7k * yr * yr - rr5k; term4k = 2.0 * rr5k;
  term5k = 5.0 * rr7k * yr; term6k = rr9k * yr * yr;
  const tiyy = vali * term1i + corei * term1core + diy * term2i - dir * term3i - qiyy * term4i + qiy * term5i - qir * term6i +
    (qix * xr + qiz * zr) * rr7i;
  const tkyy = valk * term1k + corek * term1core - dky * term2k + dkr * term3k - qkyy * term4k + qky * term5k - qkr * term6k +
    (qkx * xr + qkz * zr) * rr7k;
  // zz
  term1i = rr3i - rr5i * zr * zr; term1core = rr3core - rr5core * zr * zr;
  term2i = 2.0 * rr5i * zr; term3i = rr7i * zr * zr - rr5i; term4i = 2.0 * rr5i; term5i = 5.0 * rr7i * zr; term6i = rr9i * zr * zr;
  term1k = rr3k - rr5k * zr * zr; term2k = 2.0 * rr5k * zr; term3k = rr7k * zr * zr - rr5k; term4k = 2.0 * rr5k;
  term5k = 5.0 * rr7k * zr; term6k = rr9k * zr * zr;
  const tizz = vali * term1i + corei * term1core + diz * term2i - dir * term3i - qizz * term4i + qiz * term5i - qir * term6i +
    (qix * xr + qiy * yr) * rr7i;
  const tkzz = valk * term1k + corek * term1core - dkz * term2k + dkr * term3k - qkzz * term4k + qkz * term5k - qkr * term6k +
    (qkx * xr + qky * yr) * rr7k;
  // xy
  term2i = rr5i * xr; term1i = yr * term2i; term1core = rr5core * xr * yr; term3i = rr5i * yr; term4i = yr * (rr7i * xr);
  term5i = 2.0 * rr5i; term6i = 2.0 * rr7i * xr; term7i = 2.0 * rr7i * yr; term8i = yr * rr9i * xr;
  term2k = rr5k * xr; term1k = yr * term2k; term3k = rr5k * yr; term4k = yr * (rr7k * xr); term5k = 2.0 * rr5k;
  term6k = 2.0 * rr7k * xr; term7k = 2.0 * rr7k * yr; term8k = yr * rr9k * xr;
  const tixy = -vali * term1i - corei * term1core + diy * term2i + dix * term3i - dir * term4i - qixy * term5i + qiy * term6i +
    qix * term7i - qir * term8i;
  const tkxy = -valk * term1k - corek * term1core - dky * term2k - dkx * term3k + dkr * term4k - qkxy * term5k + qky * term6k +
    qkx * term7k - qkr * term8k;
  // xz
  term2i = rr5i * xr; term1i = zr * term2i; term1core = rr5core * xr * zr; term3i = rr5i * zr; term4i = zr * (rr7i * xr);
  term5i = 2.0 * rr5i; term6i = 2.0 * rr7i * xr; term7i = 2.0 * rr7i * zr; term8i = zr * rr9i * xr;
  term2k = rr5k * xr; term1k = zr * term2k; term3k = rr5k * zr; term4k = zr * (rr7k * xr); term5k = 2.0 * rr5k;
  term6k = 2.0 * rr7k * xr; term7k = 2.0 * rr7k * zr; term8k = zr * rr9k * xr;
  const tixz = -vali * term1i - corei * term1core + diz * term2i + dix * term3i - dir * term4i - qixz * term5i + qiz * term6i +
    qix * term7i - qir * term8i;
  const tkxz = -valk * term1k - corek * term1core - dkz * term2k - dkx * term3k + dkr * term4k - qkxz * term5k + qkz * term6k +
    qkx * term7k - qkr * term8k;
  // yz
  term2i = rr5i * yr; term1i = zr * term2i; term1core = rr5core * yr * zr; term3i = rr5i * zr; term4i = zr * (rr7i * yr);
  term5i = 2.0 * rr5i; term6i = 2.0 * rr7i * yr; term7i = 2.0 * rr7i * zr; term8i = zr * rr9i * yr;
  term2k = rr5k * yr; term1k = zr * term2k; term3k = rr5k * zr; term4k = zr * (rr7k * yr); term5k = 2.0 * rr5k;
  term6k = 2.0 * rr7k * yr; term7k = 2.0 * rr7k * zr; term8k = zr * rr9k * yr;
  const tiyz = -vali * term1i - corei * term1core + diz * term2i + diy * term3i - dir * term4i - qiyz * term5i + qiz * term6i +
    qiy * term7i - qir * term8i;
  const tkyz = -valk * term1k - corek * term1core - dkz * term2k - dky * term3k + dkr * term4k - qkyz * term5k + qkz * term6k +
    qky * term7k - qkr * term8k;
  const depx = tixx * ukx + tixy * uky + tixz * ukz - tkxx * uix - tkxy * uiy - tkxz * uiz;
  const depy = tixy * ukx + tiyy * uky + tiyz * ukz - tkxy * uix - tkyy * uiy - tkyz * uiz;
  const depz = tixz * ukx + tiyz * uky + tizz * ukz - tkxz * uix - tkyz * uiy - tkzz * uiz;
  return [depx, depy, depz];
};

/** The mutual HIPPO force terms (polar_real, poltyp MUTUAL with !amoeba): the derivative vector of the uind-uinp products. */
const mutualForceHippo = (
  m: {
    xr: number; yr: number; zr: number; rr5ik: number; rr7ik: number; uir: number; ukr: number;
    uix: number; uiy: number; uiz: number; ukx: number; uky: number; ukz: number;
    uixp: number; uiyp: number; uizp: number; ukxp: number; ukyp: number; ukzp: number;
  },
): [number, number, number] => {
  const { xr, yr, zr, rr5ik, rr7ik, uir, ukr, uix, uiy, uiz, ukx, uky, ukz, uixp, uiyp, uizp, ukxp, ukyp, ukzp } = m;
  let term1 = 2.0 * rr5ik;
  let term2 = term1 * xr;
  let term3 = rr5ik - rr7ik * xr * xr;
  const tixx = uix * term2 + uir * term3;
  const tkxx = ukx * term2 + ukr * term3;
  term2 = term1 * yr;
  term3 = rr5ik - rr7ik * yr * yr;
  const tiyy = uiy * term2 + uir * term3;
  const tkyy = uky * term2 + ukr * term3;
  term2 = term1 * zr;
  term3 = rr5ik - rr7ik * zr * zr;
  const tizz = uiz * term2 + uir * term3;
  const tkzz = ukz * term2 + ukr * term3;
  term1 = rr5ik * yr;
  term2 = rr5ik * xr;
  term3 = yr * (rr7ik * xr);
  const tixy = uix * term1 + uiy * term2 - uir * term3;
  const tkxy = ukx * term1 + uky * term2 - ukr * term3;
  term1 = rr5ik * zr;
  term3 = zr * (rr7ik * xr);
  const tixz = uix * term1 + uiz * term2 - uir * term3;
  const tkxz = ukx * term1 + ukz * term2 - ukr * term3;
  term2 = rr5ik * yr;
  term3 = zr * (rr7ik * yr);
  const tiyz = uiy * term1 + uiz * term2 - uir * term3;
  const tkyz = uky * term1 + ukz * term2 - ukr * term3;
  const depx = tixx * ukxp + tixy * ukyp + tixz * ukzp + tkxx * uixp + tkxy * uiyp + tkxz * uizp;
  const depy = tixy * ukxp + tiyy * ukyp + tiyz * ukzp + tkxy * uixp + tkyy * uiyp + tkyz * uizp;
  const depz = tixz * ukxp + tiyz * ukyp + tizz * ukzp + tkxz * uixp + tkyz * uiyp + tkzz * uizp;
  return [depx, depy, depz];
};
