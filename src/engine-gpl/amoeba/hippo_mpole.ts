/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/amoeba_multipole.cpp (multipole_real(), the !amoeba branch with the charge penetration terms, and
 * damppole()) and src/AMOEBA/amoeba_polar.cpp (the damppole call of polar_real, shared by the HIPPO polarization) at
 * commit b332f28, adapted from Tinker emrealb(). AMOEBA package authors: the LAMMPS developers (Sandia Corporation,
 * GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * HIPPO permanent multipoles (pair hippo): the multipole energy, forces and torques with the core and valence charges
 * of each site (pcore, pval) and the charge penetration damping of Gordon's f1 model (damppole). The direct sum only
 * (no Ewald; the reciprocal and self terms of HIPPO are refused by pair_amoeba.ts). Pairs as in multipole.ts: copy
 * indices with their special codes; per-site data by owned atom.
 * Literature: J. A. Rackers, Q. Wang, C. Liu, J.-P. Piquemal, P. Ren and J. W. Ponder, PCCP 19, 276 (2017); L. V.
 * Slipchenko and M. S. Gordon, J. Comput. Chem. 28, 276 (2007).
 */
import { ELECTRIC, type MpoleSites } from './multipole';

/** sqrt(pi) (amoeba_multipole.cpp MY_PIS). */
const MY_PIS = Math.sqrt(Math.PI);
import { resolveSiteTorques, type TermPairs, type TermResult } from './hippo_terms';
import { exp as libmExp } from '../libm/exp';
import { erfc as libmErfc } from '../libm/erf';

/**
 * damppole (amoeba_multipole.cpp): the charge penetration damping factors dmpi and dmpk (entries 0, 2, 4, 6, 8, and the
 * pair factors dmpik with entries 0..10 for rorder 11, 0..8 for rorder 9), for the core-valence (alphai, alphak) pair.
 */
export const damppole = (
  r: number, rorder: number, alphai: number, alphak: number,
  dmpi: Float64Array, dmpk: Float64Array, dmpik: Float64Array,
): void => {
  const eps = 0.001;
  const diff = Math.abs(alphai - alphak);
  const dampi = alphai * r;
  const dampk = alphak * r;
  const expi = libmExp(-dampi);
  const expk = libmExp(-dampk);
  // core-valence charge penetration damping for Gordon f1
  const dampi2 = dampi * dampi;
  const dampi3 = dampi * dampi2;
  const dampi4 = dampi2 * dampi2;
  const dampi5 = dampi2 * dampi3;
  dmpi[0] = 1.0 - (1.0 + 0.5 * dampi) * expi;
  dmpi[2] = 1.0 - (1.0 + dampi + 0.5 * dampi2) * expi;
  dmpi[4] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0) * expi;
  dmpi[6] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + dampi4 / 30.0) * expi;
  dmpi[8] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + 4.0 * dampi4 / 105.0 + dampi5 / 210.0) * expi;
  // valence-valence charge penetration damping for Gordon f1
  if (diff < eps) {
    dmpk[0] = dmpi[0];
    dmpk[2] = dmpi[2];
    dmpk[4] = dmpi[4];
    dmpk[6] = dmpi[6];
    dmpk[8] = dmpi[8];
    const dampi6 = dampi3 * dampi3;
    const dampi7 = dampi3 * dampi4;
    dmpik[0] = 1.0 - (1.0 + 11.0 * dampi / 16.0 + 3.0 * dampi2 / 16.0 + dampi3 / 48.0) * expi;
    dmpik[2] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + 7.0 * dampi3 / 48.0 + dampi4 / 48.0) * expi;
    dmpik[4] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + dampi4 / 24.0 + dampi5 / 144.0) * expi;
    dmpik[6] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + dampi4 / 24.0 + dampi5 / 120.0 + dampi6 / 720.0) * expi;
    dmpik[8] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + dampi4 / 24.0 + dampi5 / 120.0 + dampi6 / 720.0 +
      dampi7 / 5040.0) * expi;
    if (rorder >= 11) {
      const dampi8 = dampi4 * dampi4;
      dmpik[10] = 1.0 - (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + dampi4 / 24.0 + dampi5 / 120.0 + dampi6 / 720.0 +
        dampi7 / 5040.0 + dampi8 / 45360.0) * expi;
    }
  } else {
    const dampk2 = dampk * dampk;
    const dampk3 = dampk * dampk2;
    const dampk4 = dampk2 * dampk2;
    const dampk5 = dampk2 * dampk3;
    dmpk[0] = 1.0 - (1.0 + 0.5 * dampk) * expk;
    dmpk[2] = 1.0 - (1.0 + dampk + 0.5 * dampk2) * expk;
    dmpk[4] = 1.0 - (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0) * expk;
    dmpk[6] = 1.0 - (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0 + dampk4 / 30.0) * expk;
    dmpk[8] = 1.0 - (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0 + 4.0 * dampk4 / 105.0 + dampk5 / 210.0) * expk;
    const alphai2 = alphai * alphai;
    const alphak2 = alphak * alphak;
    const termi = alphak2 / (alphak2 - alphai2);
    const termk = alphai2 / (alphai2 - alphak2);
    const termi2 = termi * termi;
    const termk2 = termk * termk;
    dmpik[0] = 1.0 - termi2 * (1.0 + 2.0 * termk + 0.5 * dampi) * expi - termk2 * (1.0 + 2.0 * termi + 0.5 * dampk) * expk;
    dmpik[2] = 1.0 - termi2 * (1.0 + dampi + 0.5 * dampi2) * expi - termk2 * (1.0 + dampk + 0.5 * dampk2) * expk -
      2.0 * termi2 * termk * (1.0 + dampi) * expi - 2.0 * termk2 * termi * (1.0 + dampk) * expk;
    dmpik[4] = 1.0 - termi2 * (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0) * expi -
      termk2 * (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0) * expk -
      2.0 * termi2 * termk * (1.0 + dampi + dampi2 / 3.0) * expi -
      2.0 * termk2 * termi * (1.0 + dampk + dampk2 / 3.0) * expk;
    dmpik[6] = 1.0 - termi2 * (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + dampi4 / 30.0) * expi -
      termk2 * (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0 + dampk4 / 30.0) * expk -
      2.0 * termi2 * termk * (1.0 + dampi + 2.0 * dampi2 / 5.0 + dampi3 / 15.0) * expi -
      2.0 * termk2 * termi * (1.0 + dampk + 2.0 * dampk2 / 5.0 + dampk3 / 15.0) * expk;
    dmpik[8] = 1.0 - termi2 * (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + 4.0 * dampi4 / 105.0 + dampi5 / 210.0) * expi -
      termk2 * (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0 + 4.0 * dampk4 / 105.0 + dampk5 / 210.0) * expk -
      2.0 * termi2 * termk * (1.0 + dampi + 3.0 * dampi2 / 7.0 + 2.0 * dampi3 / 21.0 + dampi4 / 105.0) * expi -
      2.0 * termk2 * termi * (1.0 + dampk + 3.0 * dampk2 / 7.0 + 2.0 * dampk3 / 21.0 + dampk4 / 105.0) * expk;
    if (rorder >= 11) {
      const dampi6 = dampi3 * dampi3;
      const dampk6 = dampk3 * dampk3;
      dmpik[10] = 1.0 - termi2 * (1.0 + dampi + 0.5 * dampi2 + dampi3 / 6.0 + 5.0 * dampi4 / 126.0 + 2.0 * dampi5 / 315.0 +
        dampi6 / 1890.0) * expi -
        termk2 * (1.0 + dampk + 0.5 * dampk2 + dampk3 / 6.0 + 5.0 * dampk4 / 126.0 + 2.0 * dampk5 / 315.0 +
        dampk6 / 1890.0) * expk -
        2.0 * termi2 * termk * (1.0 + dampi + 4.0 * dampi2 / 9.0 + dampi3 / 9.0 + dampi4 / 63.0 + dampi5 / 945.0) * expi -
        2.0 * termk2 * termi * (1.0 + dampk + 4.0 * dampk2 / 9.0 + dampk3 / 9.0 + dampk4 / 63.0 + dampk5 / 945.0) * expk;
    }
  }
};

/** Per-copy inputs of the HIPPO multipole term: core charge, damping exponent and valence charge of each copy. */
export interface HippoMpoleSites {
  /** pcore of the copy's class. */
  core: Float64Array;
  /** palpha of the copy's class. */
  alpha: Float64Array;
  /** pval of the copy's owner (the valence charge, pole minus the core charge). */
  val: Float64Array;
}

/**
 * multipole_real, the HIPPO branch (no Ewald): the charge penetration damped multipole energy over the half pair list,
 * with factor_mpole = mpole-1n-scale of the special code. Forces and torques as native; torques resolved by
 * torque2force. `own` maps copies to owned atoms (per-site moments and the rotated frames). The bn[k] are the Ewald
 * real-space terms of multipole_real (erfc damping with aewald; with aewald = 0 they are the rr(2k+1) times felec).
 */
export const multipoleRealHippo = (
  sites: MpoleSites, x: Float64Array, f: Float64Array, pairs: TermPairs,
  own: Int32Array, hs: HippoMpoleSites, factor: readonly number[], dielectric: number, mpolecut: number,
  aewald: number,
): TermResult => {
  const felec = ELECTRIC / dielectric;
  const off2 = mpolecut * mpolecut;
  const rp = sites.rpole;
  const n = sites.n;
  const tq = new Float64Array(3 * n);
  const vir = new Float64Array(6);
  const dmpi = new Float64Array(9), dmpk = new Float64Array(9), dmpij = new Float64Array(11);
  let etot = 0.0;
  for (let p = 0; p < pairs.npairs; p++) {
    const i = pairs.pi[p], j = pairs.pj[p];
    const scalek = factor[pairs.ps[p]];
    const xr = x[3 * j] - x[3 * i];
    const yr = x[3 * j + 1] - x[3 * i + 1];
    const zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    const r = Math.sqrt(r2);
    const oi = own[i], ok = own[j];
    const oi13 = 13 * oi, ok13 = 13 * ok;
    const dix = rp[oi13 + 1], diy = rp[oi13 + 2], diz = rp[oi13 + 3];
    const qixx = rp[oi13 + 4], qixy = rp[oi13 + 5], qixz = rp[oi13 + 6], qiyy = rp[oi13 + 8], qiyz = rp[oi13 + 9], qizz = rp[oi13 + 12];
    const dkx = rp[ok13 + 1], dky = rp[ok13 + 2], dkz = rp[ok13 + 3];
    const qkxx = rp[ok13 + 4], qkxy = rp[ok13 + 5], qkxz = rp[ok13 + 6], qkyy = rp[ok13 + 8], qkyz = rp[ok13 + 9], qkzz = rp[ok13 + 12];
    const corei = hs.core[i], alphai = hs.alpha[i], vali = hs.val[i];
    const corek = hs.core[j], alphak = hs.alpha[j], valk = hs.val[j];
    // intermediates involving moments and separation distance
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
    const dik = dix * dkx + diy * dky + diz * dkz;
    const qik = qix * qkx + qiy * qky + qiz * qkz;
    const diqk = dix * qkx + diy * qky + diz * qkz;
    const dkqi = dkx * qix + dky * qiy + dkz * qiz;
    const qiqk = 2.0 * (qixy * qkxy + qixz * qkxz + qiyz * qkyz) + qixx * qkxx + qiyy * qkyy + qizz * qkzz;
    // additional intermediates involving moments and distance
    const dirx = diy * zr - diz * yr, diry = diz * xr - dix * zr, dirz = dix * yr - diy * xr;
    const dkrx = dky * zr - dkz * yr, dkry = dkz * xr - dkx * zr, dkrz = dkx * yr - dky * xr;
    const dikx = diy * dkz - diz * dky, diky = diz * dkx - dix * dkz, dikz = dix * dky - diy * dkx;
    const qirx = qiz * yr - qiy * zr, qiry = qix * zr - qiz * xr, qirz = qiy * xr - qix * yr;
    const qkrx = qkz * yr - qky * zr, qkry = qkx * zr - qkz * xr, qkrz = qky * xr - qkx * yr;
    const qikx = qky * qiz - qkz * qiy, qiky = qkz * qix - qkx * qiz, qikz = qkx * qiy - qky * qix;
    const qixk = qixx * qkx + qixy * qky + qixz * qkz;
    const qiyk = qixy * qkx + qiyy * qky + qiyz * qkz;
    const qizk = qixz * qkx + qiyz * qky + qizz * qkz;
    const qkxi = qkxx * qix + qkxy * qiy + qkxz * qiz;
    const qkyi = qkxy * qix + qkyy * qiy + qkyz * qiz;
    const qkzi = qkxz * qix + qkyz * qiy + qkzz * qiz;
    const qikrx = qizk * yr - qiyk * zr, qikry = qixk * zr - qizk * xr, qikrz = qiyk * xr - qixk * yr;
    const qkirx = qkzi * yr - qkyi * zr, qkiry = qkxi * zr - qkzi * xr, qkirz = qkyi * xr - qkxi * yr;
    const diqkx = dix * qkxx + diy * qkxy + diz * qkxz;
    const diqky = dix * qkxy + diy * qkyy + diz * qkyz;
    const diqkz = dix * qkxz + diy * qkyz + diz * qkzz;
    const dkqix = dkx * qixx + dky * qixy + dkz * qixz;
    const dkqiy = dkx * qixy + dky * qiyy + dkz * qiyz;
    const dkqiz = dkx * qixz + dky * qiyz + dkz * qizz;
    const diqkrx = diqkz * yr - diqky * zr, diqkry = diqkx * zr - diqkz * xr, diqkrz = diqky * xr - diqkx * yr;
    const dkqirx = dkqiz * yr - dkqiy * zr, dkqiry = dkqix * zr - dkqiz * xr, dkqirz = dkqiy * xr - dkqix * yr;
    const dqikx = diy * qkz - diz * qky + dky * qiz - dkz * qiy -
      2.0 * (qixy * qkxz + qiyy * qkyz + qiyz * qkzz - qixz * qkxy - qiyz * qkyy - qizz * qkyz);
    const dqiky = diz * qkx - dix * qkz + dkz * qix - dkx * qiz -
      2.0 * (qixz * qkxx + qiyz * qkxy + qizz * qkxz - qixx * qkxz - qixy * qkyz - qixz * qkzz);
    const dqikz = dix * qky - diy * qkx + dkx * qiy - dky * qix -
      2.0 * (qixx * qkxy + qixy * qkyy + qixz * qkyz - qixy * qkxx - qiyy * qkxy - qiyz * qkxz);
    // reciprocal distance terms (the undamped ones of the scaled terms, as native)
    const rr1 = felec / r;
    const rr3 = rr1 / r2;
    const rr5 = 3.0 * rr3 / r2;
    const rr7 = 5.0 * rr5 / r2;
    const rr9 = 7.0 * rr7 / r2;
    const rr11 = 9.0 * rr9 / r2;
    // Ewald real-space terms bn[k] (multipole_real): erfc(aewald r)/r and the recursion, times felec
    const ralpha = aewald * r;
    const bn = [libmErfc(ralpha) / r, 0, 0, 0, 0, 0];
    const alsq2 = 2.0 * aewald * aewald;
    let alsq2n = aewald > 0.0 ? 1.0 / (MY_PIS * aewald) : 0.0;
    const exp2a = libmExp(-ralpha * ralpha);
    for (let k = 1; k < 6; k++) {
      alsq2n = alsq2 * alsq2n;
      bn[k] = ((2 * k - 1) * bn[k - 1] + alsq2n * exp2a) / r2;
    }
    for (let k = 0; k < 6; k++) bn[k] *= felec;
    // damped intermediates: the charge penetration factors of the core-valence and valence-valence pairs
    damppole(r, 11, alphai, alphak, dmpi, dmpk, dmpij);
    const rr1i = bn[0] - (1.0 - scalek * dmpi[0]) * rr1;
    const rr3i = bn[1] - (1.0 - scalek * dmpi[2]) * rr3;
    const rr5i = bn[2] - (1.0 - scalek * dmpi[4]) * rr5;
    const rr7i = bn[3] - (1.0 - scalek * dmpi[6]) * rr7;
    const rr1k = bn[0] - (1.0 - scalek * dmpk[0]) * rr1;
    const rr3k = bn[1] - (1.0 - scalek * dmpk[2]) * rr3;
    const rr5k = bn[2] - (1.0 - scalek * dmpk[4]) * rr5;
    const rr7k = bn[3] - (1.0 - scalek * dmpk[6]) * rr7;
    const rr1ik = bn[0] - (1.0 - scalek * dmpij[0]) * rr1;
    const rr3ik = bn[1] - (1.0 - scalek * dmpij[2]) * rr3;
    const rr5ik = bn[2] - (1.0 - scalek * dmpij[4]) * rr5;
    const rr7ik = bn[3] - (1.0 - scalek * dmpij[6]) * rr7;
    const rr9ik = bn[4] - (1.0 - scalek * dmpij[8]) * rr9;
    const rr11ik = bn[5] - (1.0 - scalek * dmpij[10]) * rr11;
    // the undamped core-core term is scaled by (1 - scalek) only
    const rr1s = bn[0] - (1.0 - scalek) * rr1;
    const rr3s = bn[1] - (1.0 - scalek) * rr3;
    // energy intermediates (native term1 .. term5 with the core and valence charges)
    const term1 = corei * corek;
    const term1i = corek * vali;
    const term2i = corek * dir;
    const term3i = corek * qir;
    const term1k = corei * valk;
    const term2k = -corei * dkr;
    const term3k = corei * qkr;
    const term1ik = vali * valk;
    const term2ik = valk * dir - vali * dkr + dik;
    const term3ik = vali * qkr + valk * qir - dir * dkr + 2.0 * (dkqi - diqk + qiqk);
    const term4ik = dir * qkr - dkr * qir - 4.0 * qik;
    const term5ik = qir * qkr;
    const e = term1 * rr1s + term4ik * rr7ik + term5ik * rr9ik + term1i * rr1i + term1k * rr1k + term1ik * rr1ik +
      term2i * rr3i + term2k * rr3k + term2ik * rr3ik + term3i * rr5i + term3k * rr5k + term3ik * rr5ik;
    const de = term1 * rr3s + term4ik * rr9ik + term5ik * rr11ik + term1i * rr3i + term1k * rr3k + term1ik * rr3ik +
      term2i * rr5i + term2k * rr5k + term2ik * rr5ik + term3i * rr7i + term3k * rr7k + term3ik * rr7ik;
    // force and torque intermediates (native reuses term1..term6 here)
    const t1 = -corek * rr3i - valk * rr3ik + dkr * rr5ik - qkr * rr7ik;
    const t2 = corei * rr3k + vali * rr3ik + dir * rr5ik + qir * rr7ik;
    const t3 = 2.0 * rr5ik;
    const t4 = -2.0 * (corek * rr5i + valk * rr5ik - dkr * rr7ik + qkr * rr9ik);
    const t5 = -2.0 * (corei * rr5k + vali * rr5ik + dir * rr7ik + qir * rr9ik);
    const t6 = 4.0 * rr7ik;
    const frcx = de * xr + t1 * dix + t2 * dkx + t3 * (diqkx - dkqix) + t4 * qix + t5 * qkx + t6 * (qixk + qkxi);
    const frcy = de * yr + t1 * diy + t2 * dky + t3 * (diqky - dkqiy) + t4 * qiy + t5 * qky + t6 * (qiyk + qkyi);
    const frcz = de * zr + t1 * diz + t2 * dkz + t3 * (diqkz - dkqiz) + t4 * qiz + t5 * qkz + t6 * (qizk + qkzi);
    // torques on both sites (rr3 = rr3ik in native)
    const ttmi0 = -rr3ik * dikx + t1 * dirx + t3 * (dqikx + dkqirx) - t4 * qirx - t6 * (qikrx + qikx);
    const ttmi1 = -rr3ik * diky + t1 * diry + t3 * (dqiky + dkqiry) - t4 * qiry - t6 * (qikry + qiky);
    const ttmi2 = -rr3ik * dikz + t1 * dirz + t3 * (dqikz + dkqirz) - t4 * qirz - t6 * (qikrz + qikz);
    const ttmk0 = rr3ik * dikx + t2 * dkrx - t3 * (dqikx + diqkrx) - t5 * qkrx - t6 * (qkirx - qikx);
    const ttmk1 = rr3ik * diky + t2 * dkry - t3 * (dqiky + diqkry) - t5 * qkry - t6 * (qkiry - qiky);
    const ttmk2 = rr3ik * dikz + t2 * dkrz - t3 * (dqikz + diqkrz) - t5 * qkrz - t6 * (qkirz - qikz);
    etot += e;
    f[3 * i] -= frcx;
    f[3 * i + 1] -= frcy;
    f[3 * i + 2] -= frcz;
    tq[3 * oi] += ttmi0;
    tq[3 * oi + 1] += ttmi1;
    tq[3 * oi + 2] += ttmi2;
    f[3 * j] += frcx;
    f[3 * j + 1] += frcy;
    f[3 * j + 2] += frcz;
    tq[3 * ok] += ttmk0;
    tq[3 * ok + 1] += ttmk1;
    tq[3 * ok + 2] += ttmk2;
    // pairwise Cartesian virial (virmpole -= v)
    vir[0] -= -xr * frcx;
    vir[1] -= -yr * frcy;
    vir[2] -= -zr * frcz;
    vir[3] -= -0.5 * (yr * frcx + xr * frcy);
    vir[4] -= -0.5 * (zr * frcx + xr * frcz);
    vir[5] -= -0.5 * (zr * frcy + yr * frcz);
  }
  resolveSiteTorques(sites, x, f, tq, vir);
  return { e: etot, virial: vir };
};
