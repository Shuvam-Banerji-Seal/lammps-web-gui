/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/amoeba_repulsion.cpp (repulsion(), damprep()), src/AMOEBA/amoeba_dispersion.cpp (dispersion_real(),
 * the self-energy with aewald = 0) and src/AMOEBA/amoeba_charge_transfer.cpp (charge_transfer()) at commit b332f28,
 * adapted from Tinker erepel1b(), edreal1d() and echgtrn1(). AMOEBA package authors: the LAMMPS developers (Sandia
 * Corporation, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The HIPPO-only pair terms of pair hippo: anisotropic Pauli repulsion (rpole moments with the Rackers-Ponder damping,
 * damprep), the damped dispersion with no taper (native dispersion_real), and the charge transfer. Pairs come from the
 * half list with their special codes (0 none, 1..4 the 1-2..1-5 relations), as in pair_amoeba.ts. Sites are indexed by
 * owned atom; a pair copy k maps to its owned atom own[k] for the per-site data.
 * Literature: J. A. Rackers and J. W. Ponder, J. Chem. Phys. 150, 084104 (2019) (repulsion); J. A. Rackers, C. Liu,
 * P. Ren and J. W. Ponder, J. Chem. Phys. 149, 084115 (2018) (dispersion); the charge transfer model of the HIPPO
 * force field (docs.lammps.org/pair_hippo.html).
 */
import { powint, taperCoeffs } from './hal';
import { torque2force, type MpoleSites } from './multipole';

/** Pair list of a term: copy indices and the special code of each pair. */
export interface TermPairs {
  pi: Int32Array;
  pj: Int32Array;
  ps: Int8Array;
  npairs: number;
}

/** Output of a term: the energy and the virial (xx, yy, zz, xy, xz, yz). */
export interface TermResult {
  e: number;
  virial: Float64Array;
}

/**
 * Converts the site torques of the owned atoms into forces (torque2force) and adds the frame correction of the virial,
 * as native does at the end of repulsion(), multipole_real() and charge-site loops. `tq` holds 3 values per owned site.
 */
export const resolveSiteTorques = (
  sites: MpoleSites, x: Float64Array, f: Float64Array, tq: Float64Array, vir: Float64Array,
): void => {
  for (let i = 0; i < sites.n; i++) {
    const { fx, fy, fz } = torque2force(sites, i, tq.subarray(3 * i, 3 * i + 3), x, f);
    if (sites.axis[i] === 'NOFRAME') continue;
    const iz = sites.zl[i], ix = sites.xl[i], iy = sites.yl[i];
    const xiz = x[3 * iz] - x[3 * i], yiz = x[3 * iz + 1] - x[3 * i + 1], ziz = x[3 * iz + 2] - x[3 * i + 2];
    const xix = x[3 * ix] - x[3 * i], yix = x[3 * ix + 1] - x[3 * i + 1], zix = x[3 * ix + 2] - x[3 * i + 2];
    const xiy = x[3 * iy] - x[3 * i], yiy = x[3 * iy + 1] - x[3 * i + 1], ziy = x[3 * iy + 2] - x[3 * i + 2];
    const vxx = xix * fx[0] + xiy * fy[0] + xiz * fz[0];
    const vxy = 0.5 * (yix * fx[0] + yiy * fy[0] + yiz * fz[0] + xix * fx[1] + xiy * fy[1] + xiz * fz[1]);
    const vxz = 0.5 * (zix * fx[0] + ziy * fy[0] + ziz * fz[0] + xix * fx[2] + xiy * fy[2] + xiz * fz[2]);
    const vyy = yix * fx[1] + yiy * fy[1] + yiz * fz[1];
    const vyz = 0.5 * (zix * fx[1] + ziy * fy[1] + ziz * fz[1] + yix * fx[2] + yiy * fy[2] + yiz * fz[2]);
    const vzz = zix * fx[2] + ziy * fy[2] + ziz * fz[2];
    vir[0] -= vxx;
    vir[1] -= vyy;
    vir[2] -= vzz;
    vir[3] -= vxy;
    vir[4] -= vxz;
    vir[5] -= vyz;
  }
};

/** Per-site data of the HIPPO pair terms: the owned atom type and class, and the copy-to-owner map. */
export interface HippoSiteData {
  /** Owned atom index of each copy (owned atoms map to themselves). */
  own: Int32Array;
  /** Atom type (amtype) of each owned atom. */
  types: Int32Array;
  /** Atom class of each copy (owned and image). */
  cls: Int32Array;
}

/**
 * damprep (amoeba_repulsion.cpp): the Pauli repulsion damping coefficients dmpik[0..10] (even entries) for the pair with
 * damping exponents dmpi and dmpk; rorder 11 includes dmpik[10].
 */
export const damprep = (
  r: number, r2: number, rr1: number, rr3: number, rr5: number, rr7: number, rr9: number, rr11: number,
  dmpi: number, dmpk: number, dmpik: Float64Array,
): void => {
  const eps = 0.001;
  const diff = Math.abs(dmpi - dmpk);
  let s: number, ds: number, d2s: number, d3s: number, d4s: number, d5s: number;
  let pre: number;
  if (diff < eps) {
    const r3 = r2 * r, r4 = r3 * r, r5 = r4 * r, r6 = r5 * r, r7 = r6 * r;
    const dmpi2 = 0.5 * dmpi;
    const dampi = dmpi2 * r;
    const expi = Math.exp(-dampi);
    const dmpi22 = dmpi2 * dmpi2;
    const dmpi23 = dmpi22 * dmpi2;
    const dmpi24 = dmpi23 * dmpi2;
    const dmpi25 = dmpi24 * dmpi2;
    const dmpi26 = dmpi25 * dmpi2;
    pre = 128.0;
    s = (r + dmpi2 * r2 + dmpi22 * r3 / 3.0) * expi;
    ds = (dmpi22 * r3 + dmpi23 * r4) * expi / 3.0;
    d2s = dmpi24 * expi * r5 / 9.0;
    d3s = dmpi25 * expi * r6 / 45.0;
    d4s = (dmpi25 * r6 + dmpi26 * r7) * expi / 315.0;
    const r8 = r7 * r;
    const dmpi27 = dmpi2 * dmpi26;
    d5s = (dmpi25 * r6 + dmpi26 * r7 + dmpi27 * r8 / 3.0) * expi / 945.0;
  } else {
    const r3 = r2 * r, r4 = r3 * r, r5 = r4 * r;
    const dmpi2 = 0.5 * dmpi;
    const dmpk2 = 0.5 * dmpk;
    const dampi = dmpi2 * r;
    const dampk = dmpk2 * r;
    const expi = Math.exp(-dampi);
    const expk = Math.exp(-dampk);
    const dmpi22 = dmpi2 * dmpi2;
    const dmpi23 = dmpi22 * dmpi2;
    const dmpi24 = dmpi23 * dmpi2;
    const dmpi25 = dmpi24 * dmpi2;
    const dmpk22 = dmpk2 * dmpk2;
    const dmpk23 = dmpk22 * dmpk2;
    const dmpk24 = dmpk23 * dmpk2;
    const dmpk25 = dmpk24 * dmpk2;
    const term = dmpi22 - dmpk22;
    pre = 8192.0 * dmpi23 * dmpk23 / (term * term * term * term);
    const tmp = 4.0 * dmpi2 * dmpk2 / term;
    s = (dampi - tmp) * expk + (dampk + tmp) * expi;
    ds = (dmpi2 * dmpk2 * r2 - 4.0 * dmpi2 * dmpk22 * r / term - 4.0 * dmpi2 * dmpk2 / term) * expk +
      (dmpi2 * dmpk2 * r2 + 4.0 * dmpi22 * dmpk2 * r / term + 4.0 * dmpi2 * dmpk2 / term) * expi;
    d2s = (dmpi2 * dmpk2 * r2 / 3.0 + dmpi2 * dmpk22 * r3 / 3.0 -
      (4.0 / 3.0) * dmpi2 * dmpk23 * r2 / term - 4.0 * dmpi2 * dmpk22 * r / term - 4.0 * dmpi2 * dmpk2 / term) * expk +
      (dmpi2 * dmpk2 * r2 / 3.0 + dmpi22 * dmpk2 * r3 / 3.0 +
      (4.0 / 3.0) * dmpi23 * dmpk2 * r2 / term + 4.0 * dmpi22 * dmpk2 * r / term + 4.0 * dmpi2 * dmpk2 / term) * expi;
    d3s = (dmpi2 * dmpk23 * r4 / 15.0 + dmpi2 * dmpk22 * r3 / 5.0 + dmpi2 * dmpk2 * r2 / 5.0 -
      (4.0 / 15.0) * dmpi2 * dmpk24 * r3 / term - (8.0 / 5.0) * dmpi2 * dmpk23 * r2 / term -
      4.0 * dmpi2 * dmpk22 * r / term - 4.0 / term * dmpi2 * dmpk2) * expk +
      (dmpi23 * dmpk2 * r4 / 15.0 + dmpi22 * dmpk2 * r3 / 5.0 + dmpi2 * dmpk2 * r2 / 5.0 +
      (4.0 / 15.0) * dmpi24 * dmpk2 * r3 / term + (8.0 / 5.0) * dmpi23 * dmpk2 * r2 / term +
      4.0 * dmpi22 * dmpk2 * r / term + 4.0 / term * dmpi2 * dmpk2) * expi;
    d4s = (dmpi2 * dmpk24 * r5 / 105.0 + (2.0 / 35.0) * dmpi2 * dmpk23 * r4 +
      dmpi2 * dmpk22 * r3 / 7.0 + dmpi2 * dmpk2 * r2 / 7.0 -
      (4.0 / 105.0) * dmpi2 * dmpk25 * r4 / term - (8.0 / 21.0) * dmpi2 * dmpk24 * r3 / term -
      (12.0 / 7.0) * dmpi2 * dmpk23 * r2 / term - 4.0 * dmpi2 * dmpk22 * r / term -
      4.0 * dmpi2 * dmpk2 / term) * expk +
      (dmpi24 * dmpk2 * r5 / 105.0 + (2.0 / 35.0) * dmpi23 * dmpk2 * r4 +
      dmpi22 * dmpk2 * r3 / 7.0 + dmpi2 * dmpk2 * r2 / 7.0 +
      (4.0 / 105.0) * dmpi25 * dmpk2 * r4 / term + (8.0 / 21.0) * dmpi24 * dmpk2 * r3 / term +
      (12.0 / 7.0) * dmpi23 * dmpk2 * r2 / term + 4.0 * dmpi22 * dmpk2 * r / term +
      4.0 * dmpi2 * dmpk2 / term) * expi;
    const r6v = r5 * r;
    const dmpi26 = dmpi25 * dmpi2;
    const dmpk26 = dmpk25 * dmpk2;
    d5s = (dmpi2 * dmpk25 * r6v / 945.0 + (2.0 / 189.0) * dmpi2 * dmpk24 * r5 +
      dmpi2 * dmpk23 * r4 / 21.0 + dmpi2 * dmpk22 * r3 / 9.0 + dmpi2 * dmpk2 * r2 / 9.0 -
      (4.0 / 945.0) * dmpi2 * dmpk26 * r5 / term -
      (4.0 / 63.0) * dmpi2 * dmpk25 * r4 / term - (4.0 / 9.0) * dmpi2 * dmpk24 * r3 / term -
      (16.0 / 9.0) * dmpi2 * dmpk23 * r2 / term - 4.0 * dmpi2 * dmpk22 * r / term -
      4.0 * dmpi2 * dmpk2 / term) * expk +
      (dmpi25 * dmpk2 * r6v / 945.0 + (2.0 / 189.0) * dmpi24 * dmpk2 * r5 +
      dmpi23 * dmpk2 * r4 / 21.0 + dmpi22 * dmpk2 * r3 / 9.0 + dmpi2 * dmpk2 * r2 / 9.0 +
      (4.0 / 945.0) * dmpi26 * dmpk2 * r5 / term + (4.0 / 63.0) * dmpi25 * dmpk2 * r4 / term +
      (4.0 / 9.0) * dmpi24 * dmpk2 * r3 / term + (16.0 / 9.0) * dmpi23 * dmpk2 * r2 / term +
      4.0 * dmpi22 * dmpk2 * r / term + 4.0 * dmpi2 * dmpk2 / term) * expi;
  }
  // partial derivatives to full derivatives
  s = s * rr1;
  ds = ds * rr3;
  d2s = d2s * rr5;
  d3s = d3s * rr7;
  d4s = d4s * rr9;
  d5s = d5s * rr11;
  dmpik[0] = 0.5 * pre * s * s;
  dmpik[2] = pre * s * ds;
  dmpik[4] = pre * (s * d2s + ds * ds);
  dmpik[6] = pre * (s * d3s + 3.0 * ds * d2s);
  dmpik[8] = pre * (s * d4s + 4.0 * ds * d3s + 3.0 * d2s * d2s);
  dmpik[10] = pre * (s * d5s + 5.0 * ds * d4s + 10.0 * d2s * d3s);
};

/**
 * repulsion (amoeba_repulsion.cpp): the Pauli repulsion energy and forces on the pairs within repcut, with the taper of
 * reptaper; sizes, damping exponents and valence charges per atom type (sizpr, dmppr, elepr), the rep-1n-scale factors
 * per special code, and torques on the sites resolved by torque2force.
 */
export const repulsionTerm = (
  sites: MpoleSites, x: Float64Array, f: Float64Array, pairs: TermPairs, sd: HippoSiteData,
  pr: { sizpr: Float64Array; dmppr: Float64Array; elepr: Float64Array }, special: readonly number[],
  off: number, cut: number,
): TermResult => {
  const rp = sites.rpole;
  const n = sites.n;
  const off2 = off * off, cut2 = cut * cut;
  const c = taperCoeffs(off, cut);
  const tq = new Float64Array(3 * n);
  const vir = new Float64Array(6);
  const dmpik = new Float64Array(11);
  let erep = 0.0;
  for (let p = 0; p < pairs.npairs; p++) {
    const i = pairs.pi[p], j = pairs.pj[p];
    const factorRepel = special[pairs.ps[p]];
    if (factorRepel === 0.0) continue;
    const xr = x[3 * j] - x[3 * i];
    const yr = x[3 * j + 1] - x[3 * i + 1];
    const zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    const oi = sd.own[i], ok = sd.own[j];
    const ti = sd.types[oi], tk = sd.types[ok];
    const sizi = pr.sizpr[ti], dmpi = pr.dmppr[ti], vali = pr.elepr[ti];
    const sizk = pr.sizpr[tk], dmpk = pr.dmppr[tk], valk = pr.elepr[tk];
    const r = Math.sqrt(r2);
    const oi13 = 13 * oi, ok13 = 13 * ok;
    const dix = rp[oi13 + 1], diy = rp[oi13 + 2], diz = rp[oi13 + 3];
    const qixx = rp[oi13 + 4], qixy = rp[oi13 + 5], qixz = rp[oi13 + 6], qiyy = rp[oi13 + 8], qiyz = rp[oi13 + 9], qizz = rp[oi13 + 12];
    const dkx = rp[ok13 + 1], dky = rp[ok13 + 2], dkz = rp[ok13 + 3];
    const qkxx = rp[ok13 + 4], qkxy = rp[ok13 + 5], qkxz = rp[ok13 + 6], qkyy = rp[ok13 + 8], qkyz = rp[ok13 + 9], qkzz = rp[ok13 + 12];
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
    // reciprocal distance terms (no felec: the repulsion is in native units)
    const rr1 = 1.0 / r;
    const rr3 = rr1 / r2;
    const rr5 = 3.0 * rr3 / r2;
    const rr7 = 5.0 * rr5 / r2;
    const rr9 = 7.0 * rr7 / r2;
    const rr11 = 9.0 * rr9 / r2;
    damprep(r, r2, rr1, rr3, rr5, rr7, rr9, rr11, dmpi, dmpk, dmpik);
    // energy intermediates
    let term1 = vali * valk;
    let term2 = valk * dir - vali * dkr + dik;
    let term3 = vali * qkr + valk * qir - dir * dkr + 2.0 * (dkqi - diqk + qiqk);
    let term4 = dir * qkr - dkr * qir - 4.0 * qik;
    const term5 = qir * qkr;
    const eterm = term1 * dmpik[0] + term2 * dmpik[2] + term3 * dmpik[4] + term4 * dmpik[6] + term5 * dmpik[8];
    const sizik = sizi * sizk * factorRepel;
    let e = sizik * eterm * rr1;
    // force intermediates
    const de = term1 * dmpik[2] + term2 * dmpik[4] + term3 * dmpik[6] + term4 * dmpik[8] + term5 * dmpik[10];
    term1 = -valk * dmpik[2] + dkr * dmpik[4] - qkr * dmpik[6];
    term2 = vali * dmpik[2] + dir * dmpik[4] + qir * dmpik[6];
    const t3 = 2.0 * dmpik[4];
    term4 = 2.0 * (-valk * dmpik[4] + dkr * dmpik[6] - qkr * dmpik[8]);
    const t5 = 2.0 * (-vali * dmpik[4] - dir * dmpik[6] - qir * dmpik[8]);
    const t6 = 4.0 * dmpik[6];
    let frcx = de * xr + term1 * dix + term2 * dkx + t3 * (diqkx - dkqix) + term4 * qix + t5 * qkx + t6 * (qixk + qkxi);
    let frcy = de * yr + term1 * diy + term2 * dky + t3 * (diqky - dkqiy) + term4 * qiy + t5 * qky + t6 * (qiyk + qkyi);
    let frcz = de * zr + term1 * diz + term2 * dkz + t3 * (diqkz - dkqiz) + term4 * qiz + t5 * qkz + t6 * (qizk + qkzi);
    frcx = frcx * rr1 + eterm * rr3 * xr;
    frcy = frcy * rr1 + eterm * rr3 * yr;
    frcz = frcz * rr1 + eterm * rr3 * zr;
    frcx *= sizik;
    frcy *= sizik;
    frcz *= sizik;
    // torques on both sites
    const ttri0 = (-dmpik[2] * dikx + term1 * dirx + t3 * (dqikx + dkqirx) - term4 * qirx - t6 * (qikrx + qikx)) * sizik * rr1;
    const ttri1 = (-dmpik[2] * diky + term1 * diry + t3 * (dqiky + dkqiry) - term4 * qiry - t6 * (qikry + qiky)) * sizik * rr1;
    const ttri2 = (-dmpik[2] * dikz + term1 * dirz + t3 * (dqikz + dkqirz) - term4 * qirz - t6 * (qikrz + qikz)) * sizik * rr1;
    let ttrk0 = (dmpik[2] * dikx + term2 * dkrx - t3 * (dqikx + diqkrx) - t5 * qkrx - t6 * (qkirx - qikx)) * sizik * rr1;
    let ttrk1 = (dmpik[2] * diky + term2 * dkry - t3 * (dqiky + diqkry) - t5 * qkry - t6 * (qkiry - qiky)) * sizik * rr1;
    let ttrk2 = (dmpik[2] * dikz + term2 * dkrz - t3 * (dqikz + diqkrz) - t5 * qkrz - t6 * (qkirz - qikz)) * sizik * rr1;
    let ttri = [ttri0, ttri1, ttri2];
    // energy switching near the cutoff
    if (r2 > cut2) {
      const r3 = r2 * r, r4 = r2 * r2, r5 = r2 * r3;
      const taper = c[5] * r5 + c[4] * r4 + c[3] * r3 + c[2] * r2 + c[1] * r + c[0];
      let dtaper = 5.0 * c[5] * r4 + 4.0 * c[4] * r3 + 3.0 * c[3] * r2 + 2.0 * c[2] * r + c[1];
      dtaper *= e * rr1;
      e *= taper;
      frcx = frcx * taper - dtaper * xr;
      frcy = frcy * taper - dtaper * yr;
      frcz = frcz * taper - dtaper * zr;
      ttri = ttri.map((v) => v * taper);
      ttrk0 *= taper;
      ttrk1 *= taper;
      ttrk2 *= taper;
    }
    erep += e;
    f[3 * i] -= frcx;
    f[3 * i + 1] -= frcy;
    f[3 * i + 2] -= frcz;
    tq[3 * oi] += ttri[0];
    tq[3 * oi + 1] += ttri[1];
    tq[3 * oi + 2] += ttri[2];
    f[3 * j] += frcx;
    f[3 * j + 1] += frcy;
    f[3 * j + 2] += frcz;
    tq[3 * ok] += ttrk0;
    tq[3 * ok + 1] += ttrk1;
    tq[3 * ok + 2] += ttrk2;
    // pairwise Cartesian virial (virrepulse -= v)
    vir[0] -= -xr * frcx;
    vir[1] -= -yr * frcy;
    vir[2] -= -zr * frcz;
    vir[3] -= -0.5 * (yr * frcx + xr * frcy);
    vir[4] -= -0.5 * (zr * frcx + xr * frcz);
    vir[5] -= -0.5 * (zr * frcy + yr * frcz);
  }
  resolveSiteTorques(sites, x, f, tq, vir);
  return { e: erep, virial: vir };
};

/**
 * dispersion_real (amoeba_dispersion.cpp, the real-space part with aewald = 0, no taper): the damped C6 dispersion of
 * the pairs within dispcut, scaled by disp-1n-scale, and the self-energy term (zero without the Ewald coefficient).
 */
export const dispersionReal = (
  x: Float64Array, f: Float64Array, pairs: TermPairs, own: Int32Array, cls: Int32Array, nlocal: number,
  disp: { csix: Float64Array; adisp: Float64Array }, special: readonly number[], off: number, aewald: number,
): TermResult => {
  const off2 = off * off;
  const vir = new Float64Array(6);
  let edisp = 0.0;
  for (let p = 0; p < pairs.npairs; p++) {
    const i = pairs.pi[p], j = pairs.pj[p];
    const factorDisp = special[pairs.ps[p]];
    const xr = x[3 * i] - x[3 * j];
    const yr = x[3 * i + 1] - x[3 * j + 1];
    const zr = x[3 * i + 2] - x[3 * j + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    const ci = disp.csix[cls[i]], ai = disp.adisp[cls[i]];
    const ck = disp.csix[cls[j]], ak = disp.adisp[cls[j]];
    // Ewald real-space factor (dispersion_real): term = 1 + ralpha2 + ralpha2^2/2 and expterm = exp(-ralpha2)
    const r6 = r2 * r2 * r2;
    const ralpha2 = r2 * aewald * aewald;
    const term = 1.0 + ralpha2 + 0.5 * ralpha2 * ralpha2;
    const expterm = Math.exp(-ralpha2);
    const expa = expterm * term;
    const r = Math.sqrt(r2);
    const r7 = r6 * r;
    const di = ai * r;
    const di2 = di * di;
    const di3 = di * di2;
    const dk = ak * r;
    const expi = Math.exp(-di);
    const expk = Math.exp(-dk);
    let damp3: number, damp5: number, ddamp: number;
    if (ai !== ak) {
      const ai2 = ai * ai, ak2 = ak * ak;
      const dk2 = dk * dk, dk3 = dk * dk2;
      const ti = ak2 / (ak2 - ai2);
      const ti2 = ti * ti;
      const tk = ai2 / (ai2 - ak2);
      const tk2 = tk * tk;
      damp3 = 1.0 - ti2 * (1.0 + di + 0.5 * di2) * expi - tk2 * (1.0 + dk + 0.5 * dk2) * expk -
        2.0 * ti2 * tk * (1.0 + di) * expi - 2.0 * tk2 * ti * (1.0 + dk) * expk;
      damp5 = 1.0 - ti2 * (1.0 + di + 0.5 * di2 + di3 / 6.0) * expi -
        tk2 * (1.0 + dk + 0.5 * dk2 + dk3 / 6.0) * expk -
        2.0 * ti2 * tk * (1.0 + di + di2 / 3.0) * expi - 2.0 * tk2 * ti * (1.0 + dk + dk2 / 3.0) * expk;
      ddamp = 0.25 * di2 * ti2 * ai * expi * (r * ai + 4.0 * tk - 1.0) +
        0.25 * dk2 * tk2 * ak * expk * (r * ak + 4.0 * ti - 1.0);
    } else {
      const di4 = di2 * di2, di5 = di2 * di3;
      damp3 = 1.0 - (1.0 + di + 0.5 * di2 + 7.0 * di3 / 48.0 + di4 / 48.0) * expi;
      damp5 = 1.0 - (1.0 + di + 0.5 * di2 + di3 / 6.0 + di4 / 24.0 + di5 / 144.0) * expi;
      ddamp = ai * expi * (di5 - 3.0 * di3 - 3.0 * di2) / 96.0;
    }
    const damp = 1.5 * damp5 - 0.5 * damp3;
    let scale = factorDisp * damp * damp;
    scale = scale - 1.0;
    const e = -ci * ck * (expa + scale) / r6;
    const rterm = -ralpha2 * ralpha2 * ralpha2 * expterm / r;
    const de = -6.0 * e / r2 - ci * ck * rterm / r7 - 2.0 * ci * ck * factorDisp * damp * ddamp / r7;
    edisp += e;
    const dedx = de * xr, dedy = de * yr, dedz = de * zr;
    f[3 * i] -= dedx;
    f[3 * i + 1] -= dedy;
    f[3 * i + 2] -= dedz;
    f[3 * j] += dedx;
    f[3 * j + 1] += dedy;
    f[3 * j + 2] += dedz;
    // virdisp -= v (xx, yy, zz, yx -> xy, zx -> xz, zy -> yz)
    vir[0] -= xr * dedx;
    vir[1] -= yr * dedy;
    vir[2] -= zr * dedz;
    vir[3] -= yr * dedx;
    vir[4] -= zr * dedx;
    vir[5] -= zr * dedy;
  }
  // self-energy: powint(aewald, 6) / 12 * csix^2 per owned atom (dispersion(), amoeba_dispersion.cpp)
  for (let i = 0; i < nlocal; i++) {
    const term = powint(aewald, 6) / 12.0;
    edisp += term * disp.csix[cls[i]] * disp.csix[cls[i]];
  }
  return { e: edisp, virial: vir };
};

/**
 * charge_transfer (amoeba_charge_transfer.cpp): the damped charge transfer of the pairs within ctrncut, scaled by the
 * mpole-1n-scale factors (zero-factor pairs skipped), with the taper of ctrntaper. Charges and dampings per class.
 */
export const chargeTransferTerm = (
  x: Float64Array, f: Float64Array, pairs: TermPairs, cls: Int32Array,
  ct: { chgct: Float64Array; dmpct: Float64Array }, specialMpole: readonly number[],
  off: number, cut: number, felec: number,
): TermResult => {
  const off2 = off * off, cut2 = cut * cut;
  const c = taperCoeffs(off, cut);
  const vir = new Float64Array(6);
  let eq = 0.0;
  for (let p = 0; p < pairs.npairs; p++) {
    const i = pairs.pi[p], j = pairs.pj[p];
    const factorMpole = specialMpole[pairs.ps[p]];
    if (factorMpole === 0.0) continue;
    const xr = x[3 * j] - x[3 * i];
    const yr = x[3 * j + 1] - x[3 * i + 1];
    const zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    const r = Math.sqrt(r2);
    const rr1 = 1.0 / r;
    const chgi = ct.chgct[cls[i]];
    let alphai = ct.dmpct[cls[i]];
    if (alphai === 0.0) alphai = 100.0;
    const chgj = ct.chgct[cls[j]];
    let alphaj = ct.dmpct[cls[j]];
    if (alphaj === 0.0) alphaj = 100.0;
    const expi = Math.exp(-alphai * r);
    const expj = Math.exp(-alphaj * r);
    let e = -chgi * expj - chgj * expi;
    let de = chgi * expj * alphaj + chgj * expi * alphai;
    e = felec * e * factorMpole;
    de = felec * de * factorMpole;
    if (r2 > cut2) {
      const r3 = r2 * r, r4 = r2 * r2, r5 = r2 * r3;
      const taper = c[5] * r5 + c[4] * r4 + c[3] * r3 + c[2] * r2 + c[1] * r + c[0];
      const dtaper = 5.0 * c[5] * r4 + 4.0 * c[4] * r3 + 3.0 * c[3] * r2 + 2.0 * c[2] * r + c[1];
      de = e * dtaper + de * taper;
      e *= taper;
    }
    eq += e;
    const frcx = de * xr * rr1, frcy = de * yr * rr1, frcz = de * zr * rr1;
    f[3 * i] += frcx;
    f[3 * i + 1] += frcy;
    f[3 * i + 2] += frcz;
    f[3 * j] -= frcx;
    f[3 * j + 1] -= frcy;
    f[3 * j + 2] -= frcz;
    vir[0] -= xr * frcx;
    vir[1] -= yr * frcy;
    vir[2] -= zr * frcz;
    vir[3] -= yr * frcx;
    vir[4] -= zr * frcx;
    vir[5] -= zr * frcy;
  }
  return { e: eq, virial: vir };
};
