/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/pair_amoeba.cpp (assign_groups, precond_neigh, choose(POLAR)), src/AMOEBA/amoeba_induce.cpp
 * (induce() with poltyp MUTUAL, udirect2b, umutual2b, dfield0c, ufield0c, uscale0b for the AMOEBA branch), and
 * src/AMOEBA/amoeba_polar.cpp (polar(), polar_energy(), polar_real() for the AMOEBA branch) at commit b332f28; the
 * reciprocal parts are in kspace.ts (udirect1, umutual1, polar_kspace). AMOEBA package authors: the LAMMPS
 * developers (Sandia Corporation, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Induced dipoles of pair amoeba: the permanent field with Thole damping and the Ewald real-space damping (aewald),
 * the mutual solver (preconditioned conjugate gradient, convergence on polar-eps), the polarization energy and the
 * polarization forces and torques. Polarization groups are the connected sets of atoms joined through a 1-2 bond
 * whose partner type is in the polarize line of the atom (docs.lammps.org/pair_amoeba.html, "polarize"). Periodic
 * images enter through the copies of the neighbor list: `own` maps a copy to its owned atom, and the dipoles, fields
 * and per-site data are indexed by owner while the geometry uses the copy. Not ported here: DIRECT and OPT
 * polarization and polar-predict.
 */
import { ELECTRIC, type MpoleSites, torque2force } from './multipole';
import { hippoDirectField, hippoPolarReal, hippoPrecondTensors, type HippoPolarAtoms } from './hippo_polar';
import { exp as libmExp } from '../libm/exp';
import { erfc as libmErfc } from '../libm/erf';

/** Constant converting q-Angstrom to Debye, used by the convergence test (amoeba_induce.cpp DEBYE). */
const DEBYE = 4.80321;
/** sqrt(pi) (amoeba_polar.cpp MY_PIS). */
const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)
/** Lower bound of the polarizability in the solver (amoeba_induce.cpp polmin). */
const POLMIN = 0.00000001;

/** The per-atom inputs of the polarization terms (arrays indexed by local atom index). */
export interface PolarAtoms {
  n: number;
  /** Owned + image positions (3 per copy); pi, pj index them. */
  x: Float64Array;
  /** Owned atom of each copy: per-atom and dipole data are indexed by owner, positions by copy. */
  own: Int32Array;
  /** Rotated permanent multipoles, 13 per atom (MpoleSites.rpole). */
  rpole: Float64Array;
  /** polarity (alpha), its sixth root (pdamp) and the Thole width of each atom's type. */
  alpha: Float64Array;
  pdamp: Float64Array;
  thole: Float64Array;
  /** Polarization group of each atom (equal values = same group). */
  group: Int32Array;
  /** Half-list pairs (local indices) and their special codes (0 none, 1..4 the 1-2..1-5 relations). */
  pi: Int32Array;
  pj: Int32Array;
  ps: Int8Array;
  /** HIPPO only: core charge, damping exponent and valence charge per copy (pair hippo). */
  hippo?: HippoPolarAtoms;
}

/** The scalars of the polarization settings (key file and PRM file). */
export interface PolarSettings {
  /** Ewald coefficient of the induced-dipole terms (pair_amoeba apewald; 0 without Ewald). */
  aewald: number;
  /** Reciprocal-space permanent field (udirect1), 3 per atom, or null: it starts the direct field (dfield0c). */
  recipDirect: Float64Array | null;
  /** Reciprocal-space mutual field (umutual1) of uind and uinp, the amounts to add to the fields, or null. */
  recipMutual: ((uind: Float64Array, uinp: Float64Array) => { field: Float64Array; fieldp: Float64Array }) | null;
  /** polarization scale factors per special code (index 0 = no special, always 1). */
  pscale: readonly number[];
  piscale: readonly number[];
  /** direct-11-scale and mutual-11-scale (same group only). */
  dscale: number;
  uscale: number;
  dielectric: number;
  ewaldcut: number;
  usolvcut: number;
  udiag: number;
  poleps: number;
  politer: number;
  pcgprec: boolean;
  pcgguess: boolean;
  pcgpeek: number;
  /** pair hippo: the HIPPO polarization (hippo_polar.ts); wscale is induce-1n-scale per special code (index 0 = 1). */
  hippo: boolean;
  wscale: readonly number[];
}

/** Output of the solver: induced dipoles (mutual, Thole-scaled by the pscale, uscale, dscale of the field). */
export interface InducedDipoles {
  uind: Float64Array;
  uinp: Float64Array;
  udir: Float64Array;
  udirp: Float64Array;
  /** Number of CG iterations (for tests and logs). */
  iterations: number;
}

/**
 * assign_groups (pair_amoeba.cpp): atoms are in one group when a chain of 1-2 bonds joins them and each step's partner
 * type is in the polarize line of the atom it leaves. Returns a group label per local atom (equal labels = same group).
 * `oneTwo` gives the local 1-2 partners of each local atom.
 */
export const assignPolarGroups = (
  n: number, oneTwo: readonly (readonly number[])[], types: ArrayLike<number>, polgroup: (t: number) => readonly number[],
): Int32Array => {
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (a: number): number => {
    let r = a;
    while (parent[r] !== r) r = parent[r];
    let c = a;
    while (parent[c] !== r) {
      const nx = parent[c];
      parent[c] = r;
      c = nx;
    }
    return r;
  };
  for (let m = 0; m < n; m++) {
    const group = polgroup(types[m]);
    for (const j of oneTwo[m]) {
      if (!group.includes(types[j])) continue;
      const a = find(m), b = find(j);
      if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
    }
  }
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = find(i);
  return out;
};

/** Thole damping of udirect2b / polar_real (AMOEBA branch): damp = pgamma (r/(pdi*pdj))^3, pgamma = min(thole_i, thole_j). */
const tholeDamp = (a: PolarAtoms, i: number, j: number, r: number): number => {
  const d = a.pdamp[i] * a.pdamp[j];
  if (d === 0) return 0;
  const pgamma = Math.min(a.thole[i], a.thole[j]);
  const q = r / d;
  return pgamma * q * q * q;
};

/**
 * udirect2b (AMOEBA branch): the permanent field at each atom with the direct (dscale, pscale) scalings, and the
 * dipole-dipole tensors of the mutual field for each pair within the polar cutoff. `fieldD` is the field of the direct
 * scaling (dscale), `fieldP` of the preconditioner-free p-scaling (pscale).
 */
const directField = (a: PolarAtoms, s: PolarSettings, pairs: readonly number[], off2: number) => {
  const n = a.n;
  const field = new Float64Array(3 * n), fieldp = new Float64Array(3 * n);
  // dfield0c: the reciprocal permanent field starts both fields, then the real-space pairs add to them
  if (s.recipDirect) {
    field.set(s.recipDirect);
    fieldp.set(s.recipDirect);
  }
  const tdip = new Float64Array(6 * pairs.length);
  const inRange: boolean[] = new Array<boolean>(pairs.length).fill(false);
  const x = a.x, rp = a.rpole;
  for (let q = 0; q < pairs.length; q++) {
    const p = pairs[q];
    const i = a.pi[p], j = a.pj[p];
    const I = a.own[i], J = a.own[j];
    const xr = x[3 * j] - x[3 * i], yr = x[3 * j + 1] - x[3 * i + 1], zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    inRange[q] = true;
    const sb = a.ps[p];
    let pscale: number, dscale: number, uscale: number;
    if (a.group[I] === a.group[J]) {
      pscale = s.piscale[sb];
      dscale = s.dscale;
      uscale = s.uscale;
    } else {
      pscale = s.pscale[sb];
      dscale = 1.0;
      uscale = 1.0;
    }
    const r = Math.sqrt(r2);
    const rr1 = 1.0 / r;
    const rr2 = rr1 * rr1;
    const rr3 = rr2 * rr1;
    const rr5 = 3.0 * rr2 * rr3;
    const rr7 = 5.0 * rr2 * rr5;
    // reciprocal distance terms without Ewald (bn[m] = (2m-1) bn[m-1] / r^2, bn[0] = 1/r)
    // Ewald damping of the reciprocal distance terms (amoeba_induce.cpp udirect2b): bn[0] = erfc(aewald r)/r and
    // bn[m] = ((2m-1) bn[m-1] + aesq2n exp(-(aewald r)^2)) / r^2 with aesq2n = (2 aewald^2)^m / (sqrt(pi) aewald)
    const ralpha = s.aewald * r;
    const bn = [libmErfc(ralpha) * rr1, 0, 0, 0];
    const aesq2 = 2.0 * s.aewald * s.aewald;
    let aesq2n = s.aewald > 0.0 ? 1.0 / (MY_PIS * s.aewald) : 0.0;
    const exp2a = libmExp(-ralpha * ralpha);
    for (let m = 1; m <= 3; m++) {
      aesq2n = aesq2 * aesq2n;
      bn[m] = ((2 * m - 1) * bn[m - 1] + aesq2n * exp2a) * rr2;
    }
    const oi = 13 * I, ok = 13 * J;
    const ci = rp[oi], dix = rp[oi + 1], diy = rp[oi + 2], diz = rp[oi + 3];
    const qixx = rp[oi + 4], qixy = rp[oi + 5], qixz = rp[oi + 6], qiyy = rp[oi + 8], qiyz = rp[oi + 9], qizz = rp[oi + 12];
    const ck = rp[ok], dkx = rp[ok + 1], dky = rp[ok + 2], dkz = rp[ok + 3];
    const qkxx = rp[ok + 4], qkxy = rp[ok + 5], qkxz = rp[ok + 6], qkyy = rp[ok + 8], qkyz = rp[ok + 9], qkzz = rp[ok + 12];
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
    // Thole damping scale factors (the same for the direct, p and mutual terms)
    let scale3 = 1.0, scale5 = 1.0, scale7 = 1.0;
    const damp = tholeDamp(a, I, J, r);
    if (a.pdamp[I] * a.pdamp[J] !== 0) {
      if (damp < 50.0) {
        const expdamp = libmExp(-damp);
        scale3 = 1.0 - expdamp;
        scale5 = 1.0 - expdamp * (1.0 + damp);
        scale7 = 1.0 - expdamp * (1.0 + damp + 0.6 * damp * damp);
      }
    }
    // direct field (dscale for the field, pscale for the p field)
    let scalek = dscale;
    let bcn0 = bn[1] - (1.0 - scalek * scale3) * rr3;
    let bcn1 = bn[2] - (1.0 - scalek * scale5) * rr5;
    let bcn2 = bn[3] - (1.0 - scalek * scale7) * rr7;
    const fid = [
      -xr * (bcn0 * ck - bcn1 * dkr + bcn2 * qkr) - bcn0 * dkx + 2.0 * bcn1 * qkx,
      -yr * (bcn0 * ck - bcn1 * dkr + bcn2 * qkr) - bcn0 * dky + 2.0 * bcn1 * qky,
      -zr * (bcn0 * ck - bcn1 * dkr + bcn2 * qkr) - bcn0 * dkz + 2.0 * bcn1 * qkz,
    ];
    const fkd = [
      xr * (bcn0 * ci + bcn1 * dir + bcn2 * qir) - bcn0 * dix - 2.0 * bcn1 * qix,
      yr * (bcn0 * ci + bcn1 * dir + bcn2 * qir) - bcn0 * diy - 2.0 * bcn1 * qiy,
      zr * (bcn0 * ci + bcn1 * dir + bcn2 * qir) - bcn0 * diz - 2.0 * bcn1 * qiz,
    ];
    scalek = pscale;
    bcn0 = bn[1] - (1.0 - scalek * scale3) * rr3;
    bcn1 = bn[2] - (1.0 - scalek * scale5) * rr5;
    bcn2 = bn[3] - (1.0 - scalek * scale7) * rr7;
    const fip = [
      -xr * (bcn0 * ck - bcn1 * dkr + bcn2 * qkr) - bcn0 * dkx + 2.0 * bcn1 * qkx,
      -yr * (bcn0 * ck - bcn1 * dkr + bcn2 * qkr) - bcn0 * dky + 2.0 * bcn1 * qky,
      -zr * (bcn0 * ck - bcn1 * dkr + bcn2 * qkr) - bcn0 * dkz + 2.0 * bcn1 * qkz,
    ];
    const fkp = [
      xr * (bcn0 * ci + bcn1 * dir + bcn2 * qir) - bcn0 * dix - 2.0 * bcn1 * qix,
      yr * (bcn0 * ci + bcn1 * dir + bcn2 * qir) - bcn0 * diy - 2.0 * bcn1 * qiy,
      zr * (bcn0 * ci + bcn1 * dir + bcn2 * qir) - bcn0 * diz - 2.0 * bcn1 * qiz,
    ];
    // mutual dipole-dipole tensor (uscale), built with the 3- and 5-term damping only
    let mscale3 = 1.0, mscale5 = 1.0;
    if (a.pdamp[I] * a.pdamp[J] !== 0 && damp < 50.0) {
      const expdamp = libmExp(-damp);
      mscale3 = 1.0 - expdamp;
      mscale5 = 1.0 - expdamp * (1.0 + damp);
    }
    scalek = uscale;
    const mb0 = bn[1] - (1.0 - scalek * mscale3) * rr3;
    const mb1 = bn[2] - (1.0 - scalek * mscale5) * rr5;
    tdip[6 * q] = -mb0 + mb1 * xr * xr;
    tdip[6 * q + 1] = mb1 * xr * yr;
    tdip[6 * q + 2] = mb1 * xr * zr;
    tdip[6 * q + 3] = -mb0 + mb1 * yr * yr;
    tdip[6 * q + 4] = mb1 * yr * zr;
    tdip[6 * q + 5] = -mb0 + mb1 * zr * zr;
    for (let m = 0; m < 3; m++) {
      field[3 * I + m] += fid[m];
      field[3 * J + m] += fkd[m];
      fieldp[3 * I + m] += fip[m];
      fieldp[3 * J + m] += fkp[m];
    }
  }
  // the self-energy portion of the permanent field (amoeba_induce.cpp dfield0c): term times the dipole of the atom
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
 * umutual2b: the mutual field of the induced dipoles (uind for the field, uinp for the p field) over the pairs in range.
 * Each pair adds T.u of the other atom to both atoms (the half list).
 */
const mutualField = (
  a: PolarAtoms, aewald: number, recip: PolarSettings['recipMutual'], pairs: readonly number[], tdip: Float64Array, inRange: readonly boolean[],
  uind: Float64Array, uinp: Float64Array,
): { field: Float64Array; fieldp: Float64Array } => {
  const field = new Float64Array(3 * a.n), fieldp = new Float64Array(3 * a.n);
  for (let q = 0; q < pairs.length; q++) {
    if (!inRange[q]) continue;
    const p = pairs[q];
    const i = a.pi[p], j = a.pj[p];
    const I = a.own[i], J = a.own[j];
    const t = 6 * q;
    // row m of the symmetric 3x3 tensor (xx xy xz yy yz zz) times the other atom's dipole, in the order m = 0, 1, 2
    for (let m = 0; m < 3; m++) {
      const c0 = tdip[t + (m === 0 ? 0 : m === 1 ? 1 : 2)];
      const c1 = tdip[t + (m === 0 ? 1 : m === 1 ? 3 : 4)];
      const c2 = tdip[t + (m === 0 ? 2 : m === 1 ? 4 : 5)];
      field[3 * I + m] += c0 * uind[3 * J] + c1 * uind[3 * J + 1] + c2 * uind[3 * J + 2];
      field[3 * J + m] += c0 * uind[3 * I] + c1 * uind[3 * I + 1] + c2 * uind[3 * I + 2];
      fieldp[3 * I + m] += c0 * uinp[3 * J] + c1 * uinp[3 * J + 1] + c2 * uinp[3 * J + 2];
      fieldp[3 * J + m] += c0 * uinp[3 * I] + c1 * uinp[3 * I + 1] + c2 * uinp[3 * I + 2];
    }
  }
  // reciprocal-space mutual field (ufield0c: umutual1 after umutual2b)
  if (recip) {
    const r = recip(uind, uinp);
    for (let k = 0; k < 3 * a.n; k++) {
      field[k] += r.field[k];
      fieldp[k] += r.fieldp[k];
    }
  }
  // the self-energy portion of the mutual field (amoeba_induce.cpp ufield0c): term times the induced dipole
  const term = (4.0 / 3.0) * aewald * aewald * aewald / MY_PIS;
  for (let i = 0; i < a.n; i++) {
    for (let m = 0; m < 3; m++) {
      field[3 * i + m] += term * uind[3 * i + m];
      fieldp[3 * i + m] += term * uinp[3 * i + m];
    }
  }
  return { field, fieldp };
};

/** Preconditioner pairs (uscale0b): the atoms of the pairs within usolvcut at the last neighbor build, and their 6-tensor. */
export interface PrecondPairs {
  pi: Int32Array;
  pj: Int32Array;
  /** Special code of each preconditioner pair (for the HIPPO wscale factor). */
  ps: Int8Array;
}

/** pair list of the preconditioner (precond_neigh): pairs of the half list with r^2 < usolvcut^2. */
export const precondPairs = (a: PolarAtoms, usolvcut: number): PrecondPairs => {
  const cut2 = usolvcut * usolvcut;
  const pi: number[] = [], pj: number[] = [], ps: number[] = [];
  const x = a.x;
  for (let p = 0; p < a.pi.length; p++) {
    const i = a.pi[p], j = a.pj[p];
    const xr = x[3 * j] - x[3 * i], yr = x[3 * j + 1] - x[3 * i + 1], zr = x[3 * j + 2] - x[3 * i + 2];
    if (xr * xr + yr * yr + zr * zr < cut2) {
      pi.push(i);
      pj.push(j);
      ps.push(a.ps[p]);
    }
  }
  return { pi: Int32Array.from(pi), pj: Int32Array.from(pj), ps: Int8Array.from(ps) };
};

/** uscale0b BUILD: the off-diagonal 3x3 preconditioner of each pair (6 values, no cutoff), at the current positions. */
export const buildPrecondTensors = (a: PolarAtoms, pc: PrecondPairs, s: PolarSettings): Float64Array => {
  if (s.hippo) return hippoPrecondTensors(a, requireHippo(a), pc, s);
  const out = new Float64Array(6 * pc.pi.length);
  const x = a.x;
  for (let q = 0; q < pc.pi.length; q++) {
    const i = pc.pi[q], j = pc.pj[q];
    const I = a.own[i], J = a.own[j];
    const xr = x[3 * j] - x[3 * i], yr = x[3 * j + 1] - x[3 * i + 1], zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    const r = Math.sqrt(r2);
    const poli = a.alpha[I];
    const factorUscale = a.group[I] === a.group[J] ? s.uscale : 1.0;
    let scale3 = factorUscale, scale5 = factorUscale;
    const damp0 = a.pdamp[I] * a.pdamp[J];
    if (damp0 !== 0) {
      const pgamma = Math.min(a.thole[I], a.thole[J]);
      const damp = -pgamma * (r / damp0) * (r / damp0) * (r / damp0);
      if (damp > -50.0) {
        const expdamp = libmExp(damp);
        scale3 *= 1.0 - expdamp;
        scale5 *= 1.0 - expdamp * (1.0 - damp);
      }
    }
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

/** uscale0b APPLY: z = M^-1 r with the diagonal (udiag * alpha) and the pair tensors. */
const applyPrecond = (
  a: PolarAtoms, pc: PrecondPairs, tens: Float64Array, udiag: number, rsd: Float64Array,
): Float64Array => {
  const z = new Float64Array(3 * a.n);
  for (let i = 0; i < a.n; i++) {
    const pol = udiag * Math.max(POLMIN, a.alpha[i]);
    for (let m = 0; m < 3; m++) z[3 * i + m] = pol * rsd[3 * i + m];
  }
  for (let q = 0; q < pc.pi.length; q++) {
    const i = pc.pi[q], j = pc.pj[q];
    const I = a.own[i], J = a.own[j];
    const m1 = tens[6 * q], m2 = tens[6 * q + 1], m3 = tens[6 * q + 2];
    const m4 = tens[6 * q + 3], m5 = tens[6 * q + 4], m6 = tens[6 * q + 5];
    const ri = [rsd[3 * I], rsd[3 * I + 1], rsd[3 * I + 2]];
    const rj = [rsd[3 * J], rsd[3 * J + 1], rsd[3 * J + 2]];
    z[3 * I] += m1 * rj[0] + m2 * rj[1] + m3 * rj[2];
    z[3 * I + 1] += m2 * rj[0] + m4 * rj[1] + m5 * rj[2];
    z[3 * I + 2] += m3 * rj[0] + m5 * rj[1] + m6 * rj[2];
    z[3 * J] += m1 * ri[0] + m2 * ri[1] + m3 * ri[2];
    z[3 * J + 1] += m2 * ri[0] + m4 * ri[1] + m5 * ri[2];
    z[3 * J + 2] += m3 * ri[0] + m5 * ri[1] + m6 * ri[2];
  }
  return z;
};

const dot = (u: Float64Array, v: Float64Array): number => {
  let s = 0;
  for (let k = 0; k < u.length; k++) s += u[k] * v[k];
  return s;
};

/**
 * induce() with poltyp MUTUAL (amoeba_induce.cpp, no Ewald, no predictor): the direct dipoles, the mutual field of the
 * CG initial guess, preconditioned conjugate gradient on the mutual dipoles, the convergence test
 * eps = DEBYE sqrt(max(epsd, epsp) / natoms) < polar-eps, the iteration limit polar-iter and the peek step.
 */
export const induceDipoles = (
  a: PolarAtoms, s: PolarSettings, pairs: readonly number[], pre: PrecondPairs, tens: Float64Array,
): InducedDipoles => {
  const n = a.n;
  const off2 = s.ewaldcut * s.ewaldcut;
  const direct = s.hippo ? hippoDirectField(a, requireHippo(a), s, pairs, off2) : directField(a, s, pairs, off2);
  const { tdip, inRange } = direct;
  const udir = new Float64Array(3 * n), udirp = new Float64Array(3 * n);
  const poli = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    poli[i] = Math.max(POLMIN, a.alpha[i]);
    for (let m = 0; m < 3; m++) {
      udir[3 * i + m] = a.alpha[i] * direct.field[3 * i + m];
      udirp[3 * i + m] = a.alpha[i] * direct.fieldp[3 * i + m];
    }
  }
  let uind = new Float64Array(3 * n), uinp = new Float64Array(3 * n);
  if (s.pcgguess) {
    uind = Float64Array.from(udir);
    uinp = Float64Array.from(udirp);
  }
  // residuals of the initial guess
  let mf = mutualField(a, s.aewald, s.recipMutual, pairs, tdip, inRange, uind, uinp);
  const rsd = new Float64Array(3 * n), rsdp = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let m = 0; m < 3; m++) {
      const k = 3 * i + m;
      if (s.pcgguess) {
        rsd[k] = (udir[k] - uind[k]) / poli[i] + mf.field[k];
        rsdp[k] = (udirp[k] - uinp[k]) / poli[i] + mf.fieldp[k];
      } else {
        rsd[k] = udir[k] / poli[i];
        rsdp[k] = udirp[k] / poli[i];
      }
    }
  }
  let zrsd: Float64Array = Float64Array.from(rsd);
  let zrsdp: Float64Array = Float64Array.from(rsdp);
  if (s.pcgprec) {
    zrsd = applyPrecond(a, pre, tens, s.udiag, rsd);
    zrsdp = applyPrecond(a, pre, tens, s.udiag, rsdp);
  }
  let conj = Float64Array.from(zrsd);
  let conjp = Float64Array.from(zrsdp);
  let done = false;
  let iter = 0;
  while (!done) {
    iter++;
    // field of the conjugate vector (ufield0c with uind = conj)
    mf = mutualField(a, s.aewald, s.recipMutual, pairs, tdip, inRange, conj, conjp);
    const vec = new Float64Array(3 * n), vecp = new Float64Array(3 * n);
    for (let k = 0; k < 3 * n; k++) {
      const pi = poli[Math.floor(k / 3)];
      vec[k] = conj[k] / pi - mf.field[k];
      vecp[k] = conjp[k] / pi - mf.fieldp[k];
    }
    let a1 = dot(conj, vec);
    let ap = dot(conjp, vecp);
    const sum = dot(rsd, zrsd);
    const sump = dot(rsdp, zrsdp);
    if (a1 !== 0.0) a1 = sum / a1;
    if (ap !== 0.0) ap = sump / ap;
    for (let k = 0; k < 3 * n; k++) {
      uind[k] = uind[k] + a1 * conj[k];
      uinp[k] = uinp[k] + ap * conjp[k];
      rsd[k] = rsd[k] - a1 * vec[k];
      rsdp[k] = rsdp[k] - ap * vecp[k];
    }
    if (s.pcgprec) {
      zrsd = applyPrecond(a, pre, tens, s.udiag, rsd);
      zrsdp = applyPrecond(a, pre, tens, s.udiag, rsdp);
    } else {
      zrsd = Float64Array.from(rsd);
      zrsdp = Float64Array.from(rsdp);
    }
    let b = dot(rsd, zrsd);
    let bp = dot(rsdp, zrsdp);
    if (sum !== 0.0) b /= sum;
    if (sump !== 0.0) bp /= sump;
    for (let k = 0; k < 3 * n; k++) {
      conj[k] = zrsd[k] + b * conj[k];
      conjp[k] = zrsdp[k] + bp * conjp[k];
    }
    const epsd = dot(rsd, rsd);
    const epsp = dot(rsdp, rsdp);
    const eps = DEBYE * Math.sqrt(Math.max(epsd, epsp) / n);
    if (eps < s.poleps) done = true;
    if (iter >= s.politer) done = true;
    if (done) {
      for (let i = 0; i < n; i++) {
        const term = s.pcgpeek * poli[i];
        for (let m = 0; m < 3; m++) {
          uind[3 * i + m] += term * rsd[3 * i + m];
          uinp[3 * i + m] += term * rsdp[3 * i + m];
        }
      }
    }
  }
  return { uind, uinp, udir, udirp, iterations: iter };
};

/** polar_energy: the polarization energy, -1/2 sum_i (uind_i . udirp_i) / alpha_i (in kcal/mol with the AMOEBA factor). */
export const polarEnergy = (a: PolarAtoms, s: PolarSettings, ind: InducedDipoles): number => {
  const felec = (-0.5 * ELECTRIC) / s.dielectric;
  let total = 0.0;
  for (let i = 0; i < a.n; i++) {
    const fi = felec / a.alpha[i];
    let e = 0.0;
    for (let m = 0; m < 3; m++) e += fi * ind.uind[3 * i + m] * ind.udirp[3 * i + m];
    total += e;
  }
  return total;
};

/** requireHippo: the HIPPO per-copy data of a pair hippo polarization term (set by pair_amoeba.ts). */
const requireHippo = (a: PolarAtoms): HippoPolarAtoms => {
  if (!a.hippo) throw new Error('pair hippo: the HIPPO per-atom data are missing');
  return a.hippo;
};

/**
 * The site torques of the induced fields and their conversion to forces (polar() and the tail of polar_real): tep from
 * ufld and dufld per owned site, the self torque (zero without Ewald), torque2force and the frame virial. Shared by the
 * AMOEBA and HIPPO polar_real.
 */
export const polarSiteTorques = (
  a: PolarAtoms, s: PolarSettings, sites: MpoleSites, ind: InducedDipoles,
  ufld: Float64Array, dufld: Float64Array, f: Float64Array, vir: Float64Array,
): void => {
  const n = a.n;
  const x = a.x, rp = a.rpole, uind = ind.uind, uinp = ind.uinp;
  for (let i = 0; i < n; i++) {
    const oi = 13 * i;
    const dix = rp[oi + 1], diy = rp[oi + 2], diz = rp[oi + 3];
    const qixx = rp[oi + 4], qixy = rp[oi + 5], qixz = rp[oi + 6], qiyy = rp[oi + 8], qiyz = rp[oi + 9], qizz = rp[oi + 12];
    const u = (k: number): number => ufld[3 * i + k];
    const d = (k: number): number => dufld[6 * i + k];
    const tep: Float64Array = Float64Array.from([
      diz * u(1) - diy * u(2) + qixz * d(1) - qixy * d(3) + 2.0 * qiyz * (d(2) - d(5)) + (qizz - qiyy) * d(4),
      dix * u(2) - diz * u(0) - qiyz * d(1) + qixy * d(4) + 2.0 * qixz * (d(5) - d(0)) + (qixx - qizz) * d(3),
      diy * u(0) - dix * u(1) + qiyz * d(3) - qixz * d(4) + 2.0 * qixy * (d(0) - d(2)) + (qiyy - qixx) * d(1),
    ]);
    // the self torque of the induced dipoles (amoeba_polar.cpp polar(): tep = term (d x u), u = (uind + uinp)/2,
    // term = (4/3) felec aewald^3 / sqrt(pi) with felec = electric / dielectric)
    const tterm = ((4.0 / 3.0) * (ELECTRIC / s.dielectric) * s.aewald * s.aewald * s.aewald) / MY_PIS;
    if (tterm !== 0.0) {
      const uix = 0.5 * (uind[3 * i] + uinp[3 * i]), uiy = 0.5 * (uind[3 * i + 1] + uinp[3 * i + 1]);
      const uiz = 0.5 * (uind[3 * i + 2] + uinp[3 * i + 2]);
      tep[0] += tterm * (diy * uiz - diz * uiy);
      tep[1] += tterm * (diz * uix - dix * uiz);
      tep[2] += tterm * (dix * uiy - diy * uix);
    }
    const { fx, fy, fz } = torque2force(sites, i, tep, x, f);
    if (sites.axis[i] === 'NOFRAME') continue;
    const iz = sites.zl[i], ix = sites.xl[i], iy = sites.yl[i];
    const xiz = x[3 * iz] - x[3 * i], yiz = x[3 * iz + 1] - x[3 * i + 1], ziz = x[3 * iz + 2] - x[3 * i + 2];
    const xix = x[3 * ix] - x[3 * i], yix = x[3 * ix + 1] - x[3 * i + 1], zix = x[3 * ix + 2] - x[3 * i + 2];
    const xiy = x[3 * iy] - x[3 * i], yiy = x[3 * iy + 1] - x[3 * i + 1], ziy = x[3 * iy + 2] - x[3 * i + 2];
    const vxx = xix * fx[0] + xiy * fy[0] + xiz * fz[0];
    const vyy = yix * fx[1] + yiy * fy[1] + yiz * fz[1];
    const vzz = zix * fx[2] + ziy * fy[2] + ziz * fz[2];
    const vxy = 0.5 * (yix * fx[0] + yiy * fy[0] + yiz * fz[0] + xix * fx[1] + xiy * fy[1] + xiz * fz[1]);
    const vxz = 0.5 * (zix * fx[0] + ziy * fy[0] + ziz * fz[0] + xix * fx[2] + xiy * fy[2] + xiz * fz[2]);
    const vyz = 0.5 * (zix * fx[1] + ziy * fy[1] + ziz * fz[1] + yix * fx[2] + yiy * fy[2] + yiz * fz[2]);
    vir[0] -= vxx;
    vir[1] -= vyy;
    vir[2] -= vzz;
    vir[3] -= vxy;
    vir[4] -= vxz;
    vir[5] -= vyz;
  }
};

/** Output of polarReal: the virial (xx, yy, zz, xy, xz, yz). Forces and torques are added to f. */
export interface PolarResult {
  virial: Float64Array;
}

/**
 * polar_real (AMOEBA branch, no Ewald; poltyp MUTUAL): the polarization forces between the permanent multipoles and the
 * induced dipoles over the pairs in range, the dipole fields for the torques and their gradients, and the torque-to-force
 * conversion of every site (amoeba_polar.cpp polar_real and the torque part of polar()).
 */
export const polarReal = (
  a: PolarAtoms, s: PolarSettings, sites: MpoleSites, ind: InducedDipoles, f: Float64Array,
  pairs: readonly number[],
): PolarResult => {
  if (s.hippo) return hippoPolarReal(a, requireHippo(a), s, sites, ind, f, pairs, polarSiteTorques);
  const n = a.n;
  const felec = (0.5 * ELECTRIC) / s.dielectric;
  const off2 = s.ewaldcut * s.ewaldcut;
  const x = a.x, rp = a.rpole, uind = ind.uind, uinp = ind.uinp;
  const ufld = new Float64Array(3 * n);
  const dufld = new Float64Array(6 * n);
  const vir = new Float64Array(6);
  for (const p of pairs) {
    const i = a.pi[p], j = a.pj[p];
    const I = a.own[i], J = a.own[j];
    const xr = x[3 * j] - x[3 * i], yr = x[3 * j + 1] - x[3 * i + 1], zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    const sb = a.ps[p];
    let pscale: number, dscale: number, uscale: number;
    if (a.group[I] === a.group[J]) {
      pscale = s.piscale[sb];
      dscale = s.dscale;
      uscale = s.uscale;
    } else {
      pscale = s.pscale[sb];
      dscale = 1.0;
      uscale = 1.0;
    }
    const r = Math.sqrt(r2);
    const oi = 13 * I, ok = 13 * J;
    const ci = rp[oi], dix = rp[oi + 1], diy = rp[oi + 2], diz = rp[oi + 3];
    const qixx = rp[oi + 4], qixy = rp[oi + 5], qixz = rp[oi + 6], qiyy = rp[oi + 8], qiyz = rp[oi + 9], qizz = rp[oi + 12];
    const ck = rp[ok], dkx = rp[ok + 1], dky = rp[ok + 2], dkz = rp[ok + 3];
    const qkxx = rp[ok + 4], qkxy = rp[ok + 5], qkxz = rp[ok + 6], qkyy = rp[ok + 8], qkyz = rp[ok + 9], qkzz = rp[ok + 12];
    const uix = uind[3 * I], uiy = uind[3 * I + 1], uiz = uind[3 * I + 2];
    const uixp = uinp[3 * I], uiyp = uinp[3 * I + 1], uizp = uinp[3 * I + 2];
    const ukx = uind[3 * J], uky = uind[3 * J + 1], ukz = uind[3 * J + 2];
    const ukxp = uinp[3 * J], ukyp = uinp[3 * J + 1], ukzp = uinp[3 * J + 2];
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
    const uirp = uixp * xr + uiyp * yr + uizp * zr;
    const ukr = ukx * xr + uky * yr + ukz * zr;
    const ukrp = ukxp * xr + ukyp * yr + ukzp * zr;
    // reciprocal distance terms (felec folded in after the bn recursion, as native)
    const rr1 = felec / r;
    const rr3 = rr1 / r2;
    const rr5 = (3.0 * rr3) / r2;
    const rr7 = (5.0 * rr5) / r2;
    const rr9 = (7.0 * rr7) / r2;
    // Ewald damping (amoeba_polar.cpp polar_real): bn[0] = erfc(aewald r)/r, the recursion of the multipole term
    const ralpha = s.aewald * r;
    const bn = [libmErfc(ralpha) / r, 0, 0, 0, 0];
    const alsq2 = 2.0 * s.aewald * s.aewald;
    let alsq2n = s.aewald > 0.0 ? 1.0 / (MY_PIS * s.aewald) : 0.0;
    const exp2a = libmExp(-ralpha * ralpha);
    for (let m = 1; m <= 4; m++) {
      alsq2n = alsq2 * alsq2n;
      bn[m] = ((2 * m - 1) * bn[m - 1] + alsq2n * exp2a) / r2;
    }
    for (let m = 0; m < 5; m++) bn[m] *= felec;
    // Thole damping (AMOEBA branch): scale factors and the derivative terms rc3, rc5, rc7
    let sc3 = 1.0, sc5 = 1.0, sc7 = 1.0;
    const rc3 = [0, 0, 0], rc5 = [0, 0, 0], rc7 = [0, 0, 0];
    let psc3 = 0, psc5 = 0, psc7 = 0, dsc3 = 0, dsc5 = 0, dsc7 = 0, usc3 = 0, usc5 = 0;
    let psr3 = 0, psr5 = 0, psr7 = 0, dsr3 = 0, dsr5 = 0, dsr7 = 0, usr5 = 0;
    const prc3 = [0, 0, 0], prc5 = [0, 0, 0], prc7 = [0, 0, 0];
    const drc3 = [0, 0, 0], drc5 = [0, 0, 0], drc7 = [0, 0, 0];
    const urc3 = [0, 0, 0], urc5 = [0, 0, 0];
    const damp = tholeDamp(a, I, J, r);
    if (a.pdamp[I] * a.pdamp[J] !== 0) {
      if (damp < 50.0) {
        const expdamp = libmExp(-damp);
        sc3 = 1.0 - expdamp;
        sc5 = 1.0 - (1.0 + damp) * expdamp;
        sc7 = 1.0 - (1.0 + damp + 0.6 * damp * damp) * expdamp;
        const temp3 = (3.0 * damp * expdamp) / r2;
        const temp5 = damp;
        const temp7 = -0.2 + 0.6 * damp;
        for (let k = 0; k < 3; k++) {
          const dk = [xr, yr, zr][k];
          rc3[k] = dk * temp3;
          rc5[k] = rc3[k] * temp5;
          rc7[k] = rc5[k] * temp7;
        }
      }
      psc3 = 1.0 - sc3 * pscale;
      psc5 = 1.0 - sc5 * pscale;
      psc7 = 1.0 - sc7 * pscale;
      dsc3 = 1.0 - sc3 * dscale;
      dsc5 = 1.0 - sc5 * dscale;
      dsc7 = 1.0 - sc7 * dscale;
      usc3 = 1.0 - sc3 * uscale;
      usc5 = 1.0 - sc5 * uscale;
      psr3 = bn[1] - psc3 * rr3;
      psr5 = bn[2] - psc5 * rr5;
      psr7 = bn[3] - psc7 * rr7;
      dsr3 = bn[1] - dsc3 * rr3;
      dsr5 = bn[2] - dsc5 * rr5;
      dsr7 = bn[3] - dsc7 * rr7;
      usr5 = bn[2] - usc5 * rr5;
      for (let k = 0; k < 3; k++) {
        prc3[k] = rc3[k] * pscale;
        prc5[k] = rc5[k] * pscale;
        prc7[k] = rc7[k] * pscale;
        drc3[k] = rc3[k] * dscale;
        drc5[k] = rc5[k] * dscale;
        drc7[k] = rc7[k] * dscale;
        urc3[k] = rc3[k] * uscale;
        urc5[k] = rc5[k] * uscale;
      }
    }
    // induced dipole field for the dipole torques
    const tix3 = psr3 * ukx + dsr3 * ukxp, tiy3 = psr3 * uky + dsr3 * ukyp, tiz3 = psr3 * ukz + dsr3 * ukzp;
    const tkx3 = psr3 * uix + dsr3 * uixp, tky3 = psr3 * uiy + dsr3 * uiyp, tkz3 = psr3 * uiz + dsr3 * uizp;
    let tuir = -psr5 * ukr - dsr5 * ukrp;
    let tukr = -psr5 * uir - dsr5 * uirp;
    ufld[3 * I] += tix3 + xr * tuir;
    ufld[3 * I + 1] += tiy3 + yr * tuir;
    ufld[3 * I + 2] += tiz3 + zr * tuir;
    ufld[3 * J] += tkx3 + xr * tukr;
    ufld[3 * J + 1] += tky3 + yr * tukr;
    ufld[3 * J + 2] += tkz3 + zr * tukr;
    // gradient of the induced field for the quadrupole torques
    const tix5 = 2.0 * (psr5 * ukx + dsr5 * ukxp), tiy5 = 2.0 * (psr5 * uky + dsr5 * ukyp), tiz5 = 2.0 * (psr5 * ukz + dsr5 * ukzp);
    const tkx5 = 2.0 * (psr5 * uix + dsr5 * uixp), tky5 = 2.0 * (psr5 * uiy + dsr5 * uiyp), tkz5 = 2.0 * (psr5 * uiz + dsr5 * uizp);
    tuir = -psr7 * ukr - dsr7 * ukrp;
    tukr = -psr7 * uir - dsr7 * uirp;
    const di = 6 * I, dj = 6 * J;
    dufld[di] += xr * tix5 + xr * xr * tuir;
    dufld[di + 1] += xr * tiy5 + yr * tix5 + 2.0 * xr * yr * tuir;
    dufld[di + 2] += yr * tiy5 + yr * yr * tuir;
    dufld[di + 3] += xr * tiz5 + zr * tix5 + 2.0 * xr * zr * tuir;
    dufld[di + 4] += yr * tiz5 + zr * tiy5 + 2.0 * yr * zr * tuir;
    dufld[di + 5] += zr * tiz5 + zr * zr * tuir;
    dufld[dj] -= xr * tkx5 + xr * xr * tukr;
    dufld[dj + 1] -= xr * tky5 + yr * tkx5 + 2.0 * xr * yr * tukr;
    dufld[dj + 2] -= yr * tky5 + yr * yr * tukr;
    dufld[dj + 3] -= xr * tkz5 + zr * tkx5 + 2.0 * xr * zr * tukr;
    dufld[dj + 4] -= yr * tkz5 + zr * tky5 + 2.0 * yr * zr * tukr;
    dufld[dj + 5] -= zr * tkz5 + zr * zr * tukr;
    // force from the direct (dscale) terms
    const dsForce = polarForceTerms(
      { ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, ck, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz },
      { xr, yr, zr, rr3, rr5, rr7, rr9, bn, dir, dkr, qir, qkr, qix, qiy, qiz, qkx, qky, qkz },
      { s3: dsc3, s5: dsc5, s7: dsc7, r3: dsr3, r5: dsr5, r7: dsr7, rc3: drc3, rc5: drc5, rc7: drc7 },
      { ux: ukxp, uy: ukyp, uz: ukzp, vx: uixp, vy: uiyp, vz: uizp },
    );
    // force from the p (pscale) terms
    const pForce = polarForceTerms(
      { ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, ck, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz },
      { xr, yr, zr, rr3, rr5, rr7, rr9, bn, dir, dkr, qir, qkr, qix, qiy, qiz, qkx, qky, qkz },
      { s3: psc3, s5: psc5, s7: psc7, r3: psr3, r5: psr5, r7: psr7, rc3: prc3, rc5: prc5, rc7: prc7 },
      { ux: ukx, uy: uky, uz: ukz, vx: uix, vy: uiy, vz: uiz },
    );
    let frcx = dsForce.frc[0] + pForce.frc[0];
    let frcy = dsForce.frc[1] + pForce.frc[1];
    let frcz = dsForce.frc[2] + pForce.frc[2];
    // mutual polarization force (dtau/dr with uscale)
    const mut = mutualTerms(
      { uix, uiy, uiz, ukx, uky, ukz, uir, ukr, uixp, uiyp, uizp, ukxp, ukyp, ukzp },
      { xr, yr, zr, rr3, rr5, rr7, bn, usc3, usc5, usr5, urc3, urc5, rc3, uscale },
    );
    frcx += mut[0];
    frcy += mut[1];
    frcz += mut[2];
    f[3 * i] += frcx;
    f[3 * i + 1] += frcy;
    f[3 * i + 2] += frcz;
    f[3 * j] -= frcx;
    f[3 * j + 1] -= frcy;
    f[3 * j + 2] -= frcz;
    const vxx = xr * frcx;
    const vxy = 0.5 * (yr * frcx + xr * frcy);
    const vxz = 0.5 * (zr * frcx + xr * frcz);
    const vyy = yr * frcy;
    const vyz = 0.5 * (zr * frcy + yr * frcz);
    const vzz = zr * frcz;
    vir[0] -= vxx;
    vir[1] -= vyy;
    vir[2] -= vzz;
    vir[3] -= vxy;
    vir[4] -= vxz;
    vir[5] -= vyz;
  }
  polarSiteTorques(a, s, sites, ind, ufld, dufld, f, vir);
  return { virial: vir };
};

/**
 * The tixx..tkyz / depx block of polar_real for one scaling set (the dsc or the psc block of the AMOEBA branch). The
 * caller passes the scaled damping factors (s3..s7 are 1 - sc*scale), the scaled reduced distances (r3..r7) and the
 * derivative terms rc3..rc7 (already scaled). `u` holds the field dipoles of j (ux..uz) and i (vx..vz) of the block.
 */
const polarForceTerms = (
  m: { ci: number; dix: number; diy: number; diz: number; qixx: number; qixy: number; qixz: number; qiyy: number; qiyz: number; qizz: number;
    ck: number; dkx: number; dky: number; dkz: number; qkxx: number; qkxy: number; qkxz: number; qkyy: number; qkyz: number; qkzz: number },
  g: { xr: number; yr: number; zr: number; rr3: number; rr5: number; rr7: number; rr9: number; bn: number[]; dir: number; dkr: number;
    qir: number; qkr: number; qix: number; qiy: number; qiz: number; qkx: number; qky: number; qkz: number },
  t: { s3: number; s5: number; s7: number; r3: number; r5: number; r7: number; rc3: number[]; rc5: number[]; rc7: number[] },
  u: { ux: number; uy: number; uz: number; vx: number; vy: number; vz: number },
): { frc: [number, number, number] } => {
  const { ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, ck, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz } = m;
  const { xr, yr, zr, rr3, rr5, rr7, rr9, bn, dir, dkr, qir, qkr, qix, qiy, qiz, qkx, qky, qkz } = g;
  const { s3, s5, s7, r3: sr3, r5: sr5, r7: sr7, rc3, rc5, rc7 } = t;
  const { ux, uy, uz, vx, vy, vz } = u;
  // term1 = bn[2] - s3 rr5; term2 = bn[3] - s5 rr7 (the scaled 3 and 5 damping)
  const term1 = bn[2] - s3 * rr5;
  const term2 = bn[3] - s5 * rr7;
  const term1x = term1, term2x = term2;
  const ukx = ux, uky = uy, ukz = uz, uix = vx, uiy = vy, uiz = vz;
  // xx
  let t3 = -sr3 + term1x * xr * xr - rr3 * xr * rc3[0];
  let t4 = rr3 * rc3[0] - term1x * xr - sr5 * xr;
  let t5 = term2x * xr * xr - sr5 - rr5 * xr * rc5[0];
  let t6 = (bn[4] - s7 * rr9) * xr * xr - bn[3] - rr7 * xr * rc7[0];
  let t7 = rr5 * rc5[0] - 2.0 * bn[3] * xr + (s5 + 1.5 * s7) * rr7 * xr;
  const txx_i = ci * t3 + dix * t4 + dir * t5 + 2.0 * sr5 * qixx + (qiy * yr + qiz * zr) * s7 * rr7 + 2.0 * qix * t7 + qir * t6;
  const txx_k = ck * t3 - dkx * t4 - dkr * t5 + 2.0 * sr5 * qkxx + (qky * yr + qkz * zr) * s7 * rr7 + 2.0 * qkx * t7 + qkr * t6;
  // yy
  t3 = -sr3 + term1x * yr * yr - rr3 * yr * rc3[1];
  t4 = rr3 * rc3[1] - term1x * yr - sr5 * yr;
  t5 = term2x * yr * yr - sr5 - rr5 * yr * rc5[1];
  t6 = (bn[4] - s7 * rr9) * yr * yr - bn[3] - rr7 * yr * rc7[1];
  t7 = rr5 * rc5[1] - 2.0 * bn[3] * yr + (s5 + 1.5 * s7) * rr7 * yr;
  const tyy_i = ci * t3 + diy * t4 + dir * t5 + 2.0 * sr5 * qiyy + (qix * xr + qiz * zr) * s7 * rr7 + 2.0 * qiy * t7 + qir * t6;
  const tyy_k = ck * t3 - dky * t4 - dkr * t5 + 2.0 * sr5 * qkyy + (qkx * xr + qkz * zr) * s7 * rr7 + 2.0 * qky * t7 + qkr * t6;
  // zz
  t3 = -sr3 + term1x * zr * zr - rr3 * zr * rc3[2];
  t4 = rr3 * rc3[2] - term1x * zr - sr5 * zr;
  t5 = term2x * zr * zr - sr5 - rr5 * zr * rc5[2];
  t6 = (bn[4] - s7 * rr9) * zr * zr - bn[3] - rr7 * zr * rc7[2];
  t7 = rr5 * rc5[2] - 2.0 * bn[3] * zr + (s5 + 1.5 * s7) * rr7 * zr;
  const tzz_i = ci * t3 + diz * t4 + dir * t5 + 2.0 * sr5 * qizz + (qix * xr + qiy * yr) * s7 * rr7 + 2.0 * qiz * t7 + qir * t6;
  const tzz_k = ck * t3 - dkz * t4 - dkr * t5 + 2.0 * sr5 * qkzz + (qkx * xr + qky * yr) * s7 * rr7 + 2.0 * qkz * t7 + qkr * t6;
  // xy
  t3 = term1x * xr * yr - rr3 * yr * rc3[0];
  t4 = rr3 * rc3[0] - term1x * xr;
  let t5b = term2x * xr * yr - rr5 * yr * rc5[0];
  let t6b = (bn[4] - s7 * rr9) * xr * yr - rr7 * yr * rc7[0];
  let t7b = rr5 * rc5[0] - term2x * xr;
  const txy_i = ci * t3 - sr5 * dix * yr + diy * t4 + dir * t5b + 2.0 * sr5 * qixy - 2.0 * sr7 * yr * qix + 2.0 * qiy * t7b + qir * t6b;
  const txy_k = ck * t3 + sr5 * dkx * yr - dky * t4 - dkr * t5b + 2.0 * sr5 * qkxy - 2.0 * sr7 * yr * qkx + 2.0 * qky * t7b + qkr * t6b;
  // xz (t4, t7 carried from xy, as native)
  t3 = term1x * xr * zr - rr3 * zr * rc3[0];
  t5b = term2x * xr * zr - rr5 * zr * rc5[0];
  t6b = (bn[4] - s7 * rr9) * xr * zr - rr7 * zr * rc7[0];
  const txz_i = ci * t3 - sr5 * dix * zr + diz * t4 + dir * t5b + 2.0 * sr5 * qixz - 2.0 * sr7 * zr * qix + 2.0 * qiz * t7b + qir * t6b;
  const txz_k = ck * t3 + sr5 * dkx * zr - dkz * t4 - dkr * t5b + 2.0 * sr5 * qkxz - 2.0 * sr7 * zr * qkx + 2.0 * qkz * t7b + qkr * t6b;
  // yz
  t3 = term1x * yr * zr - rr3 * zr * rc3[1];
  t4 = rr3 * rc3[1] - term1x * yr;
  t5b = term2x * yr * zr - rr5 * zr * rc5[1];
  t6b = (bn[4] - s7 * rr9) * yr * zr - rr7 * zr * rc7[1];
  t7b = rr5 * rc5[1] - term2x * yr;
  const tyz_i = ci * t3 - sr5 * diy * zr + diz * t4 + dir * t5b + 2.0 * sr5 * qiyz - 2.0 * sr7 * zr * qiy + 2.0 * qiz * t7b + qir * t6b;
  const tyz_k = ck * t3 + sr5 * dky * zr - dkz * t4 - dkr * t5b + 2.0 * sr5 * qkyz - 2.0 * sr7 * zr * qky + 2.0 * qkz * t7b + qkr * t6b;
  // the mixed dependence of the energy: depx..depz use u (on j) and v (on i) in the block's convention
  const depx = txx_i * ukx + txy_i * uky + (txz_i) * ukz - txx_k * uix - txy_k * uiy - txz_k * uiz;
  const depy = txy_i * ukx + tyy_i * uky + tyz_i * ukz - txy_k * uix - tyy_k * uiy - tyz_k * uiz;
  const depz = txz_i * ukx + tyz_i * uky + tzz_i * ukz - txz_k * uix - tyz_k * uiy - tzz_k * uiz;
  return { frc: [depx, depy, depz] };
};

/**
 * The mutual (uscale) dtau/dr term of polar_real (AMOEBA branch, poltyp MUTUAL): the force from the mutual scaling.
 */
const mutualTerms = (
  m: { uix: number; uiy: number; uiz: number; ukx: number; uky: number; ukz: number; uir: number; ukr: number;
    uixp: number; uiyp: number; uizp: number; ukxp: number; ukyp: number; ukzp: number },
  g: { xr: number; yr: number; zr: number; rr3: number; rr5: number; rr7: number; bn: number[]; usc3: number; usc5: number; usr5: number;
    urc3: number[]; urc5: number[]; rc3: number[]; uscale: number },
): [number, number, number] => {
  const { uix, uiy, uiz, ukx, uky, ukz, uir, ukr, uixp, uiyp, uizp, ukxp, ukyp, ukzp } = m;
  const { xr, yr, zr, rr3, rr5, rr7, bn, usc3, usc5, usr5, urc3, urc5, rc3, uscale } = g;
  const term1 = bn[2] - usc3 * rr5;
  const term2 = bn[3] - usc5 * rr7;
  const term3 = usr5 + term1;
  const term4a = rr3 * uscale;
  let term5 = -xr * term3 + rc3[0] * term4a;
  let term6 = -usr5 + xr * xr * term2 - rr5 * xr * urc5[0];
  const tixx = uix * term5 + uir * term6;
  const tkxx = ukx * term5 + ukr * term6;
  term5 = -yr * term3 + rc3[1] * term4a;
  term6 = -usr5 + yr * yr * term2 - rr5 * yr * urc5[1];
  const tiyy = uiy * term5 + uir * term6;
  const tkyy = uky * term5 + ukr * term6;
  term5 = -zr * term3 + rc3[2] * term4a;
  term6 = -usr5 + zr * zr * term2 - rr5 * zr * urc5[2];
  const tizz = uiz * term5 + uir * term6;
  const tkzz = ukz * term5 + ukr * term6;
  let term4 = -usr5 * yr;
  term5 = -xr * term1 + rr3 * urc3[0];
  term6 = xr * yr * term2 - rr5 * yr * urc5[0];
  const tixy = uix * term4 + uiy * term5 + uir * term6;
  const tkxy = ukx * term4 + uky * term5 + ukr * term6;
  term4 = -usr5 * zr;
  term6 = xr * zr * term2 - rr5 * zr * urc5[0];
  const tixz = uix * term4 + uiz * term5 + uir * term6;
  const tkxz = ukx * term4 + ukz * term5 + ukr * term6;
  term5 = -yr * term1 + rr3 * urc3[1];
  term6 = yr * zr * term2 - rr5 * zr * urc5[1];
  const tiyz = uiy * term4 + uiz * term5 + uir * term6;
  const tkyz = uky * term4 + ukz * term5 + ukr * term6;
  const depx = tixx * ukxp + tixy * ukyp + tixz * ukzp + tkxx * uixp + tkxy * uiyp + tkxz * uizp;
  const depy = tixy * ukxp + tiyy * ukyp + tiyz * ukzp + tkxy * uixp + tkyy * uiyp + tkyz * uizp;
  const depz = tixz * ukxp + tiyz * ukyp + tizz * ukzp + tkxz * uixp + tkyz * uiyp + tkzz * uizp;
  return [depx, depy, depz];
};
