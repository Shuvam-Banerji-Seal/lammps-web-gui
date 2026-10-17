/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/amoeba_utils.cpp (kmpole, chkpole, rotmat, rotsite, torque2force: the ZTHENX and BISECTOR
 * branches), src/AMOEBA/amoeba_multipole.cpp (multipole_real, the AMOEBA branch without Ewald) and src/AMOEBA/
 * amoeba_file.cpp (file_multipole) at commit b332f28. AMOEBA package authors: the LAMMPS developers (Sandia
 * Corporation, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Permanent atomic multipoles of pair amoeba (stage 2b, direct sum, no Ewald): the local frame of each site from its
 * 1-2 and 1-3 neighbours (kmpole), the sign inversion at chiral sites (chkpole), the rotation of the local moments to
 * the global frame (rotmat, rotsite), the multipole energy, forces and torques between pairs (multipole_real), and the
 * conversion of the site torques to forces on the site and its frame atoms (torque2force).
 * Frame types ZONLY, ZBISECT and THREEFOLD are not ported in torque2force: a site with one of them is refused by name.
 * Literature (amoeba_utils.cpp): P. L. Popelier and A. J. Stone, Mol. Phys. 82, 411 (1994); C. Segui, L. G. Pedersen
 * and T. A. Darden, J. Chem. Phys. 120, 73 (2004).
 */
import { StyleError } from '../../engine/force/types';
import type { AxisKind, MultipoleFrame } from './params';
import { exp as libmExp } from '../libm/exp';
import { erfc as libmErfc } from '../libm/erf';

/** Coulomb constant of AMOEBA (pair_amoeba.cpp: electric = 332.063713). */
export const ELECTRIC = 332.063713;
/** sqrt(pi) (amoeba_multipole.cpp MY_PIS). */
const MY_PIS = Math.sqrt(Math.PI);

/** Topology queries kmpole needs; atom IDs, not local indices. */
export interface MpoleTopo {
  /** 1-2 partner IDs of an atom, sorted ascending (amoeba_utils kmpole sorts the special list the same way). */
  nb12(id: number): number[];
  /** 1-3 partner IDs of an atom, sorted ascending. */
  nb13(id: number): number[];
  /** True when atoms a and b are bonded. */
  bonded(a: number, b: number): boolean;
}

/** Per-site frame data of one step (kmpole output, chkpole sign inversion applied). */
export interface MpoleSites {
  n: number;
  axis: AxisKind[];
  /** Axis atom IDs (0 when the axis is not used): z, x and y (kmpole: xyzaxis). */
  zid: Float64Array;
  xid: Float64Array;
  yid: Float64Array;
  /** Local indices of the axis atoms (the site itself when unused): zaxis2local etc. */
  zl: Int32Array;
  xl: Int32Array;
  yl: Int32Array;
  /** Multipole values in the local frame, 13 per site (pole). */
  pole: Float64Array;
  /** Rotated moments in the global frame, 13 per site (rpole). */
  rpole: Float64Array;
  /** Rotation matrix of each site, 9 per site, row-major (rotate). */
  rotate: Float64Array;
}

/**
 * kmpole: assigns each owned atom its frame from the first matching neighbour pattern (amoeba_utils.cpp kmpole, in its
 * four passes: 1-2 with two partners, 1-2 and 1-3, one z-defining partner, no partner). `typeOf` gives the atom type
 * (amtype) of an atom ID.
 */
export const kmpole = (
  frames: (type: number) => MultipoleFrame[],
  ids: Int32Array,
  amtype: Int32Array,
  typeOfId: (id: number) => number,
  topo: MpoleTopo,
): MpoleSites => {
  const n = ids.length;
  const axis: AxisKind[] = new Array<AxisKind>(n).fill('NOFRAME');
  const zid = new Float64Array(n), xid = new Float64Array(n), yid = new Float64Array(n);
  const pole = new Float64Array(13 * n);
  let nmissing = 0;
  for (let i = 0; i < n; i++) {
    const fr = frames(amtype[i]);
    const b12 = topo.nb12(ids[i]);
    const b13 = topo.nb13(ids[i]);
    let flag = false;
    const set = (f: MultipoleFrame, z: number, x: number, y: number): void => {
      flag = true;
      zid[i] = z;
      xid[i] = x;
      yid[i] = y;
      axis[i] = f.axis;
      pole.set(f.pole, 13 * i);
    };
    // pass 1: z and x among the 1-2 partners, y among the 1-2 partners (only first match is used)
    for (const f of fr) {
      for (const j of b12) {
        if (typeOfId(j) !== f.z) continue;
        for (const k of b12) {
          if (k === j) continue;
          if (typeOfId(k) !== f.x) continue;
          if (f.y === 0 && !flag) set(f, j, k, 0);
          else {
            for (const m of b12) {
              if (m === j || m === k) continue;
              if (typeOfId(m) === f.y && !flag) set(f, j, k, m);
            }
          }
        }
      }
    }
    // pass 2: x and y among the 1-3 partners that are bonded to z
    if (!flag) {
      for (const f of fr) {
        for (const j of b12) {
          if (typeOfId(j) !== f.z) continue;
          for (const k of b13) {
            if (!topo.bonded(k, j)) continue;
            if (typeOfId(k) !== f.x) continue;
            if (f.y === 0 && !flag) set(f, j, k, 0);
            else {
              for (const m of b13) {
                if (m === k) continue;
                if (!topo.bonded(m, j)) continue;
                if (typeOfId(m) === f.y && !flag) set(f, j, k, m);
              }
            }
          }
        }
      }
    }
    // pass 3: only a z-defining partner
    if (!flag) {
      for (const f of fr) {
        for (const j of b12) {
          if (typeOfId(j) === f.z && f.x === 0 && !flag) set(f, j, 0, 0);
        }
      }
    }
    // pass 4: no connected atoms
    if (!flag) {
      for (const f of fr) if (f.z === 0 && !flag) set(f, 0, 0, 0);
    }
    if (!flag) nmissing++;
  }
  if (nmissing) throw new StyleError(`Pair amoeba: ${nmissing} multipole settings missing`);
  return { n, axis, zid, xid, yid, zl: new Int32Array(n), xl: new Int32Array(n), yl: new Int32Array(n), pole, rpole: new Float64Array(13 * n), rotate: new Float64Array(9 * n) };
};

/**
 * find_multipole_neighbors: local indices of the axis atoms (the site itself when the axis is unused). `localOf` maps
 * an atom ID to a local index; a missing axis atom is an error (amoeba_utils.cpp: "multipole neighbors missing").
 */
export const mapAxes = (sites: MpoleSites, localOf: (site: number, id: number) => number): void => {
  let nmissing = 0;
  for (let i = 0; i < sites.n; i++) {
    const pick = (id: number): number => {
      if (!id) return i;
      const li = localOf(i, id);
      if (li < 0) nmissing++;
      return li < 0 ? i : li;
    };
    sites.zl[i] = pick(sites.zid[i]);
    sites.xl[i] = pick(sites.xid[i]);
    sites.yl[i] = pick(sites.yid[i]);
  }
  if (nmissing) throw new StyleError(`Pair amoeba: ${nmissing} multipole neighbors missing`);
};

/**
 * chkpole: inverts the moments that involve the y-axis at a chiral ZTHENX site when the signed volume of the frame is
 * negative (amoeba_utils.cpp chkpole: pole[2], [5], [7], [9], [11] change sign). kmpole gives a fresh pole every call,
 * so the sign test here is the one native applies to the positive y-axis ID of the first step.
 */
export const chkpole = (sites: MpoleSites, x: Float64Array): void => {
  for (let i = 0; i < sites.n; i++) {
    if (sites.axis[i] !== 'ZTHENX' || sites.yid[i] === 0) continue;
    const ib = sites.zl[i], ic = sites.xl[i], id = sites.yl[i];
    const xad = x[3 * i] - x[3 * id], yad = x[3 * i + 1] - x[3 * id + 1], zad = x[3 * i + 2] - x[3 * id + 2];
    const xbd = x[3 * ib] - x[3 * id], ybd = x[3 * ib + 1] - x[3 * id + 1], zbd = x[3 * ib + 2] - x[3 * id + 2];
    const xcd = x[3 * ic] - x[3 * id], ycd = x[3 * ic + 1] - x[3 * id + 1], zcd = x[3 * ic + 2] - x[3 * id + 2];
    const c1 = ybd * zcd - zbd * ycd;
    const c2 = ycd * zad - zcd * yad;
    const c3 = yad * zbd - zad * ybd;
    const vol = xad * c1 + xbd * c2 + xcd * c3;
    if (vol < 0.0) {
      const p = sites.pole;
      const o = 13 * i;
      p[o + 2] = -p[o + 2];
      p[o + 5] = -p[o + 5];
      p[o + 7] = -p[o + 7];
      p[o + 9] = -p[o + 9];
      p[o + 11] = -p[o + 11];
    }
  }
};

/** rotmat: the rotation matrix from the local frame to the global frame at site i (amoeba_utils.cpp rotmat). */
export const rotmat = (sites: MpoleSites, x: Float64Array, i: number): void => {
  const R = sites.rotate;
  const o = 9 * i;
  const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
  const iz = sites.zl[i], ix = sites.xl[i], iy = sites.yl[i];
  // rows: R[0] = local x, R[1] = local y, R[2] = local z (global components)
  const r = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const ax = (k: number, dx: number, dy: number, dz: number): void => {
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    r[3 * k] = dx / len;
    r[3 * k + 1] = dy / len;
    r[3 * k + 2] = dz / len;
  };
  const vec = (k: number): [number, number, number] => [x[3 * k] - xi, x[3 * k + 1] - yi, x[3 * k + 2] - zi];
  const unit = (v: [number, number, number]): [number, number, number] => {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / len, v[1] / len, v[2] / len];
  };
  const ortho = (v: [number, number, number], dir: [number, number, number]): [number, number, number] => {
    const dot = v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2];
    return [v[0] - dot * dir[0], v[1] - dot * dir[1], v[2] - dot * dir[2]];
  };
  const kind = sites.axis[i];
  if (kind === 'ZONLY') {
    const zdir = unit(vec(iz));
    ax(2, zdir[0], zdir[1], zdir[2]);
    // the reference axis is x, or y when the z-axis is close to x (amoeba_utils.cpp rotmat, ZONLY)
    const v: [number, number, number] = Math.abs(r[6]) > 0.866 ? [0, 1, 0] : [1, 0, 0];
    const u = unit(ortho(v, [r[6], r[7], r[8]]));
    ax(0, u[0], u[1], u[2]);
  } else if (kind === 'ZTHENX') {
    const zdir = unit(vec(iz));
    ax(2, zdir[0], zdir[1], zdir[2]);
    const ox = ortho(vec(ix), [r[6], r[7], r[8]]);
    const u = unit(ox);
    ax(0, u[0], u[1], u[2]);
  } else if (kind === 'BISECTOR') {
    const d1 = unit(vec(iz));
    const d2 = unit(vec(ix));
    const sum = unit([d1[0] + d2[0], d1[1] + d2[1], d1[2] + d2[2]]);
    ax(2, sum[0], sum[1], sum[2]);
    const ox = ortho(d2, [r[6], r[7], r[8]]);
    const u = unit(ox);
    ax(0, u[0], u[1], u[2]);
  } else if (kind === 'ZBISECT') {
    const zdir = unit(vec(iz));
    ax(2, zdir[0], zdir[1], zdir[2]);
    const d1 = unit(vec(ix));
    const d2 = unit(vec(iy));
    const sum = unit([d1[0] + d2[0], d1[1] + d2[1], d1[2] + d2[2]]);
    const ox = ortho(sum, [r[6], r[7], r[8]]);
    const u = unit(ox);
    ax(0, u[0], u[1], u[2]);
  } else if (kind === 'THREEFOLD') {
    const d1 = unit(vec(iz));
    const d2 = unit(vec(ix));
    const d3 = unit(vec(iy));
    const sum = unit([d1[0] + d2[0] + d3[0], d1[1] + d2[1] + d3[1], d1[2] + d2[2] + d3[2]]);
    ax(2, sum[0], sum[1], sum[2]);
    const ox = ortho(d2, [r[6], r[7], r[8]]);
    const u = unit(ox);
    ax(0, u[0], u[1], u[2]);
  }
  // y-axis: R[1] = R[0] x R[2] as amoeba_utils.cpp writes it, with the components of the rows
  r[3] = r[2] * r[7] - r[1] * r[8];
  r[4] = r[0] * r[8] - r[2] * r[6];
  r[5] = r[1] * r[6] - r[0] * r[7];
  for (let k = 0; k < 9; k++) R[o + k] = r[k];
};

/** rotsite: rotates the local 13 moments of site i into rpole (amoeba_utils.cpp rotsite). */
export const rotsite = (sites: MpoleSites, i: number): void => {
  const p = sites.pole, rp = sites.rpole, R = sites.rotate;
  const o = 13 * i, ro = 9 * i;
  const rot = (k: number, c: number): number => R[ro + 3 * k + c];
  // monopole unchanged
  rp[o] = p[o];
  for (let a = 1; a <= 3; a++) {
    let s = 0;
    for (let b = 1; b <= 3; b++) s += p[o + b] * rot(b - 1, a - 1);
    rp[o + a] = s;
  }
  // quadrupole: mp[j][i] = pole[4 + 3 i + j]; rp[j][i] = sum R[k][i] R[m][j] mp[m][k] for j >= i, mirrored below
  const mp = (j: number, ii: number): number => p[o + 4 + 3 * ii + j];
  const q = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let ii = 0; ii < 3; ii++) {
    for (let j = 0; j < 3; j++) {
      if (j < ii) q[j][ii] = q[ii][j];
      else {
        let s = 0;
        for (let k = 0; k < 3; k++) for (let m = 0; m < 3; m++) s += rot(k, ii) * rot(m, j) * mp(m, k);
        q[j][ii] = s;
      }
    }
  }
  let k = 4;
  for (let ii = 0; ii < 3; ii++) for (let j = 0; j < 3; j++) rp[o + k++] = q[j][ii];
};

/**
 * torque2force: converts the torque on site i into forces on the site and its frame atoms (amoeba_utils.cpp
 * torque2force, the ZTHENX and BISECTOR branches; NOFRAME has no torque conversion). The forces are added to f; fx,
 * fy, fz are the frame-atom forces used by the virial. Other frame types are refused.
 */
export const torque2force = (
  sites: MpoleSites, i: number, trq: Float64Array, x: Float64Array, f: Float64Array,
): { fx: [number, number, number]; fy: [number, number, number]; fz: [number, number, number] } => {
  const fx: [number, number, number] = [0, 0, 0], fy: [number, number, number] = [0, 0, 0], fz: [number, number, number] = [0, 0, 0];
  const kind = sites.axis[i];
  if (kind === 'NOFRAME') return { fx, fy, fz };
  if (kind !== 'ZTHENX' && kind !== 'BISECTOR') {
    throw new StyleError(`pair amoeba: the torque of a ${kind} multipole frame is not ported yet (ZTHENX and BISECTOR are)`);
  }
  const ia = sites.zl[i], ib = i, ic = sites.xl[i];
  const xyz = (k: number): [number, number, number] => [x[3 * k], x[3 * k + 1], x[3 * k + 2]];
  const diff = (a: number, b: number): [number, number, number] => {
    const pa = xyz(a), pb = xyz(b);
    return [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
  };
  const siz = (v: [number, number, number]): number => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const cross = (a: number[], b: number[]): [number, number, number] => [
    a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
  ];
  const uRaw = diff(ia, ib), vRaw = diff(ic, ib);
  const usiz = siz(uRaw), vsiz = siz(vRaw);
  // w is the cross product of the raw u and v (before they are normalized)
  const wRaw = cross(uRaw, vRaw);
  const wsiz = siz(wRaw);
  const u = [uRaw[0] / usiz, uRaw[1] / usiz, uRaw[2] / usiz];
  const v = [vRaw[0] / vsiz, vRaw[1] / vsiz, vRaw[2] / vsiz];
  const w = [wRaw[0] / wsiz, wRaw[1] / wsiz, wRaw[2] / wsiz];
  const unitOf = (c: [number, number, number]): number[] => {
    const s = siz(c);
    return [c[0] / s, c[1] / s, c[2] / s];
  };
  const uv = unitOf(cross(v, u));
  const uw = unitOf(cross(w, u));
  const vw = unitOf(cross(w, v));
  const uvcos = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const uvsin = Math.sqrt(1.0 - uvcos * uvcos);
  const dphidu = -trq[0] * u[0] - trq[1] * u[1] - trq[2] * u[2];
  const dphidv = -trq[0] * v[0] - trq[1] * v[1] - trq[2] * v[2];
  const dphidw = -trq[0] * w[0] - trq[1] * w[1] - trq[2] * w[2];
  for (let j = 0; j < 3; j++) {
    let du: number, dv: number;
    if (kind === 'ZTHENX') {
      du = (uv[j] * dphidv) / (usiz * uvsin) + (uw[j] * dphidw) / usiz;
      dv = (-uv[j] * dphidu) / (vsiz * uvsin);
    } else {
      du = (uv[j] * dphidv) / (usiz * uvsin) + (0.5 * uw[j] * dphidw) / usiz;
      dv = (-uv[j] * dphidu) / (vsiz * uvsin) + (0.5 * vw[j] * dphidw) / vsiz;
    }
    f[3 * ia + j] -= du;
    f[3 * ic + j] -= dv;
    f[3 * ib + j] += du + dv;
    fz[j] += du;
    fx[j] += dv;
  }
  return { fx, fy, fz };
};

/** Output of multipoleReal: the energy and the virial (xx, yy, zz, xy, xz, yz). */
export interface MpoleResult {
  e: number;
  virial: Float64Array;
}

/**
 * multipole_real (the AMOEBA branch with no Ewald): the direct multipole energy over the half pair list, the pair
 * forces and torques, the torque-to-force conversion of every site and the virial. `pi`, `pj` are local indices and
 * `ps` the special code of the pair (0 none, 1..4 the 1-2..1-5 relations; `factor[ps]` is the mpole-1n-scale).
 */
export const multipoleReal = (
  sites: MpoleSites, x: Float64Array, f: Float64Array,
  pi: Int32Array, pj: Int32Array, ps: Int8Array, npairs: number,
  factor: readonly number[], dielectric: number, mpolecut: number,
  own: Int32Array, aewald: number,
): MpoleResult => {
  const felec = ELECTRIC / dielectric;
  const off2 = mpolecut * mpolecut;
  const rp = sites.rpole;
  const n = sites.n;
  // torques are summed on the owners (reverse communication of tq in native); own[k] is the owned atom of copy k
  const tq = new Float64Array(3 * n);
  const vir = new Float64Array(6);
  let e_tot = 0;
  for (let p = 0; p < npairs; p++) {
    const i = pi[p], j = pj[p];
    const factorMpole = factor[ps[p]];
    const xr = x[3 * j] - x[3 * i];
    const yr = x[3 * j + 1] - x[3 * i + 1];
    const zr = x[3 * j + 2] - x[3 * i + 2];
    const r2 = xr * xr + yr * yr + zr * zr;
    if (r2 > off2) continue;
    const r = Math.sqrt(r2);
    const oi = 13 * own[i], ok = 13 * own[j];
    const ci = rp[oi], dix = rp[oi + 1], diy = rp[oi + 2], diz = rp[oi + 3];
    const qixx = rp[oi + 4], qixy = rp[oi + 5], qixz = rp[oi + 6], qiyy = rp[oi + 8], qiyz = rp[oi + 9], qizz = rp[oi + 12];
    const ck = rp[ok], dkx = rp[ok + 1], dky = rp[ok + 2], dkz = rp[ok + 3];
    const qkxx = rp[ok + 4], qkxy = rp[ok + 5], qkxz = rp[ok + 6], qkyy = rp[ok + 8], qkyz = rp[ok + 9], qkzz = rp[ok + 12];
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
    // reciprocal distance terms; with no Ewald term bn[k] = (2k-1) bn[k-1] / r^2 and bn[0] = 1/r, all times felec
    let rr1 = felec / r;
    let rr3 = rr1 / r2;
    let rr5 = (3.0 * rr3) / r2;
    let rr7 = (5.0 * rr5) / r2;
    let rr9 = (7.0 * rr7) / r2;
    let rr11 = (9.0 * rr9) / r2;
    // Ewald real-space damping (amoeba_multipole.cpp multipole_real): bn[0] = erfc(aewald r)/r and the recursion
    // bn[k] = ((2k-1) bn[k-1] + alsq2n exp(-(aewald r)^2)) / r^2 with alsq2n = (2 aewald^2)^k / (sqrt(pi) aewald);
    // with aewald = 0 this is 1/r, (2k-1) bn[k-1] / r^2 as before
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
    // standard multipole intermediates and energy (the AMOEBA branch)
    const term1 = ci * ck;
    const term2 = ck * dir - ci * dkr + dik;
    const term3 = ci * qkr + ck * qir - dir * dkr + 2.0 * (dkqi - diqk + qiqk);
    const term4 = dir * qkr - dkr * qir - 4.0 * qik;
    const term5 = qir * qkr;
    const scalek = 1.0 - factorMpole;
    rr1 = bn[0] - scalek * rr1;
    rr3 = bn[1] - scalek * rr3;
    rr5 = bn[2] - scalek * rr5;
    rr7 = bn[3] - scalek * rr7;
    rr9 = bn[4] - scalek * rr9;
    rr11 = bn[5] - scalek * rr11;
    const e = term1 * rr1 + term2 * rr3 + term3 * rr5 + term4 * rr7 + term5 * rr9;
    const de = term1 * rr3 + term2 * rr5 + term3 * rr7 + term4 * rr9 + term5 * rr11;
    // intermediates for force and torque
    const t1 = -ck * rr3 + dkr * rr5 - qkr * rr7;
    const t2 = ci * rr3 + dir * rr5 + qir * rr7;
    const t3 = 2.0 * rr5;
    const t4 = 2.0 * (-ck * rr5 + dkr * rr7 - qkr * rr9);
    const t5 = 2.0 * (-ci * rr5 - dir * rr7 - qir * rr9);
    const t6 = 4.0 * rr7;
    e_tot += e;
    // force components for this interaction
    const frcx = de * xr + t1 * dix + t2 * dkx + t3 * (diqkx - dkqix) + t4 * qix + t5 * qkx + t6 * (qixk + qkxi);
    const frcy = de * yr + t1 * diy + t2 * dky + t3 * (diqky - dkqiy) + t4 * qiy + t5 * qky + t6 * (qiyk + qkyi);
    const frcz = de * zr + t1 * diz + t2 * dkz + t3 * (diqkz - dkqiz) + t4 * qiz + t5 * qkz + t6 * (qizk + qkzi);
    // torque components on both sites
    const ttmi0 = -rr3 * dikx + t1 * dirx + t3 * (dqikx + dkqirx) - t4 * qirx - t6 * (qikrx + qikx);
    const ttmi1 = -rr3 * diky + t1 * diry + t3 * (dqiky + dkqiry) - t4 * qiry - t6 * (qikry + qiky);
    const ttmi2 = -rr3 * dikz + t1 * dirz + t3 * (dqikz + dkqirz) - t4 * qirz - t6 * (qikrz + qikz);
    const ttmk0 = rr3 * dikx + t2 * dkrx - t3 * (dqikx + diqkrx) - t5 * qkrx - t6 * (qkirx - qikx);
    const ttmk1 = rr3 * diky + t2 * dkry - t3 * (dqiky + diqkry) - t5 * qkry - t6 * (qkiry - qiky);
    const ttmk2 = rr3 * dikz + t2 * dkrz - t3 * (dqikz + diqkrz) - t5 * qkrz - t6 * (qkirz - qikz);
    f[3 * i] -= frcx;
    f[3 * i + 1] -= frcy;
    f[3 * i + 2] -= frcz;
    tq[3 * own[i]] += ttmi0;
    tq[3 * own[i] + 1] += ttmi1;
    tq[3 * own[i] + 2] += ttmi2;
    f[3 * j] += frcx;
    f[3 * j + 1] += frcy;
    f[3 * j + 2] += frcz;
    tq[3 * own[j]] += ttmk0;
    tq[3 * own[j] + 1] += ttmk1;
    tq[3 * own[j] + 2] += ttmk2;
    // pairwise Cartesian virial (amoeba_multipole.cpp: virmpole -= v)
    const vxx = -xr * frcx;
    const vxy = -0.5 * (yr * frcx + xr * frcy);
    const vxz = -0.5 * (zr * frcx + xr * frcz);
    const vyy = -yr * frcy;
    const vyz = -0.5 * (zr * frcy + yr * frcz);
    const vzz = -zr * frcz;
    vir[0] -= vxx;
    vir[1] -= vyy;
    vir[2] -= vzz;
    vir[3] -= vxy;
    vir[4] -= vxz;
    vir[5] -= vyz;
  }
  // resolve the site torques into forces, then the frame correction of the virial
  for (let i = 0; i < n; i++) {
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
  return { e: e_tot, virial: vir };
};
