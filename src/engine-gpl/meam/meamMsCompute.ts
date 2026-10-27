/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM/meam_meam_dens_init.cpp
 * (meam_dens_setup, meam_dens_init, getscreen, calc_rho1), src/MEAM/meam_meam_dens_final.cpp (the msmeamflag branch of
 * meam_dens_final) and src/MEAM/meam_meam_force.cpp (meam_force with msmeamflag), and the PairMEAM::compute stage order
 * of src/MEAM/pair_meam.cpp. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Original MEAM authors (from
 * meam_meam_force.cpp and meam_meam_dens_init.cpp): Greg Wagner (Sandia), Sungkwang Mun, Stephen Foiles, Michael I.
 * Baskes; MS-MEAM: Steve Valone (LANL).
 *
 * The three stages run as in native LAMMPS on the engine's half list (owned-owned pairs at the lower index, owned-ghost
 * pairs once) and full list (screening neighbours). Partial densities of ghost images are summed into their owners
 * (the reverse communication of native LAMMPS) and copied back to the ghosts before the force stage (forward
 * communication); forces on ghosts are folded by the engine as for every pair style.
 */
import { MeamModel } from './meamModel';
import { fmExp, fcut, dfcut, dCfunc, getZij, getShpfcn, iszero, isone } from './meamMath';
import type { NeighList } from '../../engine/neighbor';

/** Per-atom accumulators of one evaluation (sized owned + ghost atoms). */
export interface MeamWork {
  rho0: Float64Array;
  arho2b: Float64Array;
  arho2mb: Float64Array;
  arho1: Float64Array;
  arho1m: Float64Array;
  arho2: Float64Array;
  arho2m: Float64Array;
  arho3: Float64Array;
  arho3m: Float64Array;
  arho3b: Float64Array;
  arho3mb: Float64Array;
  tAve: Float64Array;
  tsqAve: Float64Array;
  rho1: Float64Array;
  rho2: Float64Array;
  rho3: Float64Array;
  gamma: Float64Array;
  dgamma1: Float64Array;
  dgamma2: Float64Array;
  dgamma3: Float64Array;
  frhop: Float64Array;
  /** per half-list entry: screening, its derivative sum and the cutoff function */
  scrfcn: Float64Array;
  dscrfcn: Float64Array;
  fcpair: Float64Array;
}

export const newWork = (nall: number, nentries: number): MeamWork => {
  const a = (n: number): Float64Array => new Float64Array(n);
  return {
    rho0: a(nall), arho2b: a(nall), arho2mb: a(nall),
    arho1: a(3 * nall), arho1m: a(3 * nall), arho2: a(6 * nall), arho2m: a(6 * nall),
    arho3: a(10 * nall), arho3m: a(10 * nall), arho3b: a(3 * nall), arho3mb: a(3 * nall),
    tAve: a(3 * nall), tsqAve: a(3 * nall),
    rho1: a(nall), rho2: a(nall), rho3: a(nall), gamma: a(nall),
    dgamma1: a(nall), dgamma2: a(nall), dgamma3: a(nall), frhop: a(nall),
    scrfcn: a(nentries), dscrfcn: a(nentries), fcpair: a(nentries),
  };
};

/** Voigt multiplicities and index maps of MEAM::meam_setup_done (v2D, v3D, vind2D, vind3D). */
export const V2D = [1, 2, 2, 1, 2, 1];
export const V3D = [1, 3, 3, 3, 6, 3, 1, 3, 3, 1];
export const VIND2D: number[][] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];
export const VIND3D: number[][][] = (() => {
  const t: number[][][] = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => new Array<number>(3).fill(0)));
  let nv3 = 0;
  for (let m = 0; m < 3; m++) {
    for (let n = m; n < 3; n++) {
      for (let p = n; p < 3; p++) {
        t[m][n][p] = nv3; t[m][p][n] = nv3; t[n][m][p] = nv3;
        t[n][p][m] = nv3; t[p][m][n] = nv3; t[p][n][m] = nv3;
        nv3++;
      }
    }
  }
  return t;
})();

export interface ComputeInput {
  model: MeamModel;
  nlocal: number;
  nall: number;
  x: Float64Array;
  f: Float64Array;
  type: Int32Array;
  /** element index of each LAMMPS type (index 0 unused), -1 if unmapped */
  typeElem: number[];
  owner: Int32Array;
  half: NeighList;
  full: NeighList;
  /** NEIGHMASK of the engine neighbour encoding */
  mask: number;
  eatom: Float64Array | null;
  /** per-atom virial (6 per atom, native vatom order xx yy zz xy xz yz), or null */
  vatom: Float64Array | null;
  /** fix adapt's scale by type pair (pair_meam.cpp extract "scale"), index itype * (ntypes + 1) + jtype */
  scale: Float64Array;
  ntypes: number;
}

const sq = (v: number): number => v * v;

/** getscreen + calc_rho1 for every owned atom (meam_dens_init). */
export const densInit = (c: ComputeInput, w: MeamWork): void => {
  const M = c.model;
  const { x, type, typeElem, half, full, mask } = c;
  const cutforcesq = M.cutforcesq;
  const cutforce = M.cutforce;
  const drinv = 1.0 / M.delr;
  for (let i = 0; i < c.nlocal; i++) {
    const elti = typeElem[type[i]];
    if (elti < 0) continue;
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const h0 = half.firstneigh[i], h1 = h0 + half.numneigh[i];
    const f0 = full.firstneigh[i], f1 = f0 + full.numneigh[i];
    // getscreen
    for (let jn = h0; jn < h1; jn++) {
      const j = half.neighbors[jn] & mask;
      const eltj = typeElem[type[j]];
      w.scrfcn[jn] = 0.0; w.dscrfcn[jn] = 0.0; w.fcpair[jn] = 0.0;
      if (eltj < 0) continue;
      const xj = x[3 * j], yj = x[3 * j + 1], zj = x[3 * j + 2];
      const delxij = xj - xi, delyij = yj - yi, delzij = zj - zi;
      const rij2 = delxij * delxij + delyij * delyij + delzij * delzij;
      if (rij2 > cutforcesq) continue;
      const rbound = M.ebound[elti][eltj] * rij2;
      const rij = Math.sqrt(rij2);
      const rnorm = (cutforce - rij) * drinv;
      let sij = 1.0;
      for (let kn = f0; kn < f1; kn++) {
        const k = full.neighbors[kn] & mask;
        if (k === j) continue;
        const eltk = typeElem[type[k]];
        if (eltk < 0) continue;
        const xk = x[3 * k], yk = x[3 * k + 1], zk = x[3 * k + 2];
        const rjk2 = sq(xk - xj) + sq(yk - yj) + sq(zk - zj);
        if (rjk2 > rbound) continue;
        const rik2 = sq(xk - xi) + sq(yk - yi) + sq(zk - zi);
        if (rik2 > rbound) continue;
        const xik = rik2 / rij2, xjk = rjk2 / rij2;
        const a = 1 - (xik - xjk) * (xik - xjk);
        if (a <= 0.0) continue;
        let cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
        const Cmax = M.Cmax[elti][eltj][eltk];
        const Cmin = M.Cmin[elti][eltj][eltk];
        let sikj = 0;
        if (cikj >= Cmax) continue;
        else if (cikj <= Cmin) { sij = 0.0; break; }
        else {
          const delc = Cmax - Cmin;
          cikj = (cikj - Cmin) / delc;
          sikj = fcut(cikj);
        }
        sij *= sikj;
      }
      const fc = dfcut(rnorm);
      const fcij = fc.f;
      const dfcij = fc.df * drinv;
      let dscr = 0.0;
      const sfcij = sij * fcij;
      if (!iszero(sfcij) && !isone(sfcij)) {
        for (let kn = f0; kn < f1; kn++) {
          const k = full.neighbors[kn] & mask;
          if (k === j) continue;
          const eltk = typeElem[type[k]];
          if (eltk < 0) continue;
          const xk = x[3 * k], yk = x[3 * k + 1], zk = x[3 * k + 2];
          const rjk2 = sq(xk - xj) + sq(yk - yj) + sq(zk - zj);
          if (rjk2 > rbound) continue;
          const rik2 = sq(xk - xi) + sq(yk - yi) + sq(zk - zi);
          if (rik2 > rbound) continue;
          const xik = rik2 / rij2, xjk = rjk2 / rij2;
          const a = 1 - (xik - xjk) * (xik - xjk);
          if (a <= 0.0) continue;
          let cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
          const Cmax = M.Cmax[elti][eltj][eltk];
          const Cmin = M.Cmin[elti][eltj][eltk];
          if (cikj >= Cmax) continue;
          const delc = Cmax - Cmin;
          cikj = (cikj - Cmin) / delc;
          const sk = dfcut(cikj);
          const coef1 = sk.df / (delc * sk.f);
          const dCikj = dCfunc(rij2, rik2, rjk2);
          dscr = dscr + coef1 * dCikj;
        }
        const coef2 = (sij * dfcij) / rij;
        dscr = dscr * sfcij - coef2;
      }
      w.scrfcn[jn] = sij;
      w.dscrfcn[jn] = dscr;
      w.fcpair[jn] = fcij;
    }
    // calc_rho1
    for (let jn = h0; jn < h1; jn++) {
      if (iszero(w.scrfcn[jn])) continue;
      const j = half.neighbors[jn] & mask;
      const sij = w.scrfcn[jn] * w.fcpair[jn];
      const d0 = x[3 * j] - xi, d1 = x[3 * j + 1] - yi, d2 = x[3 * j + 2] - zi;
      const rij2 = d0 * d0 + d1 * d1 + d2 * d2;
      if (!(rij2 < cutforcesq)) continue;
      const eltj = typeElem[type[j]];
      if (eltj < 0) continue;
      const rij = Math.sqrt(rij2);
      const delij = [d0, d1, d2];
      const ai = rij / M.re[elti][elti] - 1.0;
      const aj = rij / M.re[eltj][eltj] - 1.0;
      const ro0i = M.rho0[elti];
      const ro0j = M.rho0[eltj];
      const rhoa0j = ro0j * fmExp(-M.beta0[eltj] * aj) * sij;
      let rhoa1j = ro0j * fmExp(-M.beta1[eltj] * aj) * sij;
      let rhoa2j = ro0j * fmExp(-M.beta2[eltj] * aj) * sij;
      let rhoa3j = ro0j * fmExp(-M.beta3[eltj] * aj) * sij;
      const rhoa1mj = ro0j * M.t1m[eltj] * fmExp(-M.beta1m[eltj] * aj) * sij;
      const rhoa2mj = ro0j * M.t2m[eltj] * fmExp(-M.beta2m[eltj] * aj) * sij;
      const rhoa3mj = ro0j * M.t3m[eltj] * fmExp(-M.beta3m[eltj] * aj) * sij;
      const rhoa0i = ro0i * fmExp(-M.beta0[elti] * ai) * sij;
      let rhoa1i = ro0i * fmExp(-M.beta1[elti] * ai) * sij;
      let rhoa2i = ro0i * fmExp(-M.beta2[elti] * ai) * sij;
      let rhoa3i = ro0i * fmExp(-M.beta3[elti] * ai) * sij;
      const rhoa1mi = ro0i * M.t1m[elti] * fmExp(-M.beta1m[elti] * ai) * sij;
      const rhoa2mi = ro0i * M.t2m[elti] * fmExp(-M.beta2m[elti] * ai) * sij;
      const rhoa3mi = ro0i * M.t3m[elti] * fmExp(-M.beta3m[elti] * ai) * sij;
      if (M.ialloy === 1) {
        rhoa1j = rhoa1j * M.t1[eltj];
        rhoa2j = rhoa2j * M.t2[eltj];
        rhoa3j = rhoa3j * M.t3[eltj];
        rhoa1i = rhoa1i * M.t1[elti];
        rhoa2i = rhoa2i * M.t2[elti];
        rhoa3i = rhoa3i * M.t3[elti];
      }
      w.rho0[i] += rhoa0j;
      w.rho0[j] += rhoa0i;
      if (M.ialloy !== 2) {
        w.tAve[3 * i] += M.t1[eltj] * rhoa0j;
        w.tAve[3 * i + 1] += M.t2[eltj] * rhoa0j;
        w.tAve[3 * i + 2] += M.t3[eltj] * rhoa0j;
        w.tAve[3 * j] += M.t1[elti] * rhoa0i;
        w.tAve[3 * j + 1] += M.t2[elti] * rhoa0i;
        w.tAve[3 * j + 2] += M.t3[elti] * rhoa0i;
      }
      if (M.ialloy === 1) {
        w.tsqAve[3 * i] += M.t1[eltj] * M.t1[eltj] * rhoa0j;
        w.tsqAve[3 * i + 1] += M.t2[eltj] * M.t2[eltj] * rhoa0j;
        w.tsqAve[3 * i + 2] += M.t3[eltj] * M.t3[eltj] * rhoa0j;
        w.tsqAve[3 * j] += M.t1[elti] * M.t1[elti] * rhoa0i;
        w.tsqAve[3 * j + 1] += M.t2[elti] * M.t2[elti] * rhoa0i;
        w.tsqAve[3 * j + 2] += M.t3[elti] * M.t3[elti] * rhoa0i;
      }
      w.arho2b[i] += rhoa2j;
      w.arho2b[j] += rhoa2i;
      const A1j = rhoa1j / rij, A2j = rhoa2j / rij2, A3j = rhoa3j / (rij2 * rij);
      const A1i = rhoa1i / rij, A2i = rhoa2i / rij2, A3i = rhoa3i / (rij2 * rij);
      w.arho2mb[i] += rhoa2mj;
      w.arho2mb[j] += rhoa2mi;
      const A1mj = rhoa1mj / rij, A2mj = rhoa2mj / rij2, A3mj = rhoa3mj / (rij2 * rij);
      const A1mi = rhoa1mi / rij, A2mi = rhoa2mi / rij2, A3mi = rhoa3mi / (rij2 * rij);
      let nv2 = 0, nv3 = 0;
      for (let m = 0; m < 3; m++) {
        w.arho1[3 * i + m] += A1j * delij[m];
        w.arho1[3 * j + m] -= A1i * delij[m];
        w.arho3b[3 * i + m] += (rhoa3j * delij[m]) / rij;
        w.arho3b[3 * j + m] -= (rhoa3i * delij[m]) / rij;
        w.arho1m[3 * i + m] += A1mj * delij[m];
        w.arho1m[3 * j + m] -= A1mi * delij[m];
        w.arho3mb[3 * i + m] += (rhoa3mj * delij[m]) / rij;
        w.arho3mb[3 * j + m] -= (rhoa3mi * delij[m]) / rij;
        for (let n = m; n < 3; n++) {
          w.arho2[6 * i + nv2] += A2j * delij[m] * delij[n];
          w.arho2[6 * j + nv2] += A2i * delij[m] * delij[n];
          w.arho2m[6 * i + nv2] += A2mj * delij[m] * delij[n];
          w.arho2m[6 * j + nv2] += A2mi * delij[m] * delij[n];
          nv2 = nv2 + 1;
          for (let p = n; p < 3; p++) {
            w.arho3[10 * i + nv3] += A3j * delij[m] * delij[n] * delij[p];
            w.arho3[10 * j + nv3] -= A3i * delij[m] * delij[n] * delij[p];
            w.arho3m[10 * i + nv3] += A3mj * delij[m] * delij[n] * delij[p];
            w.arho3m[10 * j + nv3] -= A3mi * delij[m] * delij[n] * delij[p];
            nv3 = nv3 + 1;
          }
        }
      }
    }
  }
};

/** Sum the partial densities of ghost images into their owners (reverse communication). */
export const foldGhosts = (c: ComputeInput, w: MeamWork): void => {
  const o = c.owner;
  const add = (arr: Float64Array, per: number): void => {
    for (let g = c.nlocal; g < c.nall; g++) {
      const to = o[g];
      for (let q = 0; q < per; q++) arr[per * to + q] += arr[per * g + q];
    }
  };
  add(w.rho0, 1); add(w.arho2b, 1); add(w.arho2mb, 1);
  add(w.arho1, 3); add(w.arho1m, 3); add(w.arho2, 6); add(w.arho2m, 6);
  add(w.arho3, 10); add(w.arho3m, 10); add(w.arho3b, 3); add(w.arho3mb, 3);
  add(w.tAve, 3); add(w.tsqAve, 3);
};

/** Copy the owned values of the per-atom arrays to their ghost images (forward communication). */
export const forwardGhosts = (c: ComputeInput, w: MeamWork): void => {
  const o = c.owner;
  const copy = (arr: Float64Array, per: number): void => {
    for (let g = c.nlocal; g < c.nall; g++) {
      const from = o[g];
      for (let q = 0; q < per; q++) arr[per * g + q] = arr[per * from + q];
    }
  };
  copy(w.rho1, 1); copy(w.rho2, 1); copy(w.rho3, 1); copy(w.gamma, 1);
  copy(w.dgamma1, 1); copy(w.dgamma2, 1); copy(w.dgamma3, 1); copy(w.frhop, 1);
  copy(w.arho2b, 1); copy(w.arho2mb, 1);
  copy(w.arho1, 3); copy(w.arho1m, 3); copy(w.arho2, 6); copy(w.arho2m, 6);
  copy(w.arho3, 10); copy(w.arho3m, 10); copy(w.arho3b, 3); copy(w.arho3mb, 3);
};

/** meam_dens_final (msmeamflag branch). Returns the embedding energy of the owned atoms. */
export const densFinal = (c: ComputeInput, w: MeamWork): number => {
  const M = c.model;
  let eng = 0;
  for (let i = 0; i < c.nlocal; i++) {
    const elti = c.typeElem[c.type[i]];
    if (elti < 0) continue;
    const lat = M.lattce[elti][elti];
    w.rho1[i] = 0.0;
    w.rho2[i] = (-1.0 / 3.0) * (sq(w.arho2b[i]) - sq(w.arho2mb[i]));
    w.rho3[i] = 0.0;
    for (let m = 0; m < 3; m++) {
      w.rho1[i] = w.rho1[i] + sq(w.arho1[3 * i + m]) - sq(w.arho1m[3 * i + m]);
      w.rho3[i] = w.rho3[i] - (3.0 / 5.0) * (sq(w.arho3b[3 * i + m]) - sq(w.arho3mb[3 * i + m]));
    }
    for (let m = 0; m < 6; m++) {
      w.rho2[i] = w.rho2[i] + V2D[m] * (sq(w.arho2[6 * i + m]) - sq(w.arho2m[6 * i + m]));
    }
    for (let m = 0; m < 10; m++) {
      w.rho3[i] = w.rho3[i] + V3D[m] * (sq(w.arho3[10 * i + m]) - sq(w.arho3m[10 * i + m]));
    }
    w.gamma[i] = w.rho1[i] + w.rho2[i] + w.rho3[i];
    if (w.rho0[i] > 0.0) w.gamma[i] = w.gamma[i] / (w.rho0[i] * w.rho0[i]);
    const Z = getZij(lat);
    const G = M.gGam(w.gamma[i], M.ibar[elti]);
    const shp = getShpfcn(lat, M.stheta[elti][elti], M.ctheta[elti][elti]);
    let Gbar: number, dGbar = 0.0;
    if (M.ibar[elti] <= 0) {
      Gbar = 1.0;
      dGbar = 0.0;
    } else {
      let gam: number;
      if (M.mixRefT === 1) gam = (w.tAve[3 * i] * shp[0] + w.tAve[3 * i + 1] * shp[1] + w.tAve[3 * i + 2] * shp[2]) / (Z * Z);
      else gam = (M.t1[elti] * shp[0] + M.t2[elti] * shp[1] + M.t3[elti] * shp[2]) / (Z * Z);
      Gbar = M.gGam(gam, M.ibar[elti]);
    }
    const rho = w.rho0[i] * G;
    let rhoBkgd: number;
    if (M.mixRefT === 1) {
      if (M.ibar[elti] <= 0) {
        Gbar = 1.0;
        dGbar = 0.0;
      } else {
        const gam = (w.tAve[3 * i] * shp[0] + w.tAve[3 * i + 1] * shp[1] + w.tAve[3 * i + 2] * shp[2]) / (Z * Z);
        const r = M.dGGam(gam, M.ibar[elti]);
        Gbar = r.G;
        dGbar = r.dG;
      }
      rhoBkgd = M.rho0[elti] * Z * Gbar;
    } else if (M.bkgdDyn === 1) {
      rhoBkgd = M.rho0[elti] * Z;
    } else {
      rhoBkgd = M.rhoRef[elti];
    }
    const rhob = rho / rhoBkgd;
    const denom = 1.0 / rhoBkgd;
    const dgr = M.dGGam(w.gamma[i], M.ibar[elti]);
    const Gd = dgr.G, dG = dgr.dG;
    w.dgamma1[i] = (Gd - 2 * dG * w.gamma[i]) * denom;
    w.dgamma2[i] = !iszero(w.rho0[i]) ? (dG / w.rho0[i]) * denom : 0.0;
    w.dgamma3[i] = M.mixRefT === 1 ? ((w.rho0[i] * Gd * dGbar) / (Gbar * Z * Z)) * denom : 0.0;
    const emb = M.embedding(M.A[elti], M.Ec[elti][elti], rhob);
    w.frhop[i] = emb.dF;
    // meam_dens_final.cpp: Fl *= scale[type[i]][type[i]] for the energy only (frhop stays unscaled)
    const Fl = emb.F * c.scale[c.type[i] * (c.ntypes + 1) + c.type[i]];
    eng += Fl;
    if (c.eatom) c.eatom[i] += Fl;
  }
  return eng;
};

