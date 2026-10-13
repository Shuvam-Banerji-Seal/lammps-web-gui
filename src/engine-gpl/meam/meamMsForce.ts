/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM/meam_meam_force.cpp (meam_force,
 * msmeamflag branches) and the spline evaluation of its pair term. LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Original MEAM authors (from meam_meam_force.cpp): Greg Wagner (Sandia), Sungkwang Mun,
 * Stephen Foiles, Michael I. Baskes; MS-MEAM: Steve Valone (LANL).
 *
 * One half pair (i, j) per call of the inner body, as in native LAMMPS: the energy of the pair term phi * S, the
 * density derivatives of i and j, and the three-body screening forces on the full-list neighbours k of i.
 */
import { fmExp, dfcut, dCfunc2, iszero, isone } from './meamMath';
import { MeamWork, ComputeInput, V2D, V3D, VIND2D, VIND3D } from './meamMsCompute';


/**
 * The three-body screening forces of one half pair (i, j) on the full-list neighbours k of i (meam_force.cpp, the k loop
 * after the pair term, shared by the MS-MEAM and the standard MEAM force). dUdsij is the derivative with respect to the
 * screening of the pair.
 */
export const addScreeningForces = (c: ComputeInput, w: MeamWork, f: Float64Array, jn: number, i: number, j: number,
  elti: number, eltj: number, f0: number, f1: number, rij2: number, dUdsij: number): void => {
  const M = c.model;
  const { x, type, typeElem, full, mask } = c;
  for (let kn = f0; kn < f1; kn++) {
    const k = full.neighbors[kn] & mask;
    const eltk = typeElem[type[k]];
    if (k === j || eltk < 0) continue;
    let dsij1 = 0.0, dsij2 = 0.0;
    let dxik = 0, dyik = 0, dzik = 0, dxjk = 0, dyjk = 0, dzjk = 0;
    const sijk = w.scrfcn[jn] * w.fcpair[jn];
    const Cmax = M.Cmax[elti][eltj][eltk];
    const Cmin = M.Cmin[elti][eltj][eltk];
    if (!iszero(sijk) && !isone(sijk)) {
      const rbound = rij2 * M.ebound[elti][eltj];
      const delc = Cmax - Cmin;
      dxjk = x[3 * k] - x[3 * j];
      dyjk = x[3 * k + 1] - x[3 * j + 1];
      dzjk = x[3 * k + 2] - x[3 * j + 2];
      const rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
      if (rjk2 <= rbound) {
        dxik = x[3 * k] - x[3 * i];
        dyik = x[3 * k + 1] - x[3 * i + 1];
        dzik = x[3 * k + 2] - x[3 * i + 2];
        const rik2 = dxik * dxik + dyik * dyik + dzik * dzik;
        if (rik2 <= rbound) {
          const xik = rik2 / rij2;
          const xjk = rjk2 / rij2;
          let a = 1 - (xik - xjk) * (xik - xjk);
          if (!iszero(a)) {
            let cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
            if (cikj >= Cmin && cikj <= Cmax) {
              cikj = (cikj - Cmin) / delc;
              const sk = dfcut(cikj);
              const d = dCfunc2(rij2, rik2, rjk2);
              a = (sijk / delc) * sk.df / sk.f;
              dsij1 = a * d[0];
              dsij2 = a * d[1];
            }
          }
        }
      }
    }
    if (!iszero(dsij1) || !iszero(dsij2)) {
      const force1 = dUdsij * dsij1;
      const force2 = dUdsij * dsij2;
      f[3 * i] += force1 * dxik;
      f[3 * i + 1] += force1 * dyik;
      f[3 * i + 2] += force1 * dzik;
      f[3 * j] += force2 * dxjk;
      f[3 * j + 1] += force2 * dyjk;
      f[3 * j + 2] += force2 * dzjk;
      f[3 * k] -= force1 * dxik + force2 * dxjk;
      f[3 * k + 1] -= force1 * dyik + force2 * dyjk;
      f[3 * k + 2] -= force1 * dzik + force2 * dzjk;
    }
  }};

/** meam_force for every owned atom: forces into c.f and energy (pair term) returned. */
export const forceAll = (c: ComputeInput, w: MeamWork, rdrar: number): number => {
  const M = c.model;
  const { x, f, type, typeElem, half, full, mask } = c;
  const cutforcesq = M.cutforcesq;
  let eng = 0;
  const delij = [0, 0, 0];
  for (let i = 0; i < c.nlocal; i++) {
    const elti = typeElem[type[i]];
    if (elti < 0) continue;
    const xitmp = x[3 * i], yitmp = x[3 * i + 1], zitmp = x[3 * i + 2];
    const h0 = half.firstneigh[i], h1 = h0 + half.numneigh[i];
    const f0 = full.firstneigh[i], f1 = f0 + full.numneigh[i];
    for (let jn = h0; jn < h1; jn++) {
      const j = half.neighbors[jn] & mask;
      const eltj = typeElem[type[j]];
      if (!(!iszero(w.scrfcn[jn]) && eltj >= 0)) continue;
      const sij = w.scrfcn[jn] * w.fcpair[jn];
      delij[0] = x[3 * j] - xitmp;
      delij[1] = x[3 * j + 1] - yitmp;
      delij[2] = x[3 * j + 2] - zitmp;
      const rij2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
      if (!(rij2 < cutforcesq)) continue;
      const rij = Math.sqrt(rij2);
      const recip = 1.0 / rij;
      const ind = M.eltind[elti][eltj];
      let pp = rij * rdrar;
      let kk = Math.trunc(pp);
      kk = Math.min(kk, M.nr - 2);
      pp = pp - kk;
      pp = Math.min(pp, 1.0);
      const r1 = M.phirar1[ind], r2 = M.phirar2[ind], r3 = M.phirar3[ind];
      const r4 = M.phirar4[ind], r5 = M.phirar5[ind], r6 = M.phirar6[ind], r0 = M.phirar[ind];
      const phi = (((r3[kk] * pp + r2[kk]) * pp + r1[kk]) * pp + r0[kk]);
      const phip = (r6[kk] * pp + r5[kk]) * pp + r4[kk];
      const scaleij = 1.0;
      const phiSc = phi * scaleij;
      eng += phiSc * sij;
      if (c.eatom) {
        c.eatom[i] += 0.5 * phiSc * sij;
        c.eatom[j] += 0.5 * phiSc * sij;
      }
      const invrei = 1.0 / M.re[elti][elti];
      const ai = rij * invrei - 1.0;
      const ro0i = M.rho0[elti];
      const rhoa0i = ro0i * fmExp(-M.beta0[elti] * ai);
      const drhoa0i = -M.beta0[elti] * invrei * rhoa0i;
      let rhoa1i = ro0i * fmExp(-M.beta1[elti] * ai);
      let drhoa1i = -M.beta1[elti] * invrei * rhoa1i;
      let rhoa2i = ro0i * fmExp(-M.beta2[elti] * ai);
      let drhoa2i = -M.beta2[elti] * invrei * rhoa2i;
      let rhoa3i = ro0i * fmExp(-M.beta3[elti] * ai);
      let drhoa3i = -M.beta3[elti] * invrei * rhoa3i;
      const rhoa1mi = ro0i * fmExp(-M.beta1m[elti] * ai) * M.t1m[elti];
      const drhoa1mi = -M.beta1m[elti] * invrei * rhoa1mi;
      const rhoa2mi = ro0i * fmExp(-M.beta2m[elti] * ai) * M.t2m[elti];
      const drhoa2mi = -M.beta2m[elti] * invrei * rhoa2mi;
      const rhoa3mi = ro0i * fmExp(-M.beta3m[elti] * ai) * M.t3m[elti];
      const drhoa3mi = -M.beta3m[elti] * invrei * rhoa3mi;
      let rhoa0j: number, drhoa0j: number, rhoa1j: number, drhoa1j: number, rhoa2j: number, drhoa2j: number;
      let rhoa3j: number, drhoa3j: number, rhoa1mj: number, drhoa1mj: number, rhoa2mj: number, drhoa2mj: number;
      let rhoa3mj: number, drhoa3mj: number;
      if (elti !== eltj) {
        const invrej = 1.0 / M.re[eltj][eltj];
        const aj = rij * invrej - 1.0;
        const ro0j = M.rho0[eltj];
        rhoa0j = ro0j * fmExp(-M.beta0[eltj] * aj);
        drhoa0j = -M.beta0[eltj] * invrej * rhoa0j;
        rhoa1j = ro0j * fmExp(-M.beta1[eltj] * aj);
        drhoa1j = -M.beta1[eltj] * invrej * rhoa1j;
        rhoa2j = ro0j * fmExp(-M.beta2[eltj] * aj);
        drhoa2j = -M.beta2[eltj] * invrej * rhoa2j;
        rhoa3j = ro0j * fmExp(-M.beta3[eltj] * aj);
        drhoa3j = -M.beta3[eltj] * invrej * rhoa3j;
        rhoa1mj = ro0j * M.t1m[eltj] * fmExp(-M.beta1m[eltj] * aj);
        drhoa1mj = -M.beta1m[eltj] * invrej * rhoa1mj;
        rhoa2mj = ro0j * M.t2m[eltj] * fmExp(-M.beta2m[eltj] * aj);
        drhoa2mj = -M.beta2m[eltj] * invrej * rhoa2mj;
        rhoa3mj = ro0j * M.t3m[eltj] * fmExp(-M.beta3m[eltj] * aj);
        drhoa3mj = -M.beta3m[eltj] * invrej * rhoa3mj;
      } else {
        rhoa0j = rhoa0i; drhoa0j = drhoa0i;
        rhoa1j = rhoa1i; drhoa1j = drhoa1i;
        rhoa2j = rhoa2i; drhoa2j = drhoa2i;
        rhoa3j = rhoa3i; drhoa3j = drhoa3i;
        rhoa1mj = rhoa1mi; drhoa1mj = drhoa1mi;
        rhoa2mj = rhoa2mi; drhoa2mj = drhoa2mi;
        rhoa3mj = rhoa3mi; drhoa3mj = drhoa3mi;
      }
      // ialloy = 1 (always for MS-MEAM): the t weights multiply the non-m densities
      const t1mi = M.t1[elti], t2mi = M.t2[elti], t3mi = M.t3[elti];
      const t1mj = M.t1[eltj], t2mj = M.t2[eltj], t3mj = M.t3[eltj];
      rhoa1j *= t1mj; rhoa2j *= t2mj; rhoa3j *= t3mj;
      rhoa1i *= t1mi; rhoa2i *= t2mi; rhoa3i *= t3mi;
      drhoa1j *= t1mj; drhoa2j *= t2mj; drhoa3j *= t3mj;
      drhoa1i *= t1mi; drhoa2i *= t2mi; drhoa3i *= t3mi;

      let nv2 = 0, nv3 = 0;
      let arg1i1 = 0, arg1j1 = 0, arg1i2 = 0, arg1j2 = 0, arg1i3 = 0, arg1j3 = 0, arg3i3 = 0, arg3j3 = 0;
      for (let n = 0; n < 3; n++) {
        for (let p = n; p < 3; p++) {
          for (let q = p; q < 3; q++) {
            const arg = delij[n] * delij[p] * delij[q] * V3D[nv3];
            arg1i3 = arg1i3 + w.arho3[10 * i + nv3] * arg;
            arg1j3 = arg1j3 - w.arho3[10 * j + nv3] * arg;
            nv3 = nv3 + 1;
          }
          const arg = delij[n] * delij[p] * V2D[nv2];
          arg1i2 = arg1i2 + w.arho2[6 * i + nv2] * arg;
          arg1j2 = arg1j2 + w.arho2[6 * j + nv2] * arg;
          nv2 = nv2 + 1;
        }
        arg1i1 = arg1i1 + w.arho1[3 * i + n] * delij[n];
        arg1j1 = arg1j1 - w.arho1[3 * j + n] * delij[n];
        arg3i3 = arg3i3 + w.arho3b[3 * i + n] * delij[n];
        arg3j3 = arg3j3 - w.arho3b[3 * j + n] * delij[n];
      }
      nv2 = 0;
      nv3 = 0;
      let arg1i1m = 0, arg1j1m = 0, arg1i2m = 0, arg1j2m = 0, arg1i3m = 0, arg1j3m = 0, arg3i3m = 0, arg3j3m = 0;
      for (let n = 0; n < 3; n++) {
        for (let p = n; p < 3; p++) {
          for (let q = p; q < 3; q++) {
            const arg = delij[n] * delij[p] * delij[q] * V3D[nv3];
            arg1i3m = arg1i3m - w.arho3m[10 * i + nv3] * arg;
            arg1j3m = arg1j3m + w.arho3m[10 * j + nv3] * arg;
            nv3 = nv3 + 1;
          }
          const arg = delij[n] * delij[p] * V2D[nv2];
          arg1i2m = arg1i2m + w.arho2m[6 * i + nv2] * arg;
          arg1j2m = arg1j2m + w.arho2m[6 * j + nv2] * arg;
          nv2 = nv2 + 1;
        }
        arg1i1m = arg1i1m - w.arho1m[3 * i + n] * delij[n];
        arg1j1m = arg1j1m + w.arho1m[3 * j + n] * delij[n];
        arg3i3m = arg3i3m - w.arho3mb[3 * i + n] * delij[n];
        arg3j3m = arg3j3m + w.arho3mb[3 * j + n] * delij[n];
      }

      const drho0dr1 = drhoa0j * sij;
      const drho0dr2 = drhoa0i * sij;
      let a1 = (2 * sij) / rij;
      const drho1dr1 = a1 * (drhoa1j - rhoa1j / rij) * arg1i1;
      const drho1dr2 = a1 * (drhoa1i - rhoa1i / rij) * arg1j1;
      a1 = (2.0 * sij) / rij;
      const drho1drm1 = [0, 0, 0], drho1drm2 = [0, 0, 0];
      for (let m = 0; m < 3; m++) {
        drho1drm1[m] = a1 * rhoa1j * w.arho1[3 * i + m];
        drho1drm2[m] = -a1 * rhoa1i * w.arho1[3 * j + m];
      }
      let a2 = (2 * sij) / rij2;
      const drho2dr1 = a2 * (drhoa2j - (2 * rhoa2j) / rij) * arg1i2 - (2.0 / 3.0) * w.arho2b[i] * drhoa2j * sij;
      const drho2dr2 = a2 * (drhoa2i - (2 * rhoa2i) / rij) * arg1j2 - (2.0 / 3.0) * w.arho2b[j] * drhoa2i * sij;
      a2 = (4 * sij) / rij2;
      const drho2drm1 = [0, 0, 0], drho2drm2 = [0, 0, 0];
      for (let m = 0; m < 3; m++) {
        let s1 = 0, s2 = 0;
        for (let n = 0; n < 3; n++) {
          s1 = s1 + w.arho2[6 * i + VIND2D[m][n]] * delij[n];
          s2 = s2 - w.arho2[6 * j + VIND2D[m][n]] * delij[n];
        }
        drho2drm1[m] = a2 * rhoa2j * s1;
        drho2drm2[m] = -a2 * rhoa2i * s2;
      }
      const rij3 = rij * rij2;
      let a3 = (2 * sij) / rij3;
      let a3a = ((6.0 / 5.0) * sij) / rij;
      const drho3dr1 = a3 * (drhoa3j - (3 * rhoa3j) / rij) * arg1i3 - a3a * (drhoa3j - rhoa3j / rij) * arg3i3;
      const drho3dr2 = a3 * (drhoa3i - (3 * rhoa3i) / rij) * arg1j3 - a3a * (drhoa3i - rhoa3i / rij) * arg3j3;
      a3 = (6 * sij) / rij3;
      a3a = (6 * sij) / (5 * rij);
      const drho3drm1 = [0, 0, 0], drho3drm2 = [0, 0, 0];
      for (let m = 0; m < 3; m++) {
        let s1 = 0, s2 = 0;
        let nv2b = 0;
        for (let n = 0; n < 3; n++) {
          for (let p = n; p < 3; p++) {
            const arg = delij[n] * delij[p] * V2D[nv2b];
            s1 = s1 + w.arho3[10 * i + VIND3D[m][n][p]] * arg;
            s2 = s2 + w.arho3[10 * j + VIND3D[m][n][p]] * arg;
            nv2b = nv2b + 1;
          }
        }
        drho3drm1[m] = (a3 * s1 - a3a * w.arho3b[3 * i + m]) * rhoa3j;
        drho3drm2[m] = (-a3 * s2 + a3a * w.arho3b[3 * j + m]) * rhoa3i;
      }
      // m-densities (MS-MEAM branch)
      a1 = (2 * sij) / rij;
      let drho1mdr1 = a1 * (drhoa1mj - rhoa1mj / rij) * arg1i1m;
      let drho1mdr2 = a1 * (drhoa1mi - rhoa1mi / rij) * arg1j1m;
      drho1mdr1 *= -1.0;
      drho1mdr2 *= -1.0;
      a1 = (2.0 * sij) / rij;
      const drho1mdrm1 = [0, 0, 0], drho1mdrm2 = [0, 0, 0];
      for (let m = 0; m < 3; m++) {
        drho1mdrm1[m] = a1 * rhoa1mj * w.arho1m[3 * i + m];
        drho1mdrm2[m] = -a1 * rhoa1mi * w.arho1m[3 * j + m];
      }
      a2 = (2 * sij) / rij2;
      const drho2mdr1 = a2 * (drhoa2mj - (2 * rhoa2mj) / rij) * arg1i2m - (2.0 / 3.0) * w.arho2mb[i] * drhoa2mj * sij;
      const drho2mdr2 = a2 * (drhoa2mi - (2 * rhoa2mi) / rij) * arg1j2m - (2.0 / 3.0) * w.arho2mb[j] * drhoa2mi * sij;
      a2 = (4 * sij) / rij2;
      const drho2mdrm1 = [0, 0, 0], drho2mdrm2 = [0, 0, 0];
      for (let m = 0; m < 3; m++) {
        let s1 = 0, s2 = 0;
        for (let n = 0; n < 3; n++) {
          s1 = s1 + w.arho2m[6 * i + VIND2D[m][n]] * delij[n];
          s2 = s2 - w.arho2m[6 * j + VIND2D[m][n]] * delij[n];
        }
        drho2mdrm1[m] = a2 * rhoa2mj * s1;
        drho2mdrm2[m] = -a2 * rhoa2mi * s2;
      }
      a3 = (2 * sij) / rij3;
      a3a = ((6.0 / 5.0) * sij) / rij;
      let drho3mdr1 = a3 * (drhoa3mj - (3 * rhoa3mj) / rij) * arg1i3m - a3a * (drhoa3mj - rhoa3mj / rij) * arg3i3m;
      let drho3mdr2 = a3 * (drhoa3mi - (3 * rhoa3mi) / rij) * arg1j3m - a3a * (drhoa3mi - rhoa3mi / rij) * arg3j3m;
      drho3mdr1 *= -1.0;
      drho3mdr2 *= -1.0;
      a3 = (6 * sij) / rij3;
      a3a = (6 * sij) / (5 * rij);
      const drho3mdrm1 = [0, 0, 0], drho3mdrm2 = [0, 0, 0];
      for (let m = 0; m < 3; m++) {
        let s1 = 0, s2 = 0;
        let nv2b = 0;
        for (let n = 0; n < 3; n++) {
          for (let p = n; p < 3; p++) {
            const arg = delij[n] * delij[p] * V2D[nv2b];
            s1 = s1 + w.arho3m[10 * i + VIND3D[m][n][p]] * arg;
            s2 = s2 + w.arho3m[10 * j + VIND3D[m][n][p]] * arg;
            nv2b = nv2b + 1;
          }
        }
        drho3mdrm1[m] = (a3 * s1 - a3a * w.arho3mb[3 * i + m]) * rhoa3mj;
        drho3mdrm2[m] = (-a3 * s2 + a3a * w.arho3mb[3 * j + m]) * rhoa3mi;
      }
      // msmeam: t1..t3 = 1 and their derivatives vanish in the density derivatives
      const drhodr1 = w.dgamma1[i] * drho0dr1 +
        w.dgamma2[i] * ((drho1dr1 - drho1mdr1) + (drho2dr1 - drho2mdr1) + (drho3dr1 - drho3mdr1));
      const drhodr2 = w.dgamma1[j] * drho0dr2 +
        w.dgamma2[j] * ((drho1dr2 - drho1mdr2) + (drho2dr2 - drho2mdr2) + (drho3dr2 - drho3mdr2));
      const drhodrm1 = [0, 0, 0], drhodrm2 = [0, 0, 0];
      for (let m = 0; m < 3; m++) {
        drhodrm1[m] = w.dgamma2[i] * ((drho1drm1[m] - drho1mdrm1[m]) + (drho2drm1[m] - drho2mdrm1[m]) + (drho3drm1[m] - drho3mdrm1[m]));
        drhodrm2[m] = w.dgamma2[j] * ((drho1drm2[m] - drho1mdrm2[m]) + (drho2drm2[m] - drho2mdrm2[m]) + (drho3drm2[m] - drho3mdrm2[m]));
      }
      let drhods1 = 0, drhods2 = 0;
      const dscr = w.dscrfcn[jn];
      if (!iszero(dscr)) {
        const drho0ds1 = rhoa0j;
        const drho0ds2 = rhoa0i;
        a1 = 2.0 / rij;
        a2 = 2.0 / rij2;
        a3 = 2.0 / rij3;
        a3a = 6.0 / (5.0 * rij);
        const drho1ds1 = a1 * rhoa1j * arg1i1;
        const drho1ds2 = a1 * rhoa1i * arg1j1;
        const drho2ds1 = a2 * rhoa2j * arg1i2 - (2.0 / 3.0) * w.arho2b[i] * rhoa2j;
        const drho2ds2 = a2 * rhoa2i * arg1j2 - (2.0 / 3.0) * w.arho2b[j] * rhoa2i;
        const drho3ds1 = a3 * rhoa3j * arg1i3 - a3a * rhoa3j * arg3i3;
        const drho3ds2 = a3 * rhoa3i * arg1j3 - a3a * rhoa3i * arg3j3;
        const drho1mds1 = -a1 * rhoa1mj * arg1i1m;
        const drho1mds2 = -a1 * rhoa1mi * arg1j1m;
        const drho2mds1 = a2 * rhoa2mj * arg1i2m - (2.0 / 3.0) * w.arho2mb[i] * rhoa2mj;
        const drho2mds2 = a2 * rhoa2mi * arg1j2m - (2.0 / 3.0) * w.arho2mb[j] * rhoa2mi;
        const drho3mds1 = -a3 * rhoa3mj * arg1i3m + a3a * rhoa3mj * arg3i3m;
        const drho3mds2 = -a3 * rhoa3mi * arg1j3m + a3a * rhoa3mi * arg3j3m;
        drhods1 = w.dgamma1[i] * drho0ds1 + w.dgamma2[i] * ((drho1ds1 - drho1mds1) + (drho2ds1 - drho2mds1) + (drho3ds1 - drho3mds1));
        drhods2 = w.dgamma1[j] * drho0ds2 + w.dgamma2[j] * ((drho1ds2 - drho1mds2) + (drho2ds2 - drho2mds2) + (drho3ds2 - drho3mds2));
      }
      const frhopi = w.frhop[i], frhopj = w.frhop[j];
      const dUdrij = phip * sij + frhopi * drhodr1 + frhopj * drhodr2;
      let dUdsij = 0.0;
      if (!iszero(dscr)) dUdsij = phi + frhopi * drhods1 + frhopj * drhods2;
      const dUdrijm = [frhopi * drhodrm1[0] + frhopj * drhodrm2[0], frhopi * drhodrm1[1] + frhopj * drhodrm2[1], frhopi * drhodrm1[2] + frhopj * drhodrm2[2]];
      const force = dUdrij * recip + dUdsij * dscr;
      for (let m = 0; m < 3; m++) {
        const forcem = delij[m] * force + dUdrijm[m];
        f[3 * i + m] += forcem;
        f[3 * j + m] -= forcem;
      }
      if (iszero(sij) || isone(sij)) continue;
      addScreeningForces(c, w, f, jn, i, j, elti, eltj, f0, f1, rij2, dUdsij);
    }
  }
  return eng;
};
