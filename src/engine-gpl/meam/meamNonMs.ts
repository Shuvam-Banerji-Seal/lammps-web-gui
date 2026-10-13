/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM/meam_dens_final.cpp (the non-msmeam
 * branch), src/MEAM/meam_force.cpp (meam_force with msmeamflag = 0: the t-averaged densities and their derivatives,
 * ialloy = 0, 1 and 2) and the forward communication of pair_meam.cpp (comm_forward of the standard MEAM: rho0, t_ave
 * and tsq_ave in addition to the MS-MEAM set). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the
 * GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Original MEAM
 * authors (from meam_force.cpp and meam_dens_final.cpp): Greg Wagner (Sandia), Sungkwang Mun, Stephen Foiles, Michael I.
 * Baskes.
 *
 * pair_style meam (docs.lammps.org/pair_meam.html): the standard MEAM with the ialloy averaging of the t weights
 * ("0 = no averaging ... 1 = alternative averaging ... 2 = no averaging of t (use single-element values)"). The
 * density accumulation (meamMsCompute.ts densInit) is shared; the m-densities are zero without msmeamflag.
 */
import { fmExp, iszero, isone, getZij, getShpfcn } from './meamMath';
import type { MeamModel } from './meamModel';
import { ComputeInput, MeamWork, V2D, V3D, VIND2D, VIND3D, forwardGhosts } from './meamMsCompute';
import { addScreeningForces } from './meamMsForce';

const sq = (v: number): number => v * v;

/** fdiv_zero (meam.h): n / d, or 0 when d is zero. */
const fdivZero = (n: number, d: number): number => (iszero(d) ? 0.0 : n / d);

/** meam_dens_final (non-msmeam branch): gamma from the t-averaged densities; t_ave is normalised in place. */
export const densFinalNonMs = (c: ComputeInput, w: MeamWork): number => {
  const M = c.model;
  let eng = 0;
  for (let i = 0; i < c.nlocal; i++) {
    const elti = c.typeElem[c.type[i]];
    if (elti < 0) continue;
    w.rho1[i] = 0.0;
    w.rho2[i] = (-1.0 / 3.0) * sq(w.arho2b[i]);
    w.rho3[i] = 0.0;
    for (let m = 0; m < 3; m++) {
      w.rho1[i] = w.rho1[i] + sq(w.arho1[3 * i + m]);
      w.rho3[i] = w.rho3[i] - (3.0 / 5.0) * sq(w.arho3b[3 * i + m]);
    }
    for (let m = 0; m < 6; m++) w.rho2[i] = w.rho2[i] + V2D[m] * sq(w.arho2[6 * i + m]);
    for (let m = 0; m < 10; m++) w.rho3[i] = w.rho3[i] + V3D[m] * sq(w.arho3[10 * i + m]);
    if (w.rho0[i] > 0.0) {
      if (M.ialloy === 1) {
        for (let k = 0; k < 3; k++) w.tAve[3 * i + k] = fdivZero(w.tAve[3 * i + k], w.tsqAve[3 * i + k]);
      } else if (M.ialloy === 2) {
        w.tAve[3 * i] = M.t1[elti];
        w.tAve[3 * i + 1] = M.t2[elti];
        w.tAve[3 * i + 2] = M.t3[elti];
      } else {
        for (let k = 0; k < 3; k++) w.tAve[3 * i + k] = w.tAve[3 * i + k] / w.rho0[i];
      }
    }
    w.gamma[i] = w.tAve[3 * i] * w.rho1[i] + w.tAve[3 * i + 1] * w.rho2[i] + w.tAve[3 * i + 2] * w.rho3[i];
    if (w.rho0[i] > 0.0) w.gamma[i] = w.gamma[i] / (w.rho0[i] * w.rho0[i]);
    const Z = getZij(M.lattce[elti][elti]);
    const G = M.gGam(w.gamma[i], M.ibar[elti]);
    const shp = getShpfcn(M.lattce[elti][elti], M.stheta[elti][elti], M.ctheta[elti][elti]);
    let Gbar: number, dGbar = 0.0;
    if (M.ibar[elti] <= 0) {
      Gbar = 1.0;
    } else {
      const gam = M.mixRefT === 1
        ? (w.tAve[3 * i] * shp[0] + w.tAve[3 * i + 1] * shp[1] + w.tAve[3 * i + 2] * shp[2]) / (Z * Z)
        : (M.t1[elti] * shp[0] + M.t2[elti] * shp[1] + M.t3[elti] * shp[2]) / (Z * Z);
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
    eng += emb.F;
    if (c.eatom) c.eatom[i] += emb.F;
  }
  return eng;
};

/** Forward communication of the standard MEAM: the MS-MEAM set plus rho0, t_ave and tsq_ave (pair_meam.cpp). */
export const forwardNonMs = (c: ComputeInput, w: MeamWork): void => {
  forwardGhosts(c, w);
  const o = c.owner;
  for (let g = c.nlocal; g < c.nall; g++) {
    const from = o[g];
    w.rho0[g] = w.rho0[from];
    for (let q = 0; q < 3; q++) {
      w.tAve[3 * g + q] = w.tAve[3 * from + q];
      w.tsqAve[3 * g + q] = w.tsqAve[3 * from + q];
    }
  }
};

/** meam_force (msmeamflag = 0) for every owned atom: forces into c.f; returns the pair energy. */
export const forceNonMs = (c: ComputeInput, w: MeamWork, rdrar: number): number => {
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
      eng += phi * sij;
      if (c.eatom) {
        c.eatom[i] += 0.5 * phi * sij;
        c.eatom[j] += 0.5 * phi * sij;
      }
      // pair densities and derivatives (the t weights of ialloy = 1 multiply them below)
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
      let rhoa0j: number, drhoa0j: number, rhoa1j: number, drhoa1j: number, rhoa2j: number, drhoa2j: number;
      let rhoa3j: number, drhoa3j: number;
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
      } else {
        rhoa0j = rhoa0i; drhoa0j = drhoa0i;
        rhoa1j = rhoa1i; drhoa1j = drhoa1i;
        rhoa2j = rhoa2i; drhoa2j = drhoa2i;
        rhoa3j = rhoa3i; drhoa3j = drhoa3i;
      }
      const t1mi = M.t1[elti], t2mi = M.t2[elti], t3mi = M.t3[elti];
      const t1mj = M.t1[eltj], t2mj = M.t2[eltj], t3mj = M.t3[eltj];
      if (M.ialloy === 1) {
        rhoa1j *= t1mj; rhoa2j *= t2mj; rhoa3j *= t3mj;
        rhoa1i *= t1mi; rhoa2i *= t2mi; rhoa3i *= t3mi;
        drhoa1j *= t1mj; drhoa2j *= t2mj; drhoa3j *= t3mj;
        drhoa1i *= t1mi; drhoa2i *= t2mi; drhoa3i *= t3mi;
      }
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
      // t-average derivatives dt/dr of the averaged weights (meam_force.cpp, ialloy 1, 2 and 0)
      const t1i = w.tAve[3 * i], t2i = w.tAve[3 * i + 1], t3i = w.tAve[3 * i + 2];
      const t1j = w.tAve[3 * j], t2j = w.tAve[3 * j + 1], t3j = w.tAve[3 * j + 2];
      const dt = dtDerivs(M, w, i, j, elti, eltj, sij, drhoa0i, drhoa0j, rhoa0i, rhoa0j, [t1i, t2i, t3i], [t1j, t2j, t3j],
        [t1mi, t2mi, t3mi], [t1mj, t2mj, t3mj], 'dr');
      const shpi = getShpfcn(M.lattce[elti][elti], M.stheta[elti][elti], M.ctheta[elti][elti]);
      const shpj = getShpfcn(M.lattce[eltj][eltj], M.stheta[elti][elti], M.ctheta[elti][elti]);
      const drhodr1 = w.dgamma1[i] * drho0dr1 +
        w.dgamma2[i] * (dt.dt1[0] * w.rho1[i] + t1i * drho1dr1 + dt.dt2[0] * w.rho2[i] + t2i * drho2dr1 +
          dt.dt3[0] * w.rho3[i] + t3i * drho3dr1) -
        w.dgamma3[i] * (shpi[0] * dt.dt1[0] + shpi[1] * dt.dt2[0] + shpi[2] * dt.dt3[0]);
      const drhodr2 = w.dgamma1[j] * drho0dr2 +
        w.dgamma2[j] * (dt.dt1[1] * w.rho1[j] + t1j * drho1dr2 + dt.dt2[1] * w.rho2[j] + t2j * drho2dr2 +
          dt.dt3[1] * w.rho3[j] + t3j * drho3dr2) -
        w.dgamma3[j] * (shpj[0] * dt.dt1[1] + shpj[1] * dt.dt2[1] + shpj[2] * dt.dt3[1]);
      const drhodrm1 = [0, 0, 0], drhodrm2 = [0, 0, 0];
      for (let m = 0; m < 3; m++) {
        drhodrm1[m] = w.dgamma2[i] * (t1i * drho1drm1[m] + t2i * drho2drm1[m] + t3i * drho3drm1[m]);
        drhodrm2[m] = w.dgamma2[j] * (t1j * drho1drm2[m] + t2j * drho2drm2[m] + t3j * drho3drm2[m]);
      }
      const dscr = w.dscrfcn[jn];
      let drhods1 = 0, drhods2 = 0;
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
        const dts = dtDerivs(M, w, i, j, elti, eltj, 1.0, 0, 0, rhoa0i, rhoa0j, [t1i, t2i, t3i], [t1j, t2j, t3j],
          [t1mi, t2mi, t3mi], [t1mj, t2mj, t3mj], 'ds');
        drhods1 = w.dgamma1[i] * drho0ds1 +
          w.dgamma2[i] * (dts.dt1[0] * w.rho1[i] + t1i * drho1ds1 + dts.dt2[0] * w.rho2[i] + t2i * drho2ds1 +
            dts.dt3[0] * w.rho3[i] + t3i * drho3ds1) -
          w.dgamma3[i] * (shpi[0] * dts.dt1[0] + shpi[1] * dts.dt2[0] + shpi[2] * dts.dt3[0]);
        drhods2 = w.dgamma1[j] * drho0ds2 +
          w.dgamma2[j] * (dts.dt1[1] * w.rho1[j] + t1j * drho1ds2 + dts.dt2[1] * w.rho2[j] + t2j * drho2ds2 +
            dts.dt3[1] * w.rho3[j] + t3j * drho3ds2) -
          w.dgamma3[j] * (shpj[0] * dts.dt1[1] + shpj[1] * dts.dt2[1] + shpj[2] * dts.dt3[1]);
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

/**
 * The t-weight derivatives of meam_force.cpp for one side pair: dr (derivatives with respect to r_ij) uses the drhoa0 of
 * the pair, ds (screening) uses the unscaled rhoa0. dt1[0] is the derivative of atom i's average, dt1[1] of atom j's.
 */
const dtDerivs = (M: MeamModel, w: MeamWork, i: number, j: number, elti: number, eltj: number, sij: number,
  drhoa0i: number, drhoa0j: number, rhoa0i: number, rhoa0j: number, t_i: number[], t_j: number[], tmi: number[], tmj: number[],
  mode: 'dr' | 'ds'): { dt1: [number, number]; dt2: [number, number]; dt3: [number, number] } => {
  const dt1: [number, number] = [0, 0], dt2: [number, number] = [0, 0], dt3: [number, number] = [0, 0];
  // the derivative of rho0 enters as drhoa0 * sij (dr) or rhoa0 (ds)
  const di = mode === 'dr' ? drhoa0j * sij : rhoa0j; // atom i's neighbour j: rho0 of j contributes to i
  const dj = mode === 'dr' ? drhoa0i * sij : rhoa0i;
  const tk = [0, 1, 2];
  if (M.ialloy === 1) {
    for (const k of tk) {
      const tsqI = w.tsqAve[3 * i + k], tsqJ = w.tsqAve[3 * j + k];
      const a1i = fdivZero(di, tsqI);
      const a1j = fdivZero(dj, tsqJ);
      const tmjk = tmj[k], tmik = tmi[k];
      const val1 = a1i * (tmjk - t_i[k] * tmjk * tmjk);
      const val2 = a1j * (tmik - t_j[k] * tmik * tmik);
      if (k === 0) { dt1[0] = val1; dt1[1] = val2; }
      else if (k === 1) { dt2[0] = val1; dt2[1] = val2; }
      else { dt3[0] = val1; dt3[1] = val2; }
    }
  } else if (M.ialloy === 2) {
    // no averaging: all derivatives vanish
  } else {
    const ai = !iszero(w.rho0[i]) ? di / w.rho0[i] : 0.0;
    const aj = !iszero(w.rho0[j]) ? dj / w.rho0[j] : 0.0;
    for (const k of tk) {
      const v1 = ai * (tmj[k] - t_i[k]);
      const v2 = aj * (tmi[k] - t_j[k]);
      if (k === 0) { dt1[0] = v1; dt1[1] = v2; }
      else if (k === 1) { dt2[0] = v1; dt2[1] = v2; }
      else { dt3[0] = v1; dt3[1] = v2; }
    }
  }
  return { dt1, dt2, dt3 };
};
