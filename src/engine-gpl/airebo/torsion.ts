/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.cpp (PairAIREBO::TORSION), at commit b332f28.
 * The torsion energy is the C-C-C-C dihedral term with the cosine switch and the
 * C-C-C-H-free form of the C++ text; forces follow the C++ operation order.
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { sp, sp2 } from './splines';
import { powint } from './splines';
import { TOL, type ReboCore } from './rebo';
import { type LJCore } from './flj';

const v3 = (): Float64Array => new Float64Array(3);

/** PairAIREBO::TORSION: the torsion term of C-C bonds (owned central atoms, REBO neighbors). */
export const torsion = (core: LJCore & ReboCore, nlocal: number): void => {
  const { P, mt, tag, x, f, rstart, rlist, epsilonT } = core;
  const pidx = (a: number, b: number) => a * 2 + b;
  const eatom = core.eatom;
  const del32 = v3(), del23 = v3(), del21 = v3(), del34 = v3(), deljk = v3(), delil = v3(), delkl = v3();
  const cross321 = v3(), cross234 = v3(), dndij = v3(), tmpvec = v3(), dndik = v3(), dndjl = v3();
  const Ftmp = v3(), fi = v3(), fj = v3(), fk = v3(), fl = v3();
  const sw: { v: number } = { v: 0 };
  void sw;
  for (let i = 0; i < nlocal; i++) {
    const itag = tag[i];
    const itype = mt[i];
    if (itype !== 0) continue;
    const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
    for (let jj = rstart[i]; jj < rstart[i + 1]; jj++) {
      const j = rlist[jj];
      const jtag = tag[j];
      if (itag > jtag) {
        if ((itag + jtag) % 2 === 0) continue;
      } else if (itag < jtag) {
        if ((itag + jtag) % 2 === 1) continue;
      } else {
        if (x[3 * j + 2] < ztmp) continue;
        if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
        if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
      }
      const jtype = mt[j];
      if (jtype !== 0) continue;
      del32[0] = x[3 * j] - x[3 * i];
      del32[1] = x[3 * j + 1] - x[3 * i + 1];
      del32[2] = x[3 * j + 2] - x[3 * i + 2];
      let rsq = del32[0] * del32[0] + del32[1] * del32[1] + del32[2] * del32[2];
      const r32 = Math.sqrt(rsq);
      del23[0] = -del32[0];
      del23[1] = -del32[1];
      del23[2] = -del32[2];
      const r23 = r32;
      const dw23 = { v: 0 };
      const w23 = sp(r23, P.rcmin[pidx(itype, jtype)], P.rcmax[pidx(itype, jtype)], dw23);
      for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
        const k = rlist[kk];
        const ktype = mt[k];
        if (k === j) continue;
        del21[0] = x[3 * i] - x[3 * k];
        del21[1] = x[3 * i + 1] - x[3 * k + 1];
        del21[2] = x[3 * i + 2] - x[3 * k + 2];
        rsq = del21[0] * del21[0] + del21[1] * del21[1] + del21[2] * del21[2];
        const r21 = Math.sqrt(rsq);
        let cos321 = -((del21[0] * del32[0]) + (del21[1] * del32[1]) + (del21[2] * del32[2])) / (r21 * r32);
        cos321 = Math.min(cos321, 1.0);
        cos321 = Math.max(cos321, -1.0);
        const sin321 = Math.sqrt(1.0 - cos321 * cos321);
        if (sin321 < TOL) continue;
        deljk[0] = del21[0] - del23[0];
        deljk[1] = del21[1] - del23[1];
        deljk[2] = del21[2] - del23[2];
        const rjk2 = deljk[0] * deljk[0] + deljk[1] * deljk[1] + deljk[2] * deljk[2];
        const rjk = Math.sqrt(rjk2);
        const dw21 = { v: 0 };
        const w21 = sp(r21, P.rcmin[pidx(itype, ktype)], P.rcmax[pidx(itype, ktype)], dw21);
        const rij = r32;
        const rik = r21;
        const rij2 = r32 * r32;
        const rik2 = r21 * r21;
        let costmp = 0.5 * (rij2 + rik2 - rjk2) / rij / rik;
        const dtsjikRef = { v: 0 };
        const tspjik = sp2(costmp, P.thmin, P.thmax, dtsjikRef);
        const dtsjik = -dtsjikRef.v;
        for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
          const l = rlist[ll];
          const ltype = mt[l];
          if (l === i || l === k) continue;
          del34[0] = x[3 * j] - x[3 * l];
          del34[1] = x[3 * j + 1] - x[3 * l + 1];
          del34[2] = x[3 * j + 2] - x[3 * l + 2];
          rsq = del34[0] * del34[0] + del34[1] * del34[1] + del34[2] * del34[2];
          const r34 = Math.sqrt(rsq);
          let cos234 = (del32[0] * del34[0] + del32[1] * del34[1] + del32[2] * del34[2]) / (r32 * r34);
          cos234 = Math.min(cos234, 1.0);
          cos234 = Math.max(cos234, -1.0);
          const sin234 = Math.sqrt(1.0 - cos234 * cos234);
          if (sin234 < TOL) continue;
          const dw34 = { v: 0 };
          const w34 = sp(r34, P.rcmin[pidx(jtype, ltype)], P.rcmax[pidx(jtype, ltype)], dw34);
          delil[0] = del23[0] + del34[0];
          delil[1] = del23[1] + del34[1];
          delil[2] = del23[2] + del34[2];
          const ril2 = delil[0] * delil[0] + delil[1] * delil[1] + delil[2] * delil[2];
          const ril = Math.sqrt(ril2);
          const rjl = r34;
          const rjl2 = r34 * r34;
          costmp = 0.5 * (rij2 + rjl2 - ril2) / rij / rjl;
          const dtsijlRef = { v: 0 };
          const tspijl = sp2(costmp, P.thmin, P.thmax, dtsijlRef);
          const dtsijl = -dtsijlRef.v;
          cross321[0] = (del32[1] * del21[2]) - (del32[2] * del21[1]);
          cross321[1] = (del32[2] * del21[0]) - (del32[0] * del21[2]);
          cross321[2] = (del32[0] * del21[1]) - (del32[1] * del21[0]);
          const cross321mag = Math.sqrt(cross321[0] * cross321[0] + cross321[1] * cross321[1] + cross321[2] * cross321[2]);
          cross234[0] = (del23[1] * del34[2]) - (del23[2] * del34[1]);
          cross234[1] = (del23[2] * del34[0]) - (del23[0] * del34[2]);
          cross234[2] = (del23[0] * del34[1]) - (del23[1] * del34[0]);
          const cross234mag = Math.sqrt(cross234[0] * cross234[0] + cross234[1] * cross234[1] + cross234[2] * cross234[2]);
          const cwnum = (cross321[0] * cross234[0]) + (cross321[1] * cross234[1]) + (cross321[2] * cross234[2]);
          const cwnom = r21 * r34 * r32 * r32 * sin321 * sin234;
          const cw = cwnum / cwnom;
          const cw2 = (.5 * (1.0 - cw));
          const ekijl = epsilonT[pidx(ktype, ltype)];
          const Ec = 256.0 * ekijl / 405.0;
          const Vtors = (Ec * (powint(cw2, 5))) - (ekijl / 10.0);
          const evdwl = Vtors * w21 * w23 * w34 * (1.0 - tspjik) * (1.0 - tspijl);
          core.acc.evdwl += evdwl;
          dndij[0] = (cross234[1] * del21[2]) - (cross234[2] * del21[1]);
          dndij[1] = (cross234[2] * del21[0]) - (cross234[0] * del21[2]);
          dndij[2] = (cross234[0] * del21[1]) - (cross234[1] * del21[0]);
          tmpvec[0] = (del34[1] * cross321[2]) - (del34[2] * cross321[1]);
          tmpvec[1] = (del34[2] * cross321[0]) - (del34[0] * cross321[2]);
          tmpvec[2] = (del34[0] * cross321[1]) - (del34[1] * cross321[0]);
          dndij[0] = dndij[0] + tmpvec[0];
          dndij[1] = dndij[1] + tmpvec[1];
          dndij[2] = dndij[2] + tmpvec[2];
          dndik[0] = (del23[1] * cross234[2]) - (del23[2] * cross234[1]);
          dndik[1] = (del23[2] * cross234[0]) - (del23[0] * cross234[2]);
          dndik[2] = (del23[0] * cross234[1]) - (del23[1] * cross234[0]);
          dndjl[0] = (cross321[1] * del23[2]) - (cross321[2] * del23[1]);
          dndjl[1] = (cross321[2] * del23[0]) - (cross321[0] * del23[2]);
          dndjl[2] = (cross321[0] * del23[1]) - (cross321[1] * del23[0]);
          const dcidij = ((r23 * r23) - (r21 * r21) + (rjk * rjk)) / (2.0 * r23 * r23 * r21);
          const dcidik = ((r21 * r21) - (r23 * r23) + (rjk * rjk)) / (2.0 * r23 * r21 * r21);
          const dcidjk = (-rjk) / (r23 * r21);
          const dcjdji = ((r23 * r23) - (r34 * r34) + (ril * ril)) / (2.0 * r23 * r23 * r34);
          const dcjdjl = ((r34 * r34) - (r23 * r23) + (ril * ril)) / (2.0 * r23 * r34 * r34);
          const dcjdil = (-ril) / (r23 * r34);
          const dsidij = (-cos321 / sin321) * dcidij;
          const dsidik = (-cos321 / sin321) * dcidik;
          const dsidjk = (-cos321 / sin321) * dcidjk;
          const dsjdji = (-cos234 / sin234) * dcjdji;
          const dsjdjl = (-cos234 / sin234) * dcjdjl;
          const dsjdil = (-cos234 / sin234) * dcjdil;
          const dxidij = (r21 * sin321) + (r23 * r21 * dsidij);
          const dxidik = (r23 * sin321) + (r23 * r21 * dsidik);
          const dxidjk = (r23 * r21 * dsidjk);
          const dxjdji = (r34 * sin234) + (r23 * r34 * dsjdji);
          const dxjdjl = (r23 * sin234) + (r23 * r34 * dsjdjl);
          const dxjdil = (r23 * r34 * dsjdil);
          const ddndij = (dxidij * cross234mag) + (cross321mag * dxjdji);
          const ddndik = dxidik * cross234mag;
          const ddndjk = dxidjk * cross234mag;
          const ddndjl = cross321mag * dxjdjl;
          const ddndil = cross321mag * dxjdil;
          const dcwddn = -cwnum / (cwnom * cwnom);
          const dcwdn = 1.0 / cwnom;
          const dvpdcw = (-1.0) * Ec * (-.5) * 5.0 * powint(cw2, 4) * w23 * w21 * w34 * (1.0 - tspjik) * (1.0 - tspijl);
          Ftmp[0] = dvpdcw * ((dcwdn * dndij[0]) + (dcwddn * ddndij * del23[0] / r23));
          Ftmp[1] = dvpdcw * ((dcwdn * dndij[1]) + (dcwddn * ddndij * del23[1] / r23));
          Ftmp[2] = dvpdcw * ((dcwdn * dndij[2]) + (dcwddn * ddndij * del23[2] / r23));
          fi[0] = Ftmp[0];
          fi[1] = Ftmp[1];
          fi[2] = Ftmp[2];
          fj[0] = -Ftmp[0];
          fj[1] = -Ftmp[1];
          fj[2] = -Ftmp[2];
          Ftmp[0] = dvpdcw * ((dcwdn * dndik[0]) + (dcwddn * ddndik * del21[0] / r21));
          Ftmp[1] = dvpdcw * ((dcwdn * dndik[1]) + (dcwddn * ddndik * del21[1] / r21));
          Ftmp[2] = dvpdcw * ((dcwdn * dndik[2]) + (dcwddn * ddndik * del21[2] / r21));
          fi[0] += Ftmp[0];
          fi[1] += Ftmp[1];
          fi[2] += Ftmp[2];
          fk[0] = -Ftmp[0];
          fk[1] = -Ftmp[1];
          fk[2] = -Ftmp[2];
          Ftmp[0] = (dvpdcw * dcwddn * ddndjk * deljk[0]) / rjk;
          Ftmp[1] = (dvpdcw * dcwddn * ddndjk * deljk[1]) / rjk;
          Ftmp[2] = (dvpdcw * dcwddn * ddndjk * deljk[2]) / rjk;
          fj[0] += Ftmp[0];
          fj[1] += Ftmp[1];
          fj[2] += Ftmp[2];
          fk[0] -= Ftmp[0];
          fk[1] -= Ftmp[1];
          fk[2] -= Ftmp[2];
          Ftmp[0] = dvpdcw * ((dcwdn * dndjl[0]) + (dcwddn * ddndjl * del34[0] / r34));
          Ftmp[1] = dvpdcw * ((dcwdn * dndjl[1]) + (dcwddn * ddndjl * del34[1] / r34));
          Ftmp[2] = dvpdcw * ((dcwdn * dndjl[2]) + (dcwddn * ddndjl * del34[2] / r34));
          fj[0] += Ftmp[0];
          fj[1] += Ftmp[1];
          fj[2] += Ftmp[2];
          fl[0] = -Ftmp[0];
          fl[1] = -Ftmp[1];
          fl[2] = -Ftmp[2];
          Ftmp[0] = (dvpdcw * dcwddn * ddndil * delil[0]) / ril;
          Ftmp[1] = (dvpdcw * dcwddn * ddndil * delil[1]) / ril;
          Ftmp[2] = (dvpdcw * dcwddn * ddndil * delil[2]) / ril;
          fi[0] += Ftmp[0];
          fi[1] += Ftmp[1];
          fi[2] += Ftmp[2];
          fl[0] -= Ftmp[0];
          fl[1] -= Ftmp[1];
          fl[2] -= Ftmp[2];
          let fpair = Vtors * dw21.v * w23 * w34 * (1.0 - tspjik) * (1.0 - tspijl) / r21;
          fi[0] -= del21[0] * fpair;
          fi[1] -= del21[1] * fpair;
          fi[2] -= del21[2] * fpair;
          fk[0] += del21[0] * fpair;
          fk[1] += del21[1] * fpair;
          fk[2] += del21[2] * fpair;
          fpair = Vtors * w21 * dw23.v * w34 * (1.0 - tspjik) * (1.0 - tspijl) / r23;
          fi[0] -= del23[0] * fpair;
          fi[1] -= del23[1] * fpair;
          fi[2] -= del23[2] * fpair;
          fj[0] += del23[0] * fpair;
          fj[1] += del23[1] * fpair;
          fj[2] += del23[2] * fpair;
          fpair = Vtors * w21 * w23 * dw34.v * (1.0 - tspjik) * (1.0 - tspijl) / r34;
          fj[0] -= del34[0] * fpair;
          fj[1] -= del34[1] * fpair;
          fj[2] -= del34[2] * fpair;
          fl[0] += del34[0] * fpair;
          fl[1] += del34[1] * fpair;
          fl[2] += del34[2] * fpair;
          let fcpc = -Vtors * w21 * w23 * w34 * dtsjik * (1.0 - tspijl);
          fpair = fcpc * dcidij / rij;
          fi[0] += fpair * del23[0];
          fi[1] += fpair * del23[1];
          fi[2] += fpair * del23[2];
          fj[0] -= fpair * del23[0];
          fj[1] -= fpair * del23[1];
          fj[2] -= fpair * del23[2];
          fpair = fcpc * dcidik / rik;
          fi[0] += fpair * del21[0];
          fi[1] += fpair * del21[1];
          fi[2] += fpair * del21[2];
          fk[0] -= fpair * del21[0];
          fk[1] -= fpair * del21[1];
          fk[2] -= fpair * del21[2];
          fpair = fcpc * dcidjk / rjk;
          fj[0] += fpair * deljk[0];
          fj[1] += fpair * deljk[1];
          fj[2] += fpair * deljk[2];
          fk[0] -= fpair * deljk[0];
          fk[1] -= fpair * deljk[1];
          fk[2] -= fpair * deljk[2];
          fcpc = -Vtors * w21 * w23 * w34 * (1.0 - tspjik) * dtsijl;
          fpair = fcpc * dcjdji / rij;
          fi[0] += fpair * del23[0];
          fi[1] += fpair * del23[1];
          fi[2] += fpair * del23[2];
          fj[0] -= fpair * del23[0];
          fj[1] -= fpair * del23[1];
          fj[2] -= fpair * del23[2];
          fpair = fcpc * dcjdjl / rjl;
          fj[0] += fpair * del34[0];
          fj[1] += fpair * del34[1];
          fj[2] += fpair * del34[2];
          fl[0] -= fpair * del34[0];
          fl[1] -= fpair * del34[1];
          fl[2] -= fpair * del34[2];
          fpair = fcpc * dcjdil / ril;
          fi[0] += fpair * delil[0];
          fi[1] += fpair * delil[1];
          fi[2] += fpair * delil[2];
          fl[0] -= fpair * delil[0];
          fl[1] -= fpair * delil[1];
          fl[2] -= fpair * delil[2];
          f[3 * i] += fi[0]; f[3 * i + 1] += fi[1]; f[3 * i + 2] += fi[2];
          f[3 * j] += fj[0]; f[3 * j + 1] += fj[1]; f[3 * j + 2] += fj[2];
          f[3 * k] += fk[0]; f[3 * k + 1] += fk[1]; f[3 * k + 2] += fk[2];
          f[3 * l] += fl[0]; f[3 * l + 1] += fl[1]; f[3 * l + 2] += fl[2];
          if (eatom) {
            // ev_tally4: one quarter of the energy to each of the four atoms
            const q = 0.25 * evdwl;
            eatom[i] += q;
            eatom[j] += q;
            eatom[k] += q;
            eatom[l] += q;
          }
          void delkl;
        }
      }
    }
  }
};
