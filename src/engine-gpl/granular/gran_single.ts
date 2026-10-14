/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the single() functions of
 * src/GRANULAR/pair_gran_hooke.cpp, src/GRANULAR/pair_gran_hooke_history.cpp and src/GRANULAR/pair_gran_hertz_history.cpp
 * (the normal force, the tangential force with the stored shear and the svector of the 10 pN quantities of compute
 * pair/local). Contributing authors (from the sources): Leo Silbert (SNL), Gary Grest (SNL).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entries 'gran/hooke', 'gran/hooke/history' and 'gran/hertz/history' (lazyStyles.ts group
 * gpl_gran_single) give the core pair gran (src/engine/force/pair/gran.ts) its native single(): the normal force
 * fforce and the svector (native single_extra = 10). The history styles read the stored shear and do not update it
 * (native single() calls are made with the history of the last force evaluation).
 */
import type { GranKind, GranSingleHook, GranSingleInput, GranSingleParams } from '../../engine/force/pair/gran';

const outside = (svector: Float64Array): number => {
  // native: "if (rsq >= radsum*radsum) { fforce = 0.0; for (m...) svector[m] = 0.0; return 0.0; }"
  svector.fill(0);
  return 0;
};

/**
 * The native single() body of the three gran styles. The relative rotational velocity wr, the tangential relative
 * velocity vtr (vt minus the rotation term) and the normal and tangential parts of v_r are those of the native code.
 */
export const granSingle = (kind: GranKind, p: GranSingleParams, inp: GranSingleInput, svector: Float64Array): number => {
  const radsum = inp.radi + inp.radj;
  if (inp.rsq >= radsum * radsum) return outside(svector);
  const hertz = kind === 'gran/hertz/history';
  const history = kind !== 'gran/hooke';
  const r = Math.sqrt(inp.rsq);
  const rinv = 1.0 / r;
  const rsqinv = 1.0 / inp.rsq;
  const { delx, dely, delz } = inp;
  const vr1 = inp.vr[0], vr2 = inp.vr[1], vr3 = inp.vr[2];

  // normal component
  const vnnr = vr1 * delx + vr2 * dely + vr3 * delz;
  const vn1 = delx * vnnr * rsqinv;
  const vn2 = dely * vnnr * rsqinv;
  const vn3 = delz * vnnr * rsqinv;
  // tangential component
  const vt1 = vr1 - vn1;
  const vt2 = vr2 - vn2;
  const vt3 = vr3 - vn3;
  // relative rotational velocity
  const wr1 = (inp.radi * inp.omegai[0] + inp.radj * inp.omegaj[0]) * rinv;
  const wr2 = (inp.radi * inp.omegai[1] + inp.radj * inp.omegaj[1]) * rinv;
  const wr3 = (inp.radi * inp.omegai[2] + inp.radj * inp.omegaj[2]) * rinv;

  // meff: "if I is frozen, meff is other particle", then the same for J (native order, both tests applied)
  let meff = (inp.mi * inp.mj) / (inp.mi + inp.mj);
  if (inp.freezeI) meff = inp.mj;
  if (inp.freezeJ) meff = inp.mi;

  // normal forces = Hookian contact + normal velocity damping (the Hertzian factor scales the whole normal force)
  const damp = meff * p.gamman * vnnr * rsqinv;
  const poly = hertz ? Math.sqrt(((radsum - r) * inp.radi * inp.radj) / radsum) : 1;
  let ccel = (p.kn * (radsum - r) * rinv - damp) * poly;
  if (p.limitDamping && ccel < 0.0) ccel = 0.0;

  // relative velocities
  const vtr1 = vt1 - (delz * wr2 - dely * wr3);
  const vtr2 = vt2 - (delx * wr3 - delz * wr1);
  const vtr3 = vt3 - (dely * wr1 - delx * wr2);
  const vrel = Math.sqrt(vtr1 * vtr1 + vtr2 * vtr2 + vtr3 * vtr3);

  if (!history) {
    // gran/hooke: the tangential force is the damping force, limited by the Coulomb value fn (native ft = min(fn, fs) / vrel)
    const fn = p.xmu * Math.abs(ccel * r);
    const fsd = meff * p.gammat * vrel;
    const ft = vrel !== 0.0 ? Math.min(fn, fsd) / vrel : 0.0;
    svector[0] = -ft * vtr1;
    svector[1] = -ft * vtr2;
    svector[2] = -ft * vtr3;
    svector[3] = Math.sqrt(svector[0] * svector[0] + svector[1] * svector[1] + svector[2] * svector[2]);
  } else {
    const sh = inp.shear;
    const shrmag = Math.sqrt(sh[0] * sh[0] + sh[1] * sh[1] + sh[2] * sh[2]);
    // tangential forces = shear + tangential velocity damping (hertz: scaled by the Hertzian factor)
    let fs1 = -poly * (p.kt * sh[0] + meff * p.gammat * vtr1);
    let fs2 = -poly * (p.kt * sh[1] + meff * p.gammat * vtr2);
    let fs3 = -poly * (p.kt * sh[2] + meff * p.gammat * vtr3);
    // rescale frictional forces if needed (native: with no shear the force is dropped altogether)
    let fs = Math.sqrt(fs1 * fs1 + fs2 * fs2 + fs3 * fs3);
    const fn = p.xmu * Math.abs(ccel * r);
    if (fs > fn) {
      if (shrmag !== 0.0) {
        fs1 *= fn / fs;
        fs2 *= fn / fs;
        fs3 *= fn / fs;
        fs *= fn / fs;
      } else {
        fs1 = fs2 = fs3 = fs = 0.0;
      }
    }
    svector[0] = fs1;
    svector[1] = fs2;
    svector[2] = fs3;
    svector[3] = fs;
  }
  // native: svector[4..6] = vn, svector[7..9] = vt (the tangential velocity without the rotation term, as native does)
  svector[4] = vn1;
  svector[5] = vn2;
  svector[6] = vn3;
  svector[7] = vt1;
  svector[8] = vt2;
  svector[9] = vt3;
  return ccel;
};
