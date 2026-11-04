/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/pair_gran_hooke_history.cpp (PairGranHookeHistory::compute), pair_gran_hertz_history.cpp
 * (PairGranHertzHistory::compute) and pair_gran_hooke.cpp (PairGranHooke::compute) at commit b332f28. Contributing
 * authors: Leo Silbert (SNL), Gary Grest (SNL). Copyright (2003) Sandia Corporation (GNU General Public License). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style gran/hooke, gran/hooke/history and gran/hertz/history with native's force arithmetic in native's operation
 * order: the normal force with limit_damping clamping ccel to 0 (the tangential part is still computed), the shear
 * update, its magnitude before the rotation into the tangent plane, the tangential force, the Coulomb rescale of the
 * shear ("(fn/fs) * (shear + meff*gammat*vtr/kt) - meff*gammat*vtr/kt") and of the force, and the forces and torques
 * scaled by the special_bonds LJ factor. gran/hooke has no history: ft = MIN(fn, fs) / vrel with fs = meff gammat vrel.
 * The settings, the neighbor list, the shear store per atom-ID pair (oriented from the lower ID) and single() are those
 * of the engine's PairGran (src/engine/force/pair/gran.ts), which this class extends.
 */
import { PairGran, pairKey } from '../../engine/force/pair/gran';
import { tallyAtomXyz, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { contactMassArray } from '../../engine/atoms';

export class PairGranNative extends PairGran {
  override compute(pc: PairCompute): void {
    const s = pc.s;
    const list = pc.half!;
    const { x, f } = pc;
    const nb = pc.nb;
    const nall = pc.nall;
    const owner = nb.owner;
    const radius = s.radius!, rmass = contactMassArray(s), omega = s.omega!, v = s.v, id = s.id, mask = s.mask;
    const dt = s.dt;
    const shearupdate = pc.historyUpdate === true;
    if (this.tq.length < 3 * nall) this.tq = new Float64Array(3 * nall);
    const tq = this.tq;
    tq.fill(0, 0, 3 * nall);
    const history = this.history, hertz = this.hertz;
    const seen = history ? new Set<number | string>() : null;
    const kn = this.kn, kt = this.kt, gamman = this.gn, gammat = this.gt, xmu = this.xmu;
    const limit_damping = this.limitDamping;
    const freeze_group_bit = this.freezeBit;
    const special_lj = pc.specialLJ;
    for (let ii = 0; ii < list.inum; ii++) {
      const oi = owner[ii];
      const xtmp = x[3 * ii], ytmp = x[3 * ii + 1], ztmp = x[3 * ii + 2];
      const radi = radius[oi];
      for (let k = list.firstneigh[ii], k1 = k + list.numneigh[ii]; k < k1; k++) {
        const jraw = list.neighbors[k];
        const factor_lj = special_lj[jraw >>> SBBITS];
        const j = jraw & NEIGHMASK;
        if (factor_lj === 0) continue;
        const oj = owner[j];
        const delx = xtmp - x[3 * j];
        const dely = ytmp - x[3 * j + 1];
        const delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const radj = radius[oj];
        const radsum = radi + radj;
        // a pair out of contact keeps no shear (native unsets touch and zeroes the shear): its entry is not seen this call
        if (rsq >= radsum * radsum) continue;
        const r = Math.sqrt(rsq);
        const rinv = 1.0 / r;
        const rsqinv = 1.0 / rsq;

        // relative translational velocity
        const vr1 = v[3 * oi] - v[3 * oj];
        const vr2 = v[3 * oi + 1] - v[3 * oj + 1];
        const vr3 = v[3 * oi + 2] - v[3 * oj + 2];

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
        const wr1 = (radi * omega[3 * oi] + radj * omega[3 * oj]) * rinv;
        const wr2 = (radi * omega[3 * oi + 1] + radj * omega[3 * oj + 1]) * rinv;
        const wr3 = (radi * omega[3 * oi + 2] + radj * omega[3 * oj + 2]) * rinv;

        // meff = effective mass of pair of particles (the body mass of a rigid body); a frozen I or J gives the other's
        const mi = rmass[oi];
        const mj = rmass[oj];
        let meff = mi * mj / (mi + mj);
        if (mask[oi] & freeze_group_bit) meff = mj;
        if (mask[oj] & freeze_group_bit) meff = mi;

        // normal force = Hookian (Hertzian) contact + normal velocity damping
        const damp = meff * gamman * vnnr * rsqinv;
        let ccel = kn * (radsum - r) * rinv - damp;
        let polyhertz = 1.0;
        if (hertz) {
          polyhertz = Math.sqrt((radsum - r) * radi * radj / radsum);
          ccel *= polyhertz;
        }
        if (limit_damping && (ccel < 0.0)) ccel = 0.0;

        // relative velocities
        const vtr1 = vt1 - (delz * wr2 - dely * wr3);
        const vtr2 = vt2 - (delx * wr3 - delz * wr1);
        const vtr3 = vt3 - (dely * wr1 - delx * wr2);

        let fs1: number, fs2: number, fs3: number;
        if (!history) {
          // force normalization
          let vrel = vtr1 * vtr1 + vtr2 * vtr2 + vtr3 * vtr3;
          vrel = Math.sqrt(vrel);
          const fn = xmu * Math.abs(ccel * r);
          const fs = meff * gammat * vrel;
          let ft: number;
          if (vrel !== 0.0) ft = (fn < fs ? fn : fs) / vrel;
          else ft = 0.0;
          // tangential force due to tangential velocity damping
          fs1 = -ft * vtr1;
          fs2 = -ft * vtr2;
          fs3 = -ft * vtr3;
        } else {
          // shear history effects (the store is oriented as seen from the lower atom ID)
          const a = id[oi], b = id[oj];
          const flip = a > b;
          const key = flip ? pairKey(b, a) : pairKey(a, b);
          seen!.add(key);
          const sh = this.shear.get(key) ?? new Float64Array(3);
          const sg = flip ? -1 : 1;
          let shear0 = sg * sh[0], shear1 = sg * sh[1], shear2 = sg * sh[2];
          if (shearupdate) {
            shear0 += vtr1 * dt;
            shear1 += vtr2 * dt;
            shear2 += vtr3 * dt;
          }
          const shrmag = Math.sqrt(shear0 * shear0 + shear1 * shear1 + shear2 * shear2);
          if (shearupdate) {
            // rotate shear displacements
            let rsht = shear0 * delx + shear1 * dely + shear2 * delz;
            rsht *= rsqinv;
            shear0 -= rsht * delx;
            shear1 -= rsht * dely;
            shear2 -= rsht * delz;
          }
          // tangential forces = shear + tangential velocity damping
          if (hertz) {
            fs1 = -polyhertz * (kt * shear0 + meff * gammat * vtr1);
            fs2 = -polyhertz * (kt * shear1 + meff * gammat * vtr2);
            fs3 = -polyhertz * (kt * shear2 + meff * gammat * vtr3);
          } else {
            fs1 = -(kt * shear0 + meff * gammat * vtr1);
            fs2 = -(kt * shear1 + meff * gammat * vtr2);
            fs3 = -(kt * shear2 + meff * gammat * vtr3);
          }
          // rescale frictional displacements and forces if needed
          const fs = Math.sqrt(fs1 * fs1 + fs2 * fs2 + fs3 * fs3);
          const fn = xmu * Math.abs(ccel * r);
          if (fs > fn) {
            if (shrmag !== 0.0) {
              shear0 = (fn / fs) * (shear0 + meff * gammat * vtr1 / kt) - meff * gammat * vtr1 / kt;
              shear1 = (fn / fs) * (shear1 + meff * gammat * vtr2 / kt) - meff * gammat * vtr2 / kt;
              shear2 = (fn / fs) * (shear2 + meff * gammat * vtr3 / kt) - meff * gammat * vtr3 / kt;
              fs1 *= fn / fs;
              fs2 *= fn / fs;
              fs3 *= fn / fs;
            } else fs1 = fs2 = fs3 = 0.0;
          }
          sh[0] = sg * shear0; sh[1] = sg * shear1; sh[2] = sg * shear2;
          this.shear.set(key, sh);
        }

        // forces & torques
        let fx = delx * ccel + fs1;
        let fy = dely * ccel + fs2;
        let fz = delz * ccel + fs3;
        fx *= factor_lj;
        fy *= factor_lj;
        fz *= factor_lj;
        f[3 * ii] += fx;
        f[3 * ii + 1] += fy;
        f[3 * ii + 2] += fz;

        let tor1 = rinv * (dely * fs3 - delz * fs2);
        let tor2 = rinv * (delz * fs1 - delx * fs3);
        let tor3 = rinv * (delx * fs2 - dely * fs1);
        tor1 *= factor_lj;
        tor2 *= factor_lj;
        tor3 *= factor_lj;
        tq[3 * ii] -= radi * tor1;
        tq[3 * ii + 1] -= radi * tor2;
        tq[3 * ii + 2] -= radi * tor3;

        f[3 * j] -= fx;
        f[3 * j + 1] -= fy;
        f[3 * j + 2] -= fz;
        tq[3 * j] -= radj * tor1;
        tq[3 * j + 1] -= radj * tor2;
        tq[3 * j + 2] -= radj * tor3;

        // ev_tally_xyz with no energy (the per-atom virial of the contact)
        if (pc.vatom) tallyAtomXyz(pc, ii, j, 0, fx, fy, fz, delx, dely, delz);
      }
    }
    if (history) for (const key of this.shear.keys()) if (!seen!.has(key)) this.shear.delete(key);
    nb.reverseSum(tq, 3, s.torque!);
  }
}
