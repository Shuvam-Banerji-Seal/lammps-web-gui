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
 * The settings, the neighbor list and single() are those of the engine's PairGran (src/engine/force/pair/gran.ts), which
 * this class extends. The shear history is kept as native's fix neigh/history keeps it (src/fix_neigh_history.cpp at
 * b332f28): with the entries of the neighbor list, three values per entry (oriented from the lower atom ID, as the
 * engine's store), carried over by atom-ID pair when the list is built again; an entry out of contact holds none.
 */
import { PairGran, pairKey } from '../../engine/force/pair/gran';
import { tallyAtomXyz, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS, type NeighList } from '../../engine/neighbor';
import { contactMassArray } from '../../engine/atoms';

export class PairGranNative extends PairGran {
  /** The neighbor list (and its build) the history slots follow. */
  private slotList: NeighList | null = null;
  private slotBuild = -1;
  /** Three shear values per list entry (lower-ID frame), and the entry's atom IDs (lower, higher; 0: no contact). */
  private slotShear = new Float64Array(0);
  private slotLo = new Int32Array(0);
  private slotHi = new Int32Array(0);
  /** single()'s index of the entries in contact by atom-ID pair (built on demand after a compute). */
  private slotIndex: Map<number | string, number> | null = null;

  override settings(args: string[]): void {
    super.settings(args);
    this.slotList = null;
    this.slotBuild = -1;
    this.slotLo = new Int32Array(0);
    this.slotHi = new Int32Array(0);
    this.slotShear = new Float64Array(0);
    this.slotIndex = null;
  }

  /**
   * The slots of a newly built list: each entry takes the record of its atom-ID pair from the previous list (native's
   * fix neigh/history pre_exchange and post_neighbor: the partners and their values stored per atom, then copied into
   * the new list's entries). The previous indices refer to the previous atoms and ghosts, so the IDs of each record are
   * kept with it.
   */
  private ensureSlots(list: NeighList, nbuild: number, id: ArrayLike<number>, owner: Int32Array): void {
    if (list === this.slotList && nbuild === this.slotBuild) return;
    const oldShear = this.slotShear, oldLo = this.slotLo, oldHi = this.slotHi;
    let prev: Map<number | string, number> | null = null;
    for (let k = 0; k < oldLo.length; k++) if (oldLo[k] !== 0) (prev ??= new Map()).set(pairKey(oldLo[k], oldHi[k]), k);
    const nent = list.neighbors.length;
    const sh = new Float64Array(3 * nent), lo = new Int32Array(nent), hi = new Int32Array(nent);
    if (prev) {
      for (let ii = 0; ii < list.inum; ii++) {
        const a = id[owner[ii]];
        for (let k = list.firstneigh[ii], k1 = k + list.numneigh[ii]; k < k1; k++) {
          const b = id[owner[list.neighbors[k] & NEIGHMASK]];
          const l = a < b ? a : b, h = a < b ? b : a;
          const o = prev.get(pairKey(l, h));
          if (o === undefined) continue;
          sh[3 * k] = oldShear[3 * o]; sh[3 * k + 1] = oldShear[3 * o + 1]; sh[3 * k + 2] = oldShear[3 * o + 2];
          lo[k] = l; hi[k] = h;
        }
      }
    }
    this.slotShear = sh; this.slotLo = lo; this.slotHi = hi;
    this.slotList = list;
    this.slotBuild = nbuild;
  }

  protected override shearOf(key: number | string): Float64Array | undefined {
    if (!this.slotIndex) {
      const m = new Map<number | string, number>();
      const lo = this.slotLo, hi = this.slotHi;
      for (let k = 0; k < lo.length; k++) if (lo[k] !== 0) m.set(pairKey(lo[k], hi[k]), k);
      this.slotIndex = m;
    }
    const k = this.slotIndex.get(key);
    return k === undefined ? undefined : this.slotShear.subarray(3 * k, 3 * k + 3);
  }

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
    if (history) this.ensureSlots(list, nb.nbuild, id, owner);
    this.slotIndex = null;
    const slotShear = this.slotShear, slotLo = this.slotLo, slotHi = this.slotHi;
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
        // a pair skipped or out of contact keeps no shear (native unsets touch and zeroes the shear)
        if (factor_lj === 0) {
          if (history) { slotShear[3 * k] = slotShear[3 * k + 1] = slotShear[3 * k + 2] = 0; slotLo[k] = 0; }
          continue;
        }
        const oj = owner[j];
        const delx = xtmp - x[3 * j];
        const dely = ytmp - x[3 * j + 1];
        const delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const radj = radius[oj];
        const radsum = radi + radj;
        if (rsq >= radsum * radsum) {
          if (history) { slotShear[3 * k] = slotShear[3 * k + 1] = slotShear[3 * k + 2] = 0; slotLo[k] = 0; }
          continue;
        }
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
          const o3 = 3 * k;
          const sg = flip ? -1 : 1;
          let shear0 = sg * slotShear[o3], shear1 = sg * slotShear[o3 + 1], shear2 = sg * slotShear[o3 + 2];
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
          slotShear[o3] = sg * shear0; slotShear[o3 + 1] = sg * shear1; slotShear[o3 + 2] = sg * shear2;
          slotLo[k] = flip ? b : a; slotHi[k] = flip ? a : b;
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
    nb.reverseSum(tq, 3, s.torque!);
  }
}
