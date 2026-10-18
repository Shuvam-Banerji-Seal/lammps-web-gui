/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMPUTE/compute_ackland_atom.cpp
 * and compute_ackland_atom.h (ComputeAcklandAtom: compute ackland/atom, the Ackland-Jones local crystal structure of
 * each atom). Contributing author: G. Ziegenhain (gerolf@ziegenhain.com), copyright (C) 2007; the updated algorithm is
 * by Brian Barnes. Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_ackland_atom.html: "compute ID group-ID ackland/atom keyword value ..." with the keyword
 * "legacy" (value yes or no); the per-atom vector gives the structure as 0 = unknown, 1 = bcc, 2 = fcc, 3 = hcp,
 * 4 = icosahedral.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { nativeOrder } from '../../engine/atoms';
import { NEIGHMASK } from '../../engine/neighbor';
import { argLogical, OccasionalSnapshot, occasionalCutoff, pairCutforce, select2 } from './common';
import { topologyRefusal } from './topology';

const UNKNOWN = 0, BCC = 1, FCC = 2, HCP = 3, ICO = 4;

export class ComputeAcklandAtom extends Compute {
  readonly style = 'ackland/atom';
  private readonly legacy: boolean;
  private readonly snap = new OccasionalSnapshot();
  private distsq = new Float64Array(0);
  private nearest = new Int32Array(0);
  private nearestN0 = new Int32Array(0);
  private nearestN1 = new Int32Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length > 2) throw new StyleError('Illegal compute ackland/atom command');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
    let legacy = false;
    for (let i = 0; i < args.length;) {
      if (args[i] === 'legacy') {
        if (i + 1 >= args.length) throw new StyleError('Invalid compute ackland/atom command');
        legacy = argLogical(args[i + 1]);
        i += 2;
      } else {
        throw new StyleError('Illegal compute ackland/atom command');
      }
    }
    this.legacy = legacy;
  }

  /** The check of ComputeAcklandAtom::init(): the force cutoff comes from the pair style. */
  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute ackland/atom requires a pair style be defined');
  }

  protected computePeratom(): void {
    const sys = this.sys;
    const s = sys.state;
    const pair = sys.ff.pair;
    if (!pair) throw new StyleError('Compute ackland/atom requires a pair style be defined');
    sys.forces();
    topologyRefusal(sys, 'ackland/atom');
    const nb = sys.nb;
    const cutforce = pairCutforce(pair.cut);
    const cutsq = cutforce * cutforce;
    const list = this.snap.get(sys, occasionalCutoff(sys, cutforce, false));
    const x = nb.xall;
    const n = s.n;
    const structure = new Float64Array(n);
    this.vectorAtom = structure;
    const chi = new Int32Array(8);

    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) { structure[i] = 0.0; continue; }
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      if (jnum > this.distsq.length) {
        this.distsq = new Float64Array(jnum);
        this.nearest = new Int32Array(jnum);
        this.nearestN0 = new Int32Array(jnum);
        this.nearestN1 = new Int32Array(jnum);
      }
      // neighbours within the force cutoff: distsq[] and nearest[] (atom indices)
      let nn = 0;
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[k0 + jj] & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          this.distsq[nn] = rsq;
          this.nearest[nn++] = j;
        }
      }
      // up to 6 nearest neighbours; undercoordinated atoms (n < 6) use all of them
      const nsel = Math.min(6, nn);
      select2(nsel, nn, this.distsq, this.nearest);
      let r0sq = 0.0;
      for (let j = 0; j < nsel; j++) r0sq += this.distsq[j];
      if (nsel > 0) r0sq /= nsel;
      // n0 near neighbours with distsq < 1.45 r0sq, n1 with distsq < 1.55 r0sq
      const n0distsq = 1.45 * r0sq, n1distsq = 1.55 * r0sq;
      let n0 = 0, n1 = 0;
      for (let j = 0; j < nn; j++) {
        if (this.distsq[j] < n1distsq) {
          this.nearestN1[n1++] = this.nearest[j];
          if (this.distsq[j] < n0distsq) this.nearestN0[n0++] = this.nearest[j];
        }
      }
      // angles <(r_ij, r_ik) for all pairs of the n0 neighbours: an 8-bin histogram of cos(angle)
      chi.fill(0);
      for (let j = 0; j < n0; j++) {
        const xij = xtmp - x[3 * this.nearestN0[j]], yij = ytmp - x[3 * this.nearestN0[j] + 1], zij = ztmp - x[3 * this.nearestN0[j] + 2];
        const normJ = Math.sqrt(xij * xij + yij * yij + zij * zij);
        if (normJ <= 0.0) continue;
        for (let k = j + 1; k < n0; k++) {
          const xik = xtmp - x[3 * this.nearestN0[k]], yik = ytmp - x[3 * this.nearestN0[k] + 1], zik = ztmp - x[3 * this.nearestN0[k] + 2];
          const normK = Math.sqrt(xik * xik + yik * yik + zik * zik);
          if (normK <= 0.0) continue;
          const bondAngle = (xij * xik + yij * yik + zij * zik) / (normJ * normK);
          if (bondAngle < -0.945) chi[0]++;
          else if (bondAngle < -0.915) chi[1]++;
          else if (bondAngle < -0.755) chi[2]++;
          else if (bondAngle < -0.195) chi[3]++;
          else if (bondAngle < 0.195) chi[4]++;
          else if (bondAngle < 0.245) chi[5]++;
          else if (bondAngle < 0.795) chi[6]++;
          else chi[7]++;
        }
      }
      structure[i] = this.legacy ? classifyLegacy(chi, n1) : classifyUpdated(chi, n0, n1);
    }
  }
}

/** The original classification (legacy yes), compute_ackland_atom.cpp "if (legacy)" branch. */
const classifyLegacy = (chi: Int32Array, n1: number): number => {
  let deltaBcc = 0.35 * chi[4] / (chi[5] + chi[6] - chi[4]);
  const deltaCp = Math.abs(1.0 - chi[6] / 24.0);
  let deltaFcc = 0.61 * (Math.abs(chi[0] + chi[1] - 6.0) + chi[2]) / 6.0;
  let deltaHcp = (Math.abs(chi[0] - 3.0) + Math.abs(chi[0] + chi[1] + chi[2] + chi[3] - 9.0)) / 12.0;
  if (chi[0] === 7) deltaBcc = 0.0;
  else if (chi[0] === 6) deltaFcc = 0.0;
  else if (chi[0] <= 3) deltaHcp = 0.0;
  if (chi[7] > 0) return UNKNOWN;
  if (chi[4] < 3) return n1 > 13 || n1 < 11 ? UNKNOWN : ICO;
  if (deltaBcc <= deltaCp) return n1 < 11 ? UNKNOWN : BCC;
  if (n1 > 12 || n1 < 11) return UNKNOWN;
  return deltaFcc < deltaHcp ? FCC : HCP;
};

/** The updated classification (the default, Brian Barnes), compute_ackland_atom.cpp "else" branch. */
const classifyUpdated = (chi: Int32Array, n0: number, n1: number): number => {
  if (chi[7] > 0 || n0 < 11) return UNKNOWN;
  if (chi[0] === 7) return BCC;
  if (chi[0] === 6) return FCC;
  if (chi[0] === 3) return HCP;
  const deltaCp = Math.abs(1.0 - chi[6] / 24.0);
  let deltaBcc = deltaCp + 1.0;
  const chi56m4 = chi[5] + chi[6] - chi[4];
  if (chi56m4 !== 0) deltaBcc = 0.35 * chi[4] / chi56m4;
  const deltaFcc = 0.61 * (Math.abs(chi[0] + chi[1] - 6) + chi[2]) / 6.0;
  const deltaHcp = (Math.abs(chi[0] - 3.0) + Math.abs(chi[0] + chi[1] + chi[2] + chi[3] - 9.0)) / 12.0;
  // the native code sets UNKNOWN here and then overwrites it in every branch below, so only the branches count
  if (chi[4] < 3 && n1 === 12) return ICO;
  if (deltaBcc <= deltaCp && n1 > 10 && n1 < 13) return BCC;
  if (n0 > 12) return UNKNOWN;
  return deltaFcc < deltaHcp ? FCC : HCP;
};
