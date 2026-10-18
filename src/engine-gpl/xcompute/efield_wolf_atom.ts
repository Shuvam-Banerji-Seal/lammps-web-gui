/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMPUTE/compute_efield_wolf_atom.cpp
 * and compute_efield_wolf_atom.h (ComputeEfieldWolfAtom: compute efield/wolf/atom, the electric field at each atom from the
 * charges of the other atoms, by the damped shifted-force Wolf sum). Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_efield_wolf_atom.html: "compute ID group-ID efield/wolf/atom alpha keyword value ..." with the
 * keywords "group" (the group of the source atoms, default all) and "cutoff" (default the pair cutoff). The per-atom
 * array has three columns, the field components.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { hasCharge, nativeOrder } from '../../engine/atoms';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { erfc } from '../libm/erf';
import { exp as libmExp } from '../libm/exp';
import { argNumeric, OccasionalSnapshot, occasionalCutoff, pairCutforce } from './common';
import { topologyRefusal } from './topology';

const MY_PIS = 1.77245385090551602729; // sqrt(pi), math_const.h

export class ComputeEfieldWolfAtom extends Compute {
  readonly style = 'efield/wolf/atom';
  private readonly alpha: number;
  private readonly group2: string;
  private readonly cutoffFlag: boolean;
  private readonly cutoffArg: number;
  private readonly snap = new OccasionalSnapshot();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal compute efield/atom/wolf command: missing arguments');
    this.peratomFlag = true;
    this.sizePeratomCols = 3;
    this.alpha = argNumeric(args[0]);
    let group2 = 'all';
    let cutoffFlag = false;
    let cutoff = 0.0;
    for (let i = 1; i < args.length;) {
      if (args[i] === 'group') {
        if (i + 2 > args.length) throw new StyleError('Illegal compute efield/atom/wolf group: missing argument');
        group2 = args[i + 1];
        i += 2;
      } else if (args[i] === 'cutoff') {
        if (i + 2 > args.length) throw new StyleError('Illegal compute efield/atom/wolf cutoff: missing argument');
        cutoff = argNumeric(args[i + 1]);
        cutoffFlag = true;
        i += 2;
      } else {
        throw new StyleError(`Unknown compute efield/wolf/atom keyword: ${args[i]}`);
      }
    }
    if (this.alpha <= 0.0) throw new StyleError(`Compute efield/atom/wolf alpha value ${this.alpha} is invalid`);
    if (cutoffFlag && cutoff <= 0.0) throw new StyleError(`Compute efield/atom/wolf cutoff ${cutoff} is invalid`);
    this.group2 = group2;
    this.cutoffFlag = cutoffFlag;
    this.cutoffArg = cutoff;
    if (sys.groups.find(group2) < 0) throw new StyleError(`Compute efield/atom/wolf group ${group2} does not exist`);
  }

  init(): void {
    if (!hasCharge(this.sys.state)) throw new StyleError('Compute efield/wolf/atom requires atom attribute q');
  }

  protected computePeratom(): void {
    const sys = this.sys;
    const s = sys.state;
    if (!hasCharge(s)) throw new StyleError('Compute efield/wolf/atom requires atom attribute q');
    const pair = sys.ff.pair;
    if (!this.cutoffFlag && !pair) throw new StyleError('Compute efield/wolf/atom cutoff is not set and no pair style is defined');
    sys.forces();
    topologyRefusal(sys, 'efield/wolf/atom');
    const nb = sys.nb;
    const jgroupbit = sys.groups.bit(this.group2);
    let cutoff = this.cutoffArg;
    if (!this.cutoffFlag && pair) cutoff = pairCutforce(pair.cut);
    if (cutoff > nb.cutghost) {
      throw new StyleError('Compute efield/wolf/atom cutoff exceeds ghost atom range - use comm_modify cutoff command');
    }
    const alpha = this.alpha;
    const cutsq = cutoff * cutoff;
    const eShift = erfc(alpha * cutoff) / cutoff;
    const fShift = -(eShift + 2.0 * alpha / MY_PIS * libmExp(-alpha * alpha * cutsq)) / cutoff;
    const qqrd2e = sys.units.qqr2e / sys.ff.dielectric;

    const out = new Float64Array(3 * s.n);
    this.arrayAtom = out;
    const list = this.snap.get(sys, occasionalCutoff(sys, cutoff, this.cutoffFlag));
    const x = nb.xall, q = nb.qall, owner = nb.owner, mask = s.mask;
    const sbc = (sys.ff as unknown as { specialCoul: Float64Array }).specialCoul;

    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      for (let jj = 0; jj < jnum; jj++) {
        const raw = list.neighbors[k0 + jj];
        const factorCoul = raw >>> SBBITS === 0 ? 1 : sbc[raw >>> SBBITS];
        const j = raw & NEIGHMASK;
        if (!(mask[owner[j]] & jgroupbit)) continue;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq > 0.0 && rsq < cutsq) {
          const r = Math.sqrt(rsq);
          const prefactor = qqrd2e * q[j] / r;
          const erfcc = erfc(alpha * r);
          const erfcd = libmExp(-alpha * alpha * r * r);
          const dvdrr = (erfcc / rsq + 2.0 * alpha / MY_PIS * erfcd / r) + fShift;
          let forcecoul = dvdrr * rsq * prefactor;
          if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
          forcecoul /= rsq;
          out[3 * i] += delx * forcecoul;
          out[3 * i + 1] += dely * forcecoul;
          out[3 * i + 2] += delz * forcecoul;
        }
      }
    }
  }
}
