/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMPUTE/compute_ave_sphere_atom.cpp
 * and compute_ave_sphere_atom.h (ComputeAveSphereAtom: compute ave/sphere/atom, the number density and the local
 * temperature of the atoms within a sphere around each atom). Contributing author: Stan Moore (SNL). Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_ave_sphere_atom.html: "compute ID group-ID ave/sphere/atom keyword value ..." with the keyword
 * "cutoff" (the sphere radius; default the pair cutoff). The per-atom array has two columns: the density (mass per
 * volume, in mass/distance^3 of the units) and the temperature of the atoms within the sphere, centre-of-mass velocity
 * removed. Atoms outside the group report zeros.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { NEIGHMASK } from '../../engine/neighbor';
import { argNumeric, OccasionalSnapshot, occasionalCutoff, pairCutforce } from './common';
import { topologyRefusal } from './topology';

const MY_PI = 3.14159265358979323846;

export class ComputeAveSphereAtom extends Compute {
  readonly style = 'ave/sphere/atom';
  private readonly cutoffArg: number;
  private readonly snap = new OccasionalSnapshot();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length > 2) throw new StyleError('Illegal compute ave/sphere/atom command');
    let cutoff = 0.0;
    for (let i = 0; i < args.length;) {
      if (args[i] === 'cutoff') {
        if (i + 2 > args.length) throw new StyleError('Illegal compute ave/sphere/atom command');
        cutoff = argNumeric(args[i + 1]);
        if (cutoff <= 0.0) throw new StyleError('Illegal compute ave/sphere/atom command');
        i += 2;
      } else {
        throw new StyleError('Illegal compute ave/sphere/atom command');
      }
    }
    this.cutoffArg = cutoff;
    this.peratomFlag = true;
    this.sizePeratomCols = 2;
  }

  /** The checks of ComputeAveSphereAtom::init(): a cutoff or a pair style is needed. */
  init(): void {
    if (!this.sys.ff.pair && this.cutoffArg === 0.0) {
      throw new StyleError('Compute ave/sphere/atom requires a cutoff be specified or a pair style be defined');
    }
  }

  protected computePeratom(): void {
    const sys = this.sys;
    const s = sys.state;
    const pair = sys.ff.pair;
    if (!pair && this.cutoffArg === 0.0) {
      throw new StyleError('Compute ave/sphere/atom requires a cutoff be specified or a pair style be defined');
    }
    sys.forces();
    topologyRefusal(sys, 'ave/sphere/atom');
    const nb = sys.nb;
    let cutoff = this.cutoffArg;
    if (pair && cutoff === 0.0) cutoff = pairCutforce(pair.cut);
    if (cutoff > nb.cutghost) {
      throw new StyleError('Compute ave/sphere/atom cutoff exceeds ghost atom range - use comm_modify cutoff command');
    }
    const cutsq = cutoff * cutoff;
    const dim = sys.dimension;
    // volume of the sphere (3d) or disc (2d): 4/3 pi cutsq cutoff, as written in the native constructor's init()
    const volume = dim === 3 ? 4.0 / 3.0 * MY_PI * cutsq * cutoff : MY_PI * cutsq;

    const n = s.n;
    const result = new Float64Array(2 * n);
    this.arrayAtom = result;
    // native: the cutoff keyword is a fixed list cutoff only beyond the pair cutoff plus the skin (cutflag)
    const cutflag = !pair || cutoff > pairCutforce(pair.cut) + nb.skin;
    const list = this.snap.get(sys, occasionalCutoff(sys, cutoff, cutflag));
    const x = nb.xall, v = s.v, owner = nb.owner;
    const adof = dim;
    const mvv2e = s.units.mvv2e, mv2d = s.units.mv2d, boltz = s.units.boltz;

    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const massI = massOf(s, i);
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      let count = 1;
      let totalmass = massI;
      const p = [v[3 * i] * massI, v[3 * i + 1] * massI, v[3 * i + 2] * massI];
      // neighbours inside the sphere: their mass and momentum
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[k0 + jj] & NEIGHMASK;
        const massJ = massOf(s, owner[j]);
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          count++;
          totalmass += massJ;
          const o = owner[j];
          p[0] += v[3 * o] * massJ;
          p[1] += v[3 * o + 1] * massJ;
          p[2] += v[3 * o + 2] * massJ;
        }
      }
      const vcom = [p[0] / totalmass, p[1] / totalmass, p[2] / totalmass];
      // kinetic energy about the centre-of-mass velocity
      let vnet0 = v[3 * i] - vcom[0], vnet1 = v[3 * i + 1] - vcom[1], vnet2 = v[3 * i + 2] - vcom[2];
      let keSum = massI * (vnet0 * vnet0 + vnet1 * vnet1 + vnet2 * vnet2);
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[k0 + jj] & NEIGHMASK;
        const massJ = massOf(s, owner[j]);
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          const o = owner[j];
          vnet0 = v[3 * o] - vcom[0];
          vnet1 = v[3 * o + 1] - vcom[1];
          vnet2 = v[3 * o + 2] - vcom[2];
          keSum += massJ * (vnet0 * vnet0 + vnet1 * vnet1 + vnet2 * vnet2);
        }
      }
      const density = mv2d * totalmass / volume;
      const temp = mvv2e * keSum / (adof * count * boltz);
      result[2 * i] = density;
      result[2 * i + 1] = temp;
    }
  }
}
