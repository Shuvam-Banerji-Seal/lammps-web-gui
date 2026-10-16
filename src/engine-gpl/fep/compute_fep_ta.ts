/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/compute_fep_ta.cpp and src/FEP/compute_fep_ta.h (class ComputeFEPTA, compute fep/ta) at commit
 * b332f28, by Shifeng Ke (Zhejiang University) and the LAMMPS developers. Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * compute ID group-ID fep/ta T xy|xz|yz scale tail yes|no (docs.lammps.org/compute_fep_ta.html): the test-area
 * perturbation. The box is scaled in the two tangential directions by sqrt(scale) and in the normal direction by
 * 1/scale, the atoms keep their fractional coordinates, and the energy is evaluated on the perturbed box with the
 * current neighbor lists. Vector (3): [0] = U(perturbed) - U(current), [1] = exp(-[0] / (k_B T)), [2] = the
 * tangential area of the original box times (scale - 1).
 *
 * The forces, positions, energies and box are restored afterwards, as the native compute restores its backups
 * (backup_xfev / restore_xfev / backup_box / restore_box). Native compute_pe adds the tail term etail / V with the
 * volume of the box being evaluated only with tail yes; this port follows that (see compute_fep.ts energyOf).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { hasTail } from './pair_table';
import { restoreAccum, snapshotAccum } from './compute_fep';
import { exp as libmExp } from '../libm/exp';

const USAGE = 'usage: compute ID group-ID fep/ta T xy|xz|yz scale tail yes/no';

export class ComputeFEPTA extends Compute {
  readonly style = 'fep/ta';
  private readonly temp: number;
  private readonly tan1: number;
  private readonly tan2: number;
  private readonly norm: number;
  private readonly scale: number;
  private tailFlag = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError(`Illegal number of arguments in compute fep/ta (${USAGE})`);
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extvector = 0;
    this.vector = new Float64Array(3);
    this.temp = parseNum(args[0], 'temperature');
    // the axes as in native: "xy" -> tangential x, y and normal z, and so on
    const axes: Record<string, [number, number, number]> = { xy: [0, 1, 2], xz: [0, 2, 1], yz: [1, 2, 0] };
    const ax = axes[args[1]];
    if (!ax) throw new StyleError(`Illegal arguments in compute fep/ta (${USAGE})`);
    [this.tan1, this.tan2, this.norm] = ax;
    this.scale = parseNum(args[2], 'scale factor');
    let k = 3;
    while (k < args.length) {
      if (args[k] !== 'tail') throw new StyleError(`Illegal optional keyword in compute fep/ta: '${args[k]}' (${USAGE})`);
      const w = args[k + 1];
      if (w !== 'yes' && w !== 'no') throw new StyleError(`compute fep/ta keyword tail must be yes or no, got '${w ?? ''}'`);
      this.tailFlag = w === 'yes';
      k += 2;
    }
  }

  init(): void {
    if (this.sys.state.dimension === 2) throw new StyleError('Cannot compute fep/ta in 2d simulation');
    if (this.tailFlag && !this.pairTail()) throw new StyleError('Compute fep/ta tail when pair style does not compute tail corrections');
  }

  private pairTail(): boolean {
    const owner = this.sys.ff.pair;
    return owner ? hasTail(owner) : false;
  }

  /** Pair energy as native compute_pe: evdwl + ecoul (+ etail / V with tail yes), plus the kspace energy. */
  private energyOf(acc: { evdwl: number; ecoul: number; elong: number }): number {
    const sys = this.sys;
    let e = acc.evdwl + acc.ecoul;
    if (!this.tailFlag) e -= sys.ff.etailV / sys.geom.volume(sys.state.dimension);
    if (sys.ff.kspace) e += acc.elong;
    return e;
  }

  protected computeVector(): void {
    const sys = this.sys, s = sys.state, b = s.box, g = sys.geom;
    const kT = s.units.boltz * this.temp;
    const pe0 = this.energyOf(sys.forces());
    // backups: positions, forces, the force-field accumulator and the box (native backup_xfev and backup_box)
    const xSave = s.x.slice();
    const fSave = s.f.slice();
    const accSave = snapshotAccum(sys.ff.acc);
    const lo0 = [...b.lo], hi0 = [...b.hi], tilt0 = [...b.tilt];
    const areaOrig = (hi0[this.tan1] - lo0[this.tan1]) * (hi0[this.tan2] - lo0[this.tan2]);
    try {
      this.changeBox(sys, lo0, hi0, tilt0);
      const pe1 = this.energyOf(sys.computeForcesInRun(sys.computeFlags()));
      const dU = pe1 - pe0;
      this.vector[0] = dU;
      this.vector[1] = libmExp(-dU / kT);
      this.vector[2] = areaOrig * (this.scale - 1.0);
    } finally {
      // restore_box, then restore_xfev: the original box, positions, forces and accumulator
      for (let d = 0; d < 3; d++) { b.lo[d] = lo0[d]; b.hi[d] = hi0[d]; b.tilt[d] = tilt0[d]; }
      g.update();
      s.x.set(xSave);
      sys.nb.forwardComm(s, g);
      s.f.set(fSave);
      restoreAccum(sys.ff.acc, accSave);
    }
  }

  /**
   * change_box as native (ComputeFEPTA::change_box): the tangential bounds scale by sqrt(scale), the normal one by
   * 1/scale, and every atom keeps its fractional coordinates. Fixes are not told (the native compute does not call
   * them), and the forces stay as they are until the perturbed evaluation.
   */
  private changeBox(sys: System, lo0: number[], hi0: number[], tilt0: number[]): void {
    const s = sys.state, b = s.box, g = sys.geom;
    const lam = new Float64Array(3 * s.n);
    const tmp = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      lam[3 * i] = tmp[0]; lam[3 * i + 1] = tmp[1]; lam[3 * i + 2] = tmp[2];
    }
    const rt = Math.sqrt(this.scale);
    const lo = [...lo0], hi = [...hi0];
    lo[this.tan1] *= rt; hi[this.tan1] *= rt;
    lo[this.tan2] *= rt; hi[this.tan2] *= rt;
    lo[this.norm] /= this.scale; hi[this.norm] /= this.scale;
    for (let d = 0; d < 3; d++) { b.lo[d] = lo[d]; b.hi[d] = hi[d]; b.tilt[d] = tilt0[d]; }
    g.update();
    for (let i = 0; i < s.n; i++) {
      g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
    }
    // the pair kernels read the ghost-extended copy xall: the ghosts follow their owners into the new box
    sys.nb.forwardComm(s, g);
  }
}
