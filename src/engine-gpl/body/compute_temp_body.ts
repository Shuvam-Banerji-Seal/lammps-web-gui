/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/compute_temp_body.cpp and
 * compute_temp_body.h (ComputeTempBody: compute temp/body, the dof_compute, compute_scalar and compute_vector of the
 * body rotational and translational kinetic energy). LAMMPS is Copyright (2003) Sandia Corporation and distributed
 * under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Contributing author (from the source): Trung Dac Nguyen (ndactrung@gmail.com).
 *
 * docs.lammps.org/compute_temp_body.html: "compute ID group-ID temp/body keyword value ..." with the keywords "dof"
 * ("rotate" or "all") and "bias". This port supports dof only; bias is refused by name (StyleError).
 *
 * The native code: the translational term is m v^2, the rotational term is sum_k I_k w_k^2 with w = R^T L / I in the
 * body frame, where I are the principal moments of the body (atom_style body record, bodies.ts) and R its orientation.
 * Degrees of freedom: 6 per body in 3d and 3 in 2d for dof all; 3 in 3d and 1 in 2d for dof rotate (native
 * dof_compute, nper). The dof removed by the fixes and extra_dof are subtracted as for the other temperature computes.
 */
import { ComputeTemp } from '../../engine/compute/temp';
import { quatToMatrix, rotateTransposeVec } from '../../engine/compute/asphere';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf } from '../../engine/atoms';
import { BODY_RECORD } from '../../engine/bodies';

export class ComputeTempBody extends ComputeTemp {
  readonly style: string = 'temp/body';
  private rotateOnly = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    // compute_temp_body.cpp: the keywords here, the atom style at init
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k], val = args[k + 1];
      if ((key === 'dof' || key === 'bias') && val === undefined) throw new StyleError(`Illegal compute temp/body ${key} command: missing argument(s)`);
      if (key === 'dof') {
        if (val !== 'all' && val !== 'rotate') throw new StyleError(`Unknown compute temp/body dof keyword ${val}`);
        this.rotateOnly = val === 'rotate';
      } else if (key === 'bias') {
        throw new StyleError('Compute temp/body bias is not supported by the browser engine');
      } else throw new StyleError(`Unknown compute temp/body keyword ${key}`);
    }
  }

  init(): void {
    const s = this.sys.state;
    if (!s.quat || !s.angmom || !s.custom.has(BODY_RECORD)) throw new StyleError('Compute temp/body requires atom style body');
    const rec = s.custom.get(BODY_RECORD)!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (rec.data[rec.cols * i] === 0) throw new StyleError('Compute temp/body requires bodies');
    }
    this.dofCompute();
  }

  dofCompute(): void {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.counted(i)) n++;
    const d = this.sys.dimension;
    const nper = d === 3 ? (this.rotateOnly ? 3 : 6) : (this.rotateOnly ? 1 : 3);
    this.dof = nper * n - this.extraDof - this.sys.dofRemoved(this.groupBit);
  }

  /** Translational m v^2 tensor and the body-frame rotational tensor of the counted bodies (native compute_vector). */
  private tensor(): { trans: number[]; rot: number[] } {
    const s = this.sys.state;
    const { v } = s;
    const rec = s.custom.get(BODY_RECORD)!;
    const q = s.quat!, L = s.angmom!;
    const trans = [0, 0, 0, 0, 0, 0], rot = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massOf(s, i);
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      trans[0] += m * vx * vx; trans[1] += m * vy * vy; trans[2] += m * vz * vz;
      trans[3] += m * vx * vy; trans[4] += m * vx * vz; trans[5] += m * vy * vz;
      const b = rec.cols * i;
      const I: [number, number, number] = [rec.data[b + 1], rec.data[b + 2], rec.data[b + 3]];
      const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
      const Lb = rotateTransposeVec(R, L[3 * i], L[3 * i + 1], L[3 * i + 2]);
      const w = [0, 1, 2].map((k) => (I[k] === 0 ? 0 : Lb[k] / I[k]));
      rot[0] += I[0] * w[0] * w[0]; rot[1] += I[1] * w[1] * w[1]; rot[2] += I[2] * w[2] * w[2];
      rot[3] += I[0] * w[0] * w[1]; rot[4] += I[1] * w[0] * w[2]; rot[5] += I[2] * w[1] * w[2];
    }
    return { trans, rot };
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const t = this.tensor();
    const sum = (this.rotateOnly ? 0 : t.trans[0] + t.trans[1] + t.trans[2]) + t.rot[0] + t.rot[1] + t.rot[2];
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return sum * tfactor;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const t = this.tensor();
    for (let c = 0; c < 6; c++) this.vector[c] = ((this.rotateOnly ? 0 : t.trans[c]) + t.rot[c]) * s.units.mvv2e;
  }
}
