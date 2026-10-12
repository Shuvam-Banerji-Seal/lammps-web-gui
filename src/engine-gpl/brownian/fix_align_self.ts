/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BROWNIAN/fix_align_self.cpp and
 * fix_align_self.h (class FixAlignSelf, fix align/self: a torque that turns the dipole (dipole) or the body-frame
 * axis (quat) of a self-propelled particle towards its velocity). Original authors: Sam Cameron (University of
 * Bristol) and Arthur Straube (Zuse Institute Berlin). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { isEllipsoid } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { cross3, matvec, quatToMat, type Mat3 } from './mathextra';

/** fix_align_self.cpp: "static constexpr double SMALL = 1.0e-14;" for the qvector norm. */
const SMALL = 1.0e-14;

export class FixAlignSelf extends Fix {
  readonly style = 'align/self';
  private readonly mode: 'dipole' | 'quat';
  private readonly magnitude: number;
  private sx = 1.0;
  private sy = 0.0;
  private sz = 0.0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 2 && args.length !== 6) throw new StyleError('Incorrect number of fix align/self arguments');
    if (args[0] === 'dipole') this.mode = 'dipole';
    else if (args[0] === 'quat') this.mode = 'quat';
    else throw new StyleError(`Unknown fix align/self keyword ${args[0]}`);
    this.magnitude = args[1] === 'inf' ? Infinity : parseNum(args[1], 'magnitude');
    if (args.length === 6) {
      if (this.mode !== 'quat') throw new StyleError("Incorrect number of arguments for 'quat' mode of fix align/self");
      if (args[2] !== 'qvector') throw new StyleError(`Unknown fix align/self keyword ${args[2]}`);
      this.sx = parseNum(args[3], 'qvector');
      this.sy = parseNum(args[4], 'qvector');
      this.sz = parseNum(args[5], 'qvector');
      const snorm = Math.sqrt(this.sx * this.sx + this.sy * this.sy + this.sz * this.sz);
      if (snorm < SMALL) throw new StyleError(`Fix align/self qvector magnitude ${snorm} is too small`);
      this.sx = this.sx / snorm;
      this.sy = this.sy / snorm;
      this.sz = this.sz / snorm;
    }
  }

  /** FixAlignSelf::init(): the mode's required atom attributes. */
  override init(): void {
    const s = this.sys.state;
    if (this.mode === 'dipole' && (!s.mu || !s.torque)) throw new StyleError('Fix align/self with option dipole requires atom attributes mu + torque');
    if (this.mode === 'quat') {
      if (!s.quat || !s.shape) throw new StyleError('Fix align/self with option quat requires atom style ellipsoid');
      for (let i = 0; i < s.n; i++) {
        if ((s.mask[i] & this.groupBit) && !isEllipsoid(s, i)) throw new StyleError('Fix align/self with option quat requires extended particles');
      }
    }
  }

  /** FixAlignSelf::post_force: adds the torque magnitude * (axis x v) to the torques of the group. */
  override postForce(): void {
    if (this.mode === 'dipole') this.postForceDipole();
    else this.postForceQuat();
  }

  private postForceDipole(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const torque = s.torque!;
    const mu = s.mu!;
    const selfTorque = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      cross3([mu[4 * i], mu[4 * i + 1], mu[4 * i + 2]], [v[3 * i], v[3 * i + 1], v[3 * i + 2]], selfTorque);
      torque[3 * i] += selfTorque[0] * this.magnitude;
      torque[3 * i + 1] += selfTorque[1] * this.magnitude;
      torque[3 * i + 2] += selfTorque[2] * this.magnitude;
    }
  }

  private postForceQuat(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const torque = s.torque!;
    const quat = s.quat!;
    const fAct = [this.sx, this.sy, this.sz];
    const Q: Mat3 = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const q = [0, 0, 0, 0];
    const fRot = [0, 0, 0];
    const selfTorque = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      for (let k = 0; k < 4; k++) q[k] = quat[4 * i + k];
      quatToMat(q, Q);
      matvec(Q, fAct, fRot);
      cross3(fRot, [v[3 * i], v[3 * i + 1], v[3 * i + 2]], selfTorque);
      torque[3 * i] += selfTorque[0] * this.magnitude;
      torque[3 * i + 1] += selfTorque[1] * this.magnitude;
      torque[3 * i + 2] += selfTorque[2] * this.magnitude;
    }
  }
}
