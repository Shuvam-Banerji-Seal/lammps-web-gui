/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/GRANULAR/fix_damping_cundall.cpp
 * and fix_damping_cundall.h (FixDampingCundall: fix damping/cundall). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Syntax (from the source): "fix ID group-ID damping/cundall gamma_lin gamma_ang [scale type value | scale v_name]". Each
 * component of the force and of the torque is multiplied by 1 - gamma sign, where the sign is +1 when the component of the
 * force (torque) and of the velocity (angular velocity) have the same sign, and -1 otherwise: energy is removed from the
 * motion and a fraction of the force is added back where the work is negative (Cundall damping, after Yade's
 * NewtonIntegrator::cundallDamp1st). The fix needs atom attribute omega.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseNum, parseInt_ } from '../../engine/force/util';
import type { System } from '../../engine/system';

export class FixDampingCundall extends Fix {
  readonly style = 'damping/cundall';
  private readonly gammaLin: number;
  private readonly gammaAng: number;
  /** scale: none, per type (scaleByType, index = type), or an atom-style variable (scaleVar). */
  private scaleStyle: 'none' | 'type' | 'variable' = 'none';
  private scaleByType: Float64Array | null = null;
  private scaleVar: string | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!sys.state.omega) throw new StyleError('Fix damping/cundall requires atom attribute omega');
    if (args.length < 2) throw new StyleError('Illegal fix damping/cundall command: missing argument(s)');
    this.gammaLin = parseNum(args[0], 'gamma_lin');
    this.gammaAng = parseNum(args[1], 'gamma_ang');
    for (let k = 2; k < args.length; ) {
      if (args[k] !== 'scale') throw new StyleError('Illegal fix damping/cundall command');
      if (k + 2 > args.length) throw new StyleError('Illegal fix damping/cundall command');
      const val = args[k + 1];
      if (val.startsWith('v_')) {
        if (this.scaleStyle !== 'none') throw new StyleError('Must use only one style of scaling');
        this.scaleVar = val.slice(2);
        const v = sys.vars.get(this.scaleVar);
        if (!v) throw new StyleError(`Variable name ${this.scaleVar} for fix damping/cundall does not exist`);
        if (v.style !== 'atom') throw new StyleError(`Fix viscous/scale variable ${this.scaleVar} is not atom-style variable`);
        this.scaleStyle = 'variable';
        k += 2;
      } else {
        if (this.scaleStyle === 'variable') throw new StyleError('Must use only one style of scaling');
        if (k + 3 > args.length) throw new StyleError('Illegal fix damping/cundall command');
        if (!this.scaleByType) {
          this.scaleByType = new Float64Array(sys.state.ntypes + 1).fill(1.0);
        }
        const itype = parseInt_(val, 'type');
        const scale = parseNum(args[k + 2], 'scale');
        if (itype < 1 || itype > sys.state.ntypes) {
          throw new StyleError(`Atom type ${itype} out of range for fix damping/cundall command:`);
        }
        this.scaleByType[itype] = scale;
        this.scaleStyle = 'type';
        k += 3;
      }
    }
  }

  init(): void {
    if (this.scaleStyle === 'variable' && this.scaleVar !== null) {
      const v = this.sys.vars.get(this.scaleVar);
      if (!v) throw new StyleError(`Variable name ${this.scaleVar} for fix damping/cundall does not exist`);
      if (v.style !== 'atom') throw new StyleError(`Fix damping/cundall variable ${this.scaleVar} is not atom-style variable`);
    }
  }

  postForce(): void {
    const s = this.sys.state;
    const v = s.v, omega = s.omega!, f = s.f, torque = s.torque!, type = s.type;
    const scaleVals = this.scaleStyle === 'variable' ? this.sys.atomVariable(this.scaleVar!) : null;
    // signf = (f * v >= 0.0) ? 1.0 : -1.0, as the source writes it
    const sgn = (p: number): number => (p >= 0.0 ? 1.0 : -1.0);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      let gammaL: number, gammaA: number;
      if (this.scaleStyle === 'type') {
        gammaL = this.gammaLin * this.scaleByType![type[i]];
        gammaA = this.gammaAng * this.scaleByType![type[i]];
      } else if (this.scaleStyle === 'variable') {
        gammaL = this.gammaLin * scaleVals![i];
        gammaA = this.gammaAng * scaleVals![i];
      } else {
        gammaL = this.gammaLin;
        gammaA = this.gammaAng;
      }
      for (let c = 0; c < 3; c++) {
        const k = 3 * i + c;
        f[k] *= 1.0 - gammaL * sgn(f[k] * v[k]);
      }
      for (let c = 0; c < 3; c++) {
        const k = 3 * i + c;
        torque[k] *= 1.0 - gammaA * sgn(torque[k] * omega[k]);
      }
    }
  }

  minPostForce(): void {
    this.postForce();
  }
}
