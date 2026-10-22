/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_viscous_sphere.cpp (FixViscousSphere: constructor, init, setup, post_force, min_post_force)
 * at commit b332f28. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Damping torque -gamma_i omega_i on finite-size atoms (atom_style sphere / bpm/sphere), with the optional scale keyword
 * per atom type or per atom (atom-style variable). docs.lammps.org/fix_viscous_sphere.html.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

export class FixViscousSphere extends Fix {
  readonly style = 'viscous/sphere';
  private readonly gamma: number;
  /** scale type ratio: per-type factor (1 when unset). */
  private scaleType: Float64Array | null = null;
  /** scale v_name: atom-style variable name. */
  private scaleVar: string | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!sys.state.omega || !sys.state.torque) throw new StyleError('Fix viscous/sphere requires atom attribute omega');
    if (args.length < 1) throw new StyleError('Illegal fix viscous/sphere command');
    this.gamma = Number(args[0]);
    if (args[0].startsWith('v_') || !Number.isFinite(this.gamma)) {
      throw new StyleError(`fix viscous/sphere: gamma must be a number (torque/angular velocity units), got '${args[0]}'`);
    }
    const s = sys.state;
    for (let k = 1; k < args.length;) {
      // As native LAMMPS does (src/EXTRA-FIX/fix_viscous_sphere.cpp FixViscousSphere::FixViscousSphere at b332f28): the only keyword is scale,
      // and an unknown word or a missing value gives the same "Illegal fix viscous/sphere command"
      if (args[k] !== 'scale') throw new StyleError('Illegal fix viscous/sphere command');
      if (k + 1 >= args.length) throw new StyleError('Illegal fix viscous/sphere command');
      if (args[k + 1].startsWith('v_')) {
        if (this.scaleType) throw new StyleError('Must use only one style of scaling');
        const name = args[k + 1].slice(2);
        const v = sys.vars.get(name);
        if (!v) throw new StyleError(`Variable name ${name} for fix viscous/sphere does not exist`);
        if (v.style !== 'atom') throw new StyleError(`Fix viscous/scale variable ${name} is not atom-style variable`);
        this.scaleVar = name;
        k += 2;
      } else {
        if (this.scaleVar) throw new StyleError('Must use only one style of scaling');
        if (k + 2 >= args.length) throw new StyleError('Illegal fix viscous/sphere command');
        this.scaleType ??= new Float64Array(s.ntypes + 1).fill(1);
        const t = Number(args[k + 1]);
        const r = Number(args[k + 2]);
        if (!Number.isInteger(t) || t < 1 || t > s.ntypes) {
          throw new StyleError(`Atom type ${args[k + 1]} out of range for fix viscous/sphere command:`);
        }
        this.scaleType[t] = r;
        k += 3;
      }
    }
  }

  /** torque_i -= gamma_i omega_i, gamma_i = gamma (or scaled by type / atom-style variable). */
  postForce(): void {
    const s = this.sys.state;
    const omega = s.omega!;
    const torque = s.torque!;
    const scaleVals = this.scaleVar ? this.sys.atomVariable(this.scaleVar) : null;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      let drag = this.gamma;
      if (scaleVals) drag = this.gamma * scaleVals[i];
      else if (this.scaleType) drag = this.gamma * this.scaleType[s.type[i]];
      torque[3 * i] -= drag * omega[3 * i];
      torque[3 * i + 1] -= drag * omega[3 * i + 1];
      torque[3 * i + 2] -= drag * omega[3 * i + 2];
    }
  }

  minPostForce(): void { this.postForce(); }
}
