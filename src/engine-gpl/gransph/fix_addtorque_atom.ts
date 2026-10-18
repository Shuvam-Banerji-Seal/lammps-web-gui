/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/GRANULAR/fix_addtorque_atom.cpp
 * and fix_addtorque_atom.h (FixAddTorqueAtom: fix addtorque/atom). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Syntax (from the source): "fix ID group-ID addtorque/atom tx ty tz [every N] [region ID]". Each atom of the group (and of
 * the region) gets the torque components added to its torque; a component is a constant, or a v_name equal-style or
 * atom-style variable. The vector output is the torque of the group before the change, summed over the atoms that the
 * fix acts on, as native compute_vector(). The step test of "every" is the native "ntimestep % nevery" check, and the
 * fix keeps the last sums on the steps it is skipped.
 */
import { Fix } from '../../engine/fix/fix';
import type { Region } from '../../engine/region';
import { StyleError } from '../../engine/force/types';
import { parseNum, parseInt_ } from '../../engine/force/util';
import type { System } from '../../engine/system';

/** One torque component: a constant, or a v_name variable (equal or atom style). */
type Comp = { kind: 'const'; v: number } | { kind: 'var'; name: string };

export class FixAddTorqueAtom extends Fix {
  readonly style = 'addtorque/atom';
  private readonly comps: Comp[];
  private regionId: string | null = null;
  private regionObj: Region | null = null;
  /** Torque sums of the group before the change (native toriginal). */
  private readonly toriginal = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal fix addtorque/atom command: missing argument(s)');
    this.comps = args.slice(0, 3).map((w) => (w.startsWith('v_') ? { kind: 'var' as const, name: w.slice(2) } : { kind: 'const' as const, v: parseNum(w, 'torque') }));
    for (let k = 3; k < args.length; ) {
      if (args[k] === 'every') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix addtorque/atom command: missing argument(s)');
        this.nevery = parseInt_(args[k + 1], 'every');
        if (this.nevery <= 0) throw new StyleError(`Invalid fix addtorque/atom every argument: ${this.nevery}`);
        k += 2;
      } else if (args[k] === 'region') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix addtorque/atom command: missing argument(s)');
        this.regionId = args[k + 1];
        k += 2;
      } else {
        throw new StyleError(`Unknown fix addtorque/atom keyword: ${args[k]}`);
      }
    }
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.globalFreq = 1;
    this.extvector = 1;
  }

  init(): void {
    for (const c of this.comps) {
      if (c.kind !== 'var') continue;
      const v = this.sys.vars.get(c.name);
      if (!v) throw new StyleError(`Variable ${c.name} for fix addtorque/atom does not exist`);
      if (v.style !== 'equal' && v.style !== 'atom') throw new StyleError(`Variable ${c.name} for fix addtorque/atom is invalid style`);
    }
    if (this.regionId !== null) {
      this.regionObj = this.sys.region(this.regionId);
      if (!this.regionObj) throw new StyleError(`Region ${this.regionId} for fix addtorque/atom does not exist`);
    }
    if (!this.sys.state.torque) throw new StyleError('fix addtorque/atom requires atom attribute torque (atom_style with torques, e.g. sphere or bpm/sphere)');
  }

  private region(): Region | null {
    if (this.regionId === null) return null;
    this.regionObj ??= this.sys.region(this.regionId);
    return this.regionObj;
  }

  postForce(): void {
    const s = this.sys.state;
    if (s.step % this.nevery) return;
    const torque = s.torque!;
    const reg = this.region();
    this.toriginal.fill(0);
    // per component: the value for the step (constant, equal-style, or per-atom for atom-style variables)
    const vals: (number | Float64Array)[] = this.comps.map((c) => {
      if (c.kind === 'const') return c.v;
      const v = this.sys.vars.get(c.name);
      if (!v) throw new StyleError(`Variable ${c.name} for fix addtorque/atom does not exist`);
      return v.style === 'atom' ? this.sys.atomVariable(c.name) : this.sys.equalVariable(c.name);
    });
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (reg && !reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
      this.toriginal[0] += torque[3 * i];
      this.toriginal[1] += torque[3 * i + 1];
      this.toriginal[2] += torque[3 * i + 2];
      for (let c = 0; c < 3; c++) {
        const val = vals[c];
        const add = typeof val === 'number' ? val : val[i];
        torque[3 * i + c] += add;
      }
    }
  }

  minPostForce(): void {
    this.postForce();
  }

  computeVector(i: number): number {
    return this.toriginal[i];
  }
}
