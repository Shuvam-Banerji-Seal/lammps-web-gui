/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/fix_settorque_atom.cpp and fix_settorque_atom.h at commit b332f28 (the fix settorque/atom of
 * the GRANULAR package). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_settorque_atom.html (verbatim): "fix ID group-ID settorque/atom tx ty tz keyword value ..."; "tx,ty,tz =
 * torque component values"; "Any of the tx, ty, tz values can be specified as NULL which means do not alter the torque
 * component in that dimension."; "region-ID = ID of region atoms must be in to have added torque". The page says the fix
 * "erases all previously computed torques on the atom", and "This fix computes a global 3-vector of torques ... This is
 * the total torque on the group of atoms before the torques on individual atoms are changed by the fix. The vector values
 * calculated by this fix are "extensive"."
 *
 * Native behaviour kept here: a component is a constant, NULL (left alone), an equal-style v_name (one value per step) or
 * an atom-style v_name (one value per atom); a variable of another style is an error (checked in init, as native does).
 * Native has no energy tally: fix_modify energy yes is refused by the base class (fix.cpp "does not support fix_modify
 * energy command"). The minimizer uses the same post-force (min_post_force), so no non-zero check applies.
 */
import { Fix } from '../../engine/fix/fix';
import type { Region } from '../../engine/region';
import { StyleError } from '../../engine/force/types';
import { numeric, missingArgs } from '../../engine/force/util';
import type { System } from '../../engine/system';

/** One torque component: NULL (left alone), a constant, or a v_name variable. */
type Component = { kind: 'null' } | { kind: 'const'; v: number } | { kind: 'var'; name: string };

/** A component resolved for the current step: an atom-style variable gives one value per atom. */
type Resolved = { kind: 'null' } | { kind: 'const'; v: number } | { kind: 'vals'; v: Float64Array };

const val = (r: Resolved, i: number): number => (r.kind === 'const' ? r.v : r.kind === 'vals' ? r.v[i] : 0);

export class FixSetTorqueAtom extends Fix {
  readonly style = 'settorque/atom';
  private comps: Component[];
  private regionId: string | null = null;
  private regionObj: Region | null = null;
  /** The torque total of the group before this fix changed it (the vector output). */
  private vec = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal fix settorque/atom command: expected tx ty tz [region ID]');
    this.comps = args.slice(0, 3).map((w, k) => this.parseComponent(w, ['tx', 'ty', 'tz'][k]));
    for (let k = 3; k < args.length; k += 2) {
      if (args[k] !== 'region') throw new StyleError(`Unknown fix settorque/atom keyword: ${args[k]}`);
      if (k + 1 >= args.length) throw missingArgs('fix settorque/atom region');
      this.regionId = args[k + 1];
      if (!sys.regions.has(this.regionId)) throw new StyleError(`Region ${this.regionId} for fix settorque/atom does not exist`);
    }
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extvector = 1;
    this.globalFreq = 1;
  }

  private parseComponent(w: string, what: string): Component {
    if (w === 'NULL') return { kind: 'null' };
    if (w.startsWith('v_')) return { kind: 'var', name: w.slice(2) };
    // As native LAMMPS does (src/GRANULAR/fix_settorque_atom.cpp FixSetTorqueAtom::FixSetTorqueAtom at b332f28):
    // utils::numeric reads a constant component.
    return { kind: 'const', v: numeric(w) };
  }

  /** native init(): the variables must exist and be equal-style or atom-style. */
  init(): void {
    for (const c of this.comps) {
      if (c.kind !== 'var') continue;
      const v = this.sys.vars.get(c.name);
      if (!v) throw new StyleError(`Variable ${c.name} for fix settorque/atom does not exist`);
      if (v.style !== 'equal' && v.style !== 'atom') throw new StyleError(`Variable ${c.name} for fix settorque/atom is invalid style`);
    }
    if (this.regionId !== null) this.regionObj = this.sys.region(this.regionId);
    if (!this.sys.state.torque) throw new StyleError('fix settorque/atom requires atom attribute torque (atom_style with torques, e.g. sphere or bpm/sphere)');
  }

  private resolve(): [Resolved, Resolved, Resolved] {
    const r = (c: Component): Resolved => {
      if (c.kind === 'null') return c;
      if (c.kind === 'const') return c;
      const v = this.sys.vars.get(c.name);
      if (!v) throw new StyleError(`Variable ${c.name} for fix settorque/atom does not exist`);
      if (v.style === 'atom') return { kind: 'vals', v: this.sys.atomVariable(c.name) };
      return { kind: 'const', v: this.sys.equalVariable(c.name) };
    };
    return [r(this.comps[0]), r(this.comps[1]), r(this.comps[2])];
  }

  /** post_force: the group total before the change is the vector; the torque is set for each atom in the group (and the region). */
  postForce(): void {
    const s = this.sys.state;
    const torque = s.torque;
    if (!torque) throw new StyleError('fix settorque/atom requires atom attribute torque');
    const [cx, cy, cz] = this.resolve();
    let tx = 0, ty = 0, tz = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const r = this.region();
      if (r && !r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
      tx += torque[3 * i]; ty += torque[3 * i + 1]; tz += torque[3 * i + 2];
      if (cx.kind !== 'null') torque[3 * i] = val(cx, i);
      if (cy.kind !== 'null') torque[3 * i + 1] = val(cy, i);
      if (cz.kind !== 'null') torque[3 * i + 2] = val(cz, i);
    }
    this.vec[0] = tx; this.vec[1] = ty; this.vec[2] = tz;
  }

  minPostForce(): void {
    this.postForce();
  }

  computeVector(i: number): number {
    return this.vec[i];
  }

  private region(): Region | null {
    if (this.regionId === null) return null;
    this.regionObj ??= this.sys.region(this.regionId);
    return this.regionObj;
  }
}
