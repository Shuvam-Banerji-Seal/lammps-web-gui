import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Region } from '../region';

/*
 * Force-modifying fixes: setforce, addforce, aveforce. Written from the
 * LAMMPS documentation (plans/lammps-docs/*.rst, docs.lammps.org), never from
 * LAMMPS source code.
 *
 * Shared behaviour — the "output" section of each page: the fix "computes a
 * global 3-vector of forces" (fix_setforce.rst / fix_aveforce.rst) or "a
 * global scalar and a global three-vector of forces" (fix_addforce.rst).
 * The vector is the total force on the atoms the fix operates on (the fix
 * group, intersected with the region when the region keyword is given)
 * BEFORE the fix changes any of them; it is tallied in post_force
 * (docs.lammps.org/Developer_flow.html), the hook where these fixes act,
 * including at step 0 (Fix.setup() runs after the setup force evaluation).
 * The tallied values are "extensive" (extvector = 1, extscalar = 1).
 *
 * fix_modify (fix_modify.rst): "*energy* value = *yes* or *no*" and
 * "*virial* value = *yes* or *no*"; "explained at the bottom of their doc
 * page.  *Energy yes* will add a" "contribution to the potential energy of
 * the system.  More" — only fix addforce supports these (its page:
 * "The :doc:`fix_modify <fix_modify>` *energy* option is supported by");
 * setforce and aveforce keep the base class's error for them.
 */

/** One force component: NULL (left alone), a constant, or a v_name variable. */
type Component = { kind: 'null' } | { kind: 'const'; v: number } | { kind: 'var'; name: string };

/** A component resolved for the current step: variables become per-atom values. */
type Resolved = { kind: 'null' } | { kind: 'const'; v: number } | { kind: 'vals'; v: Float64Array };

const val = (r: Resolved, i: number): number => (r.kind === 'const' ? r.v : r.kind === 'vals' ? r.v[i] : 0);

const parseComp = (sys: System, w: string | undefined, what: string, style: string, allowNull: boolean, equalOnly: boolean): Component => {
  if (w === undefined) throw new StyleError(`fix ${style}: missing ${what}`);
  if (w === 'NULL') {
    if (!allowNull) throw new StyleError(`fix ${style}: NULL is not valid for ${what} (expected a number or v_name)`);
    return { kind: 'null' };
  }
  if (w.startsWith('v_')) {
    const name = w.slice(2);
    const v = sys.vars.get(name);
    if (!v) throw new StyleError(`fix ${style}: variable ${name} does not exist`);
    if (equalOnly && (v.style === 'atom' || v.style === 'atomfile' || v.style === 'vector')) {
      throw new StyleError(`fix ${style}: variable ${name} must be equal-style (is ${v.style})`);
    }
    return { kind: 'var', name };
  }
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`fix ${style}: expected a number, NULL or v_name for ${what}, got '${w}'`);
  return { kind: 'const', v };
};

abstract class ForceModFix extends Fix {
  protected comps: [Component, Component, Component] = [{ kind: 'null' }, { kind: 'null' }, { kind: 'null' }];
  protected regionId: string | null = null;
  private regionObj: Region | null = null;
  /** Last total force on the group before the fix changed it (f_ID[1..3]). */
  private vec = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extvector = 1;
  }

  /** Parses `fx fy fz [keyword value ...]`; called from the subclass constructor. */
  protected parse(style: string, args: string[], allowNull: boolean, equalOnly: boolean): void {
    if (args.length < 3) throw new StyleError(`usage: fix ID group ${style} fx fy fz [keyword value ...]`);
    this.comps = [
      parseComp(this.sys, args[0], 'fx', style, allowNull, equalOnly),
      parseComp(this.sys, args[1], 'fy', style, allowNull, equalOnly),
      parseComp(this.sys, args[2], 'fz', style, allowNull, equalOnly),
    ];
    for (let k = 3; k < args.length; k += 2) {
      const key = args[k];
      const value = args[k + 1];
      if (value === undefined) throw new StyleError(`fix ${style} keyword '${key}' needs a value`);
      if (key === 'region') {
        this.sys.region(value);
        this.regionId = value;
      } else {
        this.keyword(key, value);
      }
    }
  }

  /** A keyword other than region (subclass-specific); must throw for unknown ones. */
  protected keyword(key: string, _value: string): void {
    throw new StyleError(`unknown fix ${this.style} keyword '${key}'`);
  }

  /** True when atom i is in the fix group and (with the region keyword) the region. */
  protected applies(i: number): boolean {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return false;
    const r = this.region();
    return !r || r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
  }

  private region(): Region | null {
    if (this.regionId === null) return null;
    this.regionObj ??= this.sys.region(this.regionId);
    return this.regionObj;
  }

  /** Variables become per-atom values for this step (atom-style vars) or one shared value. */
  protected resolve(): [Resolved, Resolved, Resolved] {
    const r = (c: Component): Resolved => (c.kind === 'var' ? { kind: 'vals', v: this.sys.atomVariable(c.name) } : c);
    return [r(this.comps[0]), r(this.comps[1]), r(this.comps[2])];
  }

  protected tally(sx: number, sy: number, sz: number): void {
    this.vec[0] = sx;
    this.vec[1] = sy;
    this.vec[2] = sz;
  }

  computeVector(i: number): number { return this.vec[i]; }
}

/**
 * fix ID group-ID setforce fx fy fz keyword value ... —
 * docs.lammps.org/fix_setforce.html. From fix_setforce.rst, verbatim:
 *
 *   fix ID group-ID setforce fx fy fz keyword value ...
 *
 *   keyword = *region*
 *   *region* value = region-ID
 *     region-ID = ID of region atoms must be in to have added force
 *
 *   Any of the fx,fy,fz values can be specified as NULL which means do not
 *   alter the force component in that dimension.
 *
 *   This fix computes a global 3-vector of forces, which can be accessed
 *
 * (the vector is the total force on the group before the fix changes it;
 * per the page it is "extensive"). Default:
 *
 *   none
 *
 * Minimization (same page): "The forces due to this fix are imposed during
 * an energy minimization, invoked by the minimize command, but you cannot
 * set forces to any value besides zero when performing a minimization."
 */
export class FixSetForce extends ForceModFix {
  readonly style = 'setforce';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.parse('setforce', args, true, false);
  }

  postForce(): void {
    const s = this.sys.state;
    const f = s.f;
    const [cx, cy, cz] = this.resolve();
    let sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.applies(i)) continue;
      // the vector is the force BEFORE the fix changes this atom's forces
      sx += f[3 * i]; sy += f[3 * i + 1]; sz += f[3 * i + 2];
      if (cx.kind !== 'null') f[3 * i] = val(cx, i);
      if (cy.kind !== 'null') f[3 * i + 1] = val(cy, i);
      if (cz.kind !== 'null') f[3 * i + 2] = val(cz, i);
    }
    this.tally(sx, sy, sz);
  }

  minPostForce(): void {
    const s = this.sys.state;
    const [cx, cy, cz] = this.resolve();
    const check = (c: Resolved, what: string): void => {
      if (c.kind === 'const' && c.v !== 0) {
        throw new StyleError(`fix ${this.id} setforce: cannot set ${what} to a non-zero value during minimization`);
      }
      if (c.kind === 'vals') {
        for (let i = 0; i < s.n; i++) {
          if (c.v[i] !== 0 && this.applies(i)) {
            throw new StyleError(`fix ${this.id} setforce: cannot set ${what} to a non-zero value during minimization`);
          }
        }
      }
    };
    check(cx, 'fx'); check(cy, 'fy'); check(cz, 'fz');
    this.postForce();
  }
}

/**
 * fix ID group-ID addforce fx fy fz keyword value ... —
 * docs.lammps.org/fix_addforce.html. From fix_addforce.rst, verbatim:
 *
 *   fix ID group-ID addforce fx fy fz keyword value ...
 *
 *   any of fx,fy,fz can be a variable (see below)
 *
 *   keyword = *every* or *region* or *energy*
 *   *every* value = Nevery
 *     Nevery = add force every this many time steps
 *   *region* value = region-ID
 *     region-ID = ID of region atoms must be in to have added force
 *   *energy* value = v_name
 *     v_name = variable with name that calculates the potential energy of each atom in the added force field
 *
 *   The *energy* keyword is not allowed if the added force is a constant
 *
 *   E = -\vec x \cdot \vec F = -(x f_x + y f_y + z f_z),
 *
 *   so that :math:`-\vec\nabla E = \vec F`.
 *
 * (when all components are constants, LAMMPS computes the energy itself;
 * with variables and no energy keyword "LAMMPS will set the energy to 0.0").
 * Output (same page): "This fix computes a global scalar and a global
 * three-vector of forces," and "The vector is the total force on the group
 * of atoms before the forces" "on individual atoms are changed by the fix."
 * fix_modify: "The :doc:`fix_modify <fix_modify>` *energy* option is supported by"
 * this fix (default energy no), as is the *virial* option (default no).
 * Default:
 *
 *   The option default for the every keyword is every = 1.
 */
export class FixAddForce extends ForceModFix {
  readonly style = 'addforce';
  private energyVar: string | null = null;
  private energyTallied = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.scalarFlag = true;
    this.extscalar = 1;
    this.energyGlobal = true;
    this.virialGlobal = true;
    this.parse('addforce', args, false, false);
    if (this.energyVar !== null && this.comps.every((c) => c.kind === 'const')) {
      throw new StyleError('fix addforce: the energy keyword is not allowed when all force components are constants');
    }
  }

  protected keyword(key: string, value: string): void {
    if (key === 'every') {
      const n = Number(value);
      if (!Number.isInteger(n) || n < 1) throw new StyleError(`fix addforce every must be a positive integer, got '${value}'`);
      this.nevery = n;
      return;
    }
    if (key === 'energy') {
      if (!value.startsWith('v_')) throw new StyleError(`fix addforce energy must be v_name, got '${value}'`);
      const name = value.slice(2);
      const v = this.sys.vars.get(name);
      if (!v) throw new StyleError(`fix addforce: variable ${name} does not exist`);
      if (v.style !== 'atom') throw new StyleError(`fix addforce: energy variable ${name} must be atom-style (is ${v.style})`);
      this.energyVar = name;
      return;
    }
    throw new StyleError(`unknown fix addforce keyword '${key}'`);
  }

  postForce(): void {
    const s = this.sys.state;
    // "Nevery = add force every this many time steps" (default every = 1)
    if (this.nevery > 1 && s.step % this.nevery !== 0) return;
    const { f, x } = s;
    const [cx, cy, cz] = this.resolve();
    const eVals = this.energyVar !== null ? this.sys.atomVariable(this.energyVar) : null;
    const constE = cx.kind === 'const' && cy.kind === 'const' && cz.kind === 'const';
    const w = this.thermoVirial ? this.virial : null;
    if (w) w.fill(0);
    let sx = 0, sy = 0, sz = 0, e = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.applies(i)) continue;
      sx += f[3 * i]; sy += f[3 * i + 1]; sz += f[3 * i + 2];
      const fax = val(cx, i), fay = val(cy, i), faz = val(cz, i);
      f[3 * i] += fax; f[3 * i + 1] += fay; f[3 * i + 2] += faz;
      if (eVals) e += eVals[i];
      else if (constE) e -= x[3 * i] * fax + x[3 * i + 1] * fay + x[3 * i + 2] * faz;
      if (w) {
        w[0] += x[3 * i] * fax; w[1] += x[3 * i + 1] * fay; w[2] += x[3 * i + 2] * faz;
        w[3] += x[3 * i] * fay; w[4] += x[3 * i] * faz; w[5] += x[3 * i + 1] * faz;
      }
    }
    this.tally(sx, sy, sz);
    this.energyTallied = e;
  }

  /** The potential energy of the added force field (fix_modify energy / f_ID scalar). */
  energy(): number { return this.energyTallied; }
  computeScalar(): number { return this.energyTallied; }

  minSetup(): void {
    // fix_addforce.html: "The *energy* keyword is required if the added force is
    // defined with one or more variables, and you are performing energy
    // minimization via the "minimize" command."
    if (this.energyVar === null && this.comps.some((c) => c.kind === 'var')) {
      throw new StyleError(`fix ${this.id} addforce: the energy keyword is required when the force uses variables and minimize is performed`);
    }
  }

  minPostForce(): void {
    this.postForce();
  }
}

/**
 * fix ID group-ID aveforce fx fy fz keyword value ... —
 * docs.lammps.org/fix_aveforce.html. From fix_aveforce.rst, verbatim:
 *
 *   fix ID group-ID aveforce fx fy fz keyword value ...
 *
 *   Apply an additional external force to a group of atoms in such a way
 *   that every atom experiences the same force.
 *
 *   The existing force is averaged for the group of atoms, component by
 *   component.  The actual force on each atom is then set to the average
 *   value plus the component specified in this command.  This means each
 *   atom in the group receives the same force.
 *
 *   Any of the *fx*, *fy*, or *fz* values can be specified as :code:`NULL`, which
 *   means the force in that dimension is not changed.  Note that this is not the
 *   same as specifying a 0.0 value, since that sets all forces to the same
 *   average value without adding in any additional force.
 *
 *   This fix computes a global three-vector of forces, which can be accessed
 *
 * (the total force on the group before the fix changes it; "extensive").
 * Default:
 *
 *   none
 */
export class FixAveForce extends ForceModFix {
  readonly style = 'aveforce';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.parse('aveforce', args, true, true);
  }

  postForce(): void {
    const s = this.sys.state;
    const f = s.f;
    const [cx, cy, cz] = this.resolve();
    let sx = 0, sy = 0, sz = 0, count = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.applies(i)) continue;
      sx += f[3 * i]; sy += f[3 * i + 1]; sz += f[3 * i + 2];
      count++;
    }
    this.tally(sx, sy, sz);
    if (count === 0) return;
    const ax = sx / count, ay = sy / count, az = sz / count;
    for (let i = 0; i < s.n; i++) {
      if (!this.applies(i)) continue;
      if (cx.kind !== 'null') f[3 * i] = ax + val(cx, i);
      if (cy.kind !== 'null') f[3 * i + 1] = ay + val(cy, i);
      if (cz.kind !== 'null') f[3 * i + 2] = az + val(cz, i);
    }
  }

  minPostForce(): void {
    this.postForce();
  }
}
