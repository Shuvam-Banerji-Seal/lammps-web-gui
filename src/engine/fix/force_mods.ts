import { Fix } from './fix';
import { StyleError } from '../force/types';
import { inumeric, missingArgs, numeric } from '../force/util';
import { EQUAL_STYLES } from '../argInfo';
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

const parseComp = (w: string | undefined, what: string, style: string, allowNull: boolean): Component => {
  if (w === undefined) throw new StyleError(`fix ${style}: missing ${what}`);
  // As native LAMMPS does (src/fix_setforce.cpp, fix_aveforce.cpp, fix_addforce.cpp at b332f28): NULL is a value only
  // where the style allows it; elsewhere it is read as a number and utils::numeric reports it. A v_name is kept; the
  // variable is looked up at init (ForceModFix.checkComponents).
  if (w === 'NULL' && allowNull) return { kind: 'null' };
  if (w.startsWith('v_')) return { kind: 'var', name: w.slice(2) };
  return { kind: 'const', v: numeric(w) };
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
  protected parse(style: string, args: string[], allowNull: boolean): void {
    if (args.length < 3) throw new StyleError(`usage: fix ID group ${style} fx fy fz [keyword value ...]`);
    this.comps = [
      parseComp(args[0], 'fx', style, allowNull),
      parseComp(args[1], 'fy', style, allowNull),
      parseComp(args[2], 'fz', style, allowNull),
    ];
    // As native LAMMPS does (src/fix_setforce.cpp, fix_aveforce.cpp and fix_addforce.cpp constructors at b332f28): each
    // keyword in turn; an unknown word is refused before a missing value is looked for
    for (let k = 3; k < args.length; k += 2) {
      const key = args[k];
      const value = args[k + 1];
      if (key === 'region') {
        if (value === undefined) throw this.missingValue(key);
        if (!this.sys.regions.has(value)) throw new StyleError(`Region ${value} for fix ${style} does not exist`);
        this.regionId = value;
      } else {
        this.keyword(key, value);
      }
    }
  }

  /**
   * The component variables, checked when a run starts as native LAMMPS does (src/fix_setforce.cpp,
   * fix_aveforce.cpp and fix_addforce.cpp init at b332f28): each exists and is equal-style (or atom-style where the
   * style allows it), with native's messages (aveforce names no variable), then the region.
   */
  protected checkComponents(): void {
    const st = this.style;
    for (const c of this.comps) {
      if (c.kind !== 'var') continue;
      const v = this.sys.vars.get(c.name);
      const equal = v !== undefined && EQUAL_STYLES.has(v.style);
      const atom = v !== undefined && (v.style === 'atom' || v.style === 'atomfile');
      if (st === 'aveforce') {
        if (!v) throw new StyleError('Variable name for fix aveforce does not exist');
        if (!equal) throw new StyleError('Variable for fix aveforce is invalid style');
      } else {
        if (!v) throw new StyleError(`Variable ${c.name} for fix ${st} does not exist`);
        if (!equal && !atom) throw new StyleError(`Variable ${c.name} for fix ${st} is invalid style`);
      }
    }
  }

  /** Variables a style checks after the components (addforce: its energy variable). */
  protected checkMoreVariables(): void {}

  init(): void {
    this.checkComponents();
    this.checkMoreVariables();
    if (this.regionId !== null && !this.sys.regions.has(this.regionId)) throw new StyleError(`Region ${this.regionId} for fix ${this.style} does not exist`);
    this.regionObj = null;
  }

  /** A keyword other than region (subclass-specific); must throw for unknown ones. */
  protected keyword(key: string, _value: string | undefined): void {
    throw new StyleError(`unknown fix ${this.style} keyword '${key}'`);
  }

  /** A keyword without its value: "Illegal fix {style} {keyword} command: missing argument(s)" (aveforce overrides). */
  protected missingValue(key: string): StyleError {
    return missingArgs(`fix ${this.style} ${key}`);
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
    this.parse('setforce', args, true);
  }

  protected keyword(key: string, _value: string | undefined): void {
    // As native LAMMPS does (src/fix_setforce.cpp FixSetForce::FixSetForce at b332f28): "Unknown fix setforce keyword: {}".
    throw new StyleError(`Unknown fix setforce keyword: ${key}`);
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
    this.parse('addforce', args, false);
  }

  protected keyword(key: string, value: string | undefined): void {
    // As native LAMMPS does (src/fix_addforce.cpp FixAddForce::FixAddForce at b332f28): every reads utils::inumeric (> 0),
    // energy keeps a v_name (looked up at init); an unknown word comes before a missing value
    if (key === 'every') {
      if (value === undefined) throw this.missingValue(key);
      const n = inumeric(value);
      if (n <= 0) throw new StyleError(`Invalid fix addforce every argument: ${n}`);
      this.nevery = n;
      return;
    }
    if (key === 'energy') {
      if (value === undefined) throw this.missingValue(key);
      if (!value.startsWith('v_')) throw new StyleError(`Invalid fix addforce energy argument: ${value}`);
      this.energyVar = value.slice(2);
      return;
    }
    // As native LAMMPS does (src/fix_addforce.cpp FixAddForce::FixAddForce at b332f28): "Unknown fix addforce keyword: {}".
    throw new StyleError(`Unknown fix addforce keyword: ${key}`);
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

  /** fix_addforce.cpp init: the energy variable exists and is atom-style; a variable energy needs a variable force. */
  protected checkMoreVariables(): void {
    if (this.energyVar !== null) {
      const v = this.sys.vars.get(this.energyVar);
      if (!v) throw new StyleError(`Variable ${this.energyVar} for fix addforce does not exist`);
      if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Variable ${this.energyVar} for fix addforce is invalid style`);
      if (this.comps.every((c) => c.kind !== 'var')) throw new StyleError('Cannot use variable energy with constant force in fix addforce');
    }
  }

  minSetup(): void {
    // fix_addforce.html: "The *energy* keyword is required if the added force is
    // defined with one or more variables, and you are performing energy
    // minimization via the "minimize" command." (native init: "Must use variable energy with fix addforce")
    if (this.energyVar === null && this.comps.some((c) => c.kind === 'var')) {
      throw new StyleError('Must use variable energy with fix addforce');
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
    this.parse('aveforce', args, true);
  }

  protected keyword(_key: string, _value: string | undefined): void {
    // As native LAMMPS does (src/fix_aveforce.cpp FixAveForce::FixAveForce at b332f28): any keyword other than region
    // gives "Illegal fix aveforce command".
    throw new StyleError('Illegal fix aveforce command');
  }

  /** fix_aveforce.cpp: region without its value is "Illegal fix aveforce command" too. */
  protected missingValue(_key: string): StyleError {
    return new StyleError('Illegal fix aveforce command');
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
