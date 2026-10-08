import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNumOrVar, valueOf, type NumOrVar } from './util';
import { massOf } from '../atoms';

/*
 * Force-style fixes from wave "force_ext", written only from the LAMMPS
 * documentation (docs.lammps.org pages cited per class). The engine stores
 * forces in internal units: F_internal = F_lammps / ftm2v (integrators apply
 * v += dt * ftm2v * F / m), energies in LAMMPS energy units, velocities and
 * positions in LAMMPS units.
 */

/**
 * fix ID group-ID viscous gamma keyword values ... —
 * docs.lammps.org/fix_viscous.html, Syntax:
 *
 *   fix ID group-ID viscous gamma keyword values ...
 *
 * "keyword = *scale*", "*scale* values = type ratio", "type = atom type
 * (1-N)", "ratio = factor to scale the damping coefficient by".
 * Description: "The damping force :math:`F_i` is given by
 * :math:`F_i = - \gamma v_i`." and "If the optional keyword *scale* is used, :math:`\gamma` can scaled up or
 * down by the specified factor for atoms of that type.  It can be used
 * multiple times to adjust :math:`\gamma` for several atom types."
 * "You should specify gamma in force/velocity units."
 * "The forces due to this fix are imposed during an energy minimization,
 * invoked by the :doc:`minimize <minimize>` command." Default: "none".
 * "None of the :doc:`fix_modify <fix_modify>` options are
 * relevant to this fix.  No global or per-atom quantities are stored by
 * this fix for access by various :doc:`output commands <Howto_output>`."
 */
export class FixViscous extends Fix {
  readonly style = 'viscous';
  private gamma = 0;
  private ratio: Float64Array = new Float64Array(1).fill(1);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('usage: fix ID group-ID viscous gamma keyword values ...');
    this.gamma = Number(args[0]);
    if (args[0].startsWith('v_') || !Number.isFinite(this.gamma)) {
      throw new StyleError(`fix viscous: gamma must be a number in force/velocity units, got '${args[0]}'`);
    }
    const s = sys.state;
    this.ratio = new Float64Array(s.ntypes + 1).fill(1);
    for (let k = 1; k < args.length;) {
      if (args[k] !== 'scale') throw new StyleError(`fix viscous: unknown keyword '${args[k]}' (scale is the only keyword)`);
      const t = Number(args[k + 1]), r = Number(args[k + 2]);
      if (!Number.isInteger(t) || t < 1 || t > s.ntypes) throw new StyleError(`fix viscous scale: type must be an atom type (1-${s.ntypes}), got '${args[k + 1]}'`);
      if (!(r > 0)) throw new StyleError(`fix viscous scale: ratio must be > 0, got '${args[k + 2]}'`);
      this.ratio[t] = r;
      k += 3;
    }
  }

  /** F_i = -gamma_type v_i, gamma_type = gamma * ratio (scale keyword). */
  postForce(): void {
    const s = this.sys.state;
    const { f, v, type, mask } = s;
    const k = -this.gamma / s.units.ftm2v;
    const bit = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const c = k * this.ratio[type[i]];
      f[3 * i] += c * v[3 * i];
      f[3 * i + 1] += c * v[3 * i + 1];
      f[3 * i + 2] += c * v[3 * i + 2];
    }
  }

  minPostForce(): void { this.postForce(); }
}

/**
 * fix ID group gravity magnitude style args — docs.lammps.org/fix_gravity.html,
 * Syntax:
 *
 *   fix ID group gravity magnitude style args
 *
 * "*chute* args = angle", "angle = angle in +x away from -z or -y axis in
 * 3d/2d (in degrees)"; "*spherical* args = phi theta", "phi = azimuthal angle
 * from +x axis (in degrees)", "theta = angle from +z or +y axis in 3d/2d (in
 * degrees)"; "*vector* args = x y z", "x y z = vector direction to apply the
 * acceleration". Description: "Impose an additional acceleration on each
 * particle in the group." "The *magnitude* of the acceleration is specified
 * in force/mass units." Chute: "For 3d systems, the tilt is away from the z axis; for 2d
 * systems, the tilt is away from the y axis." Spherical: "Thus for acceleration acting in the -z
 * direction, *theta* would be 180.0 (or -180.0).  *Theta* = 90.0 and
 * *phi* = -90.0 would mean acceleration acts in the -y direction.  For
 * 2d systems, *phi* is ignored and *theta* is an angle in the xy plane
 * where *theta* = 0.0 is the y-axis." Vector: "Only the direction of the vector is important; it's
 * length is ignored.  For 2d systems, the *z* component is ignored."
 * "Any of the quantities *magnitude*, *angle*, *phi*, *theta*, *x*, *y*,
 * *z* which define the gravitational magnitude and direction, can be
 * specified as an equal-style :doc:`variable <variable>`." ... "the variable will be evaluated each
 * timestep, and its value used to determine the quantity."
 *
 * Output: "This scalar is the
 * gravitational potential energy of the particles in the defined field,
 * namely mass \* (g dot x) for each particles, where x and mass are the
 * particles position and mass, and g is the gravitational field.  The
 * scalar value calculated by this fix is "extensive"." The
 * The fix_modify energy option is supported: "The :doc:`fix_modify
 * <fix_modify>` *energy* option is supported by this fix to add the
 * gravitational potential energy of the system to the global potential energy
 * of the system as part of :doc:`thermodynamic output <thermo_style>`."
 * "The default setting for this fix is :doc:`fix_modify energy no
 * <fix_modify>`." "This fix is not invoked during :doc:`energy minimization
 * <minimize>`." Default: "none".
 */
export class FixGravity extends Fix {
  readonly style = 'gravity';
  private magnitude: NumOrVar;
  private kind: 'chute' | 'spherical' | 'vector';
  private a1: NumOrVar;
  private a2: NumOrVar | null = null;
  private a3: NumOrVar | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('usage: fix ID group gravity magnitude style args');
    this.magnitude = parseNumOrVar(args[0], 'gravity magnitude');
    const style = args[1];
    if (style === 'chute') {
      if (args.length !== 3) throw new StyleError('usage: fix ID group gravity magnitude chute angle');
      this.a1 = parseNumOrVar(args[2], 'gravity chute angle');
      this.kind = 'chute';
    } else if (style === 'spherical') {
      if (args.length !== 4) throw new StyleError('usage: fix ID group gravity magnitude spherical phi theta');
      this.a1 = parseNumOrVar(args[2], 'gravity spherical phi');
      this.a2 = parseNumOrVar(args[3], 'gravity spherical theta');
      this.kind = 'spherical';
    } else if (style === 'vector') {
      if (args.length !== 5) throw new StyleError('usage: fix ID group gravity magnitude vector x y z');
      this.a1 = parseNumOrVar(args[2], 'gravity vector x');
      this.a2 = parseNumOrVar(args[3], 'gravity vector y');
      this.a3 = parseNumOrVar(args[4], 'gravity vector z');
      this.kind = 'vector';
      // a constant zero direction is rejected when the fix is defined; a
      // variable that evaluates to zero at some step is caught in gvec()
      if (typeof this.a1 === 'number' && typeof this.a2 === 'number' && typeof this.a3 === 'number'
        && this.a1 === 0 && this.a2 === 0 && this.a3 === 0) {
        throw new StyleError(`fix ${id} gravity: the vector direction must be non-zero`);
      }
    } else {
      throw new StyleError(`fix gravity: style '${style}' is not supported (only chute, spherical and vector are documented with args on docs.lammps.org/fix_gravity.html)`);
    }
    this.scalarFlag = true;
    this.extscalar = 1;
    this.energyGlobal = true;
  }

  /** Current gravitational field g (force/mass units); variables evaluated now. */
  private gvec(out: Float64Array): void {
    const two = this.sys.state.dimension === 2;
    const rad = Math.PI / 180;
    if (this.kind === 'chute') {
      const a = valueOf(this.sys, this.a1) * rad;
      // "angle in +x away from -z or -y axis": tilt toward +x from -z (3d) / -y (2d)
      out[0] = Math.sin(a);
      out[1] = two ? -Math.cos(a) : 0;
      out[2] = two ? 0 : -Math.cos(a);
    } else if (this.kind === 'spherical') {
      const phi = valueOf(this.sys, this.a1) * rad;
      const theta = valueOf(this.sys, this.a2!) * rad;
      if (two) {
        // "theta is an angle in the xy plane where theta = 0.0 is the y-axis"
        out[0] = Math.sin(theta);
        out[1] = Math.cos(theta);
        out[2] = 0;
      } else {
        const st = Math.sin(theta);
        out[0] = st * Math.cos(phi);
        out[1] = st * Math.sin(phi);
        out[2] = Math.cos(theta);
      }
    } else {
      const x = valueOf(this.sys, this.a1);
      const y = valueOf(this.sys, this.a2!);
      const z = two ? 0 : valueOf(this.sys, this.a3!);
      const len = Math.sqrt(x * x + y * y + z * z);
      if (!(len > 0)) throw new StyleError(`fix ${this.id} gravity: the vector direction must be non-zero`);
      out[0] = x / len;
      out[1] = y / len;
      out[2] = z / len;
    }
  }

  /** F = mass * magnitude * direction; "the same acceleration to each atom". */
  postForce(): void {
    const s = this.sys.state;
    const g = new Float64Array(3);
    this.gvec(g);
    const c = valueOf(this.sys, this.magnitude) / s.units.ftm2v;
    const { f, mask } = s;
    const bit = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const m = massOf(s, i) * c;
      f[3 * i] += m * g[0];
      f[3 * i + 1] += m * g[1];
      f[3 * i + 2] += m * g[2];
    }
  }

  /**
   * Potential energy of the field: -mass * (g dot x) for each particle in the
   * group, in energy units. The doc names the expression "mass \* (g dot x)";
   * the potential energy of the field is its negative, so that -Grad(U) =
   * F = +mass * g (the same -x.qE convention the fix_efield page states
   * explicitly), which is also what native LAMMPS outputs (oracle cases
   * w2fdamp_gravity / w2fdamp_gravity_chute).
   */
  private energyNow(): number {
    const s = this.sys.state;
    const g = new Float64Array(3);
    this.gvec(g);
    const k = valueOf(this.sys, this.magnitude) * s.units.mvv2e;
    const { x, mask } = s;
    const bit = this.groupBit;
    let e = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      e += massOf(s, i) * (g[0] * x[3 * i] + g[1] * x[3 * i + 1] + g[2] * x[3 * i + 2]);
    }
    return -e * k;
  }

  energy(): number { return this.energyNow(); }
  computeScalar(): number { return this.energyNow(); }
}

/**
 * fix ID group-ID lineforce x y z — docs.lammps.org/fix_lineforce.html,
 * Syntax:
 *
 *   fix ID group-ID lineforce x y z
 *
 * "x y z = direction of line as a 3-vector". Description: "Adjust the forces on each atom in the group so that only the component
 * of force along the linear direction specified by the vector (x,y,z)
 * remains.  This is done by subtracting out components of force in the
 * plane perpendicular to the line." "The forces due to this fix are imposed during an energy minimization,
 * invoked by the :doc:`minimize <minimize>` command." Default: "none".
 */
export class FixLineforce extends Fix {
  readonly style = 'lineforce';
  private n = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('usage: fix ID group-ID lineforce x y z');
    let len = 0;
    for (let d = 0; d < 3; d++) {
      const v = Number(args[d]);
      if (!Number.isFinite(v)) throw new StyleError(`fix lineforce: '${args[d]}' is not a number (x y z = direction of line as a 3-vector)`);
      this.n[d] = v;
      len += v * v;
    }
    len = Math.sqrt(len);
    if (!(len > 0)) throw new StyleError('fix lineforce: the line direction vector must be non-zero');
    for (let d = 0; d < 3; d++) this.n[d] /= len;
  }

  /** Keep only the force component along the (unit) line direction. */
  postForce(): void {
    const s = this.sys.state;
    const { f, mask } = s;
    const [nx, ny, nz] = this.n;
    const bit = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const d = f[3 * i] * nx + f[3 * i + 1] * ny + f[3 * i + 2] * nz;
      f[3 * i] = d * nx;
      f[3 * i + 1] = d * ny;
      f[3 * i + 2] = d * nz;
    }
  }

  minPostForce(): void { this.postForce(); }
}

/**
 * fix ID group-ID planeforce x y z — docs.lammps.org/fix_planeforce.html,
 * Syntax:
 *
 *   fix ID group-ID planeforce x y z
 *
 * "x y z = 3-vector that is normal to the plane". Description: "Adjust the
 * forces on each atom in the group so that only the
 * components of force in the plane specified by the normal vector
 * (x,y,z) remain.  This is done by subtracting out the component of
 * force perpendicular to the plane." "The
 * forces due to this fix are imposed during an energy minimization,
 * invoked by the :doc:`minimize <minimize>` command." Default: "none".
 */
export class FixPlaneforce extends Fix {
  readonly style = 'planeforce';
  private n = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('usage: fix ID group-ID planeforce x y z');
    let len = 0;
    for (let d = 0; d < 3; d++) {
      const v = Number(args[d]);
      if (!Number.isFinite(v)) throw new StyleError(`fix planeforce: '${args[d]}' is not a number (x y z = 3-vector that is normal to the plane)`);
      this.n[d] = v;
      len += v * v;
    }
    len = Math.sqrt(len);
    if (!(len > 0)) throw new StyleError('fix planeforce: the plane normal vector must be non-zero');
    for (let d = 0; d < 3; d++) this.n[d] /= len;
  }

  /** Remove the force component along the (unit) plane normal. */
  postForce(): void {
    const s = this.sys.state;
    const { f, mask } = s;
    const [nx, ny, nz] = this.n;
    const bit = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const d = f[3 * i] * nx + f[3 * i + 1] * ny + f[3 * i + 2] * nz;
      f[3 * i] -= d * nx;
      f[3 * i + 1] -= d * ny;
      f[3 * i + 2] -= d * nz;
    }
  }

  minPostForce(): void { this.postForce(); }
}

/** One E-field component: a constant, or a variable (equal-style or atom-style). */
type EComp = { const: number } | { var: string; atom: boolean };

/**
 * fix ID group-ID efield ex ey ez keyword value ... —
 * docs.lammps.org/fix_efield.html, Syntax:
 *
 *   fix ID group-ID style ex ey ez keyword value ...
 *
 * "ex,ey,ez = E-field component values (electric field units)", "any of
 * ex,ey,ez can be a variable", "keyword = *region* or *energy* or
 * *potential*". Description: "Add a force :math:`\vec{F} = q\vec{E}` to each charged atom in the group due to an
 * external electric field being applied to the system." "If the *region* keyword is used, the atom must also be in the
 * specified geometric :doc:`region <region>` in order to have force added
 * to it." Energy: "The *energy* keyword is not allowed if the added field is a constant
 * vector (ex,ey,ez), with all components defined as numeric constants
 * and not as variables." ... the energy is computed directly as
 * "U_{efield} = -\vec{x} \cdot q\vec{E} = -q (x\cdot E_x + y\cdot E_y + z\cdot Ez),".
 * "If the keyword is not used, LAMMPS will set
 * the energy to 0.0, which is typically fine for dynamics." Potential:
 * "The energy added by this
 * fix is then calculated as the electric potential multiplied by charge."
 * "Thus, when the *potential*
 * keyword is specified, the *energy* keyword must not be used."
 * "the *potential* keyword is not allowed if the added field is a
 * constant vector". Output: "This fix computes a global scalar and a global 3-vector of forces,
 * which can be accessed by various :doc:`output commands
 * <Howto_output>`.  The scalar is the potential energy discussed above.
 * The vector is the total force added to the group of atoms.  The scalar
 * and vector values calculated by this fix are "extensive"."
 * "The :doc:`fix_modify <fix_modify>` *energy* option is supported by this
 * fix to add the potential energy inferred by the added force due to the
 * electric field to the global potential energy of the system as part of"
 * ("The default setting for
 * this fix is :doc:`fix_modify energy no <fix_modify>`"), same for virial
 * ("The :doc:`fix_modify <fix_modify>` *virial* option is supported by this
 * fix to add the contribution due to the added forces on atoms").
 * "The forces due to this fix are imposed during an energy minimization,
 * invoked by the :doc:`minimize <minimize>` command." Default: "none".
 *
 * Point dipoles ("If the system
 * contains point-dipoles, also add a torque ...") are not supported by this
 * engine: any dipole request is a StyleError.
 */
export class FixEfield extends Fix {
  readonly style = 'efield';
  private comp: EComp[] = [];
  private allConst = true;
  private regionId: string | null = null;
  private energyVar: string | null = null;
  private potentialVar: string | null = null;
  /** Per-atom force added by the last postForce, in internal units. */
  private fadd: Float64Array = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const s = sys.state;
    if (s.atomStyle !== 'charge' && s.atomStyle !== 'full') {
      throw new StyleError(`fix efield requires atoms that store a charge; atom_style ${s.atomStyle} does not (use atom_style charge or full)`);
    }
    if (args.length < 3) throw new StyleError('usage: fix ID group-ID efield ex ey ez keyword value ...');
    this.comp = [FixEfield.parseComp(sys, args[0], 'ex'), FixEfield.parseComp(sys, args[1], 'ey'), FixEfield.parseComp(sys, args[2], 'ez')];
    this.allConst = this.comp.every((c) => 'const' in c);
    for (let k = 3; k < args.length; k += 2) {
      const key = args[k];
      const val = args[k + 1];
      if (val === undefined) throw new StyleError(`fix efield keyword '${key}' needs a value`);
      if (key === 'region') {
        // "region-ID = ID of region atoms must be in to have added force"
        sys.region(val);
        this.regionId = val;
      } else if (key === 'energy') {
        if (this.allConst) throw new StyleError('fix efield: the energy keyword is not allowed if the added field is a constant vector');
        if (this.potentialVar !== null) throw new StyleError('fix efield: when the potential keyword is specified, the energy keyword must not be used');
        this.energyVar = FixEfield.atomVar(sys, val, 'energy');
      } else if (key === 'potential') {
        if (this.allConst) throw new StyleError('fix efield: the potential keyword is not allowed if the added field is a constant vector');
        if (this.energyVar !== null) throw new StyleError('fix efield: when the potential keyword is specified, the energy keyword must not be used');
        this.potentialVar = FixEfield.atomVar(sys, val, 'potential');
      } else {
        throw new StyleError(`fix efield: unknown keyword '${key}' (region, energy and potential are the documented keywords; point-dipole handling is not supported by this engine)`);
      }
    }
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
    this.virialGlobal = true;
    this.fadd = new Float64Array(3 * s.n);
  }

  private static parseComp(sys: System, w: string, what: string): EComp {
    if (w.startsWith('v_')) {
      const name = w.slice(2);
      const v = sys.vars.get(name);
      if (!v) throw new StyleError(`fix efield: variable ${name} is not defined`);
      if (v.style === 'atom' || v.style === 'atomfile') return { var: name, atom: true };
      if (v.style === 'equal' || v.style === 'internal' || v.style === 'string') return { var: name, atom: false };
      throw new StyleError(`fix efield: a ${v.style}-style variable (${name}) cannot define an E-field component`);
    }
    const num = Number(w);
    if (!Number.isFinite(num)) throw new StyleError(`fix efield: ${what} must be a number or v_name (electric field units), got '${w}'`);
    return { const: num };
  }

  /** "The energy keyword specifies the name of an atom-style variable". */
  private static atomVar(sys: System, w: string, key: string): string {
    if (!w.startsWith('v_')) throw new StyleError(`fix efield ${key} value must be an atom-style variable (v_name), got '${w}'`);
    const name = w.slice(2);
    const v = sys.vars.get(name);
    if (!v) throw new StyleError(`fix efield: variable ${name} is not defined`);
    if (v.style !== 'atom') throw new StyleError(`fix efield ${key} variable ${name} must be atom-style, not ${v.style}-style`);
    return name;
  }

  /** Resolves one component now: a number, or per-atom values. */
  private resolve(c: EComp): number | Float64Array {
    if ('const' in c) return c.const;
    if (c.atom) return this.sys.atomVariable(c.var);
    return this.sys.equalVariable(c.var);
  }

  private compAt(r: number | Float64Array, i: number): number {
    return typeof r === 'number' ? r : r[i];
  }

  private inRegion(i: number): boolean {
    if (this.regionId === null) return true;
    const s = this.sys.state;
    return this.sys.region(this.regionId).match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
  }

  /** F = q E on each charged atom of the group (and region, if given). */
  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const [rx, ry, rz] = [this.resolve(this.comp[0]), this.resolve(this.comp[1]), this.resolve(this.comp[2])];
    const k = s.units.qe2f / s.units.ftm2v;
    const { f, q, mask, x } = s;
    const bit = this.groupBit;
    const fadd = this.fadd;
    fadd.fill(0);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      if (!this.inRegion(i)) continue;
      const fx = q[i] * this.compAt(rx, i) * k;
      const fy = q[i] * this.compAt(ry, i) * k;
      const fz = q[i] * this.compAt(rz, i) * k;
      fadd[3 * i] = fx;
      fadd[3 * i + 1] = fy;
      fadd[3 * i + 2] = fz;
      f[3 * i] += fx;
      f[3 * i + 1] += fy;
      f[3 * i + 2] += fz;
    }
    if (this.thermoVirial) this.tallyVirial();
  }

  /** Virial of the added forces, in the pressure compute's internal units. */
  private tallyVirial(): void {
    const s = this.sys.state;
    const v = this.virial;
    v.fill(0);
    const { x } = s;
    const fadd = this.fadd;
    for (let i = 0; i < s.n; i++) {
      const fx = fadd[3 * i], fy = fadd[3 * i + 1], fz = fadd[3 * i + 2];
      if (fx === 0 && fy === 0 && fz === 0) continue;
      const px = x[3 * i], py = x[3 * i + 1], pz = x[3 * i + 2];
      v[0] += fx * px;
      v[1] += fy * py;
      v[2] += fz * pz;
      v[3] += 0.5 * (fx * py + fy * px);
      v[4] += 0.5 * (fx * pz + fz * px);
      v[5] += 0.5 * (fy * pz + fz * py);
    }
  }

  /**
   * The fix's scalar energy: -q (x.E) for a constant vector, an atom-style
   * energy variable, an electric potential times charge, or 0.0 with
   * variables and no keyword.
   */
  private energyNow(): number {
    const s = this.sys.state;
    const { x, q, mask } = s;
    const bit = this.groupBit;
    if (this.energyVar !== null) {
      const ev = this.sys.atomVariable(this.energyVar);
      let e = 0;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit) || !this.inRegion(i)) continue;
        e += ev[i];
      }
      return e;
    }
    if (this.potentialVar !== null) {
      // "the electric potential multiplied by charge" (qe2f: potential*charge -> energy)
      const pv = this.sys.atomVariable(this.potentialVar);
      const k = s.units.qe2f;
      let e = 0;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit) || !this.inRegion(i)) continue;
        e += pv[i] * q[i] * k;
      }
      return e;
    }
    if (this.allConst) {
      const [ex, ey, ez] = [this.compAt(this.resolve(this.comp[0]), 0), this.compAt(this.resolve(this.comp[1]), 0), this.compAt(this.resolve(this.comp[2]), 0)];
      const k = s.units.qe2f;
      let e = 0;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit) || !this.inRegion(i)) continue;
        e -= q[i] * (x[3 * i] * ex + x[3 * i + 1] * ey + x[3 * i + 2] * ez) * k;
      }
      return e;
    }
    return 0;
  }

  energy(): number { return this.energyNow(); }
  computeScalar(): number { return this.energyNow(); }

  /** "The vector is the total force added to the group of atoms" (LAMMPS force units). */
  computeVector(i: number): number {
    const s = this.sys.state;
    const fadd = this.fadd;
    let sum = 0;
    for (let a = 0; a < s.n; a++) sum += fadd[3 * a + i];
    return sum * s.units.ftm2v;
  }

  minPostForce(): void { this.postForce(); }
}
