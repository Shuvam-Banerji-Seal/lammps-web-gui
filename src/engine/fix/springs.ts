import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';
import { parseNumOrVar, valueOf, type NumOrVar } from './util';

/*
 * fix spring and fix spring/self — force-adding fixes written from their doc
 * pages, not from LAMMPS code.
 *
 * fix spring — docs.lammps.org/fix_spring.html, syntax:
 *   "fix ID group-ID spring keyword values"
 *     "*tether* values = K x y z R0"
 *       "K = spring constant (force/distance units)"
 *       "x,y,z = point to which spring is tethered"
 *       "R0 = equilibrium distance from tether point (distance units)"
 *     "*couple* values = group-ID2 K x y z R0"
 *       "group-ID2 = 2nd group to couple to fix group with a spring"
 *       "K = spring constant (force/distance units)"
 *       "x,y,z = direction of spring"
 *       "R0 = equilibrium distance of spring (distance units)"
 *
 * Tether: "The *tether* style attaches a spring between a fixed point *x,y,z*
 * and the center of mass of the fix group of atoms.  The equilibrium
 * position of the spring is R0.  At each timestep the distance R from
 * the center of mass of the group of atoms to the tethering point is
 * computed, taking account of wrap-around in a periodic simulation box."
 * "A restoring force of magnitude K (R - R0) Mi / M is applied to each
 * atom in the group where *K* is the spring constant, Mi is the mass of
 * the atom, and M is the total mass of all atoms in the group."
 *
 * Couple: "The *couple* style links two groups of atoms together.  The first
 * group is the fix group; the second is specified by group-ID2.  The
 * groups are coupled together by a spring that is at equilibrium when
 * the two groups are displaced by a vector *x,y,z* with respect to each
 * other and at a distance R0 from that displacement.  Note that *x,y,z*
 * is the equilibrium displacement of group-ID2 relative to the fix
 * group." "When the relative positions and distance between the two
 * groups are not in equilibrium, the same spring force described above
 * is applied to atoms in each of the two groups" (equal and opposite on
 * the two groups, each distributed over its own group by Mi/M).
 *
 * "For both the *tether* and *couple* styles, any of the x,y,z values can
 * be specified as NULL which means do not include that dimension in the
 * distance calculation or force application."
 *
 * "The center of mass of a group of atoms is calculated in
 * "unwrapped" coordinates using atom image flags, which means that the
 * group can straddle a periodic boundary."  The remaining wrap-around
 * ("a spring connecting two groups or a group and the tether point can
 * cross a periodic boundary and its length be calculated correctly") is
 * the minimum-image of the displacement vector.
 *
 * Output: "This fix computes a global scalar which can be accessed by various
 * :doc:`output commands <Howto_output>`.  The scalar is the spring energy
 * = 0.5 \* K \* r\^2." "This fix also computes global 4-vector which can be
 * accessed by various :doc:`output commands <Howto_output>`.  The first 3
 * quantities in the vector are xyz components of the total force added to
 * the group of atoms by the spring.  In the case of the *couple* style, it
 * is the force on the fix group (group-ID) or the negative of the force on
 * the second group (group-ID2).  The fourth quantity in the vector is the
 * magnitude of the force added by the spring, as a positive value if
 * (r-R0) > 0 and a negative value if (r-R0) < 0."
 * "The scalar and vector values calculated by this fix are "extensive"."
 *
 * "The :doc:`fix_modify <fix_modify>` *energy* option is supported by
 * this fix to add the energy stored in the spring to the global
 * potential energy of the system as part of :doc:`thermodynamic output
 * <thermo_style>`. The default setting for this fix is :doc:`fix_modify
 * energy no <fix_modify>`." (fix_modify.html: "Energy yes will add a
 * contribution to the potential energy of the system"; "For most fixes
 * that support the energy keyword, the default setting is no.")
 *
 * "The forces due to this fix are imposed during an energy minimization,
 * invoked by the :doc:`minimize <minimize>` command." Default: "none".
 */

const num = (w: string | undefined, what: string): number => {
  if (w === undefined) throw new StyleError(`missing ${what}`);
  const v = Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`expected a number for ${what}, got '${w}'`);
  return v;
};

/** 'x y z' words: a number or NULL = exclude that dimension. */
const parseXYZ = (w: string | undefined, what: string): { value: number; on: boolean } => {
  if (w === undefined) throw new StyleError(`missing ${what}`);
  if (w === 'NULL') return { value: 0, on: false };
  const v = Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`expected a number or NULL for ${what}, got '${w}'`);
  return { value: v, on: true };
};

export class FixSpring extends Fix {
  readonly style = 'spring';
  private couple: boolean;
  private group2Bit = 0;
  private k: number;
  private r0: number;
  /** Which dimensions the spring acts in (the non-NULL x,y,z values). */
  private inc: [boolean, boolean, boolean];
  /** Tether point (tether) or equilibrium displacement of group2 (couple). */
  private point: [number, number, number];
  private espring = 0;
  private fvec = new Float64Array(4);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 4;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
    const mode = args[0];
    if (mode === 'tether') {
      if (args.length !== 6) throw new StyleError('usage: fix ID group spring tether K x y z R0');
      this.couple = false;
      this.k = num(args[1], 'K');
    } else if (mode === 'couple') {
      if (args.length !== 7) throw new StyleError('usage: fix ID group spring couple group-ID2 K x y z R0');
      this.couple = true;
      this.group2Bit = sys.groups.bit(args[1]);
      this.k = num(args[2], 'K');
    } else {
      throw new StyleError(`fix spring keyword must be tether or couple, got '${mode ?? ''}'`);
    }
    const xyz = [parseXYZ(args[this.couple ? 3 : 2], 'x'), parseXYZ(args[this.couple ? 4 : 3], 'y'), parseXYZ(args[this.couple ? 5 : 4], 'z')];
    this.inc = [xyz[0].on, xyz[1].on, xyz[2].on];
    this.point = [xyz[0].value, xyz[1].value, xyz[2].value];
    this.r0 = num(args[this.couple ? 6 : 5], 'R0');
  }

  /** Mass-weighted center of mass in unwrapped coordinates, or null if empty. */
  private centerOfMass(bit: number): { com: [number, number, number]; mass: number } | null {
    const s = this.sys.state;
    let m = 0;
    const c: [number, number, number] = [0, 0, 0];
    const p = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & bit)) continue;
      const mi = massOf(s, i);
      m += mi;
      this.sys.geom.unwrap(s.x, s.image, i, p);
      c[0] += mi * p[0]; c[1] += mi * p[1]; c[2] += mi * p[2];
    }
    if (m === 0) return null;
    return { com: [c[0] / m, c[1] / m, c[2] / m], mass: m };
  }

  /** Adds total group force (fx, fy, fz) distributed over the group as K (R - R0) Mi / M. */
  private addGroupForce(bit: number, mass: number, fx: number, fy: number, fz: number): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & bit)) continue;
      const w = massOf(s, i) / mass;
      s.f[3 * i] += fx * w; s.f[3 * i + 1] += fy * w; s.f[3 * i + 2] += fz * w;
    }
  }

  postForce(): void {
    const sys = this.sys;
    const g1 = this.centerOfMass(this.groupBit);
    const g2 = this.couple ? this.centerOfMass(this.group2Bit) : g1;
    if (!g1 || !g2) { this.espring = 0; this.fvec.fill(0); return; }
    // displacement from equilibrium; a NULL component stays out of r and the force
    const delta = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      delta[d] = this.inc[d] ? (this.couple ? g2.com[d] - g1.com[d] : g1.com[d]) - this.point[d] : 0;
    }
    sys.geom.minimumImage(delta);
    for (let d = 0; d < 3; d++) if (!this.inc[d]) delta[d] = 0;
    const r = Math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    const dr = r - this.r0;
    this.espring = 0.5 * this.k * dr * dr;
    const mag = this.k * dr;
    // restoring force: tether pulls the group back along -unit; couple pulls the
    // two groups together (+mag on the fix group, -mag on group-ID2)
    const c1 = this.couple ? mag : -mag;
    if (r > 0) {
      const ux = delta[0] / r, uy = delta[1] / r, uz = delta[2] / r;
      this.addGroupForce(this.groupBit, g1.mass, c1 * ux, c1 * uy, c1 * uz);
      if (this.couple) this.addGroupForce(this.group2Bit, g2.mass, -mag * ux, -mag * uy, -mag * uz);
      this.fvec[0] = c1 * ux; this.fvec[1] = c1 * uy; this.fvec[2] = c1 * uz;
    } else {
      this.fvec[0] = 0; this.fvec[1] = 0; this.fvec[2] = 0;
    }
    this.fvec[3] = mag;
  }

  minPostForce(): void { this.postForce(); }

  energy(): number { return this.espring; }
  computeScalar(): number { return this.espring; }
  computeVector(i: number): number { return this.fvec[i]; }
}

/*
 * fix spring/self — docs.lammps.org/fix_spring_self.html, syntax:
 *   "fix ID group-ID spring/self K dir"
 *     "K = spring constant (force/distance units), can be a variable (see below)"
 *     "dir = xyz, xy, xz, yz, x, y, or z (optional, default: xyz)"
 *
 * "Apply a spring force independently to each atom in the group to tether
 * it to its initial position.  The initial position for each atom is its
 * location at the time the fix command was issued.  At each timestep,
 * the magnitude of the force on each atom is -Kr, where r is the
 * displacement of the atom from its current position to its initial
 * position.  The distance r correctly takes into account any crossings
 * of periodic boundary by the atom since it was in its initial
 * position." (initial unwrapped positions; the displacement of unwrapped
 * coordinates includes any boundary crossings.)
 *
 * "With the (optional) dir flag, one can select in which direction the
 * spring force is applied. By default, the restraint is applied in all
 * directions, but it can be limited to the xy-, xz-, yz-plane and the
 * x-, y-, or z-direction, thus restraining the atoms to a line or a
 * plane, respectively."
 *
 * "The force constant *k* can be specified as an equal-style or atom-style
 * :doc:`variable <variable>`.  If the value is a variable, it should be specified
 * as v_name, where name is the variable name.  In this case, the variable
 * will be evaluated each time step, and its value(s) will be used as
 * force constant for the spring force."
 *
 * Output: "This fix computes a global scalar which can be accessed by various
 * :doc:`output commands <Howto_output>`.  The scalar is an energy which is
 * the sum of the spring energy for each atom, where the per-atom energy
 * is 0.5 \* K \* r\^2.  The scalar value calculated by this fix is
 * "extensive"."
 *
 * "The :doc:`fix_modify <fix_modify>` *energy* option is supported by
 * this fix to add the energy stored in the per-atom springs to the
 * global potential energy of the system as part of :doc:`thermodynamic
 * output <thermo_style>`.  The default setting for this fix is
 * :doc:`fix_modify energy no <fix_modify>`."
 *
 * "The forces due to this fix are imposed during an energy minimization,
 * invoked by the :doc:`minimize <minimize>` command."
 */

export class FixSpringSelf extends Fix {
  readonly style = 'spring/self';
  private k: NumOrVar;
  private kIsAtom = false;
  private dims: [boolean, boolean, boolean];
  /** Unwrapped positions at the time the fix command was issued. */
  private x0: Float64Array;
  private espring = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length > 2) throw new StyleError('usage: fix ID group spring/self K [dir]');
    this.k = parseNumOrVar(args[0], 'K');
    if (typeof this.k !== 'number') {
      const v = sys.vars.get(this.k.variable);
      if (!v) throw new StyleError(`fix spring/self: variable ${this.k.variable} is not defined`);
      if (v.style === 'atom') this.kIsAtom = true;
      else if (v.style !== 'equal') {
        throw new StyleError(`fix spring/self: variable ${this.k.variable} must be equal-style or atom-style, not ${v.style}`);
      }
    }
    const dir = args[1] ?? 'xyz';
    const table: Record<string, [boolean, boolean, boolean]> = {
      xyz: [true, true, true], xy: [true, true, false], xz: [true, false, true], yz: [false, true, true],
      x: [true, false, false], y: [false, true, false], z: [false, false, true],
    };
    const d = table[dir];
    if (!d) throw new StyleError(`fix spring/self dir must be xyz, xy, xz, yz, x, y or z, got '${dir}'`);
    this.dims = d;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.energyGlobal = true;
    const s = sys.state;
    this.x0 = new Float64Array(3 * s.n);
    const p = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      sys.geom.unwrap(s.x, s.image, i, p);
      this.x0[3 * i] = p[0]; this.x0[3 * i + 1] = p[1]; this.x0[3 * i + 2] = p[2];
    }
  }

  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    if (this.x0.length < 3 * s.n) {
      // atoms added after the fix: their initial position is where they are first seen
      const old = this.x0;
      this.x0 = new Float64Array(3 * s.n);
      this.x0.set(old);
      const p = [0, 0, 0];
      for (let i = old.length / 3; i < s.n; i++) {
        sys.geom.unwrap(s.x, s.image, i, p);
        this.x0[3 * i] = p[0]; this.x0[3 * i + 1] = p[1]; this.x0[3 * i + 2] = p[2];
      }
    }
    const kv = this.k;
    const kAtom = this.kIsAtom && typeof kv !== 'number' ? sys.atomVariable(kv.variable) : null;
    const k = kAtom ? 0 : valueOf(sys, this.k);
    const { f, mask } = s;
    const g = sys.geom;
    const p = [0, 0, 0];
    let e = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      g.unwrap(s.x, s.image, i, p);
      const dx = this.dims[0] ? p[0] - this.x0[3 * i] : 0;
      const dy = this.dims[1] ? p[1] - this.x0[3 * i + 1] : 0;
      const dz = this.dims[2] ? p[2] - this.x0[3 * i + 2] : 0;
      const ki = kAtom ? kAtom[i] : k;
      f[3 * i] -= ki * dx; f[3 * i + 1] -= ki * dy; f[3 * i + 2] -= ki * dz;
      e += 0.5 * ki * (dx * dx + dy * dy + dz * dz);
    }
    this.espring = e;
  }

  minPostForce(): void { this.postForce(); }

  energy(): number { return this.espring; }
  computeScalar(): number { return this.espring; }
}
