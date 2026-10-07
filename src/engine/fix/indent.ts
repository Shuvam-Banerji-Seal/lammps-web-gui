import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNumOrVar, valueOf, type NumOrVar } from './util';

/*
 * fix wall/reflect and fix indent — implemented only from the cited
 * docs.lammps.org pages (never from LAMMPS source code).
 *
 * The engine stores per-atom forces in internal units, F_internal =
 * F_lammps / ftm2v (see fix/force_ext.ts); energies are in LAMMPS energy
 * units, positions and velocities in LAMMPS units. Global fix outputs
 * (f_ID) are reported in LAMMPS units.
 */

/**
 * Lattice spacings for "units lattice". lattice.html: "By default, a
 * "lattice none 1.0" is defined, which means the lattice spacing is the same
 * as one distance unit", so with no lattice command the spacing is 1.
 */
const latticeSpacing = (sys: System): [number, number, number] =>
  sys.lattice ? ([...sys.lattice.spacing] as [number, number, number]) : [1, 1, 1];

/** A number or an equal-style v_name (validated to exist and be equal-style). */
const geoValue = (sys: System, w: string | undefined, what: string, style: string): NumOrVar => {
  const v = parseNumOrVar(w, what);
  if (typeof v === 'object') {
    const def = sys.vars.get(v.variable);
    if (!def) throw new StyleError(`fix ${style}: variable ${v.variable} does not exist`);
    if (def.style !== 'equal') {
      throw new StyleError(`fix ${style}: variable ${v.variable} must be an equal-style variable, not ${def.style}-style`);
    }
  }
  return v;
};

/*
 * fix ID group-ID wall/reflect face arg ... keyword value ... —
 * docs.lammps.org/fix_wall_reflect.html, Syntax:
 *
 *   fix ID group-ID wall/reflect face arg ... keyword value ...
 *
 * "* face = *xlo* or *xhi* or *ylo* or *yhi* or *zlo* or *zhi*", with
 * "arg = EDGE or constant or variable", "EDGE = current lo edge of simulation
 * box" and "constant = number like 0.0 or 30.0 (distance units)".
 * "Bound the simulation with one or more walls which reflect particles
 * in the specified group when they attempt to move through them."
 * "Reflection means that if an atom moves outside the wall on a timestep
 * by a distance delta (e.g. due to fix nve), then it is"
 * "put back inside the face by the same delta, and the sign of the"
 * "corresponding component of its velocity is flipped."
 * "A *lo* face reflects particles" / "that move to a coordinate less than the
 * wall position, back in the" / "*hi* direction.  A *hi* face reflects
 * particles that move to a coordinate higher than the wall position, back in
 * the *lo* direction." The reflection is applied in end_of_step, the last
 * hook of the timestep (docs.lammps.org/Developer_flow.html), so the checked
 * position is where the integrator left the atom after its move.
 *
 * "If the wall position is a variable, it should be specified as v_name,"
 * "where name is an :doc:`equal-style variable <variable>` name.  In this"
 * "case the variable is evaluated each timestep and the result becomes"
 * "the current position of the reflecting wall."
 *
 * "The *units* keyword determines the meaning of the distance units used
 * to define a wall position, but only when a numeric constant or
 * variable is used.  It is not relevant when EDGE is used to specify a
 * face position." with "*units* value = *lattice* or *box*",
 * "*lattice* = the wall position is defined in lattice units" and
 * "*box* = the wall position is defined in simulation box units". In the
 * variable case "the variable is assumed to produce a value compatible with
 * the *units* setting you specify", so its value is scaled by the lattice
 * spacing of the wall dimension for units lattice.
 *
 * "None of the :doc:`fix_modify <fix_modify>` options are"
 * "relevant to this fix.  No global or per-atom quantities are stored by"
 * "this fix for access by various :doc:`output commands <Howto_output>`." and
 * "This fix is not invoked during" energy minimization (no min hooks).
 * Restriction: "Any dimension (xyz) that has a reflecting wall must be
 * non-periodic." Default: "The default for the units keyword is lattice."
 */
interface ReflectFace {
  readonly dim: 0 | 1 | 2;
  readonly lo: boolean;
  readonly pos: { edge: true } | { edge: false; v: NumOrVar };
}

export class FixWallReflect extends Fix {
  readonly style = 'wall/reflect';
  private readonly faces: ReflectFace[] = [];
  private unitsMode: 'lattice' | 'box' = 'lattice';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) {
      throw new StyleError('usage: fix ID group-ID wall/reflect face arg ... [face arg ...] [keyword value ...]');
    }
    for (let k = 0; k < args.length;) {
      const w = args[k];
      if (w === 'units') {
        const v = args[k + 1];
        if (v !== 'lattice' && v !== 'box') {
          throw new StyleError(`fix ${id} (wall/reflect): units must be lattice or box, got '${v ?? ''}'`);
        }
        this.unitsMode = v;
        k += 2;
        continue;
      }
      const dim = 'xyz'.indexOf(w[0]) as 0 | 1 | 2;
      const isFace = dim >= 0 && w.length === 3 && (w[1] === 'l' || w[1] === 'h') && w[2] === (w[1] === 'l' ? 'o' : 'i');
      if (!isFace) {
        throw new StyleError(`fix ${id} (wall/reflect): unknown argument '${w}' (expected a face xlo, xhi, ylo, yhi, zlo or zhi, or the keyword units)`);
      }
      const lo = w[1] === 'l';
      if (this.faces.some((f) => f.dim === dim && f.lo === lo)) {
        throw new StyleError(`fix ${id} (wall/reflect): face ${w} is specified more than once`);
      }
      const arg = args[k + 1];
      if (arg === undefined) {
        throw new StyleError(`fix ${id} (wall/reflect): face ${w} needs an argument (EDGE, a constant, or v_name)`);
      }
      const pos = arg === 'EDGE'
        ? { edge: true as const }
        : arg.startsWith('v_')
          ? { edge: false as const, v: geoValue(sys, arg, `${w} wall position`, 'wall/reflect') }
          : { edge: false as const, v: parseNumOrVar(arg, `${w} wall position`) };
      this.faces.push({ dim, lo, pos });
      k += 2;
    }
    if (!this.faces.length) {
      throw new StyleError(`fix ${id} (wall/reflect): no wall face (xlo, xhi, ylo, yhi, zlo or zhi) was specified`);
    }
  }

  init(): void {
    const s = this.sys.state;
    for (const f of this.faces) {
      if (s.box.periodic[f.dim]) {
        throw new StyleError(`fix ${this.id} (wall/reflect): a reflecting wall requires a non-periodic dimension (docs.lammps.org/fix_wall_reflect.html: "Any dimension (xyz) that has a reflecting wall must be non-periodic.")`);
      }
    }
  }

  endOfStep(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const sp = this.unitsMode === 'lattice' ? latticeSpacing(sys) : null;
    const pos = new Float64Array(this.faces.length);
    for (let k = 0; k < this.faces.length; k++) {
      const f = this.faces[k];
      pos[k] = f.pos.edge
        ? (f.lo ? g.lo[f.dim] : g.hi[f.dim])
        : valueOf(sys, f.pos.v) * (sp ? sp[f.dim] : 1);
    }
    const { x, v, mask } = s;
    const bit = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      for (let k = 0; k < this.faces.length; k++) {
        const f = this.faces[k];
        const j = 3 * i + f.dim;
        if (f.lo ? x[j] < pos[k] : x[j] > pos[k]) {
          x[j] = 2 * pos[k] - x[j];
          v[j] = -v[j];
        }
      }
    }
  }
}

/*
 * fix ID group-ID indent K gstyle args keyword value ... —
 * docs.lammps.org/fix_indent.html, Syntax:
 *
 *   fix ID group-ID indent K gstyle args keyword value ...
 *
 * "* K = force constant for indenter surface (force/distance\^2 units)"
 * (a number; the page's variable example uses $k substitution, so K itself
 * is not a v_name here).
 *
 * "*sphere* args = x y z" with "x, y, z = position of center of indenter
 * (distance units)", "R = sphere radius of indenter (distance units)" and
 * "any of x, y, z, R can be a variable (see below)". It "exerts a force of
 * magnitude" "F(r) = - K \left( r - R \right)^2" "on each atom where *K* is
 * the specified force constant, *r* is the" "distance from the atom to the
 * center of the indenter, and *R* is the" "radius of the indenter.  The force
 * is repulsive and F(r) = 0 for *r* >" "*R*\." — the force on an atom points
 * away from the indenter surface ("The indenter repels all atoms in the group
 * that touch it"; with side out "then particles outside the indenter are
 * pushed away from its outer" surface), so an atom with r < R is pushed
 * radially outward with magnitude K (R - r)^2.
 *
 * "*cylinder* args = dim c1 c2 R" with "dim = *x* or *y* or *z* = axis of
 * cylinder", "c1, c2 = coords of cylinder axis in other 2 dimensions
 * (distance units)", "R = cylinder radius of indenter (distance units)".
 * "A cylindrical indenter (*gstyle* = *cylinder*) follows the same formula"
 * "for the force as a sphere, except that *r* is defined the distance"
 * "from the atom to the center axis of the cylinder.  The cylinder extends"
 * "infinitely along its axis." (c1, c2, R can be variables.)
 *
 * "*plane* args = dim pos side" with "dim = *x* or *y* or *z* = plane
 * perpendicular to this dimension", "pos = position of plane in dimension x,
 * y, or z (distance units)" (pos can be a variable) and "side = *lo* or
 * *hi*". "A planar indenter (*gstyle* = *plane*) behaves like an axis-aligned"
 * "infinite-extent wall with the same force expression on atoms in the"
 * system, with R the plane position and r-R the distance of an atom from the
 * plane. Side lo: "it will indent from the lo end of the simulation box,
 * meaning that atoms with a coordinate less than the" "plane's current
 * position will be pushed towards the hi end of the box" "and atoms with a
 * coordinate higher than the plane's current position" "will feel no force.
 * Vice versa if *side* is specified as *hi*".
 * "keyword = *side* or *units*" with "*side* value = *in* or *out*",
 * "*in* = the indenter acts on particles inside the sphere or cylinder or
 * cone" and "*out* = the indenter acts on particles outside the sphere or
 * cylinder or cone"; side in "reversed": "inside the indenter are pushed away
 * from its inner surface.  In other" "words, the indenter is now a containing
 * wall that traps the particles" "inside it.  If the radius shrinks over time,
 * it will squeeze the" particles. The *units* keyword: "*units* value =
 * *lattice* or *box*", "lattice = the geometry is defined in lattice units",
 * "box = the geometry is defined in simulation box units"; "The (x,y,z)
 * coords of the indenter position are scaled by the x,y,z" "lattice spacings
 * respectively.  The radius of a spherical or" "cylindrical indenter is
 * scaled by the x lattice spacing." and "the units keyword only affects
 * indenter geometry parameters" "specified directly with numbers, not those
 * specified as variables."
 *
 * Periodic boundaries: "Spherical, cylindrical, and conical indenters account
 * for periodic" "boundaries in two ways." "First, the center point of a
 * spherical" "indenter (x,y,z) or axis of a cylindrical/conical indenter
 * (c1,c2) is" "remapped back into the simulation box, if the box is periodic
 * in a" "particular dimension." "Second, the calculation of distance to the
 * indenter center" "or axis accounts for periodic boundaries."
 *
 * Energy and output: "The energy of" "each particle interacting with the
 * indenter is K/3 (r - R)\^3." — for side out the interacting atom sits at
 * r < R, so the interaction energy per atom is K/3 (R - r)^3 (positive), the
 * value consistent with the repulsive force and with the plane (where r-R is
 * the atom's distance from the plane, positive on the indented side); side in
 * reverses the action, so the energy per atom is -K/3 (R - r)^3.
 * "The :doc:`fix_modify <fix_modify>` *energy* option is supported by"
 * this fix (default "fix_modify energy no"). "This fix computes a global
 * scalar energy and a global 3-vector of" "forces (on the indenter), which
 * can be accessed by various" output commands. "The scalar and vector values"
 * "calculated by this fix are "extensive"." The force on the indenter is
 * the reaction to the forces applied to the atoms (Newton's third law).
 * "The forces due to this fix are imposed during an energy minimization,"
 * invoked by the minimize command. Default: "The option defaults are side =
 * out and units = lattice."
 *
 * The cone gstyle ("*cone* args = dim c1 c2 radlo radhi lo hi") is documented
 * on the page but is not supported by this engine (StyleError).
 */
type IndentGeom =
  | { kind: 'sphere'; c: [NumOrVar, NumOrVar, NumOrVar]; r: NumOrVar }
  | { kind: 'cylinder'; axis: 0 | 1 | 2; c1: NumOrVar; c2: NumOrVar; r: NumOrVar }
  | { kind: 'plane'; dim: 0 | 1 | 2; pos: NumOrVar; lo: boolean };

export class FixIndent extends Fix {
  readonly style = 'indent';
  private readonly k: number;
  private readonly geom: IndentGeom;
  private sideOut = true;
  private unitsMode: 'lattice' | 'box' = 'lattice';
  /** Sum of the per-atom interaction energies (f_ID scalar, energy units). */
  private energyAccum = 0;
  /** Force on the indenter, i.e. minus the sum of the forces on the atoms (LAMMPS force units). */
  private readonly fInd = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) {
      throw new StyleError('usage: fix ID group-ID indent K gstyle args [keyword value ...]');
    }
    const kw = args[0];
    if (kw.startsWith('v_') || !Number.isFinite(Number(kw))) {
      throw new StyleError(`fix ${id} (indent): K must be a number (force/distance^2 units), got '${kw}'`);
    }
    this.k = Number(kw);
    const gstyle = args[1];
    let k: number;
    if (gstyle === 'sphere') {
      if (args.length < 6) throw new StyleError(`fix ${id} (indent): sphere needs 4 args (x y z R)`);
      this.geom = {
        kind: 'sphere',
        c: [
          geoValue(sys, args[2], 'sphere x', 'indent'),
          geoValue(sys, args[3], 'sphere y', 'indent'),
          geoValue(sys, args[4], 'sphere z', 'indent'),
        ],
        r: geoValue(sys, args[5], 'sphere R', 'indent'),
      };
      k = 6;
    } else if (gstyle === 'cylinder') {
      if (args.length < 6) throw new StyleError(`fix ${id} (indent): cylinder needs 4 args (dim c1 c2 R)`);
      const axis = 'xyz'.indexOf(args[2]);
      if (axis < 0 || args[2].length !== 1) {
        throw new StyleError(`fix ${id} (indent): cylinder dim must be x, y or z, got '${args[2]}'`);
      }
      this.geom = {
        kind: 'cylinder',
        axis: axis as 0 | 1 | 2,
        c1: geoValue(sys, args[3], 'cylinder c1', 'indent'),
        c2: geoValue(sys, args[4], 'cylinder c2', 'indent'),
        r: geoValue(sys, args[5], 'cylinder R', 'indent'),
      };
      k = 6;
    } else if (gstyle === 'plane') {
      if (args.length < 5) throw new StyleError(`fix ${id} (indent): plane needs 3 args (dim pos side)`);
      const dim = 'xyz'.indexOf(args[2]);
      if (dim < 0 || args[2].length !== 1) {
        throw new StyleError(`fix ${id} (indent): plane dim must be x, y or z, got '${args[2]}'`);
      }
      if (args[4] !== 'lo' && args[4] !== 'hi') {
        throw new StyleError(`fix ${id} (indent): plane side must be lo or hi, got '${args[4]}'`);
      }
      this.geom = { kind: 'plane', dim: dim as 0 | 1 | 2, pos: geoValue(sys, args[3], 'plane pos', 'indent'), lo: args[4] === 'lo' };
      k = 5;
    } else if (gstyle === 'cone') {
      throw new StyleError(`fix ${id} (indent): gstyle cone is documented on docs.lammps.org/fix_indent.html but is not supported by this browser engine`);
    } else {
      throw new StyleError(`fix ${id} (indent): gstyle must be sphere, cylinder or plane, got '${gstyle ?? ''}'`);
    }
    for (; k < args.length;) {
      const key = args[k];
      if (key === 'side') {
        if (this.geom.kind === 'plane') {
          throw new StyleError(`fix ${id} (indent): the side keyword (in/out) applies to sphere and cylinder indenters; the plane's side is its lo/hi argument`);
        }
        const v = args[k + 1];
        if (v !== 'in' && v !== 'out') {
          throw new StyleError(`fix ${id} (indent): side must be in or out, got '${v ?? ''}'`);
        }
        this.sideOut = v === 'out';
        k += 2;
      } else if (key === 'units') {
        const v = args[k + 1];
        if (v !== 'lattice' && v !== 'box') {
          throw new StyleError(`fix ${id} (indent): units must be lattice or box, got '${v ?? ''}'`);
        }
        this.unitsMode = v;
        k += 2;
      } else {
        throw new StyleError(`fix ${id} (indent): unknown keyword '${key}' (side and units are documented)`);
      }
    }
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
  }

  /**
   * Evaluates a geometry value now. Numeric constants are scaled by the
   * lattice spacing of dimension d for units lattice; variables are used
   * as-is ("the units keyword only affects indenter geometry parameters"
   * "specified directly with numbers, not those specified as variables").
   */
  private evalGeom(v: NumOrVar, d: number): number {
    const val = valueOf(this.sys, v);
    if (this.unitsMode !== 'lattice' || typeof v !== 'number') return val;
    return val * latticeSpacing(this.sys)[d];
  }

  /**
   * Remaps a center/axis coordinate into the box along a periodic dimension
   * ("remapped back into the simulation box, if the box is periodic in a"
   * "particular dimension").
   */
  private remapCoord(c: number, d: number): number {
    const g = this.sys.geom;
    if (!g.periodic[d]) return c;
    const len = d === 0 ? g.lx : d === 1 ? g.ly : g.lz;
    return c - Math.floor((c - g.lo[d]) / len) * len;
  }

  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const bit = this.groupBit;
    const k = this.k;
    const ftm2v = s.units.ftm2v;
    const { x, f, mask } = s;
    const fInd = this.fInd;
    fInd[0] = 0; fInd[1] = 0; fInd[2] = 0;
    let energy = 0;
    const gm = this.geom;
    if (gm.kind === 'plane') {
      const d = gm.dim;
      const pos = this.evalGeom(gm.pos, d);
      const sign = gm.lo ? 1 : -1;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit)) continue;
        const depth = sign > 0 ? pos - x[3 * i + d] : x[3 * i + d] - pos;
        if (depth <= 0) continue;
        const mag = k * depth * depth;
        energy += (k / 3) * depth * depth * depth;
        f[3 * i + d] += (sign * mag) / ftm2v;
        fInd[d] -= sign * mag;
      }
    } else if (gm.kind === 'sphere') {
      const c = [0, 0, 0];
      for (let d = 0; d < 3; d++) c[d] = this.remapCoord(this.evalGeom(gm.c[d], d), d);
      const r0 = this.evalGeom(gm.r, 0);
      const sgn = this.sideOut ? 1 : -1;
      const delta = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit)) continue;
        delta[0] = x[3 * i] - c[0];
        delta[1] = x[3 * i + 1] - c[1];
        delta[2] = x[3 * i + 2] - c[2];
        g.minimumImage(delta);
        const r = Math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
        if (r >= r0) continue;
        const depth = r0 - r;
        const mag = k * depth * depth;
        energy += (this.sideOut ? 1 : -1) * (k / 3) * depth * depth * depth;
        if (r > 0) {
          const ux = delta[0] / r, uy = delta[1] / r, uz = delta[2] / r;
          f[3 * i] += (sgn * mag * ux) / ftm2v;
          f[3 * i + 1] += (sgn * mag * uy) / ftm2v;
          f[3 * i + 2] += (sgn * mag * uz) / ftm2v;
          fInd[0] -= sgn * mag * ux;
          fInd[1] -= sgn * mag * uy;
          fInd[2] -= sgn * mag * uz;
        }
      }
    } else {
      const axis = gm.axis;
      const dims = (axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1]) as [number, number];
      const c = [0, 0, 0];
      c[dims[0]] = this.remapCoord(this.evalGeom(gm.c1, dims[0]), dims[0]);
      c[dims[1]] = this.remapCoord(this.evalGeom(gm.c2, dims[1]), dims[1]);
      const r0 = this.evalGeom(gm.r, 0);
      const sgn = this.sideOut ? 1 : -1;
      const delta = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit)) continue;
        delta[0] = 0; delta[1] = 0; delta[2] = 0;
        delta[dims[0]] = x[3 * i + dims[0]] - c[dims[0]];
        delta[dims[1]] = x[3 * i + dims[1]] - c[dims[1]];
        g.minimumImage(delta);
        const a = delta[dims[0]], b = delta[dims[1]];
        const r = Math.sqrt(a * a + b * b);
        if (r >= r0) continue;
        const depth = r0 - r;
        const mag = k * depth * depth;
        energy += (this.sideOut ? 1 : -1) * (k / 3) * depth * depth * depth;
        if (r > 0) {
          const ua = a / r, ub = b / r;
          f[3 * i + dims[0]] += (sgn * mag * ua) / ftm2v;
          f[3 * i + dims[1]] += (sgn * mag * ub) / ftm2v;
          fInd[dims[0]] -= sgn * mag * ua;
          fInd[dims[1]] -= sgn * mag * ub;
        }
      }
    }
    this.energyAccum = energy;
  }

  /** "The forces due to this fix are imposed during an energy minimization". */
  minPostForce(): void { this.postForce(); }

  energy(): number { return this.energyAccum; }
  computeScalar(): number { return this.energyAccum; }
  computeVector(i: number): number { return this.fInd[i]; }
}
