import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { missingArgs, inumeric, bnumeric, numeric } from '../force/util';
import { errorUrl } from '../formula';
import { argInfo, EQUAL_STYLES } from '../argInfo';
import { BlockRegion, ConeRegion, SphereRegion, type Region, type SurfaceContact } from '../region';
import { massOf, nativeOrder } from '../atoms';
import { RanMars } from '../rng';
import { libm } from '../libm';
import { gplHook } from '../gplHooks';

/*
 * Fixes of the misc3 wave: fix vector, fix wall/region, fix wall/reflect/stochastic.
 * Implemented from docs.lammps.org pages; fix vector follows native LAMMPS as noted at it.
 */

/*
 * fix vector — docs.lammps.org/fix_vector.html, Syntax:
 *
 *   fix ID group-ID vector Nevery value1 value2 ... keyword args ...
 *
 * "Use one or more global values as inputs every few timesteps, and simply
 * store them as a sequence.  For a single specified value, the values are
 * stored as a global vector of growing length.  For multiple specified
 * values, they are stored as rows in a global array, whose number of rows
 * is growing."
 * "The group specified with this command is ignored."
 * "*Nevery* = use input values every this many timesteps" and "Only timesteps
 * that are a multiple of *Nevery*, including timestep 0, will contribute
 * values."
 * "The optional *nmax* keyword can be used to restrict the length of the
 * vector to the given *length* value.  Once the restricted vector is filled,
 * the oldest entry will be discarded when a entry is added."
 * "If a value begins with "c\_", a compute ID must follow ... If no bracketed
 * term is appended, the global scalar calculated by the compute is used. If a
 * bracketed term is appended, the Ith element of the global vector calculated
 * by the compute is used."
 * "The values can only be accessed on timesteps that are multiples of
 * *Nevery*."
 * "This fix is not invoked during energy minimization."
 *
 * As native LAMMPS does (src/fix_vector.cpp at b332f28): the values are read as
 * ArgInfo references up to the first word that is none, then only the nmax
 * keyword; the first sample is the first multiple of Nevery at or after the
 * step the fix was defined on, and later samples follow every Nevery steps from
 * it (a run that does not reach that step, such as a minimization, leaves the
 * fix waiting for it). Each run's init sizes the storage for the samples up to
 * the run's end step (its stop value), at most nmax; a run with pre no keeps the
 * old size, and once the storage is full the oldest entry is overwritten. The
 * fix itself does not check the step a value is read on: variables do it during
 * a run (Variable::evaluate).
 */

/** One input value: a compute, fix or variable and its index (0: the scalar, I: element I of the vector). */
type VectorInput = { which: 'c' | 'f' | 'v'; id: string; argindex: number };

/** The largest nmax (native MAXSMALLINT). */
const MAX_SMALL_INT = 2147483647;

export class FixVector extends Fix {
  readonly style = 'vector';
  private readonly values: VectorInput[] = [];
  private readonly nmaxval: number;
  /** The stored rows (one value per input), ncountmax of them; filled in order, then overwritten from the oldest. */
  private store: number[][] = [];
  private allocated = false;
  private ncount = 0;
  private ncountmax = 1;
  /** The next step that stores a row, and the first one. */
  private nextstep: number;
  private readonly initialstep: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw missingArgs('fix vector');
    this.nevery = inumeric(args[0]);
    if (this.nevery <= 0) throw new StyleError(`Invalid fix vector every argument: ${this.nevery}`);
    let nmaxval = MAX_SMALL_INT;
    let k = 1;
    for (; k < args.length; k++) {
      const a = argInfo(args[k]);
      if (a.which === 'none') break;
      if (a.which === 'unknown' || a.dim > 1) throw new StyleError(`Invalid fix vector argument: ${args[k]}`);
      // an index below 1 in brackets: native reads the scalar for [0] and outside the vector for a negative index
      if (a.index1 < 0) throw new StyleError(`fix vector: index ${a.index1} in ${args[k]} is refused: native LAMMPS reads outside the vector for it`);
      this.values.push({ which: a.which as 'c' | 'f' | 'v', id: a.name, argindex: a.index1 });
    }
    for (; k < args.length; k += 2) {
      if (args[k] !== 'nmax') throw new StyleError(`Unknown fix vector keyword: ${args[k]}`);
      if (k + 2 > args.length) throw missingArgs('fix vector nmax');
      nmaxval = bnumeric(args[k + 1]);
      if (nmaxval < 1 || nmaxval > MAX_SMALL_INT) throw new StyleError('Invalid nmax value');
    }
    this.nmaxval = nmaxval;
    // the inputs' sources and their intensive/extensive flags, which must agree
    let ext: number | null = null;
    for (const val of this.values) {
      let value = 0;
      if (val.which === 'c') {
        const c = sys.computes.find((x) => x.id === val.id);
        if (!c) throw new StyleError(`Compute ID ${val.id} for fix vector does not exist`);
        if (val.argindex === 0 && !c.scalarFlag) throw new StyleError(`Fix vector compute ${val.id} does not calculate a scalar`);
        if (val.argindex && !c.vectorFlag) throw new StyleError(`Fix vector compute ${val.id} does not calculate a vector`);
        // an engine compute that sizes its vector at init reports 0 here
        if (val.argindex && c.sizeVector > 0 && val.argindex > c.sizeVector) {
          throw new StyleError(`Fix vector compute ${val.id} vector is accessed out-of-range${errorUrl(20)}`);
        }
        value = val.argindex === 0 ? c.extscalar : c.extlist ? c.extlist[val.argindex - 1] ?? 0 : c.extvector;
      } else if (val.which === 'f') {
        const f = sys.fixes.find((x) => x.id === val.id);
        if (!f) throw new StyleError(`Fix ID ${val.id} for fix vector does not exist`);
        if (val.argindex === 0 && !f.scalarFlag) throw new StyleError(`Fix vector fix ${val.id} does not calculate a scalar`);
        if (val.argindex && !f.vectorFlag) throw new StyleError(`Fix vector fix ${val.id} does not calculate a vector`);
        if (val.argindex && (f.sizeVector > 0 || f.sizeExact) && val.argindex > f.sizeVector) {
          throw new StyleError(`Fix vector fix ${val.id} vector is accessed out-of-range${errorUrl(20)}`);
        }
        if (f.globalFreq > 0 && this.nevery % f.globalFreq) {
          throw new StyleError(`Fix for fix ${val.id} vector not computed at compatible time${errorUrl(7)}`);
        }
        value = val.argindex === 0 ? f.extscalar : f.extlist ? f.extlist[val.argindex - 1] ?? 0 : f.extvector;
      } else {
        const v = sys.vars.get(val.id);
        if (!v) throw new StyleError(`Variable name ${val.id} for fix vector does not exist`);
        if (val.argindex === 0 && !EQUAL_STYLES.has(v.style)) throw new StyleError(`Fix vector variable ${val.id} is not equal-style variable`);
        if (val.argindex && v.style !== 'vector') throw new StyleError(`Fix vector variable ${val.id} is not vector-style variable`);
      }
      if (ext === null) ext = value;
      else if (value !== ext) throw new StyleError('Fix vector cannot set output array intensive/extensive from these inputs');
    }
    if (this.values.length === 1) {
      this.vectorFlag = true;
      this.extvector = ext!;
    } else {
      // Measured with native LAMMPS: without input values (fix vector 2 nmax 5) the fix is refused with the message of
      // an output whose intensive/extensive flag is unset; native leaves the flag unassigned then.
      if (!this.values.length) {
        throw new StyleError(`Must set 'extarray' when setting 'array_flag' for fix vector.  Please contact the LAMMPS developers.${errorUrl(35)}`);
      }
      this.arrayFlag = true;
      this.sizeArrayCols = this.values.length;
      this.extarray = ext!;
    }
    this.globalFreq = this.nevery;
    this.sizeExact = true;
    const step = sys.state.step;
    this.nextstep = Math.floor(step / this.nevery) * this.nevery;
    if (this.nextstep < step) this.nextstep += this.nevery;
    this.initialstep = this.nextstep;
  }

  /** The inputs must still exist; the storage grows (or shrinks) to the samples up to the run's end step, at most nmax. */
  init(): void {
    for (const val of this.values) {
      if (val.which === 'c' && !this.sys.computes.some((x) => x.id === val.id)) throw new StyleError(`Compute ID ${val.id} for fix vector does not exist`);
      if (val.which === 'f' && !this.sys.fixes.some((x) => x.id === val.id)) throw new StyleError(`Fix ID ${val.id} for fix vector does not exist`);
      if (val.which === 'v' && !this.sys.vars.has(val.id)) throw new StyleError('Variable name for fix vector does not exist');
    }
    const end = this.sys.run.endStep;
    let finalstep = Math.floor(end / this.nevery) * this.nevery;
    if (finalstep > end) finalstep -= this.nevery;
    const oldmax = this.allocated ? this.ncountmax : 0;
    let ncountmax = Math.floor((finalstep - this.initialstep) / this.nevery) + 1;
    if (ncountmax <= 0) ncountmax = 1;
    if (ncountmax > this.nmaxval) ncountmax = this.nmaxval;
    this.ncountmax = ncountmax;
    // rows stored in earlier runs are kept; new rows start at zero
    if (ncountmax < this.store.length) this.store.length = ncountmax;
    for (let i = Math.max(oldmax, this.store.length); i < ncountmax; i++) this.store.push(new Array<number>(this.values.length).fill(0));
    this.allocated = true;
  }

  setup(): void { this.endOfStep(); }

  endOfStep(): void {
    if (this.sys.state.step !== this.nextstep) return;
    const row = this.store[this.ncount % this.ncountmax];
    this.values.forEach((val, i) => { row[i] = this.input(val); });
    this.nextstep += this.nevery;
    this.ncount++;
    if (this.values.length === 1) this.sizeVector = Math.min(this.sizeVector + 1, this.ncountmax);
    else this.sizeArrayRows = Math.min(this.sizeArrayRows + 1, this.ncountmax);
  }

  /** One input's current value; an element past the end of a vector-style variable is 0. */
  private input(val: VectorInput): number {
    if (val.which === 'c') {
      const c = this.sys.compute(val.id);
      return val.argindex === 0 ? c.scalarValue() : c.vectorValues()[val.argindex - 1];
    }
    if (val.which === 'f') {
      const f = this.sys.fix(val.id);
      return val.argindex === 0 ? f.computeScalar() : f.computeVector(val.argindex - 1);
    }
    if (val.argindex === 0) return this.sys.equalVariable(val.id);
    const vec = this.sys.vars.evalVector(val.id, this.sys.formulaEnv);
    return vec.length < val.argindex ? 0 : vec[val.argindex - 1];
  }

  /** Row I in storage order: once the storage is full, row 0 is the oldest kept entry. */
  private row(i: number): number[] {
    return this.store[this.ncount >= this.ncountmax ? (i + this.ncount) % this.ncountmax : i];
  }

  computeVector(i: number): number { return this.row(i)[0]; }

  computeArray(i: number, j: number): number { return this.row(i)[j]; }
}

/*
 * fix wall/region — docs.lammps.org/fix_wall_region.html, Syntax:
 *
 *   fix ID group-ID wall/region region-ID style args ... cutoff
 *
 * "style = *lj93* or *lj126* or *lj1043* or *colloid* or *harmonic* or *morse*";
 * the args are "epsilon = strength factor ..." and "sigma = size factor ..."
 * (morse: "D_0", "alpha", "r_0"), and "cutoff = distance from wall at which
 * wall-particle interaction is cut off (distance units)".
 *
 * "The distance between a particle and the surface is the distance to the
 * nearest point on the surface and the force the wall exerts on the particle
 * is along the direction between that point and the particle, which is the
 * direction normal to the surface at that point. Note that if the region
 * surface is comprised of multiple "faces", then each face can exert a force
 * on the particle if it is close enough. E.g. for region_style block, a
 * particle in the interior, near a corner of the block, could feel wall
 * forces from 1, 2, or 3 faces of the block."
 * "The surface of the region only exerts forces on particles "inside" the
 * region; if a particle is "outside" the region it will generate an error,
 * because it has moved through the wall."
 * "The energy of the wall potential is shifted so that the wall-particle
 * interaction energy is 0.0 at the cutoff distance."
 * "This fix computes a global scalar energy and a global 3-length vector of
 * forces ... The scalar energy is the sum of energy interactions for all
 * particles interacting with the wall represented by the region surface. The 3
 * vector quantities are the x,y,z components of the total force acting on the
 * wall due to the particles."
 * "The *fix_modify* *energy* option is supported by this fix ... The default
 * setting for this fix is fix_modify energy no."
 * "The forces due to this fix are imposed during an energy minimization"
 *
 * Measured with native LAMMPS (black box), region sphere 0 0 0 2.2 side in,
 * lj93 1.0 1.0 2.5, one atom at (0.5, 0.3, 0.2): pe = -0.185729948287376
 * (the shifted lj93 energy at r = 1.58356), f_2 vector = (-0.377141548140791,
 * -0.226284928884474, -0.150856619256316): the wall receives the reaction of
 * the force on the atom. Block -2 2 -2 2 -2 2 side in, harmonic 1.0 1.0 2.5,
 * atom at (1.5, 1.5, 0.2): pe = 8.58, the sum over the two faces at 0.5 and
 * the faces at 1.8 and 2.2; the same atom at (2.5, 2.5, 0.2) with side out
 * (exterior, nearest corner at distance 0.70711): pe = 3.21446609406726, one
 * contribution (a single point of the surface), not one per face. An atom
 * inside a side-out region errors with "Particle outside surface of region
 * used in fix wall/region", and an atom outside a side-in region errors the
 * same way.
 *
 * Supported here: block, sphere, cylinder (radlo = radhi), cone (side in),
 * plane (side in), compound (union/intersect) regions and dynamic (move/rotate)
 * regions.  A side-in region contributes one force per face within the cutoff
 * (region.ts filters the faces of a compound region as documented on
 * region.html); a side-out region contributes the nearest point of the solid
 * only.  The plane's single flat face is returned by region.ts (curvature 0);
 * a side-out plane is rejected (the nearest-point path below only knows block,
 * sphere and cylinder).  Not supported (StyleError): a side-out union/intersect
 * region, a side-out plane, and style colloid (rejected in the constructor).
 */
const WR_STYLES = ['lj93', 'lj126', 'lj1043', 'colloid', 'harmonic', 'morse'];

/**
 * Style colloid (src/engine-gpl/wall/fix_wall_region_colloid.ts, the port of fix_wall_region.cpp, hook word colloid):
 * the coefficients of epsilon, sigma and the cutoff, and the force magnitude and shifted energy at distance r for a
 * particle of radius rad.
 */
interface ColloidWallHook {
  colloidWall(epsilon: number, sigma: number, cutoff: number): object;
  colloidWallForce(w: object, r: number, rad: number): { fwall: number; eng: number };
}

export class FixWallRegion extends Fix {
  readonly style = 'wall/region';
  readonly regionId: string;
  readonly wallStyle: string;
  private readonly params: number[];
  private readonly rc: number;
  /** Style colloid: the hook and its coefficients (set by init). */
  private colloid: { hook: ColloidWallHook; w: object } | null = null;
  /** Energy and per-wall-force tallies of the last application. */
  private wallEnergy = 0;
  private wallForce = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/fix_wall_region.cpp FixWallRegion::FixWallRegion at b332f28): the word count is checked before the style name, and a wrong count is "Illegal fix wall/region command" (morse: "... morse command")
    if (args.length < 4) throw new StyleError('Illegal fix wall/region command: missing argument(s)');
    this.regionId = args[0];
    this.wallStyle = args[1];
    if (!WR_STYLES.includes(this.wallStyle)) {
      throw new StyleError(`fix ${id} (wall/region): unknown style '${this.wallStyle}' (lj93, lj126, lj1043, colloid, harmonic or morse)`);
    }
    const nParams = this.wallStyle === 'morse' ? 3 : 2;
    if (args.length !== 2 + nParams + 1) {
      throw new StyleError(this.wallStyle === 'morse' ? 'Illegal fix wall/region morse command' : 'Illegal fix wall/region command');
    }
    this.params = [];
    // As native LAMMPS does (src/fix_wall_region.cpp FixWallRegion::FixWallRegion at b332f28): the parameters and the
    // cutoff are read with utils::numeric, and a cutoff <= 0 is "Fix wall/region cutoff <= 0.0".
    for (let k = 0; k < nParams; k++) {
      this.params.push(numeric(args[2 + k]));
    }
    const cut = numeric(args[args.length - 1]);
    if (cut <= 0) {
      throw new StyleError('Fix wall/region cutoff <= 0.0');
    }
    this.rc = cut;
    // fail early on a region this fix cannot use
    this.regionOf();
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
  }

  /** The region, which may be a primitive or a compound (union/intersect). */
  private regionOf() {
    const r = this.sys.region(this.regionId);
    if (!['block', 'sphere', 'cylinder', 'cone', 'plane', 'union', 'intersect'].includes(r.style)) {
      throw new StyleError(`fix ${this.id} (wall/region): region style ${r.style} is not supported (block, sphere, cylinder, cone, plane, union and intersect are)`);
    }
    if (r.style === 'plane' && !r.interior) {
      throw new StyleError(`fix ${this.id} (wall/region): a side-out plane region is not supported`);
    }
    if (r.hasSideOutSubRegion()) {
      throw new StyleError(`fix ${this.id} (wall/region): region style ${r.style} is not supported (a side-out sub-region is not supported)`);
    }
    return r;
  }

  /**
   * As native LAMMPS does (src/fix_wall_region.cpp init at b332f28): style colloid needs atom attribute radius and only
   * extended particles in the group, and sets its coefficients.
   */
  init(): void {
    if (this.wallStyle !== 'colloid') return;
    const s = this.sys.state;
    if (!s.radius) throw new StyleError('Fix wall/region colloid requires atom attribute radius');
    for (let i = 0; i < s.n; i++) {
      if (this.inGroup(i) && s.radius[i] === 0.0) throw new StyleError('Fix wall/region colloid requires only extended particles');
    }
    const hook = gplHook<ColloidWallHook>('colloid');
    this.colloid = { hook, w: hook.colloidWall(this.params[0], this.params[1], this.rc) };
  }

  /** Region parameter value (number or equal-style variable with scale). */
  private pv(p: number | { variable: string; scale: number }): number {
    return typeof p === 'number' ? p : this.sys.regionEnv.variable(p.variable) * p.scale;
  }

  /** E(d) of the shifted (or, for harmonic, unshifted) wall potential. */
  private energyAt(d: number): number {
    const [a, b, c] = this.params;
    switch (this.wallStyle) {
      case 'lj93': { const u = b / d; return a * ((2 / 15) * u ** 9 - u ** 3); }
      case 'lj126': { const u = b / d; return 4 * a * (u ** 12 - u ** 6); }
      case 'lj1043': {
        const u = b / d;
        const off = d + (0.61 / Math.SQRT2) * b;
        return 2 * Math.PI * a * ((2 / 5) * u ** 10 - u ** 4 - (Math.SQRT2 * b ** 3) / (3 * off ** 3));
      }
      case 'harmonic': return a * (d - this.rc) ** 2;
      default: { // morse: D_0 = a, alpha = b, r_0 = c
        const e = libm.exp(-b * (d - c));
        return a * (e * e - 2 * e);
      }
    }
  }

  /** F(d) = -dE/dd of the unshifted potential (positive pushes the particle away from the surface). */
  private forceAt(d: number): number {
    const [a, b, c] = this.params;
    switch (this.wallStyle) {
      case 'lj93': { const u = b / d; return a * (b / (d * d)) * ((6 / 5) * u ** 8 - 3 * u ** 2); }
      case 'lj126': { const u = b / d; return 4 * a * (b / (d * d)) * (12 * u ** 11 - 6 * u ** 5); }
      case 'lj1043': {
        const u = b / d;
        const off = d + (0.61 / Math.SQRT2) * b;
        return 2 * Math.PI * a * ((4 * u ** 10 - 4 * u ** 4) / d - (Math.SQRT2 * b ** 3) / off ** 4);
      }
      case 'harmonic': return 2 * a * (this.rc - d);
      default: {
        const e = libm.exp(-b * (d - c));
        return 2 * b * a * (e * e - e);
      }
    }
  }

  /** Energy with the cutoff shift E(d) - E(rc) (harmonic has E(rc) = 0). */
  private shifted(d: number): number { return this.energyAt(d) - this.energyAt(this.rc); }

  /** One contribution of a surface point: distance d and unit direction u from the surface to the atom. */
  private contribute(i: number, d: number, ux: number, uy: number, uz: number): void {
    if (!(d < this.rc)) return;
    const s = this.sys.state;
    let f: number, e: number;
    if (this.colloid) {
      // fix_wall_region.cpp post_force: for style colloid a contact at r <= radius is an error
      if (d <= s.radius![i]) throw new StyleError('Particle outside surface of region used in fix wall/region');
      const c = this.colloid.hook.colloidWallForce(this.colloid.w, d, s.radius![i]);
      f = c.fwall;
      e = c.eng;
    } else {
      f = this.forceAt(d);
      e = this.shifted(d);
    }
    s.f[3 * i] += f * ux;
    s.f[3 * i + 1] += f * uy;
    s.f[3 * i + 2] += f * uz;
    this.wallEnergy += e;
    this.wallForce[0] -= f * ux;
    this.wallForce[1] -= f * uy;
    this.wallForce[2] -= f * uz;
  }

  /** The surface contributions of one atom (side in: every face within the cutoff; side out: nearest point). */
  private applyAtom(i: number, r: Region, x: number, y: number, z: number): void {
    // Region.match is true for particles that belong to the region (side in or out)
    if (!r.match(x, y, z)) {
      throw new StyleError('Particle outside surface of region used in fix wall/region');
    }
    if (r.interior) {
      // side in: every face of the region within the cutoff exerts a force on the
      // atom (fix_wall_region.html).  A compound region returns its sub-region
      // faces filtered as documented on region.html; a dynamic region's faces are
      // returned in the lab frame.
      const cs: SurfaceContact[] = [];
      // the faces within the cutoff (native region->surface(x, cutoff); a compound region counts them, see region.ts)
      r.contacts(x, y, z, cs, this.rc);
      for (const c of cs) {
        // fix_wall_region.cpp post_force: a contact at r = 0 (on the surface) is the same error
        if (c.dist <= 0) throw new StyleError('Particle outside surface of region used in fix wall/region');
        this.contribute(i, c.dist, c.nx, c.ny, c.nz);
      }
      return;
    }
    if (r.style === 'union' || r.style === 'intersect') {
      throw new StyleError(`fix ${this.id} (wall/region): a side-out ${r.style} region is not supported`);
    }
    // side out: the particle is in the exterior; the nearest point of the solid
    // is the only one.  The nearest point is found in the region's body frame
    // and mapped back to the lab frame for a moving region.
    const g = this.regionGeometry(r);
    const [bx, by, bz] = r.bodyPoint(x, y, z);
    let sx: number, sy: number, sz: number;
    if (g.kind === 'block') {
      sx = Math.min(Math.max(bx, g.lo[0]), g.hi[0]);
      sy = Math.min(Math.max(by, g.lo[1]), g.hi[1]);
      sz = Math.min(Math.max(bz, g.lo[2]), g.hi[2]);
    } else if (g.kind === 'sphere') {
      const dx = bx - g.c[0], dy = by - g.c[1], dz = bz - g.c[2];
      const rho = Math.hypot(dx, dy, dz);
      const s = rho > 0 ? g.R / rho : 0;
      sx = g.c[0] + dx * s; sy = g.c[1] + dy * s; sz = g.c[2] + dz * s;
    } else {
      const p = [bx, by, bz];
      const { axis, c1, c2, R, lo, hi } = g;
      const [d1, d2] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
      const e1 = p[d1] - c1, e2 = p[d2] - c2;
      const rho = Math.hypot(e1, e2);
      const ac = Math.min(Math.max(p[axis], lo), hi);
      const rc = Math.min(rho, R);
      const q = [0, 0, 0];
      q[axis] = ac;
      q[d1] = c1 + (rho > 0 ? e1 * (rc / rho) : 0);
      q[d2] = c2 + (rho > 0 ? e2 * (rc / rho) : 0);
      [sx, sy, sz] = q;
    }
    const [lx, ly, lz] = r.labPoint(sx, sy, sz);
    const dx = x - lx, dy = y - ly, dz = z - lz;
    const d = Math.hypot(dx, dy, dz);
    if (!(d > 0)) throw new StyleError('Particle outside surface of region used in fix wall/region');
    this.contribute(i, d, dx / d, dy / d, dz / d);
  }

  /** Geometry of a static region in absolute coordinates (parameters evaluated now). */
  private regionGeometry(r: Region):
    | { kind: 'block'; lo: number[]; hi: number[] }
    | { kind: 'sphere'; c: number[]; R: number }
    | { kind: 'cylinder'; axis: 0 | 1 | 2; c1: number; c2: number; R: number; lo: number; hi: number } {
    if (r instanceof BlockRegion) {
      const b = r.b.map((p) => this.pv(p));
      return { kind: 'block', lo: [b[0], b[2], b[4]], hi: [b[1], b[3], b[5]] };
    }
    if (r instanceof SphereRegion) return { kind: 'sphere', c: r.c.map((p) => this.pv(p)), R: this.pv(r.r) };
    if (r instanceof ConeRegion) {
      const rl = this.pv(r.radlo), rh = this.pv(r.radhi);
      if (rl !== rh) throw new StyleError(`fix ${this.id} (wall/region): a cone region is not supported (cylinder radlo = radhi only)`);
      return { kind: 'cylinder', axis: r.axis, c1: this.pv(r.c1), c2: this.pv(r.c2), R: rl, lo: this.pv(r.lo), hi: this.pv(r.hi) };
    }
    throw new StyleError(`fix ${this.id} (wall/region): region style ${r.style} is not supported`);
  }

  postForce(): void {
    const r = this.regionOf();
    const s = this.sys.state;
    this.wallEnergy = 0;
    this.wallForce.fill(0);
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      this.applyAtom(i, r, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
    }
  }

  minPostForce(): void { this.postForce(); }

  energy(): number { return this.wallEnergy; }
  computeScalar(): number { return this.wallEnergy; }
  computeVector(i: number): number { return this.wallForce[i]; }
}

/*
 * fix wall/reflect/stochastic — docs.lammps.org/fix_wall_reflect_stochastic.html,
 * Syntax:
 *
 *   fix ID group-ID wall/reflect/stochastic rstyle seed face args ... keyword value ...
 *
 * "rstyle = diffusive or maxwell or ccl"; "seed = random seed for stochasticity
 * (positive integer)"; "face = *xlo* or ... *zhi*"; "args = pos temp velx vely
 * velz accomx accomy accomz" with "pos = EDGE or constant", "temp = wall
 * temperature (temperature units)", "velx,vely,velz = wall velocity in x,y,z
 * directions (velocity units)", "accomx,accomy,accomz = accommodation coeffs in
 * x,y,z directions (unitless)"; "not specified for rstyle = diffusive"; "single
 * accom coeff specified for rstyle maxwell"; "all 3 coeffs specified for rstyle
 * cll". "keyword = *units*": "*lattice* = the wall position is defined in lattice
 * units", "*box* = the wall position is defined in simulation box units".
 *
 * "This fix models treats the wall as a moving solid boundary with a finite
 * temperature, which can exchange energy with particles that collide with it.
 * ... For this fix, the post collision velocity of each particle is treated
 * stochastically."
 * "For rstyle *diffusive*, particles are reflected diffusively. Their velocity
 * distribution corresponds to an equilibrium distribution of particles at the
 * wall temperature. No accommodation coefficients are specified."
 * "For rstyle *maxwell*, particle reflection is Maxwellian which means partially
 * diffusive and partially specular ... A single accommodation coeff is specified
 * which must be between 0.0 and 1.0 inclusive. It determines the fraction of the
 * collision which is diffusive versus specular. An accommodation coefficient of
 * 1.0 is fully diffusive; a coefficient of 0.0 is fully specular."
 * "For rstyle *cll*, particle collisions are computed by the Cercignani/Lampis
 * model. ... Three accommodations coefficient are specified. Each must be between
 * 0.0 and 1.0 inclusive. Two are velocity accommodation coefficients; one is a
 * normal kinetic energy accommodation. The normal coeff is the one corresponding
 * to the normal of the wall itself. For example if the wall is *ylo* or *yhi*,
 * *accomx* and *accomz* are the tangential velocity accommodation coefficients,
 * and *accomy* is the normal kinetic energy accommodation coefficient."
 * "The wall velocity must lie on the same plane as the wall itself."
 * "Any dimension (xyz) that has a wall must be non-periodic."
 * "The default for the units keyword is lattice."
 *
 * Reflection timing: the check runs in end_of_step like fix wall/reflect (the
 * Developer_flow hook order of docs.lammps.org/Developer_flow.html). Position
 * (measured with native LAMMPS, black box, dt = 0.005, v = -2, wall at 0, atom at
 * z = 0.305 -> z after the step 31 is 0.5 dt vz_new, not the mirror image): the
 * atom spends tau = penetration / |v_in| beyond the wall, so it is placed at
 * wall + vn_new tau, and then zu(40) = 9.5 dt vz_new = 0.0436092854 (native) for
 * the atom crossing in step 31. Its tangential position is corrected by
 * (v_new - v_old) tau over the same time (native yu of an atom with v_y changed
 * from 0 to -2.00136 at a half-step crossing: 2.93495577).
 *
 * Random numbers. Measured with native LAMMPS (black box), one atom at z = 0.3
 * moving at vz = -2 with fix nve, dt = 0.005, wall zlo 0.0 with temp 1.0 and
 * walls at zero velocity (mass 1, kB 1):
 *   - the stream is the Marsaglia-Zaman RANMAR (rng.ts RanMars, as fix
 *     langevin uses it), seeded with the seed as given. ONE uniform is drawn when
 *     the fix is created (warm-up): the first collision of seed 12345 starts
 *     at the second draw; a second collision of the same atom continues the
 *     stream with no further warm-up draw.
 *   - diffusive, per collision in storage order: the tangential pair comes from
 *     the first accepted polar (Marsaglia) pair (u1, u2) with s = v1^2 + v2^2 < 1,
 *     v1 = 2 u1 - 1, v2 = 2 u2 - 1, f = sqrt(-2 ln s / s): vx = sigma v2 f,
 *     vy = sigma v1 f (sigma = sqrt(kB T / m)); the normal speed is
 *     vn = sigma sqrt(-2 ln u) with u the uniform after the accepted pair.
 *     Rejected pairs consume their two draws (seed 12345: pairs starting at
 *     draws 1, 3 rejected, pair 5 accepted, vz from draw 7).
 *   - maxwell: one uniform u_c (after the warm-up or after the previous
 *     collision) is drawn first; u_c < accom gives the diffusive collision above,
 *     otherwise the specular one (vn -> -vn, tangential unchanged, no further
 *     draws). Checked for accom 0.5 (atom 1 specular, atom 2 diffusive) and 1.0.
 *   - ccl: the tangential components are exact against native for (accomx,
 *     accomy) = (0.5, 0.6): vx = sigma_x v2 f, vy = sigma_y v1 f with
 *     sigma_t = sqrt(alpha_t (2 - alpha_t) kB T / m). The normal component is
 *     NOT reproduced draw for draw: it is implemented as the radial part of a 2-D
 *     Gaussian, vn = sqrt((mu + s g1)^2 + (s g2)^2), mu = sqrt(1 - alpha_n) |vn_in|,
 *     s = sqrt(alpha_n kB T / m), with g1, g2 the next accepted polar pair; native
 *     agrees in distribution (400 atoms: <vz^2> = 2.67 measured, 2.60 analytic
 *     with alpha_n = 0.7, |vn_in| = 2), but its draw order is not identified.
 *
 * Wall velocity: the tangential mean is the wall velocity for diffusive and
 * maxwell (diffusive part); for ccl it is u + (1 - alpha)(v - u) for each
 * tangential component (kernel in the relative velocity).
 */
export class FixWallReflectStochastic extends Fix {
  readonly style = 'wall/reflect/stochastic';
  private readonly rstyle: 'diffusive' | 'maxwell' | 'ccl';
  private readonly faces: StochFace[] = [];
  private unitsMode: 'lattice' | 'box' = 'lattice';
  private readonly rng: RanMars;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/EXTRA-FIX/fix_wall_reflect_stochastic.cpp at b332f28): every bad word is "Illegal fix
    // wall/reflect/stochastic command"; the periodic and 2d checks follow the words
    const illegal = (): StyleError => new StyleError('Illegal fix wall/reflect/stochastic command');
    if (args.length < 5) throw illegal();
    if (sys.geom.triclinic) throw new StyleError('Fix wall/reflect/stochastic cannot be used with triclinic simulation box');
    const rstyle = args[0];
    if (rstyle !== 'diffusive' && rstyle !== 'maxwell' && rstyle !== 'ccl') throw illegal();
    this.rstyle = rstyle;
    const seed = inumeric(args[1]);
    if (seed <= 0) throw new StyleError('Random seed must be a positive number');
    const nAccom = rstyle === 'diffusive' ? 0 : rstyle === 'maxwell' ? 1 : 3;
    const nArgs = 5 + nAccom;
    for (let k = 2; k < args.length;) {
      const w = args[k];
      if (w === 'units') {
        if (k + 2 > args.length) throw new StyleError('Illegal wall/reflect/stochastic command');
        const v = args[k + 1];
        if (v !== 'lattice' && v !== 'box') throw illegal();
        this.unitsMode = v;
        k += 2;
        continue;
      }
      const dim = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'].indexOf(w);
      if (dim < 0) throw illegal();
      if (k + 1 + nArgs > args.length) throw illegal();
      if (this.faces.some((f) => f.face === w)) throw new StyleError('Fix wall/reflect/stochastic command wall defined twice');
      const a = args.slice(k + 1, k + 1 + nArgs);
      const pos = a[0] === 'EDGE' ? null : numeric(a[0]);
      const temp = numeric(a[1]);
      const wv: [number, number, number] = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        wv[d] = numeric(a[2 + d]);
        if (wv[d] !== 0 && d === dim >> 1) throw new StyleError('The wall velocity must be tangential');
      }
      const accom = a.slice(5).map((s) => numeric(s));
      // the engine's own limits (native does not check them)
      if (!(temp >= 0)) throw new StyleError(`fix ${id} (wall/reflect/stochastic): ${w} temperature must be >= 0`);
      for (const v of accom) {
        if (!(v >= 0 && v <= 1)) throw new StyleError(`fix ${id} (wall/reflect/stochastic): accommodation coefficients must be between 0 and 1, got ${v}`);
      }
      this.faces.push({ face: w, dim: (dim >> 1) as 0 | 1 | 2, lo: (dim & 1) === 0, pos, temp, wv, accom });
      k += 1 + nArgs;
    }
    if (!this.faces.length) throw new StyleError('Illegal fix wall command');
    for (const f of this.faces) {
      if (sys.state.box.periodic[f.dim]) throw new StyleError('Cannot use fix wall/reflect/stochastic in periodic dimension');
    }
    if (sys.dimension === 2 && this.faces.some((f) => f.dim === 2)) {
      throw new StyleError('Cannot use fix wall/reflect/stochastic zlo/zhi for a 2d simulation');
    }
    this.rng = new RanMars(seed);
    this.rng.uniform(); // warm-up draw made when the fix is created (measured with native LAMMPS)
  }

  /** native FixWallReflectStochastic inherits FixWallReflect::post_integrate (src/EXTRA-FIX/fix_wall_reflect_stochastic.h at b332f28). */
  postIntegrate(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const sp: number[] = this.unitsMode === 'lattice' && sys.lattice ? [...sys.lattice.spacing] : [1, 1, 1];
    const pos = this.faces.map((f) => (f.pos === null ? (f.lo ? g.lo[f.dim] : g.hi[f.dim]) : f.pos * (this.unitsMode === 'lattice' ? sp[f.dim] : 1)));
    const { x, v, mask } = s;
    // per-atom draws follow native LAMMPS's atom list (SimState.order)
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      for (let k = 0; k < this.faces.length; k++) {
        const f = this.faces[k];
        const d = f.dim;
        const j = 3 * i + d;
        if (!(f.lo ? x[j] < pos[k] : x[j] > pos[k])) continue;
        // collision: the atom spent tau = penetration / |v_in| beyond the wall; it is placed at
        // the wall and moves for tau with the post-collision normal velocity (measured, see header)
        const m = massOf(s, i);
        const sigma = Math.sqrt((sys.state.units.boltz * f.temp) / (sys.state.units.mvv2e * m));
        const delta = f.lo ? pos[k] - x[j] : x[j] - pos[k];
        const vin = Math.abs(v[j]);
        const vOld = [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
        this.collide(f, sigma, v, 3 * i);
        const tau = vin > 0 ? delta / vin : 0;
        x[j] = pos[k] + v[j] * tau;
        // the tangential motion over the same remaining time tau uses the new velocity
        for (let t = 0; t < 3; t++) {
          if (t !== d) x[3 * i + t] += (v[3 * i + t] - vOld[t]) * tau;
        }
      }
    }
  }

  /** Post-collision velocity of one atom (velocity array v at offset o) for one face. */
  private collide(f: StochFace, sigma: number, v: Float64Array, o: number): void {
    const d = f.dim;
    const tang = [0, 1, 2].filter((c) => c !== d) as [number, number];
    const vin = v[o + d];
    const sgn = f.lo ? 1 : -1;
    if (this.rstyle === 'maxwell') {
      const accom = f.accom[0];
      const uc = this.rng.uniform();
      if (!(uc < accom)) {
        v[o + d] = -vin; // specular: normal reversed, tangential kept
        return;
      }
    }
    if (this.rstyle === 'ccl') {
      const [g1, g2] = this.rng.polarPair();
      const [t1, t2] = [tang[0], tang[1]];
      const at = [f.accom[t1], f.accom[t2]];
      const sx = Math.sqrt(at[0] * (2 - at[0]) * (sigma * sigma)), sy = Math.sqrt(at[1] * (2 - at[1]) * (sigma * sigma));
      v[o + t1] = f.wv[t1] + (1 - at[0]) * (v[o + t1] - f.wv[t1]) + sx * g1;
      v[o + t2] = f.wv[t2] + (1 - at[1]) * (v[o + t2] - f.wv[t2]) + sy * g2;
      const an = f.accom[d];
      const s2 = an * sigma * sigma;
      const mu = Math.sqrt(1 - an) * Math.abs(vin);
      const [h1, h2] = this.rng.polarPair();
      const sn = Math.sqrt(s2);
      const rad = Math.hypot(mu + sn * h1, sn * h2);
      v[o + d] = sgn * rad;
      return;
    }
    // diffusive (and the diffusive branch of maxwell): tangential pair, then the flux-weighted normal speed
    const [g1, g2] = this.rng.polarPair();
    const [t1, t2] = [tang[0], tang[1]];
    v[o + t1] = f.wv[t1] + sigma * g1;
    v[o + t2] = f.wv[t2] + sigma * g2;
    const u = this.rng.uniform();
    v[o + d] = sgn * sigma * Math.sqrt(-2 * libm.log(u));
  }
}

interface StochFace {
  face: string;
  dim: 0 | 1 | 2;
  lo: boolean;
  /** Position (distance units) or null for EDGE. */
  pos: number | null;
  temp: number;
  wv: [number, number, number];
  accom: number[];
}
