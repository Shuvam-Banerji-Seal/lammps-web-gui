import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { globalScalar, parseRef, type Ref } from '../refs';
import { BlockRegion, ConeRegion, SphereRegion, type Region, type SurfaceContact } from '../region';
import { massOf, nativeOrder } from '../atoms';
import { RanMars } from '../rng';

/*
 * Fixes of the misc3 wave: fix vector, fix wall/region, fix wall/reflect/stochastic.
 * Implemented from docs.lammps.org pages only (never from LAMMPS source).
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
 * "If the fix produces a vector, then the entire vector will be either
 * "intensive" or "extensive", depending on whether the values stored in the
 * vector are "intensive" or "extensive". If the fix produces an array, then
 * all elements in the array must be the same, either "intensive" or
 * "extensive". ... Values produced by a variable are treated as intensive."
 * "This fix is not invoked during energy minimization."
 *
 * Measured with native LAMMPS (black box): the step-0 sample is taken at the
 * setup of a run (run 0 stores one entry); a run that starts on a multiple of
 * Nevery does not store that step twice; an f_ID[I] reference on a step that
 * is not a multiple of Nevery errors (not computed at a compatible time);
 * mixing an extensive and an intensive input errors ("cannot set output array
 * intensive/extensive from these inputs").
 *
 * Deviation: the default storage of LAMMPS (sized from the run length, and
 * discarding the oldest entry for "pre no" runs) is not emulated; without
 * nmax the vector grows for the whole session (the measured behaviour of
 * pre yes runs).
 */
export class FixVector extends Fix {
  readonly style = 'vector';
  private readonly inputs: Ref[] = [];
  private readonly nmax: number | null;
  /** One row per stored sample; one column per input value. */
  private rows: number[][] = [];
  private lastStep = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const usage = `usage: fix ID group-ID vector Nevery value1 value2 ... [nmax length]`;
    const every = Number(args[0]);
    if (args.length < 2 || !Number.isInteger(every) || every < 1) {
      throw new StyleError(`fix ${id} (vector): ${usage} (Nevery must be a positive integer, got '${args[0] ?? ''}')`);
    }
    this.nevery = every;
    let nmax: number | null = null;
    let k = 1;
    for (; k < args.length; k++) {
      const w = args[k];
      if (w === 'nmax') {
        const v = Number(args[k + 1]);
        if (!Number.isInteger(v) || v < 1) throw new StyleError(`fix ${id} (vector): nmax must be a positive integer, got '${args[k + 1] ?? ''}'`);
        nmax = v;
        k++;
        continue;
      }
      if (nmax !== null) {
        throw new StyleError(`fix ${id} (vector): unexpected '${w}' after nmax (input values come first, then keywords)`);
      }
      this.inputs.push(this.checkInput(w));
    }
    if (!this.inputs.length) throw new StyleError(`fix ${id} (vector): ${usage} (no input value given)`);
    this.nmax = nmax;
    // the extensive/intensive flag of the output (fix.html: vector or array rules)
    const exts = this.inputs.map((r) => this.extOf(r));
    const allSame = exts.every((e) => e === exts[0]);
    if (this.inputs.length === 1) {
      this.vectorFlag = true;
      this.extvector = exts[0];
    } else {
      if (!allSame) {
        throw new StyleError(`fix ${id} (vector): Fix vector cannot set output array intensive/extensive from these inputs`);
      }
      this.arrayFlag = true;
      this.sizeArrayCols = this.inputs.length;
      this.extvector = exts[0];
    }
  }

  /** Validates an input word (c_ID, c_ID[I], f_ID, f_ID[I], v_name, v_name[I]) and that its source exists. */
  private checkInput(w: string): Ref {
    const r = parseRef(w);
    if (r.kind === 'v') {
      if (!this.sys.vars.has(r.id)) throw new StyleError(`fix ${this.id} (vector): variable ${r.id} for fix vector does not exist`);
    } else if (r.kind === 'c') {
      const c = this.sys.compute(r.id);
      if (c.peratomFlag) throw new StyleError(`fix ${this.id} (vector): compute ${r.id} must produce a global quantity, not per-atom`);
    } else {
      this.sys.fix(r.id);
    }
    return r;
  }

  /** Intensive (0) or extensive (1) of one input value (refs.ts scalar rules). */
  private extOf(r: Ref): number {
    if (r.kind === 'v') return 0; // "Values produced by a variable are treated as intensive."
    if (r.kind === 'c') {
      const c = this.sys.compute(r.id);
      return r.index === null ? c.extscalar : c.extvector;
    }
    const f = this.sys.fix(r.id);
    return r.index === null ? f.extscalar : f.extvector;
  }

  /** Stores the inputs for the current step; nmax discards the oldest row. */
  private sample(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0 || step === this.lastStep) return;
    this.lastStep = step;
    this.rows.push(this.inputs.map((r) => globalScalar(this.sys, r)));
    if (this.nmax !== null && this.rows.length > this.nmax) this.rows.shift();
    this.sizeVector = this.rows.length;
    this.sizeArrayRows = this.rows.length;
  }

  /** Setup stores the step-0 (or first multiple of Nevery) sample. */
  setup(): void { this.sample(); }

  endOfStep(): void { this.sample(); }

  /** "The values can only be accessed on timesteps that are multiples of Nevery." */
  private checkTime(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0) {
      throw new StyleError(`Fix with ID ${this.id} not computed at a compatible time (step ${step} is not a multiple of Nevery = ${this.nevery})`);
    }
  }

  computeVector(i: number): number {
    this.checkTime();
    if (!this.vectorFlag) throw new StyleError(`fix ${this.id} (vector) with several inputs is an array: use f_${this.id}[I][J]`);
    const row = this.rows[i];
    if (!row) throw new StyleError(`f_${this.id}[${i + 1}] is out of range (1..${this.rows.length})`);
    return row[0];
  }

  computeArray(i: number, j: number): number {
    this.checkTime();
    if (!this.arrayFlag) throw new StyleError(`fix ${this.id} (vector) has a single input: its output is a vector, use f_${this.id}[I]`);
    const row = this.rows[i];
    if (!row) throw new StyleError(`f_${this.id}[${i + 1}][${j + 1}] is out of range (1..${this.rows.length})`);
    return row[j];
  }
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

export class FixWallRegion extends Fix {
  readonly style = 'wall/region';
  readonly regionId: string;
  readonly wallStyle: string;
  private readonly params: number[];
  private readonly rc: number;
  /** Energy and per-wall-force tallies of the last application. */
  private wallEnergy = 0;
  private wallForce = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const usage = 'usage: fix ID group-ID wall/region region-ID style args ... cutoff';
    if (args.length < 2) throw new StyleError(`fix ${id} (wall/region): ${usage}`);
    this.regionId = args[0];
    this.wallStyle = args[1];
    if (!WR_STYLES.includes(this.wallStyle)) {
      throw new StyleError(`fix ${id} (wall/region): unknown style '${this.wallStyle}' (lj93, lj126, lj1043, colloid, harmonic or morse)`);
    }
    if (this.wallStyle === 'colloid') {
      throw new StyleError(`fix ${id} (wall/region): style colloid is not supported (the measured native values do not match the doc formula yet)`);
    }
    const nParams = this.wallStyle === 'morse' ? 3 : 2;
    if (args.length !== 2 + nParams + 1) {
      throw new StyleError(`fix ${id} (wall/region): style ${this.wallStyle} takes ${nParams} parameters and a cutoff (${usage})`);
    }
    this.params = [];
    for (let k = 0; k < nParams; k++) {
      const v = Number(args[2 + k]);
      if (args[2 + k].startsWith('v_') || !Number.isFinite(v)) {
        throw new StyleError(`fix ${id} (wall/region): parameter '${args[2 + k]}' must be a number`);
      }
      this.params.push(v);
    }
    const cut = Number(args[args.length - 1]);
    if (args[args.length - 1].startsWith('v_') || !(cut > 0)) {
      throw new StyleError(`fix ${id} (wall/region): cutoff must be a number > 0, got '${args[args.length - 1]}'`);
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
        const e = Math.exp(-b * (d - c));
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
        const e = Math.exp(-b * (d - c));
        return 2 * b * a * (e * e - e);
      }
    }
  }

  /** Energy with the cutoff shift E(d) - E(rc) (harmonic has E(rc) = 0). */
  private shifted(d: number): number { return this.energyAt(d) - this.energyAt(this.rc); }

  /** One contribution of a surface point: distance d and unit direction u from the surface to the atom. */
  private contribute(i: number, d: number, ux: number, uy: number, uz: number): void {
    if (!(d < this.rc)) return;
    const f = this.forceAt(d);
    const s = this.sys.state;
    s.f[3 * i] += f * ux;
    s.f[3 * i + 1] += f * uy;
    s.f[3 * i + 2] += f * uz;
    this.wallEnergy += this.shifted(d);
    this.wallForce[0] -= f * ux;
    this.wallForce[1] -= f * uy;
    this.wallForce[2] -= f * uz;
  }

  /** The surface contributions of one atom (side in: every face within the cutoff; side out: nearest point). */
  private applyAtom(i: number, r: Region, x: number, y: number, z: number): void {
    // Region.match is true for particles that belong to the region (side in or out)
    if (!r.match(x, y, z)) {
      throw new StyleError(`Particle outside surface of region used in fix ${this.id} wall/region`);
    }
    if (r.interior) {
      // side in: every face of the region within the cutoff exerts a force on the
      // atom (fix_wall_region.html).  A compound region returns its sub-region
      // faces filtered as documented on region.html; a dynamic region's faces are
      // returned in the lab frame.
      const cs: SurfaceContact[] = [];
      r.contacts(x, y, z, cs);
      for (const c of cs) {
        if (c.dist <= 0) throw new StyleError(`Particle on or inside fix ${this.id} wall/region surface`);
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
    if (!(d > 0)) throw new StyleError(`Particle on or inside fix ${this.id} wall/region surface`);
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
    const usage = 'usage: fix ID group-ID wall/reflect/stochastic rstyle seed face args ... [units lattice|box]';
    const rstyle = args[0];
    if (rstyle !== 'diffusive' && rstyle !== 'maxwell' && rstyle !== 'ccl') {
      throw new StyleError(`fix ${id} (wall/reflect/stochastic): rstyle must be diffusive, maxwell or ccl, got '${rstyle ?? ''}' (${usage})`);
    }
    this.rstyle = rstyle;
    const seed = Number(args[1]);
    if (!Number.isInteger(seed) || seed < 1 || seed > 900000000) {
      throw new StyleError(`fix ${id} (wall/reflect/stochastic): seed must be a positive integer below 900000000, got '${args[1] ?? ''}'`);
    }
    const nAccom = rstyle === 'diffusive' ? 0 : rstyle === 'maxwell' ? 1 : 3;
    const nArgs = 5 + nAccom;
    for (let k = 2; k < args.length;) {
      const w = args[k];
      if (w === 'units') {
        const v = args[k + 1];
        if (v !== 'lattice' && v !== 'box') throw new StyleError(`fix ${id} (wall/reflect/stochastic): units must be lattice or box, got '${v ?? ''}'`);
        this.unitsMode = v;
        k += 2;
        continue;
      }
      const dim = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'].indexOf(w);
      if (dim < 0) throw new StyleError(`fix ${id} (wall/reflect/stochastic): unknown argument '${w}' (expected a face xlo..zhi or units)`);
      if (this.faces.some((f) => f.face === w)) throw new StyleError(`fix ${id} (wall/reflect/stochastic): face ${w} is specified more than once`);
      const a = args.slice(k + 1, k + 1 + nArgs);
      if (a.length < nArgs) {
        throw new StyleError(`fix ${id} (wall/reflect/stochastic): ${rstyle} needs ${nArgs} arguments after face ${w} (pos temp velx vely velz${nAccom ? ' accom...' : ''})`);
      }
      const num = (s: string, what: string): number => {
        const v = Number(s);
        if (s.startsWith('v_') || !Number.isFinite(v)) throw new StyleError(`fix ${id} (wall/reflect/stochastic): ${what} must be a number, got '${s}'`);
        return v;
      };
      const pos = a[0] === 'EDGE' ? null : num(a[0], `${w} position`);
      const temp = num(a[1], `${w} temperature`);
      if (!(temp >= 0)) throw new StyleError(`fix ${id} (wall/reflect/stochastic): ${w} temperature must be >= 0`);
      const wv: [number, number, number] = [num(a[2], `${w} velx`), num(a[3], `${w} vely`), num(a[4], `${w} velz`)];
      if (wv[dim >> 1] !== 0) {
        throw new StyleError(`fix ${id} (wall/reflect/stochastic): the wall velocity must lie in the plane of the ${w} wall (normal component ${wv[dim >> 1]} is not 0)`);
      }
      const accom = a.slice(5).map((s, q): number => {
        const v = num(s, `${w} accommodation coefficient ${q + 1}`);
        if (!(v >= 0 && v <= 1)) throw new StyleError(`fix ${id} (wall/reflect/stochastic): accommodation coefficients must be between 0 and 1, got ${v}`);
        return v;
      });
      this.faces.push({ face: w, dim: (dim >> 1) as 0 | 1 | 2, lo: (dim & 1) === 0, pos, temp, wv: [wv[0], wv[1], wv[2]], accom });
      k += 1 + nArgs;
    }
    if (!this.faces.length) throw new StyleError(`fix ${id} (wall/reflect/stochastic): ${usage} (no face given)`);
    this.rng = new RanMars(seed);
    this.rng.uniform(); // warm-up draw made when the fix is created (measured with native LAMMPS)
  }

  init(): void {
    const s = this.sys.state;
    for (const f of this.faces) {
      if (s.box.periodic[f.dim]) {
        throw new StyleError(`fix ${this.id} (wall/reflect/stochastic): a wall requires a non-periodic dimension (${f.face} is periodic)`);
      }
    }
  }

  endOfStep(): void {
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
    v[o + d] = sgn * sigma * Math.sqrt(-2 * Math.log(u));
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
