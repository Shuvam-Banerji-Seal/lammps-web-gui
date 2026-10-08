import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNumOrVar, valueOf, type NumOrVar } from './util';

/*
 * Wall fixes — docs.lammps.org/fix_wall.html (fix_wall.rst). One class per
 * style; all share the syntax
 *
 *   fix ID group-ID style [tabstyle] [N] face args ... keyword value ...
 *
 * with "face = *xlo* or *xhi* or *ylo* or *yhi* or *zlo* or *zhi*", "one or
 * more face/arg pairs may be appended", "zero or more keyword/value pairs may
 * be appended" and "keyword = *units* or *fld* or *pbc*":
 *
 *   *units* value = *lattice* or *box*
 *   *fld* value = *yes* or *no*
 *   *pbc* value = *yes* or *no*
 *
 * "In all cases, *r* is the distance from the particle to the wall at
 * position *coord*, and :math:`r_c` is the *cutoff* distance at which the
 * particle and wall no longer interact. The energy of the wall
 * potential is shifted so that the wall-particle interaction energy is
 * 0.0 at the cutoff distance."
 * "The position of each wall can be specified in one of 3 ways: as the EDGE
 * of the simulation box, as a constant value, or as a variable."
 * "The *units* keyword determines the meaning of the distance units used to
 * define a wall position, but only when a numeric constant or variable is
 * used.  It is not relevant when EDGE is used to specify a face position."
 * "A *lattice* value means the distance units are in lattice spacings"
 * (converted with the lattice spacing of the wall dimension; by default
 * lattice.html defines lattice none 1.0, so one lattice spacing is one
 * distance unit). "A *box* value selects standard
 * distance units as defined by the units command".
 *
 * "The *fld* keyword can be used with a *yes* setting to invoke the wall
 * constraint before pairwise interactions are computed. ... If the setting is
 * no, wall forces are imposed after pairwise interactions, in the usual
 * manner." — fld yes applies the forces in pre_force, else in post_force
 * (docs.lammps.org/Developer_flow.html).
 *
 * "The *pbc* keyword can be used with a *yes* setting to allow walls to be
 * specified in a periodic dimension. ... The default for *pbc* is *no*, which
 * means the system must be non-periodic when using a wall."
 *
 * Output: "This fix computes a global scalar energy and a global vector of
 * forces ... the scalar energy is the sum of interactions with all defined
 * walls ... The length of the vector is equal to the number of walls defined
 * by the fix.  Each vector value is the normal force on a specific wall.
 * Note that an outward force on a wall will be
 * a negative value for *lo* walls and a positive value for *hi* walls. The scalar and vector values
 * calculated by this fix are "extensive"."
 * fix_modify: "The fix_modify *energy* option is supported by this fix ...
 * The default setting for this fix is fix_modify energy no." and "The
 * fix_modify *virial* option is supported by this fix ... The default setting
 * for this fix is fix_modify virial no." The virial is tallied as r . F with
 * r the vector from the wall to the atom along the wall axis (the wall acts
 * like a second body of a pair interaction fixed at coord).
 * "The forces due to this fix are imposed during an energy minimization,
 * invoked by the minimize command."
 *
 * Restriction measured with native LAMMPS (2Sep2026): an atom of the fix
 * group on (r = 0) or behind (r < 0) a wall errors with
 * "Particle on or inside fix wall surface" regardless of the cutoff; atoms
 * outside the fix group do not trigger it (fix_wall.html: "For all of the
 * styles, you must ensure that r is always > 0 for
 * all particles in the group, or LAMMPS will generate an error.  This
 * means you cannot start your simulation with particles at the wall
 * position *coord* (r = 0) or with particles on the wrong side of the
 * wall (r < 0).")
 *
 * Units: energies are in LAMMPS energy units (the style parameters are
 * "epsilon = strength factor for wall-particle interaction (energy or energy/distance\^2 units)",
 * "sigma = size factor for wall-particle interaction (distance units)"); the
 * force added to the engine's force array is
 * -dE/dr, which is already in LAMMPS force units (energy/distance).
 */

const FACES = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'] as const;
type Face = (typeof FACES)[number];
const isFace = (w: string): w is Face => (FACES as readonly string[]).includes(w);

/** One wall: axis, side, position, style parameters, cutoff. */
interface Wall {
  dim: 0 | 1 | 2;
  lo: boolean;
  coord: { edge: true } | { edge: false; v: NumOrVar };
  /** Style parameters that may be variables (epsilon/sigma, or D_0/alpha/r_0). */
  params: NumOrVar[];
  /** "cutoff = distance from wall at which wall-particle interactions are cut off". */
  cutoff: number;
}

export abstract class FixWallBase extends Fix {
  protected readonly walls: Wall[] = [];
  private unitsMode: 'lattice' | 'box' = 'lattice';
  private fld = false;
  private pbcOk = false;
  /** Per-wall normal force tallied in the last force application (f_ID vector). */
  private wallForce = new Float64Array(0);
  /** Sum of shifted interaction energies with all walls (f_ID scalar). */
  private wallEnergy = 0;

  constructor(sys: System, id: string, group: string, args: string[], protected readonly wallStyle: string, nArgs: number) {
    super(sys, id, group, args);
    if (args.length < 1) {
      throw new StyleError(`usage: fix ID group-ID ${wallStyle} face args ... [face args ...] [keyword value ...]`);
    }
    for (let k = 0; k < args.length;) {
      const w = args[k];
      if (isFace(w)) {
        if (this.walls.some((x) => FACES[x.dim * 2 + (x.lo ? 0 : 1)] === w)) {
          throw new StyleError(`fix ${id} (${wallStyle}): face ${w} is specified more than once`);
        }
        const words = args.slice(k + 1, k + 1 + nArgs);
        if (words.length < nArgs) {
          throw new StyleError(`fix ${id} (${wallStyle}): face ${w} needs ${nArgs} arguments (args for this style)`);
        }
        this.walls.push(this.parseWall(w, words));
        k += 1 + nArgs;
      } else if (w === 'units') {
        const v = args[k + 1];
        if (v !== 'lattice' && v !== 'box') throw new StyleError(`fix ${id} (${wallStyle}): units must be lattice or box, got '${v ?? ''}'`);
        this.unitsMode = v;
        k += 2;
      } else if (w === 'fld') {
        this.fld = FixWallBase.yesno(id, wallStyle, args[k + 1], 'fld');
        k += 2;
      } else if (w === 'pbc') {
        this.pbcOk = FixWallBase.yesno(id, wallStyle, args[k + 1], 'pbc');
        k += 2;
      } else {
        throw new StyleError(`fix ${id} (${wallStyle}): unknown argument '${w}' (expected a face xlo..zhi or the keyword units, fld or pbc)`);
      }
    }
    if (!this.walls.length) throw new StyleError(`fix ${id} (${wallStyle}): no wall face (xlo, xhi, ylo, yhi, zlo or zhi) was specified`);
    this.wallForce = new Float64Array(this.walls.length);
    // "This fix computes a global scalar energy and a global vector of forces"
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = this.walls.length;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
    this.virialGlobal = true;
  }

  private static yesno(id: string, style: string, w: string | undefined, key: string): boolean {
    if (w !== 'yes' && w !== 'no') throw new StyleError(`fix ${id} (${style}): ${key} value must be yes or no, got '${w ?? ''}'`);
    return w === 'yes';
  }

  /** coord epsilon sigma cutoff (or coord D_0 alpha r_0 cutoff for morse). */
  private parseWall(face: Face, words: string[]): Wall {
    const dim = 'xyz'.indexOf(face[0]) as 0 | 1 | 2;
    const lo = face[1] === 'l';
    const coordW = words[0];
    const coord: Wall['coord'] = coordW === 'EDGE'
      ? { edge: true }
      : coordW.startsWith('v_') ? { edge: false, v: { variable: coordW.slice(2) } }
        : { edge: false, v: parseNumOrVar(coordW, `${face} wall coord`) };
    if (!coord.edge && typeof coord.v === 'object') {
      if (!this.sys.vars.has(coord.v.variable)) throw new StyleError(`fix ${this.id} (${this.wallStyle}): variable ${coord.v.variable} does not exist`);
    }
    const params: NumOrVar[] = [];
    for (let p = 1; p < words.length - 1; p++) {
      const w = words[p];
      const v = parseNumOrVar(w, `${face} wall parameter ${p}`);
      if (typeof v !== 'number' && !this.sys.vars.has(v.variable)) {
        throw new StyleError(`fix ${this.id} (${this.wallStyle}): variable ${v.variable} does not exist`);
      }
      params.push(v);
    }
    const cutW = words[words.length - 1];
    const cutoff = Number(cutW);
    if (cutW.startsWith('v_') || !Number.isFinite(cutoff)) {
      throw new StyleError(`fix ${this.id} (${this.wallStyle}): cutoff must be a number (distance units), got '${cutW}'`);
    }
    if (!(cutoff > 0)) throw new StyleError(`fix ${this.id} (${this.wallStyle}): cutoff must be > 0, got '${cutW}'`);
    return { dim, lo, coord, params, cutoff };
  }

  init(): void {
    const s = this.sys.state;
    if (s.dimension === 2) {
      const z = this.walls.find((w) => w.dim === 2);
      if (z) throw new StyleError(`fix ${this.id} (${this.wallStyle}): cannot use a z wall (zlo/zhi) in a 2d simulation`);
    }
    for (const w of this.walls) {
      if (!this.pbcOk && s.box.periodic[w.dim]) {
        throw new StyleError(`fix ${this.id} (${this.wallStyle}): wall in a periodic dimension requires pbc yes (the default is pbc no)`);
      }
    }
  }

  /** "invoke the wall constraint before pairwise interactions" (fld yes). */
  preForce(): void { this.apply(); }

  /** wall forces are "imposed after pairwise interactions, in the usual manner" (fld no, the default). */
  postForce(): void { this.apply(); }

  setup(): void {
    if (this.fld) this.preForce();
    else this.postForce();
  }

  /** "The forces due to this fix are imposed during an energy minimization". */
  minPostForce(): void { this.apply(); }

  /** Resolved wall positions for this step: EDGE, or value * lattice spacing (units keyword). */
  private coords(): Float64Array {
    const g = this.sys.geom;
    const sp = this.sys.lattice?.spacing;
    const out = new Float64Array(this.walls.length);
    for (let k = 0; k < this.walls.length; k++) {
      const w = this.walls[k];
      out[k] = w.coord.edge
        ? (w.lo ? g.lo[w.dim] : g.hi[w.dim])
        : valueOf(this.sys, w.coord.v) * (this.unitsMode === 'lattice' ? sp?.[w.dim] ?? 1 : 1);
    }
    return out;
  }

  /** Applies every wall force, tallies energy, per-wall forces and the virial. */
  private apply(): void {
    const s = this.sys.state;
    const coord = this.coords();
    const bit = this.groupBit;
    const walls = this.walls;
    this.wallEnergy = 0;
    this.wallForce.fill(0);
    if (this.thermoVirial) this.virial.fill(0);
    if (this.vAtom.length !== 6 * s.n) this.vAtom = new Float64Array(6 * s.n);
    else this.vAtom.fill(0);
    // shifted energy at the cutoff per wall (depends on the current parameters)
    const eCut = new Float64Array(walls.length);
    for (let k = 0; k < walls.length; k++) eCut[k] = this.energyAt(walls[k].cutoff, walls[k]);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & bit)) continue;
      for (let k = 0; k < walls.length; k++) {
        const w = walls[k];
        const d = w.dim;
        const r = w.lo ? s.x[3 * i + d] - coord[k] : coord[k] - s.x[3 * i + d];
        if (r <= 0) throw new StyleError(`Particle on or inside fix ${this.id} ${this.wallStyle} surface`);
        if (r >= w.cutoff) continue;
        // F = -dE/dr, pointing away from the wall; sign flips the axis for hi walls
        const f = this.forceAt(r, w);
        const sign = w.lo ? 1 : -1;
        s.f[3 * i + d] += sign * f;
        this.wallEnergy += this.energyAt(r, w) - eCut[k];
        // force ON the wall is opposite the force on the atom
        this.wallForce[k] -= sign * f;
        if (this.thermoVirial) this.virial[d] += r * f;
        this.vAtom[6 * i + d] += r * f;
      }
    }
  }

  /** E(r) of the unshifted wall potential with the wall's current parameters. */
  protected abstract energyAt(r: number, w: Wall): number;

  /** F(r) = -dE/dr of the unshifted wall potential (positive pushes away from the wall). */
  protected abstract forceAt(r: number, w: Wall): number;

  energy(): number { return this.wallEnergy; }
  /**
   * Per-atom wall virial of the last evaluation, for compute stress/atom fix. fix_wall.html: the
   * fix_modify virial option adds the wall contribution "to both the global pressure and per-atom
   * stress of the system"; the energy option names only the global potential energy, and measured
   * with native LAMMPS (black box) compute pe/atom fix stays 0 with fix_modify energy yes walls.
   */
  private vAtom = new Float64Array(0);
  virialAtom(out: Float64Array): void {
    for (let k = 0; k < this.vAtom.length && k < out.length; k++) out[k] += this.vAtom[k];
  }
  computeScalar(): number { return this.wallEnergy; }
  computeVector(i: number): number { return this.wallForce[i]; }
}

/**
 * fix wall/lj93 — docs.lammps.org/fix_wall.html, args and formula verbatim:
 *
 *   args = coord epsilon sigma cutoff
 *     coord = position of wall = EDGE or constant or variable
 *     epsilon = strength factor for wall-particle interaction (energy or energy/distance^2 units)
 *     sigma = size factor for wall-particle interaction (distance units)
 *     cutoff = distance from wall at which wall-particle interactions are cut off (distance units)
 *
 *   E = \epsilon \left[ \frac{2}{15} \left(\frac{\sigma}{r}\right)^{9} -
 *                        \left(\frac{\sigma}{r}\right)^3 \right]
 *                        \qquad r < r_c
 *
 * "The *wall/lj93* interaction is derived by integrating over a 3d
 * half-lattice of Lennard-Jones 12/6 particles."
 */
export class FixWallLJ93 extends FixWallBase {
  readonly style = 'wall/lj93';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'wall/lj93', 4);
  }

  protected energyAt(r: number, w: Wall): number {
    const eps = valueOf(this.sys, w.params[0]);
    const sig = valueOf(this.sys, w.params[1]);
    const u = sig / r;
    return eps * ((2 / 15) * u ** 9 - u ** 3);
  }

  protected forceAt(r: number, w: Wall): number {
    const eps = valueOf(this.sys, w.params[0]);
    const sig = valueOf(this.sys, w.params[1]);
    const u = sig / r;
    return eps * (sig / (r * r)) * ((6 / 5) * u ** 8 - 3 * u ** 2);
  }
}

/**
 * fix wall/lj126 — docs.lammps.org/fix_wall.html, args and formula verbatim:
 *
 *   args = coord epsilon sigma cutoff
 *
 *   E = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
 *                         \left(\frac{\sigma}{r}\right)^6 \right]
 *                         \qquad r < r_c
 *
 * "The *wall/lj126*
 * interaction is effectively a harder, more repulsive wall interaction."
 */
export class FixWallLJ126 extends FixWallBase {
  readonly style = 'wall/lj126';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'wall/lj126', 4);
  }

  protected energyAt(r: number, w: Wall): number {
    const eps = valueOf(this.sys, w.params[0]);
    const sig = valueOf(this.sys, w.params[1]);
    const u = sig / r;
    return 4 * eps * (u ** 12 - u ** 6);
  }

  protected forceAt(r: number, w: Wall): number {
    const eps = valueOf(this.sys, w.params[0]);
    const sig = valueOf(this.sys, w.params[1]);
    const u = sig / r;
    return 4 * eps * (sig / (r * r)) * (12 * u ** 11 - 6 * u ** 5);
  }
}

/**
 * fix wall/lj1043 — docs.lammps.org/fix_wall.html, args and formula verbatim:
 *
 *   args = coord epsilon sigma cutoff
 *
 *   E = 2 \pi \epsilon \left[ \frac{2}{5} \left(\frac{\sigma}{r}\right)^{10} -
 *                        \left(\frac{\sigma}{r}\right)^4 -
 *                        \frac{\sqrt(2)\sigma^3}{3\left(r+\left(0.61/\sqrt(2)\right)\sigma\right)^3}\right]
 *                        \qquad r < r_c
 *
 * "The *wall/lj1043* interaction is yet a different form of wall
 * interaction, described in Magda et al in :ref:`(Magda) <Magda>`."
 */
export class FixWallLJ1043 extends FixWallBase {
  readonly style = 'wall/lj1043';
  /** 0.61/sqrt(2), the offset of the attractive term (docs: "0.61/\sqrt(2)"). */
  private static readonly A = 0.61 / Math.SQRT2;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'wall/lj1043', 4);
  }

  protected energyAt(r: number, w: Wall): number {
    const eps = valueOf(this.sys, w.params[0]);
    const sig = valueOf(this.sys, w.params[1]);
    const u = sig / r;
    const off = r + FixWallLJ1043.A * sig;
    return 2 * Math.PI * eps * ((2 / 5) * u ** 10 - u ** 4 - (Math.SQRT2 * sig ** 3) / (3 * off ** 3));
  }

  protected forceAt(r: number, w: Wall): number {
    const eps = valueOf(this.sys, w.params[0]);
    const sig = valueOf(this.sys, w.params[1]);
    const u = sig / r;
    const off = r + FixWallLJ1043.A * sig;
    return 2 * Math.PI * eps * ((4 * u ** 10 - 4 * u ** 4) / r - (Math.SQRT2 * sig ** 3) / off ** 4);
  }
}

/**
 * fix wall/harmonic — docs.lammps.org/fix_wall.html, args and formula verbatim:
 *
 *   args = coord epsilon sigma cutoff
 *
 *   E = \epsilon \quad (r - r_c)^2 \qquad r < r_c
 *
 * "For the *wall/harmonic* style, :math:`\epsilon` is effectively the spring
 * constant K, and has units (energy/distance\^2).  The input parameter
 * :math:`\sigma` is ignored.  The minimum energy position of the harmonic
 * spring is at the *cutoff*.  This is a repulsive-only spring since the
 * interaction is truncated at the *cutoff*". E is 0.0 at the cutoff, so no
 * shift is applied.
 */
export class FixWallHarmonic extends FixWallBase {
  readonly style = 'wall/harmonic';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'wall/harmonic', 4);
  }

  protected energyAt(r: number, w: Wall): number {
    const eps = valueOf(this.sys, w.params[0]);
    const d = r - w.cutoff;
    return eps * d * d;
  }

  protected forceAt(r: number, w: Wall): number {
    const eps = valueOf(this.sys, w.params[0]);
    return 2 * eps * (w.cutoff - r);
  }
}

/**
 * fix wall/morse — docs.lammps.org/fix_wall.html, args and formula verbatim:
 *
 *   args = coord D_0 alpha r_0 cutoff
 *     D_0 = depth of the potential (energy units)
 *     alpha = width factor for wall-particle interaction (1/distance units)
 *     r_0 = distance of the potential minimum from the face of region (distance units)
 *     cutoff = distance from wall at which wall-particle interactions are cut off (distance units)
 *
 *   E = D_0 \left[ e^{- 2 \alpha (r - r_0)} - 2 e^{- \alpha (r - r_0)} \right]
 *       \qquad r < r_c
 *
 * "For the *wall/morse* style, the three parameters are in this order:
 * :math:`D_0` the depth of the potential, :math:`\alpha` the width
 * parameter, and :math:`r_0` the location of the minimum.  :math:`D_0` has
 * energy units, :math:`\alpha` inverse distance units, and :math:`r_0`
 * distance units."
 */
export class FixWallMorse extends FixWallBase {
  readonly style = 'wall/morse';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'wall/morse', 5);
  }

  protected energyAt(r: number, w: Wall): number {
    const d0 = valueOf(this.sys, w.params[0]);
    const alpha = valueOf(this.sys, w.params[1]);
    const r0 = valueOf(this.sys, w.params[2]);
    const a = Math.exp(-alpha * (r - r0));
    return d0 * (a * a - 2 * a);
  }

  protected forceAt(r: number, w: Wall): number {
    const d0 = valueOf(this.sys, w.params[0]);
    const alpha = valueOf(this.sys, w.params[1]);
    const r0 = valueOf(this.sys, w.params[2]);
    const a = Math.exp(-alpha * (r - r0));
    return 2 * alpha * d0 * (a * a - a);
  }
}
