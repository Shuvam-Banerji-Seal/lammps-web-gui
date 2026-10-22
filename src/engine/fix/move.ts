import { Fix } from './fix';
import { StyleError } from '../force/types';
import { missingArgs, numeric } from '../force/util';
import type { System } from '../system';
import { massOf, isEllipsoid, isLineAtom, isTriAtom, LINE_BONUS, TRI_BONUS, TRI_BONUS_COLS } from '../atoms';
import { libm } from '../libm';

/*
 * fix ID group-ID move style args keyword values ... —
 * docs.lammps.org/fix_move.html, Syntax (verbatim):
 *
 *   fix ID group-ID move style args keyword values ...
 *   * style = *linear* or *wiggle* or *rotate* or *transrot* or *variable*
 *       *linear* args = Vx Vy Vz
 *         Vx,Vy,Vz = components of velocity vector (velocity units), any component can be specified as NULL
 *       *wiggle* args = Ax Ay Az period
 *         Ax,Ay,Az = components of amplitude vector (distance units), any component can be specified as NULL
 *         period = period of oscillation (time units)
 *   * zero or more keyword/value pairs may be appended
 *   * keyword = *units* or *update*
 *       *units* value = *box* or *lattice*
 *       *update* value = *dipole*
 *
 * Description (from the page, one source line per line; its quote marks around unwrapped dropped):
 *
 *   Perform updates of position and velocity for atoms in the group each
 *   timestep using the specified settings or formulas, without regard to
 *   forces on the atoms.  This can be useful for boundary or other atoms,
 *   whose movement can influence nearby atoms.
 *   As discussed below, atoms are moved relative to their initial
 *   position at the time the fix is specified.  These initial coordinates
 *   are stored by the fix in unwrapped form, by using the image flags
 *   associated with each atom.
 *   The *linear* style moves atoms at a constant velocity, so that their
 *   position *X* = (x,y,z) as a function of time is given in vector
 *   notation as
 *   X(t) = X0 + V \* delta
 *   where *X0* = (x0,y0,z0) is their position at the time the fix is
 *   specified, *V* is the specified velocity vector with components
 *   (Vx,Vy,Vz), and *delta* is the time elapsed since the fix was
 *   specified.  This style also sets the velocity of each atom to V =
 *   (Vx,Vy,Vz).  If any of the velocity components is specified as NULL,
 *   then the position and velocity of that component is time integrated
 *   the same as the :doc:`fix nve <fix_nve>` command would perform, using
 *   the corresponding force component on the atom.
 *   The *wiggle* style moves atoms in an oscillatory fashion, so that
 *   their position *X* = (x,y,z) as a function of time is given in vector
 *   notation as
 *   X(t) = X0 + A sin(omega\*delta)
 *   where *X0* = (x0,y0,z0) is their position at the time the fix is
 *   specified, *A* is the specified amplitude vector with components
 *   (Ax,Ay,Az), *omega* is 2 PI / *period*, and *delta* is the time elapsed
 *   since the fix was specified.  This style also sets the velocity of each
 *   atom to the time derivative of this expression.  If any of the amplitude
 *   components is specified as NULL, then the position and velocity of that
 *   component is time integrated the same as the :doc:`fix nve <fix_nve>`
 *   command would perform, using the corresponding force component on the
 *   atom.
 *   The *units* keyword determines the meaning of the distance units used
 *   to define the *linear* velocity and *wiggle* amplitude and *rotate*
 *   origin.  This setting is ignored for the *variable* style.  A *box*
 *   value selects standard units as defined by the :doc:`units <units>`
 *   command, e.g. velocity in Angstroms/fs and amplitude and position
 *   in Angstroms for units = real.  A *lattice* value means the velocity
 *   units are in lattice spacings per time and the amplitude and position
 *   are in lattice spacings.  The :doc:`lattice <lattice>` command must have
 *   been previously used to define the lattice spacing.  Each of these 3
 *   quantities may be dependent on the x,y,z dimension, since the lattice
 *   spacings can be different in x,y,z.
 * Default: "The option default is units = lattice."
 *
 * Measured with native LAMMPS (black box): X0 is the unwrapped position at
 * the timestep at which the fix is specified, and delta = (ntimestep -
 * that timestep) * dt, so the first initial_integrate of a run gives
 * delta = dt (a fix defined at step 2 moves by V*dt on step 3, not 2*V*dt).
 * Displacing the atoms after the fix command does not change X0 (a fix
 * specified at x = 5.0 then displaced to x = 6.0 resets x to 5.005 on the
 * first step).  linear sets v = V and x = X0 + V*delta in initial_integrate
 * (no final half-kick, so forces are ignored); a NULL component does the
 * velocity-Verlet half-kicks of fix nve in initial_integrate and
 * final_integrate (its x matches a plain fix nve under the same force
 * exactly).  wiggle sets x = X0 + A sin(omega*delta) and
 * v = A*omega*cos(omega*delta) with omega = 2 PI / period.  Atoms are
 * remapped through periodic boundaries on every step of the fix (an atom
 * crossing at high speed is wrapped on the crossing step and stays wrapped
 * on the following steps, before the next neighbor rebuild).  A *lattice*
 * value scales velocity and amplitude by the per-dimension lattice spacing
 * (lattice sc 2.0 in lj units gives spacing 2^-1/3 and vx = 0.7937005259841
 * for Vx = 1.0).
 *
 * rotate and transrot, fix_move.html: "The *rotate* style rotates atoms around a rotation axis *R* =
 * (Rx,Ry,Rz) that goes through a point *P* = (Px,Py,Pz)."; "This style also sets the velocity of each atom
 * to (omega cross Rperp) where omega is its angular velocity around the rotation axis and Rperp is a
 * perpendicular vector from the rotation axis to the atom."; "The *transrot* style combines the effects
 * of *rotate* and *linear*"; "it is not possible to set any of the translation vector components to
 * NULL." Measured with native LAMMPS (black box, transrot with spheres and dipoles, rotate with
 * ellipsoids): x = P + V delta + Rot(omega delta)(X0 - P) and v = V + omega R x Rperp to 1e-15; a sphere
 * gets the angular velocity omega R; update dipole sets mu = Rot(omega delta) mu0, mu0 being the dipole
 * when the fix is specified; an ellipsoid gets the quaternion r q0 (r the rotation by omega delta, q0 the
 * quaternion when the fix is specified) and the angular momentum A I A^T omega R, where A is the
 * orientation of the previous step (the quaternion is updated after the angular momentum); point
 * particles keep their angular momentum.
 */

type MoveStyle = 'linear' | 'wiggle' | 'rotate' | 'transrot' | 'variable';

/** The words each style takes after its name, as native counts them (narg checks in fix_move.cpp at b332f28). */
const ARGS: Record<MoveStyle, number> = { linear: 3, wiggle: 4, rotate: 7, transrot: 10, variable: 6 };

/**
 * fix_move.html: "The *variable* style allows the position and velocity components of each atom to be set by formulas
 * specified via the variable command."; "Each variable must be of either the *equal* or *atom* style." The variable of
 * a component (x, y, z, vx, vy, vz): null for NULL.
 */
type MoveVar = { name: string; atom: boolean } | null;

/** Rotation matrix of the unit quaternion q (w i j k), body to space. */
const quatMatrix = (q: ArrayLike<number>, o: number, m: number[]): void => {
  const w = q[o], x = q[o + 1], y = q[o + 2], z = q[o + 3];
  m[0] = 1 - 2 * (y * y + z * z); m[1] = 2 * (x * y - w * z); m[2] = 2 * (x * z + w * y);
  m[3] = 2 * (x * y + w * z); m[4] = 1 - 2 * (x * x + z * z); m[5] = 2 * (y * z - w * x);
  m[6] = 2 * (x * z - w * y); m[7] = 2 * (y * z + w * x); m[8] = 1 - 2 * (x * x + y * y);
};

export class FixMove extends Fix {
  readonly style = 'move';
  private readonly moveStyle: MoveStyle;
  /** Vx,Vy,Vz (linear, transrot) or Ax,Ay,Az (wiggle); null marks a NULL component. */
  private readonly comp: (number | null)[];
  /** variable: the variables of x, y, z, vx, vy, vz (null for NULL); atom-style or not is set at init. */
  private readonly vars: MoveVar[] = [];
  /** rotate/transrot: the point P (scaled) and the unit axis R. */
  private readonly point: number[] = [0, 0, 0];
  private readonly axis: number[] = [0, 0, 1];
  private readonly omega: number;
  private readonly unitsMode: 'box' | 'lattice';
  private updateDipole = false;
  /** Unwrapped positions captured when the fix was specified ("X0"). */
  private readonly xorig: Float64Array;
  private readonly captured: Uint8Array;
  /** rotate/transrot: dipoles and ellipsoid quaternions when the fix was specified. */
  private muOrig: Float64Array | null = null;
  private quatOrig: Float64Array | null = null;
  /** rotate/transrot: the orientation (theta) of the line particles when the fix was specified (fix_move.cpp toriginal). */
  private thetaOrig: Float64Array | null = null;
  /** Timestep at which the fix was specified ("the time the fix is specified"). */
  private readonly initial: number;
  private dtv = 0;
  private dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/fix_move.cpp FixMove::FixMove at b332f28): the style's words, the keywords, the 2d
    // checks, the lattice scaling and then the rotation axis
    if (args.length < 1) throw missingArgs('fix move');
    const st = args[0];
    if (!(st in ARGS)) throw new StyleError(`Unknown fix move mode ${st}`);
    this.moveStyle = st as MoveStyle;
    const na = ARGS[this.moveStyle];
    // native checks transrot against 11 words while it reads 14 (fix_move.cpp); the engine asks for all of them
    if (args.length < 1 + na) throw missingArgs(`fix move ${st}`);
    const comp = (w: string) => (w === 'NULL' ? null : numeric(w));
    this.comp = [];
    if (st === 'linear' || st === 'wiggle') for (let d = 0; d < 3; d++) this.comp.push(comp(args[1 + d]));
    if (st === 'transrot') for (let d = 0; d < 3; d++) this.comp.push(numeric(args[1 + d]));
    if (st === 'variable') {
      for (let k = 0; k < 6; k++) {
        const w = args[1 + k];
        if (w === 'NULL') this.vars.push(null);
        else if (w.startsWith('v_')) this.vars.push({ name: w.slice(2), atom: false });
        else throw new StyleError('Argument must be either NULL or variable reference');
      }
    }
    const axis = [0, 0, 0];
    if (st === 'rotate' || st === 'transrot') {
      const o = st === 'rotate' ? 1 : 4;
      for (let d = 0; d < 3; d++) this.point[d] = numeric(args[o + d]);
      for (let d = 0; d < 3; d++) axis[d] = numeric(args[o + 3 + d]);
    }
    this.omega = 0;
    if (st === 'wiggle' || st === 'rotate' || st === 'transrot') {
      const T = numeric(args[na]);
      // native names rotate for transrot too
      if (T <= 0) throw new StyleError(`Illegal fix move ${st === 'wiggle' ? 'wiggle' : 'rotate'} period ${T}`);
      this.omega = (2 * Math.PI) / T;
    }
    this.unitsMode = 'lattice';
    for (let k = 1 + na; k < args.length;) {
      const key = args[k];
      if (key === 'units') {
        if (k + 2 > args.length) throw missingArgs('fix move units');
        const val = args[k + 1];
        if (val !== 'box' && val !== 'lattice') throw new StyleError(`Unknown fix move units setting ${val}`);
        this.unitsMode = val;
        k += 2;
      } else if (key === 'update') {
        // fix_move.html: "If the *update dipole* keyword/value pair is used together with the *rotate* or
        // *transrot* style, then the orientation of the dipole moment of each particle is also updated
        // appropriately to correspond with the rotation."
        if (k + 2 > args.length) throw missingArgs('fix move update');
        // native's text has the typo "Unkown"
        if (args[k + 1] !== 'dipole') throw new StyleError(`Unkown fix move update argument ${args[k + 1]}`);
        if (st !== 'rotate' && st !== 'transrot') throw new StyleError('Fix move keyword update dipole requires style rotate or transrot');
        if (!sys.state.mu) throw new StyleError('Fix move keyword update dipole requires atom attribute mu');
        this.updateDipole = true;
        k += 2;
      } else {
        throw new StyleError(`Unknown fix move keyword ${key}`);
      }
    }
    if (sys.dimension === 2) {
      if ((st === 'linear' || st === 'transrot') && this.comp[2] !== null && this.comp[2] !== 0) throw new StyleError('Fix move cannot set linear z motion for 2d problem');
      if (st === 'wiggle' && this.comp[2] !== null && this.comp[2] !== 0) throw new StyleError('Fix move cannot set wiggle z motion for 2d problem');
      if ((st === 'rotate' || st === 'transrot') && (axis[0] !== 0 || axis[1] !== 0)) throw new StyleError('Fix move cannot rotate around non z-axis for 2d problem');
      if (st === 'variable' && (this.vars[2] || this.vars[5])) throw new StyleError('Fix move cannot define z or vz variable for 2d problem');
    }
    if (st === 'rotate' || st === 'transrot') {
      const len = Math.hypot(axis[0], axis[1], axis[2]);
      if (len === 0) throw new StyleError('Zero length rotation vector with fix move');
      for (let d = 0; d < 3; d++) this.axis[d] = axis[d] / len;
    }
    // the rotate origin is in lattice units like the velocities ("and *rotate* origin")
    for (let d = 0; d < 3; d++) this.point[d] *= this.scale(d);
    // native LAMMPS time-integrated atoms more than once warning counts fix move
    this.timeIntegrate = true;
    const s = sys.state;
    this.initial = s.step;
    const n = s.n;
    const xorig = new Float64Array(3 * n);
    this.xorig = xorig;
    this.captured = new Uint8Array(n);
    // fix_move.html: "This fix produces a per-atom array which can be accessed by various output commands. The number
    // of columns for each atom is 3, and the columns store the original unwrapped x,y,z coords of each atom."
    this.peratomFlag = true;
    this.sizePeratomCols = 3;
    this.arrayAtom = xorig;
    const bit = this.groupBit;
    const r = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & bit)) continue;
      this.captured[i] = 1;
      sys.geom.unwrap(s.x, s.image, i, r);
      this.xorig[3 * i] = r[0]; this.xorig[3 * i + 1] = r[1]; this.xorig[3 * i + 2] = r[2];
    }
    if (st === 'rotate' || st === 'transrot') {
      if (this.updateDipole && s.mu) this.muOrig = s.mu.slice(0, 4 * n);
      if (s.quat) this.quatOrig = s.quat.slice(0, 4 * n);
      const lb = s.custom.get(LINE_BONUS)?.data;
      if (lb) {
        this.thetaOrig = new Float64Array(n);
        for (let i = 0; i < n; i++) if (this.captured[i] && isLineAtom(s, i)) this.thetaOrig[i] = lb[2 * i + 1];
      }
    }
  }

  init(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    // As native LAMMPS does (FixMove::init at b332f28): each variable exists and is equal-style (equal, timer,
    // internal) or atom-style (atom, atomfile)
    for (const mv of this.vars) {
      if (!mv) continue;
      const v = this.sys.vars.get(mv.name);
      if (!v) throw new StyleError(`Variable name ${mv.name} for fix move does not exist`);
      if (v.style === 'atom' || v.style === 'atomfile') mv.atom = true;
      else if (v.style === 'equal' || v.style === 'timer' || v.style === 'internal') mv.atom = false;
      else throw new StyleError(`Variable name ${mv.name} for fix move is invalid style`);
    }
  }

  /** As native LAMMPS does (FixMove::reset_dt at b332f28): a timestep command after a run refuses fix move. */
  resetDt(): void {
    throw new StyleError('Resetting timestep size is not allowed with fix move');
  }

  /** Per-dimension multiplier of a velocity/amplitude/origin ("units box|lattice"). */
  private scale(d: number): number {
    if (this.unitsMode === 'box') return 1;
    return this.sys.lattice?.spacing[d] ?? 1;
  }

  initialIntegrate(): void {
    if (this.moveStyle === 'rotate' || this.moveStyle === 'transrot') { this.rotateStep(); return; }
    if (this.moveStyle === 'variable') { this.variableStep(); return; }
    const s = this.sys.state;
    const { x, v, f, mask, image } = s;
    const bit = this.groupBit;
    const dtf = this.dtf, dtv = this.dtv;
    const delta = (s.step - this.initial) * dtv;
    const linear = this.moveStyle === 'linear';
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(mask[i] & bit)) continue;
      const c = dtf / massOf(s, i);
      let moved = false;
      for (let d = 0; d < 3; d++) {
        const val = this.comp[d];
        if (val === null) {
          // "time integrated the same as the fix nve command would perform"
          v[3 * i + d] += c * f[3 * i + d];
          x[3 * i + d] += dtv * v[3 * i + d];
        } else if (linear) {
          moved = true;
          const sc = this.scale(d);
          x[3 * i + d] = this.xorig[3 * i + d] + sc * val * delta;
          v[3 * i + d] = sc * val;
        } else {
          moved = true;
          const sc = this.scale(d);
          const ang = this.omega * delta;
          x[3 * i + d] = this.xorig[3 * i + d] + sc * val * libm.sin(ang);
          v[3 * i + d] = sc * val * this.omega * libm.cos(ang);
        }
      }
      if (moved) {
        // measured: the fix remaps moved atoms through periodic boundaries on
        // every step; the absolute (unwrapped) position is wrapped afresh.
        for (let d = 0; d < 3; d++) if (this.comp[d] !== null) image[3 * i + d] = 0;
        this.sys.geom.remap(x, image, i);
      }
    }
  }

  /**
   * variable, fix_move.html: "The first 3 variables (v_dx,v_dy,v_dz) specified for the *variable* style are used to
   * calculate a displacement from the atom's original position at the time the fix was specified.  The second 3
   * variables (v_vx,v_vy,v_vz) specified are used to compute a velocity for each atom."; "If both the displacement and
   * velocity variables for a particular x,y,z component are specified as NULL, then the position and velocity of that
   * component is time integrated the same as the fix nve command would perform"; "If only the velocity variable for a
   * component is specified as NULL, then the displacement variable will be used to set the position of the atom, and
   * its velocity component will not be changed.  If only the displacement variable for a component is specified as
   * NULL, then the velocity variable will be used to set the velocity of the atom, and the position of the atom will be
   * time integrated using that velocity." As native LAMMPS does (FixMove::initial_integrate at b332f28): the variables
   * are evaluated in the order x, y, z, vx, vy, vz each step, and the new position is put in the periodic image nearest
   * the old one, the image flags kept (Domain::remap_near).
   */
  private variableStep(): void {
    const sys = this.sys;
    const s = sys.state;
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    this.checkNewAtoms();
    const val: (number | Float64Array | null)[] = this.vars.map((mv) => (mv === null ? null : mv.atom ? sys.atomVariable(mv.name) : sys.equalVariable(mv.name)));
    const at = (q: number | Float64Array, i: number) => (typeof q === 'number' ? q : q[i]);
    const dtv = this.dtv;
    const old = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(mask[i] & bit)) continue;
      for (let d = 0; d < 3; d++) old[d] = x[3 * i + d];
      const c = this.dtf / massOf(s, i);
      for (let d = 0; d < 3; d++) {
        const dx = val[d], vv = val[3 + d];
        if (vv !== null) v[3 * i + d] = at(vv, i);
        if (dx !== null) x[3 * i + d] = this.xorig[3 * i + d] + at(dx, i);
        else if (vv !== null) x[3 * i + d] += dtv * v[3 * i + d];
        else {
          v[3 * i + d] += c * f[3 * i + d];
          x[3 * i + d] += dtv * v[3 * i + d];
        }
      }
      this.remapNear(x, i, old);
    }
  }

  /** As native LAMMPS does (FixMove::set_arrays at b332f28): an atom of the group added after the fix refuses the variable style. */
  private checkNewAtoms(): void {
    const s = this.sys.state;
    if (s.step === this.initial) return;
    for (let i = this.captured.length; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) throw new StyleError('Cannot add atoms with fix move style variable');
    }
  }

  /**
   * The periodic image of atom i's position nearest `old` (native Domain::remap_near): along each periodic edge the
   * position moves by whole box lengths until it is within half a length of the old one, in box coordinates for an
   * orthogonal box and in fractional coordinates for a triclinic one.
   */
  private remapNear(x: Float64Array, i: number, old: number[]): void {
    const g = this.sys.geom;
    const near = (c: number[], o: number[], period: number[]) => {
      for (let d = 0; d < 3; d++) {
        if (!g.periodic[d]) continue;
        const p = period[d], half = 0.5 * p;
        if (c[d] - o[d] > p) c[d] -= Math.trunc((c[d] - o[d]) / p) * p;
        while (c[d] - o[d] > half) c[d] -= p;
        if (o[d] - c[d] > p) c[d] += Math.trunc((o[d] - c[d]) / p) * p;
        while (o[d] - c[d] > half) c[d] += p;
      }
    };
    if (!g.triclinic) {
      const c = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
      near(c, old, [g.lx, g.ly, g.lz]);
      x[3 * i] = c[0]; x[3 * i + 1] = c[1]; x[3 * i + 2] = c[2];
      return;
    }
    const lnew = [0, 0, 0], lold = [0, 0, 0];
    g.toLamda(x[3 * i], x[3 * i + 1], x[3 * i + 2], lnew);
    g.toLamda(old[0], old[1], old[2], lold);
    const before = [...lnew];
    near(lnew, lold, [1, 1, 1]);
    if (lnew[0] === before[0] && lnew[1] === before[1] && lnew[2] === before[2]) return;
    const out = [0, 0, 0];
    g.fromLamda(lnew[0], lnew[1], lnew[2], out);
    x[3 * i] = out[0]; x[3 * i + 1] = out[1]; x[3 * i + 2] = out[2];
  }

  /** rotate / transrot: positions, velocities and orientations of the rigid rotation (see the header). */
  private rotateStep(): void {
    const s = this.sys.state;
    const { x, v, mask, image } = s;
    const bit = this.groupBit;
    const delta = (s.step - this.initial) * this.dtv;
    const th = this.omega * delta;
    const c = libm.cos(th), sn = libm.sin(th);
    const R = this.axis, P = this.point, w = this.omega;
    const vt = this.moveStyle === 'transrot' ? [0, 1, 2].map((d) => this.scale(d) * this.comp[d]!) : [0, 0, 0];
    const rot = (u: number[], out: number[]): void => {
      const dot = R[0] * u[0] + R[1] * u[1] + R[2] * u[2];
      const cr = [R[1] * u[2] - R[2] * u[1], R[2] * u[0] - R[0] * u[2], R[0] * u[1] - R[1] * u[0]];
      for (let d = 0; d < 3; d++) out[d] = u[d] * c + cr[d] * sn + R[d] * dot * (1 - c);
    };
    const hr = libm.sin(th / 2);
    const r = [libm.cos(th / 2), R[0] * hr, R[1] * hr, R[2] * hr];
    const u = [0, 0, 0], xr = [0, 0, 0], A = new Array<number>(9).fill(0);
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(mask[i] & bit)) continue;
      for (let d = 0; d < 3; d++) u[d] = this.xorig[3 * i + d] - P[d];
      rot(u, xr);
      // omega cross Rperp: the axial part of xr drops out of R x xr
      const vr = [R[1] * xr[2] - R[2] * xr[1], R[2] * xr[0] - R[0] * xr[2], R[0] * xr[1] - R[1] * xr[0]];
      for (let d = 0; d < 3; d++) {
        x[3 * i + d] = P[d] + vt[d] * delta + xr[d];
        v[3 * i + d] = vt[d] + w * vr[d];
        image[3 * i + d] = 0;
      }
      this.sys.geom.remap(x, image, i);
      // fix_move.cpp: omega is set for spheres with a radius, and for lines and tris ("omega for spheres, lines, tris")
      const tri = isTriAtom(s, i);
      if (s.omega && s.radius && (s.radius[i] > 0 || isLineAtom(s, i) || tri)) for (let d = 0; d < 3; d++) s.omega[3 * i + d] = w * R[d];
      // fix_move.cpp: "theta for lines", theta_new = fmod(toriginal[i] + arg, MY_2PI)
      const lb = s.custom.get(LINE_BONUS)?.data;
      if (this.thetaOrig && lb && isLineAtom(s, i)) lb[2 * i + 1] = (this.thetaOrig[i] + th) % (2 * Math.PI);
      if (this.muOrig && s.mu) {
        for (let d = 0; d < 3; d++) u[d] = this.muOrig[4 * i + d];
        rot(u, xr);
        for (let d = 0; d < 3; d++) s.mu[4 * i + d] = xr[d];
      }
      // tris (fix_move.cpp: the tri quaternion and its angular momentum with the principal moments of the tri)
      if (this.quatOrig && s.quat && s.angmom && (tri || (s.rmass && isEllipsoid(s, i)))) {
        // angular momentum from the orientation before this step's update: A I A^T (omega R)
        quatMatrix(s.quat, 4 * i, A);
        let inertia: number[];
        if (tri) {
          const tb = s.custom.get(TRI_BONUS)!.data;
          inertia = [tb[TRI_BONUS_COLS * i + 9], tb[TRI_BONUS_COLS * i + 10], tb[TRI_BONUS_COLS * i + 11]];
        } else {
          const m = s.rmass![i];
          const a = s.shape![3 * i], b = s.shape![3 * i + 1], cc = s.shape![3 * i + 2];
          inertia = [m / 5 * (b * b + cc * cc), m / 5 * (a * a + cc * cc), m / 5 * (a * a + b * b)];
        }
        const wb = [0, 1, 2].map((k) => A[k] * w * R[0] + A[3 + k] * w * R[1] + A[6 + k] * w * R[2]);
        for (let d = 0; d < 3; d++) s.angmom[3 * i + d] = A[3 * d] * inertia[0] * wb[0] + A[3 * d + 1] * inertia[1] * wb[1] + A[3 * d + 2] * inertia[2] * wb[2];
        const q0 = this.quatOrig.subarray(4 * i, 4 * i + 4);
        const q = s.quat;
        q[4 * i] = r[0] * q0[0] - r[1] * q0[1] - r[2] * q0[2] - r[3] * q0[3];
        q[4 * i + 1] = r[0] * q0[1] + r[1] * q0[0] + r[2] * q0[3] - r[3] * q0[2];
        q[4 * i + 2] = r[0] * q0[2] - r[1] * q0[3] + r[2] * q0[0] + r[3] * q0[1];
        q[4 * i + 3] = r[0] * q0[3] + r[1] * q0[2] - r[2] * q0[1] + r[3] * q0[0];
      }
    }
  }

  finalIntegrate(): void {
    if (this.moveStyle === 'rotate' || this.moveStyle === 'transrot') return;
    // a variable-style component without displacement and velocity variables gets fix nve's second half-kick
    const free = this.moveStyle === 'variable' ? [0, 1, 2].map((d) => this.vars[d] === null && this.vars[3 + d] === null) : this.comp.map((c) => c === null);
    const s = this.sys.state;
    const { v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(mask[i] & bit)) continue;
      const c = dtf / massOf(s, i);
      for (let d = 0; d < 3; d++) if (free[d]) v[3 * i + d] += c * f[3 * i + d];
    }
  }
}
