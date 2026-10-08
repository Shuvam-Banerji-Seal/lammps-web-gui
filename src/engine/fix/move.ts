import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf, isEllipsoid } from '../atoms';

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

type MoveStyle = 'linear' | 'wiggle' | 'rotate' | 'transrot';

const ARGS: Record<MoveStyle, number> = { linear: 3, wiggle: 4, rotate: 7, transrot: 10 };
const USAGE: Record<MoveStyle, string> = {
  linear: 'Vx Vy Vz', wiggle: 'Ax Ay Az period', rotate: 'Px Py Pz Rx Ry Rz period', transrot: 'Vx Vy Vz Px Py Pz Rx Ry Rz period',
};

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
  /** rotate/transrot: the point P (scaled) and the unit axis R. */
  private readonly point: number[] = [0, 0, 0];
  private readonly axis: number[] = [0, 0, 1];
  private readonly period: number;
  private readonly omega: number;
  private readonly unitsMode: 'box' | 'lattice';
  private updateDipole = false;
  /** Unwrapped positions captured when the fix was specified ("X0"). */
  private readonly xorig: Float64Array;
  private readonly captured: Uint8Array;
  /** rotate/transrot: dipoles and ellipsoid quaternions when the fix was specified. */
  private muOrig: Float64Array | null = null;
  private quatOrig: Float64Array | null = null;
  /** Timestep at which the fix was specified ("the time the fix is specified"). */
  private readonly initial: number;
  private dtv = 0;
  private dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const st = args[0];
    if (st !== 'linear' && st !== 'wiggle' && st !== 'rotate' && st !== 'transrot') {
      throw new StyleError(`fix move style '${st ?? ''}' is not supported by the browser engine; supported: linear, wiggle, rotate, transrot (variable is not implemented)`);
    }
    this.moveStyle = st;
    const na = ARGS[st];
    if (args.length < 1 + na) throw new StyleError(`usage: fix ID group-ID move ${st} ${USAGE[st]} [units box|lattice] [update dipole]`);
    const number = (w: string | undefined, what: string): number => {
      const v = Number(w);
      if (w === undefined || w === 'NULL' || !Number.isFinite(v)) throw new StyleError(`fix move ${st}: ${what} must be a number, got '${w ?? ''}'`);
      return v;
    };
    this.comp = [];
    if (st !== 'rotate') {
      for (let d = 0; d < 3; d++) {
        const w = args[1 + d];
        if (w === 'NULL' && st !== 'transrot') { this.comp.push(null); continue; }
        this.comp.push(number(w, st === 'wiggle' ? 'amplitude components' : 'velocity components'));
      }
    }
    if (st === 'rotate' || st === 'transrot') {
      const o = st === 'rotate' ? 1 : 4;
      for (let d = 0; d < 3; d++) this.point[d] = number(args[o + d], 'Px Py Pz');
      const r = [0, 1, 2].map((d) => number(args[o + 3 + d], 'Rx Ry Rz'));
      const len = Math.hypot(r[0], r[1], r[2]);
      if (!(len > 0)) throw new StyleError('fix move: zero length rotation vector');
      for (let d = 0; d < 3; d++) this.axis[d] = r[d] / len;
    }
    if (st !== 'linear') {
      const T = Number(args[na]);
      if (!Number.isFinite(T) || !(T > 0)) throw new StyleError(`fix move ${st} period must be > 0 (time units), got '${args[na] ?? ''}'`);
      this.period = T;
      this.omega = (2 * Math.PI) / T;
    } else {
      this.period = 0;
      this.omega = 0;
    }
    this.unitsMode = 'lattice';
    for (let k = 1 + na; k < args.length;) {
      const key = args[k];
      if (key === 'units') {
        const val = args[k + 1];
        if (val !== 'box' && val !== 'lattice') throw new StyleError(`fix move units must be box or lattice, got '${val ?? ''}'`);
        this.unitsMode = val;
        k += 2;
      } else if (key === 'update') {
        // fix_move.html: "If the *update dipole* keyword/value pair is used together with the *rotate* or
        // *transrot* style, then the orientation of the dipole moment of each particle is also updated
        // appropriately to correspond with the rotation."; measured with native LAMMPS (black box): update
        // dipole is refused with linear and wiggle.
        if (args[k + 1] !== 'dipole') throw new StyleError(`fix move: unknown update value '${args[k + 1] ?? ''}'`);
        if (st !== 'rotate' && st !== 'transrot') throw new StyleError('fix move keyword update dipole requires style rotate or transrot');
        if (!sys.state.mu) throw new StyleError('fix move update dipole requires atom attribute mu');
        this.updateDipole = true;
        k += 2;
      } else {
        throw new StyleError(`fix move: unknown keyword '${key}'`);
      }
    }
    // the rotate origin is in lattice units like the velocities ("and *rotate* origin")
    for (let d = 0; d < 3; d++) this.point[d] *= this.scale(d);
    // native LAMMPS time-integrated atoms more than once warning counts fix move
    this.timeIntegrate = true;
    const s = sys.state;
    this.initial = s.step;
    const n = s.n;
    this.xorig = new Float64Array(3 * n);
    this.captured = new Uint8Array(n);
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
    }
  }

  init(): void { this.resetDt(); }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  /** Per-dimension multiplier of a velocity/amplitude/origin ("units box|lattice"). */
  private scale(d: number): number {
    if (this.unitsMode === 'box') return 1;
    return this.sys.lattice?.spacing[d] ?? 1;
  }

  initialIntegrate(): void {
    if (this.moveStyle === 'rotate' || this.moveStyle === 'transrot') { this.rotateStep(); return; }
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
          x[3 * i + d] = this.xorig[3 * i + d] + sc * val * Math.sin(ang);
          v[3 * i + d] = sc * val * this.omega * Math.cos(ang);
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

  /** rotate / transrot: positions, velocities and orientations of the rigid rotation (see the header). */
  private rotateStep(): void {
    const s = this.sys.state;
    const { x, v, mask, image } = s;
    const bit = this.groupBit;
    const delta = (s.step - this.initial) * this.dtv;
    const th = this.omega * delta;
    const c = Math.cos(th), sn = Math.sin(th);
    const R = this.axis, P = this.point, w = this.omega;
    const vt = this.moveStyle === 'transrot' ? [0, 1, 2].map((d) => this.scale(d) * this.comp[d]!) : [0, 0, 0];
    const rot = (u: number[], out: number[]): void => {
      const dot = R[0] * u[0] + R[1] * u[1] + R[2] * u[2];
      const cr = [R[1] * u[2] - R[2] * u[1], R[2] * u[0] - R[0] * u[2], R[0] * u[1] - R[1] * u[0]];
      for (let d = 0; d < 3; d++) out[d] = u[d] * c + cr[d] * sn + R[d] * dot * (1 - c);
    };
    const hr = Math.sin(th / 2);
    const r = [Math.cos(th / 2), R[0] * hr, R[1] * hr, R[2] * hr];
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
      if (s.omega && s.radius) for (let d = 0; d < 3; d++) s.omega[3 * i + d] = w * R[d];
      if (this.muOrig && s.mu) {
        for (let d = 0; d < 3; d++) u[d] = this.muOrig[4 * i + d];
        rot(u, xr);
        for (let d = 0; d < 3; d++) s.mu[4 * i + d] = xr[d];
      }
      if (this.quatOrig && s.quat && s.angmom && s.rmass && isEllipsoid(s, i)) {
        // angular momentum from the orientation before this step's update: A I A^T (omega R)
        quatMatrix(s.quat, 4 * i, A);
        const m = s.rmass[i];
        const a = s.shape![3 * i], b = s.shape![3 * i + 1], cc = s.shape![3 * i + 2];
        const inertia = [m / 5 * (b * b + cc * cc), m / 5 * (a * a + cc * cc), m / 5 * (a * a + b * b)];
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
    const s = this.sys.state;
    const { v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(mask[i] & bit)) continue;
      const c = dtf / massOf(s, i);
      for (let d = 0; d < 3; d++) if (this.comp[d] === null) v[3 * i + d] += c * f[3 * i + d];
    }
  }
}
