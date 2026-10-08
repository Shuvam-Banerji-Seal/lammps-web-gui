import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

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
 */

type MoveStyle = 'linear' | 'wiggle';

export class FixMove extends Fix {
  readonly style = 'move';
  private readonly moveStyle: MoveStyle;
  /** Vx,Vy,Vz (linear) or Ax,Ay,Az (wiggle); null marks a NULL component. */
  private readonly comp: (number | null)[];
  private readonly period: number;
  private readonly omega: number;
  private readonly unitsMode: 'box' | 'lattice';
  /** Unwrapped positions captured when the fix was specified ("X0"). */
  private readonly xorig: Float64Array;
  private readonly captured: Uint8Array;
  /** Timestep at which the fix was specified ("the time the fix is specified"). */
  private readonly initial: number;
  private dtv = 0;
  private dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const st = args[0];
    if (st !== 'linear' && st !== 'wiggle') {
      throw new StyleError(`fix move style '${st ?? ''}' is not supported by the browser engine; supported: linear, wiggle (rotate, transrot and variable are not implemented)`);
    }
    this.moveStyle = st;
    const na = st === 'linear' ? 3 : 4;
    if (args.length < 1 + na) {
      const usage = st === 'linear' ? 'Vx Vy Vz' : 'Ax Ay Az period';
      throw new StyleError(`usage: fix ID group-ID move ${st} ${usage} [units box|lattice]`);
    }
    this.comp = [];
    for (let d = 0; d < 3; d++) {
      const w = args[1 + d];
      if (w === 'NULL') { this.comp.push(null); continue; }
      const v = Number(w);
      if (w === undefined || !Number.isFinite(v)) {
        throw new StyleError(`fix move ${st}: components must be numbers or NULL, got '${w ?? ''}'`);
      }
      this.comp.push(v);
    }
    if (st === 'wiggle') {
      const T = Number(args[4]);
      if (!Number.isFinite(T) || !(T > 0)) throw new StyleError(`fix move wiggle period must be > 0 (time units), got '${args[4] ?? ''}'`);
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
        // fix_move.html: the update keyword "should be used for models where a dipole moment is
        // assigned"; measured with native LAMMPS (black box): update dipole is refused with linear and wiggle.
        throw new StyleError(`fix move keyword update dipole requires style rotate or transrot`);
      } else {
        throw new StyleError(`fix move: unknown keyword '${key}'`);
      }
    }
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
  }

  init(): void { this.resetDt(); }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  /** Per-dimension multiplier of a velocity/amplitude ("units box|lattice"). */
  private scale(d: number): number {
    if (this.unitsMode === 'box') return 1;
    return this.sys.lattice?.spacing[d] ?? 1;
  }

  initialIntegrate(): void {
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

  finalIntegrate(): void {
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
