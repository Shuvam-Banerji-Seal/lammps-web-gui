import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { SimState } from '../types';
import { massOf } from '../atoms';

/*
 * Motion-style fixes written only from the LAMMPS documentation (the
 * docs.lammps.org page is cited per class and the implemented lines are
 * quoted character for character from plans/lammps-docs/<page>.rst).
 * Velocities are stored in LAMMPS velocity units, forces in internal units
 * (F_internal = F_lammps / ftm2v), so velocity updates use dt * ftm2v.
 */

/**
 * fix ID group-ID momentum N keyword values ... —
 * docs.lammps.org/fix_momentum.html, Syntax (verbatim):
 *
 *   fix ID group-ID momentum N keyword values ...
 *   N = adjust the momentum every this many timesteps
 *   *linear* values = xflag yflag zflag
 *     xflag,yflag,zflag = 0/1 to exclude/include each dimension
 *   *angular* values = none
 *   *rescale* values = none
 *
 * Description (verbatim, one source line per line):
 *
 *   One (or both) of the *linear* or *angular* keywords **must** be specified.
 *   If the *linear* keyword is used, the linear momentum is zeroed by
 *   subtracting the center-of-mass velocity of the group or chunk from each
 *   atom.  This does not change the relative velocity of any pair of atoms.
 *   If the *angular* keyword is used, the angular momentum is zeroed by
 *   subtracting a rotational component from each atom.
 *   The *rescale* keyword enables conserving the kinetic energy of the group
 *   or chunk of atoms by rescaling the velocities after the momentum was
 *   removed.
 *
 * The angular removal subtracts omega x (r_i - r_com) with omega = I^-1 L,
 * where L and the inertia tensor I are taken about the center of mass of the
 * group and r are the unwrapped coordinates; this zeroes the group's angular
 * momentum exactly. The rescale conserves the raw kinetic energy of the
 * group, 1/2 sum m v^2, measured before the removal and restored after
 * (measured on native LAMMPS with a zero-force 3-atom system and on the
 * w2fmom_momentum oracle configuration: applying [linear 1 1 0 + angular +
 * rescale] to the reconstructed pre-fix velocities of oracle step 5
 * reproduces the post-fix velocities to 1e-15 with the raw-KE basis, while
 * a COM-subtracted KE basis misses by 3e-2).
 * The fix runs at end_of_step. "None of the fix_modify options are
 * relevant to this fix.  No global or per-atom quantities are stored by
 * this fix for access by various output commands."
 * "This fix is not invoked during energy minimization." Default: "none".
 */
export class FixMomentum extends Fix {
  readonly style = 'momentum';
  private readonly linearFlags: boolean[] | null;
  private readonly angular: boolean;
  private readonly rescale: boolean;
  /** Unwrapped positions of the group, scratch for the angular removal. */
  private xu = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const n = Number(args[0]);
    if (!Number.isInteger(n) || n < 1) throw new StyleError(`fix momentum: N must be a positive integer (adjust the momentum every this many timesteps), got '${args[0] ?? ''}'`);
    this.nevery = n;
    let linear: boolean[] | null = null;
    let angular = false;
    let rescale = false;
    for (let k = 1; k < args.length;) {
      const key = args[k];
      if (key === 'linear') {
        const flags: boolean[] = [];
        for (let d = 0; d < 3; d++) {
          const w = args[k + 1 + d];
          if (w !== '0' && w !== '1') throw new StyleError(`fix momentum linear: xflag,yflag,zflag must be 0/1 to exclude/include each dimension, got '${w ?? ''}'`);
          flags.push(w === '1');
        }
        linear = flags;
        k += 4;
      } else if (key === 'angular') {
        angular = true;
        k += 1;
      } else if (key === 'rescale') {
        rescale = true;
        k += 1;
      } else {
        throw new StyleError(`fix momentum: unknown keyword '${key}' (keyword = *linear* or *angular* or *rescale*)`);
      }
    }
    if (!linear && !angular) {
      throw new StyleError('fix momentum: one (or both) of the *linear* or *angular* keywords must be specified');
    }
    this.linearFlags = linear;
    this.angular = angular;
    this.rescale = rescale;
  }

  endOfStep(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const bit = this.groupBit;
    // group mass, total momentum and center-of-mass velocity
    let msum = 0;
    const p = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const m = massOf(s, i);
      msum += m;
      p[0] += m * v[3 * i]; p[1] += m * v[3 * i + 1]; p[2] += m * v[3 * i + 2];
    }
    if (!(msum > 0)) return;
    const vcom = [p[0] / msum, p[1] / msum, p[2] / msum];
    // kinetic energy before the removal (raw 1/2 sum m v^2)
    let keBefore = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const m = massOf(s, i);
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      keBefore += 0.5 * m * (vx * vx + vy * vy + vz * vz);
    }
    if (this.linearFlags) {
      const f = this.linearFlags;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit)) continue;
        for (let d = 0; d < 3; d++) if (f[d]) v[3 * i + d] -= vcom[d];
      }
    }
    if (this.angular) this.removeAngular(s);
    if (this.rescale) {
      // kinetic energy after the momentum was removed, then rescale the
      // velocities so the group's kinetic energy is conserved
      let keAfter = 0;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit)) continue;
        const m = massOf(s, i);
        const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
        keAfter += 0.5 * m * (vx * vx + vy * vy + vz * vz);
      }
      if (keAfter > 0 && keBefore > 0) {
        const fac = Math.sqrt(keBefore / keAfter);
        for (let i = 0; i < s.n; i++) {
          if (!(mask[i] & bit)) continue;
          v[3 * i] *= fac; v[3 * i + 1] *= fac; v[3 * i + 2] *= fac;
        }
      }
    }
  }

  /**
   * Zeroes the angular momentum about the group's center of mass:
   * L = sum m (r - rcom) x v, I = sum m (|d|^2 E - d d^T), omega = I^-1 L,
   * v_i -= omega x (r_i - rcom); then L_after = L - I omega = 0.
   */
  private removeAngular(s: SimState): void {
    const { x, v, mask, image } = s;
    const bit = this.groupBit;
    const g = this.sys.geom;
    const n = s.n;
    if (this.xu.length < 3 * n) this.xu = new Float64Array(3 * n);
    const xu = this.xu;
    const r = [0, 0, 0];
    let msum = 0;
    const rc = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & bit)) continue;
      g.unwrap(x, image, i, r);
      xu[3 * i] = r[0]; xu[3 * i + 1] = r[1]; xu[3 * i + 2] = r[2];
      const m = massOf(s, i);
      msum += m;
      rc[0] += m * r[0]; rc[1] += m * r[1]; rc[2] += m * r[2];
    }
    if (!(msum > 0)) return;
    rc[0] /= msum; rc[1] /= msum; rc[2] /= msum;
    // angular momentum and inertia tensor about the COM
    const L = [0, 0, 0];
    const I = [0, 0, 0, 0, 0, 0]; // xx xy xz yy yz zz
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & bit)) continue;
      const m = massOf(s, i);
      const dx = xu[3 * i] - rc[0], dy = xu[3 * i + 1] - rc[1], dz = xu[3 * i + 2] - rc[2];
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      L[0] += m * (dy * vz - dz * vy);
      L[1] += m * (dz * vx - dx * vz);
      L[2] += m * (dx * vy - dy * vx);
      I[0] += m * (dy * dy + dz * dz);
      I[1] -= m * dx * dy;
      I[2] -= m * dx * dz;
      I[3] += m * (dx * dx + dz * dz);
      I[4] -= m * dy * dz;
      I[5] += m * (dx * dx + dy * dy);
    }
    // solve I omega = L (symmetric 3x3, Cramer)
    const [a, b, c, d, e, f] = I;
    const det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    if (det === 0 || !Number.isFinite(det)) return; // degenerate (collinear) group: no rotation to remove
    const w = [
      ((d * f - e * e) * L[0] + (c * e - b * f) * L[1] + (b * e - c * d) * L[2]) / det,
      ((c * e - b * f) * L[0] + (a * f - c * c) * L[1] + (b * c - a * e) * L[2]) / det,
      ((b * e - c * d) * L[0] + (b * c - a * e) * L[1] + (a * d - b * b) * L[2]) / det,
    ];
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & bit)) continue;
      const dx = xu[3 * i] - rc[0], dy = xu[3 * i + 1] - rc[1], dz = xu[3 * i + 2] - rc[2];
      v[3 * i] -= w[1] * dz - w[2] * dy;
      v[3 * i + 1] -= w[2] * dx - w[0] * dz;
      v[3 * i + 2] -= w[0] * dy - w[1] * dx;
    }
  }
}

/** One x/y/z setting of fix recenter: a number, INIT or NULL. */
type RecenterCoord = { kind: 'num'; value: number } | { kind: 'init' } | { kind: 'null' };

/**
 * fix ID group-ID recenter x y z keyword value ... —
 * docs.lammps.org/fix_recenter.html, Syntax (verbatim):
 *
 *   fix ID group-ID recenter x y z keyword value ...
 *   x,y,z = constrain center-of-mass to these coords (distance units),         any coord can also be NULL or INIT (see below)
 *   *shift* value = group-ID
 *     group-ID = group of atoms whose coords are shifted
 *   *units* value = *box* or *lattice* or *fraction*
 *
 * Description (verbatim, one source line per line):
 *
 *   Constrain the center-of-mass position of a group of atoms by adjusting
 *   the coordinates of the atoms every timestep.
 *   also be specified as NULL, which means exclude that dimension from
 *   this operation.  Or it can be specified as INIT which means to
 *   constrain the center-of-mass to its initial value at the beginning of
 *   the run.
 *   The center-of-mass (COM) is computed for the group specified by the
 *   fix.  If the current COM is different than the specified x,y,z, then a
 *   group of atoms has their coordinates shifted by the difference.  By
 *   default the shifted group is also the group specified by the fix.  A
 *   different group can be shifted by using the *shift* keyword.
 *   If the *units* keyword is set to *box*, then the distance units of
 *   previously used to define the lattice spacing.  A *fraction* value
 *   means a fractional distance between the lo/hi box boundaries, e.g. 0.5
 *   = middle of the box.  The default is to use lattice units.
 *   This fix performs its operations at the same point in the
 *   timestep as other time integration fixes, such as :doc:`fix nve <fix_nve>`, :doc:`fix nvt <fix_nh>`, or :doc:`fix npt <fix_nh>`.
 *   Thus fix recenter should normally be the last such fix specified in
 *   the input script, since the adjustments it makes to atom coordinates
 *   should come after the changes made by time integration.  LAMMPS will
 *   warn you if your fixes are not ordered this way.
 *
 * So the shift runs in initial_integrate (after any integrator defined
 * earlier); the initial COM for INIT is captured in setup() and no shift is
 * applied there (the w2fmom_recenter oracle prints f_2 = 0 on step 0, i.e.
 * the large first shift happens on the first timestep, as the restriction
 * "This fix should not be used with an x,y,z setting that causes a large
 * shift in the system on the first timestep" also implies). The COM is
 * computed from unwrapped coordinates (with wrapped coordinates the
 * per-step displacement would jump whenever an atom crosses a periodic
 * boundary, which the w2fmom_recenter oracle displacement vector rules
 * out).
 *
 * Output (verbatim, one source line per line):
 *
 *   :doc:`output commands <Howto_output>`.  The scalar is the distance the
 *   group is moved by fix recenter.
 *   various :doc:`output commands <Howto_output>`.  The 3 quantities in the
 *   vector are xyz components of displacement applied to the group of
 *   atoms by the fix.
 *   The scalar and vector values calculated by this fix are "extensive".
 *
 * Default: "The option defaults are shift = fix group-ID, and units = lattice."
 */
export class FixRecenter extends Fix {
  readonly style = 'recenter';
  private readonly spec: RecenterCoord[];
  private readonly unitsMode: 'box' | 'lattice' | 'fraction';
  private readonly shiftBit: number;
  /** Target COM in box (distance) units; INIT entries resolved in setup(). */
  private readonly target = [0, 0, 0];
  /** Displacement applied by the most recent shift. */
  private readonly disp = [0, 0, 0];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('usage: fix ID group-ID recenter x y z keyword value ...');
    this.spec = args.slice(0, 3).map((w, d) => {
      if (w === 'NULL') return { kind: 'null' as const };
      if (w === 'INIT') return { kind: 'init' as const };
      const v = Number(w);
      if (!Number.isFinite(v)) throw new StyleError(`fix recenter: x/y/z (${['x', 'y', 'z'][d]}) must be a coordinate, NULL or INIT, got '${w}'`);
      return { kind: 'num' as const, value: v };
    });
    this.unitsMode = 'lattice';
    this.shiftBit = this.groupBit;
    for (let k = 3; k < args.length;) {
      const key = args[k];
      const val = args[k + 1];
      if (key === 'shift') {
        if (!val) throw new StyleError('fix recenter shift needs a group-ID (group of atoms whose coords are shifted)');
        this.shiftBit = sys.groups.bit(val);
        k += 2;
      } else if (key === 'units') {
        if (val !== 'box' && val !== 'lattice' && val !== 'fraction') {
          throw new StyleError(`fix recenter units must be box or lattice or fraction, got '${val ?? ''}'`);
        }
        this.unitsMode = val;
        k += 2;
      } else {
        throw new StyleError(`fix recenter: unknown keyword '${key}' (keyword = *shift* or *units*)`);
      }
    }
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
  }

  init(): void {
    // "LAMMPS will warn you if your fixes are not ordered this way" (recenter
    // should come after the time integration fixes it follows in the timestep)
    const me = this.sys.fixes.indexOf(this);
    for (let k = me + 1; k < this.sys.fixes.length; k++) {
      if (this.sys.fixes[k].timeIntegrate) {
        this.sys.warn(`fix recenter ${this.id} should normally be the last time integration fix defined`);
        break;
      }
    }
  }

  /** Box-units coordinate of one dimension's setting (INIT resolved by setup). */
  private targetNow(d: number): number | null {
    const sp = this.spec[d];
    if (sp.kind === 'null') return null;
    if (sp.kind === 'init') return this.target[d];
    if (this.unitsMode === 'box') return sp.value;
    if (this.unitsMode === 'lattice') return sp.value * (this.sys.lattice?.spacing[d] ?? 1);
    const b = this.sys.state.box;
    return b.lo[d] + sp.value * (b.hi[d] - b.lo[d]);
  }

  /** Unwrapped center of mass of the fix group; null for an empty group. */
  private com(): number[] | null {
    const s = this.sys.state;
    const g = this.sys.geom;
    const r = [0, 0, 0];
    let msum = 0;
    const c = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const m = massOf(s, i);
      msum += m;
      g.unwrap(s.x, s.image, i, r);
      c[0] += m * r[0]; c[1] += m * r[1]; c[2] += m * r[2];
    }
    if (!(msum > 0)) return null;
    return [c[0] / msum, c[1] / msum, c[2] / msum];
  }

  setup(): void {
    // capture the INIT targets ("its initial value at the beginning of the
    // run"); no shift is applied during setup
    const c = this.com();
    for (let d = 0; d < 3; d++) {
      const sp = this.spec[d];
      this.target[d] = sp.kind === 'init' && c ? c[d] : 0;
      this.disp[d] = 0;
    }
  }

  /** Shift the atoms of the shift group so the COM sits on the target. */
  private doShift(): void {
    const c = this.com();
    if (!c) { this.disp[0] = this.disp[1] = this.disp[2] = 0; return; }
    const s = this.sys.state;
    for (let d = 0; d < 3; d++) {
      const t = this.targetNow(d);
      this.disp[d] = t === null ? 0 : t - c[d];
    }
    const { x } = s;
    const bit = this.shiftBit;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & bit)) continue;
      x[3 * i] += this.disp[0];
      x[3 * i + 1] += this.disp[1];
      x[3 * i + 2] += this.disp[2];
    }
  }

  initialIntegrate(): void { this.doShift(); }

  /** "The scalar is the distance the group is moved by fix recenter." */
  computeScalar(): number {
    const [dx, dy, dz] = this.disp;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /** "xyz components of displacement applied to the group of atoms". */
  computeVector(i: number): number { return this.disp[i]; }
}

/**
 * fix ID group-ID nve/limit xmax — docs.lammps.org/fix_nve_limit.html,
 * Syntax (verbatim):
 *
 *   fix ID group-ID nve/limit xmax
 *   xmax = maximum distance an atom can move in one timestep (distance units)
 *
 * Description (verbatim, one source line per line):
 *
 *   Perform constant NVE updates of position and velocity for atoms in the
 *   group each timestep.  A limit is imposed on the maximum distance an
 *   atom can move in one timestep.
 *   distance > 0.0).  But large velocities generated by large forces are
 *   reset to a value that corresponds to a displacement of length *xmax*
 *   in a single timestep.
 *
 * Output (verbatim, one source line per line):
 *
 *   :doc:`output commands <Howto_output>`.  The scalar is the count of how
 *   many updates of atom's velocity/position were limited by the maximum
 *   distance criterion.  This should be roughly the number of atoms so
 *   affected, except that updates occur at both the beginning and end of a
 *   timestep in a velocity Verlet timestepping algorithm.  This is a
 *   cumulative quantity for the current run, but is re-initialized to zero
 *   each time a run is performed.  The scalar value calculated by this fix
 *   is "extensive".
 *
 * So both the initial and the final half-step of velocity Verlet are limited
 * (and counted) exactly like fix nve. Default: "none".
 */
export class FixNVELimit extends Fix {
  readonly style = 'nve/limit';
  private xmax: number;
  private dtv = 0;
  private dtf = 0;
  private vmax = 0;
  private count = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('usage: fix ID group-ID nve/limit xmax');
    const v = Number(args[0]);
    if (!Number.isFinite(v) || !(v > 0)) throw new StyleError(`fix nve/limit: xmax must be a positive distance, got '${args[0]}'`);
    this.xmax = v;
    this.timeIntegrate = true;
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  init(): void {
    // "re-initialized to zero each time a run is performed"
    this.count = 0;
    this.resetDt();
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    this.vmax = this.xmax / s.dt;
  }

  /** Limits the speed so dt*|v| <= xmax ("a displacement of length xmax"). */
  private limit(s: SimState): void {
    const { v, mask } = s;
    const bit = this.groupBit;
    const vmax = this.vmax;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      const vsq = vx * vx + vy * vy + vz * vz;
      if (vsq > vmax * vmax) {
        const c = vmax / Math.sqrt(vsq);
        v[3 * i] = c * vx; v[3 * i + 1] = c * vy; v[3 * i + 2] = c * vz;
        this.count++;
      }
    }
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf, dtv = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const c = dtf / massOf(s, i);
      v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
    }
    this.limit(s);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      x[3 * i] += dtv * v[3 * i]; x[3 * i + 1] += dtv * v[3 * i + 1]; x[3 * i + 2] += dtv * v[3 * i + 2];
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const { v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const c = dtf / massOf(s, i);
      v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
    }
    this.limit(s);
  }

  computeScalar(): number { return this.count; }
}

/**
 * fix ID group-ID nve/noforce — docs.lammps.org/fix_nve_noforce.html.
 * The syntax block on the page reads "fix ID group-ID nve" (the page is
 * for the nve/noforce style); no arguments are documented. Description
 * (verbatim, one source line per line):
 *
 *   Perform updates of position, but not velocity for atoms in the group
 *   each timestep.  In other words, the force on the atoms is ignored and
 *   their velocity is not updated.  The atom velocities are used to update
 *   their positions.
 *
 * No global or per-atom outputs; not invoked during minimization.
 * Default: "none".
 */
export class FixNVENoforce extends Fix {
  readonly style = 'nve/noforce';
  private dtv = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('fix nve/noforce takes no arguments');
    this.timeIntegrate = true;
  }

  init(): void { this.resetDt(); }

  resetDt(): void { this.dtv = this.sys.state.dt; }

  initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    const bit = this.groupBit;
    const dtv = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      x[3 * i] += dtv * v[3 * i]; x[3 * i + 1] += dtv * v[3 * i + 1]; x[3 * i + 2] += dtv * v[3 * i + 2];
    }
  }
}
