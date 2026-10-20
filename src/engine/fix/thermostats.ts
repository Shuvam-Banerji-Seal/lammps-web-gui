import { Fix } from './fix';
import { takeFixRecord } from './restartRecord';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Compute } from '../compute/compute';
import { RanMars } from '../rng';
import { ownCompute, parseNumOrVar, ramp, removeCompute, valueOf, type NumOrVar } from './util';
import { isEllipsoid, massOf, nativeOrder } from '../atoms';
import { gplHook } from '../gplHooks';
import { SUPER_INERTIA } from '../superellipsoid';

/*
 * Velocity thermostats that do not integrate (use them with fix nve):
 * langevin, temp/berendsen, temp/rescale. (temp/csvr and temp/csld are in src/engine-gpl/thermostat.) Written from
 * the cited papers and the documented behaviour, not from LAMMPS code.
 *
 * Shared documented rules: "The desired temperature at each timestep is a
 * ramped value during the run from Tstart to Tstop." "Tstart can be specified
 * as an equal-style variable. In this case, the Tstop setting is ignored."
 * "The translational degrees of freedom can also have a bias velocity
 * removed from them before thermostatting takes place" (fix_modify temp with
 * a biased temperature compute). temp/berendsen, temp/rescale and
 * temp/csvr/csld "create their own compute of style temp, as if this command
 * had been issued: compute fix-ID_temp group-ID temp". "The cumulative energy
 * change in the system imposed by this fix is included in the thermodynamic
 * output keywords ecouple and econserve"; "This fix computes a global scalar
 * ... the same cumulative energy change ... The scalar value calculated by
 * this fix is "extensive"."
 */

/** Target temperature now: a variable, or the Tstart..Tstop ramp. */
const target = (sys: System, tStart: NumOrVar, tStop: number): number =>
  (typeof tStart === 'number' ? ramp(sys, tStart, tStop) : valueOf(sys, tStart));

export abstract class RescaleFix extends Fix {
  protected tStart: NumOrVar = 0;
  protected tStop = 0;
  protected tempId: string;
  protected temp!: Compute;
  protected exchanged = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.scalarFlag = true;
    this.extscalar = 1;
    this.tempId = `${id}_temp`;
    this.temp = ownCompute(sys, this.tempId, group, 'temp', []);
  }

  init(): void { this.temp = this.sys.compute(this.tempId); }
  destroy(): void { if (this.tempId === `${this.id}_temp`) removeCompute(this.sys, this.tempId); }

  /** extract("t_target") (fix temp/berendsen, temp/rescale, temp/csvr: the target temperature of this step; compute hma reads it). */
  extract(name: string): number | null {
    return name === 't_target' ? target(this.sys, this.tStart, this.tStop) : null;
  }

  modify(key: string, values: string[]): number {
    if (key === 'temp') {
      const c = this.sys.compute(values[0] ?? '');
      if (!c.tempFlag) throw new StyleError(`fix_modify temp: compute ${values[0]} does not compute a temperature`);
      if (this.tempId === `${this.id}_temp`) removeCompute(this.sys, this.tempId);
      this.tempId = values[0];
      this.temp = c;
      return 1;
    }
    return super.modify(key, values);
  }

  /** Current temperature (fresh), with the bias compute's own bias handling. */
  protected current(): number {
    this.sys.refreshComputes();
    return this.temp.scalarValue();
  }

  /** v *= lambda for the group (bias removed first when the compute has one); tallies the energy change. */
  protected rescale(lambda: number, tcur: number): void {
    const s = this.sys.state;
    const c = this.temp;
    const bias = c.hasBias();
    if (bias) { c.computeBias(); c.removeBiasAll(); }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      s.v[3 * i] *= lambda; s.v[3 * i + 1] *= lambda; s.v[3 * i + 2] *= lambda;
    }
    if (bias) c.restoreBiasAll();
    // energy removed from the system = KE_before - KE_after = (1 - lambda^2) * dof kB T / 2
    this.exchanged += (1 - lambda * lambda) * 0.5 * c.dof * s.units.boltz * tcur;
    this.sys.refreshComputes();
  }

  ecouple(): number { return this.exchanged; }
  computeScalar(): number { return this.exchanged; }
}

/**
 * fix ID group temp/berendsen Tstart Tstop Tdamp — fix_temp_berendsen.html:
 * "rescales their velocities every timestep"; "This thermostat will generate
 * an error if the current temperature is zero at the end of a timestep."
 * Berendsen et al., J Chem Phys 81, 3684 (1984): lambda = sqrt(1 + (dt/Tdamp)
 * (T_target/T - 1)).
 */
/**
 * fix_temp_berendsen.cpp write_restart: "energy" (the cumulative energy change), read back by restart() when the fix
 * is re-specified (fix_temp_rescale.cpp keeps the same single value).
 */
export interface EnergyRestart { energy: number }

/** The energy record of a fix re-specified after read_restart (see EnergyRestart). */
const energyRestart = (sys: System, id: string, style: string): number | undefined =>
  (takeFixRecord(sys, id, style)?.state as EnergyRestart | undefined)?.energy;

export class FixTempBerendsen extends RescaleFix {
  readonly style = 'temp/berendsen';
  private damp: number;

  /** fix_temp_berendsen.cpp write_restart: the cumulative energy change. */
  restartState(): EnergyRestart { return { energy: this.exchanged }; }

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('usage: fix ID group temp/berendsen Tstart Tstop Tdamp');
    const energy = energyRestart(sys, id, this.style);
    if (energy !== undefined) this.exchanged = energy;
    this.tStart = parseNumOrVar(args[0], 'Tstart');
    this.tStop = Number(args[1]);
    this.damp = Number(args[2]);
    if (!Number.isFinite(this.tStop) || !(this.damp > 0)) throw new StyleError('fix temp/berendsen: Tstop must be a number and Tdamp > 0');
  }

  endOfStep(): void {
    const t = this.current();
    // Measured with native LAMMPS (black box): with fewer than one degree of freedom in the temperature group
    // (a one-atom group) the fix returns without rescaling and without an error, whatever the temperature.
    if (this.temp.dof < 1) return;
    if (t === 0) throw new StyleError(`fix ${this.id} temp/berendsen: cannot rescale a zero temperature`);
    const tt = target(this.sys, this.tStart, this.tStop);
    const lambda = Math.sqrt(1 + (this.sys.state.dt / this.damp) * (tt / t - 1));
    this.rescale(lambda, t);
  }
}

/**
 * fix ID group temp/rescale N Tstart Tstop window fraction —
 * fix_temp_rescale.html: "Rescaling is performed every N timesteps";
 * "Rescaling is only performed if the difference between the current and
 * desired temperatures is greater than the window value. The amount of
 * rescaling that is applied is a fraction (from 0.0 to 1.0) of the difference
 * between the actual and desired temperature."
 */
export class FixTempRescale extends RescaleFix {
  readonly style = 'temp/rescale';
  private window: number;
  private fraction: number;

  /** fix_temp_rescale.cpp write_restart: the cumulative energy change. */
  restartState(): EnergyRestart { return { energy: this.exchanged }; }

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 5) throw new StyleError('usage: fix ID group temp/rescale N Tstart Tstop window fraction');
    const energy = energyRestart(sys, id, this.style);
    if (energy !== undefined) this.exchanged = energy;
    this.nevery = Number(args[0]);
    if (!Number.isInteger(this.nevery) || this.nevery < 1) throw new StyleError('fix temp/rescale: N must be a positive integer');
    this.tStart = parseNumOrVar(args[1], 'Tstart');
    this.tStop = Number(args[2]);
    this.window = Number(args[3]);
    this.fraction = Number(args[4]);
    if (![this.tStop, this.window, this.fraction].every(Number.isFinite)) throw new StyleError('fix temp/rescale: Tstop, window and fraction must be numbers');
  }

  endOfStep(): void {
    const t = this.current();
    const tt = target(this.sys, this.tStart, this.tStop);
    // Measured with native LAMMPS (black box): fewer than one degree of freedom in the group means no rescaling.
    if (this.temp.dof < 1) return;
    if (Math.abs(t - tt) <= this.window) return;
    if (t === 0) throw new StyleError(`fix ${this.id} temp/rescale: cannot rescale a zero temperature`);
    const tnew = t - this.fraction * (t - tt);
    this.rescale(Math.sqrt(tnew / t), t);
  }
}

/** The inputs of the angmom torques (the routine is in src/engine-gpl/thermostat/langevin_angmom.ts). */
export interface LangevinAngmomOptions {
  groupBit: number;
  /** the fix's damp (before the per-type ratio) */
  damp: number;
  /** per-type ratio of the damping coefficient */
  ratio: Float64Array;
  /** keyword angmom factor (nonzero here) */
  ascale: number;
  /** sqrt of the target temperature of this step (a constant or equal-style target) */
  tsqrt: number;
  /** the per-atom target temperatures of an atom-style variable (then tsqrt is unused), or null */
  tforce: Float64Array | null;
  /** the fix's random draws, one per call, in native order */
  uniform: () => number;
}

/** The helper of the lazy HOOKS entry 'angmom' (src/engine-gpl/registry/langevin_angmom.ts). */
export interface LangevinAngmomHook {
  angmomTorques(s: System['state'], o: LangevinAngmomOptions): void;
}

/**
 * fix ID group langevin Tstart Tstop damp seed [angmom omega scale tally zero] —
 * fix_langevin.html: "F_f = - (m/damp) v"; "F_r is proportional to
 * sqrt(k_B T m / (dt damp))"; "a uniform random number is used (instead of a
 * Gaussian random number) for speed"; "this fix does NOT perform time
 * integration. It only modifies forces"; "The keyword scale allows the damp
 * factor to be scaled up or down by the specified factor for atoms of that
 * type" (damp_type = damp * ratio, measured with native LAMMPS); "The keyword tally enables the
 * calculation of the cumulative energy added/subtracted to the atoms";
 * "If the keyword zero is set to yes, the total random force is set exactly
 * to zero by subtracting off an equal part of it from each atom in the
 * group." Defaults "angmom = no, omega = no, scale = 1.0 for all types,
 * tally = no, zero = no". "This fix is not invoked during energy
 * minimization."
 * A uniform deviate on [-0.5, 0.5) has variance 1/12, so the amplitude
 * sqrt(24 k_B T m / (dt damp)) gives the fluctuation-dissipation variance
 * 2 k_B T m / (dt damp) per component.
 */
export class FixLangevin extends Fix {
  readonly style = 'langevin';
  private tStart: NumOrVar;
  private tStop: number;
  private damp: number;
  private rng: RanMars;
  private ratio: Float64Array;
  private tally = false;
  private zero = false;
  /** keyword omega yes: rotational thermostat of finite-size spheres (torque from the angular velocity). */
  private omegaOn = false;
  /** keyword angmom factor (0 = off): rotational thermostat of ellipsoid angular momentum. */
  private ascale = 0;
  /** Tstart is an atom-style variable: one target temperature per atom (native tstyle ATOM), set at init. */
  private tAtom = false;
  /**
   * The tally energy as native LAMMPS keeps it (src/fix_langevin.cpp at b332f28, end_of_step and compute_scalar):
   * energy adds the power sum F_L . v times dt at the end of every step; a scalar read on the run's first step resets
   * it to half the first term, and the scalar is -(energy - energyOneStep dt / 2), a trapezoidal sum.
   */
  private tallyEnergy = 0;
  private energyOneStep = 0;
  /** The Langevin forces have been computed once (native flangevin_allocated); the scalar is 0 before. */
  private flSet = false;
  private fl = new Float64Array(0);
  private temp: Compute | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('usage: fix ID group langevin Tstart Tstop damp seed [keywords]');
    this.tStart = parseNumOrVar(args[0], 'Tstart');
    this.tStop = Number(args[1]);
    this.damp = Number(args[2]);
    const seed = Number(args[3]);
    if (!Number.isFinite(this.tStop) || !(this.damp > 0)) throw new StyleError('fix langevin: Tstop must be a number and damp > 0');
    if (!Number.isInteger(seed) || seed <= 0) throw new StyleError('fix langevin: seed must be a positive integer');
    this.rng = new RanMars(seed);
    // Measured with native LAMMPS (black box): the stream starts one draw in,
    // so the first draw is discarded (the draw is taken here, at fix creation).
    this.rng.uniform();
    const s = sys.state;
    this.ratio = new Float64Array(s.ntypes + 1).fill(1);
    for (let k = 4; k < args.length;) {
      const key = args[k];
      switch (key) {
        case 'scale': {
          const t = Number(args[k + 1]), r = Number(args[k + 2]);
          if (!Number.isInteger(t) || t < 1 || t > s.ntypes || !(r > 0)) throw new StyleError('fix langevin scale needs a type and a ratio > 0');
          this.ratio[t] = r;
          k += 3;
          break;
        }
        case 'tally': case 'zero': {
          if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError(`fix langevin ${key} must be yes or no`);
          if (key === 'tally') this.tally = args[k + 1] === 'yes'; else this.zero = args[k + 1] === 'yes';
          k += 2;
          break;
        }
        case 'omega':
          if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError('fix langevin omega must be yes or no');
          this.omegaOn = args[k + 1] === 'yes';
          k += 2;
          break;
        case 'angmom': {
          // As native LAMMPS does (src/fix_langevin.cpp at b332f28): "no" gives ascale 0 and any other value is a
          // numeric factor; a factor of 0 leaves the rotational thermostat off.
          const v = args[k + 1];
          const a = v === 'no' ? 0 : Number(v);
          if (!Number.isFinite(a) || a < 0) throw new StyleError(`fix langevin angmom must be 'no' or a factor >= 0 (got '${v}')`);
          this.ascale = a;
          k += 2;
          break;
        }
        case 'gjf': throw new StyleError('fix langevin gjf was removed from LAMMPS (22Jul2025); use fix gjf');
        default: throw new StyleError(`unknown fix langevin keyword '${key}'`);
      }
    }
    this.checkAngmom();
    this.checkOmega();
    // As native LAMMPS does (src/fix_langevin.cpp at b332f28): the fix always has a global scalar (scalar_flag = 1),
    // which is 0 without tally.
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  /**
   * The checks of keyword omega (src/fix_langevin.cpp init at b332f28: "Fix langevin omega requires atom attribute
   * omega", "... requires atom attribute radius" and, for a group atom of radius 0, "Fix langevin omega requires extended
   * particles"). Run when the fix is built and again at each run's init.
   */
  private checkOmega(): void {
    if (!this.omegaOn) return;
    const s = this.sys.state;
    if (!s.omega || !s.torque) throw new StyleError('Fix langevin omega requires atom attribute omega');
    if (!s.radius) throw new StyleError('Fix langevin omega requires atom attribute radius');
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && s.radius[i] === 0.0) throw new StyleError('Fix langevin omega requires extended particles');
    }
  }

  /**
   * The style of a v_name target temperature (src/fix_langevin.cpp init at b332f28): an equal-style variable (equal,
   * internal or timer) gives one temperature, an atom-style one (atom or atomfile) a temperature per atom; any other
   * style is "Variable X for fix langevin is invalid style".
   */
  private checkVariable(): void {
    if (typeof this.tStart === 'number') { this.tAtom = false; return; }
    const name = this.tStart.variable;
    const v = this.sys.vars.get(name);
    if (!v) throw new StyleError(`Variable name ${name} for fix langevin does not exist`);
    if (v.style === 'atom' || v.style === 'atomfile') this.tAtom = true;
    else if (v.style === 'equal' || v.style === 'internal' || v.style === 'timer') this.tAtom = false;
    else throw new StyleError(`Variable ${name} for fix langevin is invalid style`);
  }

  modify(key: string, values: string[]): number {
    if (key === 'temp') {
      const c = this.sys.compute(values[0] ?? '');
      if (!c.tempFlag) throw new StyleError(`fix_modify temp: compute ${values[0]} does not compute a temperature`);
      this.temp = c;
      return 1;
    }
    return super.modify(key, values);
  }

  /**
   * The ellipsoid checks of keyword angmom (fix_langevin.cpp: "Fix langevin angmom requires atom style ellipsoid" and
   * "requires extended particles"). Run when the fix is built and again at each run's init, so atoms added later are checked.
   */
  private checkAngmom(): void {
    if (this.ascale === 0) return;
    const s = this.sys.state;
    if (!s.quat || !s.angmom || !s.shape || !s.torque) throw new StyleError('Fix langevin angmom requires atom style ellipsoid');
    if (s.custom.has(SUPER_INERTIA)) throw new StyleError('Fix langevin does not support superellipsoids');
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !isEllipsoid(s, i)) throw new StyleError('Fix langevin angmom requires extended particles');
    }
  }

  init(): void { this.checkVariable(); this.checkAngmom(); this.checkOmega(); }

  // Measured with native LAMMPS (black box): run 0 already dumps the random
  // forces, so the setup force evaluation includes this fix.
  setup(): void { this.postForce(); }

  /**
   * The current target temperature (native fix langevin's t_target). fix charge/regulation reads it through its
   * tempfixid keyword: "Using *tempfixid* overrides any fixed temperature set by *temp*" (docs.lammps.org/
   * fix_charge_regulation.html).
   */
  /** extract("t_target") (fix_langevin.cpp extract; compute hma reads it). */
  extract(name: string): number | null {
    return name === 't_target' ? this.targetTemperature() : null;
  }

  targetTemperature(): number {
    if (this.tAtom) throw new StyleError(`fix ${this.id} langevin: an atom-style temperature variable gives no single target temperature`);
    return typeof this.tStart === 'number' ? ramp(this.sys, this.tStart, this.tStop) : valueOf(this.sys, this.tStart);
  }

  /**
   * reset_target (temper: src/fix_langevin.cpp FixLangevin::reset_target at b332f28): "t_target = t_start = t_stop = t_new".
   * A variable target temperature is refused: the native command would take it from the variable again on the next step.
   */
  resetTarget(tNew: number): void {
    if (typeof this.tStart !== 'number') throw new StyleError(`fix ${this.id} langevin: a tempering reset of a variable target temperature is not supported`);
    this.tStart = tNew; this.tStop = tNew;
  }

  /**
   * The Langevin forces, written as native LAMMPS computes them (src/fix_langevin.cpp post_force_templated at
   * b332f28): gamma1 = -m / damp / ftm2v and gamma2 = sqrt(m) sqrt(24 kB / damp / dt / mvv2e) / ftm2v, then gamma1 times
   * 1 / ratio and gamma2 times sqrt(T) / sqrt(ratio) (the per-type factors of a per-type mass are formed in the same
   * order: gfactor1, gfactor2 in init). With a bias compute, a velocity component that is zero once the bias is
   * removed gets no random force. The draws follow native's atom list (SimState.order, see System.sortAtoms).
   */
  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const u = s.units;
    // compute_target: one sqrt(T) for a constant or equal-style target, one T per atom for an atom-style variable
    let tsqrt = 0;
    let tforce: Float64Array | null = null;
    if (this.tAtom) {
      tforce = sys.atomVariable((this.tStart as { variable: string }).variable);
      for (let i = 0; i < s.n; i++) {
        if ((s.mask[i] & this.groupBit) && tforce[i] < 0.0) throw new StyleError('Fix langevin variable returned negative temperature');
      }
    } else {
      const tt = this.targetTemperature();
      if (tt < 0) {
        throw new StyleError(typeof this.tStart === 'number' ? `fix ${this.id} langevin: target temperature is negative` : 'Fix langevin variable returned negative temperature');
      }
      tsqrt = Math.sqrt(tt);
    }
    const zero = this.zero;
    let count = 0;
    if (zero) {
      for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) count++;
      if (count === 0) throw new StyleError('Cannot zero Langevin force of 0 atoms');
    }
    const bias = this.temp?.hasBias() ?? false;
    if (bias) { this.temp!.computeBias(); this.temp!.removeBiasAll(); }
    const { f, v, type, mask, rmass } = s;
    if (this.tally) {
      if (this.fl.length !== 3 * s.n) this.fl = new Float64Array(3 * s.n);
      this.flSet = true;
    }
    const damp = this.damp, ftm2v = u.ftm2v;
    const c24 = Math.sqrt(24.0 * u.boltz / damp / s.dt / u.mvv2e);
    let sx = 0, sy = 0, sz = 0;
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      const ts = tforce ? Math.sqrt(tforce[i]) : tsqrt;
      const ratio = this.ratio[type[i]];
      let g1: number, g2: number;
      if (rmass) {
        g1 = -rmass[i] / damp / ftm2v;
        g2 = Math.sqrt(rmass[i]) * c24 / ftm2v;
        g1 *= 1.0 / ratio;
        g2 *= 1.0 / Math.sqrt(ratio) * ts;
      } else {
        const m = s.massByType[type[i]];
        g1 = -m / damp / ftm2v;
        g2 = Math.sqrt(m) / ftm2v;
        g2 *= c24;
        g1 *= 1.0 / ratio;
        g2 *= 1.0 / Math.sqrt(ratio);
        g2 *= ts;
      }
      // three draws per atom, also in 2d (native keeps the z component)
      let rx = g2 * (this.rng.uniform() - 0.5), ry = g2 * (this.rng.uniform() - 0.5), rz = g2 * (this.rng.uniform() - 0.5);
      if (bias) {
        if (v[3 * i] === 0.0) rx = 0.0;
        if (v[3 * i + 1] === 0.0) ry = 0.0;
        if (v[3 * i + 2] === 0.0) rz = 0.0;
      }
      const fx = g1 * v[3 * i] + rx, fy = g1 * v[3 * i + 1] + ry, fz = g1 * v[3 * i + 2] + rz;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      if (zero) { sx += rx; sy += ry; sz += rz; }
      if (this.tally) { this.fl[3 * i] = fx; this.fl[3 * i + 1] = fy; this.fl[3 * i + 2] = fz; }
    }
    if (zero) {
      // "the total random force is set exactly to zero by subtracting off an equal part of it from each atom"
      sx /= count; sy /= count; sz /= count;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        f[3 * i] -= sx; f[3 * i + 1] -= sy; f[3 * i + 2] -= sz;
        if (this.tally) { this.fl[3 * i] -= sx; this.fl[3 * i + 1] -= sy; this.fl[3 * i + 2] -= sz; }
      }
    }
    if (bias) this.temp!.restoreBiasAll();
    // the rotational thermostats come after the translational forces (and the zero correction), omega before angmom
    if (this.omegaOn) this.rotationalTorques(tsqrt, tforce);
    if (this.ascale !== 0) {
      gplHook<LangevinAngmomHook>('angmom').angmomTorques(s, {
        groupBit: this.groupBit, damp: this.damp, ratio: this.ratio, ascale: this.ascale, tsqrt, tforce,
        uniform: () => this.rng.uniform(),
      });
    }
  }

  /**
   * fix langevin omega: torque = -(I s / damp) omega + random, with I = 0.4 m r^2 and the factor s = 10/3 for spheres;
   * the random amplitude carries sqrt(s) (sqrt(80) = sqrt(24 s)). As native LAMMPS does (src/fix_langevin.cpp
   * omega_thermostat at b332f28): gamma1 = -(10/3) I / damp / ftm2v and gamma2 = sqrt(I) sqrt(80 kB / damp / dt / mvv2e)
   * / ftm2v, then times 1 / ratio and sqrt(T) / sqrt(ratio). One draw per component and atom, after all the
   * translational draws, in native order.
   */
  private rotationalTorques(tsqrt: number, tforce: Float64Array | null): void {
    const s = this.sys.state;
    const u = s.units;
    const { torque, omega, mask, type, radius } = s;
    const tendivthree = 10.0 / 3.0;
    const c80 = Math.sqrt(80.0 * u.boltz / this.damp / s.dt / u.mvv2e);
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit) || !(radius![i] > 0.0)) continue;
      const r = radius![i];
      const inertia = 0.4 * r * r * massOf(s, i);
      const ts = tforce ? Math.sqrt(tforce[i]) : tsqrt;
      const ratio = this.ratio[type[i]];
      let g1 = -tendivthree * inertia / this.damp / u.ftm2v;
      let g2 = Math.sqrt(inertia) * c80 / u.ftm2v;
      g1 *= 1.0 / ratio;
      g2 *= 1.0 / Math.sqrt(ratio) * ts;
      const t0 = g2 * (this.rng.uniform() - 0.5), t1 = g2 * (this.rng.uniform() - 0.5), t2 = g2 * (this.rng.uniform() - 0.5);
      torque![3 * i] += g1 * omega![3 * i] + t0;
      torque![3 * i + 1] += g1 * omega![3 * i + 1] + t1;
      torque![3 * i + 2] += g1 * omega![3 * i + 2] + t2;
    }
  }

  /** The power sum of the Langevin forces, sum F_L . v over the group, in native atom order (tally yes). */
  private power(): number {
    const s = this.sys.state;
    let e = 0;
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      e += this.fl[3 * i] * s.v[3 * i] + this.fl[3 * i + 1] * s.v[3 * i + 1] + this.fl[3 * i + 2] * s.v[3 * i + 2];
    }
    return e;
  }

  /** tally yes: the step's power sum is added to the energy (end_of_step). */
  endOfStep(): void {
    if (!this.tally) return;
    this.energyOneStep = this.power();
    this.tallyEnergy += this.energyOneStep * this.sys.state.dt;
  }

  /** The thermostat's energy (thermo ecouple): the scalar, as native modify->energy_couple() sums it. */
  ecouple(): number { return this.computeScalar(); }

  /**
   * As native LAMMPS does (src/fix_langevin.cpp compute_scalar at b332f28): 0 without tally or before the forces were
   * computed; on the run's first step the energy restarts from half that step's power times dt; the value is
   * -(energy - power dt / 2).
   */
  computeScalar(): number {
    if (!this.tally || !this.flSet) return 0.0;
    const s = this.sys.state;
    if (s.step === this.sys.run.beginStep) {
      this.energyOneStep = this.power();
      this.tallyEnergy = 0.5 * this.energyOneStep * s.dt;
    }
    return -(this.tallyEnergy - 0.5 * this.energyOneStep * s.dt);
  }
}
