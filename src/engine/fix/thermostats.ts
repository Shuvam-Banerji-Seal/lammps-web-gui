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
  /** target temperature of this step */
  tt: number;
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
  private exchanged = 0;
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
    if (this.omegaOn) {
      if (!s.omega || !s.torque || !s.radius) throw new StyleError('fix langevin omega yes needs finite-size spheres (atom_style sphere)');
      if (this.tally || this.zero) throw new StyleError('fix langevin omega yes with tally or zero is not supported by the browser engine yet');
    }
    this.scalarFlag = this.tally;
    this.extscalar = 1;
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

  init(): void { this.checkAngmom(); }

  // Measured with native LAMMPS (black box): run 0 already dumps the random
  // forces, so the setup force evaluation includes this fix.
  setup(): void { this.postForce(); }

  /**
   * The current target temperature (native fix langevin's t_target). fix charge/regulation reads it through its
   * tempfixid keyword: "Using *tempfixid* overrides any fixed temperature set by *temp*" (docs.lammps.org/
   * fix_charge_regulation.html).
   */
  targetTemperature(): number {
    return typeof this.tStart === 'number' ? ramp(this.sys, this.tStart, this.tStop) : valueOf(this.sys, this.tStart);
  }

  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const u = s.units;
    const tt = this.targetTemperature();
    if (tt < 0) throw new StyleError(`fix ${this.id} langevin: target temperature is negative`);
    const bias = this.temp?.hasBias() ?? false;
    if (bias) { this.temp!.computeBias(); this.temp!.removeBiasAll(); }
    const { f, v, type, mask } = s;
    if (this.tally && this.fl.length !== 3 * s.n) this.fl = new Float64Array(3 * s.n);
    let sx = 0, sy = 0, sz = 0, count = 0;
    const rand = this.zero ? new Float64Array(3 * s.n) : null;
    // the draws follow native LAMMPS's atom list (SimState.order, see System.sortAtoms)
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      // "scale ... factor by which to scale the damping coefficient": a ratio of 2
      // doubles damp (measured: native random amplitude follows damp * ratio)
      const damp = this.damp * this.ratio[type[i]];
      const g1 = -(m / damp) / u.ftm2v;
      const g2 = Math.sqrt(m) * Math.sqrt((24 * u.boltz * tt) / (u.mvv2e * s.dt * damp)) / u.ftm2v;
      // three draws per atom in native storage order, also in 2d (measured: native keeps fz)
      const rx = g2 * (this.rng.uniform() - 0.5), ry = g2 * (this.rng.uniform() - 0.5), rz = g2 * (this.rng.uniform() - 0.5);
      const fx = g1 * v[3 * i] + rx, fy = g1 * v[3 * i + 1] + ry, fz = g1 * v[3 * i + 2] + rz;
      if (rand) { rand[3 * i] = rx; rand[3 * i + 1] = ry; rand[3 * i + 2] = rz; sx += rx; sy += ry; sz += rz; count++; }
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      if (this.tally) { this.fl[3 * i] = fx; this.fl[3 * i + 1] = fy; this.fl[3 * i + 2] = fz; }
    }
    if (rand && count > 0) {
      // "the total random force is set exactly to zero by subtracting off an equal part of it from each atom"
      sx /= count; sy /= count; sz /= count;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        f[3 * i] -= sx; f[3 * i + 1] -= sy; f[3 * i + 2] -= sz;
        if (this.tally) { this.fl[3 * i] -= sx; this.fl[3 * i + 1] -= sy; this.fl[3 * i + 2] -= sz; }
      }
    }
    if (this.omegaOn) this.rotationalTorques(tt);
    if (this.ascale !== 0) {
      gplHook<LangevinAngmomHook>('angmom').angmomTorques(s, {
        groupBit: this.groupBit, damp: this.damp, ratio: this.ratio, ascale: this.ascale, tt,
        uniform: () => this.rng.uniform(),
      });
    }
    if (bias) this.temp!.restoreBiasAll();
  }

  /**
   * fix langevin omega: torque = -(I s / damp) omega + random, with I = 0.4 m r^2 and the scale factor s = 10/3 for
   * spheres. The random amplitude carries sqrt(s). The rotational draws follow all the translational draws, one
   * uniform deviate per component and atom, in native order (measured with native LAMMPS, black box).
   */
  private rotationalTorques(tt: number): void {
    const s = this.sys.state;
    const u = s.units;
    const { torque, omega, mask, type, radius } = s;
    const sc = 10 / 3;
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      const r = (radius as Float64Array)[i];
      const inertia = 0.4 * massOf(s, i) * r * r;
      const damp = this.damp * this.ratio[type[i]];
      const g1 = -(inertia * sc / damp) / u.ftm2v;
      const g2 = Math.sqrt(inertia) * Math.sqrt((24 * u.boltz * tt * sc) / (u.mvv2e * s.dt * damp)) / u.ftm2v;
      for (let d = 0; d < 3; d++) {
        torque![3 * i + d] += g1 * omega![3 * i + d] + g2 * (this.rng.uniform() - 0.5);
      }
    }
  }

  /** Energy added by the Langevin forces over the step (tally yes): -sum F_L . v dt. */
  endOfStep(): void {
    if (!this.tally) return;
    const s = this.sys.state;
    let w = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      w += this.fl[3 * i] * s.v[3 * i] + this.fl[3 * i + 1] * s.v[3 * i + 1] + this.fl[3 * i + 2] * s.v[3 * i + 2];
    }
    // force x distance is energy in every unit style
    this.exchanged -= w * s.dt;
  }

  ecouple(): number { return this.tally ? this.exchanged : 0; }
  computeScalar(): number {
    if (!this.tally) throw new StyleError(`fix ${this.id} langevin computes a scalar only with tally yes`);
    return this.exchanged;
  }
}
