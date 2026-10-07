import type { SimState } from './types';
import { degreesOfFreedom, kineticEnergy, temperature } from './observables';
import { Rng } from './rng';
import { halfKick, drift, rampFraction, type Fix, type RunContext } from './integrate';

/*
 * Thermostats for the in-browser MD engine, written from the DOCUMENTED
 * behaviour on docs.lammps.org and textbook stochastic/extended-system
 * dynamics (Berendsen et al., J. Chem. Phys. 81, 3684 (1984); Martyna,
 * Klein & Tuckerman, J. Chem. Phys. 97, 2635 (1992); Frenkel & Smit,
 * Understanding Molecular Simulation, 2nd ed., App. E.2). No LAMMPS source
 * code is used or referenced: LAMMPS is GPL-2.0 and this project is not.
 *
 * The target temperature follows the documented ramp
 * (docs.lammps.org/fix_modify.html / each fix page: "Tstart ... Tstop ...
 * The temperature is ramped between Tstart and Tstop over the course of the
 * run"): T_target = tStart + (tStop - tStart) * rampFraction(s, ctx).
 */

/** T_target at the current step for a fix with a Tstart..Tstop ramp. */
const targetTemperature = (tStart: number, tStop: number, s: SimState, ctx: RunContext): number =>
  tStart + (tStop - tStart) * rampFraction(s, ctx);

/**
 * Langevin thermostat (docs.lammps.org/fix_langevin.html): "Add a force to
 * the atoms in the group ... F = F_c + F_f + F_r" with friction
 * "F_f = - (m/damp) v" and a random force "proportional to
 * sqrt(kB T m / (dt damp))"; "a uniform random number is used (instead of a
 * Gaussian random number) for speed". And: "this fix does NOT perform time
 * integration. It only modifies forces."
 *
 * A uniform deviate on [-0.5, 0.5) has variance 1/12, so the amplitude
 * sqrt(24 kB T m / (dt damp)) gives the fluctuation-dissipation variance
 * 2 kB T m / (dt damp) per component, matching the friction rate 1/damp
 * (Allen & Tildesley, Computer Simulation of Liquids, eq. 6.45 style FDT:
 * <F_r(t) F_r(t')> = 2 m kT gamma delta(t-t') with gamma = 1/damp).
 */
export class FixLangevin implements Fix {
  readonly style = 'langevin';
  readonly integrates = false;
  private readonly rng: Rng;

  constructor(
    readonly id: string,
    private readonly tStart: number,
    private readonly tStop: number,
    private readonly damp: number,
    seed: number,
  ) {
    this.rng = new Rng(seed);
  }

  postForce(s: SimState, ctx: RunContext): void {
    const u = s.units;
    const T = targetTemperature(this.tStart, this.tStop, s, ctx);
    // sqrt(24 kB T / (mvv2e dt damp)) / ftm2v — per sqrt(mass); the f array
    // holds forces such that f/m*ftm2v is an acceleration (see halfKick).
    const ampBase = Math.sqrt((24 * u.boltz * T) / (u.mvv2e * s.dt * this.damp)) / u.ftm2v;
    const { f, v, type, massByType } = s;
    const rng = this.rng;
    const threeD = s.dimension === 3;
    for (let i = 0; i < s.n; i++) {
      const m = massByType[type[i]];
      const fric = -(m / this.damp) / u.ftm2v;
      const amp = ampBase * Math.sqrt(m);
      const k = 3 * i;
      f[k] += fric * v[k] + amp * (rng.uniform() - 0.5);
      f[k + 1] += fric * v[k + 1] + amp * (rng.uniform() - 0.5);
      if (threeD) f[k + 2] += fric * v[k + 2] + amp * (rng.uniform() - 0.5);
    }
  }
}

/**
 * Berendsen velocity rescaling (docs.lammps.org/fix_temp_berendsen.html):
 * "The ... fix ... rescales their velocities every timestep"; "The Tdamp
 * parameter ... determines how rapidly the temperature is relaxed"; and
 * "This thermostat will generate an error if the current temperature is
 * zero". The scale factor is Berendsen et al., J. Chem. Phys. 81, 3684
 * (1984), eq. 3.15: lambda = sqrt(1 + (dt/tau)(T_target/T - 1)).
 */
export class FixTempBerendsen implements Fix {
  readonly style = 'temp/berendsen';
  readonly integrates = false;

  constructor(
    readonly id: string,
    private readonly tStart: number,
    private readonly tStop: number,
    private readonly damp: number,
  ) {}

  endOfStep(s: SimState, ctx: RunContext): void {
    const T = temperature(s);
    if (T === 0) {
      throw new Error(`fix ${this.id} temp/berendsen cannot rescale a system at zero temperature`);
    }
    const Tt = targetTemperature(this.tStart, this.tStop, s, ctx);
    const lambda = Math.sqrt(1 + (s.dt / this.damp) * (Tt / T - 1));
    const v = s.v;
    for (let k = 0; k < 3 * s.n; k++) v[k] *= lambda;
  }
}

/**
 * Periodic velocity rescaling (docs.lammps.org/fix_temp_rescale.html):
 * "Rescaling is performed every N timesteps"; "Rescaling is only performed
 * if the difference between the current and desired temperatures is greater
 * than the window value. The amount of rescaling that is applied is a
 * fraction (from 0.0 to 1.0) of the difference between the actual and
 * desired temperature. E.g. if fraction = 1.0, the temperature is reset to
 * exactly the desired value."
 */
export class FixTempRescale implements Fix {
  readonly style = 'temp/rescale';
  readonly integrates = false;

  constructor(
    readonly id: string,
    private readonly every: number,
    private readonly tStart: number,
    private readonly tStop: number,
    private readonly window: number,
    private readonly fraction: number,
  ) {}

  endOfStep(s: SimState, ctx: RunContext): void {
    if (this.every <= 0 || s.step % this.every !== 0) return;
    const T = temperature(s);
    const Tt = targetTemperature(this.tStart, this.tStop, s, ctx);
    if (Math.abs(T - Tt) <= this.window) return;
    if (T === 0) {
      throw new Error(`fix ${this.id} temp/rescale cannot rescale a system at zero temperature`);
    }
    const tNew = T - this.fraction * (T - Tt);
    const lambda = Math.sqrt(tNew / T);
    const v = s.v;
    for (let k = 0; k < 3 * s.n; k++) v[k] *= lambda;
  }
}

/**
 * Nose-Hoover-chain NVT integration (docs.lammps.org/fix_nh.html): "These
 * commands perform time integration on Nose-Hoover style non-Hamiltonian
 * equations of motion"; defaults "tchain = 3 ... tloop = 1"; "A good choice
 * for many models is a Tdamp of around 100 timesteps". The integrator is the
 * Martyna-Klein-Tuckerman explicit factorization (J. Chem. Phys. 97, 2635
 * (1992); Frenkel & Smit, 2nd ed., App. E.2; Tuckerman, Statistical
 * Mechanics: Theory and Molecular Simulation, ch. 4): each step is
 * chain(dt/2) · half-kick · drift · half-kick · chain(dt/2), where the
 * chain half-update sweeps the thermostat momenta top-down, scales the
 * particle velocities by exp(-xi1_dot dt/2), advances the chain positions
 * and sweeps back up the chain.
 *
 * Thermostat masses: Q1 = dof kB T damp^2, Qk = kB T damp^2 (k > 1). Chain
 * variables xi_k and their velocities vxi_k start at rest. The documented
 * conserved quantity is KE + PE + sum_k Qk vxi_k^2 / 2 + dof kB T xi_1 +
 * kB T sum_{k>1} xi_k.
 */
const NVT_CHAIN = 3; // "tchain = 3" default

export class FixNvt implements Fix {
  readonly style = 'nvt';
  readonly integrates = true;
  private readonly xi = new Float64Array(NVT_CHAIN);
  private readonly vxi = new Float64Array(NVT_CHAIN);
  private lastCtx: RunContext | null = null;

  constructor(
    readonly id: string,
    private readonly tStart: number,
    private readonly tStop: number,
    private readonly damp: number,
  ) {}

  initialIntegrate(s: SimState, ctx: RunContext): void {
    this.lastCtx = ctx;
    this.chainHalfStep(s, ctx);
    halfKick(s);
    drift(s);
  }

  finalIntegrate(s: SimState, ctx: RunContext): void {
    this.lastCtx = ctx;
    halfKick(s);
    this.chainHalfStep(s, ctx);
  }

  /** The conserved extended-system energy (energy units) at the given PE. */
  conservedEnergy(s: SimState, pe: number): number {
    const kB = s.units.boltz;
    const dof = degreesOfFreedom(s);
    const ctx = this.lastCtx ?? { runStart: s.step, runStop: s.step };
    const T = targetTemperature(this.tStart, this.tStop, s, ctx);
    const tau2 = this.damp * this.damp;
    let e = kineticEnergy(s) + pe + 0.5 * dof * kB * T * tau2 * this.vxi[0] ** 2 + dof * kB * T * this.xi[0];
    for (let k = 1; k < NVT_CHAIN; k++) {
      e += 0.5 * kB * T * tau2 * this.vxi[k] ** 2 + kB * T * this.xi[k];
    }
    return e;
  }

  /**
   * Half-step chain update over dt/2: sweep vxi top-down (exact solve of
   * dvxi_k/dt = G_k - vxi_{k+1} vxi_k with the exp factors), scale particle
   * velocities by exp(-vxi_1 dt/2), advance xi by vxi dt/2, sweep vxi back
   * up the chain. G_1 = (2 KE - dof kB T)/Q_1 from the instantaneous KE;
   * G_k = (Q_{k-1} vxi_{k-1}^2 - kB T)/Q_k.
   */
  private chainHalfStep(s: SimState, ctx: RunContext): void {
    const dof = degreesOfFreedom(s);
    if (dof <= 0 || s.n === 0) return;
    const kB = s.units.boltz;
    const T = targetTemperature(this.tStart, this.tStop, s, ctx);
    const tau = 0.5 * s.dt;   // the half step
    const h4 = 0.5 * tau;     // dt/4 per sweep
    const h8 = 0.25 * tau;
    const tau2 = this.damp * this.damp;
    const q1 = dof * kB * T * tau2;
    const qk = kB * T * tau2;
    const qOf = (k: number) => (k === 1 ? q1 : qk);
    const { xi, vxi } = this;

    // Sweep 1: top of the chain down (vxi_{k+1} already updated here).
    for (let k = NVT_CHAIN; k >= 1; k--) {
      const g = k === 1
        ? (2 * kineticEnergy(s) - dof * kB * T) / q1
        : (qOf(k - 1) * vxi[k - 2] * vxi[k - 2] - kB * T) / qOf(k);
      const f = k < NVT_CHAIN ? vxi[k] : 0;
      vxi[k - 1] = vxi[k - 1] * Math.exp(-f * h4) + g * h4 * Math.exp(-f * h8);
    }
    const scale = Math.exp(-vxi[0] * tau);
    const v = s.v;
    for (let j = 0; j < 3 * s.n; j++) v[j] *= scale;
    for (let k = 0; k < NVT_CHAIN; k++) xi[k] += vxi[k] * tau;

    // Sweep 2: back up the chain (vxi_{k-1} already updated here).
    for (let k = 1; k <= NVT_CHAIN; k++) {
      const g = k === 1
        ? (2 * kineticEnergy(s) - dof * kB * T) / q1
        : (qOf(k - 1) * vxi[k - 2] * vxi[k - 2] - kB * T) / qOf(k);
      const f = k < NVT_CHAIN ? vxi[k] : 0;
      vxi[k - 1] = vxi[k - 1] * Math.exp(-f * h4) + g * h4 * Math.exp(-f * h8);
    }
  }
}
