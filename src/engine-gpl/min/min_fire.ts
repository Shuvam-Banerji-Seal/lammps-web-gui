/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/min_fire.cpp (MinFire::init,
 * setup_style and run_iterate for the eulerimplicit, verlet, leapfrog and eulerexplicit integrators, with and without
 * abcfire) and the stopping tests of src/min.cpp (fnorm_sqr, fnorm_inf, fnorm_max) at commit b332f28. Authors named in
 * the source: Julien Guénolé (CNRS), Erik Bitzek (FAU Erlangen-Nuernberg), abcfire by Sebastien Echeverri Restrepo
 * (SKF, King's College London) and Predrag Andric (SKF); LAMMPS is Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * One damped-dynamics minimizer over one or more replicas: with several (neb, update->multireplica), v.f, v.v and f.f
 * are summed over all replicas, the displacement limit is the smallest over them, and the energy and force tests stop
 * only when every replica passes.
 */
import type { SimState } from '../../engine/types';
import { libm } from '../../engine/libm';

/** One replica the minimizer moves. */
export interface FireReplica {
  readonly s: SimState;
  mass(i: number): number;
}

export type FireIntegrator = 'eulerimplicit' | 'verlet' | 'leapfrog' | 'eulerexplicit';

/** The min_modify settings of fire (min.cpp defaults: delaystep 20, dtgrow 1.1, ...). */
export interface FireParams {
  dmax: number;
  delaystep: number;
  dtgrow: number;
  dtshrink: number;
  alpha0: number;
  alphashrink: number;
  tmax: number;
  tmin: number;
  integrator: FireIntegrator;
  halfstepback: boolean;
  /** initialdelay (delaystep_start_flag). */
  initialdelay: boolean;
  /** vdfmax (max_vdotf_negatif). */
  vdfmax: number;
  abcfire: boolean;
  etol: number;
  ftol: number;
  normstyle: 'two' | 'max' | 'inf';
}

export type FireStop = 'max iterations' | 'energy tolerance' | 'force tolerance' | 'max iterations with v.f negative' | 'cancelled';

/** What the iteration loop asks of its caller. */
export interface FireHooks {
  /** ++update->ntimestep of every replica; the new step. */
  advance(): number;
  /** After the tests of one iteration (thermo, dump, halt, yielding). */
  afterIteration(step: number): Promise<void>;
  cancelled(): boolean;
}

const EPS_ENERGY = 1.0e-8;

export class MinFire {
  dt = 0;
  private dtmax = 0;
  private dtmin = 0;
  private alpha = 0;
  private lastNegative = 0;
  private ntimestepStart = 0;
  private vdotfNegatif = 0;
  private flagv0 = 0;
  ecurrent = 0;
  eprevious = 0;
  niter = 0;
  neval = 0;

  /**
   * evaluateAll: energy_force(0) of every replica together (the replicas of neb evaluate in lockstep, and fix neb's
   * nudging reads all of them); the energy of each.
   */
  constructor(readonly p: FireParams, readonly reps: readonly FireReplica[], private readonly evaluateAll: () => number[], readonly multireplica: boolean) {}

  /** MinFire::init: the step limits from the timestep, alpha, and the step counters. */
  init(dt: number, ntimestep: number): void {
    const p = this.p;
    if (p.tmax < p.tmin) throw new Error('tmax has to be larger than tmin');
    if (p.dtgrow < 1.0) throw new Error('dtgrow has to be larger than 1.0');
    if (p.dtshrink > 1.0) throw new Error('dtshrink has to be smaller than 1.0');
    this.dt = dt;
    this.dtmax = p.tmax * dt;
    this.dtmin = p.tmin * dt;
    this.alpha = p.alpha0;
    this.lastNegative = this.ntimestepStart = ntimestep;
    this.vdotfNegatif = 0;
    this.niter = 0;
    this.neval = 0;
  }

  /** MinFire::setup_style: v = 0. */
  setupStyle(): void {
    for (const r of this.reps) r.s.v.fill(0, 0, 3 * r.s.n);
    this.flagv0 = 1;
  }

  private sum(fn: (s: SimState) => number): number {
    // single replica: the world sum; several: summed again over the universe (the replicas)
    let t = 0;
    for (const r of this.reps) t += fn(r.s);
    return t;
  }

  /** fnorm_sqr / fnorm_max / fnorm_inf of one replica (squared, compared with ftol^2). */
  private fnorm(s: SimState): number {
    const f = s.f;
    const n3 = 3 * s.n;
    if (this.p.normstyle === 'two') {
      let t = 0;
      for (let k = 0; k < n3; k++) t += f[k] * f[k];
      return t;
    }
    if (this.p.normstyle === 'inf') {
      let m = 0;
      for (let k = 0; k < n3; k++) m = Math.max(f[k] * f[k], m);
      return m;
    }
    let m = 0;
    for (let k = 0; k < n3; k += 3) m = Math.max(f[k] * f[k] + f[k + 1] * f[k + 1] + f[k + 2] * f[k + 2], m);
    return m;
  }

  /** run_iterate<INTEGRATOR, ABCFLAG>(maxiter). */
  async iterate(maxiter: number, hooks: FireHooks, ftm2v: number): Promise<FireStop> {
    const p = this.p;
    const reps = this.reps;
    const abc = p.abcfire;
    const integ = p.integrator;
    const kick = (r: FireReplica, c: number, set: boolean) => {
      const s = r.s;
      for (let i = 0; i < s.n; i++) {
        const dtfm = c / r.mass(i);
        if (set) {
          s.v[3 * i] = dtfm * s.f[3 * i]; s.v[3 * i + 1] = dtfm * s.f[3 * i + 1]; s.v[3 * i + 2] = dtfm * s.f[3 * i + 2];
        } else {
          s.v[3 * i] += dtfm * s.f[3 * i]; s.v[3 * i + 1] += dtfm * s.f[3 * i + 1]; s.v[3 * i + 2] += dtfm * s.f[3 * i + 2];
        }
      }
    };
    // Leap Frog integration initialization (at every call of iterate)
    if (integ === 'leapfrog') {
      this.evaluateAll();
      this.neval++;
      const dtf = -0.5 * this.dt * ftm2v;
      for (const r of reps) kick(r, dtf, true);
    }

    for (let iter = 0; iter < maxiter; iter++) {
      if (hooks.cancelled()) return 'cancelled';
      const ntimestep = hooks.advance();
      this.niter++;

      const vdotfall = this.sum((s) => {
        let t = 0;
        for (let k = 0; k < 3 * s.n; k++) t += s.v[k] * s.f[k];
        return t;
      });
      let scale1 = 0, scale2 = 0;
      if (vdotfall > 0.0) {
        const vdotvall = this.sum((s) => {
          let t = 0;
          for (let k = 0; k < 3 * s.n; k++) t += s.v[k] * s.v[k];
          return t;
        });
        this.vdotfNegatif = 0;
        const fdotfall = this.sum((s) => {
          let t = 0;
          for (let k = 0; k < 3 * s.n; k++) t += s.f[k] * s.f[k];
          return t;
        });
        if (abc) {
          // limit the value of alpha to avoid divergence of abcfire
          if (this.alpha < 1e-10) this.alpha = 1e-10;
          const abcf = 1.0 - libm.pow(1.0 - this.alpha, ntimestep - this.lastNegative);
          scale1 = (1.0 - this.alpha) / abcf;
          scale2 = fdotfall <= 1e-20 ? 0.0 : (this.alpha * Math.sqrt(vdotvall / fdotfall)) / abcf;
        } else {
          scale1 = 1.0 - this.alpha;
          scale2 = fdotfall <= 1e-20 ? 0.0 : this.alpha * Math.sqrt(vdotvall / fdotfall);
        }
        if (ntimestep - this.lastNegative > p.delaystep) {
          this.dt = Math.min(this.dt * p.dtgrow, this.dtmax);
          for (const r of reps) r.s.dt = this.dt;
          this.alpha *= p.alphashrink;
        }
      } else {
        this.lastNegative = ntimestep;
        let delayflag = true;
        if (ntimestep - this.ntimestepStart < p.delaystep && p.initialdelay) delayflag = false;
        if (delayflag) {
          this.alpha = p.alpha0;
          if (this.dt * p.dtshrink >= this.dtmin) {
            this.dt *= p.dtshrink;
            for (const r of reps) r.s.dt = this.dt;
          }
        }
        // stopping criterion while stuck in a local basin of the PES
        this.vdotfNegatif++;
        if (p.vdfmax > 0 && this.vdotfNegatif > p.vdfmax) return 'max iterations with v.f negative';
        // inertia correction
        if (p.halfstepback) {
          for (const r of reps) {
            const s = r.s;
            for (let k = 0; k < 3 * s.n; k++) s.x[k] -= 0.5 * this.dt * s.v[k];
          }
        }
        for (const r of reps) r.s.v.fill(0, 0, 3 * r.s.n);
        this.flagv0 = 1;
      }

      if (!abc && this.flagv0) {
        // evaluates velocities to estimate whether dtv has to be limited (required when v have been reset)
        const dtf = this.dt * ftm2v;
        this.evaluateAll();
        this.neval++;
        for (const r of reps) kick(r, dtf, true);
      }

      // limit timestep so no particle moves further than dmax (the smallest over the replicas)
      let dtv = this.dt;
      for (const r of reps) {
        let dtvone = this.dt;
        if (!abc) {
          const s = r.s;
          for (let i = 0; i < s.n; i++) {
            let vmax = Math.max(Math.abs(s.v[3 * i]), Math.abs(s.v[3 * i + 1]));
            vmax = Math.max(vmax, Math.abs(s.v[3 * i + 2]));
            if (dtvone * vmax > p.dmax) dtvone = p.dmax / vmax;
          }
        }
        dtv = Math.min(dtv, dtvone);
      }
      // reset velocities when necessary
      if (this.flagv0) for (const r of reps) r.s.v.fill(0, 0, 3 * r.s.n);

      const mix = (s: SimState, i: number) => {
        if (vdotfall > 0.0) {
          for (let d = 0; d < 3; d++) {
            const k = 3 * i + d;
            s.v[k] = scale1 * s.v[k] + scale2 * s.f[k];
          }
          if (abc) {
            // make sure that the displacement is not larger than dmax
            for (let d = 0; d < 3; d++) {
              const k = 3 * i + d;
              if (Math.abs(s.v[k] * dtv) > p.dmax) s.v[k] = p.dmax / dtv * s.v[k] / Math.abs(s.v[k]);
            }
          }
        }
      };
      const move = (s: SimState, i: number) => {
        s.x[3 * i] += dtv * s.v[3 * i];
        s.x[3 * i + 1] += dtv * s.v[3 * i + 1];
        s.x[3 * i + 2] += dtv * s.v[3 * i + 2];
      };
      const kickOne = (r: FireReplica, i: number, dtf: number) => {
        const s = r.s;
        const dtfm = dtf / r.mass(i);
        s.v[3 * i] += dtfm * s.f[3 * i];
        s.v[3 * i + 1] += dtfm * s.f[3 * i + 1];
        s.v[3 * i + 2] += dtfm * s.f[3 * i + 2];
      };
      // eprevious = ecurrent; ecurrent = energy_force(0), on every replica
      const energyForce = () => {
        this.eprev = this.ecur;
        this.ecur = this.evaluateAll();
        this.eprevious = this.eprev[0];
        this.ecurrent = this.ecur[0];
        this.neval++;
      };

      if (integ === 'eulerimplicit' || integ === 'leapfrog') {
        const dtf = dtv * ftm2v;
        for (const r of reps) {
          for (let i = 0; i < r.s.n; i++) {
            kickOne(r, i, dtf);
            mix(r.s, i);
            move(r.s, i);
          }
        }
        energyForce();
      } else if (integ === 'verlet') {
        const dtf = 0.5 * dtv * ftm2v;
        for (const r of reps) {
          for (let i = 0; i < r.s.n; i++) {
            kickOne(r, i, dtf);
            mix(r.s, i);
            move(r.s, i);
          }
        }
        energyForce();
        for (const r of reps) for (let i = 0; i < r.s.n; i++) kickOne(r, i, dtf);
      } else {
        const dtf = dtv * ftm2v;
        for (const r of reps) {
          for (let i = 0; i < r.s.n; i++) {
            mix(r.s, i);
            move(r.s, i);
            kickOne(r, i, dtf);
          }
        }
        energyForce();
      }
      // velocities have been evaluated
      this.flagv0 = 0;

      // energy tolerance criterion: only after delaystep since velocities were reset; every replica must pass
      if (p.etol > 0.0 && ntimestep - this.lastNegative > p.delaystep) {
        let flagall = 0;
        this.ecur.forEach((ec, k) => {
          const ep = this.eprev[k];
          if (!(Math.abs(ec - ep) < p.etol * 0.5 * (Math.abs(ec) + Math.abs(ep) + EPS_ENERGY))) flagall++;
        });
        if (flagall === 0) return 'energy tolerance';
      }
      // force tolerance criterion: every replica must pass
      if (p.ftol > 0.0) {
        let flagall = 0;
        for (const r of reps) if (!(this.fnorm(r.s) < p.ftol * p.ftol)) flagall++;
        if (flagall === 0) return 'force tolerance';
      }
      await hooks.afterIteration(ntimestep);
    }
    return 'max iterations';
  }

  /** ecurrent and eprevious of every replica. */
  private ecur: number[] = [];
  private eprev: number[] = [];

  /** The energies of the replicas from Min::setup's energy_force, before the first iteration. */
  setEnergies(es: number[]): void {
    this.ecur = es.slice();
    this.eprev = es.slice();
    this.ecurrent = this.eprevious = es[0];
  }
}
