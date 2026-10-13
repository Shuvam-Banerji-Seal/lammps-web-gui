/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/min_spin.cpp and min_spin.h
 * (MinSpin: min_style spin, the damped geometric spin minimization), with the energy and force evaluation and the
 * torque norms of src/min.cpp (Min::energy_force, Min::total_torque, Min::max_torque, Min::inf_torque). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing authors (from min_spin.cpp): Julien Tranchida
 * (SNL) and Stan Moore (SNL).
 *
 * Only the spins move (min_style spin): each iteration takes the spin timestep dts = 2 pi / (discrete_factor
 * sqrt(max_i |fm_i|^2)) from the current precession vectors, advances every spin by the damped geometric step
 * (alpha_damp: the torque is -alpha_damp fm x s), and re-evaluates the energy and the precession vectors. The
 * minimization stops on the energy change (etol, checked after 5 iterations) or on the torque norm (ftol, the norm
 * selected by min_modify norm, scaled by hbar = hplanck/2 pi), or after maxiter iterations. min_modify alpha_damp
 * and discrete_factor are read from the minimizer settings (defaults 1.0 and 10.0).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { MinHooks, MinResult } from '../../engine/run/min';
import { minEnergyForce, type MinimizeParams } from '../../engine/run/minimizers';

const EPS_ENERGY = 1e-8;
/** min_spin.cpp: DELAYSTEP, the energy test is skipped for the first iterations */
const DELAYSTEP = 5;

/** The torque tau_i = fm_i x s_i of atom i (Min::total_torque and friends). */
const torqueOf = (fm: Float64Array, sp: Float64Array, i: number, out: number[]): void => {
  const fx = fm[3 * i], fy = fm[3 * i + 1], fz = fm[3 * i + 2];
  const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
  out[0] = fy * sz - fz * sy;
  out[1] = fz * sx - fx * sz;
  out[2] = fx * sy - fy * sx;
};

/** Torque norm of the given style ('two', 'max' or 'inf'), multiplied by hbar so that it is in eV. */
export const minSpinTorqueNorm = (sys: System, style: 'two' | 'max' | 'inf'): number => {
  const s = sys.state;
  const fm = s.fm!, sp = s.sp!;
  const hbar = s.units.hplanck / (2 * Math.PI);
  const t = [0, 0, 0];
  let two = 0, mx = 0, inf = 0;
  for (let i = 0; i < s.n; i++) {
    torqueOf(fm, sp, i, t);
    const sq = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    two += sq;
    if (sq > mx) mx = sq;
    for (let k = 0; k < 3; k++) if (t[k] * t[k] > inf) inf = t[k] * t[k];
  }
  if (style === 'two') return Math.sqrt(two) * hbar;
  if (style === 'max') return Math.sqrt(mx) * hbar;
  return Math.sqrt(inf) * hbar;
};

/** Spin timestep of the source: 2 pi / (discrete_factor sqrt(max_i |fm_i|^2)). */
export const minSpinEvaluateDt = (sys: System, discreteFactor: number): number => {
  const s = sys.state;
  const fm = s.fm!;
  let fmaxsq = 0;
  for (let i = 0; i < s.n; i++) {
    const q = fm[3 * i] * fm[3 * i] + fm[3 * i + 1] * fm[3 * i + 1] + fm[3 * i + 2] * fm[3 * i + 2];
    if (q > fmaxsq) fmaxsq = q;
  }
  if (fmaxsq === 0) throw new StyleError('Incorrect fmaxsqall calculation');
  return (2 * Math.PI) / (discreteFactor * Math.sqrt(fmaxsq));
};

/** Geometric damped advance of all spins (advance_spins of the source). */
const advanceSpins = (sys: System, alpha: number, dts: number): void => {
  const s = sys.state;
  const fm = s.fm!, sp = s.sp!;
  const dts2 = dts * dts;
  for (let i = 0; i < s.n; i++) {
    const fx = fm[3 * i], fy = fm[3 * i + 1], fz = fm[3 * i + 2];
    const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
    const tx = -alpha * (fy * sz - fz * sy);
    const ty = -alpha * (fz * sx - fx * sz);
    const tz = -alpha * (fx * sy - fy * sx);
    const fm2 = tx * tx + ty * ty + tz * tz;
    const energy = sx * tx + sy * ty + sz * tz;
    const cx = ty * sz - tz * sy;
    const cy = tz * sx - tx * sz;
    const cz = tx * sy - ty * sx;
    let gx = sx + cx * dts, gy = sy + cy * dts, gz = sz + cz * dts;
    gx += (tx * energy - 0.5 * sx * fm2) * 0.5 * dts2;
    gy += (ty * energy - 0.5 * sy * fm2) * 0.5 * dts2;
    gz += (tz * energy - 0.5 * sz * fm2) * 0.5 * dts2;
    const den = 1 + 0.25 * fm2 * dts2;
    sp[4 * i] = gx / den; sp[4 * i + 1] = gy / den; sp[4 * i + 2] = gz / den;
  }
};

const settingOf = (sys: System, key: string, dflt: number): number => {
  const v = sys.minSettings.fire[key];
  if (v === undefined) return dflt;
  const x = Number(v);
  if (!Number.isFinite(x)) throw new StyleError(`min_modify ${key}: '${v}' is not a number`);
  return x;
};

/** The min_modify settings of the spin minimizers: alpha_damp, discrete_factor and line (spin_cubic, spin_none). */
export const spinMinSettings = (sys: System): { alpha: number; discrete: number; line: string } => ({
  alpha: settingOf(sys, 'alpha_damp', 1.0),
  discrete: settingOf(sys, 'discrete_factor', 10.0),
  line: sys.minSettings.fire.line ?? 'spin_none',
});

/**
 * The stopping tests of the spin minimizers (iterate of the source, after the energy evaluation of an iteration):
 * the energy change after DELAYSTEP iterations (etol), then the torque norm (ftol). Returns the stop reason or ''.
 */
export const spinMinStop = (sys: System, p: MinimizeParams, iterations: number, ePrev: number, ecur: number): string => {
  if (p.etol > 0 && iterations > DELAYSTEP) {
    if (Math.abs(ecur - ePrev) < p.etol * 0.5 * (Math.abs(ecur) + Math.abs(ePrev) + EPS_ENERGY)) return 'energy tolerance';
  }
  if (p.ftol > 0) {
    const fmsq = minSpinTorqueNorm(sys, sys.minSettings.norm);
    if (fmsq * fmsq < p.ftol * p.ftol) return 'force tolerance';
  }
  return '';
};

/** min_style spin (min_spin.cpp MinSpin::iterate with setup_style). */
export const minSpin = async (sys: System, p: MinimizeParams, hooks: MinHooks): Promise<MinResult> => {
  const s = sys.state;
  if (!s.sp || !s.fm) throw new StyleError('min/spin requires atom/spin style');
  if (s.units.style !== 'metal') throw new StyleError('Spin pair styles require metal units');
  const { alpha, discrete } = spinMinSettings(sys);
  // setup_style: the atomic velocities are zeroed (the spins are the only degrees of freedom)
  if (s.v) s.v.fill(0);
  // Min::setup: the non-default neighbor settings are used during the minimization
  const nb = sys.nb;
  const saved = { every: nb.every, delay: nb.delay, check: nb.check };
  if (nb.every !== 1 || nb.delay !== 0) {
    sys.log('Switching to \'neigh_modify every 1 delay 0 check yes\' setting during minimization');
    nb.every = 1; nb.delay = 0; nb.check = true;
  }
  let evaluations = 0;
  let ecur = minEnergyForce(sys, true);
  evaluations++;
  const e0 = ecur;
  let ePrev = ecur;
  const fnorm0 = minSpinTorqueNorm(sys, 'two'), fmax0 = minSpinTorqueNorm(sys, 'inf');
  hooks.thermo(false);
  const start = s.step;
  let reason = 'max iterations';
  let iter = 0;
  for (let k = 0; k < p.maxiter; k++) {
    if (hooks.cancelled()) { reason = 'cancelled'; break; }
    s.step++;
    iter++;
    // iteration 0 starts from a re-evaluated state (energy_force at iter == 0 of the source)
    if (k === 0) { minEnergyForce(sys, false); evaluations++; }
    const dts = minSpinEvaluateDt(sys, discrete);
    advanceSpins(sys, alpha, dts);
    ePrev = ecur;
    ecur = minEnergyForce(sys, false);
    evaluations++;
    const stop = spinMinStop(sys, p, s.step - start, ePrev, ecur);
    if (stop) { reason = stop; break; }
    hooks.thermo(false);
  }
  nb.every = saved.every; nb.delay = saved.delay; nb.check = saved.check;
  hooks.thermo(true);
  return {
    reason, e0, ePrev, e1: ecur,
    fnorm0, fnorm1: minSpinTorqueNorm(sys, 'two'), fmax0, fmax1: minSpinTorqueNorm(sys, 'inf'),
    iterations: iter, evaluations,
  };
};
