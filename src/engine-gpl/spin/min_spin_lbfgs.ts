/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/min_spin_lbfgs.cpp and
 * min_spin_lbfgs.h (MinSpinLBFGS: min_style spin/lbfgs, limited-memory BFGS on the spins, Ivanov, Uzdin and Jonsson
 * 2019, arXiv:1904.02669; the L-BFGS two-loop recursion of Nocedal and Wright, Numerical Optimization, 2nd ed., p. 177),
 * with the Rodrigues rotation and the torque helpers of min_spin_cg.ts and min_spin.ts. LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE). Contributing authors (from min_spin_lbfgs.cpp): Aleksei Ivanov (University of Iceland)
 * and Julien Tranchida (SNL).
 *
 * This port covers the path without a line search, which is the native default (min.cpp starts with linestyle
 * QUADRATIC, which is not SPIN_CUBIC, so use_line_search = 0) and min_modify line spin_none: the gradient
 * g_i = hbar (fm_i x s_i), the L-BFGS direction with three memory pairs, the maximum rotation angle
 * maxepsrot = 2 pi / (10 discrete_factor) (min_modify discrete_factor) and the rotation of each spin about its search
 * direction. The spin_cubic line search (calc_and_make_step) is refused by name.
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { MinHooks, MinResult } from '../../engine/run/min';
import { minEnergyForce, type MinimizeParams } from '../../engine/run/minimizers';
import { minSpinTorqueNorm, spinMinSettings, spinMinStop } from './min_spin';
import { applyRot, rodrigues } from './min_spin_cg';

/** num_mem in min_spin_lbfgs.cpp::init */
const NUM_MEM = 3;

/** min_style spin/lbfgs (min_spin_lbfgs.cpp MinSpinLBFGS::iterate without a line search). */
export const minSpinLBFGS = async (sys: System, p: MinimizeParams, hooks: MinHooks): Promise<MinResult> => {
  const s = sys.state;
  if (!s.sp || !s.fm) throw new StyleError('min spin/lbfgs requires atom/spin style');
  const set = spinMinSettings(sys);
  if (set.line === 'spin_cubic') throw new StyleError('min_modify line spin_cubic (the spin/lbfgs line search) is not supported by the browser engine; use line spin_none');
  const n = s.n;
  const len = 3 * n;
  const hbar = s.units.hplanck / (2 * Math.PI);
  // min_spin_lbfgs.cpp::modify_param: maxepsrot = 2 pi / (10 discrete_factor)
  const maxepsrot = (2 * Math.PI) / (10 * set.discrete);
  const gOld = new Float64Array(len), gCur = new Float64Array(len), pS = new Float64Array(len);
  const ds: Float64Array[] = [], dy: Float64Array[] = [];
  for (let k = 0; k < NUM_MEM; k++) { ds.push(new Float64Array(len)); dy.push(new Float64Array(len)); }
  const rho = new Float64Array(NUM_MEM);
  const rot = new Float64Array(9);
  if (s.v) s.v.fill(0);
  const nb = sys.nb;
  const saved = { every: nb.every, delay: nb.delay, check: nb.check };
  if (nb.every !== 1 || nb.delay !== 0) {
    sys.log('Switching to \'neigh_modify every 1 delay 0 check yes\' setting during minimization');
    nb.every = 1; nb.delay = 0; nb.check = true;
  }
  const fm = s.fm, sp = s.sp;

  /** maximum_rotation: the scaling that keeps the largest rotation angle at maxepsrot (alpha = 1 when larger). */
  const maxRotation = (v: Float64Array): number => {
    let norm2 = 0;
    for (let i = 0; i < len; i++) norm2 += v[i] * v[i];
    const scaling = maxepsrot * Math.sqrt(n / norm2);
    return scaling < 1.0 ? scaling : 1.0;
  };

  /** calc_gradient: g_i = hbar (fm_i x s_i) in the source's sign convention. */
  const calcGradient = (): void => {
    for (let i = 0; i < n; i++) {
      const fx = fm[3 * i], fy = fm[3 * i + 1], fz = fm[3 * i + 2];
      const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
      gCur[3 * i] = (fx * sy - fy * sx) * hbar;
      gCur[3 * i + 1] = -(fz * sx - fx * sz) * hbar;
      gCur[3 * i + 2] = (fy * sz - fz * sy) * hbar;
    }
  };

  /** calc_search_direction (the two-loop L-BFGS recursion); localIter counts the calls since the last reset. */
  let localIter = 0;
  const calcSearchDirection = (): void => {
    if (localIter === 0) {
      // steepest descent with the maximum rotation scaling; the memory is cleared
      const scaling = maxRotation(gCur);
      for (let i = 0; i < len; i++) {
        pS[i] = -gCur[i] * scaling;
        gOld[i] = gCur[i];
        for (let k = 0; k < NUM_MEM; k++) { ds[k][i] = 0; dy[k][i] = 0; }
      }
      rho.fill(0);
    } else {
      const m = localIter % NUM_MEM;
      let dyds = 0;
      for (let i = 0; i < len; i++) {
        ds[m][i] = pS[i];
        dy[m][i] = gCur[i] - gOld[i];
        dyds += ds[m][i] * dy[m][i];
      }
      rho[m] = Math.abs(dyds) > 1.0e-60 ? 1.0 / dyds : 1.0e60;
      if (rho[m] < 0.0) {
        // a negative curvature pair restarts the memory (the recursive call of the source ends the iteration)
        localIter = 0;
        calcSearchDirection();
        return;
      }
      const q = Float64Array.from(gCur);
      const alpha = new Float64Array(NUM_MEM);
      for (let k = NUM_MEM - 1; k > -1; k--) {
        const c = (k + m + 1) % NUM_MEM;
        let sq = 0;
        for (let i = 0; i < len; i++) sq += ds[c][i] * q[i];
        alpha[c] = rho[c] * sq;
        for (let i = 0; i < len; i++) q[i] -= alpha[c] * dy[c][i];
      }
      let yy = 0;
      for (let i = 0; i < len; i++) yy += dy[m][i] * dy[m][i];
      const devis = rho[m] * yy;
      if (Math.abs(devis) > 1.0e-60) {
        for (let i = 0; i < len; i++) pS[i] = q[i] / devis;
      } else {
        for (let i = 0; i < len; i++) pS[i] = q[i] * 1.0e60;
      }
      for (let k = 0; k < NUM_MEM; k++) {
        const c = localIter < NUM_MEM ? k : (k + m + 1) % NUM_MEM;
        let yr = 0;
        for (let i = 0; i < len; i++) yr += dy[c][i] * pS[i];
        const beta = rho[c] * yr;
        for (let i = 0; i < len; i++) pS[i] += ds[c][i] * (alpha[c] - beta);
      }
      const scaling = maxRotation(pS);
      for (let i = 0; i < len; i++) {
        pS[i] = -pS[i] * scaling;
        gOld[i] = gCur[i];
      }
    }
    localIter++;
  };

  sys.fixEnergyFrozen = null;
  let ecur = minEnergyForce(sys, true);
  let evaluations = 1;
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
    // no line search: energy and forces, gradient, search direction, then the rotation of every spin
    ePrev = ecur;
    sys.fixEnergyFrozen = null;
    ecur = minEnergyForce(sys, false);
    evaluations++;
    // the thermo energy of this iteration is that of the evaluation (native: the fix energy is summed in post_force)
    sys.freezeFixEnergies();
    calcGradient();
    calcSearchDirection();
    for (let i = 0; i < n; i++) {
      rodrigues(pS, 3 * i, rot);
      applyRot(rot, sp, 4 * i);
    }
    const stop = spinMinStop(sys, p, s.step - start, ePrev, ecur);
    if (stop) { reason = stop; break; }
    hooks.thermo(false);
  }
  nb.every = saved.every; nb.delay = saved.delay; nb.check = saved.check;
  hooks.thermo(true);
  sys.fixEnergyFrozen = null;
  return {
    reason, e0, ePrev, e1: ecur,
    fnorm0, fnorm1: minSpinTorqueNorm(sys, 'two'), fmax0, fmax1: minSpinTorqueNorm(sys, 'inf'),
    iterations: iter, evaluations,
  };
};
