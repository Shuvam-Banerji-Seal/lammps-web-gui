/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/min_spin_cg.cpp and min_spin_cg.h
 * (MinSpinCG: min_style spin/cg, orthogonal spin optimisation by Fletcher-Reeves conjugate gradient, Ivanov, Uzdin
 * and Jonsson 2019, arXiv:1904.02669), with the energy and torque helpers of min_spin.ts. LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE). Contributing authors (from min_spin_cg.cpp): Julien Tranchida (SNL) and Stan Moore
 * (SNL).
 *
 * Each iteration: the gradient g_i = (fm_i x s_i) (scaled by the spin timestep of the source, 2 pi / (discrete_factor
 * sqrt(max |fm|^2))), the Fletcher-Reeves direction (steepest descent every fifth iteration), and a rotation of every
 * spin about its direction by |p_i| (Rodrigues' formula, the matrix of the source). The line search of the source
 * (min_modify line spin_cubic) is not ported: spin_cubic is refused by name, spin_none is the default and is used.
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { MinHooks, MinResult } from '../../engine/run/min';
import { minEnergyForce, type MinimizeParams } from '../../engine/run/minimizers';
import { minSpinEvaluateDt, minSpinTorqueNorm, spinMinSettings, spinMinStop } from './min_spin';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** Rodrigues' formula for the rotation by the vector p (rot is row-major 3x3 in the source's index order). */
export const rodrigues = (p: Float64Array, off: number, out: Float64Array): void => {
  const px = p[off], py = p[off + 1], pz = p[off + 2];
  if (Math.abs(px) < 1.0e-40 && Math.abs(py) < 1.0e-40 && Math.abs(pz) < 1.0e-40) {
    out.fill(0);
    out[0] = 1; out[4] = 1; out[8] = 1;
    return;
  }
  const theta = Math.sqrt(px * px + py * py + pz * pz);
  const A = libmCos(theta), B = libmSin(theta), D = 1.0 - A;
  const x = px / theta, y = py / theta, z = pz / theta;
  out[0] = A + z * z * D;
  out[4] = A + y * y * D;
  out[8] = A + x * x * D;
  const s1 = -y * z * D, s2 = x * z * D, s3 = -x * y * D;
  const a1 = x * B, a2 = y * B, a3 = z * B;
  out[1] = s1 + a1;
  out[3] = s1 - a1;
  out[2] = s2 + a2;
  out[6] = s2 - a2;
  out[5] = s3 + a3;
  out[7] = s3 - a3;
};

/** s_new = rot^T s (the source's vm3: out[i] = sum_j m[3j+i] v[j]). */
export const applyRot = (rot: Float64Array, s: Float64Array, off: number): void => {
  const v0 = s[off], v1 = s[off + 1], v2 = s[off + 2];
  s[off] = rot[0] * v0 + rot[3] * v1 + rot[6] * v2;
  s[off + 1] = rot[1] * v0 + rot[4] * v1 + rot[7] * v2;
  s[off + 2] = rot[2] * v0 + rot[5] * v1 + rot[8] * v2;
};

/** min_style spin/cg (min_spin_cg.cpp MinSpinCG::iterate without a line search). */
export const minSpinCG = async (sys: System, p: MinimizeParams, hooks: MinHooks): Promise<MinResult> => {
  const s = sys.state;
  if (!s.sp || !s.fm) throw new StyleError('min spin/cg requires atom/spin style');
  const set = spinMinSettings(sys);
  if (set.line === 'spin_cubic') throw new StyleError('min_modify line spin_cubic (the spin/cg line search) is not supported by the browser engine; use line spin_none');
  const n = s.n;
  const pS = new Float64Array(3 * n), gOld = new Float64Array(3 * n), gCur = new Float64Array(3 * n);
  const rot = new Float64Array(9);
  if (s.v) s.v.fill(0);
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
  let localIter = 0;
  const fm = s.fm, sp = s.sp;
  for (let k = 0; k < p.maxiter; k++) {
    if (hooks.cancelled()) { reason = 'cancelled'; break; }
    s.step++;
    iter++;
    // calc_gradient (no line search: factor = the spin timestep of the current torques)
    const factor = minSpinEvaluateDt(sys, set.discrete);
    for (let i = 0; i < n; i++) {
      const fx = fm[3 * i], fy = fm[3 * i + 1], fz = fm[3 * i + 2];
      const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
      gCur[3 * i] = (fx * sy - fy * sx) * factor;
      gCur[3 * i + 1] = -(fz * sx - fx * sz) * factor;
      gCur[3 * i + 2] = (fy * sz - fz * sy) * factor;
    }
    // calc_search_direction: steepest descent at the start and every fifth iteration, else Fletcher-Reeves
    const len = 3 * n;
    if (localIter === 0 || localIter % 5 === 0) {
      for (let i = 0; i < len; i++) { pS[i] = -gCur[i]; gOld[i] = gCur[i]; }
    } else {
      let g2old = 0, g2 = 0;
      for (let i = 0; i < len; i++) { g2old += gOld[i] * gOld[i]; g2 += gCur[i] * gCur[i]; }
      const beta = Math.abs(g2) < 1.0e-60 ? 0 : g2 / g2old;
      for (let i = 0; i < len; i++) { pS[i] = beta * pS[i] - gCur[i]; gOld[i] = gCur[i]; }
    }
    localIter++;
    // advance_spins: each spin rotates about its search direction
    for (let i = 0; i < n; i++) {
      rodrigues(pS, 3 * i, rot);
      applyRot(rot, sp, 4 * i);
    }
    evaluations++;
    ePrev = ecur;
    ecur = minEnergyForce(sys, false);
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
