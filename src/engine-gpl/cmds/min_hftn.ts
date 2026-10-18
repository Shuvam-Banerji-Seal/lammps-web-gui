/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/min_hftn.cpp and min_hftn.h
 * (MinHFTN: min_style hftn, a Hessian-free truncated Newton minimizer; Todd Plantenga (SNL), after Nocedal and Wright,
 * Numerical Optimization, 2nd ed., p. 170, and Plantenga, SAND98-8201). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Each outer iteration solves the trust-region quadratic model by a CG inner loop whose Hessian-vector products are
 * finite-difference directional derivatives of the forces (forward differences while |F| > 1000 sqrt(eps), else central),
 * then accepts or rejects the step by the actual/predicted reduction ratio. The arithmetic, the order of the sums, the
 * position updates (including the ones that leave the positions off x_k between inner iterations) and the evaluation
 * counts follow the source, so that neval, niter and the thermo rows match native.
 *
 * Not ported: fix box/relax and per-atom degrees of freedom (the native docs say hftn "do not yet support" fix box/relax;
 * both are refused by name), and the progress file (bPrintProgress is false in the source). min_modify dmax is read
 * as the source does (Min::dmax); min_modify norm max is refused by name (MinHFTN::init).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { MinHooks, MinResult } from '../../engine/run/min';
import { minEnergyForce, type MinimizeParams, type MinimizerFn } from '../../engine/run/minimizers';
import { computeSetup } from '../../engine/run/verlet';
import { checkMinHalt } from '../../engine/run/halt';
import { restartDue } from '../../engine/commands/restart';
import { pow as libmPow } from '../libm/pow';

/** DBL_EPSILON (the source's MACHINE_EPS). */
const MACHINE_EPS = 2.220446049250313e-16;
/** the energy-tolerance test uses at least this magnitude (MIN_ETOL_MAG). */
const MIN_ETOL_MAG = 1.0e-8;

// inner CG step types (the source's CGSTEP_* constants; NO_CGSTEP_BECAUSE_F_TOL_SATISFIED is NO_CG)
const NO_CG = 0;
const CGSTEP_NEWTON = 1;
const CGSTEP_TO_TR = 2;
const CGSTEP_TO_DMAX = 3;
const CGSTEP_NEGATIVE_CURVATURE = 4;
const CGSTEP_MAX_INNER_ITERS = 5;
const CGSTEP_UNDETERMINED = 6;

interface InnerStep {
  type: number;
  /** energy and |F|_2 at x_k, only for NO_CG */
  eOut: number;
  f2Out: number;
  /** |p|_2 and |p|_inf of the step */
  len2: number;
  lenInf: number;
}

/**
 * min_style hftn (min_hftn.cpp MinHFTN::iterate and execute_hftn_, compute_inner_cg_step_, evaluate_dir_der_).
 * The stop strings are those of Min::stopstrings (min.cpp).
 */
export const minHftn: MinimizerFn = async (sys: System, p: MinimizeParams, hooks: MinHooks): Promise<MinResult> => {
  const s = sys.state;
  const set = sys.minSettings;
  // MinHFTN::init: "if (normstyle == MAX) error->all(FLERR,"Incorrect min_modify option");"
  if (set.norm === 'max') throw new StyleError('Incorrect min_modify option');
  if (sys.fixes.some((f) => (f as unknown as { boxDof?: unknown }).boxDof)) {
    throw new StyleError('min_style hftn with fix box/relax is not supported by the browser engine (min_style.html: "the hftn styles do not yet support the use of the fix box/relax command"); use cg or sd');
  }
  if (sys.ff.pair?.minPerAtom?.()) {
    throw new StyleError('min_style hftn with per-atom degrees of freedom (e.g. pair eff/cut) is not supported by the browser engine; use cg or sd');
  }
  const dmax = set.dmax;
  const n3 = 3 * s.n;
  // _nNumUnknowns = nvec + nextra_atom; no per-atom or global extra DOF here
  const nUnk = n3;
  const nb = sys.nb;
  const saved = { every: nb.every, delay: nb.delay, check: nb.check };
  if (nb.every !== 1 || nb.delay !== 0) {
    sys.log('Switching to \'neigh_modify every 1 delay 0 check yes\' setting during minimization');
    nb.every = 1; nb.delay = 0; nb.check = true;
  }

  // atom-based vectors of the source (VEC_XK, VEC_CG_P, VEC_CG_D, VEC_CG_HD, VEC_CG_R, VEC_DIF1, VEC_DIF2)
  const xk = new Float64Array(n3);
  const P = new Float64Array(n3);
  const D = new Float64Array(n3);
  const HD = new Float64Array(n3);
  const R = new Float64Array(n3);
  const DIF1 = new Float64Array(n3);
  const DIF2 = new Float64Array(n3);

  // Min::niter and Min::neval (both start at 0; the setup evaluation is not counted)
  let niter = 0;
  let neval = 0;
  let lastE = 0;
  // energy_force: one energy and force evaluation at the current positions (the source's neval++ is at the call sites)
  // the energies of the source are energy_force's: normalized by the atom count when thermo normalizes (lj units,
  // Min::energy_force "if (output->thermo->normflag) energy /= atom->natoms"), while the forces are not. The
  // predicted reduction (forces dotted with steps) stays unnormalized, as in the source.
  const energyForce = (first: boolean): number => {
    const e = minEnergyForce(sys, first);
    lastE = sys.thermo.norm ? e / s.n : e;
    return lastE;
  };
  // Min::fnorm_sqr (sum over the forces in order), Min::fnorm_inf (max of the squares, sqrt'd by the caller)
  const fnormSqr = (): number => {
    let t = 0;
    for (let i = 0; i < n3; i++) t += s.f[i] * s.f[i];
    return t;
  };
  const fnormInfSqr = (): number => {
    let m = 0;
    for (let i = 0; i < n3; i++) m = Math.max(m, s.f[i] * s.f[i]);
    return m;
  };
  const dot = (a: Float64Array, b: Float64Array): number => {
    let t = 0;
    for (let i = 0; i < n3; i++) t += a[i] * b[i];
    return t;
  };
  // calc_grad_dot_v_using_mpi_: -v . F at the current forces (F = -grad)
  const gradDotV = (v: Float64Array): number => {
    let t = 0;
    for (let i = 0; i < n3; i++) t += -v[i] * s.f[i];
    return t;
  };
  // calc_plengths_using_mpi_: |p|_2 and |p|_inf of P
  const plengths = (): { len2: number; lenInf: number } => {
    let pp = 0, pinf = 0;
    for (let i = 0; i < n3; i++) {
      pp += P[i] * P[i];
      pinf = Math.max(pinf, Math.abs(P[i]));
    }
    return { len2: Math.sqrt(pp), lenInf: pinf };
  };
  // step_exceeds_DMAX_ (no extra DOF: the box and per-atom limits do not apply): |P|_inf > dmax
  const stepExceedsDmax = (): boolean => {
    let pinf = 0;
    for (let i = 0; i < n3; i++) pinf = Math.max(pinf, Math.abs(P[i]));
    return pinf > dmax;
  };
  // adjust_step_to_tau_: P = DIF1 + tau D
  const adjustStepToTau = (tau: number): void => {
    for (let i = 0; i < n3; i++) P[i] = DIF1[i] + tau * D[i];
  };
  // compute_to_tr_: the tau with |p + tau d| = dTrustRadius (both roots when bBoth, the one lowering the CG model)
  const computeToTr = (dPP: number, dPD: number, dDD: number, dTR: number, bBoth: boolean,
    dDHD: number, dPdotHD: number, dGradDotD: number): number => {
    if (dDD <= 0.0 || dPP < 0.0 || dTR < 0.0 || dTR * dTR < dPP) {
      sys.log('HFTN internal error - bad data given to compute_to_tr_()');
      return 0.0;
    }
    const dTRsqrd = dTR * dTR;
    let dDiscr = (dPD * dPD) - (dDD * (dPP - dTRsqrd));
    dDiscr = Math.max(0.0, dDiscr);
    dDiscr = Math.sqrt(dDiscr);
    const dRootPos = (-dPD + dDiscr) / dDD;
    const dRootNeg = (-dPD - dDiscr) / dDD;
    if (!bBoth) return dRootPos;
    const dTmpTerm = dGradDotD + dPdotHD;
    const dCgRedPos = (dRootPos * dTmpTerm) + (0.5 * dRootPos * dRootPos * dDHD);
    const dCgRedNeg = (dRootNeg * dTmpTerm) + (0.5 * dRootNeg * dRootNeg * dDHD);
    if ((-dCgRedPos) > (-dCgRedNeg)) return dRootPos;
    return dRootNeg;
  };
  // evaluate_dir_der_: res = H dir by a finite difference of the forces; with bEvalAtX the forces are left at x_k
  // (the return value is the energy at x_k then, else unused). Central differences leave the positions at
  // x - eps dir when bEvalAtX is false, as the source does (the next call takes its base from there).
  const evaluateDirDer = (bFwd: boolean, dir: Float64Array, res: Float64Array, bEvalAtX: boolean): number => {
    let dirSq = 0;
    for (let i = 0; i < n3; i++) dirSq += dir[i] * dir[i];
    const dirNorm2 = Math.sqrt(dirSq);
    let dNewE = 0;
    // a zero direction gives a zero derivative without moving
    if (dirNorm2 === 0.0) {
      res.fill(0, 0, n3);
      if (bEvalAtX) { dNewE = energyForce(false); neval++; }
      return dNewE;
    }
    if (bFwd) {
      // "EQUATION IS FROM THE OLD LAMMPS VERSION, SAND98-8201."
      const dEps = 2.0 * Math.sqrt(1000.0 * MACHINE_EPS) / dirNorm2;
      for (let i = 0; i < n3; i++) DIF1[i] = s.x[i];
      for (let i = 0; i < n3; i++) s.x[i] += dEps * dir[i];
      energyForce(false); neval++;
      for (let i = 0; i < n3; i++) DIF2[i] = s.f[i];
      for (let i = 0; i < n3; i++) s.x[i] = DIF1[i];
      // the forward branch always evaluates at x again (two evaluations, whatever bEvalAtX)
      dNewE = energyForce(false); neval++;
      for (let i = 0; i < n3; i++) res[i] = (s.f[i] - DIF2[i]) / dEps;
      return dNewE;
    }
    // central differences: "EQUATION IS FROM THE OLD LAMMPS VERSION, SAND98-8201."
    const dEps = libmPow(3000.0 * MACHINE_EPS, 0.33333333) / dirNorm2;
    for (let i = 0; i < n3; i++) DIF1[i] = s.x[i];
    for (let i = 0; i < n3; i++) s.x[i] += dEps * dir[i];
    energyForce(false); neval++;
    for (let i = 0; i < n3; i++) DIF2[i] = s.f[i];
    for (let i = 0; i < n3; i++) s.x[i] = DIF1[i] - dEps * dir[i];
    energyForce(false); neval++;
    for (let i = 0; i < n3; i++) res[i] = (s.f[i] - DIF2[i]) / (2.0 * dEps);
    if (bEvalAtX) {
      for (let i = 0; i < n3; i++) s.x[i] = DIF1[i];
      dNewE = energyForce(false); neval++;
    }
    return dNewE;
  };

  // compute_inner_cg_step_: the CG inner loop for the trust-region step P (from x_k = xk)
  const innerCgStep = (dTrust: number, dForceTol: number, nMaxEvals: number, bHaveEvalAtXin: boolean,
    eIn: number, f2In: number): InnerStep => {
    P.fill(0, 0, n3);
    let dPP = 0;
    let eAtX = eIn;
    let f2AtX = f2In;
    if (!bHaveEvalAtXin) {
      eAtX = energyForce(false); neval++;
      f2AtX = Math.sqrt(fnormSqr());
    }
    // the force tolerance is already met: no step, the energy and forces are at x_k
    if (f2AtX <= dForceTol) return { type: NO_CG, eOut: eAtX, f2Out: f2AtX, len2: 0, lenInf: 0 };
    // r_0 = -grad = F (steepest descent first), d_0 = r_0
    for (let i = 0; i < n3; i++) { R[i] = s.f[i]; D[i] = s.f[i]; }
    let dRR = f2AtX * f2AtX;
    const dR0norm2 = Math.sqrt(dRR);
    // inner iteration limit: from the number of unknowns, and from the evaluations left (forward differences)
    let nLimit1 = Math.trunc(nUnk / 5);
    if (nLimit1 < 100) nLimit1 = Math.min(nUnk, 100);
    const nLimit2 = Math.trunc((nMaxEvals - neval) / 2);
    let nMaxInner = Math.min(nLimit1, nLimit2);
    if (f2AtX < Math.sqrt(MACHINE_EPS) * Math.max(1.0, Math.abs(eAtX))) {
      nMaxInner = Math.min(nMaxInner, Math.trunc(nUnk / 20));
    }
    const bUseForwardDiffs = f2AtX > 1000.0 * Math.sqrt(MACHINE_EPS);

    for (let nInner = 0; nInner < nMaxInner; nInner++) {
      // H(x_k) d_i
      evaluateDirDer(bUseForwardDiffs, D, HD, false);
      let dDHD = 0, dDD = 0;
      for (let i = 0; i < n3; i++) {
        dDHD += D[i] * HD[i];
        dDD += D[i] * D[i];
      }
      // negative curvature: project onto the trust region and keep the better of the two directions
      if (dDHD <= (MACHINE_EPS * dDD)) {
        const dPdotD = dot(P, D);
        const dPdotHD = dot(P, HD);
        // move to x_k and evaluate there
        for (let i = 0; i < n3; i++) s.x[i] = xk[i];
        eAtX = energyForce(false); neval++;
        const dGradDotD = gradDotV(D);
        const tau = computeToTr(dPP, dPdotD, dDD, dTrust, true, dDHD, dPdotHD, dGradDotD);
        for (let i = 0; i < n3; i++) P[i] += tau * D[i];
        const pl = plengths();
        return { type: CGSTEP_NEGATIVE_CURVATURE, eOut: eAtX, f2Out: 0, len2: pl.len2, lenInf: pl.lenInf };
      }
      // optimal step length of the quadratic model; P keeps p_i in DIF1 in case the step is shortened
      const dAlpha = dRR / dDHD;
      for (let i = 0; i < n3; i++) {
        DIF1[i] = P[i];
        P[i] += dAlpha * D[i];
      }
      // p_i+1 . p_i+1 and p_i . d_i
      let dPnewDotPnew = 0;
      let dPoldDotD = 0;
      for (let i = 0; i < n3; i++) {
        dPnewDotPnew += P[i] * P[i];
        dPoldDotD += DIF1[i] * D[i];
      }
      let nStepType = CGSTEP_UNDETERMINED;
      // step_exceeds_TR_: the step is cut to the trust region boundary
      if (plengths().len2 > dTrust) {
        const tau = computeToTr(dPP, dPoldDotD, dDD, dTrust, false, 0.0, 0.0, 0.0);
        adjustStepToTau(tau);
        nStepType = CGSTEP_TO_TR;
      }
      // dmax violations truncate the step to tau = 0 (without comparisons)
      if (stepExceedsDmax()) {
        adjustStepToTau(0.0);
        nStepType = CGSTEP_TO_DMAX;
      }
      if (nStepType === CGSTEP_TO_TR || nStepType === CGSTEP_TO_DMAX) {
        const pl = plengths();
        return { type: nStepType, eOut: 0, f2Out: 0, len2: pl.len2, lenInf: pl.lenInf };
      }
      // r_i+1 = r_i - alpha H d_i
      for (let i = 0; i < n3; i++) R[i] -= dAlpha * HD[i];
      let dRnewDotRnew = 0;
      for (let i = 0; i < n3; i++) dRnewDotRnew += R[i] * R[i];
      // a small residual ends the inner loop with a Newton step
      if (Math.sqrt(dRnewDotRnew) < dForceTol * dR0norm2) {
        const pl = plengths();
        return { type: CGSTEP_NEWTON, eOut: 0, f2Out: 0, len2: pl.len2, lenInf: pl.lenInf };
      }
      // beta = r_i+1 . r_i+1 / r_i . r_i ; d_i+1 = r_i+1 + beta d_i
      const dBeta = dRnewDotRnew / dRR;
      for (let i = 0; i < n3; i++) D[i] = R[i] + dBeta * D[i];
      dRR = dRnewDotRnew;
      dPP = dPnewDotPnew;
    }
    const pl = plengths();
    return { type: CGSTEP_MAX_INNER_ITERS, eOut: 0, f2Out: 0, len2: pl.len2, lenInf: pl.lenInf };
  };

  // Min::setup: the setup evaluation (not counted) and the initial thermo row
  computeSetup(sys);
  for (const f of sys.fixes) f.minSetup?.();
  const eInit = energyForce(true);
  const fnorm2Init = Math.sqrt(fnormSqr());
  const fnormInfInit = Math.sqrt(fnormInfSqr());
  let eprevious = eInit;
  let dCurE = eInit;
  let dCurF2 = fnorm2Init;
  const first = s.step;
  // output->next: thermo, dump and restart output on this step (the source writes only on those steps)
  const outputDue = (): boolean =>
    sys.thermo.due(s.step, first, Number.MAX_SAFE_INTEGER)
    || sys.dumps.some((d) => d.due(s.step, false))
    || restartDue(sys, s.step);
  hooks.thermo(false);

  let reason = 'max iterations';
  // execute_hftn_: the trust radius from the gradient and the box-size limit
  const dMaxTrustRadius = dmax * Math.sqrt(nUnk);
  if (fnorm2Init < p.ftol) {
    reason = 'force tolerance';
  } else {
    for (let i = 0; i < n3; i++) xk[i] = s.x[i];
    // calc_xinf_using_mpi_: the largest |x| at the start of the minimization
    let dXInf = 0.0;
    for (let i = 0; i < n3; i++) dXInf = Math.max(dXInf, Math.abs(s.x[i]));
    let dTrust = Math.min(1.5 * fnorm2Init, dMaxTrustRadius);
    let dLastNewton2 = dMaxTrustRadius;
    let bHaveEvaluatedAtX = true;

    for (niter = 0; niter < p.maxiter; niter++) {
      if (hooks.cancelled()) { reason = 'cancelled'; break; }
      s.step++;

      // inner stop tolerance on |F|_2 and the CG inner step
      const dCgForce2StopTol = Math.max(Math.min(dCurF2 / 2.0, 0.1 / (niter + 1)), p.ftol);
      const inner = innerCgStep(dTrust, dCgForce2StopTol, p.maxeval, bHaveEvaluatedAtX, dCurE, dCurF2);
      // the force tolerance is met at x_k: the energy and forces are already at x_k
      if (inner.type === NO_CG) {
        reason = 'force tolerance';
        break;
      }

      // predicted model: H(x_k) p (the forces at x_k are refreshed here)
      const bUseForwardDiffs = dCurF2 > 1000.0 * Math.sqrt(MACHINE_EPS);
      dCurE = evaluateDirDer(bUseForwardDiffs, P, HD, true);

      // p^T grad(x_k), saved for the predicted reduction
      const dGradDotP = gradDotV(P);

      // move to the new point and evaluate energy and forces there (energy_force may reset the neighbor list)
      for (let i = 0; i < n3; i++) s.x[i] = xk[i] + P[i];
      const dNewE = energyForce(false); neval++;
      const dNewF2 = Math.sqrt(fnormSqr());
      const dAred = dCurE - dNewE;

      // the force tolerance is met at the new point: the step is accepted
      if (dNewF2 < p.ftol) {
        reason = 'force tolerance';
        break;
      }

      // the actual energy reduction is tiny (not for steps cut at dmax)
      if (inner.type !== CGSTEP_TO_DMAX) {
        let dMag = 0.5 * (Math.abs(dCurE) + Math.abs(dNewE));
        dMag = Math.max(dMag, MIN_ETOL_MAG);
        if (Math.abs(dAred) < (p.etol * dMag) || inner.lenInf === 0.0) {
          reason = 'energy tolerance';
          break;
        }
      }

      // predicted reduction: - p^T grad - 0.5 p^T H p
      const dPHP = dot(P, HD);
      const dPred = -dGradDotP - (0.5 * dPHP);

      // accept, unless the energy went down by a tiny amount while the force norm grew
      const bStepAccepted = (dAred > 0.0) && ((dNewF2 < dCurF2) || (dCurF2 > 1.0e-6));
      if (bStepAccepted) {
        for (let i = 0; i < n3; i++) xk[i] = s.x[i];
        if (niter > 0) eprevious = dCurE;
        dCurE = dNewE;
        dCurF2 = dNewF2;
        bHaveEvaluatedAtX = true;
        if (inner.type === CGSTEP_NEWTON) dLastNewton2 = inner.len2;
        // trust region from the agreement of the actual and predicted reductions
        if ((dAred > 0.75 * dPred) && (inner.len2 >= 0.99 * dTrust)) dTrust = 2.0 * dTrust;
        dTrust = Math.min(dTrust, dMaxTrustRadius);
        // dmax violations: adjust the trust region so that dmax stops happening
        if (inner.type === CGSTEP_TO_DMAX) {
          if (inner.len2 <= MACHINE_EPS) dTrust = 0.1 * dTrust;
          else dTrust = Math.min(dTrust, 2.0 * inner.len2);
        }
      } else {
        // rejected: restore x_k
        for (let i = 0; i < n3; i++) s.x[i] = xk[i];
        bHaveEvaluatedAtX = false;
        // negative curvature takes a bad step a long way: be more aggressive; near a solution and past the Newton
        // steps, go back to near the last good Newton step
        if ((inner.type === CGSTEP_NEGATIVE_CURVATURE) && (-dAred > dPred)) {
          dTrust = 0.10 * Math.min(dTrust, inner.len2);
        } else if ((inner.type === CGSTEP_TO_DMAX) && (inner.len2 <= MACHINE_EPS)) {
          dTrust = 0.10 * dTrust;
        } else if (-dAred > dPred) {
          dTrust = 0.20 * Math.min(dTrust, inner.len2);
        } else {
          dTrust = 0.25 * Math.min(dTrust, inner.len2);
        }
        if ((inner.type !== CGSTEP_NEWTON) && (dCurF2 < Math.sqrt(MACHINE_EPS))) {
          dTrust = Math.min(dTrust, 2.0 * dLastNewton2);
        }
        dLastNewton2 = dMaxTrustRadius;
        // the trust region is too small to continue
        if (dTrust <= 0.0 || dTrust <= MACHINE_EPS * Math.max(1.0, dXInf)) {
          reason = 'trust region too small';
          break;
        }
      }

      // output: a rejected step re-evaluates the energy and forces at x_k so that the row shows the old point
      checkMinHalt(sys);
      if (outputDue()) {
        if (!bStepAccepted) { dCurE = energyForce(false); neval++; }
        hooks.thermo(false);
      }

      // the evaluation limit
      if (neval >= p.maxeval) {
        reason = 'max force evaluations';
        break;
      }
    }
  }

  // Min::run: after an early stop the energy and forces are evaluated at the current positions and the final output is
  // written (not counted in neval); a run that reaches maxiter writes nothing more
  if (reason !== 'max iterations') {
    energyForce(false);
    hooks.thermo(true);
  }
  nb.every = saved.every; nb.delay = saved.delay; nb.check = saved.check;
  return {
    reason,
    e0: eInit,
    ePrev: eprevious,
    e1: lastE,
    fnorm0: fnorm2Init,
    fnorm1: Math.sqrt(fnormSqr()),
    fmax0: fnormInfInit,
    fmax1: Math.sqrt(fnormInfSqr()),
    iterations: niter,
    evaluations: neval,
  };
};
