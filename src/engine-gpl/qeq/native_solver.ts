/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QEQ/fix_qeq.cpp (CG and calculate_Q), src/QEQ/fix_qeq_point.cpp and fix_qeq_shielded.cpp (init_matvec,
 * pre_force), and src/QEQ/fix_qeq_slater.cpp (the two conjugate-gradient solves and the charge sums), at commit
 * b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * The conjugate-gradient charge solver of qeq/point, qeq/shielded and qeq/slater. The core FixQeq
 * (src/engine/fix/qeq.ts) builds the matrix elements (diagonal, off-diagonal pairs, preconditioner, right-hand side,
 * the fixed-atom term) in the engine's own terms and calls this solver through the lazy HOOKS entries (lazyStyles.ts
 * group gpl_qeq, registry/qeq_native.ts). The solver owns the charge history of its fix and returns the new charges and
 * the iteration count.
 */
import type { QeqLinearSystem, QeqSolveInput, QeqSolveResult, QeqSolver, QeqSolverOptions } from '../../engine/fix/qeq';
import { QeqChargeHistory, QeqIdChargeHistory, initialGuessS, initialGuessT, pushAtomHistory } from './history';

/** y = H x for the sparse system: diagonal, then each off-diagonal pair (both directions), then the constant term. */
const matvec = (sys: QeqLinearSystem, x: Float64Array, out: Float64Array, fixed: Float64Array | null): void => {
  const { ng, diag, pairA, pairB, pairH } = sys;
  for (let k = 0; k < ng; k++) out[k] = diag[k] * x[k];
  for (let e = 0; e < pairH.length; e++) {
    const u = pairA[e], w = pairB[e], h = pairH[e];
    out[u] += h * x[w];
    out[w] += h * x[u];
  }
  if (fixed) for (let k = 0; k < ng; k++) out[k] += fixed[k];
};

interface CgResult {
  x: Float64Array;
  /** the loop counter at exit (native's loop: 1 at the start, one per update) */
  loop: number;
  /** sqrt(r.p)/|b| at exit (NaN for b = 0) */
  ratio: number;
}

/**
 * Preconditioned conjugate gradient with the Jacobi preconditioner 1/eta (sys.precond), from x0; the stop test is
 * sqrt(r.p)/|b| > tol inside the loop test loop < maxiter. fixed (qeq/point, qeq/shielded) is added to H x of the
 * first residual only, as native does for the s solve.
 */
const cg = (sys: QeqLinearSystem, b: Float64Array, x0: Float64Array, fixed: Float64Array | null, maxiter: number, tol: number): CgResult => {
  const { ng, precond } = sys;
  const x = Float64Array.from(x0);
  const hx = new Float64Array(ng);
  matvec(sys, x, hx, fixed);
  const r = new Float64Array(ng);
  for (let a = 0; a < ng; a++) r[a] = b[a] - hx[a];
  const d = new Float64Array(ng), q = new Float64Array(ng), pr = new Float64Array(ng);
  let bnorm2 = 0;
  for (let a = 0; a < ng; a++) bnorm2 += b[a] * b[a];
  const bnorm = Math.sqrt(bnorm2);
  let sigNew = 0;
  for (let a = 0; a < ng; a++) { d[a] = precond[a] * r[a]; sigNew += r[a] * d[a]; }
  let loop = 1;
  while (loop < maxiter && Math.sqrt(sigNew) / bnorm > tol) {
    matvec(sys, d, q, null);
    let tmp = 0;
    for (let a = 0; a < ng; a++) tmp += d[a] * q[a];
    const alfa = sigNew / tmp;
    for (let a = 0; a < ng; a++) { x[a] += alfa * d[a]; r[a] -= alfa * q[a]; }
    for (let a = 0; a < ng; a++) pr[a] = precond[a] * r[a];
    const sigOld = sigNew;
    sigNew = 0;
    for (let a = 0; a < ng; a++) sigNew += r[a] * pr[a];
    const beta = sigNew / sigOld;
    for (let a = 0; a < ng; a++) d[a] = pr[a] + beta * d[a];
    loop++;
  }
  return { x, loop, ratio: Math.sqrt(sigNew) / bnorm };
};

export class NativeQeqSolver implements QeqSolver {
  private readonly o: QeqSolverOptions;
  private readonly idHistory = new QeqIdChargeHistory();
  private indexHistory: QeqChargeHistory | null = null;

  constructor(o: QeqSolverOptions) {
    this.o = o;
  }

  seed(ids: ArrayLike<number>, q: Float64Array, n: number): void {
    for (let i = 0; i < n; i++) this.idHistory.seed(ids[i], q[i]);
  }

  solve(input: QeqSolveInput): QeqSolveResult {
    return this.o.kind === 'slater' ? this.solveSlater(input) : this.solveMatrix(input);
  }

  /**
   * qeq/point and qeq/shielded (fix_qeq.cpp CG, calculate_Q): s solves H s = b_s from the extrapolated history guess
   * (with the fixed term), t solves H t = -1 from its own guess; q = s - (sum s / sum t) t; the iteration scalar is
   * (loop_s + loop_t) / 2; each atom's history takes the new s and t in front.
   */
  private solveMatrix(input: QeqSolveInput): QeqSolveResult {
    const { sys, idx, ids, q } = input;
    const ng = sys.ng;
    const x0S = new Float64Array(ng), x0T = new Float64Array(ng), bT = new Float64Array(ng).fill(-1);
    const hist = new Array(ng);
    for (let a = 0; a < ng; a++) {
      const h = this.idHistory.entry(ids[a], q[idx[a]]);
      hist[a] = h;
      x0T[a] = initialGuessT(h);
      x0S[a] = initialGuessS(h);
    }
    const sRes = cg(sys, sys.bS, x0S, sys.fixed, this.o.maxiter, this.o.tol);
    if (sRes.loop >= this.o.maxiter && this.o.warn) {
      this.o.warnFn(`Fix ${this.o.style} CG convergence failed (${sRes.ratio}) after ${sRes.loop} iterations`);
    }
    const tRes = cg(sys, bT, x0T, null, this.o.maxiter, this.o.tol);
    if (tRes.loop >= this.o.maxiter && this.o.warn) {
      this.o.warnFn(`Fix ${this.o.style} CG convergence failed (${tRes.ratio}) after ${tRes.loop} iterations`);
    }
    let sumS = 0, sumT = 0;
    for (let a = 0; a < ng; a++) { sumS += sRes.x[a]; sumT += tRes.x[a]; }
    const u = sumS / sumT;
    const out = new Float64Array(ng);
    for (let a = 0; a < ng; a++) {
      out[a] = sRes.x[a] - u * tRes.x[a];
      pushAtomHistory(hist[a], sRes.x[a], tRes.x[a]);
    }
    return { q: out, iterations: (sRes.loop + tRes.loop) / 2 };
  }

  /**
   * qeq/slater: the same two solves with the history guesses of its index-keyed history, the reported count
   * trunc((loop_s + loop_t) / 2), one warning when either solve did not reach the tolerance, and q = s - (sum s / sum t) t.
   */
  private solveSlater(input: QeqSolveInput): QeqSolveResult {
    const { sys, idx, n, q } = input;
    const ng = sys.ng;
    if (!this.indexHistory || this.indexHistory.n !== n) this.indexHistory = new QeqChargeHistory(n, q);
    const guess = this.indexHistory.initialGuesses(idx);
    const bT = new Float64Array(ng).fill(-1);
    const sRes = cg(sys, sys.bS, guess.s, null, this.o.maxiter, this.o.tol);
    const tRes = cg(sys, bT, guess.t, null, this.o.maxiter, this.o.tol);
    const ok = (r: CgResult): boolean => r.ratio <= this.o.tol;
    const iterations = this.indexHistory.matvecCount(sRes.loop, tRes.loop);
    if (this.o.warn && (!ok(sRes) || !ok(tRes))) {
      this.o.warnFn(`Fix ${this.o.style} CG convergence failed after ${this.o.maxiter} iterations (maxiter reached)`);
    }
    let sumS = 0, sumT = 0;
    for (let k = 0; k < ng; k++) { sumS += sRes.x[k]; sumT += tRes.x[k]; }
    const out = new Float64Array(ng);
    for (let k = 0; k < ng; k++) out[k] = sRes.x[k] - (sumS / sumT) * tRes.x[k];
    this.indexHistory.shift(idx, sRes.x, tRes.x);
    return { q: out, iterations };
  }
}
