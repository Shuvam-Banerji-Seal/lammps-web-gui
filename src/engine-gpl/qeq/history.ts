/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QEQ/fix_qeq.cpp (nprev, s_hist and t_hist, calculate_Q), src/QEQ/fix_qeq_slater.cpp and
 * src/QEQ/fix_qeq_point.cpp / fix_qeq_shielded.cpp (init_matvec, the extrapolated initial guesses), and the averaged
 * matvec count, at commit b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The five-entry charge history of the qeq/slater conjugate-gradient solves. The core FixQeq (src/engine/fix/qeq.ts)
 * keeps the solver and reaches this through the lazy HOOKS entry 'qeq/slater' (lazyStyles.ts group gpl_qeq_history).
 */

/** Entries kept per atom: the solutions of the last five solves (nprev = 5 in fix_qeq.cpp). */
const NPREV = 5;

export class QeqChargeHistory {
  private readonly sHist: Float64Array;
  private readonly tHist: Float64Array;

  /** The history of n atoms, every entry set to the atoms' current charges (fix_qeq.cpp, at creation). */
  constructor(readonly n: number, q: Float64Array) {
    this.sHist = new Float64Array(NPREV * n);
    this.tHist = new Float64Array(NPREV * n);
    for (let i = 0; i < n; i++) for (let k = 0; k < NPREV; k++) { this.sHist[NPREV * i + k] = q[i]; this.tHist[NPREV * i + k] = q[i]; }
  }

  /**
   * Initial guesses of the s and t solves for the group atoms idx (fix_qeq_point.cpp and fix_qeq_shielded.cpp,
   * init_matvec): s = 4(h0 + h2) - (6 h1 + h3) and t = h2 + 3 (h0 - h1).
   */
  initialGuesses(idx: number[]): { s: Float64Array; t: Float64Array } {
    const sh = this.sHist, th = this.tHist;
    const ng = idx.length;
    const s = new Float64Array(ng), t = new Float64Array(ng);
    for (let a = 0; a < ng; a++) {
      const i = idx[a];
      s[a] = 4 * (sh[5 * i] + sh[5 * i + 2]) - (6 * sh[5 * i + 1] + sh[5 * i + 3]);
      t[a] = th[5 * i + 2] + 3 * (th[5 * i] - th[5 * i + 1]);
    }
    return { s, t };
  }

  /** calculate_Q: the history of each group atom shifts in the new solution (sx, tx in group order). */
  shift(idx: number[], sx: Float64Array, tx: Float64Array): void {
    const sh = this.sHist, th = this.tHist;
    for (let a = 0; a < idx.length; a++) {
      const i = idx[a];
      for (let k = NPREV - 1; k > 0; k--) { sh[NPREV * i + k] = sh[NPREV * i + k - 1]; th[NPREV * i + k] = th[NPREV * i + k - 1]; }
      sh[NPREV * i] = sx[a];
      th[NPREV * i] = tx[a];
    }
  }

  /** The reported iteration count: (CG(s) + CG(t)) / 2 with integer division (fix_qeq.cpp, fix_qeq_slater.cpp). */
  matvecCount(sIt: number, tIt: number): number {
    return Math.trunc((sIt + tIt) / 2);
  }
}
