/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QEQ/fix_qeq.cpp (nprev, s_hist and t_hist, calculate_Q), src/QEQ/fix_qeq_slater.cpp and
 * src/QEQ/fix_qeq_point.cpp / fix_qeq_shielded.cpp (init_matvec, the extrapolated initial guesses), and the averaged
 * matvec count, at commit b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The five-entry charge histories of the qeq conjugate-gradient solves. Two keyings exist: qeq/slater keeps one per
 * local index (QeqChargeHistory, created at the first solve), qeq/point and qeq/shielded keep one per atom id
 * (QeqIdChargeHistory, seeded by the fix constructor). The solves that use them are in native_solver.ts; the core
 * FixQeq (src/engine/fix/qeq.ts) reaches them through the lazy HOOKS entries 'qeq/point', 'qeq/shielded' and
 * 'qeq/slater' (lazyStyles.ts group gpl_qeq).
 */

/** Entries kept per atom: the solutions of the last five solves (nprev = 5 in fix_qeq.cpp). */
const NPREV = 5;

/** The s and t solutions of one atom, newest first. */
export interface QeqAtomHistory {
  readonly s: Float64Array;
  readonly t: Float64Array;
}

/** Initial guesses of the s and t solves (fix_qeq_point.cpp and fix_qeq_shielded.cpp, init_matvec): s = 4(h0 + h2) - (6 h1 + h3), t = h2 + 3 (h0 - h1). */
export const initialGuessS = (h: QeqAtomHistory): number => 4 * (h.s[0] + h.s[2]) - (6 * h.s[1] + h.s[3]);
export const initialGuessT = (h: QeqAtomHistory): number => h.t[2] + 3 * (h.t[0] - h.t[1]);

const newAtomHistory = (q: number): QeqAtomHistory => ({ s: new Float64Array(NPREV).fill(q), t: new Float64Array(NPREV).fill(q) });

/** calculate_Q: the history of one atom takes the new solutions in front (the newest first). */
export const pushAtomHistory = (h: QeqAtomHistory, sx: number, tx: number): void => {
  for (let k = NPREV - 1; k > 0; k--) {
    h.s[k] = h.s[k - 1];
    h.t[k] = h.t[k - 1];
  }
  h.s[0] = sx;
  h.t[0] = tx;
};

/** The histories of qeq/point and qeq/shielded, keyed by atom id (kept across runs and reorderings of the atoms). */
export class QeqIdChargeHistory {
  private readonly entries = new Map<number, QeqAtomHistory>();

  /** Native's fix constructor: every atom's history starts at its charge (a repeated id keeps the last value). */
  seed(id: number, q: number): void {
    this.entries.set(id, newAtomHistory(q));
  }

  /** The history of an atom; an atom without one starts at its current charge. */
  entry(id: number, q: number): QeqAtomHistory {
    let h = this.entries.get(id);
    if (!h) {
      h = newAtomHistory(q);
      this.entries.set(id, h);
    }
    return h;
  }
}

/** The histories of qeq/slater, one per local index of n atoms. */
export class QeqChargeHistory {
  private readonly sHist: Float64Array;
  private readonly tHist: Float64Array;

  /** The history of n atoms, every entry set to the atoms' current charges (fix_qeq.cpp, at creation). */
  constructor(readonly n: number, q: Float64Array) {
    this.sHist = new Float64Array(NPREV * n);
    this.tHist = new Float64Array(NPREV * n);
    for (let i = 0; i < n; i++) for (let k = 0; k < NPREV; k++) { this.sHist[NPREV * i + k] = q[i]; this.tHist[NPREV * i + k] = q[i]; }
  }

  /** Initial guesses of the s and t solves for the group atoms idx (init_matvec, see initialGuessS / initialGuessT). */
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
