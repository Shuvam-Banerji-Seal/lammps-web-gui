import { Fix } from './fix';
import { StyleError, typeBounds } from '../force/types';
import type { System } from '../system';

/*
 * fix qeq/point, qeq/shielded, qeq/dynamic and qeq/fire — docs.lammps.org/fix_qeq.html
 * (plans/lammps-docs/fix_qeq.rst):
 *
 *   "fix ID group-ID style Nevery cutoff tolerance maxiter qfile keyword ..."
 *   "Perform the charge equilibration (QEq) method ... based on the
 *    electronegativity equalization principle."
 *   "The QEq method minimizes the electrostatic energy of the system (or
 *    equalizes the derivative of energy with respect to charge of all the
 *    atoms) by adjusting the partial charge on individual atoms based on
 *    interactions with their neighbors within cutoff."
 *   "The qeq/point style describes partial charges on atoms as point charges.
 *    Interaction between a pair of charged particles is 1/r ... Only the chi
 *    and eta parameters from the qfile file are used. ... This style solves
 *    partial charges on atoms via the matrix inversion method."
 *   "The qeq/shielded style ... uses a shielded Coulomb potential ... given by
 *    equation (13) of the ReaxFF force field paper. ... Only the chi, eta, and
 *    gamma parameters from the qfile file are used. ... This style is the same
 *    as fix qeq/reaxff, and can be used with pair_style reaxff."
 *   "The qeq/dynamic style describes partial charges on atoms as point
 *    charges that interact through 1/r, but the extended Lagrangian method is
 *    used to solve partial charges on atoms.  Only the chi and eta
 *    parameters from the qfile file are used. ... A tolerance of 1.0e-3 is
 *    usually a good number.  Keyword qdamp can be used to change the damping
 *    factor, while keyword qstep can be used to change the time step size."
 *   "The qeq/fire style describes the same charge model and charge solver as
 *    the qeq/dynamic style, but employs a FIRE minimization algorithm to solve
 *    for equilibrium charges.  Keyword qdamp can be used to change the damping
 *    factor, while keyword qstep can be used to change the time step size."
 *   "The fix qeq styles will print a warning if the charges are not
 *    equilibrated within tolerance by maxiter steps, unless the warn keyword
 *    is used with "no" as argument."
 *   "In order to solve the self-consistent equations for electronegativity
 *    equalization, LAMMPS imposes the additional constraint that all the
 *    charges in the fix group must add up to zero.  The initial charge
 *    assignments should also satisfy this constraint.  LAMMPS will print a
 *    warning if that is not the case."
 * qfile format: "1 chi eta gamma zeta qcore" per type; "There have to be
 *    parameters given for every atom type." Wildcard type ranges work as for the
 *    coeff commands; "Later entries will overwrite previous ones."
 *
 * The QEq matrix is H_ii = eta_i, H_ij = pair interaction, solved together
 * with the charge-neutrality Lagrange multiplier.  Measured with native
 * LAMMPS (black box, not part of the doc text):
 *   - qeq/point off-diagonal is exactly 1/r inside the cutoff (no units
 *     constant, no taper), independent of the cutoff;
 *   - qeq/shielded off-diagonal is 14.4 * (r^3 + gamma_ij^-3)^(-1/3) times a
 *     7th-order taper T(r/cutoff) = 1 - 35x^4 + 84x^5 - 70x^6 + 20x^7 for
 *     r < cutoff (0 beyond), with gamma_ij = sqrt(gamma_i*gamma_j); the 14.4
 *     is the fix's hard-coded A/eV/charge Coulomb constant;
 *   - the charges match those of a Jacobi-preconditioned conjugate-gradient
 *     solve (preconditioner 1/diag, last diagonal 1; stop when
 *     max|residual| / max|rhs| < tolerance), and the fix global scalar is the
 *     iteration count of that solve (measured: 2-8 atoms give N+1 iterations
 *     at tight tolerance), although the doc says these fixes store no global
 *     scalar;
 *   - qeq/dynamic and qeq/fire also expose the iteration count as the fix
 *     global scalar (measured with native LAMMPS, black box), again despite
 *     the doc's "no global scalar" sentence.
 *
 * qeq/dynamic (measured with native LAMMPS, black box): let the charge force
 * be f_a = -(g_a - mean(g)), where g_a = chi_a + eta_a q_a + sum_neighbors q_b / r_ab
 * is the electronegativity gradient and the mean is over the fix group (the
 * neutrality constraint).  With gamma = 1 - qdamp and
 * B = 31.25 * qstep^2 * (2 - qdamp), the iterates are
 *     d_0 = B f(q_0),             q_{n+1} = q_n + d_n,
 *     d_{n+1} = gamma^2 d_n + B f(q_{n+1}),
 * at most maxiter - 1 of them, stopped when mean |f| < tolerance; the fix
 * scalar is the number of updates.  Native's charges after 1..4 capped
 * iterations of a 6-atom system are reproduced to 1e-16 (the 31.25 is a
 * measured constant), and its iteration counts at tolerances 1e-1 ... 1e-5 on
 * an irregular 5-atom cluster only with the mean |f| test (see meanAbs).
 * Defaults qdamp = 0.1, qstep = 0.02.
 *
 * qeq/fire (measured with native LAMMPS, black box): the same charge model and
 * force, minimized by the FIRE-like damped dynamics of solveFire (every
 * native iterate of a solve from a fresh fix reproduced to 1e-16); default
 * qstep 0.2, qdamp unused.  Not reproduced: native carries part of its FIRE
 * state into the next solve of a run (the first update of the second solve is
 * about 0.01 f + 0.009 f_end, f_end the converged force of the previous solve,
 * which no combination of its v, dt and forces reproduced exactly); the engine
 * starts every solve fresh, so later solves of a run differ from native by
 * about 1e-6 in pe at tolerance 1e-5 and 3e-4 at 1e-3.
 *
 * Not reproduced: with a kspace style, native's charge sums at the setup of
 * the first run lag the solve (see the end of solve).
 *
 * Unsupported qeq styles and qfile keywords throw a StyleError naming them.
 */

type QeqKind = 'point' | 'shielded' | 'dynamic' | 'fire';

interface Params {
  chi: Float64Array;
  eta: Float64Array;
  gamma: Float64Array;
}

const COUL = 14.4; // hard-coded Coulomb constant of fix qeq (measured with native LAMMPS, black box)

/** 7th-order taper used by the shielded interaction (measured with native LAMMPS, black box). */
const taper = (x: number): number => {
  if (x >= 1) return 0;
  return 1 - 35 * x * x * x * x + 84 * x ** 5 - 70 * x ** 6 + 20 * x ** 7;
};

/** Constant of the qeq/dynamic charge dynamics (measured with native LAMMPS, black box). */
const DYNAMIC_SCALE = 31.25;

export class FixQeq extends Fix {
  readonly style: string;
  private readonly kind: QeqKind;
  private readonly nEvery: number;
  private readonly cutoff: number;
  private readonly tol: number;
  private readonly maxiter: number;
  private readonly qfile: string;
  private warn: boolean;
  private qdamp: number;
  private qstep: number;
  private params: Params | null = null;
  private iterations = 0;
  scalarFlag = true;
  extscalar = 0;

  constructor(sys: System, id: string, group: string, args: string[], kind: QeqKind) {
    super(sys, id, group, args);
    this.kind = kind;
    this.style = `qeq/${kind}`;
    const a = args;
    if (a.length < 5) {
      throw new StyleError(`Illegal fix ${this.style} command: expected "Nevery cutoff tolerance maxiter qfile [keyword value ...]"`);
    }
    this.nEvery = intArg(a[0], 'Nevery');
    this.cutoff = numArg(a[1], 'cutoff');
    this.tol = numArg(a[2], 'tolerance');
    this.maxiter = intArg(a[3], 'maxiter');
    this.qfile = a[4];
    if (!(this.nEvery > 0)) throw new StyleError(`fix ${this.style}: Nevery must be > 0`);
    if (!(this.cutoff > 0)) throw new StyleError(`fix ${this.style}: cutoff must be > 0`);
    if (!(this.tol > 0)) throw new StyleError(`fix ${this.style}: tolerance must be > 0`);
    if (!(this.maxiter > 0)) throw new StyleError(`fix ${this.style}: maxiter must be > 0`);
    // qdamp/qstep are documented for qeq/dynamic and qeq/fire only; their
    // measured defaults are qdamp 0.1 and qstep 0.02 (dynamic) / 0.2 (fire).
    this.qdamp = 0.1;
    this.qstep = kind === 'fire' ? 0.2 : 0.02;
    this.warn = true;
    const damped = kind === 'dynamic' || kind === 'fire';
    for (let k = 5; k < a.length;) {
      const key = a[k];
      const val = a[k + 1];
      if (key === 'warn') {
        if (val !== 'yes' && val !== 'no') throw new StyleError(`fix ${this.style}: warn must be yes or no`);
        this.warn = val === 'yes';
        k += 2;
      } else if (damped && key === 'qdamp') {
        this.qdamp = numArg(val, 'qdamp');
        k += 2;
      } else if (damped && key === 'qstep') {
        this.qstep = numArg(val, 'qstep');
        k += 2;
      } else {
        throw new StyleError(`fix ${this.style}: keyword '${key}' is not supported (${damped ? 'qdamp, qstep, warn' : 'warn'})`);
      }
    }
    if (damped && !(this.qstep > 0)) throw new StyleError(`fix ${this.style}: qstep must be > 0`);
    // The qfile may be the literal word reaxff / coul/streitz / coul/ctip to
    // pull the parameters from an active pair style; the engine has none.
    if (qfileSpecial(this.qfile)) {
      throw new StyleError(`fix ${this.style}: qfile '${this.qfile}' (parameters from a pair style) is not supported by the browser engine`);
    }
    this.parseQfile(sys.readFile(this.qfile));
  }

  private parseQfile(text: string): void {
    const n = this.sys.state.ntypes;
    const chi = new Float64Array(n + 1).fill(NaN);
    const eta = new Float64Array(n + 1).fill(NaN);
    const gamma = new Float64Array(n + 1);
    const lines = text.split('\n');
    for (let ln = 0; ln < lines.length; ln++) {
      const line = lines[ln].replace(/#.*/, '').trim();
      if (!line) continue;
      const w = line.split(/\s+/);
      if (w.length < 6) {
        throw new StyleError(`Invalid param file for fix ${this.style} (${this.qfile}) line ${ln + 1}: expected "type chi eta gamma zeta qcore"`);
      }
      let lo = 0, hi = 0;
      try {
        [lo, hi] = typeBounds(w[0], n);
      } catch (e) {
        throw new StyleError(`Invalid param file for fix ${this.style} (${this.qfile}) line ${ln + 1}: ${(e as Error).message}`);
      }
      const vals = w.slice(1, 6).map((s, i) => numArg(s, `qfile column ${i + 1}`));
      for (let t = lo; t <= hi; t++) {
        chi[t] = vals[0];
        eta[t] = vals[1];
        gamma[t] = vals[2];
      }
    }
    for (let t = 1; t <= n; t++) {
      if (Number.isNaN(chi[t]) || Number.isNaN(eta[t])) {
        throw new StyleError(`Invalid param file for fix ${this.style} (${this.qfile}): no parameters for atom type ${t}`);
      }
    }
    this.params = { chi, eta, gamma };
  }

  /** Charges must be equilibrated before the first force computation of a run. */
  setup(): void { this.solve(); }

  /** fix_qeq.html: Nevery = "perform charge equilibration every this many steps". */
  preForce(): void {
    if (this.sys.state.step % this.nEvery === 0) this.solve();
  }

  computeScalar(): number { return this.iterations; }

  private solve(): void {
    const s = this.sys.state;
    const p = this.params!;
    const bit = this.groupBit;
    const shielded = this.kind === 'shielded';
    // group atoms (owned)
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) idx.push(i);
    const ng = idx.length;
    if (ng === 0) { this.iterations = 0; return; }

    // pair interactions within the cutoff: point 1/r, shielded 14.4*(r^3+g^-3)^-1/3*Taper
    const pairs: { i: number; j: number; int: number }[] = [];
    const d = [0, 0, 0];
    const cut2 = this.cutoff * this.cutoff;
    for (let a = 0; a < ng; a++) {
      const i = idx[a];
      for (let b = a + 1; b < ng; b++) {
        const j = idx[b];
        d[0] = s.x[3 * i] - s.x[3 * j];
        d[1] = s.x[3 * i + 1] - s.x[3 * j + 1];
        d[2] = s.x[3 * i + 2] - s.x[3 * j + 2];
        this.sys.geom.minimumImage(d);
        const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (!(r2 > 0) || r2 >= cut2) continue;
        const int = this.interaction(s, p, i, j, Math.sqrt(r2), shielded);
        if (int !== 0) pairs.push({ i, j, int });
      }
    }

    // map owned index -> matrix row
    const row = new Int32Array(s.n).fill(-1);
    for (let a = 0; a < ng; a++) row[idx[a]] = a;

    // eta diagonal
    const etaRow = new Float64Array(ng);
    for (let a = 0; a < ng; a++) etaRow[a] = p.eta[s.type[idx[a]]];

    // fixed (non-group) charges contribute a constant to each group atom's gradient
    const fixedTerm = new Float64Array(ng);
    if (ng < s.n) {
      const inGroup = (i: number) => (s.mask[i] & bit) !== 0;
      for (let a = 0; a < ng; a++) {
        const i = idx[a];
        for (let k = 0; k < s.n; k++) {
          if (inGroup(k)) continue;
          d[0] = s.x[3 * i] - s.x[3 * k];
          d[1] = s.x[3 * i + 1] - s.x[3 * k + 1];
          d[2] = s.x[3 * i + 2] - s.x[3 * k + 2];
          this.sys.geom.minimumImage(d);
          const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
          if (!(r2 > 0) || r2 >= cut2) continue;
          fixedTerm[a] += this.interaction(s, p, i, k, Math.sqrt(r2), shielded) * s.q[k];
        }
      }
    }

    let q: Float64Array;
    if (this.kind === 'point' || this.kind === 'shielded') {
      q = this.solveMatrix(s, p, idx, pairs, row, etaRow, fixedTerm, ng);
    } else {
      const gradient = (qq: Float64Array, out: Float64Array): void => {
        for (let a = 0; a < ng; a++) out[a] = p.chi[s.type[idx[a]]] + etaRow[a] * qq[a] + fixedTerm[a];
        for (const { i, j, int } of pairs) {
          const a = row[i], c = row[j];
          out[a] += int * qq[c];
          out[c] += int * qq[a];
        }
        let mean = 0;
        for (let a = 0; a < ng; a++) mean += out[a];
        mean /= ng;
        for (let a = 0; a < ng; a++) out[a] = -(out[a] - mean);
      };
      const q0 = new Float64Array(ng);
      for (let a = 0; a < ng; a++) q0[a] = s.q[idx[a]];
      q = this.kind === 'dynamic' ? this.solveDynamic(q0, gradient, ng) : this.solveFire(q0, gradient, ng);
    }

    // initial charges should already be neutral (doc note)
    let sum0 = 0;
    for (const i of idx) sum0 += s.q[i];
    if (this.warn && Math.abs(sum0) > 1e-8) {
      this.sys.warn('Fix qeq: initial charge of atoms in the fix group is not zero; charge equilibration enforces neutrality');
    }

    for (let a = 0; a < ng; a++) s.q[idx[a]] = q[a];
    if (this.sys.ff.kspace) {
      // A kspace style caches the charge sums its self/background energy uses
      // (the engine refreshes them in setupNeighbors); a fix that changes
      // charges mid-run must re-run it or the long-range energy goes stale.
      // Not reproduced (measured with native LAMMPS, black box, coul/long +
      // pppm): at the setup of a run native's sums keep the values of the
      // kspace initialisation, i.e. the charges before this solve, so the
      // step-0 pe of the first run is -6.42425038262653 where the engine (and
      // native's next run, re-initialised) gives -6.42424993275209; during the
      // steps the sums follow the charges in both.
      this.sys.setupNeighbors();
    } else {
      this.sys.nb.refreshCharges(s);
    }
    this.sys.bump();
  }

  /** Point (qeq/point, qeq/dynamic, qeq/fire) or shielded (qeq/shielded) pair interaction. */
  private interaction(s: System['state'], p: Params, i: number, j: number, r: number, shielded: boolean): number {
    if (!shielded) return 1 / r;
    const g = Math.sqrt(p.gamma[s.type[i]] * p.gamma[s.type[j]]);
    return g > 0 ? COUL * (r * r * r + 1 / (g * g * g)) ** (-1 / 3) * taper(r / this.cutoff) : 0;
  }

  /** qeq/point and qeq/shielded: matrix inversion (Jacobi-preconditioned CG; measured with native LAMMPS, black box). */
  private solveMatrix(
    s: System['state'], p: Params, idx: number[], pairs: { i: number; j: number; int: number }[],
    row: Int32Array, etaRow: Float64Array, fixedTerm: Float64Array, ng: number,
  ): Float64Array {
    // rhs: -chi_i - sum over non-group neighbors of H_ij * q_j (fixed charges)
    const n1 = ng + 1;
    const b = new Float64Array(n1);
    for (let a = 0; a < ng; a++) b[a] = -p.chi[s.type[idx[a]]] - fixedTerm[a];

    const diag = new Float64Array(n1);
    for (let a = 0; a < ng; a++) diag[a] = etaRow[a] || 1;
    diag[ng] = 1;

    const matvec = (x: Float64Array, out: Float64Array): void => {
      for (let a = 0; a < n1; a++) out[a] = 0;
      for (let a = 0; a < ng; a++) out[a] += etaRow[a] * x[a];
      for (const { i, j, int } of pairs) {
        const a = row[i], c = row[j];
        out[a] += int * x[c];
        out[c] += int * x[a];
      }
      for (let a = 0; a < ng; a++) out[a] += x[ng];
      let sum = 0;
      for (let a = 0; a < ng; a++) sum += x[a];
      out[ng] += sum;
    };

    // Jacobi-preconditioned conjugate gradient (measured with native LAMMPS).
    const x = new Float64Array(n1);
    const r = new Float64Array(b);
    const z = new Float64Array(n1);
    const pp = new Float64Array(n1);
    const Ap = new Float64Array(n1);
    let bn = 0;
    for (let a = 0; a < n1; a++) bn = Math.max(bn, Math.abs(b[a]));
    if (bn === 0) bn = 1;
    for (let a = 0; a < n1; a++) { z[a] = r[a] / diag[a]; pp[a] = z[a]; }
    let rz = 0;
    for (let a = 0; a < n1; a++) rz += r[a] * z[a];
    let iter = 0;
    for (iter = 1; iter <= this.maxiter; iter++) {
      matvec(pp, Ap);
      let pAp = 0;
      for (let a = 0; a < n1; a++) pAp += pp[a] * Ap[a];
      const alpha = rz / pAp;
      let rmax = 0;
      for (let a = 0; a < n1; a++) {
        x[a] += alpha * pp[a];
        r[a] -= alpha * Ap[a];
        if (Math.abs(r[a]) > rmax) rmax = Math.abs(r[a]);
      }
      if (rmax / bn < this.tol) break;
      for (let a = 0; a < n1; a++) z[a] = r[a] / diag[a];
      let rz2 = 0;
      for (let a = 0; a < n1; a++) rz2 += r[a] * z[a];
      const beta = rz2 / rz;
      for (let a = 0; a < n1; a++) pp[a] = z[a] + beta * pp[a];
      rz = rz2;
    }
    this.iterations = iter > this.maxiter ? this.maxiter : iter;

    if (this.iterations >= this.maxiter && this.warn) {
      // fix_qeq.html: warn when not equilibrated within tolerance by maxiter
      const qmax = maxAbs(x.subarray(0, ng));
      const rmax = maxAbs(r);
      if (rmax / bn >= this.tol) {
        this.sys.warn(`Fix ${this.style} charge equilibration did not converge; qmax = ${qmax} (iteration ${this.maxiter}, maxiter reached)`);
      }
    }
    return x.subarray(0, ng);
  }

  /**
   * qeq/dynamic: damped extended-Lagrangian charge dynamics.  The charge force
   * is f = -(gradient - mean) (neutrality constraint); iterates
   * d_0 = B f(q_0), q_{n+1} = q_n + d_n, d_{n+1} = gamma^2 d_n + B f(q_{n+1}),
   * gamma = 1 - qdamp, B = 31.25 qstep^2 (2 - qdamp); stop when max|f| < tolerance.
   * Measured with native LAMMPS (black box): this reproduces the native charges
   * to machine precision, and the native max|f| criterion stops at the same step.
   */
  private solveDynamic(q0: Float64Array, gradient: (q: Float64Array, out: Float64Array) => void, ng: number): Float64Array {
    const gamma = 1 - this.qdamp;
    const B = DYNAMIC_SCALE * this.qstep * this.qstep * (2 - this.qdamp);
    const q = Float64Array.from(q0);
    const dq = new Float64Array(ng);
    const f = new Float64Array(ng);
    gradient(q, f);
    let fm = meanAbs(f);
    let iters = 0;
    // native performs at most maxiter - 1 updates (maxiter 1 leaves the charges unchanged; measured)
    for (; iters < this.maxiter - 1 && fm >= this.tol; iters++) {
      if (iters === 0) {
        for (let a = 0; a < ng; a++) dq[a] = B * f[a];
      } else {
        for (let a = 0; a < ng; a++) dq[a] = gamma * gamma * dq[a] + B * f[a];
      }
      for (let a = 0; a < ng; a++) q[a] += dq[a];
      gradient(q, f);
      fm = meanAbs(f);
    }
    this.iterations = iters;
    if (fm >= this.tol && this.warn) {
      this.sys.warn(`Fix ${this.style} charge equilibration did not converge (iteration ${this.maxiter}, maxiter reached)`);
    }
    return q;
  }

  /**
   * qeq/fire, measured with native LAMMPS (black box): with maxiter capped at 1..80 the native charges of
   * an irregular 5-atom cluster after every iteration are reproduced to 1e-16 by this damped dynamics on
   * the same charge force f (FIRE-like, Bitzek et al., PRL 97, 170201 (2006), with fixed constants):
   *   v' = v + dt_prev f_prev           (f_prev = the force of the previous iteration; f at the first)
   *   if f . v' <= 0: v' = 0, dt_base = dt_base / 2
   *   v  = 0.2 v' + 0.8 |v'| f / |f|
   *   dt = dt_base, reduced so that max |dt v| <= 0.1;  q += dt v;  dt_prev = dt
   * with dt_base = qstep / 2 at the start (qdamp has no effect).  It stops when mean |f| < tolerance;
   * the fix scalar is then the number of updates plus one (native counts 21, 24, 45, 47, 18, 19, 30, 54
   * and 98 iterations at tolerances 1e-1 ... 1e-5 for qstep 0.2 and 0.4 are reproduced).
   */
  private solveFire(q0: Float64Array, gradient: (q: Float64Array, out: Float64Array) => void, ng: number): Float64Array {
    const q = Float64Array.from(q0);
    const v = new Float64Array(ng);
    const vp = new Float64Array(ng);
    const f = new Float64Array(ng);
    const fPrev = new Float64Array(ng);
    let dtBase = this.qstep / 2;
    let dtPrev = dtBase;
    gradient(q, f);
    fPrev.set(f);
    let fm = meanAbs(f);
    let updates = 0;
    for (; updates < this.maxiter - 1 && fm >= this.tol; updates++) {
      let P = 0;
      for (let a = 0; a < ng; a++) { vp[a] = v[a] + dtPrev * fPrev[a]; P += f[a] * vp[a]; }
      if (P <= 0) { vp.fill(0); dtBase *= 0.5; }
      let nv = 0, nf = 0;
      for (let a = 0; a < ng; a++) { nv += vp[a] * vp[a]; nf += f[a] * f[a]; }
      nv = Math.sqrt(nv); nf = Math.sqrt(nf);
      let vmax = 0;
      for (let a = 0; a < ng; a++) {
        v[a] = nv > 0 && nf > 0 ? 0.2 * vp[a] + (0.8 * nv * f[a]) / nf : vp[a];
        if (Math.abs(v[a]) > vmax) vmax = Math.abs(v[a]);
      }
      const dt = dtBase * vmax > FIRE_DQMAX ? FIRE_DQMAX / vmax : dtBase;
      for (let a = 0; a < ng; a++) q[a] += dt * v[a];
      fPrev.set(f);
      dtPrev = dt;
      gradient(q, f);
      fm = meanAbs(f);
    }
    this.iterations = fm < this.tol ? updates + 1 : updates;
    if (fm >= this.tol && this.warn) {
      this.sys.warn(`Fix ${this.style} charge equilibration did not converge (iteration ${this.maxiter}, maxiter reached)`);
    }
    return q;
  }
}

/**
 * Convergence measure of qeq/dynamic and qeq/fire: the mean |f| over the fix group. Measured with native
 * LAMMPS (black box): on a 5-atom irregular cluster the native iteration counts at tolerances 1e-1 ... 1e-5
 * (28, 40, 46, 65, 71, 77, 95, 108, 114) are reproduced only by mean |f| < tolerance (not max |f|, the RMS,
 * the norm or the sum).
 */
/** Largest charge change of one qeq/fire update (measured, see solveFire; independent of qstep). */
const FIRE_DQMAX = 0.1;

const meanAbs = (f: Float64Array): number => {
  let s = 0;
  for (let k = 0; k < f.length; k++) s += Math.abs(f[k]);
  return f.length ? s / f.length : 0;
};

const maxAbs = (a: Float64Array): number => {
  let m = 0;
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]);
  return m;
};

const numArg = (w: string, what: string): number => {
  const v = Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`fix qeq: expected a number for ${what}, got '${w}'`);
  return v;
};

const intArg = (w: string, what: string): number => {
  if (!/^\d+$/.test(w)) throw new StyleError(`fix qeq: expected an integer for ${what}, got '${w}'`);
  return Number(w);
};

const qfileSpecial = (name: string): boolean => name === 'reaxff' || name === 'coul/streitz' || name === 'coul/ctip';
