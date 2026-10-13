import { Fix } from './fix';
import { StyleError, typeBounds } from '../force/types';
import { erfcExact } from '../force/erfc';
import { PairCoulStreitz, pointUval, slaterS } from '../force/pair/coul_streitz';
import type { System } from '../system';
import { gplHook } from '../gplHooks';

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
 * qeq/slater (docs.lammps.org/fix_qeq.html, section qeq/slater): the page describes the Slater 1s charge
 * densities, the parameters chi, eta, zeta and qcore from the qfile, the coul/streitz qfile word (parameters taken from
 * an active coul/streitz pair style) and the keyword alpha, documented as the Slater type orbital exponent.
 * Measured with native LAMMPS (black box), this contradicts that description: alpha is the damping of the
 * Wolf-type self term and erfc(alpha r)/r pair term, and the zeta of the file is used unchanged
 * (the default alpha is 0.2 when the keyword is absent). The model that reproduces native to 1e-15:
 *   E(q) = sum_i [chi_i q_i + 1/2 eta_i q_i^2 + C selfW q_i^2]
 *        + sum_{i<j, r<rc} [C q_i q_j erfc(alpha r)/r + C G_ij(r)],
 *   G_ij = (q_i q_j - Z_i q_j - Z_j q_i) S_ij(r) + Z_i q_j U_j(r) + Z_j q_i U_i(r)
 * with C = 14.399645, selfW = -(erfc(alpha rc)/(2 rc) + alpha/sqrt(pi)), S and U as in pair coul/streitz,
 * eta = J of the pair style (or the qfile eta), rc = the fix cutoff (it sets the pair range and selfW;
 * a pair cutoff of the pair style does not), and sum q = 0 over the group. Measured: no shifts of the
 * erfc term and of G (the coul/streitz wolf shift is absent), no kspace term with pair ewald (real space only).
 * Keyword dsf 0.0 (Fennell-Gezelter form, taper width zero) adds (r - rc) K to the erfc term with
 * K = erfc(alpha rc)/rc^2 + 2 alpha exp(-alpha^2 rc^2)/(sqrt(pi) rc); it matches native to 1e-16.
 * Keyword wolf and dsf with a taper width above zero are not reproduced and throw a StyleError
 * (the wolf keyword charges differ from every form tried by about 2e-6 at 2 A).
 * Solver (measured with native LAMMPS, black box): two conjugate-gradient solves with the Jacobi
 * preconditioner 1/eta (not the shifted diagonal), H s = -chi_eff and H t = -1, both from zero, the
 * stop test sqrt(r.p)/|b|_2 > tolerance inside the loop test i < maxiter (so maxiter 1 leaves s = t = 0, a NaN
 * charge as native gives), and q = s - (sum s / sum t) t. It reproduces native charges to 1e-15 at
 * tolerances 1e-2 ... 1e-6 and maxiter 3 ... 100 on a 40-atom cluster.
 *
 * Unsupported qeq styles and qfile keywords throw a StyleError naming them.
 */

/** pair_style reaxff as qeq/shielded sees it: its extract() of chi, eta, gamma (duck-typed; the pair lives in engine-gpl). */
interface ReaxQeqSource {
  qeqParameters(ntypes: number): { chi: Float64Array; eta: Float64Array; gamma: Float64Array };
  attachQeqFix(): void;
}
const isReaxQeqSource = (o: unknown): o is ReaxQeqSource => typeof (o as ReaxQeqSource | null)?.qeqParameters === 'function';

type QeqKind = 'point' | 'shielded' | 'dynamic' | 'fire' | 'slater';

/** The charge history of the qeq/slater solves (src/engine-gpl/qeq/history.ts, the lazy HOOKS entry 'qeq/slater'). */
export interface QeqHistory {
  readonly n: number;
  initialGuesses(idx: number[]): { s: Float64Array; t: Float64Array };
  shift(idx: number[], sx: Float64Array, tx: Float64Array): void;
  matvecCount(sIt: number, tIt: number): number;
}
export interface QeqHistoryHook {
  create(n: number, q: Float64Array): QeqHistory;
}

interface Params {
  chi: Float64Array;
  eta: Float64Array;
  gamma: Float64Array;
  /** Slater exponent and nucleus charge (qeq/slater only; from the qfile columns zeta and qcore). */
  zeta: Float64Array;
  qcore: Float64Array;
}

const COUL = 14.4; // hard-coded Coulomb constant of fix qeq (measured with native LAMMPS, black box)

/** 7th-order taper used by the shielded interaction (measured with native LAMMPS, black box). */
const taper = (x: number): number => {
  if (x >= 1) return 0;
  return 1 - 35 * x * x * x * x + 84 * x ** 5 - 70 * x ** 6 + 20 * x ** 7;
};

/** Constant of the qeq/dynamic charge dynamics (measured with native LAMMPS, black box). */
const DYNAMIC_SCALE = 31.25;

/**
 * erfc for the qeq/slater matrix (same values as erfcExact to 2.5e-16 relative). erfcExact's continued fraction runs
 * 400 terms for x >= 2, which made every image of the periodic matrix cost microseconds; 60 terms converge to the
 * same double (checked on x in [2, 8] against 400 terms).
 */
const erfcSlater = (x: number): number => {
  if (x < 2) return erfcExact(x);
  let f = x;
  for (let k = 60; k >= 1; k--) f = x + (k / 2) / f;
  return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
};

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
  /** qeq/slater: wolf-type damping alpha (keyword alpha, default 0.2), dsf 0.0 form, qfile coul/streitz. */
  private alpha = 0.2;
  private dsf = false;
  private fromPair = false;
  /** qeq/shielded with qfile reaxff: chi, eta, gamma of the active pair_style reaxff, read at each solve. */
  private fromReax = false;
  /** qeq/slater: the charge history of the CG solves (created at the first solve, see QeqHistory). */
  private history: QeqHistory | null = null;
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
      } else if (kind === 'slater' && key === 'alpha') {
        this.alpha = numArg(val, 'alpha');
        if (!(this.alpha > 0)) throw new StyleError(`fix ${this.style}: alpha must be > 0`);
        k += 2;
        // fix_qeq.html: "wolf value = width of taper to terminate Coulomb integrals for the Wolf summation"
        // (only taper width zero is reproduced here; see the header comment)
        const sub = a[k];
        if (sub === 'wolf' || sub === 'dsf') {
          const width = numArg(a[k + 1], `${sub} taper width`);
          if (sub === 'wolf') {
            throw new StyleError(`fix ${this.style}: keyword wolf (Wolf-summation charges) is not reproduced by the browser engine`);
          }
          if (width !== 0) {
            throw new StyleError(`fix ${this.style}: keyword dsf with taper width ${width} is not reproduced (only dsf 0.0)`);
          }
          this.dsf = true;
          k += 2;
        }
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
    if (kind === 'slater' && this.qfile === 'coul/streitz') {
      // the parameters come from the active pair style (resolved at each solve)
      this.fromPair = true;
    } else {
      if (kind === 'shielded' && this.qfile === 'reaxff') {
        // fix_qeq_shielded.cpp (b332f28) extract_reax: "pair->extract("chi", tmp)", "eta", "gamma" of pair reaxff
        this.fromReax = true;
        // the pair must know of the fix (pair reaxff checkqeq yes, pair_reaxff.cpp init_style)
        this.reaxSource()?.attachQeqFix();
      } else {
        if (qfileSpecial(this.qfile)) {
          throw new StyleError(`fix ${this.style}: qfile '${this.qfile}' (parameters from a pair style) is not supported by the browser engine`);
        }
        this.parseQfile(sys.readFile(this.qfile));
      }
    }

  }

  private parseQfile(text: string): void {
    const n = this.sys.state.ntypes;
    const chi = new Float64Array(n + 1).fill(NaN);
    const eta = new Float64Array(n + 1).fill(NaN);
    const gamma = new Float64Array(n + 1);
    const zeta = new Float64Array(n + 1);
    const qcore = new Float64Array(n + 1);
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
        zeta[t] = vals[3];
        qcore[t] = vals[4];
      }
    }
    for (let t = 1; t <= n; t++) {
      if (Number.isNaN(chi[t]) || Number.isNaN(eta[t])) {
        throw new StyleError(`Invalid param file for fix ${this.style} (${this.qfile}): no parameters for atom type ${t}`);
      }
    }
    this.params = { chi, eta, gamma, zeta, qcore };
  }

  /**
   * qeq/shielded with qfile reaxff: chi, eta and gamma of each type from the pair_style reaxff (its extract()). The pair
   * is found directly or among the sub-styles of pair hybrid. The fix is attached to the pair (checkqeq yes requires it).
   */
  private reaxSource(): ReaxQeqSource | null {
    const pair = this.sys.ff.pair as unknown as { subs?: { style: unknown }[] } | null;
    let src: ReaxQeqSource | null = isReaxQeqSource(pair) ? pair : null;
    for (const sub of pair?.subs ?? []) if (!src && isReaxQeqSource(sub.style)) src = sub.style;
    return src;
  }

  private reaxParams(): Params {
    const n = this.sys.state.ntypes;
    const src = this.reaxSource();
    if (!src) throw new StyleError('No reaxff pair style for fix qeq/shielded');
    const q = src.qeqParameters(n);
    const zero = new Float64Array(n + 1);
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const t = s.type[i];
      if (q.chi[t] === 0 && q.eta[t] === 0 && q.gamma[t] === 0) throw new StyleError(`No QEq parameters for atom type ${t} provided by pair reaxff`);
    }
    return { chi: q.chi, eta: q.eta, gamma: q.gamma, zeta: zero, qcore: zero };
  }

  /** qeq/slater: parameters of the active coul/streitz pair style (qfile coul/streitz); chi, eta = J, zeta, qcore = Z. */
  private slaterParams(): Params {
    if (!this.fromPair) return this.params!;
    const n = this.sys.state.ntypes;
    const pair = this.sys.ff.pair;
    let cs: PairCoulStreitz | null = null;
    if (pair instanceof PairCoulStreitz) cs = pair;
    else {
      const subs = (pair as unknown as { subs?: { style: unknown }[] } | null)?.subs ?? [];
      for (const sub of subs) if (sub.style instanceof PairCoulStreitz) cs = sub.style;
    }
    if (!cs) throw new StyleError(`fix ${this.style}: qfile coul/streitz needs an active pair_style coul/streitz`);
    const chi = new Float64Array(n + 1), eta = new Float64Array(n + 1), gamma = new Float64Array(n + 1);
    const zeta = new Float64Array(n + 1), qcore = new Float64Array(n + 1);
    for (let t = 1; t <= n; t++) {
      const tp = cs.typeParams(t);
      chi[t] = tp.chi; eta[t] = tp.J; gamma[t] = tp.gamma; zeta[t] = tp.zeta; qcore[t] = tp.Z;
    }
    return { chi, eta, gamma, zeta, qcore };
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
    const p = this.kind === 'slater' ? this.slaterParams() : this.fromReax ? this.reaxParams() : this.params!;
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
    // qeq/slater builds its own image list in solveSlater (this O(ng^2) loop would only feed the other kinds)
    for (let a = 0; a < (this.kind === 'slater' ? 0 : ng); a++) {
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
    if (ng < s.n && this.kind !== 'slater') {
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
    if (this.kind === 'slater') {
      q = this.solveSlater(s, p, idx, ng);
    } else if (this.kind === 'point' || this.kind === 'shielded') {
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

  /**
   * qeq/slater (see the header comment): H_ii = eta_i + 2 C selfW, H_ij = C (cc_ij(r) + S_ij(r)), with
   * cc = erfc(alpha r)/r (dsf 0.0: minus erfc(alpha rc)/rc plus (r - rc) K); the right-hand side is
   * -chi_eff, chi_eff_i = chi_i + sum_j C [q_j (cc + S) + Z_j (U_i - S)] over the group partners j and
   * the fixed atoms. The Jacobi preconditioner is 1/eta_i.
   */
  private solveSlater(s: System['state'], p: Params, idx: number[], ng: number): Float64Array {
    const C = 14.399645; // qqrd2e of units metal, as the pair style uses it (measured with native LAMMPS, black box)
    const a = this.alpha, rc = this.cutoff;
    const erc = erfcSlater(a * rc);
    const selfW = -(erc / (2 * rc) + a / Math.sqrt(Math.PI));
    const kd = this.dsf ? erc / (rc * rc) + (2 * a / Math.sqrt(Math.PI)) * Math.exp(-a * a * rc * rc) / rc : 0;
    const cc = (r: number): number => {
      const e = erfcSlater(a * r) / r;
      return this.dsf ? e - erc / rc + (r - rc) * kd : e;
    };
    const cut2 = rc * rc;
    const bit = this.groupBit;
    const row = new Int32Array(s.n).fill(-1);
    for (let k = 0; k < ng; k++) row[idx[k]] = k;
    for (const i of idx) {
      if (!(p.zeta[s.type[i]] > 0)) throw new StyleError(`fix ${this.style}: zeta of type ${s.type[i]} must be > 0`);
      if (p.eta[s.type[i]] === 0) throw new StyleError(`fix ${this.style}: eta of type ${s.type[i]} is zero (preconditioner 1/eta)`);
    }
    const diag = new Float64Array(ng);
    const chi = new Float64Array(ng);
    const precond = new Float64Array(ng);
    for (let k = 0; k < ng; k++) {
      const t = s.type[idx[k]];
      diag[k] = p.eta[t] + 2 * C * selfW;
      precond[k] = 1 / p.eta[t];
      chi[k] = p.chi[t];
    }
    // the off-diagonal pairs (row a, row c, H) as typed arrays: the CG matvec runs over them every iteration
    const pairA: number[] = [], pairC: number[] = [], pairH: number[] = [];
    // Every periodic image within the cutoff counts (the pair style sums its ghost atoms the same way), so the
    // shifts are enumerated, not only the minimum image. A pair of an atom with its own image is counted once
    // (shift s and -s are the same pair); its quadratic term 2 C (cc + S) q^2 goes on the diagonal.
    // Orthogonal boxes: per dimension only the integer shifts n with |dx - n L| < rc are visited (the range is
    // computed from dx, so a pair costs a few integer tests instead of the full (2 nmax + 1)^3 image list).
    // Triclinic boxes keep the enumeration of the image list.
    const box = this.sys.geom as unknown as { lx: number; ly: number; lz: number; xy: number; xz: number; yz: number; periodic: boolean[] };
    const ortho = box.xy === 0 && box.xz === 0 && box.yz === 0;
    const Ld = [box.lx, box.ly, box.lz];
    const nmax = [0, 1, 2].map((dd) => (box.periodic[dd] ? Math.ceil(rc / Ld[dd]) + 1 : 0));
    const shifts: number[][] = [];
    if (!ortho) {
      const ax = [box.lx, 0, 0], bx = [box.xy, box.ly, 0], cx = [box.xz, box.yz, box.lz];
      for (let nx = -nmax[0]; nx <= nmax[0]; nx++)
        for (let ny = -nmax[1]; ny <= nmax[1]; ny++)
          for (let nz = -nmax[2]; nz <= nmax[2]; nz++) {
            const sh = [nx * ax[0] + ny * bx[0] + nz * cx[0], nx * ax[1] + ny * bx[1] + nz * cx[1], nx * ax[2] + ny * bx[2] + nz * cx[2]];
            shifts.push([sh[0], sh[1], sh[2], nx, ny, nz]);
          }
    }
    const positive = (nx: number, ny: number, nz: number): boolean => (nx > 0) || (nx === 0 && (ny > 0 || (ny === 0 && nz > 0)));
    const d = [0, 0, 0];
    // images of one (u, v) pair: the shift vector and its integer index, collected before the pair body runs
    // per dimension at most 2 ceil(rc / L) + 2 integer shifts can be within the cutoff (a small box needs more than 3)
    const perDim = [0, 1, 2].map((dd) => (box.periodic[dd] ? 2 * Math.ceil(rc / Ld[dd]) + 2 : 1));
    const maxImg = ortho ? perDim[0] * perDim[1] * perDim[2] : shifts.length;
    const iSx = new Float64Array(maxImg), iSy = new Float64Array(maxImg), iSz = new Float64Array(maxImg);
    const iN = new Int32Array(3 * maxImg);
    const dimRange = (dv: number, dd: number, out: Int32Array, o: number): number => {
      if (!box.periodic[dd]) { out[o] = 0; out[o + 1] = 0; return 1; }
      out[o] = Math.floor((dv - rc) / Ld[dd]);
      out[o + 1] = Math.ceil((dv + rc) / Ld[dd]);
      return 1;
    };
    const rng = new Int32Array(6);
    for (let u = 0; u < ng; u++) {
      const i = idx[u];
      const ti = s.type[i];
      for (let v = 0; v < s.n; v++) {
        const inGroup = (s.mask[v] & bit) !== 0;
        if (inGroup && v < i) continue; // each group pair once
        const tv = s.type[v];
        // collect the images of this pair that lie within the cutoff range
        let ni = 0;
        if (ortho) {
          dimRange(s.x[3 * i] - s.x[3 * v], 0, rng, 0);
          dimRange(s.x[3 * i + 1] - s.x[3 * v + 1], 1, rng, 2);
          dimRange(s.x[3 * i + 2] - s.x[3 * v + 2], 2, rng, 4);
          for (let nx = rng[0]; nx <= rng[1]; nx++)
            for (let ny = rng[2]; ny <= rng[3]; ny++)
              for (let nz = rng[4]; nz <= rng[5]; nz++) {
                if (v === i && !positive(nx, ny, nz)) continue;
                iSx[ni] = nx * Ld[0]; iSy[ni] = ny * Ld[1]; iSz[ni] = nz * Ld[2];
                iN[3 * ni] = nx; iN[3 * ni + 1] = ny; iN[3 * ni + 2] = nz;
                ni++;
              }
        } else {
          for (const [shx, shy, shz, nx, ny, nz] of shifts) {
            if (v === i && !positive(nx, ny, nz)) continue;
            iSx[ni] = shx; iSy[ni] = shy; iSz[ni] = shz;
            iN[3 * ni] = nx; iN[3 * ni + 1] = ny; iN[3 * ni + 2] = nz;
            ni++;
          }
        }
        const zi = p.zeta[ti], zv = p.zeta[tv];
        for (let k = 0; k < ni; k++) {
          d[0] = s.x[3 * i] - s.x[3 * v] - iSx[k];
          d[1] = s.x[3 * i + 1] - s.x[3 * v + 1] - iSy[k];
          d[2] = s.x[3 * i + 2] - s.x[3 * v + 2] - iSz[k];
          const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
          if (!(r2 > 0) || r2 >= cut2) continue;
          const r = Math.sqrt(r2);
          const S = slaterS(r, zi, zv);
          const Ui = pointUval(r, zi);
          const Uv = pointUval(r, zv);
          const ccr = cc(r);
          const H = C * (ccr + S);
          if (v === i) {
            diag[u] += 2 * H;
            chi[u] += 2 * C * p.qcore[ti] * (Ui - S);
          } else if (inGroup) {
            const w = row[v];
            chi[u] += C * p.qcore[tv] * (Ui - S);
            chi[w] += C * p.qcore[ti] * (Uv - S);
            if (H !== 0) { pairA.push(u); pairC.push(w); pairH.push(H); }
          } else {
            // fixed atom: its charge enters the right-hand side of group atom u
            chi[u] += C * (s.q[v] * (ccr + S) + p.qcore[tv] * (Ui - S));
          }
        }
      }
    }
    const npair = pairA.length;
    const PA = Int32Array.from(pairA), PC = Int32Array.from(pairC), PH = Float64Array.from(pairH);
    const matvec = (x: Float64Array, out: Float64Array): void => {
      for (let k = 0; k < ng; k++) out[k] = diag[k] * x[k];
      for (let e = 0; e < npair; e++) {
        const u = PA[e], w = PC[e], h = PH[e];
        out[u] += h * x[w];
        out[w] += h * x[u];
      }
    };
    // reaxff-style preconditioned CG from zero (measured, see the header comment)
    const solveOne = (b: Float64Array, x0: Float64Array | null): { x: Float64Array; it: number; ok: boolean } => {
      const x = x0 ? Float64Array.from(x0) : new Float64Array(ng);
      const hx = new Float64Array(ng);
      matvec(x, hx);
      const r = new Float64Array(ng);
      for (let k = 0; k < ng; k++) r[k] = b[k] - hx[k];
      let bn = 0;
      for (let k = 0; k < ng; k++) bn += b[k] * b[k];
      bn = Math.sqrt(bn);
      const pre = new Float64Array(ng);
      const dv = new Float64Array(ng);
      const qv = new Float64Array(ng);
      let sigNew = 0;
      for (let k = 0; k < ng; k++) { pre[k] = precond[k] * r[k]; dv[k] = pre[k]; sigNew += r[k] * pre[k]; }
      let it = 1;
      while (it < this.maxiter && Math.sqrt(sigNew) / bn > this.tol) {
        matvec(dv, qv);
        let dq = 0;
        for (let k = 0; k < ng; k++) dq += dv[k] * qv[k];
        const alpha = sigNew / dq;
        for (let k = 0; k < ng; k++) { x[k] += alpha * dv[k]; r[k] -= alpha * qv[k]; }
        const sigOld = sigNew;
        sigNew = 0;
        for (let k = 0; k < ng; k++) { pre[k] = precond[k] * r[k]; sigNew += r[k] * pre[k]; }
        const beta = sigNew / sigOld;
        for (let k = 0; k < ng; k++) dv[k] = pre[k] + beta * dv[k];
        it++;
      }
      return { x, it, ok: Math.sqrt(sigNew) / bn <= this.tol };
    };
    const bS = new Float64Array(ng);
    const bT = new Float64Array(ng).fill(-1);
    for (let k = 0; k < ng; k++) bS[k] = -chi[k];
    // the history (src/engine-gpl/qeq/history.ts) gives the initial guesses and the iteration count
    const n = this.sys.state.n;
    if (!this.history || this.history.n !== n) this.history = gplHook<QeqHistoryHook>('qeq/slater').create(n, s.q);
    const guess = this.history.initialGuesses(idx);
    const sRes = solveOne(bS, guess.s);
    const tRes = solveOne(bT, guess.t);
    this.iterations = this.history.matvecCount(sRes.it, tRes.it);
    if (this.warn && (!sRes.ok || !tRes.ok)) {
      this.sys.warn(`Fix ${this.style} CG convergence failed after ${this.maxiter} iterations (maxiter reached)`);
    }
    let sumS = 0, sumT = 0;
    for (let k = 0; k < ng; k++) { sumS += sRes.x[k]; sumT += tRes.x[k]; }
    const q = new Float64Array(ng);
    for (let k = 0; k < ng; k++) q[k] = sRes.x[k] - (sumS / sumT) * tRes.x[k];
    this.history.shift(idx, sRes.x, tRes.x);
    return q;
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
