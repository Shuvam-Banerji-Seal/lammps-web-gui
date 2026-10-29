/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QEQ/fix_qeq.cpp (base: init, setup_pre_force, init_storage, CG, sparse_matvec, calculate_Q, the charge
 * histories), src/QEQ/fix_qeq_point.cpp (init_matvec, compute_H, pre_force), src/QEQ/fix_qeq_shielded.cpp,
 * src/QEQ/fix_qeq_dynamic.cpp (pre_force, compute_eneg), src/QEQ/fix_qeq_fire.cpp (pre_force) and
 * src/QEQ/fix_qeq_slater.cpp (the two solves and the charge sums), at commit b332f28. Copyright Sandia Corporation
 * (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * One class, FixQeq, for the fix qeq family of the browser engine: qeq/point, qeq/shielded, qeq/dynamic, qeq/fire and
 * qeq/slater. The conjugate-gradient solver and the charge histories of the matrix styles are in ./native_solver.ts and
 * ./history.ts; the registry is ../registry/fix_qeq.ts (lazy group 'fix_qeq' of src/engine/lazyStyles.ts).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError, typeBounds } from '../../engine/force/types';
import { PairCoulStreitz, mdfTaper, pointUval, slaterS } from '../../engine/force/pair/coul_streitz';
import type { System } from '../../engine/system';
import { hasCharge } from '../../engine/atoms';
import { fmtShortest, formatNumber } from '../../engine/cformat';
import { SQRT_PI } from '../../engine/libm';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { erfc as libmErfc } from '../libm/erf';
import { NativeQeqSolver, type QeqLinearSystem, type QeqSolver } from './native_solver';

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
 * The QEq matrix of qeq/point and qeq/shielded is H_ii = eta_i (plus the self images), H_ij = pair
 * interaction over every periodic image within the cutoff. Measured with native
 * LAMMPS (black box, not part of the doc text):
 *   - qeq/point off-diagonal is exactly 1/r inside the cutoff (no units
 *     constant, no taper), independent of the cutoff;
 *   - qeq/shielded off-diagonal is FixQEqShielded::calculate_H (src/QEQ/fix_qeq_shielded.cpp at b332f28), ported:
 *     Taper(r) 14.4 / (r^3 + shld_ij)^(1/3), the 7th-order taper polynomial of init_shielding between 0 and the
 *     cutoff by Horner, shld_ij = (gamma_i gamma_j)^-1.5, 14.4 = EV_TO_KCAL_PER_MOL of that file;
 * *   - qeq/point and qeq/shielded solve native's two conjugate-gradient
 *     systems H s = -chi and H t = -1 (src/QEQ/fix_qeq.cpp CG), each from the
 *     history extrapolation of the last five solutions, with the Jacobi
 *     preconditioner 1/eta and the stop test sqrt(r.d)/|b| > tolerance; the
 *     charges are q = s - (sum s / sum t) t. The fix global scalar is
 *     (iterations of s + iterations of t) / 2 (w123q_point_fixed, w123q_shield_fixed:
 *     matches native to 1e-15 including the scalar), although the doc says these
 *     fixes store no global scalar;
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
 * Keyword dsf with a taper width (docs.lammps.org/fix_qeq.html: "*dsf* value = width of taper to terminate Coulomb
 * integrals for the Fennell-Gezelter summation") multiplies the Slater integrals of the matrix by the taper over
 * [rc - width, rc] and leaves the erfc part alone, as native calculate_H_dsf does (src/QEQ/fix_qeq_slater.cpp at
 * b332f28); measured with w140ct_gan_taper to 4e-13. Keyword wolf with any taper width is not reproduced and throws a
 * StyleError (the wolf keyword charges differ from every form tried by about 2e-6 at 2 A).
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

interface Params {
  chi: Float64Array;
  eta: Float64Array;
  gamma: Float64Array;
  /** Slater exponent and nucleus charge (qeq/slater only; from the qfile columns zeta and qcore). */
  zeta: Float64Array;
  qcore: Float64Array;
}

/** Matrix terms of qeq/point and qeq/shielded (see matrixTerms). */
interface MatrixTerms {
  /** The pair terms in visit order: rows a and b and the value h (the first `npairs` entries). */
  pairA: Int32Array;
  pairB: Int32Array;
  pairH: Float64Array;
  npairs: number;
  selfDiag: Float64Array;
  fixedTerm: Float64Array;
}

/** fix_qeq_shielded.cpp (b332f28): "static constexpr double EV_TO_KCAL_PER_MOL = 14.4;" */
const EV_TO_KCAL_PER_MOL = 14.4;

/**
 * FixQEqShielded::init_shielding (src/QEQ/fix_qeq_shielded.cpp at b332f28): the coefficients of the 7th-order taper
 * between swa = 0 and swb = the fix cutoff (FixQEq: "swa = 0.0; swb = cutoff;").
 */
const taperCoefficients = (swb: number): Float64Array => {
  const swa = 0.0;
  const d7 = libmPow(swb - swa, 7);
  const swa2 = swa * swa, swa3 = swa2 * swa, swb2 = swb * swb, swb3 = swb2 * swb;
  const Tap = new Float64Array(8);
  Tap[7] = 20.0 / d7;
  Tap[6] = -70.0 * (swa + swb) / d7;
  Tap[5] = 84.0 * (swa2 + 3.0 * swa * swb + swb2) / d7;
  Tap[4] = -35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3) / d7;
  Tap[3] = 140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3) / d7;
  Tap[2] = -210.0 * (swa3 * swb2 + swa2 * swb3) / d7;
  Tap[1] = 140.0 * swa3 * swb3 / d7;
  Tap[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2 - 7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb) / d7;
  return Tap;
};

/** Constant of the qeq/dynamic charge dynamics (measured with native LAMMPS, black box). */
const DYNAMIC_SCALE = 31.25;

/** erfc for the qeq/slater matrix: the C library's, as native fix_qeq_slater.cpp calls erfc() (libm/erf.ts). */
const erfcSlater = (x: number): number => libmErfc(x);

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
  /** qeq/shielded: the taper coefficients (taperCoefficients) and shld per type pair (shieldingTable). */
  private tap: Float64Array | null = null;
  private shld: Float64Array | null = null;
  private shldStride = 0;
  private iterations = 0;
  /** The class member nlocal of native FixQEq: 0 until the first solve sets it (see init). */
  private nlocalSolved = 0;
  /** The CG solver (qeq/point, qeq/shielded, qeq/slater: src/engine-gpl/qeq/native_solver.ts), set by the constructor. */
  private solver: QeqSolver | null = null;
  /** The kspace paramsFrozen flag this fix found before its first run setup (restored in postRun). */
  private kspaceFrozenBefore: boolean | null = null;
  /** qeq/slater: wolf-type damping alpha (keyword alpha, default 0.2), dsf 0.0 form, qfile coul/streitz. */
  private alpha = 0.2;
  private dsf = false;
  /** dsf taper width (keyword dsf width; 0 when none): the Slater part of the matrix is tapered over [rc - width, rc]. */
  private drtap = 0;
  private fromPair = false;
  /** qeq/shielded with qfile reaxff: chi, eta, gamma of the active pair_style reaxff, read at each solve. */
  private fromReax = false;
  scalarFlag = true;
  extscalar = 0;

  constructor(sys: System, id: string, group: string, args: string[], kind: QeqKind) {
    super(sys, id, group, args);
    this.kind = kind;
    this.style = `qeq/${kind}`;
    // fix_qeq_point.cpp, fix_qeq_shielded.cpp and fix_qeq_slater.cpp init: a perpetual full list (Fix.nativeFullList);
    // qeq/dynamic and qeq/fire request the default half list
    this.nativeFullList = kind === 'point' || kind === 'shielded' || kind === 'slater';
    const a = args;
    // As native LAMMPS does (src/QEQ/fix_qeq.cpp FixQEq::FixQEq at b332f28): the five arguments, checked together, then
    // the charges, then the qfile (read here, before the keywords of the style)
    if (a.length < 5) throw new StyleError(`Illegal fix ${this.style} command: missing argument(s)`);
    this.nEvery = inumeric(a[0]);
    this.cutoff = numeric(a[1]);
    this.tol = numeric(a[2]);
    this.maxiter = inumeric(a[3]);
    this.qfile = a[4];
    if (this.nEvery <= 0 || this.cutoff <= 0.0 || this.tol <= 0.0 || this.maxiter <= 0) throw new StyleError('Illegal fix qeq command');
    if (!hasCharge(sys.state)) throw new StyleError(`Fix ${this.style} requires atom attribute q`);
    // "if (strcmp(arg[7],"coul/streitz") == 0) streitz_flag = 1; else if (utils::strmatch(arg[7],"^reax..")) reax_flag = 1;
    // else if (strcmp(arg[7],"coul/ctip") == 0) ctip_flag = 1; else read_file(arg[7]);"
    const streitz = this.qfile === 'coul/streitz', reax = /^reax../.test(this.qfile), ctip = this.qfile === 'coul/ctip';
    if (!streitz && !reax && !ctip) this.parseQfile(sys.readFile(this.qfile, 'Cannot open fix qeq parameter file'));
    // the defaults of fix_qeq_dynamic.cpp ("qdamp = 0.10; qstep = 0.02;") and fix_qeq_fire.cpp ("qdamp = 0.20; qstep = 0.20;")
    this.qdamp = kind === 'fire' ? 0.2 : 0.1;
    this.qstep = kind === 'fire' ? 0.2 : 0.02;
    this.warn = true;
    if (kind === 'point' || kind === 'shielded') {
      // fix_qeq_point.cpp and fix_qeq_shielded.cpp: "if (narg == 10) { if (strcmp(arg[8], "warn") == 0) maxwarn =
      // utils::logical(...); else error } else if (narg > 8) error"
      if (a.length === 7 && a[5] === 'warn') this.warn = logical(a[6]);
      else if (a.length > 5) throw new StyleError(`Illegal fix ${this.style} command`);
    } else {
      for (let k = 5; k < a.length;) {
        const key = a[k];
        if (key === 'warn') {
          if (k + 2 > a.length) throw new StyleError(`Illegal fix ${this.style} warn command: missing argument(s)`);
          this.warn = logical(a[k + 1]);
          k += 2;
        } else if (kind === 'slater' && key === 'alpha') {
          if (k + 2 > a.length) throw new StyleError(`Illegal fix ${this.style} alpha command: missing argument(s)`);
          this.alpha = numeric(a[k + 1]);
          k += 2;
          // fix_qeq.html: "wolf value = width of taper to terminate Coulomb integrals for the Wolf summation"
          // (the wolf taper is not reproduced here; see the header comment)
          if (a[k] === 'wolf') {
            throw new StyleError(`fix ${this.style}: keyword wolf (Wolf-summation charges) is not reproduced by the browser engine`);
          }
          if (a[k] === 'dsf') {
            // docs.lammps.org/fix_qeq.html: "*dsf* value = width of taper to terminate Coulomb integrals for the
            // Fennell-Gezelter summation (default value is zero)". fix_qeq_slater.cpp (b332f28): "optional taper width; do
            // not treat following keyword (e.g. "warn") as numeric"
            k += 1;
            this.dsf = true;
            if (k < a.length && isDouble(a[k])) {
              this.drtap = numeric(a[k]);
              k += 1;
            }
          }
        } else if (kind !== 'slater' && (key === 'qdamp' || key === 'qstep')) {
          if (k + 2 > a.length) throw new StyleError(`Illegal fix ${this.style} ${key} command: missing argument(s)`);
          if (key === 'qdamp') this.qdamp = numeric(a[k + 1]);
          else this.qstep = numeric(a[k + 1]);
          k += 2;
        } else {
          throw new StyleError(`Unknown fix ${this.style} keyword: ${key}`);
        }
      }
    }
    if (kind === 'slater' && this.dsf && !(this.drtap >= 0)) throw new StyleError(`fix ${this.style}: dsf taper width must be >= 0`);
    if (ctip) throw new StyleError(`fix ${this.style}: qfile 'coul/ctip' (parameters from pair coul/ctip) is not supported by the browser engine`);
    if (kind === 'slater' && streitz) {
      // fix_qeq_slater.cpp: "if (streitz_flag) extract_streitz();" (the parameters are read again at each solve)
      this.fromPair = true;
      if (!this.streitzPair()) throw new StyleError('No pair style coul/streitz for fix qeq/slater');
    } else if (kind === 'shielded' && reax) {
      // fix_qeq_shielded.cpp (b332f28): "if (reax_flag) extract_reax();", "pair->extract("chi", tmp)", "eta", "gamma"
      this.fromReax = true;
      const src = this.reaxSource();
      if (!src) throw new StyleError('No reaxff pair style for fix qeq/shielded');
      // the pair must know of the fix (pair reaxff checkqeq yes, pair_reaxff.cpp init_style)
      src.attachQeqFix();
    } else if (streitz || reax) {
      // native sets the flag and reads no parameters for a style that does not use them (its arrays stay unset)
      throw new StyleError(`fix ${this.style}: qfile '${this.qfile}' (parameters from a pair style) is not supported by fix ${this.style}`);
    }
    if (kind === 'point' || kind === 'shielded' || kind === 'slater') {
      this.solver = new NativeQeqSolver({
        kind, maxiter: this.maxiter, tol: this.tol, warn: this.warn,
        warnCg: (ratio, loop) => this.warnCg(ratio, loop),
      });
      // native constructor: every atom's history starts at its charge
      if (kind !== 'slater') this.solver.seed(sys.state.id, sys.state.q, sys.state.n);
    }
  }

  /**
   * FixQEq::read_file (src/QEQ/fix_qeq.cpp at b332f28): per line (comments and blank lines skipped) a type or type range
   * and chi eta gamma zeta zcore; every type must be set ("Parameters for atom type {} missing in qeq parameter file").
   */
  private parseQfile(text: string): void {
    const n = this.sys.state.ntypes;
    const chi = new Float64Array(n + 1), eta = new Float64Array(n + 1), gamma = new Float64Array(n + 1);
    const zeta = new Float64Array(n + 1), qcore = new Float64Array(n + 1);
    const setflag = new Uint8Array(n + 1);
    for (const raw of text.split('\n')) {
      const w = raw.replace(/#.*/, '').trim().split(/\s+/).filter((x) => x !== '');
      if (w.length === 0) continue;
      if (w.length < 6) throw new StyleError(`Invalid qeq parameter file for ${this.style}`);
      let lo = -1, hi = -1;
      try { [lo, hi] = typeBounds(w[0], n); } catch { lo = hi = -1; }
      if (lo < 0 || hi < 0) throw new StyleError(`Invalid atom type range: ${w[0]}`);
      const vals = w.slice(1, 6).map((v) => {
        if (!isDouble(v)) throw new StyleError(`Not a valid floating-point number: '${v}'`);
        return Number(v);
      });
      for (let t = lo; t <= hi; t++) {
        chi[t] = vals[0];
        eta[t] = vals[1];
        gamma[t] = vals[2];
        zeta[t] = vals[3];
        qcore[t] = vals[4];
        setflag[t] = 1;
      }
    }
    for (let t = 1; t <= n; t++) if (!setflag[t]) throw new StyleError(`Parameters for atom type ${t} missing in qeq parameter file`);
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

  /** The pair coul/streitz (force->pair_match("coul/streitz", 1): the pair style, or the sub-style of pair hybrid). */
  private streitzPair(): PairCoulStreitz | null {
    const pair = this.sys.ff.pair;
    if (pair instanceof PairCoulStreitz) return pair;
    let cs: PairCoulStreitz | null = null;
    const subs = (pair as unknown as { subs?: { style: unknown }[] } | null)?.subs ?? [];
    for (const sub of subs) if (sub.style instanceof PairCoulStreitz) cs = sub.style;
    return cs;
  }

  /** qeq/slater: parameters of the active coul/streitz pair style (qfile coul/streitz); chi, eta = J, zeta, qcore = Z. */
  private slaterParams(): Params {
    if (!this.fromPair) return this.params!;
    const n = this.sys.state.ntypes;
    const cs = this.streitzPair();
    if (!cs) throw new StyleError('No pair style coul/streitz for fix qeq/slater');
    const chi = new Float64Array(n + 1), eta = new Float64Array(n + 1), gamma = new Float64Array(n + 1);
    const zeta = new Float64Array(n + 1), qcore = new Float64Array(n + 1);
    for (let t = 1; t <= n; t++) {
      const tp = cs.typeParams(t);
      chi[t] = tp.chi; eta[t] = tp.J; gamma[t] = tp.gamma; zeta[t] = tp.zeta; qcore[t] = tp.Z;
    }
    return { chi, eta, gamma, zeta, qcore };
  }


  /**
   * Charges must be equilibrated before the first force computation of a run. Native qeq/point and qeq/shielded solve in
   * setup_pre_force (before the setup force evaluation, with the same nevery rule as pre_force); the other styles keep
   * their solve after it.
   */
  setupPreForce(): void {
    // FixQEq::setup_pre_force: "if (force->newton_pair == 0) error->all(FLERR,"QEQ with 'newton pair off' not supported");"
    if (!this.sys.newtonPair) throw new StyleError("QEQ with 'newton pair off' not supported");
    // native FixQEq::setup_pre_force (inherited by every style of this file, src/QEQ/fix_qeq.cpp at b332f28) runs
    // pre_force before the setup force evaluation. The kspace G-ewald is the one of this run's init, as native keeps
    // it (KSpace::qsum_qsq refreshes only the charge sums): paramsFrozen stays set until postRun.
    const kspace = this.sys.ff.kspace;
    if (kspace) {
      if (this.kspaceFrozenBefore === null) this.kspaceFrozenBefore = kspace.paramsFrozen;
      kspace.paramsFrozen = true;
    }
    this.preForce();
  }

  postRun(): void {
    const kspace = this.sys.ff.kspace;
    if (kspace && this.kspaceFrozenBefore !== null) kspace.paramsFrozen = this.kspaceFrozenBefore;
    this.kspaceFrozenBefore = null;
  }

  /**
   * FixQEq::init and the init of each style (src/QEQ/fix_qeq*.cpp at b332f28), at every run: the charges, the group,
   * the fix efield and net charge warnings; qeq/shielded: the taper warning and the gamma check, which loops over the
   * class member nlocal, 0 until the first solve, so it applies from the second run on; qeq/slater: the zeta check;
   * qeq/dynamic and qeq/fire: the tolerance warning.
   */
  init(): void {
    const s = this.sys.state;
    if (!hasCharge(s)) throw new StyleError(`Fix ${this.style} requires atom attribute q`);
    let ngroup = 0, qsum = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      ngroup++;
      qsum += s.q[i];
    }
    if (ngroup === 0) throw new StyleError(`Fix ${this.style} group has no atoms`);
    if (this.sys.fixes.some((f) => /^efield/.test(f.style))) this.sys.warn('Fix efield is ignored during charge equilibration');
    // "if ((comm->me == 0) && (fabs(qsum) > QSUMSMALL))" with QSUMSMALL 0.00001, "net charge = {:.8}" and errorurl(29)
    if (Math.abs(qsum) > 0.00001) {
      this.sys.warn(`Fix ${this.style} group is not charge neutral, net charge = ${formatNumber(qsum, '%.8g')}\nFor more information see https://docs.lammps.org/err0029`);
    }
    if (this.kind === 'shielded') {
      // init_shielding: swa = 0, swb = the cutoff
      if (this.cutoff < 5) this.sys.warn('Fix qeq has very low Taper radius cutoff');
      const p = this.fromReax ? this.reaxParams() : this.params!;
      for (let i = 0; i < Math.min(this.nlocalSolved, s.n); i++) {
        if ((s.mask[i] & this.groupBit) && p.gamma[s.type[i]] === 0.0) throw new StyleError(`Invalid QEq parameters for atom type ${s.type[i]} provided`);
      }
    } else if (this.kind === 'slater') {
      const p = this.slaterParams();
      for (let t = 1; t <= s.ntypes; t++) if (p.zeta[t] === 0.0) throw new StyleError('Invalid parameter file values for fix qeq/slater');
    } else if ((this.kind === 'dynamic' || this.kind === 'fire') && this.tol < 1e-4) {
      this.sys.warn(this.kind === 'dynamic'
        ? `Fix qeq/dynamic tolerance ${fmtShortest(this.tol, false)} may be too small for damped dynamics`
        : `Fix qeq/fire tolerance ${fmtShortest(this.tol, false)} may be too small for damped fires`);
    }
  }

  /** fix_qeq.html: Nevery = "perform charge equilibration every this many steps" (pre_force's ntimestep % nevery). */
  preForce(): void {
    if (this.sys.state.step % this.nEvery === 0) this.solve();
  }

  computeScalar(): number { return this.iterations; }

  /** Native's CG warning (fix_qeq.cpp CG): "Fix qeq CG convergence failed (ratio) after loop iterations at step N". */
  private warnCg(ratio: number, loop: number): void {
    if (!this.warn) return;
    this.sys.warn(`Fix qeq CG convergence failed (${ratio}) after ${loop} iterations at step ${this.sys.state.step}`);
  }

  private solve(): void {
    const s = this.sys.state;
    const p = this.kind === 'slater' ? this.slaterParams() : this.fromReax ? this.reaxParams() : this.params!;
    const bit = this.groupBit;
    const shielded = this.kind === 'shielded';
    // pre_force: "nlocal = atom->nlocal;" (read by the gamma check of a later init)
    this.nlocalSolved = s.n;
    if (shielded) {
      this.tap ??= taperCoefficients(this.cutoff);
      this.shieldingTable(p);
    }
    // group atoms (owned)
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) idx.push(i);
    const ng = idx.length;
    if (ng === 0) { this.iterations = 0; return; }

    // pair interactions within the cutoff: point 1/r, shielded 14.4*(r^3+g^-3)^-1/3*Taper. qeq/point and qeq/shielded
    // build their matrix over all images (matrixTerms); qeq/slater builds its own image list in solveSlater.
    const matrix = this.kind === 'point' || this.kind === 'shielded';
    const pairs: { i: number; j: number; int: number }[] = [];
    const d = [0, 0, 0];
    const cut2 = this.cutoff * this.cutoff;
    for (let a = 0; a < (this.kind === 'slater' || matrix ? 0 : ng); a++) {
      const i = idx[a];
      for (let b = a + 1; b < ng; b++) {
        const j = idx[b];
        d[0] = s.x[3 * i] - s.x[3 * j];
        d[1] = s.x[3 * i + 1] - s.x[3 * j + 1];
        d[2] = s.x[3 * i + 2] - s.x[3 * j + 2];
        this.sys.geom.minimumImage(d);
        const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (!(r2 > 0) || r2 > cut2) continue;
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
    if (ng < s.n && this.kind !== 'slater' && !matrix) {
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
          if (!(r2 > 0) || r2 > cut2) continue;
          fixedTerm[a] += this.interaction(s, p, i, k, Math.sqrt(r2), shielded) * s.q[k];
        }
      }
    }

    let q: Float64Array;
    if (this.kind === 'slater') {
      q = this.solveSlater(s, p, idx, ng);
    } else if (this.kind === 'point' || this.kind === 'shielded') {
      q = this.solveMatrix(s, p, idx, this.matrixTerms(s, p, idx, row, shielded), etaRow, ng);
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

    for (let a = 0; a < ng; a++) s.q[idx[a]] = q[a];
    if (this.sys.ff.kspace) {
      // A kspace style caches the charge sums its self/background energy uses
      // (the engine refreshes them in setupNeighbors); a fix that changes
      // charges mid-run must re-run it or the long-range energy goes stale.
      // The engine refreshes the sums in setupNeighbors. Not reproduced (measured with native LAMMPS, black box,
      // w36qeq_point): native's step-0 pe of the first run is -6.4223085227782 while its step-0 pe of the next run,
      // with the same charges, is -6.4223080743197 (the engine gives the latter for both; forces agree to 2e-10).
      // Freezing the sums at the pre-solve charges moves the pe by 0.4, so the difference is not the sums alone.
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
    const selfW = -(erc / (2 * rc) + a / SQRT_PI);
    const kd = this.dsf ? erc / (rc * rc) + (2 * a / SQRT_PI) * libmExp(-a * a * rc * rc) / rc : 0;
    const cc = (r: number): number => {
      const e = erfcSlater(a * r) / r;
      return this.dsf ? e - erc / rc + (r - rc) * kd : e;
    };
    const cut2 = rc * rc;
    const bit = this.groupBit;
    const row = new Int32Array(s.n).fill(-1);
    for (let k = 0; k < ng; k++) row[idx[k]] = k;
    // dsf taper (mdftaper over [rc - drtap, rc]): 1 without a taper width. fix_qeq_slater.cpp calculate_H_dsf multiplies
    // the Slater integrals ci_jfi and ci_fifj by ftap; the erfc part edsf is not tapered.
    const TAP = { f: 1, df: 0 };
    const taperAt = (r: number): number => {
      if (!this.dsf || this.drtap === 0) return 1;
      mdfTaper(r, rc, this.drtap, TAP);
      return TAP.f;
    };
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
          const tf = taperAt(r);
          const S = slaterS(r, zi, zv) * tf;
          const Ui = pointUval(r, zi) * tf;
          const Uv = pointUval(r, zv) * tf;
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
    const bS = new Float64Array(ng);
    for (let k = 0; k < ng; k++) bS[k] = -chi[k];
    const sys: QeqLinearSystem = {
      ng, diag, precond, bS, fixed: null,
      pairA: Int32Array.from(pairA), pairB: Int32Array.from(pairC), pairH: Float64Array.from(pairH),
    };
    return this.runSolver(s, idx, sys);
  }

  /**
   * Point (qeq/point, qeq/dynamic, qeq/fire) or shielded (qeq/shielded) pair interaction. The shielded one is
   * FixQEqShielded::calculate_H (src/QEQ/fix_qeq_shielded.cpp at b332f28), whose half native stores in each list entry
   * (H.val = 0.5 calculate_H): the taper polynomial by Horner, over (r^3 + shld)^(1/3) with shld = (gamma_i
   * gamma_j)^-1.5 (shieldingTable). A type with gamma 0 has shld = inf and no interaction.
   */
  private interaction(s: System['state'], _p: Params, i: number, j: number, r: number, shielded: boolean): number {
    if (!shielded) return 1 / r;
    const Tap = this.tap!;
    let Taper = Tap[7] * r + Tap[6];
    Taper = Taper * r + Tap[5];
    Taper = Taper * r + Tap[4];
    Taper = Taper * r + Tap[3];
    Taper = Taper * r + Tap[2];
    Taper = Taper * r + Tap[1];
    Taper = Taper * r + Tap[0];
    let denom = r * r * r + this.shld![s.type[i] * this.shldStride + s.type[j]];
    denom = libmPow(denom, 1.0 / 3.0);
    return Taper * EV_TO_KCAL_PER_MOL / denom;
  }

  /** FixQEqShielded::init_shielding: "shld[i][j] = pow(gamma[i] * gamma[j], -1.5);" for every type pair. */
  private shieldingTable(p: Params): void {
    const nt = this.sys.state.ntypes;
    const stride = nt + 1;
    if (!this.shld || this.shld.length !== stride * stride) this.shld = new Float64Array(stride * stride);
    this.shldStride = stride;
    for (let i = 1; i <= nt; i++) for (let j = 1; j <= nt; j++) this.shld[i * stride + j] = libmPow(p.gamma[i] * p.gamma[j], -1.5);
  }

  /**
   * Matrix terms of qeq/point and qeq/shielded over every periodic image within the cutoff (native's full neighbour
   * list, fix_qeq_point.cpp compute_H, and H.val = 0.5 f(r) in each list entry of a group atom):
   *  - a pair of two group atoms, one term per image f(r) (both directions of the matvec);
   *  - an image of a group atom with itself adds f(r) to its diagonal (both of its list entries count);
   *  - a neighbour outside the group contributes 0.5 f(r) q_k to the first residual only: its own list is not
   *    visited, so the coupling is half (the stored charge is the fixed value).
   */
  private matrixTerms(s: System['state'], p: Params, idx: number[], row: Int32Array, shielded: boolean): MatrixTerms {
    const ng = idx.length;
    const rc = this.cutoff, cut2 = rc * rc;
    const bit = this.groupBit;
    const geom = this.sys.geom as unknown as { lx: number; ly: number; lz: number; xy: number; xz: number; yz: number; periodic: boolean[] };
    const Ld = [geom.lx, geom.ly, geom.lz];
    const ortho = geom.xy === 0 && geom.xz === 0 && geom.yz === 0;
    const nmax = [0, 1, 2].map((dd) => (geom.periodic[dd] ? Math.ceil(rc / Ld[dd]) + 1 : 0));
    const triShifts: number[][] = [];
    if (!ortho) {
      const ax = [geom.lx, 0, 0], bx = [geom.xy, geom.ly, 0], cx = [geom.xz, geom.yz, geom.lz];
      for (let nx = -nmax[0]; nx <= nmax[0]; nx++)
        for (let ny = -nmax[1]; ny <= nmax[1]; ny++)
          for (let nz = -nmax[2]; nz <= nmax[2]; nz++)
            triShifts.push([nx * ax[0] + ny * bx[0] + nz * cx[0], nx * ax[1] + ny * bx[1] + nz * cx[1], nx * ax[2] + ny * bx[2] + nz * cx[2]]);
    }
    let pairA = new Int32Array(1024), pairB = new Int32Array(1024), pairH = new Float64Array(1024);
    let npairs = 0;
    const selfDiag = new Float64Array(ng);
    const fixedTerm = new Float64Array(ng);
    const dv = [0, 0, 0];
    const lo = [0, 0, 0], hi = [0, 0, 0];
    const x = s.x;
    // one image of the pair (i, v) at separation r2 (r2 > 0 and within the cutoff): its term
    const add = (a: number, i: number, v: number, vg: boolean, r2: number): void => {
      const f = this.interaction(s, p, i, v, Math.sqrt(r2), shielded);
      if (!vg) { fixedTerm[a] += 0.5 * f * s.q[v]; return; }
      if (v === i) { selfDiag[a] += f; return; }
      if (npairs === pairA.length) {
        const m = 2 * npairs;
        const na = new Int32Array(m), nb = new Int32Array(m), nh = new Float64Array(m);
        na.set(pairA); nb.set(pairB); nh.set(pairH);
        pairA = na; pairB = nb; pairH = nh;
      }
      pairA[npairs] = a; pairB[npairs] = row[v]; pairH[npairs] = f;
      npairs++;
    };
    for (let a = 0; a < ng; a++) {
      const i = idx[a];
      for (let v = 0; v < s.n; v++) {
        const vg = (s.mask[v] & bit) !== 0;
        if (vg && row[v] < a) continue; // a group pair is one term: counted from its lower row
        dv[0] = x[3 * i] - x[3 * v];
        dv[1] = x[3 * i + 1] - x[3 * v + 1];
        dv[2] = x[3 * i + 2] - x[3 * v + 2];
        if (ortho) {
          for (let dd = 0; dd < 3; dd++) {
            if (!geom.periodic[dd]) { lo[dd] = 0; hi[dd] = 0; continue; }
            lo[dd] = Math.floor((dv[dd] - rc) / Ld[dd]);
            hi[dd] = Math.ceil((dv[dd] + rc) / Ld[dd]);
          }
          // the images in x, y, z order; an image whose partial sum of squares is already past the cutoff is past it in
          // full (the sum dx*dx + dy*dy + dz*dz only grows), so it is skipped before the inner loops
          for (let nx = lo[0]; nx <= hi[0]; nx++) {
            const dx = dv[0] - nx * Ld[0];
            const dx2 = dx * dx;
            if (dx2 > cut2) continue;
            for (let ny = lo[1]; ny <= hi[1]; ny++) {
              const dy = dv[1] - ny * Ld[1];
              const dxy = dx2 + dy * dy;
              if (dxy > cut2) continue;
              for (let nz = lo[2]; nz <= hi[2]; nz++) {
                const dz = dv[2] - nz * Ld[2];
                const r2 = dxy + dz * dz;
                if (!(r2 > 0) || r2 > cut2) continue;
                add(a, i, v, vg, r2);
              }
            }
          }
        } else {
          for (const sh of triShifts) {
            const dx = dv[0] - sh[0], dy = dv[1] - sh[1], dz = dv[2] - sh[2];
            const r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0) || r2 > cut2) continue;
            add(a, i, v, vg, r2);
          }
        }
      }
    }
    return { pairA, pairB, pairH, npairs, selfDiag, fixedTerm };
  }

  /**
   * qeq/point and qeq/shielded: the matrix elements of matrixTerms in the solver's terms (diagonal eta + self terms,
   * off-diagonal pairs, Jacobi preconditioner 1/eta, right-hand side -chi, the fixed-atom term of the s residual); the
   * two conjugate-gradient solves, the history and the charge sums are in src/engine-gpl/qeq/native_solver.ts.
   */
  private solveMatrix(s: System['state'], p: Params, idx: number[], mt: MatrixTerms, etaRow: Float64Array, ng: number): Float64Array {
    const precond = new Float64Array(ng), diag = new Float64Array(ng), bS = new Float64Array(ng);
    for (let a = 0; a < ng; a++) {
      precond[a] = 1 / etaRow[a];
      diag[a] = etaRow[a] + mt.selfDiag[a];
      bS[a] = -p.chi[s.type[idx[a]]];
    }
    const sys: QeqLinearSystem = {
      ng, diag, precond, bS, fixed: mt.fixedTerm,
      pairA: mt.pairA.subarray(0, mt.npairs),
      pairB: mt.pairB.subarray(0, mt.npairs),
      pairH: mt.pairH.subarray(0, mt.npairs),
    };
    return this.runSolver(s, idx, sys);
  }

  /** The gpl CG solver of this fix (gplHook 'qeq/point', 'qeq/shielded' or 'qeq/slater'): new charges, iteration scalar. */
  private runSolver(s: System['state'], idx: number[], sys: QeqLinearSystem): Float64Array {
    const r = this.solver!.solve({ sys, idx, ids: idx.map((i) => s.id[i]), n: s.n, q: s.q });
    this.iterations = r.iterations;
    return r.q;
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
    let fmax = maxAbs(f);
    let iters = 0;
    // native performs at most maxiter - 1 updates (maxiter 1 leaves the charges unchanged; measured). Native's stop
    // test (fix_qeq_dynamic.cpp pre_force): enegchk <= tolerance and enegmax <= 100 tolerance.
    for (; iters < this.maxiter - 1 && !(fm <= this.tol && fmax <= 100 * this.tol); iters++) {
      if (iters === 0) {
        for (let a = 0; a < ng; a++) dq[a] = B * f[a];
      } else {
        for (let a = 0; a < ng; a++) dq[a] = gamma * gamma * dq[a] + B * f[a];
      }
      for (let a = 0; a < ng; a++) q[a] += dq[a];
      gradient(q, f);
      fm = meanAbs(f);
      fmax = maxAbs(f);
    }
    this.iterations = iters;
    if (!(fm <= this.tol && fmax <= 100 * this.tol) && this.warn) this.sys.warn(`Charges did not converge at step ${this.sys.state.step}: ${fmtShortest(fm, false)}`);
    return q;
  }

  /**
   * qeq/fire: native FixQEqFire::pre_force (src/QEQ/fix_qeq_fire.cpp at b332f28), statement order kept. Per iteration:
   * the charge force f at the current charges; if v.f > 0 the velocity mixes toward f (alpha, then alpha shrinks and dt
   * grows when the last negative step is older than the current MD step), else dt halves, alpha resets to ALPHA0 and v
   * is zero; the step is limited so that dt |v| <= 0.1 per atom; then q moves by dt v and v by dt f (the position
   * moves before v is updated, and the convergence test uses the force of the step just taken). The sign of v is the
   * negative of native's qv (native's force is -f), which leaves every product of the update unchanged. The fix scalar is
   * the number of iterations at exit (native matvecs = loop).
   */
  private solveFire(q0: Float64Array, gradient: (q: Float64Array, out: Float64Array) => void, ng: number): Float64Array {
    const q = Float64Array.from(q0);
    const v = new Float64Array(ng);
    const f = new Float64Array(ng);
    const step = this.sys.state.step;
    let alpha = this.qdamp;
    let dt = this.qstep;
    const dtmax = FIRE_TMAX * dt;
    let lastNegative = 0;
    let fm = 0;
    let iloop = 0;
    for (; iloop < this.maxiter; iloop++) {
      gradient(q, f);
      let vdotf = 0;
      for (let a = 0; a < ng; a++) vdotf += v[a] * f[a];
      if (vdotf > 0) {
        let vdotv = 0, fdotf = 0;
        for (let a = 0; a < ng; a++) { vdotv += v[a] * v[a]; fdotf += f[a] * f[a]; }
        const scale1 = 1 - alpha;
        const scale2 = fdotf === 0 ? 0 : alpha * Math.sqrt(vdotv / fdotf);
        for (let a = 0; a < ng; a++) v[a] = scale1 * v[a] + scale2 * f[a];
        if (step - lastNegative > FIRE_DELAYSTEP) {
          dt = Math.min(dt * FIRE_DT_GROW, dtmax);
          alpha *= FIRE_ALPHA_SHRINK;
        }
      } else {
        lastNegative = step;
        dt *= FIRE_DT_SHRINK;
        alpha = FIRE_ALPHA0;
        v.fill(0);
      }
      let dtv = dt;
      for (let a = 0; a < ng; a++) {
        const vmax = Math.abs(v[a]);
        if (dtv * vmax > FIRE_DQMAX) dtv = FIRE_DQMAX / vmax;
      }
      let enegchk = 0;
      for (let a = 0; a < ng; a++) {
        q[a] += dtv * v[a];
        v[a] += dtv * f[a];
        enegchk += Math.abs(f[a]);
      }
      fm = ng ? enegchk / ng : 0;
      if (fm < this.tol) break;
    }
    this.iterations = iloop;
    if (fm >= this.tol && this.warn) this.sys.warn(`Charges did not converge at step ${step}: ${fmtShortest(fm, false)}`);
    return q;
  }
}

/**
 * Convergence measure of qeq/dynamic and qeq/fire: the mean |f| over the fix group. Measured with native
 * LAMMPS (black box): on a 5-atom irregular cluster the native iteration counts at tolerances 1e-1 ... 1e-5
 * (28, 40, 46, 65, 71, 77, 95, 108, 114) are reproduced only by mean |f| < tolerance (not max |f|, the RMS,
 * the norm or the sum).
 */
/** Constants of native FixQEqFire::pre_force (fix_qeq_fire.cpp at b332f28). */
const FIRE_DQMAX = 0.1;
const FIRE_DELAYSTEP = 0;
const FIRE_DT_GROW = 1.1;
const FIRE_DT_SHRINK = 0.5;
const FIRE_ALPHA0 = 0.8;
const FIRE_ALPHA_SHRINK = 0.1;
const FIRE_TMAX = 10;

const maxAbs = (f: Float64Array): number => {
  let m = 0;
  for (let k = 0; k < f.length; k++) m = Math.max(m, Math.abs(f[k]));
  return m;
};

const meanAbs = (f: Float64Array): number => {
  let s = 0;
  for (let k = 0; k < f.length; k++) s += Math.abs(f[k]);
  return f.length ? s / f.length : 0;
};

/** utils::is_double: a floating-point number as LAMMPS reads one (sign, digits, optional point and exponent). */
const isDouble = (w: string): boolean => /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(w);

/** utils::numeric. */
const numeric = (w: string | undefined): number => {
  if (w === undefined || !isDouble(w)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  return Number(w);
};

/** utils::inumeric. */
const inumeric = (w: string | undefined): number => {
  if (w === undefined || !/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  return Number(w);
};

/** utils::logical: yes, no, true, false, on, off, 1, 0. */
const logical = (w: string | undefined): boolean => {
  if (w === 'yes' || w === 'true' || w === 'on' || w === '1') return true;
  if (w === 'no' || w === 'false' || w === 'off' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in input script or data file`);
};

