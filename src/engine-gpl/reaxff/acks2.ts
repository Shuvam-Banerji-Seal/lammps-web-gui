/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/fix_acks2_reaxff.cpp and src/REAXFF/fix_acks2_reaxff.h (FixACKS2ReaxFF:
 * constructor, pertype_parameters, init, init_bondcut, init_storage, pre_force, init_matvec,
 * compute_X, calculate_X, BiCGStab, sparse_matvec_acks2, calculate_Q, and the solution-history
 * bookkeeping), and the ACKS2 terms of src/REAXFF/reaxff_nonbonded.cpp that read its solution (see
 * nonbonded.ts), at commit b332f28. Author credited in the C++ file headers: Stan Moore (Sandia).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * The solve is the extended system of ACKS2: for each owned atom the charge s_i (rows 0..N-1), the
 * kinetic-energy variable s_i^X (rows N..2N-1, the X matrix with the bond-softness off-diagonal
 * terms), and two Lagrange rows at 2N, 2N+1 that enforce the charge constraints. N = nlocal + nghost,
 * as in LAMMPS NN. The matrix-vector product, the BiCGStab and the preconditioners follow the C++
 * functions one for one; the ghost contributions are folded to their owners after each product.
 * The fix efield coupling (chi_field, efield.ts) is ported; the dual and matfree variants are refused. The
 * parameter file form of pertype is ported (see parseAcks2File).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { buildRows, calcHValue, forwardGhosts, historyOf, NPREV, type HistoryMap, type SparseRows, taperFix } from './charge';
import type { PairReaxFF } from './pair';
import type { QeqParams } from './qeq';
import type { FixEfield } from '../../engine/fix/force_ext';
import { efieldForModel, fieldOf, findEfield, qeqChiField } from './efield';

/** The per-type ACKS2 parameters: chi, eta, gamma (QeqParams) and the bond cutoff bcut_acks2 (1-based). */
interface Acks2Params extends QeqParams { bcut: Float64Array<ArrayBufferLike> }

/** The data of one BiCGStab solve: the matrices, the layout and the preconditioners. */
interface BiWork {
  grp: number[];
  n: number;
  N: number;
  owner: Int32Array;
  H: SparseRows;
  X: SparseRows;
  eta: Float64Array;
  typeAll: Int32Array;
  xdiag: Float64Array;
  Hdia_inv: Float64Array;
  Xdia_inv: Float64Array;
}

/** fix acks2/reaxff (FixACKS2ReaxFF). */
export class FixACKS2ReaxFF extends Fix {
  readonly style = 'acks2/reaxff';
  nevery: number;
  private swa: number;
  private swb: number;
  private tolerance: number;
  private imax = 200;
  private maxwarn = true;
  private pertype: string;
  private params: Acks2Params | null = null;
  private bondSoftness = 0;
  private Tap: Float64Array<ArrayBufferLike> = new Float64Array(8);
  private shld: Float64Array<ArrayBufferLike> = new Float64Array(0);
  /** bcut[i][j] = (bcut_i + bcut_j)/2 flattened (ntypes + 1)^2. */
  private bcut: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private ntypes = 0;
  /** per atom id: s history 0..3 and s^X history 4..7 (most recent first). */
  private hist: HistoryMap = new Map();
  /** the two Lagrange rows: their histories (4 each). */
  private histLast: Float64Array<ArrayBufferLike>[] = [new Float64Array(NPREV), new Float64Array(NPREV)];
  /** the solution of the last solve (2N + 2 entries), read by the pair style (ACKS2 terms). */
  private sVec: Float64Array<ArrayBufferLike> = new Float64Array(2);
  /** fix efield (the chi_field term), or null. */
  private efieldFix: FixEfield | null = null;
  private matvecs = 0;

  constructor(sys: System, id: string, group: string, args: string[], private pairRef: () => PairReaxFF | null) {
    super(sys, id, group, args);
    this.scalarFlag = true;
    this.extscalar = 0;
    // narg of LAMMPS is 8..12 with "fix ID group style" in front: 5..9 arguments here (as fix qeq/reaxff)
    if (args.length < 5 || args.length > 9) throw new StyleError('Illegal fix acks2/reaxff command');
    this.pairRef()?.attachAcks2(() => this.sVec);
    this.nevery = intArg(args[0], 'Nevery');
    if (this.nevery <= 0) throw new StyleError('Illegal fix acks2/reaxff command');
    this.swa = numArg(args[1], 'swa');
    this.swb = numArg(args[2], 'swb');
    this.tolerance = numArg(args[3], 'tolerance');
    this.pertype = args[4];
    let iarg = 5;
    while (iarg < args.length) {
      const k = args[iarg];
      if (k === 'nowarn') this.maxwarn = false;
      else if (k === 'maxiter') {
        if (iarg + 1 > args.length - 1) throw new StyleError('Illegal fix acks2/reaxff command: maxiter needs a value');
        this.imax = intArg(args[iarg + 1], 'maxiter');
        iarg++;
      } else if (k === 'dual') throw new StyleError('fix acks2/reaxff keyword dual is not ported (it is an OPENMP variant only)');
      else if (k === 'matfree') throw new StyleError('fix acks2/reaxff keyword matfree is not ported (Kokkos variant only)');
      else throw new StyleError(`Illegal fix acks2/reaxff command: unknown keyword ${k}`);
      iarg++;
    }
  }

  override init(): void {
    const st = this.sys.state;
    let count = 0;
    for (let i = 0; i < st.n; i++) if (st.mask[i] & this.groupBit) count++;
    if (count === 0) throw new StyleError('Fix acks2/reaxff group has no atoms');
    if (this.swb < 0) throw new StyleError('Fix acks2/reaxff has negative upper Taper radius cutoff');
    this.efieldFix = findEfield(this.sys, 'acks2/reaxff');
    if (this.efieldFix) efieldForModel(this.sys, this.efieldFix, 'acks2/reaxff', { allGroup: false, periodicCheck: true });
    this.ntypes = st.ntypes;
    this.Tap = taperFix(this.swa, this.swb);
    this.pertypeParameters();
    const p = this.params!;
    const nt = this.ntypes, nt1 = nt + 1;
    this.shld = new Float64Array(nt1 * nt1);
    this.bcut = new Float64Array(nt1 * nt1);
    for (let i = 1; i <= nt; i++) {
      for (let j = 1; j <= nt; j++) {
        this.shld[i * nt1 + j] = Math.pow(p.gamma[i] * p.gamma[j], -1.5);
        this.bcut[i * nt1 + j] = 0.5 * (p.bcut[i] + p.bcut[j]);
      }
    }
  }

  /** pertype_parameters: the keyword reaxff (pair style extract) or the parameter file (bond softness, then per type). */
  private pertypeParameters(): void {
    if (this.params) return;
    const nt = this.ntypes;
    if (this.pertype.startsWith('reaxff')) {
      const pair = this.pairRef();
      if (!pair) throw new StyleError('No reaxff pair style for fix acks2/reaxff');
      const q = pair.qeqParameters(nt);
      const a = pair.acks2Parameters(nt);
      this.params = { ...q, bcut: a.bcut };
      this.bondSoftness = a.bondSoftness;
      return;
    }
    if (this.pertype.startsWith('reax/c')) throw new StyleError("Fix acks2/reaxff keyword 'reax/c' is obsolete; please use 'reaxff'");
    const parsed = parseAcks2File(this.sys.readFile(this.pertype), nt);
    this.bondSoftness = parsed.bondSoftness;
    this.params = parsed;
  }

  override setup(): void { this.preForce(); }

  override preForce(): void {
    if (this.sys.state.step % this.nevery !== 0) return;
    this.solve();
  }

  override computeScalar(): number { return this.matvecs / 2.0; }

  /** The solution vector of the last solve (for the ACKS2 terms of the pair style). */
  sVector(): Float64Array { return this.sVec; }

  /**
   * pre_force: compute_H, compute_X, the diagonal preconditioners, the cubic extrapolation of the
   * starting guess, BiCGStab on the extended system and calculate_Q (q = s).
   */
  private solve(): void {
    if (!this.params) this.init();
    const s = this.sys.state;
    const nb = this.sys.nb;
    const n = nb.nlocal, N = nb.nall, L = 2 * N + 2;
    const x = nb.xall, owner = nb.owner, typeAll = nb.typeall;
    const p = this.params!;
    const nt1 = this.ntypes + 1;
    const orig = new Int32Array(N);
    for (let a = 0; a < N; a++) orig[a] = s.id[owner[a]];
    const grp: number[] = [];
    for (let i = 0; i < n; i++) if (s.mask[i] & this.groupBit) grp.push(i);
    const inGroup = new Uint8Array(N);
    for (const i of grp) inGroup[i] = 1;

    // compute_H (the shielded Coulomb matrix, as fix qeq/reaxff)
    const H = buildRows(x, n, N, orig, inGroup, this.swb, (i, j, r) => calcHValue(this.Tap, r, this.shld[typeAll[i] * nt1 + typeAll[j]]));
    // compute_X: the bond-softness matrix within each bond cutoff; X_diag collects its row sums
    const xdiag = new Float64Array(N);
    const X = buildRows(x, n, N, orig, inGroup, this.swb, (i, j, r, rsq) => {
      const bc = this.bcut[typeAll[i] * nt1 + typeAll[j]];
      if (!(rsq <= bc * bc)) return NaN;
      const v = this.calculateX(r, bc);
      xdiag[i] -= v;
      xdiag[owner[j]] -= v;
      return v;
    });
    const Xdia_inv = new Float64Array(N);
    for (let i = 0; i < n; i++) Xdia_inv[i] = xdiag[i] === 0.0 ? 1.0 : 1.0 / xdiag[i];

    // init_matvec: preconditioner, sources and extrapolated starting guesses
    const Hdia_inv = new Float64Array(N);
    const b = new Float64Array(L), sv = new Float64Array(L);
    // chi_field of fix efield (zero outside its group): b = -chi, then b -= chi_field
    const chiField = this.efieldFix ? qeqChiField(this.sys, this.efieldFix, fieldOf(this.sys, this.efieldFix)) : null;
    for (const i of grp) {
      Hdia_inv[i] = 1.0 / p.eta[typeAll[i]];
      b[i] = -p.chi[typeAll[i]];
      if (chiField) b[i] -= chiField[i];
      b[N + i] = 0.0;
      const h = historyOf(this.hist, orig[i], 2 * NPREV);
      sv[i] = 4 * (h[0] + h[2]) - (6 * h[1] + h[3]);
      sv[N + i] = 4 * (h[4] + h[6]) - (6 * h[5] + h[7]);
    }
    for (let k = 0; k < 2; k++) {
      b[2 * N + k] = 0.0;
      const h = this.histLast[k];
      sv[2 * N + k] = 4 * (h[0] + h[2]) - (6 * h[1] + h[3]);
    }
    forwardAll(sv, n, N, owner);

    const work: BiWork = { grp, n, N, owner, H, X, eta: p.eta, typeAll, xdiag, Hdia_inv, Xdia_inv };
    this.matvecs = bicgstab(work, b, sv, { tolerance: this.tolerance, imax: this.imax, maxwarn: this.maxwarn, warn: (t) => this.sys.warn(t), step: s.step });

    // calculate_Q: the charges are the solution s (the ghosts first take their owners' values); the histories move on
    forwardAll(sv, n, N, owner);
    for (const i of grp) {
      s.q[i] = sv[i];
      const h = historyOf(this.hist, orig[i], 2 * NPREV);
      for (let k = NPREV - 1; k > 0; k--) {
        h[k] = h[k - 1];
        h[4 + k] = h[4 + k - 1];
      }
      h[0] = sv[i];
      h[4] = sv[N + i];
    }
    for (let k = 0; k < 2; k++) {
      const h = this.histLast[k];
      for (let m = NPREV - 1; m > 0; m--) h[m] = h[m - 1];
      h[0] = sv[2 * N + k];
    }
    nb.refreshCharges(s);
    this.sVec = sv;
    this.sys.bump();
  }

  /** calculate_X: the bond-softness term bond_softness d^3 (1 - d)^6 with d = r / bcut. */
  private calculateX(r: number, bcut: number): number {
    const d = r / bcut;
    const d3 = d * d * d;
    const omd = 1.0 - d;
    const omd2 = omd * omd;
    const omd6 = omd2 * omd2 * omd2;
    return this.bondSoftness * d3 * omd6;
  }
}

/** The ACKS2 parameter file: the bond softness on the first line, then "itype chi eta gamma bcut" per type. */
const parseAcks2File = (text: string, nt: number): Acks2Params & { bondSoftness: number } => {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '');
  const first = lines[0];
  if (first === undefined || first.trim().split(/\s+/).length !== 1) throw new StyleError('Fix acks2/reaxff: Incorrect parameter file format');
  const bondSoftness = Number(first.trim());
  const chi = new Float64Array(nt + 1), eta = new Float64Array(nt + 1), gamma = new Float64Array(nt + 1), bcut = new Float64Array(nt + 1);
  for (let i = 1; i <= nt; i++) {
    const line = lines[i];
    if (line === undefined) throw new StyleError('Fix acks2/reaxff: Incorrect parameter file format');
    const w = line.trim().split(/\s+/);
    if (w.length !== 5) throw new StyleError('Fix acks2/reaxff: Incorrect parameter file format');
    const itype = Number(w[0]);
    if (!Number.isInteger(itype) || itype < 1 || itype > nt) throw new StyleError(`Fix acks2/reaxff: invalid atom type in parameter file ${w[0]}`);
    chi[itype] = Number(w[1]);
    eta[itype] = Number(w[2]);
    gamma[itype] = Number(w[3]);
    bcut[itype] = Number(w[4]);
  }
  return { chi, eta, gamma, bcut, bondSoftness };
};

/** Copies the owner value of each ghost, for both the s and the s^X parts (forward communication). */
const forwardAll = (v: Float64Array, n: number, N: number, owner: Int32Array): void => {
  forwardGhosts(v, n, N, owner);
  for (let g = n; g < N; g++) v[N + g] = v[N + owner[g]];
};

/** Folds the ghost entries of a product back onto their owners (reverse communication). */
const foldGhosts = (v: Float64Array, n: number, N: number, owner: Int32Array): void => {
  for (let g = n; g < N; g++) {
    v[owner[g]] += v[g];
    v[N + owner[g]] += v[N + g];
  }
};

/** sparse_matvec_acks2: the product of the extended matrix with x, ghost parts not yet folded. */
const sparseMatvecAcks2 = (w: BiWork, x: Float64Array, b: Float64Array): void => {
  const { grp, N, H, X, eta, typeAll, xdiag } = w;
  b.fill(0);
  for (const i of grp) {
    b[i] = eta[typeAll[i]] * x[i];
    b[N + i] = xdiag[i] * x[N + i];
  }
  b[2 * N] = 0;
  b[2 * N + 1] = 0;
  for (const i of grp) {
    for (let p = H.start[i]; p < H.start[i + 1]; p++) {
      const j = H.jlist[p];
      b[i] += H.val[p] * x[j];
      b[j] += H.val[p] * x[i];
    }
    for (let p = X.start[i]; p < X.start[i + 1]; p++) {
      const j = X.jlist[p];
      b[N + i] += X.val[p] * x[N + j];
      b[N + j] += X.val[p] * x[N + i];
    }
    b[N + i] += x[i];
    b[i] += x[N + i];
    b[2 * N] += x[N + i];
    b[N + i] += x[2 * N];
    b[2 * N + 1] += x[i];
    b[i] += x[2 * N + 1];
  }
};

/** The vector operations of BiCGStab over the owned group and the two Lagrange rows (serial: all rows local). */
const vecOps = (w: BiWork) => {
  const { grp, N } = w;
  const sum = (dest: Float64Array, c: number, v: Float64Array, d: number, y: Float64Array): void => {
    for (const k of grp) {
      dest[k] = c * v[k] + d * y[k];
      dest[N + k] = c * v[N + k] + d * y[N + k];
    }
    dest[2 * N] = c * v[2 * N] + d * y[2 * N];
    dest[2 * N + 1] = c * v[2 * N + 1] + d * y[2 * N + 1];
  };
  const add = (dest: Float64Array, c: number, v: Float64Array): void => {
    for (const k of grp) {
      dest[k] += c * v[k];
      dest[N + k] += c * v[N + k];
    }
    dest[2 * N] += c * v[2 * N];
    dest[2 * N + 1] += c * v[2 * N + 1];
  };
  const copy = (dest: Float64Array, v: Float64Array): void => {
    for (const k of grp) {
      dest[k] = v[k];
      dest[N + k] = v[N + k];
    }
    dest[2 * N] = v[2 * N];
    dest[2 * N + 1] = v[2 * N + 1];
  };
  const dot = (v1: Float64Array, v2: Float64Array): number => {
    let acc = 0;
    for (const k of grp) {
      acc += v1[k] * v2[k];
      acc += v1[N + k] * v2[N + k];
    }
    acc += v1[2 * N] * v2[2 * N];
    acc += v1[2 * N + 1] * v2[2 * N + 1];
    return acc;
  };
  return { sum, add, copy, dot };
};

/**
 * BiCGStab (fix_acks2_reaxff.cpp): solves the extended system for x (the initial guess on entry),
 * with the Jacobi preconditioners of the diagonal of H (s rows) and of X (s^X rows). Returns the
 * iteration count of the C++ function.
 */
const bicgstab = (w: BiWork, b: Float64Array, x: Float64Array, o: { tolerance: number; imax: number; maxwarn: boolean; warn: (t: string) => void; step: number }): number => {
  const { grp, n, N, owner, Hdia_inv, Xdia_inv } = w;
  const L = 2 * N + 2;
  const { sum, add, copy, dot } = vecOps(w);
  const d = new Float64Array(L), r = new Float64Array(L), rHat = new Float64Array(L), p = new Float64Array(L);
  const q = new Float64Array(L), qHat = new Float64Array(L), y = new Float64Array(L), z = new Float64Array(L), g = new Float64Array(L);
  const norm = (v: Float64Array): number => Math.sqrt(dot(v, v));

  sparseMatvecAcks2(w, x, d);
  foldGhosts(d, n, N, owner);
  sum(r, 1.0, b, -1.0, d);
  let bnorm = norm(b);
  let rnorm = norm(r);
  if (bnorm === 0.0) bnorm = 1.0;
  copy(rHat, r);
  let omega = 1.0, rho = 1.0, rhoOld = 0.0, alpha = 0.0;
  let i = 1;
  for (i = 1; i < o.imax && rnorm / bnorm > o.tolerance; ++i) {
    rho = dot(rHat, r);
    if (rho === 0.0) break;
    if (i > 1) {
      const beta = (rho / rhoOld) * (alpha / omega);
      sum(q, 1.0, p, -omega, z);
      sum(p, 1.0, r, beta, q);
    } else {
      copy(p, r);
    }
    for (const j of grp) {
      d[j] = p[j] * Hdia_inv[j];
      d[N + j] = p[N + j] * Xdia_inv[j];
    }
    d[2 * N] = p[2 * N];
    d[2 * N + 1] = p[2 * N + 1];
    forwardAll(d, n, N, owner);
    sparseMatvecAcks2(w, d, z);
    foldGhosts(z, n, N, owner);
    const tmp1 = dot(rHat, z);
    alpha = rho / tmp1;
    sum(q, 1.0, r, -alpha, z);
    const tmp2 = dot(q, q);
    if (tmp2 < o.tolerance) {
      add(x, alpha, d);
      break;
    }
    for (const j of grp) {
      qHat[j] = q[j] * Hdia_inv[j];
      qHat[N + j] = q[N + j] * Xdia_inv[j];
    }
    qHat[2 * N] = q[2 * N];
    qHat[2 * N + 1] = q[2 * N + 1];
    forwardAll(qHat, n, N, owner);
    sparseMatvecAcks2(w, qHat, y);
    foldGhosts(y, n, N, owner);
    const sigma = dot(y, q);
    const tmp3 = dot(y, y);
    omega = sigma / tmp3;
    sum(g, alpha, d, omega, qHat);
    add(x, 1.0, g);
    sum(r, 1.0, q, -omega, y);
    rnorm = norm(r);
    if (omega === 0) break;
    rhoOld = rho;
  }
  if (omega === 0 || rho === 0) {
    o.warn(`Fix acks2/reaxff BiCGStab numerical breakdown, omega = ${omega}, rho = ${rho}`);
  } else if (i >= o.imax) {
    o.warn(`Fix acks2/reaxff BiCGStab convergence failed after ${i} iterations at step ${o.step}`);
  }
  return i;
};

const intArg = (w: string, what: string): number => {
  const v = Number(w);
  if (!Number.isInteger(v)) throw new StyleError(`Illegal fix acks2/reaxff command: expected an integer for ${what}, got '${w}'`);
  return v;
};
const numArg = (w: string, what: string): number => {
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Illegal fix acks2/reaxff command: expected a number for ${what}, got '${w}'`);
  return v;
};
