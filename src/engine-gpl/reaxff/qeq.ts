/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/fix_qeq_reaxff.cpp and src/REAXFF/fix_qeq_reaxff.h (FixQEqReaxFF: constructor,
 * pertype_parameters, init, init_shielding, init_taper, setup_pre_force, init_storage, pre_force,
 * init_matvec, compute_H, calculate_H, CG, sparse_matvec, calculate_Q), at commit b332f28. Authors
 * credited in the C++ file headers: Hasan Metin Aktulga (Purdue University, now Lawrence Berkeley
 * National Laboratory); hybrid and sub-group capabilities: Ray Shan (Sandia). Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Refused (StyleError): the dual, matfree and efield couplings. The fictitious-charge histories
 * are keyed by atom id (the engine sorts atoms at reneighboring; LAMMPS keeps them with the atoms
 * through the Atom callbacks). The H matrix is built from the owned and ghost positions of the
 * engine (a binned-free O(n N) scan over the owned i and the j after i), with the same flag rule
 * as LAMMPS compute_H, instead of the pair style's neighbor list.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairReaxFF } from './pair';

const EV_TO_KCAL_PER_MOL = 14.4;
const EPSILON = 0.0001;
const NPREV = 4;

/** Per-type QEq parameters, 1-based (chi, eta, gamma). */
export interface QeqParams { chi: Float64Array; eta: Float64Array; gamma: Float64Array }

/** The Taper polynomial of fix_qeq_reaxff.cpp init_taper (not core/nonbonded's taper: the fix has its own swa/swb). */
const taperFix = (swa: number, swb: number): Float64Array => {
  const Tap = new Float64Array(8);
  const d7 = Math.pow(swb - swa, 7);
  const swa2 = swa * swa, swa3 = swa * swa * swa;
  const swb2 = swb * swb, swb3 = swb * swb * swb;
  Tap[7] = 20.0 / d7;
  Tap[6] = (-70.0 * (swa + swb)) / d7;
  Tap[5] = (84.0 * (swa2 + 3.0 * swa * swb + swb2)) / d7;
  Tap[4] = (-35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3)) / d7;
  Tap[3] = (140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3)) / d7;
  Tap[2] = (-210.0 * (swa3 * swb2 + swa2 * swb3)) / d7;
  Tap[1] = (140.0 * swa3 * swb3) / d7;
  Tap[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2 - 7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb) / d7;
  return Tap;
};

/** fix qeq/reaxff (FixQEqReaxFF): charge equilibration with the ReaxFF shielded Coulomb matrix. */
export class FixQEqReaxFF extends Fix {
  readonly style = 'qeq/reaxff';

  private swa: number;
  private swb: number;
  private tolerance: number;
  private imax = 200;
  private maxwarn = true;
  private params: QeqParams | null = null;
  private pertype: string;
  private Tap: Float64Array = new Float64Array(8);
  private shld = new Float64Array(0);
  private ntypes = 0;
  /** fictitious-charge histories keyed by atom id: [s_hist 0..3, t_hist 4..7] (most recent first). */
  private hist = new Map<number, Float64Array>();
  private matvecs = 0;
  private matvecsS = 0;
  private matvecsT = 0;

  constructor(sys: System, id: string, group: string, args: string[], private pairRef: () => PairReaxFF | null) {
    super(sys, id, group, args);
    this.scalarFlag = true;
    this.extscalar = 0;
    // narg of LAMMPS is 8..12 with "fix ID group style" in front: 5..9 arguments here
    if (args.length < 5 || args.length > 9) throw new StyleError('Illegal fix qeq/reaxff command');
    this.pairRef()?.attachQeqFix();
    this.nevery = intArg(args[0], 'Nevery');
    if (this.nevery <= 0) throw new StyleError('Illegal fix qeq/reaxff command');
    this.swa = numArg(args[1], 'swa');
    this.swb = numArg(args[2], 'swb');
    this.tolerance = numArg(args[3], 'tolerance');
    this.pertype = args[4];
    let iarg = 5;
    while (iarg < args.length) {
      const k = args[iarg];
      if (k === 'nowarn') this.maxwarn = false;
      else if (k === 'maxiter') {
        if (iarg + 1 > args.length - 1) throw new StyleError('Illegal fix qeq/reaxff command: maxiter needs a value');
        this.imax = intArg(args[iarg + 1], 'maxiter');
        iarg++;
      } else if (k === 'dual') throw new StyleError('fix qeq/reaxff keyword dual is not ported (OPENMP variant only)');
      else if (k === 'matfree') throw new StyleError('fix qeq/reaxff keyword matfree is not ported (Kokkos variant only)');
      else throw new StyleError(`Illegal fix qeq/reaxff command: unknown keyword ${k}`);
      iarg++;
    }
  }

  override init(): void {
    const s = this.sys.state;
    const st = this.sys.state;
    let count = 0;
    for (let i = 0; i < st.n; i++) if (st.mask[i] & this.groupBit) count++;
    if (count === 0) throw new StyleError('Fix qeq/reaxff group has no atoms');
    if (this.swb < 0) throw new StyleError('Fix qeq/reaxff has negative upper Taper radius cutoff');
    // LAMMPS adds the electric field of fix efield to the electronegativities (chi_field); that coupling is not
    // ported, so the combination is refused rather than run without the field
    if (this.sys.fixes.some((f) => f.style.startsWith('efield'))) {
      throw new StyleError('fix qeq/reaxff with fix efield is not supported by the browser engine (the field term of the charge equilibration is not ported)');
    }
    this.ntypes = s.ntypes;
    this.Tap = taperFix(this.swa, this.swb);
    this.pertypeParameters();
    const p = this.params!;
    const nt = this.ntypes;
    this.shld = new Float64Array((nt + 1) * (nt + 1));
    for (let i = 1; i <= nt; i++) {
      for (let j = 1; j <= nt; j++) this.shld[i * (nt + 1) + j] = Math.pow(p.gamma[i] * p.gamma[j], -1.5);
    }
  }

  /** pertype_parameters: the chi, eta, gamma columns of the file, or of the pair style (keyword reaxff). */
  private pertypeParameters(): void {
    if (this.params) return;
    const nt = this.ntypes;
    if (this.pertype.startsWith('reaxff')) {
      const pair = this.pairRef();
      if (!pair) throw new StyleError('No reaxff pair style for fix qeq/reaxff');
      this.params = pair.qeqParameters(nt);
      return;
    }
    if (this.pertype.startsWith('reax/c')) throw new StyleError("Fix qeq/reaxff keyword 'reax/c' is obsolete; please use 'reaxff'");
    const text = this.sys.readFile(this.pertype);
    const chi = new Float64Array(nt + 1), eta = new Float64Array(nt + 1), gamma = new Float64Array(nt + 1);
    const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '');
    for (let i = 1; i <= nt; i++) {
      const line = lines[i - 1];
      if (line === undefined) throw new StyleError('Fix qeq/reaxff: Invalid param file format');
      const w = line.trim().split(/\s+/);
      if (w.length !== 4) throw new StyleError('Fix qeq/reaxff: Incorrect format of param file');
      const itype = Number(w[0]);
      if (!Number.isInteger(itype) || itype < 1 || itype > nt) throw new StyleError(`Fix qeq/reaxff: invalid atom type in param file ${w[0]}`);
      chi[itype] = Number(w[1]);
      eta[itype] = Number(w[2]);
      gamma[itype] = Number(w[3]);
    }
    this.params = { chi, eta, gamma };
  }

  /** setup_pre_force: the storage of the solve, then one pre_force. */
  override setup(): void { this.preForce(); }

  override preForce(): void {
    if (this.sys.state.step % this.nevery !== 0) return;
    this.solve();
  }

  override computeScalar(): number { return this.matvecs / 2.0; }

  /**
   * pre_force: compute_H, the two CG solves (s with b = -chi, t with b = -1) and calculate_Q.
   */
  private solve(): void {
    if (!this.params) this.init();
    const s = this.sys.state;
    const nb = this.sys.nb;
    const n = nb.nlocal;
    const N = nb.nall;
    const p = this.params!;
    const bit = this.groupBit;
    const x = nb.xall;
    const owner = nb.owner;
    const typeAll = nb.typeall;
    // atom ids of owned and ghost atoms (tags)
    const orig = new Int32Array(N);
    for (let a = 0; a < N; a++) orig[a] = s.id[owner[a]];
    const grp: number[] = [];
    for (let i = 0; i < n; i++) if (s.mask[i] & bit) grp.push(i);
    const inGroup = new Uint8Array(N);
    for (const i of grp) inGroup[i] = 1;

    // compute_H: row i (group atoms) holds the j > i (owned or ghost) within swb that pass the flag rule
    const rowStart = new Int32Array(N + 1);
    const jl: number[] = [], vl: number[] = [];
    const swb2 = this.swb * this.swb;
    const nt1 = this.ntypes + 1;
    for (let i = 0; i < n; i++) {
      rowStart[i] = jl.length;
      if (!inGroup[i]) continue;
      for (let j = i + 1; j < N; j++) {
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        let flag = false;
        if (rsq <= swb2) {
          if (j < n) flag = true;
          else if (orig[i] < orig[j]) flag = true;
          else if (orig[i] === orig[j]) {
            if (dz > EPSILON) flag = true;
            else if (Math.abs(dz) < EPSILON) {
              if (dy > EPSILON) flag = true;
              else if (Math.abs(dy) < EPSILON && dx > EPSILON) flag = true;
            }
          }
        }
        if (flag) {
          jl.push(j);
          vl.push(this.calculateH(Math.sqrt(rsq), this.shld[typeAll[i] * nt1 + typeAll[j]]));
        }
      }
    }
    // ghost rows are empty (the list holds the owned i only); row i is [start[i], start[i+1])
    for (let i = n; i <= N; i++) rowStart[i] = jl.length;
    const H = { start: rowStart, jlist: Int32Array.from(jl), val: Float64Array.from(vl) };

    // init_matvec: preconditioner, sources and extrapolated starting guesses
    const Hdia_inv = new Float64Array(N);
    const bS = new Float64Array(N), bT = new Float64Array(N);
    const sv = new Float64Array(N), tv = new Float64Array(N);
    for (const i of grp) {
      Hdia_inv[i] = 1.0 / p.eta[typeAll[i]];
      bS[i] = -p.chi[typeAll[i]];
      bT[i] = -1.0;
      const h = this.histOf(orig[i]);
      tv[i] = h[4 + 2] + 3 * (h[4] - h[4 + 1]);
      sv[i] = 4 * (h[0] + h[2]) - (6 * h[1] + h[3]);
    }
    const ghosts = (v: Float64Array) => { for (let g = n; g < N; g++) v[g] = v[owner[g]]; };
    ghosts(sv);
    ghosts(tv);

    const work: CgWork = { Hdia_inv, grp, owner, n, N, H, eta: p.eta, typeAll, inGroup };
    this.matvecsS = this.cg(work, bS, sv);
    this.matvecsT = this.cg(work, bT, tv);
    this.matvecs = this.matvecsS + this.matvecsT;

    // calculate_Q
    let sSum = 0, tSum = 0;
    for (const i of grp) { sSum += sv[i]; tSum += tv[i]; }
    const u = sSum / tSum;
    for (const i of grp) {
      s.q[i] = sv[i] - u * tv[i];
      const h = this.histOf(orig[i]);
      for (let k = NPREV - 1; k > 0; k--) {
        h[k] = h[k - 1];
        h[4 + k] = h[4 + k - 1];
      }
      h[0] = sv[i];
      h[4] = tv[i];
    }
    // charges of the ghosts follow their owners (forward comm of pack_flag 4)
    nb.refreshCharges(s);
    this.sys.bump();
  }

  private histOf(id: number): Float64Array {
    let h = this.hist.get(id);
    if (!h) {
      h = new Float64Array(2 * NPREV);
      this.hist.set(id, h);
    }
    return h;
  }

  /** calculate_H: the taper times 14.4 over the shielded distance (fix_qeq_reaxff.cpp). */
  private calculateH(r: number, gamma: number): number {
    const T = this.Tap;
    let taper = T[7] * r + T[6];
    taper = taper * r + T[5];
    taper = taper * r + T[4];
    taper = taper * r + T[3];
    taper = taper * r + T[2];
    taper = taper * r + T[1];
    taper = taper * r + T[0];
    let denom = r * r * r + gamma;
    denom = Math.pow(denom, 1.0 / 3.0);
    return (taper * EV_TO_KCAL_PER_MOL) / denom;
  }

  /** sparse_matvec: b = H x over the owned group atoms, with the ghost contributions folded back to the owners. */
  private matvec(w: CgWork, x: Float64Array, b: Float64Array): void {
    const { grp, n, N, owner, H, eta, typeAll } = w;
    b.fill(0, 0, N);
    for (const i of grp) b[i] = eta[typeAll[i]] * x[i];
    for (const i of grp) {
      for (let p = H.start[i]; p < H.start[i + 1]; p++) {
        const j = H.jlist[p];
        b[i] += H.val[p] * x[j];
        b[j] += H.val[p] * x[i];
      }
    }
    // reverse communication: ghost contributions go to their owners
    for (let g = n; g < N; g++) b[owner[g]] += b[g];
  }

  /** CG (fix_qeq_reaxff.cpp): Jacobi-preconditioned conjugate gradient; returns the iteration count. */
  private cg(w: CgWork, b: Float64Array, x: Float64Array): number {
    const { grp, N, Hdia_inv, owner } = w;
    const r = new Float64Array(N), d = new Float64Array(N), p = new Float64Array(N), q = new Float64Array(N);
    const fwd = (v: Float64Array) => { for (let g = w.n; g < N; g++) v[g] = v[owner[g]]; };
    fwd(x);
    this.matvec(w, x, q);
    for (const j of grp) r[j] = b[j] - q[j];
    for (const j of grp) d[j] = r[j] * Hdia_inv[j];
    let bn = 0;
    for (const j of grp) bn += b[j] * b[j];
    const bNorm = Math.sqrt(bn);
    let sigNew = 0;
    for (const j of grp) sigNew += r[j] * d[j];
    let i = 1;
    for (; i < this.imax && Math.sqrt(sigNew) / bNorm > this.tolerance; ++i) {
      fwd(d);
      this.matvec(w, d, q);
      let tmp = 0;
      for (const j of grp) tmp += d[j] * q[j];
      const alpha = sigNew / tmp;
      for (const j of grp) x[j] += alpha * d[j];
      for (const j of grp) r[j] += -alpha * q[j];
      for (const j of grp) p[j] = r[j] * Hdia_inv[j];
      const sigOld = sigNew;
      sigNew = 0;
      for (const j of grp) sigNew += r[j] * p[j];
      const beta = sigNew / sigOld;
      for (const j of grp) d[j] = 1 * p[j] + beta * d[j];
    }
    if (i >= this.imax && this.maxwarn) {
      this.sys.warn(`Fix qeq/reaxff CG convergence failed after ${i} iterations at step ${this.sys.state.step}`);
    }
    return i;
  }

  /** The QEq parameters of the fix (for tests and the notebook). */
  qeqParams(): QeqParams | null { return this.params; }
}

interface CgWork {
  grp: number[];
  n: number;
  N: number;
  owner: Int32Array;
  H: { start: Int32Array; jlist: Int32Array; val: Float64Array };
  eta: Float64Array;
  typeAll: Int32Array;
  inGroup: Uint8Array;
  Hdia_inv: Float64Array;
}

const intArg = (w: string, what: string): number => {
  const v = Number(w);
  if (!Number.isInteger(v)) throw new StyleError(`Illegal fix qeq/reaxff command: expected an integer for ${what}, got '${w}'`);
  return v;
};
const numArg = (w: string, what: string): number => {
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Illegal fix qeq/reaxff command: expected a number for ${what}, got '${w}'`);
  return v;
};
