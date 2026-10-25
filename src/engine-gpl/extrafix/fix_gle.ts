/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_gle.cpp and fix_gle.h (FixGLE: the constructor, init, init_gle, init_gles,
 * gle_integrate, initial_integrate, final_integrate, reset_dt, reset_target, and the GLE utility functions
 * StabCholesky, AkMult, MyTrans, MatrixExp) at commit b332f28, with the RanMars generator of
 * src/random_mars.cpp (src/engine-gpl/random_mars.ts). Contributing authors of the source: Michele Ceriotti (EPFL),
 * Joe Morrone (Stony Brook), Axel Kohlmeyer (Temple U). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Generalized Langevin equation thermostat (Ceriotti, Bussi, Parrinello, PRL 103, 190601 (2010)): each atom
 * carries ns auxiliary momenta per Cartesian direction; the propagation of (v, s) is an exact Ornstein-Uhlenbeck
 * step with the matrices T = exp(-A dt/2) and S with S S^T = C - T C T^T. Native order: the random draws follow
 * the atoms in native storage order (src/engine/atoms.ts nativeOrder), as in the source.
 *
 * Not supported (refused by the engine with run_style respa, since the respa hooks are not implemented here):
 * the restart of the auxiliary momenta and the global scalar (the source does not set scalar_flag either).
 */
import { StyleError, potentialFile } from '../../engine/force/types';
import { inumeric, numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';
import { rampFraction } from '../../engine/fix/util';
import { massOf, nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { pow as libmPow } from '../libm/pow';

/** Row-major n x n index, as the source's midx(n,i,j). */
const midx = (n: number, i: number, j: number) => i * n + j;

/**
 * StabCholesky: LDL^T of MMt with negative diagonal pivots set to zero, then M = L sqrt(D), so that M M^T is the
 * stabilized MMt (the source's GLE::StabCholesky).
 */
export const stabCholesky = (n: number, MMt: Float64Array, M: Float64Array): void => {
  const L = new Float64Array(n * n);
  const D = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    L[midx(n, i, i)] = 1.0;
    for (let j = 0; j < i; j++) {
      L[midx(n, i, j)] = MMt[midx(n, i, j)];
      for (let k = 0; k < j; k++) L[midx(n, i, j)] -= L[midx(n, i, k)] * L[midx(n, j, k)] * D[k];
      if (D[j] !== 0) L[midx(n, i, j)] /= D[j];
      else L[midx(n, i, j)] = 0.0;
    }
    D[i] = MMt[midx(n, i, i)];
    for (let k = 0; k < i; k++) D[i] -= L[midx(n, i, k)] * L[midx(n, i, k)] * D[k];
  }
  for (let i = 0; i < n; i++) D[i] = D[i] > 0.0 ? Math.sqrt(D[i]) : 0.0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[midx(n, i, j)] = L[midx(n, i, j)] * D[j];
};

/**
 * AkMult: C = cf * C + A B for A (n x r), B (r x m), C (n x m), row-major; the inner sum runs over k in order
 * (the source's blocked product accumulates the same sums, for blocks of 64).
 */
export const akMult = (n: number, m: number, r: number, A: Float64Array, B: Float64Array, C: Float64Array, cf = 0.0): void => {
  for (let i = 0; i < n * m; i++) C[i] *= cf;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let tmp = 0.0;
      for (let k = 0; k < r; k++) tmp += A[midx(r, i, k)] * B[midx(m, k, j)];
      C[midx(m, i, j)] += tmp;
    }
  }
};

/** MyTrans: transpose of the n x n matrix A into AT. */
export const transposeN = (n: number, A: Float64Array, AT: Float64Array): void => {
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) AT[j * n + i] = A[i * n + j];
};

/** MatrixExp: exp(M) by a Taylor series of the matrix scaled by 2^-k, then k squarings (the source's defaults j = k = 8). */
export const matrixExp = (n: number, M: Float64Array, EM: Float64Array, j = 8, k = 8): void => {
  const tc = new Float64Array(j + 1);
  const SM = new Float64Array(n * n);
  const TMP = new Float64Array(n * n);
  const onetotwok = libmPow(0.5, 1.0 * k);
  tc[0] = 1;
  for (let i = 0; i < j; i++) tc[i + 1] = tc[i] / (i + 1.0);
  for (let i = 0; i < n * n; i++) { SM[i] = M[i] * onetotwok; EM[i] = 0.0; TMP[i] = 0.0; }
  for (let i = 0; i < n; i++) EM[midx(n, i, i)] = tc[j];
  for (let p = j - 1; p >= 0; p--) {
    akMult(n, n, n, SM, EM, TMP);
    for (let i = 0; i < n * n; i++) EM[i] = TMP[i];
    for (let i = 0; i < n; i++) EM[midx(n, i, i)] += tc[p];
  }
  for (let p = 0; p < k; p++) {
    akMult(n, n, n, EM, EM, TMP);
    for (let i = 0; i < n * n; i++) EM[i] = TMP[i];
  }
};

/**
 * Reads n numbers from a potential-style text file the way PotentialFileReader::next_dvector does: comments
 * (# to the end of the line) and blank lines are skipped, numbers are taken in order, extra numbers are ignored.
 */
const readDVector = (text: string, n: number, what: string, units: string): Float64Array => {
  // PotentialFileReader: a UNITS: tag must name the unit style in use (potential_file_reader.cpp refuses a mismatch)
  const tag = /UNITS:\s*(\S+)/.exec(text.split('\n')[0] ?? '');
  if (tag && tag[1] !== units) throw new StyleError(`${what}: potential file requires ${tag[1]} units but ${units} units are in use`);
  const out = new Float64Array(n);
  let i = 0;
  for (const line of text.split('\n')) {
    if (i >= n) break;
    const words = line.replace(/#.*/, '').trim().split(/\s+/).filter((w) => w.length > 0);
    for (const w of words) {
      if (i >= n) break;
      const x = Number(w);
      if (!Number.isFinite(x)) throw new StyleError(`Error reading ${what}: '${w}' is not a number`);
      out[i++] = x;
    }
  }
  if (i < n) throw new StyleError(`Error reading ${what}: expected ${n} numbers, found ${i}`);
  return out;
};

export class FixGLE extends Fix {
  readonly style = 'gle';
  private readonly ns: number;
  private readonly ns1sq: number;
  private readonly A: Float64Array;
  private readonly C: Float64Array;
  private readonly S: Float64Array;
  private readonly T: Float64Array;
  private readonly TT: Float64Array;
  private readonly ST: Float64Array;
  private readonly tStart: number;
  private readonly tStop: number;
  private tTarget: number;
  private readonly fnoneq: boolean;
  private readonly gleEvery: number;
  private gleStep = 0;
  private dogle = 0;
  private dtv = 0;
  private dtf = 0;
  private energyExchanged = 0;
  private readonly rng: RanMars;
  /** Auxiliary momenta: 3 * ns per atom (storage index), k * ns + j. */
  private readonly gleS: Float64Array;
  private readonly nAtoms: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.timeIntegrate = true;
    if (args.length < 5) throw new StyleError('Illegal fix gle command. Expecting: fix <fix-ID> <group-ID> gle <ns> <Tstart> <Tstop> <seed> <Amatrix>');
    // As native LAMMPS does (src/EXTRA-FIX/fix_gle.cpp FixGLE::FixGLE at b332f28): utils::inumeric reads ns, then
    // utils::numeric the two temperatures and utils::inumeric the seed, and only then the A matrix file is read.
    const ns = inumeric(args[0]);
    // ns 0 (no auxiliary momenta) runs; a negative ns makes native fail to allocate the per-atom array gle:gle_s (a size
    // that depends on its memory growth), which the engine refuses with its own message
    if (ns < 0) throw new StyleError('Illegal fix gle number of auxiliary momenta (ns)');
    this.ns = ns;
    this.ns1sq = (ns + 1) * (ns + 1);
    this.tStart = numeric(args[1]);
    this.tStop = numeric(args[2]);
    const seed = inumeric(args[3]);
    if (seed <= 0) throw new StyleError('Illegal fix gle command');
    this.A = readDVector(sys.readFile(args[4], potentialFile('fix gle A matrix')), this.ns1sq, 'fix gle A matrix', sys.state.units.style);
    this.C = new Float64Array(this.ns1sq);
    this.S = new Float64Array(this.ns1sq);
    this.T = new Float64Array(this.ns1sq);
    this.TT = new Float64Array(this.ns1sq);
    this.ST = new Float64Array(this.ns1sq);
    // keyword pairs: noneq cmatrix, every nmts; other words are skipped (see below)
    let noneqFile: string | null = null;
    let every = 1;
    for (let iarg = 5; iarg < args.length; iarg += 2) {
      const key = args[iarg];
      if (key === 'noneq') {
        if (iarg + 1 >= args.length) throw new StyleError('Did not specify C matrix for non-equilibrium GLE');
        noneqFile = args[iarg + 1];
      } else if (key === 'every') {
        if (iarg + 1 >= args.length) throw new StyleError('Did not specify interval for applying the GLE');
        every = Number(args[iarg + 1]);
        if (!Number.isInteger(every)) throw new StyleError('Illegal fix gle every value');
      }
      // any other word is skipped in pairs, as native does (fix_gle.cpp FixGLE::FixGLE at b332f28: the loop
      // only tests noneq and every, then steps two words at a time)
    }
    this.fnoneq = noneqFile !== null;
    this.gleEvery = every;
    this.tTarget = this.tStart;
    const cfac = sys.state.units.boltz / sys.state.units.mvv2e;
    if (!this.fnoneq) {
      const kT = this.tTarget * cfac;
      for (let i = 0; i < this.ns1sq; i += ns + 2) this.C[i] = kT;
    } else {
      const c = readDVector(sys.readFile(noneqFile!, potentialFile('fix gle C matrix')), this.ns1sq, 'fix gle C matrix', sys.state.units.style);
      for (let i = 0; i < this.ns1sq; i++) this.C[i] = c[i] * cfac;
    }
    this.rng = new RanMars(seed);
    this.nAtoms = sys.state.n;
    this.gleS = new Float64Array(3 * ns * this.nAtoms);
    this.initGles();
  }

  /** Sets the initial auxiliary momenta of the group's atoms (the source's init_gles, run once at fix creation). */
  private initGles(): void {
    const s = this.sys.state;
    const n = this.ns + 1;
    const order = nativeOrder(s);
    const rootC = new Float64Array(this.ns1sq);
    const rootCT = new Float64Array(this.ns1sq);
    stabCholesky(n, this.C, rootC);
    transposeN(n, rootC, rootCT);
    const nl = this.nAtoms;
    // random rows: 3 per atom in native order, ns + 1 gaussians each (all atoms, not only the group)
    const newg = new Float64Array(nl * 3 * n);
    for (let i = 0; i < nl * 3 * n; i++) newg[i] = this.rng.gaussian();
    const news = new Float64Array(nl * 3 * n);
    akMult(nl * 3, n, n, newg, rootCT, news);
    // the source consumes the flat array with stride ns per (group atom, direction)
    let nk = 0;
    for (let p = 0; p < nl; p++) {
      const i = order[p];
      if (!(s.mask[i] & this.groupBit)) continue;
      for (let k = 0; k < 3; k++) {
        for (let j = 0; j < this.ns; j++) this.gleS[i * 3 * this.ns + k * this.ns + j] = news[nk++];
      }
    }
  }

  /** Recomputes the propagation matrices for the current dt and temperature (the source's init_gle). */
  private initGle(): void {
    const n = this.ns + 1;
    const tmp1 = new Float64Array(this.ns1sq);
    const tmp2 = new Float64Array(this.ns1sq);
    const dt = this.sys.state.dt;
    for (let i = 0; i < this.ns1sq; i++) {
      tmp1[i] = -this.A[i] * dt * 0.5 * this.gleEvery;
      tmp2[i] = 0.0;
      this.S[i] = 0.0;
    }
    matrixExp(n, tmp1, this.T);
    akMult(n, n, n, this.T, this.C, tmp1);
    transposeN(n, this.T, tmp2);
    akMult(n, n, n, tmp1, tmp2, this.S);
    for (let i = 0; i < this.ns1sq; i++) tmp1[i] = this.C[i] - this.S[i];
    stabCholesky(n, tmp1, this.S);
    transposeN(n, this.T, this.TT);
    transposeN(n, this.S, this.ST);
  }

  private checkAtoms(): void {
    if (this.sys.state.n !== this.nAtoms) throw new StyleError('fix gle: the number of atoms changed after the fix was defined (not supported)');
  }

  init(): void {
    this.checkAtoms();
    this.dogle = 1;
    this.resetDt();
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    this.initGle();
  }

  /** The source's gle_integrate: one exact OU step of (v, s) for every group atom. */
  private gleIntegrate(): void {
    this.checkAtoms();
    const s = this.sys.state;
    const n = this.ns + 1;
    const v = s.v;
    const order = nativeOrder(s);
    const atoms: number[] = [];
    for (let p = 0; p < s.n; p++) if (s.mask[order[p]] & this.groupBit) atoms.push(order[p]);
    const ni = atoms.length;
    const tmp1 = new Float64Array(ni * 3 * n);
    const tmp2 = new Float64Array(ni * 3 * n);
    let deltae = 0.0;
    let nk = 0;
    for (const i of atoms) {
      const smi = Math.sqrt(massOf(s, i));
      for (let k = 0; k < 3; k++) {
        // the velocity (mass-scaled) and the auxiliary momenta of this direction
        tmp2[nk] = v[3 * i + k] * smi;
        deltae += tmp2[nk] * tmp2[nk];
        nk++;
        for (let j = 0; j < this.ns; j++) tmp2[nk++] = this.gleS[i * 3 * this.ns + k * this.ns + j];
      }
    }
    // s(t+dt) = T s
    akMult(ni * 3, n, n, tmp2, this.TT, tmp1);
    // ... + S xi
    for (let i = 0; i < 3 * ni * n; i++) tmp2[i] = this.rng.gaussian();
    akMult(ni * 3, n, n, tmp2, this.ST, tmp1, 1.0);
    nk = 0;
    for (const i of atoms) {
      const ismi = 1.0 / Math.sqrt(massOf(s, i));
      for (let k = 0; k < 3; k++) {
        v[3 * i + k] = tmp1[nk] * ismi;
        deltae -= tmp1[nk] * tmp1[nk];
        nk++;
        for (let j = 0; j < this.ns; j++) this.gleS[i * 3 * this.ns + k * this.ns + j] = tmp1[nk++];
      }
    }
    this.energyExchanged += deltae * 0.5 * s.units.mvv2e;
  }

  initialIntegrate(): void {
    this.checkAtoms();
    const s = this.sys.state;
    this.gleStep--;
    if (this.dogle && this.gleStep < 1) this.gleIntegrate();
    const { x, v, f, mask } = s;
    const dtf = this.dtf, dtv = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const dtfm = dtf / massOf(s, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      x[3 * i] += dtv * v[3 * i];
      x[3 * i + 1] += dtv * v[3 * i + 1];
      x[3 * i + 2] += dtv * v[3 * i + 2];
    }
  }

  finalIntegrate(): void {
    this.checkAtoms();
    const s = this.sys.state;
    const { v, f, mask } = s;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const dtfm = dtf / massOf(s, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
    }
    if (this.dogle && this.gleStep < 1) {
      this.gleIntegrate();
      this.gleStep = this.gleEvery;
    }
    // the target temperature of the next step on the Tstart..Tstop ramp
    this.tTarget = this.tStart + rampFraction(this.sys) * (this.tStop - this.tStart);
    if (this.tStop !== this.tStart && !this.fnoneq) {
      const kT = this.tTarget * s.units.boltz / s.units.mvv2e;
      this.C.fill(0);
      for (let i = 0; i < this.ns1sq; i += this.ns + 2) this.C[i] = kT;
      this.initGle();
    }
  }

  /** Global energy exchanged with the auxiliary momenta (the source keeps it; it is not exposed as f_ID here). */
  get exchangedEnergy(): number { return this.energyExchanged; }

  /** Target temperature of the current step (the source's extract("t_target")). */
  get targetTemperature(): number { return this.tTarget; }
}
