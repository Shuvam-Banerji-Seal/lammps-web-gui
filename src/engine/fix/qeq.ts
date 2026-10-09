import { Fix } from './fix';
import { StyleError, typeBounds } from '../force/types';
import type { System } from '../system';

/*
 * fix qeq/point and fix qeq/shielded — docs.lammps.org/fix_qeq.html
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
 *   "In order to solve the self-consistent equations for electronegativity
 *    equalization, LAMMPS imposes the additional constraint that all the
 *    charges in the fix group must add up to zero.  The initial charge
 *    assignments should also satisfy this constraint.  LAMMPS will print a
 *    warning if that is not the case."
 *   "The fix qeq styles will print a warning if the charges are not
 *    equilibrated within tolerance by maxiter steps, unless the warn keyword
 *    is used with "no" as argument."
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
 *     scalar.
 *
 * Unsupported qeq styles and qfile keywords throw a StyleError naming them.
 */

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

export class FixQeq extends Fix {
  readonly style: string;
  private readonly shielded: boolean;
  private readonly nEvery: number;
  private readonly cutoff: number;
  private readonly tol: number;
  private readonly maxiter: number;
  private readonly qfile: string;
  private warn: boolean;
  private params: Params | null = null;
  private iterations = 0;
  scalarFlag = true;
  extscalar = 0;

  constructor(sys: System, id: string, group: string, args: string[], shielded: boolean) {
    super(sys, id, group, args);
    this.shielded = shielded;
    this.style = shielded ? 'qeq/shielded' : 'qeq/point';
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
    this.warn = true;
    for (let k = 5; k < a.length;) {
      const key = a[k];
      const val = a[k + 1];
      if (key === 'warn') {
        if (val !== 'yes' && val !== 'no') throw new StyleError(`fix ${this.style}: warn must be yes or no`);
        this.warn = val === 'yes';
        k += 2;
      } else {
        throw new StyleError(`fix ${this.style}: keyword '${key}' is not supported (only warn)`);
      }
    }
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
        const r = Math.sqrt(r2);
        let int: number;
        if (this.shielded) {
          const gi = p.gamma[s.type[i]];
          const gj = p.gamma[s.type[j]];
          const g = Math.sqrt(gi * gj);
          int = g > 0 ? COUL * (r * r * r + 1 / (g * g * g)) ** (-1 / 3) * taper(r / this.cutoff) : 0;
        } else {
          int = 1 / r;
        }
        if (int !== 0) pairs.push({ i, j, int });
      }
    }

    // map owned index -> matrix row
    const row = new Int32Array(s.n).fill(-1);
    for (let a = 0; a < ng; a++) row[idx[a]] = a;

    // rhs: -chi_i - sum over non-group neighbors of H_ij * q_j (fixed charges)
    const n1 = ng + 1;
    const b = new Float64Array(n1);
    for (let a = 0; a < ng; a++) b[a] = -p.chi[s.type[idx[a]]];
    // non-group fixed charges
    if (ng < s.n) {
      const inGroup = (i: number) => (s.mask[i] & bit) !== 0;
      for (let a = 0; a < ng; a++) {
        const i = idx[a];
        for (let k = 0; k < s.n; k++) {
          if (inGroup(k)) continue;
          d[0] = s.x[3 * i] - s.x[3 * k]; d[1] = s.x[3 * i + 1] - s.x[3 * k + 1]; d[2] = s.x[3 * i + 2] - s.x[3 * k + 2];
          this.sys.geom.minimumImage(d);
          const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
          if (!(r2 > 0) || r2 >= cut2) continue;
          const r = Math.sqrt(r2);
          let int: number;
          if (this.shielded) {
            const g = Math.sqrt(p.gamma[s.type[i]] * p.gamma[s.type[k]]);
            int = g > 0 ? COUL * (r * r * r + 1 / (g * g * g)) ** (-1 / 3) * taper(r / this.cutoff) : 0;
          } else int = 1 / r;
          b[a] -= int * s.q[k];
        }
      }
    }

    // H_ii = eta_i; the Lagrange row/col couples the neutrality constraint.
    const etaRow = new Float64Array(ng);
    for (let a = 0; a < ng; a++) etaRow[a] = p.eta[s.type[idx[a]]];

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

    // initial charges should already be neutral (doc note)
    let sum0 = 0;
    for (const i of idx) sum0 += s.q[i];
    if (this.warn && Math.abs(sum0) > 1e-8) {
      this.sys.warn('Fix qeq: initial charge of atoms in the fix group is not zero; charge equilibration enforces neutrality');
    }

    for (let a = 0; a < ng; a++) s.q[idx[a]] = x[a];
    this.sys.nb.refreshCharges(s);
    this.sys.bump();
  }
}

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
