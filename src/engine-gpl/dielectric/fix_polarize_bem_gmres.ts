/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/fix_polarize_bem_gmres.cpp and fix_polarize_bem_gmres.h at commit b332f28 (Trung Nguyen and Monica Olvera de la Cruz,
 * Northwestern University, and the DIELECTRIC package authors; the GMRES solver is adapted from C++ code by John
 * Burkardt, after Lili Ju's C version). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * fix polarize/bem/gmres N tol: surface induced charges of the interface patches (the group) from the boundary element
 * equation of Barros, Sinkovits, Luijten, J. Chem. Phys. 2014, 140, 064903, solved with GMRES (restarted at the full
 * dimension unless fix_modify mr sets a smaller value). Each evaluation runs the force field with a trial induced
 * charge; the field of the pair term (and of a pppm/dielectric kspace term) gives the operator. Reproduces the LAMMPS
 * update order: the induced charges are solved in pre_force every N steps, and once in setup. Not ported: fix_modify
 * rand (random initial charges) is refused.
 */
import { Fix } from '../../engine/fix/fix';
import { parseNum } from '../../engine/force/util';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { dielArrays, dielectricPairField, epsilon0e2q, fieldAt, type DielArrays } from './common';
import { KSpacePPPMDielectric } from './pppm_dielectric';

const MY_4PI = 4 * Math.PI;

/** Pair styles this fix accepts (the others are refused in setup, as LAMMPS does). */
const pairFieldOf = (sys: System): Float64Array | null => dielectricPairField(sys.ff.pair);

export class FixPolarizeBEMGMRES extends Fix {
  readonly style = 'polarize/bem/gmres';
  vectorFlag = true;
  sizeVector = 2;

  private readonly every: number;
  private readonly tolAbs: number;
  private readonly tolRel: number;
  private itrMax = 50;
  /** Krylov dimension (0 = full, the number of induced charges minus one). */
  private mr = 0;
  private kspaceOn = true;
  private iterations = 0;
  private rho = 0;
  private normb = 0;
  /** Induced-charge index of each owned atom (-1 outside the group), and the number of induced charges. */
  private idx = new Int32Array(0);
  private m = 0;
  private qBackup = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError(`usage: fix ${id} group-ID polarize/bem/gmres N tol`);
    const every = Number(args[0]);
    if (!Number.isInteger(every) || every < 0) throw new StyleError(`Illegal fix ${id} polarize/bem/gmres command: N must be an integer >= 0`);
    this.every = every;
    this.tolAbs = this.tolRel = parseNum(args[1], 'tolerance');
  }

  /** Owned atoms of the group numbered by atom ID (LAMMPS tag order); -1 outside the group. */
  private mapping(): void {
    const s = this.sys.state;
    const ids: number[] = [];
    for (let i = 0; i < s.n; i++) if (this.inGroup(i)) ids.push(s.id[i]);
    ids.sort((a, b) => a - b);
    const pos = new Map<number, number>();
    ids.forEach((id, k) => pos.set(id, k));
    this.idx = new Int32Array(s.n);
    for (let i = 0; i < s.n; i++) this.idx[i] = this.inGroup(i) ? pos.get(s.id[i])! : -1;
    this.m = ids.length;
  }

  /** Force evaluation at the current charges; the pair (and kspace) fields are then current. */
  private evaluate(): void {
    this.sys.computeForcesInRun({});
  }

  /** The kspace field (pppm/dielectric) when the fix uses kspace (fix_modify kspace yes), else null. */
  private kspaceField(): Float64Array | null {
    const k = this.sys.ff.kspace;
    return k && this.kspaceOn && k instanceof KSpacePPPMDielectric ? k.efield : null;
  }

  /** The field at owned atom i (pair plus kspace) into out. */
  private field(i: number, out: Float64Array): void {
    const pe = pairFieldOf(this.sys);
    if (!pe) throw new StyleError(`Pair style not compatible with fix ${this.id} (${this.style})`);
    fieldAt(pe, this.kspaceField(), i, out);
  }

  setup(): void {
    this.checkStyles();
    // LAMMPS: the induced charges are set after the run's force evaluation; the last evaluation inside the solver
    // (update_residual at the final induced charges) leaves the energies and forces the run reports
    this.computeInduced();
  }

  preForce(): void {
    if (this.every === 0) return;
    if (this.sys.state.step % this.every) return;
    this.computeInduced();
  }

  /** Pair and kspace styles that carry the field the fix reads (LAMMPS setup() error texts). */
  private checkStyles(): void {
    if (!pairFieldOf(this.sys)) throw new StyleError(`Pair style not compatible with fix ${this.id} (${this.style})`);
    const k = this.sys.ff.kspace;
    if (k && this.kspaceOn && !(k instanceof KSpacePPPMDielectric)) throw new StyleError(`Kspace style not compatible with fix ${this.id} (${this.style})`);
  }

  private computeInduced(): void {
    const s = this.sys.state;
    const n = s.n;
    const d: DielArrays = dielArrays(s, `fix ${this.id}`);
    this.mapping();
    const e0 = epsilon0e2q(s.units.style);
    const qs = d.qs, q = s.q, idx = this.idx, m = this.m;
    if (m === 0) return;
    // the real (scaled) charges stay in qBackup; the right-hand side is computed with zero induced charges
    this.qBackup = Float64Array.from(qs.subarray(0, n));
    for (let i = 0; i < n; i++) if (idx[i] >= 0) qs[i] = 0;
    this.evaluate();
    const rhs = this.rhsFromField(e0, d, q, idx, m);
    const induced = new Float64Array(m);
    const r = Float64Array.from(rhs);
    this.normb = Math.sqrt(dot(rhs, rhs));
    // LAMMPS returns here without restoring the interface charges (they stay at zero)
    if (this.normb < this.tolAbs) return;
    this.gmres(induced, r, e0, d);
    for (let i = 0; i < n; i++) {
      if (idx[i] >= 0) qs[i] = induced[idx[i]] * d.area[i] + q[i];
      else qs[i] = this.qBackup[i];
    }
  }

  /** Right-hand side b of the boundary element equation from the fields at the zero-induced-charge state. */
  private rhsFromField(e0: number, d: DielArrays, q: Float64Array, idx: Int32Array, m: number): Float64Array {
    const out = new Float64Array(m);
    const E = new Float64Array(3);
    const n = this.sys.state.n;
    for (let i = 0; i < n; i++) {
      const k = idx[i];
      if (k < 0) continue;
      if (d.ed[i] === 0) { out[k] = 0; continue; }
      this.field(i, E);
      const ndotE = e0 * (E[0] * d.mu[4 * i] + E[1] * d.mu[4 * i + 1] + E[2] * d.mu[4 * i + 2]) / d.epsilon[i];
      const sigmaF = q[i] / d.area[i];
      out[k] = (1 - d.em[i]) * sigmaF - d.ed[i] * ndotE / MY_4PI;
    }
    return out;
  }

  /** A * w: the field of the induced charges w * area (the ions and the real charges are zero here). */
  private applyOperator(e0: number, d: DielArrays, w: Float64Array, Aw: Float64Array): void {
    const n = this.sys.state.n, idx = this.idx, qs = d.qs;
    for (let i = 0; i < n; i++) qs[i] = idx[i] < 0 ? 0 : w[idx[i]] * d.area[i];
    this.evaluate();
    const buf = new Float64Array(this.m);
    const E = new Float64Array(3);
    for (let i = 0; i < n; i++) {
      const k = idx[i];
      if (k < 0) continue;
      this.field(i, E);
      const ndotE = e0 * (E[0] * d.mu[4 * i] + E[1] * d.mu[4 * i + 1] + E[2] * d.mu[4 * i + 2]) / d.epsilon[i];
      buf[k] = d.em[i] * w[k] + d.ed[i] * ndotE / MY_4PI;
    }
    Aw.set(buf);
  }

  /** r = b - A x without forming A x (Eq. 60 of Barros et al.): the real charges are restored for it. */
  private updateResidual(e0: number, d: DielArrays, q: Float64Array, w: Float64Array, r: Float64Array): void {
    const n = this.sys.state.n, idx = this.idx, qs = d.qs;
    for (let i = 0; i < n; i++) qs[i] = idx[i] < 0 ? this.qBackup[i] : w[idx[i]] * d.area[i] + q[i];
    this.evaluate();
    const buf = new Float64Array(this.m);
    const E = new Float64Array(3);
    for (let i = 0; i < n; i++) {
      const k = idx[i];
      if (k < 0) continue;
      if (d.ed[i] === 0) { buf[k] = 0; continue; }
      this.field(i, E);
      const ndotE = e0 * (E[0] * d.mu[4 * i] + E[1] * d.mu[4 * i + 1] + E[2] * d.mu[4 * i + 2]) / d.epsilon[i] / MY_4PI;
      const sigmaF = q[i] / d.area[i];
      buf[k] = (1 - d.em[i]) * sigmaF - d.em[i] * w[k] - d.ed[i] * ndotE;
    }
    r.set(buf);
  }

  /** GMRES(mr) of gmres_solve in LAMMPS (1-based index formulas kept as in the source). */
  private gmres(x: Float64Array, r: Float64Array, e0: number, d: DielArrays): void {
    const n = this.m;
    const q = this.sys.state.q;
    const mr = this.mr > n - 1 || this.mr <= 0 ? n - 1 : this.mr;
    const delta = 1.0e-3;
    const hi = (row: number, col: number) => row + col * (mr + 1);
    const v = new Float64Array(n * (mr + 1));
    const h = new Float64Array((mr + 1) * mr);
    const g = new Float64Array(mr + 1);
    const c = new Float64Array(Math.max(mr, 1));
    const sg = new Float64Array(Math.max(mr, 1));
    const y = new Float64Array(mr + 1);
    const col = (k: number) => v.subarray(k * n, (k + 1) * n);
    let rho = Math.sqrt(dot(r, r));
    const rhoTol = rho * this.tolRel;
    let itr = 1;
    let kCopy = 0;
    for (itr = 1; itr <= this.itrMax; itr++) {
      for (let i = 0; i < n; i++) v[i] = r[i] / rho;
      g[0] = rho;
      for (let i = 1; i <= mr; i++) g[i] = 0;
      h.fill(0);
      let k = 1;
      for (k = 1; k <= mr; k++) {
        kCopy = k;
        this.applyOperator(e0, d, col(k - 1), col(k));
        const vk = col(k);
        const av = Math.sqrt(dot(vk, vk));
        for (let j = 1; j <= k; j++) {
          h[hi(j - 1, k - 1)] = dot(vk, col(j - 1));
          const vj = col(j - 1), hv = h[hi(j - 1, k - 1)];
          for (let i = 0; i < n; i++) vk[i] = vk[i] - hv * vj[i];
        }
        h[hi(k, k - 1)] = Math.sqrt(dot(vk, vk));
        if (av + delta * h[hi(k, k - 1)] === av) {
          for (let j = 1; j <= k; j++) {
            const htmp = dot(vk, col(j - 1));
            h[hi(j - 1, k - 1)] = h[hi(j - 1, k - 1)] + htmp;
            const vj = col(j - 1);
            for (let i = 0; i < n; i++) vk[i] = vk[i] - htmp * vj[i];
          }
          h[hi(k, k - 1)] = Math.sqrt(dot(vk, vk));
        }
        if (h[hi(k, k - 1)] !== 0) for (let i = 0; i < n; i++) vk[i] = vk[i] / h[hi(k, k - 1)];
        if (k > 1) {
          for (let i = 1; i <= k + 1; i++) y[i - 1] = h[hi(i - 1, k - 1)];
          for (let j = 1; j <= k - 1; j++) multGivens(c[j - 1], sg[j - 1], j - 1, y);
          for (let i = 1; i <= k + 1; i++) h[hi(i - 1, k - 1)] = y[i - 1];
        }
        const mu = Math.sqrt(h[hi(k - 1, k - 1)] * h[hi(k - 1, k - 1)] + h[hi(k, k - 1)] * h[hi(k, k - 1)]);
        c[k - 1] = h[hi(k - 1, k - 1)] / mu;
        sg[k - 1] = -h[hi(k, k - 1)] / mu;
        h[hi(k - 1, k - 1)] = c[k - 1] * h[hi(k - 1, k - 1)] - sg[k - 1] * h[hi(k, k - 1)];
        h[hi(k, k - 1)] = 0;
        multGivens(c[k - 1], sg[k - 1], k - 1, g);
        rho = Math.abs(g[k]);
        if (rho <= rhoTol && rho <= this.tolAbs) break;
      }
      k = kCopy - 1;
      y[k] = g[k] / h[hi(k, k)];
      for (let i = k; i >= 1; i--) {
        y[i - 1] = g[i - 1];
        for (let j = i + 1; j <= k + 1; j++) y[i - 1] = y[i - 1] - h[hi(i - 1, j - 1)] * y[j - 1];
        y[i - 1] = y[i - 1] / h[hi(i - 1, i - 1)];
      }
      for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= k + 1; j++) x[i - 1] = x[i - 1] + v[(i - 1) + (j - 1) * n] * y[j - 1];
      }
      this.updateResidual(e0, d, q, x, r);
      rho = Math.sqrt(dot(r, r));
      // Barros et al. suggested the condition: norm(r) < EPSILON norm(b)
      if (rho < this.tolRel * this.normb) break;
      if (rho <= rhoTol && rho <= this.tolAbs) break;
    }
    this.iterations = itr;
    this.rho = rho;
  }

  modify(key: string, values: string[]): number {
    const int = (w: string | undefined): number => {
      const x = Number(w);
      if (w === undefined || !Number.isInteger(x)) throw new StyleError(`Illegal fix_modify ${key} value '${w ?? ''}'`);
      return x;
    };
    if (key === 'itr_max') { this.itrMax = int(values[0]); return 1; }
    if (key === 'mr') { this.mr = int(values[0]); return 1; }
    if (key === 'kspace') {
      if (values[0] !== 'yes' && values[0] !== 'no') throw new StyleError(`Illegal fix_modify kspace value '${values[0] ?? ''}'`);
      this.kspaceOn = values[0] === 'yes';
      return 1;
    }
    if (key === 'rand') throw new StyleError(`fix_modify rand is not supported by fix ${this.id} (${this.style}) in the browser engine`);
    if (key === 'dielectrics') {
      if (values.length < 5) throw new StyleError(`Illegal fix_modify dielectrics: needs ediff emean epsilon area q (NULL to keep)`);
      setDielectricParams(this.sys, (i) => this.inGroup(i), values.slice(0, 5), this.id, this.style, false);
      return 5;
    }
    return super.modify(key, values);
  }

  computeVector(i: number): number {
    if (i === 0) return this.iterations;
    if (i === 1) return this.rho;
    return 0;
  }
}

/** Dot product of two vectors (LAMMPS vec_dot). */
const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let t = 0;
  for (let i = 0; i < a.length; i++) t += a[i] * b[i];
  return t;
};

/** Givens rotation of g[k], g[k+1] (LAMMPS mult_givens). */
const multGivens = (c: number, s: number, k: number, g: Float64Array): void => {
  const g1 = c * g[k] - s * g[k + 1];
  const g2 = s * g[k] + c * g[k + 1];
  g[k] = g1;
  g[k + 1] = g2;
};

/**
 * fix_modify dielectrics ediff emean epsiloni areai qi: set ed, em (and epsilon, area, charge) of the group atoms; NULL
 * keeps the value. The ICC variant also rescales the scaled charge (set_dielectric_params in fix_polarize_bem_icc.cpp).
 */
export const setDielectricParams = (sys: System, inGroup: (i: number) => boolean, vals: string[], id: string, style: string, rescale: boolean): void => {
  const num = (w: string): number => {
    const x = Number(w);
    if (!Number.isFinite(x)) throw new StyleError(`Illegal fix_modify dielectrics value '${w}' for fix ${id} (${style})`);
    return x;
  };
  const s = sys.state;
  const ed = num(vals[0]), em = num(vals[1]);
  const epsI = vals[2] !== 'NULL' ? num(vals[2]) : -1;
  const areaI = vals[3] !== 'NULL' ? num(vals[3]) : -1;
  const setQ = vals[4] !== 'NULL';
  const qv = setQ ? num(vals[4]) : 0;
  const d = dielArrays(s, `fix ${id}`);
  for (let i = 0; i < s.n; i++) {
    if (!inGroup(i)) continue;
    d.ed[i] = ed;
    d.em[i] = em;
    if (areaI > 0) d.area[i] = areaI;
    if (epsI > 0) d.epsilon[i] = epsI;
    if (setQ) s.q[i] = qv;
    if (rescale) d.qs[i] = s.q[i] / d.epsilon[i];
  }
};
