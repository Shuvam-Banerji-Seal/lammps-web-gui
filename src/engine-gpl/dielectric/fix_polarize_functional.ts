/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/fix_polarize_functional.cpp and fix_polarize_functional.h at commit b332f28 (Trung Nguyen and Monica Olvera de la Cruz, Northwestern University, based on the original implementation by Honghao Li, Northwestern).
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix polarize/functional N [tol]: induced surface charges of the interface patches (the group) from the variational
 * (direct optimization) equation of Jadhao, Solis, Olvera de la Cruz, J. Chem. Phys. 138, 054119 (2013): the matrix
 * (Rww + Rww^T) is built from the Green's function erfc(g r)/r over the neighbor pairs of the full list, and solved by
 * conjugate gradients (tolerance from the command, default 1e-6). The Green's function parameter g is the kspace style's
 * g_ewald, or 0.01 without kspace. The fix does not evaluate forces: the energies and forces of a run's setup are those
 * of the force evaluation before the charges are set (Verlet::setup), as in LAMMPS.
 */
import { Fix } from '../../engine/fix/fix';
import { parseNum } from '../../engine/force/util';
import { StyleError } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import type { System } from '../../engine/system';
import { dielArrays, dielectricPairField } from './common';
import { setDielectricParams } from './fix_polarize_bem_gmres';
import { exp as libmExp } from '../libm/exp';
import { erfc as libmErfc } from '../libm/erf';

const MY_PIS = Math.sqrt(Math.PI);
const MY_4PI = 4 * Math.PI;
/** Default CG tolerance on the squared residual (fix_polarize_functional.cpp EPSILON). */
const EPSILON = 1.0e-6;

export class FixPolarizeFunctional extends Fix {
  readonly style = 'polarize/functional';
  private readonly every: number;
  private readonly tolerance: number;
  private kspaceOn = true;
  /** Induced-charge index of each owned interface atom (-1 outside the group), and the ion index of each ion (-1 in the group). */
  private idx = new Int32Array(0);
  private ion = new Int32Array(0);
  private m = 0;
  private numIons = 0;
  private gEwald = 0.01;
  /** Rww + Rww^T (the CG matrix), built with Rww. */
  private A = new Float64Array(0);
  /** G1ww, ndotGww and G2ww of the last setup (the right-hand side of the solve reuses them). */
  private gMatrices: { G1: Float64Array; ND: Float64Array; G2: Float64Array } | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError(`usage: fix ${id} group-ID polarize/functional N [tol]`);
    const every = Number(args[0]);
    if (!Number.isInteger(every) || every < 0) throw new StyleError(`Illegal fix ${id} polarize/functional command: N must be an integer >= 0`);
    this.every = every;
    this.tolerance = args.length >= 2 ? parseNum(args[1], 'tolerance') : EPSILON;
  }

  /** Atom-ID numbering of the interface (group) and of the ions (the rest), in ID order as LAMMPS' tag2mat. */
  private mapping(): void {
    const s = this.sys.state;
    const inter: number[] = [], ions: number[] = [];
    for (let i = 0; i < s.n; i++) (this.inGroup(i) ? inter : ions).push(s.id[i]);
    inter.sort((a, b) => a - b);
    ions.sort((a, b) => a - b);
    const pi = new Map<number, number>(), pn = new Map<number, number>();
    inter.forEach((id, k) => pi.set(id, k));
    ions.forEach((id, k) => pn.set(id, k));
    this.idx = new Int32Array(s.n);
    this.ion = new Int32Array(s.n);
    for (let i = 0; i < s.n; i++) {
      if (this.inGroup(i)) { this.idx[i] = pi.get(s.id[i])!; this.ion[i] = -1; }
      else { this.idx[i] = -1; this.ion[i] = pn.get(s.id[i])!; }
    }
    this.m = inter.length;
    this.numIons = ions.length;
  }

  private pairOk(): void {
    if (!dielectricPairField(this.sys.ff.pair)) throw new StyleError(`Pair style not compatible with fix ${this.id} (${this.style})`);
  }

  setup(): void {
    this.pairOk();
    const k = this.sys.ff.kspace;
    if (k && this.kspaceOn && k.name !== 'pppm/dielectric') throw new StyleError(`Kspace style not compatible with fix ${this.id} (${this.style})`);
    this.gEwald = k && this.kspaceOn ? k.gEwald : 0.01;
    this.mapping();
    // calculate_Rww_cutoff: the interface matrix from the neighbor list of the run (before the forces of setup)
    this.buildRww();
    // setup() of LAMMPS: the induced charges are set after the run's force evaluation (no force evaluation here)
    this.updateInduced();
  }

  preForce(): void {
    if (this.every === 0) return;
    if (this.sys.state.step % this.every) return;
    this.updateInduced();
  }

  /** Real Green's function erfc(g r) / r. */
  private greensReal(r: number): number { return libmErfc(this.gEwald * r) / r; }

  /** Gradient of the real Green's function as a vector (grad_greens_real_factor times d). */
  private gradGreensReal(dx: number, dy: number, dz: number, out: Float64Array): void {
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const alpha = this.gEwald * r;
    const factor = libmErfc(alpha) + 2.0 * alpha / MY_PIS * libmExp(-(alpha * alpha));
    const r3 = r * r * r;
    const real = factor * (-1.0 / r3);
    out[0] = real * dx; out[1] = real * dy; out[2] = real * dz;
  }

  /** The self term of the Green's function on one patch: 2 sqrt(pi)/sqrt(area) - 2 g / sqrt(pi). */
  private greensSelf(area: number): number {
    return 2.0 * MY_PIS / Math.sqrt(area) - 2.0 * this.gEwald / MY_PIS;
  }

  /** calculate_Rww_cutoff: G1ww, ndotGww, G2ww, G3ww and Rww over the group's pairs of the full list. */
  private buildRww(): void {
    const s = this.sys.state;
    const nb = this.sys.nb;
    const list = nb.full;
    if (!list) throw new StyleError(`fix ${this.id} (${this.style}) needs a full neighbor list`);
    const d = dielArrays(s, `fix ${this.id}`);
    const { area, ed, em, curvature: curv, mu } = d;
    const N = this.m;
    const owner = nb.owner, xa = nb.xall, idx = this.idx;
    const G1 = new Float64Array(N * N), ND = new Float64Array(N * N);
    const grad = new Float64Array(3);
    const inG = (j: number) => idx[owner[j]] >= 0;
    // pass 1: the Green's function and the normal derivative over the pairs (and the self terms)
    for (let ii = 0; ii < list.inum; ii++) {
      const i = ii;
      if (idx[i] < 0) continue;
      const mi = idx[i];
      for (let kk = list.firstneigh[i], k1 = kk + list.numneigh[i]; kk < k1; kk++) {
        const k = list.neighbors[kk] & NEIGHMASK;
        if (!inG(k)) continue;
        const mk = idx[owner[k]];
        const dx = xa[3 * i] - xa[3 * k], dy = xa[3 * i + 1] - xa[3 * k + 1], dz = xa[3 * i + 2] - xa[3 * k + 2];
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        G1[mi * N + mk] = this.greensReal(r);
        this.gradGreensReal(dx, dy, dz, grad);
        ND[mi * N + mk] = (mu[4 * i] * grad[0] + mu[4 * i + 1] * grad[1] + mu[4 * i + 2] * grad[2]) / MY_4PI;
      }
      G1[mi * N + mi] = this.greensSelf(area[i]);
      ND[mi * N + mi] = curv[i] * MY_PIS / Math.sqrt(area[i]) / MY_4PI;
    }
    // The passes below only touch the entries of pairs in the group: for each atom i the group neighbours l are
    // gathered once, in list order, with their owner's area and ed (the same factors, in the same order, as before).
    let maxNb = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > maxNb) maxNb = list.numneigh[i];
    const gl = new Int32Array(maxNb), gA = new Float64Array(maxNb), gE = new Float64Array(maxNb), gO = new Int32Array(maxNb);
    const gather = (i: number): number => {
      let cnt = 0;
      for (let kk = list.firstneigh[i], k1 = kk + list.numneigh[i]; kk < k1; kk++) {
        const l = list.neighbors[kk] & NEIGHMASK;
        if (!inG(l)) continue;
        const ol = owner[l];
        gl[cnt] = idx[ol]; gA[cnt] = area[ol]; gE[cnt] = ed[ol]; gO[cnt] = ol;
        cnt++;
      }
      return cnt;
    };
    // transposed copy of ND, so that the sums over l read a contiguous row (ND[ml * N + mk] = NDt[mk * N + ml])
    const NDt = new Float64Array(N * N);
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) NDt[c * N + r] = ND[r * N + c];
    // pass 2: G2ww from the neighbors of each patch
    const G2 = new Float64Array(N * N);
    for (let i = 0; i < list.inum; i++) {
      if (idx[i] < 0) continue;
      const mi = idx[i], miN = mi * N;
      const cnt = gather(i);
      for (let t = 0; t < cnt; t++) {
        const mk = gl[t], mkN = mk * N;
        let temp = 0;
        for (let u = 0; u < cnt; u++) {
          const ml = gl[u];
          temp += G1[miN + ml] * NDt[mkN + ml] * gA[u] * gE[u];
        }
        G2[miN + mk] = temp;
      }
      let temp = 0;
      for (let t = 0; t < cnt; t++) {
        const mk = gl[t];
        temp += G1[miN + mk] * NDt[miN + mk] * gA[t] * gE[t];
      }
      G2[miN + mi] = temp;
    }
    // pass 3: Rww with the G2 and G3 corrections (the diagonal keeps only the G1 and G2 terms, as in LAMMPS)
    const G2t = new Float64Array(N * N);
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) G2t[c * N + r] = G2[r * N + c];
    const Rww = new Float64Array(N * N);
    for (let i = 0; i < list.inum; i++) {
      if (idx[i] < 0) continue;
      const mi = idx[i], miN = mi * N;
      const cnt = gather(i);
      for (let t = 0; t < cnt; t++) {
        const mk = gl[t], mkN = mk * N;
        const emk = em[gO[t]];
        const a1 = em[i] * (emk - 1.0);
        const a2 = 1.0 - em[i] - emk;
        Rww[miN + mk] = a1 * G1[miN + mk] + a2 * G2[miN + mk];
        let temp = 0;
        for (let u = 0; u < cnt; u++) {
          const ml = gl[u];
          temp += NDt[miN + ml] * G2t[mkN + ml] * gA[u] * gE[u];
        }
        Rww[miN + mk] += temp;
      }
      // the diagonal keeps only the G1 and G2 terms (the G3 term of the diagonal is not added, as in LAMMPS)
      const a1 = em[i] * (em[i] - 1.0);
      const a2 = 1.0 - em[i] - em[i];
      Rww[miN + mi] = a1 * G1[miN + mi] + a2 * G2[miN + mi];
    }
    // the CG matrix Rww + Rww^T is fixed for the setup: built once here
    const A = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) A[i * N + j] = Rww[i * N + j] + Rww[j * N + i];
    this.A = A;
    this.gMatrices = { G1, ND, G2 };
  }

  /** update_induced_charges: the right-hand side from the ions, the CG solve, and the interface charges. */
  private updateInduced(): void {
    const s = this.sys.state;
    const nb = this.sys.nb;
    const list = nb.full;
    if (!list || !this.gMatrices) throw new StyleError(`fix ${this.id} (${this.style}): no interface matrix (setup has not run)`);
    const d = dielArrays(s, `fix ${this.id}`);
    const { qs, epsilon: eps, area, ed, em, mu } = d;
    const q = s.q;
    const n = s.n;
    const N = this.m;
    const idx = this.idx, ion = this.ion;
    const owner = nb.owner, xa = nb.xall;
    // charge_rescaled(SCALED2REAL): the ions carry their real charges
    for (let i = 0; i < n; i++) if (idx[i] < 0) qs[i] = q[i];
    // calculate_qiRqw_cutoff: G1qw_real (ion to patch), then the right-hand side
    const NI = this.numIons;
    const G1qw = new Float64Array(NI * N);
    for (let i = 0; i < list.inum; i++) {
      if (idx[i] >= 0) continue;
      const mi = ion[i];
      for (let kk = list.firstneigh[i], k1 = kk + list.numneigh[i]; kk < k1; kk++) {
        const k = list.neighbors[kk] & NEIGHMASK;
        if (idx[owner[k]] < 0) continue;
        const mk = idx[owner[k]];
        const dx = xa[3 * i] - xa[3 * k], dy = xa[3 * i + 1] - xa[3 * k + 1], dz = xa[3 * i + 2] - xa[3 * k + 2];
        G1qw[mi * N + mk] = this.greensReal(Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
    const { G1, ND, G2 } = this.gMatrices;
    const sum1G1qwEps = new Float64Array(N), sum2NdotGwqEps = new Float64Array(N);
    const tmp = new Float64Array(3), gv = new Float64Array(3);
    for (let k = 0; k < list.inum; k++) {
      if (idx[k] < 0) continue;
      const mk = idx[k];
      tmp[0] = tmp[1] = tmp[2] = 0;
      let tempSum1 = 0;
      for (let ii = list.firstneigh[k], k1 = ii + list.numneigh[k]; ii < k1; ii++) {
        const i = list.neighbors[ii] & NEIGHMASK;
        const oi = owner[i];
        if (idx[oi] >= 0) continue;
        const mi = ion[oi];
        const dx = xa[3 * i] - xa[3 * k], dy = xa[3 * i + 1] - xa[3 * k + 1], dz = xa[3 * i + 2] - xa[3 * k + 2];
        this.gradGreensReal(dx, dy, dz, gv);
        // scale3(-1.0, gradG1wq)
        gv[0] = -gv[0]; gv[1] = -gv[1]; gv[2] = -gv[2];
        tmp[0] += gv[0] * (q[oi] / eps[oi]);
        tmp[1] += gv[1] * (q[oi] / eps[oi]);
        tmp[2] += gv[2] * (q[oi] / eps[oi]);
        tempSum1 += G1qw[mi * N + mk] * q[oi] / eps[oi];
      }
      sum1G1qwEps[mk] = tempSum1;
      sum2NdotGwqEps[mk] = mu[4 * k] * tmp[0] + mu[4 * k + 1] * tmp[1] + mu[4 * k + 2] * tmp[2];
    }
    const sum2G2wq = new Float64Array(N), sum1G2qw = new Float64Array(N);
    for (let k = 0; k < list.inum; k++) {
      if (idx[k] < 0) continue;
      const mk = idx[k];
      let tempwq = 0, temp = 0;
      for (let ii = list.firstneigh[k], k1 = ii + list.numneigh[k]; ii < k1; ii++) {
        const i = list.neighbors[ii] & NEIGHMASK;
        const oi = owner[i];
        if (idx[oi] < 0) continue;
        const mi = idx[oi];
        tempwq += G1[mk * N + mi] * sum2NdotGwqEps[mi] * area[oi] * ed[oi];
        temp += sum1G1qwEps[mi] * ND[mi * N + mk] * area[oi] * ed[oi];
      }
      tempwq += G1[mk * N + mk] * sum2NdotGwqEps[mk] * area[k] * ed[k];
      temp += sum1G1qwEps[mk] * ND[mk * N + mk] * area[k] * ed[k];
      sum2G2wq[mk] = tempwq;
      sum1G2qw[mk] = temp;
    }
    const qiRqw = new Float64Array(N);
    for (let k = 0; k < list.inum; k++) {
      if (idx[k] < 0) continue;
      const mk = idx[k];
      let sum1G3qw = 0, qiTemp1 = 0;
      for (let ii = list.firstneigh[k], k1 = ii + list.numneigh[k]; ii < k1; ii++) {
        const i = list.neighbors[ii] & NEIGHMASK;
        const oi = owner[i];
        if (idx[oi] >= 0) {
          const mi = idx[oi];
          sum1G3qw += sum2NdotGwqEps[mi] * G2[mi * N + mk] * area[oi] * ed[oi];
        } else {
          const mi = ion[oi];
          qiTemp1 += q[oi] * (1.0 - em[k] / eps[oi]) * G1qw[mi * N + mk];
        }
      }
      sum1G3qw += sum2NdotGwqEps[mk] * G2[mk * N + mk] * area[k] * ed[k];
      const qiTemp2 = (1.0 - 2.0 * em[k]) * sum2G2wq[mk] + sum1G2qw[mk] + 2.0 * sum1G3qw;
      qiRqw[mk] = qiTemp1 + qiTemp2;
    }
    // CG for (Rww + Rww^T) w = qiRqw, from w = 0
    const w = new Float64Array(N);
    this.cgSolve(this.A, qiRqw, w, N);
    // the patches get -w / (4 pi) as their scaled charge; the ions return to their scaled charges
    for (let i = 0; i < n; i++) {
      if (idx[i] >= 0) qs[i] = -w[idx[i]] / MY_4PI;
      else qs[i] = q[i] / eps[i];
    }
  }

  /** cg_solver of LAMMPS: conjugate gradients on the dense matrix A, stopping when the squared residual is below the tolerance. */
  private cgSolve(A: Float64Array, b: Float64Array, x: Float64Array, N: number): void {
    const matvec = (v: Float64Array, out: Float64Array) => {
      for (let k = 0; k < N; k++) {
        let t = 0;
        for (let l = 0; l < N; l++) t += A[k * N + l] * v[l];
        out[k] = t;
      }
    };
    const dot = (a: Float64Array, c: Float64Array) => {
      let t = 0;
      for (let i = 0; i < N; i++) t += a[i] * c[i];
      return t;
    };
    const r = new Float64Array(N), p = new Float64Array(N), Ap = new Float64Array(N);
    matvec(x, p);
    for (let i = 0; i < N; i++) {
      r[i] = b[i] - p[i];
      p[i] = r[i];
    }
    let rsq = dot(r, r);
    for (let k = 0; k < N; k++) {
      matvec(p, Ap);
      const pAp = dot(p, Ap);
      const alpha = rsq / pAp;
      for (let i = 0; i < N; i++) {
        x[i] = x[i] + alpha * p[i];
        r[i] = r[i] - alpha * Ap[i];
      }
      const rsqNew = dot(r, r);
      if (rsqNew < this.tolerance) break;
      const beta = rsqNew / rsq;
      for (let i = 0; i < N; i++) p[i] = r[i] + beta * p[i];
      rsq = rsqNew;
    }
  }

  modify(key: string, values: string[]): number {
    if (key === 'kspace') {
      if (values[0] !== 'yes' && values[0] !== 'no') throw new StyleError(`Illegal fix_modify kspace value '${values[0] ?? ''}'`);
      this.kspaceOn = values[0] === 'yes';
      return 1;
    }
    if (key === 'dielectrics') {
      if (values.length < 5) throw new StyleError('Illegal fix_modify dielectrics: needs ediff emean epsilon area q (NULL to keep)');
      setDielectricParams(this.sys, (i) => this.inGroup(i), values.slice(0, 5), this.id, this.style, false);
      return 5;
    }
    return super.modify(key, values);
  }
}
