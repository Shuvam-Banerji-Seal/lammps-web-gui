/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pppm_disp.cpp and pppm_disp.h at b332f28 (PPPMDisp::settings, modify (kspace.cpp
 * modify_params keywords mesh/disp, order/disp, gewald/disp, mix/disp, force/disp/real, force/disp/kspace, splittol,
 * disp/auto), init, setup (the box-dependent part, rebuilt when the box changes), compute, the energy and virial sums,
 * poisson_ik and poisson_2s_ik (orthogonal branches), make_rho_g, make_rho_a, fieldforce_g_ik, fieldforce_a_ik,
 * calc_csum, init_coeffs (geometric and arithmetic B), KSpace::two_charge (src/kspace.cpp), the Coulomb G-ewald and
 * mesh choice: set_grid_global, set_grid_local (factorable), f, derivf, adjust_gewald, final_accuracy (compute_qopt_ik
 * is in grid.ts), and the G-ewald and mesh choice for dispersion: set_grid_global_6, set_init_g6, set_n_pppm_6,
 * adjust_gewald_6, f_6). The pair side also takes buck/long/coul/long (pair_buck_long_coul_long.cpp: extract("B") is
 * buck_c, so the geometric 1/r^6 coefficients are sqrt(|C_ii|) as native).
 * Contributing authors of pppm_disp.cpp: Rolf Isele-Holder (Aachen University), Paul Crozier (SNL). Copyright (2003)
 * Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too.
 *
 * Scope of this port (everything else is refused by name, as StyleError):
 *  - orthogonal, fully periodic boxes; no slab correction; kspace_modify diff ik only;
 *  - per-atom energy and virial (compute pe/atom, stress/atom): poisson_ik's per-atom part, poisson_peratom,
 *    poisson_2s_ik's per-atom part, poisson_2s_peratom, fieldforce_c_peratom, fieldforce_g_peratom,
 *    fieldforce_a_peratom and the per-atom self terms of compute() (not for pppm/disp/tip4p or pppm/disp/dielectric);
 *  - dispersion with geometric mixing (one grid: B_i = sqrt(|4 eps_ii sigma_ii^6|)) and with arithmetic mixing
 *    (seven grids, B_ik = sqrt(eps_i) sigma_i^k sqrt(C(6,k)) / 4, the pair energies of the pairs (k, 6-k)); the
 *    arithmetic path needs at least two distinct eigen-directions of the C6 matrix (nsplit > 1), else geometric as in
 *    native init_coeffs; buck/long with arithmetic mixing is refused as native does (no epsilon/sigma); mix/disp none
 *    (function[3]) is refused;
 *  - Coulomb: user or automatic G-ewald and mesh (kspace_modify gewald and mesh are each optional; the native estimate,
 *    Newton adjustment and factorable boost); the automatic choice is refused for pppm/disp/dielectric (its q2 needs the
 *    dielectric scale, not ported here);
 *  - the box may change after init (fix npt, fix deform): the Green's functions and virial coefficients are rebuilt for
 *    the current box at each force evaluation (native setup() is called at the same points); the grids and G-ewald
 *    stay as chosen at init, as native;
 *  - dispersion G-ewald and mesh: user values (kspace_modify gewald/disp, mesh/disp) or the native automatic estimate
 *    (native refuses it unless kspace_modify disp/auto yes or a dispersion parameter is set; so do we).
 * Energies: elong = native energy (energy_1 * qqrd2e + energy_6); the virial is the native virial array.
 * The forward/backward transforms of two densities (poisson_2s_ik packs them into one complex transform) are done one
 * grid at a time here, which is the same algebra (the packing is exact for real densities).
 */
import { StyleError, type KSpaceCompute, type StyleContext, type Pair } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { parseNum } from '../../engine/force/util';
import { KSpaceBase } from '../../engine/force/kspace/ewald';
import { PairLJLongCoulLong } from '../../engine/force/pair/lj_long';
import { PairBuckLongCoulLong } from '../csdiel/buck_long';
import { ANGSTROM, QELECTRON } from '../../engine/force/kspace/ewald';
import { hasChargeStyle } from '../../engine/atoms';
import { fft3d } from '../../engine/force/fft';
import {
  OFFSET, MAXORDER, SMALL, LARGE, factorable, makeStencil, rho1d, meshGeom, greenCoulomb, greenDispersion,
  virialCoulomb, virialDispersion, virialCross, ljRspaceError, qoptDispersionIk, qoptCoulombIk, symEigenvalues, type MeshGeom,
  type Stencil,
} from './grid';
import { pow as libmPow } from '../libm/pow';
import { exp as libmExp } from '../libm/exp';
import { log as libmLog } from '../libm/log';
const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)

/** Grid index of each atom (particle_map): base[3i + d] = floor((x_d - lo_d) N_d / L_d + shift) - OFFSET. */
const mapAtoms = (x: Float64Array, n: number, lo: [number, number, number], mg: MeshGeom, st: Stencil): Int32Array => {
  const base = new Int32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < 3; d++) {
      const delx = mg.N[d] / mg.prd[d];
      base[3 * i + d] = Math.floor((x[3 * i + d] - lo[d]) * delx + st.shift) - OFFSET;
    }
  }
  return base;
};

/** The P-point weights of atom i along each dimension (dx of make_rho_c / fieldforce_c_ik). */
const atomWeights = (x: Float64Array, i: number, lo: [number, number, number], mg: MeshGeom, st: Stencil,
  base: Int32Array, r0: Float64Array, r1: Float64Array, r2: Float64Array): void => {
  const d = [0, 1, 2].map((dd) => {
    const delinv = mg.N[dd] / mg.prd[dd];
    return base[3 * i + dd] + st.shiftone - (x[3 * i + dd] - lo[dd]) * delinv;
  });
  rho1d(st, d[0], r0);
  rho1d(st, d[1], r1);
  rho1d(st, d[2], r2);
};

const wrap = (a: number, m: number): number => ((a % m) + m) % m;

const prdOf = (geom: Geometry): [number, number, number] => [geom.lx, geom.ly, geom.lz];

/** make_rho_c / make_rho_g / make_rho_a: the mesh density from per-atom weights w_i (already times delvolinv). */
const spread = (x: Float64Array, n: number, lo: [number, number, number], mg: MeshGeom, st: Stencil,
  base: Int32Array, w: Float64Array): Float64Array => {
  const [nx, ny, nz] = mg.N;
  const P = st.order;
  const rho = new Float64Array(2 * nx * ny * nz);
  const r0 = new Float64Array(P), r1 = new Float64Array(P), r2 = new Float64Array(P);
  for (let i = 0; i < n; i++) {
    if (w[i] === 0) continue;
    atomWeights(x, i, lo, mg, st, base, r0, r1, r2);
    const bx = base[3 * i], by = base[3 * i + 1], bz = base[3 * i + 2];
    const z0 = w[i];
    for (let nn = 0; nn < P; nn++) {
      const mz = wrap(bz + st.nlower + nn, nz);
      const y0 = z0 * r2[nn];
      for (let mm = 0; mm < P; mm++) {
        const my = wrap(by + st.nlower + mm, ny);
        const x0 = y0 * r1[mm];
        for (let ll = 0; ll < P; ll++) {
          const mx = wrap(bx + st.nlower + ll, nx);
          rho[2 * ((mz * ny + my) * nx + mx)] += x0 * r0[ll];
        }
      }
    }
  }
  return rho;
};

/** fieldforce: the interpolated field -sum W v at atom i, for the three real field grids. */
const interpField = (x: Float64Array, i: number, lo: [number, number, number], mg: MeshGeom, st: Stencil,
  base: Int32Array, grids: [Float64Array, Float64Array, Float64Array]): [number, number, number] => {
  const [nx, ny, nz] = mg.N;
  const P = st.order;
  const r0 = new Float64Array(P), r1 = new Float64Array(P), r2 = new Float64Array(P);
  atomWeights(x, i, lo, mg, st, base, r0, r1, r2);
  const bx = base[3 * i], by = base[3 * i + 1], bz = base[3 * i + 2];
  let ex = 0, ey = 0, ez = 0;
  for (let nn = 0; nn < P; nn++) {
    const mz = wrap(bz + st.nlower + nn, nz);
    const z0 = r2[nn];
    for (let mm = 0; mm < P; mm++) {
      const my = wrap(by + st.nlower + mm, ny);
      const y0 = z0 * r1[mm];
      for (let ll = 0; ll < P; ll++) {
        const mx = wrap(bx + st.nlower + ll, nx);
        const x0 = y0 * r0[ll];
        const idx = (mz * ny + my) * nx + mx;
        ex -= x0 * grids[0][idx];
        ey -= x0 * grids[1][idx];
        ez -= x0 * grids[2][idx];
      }
    }
  }
  return [ex, ey, ez];
};

/**
 * The energy and virial of one pair of forward-transformed densities (poisson_ik: a = b; poisson_2s_ik: a != b):
 * sum over the mesh of factor s2 G Re(F_a conj F_b) and the same weighted by vg (the virial coefficients).
 * s2 = 1/N^2 (scaleinv squared). For a = b and factor 1 this is the energy of poisson_ik.
 */
const pairSum = (Fa: Float64Array, Fb: Float64Array, G: Float64Array, vg: Float64Array, s2: number, factor: number,
  egy: { v: number }, vir: Float64Array): void => {
  const ng = G.length;
  for (let i = 0; i < ng; i++) {
    const eng = factor * s2 * G[i] * (Fa[2 * i] * Fb[2 * i] + Fa[2 * i + 1] * Fb[2 * i + 1]);
    egy.v += eng;
    for (let c = 0; c < 6; c++) vir[c] += eng * vg[6 * i + c];
  }
};

/**
 * The three field grids of one density (poisson_ik, orthogonal): IFFT of i a G F/N, i b G F/N and i kz G F/N with
 * a = (kx - kx2)/2, b = (ky - ky2)/2 (the Nyquist modes of x and y drop out, z keeps kz as the C++ does). F is the
 * unscaled forward transform (not modified).
 */
const fieldsIk = (F: Float64Array, mg: MeshGeom, G: Float64Array): [Float64Array, Float64Array, Float64Array] => {
  const [nx, ny, nz] = mg.N;
  const ng = nx * ny * nz;
  const scaleinv = 1.0 / ng;
  const w1 = new Float64Array(2 * ng), w2 = new Float64Array(2 * ng);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const idx = (k * ny + j) * nx + i;
    const a = 0.5 * (mg.fk[0][i] - mg.fk2[0][i]);
    const b = 0.5 * (mg.fk[1][j] - mg.fk2[1][j]);
    const f = scaleinv * G[idx];
    const re = F[2 * idx] * f, im = F[2 * idx + 1] * f;
    w1[2 * idx] = -a * im + b * re;
    w1[2 * idx + 1] = a * re + b * im;
    w2[2 * idx] = -mg.fk[2][k] * im;
    w2[2 * idx + 1] = mg.fk[2][k] * re;
  }
  fft3d(w1, nx, ny, nz, 1);
  fft3d(w2, nx, ny, nz, 1);
  const vx = new Float64Array(ng), vy = new Float64Array(ng), vz = new Float64Array(ng);
  for (let idx = 0; idx < ng; idx++) {
    vx[idx] = w1[2 * idx];
    vy[idx] = -w1[2 * idx + 1];
    vz[idx] = w2[2 * idx];
  }
  return [vx, vy, vz];
};

/** The per-atom bricks of one density: the potential u (per-atom energy) and the six virial bricks, or null when not asked. */
interface PaBricks {
  u: Float64Array | null;
  v: Float64Array[] | null;
}

/** IFFT of one packed complex mesh: real part to `re`, minus the imaginary part to `im` (the brick pairs of the C++). */
const ifftPair = (w: Float64Array, mg: MeshGeom, re: Float64Array, im: Float64Array): void => {
  fft3d(w, mg.N[0], mg.N[1], mg.N[2], 1);
  for (let idx = 0; idx < re.length; idx++) {
    re[idx] = w[2 * idx];
    im[idx] = -w[2 * idx + 1];
  }
};

/**
 * poisson_ik's per-atom part (orthogonal branch, eflag_atom) and poisson_peratom for one density F (the unscaled
 * forward transform): W = F G / N; u = IFFT(W); the virial bricks v0..v5 from three packed transforms,
 * W (vg0 - i vg1), W (vg2 - i vg2_xy) and W (vg2_xz - i vg2_yz), as the C++ packs them.
 */
const peratomBricks = (F: Float64Array, mg: MeshGeom, G: Float64Array, vg: Float64Array, vg2: Float64Array,
  wantE: boolean, wantV: boolean): PaBricks => {
  const ng = mg.N[0] * mg.N[1] * mg.N[2];
  const scaleinv = 1.0 / ng;
  const W = new Float64Array(2 * ng);
  for (let i = 0; i < ng; i++) {
    const f = scaleinv * G[i];
    W[2 * i] = F[2 * i] * f;
    W[2 * i + 1] = F[2 * i + 1] * f;
  }
  const out: PaBricks = { u: null, v: null };
  const w2 = new Float64Array(2 * ng);
  if (wantE) {
    w2.set(W);
    fft3d(w2, mg.N[0], mg.N[1], mg.N[2], 1);
    out.u = new Float64Array(ng);
    for (let idx = 0; idx < ng; idx++) out.u[idx] = w2[2 * idx];
  }
  if (wantV) {
    const v = [0, 1, 2, 3, 4, 5].map(() => new Float64Array(ng));
    // the (a, b) coefficient pairs of the three transforms: (vg0, vg1), (vg2, vg2[0]), (vg2[1], vg2[2])
    const coef = (n: number, pass: number): [number, number] =>
      pass === 0 ? [vg[6 * n], vg[6 * n + 1]] : pass === 1 ? [vg[6 * n + 2], vg2[3 * n]] : [vg2[3 * n + 1], vg2[3 * n + 2]];
    for (let pass = 0; pass < 3; pass++) {
      for (let n = 0; n < ng; n++) {
        const [a, b] = coef(n, pass);
        w2[2 * n] = W[2 * n] * a + W[2 * n + 1] * b;
        w2[2 * n + 1] = W[2 * n + 1] * a - W[2 * n] * b;
      }
      ifftPair(w2, mg, v[2 * pass], v[2 * pass + 1]);
    }
    out.v = v;
  }
  return out;
};

/**
 * poisson_2s_ik's per-atom part and poisson_2s_peratom (orthogonal) for two densities F1 and F2: the packed
 * W = (F1 - i F2) G / N; u1, u2 from IFFT(W) and v_c1, v_c2 from IFFT(W vg_c), the real part to the first density
 * and minus the imaginary part to the second, as the C++ packs them. As poisson_2s_peratom does, the diagonal terms
 * v0..v2 take vg_6[0..2] and the off-diagonal v3..v5 take vg2_6[0..2] (which differ from vg_6[3..5] on the Nyquist
 * planes of an even mesh).
 */
const peratomBricks2s = (F1: Float64Array, F2: Float64Array, mg: MeshGeom, G: Float64Array, vg: Float64Array,
  vg2: Float64Array, wantE: boolean, wantV: boolean): [PaBricks, PaBricks] => {
  const ng = mg.N[0] * mg.N[1] * mg.N[2];
  const scaleinv = 1.0 / ng;
  const W = new Float64Array(2 * ng);
  for (let i = 0; i < ng; i++) {
    const f = scaleinv * G[i];
    // F1 + (-i F2), the sum of the two forward transforms of the C++ (work1_6 += work2_6), then times G / N
    W[2 * i] = (F1[2 * i] + F2[2 * i + 1]) * f;
    W[2 * i + 1] = (F1[2 * i + 1] - F2[2 * i]) * f;
  }
  const b1: PaBricks = { u: null, v: null }, b2: PaBricks = { u: null, v: null };
  const w2 = new Float64Array(2 * ng);
  if (wantE) {
    w2.set(W);
    b1.u = new Float64Array(ng);
    b2.u = new Float64Array(ng);
    ifftPair(w2, mg, b1.u, b2.u);
  }
  if (wantV) {
    b1.v = [0, 1, 2, 3, 4, 5].map(() => new Float64Array(ng));
    b2.v = [0, 1, 2, 3, 4, 5].map(() => new Float64Array(ng));
    for (let c = 0; c < 6; c++) {
      for (let n = 0; n < ng; n++) {
        const k = c < 3 ? vg[6 * n + c] : vg2[3 * n + c - 3];
        w2[2 * n] = W[2 * n] * k;
        w2[2 * n + 1] = W[2 * n + 1] * k;
      }
      ifftPair(w2, mg, b1.v[c], b2.v[c]);
    }
  }
  return [b1, b2];
};

/** fieldforce_*_peratom: the stencil sums of u and of the six virial bricks at atom i, into out[0] and out[1..6]. */
const interpPeratom = (x: Float64Array, i: number, lo: [number, number, number], mg: MeshGeom, st: Stencil,
  base: Int32Array, b: PaBricks, out: Float64Array): void => {
  const [nx, ny, nz] = mg.N;
  const P = st.order;
  const r0 = new Float64Array(P), r1 = new Float64Array(P), r2 = new Float64Array(P);
  atomWeights(x, i, lo, mg, st, base, r0, r1, r2);
  const bx = base[3 * i], by = base[3 * i + 1], bz = base[3 * i + 2];
  out.fill(0);
  for (let nn = 0; nn < P; nn++) {
    const mz = wrap(bz + st.nlower + nn, nz);
    const z0 = r2[nn];
    for (let mm = 0; mm < P; mm++) {
      const my = wrap(by + st.nlower + mm, ny);
      const y0 = z0 * r1[mm];
      for (let ll = 0; ll < P; ll++) {
        const mx = wrap(bx + st.nlower + ll, nx);
        const x0 = y0 * r0[ll];
        const idx = (mz * ny + my) * nx + mx;
        if (b.u) out[0] += x0 * b.u[idx];
        if (b.v) for (let c = 0; c < 6; c++) out[1 + c] += x0 * b.v[c][idx];
      }
    }
  }
};

/** kspace_style pppm/disp — see the header of this file for the ported scope. */
export class KSpacePPPMDisp extends KSpaceBase {
  readonly name: string = 'pppm/disp';
  protected override readonly kspaceLabel: string = 'PPPMDisp';
  /** kspace_style accuracy (relative), |arg| as native (PPPMDisp::settings). */
  private accuracyRel = 0;
  private accuracyReal6 = -1;
  private accuracyKspace6 = -1;
  private splittol = 1.0e-6;
  private meshUser: [number, number, number] | null = null;
  private mesh6User: [number, number, number] | null = null;
  private gDispUser = 0;
  private orderCoul = 5;
  private order6 = 5;
  private mixflag = 0;
  private autoDisp = false;
  private pair: Pair | null = null;
  private coulOn = false;
  private dispOn = false;
  /** Arithmetic dispersion mixing (seven grids) when true; geometric (one grid) otherwise. */
  private dispArith = false;
  private gDisp = 0;
  private mesh6: [number, number, number] = [0, 0, 0];
  private meshC: [number, number, number] = [0, 0, 0];
  /** Geometric: B[t] = sqrt(|C_tt|). Arithmetic: B7[7t + k] for k = 0..6 (init_coeffs). */
  private B = new Float64Array(0);
  private B7 = new Float64Array(0);
  private csum = 0;
  private csumij = 0;
  /** calc_csum per type: csumi[t] = sum_j N_j B_t B_j (arithmetic: over the seven terms) and cii[t] the self coefficient. */
  private csumi = new Float64Array(0);
  private cii = new Float64Array(0);
  /** Box lengths the influence functions are built for (set at init, rebuilt when the box changes: native setup()). */
  private prdCur: [number, number, number] = [0, 0, 0];
  /** The Coulomb accuracy estimate of final_accuracy (for the log). */
  private coulAcc = 0;
  private infoC: Influence | null = null;
  private info6: Influence | null = null;

  settings(args: string[]): void {
    if (args.length < 1) throw new StyleError('Illegal kspace_style pppm/disp command');
    this.accuracyRel = Math.abs(parseNum(args[0], 'accuracy'));
    if (this.accuracyRel > 1.0) throw new StyleError(`Invalid relative accuracy ${this.accuracyRel} for kspace_style pppm/disp`);
    this.accuracy = this.accuracyRel;
  }

  modify(key: string, values: string[]): number {
    const meshOf = (name: string): [number, number, number] | null => {
      const m = [0, 1, 2].map((d) => parseNum(values[d], name));
      if (m.every((v) => v === 0)) return null;
      if (m.some((v) => !(v > 0))) throw new StyleError(`kspace_modify ${name} parameters must be all zero or all positive`);
      return m as [number, number, number];
    };
    switch (key) {
      case 'mesh': this.meshUser = meshOf('mesh'); return 3;
      case 'mesh/disp': this.mesh6User = meshOf('mesh/disp'); return 3;
      case 'order': {
        const o = parseNum(values[0], 'order');
        if (!Number.isInteger(o) || o > MAXORDER) throw new StyleError(`PPPMDisp coulomb or dispersion order cannot be greater than ${MAXORDER}`);
        this.orderCoul = o;
        return 1;
      }
      case 'order/disp': {
        const o = parseNum(values[0], 'order/disp');
        if (!Number.isInteger(o) || o > MAXORDER) throw new StyleError(`PPPMDisp coulomb or dispersion order cannot be greater than ${MAXORDER}`);
        this.order6 = o;
        return 1;
      }
      case 'gewald/disp': this.gDispUser = parseNum(values[0], 'gewald/disp'); return 1;
      case 'force/disp/real': this.accuracyReal6 = parseNum(values[0], 'force/disp/real'); return 1;
      case 'force/disp/kspace': this.accuracyKspace6 = parseNum(values[0], 'force/disp/kspace'); return 1;
      case 'mix/disp': {
        const v = values[0];
        if (v === 'pair') this.mixflag = 0;
        else if (v === 'geom') this.mixflag = 1;
        else if (v === 'none') this.mixflag = 2;
        else throw new StyleError(`Unknown kspace_modify mix/disp flag ${v}`);
        return 1;
      }
      case 'splittol': case 'eigtol': {
        const t = parseNum(values[0], 'splittol');
        if (t >= 1.0) throw new StyleError('Kspace_modify splittol value must be smaller than one');
        this.splittol = t;
        return 1;
      }
      case 'disp/auto':
        // native: auto_disp_flag only lifts the "no parameters set" error of PPPMDisp::init (pppm_disp.cpp:349);
        // the estimate below is the same either way
        this.autoDisp = values[0] === 'yes';
        return 1;
      case 'diff':
        if (values[0] === 'ad') throw new StyleError('kspace_modify diff ad is not ported for kspace_style pppm/disp (the ik solver only)');
        if (values[0] !== 'ik') throw new StyleError(`kspace_modify diff ${values[0]} is not a valid option`);
        return 1;
      case 'slab':
        throw new StyleError('kspace_modify slab is not ported for kspace_style pppm/disp');
      case 'kmax/ewald':
        throw new StyleError(`kspace_modify ${key} is not implemented for kspace_style pppm/disp`);
      default:
        return super.modify(key, values);
    }
  }

  linkPair(pair: Pair | null): void { this.pair = pair; }
  /** pppm_disp.cpp constructor: dispersionflag = 1 (KSpace::pair_check refuses a pair without dispersion). */
  override dispersion(): boolean { return true; }

  /** two_charge_force (KSpace::two_charge) and the accuracy: kspace_modify force, else accuracy_relative times it. */
  private accuracyOf(s: SimState): number {
    if (this.forceAbs >= 0) return this.forceAbs;
    const st = s.units.style;
    const twoChargeForce = s.units.qqr2e * (QELECTRON[st] * QELECTRON[st]) / (ANGSTROM[st] * ANGSTROM[st]);
    return this.accuracyRel * twoChargeForce;
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    const pair = this.pair;
    const lj = pair instanceof PairLJLongCoulLong ? pair : null;
    const buck = pair instanceof PairBuckLongCoulLong ? pair : null;
    if (!pair || (!lj && !buck)) {
      throw new StyleError('kspace_style pppm/disp needs pair style lj/long/coul/long or buck/long/coul/long (flag_lj or flag_buck long, flag_coul long)');
    }
    this.dispOn = lj ? lj.ljLong : !!buck && buck.dispFlag;
    this.coulOn = lj ? !!lj.coulLong && cutCoul > 0 : !!buck && buck.coulLongFlag && cutCoul > 0;
    if (!this.coulOn && !this.dispOn) {
      throw new StyleError('kspace_style pppm/disp needs a pair style with a long-range term (lj/long/coul/long with flag_lj long or flag_coul long)');
    }
    if (geom.triclinic || geom.xy !== 0 || geom.xz !== 0 || geom.yz !== 0) {
      throw new StyleError('kspace_style pppm/disp: triclinic boxes are not ported yet');
    }
    const per = s.box.periodic;
    if (!per[0] || !per[1] || !per[2]) throw new StyleError('kspace_style pppm/disp needs a fully periodic box (kspace_modify slab is not ported)');
    if (this.orderCoul < 2 || this.order6 < 2) throw new StyleError('kspace_modify minorder: the PPPMDisp order must be at least 2');
    this.prdCur = [geom.lx, geom.ly, geom.lz];
    this.natoms = s.n;
    this.cutCoul = cutCoul;
    const acc = this.accuracyOf(s);
    // native "Charges are set, but coulombic solver is not used" (PPPMDisp::init, when the Coulomb term is off)
    if (!this.coulOn && hasChargeStyle(s.atomStyle)) ctx.log('WARNING: Charges are set, but coulombic solver is not used');
    // PPPMDisp::init: a dispersion term without its parameters stops before the charge sums and the Coulomb setup
    if (this.dispOn && !this.mesh6User && !(this.gDispUser > 0) && this.accuracyReal6 < 0 && this.accuracyKspace6 < 0 && !this.autoDisp) {
      throw new StyleError('PPPMDisp used but no parameters set, for further information please see the pppm/disp documentation');
    }

    if (this.coulOn) {
      this.setupCharges(s, geom, ctx);
      // the automatic Coulomb G-ewald and mesh; the dielectric subclass keeps its refusal (its q2 needs the dielectric)
      if (this.name === 'pppm/disp/dielectric' && !(this.gewaldUser > 0 && this.meshUser)) {
        throw new StyleError('kspace_style pppm/disp/dielectric with Coulomb charges needs kspace_modify gewald and mesh (the automatic Coulomb G-ewald and grid are not ported for it)');
      }
      const c = this.chooseCoulomb(prdOf(geom), cutCoul, qqrd2e, acc);
      this.gEwald = c.g;
      this.meshC = c.N;
      this.coulAcc = c.acc;
    } else {
      this.gEwald = 0;
    }

    if (this.dispOn) {
      if (this.mixflag === 2) {
        throw new StyleError('kspace_modify mix/disp none (native function[3], the split of the dispersion matrix) is not ported for pppm/disp');
      }
      const nt = pair.ntypes + 1;
      // init_coeffs: the C matrix is lj4 (lj/long) or buck_c (buck/long, extract "B")
      const C = buck ? buck.dispCMatrix() : new Float64Array(nt * nt);
      if (lj) for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) C[i * nt + j] = lj.lj4[i * nt + j];
      // init_coeffs: the arithmetic path needs nsplit > 1 (the number of eigenvalues of C above splittol * max)
      let arith = this.mixflag === 0 && pair.mix === 'arithmetic';
      if (arith && pair.ntypes > 1) {
        const n = pair.ntypes;
        const sub = new Float64Array(n * n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) sub[i * n + j] = C[(i + 1) * nt + (j + 1)];
        const ev = symEigenvalues(sub, n);
        const acrit = Math.abs(ev[0]) * this.splittol;
        let nsplit = 0;
        for (let i = 0; i < n; i++) {
          if (Math.abs(ev[i]) > acrit) nsplit++;
          else break;
        }
        if (nsplit === 1) arith = false;
        ctx.log(`  nsplit = ${nsplit} (${arith ? 'arithmetic: 7 structure factors' : 'geometric mixing for reciprocal space'})`);
      } else if (arith) {
        arith = false;
      }
      this.dispArith = arith;
      if (this.dispArith && !lj) {
        // native init_coeffs: the arithmetic B reads epsilon and sigma, which buck/long does not provide
        throw new StyleError('Epsilon or sigma reference not set by pair style for PPPMDisp');
      }
      // init_coeffs (geometric: sqrt(|C_ii|); arithmetic: the seven-term B of the (sigma, epsilon) expansion)
      this.B = new Float64Array(nt);
      for (let i = 1; i < nt; i++) this.B[i] = Math.sqrt(Math.abs(C[i * nt + i]));
      if (this.dispArith && lj) {
        const c = [1.0, Math.sqrt(6.0), Math.sqrt(15.0), Math.sqrt(20.0), Math.sqrt(15.0), Math.sqrt(6.0), 1.0];
        this.B7 = new Float64Array(7 * nt + 7);
        for (let i = 1; i <= pair.ntypes; i++) {
          const epsI = Math.sqrt(lj.p.get('epsilon', i, i));
          const sigI = lj.p.get('sigma', i, i);
          let sigN = 1.0;
          for (let j = 0; j < 7; j++) {
            this.B7[7 * i + j] = sigN * epsI * c[j] * 0.25;
            sigN *= sigI;
          }
        }
      }
      // calc_csum: csum = sum over atoms of the self coefficient, csumij = sum_ij N_i N_j (pair coefficient sum)
      const cnt = new Float64Array(nt);
      let csum = 0;
      for (let i = 0; i < s.n; i++) cnt[s.type[i]] += 1;
      if (this.dispArith) {
        for (let i = 0; i < s.n; i++) csum += (64.0 / 20.0) * this.B7[7 * s.type[i] + 3] * this.B7[7 * s.type[i] + 3];
        let cij = 0;
        for (let i = 1; i <= pair.ntypes; i++) for (let j = 1; j <= pair.ntypes; j++) for (let k = 0; k <= 6; k++) {
          cij += cnt[i] * this.B7[7 * i + k] * cnt[j] * this.B7[7 * j + 6 - k];
        }
        this.csumij = cij;
        this.cii = new Float64Array(nt);
        this.csumi = new Float64Array(nt);
        for (let i = 1; i <= pair.ntypes; i++) {
          this.cii[i] = (64.0 / 20.0) * this.B7[7 * i + 3] * this.B7[7 * i + 3];
          for (let j = 1; j <= pair.ntypes; j++) for (let k = 0; k <= 6; k++) this.csumi[i] += cnt[j] * this.B7[7 * i + k] * this.B7[7 * (j + 1) - k - 1];
        }
      } else {
        for (let i = 0; i < s.n; i++) csum += this.B[s.type[i]] * this.B[s.type[i]];
        let sb = 0;
        for (let t = 1; t < nt; t++) sb += cnt[t] * this.B[t];
        this.csumij = sb * sb;
        this.cii = new Float64Array(nt);
        this.csumi = new Float64Array(nt);
        for (let i = 1; i < nt; i++) {
          this.cii[i] = this.B[i] * this.B[i];
          for (let j = 1; j < nt; j++) this.csumi[i] += cnt[j] * this.B[i] * this.B[j];
        }
      }
      this.csum = csum;
      const prd = prdOf(geom);
      // cut_vdwl: the global LJ cutoff (lj/long) or the global buck cutoff (buck/long, cut_buck_global)
      const cutoffLj = lj ? lj.cutGlobal : buck!.cutBuckGlobal;
      // set_grid_global_6 (native order: set_init_g6, set_n_pppm_6, factorable), then adjust_gewald_6
      let g6 = this.gDispUser;
      if (!(g6 > 0)) g6 = this.initG6(cutoffLj, acc, prd);
      let N6 = this.mesh6User;
      if (!N6) N6 = this.nPppm6(g6, prd, acc);
      N6 = [N6[0], N6[1], N6[2]];
      for (let d = 0; d < 3; d++) while (!factorable(N6[d])) N6[d]++;
      if (!(this.gDispUser > 0) && this.accuracyKspace6 === this.accuracyReal6) {
        g6 = this.adjustG6(N6, prd, g6, cutoffLj);
      }
      this.gDisp = g6;
      this.mesh6 = N6;
      if (lj) lj.gEwaldDisp = g6;
      if (buck) buck.gEwaldDisp = g6;
      ctx.log(`  Dispersion G vector (1/distance) = ${g6}\n  Dispersion grid = ${N6.join(' ')}\n  Dispersion stencil order = ${this.order6}`);
    }
    if (this.coulOn) ctx.log(`  Coulomb G vector (1/distance) = ${this.gEwald}\n  Coulomb grid = ${this.meshC.join(' ')}\n  Coulomb stencil order = ${this.orderCoul}\n  Coulomb estimated absolute RMS force accuracy = ${this.coulAcc}`);
    this.infoC = null;
    this.info6 = null;
    if (this.coulOn) this.infoC = this.buildInfluence('coul', this.meshC, this.gEwald, this.orderCoul, this.prdCur);
    if (this.dispOn) this.info6 = this.buildInfluence('disp', this.mesh6, this.gDisp, this.order6, this.prdCur);
  }

  /**
   * set_grid_global (Coulomb): the G-ewald estimate (unless the user gave one) and the mesh from the k-space error
   * estimate (unless the user gave one), then the factorable boost; adjust_gewald then runs for an automatic G-ewald.
   * Returns the G-ewald, the mesh and the final Coulomb accuracy (final_accuracy).
   */
  private chooseCoulomb(prd: [number, number, number], cutoff: number, qqrd2e: number, acc: number):
    { g: number; N: [number, number, number]; acc: number } {
    const [xprd, yprd, zprd] = prd;
    const natoms = this.natoms;
    const q2 = this.qsqsum * qqrd2e;
    const order = this.orderCoul;
    let g = this.gewaldUser;
    if (!(g > 0)) {
      g = acc * Math.sqrt(natoms * cutoff * xprd * yprd * zprd) / (2.0 * q2);
      if (g >= 1.0) throw new StyleError('KSpace accuracy too large to estimate G vector');
      g = Math.sqrt(-libmLog(g)) / cutoff;
    }
    let N: [number, number, number];
    if (this.meshUser) {
      N = [this.meshUser[0], this.meshUser[1], this.meshUser[2]];
    } else {
      let h = 4.0 / g;
      let count = 0;
      let Nn: [number, number, number];
      for (;;) {
        Nn = [Math.trunc(xprd / h), Math.trunc(yprd / h), Math.trunc(zprd / h)];
        for (let d = 0; d < 3; d++) if (Nn[d] <= 1) Nn[d] = 2;
        const qopt = qoptCoulombIk(Nn, prd, g, order);
        const dfkspace = Math.sqrt(qopt / natoms) * q2 / (xprd * yprd * zprd);
        count++;
        if (dfkspace <= acc) break;
        if (count > 500) throw new StyleError('Could not compute grid size');
        h *= 0.95;
      }
      N = Nn;
    }
    for (let d = 0; d < 3; d++) while (!factorable(N[d])) N[d]++;
    for (let d = 0; d < 3; d++) {
      if (N[d] >= OFFSET) throw new StyleError('PPPMDisp Coulomb grid is too large');
    }
    // f, derivf and adjust_gewald (Newton on the real-space against k-space error balance) for the mesh N
    const fCoul = (gg: number): number => {
      const dfR = 2.0 * q2 * libmExp(-gg * gg * cutoff * cutoff) / Math.sqrt(natoms * cutoff * xprd * yprd * zprd);
      const qopt = qoptCoulombIk(N, prd, gg, order);
      const dfK = Math.sqrt(qopt / natoms) * q2 / (xprd * yprd * zprd);
      return dfR - dfK;
    };
    if (!(this.gewaldUser > 0)) {
      let converged = false;
      for (let i = 0; i < LARGE; i++) {
        const h = 0.000001;
        const f1 = fCoul(g);
        const f2 = fCoul(g + h);
        const dfx = (f2 - f1) / h;
        const dx = f1 / dfx;
        g -= dx;
        if (Math.abs(fCoul(g)) < SMALL) { converged = true; break; }
      }
      if (!converged) throw new StyleError('Could not compute g_ewald');
    }
    // final_accuracy: the real-space and k-space estimates combined in quadrature
    const dfRFinal = 2.0 * q2 * libmExp(-g * g * cutoff * cutoff) / Math.sqrt(natoms * cutoff * xprd * yprd * zprd);
    const qoptFinal = qoptCoulombIk(N, prd, g, order);
    const dfKFinal = Math.sqrt(qoptFinal / natoms) * q2 / (xprd * yprd * zprd);
    const accFinal = Math.sqrt(dfRFinal * dfRFinal + dfKFinal * dfKFinal);
    return { g, N, acc: accFinal };
  }

  /** Green's function and virial coefficients (setup()) for the box prd. */
  private buildInfluence(kind: 'coul' | 'disp', N: [number, number, number], g: number, P: number,
    prd: [number, number, number]): Influence {
    const mg = meshGeom(N, prd);
    return {
      mg,
      G: kind === 'coul' ? greenCoulomb(mg, g, P) : greenDispersion(mg, g, P),
      vg: kind === 'coul' ? virialCoulomb(mg, g) : virialDispersion(mg, g),
      vg2: virialCross(mg, g, kind),
    };
  }

  /** set_init_g6: the bisection-style start of the dispersion G-ewald from the real-space error. */
  private initG6(cutoffLj: number, acc: number, prd: [number, number, number]): number {
    const accR = this.accuracyReal6 > 0 ? this.accuracyReal6 : acc;
    const err = (gg: number) => ljRspaceError(cutoffLj, gg, this.csum, this.natoms, prd[0], prd[1], prd[2]);
    let g = 1.0 / cutoffLj;
    let gOld = g;
    let dfReal = err(g) - accR;
    let counter = 0;
    if (dfReal > 0) {
      while (dfReal > 0 && counter < LARGE) {
        counter++;
        gOld = g;
        g *= 2;
        dfReal = err(g) - accR;
      }
    }
    if (dfReal < 0) {
      while (dfReal < 0 && counter < LARGE) {
        counter++;
        gOld = g;
        g *= 0.5;
        dfReal = err(g) - accR;
      }
    }
    if (counter >= LARGE - 1) throw new StyleError('Cannot compute initial g_ewald_disp');
    let gmin = Math.min(g, gOld);
    let gmax = Math.max(g, gOld);
    g = gmin + 0.5 * (gmax - gmin);
    counter = 0;
    while (gmax - gmin > SMALL && counter < LARGE) {
      counter++;
      dfReal = err(g) - accR;
      if (dfReal < 0) gmax = g;
      else gmin = g;
      g = gmin + 0.5 * (gmax - gmin);
    }
    if (counter >= LARGE - 1) throw new StyleError('Cannot compute initial g_ewald_disp');
    return g;
  }

  /** f_6 of pppm_disp.cpp: real-space error minus the k-space error estimate on mesh N. */
  private f6(g: number, N: [number, number, number], prd: [number, number, number], cutoffLj: number): number {
    const dfR = ljRspaceError(cutoffLj, g, this.csum, this.natoms, prd[0], prd[1], prd[2]);
    const qopt = qoptDispersionIk(N, prd, g, this.order6);
    const dfK = Math.sqrt(qopt / this.natoms) * this.csum / (prd[0] * prd[1] * prd[2]);
    return dfR - dfK;
  }

  /** set_n_pppm_6: the smallest mesh (h shrinking by 0.95 from 4/G) whose k-space error meets the target. */
  private nPppm6(g: number, prd: [number, number, number], acc: number): [number, number, number] {
    const accK = this.accuracyKspace6 > 0 ? this.accuracyKspace6 : acc;
    let h = 4.0 / g;
    let count = 0;
    const V = prd[0] * prd[1] * prd[2];
    for (;;) {
      const N: [number, number, number] = [Math.trunc(prd[0] / h), Math.trunc(prd[1] / h), Math.trunc(prd[2] / h)];
      for (let d = 0; d < 3; d++) if (N[d] <= 1) N[d] = 2;
      const qopt = qoptDispersionIk(N, prd, g, this.order6);
      const dfK = Math.sqrt(qopt / this.natoms) * this.csum / V;
      count++;
      if (dfK <= accK) return N;
      if (count > 500) throw new StyleError('Could not compute grid size for Dispersion');
      h *= 0.95;
    }
  }

  /** adjust_gewald_6: Newton iteration on f_6 for the grid N (forward-difference derivative, h = 1e-6). */
  private adjustG6(N: [number, number, number], prd: [number, number, number], g0: number, cutoffLj: number): number {
    let g = g0;
    for (let i = 0; i < LARGE; i++) {
      const f1 = this.f6(g, N, prd, cutoffLj);
      const f2 = this.f6(g + 0.000001, N, prd, cutoffLj);
      const dfx = (f2 - f1) / 0.000001;
      if (dfx === 0.0 || dfx !== dfx) break;
      let dx = f1 / dfx;
      while (g - dx <= 0.0) dx *= 0.5;
      g -= dx;
      if (Math.abs(this.f6(g, N, prd, cutoffLj)) < SMALL) return g;
    }
    throw new StyleError('Could not adjust g_ewald_6');
  }

  /** The positions the Coulomb mesh sees: the atoms here (kspace_style pppm/disp/tip4p puts the O charges on M sites). */
  protected coulombPositions(s: SimState, geom: Geometry): Float64Array {
    void geom;
    return s.x;
  }

  /** Charges of the Coulomb density of the force pass (make_rho_c); pppm/disp uses q. */
  protected coulombDensityCharges(s: SimState): Float64Array {
    return s.q;
  }

  /** Charges that multiply the interpolated Coulomb field in the force of atom i (fieldforce_c_ik); pppm/disp uses q. */
  protected coulombForceCharges(s: SimState): Float64Array {
    return s.q;
  }

  /** Charges of a second Coulomb density whose energy and virial add to the Coulomb sums, or null (pppm/disp: none). */
  protected coulombEnergyCharges(s: SimState): Float64Array | null {
    void s;
    return null;
  }

  /** Receives the raw interpolated Coulomb field (before qqrd2e) of atom i; no-op for pppm/disp. */
  protected coulombFieldAt(i: number, ex: number, ey: number, ez: number): void {
    void i;
    void ex;
    void ey;
    void ez;
  }

  /** Adds the Coulomb force of atom i (the atom itself here; the TIP4P subclass projects an M-site force on O and H). */
  protected addCoulombForce(kc: KSpaceCompute, i: number, fx: number, fy: number, fz: number): void {
    kc.f[3 * i] += fx;
    kc.f[3 * i + 1] += fy;
    kc.f[3 * i + 2] += fz;
  }

  /** Whether this style's per-atom energy and virial are ported (the TIP4P and dielectric variants: not yet). */
  protected peratomPorted(): boolean { return true; }

  /** fieldforce_c_peratom: the interpolated potential o[0] and virial o[1..6] of atom i times qfactor, on atom i. */
  protected addCoulombPeratom(kc: KSpaceCompute, i: number, o: Float64Array, qfactor: number): void {
    if (kc.eatom) kc.eatom[i] += o[0] * qfactor;
    if (kc.vatom) for (let c = 0; c < 6; c++) kc.vatom[6 * i + c] += o[1 + c] * qfactor;
  }

  compute(kc: KSpaceCompute): void {
    const pa = kc.eatom !== null || kc.vatom !== null;
    if (pa && !this.peratomPorted()) throw new StyleError(`compute pe/atom or stress/atom with kspace_style ${this.name} is not ported (per-atom energies and virials)`);
    const s = kc.s;
    const n = s.n;
    const qqrd2e = kc.qqrd2e;
    const lo: [number, number, number] = [kc.geom.lo[0], kc.geom.lo[1], kc.geom.lo[2]];
    const x = s.x;
    // native PPPMDisp::setup() (called by the integrator when the box changes): the box-dependent Green's functions and
    // virial coefficients are rebuilt for the current box; the grids and G-ewald are the init ones
    const prdNow: [number, number, number] = [kc.geom.lx, kc.geom.ly, kc.geom.lz];
    if (prdNow[0] !== this.prdCur[0] || prdNow[1] !== this.prdCur[1] || prdNow[2] !== this.prdCur[2]) {
      this.prdCur = prdNow;
      if (this.coulOn) this.infoC = this.buildInfluence('coul', this.meshC, this.gEwald, this.orderCoul, prdNow);
      if (this.dispOn) this.info6 = this.buildInfluence('disp', this.mesh6, this.gDisp, this.order6, prdNow);
    }
    const V = prdNow[0] * prdNow[1] * prdNow[2];
    const virial = new Float64Array(6);
    // the per-atom self terms are applied after every mesh term (the end of compute() of the C++)
    let coulSelf: { qsum: number; qqrd2e: number } | null = null;
    let dispSelf: { V: number; g: number } | null = null;

    if (this.coulOn && this.infoC) {
      const info = this.infoC;
      const st = makeStencil(this.orderCoul);
      const mg = info.mg;
      const q = s.q;
      const xc = this.coulombPositions(s, kc.geom);
      let qsum = 0, qsqsum = 0;
      for (let i = 0; i < n; i++) { qsum += q[i]; qsqsum += q[i] * q[i]; }
      const delvolinv = (mg.N[0] / mg.prd[0]) * (mg.N[1] / mg.prd[1]) * (mg.N[2] / mg.prd[2]);
      const base = mapAtoms(xc, n, lo, mg, st);
      const s2 = (1 / (mg.N[0] * mg.N[1] * mg.N[2])) ** 2;
      const egy = { v: 0 };
      const vir = new Float64Array(6);
      // the density of the force pass (make_rho_c) and, when the style has one, the second density whose energy and
      // virial add to the same sums (pppm_disp_dielectric.cpp: the unscaled charges after the scaled pass)
      const qd = this.coulombDensityCharges(s);
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) w[i] = delvolinv * qd[i];
      const F = spread(xc, n, lo, mg, st, base, w);
      fft3d(F, mg.N[0], mg.N[1], mg.N[2], -1);
      pairSum(F, F, info.G, info.vg, s2, 1, egy, vir);
      const qe = this.coulombEnergyCharges(s);
      if (qe) {
        const we = new Float64Array(n);
        for (let i = 0; i < n; i++) we[i] = delvolinv * qe[i];
        const Fe = spread(xc, n, lo, mg, st, base, we);
        fft3d(Fe, mg.N[0], mg.N[1], mg.N[2], -1);
        pairSum(Fe, Fe, info.G, info.vg, s2, 1, egy, vir);
      }
      let e1 = 0.5 * V * egy.v;
      e1 -= this.gEwald * qsqsum / MY_PIS + (Math.PI / 2) * qsum * qsum / (this.gEwald * this.gEwald * V);
      e1 *= qqrd2e;
      const grids = fieldsIk(F, mg, info.G);
      const fq = this.coulombForceCharges(s);
      for (let i = 0; i < n; i++) {
        const [ex, ey, ez] = interpField(xc, i, lo, mg, st, base, grids);
        this.coulombFieldAt(i, ex, ey, ez);
        const qf = qqrd2e * fq[i];
        this.addCoulombForce(kc, i, qf * ex, qf * ey, qf * ez);
      }
      for (let c = 0; c < 6; c++) virial[c] += 0.5 * qqrd2e * V * vir[c];
      kc.acc.elong += e1;
      if (pa) {
        // fieldforce_c_peratom: qfactor = 0.5 qqrd2e scale q_i times the interpolated potential and virial bricks
        const b = peratomBricks(F, mg, info.G, info.vg, info.vg2, kc.eatom !== null, kc.vatom !== null);
        const o = new Float64Array(7);
        for (let i = 0; i < n; i++) {
          interpPeratom(xc, i, lo, mg, st, base, b, o);
          this.addCoulombPeratom(kc, i, o, 0.5 * qqrd2e * q[i]);
        }
        coulSelf = { qsum, qqrd2e };
      }
    }

    if (this.dispOn && this.info6) {
      const info = this.info6;
      const st = makeStencil(this.order6);
      const mg = info.mg;
      const g = this.gDisp;
      const delvolinv = (mg.N[0] / mg.prd[0]) * (mg.N[1] / mg.prd[1]) * (mg.N[2] / mg.prd[2]);
      const ng = mg.N[0] * mg.N[1] * mg.N[2];
      const s2 = (1 / ng) ** 2;
      const base = mapAtoms(x, n, lo, mg, st);
      const piSqrt = Math.PI * MY_PIS;
      const egy = { v: 0 };
      const vir = new Float64Array(6);
      // the (density, field) pairs: geometric one grid; arithmetic seven grids with the energy of the pairs (k, 6 - k)
      const kinds = this.dispArith ? [0, 1, 2, 3, 4, 5, 6] : [0];
      const fields: [Float64Array, Float64Array, Float64Array][] = [];
      const Fs: Float64Array[] = [];
      for (const k of kinds) {
        const w = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          const coef = this.dispArith ? this.B7[7 * s.type[i] + k] : this.B[s.type[i]];
          w[i] = delvolinv * coef;
        }
        const F = spread(x, n, lo, mg, st, base, w);
        fft3d(F, mg.N[0], mg.N[1], mg.N[2], -1);
        Fs.push(F);
      }
      if (!this.dispArith) {
        pairSum(Fs[0], Fs[0], info.G, info.vg, s2, 1, egy, vir);
      } else {
        pairSum(Fs[3], Fs[3], info.G, info.vg, s2, 1, egy, vir);
        for (let m = 0; m < 3; m++) pairSum(Fs[m], Fs[6 - m], info.G, info.vg, s2, 2, egy, vir);
      }
      for (const F of Fs) fields.push(fieldsIk(F, mg, info.G));
      let e6 = 0.5 * V * egy.v;
      e6 += -piSqrt / (6 * V) * libmPow(g, 3) * this.csumij + (1.0 / 12.0) * libmPow(g, 6) * this.csum;
      for (let i = 0; i < n; i++) {
        const t = s.type[i];
        let fx = 0, fy = 0, fz = 0;
        for (let idx = 0; idx < kinds.length; idx++) {
          const [ex, ey, ez] = interpField(x, i, lo, mg, st, base, fields[idx]);
          // fieldforce_a_ik: the field of density k pairs with B[t][6 - k]; the geometric case has one grid, B[t]
          const coef = this.dispArith ? this.B7[7 * t + 6 - kinds[idx]] : this.B[t];
          fx += coef * ex;
          fy += coef * ey;
          fz += coef * ez;
        }
        kc.f[3 * i] += fx;
        kc.f[3 * i + 1] += fy;
        kc.f[3 * i + 2] += fz;
      }
      for (let c = 0; c < 6; c++) virial[c] += 0.5 * V * vir[c];
      if (pa) {
        // fieldforce_g_peratom (lj = B_t / 2) or fieldforce_a_peratom (grid k with lj = B_t[6 - k] / 2; the pairs
        // (0, 6), (1, 5), (2, 4) from poisson_2s_ik and grid 3 from poisson_ik)
        const wantE = kc.eatom !== null, wantV = kc.vatom !== null;
        const bricks: PaBricks[] = new Array(kinds.length);
        if (!this.dispArith) bricks[0] = peratomBricks(Fs[0], mg, info.G, info.vg, info.vg2, wantE, wantV);
        else {
          bricks[3] = peratomBricks(Fs[3], mg, info.G, info.vg, info.vg2, wantE, wantV);
          for (let m = 0; m < 3; m++) [bricks[m], bricks[6 - m]] = peratomBricks2s(Fs[m], Fs[6 - m], mg, info.G, info.vg, info.vg2, wantE, wantV);
        }
        const o = new Float64Array(7);
        const sum = new Float64Array(7);
        for (let i = 0; i < n; i++) {
          const t = s.type[i];
          sum.fill(0);
          for (let idx = 0; idx < kinds.length; idx++) {
            interpPeratom(x, i, lo, mg, st, base, bricks[idx], o);
            const lj = (this.dispArith ? this.B7[7 * t + 6 - kinds[idx]] : this.B[t]) * 0.5;
            for (let c = 0; c < 7; c++) sum[c] += o[c] * lj;
          }
          if (kc.eatom) kc.eatom[i] += sum[0];
          if (kc.vatom) for (let c = 0; c < 6; c++) kc.vatom[6 * i + c] += sum[1 + c];
        }
        dispSelf = { V, g };
      }
      const a = piSqrt / (6 * V) * libmPow(g, 3) * this.csumij;
      virial[0] -= a;
      virial[1] -= a;
      virial[2] -= a;
      kc.acc.elong += e6;
    }
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += virial[c];
    // the per-atom self terms (the end of PPPMDisp::compute): the Coulomb self energy, then the dispersion self energy
    // and its virial on the diagonal
    if (kc.eatom && coulSelf) {
      const q = s.q, qscale = coulSelf.qqrd2e, gE = this.gEwald;
      for (let i = 0; i < n; i++) {
        kc.eatom[i] -= qscale * gE * q[i] * q[i] / MY_PIS + qscale * (Math.PI / 2) * q[i] * coulSelf.qsum / (gE * gE * V);
      }
    }
    if (dispSelf) {
      const piSqrt = Math.PI * MY_PIS;
      const g3 = libmPow(dispSelf.g, 3), g6 = libmPow(dispSelf.g, 6);
      for (let i = 0; i < n; i++) {
        const t = s.type[i];
        if (kc.eatom) kc.eatom[i] += -piSqrt / (6 * V) * g3 * this.csumi[t] + (1.0 / 12.0) * g6 * this.cii[t];
        if (kc.vatom) for (let c = 0; c < 3; c++) kc.vatom[6 * i + c] -= piSqrt / (6 * V) * g3 * this.csumi[t];
      }
    }
  }
}

/** The influence functions of one term (Green's function G over the mesh, and the virial coefficients vg). */
interface Influence {
  mg: MeshGeom;
  G: Float64Array;
  vg: Float64Array;
  /** The cross-term virial coefficients of the per-atom pass (setup() vg2). */
  vg2: Float64Array;
}
