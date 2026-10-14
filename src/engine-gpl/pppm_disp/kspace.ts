/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pppm_disp.cpp and pppm_disp.h at b332f28 (PPPMDisp::settings, modify (kspace.cpp
 * modify_params keywords mesh/disp, order/disp, gewald/disp, mix/disp, force/disp/real, force/disp/kspace, splittol,
 * disp/auto), init, setup, compute, the energy and virial sums, poisson_ik and poisson_2s_ik (orthogonal branches),
 * make_rho_g, make_rho_a, fieldforce_g_ik, fieldforce_a_ik, calc_csum, init_coeffs (geometric and arithmetic B), and
 * the G-ewald and mesh choice for dispersion: set_grid_global_6, set_init_g6, set_n_pppm_6, adjust_gewald_6, f_6).
 * Contributing authors of pppm_disp.cpp: Rolf Isele-Holder (Aachen University), Paul Crozier (SNL). Copyright (2003)
 * Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too.
 *
 * Scope of this port (everything else is refused by name, as StyleError):
 *  - orthogonal, fully periodic boxes; no slab correction; no per-atom energy or virial; kspace_modify diff ik only;
 *  - dispersion with geometric mixing (one grid: B_i = sqrt(|4 eps_ii sigma_ii^6|)) and with arithmetic mixing
 *    (seven grids, B_ik = sqrt(eps_i) sigma_i^k sqrt(C(6,k)) / 4, the pair energies of the pairs (k, 6-k)); the
 *    arithmetic path needs at least two distinct eigen-directions of the C6 matrix (nsplit > 1), else geometric as in
 *    native init_coeffs; mix/disp none (function[3]) is refused;
 *  - Coulomb (pair long on the coulomb side) only with user kspace_modify gewald and mesh; the native automatic
 *    Coulomb G-ewald and grid estimate is not ported here;
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
import { fft3d } from '../../engine/force/fft';
import {
  OFFSET, MAXORDER, SMALL, LARGE, factorable, makeStencil, rho1d, meshGeom, greenCoulomb, greenDispersion,
  virialCoulomb, virialDispersion, ljRspaceError, qoptDispersionIk, symEigenvalues, type MeshGeom, type Stencil,
} from './grid';

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

/** kspace_style pppm/disp — see the header of this file for the ported scope. */
export class KSpacePPPMDisp extends KSpaceBase {
  readonly name: string = 'pppm/disp';
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
  /** Box lengths at init; the influence functions and the grids are built for this box. */
  private prdInit: [number, number, number] = [0, 0, 0];
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

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    const pair = this.pair;
    if (!(pair instanceof PairLJLongCoulLong)) {
      throw new StyleError('kspace_style pppm/disp needs pair style lj/long/coul/long (flag_lj long or flag_coul long)');
    }
    this.dispOn = pair.ljLong;
    this.coulOn = !!pair.coulLong && cutCoul > 0;
    if (!this.coulOn && !this.dispOn) {
      throw new StyleError('kspace_style pppm/disp needs a pair style with a long-range term (lj/long/coul/long with flag_lj long or flag_coul long)');
    }
    if (geom.triclinic || geom.xy !== 0 || geom.xz !== 0 || geom.yz !== 0) {
      throw new StyleError('kspace_style pppm/disp: triclinic boxes are not ported yet');
    }
    const per = s.box.periodic;
    if (!per[0] || !per[1] || !per[2]) throw new StyleError('kspace_style pppm/disp needs a fully periodic box (kspace_modify slab is not ported)');
    if (this.orderCoul < 2 || this.order6 < 2) throw new StyleError('kspace_modify minorder: the PPPMDisp order must be at least 2');
    this.prdInit = [geom.lx, geom.ly, geom.lz];
    this.natoms = s.n;
    this.cutCoul = cutCoul;
    const acc = this.absAccuracy(s, qqrd2e);

    if (this.coulOn) {
      // the automatic Coulomb G-ewald and grid (set_grid_global, adjust_gewald, final_accuracy) are not ported here
      if (!(this.gewaldUser > 0) || !this.meshUser) {
        throw new StyleError('kspace_style pppm/disp with Coulomb charges needs kspace_modify gewald and mesh (the automatic Coulomb G-ewald and grid are not ported)');
      }
      this.gEwald = this.gewaldUser;
      this.meshC = this.meshUser;
      this.setupCharges(s, geom, ctx);
    } else {
      this.gEwald = 0;
    }

    if (this.dispOn) {
      if (this.mixflag === 2) {
        throw new StyleError('kspace_modify mix/disp none (native function[3], the split of the dispersion matrix) is not ported for pppm/disp');
      }
      const nt = pair.ntypes + 1;
      const C = new Float64Array(nt * nt);
      for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) C[i * nt + j] = pair.lj4[i * nt + j];
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
      if (!this.mesh6User && !(this.gDispUser > 0) && this.accuracyReal6 < 0 && this.accuracyKspace6 < 0 && !this.autoDisp) {
        throw new StyleError('PPPMDisp used but no parameters set, for further information please see the pppm/disp documentation');
      }
      // init_coeffs (geometric: sqrt(|C_ii|); arithmetic: the seven-term B of the (sigma, epsilon) expansion)
      this.B = new Float64Array(nt);
      for (let i = 1; i < nt; i++) this.B[i] = Math.sqrt(Math.abs(C[i * nt + i]));
      if (this.dispArith) {
        const c = [1.0, Math.sqrt(6.0), Math.sqrt(15.0), Math.sqrt(20.0), Math.sqrt(15.0), Math.sqrt(6.0), 1.0];
        this.B7 = new Float64Array(7 * nt + 7);
        for (let i = 1; i <= pair.ntypes; i++) {
          const epsI = Math.sqrt(pair.p.get('epsilon', i, i));
          const sigI = pair.p.get('sigma', i, i);
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
      } else {
        for (let i = 0; i < s.n; i++) csum += this.B[s.type[i]] * this.B[s.type[i]];
        let sb = 0;
        for (let t = 1; t < nt; t++) sb += cnt[t] * this.B[t];
        this.csumij = sb * sb;
      }
      this.csum = csum;
      const prd = this.prdInit;
      const cutoffLj = pair.cutGlobal;
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
      pair.gEwaldDisp = g6;
      ctx.log(`  Dispersion G vector (1/distance) = ${g6}\n  Dispersion grid = ${N6.join(' ')}\n  Dispersion stencil order = ${this.order6}`);
    }
    if (this.coulOn) ctx.log(`  Coulomb G vector (1/distance) = ${this.gEwald}\n  Coulomb grid = ${this.meshC.join(' ')}\n  Coulomb stencil order = ${this.orderCoul}`);
    this.infoC = null;
    this.info6 = null;
    if (this.coulOn) this.infoC = this.buildInfluence('coul', this.meshC, this.gEwald, this.orderCoul);
    if (this.dispOn) this.info6 = this.buildInfluence('disp', this.mesh6, this.gDisp, this.order6);
  }

  /** Green's function and virial coefficients (setup()) for the box at init. */
  private buildInfluence(kind: 'coul' | 'disp', N: [number, number, number], g: number, P: number): Influence {
    const prd: [number, number, number] = [this.prdInit[0], this.prdInit[1], this.prdInit[2]];
    const mg = meshGeom(N, prd);
    return {
      mg,
      G: kind === 'coul' ? greenCoulomb(mg, g, P) : greenDispersion(mg, g, P),
      vg: kind === 'coul' ? virialCoulomb(mg, g) : virialDispersion(mg, g),
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

  compute(kc: KSpaceCompute): void {
    if (kc.eatom || kc.vatom) throw new StyleError('compute pe/atom or stress/atom with kspace_style pppm/disp is not ported (per-atom energies and virials)');
    const s = kc.s;
    const n = s.n;
    const qqrd2e = kc.qqrd2e;
    const lo: [number, number, number] = [kc.geom.lo[0], kc.geom.lo[1], kc.geom.lo[2]];
    const x = s.x;
    if (kc.geom.lx !== this.prdInit[0] || kc.geom.ly !== this.prdInit[1] || kc.geom.lz !== this.prdInit[2]) {
      throw new StyleError('kspace_style pppm/disp: the box changed after init (rerun the kspace init; box-dependent setup is not ported)');
    }
    const V = this.prdInit[0] * this.prdInit[1] * this.prdInit[2];
    const virial = new Float64Array(6);

    if (this.coulOn && this.infoC) {
      const info = this.infoC;
      const st = makeStencil(this.orderCoul);
      const mg = info.mg;
      const q = s.q;
      let qsum = 0, qsqsum = 0;
      for (let i = 0; i < n; i++) { qsum += q[i]; qsqsum += q[i] * q[i]; }
      const delvolinv = (mg.N[0] / mg.prd[0]) * (mg.N[1] / mg.prd[1]) * (mg.N[2] / mg.prd[2]);
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) w[i] = delvolinv * q[i];
      const base = mapAtoms(x, n, lo, mg, st);
      const F = spread(x, n, lo, mg, st, base, w);
      fft3d(F, mg.N[0], mg.N[1], mg.N[2], -1);
      const egy = { v: 0 };
      const vir = new Float64Array(6);
      const s2 = (1 / (mg.N[0] * mg.N[1] * mg.N[2])) ** 2;
      pairSum(F, F, info.G, info.vg, s2, 1, egy, vir);
      let e1 = 0.5 * V * egy.v;
      e1 -= this.gEwald * qsqsum / Math.sqrt(Math.PI) + (Math.PI / 2) * qsum * qsum / (this.gEwald * this.gEwald * V);
      e1 *= qqrd2e;
      const grids = fieldsIk(F, mg, info.G);
      for (let i = 0; i < n; i++) {
        const [ex, ey, ez] = interpField(x, i, lo, mg, st, base, grids);
        const qf = qqrd2e * q[i];
        kc.f[3 * i] += qf * ex;
        kc.f[3 * i + 1] += qf * ey;
        kc.f[3 * i + 2] += qf * ez;
      }
      for (let c = 0; c < 6; c++) virial[c] += 0.5 * qqrd2e * V * vir[c];
      kc.acc.elong += e1;
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
      const piSqrt = Math.PI * Math.sqrt(Math.PI);
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
      e6 += -piSqrt / (6 * V) * Math.pow(g, 3) * this.csumij + (1.0 / 12.0) * Math.pow(g, 6) * this.csum;
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
      const a = piSqrt / (6 * V) * Math.pow(g, 3) * this.csumij;
      virial[0] -= a;
      virial[1] -= a;
      virial[2] -= a;
      kc.acc.elong += e6;
    }
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += virial[c];
  }
}

/** The influence functions of one term (Green's function G over the mesh, and the virial coefficients vg). */
interface Influence {
  mg: MeshGeom;
  G: Float64Array;
  vg: Float64Array;
}
