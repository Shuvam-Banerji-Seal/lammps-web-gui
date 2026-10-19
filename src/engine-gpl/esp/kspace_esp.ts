/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/esp.cpp and esp.h at
 * b332f28 (ESP::settings, init, setup, reset_grid (not used), compute, set_grid_global, factorable, set_grid_local,
 * estimate_order, build_table, compute_rho1d, compute_drho1d, compute_rho_coeff (not used by the ad path),
 * compute_gf_ad, compute_sf_precoeff, particle_map, make_rho, brick2fft, poisson_ad, fieldforce_ad), with the
 * prolate spheroidal wave function tables of src/KSPACE/math_pswf.cpp (see pswf_*.ts in this folder).
 * Contributing authors of esp.cpp: Jiuyang Liang, Libin Lu, Shidong Jiang (Flatiron). Copyright (2003) Sandia
 * Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too.
 *
 * Scope of this port (everything else is refused by name, as StyleError):
 *  - orthogonal, fully periodic boxes, one process; no slab correction, no triclinic box;
 *  - kspace_modify diff ad only (the default, "analytic differentiation": two FFTs per step); diff ik is refused;
 *  - no kspace_modify mesh, order, gewald, force, slab (the grid and order follow from the accuracies);
 *  - the per-atom energy and virial (compute pe/atom, stress/atom) of poisson_peratom and fieldforce_peratom: as native,
 *    the per-atom virial meshes carry the vg term only (not the greensfn2 vg2 term of the global virial);
 *  - no group-group (compute group/group) and no TIP4P;
 *  - the pair styles coul/esp and lj/cut/coul/esp read the polynomials of this object (linkPair).
 *
 * Operation order follows esp.cpp; the transcendental calls go through the glibc ports of src/engine-gpl/libm.
 */
import { StyleError, type KSpaceCompute, type StyleContext, type Pair } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { parseNum } from '../../engine/force/util';
import { KSpaceBase, modifyNoEffect } from '../../engine/force/kspace/ewald';
import { fft3d } from '../../engine/force/fft';
import { sinFma as libmSin } from '../libm/sincos_fma';
import { prolc180, forcePoly, energyPoly, fourierPoly, spreadFourierPoly, spreadRealPoly } from './pswf_poly';
import { PairCoulEsp } from './pair_coul_esp';
import { PairLJCutCoulEsp } from './pair_lj_cut_coul_esp';

const MAXORDER = 32;
const OFFSET = 16384;
const MY_PI = 3.141592653589793;     // math_const.h MY_PI
const MY_2PI = 6.283185307179586;    // math_const.h MY_2PI

/** esp.cpp poly_horner: coeff[0] + coeff[1] x + ... + coeff[n-1] x^(n-1). */
const polyHorner = (x: number, coeff: number[], n: number): number => {
  let p = coeff[n - 1];
  for (let i = n - 2; i >= 0; --i) p = p * x + coeff[i];
  return p;
};

/** esp.cpp poly_and_deriv_horner: p(x) and dp/dx. */
const polyAndDerivHorner = (x: number, coeff: number[], n: number): { p: number; dp: number } => {
  let p = coeff[n - 1];
  let dp = 0.0;
  for (let i = n - 2; i >= 0; --i) {
    dp = dp * x + p;
    p = p * x + coeff[i];
  }
  return { p, dp };
};

/** esp.cpp ESP::estimate_order(accuracy). */
const estimateOrder = (accuracy: number): number => {
  const p = -Math.log10(accuracy);
  const pRound = Math.round(p);
  const eps = 0.2;
  let order: number;
  if (Math.abs(p - pRound) < eps) order = Math.trunc(pRound) * 2 - 2;
  else order = Math.trunc(Math.ceil(p)) * 2 - 3;
  if (order < 2) order = 2;
  if (order > 16) order = 16;
  return order;
};

/** Index into the periodic grid (serial: the ghost cells of the brick wrap onto the owned grid). */
const wrap = (i: number, n: number): number => ((i % n) + n) % n;

/** kspace_style esp, analytic differentiation (ad), one process. */
export class KSpaceESP extends KSpaceBase {
  readonly name: string = 'esp';
  /** kspace_style esp accuracy_relative (splitting) and spreading_accuracy (|args| as native settings()). */
  accuracyRelative = 0;
  spreadingAccuracy = 0;
  order = 0;
  /** prolate parameters of the splitting (select_c) and of the spreading (spreading_select_c). */
  selectC = 0;
  spreadingSelectC = 0;
  cutoff = 0;
  /** polynomial coefficients used by the pair styles and the self energy (build_table). */
  forcePolyCoeff: number[] = [];
  energyPolyCoeff: number[] = [];
  private fourierSplitPolyCoeff: number[] = [];
  private fourierSpreadPolyCoeff: number[] = [];
  private rhoCoeff: number[][] = [];     // [l][k + (order-1)/2 ]
  private drhoCoeff: number[][] = [];
  private polyOrder = 0;
  private pair: Pair | null = null;
  private nx = 0; private ny = 0; private nz = 0;
  private shift = 0; private shiftone = 0;
  private nlower = 0; private nupper = 0;
  private delxinv = 0; private delyinv = 0; private delzinv = 0; private delvolinv = 0;
  private volume2 = 0;
  private prdInit: [number, number, number] = [0, 0, 0];
  private boxBuilt: [number, number, number] | null = null;
  private fkx = new Float64Array(0);
  private fky = new Float64Array(0);
  private fkz = new Float64Array(0);
  private greensfn = new Float64Array(0);
  private greensfn2 = new Float64Array(0);
  private vg = new Float64Array(0);
  private vg2 = new Float64Array(0);
  private sfPre: Float64Array[] = [];
  private sfCoeff = new Float64Array(6);
  private work1 = new Float64Array(0);
  private work2 = new Float64Array(0);
  private density = new Float64Array(0);
  private u = new Float64Array(0);
  private part2grid = new Int32Array(0);
  private lo: [number, number, number] = [0, 0, 0];

  settings(args: string[]): void {
    if (args.length < 1 || args.length > 2) throw new StyleError('Illegal kspace_style esp command');
    this.accuracyRelative = Math.abs(parseNum(args[0], 'splitting accuracy'));
    if (args.length === 1) this.spreadingAccuracy = 0.25 * this.accuracyRelative;
    else this.spreadingAccuracy = Math.abs(parseNum(args[1], 'spreading accuracy'));
    if (this.accuracyRelative <= 0.0 || this.spreadingAccuracy <= 0.0 ||
        this.accuracyRelative > 1.0 || this.spreadingAccuracy > 1.0) {
      throw new StyleError(`Invalid relative accuracy ${this.accuracyRelative} or spreading accuracy ${this.spreadingAccuracy} for kspace_style esp`);
    }
    this.accuracy = this.accuracyRelative;
    this.order = estimateOrder(this.accuracyRelative);
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'diff':
        if (values[0] === 'ad') return 1;
        throw new StyleError(`kspace_modify diff ${values[0]} is not ported for kspace_style esp (only the default ad differentiation)`);
      case 'mesh': case 'order': case 'gewald': case 'force': case 'slab': case 'kmax/ewald':
      case 'mesh/disp': case 'order/disp': case 'gewald/disp': case 'splittol': case 'disp/auto':
        throw new StyleError(`kspace_modify ${key} is not ported for kspace_style esp (the grid and order follow from the accuracies)`);
      default: {
        const n = modifyNoEffect(key, values);
        if (n) return n;
        return super.modify(key, values);
      }
    }
  }

  linkPair(pair: Pair | null): void {
    this.pair = pair;
    // The pair styles of this family read the polynomials from the kspace object (pair_coul_esp.cpp init_style).
    if (pair instanceof PairCoulEsp || pair instanceof PairLJCutCoulEsp) pair.kspace = this;
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    if (geom.triclinic || geom.xy !== 0 || geom.xz !== 0 || geom.yz !== 0) {
      throw new StyleError('kspace_style esp: triclinic boxes are not ported (Cannot (yet) use ESP with triclinic box)');
    }
    const per = s.box.periodic;
    if (!per[0] || !per[1] || !per[2]) throw new StyleError('Cannot use non-periodic boundaries with ESP');
    if (this.order < 2 || this.order > MAXORDER) throw new StyleError(`ESP order cannot be < 2 or > ${MAXORDER}`);
    this.setupCharges(s, geom, ctx);
    this.cutoff = cutCoul;
    this.cutCoul = cutCoul;
    this.natoms = s.n;
    this.prdInit = [geom.lx, geom.ly, geom.lz];
    this.volume = geom.volume(3);
    this.lo = [geom.lo[0], geom.lo[1], geom.lo[2]];
    this.selectC = prolc180(this.accuracyRelative);
    this.spreadingSelectC = prolc180(this.spreadingAccuracy);
    this.setGridGlobal(geom);
    this.setGridLocal();
    this.buildTable(ctx);
    this.allocateGrid();
    this.computeSfPrecoeff();
    ctx.log(`ESP initialization ...\n  Force poly size: ${this.forcePolyCoeff.length}\n  Energy poly size: ${this.energyPolyCoeff.length}\n  Fourier poly size: ${this.fourierSplitPolyCoeff.length}\n  Spread real poly size: ${this.polyOrder}\n  Fourier spreading poly size: ${this.fourierSpreadPolyCoeff.length}\n  Spreading parameter c = ${fmtG(this.spreadingSelectC)}\n  Splitting parameter c = ${fmtG(this.selectC)}\n  grid = ${this.nx} ${this.ny} ${this.nz}\n  stencil order = ${this.order}\n  estimated relative splitting force accuracy = ${fmtG(this.accuracyRelative)}\n  estimated relative spreading accuracy = ${fmtG(this.spreadingAccuracy)}`);
    this.setup(geom);
  }

  /** esp.cpp set_grid_global (not gridflag, orthogonal, ad). */
  private setGridGlobal(geom: Geometry): void {
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    const zprdSlab = zprd * 1.0;
    const cutoff = this.cutoff;
    const hx = MY_PI * cutoff / this.selectC;
    const hy = hx, hz = hx;
    let nx = Math.trunc(Math.ceil(xprd / hx));
    let ny = Math.trunc(Math.ceil(yprd / hy));
    let nz = Math.trunc(Math.ceil(zprdSlab / hz));
    if (nx <= 1) nx = 2;
    if (ny <= 1) ny = 2;
    if (nz <= 1) nz = 2;
    while (!factorable(nx)) nx++;
    while (!factorable(ny)) ny++;
    while (!factorable(nz)) nz++;
    if (nx >= OFFSET || ny >= OFFSET || nz >= OFFSET) throw new StyleError('ESP grid is too large');
    this.nx = nx; this.ny = ny; this.nz = nz;
  }

  /** esp.cpp set_grid_local: shifts and stencil extents, and the (single-process) FFT block. */
  private setGridLocal(): void {
    if (this.order % 2) this.shift = OFFSET + 0.5;
    else this.shift = OFFSET;
    if (this.order % 2) this.shiftone = 0.0;
    else this.shiftone = 0.5;
    this.nlower = -Math.trunc((this.order - 1) / 2);
    this.nupper = Math.trunc(this.order / 2);
  }

  /** esp.cpp build_table: the polynomials for the splitting, the energy, the spreading and the Fourier kernels. */
  private buildTable(ctx: StyleContext): void {
    const ar = this.accuracyRelative;
    this.forcePolyCoeff = forcePoly(0.1 * ar, this.selectC);
    this.energyPolyCoeff = energyPoly(0.01 * ar, this.selectC);
    this.fourierSplitPolyCoeff = fourierPoly(0.1 * ar, this.selectC).coeffs;
    const real = spreadRealPoly(this.order, 0.1 * this.spreadingAccuracy, this.spreadingSelectC);
    this.polyOrder = Math.trunc(real.length / this.order);
    const lo = Math.trunc((1 - this.order) / 2);
    const hi = Math.trunc(this.order / 2);
    this.rhoCoeff = [];
    for (let i = 0; i < this.polyOrder; i++) {
      const row = new Array<number>(this.order).fill(0);
      for (let j = 0; j < this.order; j++) row[j] = real[i * this.order + j];
      this.rhoCoeff.push(row);
    }
    // rho_coeff[l][k] for k in [lo, hi] is stored at index k - lo
    this.drhoCoeff = [];
    for (let l = 0; l < this.polyOrder; l++) this.drhoCoeff.push(new Array<number>(this.order).fill(0));
    for (let m = lo; m <= hi; m += 1) {
      for (let l = 1; l < this.polyOrder; l++) this.drhoCoeff[l - 1][m - lo] = l * this.rhoCoeff[l][m - lo];
      this.drhoCoeff[this.polyOrder - 1][m - lo] = 0.0;
    }
    this.fourierSpreadPolyCoeff = spreadFourierPoly(0.1 * this.spreadingAccuracy, this.spreadingSelectC).coeffs;
    void ctx;
  }

  /** esp.cpp allocate: the Green's function and virial arrays of the FFT grid, and the work arrays. */
  private allocateGrid(): void {
    const N = this.nx * this.ny * this.nz;
    this.fkx = new Float64Array(this.nx);
    this.fky = new Float64Array(this.ny);
    this.fkz = new Float64Array(this.nz);
    this.greensfn = new Float64Array(N);
    this.greensfn2 = new Float64Array(N);
    this.vg = new Float64Array(6 * N);
    this.vg2 = new Float64Array(6 * N);
    this.sfPre = [0, 1, 2, 3, 4, 5].map(() => new Float64Array(N));
    this.work1 = new Float64Array(2 * N);
    this.work2 = new Float64Array(2 * N);
    this.density = new Float64Array(N);
    this.u = new Float64Array(N);
  }

  /**
   * esp.cpp setup() for an orthogonal periodic box: the k vectors, the virial coefficients and the Green's function
   * for the current box (called at init and whenever the box changes, as native does through kspace->setup()).
   */
  private setup(geom: Geometry): void {
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    this.volume2 = xprd * yprd * zprd;
    this.volume = this.volume2;
    this.delxinv = this.nx / xprd;
    this.delyinv = this.ny / yprd;
    this.delzinv = this.nz / zprd;
    this.delvolinv = this.delxinv * this.delyinv * this.delzinv;
    const unitkx = (MY_2PI / xprd);
    const unitky = (MY_2PI / yprd);
    const unitkz = (MY_2PI / zprd);
    for (let i = 0; i < this.nx; i++) {
      const per = i - this.nx * Math.trunc(2 * i / this.nx);
      this.fkx[i] = unitkx * per;
    }
    for (let i = 0; i < this.ny; i++) {
      const per = i - this.ny * Math.trunc(2 * i / this.ny);
      this.fky[i] = unitky * per;
    }
    for (let i = 0; i < this.nz; i++) {
      const per = i - this.nz * Math.trunc(2 * i / this.nz);
      this.fkz[i] = unitkz * per;
    }
    let n = 0;
    for (let k = 0; k < this.nz; k++) {
      for (let j = 0; j < this.ny; j++) {
        for (let i = 0; i < this.nx; i++) {
          const sqk = this.fkx[i] * this.fkx[i] + this.fky[j] * this.fky[j] + this.fkz[k] * this.fkz[k];
          const v = 6 * n;
          if (sqk === 0.0) {
            for (let c = 0; c < 6; c++) { this.vg[v + c] = 0.0; this.vg2[v + c] = 0.0; }
          } else {
            const vterm = -2.0 / sqk;
            this.vg[v + 0] = 1.0 + vterm * this.fkx[i] * this.fkx[i];
            this.vg[v + 1] = 1.0 + vterm * this.fky[j] * this.fky[j];
            this.vg[v + 2] = 1.0 + vterm * this.fkz[k] * this.fkz[k];
            this.vg[v + 3] = vterm * this.fkx[i] * this.fky[j];
            this.vg[v + 4] = vterm * this.fkx[i] * this.fkz[k];
            this.vg[v + 5] = vterm * this.fky[j] * this.fkz[k];
            this.vg2[v + 0] = this.fkx[i] * this.fkx[i];
            this.vg2[v + 1] = this.fky[j] * this.fky[j];
            this.vg2[v + 2] = this.fkz[k] * this.fkz[k];
            this.vg2[v + 3] = this.fkx[i] * this.fky[j];
            this.vg2[v + 4] = this.fkx[i] * this.fkz[k];
            this.vg2[v + 5] = this.fky[j] * this.fkz[k];
          }
          n++;
        }
      }
    }
    this.computeGfAd(geom);
    this.boxBuilt = [geom.lx, geom.ly, geom.lz];
  }

  /** spreading_weight2_from_t: ((order/2) poly(2t-1))^2 with poly the Fourier spreading polynomial; 0 for t > 1. */
  private spreadingWeight2FromT(t: number): number {
    if (t > 1.0) return 0.0;
    const x = 2.0 * t - 1.0;
    const appx = polyHorner(x, this.fourierSpreadPolyCoeff, this.fourierSpreadPolyCoeff.length);
    const w = 0.5 * this.order * appx;
    return w * w;
  }

  /** esp.cpp compute_gf_ad: Green's function, its virial derivative and the self-force coefficients. */
  private computeGfAd(geom: Geometry): void {
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    const unitkx = (MY_2PI / xprd);
    const unitky = (MY_2PI / yprd);
    const unitkz = (MY_2PI / zprd);
    const cutoff = this.cutoff, selectC = this.selectC;
    const sqkCut2 = (selectC / cutoff) * (selectC / cutoff);
    const scaleX = (this.order * MY_PI) / (this.nx * this.spreadingSelectC);
    const scaleY = (this.order * MY_PI) / (this.ny * this.spreadingSelectC);
    const scaleZ = (this.order * MY_PI) / (this.nz * this.spreadingSelectC * 1.0);
    this.sfCoeff.fill(0);
    const nx = this.nx, ny = this.ny, nz = this.nz;
    const kperA: number[] = [], lperA: number[] = [], mperA: number[] = [];
    const qx: number[] = [], qy: number[] = [], qz: number[] = [];
    const qx2: number[] = [], qy2: number[] = [], qz2: number[] = [];
    const wxD: number[] = [], wyD: number[] = [], wzD: number[] = [];
    for (let ii = 0; ii < nx; ++ii) {
      const k = ii;
      const kp = k - nx * Math.trunc(2 * k / nx);
      kperA[ii] = kp;
      const q = unitkx * kp;
      qx[ii] = q; qx2[ii] = q * q;
      wxD[ii] = this.spreadingWeight2FromT(scaleX * Math.abs(kp));
    }
    for (let ll = 0; ll < ny; ++ll) {
      const l = ll;
      const lp = l - ny * Math.trunc(2 * l / ny);
      lperA[ll] = lp;
      const q = unitky * lp;
      qy[ll] = q; qy2[ll] = q * q;
      wyD[ll] = this.spreadingWeight2FromT(scaleY * Math.abs(lp));
    }
    for (let mm = 0; mm < nz; ++mm) {
      const m = mm;
      const mp = m - nz * Math.trunc(2 * m / nz);
      mperA[mm] = mp;
      const q = unitkz * mp;
      qz[mm] = q; qz2[mm] = q * q;
      wzD[mm] = this.spreadingWeight2FromT(scaleZ * Math.abs(mp));
    }
    let c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0, c5 = 0.0;
    const [p1, p2, p3, p4, p5, p6] = this.sfPre;
    const g1 = this.greensfn, g2 = this.greensfn2;
    const nC = this.fourierSplitPolyCoeff.length;
    let n = 0;
    for (let mm = 0; mm < nz; ++mm) {
      const wz = wzD[mm];
      for (let ll = 0; ll < ny; ++ll) {
        const wy = wyD[ll];
        const wyz = wy * wz;
        for (let ii = 0; ii < nx; ++ii, ++n) {
          const wx = wxD[ii];
          const wxyz = wx * wyz;
          const sqk = qx2[ii] + qy2[ll] + qz2[mm];
          g1[n] = 0.0;
          g2[n] = 0.0;
          if (sqk === 0.0) continue;
          const denom = wxyz;
          if (denom === 0.0) continue;
          if (sqk > sqkCut2) continue;
          const krcOld = Math.sqrt(sqk) * cutoff / selectC;
          const x = 2.0 * krcOld - 1.0;
          const pd = polyAndDerivHorner(x, this.fourierSplitPolyCoeff, nC);
          const polyVirial = 2.0 * krcOld * pd.dp;
          const invden = 1.0 / denom;
          g1[n] = (MY_2PI * pd.p) * invden / sqk;
          const sqk2 = sqk * sqk;
          g2[n] = (MY_2PI * polyVirial) * invden / sqk2;
          const gg = g1[n];
          c0 += p1[n] * gg; c1 += p2[n] * gg;
          c2 += p3[n] * gg; c3 += p4[n] * gg;
          c4 += p5[n] * gg; c5 += p6[n] * gg;
        }
      }
    }
    this.sfCoeff[0] = c0; this.sfCoeff[1] = c1; this.sfCoeff[2] = c2;
    this.sfCoeff[3] = c3; this.sfCoeff[4] = c4; this.sfCoeff[5] = c5;
    const vol = this.volume2;
    let prex = MY_PI / vol;
    let prey = MY_PI / vol;
    let prez = MY_PI / vol;
    prex *= nx / xprd;
    prey *= ny / yprd;
    prez *= nz / zprd;
    this.sfCoeff[0] *= prex;
    this.sfCoeff[1] *= prex * 2.0;
    this.sfCoeff[2] *= prey;
    this.sfCoeff[3] *= prey * 2.0;
    this.sfCoeff[4] *= prez;
    this.sfCoeff[5] *= prez * 2.0;
  }

  /** esp.cpp compute_sf_precoeff: per-mode products of the spreading weights (5-point stencil in k). */
  private computeSfPrecoeff(): void {
    const N = this.nx * this.ny * this.nz;
    if (this.sfPre.length !== 6 || this.sfPre[0].length !== N) this.sfPre = [0, 1, 2, 3, 4, 5].map(() => new Float64Array(N));
    const coef = this.fourierSpreadPolyCoeff;
    const nco = coef.length;
    const order = this.order;
    const spr = this.spreadingSelectC;
    const weight = (q: number, n: number, dimN: number): number => {
      let ph = (0.5 * order / dimN) * (Math.abs(q) / spr);
      if (ph <= 1.00) {
        ph = 2.0 * ph - 1.0;
        let appx = coef[0];
        let r = 1.0;
        for (let ii = 1; ii < nco; ii++) {
          r *= ph;
          appx += coef[ii] * r;
        }
        return order / 2.0 * appx;
      }
      void n;
      return 0.00;
    };
    const wx0 = new Float64Array(5), wx1 = new Float64Array(5), wx2 = new Float64Array(5);
    const wy0 = new Float64Array(5), wy1 = new Float64Array(5), wy2 = new Float64Array(5);
    const wz0 = new Float64Array(5), wz1 = new Float64Array(5), wz2 = new Float64Array(5);
    const [s1, s2, s3, s4, s5, s6] = this.sfPre;
    let n = 0;
    for (let m = 0; m < this.nz; m++) {
      const mper = m - this.nz * Math.trunc(2 * m / this.nz);
      for (let l = 0; l < this.ny; l++) {
        const lper = l - this.ny * Math.trunc(2 * l / this.ny);
        for (let k = 0; k < this.nx; k++) {
          const kper = k - this.nx * Math.trunc(2 * k / this.nx);
          let sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, sum5 = 0.0, sum6 = 0.0;
          for (let i = 0; i < 5; i++) {
            const qx0 = MY_2PI * (kper + this.nx * (i - 2));
            const qx1 = MY_2PI * (kper + this.nx * (i - 1));
            const qx2 = MY_2PI * (kper + this.nx * (i));
            wx0[i] = weight(qx0, 0, this.nx);
            wx1[i] = weight(qx1, 0, this.nx);
            wx2[i] = weight(qx2, 0, this.nx);
            const qy0 = MY_2PI * (lper + this.ny * (i - 2));
            const qy1 = MY_2PI * (lper + this.ny * (i - 1));
            const qy2 = MY_2PI * (lper + this.ny * (i));
            wy0[i] = weight(qy0, 0, this.ny);
            wy1[i] = weight(qy1, 0, this.ny);
            wy2[i] = weight(qy2, 0, this.ny);
            const qz0 = MY_2PI * (mper + this.nz * (i - 2));
            const qz1 = MY_2PI * (mper + this.nz * (i - 1));
            const qz2 = MY_2PI * (mper + this.nz * (i));
            wz0[i] = weight(qz0, 0, this.nz);
            wz1[i] = weight(qz1, 0, this.nz);
            wz2[i] = weight(qz2, 0, this.nz);
          }
          for (let nx = 0; nx < 5; nx++) {
            for (let ny = 0; ny < 5; ny++) {
              for (let nz = 0; nz < 5; nz++) {
                const u0 = wx0[nx] * wy0[ny] * wz0[nz];
                const u1 = wx1[nx] * wy0[ny] * wz0[nz];
                const u2 = wx2[nx] * wy0[ny] * wz0[nz];
                const u3 = wx0[nx] * wy1[ny] * wz0[nz];
                const u4 = wx0[nx] * wy2[ny] * wz0[nz];
                const u5 = wx0[nx] * wy0[ny] * wz1[nz];
                const u6 = wx0[nx] * wy0[ny] * wz2[nz];
                sum1 += u0 * u1;
                sum2 += u0 * u2;
                sum3 += u0 * u3;
                sum4 += u0 * u4;
                sum5 += u0 * u5;
                sum6 += u0 * u6;
              }
            }
          }
          s1[n] = sum1; s2[n] = sum2; s3[n] = sum3; s4[n] = sum4; s5[n] = sum5;
          s6[n] = sum6;
          n++;
        }
      }
    }
  }

  compute(kc: KSpaceCompute): void {
    const s = kc.s;
    const geom = kc.geom;
    this.refreshChargeSums(s);
    if (this.qsqsum === 0.0) return;
    if (geom.triclinic || geom.xy !== 0 || geom.xz !== 0 || geom.yz !== 0) {
      throw new StyleError('kspace_style esp: triclinic boxes are not ported');
    }
    if (geom.lx !== this.prdInit[0] || geom.ly !== this.prdInit[1] || geom.lz !== this.prdInit[2]) {
      // native: fix npt and fix deform call kspace->setup(), which recomputes the box-dependent coefficients
      this.prdInit = [geom.lx, geom.ly, geom.lz];
      this.setup(geom);
    }
    if (this.boxBuilt === null) this.setup(geom);
    const n = s.n;
    const N = this.nx * this.ny * this.nz;
    const qqrd2e = kc.qqrd2e;
    const qscale = qqrd2e * 1.0;
    const bl = geom.lo;
    // particle_map
    this.part2grid = new Int32Array(3 * n);
    const delinv = [this.delxinv, this.delyinv, this.delzinv];
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        this.part2grid[3 * i + d] = Math.trunc((s.x[3 * i + d] - bl[d]) * delinv[d] + this.shift) - OFFSET;
      }
    }
    // make_rho (periodic brick: the stencil index wraps onto the grid)
    this.density.fill(0);
    const nlow = this.nlower, nup = this.nupper;
    const rho1d = [new Float64Array(this.order), new Float64Array(this.order), new Float64Array(this.order)];
    const d1 = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const nx0 = this.part2grid[3 * i], ny0 = this.part2grid[3 * i + 1], nz0 = this.part2grid[3 * i + 2];
      for (let d = 0; d < 3; d++) {
        const idx = [nx0, ny0, nz0][d];
        d1[d] = idx + this.shiftone - (s.x[3 * i + d] - bl[d]) * delinv[d];
      }
      this.rho1dAll(d1[0], d1[1], d1[2], rho1d);
      const z0 = this.delvolinv * s.q[i];
      for (let nn = nlow; nn <= nup; nn++) {
        const mz = wrap(nn + nz0, this.nz);
        const y0 = z0 * rho1d[2][nn - nlow];
        for (let m = nlow; m <= nup; m++) {
          const my = wrap(m + ny0, this.ny);
          const x0 = y0 * rho1d[1][m - nlow];
          for (let l = nlow; l <= nup; l++) {
            const mx = wrap(l + nx0, this.nx);
            this.density[(mz * this.ny + my) * this.nx + mx] += x0 * rho1d[0][l - nlow];
          }
        }
      }
    }
    // poisson_ad
    const w1 = this.work1;
    for (let i = 0; i < N; i++) { w1[2 * i] = this.density[i]; w1[2 * i + 1] = 0.0; }
    fft3d(w1, this.nx, this.ny, this.nz, -1);
    const ngridtotal = this.nx * this.ny * this.nz;
    const scaleinv = 1.0 / ngridtotal;
    const s2 = scaleinv * scaleinv;
    let energy = 0.0;
    const virial = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < N; i++) {
      const re = w1[2 * i], im = w1[2 * i + 1];
      const eng = s2 * this.greensfn[i] * (re * re + im * im);
      const engVirial = s2 * this.greensfn2[i] * (re * re + im * im);
      for (let j = 0; j < 6; j++) {
        virial[j] += eng * this.vg[6 * i + j];
        virial[j] += engVirial * this.vg2[6 * i + j];
      }
      energy += eng;
    }
    for (let i = 0; i < N; i++) {
      w1[2 * i] *= scaleinv * this.greensfn[i];
      w1[2 * i + 1] *= scaleinv * this.greensfn[i];
    }
    // poisson_peratom: the six virial meshes (vg only); the potential mesh u below serves the energy (diff ad)
    const vmesh: Float64Array[] = [];
    if (kc.vatom) {
      for (let j = 0; j < 6; j++) {
        const wv = new Float64Array(2 * N);
        for (let i = 0; i < N; i++) {
          wv[2 * i] = w1[2 * i] * this.vg[6 * i + j];
          wv[2 * i + 1] = w1[2 * i + 1] * this.vg[6 * i + j];
        }
        fft3d(wv, this.nx, this.ny, this.nz, 1);
        const v = new Float64Array(N);
        for (let i = 0; i < N; i++) v[i] = wv[2 * i];
        vmesh.push(v);
      }
    }
    const w2 = this.work2;
    w2.set(w1.subarray(0, 2 * N));
    fft3d(w2, this.nx, this.ny, this.nz, 1);
    for (let i = 0; i < N; i++) this.u[i] = w2[2 * i];
    // fieldforce_ad
    const hxInv = this.nx / geom.lx;
    const hyInv = this.ny / geom.ly;
    const hzInv = this.nz / geom.lz;
    const q = s.q;
    const sfc = this.sfCoeff;
    const rho1dDr = [new Float64Array(this.order), new Float64Array(this.order), new Float64Array(this.order)];
    const drho1dDr = [new Float64Array(this.order), new Float64Array(this.order), new Float64Array(this.order)];
    for (let i = 0; i < n; i++) {
      const nx0 = this.part2grid[3 * i], ny0 = this.part2grid[3 * i + 1], nz0 = this.part2grid[3 * i + 2];
      for (let d = 0; d < 3; d++) {
        const idx = [nx0, ny0, nz0][d];
        d1[d] = idx + this.shiftone - (s.x[3 * i + d] - bl[d]) * delinv[d];
      }
      this.rho1dAll(d1[0], d1[1], d1[2], rho1dDr);
      this.drho1dAll(d1[0], d1[1], d1[2], drho1dDr);
      let ekx = 0.0, eky = 0.0, ekz = 0.0;
      for (let nn = nlow; nn <= nup; nn++) {
        const mz = wrap(nn + nz0, this.nz);
        for (let m = nlow; m <= nup; m++) {
          const my = wrap(m + ny0, this.ny);
          for (let l = nlow; l <= nup; l++) {
            const mx = wrap(l + nx0, this.nx);
            const uu = this.u[(mz * this.ny + my) * this.nx + mx];
            const a = l - nlow, b = m - nlow, c = nn - nlow;
            ekx += drho1dDr[0][a] * rho1dDr[1][b] * rho1dDr[2][c] * uu;
            eky += rho1dDr[0][a] * drho1dDr[1][b] * rho1dDr[2][c] * uu;
            ekz += rho1dDr[0][a] * rho1dDr[1][b] * drho1dDr[2][c] * uu;
          }
        }
      }
      ekx *= hxInv;
      eky *= hyInv;
      ekz *= hzInv;
      const qfactor = qqrd2e * 1.0;
      const s1 = s.x[3 * i] * hxInv;
      const s2y = s.x[3 * i + 1] * hyInv;
      const s3 = s.x[3 * i + 2] * hzInv;
      let sf = sfc[0] * libmSin(2 * MY_PI * s1);
      sf += sfc[1] * libmSin(4 * MY_PI * s1);
      sf *= 2 * q[i] * q[i];
      kc.f[3 * i] += qfactor * (ekx * q[i] - sf);
      sf = sfc[2] * libmSin(2 * MY_PI * s2y);
      sf += sfc[3] * libmSin(4 * MY_PI * s2y);
      sf *= 2 * q[i] * q[i];
      kc.f[3 * i + 1] += qfactor * (eky * q[i] - sf);
      sf = sfc[4] * libmSin(2 * MY_PI * s3);
      sf += sfc[5] * libmSin(4 * MY_PI * s3);
      sf *= 2 * q[i] * q[i];
      kc.f[3 * i + 2] += qfactor * (ekz * q[i] - sf);
    }
    // energy and virial (compute)
    let selfCoeff = 0.00;
    for (let i = 1; i < this.energyPolyCoeff.length; i++) {
      selfCoeff += 2 * (i + 0.00) * this.energyPolyCoeff[i] * (i % 2 === 1 ? 1.0 : -1.0);
    }
    if (kc.eatom || kc.vatom) {
      // fieldforce_peratom: u and v0..v5 at each charge, times q_i; then the halving, the self term and qscale
      const eatom = kc.eatom, vatom = kc.vatom;
      const vv = [0, 0, 0, 0, 0, 0];
      for (let i = 0; i < n; i++) {
        const nx0 = this.part2grid[3 * i], ny0 = this.part2grid[3 * i + 1], nz0 = this.part2grid[3 * i + 2];
        for (let d = 0; d < 3; d++) {
          const idx = [nx0, ny0, nz0][d];
          d1[d] = idx + this.shiftone - (s.x[3 * i + d] - bl[d]) * delinv[d];
        }
        this.rho1dAll(d1[0], d1[1], d1[2], rho1d);
        let u = 0.0;
        vv.fill(0);
        for (let nn = nlow; nn <= nup; nn++) {
          const mz = wrap(nn + nz0, this.nz);
          const z0 = rho1d[2][nn - nlow];
          for (let m = nlow; m <= nup; m++) {
            const my = wrap(m + ny0, this.ny);
            const y0 = z0 * rho1d[1][m - nlow];
            for (let l = nlow; l <= nup; l++) {
              const mx = wrap(l + nx0, this.nx);
              const x0 = y0 * rho1d[0][l - nlow];
              const g = (mz * this.ny + my) * this.nx + mx;
              if (eatom) u += x0 * this.u[g];
              if (vatom) for (let j = 0; j < 6; j++) vv[j] += x0 * vmesh[j][g];
            }
          }
        }
        if (eatom) eatom[i] += qscale * (0.5 * q[i] * u - (-selfCoeff / this.cutoff) * q[i] * q[i] / 2.0);
        if (vatom) for (let j = 0; j < 6; j++) vatom[6 * i + j] += 0.5 * qscale * q[i] * vv[j];
      }
    }
    energy *= 0.5 * this.volume2;
    energy -= (-selfCoeff / this.cutoff) * this.qsqsum / 2.0;
    energy *= qscale;
    kc.acc.elong += energy;
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += 0.5 * qscale * this.volume2 * virial[c];
  }

  /** compute_rho1d on the three directions: rho1d[d][l - nlower] for the spreading points of the stencil. */
  private rho1dAll(dx: number, dy: number, dz: number, out: Float64Array[]): void {
    const dd = [dx, dy, dz];
    for (let d = 0; d < 3; d++) {
      for (let k = this.nlower; k <= this.nupper; k++) {
        let r = 0;
        for (let l = this.polyOrder - 1; l >= 0; l--) {
          r = this.rhoCoeff[l][k - Math.trunc((1 - this.order) / 2)] + r * dd[d];
        }
        out[d][k - this.nlower] = r;
      }
    }
  }

  /** compute_drho1d on the three directions. */
  private drho1dAll(dx: number, dy: number, dz: number, out: Float64Array[]): void {
    const dd = [dx, dy, dz];
    for (let d = 0; d < 3; d++) {
      for (let k = this.nlower; k <= this.nupper; k++) {
        let r = 0;
        for (let l = this.polyOrder - 2; l >= 0; l--) {
          r = this.drhoCoeff[l][k - Math.trunc((1 - this.order) / 2)] + r * dd[d];
        }
        out[d][k - this.nlower] = r;
      }
    }
  }
}

/** esp.cpp factorable: all factors in {2, 3, 5}. */
const factorable = (n0: number): boolean => {
  let n = n0;
  const factors = [2, 3, 5];
  while (n > 1) {
    let i = 0;
    for (; i < factors.length; i++) {
      if (n % factors[i] === 0) { n = Math.trunc(n / factors[i]); break; }
    }
    if (i === factors.length) return false;
  }
  return true;
};

/** C printf %.8g-style number (used in the log lines only). */
const fmtG = (x: number): string => {
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e < -5 || e >= 8) return x.toExponential(7).replace(/\.?0+e/, 'e').replace(/e([+-])(\d)$/, 'e$10$2');
  return String(Number(x.toPrecision(8)));
};

