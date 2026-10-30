/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/pppm_stagger.cpp and
 * src/KSPACE/pppm_stagger.h (Contributing author: Stan Moore, SNL), with the base PPPM steps it calls from
 * src/KSPACE/pppm.cpp at the same commit (init, set_grid_global, compute_df_kspace, compute_qopt, adjust_gewald,
 * newton_raphson_f, derivf, compute_gf_denom, set_grid_local, the poisson_ik energy and virial). Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * kspace_style pppm/stagger: PPPM on two meshes, the second shifted by half a mesh cell (stagger 0.0 and 0.5 in
 * pppm_stagger.cpp compute()). Each pass assigns the charges with the stagger offset, solves Poisson's equation with the
 * staggered influence function (compute_gf_ik, denominator 0.5 * (gf_denom + gf_denom2)), and interpolates the field
 * back. Energy, virial and forces are averaged over the nstagger = 2 passes.
 *
 * Not ported: kspace_modify diff ad (the stagger analytic-differentiation solver, compute_gf_ad and fieldforce_ad),
 * per-atom energy and virial (fieldforce_peratom), and triclinic boxes (native refuses them too). They are refused by
 * name. The mesh and G-ewald choice is the stagger-specific one of set_grid_global: the grid is searched from h = 4/G
 * shrinking by 0.95 until compute_df_kspace (compute_qopt) meets the accuracy, then G is solved by adjust_gewald.
 */
import { KSpaceBase, invertUpper } from '../../engine/force/kspace/ewald';
import { fft3d, isFFTSize } from '../../engine/force/fft';
import { parseNum } from '../../engine/force/util';
import { StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { gfDenomCoefficients } from '../pppm/grid';
import { exp as libmExp } from '../libm/exp';
import { log as libmLog } from '../libm/log';
import { pow as libmPow } from '../libm/pow';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { sinFma } from '../libm/sincos_fma';

/** pppm.cpp SMALL and LARGE (adjust_gewald). */
const SMALL = 0.00001;
const LARGE = 10000;
/** pppm_stagger.cpp EPS_HOC: the image count of compute_gf_ik. */
const EPS_HOC = 1.0e-7;
/** Number of staggered meshes (pppm_stagger.cpp compute, nstagger = 2). */
const NSTAGGER = 2;
/** MY_4PI and MY_2PI of math_const.h (the double literals). */
const MY_4PI = 12.56637061435917295384;
const MY_2PI = 6.28318530717958647692;
/** pppm.cpp qopt images in compute_qopt: nbx = nby = nbz = 2. */
const QOPT_IMAGES = 2;

/** pppm_stagger.cpp gf_b2[order][l], rows 1..7 (the constructor's values, same expressions). */
const GF_B2: number[][] = [
  [],
  [1.0],
  [5.0 / 6.0, 1.0 / 6.0],
  [61.0 / 120.0, 29.0 / 60.0, 1.0 / 120.0],
  [277.0 / 1008.0, 1037.0 / 1680.0, 181.0 / 1680.00, 1.0 / 5040.0],
  [50521.0 / 362880.0, 7367.0 / 12960.0, 16861.0 / 60480.0, 1229.0 / 90720.0, 1.0 / 362880.0],
  [540553.0 / 7983360.0, 17460701.0 / 39916800.0, 8444893.0 / 19958400.0, 1409633.0 / 19958400.0, 44281.0 / 39916800.0, 1.0 / 39916800.0],
  [199360981.0 / 6227020800.0, 103867703.0 / 345945600.0, 66714163.0 / 138378240.0, 54085121.0 / 311351040.0, 1640063.0 / 138378240.0, 671.0 / 10483200.0, 1.0 / 6227020800.0],
];

/** math_special.cpp powsinxx: (sin x / x)^n by binary powering, 1 at x = 0. */
const powsinxx = (x: number, n: number): number => {
  if (x === 0.0) return 1.0;
  let yy = sinFma(x) / x;
  let ww = 1.0;
  for (; n !== 0; n >>= 1, yy *= yy) if (n & 1) ww *= yy;
  return ww;
};

/** One axis of the mesh: per grid index k the integer wave number kper and the sine/cosine terms, and per image n
 *  (index n + R) the shifted wave number q, its Gaussian s and the window w. */
interface AxisTable {
  R: number;
  kper: Int32Array;
  sn2: Float64Array;
  cs: Float64Array;
  q: Float64Array;
  s: Float64Array;
  w: Float64Array;
}

/** The per-axis values of compute_gf_ik / compute_qopt for a mesh of N points on an edge of length L. */
const axisTable = (N: number, L: number, unit: number, g: number, twoorder: number, R: number): AxisTable => {
  const W = 2 * R + 1;
  const t: AxisTable = {
    R, kper: new Int32Array(N), sn2: new Float64Array(N), cs: new Float64Array(N),
    q: new Float64Array(N * W), s: new Float64Array(N * W), w: new Float64Array(N * W),
  };
  for (let k = 0; k < N; k++) {
    // pppm_stagger.cpp: kper = k - nx_pppm*(2*k/nx_pppm), with integer division
    const kper = k - N * Math.trunc((2 * k) / N);
    t.kper[k] = kper;
    const arg0 = 0.5 * unit * kper * L / N;
    const sn = libmSin(arg0);
    t.sn2[k] = sn * sn;
    t.cs[k] = libmCos(arg0);
    for (let n = -R; n <= R; n++) {
      const q = unit * (kper + N * n);
      const idx = k * W + (n + R);
      t.q[idx] = q;
      t.s[idx] = libmExp(-0.25 * ((q / g) * (q / g)));
      t.w[idx] = powsinxx((0.5 * q * L) / N, twoorder);
    }
  }
  return t;
};

/** The base PPPM Green's denominator (gf_denom), with the squared sin^2 terms of the stagger solver. */
const gfDenom = (gb: number[], order: number, x: number, y: number, z: number): number => {
  let sx = 0, sy = 0, sz = 0;
  for (let l = order - 1; l >= 0; l--) {
    sx = gb[l] + sx * x;
    sy = gb[l] + sy * y;
    sz = gb[l] + sz * z;
  }
  const s = sx * sy * sz;
  return s * s;
};

/** pppm_stagger.h gf_denom2 (the cos terms of the staggered mesh). */
const gfDenom2 = (order: number, x: number, y: number, z: number): number => {
  let sx = 0, sy = 0, sz = 0;
  const x2 = x * x, y2 = y * y, z2 = z * z;
  let xl = x, yl = y, zl = z;
  const b2 = GF_B2[order];
  for (let l = 0; l < order; l++) {
    sx += b2[l] * xl;
    sy += b2[l] * yl;
    sz += b2[l] * zl;
    xl *= x2;
    yl *= y2;
    zl *= z2;
  }
  const s = sx * sy * sz;
  return s * s;
};

/*
 * The class derives from KSpaceBase, not from the core KSpacePPPM: kspace_modify diff ad promotes every KSpacePPPM
 * instance to the core analytic-differentiation solver (src/engine/commands/forcefield.ts), which would silently replace
 * the stagger solver. The mesh, order and diff keywords are read here as KSpacePPPM.modify reads them.
 */
export class KSpacePPPMStagger extends KSpaceBase {
  override readonly name: string = 'pppm/stagger';
  // PPPMStagger::init calls PPPM::init, whose messages name PPPM
  protected override readonly kspaceLabel: string = 'PPPM';
  /** kspace_modify order (pppm: 2 to 7) and mesh (null when automatic). */
  order = 5;
  meshUser: [number, number, number] | null = null;
  protected mesh: [number, number, number] = [0, 0, 0];
  /** gf_b of PPPM::compute_gf_denom for the current order. */
  private gfB: number[] = [];
  private staggerGreen = new Float64Array(0);
  private staggerKey = '';

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'mesh': {
        const m = [0, 1, 2].map((d) => parseNum(values[d], 'mesh'));
        if (m.every((v) => v === 0)) { this.meshUser = null; return 3; }
        this.meshUser = m as [number, number, number];
        return 3;
      }
      case 'order': {
        const o = parseNum(values[0], 'order');
        if (!Number.isInteger(o) || o < 2 || o > 7) throw new StyleError('kspace_modify order for pppm must be an integer from 2 to 7');
        this.order = o;
        return 1;
      }
      case 'diff':
        if (values[0] === 'ad') {
          throw new StyleError('kspace_modify diff ad is not supported by kspace_style pppm/stagger (the staggered analytic-differentiation solver of pppm_stagger.cpp is not ported)');
        }
        if (values[0] !== 'ik') throw new StyleError(`kspace_modify diff ${values[0]} is not a diff option (ik or ad)`);
        return 1;
      default: return super.modify(key, values);
    }
  }

  /**
   * PPPMStagger::init and PPPM::init (set_grid_global, adjust_gewald). The stagger grid loop starts at h = 4/G and
   * shrinks it by 0.95 until compute_df_kspace meets the accuracy; the mesh is then made factorable.
   */
  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    // pppm_stagger.cpp PPPMStagger::init
    if (s.box.triclinic) throw new StyleError('Cannot (yet) use kspace_style pppm/stagger with triclinic systems');
    if (this.minorder > this.order) throw new StyleError(`PPPM order < minimum allowed order (kspace_modify minorder ${this.minorder} > order ${this.order})`);
    this.cutCoul = cutCoul;
    this.setupCharges(s, geom, ctx);
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    const zprdSlab = zprd * this.slab;
    const natoms = this.natoms === 0 ? 1 : this.natoms;
    const q2 = qqrd2e * this.qsqsum;
    const accuracy = this.absAccuracy(s, qqrd2e);
    const gewaldFlag = this.gewaldUser > 0;
    if (accuracy <= 0.0) throw new StyleError('KSpace accuracy must be > 0');
    let g: number;
    if (!gewaldFlag) {
      if (q2 === 0.0) throw new StyleError("Must use 'kspace_modify gewald' for uncharged system");
      // PPPM::init: the initial G from the accuracy and the cutoff
      g = accuracy * Math.sqrt(natoms * this.cutCoul * xprd * yprd * zprd) / (2.0 * q2);
      if (g >= 1.0) g = (1.35 - 0.15 * libmLog(accuracy)) / this.cutCoul;
      else g = Math.sqrt(-libmLog(g)) / this.cutCoul;
    } else g = this.gewaldUser;
    // compute_gf_denom runs before set_grid_global for stagger (PPPM::init)
    this.gfB = gfDenomCoefficients(this.order);
    const order = this.order;
    const dfKspace = (gg: number, nx: number, ny: number, nz: number): number => {
      // compute_df_kspace: sqrt(qopt/natoms) q2 / (xprd yprd zprd_slab)
      const qopt = this.computeQopt(gg, nx, ny, nz, xprd, yprd, zprdSlab, order);
      return Math.sqrt(qopt / natoms) * q2 / (xprd * yprd * zprdSlab);
    };
    let nx = 0, ny = 0, nz = 0;
    if (this.meshUser) {
      [nx, ny, nz] = this.meshUser;
    } else {
      // set_grid_global, stagger branch
      let h = (4.0 / g);
      let count = 0;
      for (;;) {
        nx = Math.trunc(xprd / h);
        ny = Math.trunc(yprd / h);
        nz = Math.trunc(zprdSlab / h);
        if (nx <= 1) nx = 2;
        if (ny <= 1) ny = 2;
        if (nz <= 1) nz = 2;
        const df = dfKspace(g, nx, ny, nz);
        count++;
        if (df <= accuracy) break;
        if (count > 500) throw new StyleError('Could not compute grid size');
        h *= 0.95;
      }
    }
    // boost grid size until it is factorable (set_grid_global, also with a user mesh)
    while (!isFactorable(nx)) nx++;
    while (!isFactorable(ny)) ny++;
    while (!isFactorable(nz)) nz++;
    if (!gewaldFlag) {
      // adjust_gewald: Newton-Raphson on f(G) = df_rspace - df_kspace with the forward-difference derivative
      const fRes = (gg: number) => {
        const dfRspace = 2.0 * q2 * libmExp(-gg * gg * this.cutCoul * this.cutCoul) / Math.sqrt(natoms * this.cutCoul * xprd * yprd * zprd);
        return dfRspace - dfKspace(gg, nx, ny, nz);
      };
      let ok = false;
      for (let i = 0; i < LARGE; i++) {
        const f1 = fRes(g);
        const dfx = (fRes(g + 0.000001) - f1) / 0.000001;
        if (dfx === 0.0 || dfx !== dfx) break;
        let dx = fRes(g) / dfx;
        while (g - dx <= 0.0) dx *= 0.5;
        g -= dx;
        if (Math.abs(fRes(g)) < SMALL) { ok = true; break; }
      }
      if (!ok) throw new StyleError('Could not compute g_ewald');
    }
    this.gEwald = g;
    this.mesh = [nx, ny, nz];
    this.staggerKey = '';
    // pppm.cpp init prints G with {:.8g}
    ctx.log(`  G vector (1/distance) = ${Number(g.toPrecision(8))}\n  grid = ${this.mesh.join(' ')}\n  stencil order = ${this.order}`);
  }

  /**
   * PPPM::compute_qopt (ik, stagger): the optimal-influence error sum over the mesh, with images |n| <= 2. Per grid
   * point and image the terms are those of pppm_stagger.cpp compute_qopt; the axis values are tabulated once.
   */
  private computeQopt(g: number, nx: number, ny: number, nz: number, xprd: number, yprd: number, zprdSlab: number, order: number): number {
    const unitx = MY_2PI / xprd, unity = MY_2PI / yprd, unitz = MY_2PI / zprdSlab;
    const twoorder = 2 * order;
    const R = QOPT_IMAGES;
    const tx = axisTable(nx, xprd, unitx, g, twoorder, R);
    const ty = axisTable(ny, yprd, unity, g, twoorder, R);
    const tz = axisTable(nz, zprdSlab, unitz, g, twoorder, R);
    const W = 2 * R + 1;
    const gb = this.gfB;
    let qopt = 0.0;
    for (let m = 0; m < nz; m++) {
      const mper = tz.kper[m];
      for (let l = 0; l < ny; l++) {
        const lper = ty.kper[l];
        for (let k = 0; k < nx; k++) {
          const kper = tx.kper[k];
          const sqk = (unitx * kper) * (unitx * kper) + (unity * lper) * (unity * lper) + (unitz * mper) * (unitz * mper);
          if (sqk === 0.0) continue;
          const numerator = MY_4PI / sqk;
          const denominator = 0.5 * (gfDenom(gb, order, tx.sn2[k], ty.sn2[l], tz.sn2[m]) + gfDenom2(order, tx.cs[k], ty.cs[l], tz.cs[m]));
          let sum1 = 0.0, sum2 = 0.0;
          for (let nxi = -R; nxi <= R; nxi++) {
            const ix = k * W + (nxi + R);
            const qx = tx.q[ix], sx = tx.s[ix], wx = tx.w[ix];
            for (let nyi = -R; nyi <= R; nyi++) {
              const iy = l * W + (nyi + R);
              const qy = ty.q[iy], sy = ty.s[iy], wy = ty.w[iy];
              for (let nzi = -R; nzi <= R; nzi++) {
                const iz = m * W + (nzi + R);
                const qz = tz.q[iz], sz = tz.s[iz], wz = tz.w[iz];
                const dot1 = unitx * kper * qx + unity * lper * qy + unitz * mper * qz;
                const dot2 = qx * qx + qy * qy + qz * qz;
                const u1 = sx * sy * sz;
                const u2 = wx * wy * wz;
                const u3 = numerator * u1 * u2 * dot1;
                sum1 += u1 * u1 * MY_4PI * MY_4PI / dot2;
                sum2 += u3 * u3 / dot2;
              }
            }
          }
          qopt += sum1 - sum2 / denominator;
        }
      }
    }
    return qopt;
  }

  /**
   * PPPMStagger::compute_gf_ik: the staggered influence function on the mesh (flattened with k fastest, the solver's
   * index order). Images |n| <= nbx, nby, nbz from the EPS_HOC rule of pppm_stagger.cpp.
   */
  private buildStaggerGreen(geom: Geometry): Float64Array {
    const [nx, ny, nz] = this.mesh;
    const g = this.gEwald;
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    const zprdSlab = zprd * this.slab;
    const key = [xprd, yprd, zprdSlab, nx, ny, nz, this.order, g].join(',');
    if (key === this.staggerKey) return this.staggerGreen;
    const unitx = MY_2PI / xprd, unity = MY_2PI / yprd, unitz = MY_2PI / zprdSlab;
    const order = this.order;
    const twoorder = 2 * order;
    const fac = libmPow(-libmLog(EPS_HOC), 0.25);
    const nbx = Math.trunc((g * xprd / (Math.PI * nx)) * fac);
    const nby = Math.trunc((g * yprd / (Math.PI * ny)) * fac);
    const nbz = Math.trunc((g * zprdSlab / (Math.PI * nz)) * fac);
    const tx = axisTable(nx, xprd, unitx, g, twoorder, nbx);
    const ty = axisTable(ny, yprd, unity, g, twoorder, nby);
    const tz = axisTable(nz, zprdSlab, unitz, g, twoorder, nbz);
    const Wx = 2 * nbx + 1, Wy = 2 * nby + 1, Wz = 2 * nbz + 1;
    const gb = this.gfB;
    const green = new Float64Array(nx * ny * nz);
    let n = 0;
    for (let m = 0; m < nz; m++) {
      const mper = tz.kper[m];
      for (let l = 0; l < ny; l++) {
        const lper = ty.kper[l];
        for (let k = 0; k < nx; k++) {
          const kper = tx.kper[k];
          const sqk = (unitx * kper) * (unitx * kper) + (unity * lper) * (unity * lper) + (unitz * mper) * (unitz * mper);
          if (sqk !== 0.0) {
            const numerator = MY_4PI / sqk;
            const denominator = 0.5 * (gfDenom(gb, order, tx.sn2[k], ty.sn2[l], tz.sn2[m]) + gfDenom2(order, tx.cs[k], ty.cs[l], tz.cs[m]));
            let sum1 = 0.0;
            for (let nxi = -nbx; nxi <= nbx; nxi++) {
              const ix = k * Wx + (nxi + nbx);
              const qx = tx.q[ix], sx = tx.s[ix], wx = tx.w[ix];
              for (let nyi = -nby; nyi <= nby; nyi++) {
                const iy = l * Wy + (nyi + nby);
                const qy = ty.q[iy], sy = ty.s[iy], wy = ty.w[iy];
                for (let nzi = -nbz; nzi <= nbz; nzi++) {
                  const iz = m * Wz + (nzi + nbz);
                  const qz = tz.q[iz], sz = tz.s[iz], wz = tz.w[iz];
                  const dot1 = unitx * kper * qx + unity * lper * qy + unitz * mper * qz;
                  const dot2 = qx * qx + qy * qy + qz * qz;
                  sum1 += (dot1 / dot2) * sx * sy * sz * wx * wy * wz;
                }
              }
            }
            green[n++] = numerator * sum1 / denominator;
          } else green[n++] = 0.0;
        }
      }
    }
    this.staggerGreen = green;
    this.staggerKey = key;
    return green;
  }

  /**
   * PPPMStagger::particle_map, make_rho and the stencil of fieldforce_ik for one mesh: the grid index of the first stencil
   * point and the P weights per dimension, with the stagger offset added to the scaled coordinate.
   */
  private assignStagger(s: SimState, geom: Geometry, stagger: number): { first: Int32Array; wts: Float64Array } {
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    const n = s.n;
    const first = new Int32Array(3 * n);
    const wts = new Float64Array(3 * n * P);
    const lam = [0, 0, 0];
    const N = [nx, ny, nz];
    for (let i = 0; i < n; i++) {
      geom.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      lam[2] /= this.slab;
      for (let d = 0; d < 3; d++) {
        const u = lam[d] * N[d] + stagger;
        const i0 = P % 2 === 1 ? Math.round(u) - (P - 1) / 2 : Math.floor(u) - P / 2 + 1;
        first[3 * i + d] = i0;
        for (let k = 0; k < P; k++) wts[(3 * i + d) * P + k] = bspline(P, u - (i0 + k) + P / 2);
      }
    }
    return { first, wts };
  }

  compute(kc: KSpaceCompute): void {
    if (kc.eatom || kc.vatom) {
      throw new StyleError('compute pe/atom and stress/atom with kspace_style pppm/stagger are not supported by the browser engine (per-atom energy and virial of pppm/stagger are not ported)');
    }
    const s = kc.s;
    this.refreshChargeSums(s);
    const geom = kc.geom;
    const n = s.n;
    const C = kc.qqrd2e;
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    this.volume = geom.volume(3) * this.slab;
    const V = this.volume;
    const green = this.buildStaggerGreen(geom);
    const ng = nx * ny * nz;
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, geom.lz * this.slab);
    const mod = (a: number, m: number) => ((a % m) + m) % m;
    const g = this.gEwald;
    let e = 0;
    const w = [0, 0, 0, 0, 0, 0];
    for (let pass = 0; pass < NSTAGGER; pass++) {
      // stagger 0.0 on the first mesh, then += 1/nstagger (0.5) on the second
      const stagger = pass * (1.0 / NSTAGGER);
      const { first, wts } = this.assignStagger(s, geom, stagger);
      const rho = new Float64Array(2 * ng);
      for (let i = 0; i < n; i++) {
        const q = s.q[i];
        if (q === 0) continue;
        const fx = first[3 * i], fy = first[3 * i + 1], fz = first[3 * i + 2];
        for (let c = 0; c < P; c++) {
          const iz = mod(fz + c, nz);
          const wz = q * wts[(3 * i + 2) * P + c];
          for (let b = 0; b < P; b++) {
            const iy = mod(fy + b, ny);
            const wyz = wz * wts[(3 * i + 1) * P + b];
            for (let a = 0; a < P; a++) {
              const ix = mod(fx + a, nx);
              rho[2 * ((iz * ny + iy) * nx + ix)] += wyz * wts[(3 * i) * P + a];
            }
          }
        }
      }
      fft3d(rho, nx, ny, nz, -1);
      // poisson_ik: energy, virial and the three field components in k-space
      const ex = new Float64Array(2 * ng), ey = new Float64Array(2 * ng), ez = new Float64Array(2 * ng);
      for (let iz = 0; iz < nz; iz++) {
        const mz = iz < nz / 2 ? iz : iz - nz;
        for (let iy = 0; iy < ny; iy++) {
          const my = iy < ny / 2 ? iy : iy - ny;
          for (let ix = 0; ix < nx; ix++) {
            const mx = ix < nx / 2 ? ix : ix - nx;
            const idx = (iz * ny + iy) * nx + ix;
            const G = green[idx];
            if (G === 0) continue;
            const re = rho[2 * idx], im = rho[2 * idx + 1];
            const r2 = re * re + im * im;
            const kx = 2 * Math.PI * (mx * inv[0][0] + my * inv[1][0] + mz * inv[2][0]);
            const ky = 2 * Math.PI * (mx * inv[0][1] + my * inv[1][1] + mz * inv[2][1]);
            const kz = 2 * Math.PI * (mx * inv[0][2] + my * inv[1][2] + mz * inv[2][2]);
            const k2 = kx * kx + ky * ky + kz * kz;
            e += G * r2;
            const bb = 2 * (1 / k2 + 1 / (4 * g * g));
            const t = G * r2;
            w[0] += t * (1 - bb * kx * kx); w[1] += t * (1 - bb * ky * ky); w[2] += t * (1 - bb * kz * kz);
            w[3] += t * (-bb * kx * ky); w[4] += t * (-bb * kx * kz); w[5] += t * (-bb * ky * kz);
            ex[2 * idx] = kx * G * im; ex[2 * idx + 1] = -kx * G * re;
            ey[2 * idx] = ky * G * im; ey[2 * idx + 1] = -ky * G * re;
            ez[2 * idx] = kz * G * im; ez[2 * idx + 1] = -kz * G * re;
          }
        }
      }
      fft3d(ex, nx, ny, nz, 1);
      fft3d(ey, nx, ny, nz, 1);
      fft3d(ez, nx, ny, nz, 1);
      // fieldforce_ik: interpolate the field with the same stencil and weights; qfactor carries 1/nstagger
      for (let i = 0; i < n; i++) {
        const q = s.q[i];
        if (q === 0) continue;
        const fx0 = first[3 * i], fy0 = first[3 * i + 1], fz0 = first[3 * i + 2];
        let Ex = 0, Ey = 0, Ez = 0;
        for (let c = 0; c < P; c++) {
          const iz = mod(fz0 + c, nz);
          const wz = wts[(3 * i + 2) * P + c];
          for (let b = 0; b < P; b++) {
            const iy = mod(fy0 + b, ny);
            const wyz = wz * wts[(3 * i + 1) * P + b];
            for (let a = 0; a < P; a++) {
              const ix = mod(fx0 + a, nx);
              const ww = wyz * wts[(3 * i) * P + a];
              const idx = 2 * ((iz * ny + iy) * nx + ix);
              Ex += ww * ex[idx]; Ey += ww * ey[idx]; Ez += ww * ez[idx];
            }
          }
        }
        const pre = (C * q) / V / NSTAGGER;
        kc.f[3 * i] += pre * Ex; kc.f[3 * i + 1] += pre * Ey;
        if (this.zForce) kc.f[3 * i + 2] += pre * Ez;
      }
    }
    // energy and virial: 0.5 V / nstagger times the sum over the meshes, then the self and background terms
    const pref = C / (2 * V);
    kc.acc.elong += pref * e / NSTAGGER + this.constantEnergy(C);
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += pref * w[c] / NSTAGGER;
    this.slabCorrection(kc);
  }
}

/** PPPM::factorable: every prime factor is 2, 3 or 5. */
const isFactorable = (n: number): boolean => {
  if (n < 1) return false;
  for (const p of [2, 3, 5]) while (n % p === 0) n /= p;
  return n === 1;
};

/** Cardinal B-spline M_p(x), support [0, p] (the charge assignment function, as pppm.ts). */
const bspline = (p: number, x: number): number => {
  if (p === 1) return x >= 0 && x < 1 ? 1 : 0;
  if (x <= 0 || x >= p) return 0;
  return (x * bspline(p - 1, x) + (p - x) * bspline(p - 1, x - 1)) / (p - 1);
};

