/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ELECTRODE/pppm_electrode.cpp and pppm_electrode.h at commit b332f28 (the vector and matrix routines,
 * compute_vector, project_psi, compute_matrix, and the boundary-correction calls), on top of the core PPPM it extends
 * (src/KSPACE/pppm.cpp, here src/engine/force/kspace/pppm.ts). ELECTRODE contributing authors: Ludwig Ahrens-Iwers
 * (TUHH), Shern Tee (UQ), Robert Meissner (TUHH), and the LAMMPS developers. Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * kspace_style pppm/electrode: the PPPM solver of the core engine plus the electrode interface of ewald/electrode.
 * Single process. The grid, the Green's function and the charge weights are the core PPPM's (native-matched there);
 * the potential at the sensor atoms, compute_vector, is the mesh potential: charges of the source group are spread on
 * the mesh, transformed, multiplied by the optimal influence function, transformed back and interpolated at the
 * sensors with the same weights (pppm_electrode.cpp compute_vector and project_psi). The matrix is the one-step
 * W^T G W of compute_matrix, summed directly. Not ported (refused with a StyleError): kspace_modify slab ew2d, slab auto
 * and nozforce, the wire correction, triclinic boxes (the C++ refuses those too), and per-atom energy and virial.
 */
import { KSpaceCompute, StyleContext, StyleError } from '../../engine/force/types';
import { KSpacePPPM } from '../../engine/force/kspace/pppm';
import { fft3d } from '../../engine/force/fft';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { BoundaryCorrection, NoBoundaryCorrection, SlabDipole, type BoundaryEnv, inGroup } from './boundary_correction';
import type { ElectrodeKSpace } from './ewald_electrode';

export class KSpacePPPMElectrode extends KSpacePPPM implements ElectrodeKSpace {
  override readonly name = 'pppm/electrode';
  qqrd2e = 0;
  private boundcorr: BoundaryCorrection = new NoBoundaryCorrection();
  /** The box of the last init or compute (the vector routines are called outside compute). */
  private lastGeom: Geometry | null = null;
  private area = 0;
  private zprdSlab = 0;
  /** The real-space Green's function on the mesh (the inverse transform of the influence function), cached per box. */
  private greensReal = new Float64Array(0);
  private greensRealFor: Float64Array | null = null;

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'slab':
        if (values[0] === 'ew2d') throw new StyleError("Cannot (yet) use PPPM/electrode with 'kspace_modify slab ew2d' (native LAMMPS refuses it too)");
        if (values[0] === 'auto' || values[0] === 'nozforce') {
          throw new StyleError(`kspace_modify slab ${values[0]} is not supported by kspace_style pppm/electrode in the browser engine`);
        }
        return super.modify(key, values);
      case 'wire':
        throw new StyleError('kspace_modify wire (the wire correction) is not supported by the browser engine');
      default:
        return super.modify(key, values);
    }
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    if (geom.triclinic) throw new StyleError('Cannot (yet) use PPPM/electrode with triclinic box');
    if (s.dimension === 2) throw new StyleError('Cannot use PPPM/electrode with 2d simulation');
    super.init(s, geom, cutCoul, qqrd2e, ctx);
    this.qqrd2e = qqrd2e;
    this.boundcorr = this.slab > 1 ? new SlabDipole() : new NoBoundaryCorrection();
    this.lastGeom = geom;
    this.area = geom.lx * geom.ly;
    this.zprdSlab = geom.lz * this.slab;
    ctx.log('PPPM/electrode initialization ...');
  }

  /** The core PPPM slab term is replaced by the electrode boundary correction (boundcorr), as in the C++. */
  protected override slabCorrection(): void {
    // the EW3DC dipole correction is applied in compute() through boundcorr
  }

  compute(kc: KSpaceCompute): void {
    if (kc.eatom || kc.vatom) throw new StyleError('compute pe/atom and stress/atom with kspace_style pppm/electrode are not supported by the browser engine');
    const s = kc.s;
    this.qsum = 0;
    this.qsqsum = 0;
    for (let i = 0; i < s.n; i++) { this.qsum += s.q[i]; this.qsqsum += s.q[i] * s.q[i]; }
    this.lastGeom = kc.geom;
    this.area = kc.geom.lx * kc.geom.ly;
    this.zprdSlab = kc.geom.lz * this.slab;
    super.compute(kc);
    kc.acc.elong += this.boundcorr.computeCorr(s, kc.f, this.qsum, this.env(kc.qqrd2e));
  }

  private env(qqrd2e: number): BoundaryEnv {
    return { volume: this.volume, zprdSlab: this.zprdSlab, area: this.area, gEwald: this.gEwald, qqrd2e };
  }

  // ------------------------------------------------------------ electrode interface

  /**
   * compute_vector(): the mesh potential (without qqrd2e, as in ewald/electrode) at the sensor atoms from the source
   * charges. With rho the spread source charges, phi(x_i) = (1/V) sum_g W_i(g) IFFT[G FFT(rho)](g) (the C++ scales
   * the grid density by N/V and the projection by 1/N, which is the same 1/V).
   */
  computeVector(s: SimState, vec: Float64Array, sensorBit: number, sourceBit: number, invert: boolean): void {
    const geom = this.requireGeom();
    this.buildGreen(geom);
    this.volume = geom.volume(3) * this.slab;
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    const ng = nx * ny * nz;
    const { first, wts } = this.assignWeights(s, geom);
    const mod = (a: number, m: number) => ((a % m) + m) % m;
    const rho = new Float64Array(2 * ng);
    for (let i = 0; i < s.n; i++) {
      const q = s.q[i];
      if (q === 0 || !inGroup(s.mask[i], sourceBit, invert)) continue;
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
    for (let idx = 0; idx < ng; idx++) {
      const G = this.green[idx];
      rho[2 * idx] *= G;
      rho[2 * idx + 1] *= G;
    }
    fft3d(rho, nx, ny, nz, 1);
    const V = this.volume;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & sensorBit)) continue;
      const fx0 = first[3 * i], fy0 = first[3 * i + 1], fz0 = first[3 * i + 2];
      let v = 0;
      for (let c = 0; c < P; c++) {
        const iz = mod(fz0 + c, nz);
        const wz = wts[(3 * i + 2) * P + c];
        for (let b = 0; b < P; b++) {
          const iy = mod(fy0 + b, ny);
          const wyz = wz * wts[(3 * i + 1) * P + b];
          for (let a = 0; a < P; a++) {
            const ix = mod(fx0 + a, nx);
            v += wyz * wts[(3 * i) * P + a] * rho[2 * ((iz * ny + iy) * nx + ix)];
          }
        }
      }
      vec[i] += v / V;
    }
  }

  computeVectorCorr(s: SimState, vec: Float64Array, sensorBit: number, sourceBit: number, invert: boolean): void {
    this.boundcorr.vectorCorr(s, vec, sensorBit, sourceBit, invert, this.env(this.qqrd2e));
  }

  /**
   * compute_matrix(): the k-space part of the electrode-electrode matrix, M_ij = (1/V) sum_{g,g'} W_i(g) Gr(g - g') W_j(g'),
   * with Gr the forward transform of the influence function (pppm_electrode.cpp compute_matrix and
   * one_step_multiplication: greens_real and matrix[ipos][jpos] += aij / volume, mirrored for ipos != jpos). The C++
   * sums the same products through a stencil map (amesh); this sums them directly, which differs only in rounding.
   * Matrix entries are added (the fix adds the real-space and boundary terms), as in ewald/electrode.
   */
  computeMatrix(s: SimState, imat: Int32Array, matrix: Float64Array, ngroup: number): void {
    const geom = this.requireGeom();
    this.buildGreen(geom);
    this.volume = geom.volume(3) * this.slab;
    const gr = this.getGreensReal();
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    const P3 = P * P * P;
    const { first, wts } = this.assignWeights(s, geom);
    const mod = (a: number, m: number) => ((a % m) + m) % m;
    const group: number[] = [];
    for (let i = 0; i < s.n; i++) if (imat[i] >= 0) group.push(i);
    // per electrode atom: the P^3 stencil points (grid coordinates and flat index) and their weights
    const sx = new Int32Array(group.length * P3), sy = new Int32Array(group.length * P3), sz = new Int32Array(group.length * P3);
    const sw = new Float64Array(group.length * P3);
    group.forEach((i, gi) => {
      let p = 0;
      for (let c = 0; c < P; c++) {
        for (let b = 0; b < P; b++) {
          for (let a = 0; a < P; a++, p++) {
            sx[gi * P3 + p] = mod(first[3 * i] + a, nx);
            sy[gi * P3 + p] = mod(first[3 * i + 1] + b, ny);
            sz[gi * P3 + p] = mod(first[3 * i + 2] + c, nz);
            sw[gi * P3 + p] = wts[3 * i * P + a] * wts[(3 * i + 1) * P + b] * wts[(3 * i + 2) * P + c];
          }
        }
      }
    });
    const V = this.volume;
    for (let gi = 0; gi < group.length; gi++) {
      const ri = imat[group[gi]];
      for (let gj = 0; gj < group.length; gj++) {
        const rj = imat[group[gj]];
        if (rj > ri) continue;   // the matrix is symmetric: the lower triangle is summed, then mirrored
        let aij = 0;
        for (let p = gi * P3; p < (gi + 1) * P3; p++) {
          for (let r = gj * P3; r < (gj + 1) * P3; r++) {
            const dx = sx[p] - sx[r] < 0 ? sx[p] - sx[r] + nx : sx[p] - sx[r];
            const dy = sy[p] - sy[r] < 0 ? sy[p] - sy[r] + ny : sy[p] - sy[r];
            const dz = sz[p] - sz[r] < 0 ? sz[p] - sz[r] + nz : sz[p] - sz[r];
            aij += sw[p] * sw[r] * gr[(dz * ny + dy) * nx + dx];
          }
        }
        matrix[ri * ngroup + rj] += aij / V;
        if (ri !== rj) matrix[rj * ngroup + ri] += aij / V;
      }
    }
  }

  /** Inverse transform of the influence function (greens_real of the C++), recomputed when the box changes. */
  private getGreensReal(): Float64Array {
    if (this.greensRealFor === this.green) return this.greensReal;
    const [nx, ny, nz] = this.mesh;
    const ng = nx * ny * nz;
    const buf = new Float64Array(2 * ng);
    for (let i = 0; i < ng; i++) buf[2 * i] = this.green[i];
    fft3d(buf, nx, ny, nz, -1);
    const out = new Float64Array(ng);
    for (let i = 0; i < ng; i++) out[i] = buf[2 * i];
    this.greensReal = out;
    this.greensRealFor = this.green;
    return out;
  }

  computeMatrixCorr(s: SimState, imat: Int32Array, matrix: Float64Array, ngroup: number): void {
    this.boundcorr.matrixCorr(s, imat, matrix, ngroup, this.env(this.qqrd2e));
  }

  private requireGeom(): Geometry {
    if (!this.lastGeom) throw new StyleError('kspace_style pppm/electrode used before init');
    return this.lastGeom;
  }
}
