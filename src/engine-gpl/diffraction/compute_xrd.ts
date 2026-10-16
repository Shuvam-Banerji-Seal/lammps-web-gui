/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DIFFRACTION/compute_xrd.cpp and
 * compute_xrd.h (ComputeXRD: compute xrd, the x-ray diffraction intensity of the atoms in a group, as a global array with
 * one row per reciprocal lattice point: column 1 = 2 theta (degrees, or radians when 2Theta is given below 2 pi), column 2
 * = the intensity). Contributing authors: Shawn Coleman and Douglas Spearot (Arkansas). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). The atomic scattering factors are in asf_xrd.ts (compute_xrd_consts.h).
 *
 * docs.lammps.org/compute_xrd.html: "compute ID group-ID xrd lambda type1 type2 ... typeN keyword value ..."; "lambda =
 * wavelength of incident radiation (length units)"; "type1 type2 ... typeN = chemical symbol of each atom type"; keyword
 * 2Theta with "Min2Theta,Max2Theta = minimum and maximum 2 theta range to explore (radians or degrees)"; keyword c "c1,c2,c3
 * = parameters to adjust the spacing of the reciprocal lattice nodes"; keyword LP "0/1 = off/on"; keyword manual "flag to
 * use manual spacing of reciprocal lattice points based on the values of the c parameters"; keyword echo. "This compute
 * calculates a global array." "The global array has two columns." "The compute_xrd command does not work for triclinic
 * cells." The native 2d check and the native checks of the 2Theta range are kept as they are.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { ASF_XRD, XRD_TYPE_LIST } from './asf_xrd';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

const PI = Math.PI;

/** utils::numeric: a floating point argument. */
const numeric = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in ${what}`);
  return v;
};

/** utils::inumeric: an integer argument. */
const inumeric = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || !/^[-+]?\d+$/.test(w.trim())) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in ${what}`);
  return v;
};

export class ComputeXRD extends Compute {
  readonly style = 'xrd';
  private readonly lambda: number;
  /** atom type (0-based) -> row of ASF_XRD */
  private readonly ztype: number[];
  private readonly min2Theta: number;
  private readonly max2Theta: number;
  private readonly radflag: boolean;
  private readonly LP: number;
  private readonly dK: [number, number, number];
  private readonly knmax: [number, number, number];
  /** reciprocal-lattice indices (i, j, k) of each row, in native row order */
  private readonly hkl: Int32Array;
  private readonly rows: number;
  private readonly convf: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = 'compute xrd';
    if (!sys.hasBox) throw new StyleError('compute xrd requires a simulation box');
    const ntypes = sys.state.ntypes;
    if (sys.dimension === 2) throw new StyleError('Compute XRD does not work with 2d structures');
    if (args.length < 1 + ntypes) throw new StyleError('Illegal Compute XRD Command');
    if (sys.state.box.triclinic) throw new StyleError('Compute XRD does not work with triclinic structures');
    this.arrayFlag = true;
    this.lambda = numeric(args[0], what);
    if (this.lambda < 0) throw new StyleError('Compute SAED: Wavelength must be greater than zero');
    this.ztype = [];
    for (let t = 0; t < ntypes; t++) {
      const want = args[1 + t].toLowerCase();
      let z = -1;
      for (let j = 0; j < ASF_XRD.length; j++) if (XRD_TYPE_LIST[j].toLowerCase() === want) z = j;
      if (z < 0) throw new StyleError(`Compute XRD: Invalid ASF atom type ${args[1 + t]}`);
      this.ztype.push(z);
    }
    let min2 = 1, max2 = 179;
    this.radflag = true;
    const c: [number, number, number] = [1, 1, 1];
    let LP = 1;
    let echo = false;
    let manual = false;
    let iarg = 1 + ntypes;
    while (iarg < args.length) {
      const kw = args[iarg];
      if (kw === '2Theta') {
        if (iarg + 3 > args.length) throw new StyleError('Illegal Compute XRD Command');
        min2 = numeric(args[iarg + 1], what);
        max2 = numeric(args[iarg + 2], what);
        iarg += 3;
      } else if (kw === 'c') {
        if (iarg + 4 > args.length) throw new StyleError('Illegal Compute XRD Command');
        c[0] = numeric(args[iarg + 1], what);
        c[1] = numeric(args[iarg + 2], what);
        c[2] = numeric(args[iarg + 3], what);
        if (c[0] < 0 || c[1] < 0 || c[2] < 0) throw new StyleError("Compute XRD: c's must be greater than 0");
        iarg += 4;
      } else if (kw === 'LP') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal Compute XRD Command');
        LP = inumeric(args[iarg + 1], what);
        if (LP !== 1 && LP !== 0) throw new StyleError('Compute XRD: LP must have value of 0 or 1');
        iarg += 2;
      } else if (kw === 'echo') {
        echo = true;
        iarg += 1;
      } else if (kw === 'manual') {
        manual = true;
        iarg += 1;
      } else throw new StyleError(`Illegal Compute XRD Command: unknown keyword '${kw}'`);
    }
    this.LP = LP;
    // 2Theta is given in degrees unless its maximum is below 2 pi (then radians); the half angles are used below
    min2 /= 2.0;
    max2 /= 2.0;
    if (max2 > PI) {
      min2 = (min2 * PI) / 180;
      max2 = (max2 * PI) / 180;
      this.radflag = false;
    }
    if (min2 <= 0) throw new StyleError('Minimum 2Theta value must be greater than zero');
    if (max2 >= PI) throw new StyleError('Maximum 2Theta value must be less than 180 degrees');
    if (max2 - min2 <= 0) throw new StyleError('2Theta range must be greater than zero');
    this.min2Theta = min2;
    this.max2Theta = max2;
    this.convf = this.radflag ? 2 : (360 / PI);
    const kmax = (2 * libmSin(max2)) / this.lambda;
    const prdInv = reciprocalSpacing(sys, manual, 'Compute XRD');
    this.dK = [prdInv[0] * c[0], prdInv[1] * c[1], prdInv[2] * c[2]];
    this.knmax = [0, 0, 0];
    for (let d = 0; d < 3; d++) this.knmax[d] = Math.ceil(kmax / this.dK[d]);
    // rows: the reciprocal points whose Bragg angle is inside the 2Theta window (native order: i outer, k inner)
    const found: number[] = [];
    for (let i = -this.knmax[0]; i <= this.knmax[0]; i++) {
      for (let j = -this.knmax[1]; j <= this.knmax[1]; j++) {
        for (let k = -this.knmax[2]; k <= this.knmax[2]; k++) {
          const Kx = i * this.dK[0], Ky = j * this.dK[1], Kz = k * this.dK[2];
          const dinv2 = Kx * Kx + Ky * Ky + Kz * Kz;
          if (4 >= dinv2 * this.lambda * this.lambda) {
            const ang = Math.asin(this.lambda * Math.sqrt(dinv2) * 0.5);
            if (ang <= this.max2Theta && ang >= this.min2Theta) found.push(k, j, i);
          }
        }
      }
    }
    this.rows = found.length / 3;
    this.hkl = Int32Array.from(found);
    this.sizeArrayRows = this.rows;
    this.sizeArrayCols = 2;
    this.array = new Float64Array(2 * this.rows);
    // column 1 is the 2theta value of each row (set now, as native init() does)
    for (let n = 0; n < this.rows; n++) {
      const i = this.hkl[3 * n + 2], j = this.hkl[3 * n + 1], k = this.hkl[3 * n];
      const Kx = i * this.dK[0], Ky = j * this.dK[1], Kz = k * this.dK[2];
      const dinv2 = Kx * Kx + Ky * Ky + Kz * Kz;
      const ang = Math.asin(this.lambda * Math.sqrt(dinv2) * 0.5);
      this.array[2 * n] = ang * this.convf;
    }
    if (echo) sys.log(`-----\nCompute XRD id:${id}, # of atoms:${sys.state.n}, # of relp:${this.rows}\nReciprocal point spacing in k1,k2,k3 = ${this.dK[0].toPrecision(8)} ${this.dK[1].toPrecision(8)} ${this.dK[2].toPrecision(8)}\n-----`);
  }

  protected computeArray(): void {
    this.sys.forces();
    const s = this.sys.state;
    const n = s.n;
    const rows = this.rows;
    // the atoms of the group, in the state order (positions and types)
    const xs: number[] = [], ts: number[] = [];
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      xs.push(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
      ts.push(s.type[i]);
    }
    const natoms = countGroup(s.mask, n, this.groupBit);
    const nt = s.ntypes;
    const f = new Float64Array(nt);
    const F = new Float64Array(rows * 2);
    const lambda = this.lambda;
    const K = [0, 0, 0];
    for (let r = 0; r < rows; r++) {
      const i = this.hkl[3 * r + 2], j = this.hkl[3 * r + 1], k = this.hkl[3 * r];
      K[0] = i * this.dK[0];
      K[1] = j * this.dK[1];
      K[2] = k * this.dK[2];
      const dinv2 = K[0] * K[0] + K[1] * K[1] + K[2] * K[2];
      const dinv = Math.sqrt(dinv2);
      const SinTheta_lambda = 0.5 * dinv;
      const SinTheta = SinTheta_lambda * lambda;
      const ang = Math.asin(SinTheta);
      const Cos2Theta = libmCos(2 * ang);
      const CosTheta = libmCos(ang);
      for (let t = 0; t < nt; t++) {
        const row = ASF_XRD[this.ztype[t]];
        let ft = 0;
        for (let C = 0; C < 8; C += 2) ft += row[C] * libmExp(-1 * row[C + 1] * SinTheta_lambda * SinTheta_lambda);
        f[t] = ft + row[8];
      }
      let F1 = 0, F2 = 0;
      for (let a = 0; a < xs.length / 3; a++) {
        const typei = ts[a] - 1;
        const inners = 2 * PI * (K[0] * xs[3 * a] + K[1] * xs[3 * a + 1] + K[2] * xs[3 * a + 2]);
        F1 += f[typei] * libmCos(inners);
        F2 += f[typei] * libmSin(inners);
      }
      if (this.LP === 1) {
        const sqrtLp = Math.sqrt((1 + Cos2Theta * Cos2Theta) / (CosTheta * SinTheta * SinTheta));
        F[2 * r] = F1 * sqrtLp;
        F[2 * r + 1] = F2 * sqrtLp;
      } else {
        F[2 * r] = F1;
        F[2 * r + 1] = F2;
      }
    }
    for (let r = 0; r < rows; r++) {
      this.array[2 * r + 1] = (F[2 * r] * F[2 * r] + F[2 * r + 1] * F[2 * r + 1]) / natoms;
    }
  }
}

/**
 * Reciprocal spacing per direction (compute xrd and compute saed share it): 1/prd for periodic directions, the average
 * inverse length for the others; all 1 with manual. Native: "Compute SAED must have at least one periodic boundary".
 */
export const reciprocalSpacing = (sys: System, manual: boolean, what: string): [number, number, number] => {
  const prdInv: [number, number, number] = [1, 1, 1];
  if (manual) return prdInv;
  const s = sys.state;
  const per = s.box.periodic;
  const np = (per[0] ? 1 : 0) + (per[1] ? 1 : 0) + (per[2] ? 1 : 0);
  if (np === 0) throw new StyleError(`${what} must have at least one periodic boundary unless manual spacing specified`);
  let ave = 0;
  for (let d = 0; d < 3; d++) {
    if (per[d]) {
      prdInv[d] = 1 / (s.box.hi[d] - s.box.lo[d]);
      ave += prdInv[d];
    }
  }
  ave = ave / np;
  for (let d = 0; d < 3; d++) if (!per[d]) prdInv[d] = ave;
  return prdInv;
};

const countGroup = (mask: Int32Array, n: number, bit: number): number => {
  let c = 0;
  for (let i = 0; i < n; i++) if (mask[i] & bit) c++;
  return c;
};
