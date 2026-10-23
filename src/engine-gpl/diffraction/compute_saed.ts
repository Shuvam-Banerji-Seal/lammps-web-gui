/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DIFFRACTION/compute_saed.cpp and
 * compute_saed.h (ComputeSAED: compute saed, the selected-area electron diffraction intensity of the atoms in a group, as
 * a global vector with one entry per reciprocal lattice point on the Ewald-sphere slice of the zone axis, or on the whole
 * reciprocal volume for Zone 0 0 0). Contributing authors: Shawn Coleman and Douglas Spearot (Arkansas). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE). The atomic scattering factors are in asf_saed.ts (compute_saed_consts.h).
 *
 * docs.lammps.org/compute_saed.html: "compute ID group-ID saed lambda type1 type2 ... typeN keyword value ..."; keywords
 * Kmax ("Maximum distance explored from reciprocal space origin"), Zone ("z1,z2,z3 = Zone axis of incident radiation. If
 * z1=z2=z3=0 all reciprocal space will be meshed up to Kmax"), dR_Ewald, c, manual and echo. "This compute calculates a
 * global vector." The native limit |K|max/2 between 0 and 6 is kept as it is. "The compute_saed command does not work for
 * triclinic cells."
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { ASF_SAED, SAED_TYPE_LIST } from './asf_saed';
import { reciprocalSpacing } from './compute_xrd';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

const PI = Math.PI;

const numeric = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  return v;
};

export class ComputeSAED extends Compute {
  readonly style = 'saed';
  private readonly lambda: number;
  private readonly ztype: number[];
  private readonly Kmax: number;
  private readonly dRewald: number;
  private readonly rEwald: number;
  private readonly dK: [number, number, number];
  private readonly knmax: [number, number, number];
  /** reciprocal-lattice indices (i, j, k) of each entry, in native order (k outer, i inner) */
  private readonly hkl: Int32Array;
  private readonly nRows: number;
  /** The settings fix saed/vtk reads (native saed_var in compute_saed.h): Zone already scaled to 1/lambda. */
  readonly saedVar: { lambda: number; Kmax: number; zone: [number, number, number]; c: [number, number, number]; dRewald: number; manual: boolean };

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = 'compute saed';
    if (!sys.hasBox) throw new StyleError('compute saed requires a simulation box');
    const ntypes = sys.state.ntypes;
    if (sys.dimension === 2) throw new StyleError('Compute SAED does not work with 2d structures');
    if (args.length < 1 + ntypes) throw new StyleError('Illegal Compute SAED Command');
    if (sys.state.box.triclinic) throw new StyleError('Compute SAED does not work with triclinic structures');
    this.vectorFlag = true;
    this.extvector = 0;
    this.lambda = numeric(args[0], what);
    if (this.lambda < 0) throw new StyleError('Compute SAED: Wavelength must be greater than zero');
    this.ztype = [];
    for (let t = 0; t < ntypes; t++) {
      const want = args[1 + t].toLowerCase();
      let z = -1;
      for (let j = 0; j < SAED_TYPE_LIST.length; j++) if (SAED_TYPE_LIST[j].toLowerCase() === want) z = j;
      if (z < 0) throw new StyleError('Compute SAED: Invalid ASF atom type');
      this.ztype.push(z);
    }
    let Kmax = 1.7;
    const zone: [number, number, number] = [1, 0, 0];
    const c: [number, number, number] = [1, 1, 1];
    let dR = 0.01 / 2;
    let manual = false;
    let echo = false;
    let iarg = 1 + ntypes;
    while (iarg < args.length) {
      const kw = args[iarg];
      if (kw === 'Kmax') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal Compute SAED Command');
        Kmax = numeric(args[iarg + 1], what);
        if (Kmax / 2 < 0 || Kmax / 2 > 6) throw new StyleError('Compute SAED: |K|max/2 must be between 0 and 6 ');
        iarg += 2;
      } else if (kw === 'Zone') {
        if (iarg + 4 > args.length) throw new StyleError('Illegal Compute SAED Command');
        zone[0] = numeric(args[iarg + 1], what);
        zone[1] = numeric(args[iarg + 2], what);
        zone[2] = numeric(args[iarg + 3], what);
        iarg += 4;
      } else if (kw === 'c') {
        if (iarg + 4 > args.length) throw new StyleError('Illegal Compute SAED Command');
        c[0] = numeric(args[iarg + 1], what);
        c[1] = numeric(args[iarg + 2], what);
        c[2] = numeric(args[iarg + 3], what);
        if (c[0] < 0 || c[1] < 0 || c[2] < 0) throw new StyleError('Compute SAED: dKs must be greater than 0');
        iarg += 4;
      } else if (kw === 'dR_Ewald') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal Compute SAED Command');
        dR = numeric(args[iarg + 1], what);
        if (dR < 0) throw new StyleError('Compute SAED: dR_Ewald slice must be greater than 0');
        iarg += 2;
      } else if (kw === 'echo') {
        echo = true;
        iarg += 1;
      } else if (kw === 'manual') {
        manual = true;
        iarg += 1;
      } else throw new StyleError('Illegal Compute SAED Command');
    }
    this.Kmax = Kmax;
    this.dRewald = dR;
    // a zone axis is scaled to the Ewald radius 1/lambda (native: Zone 0 0 0 keeps the whole volume)
    this.rEwald = 1 / this.lambda;
    if (!(zone[0] === 0 && zone[1] === 0 && zone[2] === 0)) {
      const rnorm = this.rEwald / Math.sqrt(zone[0] * zone[0] + zone[1] * zone[1] + zone[2] * zone[2]);
      zone[0] = zone[0] * rnorm;
      zone[1] = zone[1] * rnorm;
      zone[2] = zone[2] * rnorm;
    }
    this.saedVar = { lambda: this.lambda, Kmax, zone: [zone[0], zone[1], zone[2]], c: [c[0], c[1], c[2]], dRewald: dR, manual };
    const prdInv = reciprocalSpacing(sys, manual, 'Compute SAED');
    this.dK = [prdInv[0] * c[0], prdInv[1] * c[1], prdInv[2] * c[2]];
    this.knmax = [0, 0, 0];
    for (let d = 0; d < 3; d++) this.knmax[d] = Math.ceil(Kmax / this.dK[d]);
    const whole = zone[0] === 0 && zone[1] === 0 && zone[2] === 0;
    const found: number[] = [];
    const emdR2 = (this.rEwald - this.dRewald) ** 2;
    const epdR2 = (this.rEwald + this.dRewald) ** 2;
    for (let k = -this.knmax[2]; k <= this.knmax[2]; k++) {
      for (let j = -this.knmax[1]; j <= this.knmax[1]; j++) {
        for (let i = -this.knmax[0]; i <= this.knmax[0]; i++) {
          const Kv = [i * this.dK[0], j * this.dK[1], k * this.dK[2]];
          const dinv2 = Kv[0] * Kv[0] + Kv[1] * Kv[1] + Kv[2] * Kv[2];
          if (!(dinv2 < Kmax * Kmax)) continue;
          if (whole) {
            found.push(i, j, k);
          } else {
            let r2 = 0;
            for (let m = 0; m < 3; m++) r2 += (Kv[m] - zone[m]) * (Kv[m] - zone[m]);
            if (r2 > emdR2 && r2 < epdR2) found.push(i, j, k);
          }
        }
      }
    }
    this.nRows = found.length / 3;
    this.hkl = Int32Array.from(found);
    this.sizeVector = this.nRows;
    this.vector = new Float64Array(this.nRows);
    if (echo) sys.log(`-----\nCompute SAED id:${id}, # of atoms:${sys.state.n}, # of relp:${this.nRows}\nReciprocal point spacing in k1,k2,k3 = ${this.dK[0].toPrecision(8)} ${this.dK[1].toPrecision(8)} ${this.dK[2].toPrecision(8)}\n-----`);
  }

  protected computeVector(): void {
    this.sys.forces();
    const s = this.sys.state;
    const n = s.n;
    const xs: number[] = [], ts: number[] = [];
    let natoms = 0;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      natoms++;
      xs.push(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
      ts.push(s.type[i]);
    }
    const nt = s.ntypes;
    const nRows = this.nRows;
    // two parameter sets: S = Kmax/2 <= 2 uses the first five pairs, S > 2 the second set (native offset 0 or 10)
    const Smax = this.Kmax / 2;
    const offset = Smax > 2 ? 10 : 0;
    const f = new Float64Array(nt);
    const F = new Float64Array(2 * nRows);
    const K = [0, 0, 0];
    for (let r = 0; r < nRows; r++) {
      const i = this.hkl[3 * r], j = this.hkl[3 * r + 1], k = this.hkl[3 * r + 2];
      K[0] = i * this.dK[0];
      K[1] = j * this.dK[1];
      K[2] = k * this.dK[2];
      const dinv2 = K[0] * K[0] + K[1] * K[1] + K[2] * K[2];
      const dinv = Math.sqrt(dinv2);
      const SinTheta_lambda = 0.5 * dinv;
      for (let t = 0; t < nt; t++) {
        const row = ASF_SAED[this.ztype[t]];
        let ft = 0;
        for (let C = 0; C < 5; C++) {
          const D = C + offset;
          ft += row[D] * libmExp(-1 * row[5 + D] * SinTheta_lambda * SinTheta_lambda);
        }
        f[t] = ft;
      }
      let F1 = 0, F2 = 0;
      for (let a = 0; a < ts.length; a++) {
        const typei = ts[a] - 1;
        const inners = 2 * PI * (K[0] * xs[3 * a] + K[1] * xs[3 * a + 1] + K[2] * xs[3 * a + 2]);
        F1 += f[typei] * libmCos(inners);
        F2 += f[typei] * libmSin(inners);
      }
      F[2 * r] = F1;
      F[2 * r + 1] = F2;
    }
    for (let r = 0; r < nRows; r++) {
      this.vector[r] = (F[2 * r] * F[2 * r] + F[2 * r + 1] * F[2 * r + 1]) / natoms;
    }
  }
}
