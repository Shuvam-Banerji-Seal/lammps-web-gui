/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM/meam_setup_global.cpp,
 * meam_setup_param.cpp, meam_setup_done.cpp (compute_pair_meam, phi_meam, compute_reference_density, get_tavref,
 * get_densref, interpolate_meam) and the MS-MEAM (msmeamflag) branches of those routines, plus the parameter
 * reading of src/MEAM/pair_meam.cpp (read_global_meam_file, read_user_meam_file) and src/MEAM/pair_meam_ms.cpp.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Original MEAM authors (from meam_setup_done.cpp and
 * meam.h): Greg Wagner (Sandia), Sungkwang Mun, Stephen Foiles, Michael I. Baskes; MS-MEAM: Steve Valone (LANL),
 * Sergei Rudin, Mikhail Baskes, Mark Baskes.
 *
 * The model is MS-MEAM only (pair_style meam/ms sets msmeamflag = 1, ialloy = 1 by meam_setup_global).
 */
import { fmExp, getShpfcn, getZij, getZij2, getZij2b2nn, iszero, isone, fcut, erosePot, zblPot, powint, type Lattice } from './meamMath';

export const MAXELT = 8;
const NR = 1000;

const mat2 = (n: number, v: number): number[][] => Array.from({ length: n }, () => new Array<number>(n).fill(v));
const mat3 = (n: number, v: number): number[][][] => Array.from({ length: n }, () => Array.from({ length: n }, () => new Array<number>(n).fill(v)));
const vec = (n: number, v: number): number[] => new Array<number>(n).fill(v);

/** One element's library entry (pair_meam.cpp read_global_meam_file, with the msmeam fields). */
export interface LibEntry {
  lat: Lattice;
  z: number;
  ielement: number;
  atwt: number;
  alpha: number;
  b0: number;
  b1: number;
  b2: number;
  b3: number;
  b1m: number;
  b2m: number;
  b3m: number;
  alat: number;
  esub: number;
  asub: number;
  t0: number;
  t1: number;
  t2: number;
  t3: number;
  t1m: number;
  t2m: number;
  t3m: number;
  rozero: number;
  ibar: number;
}

const KEYWORDS = ['Ec', 'alpha', 'rho0', 'delta', 'lattce', 'attrac', 'repuls', 'nn2', 'Cmin', 'Cmax', 'rc', 'delr', 'augt1',
  'gsmooth_factor', 're', 'ialloy', 'mixture_ref_t', 'erose_form', 'zbl', 'emb_lin_neg', 'bkgd_dyn', 'theta'];
export const meamKeywords = KEYWORDS;

/** The lattice names accepted for a multi-element pair (MEAM::str_to_lat with single = false). */
export const LATTICE_NAMES: readonly string[] = ['fcc', 'bcc', 'hcp', 'dim', 'dia', 'dia3', 'lin', 'zig', 'tri', 'sc', 'b1', 'c11', 'l12', 'b2', 'ch4'];
export const SINGLE_LATTICE_NAMES: readonly string[] = ['fcc', 'bcc', 'hcp', 'dim', 'dia', 'dia3', 'lin', 'zig', 'tri', 'sc'];

/** Element names of MEAM::str_to_lat. */
export const toLattice = (s: string, single: boolean): Lattice | null => {
  const names = single ? SINGLE_LATTICE_NAMES : LATTICE_NAMES;
  return names.includes(s) ? (s as Lattice) : null;
};

/** Pair index a <= b in MEAM::eltind (row-major over the upper triangle). */
export class MeamModel {
  neltypes = 0;
  cutforce = 0;
  cutforcesq = 0;
  Ec = mat2(MAXELT, 0);
  re = mat2(MAXELT, 0);
  A = vec(MAXELT, 0);
  alpha = mat2(MAXELT, 0);
  rho0 = vec(MAXELT, 0);
  delta = mat2(MAXELT, 0);
  beta0 = vec(MAXELT, 0);
  beta1 = vec(MAXELT, 0);
  beta2 = vec(MAXELT, 0);
  beta3 = vec(MAXELT, 0);
  t0 = vec(MAXELT, 0);
  t1 = vec(MAXELT, 0);
  t2 = vec(MAXELT, 0);
  t3 = vec(MAXELT, 0);
  rhoRef = vec(MAXELT, 0);
  ibar = vec(MAXELT, 0);
  ielt = vec(MAXELT, 0);
  lattce: Lattice[][] = Array.from({ length: MAXELT }, () => new Array<Lattice>(MAXELT).fill('fcc'));
  nn2 = mat2(MAXELT, 0);
  zbl = mat2(MAXELT, 1);
  eltind = mat2(MAXELT, 0);
  attrac = mat2(MAXELT, 0);
  repuls = mat2(MAXELT, 0);
  Cmin = mat3(MAXELT, 2.0);
  Cmax = mat3(MAXELT, 2.8);
  ebound = mat2(MAXELT, (2.8 * 2.8) / (4.0 * (2.8 - 1.0)));
  rc = 4.0;
  delr = 0.1;
  augt1 = 1;
  ialloy = 1;
  mixRefT = 0;
  ereseForm = 0;
  embLinNeg = 0;
  bkgdDyn = 0;
  gsmooth = 99.0;
  stheta = mat2(MAXELT, 1.0);
  ctheta = mat2(MAXELT, 0.0);
  t1m = vec(MAXELT, 0);
  t2m = vec(MAXELT, 0);
  t3m = vec(MAXELT, 0);
  beta1m = vec(MAXELT, 0);
  beta2m = vec(MAXELT, 0);
  beta3m = vec(MAXELT, 0);
  nr = NR;
  dr = 0;
  /** phi(r) tables per element pair (index eltind), and their spline coefficients */
  phir: Float64Array[] = [];
  phirar: Float64Array[] = [];
  phirar1: Float64Array[] = [];
  phirar2: Float64Array[] = [];
  phirar3: Float64Array[] = [];
  phirar4: Float64Array[] = [];
  phirar5: Float64Array[] = [];
  phirar6: Float64Array[] = [];

  /** meam_setup_global: element parameters (msmeam set). */
  setupGlobal(lib: LibEntry[]): void {
    this.neltypes = lib.length;
    for (let i = 0; i < lib.length; i++) {
      const L = lib[i];
      this.lattce[i][i] = L.lat;
      this.ielt[i] = L.ielement;
      this.alpha[i][i] = L.alpha;
      this.beta0[i] = L.b0;
      this.beta1[i] = L.b1;
      this.beta2[i] = L.b2;
      this.beta3[i] = L.b3;
      this.beta1m[i] = L.b1m;
      this.beta2m[i] = L.b2m;
      this.beta3m[i] = L.b3m;
      const tmplat = L.alat;
      this.Ec[i][i] = L.esub;
      this.A[i] = L.asub;
      this.t0[i] = L.t0;
      this.t1[i] = L.t1;
      this.t2[i] = L.t2;
      this.t3[i] = L.t3;
      this.t1m[i] = L.t1m;
      this.t2m[i] = L.t2m;
      this.t3m[i] = L.t3m;
      this.rho0[i] = L.rozero;
      this.ibar[i] = L.ibar;
      switch (L.lat) {
        case 'fcc': this.re[i][i] = tmplat / Math.sqrt(2.0); break;
        case 'bcc': this.re[i][i] = (tmplat * Math.sqrt(3.0)) / 2.0; break;
        case 'hcp': case 'dim': case 'ch4': case 'lin': case 'zig': case 'tri': case 'sc':
          this.re[i][i] = tmplat; break;
        case 'dia': case 'dia3': this.re[i][i] = (tmplat * Math.sqrt(3.0)) / 4.0; break;
        default: break;
      }
    }
    this.rc = 4.0;
    this.delr = 0.1;
    this.attrac = mat2(MAXELT, 0);
    this.repuls = mat2(MAXELT, 0);
    this.Cmax = mat3(MAXELT, 2.8);
    this.Cmin = mat3(MAXELT, 2.0);
    this.ebound = mat2(MAXELT, (2.8 * 2.8) / (4.0 * (2.8 - 1.0)));
    this.delta = mat2(MAXELT, 0);
    this.nn2 = mat2(MAXELT, 0);
    this.zbl = mat2(MAXELT, 1);
    this.gsmooth = 99.0;
    this.augt1 = 1;
    this.ialloy = 1;
    this.mixRefT = 0;
    this.embLinNeg = 0;
    this.bkgdDyn = 0;
    this.ereseForm = 0;
    this.stheta = mat2(MAXELT, 1.0);
    this.ctheta = mat2(MAXELT, 0.0);
  }

  /** meam_setup_param: one keyword of the user parameter file (which = index of KEYWORDS). */
  setupParam(which: number, value: number, idx: number[]): void {
    const need = (n: number): void => {
      if (idx.length < n) throw new Error('expected more indices');
      for (let i = 0; i < n; i++) if (idx[i] < 0 || idx[i] >= this.neltypes) throw new Error('has out of range element index');
    };
    switch (which) {
      case 0: need(2); this.Ec[idx[0]][idx[1]] = value; break;
      case 1: need(2); this.alpha[idx[0]][idx[1]] = value; break;
      case 2: need(1); this.rho0[idx[0]] = value; break;
      case 3: need(2); this.delta[idx[0]][idx[1]] = value; break;
      case 4: need(2); this.lattce[idx[0]][idx[1]] = LATTICE_BY_CODE[value]; break;
      case 5: need(2); this.attrac[idx[0]][idx[1]] = value; break;
      case 6: need(2); this.repuls[idx[0]][idx[1]] = value; break;
      case 7: need(2); this.nn2[Math.min(idx[0], idx[1])][Math.max(idx[0], idx[1])] = Math.trunc(value); break;
      case 8: need(3); this.Cmin[idx[0]][idx[1]][idx[2]] = value; break;
      case 9: need(3); this.Cmax[idx[0]][idx[1]][idx[2]] = value; break;
      case 10: this.rc = value; break;
      case 11: this.delr = value; break;
      case 12: this.augt1 = Math.trunc(value); break;
      case 13: this.gsmooth = value; break;
      case 14: need(2); this.re[idx[0]][idx[1]] = value; break;
      case 15: this.ialloy = Math.trunc(value); break;
      case 16: this.mixRefT = Math.trunc(value); break;
      case 17: this.ereseForm = Math.trunc(value); break;
      case 18: need(2); this.zbl[Math.min(idx[0], idx[1])][Math.max(idx[0], idx[1])] = Math.trunc(value); break;
      case 19: this.embLinNeg = Math.trunc(value); break;
      case 20: this.bkgdDyn = Math.trunc(value); break;
      case 21: {
        need(2);
        const i1 = Math.min(idx[0], idx[1]), i2 = Math.max(idx[0], idx[1]);
        this.stheta[i1][i2] = Math.sin((value / 2 * Math.PI) / 180.0);
        this.ctheta[i1][i2] = Math.cos((value / 2 * Math.PI) / 180.0);
        break;
      }
      default: throw new Error('unknown keyword');
    }
  }

  /** meam_setup_done: augment t1, alloy parameters, reference densities, pair tables. Returns the cutoff. */
  setupDone(): number {
    this.cutforce = this.rc;
    this.cutforcesq = this.cutforce * this.cutforce;
    for (let i = 0; i < MAXELT; i++) this.t1[i] = this.t1[i] + this.augt1 * 3.0 / 5.0 * this.t3[i];
    this.alloyparams();
    let nv2 = 0;
    for (let m = 0; m < this.neltypes; m++) {
      for (let n = m; n < this.neltypes; n++) {
        this.eltind[m][n] = nv2;
        this.eltind[n][m] = nv2;
        nv2 = nv2 + 1;
      }
    }
    this.computeReferenceDensity();
    this.nr = NR;
    this.dr = (1.1 * this.rc) / this.nr;
    this.computePairMeam();
    return this.cutforce;
  }

  alloyparams(): void {
    const n = this.neltypes;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i > j) {
          this.re[i][j] = this.re[j][i];
          this.Ec[i][j] = this.Ec[j][i];
          this.alpha[i][j] = this.alpha[j][i];
          this.lattce[i][j] = this.lattce[j][i];
          this.nn2[i][j] = this.nn2[j][i];
          this.stheta[i][j] = this.stheta[j][i];
          this.ctheta[i][j] = this.ctheta[j][i];
        } else if (j > i) {
          if (iszero(this.Ec[i][j])) {
            if (this.lattce[i][j] === 'l12') this.Ec[i][j] = (3 * this.Ec[i][i] + this.Ec[j][j]) / 4.0 - this.delta[i][j];
            else if (this.lattce[i][j] === 'c11') {
              if (this.lattce[i][i] === 'dia') this.Ec[i][j] = (2 * this.Ec[i][i] + this.Ec[j][j]) / 3.0 - this.delta[i][j];
              else this.Ec[i][j] = (this.Ec[i][i] + 2 * this.Ec[j][j]) / 3.0 - this.delta[i][j];
            } else this.Ec[i][j] = (this.Ec[i][i] + this.Ec[j][j]) / 2.0 - this.delta[i][j];
          }
          if (iszero(this.alpha[i][j])) this.alpha[i][j] = (this.alpha[i][i] + this.alpha[j][j]) / 2.0;
          if (iszero(this.re[i][j])) this.re[i][j] = (this.re[i][i] + this.re[j][j]) / 2.0;
        }
      }
    }
    for (let i = 1; i < n; i++) {
      for (let j = 0; j < i; j++) {
        for (let k = 0; k < n; k++) {
          this.Cmin[i][j][k] = this.Cmin[j][i][k];
          this.Cmax[i][j][k] = this.Cmax[j][i][k];
        }
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          const cm = this.Cmax[i][j][k];
          const eb = (cm * cm) / (4.0 * (cm - 1.0));
          this.ebound[i][j] = Math.max(this.ebound[i][j], eb);
        }
      }
    }
  }

  computePairMeam(): void {
    const npair = (this.neltypes * (this.neltypes + 1)) / 2;
    this.phir = []; this.phirar = []; this.phirar1 = []; this.phirar2 = [];
    this.phirar3 = []; this.phirar4 = []; this.phirar5 = []; this.phirar6 = [];
    for (let p = 0; p < npair; p++) {
      this.phir.push(new Float64Array(this.nr));
      this.phirar.push(new Float64Array(this.nr));
      this.phirar1.push(new Float64Array(this.nr));
      this.phirar2.push(new Float64Array(this.nr));
      this.phirar3.push(new Float64Array(this.nr));
      this.phirar4.push(new Float64Array(this.nr));
      this.phirar5.push(new Float64Array(this.nr));
      this.phirar6.push(new Float64Array(this.nr));
    }
    const nr = this.nr, dr = this.dr;
    let nv2 = 0;
    for (let a = 0; a < this.neltypes; a++) {
      for (let b = a; b < this.neltypes; b++) {
        const phir = this.phir[nv2];
        for (let j = 0; j < nr; j++) {
          const r = j * dr;
          phir[j] = this.phiMeam(r, a, b);
          if (this.nn2[a][b] === 1) {
            const z1 = getZij(this.lattce[a][b]);
            const z2s = getZij2(this.lattce[a][b], this.Cmin[a][a][b], this.Cmax[a][a][b], this.stheta[a][b]);
            let arat = z2s.a;
            let scrn = z2s.S;
            let Z1 = z1;
            let Z2 = z2s.z;
            const lab = this.lattce[a][b];
            if (lab === 'b1' || lab === 'b2' || lab === 'l12' || lab === 'dia') {
              const rarat = r * arat;
              let phiaa = this.phiMeam(rarat, a, a);
              Z1 = getZij(this.lattce[a][a]);
              {
                const g = getZij2(this.lattce[a][a], this.Cmin[a][a][a], this.Cmax[a][a][a], this.stheta[a][a]);
                Z2 = g.z; arat = g.a; scrn = g.S;
              }
              phiaa += this.phiMeamSeries(scrn, Z1, Z2, a, a, rarat, arat);
              let phibb = this.phiMeam(rarat, b, b);
              Z1 = getZij(this.lattce[b][b]);
              {
                const g = getZij2(this.lattce[b][b], this.Cmin[b][b][b], this.Cmax[b][b][b], this.stheta[b][b]);
                Z2 = g.z; arat = g.a; scrn = g.S;
              }
              phibb += this.phiMeamSeries(scrn, Z1, Z2, b, b, rarat, arat);
              if (lab === 'b1' || lab === 'b2' || lab === 'dia') {
                Z1 = getZij(lab);
                {
                  const g = getZij2(lab, this.Cmin[a][a][b], this.Cmax[a][a][b], this.stheta[a][b]);
                  Z2 = g.z; arat = g.a; scrn = g.S;
                }
                phir[j] = phir[j] - (Z2 * scrn) / (2 * Z1) * phiaa;
                {
                  const g = getZij2(lab, this.Cmin[b][b][a], this.Cmax[b][b][a], this.stheta[a][b]);
                  Z2 = g.z;
                  const scrn2 = g.S;
                  phir[j] = phir[j] - (Z2 * scrn2) / (2 * Z1) * phibb;
                }
              } else if (lab === 'l12') {
                const C = 1.0;
                const s111 = this.getSijk(C, a, a, a);
                const s112 = this.getSijk(C, a, a, b);
                const s221 = this.getSijk(C, b, b, a);
                const S11 = s111 * s111 * s112 * s112;
                const S22 = Math.pow(s221, 4);
                phir[j] = phir[j] - 0.75 * S11 * phiaa - 0.25 * S22 * phibb;
              }
            } else {
              phir[j] += this.phiMeamSeries(scrn, Z1, Z2, a, b, r, arat);
            }
          }
          if (this.zbl[a][b] === 1) {
            const astar = this.alpha[a][b] * (r / this.re[a][b] - 1.0);
            if (astar <= -3.0) phir[j] = zblPot(r, this.ielt[a], this.ielt[b]);
            else if (astar > -3.0 && astar < -1.0) {
              const frac = fcut(1 - (astar + 1.0) / (-3.0 + 1.0));
              const phizbl = zblPot(r, this.ielt[a], this.ielt[b]);
              phir[j] = frac * phir[j] + (1 - frac) * phizbl;
            }
          }
        }
        this.interpolateMeam(nv2);
        nv2 = nv2 + 1;
      }
    }
  }

  getSijk(C: number, i: number, j: number, k: number): number {
    const x = (C - this.Cmin[i][j][k]) / (this.Cmax[i][j][k] - this.Cmin[i][j][k]);
    return fcut(x);
  }

  /** G_gam (meam_funcs.cpp). Returns NaN-free value; errorflag for unknown ibar is thrown. */
  gGam(gamma: number, ibar: number): number {
    switch (ibar) {
      case 0:
      case 4: {
        const sw = -this.gsmooth / (this.gsmooth + 1);
        if (gamma < sw) {
          if (gamma === 0.0) return 0.0;
          const G = (1 / (this.gsmooth + 1)) * Math.pow(sw / gamma, this.gsmooth);
          return Math.sqrt(G);
        }
        return Math.sqrt(1.0 + gamma);
      }
      case 1: return fmExp(gamma / 2.0);
      case 3: return 2.0 / (1.0 + fmExp(-gamma));
      case -5:
        if (1.0 + gamma >= 0) return Math.sqrt(1.0 + gamma);
        return -Math.sqrt(-1.0 - gamma);
      default: throw new Error(`unknown ibar ${ibar}`);
    }
  }

  dGGam(gamma: number, ibar: number): { G: number; dG: number } {
    switch (ibar) {
      case 0:
      case 4: {
        const sw = -this.gsmooth / (this.gsmooth + 1);
        if (gamma < sw) {
          let G = (1 / (this.gsmooth + 1)) * Math.pow(sw / gamma, this.gsmooth);
          G = Math.sqrt(G);
          return { G, dG: (-this.gsmooth * G) / (2.0 * gamma) };
        }
        const G = Math.sqrt(1.0 + gamma);
        return { G, dG: 1.0 / (2.0 * G) };
      }
      case 1: {
        const G = fmExp(gamma / 2.0);
        return { G, dG: G / 2.0 };
      }
      case 3: {
        const G = 2.0 / (1.0 + fmExp(-gamma));
        return { G, dG: (G * (2.0 - G)) / 2 };
      }
      case -5: {
        if (1.0 + gamma >= 0) {
          const G = Math.sqrt(1.0 + gamma);
          return { G, dG: 1.0 / (2.0 * G) };
        }
        const G = -Math.sqrt(-1.0 - gamma);
        return { G, dG: -1.0 / (2.0 * G) };
      }
      default: return { G: 0.0, dG: 1.0 };
    }
  }

  embedding(A: number, Ec: number, rhobar: number): { F: number; dF: number } {
    const AEc = A * Ec;
    if (rhobar > 0.0) {
      const lrb = Math.log(rhobar);
      return { F: AEc * rhobar * lrb, dF: AEc * (1.0 + lrb) };
    }
    if (this.embLinNeg === 0) return { F: 0.0, dF: 0.0 };
    return { F: -AEc * rhobar, dF: -AEc };
  }

  phiMeamSeries(scrn: number, Z1: number, Z2: number, a: number, b: number, r: number, arat: number): number {
    let phiSum = 0.0;
    if (scrn > 0.0) {
      const b2nn = (-Z2 * scrn) / Z1;
      for (let n = 1; n <= 10; n++) {
        const phiVal = powint(b2nn, n) * this.phiMeam(r * powint(arat, n), a, b);
        if (iszero(phiVal)) break;
        phiSum += phiVal;
      }
    }
    return phiSum;
  }

  /** Pair function phi_meam(r, a, b) of the reference structures (interpolated later). */
  phiMeam(r: number, a: number, b: number): number {
    const Z1 = getZij(this.lattce[a][a]);
    const Z2 = getZij(this.lattce[b][b]);
    const Z12 = getZij(this.lattce[a][b]);
    const d = this.getDensref(r, a, b);
    const { rho01, rho11, rho21, rho31, rho02, rho12, rho22, rho32 } = d;
    if (rho01 <= 1e-14 && rho02 <= 1e-14) return 0.0;
    const lab = this.lattce[a][b];
    let t11av = 0, t21av = 0, t31av = 0, t12av = 0, t22av = 0, t32av = 0;
    let rhobar1: number, rhobar2: number;
    if (lab === 'c11') {
      if (this.ialloy === 2) {
        t11av = this.t1[a]; t12av = this.t1[b];
        t21av = this.t2[a]; t22av = this.t2[b];
        t31av = this.t3[a]; t32av = this.t3[b];
      } else {
        const scalfac = 1.0 / (rho01 + rho02);
        t11av = scalfac * (this.t1[a] * rho01 + this.t1[b] * rho02);
        t12av = t11av;
        t21av = scalfac * (this.t2[a] * rho01 + this.t2[b] * rho02);
        t22av = t21av;
        t31av = scalfac * (this.t3[a] * rho01 + this.t3[b] * rho02);
        t32av = t31av;
      }
    } else {
      // (the m-averages of get_tavref are computed but unused in native phi_meam, so they are not formed here)
      [t11av, t21av, t31av, t12av, t22av, t32av] = this.getTavref(this.t1[a], this.t2[a], this.t3[a], this.t1[b], this.t2[b], this.t3[b], r, a, b, lab);
    }
    if (lab === 'c11') {
      const latta = this.lattce[a][a];
      if (latta === 'dia') {
        rhobar1 = Math.pow(Math.floor(Z12 / 2) * (rho02 + rho01), 2) + t11av * Math.pow(rho12 - rho11, 2)
          + (t21av / 6.0) * Math.pow(rho22 + rho21, 2) + (121.0 / 40.0) * t31av * Math.pow(rho32 - rho31, 2);
        rhobar1 = Math.sqrt(rhobar1);
        rhobar2 = Math.pow(Z12 * rho01, 2) + (2.0 / 3.0) * t21av * Math.pow(rho21, 2);
        rhobar2 = Math.sqrt(rhobar2);
      } else {
        rhobar2 = Math.pow(Math.floor(Z12 / 2) * (rho01 + rho02), 2) + t12av * Math.pow(rho11 - rho12, 2)
          + (t22av / 6.0) * Math.pow(rho21 + rho22, 2) + (121.0 / 40.0) * t32av * Math.pow(rho31 - rho32, 2);
        rhobar2 = Math.sqrt(rhobar2);
        rhobar1 = Math.pow(Z12 * rho02, 2) + (2.0 / 3.0) * t22av * Math.pow(rho22, 2);
        rhobar1 = Math.sqrt(rhobar1);
      }
    } else {
      let G1: number, G2: number, rho0_1 = 0, rho0_2 = 0;
      if (this.mixRefT === 1) {
        if (this.ibar[a] <= 0) G1 = 1.0;
        else {
          const s1 = getShpfcn(this.lattce[a][a], this.stheta[a][a], this.ctheta[a][a]);
          const Gam1 = (s1[0] * t11av + s1[1] * t21av + s1[2] * t31av) / (Z1 * Z1);
          G1 = this.gGam(Gam1, this.ibar[a]);
        }
        if (this.ibar[b] <= 0) G2 = 1.0;
        else {
          const s2 = getShpfcn(this.lattce[b][b], this.stheta[b][b], this.ctheta[b][b]);
          const Gam2 = (s2[0] * t12av + s2[1] * t22av + s2[2] * t32av) / (Z2 * Z2);
          G2 = this.gGam(Gam2, this.ibar[b]);
        }
        rho0_1 = this.rho0[a] * Z1 * G1;
        rho0_2 = this.rho0[b] * Z2 * G2;
      }
      // MS-MEAM: Gamma uses the difference of the full and the m densities (msmeamflag)
      let Gam1 = rho11 + rho21 + rho31 - (d.rho1m1 + d.rho2m1 + d.rho3m1);
      Gam1 = rho01 < 1.0e-14 ? 0.0 : Gam1 / (rho01 * rho01);
      let Gam2 = rho12 + rho22 + rho32 - (d.rho1m2 + d.rho2m2 + d.rho3m2);
      Gam2 = rho02 < 1.0e-14 ? 0.0 : Gam2 / (rho02 * rho02);
      G1 = this.gGam(Gam1, this.ibar[a]);
      G2 = this.gGam(Gam2, this.ibar[b]);
      let rhoBkgd1: number, rhoBkgd2: number;
      if (this.mixRefT === 1) {
        rhoBkgd1 = rho0_1;
        rhoBkgd2 = rho0_2;
      } else if (this.bkgdDyn === 1) {
        rhoBkgd1 = this.rho0[a] * Z1;
        rhoBkgd2 = this.rho0[b] * Z2;
      } else {
        rhoBkgd1 = this.rhoRef[a];
        rhoBkgd2 = this.rhoRef[b];
      }
      rhobar1 = (rho01 / rhoBkgd1) * G1;
      rhobar2 = (rho02 / rhoBkgd2) * G2;
    }
    const F1 = this.embedding(this.A[a], this.Ec[a][a], rhobar1).F;
    const F2 = this.embedding(this.A[b], this.Ec[b][b], rhobar2).F;
    const Eu = erosePot(r, this.re[a][b], this.alpha[a][b], this.Ec[a][b], this.repuls[a][b], this.attrac[a][b], this.ereseForm);
    let phi_m: number;
    if (lab === 'c11') {
      const latta = this.lattce[a][a];
      if (latta === 'dia') {
        const phiaa = this.phiMeam(r, a, a);
        phi_m = (3 * Eu - F2 - 2 * F1 - 5 * phiaa) / Z12;
      } else {
        const phibb = this.phiMeam(r, b, b);
        phi_m = (3 * Eu - F1 - 2 * F2 - 5 * phibb) / Z12;
      }
    } else if (lab === 'l12') {
      let phiaa = this.phiMeam(r, a, a);
      const Z1nn = getZij(this.lattce[a][a]);
      const g = getZij2(this.lattce[a][a], this.Cmin[a][a][a], this.Cmax[a][a][a], this.stheta[a][b]);
      phiaa += this.phiMeamSeries(g.S, Z1nn, g.z, a, a, r, g.a);
      phi_m = Eu / 3.0 - F1 / 4.0 - F2 / 12.0 - phiaa;
    } else if (lab === 'ch4') {
      phi_m = (5 * Eu - F1 - 4 * F2) / 4;
    } else if (lab === 'zig') {
      if (a === b) phi_m = (2 * Eu - F1 - F2) / Z12;
      else {
        const Z1b = getZij(lab);
        const g1 = getZij2b2nn(lab, this.Cmin[a][a][b], this.Cmax[a][a][b]);
        const b11s = (-g1.z / Z1b) * g1.S;
        const g2 = getZij2b2nn(lab, this.Cmin[b][b][a], this.Cmax[b][b][a]);
        const b22s = (-g2.z / Z1b) * g2.S;
        const phiaa = this.phiMeam(2.0 * this.stheta[a][b] * r, a, a);
        const phibb = this.phiMeam(2.0 * this.stheta[a][b] * r, b, b);
        phi_m = (2.0 * Eu - F1 - F2 + phiaa * b11s + phibb * b22s) / Z12;
      }
    } else if (lab === 'tri') {
      if (a === b) phi_m = (3.0 * Eu - 2.0 * F1 - F2) / Z12;
      else {
        const Z1b = getZij(lab);
        const g1 = getZij2b2nn(lab, this.Cmin[a][a][b], this.Cmax[a][a][b]);
        const b11s = (-g1.z / Z1b) * g1.S;
        const phiaa = this.phiMeam(2.0 * this.stheta[a][b] * r, a, a);
        phi_m = (3.0 * Eu - 2.0 * F1 - F2 + phiaa * b11s) / Z12;
      }
    } else {
      phi_m = (2 * Eu - F1 - F2) / Z12;
    }
    if (iszero(r)) phi_m = 0.0;
    return phi_m;
  }

  /** get_tavref for the lattices of the MEAM pairs (all-opposite-type or L12 averaging). */
  getTavref(t11: number, t21: number, t31: number, t12: number, t22: number, t32: number, r: number, a: number, b: number, latt: Lattice): number[] {
    if (this.ialloy === 2) return [t11, t21, t31, t12, t22, t32];
    switch (latt) {
      case 'fcc': case 'bcc': case 'dia': case 'dia3': case 'hcp': case 'b1': case 'dim': case 'b2': case 'ch4':
      case 'lin': case 'zig': case 'tri': case 'sc':
        return [t12, t22, t32, t11, t21, t31];
      default: {
        const a1 = r / this.re[a][a] - 1.0;
        const a2 = r / this.re[b][b] - 1.0;
        const rhoa01 = this.rho0[a] * fmExp(-this.beta0[a] * a1);
        const rhoa02 = this.rho0[b] * fmExp(-this.beta0[b] * a2);
        if (latt === 'l12') {
          const rho01 = 8 * rhoa01 + 4 * rhoa02;
          const t12av = t11, t22av = t21, t32av = t31;
          let t11av: number, t21av: number, t31av: number;
          if (rho01 > 0.0) {
            t11av = (8 * t11 * rhoa01 + 4 * t12 * rhoa02) / rho01;
            t21av = (8 * t21 * rhoa01 + 4 * t22 * rhoa02) / rho01;
            t31av = (8 * t31 * rhoa01 + 4 * t32 * rhoa02) / rho01;
          } else {
            t11av = (2.0 * t11 + t12) / 3.0;
            t21av = (2.0 * t21 + t22) / 3.0;
            t31av = (2.0 * t31 + t32) / 3.0;
          }
          return [t11av, t21av, t31av, t12av, t22av, t32av];
        }
        return [0, 0, 0, 0, 0, 0];
      }
    }
  }

  /** get_densref: the reference densities of pair (a, b) at distance r (msmeam: including the m densities). */
  getDensref(r: number, a: number, b: number): DensRef {
    const a1 = r / this.re[a][a] - 1.0;
    const a2 = r / this.re[b][b] - 1.0;
    const rhoa01 = this.rho0[a] * fmExp(-this.beta0[a] * a1);
    const rhoa11 = this.rho0[a] * this.t1[a] * fmExp(-this.beta1[a] * a1);
    const rhoa21 = this.rho0[a] * this.t2[a] * fmExp(-this.beta2[a] * a1);
    const rhoa31 = this.rho0[a] * this.t3[a] * fmExp(-this.beta3[a] * a1);
    const rhoa02 = this.rho0[b] * fmExp(-this.beta0[b] * a2);
    const rhoa12 = this.rho0[b] * this.t1[b] * fmExp(-this.beta1[b] * a2);
    const rhoa22 = this.rho0[b] * this.t2[b] * fmExp(-this.beta2[b] * a2);
    const rhoa32 = this.rho0[b] * this.t3[b] * fmExp(-this.beta3[b] * a2);
    const rhoa1m1 = this.rho0[a] * this.t1m[a] * fmExp(-this.beta1m[a] * a1);
    const rhoa2m1 = this.rho0[a] * this.t2m[a] * fmExp(-this.beta2m[a] * a1);
    const rhoa3m1 = this.rho0[a] * this.t3m[a] * fmExp(-this.beta3m[a] * a1);
    const rhoa1m2 = this.rho0[b] * this.t1m[b] * fmExp(-this.beta1m[b] * a2);
    const rhoa2m2 = this.rho0[b] * this.t2m[b] * fmExp(-this.beta2m[b] * a2);
    const rhoa3m2 = this.rho0[b] * this.t3m[b] * fmExp(-this.beta3m[b] * a2);
    const lat = this.lattce[a][b];
    const d: DensRef = {
      rho01: 0, rho11: 0, rho21: 0, rho31: 0, rho02: 0, rho12: 0, rho22: 0, rho32: 0,
      rho1m1: 0, rho2m1: 0, rho3m1: 0, rho1m2: 0, rho2m2: 0, rho3m2: 0,
    };
    const s = [0, 0, 0];
    switch (lat) {
      case 'fcc': d.rho01 = 12.0 * rhoa02; d.rho02 = 12.0 * rhoa01; break;
      case 'bcc': d.rho01 = 8.0 * rhoa02; d.rho02 = 8.0 * rhoa01; break;
      case 'b1': case 'sc': d.rho01 = 6.0 * rhoa02; d.rho02 = 6.0 * rhoa01; break;
      case 'dia': case 'dia3':
        d.rho01 = 4.0 * rhoa02; d.rho02 = 4.0 * rhoa01;
        d.rho31 = (32.0 / 9.0) * rhoa32 * rhoa32;
        d.rho32 = (32.0 / 9.0) * rhoa31 * rhoa31;
        d.rho3m1 = (32.0 / 9.0) * rhoa3m2 * rhoa3m2;
        d.rho3m2 = (32.0 / 9.0) * rhoa3m1 * rhoa3m1;
        break;
      case 'hcp':
        d.rho01 = 12 * rhoa02; d.rho02 = 12 * rhoa01;
        d.rho31 = (1.0 / 3.0) * rhoa32 * rhoa32;
        d.rho32 = (1.0 / 3.0) * rhoa31 * rhoa31;
        d.rho3m1 = (1.0 / 3.0) * rhoa3m2 * rhoa3m2;
        d.rho3m2 = (1.0 / 3.0) * rhoa3m1 * rhoa3m1;
        break;
      case 'dim': {
        const sh = getShpfcn('dim', 0, 0);
        s[0] = sh[0]; s[1] = sh[1]; s[2] = sh[2];
        d.rho01 = rhoa02; d.rho02 = rhoa01;
        d.rho11 = s[0] * rhoa12 * rhoa12; d.rho12 = s[0] * rhoa11 * rhoa11;
        d.rho21 = s[1] * rhoa22 * rhoa22; d.rho22 = s[1] * rhoa21 * rhoa21;
        d.rho31 = s[2] * rhoa32 * rhoa32; d.rho32 = s[2] * rhoa31 * rhoa31;
        d.rho1m1 = s[0] * rhoa1m2 * rhoa1m2; d.rho1m2 = s[0] * rhoa1m1 * rhoa1m1;
        d.rho2m1 = s[1] * rhoa2m2 * rhoa2m2; d.rho2m2 = s[1] * rhoa2m1 * rhoa2m1;
        d.rho3m1 = s[2] * rhoa3m2 * rhoa3m2; d.rho3m2 = s[2] * rhoa3m1 * rhoa3m1;
        break;
      }
      case 'c11':
        d.rho01 = rhoa01; d.rho02 = rhoa02;
        d.rho11 = rhoa11; d.rho12 = rhoa12;
        d.rho21 = rhoa21; d.rho22 = rhoa22;
        d.rho31 = rhoa31; d.rho32 = rhoa32;
        d.rho1m1 = rhoa1m1; d.rho1m2 = rhoa1m2;
        d.rho2m1 = rhoa2m1; d.rho2m2 = rhoa2m2;
        d.rho3m1 = rhoa3m1; d.rho3m2 = rhoa3m2;
        break;
      case 'l12':
        d.rho01 = 8 * rhoa01 + 4 * rhoa02;
        d.rho02 = 12 * rhoa01;
        // msmeam: the t's are already in the rho's (ialloy not used)
        d.rho21 = (8 / 3) * (rhoa21 - rhoa22) * (rhoa21 - rhoa22);
        d.rho2m1 = (8 / 3) * (rhoa2m1 - rhoa2m2) * (rhoa2m1 - rhoa2m2);
        break;
      case 'b2': d.rho01 = 8.0 * rhoa02; d.rho02 = 8.0 * rhoa01; break;
      default:
        throw new Error(`MS-MEAM reference lattice '${lat}' is not supported by the engine port`);
    }
    if (this.nn2[a][b] === 1) {
      const g = getZij2(lat, this.Cmin[a][a][b], this.Cmax[a][a][b], this.stheta[a][b]);
      const arat = g.a;
      const scrn = g.S;
      const ar1 = arat * r / this.re[a][a] - 1.0;
      const ar2 = arat * r / this.re[b][b] - 1.0;
      const rhoa01nn = this.rho0[a] * fmExp(-this.beta0[a] * ar1);
      const rhoa02nn = this.rho0[b] * fmExp(-this.beta0[b] * ar2);
      if (lat === 'l12') {
        const C = 1.0;
        const s111 = this.getSijk(C, a, a, a);
        const s112 = this.getSijk(C, a, a, b);
        const s221 = this.getSijk(C, b, b, a);
        const S11 = s111 * s111 * s112 * s112;
        const S22 = s221 * s221 * s221 * s221;
        d.rho01 = d.rho01 + 6 * S11 * rhoa01nn;
        d.rho02 = d.rho02 + 6 * S22 * rhoa02nn;
      } else {
        d.rho01 = d.rho01 + g.z * scrn * rhoa01nn;
        const g2 = getZij2(lat, this.Cmin[b][b][a], this.Cmax[b][b][a], this.stheta[a][b]);
        d.rho02 = d.rho02 + g2.z * g2.S * rhoa02nn;
      }
    }
    return d;
  }

  computeReferenceDensity(): void {
    for (let a = 0; a < this.neltypes; a++) {
      const Z = getZij(this.lattce[a][a]);
      let Gbar: number;
      if (this.ibar[a] <= 0) Gbar = 1.0;
      else {
        const shp = getShpfcn(this.lattce[a][a], this.stheta[a][a], this.ctheta[a][a]);
        const gam = (this.t1[a] * shp[0] + this.t2[a] * shp[1] + this.t3[a] * shp[2]) / (Z * Z);
        Gbar = this.gGam(gam, this.ibar[a]);
      }
      let rho0 = this.rho0[a] * Z;
      if (this.nn2[a][a] === 1) {
        const g = getZij2(this.lattce[a][a], this.Cmin[a][a][a], this.Cmax[a][a][a], this.stheta[a][a]);
        const rho0_2nn = this.rho0[a] * fmExp(-this.beta0[a] * (g.a - 1));
        rho0 = rho0 + g.z * rho0_2nn * g.S;
      }
      this.rhoRef[a] = rho0 * Gbar;
    }
  }

  interpolateMeam(ind: number): void {
    const nrar = this.nr;
    const drar = this.dr;
    const ph = this.phir[ind], pa = this.phirar[ind], p1 = this.phirar1[ind], p2 = this.phirar2[ind];
    const p3 = this.phirar3[ind], p4 = this.phirar4[ind], p5 = this.phirar5[ind], p6 = this.phirar6[ind];
    for (let j = 0; j < nrar; j++) pa[j] = ph[j];
    p1[0] = pa[1] - pa[0];
    p1[1] = 0.5 * (pa[2] - pa[0]);
    p1[nrar - 2] = 0.5 * (pa[nrar - 1] - pa[nrar - 3]);
    p1[nrar - 1] = 0.0;
    for (let j = 2; j < nrar - 2; j++) {
      p1[j] = (pa[j - 2] - pa[j + 2] + 8.0 * (pa[j + 1] - pa[j - 1])) / 12.;
    }
    for (let j = 0; j < nrar - 1; j++) {
      p2[j] = 3.0 * (pa[j + 1] - pa[j]) - 2.0 * p1[j] - p1[j + 1];
      p3[j] = p1[j] + p1[j + 1] - 2.0 * (pa[j + 1] - pa[j]);
    }
    p2[nrar - 1] = 0.0;
    p3[nrar - 1] = 0.0;
    for (let j = 0; j < nrar; j++) {
      p4[j] = p1[j] / drar;
      p5[j] = (2.0 * p2[j]) / drar;
      p6[j] = (3.0 * p3[j]) / drar;
    }
  }
}

export interface DensRef {
  rho01: number; rho11: number; rho21: number; rho31: number;
  rho02: number; rho12: number; rho22: number; rho32: number;
  rho1m1: number; rho2m1: number; rho3m1: number;
  rho1m2: number; rho2m2: number; rho3m2: number;
}

/** LAMMPS lattice enum codes used by the lattce keyword (MEAM::lattice_t order). */
const LATTICE_BY_CODE: Lattice[] = ['fcc', 'bcc', 'hcp', 'dim', 'dia', 'dia3', 'b1', 'c11', 'l12', 'b2', 'ch4', 'lin', 'zig', 'tri', 'sc'];
export const latticeCode = (l: Lattice): number => LATTICE_BY_CODE.indexOf(l);
