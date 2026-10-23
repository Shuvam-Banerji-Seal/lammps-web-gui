/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-POD/eapod.cpp and src/ML-POD/eapod.h (class EAPOD: read_pod_file, read_model_coeff_file,
 * eigenvaluedecomposition, init2body, init3body, init4body, radialbasis, angularbasis, radialangularbasis,
 * twobodydesc, threebodydesc, fourbodydesc, the force coefficients and peratomenergyforce2), at commit b332f28.
 * Monomial and 4-body term tables: tables.ts (the same commit's data).
 * Authors credited in the C++ file header: Ngoc Cuong Nguyen (MIT).
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Evaluation only (pair_style pod). Not ported: fitting (fitpod), the 5-, 6- and 7-body terms, the
 * environment-adaptive model (number_of_environment_clusters > 1) and onebody 0; each is a StyleError.
 */
import { StyleError } from '../../engine/force/types';
import { POLY3, POLY4, NS4 } from './tables';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { sinFma as libmSinFma } from '../libm/sincos_fma';
import { exp as libmExp } from '../libm/exp';

/** npa of eapod.cpp: number of monomials of each degree. */
const NPA = [0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455];
/** nb of eapod.cpp: number of angular basis functions of each four-body degree. */
const NB = [1, 2, 4, 7, 11, 16, 23];
/** The Bessel parameters are not read from the file (eapod.cpp constructor). */
const NBESSELPARS = 3;
const BESSEL_PARAMS = [1e-3, 2.0, 4.0];

/** Keywords of the POD parameter file (eapod.cpp read_pod_file). */
export interface PODParams {
  species: string[];
  rin: number;
  rcut: number;
  nClusters: number;
  nComponents: number;
  besseldegree: number;
  inversedegree: number;
  onebody: number;
  nrbf2: number;
  nrbf3: number;
  P3: number;
  nrbf4: number;
  P4: number;
  nrbf33: number;
  P33: number;
  nrbf34: number;
  P34: number;
  nrbf44: number;
  P44: number;
}

const tokens = (line: string): string[] => {
  const hash = line.indexOf('#');
  const body = hash >= 0 ? line.slice(0, hash) : line;
  return body.split(/[\s"']+/).filter((w) => w.length > 0);
};

const intArg = (w: string, what: string): number => {
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`pod parameter file: ${what} needs an integer, got '${w}'`);
  return Number(w);
};

const numArg = (w: string, what: string): number => {
  const v = Number(w);
  if (w.length === 0 || !Number.isFinite(v)) throw new StyleError(`pod parameter file: ${what} needs a number, got '${w}'`);
  return v;
};

/** Parses the POD parameter file (text of the file named by pair_coeff). */
/**
 * The message of a POD parameter or coefficient file that cannot be opened (eapod.cpp read_pod_file and
 * read_model_coeff_file at b332f28, whose format string drops the system error: "Cannot open POD coefficient file {}: ").
 */
export const POD_OPENER = (name: string): string => `Cannot open POD coefficient file ${name}: `;

export const parsePODParams = (text: string): PODParams => {
  const p: PODParams = {
    species: [], rin: 0.5, rcut: 5.0, nClusters: 1, nComponents: 1, besseldegree: 4, inversedegree: 8, onebody: 1,
    nrbf2: 8, nrbf3: 6, P3: 4, nrbf4: 4, P4: 3, nrbf33: 0, P33: 0, nrbf34: 0, P34: 0, nrbf44: 0, P44: 0,
  };
  for (const line of text.split(/\r?\n/)) {
    const w = tokens(line);
    if (w.length === 0) continue;
    const key = w[0];
    if (key === 'species') {
      p.species = w.slice(1);
      continue;
    }
    if (key === 'pbc') {
      if (w.length !== 4) throw new StyleError('Improper POD file (pbc needs three values)');
      continue;
    }
    if (w.length !== 2) throw new StyleError(`Improper POD file: keyword '${key}' needs one value`);
    const v = w[1];
    switch (key) {
      case 'rin': p.rin = numArg(v, key); break;
      case 'rcut': p.rcut = numArg(v, key); break;
      case 'number_of_environment_clusters': p.nClusters = intArg(v, key); break;
      case 'number_of_principal_components': p.nComponents = intArg(v, key); break;
      case 'bessel_polynomial_degree': p.besseldegree = intArg(v, key); break;
      case 'inverse_polynomial_degree': p.inversedegree = intArg(v, key); break;
      case 'onebody': p.onebody = intArg(v, key); break;
      case 'twobody_number_radial_basis_functions': p.nrbf2 = intArg(v, key); break;
      case 'threebody_number_radial_basis_functions': p.nrbf3 = intArg(v, key); break;
      case 'threebody_angular_degree': p.P3 = intArg(v, key); break;
      case 'fourbody_number_radial_basis_functions': p.nrbf4 = intArg(v, key); break;
      case 'fourbody_angular_degree': p.P4 = intArg(v, key); break;
      case 'fivebody_number_radial_basis_functions': p.nrbf33 = intArg(v, key); break;
      case 'fivebody_angular_degree': p.P33 = intArg(v, key); break;
      case 'sixbody_number_radial_basis_functions': p.nrbf34 = intArg(v, key); break;
      case 'sixbody_angular_degree': p.P34 = intArg(v, key); break;
      case 'sevenbody_number_radial_basis_functions': p.nrbf44 = intArg(v, key); break;
      case 'sevenbody_angular_degree': p.P44 = intArg(v, key); break;
      default: break; // unknown two-word keywords are ignored, as native LAMMPS does
    }
  }
  return p;
};

/** Model coefficients, projection and centroid arrays of the coefficient file (eapod.cpp read_model_coeff_file). */
export interface PODCoeffFile {
  coeff: Float64Array;
  proj: Float64Array;
  centroids: Float64Array;
}

/** Parses the coefficient file: a header "model_coefficients: N nproj ncent", then one number per line. */
export const parsePODCoeffFile = (text: string): PODCoeffFile => {
  const lines = text.split(/\r?\n/);
  let k = 0;
  while (k < lines.length && tokens(lines[k]).length === 0) k++;
  if (k >= lines.length) throw new StyleError('Incorrect format in POD coefficient file');
  const head = tokens(lines[k++]);
  if (head.length !== 4) throw new StyleError('Incorrect format in POD coefficient file');
  const nCoeff = intArg(head[1], 'number of coefficients');
  const nProj = intArg(head[2], 'number of projection elements');
  const nCent = intArg(head[3], 'number of centroid elements');
  const readBlock = (n: number): Float64Array => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const w = k < lines.length ? tokens(lines[k++]) : [];
      if (w.length !== 1) throw new StyleError('Incorrect format in model coefficient file');
      out[i] = numArg(w[0], 'model coefficient');
    }
    return out;
  };
  const coeff = readBlock(nCoeff);
  const proj = readBlock(nProj);
  const centroids = readBlock(nCent);
  return { coeff, proj, centroids };
};

/** Symmetric eigen decomposition by cyclic Jacobi rotations (in place on a copy). Returns eigenvalues and
 *  eigenvectors, the vectors column-major (component i of vector j at i + n*j). */
export const jacobiEigen = (matrix: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } => {
  const a = Float64Array.from(matrix);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i + n * i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p + n * q] * a[p + n * q];
    if (off === 0) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p + n * q];
        if (apq === 0) continue;
        const theta = (a[q + n * q] - a[p + n * p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let r = 0; r < n; r++) {
          const arp = a[r + n * p], arq = a[r + n * q];
          a[r + n * p] = c * arp - s * arq;
          a[r + n * q] = s * arp + c * arq;
        }
        for (let r = 0; r < n; r++) {
          const apr = a[p + n * r], aqr = a[q + n * r];
          a[p + n * r] = c * apr - s * aqr;
          a[q + n * r] = s * apr + c * aqr;
        }
        for (let r = 0; r < n; r++) {
          const vrp = v[r + n * p], vrq = v[r + n * q];
          v[r + n * p] = c * vrp - s * vrq;
          v[r + n * q] = s * vrp + c * vrq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = a[i + n * i];
  return { values, vectors: v };
};

/** pow for integer exponents, binary powering as in LAMMPS math_special powint. */
const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

/** Per-call scratch arrays, grown to the largest neighbour count seen. */
interface Scratch {
  n: number;
  rbft: Float64Array; rbfxt: Float64Array; rbfyt: Float64Array; rbfzt: Float64Array;
  rbf: Float64Array; rbfx: Float64Array; rbfy: Float64Array; rbfz: Float64Array;
  abf: Float64Array; abfx: Float64Array; abfy: Float64Array; abfz: Float64Array;
  ux: Float64Array; uy: Float64Array; uz: Float64Array;
}

/** The POD model of one element set: the descriptors, their coefficients, and energy and forces of one atom. */
export class PODModel {
  readonly nelements: number;
  readonly rin: number;
  readonly rcut: number;
  readonly besseldegree: number;
  readonly inversedegree: number;
  readonly ns: number;
  readonly nrbf2: number;
  readonly nrbf3: number;
  readonly nrbf4: number;
  readonly nrbfmax: number;
  readonly nabf3: number;
  readonly nabf4: number;
  readonly K3: number;
  readonly Q4: number;
  readonly nl2: number;
  readonly nl3: number;
  readonly nl4: number;
  readonly Mdesc: number;
  readonly nCoeffPerElement: number;
  readonly nCoeffAll: number;
  readonly coeff: Float64Array;
  /** Projection of the radial basis onto the eigenvectors of the snapshot matrix (ns x ns, column-major). */
  readonly Phi: Float64Array;
  /** pn3: first monomial index of each angular degree; pq3: parent index and direction; pc3: monomial coefficient. */
  private readonly pn3: Int32Array;
  private readonly pq3: Int32Array;
  private readonly pc3: Float64Array;
  /** pa4: first term of each four-body angular function; pb4: three monomial indices per term; pc4: coefficient. */
  private readonly pa4: Int32Array;
  private readonly pb4: Int32Array;
  private readonly pc4: Int32Array;
  private readonly elemindex: Int32Array;
  private scratchData: Scratch | null = null;
  private readonly sumU: Float64Array;
  private readonly forcecoeff: Float64Array;
  private readonly bd: Float64Array;
  private readonly tm: Float64Array;

  constructor(params: PODParams, coeffFile: PODCoeffFile) {
    if (params.species.length === 0) throw new StyleError('pod parameter file: no species given');
    if (params.nClusters > 1) {
      throw new StyleError('pair style pod: number_of_environment_clusters > 1 (environment-adaptive POD models) is not ported');
    }
    if (params.onebody !== 1) throw new StyleError('pair style pod: onebody other than 1 is not ported');
    for (const [name, n] of [['fivebody', params.nrbf33], ['sixbody', params.nrbf34], ['sevenbody', params.nrbf44]] as const) {
      if (n > 0) throw new StyleError(`pair style pod: ${name} terms (${name}_number_radial_basis_functions > 0) are not ported`);
    }
    if (params.nrbf3 < params.nrbf4) throw new StyleError('number of four-body radial basis functions must be equal or less than number of three-body radial basis functions');
    if (params.P3 < params.P4) throw new StyleError('four-body angular degree must be equal or less than three-body angular degree');
    if (params.P3 > 12) throw new StyleError('three-body angular degree must be equal or less than 12');
    if (params.P4 > 6) throw new StyleError('four-body angular degree must be equal or less than 6');

    const Ne = params.species.length;
    this.nelements = Ne;
    this.rin = params.rin;
    this.rcut = params.rcut;
    this.besseldegree = params.besseldegree;
    this.inversedegree = params.inversedegree;
    this.nrbf2 = params.nrbf2;
    this.nrbf3 = params.nrbf3;
    this.nrbf4 = params.nrbf4;
    this.nrbfmax = Math.max(params.nrbf2, params.nrbf3, params.nrbf4);
    this.ns = NBESSELPARS * this.besseldegree + this.inversedegree;
    if (this.nrbfmax > this.ns) throw new StyleError('pair style pod: more radial basis functions than the radial basis has terms');

    // angular tables (init3body, init4body)
    this.nabf3 = params.P3 + 1;
    this.K3 = NPA[this.nabf3];
    this.pn3 = new Int32Array(this.nabf3 + 1);
    for (let i = 0; i <= this.nabf3; i++) this.pn3[i] = NPA[i];
    this.pq3 = new Int32Array(this.K3 * 2);
    this.pc3 = new Float64Array(this.K3);
    for (let i = 0; i < this.K3; i++) {
      this.pq3[i] = POLY3[i][3];
      this.pq3[i + this.K3] = POLY3[i][4];
      this.pc3[i] = POLY3[i][5];
    }
    this.nabf4 = NB[params.P4];
    this.Q4 = NS4[this.nabf4];
    this.pa4 = new Int32Array(this.nabf4 + 1);
    for (let i = 0; i <= this.nabf4; i++) this.pa4[i] = NS4[i];
    this.pb4 = new Int32Array(this.Q4 * 3);
    this.pc4 = new Int32Array(this.Q4);
    for (let i = 0; i < this.Q4; i++) {
      this.pb4[i] = POLY4[i][0] - 1;
      this.pb4[i + this.Q4] = POLY4[i][1] - 1;
      this.pb4[i + 2 * this.Q4] = POLY4[i][2] - 1;
      this.pc4[i] = POLY4[i][3];
    }

    // element pairs and triples (read_pod_file)
    this.elemindex = new Int32Array(Ne * Ne);
    let kk = 0;
    for (let i1 = 0; i1 < Ne; i1++) {
      for (let i2 = i1; i2 < Ne; i2++) {
        this.elemindex[i2 + Ne * i1] = kk;
        this.elemindex[i1 + Ne * i2] = kk;
        kk += 1;
      }
    }

    // descriptor sizes (read_pod_file); onebody = 1 gives one constant per element
    const nl1 = 1;
    this.nl2 = this.nrbf2 * Ne;
    this.nl3 = this.nabf3 * this.nrbf3 * Ne * (Ne + 1) / 2;
    this.nl4 = this.nabf4 * this.nrbf4 * Ne * (Ne + 1) * (Ne + 2) / 6;
    this.Mdesc = this.nl2 + this.nl3 + this.nl4;
    this.nCoeffPerElement = nl1 + this.Mdesc;
    this.nCoeffAll = this.nCoeffPerElement * Ne;

    if (coeffFile.coeff.length !== this.nCoeffAll) {
      throw new StyleError('number of coefficients in the coefficient file is not correct');
    }
    this.coeff = coeffFile.coeff;

    this.Phi = this.eigenBasis();

    this.sumU = new Float64Array(Ne * this.K3 * this.nrbf3);
    this.forcecoeff = new Float64Array(Ne * this.K3 * this.nrbf3);
    this.bd = new Float64Array(this.Mdesc);
    this.tm = new Float64Array(4 * this.K3);
  }

  /**
   * Eigenvectors of the snapshot matrix S^T S / N with S the radial basis on 2000 points of [rin, rcut]
   * (eapod.cpp eigenvaluedecomposition, init2body): ordered from the largest eigenvalue, normalised in the
   * trapezoid-weighted L2 norm, and signed so that each diagonal entry is positive.
   */
  private eigenBasis(): Float64Array {
    const N = 2000;
    const ns = this.ns;
    const rin = this.rin, rcut = this.rcut;
    const xij = new Float64Array(N);
    for (let i = 0; i < N; i++) xij[i] = (rin + 1e-6) + (rcut - rin - 1e-6) * (i * 1.0 / (N - 1));
    const S = new Float64Array(N * ns);
    this.snapshots(S, xij, N);
    const A = new Float64Array(ns * ns);
    for (let i = 0; i < ns; i++) {
      for (let j = 0; j < ns; j++) {
        let s = 0;
        for (let n = 0; n < N; n++) s += S[n + N * i] * S[n + N * j];
        A[i + ns * j] = s;
      }
    }
    for (let i = 0; i < ns * ns; i++) A[i] = A[i] * (1.0 / N);
    const { values, vectors } = jacobiEigen(A, ns);
    const order = Array.from({ length: ns }, (_, i) => i).sort((a, b) => values[a] - values[b]);
    // Phi column (ns-1-j) = eigenvector order[j] (ascending eigenvalues, reversed to descending)
    const Phi = new Float64Array(ns * ns);
    for (let j = 0; j < ns; j++) {
      const src = order[j];
      for (let i = 0; i < ns; i++) Phi[i + ns * (ns - j - 1)] = vectors[i + ns * src];
    }
    // Q = S Phi, and the trapezoid area of each column
    const h = new Float64Array(N - 1);
    for (let i = 0; i < N - 1; i++) h[i] = xij[i + 1] - xij[i];
    for (let m = 0; m < ns; m++) {
      const q = new Float64Array(N);
      for (let n = 0; n < N; n++) {
        let s = 0;
        for (let k = 0; k < ns; k++) s += S[n + N * k] * Phi[k + ns * m];
        q[n] = s;
      }
      let area = 0;
      for (let i = 0; i < N - 1; i++) area += 0.5 * h[i] * (q[i] * q[i] + q[i + 1] * q[i + 1]);
      const scale = Math.sqrt(area);
      for (let i = 0; i < ns; i++) Phi[i + ns * m] = Phi[i + ns * m] / scale;
    }
    for (let m = 0; m < ns; m++) {
      if (Phi[m + ns * m] < 0) {
        for (let i = 0; i < ns; i++) Phi[i + ns * m] = -Phi[i + ns * m];
      }
    }
    return Phi;
  }

  /** Snapshot matrix S (N x ns, column-major) of the radial basis at the points xij (eapod.cpp snapshots). */
  private snapshots(rbf: Float64Array, xij: Float64Array, N: number): void {
    const rmax = this.rcut - this.rin;
    for (let n = 0; n < N; n++) {
      const dij = xij[n];
      const r = dij - this.rin;
      const y = r / rmax;
      const y2 = y * y;
      const y3 = 1.0 - y2 * y;
      const y4 = y3 * y3 + 1e-6;
      const y5 = Math.sqrt(y4);
      const y6 = libmExp(-1.0 / y5);
      const fcut = y6 / libmExp(-1.0);
      for (let j = 0; j < NBESSELPARS; j++) {
        let alpha = BESSEL_PARAMS[j];
        if (Math.abs(alpha) <= 1.0e-6) alpha = 1e-3;
        const x = (1.0 - libmExp(-alpha * r / rmax)) / (1.0 - libmExp(-alpha));
        for (let i = 0; i < this.besseldegree; i++) {
          const a = (i + 1) * Math.PI;
          const b = Math.sqrt(2.0 / rmax) / (i + 1);
          rbf[n + N * i + N * this.besseldegree * j] = b * fcut * libmSinFma(a * x) / r;
        }
      }
      for (let i = 0; i < this.inversedegree; i++) {
        const p = this.besseldegree * NBESSELPARS + i;
        rbf[n + N * p] = fcut / powint(dij, i + 1);
      }
    }
  }

  private makeScratch(n: number): Scratch {
    const ns = this.ns, K3 = this.K3, nrbf3 = this.nrbf3;
    const nrbfmax = this.nrbfmax;
    return {
      n,
      rbft: new Float64Array(n * ns), rbfxt: new Float64Array(n * ns), rbfyt: new Float64Array(n * ns), rbfzt: new Float64Array(n * ns),
      rbf: new Float64Array(n * nrbfmax), rbfx: new Float64Array(n * nrbfmax), rbfy: new Float64Array(n * nrbfmax), rbfz: new Float64Array(n * nrbfmax),
      abf: new Float64Array(n * K3), abfx: new Float64Array(n * K3), abfy: new Float64Array(n * K3), abfz: new Float64Array(n * K3),
      ux: new Float64Array(n * K3 * nrbf3), uy: new Float64Array(n * K3 * nrbf3), uz: new Float64Array(n * K3 * nrbf3),
    };
  }

  private scratch(n: number): Scratch {
    if (!this.scratchData || this.scratchData.n < n) this.scratchData = this.makeScratch(Math.max(n, 16));
    return this.scratchData;
  }

  /** Radial basis functions and their derivatives with respect to r_ij (eapod.cpp radialbasis). */
  private radialbasis(s: Scratch, rij: Float64Array, nj: number): void {
    const rmax = this.rcut - this.rin;
    const bd = this.besseldegree, id = this.inversedegree;
    for (let n = 0; n < nj; n++) {
      const xij1 = rij[3 * n], xij2 = rij[3 * n + 1], xij3 = rij[3 * n + 2];
      const dij = Math.sqrt(xij1 * xij1 + xij2 * xij2 + xij3 * xij3);
      const dr1 = xij1 / dij, dr2 = xij2 / dij, dr3 = xij3 / dij;
      const r = dij - this.rin;
      const y = r / rmax;
      const y2 = y * y;
      const y3 = 1.0 - y2 * y;
      const y4 = y3 * y3 + 1e-6;
      const y5 = Math.sqrt(y4);
      const y6 = libmExp(-1.0 / y5);
      const y7 = y4 * Math.sqrt(y4);
      const fcut = y6 / libmExp(-1.0);
      const dfcut = ((3.0 / (rmax * libmExp(-1.0))) * y2 * y6 * (y * y2 - 1.0)) / y7;
      const f1 = fcut / r;
      const f2 = f1 / r;
      const df1 = dfcut / r;
      for (let j = 0; j < NBESSELPARS; j++) {
        const alpha = BESSEL_PARAMS[j];
        const t1 = 1.0 - libmExp(-alpha);
        const t2 = libmExp(-alpha * r / rmax);
        const x = (1.0 - t2) / t1;
        const dx = (alpha / rmax) * t2 / t1;
        for (let i = 0; i < bd; i++) {
          const a = (i + 1) * Math.PI;
          const b = Math.sqrt(2.0 / rmax) / (i + 1);
          const af1 = a * f1;
          const sinax = libmSin(a * x);
          const nij = n + nj * (i + bd * j);
          s.rbft[nij] = b * f1 * sinax;
          const drbfdr = b * (df1 * sinax - f2 * sinax + af1 * libmCos(a * x) * dx);
          s.rbfxt[nij] = drbfdr * dr1;
          s.rbfyt[nij] = drbfdr * dr2;
          s.rbfzt[nij] = drbfdr * dr3;
        }
      }
      const f1i = fcut / dij;
      for (let i = 0; i < id; i++) {
        const p = bd * NBESSELPARS + i;
        const nij = n + nj * p;
        const a = powint(dij, i + 1);
        s.rbft[nij] = fcut / a;
        const drbfdr = (dfcut - (i + 1.0) * f1i) / a;
        s.rbfxt[nij] = drbfdr * dr1;
        s.rbfyt[nij] = drbfdr * dr2;
        s.rbfzt[nij] = drbfdr * dr3;
      }
    }
  }

  /** Projects the radial basis onto Phi: rbf = rbft Phi (first nrbfmax columns). */
  private projectRadial(s: Scratch, nj: number): void {
    const ns = this.ns, Phi = this.Phi;
    for (let m = 0; m < this.nrbfmax; m++) {
      for (let n = 0; n < nj; n++) {
        let v = 0, vx = 0, vy = 0, vz = 0;
        for (let k = 0; k < ns; k++) {
          const p = Phi[k + ns * m];
          const idx = n + nj * k;
          v += s.rbft[idx] * p;
          vx += s.rbfxt[idx] * p;
          vy += s.rbfyt[idx] * p;
          vz += s.rbfzt[idx] * p;
        }
        const o = n + nj * m;
        s.rbf[o] = v;
        s.rbfx[o] = vx;
        s.rbfy[o] = vy;
        s.rbfz[o] = vz;
      }
    }
  }

  /** Angular basis functions of the unit vectors and their derivatives (eapod.cpp angularbasis). */
  private angularbasis(s: Scratch, rij: Float64Array, nj: number): void {
    const K = this.K3;
    const pq = this.pq3, tm = this.tm;
    const tmu = new Float64Array(K), tmv = new Float64Array(K), tmw = new Float64Array(K);
    for (let j = 0; j < nj; j++) {
      const x = rij[3 * j], y = rij[3 * j + 1], z = rij[3 * j + 2];
      const xx = x * x, yy = y * y, zz = z * z;
      const xy = x * y, xz = x * z, yz = y * z;
      const dij = Math.sqrt(xx + yy + zz);
      const u = x / dij, v = y / dij, w = z / dij;
      const dij3 = dij * dij * dij;
      const dudx = (yy + zz) / dij3, dudy = -xy / dij3, dudz = -xz / dij3;
      const dvdx = -xy / dij3, dvdy = (xx + zz) / dij3, dvdz = -yz / dij3;
      const dwdx = -xz / dij3, dwdy = -yz / dij3, dwdz = (xx + yy) / dij3;
      tm[0] = 1.0;
      tmu[0] = 0.0;
      tmv[0] = 0.0;
      tmw[0] = 0.0;
      s.abf[j] = tm[0];
      s.abfx[j] = 0.0;
      s.abfy[j] = 0.0;
      s.abfz[j] = 0.0;
      for (let n = 1; n < K; n++) {
        const m = pq[n] - 1;
        const d = pq[n + K];
        if (d === 1) {
          tm[n] = tm[m] * u;
          tmu[n] = tmu[m] * u + tm[m];
          tmv[n] = tmv[m] * u;
          tmw[n] = tmw[m] * u;
        } else if (d === 2) {
          tm[n] = tm[m] * v;
          tmu[n] = tmu[m] * v;
          tmv[n] = tmv[m] * v + tm[m];
          tmw[n] = tmw[m] * v;
        } else if (d === 3) {
          tm[n] = tm[m] * w;
          tmu[n] = tmu[m] * w;
          tmv[n] = tmv[m] * w;
          tmw[n] = tmw[m] * w + tm[m];
        }
        s.abf[j + nj * n] = tm[n];
        s.abfx[j + nj * n] = tmu[n] * dudx + tmv[n] * dvdx + tmw[n] * dwdx;
        s.abfy[j + nj * n] = tmu[n] * dudy + tmv[n] * dvdy + tmw[n] * dwdy;
        s.abfz[j + nj * n] = tmu[n] * dudz + tmv[n] * dvdz + tmw[n] * dwdz;
      }
    }
  }

  /**
   * Radial-angular basis summed over the neighbours per element (sumU) and its derivatives
   * (ux, uy, uz) (eapod.cpp radialangularbasis). Layout: index n + nj*k + nj*K*m for the derivatives,
   * tn + Ne*k + Ne*K*m for sumU.
   */
  private radialangularbasis(s: Scratch, tj: Int32Array, nj: number): void {
    const K = this.K3, M = this.nrbf3, Ne = this.nelements;
    const sumU = this.sumU;
    sumU.fill(0);
    for (let m = 0; m < M; m++) {
      for (let k = 0; k < K; k++) {
        for (let n = 0; n < nj; n++) {
          const ia = n + nj * k;
          const ib = n + nj * m;
          const ii = ia + nj * K * m;
          const c1 = s.rbf[ib];
          const c2 = s.abf[ia];
          s.ux[ii] = s.abfx[ia] * c1 + c2 * s.rbfx[ib];
          s.uy[ii] = s.abfy[ia] * c1 + c2 * s.rbfy[ib];
          s.uz[ii] = s.abfz[ia] * c1 + c2 * s.rbfz[ib];
          const tn = tj[n] - 1;
          sumU[tn + Ne * k + Ne * K * m] += c1 * c2;
        }
      }
    }
  }

  /** Three-body descriptors (eapod.cpp threebodydesc): d3[p + nabf3*m + nabf3*nrbf3*pair]. */
  private threebodydesc(d3: Float64Array, off: number): void {
    const Ne = this.nelements, K = this.K3, M = this.nrbf3, nabf = this.nabf3;
    const sumU = this.sumU;
    const pn = this.pn3, pc = this.pc3;
    for (let m = 0; m < M; m++) {
      for (let p = 0; p < nabf; p++) {
        const n1 = pn[p], n2 = pn[p + 1];
        for (let q = 0; q < n2 - n1; q++) {
          let k = 0;
          for (let i1 = 0; i1 < Ne; i1++) {
            const t1 = pc[n1 + q] * sumU[i1 + Ne * (n1 + q) + Ne * K * m];
            for (let i2 = i1; i2 < Ne; i2++) {
              d3[off + p + nabf * m + nabf * M * k] += t1 * sumU[i2 + Ne * (n1 + q) + Ne * K * m];
              k += 1;
            }
          }
        }
      }
    }
  }

  /** Four-body descriptors (eapod.cpp fourbodydesc): d4[p + nabf4*m + nabf4*nrbf4*triple]. */
  private fourbodydesc(d4: Float64Array, off: number): void {
    const Ne = this.nelements, K = this.K3, M = this.nrbf4, nabf = this.nabf4, Q4 = this.Q4;
    const sumU = this.sumU;
    for (let m = 0; m < M; m++) {
      const idxU = Ne * K * m;
      for (let p = 0; p < nabf; p++) {
        const n1 = this.pa4[p], n2 = this.pa4[p + 1];
        for (let q = 0; q < n2 - n1; q++) {
          const c = this.pc4[n1 + q];
          const j1 = this.pb4[n1 + q];
          const j2 = this.pb4[n1 + q + Q4];
          const j3 = this.pb4[n1 + q + 2 * Q4];
          let k = 0;
          for (let i1 = 0; i1 < Ne; i1++) {
            const c1 = sumU[idxU + i1 + Ne * j1];
            for (let i2 = i1; i2 < Ne; i2++) {
              const c2 = sumU[idxU + i2 + Ne * j2];
              const t12 = c * c1 * c2;
              for (let i3 = i2; i3 < Ne; i3++) {
                const c3 = sumU[idxU + i3 + Ne * j3];
                const kk = off + p + nabf * m + nabf * M * k;
                d4[kk] += t12 * c3;
                k += 1;
              }
            }
          }
        }
      }
    }
  }

  /** Derivative of the energy with respect to sumU from the three-body coefficients (threebody_forcecoeff). */
  private threebodyForcecoeff(fb3: Float64Array, cbOff: number): void {
    const Ne = this.nelements, K = this.K3, M = this.nrbf3, nabf = this.nabf3;
    const sumU = this.sumU;
    const coeff = this.coeff;
    const pn = this.pn3, pc = this.pc3;
    const N3 = nabf * M;
    for (let m = 0; m < M; m++) {
      for (let p = 0; p < nabf; p++) {
        const n1 = pn[p], n2 = pn[p + 1];
        const jmp = p + nabf * m;
        for (let q = 0; q < n2 - n1; q++) {
          const k = n1 + q;
          const idxU = Ne * k + Ne * K * m;
          for (let i1 = 0; i1 < Ne; i1++) {
            const tm = pc[k] * sumU[i1 + idxU];
            for (let i2 = i1; i2 < Ne; i2++) {
              const em = this.elemindex[i2 + Ne * i1];
              const cb = coeff[cbOff + jmp + N3 * em];
              fb3[i2 + idxU] += tm * cb;
              fb3[i1 + idxU] += pc[k] * cb * sumU[i2 + idxU];
            }
          }
        }
      }
    }
  }

  /** Derivative of the energy with respect to sumU from the four-body coefficients (fourbody_forcecoeff). */
  private fourbodyForcecoeff(fb4: Float64Array, cbOff: number): void {
    const Ne = this.nelements, K = this.K3, M = this.nrbf4, nabf = this.nabf4, Q4 = this.Q4;
    const sumU = this.sumU;
    const coeff = this.coeff;
    const N3 = nabf * M;
    for (let m = 0; m < M; m++) {
      for (let p = 0; p < nabf; p++) {
        const n1 = this.pa4[p], n2 = this.pa4[p + 1];
        const jpm = p + nabf * m;
        for (let q = 0; q < n2 - n1; q++) {
          const c = this.pc4[n1 + q];
          const j1 = this.pb4[n1 + q];
          const j2 = this.pb4[n1 + q + Q4];
          const j3 = this.pb4[n1 + q + 2 * Q4];
          const idx1 = Ne * j1 + Ne * K * m;
          const idx2 = Ne * j2 + Ne * K * m;
          const idx3 = Ne * j3 + Ne * K * m;
          let k = 0;
          for (let i1 = 0; i1 < Ne; i1++) {
            const c1 = sumU[idx1 + i1];
            for (let i2 = i1; i2 < Ne; i2++) {
              const c2 = sumU[idx2 + i2];
              for (let i3 = i2; i3 < Ne; i3++) {
                const c3 = sumU[idx3 + i3];
                const c4 = c * coeff[cbOff + jpm + N3 * k];
                fb4[idx3 + i3] += c4 * (c1 * c2);
                fb4[idx2 + i2] += c4 * (c1 * c3);
                fb4[idx1 + i1] += c4 * (c2 * c3);
                k += 1;
              }
            }
          }
        }
      }
    }
  }

  /**
   * Base descriptors of one atom (eapod.cpp peratombase_descriptors, the bd part of peratomenergyforce2): nj
   * neighbours at rij (3 per neighbour) of elements tj (1-based). The returned buffer (Mdesc values) is reused by
   * the next call. Zero for nj = 0.
   */
  baseDescriptors(rij: Float64Array, tj: Int32Array, nj: number): Float64Array {
    if (nj === 0) return this.baseInto(null, rij, tj, nj, null);
    return this.baseInto(this.scratch(nj), rij, tj, nj, null);
  }

  /**
   * Base descriptors and their derivatives with respect to each neighbour's r_ij (eapod.cpp peratombase_descriptors,
   * bd1 and bdd1). Returns copies: bd[D] for the Mdesc descriptors and bdd[3*(nj*D + n) + d] = d bd[D] / d rij[n][d].
   * The two- and three-body descriptors are differentiated; four-body descriptors are refused (not ported here).
   */
  baseDescriptorsWithDerivatives(rij: Float64Array, tj: Int32Array, nj: number): { bd: Float64Array; bdd: Float64Array } {
    if (this.nl4 > 0) throw new StyleError('compute podd/atom: four-body POD descriptors are not ported');
    const bdd = new Float64Array(3 * nj * this.Mdesc);
    if (nj === 0) return { bd: new Float64Array(this.Mdesc), bdd };
    const bd = this.baseInto(this.scratch(nj), rij, tj, nj, bdd);
    return { bd: Float64Array.from(bd), bdd };
  }

  /** Fills this.bd with the base descriptors of nj neighbours; s is the scratch of nj (null when nj = 0). dd: derivatives or null. */
  private baseInto(s: Scratch | null, rij: Float64Array, tj: Int32Array, nj: number, dd: Float64Array | null): Float64Array {
    const bd = this.bd;
    bd.fill(0);
    if (nj === 0 || s === null) return bd;
    const nl2 = this.nl2, nl3 = this.nl3, nl4 = this.nl4;
    const nrbf2 = this.nrbf2;
    this.radialbasis(s, rij, nj);
    this.projectRadial(s, nj);
    if (nl2 > 0) {
      for (let m = 0; m < nrbf2; m++) {
        for (let n = 0; n < nj; n++) {
          const D = m + nrbf2 * (tj[n] - 1);
          bd[D] += s.rbf[n + nj * m];
          if (dd) {
            // eapod.cpp twobodydescderiv: dd2[d + 3*(n + N*D)] += rbfd[n + N*m]
            const o = 3 * (n + nj * D);
            dd[o] += s.rbfx[n + nj * m];
            dd[o + 1] += s.rbfy[n + nj * m];
            dd[o + 2] += s.rbfz[n + nj * m];
          }
        }
      }
    }
    if (nl3 > 0 && nj > 1) {
      this.angularbasis(s, rij, nj);
      this.radialangularbasis(s, tj, nj);
      this.threebodydesc(bd, nl2);
      if (dd) this.threebodyDerivatives(s, dd, nl2, tj, nj);
      if (nl4 > 0 && nj > 2) this.fourbodydesc(bd, nl2 + nl3);
    }
    return bd;
  }

  /**
   * Derivatives of the three-body descriptors (eapod.cpp threebodydescderiv, the nelements loop): neighbour j of element
   * e contributes f * Ux[j] to the pair (i1, e) of every i1, with f = 2 t1 when i1 == e and t1 otherwise.
   */
  private threebodyDerivatives(s: Scratch, dd: Float64Array, nl2: number, tj: Int32Array, nj: number): void {
    const Ne = this.nelements, K = this.K3, M = this.nrbf3, nabf = this.nabf3;
    const sumU = this.sumU, pn = this.pn3, pc = this.pc3;
    // pair index of (a, b) in threebodydesc's order: a <= b, pairs enumerated row by row
    const pairIdx = new Int32Array(Ne * Ne);
    let kp = 0;
    for (let a = 0; a < Ne; a++) for (let b = a; b < Ne; b++) { pairIdx[a * Ne + b] = kp; pairIdx[b * Ne + a] = kp; kp++; }
    for (let m = 0; m < M; m++) {
      for (let p = 0; p < nabf; p++) {
        const n1 = pn[p], n2 = pn[p + 1];
        for (let q = 0; q < n2 - n1; q++) {
          const kq = n1 + q;
          for (let j = 0; j < nj; j++) {
            const e = tj[j] - 1;
            for (let i1 = 0; i1 < Ne; i1++) {
              const t1 = pc[kq] * sumU[i1 + Ne * kq + Ne * K * m];
              const f = i1 === e ? 2.0 * t1 : t1;
              const k = pairIdx[i1 * Ne + e];
              const o = 3 * (nj * (nl2 + p + nabf * m + nabf * M * k) + j);
              const jj = j + nj * kq + nj * K * m;
              dd[o] += f * s.ux[jj];
              dd[o + 1] += f * s.uy[jj];
              dd[o + 2] += f * s.uz[jj];
            }
          }
        }
      }
    }
  }

  /**
   * Energy of one atom of element ti (1-based) with nj neighbours at rij (3 per neighbour, neighbour minus
   * centre) of elements tj (1-based). Writes the force on each neighbour's pair into fij (3 per neighbour,
   * the force on the centre is minus the sum). eapod.cpp peratomenergyforce2.
   */
  atomEnergyForce(ti: number, nj: number, rij: Float64Array, tj: Int32Array, fij: Float64Array): number {
    const nc = this.nCoeffPerElement * (ti - 1);
    const coeff = this.coeff;
    fij.fill(0, 0, 3 * nj);
    if (nj === 0) return coeff[nc];
    const s = this.scratch(nj);
    const bd = this.baseInto(s, rij, tj, nj, null);
    const nl2 = this.nl2, nl3 = this.nl3, nl4 = this.nl4;
    const nrbf2 = this.nrbf2;
    let e = coeff[nc];
    for (let m = 0; m < this.Mdesc; m++) e += coeff[1 + m + nc] * bd[m];

    // forces: the two-body part through the radial derivatives
    const cbOff = nc + 1;
    if (nl2 > 0) {
      for (let n = 0; n < nj; n++) {
        for (let m = 0; m < nrbf2; m++) {
          const c = coeff[cbOff + m + nrbf2 * (tj[n] - 1)];
          const idx = n + nj * m;
          fij[3 * n] += c * s.rbfx[idx];
          fij[3 * n + 1] += c * s.rbfy[idx];
          fij[3 * n + 2] += c * s.rbfz[idx];
        }
      }
    }
    if (nl3 > 0 && nj > 1) {
      const forcecoeff = this.forcecoeff;
      forcecoeff.fill(0);
      this.threebodyForcecoeff(forcecoeff, cbOff + nl2);
      if (nl4 > 0 && nj > 2) this.fourbodyForcecoeff(forcecoeff, cbOff + nl2 + nl3);
      // the three-body derivative: sum over monomials and radial functions of forcecoeff times dU/dr
      const K = this.K3, M = this.nrbf3, Ne = this.nelements;
      for (let j = 0; j < nj; j++) {
        const i2 = tj[j] - 1;
        let fx = 0, fy = 0, fz = 0;
        for (let m = 0; m < M; m++) {
          for (let k = 0; k < K; k++) {
            const fc = forcecoeff[i2 + Ne * k + Ne * K * m];
            const idxU = j + nj * k + nj * K * m;
            fx += fc * s.ux[idxU];
            fy += fc * s.uy[idxU];
            fz += fc * s.uz[idxU];
          }
        }
        fij[3 * j] += fx;
        fij[3 * j + 1] += fy;
        fij[3 * j + 2] += fz;
      }
    }
    return e;
  }

}
