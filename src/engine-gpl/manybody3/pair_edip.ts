/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANYBODY/pair_edip.cpp and
 * src/MANYBODY/pair_edip.h (PairEDIP, pair_style edip: the tabulated single-element form with initGrids and the
 * interpolated grids) and src/MANYBODY/pair_edip_multi.cpp and pair_edip_multi.h (PairEDIPMulti, pair_style edip/multi:
 * the analytic form per element triplet). Contributing author: Luca Ferraro (CASPUR). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Both styles sum the full neighbour list of each owned atom i: the coordination Z_i first, then the two-body and
 * three-body terms, then the coordination forces. The grids of edip are built as native does (initGrids, GRIDDENSITY
 * 8000 points per unit, linear interpolation); edip/multi evaluates the functions directly.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { exp } from '../libm/exp';
import { pow } from '../libm/pow';
import { PotentialEntryReader, nextDouble, tallyPair, tallyTriplet, mapElementsToTypes } from '../manybody2/common';

/** pair_edip.cpp GRIDDENSITY, GRIDSTART and leadDimInteractionList (also used by pair_edip_multi.cpp). */
const GRIDDENSITY = 8000;
const GRIDSTART = 0.1;
const LEAD_DIM = 64;
/** -std::numeric_limits<double>::min() * 1000.0 (pair_edip.cpp allocateGrids and initGrids). */
const LEFT_LIMIT_TO_ZERO = -2.2250738585072014e-308 * 1000.0;

/** One parameter set of an EDIP potential file (the names of pair_edip.h's Param). */
export interface EdipParam {
  A: number; B: number; cutoffA: number; cutoffC: number; alpha: number; beta: number; sigma: number; rho: number;
  gamma: number; eta: number; lambda: number; mu: number; Q0: number; u1: number; u2: number; u3: number; u4: number;
  cutsq: number;
  ielement: number; jelement: number; kelement: number;
}

/** The fields read after the three element names of an entry, in file order. */
const NUMBER_ORDER = ['A', 'B', 'cutoffA', 'cutoffC', 'alpha', 'beta', 'eta', 'gamma', 'lambda', 'mu', 'rho', 'sigma', 'Q0', 'u1', 'u2', 'u3', 'u4'] as const;

/** EDIP unit check of PotentialFileReader (no unit conversion): the file's UNITS tag must match the unit style. */
const checkEdipUnits = (text: string, fileName: string, ctx: StyleContext): void => {
  const first = (text.split('\n')[0] ?? '').split(/\s+/).filter((s) => s !== '');
  const k = first.indexOf('UNITS:');
  const units = k >= 0 && k + 1 < first.length ? first[k + 1] : '';
  const style = ctx.s?.units.style ?? 'lj';
  if (units !== '' && units !== style) {
    throw new StyleError(`edip potential file ${fileName} requires ${units} units but ${style} units are in use`);
  }
};

/**
 * The entries of an EDIP potential file for the elements of the mapping (pair_edip.cpp read_file): an entry whose three
 * names are not all mapped is skipped; a parameter line is read in full (20 words) before the names are matched.
 */
export const readEdipFile = (text: string, fileName: string, elements: readonly string[], ctx: StyleContext): EdipParam[] => {
  checkEdipUnits(text, fileName, ctx);
  const reader = new PotentialEntryReader(text.split(/\r?\n/), 'edip potential');
  const params: EdipParam[] = [];
  for (let words = reader.next(20); words !== null; words = reader.next(20)) {
    const ielement = elements.indexOf(words[0]);
    if (ielement < 0) continue;
    const jelement = elements.indexOf(words[1]);
    if (jelement < 0) continue;
    const kelement = elements.indexOf(words[2]);
    if (kelement < 0) continue;
    const v: Record<string, number> = {};
    try {
      NUMBER_ORDER.forEach((name, k) => { v[name] = nextDouble(words[3 + k]); });
    } catch (e) {
      throw new StyleError(`Error reading EDIP potential file: ${(e as Error).message}`);
    }
    const p: EdipParam = {
      A: v.A, B: v.B, cutoffA: v.cutoffA, cutoffC: v.cutoffC, alpha: v.alpha, beta: v.beta, sigma: v.sigma, rho: v.rho,
      gamma: v.gamma, eta: v.eta, lambda: v.lambda, mu: v.mu, Q0: v.Q0, u1: v.u1, u2: v.u2, u3: v.u3, u4: v.u4,
      cutsq: 0, ielement, jelement, kelement,
    };
    if (p.A < 0.0 || p.B < 0.0 || p.cutoffA < 0.0 || p.cutoffC < 0.0 || p.alpha < 0.0 || p.beta < 0.0 || p.eta < 0.0 ||
        p.gamma < 0.0 || p.lambda < 0.0 || p.mu < 0.0 || p.rho < 0.0 || p.sigma < 0.0) {
      throw new StyleError('Illegal EDIP parameter');
    }
    params.push(p);
  }
  return params;
};

/** setup_params (both styles): elem3param for every element triplet, the squared cutoffs and cutmax. */
const setupParams = (params: EdipParam[], elements: readonly string[]): { elem3param: Int32Array; cutmax: number } => {
  const nel = elements.length;
  const elem3param = new Int32Array(nel * nel * nel);
  for (let i = 0; i < nel; i++) {
    for (let j = 0; j < nel; j++) {
      for (let k = 0; k < nel; k++) {
        let n = -1;
        for (let m = 0; m < params.length; m++) {
          const p = params[m];
          if (i === p.ielement && j === p.jelement && k === p.kelement) {
            if (n >= 0) throw new StyleError(`Potential file has a duplicate entry for: ${elements[i]} ${elements[j]} ${elements[k]}`);
            n = m;
          }
        }
        if (n < 0) throw new StyleError(`Potential file is missing an entry for: ${elements[i]} ${elements[j]} ${elements[k]}`);
        elem3param[(i * nel + j) * nel + k] = n;
      }
    }
  }
  for (const p of params) p.cutsq = p.cutoffA * p.cutoffA;
  let cutmax = 0.0;
  for (const p of params) {
    const rtmp = Math.sqrt(p.cutsq);
    if (rtmp > cutmax) cutmax = rtmp;
  }
  return { elem3param, cutmax };
};

/** The parsed potential of the mapped elements (pair_edip.h members that the styles share). */
interface EdipCommon {
  elements: string[];
  elemOf: Int32Array;
  params: EdipParam[];
  elem3param: Int32Array;
  cutmax: number;
}

/** Native Pair::ev_tally3 for the energy and the per-atom energy (THIRD on each of i, j, k); the virial via tallyTriplet. */
const tally3 = (
  pc: PairCompute, i: number, j: number, k: number, evdwl: number,
  fj: Float64Array, fk: Float64Array, drji: Float64Array, drki: Float64Array,
): void => {
  pc.acc.evdwl += evdwl;
  if (pc.eatom) {
    const epot = (1 / 3) * evdwl;
    pc.eatom[i] += epot;
    pc.eatom[j] += epot;
    pc.eatom[k] += epot;
  }
  tallyTriplet(pc, i, j, k, fj, fk, drji, drki);
};

/** The coordination fc(r) and its derivative of edip/multi (pair_edip_multi.cpp edip_fc). */
const edipFc = (r: number, p: EdipParam): [number, number] => {
  const a = p.cutoffA, c = p.cutoffC, alpha = p.alpha;
  if (r < c + 1e-6) return [1.0, 0.0];
  if (r > a - 1e-6) return [0.0, 0.0];
  const x = (a - c) / (r - c);
  const v1 = x * x * x;
  const v2 = 1.0 / (1.0 - v1);
  const f = exp(alpha * v2);
  const fdr = (3.0 * x * v1 / (a - c)) * (-alpha * v2 * v2) * f;
  return [f, fdr];
};

/** edip_fcut2 of edip/multi: the two-body cutoff exp(sigma/(r - a)). */
const edipFcut2 = (r: number, p: EdipParam): [number, number] => {
  const sigma = p.sigma, a = p.cutoffA;
  if (r > a - 1e-6) return [0.0, 0.0];
  const v1 = 1.0 / (r - a);
  const f = exp(sigma * v1);
  const fdr = (-sigma * v1 * v1) * f;
  return [f, fdr];
};

/** edip_fcut3 of edip/multi: the three-body cutoff exp(gamma/(r - a)). */
const edipFcut3 = (r: number, p: EdipParam): [number, number] => {
  const gamma = p.gamma, a = p.cutoffA;
  if (r > a - 1e-6) return [0.0, 0.0];
  const v1 = 1.0 / (r - a);
  const f = exp(gamma * v1);
  const fdr = (-gamma * v1 * v1) * f;
  return [f, fdr];
};

/** edip_tau of edip/multi: tau(Z) and its derivative. */
const edipTau = (z: number, p: EdipParam): [number, number] => {
  const { u1, u2, u3, u4 } = p;
  const v1 = exp(-u4 * z);
  const v2 = exp(-2.0 * u4 * z);
  const f = u1 + u2 * u3 * v1 - u2 * v2;
  const fdZ = -u2 * u3 * u4 * v1 + 2.0 * u2 * u4 * v2;
  return [f, fdZ];
};

/** edip_h of edip/multi: h(l, Z) and its derivatives in l and Z. */
const edipH = (l: number, z: number, p: EdipParam): [number, number, number] => {
  const { lambda, eta, Q0, mu } = p;
  const Q = Q0 * exp(-mu * z);
  const QdZ = -mu * Q;
  const [Tau, TaudZ] = edipTau(z, p);
  const v1 = (l + Tau) * (l + Tau);
  const u2 = Q * v1;
  const v2 = exp(-u2);
  const f = lambda * (1 - v2 + eta * u2);
  const v3 = lambda * (v2 + eta);
  const du2l = Q * 2 * (l + Tau);
  const fdl = v3 * du2l;
  const du2Z = QdZ * v1 + Q * 2 * (l + Tau) * TaudZ;
  const fdZ = v3 * du2Z;
  return [f, fdl, fdZ];
};

/** edip_pair of edip/multi: V_ij(r, Z) and its derivatives in r and Z. */
const edipPair = (r: number, z: number, p: EdipParam): [number, number, number] => {
  const { A, B, rho, beta } = p;
  const v1 = pow(B / r, rho);
  const v2 = exp(-beta * z * z);
  const [v3, v4] = edipFcut2(r, p);
  const eng = A * (v1 - v2) * v3;
  const fdr = A * (v1 - v2) * v4 + A * (-rho * v1 / r) * v3;
  const fZ = A * (2 * beta * z * v2) * v3;
  return [eng, fdr, fZ];
};

/** costheta_d of edip/multi: the derivatives of cos(theta_jik) with respect to the three positions. */
const costhetaD = (drij: number[], rij: number, drik: number[], rik: number): { dri: number[]; drj: number[]; drk: number[] } => {
  const costheta = (drij[0] * drik[0] + drij[1] * drik[1] + drij[2] * drik[2]) / rij / rik;
  const s1 = 1 / rij / rik;
  const s2 = -costheta / rij / rij;
  const s3 = -costheta / rik / rik;
  const drj = [0, 1, 2].map((c) => s1 * drik[c] + s2 * drij[c]);
  const drk = [0, 1, 2].map((c) => s1 * drij[c] + s3 * drik[c]);
  const dri = [0, 1, 2].map((c) => -1 * drj[c] + -1 * drk[c]);
  return { dri, drj, drk };
};

/** The reading and set-up shared by edip and edip/multi. */
abstract class EdipBase extends Pair {
  abstract readonly name: string;
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;
  protected common: EdipCommon | null = null;
  protected readonly cache = new Map<string, EdipParam[]>();
  protected elemOf = new Int32Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.elemOf = new Int32Array(ntypes + 1).fill(-1);
    this.common = null;
  }

  /** The element names of the mapping and the potential file read for them (native map_element2type, read_file, setup_params). */
  protected loadFile(args: string[], ctx: StyleContext): EdipCommon {
    if (this.ntypes === 0) throw new StyleError('Pair_coeff command before simulation box is defined');
    if (args.length < 3) throw new StyleError('Incorrect args for pair coefficients');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Incorrect args for pair coefficients');
    // map_element2type(narg - 3, arg + 3), then the single-element check of pair_edip.cpp coeff
    const mapping = mapElementsToTypes(args.slice(3), this.ntypes);
    if (this.name === 'edip' && mapping.elements.length !== 1) {
      throw new StyleError('Pair style edip only supports single element potentials');
    }
    const fileName = args[2];
    const text = ctx.readFile(fileName);
    const elements = mapping.elements;
    const params = readEdipFile(text, fileName, elements, ctx);
    const { elem3param, cutmax } = setupParams(params, elements);
    const elemOf = Int32Array.from(mapping.map);
    this.elemOf = elemOf;
    return { elements, elemOf, params, elem3param, cutmax };
  }

  /** init_one: the global cutmax, after the check that the pair coeffs cover the type (native setflag test). */
  initOne(i: number, j: number): number {
    if (!this.common || this.elemOf[i] < 0 || this.elemOf[j] < 0) throw new StyleError('All pair coeffs are not set');
    return this.common.cutmax;
  }

  initStyle(ctx: StyleContext): void {
    // native accepts pair_modify shift and tail for edip and ignores them (measured with native LAMMPS b332f28)
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.table !== 12) throw new StyleError(`pair style ${this.name} does not support the pair_modify table option`);
    if (!this.common) throw new StyleError('All pair coeffs are not set');
  }
}

/** Interpolation of a grid at a fractional index (the C expression of pair_edip.cpp, in its operation order). */
const interp = (grid: Float64Array, idx: number, tmp: number): number => {
  const y1 = grid[idx];
  const y2 = grid[idx + 1];
  return y1 + (y2 - y1) * (tmp - idx);
};

/** The grids of pair_style edip (pair_edip.cpp allocateGrids and initGrids), built from the parameters of the file. */
interface EdipGrids {
  cutoffFunction: Float64Array;
  cutoffFunctionDerived: Float64Array;
  pow2B: Float64Array;
  exp2B: Float64Array;
  exp3B: Float64Array;
  qFunctionGrid: Float64Array;
  expMinusBetaZeta: Float64Array;
  tauFunctionGrid: Float64Array;
  tauFunctionDerivedGrid: Float64Array;
}

const buildEdipGrids = (p: EdipParam): EdipGrids => {
  const { A, B, rho, sigma, gamma: gamm, cutoffA, cutoffC, alpha, beta, Q0, mu, u1, u2, u3, u4 } = p;
  // tauFunctionGrid, expMinusBetaZeta_iZeta_iGrid, qFunctionGrid
  const numZ = Math.trunc(LEAD_DIM * GRIDDENSITY) + 2;
  const tauFunctionGrid = new Float64Array(numZ);
  const tauFunctionDerivedGrid = new Float64Array(numZ);
  const expMinusBetaZeta = new Float64Array(numZ);
  const qFunctionGrid = new Float64Array(numZ);
  const deltaZ = 1.0 / GRIDDENSITY;
  let r = 0.0;
  for (let l = 0; l < numZ; l++) {
    tauFunctionGrid[l] = u1 + u2 * u3 * exp(-u4 * r) - u2 * exp(-2.0 * u4 * r);
    tauFunctionDerivedGrid[l] = -u2 * u3 * u4 * exp(-u4 * r) + 2.0 * u2 * u4 * exp(-2.0 * u4 * r);
    r += deltaZ;
  }
  r = 0.0;
  for (let l = 0; l < numZ; l++) {
    expMinusBetaZeta[l] = exp(-beta * r * r);
    r += deltaZ;
  }
  r = 0.0;
  for (let l = 0; l < numZ; l++) {
    qFunctionGrid[l] = Q0 * exp(-mu * r);
    r += deltaZ;
  }
  // cutoffFunction
  const numOne = Math.trunc((cutoffC - GRIDSTART) * GRIDDENSITY);
  const numNotOne = Math.trunc((cutoffA - cutoffC) * GRIDDENSITY);
  const numCut = numOne + numNotOne + 2;
  const cutoffFunction = new Float64Array(numCut);
  const cutoffFunctionDerived = new Float64Array(numCut);
  r = GRIDSTART;
  const deltaC = 1.0 / GRIDDENSITY;
  for (let l = 0; l < numOne; l++) {
    cutoffFunction[l] = 1.0;
    cutoffFunctionDerived[l] = 0.0;
    r += deltaC;
  }
  for (let l = numOne; l < numCut; l++) {
    const temp = (cutoffA - cutoffC) / (r - cutoffC);
    const temp3 = temp * temp * temp;
    const temp4 = temp3 * temp;
    cutoffFunction[l] = exp(alpha / (1.0 - temp3));
    cutoffFunctionDerived[l] = (-3 * alpha / (cutoffA - cutoffC)) *
      (temp4 / ((1 - temp3) * (1 - temp3))) * exp(alpha / (1.0 - temp3));
    r += deltaC;
  }
  // pow2B, exp2B, exp3B
  const numR = Math.trunc((cutoffA + LEFT_LIMIT_TO_ZERO - GRIDSTART) * GRIDDENSITY);
  const numRTotal = numR + 2;
  const pow2B = new Float64Array(numRTotal);
  const exp2B = new Float64Array(numRTotal);
  const exp3B = new Float64Array(numRTotal);
  r = GRIDSTART;
  const deltaR = 1.0 / GRIDDENSITY;
  for (let l = 0; l < numR; l++) {
    pow2B[l] = pow(B / r, rho);
    exp2B[l] = A * exp(sigma / (r - cutoffA));
    exp3B[l] = exp(gamm / (r - cutoffA));
    r += deltaR;
  }
  pow2B[numR] = pow(B / r, rho);
  exp2B[numR] = 0;
  exp3B[numR] = 0;
  r += deltaR;
  pow2B[numR + 1] = pow(B / r, rho);
  exp2B[numR + 1] = 0;
  exp3B[numR + 1] = 0;
  return {
    cutoffFunction, cutoffFunctionDerived, pow2B, exp2B, exp3B, qFunctionGrid, expMinusBetaZeta,
    tauFunctionGrid, tauFunctionDerivedGrid,
  };
};

/**
 * pair_style edip (one element): the parameters of the single entry, the grids of initGrids, and the compute of
 * pair_edip.cpp with its interpolated functions.
 */
export class PairEDIP extends EdipBase {
  readonly name: string = 'edip';
  private p0: EdipParam | null = null;
  private grids: EdipGrids | null = null;
  private bufs = new Float64Array(0);

  coeff(args: string[], ctx: StyleContext): void {
    const c = this.loadFile(args, ctx);
    this.common = c;
    // the globals of pair_edip.cpp setup_params are the first entry (params[0])
    this.p0 = c.params[0];
    this.grids = buildEdipGrids(this.p0);
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const p = this.p0, G = this.grids;
    if (!p || !G) throw new StyleError('All pair coeffs are not set');
    const { x, f, nlocal } = pc;
    const nb = list.neighbors;
    const cutsq = p.cutsq;
    const { cutoffA, cutoffC, sigma, lambda, eta, mu, beta, rho, gamma: gamm } = p;
    const maxJ = (() => {
      let m = 0;
      for (let i = 0; i < nlocal; i++) if (list.numneigh[i] > m) m = list.numneigh[i];
      return m;
    })();
    if (this.bufs.length < 11 * maxJ) this.bufs = new Float64Array(11 * Math.max(maxJ, 1));
    const B = this.bufs;
    const oInvR = 0, oE3 = maxJ, oE3D = 2 * maxJ, oE2 = 3 * maxJ, oE2D = 4 * maxJ, oP2 = 5 * maxJ, oFC = 6 * maxJ;

    for (let i = 0; i < nlocal; i++) {
      let zeta_i = 0.0;
      let numForce = 0;
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const k0 = list.firstneigh[i], jnum = list.numneigh[i];

      // pre-loop: pair quantities of the neighbours and the coordination Z_i
      for (let nj = 0; nj < jnum; nj++) {
        const j = nb[k0 + nj] & NEIGHMASK;
        const dx = xtmp - x[3 * j], dy = ytmp - x[3 * j + 1], dz = ztmp - x[3 * j + 2];
        let rr = dx * dx + dy * dy + dz * dz;
        if (rr > cutsq) continue;
        const r = Math.sqrt(rr);
        const invR = 1.0 / r;
        B[oInvR + nj] = invR;
        const invRMinusCutoffA = 1.0 / (r - cutoffA);
        const sigmaInv = sigma * invRMinusCutoffA;
        const gammInv = gamm * invRMinusCutoffA;
        const dX = r - GRIDSTART;
        const tmp = dX * GRIDDENSITY;
        const idx = Math.trunc(tmp);
        const exp3B_ij = interp(G.exp3B, idx, tmp);
        B[oE3 + nj] = exp3B_ij;
        B[oE3D + nj] = -exp3B_ij * gammInv * invRMinusCutoffA;
        const exp2B_ij = interp(G.exp2B, idx, tmp);
        B[oE2 + nj] = exp2B_ij;
        B[oE2D + nj] = -exp2B_ij * sigmaInv * invRMinusCutoffA;
        B[oP2 + nj] = interp(G.pow2B, idx, tmp);
        if (r < cutoffC) {
          zeta_i += 1.0;
        } else {
          const cutoffFunction_ij = interp(G.cutoffFunction, idx, tmp);
          zeta_i += cutoffFunction_ij;
          const zeta_iDerived = interp(G.cutoffFunctionDerived, idx, tmp);
          const zDInvR = zeta_iDerived * invR;
          const o = oFC + numForce * 5;
          B[o] = zDInvR;
          B[o + 1] = dx;
          B[o + 2] = dy;
          B[o + 3] = dz;
          B[o + 4] = j;
          numForce++;
        }
      }

      // quantities depending on zeta_i
      const tmpZ = zeta_i * GRIDDENSITY;
      const idZ = Math.trunc(tmpZ);
      const expMinusBetaZeta_iZeta_i = interp(G.expMinusBetaZeta, idZ, tmpZ);
      const qFunction = interp(G.qFunctionGrid, idZ, tmpZ);
      const tauFunction = interp(G.tauFunctionGrid, idZ, tmpZ);
      const tauFunctionDerived = interp(G.tauFunctionDerivedGrid, idZ, tmpZ);
      const forceModCoord_factor = 2.0 * beta * zeta_i * expMinusBetaZeta_iZeta_i;
      let forceModCoord = 0.0;

      // two-body terms and the three-body terms of the pairs (j, k) with k after j
      for (let nj = 0; nj < jnum; nj++) {
        const j = nb[k0 + nj] & NEIGHMASK;
        const dxj = x[3 * j] - xtmp, dyj = x[3 * j + 1] - ytmp, dzj = x[3 * j + 2] - ztmp;
        const rr = dxj * dxj + dyj * dyj + dzj * dzj;
        if (rr > cutsq) continue;
        const invR_ij = B[oInvR + nj];
        const pow2B_ij = B[oP2 + nj];
        const potential2B_factor = pow2B_ij - expMinusBetaZeta_iZeta_i;
        const exp2B_ij = B[oE2 + nj];
        const pow2BDerived_ij = -rho * invR_ij * pow2B_ij;
        forceModCoord += forceModCoord_factor * exp2B_ij;
        const exp2BDerived_ij = B[oE2D + nj];
        const forceMod2B = exp2BDerived_ij * potential2B_factor + exp2B_ij * pow2BDerived_ij;
        const dcx = invR_ij * dxj, dcy = invR_ij * dyj, dcz = invR_ij * dzj;
        const exp3B_ij = B[oE3 + nj];
        const exp3BDerived_ij = B[oE3D + nj];
        const fijx = forceMod2B * dcx, fijy = forceMod2B * dcy, fijz = forceMod2B * dcz;
        f[3 * i] += fijx; f[3 * i + 1] += fijy; f[3 * i + 2] += fijz;
        f[3 * j] -= fijx; f[3 * j + 1] -= fijy; f[3 * j + 2] -= fijz;
        const evdwl = exp2B_ij * potential2B_factor;
        tallyPair(pc, i, j, evdwl, -forceMod2B * invR_ij, dxj, dyj, dzj);

        for (let nk = nj + 1; nk < jnum; nk++) {
          const k = nb[k0 + nk] & NEIGHMASK;
          const dxk = x[3 * k] - xtmp, dyk = x[3 * k + 1] - ytmp, dzk = x[3 * k + 2] - ztmp;
          const rkk = dxk * dxk + dyk * dyk + dzk * dzk;
          if (rkk > cutsq) continue;
          const invR_ik = B[oInvR + nk];
          const dkx = invR_ik * dxk, dky = invR_ik * dyk, dkz = invR_ik * dzk;
          const cosTeta = dcx * dkx + dcy * dky + dcz * dkz;
          const cosTetaDiff = cosTeta + tauFunction;
          const cosTetaDiffCosTetaDiff = cosTetaDiff * cosTetaDiff;
          const qFunctionCosTetaDiffCosTetaDiff = cosTetaDiffCosTetaDiff * qFunction;
          const expMinusQFunctionCosTetaDiffCosTetaDiff = exp(-qFunctionCosTetaDiffCosTetaDiff);
          const potentia3B_factor = lambda * ((1.0 - expMinusQFunctionCosTetaDiffCosTetaDiff) +
            eta * qFunctionCosTetaDiffCosTetaDiff);
          const exp3B_ik = B[oE3 + nk];
          const exp3BDerived_ik = B[oE3D + nk];
          const forceMod3B_factor1_ij = -exp3BDerived_ij * exp3B_ik * potentia3B_factor;
          const forceMod3B_factor2 = 2.0 * lambda * exp3B_ij * exp3B_ik * qFunction * cosTetaDiff *
            (eta + expMinusQFunctionCosTetaDiffCosTetaDiff);
          const forceMod3B_factor2_ij = forceMod3B_factor2 * invR_ij;
          const fj = new Float64Array(3);
          const fk = new Float64Array(3);
          const dcs = [dcx, dcy, dcz], dks = [dkx, dky, dkz];
          for (let c = 0; c < 3; c++) {
            fj[c] = forceMod3B_factor1_ij * dcs[c] + forceMod3B_factor2_ij * (cosTeta * dcs[c] - dks[c]);
          }
          const forceMod3B_factor1_ik = -exp3BDerived_ik * exp3B_ij * potentia3B_factor;
          const forceMod3B_factor2_ik = forceMod3B_factor2 * invR_ik;
          for (let c = 0; c < 3; c++) {
            fk[c] = forceMod3B_factor1_ik * dks[c] + forceMod3B_factor2_ik * (cosTeta * dks[c] - dcs[c]);
          }
          forceModCoord += forceMod3B_factor2 * (tauFunctionDerived - 0.5 * mu * cosTetaDiff);
          for (let c = 0; c < 3; c++) {
            f[3 * j + c] += fj[c];
            f[3 * k + c] += fk[c];
            f[3 * i + c] -= fj[c] + fk[c];
          }
          const evdwl3 = exp3B_ij * exp3B_ik * potentia3B_factor;
          tally3(pc, i, j, k, evdwl3, fj, fk, Float64Array.of(dxj, dyj, dzj), Float64Array.of(dxk, dyk, dzk));
        }
      }

      // forces due to the coordination f(Z)
      for (let idx = 0; idx < numForce; idx++) {
        const o = oFC + idx * 5;
        const zeta_iDerivedInvR_ij = B[o];
        const dx = B[o + 1], dy = B[o + 2], dz = B[o + 3];
        const j = B[o + 4];
        const forceModCoord_ij = forceModCoord * zeta_iDerivedInvR_ij;
        const fx = forceModCoord_ij * dx, fy = forceModCoord_ij * dy, fz = forceModCoord_ij * dz;
        f[3 * i] -= fx; f[3 * i + 1] -= fy; f[3 * i + 2] -= fz;
        f[3 * j] += fx; f[3 * j + 1] += fy; f[3 * j + 2] += fz;
        tallyPair(pc, i, j, 0.0, -forceModCoord_ij, dx, dy, dz);
      }
    }
  }
}

/** pair_style edip/multi: the analytic functions of pair_edip_multi.cpp for each element triplet. */
export class PairEDIPMulti extends EdipBase {
  readonly name: string = 'edip/multi';

  coeff(args: string[], ctx: StyleContext): void {
    this.common = this.loadFile(args, ctx);
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    const C = this.common;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    if (!C) throw new StyleError('All pair coeffs are not set');
    const { x, f, type, nlocal } = pc;
    const nb = list.neighbors;
    const nel = C.elements.length;
    const { params, elem3param, elemOf } = C;
    const P = (i: number, j: number, k: number): EdipParam => params[elem3param[(i * nel + j) * nel + k]];
    const preFC: number[][] = [];

    for (let i = 0; i < nlocal; i++) {
      let zeta_i = 0.0;
      preFC.length = 0;
      const itype = elemOf[type[i]];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const k0 = list.firstneigh[i], jnum = list.numneigh[i];

      // pre-loop: the coordination Z_i and the derivatives of f(r_ij) for the coordination forces
      for (let nj = 0; nj < jnum; nj++) {
        const j = nb[k0 + nj] & NEIGHMASK;
        const delx = x[3 * j] - xtmp, dely = x[3 * j + 1] - ytmp, delz = x[3 * j + 2] - ztmp;
        const r2 = delx * delx + dely * dely + delz * delz;
        const jtype = elemOf[type[j]];
        const param = P(itype, jtype, jtype);
        if (r2 > param.cutsq) continue;
        const r = Math.sqrt(r2);
        if (r < param.cutoffC) {
          zeta_i += 1.0;
        } else {
          const [fz, fdr] = edipFc(r, param);
          zeta_i += fz;
          const dzetair = -fdr / r;
          preFC.push([dzetair, delx, dely, delz, j]);
        }
      }

      let dpairZ = 0.0;
      let dtripleZ = 0.0;
      // two-body and three-body terms
      for (let nj = 0; nj < jnum; nj++) {
        const j = nb[k0 + nj] & NEIGHMASK;
        const dr_ij = [x[3 * j] - xtmp, x[3 * j + 1] - ytmp, x[3 * j + 2] - ztmp];
        const r2 = dr_ij[0] * dr_ij[0] + dr_ij[1] * dr_ij[1] + dr_ij[2] * dr_ij[2];
        const jtype = elemOf[type[j]];
        const ijparam = elem3param[(itype * nel + jtype) * nel + jtype];
        if (r2 > params[ijparam].cutsq) continue;
        const r_ij = Math.sqrt(r2);
        const [evdwl, fdr, fdZ] = edipPair(r_ij, zeta_i, params[ijparam]);
        const fpair = -fdr / r_ij;
        dpairZ += fdZ;
        for (let c = 0; c < 3; c++) {
          f[3 * i + c] -= fpair * dr_ij[c];
          f[3 * j + c] += fpair * dr_ij[c];
        }
        tallyPair(pc, i, j, evdwl, fpair, -dr_ij[0], -dr_ij[1], -dr_ij[2]);

        for (let nk = nj + 1; nk < jnum; nk++) {
          const k = nb[k0 + nk] & NEIGHMASK;
          const ktype = elemOf[type[k]];
          const ikparam = elem3param[(itype * nel + ktype) * nel + ktype];
          const ijkparam = elem3param[(itype * nel + jtype) * nel + ktype];
          const dr_ik = [x[3 * k] - xtmp, x[3 * k + 1] - ytmp, x[3 * k + 2] - ztmp];
          const r_ik2 = dr_ik[0] * dr_ik[0] + dr_ik[1] * dr_ik[1] + dr_ik[2] * dr_ik[2];
          if (r_ik2 > params[ikparam].cutsq) continue;
          const r_ik = Math.sqrt(r_ik2);
          const costheta = (dr_ij[0] * dr_ik[0] + dr_ij[1] * dr_ik[1] + dr_ij[2] * dr_ik[2]) / r_ij / r_ik;
          const [v1, v2] = edipFcut3(r_ij, params[ijparam]);
          const [v3, v4] = edipFcut3(r_ik, params[ikparam]);
          const [v5, v6, v7] = edipH(costheta, zeta_i, params[ijkparam]);
          const e3 = v1 * v3 * v5;
          dtripleZ += v1 * v3 * v7;
          const dhl = v1 * v3 * v6;
          const { drj, drk } = costhetaD(dr_ij, r_ij, dr_ik, r_ik);
          const f_ij = new Float64Array(3), f_ik = new Float64Array(3);
          for (let c = 0; c < 3; c++) {
            f_ij[c] = -dhl * drj[c];
            f_ik[c] = -dhl * drk[c];
          }
          let dfr = v2 * v3 * v5;
          let fp = -dfr / r_ij;
          for (let c = 0; c < 3; c++) f_ij[c] += fp * dr_ij[c];
          dfr = v1 * v4 * v5;
          fp = -dfr / r_ik;
          for (let c = 0; c < 3; c++) f_ik[c] += fp * dr_ik[c];
          for (let c = 0; c < 3; c++) {
            f[3 * j + c] += f_ij[c];
            f[3 * k + c] += f_ik[c];
            f[3 * i + c] -= f_ij[c] + f_ik[c];
          }
          tally3(pc, i, j, k, e3, f_ij, f_ik, Float64Array.from(dr_ij), Float64Array.from(dr_ik));
        }
      }

      // forces due to the coordination f(Z)
      for (const [dzeta0, dlx, dly, dlz, jf] of preFC) {
        const j = jf;
        const dzetair = dzeta0 * (dpairZ + dtripleZ);
        f[3 * j] += dzetair * dlx;
        f[3 * j + 1] += dzetair * dly;
        f[3 * j + 2] += dzetair * dlz;
        f[3 * i] -= dzetair * dlx;
        f[3 * i + 1] -= dzetair * dly;
        f[3 * i + 2] -= dzetair * dlz;
        tallyPair(pc, i, j, 0.0, dzetair, -dlx, -dly, -dlz);
      }
    }
  }
}
