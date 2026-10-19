/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_dispersion_d3.cpp
 * and pair_dispersion_d3.h (class PairDispersionD3; contributing authors Sonia Salomoni and Arthur France-Lanord, IMPMC,
 * CNRS, Sorbonne Universite; copyright Sandia Corporation). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/pair_dispersion_d3.html (verbatim): "pair_style dispersion/d3 damping functional cutoff cn_cutoff";
 * "damping = damping function: zero, zerom, bj, or bjm"; "functional = XC functional form"; "cutoff = global cutoff
 * (distance units)"; "cn_cutoff = coordination number cutoff (distance units)"; "pair_coeff * * element1 element2 ...".
 * "Style dispersion/d3 computes the dispersion energy-correction used in the DFT-D3 method of Grimme".
 *
 * Native behaviour kept here: the coordination numbers are summed over the half list (both atoms of each pair, ghosts
 * folded into their owners and copied back, as reverse_comm and forward_comm do); the energy and the pair forces use
 * the C6 interpolated in (CN_i, CN_j) with get_dC6; a second pass over the half list adds the forces through dC6/dCN.
 * pow and exp are the C library's (src/engine-gpl/libm). The virial is the fdotr sum.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { d3Tables, NUM_ELEMENTS, N_PARS_COLS, N_PARS_ROWS } from './d3_tables';
import { DAMPING_PARAMS } from './d3_functionals';

/** pair_dispersion_d3.cpp: global ad hoc parameters K1 = 16.0, K3 = -4.0. */
const K1 = 16.0;
const K3 = -4.0;
/** atomic units (Bohr) to Angstrom and (Hartree) to eV. */
const autoang = 0.52917725;
const autoev = 27.21140795;

/** find_atomic_number's element_table (two-character keys, a one-letter symbol padded with a blank). */
const ELEMENT_TABLE = [
  'h ', 'he', 'li', 'be', 'b ', 'c ', 'n ', 'o ', 'f ', 'ne', 'na', 'mg', 'al', 'si',
  'p ', 's ', 'cl', 'ar', 'k ', 'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni',
  'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y ', 'zr', 'nb', 'mo',
  'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i ', 'xe', 'cs', 'ba',
  'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb',
  'lu', 'hf', 'ta', 'w ', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po',
  'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u ', 'np', 'pu',
];

const DAMPING_MAP: Record<string, number> = { original: 1, zero: 1, zerom: 2, bj: 3, bjm: 4 };
const DAMPING_NAME: Record<number, string> = { 1: 'original', 2: 'zerom', 3: 'bj', 4: 'bjm' };

/** utils::numeric: a floating-point number or the native error. */
const numeric = (w: string): number => {
  const v = Number(w);
  if (w.trim() === '' || !/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(w) || !Number.isFinite(v)) {
    throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  }
  return v;
};

export class PairDispersionD3 extends Pair {
  readonly name = 'dispersion/d3';
  virialFdotr = true;
  private rthr = 0;
  private cnThr = 0;
  private dampingCode = 0;
  private s6 = 0; private s8 = 0; private rs6 = 0; private rs8 = 0;
  private a1 = 0; private a2 = 0; private alpha = 0;
  private nt = 0;
  private setflag = new Uint8Array(0);
  private mxci = new Int32Array(0);
  private r2r4 = new Float64Array(0);
  private rcov = new Float64Array(0);
  /** r0ab[(ntypes + 1)^2] */
  private r0ab = new Float64Array(0);
  /** c6ab[n+1][n+1][5][5][3], flattened */
  private c6ab = new Float64Array(0);
  private cn = new Float64Array(0);
  private dc6 = new Float64Array(0);
  private fold = new Float64Array(0);
  private c6res = new Float64Array(3);

  settings(args: string[], ctx: StyleContext): void {
    if (args.length !== 4) throw new StyleError('Pair style dispersion/d3 needs 4 arguments');
    if ((ctx.s?.units.style ?? 'lj') !== 'metal') throw new StyleError('Pair style dispersion/d3 requires metal units');
    const dampingType = args[0].toLowerCase();
    const code = DAMPING_MAP[dampingType];
    if (code === undefined) throw new StyleError(`Unknown damping type ${dampingType} for pair style dispersion/d3`);
    this.dampingCode = code;
    this.rthr = numeric(args[2]);
    this.cnThr = numeric(args[3]);
    this.rthr = this.rthr * this.rthr;
    this.cnThr = this.cnThr * this.cnThr;
    this.setFuncpar(args[1]);
  }

  /** set_funcpar: the defaults of the damping, the functional's assignments, then the unit conversion of rs8 or a2. */
  private setFuncpar(functionalName: string): void {
    const p = DAMPING_PARAMS[this.dampingCode];
    const set = (v: string, x: number) => { (this as unknown as Record<string, number>)[v] = x; };
    for (const [v, x] of p.defaults) set(v, x);
    // only the zero/original branch lowercases the name before the lookup (std::transform in case 1)
    const name = this.dampingCode === 1 ? functionalName.toLowerCase() : functionalName;
    const fcode = Object.prototype.hasOwnProperty.call(p.map, name) ? p.map[name] : 0;
    const assign = p.cases[fcode];
    if (!assign) {
      const label = this.dampingCode === 1 ? 'original' : DAMPING_NAME[this.dampingCode];
      throw new StyleError(`Functional ${name} not supported with ${label} damping function`);
    }
    for (const [v, x] of assign) set(v, x);
    if (this.dampingCode === 2) this.rs8 = this.rs8 / autoang;
    if (this.dampingCode === 3 || this.dampingCode === 4) this.a2 = this.a2 * autoang;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    this.nt = n;
    this.setflag = new Uint8Array(n * n);
    this.mxci = new Int32Array(n);
    this.r2r4 = new Float64Array(n);
    this.rcov = new Float64Array(n);
    this.r0ab = new Float64Array(n * n);
    this.c6ab = new Float64Array(n * n * 5 * 5 * 3);
  }

  private c6Index(i: number, j: number, gi: number, gj: number, k: number): number {
    return ((((i * this.nt + j) * 5 + gi) * 5 + gj) * 3) + k;
  }

  /** coeff: "pair_coeff * * element1 element2 ..." (one entry per atom type). */
  coeff(args: string[]): void {
    const ntypes = this.ntypes;
    if (args.length !== ntypes + 2) throw new StyleError('Pair_coeff * * needs: element1 element2 ...');
    const atomicNumbers: number[] = [];
    for (let i = 0; i < ntypes; i++) {
      let key = args[i + 2].toLowerCase();
      if (key.length === 1) key += ' ';
      const z = key.length > 2 ? -1 : ELEMENT_TABLE.indexOf(key);
      // find_atomic_number takes the key by reference: the message shows it lowercased (and padded)
      if (z < 0) throw new StyleError(`Element ${key} not supported`);
      atomicNumbers.push(z + 1);
    }
    const nt = this.nt;
    for (let i = 1; i <= ntypes; i++) for (let j = 1; j <= ntypes; j++) this.setflag[i * nt + j] = 1;
    const tab = d3Tables();
    for (let i = 1; i <= ntypes; i++) {
      this.r2r4[i] = tab.r2r4[atomicNumbers[i - 1]];
      this.rcov[i] = tab.rcov[atomicNumbers[i - 1]];
    }
    // read_r0ab
    for (let i = 1; i <= ntypes; i++) {
      for (let j = 1; j <= ntypes; j++) this.r0ab[i * nt + j] = tab.r0ab[(atomicNumbers[i - 1] - 1) * NUM_ELEMENTS + (atomicNumbers[j - 1] - 1)];
    }
    this.readC6ab(atomicNumbers, tab.c6ab);
  }

  /** read_c6ab: the reference C6 grid of every pair of types present (element number + 100 * grid index). */
  private readC6ab(atomicNumbers: number[], table: Float64Array): void {
    const ntypes = this.ntypes;
    this.mxci.fill(0);
    for (let i = 0; i < N_PARS_ROWS; i++) {
      const row = i * N_PARS_COLS;
      const refC6 = table[row];
      // (int)std::round(...) of a non-negative value
      let atom1 = Math.round(table[row + 1]);
      let atom2 = Math.round(table[row + 2]);
      let gridI = 0, gridJ = 0;
      while (atom1 > 100) { atom1 -= 100; gridI++; }
      while (atom2 > 100) { atom2 -= 100; gridJ++; }
      const idx1: number[] = [];
      for (let k = 0; k < ntypes; k++) if (atomicNumbers[k] === atom1) idx1.push(k + 1);
      if (!idx1.length) continue;
      const idx2: number[] = [];
      for (let k = 0; k < ntypes; k++) if (atomicNumbers[k] === atom2) idx2.push(k + 1);
      if (!idx2.length) continue;
      const refCn1 = table[row + 3];
      const refCn2 = table[row + 4];
      for (const a of idx1) {
        for (const b of idx2) {
          this.mxci[a] = Math.max(this.mxci[a], gridI);
          this.mxci[b] = Math.max(this.mxci[b], gridJ);
          this.c6ab[this.c6Index(a, b, gridI, gridJ, 0)] = refC6;
          this.c6ab[this.c6Index(a, b, gridI, gridJ, 1)] = refCn1;
          this.c6ab[this.c6Index(a, b, gridI, gridJ, 2)] = refCn2;
          this.c6ab[this.c6Index(b, a, gridJ, gridI, 0)] = refC6;
          this.c6ab[this.c6Index(b, a, gridJ, gridI, 1)] = refCn2;
          this.c6ab[this.c6Index(b, a, gridJ, gridI, 2)] = refCn1;
        }
      }
    }
  }

  initStyle(ctx: StyleContext): void {
    // init_style: "Pair style D3 requires atom IDs" (the engine's atoms always have IDs); a default (half) list
    void ctx;
  }

  initOne(i: number, j: number): number {
    if (!this.setflag[i * this.nt + j]) throw new StyleError('All pair coeffs are not set');
    this.r0ab[j * this.nt + i] = this.r0ab[i * this.nt + j];
    return Math.sqrt(this.rthr);
  }

  /** get_dC6: C6 and its derivatives in CN_i and CN_j, interpolated over the reference grid. */
  private getDC6(iat: number, jat: number, cni: number, cnj: number): Float64Array {
    const res = this.c6res;
    let c6mem = -1.0e20, rSave = 1.0e20;
    let num = 0, den = 0, dNumI = 0, dNumJ = 0, dDenI = 0, dDenJ = 0;
    const scale = autoev * libmPow(autoang, 6);
    for (let ci = 0; ci <= this.mxci[iat]; ci++) {
      for (let cj = 0; cj <= this.mxci[jat]; cj++) {
        let c6ref = this.c6ab[this.c6Index(iat, jat, ci, cj, 0)];
        c6ref *= scale;
        if (c6ref > 0) {
          const cniRef = this.c6ab[this.c6Index(iat, jat, ci, cj, 1)];
          const cnjRef = this.c6ab[this.c6Index(iat, jat, ci, cj, 2)];
          const r = (cni - cniRef) * (cni - cniRef) + (cnj - cnjRef) * (cnj - cnjRef);
          if (r < rSave) { rSave = r; c6mem = c6ref; }
          let expterm = libmExp(K3 * r);
          num += c6ref * expterm;
          den += expterm;
          expterm = expterm * 2.0 * K3;
          let term = expterm * (cni - cniRef);
          dNumI += c6ref * term;
          dDenI += term;
          term = expterm * (cnj - cnjRef);
          dNumJ += c6ref * term;
          dDenJ += term;
        }
      }
    }
    if (den > 1.0e-99) {
      res[0] = num / den;
      res[1] = ((dNumI * den) - (dDenI * num)) / (den * den);
      res[2] = ((dNumJ * den) - (dDenJ * num)) / (den * den);
    } else {
      res[0] = c6mem;
      res[1] = 0;
      res[2] = 0;
    }
    return res;
  }

  /** The values of the owned atoms summed with their ghosts (reverse_comm), then copied to the ghosts (forward_comm). */
  private commSum(pc: PairCompute, a: Float64Array): void {
    const nlocal = pc.nlocal;
    if (this.fold.length < nlocal) this.fold = new Float64Array(nlocal);
    const out = this.fold;
    out.fill(0, 0, nlocal);
    pc.nb.reverseSum(a, 1, out);
    a.set(out.subarray(0, nlocal));
    pc.nb.forwardCopy(a, 1);
  }

  compute(pc: PairCompute): void {
    const list = pc.half;
    if (!list) throw new Error('pair dispersion/d3 needs a half neighbor list');
    const { x, f, type } = pc;
    const nall = pc.nall;
    const nt = this.nt;
    if (this.cn.length < nall) {
      this.cn = new Float64Array(nall);
      this.dc6 = new Float64Array(nall);
    }
    const cn = this.cn, dc6 = this.dc6;
    cn.fill(0, 0, nall);
    dc6.fill(0, 0, nall);
    const nb = list.neighbors;
    const ilo = list.ilo ?? 0;
    const { rcov, r2r4, r0ab, cnThr } = this;
    // calc_coordination_number
    for (let i = ilo; i < list.inum; i++) {
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const jtype = type[j];
        const d0 = x[3 * i] - x[3 * j], d1 = x[3 * i + 1] - x[3 * j + 1], d2 = x[3 * i + 2] - x[3 * j + 2];
        const rsq = d0 * d0 + d1 * d1 + d2 * d2;
        if (rsq > cnThr) continue;
        const rr = Math.sqrt(rsq);
        const rcovIJ = (rcov[itype] + rcov[jtype]) * autoang;
        const cnIJ = 1.0 / (1.0 + libmExp(-K1 * ((rcovIJ / rr) - 1.0)));
        cn[i] += cnIJ;
        cn[j] += cnIJ;
      }
    }
    this.commSum(pc, cn);

    const sLJ = pc.specialLJ;
    const cutsq = this.cutsq;
    const { s6, s8, rs6, rs8, a1, a2, alpha } = this;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwlSum = 0;
    for (let i = ilo; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const ti = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const tj = type[j];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (!(rsq < cutsq[ti * nt + tj])) continue;
        const r = Math.sqrt(rsq);
        const r2inv = 1.0 / rsq;
        const r6inv = r2inv * r2inv * r2inv;
        const r8inv = r2inv * r2inv * r2inv * r2inv;
        const r10inv = r2inv * r2inv * r2inv * r2inv * r2inv;
        const c6res = this.getDC6(ti, tj, cn[i], cn[j]);
        const C6 = c6res[0];
        const C8 = 3.0 * C6 * r2r4[ti] * r2r4[tj] * autoang * autoang;
        const alpha6 = alpha;
        const alpha8 = alpha + 2;
        let t6 = 0, t8 = 0, damp6 = 0, damp8 = 0, e6 = 0, e8 = 0, tmp6 = 0, tmp8 = 0, fpair1 = 0, fpair2 = 0, fpair = 0;
        switch (this.dampingCode) {
          case 1: {
            const r0 = r / r0ab[ti * nt + tj];
            t6 = libmPow(rs6 / r0, alpha6);
            damp6 = 1.0 / (1.0 + 6.0 * t6);
            t8 = libmPow(rs8 / r0, alpha8);
            damp8 = 1.0 / (1.0 + 6.0 * t8);
            e6 = C6 * damp6 * r6inv;
            e8 = C8 * damp8 * r8inv;
            tmp6 = 6 * s6 * C6 * r8inv * damp6;
            tmp8 = 8 * s8 * C8 * r10inv * damp8;
            fpair1 = -tmp6 - tmp8;
            fpair2 = tmp6 * alpha6 * t6 * damp6 + (3.0 / 4.0) * tmp8 * alpha8 * t8 * damp8;
            fpair = fpair1 + fpair2;
            fpair *= factorLJ;
            break;
          }
          case 2: {
            const r0 = r0ab[ti * nt + tj];
            t6 = libmPow((r / (rs6 * r0)) + rs8 * r0, -alpha6);
            damp6 = 1.0 / (1.0 + 6.0 * t6);
            t8 = libmPow((r / r0) + rs8 * r0, -alpha8);
            damp8 = 1.0 / (1.0 + 6.0 * t8);
            e6 = C6 * damp6 * r6inv;
            e8 = C8 * damp8 * r8inv;
            tmp6 = 6 * s6 * C6 * r8inv * damp6;
            tmp8 = 8 * s8 * C8 * r10inv * damp8;
            fpair1 = -tmp6 - tmp8;
            const fp26 = tmp6 * alpha6 * t6 * damp6 * r / (r + rs6 * rs8 * r0 * r0);
            const fp28 = tmp8 * alpha8 * t8 * damp8 * r / (r + rs8 * r0 * r0);
            fpair2 = fp26 + (3.0 / 4.0) * fp28;
            fpair = fpair1 + fpair2;
            fpair *= factorLJ;
            break;
          }
          case 3:
          case 4: {
            // bj and bjm have the same form
            const r0 = Math.sqrt(C8 / C6);
            const r4 = rsq * rsq;
            const r6 = rsq * rsq * rsq;
            const r8 = rsq * rsq * rsq * rsq;
            t6 = r6 + libmPow((a1 * r0 + a2), 6);
            t8 = r8 + libmPow((a1 * r0 + a2), 8);
            e6 = C6 / t6;
            e8 = C8 / t8;
            tmp6 = 6.0 * s6 * C6 * r4 / (t6 * t6);
            tmp8 = 8.0 * s8 * C8 * r6 / (t8 * t8);
            fpair = -(tmp6 + tmp8);
            fpair *= factorLJ;
            break;
          }
          default:
            throw new StyleError(`Damping code ${this.dampingCode} unknown`);
        }
        const evdwl = -(s6 * e6 + s8 * e8) * factorLJ;
        evdwlSum += evdwl;
        const rest = (s6 * e6 + s8 * e8) / C6;
        dc6[i] += rest * c6res[1];
        dc6[j] += rest * c6res[2];
        f[3 * i] += delx * fpair; f[3 * i + 1] += dely * fpair; f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair; f[3 * j + 1] -= dely * fpair; f[3 * j + 2] -= delz * fpair;
        if (tally) tallyAtom(pc, i, j, evdwl, fpair, delx, dely, delz);
      }
    }
    this.commSum(pc, dc6);

    // the gradient of the CN-dependent C6: dE/dr_ij through dCN/dr
    for (let i = ilo; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const ti = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const tj = type[j];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (!(rsq < cutsq[ti * nt + tj])) continue;
        const r = Math.sqrt(rsq);
        let dcn: number;
        if (rsq < cnThr) {
          const rcovij = (rcov[ti] + rcov[tj]) * autoang;
          const expterm = libmExp(-K1 * (rcovij / r - 1.0));
          dcn = -K1 * rcovij * expterm / (rsq * (expterm + 1.0) * (expterm + 1.0));
        } else {
          dcn = 0.0;
        }
        let fpair = dcn * (dc6[i] + dc6[j]) / r;
        fpair *= factorLJ;
        f[3 * i] += delx * fpair; f[3 * i + 1] += dely * fpair; f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair; f[3 * j + 1] -= dely * fpair; f[3 * j + 2] -= delz * fpair;
        if (tally) tallyAtom(pc, i, j, 0.0, fpair, delx, dely, delz);
      }
    }
    pc.acc.evdwl += evdwlSum;
  }
}
