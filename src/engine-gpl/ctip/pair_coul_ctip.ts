/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-PAIR/pair_coul_ctip.cpp and src/EXTRA-PAIR/pair_coul_ctip.h (style coul/ctip: compute, self,
 * init_style, read_file, setup_params, extract), at commit b332f28. Author credited in the C++ file headers:
 * Gabriel Plummer (NASA). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * pair_style coul/ctip alpha cutoff (docs.lammps.org/pair_coul.html, section coul/ctip): "Style *coul/ctip* computes the
 * Coulomb interactions as described in Plummer. ... Like the style *coul/streitz*, style *coul/ctip* is a variable charge
 * potential and must be hybridized with a short-range potential via the pair_style hybrid/overlay command." Used as
 * pair_style hybrid/overlay eam/fs coul/ctip 0.30 12.0 (examples/streitz/in.ctip).
 * Its per-type parameters (chi, eta, gamma, zeta, zcore, qmin, qmax, omega) are read by fix qeq/ctip
 * (src/engine-gpl/qeq/fix_qeq_ctip.ts) through typeParams(). The shared constants and the parameter file reader are
 * exported for that fix.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';

/** ewald_const.h (EWALD_P, A1 ... A5): the erfc approximation of coul/ctip and fix qeq/ctip. */
export const EWALD_P = 0.3275911;
export const A1 = 0.254829592;
export const A2 = -0.284496736;
export const A3 = 1.421413741;
export const A4 = -1.453152027;
export const A5 = 1.061405429;
/** math_const.h MY_PIS = sqrt(pi). */
export const MY_PIS = 1.77245385090551602729;

/** Per element: the coul/ctip file columns (pair_coul_ctip.cpp read_file: element chi eta gamma zeta zcore qmin qmax omega). */
export interface CtipParams {
  chi: number;
  eta: number;
  gamma: number;
  zeta: number;
  zcore: number;
  qmin: number;
  qmax: number;
  omega: number;
}

/** Parses a coul/ctip potential file (element, then 8 values per line, '#' comments), with native's sanity checks. */
export const parseCtipFile = (text: string, fileName: string): Map<string, CtipParams> => {
  const out = new Map<string, CtipParams>();
  text.split(/\r?\n/).forEach((raw, k) => {
    const h = raw.indexOf('#');
    const line = (h >= 0 ? raw.slice(0, h) : raw).trim();
    if (!line) return;
    const w = line.split(/\s+/);
    if (w.length < 9) throw new StyleError(`coul/ctip potential file ${fileName} line ${k + 1}: expected 9 words (element chi eta gamma zeta zcore qmin qmax omega), got ${w.length}`);
    const v = w.slice(1, 9).map((x, c) => parseNum(x, `${['chi', 'eta', 'gamma', 'zeta', 'zcore', 'qmin', 'qmax', 'omega'][c]} for ${w[0]} in ${fileName}`));
    const p: CtipParams = { chi: v[0], eta: v[1], gamma: v[2], zeta: v[3], zcore: v[4], qmin: v[5], qmax: v[6], omega: v[7] };
    if (p.eta < 0 || p.zeta < 0 || p.zcore < 0 || p.gamma < 0 || p.qmin > p.qmax || p.omega < 0) {
      throw new StyleError(`Illegal coul/ctip parameter in ${fileName} for element ${w[0]}`);
    }
    if (out.has(w[0])) throw new StyleError(`Potential file ${fileName} has duplicate entry for: ${w[0]}`);
    out.set(w[0], p);
  });
  if (out.size === 0) throw new StyleError(`coul/ctip potential file ${fileName} has no entries`);
  return out;
};

/** The self energy of one atom with charge qi (pair_coul_ctip.cpp self): the charge-regime polynomial. */
export const ctipSelf = (p: CtipParams, qi: number): number => {
  const s1 = p.chi, s2 = p.eta, s3 = p.qmin, s4 = p.qmax, s5 = p.omega;
  if (qi < s3) return qi * ((s1 - 2 * s3 * s5) + qi * (0.5 * s2 + s5)) + s3 * s3 * s5;
  if (qi < s4) return qi * (s1 + qi * (0.5 * s2));
  return qi * ((s1 - 2 * s4 * s5) + qi * (0.5 * s2 + s5)) + s4 * s4 * s5;
};

/** Pair coul/ctip (pair_coul_ctip.cpp): the shielded Coulomb term with its shift constants, per type pair. */
export class PairCoulCtip extends Pair {
  readonly name = 'coul/ctip';
  virialFdotr = true;
  private alpha = 0;
  private cutCoul = 0;
  private fileName = '';
  private fileRead = false;
  private entries = new Map<string, CtipParams>();
  /** Per atom type (1..ntypes) parameters; null until pair_coeff sets them. */
  private tp: (CtipParams | null)[] = [];
  /** Per type pair (t1 * nt + t2): shielded constants at the cutoff (init_style). */
  private shieldCu = new Float64Array(0);
  private reffc = new Float64Array(0);
  private reffc4 = new Float64Array(0);
  private reffc7 = new Float64Array(0);
  private s2dShift = new Float64Array(0);
  private fShift = new Float64Array(0);
  private eShift = new Float64Array(0);
  /** e_shift / 2 + alpha / sqrt(pi); the self-factor is -selfHalf * qqrd2e (pair_coul_ctip.cpp init_style). */
  private selfHalf = new Float64Array(0);

  /** The parameters of atom type `type` (for fix qeq/ctip). */
  typeParams(type: number): CtipParams {
    const p = this.tp[type];
    if (!p) throw new StyleError(`coul/ctip has no parameters for atom type ${type} (pair_coeff not set)`);
    return p;
  }

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('Illegal pair_style coul/ctip command: usage pair_style coul/ctip alpha cutoff');
    this.alpha = parseNum(args[0], 'alpha');
    this.cutCoul = parseNum(args[1], 'cutoff');
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError('usage: pair_coeff * * NiO.ctip elem1 ... elemN (style coul/ctip)');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style coul/ctip must be * *');
    const filename = args[2];
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style coul/ctip needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    }
    if (!this.fileRead) {
      this.entries = parseCtipFile(ctx.readFile(filename), filename);
      this.fileName = filename;
      this.fileRead = true;
    } else if (filename !== this.fileName) {
      throw new StyleError(`only one pair_coeff command is allowed for style coul/ctip (potential file already set to ${this.fileName})`);
    }
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') throw new StyleError('pair_coeff for style coul/ctip does not accept NULL types');
      const e = this.entries.get(name);
      if (!e) throw new StyleError(`element '${name}' is not in coul/ctip potential file ${this.fileName} (elements: ${[...this.entries.keys()].sort().join(' ')})`);
      this.tp[t] = e;
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.tp = new Array(ntypes + 1).fill(null);
  }

  initStyle(_ctx: StyleContext): void {
    if (!this.fileRead) throw new StyleError('pair_style coul/ctip needs a pair_coeff command with the potential file');
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.tp[t]) throw new StyleError(`pair_coeff for style coul/ctip does not set atom type ${t}`);
    }
    const nt = this.ntypes + 1;
    const n2 = nt * nt;
    this.shieldCu = new Float64Array(n2);
    this.reffc = new Float64Array(n2);
    this.reffc4 = new Float64Array(n2);
    this.reffc7 = new Float64Array(n2);
    this.s2dShift = new Float64Array(n2);
    this.fShift = new Float64Array(n2);
    this.eShift = new Float64Array(n2);
    this.selfHalf = new Float64Array(n2);
    const a = this.alpha, rc = this.cutCoul;
    const cutCoulsq = rc * rc;
    const cutCoulcu = cutCoulsq * rc;
    const cut4 = cutCoulsq * cutCoulsq;
    const alphacu = a * a * a;
    const erfcdCut = Math.exp(-a * a * cutCoulsq);
    const tCut = 1.0 / (1.0 + EWALD_P * a * rc);
    const erfccCut = tCut * (A1 + tCut * (A2 + tCut * (A3 + tCut * (A4 + tCut * A5)))) * erfcdCut;
    for (let t1 = 1; t1 <= this.ntypes; t1++) {
      for (let t2 = 1; t2 <= this.ntypes; t2++) {
        const k = t1 * nt + t2;
        const shield = Math.sqrt(this.tp[t1]!.gamma * this.tp[t2]!.gamma);
        const shieldcu = shield * shield * shield;
        this.shieldCu[k] = shieldcu;
        this.reffc[k] = Math.cbrt(cutCoulcu + 1.0 / shieldcu);
        const reffcsq = this.reffc[k] * this.reffc[k];
        this.reffc4[k] = reffcsq * reffcsq;
        this.reffc7[k] = this.reffc4[k] * reffcsq * this.reffc[k];
        this.s2dShift[k] = 2.0 * erfccCut / cutCoulcu + 4.0 * a / MY_PIS * erfcdCut / cutCoulsq + 4.0 * alphacu / MY_PIS * erfcdCut -
          2.0 / cutCoulcu + 4.0 * cut4 / this.reffc7[k] - 2.0 * rc / this.reffc4[k];
        this.fShift[k] = erfccCut / cutCoulsq + 2.0 * a / MY_PIS * erfcdCut / rc - 1.0 / cutCoulsq + cutCoulsq / this.reffc4[k];
        this.eShift[k] = 2.0 * erfccCut / rc + 2.0 * a / MY_PIS * erfcdCut - 2.0 / rc + 1.0 / this.reffc[k] + cutCoulcu / this.reffc4[k] +
          this.s2dShift[k] * cutCoulsq * 0.5;
        this.selfHalf[k] = this.eShift[k] * 0.5 + a / MY_PIS;
      }
    }
  }

  initOne(i: number, j: number): number {
    this.typeParams(i);
    this.typeParams(j);
    return this.cutCoul;
  }

  compute(pc: PairCompute): void {
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const C = pc.qqrd2e;
    const a = this.alpha, rc = this.cutCoul;
    const cutCoulsq = rc * rc;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const sC = pc.specialCoul;
    const erfcdCut = Math.exp(-a * a * cutCoulsq);
    const tCut = 1.0 / (1.0 + EWALD_P * a * rc);
    const erfccCut = tCut * (A1 + tCut * (A2 + tCut * (A3 + tCut * (A4 + tCut * A5)))) * erfcdCut;
    let ecoul = 0;
    for (let i = 0; i < pc.nlocal; i++) {
      const p = this.typeParams(type[i]);
      const qi = q[i];
      const selfion = ctipSelf(p, qi);
      const k = type[i] * nt + type[i];
      const eSelf = -this.selfHalf[k] * C * qi * qi;
      ecoul += selfion + eSelf;
      if (pc.eatom) pc.eatom[i] += selfion + eSelf;
    }
    const list = pc.half!;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = q[i];
      const ti = type[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      for (let kk = list.firstneigh[i], k1 = kk + list.numneigh[i]; kk < k1; kk++) {
        const jj = nb[kk];
        const j = jj & NEIGHMASK;
        const factor = sC[jj >>> SBBITS];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (!(rsq < cutCoulsq)) continue;
        const tj = type[j];
        const k = ti * nt + tj;
        const qj = q[j];
        const r = Math.sqrt(rsq);
        const reff = Math.cbrt(rsq * r + 1 / this.shieldCu[k]);
        const reffsq = reff * reff;
        const reff4 = reffsq * reffsq;
        const prefactor = C * qi * qj / r;
        const erfcd = Math.exp(-a * a * rsq);
        const t = 1.0 / (1.0 + EWALD_P * a * r);
        const erfcc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * erfcd;
        let forcecoul = prefactor *
          (erfcc / r + 2.0 * a / MY_PIS * erfcd + rsq * r / reff4 - 1 / r - r * this.fShift[k] + r * this.s2dShift[k] * (r - rc)) * r;
        if (factor < 1.0) forcecoul -= (1.0 - factor) * prefactor;
        const fpair = forcecoul / rsq;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        let e = prefactor *
          (erfcc + r / reff - 1 - r * erfccCut / rc - r / this.reffc[k] + r / rc + r * this.fShift[k] * (r - rc) -
            r * this.s2dShift[k] * (r - rc) * (r - rc) * 0.5);
        if (factor < 1.0) e -= (1.0 - factor) * prefactor;
        ecoul += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.ecoul += ecoul;
  }
}
