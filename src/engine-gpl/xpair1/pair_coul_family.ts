/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_coul_cut_global.cpp
 * (and .h), src/EXTRA-PAIR/pair_coul_slater_cut.cpp (and .h; Contributing author: Evangelos Voyiatzis, Royal DSM) and
 * src/EXTRA-PAIR/pair_coul_exclude.cpp (and .h). The coul/cut base is src/pair_coul_cut.cpp (b332f28). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/pair_coul.html (coul/cut/global, coul/exclude) and docs.lammps.org/pair_coul_slater.html (coul/slater/cut):
 *   coul/cut/global cutoff     pair_coeff I J   (the global cutoff, no per-pair cutoff word)
 *   coul/slater/cut lamda cutoff
 *   coul/exclude cutoff        for the special pairs only: adds (factor_coul - 1) times the bare Coulomb term, which
 *                              removes the excluded part of the force field's Coulomb interaction.
 *
 * Each compute() keeps native's operation order, with fix adapt's scale[itype][jtype] of coul/cut where native has it
 * (the table is the parent class's, engine/force/pair/coul.ts; native single() leaves it out).
 */
import { Pair, PairParams, StyleError, typeBounds, type PairCompute, type PairSep, type StyleContext } from '../../engine/force/types';
import { resetScale } from '../../engine/force/pair/scaleTable';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { PairCoulCut } from '../../engine/force/pair/coul';
import { parseNum } from '../../engine/force/util';
import { exp } from '../libm/exp';
import { setPairRange, resetSetCutoffs } from './common';

/** pair_coul_cut.cpp coeff(): "pair_coeff I J [cutoff]" with narg 2 or 3 (coul/slater/cut), or narg 2 (coul/cut/global). */
const coulCutCoeff = (
  p: PairParams, args: string[], minNarg: number, maxNarg: number, cutGlobal: number, ntypes: number, scale: Float64Array,
): void => {
  if (args.length < minNarg || args.length > maxNarg) throw new StyleError('Incorrect args for pair coefficients');
  const cut = args.length === 3 ? parseNum(args[2], 'cutoff') : cutGlobal;
  setPairRange(p, args[0], args[1], [cut], 'Incorrect args for pair coefficients', ntypes);
  // and scale[i][j] = 1.0 for the same pairs
  const [ilo, ihi] = typeBounds(args[0], ntypes), [jlo, jhi] = typeBounds(args[1], ntypes);
  for (let i = ilo; i <= ihi; i++) for (let j = Math.max(jlo, i); j <= jhi; j++) resetScale(scale, ntypes, i, j);
};

/** pair_style coul/cut/global cutoff (pair_coul_cut_global.cpp: PairCoulCut with a coeff() that takes two words). */
export class PairCoulCutGlobal extends PairCoulCut {
  readonly name: string = 'coul/cut/global';
  qqrd2eLast = 1;

  settings(args: string[]): void {
    // pair_coul_cut.cpp settings(): "Illegal pair_style command" unless one word
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    resetSetCutoffs(this.p, 'cut', this.cutGlobal, this.ntypes);
  }

  coeff(args: string[]): void {
    // pair_coul_cut_global.cpp coeff(): "if (narg != 2) error->all(...Incorrect args for pair coefficients...)"
    if (args.length !== 2) throw new StyleError('Incorrect args for pair coefficients');
    coulCutCoeff(this.p, args, 2, 2, this.cutGlobal, this.ntypes, this.scale);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    this.qqrd2eLast = qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = q[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq < cutsq[t])) continue;
        const factor = sC[jj >>> SBBITS];
        const r2inv = 1.0 / rsq;
        const rinv = Math.sqrt(r2inv);
        const forcecoul = qqrd2e * this.scale[t] * qi * q[j] * rinv;
        const fpair = factor * forcecoul * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        const e = factor * qqrd2e * this.scale[t] * qi * q[j] * rinv;
        ecoul += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, _itype: number, _jtype: number, rsq: number, factorCoul: number, _factorLJ: number, qi: number, qj: number, _del?: PairSep) {
    const r2inv = 1.0 / rsq;
    const rinv = Math.sqrt(r2inv);
    const forcecoul = this.qqrd2eLast * qi * qj * rinv;
    const fforce = factorCoul * forcecoul * r2inv;
    const phicoul = this.qqrd2eLast * qi * qj * rinv;
    return { eng: factorCoul * phicoul, fforce };
  }

  extract(name: string): unknown {
    // pair_coul_cut_global.cpp extract(): cut_coul is the global cutoff (dim 0); scale (dim 2)
    if (name === 'cut_coul') return this.cutGlobal;
    if (name === 'scale') return this.scale;
    return undefined;
  }
}

/** pair_style coul/slater/cut lamda cutoff (pair_coul_slater_cut.cpp: PairCoulCut with the Slater-smeared bracket). */
export class PairCoulSlaterCut extends PairCoulCut {
  readonly name: string = 'coul/slater/cut';
  lamda = 0;
  qqrd2eLast = 1;

  settings(args: string[]): void {
    // pair_coul_slater_cut.cpp settings(): "if (narg != 2) error->all(FLERR,"Illegal pair_style command")"
    if (args.length !== 2) throw new StyleError('Illegal pair_style command');
    this.lamda = parseNum(args[0], 'lamda');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    resetSetCutoffs(this.p, 'cut', this.cutGlobal, this.ntypes);
  }

  coeff(args: string[]): void {
    // inherited from pair_coul_cut.cpp coeff(): narg 2 or 3
    coulCutCoeff(this.p, args, 2, 3, this.cutGlobal, this.ntypes, this.scale);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    this.qqrd2eLast = qqrd2e;
    const lamda = this.lamda;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = q[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq < cutsq[t])) continue;
        const factor = sC[jj >>> SBBITS];
        const r2inv = 1.0 / rsq;
        const r = Math.sqrt(rsq);
        const rinv = 1.0 / r;
        const bracketTerm = 1 - exp(-2 * r / lamda) * (1 + (2 * r / lamda * (1 + r / lamda)));
        const forcecoul = qqrd2e * this.scale[t] * qi * q[j] * bracketTerm * rinv;
        const fpair = factor * forcecoul * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        const e = factor * qqrd2e * this.scale[t] * qi * q[j] * rinv * (1 - (1 + r / lamda) * exp(-2 * r / lamda));
        ecoul += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, _itype: number, _jtype: number, rsq: number, factorCoul: number, _factorLJ: number, qi: number, qj: number, _del?: PairSep) {
    const r2inv = 1.0 / rsq;
    const r = Math.sqrt(rsq);
    const rinv = 1.0 / r;
    const bracketTerm = 1 - exp(-2 * r / this.lamda) * (1 + (2 * r / this.lamda * (1 + r / this.lamda)));
    const forcecoul = this.qqrd2eLast * qi * qj * bracketTerm * rinv;
    const fforce = factorCoul * forcecoul * r2inv;
    const phicoul = this.qqrd2eLast * qi * qj * rinv * (1 - (1 + r / this.lamda) * exp(-2 * r / this.lamda));
    return { eng: factorCoul * phicoul, fforce };
  }

  extract(name: string): unknown {
    // inherited from pair_coul_cut.cpp extract(): cut_coul is the per-pair cutoff table (dim 2); scale likewise
    if (name === 'cut_coul') return this.p.p('cut');
    if (name === 'scale') return this.scale;
    return undefined;
  }
}

/** pair_style coul/exclude cutoff (pair_coul_exclude.cpp): only the special pairs, with the factor (special - 1). */
export class PairCoulExclude extends Pair {
  readonly name: string = 'coul/exclude';
  virialFdotr = true;
  /** The excluded (weight 0) special pairs stay in the list: the style is what subtracts them. */
  keepExcluded = true;
  cutGlobal = 0;
  qqrd2eLast = 1;
  p!: PairParams;

  settings(args: string[]): void {
    // pair_coul_exclude.cpp settings(): "Illegal pair_style command" unless one word
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['flag']);
  }

  coeff(args: string[]): void {
    // pair_coul_exclude.cpp coeff(): "pair_coeff I J" only, which records setflag for the pair
    if (args.length !== 2) throw new StyleError('Incorrect args for pair coefficients');
    setPairRange(this.p, args[0], args[1], [1], 'Incorrect args for pair coefficients', this.ntypes);
  }

  initOne(_i: number, _j: number): number {
    // init_one() returns cut_global for every pair (no check that the pair was set)
    return this.cutGlobal;
  }

  compute(pc: PairCompute): void {
    // pair_coul_exclude.cpp compute(): no cutoff test; the special pairs only (sbmask(j) == 0 is skipped)
    const list = pc.half!;
    const { x, f, q } = pc;
    const sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    this.qqrd2eLast = qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qtmp = q[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const sb = jj >>> SBBITS;
        if (sb === 0) continue;
        const factor = sC[sb] - 1.0;
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const r2inv = 1.0 / rsq;
        const rinv = Math.sqrt(r2inv);
        const forcecoul = qqrd2e * qtmp * q[j] * rinv;
        const fpair = factor * forcecoul * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        const e = factor * qqrd2e * qtmp * q[j] * rinv;
        ecoul += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, _itype: number, _jtype: number, rsq: number, factorCoul: number, _factorLJ: number, qi: number, qj: number, _del?: PairSep) {
    const fc = factorCoul - 1.0;
    const r2inv = 1.0 / rsq;
    const rinv = Math.sqrt(r2inv);
    const forcecoul = this.qqrd2eLast * qi * qj * rinv;
    const fforce = fc * forcecoul * r2inv;
    const phicoul = this.qqrd2eLast * qi * qj * rinv;
    return { eng: fc * phicoul, fforce };
  }

  dataCoeffs(): string[] | null {
    // pair_coul_exclude.cpp write_data(): the type numbers only
    return Array.from({ length: this.ntypes }, (_, k) => `${k + 1}`);
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) for (let j = i; j <= this.ntypes; j++) out.push(`${i} ${j}`);
    return out;
  }

  extract(name: string): unknown {
    // pair_coul_exclude.cpp extract(): cut_coul (dim 0)
    return name === 'cut_coul' ? this.cutGlobal : undefined;
  }
}
