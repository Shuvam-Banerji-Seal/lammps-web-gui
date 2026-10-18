/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_nm_cut.cpp,
 * pair_nm_cut_split.cpp, pair_nm_cut_coul_cut.cpp and pair_nm_cut_coul_long.cpp with their headers (PairNMCut,
 * PairNMCutSplit, PairNMCutCoulCut, PairNMCutCoulLong: pair_style nm/cut, nm/cut/split, nm/cut/coul/cut, nm/cut/coul/long).
 * The source names its author (Tom Mattsson / Sandia National Laboratories, the N-M potential). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2; this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The N-M potential (docs.lammps.org/pair_nm.html): E = E0/(n-m) [ m (r0/r)^n - n (r0/r)^m ] with E0 the well depth,
 * r0 the minimum, n and m the exponents; nm = n m; e0nm = E0/(n - m). The Coulomb parts of nm/cut/coul/cut and
 * nm/cut/coul/long are the plain and the Ewald-damped (pair_coul.html) terms.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, type PairSep } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum, fmtCoeff } from '../../engine/force/util';
import { hasCharge } from '../../engine/atoms';
import { ErfcTableCache, type ErfcTable } from '../../engine/force/erfc';
import { coulLongPair } from '../../engine/force/pair/coul_long';
import { pow } from '../libm/pow';
import { forEachTypePair, powint } from './common';

/** Pair::init tail sums of one type pair (i <= j): the native etail_ij and ptail_ij of the nm/cut family. */
const nmTailPair = (ni: number, nj: number, cut: number, r0: number, nn: number, mm: number, e0nm: number, nm: number): { etail: number; ptail: number } => {
  const cut3 = cut * cut * cut;
  const ptail = 2. * Math.PI / 3. * ni * nj * e0nm * nm * cut3 *
    (pow(r0 / cut, nn) / (nn - 3) - pow(r0 / cut, mm) / (mm - 3));
  const etail = 2. * Math.PI * ni * nj * e0nm * cut3 *
    (mm * pow(r0 / cut, nn) / (nn - 3) - nn * pow(r0 / cut, mm) / (mm - 3));
  return { etail, ptail };
};

/** The native init_one() offset of the nm/cut family (shift yes): e0nm (mm r0n / cut^nn - nn r0m / cut^mm). */
const nmOffset = (e0nm: number, mm: number, nn: number, r0n: number, r0m: number, cut: number): number =>
  e0nm * ((mm * r0n / pow(cut, nn)) - (nn * r0m / pow(cut, mm)));

export class PairNMCut extends Pair {
  readonly name: string = 'nm/cut';
  virialFdotr = true;
  cutGlobal = 0;
  protected p!: PairParams;
  protected nm = new Float64Array(0);
  protected e0nm = new Float64Array(0);
  protected r0n = new Float64Array(0);
  protected r0m = new Float64Array(0);
  protected offset = new Float64Array(0);

  settings(args: string[]): void {
    // pair_nm_cut.cpp settings(): if (narg != 1) error->all(FLERR,"Illegal pair_style command");
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (this.p) for (let i = 1; i <= this.ntypes; i++) for (let j = i; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut', this.cutGlobal);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['e0', 'r0', 'nn', 'mm', 'cut']);
    const n = (ntypes + 1) * (ntypes + 1);
    this.nm = new Float64Array(n); this.e0nm = new Float64Array(n);
    this.r0n = new Float64Array(n); this.r0m = new Float64Array(n); this.offset = new Float64Array(n);
  }

  coeff(args: string[]): void {
    // pair_nm_cut.cpp coeff(): narg 6 or 7 (I J E0 r0 n m [cutoff])
    if (args.length < 6 || args.length > 7) throw new StyleError('Incorrect args for pair coefficients');
    const e0 = parseNum(args[2], 'E0');
    const r0 = parseNum(args[3], 'r0');
    const nn = parseNum(args[4], 'n');
    const mm = parseNum(args[5], 'm');
    const cutOne = args.length === 7 ? parseNum(args[6], 'cutoff') : this.cutGlobal;
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, [e0, r0, nn, mm, cutOne]));
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError('All pair coeffs are not set');
    const nt = this.ntypes + 1;
    const k = i * nt + j, kr = j * nt + i;
    const e0 = p.get('e0', i, j), r0 = p.get('r0', i, j), nn = p.get('nn', i, j), mm = p.get('mm', i, j), cut = p.get('cut', i, j);
    this.nm[k] = nn * mm;
    this.e0nm[k] = e0 / (nn - mm);
    this.r0n[k] = pow(r0, nn);
    this.r0m[k] = pow(r0, mm);
    this.offset[k] = this.shift && cut > 0.0 ? nmOffset(this.e0nm[k], mm, nn, this.r0n[k], this.r0m[k], cut) : 0.0;
    this.nm[kr] = this.nm[k]; this.e0nm[kr] = this.e0nm[k]; this.r0n[kr] = this.r0n[k];
    this.r0m[kr] = this.r0m[k]; this.offset[kr] = this.offset[k];
    p.setMixed(j, i, 'e0', e0); p.setMixed(j, i, 'r0', r0); p.setMixed(j, i, 'nn', nn);
    p.setMixed(j, i, 'mm', mm); p.setMixed(j, i, 'cut', cut);
    return cut;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    let etail = 0, ptail = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = i; j < nt; j++) {
        const k = i * nt + j;
        const t = nmTailPair(count[i], count[j], this.p.get('cut', i, j), this.p.get('r0', i, j), this.p.get('nn', i, j), this.p.get('mm', i, j), this.e0nm[k], this.nm[k]);
        etail += t.etail; ptail += t.ptail;
        if (i !== j) { etail += t.etail; ptail += t.ptail; }
      }
    }
    return { etail, ptail };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, nm, e0nm, r0n, r0m, offset } = this;
    const nn = this.p.p('nn'), mm = this.p.p('mm');
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = itype * nt + type[j];
        if (rsq < cutsq[t]) {
          const r2inv = 1.0 / rsq;
          const r = Math.sqrt(rsq);
          const rminv = pow(r2inv, mm[t] / 2.0);
          const rninv = pow(r2inv, nn[t] / 2.0);
          const forcenm = e0nm[t] * nm[t] * (r0n[t] / pow(r, nn[t]) - r0m[t] / pow(r, mm[t]));
          const fpair = factorLJ * forcenm * r2inv;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          let e = e0nm[t] * (mm[t] * r0n[t] * rninv - nn[t] * r0m[t] * rminv) - offset[t];
          e *= factorLJ;
          evdwl += e;
          if (tally) tallyAtom(pc, i, j, e, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, _del?: PairSep) {
    const t = itype * (this.ntypes + 1) + jtype;
    const nn = this.p.get('nn', itype, jtype), mm = this.p.get('mm', itype, jtype);
    const r2inv = 1.0 / rsq;
    const r = Math.sqrt(rsq);
    const forcenm = this.e0nm[t] * this.nm[t] * (this.r0n[t] / pow(r, nn) - this.r0m[t] / pow(r, mm));
    const fforce = factorLJ * forcenm * r2inv;
    const phinm = this.e0nm[t] * (mm * this.r0n[t] / pow(r, nn) - nn * this.r0m[t] / pow(r, mm)) - this.offset[t];
    return { eng: factorLJ * phinm, fforce };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('e0', i, i))} ${fmtCoeff(this.p.get('r0', i, i))} ${fmtCoeff(this.p.get('nn', i, i))} ${fmtCoeff(this.p.get('mm', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('e0', i, j))} ${fmtCoeff(this.p.get('r0', i, j))} ${fmtCoeff(this.p.get('nn', i, j))} ${fmtCoeff(this.p.get('mm', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'e0' || name === 'r0' || name === 'nn' || name === 'mm') return this.p.p(name);
    return undefined;
  }
}

/**
 * pair_style nm/cut/split (pair_nm_cut_split.cpp): the N-M form inside r0, the standard 12-6 Lennard-Jones outside it.
 * Settings, coefficients, mixing and the data-file lines are those of nm/cut (PairNMCutSplit : PairNMCut in the source).
 */
export class PairNMCutSplit extends PairNMCut {
  override readonly name: string = 'nm/cut/split';

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, nm, e0nm, r0n, r0m, offset } = this;
    const nn = this.p.p('nn'), mm = this.p.p('mm'), e0 = this.p.p('e0'), r0 = this.p.p('r0');
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = itype * nt + type[j];
        if (rsq < cutsq[t]) {
          const r2inv = 1.0 / rsq;
          const r = Math.sqrt(rsq);
          let forcenm: number;
          // r < r0 --> use generalized LJ
          if (rsq < r0[t] * r0[t]) {
            forcenm = e0nm[t] * nm[t] * (r0n[t] / pow(r, nn[t]) - r0m[t] / pow(r, mm[t]));
          } else {
            // r > r0 --> use standard LJ (m = 6 n = 12)
            forcenm = (e0[t] / 6.0) * 72.0 * (4.0 / powint(r, 12) - 2.0 / powint(r, 6));
          }
          const fpair = factorLJ * forcenm * r2inv;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          let e: number;
          if (rsq < r0[t] * r0[t]) {
            const rminv = pow(r2inv, mm[t] / 2.0);
            const rninv = pow(r2inv, nn[t] / 2.0);
            e = e0nm[t] * (mm[t] * r0n[t] * rninv - nn[t] * r0m[t] * rminv) - offset[t];
          } else {
            e = (e0[t] / 6.0) * (24.0 * powint(r2inv, 6) - 24.0 * pow(r2inv, 3));
          }
          e *= factorLJ;
          evdwl += e;
          if (tally) tallyAtom(pc, i, j, e, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, _del?: PairSep) {
    const t = itype * (this.ntypes + 1) + jtype;
    const nn = this.p.get('nn', itype, jtype), mm = this.p.get('mm', itype, jtype);
    const e0 = this.p.get('e0', itype, jtype), r0 = this.p.get('r0', itype, jtype);
    const r2inv = 1.0 / rsq;
    const r = Math.sqrt(rsq);
    let forcenm: number, phinm: number;
    if (rsq < r0 * r0) {
      const rminv = pow(r2inv, mm / 2.0);
      const rninv = pow(r2inv, nn / 2.0);
      forcenm = this.e0nm[t] * this.nm[t] * (this.r0n[t] / pow(r, nn) - this.r0m[t] / pow(r, mm));
      phinm = this.e0nm[t] * (mm * this.r0n[t] * rninv - nn * this.r0m[t] * rminv) - this.offset[t];
    } else {
      forcenm = (e0 / 6.0) * 72.0 * (4.0 / powint(r, 12) - 2.0 / powint(r, 6));
      phinm = (e0 / 6.0) * (24.0 * powint(r2inv, 6) - 24.0 * powint(r2inv, 3));
    }
    return { eng: factorLJ * phinm, fforce: factorLJ * forcenm * r2inv };
  }
}

/** pair_style nm/cut/coul/cut (pair_nm_cut_coul_cut.cpp): nm/cut with an unscreened Coulomb cutoff per type pair. */
export class PairNMCutCoulCut extends Pair {
  readonly name: string = 'nm/cut/coul/cut';
  virialFdotr = true;
  cutLJGlobal = 0;
  cutCoulGlobal = 0;
  private p!: PairParams;
  private cutLJsq = new Float64Array(0);
  private cutCoulsq = new Float64Array(0);
  private nm = new Float64Array(0);
  private e0nm = new Float64Array(0);
  private r0n = new Float64Array(0);
  private r0m = new Float64Array(0);
  private offset = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length < 1 || args.length > 2) throw new StyleError('Illegal pair_style command');
    this.cutLJGlobal = parseNum(args[0], 'cutoff');
    this.cutCoulGlobal = args.length === 1 ? this.cutLJGlobal : parseNum(args[1], 'coulomb cutoff');
    if (this.p) {
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) {
          if (this.p.isSet(i, j)) {
            this.p.setMixed(i, j, 'cut_lj', this.cutLJGlobal);
            this.p.setMixed(i, j, 'cut_coul', this.cutCoulGlobal);
          }
        }
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['e0', 'r0', 'nn', 'mm', 'cut_lj', 'cut_coul']);
    const n = (ntypes + 1) * (ntypes + 1);
    this.cutLJsq = new Float64Array(n); this.cutCoulsq = new Float64Array(n);
    this.nm = new Float64Array(n); this.e0nm = new Float64Array(n);
    this.r0n = new Float64Array(n); this.r0m = new Float64Array(n); this.offset = new Float64Array(n);
  }

  coeff(args: string[]): void {
    // pair_nm_cut_coul_cut.cpp coeff(): narg 6 to 8 (I J E0 r0 n m [cutoff_lj [cutoff_coul]])
    if (args.length < 6 || args.length > 8) throw new StyleError('Incorrect args for pair coefficients');
    const e0 = parseNum(args[2], 'E0');
    const r0 = parseNum(args[3], 'r0');
    const nn = parseNum(args[4], 'n');
    const mm = parseNum(args[5], 'm');
    let cutLJ = this.cutLJGlobal, cutCoul = this.cutCoulGlobal;
    if (args.length >= 7) cutCoul = cutLJ = parseNum(args[6], 'cutoff');
    if (args.length === 8) cutCoul = parseNum(args[7], 'coulomb cutoff');
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, [e0, r0, nn, mm, cutLJ, cutCoul]));
  }

  initStyle(ctx: StyleContext): void {
    if (!ctx.s || !hasCharge(ctx.s)) throw new StyleError('Pair style nm/cut/coul/cut requires atom attribute q');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError('All pair coeffs are not set');
    const nt = this.ntypes + 1;
    const k = i * nt + j, kr = j * nt + i;
    const e0 = p.get('e0', i, j), r0 = p.get('r0', i, j), nn = p.get('nn', i, j), mm = p.get('mm', i, j);
    const cutLJ = p.get('cut_lj', i, j), cutCoul = p.get('cut_coul', i, j);
    const cut = Math.max(cutLJ, cutCoul);
    this.cutLJsq[k] = cutLJ * cutLJ;
    this.cutCoulsq[k] = cutCoul * cutCoul;
    this.nm[k] = nn * mm;
    this.e0nm[k] = e0 / (nn - mm);
    this.r0n[k] = pow(r0, nn);
    this.r0m[k] = pow(r0, mm);
    this.offset[k] = this.shift && cutLJ > 0.0 ? nmOffset(this.e0nm[k], mm, nn, this.r0n[k], this.r0m[k], cutLJ) : 0.0;
    this.cutLJsq[kr] = this.cutLJsq[k]; this.cutCoulsq[kr] = this.cutCoulsq[k];
    this.nm[kr] = this.nm[k]; this.e0nm[kr] = this.e0nm[k]; this.r0n[kr] = this.r0n[k];
    this.r0m[kr] = this.r0m[k]; this.offset[kr] = this.offset[k];
    p.setMixed(j, i, 'e0', e0); p.setMixed(j, i, 'r0', r0); p.setMixed(j, i, 'nn', nn); p.setMixed(j, i, 'mm', mm);
    return cut;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    let etail = 0, ptail = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = i; j < nt; j++) {
        const k = i * nt + j;
        const t = nmTailPair(count[i], count[j], this.p.get('cut_lj', i, j), this.p.get('r0', i, j), this.p.get('nn', i, j), this.p.get('mm', i, j), this.e0nm[k], this.nm[k]);
        etail += t.etail; ptail += t.ptail;
        if (i !== j) { etail += t.etail; ptail += t.ptail; }
      }
    }
    return { etail, ptail };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, cutLJsq, cutCoulsq, nm, e0nm, r0n, r0m, offset } = this;
    const nn = this.p.p('nn'), mm = this.p.p('mm');
    const qqrd2e = pc.qqrd2e;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qtmp = q[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const factorCoul = sC[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = itype * nt + type[j];
        if (rsq < cutsq[t]) {
          const r2inv = 1.0 / rsq;
          const forcecoul = rsq < cutCoulsq[t] ? qqrd2e * qtmp * q[j] * Math.sqrt(r2inv) : 0.0;
          let forcenm = 0.0, r = 0.0;
          if (rsq < cutLJsq[t]) {
            r = Math.sqrt(rsq);
            forcenm = e0nm[t] * nm[t] * (r0n[t] / pow(r, nn[t]) - r0m[t] / pow(r, mm[t]));
          }
          const fpair = (factorCoul * forcecoul + factorLJ * forcenm) * r2inv;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          let ec = 0.0, e = 0.0;
          if (rsq < cutCoulsq[t]) ec = factorCoul * qqrd2e * qtmp * q[j] * Math.sqrt(r2inv);
          if (rsq < cutLJsq[t]) {
            const rminv = pow(r2inv, mm[t] / 2.0);
            const rninv = pow(r2inv, nn[t] / 2.0);
            e = e0nm[t] * (mm[t] * r0n[t] * rninv - nn[t] * r0m[t] * rminv) - offset[t];
            e *= factorLJ;
          }
          evdwl += e;
          ecoul += ec;
          if (tally) tallyAtom(pc, i, j, e + ec, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number, _del?: PairSep) {
    const t = itype * (this.ntypes + 1) + jtype;
    const nn = this.p.get('nn', itype, jtype), mm = this.p.get('mm', itype, jtype);
    const qqrd2e = this.qqrd2eLast;
    const r2inv = 1.0 / rsq;
    const forcecoul = rsq < this.cutCoulsq[t] ? qqrd2e * qi * qj * Math.sqrt(r2inv) : 0.0;
    let forcenm = 0.0, r = 0.0;
    if (rsq < this.cutLJsq[t]) {
      r = Math.sqrt(rsq);
      forcenm = this.e0nm[t] * this.nm[t] * (this.r0n[t] / pow(r, nn) - this.r0m[t] / pow(r, mm));
    }
    const fforce = (factorCoul * forcecoul + factorLJ * forcenm) * r2inv;
    let eng = 0.0;
    if (rsq < this.cutCoulsq[t]) eng += factorCoul * (qqrd2e * qi * qj * Math.sqrt(r2inv));
    if (rsq < this.cutLJsq[t]) {
      const phinm = this.e0nm[t] * (mm * this.r0n[t] / pow(r, nn) - nn * this.r0m[t] / pow(r, mm)) - this.offset[t];
      eng += factorLJ * phinm;
    }
    return { eng, fforce };
  }

  /** force->qqrd2e as the last compute() saw it (single() takes it from the force field). */
  private qqrd2eLast = 1;

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('e0', i, i))} ${fmtCoeff(this.p.get('r0', i, i))} ${fmtCoeff(this.p.get('nn', i, i))} ${fmtCoeff(this.p.get('mm', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('e0', i, j))} ${fmtCoeff(this.p.get('r0', i, j))} ${fmtCoeff(this.p.get('nn', i, j))} ${fmtCoeff(this.p.get('mm', i, j))} ${fmtCoeff(this.p.get('cut_lj', i, j))}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'cut_coul' || name === 'e0' || name === 'r0' || name === 'nn' || name === 'mm') return this.p.p(name);
    return undefined;
  }
}

/**
 * pair_style nm/cut/coul/long (pair_nm_cut_coul_long.cpp): nm/cut with the Ewald real-space Coulomb term at the global
 * Coulomb cutoff; the erfc comes from the pair_modify table (default 12) or the polynomial (table 0), as coul/long does.
 */
export class PairNMCutCoulLong extends Pair {
  readonly name: string = 'nm/cut/coul/long';
  virialFdotr = true;
  coulLong = true;
  cutLJGlobal = 0;
  cutCoul = 0;
  private p!: PairParams;
  private cutLJsq = new Float64Array(0);
  private nm = new Float64Array(0);
  private e0nm = new Float64Array(0);
  private r0n = new Float64Array(0);
  private r0m = new Float64Array(0);
  private offset = new Float64Array(0);
  private qqrd2eLast = 1;
  private readonly erfcTables = new ErfcTableCache();
  /** pair_modify table N (null for 0): erfc.ts makeErfcTable. */
  private erfcTable(g: number): ErfcTable | null { return this.erfcTables.get(this.table, g, this.cutCoul * this.cutCoul, this.tabinnerRsq); }

  settings(args: string[]): void {
    if (args.length < 1 || args.length > 2) throw new StyleError('Illegal pair_style command');
    this.cutLJGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = args.length === 1 ? this.cutLJGlobal : parseNum(args[1], 'coulomb cutoff');
    if (this.p) for (let i = 1; i <= this.ntypes; i++) for (let j = i; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut_lj', this.cutLJGlobal);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['e0', 'r0', 'nn', 'mm', 'cut_lj']);
    const n = (ntypes + 1) * (ntypes + 1);
    this.cutLJsq = new Float64Array(n);
    this.nm = new Float64Array(n); this.e0nm = new Float64Array(n);
    this.r0n = new Float64Array(n); this.r0m = new Float64Array(n); this.offset = new Float64Array(n);
  }

  coeff(args: string[]): void {
    // pair_nm_cut_coul_long.cpp coeff(): narg 6 or 7 (I J E0 r0 n m [cutoff_lj])
    if (args.length < 6 || args.length > 7) throw new StyleError('Incorrect args for pair coefficients');
    const e0 = parseNum(args[2], 'E0');
    const r0 = parseNum(args[3], 'r0');
    const nn = parseNum(args[4], 'n');
    const mm = parseNum(args[5], 'm');
    const cutLJ = args.length === 7 ? parseNum(args[6], 'cutoff') : this.cutLJGlobal;
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, [e0, r0, nn, mm, cutLJ]));
  }

  initStyle(ctx: StyleContext): void {
    if (!ctx.s || !hasCharge(ctx.s)) throw new StyleError('Pair style nm/cut/coul/long requires atom attribute q');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError('All pair coeffs are not set');
    const nt = this.ntypes + 1;
    const k = i * nt + j, kr = j * nt + i;
    const e0 = p.get('e0', i, j), r0 = p.get('r0', i, j), nn = p.get('nn', i, j), mm = p.get('mm', i, j);
    const cutLJ = p.get('cut_lj', i, j);
    const cut = Math.max(cutLJ, this.cutCoul);
    this.cutLJsq[k] = cutLJ * cutLJ;
    this.nm[k] = nn * mm;
    this.e0nm[k] = e0 / (nn - mm);
    this.r0n[k] = pow(r0, nn);
    this.r0m[k] = pow(r0, mm);
    this.offset[k] = this.shift && cutLJ > 0.0 ? nmOffset(this.e0nm[k], mm, nn, this.r0n[k], this.r0m[k], cutLJ) : 0.0;
    this.cutLJsq[kr] = this.cutLJsq[k];
    this.nm[kr] = this.nm[k]; this.e0nm[kr] = this.e0nm[k]; this.r0n[kr] = this.r0n[k];
    this.r0m[kr] = this.r0m[k]; this.offset[kr] = this.offset[k];
    p.setMixed(j, i, 'e0', e0); p.setMixed(j, i, 'r0', r0); p.setMixed(j, i, 'nn', nn); p.setMixed(j, i, 'mm', mm); p.setMixed(j, i, 'cut_lj', cutLJ);
    return cut;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    let etail = 0, ptail = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = i; j < nt; j++) {
        const k = i * nt + j;
        const t = nmTailPair(count[i], count[j], this.p.get('cut_lj', i, j), this.p.get('r0', i, j), this.p.get('nn', i, j), this.p.get('mm', i, j), this.e0nm[k], this.nm[k]);
        etail += t.etail; ptail += t.ptail;
        if (i !== j) { etail += t.etail; ptail += t.ptail; }
      }
    }
    return { etail, ptail };
  }

  compute(pc: PairCompute): void {
    this.qqrd2eLast = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, cutLJsq, nm, e0nm, r0n, r0m, offset } = this;
    const nn = this.p.p('nn'), mm = this.p.p('mm');
    const cutCoulsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.erfcTable(g);
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qtmp = q[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const factorCoul = sC[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = itype * nt + type[j];
        if (rsq < cutsq[t]) {
          const r2inv = 1.0 / rsq;
          // the Coulomb force and energy of coul/long (erfc table or polynomial, the special-bond correction)
          let forcecoul = 0.0, ec = 0.0;
          if (rsq < cutCoulsq) {
            const c = coulLongPair(rsq, qtmp, q[j], g, pc.qqrd2e, factorCoul, this.table === 0, tab);
            forcecoul = c.f;
            ec = c.e;
          }
          let forcenm = 0.0, r = 0.0;
          if (rsq < cutLJsq[t]) {
            r = Math.sqrt(rsq);
            forcenm = e0nm[t] * nm[t] * (r0n[t] / pow(r, nn[t]) - r0m[t] / pow(r, mm[t]));
          }
          // coulLongPair returns the force over r^2 (forcecoul / rsq), so the LJ part enters with r2inv
          const fpair = forcecoul + factorLJ * forcenm * r2inv;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          let e = 0.0;
          if (rsq < cutLJsq[t]) {
            const rminv = pow(r2inv, mm[t] / 2.0);
            const rninv = pow(r2inv, nn[t] / 2.0);
            e = e0nm[t] * (mm[t] * r0n[t] * rninv - nn[t] * r0m[t] * rminv) - offset[t];
            e *= factorLJ;
          }
          evdwl += e;
          ecoul += ec;
          if (tally) tallyAtom(pc, i, j, e + ec, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number, _del?: PairSep) {
    const t = itype * (this.ntypes + 1) + jtype;
    const nn = this.p.get('nn', itype, jtype), mm = this.p.get('mm', itype, jtype);
    const r2inv = 1.0 / rsq;
    let forcecoul = 0.0, phicoul = 0.0;
    if (rsq < this.cutCoul * this.cutCoul) {
      const c = coulLongPair(rsq, qi, qj, this.gEwald, this.qqrd2eLast, factorCoul, this.table === 0, this.erfcTable(this.gEwald));
      forcecoul = c.f;
      phicoul = c.e;
    }
    let forcenm = 0.0, r = 0.0;
    if (rsq < this.cutLJsq[t]) {
      r = Math.sqrt(rsq);
      forcenm = this.e0nm[t] * this.nm[t] * (this.r0n[t] / pow(r, nn) - this.r0m[t] / pow(r, mm));
    }
    // forcecoul is already over r^2 here (coulLongPair), so the LJ part enters with r2inv as in compute()
    const fforce = forcecoul + factorLJ * forcenm * r2inv;
    let eng = phicoul;
    if (rsq < this.cutLJsq[t]) {
      const phinm = this.e0nm[t] * (mm * this.r0n[t] / pow(r, nn) - nn * this.r0m[t] / pow(r, mm)) - this.offset[t];
      eng += factorLJ * phinm;
    }
    return { eng, fforce };
  }

  // Measured with native LAMMPS (black box, the source has no writedata flag): write_data writes no Pair Coeffs section.
  dataCoeffs(): string[] | null { return null; }
  dataCoeffsIJ(): string[] | null { return null; }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    if (name === 'e0' || name === 'r0' || name === 'nn' || name === 'mm') return this.p.p(name);
    return undefined;
  }
}
