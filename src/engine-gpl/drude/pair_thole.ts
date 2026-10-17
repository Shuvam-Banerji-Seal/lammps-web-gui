/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/pair_thole.cpp and
 * src/DRUDE/pair_thole.h (class PairThole, pair_style thole). Copyright (2003) Sandia Corporation; GPL-2.0
 * (src/engine-gpl/LICENSE). DRUDE package authors: Alain Dequidt, Julien Devemy, Agilio Padua; contributors
 * Kateryna Goloviznina and Zheng Gong.
 *
 * pair_style thole cutoff_thole cutoff (docs.lammps.org/pair_thole.html): Coulomb interaction between induced dipoles
 * of core-Drude pairs, damped with Thole's function (screening a = thole / polar^(1/3) per type pair). A core carries
 * the charge -q of its Drude particle; a Drude particle its own charge; the bonded core-Drude pair itself is skipped.
 * The energy is tallied as Coulomb (ecoul). Special weights of the pair are subtracted from the damping, as LAMMPS.
 */
import { Pair, StyleError, mixDistance, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { DRUDE_TYPE, NOPOL_TYPE, drudeModelOf, type DrudeModel } from './drude_model';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

export class PairThole extends Pair {
  readonly name = 'thole';
  /** Global cutoffs from pair_style (thole_global, cut_global). */
  private tholeGlobal = 0;
  private cutGlobal = 0;
  private polar = new Float64Array(0);
  private thole = new Float64Array(0);
  private ascreen = new Float64Array(0);
  private cutOne = new Float64Array(0);
  private set = new Uint8Array(0);
  private model: DrudeModel | null = null;

  /** keepExcluded: the bonded core-Drude pairs stay in the list, with their special weights (special_bonds.html). */
  override keepExcluded = true;
  /**
   * The central pair forces sum to zero per pair, so the virial is the dot product x . f over all atoms (the force
   * field takes it; sub-styles of pair hybrid need this). Only the energy is tallied here.
   */
  override virialFdotr = true;

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('Illegal pair_style command');
    this.tholeGlobal = parseNum(args[0], 'thole');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    const nt = this.ntypes + 1;
    if (this.set.length === nt * nt) {
      for (let i = 1; i < nt; i++) {
        for (let j = i; j < nt; j++) {
          if (!this.set[i * nt + j]) continue;
          this.thole[i * nt + j] = this.tholeGlobal;
          this.cutOne[i * nt + j] = this.cutGlobal;
        }
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    if (this.set.length !== nt * nt) {
      this.set = new Uint8Array(nt * nt);
      this.polar = new Float64Array(nt * nt);
      this.thole = new Float64Array(nt * nt);
      this.ascreen = new Float64Array(nt * nt);
      this.cutOne = new Float64Array(nt * nt);
    }
  }

  coeff(args: string[]): void {
    if (args.length < 3 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    if (this.set.length === 0) this.allocate(this.ntypes);
    const [ilo, ihi] = bounds(args[0], this.ntypes);
    const [jlo, jhi] = bounds(args[1], this.ntypes);
    const polarOne = parseNum(args[2], 'polar');
    const tholeOne = args.length >= 4 ? parseNum(args[3], 'thole') : this.tholeGlobal;
    const cutOne = args.length === 5 ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    const nt = this.ntypes + 1;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const k = i * nt + j;
        this.polar[k] = polarOne;
        this.thole[k] = tholeOne;
        this.ascreen[k] = tholeOne / libmPow(polarOne, 1 / 3);
        this.cutOne[k] = cutOne;
        this.set[k] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    this.model = drudeModelOf(ctx.s, 'Pair thole');
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const k = i * nt + j, kr = j * nt + i;
    if (!this.set[k]) {
      const kii = i * nt + i, kjj = j * nt + j;
      this.cutOne[k] = mixDistance(this.mix, this.cutOne[kii], this.cutOne[kjj]);
    }
    this.polar[kr] = this.polar[k];
    this.thole[kr] = this.thole[k];
    this.ascreen[kr] = this.ascreen[k];
    this.cutOne[kr] = this.cutOne[k];
    return this.cutOne[k];
  }

  compute(pc: PairCompute): void {
    const model = this.model ?? drudeModelOf(pc.s, 'Pair thole');
    const s = pc.s;
    const partner = model.partnerIndex(s);
    const kindOf = new Uint8Array(s.n);
    for (let i = 0; i < s.n; i++) kindOf[i] = model.kind[s.type[i]];
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sC = pc.specialCoul;
    const owner = pc.nb.owner;
    const list = pc.half!;
    const nb = list.neighbors;
    const qqrd2e = pc.qqrd2e;
    let ecoul = 0;
    for (let ii = list.ilo ?? 0; ii < list.inum; ii++) {
      const i = ii;
      if (kindOf[i] === NOPOL_TYPE) continue;
      const di = partner[i];
      if (di < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[i]} is not defined`);
      const pid = s.id[di];
      const qi = kindOf[i] === DRUDE_TYPE ? q[i] : -q[di];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const kj = kindOf[oj];
        if (kj === NOPOL_TYPE) continue;
        if (s.id[oj] === pid) continue; // the bonded partner of i (LAMMPS: j == di)
        let qj: number;
        if (kj === DRUDE_TYPE) qj = q[j];
        else {
          const dj = partner[oj];
          if (dj < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[oj]} is not defined`);
          qj = -q[dj];
        }
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factorCoul = sC[jj >>> SBBITS];
        const rinv = 1 / Math.sqrt(rsq);
        const r = Math.sqrt(rsq);
        const asr = this.ascreen[t] * r;
        const expAsr = libmExp(-asr);
        const dcoul = qqrd2e * qi * qj * rinv;
        const factorF = 0.5 * (2.0 + expAsr * (-2.0 - asr * (2.0 + asr))) - factorCoul;
        const factorE = 0.5 * (2.0 - expAsr * (2.0 + asr)) - factorCoul;
        const fpair = factorF * dcoul / rsq;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        const e = factorE * dcoul;
        ecoul += e;
        if (pc.eatom) { pc.eatom[i] += 0.5 * e; pc.eatom[j] += 0.5 * e; }
      }
    }
    pc.acc.ecoul += ecoul;
  }
}

/** pair_coeff I J bounds (wildcards as pair_coeff.html; I <= J is enforced by the caller). */
export const bounds = (w: string, n: number): [number, number] => {
  if (w === '*') return [1, n];
  const star = w.indexOf('*');
  if (star < 0) {
    const v = Number(w);
    if (!Number.isInteger(v) || v < 1 || v > n) throw new StyleError(`Invalid atom type range '${w}'`);
    return [v, v];
  }
  const a = w.slice(0, star), b = w.slice(star + 1);
  const lo = a === '' ? 1 : Number(a), hi = b === '' ? n : Number(b);
  if (!(lo >= 1 && hi <= n && lo <= hi)) throw new StyleError(`Invalid atom type range '${w}'`);
  return [lo, hi];
};
