/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/pair_coul_tt.cpp and
 * src/DRUDE/pair_coul_tt.h (class PairCoulTT, pair_style coul/tt). Copyright (2003) Sandia Corporation; GPL-2.0
 * (src/engine-gpl/LICENSE). DRUDE package authors: Alain Dequidt, Julien Devemy, Agilio Padua; contributors
 * Kateryna Goloviznina and Zheng Gong.
 *
 * pair_style coul/tt N cutoff (docs.lammps.org/pair_coul_tt.html): charge-dipole damping of Coulomb interactions with
 * the Tang-Toennies function f_n(br) = 1 + sum_{k=1..n} (br)^k / k!, multiplied by C exp(-br):
 * E = -C exp(-b r) f_n(b r) q_i q_j / r (pair_coeff I J b C [n] [cutoff]). Cores carry their Drude charge: the
 * core-Drude terms of a molecule are skipped and a core's charge is combined with its Drude particle's, as LAMMPS.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { inumeric, parseNum } from '../../engine/force/util';
import { CORE_TYPE, DRUDE_TYPE, NOPOL_TYPE, drudeModelOf, type DrudeModel } from './drude_model';
import { exp as libmExp } from '../libm/exp';

export class PairCoulTT extends Pair {
  readonly name = 'coul/tt';
  private nGlobal = 0;
  private cutGlobal = 0;
  private bArr = new Float64Array(0);
  private cArr = new Float64Array(0);
  private nArr = new Int32Array(0);
  private cutOne = new Float64Array(0);
  private set = new Uint8Array(0);
  private model: DrudeModel | null = null;

  /** keepExcluded: the core-Drude pairs keep their special entries in the list (kept for the coul/tt terms). */
  override keepExcluded = true;
  /** Central pair forces: the force field takes the virial as x . f (see pair_thole.ts). */
  override virialFdotr = true;

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('Illegal pair_style command');
    this.nGlobal = Math.trunc(parseNum(args[0], 'n'));
    this.cutGlobal = parseNum(args[1], 'cutoff');
    const nt = this.ntypes + 1;
    if (this.set.length === nt * nt) {
      for (let i = 1; i < nt; i++) {
        for (let j = i; j < nt; j++) {
          if (!this.set[i * nt + j]) continue;
          this.nArr[i * nt + j] = this.nGlobal;
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
      this.bArr = new Float64Array(nt * nt);
      this.cArr = new Float64Array(nt * nt);
      this.nArr = new Int32Array(nt * nt);
      this.cutOne = new Float64Array(nt * nt);
    }
  }

  coeff(args: string[]): void {
    if (args.length < 3 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    if (this.set.length === 0) this.allocate(this.ntypes);
    const [ilo, ihi] = bounds(args[0], this.ntypes);
    const [jlo, jhi] = bounds(args[1], this.ntypes);
    const bOne = parseNum(args[2], 'b');
    const cOne = parseNum(args[3], 'C');
    let nOne = this.nGlobal;
    let cutOne = this.cutGlobal;
    // pair_coul_tt.cpp coeff(): n is read by utils::inumeric
    if (args.length >= 5) nOne = inumeric(args[4]);
    if (args.length === 6) cutOne = parseNum(args[5], 'cutoff');
    if (nOne > this.nGlobal) throw new StyleError('Incorrect coefficients for pair style coul/tt: n should not be larger than global setting');
    const nt = this.ntypes + 1;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const k = i * nt + j;
        this.bArr[k] = bOne;
        this.cArr[k] = cOne;
        this.nArr[k] = nOne;
        this.cutOne[k] = cutOne;
        this.set[k] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    this.model = drudeModelOf(ctx.s, 'Pair coul/tt');
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const k = i * nt + j, kr = j * nt + i;
    if (!this.set[k]) {
      this.cutOne[k] = Math.sqrt(this.cutOne[i * nt + i] * this.cutOne[j * nt + j]);
    }
    this.bArr[kr] = this.bArr[k];
    this.cArr[kr] = this.cArr[k];
    this.nArr[kr] = this.nArr[k];
    this.cutOne[kr] = this.cutOne[k];
    return this.cutOne[k];
  }

  compute(pc: PairCompute): void {
    const model = this.model ?? drudeModelOf(pc.s, 'Pair coul/tt');
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
    for (let i = 0; i < list.inum; i++) {
      const ki = kindOf[i];
      const idI = s.id[i];
      let qi = q[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const di = ki === CORE_TYPE ? partner[i] : -1;
      if (ki === CORE_TYPE && di < 0) throw new StyleError(`Drude atom for core atom ID ${idI} is not defined`);
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const kj = kindOf[oj];
        if (ki === kj && kj !== CORE_TYPE) continue;
        let qj = q[j];
        const factorCoul = sC[jj >>> SBBITS];
        if (ki === CORE_TYPE) {
          if (s.id[oj] === s.id[di]) continue; // the Drude particle of i (LAMMPS: di == j)
          if (kj === DRUDE_TYPE) qi = q[i] + q[di];
          else if (kj === NOPOL_TYPE) qi = -q[di];
        }
        let dj = -1;
        if (kj === CORE_TYPE) {
          dj = partner[oj];
          if (dj < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[oj]} is not defined`);
          if (s.id[dj] === idI) continue; // i is the Drude particle of j (LAMMPS: dj == i)
          if (ki === DRUDE_TYPE) qj = q[j] + q[dj];
          else if (ki === NOPOL_TYPE) qj = -q[dj];
        }
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const rinv = 1 / Math.sqrt(rsq);
        const r2inv = 1 / rsq;
        const r = Math.sqrt(rsq);
        const b = this.bArr[t], c = this.cArr[t], n = this.nArr[t];
        const beta = c * libmExp(-b * r);
        const betaprime = -b * beta;
        let gamma = 1.0 + b * r;
        let gammaprime = b;
        let gammatmp = 1.0;
        for (let kk = 2; kk <= n; kk++) {
          gammatmp *= (b * r) / kk;
          gamma += gammatmp * b * r;
          gammaprime += gammatmp * b * kk;
        }
        let dcoul: number;
        if (ki === CORE_TYPE && kj === CORE_TYPE) {
          const dk = partner[i];
          dcoul = qqrd2e * (-(q[i] + q[dk]) * q[dj] - q[dk] * (q[j] + q[dj])) * rinv;
        } else {
          dcoul = qqrd2e * qi * qj * rinv;
        }
        const factorF = (-beta * gamma + r * betaprime * gamma + r * beta * gammaprime) * factorCoul;
        const factorE = -beta * gamma * factorCoul;
        const fpair = factorF * dcoul * r2inv;
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
const bounds = (w: string, n: number): [number, number] => {
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
