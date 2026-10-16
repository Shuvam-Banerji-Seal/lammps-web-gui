/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EXTRA-PAIR/pair_coul_slater_long.cpp and
 * pair_coul_slater_long.h (PairCoulSlaterLong: pair_style coul/slater/long). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Contributing author (from the source): Evangelos Voyiatzis (Royal DSM).
 *
 * Real-space Coulomb of a Gaussian-smeared (Ewald) charge against a Slater-type charge distribution with decay length
 * lamda (docs.lammps.org/pair_coul.html, "coul/slater/long"): the Ewald erfc term minus the Slater term
 * S(r) = exp(-2r/lamda) (1 + 2r/lamda (1 + r/lamda)), with the Ewald erfc taken from the polynomial fit of
 * ewald_const.h (EWALD_P, A1..A5) as native does (no tables).
 */
import { Pair, PairParams, StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum } from '../../engine/force/util';
import { EWALD_F } from '../../engine/force/erfc';
import { exp as libmExp } from '../libm/exp';

/** ewald_const.h: EWALD_P (the polynomial erfc of the direct branch uses the same A1..A5 as erfcPoly). */
const EWALD_P = 0.3275911;
const A1 = 0.254829592, A2 = -0.284496736, A3 = 1.421413741, A4 = -1.453152027, A5 = 1.061405429;

/**
 * Energy and force-over-r^2 of one pair (coul/slater/long compute() and single()): the prefactor C q_i q_j / r with the
 * Ewald real-space erfc and the Slater correction; the special weight fc scales the excluded part (1 - fc) as in the
 * source (forcecoul -= (1-fc) prefactor (1 - S), energy -= (1-fc) prefactor (1 - (1 + r/lamda) exp(-2r/lamda))).
 */
export const slaterCoulPair = (rsq: number, qi: number, qj: number, g: number, qqrd2e: number, lamda: number, fc: number): { e: number; f: number } => {
  const r2inv = 1.0 / rsq;
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const expm2 = libmExp(-grij * grij);
  const t = 1.0 / (1.0 + EWALD_P * grij);
  const erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
  const slaterTerm = libmExp(-2 * r / lamda) * (1 + (2 * r / lamda * (1 + r / lamda)));
  const prefactor = qqrd2e * qi * qj / r;
  let forcecoul = prefactor * (erfc + EWALD_F * grij * expm2 - slaterTerm);
  if (fc < 1.0) forcecoul -= (1.0 - fc) * prefactor * (1 - slaterTerm);
  let e = prefactor * (erfc - (1 + r / lamda) * libmExp(-2 * r / lamda));
  if (fc < 1.0) e -= (1.0 - fc) * prefactor * (1.0 - (1 + r / lamda) * libmExp(-2 * r / lamda));
  return { e, f: forcecoul * r2inv };
};

export class PairCoulSlaterLong extends Pair {
  readonly name: string = 'coul/slater/long';
  virialFdotr = true;
  coulLong = true;
  /** Slater decay length and Coulomb cutoff (pair_style coul/slater/long lamda cutoff). */
  lamda = 0;
  cutCoul = 0;
  private set!: PairParams;

  settings(args: string[]): void {
    // pair_coul_slater_long.cpp settings(): if (narg != 2) error->all(FLERR,"Illegal pair_style command");
    if (args.length !== 2) throw new StyleError('Illegal pair_style command (coul/slater/long lamda cutoff)');
    this.lamda = parseNum(args[0], 'lamda');
    this.cutCoul = parseNum(args[1], 'cutoff');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.set = new PairParams(ntypes, ['scale']);
  }

  coeff(args: string[]): void {
    // pair_coul_slater_long.cpp coeff(): "pair_coeff I J" with no coefficients; scale = 1 for the pairs set
    if (args.length !== 2) throw new StyleError('Incorrect args for pair coefficients');
    this.set.setRange(args[0], args[1], [1]);
  }

  initOne(i: number, j: number): number {
    // init_one(): the pair takes the Coulomb cutoff (scale of unset pairs mixes to 1)
    if (!this.set.isSet(i, j) && !(this.set.isSet(i, i) && this.set.isSet(j, j))) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    return this.cutCoul;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, q } = pc;
    const cutsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = q[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq) continue;
        const r = slaterCoulPair(rsq, qi, q[j], g, pc.qqrd2e, this.lamda, sC[jj >>> SBBITS]);
        fxi += dx * r.f; fyi += dy * r.f; fzi += dz * r.f;
        f[3 * j] -= dx * r.f; f[3 * j + 1] -= dy * r.f; f[3 * j + 2] -= dz * r.f;
        ecoul += r.e;
        if (tally) tallyAtom(pc, i, j, r.e, r.f, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.ecoul += ecoul;
  }

  /** pair_coul_slater_long.cpp extract(): cut_coul, lamda. */
  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    if (name === 'lamda') return this.lamda;
    return undefined;
  }
}
