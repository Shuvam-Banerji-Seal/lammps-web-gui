/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-BASIC/pair_dpd_coul_slater_long.cpp and
 * pair_dpd_coul_slater_long.h (PairDPDCoulSlaterLong: pair_style dpd/coul/slater/long). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE). Contributing author (from the source): Eddy Barraud (IFPEN/Sorbonne).
 *
 * One pair style with two parts: the DPD thermostat of pair_style dpd (conservative a0 wd, drag, random sigma wd theta /
 * sqrt(dt), with the per-pair RanMars stream of pair_dpd.ts) inside the DPD cutoff, and the real-space Slater-Ewald
 * Coulomb of coul/slater/long (slater.ts slaterCoulPair) inside the Slater cutoff.
 *
 * single() (PairDPDCoulSlaterLong::single, src/DPD-BASIC/pair_dpd_coul_slater_long.cpp at b332f28): the DPD conservative
 * force and energy (no random or dissipative part) plus the Slater Coulomb with the plain (1 - factor_coul) prefactor
 * correction native single() uses, which is not the compute's Slater form. pair_coeff I J a0 gamma [yes|no] [cut]:
 * "yes" gives the pair the Slater Coulomb term out to the pair_style cutoff, "no" (default) gives it none.
 *
 * The random deviate is drawn only inside the DPD cutoff, after the r > EPSILON test, as native does, in the native
 * neighbor-list order (nativeOrder.ts), so the random forces match native step by step.
 */
import { Pair, PairParams, StyleError, type AdaptAccess, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { fmtCoeff, parseInt_, parseNum, logical } from '../../engine/force/util';
import { RanMars } from '../random_mars';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { EPSILON, tallyXYZ } from '../dpd/pair_dpd';
import { slaterCoulPair, EWALD_P, A1, A2, A3, A4, A5 } from './pair_coul_slater_long';
import { EWALD_F } from '../../engine/force/erfc';
import { exp as libmExp } from '../libm/exp';

/** utils::logical accepts these words for the per-pair "do_slater" flag. */
const logicalWord = (w: string | undefined): boolean => {
  if (w === undefined) throw new StyleError('Illegal pair_coeff command (expected yes or no)');
  return logical(w);
};

export class PairDPDCoulSlaterLong extends Pair {
  readonly name: string = 'dpd/coul/slater/long';
  virialFdotr = true;
  coulLong = true;
  cutGlobal = 0;
  temperature = 0;
  seed = 0;
  lamda = 0;
  cutCoul = 0;
  private rng: RanMars | null = null;
  private p!: PairParams;
  /** Per type pair: the DPD cutoff (cut_dpd), its square, and the Slater cutoff squared (0 without the Slater term). */
  private cutDpd = new Float64Array(0);
  private cutDpdsq = new Float64Array(0);
  private cutSlatersq = new Float64Array(0);
  private sigma = new Float64Array(0);
  private boltz = 1;
  private nt = 0;
  /** force->qqrd2e of the last compute (single() reads it, as native does from force). */
  protected qqrd2eLast = 1;
  private readonly order = new NativePairOrder();

  settings(args: string[]): void {
    // pair_dpd_coul_slater_long.cpp settings(): if (narg != 5) error->all(FLERR,"Illegal pair_style command");
    if (args.length !== 5) throw new StyleError('Illegal pair_style command (dpd/coul/slater/long T cutoff seed lamda cutoff_coul)');
    this.temperature = parseNum(args[0], 'temperature');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    this.seed = parseInt_(args[2], 'seed');
    this.lamda = parseNum(args[3], 'lamda');
    this.cutCoul = parseNum(args[4], 'cutoff_coul');
    if (!(this.seed > 0)) throw new StyleError(`Invalid random seed ${args[2]} for pair_style dpd/coul/slater/long command`);
    this.rng = new RanMars(this.seed);
    // settings() re-sets cut_dpd of the pairs already set to MAX(cut_global, cut_coul)
    if (this.p) {
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) {
          if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut', Math.max(this.cutGlobal, this.cutCoul));
        }
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['a0', 'gamma', 'cut', 'slater']);
  }

  coeff(args: string[]): void {
    // pair_dpd_coul_slater_long.cpp coeff(): narg 4..6 (I J a0 gamma [do_slater] [cut_dpd])
    if (args.length < 4 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    const a0 = parseNum(args[2], 'a0');
    const gamma = parseNum(args[3], 'gamma');
    let cutOne = this.cutGlobal;
    let cutTwo = 0;
    if (args.length > 4 && logicalWord(args[4])) cutTwo = this.cutCoul;
    if (args.length > 5) cutOne = parseNum(args[5], 'cutoff');
    this.p.setRange(args[0], args[1], [a0, gamma, cutOne, cutTwo]);
  }

  initStyle(ctx: StyleContext): void {
    // init_style(): "Pair dpd requires ghost atoms store velocity"
    if (ctx.ghostVelocity !== true) throw new StyleError('Pair dpd requires ghost atoms store velocity\n(use comm_modify vel yes)');
    // init_style(): "Pair dpd needs newton pair on for momentum conservation" is a warning natively; newton off is refused here
    if (ctx.newtonPair === false) throw new StyleError('pair_style dpd/coul/slater/long with newton pair off is not supported by the browser engine');
    this.boltz = ctx.s ? ctx.s.units.boltz : 1;
    this.nt = this.ntypes + 1;
    this.sigma = new Float64Array(this.nt * this.nt);
    this.cutDpd = new Float64Array(this.nt * this.nt);
    this.cutDpdsq = new Float64Array(this.nt * this.nt);
    this.cutSlatersq = new Float64Array(this.nt * this.nt);
  }

  initOne(i: number, j: number): number {
    // init_one(): "All pair coeffs are not set"
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.nt;
    const cutDpd = this.p.get('cut', i, j);
    const cutSlater2 = this.p.get('slater', i, j) * this.p.get('slater', i, j);
    const sig = Math.sqrt(2.0 * this.boltz * this.temperature * this.p.get('gamma', i, j));
    for (const [a, b] of [[i, j], [j, i]] as const) {
      this.sigma[a * nt + b] = sig;
      this.cutDpd[a * nt + b] = cutDpd;
      this.cutDpdsq[a * nt + b] = cutDpd * cutDpd;
      this.cutSlatersq[a * nt + b] = cutSlater2;
    }
    // return MAX(cut_dpd, sqrt(cut_slatersq)) for the pair
    return Math.max(cutDpd, Math.sqrt(cutSlater2));
  }

  /**
   * PairDPDCoulSlaterLong::single (src/DPD-BASIC/pair_dpd_coul_slater_long.cpp at b332f28): fforce and the returned energy
   * of the DPD conservative part (factor_dpd = factorLJ) inside cut_dpd and r > EPSILON, plus the Slater Coulomb when
   * q_i q_j != 0 and rsq < cut_slatersq. The Coulomb's (1 - factor_coul) correction is the plain prefactor, as native
   * single() writes it. No cutoff test beyond these.
   */
  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorDpd: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    const t = itype * this.nt + jtype;
    const a0 = this.p.p('a0');
    let fforce = 0.0;
    let energy = 0.0;
    const r = Math.sqrt(rsq);
    if (rsq < this.cutDpdsq[t] && r > EPSILON) {
      const rinv = 1.0 / r;
      const wd = 1.0 - r / this.cutDpd[t];
      fforce += a0[t] * wd * factorDpd * rinv;
      const phi = 0.5 * a0[t] * this.cutDpd[t] * wd * wd;
      energy += factorDpd * phi;
    }
    if (qi * qj !== 0.0 && rsq < this.cutSlatersq[t]) {
      const r2inv = 1.0 / rsq;
      const g = this.gEwald;
      const qqrd2e = this.qqrd2eLast;
      const grij = g * r;
      const expm2 = libmExp(-grij * grij);
      const tt = 1.0 / (1.0 + EWALD_P * grij);
      const erfc = tt * (A1 + tt * (A2 + tt * (A3 + tt * (A4 + tt * A5)))) * expm2;
      const slaterTerm = libmExp(-2 * r / this.lamda) * (1 + (2 * r / this.lamda * (1 + r / this.lamda)));
      const prefactor = qqrd2e * qi * qj / r;
      let forcecoul = prefactor * (erfc + EWALD_F * grij * expm2 - slaterTerm);
      if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
      fforce += forcecoul * r2inv;
      let phicoul = prefactor * (erfc - (1 + r / this.lamda) * libmExp(-2 * r / this.lamda));
      if (factorCoul < 1.0) phicoul -= (1.0 - factorCoul) * prefactor;
      energy += phicoul;
    }
    return { eng: energy, fforce };
  }

  compute(pc: PairCompute): void {
    const rng = this.rng;
    if (!rng) throw new StyleError('pair dpd/coul/slater/long used before pair_style');
    const pairs = this.order.pairs(pc.s, pc.nb, pc.geom, pc.half);
    const { x, f, type, q } = pc;
    const s = pc.s;
    const vel = s.v;
    const owner = pc.nb.owner;
    const nt = this.nt;
    const cutsq = this.cutsq;
    const a0 = this.p.p('a0'), gamma = this.p.p('gamma');
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const sqrtS = [Math.sqrt(sLJ[0]), Math.sqrt(sLJ[1]), Math.sqrt(sLJ[2]), Math.sqrt(sLJ[3])];
    const dtinvsqrt = 1.0 / Math.sqrt(s.dt);
    const tally = pc.eatom !== null || pc.vatom !== null;
    const g = this.gEwald;
    this.qqrd2eLast = pc.qqrd2e;
    let evdwl = 0, ecoul = 0;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const jj = pairs[p + 1];
      const j = jj & NEIGHMASK;
      const m = jj >>> SBBITS;
      const factorDpd = sLJ[m];
      const factorSqrt = sqrtS[m];
      const factorCoul = sC[m];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const t = type[i] * nt + type[j];
      if (rsq >= cutsq[t]) continue;
      const r = Math.sqrt(rsq);
      let evdwlPair = 0, ecoulPair = 0;
      let forcedpd = 0;
      if (rsq < this.cutDpdsq[t] && r > EPSILON) {
        const rinv = 1.0 / r;
        const oj = owner[j];
        const vxi = vel[3 * i], vyi = vel[3 * i + 1], vzi = vel[3 * i + 2];
        const delvx = vxi - vel[3 * oj], delvy = vyi - vel[3 * oj + 1], delvz = vzi - vel[3 * oj + 2];
        const dot = delx * delvx + dely * delvy + delz * delvz;
        const wd = 1.0 - r / this.cutDpd[t];
        const randnum = rng.gaussian();
        forcedpd = a0[t] * wd;
        forcedpd -= gamma[t] * wd * wd * dot * rinv;
        forcedpd *= factorDpd;
        forcedpd += factorSqrt * this.sigma[t] * wd * randnum * dtinvsqrt;
        forcedpd *= rinv;
        evdwlPair = 0.5 * a0[t] * this.cutDpd[t] * wd * wd;
        evdwlPair *= factorDpd;
      }
      let forcecoul = 0;
      if (rsq < this.cutSlatersq[t]) {
        const c = slaterCoulPair(rsq, q[i], q[j], g, pc.qqrd2e, this.lamda, factorCoul);
        forcecoul = c.f;
        ecoulPair = c.e;
      }
      const fpair = forcedpd + forcecoul;
      const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      evdwl += evdwlPair;
      ecoul += ecoulPair;
      if (tally) tallyXYZ(pc, i, j, evdwlPair + ecoulPair, fx, fy, fz, delx, dely, delz);
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  /** pair_dpd_coul_slater_long.cpp write_data(): "i a0 gamma yes|no cut_dpd" per type (writedata = 1). */
  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const slater = this.p.get('slater', i, j) === 0 ? 'no' : 'yes';
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('a0', i, j))} ${fmtCoeff(this.p.get('gamma', i, j))} ${slater} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }

  /** fix adapt's lamda (extract() dim 0): compute reads it at every step, as native does. */
  override adaptParam(name: string): AdaptAccess | null {
    if (name !== 'lamda') return null;
    return { dim: 0, get: () => this.lamda, set: (v) => { this.lamda = v; } };
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    if (name === 'lamda') return this.lamda;
    return undefined;
  }
}
