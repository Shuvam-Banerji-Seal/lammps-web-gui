/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-MESO/pair_edpd.cpp and pair_edpd.h
 * (PairEDPD: pair_style edpd, the energy-conserving DPD of Li, Tang, Lei, Caswell and Karniadakis, J. Comput. Phys. 2014).
 * Contributing author (from pair_edpd.cpp): Zhen Li (Clemson University). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style edpd cutoff seed; pair_coeff i j A gamma power cut kappa powerT cutT [power c0 c1 c2 c3] [kappa k0 k1 k2 k3]
 * (docs.lammps.org/pair_edpd.html). Per pair, a random force from one Gaussian deviate of the style's RanMars stream, and
 * a heat-flux exchange from a second stream (randomT) inside cutT. The temperature T, the heat capacity cv and the heat flux
 * (edpd_flux) are the atom's internal props (atom_style edpd); the flux is zeroed at the start of each force evaluation
 * (native force_clear) and the fixes edpd/source add to it after the pair style.
 *
 * Random numbers are drawn in native's pair order (src/engine-gpl/neighbor/nativeOrder.ts), so the runs match native step
 * by step (oracle cases tests/oracle/w93meso_edpd_*.in).
 */
import { Pair, PairParams, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { EDPD_CV, EDPD_FLUX, EDPD_TEMP, isEdpdStyle } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { EPSILON, tallyXYZ } from '../dpd/pair_dpd';
import { pow as libmPow } from '../libm/pow';

/** pair_edpd.cpp: "static constexpr double EPSILON = 1.0e-10;" and kboltz = 1.0 (the lj units of the eDPD examples). */
const KBOLTZ = 1.0;
const NAMES = ['a0', 'gamma', 'power', 'cut', 'kappa', 'powerT', 'cutT'];

export class PairEDPD extends Pair {
  readonly name = 'edpd';
  virialFdotr = true;
  private cutGlobal = 0;
  private seed = 0;
  private rng: RanMars | null = null;
  private rngT: RanMars | null = null;
  private p: PairParams | null = null;
  /** power / kappa keywords: four polynomial coefficients per type pair (pair_edpd.cpp sc and kc), flags of the last pair_coeff. */
  private sc = new Float64Array(0);
  private kc = new Float64Array(0);
  private powerFlag = false;
  private kappaFlag = false;
  private nt = 0;
  private readonly order = new NativePairOrder();

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.seed = parseNum(args[1], 'seed');
    if (!(this.seed > 0) || !Number.isInteger(this.seed)) throw new StyleError('Invalid random number seed');
    // pair_edpd.cpp settings(): RanMars(seed + me) for the forces and RanMars(2 seed + me) for the heat exchange
    this.rng = new RanMars(this.seed % 900000000);
    this.rngT = new RanMars((2 * this.seed) % 900000000);
    // pair_edpd.cpp settings(): the cutoff of set pairs (i < j) is taken from the pair_style cutoff again
    if (this.p) {
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i + 1; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut', this.cutGlobal);
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, NAMES);
  }

  coeff(args: string[]): void {
    if (args.length < 9) throw new StyleError('Incorrect args for pair edpd coefficients');
    if (!this.p) throw new StyleError('pair_coeff before the box is defined');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const v = args.slice(2, 9).map((w, k) => parseNum(w, ['a0', 'gamma', 'power', 'cut', 'kappa', 'powerT', 'cutT'][k]));
    const [a0, gamma, power, cut, kappa, powerT, cutT] = v;
    // pair_edpd.cpp coeff(): the power and kappa keywords take four values each; a keyword sets its flag for this call only
    this.powerFlag = false;
    this.kappaFlag = false;
    const scOne = [0, 0, 0, 0], kcOne = [0, 0, 0, 0];
    for (let iarg = 9; iarg < args.length;) {
      const kw = args[iarg];
      if (kw === 'power' || kw === 'kappa') {
        if (iarg + 5 > args.length) throw new StyleError('Illegal pair edpd coefficients');
        const one = kw === 'power' ? scOne : kcOne;
        for (let k = 0; k < 4; k++) one[k] = parseNum(args[iarg + k + 1], kw);
        if (kw === 'power') this.powerFlag = true; else this.kappaFlag = true;
        iarg += 5;
      } else throw new StyleError('Illegal pair edpd coefficients');
    }
    const nt = this.ntypes + 1;
    if (this.powerFlag && this.sc.length === 0) this.sc = new Float64Array(nt * nt * 4);
    if (this.kappaFlag && this.kc.length === 0) this.kc = new Float64Array(nt * nt * 4);
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.p.set(i, j, [a0, gamma, power, cut, kappa, powerT, cutT]);
        const idx = this.p.idx(i, j);
        if (this.powerFlag) for (let k = 0; k < 4; k++) this.sc[4 * idx + k] = scOne[k];
        if (this.kappaFlag) for (let k = 0; k < 4; k++) this.kc[4 * idx + k] = kcOne[k];
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    // pair_edpd.cpp init_style(): "Pair edpd requires ghost atoms store velocity"
    if (ctx.ghostVelocity !== true) throw new StyleError('Pair edpd requires ghost atoms store velocity\n(use comm_modify vel yes)');
    if (ctx.newtonPair === false) throw new StyleError('pair_style edpd with newton pair off is not supported by the browser engine');
    if (!ctx.s || !isEdpdStyle(ctx.s.atomStyle)) throw new StyleError('Pair style edpd requires atom style edpd');
    this.nt = this.ntypes + 1;
  }

  initOne(i: number, j: number): number {
    const p = this.p!;
    if (!p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    return p.get('cut', i, j);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const rng = this.rng, rngT = this.rngT, p = this.p;
    if (!rng || !rngT || !p) throw new StyleError('pair edpd used before pair_style');
    const T = s.custom.get(EDPD_TEMP)!.data, Q = s.custom.get(EDPD_FLUX)!.data, cv = s.custom.get(EDPD_CV)!.data;
    // native force_clear(): the heat flux of every atom starts at zero for this force evaluation
    Q.fill(0, 0, s.n);
    const { x, f, type } = pc;
    const vel = s.v;
    const owner = pc.nb.owner;
    const nt = this.nt;
    const { cutsq } = this;
    const a0 = p.p('a0'), gam = p.p('gamma'), pw = p.p('power'), cutA = p.p('cut');
    const kap = p.p('kappa'), pwT = p.p('powerT'), cutT = p.p('cutT');
    const sLJ = pc.specialLJ;
    const dtinvsqrt = 1.0 / Math.sqrt(s.dt);
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    const tally = pc.eatom !== null || pc.vatom !== null;
    const massByType = s.massByType;
    const powerFlag = this.powerFlag, kappaFlag = this.kappaFlag;
    const sc = this.sc, kc = this.kc;
    let evdwl = 0;
    const Tp = [0, 0, 0, 0];
    for (let q = 0; q < pairs.length; q += 2) {
      const i = pairs[q];
      const jj = pairs[q + 1];
      const j = jj & NEIGHMASK;
      const factorDpd = sLJ[jj >>> SBBITS];
      const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const t = type[i] * nt + type[j];
      if (rsq >= cutsq[t]) continue;
      const r = Math.sqrt(rsq);
      if (r < EPSILON) continue;
      const rinv = 1.0 / r;
      const oj = owner[j];
      const delvx = vel[3 * i] - vel[3 * oj], delvy = vel[3 * i + 1] - vel[3 * oj + 1], delvz = vel[3 * i + 2] - vel[3 * oj + 2];
      const dot = delx * delvx + dely * delvy + delz * delvz;
      const vijeij = dot * rinv;
      const randnum = rng.gaussian();
      const Ti = T[i], Tj = T[oj];
      const Tij = 0.5 * (Ti + Tj);
      Tp[0] = Tij - 1.0;
      Tp[1] = Tp[0] * Tp[0];
      Tp[2] = Tp[0] * Tp[1];
      Tp[3] = Tp[0] * Tp[2];
      let powerD = pw[t];
      if (powerFlag) {
        let factor = 1.0;
        for (let k = 0; k < 4; k++) factor += sc[4 * t + k] * Tp[k];
        powerD *= factor;
      }
      powerD = Math.max(0.01, powerD);
      let wc = 1.0 - r / cutA[t];
      wc = Math.max(0.0, Math.min(1.0, wc));
      const wr = libmPow(wc, 0.5 * powerD);
      const GammaIJ = gam[t];
      let SigmaIJ = 4.0 * GammaIJ * KBOLTZ * Ti * Tj / (Ti + Tj);
      SigmaIJ = Math.sqrt(SigmaIJ);
      let fpair = a0[t] * Tij * wc;
      fpair -= GammaIJ * wr * wr * dot * rinv;
      fpair += SigmaIJ * wr * randnum * dtinvsqrt;
      fpair *= factorDpd * rinv;
      const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      let dQc = 0.0, dQd = 0.0, dQr = 0.0;
      if (r < cutT[t]) {
        let wrT = 1.0 - r / cutT[t];
        wrT = Math.max(0.0, Math.min(1.0, wrT));
        wrT = libmPow(wrT, 0.5 * pwT[t]);
        let randnumT = rngT.gaussian();
        randnumT = Math.max(-5.0, Math.min(randnumT, 5.0));
        let kappaT = kap[t];
        if (kappaFlag) {
          let factor = 1.0;
          for (let k = 0; k < 4; k++) factor += kc[4 * t + k] * Tp[k];
          kappaT *= factor;
        }
        const kij = cv[i] * cv[oj] * kappaT * Tij * Tij;
        const alphaij = Math.sqrt(2.0 * KBOLTZ * kij);
        dQc = kij * wrT * wrT * (Tj - Ti) / (Ti * Tj);
        dQd = wr * wr * (GammaIJ * vijeij * vijeij - SigmaIJ * SigmaIJ / massByType[type[i]]) - SigmaIJ * wr * vijeij * randnum;
        dQd /= (cv[i] + cv[oj]);
        dQr = alphaij * wrT * dtinvsqrt * randnumT;
        Q[i] += (dQc + dQd + dQr);
      }
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      Q[oj] -= (dQc - dQd + dQr);
      let e = 0.5 * a0[t] * Tij * cutA[t] * wc * wc;
      e *= factorDpd;
      evdwl += e;
      if (tally) tallyXYZ(pc, i, j, e, fx, fy, fz, delx, dely, delz);
    }
    pc.acc.evdwl += evdwl;
  }
}
