/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-BASIC/pair_dpd_ext.cpp and
 * pair_dpd_ext.h (PairDPDExt: pair_style dpd/ext) and src/DPD-BASIC/pair_dpd_ext_tstat.cpp (PairDPDExtTstat:
 * pair_style dpd/ext/tstat). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing authors (from
 * pair_dpd_ext.cpp): Martin Svoboda (ICPF, UJEP), Martin Lisal (ICPF, UJEP), based on pair style dpd by Kurt Smith (U Pittsburgh).
 *
 * Anisotropic DPD (docs.lammps.org/pair_dpd_ext.html): the parallel part uses the weight wd^ws (ws = the "ws"
 * exponent, wdPar = wd^ws), the perpendicular part projects the relative velocity and the random vector with
 * P = 1 - rhat rhat and uses wd^wsT. Four Gaussian deviates are drawn per pair (one parallel, three for the
 * perpendicular vector), in that order, from the same RanMars stream as pair dpd.
 *
 * The per-pair draw order is native LAMMPS's neighbor-list order (see pair_dpd.ts and nativeOrder.ts).
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { fmtCoeff, parseInt_, parseNum } from '../../engine/force/util';
import { RanMars } from '../random_mars';
import { EPSILON, tallyXYZ } from './pair_dpd';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { pow as libmPow } from '../libm/pow';

export class PairDPDExt extends Pair {
  readonly name: string;
  virialFdotr = true;
  /** true for pair_style dpd/ext/tstat (no conservative term; the temperature is the pair_style's). */
  readonly tstat: boolean;
  temperature = 0;
  tStart = 0;
  tStop = 0;
  cutGlobal = 0;
  seed = 0;
  p!: PairParams;
  private rng: RanMars | null = null;
  private boltz = 1;
  private sigma = new Float64Array(0);
  /** the run's begin and end steps (StyleContext.runSteps), read for the dpd/ext/tstat ramp */
  private runSteps: (() => { begin: number; end: number }) | null = null;
  private sigmaT = new Float64Array(0);
  private nt = 0;
  private readonly order = new NativePairOrder();
  private logFn: ((text: string) => void) | null = null;
  private triWarned = false;

  constructor(tstat = false) {
    super();
    this.tstat = tstat;
    this.name = tstat ? 'dpd/ext/tstat' : 'dpd/ext';
  }

  settings(args: string[]): void {
    if (this.tstat) {
      if (args.length !== 4) throw new StyleError('Illegal pair_style command (dpd/ext/tstat t_start t_stop cutoff seed)');
      this.tStart = parseNum(args[0], 't_start');
      this.tStop = parseNum(args[1], 't_stop');
      this.temperature = this.tStart;
      this.cutGlobal = parseNum(args[2], 'cutoff');
      this.seed = parseInt_(args[3], 'seed');
    } else {
      if (args.length !== 3) throw new StyleError('Illegal pair_style command (dpd/ext T cutoff seed)');
      this.temperature = parseNum(args[0], 'temperature');
      this.cutGlobal = parseNum(args[1], 'cutoff');
      this.seed = parseInt_(args[2], 'seed');
    }
    // pair_dpd_ext.cpp settings(): "if (seed <= 0) error->all(FLERR,"Illegal pair_style command");"
    if (!(this.seed > 0)) throw new StyleError('Illegal pair_style command');
    this.rng = new RanMars(this.seed);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, this.tstat ? ['gamma', 'gammaT', 'ws', 'wsT', 'cut'] : ['a0', 'gamma', 'gammaT', 'ws', 'wsT', 'cut']);
  }

  coeff(args: string[]): void {
    if (this.tstat) {
      if (args.length < 6 || args.length > 7) throw new StyleError('Incorrect args for pair coefficients');
      const v = args.slice(2, 6).map((w, q) => parseNum(w, ['gamma', 'gammaT', 'ws', 'wsT'][q]));
      const cut = args[6] !== undefined ? parseNum(args[6], 'cutoff') : this.cutGlobal;
      this.p.setRange(args[0], args[1], [...v, cut]);
      return;
    }
    if (args.length < 7 || args.length > 8) throw new StyleError('Incorrect args for pair coefficients');
    const v = args.slice(2, 7).map((w, q) => parseNum(w, ['a0', 'gamma', 'gammaT', 'ws', 'wsT'][q]));
    const cut = args[7] !== undefined ? parseNum(args[7], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [...v, cut]);
  }

  initStyle(ctx: StyleContext): void {
    // the random deviates follow native's pair order for orthogonal boxes (nativeOrder.ts); triclinic boxes keep the
    // engine's list order, so with T > 0 they match native LAMMPS only statistically (warned once in compute)
    this.logFn = (text) => ctx.log(text);
    this.runSteps = ctx.runSteps ?? null;
    // pair_dpd_ext.cpp init_style() inherits pair_dpd: ghost velocities are required
    if (ctx.ghostVelocity !== true) throw new StyleError(`Pair ${this.name} requires ghost atoms store velocity\n(use comm_modify vel yes)`);
    if (ctx.newtonPair === false) throw new StyleError(`pair_style ${this.name} with newton pair off is not supported by the browser engine`);
    this.boltz = ctx.s ? ctx.s.units.boltz : 1;
    this.nt = this.ntypes + 1;
    this.sigma = new Float64Array(this.nt * this.nt);
    this.sigmaT = new Float64Array(this.nt * this.nt);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    // sigma = sqrt(2.0*force->boltz*temperature*gamma), sigmaT likewise with gammaT (pair_dpd_ext.cpp init_one)
    const sig = Math.sqrt(2.0 * this.boltz * this.temperature * p.get('gamma', i, j));
    const sigT = Math.sqrt(2.0 * this.boltz * this.temperature * p.get('gammaT', i, j));
    const nt = this.nt;
    this.sigma[i * nt + j] = this.sigma[j * nt + i] = sig;
    this.sigmaT[i * nt + j] = this.sigmaT[j * nt + i] = sigT;
    return p.get('cut', i, j);
  }

  /** pair_dpd_ext_tstat.cpp compute(): "adjust sigma if target T is changing" (sigma and sigmaT). */
  private ramp(step: number): void {
    const rs = this.runSteps?.() ?? { begin: step, end: step };
    let delta = step - rs.begin;
    if (delta !== 0.0) delta /= rs.end - rs.begin;
    this.temperature = this.tStart + delta * (this.tStop - this.tStart);
    const nt = this.nt;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        this.sigma[i * nt + j] = this.sigma[j * nt + i] = Math.sqrt(2.0 * this.boltz * this.temperature * this.p.get('gamma', i, j));
        this.sigmaT[i * nt + j] = this.sigmaT[j * nt + i] = Math.sqrt(2.0 * this.boltz * this.temperature * this.p.get('gammaT', i, j));
      }
    }
  }

  compute(pc: PairCompute): void {
    const rng = this.rng;
    if (!rng) throw new StyleError(`pair ${this.name} used before pair_style`);
    if (this.tstat && this.tStart !== this.tStop) this.ramp(pc.s.step);
    if (pc.geom.triclinic && this.temperature > 0 && !this.triWarned) {
      this.triWarned = true;
      this.logFn?.('WARNING: pair_style ' + this.name + ': triclinic box: random forces are drawn in the browser engine\'s pair order (native\'s enumeration is not ported for triclinic boxes); with T > 0 the run matches native LAMMPS only statistically');
    }
    const pairs = this.order.pairs(pc.s, pc.nb, pc.geom, pc.half);
    const { x, f, type } = pc;
    const s = pc.s;
    const vel = s.v;
    const owner = pc.nb.owner;
    const nt = this.nt;
    const { cutsq, cut, sigma, sigmaT } = this;
    const gArr = this.p.p('gamma'), gTArr = this.p.p('gammaT'), wsArr = this.p.p('ws'), wsTArr = this.p.p('wsT');
    const aArr = this.tstat ? null : this.p.p('a0');
    const sLJ = pc.specialLJ;
    const sqrtS = [Math.sqrt(sLJ[0]), Math.sqrt(sLJ[1]), Math.sqrt(sLJ[2]), Math.sqrt(sLJ[3])];
    const dtinvsqrt = 1.0 / Math.sqrt(s.dt);
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const P = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const jj = pairs[p + 1];
      const j = jj & NEIGHMASK;
      const m = jj >>> SBBITS;
      const factorDpd = sLJ[m];
      const factorSqrt = sqrtS[m];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const vxi = vel[3 * i], vyi = vel[3 * i + 1], vzi = vel[3 * i + 2];
      const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const t = type[i] * nt + type[j];
      if (rsq >= cutsq[t]) continue;
      const r = Math.sqrt(rsq);
      if (r < EPSILON) continue;
      const rinv = 1.0 / r;
      const oj = owner[j];
      const delvx = vxi - vel[3 * oj], delvy = vyi - vel[3 * oj + 1], delvz = vzi - vel[3 * oj + 2];
      const dot = delx * delvx + dely * delvy + delz * delvz;
      // projector onto the plane perpendicular to the pair separation (pair_dpd_ext.cpp compute)
      P[0] = 1.0 - delx * delx * rinv * rinv;
      P[1] = -delx * dely * rinv * rinv;
      P[2] = -delx * delz * rinv * rinv;
      P[3] = P[1];
      P[4] = 1.0 - dely * dely * rinv * rinv;
      P[5] = -dely * delz * rinv * rinv;
      P[6] = P[2];
      P[7] = P[5];
      P[8] = 1.0 - delz * delz * rinv * rinv;
      const wd = 1.0 - r / cut[t];
      const wdPar = libmPow(wd, wsArr[t]);
      const wdPerp = libmPow(wd, wsTArr[t]);
      const randnum = rng.gaussian();
      const randnumx = rng.gaussian();
      const randnumy = rng.gaussian();
      const randnumz = rng.gaussian();
      let fpair: number;
      let e = 0;
      if (this.tstat) {
        // pair_dpd_ext_tstat.cpp: "fpair = -factor_dpd * gamma*wdPar*wdPar*dot*rinv;"
        fpair = -factorDpd * gArr[t] * wdPar * wdPar * dot * rinv;
        fpair += factorSqrt * sigma[t] * wdPar * randnum * dtinvsqrt;
      } else {
        const a0 = aArr![t];
        fpair = a0 * wd;
        fpair -= gArr[t] * wdPar * wdPar * dot * rinv;
        fpair *= factorDpd;
        fpair += factorSqrt * sigma[t] * wdPar * randnum * dtinvsqrt;
        e = 0.5 * a0 * cut[t] * wd * wd;
        e *= factorDpd;
      }
      let fx = fpair * rinv * delx, fy = fpair * rinv * dely, fz = fpair * rinv * delz;
      const prefactorG = factorDpd * gTArr[t] * wdPerp * wdPerp;
      fx -= prefactorG * (P[0] * delvx + P[1] * delvy + P[2] * delvz);
      fy -= prefactorG * (P[3] * delvx + P[4] * delvy + P[5] * delvz);
      fz -= prefactorG * (P[6] * delvx + P[7] * delvy + P[8] * delvz);
      const prefactorS = factorSqrt * sigmaT[t] * wdPerp * dtinvsqrt;
      fx += prefactorS * (P[0] * randnumx + P[1] * randnumy + P[2] * randnumz);
      fy += prefactorS * (P[3] * randnumx + P[4] * randnumy + P[5] * randnumz);
      fz += prefactorS * (P[6] * randnumx + P[7] * randnumy + P[8] * randnumz);
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      evdwl += e;
      if (tally) tallyXYZ(pc, i, j, e, fx, fy, fz, delx, dely, delz);
    }
    pc.acc.evdwl += evdwl;
  }

  dataCoeffs(): string[] { return []; }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    const names = this.tstat ? ['gamma', 'gammaT', 'ws', 'wsT', 'cut'] : ['a0', 'gamma', 'gammaT', 'ws', 'wsT', 'cut'];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) out.push(`${i} ${j} ${names.map((n) => fmtCoeff(this.p.get(n, i, j))).join(' ')}`);
    }
    return out;
  }
}
