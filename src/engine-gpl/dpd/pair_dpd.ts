/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-BASIC/pair_dpd.cpp and pair_dpd.h
 * (PairDPD: pair_style dpd), src/DPD-BASIC/pair_dpd_tstat.cpp (PairDPDTstat: pair_style dpd/tstat), the
 * PairDPD::single() of pair_dpd.cpp (compute pair/local) and the src/random_mars.cpp generator they draw from. LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Contributing author (from pair_dpd.cpp): Kurt Smith (U Pittsburgh).
 *
 * Forces (docs.lammps.org/pair_dpd.html): the conservative term a0 wd, the drag -gamma wd^2 (dv . rhat), and the
 * random term sigma wd theta / sqrt(dt), sigma = sqrt(2 kB T gamma), wd = 1 - r/rc, with theta a Gaussian deviate
 * from a Marsaglia RanMars stream seeded by the pair_style seed. "dpd/tstat" has no conservative term; its sigma
 * follows the temperature of the pair_style, ramped from t_start to t_stop over the run as pair_dpd_tstat.cpp compute()
 * does: T = t_start + delta (t_stop - t_start) with delta = (ntimestep - beginstep) / (endstep - beginstep), and sigma
 * recomputed for every type pair.
 *
 * Random numbers are drawn once per pair inside the cutoff, in native LAMMPS's neighbor-list order: the pair
 * order comes from src/engine-gpl/neighbor/nativeOrder.ts (native local order, ghost atoms, bins and the half
 * stencil), so the random forces match native step by step (oracle cases tests/oracle/w77dpd_*.in). Triclinic
 * boxes keep the engine's list order (the native enumeration is not ported for them).
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { fmtCoeff, parseInt_, parseNum } from '../../engine/force/util';
import { RanMars } from '../random_mars';
import { NativePairOrder } from '../neighbor/nativeOrder';

/**
 * ev_tally_xyz (src/pair.cpp) for a pair with force components (fx, fy, fz) on i and the energy e: per-atom energy
 * and virial are split half to each atom; the global virial is left to the force field (virialFdotr).
 */
export const tallyXYZ = (pc: PairCompute, i: number, j: number, e: number, fx: number, fy: number, fz: number,
  dx: number, dy: number, dz: number): void => {
  if (pc.eatom) { pc.eatom[i] += 0.5 * e; pc.eatom[j] += 0.5 * e; }
  if (pc.vatom) {
    const va = pc.vatom;
    const w = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
    for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * w[c]; va[6 * j + c] += 0.5 * w[c]; }
  }
};

/** pair_dpd.cpp: "static constexpr double EPSILON = 1.0e-10;" (r can be 0.0 in DPD systems). */
export const EPSILON = 1.0e-10;

/** Shared settings and per-type-pair data of the DPD styles that use a single random deviate per pair. */
export class PairDPD extends Pair {
  readonly name: string = 'dpd';
  virialFdotr = true;
  /** true for pair_style dpd/tstat (no conservative term, the temperature is the pair_style's). */
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
  private nt = 0;
  private readonly order = new NativePairOrder();
  private logFn: ((text: string) => void) | null = null;
  private triWarned = false;
  /** the run's begin and end steps (StyleContext.runSteps), read for the dpd/tstat ramp */
  private runSteps: (() => { begin: number; end: number }) | null = null;

  constructor(tstat = false) {
    super();
    this.tstat = tstat;
    if (tstat) {
      this.name = 'dpd/tstat';
      // pair_dpd_tstat.cpp: "single_enable = 0;" (compute pair/local refuses this style)
      this.singleOff = true;
    }
  }

  settings(args: string[]): void {
    const want = this.tstat ? 4 : 3;
    if (args.length !== want) throw new StyleError(this.tstat ? 'Illegal pair_style command (dpd/tstat t_start t_stop cutoff seed)' : 'Illegal pair_style command (dpd T cutoff seed)');
    if (this.tstat) {
      this.tStart = parseNum(args[0], 't_start');
      this.tStop = parseNum(args[1], 't_stop');
      this.temperature = this.tStart;
      this.cutGlobal = parseNum(args[2], 'cutoff');
      this.seed = parseInt_(args[3], 'seed');
    } else {
      this.temperature = parseNum(args[0], 'temperature');
      this.cutGlobal = parseNum(args[1], 'cutoff');
      this.seed = parseInt_(args[2], 'seed');
    }
    // pair_dpd.cpp settings(): "if (seed <= 0) error->all(FLERR,"Illegal pair_style command");" then RanMars(seed + me)
    if (!(this.seed > 0)) throw new StyleError('Illegal pair_style command');
    this.rng = new RanMars(this.seed);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, this.tstat ? ['gamma', 'cut'] : ['a0', 'gamma', 'cut']);
  }

  coeff(args: string[]): void {
    if (this.tstat) {
      if (args.length < 3 || args.length > 4) throw new StyleError('Incorrect args for pair coefficients');
      const gamma = parseNum(args[2], 'gamma');
      const cut = args[3] !== undefined ? parseNum(args[3], 'cutoff') : this.cutGlobal;
      this.p.setRange(args[0], args[1], [gamma, cut]);
      return;
    }
    if (args.length < 4 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    const a0 = parseNum(args[2], 'a0');
    const gamma = parseNum(args[3], 'gamma');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [a0, gamma, cut]);
  }

  initStyle(ctx: StyleContext): void {
    // the random deviates follow native's pair order for orthogonal boxes (nativeOrder.ts); triclinic boxes keep the
    // engine's list order, so with T > 0 they match native LAMMPS only statistically (warned once in compute)
    this.logFn = (text) => ctx.log(text);
    this.runSteps = ctx.runSteps ?? null;
    // pair_dpd.cpp init_style(): "Pair dpd requires ghost atoms store velocity"
    if (ctx.ghostVelocity !== true) throw new StyleError('Pair dpd requires ghost atoms store velocity\n(use comm_modify vel yes)');
    // pair_dpd.cpp: "Pair dpd needs newton pair on for momentum conservation" (a warning natively; this port
    // applies only the newton-on force update, so newton off is refused)
    if (ctx.newtonPair === false) throw new StyleError('pair_style dpd with newton pair off is not supported by the browser engine');
    this.boltz = ctx.s ? ctx.s.units.boltz : 1;
    this.nt = this.ntypes + 1;
    this.sigma = new Float64Array(this.nt * this.nt);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    // pair_dpd.cpp init_one(): "All pair coeffs are not set"; no mixing
    if (!p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    // sigma = sqrt(2.0*force->boltz*temperature*gamma[i][j]) (pair_dpd.cpp init_one)
    const sig = Math.sqrt(2.0 * this.boltz * this.temperature * p.get('gamma', i, j));
    const nt = this.nt;
    this.sigma[i * nt + j] = this.sigma[j * nt + i] = sig;
    return p.get('cut', i, j);
  }

  /** pair_dpd_tstat.cpp compute(): "adjust sigma if target T is changing". */
  private ramp(step: number): void {
    const rs = this.runSteps?.() ?? { begin: step, end: step };
    let delta = step - rs.begin;
    if (delta !== 0.0) delta /= rs.end - rs.begin;
    this.temperature = this.tStart + delta * (this.tStop - this.tStart);
    const nt = this.nt;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        this.sigma[i * nt + j] = this.sigma[j * nt + i] = Math.sqrt(2.0 * this.boltz * this.temperature * this.p.get('gamma', i, j));
      }
    }
  }

  compute(pc: PairCompute): void {
    const rng = this.rng;
    if (!rng) throw new StyleError('pair dpd used before pair_style');
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
    const { cutsq, cut, sigma } = this;
    const aArr = this.tstat ? null : this.p.p('a0');
    const gArr = this.p.p('gamma');
    const sLJ = pc.specialLJ;
    const sqrtS = [Math.sqrt(sLJ[0]), Math.sqrt(sLJ[1]), Math.sqrt(sLJ[2]), Math.sqrt(sLJ[3])];
    const dtinvsqrt = 1.0 / Math.sqrt(s.dt);
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
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
      const wd = 1.0 - r / cut[t];
      const randnum = rng.gaussian();
      let fpair: number;
      let e = 0;
      if (this.tstat) {
        // pair_dpd_tstat.cpp compute(): "fpair = -factor_dpd*gamma*wd*wd*dot*rinv;"
        fpair = -factorDpd * gArr[t] * wd * wd * dot * rinv;
        fpair += factorSqrt * sigma[t] * wd * randnum * dtinvsqrt;
      } else {
        const a0 = aArr![t];
        fpair = a0 * wd;
        fpair -= gArr[t] * wd * wd * dot * rinv;
        fpair *= factorDpd;
        fpair += factorSqrt * sigma[t] * wd * randnum * dtinvsqrt;
        // unshifted conservative energy is 0.5 a0 rc wd^2 (eng shifted to 0 at the cutoff), times factor_dpd
        e = 0.5 * a0 * cut[t] * wd * wd;
        e *= factorDpd;
      }
      fpair *= rinv;
      const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      evdwl += e;
      if (tally) tallyXYZ(pc, i, j, e, fx, fy, fz, delx, dely, delz);
    }
    pc.acc.evdwl += evdwl;
  }

  /**
   * PairDPD::single (pair_dpd.cpp at b332f28): the conservative part only, with no random or drag term, no cutoff test.
   * "if (r < EPSILON) { fforce = 0.0; return 0.0; }"; "wd = 1.0 - r/cut[itype][jtype]";
   * "fforce = a0[itype][jtype]*wd * factor_dpd*rinv"; "phi = 0.5*a0[itype][jtype]*cut[itype][jtype] * wd*wd; return factor_dpd*phi".
   */
  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _factorCoul: number, factorDpd: number, _qi = 0, _qj = 0): { eng: number; fforce: number } {
    const r = Math.sqrt(rsq);
    if (r < EPSILON) return { eng: 0, fforce: 0 };
    const t = itype * this.nt + jtype;
    const a0 = this.p.p('a0')[t], cut = this.cut[t];
    const rinv = 1.0 / r;
    const wd = 1.0 - r / cut;
    const fforce = a0 * wd * factorDpd * rinv;
    const phi = 0.5 * a0 * cut * wd * wd;
    return { eng: factorDpd * phi, fforce };
  }

  dataCoeffs(): string[] { return []; }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const parts = this.tstat ? ['gamma', 'cut'] : ['a0', 'gamma', 'cut'];
        out.push(`${i} ${j} ${parts.map((n) => fmtCoeff(this.p.get(n, i, j))).join(' ')}`);
      }
    }
    return out;
  }
}
