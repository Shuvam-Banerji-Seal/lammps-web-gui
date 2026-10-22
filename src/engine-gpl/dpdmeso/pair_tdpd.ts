/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-MESO/pair_tdpd.cpp and pair_tdpd.h
 * (PairTDPD: pair_style tdpd, the transport DPD of Li, Yazdani, Tartakovsky and Karniadakis, J. Chem. Phys. 2015).
 * Contributing author (from pair_tdpd.cpp): Zhen Li (Clemson University). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style tdpd T cutoff seed; pair_coeff i j A gamma power cut cutcc (kappa epsilon powercc)*N (docs.lammps.org/pair_tdpd.html):
 * the momentum exchange is the DPD force of pair dpd; for each species k inside cutcc, the chemical concentration flux is a
 * diffusion term -kappa wcr^2 (c_i - c_j) plus a random term epsilon wcr (c_i + c_j) theta / sqrt(dt), with theta from the
 * same RanMars stream (clamped to +-5). The fluxes are kept in cc_flux (atom_style tdpd) and summed by fix mvv/tdpd.
 * Random numbers follow native's pair order (src/engine-gpl/neighbor/nativeOrder.ts).
 */
import { Pair, PairParams, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { TDPD_CC, TDPD_CC_FLUX, isTdpdStyle, tdpdSpecies } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { EPSILON, tallyXYZ } from '../dpd/pair_dpd';
import { pow as libmPow } from '../libm/pow';

const NAMES = ['a0', 'gamma', 'power', 'cut', 'cutcc'];

export class PairTDPD extends Pair {
  readonly name = 'tdpd';
  virialFdotr = true;
  private temperature = 0;
  private cutGlobal = 0;
  private seed = 0;
  private rng: RanMars | null = null;
  private p: PairParams | null = null;
  /** per type pair and species: kappa, epsilon and powercc (pair_tdpd.cpp kappa, epsilon, powercc), nt*nt*nsp each */
  private kappa = new Float64Array(0);
  private epsilon = new Float64Array(0);
  private powercc = new Float64Array(0);
  private nsp = 0;
  private boltz = 1;
  /** sigma per type pair (the amplitude of the random force, pair_tdpd.cpp sigma), set by initOne. */
  private sigma = new Float64Array(0);
  private readonly order = new NativePairOrder();

  settings(args: string[]): void {
    if (args.length !== 3) throw new StyleError('Illegal pair_style command');
    this.temperature = parseNum(args[0], 'temperature');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    this.seed = parseNum(args[2], 'seed');
    if (!(this.seed > 0) || !Number.isInteger(this.seed)) throw new StyleError('Invalid random number seed');
    this.rng = new RanMars(this.seed % 900000000);
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

  coeff(args: string[], ctx: StyleContext): void {
    const nsp = tdpdSpecies(ctx.s?.atomStyle ?? 'atomic');
    if (!nsp) throw new StyleError('Pair style tdpd requires atom properties cc/cc_flux (use atom_style tdpd N)');
    // pair_tdpd.cpp coeff(): "if (narg != 7 + 3*cc_species)"
    if (args.length !== 7 + 3 * nsp) throw new StyleError('Incorrect args for pair tdpd coefficients');
    if (!this.p) throw new StyleError('pair_coeff before the box is defined');
    const nt = this.ntypes + 1;
    if (this.nsp !== nsp || this.kappa.length !== nt * nt * nsp) {
      this.nsp = nsp;
      this.kappa = new Float64Array(nt * nt * nsp);
      this.epsilon = new Float64Array(nt * nt * nsp);
      this.powercc = new Float64Array(nt * nt * nsp);
    }
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const a0 = parseNum(args[2], 'a0'), gamma = parseNum(args[3], 'gamma'), power = parseNum(args[4], 'power');
    const cut = parseNum(args[5], 'cut'), cutcc = parseNum(args[6], 'cutcc');
    const kap: number[] = [], eps: number[] = [], pcc: number[] = [];
    for (let k = 0; k < nsp; k++) {
      kap.push(parseNum(args[7 + 3 * k], 'kappa'));
      eps.push(parseNum(args[8 + 3 * k], 'epsilon'));
      pcc.push(parseNum(args[9 + 3 * k], 'powercc'));
    }
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.p.set(i, j, [a0, gamma, power, cut, cutcc]);
        const idx = this.p.idx(i, j);
        for (let k = 0; k < nsp; k++) {
          this.kappa[nsp * idx + k] = kap[k];
          this.epsilon[nsp * idx + k] = eps[k];
          this.powercc[nsp * idx + k] = pcc[k];
        }
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    // pair_tdpd.cpp init_style(): "Pair style tdpd requires ghost atoms store velocity" and the cc / cc_flux properties
    if (ctx.ghostVelocity !== true) throw new StyleError('Pair style tdpd requires ghost atoms store velocity\n(use comm_modify vel yes)');
    if (!ctx.s || !isTdpdStyle(ctx.s.atomStyle) || !ctx.s.custom.has(TDPD_CC) || !ctx.s.custom.has(TDPD_CC_FLUX)) {
      throw new StyleError('Pair style tdpd requires atom properties cc/cc_flux (use atom_style tdpd N)');
    }
    if (ctx.newtonPair === false) throw new StyleError('pair_style tdpd with newton pair off is not supported by the browser engine');
    this.boltz = ctx.s.units.boltz;
    this.sigma = new Float64Array((this.ntypes + 1) * (this.ntypes + 1));
  }

  initOne(i: number, j: number): number {
    const p = this.p!;
    if (!p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    // pair_tdpd.cpp init_one(): sigma[i][j] = sqrt(2.0*force->boltz*temperature*gamma[i][j])
    const sigma = Math.sqrt(2.0 * this.boltz * this.temperature * p.get('gamma', i, j));
    this.sigma[p.idx(i, j)] = this.sigma[p.idx(j, i)] = sigma;
    return p.get('cut', i, j);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const rng = this.rng, p = this.p;
    if (!rng || !p) throw new StyleError('pair tdpd used before pair_style');
    const nsp = this.nsp;
    const cc = s.custom.get(TDPD_CC)!.data, ccFlux = s.custom.get(TDPD_CC_FLUX)!.data;
    // native force_clear(): the concentration fluxes start at zero for this force evaluation
    ccFlux.fill(0, 0, nsp * s.n);
    const { x, f, type } = pc;
    const vel = s.v;
    const owner = pc.nb.owner;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const a0 = p.p('a0'), gam = p.p('gamma'), pw = p.p('power'), cutA = p.p('cut'), cutcc = p.p('cutcc');
    const sigma = this.sigma;
    const sLJ = pc.specialLJ;
    const dtinvsqrt = 1.0 / Math.sqrt(s.dt);
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
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
      let wc = 1.0 - r / cutA[t];
      wc = Math.max(0, Math.min(1.0, wc));
      const wr = libmPow(wc, 0.5 * pw[t]);
      const randnum = rng.gaussian();
      let fpair = a0[t] * wc;
      fpair -= gam[t] * wr * wr * dot * rinv;
      fpair += sigma[t] * wr * randnum * dtinvsqrt;
      fpair *= factorDpd * rinv;
      const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      if (r < cutcc[t]) {
        for (let k = 0; k < nsp; k++) {
          let wcr = 1.0 - r / cutcc[t];
          wcr = Math.max(0, wcr);
          wcr = libmPow(wcr, 0.5 * this.powercc[nsp * t + k]);
          let rn = rng.gaussian();
          rn = Math.max(-5.0, Math.min(rn, 5.0));
          const ci = cc[nsp * i + k], cj = cc[nsp * oj + k];
          const dQc = -this.kappa[nsp * t + k] * wcr * wcr * (ci - cj);
          const dQr = this.epsilon[nsp * t + k] * wcr * (ci + cj) * rn * dtinvsqrt;
          ccFlux[nsp * i + k] += (dQc + dQr);
          ccFlux[nsp * oj + k] -= (dQc + dQr);
        }
      }
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      let e = 0.5 * a0[t] * cutA[t] * wc * wc;
      e *= factorDpd;
      evdwl += e;
      if (tally) tallyXYZ(pc, i, j, e, fx, fy, fz, delx, dely, delz);
    }
    pc.acc.evdwl += evdwl;
  }
}
