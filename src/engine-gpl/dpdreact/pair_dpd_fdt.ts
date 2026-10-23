/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/pair_dpd_fdt.cpp and
 * pair_dpd_fdt.h (PairDPDfdt: pair_style dpd/fdt) and src/DPD-REACT/pair_dpd_fdt_energy.cpp and
 * pair_dpd_fdt_energy.h (PairDPDfdtEnergy: pair_style dpd/fdt/energy). LAMMPS is Copyright (2003) Sandia Corporation
 * and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Contributing author (from both files): James Larentzos (U.S. Army Research Laboratory).
 *
 * Forces (docs.lammps.org/pair_dpd_fdt.html): the conservative term a0 wr, the drag -gamma wd (dv . rhat) and the random
 * term sigma wr xi / sqrt(dt), with wr = 1 - r/rc and wd = wr^2; the whole pair force is scaled by factor_dpd (special
 * bonds). "where T is the set point temperature" (dpd/fdt); gamma = sigma^2 / (2 kB T). dpd/fdt/energy takes the
 * temperature from the particles: gamma uses the harmonic mean of dpdTheta of the pair. The energy style also carries
 * the mechanical and conductive energy rates duMech, duCond per atom (pair_dpd_fdt.html: "alpha = sqrt(2 kB kappa)").
 *
 * Random numbers: one Gaussian deviate per pair for dpd/fdt; two per pair for dpd/fdt/energy (the first for the
 * force and the mechanical energy, the second for the conductive energy), drawn from a Marsaglia RanMars stream in
 * native LAMMPS's neighbor-list order (src/engine-gpl/neighbor/nativeOrder.ts).
 *
 * With fix shardlow the pair keeps only the conservative term (splitFdt); fix_shardlow.ts applies the rest.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { fmtCoeff, parseInt_, parseNum } from '../../engine/force/util';
import { massOf, DPD_THETA } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { EPSILON, tallyXYZ } from '../dpd/pair_dpd';

/** Shared part of pair dpd/fdt and dpd/fdt/energy (the force loop of both native styles). */
abstract class PairDPDfdtBase extends Pair {
  virialFdotr = true;
  cutGlobal = 0;
  seed = 0;
  p!: PairParams;
  /**
   * Set when fix shardlow is defined (pair_dpd_fdt.cpp init_style: splitFDT_flag = a fix shardlow exists): the pair
   * then keeps only the conservative force a0 wr, and fix shardlow applies the dissipative and random parts.
   */
  splitFdt = false;
  protected rng: RanMars | null = null;
  protected boltz = 1;
  protected readonly order = new NativePairOrder();
  protected nt = 0;
  protected cutArr = new Float64Array(0);
  /** Per-atom rates of the internal energies, owned atoms (filled by compute; read by fix dpd/energy). */
  duCond = new Float64Array(0);
  duMech = new Float64Array(0);

  protected abstract readonly paramNames: string[];

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, this.paramNames);
  }

  protected seedRng(seed: number): void {
    // pair_dpd_fdt.cpp settings(): "if (seed <= 0) error->all(FLERR,"Illegal pair_style command");" then RanMars(seed + me)
    if (!(seed > 0)) throw new StyleError('Illegal pair_style command');
    this.rng = new RanMars(seed);
  }

  initStyle(ctx: StyleContext): void {
    // pair_dpd_fdt.cpp init_style(): "Pair dpd/fdt requires ghost atoms store velocity"
    if (ctx.ghostVelocity !== true) throw new StyleError(`Pair ${this.name} requires ghost atoms store velocity\n(use comm_modify vel yes)`);
    if (ctx.newtonPair === false) throw new StyleError(`pair_style ${this.name} with newton pair off is not supported by the browser engine`);
    this.boltz = ctx.s ? ctx.s.units.boltz : 1;
    this.nt = this.ntypes + 1;
    this.cutArr = new Float64Array(this.nt * this.nt);
  }

  initOne(i: number, j: number): number {
    // pair_dpd_fdt.cpp init_one(): "All pair coeffs are not set"; no mixing
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    const c = this.p.get('cut', i, j);
    this.cutArr[i * this.nt + j] = this.cutArr[j * this.nt + i] = c;
    return c;
  }

  /** Sum of the per-pair work shared by both styles: the conservative and drag/random force (factor_dpd applies to all). */
  protected pairKernel(pc: PairCompute): void {
    const rng = this.rng;
    if (!rng) throw new StyleError(`${this.name} used before pair_style`);
    if (this.splitFdt) { this.conservativeKernel(pc); return; }
    const pairs = this.order.pairs(pc.s, pc.nb, pc.geom, pc.half);
    const { x, f, type } = pc;
    const s = pc.s;
    const vel = s.v;
    const owner = pc.nb.owner;
    const nt = this.nt;
    const cutArr = this.cutArr;
    const sLJ = pc.specialLJ;
    const dtinvsqrt = 1.0 / Math.sqrt(s.dt);
    const tally = pc.eatom !== null || pc.vatom !== null;
    const nlocal = pc.nlocal;
    if (this.duCond.length !== nlocal) {
      this.duCond = new Float64Array(nlocal);
      this.duMech = new Float64Array(nlocal);
    } else {
      this.duCond.fill(0);
      this.duMech.fill(0);
    }
    // contributions to ghost atoms are summed into their owners after the pair loop, as native's reverse communication does
    const energy = this.energyStyle;
    const nghost = pc.nall - nlocal;
    const gCond = energy ? new Float64Array(nghost) : null, gMech = energy ? new Float64Array(nghost) : null;
    const aArr = this.p.p('a0'), gArr = this.p.p('sigma');
    const kArr = energy ? this.p.p('kappa') : null, alArr = energy ? this.p.p('alpha') : null;
    const theta = energy ? s.custom.get(DPD_THETA)!.data : null;
    const temperature = this.temperature;
    let evdwl = 0;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const jj = pairs[p + 1];
      const j = jj & NEIGHMASK;
      const m = jj >>> SBBITS;
      const factorDpd = sLJ[m];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const vxi = vel[3 * i], vyi = vel[3 * i + 1], vzi = vel[3 * i + 2];
      const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const t = type[i] * nt + type[j];
      if (rsq >= this.cutsqOf(t)) continue;
      const r = Math.sqrt(rsq);
      if (r < EPSILON) continue;
      const rinv = 1.0 / r;
      const oj = owner[j];
      const delvx = vxi - vel[3 * oj], delvy = vyi - vel[3 * oj + 1], delvz = vzi - vel[3 * oj + 2];
      const dot = delx * delvx + dely * delvy + delz * delvz;
      const wr = 1.0 - r / cutArr[t];
      const wd = wr * wr;
      const randnum = rng.gaussian();
      let gammaIJ: number;
      if (energy) {
        // dpd/fdt/energy: theta_ij is the harmonic mean of the two internal temperatures (pair_dpd_fdt_energy.cpp)
        let thetaIJ = 0.5 * (1.0 / theta![i] + 1.0 / theta![oj]);
        thetaIJ = 1.0 / thetaIJ;
        gammaIJ = gArr[t] * gArr[t] / (2.0 * this.boltz * thetaIJ);
      } else {
        gammaIJ = gArr[t] * gArr[t] / (2.0 * this.boltz * temperature);
      }
      // conservative force = a0 * wr; drag force = -gamma * wd * (delx dot delv) / r; random = sigma * wr * rnd * dtinvsqrt
      let fpair = aArr[t] * wr;
      fpair -= gammaIJ * wd * dot * rinv;
      fpair += gArr[t] * wr * randnum * dtinvsqrt;
      fpair *= factorDpd * rinv;
      const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      if (energy) {
        const mi = massOf(s, i), mj = massOf(s, oj);
        const muIJ = (1.0 / mi + 1.0 / mj) * s.units.ftm2v;
        const sig = gArr[t];
        let uTmp = gammaIJ * wd * rinv * rinv * dot * dot - 0.5 * sig * sig * muIJ * wd;
        uTmp -= sig * wr * rinv * dot * randnum * dtinvsqrt;
        uTmp *= 0.5;
        this.duMech[i] += uTmp;
        if (j < nlocal) this.duMech[j] += uTmp; else gMech![j - nlocal] += uTmp;
        // second deviate: the conductive energy (pair_dpd_fdt_energy.cpp "randnum = random->gaussian();")
        const randnum2 = rng.gaussian();
        const kappaIJ = kArr![t];
        const randPair = alArr![t] * wr * randnum2 * dtinvsqrt;
        let uc = kappaIJ * (1.0 / theta![i] - 1.0 / theta![oj]) * wd;
        uc += randPair;
        this.duCond[i] += uc;
        if (j < nlocal) this.duCond[j] -= uc; else gCond![j - nlocal] -= uc;
      }
      // unshifted conservative energy 0.5 a0 rc wd^2 (shifted to 0 at the cutoff), times factor_dpd
      let e = 0.5 * aArr[t] * cutArr[t] * wd;
      e *= factorDpd;
      evdwl += e;
      if (tally) tallyXYZ(pc, i, j, e, fx, fy, fz, delx, dely, delz);
    }
    if (energy) {
      for (let g = nghost - 1; g >= 0; g--) {
        const o = owner[nlocal + g];
        this.duMech[o] += gMech![g];
        this.duCond[o] += gCond![g];
      }
    }
    pc.acc.evdwl += evdwl;
  }

  /**
   * The per-type-pair parameters fix shardlow reads (same arrays the force loop uses; index i * nt + j).
   * kappa and alpha are only set for dpd/fdt/energy.
   */
  shardlowView(): { nt: number; cut: Float64Array; sigma: Float64Array; kappa: Float64Array | null; alpha: Float64Array | null; temperature: number; seed: number; boltz: number; energy: boolean } {
    return {
      nt: this.nt,
      cut: this.cutArr,
      sigma: this.p.p('sigma'),
      kappa: this.energyStyle ? this.p.p('kappa') : null,
      alpha: this.energyStyle ? this.p.p('alpha') : null,
      temperature: this.temperature,
      seed: this.seed,
      boltz: this.boltz,
      energy: this.energyStyle,
    };
  }

  /**
   * The split form used with fix shardlow (pair_dpd_fdt.cpp compute, splitFDT_flag): the conservative force
   * a0 wr only, with its shifted energy 0.5 a0 rc wd (times factor_dpd); no random numbers, no drag, no internal energy rates.
   */
  private conservativeKernel(pc: PairCompute): void {
    const pairs = this.order.pairs(pc.s, pc.nb, pc.geom, pc.half);
    const { x, f, type } = pc;
    const nt = this.nt;
    const cutArr = this.cutArr;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const aArr = this.p.p('a0');
    let evdwl = 0;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const jj = pairs[p + 1];
      const j = jj & NEIGHMASK;
      const factorDpd = sLJ[jj >>> SBBITS];
      const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const t = type[i] * nt + type[j];
      if (rsq >= this.cutsqOf(t)) continue;
      const r = Math.sqrt(rsq);
      if (r < EPSILON) continue;
      const rinv = 1.0 / r;
      const wr = 1.0 - r / cutArr[t];
      const wd = wr * wr;
      let fpair = aArr[t] * wr;
      fpair *= factorDpd * rinv;
      const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      const e = 0.5 * aArr[t] * cutArr[t] * wd * factorDpd;
      evdwl += e;
      if (tally) tallyXYZ(pc, i, j, e, fx, fy, fz, delx, dely, delz);
    }
    pc.acc.evdwl += evdwl;
  }

  protected cutsqOf(t: number): number {
    const c = this.cutArr[t];
    return c * c;
  }

  protected abstract readonly energyStyle: boolean;
  protected abstract readonly temperature: number;

  compute(pc: PairCompute): void { this.pairKernel(pc); }

  dataCoeffs(): string[] { return []; }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const parts = this.paramNames;
        out.push(`${i} ${j} ${parts.map((n) => fmtCoeff(this.p.get(n, i, j))).join(' ')}`);
      }
    }
    return out;
  }
}

/** pair_style dpd/fdt T cutoff seed (docs.lammps.org/pair_dpd_fdt.html; pair_coeff a0 sigma [cutoff]). */
export class PairDPDfdt extends PairDPDfdtBase {
  readonly name = 'dpd/fdt';
  protected readonly paramNames = ['a0', 'sigma', 'cut'];
  protected readonly energyStyle = false;
  temperature = 0;

  settings(args: string[]): void {
    if (args.length !== 3) throw new StyleError('Illegal pair_style command (dpd/fdt T cutoff seed)');
    this.temperature = parseNum(args[0], 'temperature');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    this.seed = parseInt_(args[2], 'seed');
    this.seedRng(this.seed);
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    const a0 = parseNum(args[2], 'a0');
    const sigma = parseNum(args[3], 'sigma');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [a0, sigma, cut]);
  }
}

/** pair_style dpd/fdt/energy cutoff seed (needs atom_style dpd; pair_coeff a0 sigma kappa [cutoff]). */
export class PairDPDfdtEnergy extends PairDPDfdtBase {
  readonly name = 'dpd/fdt/energy';
  protected readonly paramNames = ['a0', 'sigma', 'kappa', 'alpha', 'cut'];
  protected readonly energyStyle = true;
  readonly temperature = 0;

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('Illegal pair_style command (dpd/fdt/energy cutoff seed)');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.seed = parseInt_(args[1], 'seed');
    this.seedRng(this.seed);
  }

  coeff(args: string[]): void {
    if (args.length < 5 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    const a0 = parseNum(args[2], 'a0');
    const sigma = parseNum(args[3], 'sigma');
    const kappa = parseNum(args[4], 'kappa');
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : this.cutGlobal;
    // alpha needs the Boltzmann constant of the units, which the style only knows at init (see initStyle)
    this.p.setRange(args[0], args[1], [a0, sigma, kappa, 0, cut]);
  }

  initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    // pair_dpd_fdt_energy.cpp settings(): "pair_style dpd/fdt/energy requires atom_style with internal temperature
    // and energies (e.g. dpd)"
    if (!ctx.s || !ctx.s.custom.has(DPD_THETA)) throw new StyleError('pair_style dpd/fdt/energy requires atom_style with internal temperature and energies (e.g. dpd)');
    // pair_dpd_fdt_energy.cpp coeff(): "alpha_one = sqrt(2.0*force->boltz*kappa_one);" (filled here: the units are set by now)
    const kappa = this.p.p('kappa'), alpha = this.p.p('alpha');
    for (let t = 0; t < kappa.length; t++) alpha[t] = Math.sqrt(2.0 * this.boltz * kappa[t]);
    // pair_dpd_fdt_energy.cpp init_style(): "if (modify->get_fix_by_style("^eos").empty()) error->all(...)". The engine has
    // the EOS fixes of native's "^eos" family (eos/cv, eos/table, eos/table/rx) and matches them by style name.
    if (ctx.fixOf && !['eos/cv', 'eos/table', 'eos/table/rx'].some((st) => ctx.fixOf!(st) !== null)) {
      throw new StyleError('pair_style dpd/fdt/energy requires an EOS fix to be specified');
    }
  }
}
