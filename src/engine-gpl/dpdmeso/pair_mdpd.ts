/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-MESO/pair_mdpd.cpp and pair_mdpd.h
 * (PairMDPD: pair_style mdpd) and src/DPD-MESO/pair_mdpd_rhosum.cpp and pair_mdpd_rhosum.h (PairMDPDRhoSum: pair_style
 * mdpd/rhosum, the density of each particle). Contributing author (from both files): Zhen Li (Clemson University).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style mdpd T cutoff seed; pair_coeff i j A B gamma cut cutr (docs.lammps.org/pair_mesodpd.html): the many-body
 * force uses the densities rho_i + rho_j of the atoms, and the random force a Gaussian deviate per pair from RanMars.
 * mdpd/rhosum (listed first in the hybrid/overlay) sums the density of each owned atom over its full neighbor list with the
 * quintic weight function of the source, rho_i = sum_j m_j w(r_ij); the density of a ghost is that of its owner.
 */
import { Pair, PairParams, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseInt_, parseNum } from '../../engine/force/util';
import { MDPD_RHO, isMdpdStyle } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { EPSILON, tallyXYZ } from '../dpd/pair_dpd';

const NAMES = ['A', 'B', 'gamma', 'cut', 'cutr', 'sigma'];

export class PairMDPD extends Pair {
  readonly name = 'mdpd';
  virialFdotr = true;
  private temperature = 0;
  private cutGlobal = 0;
  private seed = 0;
  private rng: RanMars | null = null;
  private p: PairParams | null = null;
  private boltz = 1;
  private readonly order = new NativePairOrder();

  settings(args: string[]): void {
    if (args.length !== 3) throw new StyleError('Illegal pair_style command');
    this.temperature = parseNum(args[0], 'temperature');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    this.seed = parseInt_(args[2], 'seed');
    if (!(this.seed > 0) || !Number.isInteger(this.seed)) throw new StyleError('Invalid random number seed');
    this.rng = new RanMars(this.seed % 900000000);
    // pair_mdpd.cpp settings(): the cutoff of set pairs (i < j) is taken from the pair_style cutoff again
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
    if (args.length !== 7) throw new StyleError('Incorrect args for pair coefficients');
    if (!this.p) throw new StyleError('pair_coeff before the box is defined');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const A = parseNum(args[2], 'A'), B = parseNum(args[3], 'B'), gamma = parseNum(args[4], 'gamma');
    const cut = parseNum(args[5], 'cut'), cutr = parseNum(args[6], 'cutr');
    // pair_mdpd.cpp coeff(): "Value for cutA should be larger than cutB."
    if (cut < cutr) throw new StyleError('Value for cutA should be larger than cutB.');
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.p.set(i, j, [A, B, gamma, cut, cutr, 0]);
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    // pair_mdpd.cpp init_style(): "Pair mdpd requires ghost atoms store velocity" and "Pair style mdpd requires atom attribute rho"
    if (ctx.ghostVelocity !== true) throw new StyleError('Pair mdpd requires ghost atoms store velocity\n(use comm_modify vel yes)');
    if (!ctx.s || !isMdpdStyle(ctx.s.atomStyle)) throw new StyleError('Pair style mdpd requires atom attribute rho (use atom_style mdpd)');
    if (ctx.newtonPair === false) throw new StyleError('pair_style mdpd with newton pair off is not supported by the browser engine');
    this.boltz = ctx.s.units.boltz;
  }

  initOne(i: number, j: number): number {
    const p = this.p!;
    if (!p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    // pair_mdpd.cpp init_one(): sigma[i][j] = sqrt(2.0*force->boltz*temperature*gamma[i][j])
    const sigma = Math.sqrt(2.0 * this.boltz * this.temperature * p.get('gamma', i, j));
    p.p('sigma')[p.idx(i, j)] = p.p('sigma')[p.idx(j, i)] = sigma;
    return p.get('cut', i, j);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const rng = this.rng, p = this.p;
    if (!rng || !p) throw new StyleError('pair mdpd used before pair_style');
    const rho = s.custom.get(MDPD_RHO)!.data;
    const { x, f, type } = pc;
    const vel = s.v;
    const owner = pc.nb.owner;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const A = p.p('A'), B = p.p('B'), gam = p.p('gamma'), cutA = p.p('cut'), cutR = p.p('cutr'), sig = p.p('sigma');
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
      const wc = 1.0 - r / cutA[t];
      let wcR = 1.0 - r / cutR[t];
      wcR = Math.max(wcR, 0.0);
      const wr = wc;
      const rhoi = rho[i], rhoj = rho[oj];
      const randnum = rng.gaussian();
      let fpair = A[t] * wc + B[t] * (rhoi + rhoj) * wcR;
      fpair -= gam[t] * wr * wr * dot * rinv;
      fpair += sig[t] * wr * randnum * dtinvsqrt;
      fpair *= factorDpd * rinv;
      const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      let e = 0.5 * A[t] * cutA[t] * wr * wr + 0.5 * B[t] * cutR[t] * (rhoi + rhoj) * wcR * wcR;
      e *= factorDpd;
      evdwl += e;
      if (tally) tallyXYZ(pc, i, j, e, fx, fy, fz, delx, dely, delz);
    }
    pc.acc.evdwl += evdwl;
  }
}

/** pair_style mdpd/rhosum: the density rho of each owned atom from its full neighbor list (no forces). */
export class PairMDPDRhoSum extends Pair {
  readonly name = 'mdpd/rhosum';
  /** no forces, so no virial: the global virial of the hybrid is the fdotr sum of the other sub-styles */
  virialFdotr = true;
  /** the density needs every neighbor of an atom: a full list (pair_mdpd_rhosum.cpp init_style: REQ_FULL) */
  needsFull = true;
  needsHalf = false;
  private p: PairParams | null = null;

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal number of setting arguments for pair_style mdpd/rhosum');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['cut']);
  }

  coeff(args: string[]): void {
    // pair_mdpd_rhosum.cpp coeff(): "narg != 3" is an error; the one value is the cutoff of the density
    if (args.length !== 3) throw new StyleError('Incorrect args for pair coefficients');
    if (!this.p) throw new StyleError('pair_coeff before the box is defined');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const cut = parseNum(args[2], 'cutoff');
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.p.set(i, j, [cut]);
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    // pair_mdpd_rhosum.cpp init_style(): "Pair style mdpd/rhosum requires atom attribute rho"
    if (!ctx.s || !isMdpdStyle(ctx.s.atomStyle)) throw new StyleError('Pair style mdpd/rhosum requires atom attribute rho (use atom_style mdpd)');
  }

  initOne(i: number, j: number): number {
    const p = this.p!;
    if (!p.isSet(i, j)) throw new StyleError(`All pair mdpd/rhosum coeffs are not set (pair ${i} ${j})`);
    return p.get('cut', i, j);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const full = pc.full;
    if (!full) throw new StyleError('pair mdpd/rhosum needs a full neighbor list');
    const rho = s.custom.get(MDPD_RHO)!.data;
    const x = pc.x, type = pc.type;
    const nt = this.ntypes + 1;
    const cutSq = this.cutsq;
    const cutA = this.p!.p('cut');
    const mass = s.massByType;
    rho.fill(0, 0, s.n);
    for (let i = 0; i < pc.nlocal; i++) {
      const itype = type[i];
      const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
      for (let q = k0; q < k1; q++) {
        const j = full.neighbors[q] & NEIGHMASK;
        const jtype = type[j];
        const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq >= cutSq[itype * nt + jtype]) continue;
        const h = cutA[itype * nt + jtype];
        const ih = 1.0 / h;
        const ihsq = ih * ih;
        const r = Math.sqrt(rsq);
        let wf = (h - r) * ihsq;
        wf = 2.0889086280811262819 * (h + 3.0 * r) * wf * wf * wf * ih;
        rho[i] += mass[jtype] * wf;
      }
    }
  }
}
