/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/COLLOID/pair_brownian_poly.cpp and
 * pair_brownian_poly.h (Amit Kumar and Michael Bybee (UIUC); Dave Heine (Corning), polydispersity). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * pair_style brownian/poly mu flaglog flagfld cut_inner cut T seed [flagHI flagVF] (docs.lammps.org/pair_brownian.html):
 * the polydisperse Brownian style. It needs newton pair off (native: "Pair brownian/poly requires newton pair off") and
 * a full neighbor list: each pair is visited from both atoms, and each visit moves only the owned atom i. The
 * resistances use the two radii (beta0 = radj / radi) with the powint() polynomials of pair_brownian_poly.cpp.
 *
 * Random draws: one per atom FLD term, then per visit one a_sq draw, two per a_sh and two per a_pu term, in the order
 * of the full list. The engine walks the full list in its own order; native's full-list enumeration (bins, stencil) is
 * not reproduced, so runs where an atom has more than one partner inside the cutoff can differ in the draw order.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { nativeOrder } from '../../engine/atoms';
import { pow as libmPow } from '../libm/pow';
import { log as libmLog } from '../libm/log';
import { PairBrownian, cube, orthogonalVectors, tallyVatom, wallMoves, VXMU2F } from './pair_brownian';

/** math_special.h powint(x, n): repeated squaring for integer n (the exact operation order of the source). */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

export class PairBrownianPoly extends PairBrownian {
  override readonly name: string = 'brownian/poly';
  override needsFull = true;
  override needsHalf = false;
  /** native: no_virial_fdotr_compute = 1; the global virial is tallied pair by pair (ev_tally_xyz, newton off). */
  override virialFdotr = false;
  override explicitVirial = true;

  override initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    // native: "Pair brownian/poly requires newton pair off"
    if (ctx.newtonPair !== false) throw new StyleError('Pair brownian/poly requires newton pair off');
    const s = ctx.s;
    if (!s) return;
    if (!s.radius) throw new StyleError('Pair brownian/poly requires atom attribute radius');
    for (let i = 0; i < s.n; i++) {
      if (s.radius[i] === 0.0) throw new StyleError('Pair brownian/poly requires extended particles');
    }
    this.flagdeform = (ctx.fixOf?.('deform') ?? ctx.fixOf?.('deform/pressure')) ? 1 : 0;
    this.wall = this.findWallView(ctx, 'pair brownian/poly');
    this.flagwall = this.wall ? (wallMoves(this.wall) ? 2 : 1) : 0;
    const prd = [s.box.hi[0] - s.box.lo[0], s.box.hi[1] - s.box.lo[1], s.box.hi[2] - s.box.lo[2]];
    const volT = this.initVolume(prd);
    // vol_P: sum over the particles of their volumes (polydisperse), vol_f = vol_P / vol_T
    let volP = 0.0;
    for (let i = 0; i < s.n; i++) volP += (4.0 / 3.0) * Math.PI * libmPow(s.radius[i], 3.0);
    this.volP = volP;
    const volF = this.flagVF ? this.volP / volT : 0;
    [this.R0, this.RT0] = this.constants(volF);
  }

  /** Isotropic constants of the polydisperse style (no radius factor: native R0 = 6 pi mu (...), RT0 = 8 pi mu (...)). */
  protected override constants(volF: number): [number, number] {
    const mu = this.mu;
    if (this.flaglog === 0) return [6 * Math.PI * mu * (1.0 + 2.16 * volF), 8 * Math.PI * mu];
    return [
      6 * Math.PI * mu * (1.0 + 2.725 * volF - 6.583 * volF * volF),
      8 * Math.PI * mu * (1.0 + 0.749 * volF - 2.469 * volF * volF),
    ];
  }

  override compute(pc: PairCompute): void {
    const s = pc.s, g = pc.geom;
    this.triclinicWarning(g, this.name);
    this.volumeCorrection(g);
    const rng = this.rng0();
    const pre = this.prethermostat(s);
    const radius = s.radius!;
    const order = nativeOrder(s);
    const nlocal = pc.nlocal, nall = pc.nall;
    const x = pc.x, f = pc.f, type = pc.type, owner = pc.nb.owner;
    const cutsq = this.cutsq;
    const nt = this.ntypes + 1;
    const cutInner = this.params().p('cut_inner');
    const { R0, RT0, mu, flaglog, flagfld, flagHI } = this;
    const vxmu = VXMU2F[s.units.style];
    const tq = flaglog && s.torque ? new Float64Array(3 * nall) : null;
    const full = pc.full!;
    const p1 = new Float64Array(3), p2 = new Float64Array(3), p3 = new Float64Array(3);
    const xl = new Float64Array(3);
    for (let k = 0; k < nlocal; k++) {
      const i = order[k];
      const radi = radius[i];
      if (flagfld) {
        f[3 * i] += pre * Math.sqrt(R0 * radi) * (rng.uniform() - 0.5);
        f[3 * i + 1] += pre * Math.sqrt(R0 * radi) * (rng.uniform() - 0.5);
        f[3 * i + 2] += pre * Math.sqrt(R0 * radi) * (rng.uniform() - 0.5);
        if (flaglog) {
          const radi3 = radi * radi * radi;
          tq![3 * i] += pre * Math.sqrt(RT0 * radi3) * (rng.uniform() - 0.5);
          tq![3 * i + 1] += pre * Math.sqrt(RT0 * radi3) * (rng.uniform() - 0.5);
          tq![3 * i + 2] += pre * Math.sqrt(RT0 * radi3) * (rng.uniform() - 0.5);
        }
      }
      if (!flagHI) continue;
      const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
      for (let q = k0; q < k1; q++) {
        const j = full.neighbors[q] & NEIGHMASK;
        const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = type[i] * nt + type[j];
        if (!(rsq < cutsq[t])) continue;
        const radj = radius[owner[j]];
        const r = Math.sqrt(rsq);
        // scalar resistances a_sq and a_sh
        let hSep = r - radi - radj;
        // if less than minimum gap, use minimum gap instead
        if (r < cutInner[t]) hSep = cutInner[t] - radi - radj;
        // scale h_sep by radi
        hSep = hSep / radi;
        const beta0 = radj / radi;
        const beta1 = 1.0 + beta0;
        let aSq: number, aSh = 0, aPu = 0;
        if (flaglog) {
          aSq = beta0 * beta0 / beta1 / beta1 / hSep +
            (1.0 + 7.0 * beta0 + beta0 * beta0) / 5.0 / cube(beta1) * libmLog(1.0 / hSep);
          aSq += (1.0 + 18.0 * beta0 - 29.0 * beta0 * beta0 + 18.0 * cube(beta0) + powint(beta0, 4)) / 21.0 /
            powint(beta1, 4) * hSep * libmLog(1.0 / hSep);
          aSq *= 6.0 * Math.PI * mu * radi;
          aSh = 4.0 * beta0 * (2.0 + beta0 + 2.0 * beta0 * beta0) / 15.0 / cube(beta1) * libmLog(1.0 / hSep);
          aSh += 4.0 * (16.0 - 45.0 * beta0 + 58.0 * beta0 * beta0 - 45.0 * cube(beta0) + 16.0 * powint(beta0, 4)) /
            375.0 / powint(beta1, 4) * hSep * libmLog(1.0 / hSep);
          aSh *= 6.0 * Math.PI * mu * radi;
          aPu = beta0 * (4.0 + beta0) / 10.0 / beta1 / beta1 * libmLog(1.0 / hSep);
          aPu += (32.0 - 33.0 * beta0 + 83.0 * beta0 * beta0 + 43.0 * cube(beta0)) / 250.0 / cube(beta1) * hSep * libmLog(1.0 / hSep);
          aPu *= 8.0 * Math.PI * mu * cube(radi);
        } else {
          aSq = 6.0 * Math.PI * mu * radi * (beta0 * beta0 / beta1 / beta1 / hSep);
        }
        // generate the Pairwise Brownian Force: a_sq
        let Fbmag = pre * Math.sqrt(aSq);
        let randr = rng.uniform() - 0.5;
        // contribution due to Brownian motion
        let fx = Fbmag * randr * delx / r;
        let fy = Fbmag * randr * dely / r;
        let fz = Fbmag * randr * delz / r;
        // add terms due to a_sh
        if (flaglog) {
          p1[0] = delx / r;
          p1[1] = dely / r;
          p1[2] = delz / r;
          orthogonalVectors(p1, p2, p3);
          Fbmag = pre * Math.sqrt(aSh);
          randr = rng.uniform() - 0.5;
          fx += Fbmag * randr * p2[0];
          fy += Fbmag * randr * p2[1];
          fz += Fbmag * randr * p2[2];
          randr = rng.uniform() - 0.5;
          fx += Fbmag * randr * p3[0];
          fy += Fbmag * randr * p3[1];
          fz += Fbmag * randr * p3[2];
        }
        // scale forces to appropriate units
        fx = vxmu * fx;
        fy = vxmu * fy;
        fz = vxmu * fz;
        // sum to total force (only atom i: newton off)
        f[3 * i] -= fx;
        f[3 * i + 1] -= fy;
        f[3 * i + 2] -= fz;
        // torque due to the Brownian Force
        if (flaglog) {
          xl[0] = -delx / r * radi;
          xl[1] = -dely / r * radi;
          xl[2] = -delz / r * radi;
          // torque = xl_cross_F
          let tx = xl[1] * fz - xl[2] * fy;
          let ty = xl[2] * fx - xl[0] * fz;
          let tz = xl[0] * fy - xl[1] * fx;
          // torque is same on both particles
          tq![3 * i] -= tx;
          tq![3 * i + 1] -= ty;
          tq![3 * i + 2] -= tz;
          // torque due to a_pu
          Fbmag = pre * Math.sqrt(aPu);
          randr = rng.uniform() - 0.5;
          tx = Fbmag * randr * p2[0];
          ty = Fbmag * randr * p2[1];
          tz = Fbmag * randr * p2[2];
          randr = rng.uniform() - 0.5;
          tx += Fbmag * randr * p3[0];
          ty += Fbmag * randr * p3[1];
          tz += Fbmag * randr * p3[2];
          // torque has opposite sign on two particles
          tq![3 * i] -= tx;
          tq![3 * i + 1] -= ty;
          tq![3 * i + 2] -= tz;
        }
        // native: ev_tally_xyz(i, nlocal, nlocal, 0, ...): i takes half of the virial of this visit
        const fxn = -fx, fyn = -fy, fzn = -fz;
        const w = [delx * fxn, dely * fyn, delz * fzn, delx * fyn, delx * fzn, dely * fzn];
        for (let c = 0; c < 6; c++) pc.acc.virial[c] += 0.5 * w[c];
        if (pc.vatom) tallyVatom(pc, i, j, false, fxn, fyn, fzn, delx, dely, delz);
      }
    }
    if (tq) pc.nb.reverseSum(tq, 3, s.torque!);
  }
}
