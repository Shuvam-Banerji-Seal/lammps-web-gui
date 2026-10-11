/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/pair_bpm_peri.cpp and src/BPM/pair_bpm_peri.h at commit b332f28 (the short-range contact between
 * non-bonded nodes that keeps peridynamic nodes from interpenetrating). Authors credited in the BPM sources: Joel
 * Clemmer (SNL). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * pair_style bpm/peri: docs.lammps.org/pair_bpm_peri.html. The doc says the pair style "computes a short-range
 * contact force between peridynamic nodes" that acts within 1.35 lattice spacings. The force is the linear spring
 * -15 K vfrac_eff (r - d_ij) / horizon, with the nodal volumes vfrac of the two nodes (the custom per-atom
 * property d_vfrac). Pairs with a special-bond weight of 0 (bonded partners) are skipped, as in the C++ loop.
 * Browser-engine difference: the nodal volume is read from the custom property of the owned atom.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum, fmtCoeff } from '../../engine/force/util';

export class PairBpmPeri extends Pair {
  readonly name = 'bpm/peri';
  virialFdotr = true;
  p!: PairParams;
  private kk = new Float64Array(0);
  private cutK = new Float64Array(0);
  private dij = 0;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length) throw new StyleError(`illegal pair_style bpm/peri argument '${args[0]}'`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['k', 'cut']);
  }

  /** pair_coeff I J kspring horizon (pair_bpm_peri.cpp coeff: the horizon is the cutoff). */
  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length !== 4) throw new StyleError('usage: pair_coeff I J kspring horizon');
    const k = parseNum(args[2], 'kspring'), cut = parseNum(args[3], 'horizon');
    if (!(cut > 0)) throw new StyleError(`Invalid horizon value ${cut} for pair style bpm/peri`);
    this.p.setRange(args[0], args[1], [k, cut]);
  }

  initStyle(ctx: StyleContext): void {
    if (this.tail) throw new StyleError('pair_modify tail is not supported by pair style bpm/peri');
    const lc = ctx.xlattice ?? 1;
    this.dij = 1.35 * lc; // d_ij = 1.35 lattice spacing (pair_bpm_peri.cpp compute)
    const nt = this.ntypes + 1;
    this.kk = new Float64Array(nt * nt);
    this.cutK = new Float64Array(nt * nt);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError('All pair_coeff for pair style bpm/peri must be set explicitly');
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.kk[k1] = this.kk[k2] = this.p.get('k', i, j);
    this.cutK[k1] = this.cutK[k2] = this.p.get('cut', i, j);
    return this.p.get('cut', i, j);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nt = this.ntypes + 1;
    const { cutsq, kk, cutK } = this;
    const sLJ = pc.specialLJ;
    const vf = pc.s.custom.get('vfrac');
    if (!vf) throw new StyleError('Pair bpm/peri requires a per-atom vfrac property (fix property/atom d_vfrac)');
    const vfrac = vf.data;
    const dij = this.dij;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const oi = owner[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        // bonded partners with special lj weight 0 are skipped (special_lj[sbmask(j)] == 0)
        if (sLJ[jj >>> SBBITS] === 0) continue;
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        if (r >= dij) continue;
        const dr = r - dij;
        const oj = owner[j];
        const vEff = 0.5 * (vfrac[oi] + vfrac[oj]);
        const rk = 15 * kk[t] * vEff * (dr / cutK[t]);
        const fpair = r > 0 ? -(rk / r) : 0;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        evdwl += 0.5 * rk * dr;
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('k', i, i))} ${fmtCoeff(this.p.get('cut', i, i))}`);
    return out;
  }
}
