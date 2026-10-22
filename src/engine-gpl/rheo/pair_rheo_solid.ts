/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/RHEO/pair_rheo_solid.cpp and pair_rheo_solid.h (PairRHEOSolid: pair_style rheo/solid), at commit b332f28.
 * RHEO package authors: Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU). Copyright Sandia Corporation (GNU
 * General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/pair_rheo_solid.html): "Style rheo/solid is effectively a copy of pair style bpm/spring except it
 * only applies forces between solid RHEO particles, determined by checking the status of each pair of neighboring
 * particles before calculating forces." "F = k (r - r_c)" and "F_D = - gamma w (r-hat . v)" with "w = 1.0 - (r/r_c)^8".
 * "pair_coeff * * 1.0 1.5 1.0" gives k, r_c and gamma per type pair.
 *
 * The native code loops over the half list with a newton flag. The port visits the owned pairs once (half list) and the
 * ghost pairs from the full list, each owned atom acting on itself only (the image pair acts on its own owned atom),
 * which is the newton-off result for either setting. The virial is tallied per pair as native ev_tally does: an owned
 * pair takes the full product, a ghost pair half of it. Special bonds with a zero factor are skipped (native
 * "if (factor_lj == 0) continue").
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { RHEO_STATUS } from '../../engine/atoms';
import { STATUS_SOLID, rheoProp } from './model';

export class PairRHEOSolid extends Pair {
  readonly name: string = 'rheo/solid';
  virialFdotr = false;
  explicitVirial = true;
  needsFull = true;
  private k = new Float64Array(0);
  private cutP = new Float64Array(0);
  private gamma = new Float64Array(0);
  private cutsqP = new Float64Array(0);
  private set = new Uint8Array(0);

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style rheo/solid command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = (ntypes + 1) * (ntypes + 1);
    this.k = new Float64Array(nt);
    this.cutP = new Float64Array(nt);
    this.gamma = new Float64Array(nt);
    this.cutsqP = new Float64Array(nt);
    this.set = new Uint8Array(nt);
  }

  coeff(args: string[]): void {
    if (args.length !== 5) throw new StyleError('Incorrect args for pair coefficients');
    const n = this.ntypes;
    const [ilo, ihi] = typeBounds(args[0], n);
    const [jlo, jhi] = typeBounds(args[1], n);
    const kOne = parseNum(args[2], 'k');
    const cutOne = parseNum(args[3], 'cutoff');
    const gammaOne = parseNum(args[4], 'gamma');
    // pair_rheo_solid.cpp: "if (cut_one <= 0.0) error->all(...Incorrect args for pair coefficients)"
    if (cutOne <= 0.0) throw new StyleError('Incorrect args for pair coefficients');
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const q = i * (n + 1) + j;
        this.k[q] = kOne;
        this.cutP[q] = cutOne;
        this.gamma[q] = gammaOne;
        this.set[q] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    if (ctx.ghostVelocity !== true) throw new StyleError('Pair rheo/solid requires ghost atoms store velocity\n(use comm_modify vel yes)');
  }

  initOne(i: number, j: number): number {
    const n = this.ntypes;
    const q = i * (n + 1) + j, qi = i * (n + 1) + i, qj = j * (n + 1) + j;
    if (!this.set[q]) {
      // pair_rheo_solid.cpp init_one(): mix_distance and mix_energy with the default (geometric) mixing rule
      const ci = this.cutP[qi], cj = this.cutP[qj];
      this.cutP[q] = Math.sqrt(ci * cj);
      this.k[q] = Math.sqrt(this.k[qi] * this.k[qj]);
      this.gamma[q] = Math.sqrt(this.gamma[qi] * this.gamma[qj]);
    }
    this.cutsqP[q] = this.cutP[q] * this.cutP[q];
    this.cutsqP[j * (n + 1) + i] = this.cutsqP[q];
    this.cutP[j * (n + 1) + i] = this.cutP[q];
    this.k[j * (n + 1) + i] = this.k[q];
    this.gamma[j * (n + 1) + i] = this.gamma[q];
    return this.cutP[q];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const nlocal = pc.nlocal;
    const x = pc.x, f = pc.f, type = pc.type, owner = pc.nb.owner;
    const v = s.v;
    const status = rheoProp(s, RHEO_STATUS);
    const sLJ = pc.specialLJ;
    const n = this.ntypes;
    const vir = pc.acc.virial;
    const half = pc.half, full = pc.full;
    if (!half || !full) throw new StyleError('pair rheo/solid needs half and full neighbor lists');
    const pairVisit = (i: number, jj: number, owned: boolean): void => {
      const j = jj & NEIGHMASK;
      const factor = sLJ[jj >>> SBBITS];
      if (factor === 0) return;
      const oj = owner[j];
      if (!(status[oj] & STATUS_SOLID)) return;
      const q = type[i] * (n + 1) + type[j];
      const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (!(rsq < this.cutsqP[q])) return;
      const r = Math.sqrt(rsq);
      const rinv = 1.0 / r;
      let fpair = this.k[q] * (this.cutP[q] - r);
      let smooth = rsq / this.cutsqP[q];
      smooth *= smooth;
      smooth *= smooth;
      smooth = 1.0 - smooth;
      const dvx = v[3 * i] - v[3 * oj], dvy = v[3 * i + 1] - v[3 * oj + 1], dvz = v[3 * i + 2] - v[3 * oj + 2];
      const dot = dx * dvx + dy * dvy + dz * dvz;
      fpair -= this.gamma[q] * dot * smooth * rinv;
      fpair *= factor * rinv;
      const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
      f[3 * i] += fx;
      f[3 * i + 1] += fy;
      f[3 * i + 2] += fz;
      if (owned) {
        f[3 * j] -= fx;
        f[3 * j + 1] -= fy;
        f[3 * j + 2] -= fz;
      }
      // ev_tally with newton off: an owned end takes half of the product (an owned pair both halves)
      const w = owned ? 1.0 : 0.5;
      vir[0] += w * dx * fx;
      vir[1] += w * dy * fy;
      vir[2] += w * dz * fz;
      vir[3] += w * dx * fy;
      vir[4] += w * dx * fz;
      vir[5] += w * dy * fz;
      // per-atom: half of the product to the owned i, and to j when it is owned (ev_tally, newton off)
      const va = pc.vatom;
      if (va) {
        const p6 = [0.5 * dx * fx, 0.5 * dy * fy, 0.5 * dz * fz, 0.5 * dx * fy, 0.5 * dx * fz, 0.5 * dy * fz];
        for (let c = 0; c < 6; c++) va[6 * i + c] += p6[c];
        if (owned) for (let c = 0; c < 6; c++) va[6 * j + c] += p6[c];
      }
    };
    for (let i = 0; i < nlocal; i++) {
      if (!(status[i] & STATUS_SOLID)) continue;
      for (let k = half.firstneigh[i], k1 = k + half.numneigh[i]; k < k1; k++) {
        const jj = half.neighbors[k];
        if ((jj & NEIGHMASK) < nlocal) pairVisit(i, jj, true);
      }
      for (let k = full.firstneigh[i], k1 = k + full.numneigh[i]; k < k1; k++) {
        const jj = full.neighbors[k];
        if ((jj & NEIGHMASK) >= nlocal) pairVisit(i, jj, false);
      }
    }
  }
}
