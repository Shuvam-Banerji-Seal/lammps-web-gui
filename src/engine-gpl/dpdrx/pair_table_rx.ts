/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/pair_table_rx.cpp and
 * pair_table_rx.h (PairTableRX: pair_style table/rx, a tabulated pair potential of a reacting DPD particle, weighted by the
 * species concentrations of fix rx). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author (from
 * pair_table_rx.cpp): Paul Crozier (SNL).
 *
 * docs.lammps.org/pair_table_rx.html: "pair_style table/rx style N" with "style = lookup or linear or spline or bitmap" and
 * "weighting = fractional or molecular (optional)"; "pair_coeff I J filename keyword species1 species2 cutoff". The table
 * construction and the file format are those of pair table (core src/engine/force/pair/table.ts, reused here); the
 * per-pair energy and force are the table value times the square roots of the site weights, as in pair_table_rx.cpp
 * compute. The per-atom energies uCG (old concentrations) and uCGnew (new) are accumulated for fix eos/table/rx and fix rx.
 *
 * Refused by name: the bitmap style and the ewald, pppm, msm, dispersion and tip4p keywords (the core table refuses them).
 * The per-pair single() is not available (its weights need the owned atoms' mixing weights of this step).
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { PairTable, readSection } from '../../engine/force/pair/table';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { DPD_UCG, DPD_UCGNEW } from '../../engine/atoms';
import { tallyXYZ } from '../dpd/pair_dpd';
import { FixRX } from './fix_rx';
import { MIX_WEIGHTS, mixingWeights } from './rx_mixing';
import { NativePairOrder } from '../neighbor/nativeOrder';

/** pair_table_rx.cpp settings: the keywords the native style accepts that the browser engine refuses by name. */
const LONG_RANGE_KEYWORDS = ['ewald', 'pppm', 'msm', 'dispersion', 'tip4p'];

export class PairTableRX extends PairTable {
  readonly name: string = 'table/rx';
  /** pair_table_rx.cpp: "fractionalWeighting = true" by default (the fractional keyword), molecular otherwise. */
  private fractional = true;
  private rx: FixRX | null = null;
  /** Site species indices (pair_coeff species1, species2); -1 is the one-fluid site "1fluid". */
  private isite1 = -1;
  private isite2 = -1;
  /** Per owned atom, MIX_WEIGHTS mixing weights (see rx_mixing.ts). */
  private W = new Float64Array(0);
  private readonly order = new NativePairOrder();

  settings(args: string[]): void {
    if (args.length < 2) throw new StyleError('Illegal pair_style command');
    const style = args[0];
    if (style === 'bitmap') {
      throw new StyleError("pair_style table/rx interpolation style 'bitmap' is not supported (lookup, linear and spline only)");
    }
    if (style !== 'lookup' && style !== 'linear' && style !== 'spline') {
      throw new StyleError('Unknown table style in pair_style command');
    }
    this.fractional = true;
    for (const w of args.slice(2)) {
      if (w === 'fractional') this.fractional = true;
      else if (w === 'molecular') this.fractional = false;
      else if (LONG_RANGE_KEYWORDS.includes(w)) {
        throw new StyleError(`pair_style table/rx keyword '${w}' is not supported (long-range solver coupling is not implemented)`);
      } else throw new StyleError('Illegal pair_style command');
    }
    super.settings([args[0], args[1]]);
  }

  /** pair_table_rx.cpp coeff: "I J filename keyword species1 species2 [cutoff]" (the table part is the pair table's). */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 6 && args.length !== 7) throw new StyleError('Incorrect args for pair coefficients');
    const fx = ctx.fixOf?.('rx');
    if (!(fx instanceof FixRX)) throw new StyleError('Fix rx not defined or not compatible with pair style');
    this.rx = fx;
    if (fx.nspecies === 0) throw new StyleError('There are no rx species specified.');
    const isite1 = this.siteOf(args[4], 'Site1');
    const isite2 = this.siteOf(args[5], 'Site2');
    // "Invalid pair table cutoff" when the table starts at r <= 0 (pair_table_rx.cpp coeff, rlo <= 0)
    const sec = readSection(ctx.readFile(args[2]), args[2], args[3]);
    const rlo = sec.spacing === 'none' ? sec.r[0] : sec.rlo;
    if (rlo <= 0.0) throw new StyleError('Invalid pair table cutoff');
    super.coeff([args[0], args[1], args[2], args[3], ...(args.length === 7 ? [args[6]] : [])], ctx);
    this.isite1 = isite1;
    this.isite2 = isite2;
  }

  /** The species index of a site word, or -1 for "1fluid". */
  private siteOf(word: string, which: string): number {
    if (word === '1fluid') return -1;
    const k = this.rx!.speciesIndexOf(word);
    if (k === undefined) throw new StyleError(`${which} name not recognized in pair coefficients`);
    return k;
  }

  initStyle(ctx: StyleContext): void {
    // pair_table_rx.cpp reads atom->uCG and atom->uCGnew: the atom style must have them (atom_style dpd)
    if (!ctx.s || !ctx.s.custom.has(DPD_UCG) || !ctx.s.custom.has(DPD_UCGNEW)) {
      throw new StyleError('pair_style table/rx requires atom_style with internal temperature and energies (e.g. dpd)');
    }
  }

  single(): { eng: number; fforce: number } {
    throw new StyleError('pair_style table/rx: the per-pair energy and force (single) are not available');
  }

  /**
   * pair_table_rx.cpp compute. The mixing weights of each owned atom are computed first (getMixingWeights); a ghost
   * atom uses its owner's weights, which is what native's forward communication gives it. A pair inside the cutoff adds
   * the mixed force and the old-concentration energy (times 1/2 to each atom) to uCG, and the new-concentration energy to
   * uCGnew; the owned atom's sums are added after its neighbor loop, as native does.
   */
  compute(pc: PairCompute): void {
    const rx = this.rx;
    if (!rx) throw new StyleError('pair table/rx used before pair_coeff');
    const s = pc.s;
    const cur = rx.speciesData(s);
    const old = rx.speciesNames.map((n) => s.custom.get(`${n}Old`)!.data);
    const uCG = s.custom.get(DPD_UCG)!.data, uCGnew = s.custom.get(DPD_UCGNEW)!.data;
    const nl = pc.nlocal;
    if (this.W.length !== MIX_WEIGHTS * nl) this.W = new Float64Array(MIX_WEIGHTS * nl);
    const W = this.W;
    for (let i = 0; i < nl; i++) mixingWeights(i, cur, old, this.isite1, this.isite2, this.fractional, W, MIX_WEIGHTS * i);

    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sLJ = pc.specialLJ;
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    const tally = pc.eatom !== null || pc.vatom !== null;
    const { tRsq1, tInvDelta, tE, tG, tE2, tG2 } = this;
    const mode = this.mode;
    const same = this.isite1 === this.isite2;
    let evAcc = 0;
    // the owned atom whose neighbor loop is open: its force and uCG sums are added when the loop ends (native fx_i, uCG_i)
    let curI = -1;
    let xi = 0, yi = 0, zi = 0, ti = 0, bi = 0, w1oi = 0, w2oi = 0, w1i = 0, w2i = 0;
    let fxi = 0, fyi = 0, fzi = 0, uCGi = 0, uCGnewi = 0;
    const flush = (): void => {
      if (curI < 0) return;
      uCG[owner[curI]] += uCGi;
      uCGnew[owner[curI]] += uCGnewi;
      f[3 * curI] += fxi; f[3 * curI + 1] += fyi; f[3 * curI + 2] += fzi;
    };
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const jj = pairs[p + 1];
      if (i !== curI) {
        flush();
        curI = i;
        xi = x[3 * i]; yi = x[3 * i + 1]; zi = x[3 * i + 2];
        ti = type[i] * nt;
        bi = MIX_WEIGHTS * i;
        w1oi = W[bi]; w2oi = W[bi + 1]; w1i = W[bi + 2]; w2i = W[bi + 3];
        fxi = 0; fyi = 0; fzi = 0; uCGi = 0; uCGnewi = 0;
      }
      const j = jj & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const t = ti + type[j];
      if (rsq >= cutsq[t]) continue;
      const factor = sLJ[jj >>> SBBITS];
      const eT = tE[t]!, gT = tG[t]!;
      const rsq1 = tRsq1[t];
      if (rsq < rsq1) throw new StyleError('Pair distance < table inner cutoff');
      const pos = (rsq - rsq1) * tInvDelta[t];
      const last = eT.length - 1;
      const itable = Math.trunc(pos);
      if (itable >= last) throw new StyleError('Pair distance > table outer cutoff');
      let e: number, value: number;
      if (mode === 'lookup') {
        // "the distance R is used to find the nearest table entry"
        e = eT[itable];
        value = gT[itable];
      } else if (mode === 'linear') {
        const fraction = pos - itable;
        e = eT[itable] + fraction * (eT[itable + 1] - eT[itable]);
        value = gT[itable] + fraction * (gT[itable + 1] - gT[itable]);
      } else {
        const b = pos - itable, a = 1 - b;
        const h2_6 = 1 / (6 * tInvDelta[t] * tInvDelta[t]);
        const e2 = tE2[t]!, g2 = tG2[t]!;
        e = a * eT[itable] + b * eT[itable + 1] + ((a * a * a - a) * e2[itable] + (b * b * b - b) * e2[itable + 1]) * h2_6;
        value = a * gT[itable] + b * gT[itable + 1] + ((a * a * a - a) * g2[itable] + (b * b * b - b) * g2[itable + 1]) * h2_6;
      }
      const bj = MIX_WEIGHTS * owner[j];
      const w1oj = W[bj], w2oj = W[bj + 1], w1j = W[bj + 2], w2j = W[bj + 3];
      let fpair = factor * value;
      // the force weights: the old concentrations (native mixWtSite*old)
      if (same) fpair = Math.sqrt(w1oi * w2oj) * fpair;
      else fpair = (Math.sqrt(w1oi * w2oj) + Math.sqrt(w2oi * w1oj)) * fpair;
      const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
      fxi += fx; fyi += fy; fzi += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      // the energy: old concentrations are tallied, new ones go to uCGnew
      let evdwlOld: number, evdwl: number;
      if (same) {
        evdwlOld = Math.sqrt(w1oi * w2oj) * e;
        evdwl = Math.sqrt(w1i * w2j) * e;
      } else {
        evdwlOld = (Math.sqrt(w1oi * w2oj) + Math.sqrt(w2oi * w1oj)) * e;
        evdwl = (Math.sqrt(w1i * w2j) + Math.sqrt(w2i * w1j)) * e;
      }
      evdwlOld *= factor;
      evdwl *= factor;
      uCGi += 0.5 * evdwlOld;
      uCG[owner[j]] += 0.5 * evdwlOld;
      uCGnewi += 0.5 * evdwl;
      uCGnew[owner[j]] += 0.5 * evdwl;
      evAcc += evdwlOld;
      if (tally) tallyXYZ(pc, i, j, evdwlOld, fx, fy, fz, dx, dy, dz);
    }
    flush();
    pc.acc.evdwl += evAcc;
  }
}
