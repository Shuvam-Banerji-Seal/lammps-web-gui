/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/pair_multi_lucy_rx.cpp and
 * pair_multi_lucy_rx.h (PairMultiLucyRX: pair_style multi/lucy/rx, the density-dependent pair force and the per-atom
 * density energy of a reacting DPD particle, weighted by the species concentrations of fix rx). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Contributing authors (from pair_multi_lucy_rx.cpp): James Larentzos and Joshua Moore
 * (U.S. Army Research Laboratory). Please cite J. D. Moore et al., J. Chem. Phys. 144, 104501 (2016), as the native file does.
 *
 * docs.lammps.org/pair_multi_lucy_rx.html: "pair_style multi/lucy/rx style N keyword ..." with "style = lookup or linear" and
 * "weighting = fractional or molecular (optional)"; "pair_coeff I J filename keyword species1 species2 cutoff". The local
 * densities and the table are those of pair multi/lucy (src/engine-gpl/multi_lucy/pair_multi_lucy.ts, reused here); the
 * table file is read with the rx table reader (rx_table_file_reader.cpp), as native does. The site weights are those of
 * pair table/rx (rx_mixing.ts). The per-atom energy is the density energy of the atom's own density times the site weight:
 * uCG (old concentrations) and uCGnew (new) for fix eos/table/rx and fix rx.
 *
 * Refused: the spline style, and per-atom energy or virial (the browser engine does not tally them for this style).
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { DPD_UCG, DPD_UCGNEW } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { PairMultiLucy, lucyLocalDensity, newTable, type Table } from '../multi_lucy/pair_multi_lucy';
import { FixRX } from './fix_rx';
import { RxTableReader, Tokens } from './rx_table_reader';
import { MIX_WEIGHTS, mixingWeights } from './rx_mixing';

export class PairMultiLucyRX extends PairMultiLucy {
  readonly name: string = 'multi/lucy/rx';
  private rx: FixRX | null = null;
  private isite1 = -1;
  private isite2 = -1;
  /** pair_multi_lucy_rx.cpp: fractional weighting by default. */
  private fractional = true;
  /** Per owned atom, MIX_WEIGHTS mixing weights (see rx_mixing.ts). */
  private W = new Float64Array(0);

  settings(args: string[]): void {
    // As native LAMMPS does (src/DPD-REACT/pair_multi_lucy_rx.cpp PairMultiLucyRX::settings at b332f28): the table
    // style and length, then the keywords
    if (args.length < 2) throw new StyleError('Illegal pair_style command');
    this.tableSettings([args[0], args[1]]);
    this.fractional = true;
    for (const w of args.slice(2)) {
      if (w === 'fractional') this.fractional = true;
      else if (w === 'molecular') this.fractional = false;
      else throw new StyleError('Illegal pair_style command');
    }
  }

  /** pair_multi_lucy_rx.cpp coeff: "I J filename keyword species1 species2 [cutoff]". */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 6 && args.length !== 7) throw new StyleError('Incorrect args for pair coefficients');
    const fx = ctx.fixOf?.('rx');
    if (!(fx instanceof FixRX)) throw new StyleError('Fix rx not defined or not compatible with pair style');
    this.rx = fx;
    const isite1 = this.siteOf(args[4], 'Site1');
    const isite2 = this.siteOf(args[5], 'Site2');
    super.coeff([args[0], args[1], args[2], args[3], ...(args.length === 7 ? [args[6]] : [])], ctx);
    this.isite1 = isite1;
    this.isite2 = isite2;
  }

  /** The species index of a site word, or -1 for "1fluid". */
  private siteOf(word: string, which: string): number {
    if (word === '1fluid') return -1;
    const k = this.rx!.speciesIndexOf(word);
    if (k === undefined) throw new StyleError(`Pair_multi_lucy_rx ${which.toLowerCase()} is invalid.`);
    return k;
  }

  initStyle(ctx: StyleContext): void {
    // pair_multi_lucy_rx.cpp: "Pair multi/lucy/rx command requires atom_style with density (e.g. dpd, meso)", and the
    // uCG and uCGnew energies of the reactions
    if (!ctx.s || !ctx.s.custom.has(DPD_UCG) || !ctx.s.custom.has(DPD_UCGNEW)) {
      throw new StyleError('pair multi/lucy/rx requires atom_style with density and internal energies (e.g. dpd)');
    }
    super.initStyle(ctx);
  }

  /** read_table of pair_multi_lucy_rx.cpp (rx table reader: the N entries, then param_extract, index r E F per line). */
  protected readTable(text: string, keyword: string, file: string): Table {
    const reader = new RxTableReader(text, keyword, file, 'multi/lucy/rx');
    const tb = newTable();
    tb.ninput = reader.N;
    tb.rflag = 0;
    tb.fpflag = 0;
    const p = reader.params;
    for (let k = 0; k < p.length; k++) {
      const w = p[k];
      if (w[0] === 'R') {
        if (w === 'R') tb.rflag = 1;
        else if (w === 'RSQ') tb.rflag = 2;
        else throw new StyleError(`Invalid keyword in pair table parameters: ${w}`);
        tb.rlo = parseNum(p[++k], 'R lo');
        tb.rhi = parseNum(p[++k], 'R hi');
      } else if (w === 'FP') {
        tb.fpflag = 1;
        tb.fplo = parseNum(p[++k], 'FP lo');
        tb.fphi = parseNum(p[++k], 'FP hi');
      } else {
        throw new StyleError(`Invalid keyword in pair table parameters: ${w}`);
      }
    }
    const n = tb.ninput;
    tb.rfile = new Float64Array(n);
    tb.efile = new Float64Array(n);
    tb.ffile = new Float64Array(n);
    reader.readTable((i: number, values: Tokens) => {
      values.int(); // the index is ignored
      let r = values.double();
      tb.efile[i] = values.double();
      tb.ffile[i] = values.double();
      if (tb.rflag === 1) r = tb.rlo + ((tb.rhi - tb.rlo) * i) / (n - 1);
      else if (tb.rflag === 2) {
        const rsq = tb.rlo * tb.rlo + ((tb.rhi * tb.rhi - tb.rlo * tb.rlo) * i) / (n - 1);
        r = Math.sqrt(rsq);
      }
      tb.rfile[i] = r;
    });
    return tb;
  }

  /**
   * compute of pair_multi_lucy_rx.cpp. The site weights of each owned atom come first; the local densities are those of
   * pair multi/lucy (a ghost's density is its owner's); each pair inside the cutoff adds the density-dependent force
   * weighted by the old-concentration site weights; each owned atom adds its density energy to uCG (old) and uCGnew (new).
   */
  compute(pc: PairCompute): void {
    const rx = this.rx;
    if (!rx) throw new StyleError('pair multi/lucy/rx used before pair_coeff');
    if (pc.eatom || pc.vatom) throw new StyleError('pair multi/lucy/rx: per-atom energy and virial are not supported by the browser engine');
    const s = pc.s;
    const cur = rx.speciesData(s);
    const old = rx.speciesNames.map((n) => s.custom.get(`${n}Old`)!.data);
    const uCG = s.custom.get(DPD_UCG)!.data, uCGnew = s.custom.get(DPD_UCGNEW)!.data;
    const nl = pc.nlocal;
    if (this.W.length !== MIX_WEIGHTS * nl) this.W = new Float64Array(MIX_WEIGHTS * nl);
    const W = this.W;
    for (let i = 0; i < nl; i++) mixingWeights(i, cur, old, this.isite1, this.isite2, this.fractional, W, MIX_WEIGHTS * i);

    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const rho = lucyLocalDensity(pc, cutsq, nt);
    const owner = pc.nb.owner;
    const rhoOf = (k: number): number => rho[owner[k]];
    const list = pc.half!;
    const { x, f, type } = pc;
    const nbrs = list.neighbors;
    const tlm1 = this.tablength - 1;
    const lookup = this.tabstyle === 'lookup';
    const same = this.isite1 === this.isite2;
    let evAcc = 0;
    for (let i = 0; i < list.inum; i++) {
      const itype = type[i];
      const bi = MIX_WEIGHTS * i;
      const w1oi = W[bi], w2oi = W[bi + 1];
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const jtype = type[j];
        const t = itype * nt + jtype;
        if (!(rsq < cutsq[t])) continue;
        const tb = this.tables[this.tabindex[t]];
        const rhoI2 = rhoOf(i) * rhoOf(i), rhoJ2 = rhoOf(j) * rhoOf(j);
        if (rhoI2 < tb.innersq || rhoJ2 < tb.innersq) throw new StyleError('Density < table inner cutoff');
        const itable = Math.trunc((rhoI2 - tb.innersq) * tb.invdelta);
        const jtable = Math.trunc((rhoJ2 - tb.innersq) * tb.invdelta);
        if (itable >= tlm1 || jtable >= tlm1) throw new StyleError('Density > table outer cutoff');
        let Ai: number, Aj: number;
        if (lookup) {
          Ai = tb.f[itable];
          Aj = tb.f[jtable];
        } else {
          // the linear fraction is zero in the first bin (native fraction_i = 0 when itable == 0)
          const fi = itable === 0 ? 0 : (rhoI2 - tb.rsq[itable]) * tb.invdelta;
          const fj = jtable === 0 ? 0 : (rhoJ2 - tb.rsq[jtable]) * tb.invdelta;
          Ai = tb.f[itable] + fi * tb.df[itable];
          Aj = tb.f[jtable] + fj * tb.df[jtable];
        }
        const r = Math.sqrt(rsq);
        const rfactor = 1.0 - Math.sqrt(rsq / cutsq[t]);
        let fpair = 0.5 * (Ai + Aj) * (4.0 - 3.0 * rfactor) * rfactor * rfactor * rfactor;
        fpair /= r;
        const bj = MIX_WEIGHTS * owner[j];
        const w1oj = W[bj], w2oj = W[bj + 1];
        if (same) fpair = Math.sqrt(w1oi * w2oj) * fpair;
        else fpair = (Math.sqrt(w1oi * w2oj) + Math.sqrt(w2oi * w1oj)) * fpair;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    // the density energy of each owned atom: its own density in its self table, times pi rc^4 / 84
    for (let i = 0; i < list.inum; i++) {
      const itype = type[i];
      const t = itype * nt + itype;
      const tb = this.tables[this.tabindex[t]];
      const rho2 = rhoOf(i) * rhoOf(i);
      if (rho2 < tb.innersq) throw new StyleError('Density < table inner cutoff');
      const itable = Math.trunc((rho2 - tb.innersq) * tb.invdelta);
      if (itable >= tlm1) throw new StyleError('Density > table outer cutoff');
      let evdwl: number;
      if (lookup) evdwl = tb.e[itable];
      else {
        const fraction = itable === 0 ? 0 : (rho2 - tb.rsq[itable]) * tb.invdelta;
        evdwl = tb.e[itable] + fraction * tb.de[itable];
      }
      const c = cutsq[t];
      evdwl *= (Math.PI * c * c) / 84.0;
      const bi = MIX_WEIGHTS * i;
      const evdwlOld = W[bi] * evdwl;
      const evdwlNew = W[bi + 2] * evdwl;
      uCG[i] += evdwlOld;
      uCGnew[i] += evdwlNew;
      evAcc += evdwlOld;
    }
    pc.acc.evdwl += evAcc;
  }
}
