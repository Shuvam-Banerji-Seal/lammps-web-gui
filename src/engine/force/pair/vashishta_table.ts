import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseInt_, parseNum } from '../util';
import { PairVashishta } from './vashishta';

/*
 * pair_style vashishta/table Ntable cutinner — docs.lammps.org/pair_vashishta.html
 * (source: plans/lammps-docs/pair_vashishta.rst)
 *
 * Quoted sentences are copied from the .rst with its reST markup (:math:,
 * :doc:, *emphasis*) removed and line breaks joined.
 *
 * Syntax (verbatim):
 *
 *   *vashishta/table* args = Ntable cutinner
 *     Ntable = # of tabulation points
 *     cutinner = tablulate from cutinner to cutoff
 *
 * "The *vashishta* style computes these formulas analytically.  The
 * *vashishta/table* style tabulates the analytic values for *Ntable* points
 * from cutinner to the cutoff of the potential.  The points are equally
 * spaced in R^2 space from cutinner^2 to cutoff^2.  For the two-body term in
 * the above equation, a linear interpolation for each pairwise distance
 * between adjacent points in the table."
 *
 * The table covers the two-body term U2 only; the three-body term is the
 * analytic one (inherited from vashishta.ts).  Measured with native LAMMPS
 * (black box), on synthetic files, the table works as follows.
 *
 * Energy and force: below cutinner the value is the analytic shifted term
 * (exact).  From cutinner up to the entry's own cutoff rc the energy and the
 * force are linear in rsq between table nodes; the node values are the
 * analytic shifted energy U2s(r) and the force -dU2s/dr / r, with the shift
 * of the entry itself.  At and beyond rc both are zero.
 *
 * Node range: the nodes are equally spaced in rsq from cutinner^2 up to R^2,
 * dr2 = (R^2 - cutinner^2)/(Ntable - 1), where R is the largest cutoff among
 * the potential entries whose three elements are all mapped to atom types by
 * pair_coeff, whatever types the atoms have (measured: a dimer of type-1 atoms
 * with A and B mapped has its kinks every 0.5365 in rsq for Ntable 30 and
 * cutinner 1.5, the spacing of R = 4.22, the largest A/B entry cutoff, and
 * agrees with the rule to 3.5e-11 over 2000 distances; entries of a third
 * element C in the file with cutoff 5.0 but not mapped leave R at 4.22; raising
 * one A/B entry cutoff moves R).
 * The energy stays zero beyond the entry's own rc even when R is larger.
 *
 * Accuracy measured against native (black box): with Ntable = 5 (cutinner 2.0),
 * 11 and 30 (cutinner 1.5), dimers of every element pair and a four-atom A/B
 * system agree with the rule above to 3e-11 in energy; the oracle case
 * tests/oracle/w13pair_vashtable (Ntable 11, 64 atoms, 30 steps) agrees to
 * the printed precision.
 *
 * Other native behaviour measured (black box): Ntable = 1 gives nan
 * energies; cutinner = 0 or negative stops with the error Illegal inner
 * cutoff for tabulation; a non-integer Ntable with Expected integer
 * parameter; a wrong argument count with Illegal pair_style command;
 * Ntable = 2 works; a cutinner at or beyond the cutoff is accepted
 * (every distance is then analytic).  The doc's "It is not recommended to use
 * less than 5000 tabulation points" is advice, not an error.  Ntable < 2 is
 * rejected here as a StyleError.
 *
 * pair_modify shift, tail and table are rejected by the analytic part
 * (vashishta.ts), as the doc says: "This pair style does not support the
 * pair_modify shift, table, and tail options."
 *
 * Composition: this class holds a PairVashishta (the analytic style) for
 * parsing, mapping, the three-body term and the neighbor cutoffs.  Its
 * two-body cutoff is set to zero after initOne so its compute() skips the
 * two-body pair terms; this class then adds the tabulated pair terms with the
 * same full-list, half-per-visit scheme as vashishta.ts.
 *
 * Two-body entry choice: the doc does not say which ordered entry (EI,EJ,EJ)
 * or (EJ,EI,EI) a pair takes when the two differ.  Measured with native
 * LAMMPS, a dimer's energy equals one ordered entry, and which one depends on
 * the neighbor-list build (atom order and binning), not on the element types
 * alone.  This engine keeps the analytic style's half-per-visit average of
 * both entries, which is not what native does for asymmetric entries; the
 * oracle case uses identical two-body parameters for the two orderings, and
 * the three-body B, C, costheta0 of AAB/ABA and BAB/BBA equal, so that the
 * three-body entry does not depend on the order of the two neighbours (not
 * tested with native whether the order matters for asymmetric triplets).
 * With these symmetric parameters the analytic three-body sums matched the
 * doc formula to 1e-16 on non-periodic and periodic structures (measured).
 */

/** pair_style vashishta/table: the analytic Vashishta pair term tabulated in rsq. */
export class PairVashishtaTable extends Pair {
  readonly name = 'vashishta/table';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private nTable = 0;
  private cutInner = 0;
  private readonly base = new PairVashishta();
  private nt = 0;
  private rcTab = 0;
  private rcArr = new Float64Array(0);
  private cut2 = new Float64Array(0);
  private cin2 = new Float64Array(0);
  private dr2 = new Float64Array(0);
  private invDr2 = new Float64Array(0);
  private on = new Uint8Array(0);
  private tabE = new Float64Array(0);
  private tabF = new Float64Array(0);

  override settings(args: string[], ctx: StyleContext): void {
    if (args.length !== 2) {
      throw new StyleError(`usage: pair_style ${this.name} Ntable cutinner (got ${args.length} arguments)`);
    }
    const n = parseInt_(args[0], 'Ntable of pair_style vashishta/table');
    if (n < 2) {
      throw new StyleError(`pair_style ${this.name} needs Ntable >= 2 (got ${n}): the table spacing (cutoff^2 - cutinner^2)/(Ntable - 1) is undefined for one point`);
    }
    const cin = parseNum(args[1], 'cutinner of pair_style vashishta/table');
    if (!(cin > 0)) {
      throw new StyleError(`pair_style ${this.name}: illegal inner cutoff for tabulation (cutinner = ${cin}, must be > 0)`);
    }
    this.nTable = n;
    this.cutInner = cin;
    this.base.settings([], ctx);
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.base.allocate(ntypes);
  }

  override coeff(args: string[], ctx: StyleContext): void {
    this.base.coeff(args, ctx);
  }

  override initStyle(ctx: StyleContext): void {
    // pair_modify flags live on this object; the analytic part checks them
    this.base.shift = this.shift;
    this.base.tail = this.tail;
    this.base.table = this.table;
    this.base.initStyle(ctx);
    const nt = this.ntypes + 1;
    this.nt = nt;
    this.rcTab = this.tableCutoff();
    this.rcArr = new Float64Array(nt * nt);
    this.cut2 = new Float64Array(nt * nt);
    this.cin2 = new Float64Array(nt * nt);
    this.dr2 = new Float64Array(nt * nt);
    this.invDr2 = new Float64Array(nt * nt);
    this.on = new Uint8Array(nt * nt);
    this.tabE = new Float64Array(nt * nt * this.nTable);
    this.tabF = new Float64Array(nt * nt * this.nTable);
  }

  /** Analytic shifted two-body energy of ordered type pair t at distance r (vashishta.ts phi2s). */
  private phi(t: number, r: number): number {
    const b = this.base as unknown as Record<string, Float64Array>;
    const H = b['tbH'][t], eta = b['tbEta'][t], zpq = b['tbZpq'][t], lam1 = b['tbLam1'][t];
    const D = b['tbD'][t], lam4 = b['tbLam4'][t], W = b['tbW'][t];
    const rc = this.rcArr[t];
    return (
      H / Math.pow(r, eta) + (zpq * Math.exp(-r / lam1)) / r -
      (D / Math.pow(r, 4)) * Math.exp(-r / lam4) - W / Math.pow(r, 6) -
      b['tbU2rc'][t] - (r - rc) * b['tbDuRc'][t]
    );
  }

  /** Radial derivative of the shifted two-body energy (vashishta.ts du - duRc). */
  private dphi(t: number, r: number): number {
    const b = this.base as unknown as Record<string, Float64Array>;
    const H = b['tbH'][t], eta = b['tbEta'][t], zpq = b['tbZpq'][t], lam1 = b['tbLam1'][t];
    const D = b['tbD'][t], lam4 = b['tbLam4'][t], W = b['tbW'][t];
    const scr = Math.exp(-r / lam1);
    return (
      -eta * H / Math.pow(r, eta + 1) - zpq * scr * (1 / (r * r) + 1 / (lam1 * r)) +
      D * Math.exp(-r / lam4) * (4 / Math.pow(r, 5) + 1 / (lam4 * Math.pow(r, 4))) +
      6 * W / Math.pow(r, 7) - b['tbDuRc'][t]
    );
  }

  /**
   * Node range of the table: the largest cutoff among the entries of the potential file whose three
   * elements are all mapped to atom types by pair_coeff (measured with native LAMMPS, see the file
   * header: every pair takes the range of the mapped element set, not its own).
   */
  private tableCutoff(): number {
    const b = this.base as unknown as {
      entries: Map<string, { e1: string; e2: string; e3: string; rc: number }>;
      elemOf: Int32Array;
      elemNames: string[];
    };
    const names = new Set<string>();
    for (let t = 1; t < b.elemOf.length; t++) if (b.elemOf[t] >= 0) names.add(b.elemNames[b.elemOf[t]]);
    let rc = 0;
    for (const e of b.entries.values()) {
      if (names.has(e.e1) && names.has(e.e2) && names.has(e.e3) && e.rc > rc) rc = e.rc;
    }
    return rc;
  }

  override initOne(i: number, j: number): number {
    const cut = this.base.initOne(i, j);
    const b = this.base as unknown as Record<string, Float64Array>;
    const nt = this.nt;
    const N = this.nTable;
    // i == j is one ordered index; the base cutoff is zeroed after it is read, so read each index once
    const ts = i === j ? [i * nt + j] : [i * nt + j, j * nt + i];
    // the energy stops at the entry's own cutoff; the node range is this.rcTab (see tableCutoff)
    const rcTab = this.rcTab;
    for (const t of ts) {
      // the analytic compute() must skip its two-body term: keep the cutoff here and zero the base's
      const rc = b['tbRc'][t];
      this.rcArr[t] = rc;
      b['tbRc'][t] = 0;
      this.cut2[t] = rc > 0 ? rc * rc : 0;
      this.on[t] = 0;
      if (rc <= 0) continue;
      // a non-positive table range keeps every distance analytic (r < cutinner is analytic)
      if (this.cutInner >= rc) continue;
      const c2 = this.cutInner * this.cutInner;
      const range = Math.max(rcTab, rc);
      const dr2 = (range * range - c2) / (N - 1);
      this.cin2[t] = c2;
      this.dr2[t] = dr2;
      this.invDr2[t] = 1 / dr2;
      this.on[t] = 1;
      for (let k = 0; k < N; k++) {
        const s = k === N - 1 ? range * range : c2 + k * dr2;
        const r = Math.sqrt(s);
        // node values are the shifted analytic term with this entry's own shift (no cut at the node)
        this.tabE[t * N + k] = this.phi(t, r);
        this.tabF[t * N + k] = -this.dphi(t, r) / r;
      }
    }
    return cut;
  }

  override compute(pc: PairCompute): void {
    // three-body terms (the base's two-body cutoffs are zero, so only its 3-body legs act)
    this.base.cut = this.cut;
    this.base.cutsq = this.cutsq;
    this.base.compute(pc);
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nt = this.nt;
    const N = this.nTable;
    const eatom = pc.eatom;
    const vatom = pc.vatom;
    const cut2 = this.cut2;
    let evdwl = 0;
    for (let i = 0; i < nlocal; i++) {
      const ti = type[i] * nt;
      const k0: number = list.firstneigh[i];
      const k1: number = k0 + list.numneigh[i];
      for (let k: number = k0; k < k1; k++) {
        const j: number = list.neighbors[k] & NEIGHMASK;
        const t: number = ti + type[j];
        if (cut2[t] === 0) continue;
        const dxx = x[3 * j] - x[3 * i];
        const dyy = x[3 * j + 1] - x[3 * i + 1];
        const dzz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dxx * dxx + dyy * dyy + dzz * dzz;
        if (rsq >= cut2[t]) continue;
        const r = Math.sqrt(rsq);
        let E: number, F: number;
        if (r < this.cutInner || !this.on[t]) {
          E = this.phi(t, r);
          F = -this.dphi(t, r) / r;
        } else {
          // linear interpolation in rsq between the nodes (the doc's "linear interpolation")
          let idx = Math.floor((rsq - this.cin2[t]) * this.invDr2[t]);
          if (idx > N - 2) idx = N - 2;
          if (idx < 0) idx = 0;
          const frac = (rsq - (this.cin2[t] + idx * this.dr2[t])) * this.invDr2[t];
          const p = t * N + idx;
          E = this.tabE[p] + frac * (this.tabE[p + 1] - this.tabE[p]);
          F = this.tabF[p] + frac * (this.tabF[p + 1] - this.tabF[p]);
        }
        // full list, half per visit: the pair is seen from both ends (vashishta.ts pass 1)
        const fpair = F;
        const hx = 0.5 * fpair * dxx;
        const hy = 0.5 * fpair * dyy;
        const hz = 0.5 * fpair * dzz;
        f[3 * j] += hx;
        f[3 * j + 1] += hy;
        f[3 * j + 2] += hz;
        f[3 * i] -= hx;
        f[3 * i + 1] -= hy;
        f[3 * i + 2] -= hz;
        evdwl += 0.5 * E;
        if (eatom) {
          const e4 = 0.25 * E;
          eatom[i] += e4;
          eatom[j] += e4;
        }
        if (vatom) {
          const vc = 0.25 * fpair;
          const v0 = vc * dxx * dxx, v1 = vc * dyy * dyy, v2 = vc * dzz * dzz;
          const v3 = vc * dxx * dyy, v4 = vc * dxx * dzz, v5 = vc * dyy * dzz;
          vatom[6 * i] += v0; vatom[6 * i + 1] += v1; vatom[6 * i + 2] += v2;
          vatom[6 * i + 3] += v3; vatom[6 * i + 4] += v4; vatom[6 * i + 5] += v5;
          vatom[6 * j] += v0; vatom[6 * j + 1] += v1; vatom[6 * j + 2] += v2;
          vatom[6 * j + 3] += v3; vatom[6 * j + 4] += v4; vatom[6 * j + 5] += v5;
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
