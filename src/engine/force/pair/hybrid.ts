import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext, newAccum, clearAccum, type Accum } from '../types';
import { NEIGHMASK, type NeighList } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style hybrid, hybrid/overlay, hybrid/scaled, hybrid/molecular —
 * docs.lammps.org/pair_hybrid.html.
 *
 *   "With the hybrid style, exactly one pair style is assigned to each pair
 *   of atom types. With the hybrid/overlay and hybrid/scaled styles, one or
 *   more pair styles can be assigned to each pair of atom types. With the
 *   hybrid/molecular style, pair styles are assigned to either intra- or
 *   inter-molecular interactions."
 *   "In the pair_coeff commands, the name of a pair style must be added after
 *   the I,J type specification, with the remaining coefficients being those
 *   appropriate to that style. If the pair style is used multiple times in
 *   the pair_style command, then an additional numeric argument must also be
 *   specified which is a number from 1 to M where M is the number of times
 *   the sub-style was listed in the pair style command."
 *   "For the hybrid style, each atom type pair I,J is assigned to exactly one
 *   sub-style. Just as with a simulation using a single pair style, if you
 *   specify the same atom type pair in a second pair_coeff command, the
 *   previous assignment will be overwritten."
 *   "If you specify the same atom type pair in a second pair_coeff command
 *   with a new sub-style, then the second sub-style is added to the list of
 *   potentials that will be calculated for two interacting atoms of those
 *   types."
 *   "If an assignment to none is made in a simulation with the hybrid/overlay
 *   or hybrid/scaled pair style, it wipes out all previous assignments of
 *   that pair of atom types to sub-styles."
 *   "every atom type pair I,J (where I <= J) must be assigned to at least one
 *   sub-style via the pair_coeff command as in the examples above, or in the
 *   data file read by the read_data, or by mixing as described below. Also
 *   all sub-styles must be used at least once in a pair_coeff command."
 *   "For atom type pairs I,J and I != J, if the sub-style assigned to I,I and
 *   J,J is the same, and if the sub-style allows for mixing, then the
 *   coefficients for I,J can be mixed." "For the hybrid/overlay and
 *   hybrid/scaled style, there is an additional requirement that both the
 *   I,I and J,J pairs are assigned to a single sub-style."
 *   "The NULL keyword is used by many such potentials (eam/alloy, Tersoff,
 *   AIREBO, etc), to denote an atom type that will be assigned to a
 *   different sub-style."
 *   "Any pair potential settings made via the pair_modify command are passed
 *   along to all sub-styles of the hybrid potential."
 *   hybrid/scaled: "the scale factor for each sub-style may be a constant, an
 *   equal style variable, or an atom style variable" (atom-style factors are
 *   not supported by the browser engine and are rejected).
 *   hybrid/molecular: "accepts only two sub-styles: the first is assigned to
 *   intra-molecular interactions (i.e. both atoms have the same molecule ID),
 *   the second to inter-molecular interactions (i.e. interacting atoms have
 *   different molecule IDs)." "Pair style hybrid/molecular is not compatible
 *   with manybody potentials."
 *   "You must ensure that the short-range Coulombic cutoff used by each of
 *   these long pair styles is the same or else LAMMPS will generate an
 *   error."
 *
 * Each sub-style computes on a "skip list": the main neighbor list filtered to
 * the type pairs (and, for hybrid/molecular, the molecule relation) assigned
 * to it, so a sub-style never sees pairs that belong to another one.
 * Sub-styles initialize only their assigned type pairs; their cutoff stays 0
 * elsewhere, which also restricts their tail corrections to those pairs.
 */

export type HybridMode = 'hybrid' | 'hybrid/overlay' | 'hybrid/scaled' | 'hybrid/molecular';

interface Sub {
  style: Pair;
  name: string;
  /** 1-based instance number among sub-styles of the same name. */
  instance: number;
  /** hybrid/scaled factor: a number or an equal-style variable name. */
  scale: number | string;
  /** pair_modify pair ... special overrides: [1, w12, w13, w14]. */
  specialLJ: Float64Array | null;
  specialCoul: Float64Array | null;
  used: boolean;
  half: NeighList | null;
  full: NeighList | null;
}

export class PairHybrid extends Pair {
  readonly name: string;
  subs: Sub[] = [];
  /** Sub-style indices per type pair ((ntypes+1)^2, symmetric); null = never assigned. */
  private map: (number[] | null)[] = [];
  private listsBuilt = -1;
  private lastHalf: NeighList | null = null;
  private lastFull: NeighList | null = null;
  /** Equal-style variable evaluation for hybrid/scaled factors. */
  private evalVar: ((name: string) => number) | null = null;

  constructor(readonly mode: HybridMode, private registry: Record<string, () => Pair>) {
    super();
    this.name = mode;
    this.virialFdotr = true;
  }

  settings(args: string[], ctx: StyleContext): void {
    if (args.length === 0) throw new StyleError(`usage: pair_style ${this.mode} style1 args style2 args ...`);
    const subs: Sub[] = [];
    const scaled = this.mode === 'hybrid/scaled';
    let k = 0;
    while (k < args.length) {
      let scale: number | string = 1;
      if (scaled) {
        const w = args[k++];
        if (w === undefined) break;
        if (w.startsWith('v_')) scale = w.slice(2);
        else scale = parseNum(w, 'hybrid/scaled scale factor');
      }
      const name = args[k];
      if (name === undefined) throw new StyleError(`pair_style ${this.mode}: a scale factor must be followed by a sub-style`);
      if (name.startsWith('hybrid')) throw new StyleError(`pair_style ${this.mode}: a hybrid style cannot be a sub-style`);
      if (name === 'none') throw new StyleError(`pair_style ${this.mode}: none is not a sub-style (use pair_coeff I J none)`);
      const make = this.registry[name];
      if (!make) {
        throw new StyleError(`pair_style ${this.mode}: sub-style '${name}' is not supported by the browser engine; supported: ${Object.keys(this.registry).filter((n) => !n.startsWith('hybrid')).sort().join(', ')}`);
      }
      // the sub-style's arguments run up to the next sub-style name (or scale factor)
      let e = k + 1;
      while (e < args.length && !this.startsSub(args, e)) e++;
      const style = make();
      style.settings(args.slice(k + 1, e), ctx);
      if (!style.virialFdotr) throw new StyleError(`pair_style ${this.mode}: sub-style ${name} cannot be combined by the browser engine`);
      const instance = subs.filter((s) => s.name === name).length + 1;
      subs.push({ style, name, instance, scale, specialLJ: null, specialCoul: null, used: false, half: null, full: null });
      k = e;
    }
    if (this.mode === 'hybrid/molecular') {
      if (subs.length !== 2) throw new StyleError('pair_style hybrid/molecular accepts only two sub-styles (intra-molecular, then inter-molecular)');
      if (subs.some((s) => s.style.manybody)) throw new StyleError('pair_style hybrid/molecular is not compatible with manybody potentials');
    }
    this.subs = subs;
    this.needsHalf = subs.some((s) => s.style.needsHalf);
    this.needsFull = subs.some((s) => s.style.needsFull);
    this.coulLong = subs.some((s) => s.style.coulLong);
    // special_bonds weights do not apply to many-body sub-styles; pairwise ones need them
    this.manybody = subs.every((s) => s.style.manybody);
    if (this.ntypes > 0) this.allocate(this.ntypes);
  }

  /** Does a sub-style (or, for hybrid/scaled, its scale factor) start at args[k]? */
  private startsSub(args: string[], k: number): boolean {
    if (this.mode === 'hybrid/scaled') {
      const w = args[k], next = args[k + 1];
      const isFactor = w.startsWith('v_') || (w.trim() !== '' && Number.isFinite(Number(w)));
      return isFactor && next !== undefined && next in this.registry;
    }
    return args[k] in this.registry;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    for (const s of this.subs) s.style.allocate(ntypes);
    this.map = new Array((ntypes + 1) * (ntypes + 1)).fill(null);
  }

  /** The sub-style named by words[0] (with an instance number in words[1] when it is listed more than once). */
  selectSub(words: string[], what: string): { sub: Sub; used: number } {
    const name = words[0];
    const all = this.subs.filter((s) => s.name === name);
    if (all.length === 0) throw new StyleError(`${what}: '${name}' is not a sub-style of pair_style ${this.mode} (${this.subs.map((s) => s.name).join(', ')})`);
    if (all.length === 1) return { sub: all[0], used: 1 };
    const m = /^\d+$/.test(words[1] ?? '') ? Number(words[1]) : 0;
    if (m < 1 || m > all.length) {
      throw new StyleError(`${what}: sub-style ${name} is listed ${all.length} times; give its number 1..${all.length} after the name`);
    }
    return { sub: all[m - 1], used: 2 };
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 3) throw new StyleError(`usage: pair_coeff I J sub-style args (pair_style ${this.mode})`);
    const n = this.ntypes;
    const [ilo, ihi] = typeBounds(args[0], n);
    const [jlo, jhi] = typeBounds(args[1], n);
    const pairs: [number, number][] = [];
    for (let i = ilo; i <= ihi; i++) for (let j = Math.max(jlo, i); j <= jhi; j++) pairs.push([i, j]);
    if (pairs.length === 0 && ilo === ihi && jlo === jhi) pairs.push([jlo, ilo]);
    if (pairs.length === 0) throw new StyleError(`pair_coeff: no type pairs with I <= J in ${args[0]} ${args[1]}`);
    if (args[2] === 'none') {
      if (args.length !== 3) throw new StyleError('usage: pair_coeff I J none');
      for (const [i, j] of pairs) this.assign(i, j, []);
      return;
    }
    const { sub, used } = this.selectSub(args.slice(2), 'pair_coeff');
    const rest = args.slice(2 + used);
    sub.style.coeff([args[0], args[1], ...rest], ctx);
    sub.used = true;
    const idx = this.subs.indexOf(sub);
    // many-body styles map elements to all types with "* *"; NULL types are left to other sub-styles
    let active: ((t: number) => boolean) | null = null;
    if (sub.style.manybody && args[0] === '*' && args[1] === '*') {
      const elems = rest.slice(rest.length - n);
      active = (t) => elems[t - 1] !== 'NULL';
    }
    for (const [i, j] of pairs) {
      if (active && (!active(i) || !active(j))) continue;
      const cur = this.map[i * (n + 1) + j];
      if (this.mode === 'hybrid') this.assign(i, j, [idx]);
      else if (!cur || !cur.includes(idx)) this.assign(i, j, [...(cur ?? []), idx]);
    }
  }

  private assign(i: number, j: number, subs: number[]): void {
    const nt = this.ntypes + 1;
    this.map[i * nt + j] = subs;
    this.map[j * nt + i] = subs;
  }

  /** Sub-styles assigned to the type pair (after init: including mixed pairs). */
  subsOf(i: number, j: number): readonly number[] {
    return this.map[i * (this.ntypes + 1) + j] ?? [];
  }

  init(ctx: StyleContext): void {
    // pair_modify.html: "You cannot use shift yes with tail yes, since those are conflicting
    // options." — per-sub-style settings also set the hybrid's own flags, as in native LAMMPS
    if (this.shift && this.tail) throw new StyleError('cannot have both pair_modify shift and tail set to yes');
    const unused = this.subs.find((s) => !s.used);
    if (unused) throw new StyleError(`pair_style ${this.mode}: sub-style ${unused.name} is not used in any pair_coeff command`);
    const n = this.ntypes, nt = n + 1;
    // mixing: an unassigned I,J takes the sub-style shared by I,I and J,J
    for (let i = 1; i <= n; i++) {
      for (let j = i; j <= n; j++) {
        if (this.map[i * nt + j]) continue;
        const a = this.map[i * nt + i], b = this.map[j * nt + j];
        const single = (m: number[] | null) => m !== null && m.length === 1;
        if (i !== j && single(a) && single(b) && a![0] === b![0]) this.assign(i, j, [a![0]]);
        else throw new StyleError(`all pair coeffs are not set (pair ${i} ${j} has no sub-style of pair_style ${this.mode})`);
      }
    }
    this.cut = new Float64Array(nt * nt);
    this.cutsq = new Float64Array(nt * nt);
    this.etail = this.ptail = 0;
    let coulCut: number | null = null;
    for (let k = 0; k < this.subs.length; k++) {
      const st = this.subs[k].style;
      // pair.html: shift and tail conflict; tail is not defined in 2d (checked per sub-style)
      if (st.shift && st.tail) throw new StyleError('cannot have both pair_modify shift and tail set to yes');
      if (st.tail && ctx.s?.dimension === 2) throw new StyleError('cannot use pair_modify tail yes with 2d simulations');
      st.gEwald = this.gEwald;
      st.initStyle(ctx);
      st.cut = new Float64Array(nt * nt);
      st.cutsq = new Float64Array(nt * nt);
      for (let i = 1; i <= n; i++) {
        for (let j = i; j <= n; j++) {
          if (!this.map[i * nt + j]!.includes(k)) continue;
          const c = st.initOne(i, j);
          st.cut[i * nt + j] = st.cut[j * nt + i] = c;
          st.cutsq[i * nt + j] = st.cutsq[j * nt + i] = c * c;
          if (c > this.cut[i * nt + j]) {
            this.cut[i * nt + j] = this.cut[j * nt + i] = c;
            this.cutsq[i * nt + j] = this.cutsq[j * nt + i] = c * c;
          }
        }
      }
      if (st.coulLong) {
        const cc = st.extract('cut_coul');
        if (typeof cc === 'number') {
          if (coulCut !== null && cc !== coulCut) throw new StyleError(`pair_style ${this.mode}: the long-range Coulomb sub-styles must use the same Coulomb cutoff (${coulCut} vs ${cc})`);
          coulCut = cc;
        }
      }
    }
    this.listsBuilt = -1;
  }

  /** The merged cutoff of the type pair (init() initializes the sub-styles). */
  initOne(i: number, j: number): number {
    return this.cut[i * (this.ntypes + 1) + j];
  }

  tailSums(counts: Float64Array): { etail: number; ptail: number } {
    let etail = 0, ptail = 0;
    for (const s of this.subs) {
      const t = s.style.tailSums(counts);
      const f = this.scaleOf(s);
      etail += f * t.etail;
      ptail += f * t.ptail;
    }
    return { etail, ptail };
  }

  /** The sub-style's current scale factor (1 except for hybrid/scaled). */
  private scaleOf(s: Sub): number {
    if (typeof s.scale === 'number') return s.scale;
    if (!this.evalVar) throw new StyleError(`pair_style hybrid/scaled: cannot evaluate variable ${s.scale} here`);
    return this.evalVar(s.scale);
  }

  /** Lets the force field supply equal-style variable values for hybrid/scaled. */
  setVariableEvaluator(fn: (name: string) => number): void {
    this.evalVar = fn;
  }

  /**
   * Filters `list` to the pairs sub-style k owns (and, for hybrid/molecular,
   * to intra- or inter-molecular pairs).
   */
  private skipList(list: NeighList, k: number, type: Int32Array, mol: Int32Array | null): NeighList {
    const nt = this.ntypes + 1;
    const own = new Uint8Array(nt * nt);
    for (let a = 1; a < nt; a++) for (let b = 1; b < nt; b++) if (this.map[a * nt + b]?.includes(k)) own[a * nt + b] = 1;
    const intra = this.mode === 'hybrid/molecular' ? k === 0 : null;
    const { inum, numneigh, firstneigh, neighbors } = list;
    const out = new Int32Array(neighbors.length);
    const num = new Int32Array(inum), first = new Int32Array(inum + 1);
    let c = 0;
    for (let i = 0; i < inum; i++) {
      first[i] = c;
      const ti = type[i];
      const k0 = firstneigh[i], k1 = k0 + numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const jj = neighbors[kk];
        const j = jj & NEIGHMASK;
        if (!own[ti * nt + type[j]]) continue;
        if (intra !== null && mol && (mol[i] === mol[j]) !== intra) continue;
        out[c++] = jj;
      }
      num[i] = c - first[i];
    }
    first[inum] = c;
    return { inum, numneigh: num, firstneigh: first, neighbors: out.subarray(0, c) };
  }

  compute(pc: PairCompute): void {
    const nb = pc.nb;
    if (this.listsBuilt !== nb.nbuild || pc.half !== this.lastHalf || pc.full !== this.lastFull) {
      const mol = this.mode === 'hybrid/molecular' ? this.ghostMolecules(pc) : null;
      for (let k = 0; k < this.subs.length; k++) {
        const s = this.subs[k];
        s.half = s.style.needsHalf && pc.half ? this.skipList(pc.half, k, pc.type, mol) : null;
        s.full = s.style.needsFull && pc.full ? this.skipList(pc.full, k, pc.type, mol) : null;
      }
      this.listsBuilt = nb.nbuild;
      this.lastHalf = pc.half;
      this.lastFull = pc.full;
    }
    const scaled = this.mode === 'hybrid/scaled';
    for (const s of this.subs) {
      s.style.gEwald = this.gEwald;
      const sub: PairCompute = {
        ...pc, half: s.half, full: s.full,
        specialLJ: s.specialLJ ?? pc.specialLJ,
        specialCoul: s.specialCoul ?? pc.specialCoul,
      };
      if (!scaled) {
        s.style.compute(sub);
        continue;
      }
      const f = this.scaleOf(s);
      if (f === 0) continue;
      const fbuf = new Float64Array(pc.f.length);
      const acc: Accum = newAccum();
      clearAccum(acc);
      const eatom = pc.eatom ? new Float64Array(pc.eatom.length) : null;
      const vatom = pc.vatom ? new Float64Array(pc.vatom.length) : null;
      s.style.compute({ ...sub, f: fbuf, acc, eatom, vatom });
      for (let q = 0; q < fbuf.length; q++) pc.f[q] += f * fbuf[q];
      pc.acc.evdwl += f * acc.evdwl;
      pc.acc.ecoul += f * acc.ecoul;
      for (let c = 0; c < 6; c++) pc.acc.virial[c] += f * acc.virial[c];
      if (eatom) for (let q = 0; q < eatom.length; q++) pc.eatom![q] += f * eatom[q];
      if (vatom) for (let q = 0; q < vatom.length; q++) pc.vatom![q] += f * vatom[q];
    }
  }

  /** Molecule IDs of owned and ghost atoms (ghosts take their owner's). */
  private ghostMolecules(pc: PairCompute): Int32Array {
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    const out = new Int32Array(pc.nall);
    for (let k = 0; k < pc.nall; k++) out[k] = mol[owner[k]];
    return out;
  }

  single(i: number, j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number): { eng: number; fforce: number } {
    let eng = 0, fforce = 0;
    const nt = this.ntypes + 1;
    for (const k of this.subsOf(itype, jtype)) {
      const s = this.subs[k];
      if (!(rsq < s.style.cutsq[itype * nt + jtype])) continue;
      if (!s.style.single) throw new StyleError(`pair sub-style ${s.name} does not provide single-pair energies`);
      const r = s.style.single(i, j, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
      const f = this.scaleOf(s);
      eng += f * r.eng;
      fforce += f * r.fforce;
    }
    return { eng, fforce };
  }

  /** Coefficients are not written to data files for hybrid styles (pair_hybrid.html: "The same is true for data files."). */
  dataCoeffs(): string[] | null { return null; }

  extract(name: string): unknown {
    if (name === 'cut_coul') {
      for (const s of this.subs) {
        const v = s.style.extract('cut_coul');
        if (v !== undefined) return v;
      }
      return undefined;
    }
    for (const s of this.subs) {
      const v = s.style.extract(name);
      if (v !== undefined) return v;
    }
    return undefined;
  }

  /** pair_modify pair ... special which w1 w2 w3 for one sub-style. */
  setSpecial(sub: Sub, which: string, w: [number, number, number]): void {
    if (which !== 'lj' && which !== 'coul' && which !== 'lj/coul') throw new StyleError(`pair_modify special: which must be lj, coul or lj/coul, not '${which}'`);
    if (w.some((v) => !(v >= 0 && v <= 1))) throw new StyleError('pair_modify special: weights must be between 0.0 and 1.0');
    if (which === 'lj' || which === 'lj/coul') sub.specialLJ = Float64Array.from([1, ...w]);
    if (which === 'coul' || which === 'lj/coul') sub.specialCoul = Float64Array.from([1, ...w]);
  }

  /**
   * A sub-style's special weights must keep the neighbor list valid: where
   * the global special_bonds weight is 0 (pair excluded) or 1 (pair not
   * flagged) the sub-style's weight must be the same. Measured with native
   * LAMMPS (2 Sep 2026), which stops with "Pair_modify special lj 1-2
   * setting for pair hybrid substyle lj/cut incompatible with global
   * special_bonds setting"; weights strictly between 0 and 1 globally allow
   * any sub-style weight.
   */
  checkSpecial(global: { lj: readonly number[]; coul: readonly number[] }): void {
    for (const s of this.subs) {
      for (const [kind, mine, g] of [['lj', s.specialLJ, global.lj], ['coul', s.specialCoul, global.coul]] as const) {
        if (!mine) continue;
        for (let o = 0; o < 3; o++) {
          const gw = g[o], sw = mine[o + 1];
          if ((gw === 0 || gw === 1) && sw !== gw) {
            throw new StyleError(`pair_modify special ${kind} 1-${o + 2} setting for pair hybrid sub-style ${s.name} is incompatible with the global special_bonds setting (${gw})`);
          }
        }
      }
    }
  }

  modify(key: string, values: string[]): number {
    // style-specific keywords go to every sub-style that accepts them
    let used = -1;
    for (const s of this.subs) {
      try {
        used = Math.max(used, s.style.modify(key, values));
      } catch (e) {
        if (!(e instanceof StyleError)) throw e;
      }
    }
    if (used < 0) throw new StyleError(`pair_modify keyword '${key}' is not supported by any sub-style of pair_style ${this.mode}`);
    return used;
  }
}
