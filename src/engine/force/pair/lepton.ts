import { Pair, PairParams, StyleError, typeBounds, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { parseNum } from '../util';
import { compileLepton, fillVrefs, type LeptonProgram } from '../../lepton';
import type { ZblConst } from '../../lepton/compile';
import { ANGSTROM_PER_UNIT } from '../../lepton/units';

/*
 * pair_style lepton, lepton/coul and lepton/sphere — docs.lammps.org/pair_lepton.html.
 *
 *   pair_style lepton cutoff
 *   pair_style lepton/coul cutoff [ewald|pppm|msm]
 *   pair_style lepton/sphere cutoff
 *   pair_coeff I J "expression" [cutoff]
 *
 * "The potential function must be provided as an expression string using "r"
 * as the distance variable. With pair style *lepton/coul* one may additionally
 * reference the charges of the two atoms of the pair with "qi" and "qj",
 * respectively. With pair style *lepton/sphere* one may instead reference the
 * radii of the two atoms of the pair with "radi" and "radj", respectively"
 * "The last coefficient is optional; it allows to set the cutoff for a pair of
 * atom types to a different value than the global cutoff."
 * "Pair styles *lepton*, *lepton/coul*, and *lepton/sphere* do not support
 * mixing. Thus, expressions for *all* I,J pairs must be specified explicitly."
 * shift: "Only pair style *lepton* supports the pair_modify shift option for
 * shifting the potential energy of the pair interaction so that it is 0 at the
 * cutoff, pair styles *lepton/coul* and *lepton/sphere* do *not*."
 * tail: "These pair styles do not support the pair_modify tail option".
 * special_bonds: "For pair style *lepton* only the "lj" values of the
 * special_bonds settings apply in case the interacting pair is also connected
 * with a bond. The potential energy will *only* be added to the "evdwl"
 * property." For *lepton/coul* only the "coul" values apply and the energy goes
 * to "ecoul"; *lepton/sphere* behaves like *lepton*.
 * Keywords of lepton/coul: "keyword = *ewald* or *pppm* or *msm* or
 * *dispersion* or *tip4p*". This engine accepts ewald, pppm and msm (the
 * kspace style must then be defined; the expression supplies the short-range
 * part, e.g. erfc(alpha*r)) and rejects dispersion and tip4p.
 * Restrictions: lepton/coul needs atom_style charge (or full); lepton/sphere
 * needs atom_style sphere.
 * The expression is evaluated with exact derivatives (lepton/expression.ts).
 */

type Variant = 'lepton' | 'lepton/coul' | 'lepton/sphere';

/** Context with the equal-style variable hook (system.ts styleContext, see HOOKS NEEDED). */
export interface LeptonStyleContext extends StyleContext {
  equalVariable?: (name: string) => number;
}

export class PairLepton extends Pair {
  readonly name: Variant;
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  /** Expression text per type pair (i*nt+j, symmetric). */
  private texts: string[] = [];
  private progs: (LeptonProgram | null)[] = [];
  private shiftE = new Float64Array(0);
  private ctx: LeptonStyleContext | null = null;
  private readonly zbl: ZblConst = { qqr2e: 1, angstrom: 1 };
  /** lepton/coul: the keyword asked for a long-range solver (kspace style required). */
  private coulKeyword = false;

  constructor(variant: Variant) {
    super();
    this.name = variant;
  }

  private get builtins(): string[] {
    if (this.name === 'lepton/coul') return ['r', 'qi', 'qj'];
    if (this.name === 'lepton/sphere') return ['r', 'radi', 'radj'];
    return ['r'];
  }

  settings(args: string[]): void {
    if (this.name !== 'lepton/coul') {
      if (args.length !== 1) throw new StyleError(`usage: pair_style ${this.name} cutoff`);
    } else {
      if (args.length < 1) throw new StyleError('usage: pair_style lepton/coul cutoff [ewald|pppm|msm]');
      for (const kw of args.slice(1)) {
        if (kw === 'ewald' || kw === 'pppm' || kw === 'msm') this.coulKeyword = true;
        else if (kw === 'dispersion' || kw === 'tip4p') {
          throw new StyleError(`pair_style lepton/coul keyword '${kw}' is not supported by this engine (ewald, pppm and msm are)`);
        } else throw new StyleError(`pair_style lepton/coul: unknown keyword '${kw}' (expected ewald, pppm or msm)`);
      }
      this.coulLong = this.coulKeyword;
    }
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['cut']);
    const nt = ntypes + 1;
    this.texts = new Array(nt * nt).fill('');
    this.progs = new Array(nt * nt).fill(null);
  }

  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length < 3 || args.length > 4) {
      throw new StyleError(`usage: pair_coeff I J "expression" [cutoff] (pair style ${this.name})`);
    }
    if (ctx) this.ctx = ctx as LeptonStyleContext;
    const cut = args[3] !== undefined ? parseNum(args[3], 'cutoff') : this.cutGlobal;
    if (!(cut > 0)) throw new StyleError('pair_coeff cutoff must be > 0');
    const text = args[2];
    const prog = compileLepton(text, { builtins: this.builtins, wrt: ['r'], zbl: this.zbl });
    const nt = this.ntypes + 1;
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.p.set(i, j, [cut]);
        this.texts[i * nt + j] = this.texts[j * nt + i] = text;
        this.progs[i * nt + j] = this.progs[j * nt + i] = prog;
        count++;
      }
    }
    if (count === 0) throw new StyleError(`pair_coeff: no type pairs with I <= J in ${args[0]} ${args[1]}`);
  }

  initStyle(ctx: StyleContext): void {
    this.ctx = ctx as LeptonStyleContext;
    // pair_modify options that the pair page lists as unsupported
    if (this.tail) throw new StyleError(`pair_modify tail yes is not supported for pair style ${this.name}`);
    if (this.shift && this.name !== 'lepton') throw new StyleError(`pair_modify shift yes is not supported for pair style ${this.name}`);
    const s = ctx.s;
    if (this.name === 'lepton/coul' && s && !s.q) throw new StyleError(`pair_style ${this.name} requires atom_style charge (or full)`);
    if (this.name === 'lepton/sphere' && s && !s.radius) throw new StyleError(`pair_style ${this.name} requires atom_style sphere`);
    if (s) {
      this.zbl.qqr2e = s.units.qqr2e;
      this.zbl.angstrom = ANGSTROM_PER_UNIT[s.units.style] ?? 1;
    }
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.p.isSet(i, j) || !this.progs[i * nt + j]) {
      throw new StyleError(`all pair coeffs are not set (pair ${i} ${j}): pair style ${this.name} does not mix`);
    }
    if (this.shiftE.length !== nt * nt) this.shiftE = new Float64Array(nt * nt);
    return this.p.get('cut', i, j);
  }

  /** Looks up the value of a v_name reference at the current time. */
  private varValue(name: string): number {
    const fn = this.ctx?.equalVariable;
    if (!fn) throw new StyleError(`pair style ${this.name}: v_${name} needs the equal-style variable hook (not available)`);
    return fn(name);
  }

  private vrefProgs(): LeptonProgram[] {
    const out: LeptonProgram[] = [];
    for (const p of this.progs) if (p && !out.includes(p)) out.push(p);
    return out;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const sLJ = pc.specialLJ, sCoul = pc.specialCoul;
    const coul = this.name === 'lepton/coul';
    const sphere = this.name === 'lepton/sphere';
    const tally = pc.eatom !== null || pc.vatom !== null;
    this.zbl.qqr2e = pc.s.units.qqr2e;
    this.zbl.angstrom = ANGSTROM_PER_UNIT[pc.s.units.style] ?? 1;
    // per program: fill v_name slots and the energy shift at the cutoff
    const envOf = new Map<LeptonProgram, Float64Array>();
    for (const p of this.vrefProgs()) {
      const env = new Float64Array(p.builtins.length + p.vrefs.length);
      fillVrefs(env, p, (n) => this.varValue(n));
      envOf.set(p, env);
    }
    for (let t = 1; t < nt; t++) {
      for (let u = t; u < nt; u++) {
        const p = this.progs[t * nt + u];
        if (!p) continue;
        const env = envOf.get(p)!;
        env[0] = this.cut[t * nt + u];
        this.shiftE[t * nt + u] = this.shift ? p.value(env) : 0;
        this.shiftE[u * nt + t] = this.shiftE[t * nt + u];
      }
    }
    const owner = pc.nb.owner;
    const radius = pc.s.radius;
    const q = pc.q;
    let evdwl = 0, ecoul = 0;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const prog = this.progs[t];
        if (!prog) continue;
        const env = envOf.get(prog)!;
        const r = Math.sqrt(rsq);
        env[0] = r;
        if (coul) { env[1] = q[i]; env[2] = q[j]; }
        else if (sphere && radius) { env[1] = radius[owner[i]]; env[2] = radius[owner[j]]; }
        const factor = coul ? sCoul[jj >>> SBBITS] : sLJ[jj >>> SBBITS];
        const dEdr = prog.deriv[0](env);
        const e = factor * (prog.value(env) - this.shiftE[t]);
        const fpair = (-factor * dEdr) / r;
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (coul) ecoul += e; else evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  extract(name: string): unknown {
    if (name === 'cut_coul' && this.name === 'lepton/coul' && this.coulKeyword) return this.cutGlobal;
    return undefined;
  }
}
