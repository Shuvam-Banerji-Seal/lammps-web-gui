/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/fix_adapt_fep.cpp and src/FEP/fix_adapt_fep.h (class FixAdaptFEP, fix adapt/fep) at commit
 * b332f28, by Agilio Padua (ENS de Lyon & CNRS) and the LAMMPS developers. Copyright Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix ID group-ID adapt/fep N keyword args ... keyword value ... (docs.lammps.org/fix_adapt_fep.html). Differences
 * from fix adapt (src/engine/fix/adapt.ts): the pair keyword takes the soft-core parameter lambda of the
 * lj/cut/coul/long/soft family (any pair parameter table of the defined style is accepted); the keywords
 * reset, scale and after; the atom keyword takes charge or diameter for a type range; kspace sets the kspace scale
 * factor (KSpace.scale); the timing follows fix_adapt_fep.cpp (pre_force: "if (afterflag) ... update at n+1").
 * Not supported (StyleError): dynamic groups with atom attributes, and respa.
 *
 * Source notes kept from fix_adapt_fep.cpp: setup applies the variables at every run start (setup_pre_force);
 * pre_force applies at steps where ntimestep % N == 0, or with "after yes" at steps N+1, 2N+1, ... (and
 * always when N is 1 or at step 0); post_run restores the original values when reset is yes. The original
 * per-atom charges are the values the fix saw when it was defined (the STORE/ATOM vector of the native fix).
 */
import { Fix } from '../../engine/fix/fix';
import { Pair, StyleError, typeBounds } from '../../engine/force/types';
import { hasChargeStyle, isSphereStyle } from '../../engine/atoms';
import { missingArgs } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { refreshChargeSums, reinitPair, resolvePairTable, setPairValue, typePairs, type PairTable } from './pair_table';

const USAGE = 'usage: fix ID group-ID adapt/fep N keyword args ... keyword value ...';
/** Words of each attribute keyword, as native's first argument loop counts them (src/FEP/fix_adapt_fep.cpp at b332f28). */
const ATTRIBUTE_WORDS: Record<string, number> = { pair: 6, kspace: 2, atom: 4 };

const logical = (key: string, w: string | undefined): boolean => {
  if (w !== 'yes' && w !== 'no') throw new StyleError(`fix adapt/fep keyword ${key} must be yes or no, got '${w ?? ''}'`);
  return w === 'yes';
};

interface PairAdapt {
  vname: string;
  table: PairTable;
  pairs: Array<[number, number]>;
  orig: Float64Array;
}

interface AtomAdapt {
  kind: 'charge' | 'diameter';
  vname: string;
  /** Type range as given (resolved against ntypes at run start). */
  iw: string;
}

/** fix_adapt_fep.cpp: the volume of a sphere of radius r, 4.0*MY_PI/3.0 * r*r*r in native's operation order. */
const sphereVolume = (r: number): number => 4.0 * Math.PI / 3.0 * r * r * r;

export class FixAdaptFEP extends Fix {
  readonly style = 'adapt/fep';
  private readonly pending: Array<{ pstyle: string; pparam: string; iw: string; jw: string; vname: string }> = [];
  private readonly atoms: AtomAdapt[] = [];
  /** kspace v_name entries (the kspace scale factor) */
  private readonly kspaces: string[] = [];
  private pairs: PairAdapt[] = [];
  private reset = false;
  private scale = false;
  private after = false;
  /** Charges and radii as the fix saw them when defined, by atom ID (reset yes restores them; the STORE/ATOM vectors). */
  private readonly origQ = new Map<number, number>();
  private readonly origR = new Map<number, number>();
  private definedQ = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const n = Number(args[0]);
    if (args[0] === undefined || !Number.isInteger(n) || n < 0) throw new StyleError(`fix ${id} (adapt/fep): N must be a non-negative integer, got '${args[0] ?? ''}' (${USAGE})`);
    this.nevery = n;
    // As native LAMMPS does (src/FEP/fix_adapt_fep.cpp FixAdaptFEP::FixAdaptFEP at b332f28): the leading attribute keywords
    // are counted first, each checked for its own number of arguments ("Illegal fix adapt/fep atom command: missing argument(s)")
    for (let p = 1; p < args.length; ) {
      const need = ATTRIBUTE_WORDS[args[p]];
      if (typeof need !== 'number') break;
      if (p + need > args.length) throw missingArgs(`fix adapt/fep ${args[p]}`);
      p += need;
    }
    let k = 1;
    let nattr = 0;
    while (k < args.length) {
      const w = args[k];
      if (w === 'reset' || w === 'scale' || w === 'after') {
        const v = logical(w, args[k + 1]);
        if (w === 'reset') this.reset = v;
        else if (w === 'scale') this.scale = v;
        else this.after = v;
        k += 2;
        continue;
      }
      if (w === 'pair') {
        const pstyle = args[k + 1], pparam = args[k + 2], iw = args[k + 3], jw = args[k + 4], vw = args[k + 5];
        if (vw === undefined || !vw.startsWith('v_')) throw new StyleError(`fix ${id} (adapt/fep): pair needs pstyle pparam I J v_name (${USAGE})`);
        this.pending.push({ pstyle, pparam, iw, jw, vname: vw.slice(2) });
        k += 6; nattr++;
        continue;
      }
      if (w === 'atom') {
        const kind = args[k + 1], iw = args[k + 2], vw = args[k + 3];
        if (vw === undefined || !vw.startsWith('v_')) throw new StyleError(`fix ${id} (adapt/fep): atom needs atomparam type v_name (${USAGE})`);
        if (kind !== 'charge' && kind !== 'diameter') throw new StyleError('Illegal fix adapt/fep command');
        this.atoms.push({ kind, vname: vw.slice(2), iw });
        k += 4; nattr++;
        continue;
      }
      if (w === 'kspace') {
        const vw = args[k + 1];
        if (vw === undefined || !vw.startsWith('v_')) throw new StyleError('Illegal fix adapt/fep command');
        this.kspaces.push(vw.slice(2));
        k += 2; nattr++;
        continue;
      }
      // As native LAMMPS does (src/FEP/fix_adapt_fep.cpp FixAdaptFEP::FixAdaptFEP at b332f28): a word that is not a keyword stops with "Illegal fix adapt/fep command".
      throw new StyleError('Illegal fix adapt/fep command');
    }
    if (nattr === 0) throw new StyleError(`fix ${id} (adapt/fep): nothing to adapt (${USAGE})`);
    if (this.scale && this.atoms.length) throw new StyleError("Keyword 'scale yes' is not supported with fix adapt/fep for 'atom'");
    // the STORE/ATOM vectors of the native fix are filled when the fix is defined (post_constructor, reset yes only)
    if (this.reset && this.atoms.length) this.storeAtoms();
  }

  private get hasCharge(): boolean { return this.atoms.some((a) => a.kind === 'charge'); }
  private get hasDiameter(): boolean { return this.atoms.some((a) => a.kind === 'diameter'); }

  private storeAtoms(): void {
    const s = this.sys.state;
    this.origQ.clear();
    this.origR.clear();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (this.hasCharge && s.q) this.origQ.set(s.id[i], s.q[i]);
      if (this.hasDiameter && s.radius) this.origR.set(s.id[i], s.radius[i]);
    }
    this.definedQ = true;
  }

  /**
   * Resolves the targets, copies the original pair values and applies the variables. The native fix does the
   * latter in setup_pre_force, before the first force evaluation of each run; this engine runs init() at the
   * start of every run, before the setup forces.
   */
  init(): void {
    const s = this.sys.state;
    // fix_adapt_fep.cpp init: a dynamic group only without the atom attribute
    if (this.atoms.length && this.sys.groups.isDynamic(this.groupBit)) throw new StyleError('Cannot use dynamic group with fix adapt/fep atom');
    if (this.kspaces.length && !this.sys.ff.kspace) throw new StyleError('Fix adapt/fep kspace style does not exist');
    if (this.hasDiameter && !s.radius) throw new StyleError('Fix adapt/fep requires atom attribute diameter');
    // AtomVecSphere::init (src/atom_vec_sphere.cpp at b332f28): changing diameters needs atom_style sphere 1
    if (this.hasDiameter && isSphereStyle(s.atomStyle) && !this.sys.radvary) {
      throw new StyleError('Fix adapt/fep changes atom radii but atom_style sphere is not dynamic');
    }
    if (this.hasCharge && !hasChargeStyle(s.atomStyle)) throw new StyleError('Fix adapt/fep requires atom attribute charge');
    this.pairs = [];
    for (const p of this.pending) {
      const table = resolvePairTable(this.sys, `fix ${this.id} (adapt/fep)`, p.pstyle, p.pparam);
      const pairs = typePairs(p.iw, p.jw, s.ntypes, true);
      if (!pairs.length) throw new StyleError(`fix ${this.id} (adapt/fep): no type pairs with I <= J in ${p.iw} ${p.jw}`);
      const orig = new Float64Array(pairs.length);
      pairs.forEach(([i, j], k) => { orig[k] = table.table.get(table.name, i, j); });
      this.pairs.push({ vname: p.vname, table, pairs, orig });
    }
    if (this.reset && this.atoms.length && !this.definedQ) this.storeAtoms();
    this.apply();
  }

  /** The variables are written to the tables and the atoms (change_settings). */
  private apply(): void {
    const sys = this.sys, s = sys.state;
    const owners = new Set<Pair>();
    for (const p of this.pairs) {
      const v = sys.equalVariable(p.vname);
      p.pairs.forEach(([i, j], k) => {
        const val = this.scale ? v * p.orig[k] : v;
        setPairValue(p.table, i, j, val);
      });
      owners.add(p.table.owner);
    }
    for (const owner of owners) reinitPair(sys, owner);
    // set kspace scale factor
    for (const vname of this.kspaces) sys.ff.kspace!.scale = sys.equalVariable(vname);
    for (const a of this.atoms) {
      const v = sys.equalVariable(a.vname);
      const [ilo, ihi] = typeBounds(a.iw, s.ntypes);
      if (a.kind === 'charge') {
        for (let i = 0; i < s.n; i++) {
          if (s.type[i] >= ilo && s.type[i] <= ihi && (s.mask[i] & this.groupBit)) s.q[i] = v;
        }
      } else {
        // reset radius from diameter, also scale rmass to new value (at the same density)
        const radius = s.radius!, rmass = s.rmass;
        for (let i = 0; i < s.n; i++) {
          if (!(s.type[i] >= ilo && s.type[i] <= ihi && (s.mask[i] & this.groupBit))) continue;
          if (!rmass) { radius[i] = 0.5 * v; continue; }
          const density = rmass[i] / sphereVolume(radius[i]);
          radius[i] = 0.5 * v;
          rmass[i] = sphereVolume(radius[i]) * density;
        }
      }
    }
    if (this.hasDiameter) sys.bump();
    if (this.hasCharge) refreshChargeSums(sys);
  }

  preForce(): void {
    if (this.nevery === 0) return;
    const step = this.sys.state.step;
    if (this.after) {
      if (this.nevery === 1 || step === 0) this.apply();
      else if (step > 1 && (step - 1) % this.nevery === 0) this.apply();
    } else if (step % this.nevery === 0) this.apply();
  }

  /** post_run: reset yes restores the values of the run start (pair orig, charges as defined). */
  postRun(): void {
    if (!this.reset) return;
    const sys = this.sys, s = sys.state;
    const owners = new Set<Pair>();
    for (const p of this.pairs) {
      p.pairs.forEach(([i, j], k) => setPairValue(p.table, i, j, p.orig[k]));
      owners.add(p.table.owner);
    }
    for (const owner of owners) reinitPair(sys, owner);
    if (this.kspaces.length && sys.ff.kspace) sys.ff.kspace.scale = 1.0;
    // native restores both stored vectors once for each atom entry (the radius at the same density)
    for (let e = 0; e < this.atoms.length; e++) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        if (this.hasDiameter && s.radius && s.rmass) {
          const density = s.rmass[i] / sphereVolume(s.radius[i]);
          s.radius[i] = this.origR.get(s.id[i]) ?? s.radius[i];
          s.rmass[i] = sphereVolume(s.radius[i]) * density;
        }
        if (this.hasCharge) {
          const q = this.origQ.get(s.id[i]);
          if (q !== undefined) s.q[i] = q;
        }
      }
    }
    if (this.hasDiameter) sys.bump();
    if (this.hasCharge) refreshChargeSums(sys);
  }
}
