/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MC/fix_bond_break.cpp and
 * src/MC/fix_bond_break.h (fix bond/break: the partner choice, the probability test, the special-list rebuild and the
 * angle, dihedral and improper removal). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_bondcb).
 *
 * Scope: keywords prob (fraction and seed). vizsteps (fix_modify, the image output) is not supported by the engine and
 * is refused by the base fix. Native's ghost checks have no meaning in the browser engine (one owner per atom).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import {
  intArg, numArg, oneTwoOrder, oneTwoRemove, pairKey, refreshTopology, topoEntries, writeTopo, type TopoEntry,
} from './bond_topo';

export class FixBondBreak extends Fix {
  readonly style = 'bond/break';
  private readonly btype: number;
  private readonly cutsq: number;
  private readonly fraction: number;
  private readonly rng: RanMars;
  /** bonds broken in the last event (fix srp/react reads and resets it, as native) */
  breakcount = 0;
  private breakcounttotal = 0;

  /** fix ID group-ID bond/break N btype rmin [prob fraction seed] (docs.lammps.org/fix_bond_break.html). */
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal fix bond/break command');
    const nevery = intArg(args[0], 'fix bond/break nevery');
    if (nevery <= 0) throw new StyleError('Illegal fix bond/break command');
    this.nevery = nevery;
    this.vectorFlag = true;
    this.sizeVector = 2;
    this.btype = intArg(args[1], 'fix bond/break bond type');
    if (this.btype < 1 || this.btype > sys.state.topo.nbondtypes) throw new StyleError('Invalid bond type in fix bond/break command');
    const cutoff = numArg(args[2], 'fix bond/break cutoff');
    if (cutoff < 0) throw new StyleError('Illegal fix bond/break command');
    this.cutsq = cutoff * cutoff;
    let fraction = 1;
    let seed = 12345;
    for (let i = 3; i < args.length;) {
      // As native LAMMPS does (src/MC/fix_bond_break.cpp FixBondBreak::FixBondBreak at b332f28): every keyword and value error is "Illegal fix bond/break command"
      if (args[i] !== 'prob') throw new StyleError('Illegal fix bond/break command');
      if (i + 3 > args.length) throw new StyleError('Illegal fix bond/break command');
      fraction = numArg(args[i + 1], 'fix bond/break prob fraction');
      seed = intArg(args[i + 2], 'fix bond/break prob seed');
      if (fraction < 0 || fraction > 1) throw new StyleError('Illegal fix bond/break command');
      if (seed <= 0) throw new StyleError('Illegal fix bond/break command');
      i += 3;
    }
    this.fraction = fraction;
    // RanMars(lmp, seed + me) with one process: me = 0
    this.rng = new RanMars(seed);
  }

  computeVector(i: number): number {
    return i === 0 ? this.breakcount : this.breakcounttotal;
  }

  private indexOf(tag: number): number {
    const k = this.sys.indexOfId(tag);
    if (k < 0) throw new StyleError('Fix bond/break needs ghost atoms from further away');
    return k;
  }

  /**
   * post_integrate (fix_bond_break.cpp): every nevery steps each atom picks the bond of type btype to a group partner
   * that is stretched past the cutoff and farthest away; a bond breaks when both atoms pick each other and the
   * probability draw (of the atom with the smaller ID) is below the fraction. Native draws one random number per atom
   * with a partner, in its local order, before the breaks are decided.
   */
  postIntegrate(): void {
    const s = this.sys.state;
    if (s.step % this.nevery !== 0) return;
    const n = s.n;
    const gb = this.groupBit;
    const x = s.x;
    const partner = new Int32Array(n);
    const distsq = new Float64Array(n);
    // bondlist: each bond of the type with both atoms in the group, at its closest image
    const B = s.topo.bonds;
    const d = [0, 0, 0];
    for (let e = 0; e < B.n; e++) {
      if (B.type[e] !== this.btype) continue;
      const i1 = this.indexOf(B.atoms[2 * e]), i2 = this.indexOf(B.atoms[2 * e + 1]);
      if (!(s.mask[i1] & gb) || !(s.mask[i2] & gb)) continue;
      d[0] = x[3 * i1] - x[3 * i2]; d[1] = x[3 * i1 + 1] - x[3 * i2 + 1]; d[2] = x[3 * i1 + 2] - x[3 * i2 + 2];
      this.sys.geom.minimumImage(d);
      const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (rsq <= this.cutsq) continue;
      if (rsq > distsq[i1]) { partner[i1] = s.id[i2]; distsq[i1] = rsq; }
      if (rsq > distsq[i2]) { partner[i2] = s.id[i1]; distsq[i2] = rsq; }
    }
    const ord = nativeOrder(s);
    const prob = new Float64Array(n);
    if (this.fraction < 1) {
      for (let k = 0; k < n; k++) {
        const i = ord[k];
        if (partner[i]) prob[i] = this.rng.uniform();
      }
    }
    // mutual partners that pass the probability test (the atom with the smaller ID draws)
    const broken: [number, number][] = [];
    for (let k = 0; k < n; k++) {
      const i = ord[k];
      if (!partner[i]) continue;
      const j = this.indexOf(partner[i]);
      if (partner[j] !== s.id[i]) continue;
      if (this.fraction < 1) {
        const lo = s.id[i] < s.id[j] ? i : j;
        if (prob[lo] >= this.fraction) continue;
      }
      if (s.id[i] < s.id[j]) broken.push([s.id[i], s.id[j]]);
    }
    this.breakcount = broken.length;
    this.breakcounttotal += this.breakcount;
    if (this.breakcount === 0) return;
    const order = oneTwoOrder(s);
    for (const [a, b] of broken) oneTwoRemove(order, a, b);
    this.removeBonds(broken);
    this.removeAngular(broken);
    refreshTopology(this.sys);
  }

  /** Removes one bond of the type between each broken pair (the first stored one, as native). */
  private removeBonds(broken: [number, number][]): void {
    const B = this.sys.state.topo.bonds;
    const entries = topoEntries(B);
    const firstOf = new Map<string, number>();
    entries.forEach((t, e) => {
      if (t.ty !== this.btype) return;
      const key = pairKey(t.at[0], t.at[1]);
      if (!firstOf.has(key)) firstOf.set(key, e);
    });
    const drop = new Set<number>();
    for (const [a, b] of broken) {
      const e = firstOf.get(pairKey(a, b));
      if (e !== undefined) drop.add(e);
    }
    writeTopo(B, entries.filter((_, e) => !drop.has(e)));
  }

  /**
   * Removes the angles, dihedrals and impropers that contain a broken bond as one of their bonds (break_angles,
   * break_dihedrals, break_impropers): an angle has the bonds 1-2 and 2-3, a dihedral 1-2, 2-3 and 3-4, an improper
   * 1-2, 1-3 and 1-4 (the central atom is 1).
   */
  private removeAngular(broken: [number, number][]): void {
    const st = this.sys.state.topo;
    const has = new Set(broken.map(([a, b]) => pairKey(a, b)));
    const hit = (a: number, b: number) => has.has(pairKey(a, b));
    if (st.angles.n) {
      writeTopo(st.angles, topoEntries(st.angles).filter((t) => !(hit(t.at[0], t.at[1]) || hit(t.at[1], t.at[2]))));
    }
    if (st.dihedrals.n) {
      writeTopo(st.dihedrals, topoEntries(st.dihedrals).filter((t) => !(hit(t.at[0], t.at[1]) || hit(t.at[1], t.at[2]) || hit(t.at[2], t.at[3]))));
    }
    if (st.impropers.n) {
      writeTopo(st.impropers, topoEntries(st.impropers).filter((t: TopoEntry) => !(hit(t.at[0], t.at[1]) || hit(t.at[0], t.at[2]) || hit(t.at[0], t.at[3]))));
    }
  }
}
