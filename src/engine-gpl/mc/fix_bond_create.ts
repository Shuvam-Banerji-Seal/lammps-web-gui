/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MC/fix_bond_create.cpp and
 * src/MC/fix_bond_create.h (fix bond/create: the partner choice, the iparam/jparam bond limits and type changes, the
 * probability test, the special-list rebuild, and the angles, dihedrals and impropers induced by a new bond). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_bondcb).
 *
 * Scope: keywords iparam, jparam, prob, atype, dtype, itype and molecule off|inter|intra. The aconstrain keyword belongs
 * to fix bond/create/angle (src/MC/fix_bond_create_angle.cpp), which is not ported: here it is refused, as native refuses
 * it for fix bond/create. Not made: native's init error when the fix cutoff exceeds the pair cutoff, and its per-atom
 * storage limits (bonds, angles, dihedrals and impropers per atom, the special list size); the engine has no per-atom
 * storage, so those errors cannot arise here.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { inumeric } from '../../engine/force/util';
import { missingArgs } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { buildSpecial, nativeOrder, type SpecialList } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import {
  intArg, numArg, oneTwoAppend, oneTwoOf, oneTwoOrder, pairKey, refreshTopology, topoEntries, writeTopo, type TopoEntry,
} from './bond_topo';

const BIG = 1.0e20;

export class FixBondCreate extends Fix {
  readonly style = 'bond/create';
  private readonly iat: number;
  private readonly jat: number;
  private readonly cutsq: number;
  private readonly btype: number;
  private imax = 0;
  private inew: number;
  private jmax = 0;
  private jnew: number;
  private fraction = 1;
  private atype = 0;
  private dtype = 0;
  private ityp = 0;
  private molecule: 'off' | 'inter' | 'intra' = 'off';
  private readonly rng: RanMars;
  /** Bonds of the type per atom ID (native bondcount): counted at the first run setup, then incremented by this fix. */
  private readonly count = new Map<number, number>();
  private counted = false;
  /** bonds created in the last event (fix srp/react reads and resets it, as native) */
  createcount = 0;
  private createcounttotal = 0;

  /** fix ID group-ID bond/create N itype jtype rmin btype keyword values ... (docs.lammps.org/fix_bond_create.html). */
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 5) throw new StyleError('Illegal fix bond/create command: missing arguments');
    // As native LAMMPS does (src/MC/fix_bond_create.cpp FixBondCreate::FixBondCreate at b332f28): utils::inumeric reads nevery
    const nevery = inumeric(args[0]);
    if (nevery <= 0) throw new StyleError(`Illegal fix bond/create nevery value ${nevery}`);
    this.nevery = nevery;
    this.vectorFlag = true;
    this.sizeVector = 2;
    const ntypes = sys.state.ntypes;
    this.iat = intArg(args[1], 'fix bond/create itype');
    this.jat = intArg(args[2], 'fix bond/create jtype');
    if (this.iat < 1 || this.iat > ntypes || this.jat < 1 || this.jat > ntypes) {
      throw new StyleError('Invalid atom type in fix bond/create command');
    }
    const cutoff = numArg(args[3], 'fix bond/create cutoff');
    if (cutoff < 0) throw new StyleError(`Illegal fix bond/create cutoff value ${cutoff}`);
    this.cutsq = cutoff * cutoff;
    this.btype = intArg(args[4], 'fix bond/create bond type');
    if (this.btype < 1 || this.btype > sys.state.topo.nbondtypes) throw new StyleError(`Invalid bond type ${this.btype} in fix bond/create command`);
    this.inew = this.iat;
    this.jnew = this.jat;
    let seed = 12345;
    for (let i = 5; i < args.length;) {
      const key = args[i];
      const need = (k: number) => {
        if (i + k > args.length) throw missingArgs(`fix bond/create ${key}`);
      };
      if (key === 'iparam' || key === 'jparam') {
        need(3);
        const max = intArg(args[i + 1], `fix bond/create ${key} maxbond`);
        const nt = intArg(args[i + 2], `fix bond/create ${key} newtype`);
        if (max < 0) throw new StyleError(`Invalid fix bond/create ${key} maxbond value ${max}`);
        if (nt < 1 || nt > ntypes) throw new StyleError(`Invalid atom type ${nt} in fix bond/create ${key} command`);
        if (key === 'iparam') { this.imax = max; this.inew = nt; } else { this.jmax = max; this.jnew = nt; }
        i += 3;
      } else if (key === 'prob') {
        need(3);
        this.fraction = numArg(args[i + 1], 'fix bond/create prob fraction');
        seed = intArg(args[i + 2], 'fix bond/create prob seed');
        if (this.fraction < 0 || this.fraction > 1) throw new StyleError(`Invalid fix bond/create prob fraction value ${this.fraction}`);
        if (seed <= 0) throw new StyleError(`Invalid fix bond/create prob seed value ${seed}`);
        i += 3;
      } else if (key === 'atype' || key === 'dtype' || key === 'itype') {
        need(2);
        const t = intArg(args[i + 1], `fix bond/create ${key}`);
        if (t < 0) throw new StyleError(`Invalid fix bond/create ${key} value ${t}`);
        if (key === 'atype') this.atype = t; else if (key === 'dtype') this.dtype = t; else this.ityp = t;
        i += 2;
      } else if (key === 'molecule') {
        need(2);
        const w = args[i + 1];
        if (w !== 'off' && w !== 'inter' && w !== 'intra') throw new StyleError(`Unknown option ${w} for fix bond/create molecule`);
        this.molecule = w;
        i += 2;
      } else if (key === 'aconstrain') {
        throw new StyleError('Unknown fix bond/create keyword aconstrain (it belongs to fix bond/create/angle, which the browser engine does not provide)');
      } else {
        throw new StyleError(`Unknown fix bond/create keyword ${key}`);
      }
    }
    if (this.iat === this.jat && (this.imax !== this.jmax || this.inew !== this.jnew)) {
      throw new StyleError('Inconsistent iparam/jparam values in fix bond/create command');
    }
    // RanMars(lmp, seed + me) with one process: me = 0
    this.rng = new RanMars(seed);
  }

  computeVector(i: number): number {
    return i === 0 ? this.createcount : this.createcounttotal;
  }

  private indexOf(tag: number): number {
    const k = this.sys.indexOfId(tag);
    if (k < 0) throw new StyleError('Fix bond/create needs ghost atoms from further away');
    return k;
  }

  /** init / setup checks of native (angle, dihedral and improper types against the style) and the bond counts. */
  setup(): void {
    const ff = this.sys.ff, topo = this.sys.state.topo;
    if (this.atype && ff.angle && this.atype > topo.nangletypes) throw new StyleError('Fix bond/create angle type is invalid');
    if (this.dtype && ff.dihedral && this.dtype > topo.ndihedraltypes) throw new StyleError('Fix bond/create dihedral type is invalid');
    if (this.ityp && ff.improper && this.ityp > topo.nimpropertypes) throw new StyleError('Fix bond/create improper type is invalid');
    if (ff.improper && (ff.improper.name.includes('class2') || ff.improper.name.includes('ring'))) {
      throw new StyleError(`Cannot yet use fix bond/create with improper style ${ff.improper.name}`);
    }
    if (!this.counted) {
      // count the bonds of the type stored with each atom (both atoms of a bond, as newton_bond on)
      this.counted = true;
      const B = this.sys.state.topo.bonds;
      for (let e = 0; e < B.n; e++) {
        if (B.type[e] !== this.btype) continue;
        for (let k = 0; k < 2; k++) {
          const tag = B.atoms[2 * e + k];
          this.count.set(tag, (this.count.get(tag) ?? 0) + 1);
        }
      }
    }
    super.setup();
  }

  /**
   * post_integrate (fix_bond_create.cpp): every nevery steps each atom picks the closest eligible partner within the
   * cutoff (a neighbor of the group, of the atom types, below the bond limits, not already a 1-2 neighbor), a new bond
   * forms when both atoms pick each other and the probability draw passes, and the special lists, the types at the
   * bond limit and the induced angles, dihedrals and impropers follow. Native draws one random number per atom with a
   * partner, in its local order, before the bonds are made.
   */
  postIntegrate(): void {
    const s = this.sys.state;
    if (s.step % this.nevery !== 0) return;
    const n = s.n;
    const gb = this.groupBit;
    const ff = this.sys.ff;
    const nb = this.sys.nb;
    const sp = buildSpecial(s, { angle: ff.special.angle, dihedral: ff.special.dihedral });
    // the 1-2 lists in native order: a new bond is appended to the lists of its atoms, and the induced impropers take
    // their outer atoms in these lists' order
    const order = oneTwoOrder(s);
    const partner = new Int32Array(n);
    const distsq = new Float64Array(n).fill(BIG);
    const cutoff = Math.sqrt(this.cutsq);
    // the ghost images follow the positions of this step (native: comm->forward_comm() before the list is built,
    // since post_integrate runs before the Verlet communication)
    nb.forwardComm(s, this.sys.geom);
    const list = nb.occasionalList(s, cutoff, false, s.ntypes);
    const xa = nb.xall, owner = nb.owner;
    const box = s.box, per = box.periodic;
    const hx = 0.5 * (box.hi[0] - box.lo[0]), hy = 0.5 * (box.hi[1] - box.lo[1]), hz = 0.5 * (box.hi[2] - box.lo[2]);
    const kspace = Boolean(ff.kspace);
    const lj = ff.special.lj, coul = ff.special.coul;
    const cnt = (tag: number) => this.count.get(tag) ?? 0;
    for (let i = 0; i < list.inum; i++) {
      if (!(s.mask[i] & gb)) continue;
      const lo = list.firstneigh[i], hi = lo + list.numneigh[i];
      for (let k = lo; k < hi; k++) {
        const jx = list.neighbors[k];
        const j = owner[jx];
        if (!(s.mask[j] & gb)) continue;
        if (this.molecule === 'inter' && s.molecule[i] === s.molecule[j]) continue;
        if (this.molecule === 'intra' && s.molecule[i] !== s.molecule[j]) continue;
        const itype = s.type[i], jtype = s.type[j];
        let possible = false;
        if (itype === this.iat && jtype === this.jat) {
          if ((this.imax === 0 || cnt(s.id[i]) < this.imax) && (this.jmax === 0 || cnt(s.id[j]) < this.jmax)) possible = true;
        } else if (itype === this.jat && jtype === this.iat) {
          if ((this.jmax === 0 || cnt(s.id[i]) < this.jmax) && (this.imax === 0 || cnt(s.id[j]) < this.imax)) possible = true;
        }
        if (!possible) continue;
        // no duplicate bond: the partner must not be a 1-2 neighbor of atom i
        const sOrd = specialOrder(sp, i, s.id[j]);
        if (sOrd === 1) continue;
        // the neighbor list of native LAMMPS excludes a special pair whose weights are both zero (no kspace);
        // a special partner seen through a far periodic image is an ordinary neighbor
        if (sOrd > 0 && !kspace && lj[sOrd - 1] === 0 && coul[sOrd - 1] === 0) {
          const fx = xa[3 * jx] - xa[3 * i], fy = xa[3 * jx + 1] - xa[3 * i + 1], fz = xa[3 * jx + 2] - xa[3 * i + 2];
          const far = (per[0] && Math.abs(fx) > hx) || (per[1] && Math.abs(fy) > hy) || (per[2] && Math.abs(fz) > hz);
          if (!far) continue;
        }
        const dx = xa[3 * i] - xa[3 * jx], dy = xa[3 * i + 1] - xa[3 * jx + 1], dz = xa[3 * i + 2] - xa[3 * jx + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= this.cutsq) continue;
        if (rsq < distsq[i]) { partner[i] = s.id[j]; distsq[i] = rsq; }
        if (rsq < distsq[j]) { partner[j] = s.id[i]; distsq[j] = rsq; }
      }
    }
    const ord = nativeOrder(s);
    const prob = new Float64Array(n);
    if (this.fraction < 1) {
      for (let k = 0; k < n; k++) {
        const i = ord[k];
        if (partner[i]) prob[i] = this.rng.uniform();
      }
    }
    // each atom visits once: a mutual partner pair passing the probability test (the smaller ID draws) makes a bond;
    // the bond is stored once, both atoms count it and an atom at its bond limit takes the new type
    const created: [number, number][] = [];
    for (let k = 0; k < n; k++) {
      const i = ord[k];
      if (!partner[i]) continue;
      const j = this.indexOf(partner[i]);
      if (partner[j] !== s.id[i]) continue;
      if (this.fraction < 1) {
        const lo = s.id[i] < s.id[j] ? i : j;
        if (prob[lo] >= this.fraction) continue;
      }
      if (s.id[i] < s.id[j]) created.push([s.id[i], s.id[j]]);
      oneTwoAppend(order, s.id[i], s.id[j]);
      const c = cnt(s.id[i]) + 1;
      this.count.set(s.id[i], c);
      if (s.type[i] === this.iat) {
        if (c === this.imax) s.type[i] = this.inew;
      } else if (c === this.jmax) {
        s.type[i] = this.jnew;
      }
    }
    this.createcount = created.length;
    this.createcounttotal += this.createcount;
    if (!created.length) return;
    const B = s.topo.bonds;
    writeTopo(B, [...topoEntries(B), ...created.map(([a, b]) => ({ ty: this.btype, at: [a, b] }))]);
    this.inducedTopology(created, order);
    refreshTopology(this.sys);
  }

  /**
   * update_topology with create_angles, create_dihedrals and create_impropers: the entries that contain a new bond
   * as one of their bonds. Native walks the 1-2 neighbors of each atom (the new bonds included), so an angle is
   * centred on an atom of a new bond, a dihedral is stored once in the orientation with the middle atom's ID below
   * the other one, and an improper is centred on the atom of the new bond.
   */
  private inducedTopology(created: [number, number][], order: Map<number, number[]>): void {
    const s = this.sys.state, st = s.topo, ff = this.sys.ff;
    const B = st.bonds;
    const nbrs = (t: number): number[] => oneTwoOf(order, t);
    const newBond = new Set(created.map(([a, b]) => pairKey(a, b)));
    const isNew = (a: number, b: number) => newBond.has(pairKey(a, b));
    const centers = new Set<number>();
    for (const [a, b] of created) { centers.add(a); centers.add(b); }
    const angles: TopoEntry[] = [], dihedrals: TopoEntry[] = [], impropers: TopoEntry[] = [];
    const doAngles = this.atype > 0 && Boolean(ff.angle);
    const doDihedrals = this.dtype > 0 && Boolean(ff.dihedral);
    const doImpropers = this.ityp > 0 && Boolean(ff.improper);
    for (const k of nativeOrder(s)) {
      const m = s.id[k];
      if (!centers.has(m)) continue;
      const N = nbrs(m);
      if (doAngles) {
        for (let a = 0; a < N.length; a++) {
          for (let b = a + 1; b < N.length; b++) {
            if (isNew(N[a], m) || isNew(m, N[b])) angles.push({ ty: this.atype, at: [N[a], m, N[b]] });
          }
        }
      }
      if (doImpropers) {
        for (let a = 0; a < N.length; a++) {
          for (let b = a + 1; b < N.length; b++) {
            for (let c = b + 1; c < N.length; c++) {
              if (isNew(m, N[a]) || isNew(m, N[b]) || isNew(m, N[c])) impropers.push({ ty: this.ityp, at: [m, N[a], N[b], N[c]] });
            }
          }
        }
      }
    }
    if (doDihedrals) {
      for (let e = 0; e < B.n; e++) {
        if (B.type[e] === 0) continue;
        const p = B.atoms[2 * e], q = B.atoms[2 * e + 1];
        const b = Math.min(p, q), c = Math.max(p, q);
        for (const a of nbrs(b)) {
          if (a === c) continue;
          for (const d of nbrs(c)) {
            if (d === b || d === a) continue;
            if (isNew(a, b) || isNew(b, c) || isNew(c, d)) dihedrals.push({ ty: this.dtype, at: [a, b, c, d] });
          }
        }
      }
    }
    if (angles.length) writeTopo(st.angles, [...topoEntries(st.angles), ...angles]);
    if (dihedrals.length) writeTopo(st.dihedrals, [...topoEntries(st.dihedrals), ...dihedrals]);
    if (impropers.length) writeTopo(st.impropers, [...topoEntries(st.impropers), ...impropers]);
  }
}

/** The special order (1, 2 or 3) of the closest special relation of atom index i to the atom ID tag, 0 when none. */
const specialOrder = (sp: SpecialList, i: number, tag: number): number => {
  let best = 0;
  for (let k = sp.offset[i]; k < sp.offset[i + 1]; k++) {
    if (sp.partner[k] !== tag) continue;
    const o = sp.order[k];
    if (best === 0 || o < best) best = o;
  }
  return best;
};
