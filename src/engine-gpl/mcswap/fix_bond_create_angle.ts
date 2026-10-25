/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MC/fix_bond_create_angle.cpp and
 * src/MC/fix_bond_create_angle.h (Joao Gregorio, Oliver Henrich, University of Strathclyde, Glasgow, UK), on top of
 * src/MC/fix_bond_create.cpp and src/MC/fix_bond_create.h at b332f28. LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix bond/create/angle (docs.lammps.org/fix_bond_create.html, the aconstrain keyword): fix bond/create whose candidate
 * partners must also satisfy an angle constraint. This file is the fix bond/create port of src/engine-gpl/mc/
 * fix_bond_create.ts with the constrain() call of native added to the partner loop (native: constrainflag set by
 * aconstrain, and constrain(i, j, amin, amax) after the cutoff test). The rest of the algorithm is that port's.
 *
 * Constraint (FixBondCreateAngle::constrain, src/MC/fix_bond_create_angle.cpp): with no 1-2 neighbours on either atom
 * the bond passes; with 1-2 neighbours, the angle at each atom between the other atom and its first 1-2 neighbour
 * (special[i][0], read from the 1-2 list in native order, positions of the owned atoms, no minimum image) must lie in
 * [amin, amax]; when only one atom has neighbours only its angle is tested; when both do, both angles must pass.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { inumeric, missingArgs } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { buildSpecial, nativeOrder, type SpecialList } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import {
  intArg, numArg, oneTwoAppend, oneTwoOf, oneTwoOrder, pairKey, refreshTopology, topoEntries, writeTopo, type TopoEntry,
} from '../mc/bond_topo';
import { acos as libmAcos } from '../libm/acos';

const BIG = 1.0e20;

export class FixBondCreateAngle extends Fix {
  readonly style = 'bond/create/angle';
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
  /** aconstrain: the angle range in radians (native amin, amax) and whether it applies */
  private amin = 0;
  private amax = 0;
  private constrainflag = false;
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
    // As native LAMMPS does (src/MC/fix_bond_create.cpp FixBondCreate::FixBondCreate at b332f28, which bond/create/angle
    // uses): utils::inumeric reads nevery
    const nevery = inumeric(args[0]);
    // the message names the style as invoked (src/MC/fix_bond_create.cpp at b332f28: "Illegal fix {} nevery value {}")
    if (nevery <= 0) throw new StyleError(`Illegal fix bond/create/angle nevery value ${nevery}`);
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
        if (i + k > args.length) throw missingArgs(`fix bond/create/angle ${key}`);
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
        // native: amin is utils::numeric, amax utils::inumeric; both in degrees, then converted to radians
        need(3);
        const amin0 = numArg(args[i + 1], 'fix bond/create/angle aconstrain amin');
        const amax0 = intArg(args[i + 2], 'fix bond/create/angle aconstrain amax');
        if (amin0 >= amax0) throw new StyleError('Illegal fix bond/create/angle aconstrain command');
        if (amin0 < 0 || amin0 > 180) throw new StyleError(`Invalid fix bond/create/angle aconstrain amin value ${amin0}`);
        if (amax0 < 0 || amax0 > 180) throw new StyleError(`Invalid fix bond/create/angle aconstrain amax value ${amax0}`);
        this.amin = (Math.PI / 180.0) * amin0;
        this.amax = (Math.PI / 180.0) * amax0;
        this.constrainflag = true;
        i += 3;
      } else {
        throw new StyleError(`Unknown fix bond/create/angle keyword ${key}`);
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

  /**
   * FixBondCreateAngle::constrain (src/MC/fix_bond_create_angle.cpp at b332f28): i and the candidate j (jx is its index
   * in the ghost-inclusive arrays, j its owned index); order is the 1-2 lists before this event's bonds.
   */
  private constrain(i: number, jx: number, j: number, order: Map<number, number[]>): boolean {
    const s = this.sys.state;
    const xa = this.sys.nb.xall;
    const li = order.get(s.id[i]) ?? [];
    const lj = order.get(s.id[j]) ?? [];
    const xi = [xa[3 * i], xa[3 * i + 1], xa[3 * i + 2]];
    const xj = [xa[3 * jx], xa[3 * jx + 1], xa[3 * jx + 2]];
    const ownPos = (tag: number): number[] => {
      const k = this.indexOf(tag);
      return [s.x[3 * k], s.x[3 * k + 1], s.x[3 * k + 2]];
    };
    // the angle at c between (c - o) and (c - p), as native computes it (acos of the normalised dot product)
    const angleAt = (c: number[], o: number[], p: number[]): number => {
      const v1x = c[0] - o[0], v1y = c[1] - o[1], v1z = c[2] - o[2];
      const v2x = c[0] - p[0], v2y = c[1] - p[1], v2z = c[2] - p[2];
      return libmAcos((v1x * v2x + v1y * v2y + v1z * v2z) /
        (Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z) * Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)));
    };
    const inRange = (a: number): boolean => this.amin <= a && a <= this.amax;
    if (li.length === 0 && lj.length === 0) return true;
    if (li.length !== 0 && lj.length === 0) return inRange(angleAt(xi, xj, ownPos(li[0])));
    if (li.length === 0 && lj.length !== 0) return inRange(angleAt(xj, xi, ownPos(lj[0])));
    const a1 = angleAt(xi, xj, ownPos(li[0]));
    const a2 = angleAt(xj, xi, ownPos(lj[0]));
    return inRange(a1) && inRange(a2);
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
        // fix bond/create/angle: the angle constraint of the candidate pair (native constrain, after the cutoff test)
        if (this.constrainflag && !this.constrain(i, jx, j, order)) continue;
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
