/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the topology edits of src/MC/fix_bond_create.cpp
 * and src/MC/fix_bond_break.cpp (update_topology, create_angles, create_dihedrals, create_impropers, break_angles,
 * break_dihedrals, break_impropers). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Shared helpers of fix bond/create and fix bond/break: list access by atom ID, the unordered bond key of the broken and
 * created bonds, and the topology refresh after a change (special lists and neighbor rebuild, tail corrections kept).
 */
import { inumeric, numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { SimState, TopoList } from '../../engine/types';
import { StyleError } from '../../engine/force/types';

export type TopoKind = 'bonds' | 'angles' | 'dihedrals' | 'impropers';

/** One topology entry: its type and its atom IDs (1-2, 1-2-3, 1-2-3-4 in the list's order). */
export interface TopoEntry {
  ty: number;
  at: number[];
}

/** Unordered key of a bond between atom IDs a and b. */
export const pairKey = (a: number, b: number): string => (a < b ? `${a} ${b}` : `${b} ${a}`);

/** The entries of a topology list. */
export const topoEntries = (L: TopoList): TopoEntry[] => {
  const w = L.width;
  const out: TopoEntry[] = [];
  for (let e = 0; e < L.n; e++) {
    const at: number[] = [];
    for (let k = 0; k < w; k++) at.push(L.atoms[w * e + k]);
    out.push({ ty: L.type[e], at });
  }
  return out;
};

/** Replaces the contents of a topology list with these entries. */
export const writeTopo = (L: TopoList, entries: TopoEntry[]): void => {
  const w = L.width;
  const type = new Int32Array(entries.length);
  const atoms = new Int32Array(entries.length * w);
  entries.forEach((x, e) => {
    if (x.at.length !== w) throw new StyleError('fix bond topology: entry with the wrong number of atoms');
    type[e] = x.ty;
    for (let k = 0; k < w; k++) atoms[w * e + k] = x.at[k];
  });
  L.n = entries.length;
  L.type = type;
  L.atoms = atoms;
};

/**
 * Re-derives the neighbor lists and the force field's special lists from the bonds after a topology change. Native
 * keeps the tail corrections of the last run setup through a bond change (fix bond/react and fix bond/create measured
 * with native LAMMPS: the pair tail terms are not re-initialised), so they are restored after the refresh.
 */
export const refreshTopology = (sys: System): void => {
  const ff = sys.ff;
  const etail = ff.etailV, ptail = ff.ptailV;
  sys.atomsChanged();
  ff.etailV = etail;
  ff.ptailV = ptail;
};

/** An integer argument of a fix, read as native's utils::inumeric (fix_bond_break.cpp, fix_bond_create.cpp at b332f28). */
export const intArg = (w: string | undefined, _what: string): number => inumeric(w);

/** A numeric argument of a fix, read as native's utils::numeric. */
export const numArg = (w: string | undefined, _what: string): number => numeric(w);

/*
 * The 1-2 neighbor order of each atom, as native LAMMPS keeps it in atom->special (its order matters for the impropers
 * that fix bond/create induces: the three outer atoms are taken in the list order of the central atom). Native builds
 * it when the data are read (Special::onetwo_build_newton, src/special.cpp at b332f28): atoms in their file order, and
 * each bond, stored with its first atom (Atom::data_bonds, src/atom.cpp), appends the second atom to the first and the
 * first to the second. A created bond is appended to the list of each of its atoms (fix_bond_create.cpp, the insertion
 * at nspecial[i][0]); a broken bond is removed and the others keep their order (fix_bond_break.cpp).
 *
 * The order is kept per state. It is checked against the bond list on every use: where the topology changed by other
 * means (delete_bonds, create_bonds, bond/react), the lists are rebuilt in the read-time order.
 */
const oneTwoStore = new WeakMap<SimState, Map<number, number[]>>();

/** The 1-2 neighbor lists by atom ID, in native order (see above). */
export const oneTwoOrder = (s: SimState): Map<number, number[]> => {
  const B = s.topo.bonds;
  const deg = new Map<number, number>();
  const bondKeys = new Set<string>();
  let nbonds = 0;
  for (let e = 0; e < B.n; e++) {
    if (B.type[e] === 0) continue;
    const a = B.atoms[2 * e], b = B.atoms[2 * e + 1];
    nbonds++;
    bondKeys.add(pairKey(a, b));
    deg.set(a, (deg.get(a) ?? 0) + 1);
    deg.set(b, (deg.get(b) ?? 0) + 1);
  }
  const cur = oneTwoStore.get(s);
  if (cur) {
    let total = 0, ok = true;
    for (const [tag, list] of cur) {
      total += list.length;
      if (list.length !== (deg.get(tag) ?? 0)) { ok = false; break; }
      for (const p of list) if (!bondKeys.has(pairKey(tag, p))) { ok = false; break; }
      if (!ok) break;
    }
    if (ok && total === 2 * nbonds) return cur;
  }
  // read-time order: bonds grouped by their first atom's index (atoms in file order), bonds in file order within a group
  const index = new Map<number, number>();
  for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
  const rank = (e: number) => index.get(B.atoms[2 * e]) ?? -1;
  const edges: number[] = [];
  for (let e = 0; e < B.n; e++) if (B.type[e] !== 0) edges.push(e);
  edges.sort((x, y) => rank(x) - rank(y));
  const order = new Map<number, number[]>();
  const list = (t: number) => {
    let l = order.get(t);
    if (!l) { l = []; order.set(t, l); }
    return l;
  };
  for (const e of edges) {
    const a = B.atoms[2 * e], b = B.atoms[2 * e + 1];
    list(a).push(b);
    list(b).push(a);
  }
  oneTwoStore.set(s, order);
  return order;
};

/** The 1-2 neighbors of one atom in native order (empty when it has none). */
export const oneTwoOf = (order: Map<number, number[]>, tag: number): number[] => order.get(tag) ?? [];

/** Appends b to the 1-2 list of a only (the visit of atom a in fix bond/create). */
export const oneTwoAppend = (order: Map<number, number[]>, a: number, b: number): void => {
  let l = order.get(a);
  if (!l) { l = []; order.set(a, l); }
  l.push(b);
};

/** Removes the bond a-b from the 1-2 lists of both atoms (the order of the others is kept). */
export const oneTwoRemove = (order: Map<number, number[]>, a: number, b: number): void => {
  for (const [x, y] of [[a, b], [b, a]] as const) {
    const l = order.get(x);
    if (!l) continue;
    const k = l.indexOf(y);
    if (k >= 0) l.splice(k, 1);
  }
};
