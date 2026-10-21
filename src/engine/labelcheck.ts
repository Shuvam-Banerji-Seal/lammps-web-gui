import type { System } from './system';
import type { TopoList } from './types';
import { nativeOrder } from './atoms';
import { checkLabelsReport, improperSymmatoms, type CheckEntry, type CheckWhich, type LabelMap } from './labelmap';

/*
 * labelmap check_labels and molecule check_labels (labelmap.html, molecule.html): the type labels of the bonds,
 * angles, dihedrals and impropers checked against the labels of their atoms. As native LAMMPS does (b332f28):
 * - src/label_map.cpp check_labels runs at the end of Modify::setup (the setup of the next run or minimization),
 *   once, over each owning atom's lists in local order; with newton_bond on a bond belongs to its first atom and an
 *   angle, dihedral or improper to its second, with newton_bond off to each of its atoms;
 * - src/molecule.cpp check_labels runs when the template is read, over its atoms' lists built the same way from the
 *   file's order.
 */

/**
 * The entries of one kind in native's per-atom order: for each owner (in `owners` order) the entries it holds, in
 * list order. `holders(atoms)` gives the positions (0-based within the entry) of the atoms that hold it.
 */
const ownedOrder = (
  n: number, entryAtoms: (e: number) => number[], holders: (atoms: number[]) => number[], ownerKeys: number[],
  make: (e: number) => CheckEntry,
): CheckEntry[] => {
  const byOwner = new Map<number, number[]>();
  for (let e = 0; e < n; e++) {
    const at = entryAtoms(e);
    for (const h of holders(at)) {
      const key = at[h];
      if (!byOwner.has(key)) byOwner.set(key, []);
      byOwner.get(key)!.push(e);
    }
  }
  const out: CheckEntry[] = [];
  for (const key of ownerKeys) for (const e of byOwner.get(key) ?? []) out.push(make(e));
  return out;
};

/** Which atoms of an entry of width w hold it: the first (bonds) or second atom with newton_bond on, else all. */
const holdersOf = (w: number, newtonBond: boolean) => (atoms: number[]): number[] =>
  (newtonBond ? [w === 2 ? 0 : 1] : atoms.map((_, k) => k));

const emit = (sys: System, report: { warnings: string[]; logs: string[] }): void => {
  for (const w of report.warnings) sys.warn(w);
  for (const l of report.logs) sys.log(l);
};

/** The check of labelmap check_labels, at the end of a run's or minimization's setup (once). */
export const runLabelCheck = (sys: System): void => {
  const map = sys.labels;
  const which = map.checkWhich;
  if (!which || !map.defined) return;
  map.checkWhich = null;
  const s = sys.state;
  const order = nativeOrder(s);
  const ownerIds = Array.from(order, (i) => s.id[i]);
  const typeOf = (id: number): number => s.type[sys.indexOfId(id)];
  const lists: TopoList[] = [s.topo.bonds, s.topo.angles, s.topo.dihedrals, s.topo.impropers];
  const entries = lists.map((l) => {
    const atomsOf = (e: number): number[] => Array.from(l.atoms.subarray(e * l.width, (e + 1) * l.width));
    // native skips an entry whose type is below 1 (a broken bond)
    return ownedOrder(l.n, atomsOf, holdersOf(l.width, sys.newtonBond), ownerIds, (e) => {
      const atoms = atomsOf(e);
      return { type: l.type[e], atoms, types: atoms.map(typeOf) };
    }).filter((c) => c.type >= 1);
  });
  emit(sys, checkLabelsReport(map, which, entries, sys.newtonBond, improperSymmatoms(sys.ff.improper?.name), null));
};

/** The check of molecule check_labels on a template just read (its entries are [type, atom indices...]). */
export const moleculeLabelCheck = (
  sys: System, map: LabelMap, which: CheckWhich, id: string,
  t: { natoms: number; type: Int32Array; bonds: number[][]; angles: number[][]; dihedrals: number[][]; impropers: number[][] },
): void => {
  if (!map.defined) return;
  const owners = Array.from({ length: t.natoms }, (_, k) => k + 1);
  const entries = [t.bonds, t.angles, t.dihedrals, t.impropers].map((list) => {
    const w = list.length ? list[0].length - 1 : 2;
    return ownedOrder(list.length, (e) => list[e].slice(1), holdersOf(w, sys.newtonBond), owners, (e) => {
      const atoms = list[e].slice(1);
      return { type: list[e][0], atoms, types: atoms.map((a) => t.type[a - 1]) };
    });
  });
  emit(sys, checkLabelsReport(map, which, entries, sys.newtonBond, improperSymmatoms(sys.ff.improper?.name), id));
};
