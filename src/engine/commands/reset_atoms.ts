import type { Handler } from './args';
import { int, yesno, keywords, initializing, missingArgs } from './args';
import { StyleError } from '../force/types';
import { hasMolecule, isMolecularStyle, nativeOrder } from '../atoms';
import { gplHook } from '../gplHooks';
import type { SimState, TopoList } from '../types';
import type { System } from '../system';

/*
 * reset_atoms property arguments — docs.lammps.org/reset_atoms.html:
 *   "property = id or image or mol"
 *   "reset_atoms id keyword value ..." with "keyword = sort", "sort value = yes or no"
 *   "reset_atoms image group-ID"
 *   "reset_atoms mol group-ID keyword value ..." with "keyword = compress or offset or single"
 *   "compress value = yes or no", "offset value = Noffset >= -1",
 *   "single value = yes or no to treat single atoms (no bonds) as molecules"
 *   "For id, the IDs of all the atoms are reset to contiguous values."
 *   "Note that the spatial sort requires communication of atom IDs and coordinates" (sort yes)
 *   "If the sort keyword is used with a setting of yes, then the assignment of new atom IDs will be the same no
 *   matter how many processors LAMMPS is running on."
 *   "The image property can only be used when the atom style supports bonds."
 *   "If the setting is yes (the default) and there are N molecules in the group, the new molecule IDs will be a
 *   set of N contiguous values."; "If the setting is no, the molecule ID of every atom in the molecule will be set
 *   to the smallest atom ID of any atom in the molecule."
 *   "If the group is not all and the largest molecule ID of atoms outside that group is M, then the new compressed
 *   molecule IDs will range from M+1 to M+N"; "If an Noffset >= 0 is specified, then the new compressed molecule
 *   IDs will range from Noffset+1 to Noffset+N."
 *   "The offset keyword is only used if the compress setting is yes."
 *
 * Measured with native LAMMPS (black box, one process):
 *  - id without sort: the new ID of an atom is its position in the native storage order plus one (read_data
 *    keeps the file order), and the bonded topology follows the atoms (w104cf_reset_id);
 *  - image: the image flags of a fragment (atoms joined by bonds over the WHOLE topology, not only the group)
 *    are shifted by floor((max + min) / 2) per dimension, where max and min are over all atoms of the fragment;
 *    only the group's atoms change (w104cf_reset_image_all, w104cf_reset_image_group: -3,-4 gives 1,0; 4,5 with
 *    a -1 partner gives 2,3,-3);
 *  - mol: molecules are bond fragments over the GROUP atoms only (an atom whose bond partner is outside the
 *    group is its own molecule), the units are ranked by their smallest atom ID, and an atom with no bond at all
 *    is a single (0 by default, its own ID rank with single yes). The offset M is the largest molecule ID of the
 *    atoms outside the group (w104cf_reset_mol: six variants written to files and compared).
 * sort yes: the spatial order of src/reset_atoms_id.cpp (b332f28, ResetAtomsID::sort and sort_bins), see
 * the GPL helper src/engine-gpl/reset_atoms/spatial_sort.ts (SpatialSortHook). Native wraps the atoms into the periodic box
 * first (domain->pbc()) for both settings.
 */

/** Union-find over atom indices (path halving). */
class Sets {
  private readonly parent: Int32Array;
  constructor(n: number) {
    this.parent = new Int32Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }
  find(i: number): number {
    const p = this.parent;
    while (p[i] !== i) { p[i] = p[p[i]]; i = p[i]; }
    return i;
  }
  union(a: number, b: number): void {
    const ra = this.find(a), rb = this.find(b);
    if (ra !== rb) this.parent[Math.max(ra, rb)] = Math.min(ra, rb);
  }
}

/** The bond list, after checking that every bond is an active one (type > 0). */
const activeBonds = (s: SimState): TopoList => {
  const b = s.topo.bonds;
  for (let e = 0; e < b.n; e++) {
    if (b.type[e] <= 0) throw new StyleError(`reset_atoms: bonds of type ${b.type[e]} (turned off or broken) are not supported by the browser engine`);
  }
  return b;
};

/** Index pairs of every bond (bonds naming an atom ID that is not present are skipped). */
const bondPairs = (sys: System, b: TopoList): [number, number][] => {
  const out: [number, number][] = [];
  for (let e = 0; e < b.n; e++) {
    const ia = sys.indexOfId(b.atoms[2 * e]), ib = sys.indexOfId(b.atoms[2 * e + 1]);
    if (ia >= 0 && ib >= 0) out.push([ia, ib]);
  }
  return out;
};

/**
 * The atom order of "sort yes" (src/reset_atoms_id.cpp at b332f28, the bin sort of ResetAtomsID::sort): the GPL
 * helper of the lazy HOOKS entry 'reset_atoms' (src/engine-gpl/reset_atoms/spatial_sort.ts). Arguments: positions,
 * atom count, dimension; the result lists the atom indices in the order that receives the IDs 1..N.
 */
export type SpatialSortHook = (x: Float64Array, n: number, dim: number) => number[];

/**
 * id: contiguous IDs in native storage order (sort no), or in the spatial order (sort yes); the bonded topology is
 * renumbered with the atoms. Native calls domain->pbc() before either (reset_atoms_id.cpp, command()).
 */
const resetIds = (sys: System, args: string[]): void => {
  // As native LAMMPS does (src/reset_atoms_id.cpp at b332f28, ResetAtomsID::command): the words are read in order and
  // an unknown word is "Unknown reset_atoms id keyword" (sort is the only keyword)
  let sortValue: string | null = null;
  for (let k = 0; k < args.length;) {
    if (args[k] !== 'sort') throw new StyleError(`Unknown reset_atoms id keyword: ${args[k]}`);
    if (k + 2 > args.length) throw new StyleError(`reset_atoms id: keyword 'sort' needs 1 value(s)`);
    sortValue = args[k + 1];
    k += 2;
  }
  const sorted = sortValue !== null && yesno(sortValue, 'reset_atoms id sort');
  const s = sys.state;
  sys.pbc();
  const order = sorted ? gplHook<SpatialSortHook>('reset_atoms')(s.x, s.n, s.dimension) : nativeOrder(s);
  const newId = new Map<number, number>();
  for (let k = 0; k < s.n; k++) newId.set(s.id[order[k]], k + 1);
  const renumber = (l: TopoList): void => {
    for (let q = 0; q < l.n * l.width; q++) {
      const v = newId.get(l.atoms[q]);
      if (v !== undefined) l.atoms[q] = v;
    }
  };
  const t = s.topo;
  for (const l of [t.bonds, t.angles, t.dihedrals, t.impropers]) renumber(l);
  const fresh = new Int32Array(s.n);
  for (let i = 0; i < s.n; i++) fresh[i] = newId.get(s.id[i]) ?? s.id[i];
  s.id.set(fresh);
  sys.atomsChanged();
};

/**
 * image group-ID: per fragment (bonds over the whole topology) the image flags of the group's atoms are shifted by
 * floor((max + min) / 2) of the fragment's flags in each dimension (measured, see the header).
 */
const resetImages = (sys: System, args: string[]): void => {
  const name = args[0];
  // As native LAMMPS does (src/reset_atoms_image.cpp ResetAtomsImage::command at b332f28): "if (narg < 1) utils::missing_cmd_args(FLERR, \"reset_atoms image\", error)"
  if (name === undefined) throw missingArgs('reset_atoms image');
  if (args.length > 1) throw new StyleError(`reset_atoms image: unexpected argument '${args[1]}'`);
  const s = sys.state;
  if (!isMolecularStyle(s.atomStyle)) throw new StyleError('reset_atoms image: the atom style has no bonds (reset_atoms image needs an atom style with bonds)');
  const bit = sys.groupBit(name);
  const sets = new Sets(s.n);
  for (const [ia, ib] of bondPairs(sys, activeBonds(s))) sets.union(ia, ib);
  const roots = new Int32Array(s.n);
  for (let i = 0; i < s.n; i++) roots[i] = sets.find(i);
  for (let d = 0; d < 3; d++) {
    const hi = new Map<number, number>(), lo = new Map<number, number>();
    for (let i = 0; i < s.n; i++) {
      const r = roots[i], v = s.image[3 * i + d];
      hi.set(r, Math.max(hi.get(r) ?? v, v));
      lo.set(r, Math.min(lo.get(r) ?? v, v));
    }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & bit)) continue;
      const r = roots[i];
      s.image[3 * i + d] -= Math.floor(((hi.get(r) ?? 0) + (lo.get(r) ?? 0)) / 2);
    }
  }
  sys.atomsChanged();
};

/**
 * mol group-ID [compress yes|no] [offset N] [single yes|no]: molecules are bond fragments of the group's atoms.
 * Every unit (a fragment, or a single atom when single yes) is ranked by its smallest atom ID; compress yes gives
 * rank + offset (offset -1: the largest molecule ID of the atoms outside the group), compress no gives the
 * smallest atom ID of the unit. Atoms with no bond at all are singles: 0 unless single yes.
 */
const resetMol = (sys: System, args: string[]): void => {
  const name = args[0];
  if (name === undefined) throw new StyleError('usage: reset_atoms mol group-ID keyword value ...');
  const kw = keywords(args.slice(1), { compress: 1, offset: 1, single: 1 }, 'reset_atoms mol');
  const compress = kw.has('compress') ? yesno(kw.get('compress')![0], 'reset_atoms mol compress') : true;
  const single = kw.has('single') ? yesno(kw.get('single')![0], 'reset_atoms mol single') : false;
  const offset = kw.has('offset') ? int(kw.get('offset')![0], 'reset_atoms mol offset') : -1;
  if (offset < -1) throw new StyleError(`reset_atoms mol: offset must be >= -1, got ${offset}`);
  const s = sys.state;
  if (!hasMolecule(s)) throw new StyleError('reset_atoms mol: the atom style has no molecule IDs');
  const bit = sys.groupBit(name);
  const pairs = bondPairs(sys, activeBonds(s));
  const bonded = new Uint8Array(s.n);
  for (const [ia, ib] of pairs) { bonded[ia] = 1; bonded[ib] = 1; }
  const sets = new Sets(s.n);
  for (const [ia, ib] of pairs) {
    if ((s.mask[ia] & bit) && (s.mask[ib] & bit)) sets.union(ia, ib);
  }
  // the units of the group: root -> smallest atom ID, and whether it is a single atom
  const minId = new Map<number, number>();
  const isSingle = new Map<number, boolean>();
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & bit)) continue;
    const r = sets.find(i);
    const id = s.id[i];
    if (!minId.has(r) || id < minId.get(r)!) minId.set(r, id);
    isSingle.set(r, (isSingle.get(r) ?? true) && !bonded[i]);
  }
  const units = [...minId.keys()].filter((r) => single || !isSingle.get(r));
  units.sort((a, b) => minId.get(a)! - minId.get(b)!);
  let base = 0;
  if (compress) {
    if (offset >= 0) base = offset;
    else {
      for (let i = 0; i < s.n; i++) if (!(s.mask[i] & bit)) base = Math.max(base, s.molecule[i]);
    }
  }
  const value = new Map<number, number>();
  units.forEach((r, k) => value.set(r, compress ? base + k + 1 : minId.get(r)!));
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & bit)) continue;
    s.molecule[i] = value.get(sets.find(i)) ?? 0;
  }
  sys.atomsChanged();
};

/** reset_atoms property arguments (id, image or mol). */
const resetAtoms: Handler = ({ sys }, a) => {
  const prop = a[0];
  if (prop === 'id') resetIds(sys, a.slice(1));
  else if (prop === 'image') resetImages(sys, a.slice(1));
  else if (prop === 'mol') resetMol(sys, a.slice(1));
  else throw new StyleError(`Illegal reset_atoms command: property must be id, image or mol, got '${prop ?? ''}'`);
};

// reset_atoms initializes the system first (native lmp->init() in reset_atoms_id, _image and _mol at b332f28)
export const RESET_ATOMS_COMMANDS: Record<string, Handler> = { reset_atoms: initializing(resetAtoms) };
