import type { AtomStyle, SimBox, SimState, TopoList, Topology, UnitSystem } from './types';
import { makeBox, type BoxInit } from './domain';
import { StyleError } from './force/types';

/*
 * Per-atom storage: creating the state, appending and deleting atoms, the
 * ID -> index map, and bonded topology (stored by atom ID).
 *
 * docs.lammps.org/atom_style.html: "atomic" carries "only the default
 * values" (id, type, x, v, image), "charge" adds "charge", "bond" "bonds and
 * molecules", "angle" "bonds and angles", "molecular" "bonds, angles,
 * dihedrals, impropers", "full" "molecular + charge". Every array exists in
 * every style here; the style only decides what read_data / write_data /
 * set accept and print. "sphere" is "atomic + radius, rmass, omega, torque";
 * those four arrays exist only for it (null otherwise).
 */

/** Styles with bond topology and molecule IDs. */
export const isMolecularStyle = (st: AtomStyle): boolean => st === 'bond' || st === 'angle' || st === 'molecular' || st === 'full';
/** Styles that store a per-atom charge. */
export const hasChargeStyle = (st: AtomStyle): boolean => st === 'charge' || st === 'full';
/** Per-atom charges exist: from the atom style or from fix property/atom q. */
export const hasCharge = (s: SimState): boolean => hasChargeStyle(s.atomStyle) || s.propQ;
/** A custom property attribute name (fix property/atom). */
export const CUSTOM_ATTR = /^(i|d)(2?)_([A-Za-z0-9_]+)(?:\[(\d+|\*)\])?$/;

/**
 * Resolves i_name, d_name, i2_name[N], d2_name[N] (dump.html: "i_name =
 * custom integer vector with name", "d2_name[I] = Ith column of custom
 * floating point vector with name") to a per-atom getter, or throws naming
 * the attribute. null when the word is not a custom-property attribute.
 */
export const customAttr = (s: SimState, attr: string): ((i: number) => number) | null => {
  const m = CUSTOM_ATTR.exec(attr);
  if (!m) return null;
  const cp = s.custom.get(m[3]);
  if (!cp || cp.int !== (m[1] === 'i') || (cp.cols > 0) !== (m[2] === '2')) {
    throw new StyleError(`custom per-atom property ${attr} does not exist (define it with fix property/atom)`);
  }
  if (cp.cols > 0) {
    const col = Number(m[4]);
    if (!m[4] || !Number.isInteger(col) || col < 1 || col > cp.cols) throw new StyleError(`${attr}: column must be 1..${cp.cols}`);
    const w = cp.cols;
    return (i) => cp.data[w * i + col - 1];
  }
  if (m[4]) throw new StyleError(`${attr}: ${m[1]}_${m[3]} is a vector, not an array`);
  return (i) => cp.data[i];
};

/** Molecule IDs exist: from the atom style or from fix property/atom mol. */
export const hasMolecule = (s: SimState): boolean => isMolecularStyle(s.atomStyle) || s.propMol;

/**
 * Mass of a sphere of the given radius and density: "If the atom has a
 * radius attribute ... and its radius is non-zero, its mass is set from the
 * density and particle volume for 3d systems" and "If none of these cases are
 * valid, then the mass is set to the density value directly" (set.html);
 * read_data.html: "for 2d simulations of spheres, this command will treat
 * them as spheres when converting density to mass". disc = set density/disc
 * ("Their mass is set from the density and particle area").
 */
export const sphereMass = (radius: number, density: number, disc = false): number =>
  radius > 0 ? density * (disc ? Math.PI * radius * radius : (4 * Math.PI / 3) * radius ** 3) : density;

/**
 * create_atoms defaults for atom_style sphere, measured with native LAMMPS
 * (write_data after create_atoms): diameter 1 and density 1, i.e. radius 0.5
 * and mass 4 pi / 3 * 0.125, in 2d as in 3d.
 */
export const SPHERE_DEFAULT_RADIUS = 0.5;
export const SPHERE_DEFAULT_MASS = sphereMass(0.5, 1);

export const ALL_GROUP_BIT = 1;

const emptyList = (width: 2 | 3 | 4): TopoList => ({ n: 0, width, type: new Int32Array(0), atoms: new Int32Array(0) });

export const emptyTopology = (): Topology => ({
  nbondtypes: 0, nangletypes: 0, ndihedraltypes: 0, nimpropertypes: 0,
  bonds: emptyList(2), angles: emptyList(3), dihedrals: emptyList(4), impropers: emptyList(4),
});

const isFullBox = (b: BoxInit | SimBox): b is SimBox => 'minLo' in b;

export const emptyState = (
  units: UnitSystem, dimension: 2 | 3, box: BoxInit | SimBox, ntypes: number, atomStyle: AtomStyle = 'atomic',
): SimState => ({
  n: 0,
  dimension,
  box: isFullBox(box) ? box : makeBox(box),
  units,
  atomStyle,
  ntypes,
  type: new Int32Array(0),
  massByType: new Float64Array(ntypes + 1).fill(Number.NaN),
  rmass: atomStyle === 'sphere' ? new Float64Array(0) : null,
  radius: atomStyle === 'sphere' ? new Float64Array(0) : null,
  omega: atomStyle === 'sphere' ? new Float64Array(0) : null,
  torque: atomStyle === 'sphere' ? new Float64Array(0) : null,
  custom: new Map(),
  propMol: false,
  propQ: false,
  x: new Float64Array(0),
  v: new Float64Array(0),
  f: new Float64Array(0),
  image: new Int32Array(0),
  id: new Int32Array(0),
  mask: new Int32Array(0),
  molecule: new Int32Array(0),
  q: new Float64Array(0),
  topo: emptyTopology(),
  step: 0,
  dt: units.dt,
  time: 0,
  timeStep: 0,
});

/** Mass of atom i: the per-atom mass when the atom style has one (rmass), else its type's mass. */
export const massOf = (s: SimState, i: number): number => (s.rmass ? s.rmass[i] : s.massByType[s.type[i]]);

export const maxAtomId = (s: SimState): number => {
  let m = 0;
  for (let i = 0; i < s.n; i++) if (s.id[i] > m) m = s.id[i];
  return m;
};

export interface NewAtoms {
  /** Flat 3N positions. */
  x: Float64Array;
  type: number | Int32Array;
  id?: Int32Array;
  v?: Float64Array;
  image?: Int32Array;
  molecule?: number | Int32Array;
  q?: number | Float64Array;
  /** Per-atom masses (only for atom styles with rmass; default SPHERE_DEFAULT_MASS). */
  rmass?: number | Float64Array;
  /** atom_style sphere: radii (default SPHERE_DEFAULT_RADIUS) and flat 3N angular velocities (default 0). */
  radius?: number | Float64Array;
  omega?: Float64Array;
  /** fix property/atom values by name (n * max(cols, 1) each; default 0). */
  custom?: Map<string, Float64Array>;
  /** Group bits to set besides 'all'. */
  mask?: number;
}

const growF = (a: Float64Array, n: number) => { const b = new Float64Array(n); b.set(a.subarray(0, Math.min(a.length, n))); return b; };
const growI = (a: Int32Array, n: number) => { const b = new Int32Array(n); b.set(a.subarray(0, Math.min(a.length, n))); return b; };

/** Appends atoms; IDs continue from the current maximum unless given. Returns the count. */
export const appendAtoms = (s: SimState, a: NewAtoms): number => {
  const add = a.x.length / 3;
  if (add === 0) return 0;
  const n0 = s.n;
  const n = n0 + add;
  s.x = growF(s.x, 3 * n); s.x.set(a.x, 3 * n0);
  s.v = growF(s.v, 3 * n); if (a.v) s.v.set(a.v, 3 * n0);
  s.f = growF(s.f, 3 * n);
  s.image = growI(s.image, 3 * n); if (a.image) s.image.set(a.image, 3 * n0);
  s.type = growI(s.type, n);
  if (typeof a.type === 'number') s.type.fill(a.type, n0, n); else s.type.set(a.type, n0);
  s.id = growI(s.id, n);
  if (a.id) s.id.set(a.id, n0);
  else {
    let next = maxAtomId(s);
    for (let i = n0; i < n; i++) s.id[i] = ++next;
  }
  s.mask = growI(s.mask, n); s.mask.fill(ALL_GROUP_BIT | (a.mask ?? 0), n0, n);
  s.molecule = growI(s.molecule, n);
  if (typeof a.molecule === 'number') s.molecule.fill(a.molecule, n0, n);
  else if (a.molecule) s.molecule.set(a.molecule, n0);
  s.q = growF(s.q, n);
  if (typeof a.q === 'number') s.q.fill(a.q, n0, n);
  else if (a.q) s.q.set(a.q, n0);
  if (s.rmass) {
    s.rmass = growF(s.rmass, n);
    // measured: fix property/atom rmass starts at 0 for new atoms; sphere atoms get the sphere default
    if (a.rmass instanceof Float64Array) s.rmass.set(a.rmass, n0);
    else s.rmass.fill(a.rmass ?? (s.atomStyle === 'sphere' ? SPHERE_DEFAULT_MASS : 0), n0, n);
  }
  if (s.radius) {
    s.radius = growF(s.radius, n);
    if (a.radius instanceof Float64Array) s.radius.set(a.radius, n0);
    else s.radius.fill(a.radius ?? SPHERE_DEFAULT_RADIUS, n0, n);
  }
  if (s.omega) { s.omega = growF(s.omega, 3 * n); if (a.omega) s.omega.set(a.omega, 3 * n0); }
  if (s.torque) s.torque = growF(s.torque, 3 * n);
  for (const [name, c] of s.custom) {
    const w = Math.max(c.cols, 1);
    c.data = growF(c.data, w * n);
    const src = a.custom?.get(name);
    if (src) c.data.set(src, w * n0);
  }
  s.n = n;
  return add;
};

/** Appends atoms (flat 3N positions) of one type — the v1 signature. */
export const addAtoms = (s: SimState, positions: Float64Array, type: number): number =>
  appendAtoms(s, { x: positions, type });

/**
 * All per-atom data of the atoms at the given indices, as NewAtoms (ids and
 * molecule IDs included; callers adjust them), e.g. to replicate atoms.
 */
export const gatherAtoms = (s: SimState, idx: ArrayLike<number>): NewAtoms => {
  const m = idx.length;
  const pick = (a: Float64Array, w: number) => { const out = new Float64Array(w * m); for (let k = 0; k < m; k++) for (let c = 0; c < w; c++) out[w * k + c] = a[w * idx[k] + c]; return out; };
  const pickI = (a: Int32Array, w: number) => { const out = new Int32Array(w * m); for (let k = 0; k < m; k++) for (let c = 0; c < w; c++) out[w * k + c] = a[w * idx[k] + c]; return out; };
  const custom = new Map<string, Float64Array>();
  for (const [name, c] of s.custom) custom.set(name, pick(c.data, Math.max(c.cols, 1)));
  return {
    x: pick(s.x, 3), v: pick(s.v, 3), image: pickI(s.image, 3), type: pickI(s.type, 1), id: pickI(s.id, 1),
    molecule: pickI(s.molecule, 1), q: pick(s.q, 1),
    rmass: s.rmass ? pick(s.rmass, 1) : undefined, radius: s.radius ? pick(s.radius, 1) : undefined,
    omega: s.omega ? pick(s.omega, 3) : undefined, custom,
  };
};

/**
 * Deletes atoms flagged in `del` (length n), keeping the order of the rest.
 * Bonded entries that reference a deleted atom are removed too.
 */
export const deleteAtoms = (s: SimState, del: Uint8Array): number => {
  let k = 0;
  const gone = new Set<number>();
  for (let i = 0; i < s.n; i++) {
    if (del[i]) { gone.add(s.id[i]); continue; }
    if (k !== i) {
      for (let d = 0; d < 3; d++) {
        s.x[3 * k + d] = s.x[3 * i + d]; s.v[3 * k + d] = s.v[3 * i + d];
        s.f[3 * k + d] = s.f[3 * i + d]; s.image[3 * k + d] = s.image[3 * i + d];
      }
      s.type[k] = s.type[i]; s.id[k] = s.id[i]; s.mask[k] = s.mask[i];
      s.molecule[k] = s.molecule[i]; s.q[k] = s.q[i];
      if (s.rmass) s.rmass[k] = s.rmass[i];
      if (s.radius) s.radius[k] = s.radius[i];
      for (let d = 0; d < 3; d++) {
        if (s.omega) s.omega[3 * k + d] = s.omega[3 * i + d];
        if (s.torque) s.torque[3 * k + d] = s.torque[3 * i + d];
      }
      for (const c of s.custom.values()) {
        const w = Math.max(c.cols, 1);
        for (let m = 0; m < w; m++) c.data[w * k + m] = c.data[w * i + m];
      }
    }
    k++;
  }
  const removed = s.n - k;
  if (removed === 0) return 0;
  s.n = k;
  s.x = s.x.slice(0, 3 * k); s.v = s.v.slice(0, 3 * k); s.f = s.f.slice(0, 3 * k);
  s.image = s.image.slice(0, 3 * k); s.type = s.type.slice(0, k); s.id = s.id.slice(0, k);
  s.mask = s.mask.slice(0, k); s.molecule = s.molecule.slice(0, k); s.q = s.q.slice(0, k);
  if (s.rmass) s.rmass = s.rmass.slice(0, k);
  if (s.radius) s.radius = s.radius.slice(0, k);
  if (s.omega) s.omega = s.omega.slice(0, 3 * k);
  if (s.torque) s.torque = s.torque.slice(0, 3 * k);
  for (const c of s.custom.values()) c.data = c.data.slice(0, Math.max(c.cols, 1) * k);
  for (const list of [s.topo.bonds, s.topo.angles, s.topo.dihedrals, s.topo.impropers]) {
    filterTopo(list, (ids) => !ids.some((id) => gone.has(id)));
  }
  return removed;
};

/** id -> index lookup (index -1 = no such atom). Rebuild after atoms change. */
export const buildAtomMap = (s: SimState): Int32Array => {
  const map = new Int32Array(maxAtomId(s) + 1).fill(-1);
  for (let i = 0; i < s.n; i++) map[s.id[i]] = i;
  return map;
};

/** Appends one entry to a topology list. */
export const pushTopo = (list: TopoList, type: number, ids: readonly number[]): void => {
  if (list.n === list.type.length) {
    const cap = Math.max(16, 2 * list.n);
    list.type = growI(list.type, cap);
    list.atoms = growI(list.atoms, cap * list.width);
  }
  list.type[list.n] = type;
  for (let k = 0; k < list.width; k++) list.atoms[list.n * list.width + k] = ids[k];
  list.n++;
};

export const filterTopo = (list: TopoList, keep: (ids: number[], type: number) => boolean): void => {
  let k = 0;
  const ids: number[] = new Array(list.width);
  for (let e = 0; e < list.n; e++) {
    for (let w = 0; w < list.width; w++) ids[w] = list.atoms[e * list.width + w];
    if (!keep(ids, list.type[e])) continue;
    list.type[k] = list.type[e];
    for (let w = 0; w < list.width; w++) list.atoms[k * list.width + w] = ids[w];
    k++;
  }
  list.n = k;
};

/**
 * Special neighbours of every atom (1-2, 1-3, 1-4), by atom ID, from the
 * bond list — docs.lammps.org/special_bonds.html: "1-3 and 1-4 interactions
 * are not defined from the list of angles or dihedrals used by the
 * simulation. Rather, they are inferred topologically from the set of bonds".
 * A partner is listed once, at its closest separation. With `angle` yes a 1-3
 * pair counts only if it is the 1,3 pair of an angle or the 1,3 / 2,4 pair of
 * a dihedral; with `dihedral` yes a 1-4 pair only if it is the 1,4 pair of a
 * dihedral ("The angle keyword allows the 1-3 weighting factor to be ignored
 * for individual atom pairs if they are not listed as the first and last
 * atoms in any angle ...").
 */
export interface SpecialList {
  /** For atom index i: partners at offset[i] .. offset[i+1). */
  offset: Int32Array;
  partner: Int32Array;
  /** 1 = 1-2, 2 = 1-3, 3 = 1-4. */
  order: Int8Array;
}

export const buildSpecial = (s: SimState, opts: { angle?: boolean; dihedral?: boolean } = {}): SpecialList => {
  const adj = new Map<number, number[]>();
  const b = s.topo.bonds;
  for (let e = 0; e < b.n; e++) {
    const i = b.atoms[2 * e], j = b.atoms[2 * e + 1];
    if (!adj.has(i)) adj.set(i, []);
    if (!adj.has(j)) adj.set(j, []);
    adj.get(i)!.push(j);
    adj.get(j)!.push(i);
  }
  const pairKey = (a: number, c: number) => (a < c ? `${a},${c}` : `${c},${a}`);
  let angle13: Set<string> | null = null;
  let dihed14: Set<string> | null = null;
  if (opts.angle) {
    angle13 = new Set();
    const A = s.topo.angles;
    for (let e = 0; e < A.n; e++) angle13.add(pairKey(A.atoms[3 * e], A.atoms[3 * e + 2]));
    const D = s.topo.dihedrals;
    for (let e = 0; e < D.n; e++) {
      angle13.add(pairKey(D.atoms[4 * e], D.atoms[4 * e + 2]));
      angle13.add(pairKey(D.atoms[4 * e + 1], D.atoms[4 * e + 3]));
    }
  }
  if (opts.dihedral) {
    dihed14 = new Set();
    const D = s.topo.dihedrals;
    for (let e = 0; e < D.n; e++) dihed14.add(pairKey(D.atoms[4 * e], D.atoms[4 * e + 3]));
  }
  const offset = new Int32Array(s.n + 1);
  const partner: number[] = [];
  const order: number[] = [];
  for (let i = 0; i < s.n; i++) {
    offset[i] = partner.length;
    const me = s.id[i];
    const n1 = adj.get(me);
    if (n1) {
      const seen = new Map<number, number>([[me, 0]]);
      let frontier = [me];
      for (let o = 1; o <= 3; o++) {
        const next: number[] = [];
        for (const a of frontier) {
          for (const c of adj.get(a) ?? []) {
            if (seen.has(c)) continue;
            seen.set(c, o);
            next.push(c);
          }
        }
        frontier = next;
      }
      for (const [id, o] of seen) {
        if (o === 0) continue;
        if (o === 2 && angle13 && !angle13.has(pairKey(me, id))) continue;
        if (o === 3 && dihed14 && !dihed14.has(pairKey(me, id))) continue;
        partner.push(id);
        order.push(o);
      }
    }
  }
  offset[s.n] = partner.length;
  return { offset, partner: Int32Array.from(partner), order: Int8Array.from(order) };
};
