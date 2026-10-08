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

/**
 * The sub-styles of an atom style (one, or the list of atom_style hybrid). atom_style.html: "When a
 * hybrid style is used, atoms store and communicate the union of all quantities implied by the
 * individual styles."
 */
export const atomSubStyles = (st: AtomStyle): string[] => (st.startsWith('hybrid ') ? st.slice(7).trim().split(/\s+/) : [st])
  .map((x) => (x.startsWith('template:') ? 'template' : x));
/** atom_style template ID of a style (the ID after the template keyword), or null when there is no template sub-style. */
export const templateStyleId = (st: AtomStyle): string | null => {
  const w = (st.startsWith('hybrid ') ? st.slice(7).trim().split(/\s+/) : [st]).find((x) => x.startsWith('template:'));
  return w ? w.slice('template:'.length) : null;
};
/** Styles with molecule templates (atom_style template): topology comes from the template, per-atom template index and atom. */
export const isTemplateStyle = (st: AtomStyle): boolean => atomSubStyles(st).includes('template');
/** Styles with bond topology and molecule IDs (template styles too: their topology is expanded from the template). */
export const isMolecularStyle = (st: AtomStyle): boolean => atomSubStyles(st).some((x) => x === 'bond' || x === 'angle' || x === 'molecular' || x === 'full' || x === 'template');
/** Styles that store a per-atom charge. */
export const hasChargeStyle = (st: AtomStyle): boolean => atomSubStyles(st).some((x) => x === 'charge' || x === 'full' || x === 'dipole');
/** Styles with finite-size spheres (radius, rmass, omega, torque). */
export const isSphereStyle = (st: AtomStyle): boolean => atomSubStyles(st).includes('sphere');
/** Styles with point dipoles (mu). */
export const hasDipoleStyle = (st: AtomStyle): boolean => atomSubStyles(st).includes('dipole');
/** Styles with ellipsoidal particles (shape, quat, angmom). */
export const isEllipsoidStyle = (st: AtomStyle): boolean => atomSubStyles(st).includes('ellipsoid');
/** Styles with peridynamic particles (vfrac, x0): atom_style.html lists peri as atomic plus rmass, vfrac, s0 and x0. */
export const isPeriStyle = (st: AtomStyle): boolean => atomSubStyles(st).includes('peri');
/** Styles with a per-atom mass (rmass): sphere, ellipsoid and peri. */
export const hasRmassStyle = (st: AtomStyle): boolean => isSphereStyle(st) || isEllipsoidStyle(st) || isPeriStyle(st);
/** Bonded topology a style stores: 0 none, 1 bonds, 2 bonds and angles, 3 also dihedrals and impropers. */
export const topologyLevel = (st: AtomStyle): number => {
  let lv = 0;
  for (const x of atomSubStyles(st)) lv = Math.max(lv, x === 'bond' ? 1 : x === 'angle' ? 2 : x === 'molecular' || x === 'full' ? 3 : 0);
  return lv;
};
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
  rmass: hasRmassStyle(atomStyle) ? new Float64Array(0) : null,
  vfrac: isPeriStyle(atomStyle) ? new Float64Array(0) : null,
  x0: isPeriStyle(atomStyle) ? new Float64Array(0) : null,
  radius: isSphereStyle(atomStyle) ? new Float64Array(0) : null,
  omega: isSphereStyle(atomStyle) ? new Float64Array(0) : null,
  torque: isSphereStyle(atomStyle) || isEllipsoidStyle(atomStyle) ? new Float64Array(0) : null,
  tmplIndex: isTemplateStyle(atomStyle) ? new Int32Array(0) : null,
  tmplAtom: isTemplateStyle(atomStyle) ? new Int32Array(0) : null,
  mu: hasDipoleStyle(atomStyle) ? new Float64Array(0) : null,
  shape: isEllipsoidStyle(atomStyle) ? new Float64Array(0) : null,
  quat: isEllipsoidStyle(atomStyle) ? new Float64Array(0) : null,
  angmom: isEllipsoidStyle(atomStyle) ? new Float64Array(0) : null,
  custom: new Map(),
  propMol: false,
  propQ: false,
  x: new Float64Array(0),
  v: new Float64Array(0),
  f: new Float64Array(0),
  image: new Int32Array(0),
  id: new Int32Array(0),
  order: new Int32Array(0),
  mask: new Int32Array(0),
  molecule: new Int32Array(0),
  q: new Float64Array(0),
  topo: emptyTopology(),
  step: 0,
  dt: units.dt,
  time: 0,
  timeStep: 0,
});

/** Atom i is a finite-size ellipsoid (atom_style ellipsoid; 0 half-axes mean a point particle). */
export const isEllipsoid = (s: SimState, i: number): boolean => !!s.shape && s.shape[3 * i] > 0;
/** Volume 4/3 pi a b c of ellipsoid i from its half-axes. */
export const ellipsoidVolume = (s: SimState, i: number): number => (4 / 3) * Math.PI * s.shape![3 * i] * s.shape![3 * i + 1] * s.shape![3 * i + 2];
/** Number of finite-size ellipsoids. */
export const countEllipsoids = (s: SimState): number => {
  let n = 0;
  for (let i = 0; i < s.n; i++) if (isEllipsoid(s, i)) n++;
  return n;
};

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
  /** atom_style template: template index and template atom per atom (default 0). */
  tmplIndex?: number | Int32Array;
  tmplAtom?: number | Int32Array;
  q?: number | Float64Array;
  /** Per-atom masses (only for atom styles with rmass; default SPHERE_DEFAULT_MASS). */
  rmass?: number | Float64Array;
  /** atom_style sphere: radii (default SPHERE_DEFAULT_RADIUS) and flat 3N angular velocities (default 0). */
  radius?: number | Float64Array;
  omega?: Float64Array;
  /** atom_style dipole: flat 4N dipoles (mux, muy, muz, length; default 0). */
  mu?: Float64Array;
  /** atom_style ellipsoid: flat 3N half-axes (default 0, a point particle), 4N quaternions (default 1 0 0 0), 3N angular momenta. */
  shape?: Float64Array;
  quat?: Float64Array;
  angmom?: Float64Array;
  /** fix property/atom values by name (n * max(cols, 1) each; default 0). */
  custom?: Map<string, Float64Array>;
  /** atom_style peri: volumes (default 1) and reference positions (flat 3N; default the positions x). */
  vfrac?: number | Float64Array;
  x0?: Float64Array;
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
  // new atoms go to the end of native storage
  nativeOrder(s);
  s.order = growI(s.order, n);
  for (let i = n0; i < n; i++) s.order[i] = i;
  s.mask = growI(s.mask, n); s.mask.fill(ALL_GROUP_BIT | (a.mask ?? 0), n0, n);
  s.molecule = growI(s.molecule, n);
  if (typeof a.molecule === 'number') s.molecule.fill(a.molecule, n0, n);
  else if (a.molecule) s.molecule.set(a.molecule, n0);
  if (s.tmplIndex && s.tmplAtom) {
    s.tmplIndex = growI(s.tmplIndex, n); s.tmplAtom = growI(s.tmplAtom, n);
    for (const [arr, src] of [[s.tmplIndex, a.tmplIndex], [s.tmplAtom, a.tmplAtom]] as const) {
      if (typeof src === 'number') arr.fill(src, n0, n);
      else if (src) arr.set(src, n0);
      else arr.fill(0, n0, n);
    }
  }
  s.q = growF(s.q, n);
  if (typeof a.q === 'number') s.q.fill(a.q, n0, n);
  else if (a.q) s.q.set(a.q, n0);
  if (s.rmass) {
    s.rmass = growF(s.rmass, n);
    // measured: fix property/atom rmass starts at 0 for new atoms; sphere atoms get the sphere default
    if (a.rmass instanceof Float64Array) s.rmass.set(a.rmass, n0);
    // measured with native LAMMPS: create_atoms gives ellipsoid-style atoms (point particles) mass 1
    // measured with native LAMMPS: atom_style peri gives mass 1 (its density and volume defaults are 1)
    else s.rmass.fill(a.rmass ?? (isSphereStyle(s.atomStyle) ? SPHERE_DEFAULT_MASS : isEllipsoidStyle(s.atomStyle) || isPeriStyle(s.atomStyle) ? 1 : 0), n0, n);
  }
  if (s.vfrac) {
    s.vfrac = growF(s.vfrac, n);
    if (a.vfrac instanceof Float64Array) s.vfrac.set(a.vfrac, n0);
    else s.vfrac.fill(a.vfrac ?? 1, n0, n);
  }
  if (s.x0) {
    s.x0 = growF(s.x0, 3 * n);
    // the reference configuration of new atoms is the configuration they are created in (x0 = x)
    if (a.x0) s.x0.set(a.x0, 3 * n0); else s.x0.set(a.x, 3 * n0);
  }
  if (s.radius) {
    s.radius = growF(s.radius, n);
    if (a.radius instanceof Float64Array) s.radius.set(a.radius, n0);
    else s.radius.fill(a.radius ?? SPHERE_DEFAULT_RADIUS, n0, n);
  }
  if (s.omega) { s.omega = growF(s.omega, 3 * n); if (a.omega) s.omega.set(a.omega, 3 * n0); }
  if (s.torque) s.torque = growF(s.torque, 3 * n);
  if (s.mu) { s.mu = growF(s.mu, 4 * n); if (a.mu) s.mu.set(a.mu, 4 * n0); }
  if (s.shape) { s.shape = growF(s.shape, 3 * n); if (a.shape) s.shape.set(a.shape, 3 * n0); }
  if (s.quat) {
    s.quat = growF(s.quat, 4 * n);
    if (a.quat) s.quat.set(a.quat, 4 * n0);
    else for (let i = n0; i < n; i++) s.quat[4 * i] = 1;
  }
  if (s.angmom) { s.angmom = growF(s.angmom, 3 * n); if (a.angmom) s.angmom.set(a.angmom, 3 * n0); }
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
    tmplIndex: s.tmplIndex ? pickI(s.tmplIndex, 1) : undefined, tmplAtom: s.tmplAtom ? pickI(s.tmplAtom, 1) : undefined,
    rmass: s.rmass ? pick(s.rmass, 1) : undefined, radius: s.radius ? pick(s.radius, 1) : undefined,
    omega: s.omega ? pick(s.omega, 3) : undefined, mu: s.mu ? pick(s.mu, 4) : undefined,
    shape: s.shape ? pick(s.shape, 3) : undefined, quat: s.quat ? pick(s.quat, 4) : undefined, angmom: s.angmom ? pick(s.angmom, 3) : undefined, custom,
    vfrac: s.vfrac ? pick(s.vfrac, 1) : undefined, x0: s.x0 ? pick(s.x0, 3) : undefined,
  };
};

/**
 * Deletes atoms flagged in `del` (length n), keeping the order of the rest.
 * Bonded entries that reference a deleted atom are removed too.
 */
export const deleteAtoms = (s: SimState, del: Uint8Array): number => {
  // atom_style template: the bonds of a template molecule come from its template, so a molecule
  // that loses some atoms but keeps others cannot be bonded any more. Measured with native LAMMPS
  // (black box, delete_atoms group of one atom of a 6-atom template molecule, then run 0): native stops
  // at the next run setup with Bond atom missing in image check. Deleting a whole molecule is fine.
  if (s.tmplIndex && s.tmplAtom) {
    const total = new Map<string, number>(), gone = new Map<string, number>();
    for (let i = 0; i < s.n; i++) {
      if (s.tmplIndex[i] === 0) continue;
      const key = `${s.molecule[i]}:${s.tmplIndex[i]}`;
      total.set(key, (total.get(key) ?? 0) + 1);
      if (del[i]) gone.set(key, (gone.get(key) ?? 0) + 1);
    }
    for (const [key, g] of gone) {
      if (g < (total.get(key) ?? 0)) {
        throw new StyleError(`Bond atom missing in image check: molecule ${key.split(':')[0]} (template index ${key.split(':')[1]}) lost some atoms, but its bonds come from the molecule template`);
      }
    }
  }
  // Measured with native LAMMPS (black box, atoms 1..10, delete 2 3 7 with compress no): native
  // storage becomes 1 10 9 4 5 6 8, i.e. walking its list, a deleted atom's slot takes the last
  // atom of the list, which is then checked in turn.
  const ord = Array.from(nativeOrder(s));
  let m = ord.length;
  for (let p = 0; p < m;) {
    if (del[ord[p]]) { ord[p] = ord[m - 1]; m--; } else p++;
  }
  const newIndex = new Int32Array(s.n).fill(-1);
  let k = 0;
  const gone = new Set<number>();
  for (let i = 0; i < s.n; i++) {
    if (del[i]) { gone.add(s.id[i]); continue; }
    newIndex[i] = k;
    if (k !== i) {
      for (let d = 0; d < 3; d++) {
        s.x[3 * k + d] = s.x[3 * i + d]; s.v[3 * k + d] = s.v[3 * i + d];
        s.f[3 * k + d] = s.f[3 * i + d]; s.image[3 * k + d] = s.image[3 * i + d];
      }
      s.type[k] = s.type[i]; s.id[k] = s.id[i]; s.mask[k] = s.mask[i];
      s.molecule[k] = s.molecule[i]; s.q[k] = s.q[i];
      if (s.tmplIndex && s.tmplAtom) { s.tmplIndex[k] = s.tmplIndex[i]; s.tmplAtom[k] = s.tmplAtom[i]; }
      if (s.rmass) s.rmass[k] = s.rmass[i];
      if (s.radius) s.radius[k] = s.radius[i];
      for (let d = 0; d < 3; d++) {
        if (s.omega) s.omega[3 * k + d] = s.omega[3 * i + d];
        if (s.torque) s.torque[3 * k + d] = s.torque[3 * i + d];
      }
      if (s.mu) for (let d = 0; d < 4; d++) s.mu[4 * k + d] = s.mu[4 * i + d];
      if (s.shape) for (let d = 0; d < 3; d++) s.shape[3 * k + d] = s.shape[3 * i + d];
      if (s.quat) for (let d = 0; d < 4; d++) s.quat[4 * k + d] = s.quat[4 * i + d];
      if (s.angmom) for (let d = 0; d < 3; d++) s.angmom[3 * k + d] = s.angmom[3 * i + d];
      if (s.vfrac) s.vfrac[k] = s.vfrac[i];
      if (s.x0) for (let d = 0; d < 3; d++) s.x0[3 * k + d] = s.x0[3 * i + d];
      for (const c of s.custom.values()) {
        const w = Math.max(c.cols, 1);
        for (let m = 0; m < w; m++) c.data[w * k + m] = c.data[w * i + m];
      }
    }
    k++;
  }
  const removed = s.n - k;
  if (removed === 0) return 0;
  s.order = new Int32Array(k);
  for (let p = 0; p < m; p++) s.order[p] = newIndex[ord[p]];
  s.n = k;
  s.x = s.x.slice(0, 3 * k); s.v = s.v.slice(0, 3 * k); s.f = s.f.slice(0, 3 * k);
  s.image = s.image.slice(0, 3 * k); s.type = s.type.slice(0, k); s.id = s.id.slice(0, k);
  s.mask = s.mask.slice(0, k); s.molecule = s.molecule.slice(0, k); s.q = s.q.slice(0, k);
  if (s.tmplIndex && s.tmplAtom) { s.tmplIndex = s.tmplIndex.slice(0, k); s.tmplAtom = s.tmplAtom.slice(0, k); }
  if (s.rmass) s.rmass = s.rmass.slice(0, k);
  if (s.radius) s.radius = s.radius.slice(0, k);
  if (s.omega) s.omega = s.omega.slice(0, 3 * k);
  if (s.torque) s.torque = s.torque.slice(0, 3 * k);
  if (s.mu) s.mu = s.mu.slice(0, 4 * k);
  if (s.shape) s.shape = s.shape.slice(0, 3 * k);
  if (s.quat) s.quat = s.quat.slice(0, 4 * k);
  if (s.angmom) s.angmom = s.angmom.slice(0, 3 * k);
  if (s.vfrac) s.vfrac = s.vfrac.slice(0, k);
  if (s.x0) s.x0 = s.x0.slice(0, 3 * k);
  for (const c of s.custom.values()) c.data = c.data.slice(0, Math.max(c.cols, 1) * k);
  for (const list of [s.topo.bonds, s.topo.angles, s.topo.dihedrals, s.topo.impropers]) {
    filterTopo(list, (ids) => !ids.some((id) => gone.has(id)));
  }
  return removed;
};

/** The native storage order (SimState.order), reset to the engine's order if it does not fit the atoms. */
export const nativeOrder = (s: SimState): Int32Array => {
  if (!s.order || s.order.length !== s.n) {
    s.order = new Int32Array(s.n);
    for (let i = 0; i < s.n; i++) s.order[i] = i;
  }
  return s.order;
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
