import { Compute } from './compute';
import { StyleError, newAccum, type Accum, type Bonded, type BondedCompute, type Pair } from '../force/types';
import { PairHybrid } from '../force/pair/hybrid';
import { bondDelta, delta, dihedralGeometry } from '../force/bonded_util';
import type { BondedCopies } from '../force/types';
import { atomSubStyles, buildAtomMap, isTemplateStyle, massOf, nativeOrder } from '../atoms';
import type { System } from '../system';
import type { SimState, TopoList } from '../types';
import type { Geometry } from '../domain';
import type { Neighbor, NeighList } from '../neighbor';
import { NEIGHMASK, SBBITS } from '../neighbor';
import { gplHook } from '../gplHooks';
import { missingArgs } from '../force/util';
import { libm } from '../libm';

/*
 * Local computes (docs.lammps.org/compute_bond_local.html and the angle,
 * dihedral, improper, property local pages): one row per bond, angle,
 * dihedral or improper, the row order being the order native LAMMPS stores
 * the topology in.
 *
 * Quotes used:
 *   compute_bond_local.html: "The value *dist* is the current length of the bond."
 *   "The values *dx*, *dy*, and *dz* are the :math:`(x,y,z)` components of the distance
 *   vector :math:`\vec{x_i} - \vec{x_j}` between the atoms in the bond."
 *   "The value *engpot* is the potential energy for the bond, based on the current
 *   separation of the pair of atoms in the bond."
 *   "The value *force* is the magnitude of the force acting between the pair of atoms in
 *   the bond, which is positive for a repulsive force and negative for an attractive force."
 *   "The values *fx*, *fy*, and *fz* are the :math:`(x,y,z)` components of the force on the
 *   first atom *i* in the bond due to the second atom *j*."
 *   "The value *velvib* is the magnitude of the relative velocity of the two atoms in the bond
 *   towards each other.  A negative value means the two atoms are moving toward each other;"
 *   "The value *engvib* is the vibrational kinetic energy of the two atoms in the bond, which is
 *   simply :math:`\frac12 m_1 v_1^2 + \frac12 m_2 v_2^2,` where :math:`v_1` and :math:`v_2` are
 *   the magnitude of the velocity of the two atoms along the bond direction, after the COM
 *   velocity has been subtracted from each."
 *   "The value *engrot* ... the two atoms perpendicular to the bond direction, after the COM
 *   velocity has been subtracted from each."
 *   "The value *engtrans* is the translational kinetic energy associated with the motion of the
 *   COM of the system itself, namely :math:`\frac12(m_1+m_2) V_{\mathrm{cm}}^2`"
 *   compute_angle_local.html: "The value *theta* is the angle for the three atoms in the
 *   interaction.  The value *eng* is the interaction energy for the angle."
 *   compute_dihedral_local.html: "The value *phi* (:math:`\phi`) is the dihedral angle, as
 *   defined in the diagram on the dihedral_style doc page."
 *   compute_improper_local.html: "The value *chi* is the improper angle, as defined in the doc
 *   pages for the individual improper styles listed on improper_style doc page."
 *   compute_property_local.html: "The vector or array values will be integers that correspond to
 *   the specified attribute."
 *
 * Energies and forces come from the style's own compute() (force/*): each term is run
 * alone through the style on a one-term topology, so there is one implementation of each
 * potential. Measured with native LAMMPS (black box): a bond's row is written by its first
 * atom, an angle, dihedral or improper by its second atom (the central atom); the rows walk
 * the atoms in native storage order.
 *
 * v_name values with the set keyword (bond, angle and dihedral) — compute_bond_local.html: "The value *v_name* can be
 * used together with the *set* keyword to compute a user-specified function of the bond distance. ... the
 * internal-style variable specified by the *set* keyword. It is an internal-style variable, because this command
 * resets its value directly." As native LAMMPS does (src/compute_bond_local.cpp, compute_angle_local.cpp and
 * compute_dihedral_local.cpp at b332f28): the set variable is created as an internal-style variable when it does not
 * exist; for each row it is set to the bond distance, or to the angle theta or dihedral phi in radians (the theta and
 * phi values are printed in degrees), and then each v_name is evaluated as an equal-style variable, with native's
 * error texts.
 *
 * Not implemented (throw StyleError): omega. compute pair/local is at the end of this file.
 */

export type TermKind = 'bond' | 'angle' | 'dihedral' | 'improper';

const FIELD = { bond: 'bonds', angle: 'angles', dihedral: 'dihedrals', improper: 'impropers' } as const;
const WIDTH = { bond: 2, angle: 3, dihedral: 4, improper: 4 } as const;
/** Position in the topology line of the atom that stores the term (newton_bond on). */
const OWNER = { bond: 0, angle: 1, dihedral: 1, improper: 1 } as const;
const ENERGY = { bond: 'ebond', angle: 'eangle', dihedral: 'edihed', improper: 'eimp' } as const;
const STYLE_FIELD = { bond: 'bond', angle: 'angle', dihedral: 'dihedral', improper: 'improper' } as const;

const BOND_VALUES = ['dist', 'dx', 'dy', 'dz', 'engpot', 'force', 'fx', 'fy', 'fz', 'engvib', 'engrot', 'engtrans', 'velvib'];
const VALUES: Record<TermKind, string[]> = {
  bond: BOND_VALUES,
  angle: ['theta', 'eng'],
  dihedral: ['phi'],
  improper: ['chi'],
};

/** Attributes of property/local for each kind (compute_property_local.html). */
const ATTRS: Record<TermKind, { atoms: string[]; type: string }> = {
  bond: { atoms: ['batom1', 'batom2'], type: 'btype' },
  angle: { atoms: ['aatom1', 'aatom2', 'aatom3'], type: 'atype' },
  dihedral: { atoms: ['datom1', 'datom2', 'datom3', 'datom4'], type: 'dtype' },
  improper: { atoms: ['iatom1', 'iatom2', 'iatom3', 'iatom4'], type: 'itype' },
};

const RAD2DEG = 180 / Math.PI;

/**
 * Topology rows of one kind that a local compute reports, in native order: broken terms (type 0)
 * are left out, and a term needs every one of its atoms in the group.
 */
export const termRows = (sys: System, kind: TermKind, bit: number): { k: number; idx: number[] }[] => {
  const s = sys.state;
  const list: TopoList = s.topo[FIELD[kind]];
  const w = WIDTH[kind], o = OWNER[kind];
  const map = buildAtomMap(s);
  const buckets: number[][] = Array.from({ length: s.n }, () => []);
  for (let k = 0; k < list.n; k++) {
    const id = list.atoms[w * k + o];
    const i = id < map.length ? map[id] : -1;
    if (i < 0) throw new StyleError(`${kind} atom ${id} missing`);
    buckets[i].push(k);
  }
  const rows: { k: number; idx: number[] }[] = [];
  for (const i of nativeOrder(s)) {
    for (const k of buckets[i]) {
      if (list.type[k] === 0) continue;
      const idx: number[] = [];
      let inGroup = true;
      for (let m = 0; m < w; m++) {
        const a = map[list.atoms[w * k + m]];
        idx.push(a);
        if (!(s.mask[a] & bit)) inGroup = false;
      }
      if (inGroup) rows.push({ k, idx });
    }
  }
  return rows;
};

/**
 * Energy of one term, from the style's compute() run on a topology holding only this term.
 * The forces of the term's atoms are left in `scratch` (the caller reads and clears them).
 */
const termEnergy = (s: SimState, geom: Geometry, copies: BondedCopies | null, kind: TermKind, style: Bonded, list: TopoList, k: number, map: Int32Array, scratch: Float64Array): number => {
  const w = WIDTH[kind];
  const one: TopoList = { n: 1, width: w as 2 | 3 | 4, type: Int32Array.of(list.type[k]), atoms: list.atoms.slice(w * k, w * k + w) };
  const shadow = Object.create(s) as SimState;
  shadow.topo = { ...s.topo, [FIELD[kind]]: one };
  const acc: Accum = newAccum();
  const bc: BondedCompute = { s: shadow, geom, map, f: scratch, acc, virial: acc.vbond, eatom: null, vatom: null, copies };
  style.compute(bc);
  return acc[ENERGY[kind]];
};

/**
 * The forces one bonded term puts on its atoms, from the style's own compute() on a topology holding only that term
 * (the term of row k of the kind's list): the owned indices of its atoms in topology order and 3 forces per atom.
 * Used by the stress profile computes for the bonded contributions.
 */
export const termForces = (sys: System, kind: TermKind, k: number): { idx: number[]; f: number[] } => {
  const s = sys.state;
  const style = sys.ff[STYLE_FIELD[kind]];
  if (!style) throw new StyleError(`no ${kind}_style is defined`);
  const list: TopoList = s.topo[FIELD[kind]];
  const map = buildAtomMap(s);
  const w = WIDTH[kind];
  const scratch = new Float64Array(3 * s.n);
  termEnergy(s, sys.geom, sys.nb.bondedCopies(), kind, style, list, k, map, scratch);
  const idx: number[] = [], f: number[] = [];
  for (let m = 0; m < w; m++) {
    const a = map[list.atoms[w * k + m]];
    idx.push(a);
    f.push(scratch[3 * a], scratch[3 * a + 1], scratch[3 * a + 2]);
  }
  return { idx, f };
};

/** r_a - r_b for a bond (owner a), the copy of b nearest to a (bonded_util bondDelta). */
const bondVec = (bc: BondedCompute, a: number, b: number): number[] => {
  const d = [0, 0, 0];
  bondDelta(bc, a, b, d);
  return d;
};
/** r_b' - r_o, the copy of b nearest to the term's owner o (bonded_util delta). */
const fromOwner = (bc: BondedCompute, o: number, b: number): number[] => {
  const d = [0, 0, 0];
  delta(bc, o, b, d);
  return d;
};

const norm = (d: number[]): number => Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

/** A bond style with the single() extra outputs of compute bond/local bN (bond_bpm_rotational.ts bondExtra). */
interface BondExtraProvider {
  bondExtra(bc: BondedCompute, ia: number, ib: number, btype: number): number[];
}

/** A local compute over bonds, angles, dihedrals or impropers with value keywords. */
export class ComputeTermLocal extends Compute {
  readonly style: string;
  private readonly values: string[];
  /** The set keyword's internal-style variable (null without set). */
  private readonly setVar: string | null = null;

  constructor(sys: System, id: string, group: string, args: string[], private readonly kind: TermKind) {
    super(sys, id, group, args);
    this.style = `${kind}/local`;
    if (!args.length) throw new StyleError(`compute ${id} ${this.style} needs at least one value`);
    const values: string[] = [];
    const SET_ARG: Partial<Record<TermKind, string>> = { bond: 'dist', angle: 'theta', dihedral: 'phi' };
    let setVar: string | null = null;
    let setSeen = false;
    for (let a = 0; a < args.length; a++) {
      const w = args[a];
      if (w === 'set' && SET_ARG[kind]) {
        // native: the keyword loop after the values; "set dist name" (bond), "set theta name", "set phi name"
        setSeen = true;
        if (a + 2 >= args.length) {
          throw new StyleError(kind === 'bond' ? 'Illegal compute bond/local set command: missing argument(s)' : `Illegal compute ${kind}/local command`);
        }
        if (args[a + 1] !== SET_ARG[kind]) {
          throw new StyleError(kind === 'bond' ? `Unknown compute bond/local set keyword: ${args[a + 2]}` : `Illegal compute ${kind}/local command`);
        }
        setVar = args[a + 2];
        a += 2;
        continue;
      }
      if (setSeen) throw new StyleError(kind === 'bond' ? `Unknown compute bond/local keyword: ${w}` : `Illegal compute ${kind}/local command`);
      if (w.startsWith('v_') && SET_ARG[kind]) {
        values.push(w);
        continue;
      }
      // compute bond/local: "bN" = the Nth extra output of the bond style's single() (compute_bond_local.cpp at b332f28)
      if (kind === 'bond' && /^b\d+$/.test(w)) {
        if (Number(w.slice(1)) < 1) throw new StyleError(`compute ${id} ${this.style}: Invalid keyword ${w} in compute bond/local command`);
        values.push(w);
        continue;
      }
      if (kind === 'bond' && w === 'omega') throw new StyleError(`compute ${id} ${this.style}: value 'omega' is not supported by the browser engine`);
      if (!VALUES[kind].includes(w)) {
        // As native LAMMPS does (src/compute_{bond,angle,dihedral,improper}_local.cpp at b332f28): the value loop stops at
        // an unknown word and the keyword loop names it in each style's own words
        throw new StyleError(kind === 'bond' ? `Unknown compute bond/local keyword: ${w}`
          : kind === 'improper' ? `Unknown compute improper/local keyword ${w}` : `Illegal compute ${kind}/local command`);
      }
      values.push(w);
    }
    this.values = values;
    // native's error checks (constructor): the variables must exist and be equal-style; the set variable is created
    const vnames = values.filter((v) => v.startsWith('v_')).map((v) => v.slice(2));
    const what = `compute ${kind}/local`;
    if (vnames.length) {
      if (!setSeen) throw new StyleError(`Compute ${kind}/local variable requires a set variable`);
      for (const name of vnames) {
        const v = sys.vars.get(name);
        const named = kind === 'bond' ? ` ${name}` : '';
        if (!v) throw new StyleError(`Variable name${named} for ${what} does not exist`);
        if (v.style !== 'equal') throw new StyleError(`Variable${named} for ${what} is invalid style`);
      }
      if (setVar !== null) {
        const sv = sys.vars.get(setVar);
        if (!sv) sys.vars.setInternal(setVar, 0.0);
        else if (sv.style !== 'internal') throw new StyleError(`Variable for ${what} is invalid style`);
      }
    } else if (setSeen) {
      throw new StyleError(kind === 'bond' ? 'Compute bond/local set used with without a variable' : `Compute ${kind}/local set with no variable`);
    }
    this.setVar = setVar;
    this.localFlag = true;
    this.sizeLocalCols = values.length > 1 ? values.length : 0;
  }

  /** The v_name values of a row: the set variable takes value, then each variable is evaluated in order. */
  private variableValues(vals: Record<string, number>, value: number): void {
    if (this.setVar === null) return;
    this.sys.vars.setInternal(this.setVar, value);
    for (const v of this.values) if (v.startsWith('v_')) vals[v] = this.sys.equalVariable(v.slice(2));
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    const sys = this.sys;
    const s = sys.state;
    const kind = this.kind;
    const geom = sys.geom;
    const nv = this.values.length;
    const ncol = Math.max(1, nv);
    const rows = termRows(sys, kind, this.groupBit);
    const need = (name: string) => this.values.some((v) => v === name);
    const bNames = this.values.filter((v) => /^b\d+$/.test(v));
    // As native LAMMPS does (src/compute_bond_local.cpp init at b332f28): a bN needs a bond style with that extra field
    const extraStyle = bNames.length ? (sys.ff.bond as unknown as Partial<BondExtraProvider> | null) : null;
    if (bNames.length && (!extraStyle || typeof extraStyle.bondExtra !== 'function')) {
      // a style without the bN outputs in the browser engine is named, not reported as a missing field
      throw new StyleError(`compute ${this.id} bond/local ${bNames.join(' ')}: the bN outputs are not supported by the browser engine for bond style ${sys.ff.bond?.name ?? 'none'} (only bpm/rotational has them)`);
    }
    const needEnergy = need('engpot') || need('force') || need('fx') || need('fy') || need('fz') || need('eng');
    const style = sys.ff[STYLE_FIELD[kind]];
    if (needEnergy && !style) throw new StyleError(`compute ${this.id} ${this.style} needs a ${kind}_style`);
    const list = s.topo[FIELD[kind]];
    const nb = sys.nb;
    const map = buildAtomMap(s);
    const scratch = new Float64Array(3 * s.n);
    const mvv2e = s.units.mvv2e;
    const out = new Float64Array(rows.length * ncol);
    const bc: BondedCompute = { s, geom, map, f: scratch, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null, nbuild: nb.nbuild, copies: nb.bondedCopies() };
    rows.forEach(({ k, idx }, r) => {
      const vals: Record<string, number> = {};
      if (kind === 'bond') {
        const [i, j] = idx;
        const d = bondVec(bc, i, j);
        const dist = norm(d);
        const u = dist > 0 ? d.map((c) => c / dist) : [0, 0, 0];
        let e = 0, fi = [0, 0, 0];
        if (needEnergy && list.type[k] > 0) {
          e = termEnergy(s, geom, nb.bondedCopies(), kind, style!, list, k, map, scratch);
          fi = [scratch[3 * i], scratch[3 * i + 1], scratch[3 * i + 2]];
        }
        if (needEnergy) for (const a of [i, j]) for (let c = 0; c < 3; c++) scratch[3 * a + c] = 0;
        const force = dist > 0 ? (fi[0] * d[0] + fi[1] * d[1] + fi[2] * d[2]) / dist : 0;
        vals.dist = dist; vals.dx = d[0]; vals.dy = d[1]; vals.dz = d[2];
        this.variableValues(vals, dist);
        if (bNames.length) {
          const extra = extraStyle!.bondExtra!(bc, list.atoms[2 * k], list.atoms[2 * k + 1], list.type[k]);
          for (const name of bNames) {
            const n = Number(name.slice(1));
            if (n > extra.length) throw new StyleError('Bond style does not have extra field requested by compute bond/local');
            vals[name] = extra[n - 1];
          }
        }
        vals.engpot = e; vals.force = force; vals.fx = fi[0]; vals.fy = fi[1]; vals.fz = fi[2];
        if (need('engvib') || need('engrot') || need('engtrans') || need('velvib')) {
          const mi = massOf(s, i), mj = massOf(s, j), M = mi + mj;
          const v = (a: number) => [s.v[3 * a], s.v[3 * a + 1], s.v[3 * a + 2]];
          const vi = v(i), vj = v(j);
          const vcm = [0, 1, 2].map((c) => (mi * vi[c] + mj * vj[c]) / M);
          const wi = [0, 1, 2].map((c) => vi[c] - vcm[c]);
          const wj = [0, 1, 2].map((c) => vj[c] - vcm[c]);
          const alongI = wi[0] * u[0] + wi[1] * u[1] + wi[2] * u[2];
          const alongJ = wj[0] * u[0] + wj[1] * u[1] + wj[2] * u[2];
          const perpI = wi[0] * wi[0] + wi[1] * wi[1] + wi[2] * wi[2] - alongI * alongI;
          const perpJ = wj[0] * wj[0] + wj[1] * wj[1] + wj[2] * wj[2] - alongJ * alongJ;
          const vc2 = vcm[0] * vcm[0] + vcm[1] * vcm[1] + vcm[2] * vcm[2];
          vals.engvib = 0.5 * mvv2e * (mi * alongI * alongI + mj * alongJ * alongJ);
          vals.engrot = 0.5 * mvv2e * (mi * perpI + mj * perpJ);
          vals.engtrans = 0.5 * mvv2e * M * vc2;
          vals.velvib = (vi[0] - vj[0]) * u[0] + (vi[1] - vj[1]) * u[1] + (vi[2] - vj[2]) * u[2];
        }
      } else if (kind === 'angle') {
        const [i, j, kk] = idx;
        const d1 = fromOwner(bc, j, i), d3 = fromOwner(bc, j, kk);
        const r1 = norm(d1), r3 = norm(d3);
        const c = r1 > 0 && r3 > 0 ? Math.max(-1, Math.min(1, (d1[0] * d3[0] + d1[1] * d3[1] + d1[2] * d3[2]) / (r1 * r3))) : 1;
        vals.theta = libm.acos(c) * RAD2DEG;
        this.variableValues(vals, libm.acos(c));
        if (need('eng')) {
          vals.eng = list.type[k] > 0 ? termEnergy(s, geom, nb.bondedCopies(), kind, style!, list, k, map, scratch) : 0;
          for (const a of idx) for (let q = 0; q < 3; q++) scratch[3 * a + q] = 0;
        }
      } else if (kind === 'dihedral') {
        const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
        const phi = dihedralGeometry(bc, idx[0], idx[1], idx[2], idx[3], grad, rel);
        vals.phi = phi * RAD2DEG;
        this.variableValues(vals, phi);
      } else {
        const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
        vals.chi = Math.abs(dihedralGeometry(bc, idx[0], idx[1], idx[2], idx[3], grad, rel)) * RAD2DEG;
      }
      this.values.forEach((name, c) => { out[r * ncol + c] = vals[name]; });
    });
    this.localRows = rows.length;
    return out;
  }
}

/**
 * The pair attributes of property/local (compute_property_local.html): "natom1, natom2 = store IDs of two atoms in each
 * pair (within neighbor cutoff)", "ntype1, ntype2 = store types of two atoms in each pair (within neighbor cutoff)",
 * "patom1, patom2 = store IDs of two atoms in each pair (within force cutoff)", "ptype1, ptype2 = store types of two
 * atoms in each pair (within force cutoff)".
 */
const PAIR_ATTRS: Record<string, 'neigh' | 'pair'> = {
  natom1: 'neigh', natom2: 'neigh', ntype1: 'neigh', ntype2: 'neigh', patom1: 'pair', patom2: 'pair', ptype1: 'pair', ptype2: 'pair',
};

/** The topology each atom style stores (native AtomVec bonds_allow, angles_allow, dihedrals_allow, impropers_allow). */
const ALLOW: Record<TermKind, string[]> = {
  bond: ['bond', 'angle', 'molecular', 'full', 'template', 'bpm/sphere', 'dielectric', 'amoeba'],
  angle: ['angle', 'molecular', 'full', 'template', 'dielectric', 'amoeba'],
  dihedral: ['molecular', 'full', 'template', 'dielectric', 'amoeba'],
  improper: ['molecular', 'full', 'template', 'dielectric', 'amoeba'],
};

/**
 * property/local: the topology ids and types of bonds, angles, dihedrals or impropers, or the IDs and types of the atoms
 * of each pair of the neighbor list. compute_property_local.html: "For *natom1* and *natom2*, all atom pairs in the
 * neighbor list are considered (out to the neighbor cutoff = force cutoff + neighbor skin).  For *patom1* and *patom2*,
 * the distance between the atoms must be less than the force cutoff distance for that pair to be included"; "For the
 * default setting of *type*, the pairwise cutoff defined by the pair_style command for the types of the two atoms is
 * used.  For the *radius* setting, the sum of the radii of the two particles is used as a cutoff." As native LAMMPS does
 * (src/compute_property_local.cpp at b332f28): the attributes are read until a word that is none (one kind only: "Compute
 * property/local cannot use these inputs together"), then the cutoff keyword; the pairs are those of the half list in
 * native's order (PairLocalHook, as compute pair/local), both atoms in the group, a ghost partner under newton off
 * counted once by the tag rule; a pair attribute needs a pair style.
 */
export class ComputePropertyLocal extends Compute {
  readonly style = 'property/local';
  private readonly attrs: string[];
  private readonly kind: TermKind | 'neigh' | 'pair';
  private readonly radiusCut: boolean;
  private order: PairOrderSource | null = null;
  private orderNewton = true;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError('Illegal compute property/local command: missing argument(s)');
    let kind: TermKind | 'neigh' | 'pair' | null = null;
    const attrs: string[] = [];
    let k = 0;
    for (; k < args.length; k++) {
      const w = args[k];
      const wk = PAIR_ATTRS[w] ?? (Object.keys(ATTRS) as TermKind[]).find((kk) => ATTRS[kk].atoms.includes(w) || ATTRS[kk].type === w);
      if (!wk) break;
      if (kind && kind !== wk) throw new StyleError('Compute property/local cannot use these inputs together');
      kind = wk;
      attrs.push(w);
    }
    let radiusCut = false;
    while (k < args.length) {
      if (args[k] !== 'cutoff') throw new StyleError(`Unknown compute property/local keyword: ${args[k]}`);
      if (k + 2 > args.length) throw new StyleError('Illegal compute property/local cutoff command: missing argument(s)');
      if (args[k + 1] === 'type') radiusCut = false;
      else if (args[k + 1] === 'radius') radiusCut = true;
      else throw new StyleError(`Unknown compute property/local cutoff keyword: ${args[k + 1]}`);
      k += 2;
    }
    const st = sys.state.atomStyle;
    if (kind && kind !== 'neigh' && kind !== 'pair') {
      if (isTemplateStyle(st)) throw new StyleError('Compute property/local does not (yet) work with atom_style template');
      if (!atomSubStyles(st).some((x) => ALLOW[kind as TermKind].includes(x))) throw new StyleError("Compute property/local for property that isn't allocated");
    }
    if (radiusCut && !sys.state.radius) throw new StyleError(`Compute property/local with ID ${id} requires atom attribute radius`);
    // native accepts no attribute (a cutoff keyword alone) and then gives no rows
    this.kind = kind ?? 'bond';
    this.attrs = attrs;
    this.radiusCut = radiusCut;
    this.localFlag = true;
    this.sizeLocalCols = attrs.length > 1 ? attrs.length : 0;
  }

  init(): void {
    if ((this.kind === 'neigh' || this.kind === 'pair') && !this.sys.ff.pair) throw new StyleError('No pair style is defined for compute property/local');
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    if (this.kind === 'neigh' || this.kind === 'pair') return this.pairRows();
    const s = this.sys.state;
    const kind = this.kind;
    if (!this.attrs.length) { this.localRows = 0; return new Float64Array(0); }
    const rows = termRows(this.sys, kind, this.groupBit);
    const list = s.topo[FIELD[kind]];
    const w = WIDTH[kind];
    const ncol = Math.max(1, this.attrs.length);
    const out = new Float64Array(rows.length * ncol);
    rows.forEach(({ k }, r) => {
      this.attrs.forEach((name, c) => {
        const m = ATTRS[kind].atoms.indexOf(name);
        out[r * ncol + c] = m >= 0 ? list.atoms[w * k + m] : list.type[k];
      });
    });
    this.localRows = rows.length;
    return out;
  }

  /** The pairs of the half list (ComputePropertyLocal::count_pairs): IDs and types of the two atoms. */
  private pairRows(): Float64Array<ArrayBuffer> {
    const sys = this.sys;
    const pair = sys.ff.pair;
    if (!pair) throw new StyleError('No pair style is defined for compute property/local');
    sys.forces();
    const s = sys.state;
    const nb = sys.nb;
    const list = nb.half;
    if (!list) throw new StyleError(`compute ${this.id} property/local: no half neighbour list is available`);
    const newton = sys.newtonPair !== false;
    if (!this.order || this.orderNewton !== newton) {
      this.order = gplHook<PairLocalHook>('property/local').pairOrder(newton);
      this.orderNewton = newton;
    }
    const pairs = this.order.pairs(s, nb, sys.geom, list);
    const force = this.kind === 'pair';
    const nt = pair.ntypes + 1;
    const ncol = this.attrs.length;
    const rows: number[] = [];
    const xa = nb.xall;
    const nlocal = nb.nlocal;
    const tag = s.id;
    const radius = s.radius;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const oi = nb.owner[i];
      if (!(s.mask[oi] & this.groupBit)) continue;
      const j = pairs[p + 1] & NEIGHMASK;
      const oj = nb.owner[j];
      if (!(s.mask[oj] & this.groupBit)) continue;
      if (!newton && j >= nlocal) {
        const itag = tag[oi], jtag = tag[oj];
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (xa[3 * j + 2] < xa[3 * i + 2]) continue;
          if (xa[3 * j + 2] === xa[3 * i + 2]) {
            if (xa[3 * j + 1] < xa[3 * i + 1]) continue;
            if (xa[3 * j + 1] === xa[3 * i + 1] && xa[3 * j] < xa[3 * i]) continue;
          }
        }
      }
      const ti = nb.typeall[i], tj = nb.typeall[j];
      if (force) {
        const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (this.radiusCut) {
          const radsum = radius![oi] + radius![oj];
          if (rsq >= radsum * radsum) continue;
        } else if (rsq >= pair.cutsq[ti * nt + tj]) continue;
      }
      for (let c = 0; c < ncol; c++) {
        const a = this.attrs[c];
        rows.push(a.endsWith('atom1') ? tag[oi] : a.endsWith('atom2') ? tag[oj] : a.endsWith('type1') ? ti : tj);
      }
    }
    this.localRows = ncol ? rows.length / ncol : 0;
    return Float64Array.from(rows);
  }
}

/**
 * The pair styles whose native class has a single() (single_enable) that this engine does not have yet: compute
 * pair/local refuses them naming the engine, not with native's "Pair style does not support compute pair/local" (the
 * pair-local parity matrix lists them as gaps).
 */
const SINGLE_GAPS = new Set<string>([
  'bpm/peri', 'bpm/spring', 'buck/long/coul/long', 'colloid', 'coul/cut', 'coul/cut/dielectric', 'coul/cut/soft',
  'coul/cut/soft/gapsys', 'coul/debye', 'coul/dsf', 'coul/esp', 'coul/long', 'coul/long/soft', 'coul/slater/long', 'dpd',
  'dpd/coul/slater/long', 'dpd/ext', 'dpd/fdt', 'dpd/fdt/energy', 'edpd', 'gauss', 'granular/superellipsoid', 'harmonic/cut',
  'lepton', 'lepton/coul', 'lepton/sphere', 'lj/charmm/coul/charmm', 'lj/charmm/coul/charmm/implicit', 'lj/charmm/coul/long',
  'lj/charmm/coul/long/soft', 'lj/charmmfsw/coul/charmmfsh', 'lj/charmmfsw/coul/long', 'lj/class2/coul/cut/soft',
  'lj/class2/coul/long/soft', 'lj/class2/soft', 'lj/cut/coul/cut/dielectric', 'lj/cut/coul/cut/soft',
  'lj/cut/coul/debye/dielectric', 'lj/cut/coul/long/dielectric', 'lj/cut/coul/long/soft', 'lj/cut/coul/msm/dielectric',
  'lj/cut/soft', 'lj/cut/soft/gapsys', 'lj/long/coul/long', 'lj/long/coul/long/dielectric', 'lj/sf/dipole/sf',
  'lj/switch3/coulgauss/long', 'local/density', 'mm3/switch3/coulgauss/long', 'morse/soft', 'multi/lucy', 'peri/pmb',
  'rheo/solid', 'smd/tlsph', 'soft', 'sph/heatconduction', 'tdpd', 'ylz', 'yukawa', 'yukawa/colloid',
]);

/** The first style (a hybrid's sub-styles in order) that is a SINGLE_GAPS entry without a single(), or null. */
const singleGap = (pair: Pair): string | null => {
  const styles = pair instanceof PairHybrid ? pair.subs.map((x) => x.style) : [pair];
  for (const st of styles) if (!st.singleEnabled() && (st.singleIsGap() || SINGLE_GAPS.has(st.name))) return st.name;
  return null;
};

/**
 * The row order of compute pair/local: gplHook('pair/local') (src/engine-gpl/registry/pair_local.ts) gives the pairs of
 * the half list in native's enumeration (npair_bin.cpp order, newton on or off); the engine keeps its own cutoffs.
 * The enumeration is kept between neighbor builds (native bins the atoms only in Neighbor::build, so build_one of the
 * occasional list enumerates with the bins of the last build; the distances are the current ones).
 */
export interface PairOrderSource {
  pairs(s: SimState, nb: Neighbor, g: Geometry, half: NeighList | null): Int32Array;
}
export interface PairLocalHook {
  pairOrder(newton: boolean): PairOrderSource;
}

/*
 * compute pair/local (docs.lammps.org/compute_pair_local.html):
 *   "Define a computation that calculates properties of individual pairwise interactions. ... Info about an individual
 *   pairwise interaction will only be included if both atoms in the pair are in the specified compute group, and if the
 *   current pairwise distance is less than the force cutoff distance for that interaction, as defined by the pair_style and
 *   pair_coeff commands."
 *   "The value *dist* is the distance between the pair of atoms. The values *dx*, *dy*, and *dz* are the :math:`(x,y,z)`
 *   components of the distance vector :math:`\vec{x_i} - \vec{x_j}` between the pair of atoms."
 *   "The value *eng* is the interaction energy for the pair of atoms."
 *   "The value *force* is the force acting between the pair of atoms, which is positive for a repulsive force and negative
 *   for an attractive force."
 *   "The values *fx*, *fy*, and *fz* are the :math:`(x,y,z)` components of the force vector on the first atom *i* of a pair
 *   in the neighbor list due to the second atom *j*." "For pair styles that apply non-central forces, such as granular pair
 *   styles, these values only include the :math:`(x,y,z)` components of the normal force component."
 *   "A pair style may define additional pairwise quantities which can be accessed as *p1* to *pN*, where :math:`N` is defined
 *   by the pair style. Most pair styles do not define any additional quantities, so :math:`N = 0`."
 *   "When using *pN* with pair style *hybrid*, the output will be the Nth quantity from the sub-style that computes the
 *   pairwise interaction (based on atom types). If that sub-style does not define a *pN*, the output will be 0.0."
 *   cutoff: "The optional *cutoff* keyword determines how the force cutoff distance for an interaction is determined. For the
 *   default setting of *type*, the pairwise cutoff defined by the pair_style command for the types of the two atoms is used.
 *   For the *radius* setting, the sum of the radii of the two particles is used as a cutoff."
 *   "The keyword default is cutoff = type." (the page's Default section)
 *
 * Native compute_pair_local.cpp (compute_pairs, b332f28): the test is "if (rsq >= cutsq[itype][jtype]) continue;" for type,
 * and "radsum = radius[i] + radius[j]; if (rsq >= radsum * radsum) continue;" for radius, applied to the pairs of the
 * half list (which holds the pairs inside the neighbor cutoff, so no separate force-cutoff test is made for radius).
 * The energy and force come from pair->single() when a value other than dist, dx, dy, dz is requested (singleflag);
 * pN is svector[N-1], and the native init() refuses N >= single_extra ("Pair style does not have extra field requested
 * by compute pair/local").
 * Row order: native's (the pair order of the half list, see PairLocalHook). A pair with a ghost partner under newton off
 * is counted once, by the tag rule of compute_pairs(). Triclinic boxes keep the engine's list order.
 *
 * Refused (StyleError): pN on a pair style with singleExtra() 0 (it names the style), and the radius cutoff without atom
 * radius.
 */
const PAIR_VALUES = ['dist', 'dx', 'dy', 'dz', 'eng', 'force', 'fx', 'fy', 'fz'];
const PN = /^p\d+$/;

export class ComputePairLocal extends Compute {
  readonly style = 'pair/local';
  /** The requested values: a name of PAIR_VALUES, or pN. */
  private readonly values: string[];
  /** For each value, N-1 when it is pN, else -1. */
  private readonly pIndex: number[];
  private readonly radiusCut: boolean;
  /** Any value other than dist, dx, dy, dz needs pair.single() (native singleflag). */
  private readonly needSingle: boolean;
  /** The native pair order (kept between neighbor builds, see PairOrderSource) and the newton setting it was made for. */
  private order: PairOrderSource | null = null;
  private orderNewton = true;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError(`compute ${id} pair/local needs at least one value`);
    const values: string[] = [];
    let a = 0;
    for (; a < args.length && args[a] !== 'cutoff'; a++) {
      const w = args[a];
      // As native LAMMPS does (src/compute_pair_local.cpp at b332f28): the value loop stops at an unknown word, which the
      // keyword loop then names
      if (PN.test(w)) {
        if (Number(w.slice(1)) < 1) throw new StyleError(`Invalid keyword ${w} in compute pair/local command`);
      } else if (!PAIR_VALUES.includes(w)) {
        throw new StyleError(`Unknown compute pair/local keyword: ${w}`);
      }
      values.push(w);
    }
    // keyword cutoff type|radius (native: "cutstyle = TYPE" by default; the keywords follow the values)
    let radiusCut = false;
    for (; a < args.length; a += 2) {
      if (args[a] !== 'cutoff') throw new StyleError(`Unknown compute pair/local keyword: ${args[a]}`);
      if (a + 1 >= args.length) throw missingArgs('compute pair/local cutoff');
      const c = args[a + 1];
      if (c !== 'type' && c !== 'radius') throw new StyleError(`Unknown compute pair/local cutoff keyword: ${c}`);
      radiusCut = c === 'radius';
    }
    this.values = values;
    this.pIndex = values.map((v) => (PN.test(v) ? Number(v.slice(1)) - 1 : -1));
    this.radiusCut = radiusCut;
    this.needSingle = values.some((v) => !['dist', 'dx', 'dy', 'dz'].includes(v));
    this.localFlag = true;
    this.sizeLocalCols = values.length > 1 ? values.length : 0;
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    const sys = this.sys;
    const pair = sys.ff.pair;
    // As native LAMMPS does (src/compute_pair_local.cpp init at b332f28), when a value needs single(): the pair style,
    // its single_enable, then the pN fields of single_extra
    if (!pair) throw new StyleError(this.needSingle ? 'No pair style is defined for compute pair/local' : `compute ${this.id} pair/local needs a pair_style (none is defined)`);
    if (this.needSingle && !pair.singleEnabled()) {
      const gap = singleGap(pair);
      if (gap) throw new StyleError(`compute ${this.id} pair/local: the browser engine has no single() for pair style ${gap} yet (native LAMMPS has one)`);
      throw new StyleError('Pair style does not support compute pair/local');
    }
    const nExtra = pair.singleExtra();
    for (let c = 0; c < this.pIndex.length; c++) {
      const n = this.pIndex[c];
      if (n >= 0 && n >= nExtra) throw new StyleError('Pair style does not have extra field requested by compute pair/local');
    }
    if (this.radiusCut && !sys.state.radius) throw new StyleError(`Compute pair/local with ID ${this.id} requires atom attribute radius`);
    sys.forces();
    const s = sys.state;
    const nb = sys.nb;
    const list = nb.half;
    if (!list) throw new StyleError(`compute ${this.id} pair/local: no half neighbour list is available`);
    // special-bond factors of the last force evaluation (index = special bits of the neighbour)
    const { specialLJ: sLJ, specialCoul: sCoul } = sys.ff as unknown as { specialLJ: Float64Array; specialCoul: Float64Array };
    const newton = sys.newtonPair !== false;
    if (!this.order || this.orderNewton !== newton) {
      this.order = gplHook<PairLocalHook>('pair/local').pairOrder(newton);
      this.orderNewton = newton;
    }
    const pairs = this.order.pairs(s, nb, sys.geom, list);
    const nt = pair.ntypes + 1;
    const ncol = this.values.length;
    const rows: number[] = [];
    const xa = nb.xall;
    const nlocal = nb.nlocal;
    const tag = s.id;
    const radius = s.radius;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const oi = nb.owner[i];
      if (!(s.mask[oi] & this.groupBit)) continue;
      const jEnc = pairs[p + 1];
      const j = jEnc & NEIGHMASK;
      const sb = jEnc >>> SBBITS;
      const oj = nb.owner[j];
      if (!(s.mask[oj] & this.groupBit)) continue;
      // newton off, a ghost partner: the pair is counted once, by the tag rule of compute_pairs() (compute_pair_local.cpp)
      if (!newton && j >= nlocal) {
        const itag = tag[oi], jtag = tag[oj];
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (xa[3 * j + 2] < xa[3 * i + 2]) continue;
          if (xa[3 * j + 2] === xa[3 * i + 2]) {
            if (xa[3 * j + 1] < xa[3 * i + 1]) continue;
            if (xa[3 * j + 1] === xa[3 * i + 1] && xa[3 * j] < xa[3 * i]) continue;
          }
        }
      }
      const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const ti = nb.typeall[i], tj = nb.typeall[j];
      if (this.radiusCut) {
        const radsum = radius![oi] + radius![oj];
        if (rsq >= radsum * radsum) continue;
      } else if (rsq >= pair.cutsq[ti * nt + tj]) continue;
      const factorLJ = sb === 0 ? 1 : sLJ[sb];
      const factorCoul = sb === 0 ? 1 : sCoul[sb];
      const r = Math.sqrt(rsq);
      const res = this.needSingle ? pair.single!(oi, oj, ti, tj, rsq, factorCoul, factorLJ, nb.qall[i], nb.qall[j], { dx, dy, dz }) : null;
      const fpair = res ? res.fforce : 0;
      const named: Record<string, number> = {
        dist: r, dx, dy, dz, eng: res ? res.eng : 0, force: fpair * r, fx: fpair * dx, fy: fpair * dy, fz: fpair * dz,
      };
      for (let c = 0; c < ncol; c++) {
        const n = this.pIndex[c];
        rows.push(n >= 0 ? (res?.svector?.[n] ?? 0) : named[this.values[c]]);
      }
    }
    this.localRows = rows.length / ncol;
    return Float64Array.from(rows);
  }
}
