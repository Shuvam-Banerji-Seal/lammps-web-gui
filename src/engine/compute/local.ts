import { Compute } from './compute';
import { StyleError, newAccum, type Accum, type Bonded, type BondedCompute } from '../force/types';
import { dihedralGeometry } from '../force/bonded_util';
import { buildAtomMap, massOf, nativeOrder } from '../atoms';
import type { System } from '../system';
import type { SimState, TopoList } from '../types';
import type { Geometry } from '../domain';

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
 * Not implemented (throw StyleError): the set keyword, v_name values, bN style quantities,
 * omega, and pair/local (the pair neighbour order is not reproduced).
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
const termEnergy = (s: SimState, geom: Geometry, kind: TermKind, style: Bonded, list: TopoList, k: number, map: Int32Array, scratch: Float64Array): number => {
  const w = WIDTH[kind];
  const one: TopoList = { n: 1, width: w as 2 | 3 | 4, type: Int32Array.of(list.type[k]), atoms: list.atoms.slice(w * k, w * k + w) };
  const shadow = Object.create(s) as SimState;
  shadow.topo = { ...s.topo, [FIELD[kind]]: one };
  const acc: Accum = newAccum();
  const bc: BondedCompute = { s: shadow, geom, map, f: scratch, acc, virial: acc.vbond, eatom: null, vatom: null };
  style.compute(bc);
  return acc[ENERGY[kind]];
};

/** Minimum-image displacement a - b. */
const disp = (s: SimState, geom: Geometry, a: number, b: number): number[] => {
  const d = [s.x[3 * a] - s.x[3 * b], s.x[3 * a + 1] - s.x[3 * b + 1], s.x[3 * a + 2] - s.x[3 * b + 2]];
  geom.minimumImage(d);
  return d;
};

const norm = (d: number[]): number => Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

/** A local compute over bonds, angles, dihedrals or impropers with value keywords. */
export class ComputeTermLocal extends Compute {
  readonly style: string;
  private readonly values: string[];

  constructor(sys: System, id: string, group: string, args: string[], private readonly kind: TermKind) {
    super(sys, id, group, args);
    this.style = `${kind}/local`;
    if (!args.length) throw new StyleError(`compute ${id} ${this.style} needs at least one value`);
    const values: string[] = [];
    for (let a = 0; a < args.length; a++) {
      const w = args[a];
      if (w === 'set') throw new StyleError(`compute ${id} ${this.style}: the set keyword is not supported by the browser engine`);
      if (w.startsWith('v_')) throw new StyleError(`compute ${id} ${this.style}: v_name values are not supported by the browser engine`);
      if (/^b\d+$/.test(w)) throw new StyleError(`compute ${id} ${this.style}: bN bond-style quantities are not supported by the browser engine`);
      if (!VALUES[kind].includes(w)) {
        const extra = kind === 'bond' && w === 'omega' ? ' (omega is not supported by the browser engine)' : '';
        throw new StyleError(`compute ${id} ${this.style}: unknown value '${w}'${extra}`);
      }
      values.push(w);
    }
    this.values = values;
    this.localFlag = true;
    this.sizeLocalCols = values.length > 1 ? values.length : 0;
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
    const needEnergy = need('engpot') || need('force') || need('fx') || need('fy') || need('fz') || need('eng');
    const style = sys.ff[STYLE_FIELD[kind]];
    if (needEnergy && !style) throw new StyleError(`compute ${this.id} ${this.style} needs a ${kind}_style`);
    const list = s.topo[FIELD[kind]];
    const map = buildAtomMap(s);
    const scratch = new Float64Array(3 * s.n);
    const mvv2e = s.units.mvv2e;
    const out = new Float64Array(rows.length * ncol);
    const bc: BondedCompute = { s, geom, map, f: scratch, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
    rows.forEach(({ k, idx }, r) => {
      const vals: Record<string, number> = {};
      if (kind === 'bond') {
        const [i, j] = idx;
        const d = disp(s, geom, i, j);
        const dist = norm(d);
        const u = dist > 0 ? d.map((c) => c / dist) : [0, 0, 0];
        let e = 0, fi = [0, 0, 0];
        if (needEnergy && list.type[k] > 0) {
          e = termEnergy(s, geom, kind, style!, list, k, map, scratch);
          fi = [scratch[3 * i], scratch[3 * i + 1], scratch[3 * i + 2]];
        }
        if (needEnergy) for (const a of [i, j]) for (let c = 0; c < 3; c++) scratch[3 * a + c] = 0;
        const force = dist > 0 ? (fi[0] * d[0] + fi[1] * d[1] + fi[2] * d[2]) / dist : 0;
        vals.dist = dist; vals.dx = d[0]; vals.dy = d[1]; vals.dz = d[2];
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
        const d1 = disp(s, geom, i, j), d3 = disp(s, geom, kk, j);
        const r1 = norm(d1), r3 = norm(d3);
        const c = r1 > 0 && r3 > 0 ? Math.max(-1, Math.min(1, (d1[0] * d3[0] + d1[1] * d3[1] + d1[2] * d3[2]) / (r1 * r3))) : 1;
        vals.theta = Math.acos(c) * RAD2DEG;
        if (need('eng')) {
          vals.eng = list.type[k] > 0 ? termEnergy(s, geom, kind, style!, list, k, map, scratch) : 0;
          for (const a of idx) for (let q = 0; q < 3; q++) scratch[3 * a + q] = 0;
        }
      } else if (kind === 'dihedral') {
        const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
        vals.phi = dihedralGeometry(bc, idx[0], idx[1], idx[2], idx[3], grad, rel) * RAD2DEG;
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

/** property/local: the topology ids and types of bonds, angles, dihedrals or impropers. */
export class ComputePropertyLocal extends Compute {
  readonly style = 'property/local';
  private readonly attrs: string[];
  private readonly kind: TermKind;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError(`compute ${id} property/local needs at least one attribute`);
    let kind: TermKind | null = null;
    for (const w of args) {
      if (w === 'cutoff' || w === 'type' || w === 'radius') {
        throw new StyleError(`compute ${id} property/local: the cutoff keyword is only used with pair attributes, which are not supported by the browser engine`);
      }
      const k = (Object.keys(ATTRS) as TermKind[]).find((kk) => ATTRS[kk].atoms.includes(w) || ATTRS[kk].type === w);
      if (!k) {
        throw new StyleError(`compute ${id} property/local: attribute '${w}' is not supported by the browser engine (bond, angle, dihedral and improper attributes are)`);
      }
      if (kind && kind !== k) throw new StyleError(`compute ${id} property/local: bond, angle, dihedral and improper attributes cannot be mixed`);
      kind = k;
    }
    this.kind = kind!;
    this.attrs = args;
    this.localFlag = true;
    this.sizeLocalCols = args.length > 1 ? args.length : 0;
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    const s = this.sys.state;
    const kind = this.kind;
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
}

/*
 * compute pair/local (docs.lammps.org/compute_pair_local.html):
 *   "Define a computation that calculates properties of individual pairwise
 *   interactions. ... Info about an individual pairwise interaction will only
 *   be included if both atoms in the pair are in the specified compute group,
 *   and if the current pairwise distance is less than the force cutoff distance
 *   for that interaction, as defined by the pair_style and pair_coeff commands."
 *   "The value *dist* is the distance between the pair of atoms. The values
 *   *dx*, *dy*, and *dz* are the :math:`(x,y,z)` components of the distance vector
 *   :math:`\vec{x_i} - \vec{x_j}` between the pair of atoms."
 *   "The value *eng* is the interaction energy for the pair of atoms."
 *   "The value *force* is the force acting between the pair of atoms, which is
 *   positive for a repulsive force and negative for an attractive force."
 *   "The values *fx*, *fy*, and *fz* are the (x,y,z) components of the force
 *   vector on the first atom *i* of a pair in the neighbor list due to the second
 *   atom *j*."
 *   Keywords: "cutoff = type or radius" and the pN values are not supported; they
 *   throw StyleError.
 *
 * Row order: the rows follow this engine's half neighbour list (owned i, then its
 * list). That is NOT native's row order (native's list is built by its own bins), so
 * the values are only meaningful through order-independent aggregates (fix ave/histo,
 * compute reduce) — see tests/oracle/w12local_*.in.
 */
const PAIR_VALUES = ['dist', 'dx', 'dy', 'dz', 'eng', 'force', 'fx', 'fy', 'fz'];

export class ComputePairLocal extends Compute {
  readonly style = 'pair/local';
  private readonly values: string[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError(`compute ${id} pair/local needs at least one value`);
    const values: string[] = [];
    for (let a = 0; a < args.length; a++) {
      const w = args[a];
      if (w === 'cutoff') throw new StyleError(`compute ${id} pair/local: the cutoff keyword is not supported by the browser engine`);
      if (/^p\d+$/.test(w)) throw new StyleError(`compute ${id} pair/local: pN quantities are not supported by the browser engine (no pair style defines them)`);
      if (!PAIR_VALUES.includes(w)) throw new StyleError(`compute ${id} pair/local: unknown value '${w}' (use ${PAIR_VALUES.join(', ')})`);
      values.push(w);
    }
    this.values = values;
    this.localFlag = true;
    this.sizeLocalCols = values.length > 1 ? values.length : 0;
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    const sys = this.sys;
    const pair = sys.ff.pair;
    if (!pair || !pair.single) throw new StyleError(`compute ${this.id} pair/local needs a pair_style that supports single() (the engine's pairwise evaluation)`);
    sys.forces();
    const s = sys.state;
    const nb = sys.nb;
    const list = nb.half;
    if (!list) throw new StyleError(`compute ${this.id} pair/local: no half neighbour list is available`);
    // special-bond factors of the last force evaluation (index = special bits of the neighbour)
    const sLJ = (sys.ff as unknown as { specialLJ: Float64Array }).specialLJ;
    const nt = pair.ntypes + 1;
    const ncol = this.values.length;
    const rows: number[] = [];
    const nb0 = list.neighbors;
    const xa = nb.xall;
    const NEIGH_MASK = (1 << 30) - 1;
    // rows are collected as the list is walked; each row fills `ncol` values
    for (let i = 0; i < list.inum; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb0[k];
        const j = jj & NEIGH_MASK;
        const sb = jj >>> 30;
        if (!(s.mask[nb.owner[j]] & this.groupBit)) continue;
        const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const ti = nb.typeall[i], tj = nb.typeall[j];
        if (rsq >= pair.cutsq[ti * nt + tj]) continue;
        const factorLJ = sb === 0 ? 1 : sLJ[sb];
        const r = Math.sqrt(rsq);
        const res = pair.single(nb.owner[i], nb.owner[j], ti, tj, rsq, factorLJ, factorLJ, nb.qall[i], nb.qall[j]);
        const force = res.fforce * r;
        const fx = res.fforce * dx, fy = res.fforce * dy, fz = res.fforce * dz;
        const vals: Record<string, number> = { dist: r, dx, dy, dz, eng: res.eng, force, fx, fy, fz };
        for (const name of this.values) rows.push(vals[name]);
      }
    }
    this.localRows = rows.length / ncol;
    return Float64Array.from(rows);
  }
}
