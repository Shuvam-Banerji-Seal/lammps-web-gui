/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_filter_corotate.cpp and fix_filter_corotate.h at commit b332f28 (contributing author: Lukas
 * Fath, KIT; "some subroutines are from fix_shake.cpp"), with Special::onetwo_build_newton (src/special.cpp, the order of
 * the 1-2 partners the cluster finder reads) and Domain::minimum_image. Copyright (2003) Sandia Corporation, distributed
 * under the GNU General Public License. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix filter/corotate (docs.lammps.org/fix_filter_corotate.html; L. Fath, M. Hochbruck and C. V. Singh, J. Comput. Phys.
 * 333, 180 (2017)): the mollified impulse method of rRESPA. The clusters are found as fix shake finds its clusters (bond
 * types b, angle types a, atom types t, masses m). For the force evaluation of the outermost rRESPA level the cluster atoms
 * move to the filtered positions (the cluster's centre of mass plus its reference shape rotated into the cluster's frame),
 * and the forces of that level are filtered with the transposed derivative of those positions. The other levels see the
 * real positions. run/respa.ts calls the hooks of RespaOuterFilter.
 */
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import { isMolecularStyle, nativeOrder } from '../../engine/atoms';
import { cos, sin } from '../libm/sincos';
import type { NeighList } from '../../engine/neighbor';
import { nativePairs } from '../neighbor/nativeOrder';

/** fix_filter_corotate.cpp: "static constexpr double MASSDELTA = 0.1;" */
const MASSDELTA = 0.1;
/** MathConst::MY_PI */
const MY_PI = 3.14159265358979323846;

/** sgn() of fix_filter_corotate.h: -1, 0 or 1. */
const sgn = (v: number): number => (v > 0 ? 1 : 0) - (v < 0 ? 1 : 0);

/** One cluster of the list (pre_neighbor) with its reference shape and selections. */
interface Cluster {
  /** atom IDs: the central atom first (shake_atom) */
  atoms: number[];
  /** shake_flag: 2 (two atoms), 1 (angle cluster of three), 3, 4, 5 */
  flag: number;
  types: number[];
  q0: Float64Array;
  select1: number[];
  select2: number[];
  /** the derivative of the filtered positions (3N x 3N, transposed), from the last filter of the positions */
  derv: Float64Array;
}

export class FixFilterCorotate extends Fix {
  readonly style = 'filter/corotate';
  readonly respaOuterFilter = true;
  private readonly bondFlag: Uint8Array;
  private readonly angleFlag: Uint8Array;
  private readonly typeFlag: Uint8Array;
  private readonly massList: number[] = [];
  /** per atom ID: shake_flag, shake_atom (5), shake_type (4) of find_clusters */
  private readonly shakeFlag = new Map<number, number>();
  private readonly shakeAtom = new Map<number, number[]>();
  private readonly shakeType = new Map<number, number[]>();
  private bondDistance = new Float64Array(0);
  private angleDistance = new Float64Array(0);
  private list: Cluster[] = [];
  private map: Int32Array = new Int32Array(0);
  private filtered = new Float64Array(0);
  /** the native-side half list of the last neighbor build (respaHalfList) */
  private half: NeighList | null = null;
  private halfKey: { nbuild: number; order: Int32Array | null; src: NeighList | null } = { nbuild: -1, order: null, src: null };

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const s = sys.state;
    if (!isMolecularStyle(s.atomStyle)) throw new StyleError('Cannot use fix filter/corotate with non-molecular system');
    if (s.atomStyle.startsWith('template')) throw new StyleError('Cannot use fix filter/corotate with molecular template system');
    const topo = s.topo;
    this.bondFlag = new Uint8Array(topo.nbondtypes + 1);
    this.angleFlag = new Uint8Array(topo.nangletypes + 1);
    this.typeFlag = new Uint8Array(s.ntypes + 1);

    // parse args for bond and angle types; "m" values are kept in a list
    let mode = '';
    const intArg = (w: string): number => {
      if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
      return Number(w);
    };
    for (const w of args) {
      if (w === 'b') mode = 'b';
      else if (w === 'a') mode = 'a';
      else if (w === 't') mode = 't';
      else if (w === 'm') {
        mode = 'm';
        // atom->check_mass: every type needs a mass
        for (let t = 1; t <= s.ntypes; t++) {
          if (!s.rmass && !(s.massByType[t] > 0)) throw new StyleError(`Not all per-type masses are set. Type ${t} is missing.`);
        }
      } else if (/^[A-Za-z]/.test(w)) break;
      else if (mode === 'b') {
        const i = intArg(w);
        if (i < 1 || i > topo.nbondtypes) throw new StyleError('Invalid bond type index for fix filter/corotate');
        this.bondFlag[i] = 1;
      } else if (mode === 'a') {
        const i = intArg(w);
        if (i < 1 || i > topo.nangletypes) throw new StyleError('Invalid angle type index for fix filter/corotate');
        this.angleFlag[i] = 1;
      } else if (mode === 't') {
        const i = intArg(w);
        if (i < 1 || i > s.ntypes) throw new StyleError('Invalid atom type index for fix filter/corotate');
        this.typeFlag[i] = 1;
      } else if (mode === 'm') {
        const m = Number(w);
        if (w.trim() === '' || !Number.isFinite(m)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
        if (m === 0.0) throw new StyleError('Invalid atom mass for fix filter/corotate');
        if (this.massList.length === s.ntypes) throw new StyleError('Too many masses for fix filter/corotate');
        this.massList.push(m);
      } else throw new StyleError('Illegal fix filter/corotate command');
    }
    this.findClusters();
  }

  private masscheck(massone: number): number {
    for (const m of this.massList) if (Math.abs(m - massone) <= MASSDELTA) return 1;
    return 0;
  }

  private massOf(i: number): number {
    const s = this.sys.state;
    return s.rmass ? s.rmass[i] : s.massByType[s.type[i]];
  }

  /**
   * The bonds stored on each atom, as native stores them with newton_bond on (on the first atom of the Bonds line, in
   * file order): per atom index, the topology indices of its bonds.
   */
  private storedBonds(): number[][] {
    const s = this.sys.state;
    const out: number[][] = Array.from({ length: s.n }, () => []);
    const b = s.topo.bonds;
    const map = this.idMap();
    for (let e = 0; e < b.n; e++) {
      const i = map[b.atoms[2 * e]];
      if (i >= 0) out[i].push(e);
    }
    return out;
  }

  private idMap(): Int32Array {
    const s = this.sys.state;
    let maxId = 0;
    for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
    const map = new Int32Array(maxId + 1).fill(-1);
    for (let i = 0; i < s.n; i++) map[s.id[i]] = i;
    return map;
  }

  /** bondtype_findset(i, n1, n2, 0): the type of the bond (n1, n2) stored on atom i, 0 when there is none. */
  private bondtypeFind(stored: number[][], i: number, n1: number, n2: number): number {
    const s = this.sys.state;
    const b = s.topo.bonds;
    const ti = s.id[i];
    for (const e of stored[i]) {
      const other = b.atoms[2 * e + 1];
      if ((n1 === ti && n2 === other) || (n1 === other && n2 === ti)) return b.type[e];
    }
    return 0;
  }

  /** angletype_findset(i, n1, n2, 0): the type of the angle n1-i-n2 stored on atom i (its central atom), 0 when none. */
  private angletypeFind(i: number, n1: number, n2: number): number {
    const s = this.sys.state;
    const A = s.topo.angles;
    const ti = s.id[i];
    for (let e = 0; e < A.n; e++) {
      if (A.atoms[3 * e + 1] !== ti) continue;
      const a1 = A.atoms[3 * e], a3 = A.atoms[3 * e + 2];
      if ((a1 === n1 && a3 === n2) || (a1 === n2 && a3 === n1)) return A.type[e];
    }
    return 0;
  }

  /** FixFilterCorotate::find_clusters, on one processor (every bond partner is an owned atom). */
  private findClusters(): void {
    const sys = this.sys;
    const s = sys.state;
    const nlocal = s.n;
    const map = this.idMap();
    const stored = this.storedBonds();
    const order = nativeOrder(s);
    const b = s.topo.bonds;

    // the 1-2 partners in the order of Special::onetwo_build_newton: atoms in local order, each atom's stored bonds in
    // order, the partner appended to the atom's list and the atom to the partner's
    const partners: number[][] = Array.from({ length: nlocal }, () => []);
    for (let k = 0; k < nlocal; k++) {
      const i = order[k];
      for (const e of stored[i]) {
        if (b.type[e] === 0) continue;
        const pid = b.atoms[2 * e + 1];
        partners[i].push(pid);
        const m = map[pid];
        if (m >= 0) partners[m].push(s.id[i]);
      }
    }
    // Special::dedup: a partner that occurs twice is listed once, at its first place
    for (let i = 0; i < nlocal; i++) partners[i] = partners[i].filter((p, k, a) => a.indexOf(p) === k);

    const npartner = partners.map((p) => p.length);
    const nshake = new Int32Array(nlocal);
    const pMask: number[][] = [], pType: number[][] = [], pMass: number[][] = [], pBond: number[][] = [];
    const pShake: number[][] = [], pNshake: number[][] = [];
    for (let i = 0; i < nlocal; i++) {
      pMask.push([]); pType.push([]); pMass.push([]); pBond.push([]); pShake.push([]); pNshake.push([]);
      for (let j = 0; j < npartner[i]; j++) {
        pMask[i].push(0); pType[i].push(0); pMass[i].push(0); pBond[i].push(0);
        const m = map[partners[i][j]];
        if (m >= 0 && m < nlocal) {
          pMask[i][j] = s.mask[m];
          pType[i][j] = s.type[m];
          if (this.massList.length) pMass[i][j] = this.masscheck(this.massOf(m));
          let n = this.bondtypeFind(stored, i, s.id[i], partners[i][j]);
          if (n) pBond[i][j] = n;
          else {
            n = this.bondtypeFind(stored, m, s.id[i], partners[i][j]);
            if (n) pBond[i][j] = n;
          }
        }
      }
    }

    // error check for unfilled partner info
    for (let i = 0; i < nlocal; i++) {
      for (let j = 0; j < npartner[i]; j++) {
        if (pType[i][j] === 0) throw new StyleError('Did not find fix filter/corotate partner info');
        if (!(s.mask[i] & this.groupBit)) continue;
        if (!(pMask[i][j] & this.groupBit)) continue;
        if (pBond[i][j] === 0) throw new StyleError('Did not find fix filter/corotate partner info');
      }
    }

    // set partner_shake and nshake
    for (let i = 0; i < nlocal; i++) {
      nshake[i] = 0;
      for (let j = 0; j < npartner[i]; j++) {
        pShake[i].push(0);
        if (!(s.mask[i] & this.groupBit)) continue;
        if (!(pMask[i][j] & this.groupBit)) continue;
        if (pBond[i][j] <= 0) continue;
        if (this.bondFlag[pBond[i][j]]) { pShake[i][j] = 1; nshake[i]++; continue; }
        if (this.typeFlag[s.type[i]] || this.typeFlag[pType[i][j]]) { pShake[i][j] = 1; nshake[i]++; continue; }
        if (this.massList.length) {
          if (pMass[i][j]) { pShake[i][j] = 1; nshake[i]++; continue; }
          if (this.masscheck(this.massOf(i))) { pShake[i][j] = 1; nshake[i]++; continue; }
        }
      }
    }
    for (let i = 0; i < nlocal; i++) {
      for (let j = 0; j < npartner[i]; j++) pNshake[i].push(nshake[map[partners[i][j]]]);
    }

    // error checks: no more than 4 partners, no connected clusters
    for (let i = 0; i < nlocal; i++) if (nshake[i] > 4) throw new StyleError('Cluster of more than 5 atoms');
    for (let i = 0; i < nlocal; i++) {
      if (nshake[i] <= 1) continue;
      for (let j = 0; j < npartner[i]; j++) if (pShake[i][j] && pNshake[i][j] > 1) throw new StyleError('Clusters are connected');
    }

    // set SHAKE arrays that are stored with atoms
    const flag = new Int32Array(nlocal);
    const atoms: number[][] = [], types: number[][] = [];
    for (let i = 0; i < nlocal; i++) {
      atoms.push([0, 0, 0, 0, 0]);
      types.push([0, 0, 0, 0]);
      if (nshake[i] === 1) {
        let j = 0;
        for (; j < npartner[i]; j++) if (pShake[i][j]) break;
        if (pNshake[i][j] === 1 && s.id[i] < partners[i][j]) {
          flag[i] = 2;
          atoms[i][0] = s.id[i];
          atoms[i][1] = partners[i][j];
          types[i][0] = pBond[i][j];
        }
      }
      if (nshake[i] > 1) {
        flag[i] = 1;
        atoms[i][0] = s.id[i];
        for (let j = 0; j < npartner[i]; j++) {
          if (pShake[i][j]) {
            const m = flag[i];
            atoms[i][m] = partners[i][j];
            types[i][m - 1] = pBond[i][j];
            flag[i]++;
          }
        }
      }
      if (nshake[i] === 2) {
        const n = this.angletypeFind(i, atoms[i][1], atoms[i][2]);
        if (n <= 0) continue;
        flag[i] = 1;
        types[i][2] = n;
      }
    }
    // copy the SHAKE info of each central atom to its partners
    for (let i = 0; i < nlocal; i++) {
      if (flag[i] === 0) continue;
      for (let j = 0; j < npartner[i]; j++) {
        if (pShake[i][j] === 0) continue;
        const m = map[partners[i][j]];
        flag[m] = flag[i];
        atoms[m] = [...atoms[i]];
        types[m] = [...types[i]];
      }
    }
    let c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;
    for (let i = 0; i < nlocal; i++) {
      const id = s.id[i];
      this.shakeFlag.set(id, flag[i]);
      this.shakeAtom.set(id, atoms[i]);
      this.shakeType.set(id, types[i]);
      if (flag[i] === 1) c1++;
      else if (flag[i] === 2) c2++;
      else if (flag[i] === 3) c3++;
      else if (flag[i] === 4) c4++;
      else if (flag[i] === 5) c5++;
    }
    sys.log(`  ${Math.trunc(c2 / 2)} = # of size 2 clusters`);
    sys.log(`  ${Math.trunc(c3 / 3)} = # of size 3 clusters`);
    sys.log(`  ${Math.trunc(c4 / 4)} = # of size 4 clusters`);
    sys.log(`  ${Math.trunc(c5 / 5)} = # of size 5 clusters`);
    sys.log(`  ${Math.trunc(c1 / 3)} = # of frozen angles`);
  }

  /** FixFilterCorotate::init: the checks, and the equilibrium bond lengths and angles of the bond and angle styles. */
  override init(): void {
    const sys = this.sys;
    if (sys.fixes.filter((f) => f.style === 'filter/corotate').length > 1) throw new StyleError('More than one fix filter/corotate');
    if (sys.fixes.some((f) => f.style.startsWith('shake'))) throw new StyleError('Both fix shake and fix filter/corotate detected.');
    if (sys.fixes.some((f) => f.style.startsWith('rattle'))) throw new StyleError('Both fix rattle and fix filter/corotate detected.');
    if (sys.fixes.some((f) => f.style.startsWith('ilves'))) throw new StyleError('Both fix ilves and fix filter/corotate detected.');
    if (!sys.respa) throw new StyleError('Fix filter/corotate requires rRESPA!');
    const bond = sys.ff.bond;
    if (!bond) throw new StyleError('Bond potential must be defined for fix filter/corotate');
    const topo = sys.state.topo;
    this.bondDistance = new Float64Array(topo.nbondtypes + 1);
    for (let i = 1; i <= topo.nbondtypes; i++) this.bondDistance[i] = bond.equilibrium(i);
    this.angleDistance = new Float64Array(topo.nangletypes + 1);
    const angle = sys.ff.angle;
    for (let i = 1; i <= topo.nangletypes; i++) {
      // Angle::equilibrium_angle: the stored theta0 in radians
      this.angleDistance[i] = angle ? (angle.equilibriumRadians ? angle.equilibriumRadians(i) : angle.equilibrium(i) * (MY_PI / 180)) : Number.NaN;
    }
  }

  /** Native setup (filter of the setup forces) is done by run/respa.ts; no post-force step of its own. */
  override setup(): void {}

  /** FixFilterCorotate::pre_neighbor: the list of clusters and their reference shapes q0. */
  respaPreNeighbor(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const map = this.idMap();
    this.map = map;
    this.list = [];
    const x = s.x;
    const mapOf = (id: number): number => {
      const m = id < map.length ? map[id] : -1;
      return m;
    };
    // build list of clusters I compute: each cluster once, at the atom with the lowest local index
    const order = nativeOrder(s);
    const local = new Int32Array(s.n);
    for (let k = 0; k < s.n; k++) local[order[k]] = k;
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      const id = s.id[i];
      const flag = this.shakeFlag.get(id) ?? 0;
      if (!flag) continue;
      const at = this.shakeAtom.get(id)!;
      const n = flag === 2 ? 2 : flag === 1 || flag === 3 ? 3 : flag;
      let lowest = true;
      for (let c = 0; c < n; c++) {
        const m = mapOf(at[c]);
        if (m < 0) throw new StyleError(`Cluster atoms ${at.slice(0, n).join(' ')} missing on proc 0 at step ${s.step}`);
        if (local[m] < k) lowest = false;
      }
      if (!lowest) continue;
      const types = this.shakeType.get(id)!;
      this.list.push({ atoms: at, flag, types, q0: new Float64Array(15), select1: [], select2: [], derv: new Float64Array(225) });
    }

    // also build reference config lists
    for (const cl of this.list) {
      let N = cl.flag;
      const q = cl.q0;
      // switch cluster type 3 to angle cluster
      if (N === 3 && cl.types[2] === 0) cl.types[2] = this.angletypeFind(mapOf(cl.atoms[0]), cl.atoms[1], cl.atoms[2]);
      if (N === 1) N = 3;
      const massAt = (c: number): number => this.massOf(mapOf(cl.atoms[c]));
      if (N === 2) {
        const r0 = this.bondDistance[cl.types[0]];
        const m1 = massAt(0), m2 = massAt(1);
        const a1 = -m2 / (m1 + m2) * r0;
        const a2 = m1 / (m1 + m2) * r0;
        q[0] = a1; q[1] = 0; q[2] = 0;
        q[3] = a2; q[4] = 0; q[5] = 0;
        cl.select1 = [1];
        cl.select2 = [];
      } else if (N === 3) {
        const r0 = this.bondDistance[cl.types[0]];
        const r1 = this.bondDistance[cl.types[1]];
        if (!cl.types[2]) throw new StyleError(`fix filter/corotate: the 3-atom cluster ${cl.atoms.slice(0, 3).join(' ')} has no angle (native reads an undefined equilibrium angle)`);
        const theta0 = this.angleDistance[cl.types[2]];
        const m1 = massAt(0), m2 = massAt(1), m3 = massAt(2);
        const mAll = m1 + m2 + m3;
        const alpha1 = 0.5 * (MY_PI - theta0);
        const xcenter = m2 / mAll * r0 * sin(alpha1) + m3 / mAll * r1 * sin(alpha1);
        const ycenter = -m2 / mAll * r0 * cos(alpha1) + m3 / mAll * r1 * cos(alpha1);
        q[0] = -xcenter; q[1] = -ycenter; q[2] = 0;
        q[3] = r0 * sin(alpha1) - xcenter; q[4] = r0 * cos(alpha1) - ycenter; q[5] = 0;
        q[6] = r1 * sin(alpha1) - xcenter; q[7] = -r1 * cos(alpha1) - ycenter; q[8] = 0;
        cl.select1 = [1, 2];
        cl.select2 = [1];
      } else if (N === 4 || N === 5) {
        const oxy = mapOf(cl.atoms[0]), atom1 = mapOf(cl.atoms[1]), atom2 = mapOf(cl.atoms[2]), atom3 = mapOf(cl.atoms[3]);
        const alpha1 = MY_PI / 2.57; // roughly 70 degrees
        const alpha2 = MY_PI / 3;
        if (N === 4) {
          const r0 = this.bondDistance[cl.types[0]], r1 = this.bondDistance[cl.types[1]], r2 = this.bondDistance[cl.types[2]];
          const m1 = massAt(0), m2 = massAt(1), m3 = massAt(2), m4 = massAt(3);
          const mAll = m1 + m2 + m3 + m4;
          const xcenter = (m2 * r0 + m3 * r1 + m4 * r2) / mAll * cos(alpha1);
          q[0] = -xcenter; q[1] = 0.0; q[2] = 0.0;
          q[3] = r0 * cos(alpha1) - xcenter; q[4] = -r0 * sin(alpha1); q[5] = 0.0;
          q[6] = r1 * cos(alpha1) - xcenter; q[7] = r1 * sin(alpha1) * cos(alpha2); q[8] = -r1 * sin(alpha1) * sin(alpha2);
          q[9] = r2 * cos(alpha1) - xcenter; q[10] = r2 * sin(alpha1) * cos(alpha2); q[11] = r2 * sin(alpha1) * sin(alpha2);
          cl.select1 = [1, 2, 3];
          cl.select2 = [2, 3];
        } else {
          const r1 = this.bondDistance[cl.types[3]];
          const r0 = this.bondDistance[cl.types[0]];
          const m1 = massAt(0), m2 = massAt(1), m3 = massAt(2), m4 = massAt(3), m5 = massAt(4);
          const mAll = m1 + m2 + m3 + m4 + m5;
          const xcenter = -(m2 + m3 + m4) / mAll * r0 * cos(alpha1) + r1 * m5 / mAll;
          q[0] = -xcenter; q[1] = 0.0; q[2] = 0.0;
          q[3] = -r0 * cos(alpha1) - xcenter; q[4] = -r0 * sin(alpha1); q[5] = 0.0;
          q[6] = -r0 * cos(alpha1) - xcenter; q[7] = r0 * sin(alpha1) * cos(alpha2); q[8] = r0 * sin(alpha1) * sin(alpha2);
          q[9] = -r0 * cos(alpha1) - xcenter; q[10] = r0 * sin(alpha1) * cos(alpha2); q[11] = -r0 * sin(alpha1) * sin(alpha2);
          q[12] = r1 - xcenter; q[13] = 0.0; q[14] = 0.0;
          cl.select1 = [4];
          cl.select2 = [2, 3];
        }
        // signum ensures correct ordering of three satellites: sign(cross(x2-x1,x3-x1)^T (x1-x0))
        const del1 = [x[3 * atom1] - x[3 * oxy], x[3 * atom1 + 1] - x[3 * oxy + 1], x[3 * atom1 + 2] - x[3 * oxy + 2]];
        g.minimumImage(del1);
        const del2 = [x[3 * atom2] - x[3 * atom1], x[3 * atom2 + 1] - x[3 * atom1 + 1], x[3 * atom2 + 2] - x[3 * atom1 + 2]];
        g.minimumImage(del2);
        const del3 = [x[3 * atom3] - x[3 * atom1], x[3 * atom3 + 1] - x[3 * atom1 + 1], x[3 * atom3 + 2] - x[3 * atom1 + 2]];
        g.minimumImage(del3);
        const a = del2[1] * del3[2] - del2[2] * del3[1];
        const bb = del2[2] * del3[0] - del2[0] * del3[2];
        const c = del2[0] * del3[1] - del2[1] * del3[0];
        const signum = sgn(a * del1[0] + bb * del1[1] + c * del1[2]);
        if (Math.abs(signum) !== 1) throw new StyleError(`Wrong orientation in cluster of size ${N}in fix filter/corotate!`);
        q[8] *= signum;
        q[11] *= signum;
      } else {
        throw new StyleError('Fix filter/corotate cluster with size > 5 not yet configured...');
      }
    }
  }

  /**
   * The pairs of the engine's half list from the side native LAMMPS stores them (npair_bin.cpp half/bin/newton: the upper
   * half stencil, and in the atom's own bin only the ghosts above it; ../neighbor/nativeOrder.ts nativePairs), as a half
   * list for the outermost level. The filtered level sees filtered owned atoms and real ghosts (native swaps atom->x for
   * the owned atoms only), so a pair across a periodic boundary is evaluated with the filtered position of its owned atom.
   * Orthogonal boxes; null (the engine's list) otherwise.
   */
  respaHalfList(): NeighList | null {
    const sys = this.sys;
    const s = sys.state, nb = sys.nb, g = sys.geom;
    if (!nb.half || g.triclinic) return null;
    const k = this.halfKey;
    if (this.half && k.nbuild === nb.nbuild && k.order === (s.order ?? null) && k.src === nb.half) return this.half;
    const pairs = nativePairs(s, nb, g, nb.half, true);
    const n = nb.nlocal;
    const numneigh = new Int32Array(n);
    for (let p = 0; p < pairs.length; p += 2) numneigh[pairs[p]]++;
    const firstneigh = new Int32Array(n);
    for (let i = 1; i < n; i++) firstneigh[i] = firstneigh[i - 1] + numneigh[i - 1];
    const fill = firstneigh.slice();
    const neighbors = new Int32Array(pairs.length / 2);
    for (let p = 0; p < pairs.length; p += 2) neighbors[fill[pairs[p]]++] = pairs[p + 1];
    this.half = { inum: n, numneigh, firstneigh, neighbors };
    this.halfKey = { nbuild: nb.nbuild, order: s.order ?? null, src: nb.half };
    return this.half;
  }

  /** FixFilterCorotate::filter_inner: the positions of the owned atoms for the outermost level (3 per atom, engine order). */
  respaFilterPositions(): Float64Array {
    const s = this.sys.state;
    const n3 = 3 * s.n;
    if (this.filtered.length !== n3) this.filtered = new Float64Array(n3);
    // array_atom = atom->x, for the atoms that are in no cluster
    this.filtered.set(s.x.subarray(0, n3));
    for (const cl of this.list) this.generalCluster(cl);
    return this.filtered;
  }

  /** FixFilterCorotate::filter_outer: the forces of the outermost level (3 per owned atom, engine order), filtered in place. */
  respaFilterForces(f: Float64Array): void {
    const map = this.map;
    const g = new Float64Array(15);
    for (const cl of this.list) {
      let dim = cl.flag;
      if (dim === 1) dim = 3; // case: angle
      const D = cl.derv;
      for (let j = 0; j < dim; j++) {
        let sum1 = 0, sum2 = 0, sum3 = 0;
        for (let k = 0; k < dim; k++) {
          const kk = map[cl.atoms[k]];
          // apply derivative times force
          sum1 += D[15 * (3 * j) + 3 * k] * f[3 * kk] + D[15 * (3 * j) + 3 * k + 1] * f[3 * kk + 1] + D[15 * (3 * j) + 3 * k + 2] * f[3 * kk + 2];
          sum2 += D[15 * (3 * j + 1) + 3 * k] * f[3 * kk] + D[15 * (3 * j + 1) + 3 * k + 1] * f[3 * kk + 1] + D[15 * (3 * j + 1) + 3 * k + 2] * f[3 * kk + 2];
          sum3 += D[15 * (3 * j + 2) + 3 * k] * f[3 * kk] + D[15 * (3 * j + 2) + 3 * k + 1] * f[3 * kk + 1] + D[15 * (3 * j + 2) + 3 * k + 2] * f[3 * kk + 2];
        }
        g[3 * j] = sum1;
        g[3 * j + 1] = sum2;
        g[3 * j + 2] = sum3;
      }
      // replace force f with filtered value
      for (let j = 0; j < dim; j++) {
        const kk = map[cl.atoms[j]];
        f[3 * kk] = g[3 * j];
        f[3 * kk + 1] = g[3 * j + 1];
        f[3 * kk + 2] = g[3 * j + 2];
      }
    }
  }

  /** FixFilterCorotate::general_cluster: the filtered positions of one cluster and their derivative. */
  private generalCluster(cl: Cluster): void {
    const sys = this.sys;
    const s = sys.state;
    const geom = sys.geom;
    const map = this.map;
    const out = this.filtered;
    const x = s.x;
    const q0 = cl.q0;
    const select1 = cl.select1, select2 = cl.select2;
    const nselect1 = select1.length, nselect2 = select2.length;
    let N = cl.flag;
    if (N === 1) N = 3;

    const listCluster: number[] = [];
    const m: number[] = [];
    for (let i = 0; i < N; i++) {
      listCluster.push(map[cl.atoms[i]]);
      m.push(s.massByType[s.type[listCluster[i]]]);
    }
    const r = new Float64Array(N);
    const del: number[][] = [];
    for (let i = 0; i < N; i++) del.push([0, 0, 0]);
    // CALC r_i
    for (let i = 1; i < N; i++) {
      const a = listCluster[i], o = listCluster[0];
      del[i][0] = x[3 * a] - x[3 * o];
      del[i][1] = x[3 * a + 1] - x[3 * o + 1];
      del[i][2] = x[3 * a + 2] - x[3 * o + 2];
      geom.minimumImage(del[i]);
      r[i] = 1.0 / Math.sqrt(del[i][0] * del[i][0] + del[i][1] * del[i][1] + del[i][2] * del[i][2]);
    }
    del[0][0] = del[0][1] = del[0][2] = r[0] = 0.0;

    // calc n1
    const n1 = [0.0, 0.0, 0.0];
    for (let i = 0; i < nselect1; i++) {
      const k = select1[i];
      n1[0] += del[k][0] * r[k];
      n1[1] += del[k][1] * r[k];
      n1[2] += del[k][2] * r[k];
    }
    const norm1 = 1.0 / Math.sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]);
    n1[0] *= norm1; n1[1] *= norm1; n1[2] *= norm1;

    // calc n2
    const n2 = [0.0, 0.0, 0.0];
    for (let i = 0; i < nselect2; i++) {
      const k = select2[i];
      n2[0] += del[k][0] * r[k];
      n2[1] += del[k][1] * r[k];
      n2[2] += del[k][2] * r[k];
    }
    if (!nselect2) n2[0] = n2[1] = n2[2] = 1.0; // cluster of size 2, has only one direction
    const alpha = n2[0] * n1[0] + n2[1] * n1[1] + n2[2] * n1[2];
    n2[0] -= alpha * n1[0];
    n2[1] -= alpha * n1[1];
    n2[2] -= alpha * n1[2];
    const norm2 = 1.0 / Math.sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]);
    n2[0] *= norm2; n2[1] *= norm2; n2[2] *= norm2;

    const n3 = [n1[1] * n2[2] - n1[2] * n2[1], n1[2] * n2[0] - n1[0] * n2[2], n1[0] * n2[1] - n1[1] * n2[0]];
    const norm3 = 1.0 / Math.sqrt(n3[0] * n3[0] + n3[1] * n3[1] + n3[2] * n3[2]);
    n3[0] *= norm3; n3[1] *= norm3; n3[2] *= norm3;

    // x_filter
    for (let i = 0; i < N; i++) {
      const k = listCluster[i];
      out[3 * k] = x[3 * k] + q0[3 * i] * n1[0] + q0[3 * i + 1] * n2[0] + q0[3 * i + 2] * n3[0];
      out[3 * k + 1] = x[3 * k + 1] + q0[3 * i] * n1[1] + q0[3 * i + 1] * n2[1] + q0[3 * i + 2] * n3[1];
      out[3 * k + 2] = x[3 * k + 2] + q0[3 * i] * n1[2] + q0[3 * i + 1] * n2[2] + q0[3 * i + 2] * n3[2];
    }
    let mAll = 0.0;
    for (let i = 0; i < N; i++) mAll += m[i];
    // x + X_center
    for (let i = 0; i < N; i++) {
      const k = listCluster[i];
      for (let j = 0; j < N; j++) {
        out[3 * k] += m[j] / mAll * (del[j][0] - del[i][0]);
        out[3 * k + 1] += m[j] / mAll * (del[j][1] - del[i][1]);
        out[3 * k + 2] += m[j] / mAll * (del[j][2] - del[i][2]);
      }
    }

    // derivative: dn1dx
    const W = 3 * N;
    const sum1: number[][] = [new Array(W).fill(0), new Array(W).fill(0), new Array(W).fill(0)];
    const I3mn1n1T = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) I3mn1n1T[i][j] = -n1[i] * n1[j];
      I3mn1n1T[i][i] += 1.0;
    }
    for (let l = 0; l < nselect1; l++) {
      const k = select1[l];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const help = del[k][i] * del[k][j] * r[k] * r[k] * r[k];
          sum1[i][j + 3 * k] -= help;
          sum1[i][j] += help;
        }
        sum1[i][i + 3 * k] += r[k];
        sum1[i][i] -= r[k];
      }
    }
    const dn1dx: number[][] = [new Array(W).fill(0), new Array(W).fill(0), new Array(W).fill(0)];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < W; j++) {
        let sum = 0;
        for (let l = 0; l < 3; l++) sum += I3mn1n1T[i][l] * sum1[l][j];
        dn1dx[i][j] = norm1 * sum;
      }
    }

    // dn2dx: norm2 * I3mn2n2T * (I3mn1n1T*sum2 - rkn1pn1rk*dn1dx)
    const sum2: number[][] = [new Array(W).fill(0), new Array(W).fill(0), new Array(W).fill(0)];
    const I3mn2n2T = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) I3mn2n2T[i][j] = -n2[i] * n2[j];
      I3mn2n2T[i][i] += 1.0;
    }
    for (let l = 0; l < nselect2; l++) {
      const k = select2[l];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const help = del[k][i] * del[k][j] * r[k] * r[k] * r[k];
          sum2[i][j + 3 * k] -= help;
          sum2[i][j] += help;
        }
        sum2[i][i + 3 * k] += r[k];
        sum2[i][i] -= r[k];
      }
    }
    const rk = [0.0, 0.0, 0.0];
    for (let i = 0; i < nselect2; i++) {
      const k = select2[i];
      rk[0] += del[k][0] * r[k];
      rk[1] += del[k][1] * r[k];
      rk[2] += del[k][2] * r[k];
    }
    // rkn1pn1rk = rkT*n1*I3 + n1*rkT
    const scalar = rk[0] * n1[0] + rk[1] * n1[1] + rk[2] * n1[2];
    const rkn1pn1rk = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) rkn1pn1rk[i][j] = n1[i] * rk[j];
      rkn1pn1rk[i][i] += scalar;
    }
    const sum3: number[][] = [new Array(W).fill(0), new Array(W).fill(0), new Array(W).fill(0)];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < W; j++) {
        let sum = 0;
        for (let l = 0; l < 3; l++) sum += I3mn1n1T[i][l] * sum2[l][j] - rkn1pn1rk[i][l] * dn1dx[l][j];
        sum3[i][j] = sum;
      }
    }
    const dn2dx: number[][] = [new Array(W).fill(0), new Array(W).fill(0), new Array(W).fill(0)];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < W; j++) {
        let sum = 0;
        for (let l = 0; l < 3; l++) sum += I3mn2n2T[i][l] * sum3[l][j];
        dn2dx[i][j] = norm2 * sum;
      }
    }

    // dn3dx = norm3 * I3mn3n3T * cross
    const I3mn3n3T = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) I3mn3n3T[i][j] = -n3[i] * n3[j];
      I3mn3n3T[i][i] += 1.0;
    }
    const cross: number[][] = [new Array(W).fill(0), new Array(W).fill(0), new Array(W).fill(0)];
    for (let j = 0; j < W; j++) {
      cross[0][j] = dn1dx[1][j] * n2[2] - dn1dx[2][j] * n2[1] + n1[1] * dn2dx[2][j] - n1[2] * dn2dx[1][j];
      cross[1][j] = dn1dx[2][j] * n2[0] - dn1dx[0][j] * n2[2] + n1[2] * dn2dx[0][j] - n1[0] * dn2dx[2][j];
      cross[2][j] = dn1dx[0][j] * n2[1] - dn1dx[1][j] * n2[0] + n1[0] * dn2dx[1][j] - n1[1] * dn2dx[0][j];
    }
    const dn3dx: number[][] = [new Array(W).fill(0), new Array(W).fill(0), new Array(W).fill(0)];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < W; j++) {
        let sum = 0;
        for (let l = 0; l < 3; l++) sum += I3mn3n3T[i][l] * cross[l][j];
        dn3dx[i][j] = norm3 * sum;
      }
    }

    // help2[i+3l][j] = q0 . dn/dx, plus the centre-of-mass part
    const help2 = new Float64Array(225);
    for (let l = 0; l < N; l++) {
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < W; j++) {
          help2[15 * (i + 3 * l) + j] = q0[3 * l] * dn1dx[i][j] + q0[3 * l + 1] * dn2dx[i][j] + q0[3 * l + 2] * dn3dx[i][j];
        }
      }
    }
    for (let j = 0; j < N; j++) {
      for (let l = 0; l < N; l++) {
        for (let i = 0; i < 3; i++) help2[15 * (i + 3 * l) + (i + 3 * j)] += m[j] / mAll;
      }
    }
    // need transposed of help2
    const D = cl.derv;
    for (let ii = 0; ii < W; ii++) {
      for (let jj = 0; jj < W; jj++) D[15 * ii + jj] = help2[15 * jj + ii];
    }
  }
}

export const makeFixFilterCorotate: FixFactory = (sys, id, group, args) => new FixFilterCorotate(sys, id, group, args);
