/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/REACTION/fix_bond_react.cpp and
 * src/REACTION/fix_bond_react.h (fix bond/react, Jacob Gissinger, Stevens Institute of Technology; the
 * superimpose algorithm, the map file sections, the topology update and the stabilization groups), and
 * src/reset_atoms_mol.cpp (the reset_mol_ids option), src/REACTION/superpose3d.h (Superpose3D, used for the
 * created atoms' fit) and src/math_eigen_impl.h (the Jacobi eigen-solver that Superpose3D calls). LAMMPS is Copyright Sandia Corporation and distributed
 * under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Scope of this port (everything else is a StyleError naming the keyword): react blocks with map files that
 * have InitiatorIDs, EdgeIDs, Equivalences, DeleteIDs, CreateIDs and Wildcards sections; keywords stabilization,
 * stabilize_steps, prob, rescale_charges, molecule (off/inter/intra), modify_create (fit with a fragment or all,
 * overlap), max_rxn and rate_limit with one reaction, reset_mol_ids yes/no, shuffle_seed (used only with
 * multi-reaction limits). Wildcard atoms take their bonded types from the type labels (inferBondType and the
 * other labelmap inferences in src/engine/labelmap.ts). The site fit of created atoms is superpose() (a port of
 * Superpose3D with the Jacobi eigen-solver). Refused by name: constraints, ChiralIDs, custom_charges, v_ variables,
 * file, vizsteps, reset_mol_ids molmap, and created atoms with a per-atom mass style.
 *
 * Engine representation: topology lists are global (by atom ID) instead of per-atom storage, so the
 * bond, angle, dihedral and improper updates are written as set operations (remove the entries that touch a
 * landlocked or deleted atom, add the post-reaction entries that touch a landlocked atom). Special lists are
 * rebuilt by the engine from the bonds after the change. Both give the same entries as native; only the
 * storage order and hence floating-point summation order can differ.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { MoleculeTemplate } from '../../engine/molecule';
import { specialFromBonds } from '../../engine/molecule';
import { appendAtoms, buildSpecial, massOf, nativeOrder } from '../../engine/atoms';
import { bitOfIndex } from '../../engine/group';
import { FixNVELimit } from '../../engine/fix/motion';
import type { CustomProp, TopoList } from '../../engine/types';
import type { FixFactory } from '../../engine/styles';
import { RanMars } from '../random_mars';
import { inferAngleType, inferBondType, inferDihedralType, inferImproperType } from '../../engine/labelmap';

const BIG = 1.0e20;
const MAXGUESS = 20;

type Status = 'ACCEPT' | 'REJECT' | 'PROCEED' | 'CONTINUE' | 'GUESSFAIL' | 'RESTORE';

/** Per-atom record of one reaction (fix_bond_react.h ReactionAtomFlags). Indexed by reactant or product atom as native does. */
interface RxnAtom {
  edge: number;
  landlocked: number;
  wildcard: boolean;
  recharged: number;
  deleted: number;
  created: number;
  /** amap[0] = product ID, amap[1] = reactant ID (equivalences; 1-based). */
  amap: [number, number];
  /** ramap[0] = reactant ID, ramap[1] = product ID. */
  ramap: [number, number];
}

/** Template data with native-style special lists: tsp[i] flat 1-2, 1-3, 1-4 partners; tn[i] cumulative counts. */
interface Tmpl {
  t: MoleculeTemplate;
  n: number;
  tsp: number[][];
  tn: number[][];
}

interface Rxn {
  ID: number;
  name: string;
  reactant: Tmpl;
  product: Tmpl;
  groupbit: number;
  nevery: number;
  rminsq: number;
  rmaxsq: number;
  ibonding: number;
  jbonding: number;
  iatomtype: number;
  jatomtype: number;
  closeneigh: number;
  fraction: number;
  seed: number;
  random: RanMars;
  limitDuration: number;
  rescaleFlag: boolean;
  /** number of product atoms whose charge is rescaled (rescale_charges yes), set once from the map file. */
  rescaleCount: number;
  molTotalCharge: number;
  moleculeKeyword: 'off' | 'inter' | 'intra';
  atoms: RxnAtom[];
  attempts: [number, number][];
  localCount: number;
  nlocalkeep: number;
  reactionCountTotal: number;
  /** CreateIDs in the map file: the reaction creates atoms (native create_atoms_flag). */
  createFlag: boolean;
  /** modify_create overlap R, squared (native overlapsq; 0 = no overlap check). */
  overlapsq: number;
  /** modify_create fit fragment: index in the post-reaction template, or -1 for all atoms (native modify_create_fragid). */
  fitFragment: number;
}

/** A created atom waiting for the insertion (native addatoms entry). */
interface PendingAtom { x: number[]; type: number; tag: number; image: number[]; v: number[]; mask: number }

/** One reaction instance (native mega_glove column): reaction ID, total charge of the reactant atoms, and the tags. */
interface Mega {
  rxn: number;
  charge: number;
  glove: number[];
}

interface Special {
  flat: number[];
  /** cumulative counts: 1-2, 1-2 + 1-3, all */
  c: [number, number, number];
}

/** Map-file parsing and template preparation. */
const templateOf = (t: MoleculeTemplate): Tmpl => {
  const sp = specialFromBonds(t);
  const tsp: number[][] = [];
  const tn: number[][] = [];
  for (let i = 0; i < t.natoms; i++) {
    const [s12, s13, s14] = sp[i];
    tsp.push([...s12, ...s13, ...s14]);
    tn.push([s12.length, s12.length + s13.length, s12.length + s13.length + s14.length]);
  }
  return { t, n: t.natoms, tsp, tn };
};

/** The lines of a map file, read the way native does (comments stripped, keyword then one skipped line). */
class MapReader {
  private k = 1;
  constructor(private lines: string[], private name: string) {}
  next(): string | null {
    if (this.k >= this.lines.length) return null;
    return this.lines[this.k++];
  }
  /** The next line as an integer, failing with the native message. */
  int(msg: string): number {
    const l = this.next();
    if (l === null) throw new StyleError(`Fix bond/react: Unexpected end of map file ${this.name}`);
    const v = Number.parseInt(l.trim().split(/\s+/)[0] ?? '', 10);
    if (!Number.isFinite(v)) throw new StyleError(`Fix bond/react: ${msg} (${this.name})`);
    return v;
  }
  /** Current position for the header loop. */
  peek(): string | null { return this.k < this.lines.length ? this.lines[this.k] : null; }
  advance(): void { this.k++; }
}

const strip = (l: string): string => {
  const h = l.indexOf('#');
  return (h < 0 ? l : l.slice(0, h)).trim();
};

/** Parses one map file into the reaction (fix_bond_react.cpp read_map_file and its section readers). */
const readMap = (rxn: Rxn, text: string, name: string): void => {
  const lines = text.split('\n').map((l) => l.replace(/\r$/, ''));
  const rd = new MapReader(lines, name);
  let nedge = 0, nequiv = -1, ndelete = 0, ncreate = 0, nchiral = 0, nwild = 0, nconstr = 0;
  // header: skip blank and comment lines; the first unrecognised line starts the body
  for (;;) {
    const raw = rd.peek();
    if (raw === null) break;
    const l = strip(raw);
    if (!l) { rd.advance(); continue; }
    const num = () => Number.parseInt(l.split(/\s+/)[0] ?? '', 10);
    if (l.includes('edgeIDs')) nedge = num();
    else if (l.includes('equivalences')) nequiv = num();
    else if (l.includes('deleteIDs')) ndelete = num();
    else if (l.includes('createIDs')) ncreate = num();
    else if (l.includes('chiralIDs')) nchiral = num();
    else if (l.includes('wildcards')) nwild = num();
    else if (l.includes('constraints')) nconstr = num();
    else break;
    rd.advance();
  }
  if (nchiral > 0) throw new StyleError(`fix bond/react: ChiralIDs in map file ${name} is not supported by the browser engine`);
  if (nwild < 0 || nwild > rxn.reactant.n) throw new StyleError('Fix bond/react: Invalid number of wildcards in map file');
  if (nconstr > 0) throw new StyleError(`fix bond/react: Constraints in map file ${name} are not supported by the browser engine`);
  if (ncreate < 0 || ncreate > rxn.product.n) throw new StyleError('Fix bond/react: Invalid number of createIDs in map file');
  if (ncreate === 0 && rxn.reactant.n !== rxn.product.n) throw new StyleError('Fix bond/react: Reaction templates must contain the same number of atoms');
  if (ncreate > 0 && rxn.reactant.n + ncreate !== rxn.product.n) throw new StyleError('Fix bond/react: Incorrect number of created atoms');
  if (nequiv !== rxn.reactant.n) throw new StyleError('Fix bond/react: Number of equivalences in map file must equal number of atoms in reaction templates');
  if (nedge < 0 || nedge > rxn.reactant.n) throw new StyleError('Fix bond/react: Invalid number of edgeIDs in map file');
  if (ndelete < 0 || ndelete > rxn.reactant.n) throw new StyleError('Fix bond/react: Invalid number of deleteIDs in map file');
  // sections: the keyword line (already at the current position), then one line skipped, then the data
  let bondflag = false, equivflag = false;
  const keyword = (): string | null => {
    for (;;) {
      const l = rd.next();
      if (l === null) return null;
      if (strip(l)) { rd.next(); return strip(l); }
    }
  };
  let kw: string | null = null;
  {
    // the header loop stopped on the keyword line (or the file ended)
    const raw = rd.peek();
    if (raw !== null) { kw = strip(raw); rd.advance(); rd.next(); }
  }
  while (kw) {
    if (kw === 'InitiatorIDs' || kw === 'BondingIDs') {
      bondflag = true;
      rxn.ibonding = rd.int('InitiatorIDs section is incorrectly formatted');
      rxn.jbonding = rd.int('InitiatorIDs section is incorrectly formatted');
      if (rxn.ibonding < 1 || rxn.ibonding > rxn.reactant.n || rxn.jbonding < 1 || rxn.jbonding > rxn.reactant.n) {
        throw new StyleError('Fix bond/react: Invalid template atom ID in map file');
      }
    } else if (kw === 'EdgeIDs') {
      for (let i = 0; i < nedge; i++) {
        const v = rd.int('EdgeIDs section is incorrectly formatted');
        if (v < 1 || v > rxn.reactant.n) throw new StyleError('Fix bond/react: Invalid template atom ID in map file');
        rxn.atoms[v - 1].edge = 1;
      }
    } else if (kw === 'Equivalences') {
      equivflag = true;
      for (let i = 0; i < nequiv; i++) {
        const l = rd.next();
        const w = (l ?? '').trim().split(/\s+/).map((x) => Number.parseInt(x, 10));
        const t1 = w[0], t2 = w[1];
        if (!Number.isFinite(t1) || !Number.isFinite(t2)) throw new StyleError('Equivalences section is incorrectly formatted');
        // columns: reactant ID, product ID (native: tmp1 <= reactant natoms, tmp2 <= product natoms)
        if (t1 < 1 || t1 > rxn.reactant.n || t2 < 1 || t2 > rxn.product.n) throw new StyleError('Fix bond/react: Invalid template atom ID in map file');
        rxn.atoms[t2 - 1].amap = [t2, t1];
        rxn.atoms[t1 - 1].ramap = [t1, t2];
      }
      for (let i = 0; i < rxn.product.n; i++) {
        if (rxn.atoms[i].created === 1) continue;
        for (let j = i + 1; j < rxn.product.n; j++) {
          if (rxn.atoms[j].created === 1) continue;
          if (rxn.atoms[i].amap[0] === rxn.atoms[j].amap[0] || rxn.atoms[i].amap[1] === rxn.atoms[j].amap[1]) {
            throw new StyleError('Fix bond/react: Repeated atoms IDs in Equivalences section');
          }
        }
      }
    } else if (kw === 'CreateIDs') {
      // CreateAtoms: the post-reaction IDs of the atoms to create (a created atom is not in the pre-reaction template)
      rxn.createFlag = true;
      for (let i = 0; i < ncreate; i++) {
        const v = rd.int('CreateIDs section is incorrectly formatted');
        if (v < 1 || v > rxn.product.n) throw new StyleError('Fix bond/react: Invalid atom ID in CreateIDs section of map file');
        rxn.atoms[v - 1].created = 1;
      }
    } else if (kw === 'Wildcards') {
      // ReadWildcards: a wildcard reactant atom matches any type, and bonded entries that name it take their type from labels
      for (let i = 0; i < nwild; i++) {
        const v = rd.int('Wildcards section is incorrectly formatted');
        if (v < 1 || v > rxn.reactant.n) throw new StyleError('Bond/react: Invalid template atom ID in map file');
        rxn.atoms[v - 1].wildcard = true;
      }
    } else if (kw === 'DeleteIDs') {
      for (let i = 0; i < ndelete; i++) {
        const v = rd.int('DeleteIDs section is incorrectly formatted');
        if (v < 1 || v > rxn.reactant.n) throw new StyleError('Fix bond/react: Invalid template atom ID in map file');
        rxn.atoms[v - 1].deleted = 1;
      }
    } else {
      throw new StyleError(`Fix bond/react: Unknown section '${kw}' in map file ${name}`);
    }
    kw = keyword();
  }
  if (!bondflag || !equivflag) throw new StyleError('Fix bond/react: Map file missing InitiatorIDs or Equivalences section');
  for (let i = 0; i < rxn.reactant.n; i++) {
    const my = rxn.atoms[i].ramap[1];
    if (my && rxn.atoms[my - 1].created === 1) throw new StyleError('Fix bond/react: Created atoms cannot also be listed in Equivalences section');
  }
};

/** Landlocked flags (find_landlocked_atoms): atoms whose topology is contained in the post-reaction template. */
const findLandlocked = (rxn: Rxn): void => {
  const P = rxn.product, R = rxn.reactant;
  for (let i = 0; i < P.n; i++) {
    rxn.atoms[i].landlocked = rxn.atoms[i].created === 0 && rxn.atoms[rxn.atoms[i].amap[1] - 1].edge === 1 ? 0 : 1;
  }
  // angles: landlocked means no edge atom among the 1-2 neighbours; dihedrals and impropers: among 1-3
  let nspecialLimit = -1;
  if (P.t.angles.length > 0) nspecialLimit = 0;
  if (P.t.dihedrals.length > 0 || P.t.impropers.length > 0) nspecialLimit = 1;
  if (nspecialLimit !== -1) {
    for (let i = 0; i < P.n; i++) {
      const cnt = nspecialLimit === 0 ? P.tn[i][0] : P.tn[i][1];
      for (let j = 0; j < cnt; j++) {
        const pa = P.tsp[i][j];
        for (let k = 0; k < R.n; k++) {
          if (rxn.atoms[pa - 1].amap[1] === k + 1 && rxn.atoms[k].edge === 1) rxn.atoms[i].landlocked = 0;
        }
      }
    }
  }
  // a product atom whose type changes must be landlocked
  for (let i = 0; i < P.n; i++) {
    if (rxn.atoms[i].created === 0) {
      const ra = rxn.atoms[i].amap[1];
      const sameType = P.t.type[i] === R.t.type[ra - 1] || rxn.atoms[ra - 1].wildcard;
      if (!sameType && rxn.atoms[i].landlocked === 0) throw new StyleError(`Fix bond/react: Atom type affected by reaction ${rxn.name} is too close to template edge`);
    }
  }
  // a deleted atom may not stay bonded to an atom that is not deleted
  for (let i = 0; i < R.n; i++) {
    if (rxn.atoms[i].deleted === 1) {
      const ii = rxn.atoms[i].ramap[1] - 1;
      for (let j = 0; j < P.tn[ii][0]; j++) {
        if (rxn.atoms[rxn.atoms[P.tsp[ii][j] - 1].amap[1] - 1].deleted === 0) {
          throw new StyleError('Fix bond/react: A deleted atom cannot remain bonded to an atom that is not deleted');
        }
      }
    }
  }
  for (let i = 0; i < P.n; i++) {
    if (rxn.atoms[i].created === 1 && rxn.atoms[i].landlocked === 0) throw new StyleError('Fix bond/react: Created atom too close to template edge');
  }
};

/** Minimum-image displacement along one dimension (domain->minimum_image); periodic dimensions only. */
const minImage = (dx: number, L: number, periodic: boolean): number => (periodic ? dx - L * Math.round(dx / L) : dx);

/**
 * Jacobi::Diagonalize for a symmetric 4x4 matrix (src/math_eigen_impl.h at b332f28, the cyclic Jacobi with the
 * max-entry-per-row bookkeeping, sorted by decreasing eigenvalue). The eigenvectors are the rows of the result.
 * Degenerate eigenvalues (two atoms in the fit) make the chosen eigenvector depend on these steps, so they are kept.
 */
const jacobiDiagonalize4 = (mat: number[][]): { eval: number[]; evec: number[][] } => {
  const n = 4;
  const M: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const evec: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j): number => (i === j ? 1 : 0)));
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) M[i][j] = mat[i][j];
  const maxIdx = new Array<number>(n).fill(0);
  const maxEntryRow = (i: number): number => {
    let jmax = i + 1;
    for (let j = i + 2; j < n; j++) if (Math.abs(M[i][j]) > Math.abs(M[i][jmax])) jmax = j;
    return jmax;
  };
  for (let i = 0; i < n - 1; i++) maxIdx[i] = maxEntryRow(i);
  const maxIters = 50 * n * (n - 1) / 2;
  for (let iter = 0; iter < maxIters; iter++) {
    // MaxEntry: the largest entry among the row maxima
    let i = 0, j = maxIdx[0];
    let maxEntry = Math.abs(M[i][j]);
    for (let k = 1; k < n - 1; k++) {
      const jk = maxIdx[k];
      if (Math.abs(M[k][jk]) > maxEntry) { maxEntry = Math.abs(M[k][jk]); i = k; j = jk; }
    }
    if (M[i][i] + M[i][j] === M[i][i] && M[j][j] + M[i][j] === M[j][j]) {
      M[i][j] = 0.0;
      maxIdx[i] = maxEntryRow(i);
    }
    if (M[i][j] === 0.0) break;
    // CalcRot
    let t = 1.0;
    const mjjii = M[j][j] - M[i][i];
    if (mjjii !== 0.0) {
      let kappa = mjjii;
      t = 0.0;
      const mij = M[i][j];
      if (mij !== 0.0) {
        kappa /= 2.0 * mij;
        t = 1.0 / (Math.sqrt(1 + kappa * kappa) + Math.abs(kappa));
        if (kappa < 0.0) t = -t;
      }
    }
    const c = 1.0 / Math.sqrt(1 + t * t);
    const s = c * t;
    // ApplyRot
    M[i][i] -= t * M[i][j];
    M[j][j] += t * M[i][j];
    M[i][j] = 0.0;
    for (let w = 0; w < i; w++) {
      M[i][w] = M[w][i];
      M[w][i] = c * M[w][i] - s * M[w][j];
      if (i === maxIdx[w]) maxIdx[w] = maxEntryRow2(M, w, n);
      else if (Math.abs(M[w][i]) > Math.abs(M[w][maxIdx[w]])) maxIdx[w] = i;
    }
    for (let w = i + 1; w < j; w++) {
      M[w][i] = M[i][w];
      M[i][w] = c * M[i][w] - s * M[w][j];
    }
    for (let w = j + 1; w < n; w++) {
      M[w][i] = M[i][w];
      M[i][w] = c * M[i][w] - s * M[j][w];
    }
    maxIdx[i] = maxEntryRow(i);
    for (let w = 0; w < i; w++) {
      M[w][j] = s * M[i][w] + c * M[w][j];
      if (j === maxIdx[w]) maxIdx[w] = maxEntryRow2(M, w, n);
      else if (Math.abs(M[w][j]) > Math.abs(M[w][maxIdx[w]])) maxIdx[w] = j;
    }
    for (let w = i + 1; w < j; w++) {
      M[w][j] = s * M[w][i] + c * M[w][j];
      if (j === maxIdx[w]) maxIdx[w] = maxEntryRow2(M, w, n);
      else if (Math.abs(M[w][j]) > Math.abs(M[w][maxIdx[w]])) maxIdx[w] = j;
    }
    for (let w = j + 1; w < n; w++) M[j][w] = s * M[w][i] + c * M[j][w];
    maxIdx[j] = maxEntryRow(j);
    // ApplyRotLeft
    for (let v = 0; v < n; v++) {
      const eiv = evec[i][v];
      evec[i][v] = c * evec[i][v] - s * evec[j][v];
      evec[j][v] = s * eiv + c * evec[j][v];
    }
  }
  const eval_ = Array.from({ length: n }, (_, k) => M[k][k]);
  // SortRows (decreasing eigenvalues, selection sort swapping the rows of evec)
  for (let a = 0; a < n - 1; a++) {
    let imax = a;
    for (let b = a + 1; b < n; b++) if (eval_[b] > eval_[imax]) imax = b;
    [eval_[a], eval_[imax]] = [eval_[imax], eval_[a]];
    [evec[a], evec[imax]] = [evec[imax], evec[a]];
  }
  return { eval: eval_, evec };
};

/** MaxEntryRow on a 4x4 working matrix (row w: the column of the largest entry right of the diagonal). */
const maxEntryRow2 = (M: number[][], w: number, n: number): number => {
  let jmax = w + 1;
  for (let j = w + 2; j < n; j++) if (Math.abs(M[w][j]) > Math.abs(M[w][jmax])) jmax = j;
  return jmax;
};

/**
 * Least-squares rigid superposition with unit weights: the rotation R and translation T that map the mobile points
 * onto the frozen ones (frozen_i ~ R mobile_i + T). A port of Superpose3D::Superpose (src/REACTION/superpose3d.h at
 * b332f28; Jewett's quaternion form of Theobald's method, with the eigen-solver of jacobiDiagonalize4) without
 * rescaling. fix bond/react uses it to fit the post-reaction template onto the reaction site.
 */
export const superpose = (frozen: number[][], mobile: number[][]): { R: number[][]; T: number[] } => {
  const n = frozen.length;
  const cf = [0, 0, 0], cm = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) { cf[d] += frozen[i][d]; cm[d] += mobile[i][d]; }
  for (let d = 0; d < 3; d++) { cf[d] /= n; cm[d] /= n; }
  const sf = frozen.map((r) => [0, 1, 2].map((d) => r[d] - cf[d]));
  const sm = mobile.map((r) => [0, 1, 2].map((d) => r[d] - cm[d]));
  // M (equation 16): M[i][j] = sum_n sm[n][i] * sf[n][j]
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let k = 0; k < n; k++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[i][j] += sm[k][i] * sf[k][j];
  const traceM = M[0][0] + M[1][1] + M[2][2];
  const Q = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { Q[i][j] = M[i][j] + M[j][i]; if (i === j) Q[i][j] -= 2.0 * traceM; }
  const V = [M[1][2] - M[2][1], M[2][0] - M[0][2], M[0][1] - M[1][0]];
  // P (equation 22)
  const P: number[][] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) P[i][j] = Q[i][j];
  P[0][3] = V[0]; P[3][0] = V[0];
  P[1][3] = V[1]; P[3][1] = V[1];
  P[2][3] = V[2]; P[3][2] = V[2];
  P[3][3] = 0.0;
  // principal eigenvector (the eigenvalues come sorted in decreasing order); a single point is not rotated
  const p = n < 2 ? [0, 0, 0, 1] : jacobiDiagonalize4(P).evec[0].slice();
  const pnorm = Math.sqrt(p.reduce((a, x) => a + x * x, 0));
  for (let i = 0; i < 4; i++) p[i] /= pnorm;
  const R = [
    [(p[0] * p[0]) - (p[1] * p[1]) - (p[2] * p[2]) + (p[3] * p[3]), 2 * (p[0] * p[1] - p[2] * p[3]), 2 * (p[0] * p[2] + p[1] * p[3])],
    [2 * (p[0] * p[1] + p[2] * p[3]), -(p[0] * p[0]) + (p[1] * p[1]) - (p[2] * p[2]) + (p[3] * p[3]), 2 * (p[1] * p[2] - p[0] * p[3])],
    [2 * (p[0] * p[2] - p[1] * p[3]), 2 * (p[1] * p[2] + p[0] * p[3]), -(p[0] * p[0]) - (p[1] * p[1]) + (p[2] * p[2]) + (p[3] * p[3])],
  ];
  const T = [0, 1, 2].map((i) => cf[i] - (R[i][0] * cm[0] + R[i][1] * cm[1] + R[i][2] * cm[2]));
  return { R, T };
};

/** Counts of the multiset `b` contained in the multiset `a` (std::includes on sorted ranges). */
const multisetIncludes = (a: number[], b: number[]): boolean => {
  const m = new Map<number, number>();
  for (const x of a) m.set(x, (m.get(x) ?? 0) + 1);
  for (const x of b) {
    const c = m.get(x) ?? 0;
    if (c === 0) return false;
    m.set(x, c - 1);
  }
  return true;
};

interface Restore {
  glove: number[];
  pioneerCount: number[];
  pioneers: number[];
  pion: number;
  neigh: number;
  trace: number;
  gloveCounter: number;
}

/**
 * One superimpose search for one initiator pair (fix_bond_react.cpp superimpose_algorithm and its helpers).
 * `sim` gives the simulation special lists by atom ID; `limited` tells whether an atom is in a reaction
 * (i_limit_tags != 0) and `typeOf` the simulation type of an atom ID.
 */
class Superimposer {
  status: Status = 'PROCEED';
  pion = 0;
  neigh = 0;
  trace = 0;
  gloveCounter = 0;
  glove: number[];
  pioneerCount: number[];
  pioneers: number[];
  avail = 0;
  /** guessBranch[avail] (native index avail-1, shifted by one so avail 0 is defined). */
  guessBranch = new Int32Array(MAXGUESS + 1);
  restore: Restore[] = [];
  megas: Mega[] = [];

  constructor(
    private rxn: Rxn,
    private sim: (id: number) => Special | undefined,
    private typeOf: (id: number) => number,
    private limited: (id: number) => boolean,
    private nlocal: number,
    private charge: (glove: number[]) => number,
  ) {
    const n = rxn.reactant.n;
    this.glove = new Array(n).fill(0);
    this.pioneerCount = new Array(n).fill(0);
    this.pioneers = new Array(n).fill(0);
  }

  private spec(tag: number): Special {
    const s = this.sim(tag);
    if (!s) throw new StyleError('Fix bond/react: Fix bond/react needs ghost atoms from further away');
    return s;
  }

  private n12(tag: number): number { return this.spec(tag).c[0]; }

  private xs(tag: number, i: number): number { return this.spec(tag).flat[i]; }

  private compareAtomtype(checktype: number, reactantAtom: number): boolean {
    const i = reactantAtom - 1;
    return checktype === this.rxn.reactant.t.type[i] || this.rxn.atoms[i].wildcard;
  }

  private ringCheck(): boolean {
    const R = this.rxn.reactant;
    for (let i = 0; i < R.n; i++) {
      if (this.rxn.atoms[i].edge === 0 && R.tn[i][0] !== this.n12(this.glove[i])) return false;
    }
    for (let i = 0; i < R.n; i++) {
      for (let j = 0; j < R.tn[i][0]; j++) {
        const ispecial = R.tsp[i][j];
        let fail = true;
        for (let k = 0; k < this.n12(this.glove[i]); k++) {
          if (this.xs(this.glove[i], k) === this.glove[ispecial - 1]) { fail = false; break; }
        }
        if (fail) return false;
      }
    }
    return true;
  }

  /** check_constraints: no constraints are ported, so an accepted site always satisfies them. */
  private checkConstraints(): boolean { return true; }

  /** Native: a site found (status ACCEPT) survives the probability draw, else it is rejected. */
  private acceptSite(): void {
    if (this.rxn.fraction < 1.0 && this.rxn.random.uniform() >= this.rxn.fraction) {
      this.status = 'REJECT';
      return;
    }
    this.status = 'ACCEPT';
    this.megas.push({ rxn: this.rxn.ID, charge: this.rxn.rescaleFlag ? this.charge(this.glove) : 0, glove: this.glove.slice(0, this.rxn.reactant.n) });
  }

  run(): void {
    const rxn = this.rxn, R = rxn.reactant;
    for (const attempt of rxn.attempts) {
      this.status = 'PROCEED';
      this.pion = this.neigh = this.trace = this.gloveCounter = 0;
      this.glove.fill(0);
      this.guessBranch.fill(0);
      this.glove[rxn.ibonding - 1] = attempt[0];
      this.gloveCounter++;
      this.glove[rxn.jbonding - 1] = attempt[1];
      this.gloveCounter++;
      // two-atom templates: the initiators are the whole site
      if (this.gloveCounter === R.n) {
        const a1 = this.glove[rxn.ibonding - 1], a2 = this.glove[rxn.jbonding - 1];
        const n1 = this.n12(a1), n2 = this.n12(a2);
        if (n1 === R.tn[rxn.ibonding - 1][0] && n2 === n1 && (n1 === 0 || this.xs(a1, 0) === a2) && this.checkConstraints()) {
          this.acceptSite();
        } else this.status = 'REJECT';
      }
      this.avail = 0;
      this.pioneerCount.fill(0);
      for (let i = 0; i < R.tn[rxn.ibonding - 1][0]; i++) this.pioneerCount[R.tsp[rxn.ibonding - 1][i] - 1]++;
      for (let i = 0; i < R.tn[rxn.jbonding - 1][0]; i++) this.pioneerCount[R.tsp[rxn.jbonding - 1][i] - 1]++;
      let hang = 0;
      while ((this.status as Status) !== 'ACCEPT' && (this.status as Status) !== 'REJECT') {
        this.pioneers.fill(0);
        let npion = 0;
        for (let i = 0; i < R.n; i++) {
          if (this.glove[i] !== 0 && this.pioneerCount[i] < R.tn[i][0] && rxn.atoms[i].edge === 0) {
            let assigned = 0;
            const nfirst = R.tn[i][0];
            for (let j = 0; j < nfirst; j++) {
              for (let k = 0; k < R.n; k++) {
                if (this.xs(this.glove[i], j) === this.glove[k]) { assigned++; break; }
              }
            }
            if (assigned === nfirst) continue;
            this.pioneers[i] = 1;
            npion++;
          }
        }
        if (npion === 0) this.status = 'GUESSFAIL';
        for (this.pion = 0; this.pion < R.n; this.pion++) {
          if (this.pioneers[this.pion] || this.status === 'GUESSFAIL') {
            this.makeGuess();
            if ((this.status as Status) === 'ACCEPT' || (this.status as Status) === 'REJECT') break;
          }
        }
        if ((this.status as Status) === 'ACCEPT') this.acceptSite();
        hang++;
        if (hang > this.nlocal * 30) {
          throw new StyleError('Fix bond/react: Excessive iteration of superimpose algorithm. Please check that all pre-reaction template atoms are linked to an initiator atom, via at least one path that does not involve edge atoms.');
        }
      }
    }
  }

  private makeGuess(): void {
    const rxn = this.rxn, R = rxn.reactant;
    if (this.status === 'GUESSFAIL' && this.avail === 0) { this.status = 'REJECT'; return; }
    if (this.status === 'GUESSFAIL' && this.avail > 0) {
      const rp = this.restore[this.avail - 1];
      this.glove = rp.glove.slice();
      this.pioneerCount = rp.pioneerCount.slice();
      this.pioneers = rp.pioneers.slice();
      this.pion = rp.pion;
      this.neigh = rp.neigh;
      this.trace = rp.trace;
      this.gloveCounter = rp.gloveCounter;
      this.status = 'RESTORE';
      this.neighborLoop();
      if ((this.status as Status) !== 'PROCEED') return;
    }
    const pion = this.pion;
    const sitePion = this.glove[pion];
    for (let i = 0; i < this.n12(sitePion); i++) {
      if (this.limited(this.xs(sitePion, i))) { this.status = 'GUESSFAIL'; return; }
    }
    const nfirst = R.tn[this.pion][0];
    if (nfirst !== this.n12(this.glove[this.pion])) { this.status = 'GUESSFAIL'; return; }
    const molTypes: number[] = [], lclTypes: number[] = [];
    for (let i = 0; i < nfirst; i++) {
      const imolatom = R.tsp[this.pion][i] - 1;
      if (!rxn.atoms[imolatom].wildcard) molTypes.push(R.t.type[imolatom]);
      lclTypes.push(this.typeOf(this.xs(this.glove[this.pion], i)));
    }
    if (!multisetIncludes(lclTypes, molTypes)) { this.status = 'GUESSFAIL'; return; }
    this.neighborLoop();
  }

  private neighborLoop(): void {
    const R = this.rxn.reactant;
    if (this.status === 'RESTORE') { this.checkANeighbor(); return; }
    const nfirst = R.tn[this.pion][0];
    for (this.neigh = 0; this.neigh < nfirst; this.neigh++) {
      if (this.glove[R.tsp[this.pion][this.neigh] - 1] === 0) this.checkANeighbor();
    }
  }

  private checkANeighbor(): void {
    const rxn = this.rxn, R = rxn.reactant;
    const nfirst = R.tn[this.pion][0];
    const site = this.glove[this.pion];
    const rn = R.tsp[this.pion][this.neigh];
    if (this.status !== 'RESTORE') {
      // a first neighbour that has no other neighbours and is not an edge atom: hydrogen-like matching
      if (R.tn[rn - 1][0] === 1 && rxn.atoms[rn - 1].edge === 0) {
        for (let i = 0; i < nfirst; i++) {
          const cand = this.xs(site, i);
          const checktype = this.typeOf(cand);
          if (this.compareAtomtype(checktype, rn) && this.n12(cand) === 1) {
            let already = false;
            for (let j = 0; j < R.n; j++) if (this.glove[j] === cand) { already = true; break; }
            if (!already) {
              this.glove[rn - 1] = cand;
              for (let j = 0; j < R.tn[rn - 1][0]; j++) this.pioneerCount[R.tsp[rn - 1][j] - 1]++;
              this.gloveCounter++;
              if (this.gloveCounter === R.n) {
                this.status = this.ringCheck() && this.checkConstraints() ? 'ACCEPT' : 'GUESSFAIL';
                return;
              }
              return;
            }
          }
        }
        this.status = 'GUESSFAIL';
        return;
      }
    }
    this.crosscheckTheNeighbor();
    if (this.status !== 'PROCEED') {
      if (this.status === 'CONTINUE') this.status = 'PROCEED';
      return;
    }
    for (let i = 0; i < nfirst; i++) {
      const cand = this.xs(site, i);
      const checktype = this.typeOf(cand);
      if (this.compareAtomtype(checktype, rn)) {
        let already = false;
        for (let j = 0; j < R.n; j++) if (this.glove[j] === cand) { already = true; break; }
        if (!already) {
          this.glove[rn - 1] = cand;
          for (let ii = 0; ii < R.tn[rn - 1][0]; ii++) this.pioneerCount[R.tsp[rn - 1][ii] - 1]++;
          this.gloveCounter++;
          if (this.gloveCounter === R.n) {
            this.status = this.ringCheck() && this.checkConstraints() ? 'ACCEPT' : 'GUESSFAIL';
            return;
          }
          return;
        }
      }
    }
  }

  private crosscheckTheNeighbor(): void {
    const rxn = this.rxn, R = rxn.reactant;
    if (this.status === 'RESTORE') { this.innerCrosscheckLoop(); return; }
    const nfirst = R.tn[this.pion][0];
    for (this.trace = 0; this.trace < nfirst; this.trace++) {
      const ra1 = R.tsp[this.pion][this.trace];
      const at1 = R.t.type[ra1 - 1];
      const ra2 = R.tsp[this.pion][this.neigh];
      const at2 = R.t.type[ra2 - 1];
      if (this.neigh !== this.trace && (this.compareAtomtype(at2, ra1) || this.compareAtomtype(at1, ra2)) &&
          this.glove[R.tsp[this.pion][this.trace] - 1] === 0) {
        if (this.avail === MAXGUESS) {
          this.status = 'GUESSFAIL';
          return;
        }
        this.avail++;
        this.restore[this.avail - 1] = {
          glove: this.glove.slice(), pioneerCount: this.pioneerCount.slice(), pioneers: this.pioneers.slice(),
          pion: this.pion, neigh: this.neigh, trace: this.trace, gloveCounter: this.gloveCounter,
        };
        this.innerCrosscheckLoop();
        return;
      }
    }
  }

  private innerCrosscheckLoop(): void {
    const R = this.rxn.reactant;
    const site = this.glove[this.pion];
    const nfirst = R.tn[this.pion][0];
    const choices: number[] = [];
    const rA = R.tsp[this.pion][this.neigh];
    for (let i = 0; i < nfirst; i++) {
      const cand = this.xs(site, i);
      if (this.compareAtomtype(this.typeOf(cand), rA)) {
        if (choices.length === 5) { this.status = 'GUESSFAIL'; return; }
        choices.push(cand);
      }
    }
    const gb = (): number => this.guessBranch[this.avail];
    if (gb() === 0) this.guessBranch[this.avail] = choices.length;
    choices.sort((a, b) => a - b);
    for (let i = gb() - 1; i >= 0; i--) {
      let already = false;
      for (let j = 0; j < R.n; j++) if (this.glove[j] === choices[i]) { already = true; break; }
      if (already) {
        this.guessBranch[this.avail]--;
        if (gb() === 0) { this.status = 'REJECT'; return; }
      } else {
        this.glove[rA - 1] = choices[i];
        this.guessBranch[this.avail]--;
        break;
      }
    }
    if (gb() === 0) this.avail--;
    for (let i = 0; i < R.tn[rA - 1][0]; i++) this.pioneerCount[R.tsp[rA - 1][i] - 1]++;
    this.gloveCounter++;
    if (this.gloveCounter === R.n) {
      this.status = this.ringCheck() && this.checkConstraints() ? 'ACCEPT' : 'GUESSFAIL';
      return;
    }
    this.status = 'CONTINUE';
  }
}

/** Builds the arguments of one `react` block and the options that follow it (fix_bond_react.cpp constructor). */
interface ReactArgs { name: string; group: string; nevery: number; rmin: number; rmax: number; pre: string; post: string; map: string; opts: string[] }

const parseReactBlock = (a: string[], i: number): { r: ReactArgs; next: number } => {
  if (i + 6 > a.length) throw new StyleError('Illegal fix bond/react command: \'react\' has too few arguments');
  const v = (x: string, what: string) => {
    if (x.startsWith('v_')) throw new StyleError(`fix bond/react: variable ${x} for ${what} is not supported by the browser engine`);
    return x;
  };
  const r: ReactArgs = {
    name: a[i + 1], group: a[i + 2], nevery: Number(v(a[i + 3], 'Nevery')), rmin: Number(v(a[i + 4], 'Rmin')),
    rmax: Number(v(a[i + 5], 'Rmax')), pre: a[i + 6], post: a[i + 7] ?? '', map: a[i + 8] ?? '', opts: [],
  };
  if (!Number.isFinite(r.nevery) || r.nevery <= 0 || !Number.isInteger(r.nevery)) throw new StyleError("Illegal fix bond/react command: 'Nevery' must be a positive integer");
  if (!(r.rmin >= 0)) throw new StyleError("Illegal fix bond/react command: 'Rmin' cannot be negative");
  if (!(r.rmax >= 0)) throw new StyleError("Illegal fix bond/react command: 'Rmax' cannot be negative");
  let k = i + 9;
  while (k < a.length && a[k] !== 'react') r.opts.push(a[k++]);
  return { r, next: k };
};

/** A reaction's templates, map and keywords, with the template ID lookup from the session. */
const buildReaction = (sys: System, id: number, r: ReactArgs, stabilization: boolean): Rxn => {
  const get = (tid: string): MoleculeTemplate => {
    const list = sys.molecules.get(tid);
    if (!list || !list.length) throw new StyleError(`Pre-reaction molecule template ID for fix bond/react does not exist: ${tid}`);
    return list[0];
  };
  const reactant = templateOf(get(r.pre));
  const product = templateOf(get(r.post));
  const groupBit = sys.groupBit(r.group);
  const rxn: Rxn = {
    ID: id, name: r.name, reactant, product, groupbit: groupBit, nevery: r.nevery,
    rminsq: r.rmin * r.rmin, rmaxsq: r.rmax * r.rmax, ibonding: 0, jbonding: 0, iatomtype: 0, jatomtype: 0,
    closeneigh: -1, fraction: 1.0, seed: 12345, random: new RanMars(12345), limitDuration: 60,
    rescaleFlag: false, rescaleCount: 0, molTotalCharge: 0, moleculeKeyword: 'off', atoms: [], attempts: [],
    localCount: 0, nlocalkeep: Number.MAX_SAFE_INTEGER, reactionCountTotal: 0,
    createFlag: false, overlapsq: 0, fitFragment: -1,
  };
  const o = r.opts;
  for (let k = 0; k < o.length;) {
    const key = o[k];
    if (key === 'prob') {
      const f = o[k + 1], sd = o[k + 2];
      if (f === undefined || sd === undefined) throw new StyleError("Illegal fix bond/react command: 'prob' keyword has too few arguments");
      if (f.startsWith('v_')) throw new StyleError('fix bond/react: prob v_ variables are not supported by the browser engine');
      rxn.fraction = Number(f);
      rxn.seed = Number.parseInt(sd, 10);
      if (!(rxn.fraction >= 0 && rxn.fraction <= 1)) throw new StyleError('Illegal fix bond/react command: probability fraction must between 0 and 1, inclusive');
      if (!(rxn.seed > 0)) throw new StyleError('Illegal fix bond/react command: probability seed must be positive');
      k += 3;
    } else if (key === 'stabilize_steps') {
      if (!stabilization) throw new StyleError('Stabilize_steps keyword used without stabilization keyword');
      rxn.limitDuration = Number.parseInt(o[k + 1] ?? '', 10);
      k += 2;
    } else if (key === 'rescale_charges') {
      const w = o[k + 1];
      if (w === 'no') rxn.rescaleFlag = false;
      else if (w === 'yes') {
        if (!reactant.t.q && !product.t.q) throw new StyleError('Illegal fix bond/react command: cannot use \'rescale_charges\' without atomic charges enabled');
        if (!product.t.q) throw new StyleError("Illegal fix bond/react command: cannot use 'rescale_charges' without Charges section in post-reaction template");
        rxn.rescaleFlag = true;
      } else throw new StyleError("Bond/react: Illegal option for 'rescale_charges' keyword");
      k += 2;
    } else if (key === 'molecule') {
      const w = o[k + 1];
      if (w !== 'off' && w !== 'inter' && w !== 'intra') throw new StyleError("Fix bond/react: Illegal option for 'molecule' keyword");
      rxn.moleculeKeyword = w;
      k += 2;
    } else if (key === 'custom_charges') {
      if (o[k + 1] !== 'no') throw new StyleError('fix bond/react custom_charges (fragments) is not supported by the browser engine');
      k += 2;
    } else if (key === 'modify_create') {
      // modify_create (fix_bond_react.cpp): sub-keywords fit (all or a fragment name) and overlap R, until another keyword
      let j = k + 1;
      while (j < o.length && (o[j] === 'fit' || o[j] === 'overlap')) {
        if (o[j] === 'fit') {
          const frag = o[j + 1];
          if (frag === undefined) throw new StyleError("Illegal fix bond/react command: 'modify_create' has too few arguments");
          if (frag === 'all') rxn.fitFragment = -1;
          else {
            const idx = product.t.fragments.findIndex((x) => x.name === frag);
            if (idx < 0) throw new StyleError("Fix bond/react: Molecule fragment for 'modify_create' keyword does not exist");
            rxn.fitFragment = idx;
          }
          j += 2;
        } else {
          const r = Number(o[j + 1]);
          if (!Number.isFinite(r)) throw new StyleError("Illegal fix bond/react command: 'modify_create' overlap is not a number");
          rxn.overlapsq = r * r;
          j += 2;
        }
      }
      k = j;
    } else {
      throw new StyleError(`Illegal fix bond/react command: unknown keyword '${key}' in reaction ${r.name}`);
    }
  }
  return rxn;
};

/** Native-order atom list of the system (the order native LAMMPS loops over its local atoms). */
export class FixBondReact extends Fix {
  readonly style = 'bond/react';
  private rxns: Rxn[] = [];
  private stabilization = false;
  private exclude = '';
  private nveXmax = '';
  private resetMolIds = true;
  private maxRxn: { names: string[]; ids: number[]; max: number }[] = [];
  private rateLimits: { names: string[]; ids: number[]; nlimit: number; nsteps: number; store: number[] }[] = [];
  private readonly lim: CustomProp;
  private readonly react: CustomProp;
  private readonly inst: CustomProp;
  private readonly statted: CustomProp | null;
  private maxNatoms = 0;
  private nve: FixNVELimit | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 8) throw new StyleError('usage: fix ID group-ID bond/react keyword value ... react ...');
    // common keywords come first, then one or more react blocks
    let i = 0;
    const nrxn = args.filter((w) => w === 'react').length;
    if (nrxn === 0) throw new StyleError("Fix bond/react is missing mandatory 'react' keyword");
    const modeWords = new Set(['stabilization', 'reset_mol_ids', 'max_rxn', 'rate_limit', 'shuffle_seed', 'file', 'vizsteps']);
    const maxRxnPending: { names: string[]; max: number }[] = [];
    for (; i < args.length && args[i] !== 'react';) {
      const key = args[i];
      if (!modeWords.has(key)) throw new StyleError(`Unknown fix bond/react command keyword ${key}`);
      if (key === 'stabilization') {
        const yes = args[i + 1];
        if (yes !== 'yes' && yes !== 'no') throw new StyleError('Illegal fix bond/react stabilization value');
        this.stabilization = yes === 'yes';
        if (this.stabilization) {
          this.exclude = args[i + 2] ?? '';
          this.nveXmax = args[i + 3] ?? '';
          if (!this.exclude || !this.nveXmax) throw new StyleError('fix bond/react stabilization yes needs a group and xmax');
          i += 4;
        } else i += 2;
      } else if (key === 'reset_mol_ids') {
        const w = args[i + 1];
        if (w === 'molmap') throw new StyleError('fix bond/react reset_mol_ids molmap is not supported by the browser engine');
        if (w !== 'yes' && w !== 'no') throw new StyleError(`Unknown option ${w} for 'reset_mol_ids' keyword`);
        this.resetMolIds = w === 'yes';
        i += 2;
      } else if (key === 'max_rxn') {
        const names: string[] = [];
        let j = i + 1;
        while (j < args.length && /^[A-Za-z]/.test(args[j]) && args[j] !== 'react') names.push(args[j++]);
        if (!names.length) throw new StyleError("Illegal fix bond/react command: at least one rxn-ID should be listed directly after the 'max_rxn' keyword");
        const max = Number.parseInt(args[j] ?? '', 10);
        if (!(max >= 0)) throw new StyleError("Illegal fix bond/react command: 'max_rxn' cannot be negative");
        maxRxnPending.push({ names, max });
        i = j + 1;
      } else if (key === 'rate_limit') {
        const names: string[] = [];
        let j = i + 1;
        while (j < args.length && /^[A-Za-z]/.test(args[j]) && args[j] !== 'react') names.push(args[j++]);
        if (!names.length) throw new StyleError("Illegal fix bond/react command: at least one rxn-ID should be listed directly after the 'rate_limit' keyword");
        const nl = args[j] ?? '';
        if (nl.startsWith('v_')) throw new StyleError('fix bond/react rate_limit with a variable is not supported by the browser engine');
        const nlimit = Number.parseInt(nl, 10);
        const nsteps = Number.parseInt(args[j + 1] ?? '', 10);
        if (!(nsteps > 0)) throw new StyleError('Illegal fix bond/react rate_limit: Nsteps must be positive');
        this.rateLimits.push({ names, ids: [], nlimit, nsteps, store: new Array(nsteps).fill(-1) });
        i = j + 2;
      } else if (key === 'shuffle_seed') {
        // the seed drives the shuffled overstep, which is not ported (see the multi-reaction errors below)
        if (!Number.isFinite(Number.parseInt(args[i + 1] ?? '', 10))) throw new StyleError('Illegal fix bond/react shuffle_seed');
        i += 2;
      } else {
        throw new StyleError(`fix bond/react ${key} is not supported by the browser engine`);
      }
    }
    // react blocks
    const blocks: ReactArgs[] = [];
    while (i < args.length) {
      if (args[i] !== 'react') throw new StyleError("Illegal fix bond/react command: 'react' or 'stabilization' has incorrect arguments");
      const { r, next } = parseReactBlock(args, i);
      blocks.push(r);
      i = next;
    }
    if (blocks.length !== nrxn) throw new StyleError('Illegal fix bond/react command: react blocks');
    if (sys.groups.find(this.group) < 0 && this.group !== 'all') throw new StyleError(`Could not find fix group ID ${this.group}`);
    this.rxns = blocks.map((b, k) => buildReaction(sys, k, b, this.stabilization));
    this.vectorFlag = true;
    this.sizeVector = nrxn;
    this.nevery = 1;
    // rate limits and max_rxn refer to reactions by name
    const idOf = (name: string): number => {
      const r = this.rxns.find((x) => x.name === name);
      if (!r) throw new StyleError(`Fix bond/react: Invalid reaction name ${name} listed`);
      return r.ID;
    };
    for (const rl of this.rateLimits) rl.ids = rl.names.map(idOf);
    this.maxRxn = maxRxnPending.map((m) => ({ names: m.names, ids: m.names.map(idOf), max: m.max }));
    // map files
    this.maxNatoms = this.rxns.reduce((m, r) => Math.max(m, r.product.n), 0);
    for (const rxn of this.rxns) {
      rxn.atoms = Array.from({ length: this.maxNatoms }, (_, idx) => ({
        edge: 0, landlocked: 0, wildcard: false, recharged: 1, deleted: 0, created: 0,
        amap: [idx + 1, idx + 1] as [number, number], ramap: [0, 0] as [number, number],
      }));
    }
    for (const [k, b] of blocks.entries()) readMap(this.rxns[k], sys.readFile(b.map), b.map);
    for (const rxn of this.rxns) {
      rxn.iatomtype = rxn.reactant.t.type[rxn.ibonding - 1];
      rxn.jatomtype = rxn.reactant.t.type[rxn.jbonding - 1];
      findLandlocked(rxn);
    }
    for (const rxn of this.rxns) {
      if (rxn.rescaleFlag) {
        rxn.rescaleCount = 0;
        for (let j = 0; j < rxn.product.n; j++) {
          const jj = rxn.atoms[j].amap[1] - 1;
          if (rxn.atoms[jj].recharged === 1 && rxn.atoms[jj].deleted === 0) {
            rxn.molTotalCharge += rxn.product.t.q![j];
            rxn.rescaleCount++;
          }
        }
      }
    }
    // closeneigh: are the initiators 1-2, 1-3 or 1-4 neighbours in the reactant?
    for (const rxn of this.rxns) {
      rxn.closeneigh = -1;
      const R = rxn.reactant;
      for (let k = 0; k < R.tn[rxn.ibonding - 1][2]; k++) {
        if (R.tsp[rxn.ibonding - 1][k] === rxn.jbonding) {
          rxn.closeneigh = 2;
          if (k < R.tn[rxn.ibonding - 1][1]) rxn.closeneigh = 1;
          if (k < R.tn[rxn.ibonding - 1][0]) rxn.closeneigh = 0;
          break;
        }
      }
    }
    for (const rxn of this.rxns) rxn.random = new RanMars(rxn.seed);
    // per-atom bookkeeping stored as custom per-atom vectors (fix property/atom names of native)
    const s = sys.state;
    const prop = (name: string, init: number): CustomProp => {
      const c = s.custom.get(name);
      if (c && c.int && c.cols === 0) return c;
      const nc: CustomProp = { int: true, cols: 0, data: new Float64Array(s.n).fill(init) };
      s.custom.set(name, nc);
      return nc;
    };
    this.lim = prop('limit_tags', 0);
    this.react = prop('react_tags', 0);
    this.inst = prop('rxn_instance', 0);
    this.statted = this.stabilization ? prop('statted_tags', 1) : null;
    // master group: atoms with limit_tags != 0 (dynamic, parent all)
    this.defineDynamic('bond_react_MASTER_group', 'all', 'limit_tags');
    if (this.stabilization) {
      const exists = sys.groups.find(this.exclude);
      const parentStatic = exists >= 0 && !sys.groups.isDynamic(bitOfIndex(exists));
      if (exists < 0 || parentStatic) {
        const reactName = `${this.exclude}_REACT`;
        this.defineDynamic(reactName, exists < 0 ? 'all' : this.exclude, 'statted_tags');
        this.exclude = reactName;
      } else {
        throw new StyleError(`fix bond/react stabilization with dynamic group ${this.exclude} is not supported by the browser engine`);
      }
      // nve/limit on the reacting atoms (native internal fix bond_react_MASTER_nve_limit, created before this fix)
      this.nve = new FixNVELimit(sys, 'bond_react_MASTER_nve_limit', 'bond_react_MASTER_group', [this.nveXmax]);
      sys.fixes.push(this.nve);
    }
  }

  /** group ID dynamic parent property: assigned from the per-atom vector every step (group.html). */
  private defineDynamic(name: string, parent: string, property: string): void {
    const k = this.sys.groups.find(name) >= 0 ? this.sys.groups.find(name) : this.sys.groups.create(name);
    this.sys.groups.dynamic.set(bitOfIndex(k), {
      parent, region: null, variable: null, property, every: 1, within: null, includeMolecule: false, exclude: null,
    });
  }

  computeVector(i: number): number {
    return this.rxns[i]?.reactionCountTotal ?? 0;
  }

  /** Index of the atom with this ID, or -1. */
  private map(tag: number): number { return this.sys.indexOfId(tag); }

  postIntegrate(): void {
    const s = this.sys.state;
    const step = s.step;
    for (const rl of this.rateLimits) {
      let sum = 0;
      for (const id of rl.ids) sum += this.rxns[id].reactionCountTotal;
      rl.store.unshift(sum);
      rl.store.pop();
    }
    let due = false;
    for (const rxn of this.rxns) {
      if (step % rxn.nevery === 0) { due = true; break; }
    }
    for (const rxn of this.rxns) {
      rxn.localCount = 0;
      rxn.nlocalkeep = Number.MAX_SAFE_INTEGER;
    }
    if (!due) { this.unlimit(); return; }
    // reset the attempt lists of every reaction
    for (const rxn of this.rxns) rxn.attempts = [];
    const sim = this.buildSim();
    // rate limits and max_rxn decide which reactions may run on this step
    const limitFlag = this.rxns.map(() => 1);
    for (const rl of this.rateLimits) {
      const myCount = rl.store[rl.nsteps - 1];
      let nrxnDelta = 0;
      if (myCount !== -1) {
        let sum = 0;
        for (const id of rl.ids) sum += this.rxns[id].reactionCountTotal;
        nrxnDelta = sum - myCount;
      }
      if (myCount === -1 || nrxnDelta >= rl.nlimit) for (const id of rl.ids) limitFlag[id] = 0;
    }
    for (const ml of this.maxRxn) {
      let sum = 0;
      for (const id of ml.ids) sum += this.rxns[id].reactionCountTotal;
      if (sum >= ml.max) for (const id of ml.ids) limitFlag[id] = 0;
    }
    for (const rxn of this.rxns) {
      if (step % rxn.nevery !== 0 || limitFlag[rxn.ID] === 0) continue;
      const partnerOf = this.findPartners(rxn, sim);
      // mutual winners become attempts (native: count only pairs listed by both atoms; owner = smaller tag)
      const order = nativeOrder(s);
      const finalp = new Map<number, number>();
      for (const k of order) {
        const tagI = s.id[k];
        const p = partnerOf.get(tagI);
        if (!p) continue;
        const jIdx = this.map(p);
        if (partnerOf.get(s.id[jIdx]) !== tagI) continue;
        finalp.set(tagI, p);
      }
      for (const k of order) {
        const tagI = s.id[k];
        const fp = finalp.get(tagI);
        if (!fp) continue;
        if (tagI < fp) {
          if (rxn.iatomtype === s.type[k]) {
            rxn.attempts.push([tagI, fp]);
            if (rxn.iatomtype === rxn.jatomtype) rxn.attempts.push([fp, tagI]);
          } else {
            rxn.attempts.push([fp, tagI]);
          }
        }
      }
    }
    const nattempt = this.rxns.reduce((a, r) => a + r.attempts.length, 0);
    if (!nattempt) { this.unlimit(); return; }
    this.superimposeAndUpdate(sim);
    this.unlimit();
  }

  /** Simulation special lists by atom ID: 1-2, then 1-3, then 1-4 partners (native nspecial order). */
  private buildSim(): Map<number, Special> {
    const s = this.sys.state;
    const adj = new Map<number, number[]>();
    const b = s.topo.bonds;
    for (let e = 0; e < b.n; e++) {
      if (b.type[e] === 0) continue;
      const i = b.atoms[2 * e], j = b.atoms[2 * e + 1];
      if (!adj.has(i)) adj.set(i, []);
      if (!adj.has(j)) adj.set(j, []);
      adj.get(i)!.push(j);
      adj.get(j)!.push(i);
    }
    const out = new Map<number, Special>();
    for (let k = 0; k < s.n; k++) {
      const me = s.id[k];
      const seen = new Set<number>([me]);
      const shells: number[][] = [];
      let frontier = [me];
      for (let o = 1; o <= 3; o++) {
        const next: number[] = [];
        for (const a of frontier) {
          for (const c of adj.get(a) ?? []) {
            if (seen.has(c)) continue;
            seen.add(c);
            next.push(c);
          }
        }
        shells.push(next);
        frontier = next;
      }
      const flat = [...shells[0], ...shells[1], ...shells[2]];
      out.set(me, { flat, c: [shells[0].length, shells[0].length + shells[1].length, flat.length] });
    }
    return out;
  }

  /** find_partners for one reaction: each atom's nearest eligible partner (far_partner or close_partner). */
  private findPartners(rxn: Rxn, sim: Map<number, Special>): Map<number, number> {
    const s = this.sys.state;
    const out = new Map<number, number>();
    const d0 = new Map<number, number>(), d1 = new Map<number, number>();
    const order = nativeOrder(s);
    const g = rxn.groupbit;
    const okPair = (a: number, b: number) => {
      if (rxn.moleculeKeyword === 'inter' && s.molecule[a] === s.molecule[b]) return false;
      if (rxn.moleculeKeyword === 'intra' && s.molecule[a] !== s.molecule[b]) return false;
      return true;
    };
    const minimg = (dx: number, L: number, periodic: boolean): number => {
      if (!periodic) return dx;
      return dx - L * Math.round(dx / L);
    };
    const box = s.box;
    const L = [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]];
    const dsq = (a: number, b: number): number => {
      let r = 0;
      for (let d = 0; d < 3; d++) {
        const dx = minimg(s.x[3 * a + d] - s.x[3 * b + d], L[d], box.periodic[d]);
        r += dx * dx;
      }
      return r;
    };
    const ok = (k: number): boolean => (s.mask[k] & g) !== 0 && this.lim.data[k] === 0;
    if (rxn.closeneigh < 0) {
      // far partners: every pair of group atoms of the right types within the cutoff window
      for (let oi = 0; oi < order.length; oi++) {
        const i = order[oi];
        if (!ok(i)) continue;
        const ti = s.type[i];
        const sp = sim.get(s.id[i])!;
        for (let oj = 0; oj < order.length; oj++) {
          const j = order[oj];
          if (j === i || !ok(j)) continue;
          if (!okPair(i, j)) continue;
          const tj = s.type[j];
          const possible = (ti === rxn.iatomtype && tj === rxn.jatomtype) || (ti === rxn.jatomtype && tj === rxn.iatomtype);
          if (!possible) continue;
          if (sp.flat.slice(0, sp.c[2]).includes(s.id[j])) continue;
          const rsq = dsq(i, j);
          if (rsq >= rxn.rmaxsq || rsq <= rxn.rminsq) continue;
          if (rsq < (d1.get(s.id[i]) ?? BIG)) { out.set(s.id[i], s.id[j]); d1.set(s.id[i], rsq); }
        }
      }
      return out;
    }
    // close partners: search the 1-2 / 1-3 / 1-4 neighbours of each atom
    for (let oi = 0; oi < order.length; oi++) {
      const i1 = order[oi];
      const sp = sim.get(s.id[i1])!;
      const from = rxn.closeneigh !== 0 ? sp.c[rxn.closeneigh - 1] : 0;
      for (let n = from; n < sp.c[rxn.closeneigh]; n++) {
        const i2 = this.map(sp.flat[n]);
        if (i2 < 0) continue;
        if (!(s.mask[i1] & g) || !(s.mask[i2] & g)) continue;
        if (this.lim.data[i1] !== 0 || this.lim.data[i2] !== 0) continue;
        if (s.type[i1] !== rxn.iatomtype || s.type[i2] !== rxn.jatomtype) continue;
        if (!okPair(i1, i2)) continue;
        const rsq = dsq(i1, i2);
        if (rsq >= rxn.rmaxsq || rsq <= rxn.rminsq) continue;
        if (rxn.closeneigh === 0) {
          if (rsq > (d0.get(s.id[i1]) ?? 0)) { out.set(s.id[i1], s.id[i2]); d0.set(s.id[i1], rsq); }
          if (rsq > (d0.get(s.id[i2]) ?? 0)) { out.set(s.id[i2], s.id[i1]); d0.set(s.id[i2], rsq); }
        } else {
          if (rsq < (d1.get(s.id[i1]) ?? BIG)) { out.set(s.id[i1], s.id[i2]); d1.set(s.id[i1], rsq); }
          if (rsq < (d1.get(s.id[i2]) ?? BIG)) { out.set(s.id[i2], s.id[i1]); d1.set(s.id[i2], rsq); }
        }
      }
    }
    return out;
  }

  private superimposeAndUpdate(sim: Map<number, Special>): void {
    const s = this.sys.state;
    const megas: Mega[] = [];
    for (const rxn of this.rxns) {
      const sup = new Superimposer(
        rxn, (id) => sim.get(id), (id) => s.type[this.map(id)],
        (id) => this.lim.data[this.map(id)] !== 0, s.n,
        (glove) => this.totalCharge(rxn, glove),
      );
      sup.run();
      megas.push(...sup.megas);
    }
    // dedup: an atom may take part in one reaction instance only; native shuffles with reaction 0's stream
    const size = megas.length;
    const arr = megas.slice();
    for (let i = size - 1; i > 0; i--) {
      const k = Math.floor(this.rxns[0].random.uniform() * (i + 1));
      const t = arr[i]; arr[i] = arr[k]; arr[k] = t;
    }
    const mask = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      if (mask[i]) continue;
      const r1 = this.rxns[arr[i].rxn];
      for (let j = 0; j < r1.reactant.n; j++) {
        const check1 = arr[i].glove[j];
        for (let ii = i + 1; ii < size; ii++) {
          if (mask[ii]) continue;
          const r2 = this.rxns[arr[ii].rxn];
          for (let jj = 0; jj < r2.reactant.n; jj++) {
            if (arr[ii].glove[jj] === check1) { mask[ii] = 1; break; }
          }
        }
      }
    }
    const kept = arr.filter((_, i) => !mask[i]);
    // counts per reaction, and the over-limit bookkeeping (single-reaction limits; native shuffles multi-reaction ones)
    const delta = this.rxns.map(() => 0);
    for (const m of kept) delta[m.rxn]++;
    if (!kept.length) return;
    const oversteps = this.rxns.map(() => 0);
    for (const ml of this.maxRxn) {
      let sum = 0, dsum = 0;
      for (const id of ml.ids) { sum += this.rxns[id].reactionCountTotal; dsum += delta[id]; }
      const over = sum + dsum - ml.max;
      if (over > 0) {
        if (ml.ids.length !== 1) throw new StyleError('fix bond/react max_rxn with several reactions needs the shuffled overstep, which the browser engine does not support yet');
        oversteps[ml.ids[0]] = Math.max(oversteps[ml.ids[0]], over);
      }
    }
    for (const rl of this.rateLimits) {
      const myCount = rl.store[rl.nsteps - 1];
      if (myCount === -1) continue;
      let sum = 0, dsum = 0;
      for (const id of rl.ids) { sum += this.rxns[id].reactionCountTotal; dsum += delta[id]; }
      const over = sum + dsum - myCount - rl.nlimit;
      if (over > 0) {
        if (rl.ids.length !== 1) throw new StyleError('fix bond/react rate_limit with several reactions needs the shuffled overstep, which the browser engine does not support yet');
        oversteps[rl.ids[0]] = Math.max(oversteps[rl.ids[0]], over);
      }
    }
    for (const rxn of this.rxns) {
      if (oversteps[rxn.ID] > 0) {
        const cap = Math.min(oversteps[rxn.ID], delta[rxn.ID]);
        rxn.nlocalkeep = delta[rxn.ID] - cap;
      }
    }
    for (const rxn of this.rxns) rxn.localCount = delta[rxn.ID];
    this.applyUpdate(kept);
  }

  /** Sum of the charges of the reactant atoms whose charge is rescaled (native get_totalcharge). */
  private totalCharge(rxn: Rxn, glove: number[]): number {
    const s = this.sys.state;
    let q = 0;
    for (let j = 0; j < rxn.reactant.n; j++) {
      const jj = rxn.atoms[j].amap[1] - 1;
      if (rxn.atoms[jj].recharged === 1) q += s.q[this.map(glove[jj])];
    }
    return q;
  }

  /**
   * insert_atoms_setup (fix_bond_react.cpp): the post-reaction template is fitted onto the reaction site (a
   * least-squares rotation and translation over the atoms that are neither created nor deleted; the modify_create
   * fit fragment limits them). Created atoms are placed at the fitted coordinates unless one of them lies within
   * the overlap distance of an atom of the system or of an atom accepted earlier in this step. Accepted atoms are
   * queued in `pending`, their tags are appended to the instance's glove and the template maps are set.
   * Velocities: a random direction (three uniform draws of the reaction's stream) with the site temperature.
   */
  private createAtoms(rxn: Rxn, m: Mega, pending: PendingAtom[], nextTag: () => number): boolean {
    const s = this.sys.state, P = rxn.product, R = rxn.reactant;
    if (s.rmass) throw new StyleError('fix bond/react: created atoms with a per-atom mass style are not supported by the browser engine');
    const box = s.box;
    const L = [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]];
    const units = this.sys.units;
    const inFit = (j: number): boolean => rxn.fitFragment < 0 || P.t.fragments[rxn.fitFragment].atoms.includes(j + 1);
    const fit: number[] = [];
    for (let j = 0; j < P.n; j++) {
      if (!inFit(j)) continue;
      const ipre = rxn.atoms[j].amap[1] - 1;
      if (!rxn.atoms[j].created && !rxn.atoms[ipre].deleted) fit.push(j);
    }
    if (fit.length === 0) throw new StyleError('Fix bond/react: no atoms of the post-reaction template to fit the created atoms to');
    const local = (k: number): number => {
      const i = this.map(m.glove[k]);
      if (i < 0) throw new StyleError('Bond/react: Fix bond/react needs ghost atoms from further away');
      return i;
    };
    // site temperature over the reactant atoms (get_temperature; the types are not updated yet)
    let ke = 0;
    for (let j = 0; j < R.n; j++) {
      const i = local(j);
      ke += (s.v[3 * i] * s.v[3 * i] + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2]) * massOf(s, i);
    }
    const dim = s.dimension;
    const t = ke * (units.mvv2e / (dim * R.n * units.boltz));
    // fit: reference atom first, the others at their closest images to it (closest_image)
    const ref = local(rxn.atoms[fit[0]].amap[1] - 1);
    const ifit = local(rxn.ibonding - 1);
    const xfrozen = fit.map((j) => {
      const i = local(rxn.atoms[j].amap[1] - 1);
      return [0, 1, 2].map((d) => s.x[3 * ref + d] + minImage(s.x[3 * i + d] - s.x[3 * ref + d], L[d], box.periodic[d]));
    });
    const xmobile = fit.map((j) => [P.t.x[3 * j], P.t.x[3 * j + 1], P.t.x[3 * j + 2]]);
    const { R: rot, T } = superpose(xfrozen, xmobile);
    const created: number[] = [];
    for (let j = 0; j < P.n; j++) if (rxn.atoms[j].created === 1) created.push(j);
    const coords = created.map((j) => [0, 1, 2].map((a) => rot[a][0] * P.t.x[3 * j] + rot[a][1] * P.t.x[3 * j + 1] + rot[a][2] * P.t.x[3 * j + 2] + T[a]));
    // image flags: those of the reference atom, shifted by the remap into the box
    const base = [s.image[3 * ifit], s.image[3 * ifit + 1], s.image[3 * ifit + 2]];
    const images = coords.map((c) => {
      const img = base.slice();
      for (let d = 0; d < 3; d++) {
        if (!box.periodic[d]) continue;
        const shift = Math.floor((c[d] - box.lo[d]) / L[d]);
        c[d] -= shift * L[d];
        img[d] += shift;
      }
      return img;
    });
    if (rxn.overlapsq > 0) {
      const near = (c: number[], o: number[]): boolean => {
        let r = 0;
        for (let d = 0; d < 3; d++) {
          const dx = minImage(c[d] - o[d], L[d], box.periodic[d]);
          r += dx * dx;
        }
        return r < rxn.overlapsq;
      };
      for (const c of coords) {
        for (let i = 0; i < s.n; i++) if (near(c, [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]])) return false;
        for (const p of pending) if (near(c, p.x)) return false;
      }
    }
    for (let c = 0; c < created.length; c++) {
      const j = created[c];
      const tag = nextTag();
      const preID = R.n + c + 1;
      const vtnorm = Math.sqrt(t / (units.mvv2e / (dim * units.boltz)) / s.massByType[P.t.type[j]]);
      const myv = [rxn.random.uniform(), rxn.random.uniform(), rxn.random.uniform()];
      const vnorm = Math.sqrt(myv[0] * myv[0] + myv[1] * myv[1] + myv[2] * myv[2]);
      pending.push({
        x: coords[c], type: P.t.type[j], tag, image: images[c], mask: rxn.groupbit,
        v: myv.map((x) => (x / vnorm) * vtnorm),
      });
      m.glove[preID - 1] = tag;
      rxn.atoms[j].amap = [j + 1, preID];
      rxn.atoms[preID - 1].ramap = [preID, j + 1];
    }
    return true;
  }

  /** update_everything: types, charges, stabilization flags, topology, deletions. */
  private applyUpdate(megas: Mega[]): void {
    const s = this.sys.state;
    const step = s.step;
    // per reaction: instances applied (native noccur / nlocalkeep)
    const applied: Mega[] = [];
    const occur = this.rxns.map(() => 0);
    // created atoms take the IDs after the largest ID of the system (native addatomtag), in the order they are accepted
    let addTag = 0;
    for (let i = 0; i < s.n; i++) addTag = Math.max(addTag, s.id[i]);
    const pending: PendingAtom[] = [];
    for (const m of megas) {
      const rxn = this.rxns[m.rxn];
      if (occur[rxn.ID] >= rxn.nlocalkeep) continue;
      // a reaction whose created atoms would overlap is skipped and does not count (native insert_atoms_setup)
      if (rxn.createFlag && !this.createAtoms(rxn, m, pending, () => ++addTag)) continue;
      occur[rxn.ID]++;
      applied.push(m);
    }
    // insert all created atoms at once (native update_everything), before the types, charges and topology update
    for (const p of pending) {
      appendAtoms(s, { x: Float64Array.from(p.x), type: Int32Array.of(p.type), id: Int32Array.of(p.tag), v: Float64Array.from(p.v), image: Int32Array.from(p.image), mask: p.mask });
    }
    if (pending.length) this.sys.atomsChanged();
    for (const rxn of this.rxns) rxn.reactionCountTotal += occur[rxn.ID];
    // types, charges and stabilization bookkeeping of the atoms of every instance
    for (const m of applied) {
      const rxn = this.rxns[m.rxn];
      let addend = 0;
      if (rxn.rescaleFlag) addend = (m.charge - rxn.molTotalCharge) / rxn.rescaleCount;
      for (let j = 0; j < rxn.product.n; j++) {
        const jj = rxn.atoms[j].amap[1] - 1;
        const il = this.map(m.glove[jj]);
        if (il < 0) continue;
        this.lim.data[il] = step + 1;
        if (this.statted) this.statted.data[il] = 0;
        this.react.data[il] = rxn.ID;
        this.inst.data[il] = m.glove[rxn.ibonding] ?? 0;
        if (rxn.atoms[j].landlocked === 1 && !rxn.atoms[jj].wildcard) s.type[il] = rxn.product.t.type[j];
        if (rxn.product.t.q && rxn.atoms[jj].recharged === 1) s.q[il] = rxn.product.t.q[j] + addend;
      }
    }
    // atoms to delete (the reactant's DeleteIDs), by ID
    const deadTags = new Set<number>();
    for (const m of applied) {
      const rxn = this.rxns[m.rxn];
      for (let j = 0; j < rxn.reactant.n; j++) if (rxn.atoms[j].deleted === 1) deadTags.add(m.glove[j]);
    }
    // topology: per reaction, remove the entries that touch a landlocked or deleted atom, add the post-reaction ones
    for (const m of applied) {
      const rxn = this.rxns[m.rxn];
      const tagOf = (j: number) => m.glove[rxn.atoms[j].amap[1] - 1];
      const landTags = new Set<number>();
      for (let j = 0; j < rxn.product.n; j++) if (rxn.atoms[j].landlocked === 1) landTags.add(tagOf(j));
      const touchOld = (tags: number[]) => tags.some((t) => landTags.has(t) || deadTags.has(t));
      const landed = (idx: number[]) => idx.some((j) => rxn.atoms[j - 1].landlocked === 1);
      const P = rxn.product.t;
      // template entries are 1-based product atom IDs; the landed test and the tags both use them
      const adds = (kind: 'bonds' | 'angles' | 'dihedrals' | 'impropers', entries: number[][]) => entries
        .filter((e) => landed(e.slice(1)))
        .map((e) => {
          const at = e.slice(1).map((a) => tagOf(a - 1));
          // a wildcard atom (any type in the pre-reaction template) makes the type come from the labels
          const wild = e.slice(1).some((a) => rxn.atoms[rxn.atoms[a - 1].amap[1] - 1].wildcard);
          return wild ? this.inferWildcardEntry(kind, at) : { ty: e[0], at };
        });
      this.rewriteTopo('bonds', (e) => touchOld(e), adds('bonds', P.bonds));
      this.rewriteTopo('angles', (e) => touchOld(e), adds('angles', P.angles));
      this.rewriteTopo('dihedrals', (e) => touchOld(e), adds('dihedrals', P.dihedrals));
      this.rewriteTopo('impropers', (e) => touchOld(e), adds('impropers', P.impropers));
    }
    // dead atoms may not keep any topology
    if (deadTags.size) {
      for (const k of ['bonds', 'angles', 'dihedrals', 'impropers'] as const) {
        this.rewriteTopo(k, (e) => e.some((t) => deadTags.has(t)), []);
      }
    }
    // The reaction never re-initialises the pair style, so the pair_modify tail terms keep the values of the last
    // run setup (measured with native LAMMPS: the tail term of the thermo pressure is the same on every row of a
    // run that reacts, whatever the new type counts are). atomsChanged would recount the types, so restore them.
    const ff = this.sys.ff;
    const etail = ff.etailV, ptail = ff.ptailV;
    if (deadTags.size) {
      const flags = new Uint8Array(s.n);
      for (let i = 0; i < s.n; i++) if (deadTags.has(s.id[i])) flags[i] = 1;
      this.sys.deleteAtoms(flags);
    } else this.sys.atomsChanged();
    ff.etailV = etail;
    ff.ptailV = ptail;
    // Charge-only changes keep the charge sums of the kspace self and background energies at their last refresh;
    // deleting atoms refreshes them at the next compute (KSpace.chargeSumsFrozen, measured with native LAMMPS).
    if (ff.kspace) ff.kspace.chargeSumsFrozen = deadTags.size === 0;
    // atomsChanged rebuilds the force field's special list, but the neighbour list keeps the one it was given
    // at run setup: hand it the new lists so this step's neighbour build excludes the right pairs
    const sp = this.sys.ff.special;
    this.sys.nb.setSpecial(buildSpecial(this.sys.state, { angle: sp.angle, dihedral: sp.dihedral }));
  }

  /**
   * A post-reaction entry with wildcard atoms (fix_bond_react.cpp update_everything): the type is inferred from the
   * labels of the current types of its atoms, and a reversed match reverses the atom order. Native messages:
   * "Bond/react: Unable to infer bond type from wildcard atoms" and the same for angle, dihedral and improper.
   */
  private inferWildcardEntry(kind: 'bonds' | 'angles' | 'dihedrals' | 'impropers', at: number[]): { ty: number; at: number[] } {
    const L = this.sys.labels;
    const s = this.sys.state;
    const types = at.map((t) => {
      const i = this.map(t);
      if (i < 0) throw new StyleError('Bond/react: Fix bond/react needs ghost atoms from further away');
      return s.type[i];
    });
    const fail = (what: string) => new StyleError(`Bond/react: Unable to infer ${what} type from wildcard atoms`);
    if (kind === 'bonds') {
      const v = inferBondType(L, types[0], types[1]);
      if (!v) throw fail('bond');
      return { ty: Math.abs(v), at };
    }
    if (kind === 'angles') {
      const v = inferAngleType(L, types[0], types[1], types[2]);
      if (!v) throw fail('angle');
      return { ty: Math.abs(v), at: v > 0 ? at : [at[2], at[1], at[0]] };
    }
    if (kind === 'dihedrals') {
      const v = inferDihedralType(L, types[0], types[1], types[2], types[3]);
      if (!v) throw fail('dihedral');
      return { ty: Math.abs(v), at: v > 0 ? at : [at[3], at[2], at[1], at[0]] };
    }
    const r = inferImproperType(L, [types[0], types[1], types[2], types[3]], this.improperSymmetry());
    if (!r.type) throw fail('improper');
    return { ty: Math.abs(r.type), at: r.type > 0 ? at : r.iorder.map((k) => at[k]) };
  }

  /**
   * The symmatoms of the improper style (improper_class2.html, Symmetry convention: "the second atom in the
   * quadruplet is the atom of symmetry; all other atoms are considered interchangeable"). No improper style
   * gives no symmetry, as in native (improper_style none).
   */
  private improperSymmetry(): number[] {
    const name = this.sys.ff.improper?.name;
    if (name === undefined) return [0, 0, 0, 0];
    if (name === 'class2') return [0, 1, 0, 0];
    throw new StyleError(`fix bond/react: wildcard impropers need the symmetry of improper style ${name}, which the browser engine does not provide`);
  }

  /**
   * Removes the entries of one topology list for which `drop` is true (by atom IDs), then appends `add`
   * (each entry {ty, at}).
   */
  private rewriteTopo(kind: 'bonds' | 'angles' | 'dihedrals' | 'impropers', drop: (atoms: number[]) => boolean,
    add: { ty: number; at: number[] }[]): void {
    const L: TopoList = this.sys.state.topo[kind];
    const w = L.width;
    const keep: { ty: number; at: number[] }[] = [];
    for (let e = 0; e < L.n; e++) {
      const at: number[] = [];
      for (let k = 0; k < w; k++) at.push(L.atoms[w * e + k]);
      if (drop(at)) continue;
      keep.push({ ty: L.type[e], at });
    }
    for (const x of add) {
      if (x.at.some((t) => this.map(t) < 0)) throw new StyleError('Bond/react: Fix bond/react needs ghost atoms from further away');
      keep.push(x);
    }
    const n = keep.length;
    const type = new Int32Array(n), atoms = new Int32Array(n * w);
    keep.forEach((x, e) => { type[e] = x.ty; for (let k = 0; k < w; k++) atoms[w * e + k] = x.at[k]; });
    L.n = n;
    L.type = type;
    L.atoms = atoms;
  }

  /** unlimit_bond: atoms whose reaction is older than limit_duration leave the stabilization groups. */
  private unlimit(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (this.lim.data[i] === 0) continue;
      const rid = this.react.data[i];
      const dur = this.rxns[rid]?.limitDuration ?? this.rxns[0].limitDuration;
      if (s.step + 1 - this.lim.data[i] > dur) {
        this.lim.data[i] = 0;
        if (this.statted) this.statted.data[i] = 1;
        this.react.data[i] = 0;
        this.inst.data[i] = 0;
      }
    }
  }

  /** Run setup: the kspace charge sums follow the charges again (the setup refresh) until a reaction freezes them. */
  setup(): void {
    if (this.sys.ff.kspace) this.sys.ff.kspace.chargeSumsFrozen = false;
    super.setup();
  }

  postForce(): void {
    if (!this.resetMolIds) return;
    this.resetMolecules();
  }

  /**
   * reset_mol_ids yes (reset_atoms_mol.cpp): molecule IDs of the group's atoms follow the bond fragments.
   * Fragment IDs are the lowest atom ID of each connected group, and the compressed IDs are numbered in
   * order of ascending fragment ID after the largest molecule ID outside the group.
   */
  private resetMolecules(): void {
    const s = this.sys.state;
    const g = this.groupBit;
    const parent = new Map<number, number>();
    const find = (x: number): number => {
      let r = x;
      while (parent.get(r) !== r) r = parent.get(r)!;
      return r;
    };
    for (let i = 0; i < s.n; i++) if (s.mask[i] & g) parent.set(s.id[i], s.id[i]);
    const b = s.topo.bonds;
    for (let e = 0; e < b.n; e++) {
      const i = b.atoms[2 * e], j = b.atoms[2 * e + 1];
      if (!parent.has(i) || !parent.has(j) || b.type[e] === 0) continue;
      const ri = find(i), rj = find(j);
      if (ri !== rj) parent.set(Math.max(ri, rj), Math.min(ri, rj));
    }
    // compute fragment/atom: atoms without any 1-2 partner get fragment ID 0 (single 0); others the cluster minimum
    const hasBond = new Set<number>();
    for (let e = 0; e < b.n; e++) {
      if (b.type[e] === 0) continue;
      hasBond.add(b.atoms[2 * e]);
      hasBond.add(b.atoms[2 * e + 1]);
    }
    const frag = new Map<number, number>();
    for (const id of parent.keys()) frag.set(id, hasBond.has(id) ? find(id) : 0);
    // chunk/atom molecule compress yes: consecutive chunk IDs in ascending fragment ID order
    const sorted = [...new Set(frag.values())].sort((a, c) => a - c);
    const chunk = new Map<number, number>(sorted.map((r, k) => [r, k + 1]));
    const single = sorted.length > 0 && sorted[0] === 0;
    let offset = 0;
    for (let i = 0; i < s.n; i++) if (!(s.mask[i] & g)) offset = Math.max(offset, s.molecule[i]);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & g)) continue;
      const c = chunk.get(frag.get(s.id[i])!) ?? 0;
      s.molecule[i] = single ? (c === 1 ? 0 : c + offset - 1) : c + offset;
    }
  }

}

/** fix bond/react ID group-ID bond/react keyword value ... react ... (docs.lammps.org/fix_bond_react.html). */
export const FIXES: Record<string, FixFactory> = {
  'bond/react': (s, id, group, a) => new FixBondReact(s, id, group, a),
};
