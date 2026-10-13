/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_kolmogorov_crespi_full.cpp and src/INTERLAYER/pair_kolmogorov_crespi_full.h, at commit
 * b332f28. Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 */
import { hasMolecule } from '../../engine/atoms';
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK, type NeighList } from '../../engine/neighbor';
import { InterlayerZBase } from './zbase';

/** Parameter entry needed by the normal (per-molecule) machinery: the element pair, z0, lambda, rcut and 1/delta^2. */
export interface NormalParam {
  ie: number; je: number; z0: number; lambda: number; rcut: number; delta2inv: number;
}

type Tensor3 = number[][][];
const zeroT3 = (): Tensor3 => [0, 1, 2].map(() => [0, 1, 2].map(() => [0, 0, 0]));



/**
 * Shared machinery of the per-molecule normal styles (kolmogorov/crespi/full, ilp/graphene/hbn): the settings,
 * the element map, the KC neighbors (same molecule, within each pair's rcut), the normals and their derivatives, and
 * the loop over the full neighbor list. Subclasses supply the parameter entries and the two pair terms.
 */
export abstract class KCNormalBase<P extends NormalParam> extends InterlayerZBase<P> {
  /** tap_flag (settings): 1 applies the taper function. */
  protected tapFlag = 0;
  /** Energies of the attractive (index 0) and repulsive (index 1) parts, the pair style's own vector. */
  protected readonly pvector = new Float64Array(2);
  /** Squared KC-neighbor cutoff per type pair (the rcut of the potential entry), set in initStyle. */
  protected kcCut2 = new Float64Array(0);
  // per-atom scratch of the normals (owned atoms): KC neighbors (up to 3), their count, normal and derivatives
  protected cap = 0;
  protected kcN = new Int32Array(0);
  protected kcJ = new Int32Array(0);
  protected normal = new Float64Array(0);
  /** dnormdri[i][a][b] at 9 i + 3 a + b. */
  protected dnormdri = new Float64Array(0);
  /** dnormal[i][k][a][b] at 27 i + 9 k + 3 a + b (k = KC neighbor index). */
  protected dnormal = new Float64Array(0);
  protected listInum = 0;

  constructor() {
    super();
    // the normals need the neighbors of every owned atom (a full list), and the attractive part reads it too
    this.needsFull = true;
    this.needsHalf = false;
  }

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length < 1 || args.length > 2) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (args.length === 2) {
      const t = Number(args[1]);
      if (!Number.isInteger(t)) throw new StyleError(`expected an integer for tap_flag, got '${args[1]}'`);
      this.tapFlag = t;
    }
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 4) throw new StyleError(`usage: pair_coeff * * filename elem1 ... elemN (style ${this.name})`);
    const nt = this.ntypes + 1;
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style ${this.name} needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    }
    // map_element2type: every type pair with both types mapped is set (the I J arguments are not used)
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') {
        this.map[t] = -1;
        continue;
      }
      let idx = this.elements.indexOf(name);
      if (idx < 0) {
        this.elements.push(name);
        idx = this.elements.length - 1;
      }
      this.map[t] = idx;
    }
    this.readFile(args[2], ctx);
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        if (this.map[i] >= 0 && this.map[j] >= 0) {
          this.setflag[i * nt + j] = 1;
          this.setflag[j * nt + i] = 1;
        }
      }
    }
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError(`Pair style ${this.name} requires newton pair on`);
    if (!ctx.s || !hasMolecule(ctx.s)) throw new StyleError(`Pair style ${this.name} requires atom attribute molecule`);
    // rcut per type pair, from the element pair of the two types
    const nt = this.ntypes + 1;
    this.kcCut2 = new Float64Array(nt * nt);
    const ne = this.elements.length;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = 1; j <= this.ntypes; j++) {
        const ei = this.map[i], ej = this.map[j];
        if (ei < 0 || ej < 0) continue;
        const p = this.params[this.elem2param[ei * ne + ej]];
        this.kcCut2[i * nt + j] = p.rcut * p.rcut;
      }
    }
  }

  override extract(name: string): unknown {
    return name === 'pvector' ? this.pvector : undefined;
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    if (pc.vatom) throw new StyleError(`per-atom virial (compute stress/atom) is not supported for pair style ${this.name} by the browser engine`);
    this.pvector.fill(0);
    this.listInum = list.inum;
    this.ensureScratch(list.inum);
    this.buildKCNeighbors(pc, list);
    this.calcNormals(pc);
    this.calcFvdW(pc, list);
    this.calcFRep(pc, list);
  }

  private ensureScratch(n: number): void {
    if (n <= this.cap) return;
    this.cap = n;
    this.kcN = new Int32Array(n);
    this.kcJ = new Int32Array(3 * n);
    this.normal = new Float64Array(3 * n);
    this.dnormdri = new Float64Array(9 * n);
    this.dnormal = new Float64Array(27 * n);
  }

  /** KC neighbors of each owned atom: same molecule, closer than its rcut, excluding the atom itself (KC_neigh). */
  private buildKCNeighbors(pc: PairCompute, list: NeighList): void {
    const { x, type } = pc;
    const nt = this.ntypes + 1;
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      let n = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq !== 0 && rsq < this.kcCut2[ti * nt + type[j]] && mol[owner[i]] === mol[owner[j]]) {
          if (n >= 3) throw new StyleError('There are too many neighbors for some atoms, please check your configuration');
          this.kcJ[3 * i + n] = j;
          n++;
        }
      }
      this.kcN[i] = n;
    }
  }

  /** Normal of each owned atom from its KC neighbors and the derivatives of the normal (calc_normal). */
  private calcNormals(pc: PairCompute): void {
    const x = pc.x;
    for (let i = 0; i < this.listInum; i++) {
      const cont = this.kcN[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const vet: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let c = 0; c < cont; c++) {
        const j = this.kcJ[3 * i + c];
        vet[c][0] = x[3 * j] - xi;
        vet[c][1] = x[3 * j + 1] - yi;
        vet[c][2] = x[3 * j + 2] - zi;
      }
      this.dnormdri.fill(0, 9 * i, 9 * i + 9);
      this.dnormal.fill(0, 27 * i, 27 * i + 27);
      if (cont <= 1) {
        this.normal[3 * i] = 0;
        this.normal[3 * i + 1] = 0;
        this.normal[3 * i + 2] = 1;
        continue;
      }
      if (cont > 3) throw new StyleError('There are too many neighbors for calculating normals');
      const dpv12 = this.dpvT(vet);
      const pv12 = cross(vet[0], vet[1]);
      let n1: number[];
      let dn1: Tensor3;
      if (cont === 2) {
        // dpvdri: derivative of the cross product with respect to atom i
        const dpvdri = [
          [0, vet[0][2] - vet[1][2], vet[1][1] - vet[0][1]],
          [vet[1][2] - vet[0][2], 0, vet[0][0] - vet[1][0]],
          [vet[0][1] - vet[1][1], vet[1][0] - vet[0][0], 0],
        ];
        n1 = pv12;
        const nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
        const nn = Math.sqrt(nn2);
        if (nn === 0) throw new StyleError('The magnitude of the normal vector is zero');
        this.setNormal(i, n1, nn);
        const dni = [0, 1, 2].map((k) => (n1[0] * dpvdri[0][k] + n1[1] * dpvdri[1][k] + n1[2] * dpvdri[2][k]) / nn);
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) this.dnormdri[9 * i + 3 * a + b] = dpvdri[a][b] / nn - n1[a] * dni[b] / nn2;
        }
        dn1 = dpv12;
      } else {
        const pv31 = cross(vet[2], vet[0]);
        const pv23 = cross(vet[1], vet[2]);
        n1 = [0, 1, 2].map((a) => (pv12[a] + pv31[a] + pv23[a]) / cont);
        const nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
        const nn = Math.sqrt(nn2);
        if (nn === 0) throw new StyleError('The magnitude of the normal vector is zero');
        this.setNormal(i, n1, nn);
        const dpv31 = this.dpv31T(vet);
        const dpv23 = this.dpv23T(vet);
        dn1 = zeroT3();
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) {
            for (let m = 0; m < 3; m++) dn1[a][b][m] = (dpv12[a][b][m] + dpv23[a][b][m] + dpv31[a][b][m]) / cont;
          }
        }
      }
      // dnn[id][m] and dnormal[id][ip][m] (per KC neighbor m)
      const nn = Math.sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]);
      const nn2 = nn * nn;
      const dnn: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let m = 0; m < 3; m++) {
        for (let id = 0; id < 3; id++) {
          dnn[id][m] = (n1[0] * dn1[0][id][m] + n1[1] * dn1[1][id][m] + n1[2] * dn1[2][id][m]) / nn;
        }
      }
      for (let m = 0; m < 3; m++) {
        for (let id = 0; id < 3; id++) {
          for (let ip = 0; ip < 3; ip++) {
            this.dnormal[27 * i + 9 * m + 3 * id + ip] = dn1[id][ip][m] / nn - n1[id] * dnn[ip][m] / nn2;
          }
        }
      }
    }
  }

  private setNormal(i: number, n1: number[], nn: number): void {
    this.normal[3 * i] = n1[0] / nn;
    this.normal[3 * i + 1] = n1[1] / nn;
    this.normal[3 * i + 2] = n1[2] / nn;
  }

  /** d(pv12)/d(vet): the derivative of the cross product of the first two neighbor vectors. */
  private dpvT(vet: number[][]): Tensor3 {
    const T = zeroT3();
    T[0][1][0] = vet[1][2]; T[0][2][0] = -vet[1][1];
    T[1][0][0] = -vet[1][2]; T[1][2][0] = vet[1][0];
    T[2][0][0] = vet[1][1]; T[2][1][0] = -vet[1][0];
    T[0][1][1] = -vet[0][2]; T[0][2][1] = vet[0][1];
    T[1][0][1] = vet[0][2]; T[1][2][1] = -vet[0][0];
    T[2][0][1] = -vet[0][1]; T[2][1][1] = vet[0][0];
    return T;
  }

  private dpv31T(vet: number[][]): Tensor3 {
    const T = zeroT3();
    T[0][1][0] = -vet[2][2]; T[0][2][0] = vet[2][1];
    T[1][0][0] = vet[2][2]; T[1][2][0] = -vet[2][0];
    T[2][0][0] = -vet[2][1]; T[2][1][0] = vet[2][0];
    T[0][1][2] = vet[0][2]; T[0][2][2] = -vet[0][1];
    T[1][0][2] = -vet[0][2]; T[1][2][2] = vet[0][0];
    T[2][0][2] = vet[0][1]; T[2][1][2] = -vet[0][0];
    return T;
  }

  private dpv23T(vet: number[][]): Tensor3 {
    const T = zeroT3();
    T[0][1][1] = vet[2][2]; T[0][2][1] = -vet[2][1];
    T[1][0][1] = -vet[2][2]; T[1][2][1] = vet[2][0];
    T[2][0][1] = vet[2][1]; T[2][1][1] = -vet[2][0];
    T[0][1][2] = -vet[1][2]; T[0][2][2] = vet[1][1];
    T[1][0][2] = vet[1][2]; T[1][2][2] = -vet[1][0];
    T[2][0][2] = -vet[1][1]; T[2][1][2] = vet[1][0];
    return T;
  }

  /** Attractive part over half of the pairs (chosen by atom IDs; calc_FvdW). */
  /** The attractive part (the subclass's pair term). */
  protected abstract calcFvdW(pc: PairCompute, list: NeighList): void;
  /** The repulsive part (the subclass's pair term). */
  protected abstract calcFRep(pc: PairCompute, list: NeighList): void;
}


const cross = (a: readonly number[], b: readonly number[]): number[] => [
  a[1] * b[2] - b[1] * a[2],
  a[2] * b[0] - b[2] * a[0],
  a[0] * b[1] - b[0] * a[1],
];
