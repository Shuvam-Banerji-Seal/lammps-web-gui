/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_ilp_tmd.cpp and src/INTERLAYER/pair_ilp_tmd.h, at commit b332f28 (the ring-neighbor
 * ILP_neigh and the calc_normal of the TMD variant, written for the six in-plane neighbors of a TMD atom; calc_FvdW
 * and calc_FRep are inherited from the ilp/graphene/hbn port of ./ilp_graphene_hbn.ts, as native inherits them).
 * Contributing author of the TMD variant: Wengen Ouyang (Wuhan University). Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, type NeighList } from '../../engine/neighbor';
import { PairILPGrapheneHBN } from './ilp_graphene_hbn';

/** Neighbors per atom of the TMD variant (Nnei in pair_ilp_tmd.h). */
const NNEI = 6;
/** Elements whose neighbors are the ring of same-type atoms (ILP_neigh; Te is included there, as in native). */
const RING_ELEMENTS = new Set(['Mo', 'W', 'S', 'Se', 'Te']);
/** Elements that take the TMD normal (calc_normal); other elements take the B/N/C/H branch, as in native. */
const TMD_NORMAL_ELEMENTS = new Set(['Mo', 'W', 'S', 'Se']);

type V3 = number[];

const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** d(a x b)/da as [component][component] (native dpvet1): the derivative of a cross product wrt its first vector. */
const dCrossDa = (b: V3): number[][] => [
  [0, b[2], -b[1]],
  [-b[2], 0, b[0]],
  [b[1], -b[0], 0],
];

/** d(a x b)/db as [component][component] (native dpvet2). */
const dCrossDb = (a: V3): number[][] => [
  [0, -a[2], a[1]],
  [a[2], 0, -a[0]],
  [-a[1], a[0], 0],
];

/**
 * pair_style ilp/tmd (docs.lammps.org/pair_ilp_tmd.html): the ILP for transition-metal dichalcogenide layers. Its
 * normals come from the ring of six same-type in-plane neighbors (the order of the ring is built by ILP_neigh), and the
 * pair terms are those of ilp/graphene/hbn. Usage: pair_style ilp/tmd cutoff [tap_flag] inside pair_style
 * hybrid/overlay. The water normals (Ow with two H neighbors, Hw by its O) are those of native.
 */
export class PairILPTMD extends PairILPGrapheneHBN {
  override readonly name: string = 'ilp/tmd';

  constructor() {
    super();
    this.maxKC = NNEI;
  }

  /** The neighbors of each owned atom in ring order for TMD atoms, in list order otherwise (ILP_neigh). */
  protected override buildKCNeighbors(pc: PairCompute, list: NeighList): void {
    const { x, type } = pc;
    const nt = this.ntypes + 1;
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    for (let i = 0; i < list.inum; i++) {
      const ti = type[i];
      const ring = RING_ELEMENTS.has(this.elements[this.map[ti]]);
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const nb: number[] = [];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq === 0 || !(rsq < this.kcCut2[ti * nt + type[j]]) || mol[owner[i]] !== mol[owner[j]]) continue;
        if (ring && type[j] !== ti) continue;
        nb.push(j);
      }
      let sorted: number[];
      if (ring) {
        sorted = this.ringOrder(nb, pc);
      } else {
        if (nb.length > 3) throw new StyleError('There are too many neighbors for P/B/N/C/H atoms, please check your configuration');
        sorted = nb;
      }
      this.kcN[i] = nb.length;
      for (let c = 0; c < nb.length; c++) this.kcJ[this.maxKC * i + c] = sorted[c];
    }
  }

  /** Ring order of a TMD atom's neighbors: the first neighbor, then each next one bonded to the previous (ILP_neigh). */
  private ringOrder(nb: number[], pc: PairCompute): number[] {
    const n = nb.length;
    if (n > NNEI) throw new StyleError('There are too many neighbors for TMD atoms, please check your configuration');
    const { x, type } = pc;
    const nt = this.ntypes + 1;
    // true when atoms a and b are within the pair cutoff (rsq != 0 and rsq < cutILPsq)
    const near = (a: number, b: number): boolean => {
      const dx = x[3 * a] - x[3 * b], dy = x[3 * a + 1] - x[3 * b + 1], dz = x[3 * a + 2] - x[3 * b + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      return rsq !== 0 && rsq < this.kcCut2[type[a] * nt + type[b]];
    };
    const ns: number[] = new Array<number>(n + 1).fill(-1);
    for (let ll = 0; ll < n; ll++) ns[ll] = nb[ll];
    const done = new Array<boolean>(n).fill(false);
    if (n === NNEI) {
      ns[0] = nb[0];
      done[0] = true;
    } else if (n > 0) {
      // the first neighbor is the one with exactly one neighbor (within the cutoff) among the others
      for (let jj = 0; jj < n; jj++) {
        const j = nb[jj];
        let count = 0;
        for (let ll = 0; ll < n; ll++) {
          const l = nb[ll];
          if (l !== j && near(l, j)) count++;
        }
        if (count === 1) {
          ns[0] = nb[jj];
          done[jj] = true;
          break;
        }
      }
    }
    for (let jj = 0; jj < n; jj++) {
      const j = ns[jj];
      for (let ll = 0; ll < n; ll++) {
        if (done[ll]) continue;
        const l = nb[ll];
        if (near(l, j)) {
          ns[jj + 1] = l;
          done[ll] = true;
          break;
        }
      }
    }
    return ns.slice(0, n);
  }

  /** Normals of the TMD variant (calc_normal): the branch depends on the neighbor count and the element. */
  protected override calcNormals(pc: PairCompute): void {
    const x = pc.x;
    const per = 9 * this.maxKC;
    let tagIndex: Map<number, number> | null = null;
    for (let i = 0; i < this.listInum; i++) {
      const cont = this.kcN[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const vet: V3[] = [];
      for (let c = 0; c < cont; c++) {
        const j = this.kcJ[this.maxKC * i + c];
        vet.push([x[3 * j] - xi, x[3 * j + 1] - yi, x[3 * j + 2] - zi]);
      }
      this.dnormdri.fill(0, 9 * i, 9 * i + 9);
      this.dnormal.fill(0, per * i, per * i + per);
      const element = this.elements[this.map[pc.type[i]]];
      if (cont <= 1) {
        // dangling atom: the normal is along z, with no derivatives; a water H takes its O-H direction
        this.normal[3 * i] = 0;
        this.normal[3 * i + 1] = 0;
        this.normal[3 * i + 2] = 1;
        if (element === 'Hw') {
          let ov: V3 | null = cont === 1 ? vet[0] : null;
          if (!ov) {
            // no O neighbor: the O is the atom with tag - 1 or tag - 2 (native atom->map lookups)
            if (!tagIndex) tagIndex = this.ownedTags(pc);
            const tag = pc.s.id[i];
            ov = this.waterO(pc, tagIndex, tag - 1, tag - 2, xi, yi, zi);
          }
          this.hydrogenNormal(i, ov);
        }
        continue;
      }
      const tmd = TMD_NORMAL_ELEMENTS.has(element);
      if (cont < NNEI) {
        if (element === 'Ow') {
          if (cont !== 2) throw new StyleError('There are too many neighbors for calculating normals of water molecules');
          this.oxygenNormal(i, vet);
        } else if (tmd || cont === 2) {
          this.normalOf(i, vet, false, true);
        } else if (cont === 3) {
          // the central atom's normal derivative is zero for this branch, as in native
          this.normalOf(i, vet, true, false);
        } else {
          throw new StyleError('There are too many neighbors for calculating normals of B/N/C/H atoms');
        }
      } else if (cont === NNEI) {
        this.normalOf(i, vet, true, false);
      } else {
        throw new StyleError('There are too many neighbors for calculating normals of TMD atoms');
      }
    }
  }

  /** Owned atoms by tag (atom->map of native, for owned atoms). */
  private ownedTags(pc: PairCompute): Map<number, number> {
    const m = new Map<number, number>();
    for (let a = 0; a < pc.nlocal; a++) m.set(pc.s.id[a], a);
    return m;
  }

  /** The O-H vector (O minus H) of a water H, from the atom with tag oTag1 or oTag2, which must be an O. */
  private waterO(pc: PairCompute, tags: Map<number, number>, oTag1: number, oTag2: number, xi: number, yi: number, zi: number): V3 {
    const x = pc.x;
    for (const t of [oTag1, oTag2]) {
      const j = tags.get(t);
      if (j !== undefined && this.elements[this.map[pc.type[j]]] === 'Ow') {
        return [x[3 * j] - xi, x[3 * j + 1] - yi, x[3 * j + 2] - zi];
      }
    }
    throw new StyleError('The order of atoms in water molecule should be O H H !');
  }

  /** Normal of a water H (calc_normal, the Hw branch): along its O-H vector; its derivative wrt that O is the minus one. */
  private hydrogenNormal(i: number, ov: V3): void {
    const nn2 = ov[0] * ov[0] + ov[1] * ov[1] + ov[2] * ov[2];
    const nn = Math.sqrt(nn2);
    if (nn === 0) throw new StyleError('The magnitude of the normal vector is zero');
    for (let c = 0; c < 3; c++) this.normal[3 * i + c] = ov[c] / nn;
    // dpvdri = -identity
    const dpvdri = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]];
    const dni = [0, 1, 2].map((ip) => (ov[0] * dpvdri[0][ip] + ov[1] * dpvdri[1][ip] + ov[2] * dpvdri[2][ip]) / nn);
    for (let id = 0; id < 3; id++) {
      for (let ip = 0; ip < 3; ip++) {
        this.dnormdri[9 * i + 3 * id + ip] = dpvdri[id][ip] / nn - ov[id] * dni[ip] / nn2;
      }
    }
    const per = 9 * this.maxKC;
    for (let id = 0; id < 3; id++) {
      for (let ip = 0; ip < 3; ip++) this.dnormal[per * i + 3 * id + ip] = -this.dnormdri[9 * i + 3 * id + ip];
    }
  }

  /** Normal of a water O with its two H neighbors: the normalized mean of the two O-H vectors (calc_normal, Ow). */
  private oxygenNormal(i: number, vet: V3[]): void {
    const cont = 2;
    const Nave: V3 = [0, 1, 2].map((c) => (vet[0][c] + vet[1][c]) / cont);
    const nn2 = Nave[0] * Nave[0] + Nave[1] * Nave[1] + Nave[2] * Nave[2];
    const nn = Math.sqrt(nn2);
    if (nn === 0) throw new StyleError('The magnitude of the normal vector is zero');
    for (let c = 0; c < 3; c++) this.normal[3 * i + c] = Nave[c] / nn;
    // dNave[id][m][ip] = 0.5 for ip === id (each neighbor enters the mean with weight 1/2)
    const dNave: number[][][] = [0, 1, 2].map((id) => [0, 1].map(() => [0, 1, 2].map((ip) => (ip === id ? 0.5 : 0))));
    this.writeDerivatives(i, cont, Nave, nn, dNave, true);
  }

  /**
   * Derivatives of the unit normal of atom i (dnormal wrt its neighbors, dnormdri wrt atom i when withDnormdri) from the
   * non-normalized normal Nave (length nn) and its derivatives dNave[id][m][ip] (native calc_normal, the common tail).
   */
  private writeDerivatives(i: number, cont: number, Nave: V3, nn: number, dNave: number[][][], withDnormdri: boolean): void {
    const nn2 = nn * nn;
    const per = 9 * this.maxKC;
    const dnn: number[][] = Array.from({ length: cont }, () => [0, 0, 0]);
    for (let m = 0; m < cont; m++) {
      for (let id = 0; id < 3; id++) {
        dnn[m][id] = (Nave[0] * dNave[0][m][id] + Nave[1] * dNave[1][m][id] + Nave[2] * dNave[2][m][id]) / nn;
      }
    }
    for (let m = 0; m < cont; m++) {
      for (let id = 0; id < 3; id++) {
        for (let ip = 0; ip < 3; ip++) {
          this.dnormal[per * i + 9 * m + 3 * id + ip] = dNave[id][m][ip] / nn - Nave[id] * dnn[m][ip] / nn2;
        }
      }
    }
    if (!withDnormdri) return;
    // derivative of the summed normal wrt atom i (the neighbors move with it: minus their sum)
    const dpvdri: number[][] = [0, 1, 2].map(() => [0, 0, 0]);
    for (let id = 0; id < 3; id++) {
      for (let ip = 0; ip < 3; ip++) {
        dpvdri[id][ip] = 0;
        for (let k = 0; k < cont; k++) dpvdri[id][ip] -= dNave[id][k][ip];
      }
    }
    const dni = [0, 1, 2].map((ip) => (Nave[0] * dpvdri[0][ip] + Nave[1] * dpvdri[1][ip] + Nave[2] * dpvdri[2][ip]) / nn);
    for (let id = 0; id < 3; id++) {
      for (let ip = 0; ip < 3; ip++) {
        this.dnormdri[9 * i + 3 * id + ip] = dpvdri[id][ip] / nn - Nave[id] * dni[ip] / nn2;
      }
    }
  }

  /**
   * The normal of atom i from the cross products of its consecutive neighbor vectors. A chain (cyclic false) uses the
   * cont - 1 planes of consecutive neighbors; a cycle uses all cont planes, the last one closing the ring. Writes the
   * normal, its derivatives wrt the neighbors (dnormal) and, when withDnormdri, wrt atom i (dnormdri).
   */
  private normalOf(i: number, vet: V3[], cyclic: boolean, withDnormdri: boolean): void {
    const cont = vet.length;
    const P = cyclic ? cont : cont - 1;
    const next = (k: number): number => (cyclic ? (k + 1) % cont : k + 1);
    const prev = (m: number): number => (cyclic ? (m - 1 + cont) % cont : m - 1);
    const pvet: V3[] = [];
    const d1: number[][][] = [];
    const d2: number[][][] = [];
    for (let k = 0; k < P; k++) {
      const a = vet[k], b = vet[next(k)];
      pvet.push(cross(a, b));
      d1.push(dCrossDa(b));
      d2.push(dCrossDb(a));
    }
    const Nave: V3 = [0, 0, 0];
    for (let k = 0; k < P; k++) for (let c = 0; c < 3; c++) Nave[c] += pvet[k][c];
    for (let c = 0; c < 3; c++) Nave[c] /= P;
    const nn2 = Nave[0] * Nave[0] + Nave[1] * Nave[1] + Nave[2] * Nave[2];
    const nn = Math.sqrt(nn2);
    if (nn === 0) throw new StyleError('The magnitude of the normal vector is zero');
    for (let c = 0; c < 3; c++) this.normal[3 * i + c] = Nave[c] / nn;
    // dNave[id][m][ip]: derivative of the id component of the summed normal wrt the ip component of neighbor m
    const dNave: number[][][] = [0, 1, 2].map(() => vet.map(() => [0, 0, 0]));
    for (let id = 0; id < 3; id++) {
      for (let m = 0; m < cont; m++) {
        for (let ip = 0; ip < 3; ip++) {
          let t = 0;
          if (m < P) t += d1[m][id][ip];
          if (cyclic || m >= 1) t += d2[prev(m)][id][ip];
          dNave[id][m][ip] = t / P;
        }
      }
    }
    this.writeDerivatives(i, cont, Nave, nn, dNave, withDnormdri);
  }
}
