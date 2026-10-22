/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-POD/compute_pod_global.cpp and src/ML-POD/compute_pod_global.h (class ComputePODGlobal, compute pod/global)
 * at commit b332f28, by Ngoc Cuong Nguyen (MIT) and Andrew Rohskopf (SNL) and the LAMMPS developers. Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * compute ID group-ID pod/global pod_file coeff_file elem1 ... elemN (docs.lammps.org/compute_pod_atom.html, the global
 * array of the POD descriptors): the global array has 1 + 3*N rows and nCoeffAll columns, N the number of atoms.
 * Row 0 is the sum over all atoms of the descriptors (the one-body count in the first column of each element block).
 * Row 1 + 3*(id-1) + d is the derivative of that sum with respect to coordinate d of atom id. Each neighbour pair
 * within the cutoff (1e-20 < r_ij^2 < rcut^2, from the full neighbour list) adds the derivative to the rows of i and
 * subtracts it from the rows of j, as native does. Atoms without neighbours contribute only to the one-body column.
 *
 * Only the one-cluster models are supported (environment-adaptive models are refused by the POD model). As native, the
 * group is not used and the array is computed on every evaluation.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NEIGHMASK } from '../../engine/neighbor';
import { PODModel, parsePODCoeffFile, parsePODParams } from './eapod';

export class ComputePODGlobal extends Compute {
  readonly style = 'pod/global';
  private readonly model: PODModel;
  /** 0-based element index of each atom type (index 0 unused). */
  private readonly typeElem: Int32Array;
  private cap = 0;
  private rij = new Float64Array(0);
  private tj = new Int32Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal compute pod/global command');
    this.arrayFlag = true;
    const ntypes = sys.state.ntypes;
    const elems = args.slice(2);
    if (elems.length !== ntypes) {
      throw new StyleError(`Number of element to type mappings does not match number of atom types (${elems.length} given, ${ntypes} types)`);
    }
    if (elems.includes('NULL')) throw new StyleError('compute pod/global: NULL atom types are not supported');
    const params = parsePODParams(sys.readFile(args[0]));
    const coeffFile = parsePODCoeffFile(sys.readFile(args[1]));
    this.model = new PODModel(params, coeffFile);
    // element indices follow the order of first appearance, as native map_element2type
    const names = [...new Set(elems)];
    if (names.length !== params.species.length) {
      throw new StyleError(`compute pod/global: the parameter file has ${params.species.length} species but the element list names ${names.length} elements`);
    }
    this.typeElem = new Int32Array(ntypes + 1);
    for (let t = 1; t <= ntypes; t++) this.typeElem[t] = names.indexOf(elems[t - 1]);
    this.sizeArrayCols = this.model.nCoeffAll;
  }

  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute pod requires a pair style be defined');
  }

  protected computeArray(): void {
    const sys = this.sys, s = sys.state, model = this.model;
    const n = s.n;
    const cols = model.nCoeffAll;
    const rows = 1 + 3 * n;
    this.sizeArrayRows = rows;
    this.sizeArrayCols = cols;
    const out = new Float64Array(rows * cols);
    this.array = out;
    const rcut = model.rcut, rcutsq = rcut * rcut;
    const nb = sys.nb;
    if (nb.cutghost > 0 && nb.cutghost < rcut + nb.skin - 1e-9) {
      throw new StyleError(`compute pod/global cutoff ${rcut} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
    }
    const list = nb.occasionalList(s, rcut, true, s.ntypes);
    const xa = nb.xall, ta = nb.typeall, nbs = list.neighbors;
    let maxn = 1;
    for (let i = 0; i < n; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.rij = new Float64Array(3 * maxn);
      this.tj = new Int32Array(maxn);
    }
    const rij = this.rij, tj = this.tj;
    const nCPE = model.nCoeffPerElement, mdesc = model.Mdesc;
    const idOf = (k: number): number => s.id[nb.owner[k]];
    for (let i = 0; i < n; i++) {
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      const ai = idOf(i) - 1;
      const ti = this.typeElem[ta[i]];
      // one-body descriptor: counted for every atom, including those without neighbours
      out[0 * cols + nCPE * ti] += 1.0;
      let nij = 0;
      const nbrs: number[] = [];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbs[k] & NEIGHMASK;
        const dx = xa[3 * j] - xi, dy = xa[3 * j + 1] - yi, dz = xa[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < rcutsq && rsq > 1e-20) {
          rij[3 * nij] = dx;
          rij[3 * nij + 1] = dy;
          rij[3 * nij + 2] = dz;
          tj[nij] = this.typeElem[ta[j]] + 1;
          nbrs.push(idOf(j) - 1);
          nij++;
        }
      }
      if (nij === 0) continue;
      const { bd, bdd } = model.baseDescriptorsWithDerivatives(rij, tj, nij);
      for (let m = 0; m < mdesc; m++) {
        const kc = nCPE * ti + 1 + m; // the one-body column comes first (nl1 = 1)
        out[kc] += bd[m];
        for (let q = 0; q < nij; q++) {
          const ain = 3 * ai, ajn = 3 * nbrs[q];
          for (let d = 0; d < 3; d++) {
            const v = bdd[3 * (q + nij * m) + d];
            out[(1 + ain + d) * cols + kc] += v;
            out[(1 + ajn + d) * cols + kc] -= v;
          }
        }
      }
    }
  }
}
