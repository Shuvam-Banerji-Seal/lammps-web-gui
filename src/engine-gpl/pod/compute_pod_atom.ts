/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-POD/compute_pod_atom.cpp and src/ML-POD/compute_pod_atom.h (class ComputePODAtom, compute pod/atom)
 * at commit b332f28, by Ngoc Cuong Nguyen (MIT) and Andrew Rohskopf (SNL) and the LAMMPS developers. Copyright
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * compute ID group-ID pod/atom pod_file coeff_file elem1 ... elemN (docs.lammps.org/compute_pod_atom.html): the
 * per-atom POD base descriptors. Row i holds the Mdesc base descriptors of atom i summed over its neighbours j with
 * 1e-20 < r_ij^2 < rcut^2 (the full neighbour list). Atoms with no neighbour keep zeros.
 *
 * Only the one-cluster models are supported (the environment-adaptive models with number_of_environment_clusters
 * > 1 are refused by the POD model, as in pair pod). As native, the group is not used: every atom is computed.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NEIGHMASK } from '../../engine/neighbor';
import { PODModel, parsePODCoeffFile, parsePODParams } from './eapod';

const USAGE = 'usage: compute ID group-ID pod/atom pod_file coeff_file elem1 ... elemN (one element per atom type)';

export class ComputePODAtom extends Compute {
  readonly style = 'pod/atom';
  private readonly model: PODModel;
  /** 0-based element index of each atom type (index 0 unused). */
  private readonly typeElem: Int32Array;
  private cap = 0;
  private rij = new Float64Array(0);
  private tj = new Int32Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError(`Illegal compute pod/atom command (${USAGE})`);
    this.peratomFlag = true;
    const ntypes = sys.state.ntypes;
    const elems = args.slice(2);
    if (elems.length !== ntypes) {
      throw new StyleError(`Number of element to type mappings does not match number of atom types (${elems.length} given, ${ntypes} types)`);
    }
    if (elems.includes('NULL')) throw new StyleError('compute pod/atom: NULL atom types are not supported');
    const params = parsePODParams(sys.readFile(args[0]));
    const coeffFile = parsePODCoeffFile(sys.readFile(args[1]));
    this.model = new PODModel(params, coeffFile);
    // element indices follow the order of first appearance, as native map_element2type
    const names = [...new Set(elems)];
    if (names.length !== params.species.length) {
      throw new StyleError(`compute pod/atom: the parameter file has ${params.species.length} species but the element list names ${names.length} elements`);
    }
    this.typeElem = new Int32Array(ntypes + 1);
    for (let t = 1; t <= ntypes; t++) this.typeElem[t] = names.indexOf(elems[t - 1]);
    this.sizePeratomCols = this.model.Mdesc;
  }

  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute pod requires a pair style be defined');
  }

  protected computePeratom(): void {
    const sys = this.sys, s = sys.state, model = this.model;
    const cols = model.Mdesc;
    const n = s.n;
    const out = new Float64Array(cols * n);
    this.arrayAtom = out;
    const rcut = model.rcut, rcutsq = rcut * rcut;
    const nb = sys.nb;
    if (nb.cutghost > 0 && nb.cutghost < rcut + nb.skin - 1e-9) {
      throw new StyleError(`compute pod/atom cutoff ${rcut} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
    }
    const list = nb.occasionalList(s, rcut, true, s.ntypes);
    const xa = nb.xall, ta = nb.typeall, nbs = list.neighbors;
    // the largest candidate count sizes the work buffers (a minimum of one)
    let maxn = 1;
    for (let i = 0; i < n; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.rij = new Float64Array(3 * maxn);
      this.tj = new Int32Array(maxn);
    }
    const rij = this.rij, tj = this.tj;
    for (let i = 0; i < n; i++) {
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      let nj = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbs[k] & NEIGHMASK;
        const dx = xa[3 * j] - xi, dy = xa[3 * j + 1] - yi, dz = xa[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < rcutsq && rsq > 1e-20) {
          rij[3 * nj] = dx;
          rij[3 * nj + 1] = dy;
          rij[3 * nj + 2] = dz;
          tj[nj] = this.typeElem[ta[j]] + 1;
          nj++;
        }
      }
      if (nj === 0) continue;
      const bd = model.baseDescriptors(rij, tj, nj);
      for (let m = 0; m < cols; m++) out[i * cols + m] = bd[m];
    }
  }
}
