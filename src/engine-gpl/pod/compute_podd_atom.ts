/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-POD/compute_podd_atom.cpp and src/ML-POD/compute_podd_atom.h (class ComputePODDAtom, compute podd/atom)
 * at commit b332f28, by Ngoc Cuong Nguyen (MIT) and Andrew Rohskopf (SNL) and the LAMMPS developers. Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * compute ID group-ID podd/atom pod_file coeff_file elem1 ... elemN (docs.lammps.org/compute_podd_atom.html): the
 * per-atom derivatives of the POD base descriptors. Row i (local atom i) holds 3*N*Mdesc columns, N the number of atoms
 * in the system: column m + Mdesc*(d + 3*a) is the derivative of the descriptor m summed over the neighbour pairs of i
 * with respect to coordinate d of atom a (a = atom id - 1). Each neighbour j (with 1e-20 < r_ij^2 < rcut^2, from the
 * full neighbour list) adds +dD/dr_ij to the column block of atom i and -dD/dr_ij to that of atom j, as native does.
 *
 * Only the one-cluster models with no four-body descriptors are ported (the environment-adaptive models are refused by
 * the POD model; four-body descriptors are refused when the compute is defined). As native, the group is not used.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NEIGHMASK } from '../../engine/neighbor';
import { PODModel, parsePODCoeffFile, parsePODParams } from './eapod';

export class ComputePODDAtom extends Compute {
  readonly style = 'podd/atom';
  private readonly model: PODModel;
  /** 0-based element index of each atom type (index 0 unused). */
  private readonly typeElem: Int32Array;
  private cap = 0;
  private rij = new Float64Array(0);
  private tj = new Int32Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal compute podd/atom command');
    this.peratomFlag = true;
    const ntypes = sys.state.ntypes;
    const elems = args.slice(2);
    if (elems.length !== ntypes) {
      throw new StyleError(`Number of element to type mappings does not match number of atom types (${elems.length} given, ${ntypes} types)`);
    }
    if (elems.includes('NULL')) throw new StyleError('compute podd/atom: NULL atom types are not supported');
    const params = parsePODParams(sys.readFile(args[0]));
    const coeffFile = parsePODCoeffFile(sys.readFile(args[1]));
    this.model = new PODModel(params, coeffFile);
    if (this.model.nl4 > 0) throw new StyleError('compute podd/atom: four-body POD descriptors (fourbody_number_radial_basis_functions > 0) are not ported');
    // element indices follow the order of first appearance, as native map_element2type
    const names = [...new Set(elems)];
    if (names.length !== params.species.length) {
      throw new StyleError(`compute podd/atom: the parameter file has ${params.species.length} species but the element list names ${names.length} elements`);
    }
    this.typeElem = new Int32Array(ntypes + 1);
    for (let t = 1; t <= ntypes; t++) this.typeElem[t] = names.indexOf(elems[t - 1]);
    this.sizePeratomCols = 3 * sys.state.n * this.model.Mdesc;
  }

  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute pod requires a pair style be defined');
  }

  protected computePeratom(): void {
    const sys = this.sys, s = sys.state, model = this.model;
    const n = s.n;
    const mdesc = model.Mdesc;
    const cols = 3 * n * mdesc;
    this.sizePeratomCols = cols;
    const out = new Float64Array(cols * n);
    this.arrayAtom = out;
    const rcut = model.rcut, rcutsq = rcut * rcut;
    const nb = sys.nb;
    if (nb.cutghost > 0 && nb.cutghost < rcut + nb.skin - 1e-9) {
      throw new StyleError(`compute podd/atom cutoff ${rcut} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
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
    // atom ids of the owned atoms and of the ghosts (through their owners)
    const idOf = (k: number): number => s.id[nb.owner[k]];
    for (let i = 0; i < n; i++) {
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      const ai = idOf(i) - 1;
      let nj = 0;
      const nbrs: number[] = [];
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
          nbrs.push(idOf(j) - 1);
          nj++;
        }
      }
      if (nj === 0) continue;
      const { bdd } = model.baseDescriptorsWithDerivatives(rij, tj, nj);
      const row = i * cols;
      for (let q = 0; q < nj; q++) {
        const ain = 3 * ai, ajn = 3 * nbrs[q];
        for (let m = 0; m < mdesc; m++) {
          for (let d = 0; d < 3; d++) {
            const v = bdd[3 * (q + nj * m) + d];
            out[row + m + mdesc * (d + ain)] += v;
            out[row + m + mdesc * (d + ajn)] -= v;
          }
        }
      }
    }
  }
}
