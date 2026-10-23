/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-POD/pair_pod.cpp and src/ML-POD/pair_pod.h (class PairPOD: settings, coeff, init_style,
 * init_one, compute with blockMode 0, lammpsNeighborList, tallyforce, tallystress, copy_data_from_pod_class)
 * at commit b332f28. Model evaluation: eapod.ts (src/ML-POD/eapod.cpp).
 * Authors credited in the C++ file headers: Ngoc Cuong Nguyen (MIT) and Andrew Rohskopf (SNL).
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Summation as native compute(): each owned atom i gives its energy E_i over the neighbours j with
 * 1e-20 < r_ij^2 < rcut^2 (the full neighbour list); forces are tallied on i (+) and j (-).
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { PODModel, parsePODCoeffFile, parsePODParams } from './eapod';

/** pair_style pod (evaluation only), docs.lammps.org/pair_pod.html: pair_coeff * * param coeff elem1 ... elemN. */
export class PairPOD extends Pair {
  readonly name: string = 'pod';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;
  private model: PODModel | null = null;
  /** Element index (0-based, in the order of the pair_coeff names) of each atom type; -1 when unset. */
  private typeElem = new Int32Array(0);
  private cap = 0;
  private rij = new Float64Array(0);
  private fij = new Float64Array(0);
  private tjs = new Int32Array(0);
  private ajs = new Int32Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length > 0) throw new StyleError('Pair style pod accepts no arguments');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.typeElem = new Int32Array(ntypes + 1).fill(-1);
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 5) throw new StyleError('Illegal pair_coeff command: usage pair_coeff * * param_file coeff_file elem1 ... elemN');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style pod must be * *');
    const elems = args.slice(4);
    if (elems.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    if (elems.includes('NULL')) throw new StyleError('pair style pod: NULL atom types are not supported');
    // EAPOD's messages (src/ML-POD/eapod.cpp): the parameter file is named a coefficient file, and neither
    // message has a slot for the system's reason
    const params = parsePODParams(ctx.readFile(args[2], (n) => `Cannot open POD coefficient file ${n}: `));
    const coeffFile = parsePODCoeffFile(ctx.readFile(args[3], (n) => `Cannot open model coefficient file ${n}: `));
    // element indices follow the order of first appearance, as native map_element2type
    const names = [...new Set(elems)];
    if (names.length !== params.species.length) {
      throw new StyleError(`pair style pod: the parameter file has ${params.species.length} species but pair_coeff names ${names.length} elements`);
    }
    this.model = new PODModel(params, coeffFile);
    for (let ti = 1; ti <= this.ntypes; ti++) this.typeElem[ti] = names.indexOf(elems[ti - 1]);
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.model) throw new StyleError('pair style pod needs a pair_coeff command with the POD parameter and coefficient files');
    if (this.shift || this.tail) throw new StyleError('pair style pod does not support pair_modify shift and tail');
  }

  initOne(i: number, j: number): number {
    if (!this.model || this.typeElem[i] < 0 || this.typeElem[j] < 0) throw new StyleError('All pair coeffs are not set');
    return this.model.rcut;
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style pod needs a full neighbor list');
    const model = this.model;
    if (!model) throw new Error('pair style pod: no coefficients');
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nb = list.neighbors;
    const rcutsq = model.rcut * model.rcut;
    let maxn = 1;
    for (let i = 0; i < nlocal; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.rij = new Float64Array(3 * maxn);
      this.fij = new Float64Array(3 * maxn);
      this.tjs = new Int32Array(maxn);
      this.ajs = new Int32Array(maxn);
    }
    const rij = this.rij, fij = this.fij, tjs = this.tjs, ajs = this.ajs;
    for (let i = 0; i < nlocal; i++) {
      const ti = this.typeElem[type[i]] + 1;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let nj = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < rcutsq && rsq > 1e-20) {
          rij[3 * nj] = dx;
          rij[3 * nj + 1] = dy;
          rij[3 * nj + 2] = dz;
          ajs[nj] = j;
          tjs[nj] = this.typeElem[type[j]] + 1;
          nj++;
        }
      }
      const e = model.atomEnergyForce(ti, nj, rij, tjs, fij);
      pc.acc.evdwl += e;
      if (pc.eatom) pc.eatom[i] += e;
      for (let n = 0; n < nj; n++) {
        const j = ajs[n];
        const fx = fij[3 * n], fy = fij[3 * n + 1], fz = fij[3 * n + 2];
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        if (pc.vatom) {
          // native tallystress: v = -r_ij f_ij, half to each atom of the pair
          const v0 = -rij[3 * n] * fx, v1 = -rij[3 * n + 1] * fy, v2 = -rij[3 * n + 2] * fz;
          const v3 = -rij[3 * n] * fy, v4 = -rij[3 * n] * fz, v5 = -rij[3 * n + 1] * fz;
          const va = pc.vatom;
          for (const a of [i, j]) {
            va[6 * a] += 0.5 * v0; va[6 * a + 1] += 0.5 * v1; va[6 * a + 2] += 0.5 * v2;
            va[6 * a + 3] += 0.5 * v3; va[6 * a + 4] += 0.5 * v4; va[6 * a + 5] += 0.5 * v5;
          }
        }
      }
    }
  }
}
