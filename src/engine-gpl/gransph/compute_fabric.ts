/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/GRANULAR/compute_fabric.cpp and
 * compute_fabric.h (ComputeFabric: compute fabric). Contributing authors (from the source): Joel Clemmer (SNL), Ishan
 * Srivastava (LBNL). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_fabric.html: "compute ID group-ID fabric cutoff_style args ... keyword value". The cutoff style is
 * "type" (the pair cutoff of the types) or "radius" (the sum of the radii); the tensors are "contact" (the fabric tensor of
 * the contact normals), "branch" (the branch vector tensor), "force/normal" and "force/tangential" (the contact forces
 * projected on the normals, from the pair style's single(); the tangential one needs a granular pair style, and its
 * svector[3] value). The keyword "type/include" restricts the pairs to the listed type pairs. The vector holds 6 values per
 * tensor, in the order given; the scalar is the number of contacts.
 *
 * Native quirks kept: the tangential tensor is normalised with the trace of the normal-force tensor (compute_fabric.cpp
 * uses trace_third_Xfn with the factor 9 for the tangential tensor). The contact count and the tensors are summed over the
 * half neighbour list; a ghost partner contributes to the sums through its owner's group membership, and its position and
 * radius are its image's, as the source reads them from the ghost arrays. The force/normal and force/tangential tensors
 * are refused for a periodic-image contact: native single() reads the velocity of a ghost as it was at the last forward
 * communication (the force evaluation), a value this engine does not keep.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError, typeBounds, type PairSep } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import type { System } from '../../engine/system';

const CN = 0, BR = 1, FN = 2, FT = 3;

export class ComputeFabric extends Compute {
  readonly style = 'fabric';
  private readonly cutRadius: boolean;
  /** The tensors in the order of the keywords; the vector holds 6 values per tensor. */
  private readonly tensors: number[] = [];
  private readonly cnFlag: boolean;
  private readonly brFlag: boolean;
  private readonly fnFlag: boolean;
  private readonly ftFlag: boolean;
  /** type_filter[i * nt + j] = 1 for the type pairs included (null: all). */
  private typeFilter: Uint8Array | null = null;
  private nc = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal compute fabric command');
    if (args[0] === 'type') this.cutRadius = false;
    else if (args[0] === 'radius') this.cutRadius = true;
    else throw new StyleError('Illegal compute fabric command');
    if (this.cutRadius && !sys.state.radius) throw new StyleError('Compute fabric radius style requires atom attribute radius');
    const nt = sys.state.ntypes + 1;
    for (let k = 1; k < args.length; k++) {
      const w = args[k];
      if (w === 'contact') this.tensors.push(CN);
      else if (w === 'branch') this.tensors.push(BR);
      else if (w === 'force/normal') this.tensors.push(FN);
      else if (w === 'force/tangential') this.tensors.push(FT);
      else if (w === 'type/include') {
        if (k + 2 >= args.length) throw new StyleError('Invalid keyword in compute fabric command');
        this.typeFilter ??= new Uint8Array(nt * nt);
        const ntypes = sys.state.ntypes;
        const iwords = args[k + 1].split(','), jwords = args[k + 2].split(',');
        for (const ifield of iwords) {
          const [inlo, inhi] = typeBounds(ifield, ntypes);
          for (const jfield of jwords) {
            const [jnlo, jnhi] = typeBounds(jfield, ntypes);
            for (let it = inlo; it <= inhi; it++) {
              for (let jt = jnlo; jt <= jnhi; jt++) {
                this.typeFilter[it * nt + jt] = 1;
                this.typeFilter[jt * nt + it] = 1;
              }
            }
          }
        }
        k += 2;
      } else {
        throw new StyleError('Illegal compute fabric command');
      }
    }
    this.cnFlag = this.tensors.includes(CN);
    this.brFlag = this.tensors.includes(BR);
    this.fnFlag = this.tensors.includes(FN);
    this.ftFlag = this.tensors.includes(FT);
    this.vectorFlag = true;
    this.sizeVector = this.tensors.length * 6;
    this.extvector = 0;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.vector = new Float64Array(this.sizeVector);
  }

  init(): void {
    const pair = this.sys.ff.pair;
    if (!pair) throw new StyleError('No pair style is defined for compute fabric');
    if (!pair.single && (this.fnFlag || this.ftFlag)) {
      throw new StyleError('Pair style does not support compute fabric normal or tangential force');
    }
    const granular = pair.name.startsWith('granular') || pair.name.startsWith('gran/');
    if (!granular && this.ftFlag) throw new StyleError('Pair style does not calculate tangential forces for compute fabric');
  }

  /** Whether the pair (i, j) of the half list counts: group membership, type filter and the cutoff style. */
  private counts(oi: number, oj: number, ti: number, tj: number, rsq: number): boolean {
    const s = this.sys.state;
    if (!(s.mask[oj] & this.groupBit)) return false;
    if (this.typeFilter && this.typeFilter[ti * (s.ntypes + 1) + tj] === 0) return false;
    if (this.cutRadius) {
      const radsum = s.radius![oi] + s.radius![oj];
      return !(rsq >= radsum * radsum);
    }
    return !(rsq >= this.sys.ff.pair!.cutsq[ti * (s.ntypes + 1) + tj]);
  }

  /** compute_scalar: the number of contacts (pairs of the half list with both atoms in the group that touch). */
  private countContacts(): number {
    this.sys.forces();
    const s = this.sys.state, nb = this.sys.nb, list = nb.half;
    if (!list) throw new StyleError('compute fabric: no half neighbour list is available');
    const nbr = list.neighbors, x = nb.xall, owner = nb.owner, typeAll = nb.typeall;
    let nc = 0;
    for (let i = 0; i < list.inum; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbr[k] & NEIGHMASK;
        const oj = owner[j];
        const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (this.counts(i, oj, typeAll[i], typeAll[j], rsq)) nc += 1.0;
      }
    }
    return nc;
  }

  protected computeScalar(): number {
    this.nc = this.countContacts();
    return this.nc;
  }

  protected computeVector(): void {
    const sys = this.sys;
    sys.forces();
    const s = sys.state, nb = sys.nb, list = nb.half;
    if (!list) throw new StyleError('compute fabric: no half neighbour list is available');
    const pair = sys.ff.pair!;
    const nc = this.countContacts();
    this.nc = nc;
    const vec = new Float64Array(this.sizeVector);
    if (nc === 0) {
      this.vector = vec;
      return;
    }
    const ncinv = 1.0 / nc;
    const nbr = list.neighbors, x = nb.xall, owner = nb.owner, typeAll = nb.typeall;
    const phi = new Float64Array(6);
    // Loop over the contact pairs: the callback gets the normal n = d / r, the pair distance r and the single() outputs.
    const each = (cb: (nx: number, ny: number, nz: number, r: number, fpair: number, ft: number) => void, needForce: boolean): void => {
      for (let i = 0; i < list.inum; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
        for (let k = k0; k < k1; k++) {
          const j = nbr[k] & NEIGHMASK;
          const oj = owner[j];
          const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
          const rsq = delx * delx + dely * dely + delz * delz;
          if (!this.counts(i, oj, typeAll[i], typeAll[j], rsq)) continue;
          let fpair = 0, ft = 0;
          if (needForce) {
            // native single() reads the velocity of a ghost partner as the ghost held it at the last forward
            // communication (the force evaluation), which this engine does not keep: a periodic-image partner is refused
            if (oj !== j) throw new StyleError('compute fabric force/normal and force/tangential with a periodic-image contact are not supported by the browser engine (native reads the ghost velocity of the last force evaluation)');
            const out = pair.single!(i, oj, typeAll[i], typeAll[oj], rsq, 1.0, 1.0, 0, 0, { dx: delx, dy: dely, dz: delz } as PairSep);
            fpair = out.fforce;
            ft = out.svector ? out.svector[3] : 0;
          }
          const r = Math.sqrt(rsq);
          const rinv = 1.0 / r;
          cb(delx * rinv, dely * rinv, delz * rinv, r, fpair, ft);
        }
      }
    };

    // first loop: the contact fabric tensor phi
    each((nx, ny, nz) => {
      phi[0] += nx * nx;
      phi[1] += ny * ny;
      phi[2] += nz * nz;
      phi[3] += nx * ny;
      phi[4] += nx * nz;
      phi[5] += ny * nz;
    }, false);
    for (let c = 0; c < 6; c++) phi[c] = phi[c] * ncinv;
    const trace_third_phi = (1.0 / 3.0) * (phi[0] + phi[1] + phi[2]);
    const Ac = new Float64Array(6);
    Ac[0] = (15.0 / 2.0) * (phi[0] - trace_third_phi);
    Ac[1] = (15.0 / 2.0) * (phi[1] - trace_third_phi);
    Ac[2] = (15.0 / 2.0) * (phi[2] - trace_third_phi);
    Ac[3] = (15.0 / 2.0) * (phi[3]);
    Ac[4] = (15.0 / 2.0) * (phi[4]);
    Ac[5] = (15.0 / 2.0) * (phi[5]);

    const D = new Float64Array(6), Xfn = new Float64Array(6), Xft = new Float64Array(6);
    if (this.brFlag || this.fnFlag || this.ftFlag) {
      each((nx, ny, nz, r, fpair, ft) => {
        let denom = 1 + Ac[0] * nx * nx + Ac[1] * ny * ny + Ac[2] * nz * nz;
        denom += 2 * Ac[3] * nx * ny + 2 * Ac[4] * nx * nz + 2 * Ac[5] * ny * nz;
        const prefactor = ncinv / denom;
        if (this.brFlag) {
          D[0] += prefactor * nx * nx * r;
          D[1] += prefactor * ny * ny * r;
          D[2] += prefactor * nz * nz * r;
          D[3] += prefactor * nx * ny * r;
          D[4] += prefactor * nx * nz * r;
          D[5] += prefactor * ny * nz * r;
        }
        if (this.fnFlag || this.ftFlag) {
          const fn = r * fpair;
          Xfn[0] += prefactor * nx * nx * fn;
          Xfn[1] += prefactor * ny * ny * fn;
          Xfn[2] += prefactor * nz * nz * fn;
          Xfn[3] += prefactor * nx * ny * fn;
          Xfn[4] += prefactor * nx * nz * fn;
          Xfn[5] += prefactor * ny * nz * fn;
          if (this.ftFlag) {
            Xft[0] += prefactor * nx * nx * ft;
            Xft[1] += prefactor * ny * ny * ft;
            Xft[2] += prefactor * nz * nz * ft;
            Xft[3] += prefactor * nx * ny * ft;
            Xft[4] += prefactor * nx * nz * ft;
            Xft[5] += prefactor * ny * nz * ft;
          }
        }
      }, this.fnFlag || this.ftFlag);
    }

    const tensorOut = (style: number, t: Float64Array): void => {
      for (let i = 0; i < this.tensors.length; i++) {
        if (this.tensors[i] === style) for (let j = 0; j < 6; j++) vec[6 * i + j] = t[j];
      }
    };
    if (this.cnFlag) tensorOut(CN, Ac);
    if (this.brFlag) {
      const tr = (1.0 / 3.0) * (D[0] + D[1] + D[2]);
      const f = (15.0 / (6.0 * tr));
      const br = new Float64Array(6);
      br[0] = f * (D[0] - tr);
      br[1] = f * (D[1] - tr);
      br[2] = f * (D[2] - tr);
      br[3] = f * (D[3]);
      br[4] = f * (D[4]);
      br[5] = f * (D[5]);
      tensorOut(BR, br);
    }
    let trXfn = 0;
    if (this.fnFlag || this.ftFlag) trXfn = (1.0 / 3.0) * (Xfn[0] + Xfn[1] + Xfn[2]);
    if (this.fnFlag) {
      const f = (15.0 / (6.0 * trXfn));
      const fnT = new Float64Array(6);
      fnT[0] = f * (Xfn[0] - trXfn);
      fnT[1] = f * (Xfn[1] - trXfn);
      fnT[2] = f * (Xfn[2] - trXfn);
      fnT[3] = f * (Xfn[3]);
      fnT[4] = f * (Xfn[4]);
      fnT[5] = f * (Xfn[5]);
      tensorOut(FN, fnT);
    }
    if (this.ftFlag) {
      const trXft = (1.0 / 3.0) * (Xft[0] + Xft[1] + Xft[2]);
      const f = (15.0 / (9.0 * trXfn));
      const ftT = new Float64Array(6);
      ftT[0] = f * (Xft[0] - trXft);
      ftT[1] = f * (Xft[1] - trXft);
      ftT[2] = f * (Xft[2] - trXft);
      ftT[3] = f * (Xft[3]);
      ftT[4] = f * (Xft[4]);
      ftT[5] = f * (Xft[5]);
      tensorOut(FT, ftT);
    }
    this.vector = vec;
  }
}
