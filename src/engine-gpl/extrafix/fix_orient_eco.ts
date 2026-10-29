/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ORIENT/fix_orient_eco.cpp and fix_orient_eco.h (FixOrientECO: constructor, init, get_reciprocal, get_norm,
 * post_force, compute_scalar) at commit b332f28. Copyright (GNU General Public License): the ORIENT package authors
 * (A. A. Schratt and V. Mohles, the efficient ECO implementation; Ulomek et al. for the method). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Synthetic driving force for grain boundary motion (docs.lammps.org/fix_orient_eco.html): the order parameter chi of
 * each atom from the two grains' reciprocal lattice vectors, the thermal masking, and the force that is the negative
 * gradient of the synthetic energy. Neighbours are the full list within the cutoff, with the images of ghost atoms
 * mapped to their owners (the source's forward communication of the Nbr data).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError, potentialFile } from '../../engine/force/types';
import { numeric } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import type { System } from '../../engine/system';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** Per-owned-atom order parameter data (the source's Nbr struct). */
interface Nbr {
  realPhi: [number[], number[]];
  imagPhi: [number[], number[]];
  duchi: number;
}

const newNbr = (): Nbr => ({
  realPhi: [[0, 0, 0], [0, 0, 0]],
  imagPhi: [[0, 0, 0], [0, 0, 0]],
  duchi: 0,
});

export class FixOrientECO extends Fix {
  readonly style = 'orient/eco';
  /** fix_orient_eco.cpp init: neighbor->add_request(this, NeighConst::REQ_FULL), a perpetual full list (Fix.nativeFullList). */
  nativeFullList = true;
  private readonly u0: number;
  private readonly sign: number;
  private readonly eta: number;
  private readonly rCut: number;
  private readonly dirVec: number[][] = [];
  /** reciprocal_vectors[lambda][k][dim]: lambda 0 = grain A, 1 = grain B. */
  private readonly recip: number[][][] = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]];
  private readonly squaredCutoff: number;
  private readonly invSquaredCutoff: number;
  private readonly halfU: number;
  private readonly invEta: number;
  private normFac = 0;
  private invNormFac = 0;
  private addedEnergy = 0;
  private nbr: Nbr[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4) throw new StyleError('Illegal fix orient/eco command: usage fix ID group orient/eco u0 eta cutoff orientationsFile');
    this.scalarFlag = true;
    this.extscalar = 1;
    this.peratomFlag = true;
    this.sizePeratomCols = 2;
    this.energyGlobal = true;
    // As native LAMMPS does (src/ORIENT/fix_orient_eco.cpp FixOrientECO::FixOrientECO at b332f28): utils::numeric reads
    // u0, eta and the cutoff in turn.
    this.u0 = numeric(args[0]);
    this.sign = this.u0 >= 0.0 ? 1 : -1;
    this.eta = numeric(args[1]);
    this.rCut = numeric(args[2]);
    // the reference orientations: 18 numbers (six primitive lattice vectors, three components each), '#' comments allowed
    const text = sys.readFile(args[3], potentialFile('fix orient/eco'));
    const nums: number[] = [];
    for (const line of text.split('\n')) {
      const body = line.replace(/#.*/, '').trim();
      if (!body) continue;
      for (const w of body.split(/\s+/)) nums.push(Number(w));
    }
    if (nums.length < 18 || nums.slice(0, 18).some((v) => !Number.isFinite(v))) {
      throw new StyleError(`Fix orient/eco file read failed: ${args[3]} must hold 18 numbers (six lattice vectors)`);
    }
    for (let r = 0; r < 6; r++) this.dirVec.push([nums[3 * r], nums[3 * r + 1], nums[3 * r + 2]]);
    this.getReciprocal();
    this.squaredCutoff = this.rCut * this.rCut;
    this.invSquaredCutoff = 1.0 / this.squaredCutoff;
    this.halfU = 0.5 * this.u0;
    this.invEta = 1.0 / this.eta;
  }

  /** get_reciprocal: reciprocal vectors of both grains from their primitive vectors. */
  private getReciprocal(): void {
    const d = this.dirVec;
    const cross = (a: number[], b: number[]): number[] => [
      a[1] * b[2] - b[1] * a[2],
      a[2] * b[0] - b[2] * a[0],
      a[0] * b[1] - b[0] * a[1],
    ];
    for (let g = 0; g < 2; g++) {
      const a = d[3 * g], b = d[3 * g + 1], c = d[3 * g + 2];
      // the source's cell volume, divided by pi (fix_orient_eco.cpp get_reciprocal)
      const vol = 0.5 * (a[0] * (b[1] * c[2] - c[1] * b[2]) + b[0] * (c[1] * a[2] - a[1] * c[2]) + c[0] * (a[1] * b[2] - b[1] * a[2])) / Math.PI;
      const iVol = 1.0 / vol;
      const bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
      for (let k = 0; k < 3; k++) {
        // the source writes the three reciprocal vectors as (b x c, c x a, a x b) / vol, component by component
        this.recip[g][0][k] = bc[k] * iVol;
        this.recip[g][1][k] = ca[k] * iVol;
        this.recip[g][2][k] = ab[k] * iVol;
      }
    }
  }

  /**
   * get_norm: the normalisation of chi from the ideal lattice of grain A (the sum over the lattice points within the
   * cutoff, -4..4 cells). Returns the neighbour count, as the source does for its log line.
   */
  private getNorm(): number {
    const maxCo = 4;
    let wsum = 0.0;
    const reesum = [0, 0, 0], imesum = [0, 0, 0];
    let neigh = 0;
    const d = this.dirVec;
    for (let i0 = -maxCo; i0 <= maxCo; i0++) {
      for (let i1 = -maxCo; i1 <= maxCo; i1++) {
        for (let i2 = -maxCo; i2 <= maxCo; i2++) {
          const delta = [0, 0, 0];
          for (let c = 0; c < 3; c++) delta[c] = d[0][c] * i0 + d[1][c] * i1 + d[2][c] * i2;
          let sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
          if (sq !== 0.0 && sq < this.squaredCutoff) {
            neigh++;
            sq *= this.invSquaredCutoff;
            const weight = sq * (sq - 2.0) + 1.0;
            wsum += weight;
            for (let k = 0; k < 3; k++) {
              const sp = this.recip[1][k][0] * delta[0] + this.recip[1][k][1] * delta[1] + this.recip[1][k][2] * delta[2];
              reesum[k] += weight * libmCos(sp);
              imesum[k] -= weight * libmSin(sp);
            }
          }
        }
      }
    }
    this.normFac = 3.0 * wsum * wsum;
    for (let k = 0; k < 3; k++) this.normFac -= reesum[k] * reesum[k] + imesum[k] * imesum[k];
    return neigh;
  }

  init(): void {
    this.getNorm();
    this.invNormFac = 1.0 / this.normFac;
  }

  /** The force cutoff check runs once the force field has set its cutoffs (the pair's cut table). */
  afterForceInit(): void {
    const cutforce = this.forceCutoff();
    if (this.rCut > cutforce) throw new StyleError('Cutoff radius used by fix orient/eco must be smaller than force cutoff');
  }

  private forceCutoff(): number {
    const pair = this.sys.ff.pair;
    if (!pair) return 0;
    let mx = 0;
    for (let k = 0; k < pair.cut.length; k++) if (pair.cut[k] > mx) mx = pair.cut[k];
    return mx;
  }

  /**
   * post_force: the order parameter chi of every owned atom (all atoms, as the source), the thermal masking value,
   * the synthetic energy (added_energy) and, when u0 != 0, the force on the atoms of the group.
   */
  postForce(): void {
    const s = this.sys.state;
    const nb = this.sys.nb;
    const list = nb.occasionalList(s, this.rCut, true, s.ntypes);
    const xa = nb.xall, owner = nb.owner;
    const f = s.f;
    const n = s.n;
    if (this.arrayAtom.length !== 2 * n) this.arrayAtom = new Float64Array(2 * n);
    if (this.nbr.length !== n) this.nbr = Array.from({ length: n }, newNbr);
    this.addedEnergy = 0.0;
    const omegaPre = (Math.PI / 2) * this.invEta;
    const duchiPre = this.halfU * Math.PI * this.invEta * this.invNormFac;
    const ii0 = list.inum;
    const { recip } = this;
    for (let i = 0; i < ii0; i++) {
      const nbi = this.nbr[i];
      for (let k = 0; k < 3; k++) {
        nbi.realPhi[0][k] = nbi.realPhi[1][k] = 0.0;
        nbi.imagPhi[0][k] = nbi.imagPhi[1][k] = 0.0;
      }
      let chi = 0.0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & NEIGHMASK;
        const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
        let sq = dx * dx + dy * dy + dz * dz;
        if (sq < this.squaredCutoff) {
          sq *= this.invSquaredCutoff;
          const weight = sq * (sq - 2.0) + 1.0;
          for (let lambda = 0; lambda < 2; lambda++) {
            for (let k = 0; k < 3; k++) {
              const sp = recip[lambda][k][0] * dx + recip[lambda][k][1] * dy + recip[lambda][k][2] * dz;
              nbi.realPhi[lambda][k] += weight * libmCos(sp);
              nbi.imagPhi[lambda][k] += weight * libmSin(sp);
            }
          }
        }
      }
      for (let k = 0; k < 3; k++) {
        chi += (nbi.realPhi[0][k] * nbi.realPhi[0][k] + nbi.imagPhi[0][k] * nbi.imagPhi[0][k] -
          nbi.realPhi[1][k] * nbi.realPhi[1][k] - nbi.imagPhi[1][k] * nbi.imagPhi[1][k]);
      }
      chi *= this.invNormFac;
      this.arrayAtom[2 * i] = chi;
      if (chi > this.eta) {
        this.addedEnergy += this.halfU;
        nbi.duchi = 0.0;
        this.arrayAtom[2 * i + 1] = this.sign;
      } else if (chi < -this.eta) {
        this.addedEnergy -= this.halfU;
        nbi.duchi = 0.0;
        this.arrayAtom[2 * i + 1] = -this.sign;
      } else {
        const omega = omegaPre * chi;
        const sinOm = libmSin(omega);
        this.addedEnergy += this.halfU * sinOm;
        nbi.duchi = duchiPre * libmCos(omega);
        this.arrayAtom[2 * i + 1] = this.sign * sinOm;
      }
      // the product with the potential derivative
      for (let k = 0; k < 3; k++) {
        for (let lambda = 0; lambda < 2; lambda++) {
          nbi.realPhi[lambda][k] *= nbi.duchi;
          nbi.imagPhi[lambda][k] *= nbi.duchi;
        }
      }
    }
    if (this.u0 === 0.0) return;
    // the force: the gradient of the synthetic energy, for the atoms of the group
    const bit = this.groupBit;
    for (let ii = 0; ii < ii0; ii++) {
      const i = ii;
      if (!(s.mask[i] & bit)) continue;
      const nbi = this.nbr[i];
      const noBoundaryAtom = nbi.duchi === 0.0;
      const giiCos = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]];
      const giiSin = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]];
      const gijVec = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]];
      const gijSca = [[0, 0, 0], [0, 0, 0]];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & NEIGHMASK;
        const nbj = this.nbr[owner[j]];
        // do not compute the force on atom i if it is far from the boundary
        if (nbj.duchi === 0.0 && noBoundaryAtom) continue;
        const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
        let sq = dx * dx + dy * dy + dz * dz;
        if (sq < this.squaredCutoff) {
          sq *= this.invSquaredCutoff;
          const weight = sq * (sq - 2.0) + 1.0;
          const wgp = 4.0 * (sq - 1.0) * this.invSquaredCutoff;
          const wg = [wgp * dx, wgp * dy, wgp * dz];
          for (let lambda = 0; lambda < 2; lambda++) {
            for (let k = 0; k < 3; k++) {
              const sp = recip[lambda][k][0] * dx + recip[lambda][k][1] * dy + recip[lambda][k][2] * dz;
              const cosSp = libmCos(sp), sinSp = libmSin(sp);
              for (let dim = 0; dim < 3; dim++) {
                giiCos[lambda][k][dim] += wg[dim] * cosSp;
                giiSin[lambda][k][dim] += wg[dim] * sinSp;
                gijVec[lambda][k][dim] += nbj.realPhi[lambda][k] * (wg[dim] * cosSp) - nbj.imagPhi[lambda][k] * (wg[dim] * sinSp);
              }
              gijSca[lambda][k] += weight * (nbj.realPhi[lambda][k] * sinSp + nbj.imagPhi[lambda][k] * cosSp);
            }
          }
        }
      }
      for (let k = 0; k < 3; k++) {
        for (let dim = 0; dim < 3; dim++) {
          f[3 * i + dim] -= nbi.realPhi[0][k] * giiCos[0][k][dim] + nbi.imagPhi[0][k] * giiSin[0][k][dim] + gijVec[0][k][dim] + recip[1][k][dim] * gijSca[1][k];
          f[3 * i + dim] += nbi.realPhi[1][k] * giiCos[1][k][dim] + nbi.imagPhi[1][k] * giiSin[1][k][dim] + gijVec[1][k][dim] + recip[0][k][dim] * gijSca[0][k];
        }
      }
    }
  }

  minPostForce(): void { this.postForce(); }

  /** compute_scalar: the synthetic energy of the last evaluation (fix_modify energy yes adds it to pe). */
  computeScalar(): number { return this.addedEnergy; }
  energy(): number { return this.addedEnergy; }
}
