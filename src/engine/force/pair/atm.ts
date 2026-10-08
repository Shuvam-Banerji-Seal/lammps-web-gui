import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style atm — docs.lammps.org/pair_atm.html (source: plans/lammps-docs/pair_atm.rst)
 *
 * Syntax (verbatim): "pair_style atm cutoff cutoff_triple"
 *   "cutoff = cutoff for each pair in 3-body interaction (distance units)"
 *   "cutoff_triple = additional cutoff applied to product of 3 pairwise distances (distance units)"
 *
 * Energy (verbatim, the Description's LaTeX):
 *   E & = \nu\frac{1+3\cos\gamma_1\cos\gamma_2\cos\gamma_3}{r_{12}^3r_{23}^3r_{31}^3}
 * with the angles gamma_1, gamma_2, gamma_3 of the doc's diagram (the angles of
 * the triangle at its three vertices).  Measured with native
 * LAMMPS (black box): an equilateral triangle of side 1 with nu = 1 gives
 * 1 + 3 (1/2)^3 = 1.375 over 1, and a 3-4-5 right triangle gives 1/216000 =
 * 4.6296e-6 (both with thermo in metal units; lj units divide pe by N).
 *
 * Triplet rules: "The potential for a triplet of atom is calculated only if all
 * 3 distances" r_12, r_23, r_31 "between the three atoms satisfy" r_IJ < cutoff;
 * "In addition, the product of the 3 distances" r_12 r_23 r_31 < cutoff_triple^3
 * "is required, which excludes from calculation the triplets with small
 * contribution to the interaction."
 * Measured with native LAMMPS (black box): equilateral side d = 1 with
 * cutoff_triple = 0.99 gives 0 and 1.01 gives the full value; a triangle of
 * sides 1.5 with cutoff_triple = 1.4 gives 0 (product 3.375 is not below
 * 1.4^3 = 2.744), while 1.6 and 2.0 both include it (3.375 < 1.6^3 = 4.096).
 * So the product is compared with the cube of cutoff_triple, as written.
 *
 * Coefficients (verbatim): ":math:`K` = atom type of the third atom (1 to :math:`N_{\text{types}}`)"
 * and ":math:`\nu` = prefactor (energy/distance\^9 units)".
 * "LAMMPS sets the coefficients for the other 5 symmetric interactions to the same
 * values." Measured with native LAMMPS (black box): pair_coeff 1 1 2 2.0 gave the
 * same energy for the types (1,1,2), (2,1,1) and the 1-2-2 triplet gave nu = 3.0,
 * i.e. the value is stored for every order of the triplet.
 * "Note that only type triplets with :math:`J \leq K` are considered; if
 * asterisks imply type triplets where :math:`K < J`, they are ignored."
 * Measured with native LAMMPS (black box): with pair_coeff * * * then
 * 2 * * the triplet multiset {1,1,2} kept its value, so an I <= J condition
 * applies to wildcards as well; 1 * 1 did not touch {1,1,2} (K < J ignored).
 * An explicit triplet in any order is accepted: pair_coeff 1 2 1 5.0 set
 * the {1,1,2} value to 5.0 (energy 4.2598659920134 = 5 x 0.85197).
 * "it is required to specify a pair_coeff command for all :math:`I,J`"
 * combinations (the doc). Measured with native LAMMPS (black box): a missing
 * pair stops with All pair coeffs are not set.
 * "Note that a pair_coeff command can override a previous setting for the
 * same :math:`I,J,K` triplet" (doc). Measured: a later 1 1 2 9.0 replaced an earlier 2.0.
 *
 * Measured with native LAMMPS (black box): scaling all distances by 2 divides
 * the energy by 2^9 = 512, as the r^9 units require.
 *
 * "This pair style do not support the pair_modify mix, shift, table, and tail
 * options."
 *
 * Engine structure: the triplet sum is evaluated per owned center i over pairs
 * of its neighbours j, k (full neighbour list). A periodic triplet of atom images
 * is met from each owned member; it is counted only from the member whose
 * (owner, image offset) key is the smallest, so each translation class counts
 * once. Measured with native LAMMPS (black box) on a periodic 32-atom fcc cell
 * (tests/oracle/w10atm_pure.in, step 0): counting every triplet with weight 1/m
 * gives 0.005972483165789 and the smallest-key rule gives 0.003454460294761, the
 * value LAMMPS reports (0.00345446029476096). Forces on ghost members are folded
 * back by the force field.
 *
 * The energy in terms of edge vectors a = x2 - x1, b = x3 - x2, c = x1 - x3:
 * cos gamma_1 cos gamma_2 cos gamma_3 = -(a.b)(b.c)(c.a) / (|a|^2 |b|^2 |c|^2)
 * (each vertex cosine is a dot product over the product of two lengths), so
 *   E = nu [ W^(-3/2) - 3 (a.b)(b.c)(c.a) W^(-5/2) ],   W = |a|^2 |b|^2 |c|^2.
 * The gradient is analytic (atmTriple below); the unit test checks it against
 * finite differences.
 */

/**
 * Energy of one triplet of atoms at positions p = [x1, y1, z1, x2, ..., z3]
 * and the gradient dE/dx_m written to grad (9 values) when given.
 */
export const atmTriple = (p: ArrayLike<number>, nu: number, grad: Float64Array | null): number => {
  const ax = p[3] - p[0], ay = p[4] - p[1], az = p[5] - p[2];
  const bx = p[6] - p[3], by = p[7] - p[4], bz = p[8] - p[5];
  const cx = p[0] - p[6], cy = p[1] - p[7], cz = p[2] - p[8];
  const A = ax * ax + ay * ay + az * az;
  const B = bx * bx + by * by + bz * bz;
  const C = cx * cx + cy * cy + cz * cz;
  const s = ax * bx + ay * by + az * bz; // a.b
  const t = bx * cx + by * cy + bz * cz; // b.c
  const u = cx * ax + cy * ay + cz * az; // c.a
  const W = A * B * C;
  const W32 = W ** -1.5, W52 = W ** -2.5, W72 = W ** -3.5;
  const stu = s * t * u;
  const E = nu * (W32 - 3 * stu * W52);
  if (grad) {
    // dE/dv = nu [ -3 W^-5/2 N v - 3 d_pq (p d_qv + q d_vp) W^-5/2 + 15 stu N v W^-7/2 ]
    // with N = |p|^2 |q|^2 (the two other edges), p, q the other edges, d_pq the dot of p and q.
    // edge vectors v (a, b, c) with the other two edges p, q, d_pq = p.q, d_vp = v.p, d_qv = q.v
    const edgeGrad = (v: number[], pp: number[], qq: number[], dpq: number, dvp: number, dqv: number, N: number): number[] =>
      [0, 1, 2].map((k) => nu * (
        -3 * W52 * N * v[k]
        - 3 * dpq * (pp[k] * dqv + qq[k] * dvp) * W52
        + 15 * stu * N * v[k] * W72
      ));
    const gA = edgeGrad([ax, ay, az], [bx, by, bz], [cx, cy, cz], t, s, u, B * C);
    const gB = edgeGrad([bx, by, bz], [cx, cy, cz], [ax, ay, az], u, t, s, C * A);
    const gC = edgeGrad([cx, cy, cz], [ax, ay, az], [bx, by, bz], s, u, t, A * B);
    for (let k = 0; k < 3; k++) {
      grad[k] = -gA[k] + gC[k];       // dE/dx1 (a = x2 - x1, c = x1 - x3)
      grad[3 + k] = gA[k] - gB[k];    // dE/dx2 (b = x3 - x2)
      grad[6 + k] = gB[k] - gC[k];    // dE/dx3
    }
  }
  return E;
};

/**
 * Order of two atom images: by owner index, then by image offset (x, y, z).
 * The owned center has offset 0. A periodic triplet has one representative
 * per translation class in which its smallest-keyed member is at offset 0;
 * counting only from that member counts each class once (checked against
 * native LAMMPS on a periodic fcc cell, see tests/oracle/w10atm_pure.in).
 */
const keyLess = (oi: number, oj: number, gim: Int32Array, jo: number): boolean => {
  if (oi !== oj) return oi < oj;
  // same owner: the center has offset (0, 0, 0), compare it with the neighbour's offset
  const a = gim[jo], b = gim[jo + 1], c = gim[jo + 2];
  if (a !== 0) return 0 < a;
  if (b !== 0) return 0 < b;
  return 0 < c;
};

/** pair_style atm: three-body Axilrod-Teller-Muto dispersion over triplets within the cutoffs. */
export class PairATM extends Pair {
  readonly name = 'atm';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private cutoff = 0;
  private cutTriple = 0;
  /** nu for every ordered type triplet, index (I*nt + J)*nt + K; all orders of a triplet are set. */
  private nu = new Float64Array(0);
  /** Pair (I,J) with I <= J was covered by a pair_coeff command. */
  private setflag = new Uint8Array(0);

  private cap = 0;
  private gIdx = new Int32Array(0);
  private gDx = new Float64Array(0);
  private gDy = new Float64Array(0);
  private gDz = new Float64Array(0);
  private gR = new Float64Array(0);

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 2) throw new StyleError(`usage: pair_style atm cutoff cutoff_triple (got ${args.length} arguments)`);
    const c = parseNum(args[0], 'cutoff');
    const ct = parseNum(args[1], 'cutoff_triple');
    if (!(c > 0)) throw new StyleError(`pair_style atm: cutoff must be > 0 (got ${c})`);
    if (!(ct > 0)) throw new StyleError(`pair_style atm: cutoff_triple must be > 0 (got ${ct})`);
    this.cutoff = c;
    this.cutTriple = ct;
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.nu = new Float64Array(nt * nt * nt);
    this.setflag = new Uint8Array(nt * nt);
  }

  override coeff(args: string[], _ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 4) throw new StyleError(`usage: pair_coeff I J K nu (style atm; got ${args.length} arguments)`);
    const nt = this.ntypes + 1;
    const nu = parseNum(args[3], 'nu');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const [klo, khi] = typeBounds(args[2], this.ntypes);
    const setPerms = (i: number, j: number, k: number) => {
      const t = [i, j, k];
      for (const [a, b, c] of [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]) {
        this.nu[(t[a] * nt + t[b]) * nt + t[c]] = nu;
      }
    };
    const explicit = !args.slice(0, 3).some((w) => w.includes('*'));
    if (explicit) {
      // an explicit triplet is accepted in any order and stored for all of its orders
      setPerms(ilo, jlo, klo);
      const i = Math.min(ilo, jlo), j = Math.max(ilo, jlo);
      this.setflag[i * nt + j] = 1;
      return;
    }
    // wildcards: only I <= J <= K is enumerated (measured, see the header)
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.setflag[i * nt + j] = 1;
        for (let k = Math.max(klo, j); k <= khi; k++) {
          setPerms(i, j, k);
          count++;
        }
      }
    }
    if (count === 0) throw new StyleError(`pair_coeff: no type triplets with I <= J <= K in ${args.slice(0, 3).join(' ')}`);
  }

  override initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) {
      throw new StyleError('pair_style atm does not support the pair_modify shift and tail options');
    }
    if (this.table !== 12) throw new StyleError('pair_style atm does not support the pair_modify table option');
  }

  override initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const a = Math.min(i, j), b = Math.max(i, j);
    if (!this.setflag[a * nt + b]) {
      throw new StyleError(`all pair coeffs are not set (pair ${a} ${b} has no pair_coeff for pair_style atm)`);
    }
    return this.cutoff;
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type, nb } = pc;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const nu = this.nu;
    const cubeT = this.cutTriple ** 3;
    const eatom = pc.eatom, vatom = pc.vatom;
    let evdwl = 0;

    let maxn = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.gIdx = new Int32Array(maxn);
      this.gDx = new Float64Array(maxn);
      this.gDy = new Float64Array(maxn);
      this.gDz = new Float64Array(maxn);
      this.gR = new Float64Array(maxn);
    }
    const gIdx = this.gIdx, gDx = this.gDx, gDy = this.gDy, gDz = this.gDz, gR = this.gR;
    const p = new Float64Array(9), g = new Float64Array(9);
    const owner = nb.owner, gim = nb.gimage;

    for (let i = 0; i < nlocal; i++) {
      const ti = type[i];
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        const dx = x[3 * j] - x[3 * i];
        const dy = x[3 * j + 1] - x[3 * i + 1];
        const dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti * nt + tj]) continue;
        gIdx[m] = j; gDx[m] = dx; gDy[m] = dy; gDz[m] = dz; gR[m] = Math.sqrt(rsq);
        m++;
      }
      for (let p1 = 0; p1 < m; p1++) {
        const j = gIdx[p1];
        const tj = type[j];
        for (let p2 = p1 + 1; p2 < m; p2++) {
          const k = gIdx[p2];
          const tk = type[k];
          // edge j-k: both positions are images in the same unwrapped frame
          const ex = x[3 * k] - x[3 * j], ey = x[3 * k + 1] - x[3 * j + 1], ez = x[3 * k + 2] - x[3 * j + 2];
          const rjk2 = ex * ex + ey * ey + ez * ez;
          if (rjk2 >= cutsq[tj * nt + tk]) continue;
          const rij = gR[p1], rik = gR[p2], rjk = Math.sqrt(rjk2);
          if (rij * rik * rjk >= cubeT) continue;
          const val = nu[(ti * nt + tj) * nt + tk];
          if (val === 0) continue;
          // each periodic triplet is counted once: from the member with the smallest
          // (owner, image offset) key; the center is an owned atom with offset 0
          if (!(keyLess(i, owner[j], gim, 3 * j) && keyLess(i, owner[k], gim, 3 * k))) continue;
          p[0] = x[3 * i]; p[1] = x[3 * i + 1]; p[2] = x[3 * i + 2];
          p[3] = x[3 * j]; p[4] = x[3 * j + 1]; p[5] = x[3 * j + 2];
          p[6] = x[3 * k]; p[7] = x[3 * k + 1]; p[8] = x[3 * k + 2];
          const E = atmTriple(p, val, g);
          evdwl += E;
          for (let c = 0; c < 3; c++) {
            f[3 * i + c] -= g[c];
            f[3 * j + c] -= g[3 + c];
            f[3 * k + c] -= g[6 + c];
          }
          if (eatom) {
            const e3 = E / 3;
            eatom[i] += e3; eatom[j] += e3; eatom[k] += e3;
          }
          if (vatom) {
            // virial about the center: sum over the two other members of (x_m - x_i) . F_m
            const rjx = p[3] - p[0], rjy = p[4] - p[1], rjz = p[5] - p[2];
            const rkx = p[6] - p[0], rky = p[7] - p[1], rkz = p[8] - p[2];
            const fjx = -g[3], fjy = -g[4], fjz = -g[5];
            const fkx = -g[6], fky = -g[7], fkz = -g[8];
            const vw = [
              rjx * fjx + rkx * fkx, rjy * fjy + rky * fky, rjz * fjz + rkz * fkz,
              rjx * fjy + rkx * fky, rjx * fjz + rkx * fkz, rjy * fjz + rky * fkz,
            ];
            for (let q = 0; q < 6; q++) {
              const vq = vw[q] / 3;
              vatom[6 * i + q] += vq; vatom[6 * j + q] += vq; vatom[6 * k + q] += vq;
            }
          }
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
