import { Compute } from './compute';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';

/*
 * compute ID group-ID born/matrix keyword value ...
 * — docs.lammps.org/compute_born_matrix.html (plans/lammps-docs/compute_born_matrix.rst).
 * Quoted sentences are copied character for character from that page.
 *
 *   The page defines a compute for d^2 U / d eps_i d eps_j: "the second
 *    derivatives of the potential energy" with respect to the strain tensor
 *    elements.
 *   "The Born term is a symmetric 6x6 matrix ... whose 21 independent elements
 *    are output in this order:" C11 C22 C33 C44 C55 C66 C12 C13 C14 C15 C16
 *    C23 C24 C25 C26 C34 C35 C36 C45 C46 C56.
 *   "This compute calculates a global vector with 21 values that are the second
 *    derivatives of the potential energy with respect to strain.  The values
 *    are in energy units." The page calls the values extensive. Measured with native LAMMPS (black box): thermo does NOT
 *    normalize the values by the atom count, so the engine reports them as
 *    intensive (extvector 0) to reproduce native's printed numbers.
 *
 * The six Voigt components are ordered xx, yy, zz, yz, xz, xy (measured with
 * native LAMMPS, black box: a single pair along x gives C9 = (1,4) = xx-yz and
 * C11 = (1,6) = xx-xy). For a pair with separation r and direction cosines
 * n_a = s_a / r the Born contribution is (u'' - u'/r) n_a n_b n_c n_d, summed
 * over every pair within the force cutoff (in energy units, because the page's
 * C^B = (1/V) d^2 U / d eps d eps and the output is V * C^B).
 *
 * Two mutually exclusive methods:
 *   analytic — "a direct computation from the analytical formula from the
 *   different terms of the potential". Implemented for pair style lj/cut:
 *   u'' - u'/r = 4 eps [ 168 sigma^12 / r^14 - 48 sigma^6 / r^8 ].
 *   numdiff — "compute 1 all born/matrix numdiff 1.0e-4 myvirial": "delta gives
 *   the size of the applied strains. virial-ID gives the ID string of the
 *   pressure compute that provides the virial stress tensor, requiring that it
 *   use the virial keyword". "The difference in these two virials divided by
 *   two times delta, approximates the corresponding components of the second
 *   derivative, after applying a suitable unit conversion." "The *numdiff*
 *   option cannot be used with any other keyword."
 *   Measured with native LAMMPS (black box): the numdiff output equals the
 *   analytic output for lj/cut to 1e-9, so the engine reproduces the same Born
 *   sum by finite-differencing each pair's radial virial coefficient f(r) =
 *   fforce * r = -u'(r): u'' - u'/r = -r * d f / d r, evaluated by a central
 *   difference. The virial-ID is parsed and required to name a pressure compute
 *   (as the page demands), and the pair sum then uses that compute's group.
 *
 * Any other pair style, or any bonded term, throws a StyleError naming it: the
 * browser engine has no per-style born_matrix method.
 */

/** The six Voigt components in the field order used for the pair sum. */
const FIELDS: readonly (readonly [number, number])[] = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]];
/** Maps the pair-sum field order to native's Voigt order xx yy zz yz xz xy. */
const PERM = [0, 1, 2, 5, 4, 3];
/** The 21 output positions as (row, col) in Voigt order xx yy zz yz xz xy. */
const OUT: readonly (readonly [number, number])[] = [
  [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [1, 2], [1, 3], [1, 4], [1, 5],
  [2, 3], [2, 4], [2, 5],
  [3, 4], [3, 5],
  [4, 5],
];

export class ComputeBornMatrix extends Compute {
  readonly style = 'born/matrix';
  private readonly useNumdiff: boolean;
  private readonly delta: number;
  private readonly virialId: string;
  private readonly M = Array.from({ length: 6 }, () => new Float64Array(6));

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.vectorFlag = true;
    this.sizeVector = 21;
    this.extvector = 0;
    this.vector = new Float64Array(21);
    if (args.length && args[0] === 'numdiff') {
      if (args.length !== 3) {
        throw new StyleError(`compute ${id} born/matrix: numdiff needs exactly two arguments, delta and virial-ID`);
      }
      this.useNumdiff = true;
      this.delta = parseNum(args[1], `compute ${id} born/matrix delta`);
      if (!(this.delta > 0)) throw new StyleError(`compute ${id} born/matrix: delta must be positive, got '${args[1]}'`);
      this.virialId = args[2];
    } else {
      this.useNumdiff = false;
      this.delta = 0;
      this.virialId = '';
      for (const w of args) {
        if (w !== 'pair' && w !== 'bond' && w !== 'angle' && w !== 'dihedral' && w !== 'improper') {
          throw new StyleError(`compute ${id} born/matrix: unknown keyword '${w}'`);
        }
        if (w !== 'pair') {
          throw new StyleError(`compute ${id} born/matrix: the '${w}' term is not supported by the browser engine (only pair, or the numdiff option)`);
        }
      }
    }
    if (sys.dimension === 2) throw new StyleError(`compute ${id} born/matrix: 2d simulations are not supported by the browser engine`);
  }

  init(): void {
    if (this.useNumdiff) {
      const c = this.sys.compute(this.virialId);
      if (!c.pressFlag) throw new StyleError(`compute ${this.id} born/matrix: compute ${this.virialId} is not a pressure compute`);
    } else {
      const ff = this.sys.ff;
      for (const [term, style] of [['bond', ff.bond], ['angle', ff.angle], ['dihedral', ff.dihedral], ['improper', ff.improper]] as const) {
        if (style) throw new StyleError(`compute ${this.id} born/matrix: the analytic ${term} term is not supported by the browser engine (use numdiff)`);
      }
    }
  }

  protected computeVector(): void {
    this.vector.fill(0);
    const sys = this.sys;
    const pair = sys.ff.pair;
    if (!pair) throw new StyleError(`compute ${this.id} born/matrix: no pair_style is defined`);
    if (!this.useNumdiff && pair.name !== 'lj/cut') {
      throw new StyleError(`compute ${this.id} born/matrix: the analytic Born matrix is only implemented for pair style lj/cut (got '${pair.name}'; use numdiff)`);
    }
    const eps = this.useNumdiff ? null : pair.extract('epsilon');
    const sig = this.useNumdiff ? null : pair.extract('sigma');
    if (!this.useNumdiff && (!(eps instanceof Float64Array) || !(sig instanceof Float64Array))) {
      throw new StyleError(`compute ${this.id} born/matrix: pair style lj/cut did not provide epsilon/sigma coefficients`);
    }
    if (!pair.single) throw new StyleError(`compute ${this.id} born/matrix: pair style '${pair.name}' does not support the engine's pairwise evaluation`);

    sys.forces();
    const s = sys.state;
    const nb = sys.nb;
    const list = nb.half;
    if (!list) throw new StyleError(`compute ${this.id} born/matrix: no half neighbour list is available`);
    const sLJ = (sys.ff as unknown as { specialLJ: Float64Array }).specialLJ;
    const nt = pair.ntypes + 1;
    const nbrs = list.neighbors;
    const xa = nb.xall;
    const NEIGH_MASK = (1 << 30) - 1;
    for (const row of this.M) row.fill(0);
    const comp = [0, 0, 0];
    for (let i = 0; i < list.inum; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nbrs[k];
        const j = jj & NEIGH_MASK;
        const owner = nb.owner[j];
        if (!(s.mask[owner] & this.groupBit)) continue;
        const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const ti = nb.typeall[i], tj = nb.typeall[j];
        if (rsq >= pair.cutsq[ti * nt + tj]) continue;
        const sb = jj >>> 30;
        const factor = sb === 0 ? 1 : sLJ[sb];
        const r = Math.sqrt(rsq);
        const t = ti * nt + tj;
        let coef: number;
        if (this.useNumdiff) {
          // u'' - u'/r = -r d(fforce)/dr, central difference of the radial virial coefficient.
          const h = this.delta * r;
          const g = (rr: number) => pair.single!(i, j, ti, tj, rr * rr, factor, factor, nb.qall[i], nb.qall[j]).fforce;
          coef = -r * (g(r + h) - g(r - h)) / (2 * h);
        } else {
          const e = (eps as Float64Array)[t], sg = (sig as Float64Array)[t];
          const q = (sg ** 6) / rsq ** 3;
          coef = factor * 4 * e * (168 * q * q - 48 * q) / rsq;
        }
        comp[0] = dx; comp[1] = dy; comp[2] = dz;
        for (let a = 0; a < 6; a++) {
          const [ai, bi] = FIELDS[a];
          const p = comp[ai] * comp[bi];
          const Ma = this.M[a];
          for (let b = 0; b < 6; b++) {
            const [gi, di] = FIELDS[b];
            Ma[b] += coef * p * comp[gi] * comp[di] / rsq;
          }
        }
      }
    }
    OUT.forEach(([i, j], n) => { this.vector[n] = this.M[PERM[i]][PERM[j]]; });
  }
}
