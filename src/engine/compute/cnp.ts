import { Compute } from './compute';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import { CellList } from '../cellList';
import type { System } from '../system';

/*
 * compute ID group-ID cnp/atom cutoff — docs.lammps.org/compute_cnp_atom.html (plans/lammps-docs-live/
 * compute_cnp_atom.rst). Quoted sentences are verbatim from that page:
 *
 *   "The value of the CNP parameter will be 0.0 for atoms not in the specified compute group."
 *   "where the index j goes over the :math:`n_i` nearest neighbors of atom i, and the index k goes over
 *    the :math:`n_{ij}` common nearest neighbors between atom i and atom j."
 *   "The CNP calculation is sensitive to the specified cutoff value."
 *
 * Formula (Tsuzuki, Branicio, Rino, Comput Phys Comm 177, 518 (2007)):
 *   Q_i = (1/n_i) sum_j | sum_k (R_ik + R_jk) |^2
 * with R_ik the vector connecting atom k to atom i. The sign convention does not change |sum|^2
 * (both vectors flip together). With u_k = x_k - x_i (k a neighbour of i inside the cutoff) and
 * the neighbour j at u_j: R_ik + R_jk = u_j - 2 u_k, and k runs over the neighbours of i that
 * are also within the cutoff of j (k != j). Only the neighbourhood of owned atom i is needed, all
 * positions taken from the engine's image frame (nb.xall), as in the centro/atom and cna/atom code.
 *
 * Measured with native LAMMPS (black box): the neighbour set is every atom inside the cutoff (no fixed count), and
 * the common neighbours of i and j are the neighbours of i that lie inside the cutoff of j. The per-atom values of
 * tests/oracle/w48cb_cnp_tiny.in (cutoffs 3.0 and 3.6) and w48cb_cnp_fcc.in agree with native to the 12 printed digits.
 */

const EPS = 1e-9;

export class ComputeCnpAtom extends Compute {
  readonly style = 'cnp/atom';
  peratomFlag = true;
  private readonly cut: number;
  private nbr = new Int32Array(0);
  private ux = new Float64Array(0);
  private uy = new Float64Array(0);
  private uz = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`compute ${id} (cnp/atom): usage: compute ID group-ID cnp/atom cutoff`);
    const v = parseNum(args[0], `compute ${id} (cnp/atom) cutoff`);
    if (!(v > 0)) throw new StyleError(`compute ${id} (cnp/atom): cutoff must be > 0 (got ${args[0]})`);
    this.cut = v;
  }

  /** As native LAMMPS does (src/EXTRA-COMPUTE/compute_cnp_atom.cpp ComputeCNPAtom::init at b332f28). */
  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute cnp/atom requires a pair style be defined');
    if (Math.sqrt(this.cut * this.cut) > this.sys.ff.cutforce) throw new StyleError('Compute cnp/atom cutoff is longer than pairwise cutoff');
    if (2 * Math.sqrt(this.cut * this.cut) > this.sys.ff.cutforce + this.sys.nb.skin) {
      this.sys.warn('Compute cnp/atom cutoff may be too large to find ghost atom neighbors');
    }
    if (this.sys.computes.filter((x) => x.style === 'cnp/atom').length > 1) this.sys.warn('More than one compute cnp/atom');
  }

  protected computePeratom(): void {
    const sys = this.sys;
    sys.forces();
    const s = sys.state;
    const nb = sys.nb;
    const cut = this.cut;
    if (nb.cutghost > 0 && nb.cutghost < cut + nb.skin - EPS) {
      throw new StyleError(`compute ${this.id} (cnp/atom): cutoff ${cut} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
    }
    const out = (this.vectorAtom = new Float64Array(s.n));
    const xa = nb.xall;
    const cut2 = cut * cut;
    const nall = nb.nall;
    const cells = new CellList(xa, nall, cut);
    if (this.nbr.length < nall) {
      this.nbr = new Int32Array(nall);
      this.ux = new Float64Array(nall);
      this.uy = new Float64Array(nall);
      this.uz = new Float64Array(nall);
    }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      // neighbours of owned atom i within the cutoff (self excluded), as displacements u_k = x_k - x_i
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      let n = 0;
      // the candidates are the atoms of the cells around i, in ascending index order (as the all-atom loop visits them)
      const cand = cells.candidates(xi, yi, zi);
      for (let t = 0; t < cand.length; t++) {
        const k = cand[t];
        if (k === i) continue;
        const dx = xa[3 * k] - xi, dy = xa[3 * k + 1] - yi, dz = xa[3 * k + 2] - zi;
        if (dx * dx + dy * dy + dz * dz >= cut2) continue;
        this.nbr[n] = k; this.ux[n] = dx; this.uy[n] = dy; this.uz[n] = dz;
        n++;
      }
      if (n === 0) continue;
      let q = 0;
      for (let a = 0; a < n; a++) {
        // sum over common neighbours k of i and j (k != j, |x_k - x_j| inside the cutoff) of u_j - 2 u_k
        let vx = 0, vy = 0, vz = 0;
        const ax = this.ux[a], ay = this.uy[a], az = this.uz[a];
        for (let b = 0; b < n; b++) {
          if (b === a) continue;
          const ex = this.ux[b] - ax, ey = this.uy[b] - ay, ez = this.uz[b] - az;
          if (ex * ex + ey * ey + ez * ez >= cut2) continue;
          vx += ax - 2 * this.ux[b];
          vy += ay - 2 * this.uy[b];
          vz += az - 2 * this.uz[b];
        }
        q += vx * vx + vy * vy + vz * vz;
      }
      out[i] = q / n;
    }
  }
}
