import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * fix ID group-ID balance Nfreq thresh style args keyword args ... —
 * docs.lammps.org/fix_balance.html:
 *   * Nfreq = perform dynamic load balancing every this many steps
 *   * thresh = imbalance threshold that must be exceeded to perform a re-balance
 *   * style = *shift* or *rcb* or *report*
 * "The imbalance factor is defined as the maximum number of particles (or
 * weight) owned by any processor, divided by the average number of particles
 * (or weight) per processor.  Thus, an imbalance factor of 1.0 is perfect
 * balance." "Note that re-balances can be forced even if the current
 * balance is perfect (1.0) be specifying a *thresh* < 1.0."
 * "This fix computes a global scalar which is the imbalance factor after the
 * most recent re-balance and a global vector of length 3".
 *
 * The browser engine is one process, so the balance is always perfect and
 * nothing moves. Measured with native LAMMPS on one process (black box): the
 * scalar is 0 and the vector [W, 0, 0] when thresh >= 1 (no re-balance), and
 * 1 and [W, iterations, 1] when thresh < 1 forces one, where W is the total
 * (weighted) particle count and iterations is the number of dimensions in a
 * shift dimstr (0 for rcb); rcb errors with comm_style brick; the values
 * are intensive and set at run setup. Weight styles other than group, and
 * the out keyword, are not supported here.
 */

export class FixBalance extends Fix {
  readonly style = 'balance';
  vectorFlag = true;
  sizeVector = 3;
  scalarFlag = true;
  private thresh = 1;
  private iterations = 0;
  private weights: { bit: number; w: number }[] = [];
  private result = { scalar: 0, vec: [0, 0, 0] };

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const nfreq = Number(args[0]);
    this.thresh = Number(args[1]);
    if (!Number.isInteger(nfreq) || nfreq < 0 || !Number.isFinite(this.thresh) || this.thresh <= 0) {
      throw new StyleError('usage: fix ID group-ID balance Nfreq thresh style args keyword args ...');
    }
    let k = 2;
    const style = args[k++];
    if (style === 'shift') {
      const dims = args[k];
      if (!dims || !/^[xyz]{1,3}$/.test(dims) || new Set(dims).size !== dims.length) throw new StyleError(`fix balance shift: dimstr '${dims ?? ''}' must contain x, y or z, each at most once`);
      if (this.sys.dimension === 2 && dims.includes('z')) throw new StyleError('fix balance shift: cannot balance in z for a 2d simulation');
      const niter = Number(args[k + 1]), stop = Number(args[k + 2]);
      if (!Number.isInteger(niter) || niter < 1 || !Number.isFinite(stop)) throw new StyleError('usage: fix ID group-ID balance Nfreq thresh shift dimstr Niter stopthresh');
      this.iterations = dims.length;
      k += 3;
    } else if (style === 'rcb') {
      if (sys.commStyle !== 'tiled') throw new StyleError('Fix balance rcb cannot be used with comm_style brick');
      this.iterations = 0;
    } else if (style === 'report') {
      this.iterations = 0;
    } else throw new StyleError(`fix balance: unknown style '${style ?? ''}' (shift, rcb or report)`);
    while (k < args.length) {
      const key = args[k];
      if (key === 'weight') {
        if (args[k + 1] !== 'group') throw new StyleError(`fix balance weight ${args[k + 1] ?? ''} is not supported by the browser engine (only weight group)`);
        const ng = Number(args[k + 2]);
        if (!Number.isInteger(ng) || ng < 1) throw new StyleError('fix balance weight group: Ngroup must be a positive integer');
        for (let g = 0; g < ng; g++) {
          const name = args[k + 3 + 2 * g], w = Number(args[k + 4 + 2 * g]);
          if (!name || !(w > 0)) throw new StyleError('fix balance weight group: expected group-ID weight pairs with weights > 0');
          this.weights.push({ bit: sys.groups.bit(name), w });
        }
        k += 3 + 2 * ng;
      } else if (key === 'sort') {
        if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError('fix balance sort must be yes or no');
        k += 2;
      } else if (key === 'out') {
        throw new StyleError('fix balance out (writing subdomain files) is not supported by the browser engine');
      } else throw new StyleError(`fix balance: unknown keyword '${key}'`);
    }
  }

  setup(): void {
    const s = this.sys.state;
    let total = 0;
    for (let i = 0; i < s.n; i++) {
      let w = 1;
      for (const g of this.weights) if (s.mask[i] & g.bit) w *= g.w;
      total += w;
    }
    // one process holds everything: imbalance 1.0, re-balanced only when thresh < 1
    const forced = 1 > this.thresh;
    this.result = forced ? { scalar: 1, vec: [total, this.iterations, 1] } : { scalar: 0, vec: [total, 0, 0] };
  }

  computeScalar(): number { return this.result.scalar; }
  computeVector(i: number): number { return this.result.vec[i]; }
}
