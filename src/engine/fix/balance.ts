import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { CustomProp } from '../types';
import { formatNumber } from '../cformat';

/**
 * The out file of balance and fix balance (balance.html: "The *out* keyword writes text to the specified filename
 * with the results of each rebalancing operation."), in the layout of native's Balance::dumpout (src/balance.cpp at
 * b332f28): a Pizza.py mdump snapshot of the nodes of every processor's sub-domain, then one square (2d) or cube (3d)
 * per processor. The engine is one process, so its sub-domain is the whole box (the corners of the tilted box for a
 * triclinic one); numbers are written with %g.
 */
export function balanceDump(sys: System, step: number): string {
  const s = sys.state;
  const g = sys.geom;
  const b = s.box;
  const dim = s.dimension;
  const f = (v: number): string => formatNumber(v, '%g');
  const lines: string[] = ['ITEM: TIMESTEP', String(step), 'ITEM: NUMBER OF NODES', String(dim === 2 ? 4 : 8), 'ITEM: BOX BOUNDS'];
  for (let d = 0; d < 3; d++) lines.push(`${f(b.lo[d])} ${f(b.hi[d])}`);
  lines.push('ITEM: NODES');
  // the corners in native's order: (lo,lo,lo), (hi,lo,lo), (hi,hi,lo), (lo,hi,lo), then the same at z hi for an
  // orthogonal box; Domain::lamda_box_corners' order (x fastest, then y, then z) for a triclinic one
  const corner = (l0: number, l1: number, l2: number): number[] => {
    const out = [0, 0, 0];
    g.fromLamda(l0, l1, l2, out);
    return out;
  };
  const pts: number[][] = [];
  if (!b.triclinic) {
    const lo = b.lo, hi = b.hi;
    const z0 = dim === 2 ? 0 : lo[2];
    pts.push([lo[0], lo[1], z0], [hi[0], lo[1], z0], [hi[0], hi[1], z0], [lo[0], hi[1], z0]);
    if (dim === 3) pts.push([lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]);
  } else {
    for (const l2 of dim === 2 ? [0] : [0, 1]) {
      for (const l1 of [0, 1]) for (const l0 of [0, 1]) pts.push(corner(l0, l1, l2));
    }
    if (dim === 2) for (const p of pts) p[2] = 0;
  }
  pts.forEach((p, k) => lines.push(`${k + 1} 1 ${f(p[0])} ${f(p[1])} ${f(p[2])}`));
  lines.push('ITEM: TIMESTEP', String(step), dim === 2 ? 'ITEM: NUMBER OF SQUARES' : 'ITEM: NUMBER OF CUBES', '1', dim === 2 ? 'ITEM: SQUARES' : 'ITEM: CUBES');
  lines.push(dim === 2 ? '1 1 1 2 3 4' : '1 1 1 2 3 4 5 6 7 8');
  return lines.join('\n') + '\n';
}

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
 * shift dimstr (0 for rcb); rcb errors with comm_style brick. The *out*
 * keyword (writing subdomain files) is not supported here.
 *
 * The weighted-balance options are shared with the balance command and
 * documented at docs.lammps.org/balance.html (#weighted_balance):
 *   weight style args:
 *     group args = Ngroup group1 weight1 group2 weight2 ...
 *     neigh factor = compute weight based on number of neighbors
 *     time factor  = compute weight based on time spend computing
 *     var name     = take weight from atom-style variable
 *     store name   = store weight in custom atom property d_name
 * "It is possible to apply multiple weight flags and the weightings they
 * induce will be combined through multiplication." "For all the options
 * below the weight assigned to a particle must be a positive value; an error
 * will be be generated if a weight is <= 0.0."
 *
 * Measured with native LAMMPS (black box, 1 process, lj units on the sc
 * lattices of plans/scratch/balance_weights): the weights of a `weight store`
 * vector, read back through compute property/atom, are
 *  - the running product in keyword order; a store snapshots the product
 *    accumulated before it, so keyword order matters for the stored values,
 *    while the fix's total weight (f_b[1]) is the same in every order;
 *  - `weight time factor` leaves every weight unchanged (factor 1.0) for any
 *    factor on one rank, with no warning; the factor is validated (> 0);
 *  - `weight var name` multiplies by the per-atom value of an atom-style
 *    variable (equal-style and undefined names error), and errors with
 *    "Balance weight <= 0.0" if a value is not positive;
 *  - `weight store name` writes the custom d_name vector (fix property/atom):
 *    an undefined, integer or array property errors (native: "Balance weight
 *    store vector does not exist");
 *  - the weights are refreshed at run setup and every Nfreq steps (measured
 *    with neighbor rebuilds every 1 and every 10: steps 0, Nfreq, 2*Nfreq);
 *  - `weight neigh` uses the half list's pair count, every image pair counted
 *    once, divided by the owned atom count; factor 1.0 only (other factors
 *    leave W = the group weights, with a warning), and no list yet at setup.
 */

/** One parsed weight or store action of the balance *weight* keyword, in input order. */
export type BalanceWeightAction =
  | { kind: 'group'; bit: number; w: number }
  | { kind: 'neigh'; factor: number }
  | { kind: 'time'; factor: number }
  | { kind: 'var'; name: string }
  | { kind: 'store'; name: string };

/** The fix property/atom d_name vector a store action needs, or throws as native does. */
function storeVector(sys: System, name: string): CustomProp {
  const c = sys.state.custom.get(name);
  if (!c || c.int || c.cols !== 0) throw new StyleError(`Balance weight store vector does not exist: '${name}' (define it with fix property/atom d_${name})`);
  return c;
}

/**
 * Parses one "weight style args" keyword at args[k], appends its actions in
 * input order and returns the index after it. docs.lammps.org/balance.html.
 */
export function parseBalanceWeight(sys: System, args: string[], k: number, actions: BalanceWeightAction[]): number {
  const style = args[k + 1];
  if (style === 'group') {
    const ng = Number(args[k + 2]);
    if (!Number.isInteger(ng) || ng < 1) throw new StyleError('balance weight group: Ngroup must be a positive integer');
    for (let g = 0; g < ng; g++) {
      const name = args[k + 3 + 2 * g], w = Number(args[k + 4 + 2 * g]);
      if (!name || !(w > 0)) throw new StyleError('balance weight group: expected group-ID weight pairs with weights > 0');
      actions.push({ kind: 'group', bit: sys.groups.bit(name), w });
    }
    return k + 3 + 2 * ng;
  }
  if (style === 'neigh') {
    const f = Number(args[k + 2]);
    if (!(f > 0)) throw new StyleError('balance weight neigh: factor must be a positive number');
    if (f !== 1) sys.warn(`balance weight neigh factor ${args[k + 2]}: no neighbor weights are used (as in native LAMMPS for factors other than 1.0)`);
    actions.push({ kind: 'neigh', factor: f });
    return k + 3;
  }
  if (style === 'time') {
    const f = Number(args[k + 2]);
    if (!(f > 0)) throw new StyleError('balance weight time: factor must be a positive number');
    actions.push({ kind: 'time', factor: f });
    return k + 3;
  }
  if (style === 'var') {
    const name = args[k + 2];
    const v = name ? sys.vars.get(name) : undefined;
    if (!v) throw new StyleError(`Variable name for balance weight does not exist: '${name ?? ''}'`);
    if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Variable for balance weight has invalid style: '${name}' (must be atom-style)`);
    actions.push({ kind: 'var', name });
    return k + 3;
  }
  if (style === 'store') {
    const name = args[k + 2];
    if (!name) throw new StyleError('balance weight store: expected a d_name custom property name');
    if (sys.hasBox) storeVector(sys, name);
    actions.push({ kind: 'store', name });
    return k + 3;
  }
  throw new StyleError(`balance weight ${style ?? ''} is not supported (weight group, neigh, time, var or store)`);
}

/**
 * The factor the neighbor list contributes to every atom (half-list pairs /
 * owned atoms), or null when no current list is available: docs.lammps.org/
 * balance.html: "A warning will be issued if there is no suitable neighbor
 * list available or if it is not current ... In this case no weights are
 * computed."
 */
export function balanceNeighFactor(sys: System): number | null {
  const list = sys.nb.half;
  if (!list) { sys.warn('Balance weight neigh skipped b/c no suitable list found'); return null; }
  const s = sys.state;
  let pairs = 0;
  for (let i = 0; i < list.inum; i++) pairs += list.numneigh[i];
  return s.n > 0 ? pairs / s.n : 0;
}

/** True when the action list uses a neighbor weight. */
export function balanceUsesNeigh(actions: BalanceWeightAction[]): boolean {
  return actions.some((a) => a.kind === 'neigh');
}

/**
 * Applies the actions in input order to a per-atom weight vector that starts
 * at 1, writing every *store* snapshot into its custom d_name vector, and
 * returns the final weights. The neighbor factor is the current half-list
 * pairs / atoms (null when no neighbor weight is requested or no list exists).
 */
export function applyBalanceWeights(sys: System, actions: BalanceWeightAction[], neighFactor: number | null): Float64Array {
  const s = sys.state;
  const w = new Float64Array(s.n).fill(1);
  for (const a of actions) {
    if (a.kind === 'group') {
      for (let i = 0; i < s.n; i++) if (s.mask[i] & a.bit) w[i] *= a.w;
    } else if (a.kind === 'var') {
      const v = sys.atomVariable(a.name);
      for (let i = 0; i < s.n; i++) {
        if (!(v[i] > 0)) throw new StyleError('Balance weight <= 0.0');
        w[i] *= v[i];
      }
    } else if (a.kind === 'neigh') {
      if (a.factor === 1 && neighFactor !== null) for (let i = 0; i < s.n; i++) w[i] *= neighFactor;
    } else if (a.kind === 'store') {
      const c = storeVector(sys, a.name);
      for (let i = 0; i < s.n; i++) c.data[i] = w[i];
    }
    // 'time': measured with native LAMMPS on one process: every weight is left
    // unchanged (multiplied by 1.0) for any factor, independent of the timer.
  }
  return w;
}

export class FixBalance extends Fix {
  readonly style = 'balance';
  vectorFlag = true;
  sizeVector = 3;
  scalarFlag = true;
  private thresh = 1;
  private iterations = 0;
  private nfreq = 1;
  private actions: BalanceWeightAction[] = [];
  private result = { scalar: 0, vec: [0, 0, 0] };
  /** The out file (balance.cpp options: opened for writing when the fix is defined), and the step last written. */
  private outFile: string | null = null;
  private lastOut = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const nfreq = Number(args[0]);
    this.thresh = Number(args[1]);
    if (!Number.isInteger(nfreq) || nfreq < 0 || !Number.isFinite(this.thresh) || this.thresh <= 0) {
      throw new StyleError('usage: fix ID group-ID balance Nfreq thresh style args keyword args ...');
    }
    this.nfreq = nfreq;
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
      if (key === 'weight') k = parseBalanceWeight(this.sys, args, k, this.actions);
      else if (key === 'sort') {
        if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError('fix balance sort must be yes or no');
        k += 2;
      } else if (key === 'out') {
        const name = args[k + 1];
        if (name === undefined) throw new StyleError('Illegal fix balance out command: missing file name');
        this.outFile = name;
        this.sys.writeFile(name, '', false);
        k += 2;
      } else throw new StyleError(`Unknown (fix) balance keyword ${key}`);
    }
  }

  /** Recomputes the weights (writing store snapshots) and records the fix outputs. */
  private update(neighFactor: number | null): void {
    const s = this.sys.state;
    const w = applyBalanceWeights(this.sys, this.actions, neighFactor);
    let total = 0;
    for (let i = 0; i < s.n; i++) total += w[i];
    const forced = 1 > this.thresh;
    this.result = forced ? { scalar: 1, vec: [total, this.iterations, 1] } : { scalar: 0, vec: [total, 0, 0] };
    // As native LAMMPS does (src/fix_balance.cpp rebalance at b332f28): each rebalancing writes the out file
    if (forced && this.outFile && s.step !== this.lastOut) {
      this.lastOut = s.step;
      this.sys.writeFile(this.outFile, balanceDump(this.sys, s.step), true);
    }
  }

  private updateWeights(): void {
    this.update(balanceUsesNeigh(this.actions) ? balanceNeighFactor(this.sys) : null);
  }

  setup(): void {
    // native: at run setup the neighbor list is not yet used, so no neighbor weights (measured)
    this.update(null);
  }

  /**
   * Native runs fix balance at pre_exchange, before the current step's
   * neighbor list is rebuilt, so *weight neigh* uses the list from the last
   * build (the engine's preExchange, which is the same point). The neighbor
   * rebuild branch is the only place preExchange runs. For Nfreq > 0 native
   * re-balances every Nfreq steps (measured: steps 0, Nfreq, 2*Nfreq with
   * neighbor rebuilds every 1 and every 10).
   */
  preExchange(): void {
    if (!balanceUsesNeigh(this.actions)) return;
    if (this.nfreq !== 0 && this.sys.state.step % this.nfreq !== 0) return;
    this.updateWeights();
  }

  /**
   * The *var* weight reads atom positions, and native evaluates it after the
   * step's periodic remap (measured: native sees the wrapped coordinates, the
   * engine's postIntegrate/preExchange happen before sys.pbc). preForce is the
   * engine's first hook after pbc and runs every step. Nfreq == 0 means every
   * normal reneighbor step.
   */
  preForce(): void {
    if (balanceUsesNeigh(this.actions)) return;
    if (this.nfreq === 0) {
      if (this.sys.nb.lastBuild !== this.sys.state.step) return;
    } else if (this.sys.state.step % this.nfreq !== 0) return;
    this.updateWeights();
  }

  computeScalar(): number { return this.result.scalar; }
  computeVector(i: number): number { return this.result.vec[i]; }
}
