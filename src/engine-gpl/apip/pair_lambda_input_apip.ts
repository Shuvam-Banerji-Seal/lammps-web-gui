/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/APIP/pair_lambda_input_apip.cpp
 * (pair_style lambda/input/apip: settings, coeff, init_style, init_one, compute, calculate_lambda_input and the time per
 * atom) and src/APIP/pair_lambda_input_csp_apip.cpp (pair_style lambda/input/csp/apip: settings and the
 * centro-symmetry parameter of calculate_lambda_input with the two Numerical Recipes select routines), at commit
 * b332f28. Author named in the sources: David Immel (FZ Juelich), with the LAMMPS Sandia copyright notice (GNU General
 * Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_lambda_input_apip.html. These styles add no force or energy: their compute writes the
 * per-atom lambda_input (0 for lambda/input/apip; the centro-symmetry parameter of the nnn nearest neighbours within the
 * cutoff, or 1000 with fewer neighbours, for lambda/input/csp/apip) and then updates the history of fix lambda/apip.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, type Neighbor } from '../../engine/neighbor';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { APIP_LAMBDA_INPUT } from '../../engine/atoms';
import { NativeFullOrder } from '../neighbor/nativeOrder';

/** The part of fix lambda/apip the lambda/input styles call. */
export interface LambdaInputHistory {
  updateLambdaInputHistory(): void;
  readonly groupBitIgnoreLambdaInput: number;
}

export class PairLambdaInputAPIP extends Pair {
  readonly name: string = 'lambda/input/apip';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  protected cutGlobal = -1;
  protected readonly coeffed = new Set<string>();
  protected ignoreGroupBit = 0;
  /** fix lambda/apip, found by init_style. */
  fixLambda: LambdaInputHistory | null = null;
  protected readonly natOrder = new NativeFullOrder();
  private timer = 0;
  private nCalculations = 0;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.coeffed.clear();
  }

  /** settings: cutoff (native utils::numeric message). */
  override settings(args: string[]): void {
    if (args.length < 1) throw new StyleError('Illegal pair_style lambda_input command: missing argument(s)');
    this.cutGlobal = this.num(args[0]);
  }

  protected num(w: string): number {
    const v = Number(w);
    if (w === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
    return v;
  }

  /** coeff: "pair_coeff I J" (two arguments). */
  override coeff(args: string[]): void {
    if (args.length !== 2) throw new StyleError('Incorrect args for pair coefficients');
    const n = this.ntypes;
    const bounds = (w: string): [number, number] => {
      if (w === '*') return [1, n];
      const m = /^(\d*)\*(\d*)$/.exec(w);
      if (m) return [m[1] ? Number(m[1]) : 1, m[2] ? Number(m[2]) : n];
      if (!/^\d+$/.test(w) || Number(w) < 1 || Number(w) > n) throw new StyleError(`Numeric index ${w} is out of bounds (1-${n})`);
      return [Number(w), Number(w)];
    };
    const [ilo, ihi] = bounds(args[0]);
    const [jlo, jhi] = bounds(args[1]);
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.coeffed.add(`${i},${j}`);
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  override setsPair(i: number, j: number): boolean {
    return this.coeffed.has(`${Math.min(i, j)},${Math.max(i, j)}`);
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.s && !ctx.s.custom.has(APIP_LAMBDA_INPUT)) throw new StyleError('pair_lambda input requires an atom style with lambda_input');
    // init_style: exactly one fix lambda/apip (the fix registers itself with this pair, fix_lambda_apip.ts)
    const f = ctx.fixOf?.('lambda/apip') as unknown as LambdaInputHistory | null | undefined;
    if (!f) throw new StyleError('Exact one fix lambda/apip required');
    this.fixLambda = f;
    this.ignoreGroupBit = f.groupBitIgnoreLambdaInput;
  }

  override initOne(): number {
    return this.cutGlobal;
  }

  override compute(pc: PairCompute): void {
    const t0 = performance.now();
    this.nCalculations += this.calculateLambdaInput(pc.s, pc.nb, pc.geom);
    this.timer += (performance.now() - t0) / 1000;
    this.fixLambda?.updateLambdaInputHistory();
  }

  /** calculate_lambda_input: lambda_input = 0 for every owned atom; returns the number of atoms. */
  calculateLambdaInput(s: SimState, nb: Neighbor, geom: Geometry): number {
    void nb; void geom;
    s.custom.get(APIP_LAMBDA_INPUT)!.data.fill(0, 0, s.n);
    return s.n;
  }

  /** extract("lambda/input/apip:time_per_atom") through calculate_time_per_atom. */
  apipTimePerAtom(name: string): number | null {
    if (name !== 'lambda/input/apip:time_per_atom') return null;
    const t = this.nCalculations > 0 ? this.timer / this.nCalculations : -1;
    this.timer = 0;
    this.nCalculations = 0;
    return t;
  }

  apipWallTotal(): number {
    return 0;
  }
}

/** select (Numerical Recipes, as in pair_lambda_input_csp_apip.cpp): the k smallest values of arr[0..n) first. */
const select = (k: number, n: number, a: Float64Array): void => {
  // 1-based indices as the C++ (arr-- before the loop)
  const at = (q: number): number => a[q - 1];
  const set = (q: number, v: number): void => { a[q - 1] = v; };
  const swap = (p: number, q: number): void => { const t = a[p - 1]; a[p - 1] = a[q - 1]; a[q - 1] = t; };
  let l = 1, ir = n;
  for (;;) {
    if (ir <= l + 1) {
      if (ir === l + 1 && at(ir) < at(l)) swap(l, ir);
      return;
    }
    const mid = (l + ir) >> 1;
    swap(mid, l + 1);
    if (at(l) > at(ir)) swap(l, ir);
    if (at(l + 1) > at(ir)) swap(l + 1, ir);
    if (at(l) > at(l + 1)) swap(l, l + 1);
    let i = l + 1, j = ir;
    const v = at(l + 1);
    for (;;) {
      do i++; while (at(i) < v);
      do j--; while (at(j) > v);
      if (j < i) break;
      swap(i, j);
    }
    set(l + 1, at(j));
    set(j, v);
    if (j >= k) ir = j - 1;
    if (j <= k) l = i;
  }
};

/** select2: as select, with an index array permuted alongside. */
const select2 = (k: number, n: number, a: Float64Array, ia: Int32Array): void => {
  const at = (q: number): number => a[q - 1];
  const swap = (p: number, q: number): void => {
    const t = a[p - 1]; a[p - 1] = a[q - 1]; a[q - 1] = t;
    const u = ia[p - 1]; ia[p - 1] = ia[q - 1]; ia[q - 1] = u;
  };
  let l = 1, ir = n;
  for (;;) {
    if (ir <= l + 1) {
      if (ir === l + 1 && at(ir) < at(l)) swap(l, ir);
      return;
    }
    const mid = (l + ir) >> 1;
    swap(mid, l + 1);
    if (at(l) > at(ir)) swap(l, ir);
    if (at(l + 1) > at(ir)) swap(l + 1, ir);
    if (at(l) > at(l + 1)) swap(l, l + 1);
    let i = l + 1, j = ir;
    const v = at(l + 1), iv = ia[l];
    for (;;) {
      do i++; while (at(i) < v);
      do j--; while (at(j) > v);
      if (j < i) break;
      swap(i, j);
    }
    a[l] = at(j); ia[l] = ia[j - 1];
    a[j - 1] = v; ia[j - 1] = iv;
    if (j >= k) ir = j - 1;
    if (j <= k) l = i;
  }
};

export class PairLambdaInputCSPAPIP extends PairLambdaInputAPIP {
  override readonly name: string = 'lambda/input/csp/apip';
  private nnn = 12;
  private nnnBuffer = 0;
  private cutCspSq = 25;

  /** settings: fcc|bcc|N [cutoff value] [N_buffer value] (native messages). */
  override settings(args: string[]): void {
    let cutCsp = 5;
    if (args.length < 1) throw new StyleError('pair lambda_input/csp: lattice requires one argument');
    if (args[0] === 'fcc') this.nnn = 12;
    else if (args[0] === 'bcc') this.nnn = 8;
    else {
      if (!/^[+-]?\d+$/.test(args[0])) throw new StyleError(`Expected integer parameter instead of '${args[0]}' in input script or data file`);
      this.nnn = parseInt(args[0], 10);
    }
    let iarg = 1;
    while (iarg < args.length) {
      if (args[iarg] === 'cutoff') {
        if (iarg + 1 >= args.length) throw new StyleError('pair lambda_input/csp: threshold requires an argument');
        cutCsp = this.num(args[iarg + 1]);
        iarg += 2;
      } else if (args[iarg] === 'N_buffer') {
        if (iarg + 1 >= args.length) throw new StyleError('pair lambda_input/csp: N_buffer requires an argument');
        if (!/^[+-]?\d+$/.test(args[iarg + 1])) throw new StyleError(`Expected integer parameter instead of '${args[iarg + 1]}' in input script or data file`);
        this.nnnBuffer = parseInt(args[iarg + 1], 10);
        iarg += 2;
      } else {
        throw new StyleError(`pair_lambda_input_csp: unknown argument ${args[iarg]}`);
      }
    }
    if (this.nnn <= 1 || this.nnn % 2) throw new StyleError('pair_lambda_input_csp: even number of neighbours > 1 for csp calculation required');
    if (cutCsp <= 0) throw new StyleError('pair_lambda_input_csp: cut_csp <= 0');
    if (this.nnnBuffer < 0) throw new StyleError('pair_lambda_input_csp: N_buffer negative');
    this.cutGlobal = cutCsp;
    this.cutCspSq = cutCsp * cutCsp;
  }

  /** calculate_lambda_input: the centro-symmetry parameter of each owned atom outside the ignore group. */
  override calculateLambdaInput(s: SimState, nb: Neighbor, geom: Geometry): number {
    const list = nb.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    const nbrs = this.natOrder.neighbors(s, nb, geom, list);
    const x = nb.xall;
    const lambdaInput = s.custom.get(APIP_LAMBDA_INPUT)!.data;
    const nhalf = this.nnn / 2;
    const nnnAll = this.nnn + this.nnnBuffer;
    const npairs = (nnnAll * (nnnAll - 1)) / 2;
    const pairs = new Float64Array(npairs);
    let distsq = new Float64Array(64), nearest = new Int32Array(64);
    const inum = list.inum;
    for (let i = 0; i < inum; i++) {
      if (s.mask[i] & this.ignoreGroupBit) {
        lambdaInput[i] = 0;
        continue;
      }
      const xt = x[3 * i], yt = x[3 * i + 1], zt = x[3 * i + 2];
      const k0 = list.firstneigh[i], jnum = list.numneigh[i];
      if (jnum > distsq.length) {
        distsq = new Float64Array(jnum);
        nearest = new Int32Array(jnum);
      }
      let nCut = 0;
      for (let jj = 0; jj < jnum; jj++) {
        const j = nbrs[k0 + jj] & NEIGHMASK;
        const dx = xt - x[3 * j], dy = yt - x[3 * j + 1], dz = zt - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < this.cutCspSq) {
          distsq[nCut] = rsq;
          nearest[nCut++] = j;
        }
      }
      let value: number;
      if (nCut >= nnnAll) {
        select2(nnnAll, nCut, distsq, nearest);
        let n = 0;
        for (let a = 0; a < nnnAll; a++) {
          const ja = nearest[a];
          for (let b = a + 1; b < nnnAll; b++) {
            const kb = nearest[b];
            const dx = x[3 * ja] + x[3 * kb] - 2.0 * xt;
            const dy = x[3 * ja + 1] + x[3 * kb + 1] - 2.0 * yt;
            const dz = x[3 * ja + 2] + x[3 * kb + 2] - 2.0 * zt;
            pairs[n++] = dx * dx + dy * dy + dz * dz;
          }
        }
        select(nhalf, npairs, pairs);
        value = 0.0;
        for (let a = 0; a < nhalf; a++) value += pairs[a];
      } else {
        value = 1000.0;
      }
      lambdaInput[i] = value;
    }
    return inum;
  }
}
