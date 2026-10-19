/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/APIP/pair_lambda_zone_apip.cpp
 * (pair_style lambda/zone/apip: settings, coeff, init_style, init_one, compute, calculate_lambda,
 * switching_function_poly_distance and the time per atom), at commit b332f28. Author named in the source: David Immel
 * (FZ Juelich), with the LAMMPS Sandia copyright notice (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_lambda_zone_apip.html. No force or energy: the compute spreads the precise region of
 * lambda_input_ta over a transition zone. Native walks the ghost neighbour list of every owned and ghost atom i and
 * lowers lambda_ta of each owned neighbour j to 1 - (1 - lambda_input_ta_i) f(r_ij); the result is the minimum over all
 * such i, so the engine takes, for each owned atom j, the minimum over its full-list neighbours (owned atoms and
 * ghosts, whose values are their owners'), which is the same set of pairs.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, type Neighbor } from '../../engine/neighbor';
import type { SimState } from '../../engine/types';
import { APIP_LAMBDA_INPUT_TA } from '../../engine/atoms';
import { pow as libmPow } from '../libm/pow';

/** The settings of fix lambda/apip the zone reads (fix_lambda->extract in native). */
export interface LambdaZoneSettings {
  readonly cutLo: number;
  readonly cutHi: number;
  readonly cutHiSq: number;
  readonly cutWidth: number;
  readonly lambdaNonGroup: number;
  readonly groupBit: number;
}

export class PairLambdaZoneAPIP extends Pair {
  readonly name: string = 'lambda/zone/apip';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  private cutGlobal = 0;
  private readonly coeffed = new Set<string>();
  private z: LambdaZoneSettings | null = null;
  private timer = 0;
  private nCalculations = 0;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.coeffed.clear();
  }

  override settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError(`pair_lambda_zone: expected 1 instead of ${args.length} arguments`);
    const v = Number(args[0]);
    if (args[0] === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${args[0]}' in input script or data file`);
    if (v <= 0) throw new StyleError(`pair_lambda_zone: cut_global = ${v} <= 0`);
    this.cutGlobal = v;
  }

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
    if (ctx.s && !ctx.s.custom.has(APIP_LAMBDA_INPUT_TA)) throw new StyleError('pair_lambda_zone requires an atom style with lambda_input_ta');
    const f = ctx.fixOf?.('lambda/apip') as unknown as LambdaZoneSettings | null | undefined;
    if (!f) throw new StyleError('Exact one fix lambda required');
    this.z = f;
    if (f.cutHi > this.cutGlobal) throw new StyleError('The r_lambda_hi > neighbour_list_cutoff.');
  }

  override initOne(): number {
    return this.cutGlobal;
  }

  override compute(pc: PairCompute): void {
    this.calculateLambda(pc.s, pc.nb);
  }

  /** switching_function_poly_distance. */
  private sw(input: number): number {
    const z = this.z!;
    if (input <= z.cutLo) return 1;
    if (input >= z.cutHi) return 0;
    const d = 1 - 2 * (1 + (input - z.cutHi) / z.cutWidth);
    return 0.5 + 7.5 / 2. * (d / 4. - libmPow(d, 3) / 6. + libmPow(d, 5) / 20.);
  }

  /** calculate_lambda: the zone-spread lambda_input_ta of the owned atoms. */
  calculateLambda(s: SimState, nb: Neighbor): void {
    const t0 = performance.now();
    const z = this.z;
    if (!z) throw new StyleError('Exact one fix lambda required');
    const list = nb.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    const lta = s.custom.get(APIP_LAMBDA_INPUT_TA)!.data;
    const owner = nb.owner, x = nb.xall;
    const nlocal = nb.nlocal;
    const gb = z.groupBit;
    const lambdaTa = new Float64Array(nlocal);
    for (let i = 0; i < nlocal; i++) lambdaTa[i] = (s.mask[i] & gb) ? lta[i] : z.lambdaNonGroup;
    // n_calculations: the owned and ghost atoms whose neighbours native visits
    let loops = 0;
    if (z.cutHiSq > 0) {
      for (let k = 0; k < nb.nall; k++) if (lta[owner[k]] !== 1 && (s.mask[owner[k]] & gb)) loops++;
      for (let j = 0; j < nlocal; j++) {
        if (!(s.mask[j] & gb)) continue;
        const k0 = list.firstneigh[j], k1 = k0 + list.numneigh[j];
        for (let k = k0; k < k1; k++) {
          const i = list.neighbors[k] & NEIGHMASK;
          const oi = owner[i];
          const lti = lta[oi];
          if (lti === 1 || !(s.mask[oi] & gb)) continue;
          if (lambdaTa[j] === 0) continue;
          const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
          const rsq = dx * dx + dy * dy + dz * dz;
          if (rsq >= z.cutHiSq) continue;
          const lambdaNew = 1 - (1 - lti) * this.sw(Math.sqrt(rsq));
          if (lambdaNew < lambdaTa[j]) lambdaTa[j] = lambdaNew;
        }
      }
    }
    for (let i = 0; i < nlocal; i++) lta[i] = lambdaTa[i];
    this.timer += (performance.now() - t0) / 1000;
    this.nCalculations += loops;
  }

  apipTimePerAtom(name: string): number | null {
    if (name !== 'lambda/zone/apip:time_per_atom') return null;
    const t = this.nCalculations > 0 ? this.timer / this.nCalculations : -1;
    this.timer = 0;
    this.nCalculations = 0;
    return t;
  }

  apipWallTotal(): number {
    return 0;
  }
}
