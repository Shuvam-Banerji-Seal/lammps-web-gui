/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/APIP/fix_atom_weight_apip.cpp
 * (fix atom_weight/apip: the arguments, init, setup_pre_exchange, pre_exchange, end_of_step, calc_work_per_particle and
 * compute_vector), at commit b332f28. Author named in the source: David Immel (FZ Juelich), with the LAMMPS Sandia
 * copyright notice (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/fix_atom_weight_apip.html: "fix ID group-ID atom_weight/apip nevery fast_potential
 * precise_potential lambda_input lambda_zone group_lambda_input [no_rescale]". The per-atom weight is the sum of the
 * time per atom of each force routine the atom needed (the SIMPLE and COMPLEX bits of lambda_required, and membership of
 * the lambda_input group), and the global vector holds the four times per atom. The weight only feeds fix balance, which
 * has nothing to balance in this single-process engine. Measured times: the APIP pair styles time their own computes in
 * the browser (performance.now, in seconds) where native reads its wall clock; and the rescaling (without no_rescale)
 * scales the weights to the total time of those APIP pair computes since the last calculation, where native uses its
 * pair, neighbor, bond and kspace timers. With constant times and no_rescale the weights are exact.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { APIP_LAMBDA_REQUIRED } from '../../engine/atoms';
import { COMPLEX, SIMPLE } from './common';

/** The timing interface of the APIP pair styles (pair->extract("<style>:time_per_atom")). */
export interface ApipTimed {
  /** time per atom since the last call, then reset (calculate_time_per_atom); null when the name is not this style's. */
  apipTimePerAtom(name: string): number | null;
  /** total wall time of this style's computes so far (seconds). */
  apipWallTotal(): number;
}

const isTimed = (p: unknown): p is ApipTimed => typeof (p as ApipTimed | null)?.apipTimePerAtom === 'function';

export class FixAtomWeightAPIP extends Fix {
  readonly style = 'atom_weight/apip';
  private readonly extractName: (string | null)[] = [null, null, null, null];
  private readonly timeAtom = [-1, -1, -1, -1];
  private readonly avgTimeAtom = new Float64Array(4);
  private readonly groupName: string;
  private readonly rescaleWork: boolean;
  private lastCalc = -1;
  private wallLast = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 6) throw new StyleError('Illegal fix atom_weight/apip command: missing argument(s)');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
    this.vectorFlag = true;
    this.sizeVector = 4;
    this.extvector = 0;
    const num = (w: string): number => {
      const v = Number(w);
      if (w === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
      return v;
    };
    const nevery = /^[+-]?\d+$/.test(args[0]) ? parseInt(args[0], 10) : NaN;
    if (Number.isNaN(nevery)) throw new StyleError(`Expected integer parameter instead of '${args[0]}' in input script or data file`);
    if (args[1] === 'eam') this.extractName[0] = 'eam/apip:time_per_atom';
    else if (args[1] === 'ace') this.extractName[0] = 'pace/fast/apip:time_per_atom';
    else this.avgTimeAtom[0] = this.timeAtom[0] = num(args[1]);
    if (args[2] === 'ace') this.extractName[1] = 'pace/apip:time_per_atom';
    else this.avgTimeAtom[1] = this.timeAtom[1] = num(args[2]);
    if (args[3] === 'lambda/input') this.extractName[2] = 'lambda/input/apip:time_per_atom';
    else this.avgTimeAtom[2] = this.timeAtom[2] = num(args[3]);
    if (args[4] === 'lambda/zone') this.extractName[3] = 'lambda/zone/apip:time_per_atom';
    else this.avgTimeAtom[3] = this.timeAtom[3] = num(args[4]);
    this.groupName = args[5];
    if (!sys.groups.names.includes(this.groupName)) throw new StyleError(`Fix atom_weight/apip: group ${this.groupName} does not exist`);
    let rescale = true;
    for (let k = 6; k < args.length; k++) {
      if (args[k] === 'no_rescale') rescale = false;
      else throw new StyleError(`Fix atom_weight/apip: unknown argument ${args[k]}`);
    }
    this.rescaleWork = rescale;
    if (nevery < 1) throw new StyleError('Fix atom_weight/apip: nevery > 0 required');
    this.nevery = nevery;
    if (!sys.state.custom.has(APIP_LAMBDA_REQUIRED)) throw new StyleError('Fix atom_weight/apip: atomic style with lambda_required required');
    const what = ['simple', 'complex', 'group', 'lambda'];
    const kinds = ['time_simple_atom', 'time_complex_atom', 'time_group_atom', 'time_lambda_atom'];
    for (let q = 0; q < 4; q++) {
      const name = this.extractName[q];
      if (name) {
        if (!sys.ff.pair) throw new StyleError('Fix atom_weight/apip: extract requires a defined pair style');
        if (this.timedPair(name) === null) throw new StyleError(`Fix atom_weight/apip: ${what[q]} time cannot be extracted with ${name} from ${sys.ff.pair.name}`);
      } else if (this.timeAtom[q] < 0) {
        throw new StyleError(`Fix atom_weight/apip: ${kinds[q]} needs to be non-negative instead of ${this.timeAtom[q]}`);
      }
    }
  }

  /** The pair (or hybrid sub-style) that answers the extract name. */
  private timedPair(name: string): ApipTimed | null {
    const pair = this.sys.ff.pair as unknown as { subs?: { style: unknown }[] } | null;
    if (!pair) return null;
    const styles = pair.subs ? pair.subs.map((s) => s.style) : [pair];
    const prefix = name.slice(0, name.indexOf(':'));
    for (const st of styles) {
      if (!isTimed(st)) continue;
      const n = (st as { name?: string }).name ?? '';
      // native extract names: eam/apip (also eam/fs/apip), pace/apip (also pace/precise/apip), pace/fast/apip
      const answers = prefix === 'eam/apip' ? n === 'eam/apip' || n === 'eam/fs/apip'
        : prefix === 'pace/apip' ? n === 'pace/apip' || n === 'pace/precise/apip'
        : n === prefix;
      if (answers) return st;
    }
    return null;
  }

  init(): void {
    const sys = this.sys;
    if (sys.fixes.filter((f) => f.style === 'atom_weight/apip').length > 1) throw new StyleError('More than one atom_weight/apip fix');
    const lambdaFixes = sys.fixes.filter((f) => f.style === 'lambda/apip');
    if (lambdaFixes.length > 1) throw new StyleError('More than one fix lambda/apip');
    if (lambdaFixes.length === 0 && (this.extractName[3] || this.timeAtom[3] > 0)) {
      throw new StyleError('fix lambda/apip required to approximate weight of pair style lambda/zone/apip');
    }
    if (!sys.groups.names.includes(this.groupName)) throw new StyleError(`Fix atom_weight/apip: group ${this.groupName} does not exist`);
    this.wallLast = this.wallTotal();
    this.lastCalc = -1;
  }

  private wallTotal(): number {
    const pair = this.sys.ff.pair as unknown as { subs?: { style: unknown }[] } | null;
    if (!pair) return 0;
    const styles = pair.subs ? pair.subs.map((s) => s.style) : [pair];
    let t = 0;
    for (const st of styles) if (isTimed(st)) t += st.apipWallTotal();
    return t;
  }

  /** setup_pre_exchange: weight 1 until the first forces. */
  setup(): void {
    const n = this.sys.state.n;
    this.vectorAtom = new Float64Array(n).fill(1);
  }

  preExchange(): void {
    if (this.sys.state.step % this.nevery !== 0) return;
    this.calcWork();
  }

  endOfStep(): void {
    if (this.sys.state.step % this.nevery !== 0) return;
    this.calcWork();
  }

  /** calc_work_per_particle. */
  private calcWork(): void {
    const s = this.sys.state;
    if (s.step === this.lastCalc) return;
    this.lastCalc = s.step;
    for (let q = 0; q < 4; q++) {
      const name = this.extractName[q];
      if (!name) continue;
      const t = this.timedPair(name)?.apipTimePerAtom(name) ?? -1;
      // one process: the average of the processes that measured is this value, or 0 when nothing was measured
      this.avgTimeAtom[q] = t < 0 ? 0 : t;
      this.timeAtom[q] = t < 0 ? this.avgTimeAtom[q] : t;
    }
    const n = s.n;
    const weight = new Float64Array(n);
    const req = s.custom.get(APIP_LAMBDA_REQUIRED)!.data;
    for (let i = 0; i < n; i++) {
      let work = 0;
      if (req[i] & SIMPLE) work += this.timeAtom[0];
      if (req[i] & COMPLEX) work += this.timeAtom[1];
      weight[i] = work;
    }
    if (this.timeAtom[2] > 0) {
      const gb = this.sys.groups.bit(this.groupName);
      for (let i = 0; i < n; i++) if (s.mask[i] & gb) weight[i] += this.timeAtom[2];
    }
    if (this.timeAtom[3] > 0) throw new StyleError('fix lambda/apip required to approximate weight of pair style lambda/zone/apip');
    if (this.rescaleWork) {
      let workAtoms = 0;
      for (let i = 0; i < n; i++) workAtoms += weight[i];
      const total = this.wallTotal();
      const work = total - this.wallLast;
      this.wallLast = total;
      const factor = work / workAtoms;
      for (let i = 0; i < n; i++) weight[i] *= factor;
    }
    this.vectorAtom = weight;
  }

  computeVector(i: number): number {
    return i < 4 ? this.avgTimeAtom[i] : 0;
  }
}
