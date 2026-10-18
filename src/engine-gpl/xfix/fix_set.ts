/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_set.cpp and src/fix_set.h at commit b332f28 (class FixSet, fix set: the constructor, setmask and
 * end_of_step; contributing author Germain Clavier, TUe). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Runs the actions of the set command every Nevery steps (docs.lammps.org/fix_set.html). The actions are the set
 * command's own (src/engine/commands/setup.ts, the set handler, reused unchanged), and every keyword must use a
 * per-atom variable (v_name), as native Set::process_args does for the FIXSET caller: "Fix set command keyword K does
 * not invoke a per-atom variable". The reneighbor keyword forces a neighbor rebuild on the next step.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { Ctx } from '../../engine/commands/args';
import { SETUP_COMMANDS } from '../../engine/commands/setup';

/** The selection styles of the set command (set.cpp Set::process_args): the first two words are the style and the ID. */
const STYLE_WORDS = ['atom', 'mol', 'type', 'group', 'region'];

export class FixSet extends Fix {
  readonly style = 'set';
  private readonly actions: string[];
  private readonly reneighbor: boolean;
  private nextReneighbor = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // native: narg < 8 is missing arguments (the fix words are "fix ID group set Nevery rnflag style ID keyword values")
    if (args.length < 5) throw new StyleError('Illegal fix set command: missing arguments');
    this.nevery = Number(args[0]);
    if (!/^[+-]?\d+$/.test(args[0]) || this.nevery <= 0) throw new StyleError('Fix set Nevery must be > 0');
    if (!/^[+-]?\d+$/.test(args[1]) || (args[1] !== '0' && args[1] !== '1')) throw new StyleError('Fix set rnflag must be 0/1');
    this.reneighbor = args[1] === '1';
    const rest = args.slice(2);
    if (rest.length < 2 || !STYLE_WORDS.includes(rest[0])) {
      throw new StyleError(`Unrecognized set or fix set command keyword ${rest[0] ?? ''}`);
    }
    this.actions = rest;
    // every keyword group (a word that is not a value: not a number and not a v_ variable) must invoke a variable
    let k = 2;
    while (k < rest.length) {
      const kw = rest[k];
      let j = k + 1;
      let hasVar = false;
      while (j < rest.length && (rest[j].startsWith('v_') || /^[-+.\d]/.test(rest[j]))) {
        if (rest[j].startsWith('v_')) {
          hasVar = true;
          const name = rest[j].slice(2);
          if (!sys.vars.get(name)) throw new StyleError(`Variable name ${name} for set command does not exist`);
        }
        j++;
      }
      if (!hasVar) throw new StyleError(`Fix set command keyword ${kw} does not invoke a per-atom variable`);
      k = j;
    }
  }

  /**
   * Runs the set command's actions on the current atoms, keyword group by keyword group (the same order as the native
   * action list). The set handler of src/engine/commands/setup.ts is reused for every keyword except type v_name: that
   * handler reads the type word as a type label, so a variable is set here with the handler's own range check.
   */
  private runActions(): void {
    const handler = SETUP_COMMANDS.set;
    if (!handler) throw new StyleError('set: the set command is not available');
    const sys = this.sys;
    const ctx = {
      sys,
      fail: (m: string): never => { throw new StyleError(m); },
    } as unknown as Ctx;
    const [style, id] = this.actions;
    const rest = this.actions.slice(2);
    for (let k = 0; k < rest.length;) {
      let j = k + 1;
      while (j < rest.length && (rest[j].startsWith('v_') || /^[-+.\d]/.test(rest[j]))) j++;
      const group = rest.slice(k, j);
      if (group[0] === 'type' && group[1]?.startsWith('v_')) {
        const v = sys.atomVariable(group[1].slice(2));
        const ids = this.selectedAtoms(style, id);
        const s = sys.state;
        for (const i of ids) {
          const t = Math.trunc(v[i]);
          if (t < 1 || t > s.ntypes) throw new StyleError(`set type ${t} is outside 1..${s.ntypes}`);
          s.type[i] = t;
        }
      } else {
        void handler(ctx, [style, id, ...group]);
      }
      k = j;
    }
  }

  /**
   * The atoms selected by a set style and ID, for the type v_name action: group, and atom with an ID or a lo*hi range
   * (set.cpp Set::selection; other styles are refused here, since the set handler's selection is not exported).
   */
  private selectedAtoms(style: string, id: string): number[] {
    const s = this.sys.state;
    const out: number[] = [];
    if (style === 'group') {
      const bit = this.sys.groups.bit(id);
      for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) out.push(i);
      return out;
    }
    if (style === 'atom') {
      const m = /^(\d*)\*?(\d*)$/.exec(id);
      if (!m || (m[1] === '' && m[2] === '')) throw new StyleError(`invalid set ID '${id}'`);
      const lo = m[1] === '' ? 1 : Number(m[1]);
      const hi = id.includes('*') ? (m[2] === '' ? Number.MAX_SAFE_INTEGER : Number(m[2])) : lo;
      for (let i = 0; i < s.n; i++) if (s.id[i] >= lo && s.id[i] <= hi) out.push(i);
      return out;
    }
    throw new StyleError(`fix set: keyword type with a variable needs the group or atom selection (got ${style})`);
  }

  endOfStep(): void {
    // native thermo output reports the energies of this step's force evaluation (pair->eng_vdwl), which the actions
    // do not recompute: the forces are made current before the actions, and marked current again after them
    this.sys.forces();
    this.runActions();
    this.sys.forcesCurrent();
    if (this.reneighbor) this.nextReneighbor = this.sys.state.step + 1;
  }

  /** The neighbor rebuild requested for the next step (reneighbor yes). */
  reneighborDue(_step: number): boolean {
    // As native LAMMPS does (src/fix_set.cpp at b332f28): end_of_step sets next_reneighbor, but the constructor never
    // sets force_reneighbor, so Neighbor::decide never consults it and rnflag 1 rebuilds nothing (measured: rnflag 0 and
    // 1 give the same run)
    return false;
  }
}
