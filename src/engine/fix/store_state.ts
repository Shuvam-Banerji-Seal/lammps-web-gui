import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { expandWildcards, parseRef, peratomValues, type Ref } from '../refs';

/*
 * fix ID group-ID store/state N input1 input2 ... [keyword value ...] — docs.lammps.org/fix_store_state.html:
 * "N = store atom attributes every N steps, N = 0 for initial store only"; "If N > 0, then the stored attributes
 * will be updated once every N steps. If N = 0, then the attributes are stored when the fix is defined and never
 * changed"; "The requested values are stored in a per-atom vector or array as discussed below. Zeroes are stored for
 * atoms not in the specified group."; "Only atom attributes specified by keywords like xu or vy or radius are
 * initially stored immediately at the point in your input script when the fix is defined. Attributes specified by a
 * compute, fix, or variable are not initially stored until the first run following the fix definition begins."
 * Measured with native LAMMPS (black box, lj fcc, fix store/state 5 x y z, dump of f_s[1..3] each step): the values
 * hold the definition-time positions on steps 0 to 4, and change to the positions of steps 5 and 10 at those steps.
 *
 * Supported inputs: the atom attributes x, y, z, vx, vy, vz, fx, fy, fz, and c_ID, c_ID[I], f_ID, f_ID[I], v_name.
 * The keywords com, history and thresh and the other attributes of the docs (id, type, xu, ix, ...) are refused by
 * the input parser ("invalid input value"), and the keywords by name.
 */

const KEYWORDS = new Set(['com', 'history', 'thresh']);

export class FixStoreState extends Fix {
  readonly style = 'store/state';
  peratomFlag = true;
  private readonly nStore: number;
  private readonly inputs: Ref[];
  private readonly nvals: number;
  private stored = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} (store/state)`;
    if (args.length < 2) throw new StyleError(`usage: fix ${id} group-ID store/state N input1 input2 ... [keyword value ...]`);
    this.nStore = Number(args[0]);
    if (!Number.isInteger(this.nStore) || this.nStore < 0) throw new StyleError(`${what}: N must be an integer >= 0`);
    let k = 1;
    while (k < args.length && !KEYWORDS.has(args[k])) k++;
    if (k === 1) throw new StyleError(`${what}: no input attributes given`);
    if (k < args.length) throw new StyleError(`${what}: keyword '${args[k]}' is not supported by the browser engine`);
    const words = expandWildcards(sys, args.slice(1, k), 'peratom');
    this.inputs = words.map((w) => this.parseInput(w));
    this.nvals = this.inputs.length;
    this.sizePeratomCols = this.nvals === 1 ? 0 : this.nvals;
    this.nevery = 1;
    this.alloc();
    // attributes are stored at the definition; computes, fixes and variables wait for the first run
    if (this.inputs.every((r) => r.kind === 'attr')) this.store();
  }

  private parseInput(w: string): Ref {
    const r = parseRef(w, true);
    if (r.kind === 'v') {
      const v = this.sys.vars.get(r.id);
      if (!v) throw new StyleError(`fix store/state: variable ${r.id} does not exist`);
      if (v.style !== 'atom') throw new StyleError(`fix store/state: variable ${r.id} must be atom-style`);
    }
    return r;
  }

  /** The per-atom storage follows the atom count (zeros for the atoms not yet stored). */
  private alloc(): void {
    const need = this.nvals * this.sys.state.n;
    if (this.nvals === 1) {
      if (this.vectorAtom.length !== this.sys.state.n) this.vectorAtom = new Float64Array(this.sys.state.n);
    } else if (this.arrayAtom.length !== need) this.arrayAtom = new Float64Array(need);
  }

  /** Copies the inputs into the stored arrays; atoms outside the group keep zeros. */
  private store(): void {
    this.alloc();
    const s = this.sys.state;
    const n = s.n;
    for (let k = 0; k < this.nvals; k++) {
      const vals = peratomValues(this.sys, this.inputs[k]);
      for (let i = 0; i < n; i++) {
        const value = (s.mask[i] & this.groupBit) !== 0 ? vals[i] : 0;
        if (this.nvals === 1) this.vectorAtom[i] = value;
        else this.arrayAtom[i * this.nvals + k] = value;
      }
    }
    this.stored = true;
  }

  /** The first run stores the computed inputs (they could not be read at the definition). */
  setup(): void {
    if (!this.stored) this.store();
  }

  init(): void {
    this.alloc();
  }

  endOfStep(): void {
    if (this.nStore > 0 && this.sys.state.step % this.nStore === 0) this.store();
  }
}
