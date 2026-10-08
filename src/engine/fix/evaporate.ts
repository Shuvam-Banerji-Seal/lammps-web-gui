import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { insertionStream, posInt } from './pour';
import type { RanPark } from '../rng';

/*
 * fix evaporate — written from docs.lammps.org/fix_evaporate.html
 * (plans/lammps-docs/fix_evaporate.rst). Syntax: "fix ID group-ID evaporate N
 * M region-ID seed", optional keyword molecule with value no or yes. Page text the engine relies on:
 *   "Remove M atoms from the simulation every N steps."
 *   "Every N steps, the number of atoms in the fix group and within the
 *    specified region are counted. M of these are chosen at random and
 *    deleted. If there are less than M eligible particles, then all of them
 *    are deleted."
 *   "If the setting for the molecule keyword is yes, then when an atom is
 *    chosen for deletion, the entire molecule it is part of is deleted."
 *   "The count of deleted atoms is incremented by the number of atoms in the
 *    molecule," (the doc continues: which may make it exceed M).
 *   "If the molecule ID of the chosen atom is 0, then it is assumed to not be
 *    part of a molecule, and just the single atom is deleted."
 * "The scalar is the cumulative number" of deleted atoms. "This fix is not invoked during" energy minimization.
 *
 * The random selection was measured with native LAMMPS (black box): the
 * generator is RanPark seeded with the seed and advanced past 30 draws (the
 * same stream as fix pour / deposit, see pour.ts). Over 12 seeds with N = 1,
 * M = 1 the deleted atom is index floor(u * n) + 1 of the eligible atoms in
 * native atom order, where u is the next draw and n the number of eligible
 * atoms (matched all 12). For M > 1 each pick uses the next draw on the
 * current list and the picked atom is removed from that list by moving the
 * last list entry into its place (swap-remove; matched all 8 seeds for M = 2
 * and M = 3). With molecule yes the pick is the same, then each member of the
 * picked molecule is counted and removed from the list the same way (matched
 * 32 seed/M combinations, M = 1 to 4); members outside the region are deleted
 * too (molecule 2 with an outside atom: atoms 3, 4 and 8 were all deleted).
 * Deletion happens on the step that is a multiple of N, before that step's
 * thermo output (measured: the thermo atom count at step 3 is already reduced
 * with N = 3). If M exceeds the eligible count, every eligible atom goes
 * (measured: 6 of 10 atoms removed with M = 20).
 *
 * Implementation: deletion runs in postIntegrate (before the neighbour
 * decision), as fix deposit inserts atoms; preExchange is a stub so the
 * accelerated backends are not chosen (see fix pour.ts).
 */

export class FixEvaporate extends Fix {
  readonly style = 'evaporate';
  readonly N: number;
  readonly M: number;
  readonly regionId: string;
  private readonly rng: RanPark;
  private molecule = false;
  private deleted = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('usage: fix ID group-ID evaporate N M region-ID seed [molecule yes|no]');
    this.N = posInt(args[0], 'N');
    this.M = posInt(args[1], 'M');
    this.regionId = args[2];
    const seed = posInt(args[3], 'seed');
    this.rng = insertionStream(seed);
    let molecule = false;
    for (let k = 4; k < args.length; k += 2) {
      if (args[k] !== 'molecule') throw new StyleError(`fix evaporate: unknown keyword '${args[k]}' (only molecule)`);
      const v = args[k + 1];
      if (v !== 'yes' && v !== 'no') throw new StyleError(`fix evaporate molecule must be yes or no (got '${v ?? ''}')`);
      molecule = v === 'yes';
    }
    this.molecule = molecule;
    sys.region(this.regionId); // fail early on an unknown region
    this.scalarFlag = true;
    this.extscalar = 0;
  }

  /** Acts only on every N-th step, before the neighbour decision and force evaluation. */
  postIntegrate(): void {
    const s = this.sys.state;
    if (s.step % this.N !== 0) return;
    this.evaporate();
  }

  /** The run loop calls preExchange only on neighbour rebuilds; it is a stub (see fix pour). */
  preExchange(): void { /* deletion is done in postIntegrate */ }

  private evaporate(): void {
    const sys = this.sys;
    const s = sys.state;
    const reg = sys.region(this.regionId);
    // eligible atoms (group and region), in native atom order
    const elig: number[] = [];
    const order = s.order.length === s.n ? s.order : Int32Array.from({ length: s.n }, (_, k) => k);
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & this.groupBit)) continue;
      if (!reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
      elig.push(i);
    }
    const del = new Uint8Array(s.n);
    let count = 0;
    const removeFromList = (atom: number): void => {
      const k = elig.indexOf(atom);
      if (k < 0) return;
      elig[k] = elig[elig.length - 1];
      elig.pop();
    };
    while (count < this.M && elig.length > 0) {
      const pick = elig[Math.floor(this.rng.uniform() * elig.length)];
      const mol = this.molecule ? s.molecule[pick] : 0;
      if (mol) {
        // every atom of the molecule (inside or outside the region) is deleted and counted
        for (let i = 0; i < s.n; i++) {
          if (s.molecule[i] !== mol) continue;
          del[i] = 1;
          count++;
          removeFromList(i);
        }
      } else {
        del[pick] = 1;
        count++;
        removeFromList(pick);
      }
    }
    if (count === 0) return;
    // native storage order after the deletion (measured, see the header): the deleted atoms
    // are taken in decreasing storage position and each slot is refilled by the last atom
    // of the list; this differs from the walk that delete_atoms uses (sys.deleteAtoms), so the
    // order is recomputed here and written over the one deleteAtoms leaves.
    const st = Array.from(order);
    const goneOld = [...order].filter((a) => del[a] === 1);
    goneOld.sort((a, b) => st.indexOf(b) - st.indexOf(a));
    for (const a of goneOld) {
      const i = st.indexOf(a);
      const last = st.pop()!;
      if (i < st.length) st[i] = last;
    }
    const rank = new Int32Array(s.n).fill(-1);
    let kept = 0;
    for (let i = 0; i < s.n; i++) if (!del[i]) rank[i] = kept++;
    const n = sys.deleteAtoms(del);
    s.order = Int32Array.from(st, (a) => rank[a]);
    this.deleted += n;
  }

  computeScalar(): number { return this.deleted; }
}
