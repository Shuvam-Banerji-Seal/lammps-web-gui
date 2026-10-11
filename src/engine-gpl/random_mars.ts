/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/random_mars.cpp and
 * src/random_mars.h (class RanMars: the constructor, uniform(), select_subset()).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Marsaglia's lagged Fibonacci generator with carry (the seeding of the constructor is the source's), and the
 * subset selection of set type/ratio and type/subset for one processor (the reduction over processors is trivial).
 */
import { StyleError } from '../engine/force/types';

const ADD = 0;
const SUBTRACT = 1;

export class RanMars {
  private readonly u = new Float64Array(98);
  private i97 = 97;
  private j97 = 33;
  private c: number;
  private readonly cd = 7654321.0 / 16777216.0;
  private readonly cm = 16777213.0 / 16777216.0;

  /** RanMars::RanMars(lmp, seed): "Invalid seed ... for the Marsaglia random # generator" outside 1..900000000. */
  constructor(seed: number) {
    if (!(seed > 0 && seed <= 900000000)) throw new StyleError(`Invalid seed ${seed} for the Marsaglia random # generator`);
    const ij = Math.trunc((seed - 1) / 30082);
    const kl = seed - 1 - 30082 * ij;
    let i = (Math.trunc(ij / 177) % 177) + 2;
    let j = (ij % 177) + 2;
    let k = (Math.trunc(kl / 169) % 178) + 1;
    let l = kl % 169;
    for (let ii = 1; ii <= 97; ii++) {
      let s = 0.0;
      let t = 0.5;
      for (let jj = 1; jj <= 24; jj++) {
        const m = (((i * j) % 179) * k) % 179;
        i = j;
        j = k;
        k = m;
        l = (53 * l + 1) % 169;
        if ((l * m) % 64 >= 32) s = s + t;
        t = 0.5 * t;
      }
      this.u[ii] = s;
    }
    this.c = 362436.0 / 16777216.0;
    this.uniform();
  }

  /** RanMars::uniform(): one number in [0, 1). */
  uniform(): number {
    let uni = this.u[this.i97] - this.u[this.j97];
    if (uni < 0.0) uni += 1.0;
    this.u[this.i97] = uni;
    this.i97--;
    if (this.i97 === 0) this.i97 = 97;
    this.j97--;
    if (this.j97 === 0) this.j97 = 97;
    this.c -= this.cd;
    if (this.c < 0.0) this.c += this.cm;
    uni -= this.c;
    if (uni < 0.0) uni += 1.0;
    return uni;
  }

  /**
   * RanMars::select_subset(ntarget, nmine, mark, next) on one processor: mark[i] = 1 for ntarget of the nmine
   * items, chosen by flipping random items between two linked lists. next is a work array of length nmine.
   */
  selectSubset(ntarget: number, nmine: number, mark: Uint8Array, next: Int32Array): void {
    const active = [nmine, 0];
    const first = [0, -1];
    const activeall = [nmine, 0];
    for (let i = 0; i < nmine; i++) mark[i] = 0;
    for (let i = 0; i < nmine; i++) next[i] = i + 1;
    if (nmine > 0) next[nmine - 1] = -1;
    let nmark = 0;
    while (nmark !== ntarget) {
      // ADD or SUBTRACT from the current nmark; thresh = desired flips / size of the active set, bounded for accuracy
      let mode: number;
      let thresh: number;
      if (ntarget - nmark > 0) {
        mode = ADD;
        thresh = (1.0 * (ntarget - nmark)) / activeall[mode];
      } else {
        mode = SUBTRACT;
        thresh = (1.0 * (nmark - ntarget)) / activeall[mode];
      }
      thresh = Math.max(thresh, 0.01);
      thresh = Math.min(thresh, 0.99);
      const newactive = [0, 0];
      const newfirst = [-1, -1];
      const newlast = [-1, -1];
      const newvalue = mode === ADD ? 1 : 0;
      let index = first[mode];
      let nflip = 0;
      while (nmine > 0 && index >= 0) {
        if (this.uniform() < thresh) {
          mark[index] = newvalue;
          nflip++;
        }
        const oldindex = index;
        index = next[index];
        const which = mark[oldindex];
        newactive[which]++;
        if (newfirst[which] < 0) newfirst[which] = oldindex;
        if (newlast[which] >= 0) next[newlast[which]] = oldindex;
        newlast[which] = oldindex;
        next[oldindex] = -1;
      }
      active[0] = newactive[0];
      active[1] = newactive[1];
      first[0] = newfirst[0];
      first[1] = newfirst[1];
      activeall[0] = active[0];
      activeall[1] = active[1];
      if (mode === ADD) nmark += nflip;
      else nmark -= nflip;
    }
  }
}
