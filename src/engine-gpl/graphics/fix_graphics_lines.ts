/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/fix_graphics_lines.cpp and fix_graphics_lines.h (class FixGraphicsLines, fix graphics/lines:
 * "fix ID group-ID graphics/lines Nevery Nrepeat Nfreq Nlength"). Copyright (2003) Sandia Corporation, distributed
 * under the GNU General Public License (the LAMMPS file headers). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_graphics_lines.html): the fix draws the recent trajectory of each atom
 * as line segments. The positions are averaged over Nrepeat samples taken every Nevery steps before each multiple
 * of Nfreq (the fix ave/atom rule of docs.lammps.org/fix_ave_atom.html: "values on timesteps 90, 92, 94, 96, 98,
 * and 100 ... if Nevery=2, Nrepeat=6, and Nfreq=100"), and the last Nlength averages are kept per atom. A segment
 * across a periodic boundary is skipped, as the C++ does (|dx| >= half the box). The objects are cylinders with
 * zero diameter, so the dump image "fix ID const flag1 flag2" keyword gives their width (flag2) and sphere caps
 * (flag1); the scalar is the current number of stored averages (compute_scalar).
 *
 * The C++ ave/atom and store/atom helpers are replaced by a per-atom ring keyed by atom id (the atom order of the
 * browser engine can change between snapshots); the averaging and the segment rules are as in the C++.
 */
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import type { ImageObject, Vec3 } from '../../engine/output/image_objects';
import { parseInteger } from './common';

export class FixGraphicsLines extends Fix {
  readonly style = 'graphics/lines';
  private readonly nrepeat: number;
  private readonly nfreq: number;
  private readonly nlength: number;
  /** Stored averages: per atom id, nlength positions (3 values each). */
  private readonly hist = new Map<number, Float64Array>();
  /** Running sums of the current averaging window per atom id. */
  private readonly sums = new Map<number, Float64Array>();
  private nvalues = 0;
  private ivalue = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError(`usage: fix ${id} group-ID graphics/lines Nevery Nrepeat Nfreq Nlength`);
    this.nevery = parseInteger(args[0], `fix ${id} graphics/lines Nevery`);
    this.nrepeat = parseInteger(args[1], `fix ${id} graphics/lines Nrepeat`);
    this.nfreq = parseInteger(args[2], `fix ${id} graphics/lines Nfreq`);
    this.nlength = parseInteger(args[3], `fix ${id} graphics/lines Nlength`);
    if (this.nevery <= 0) throw new StyleError(`Illegal fix graphics/lines nevery value: ${this.nevery}`);
    if (this.nrepeat <= 0) throw new StyleError(`Illegal fix graphics/lines nrepeat value: ${this.nrepeat}`);
    if (this.nfreq <= 0) throw new StyleError(`Illegal fix graphics/lines nfreq value: ${this.nfreq}`);
    if (this.nlength <= 0) throw new StyleError(`Illegal fix graphics/lines nlength value: ${this.nlength}`);
    if (this.nfreq % this.nevery || this.nrepeat * this.nevery > this.nfreq) throw new StyleError('Inconsistent fix graphics/lines nevery/nrepeat/nfreq values');
    this.scalarFlag = true;
    this.extscalar = 0;
  }

  /** Is this step one of the samples of an averaging window (fix ave/atom: the last Nrepeat multiples of Nevery). */
  private isSample(step: number): boolean {
    if (step % this.nevery) return false;
    const next = Math.ceil(step / this.nfreq) * this.nfreq;
    return step >= next - (this.nrepeat - 1) * this.nevery;
  }

  /** Adds the current unwrapped positions of the group to the window sums. */
  private sampleNow(): void {
    const s = this.sys.state;
    const u = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      let acc = this.sums.get(s.id[i]);
      if (!acc) { acc = new Float64Array(3); this.sums.set(s.id[i], acc); }
      acc[0] += u[0]; acc[1] += u[1]; acc[2] += u[2];
    }
  }

  /** Setup: the samples of the first step; as in the C++, the setup call stores no history (firstflag). */
  setup(): void {
    this.sums.clear();
    const step = this.sys.state.step;
    if (this.isSample(step)) this.sampleNow();
  }

  endOfStep(): void {
    const s = this.sys.state;
    const step = s.step;
    if (this.isSample(step)) this.sampleNow();
    if (step % this.nfreq) return;
    // one window is complete: store the averages, remapped into the box (domain->remap in the C++)
    const p = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const acc = this.sums.get(s.id[i]);
      if (!acc) continue;
      let h = this.hist.get(s.id[i]);
      if (!h) { h = new Float64Array(3 * this.nlength); this.hist.set(s.id[i], h); }
      for (let d = 0; d < 3; d++) p[d] = acc[d] / this.nrepeat;
      this.sys.geom.remapPoint(p);
      h[3 * this.ivalue] = p[0]; h[3 * this.ivalue + 1] = p[1]; h[3 * this.ivalue + 2] = p[2];
    }
    this.sums.clear();
    if (this.nvalues < this.nlength) this.nvalues++;
    this.ivalue = (this.ivalue + 1) % this.nlength;
  }

  computeScalar(): number {
    return this.nvalues;
  }

  /** The line segments of the stored history (fix_graphics_lines.cpp end_of_step, the segment rule). */
  imageObjects(): ImageObject[] {
    const s = this.sys.state;
    const out: ImageObject[] = [];
    if (this.nvalues === 0) return out;
    const box = s.box;
    const half = [0, 1, 2].map((d) => 0.5 * (box.hi[d] - box.lo[d]));
    const zoffs = s.dimension === 2 ? half[2] : 0;
    const jump = (a: ArrayLike<number>, ao: number, b: ArrayLike<number>, bo: number): boolean =>
      Math.abs(a[ao] - b[bo]) >= half[0] || Math.abs(a[ao + 1] - b[bo + 1]) >= half[1] || Math.abs(a[ao + 2] - b[bo + 2]) >= half[2];
    const seg = (p: ArrayLike<number>, po: number, q: ArrayLike<number>, qo: number, type: number): void => {
      out.push({ kind: 'cylinder', type, p1: [p[po], p[po + 1], p[po + 2] + zoffs] as Vec3, p2: [q[qo], q[qo + 1], q[qo + 2] + zoffs] as Vec3, diameter: 0 });
    };
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const h = this.hist.get(s.id[i]);
      if (!h) continue;
      const x = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
      // current position to the newest stored average
      const j2 = (this.ivalue - 1 + this.nvalues) % this.nvalues;
      if (!jump(x, 0, h, 3 * j2)) seg(h, 3 * j2, x, 0, s.type[i]);
      // the stored averages, oldest to newest
      for (let jj = 1; jj < this.nvalues; jj++) {
        const j1 = (this.ivalue + jj - 1) % this.nvalues, k2 = (this.ivalue + jj) % this.nvalues;
        if (!jump(h, 3 * j1, h, 3 * k2)) seg(h, 3 * j1, h, 3 * k2, s.type[i]);
      }
    }
    return out;
  }
}

export const makeFixGraphicsLines: FixFactory = (sys, id, group, args) => new FixGraphicsLines(sys, id, group, args);
