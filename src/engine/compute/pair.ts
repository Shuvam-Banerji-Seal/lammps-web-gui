import { Compute } from './compute';
import { StyleError, type Pair } from '../force/types';
import { PairHybrid } from '../force/pair/hybrid';
import type { System } from '../system';

/*
 * compute ID group-ID pair pstyle [nsub] [evalue] — docs.lammps.org/compute_pair.html:
 * "The specified *pstyle* must be a pair style used in your simulation either by itself or
 * as a sub-style in a pair_style hybrid or hybrid/overlay command. If the sub-style is used
 * more than once, an additional number *nsub* has to be specified in order to choose which
 * instance of the sub-style will be used by the compute."
 * "The *evalue* setting is optional. ... If *evalue* is blank or specified as *epair*, then
 * *epair* is stored as a global scalar by this compute. ... If *evalue* is specified as
 * *evdwl* or *ecoul*, then just that portion of the energy is stored as a global scalar."
 * "The energy returned by the *evdwl* keyword does not include tail corrections, even if they
 * are enabled via the pair_modify command."
 * "The scalar and vector values calculated by this compute are "extensive"."
 * "The group specified for this command is **ignored**."
 *
 * Measured with native LAMMPS (black box):
 *  - an unknown pstyle, or a repeated sub-style without nsub, or nsub above the number of
 *    listings, fails when the compute is defined; a sub-style listed once accepts any nsub >= 1;
 *  - pstyle equal to the hybrid style itself gives the hybrid total and no vector;
 *  - a hybrid/scaled sub-style reports its unscaled energy (the scale applies only in pe);
 *  - the vector c_ID[1] (count) and c_ID[2] (energy) of hbond/dreiding/lj, and the scalar, are
 *    extensive: thermo_modify norm yes divides both by the atom count.
 *
 * Vector styles (docs.lammps.org): hbond/dreiding/* [count, energy], gauss [occupancy],
 * reaxff [14 values], ilp/graphene/hbn, ilp/tmd, kolmogorov/crespi/full, saip/metal and
 * aip/water/2dm [2 values]. The engine provides only the hbond/dreiding vector; the others
 * raise a StyleError when the vector is accessed.
 */

const VECTOR_LENGTH: Record<string, number> = {
  'hbond/dreiding/lj': 2,
  'hbond/dreiding/morse': 2,
  'hbond/dreiding/lj/angleoffset': 2,
  'hbond/dreiding/morse/angleoffset': 2,
  gauss: 1,
  reaxff: 14,
  'ilp/graphene/hbn': 2,
  'ilp/tmd': 2,
  'kolmogorov/crespi/full': 2,
  'saip/metal': 2,
  'aip/water/2dm': 2,
};
const ENGINE_VECTOR = new Set(['hbond/dreiding/lj', 'hbond/dreiding/morse', 'hbond/dreiding/lj/angleoffset', 'hbond/dreiding/morse/angleoffset']);
const EVALUES = ['epair', 'evdwl', 'ecoul'];

export class ComputePair extends Compute {
  readonly style = 'pair';
  private pstyle: string;
  /** nsub as given (0 = not given). */
  private nsub = 0;
  private evalue = 'epair';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const w = [...args];
    if (w.length === 0) throw new StyleError(`compute ${id} pair needs a pair style name`);
    this.pstyle = w.shift()!;
    if (w.length && /^\d+$/.test(w[0])) {
      this.nsub = Number(w.shift());
      if (this.nsub < 1) throw new StyleError(`compute ${id} pair: nsub must be 1 or more (got ${this.nsub})`);
    }
    if (w.length) {
      if (!EVALUES.includes(w[0])) throw new StyleError(`compute ${id} pair: unknown keyword '${w[0]}' (epair, evdwl or ecoul)`);
      this.evalue = w.shift()!;
    }
    if (w.length) throw new StyleError(`compute ${id} pair: unexpected argument '${w[0]}' after evalue`);
    this.scalarFlag = true;
    this.extscalar = 1;
    const len = VECTOR_LENGTH[this.pstyle];
    this.vectorFlag = len !== undefined;
    this.sizeVector = len ?? 0;
    this.extvector = 1;
    // the style must be in use when the compute is defined
    this.source();
  }

  /**
   * The style whose tallies the compute reads. For a hybrid sub-style it is the record kept by the
   * hybrid for its last evaluation; for the pair style itself rec is null.
   */
  private source(): { style: Pair; rec: { evdwl: number; ecoul: number; evaluated: boolean } | null } {
    const pf = this.sys.ff.pair;
    if (!pf) throw new StyleError(`compute ${this.id} pair: no pair style is defined`);
    if (pf instanceof PairHybrid && pf.name !== this.pstyle) {
      const listings = pf.subs.filter((s) => s.name === this.pstyle).length;
      if (listings === 0) throw new StyleError(`compute ${this.id} pair: pair style ${this.pstyle} is not a sub-style of pair_style ${pf.name}`);
      let instance = 1;
      if (listings > 1) {
        if (this.nsub < 1 || this.nsub > listings) throw new StyleError(`compute ${this.id} pair: sub-style ${this.pstyle} is listed ${listings} times; give its number 1..${listings} as nsub`);
        instance = this.nsub;
      }
      const rec = pf.subEnergy(this.pstyle, instance);
      if (!rec) throw new StyleError(`compute ${this.id} pair: sub-style ${this.pstyle} not found`);
      return { style: rec.style, rec };
    }
    if (pf.name !== this.pstyle) throw new StyleError(`compute ${this.id} pair: pair style ${this.pstyle} is not in use (pair_style ${pf.name})`);
    if (pf instanceof PairHybrid && this.nsub) throw new StyleError(`compute ${this.id} pair: nsub is not supported for the total of pair_style ${pf.name}`);
    return { style: pf, rec: null };
  }

  /** Energies of the last evaluation: evdwl without tail corrections, and ecoul. */
  private energies(): { evdwl: number; ecoul: number } {
    const a = this.sys.forces();
    const { style, rec } = this.source();
    if (rec) {
      if (!rec.evaluated) throw new StyleError(`compute ${this.id} pair: sub-style ${style.name} did not run in the last evaluation (hybrid/scaled factor 0 is not supported here)`);
      return { evdwl: rec.evdwl, ecoul: rec.ecoul };
    }
    // the pair term plus its tail correction is what forces() reports as evdwl (ff.compute adds the tail last)
    const tail = this.sys.ff.etailV / this.sys.geom.volume(this.sys.state.dimension);
    return { evdwl: a.evdwl - tail, ecoul: a.ecoul };
  }

  protected computeScalar(): number {
    const e = this.energies();
    if (this.evalue === 'evdwl') return e.evdwl;
    if (this.evalue === 'ecoul') return e.ecoul;
    return e.evdwl + e.ecoul;
  }

  protected computeVector(): void {
    if (!this.vectorFlag) throw new StyleError(`compute ${this.id} does not calculate a global vector`);
    this.sys.forces();
    const { style, rec } = this.source();
    if (rec && !rec.evaluated) throw new StyleError(`compute ${this.id} pair: sub-style ${style.name} did not run in the last evaluation (hybrid/scaled factor 0 is not supported here)`);
    if (!ENGINE_VECTOR.has(this.pstyle)) {
      throw new StyleError(`compute ${this.id} pair: the global vector of pair style ${this.pstyle} (${this.sizeVector} values) is not provided by the browser engine`);
    }
    const count = style.extract('hbond_count');
    const energy = style.extract('hbond_energy');
    if (typeof count !== 'number' || typeof energy !== 'number') {
      throw new StyleError(`compute ${this.id} pair: pair style ${style.name} does not provide hbond_count and hbond_energy`);
    }
    this.vector = Float64Array.from([count, energy]);
  }
}
