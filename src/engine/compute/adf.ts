import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseInt_, parseNum } from '../force/util';

/*
 * compute ID group-ID adf Nbin itype1 jtype1 ktype1 Rjinner1 Rjouter1 Rkinner1 Rkouter1 ...
 * — docs.lammps.org/compute_adf.html:
 * "Define a computation that calculates one or more angular distribution functions
 * (ADF) for a group of particles.  Each ADF is calculated in histogram form
 * by measuring the angle formed by a central atom and two neighbor atoms and
 * binning these angles into *Nbin* bins."
 * "Only neighbors for which *Rinner* < *R* < *Router* are counted, where
 * *Rinner* and *Router* are specified separately for the first and second
 * neighbor atom in each requested ADF." The description's criteria list adds: "atoms I,J,K are
 * all in the specified compute group", "the distance between atoms I,J is
 * between Rjinner and Rjouter", "the distance between atoms I,K is between
 * Rkinner and Rkouter", "the type of the I atom matches itypeN",
 * "atoms I,J,K are distinct", and the matching type tests for J and K.
 * "Each unique angle
 * satisfying the above criteria is counted only once, regardless of whether
 * either or both of the neighbor atoms making up the angle appear in both the
 * J and K lists." "The *ordinate* optional keyword determines whether the bins
 * are of uniform angular size from zero to 180 (\ *degree*\ ), zero to Pi
 * (\ *radian*\ ), or the cosine of the angle uniform in the range [-1,1]
 * (\ *cosine*\ )."
 * Default: "The keyword default is ordinate = degree." "If no *itypeN*, *jtypeN*,
 * *ktypeN* settings are specified, then LAMMPS will generate a single ADF for
 * all atoms in the group. The inner cutoff is set to zero and the outer cutoff
 * is set to the force cutoff."
 * Output info: "This compute calculates a global array with the number of rows =" Nbins, with 1 + 2 Ntriples
 * columns. "The first column has the bin coordinate (angle-related ordinate at midpoint of bin)." "The values
 * in the first ADF column are normalized numbers :math:`\ge 0.0`, whose integral w.r.t. the ordinate is 1,";
 * the second ADF column values "are the cumulative density distribution of angles per atom."
 *
 * Measured with native LAMMPS (black box; probes in plans/scratch/adf):
 * - despite the intro's strict wording, R = Rinner and R = Router are BOTH
 *   included: the shell test is Rinner <= R <= Router;
 * - bins are half-open [lo + b*w, lo + (b+1)*w) with w = range/Nbin
 *   (range = 180, Pi or 2; lo = 0, or -1 for cosine), so an ordinate at the
 *   top of the range (an angle of 180 in degree/radian, cosine +1) is DROPPED,
 *   not clamped into the last bin;
 * - the first ADF value is hist_b / (Nangles * w), where Nangles is the sum of
 *   the histogram counts for that triple over all bins (angles dropped by the
 *   range test are not part of Nangles); the second is cumulative_hist_b /
 *   Ncentral, with Ncentral the number of group atoms whose type is in the
 *   itype range;
 * - the J and K roles are symmetric: a neighbor pair is counted when either
 *   ordering matches (jtype, Rj) for one atom and (ktype, Rk) for the other,
 *   independent of neighbor-list order;
 * - Rinner must be >= 0 and Router > Rinner (native rejects 1.0 0.5, 0.5 0.5
 *   and -0.1 2.0); an unknown keyword or a bad ordinate is an error.
 */

type Ordinate = 'degree' | 'radian' | 'cosine';

interface Triple {
  /** 1-based inclusive type ranges. */
  ilo: number; ihi: number; jlo: number; jhi: number; klo: number; khi: number;
  rjinner: number; rjouter: number; rkinner: number; rkouter: number;
}

export class ComputeADF extends Compute {
  readonly style = 'adf';
  private readonly nbin: number;
  private ordinate: Ordinate = 'degree';
  private triples: Triple[] = [];
  /** No triple given: one all-types histogram whose outer cutoff is the force cutoff. */
  private useForceCut = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.nbin = parseInt_(args[0], 'compute adf Nbin');
    if (this.nbin < 1) throw new StyleError(`compute adf Nbin must be >= 1 (got ${args[0]})`);
    const ntypes = sys.state.ntypes;

    const typeRange = (w: string): [number, number] => {
      const wild = /^(\d*)\*(\d*)$/.exec(w);
      if (wild) {
        const lo = wild[1] === '' ? 1 : Number(wild[1]);
        const hi = wild[2] === '' ? ntypes : Number(wild[2]);
        return [lo, hi];
      }
      if (/^\d+$/.test(w)) {
        const t = Number(w);
        return [t, t];
      }
      throw new StyleError(`compute adf: invalid atom type '${w}' (number, asterisk range, or type label; type labels are not supported by this engine)`);
    };

    const nums: string[] = [];
    for (let k = 1; k < args.length;) {
      if (args[k] === 'ordinate') {
        const v = args[k + 1];
        if (v !== 'degree' && v !== 'radian' && v !== 'cosine') {
          throw new StyleError(`compute adf: unknown ordinate '${v}' (degree, radian or cosine)`);
        }
        this.ordinate = v;
        k += 2;
        continue;
      }
      nums.push(args[k]);
      k++;
    }

    if (nums.length % 7 !== 0) {
      const at = Math.floor(nums.length / 7) * 7;
      throw new StyleError(`compute adf: unknown keyword or argument '${nums[at]}' (expected groups of itype jtype ktype Rjinner Rjouter Rkinner Rkouter)`);
    }
    for (let k = 0; k < nums.length; k += 7) {
      const [ilo, ihi] = typeRange(nums[k]);
      const [jlo, jhi] = typeRange(nums[k + 1]);
      const [klo, khi] = typeRange(nums[k + 2]);
      for (const t of [ilo, ihi, jlo, jhi, klo, khi]) {
        if (t < 1 || t > ntypes) throw new StyleError(`compute adf: atom type ${t} is out of range 1..${ntypes}`);
      }
      if (ilo > ihi || jlo > jhi || klo > khi) {
        throw new StyleError(`compute adf: empty type range in '${nums[k]} ${nums[k + 1]} ${nums[k + 2]}'`);
      }
      const rjinner = parseNum(nums[k + 3], 'compute adf Rjinner');
      const rjouter = parseNum(nums[k + 4], 'compute adf Rjouter');
      const rkinner = parseNum(nums[k + 5], 'compute adf Rkinner');
      const rkouter = parseNum(nums[k + 6], 'compute adf Rkouter');
      if (!(rjinner >= 0) || !(rjouter > rjinner)) {
        throw new StyleError(`compute adf: illegal j-cutoff, need 0 <= Rjinner < Rjouter (got ${nums[k + 3]} ${nums[k + 4]})`);
      }
      if (!(rkinner >= 0) || !(rkouter > rkinner)) {
        throw new StyleError(`compute adf: illegal k-cutoff, need 0 <= Rkinner < Rkouter (got ${nums[k + 5]} ${nums[k + 6]})`);
      }
      this.triples.push({ ilo, ihi, jlo, jhi, klo, khi, rjinner, rjouter, rkinner, rkouter });
    }
    if (!this.triples.length) {
      this.triples.push({ ilo: 1, ihi: ntypes, jlo: 1, jhi: ntypes, klo: 1, khi: ntypes, rjinner: 0, rjouter: 0, rkinner: 0, rkouter: 0 });
      this.useForceCut = true;
    }

    this.arrayFlag = true;
    this.sizeArrayRows = this.nbin;
    this.sizeArrayCols = 1 + 2 * this.triples.length;
    this.array = new Float64Array(this.sizeArrayRows * this.sizeArrayCols);
  }

  /** Maximum pairwise force cutoff over all type pairs (pair "cut" array). */
  private forceCutoff(): number {
    const pair = this.sys.ff.pair;
    if (!pair) return 0;
    let mx = 0;
    for (let k = 0; k < pair.cut.length; k++) if (pair.cut[k] > mx) mx = pair.cut[k];
    return mx;
  }

  protected computeArray(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;

    let cutmax = 0;
    for (const t of this.triples) cutmax = Math.max(cutmax, t.rjouter, t.rkouter);
    if (this.useForceCut) {
      cutmax = this.forceCutoff();
      for (const t of this.triples) { t.rjouter = cutmax; t.rkouter = cutmax; }
    }
    if (!(cutmax > 0)) {
      throw new StyleError('compute adf: no pairwise force cutoff is defined; give explicit outer cutoffs');
    }
    if (nb.cutghost > 0 && nb.cutghost < cutmax + nb.skin - 1e-9) {
      // compute_adf.html: "If you request any outer cutoff Router > force
      // cutoff ... you must ensure ghost atom information out to the largest
      // value of Router + skin is communicated, via the comm_modify cutoff
      // command, else the ADF computation cannot be performed, and LAMMPS will
      // give an error message."
      throw new StyleError(`compute adf cutoff ${cutmax} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
    }

    const counts = new Float64Array(s.ntypes + 1);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) counts[s.type[i]]++;

    const range = this.ordinate === 'degree' ? 180 : this.ordinate === 'radian' ? Math.PI : 2;
    const lo = this.ordinate === 'cosine' ? -1 : 0;
    const width = range / this.nbin;
    const ntr = this.triples.length;
    const hist = new Float64Array(ntr * this.nbin);

    const list = nb.occasionalList(s, cutmax, true, s.ntypes);
    const xa = nb.xall, ta = nb.typeall, owner = nb.owner;
    for (let i = 0; i < list.inum; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const ti = ta[i];
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      const start = list.firstneigh[i];
      const end = start + list.numneigh[i];
      for (let m = 0; m < ntr; m++) {
        const t = this.triples[m];
        if (ti < t.ilo || ti > t.ihi) continue;
        const rj2i = t.rjinner * t.rjinner, rj2o = t.rjouter * t.rjouter;
        const rk2i = t.rkinner * t.rkinner, rk2o = t.rkouter * t.rkouter;
        for (let a = start; a < end; a++) {
          const ja = list.neighbors[a];
          if (!(s.mask[owner[ja]] & this.groupBit)) continue;
          const tja = ta[ja];
          const dxa = xa[3 * ja] - xi, dya = xa[3 * ja + 1] - yi, dza = xa[3 * ja + 2] - zi;
          const r2a = dxa * dxa + dya * dya + dza * dza;
          const aJ = tja >= t.jlo && tja <= t.jhi && r2a >= rj2i && r2a <= rj2o;
          const aK = tja >= t.klo && tja <= t.khi && r2a >= rk2i && r2a <= rk2o;
          if (!aJ && !aK) continue;
          for (let b = a + 1; b < end; b++) {
            const jb = list.neighbors[b];
            if (!(s.mask[owner[jb]] & this.groupBit)) continue;
            const tjb = ta[jb];
            const dxb = xa[3 * jb] - xi, dyb = xa[3 * jb + 1] - yi, dzb = xa[3 * jb + 2] - zi;
            const r2b = dxb * dxb + dyb * dyb + dzb * dzb;
            const bJ = tjb >= t.jlo && tjb <= t.jhi && r2b >= rj2i && r2b <= rj2o;
            const bK = tjb >= t.klo && tjb <= t.khi && r2b >= rk2i && r2b <= rk2o;
            if (!((aJ && bK) || (aK && bJ))) continue;
            const cos = (dxa * dxb + dya * dyb + dza * dzb) / Math.sqrt(r2a * r2b);
            const val = this.ordinate === 'cosine'
              ? cos
              : (this.ordinate === 'degree' ? Math.acos(clamp(cos)) * (180 / Math.PI) : Math.acos(clamp(cos)));
            const bin = Math.floor((val - lo) / width);
            if (bin < 0 || bin >= this.nbin) continue;
            hist[m * this.nbin + bin]++;
          }
        }
      }
    }

    const cols = this.sizeArrayCols;
    const array = this.array;
    for (let m = 0; m < ntr; m++) {
      const t = this.triples[m];
      let countI = 0;
      for (let ty = t.ilo; ty <= t.ihi; ty++) countI += counts[ty];
      let total = 0;
      for (let b = 0; b < this.nbin; b++) total += hist[m * this.nbin + b];
      let cum = 0;
      for (let b = 0; b < this.nbin; b++) {
        const h = hist[m * this.nbin + b];
        if (total > 0) array[b * cols + 1 + 2 * m] = h / (total * width);
        if (countI > 0) cum += h / countI;
        array[b * cols] = lo + (b + 0.5) * width;
        array[b * cols + 2 + 2 * m] = cum;
      }
    }
  }
}

const clamp = (c: number): number => (c < -1 ? -1 : c > 1 ? 1 : c);
