import { Compute, groupCount } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';
import { parseInt_, parseNum } from '../force/util';

/*
 * Global computes of wave 2c: com, gyration, msd, vacf, rdf. Each class cites
 * and quotes the docs.lammps.org page it implements; the rdf normalization
 * details that the page leaves out were measured with native LAMMPS
 * (/home/roy/.local/bin/lmp, probes in /tmp/opencode/w2cglob) and are marked
 * "measured" below.
 */

/*
 * compute ID group-ID com — docs.lammps.org/compute_com.html:
 *   compute ID group-ID com
 * "Define a computation that calculates the center-of-mass of the group
 * of atoms, including all effects due to atoms passing through periodic
 * boundaries." "A vector of three quantities is calculated by this compute,
 * which are the (x,y,z) coordinates of the center of mass." "The coordinates
 * of an atom contribute to the center-of-mass in "unwrapped" form, by using
 * the image flags associated with each atom." Output info: "This compute
 * calculates a global vector of length 3 ... The vector values are
 * "intensive". The vector values will be in distance units."
 * xcm = sum_i m_i r_i^unwrapped / sum_i m_i over atoms in the group.
 */
export class ComputeCom extends Compute {
  readonly style = 'com';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`compute com takes no arguments (got ${args.join(' ')})`);
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extvector = 0;
    this.vector = new Float64Array(3);
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const u = [0, 0, 0];
    let mx = 0, my = 0, mz = 0, msum = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      mx += m * u[0]; my += m * u[1]; mz += m * u[2];
      msum += m;
    }
    this.vector[0] = mx / msum;
    this.vector[1] = my / msum;
    this.vector[2] = mz / msum;
  }
}

/*
 * compute ID group-ID gyration — docs.lammps.org/compute_gyration.html:
 *   compute ID group-ID gyration
 *   R_g^2 = \frac{1}{M} \sum_i m_i (r_i - r_{\text{cm}})^2
 * "where :math:`M` is the total mass of the group, :math:`r_{\text{cm}}` is the
 * center-of-mass position of the group, and the sum is over all atoms in
 * the group." "A :math:`R_g^2` tensor, stored as a 6-element vector, is also
 * calculated" (components xx, yy, zz, xy, xz, yz). Atom coordinates enter
 * unwrapped, using each atom's image flags. Output info: "This compute
 * calculates a global scalar (:math:`R_g`) and a global vector of length 6
 * (:math:`R_g^2` tensor)" — the scalar is the square root
 * of the sum of the diagonal tensor components. "The scalar and vector
 * values calculated by this compute are "intensive"."
 */
export class ComputeGyration extends Compute {
  readonly style = 'gyration';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`compute gyration takes no arguments (got ${args.join(' ')})`);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extscalar = 0;
    this.extvector = 0;
    this.vector = new Float64Array(6);
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const u = [0, 0, 0];
    let mx = 0, my = 0, mz = 0, msum = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      g.unwrap(s.x, s.image, i, u);
      mx += m * u[0]; my += m * u[1]; mz += m * u[2];
      msum += m;
    }
    const cmx = mx / msum, cmy = my / msum, cmz = mz / msum;
    const t = this.vector;
    t.fill(0);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      g.unwrap(s.x, s.image, i, u);
      const dx = u[0] - cmx, dy = u[1] - cmy, dz = u[2] - cmz;
      t[0] += m * dx * dx; t[1] += m * dy * dy; t[2] += m * dz * dz;
      t[3] += m * dx * dy; t[4] += m * dx * dz; t[5] += m * dy * dz;
    }
    for (let c = 0; c < 6; c++) t[c] /= msum;
  }

  protected computeScalar(): number {
    const t = this.vectorValues();
    return Math.sqrt(t[0] + t[1] + t[2]);
  }
}

/*
 * compute ID group-ID msd keyword values ... — docs.lammps.org/compute_msd.html:
 *   compute ID group-ID msd keyword values ...
 * "keyword = com or average"; "com value = yes or no"; "average value = yes
 * or no". "A vector of four quantities is calculated by this compute. The
 * first three elements of the vector are the squared dx, dy, and dz
 * displacements, summed and averaged over atoms in the group. The fourth
 * element is the total squared displacement (i.e., dx^2 + dy^2 + dz^2),
 * summed and averaged over atoms in the group." "The displacement of an atom
 * is from its reference position. This is normally the original position at
 * the time the compute command was issued". "If the com option is set to yes
 * then the effect of any drift in the center-of-mass of the group of atoms is
 * subtracted out before the displacement of each atom is calculated."
 * "If the average option is set to yes then the reference position of an atom
 * is based on the average position of that atom, corrected for center-of-mass
 * motion if requested. The average position is a running average over all
 * previous calls to the compute, including the current call. So on the first
 * call it is current position, on the second call it is the arithmetic
 * average of the current position and the position on the first call, and so
 * on." "Initial coordinates are stored in "unwrapped" form, by using the
 * image flags associated with each atom." Output info: "This compute
 * calculates a global vector of length 4 ... The vector values are
 * "intensive". The vector values will be in distance^2 units."
 * Default: "The option default are com = no, average = no."
 */
export class ComputeMSD extends Compute {
  readonly style = 'msd';
  private comFlag: boolean;
  private aveFlag: boolean;
  /** Unwrapped reference positions, 3 per atom, captured when the compute is defined. */
  private xorig: Float64Array;
  /** Running average of unwrapped positions (average yes), including the current call. */
  private xave: Float64Array;
  /** Center of mass of the group at reference time / running average of it (unwrapped). */
  private cm0: [number, number, number] = [0, 0, 0];
  private cmave: [number, number, number] = [0, 0, 0];
  private calls = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.comFlag = false;
    this.aveFlag = false;
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k];
      if (key !== 'com' && key !== 'average') throw new StyleError(`unknown compute msd keyword '${key}'`);
      const v = args[k + 1];
      if (v !== 'yes' && v !== 'no') throw new StyleError(`compute msd ${key} must be yes or no (got '${v ?? ''}')`);
      if (key === 'com') this.comFlag = v === 'yes';
      else this.aveFlag = v === 'yes';
    }
    this.vectorFlag = true;
    this.sizeVector = 4;
    this.extvector = 0;
    this.vector = new Float64Array(4);
    const s = sys.state;
    const u = [0, 0, 0];
    this.xorig = new Float64Array(3 * s.n);
    this.xave = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) {
      sys.geom.unwrap(s.x, s.image, i, u);
      this.xorig[3 * i] = u[0]; this.xorig[3 * i + 1] = u[1]; this.xorig[3 * i + 2] = u[2];
    }
    this.xave.set(this.xorig);
    if (this.comFlag) {
      this.cm0 = this.groupCom();
      this.cmave = [...this.cm0] as [number, number, number];
    }
  }

  /** Mass-weighted center of mass of the group, unwrapped (compute com). */
  private groupCom(): [number, number, number] {
    const s = this.sys.state;
    const u = [0, 0, 0];
    let mx = 0, my = 0, mz = 0, msum = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      mx += m * u[0]; my += m * u[1]; mz += m * u[2];
      msum += m;
    }
    return [mx / msum, my / msum, mz / msum];
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const u = [0, 0, 0];
    this.calls++;
    if (this.aveFlag) {
      // "a running average over all previous calls to the compute, including
      // the current call": update the references to the current positions first
      for (let i = 0; i < s.n; i++) {
        g.unwrap(s.x, s.image, i, u);
        for (let c = 0; c < 3; c++) this.xave[3 * i + c] += (u[c] - this.xave[3 * i + c]) / this.calls;
      }
      if (this.comFlag) {
        const cm = this.groupCom();
        for (let c = 0; c < 3; c++) this.cmave[c] += (cm[c] - this.cmave[c]) / this.calls;
      }
    }
    const cmNow = this.comFlag ? this.groupCom() : null;
    const cmRef = this.comFlag ? (this.aveFlag ? this.cmave : this.cm0) : null;
    const v = this.vector;
    v.fill(0);
    let count = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      count++;
      g.unwrap(s.x, s.image, i, u);
      const refBase = this.aveFlag ? this.xave : this.xorig;
      let dx = u[0] - refBase[3 * i];
      let dy = u[1] - refBase[3 * i + 1];
      let dz = u[2] - refBase[3 * i + 2];
      if (cmNow && cmRef) {
        dx -= cmNow[0] - cmRef[0];
        dy -= cmNow[1] - cmRef[1];
        dz -= cmNow[2] - cmRef[2];
      }
      v[0] += dx * dx; v[1] += dy * dy; v[2] += dz * dz;
      v[3] += dx * dx + dy * dy + dz * dz;
    }
    for (let c = 0; c < 4; c++) v[c] /= count;
  }
}

/*
 * compute ID group-ID vacf — docs.lammps.org/compute_vacf.html:
 *   compute ID group-ID vacf
 * "Each atom's contribution to the VACF is its current velocity vector dotted
 * into its initial velocity vector at the time the compute was specified."
 * "A vector of four quantities is calculated by this compute." The first
 * three are v_x v_x,0, v_y v_y,0 and v_z v_z,0 summed and averaged over the
 * group; "The fourth element of the vector is the total VACF" (their sum),
 * also averaged over the group. Output info:
 * "This compute calculates a global vector of length 4 ... The vector values
 * are "intensive". The vector values will be in velocity^2 units."
 */
export class ComputeVACF extends Compute {
  readonly style = 'vacf';
  /** Velocities at the time the compute was specified, 3 per atom. */
  private v0: Float64Array;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`compute vacf takes no arguments (got ${args.join(' ')})`);
    this.vectorFlag = true;
    this.sizeVector = 4;
    this.extvector = 0;
    this.vector = new Float64Array(4);
    const s = sys.state;
    this.v0 = Float64Array.from(s.v.subarray(0, 3 * s.n));
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const v = this.vector;
    v.fill(0);
    let count = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      count++;
      for (let c = 0; c < 3; c++) v[c] += s.v[3 * i + c] * this.v0[3 * i + c];
      v[3] += s.v[3 * i] * this.v0[3 * i] + s.v[3 * i + 1] * this.v0[3 * i + 1] + s.v[3 * i + 2] * this.v0[3 * i + 2];
    }
    for (let c = 0; c < 4; c++) v[c] /= count;
  }
}

/*
 * compute ID group-ID rdf Nbin itype1 jtype1 itype2 jtype2 ... keyword/value
 * ... — docs.lammps.org/compute_rdf.html:
 *   compute ID group-ID rdf Nbin itype1 jtype1 itype2 jtype2 ... keyword/value ...
 * "Nbin = number of RDF bins"; "keyword = cutoff"; "cutoff value = Rcut
 *   Rcut = cutoff distance for RDF computation (distance units)".
 * "Both are calculated in histogram form by binning pairwise distances into
 * Nbin bins from 0.0 to the maximum force cutoff defined by the pair_style
 * command or the cutoff distance Rcut specified via the cutoff keyword. The
 * bins are of uniform size in radial distance." A pair I,J is histogrammed
 * when both atoms are in the compute group, their distance is below the
 * maximum force cutoff, and their types match itypeN and jtypeN (each one
 * type or a range).
 * "The g(r) value for a bin is calculated from the histogram count by scaling
 * it by the idealized number of how many counts there would be if atoms of
 * type jtypeN were uniformly distributed. Thus it involves the count of
 * itypeN atoms, the count of jtypeN atoms, the volume of the entire
 * simulation box, and the volume of the bin's thin shell in 3d (or the area
 * of the bin's thin ring in 2d)."
 * "A coordination number coord(r) is also calculated, which is the number of
 * atoms of type jtypeN within the current bin or closer, averaged over atoms
 * of type itypeN. This is calculated as the area- or volume-weighted sum of
 * g(r) values over all bins up to and including the current bin, multiplied
 * by the global average volume density of atoms of type jtypeN."
 * Output info: "This compute calculates a global array in which the number of
 * rows is *Nbins*" with 1 + 2 N_pairs columns: "The first column has the bin
 * coordinate (center of the bin), and each successive set of two columns has
 * the :math:`g(r)` and :math:`\text{coord}(r)` values for a specific set of
 * *itypeN* versus *jtypeN* interactions" "The array values calculated by this
 * compute are all "intensive"." Default: "The keyword defaults are
 * cutoff = 0.0 (use the pairwise force cutoff)."
 *
 * Measured with native LAMMPS (probes e1/e2/e5/e6 in /tmp/opencode/w2cglob):
 * - the histogram counts each ordered pair (I,J) once, as a full neighbor
 *   list does: a pair of two atoms of the same type is therefore counted
 *   twice (once per ordering), matching LAMMPS's half-list traversal that
 *   tests both directions;
 * - the idealized count is
 *       nideal = (countI * countJ - shared) * shellvolume / V
 *   where shared = the number of group atoms whose type lies in both the
 *   itype and jtype ranges (their self-pairs can never be counted). For
 *   pair (1,1) of the w2cglob_rdf oracle this gives 54*53*shell/V and
 *   reproduces every thermo value; the docs' four ingredients alone do not
 *   fix the -shared term;
 * - coord(r) is the cumulative histogram sum divided by countI (not the
 *   g*shell*density product of the prose, which differs whenever shared > 0):
 *   coord(bin) = sum_{b<=bin} hist(b) / countI;
 * - bin centers are (b + 0.5) * delr with delr = cutmax / Nbin.
 * Type labels (docs: "itypeN ... integer, type label, or asterisk form") are
 * not supported by the engine and throw; wildcard ranges are.
 */
export class ComputeRDF extends Compute {
  readonly style = 'rdf';
  private nbin: number;
  /** [ilo, ihi, jlo, jhi] per histogram, 1-based inclusive type ranges. */
  private pairs: [number, number, number, number][] = [];
  /** User cutoff; 0 = use the maximum pairwise force cutoff. */
  private rcut = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.nbin = parseInt_(args[0], 'compute rdf Nbin');
    if (this.nbin < 1) throw new StyleError(`compute rdf Nbin must be >= 1 (got ${args[0]})`);
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
      throw new StyleError(`compute rdf: invalid atom type '${w}' (number, asterisk range, or type label; type labels are not supported by this engine)`);
    };
    for (let k = 1; k < args.length;) {
      if (args[k] === 'cutoff') {
        this.rcut = parseNum(args[k + 1], 'compute rdf cutoff');
        if (!(this.rcut > 0)) throw new StyleError(`compute rdf cutoff must be > 0 (got ${args[k + 1]})`);
        k += 2;
        continue;
      }
      if (args[k + 1] === undefined || args[k + 1] === 'cutoff') {
        throw new StyleError('compute rdf: must provide atom types in pairs (itypeN jtypeN)');
      }
      const [ilo, ihi] = typeRange(args[k]);
      const [jlo, jhi] = typeRange(args[k + 1]);
      for (const t of [ilo, ihi, jlo, jhi]) {
        if (t < 1 || t > ntypes) throw new StyleError(`compute rdf: atom type ${t} is out of range 1..${ntypes}`);
      }
      if (ilo > ihi || jlo > jhi) throw new StyleError(`compute rdf: empty type range in '${args[k]} ${args[k + 1]}'`);
      this.pairs.push([ilo, ihi, jlo, jhi]);
      k += 2;
    }
    if (!this.pairs.length) this.pairs.push([1, ntypes, 1, ntypes]);
    this.arrayFlag = true;
    this.sizeArrayRows = this.nbin;
    this.sizeArrayCols = 1 + 2 * this.pairs.length;
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
    const cutmax = this.rcut > 0 ? this.rcut : this.forceCutoff();
    if (!(cutmax > 0)) {
      throw new StyleError('compute rdf: no pairwise force cutoff is defined; use the cutoff keyword');
    }
    if (nb.cutghost > 0 && nb.cutghost < cutmax + nb.skin - 1e-9) {
      // compute_rdf.html: "If you specify a Rcut > force cutoff, you must
      // ensure ghost atom information out to Rcut + skin is communicated, via
      // the comm_modify cutoff command ... else the RDF computation cannot be
      // performed, and LAMMPS will give an error message."
      throw new StyleError(`compute rdf cutoff ${cutmax} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
    }
    // per-type counts of group atoms (normalization is dynamic: "compute rdf
    // can handle dynamic groups ... certain normalization parameters need to
    // be re-computed in every step")
    const counts = new Float64Array(s.ntypes + 1);
    for (let i = 0; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) counts[s.type[i]]++;
    }
    const delr = cutmax / this.nbin;
    const hist = new Float64Array(this.pairs.length * this.nbin);
    const list = nb.occasionalList(s, cutmax, true, s.ntypes);
    const xa = nb.xall, ta = nb.typeall, owner = nb.owner;
    const cutmax2 = cutmax * cutmax;
    for (let i = 0; i < list.inum; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const ti = ta[i];
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      const lo = list.firstneigh[i];
      const hi = lo + list.numneigh[i];
      for (let k = lo; k < hi; k++) {
        const j = list.neighbors[k];
        const tj = ta[j];
        if (!(s.mask[owner[j]] & this.groupBit)) continue;
        const dx = xa[3 * j] - xi, dy = xa[3 * j + 1] - yi, dz = xa[3 * j + 2] - zi;
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= cutmax2) continue;
        const bin = Math.floor(Math.sqrt(r2) / delr);
        if (bin >= this.nbin) continue;
        for (let m = 0; m < this.pairs.length; m++) {
          const p = this.pairs[m];
          if (ti >= p[0] && ti <= p[1] && tj >= p[2] && tj <= p[3]) hist[m * this.nbin + bin]++;
        }
      }
    }
    // normalization per histogram: g = hist / nideal, coord = cumulative hist / countI
    const vol = sys.geom.volume(s.dimension);
    const cols = this.sizeArrayCols;
    const array = this.array;
    for (let m = 0; m < this.pairs.length; m++) {
      const [ilo, ihi, jlo, jhi] = this.pairs[m];
      let countI = 0, countJ = 0, shared = 0;
      for (let t = ilo; t <= ihi; t++) countI += counts[t];
      for (let t = jlo; t <= jhi; t++) countJ += counts[t];
      for (let t = Math.max(ilo, jlo); t <= Math.min(ihi, jhi); t++) shared += counts[t];
      const ideal = countI * countJ - shared;
      let cum = 0;
      for (let b = 0; b < this.nbin; b++) {
        const h = hist[m * this.nbin + b];
        const shell = s.dimension === 3
          ? (4 / 3) * Math.PI * delr ** 3 * ((b + 1) ** 3 - b ** 3)
          : Math.PI * delr ** 2 * ((b + 1) ** 2 - b ** 2);
        const nideal = (ideal * shell) / vol;
        const gr = nideal > 0 ? h / nideal : 0;
        if (countI > 0) cum += h / countI;
        array[b * cols] = (b + 0.5) * delr;
        array[b * cols + 1 + 2 * m] = gr;
        array[b * cols + 2 + 2 * m] = cum;
      }
    }
  }
}
