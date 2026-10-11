import { Compute } from './compute';
import { StyleError } from '../force/types';
import { massOf } from '../atoms';
import type { System } from '../system';

/*
 * Spatial stress profiles: compute stress/cartesian, stress/mop, stress/mop/profile (implemented) and
 * stress/cylinder, stress/spherical (refused, see registry/compute_stressprof.ts).
 * Documented in docs.lammps.org/compute_stress_cartesian.html and
 * docs.lammps.org/compute_stress_mop.html; both cite the Irving-Kirkwood contour (Ikeshoji,
 * Hafskjold, Furuholt 2003) and the method of planes (Todd, Evans, Daivis 1995). Each rule below was
 * measured with native LAMMPS (black box) on constructed 2-atom cases or on the oracle cases
 * tests/oracle/w47sp_*.in.
 *
 * compute stress/cartesian: the docs say "This compute only supports pair and bond (no angle,
 * dihedral, improper, or kspace) forces." This engine implements the pair and ke parts. The bond
 * keyword, the default keyword set (which contains bond) on a system with bonds, and kspace styles
 * are EngineErrors, as is a pair style without single().
 *   Output: the bin centres in one or two dimensions, the number density, the three diagonal kinetic
 *   and the three diagonal virial components; "There are 8 columns when one dimension is specified
 *   and 9 columns when two dimensions are specified."
 *   Measured: the bin count is floor(L / b) and the bin width becomes L / count, with native printing
 *   the native message Adjusting first bin width for compute stress/cartesian from 1.600000 to 1.679500
 *   (box 6.718, b = 1.6). Box -5..5 with b = 3.0 gives 3 bins centred at -3.333, 0, 3.333. The first dimension varies
 *   fastest over the rows.
 *   Measured: P^k_ab = m v_a v_b / V_bin and P^v_ab = sum r_a f_b / V_bin over the pieces of each pair,
 *   V_bin = bin width times the full box length in the other directions; nktv2p converts to pressure.
 *   Measured: the number density (N_bin / V_bin) is reported only when ke is in effect (no keyword,
 *   or ke). The density and the kinetic part count every atom of the box, whatever the group.
 *   Measured: a pair contributes only when both atoms are in the group.
 *   Measured: a pair line is split between the bins it crosses in proportion to the parameter length
 *   of the segment in each bin; a pair across a periodic boundary uses its minimum-image segment.
 *   Not reproduced: a pair whose two atoms lie exactly on a bin edge plane in the second dimension
 *   (native does not split it as the midpoint rule does). Oracle cases avoid exact lattice alignment.
 *
 * compute stress/mop and stress/mop/profile: the docs describe the kinetic part as atoms that have
 * crossed the plane between the positions at t-dt and t, using the velocity at t-dt/2 from velocity
 * Verlet (compute_stress_mop.html). Measured (lj units, dt = 0.005, plane area 100):
 *   kinetic = nktv2p * mvv2e * sum over crossings of s m v_b / (A dt), with s = +1 for a crossing from
 *   the lower to the upper side and -1 for the reverse, A the plane area and v_b the half-step velocity
 *   v(t) - dt/2 * ftm2v * f / m. Measured: an atom at x = 4.998 with v = 1 and a constant force 1.0
 *   (fix addforce) crosses x = 5 with kinetic 2.005 = m v_half / (A dt) (A = 100, dt = 0.005) with
 *   v_half = 1.0025; without the force, v = 2 with v_y = 0.3 gives 4 and 0.6.
 *   Measured: the kinetic part of a crossing atom is zero when no atom is within dt v of the plane.
 *   Measured: configurational = -(1/A) sum over pairs straddling the plane of the force on the lower-side
 *   atom from the upper-side atom. Measured: a pair with one atom in the group counts (at least one).
 *   Planes (profile): origin + k * delta for every integer k that keeps the plane in [lo, hi] (origin 0,
 *   delta 2.5 in -5..5 gives -5, -2.5, 0, 2.5, 5). lower, center and upper are the box bounds and the
 *   mid-point.
 *   Keywords: kin, conf, total (kin + conf), pair (the pair part of conf). The bond, angle and dihedral
 *   keywords, and conf on a system with bonded terms, are EngineErrors.
 *   Not reproduced: the setup row (the first evaluation of a run, before any step) of stress/mop and
 *   stress/mop/profile. On a displaced lattice native gives a nonzero kinetic part and a different
 *   configurational part there, while the rows after the first step match; the setup evaluation is an
 *   EngineError (refuseSetupRow).
 *
 * compute stress/cylinder and stress/spherical: refused. Measured: the native kinetic part is the
 * projection of m v on the local radial and tangential directions, and the radial virial matched an
 * exact line integral for pairs inside one shell. But for a non-radial pair crossing one shell edge,
 * native assigned the whole pair to the shell of its first atom, a rule that the midpoint and exact
 * integral splits do not reproduce, so the curvilinear profiles are not implemented.
 */

type Mode = 'both' | 'either';

/** Pair list of the last force evaluation: 9 numbers per pair, xi, xj (image positions) and the force on i. */
interface PairSet {
  n: number;
  data: Float64Array;
}

/**
 * Pairs of the half list with pair.single() (pair/local's loop). mode 'both' needs both atoms in the
 * group (compute stress/cartesian), 'either' one (stress/mop).
 */
function collectPairs(sys: System, groupBit: number, mode: Mode): PairSet {
  const pair = sys.ff.pair;
  if (!pair || !pair.single) {
    throw new StyleError('compute stress profile needs a pair_style that supports single() (the engine\'s pairwise evaluation); many-body pair styles are not supported');
  }
  sys.forces();
  const s = sys.state;
  const nb = sys.nb;
  const list = nb.half;
  if (!list) throw new StyleError('compute stress profile: no half neighbour list is available');
  const sLJ = (sys.ff as unknown as { specialLJ: Float64Array }).specialLJ;
  const nt = pair.ntypes + 1;
  const NEIGH_MASK = (1 << 30) - 1;
  const xa = nb.xall;
  const out: number[] = [];
  for (let i = 0; i < list.inum; i++) {
    const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
    const mi = s.mask[i] & groupBit;
    for (let k = k0; k < k1; k++) {
      const jj = list.neighbors[k];
      const j = jj & NEIGH_MASK;
      const mj = s.mask[nb.owner[j]] & groupBit;
      if (mode === 'both' ? !(mi && mj) : !(mi || mj)) continue;
      const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const ti = nb.typeall[i], tj = nb.typeall[j];
      if (rsq >= pair.cutsq[ti * nt + tj]) continue;
      const sb = jj >>> 30;
      const factorLJ = sb === 0 ? 1 : sLJ[sb];
      const res = pair.single(nb.owner[i], nb.owner[j], ti, tj, rsq, factorLJ, factorLJ, nb.qall[i], nb.qall[j]);
      const ff = res.fforce;
      out.push(
        xa[3 * i], xa[3 * i + 1], xa[3 * i + 2],
        xa[3 * j], xa[3 * j + 1], xa[3 * j + 2],
        ff * dx, ff * dy, ff * dz,
      );
    }
  }
  return { n: out.length / 9, data: Float64Array.from(out) };
}

/** Bin axis: dimension, lower edge, bin width, bin count, periodicity. */
interface Axis { d: number; lo: number; w: number; n: number; periodic: boolean }

/** Bin index of coordinate c along one axis, or -1 outside a non-periodic axis. */
function binIndex(ax: Axis, c: number): number {
  const k = Math.floor((c - ax.lo) / ax.w);
  if (ax.periodic) return ((k % ax.n) + ax.n) % ax.n;
  return k < 0 || k >= ax.n ? -1 : k;
}

/**
 * Splits the segment p0 -> p1 at the bin edges of every axis and calls cb(flat bin, dt) per piece, the
 * piece's bin taken at its midpoint; dt is the piece's parameter length. The first axis varies fastest.
 */
function splitSegment(axes: readonly Axis[], p0: readonly number[], p1: readonly number[], cb: (bin: number, dt: number) => void): void {
  const ts: number[] = [0, 1];
  for (const ax of axes) {
    const d0 = p0[ax.d], dd = p1[ax.d] - d0;
    if (dd === 0) continue;
    const lo = Math.min(d0, p1[ax.d]), hi = Math.max(d0, p1[ax.d]);
    const kA = Math.floor((lo - ax.lo) / ax.w), kB = Math.floor((hi - ax.lo) / ax.w);
    for (let k = kA + 1; k <= kB; k++) {
      const t = (ax.lo + k * ax.w - d0) / dd;
      if (t > 0 && t < 1) ts.push(t);
    }
  }
  ts.sort((a, b) => a - b);
  for (let q = 0; q + 1 < ts.length; q++) {
    const t0 = ts[q], t1 = ts[q + 1];
    if (!(t1 > t0)) continue;
    const tm = 0.5 * (t0 + t1);
    let flat = 0, stride = 1, ok = true;
    for (const ax of axes) {
      const k = binIndex(ax, p0[ax.d] + tm * (p1[ax.d] - p0[ax.d]));
      if (k < 0) { ok = false; break; }
      flat += k * stride;
      stride *= ax.n;
    }
    if (ok) cb(flat, t1 - t0);
  }
}

const DIM: Record<string, number> = { x: 0, y: 1, z: 2 };

function parseDim(word: string | undefined, what: string): number {
  if (word === undefined || !(word in DIM)) throw new StyleError(`${what}: dimension must be x, y or z (got '${word ?? ''}')`);
  return DIM[word];
}

/** Plane position from lower, center, upper or a coordinate. */
function planeValue(sys: System, d: number, spec: string): number {
  const s = sys.state;
  const lo = s.box.lo[d], hi = s.box.hi[d];
  if (spec === 'lower') return lo;
  if (spec === 'upper') return hi;
  if (spec === 'center') return 0.5 * (lo + hi);
  const v = Number(spec);
  if (spec === '' || !Number.isFinite(v)) throw new StyleError(`compute stress/mop: plane position must be lower, center, upper or a coordinate (got '${spec}')`);
  return v;
}

const CONF_TERMS_BONDED = ['bond', 'angle', 'dihedral'] as const;

/** Refuses the bonded and kspace contributions that this engine does not implement. */
function refuseStress(sys: System, id: string, style: string, kws: readonly string[]): void {
  for (const w of CONF_TERMS_BONDED) {
    if (kws.includes(w)) throw new StyleError(`compute ${id} ${style}: the ${w} keyword is not implemented in the browser engine`);
  }
  if (kws.includes('conf') && (sys.ff.bond || sys.ff.angle || sys.ff.dihedral || sys.ff.improper)) {
    throw new StyleError(`compute ${id} ${style}: the conf keyword includes the bonded terms, which are not implemented in the browser engine (use the pair keyword)`);
  }
  if (sys.ff.kspace) throw new StyleError(`compute ${id} ${style}: kspace contributions are not in this compute (the engine refuses rather than omit them)`);
}

/**
 * The setup row is not reproduced: the first evaluation of a run (before any step) gives native values
 * that do not follow the rules above (measured: a nonzero kinetic part and a different configurational
 * part on a displaced lattice, the same for run 0 and run 1 with no integrator). Refuse rather than print.
 */
function refuseSetupRow(sys: System, id: string, style: string): void {
  if (sys.run.inRun && sys.state.step === sys.run.firstStep) {
    throw new StyleError(`compute ${id} ${style}: the setup row (the evaluation at step ${sys.state.step}, before the first step of the run) is not reproduced by the browser engine; native gives different values there. Evaluate it after the first step (for example with print after the run, or a thermo output after step 0)`);
  }
}

/** Method-of-planes components of one plane: [kin_x, kin_y, kin_z, conf_x, conf_y, conf_z]. */
function mopPlane(sys: System, d: number, p: number, pairs: PairSet, groupBit: number, withKin: boolean, withConf: boolean): number[] {
  const s = sys.state;
  const out = [0, 0, 0, 0, 0, 0];
  const area = [0, 1, 2].filter((x) => x !== d).reduce((q, x) => q * (s.box.hi[x] - s.box.lo[x]), 1);
  const L = s.box.hi[d] - s.box.lo[d];
  // a periodic box is visited through its images of the plane
  const images = s.box.periodic[d] ? [-1, 0, 1] : [0];
  const nkt = s.units.nktv2p;
  const nkm = s.units.nktv2p * s.units.mvv2e;
  if (withConf) {
    for (let q = 0; q < pairs.n; q++) {
      const o = 9 * q;
      const xi = pairs.data[o + d], xj = pairs.data[o + 3 + d];
      for (const m of images) {
        const pm = p + m * L;
        const li = xi < pm, lj = xj < pm;
        if (li === lj) continue;
        // the force on the lower-side atom from the upper-side atom
        const sign = li ? 1 : -1;
        for (let b = 0; b < 3; b++) out[3 + b] -= (sign * pairs.data[o + 6 + b] * nkt) / area;
      }
    }
  }
  if (withKin) {
    // velocity Verlet: v(t - dt/2) = v(t) - dt/2 * ftm2v * f / m, and x(t - dt) = x(t) - dt * v(t - dt/2)
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & groupBit)) continue;
      const m = massOf(s, i);
      const vh = [0, 0, 0];
      for (let b = 0; b < 3; b++) vh[b] = s.v[3 * i + b] - (dtf / m) * s.f[3 * i + b];
      const xcur = s.x[3 * i + d];
      const xprev = xcur - s.dt * vh[d];
      for (const m2 of images) {
        const pm = p + m2 * L;
        const cur = xcur < pm, prev = xprev < pm;
        if (cur === prev) continue;
        // +1 for a crossing from the lower to the upper side
        const sgn = cur ? -1 : 1;
        for (let b = 0; b < 3; b++) out[b] += (sgn * nkm * m * vh[b]) / (area * s.dt);
      }
    }
  }
  return out;
}

/** Keyword flags of one stress/mop keyword: total = kin + conf, pair is the pair part of conf. */
function mopFlags(kw: string): { kin: boolean; conf: boolean } {
  return { kin: kw === 'kin' || kw === 'total', conf: kw === 'conf' || kw === 'total' || kw === 'pair' };
}

const MOP_KEYWORDS = ['kin', 'conf', 'total', 'pair', 'bond', 'angle', 'dihedral'];

/**
 * compute stress/mop ID group-ID stress/mop dir pos keyword ... (docs: "Compute stress/mop calculates a
 * global vector ... with 3 values for each declared keyword"). Components are P_ix, P_iy, P_iz.
 */
export class ComputeStressMop extends Compute {
  readonly style = 'stress/mop';
  private readonly d: number;
  private readonly pos: string;
  private readonly kws: string[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('usage: compute ID group-ID stress/mop dir pos keywords...');
    this.d = parseDim(args[0], `compute ${id} stress/mop`);
    this.pos = args[1];
    this.kws = args.slice(2);
    for (const w of this.kws) if (!MOP_KEYWORDS.includes(w)) throw new StyleError(`compute ${id} stress/mop: unknown keyword '${w}' (use ${MOP_KEYWORDS.join(', ')})`);
    refuseStress(sys, id, 'stress/mop', this.kws);
    this.vectorFlag = true;
    this.sizeVector = 3 * this.kws.length;
    this.vector = new Float64Array(this.sizeVector);
  }

  init(): void {
    refuseStress(this.sys, this.id, 'stress/mop', this.kws);
  }

  protected computeVector(): void {
    const sys = this.sys;
    refuseStress(sys, this.id, 'stress/mop', this.kws);
    refuseSetupRow(sys, this.id, 'stress/mop');
    const p = planeValue(sys, this.d, this.pos);
    const pairs = collectPairs(sys, this.groupBit, 'either');
    this.vector.fill(0);
    for (let k = 0; k < this.kws.length; k++) {
      const f = mopFlags(this.kws[k]);
      const val = mopPlane(sys, this.d, p, pairs, this.groupBit, f.kin, f.conf);
      for (let b = 0; b < 3; b++) this.vector[3 * k + b] = (f.kin ? val[b] : 0) + (f.conf ? val[3 + b] : 0);
    }
  }
}

/**
 * compute stress/mop/profile ID group-ID stress/mop/profile dir origin delta keyword ... (docs: a global
 * array with 1 column of plane positions and 3 columns per keyword).
 */
export class ComputeStressMopProfile extends Compute {
  readonly style = 'stress/mop/profile';
  private readonly d: number;
  private readonly origin: string;
  private readonly delta: number;
  private readonly kws: string[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('usage: compute ID group-ID stress/mop/profile dir origin delta keywords...');
    this.d = parseDim(args[0], `compute ${id} stress/mop/profile`);
    this.origin = args[1];
    this.delta = Number(args[2]);
    if (!(this.delta > 0)) throw new StyleError(`compute ${id} stress/mop/profile: delta must be positive (got ${args[2]})`);
    this.kws = args.slice(3);
    for (const w of this.kws) if (!MOP_KEYWORDS.includes(w)) throw new StyleError(`compute ${id} stress/mop/profile: unknown keyword '${w}' (use ${MOP_KEYWORDS.join(', ')})`);
    refuseStress(sys, id, 'stress/mop/profile', this.kws);
    this.arrayFlag = true;
    this.sizeArrayCols = 1 + 3 * this.kws.length;
  }

  init(): void {
    refuseStress(this.sys, this.id, 'stress/mop/profile', this.kws);
    this.sizeArrayRows = this.planes().length;
    this.array = new Float64Array(this.sizeArrayRows * this.sizeArrayCols);
  }

  /** Planes origin + k delta inside [lo, hi] (measured natively, see the header comment). */
  private planes(): number[] {
    const s = this.sys.state;
    const lo = s.box.lo[this.d], hi = s.box.hi[this.d];
    const o = planeValue(this.sys, this.d, this.origin);
    const kmin = Math.ceil((lo - o) / this.delta - 1e-10);
    const kmax = Math.floor((hi - o) / this.delta + 1e-10);
    const out: number[] = [];
    for (let k = kmin; k <= kmax; k++) out.push(o + k * this.delta);
    return out;
  }

  protected computeArray(): void {
    const sys = this.sys;
    refuseStress(sys, this.id, 'stress/mop/profile', this.kws);
    refuseSetupRow(sys, this.id, 'stress/mop/profile');
    const planes = this.planes();
    const pairs = collectPairs(sys, this.groupBit, 'either');
    const ncol = this.sizeArrayCols;
    this.sizeArrayRows = planes.length;
    this.array = new Float64Array(planes.length * ncol);
    for (let r = 0; r < planes.length; r++) {
      this.array[r * ncol] = planes[r];
      for (let k = 0; k < this.kws.length; k++) {
        const f = mopFlags(this.kws[k]);
        const val = mopPlane(sys, this.d, planes[r], pairs, this.groupBit, f.kin, f.conf);
        for (let b = 0; b < 3; b++) this.array[r * ncol + 1 + 3 * k + b] = (f.kin ? val[b] : 0) + (f.conf ? val[3 + b] : 0);
      }
    }
  }
}

/**
 * compute stress/cartesian ID group-ID stress/cartesian dim1 bin_width1 dim2 bin_width2 [ke] [pair] [bond]
 * (dim2 may be NULL, then bin_width2 is ignored). Global array: one row per bin.
 */
export class ComputeStressCartesian extends Compute {
  readonly style = 'stress/cartesian';
  private readonly dims: { d: number; width: number }[];
  private axes: Axis[] = [];
  private readonly withKe: boolean;
  private readonly withPair: boolean;
  private readonly explicit: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('usage: compute ID group-ID stress/cartesian dim1 bin_width1 dim2 bin_width2 [keywords]');
    const d1 = parseDim(args[0], `compute ${id} stress/cartesian`);
    const w1 = Number(args[1]);
    this.dims = [{ d: d1, width: w1 }];
    if (args[2] !== 'NULL') {
      const d2 = parseDim(args[2], `compute ${id} stress/cartesian`);
      if (d2 === d1) throw new StyleError(`compute ${id} stress/cartesian: dim1 and dim2 must differ`);
      this.dims.push({ d: d2, width: Number(args[3]) });
    }
    for (const dm of this.dims) {
      if (!(dm.width > 0)) throw new StyleError(`compute ${id} stress/cartesian: bin width must be positive (got ${dm.width})`);
    }
    const kw = args.slice(4);
    for (const w of kw) {
      if (!['ke', 'pair', 'bond'].includes(w)) throw new StyleError(`compute ${id} stress/cartesian: unknown keyword '${w}' (use ke, pair, bond)`);
    }
    if (kw.includes('bond')) throw new StyleError(`compute ${id} stress/cartesian: the bond keyword is not implemented in the browser engine`);
    this.explicit = kw.length > 0;
    this.withKe = !this.explicit || kw.includes('ke');
    this.withPair = !this.explicit || kw.includes('pair');
    this.arrayFlag = true;
    this.sizeArrayCols = this.dims.length === 1 ? 8 : 9;
    if (sys.hasBox) this.resolve();
  }

  /** Bins follow the box (nbins = floor(L / b), width L / nbins); resolved when the box exists. */
  private resolve(): void {
    const s = this.sys.state;
    this.axes = this.dims.map(({ d, width }) => {
      const L = s.box.hi[d] - s.box.lo[d];
      const n = Math.floor(L / width + 1e-10);
      if (n < 1) throw new StyleError(`compute ${this.id} stress/cartesian: bin width ${width} is larger than the box length ${L}`);
      return { d, lo: s.box.lo[d], w: L / n, n, periodic: s.box.periodic[d] };
    });
    this.sizeArrayRows = this.axes.reduce((p, a) => p * a.n, 1);
    this.array = new Float64Array(this.sizeArrayRows * this.sizeArrayCols);
  }

  init(): void {
    this.resolve();
    if (this.sys.ff.kspace) throw new StyleError(`compute ${this.id} stress/cartesian: kspace contributions are not in this compute (the engine refuses rather than omit them)`);
    if (!this.explicit && this.sys.ff.bond) {
      throw new StyleError(`compute ${this.id} stress/cartesian: the default keywords include bond, which is not implemented in the browser engine (give the ke or pair keyword)`);
    }
  }

  protected computeArray(): void {
    const sys = this.sys;
    this.resolve();
    const s = sys.state;
    const axes = this.axes;
    const nrows = this.sizeArrayRows;
    const ncol = this.sizeArrayCols;
    const other = [0, 1, 2].filter((d) => !axes.some((a) => a.d === d)).reduce((p, d) => p * (s.box.hi[d] - s.box.lo[d]), 1);
    const vbin = new Float64Array(nrows);
    for (let r = 0; r < nrows; r++) {
      let v = other;
      for (const a of axes) v *= a.w;
      vbin[r] = v;
    }
    const cnt = new Float64Array(nrows);
    // per bin: 3x3 kinetic and 3x3 virial, row-major
    const pk = new Float64Array(nrows * 9);
    const pv = new Float64Array(nrows * 9);
    const nkm = s.units.nktv2p * s.units.mvv2e;
    const nkt = s.units.nktv2p;
    // native (black box, measured): the number density and the kinetic part count every atom of the box;
    // only the pair part is restricted to the group (both atoms in it)
    for (let i = 0; i < s.n; i++) {
      const bin = this.binOf(i);
      if (bin < 0) continue;
      cnt[bin] += 1;
      if (this.withKe) {
        const m = massOf(s, i) * nkm;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) pk[bin * 9 + 3 * a + b] += m * s.v[3 * i + a] * s.v[3 * i + b];
      }
    }
    if (this.withPair) {
      const pairs = collectPairs(sys, this.groupBit, 'both');
      const dat = pairs.data;
      const W = new Float64Array(9);
      const xi = [0, 0, 0], xj = [0, 0, 0];
      for (let p = 0; p < pairs.n; p++) {
        const o = 9 * p;
        for (let a = 0; a < 3; a++) {
          xi[a] = dat[o + a];
          xj[a] = dat[o + 3 + a];
          const r = dat[o + a] - dat[o + 3 + a];
          for (let b = 0; b < 3; b++) W[3 * a + b] = r * dat[o + 6 + b] * nkt;
        }
        splitSegment(axes, xj, xi, (bin, dt) => {
          for (let q = 0; q < 9; q++) pv[bin * 9 + q] += W[q] * dt;
        });
      }
    }
    for (let r = 0; r < nrows; r++) {
      let rem = r;
      let c = 0;
      for (const a of axes) {
        const k = rem % a.n;
        rem = Math.floor(rem / a.n);
        this.array[r * ncol + c++] = a.lo + (k + 0.5) * a.w;
      }
      this.array[r * ncol + c++] = this.withKe ? cnt[r] / vbin[r] : 0;
      for (let q = 0; q < 3; q++) this.array[r * ncol + c++] = pk[r * 9 + 4 * q] / vbin[r];
      for (let q = 0; q < 3; q++) this.array[r * ncol + c++] = pv[r * 9 + 4 * q] / vbin[r];
    }
  }

  /** Flat bin index of atom i (-1 outside a non-periodic axis); dim1 varies fastest. */
  private binOf(i: number): number {
    const x = this.sys.state.x;
    let flat = 0, stride = 1;
    for (const a of this.axes) {
      const k = binIndex(a, x[3 * i + a.d]);
      if (k < 0) return -1;
      flat += k * stride;
      stride *= a.n;
    }
    return flat;
  }
}

