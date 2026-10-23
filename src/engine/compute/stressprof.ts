import { Compute } from './compute';
import { StyleError } from '../force/types';
import { numeric } from '../force/util';
import { massOf } from '../atoms';
import type { System } from '../system';
import { termForces, termRows } from './local';

/*
 * Spatial stress profiles: compute stress/cartesian, stress/mop, stress/mop/profile (implemented here).
 * stress/cylinder and stress/spherical are ported in src/engine-gpl/compute/stress_curvilinear.ts.
 * Documented in docs.lammps.org/compute_stress_cartesian.html and
 * docs.lammps.org/compute_stress_mop.html; both cite the Irving-Kirkwood contour (Ikeshoji,
 * Hafskjold, Furuholt 2003) and the method of planes (Todd, Evans, Daivis 1995). Each rule below was
 * measured with native LAMMPS (black box) on constructed 2-atom cases or on the oracle cases
 * tests/oracle/w47sp_*.in.
 *
 * compute stress/cartesian: the docs say "This compute only supports pair and bond (no angle,
 * dihedral, improper, or kspace) forces." This engine implements the ke, pair and bond parts (a bond is a segment
 * like a pair, with the force of the bond style). kspace styles are EngineErrors, as is a pair style without
 * single().
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
 *   kinetic = nktv2p * mvv2e * sum over crossings of s m v_b / (A dt), with s = copysign(1, v_dir) (native: sign of
 *   the velocity at t, not the crossing direction), A the plane area and v_b the half-step velocity
 *   v(t) - dt/2 * ftm2v * f / m. Measured: an atom at x = 4.998 with v = 1 and a constant force 1.0
 *   (fix addforce) crosses x = 5 with kinetic 2.005 = m v_half / (A dt) (A = 100, dt = 0.005) with
 *   v_half = 1.0025; without the force, v = 2 with v_y = 0.3 gives 4 and 0.6.
 *   Measured: the kinetic part of a crossing atom is zero when no atom is within dt v of the plane.
 *   Measured: configurational = -(1/A) sum over pairs straddling the plane of the force on the lower-side
 *   atom from the upper-side atom. Measured: a pair with one atom in the group counts (at least one).
 *   Planes (profile): origin + k * delta for every integer k that keeps the plane in [lo, hi] (origin 0,
 *   delta 2.5 in -5..5 gives -5, -2.5, 0, 2.5, 5). lower, center and upper are the box bounds and the
 *   mid-point.
 *   Keywords: kin, conf, total (kin + conf), pair (the pair part of conf), bond, angle, dihedral (the bonded parts
 *   of conf; see mopBonded).
 *   Setup row: the evaluation at the setup of a run (step 0) is reproduced. Native counts an atom that lies exactly on
 *   the plane at setup (a lattice site at the plane): the kinetic test uses the closed interval tau in [0, 1] with the
 *   atom folded to the plane image nearest the plane (Domain::minimum_image, compute_stress_mop.cpp compute_pairs()),
 *   and the configurational test is strict (xi > pos && xj < pos, or the same for pos1 = pos +/- prd). Oracle w149cp_mop_setup.
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
  if (!pair || !pair.single || pair.singleNeedsSeparation()) {
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

/**
 * The angle and dihedral styles whose terms compute stress/mop accounts for: native needs their born_matrix (the
 * styles with born_matrix_enable at b332f28; a hybrid style has none) and stops otherwise.
 */
const MOP_ANGLE_STYLES = new Set(['harmonic', 'cosine', 'cosine/squared', 'cosine/squared/restricted', 'cosine/periodic', 'cosine/delta', 'fourier', 'fourier/simple', 'quartic', 'mwlc', 'mm3']);
const MOP_DIHEDRAL_STYLES = new Set(['opls', 'multi/harmonic', 'nharmonic', 'quadratic', 'helix', 'cosine/squared/restricted']);

/** The style name without an accelerator suffix (the /omp, /kk ... aliases are the same styles). */
const baseStyle = (name: string): string => name.replace(/\/(omp|kk|gpu|intel|opt)$/, '');

/** The bonded kinds of a stress/mop keyword list (native bondflag, angleflag, dihedralflag) whose styles exist. */
function mopBondedKinds(sys: System, kws: readonly string[]): { bond: boolean; angle: boolean; dihedral: boolean } {
  const all = kws.includes('conf') || kws.includes('total');
  return {
    bond: (all || kws.includes('bond')) && !!sys.ff.bond,
    angle: (all || kws.includes('angle')) && !!sys.ff.angle,
    dihedral: (all || kws.includes('dihedral')) && !!sys.ff.dihedral,
  };
}

/** Refuses the contributions that this engine does not implement, and native's init errors of stress/mop. */
function refuseStress(sys: System, id: string, style: string, kws: readonly string[]): void {
  // compute_stress_mop.cpp init(): an angle or dihedral style without born_matrix (other than zero or none) stops
  const k = mopBondedKinds(sys, kws);
  const an = sys.ff.angle ? baseStyle(sys.ff.angle.name) : '';
  if (k.angle && !MOP_ANGLE_STYLES.has(an) && an !== 'zero' && an !== 'none') throw new StyleError('compute stress/mop does not account for the selected angle potential');
  const dn = sys.ff.dihedral ? baseStyle(sys.ff.dihedral.name) : '';
  if (k.dihedral && !MOP_DIHEDRAL_STYLES.has(dn) && dn !== 'zero' && dn !== 'none') throw new StyleError('compute stress/mop does not account for the selected dihedral potentials');
  if (sys.ff.kspace) throw new StyleError(`compute ${id} ${style}: kspace contributions are not in this compute (the engine refuses rather than omit them)`);
}

/** One bonded term of a stress/mop computation: its kind, atom indices and the forces of the style on them. */
interface BondedTerm { kind: 'bond' | 'angle' | 'dihedral'; idx: number[]; f: number[] }

/** The bonded terms whose atoms are all in the group (the rows of the local computes, native order), with forces. */
function mopTerms(sys: System, groupBit: number, kinds: { bond: boolean; angle: boolean; dihedral: boolean }): BondedTerm[] {
  const out: BondedTerm[] = [];
  for (const kind of ['bond', 'angle', 'dihedral'] as const) {
    if (!kinds[kind]) continue;
    for (const row of termRows(sys, kind, groupBit)) {
      const { idx, f } = termForces(sys, kind, row.k);
      out.push({ kind, idx, f });
    }
  }
  return out;
}

/**
 * The bonded part of the configurational stress at one plane, per kind: [bond_x, bond_y, bond_z, angle_x, ...]. As
 * native LAMMPS does (src/EXTRA-COMPUTE/compute_stress_mop.cpp compute_bonds, compute_angles, compute_dihedrals at
 * b332f28): the term's first atom is taken at the image nearest the plane and each next atom at the image nearest the
 * previous one; a bond crosses the plane when tau = (x_a - pos) / (x_a - x_b) lies in [0, 1]. A term with a crossing
 * bond adds the force it exerts on the atoms of one side of the plane, signed by that side (native's case table:
 * sgn(x_1) F_1 for a bond; for an angle 1-2-3, sgn(x_3) F_3, sgn(x_1) F_1, or sgn(x_2) F_2 when both bonds cross; for a
 * dihedral the same rule over its three bonds), over the plane area, in pressure units. The forces are those of the
 * style's own evaluation of the term. Native takes a dihedral's derivative from born_matrix(i, ...), whose type comes
 * from neighbor->dihedrallist[i] while i is the atom's own dihedral index, and from the atoms' stored coordinates (no
 * minimum image in the opls form): with one dihedral type and the term's atoms not split by a periodic boundary the two
 * agree, otherwise native's value is that of another type or geometry.
 */
function mopBonded(sys: System, d: number, p: number, terms: readonly BondedTerm[]): number[] {
  const s = sys.state, g = sys.geom;
  const out = new Array<number>(9).fill(0);
  const area = [0, 1, 2].filter((x) => x !== d).reduce((q, x) => q * (s.box.hi[x] - s.box.lo[x]), 1);
  const nkt = s.units.nktv2p;
  const sgnOf = (v: number): number => (v < 0 || Object.is(v, -0) ? -1 : 1);
  const tauIn = (a: number, b: number): boolean => {
    const tau = (a - p) / (a - b);
    return tau >= 0 && tau <= 1;
  };
  for (const t of terms) {
    // the chain of images: the first atom at the image nearest the plane, the next ones nearest the previous
    const xs: number[][] = [];
    const first = [s.x[3 * t.idx[0]], s.x[3 * t.idx[0] + 1], s.x[3 * t.idx[0] + 2]];
    first[d] -= p;
    g.minimumImage(first);
    first[d] += p;
    xs.push(first);
    for (let m = 1; m < t.idx.length; m++) {
      const prev = xs[m - 1], a = t.idx[m];
      const dd = [s.x[3 * a] - prev[0], s.x[3 * a + 1] - prev[1], s.x[3 * a + 2] - prev[2]];
      g.minimumImage(dd);
      xs.push([prev[0] + dd[0], prev[1] + dd[1], prev[2] + dd[2]]);
    }
    const F = (m: number, b: number): number => t.f[3 * m + b];
    const df = [0, 0, 0];
    if (t.kind === 'bond') {
      if (!tauIn(xs[0][d], xs[1][d])) continue;
      const sg = sgnOf(xs[0][d] - p);
      for (let b = 0; b < 3; b++) df[b] = sg * F(0, b);
    } else if (t.kind === 'angle') {
      const right = tauIn(xs[2][d], xs[1][d]), left = tauIn(xs[1][d], xs[0][d]);
      if (!right && !left) continue;
      const lone = right && !left ? 2 : !right && left ? 0 : 1;
      const sg = sgnOf(xs[lone][d] - p);
      for (let b = 0; b < 3; b++) df[b] = sg * F(lone, b);
    } else {
      // dihedral 1-2-3-4: native's right, middle and left bonds are 1-2, 2-3 and 3-4
      const r = tauIn(xs[1][d], xs[0][d]), mid = tauIn(xs[2][d], xs[1][d]), l = tauIn(xs[3][d], xs[2][d]);
      if (!r && !mid && !l) continue;
      // the atoms on the side of the reference atom whose sign native takes, for each crossing pattern
      let ref: number, side: number[];
      if (r && !mid && !l) { ref = 0; side = [0]; }
      else if (!r && mid && !l) { ref = 1; side = [1, 0]; }
      else if (!r && !mid && l) { ref = 3; side = [3]; }
      else if (r && mid && !l) { ref = 1; side = [1]; }
      else if (r && !mid && l) { ref = 0; side = [0, 3]; }
      else if (!r && mid && l) { ref = 2; side = [2]; }
      else { ref = 0; side = [0, 2]; }
      const sg = sgnOf(xs[ref][d] - p);
      for (let b = 0; b < 3; b++) {
        let sum = 0;
        for (const m of side) sum += F(m, b);
        df[b] = sg * sum;
      }
    }
    const o = t.kind === 'bond' ? 0 : t.kind === 'angle' ? 3 : 6;
    for (let b = 0; b < 3; b++) out[o + b] += df[b] / area * nkt;
  }
  return out;
}

/** The bonded part of one keyword's three components: conf and total take every kind, bond/angle/dihedral their own. */
function mopBondedOf(kw: string, val: readonly number[]): number[] {
  const out = [0, 0, 0];
  const take = (o: number): void => { for (let b = 0; b < 3; b++) out[b] += val[o + b]; };
  if (kw === 'conf' || kw === 'total' || kw === 'bond') take(0);
  if (kw === 'conf' || kw === 'total' || kw === 'angle') take(3);
  if (kw === 'conf' || kw === 'total' || kw === 'dihedral') take(6);
  return out;
}

/**
 * Method-of-planes components of one plane: [kin_x, kin_y, kin_z, conf_x, conf_y, conf_z]. Follows
 * compute_stress_mop.cpp compute_pairs(): the configurational test is strict (xi > pos && xj < pos, or the same for
 * pos1), and the kinetic test folds the atom to the plane image nearest the plane (Domain::minimum_image on the
 * displacement from the plane, for a periodic direction) and counts the crossing when tau = (xi - pos)/(xi - xj) lies
 * in the closed interval [0, 1], so an atom exactly on the plane at setup is counted.
 */
function mopPlane(sys: System, d: number, p: number, pairs: PairSet, groupBit: number, withKin: boolean, withConf: boolean): number[] {
  const s = sys.state;
  const out = [0, 0, 0, 0, 0, 0];
  const area = [0, 1, 2].filter((x) => x !== d).reduce((q, x) => q * (s.box.hi[x] - s.box.lo[x]), 1);
  const lo = s.box.lo[d];
  const L = s.box.hi[d] - lo;
  const half = 0.5 * L;
  // native: pos1 = pos + prd when pos < boxlo + prd_half, else pos - prd (the other image of the plane)
  const p1 = p < lo + half ? p + L : p - L;
  const nkt = s.units.nktv2p;
  const periodic = s.box.periodic[d];
  if (withConf) {
    for (let q = 0; q < pairs.n; q++) {
      const o = 9 * q;
      const xi = pairs.data[o + d], xj = pairs.data[o + 3 + d];
      const plus = (xi > p && xj < p) || (xi > p1 && xj < p1);
      const minus = !plus && ((xi < p && xj > p) || (xi < p1 && xj > p1));
      if (!plus && !minus) continue;
      // native: +fpair (xi - xj) for a pair with the atom i above the plane, -fpair (xi - xj) for the reverse
      const add = plus ? 1 : -1;
      for (let b = 0; b < 3; b++) out[3 + b] += (add * pairs.data[o + 6 + b] * nkt) / area;
    }
  }
  if (withKin) {
    // velocity Verlet: x(t - dt) = x(t) - dt * v(t) + dt * f * iterm, iterm = 0.5 / m * dt * ftm2v (compute_stress_mop.cpp)
    const ftm2v = s.units.ftm2v;
    const dt = s.dt;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & groupBit)) continue;
      const m = massOf(s, i);
      const iterm = (0.5 / m) * dt * ftm2v;
      const vi = s.v[3 * i + d], fi = s.f[3 * i + d];
      // minimum image of the displacement from the plane (a periodic direction only)
      let dd = s.x[3 * i + d] - p;
      if (periodic) while (Math.abs(dd) > half) dd = dd < 0 ? dd + L : dd - L;
      const xi = dd + p;
      const xj = xi - vi * dt + fi * iterm * dt;
      const tau = (xi - p) / (xi - xj);
      if (!(tau <= 1 && tau >= 0)) continue;
      // copysign(1, v): -1 for a negative velocity or -0
      const sgn = vi < 0 || Object.is(vi, -0) ? -1 : 1;
      for (let b = 0; b < 3; b++) {
        const vcross = s.v[3 * i + b] - s.f[3 * i + b] * iterm;
        out[b] += (((m * vcross * sgn) / dt) / area * nkt) / ftm2v;
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
    // As native LAMMPS does (src/EXTRA-COMPUTE/compute_stress_mop.cpp ComputeStressMop::ComputeStressMop at b332f28): a plane
    // position that is not lower, upper or center is read by utils::numeric, before the keywords
    if (!['lower', 'upper', 'center'].includes(args[1])) numeric(args[1]);
    this.pos = args[1];
    this.kws = args.slice(2);
    // As native LAMMPS does (src/EXTRA-COMPUTE/compute_stress_mop.cpp ComputeStressMop::ComputeStressMop at b332f28): keywords are read in order and an unknown one stops with "Unknown compute stress/mop keyword".
    for (const w of this.kws) if (!MOP_KEYWORDS.includes(w)) throw new StyleError(`Unknown compute stress/mop keyword ${w}`);
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
    const p = planeValue(sys, this.d, this.pos);
    const pairs = collectPairs(sys, this.groupBit, 'either');
    const terms = mopTerms(sys, this.groupBit, mopBondedKinds(sys, this.kws));
    const bonded = mopBonded(sys, this.d, p, terms);
    this.vector.fill(0);
    for (let k = 0; k < this.kws.length; k++) {
      const f = mopFlags(this.kws[k]);
      const val = mopPlane(sys, this.d, p, pairs, this.groupBit, f.kin, f.conf);
      const bv = mopBondedOf(this.kws[k], bonded);
      // native: vector = pair part (kin and the pair conf) + bond + angle + dihedral
      for (let b = 0; b < 3; b++) this.vector[3 * k + b] = (f.kin ? val[b] : 0) + (f.conf ? val[3 + b] : 0) + bv[b];
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
    // As native LAMMPS does (src/EXTRA-COMPUTE/compute_stress_mop_profile.cpp ComputeStressMopProfile::ComputeStressMopProfile at b332f28):
    // origin and delta are read by utils::numeric (origin may also be lower, center or upper)
    if (!['lower', 'center', 'upper'].includes(args[1])) numeric(args[1]);
    this.origin = args[1];
    this.delta = numeric(args[2]);
    if (!(this.delta > 0)) throw new StyleError(`compute ${id} stress/mop/profile: delta must be positive (got ${args[2]})`);
    this.kws = args.slice(3);
    // As native LAMMPS does (src/EXTRA-COMPUTE/compute_stress_mop_profile.cpp ComputeStressMopProfile::ComputeStressMopProfile at b332f28): an unknown keyword stops with "Unknown compute stress/mop/profile keyword".
    for (const w of this.kws) if (!MOP_KEYWORDS.includes(w)) throw new StyleError(`Unknown compute stress/mop/profile keyword ${w}`);
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
    const planes = this.planes();
    const pairs = collectPairs(sys, this.groupBit, 'either');
    const terms = mopTerms(sys, this.groupBit, mopBondedKinds(sys, this.kws));
    const ncol = this.sizeArrayCols;
    this.sizeArrayRows = planes.length;
    this.array = new Float64Array(planes.length * ncol);
    for (let r = 0; r < planes.length; r++) {
      this.array[r * ncol] = planes[r];
      const bonded = mopBonded(sys, this.d, planes[r], terms);
      for (let k = 0; k < this.kws.length; k++) {
        const f = mopFlags(this.kws[k]);
        const val = mopPlane(sys, this.d, planes[r], pairs, this.groupBit, f.kin, f.conf);
        const bv = mopBondedOf(this.kws[k], bonded);
        for (let b = 0; b < 3; b++) this.array[r * ncol + 1 + 3 * k + b] = (f.kin ? val[b] : 0) + (f.conf ? val[3 + b] : 0) + bv[b];
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
  private readonly withBond: boolean;
  private readonly explicit: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('usage: compute ID group-ID stress/cartesian dim1 bin_width1 dim2 bin_width2 [keywords]');
    const d1 = parseDim(args[0], `compute ${id} stress/cartesian`);
    // As native LAMMPS does (src/EXTRA-COMPUTE/compute_stress_cartesian.cpp ComputeStressCartesian::ComputeStressCartesian at b332f28): utils::numeric reads each bin width
    const w1 = numeric(args[1]);
    this.dims = [{ d: d1, width: w1 }];
    if (args[2] !== 'NULL') {
      const d2 = parseDim(args[2], `compute ${id} stress/cartesian`);
      if (d2 === d1) throw new StyleError(`compute ${id} stress/cartesian: dim1 and dim2 must differ`);
      this.dims.push({ d: d2, width: numeric(args[3]) });
    }
    for (const dm of this.dims) {
      if (!(dm.width > 0)) throw new StyleError(`compute ${id} stress/cartesian: bin width must be positive (got ${dm.width})`);
    }
    const kw = args.slice(4);
    for (const w of kw) {
      // As native LAMMPS does (src/EXTRA-COMPUTE/compute_stress_cartesian.cpp ComputeStressCartesian::ComputeStressCartesian at b332f28): an unknown keyword stops with "Unknown compute stress/cartesian keyword".
      if (!['ke', 'pair', 'bond'].includes(w)) throw new StyleError(`Unknown compute stress/cartesian keyword: ${w}`);
    }
    this.explicit = kw.length > 0;
    this.withKe = !this.explicit || kw.includes('ke');
    this.withPair = !this.explicit || kw.includes('pair');
    this.withBond = !this.explicit || kw.includes('bond');
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
    if (this.withBond && sys.ff.bond) {
      // compute_stress_cartesian.cpp: the bonds of the group (both atoms in it), each a segment from atom 1 to the
      // image of atom 2 nearest it, with the bond force (bond->single) split between bins as a pair's is
      const W = new Float64Array(9);
      const xi = [0, 0, 0], xj = [0, 0, 0];
      for (const row of termRows(sys, 'bond', this.groupBit)) {
        const { idx, f } = termForces(sys, 'bond', row.k);
        const a = idx[0], b = idx[1];
        const dd = [s.x[3 * b] - s.x[3 * a], s.x[3 * b + 1] - s.x[3 * a + 1], s.x[3 * b + 2] - s.x[3 * a + 2]];
        sys.geom.minimumImage(dd);
        for (let q = 0; q < 3; q++) {
          xi[q] = s.x[3 * a + q];
          xj[q] = xi[q] + dd[q];
        }
        for (let q = 0; q < 3; q++) for (let c = 0; c < 3; c++) W[3 * q + c] = (xi[q] - xj[q]) * f[c] * nkt;
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

