import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf, nativeOrder } from '../atoms';
import { RanMars } from '../rng';

/*
 * fix wall/flow — docs.lammps.org/fix_wall_flow.html (fix_wall_flow.rst).
 * Syntax (quoted from the page):
 *
 *   fix ID group-ID wall/flow axis vflow T seed N coords ... keyword value
 *
 * "axis = flow axis (*x*, *y*, or *z*)", "vflow = generated flow velocity in
 * *axis* direction (velocity units)", "T = flow temperature (temperature
 * units)", "seed = random seed for stochasticity (positive integer)",
 * "N = number of walls", "coords = list of N wall positions along the *axis*
 * direction in ascending order (distance units)", "keyword = *units*" with
 * "*lattice* = wall positions are defined in lattice units" or "*box* = the
 * wall positions are defined in simulation box units". The default for the
 * units keyword is lattice.
 *
 * Model (docs.lammps.org/fix_wall_flow.html): "The goal is to generate a
 * stationary flow with a shifted Maxwell velocity distribution". "Each time a
 * particle in the specified group passes through one of the transparent walls,
 * its velocity is re-assigned." "Note that an additional transparent wall is
 * defined by the code at the boundary of the (periodic) simulation domain in
 * the *axis* direction." "The velocity components in directions parallel to
 * the wall are re-assigned according to the standard Maxwell velocity
 * distribution for the specified temperature *T*." "The velocity component
 * perpendicular to the wall is re-assigned according to the shifted Maxwell
 * distribution defined above". The shifted distribution is
 * f_a(v_a) proportional to exp(-m (v_a - v_flow)^2 / (2 kB T)) and the
 * generated normal component follows f_gen proportional to v_a f_a(v_a).
 * Positions are not altered: "Likewise particle positions are not altered by
 * the velocity reassignment."
 * Restrictions: "This fix can only be used with periodic boundary conditions
 * along the flow axis." The walls must lie inside the box (see the native
 * error text below).
 *
 * Measured with native LAMMPS (black box), 1 to 3 atoms, lj units, dt 0.005:
 *   - Timing: the crossing is detected at the end of the step in which the
 *     position passes the wall (atom at x = 2.99, vx = 3: x = 3.005 at step 1
 *     and the new velocity is already in the step-1 dump). The position of that
 *     step was integrated with the old velocity, and the new velocity is used
 *     from the next step. With a constant force (fix addforce) the reassigned
 *     velocity is the drawn value exactly, without the force kick of that step
 *     (so it is applied after the final half kick, as an end-of-step hook).
 *   - The periodic box edge is a wall too, in both directions: an atom at
 *     x = 9, vx = 3 is reassigned at step 67 when it passes 10 (an unwrapped
 *     position of 10.005) and again at step 68 when it comes back below 10.
 *     With neigh_modify every 20 check no, native wraps atoms into the box only
 *     on its reneighbor steps, and the wrap is counted as one more crossing: the
 *     same atom is reassigned at step 80 (a wrap from 10.067 to 0.067), so one
 *     crossing above the edge can be counted twice. The engine keeps the native
 *     raw coordinate (rebuilt on its own neighbor-list builds, which follow the
 *     same neigh_modify rules) and counts a straddle of the edge or a wall in the
 *     raw coordinate, or a wrap on a build step, as one crossing.
 *   - At most one reassignment per atom per step: an atom that passes two walls
 *     in one step (walls 3.001 and 3.003, atom from 2.99 to 3.005) draws once.
 *   - Atoms are processed in their storage order; with three atoms crossing in
 *     the same step, the first atom gets the first draws, the second the next,
 *     and so on.
 *   - Random numbers: the Marsaglia-Zaman RANMAR stream of rng.ts (RanMars) with
 *     the seed as given, one uniform drawn when the fix is created (warm-up).
 *     Per crossing, in this order:
 *       (1) two uniforms a, b (the flow component);
 *       (2) the transverse components from the Marsaglia polar loop of
 *           RanMars.polarPair() (rejected pairs consume their two draws).
 *     Verified draw for draw: 16 flow components of one atom for vflow/sigma from
 *     0.01 to 8 (both branches of the sign), the transverse components of the
 *     same crossings, three atoms crossing in one step (storage order), and the
 *     multi-crossing one-atom runs of the oracle cases.
 *   - Flow component: with r = vflow/sigma and sigma = sqrt(kB T/m) (the
 *     velocity unit of the run), X = v_a / sigma is drawn from the density
 *     proportional to |x| exp(-(x - r)^2/2) on the whole line. The sign is
 *     negative when (1 - a) < P_neg(r), where P_neg is the probability of the
 *     negative half; on the negative half b is the conditional cumulative
 *     probability, on the positive half 1 - b is. Measured: the implied
 *     conditional uniform is exactly b (negative) or 1 - b (positive) for every
 *     tested r, with a = 0.869953 and b = 0.010426 for the first crossing.
 *     The sign flips at r = 0.7484 for seed 12345 (vflow = sigma = 1), which is
 *     where P_neg = 1 - a = 0.130047. Crossings from either side give the same
 *     distribution (1214 crossings moving left, 3424 moving right; both give a
 *     negative fraction of 0.23).
 *   - Transverse: the components are sigma v2 f and sigma v1 f with
 *     f = sqrt(-2 ln s / s), s = v1^2 + v2^2 < 1, v1 = 2 u1 - 1, v2 = 2 u2 - 1
 *     (the first accepted polar pair). For flow along x the y component is
 *     sigma v2 f and z is sigma v1 f; for flow along y, z is sigma v2 f and x
 *     is sigma v1 f; for flow along z, x is sigma v2 f and y is sigma v1 f. That
 *     is, the next axis (cyclic) gets the first value and the one after gets the second.
 *   - Units: sigma uses the same constants as the velocity of the other
 *     engine fixes (kB and mvv2e); the reassignment is with the atom mass m.
 *   - Walls: the units lattice setting multiplies the coordinates by the lattice spacing
 *     of the flow axis; the units box setting uses them as they are.
 *   - Native errors measured for bad input (text as printed by native):
 *       Illegal fix wall/flow argument: velocity cannot be 0
 *       Illegal fix wall/flow argument: negative direction is not supported yet
 *       Illegal fix wall/flow argument: random seed must be positive integer
 *       Illegal fix wall/flow argument: wall count must be positive integer
 *       Illegal fix wall/flow argument: axis must by x or y or z, but w specified
 *       Wrong fix wall/flow wall count
 *       Wrong fix wall/flow units command
 *       Wrong fix wall/flow wall ordering or some walls are outside simulation domain
 *       Fix wall/flow cannot be used with a non-periodic boundary along the flow axis
 *     A temperature of 0 is accepted by native and gives NaN velocities at the
 *     first crossing (Measured with native LAMMPS, black box), so this engine
 *     refuses T <= 0 instead.
 */

const AXES = ['x', 'y', 'z'] as const;
const SEED_MAX = 900000000;

/** Lattice spacings for "units lattice" (lattice.html: with no lattice command the spacing is 1). */
const latticeSpacing = (sys: System): [number, number, number] =>
  sys.lattice ? ([...sys.lattice.spacing] as [number, number, number]) : [1, 1, 1];

/** erf(x) by the series with positive terms: erf = 2/sqrt(pi) exp(-x^2) sum (2x^2)^n x / (2n+1)!!. */
const erf = (x: number): number => {
  if (x === 0) return 0;
  if (x < 0) return -erf(-x);
  if (x > 9) return 1;
  const x2 = x * x;
  let term = x;
  let sum = x;
  for (let n = 1; n < 4000; n++) {
    term *= (2 * x2) / (2 * n + 1);
    sum += term;
    if (term < 1e-17 * sum) break;
  }
  return (2 / Math.sqrt(Math.PI)) * Math.exp(-x2) * sum;
};
const SQRT_2PI = Math.sqrt(2 * Math.PI);
/** Standard normal density and cumulative distribution. */
const phi = (z: number): number => Math.exp((-0.5 * z) * z) / SQRT_2PI;
const Phi = (z: number): number => 0.5 * (1 + erf(z / Math.SQRT2));

/**
 * int_{-inf}^{x} |t| phi(t - r) dt = phi(x - r) - r Phi(x - r) for x <= 0 (closed form, derivative
 * of the left side is |x| phi(x - r)). Its value at 0 is N0 and the positive mass is P0 = r + N0.
 */
const halfMass = (x: number, r: number): number => phi(x - r) - r * Phi(x - r);

/**
 * Flow component of the generated velocity in units of sigma: X with density proportional to
 * |x| exp(-(x - r)^2 / 2) on the whole line, r = vflow / sigma. a chooses the sign (the negative
 * half when 1 - a < P_neg) and b is the conditional cumulative probability inside that half
 * (negative half: cdf = b; positive half: upper tail = b). Both are solved by bisection on the
 * closed forms. See the Measured notes in the header of this file.
 */
export const fluxNormal = (r: number, a: number, b: number): number => {
  const N0 = halfMass(0, r);
  const P0 = r + N0;
  const pNeg = N0 / (N0 + P0);
  const span = Math.abs(r) + 40;
  let lo: number;
  let hi: number;
  if (1 - a < pNeg) {
    // halfMass increases on (-inf, 0]: find x with halfMass(x) = b N0
    const target = b * N0;
    lo = -span; hi = 0;
    for (let it = 0; it < 80; it++) {
      const m = 0.5 * (lo + hi);
      if (halfMass(m, r) < target) lo = m; else hi = m;
    }
  } else {
    // upper tail r + halfMass(x) decreases on [0, inf): find x with the tail = b P0
    const target = b * P0;
    lo = 0; hi = span;
    for (let it = 0; it < 80; it++) {
      const m = 0.5 * (lo + hi);
      if (r + halfMass(m, r) > target) lo = m; else hi = m;
    }
  }
  return 0.5 * (lo + hi);
};

export class FixWallFlow extends Fix {
  readonly style = 'wall/flow';
  private readonly dim: 0 | 1 | 2;
  private readonly vflow: number;
  private readonly temp: number;
  private readonly walls: number[] = [];
  private readonly rng: RanMars;
  private unitsMode: 'lattice' | 'box' = 'lattice';
  /**
   * Per group atom (by id): the flow-axis coordinate as native LAMMPS holds it, raw = u - k L, and
   * the periodic image k it was last wrapped with. Native wraps atoms only on its reneighbor steps.
   */
  private readonly raw = new Map<number, { raw: number; k: number }>();
  /** Neighbor-list build counter seen at the last end of step (a build = a native reneighbor step). */
  private lastBuildSeen = -2;
  private checked = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const usage = 'usage: fix ID group-ID wall/flow axis vflow T seed N coords ... [units lattice|box]';
    if (args.length < 5) throw new StyleError(`fix ${id} (wall/flow): ${usage}`);
    const axis = args[0];
    const dimIdx = AXES.indexOf(axis as (typeof AXES)[number]);
    if (dimIdx < 0) {
      throw new StyleError(`Illegal fix wall/flow argument: axis must by x or y or z, but ${axis} specified`);
    }
    this.dim = dimIdx as 0 | 1 | 2;
    const vflow = Number(args[1]);
    if (!Number.isFinite(vflow) || args[1].startsWith('v_')) {
      throw new StyleError(`fix ${id} (wall/flow): vflow must be a number, got '${args[1]}'`);
    }
    if (vflow === 0) throw new StyleError('Illegal fix wall/flow argument: velocity cannot be 0');
    if (vflow < 0) throw new StyleError('Illegal fix wall/flow argument: negative direction is not supported yet');
    this.vflow = vflow;
    const temp = Number(args[2]);
    if (!(temp > 0) || !Number.isFinite(temp) || args[2].startsWith('v_')) {
      throw new StyleError(`fix ${id} (wall/flow): the flow temperature T must be a positive number, got '${args[2]}' (native LAMMPS gives NaN velocities for T = 0)`);
    }
    this.temp = temp;
    const seed = Number(args[3]);
    if (!Number.isInteger(seed) || seed < 1 || seed >= SEED_MAX) {
      throw new StyleError('Illegal fix wall/flow argument: random seed must be positive integer');
    }
    const n = Number(args[4]);
    if (!Number.isInteger(n) || n < 1) {
      throw new StyleError('Illegal fix wall/flow argument: wall count must be positive integer');
    }
    if (args.length < 5 + n) throw new StyleError('Wrong fix wall/flow wall count');
    for (let k = 0; k < n; k++) {
      const w = args[5 + k];
      const v = Number(w);
      if (!Number.isFinite(v) || w.startsWith('v_')) {
        throw new StyleError(`fix ${id} (wall/flow): wall position must be a number, got '${w}'`);
      }
      this.walls.push(v);
    }
    for (let k = 5 + n; k < args.length;) {
      if (args[k] !== 'units') {
        throw new StyleError(`fix ${id} (wall/flow): unknown argument '${args[k]}' (expected the keyword units, or nothing)`);
      }
      const v = args[k + 1];
      if (v !== 'lattice' && v !== 'box') throw new StyleError('Wrong fix wall/flow units command');
      this.unitsMode = v;
      k += 2;
    }
    this.rng = new RanMars(seed);
    this.rng.uniform(); // warm-up draw made when the fix is created (measured with native LAMMPS)
  }

  /** Wall positions in box units, checked against the box (strictly inside, ascending). */
  init(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const d = this.dim;
    if (!s.box.periodic[d]) {
      throw new StyleError('Fix wall/flow cannot be used with a non-periodic boundary along the flow axis');
    }
    if (this.checked) return;
    this.checked = true;
    const sp = this.unitsMode === 'lattice' ? latticeSpacing(this.sys)[d] : 1;
    const lo = g.lo[d];
    const hi = g.hi[d];
    for (let k = 0; k < this.walls.length; k++) {
      this.walls[k] *= sp;
    }
    for (let k = 0; k < this.walls.length; k++) {
      const w = this.walls[k];
      const ascending = k === 0 || w > this.walls[k - 1];
      if (!ascending || !(w > lo && w < hi)) {
        throw new StyleError('Wrong fix wall/flow wall ordering or some walls are outside simulation domain');
      }
    }
  }

  /** Remember the positions at the start of each run (native wraps atoms when it sets up a run). */
  setup(): void {
    this.refresh();
  }

  private boxLength(): number {
    return this.sys.geom.hi[this.dim] - this.sys.geom.lo[this.dim];
  }

  /** Unwrapped flow-axis coordinate of atom i (positions are wrapped every step; image flags are exact). */
  private unwrapped(i: number): number {
    const s = this.sys.state;
    return s.x[3 * i + this.dim] + s.image[3 * i + this.dim] * this.boxLength();
  }

  private refresh(): void {
    const s = this.sys.state;
    const d = this.dim;
    this.raw.clear();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.raw.set(s.id[i], { raw: s.x[3 * i + d], k: s.image[3 * i + d] });
    }
    this.lastBuildSeen = this.sys.nb.lastBuild;
  }

  /** A boundary (box edge lo or hi, or a wall) lies in (min, max] between the two raw positions. */
  private straddles(a: number, b: number): boolean {
    const mn = Math.min(a, b);
    const mx = Math.max(a, b);
    const lo = this.sys.geom.lo[this.dim];
    const hi = this.sys.geom.hi[this.dim];
    if ((mn < lo && lo <= mx) || (mn < hi && hi <= mx)) return true;
    for (const w of this.walls) if (mn < w && w <= mx) return true;
    return false;
  }

  endOfStep(): void {
    const s = this.sys.state;
    const d = this.dim;
    const { v, mask, id } = s;
    const L = this.boxLength();
    const lo = this.sys.geom.lo[d];
    // native remaps atoms into the box on its reneighbor steps only; the engine wraps every step, so the
    // native raw coordinate (and its image) is rebuilt here from the unwrapped one at those steps
    const remapped = this.sys.nb.lastBuild !== this.lastBuildSeen;
    this.lastBuildSeen = this.sys.nb.lastBuild;
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      const u = this.unwrapped(i);
      const key = id[i];
      const st = this.raw.get(key);
      if (st === undefined) {
        this.raw.set(key, { raw: s.x[3 * i + d], k: s.image[3 * i + d] });
        continue;
      }
      let k = st.k;
      let wrapped = false;
      if (remapped) {
        const nk = Math.floor((u - lo) / L);
        if (nk !== k) { wrapped = true; k = nk; }
      }
      const raw = u - k * L;
      const crossed = this.straddles(st.raw, raw) || wrapped;
      this.raw.set(key, { raw, k });
      if (!crossed) continue;
      // one reassignment per atom per step, however many boundaries it passed
      const m = massOf(s, i);
      const sig = Math.sqrt((s.units.boltz * this.temp) / (s.units.mvv2e * m));
      const r = this.vflow / sig;
      const a = this.rng.uniform();
      const b = this.rng.uniform();
      v[3 * i + d] = sig * fluxNormal(r, a, b);
      const [t1, t2] = this.rng.polarPair();
      v[3 * i + ((d + 1) % 3)] = sig * t1;
      v[3 * i + ((d + 2) % 3)] = sig * t2;
    }
  }
}
