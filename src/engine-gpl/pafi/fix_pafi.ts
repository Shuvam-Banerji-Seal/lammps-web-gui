/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_pafi.cpp and src/EXTRA-FIX/fix_pafi.h (class FixPAFI, fix pafi) at commit b332f28, by the
 * LAMMPS developers (path-action fix of Yong Bai and Yuhui Zhu, Univ. of Hawaii). Copyright Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix ID group-ID pafi compute-ID T tdamp seed [overdamped yes/no] [com yes/no] (docs.lammps.org/fix_pafi.html): the
 * path-action dynamics. The compute gives nine values per atom: 0-2 the path point, 3-5 the path direction n, 6-8 the
 * vector n/|n|^2. The force and velocity components along the path are removed, a random force h (uniform in
 * [-0.5, 0.5), RanMars(seed), three draws per group atom in atom order in post_force) acts on the perpendicular
 * motion, and five scalars are exposed as the global vector: f.n psi, its square, psi, |dX.n| and dX.f.
 *
 * Native details kept as in the source: the dX.f projection reads f[i][3..5] of the flat per-atom force array (the
 * force of the next atom, which is exact here). For the last owned atom that read goes past the owned forces into
 * the ghost partial forces, and this port's choice of ghost entry (firstGhost) does not reproduce the native value:
 * results[4] (f_ID[5]) therefore differs from native for that one term (see the w155mx_pafi_* oracle headers).
 * min_post_force reads h as left by the last post_force (zero before it). Minimisers accept only damped dynamics
 * (fire, quickmin) as native requires; this port does not check the minimiser style.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { inumeric, numeric } from '../../engine/force/util';
import { massOf } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { RanMars } from '../random_mars';

const USAGE = 'usage: fix ID group-ID pafi compute-ID T tdamp seed [overdamped yes/no] [com yes/no]';

/** Projections (proj) and centre-of-mass sums (cv) of one evaluation (native proj_all, c_v_all). */
interface Sums { proj: Float64Array; cv: Float64Array }

export class FixPAFI extends Fix {
  readonly style = 'pafi';
  // fix_pafi.cpp constructor: "time_integrate = 1;"
  timeIntegrate = true;
  private readonly computeId: string;
  private readonly rng: RanMars;
  private readonly gamma: number;
  private readonly sqrtD: number;
  private od = false;
  private com = true;
  private dtv = 0;
  private dtf = 0;
  private h = new Float64Array(0);
  /** results: f.n psi, (f.n psi)^2, psi, |dX.n|, dX.f (native results_all). */
  private readonly results = new Float64Array(5);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/EXTRA-FIX/fix_pafi.cpp FixPAFI::FixPAFI at b332f28): narg < 11 is "Illegal fix pafi command"
    if (args.length < 7) throw new StyleError('Illegal fix pafi command');
    this.vectorFlag = true;
    this.sizeVector = 5;
    this.extvector = 0;
    this.computeId = args[0];
    const pc = sys.compute(this.computeId);
    if (!pc.peratomFlag) throw new StyleError(`Compute ${this.computeId} for fix pafi does not calculate a local array`);
    if (pc.sizePeratomCols < 9) throw new StyleError(`Compute ${this.computeId} for fix pafi must have 9 fields per atom`);
    // utils::numeric reads the temperature and the period, utils::inumeric the seed
    const temperature = numeric(args[1]), tPeriod = numeric(args[2]), seed = inumeric(args[3]);
    // fix_pafi.cpp constructor: gamma = 1/t_period/ftm2v, sqrtD = sqrt(24 kB T/(t_period dt mvv2e))/ftm2v
    const u = sys.state.units, dt = sys.state.dt;
    this.gamma = 1 / tPeriod / u.ftm2v;
    this.sqrtD = Math.sqrt(24 * u.boltz / tPeriod / dt / u.mvv2e * temperature) / u.ftm2v;
    let k = 4;
    while (k < args.length) {
      const w = args[k], v = args[k + 1];
      // As native LAMMPS does (src/EXTRA-FIX/fix_pafi.cpp FixPAFI::FixPAFI at b332f28): a word that is neither keyword stops with "Illegal fix pafi command".
      if (w !== 'overdamped' && w !== 'com') throw new StyleError('Illegal fix pafi command');
      if (v !== 'yes' && v !== 'no') throw new StyleError(`Illegal fix pafi command (${USAGE})`);
      if (w === 'overdamped') this.od = v === 'yes';
      else this.com = v === 'yes';
      k += 2;
    }
    this.rng = new RanMars(Math.trunc(seed));
  }

  init(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    const pc = this.sys.compute(this.computeId);
    if (!pc.peratomFlag || pc.sizePeratomCols < 9) throw new StyleError(`Compute ${this.computeId} for fix pafi must have 9 fields per atom`);
  }

  setup(): void { this.postForce(); }

  minSetup(): void { this.minPostForce(); }

  computeVector(i: number): number {
    return this.results[i];
  }

  /** The path arrays (PathCompute->compute_peratom and array_atom), flat with 9 columns per atom. */
  private path(): { p: Float64Array; cols: number } {
    const pc = this.sys.compute(this.computeId);
    pc.invalidate();
    return { p: pc.peratomValues(), cols: pc.sizePeratomCols };
  }

  private hBuf(n: number): Float64Array {
    if (this.h.length < 3 * n) {
      const next = new Float64Array(3 * n);
      next.set(this.h.subarray(0, this.h.length));
      this.h = next;
    }
    return this.h;
  }

  /**
   * Flat force component k of native atom->f (nmax x 3, so f[i][3..5] is the next atom's force, and the component
   * after the last owned atom is the first ghost's). The ghost forces of native are partial sums (the pairs of the
   * half list with an owned atom), kept after the reverse communication. ghost0 is the engine's ghost entry for
   * native's first ghost (see firstGhost), or -1 when there is none; the owned part is the state force.
   */
  private fFlat(k: number, ghost0: number): number {
    const s = this.sys.state;
    if (k < 3 * s.n) return s.f[k];
    if (ghost0 < 0) return 0;
    return this.sys.nb.fall[3 * ghost0 + (k - 3 * s.n)];
  }

  /**
   * Engine ghost entry of native's first ghost: CommBrick::setup and borders (one proc) send first the owned atoms
   * with x <= sublo + cutghost (slabhi = sublo + cutghost, slablo = -BIG for the first swap, pbc +1), in index
   * order; their ghosts are the images shifted by +Lx. Returns the ghost index in the engine's neighbour arrays, or -1.
   */
  private firstGhost(): number {
    const s = this.sys.state, nb = this.sys.nb;
    const hi = s.box.lo[0] + nb.cutghost;
    let owner0 = -1;
    for (let i = 0; i < s.n; i++) {
      if (s.x[3 * i] <= hi) { owner0 = i; break; }
    }
    if (owner0 < 0) return -1;
    for (let k = s.n; k < nb.nall; k++) {
      if (nb.owner[k] === owner0 && nb.gimage[3 * k] === 1 && nb.gimage[3 * k + 1] === 0 && nb.gimage[3 * k + 2] === 0) return k;
    }
    return -1;
  }

  /** The group sums of proj (projections on the path) and c_v (com sums), as native post_force / initial_integrate. */
  private sums(mode: 'post' | 'min' | 'initial' | 'final'): Sums {
    const sys = this.sys, s = sys.state;
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    const { p, cols } = this.path();
    const h = this.h;
    const proj = new Float64Array(6), cv = new Float64Array(10);
    const dev = [0, 0, 0];
    const ghost0 = mode === 'post' || mode === 'min' ? this.firstGhost() : -1;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const o = i * cols;
      const nx = p[o + 3], ny = p[o + 4], nz = p[o + 5];
      proj[0] += f[3 * i] * nx;
      proj[0] += f[3 * i + 1] * ny;
      proj[0] += f[3 * i + 2] * nz;
      proj[1] += v[3 * i] * nx;
      proj[1] += v[3 * i + 1] * ny;
      proj[1] += v[3 * i + 2] * nz;
      if (mode === 'post' || mode === 'min') {
        proj[2] += h[3 * i] * nx;
        proj[2] += h[3 * i + 1] * ny;
        proj[2] += h[3 * i + 2] * nz;
        dev[0] = x[3 * i] - p[o]; dev[1] = x[3 * i + 1] - p[o + 1]; dev[2] = x[3 * i + 2] - p[o + 2];
        sys.geom.minimumImage(dev);
        proj[3] += p[o + 6] * dev[0];
        proj[3] += p[o + 7] * dev[1];
        proj[3] += p[o + 8] * dev[2];
        proj[4] += nx * dev[0];
        proj[4] += ny * dev[1];
        proj[4] += nz * dev[2];
        // native reads f[i][3..5] of the flat array: the force of the next atom in the flat layout
        proj[5] += this.fFlat(3 * i + 3, ghost0) * dev[0];
        proj[5] += this.fFlat(3 * i + 4, ghost0) * dev[1];
        proj[5] += this.fFlat(3 * i + 5, ghost0) * dev[2];
      }
    }
    if (!this.com) cv[9] += 1.0;
    else {
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & bit)) continue;
        if (mode === 'post' || mode === 'min') {
          cv[0] += f[3 * i]; cv[1] += f[3 * i + 1]; cv[2] += f[3 * i + 2];
          cv[3] += v[3 * i]; cv[4] += v[3 * i + 1]; cv[5] += v[3 * i + 2];
          cv[6] += h[3 * i]; cv[7] += h[3 * i + 1]; cv[8] += h[3 * i + 2];
        } else if (mode === 'initial') {
          cv[0] += v[3 * i]; cv[1] += v[3 * i + 1]; cv[2] += v[3 * i + 2];
          cv[3] += f[3 * i]; cv[4] += f[3 * i + 1]; cv[5] += f[3 * i + 2];
        } else {
          cv[3] += f[3 * i]; cv[4] += f[3 * i + 1]; cv[5] += f[3 * i + 2];
        }
        cv[9] += 1.0;
      }
    }
    return { proj, cv };
  }

  /** post_force: the random force, the path projections removed, then the damping and noise on the group. */
  postForce(): void {
    const sys = this.sys, s = sys.state;
    const { v, f, mask } = s;
    const bit = this.groupBit;
    const h = this.hBuf(s.n);
    const { p, cols } = this.path();
    // random numbers: three per group atom in atom order, drawn before the sums that use them (native loop order)
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      h[3 * i] = this.rng.uniform() - 0.5;
      h[3 * i + 1] = this.rng.uniform() - 0.5;
      h[3 * i + 2] = this.rng.uniform() - 0.5;
    }
    const { proj, cv } = this.sums('post');
    this.setResults(proj);
    this.removeProjections(p, cols, f, v, h, proj, cv, true);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const mf = Math.sqrt(massOf(s, i));
      if (!this.od) {
        f[3 * i] += -this.gamma * mf * mf * v[3 * i];
        f[3 * i + 1] += -this.gamma * mf * mf * v[3 * i + 1];
        f[3 * i + 2] += -this.gamma * mf * mf * v[3 * i + 2];
        f[3 * i] += this.sqrtD * mf * h[3 * i];
        f[3 * i + 1] += this.sqrtD * mf * h[3 * i + 1];
        f[3 * i + 2] += this.sqrtD * mf * h[3 * i + 2];
      } else {
        f[3 * i] += this.sqrtD * h[3 * i] * mf;
        f[3 * i + 1] += this.sqrtD * h[3 * i + 1] * mf;
        f[3 * i + 2] += this.sqrtD * h[3 * i + 2] * mf;
        f[3 * i] /= this.gamma * mf * mf;
        f[3 * i + 1] /= this.gamma * mf * mf;
        f[3 * i + 2] /= this.gamma * mf * mf;
      }
    }
  }

  /** min_post_force: the path projections removed from f and v (no random force, no damping). */
  minPostForce(): void {
    const s = this.sys.state;
    const { v, f } = s;
    const h = this.hBuf(s.n);
    const { p, cols } = this.path();
    const { proj, cv } = this.sums('min');
    this.setResults(proj);
    this.removeProjections(p, cols, f, v, h, proj, cv, false);
  }

  /** results from the post_force / min_post_force projections (native results_all). */
  private setResults(proj: Float64Array): void {
    this.results[0] = proj[0] * (1 - proj[3]);
    this.results[1] = this.results[0] * this.results[0];
    this.results[2] = 1 - proj[3];
    this.results[3] = Math.abs(proj[4]);
    this.results[4] = proj[5];
  }

  /** f -= proj0 n + cv/N, v -= proj1 n + cv/N, and h -= proj2 n + cv/N when withH (native post_force / min). */
  private removeProjections(p: Float64Array, cols: number, f: Float64Array, v: Float64Array, h: Float64Array, proj: Float64Array, cv: Float64Array, withH: boolean): void {
    const s = this.sys.state;
    const bit = this.groupBit;
    const N = cv[9];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & bit)) continue;
      const o = i * cols;
      const nx = p[o + 3], ny = p[o + 4], nz = p[o + 5];
      f[3 * i] -= proj[0] * nx + cv[0] / N;
      f[3 * i + 1] -= proj[0] * ny + cv[1] / N;
      f[3 * i + 2] -= proj[0] * nz + cv[2] / N;
      v[3 * i] -= proj[1] * nx + cv[3] / N;
      v[3 * i + 1] -= proj[1] * ny + cv[4] / N;
      v[3 * i + 2] -= proj[1] * nz + cv[5] / N;
      if (withH) {
        h[3 * i] -= proj[2] * nx + cv[6] / N;
        h[3 * i + 1] -= proj[2] * ny + cv[7] / N;
        h[3 * i + 2] -= proj[2] * nz + cv[8] / N;
      }
    }
  }

  initialIntegrate(): void {
    this.integrate(true);
  }

  finalIntegrate(): void {
    this.integrate(false);
  }

  /** initial_integrate (x and v updated) and final_integrate (v only), with the path projections of the force. */
  private integrate(initial: boolean): void {
    const s = this.sys.state;
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    const { p, cols } = this.path();
    const { proj, cv } = this.sums(initial ? 'initial' : 'final');
    const N = cv[9];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const o = i * cols;
      const nx = p[o + 3], ny = p[o + 4], nz = p[o + 5];
      if (!this.od) {
        const dtfm = this.dtf / massOf(s, i);
        v[3 * i] += dtfm * (f[3 * i] - nx * proj[0] - cv[3] / N);
        v[3 * i + 1] += dtfm * (f[3 * i + 1] - ny * proj[0] - cv[4] / N);
        v[3 * i + 2] += dtfm * (f[3 * i + 2] - nz * proj[0] - cv[5] / N);
        if (initial) {
          x[3 * i] += this.dtv * (v[3 * i] - nx * proj[1] - cv[0] / N);
          x[3 * i + 1] += this.dtv * (v[3 * i + 1] - ny * proj[1] - cv[1] / N);
          x[3 * i + 2] += this.dtv * (v[3 * i + 2] - nz * proj[1] - cv[2] / N);
        }
      } else {
        if (initial) {
          v[3 * i] = 0; v[3 * i + 1] = 0; v[3 * i + 2] = 0;
          x[3 * i] += this.dtv * (f[3 * i] - nx * proj[0] - cv[3] / N);
          x[3 * i + 1] += this.dtv * (f[3 * i + 1] - ny * proj[0] - cv[4] / N);
          x[3 * i + 2] += this.dtv * (f[3 * i + 2] - nz * proj[0] - cv[5] / N);
        } else {
          v[3 * i] = 0; v[3 * i + 1] = 0; v[3 * i + 2] = 0;
        }
      }
    }
  }
}
