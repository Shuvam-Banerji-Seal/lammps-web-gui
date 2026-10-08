import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { RanMars } from '../rng';
import { parseNumOrVar, ramp, valueOf, type NumOrVar } from './util';
import { massOf, nativeOrder } from '../atoms';

/*
 * fix ID group-ID gjf Tstart Tstop damp seed [vel vfull|vhalf] [method 1-8 [scalar]]
 *
 * docs.lammps.org/fix_gjf.html: "Apply a Langevin thermostat as described in"
 * (Gronbech-Jensen 2020) "to a group of atoms which models an interaction with a
 * background implicit solvent." and "Unlike the fix langevin command which
 * performs force modifications only, this fix performs thermostatting and time
 * integration. Thus you no longer need a separate time integration fix, like
 * fix nve." The fix therefore replaces fix nve (timeIntegrate = true).
 *
 * Method: the GJ stochastic Stormer-Verlet trajectory of Gronbech-Jensen,
 * arXiv:1909.04380 (Mol. Phys. 2020), Eq. (62) with the coefficients of its
 * Table 1, c1 = (1+c2)/2, c3 = (1-c2)/x, x = alpha dt/m, and the on-site
 * velocity of Eq. (130) in velocity-Verlet form:
 *   r^{n+1} = r^n + sqrt(c1 c3) dt v^n + c3 dt^2/(2m) f^n + c3 dt/(2m) beta^{n+1}
 *   v^{n+1} = c2 v^n + sqrt(c3/c1) dt/(2m) (c2 f^n + f^{n+1}) + sqrt(c1 c3)/m beta^{n+1}
 * The half-step velocity of Eq. (146), u^{n+1/2} = (r^{n+1} - r^n)/(sqrt(c3) dt),
 * is the velocity the fix stores for vhalf (the 2GJ velocity of
 * Gronbech Jensen/Gronbech-Jensen 2019). The noise beta has
 * <beta^n beta^l> = 2 alpha kB T dt delta_nl (Eq. 12b), alpha = m/damp (the
 * friction of the Langevin equation m v' + alpha r' = f + beta, Eq. 1).
 * Since alpha dt/m = dt/damp, the coefficients do not depend on the mass.
 *
 * Methods (Table 1 and Sec. III.2 of arXiv:1909.04380), c2 = velocity attenuation:
 *   1 GJ-I   c2 = a = (1 - x/2)/(1 + x/2)       (the GJF method)
 *   2 GJ-II  c2 = exp(-x)
 *   3 GJ-III c2 = 1 - x
 *   4 GJ-IV  c2 = sqrt(c3)
 *   5 GJ-V   c2 = c3 = 1/(1 + x)
 *   6 GJ-VI  c2 = b^2, b = 1/(1 + x/2)
 * "The keyword *method* selects one of the eight GJ-methods implemented in
 * LAMMPS." Methods 7 and 8 need the splitting forms of Finkelstein et al. 2021
 * and Gronbech-Jensen 2024 and are rejected with a StyleError.
 *
 * Velocity: docs.lammps.org/fix_gjf.html "*vfull* = use on-site velocity" and
 * "The option *vhalf* outputs the 2GJ half-step velocity given in". The
 * default is "The option defaults are vel = vhalf, method = 1."
 *
 * Units: forces are converted to the paper's units with ftm2v (dist/time
 * per force/mass); the noise variance uses boltz/mvv2e, as in fix langevin.
 * Measured with native LAMMPS (black box, one and three atoms, lj units,
 * damp = 1, m = 1, T = 1, dt = 0.01, seed 12345): the velocity stored after
 * the step with vhalf is (r^{n+1} - r^n)/(sqrt(c3) dt); the initial velocity
 * is the on-site velocity for both vel settings; per step each group atom
 * draws three Gaussians (x, y, z) in storage order; the Gaussian stream is
 * the Marsaglia polar pairs of RanMars after one discarded uniform, the first
 * value of a pair being used first; no draws happen in setup.
 */

/** Target temperature in effect this step: a variable, or the Tstart..Tstop ramp. */
const targetT = (sys: System, t: NumOrVar, tStop: number): number =>
  typeof t === 'number' ? ramp(sys, t, tStop) : valueOf(sys, t);

/** Unit-less GJ coefficients (paper Table 1) for method and x = alpha dt/m = dt/damp. */
export const gjCoefficients = (method: number, x: number): { c1: number; c2: number; c3: number } => {
  let c2: number;
  switch (method) {
    case 1: c2 = (1 - x / 2) / (1 + x / 2); break;
    case 2: c2 = Math.exp(-x); break;
    case 3: c2 = 1 - x; break;
    case 4: c2 = 2 / (1 + Math.sqrt(1 + 4 * x)); break;
    case 5: c2 = 1 / (1 + x); break;
    case 6: c2 = 1 / ((1 + x / 2) * (1 + x / 2)); break;
    default: throw new StyleError(`fix gjf: method ${method} is not implemented in this engine`);
  }
  const c1 = (1 + c2) / 2;
  const c3 = method === 4 ? c2 * c2 : (1 - c2) / x;
  return { c1, c2, c3 };
};

export class FixGJF extends Fix {
  readonly style = 'gjf';
  private readonly tStart: NumOrVar;
  private readonly tStop: number;
  private readonly damp: number;
  private readonly rng: RanMars;
  private cached: number | null = null;
  private readonly vhalf: boolean;
  private readonly method: number;
  /** Per-atom storage: old force (paper units), this step's noise, on-site velocity (vhalf only). */
  private fOld = new Float64Array(0);
  private beta = new Float64Array(0);
  private vOn = new Float64Array(0);
  private vRep = new Float64Array(0);
  private coef = { c1: 1, c2: 1, c3: 1 };

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('usage: fix ID group gjf Tstart Tstop damp seed [vel vfull|vhalf] [method 1-8 [scalar]]');
    this.timeIntegrate = true;
    this.tStart = parseNumOrVar(args[0], 'Tstart');
    this.tStop = Number(args[1]);
    this.damp = Number(args[2]);
    const seed = Number(args[3]);
    if (!Number.isFinite(this.tStop)) throw new StyleError('fix gjf: Tstop must be a number');
    if (!(this.damp > 0)) throw new StyleError('fix gjf: damp must be > 0');
    if (!Number.isInteger(seed) || seed <= 0) throw new StyleError('fix gjf: seed must be a positive integer');
    let vel = 'vhalf';
    let method = 1;
    for (let k = 4; k < args.length;) {
      const key = args[k];
      if (key === 'vel') {
        vel = args[k + 1] ?? '';
        if (vel !== 'vfull' && vel !== 'vhalf') throw new StyleError(`fix gjf vel must be vfull or vhalf, not '${vel}'`);
        k += 2;
      } else if (key === 'method') {
        method = Number(args[k + 1]);
        if (!Number.isInteger(method) || method < 1 || method > 8) throw new StyleError(`fix gjf method must be 1-8, not '${args[k + 1]}'`);
        // method 7 takes an extra scalar; both need splitting forms not implemented here
        if (method === 7 || method === 8) throw new StyleError(`fix gjf method ${method} (GJ-${method === 7 ? 'VII' : 'VIII'}) is not implemented in this engine`);
        k += 2;
      } else {
        throw new StyleError(`unknown fix gjf keyword '${key}'`);
      }
    }
    this.vhalf = vel === 'vhalf';
    this.method = method;
    this.rng = new RanMars(seed);
    // Measured with native LAMMPS (black box): the Gaussian stream starts one
    // uniform in, as for fix langevin, so one uniform is discarded here.
    this.rng.uniform();
  }

  /** Gaussian deviate: the second value of each Marsaglia polar pair is kept for the next call. */
  private gaussian(): number {
    if (this.cached !== null) {
      const g = this.cached;
      this.cached = null;
      return g;
    }
    const [g1, g2] = this.rng.polarPair();
    this.cached = g2;
    return g1;
  }

  /** Internal on-site velocity of atom i: stored velocity for vfull, private copy for vhalf. */
  private vIn(i: number, c: number): number {
    return this.vhalf ? this.vOn[3 * i + c] : this.sys.state.v[3 * i + c];
  }

  setup(): void {
    const s = this.sys.state;
    const n3 = 3 * s.n;
    // The on-site velocity is kept privately for vhalf. It is reinitialised from
    // the atom velocities unless they are still the half-step values this fix stored.
    if (this.vhalf) {
      const same = this.vOn.length === n3 && this.vRep.length === n3 &&
        this.vRep.every((x, k) => x === s.v[k]);
      if (!same) this.vOn = Float64Array.from(s.v.subarray(0, n3));
    }
    if (this.fOld.length !== n3) this.fOld = new Float64Array(n3);
    if (this.beta.length !== n3) this.beta = new Float64Array(n3);
  }

  init(): void { this.resetCoef(); }

  private resetCoef(): void {
    this.coef = gjCoefficients(this.method, this.sys.state.dt / this.damp);
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const n = s.n;
    const { x, v, f, mask } = s;
    const dt = s.dt;
    const u = s.units;
    this.resetCoef();
    if (this.fOld.length !== 3 * n) { this.fOld = new Float64Array(3 * n); this.beta = new Float64Array(3 * n); }
    const tt = targetT(this.sys, this.tStart, this.tStop);
    if (tt < 0) throw new StyleError(`fix ${this.id} gjf: target temperature is negative`);
    const { c1, c3 } = this.coef;
    const sq13 = Math.sqrt(c1 * c3);
    const sc3 = Math.sqrt(c3);
    // per-atom draws follow native LAMMPS's atom list (SimState.order)
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      // noise variance 2 alpha kB T dt with alpha = m/damp, in the paper's units
      const sd = Math.sqrt((2 * m * u.boltz * tt * dt) / (this.damp * u.mvv2e));
      for (let c = 0; c < 3; c++) {
        const k = 3 * i + c;
        const b = sd * this.gaussian();
        this.beta[k] = b;
        const fp = f[k] * u.ftm2v;
        this.fOld[k] = fp;
        const vn = this.vIn(i, c);
        const r0 = x[k];
        // Eq. (130a) in paper units
        x[k] = r0 + sq13 * dt * vn + (c3 * dt * dt) / (2 * m) * fp + (c3 * dt) / (2 * m) * b;
        if (this.vhalf) {
          // Eq. (146): the half-step velocity, reported as the atom velocity
          v[k] = (x[k] - r0) / (sc3 * dt);
        }
      }
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const { v, f, mask } = s;
    const dt = s.dt;
    const u = s.units;
    const { c1, c2, c3 } = this.coef;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      for (let c = 0; c < 3; c++) {
        const k = 3 * i + c;
        const fn = f[k] * u.ftm2v;
        const vn = this.vIn(i, c);
        // Eq. (130b) in paper units, with the noise drawn in initialIntegrate
        const vNew = c2 * vn + Math.sqrt(c3 / c1) * (dt / (2 * m)) * (c2 * this.fOld[k] + fn) +
          (Math.sqrt(c1 * c3) / m) * this.beta[k];
        if (this.vhalf) {
          this.vOn[k] = vNew;
        } else {
          v[k] = vNew;
        }
      }
    }
    if (this.vhalf) {
      // remember what was reported, to detect outside changes at the next setup
      this.vRep = Float64Array.from(v.subarray(0, 3 * s.n));
    }
  }
}
