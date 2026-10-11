import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * fix ti/spring — written from docs.lammps.org/fix_ti_spring.html (plans/lammps-docs-live/fix_ti_spring.rst).
 * Syntax quoted there:
 *   "fix ID group-ID ti/spring k t_s t_eq keyword value ..."
 * The bullet list in the same page names the arguments in the other order ("k = spring constant",
 * "t_eq = number of steps for the equilibration procedure", "t_s = number of steps for the switching
 * procedure"). Measured with native LAMMPS (black box): the first integer after k is the switching
 * length and the second the equilibration length (fix ti/spring 3.0 4 3 gives a 4-step switch and
 * 3-step holds), which is the order the example's comments use (in.ti_spring: 2000 switching, 1000 equilibration).
 *
 * Force: "F = \left( 1-\lambda \right) F_{\text{solid}} + \lambda F_{\text{harm}}" in the doc's notation.
 * The fix rescales the force present on each atom: f_i <- (1 - lambda) f_i + lambda F_harm,i with
 * F_harm,i = -k (x_i - x_i0) and x_i0 the position when the fix command was issued ("The initial position
 * of each atom is its position at the time the fix command was issued").
 *
 * Schedule: "during the first t_eq steps after the fix is defined the value of lambda is zero" ... "After
 * this the value of lambda changes dynamically during the simulation from 0 to 1 ... in t_s steps"; then
 * "the second equilibration period of t_eq" at lambda = 1, "the switching back to the lambda = 0 state is
 * made using t_s timesteps", and "After this period the value of lambda is kept equal to zero".
 * With start = the step the fix was defined at, up = start + t_eq and down = up + t_s + t_eq:
 *   lambda(s) = g(tau_up) - g(tau_down),  tau_up = clamp((s - up)/t_s), tau_down = clamp((s - down)/t_s)
 * which is 0 before up, rises to 1 over [up, up + t_s], holds 1 until down, falls to 0 over [down, down + t_s]
 * and stays 0. Measured with native LAMMPS (black box) with t_s = 4, t_eq = 3 defined at step 0: lambda is
 * 0 through step 3, 0.25 at step 4, 1 at step 7, 1 through step 10, 0.75 at step 11 and 0 from step 14 on;
 * with t_s = 5, t_eq = 2 defined at step 4: lambda 0.0195814, 0.266568, 0.733432, 0.980419 at steps 7-10.
 *
 * Switching functions (function keyword, "function-ID = ID of the switching function (1 or 2)"):
 *   function 1: "\lambda(\tau) = \tau"; function 2: "\lambda(\tau) = \tau^5 \left( 70 \tau^4 - 315 \tau^3 + 540 \tau^2 - 420 \tau + 126 \right)",
 * with tau = t/t_s. Measured with native LAMMPS (black box): the second value (dlambda/ds) is
 * f'(tau)/t_s with f'(tau) = 630 tau^4 (1 - tau)^4 for function 2 (0.0825754 at tau = 0.2, 5-step switch),
 * sign + on the way up and - on the way down.
 *
 * Output: "This fix computes a global scalar and a global vector quantities which can be accessed by various"
 * (the rest of that sentence is in the doc). "The scalar is an energy which is the sum of the spring energy for
 * each atom, where the per-atom energy is" 0.5 k r^2 (the doc's math). The vector: "The first value is the
 * coupling parameter lambda." "The second value is the derivative of lambda with respect to the integer timestep s". Measured with native LAMMPS (black box): the derivative is
 * S(s) f'(tau)/t_s with tau the clamped switching variable of the current phase and S = +1 before the
 * down switch (step down = up + t_s + t_eq) and -1 from it on. Holds with function 1 therefore show
 * +1/t_s (before step down) and -1/t_s (after), while function 2 is zero in the holds because f' vanishes at
 * tau = 0 and 1. The scalar and vector are extensive (thermo divides them by the atom count in lj units).
 *
 * fix_modify energy: "The fix_modify energy option is supported by this fix to add the energy stored in the
 * per-atom springs to the global potential energy of the system". Default no.
 *
 * The doc's Restrictions: none for the engine; the doc's note on the centre of mass (use fix langevin zero
 * yes, velocity mom yes) is advice, not enforced.
 */

const KEYWORDS = ['function'] as const;

const quintic = (t: number): number => t ** 5 * (70 * t ** 4 - 315 * t ** 3 + 540 * t ** 2 - 420 * t + 126);
const dquintic = (t: number): number => 630 * t ** 4 * (1 - t) ** 4;
const clamp01 = (t: number): number => (t < 0 ? 0 : t > 1 ? 1 : t);

export class FixTiSpring extends Fix {
  readonly style = 'ti/spring';
  private readonly k: number;
  private readonly ts: number;
  private readonly teq: number;
  private readonly func: 1 | 2;
  private readonly start: number;
  /** unwrapped positions at the time the fix was defined, per atom index */
  private x0: Float64Array;
  private lam = 0;
  private dlam = 0;
  private espring = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 2;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
    if (args.length < 3) throw new StyleError('fix ti/spring: usage: fix ID group-ID ti/spring k t_s t_eq [function 1|2]');
    this.k = Number(args[0]);
    if (!Number.isFinite(this.k)) throw new StyleError(`fix ti/spring: expected a spring constant, got '${args[0]}'`);
    const ts = Number(args[1]), teq = Number(args[2]);
    if (!Number.isInteger(ts) || ts < 1) throw new StyleError(`fix ti/spring: t_s must be a positive integer, got '${args[1]}'`);
    if (!Number.isInteger(teq) || teq < 0) throw new StyleError(`fix ti/spring: t_eq must be a non-negative integer, got '${args[2]}'`);
    this.ts = ts;
    this.teq = teq;
    this.func = 1;
    let k = 3;
    while (k < args.length) {
      if (!KEYWORDS.includes(args[k] as (typeof KEYWORDS)[number])) throw new StyleError(`fix ti/spring: unknown keyword '${args[k]}' (function)`);
      if (k + 1 >= args.length) throw new StyleError('fix ti/spring: function needs a value (1 or 2)');
      const f = Number(args[k + 1]);
      if (f !== 1 && f !== 2) throw new StyleError(`fix ti/spring: function must be 1 or 2, got '${args[k + 1]}'`);
      this.func = f;
      k += 2;
    }
    this.start = sys.state.step;
    const s = sys.state;
    this.x0 = new Float64Array(3 * s.n);
    const p = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      sys.geom.unwrap(s.x, s.image, i, p);
      this.x0[3 * i] = p[0]; this.x0[3 * i + 1] = p[1]; this.x0[3 * i + 2] = p[2];
    }
  }

  /** lambda and dlambda/ds at step s (the value a force evaluation at step s uses). */
  private schedule(s: number): { lam: number; dlam: number } {
    const up = this.start + this.teq;
    const down = up + this.ts + this.teq;
    const tUp = clamp01((s - up) / this.ts);
    const tDn = clamp01((s - down) / this.ts);
    const g = this.func === 1 ? (t: number) => t : quintic;
    const df = this.func === 1 ? (_t: number) => 1 : dquintic;
    const lam = g(tUp) - g(tDn);
    const before = s < down;
    const tau = before ? tUp : tDn;
    const dlam = (before ? 1 : -1) * df(tau) / this.ts;
    return { lam, dlam };
  }

  postForce(): void {
    const s = this.sys.state;
    const { lam, dlam } = this.schedule(s.step);
    this.lam = lam;
    this.dlam = dlam;
    const x = s.x, f = s.f;
    const p = [0, 0, 0];
    let e = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      this.sys.geom.unwrap(x, s.image, i, p);
      for (let c = 0; c < 3; c++) {
        const dx = p[c] - this.x0[3 * i + c];
        e += 0.5 * this.k * dx * dx;
        const fharm = -this.k * dx;
        f[3 * i + c] = (1 - lam) * f[3 * i + c] + lam * fharm;
      }
    }
    // Measured with native LAMMPS (black box): the spring energy is zero before the switch-up begins (steps
    // start .. start + t_eq - 1 print 0 with lambda = 0), and nonzero from the switch-up step on.
    this.espring = s.step < this.start + this.teq ? 0 : e;
  }

  minPostForce(): void { this.postForce(); }

  energy(): number { return this.espring; }
  computeScalar(): number { return this.espring; }
  computeVector(i: number): number {
    if (i === 0) return this.lam;
    if (i === 1) return this.dlam;
    throw new StyleError(`fix ${this.id} ti/spring vector index ${i + 1} out of range`);
  }
}
