import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import { globalScalar, parseRef, type Ref } from '../refs';

/*
 * fix controller — implemented from docs.lammps.org/fix_controller.html
 * (plans/lammps-docs/fix_controller.rst). Quoted sentences are copied
 * character for character from that page.
 *
 * Syntax: "fix ID group-ID controller Nevery alpha Kp Ki Kd pvar setpoint cvar".
 * "The group specified with this command is ignored."
 *
 * Equation: "The PID controller is implemented as a discretized version of the
 * following dynamic equation" and the discrete form
 *   c_n = c_{n-1} - alpha (Kp tau e_n + Ki tau^2 sum_{i=1}^n e_i + Kd (e_n - e_{n-1}))
 * with tau = Nevery * timestep and e = pvar - setpoint. "On the first update,
 * the value of the derivative term is set to zero", since e_{n-1} is not yet
 * defined.
 *
 * Measured with native LAMMPS (black box, 10 steps per update, dt 0.005, a
 * thermo of v_fc and f_ctl[1..3] every step): the update made at the end of step
 * 10 is applied in the force computation of step 11, not step 10; the three
 * f_ID values are the three terms of the latest update (proportional
 * -alpha Kp tau e_n, integral -alpha Ki tau^2 sum e_i, derivative
 * -alpha Kd (e_n - e_{n-1})), and the control variable changes by their sum.
 * No update happens at step 0 (setup).
 *
 * Global vector: "This fix produces a global vector with 3 values which can be
 * accessed by various output commands ... The first value is the proportional
 * term, the second is the integral term, the third is the derivative term."
 * "The vector values calculated by this fix are "extensive"."
 */
export class FixController extends Fix {
  readonly style = 'controller';
  vectorFlag = true;
  sizeVector = 3;
  /**
   * Intensive in thermo output. The docs call the vector "extensive", but measured with native LAMMPS
   * (black box): with thermo_modify norm yes (the lj default) the printed f_ctl[1] equals the
   * unnormalized update (0.0125 for 108 atoms), so thermo does not divide it by the atom count.
   */
  extvector = 0;

  private readonly alpha: number;
  private readonly kp: number;
  private readonly ki: number;
  private readonly kd: number;
  private readonly pvar: Ref;
  private readonly setpoint: number;
  private readonly cvar: string;
  /** Sum of the errors at every update so far (the integral sum). */
  private errSum = 0;
  private ePrev = 0;
  private updates = 0;
  private terms = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 8) {
      throw new StyleError(`fix ${id} controller: expected Nevery alpha Kp Ki Kd pvar setpoint cvar, got ${args.length} arguments`);
    }
    const nevery = Math.trunc(parseNum(args[0], `fix ${id} controller Nevery`));
    if (nevery < 1 || nevery !== parseNum(args[0], 'Nevery')) throw new StyleError(`fix ${id} controller: Nevery must be a positive integer, got ${args[0]}`);
    this.nevery = nevery;
    this.alpha = parseNum(args[1], `fix ${id} controller alpha`);
    this.kp = parseNum(args[2], `fix ${id} controller Kp`);
    this.ki = parseNum(args[3], `fix ${id} controller Ki`);
    this.kd = parseNum(args[4], `fix ${id} controller Kd`);
    if (/^[fcv]_/.test(args[5]) === false) throw new StyleError(`fix ${id} controller: pvar must be c_ID, c_ID[I], f_ID, f_ID[I] or v_name, got '${args[5]}'`);
    this.pvar = parseRef(args[5]);
    this.setpoint = parseNum(args[6], `fix ${id} controller setpoint`);
    this.cvar = args[7];
    if (this.pvar.kind === 'v' && this.pvar.index !== null) {
      throw new StyleError(`fix ${id} controller: pvar ${args[5]} must be an equal-style variable, not an element`);
    }
  }

  init(): void {
    const def = this.sys.vars.get(this.cvar);
    if (!def) throw new StyleError(`fix ${this.id} controller: variable ${this.cvar} does not exist`);
    if (def.style !== 'internal') throw new StyleError(`fix ${this.id} controller: variable ${this.cvar} is not internal-style`);
    if (this.pvar.kind === 'v' && !this.sys.vars.has(this.pvar.id)) {
      throw new StyleError(`fix ${this.id} controller: variable ${this.pvar.id} does not exist`);
    }
  }

  endOfStep(): void {
    const sys = this.sys;
    const step = sys.state.step;
    if (step % this.nevery !== 0) return;
    const tau = this.nevery * sys.state.dt;
    const e = globalScalar(sys, this.pvar) - this.setpoint;
    const dEdt = this.updates === 0 ? 0 : e - this.ePrev; // no derivative on the first update
    this.errSum += e;
    const p = -this.alpha * this.kp * tau * e;
    const i = -this.alpha * this.ki * tau * tau * this.errSum;
    const d = -this.alpha * this.kd * dEdt;
    this.terms[0] = p;
    this.terms[1] = i;
    this.terms[2] = d;
    const current = sys.vars.get(this.cvar)?.num ?? 0;
    sys.vars.setInternal(this.cvar, current + p + i + d);
    this.ePrev = e;
    this.updates++;
  }

  computeVector(i: number): number {
    return this.terms[i];
  }
}
