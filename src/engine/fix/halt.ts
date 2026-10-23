import { Fix } from './fix';
import { StyleError } from '../force/types';
import { inumeric, missingArgs, parseNum } from '../force/util';
import type { System } from '../system';
import { formatNumber } from '../cformat';
import { requestHalt } from '../run/halt';

/*
 * fix halt — implemented from docs.lammps.org/fix_halt.html (plans/lammps-docs-live/fix_halt.rst).
 * Quoted sentences are copied character for character from that page:
 *
 *   "Check a condition every N steps during a simulation run.  N must be >=1.
 *   If the condition is met, exit the run."
 *   "The choice of operators listed above are the usual comparison operators.
 *   The XOR operation (exclusive or) is also included as "\|\^"."
 *   "In this context, XOR means that if either the attribute or avalue is
 *   0.0 and the other is non-zero, then the result is "true".  Otherwise
 *   it is "false"."
 *   "The optional *message* keyword determines whether a message is printed
 *   to the screen and logfile when the halt condition is triggered."
 *
 * Checks run on the steps that are multiples of N (absolute timestep, measured with native LAMMPS, black
 * box: fix halt 3 after run 1 checks at steps 3, 6, ...). The first check is not made on the setup step of a
 * run. Measured with native LAMMPS (black box): a halt stops the run after the step on which it was met; the
 * thermo row of that step is printed only if the thermo interval includes it, and the run's last row is not
 * forced; with error soft a later run or minimize is skipped without output until timer timeout off;
 * with error continue later runs proceed and the check repeats on the next multiple of N. Message yes prints a line such as Fix halt condition for fix-id h met on step 3 with value 0.015.
 *
 * Refused (EngineError): attribute tlimit (its value is elapsed CPU time, which the browser has no clock for
 * that matches a native run), attribute diskfree (free disk space of the host file system), keyword path, and universe yes
 * (multi-partition runs are not available).
 */

const OPERATORS = ['<', '<=', '>', '>=', '==', '!=', '|^'] as const;
type Operator = (typeof OPERATORS)[number];

export class FixHalt extends Fix {
  readonly style = 'halt';
  readonly attribute: 'bondmax' | string;
  readonly op: Operator;
  readonly avalue: number;
  readonly error: 'hard' | 'soft' | 'continue';
  readonly message: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError(`fix ${id} halt: usage: fix ID group-ID halt N attribute operator avalue [keyword value ...]`);
    // As native LAMMPS does (src/fix_halt.cpp FixHalt::FixHalt at b332f28): N is read with utils::inumeric
    const n = inumeric(args[0]);
    if (n <= 0) throw new StyleError('Illegal fix halt command: nevery must be > 0');
    this.nevery = n;
    const attr = args[1];
    if (attr === 'tlimit') {
      throw new StyleError(`fix ${id} halt attribute tlimit is not supported by the browser engine: it stops a run by elapsed CPU time, which depends on the machine and is not reproducible here (use a v_name variable instead)`);
    }
    if (attr === 'diskfree') {
      throw new StyleError(`fix ${id} halt attribute diskfree is not supported by the browser engine: it reads the free space of the host file system (use a v_name variable instead)`);
    }
    if (attr !== 'bondmax' && !attr.startsWith('v_')) {
      throw new StyleError(`fix ${id} halt: attribute must be bondmax, tlimit, diskfree or v_name, got '${attr}'`);
    }
    if (attr.startsWith('v_') && attr.length === 2) throw new StyleError(`fix ${id} halt: missing variable name after v_`);
    this.attribute = attr;
    if (!(OPERATORS as readonly string[]).includes(args[2])) {
      throw new StyleError(`fix ${id} halt: operator must be one of < <= > >= == != |^, got '${args[2]}'`);
    }
    this.op = args[2] as Operator;
    this.avalue = parseNum(args[3], `fix ${id} halt avalue`);
    let error: 'hard' | 'soft' | 'continue' = 'soft';
    let message = true;
    for (let k = 4; k < args.length;) {
      const key = args[k];
      // As native LAMMPS does (src/fix_halt.cpp FixHalt::FixHalt at b332f28): a word that is not one of its keywords stops with "Unknown fix halt keyword".
      if (!['error', 'message', 'path', 'universe'].includes(key)) throw new StyleError(`Unknown fix halt keyword ${key}`);
      const v = args[k + 1];
      // As native LAMMPS does (src/fix_halt.cpp FixHalt::FixHalt at b332f28): "Illegal fix halt {keyword} command: missing argument(s)"
      if (v === undefined) throw missingArgs(`fix halt ${key}`);
      if (key === 'error') {
        if (v !== 'hard' && v !== 'soft' && v !== 'continue') throw new StyleError(`fix ${id} halt error must be hard, soft or continue`);
        error = v;
      } else if (key === 'message') {
        if (v !== 'yes' && v !== 'no') throw new StyleError(`fix ${id} halt message must be yes or no`);
        message = v === 'yes';
      } else if (key === 'path') {
        throw new StyleError(`fix ${id} halt path is only used with attribute diskfree, which the browser engine does not support`);
      } else if (key === 'universe') {
        if (v === 'yes') throw new StyleError(`fix ${id} halt universe yes is not supported by the browser engine (no multi-partition runs)`);
      }
      k += 2;
    }
    this.error = error;
    this.message = message;
  }

  /** The attribute's current value (bondmax: the longest bond of the system, minimum image). */
  private value(): number {
    if (this.attribute.startsWith('v_')) {
      const name = this.attribute.slice(2);
      if (!this.sys.vars.has(name)) throw new StyleError(`fix ${this.id} halt: variable ${name} is not defined`);
      return this.sys.vars.scalar(name, this.sys.formulaEnv);
    }
    return longestBond(this.sys);
  }

  /** Checked after a step on which the step count is a multiple of N (run/halt.ts requests the stop). */
  endOfStep(): void {
    const sys = this.sys;
    const step = sys.state.step;
    const x = this.value();
    if (!test(x, this.op, this.avalue)) return;
    const shown = formatNumber(x, '%g');
    if (this.message) sys.log(`Fix halt condition for fix-id ${this.id} met on step ${step} with value ${shown}`);
    if (this.error === 'hard') {
      throw new StyleError(`Fix halt condition for fix-id ${this.id} met on step ${step} with value ${shown}`);
    }
    requestHalt(sys, this.error === 'soft');
  }
}

/** The comparison of fix_halt.html; XOR is true when exactly one of the two values is 0.0. */
export const test = (x: number, op: Operator, v: number): boolean => {
  switch (op) {
    case '<': return x < v;
    case '<=': return x <= v;
    case '>': return x > v;
    case '>=': return x >= v;
    case '==': return x === v;
    case '!=': return x !== v;
    case '|^': return (x === 0) !== (v === 0);
  }
};

/** Longest distance between the two atoms of any bond (bond topology atom IDs, minimum image). */
const longestBond = (sys: System): number => {
  const s = sys.state;
  const b = s.topo.bonds;
  let max = 0;
  const d = [0, 0, 0];
  for (let k = 0; k < b.n; k++) {
    const i = sys.indexOfId(b.atoms[2 * k]);
    const j = sys.indexOfId(b.atoms[2 * k + 1]);
    if (i < 0 || j < 0) throw new StyleError(`fix halt bondmax: bond atom ${b.atoms[2 * k]} or ${b.atoms[2 * k + 1]} is missing`);
    for (let c = 0; c < 3; c++) d[c] = s.x[3 * j + c] - s.x[3 * i + c];
    sys.geom.minimumImage(d);
    const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (r > max) max = r;
  }
  return max;
};
