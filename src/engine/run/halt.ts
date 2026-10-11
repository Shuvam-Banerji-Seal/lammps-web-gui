import type { System } from '../system';

/*
 * Halt requests of fix halt (docs.lammps.org/fix_halt.html). A halt is requested by the fix's end-of-step
 * check and ends the current run or minimization after the step on which the condition was met. The
 * request and the soft state live here, not on System, so the run loops only ask whether a halt is
 * pending (haltRequested) and the run commands ask whether a soft halt blocks them (softHalted).
 *
 * fix_halt.html: "If its value is *soft*, LAMMPS will exit the current run, but continue to execute
 * subsequent commands in the input script. However, additional :doc:`run <run>` or :doc:`minimize
 * <minimize>` commands will be skipped." "To re-enable regular runs after *fix halt* stopped a run, you need
 * to issue a :doc:`timer timeout unlimited <timer>` command."
 * "If its value is *continue*, the behavior is the same as for *soft*, except subsequent :doc:`run <run>`
 * or :doc:`minimize <minimize>` commands are executed."
 */

interface HaltState {
  requested: boolean;
  soft: boolean;
}

const states = new WeakMap<System, HaltState>();

const stateOf = (sys: System): HaltState => {
  let st = states.get(sys);
  if (!st) {
    st = { requested: false, soft: false };
    states.set(sys, st);
  }
  return st;
};

/** A fix halt condition was met: the run stops after this step; soft = later runs are skipped. */
export const requestHalt = (sys: System, soft: boolean): void => {
  const st = stateOf(sys);
  st.requested = true;
  if (soft) st.soft = true;
};

/** True while a halt requested by a fix is pending (checked by the run loops after each step). */
export const haltRequested = (sys: System): boolean => stateOf(sys).requested;

/** Start of a run or minimization: the per-run request is cleared. */
export const clearHaltRequest = (sys: System): void => {
  stateOf(sys).requested = false;
};

/** A soft halt blocks run and minimize commands until timer timeout off or timer timeout unlimited is given. */
export const softHalted = (sys: System): boolean => stateOf(sys).soft;

/** timer timeout off|unlimited (timer.html) clears a soft halt. */
export const clearSoftHalt = (sys: System): void => {
  stateOf(sys).soft = false;
};

/**
 * fix halt during a minimization (fix_halt.html says that a run "can be dynamics or minimization iterations"):
 * the minimizer calls this after each iteration's step increment, with the halt fixes' own
 * checks on their every-N steps.
 */
export const checkMinHalt = (sys: System): void => {
  for (const f of sys.fixes) {
    if (f.style === 'halt' && sys.state.step % f.nevery === 0) f.endOfStep?.();
  }
};
