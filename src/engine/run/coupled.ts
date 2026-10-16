import type { System } from '../system';
import type { Fix } from '../fix/fix';
import { StyleError } from '../force/types';
import { restartDue } from '../commands/restart';
import { haltRequested } from './halt';
import { yieldNow } from './verlet';

/*
 * The lockstep velocity-Verlet loop for coupled replicas (partitions). A path-integral fix (fix pimd, ported in
 * src/engine-gpl/pimd/) couples the beads, which are the replicas of the session: its post_force needs the positions
 * and forces of every bead in the same step, as native LAMMPS does through its inter-partition exchange (checked
 * against src/REPLICA/fix_pimd_nvt.cpp at b332f28, comm_exec and post_force). So each step runs every replica through
 * the same phase (initial_integrate, neighbor decision, force evaluation, post_force, final_integrate, end_of_step)
 * before the next phase starts. Setup is phased the same way: every replica's forces are computed before any fix setup.
 *
 * Within a phase the fixes run in their definition order, each on every replica before the next fix (native runs the
 * fixes of one partition in that order; the partitions share the same fix list, so this is the same order per bead).
 * A single replica gives the same steps as run/verlet.ts (runVerlet).
 */

export interface CoupledHooks {
  /** Cooperative cancellation (Stop button). */
  cancelled(): boolean;
  /** Called for every replica on the setup step (step 0 of the run), after the setup forces. */
  afterSetup(sys: System): void;
  /** Called for every replica after every step. */
  afterStep(sys: System, step: number): void;
  /** Milliseconds of work between yields to the event loop. */
  yieldMs: number;
}

/**
 * Calls one fix phase for every replica: fix k of every replica (in definition order), then fix k + 1. The replicas
 * must define the same fixes in the same order (the same input); otherwise the phases could not line up.
 */
const fixPhase = (systems: readonly System[], call: (f: Fix) => void): void => {
  const n = systems[0].fixes.length;
  if (systems.some((sys) => sys.fixes.length !== n)) throw new StyleError('coupled replicas (fix pimd) must define the same fixes in the same order');
  for (let k = 0; k < n; k++) for (const sys of systems) call(sys.fixes[k]);
};

/** Runs nsteps of velocity Verlet on all replicas in lockstep; returns the steps taken. */
export const runVerletCoupled = async (systems: readonly System[], nsteps: number, hooks: CoupledHooks): Promise<number> => {
  // setup: every replica's forces first, then the fix setups (a ring fix reads the forces of all beads)
  for (const sys of systems) {
    sys.setupNeighbors();
    sys.sortAtoms(true);
    sys.computeForcesInRun(sys.computeFlags());
    sys.forcesCurrent();
    sys.assignDynamicGroups(true);
  }
  fixPhase(systems, (f) => f.setupPreForce?.());
  fixPhase(systems, (f) => f.setupPreReverse?.());
  fixPhase(systems, (f) => f.setup());
  for (const sys of systems) {
    sys.refreshComputes();
    hooks.afterSetup(sys);
  }
  const flags = systems.map((sys) => sys.computeFlags());
  let taken = 0;
  let lastYield = performance.now();
  for (let k = 0; k < nsteps; k++) {
    if (hooks.cancelled()) break;
    for (const sys of systems) sys.state.step++;
    for (const sys of systems) sys.refreshComputes();
    fixPhase(systems, (f) => f.initialIntegrate?.());
    fixPhase(systems, (f) => f.postIntegrate?.());
    for (const sys of systems) sys.bump();
    // the neighbor decision of every replica, then the rebuilds (or the forward communication)
    const decided = systems.map((sys) => {
      const s = sys.state;
      const fReneigh = sys.fixes.some((f) => f.reneighborDue?.(s.step) === true);
      sys.nb.forceBuildStep = restartDue(sys, s.step) ? s.step : -1;
      sys.nb.fixBuildStep = fReneigh ? s.step : -1;
      return sys.nb.decide(s.step, s, sys.geom);
    });
    systems.forEach((sys, r) => {
      const s = sys.state;
      if (decided[r]) {
        for (const f of sys.fixes) f.preExchange?.();
        sys.pbc();
        sys.sortAtoms(false);
        for (const f of sys.fixes) f.preNeighbor?.();
        sys.nb.build(s, sys.geom, s.step);
        for (const f of sys.fixes) f.postNeighbor?.();
      } else {
        sys.nb.forwardComm(s, sys.geom);
      }
    });
    systems.forEach((sys, r) => {
      sys.computeForcesInRun({ ...flags[r], step: true });
      sys.forcesCurrent();
      sys.assignDynamicGroups(false);
    });
    fixPhase(systems, (f) => f.preForce?.());
    fixPhase(systems, (f) => f.preReverse?.());
    fixPhase(systems, (f) => f.postForce?.());
    fixPhase(systems, (f) => f.finalIntegrate?.());
    fixPhase(systems, (f) => { if (systems[0].state.step % f.nevery === 0) f.endOfStep?.(); });
    for (const sys of systems) sys.refreshComputes();
    taken++;
    for (const sys of systems) hooks.afterStep(sys, sys.state.step);
    // fix halt: the run ends after the step on which the condition was met
    if (systems.some((sys) => haltRequested(sys))) break;
    if (hooks.yieldMs >= 0 && performance.now() - lastYield > hooks.yieldMs) {
      await yieldNow();
      lastYield = performance.now();
    }
  }
  for (const sys of systems) for (const f of sys.fixes) f.postRun?.();
  return taken;
};
