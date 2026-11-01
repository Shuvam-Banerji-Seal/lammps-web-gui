import { COMM, MODIFY, NEIGH, OUTPUT } from '../timer';
import type { System } from '../system';
import { StyleError } from '../force/types';
import { restartDue } from '../commands/restart';
import { haltRequested } from './halt';
import { checkDynamicGroupFixes } from '../group';
import { runLabelCheck } from '../labelcheck';

/*
 * The velocity-Verlet run loop — docs.lammps.org/Developer_flow.html:
 *
 *   loop over N timesteps:
 *     fix->initial_integrate(); fix->post_integrate()
 *     nflag = neighbor->decide()
 *     if nflag: fix->pre_exchange(); domain->pbc(); domain->reset_box();
 *               ...; fix->pre_neighbor(); neighbor->build(); fix->post_neighbor()
 *     else:     comm->forward_comm()
 *     force_clear(); fix->pre_force(); pair/bond/angle/dihedral/improper/
 *     kspace compute(); fix->pre_reverse(); comm->reverse_comm()
 *     fix->post_force(); fix->final_integrate(); fix->end_of_step()
 *     if any output on this step: output->write()
 *   fix->post_run()
 *
 * "Periodic boundary conditions are then applied ... Note that this is not
 * done every timestep, but only when neighbor lists are rebuilt. ... It is
 * also why dumped atom coordinates may be slightly outside the simulation
 * box if not dumped on a step where the neighbor lists are rebuilt."
 * docs.lammps.org/run.html: "A value of N = 0 is acceptable; only the
 * thermodynamics of the system are computed and printed without taking a
 * timestep." docs.lammps.org/thermo.html: thermo output "on timesteps that
 * are a multiple of N and at the beginning and end of a simulation".
 * Fixes with an end_of_step() hook and a period ("every N steps") are called
 * only on multiples of their nevery.
 */

export interface RunHooks {
  /** Cooperative cancellation (Stop button). */
  cancelled(): boolean;
  /** Called after every step (dumps, frames); return value ignored. */
  afterStep(step: number): void;
  /** Called on the setup step (step 0 of this run) after forces. */
  afterSetup(): void;
  /** Milliseconds of work between yields to the event loop (keeps the page responsive). */
  yieldMs: number;
  /** run pre no: the neighbor lists and forces of the previous run are kept; no setup (run.html "pre"). */
  skipSetup?: boolean;
  /**
   * Whether a compute may read per-atom energy or virial after this step (its thermo or dump output is due). With
   * only quiet fixes (peratomQuiet), the other steps leave them out of the force evaluation, as native
   * Integrate::ev_set does for steps no compute registered (src/integrate.cpp at b332f28). Absent: every step
   * tallies them while a compute needs them.
   */
  peratomDue?: (step: number) => boolean;
}

/**
 * Fix styles that invoke no compute between outputs when none of their arguments is a variable: native's fixes of this
 * list call modify->addstep_compute only for a variable argument (e.g. src/fix_setforce.cpp, fix_langevin.cpp,
 * fix_temp_berendsen.cpp at b332f28); their temperature and pressure computes need no per-atom energy or virial.
 */
const QUIET_FIXES = new Set([
  'nve', 'nve/limit', 'nve/sphere', 'nve/asphere', 'nve/noforce', 'nvt', 'npt', 'nph', 'nvt/sllod',
  'langevin', 'temp/berendsen', 'temp/rescale', 'temp/csvr', 'temp/csld', 'enforce2d', 'momentum', 'recenter', 'viscous',
  'setforce', 'addforce', 'aveforce', 'gravity', 'efield', 'spring', 'spring/self', 'shake', 'rattle',
  'rigid', 'rigid/nve', 'rigid/nvt', 'rigid/npt', 'rigid/nph', 'rigid/small', 'rigid/nve/small', 'rigid/nvt/small',
  'rigid/npt/small', 'rigid/nph/small', 'wall/lj93', 'wall/lj126', 'wall/lj1043', 'wall/harmonic', 'wall/reflect',
  'deform', 'nve/eff', 'nvt/eff', 'npt/eff', 'nph/eff',
]);

/**
 * Whether per-atom energy and virial are read only at thermo and dump output: every fix is quiet (QUIET_FIXES, no
 * variable argument) and no group is dynamic (a dynamic group's variable is evaluated on its own steps).
 */
export const peratomQuiet = (sys: System): boolean => {
  if (sys.groups.dynamic.size) return false;
  for (const f of sys.fixes) {
    if (!QUIET_FIXES.has(f.style)) return false;
    if (((f as unknown as { args?: readonly string[] }).args ?? []).some((a) => a.startsWith('v_'))) return false;
  }
  return true;
};

/**
 * Modify::setup's compute setup, before the fixes' setup: "compute setup needs to come before fix setup b/c NH fixes
 * need DOF of temperature computes" (src/modify.cpp at b332f28). The temperature computes count their degrees of
 * freedom again, so atoms a fix added or removed in setup_pre_force (the bond particles of fix srp) are counted.
 */
export const computeSetup = (sys: System): void => {
  for (const c of sys.computes) if (c.tempFlag) (c as unknown as { dofCompute?: () => void }).dofCompute?.();
};

/**
 * Fix and compute initialization shared by run and minimize. opts.verlet: the init of a dynamics run with the Verlet
 * integrator (Verlet::init, not Respa::init or Min::init), which warns when no fix integrates the atoms.
 */
export const initRun = (sys: System, opts: { verlet?: boolean } = {}): void => {
  const s = sys.state;
  // a run's init computes the pair style's tail correction from the current atom counts (Pair::init) and keeps it
  // until a pair reinit or the next run (ForceField.tailHold)
  sys.ff.tailHold = null;
  sys.ff.holdTailAtInit = true;
  // force->init comes before atom->init (the per-type masses) and modify->init (computes, then fixes)
  sys.initPair();
  if (!s.rmass) {
    for (let t = 1; t <= s.ntypes; t++) {
      // As native LAMMPS does (src/atom.cpp Atom::check_mass at b332f28)
      if (!(s.massByType[t] > 0)) throw new StyleError(`Not all per-type masses are set. Type ${t} is missing.`);
    }
  }
  for (const c of sys.computes) c.invalidate();
  for (const c of sys.computes) { c.init(); c.initialized = true; }
  sys.checkDynamicGroups();
  for (const f of sys.fixes) f.init?.();
  checkDynamicGroupFixes(sys.groups, sys.fixes);
  // As native LAMMPS does (src/verlet.cpp Verlet::init at b332f28): "if (fix->time_integrate) do_time_integrate = true;"
  // over every fix, else "No fixes with time integration, atoms won't move" + utils::errorurl(28)
  if (opts.verlet && !sys.fixes.some((f) => f.timeIntegrate)) {
    sys.warn("No fixes with time integration, atoms won't move\nFor more information see https://docs.lammps.org/err0028");
  }
  // native LAMMPS: "WARNING: One or more atoms are time integrated more than once"
  const integ = sys.fixes.filter((f) => f.timeIntegrate);
  if (integ.length > 1) {
    let twice = false;
    for (let i = 0; i < s.n && !twice; i++) {
      let c = 0;
      for (const f of integ) if (s.mask[i] & f.groupBit) c++;
      twice = c > 1;
    }
    if (twice) sys.warn('One or more atoms are time integrated more than once');
  }
  // temperature computes need the fixes' dof (shake, rigid) — recount after fixes init
  for (const c of sys.computes) if (c.tempFlag) (c as unknown as { dofCompute?: () => void }).dofCompute?.();
  sys.thermo.init();
  // Output::init: the thermo output, then each dump (src/output.cpp at b332f28)
  for (const d of sys.dumps) d.init();
};

/** Setup half of a run: neighbor lists, forces, fix setup (Verlet::setup()). */
export const setupRun = (sys: System): void => {
  sys.setupNeighbors(true);
  for (const f of sys.fixes) f.setupPreForce?.();
  sys.computeForcesInRun(sys.computeFlags());
  sys.forcesCurrent();
  sys.assignDynamicGroups(true);
  for (const f of sys.fixes) f.setupPreReverse?.();
  computeSetup(sys);
  for (const f of sys.fixes) f.setup();
  // labelmap check_labels: once, at the end of the setup (native Modify::setup)
  runLabelCheck(sys);
  sys.refreshComputes();
};

/** A macrotask yield that is not clamped like nested setTimeout(0) (MessageChannel when available). */
export const yieldNow = (): Promise<void> => {
  if (typeof MessageChannel === 'undefined') return new Promise((r) => setTimeout(r, 0));
  return new Promise((r) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => { ch.port1.close(); r(); };
    ch.port2.postMessage(0);
  });
};

export const runVerlet = async (sys: System, nsteps: number, hooks: RunHooks): Promise<number> => {
  const s = sys.state;
  const fixes = sys.fixes;
  const nb = sys.nb;
  const has = <K extends keyof (typeof fixes)[number]>(k: K) => fixes.filter((f) => typeof f[k] === 'function');
  const fInitial = has('initialIntegrate');
  const fPostInt = has('postIntegrate');
  const fPreEx = has('preExchange');
  const fPreNeigh = has('preNeighbor');
  const fPostNeigh = has('postNeighbor');
  const fPreForce = has('preForce');
  const fPreRev = has('preReverse');
  const fPostForce = has('postForce');
  const fFinal = has('finalIntegrate');
  const fEnd = has('endOfStep');
  const flags = sys.computeFlags();
  // per-atom energy and virial only on the steps a compute may read them (hooks.peratomDue)
  const peratomDue = (flags.eatom || flags.vatom || flags.cvatom) && hooks.peratomDue && peratomQuiet(sys) ? hooks.peratomDue : null;
  const lean = { ...flags, eatom: false, vatom: false, cvatom: false, step: true };
  const timer = sys.timer;

  if (!hooks.skipSetup) setupRun(sys);
  hooks.afterSetup();
  let taken = 0;
  let lastYield = performance.now();
  for (let k = 0; k < nsteps; k++) {
    if (hooks.cancelled()) break;
    s.step++;
    timer.mark();
    // forces still belong to the current positions here (barostats read the pressure);
    // only compute values must be refreshed for the new step
    sys.refreshComputes();
    for (const f of fInitial) f.initialIntegrate!();
    for (const f of fPostInt) f.postIntegrate!();
    sys.bump();
    timer.stamp(MODIFY);
    const fReneigh = fixes.some((f) => f.reneighborDue?.(s.step) === true);
    nb.forceBuildStep = restartDue(sys, s.step) ? s.step : -1;
    nb.fixBuildStep = fReneigh ? s.step : -1;
    if (nb.decide(s.step, s, sys.geom)) {
      for (const f of fPreEx) f.preExchange!();
      timer.stamp(MODIFY);
      sys.pbc();
      sys.sortAtoms(false);
      timer.stamp(COMM);
      for (const f of fPreNeigh) f.preNeighbor!();
      timer.stamp(MODIFY);
      nb.build(s, sys.geom, s.step);
      timer.stamp(NEIGH);
      for (const f of fPostNeigh) f.postNeighbor!();
      timer.stamp(MODIFY);
    } else {
      nb.forwardComm(s, sys.geom);
      timer.stamp(COMM);
    }
    for (const f of fPreForce) f.preForce!();
    timer.stamp(MODIFY);
    if (peratomDue && !peratomDue(s.step)) {
      sys.computeForcesInRun(lean);
      sys.peratomSkippedStep = s.step;
    } else {
      sys.computeForcesInRun({ ...flags, step: true });
      sys.peratomSkippedStep = -1;
    }
    sys.forcesCurrent();
    sys.assignDynamicGroups(false);
    for (const f of fPreRev) f.preReverse!();
    for (const f of fPostForce) f.postForce!();
    for (const f of fFinal) f.finalIntegrate!();
    for (const f of fEnd) if (s.step % f.nevery === 0) f.endOfStep!();
    timer.stamp(MODIFY);
    sys.refreshComputes();
    taken++;
    hooks.afterStep(s.step);
    timer.stamp(OUTPUT);
    // fix halt (fix_halt.html): the run ends after the step on which the condition was met
    if (haltRequested(sys)) break;
    if (hooks.yieldMs >= 0 && performance.now() - lastYield > hooks.yieldMs) {
      await yieldNow();
      lastYield = performance.now();
    }
  }
  // a run stopped early (Stop button) may end on a step without per-atom terms: a compute read after the loop
  // recomputes them (System.peratomEnergy), as between runs
  sys.peratomSkippedStep = -1;
  for (const f of fixes) f.postRun?.();
  return taken;
};
