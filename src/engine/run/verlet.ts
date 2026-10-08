import type { System } from '../system';
import { StyleError } from '../force/types';

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
}

/** Fix and compute initialization shared by run and minimize. */
export const initRun = (sys: System): void => {
  const s = sys.state;
  if (!s.rmass) {
    for (let t = 1; t <= s.ntypes; t++) {
      if (!(s.massByType[t] > 0)) throw new StyleError(`not all per-type masses are set (type ${t}); use the mass command`);
    }
  }
  for (const c of sys.computes) c.invalidate();
  for (const c of sys.computes) c.init();
  sys.checkDynamicGroups();
  for (const f of sys.fixes) f.init?.();
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
};

/** Setup half of a run: neighbor lists, forces, fix setup (Verlet::setup()). */
export const setupRun = (sys: System): void => {
  sys.setupNeighbors();
  sys.sortAtoms(true);
  sys.computeForcesInRun(sys.computeFlags());
  sys.forcesCurrent();
  sys.assignDynamicGroups(true);
  for (const f of sys.fixes) f.setup();
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

  setupRun(sys);
  hooks.afterSetup();
  let taken = 0;
  let lastYield = performance.now();
  for (let k = 0; k < nsteps; k++) {
    if (hooks.cancelled()) break;
    s.step++;
    // forces still belong to the current positions here (barostats read the pressure);
    // only compute values must be refreshed for the new step
    sys.refreshComputes();
    for (const f of fInitial) f.initialIntegrate!();
    for (const f of fPostInt) f.postIntegrate!();
    sys.bump();
    if (nb.decide(s.step, s, sys.geom)) {
      for (const f of fPreEx) f.preExchange!();
      sys.pbc();
      sys.sortAtoms(false);
      for (const f of fPreNeigh) f.preNeighbor!();
      nb.build(s, sys.geom, s.step);
      for (const f of fPostNeigh) f.postNeighbor!();
    } else {
      nb.forwardComm(s, sys.geom);
    }
    for (const f of fPreForce) f.preForce!();
    sys.computeForcesInRun({ ...flags, step: true });
    sys.forcesCurrent();
    sys.assignDynamicGroups(false);
    for (const f of fPreRev) f.preReverse!();
    for (const f of fPostForce) f.postForce!();
    for (const f of fFinal) f.finalIntegrate!();
    for (const f of fEnd) if (s.step % f.nevery === 0) f.endOfStep!();
    sys.refreshComputes();
    taken++;
    hooks.afterStep(s.step);
    if (hooks.yieldMs >= 0 && performance.now() - lastYield > hooks.yieldMs) {
      await yieldNow();
      lastYield = performance.now();
    }
  }
  for (const f of fixes) f.postRun?.();
  return taken;
};
