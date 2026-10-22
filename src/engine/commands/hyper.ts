import type { Ctx, Handler } from './args';
import { int, num } from './args';
import { StyleError } from '../force/types';
import { minimize } from '../run/min';
import { initRun, yieldNow } from '../run/verlet';
import { clearHaltRequest, softHalted } from '../run/halt';
import { RunCancelled } from '../errors';
import type { System } from '../system';
import { ensureThermoComputes } from './run';
import { restartDue } from './restart';

/*
 * The hyper command — implemented from docs.lammps.org/hyper.html (plans/lammps-docs-live/hyper.rst). Quoted
 * sentences are copied character for character from that page:
 *
 *   "Dynamics (with the bias potential) is run continuously, stopping every
 *   *Nevent* steps to check if a transition event has occurred."
 *   "The specified *N* for total steps must be a multiple of *Nevent*\ ."
 *   "A quench is an energy minimization and is performed by whichever
 *   algorithm has been defined by the :doc:`min_style <min_style>` command."
 *   "If this happens, the list of bonds is reset, since some bond pairs
 *   are likely now too far apart, and new pairs are likely close enough
 *   to be considered a bond.  The pre-quenched state of the
 *   system (coordinates and velocities) is restored, and dynamics continue."
 *   "The *dump* keyword can be used to trigger a specific dump command with
 *   the specified *dump-ID* to output a snapshot each time an event is
 *   detected."
 *   "The *rebond* keyword will force a reset of the bond
 *   list every *Nrebond* steps, even if an event has not occurred."
 *
 * Loop (the logic block of the page): a quench and the bond list at the start; then for each Nevent-step
 * dynamics segment: quench, check the event (compute event/displace against the reference of the last event),
 * reset the bonds and the reference if there was one, and restore the pre-quench state. The last quench is not
 * restored. A quench is a minimize, so its iterations advance the timestep counter and print thermo rows like a
 * minimize; every dynamics segment starts with a thermo header and its setup row (measured with native LAMMPS,
 * black box). Dynamics prints the thermo rows on the thermo interval, including its last step.
 *
 * fix hyper/local (local hyperdynamics, src/engine-gpl/hyper/fix_hyper_local.ts) takes the same loop: its bond
 * list is built by buildBonds(natom) (natom = atoms of the event, 0 otherwise), as hyper.cpp calls
 * build_bond_list(ecount).
 *
 * Refused (EngineError): compute other than event/displace, thermo v_name output, N = 0, N not a multiple of Nevent.
 */

/** The bias fixes of the hyper command (fix hyper/global: fix/hyper_global.ts; fix hyper/local: engine-gpl). */
interface HyperBiasFix {
  style: string;
  id: string;
  buildBonds(natom: number): void;
  beginRun(step: number): void;
  recordEvent(nflag: number, drift: number): void;
  recordDrift(drift: number): void;
  setQuench(on: boolean): void;
  summary(): { hyperTime: number; events: number; eventAtoms: number };
  setupPreReverse?(): void;
  setup(): void;
}

interface EventCompute {
  style: string;
  hyperOwned: boolean;
  storeReference(): void;
  displacement(): { count: number; maxMove: number; flags: Uint8Array };
}

interface QuenchOpts { etol: number; ftol: number; maxiter: number; maxeval: number }

/** Saved state of a restore: positions, velocities, image flags and the timestep. */
interface Saved { x: Float64Array; v: Float64Array; image: Int32Array; step: number }

const saveState = (sys: System): Saved => {
  const s = sys.state;
  return { x: s.x.slice(0, 3 * s.n), v: s.v.slice(0, 3 * s.n), image: s.image.slice(0, 3 * s.n), step: s.step };
};

const restoreState = (sys: System, k: Saved): void => {
  const s = sys.state;
  s.x.set(k.x);
  s.v.set(k.v);
  s.image.set(k.image);
  s.step = k.step;
  sys.bump();
};

/** Forces and fix setup for the current positions (a run's setup, used after a restore). */
const setupForces = (sys: System): void => {
  sys.setupNeighbors();
  for (const f of sys.fixes) f.setupPreForce?.();
  sys.computeForcesInRun(sys.computeFlags());
  sys.forcesCurrent();
  for (const f of sys.fixes) f.setupPreReverse?.();
  for (const f of sys.fixes) f.setup();
  sys.refreshComputes();
};

const emitHeader = (sys: System): void => {
  const th = sys.thermo;
  sys.io.emit({ kind: 'thermo-header', keywords: [...th.keywords], labels: th.labels(), units: sys.units.style });
};

const emitRow = (sys: System): void => {
  sys.io.emit({ kind: 'thermo', row: sys.thermo.row() });
};

/** A quench: minimize with the hyper min settings, printing thermo like minimize does (the bias fix knows it). */
const quench = async (sys: System, cancelled: () => boolean, opts: QuenchOpts, hf: HyperBiasFix): Promise<void> => {
  hf.setQuench(true);
  try {
    await quenchRun(sys, cancelled, opts);
  } finally {
    hf.setQuench(false);
  }
};

const quenchRun = async (sys: System, cancelled: () => boolean, opts: QuenchOpts): Promise<void> => {
  initRun(sys);
  const s = sys.state;
  const th = sys.thermo;
  const first = s.step;
  emitHeader(sys);
  let lastThermo = -1;
  await minimize(sys, opts, {
    cancelled,
    thermo: (iterDone) => {
      if ((iterDone || th.due(s.step, first, Number.MAX_SAFE_INTEGER)) && s.step !== lastThermo) {
        emitRow(sys);
        lastThermo = s.step;
      }
    },
  });
};

/** One dynamics segment of nsteps with the bias fix's post_force (the velocity-Verlet order of run/verlet.ts). */
const dynamics = async (sys: System, nsteps: number, cancelled: () => boolean): Promise<void> => {
  initRun(sys, { verlet: true });
  const s = sys.state;
  const fixes = sys.fixes;
  const nb = sys.nb;
  const th = sys.thermo;
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
  let lastYield = performance.now();
  for (let k = 0; k < nsteps; k++) {
    if (cancelled()) break;
    s.step++;
    sys.refreshComputes();
    for (const f of fInitial) f.initialIntegrate!();
    for (const f of fPostInt) f.postIntegrate!();
    sys.bump();
    nb.forceBuildStep = restartDue(sys, s.step) ? s.step : -1;
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
    if (th.every > 0 && s.step % th.every === 0) emitRow(sys);
    for (const d of sys.dumps) if (d.due(s.step, false)) d.write();
    if (performance.now() - lastYield > 30) {
      await yieldNow();
      lastYield = performance.now();
    }
  }
  for (const f of fixes) f.postRun?.();
};

/** hyper N Nevent fix-ID compute-ID [min etol ftol maxiter maxeval] [dump dump-ID] [rebond Nrebond]. */
const hyper: Handler = async (ctx: Ctx, a) => {
  const { sys, session } = ctx;
  if (a.length < 4) throw new StyleError('usage: hyper N Nevent fix-ID compute-ID keyword values ...');
  const nsteps = int(a[0], 'hyper N');
  const nevent = int(a[1], 'hyper Nevent');
  const [fixId, compId] = [a[2], a[3]];
  if (nsteps <= 0) throw new StyleError('hyper: N must be > 0 (the browser engine does not run an empty hyper command)');
  if (nevent < 1) throw new StyleError('hyper: Nevent must be >= 1');
  if (nsteps % nevent !== 0) throw new StyleError('hyper: N must be a multiple of Nevent');
  const quenchOpts: QuenchOpts = { etol: 0.1, ftol: 0.1, maxiter: 40, maxeval: 50 };
  const dumpIds: string[] = [];
  let rebond = 0;
  for (let k = 4; k < a.length;) {
    switch (a[k]) {
      case 'min': {
        quenchOpts.etol = num(a[k + 1], 'hyper min etol');
        quenchOpts.ftol = num(a[k + 2], 'hyper min ftol');
        quenchOpts.maxiter = int(a[k + 3], 'hyper min maxiter');
        quenchOpts.maxeval = int(a[k + 4], 'hyper min maxeval');
        k += 5;
        break;
      }
      case 'dump': {
        if (a[k + 1] === undefined) throw new StyleError('hyper dump needs a dump-ID');
        dumpIds.push(a[k + 1]);
        k += 2;
        break;
      }
      case 'rebond': {
        rebond = int(a[k + 1], 'hyper rebond Nrebond');
        if (rebond < 1 || rebond % nevent !== 0) throw new StyleError('hyper rebond: Nrebond must be a multiple of Nevent');
        k += 2;
        break;
      }
      default: throw new StyleError(`hyper: unknown keyword '${a[k]}' (min, dump or rebond)`);
    }
  }
  if (quenchOpts.etol < 0 || quenchOpts.ftol < 0) throw new StyleError('hyper min tolerances must be >= 0');
  const fix = sys.fixes.find((f) => f.id === fixId);
  if (!fix) throw new StyleError(`hyper: fix ${fixId} does not exist`);
  if (fix.style !== 'hyper/global' && fix.style !== 'hyper/local') {
    throw new StyleError(`hyper: fix ${fixId} (${fix.style}) is not a bond-boost bias fix (fix hyper/global or hyper/local)`);
  }
  const comp = sys.computes.find((c) => c.id === compId);
  if (!comp) throw new StyleError(`hyper: compute ${compId} does not exist`);
  if (comp.style !== 'event/displace') throw new StyleError(`hyper: compute ${compId} (${comp.style}) is not an event computer; the browser engine supports compute event/displace`);
  for (const d of dumpIds) {
    if (!sys.dumps.some((x) => x.id === d)) throw new StyleError(`hyper: dump ${d} does not exist`);
  }
  if (!sys.hasBox) throw new StyleError('hyper: no simulation box is defined');
  // fix_halt.html: a soft halt skips later run and minimize commands
  if (softHalted(sys)) return;
  clearHaltRequest(sys);
  ensureThermoComputes(sys);
  const th = sys.thermo;
  if (th.everyVar) throw new StyleError('hyper: thermo v_name output is not supported by the browser engine');

  const hf = fix as unknown as HyperBiasFix;
  const ec = comp as unknown as EventCompute;
  ec.hyperOwned = true;
  try {
    await hyperLoop(sys, session, hf, ec, { nsteps, nevent, quenchOpts, dumpIds, rebond });
  } finally {
    ec.hyperOwned = false;
    sys.run.inRun = false;
  }
};

interface HyperLoop {
  nsteps: number;
  nevent: number;
  quenchOpts: QuenchOpts;
  dumpIds: string[];
  rebond: number;
}

const hyperLoop = async (sys: System, session: Ctx['session'], hf: HyperBiasFix, ec: EventCompute, o: HyperLoop): Promise<void> => {
  const { nsteps, nevent, quenchOpts, dumpIds, rebond } = o;
  const cancelled = () => session.isCancelled;
  const s = sys.state;
  const start = s.step;
  sys.run = { inRun: true, firstStep: start, lastStep: start + nsteps, beginStep: start, endStep: start + nsteps, t0: performance.now(), ranOnce: true };
  hf.beginRun(start);

  // initial quench: bonds and the event reference come from the quenched state; the start state is restored
  const st0 = saveState(sys);
  await quench(sys, cancelled, quenchOpts, hf);
  hf.buildBonds(0);
  ec.storeReference();
  restoreState(sys, st0);
  setupForces(sys);
  if (cancelled()) throw new RunCancelled();

  for (let done = 0; done < nsteps; done += nevent) {
    // dynamics segment: its setup row (the restored state), then the thermo interval
    initRun(sys, { verlet: true });
    emitHeader(sys);
    sys.refreshComputes();
    emitRow(sys);
    sys.io.emit({ kind: 'run', from: s.step, to: s.step + nevent, dt: s.dt, units: sys.units.style });
    await dynamics(sys, nevent, cancelled);
    if (cancelled()) break;
    // quench of the segment's end state; check the event against the reference
    const saved = saveState(sys);
    await quench(sys, cancelled, quenchOpts, hf);
    if (cancelled()) break;
    const ev = ec.displacement();
    let reset = false;
    if (ev.count > 0) {
      hf.recordEvent(ev.count, ev.maxMove);
      ec.storeReference();
      for (const d of dumpIds) sys.dumps.find((x) => x.id === d)?.write();
      reset = true;
    }
    if (rebond > 0 && (done + nevent) % rebond === 0) {
      reset = true;
      // a rebond check without an event still reports the largest move of the quench (measured with native LAMMPS)
      if (ev.count === 0) hf.recordDrift(ev.maxMove);
      // the reference is the quenched state of the last check that reset the bonds (measured with native LAMMPS)
      ec.storeReference();
    }
    if (reset) hf.buildBonds(ev.count);
    // As native LAMMPS does (src/REPLICA/hyper.cpp command at b332f28): every quench, the last one too, is followed by
    // fix_event->restore_state_quench(), which restores x, v and image of the pre-quench state (the forces stay those of
    // the quench); the next segment's dynamics starts with a setup
    restoreState(sys, saved);
    if (done + nevent < nsteps) setupForces(sys);
    else sys.forcesCurrent();
  }
  if (cancelled()) throw new RunCancelled();
  sys.run.inRun = false;
  const seconds = (performance.now() - sys.run.t0) / 1000;
  const stats = hf.summary();
  const cumTime = stats.hyperTime;
  sys.log([
    'Final hyper stats ...',
    `  hyper time = ${cumTime}`,
    `  time boost factor = ${cumTime / (nsteps * s.dt)}`,
    `  event timesteps = ${stats.events}`,
    `  # of atoms in events = ${stats.eventAtoms}`,
  ].join('\n'));
  sys.io.emit({ kind: 'done', steps: nsteps, seconds, backend: session.backendLabel });
};

export const HYPER_COMMANDS: Record<string, Handler> = { hyper };
