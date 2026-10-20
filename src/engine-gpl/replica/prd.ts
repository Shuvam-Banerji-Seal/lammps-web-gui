/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/prd.cpp and prd.h (PRD::command, options, dephase, dynamics, quench, check_event, share_event,
 * log_event, replicate), src/REPLICA/fix_event_prd.cpp (FixEventPRD::store_event_prd), and src/REPLICA/compute_event_displace.cpp
 * (the event check) at commit b332f28. Copyright Sandia Corporation (GNU General Public License). Contributing author:
 * Mike Brown (SNL). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Parallel replica dynamics (docs.lammps.org/prd.html): "Parallel replica dynamics (PRD) is a method for performing
 * accelerated dynamics for systems where transitions between states are rare." The shared helpers (snapshots, dynamics,
 * quench, velocity create) are in accelerated.ts. Native LAMMPS runs one partition per process, and the partitions
 * step together through MPI collectives; here the replicas are stepped one after another in the same thread. That
 * gives the same numbers, because every replica's own computations are sequential and the collective decisions
 * (check_event, the choice of the event replica) are made from the same generators on every replica (random_select
 * and random_clock are the same stream on every partition, random_dephase is RanMars(seed + partition)).
 *
 * Output as prd.rst gives it: dumps are written only at events, by replica 0, with the event's quenched coordinates ("the
 * requested dump frequency in the dump command is ignored"; "A dump snapshot corresponding to the initial minimum state
 * used for event detection is written to the dump file at the beginning of each PRD run"), and a restart file of
 * replica 0 every N uncorrelated events ("a frequency of 10 means write a restart file every 10th uncorrelated event").
 *
 * Refused with a StyleError: fix deform (a changing box), vel local (native's loop local draws the velocities in its own
 * local atom order), and a compute other than event/displace. The engine has no time-dependent fix or region to refuse
 * (native refuses them: "Cannot use PRD with a time-dependent fix defined").
 */
import type { Ctx } from '../../engine/commands/args';
import { int, num } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import { RanMars } from '../random_mars';
import { RanPark } from '../random_park';
import { snapOf, restoreSnap, copyPositions, dynamics, setupOnly, quench, createVelocity, type Snap } from './accelerated';

/** The largest small integer of LAMMPS (MAXSMALLINT, src/lmptype.h), the range of the dephasing seeds. */
const MAXSMALLINT = 0x7fffffff;

/** The event compute the command reads (compute event/displace), as its reference and threshold are set. */
interface EventCompute {
  id: string;
  style: string;
  storeReference(): void;
  displacement(): { count: number };
}

/** One replica of the command: its generators and the state of its own fix event snapshots. */
interface World {
  sys: System;
  index: number;
  /** random_dephase: RanMars(seed + iworld), unique per replica */
  rdephase: RanMars;
  /** temp_dephase (per replica, updated after each dephasing iteration when the temperature is not given) */
  tempDephase: number;
  /** the replica's temperature compute (thermo_temp: group all, the prd_temp of the command) */
  temperature: Compute;
  compute: EventCompute;
  /** FixEventPRD's state before the last quench (store_state_quench) and before the dephasing run (store_state_dephase) */
  hot: Snap | null;
  orig: Snap | null;
}

/** The fix event of the command (FixEventPRD): the last event's clock and timestep, and the event's replica. */
interface PrdEvent {
  number: number;
  timestep: number;
  clock: number;
  correlated: number;
  ncoincident: number;
  replica: number;
}

interface Options {
  etol: number; ftol: number; maxiter: number; maxeval: number;
  tempFlag: boolean; tempDephase: number;
  loop: 'all' | 'geom'; gaussian: boolean;
  stepmode: 'steps' | 'clock';
}

/** The options of prd after its command words (prd.cpp PRD::options; the defaults are the same). */
const parseOptions = (w: readonly string[]): Options => {
  const o: Options = {
    etol: 0.1, ftol: 0.1, maxiter: 40, maxeval: 50, tempFlag: false, tempDephase: 0,
    loop: 'geom', gaussian: true, stepmode: 'steps',
  };
  for (let k = 0; k < w.length;) {
    switch (w[k]) {
      case 'min': {
        if (k + 5 > w.length) throw new StyleError('Illegal prd command');
        o.etol = num(w[k + 1], 'etol');
        o.ftol = num(w[k + 2], 'ftol');
        o.maxiter = int(w[k + 3], 'maxiter');
        o.maxeval = int(w[k + 4], 'maxeval');
        if (o.maxiter < 0) throw new StyleError('Illegal prd command');
        k += 5;
        break;
      }
      case 'temp': {
        if (k + 2 > w.length) throw new StyleError('Illegal prd command');
        o.tempFlag = true;
        o.tempDephase = num(w[k + 1], 'temp');
        if (o.tempDephase <= 0) throw new StyleError('Illegal prd command');
        k += 2;
        break;
      }
      case 'vel': {
        if (k + 3 > w.length) throw new StyleError('Illegal prd command');
        if (w[k + 1] === 'local') throw new StyleError('prd vel local is not supported by the browser engine yet (loop all and loop geom are)');
        if (!['all', 'geom'].includes(w[k + 1])) throw new StyleError('Illegal prd command');
        if (w[k + 2] !== 'uniform' && w[k + 2] !== 'gaussian') throw new StyleError('Illegal prd command');
        o.loop = w[k + 1] as Options['loop'];
        o.gaussian = w[k + 2] === 'gaussian';
        k += 3;
        break;
      }
      case 'time': {
        if (k + 2 > w.length) throw new StyleError('Illegal prd command');
        if (w[k + 1] === 'steps') o.stepmode = 'steps';
        else if (w[k + 1] === 'clock') o.stepmode = 'clock';
        else throw new StyleError('Illegal prd command');
        k += 2;
        break;
      }
      default: throw new StyleError(`Illegal prd command: unknown keyword '${w[k]}' (min, temp, vel or time)`);
    }
  }
  return o;
};

/** The run of the command: the replicas (partitions) step together as native LAMMPS does (prd.cpp PRD::command). */
export const runPrd = async (ctx: Ctx, a: string[]): Promise<void> => {
  const { session } = ctx;
  const run = ctx.run;
  const reps = [...session.replicaSystems];
  if (a.length < 7) throw new StyleError('Illegal prd command');
  const nsteps = int(a[0], 'N');
  const tEvent = int(a[1], 't_event');
  const nDephase = int(a[2], 'n_dephase');
  const tDephase = int(a[3], 't_dephase');
  const tCorr = int(a[4], 't_corr');
  const compId = a[5];
  const seed = int(a[6], 'seed');
  const o = parseOptions(a.slice(7));

  if (tEvent <= 0) throw new StyleError('Invalid t_event in prd command');
  if (nsteps % tEvent) throw new StyleError('PRD nsteps must be multiple of t_event');
  if (tCorr % tEvent) throw new StyleError('PRD t_corr must be multiple of t_event');
  if (seed <= 0) throw new StyleError('Invalid seed for Park random # generator');
  if (nDephase < 0 || tDephase < 0 || tCorr < 0) throw new StyleError('Illegal prd command');
  for (const sys of reps) {
    if (!sys.hasBox) throw new StyleError('PRD command before simulation box is defined');
    if (sys.fixes.some((f) => f.style === 'deform')) throw new StyleError('Cannot use PRD with a changing box (fix deform)');
  }
  const N = reps.length;
  const cancelled = () => session.isCancelled;
  const worlds: World[] = reps.map((sys, w) => {
    const c = sys.computes.find((x) => x.id === compId) as unknown as EventCompute | undefined;
    if (!c) throw new StyleError(`Could not find compute ID for PRD (${compId})`);
    if (c.style !== 'event/displace') throw new StyleError(`Compute ${compId} (${c.style}) is not an event computer for PRD; the browser engine supports compute event/displace`);
    ctx.run.ensureThermoComputes(sys);
    return {
      sys, index: w, rdephase: new RanMars(seed + w), tempDephase: o.tempFlag ? o.tempDephase : 0,
      temperature: sys.compute('thermo_temp'), compute: c, hot: null, orig: null,
    };
  });
  // the neighbour settings of PRD (every 1, delay 0, check yes) hold for the whole command (prd.cpp)
  const saved = reps.map((sys) => ({ every: sys.nb.every, delay: sys.nb.delay, check: sys.nb.check }));
  if (reps.some((sys) => sys.nb.every !== 1 || sys.nb.delay !== 0 || !sys.nb.check)) reps[0].warn('Resetting reneighboring criteria during PRD');
  for (const sys of reps) { sys.nb.every = 1; sys.nb.delay = 0; sys.nb.check = true; }

  // the random streams: the same select and clock streams on every replica (one copy suffices here)
  const select = new RanPark(seed);
  const clock = new RanPark(seed + 1000);
  const ev: PrdEvent = { number: 0, timestep: reps[0].state.step, clock: 0, correlated: 0, ncoincident: 0, replica: 0 };
  let ncoincident = 0;
  const beginStep = reps[0].state.step;
  const endStep = beginStep + nsteps;
  const qopts = { etol: o.etol, ftol: o.ftol, maxiter: o.maxiter, maxeval: o.maxeval, firststep: beginStep };
  const t0 = performance.now();
  const cpu = () => ((performance.now() - t0) / 1000).toFixed(3);
  const emitLog = (text: string) => session.emit({ kind: 'log', text });
  const dynOpts = { cancelled, laststep: endStep };
  const stepOf = (w: World) => w.sys.state.step;

  const temperatureOf = (w: World): number => {
    w.temperature.init();
    w.sys.refreshComputes();
    return w.temperature.scalarValue();
  };
  /** compute event/displace of replica w (1 when an atom has moved more than the threshold from the reference). */
  const eventIn = (w: World): number => (w.compute.displacement().count > 0 ? 1 : 0);

  /** FixEventPRD::store_event_prd: the quenched state (the same on every replica) is the new reference. */
  const storeEventPrd = (timestep: number, dclock: number): void => {
    for (const w of worlds) w.compute.storeReference();
    ev.timestep = timestep;
    ev.clock += dclock;
    ev.number++;
  };

  /** PRD::log_event: universe proc 0 (replica 0) prints the event line. */
  const logEvent = (): void => {
    emitLog(`${ev.timestep} ${cpu()} ${ev.clock} ${ev.number} ${ev.correlated} ${ev.ncoincident} ${ev.replica}`);
  };

  /** PRD::replicate: the positions and image flags of replica ireplica go to every replica. */
  const replicate = (ireplica: number): void => {
    const src = worlds[ireplica].sys;
    for (const w of worlds) copyPositions(w.sys, src);
  };

  /**
   * The dump snapshot of share_event: replica 0 (which holds the event's quenched coordinates) is reneighbored and its
   * forces computed (integrate->setup_minimal(1)), then every dump writes the current timestep, whatever its interval.
   */
  const writeEventDumps = async (): Promise<void> => {
    const sys = reps[0];
    if (!sys.dumps.length) return;
    await run.runVerlet(sys, 0, { cancelled, afterSetup: () => undefined, afterStep: () => undefined, yieldMs: 30 });
    for (const d of sys.dumps) d.write();
  };

  /** PRD::share_event: the quenched state of replica ireplica becomes the event, then the hot state is restored and shared. */
  const shareEvent = async (ireplica: number, flag: number, decrement: number): Promise<void> => {
    replicate(ireplica);
    const corrAdjust = ev.number < 1 || flag === 2 ? 0 : tCorr;
    let delta = stepOf(worlds[ireplica]) - ev.timestep - corrAdjust;
    if (flag !== 2) delta *= N;
    if (delta > 0 && flag !== 2) delta -= decrement;
    delta += corrAdjust;
    if (flag === 0 && ev.number !== 0) storeEventPrd(ev.timestep, 0);
    else {
      storeEventPrd(stepOf(worlds[ireplica]), delta);
      ev.replica = ireplica;
      ev.correlated = flag === 2 ? 1 : 0;
      ev.ncoincident = ncoincident;
    }
    if (flag === 0) ev.number--;
    await writeEventDumps();
    for (const w of worlds) if (w.hot) restoreSnap(w.sys, w.hot);
    replicate(ireplica);
  };

  /**
   * PRD::check_event: the replicas whose event compute flags an event; with several, one is chosen at random with
   * random_select (the iwhich-th event replica). Returns the replica or -1 (replicaNum >= 0: only that replica counts).
   */
  const checkEvent = (replicaNum: number): number => {
    const flags = worlds.map((w) => (replicaNum >= 0 && replicaNum !== w.index ? 0 : eventIn(w)));
    const universe = flags.reduce((x, y) => x + y, 0);
    ncoincident = universe;
    if (!universe) return -1;
    if (universe > 1) {
      const iwhich = Math.trunc(universe * select.uniform()) + 1;
      let scan = 0;
      for (let k = 0; k < N; k++) {
        if (flags[k]) scan++;
        if (scan !== iwhich) flags[k] = 0;
      }
    }
    return flags.indexOf(1);
  };

  /** PRD::dephase: n_dephase short dynamics runs from new random velocities, each until an event-free run occurs. */
  const dephase = async (): Promise<void> => {
    for (const w of worlds) {
      const hold = stepOf(w);
      for (let i = 0; i < nDephase; i++) {
        w.orig = snapOf(w.sys);
        let done = false;
        while (!done) {
          let sd = Math.trunc(w.rdephase.uniform() * MAXSMALLINT);
          if (sd === 0) sd = 1;
          createVelocity(w.sys, w.tempDephase, sd, o.loop === 'all' ? 'all' : 'geom', o.gaussian, () => temperatureOf(w));
          await dynamics(run, w.sys, tDephase, dynOpts);
          w.hot = snapOf(w.sys);
          await quench(run, w.sys, qopts, cancelled);
          if (eventIn(w) > 0) {
            restoreSnap(w.sys, w.orig);
            w.sys.state.step -= tDephase;
            if (w.index === 0) logEvent();
          } else {
            restoreSnap(w.sys, w.hot);
            done = true;
          }
          if (!o.tempFlag) w.tempDephase = temperatureOf(w);
        }
      }
      // the timestep is reset as if the dephasing did not occur
      w.sys.state.step = hold;
    }
  };

  /** The setup of the full init after an event (integrate->setup(1)) on every replica, and the temperature of the event replica. */
  const setupAll = async (): Promise<void> => {
    for (const w of worlds) await setupOnly(run, w.sys, dynOpts);
  };

  try {
    emitLog('Setting up PRD ...');
    emitLog('Step CPU Clock Event Correlated Coincident Replica');
    // the start state: each replica is quenched, replica 0's quenched state is the first event
    for (const w of worlds) {
      w.hot = snapOf(w.sys);
      await quench(run, w.sys, qopts, cancelled);
    }
    ncoincident = 0;
    await shareEvent(0, 0, 0);
    logEvent();
    await setupAll();
    if (!o.tempFlag) {
      const t = temperatureOf(worlds[0]);
      for (const w of worlds) w.tempDephase = t;
    }

    // main loop: look for events until out of time (steps, or clock when time clock is given)
    // the loop clock counts every replica's steps (clock += t_event * nworlds); it is the istep of time clock
    let loopClock = 0;
    const istepOf = () => (o.stepmode === 'steps' ? stepOf(worlds[0]) - beginStep : loopClock);
    let istep = 0;
    while (istep < nsteps) {
      await dephase();
      istep = istepOf();
      let ireplica = -1;
      while (istep < nsteps) {
        for (const w of worlds) {
          await dynamics(run, w.sys, tEvent, dynOpts);
          w.hot = snapOf(w.sys);
          await quench(run, w.sys, qopts, cancelled);
        }
        loopClock += tEvent * N;
        ireplica = checkEvent(-1);
        if (ireplica >= 0) break;
        for (const w of worlds) if (w.hot) restoreSnap(w.sys, w.hot);
        istep = istepOf();
      }
      if (ireplica < 0) break;

      // decrement the clock by the random time at which one or more events occurred (random_clock). The count is
      // fix_event->ncoincident, the previous event's (native reads it before this event is stored)
      let fracT = tEvent;
      for (let i = 0; i < ev.ncoincident; i++) {
        const fracRand = Math.trunc(clock.uniform() * tEvent);
        fracT = Math.min(fracT, fracRand);
      }
      const decrement = (tEvent - fracT) * N;
      loopClock -= decrement;
      await shareEvent(ireplica, 1, decrement);
      logEvent();
      // a restart file every N uncorrelated events (restart N: output->restart_every_single or _double), replica 0 only;
      // it is written after the correlated-event loop, unless that loop reaches the end of the run
      let restartFlag = reps[0].restartOut.some((r) => r.everyVar === null && r.every > 0 && ev.number % r.every === 0);

      // correlated event loop: other replicas may be dephasing during this time in native LAMMPS
      let corrEnd = stepOf(worlds[0]) + tCorr;
      while (stepOf(worlds[0]) < corrEnd) {
        if (stepOf(worlds[0]) === endStep) { restartFlag = false; break; }
        for (const w of worlds) {
          await dynamics(run, w.sys, tEvent, dynOpts);
          w.hot = snapOf(w.sys);
          await quench(run, w.sys, qopts, cancelled);
        }
        loopClock += tEvent;
        const corr = checkEvent(ireplica);
        if (corr >= 0) {
          await shareEvent(ireplica, 2, 0);
          logEvent();
          corrEnd = stepOf(worlds[0]) + tCorr;
        } else for (const w of worlds) if (w.hot) restoreSnap(w.sys, w.hot);
      }

      // full init/setup since all replicas start again after the event
      await setupAll();
      if (tCorr > 0) replicate(ireplica);
      if (!o.tempFlag) {
        const t = temperatureOf(worlds[ireplica]);
        for (const w of worlds) w.tempDephase = t;
      }
      // the restart file of the hot coordinates (output->write_restart: every restart file of the restart command)
      if (restartFlag) run.writeRestarts(reps[0], stepOf(worlds[0]), true);
      istep = istepOf();
    }
    const steps = stepOf(worlds[0]) - beginStep;
    const seconds = (performance.now() - t0) / 1000;
    emitLog(`Loop time of ${seconds.toFixed(3)} s for ${steps} steps with ${reps[0].state.n} atoms (CPU · fp64, ${N} partitions)`);
    emitLog('\nPRD done');
    session.emit({ kind: 'done', steps, seconds, backend: session.backendLabel });
  } finally {
    reps.forEach((sys, k) => { sys.nb.every = saved[k].every; sys.nb.delay = saved[k].delay; sys.nb.check = saved[k].check; });
    for (const sys of reps) sys.run.inRun = false;
  }
};
