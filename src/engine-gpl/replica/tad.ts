/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/tad.cpp and tad.h (TAD::command, options, dynamics, quench, check_event, log_event, perform_neb,
 * check_confidence, store_state, revert_state, initialize_event_list, add_event, compute_tlo, perform_event), and
 * src/REPLICA/fix_event_tad.cpp (FixEventTAD::store_event_tad) at commit b332f28. Copyright Sandia Corporation (GNU
 * General Public License). Contributing author: Mike Brown (SNL) (tad), Aidan Thompson (SNL) (fix event). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Temperature accelerated dynamics (docs.lammps.org/tad.html): the dynamics runs at the high temperature, each
 * detected event is quenched, its barrier is found by a NEB calculation between the event and the state before it
 * (the replicas are the partitions), and the earliest event at the low temperature is taken once the confidence
 * criterion is met. In native LAMMPS only the first partition runs the dynamics; the others only take part in the
 * NEB calculations, so here only replica 0 is stepped and every replica joins the NEB (neb.ts runNeb). The NEB output
 * goes to the neb_log file (native: universe->ulogfile of the NEB), not to the screen.
 *
 * As tad.cpp does, the quench minimizer is made anew when the command starts and after each NEB, and the NEB minimizer
 * (neb_style: quickmin or fire) for each NEB, all with the default min_modify settings (Update::create_minimize).
 *
 * Output as tad.rst gives it: dumps are written only by replica 0, when an event is taken and at the start, with the
 * quenched coordinates ("the requested dump frequency in the dump command is ignored"; "A dump snapshot corresponding to
 * the initial minimum state used for event detection is written to the dump file at the beginning of each TAD run"),
 * and a restart file every N events ("The restart file is generated after an event is executed").
 *
 * Refused with a StyleError: atom_modify sort, fix deform (a changing box), and a compute other than event/displace.
 */
import type { Ctx } from '../../engine/commands/args';
import { int, num } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { NebHost } from '../../engine/commands/neb';
import type { Fix } from '../../engine/fix/fix';
import { snapOf, restoreSnap, dynamics, setupOnly, quench, type Snap } from './accelerated';
import { log as libmLog } from '../libm/log';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

interface EventCompute {
  id: string;
  style: string;
  storeReference(): void;
  displacement(): { count: number };
}

/** A possible event (FixEventTAD: its quenched state, its hot state and the times): fix_event_list entries. */
interface TadEvent {
  timestep: number;
  tlo: number;
  ebarrier: number;
  /** unwrapped quenched positions (FixEvent::store_event) */
  xevent: Float64Array;
  /** the hot state the event was found from (store_state_quench of the event) */
  hot: Snap;
}

interface Options {
  etol: number; ftol: number; maxiter: number; maxeval: number;
  etolNeb: number; ftolNeb: number; n1: number; n2: number; nevery: number;
  nebStyle: string; dtNeb: number | null; nebLog: string | null;
}

const parseOptions = (w: readonly string[]): Options => {
  const o: Options = {
    etol: 0.1, ftol: 0.1, maxiter: 40, maxeval: 50,
    etolNeb: 0.01, ftolNeb: 0.01, n1: 100, n2: 100, nevery: 10,
    nebStyle: 'quickmin', dtNeb: null, nebLog: null,
  };
  for (let k = 0; k < w.length;) {
    switch (w[k]) {
      case 'min': {
        if (k + 5 > w.length) throw new StyleError('Illegal tad command');
        o.etol = num(w[k + 1], 'etol');
        o.ftol = num(w[k + 2], 'ftol');
        o.maxiter = int(w[k + 3], 'maxiter');
        o.maxeval = int(w[k + 4], 'maxeval');
        if (o.maxiter < 0 || o.maxeval < 0 || o.etol < 0 || o.ftol < 0) throw new StyleError('Illegal tad command');
        k += 5;
        break;
      }
      case 'neb': {
        if (k + 6 > w.length) throw new StyleError('Illegal tad command');
        o.etolNeb = num(w[k + 1], 'etol_neb');
        o.ftolNeb = num(w[k + 2], 'ftol_neb');
        o.n1 = int(w[k + 3], 'n1steps_neb');
        o.n2 = int(w[k + 4], 'n2steps_neb');
        o.nevery = int(w[k + 5], 'nevery_neb');
        if (o.etolNeb < 0 || o.ftolNeb < 0 || o.n1 < 0 || o.n2 < 0 || o.nevery < 0) throw new StyleError('Illegal tad command');
        k += 6;
        break;
      }
      case 'neb_style': {
        if (k + 2 > w.length) throw new StyleError('Illegal tad command');
        o.nebStyle = w[k + 1];
        k += 2;
        break;
      }
      case 'neb_step': {
        if (k + 2 > w.length) throw new StyleError('Illegal tad command');
        o.dtNeb = num(w[k + 1], 'neb_step');
        if (o.dtNeb <= 0) throw new StyleError('Illegal tad command');
        k += 2;
        break;
      }
      case 'neb_log': {
        if (k + 2 > w.length) throw new StyleError('Illegal tad command');
        o.nebLog = w[k + 1] === 'none' ? null : w[k + 1];
        k += 2;
        break;
      }
      default: throw new StyleError(`Illegal tad command: unknown keyword '${w[k]}' (min, neb, neb_style, neb_step or neb_log)`);
    }
  }
  return o;
};

/** neb_log: the NEB tables of the run, written to the log file (none: not written). */
const writeNebLog = (sys: System, name: string | null, lines: string[]): void => {
  if (name === null) return;
  sys.io.writeFile?.(name, lines.join('\n') + '\n', false);
};

/** The minimum image of one separation (Domain::minimum_image: one period is added or taken off). */
const minimumImage = (d: number, len: number, periodic: boolean): number => {
  if (!periodic || !(Math.abs(d) > len / 2)) return d;
  return d < 0 ? d + len : d - len;
};

/** FixEvent::restore_event: positions from the unwrapped event coordinates, remapped into the box. */
const restoreEvent = (sys: System, xevent: Float64Array): void => {
  const s = sys.state;
  for (let i = 0; i < s.n; i++) {
    s.x[3 * i] = xevent[3 * i]; s.x[3 * i + 1] = xevent[3 * i + 1]; s.x[3 * i + 2] = xevent[3 * i + 2];
    s.image[3 * i] = 0; s.image[3 * i + 1] = 0; s.image[3 * i + 2] = 0;
    sys.geom.remap(s.x, s.image, i);
  }
  sys.bump();
  sys.forcesCurrent();
};

/** FixEvent::store_event: the unwrapped positions of the current state. */
const unwrappedOf = (sys: System): Float64Array => {
  const s = sys.state;
  const out = new Float64Array(3 * s.n);
  const u = [0, 0, 0];
  for (let i = 0; i < s.n; i++) {
    sys.geom.unwrap(s.x, s.image, i, u);
    out[3 * i] = u[0]; out[3 * i + 1] = u[1]; out[3 * i + 2] = u[2];
  }
  return out;
};

/** The wrapped positions of the current state (the buffers perform_neb copies, one value per atom and coordinate). */
const wrappedOf = (sys: System): Float64Array => Float64Array.from(sys.state.x.subarray(0, 3 * sys.state.n));

export const runTad = async (ctx: Ctx, a: string[]): Promise<void> => {
  const { session } = ctx;
  const run = ctx.run;
  const reps = [...session.replicaSystems];
  if (reps.length < 2) throw new StyleError('Cannot use TAD with a single replica for NEB');
  if (a.length < 7) throw new StyleError('Illegal tad command');
  const nsteps = int(a[0], 'nsteps');
  const tEvent = int(a[1], 't_event');
  const templo = num(a[2], 'templo');
  const temphi = num(a[3], 'temphi');
  const deltaConf = num(a[4], 'delta_conf');
  const tmax = num(a[5], 'tmax');
  const compId = a[6];
  const o = parseOptions(a.slice(7));

  if (tEvent <= 0) throw new StyleError('Invalid t_event in tad command');
  if (nsteps % tEvent) throw new StyleError('TAD nsteps must be multiple of t_event');
  if (deltaConf <= 0 || deltaConf >= 1) throw new StyleError('Invalid delta_conf in tad command');
  if (tmax <= 0) throw new StyleError('Invalid tmax in tad command');
  const sys0 = reps[0];
  for (const sys of reps) {
    if (!sys.hasBox) throw new StyleError('TAD command before simulation box is defined');
    if (sys.fixes.some((f) => f.style === 'deform')) throw new StyleError('Cannot use TAD with a changing box (fix deform)');
    if (sys.sortEvery > 0) throw new StyleError('Cannot use TAD with atom_modify sort enabled for NEB');
  }
  const comp = sys0.computes.find((x) => x.id === compId) as unknown as EventCompute | undefined;
  if (!comp) throw new StyleError(`Could not find compute ID for TAD (${compId})`);
  if (comp.style !== 'event/displace') throw new StyleError(`Compute ${compId} (${comp.style}) is not an event computer for TAD; the browser engine supports compute event/displace`);

  const N = reps.length;
  const cancelled = () => session.isCancelled;
  const s0 = sys0.state;
  const dt = s0.dt;
  const boltz = s0.units.boltz;
  // deltconf = (ln(1/delta))/freq_min (timestep units); the NEB timestep defaults to the run's
  const deltconf = -libmLog(deltaConf) * tmax / dt;
  const deltaBeta = (1.0 / templo - 1.0 / temphi) / boltz;
  const ratioBeta = templo / temphi;
  const dtNeb = o.dtNeb ?? dt;
  const beginStep = s0.step;
  const endStep = beginStep + nsteps;
  const qopts = { etol: o.etol, ftol: o.ftol, maxiter: o.maxiter, maxeval: o.maxeval, firststep: beginStep };
  const t0 = performance.now();
  const cpu = () => ((performance.now() - t0) / 1000).toFixed(3);
  const emitLog = (text: string) => session.emit({ kind: 'log', text });
  const nebLines: string[] = [];

  // tad.cpp: "set minimize style for quench" (a new minimizer of the session's min_style on every partition)
  for (const sys of reps) run.createMinimize(sys, sys.minStyle);
  const saved = reps.map((sys) => ({ every: sys.nb.every, delay: sys.nb.delay, check: sys.nb.check }));
  if (reps.some((sys) => sys.nb.every !== 1 || sys.nb.delay !== 0 || !sys.nb.check)) sys0.warn('Resetting reneighboring criteria during TAD');
  for (const sys of reps) { sys.nb.every = 1; sys.nb.delay = 0; sys.nb.check = true; }

  // the fix event of the command (FixEventTAD tad_event): the last event and its hot state (store_state_quench)
  const main = { number: 0, timestep: beginStep, tlo: 0, ebarrier: 0, xevent: unwrappedOf(sys0), hot: null as Snap | null };
  let events: TadEvent[] = [];
  let deltstop = 0;
  let deltfirst = 0;
  let eventFirst = 0;
  /**
   * revert_state: the state at the start of the last dynamics segment without an event (store_state). Native keeps the
   * store unset until the first such segment (fix_revert is allocated, not filled), so the state at the start is taken
   * here as the store.
   */
  let astore: Snap | null = null;

  /** TAD::log_event: the event row of the screen table (Step CPU N M Status Barrier Margin t_lo delt_lo). */
  const logEvent = (timestep: number, ievent: number, status: string, ebarrier: number, tfrac: number, tlo: number, delt: number): void => {
    emitLog(`${timestep} ${cpu()} ${main.number} ${ievent} ${status} ${ebarrier.toFixed(3)} ${tfrac.toFixed(3)} ${tlo.toFixed(3)} ${delt.toFixed(3)}`);
  };

  /**
   * The dump snapshot of TAD::log_event: replica 0 (holding the quenched coordinates) is reneighbored and its forces
   * computed (integrate->setup_minimal(1)), then every dump writes the current timestep, whatever its interval.
   */
  const eventDump = async (): Promise<void> => {
    if (!sys0.dumps.length) return;
    await run.runVerlet(sys0, 0, { cancelled, afterSetup: () => undefined, afterStep: () => undefined, yieldMs: 30 });
    for (const d of sys0.dumps) d.write();
  };

  const checkEvent = (): boolean => comp.displacement().count > 0;

  /** TAD::check_confidence: the stopping time deltstop, and whether the time since the event exceeds it. */
  const checkConfidence = (): boolean => {
    deltstop = deltconf * libmPow(deltfirst / deltconf, ratioBeta);
    return deltstop < s0.step - main.timestep;
  };

  /** TAD::compute_tlo: the cold time of event ievent and the first event (the smallest cold time). */
  const computeTlo = (ievent: number): void => {
    const ev = events[ievent];
    const delthi = ev.timestep - main.timestep;
    const deltlo = delthi * libmExp(ev.ebarrier * deltaBeta);
    ev.tlo = main.tlo + deltlo;
    let status = 'D ';
    if (ievent === 0) {
      deltfirst = deltlo;
      eventFirst = ievent;
      status = 'DF';
    } else if (deltlo < deltfirst) {
      deltfirst = deltlo;
      eventFirst = ievent;
      status = 'DF';
    }
    const tfrac = ievent > 0 ? delthi / deltstop : 0;
    logEvent(ev.timestep, ievent, status, ev.ebarrier, tfrac, main.tlo, deltlo);
  };

  /** TAD::add_event: a new possible event from the quenched state; the hot state of the main fix is restored first. */
  const addEvent = (): void => {
    const xevent = unwrappedOf(sys0);
    if (!main.hot) throw new StyleError('tad: no hot state');
    const ev: TadEvent = { timestep: s0.step, tlo: 0, ebarrier: 0, xevent, hot: null as unknown as Snap };
    restoreSnap(sys0, main.hot);
    ev.hot = snapOf(sys0);
    events.push(ev);
  };

  /** TAD::perform_neb: the NEB between the event's quenched state and the main event's, on every replica. */
  const performNeb = async (ievent: number): Promise<void> => {
    const ev = events[ievent];
    // the timestep of every replica is the TAD clock (tad.cpp: the other universe worlds advance ntimestep by t_event
    // with the event flag broadcast, and MPI_Bcast of the timestep follows each event)
    for (const sys of reps) sys.state.step = s0.step;
    restoreEvent(sys0, ev.xevent);
    const bufFinal = wrappedOf(sys0);
    restoreEvent(sys0, main.xevent);
    const bufInit = wrappedOf(sys0);
    // NEB constructor: replica r starts at buf_init + (r / (N - 1)) * (buf_final - buf_init), minimum image
    for (let r = 0; r < N; r++) {
      const sys = reps[r];
      const st = sys.state;
      const fraction = r / (N - 1.0);
      for (let i = 0; i < st.n; i++) {
        for (let d = 0; d < 3; d++) {
          const len = st.box.hi[d] - st.box.lo[d];
          const delx = minimumImage(bufFinal[3 * i + d] - bufInit[3 * i + d], len, st.box.periodic[d]);
          st.x[3 * i + d] = bufInit[3 * i + d] + fraction * delx;
        }
      }
      sys.pbc();
    }
    // fix neb 1.0 in every replica (tad.cpp: add_fix "neb all neb 1.0"), a new minimizer of neb_style for the NEB, and
    // its timestep
    const nebFix: { sys: System; fix: Fix }[] = [];
    const holds = reps.map((sys) => ({ minStyle: sys.minStyle, dt: sys.state.dt, step: sys.state.step }));
    for (const sys of reps) {
      const f = run.makeNebFix(sys, 'neb', 'all', ['1.0']);
      sys.fixes.push(f as unknown as (typeof sys.fixes)[number]);
      nebFix.push({ sys, fix: f });
    }
    const lines: string[] = [];
    const host: NebHost = {
      replicaSystems: reps,
      backendLabel: session.backendLabel,
      emit: (e) => {
        // the NEB table goes to the neb log (native: universe->ulogfile of the NEB), not to the screen
        if (e.kind === 'thermo') lines.push(Object.entries(e.row).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toPrecision(8) : v}`).join(' '));
        else if (e.kind === 'log') lines.push(e.text);
      },
      argsOf: () => undefined,
      // each partition's minimization prints its own thermo rows (the header and the setup row, then the interval)
      replicaThermo: (rs, setup) => {
        if (setup) rs.io.emit({ kind: 'thermo-header', keywords: [...rs.thermo.keywords], labels: rs.thermo.labels(), units: rs.units.style });
        run.emitThermoRow(rs);
      },
    };
    let ebf = 0;
    try {
      for (const sys of reps) {
        run.createMinimize(sys, o.nebStyle);
        sys.state.dt = dtNeb;
      }
      ebf = await run.runNeb(host, [String(o.etolNeb), String(o.ftolNeb), String(o.n1), String(o.n2), String(o.nevery), 'none']);
    } finally {
      // neb.cpp sets update->max_eval to N1 and then N2 steps (lines 240 and 337 at b332f28), and tad.cpp
      // perform_neb does not restore it: every later quench stops after N2 evaluations (min_cg.cpp MAXEVAL)
      qopts.maxeval = o.n2;
      nebFix.forEach(({ sys, fix }) => {
        const k = sys.fixes.indexOf(fix as unknown as (typeof sys.fixes)[number]);
        if (k >= 0) sys.fixes.splice(k, 1);
      });
      reps.forEach((sys, r) => {
        // tad.cpp: "switch minimize style back for quench" (a new minimizer, default min_modify settings)
        run.createMinimize(sys, holds[r].minStyle);
        sys.state.dt = holds[r].dt;
        sys.state.step = holds[r].step;
      });
      sys0.refreshComputes();
    }
    nebLines.push(...lines);
    ev.ebarrier = ebf;
  };

  /** TAD::revert_state: world 0 goes back to the stored state with the velocities reversed. */
  const revertState = (): void => {
    if (!astore) throw new StyleError('tad: no stored state');
    const s = sys0.state;
    s.x.set(astore.x);
    s.image.set(astore.image);
    for (let k = 0; k < 3 * s.n; k++) s.v[k] = -astore.v[k];
    sys0.bump();
    sys0.forcesCurrent();
  };

  /** TAD::perform_event: the event is taken: the timestep and the event state become the main event's. */
  const performEvent = async (ievent: number): Promise<void> => {
    const ev = events[ievent];
    s0.step = ev.timestep;
    main.tlo = ev.tlo;
    main.ebarrier = ev.ebarrier;
    main.number++;
    main.timestep = s0.step;
    restoreEvent(sys0, ev.xevent);
    main.xevent = unwrappedOf(sys0);
    comp.storeReference();
    logEvent(main.timestep, eventFirst, 'E ', main.ebarrier, 0, main.tlo, deltfirst);
    await eventDump();
    restoreSnap(sys0, ev.hot);
    main.hot = snapOf(sys0);
  };

  try {
    emitLog('Setting up TAD ...');
    emitLog('Step CPU N M Status Barrier Margin t_lo delt_lo');
    // the first quench on every replica (tad.cpp: store_state_quench and quench are not limited to universe world 0);
    // the reference event is replica 0's quenched state, then every replica is restored and set up
    const hotAll = reps.map((sys) => snapOf(sys));
    main.hot = hotAll[0];
    for (const sys of reps) await quench(run, sys, qopts, cancelled);
    comp.storeReference();
    main.xevent = unwrappedOf(sys0);
    logEvent(main.timestep, 0, 'E ', 0, 0, 0, 0);
    await eventDump();
    reps.forEach((sys, k) => restoreSnap(sys, hotAll[k]));
    // the full setup (integrate->setup(1))
    for (const sys of reps) await setupOnly(run, sys, { cancelled });
    astore = snapOf(sys0);

    while (s0.step < endStep) {
      events = [];
      let confident = false;
      while (s0.step < endStep) {
        let eventFlag = false;
        while (s0.step < endStep) {
          await dynamics(run, sys0, tEvent, { cancelled, laststep: endStep });
          main.hot = snapOf(sys0);
          await quench(run, sys0, qopts, cancelled);
          eventFlag = checkEvent();
          if (eventFlag) break;
          restoreSnap(sys0, main.hot);
          astore = snapOf(sys0);
        }
        if (!eventFlag) break;
        addEvent();
        await performNeb(events.length - 1);
        computeTlo(events.length - 1);
        confident = checkConfidence();
        if (confident) break;
        revertState();
      }
      if (!confident) break;
      await performEvent(eventFirst);
      // a restart file every N events (restart N: output->restart_every_single or _double), replica 0 only, of the hot
      // coordinates after the full setup (output->write_restart: every restart file of the restart command)
      const restartFlag = sys0.restartOut.some((r) => r.everyVar === null && r.every > 0 && main.number % r.every === 0);
      await setupOnly(run, sys0, { cancelled });
      if (restartFlag) run.writeRestarts(sys0, s0.step, true);
    }
    const seconds = (performance.now() - t0) / 1000;
    emitLog(`Loop time of ${seconds.toFixed(3)} s for ${s0.step - beginStep} steps with ${s0.n} atoms (CPU · fp64, ${N} partitions)`);
    emitLog('\nTAD done');
    session.emit({ kind: 'done', steps: s0.step - beginStep, seconds, backend: session.backendLabel });
    if (nebLines.length) writeNebLog(sys0, o.nebLog, nebLines);
  } finally {
    reps.forEach((sys, k) => { sys.nb.every = saved[k].every; sys.nb.delay = saved[k].delay; sys.nb.check = saved[k].check; });
  }
};
