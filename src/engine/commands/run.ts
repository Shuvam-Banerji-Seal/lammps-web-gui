import type { Ctx, Handler } from './args';
import { int, num } from './args';
import { StyleError } from '../force/types';
import { checkFixStateNotStored } from '../fix/restartRecord';
import { FIX_STYLES, COMPUTE_STYLES, styleNames } from '../styles';
import { Dump, type DumpStyle } from '../output/dump';
import { initRun, runVerlet } from '../run/verlet';
import { runVerletCoupled } from '../run/coupled';
import { parseRunStyleRespa, runRespa } from '../run/respa';
import { RunCancelled } from '../errors';
import { DYNAMIC_GROUP_FIXES, NO_DYNAMIC_GROUP_FIXES } from '../group';
import type { System } from '../system';
import type { EngineEvent } from '../types';
import { massOf } from '../atoms';
import { minimize, MIN_STYLES } from '../run/min';
import { MINIMIZERS } from '../run/minimizers';
import { accelerator, runAccelerated } from '../run/accel';
import { restartDue, startRestarts, writeRestarts } from './restart';
import { clearHaltRequest, haltRequested, softHalted } from '../run/halt';

/*
 * fix / compute / thermo / dump / run commands.
 */

const RESERVED_COMPUTES = new Set(['thermo_temp', 'thermo_press', 'thermo_pe']);

/** The computes thermo output needs ("compute thermo_temp all temp" etc., thermo_style.html). */
export const ensureThermoComputes = (sys: System): void => {
  if (sys.computes.some((c) => c.id === 'thermo_temp')) return;
  sys.computes.push(COMPUTE_STYLES.temp(sys, 'thermo_temp', 'all', []));
  sys.computes.push(COMPUTE_STYLES.pressure(sys, 'thermo_press', 'all', ['thermo_temp']));
  sys.computes.push(COMPUTE_STYLES.pe(sys, 'thermo_pe', 'all', []));
};

/** fix ID group style args — fix.html: "If you specify a fix ID that already exists, it will replace the existing fix" when the style matches. */
const fix: Handler = ({ sys }, a) => {
  const [id, group, style] = a;
  if (!id || !group || !style) throw new StyleError('usage: fix ID group-ID style args');
  if (!/^[A-Za-z0-9_]+$/.test(id)) throw new StyleError(`fix ID '${id}' must be alphanumeric or underscore`);
  const make = FIX_STYLES[style];
  if (!make) {
    throw new StyleError(`fix style '${style}' is not supported by the browser engine; supported: ${styleNames().fix.join(', ')}`);
  }
  checkFixStateNotStored(sys, id, style);
  // the style is checked first, so a fix the engine lacks is named even before the box exists;
  // fix_property_atom.html: "This fix is one of a small number that can be defined in an input
  // script before the simulation box is created or atoms are defined." fix_cmap.html: "To function
  // as expected this fix command must be issued *before* a read_data command".
  // examples/amoeba/in.ubiquitin defines fix amoeba/pitorsion and fix amoeba/bitorsion before read_data (their sections
  // are read by that command, like fix cmap), so they are exempt too. fix rx (src/DPD-REACT/fix_rx.cpp post_constructor)
  // only adds its species vectors as fix property/atom fixes, so it may precede create_box (examples/PACKAGES/dpd-react/
  // dpdrx-shardlow does); its per-atom vectors are attached when the box exists
  if (style !== 'property/atom' && style !== 'cmap' && style !== 'amoeba/pitorsion' && style !== 'amoeba/bitorsion' && style !== 'rx') sys.state;
  const k = sys.fixes.findIndex((f) => f.id === id);
  if (k >= 0 && sys.fixes[k].style !== style) throw new StyleError(`replacing fix ${id} with a different style (${sys.fixes[k].style} -> ${style}) is not allowed; unfix it first`);
  ensureThermoComputes(sys);
  if (sys.groups.isDynamic(sys.groupBit(group))) {
    if (NO_DYNAMIC_GROUP_FIXES.has(style)) throw new StyleError(`Fix ${style} does not allow use with a dynamic group`);
    if (!DYNAMIC_GROUP_FIXES.has(style)) throw new StyleError(`fix ${style} with a dynamic group is not supported by the browser engine`);
  }
  if (k >= 0) sys.fixes[k].destroy?.();
  const f = make(sys, id, group, a.slice(3));
  if (k >= 0) sys.fixes[k] = f; else sys.fixes.push(f);
  sys.refreshComputes();
};

const unfix: Handler = ({ sys }, a) => {
  if (a.length !== 1) throw new StyleError('usage: unfix fix-ID');
  const k = sys.fixes.findIndex((f) => f.id === a[0]);
  if (k < 0) throw new StyleError(`could not find unfix ID '${a[0]}'`);
  sys.fixes[k].destroy?.();
  sys.fixes.splice(k, 1);
  sys.refreshComputes();
};

/** fix_modify fix-ID keyword value ... — fix_modify.html. */
const fixModify: Handler = ({ sys }, a) => {
  const f = sys.fix(a[0] ?? '');
  for (let k = 1; k < a.length;) {
    const used = f.modify(a[k], a.slice(k + 1));
    if (used <= 0) throw new StyleError(`fix_modify keyword '${a[k]}' is not supported by fix ${f.id} (${f.style})`);
    k += 1 + used;
  }
};

/** compute ID group style args — compute.html. */
const compute: Handler = ({ sys }, a) => {
  sys.state;
  const [id, group, style] = a;
  if (!id || !group || !style) throw new StyleError('usage: compute ID group-ID style args');
  if (!/^[A-Za-z0-9_]+$/.test(id)) throw new StyleError(`compute ID '${id}' must be alphanumeric or underscore`);
  if (sys.computes.some((c) => c.id === id)) throw new StyleError(`reuse of compute ID '${id}' (uncompute it first)`);
  const make = COMPUTE_STYLES[style];
  if (!make) {
    throw new StyleError(`compute style '${style}' is not supported by the browser engine; supported: ${styleNames().compute.join(', ')}`);
  }
  ensureThermoComputes(sys);
  if (style === 'msd' && sys.groups.isDynamic(sys.groupBit(group))) throw new StyleError('Compute msd is not compatible with dynamic groups');
  sys.computes.push(make(sys, id, group, a.slice(3)));
};

const uncompute: Handler = ({ sys }, a) => {
  if (a.length !== 1) throw new StyleError('usage: uncompute compute-ID');
  if (RESERVED_COMPUTES.has(a[0])) throw new StyleError(`compute ${a[0]} is used by thermo output and cannot be deleted`);
  const k = sys.computes.findIndex((c) => c.id === a[0]);
  if (k < 0) throw new StyleError(`could not find uncompute ID '${a[0]}'`);
  sys.computes.splice(k, 1);
};

/** compute_modify compute-ID keyword value ... — compute_modify.html. */
const computeModify: Handler = ({ sys }, a) => {
  ensureThermoComputes(sys);
  const c = sys.compute(a[0] ?? '');
  for (let k = 1; k < a.length;) {
    const used = c.modify(a[k], a.slice(k + 1));
    if (used <= 0) throw new StyleError(`compute_modify keyword '${a[k]}' is not supported by compute ${c.id} (${c.style})`);
    k += 1 + used;
  }
  sys.refreshComputes();
};

/** thermo N | thermo v_name — thermo.html. */
const thermo: Handler = ({ sys }, a) => {
  if (a.length !== 1) throw new StyleError('usage: thermo N | thermo v_name');
  if (a[0].startsWith('v_')) {
    if (!sys.vars.has(a[0].slice(2))) throw new StyleError(`thermo: variable ${a[0].slice(2)} is not defined`);
    sys.thermo.everyVar = a[0].slice(2);
    sys.thermo.every = 0;
    return;
  }
  const n = int(a[0], 'thermo interval');
  if (n < 0) throw new StyleError('thermo interval must be >= 0');
  sys.thermo.every = n;
  sys.thermo.everyVar = null;
};

const thermoStyle: Handler = ({ sys }, a) => {
  if (sys.hasBox) ensureThermoComputes(sys);
  sys.thermo.setStyle(a[0] ?? '', a.slice(1));
};

const thermoModify: Handler = ({ sys }, a) => {
  if (sys.hasBox) ensureThermoComputes(sys);
  sys.thermo.modify(a);
};

/**
 * run_style style args — docs.lammps.org/run_style.html: "*style* = *verlet* or *verlet/split* or
 * *respa* or *respa/omp*"; "*verlet* args = none". run_style verlet selects the plain velocity-Verlet
 * path; run_style respa selects the rRESPA integrator (run/respa.ts). verlet/split and respa/omp are
 * refused. Measured with native LAMMPS (black box): before a box exists the command stops with
 * Run_style command before simulation box is defined.
 */
const runStyle: Handler = ({ sys }, a) => {
  if (!sys.hasBox) throw new StyleError('Run_style command before simulation box is defined');
  const style = a[0];
  if (style === 'verlet') {
    if (a.length > 1) throw new StyleError('run_style verlet takes no arguments');
    sys.respa = null;
    return;
  }
  if (style === 'respa') {
    sys.respa = parseRunStyleRespa(a);
    return;
  }
  // run_style verlet/split and verlet/split/rk need two partitions (-partition 2x1, one for the pair forces and one
  // for the kspace forces; the doc says the force calculation is split "over 2 partitions of processors"). Measured
  // with native LAMMPS (black box): verlet/split on one partition stops with Verlet/split requires 2 partitions.
  // The browser engine runs a single partition, so both are refused and named here.
  if (style === 'verlet/split' || style === 'verlet/split/rk') {
    throw new StyleError(`run_style ${style} needs two partitions (native LAMMPS: Verlet/split requires 2 partitions); the browser engine runs one partition, so it is refused`);
  }
  if (style === 'respa/omp') {
    throw new StyleError(`run_style ${style} is not supported by the browser engine (it runs velocity Verlet only)`);
  }
  throw new StyleError(`unknown run_style '${style ?? ''}' (verlet, verlet/split, respa or respa/omp)`);
};

const DUMP_STYLES: DumpStyle[] = ['atom', 'custom', 'xyz', 'extxyz', 'yaml', 'local', 'image'];

/** dump ID group style N file args — dump.html. */
const dump: Handler = ({ sys }, a) => {
  sys.state;
  const [id, group, style, every, file, ...cols] = a;
  if (!id || !group || !style || !every || !file) throw new StyleError('usage: dump ID group-ID style N file [attributes]');
  if (sys.dumps.some((d) => d.id === id)) throw new StyleError(`reuse of dump ID '${id}' (undump it first)`);
  if (/\/(gz|zstd)$/.test(style)) throw new StyleError(`dump ${style}: compressed dump files are not available in the browser; use ${style.split('/')[0]}`);
  if (!(DUMP_STYLES as string[]).includes(style)) throw new StyleError(`dump style '${style}' is not supported by the browser engine; supported: ${DUMP_STYLES.join(', ')}`);
  let n: number | string;
  if (every.startsWith('v_')) n = every.slice(2);
  else {
    n = int(every, 'dump interval');
    if (n < 1) throw new StyleError('dump interval must be > 0');
  }
  sys.dumps.push(new Dump(sys, id, group, style as DumpStyle, n, file, cols));
};

const undump: Handler = ({ sys }, a) => {
  const k = sys.dumps.findIndex((d) => d.id === a[0]);
  if (k < 0) throw new StyleError(`could not find undump ID '${a[0] ?? ''}'`);
  sys.dumps.splice(k, 1);
};

const dumpModify: Handler = ({ sys }, a) => {
  const d = sys.dumps.find((x) => x.id === a[0]);
  if (!d) throw new StyleError(`could not find dump_modify ID '${a[0] ?? ''}'`);
  d.modify(a.slice(1));
};

/** write_dump group style file args [modify ...] — write_dump.html: a single snapshot now. */
const writeDump: Handler = ({ sys }, a) => {
  const [group, style, file, ...rest] = a;
  if (!group || !style || !file) throw new StyleError('usage: write_dump group-ID style file [attributes] [modify ...]');
  const mi = rest.indexOf('modify');
  const cols = mi >= 0 ? rest.slice(0, mi) : rest;
  if (!(DUMP_STYLES as string[]).includes(style)) throw new StyleError(`write_dump style '${style}' is not supported; supported: ${DUMP_STYLES.join(', ')}`);
  // write_dump: a dump image file needs no '*' (multifile_override, see dump_image.ts)
  const d = new Dump(sys, '__write_dump', group, style as DumpStyle, 1, file, cols, true);
  if (mi >= 0) d.modify(rest.slice(mi + 1));
  d.write();
};

// ----------------------------------------------------------------- run

interface RunOpts {
  start: number | null;
  stop: number | null;
  /** run.html "pre": yes (default) creates the neighbor lists and computes the forces before the run. */
  pre: boolean;
  every: number;
  commands: string[];
}

/** Emits a frame for the 3D view. */
export const emitFrame = (sys: System): void => {
  if (!sys.hasBox) return;
  const s = sys.state;
  const ev: FrameEventOut = {
    kind: 'frame', step: s.step, x: Float64Array.from(s.x.subarray(0, 3 * s.n)), image: Int32Array.from(s.image.subarray(0, 3 * s.n)),
    type: Int32Array.from(s.type.subarray(0, s.n)), id: Int32Array.from(s.id.subarray(0, s.n)),
    box: { ...s.box, lo: [...s.box.lo], hi: [...s.box.hi], periodic: [...s.box.periodic], tilt: [...s.box.tilt], boundary: s.box.boundary.map((f) => [f[0], f[1]]) as typeof s.box.boundary, minLo: [...s.box.minLo], minHi: [...s.box.minHi] },
  };
  if (sys.frameVelocities) {
    ev.v = Float64Array.from(s.v.subarray(0, 3 * s.n));
    const mass = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) mass[i] = massOf(s, i);
    // the masses rarely change: they go with the first frame of a run and then only when they differ (index by index,
    // so a re-sorted atom order with different masses is sent again); the notebook keeps the last ones it received
    const prev = sys.frameMassSent;
    let same = prev !== null && prev.length === mass.length;
    for (let i = 0; same && i < mass.length; i++) same = prev![i] === mass[i];
    if (!same) {
      sys.frameMassSent = mass.slice();
      ev.mass = mass;
    }
  }
  sys.io.emit(ev);
};

type FrameEventOut = Extract<EngineEvent, { kind: 'frame' }>;

/** One thermo row: the thermo event and, with a log file open, its lines in the log (thermo.ts logRow). */
const emitThermoRow = (sys: System): number => {
  const row = sys.thermo.row();
  sys.io.emit({ kind: 'thermo', row });
  if (sys.logOpen) sys.logTable(sys.thermo.logRow(row));
  return sys.state.step;
};

/** How often (wall-clock ms) a run reports its speed (EngineEvent 'perf'). */
const PERF_EVERY_MS = 500;

const writeDumps = (sys: System, firstOfRun: boolean): boolean => {
  let any = false;
  for (const d of sys.dumps) {
    if (d.due(sys.state.step, firstOfRun)) { d.write(); any = true; }
  }
  return any;
};

/**
 * What a run with pre no depends on: the atoms, the box, the fixes, the computes and the pair style that the previous
 * run set up. run.html: "it is an error to use the pre no option" after such a change.
 */
const setupSignature = (sys: System): string => {
  const s = sys.state;
  const box = `${s.box.lo.join(',')}/${s.box.hi.join(',')}/${s.box.tilt.join(',')}`;
  return [s.n, box, sys.fixes.map((f) => `${f.id}:${f.style}`).join(','), sys.computes.map((c) => c.id).join(','), sys.ff.pair?.name ?? ''].join('|');
};

/** The setup signature at the end of the last run of each system (WeakMap: a clear makes a new system). */
const lastSetup = new WeakMap<System, string>();

/**
 * One run of n steps on coupled replicas (a fix with `coupled`, fix pimd): every replica is set up and output in its own
 * right (its thermo rows, dumps and restarts), and the replicas are stepped in lockstep (run/coupled.ts). Only replica 0's
 * output reaches the session's log and screen; the others reach it when Session.recordPartitionOutput is set.
 * Not supported with them: pre no, every, a CPU-only run (the accelerators step one system), run_style respa.
 */
const runCoupledSteps = async (ctx: Ctx, n: number, opts: RunOpts): Promise<number> => {
  const { sys, session } = ctx;
  const group = session.coupledSystems!;
  if (!opts.pre) throw new StyleError('run pre no is not supported with coupled replicas (fix pimd): use pre yes');
  if (opts.every > 0 && opts.every < n) throw new StyleError('run every is not supported with coupled replicas (fix pimd)');
  if (sys.respa) throw new StyleError('run_style respa is not supported with coupled replicas (fix pimd)');
  const first = sys.state.step;
  if (opts.start !== null && opts.start > first) throw new StyleError('run start cannot be after the current timestep');
  if (opts.stop !== null && opts.stop < first + n) throw new StyleError('run stop cannot be before the last timestep of the run');
  const lastThermo = new Map<System, number>();
  for (const r of group) {
    ensureThermoComputes(r);
    initRun(r);
    r.run = {
      inRun: true, firstStep: first, lastStep: first + n,
      beginStep: opts.start ?? first, endStep: opts.stop ?? first + n, t0: performance.now(), ranOnce: true,
    };
    r.io.emit({ kind: 'thermo-header', keywords: [...r.thermo.keywords], labels: r.thermo.labels(), units: r.units.style });
    r.io.emit({ kind: 'run', from: first, to: first + n, dt: r.state.dt, units: r.units.style });
    startRestarts(r);
    lastThermo.set(r, -1);
  }
  const nextThermoVar = new Map<System, number>(group.map((r) => [r, r.thermo.everyVar ? r.thermo.nextVariableStep() : -1]));
  const emitThermo = (r: System) => {
    lastThermo.set(r, emitThermoRow(r));
  };
  const t0 = sys.run.t0;
  const taken = await runVerletCoupled(group, n, {
    cancelled: () => session.isCancelled,
    afterSetup: (r) => {
      emitThermo(r);
      const dumped = writeDumps(r, true);
      if (dumped && r === sys) emitFrame(r);
    },
    afterStep: (r, step) => {
      writeRestarts(r, step);
      const th = r.thermo;
      let due = th.due(step, first, first + n);
      if (th.everyVar && step >= nextThermoVar.get(r)!) { due = true; nextThermoVar.set(r, th.nextVariableStep()); }
      if (due) emitThermo(r);
      writeDumps(r, false);
    },
    yieldMs: 30,
  });
  for (const r of group) {
    // a cancelled run still reports its last step (as runSteps)
    if (lastThermo.get(r) !== r.state.step && !haltRequested(r)) emitThermo(r);
    r.run.inRun = false;
    lastSetup.set(r, setupSignature(r));
  }
  const seconds = (performance.now() - t0) / 1000;
  sys.log(`Loop time of ${seconds.toFixed(3)} s for ${taken} steps with ${sys.state.n} atoms (CPU · fp64, ${group.length} coupled replicas)`);
  sys.io.emit({ kind: 'done', steps: taken, seconds, backend: 'CPU · fp64' });
  if (session.isCancelled) throw new RunCancelled();
  return taken;
};

/** One run of n steps with output; returns steps taken. */
const runSteps = async (ctx: Ctx, n: number, opts: RunOpts): Promise<number> => {
  const { sys, session } = ctx;
  // fix_halt.html: after a soft halt "additional run or minimize commands will be skipped" (no output)
  if (softHalted(sys)) return 0;
  clearHaltRequest(sys);
  if (session.coupledSystems) return runCoupledSteps(ctx, n, opts);
  const s = sys.state;
  const first = s.step;
  // run.html: start and stop are checked before anything changes, so a rejected run leaves the system as it was
  // (measured with native LAMMPS, black box: the checks come before the setup)
  if (opts.start !== null && opts.start > first) throw new StyleError('run start cannot be after the current timestep');
  if (opts.stop !== null && opts.stop < first + n) throw new StyleError('run stop cannot be before the last timestep of the run');
  // run.html: "if pre is specified as no then the initial setup is skipped, except for printing thermodynamic info.
  // Note that if pre is set to no for the very first run LAMMPS performs, then it is overridden". The run starts from the
  // neighbor lists, forces and energies the previous run left (measured with native LAMMPS, black box: after a
  // displacement, the setup row of a pre no run still shows the energy of the old positions; a pre yes run shows the new one).
  const reuse = !opts.pre && sys.run.ranOnce;
  if (reuse && sys.respa) throw new StyleError("Run flag 'pre no' not compatible with r-RESPA");
  if (reuse && lastSetup.get(sys) !== setupSignature(sys)) {
    throw new StyleError('run pre no after a change of the atoms, box, fixes, computes or pair style: use pre yes (run.html: it is an error to use pre no after such a change)');
  }
  ensureThermoComputes(sys);
  if (!reuse) initRun(sys);
  if (reuse) sys.forcesCurrent();
  sys.run = {
    inRun: true, firstStep: first, lastStep: first + n,
    beginStep: opts.start ?? first, endStep: opts.stop ?? first + n, t0: performance.now(), ranOnce: true,
  };
  const th = sys.thermo;
  sys.io.emit({ kind: 'thermo-header', keywords: [...th.keywords], labels: th.labels(), units: sys.units.style });
  if (sys.logOpen) sys.logTable(th.logHeader());
  sys.io.emit({ kind: 'run', from: first, to: first + n, dt: s.dt, units: sys.units.style });
  const frameEvery = session.frameEvery;
  let nextThermoVar = th.everyVar ? th.nextVariableStep() : -1;
  let lastThermo = -1;
  const emitThermo = () => {
    lastThermo = emitThermoRow(sys);
  };
  const afterSetup = () => {
    emitThermo();
    const dumped = writeDumps(sys, true);
    if (dumped || frameEvery > 0) emitFrame(sys);
  };
  startRestarts(sys);
  // run speed for the resource monitor: steps per second over the last ~0.5 s of wall time
  const perf = { t: performance.now(), step: first, calls: sys.ff.pairThreads?.calls ?? 0 };
  const emitPerf = (step: number, force = false) => {
    const now = performance.now();
    if (!force && now - perf.t < PERF_EVERY_MS) return;
    const dtWall = (now - perf.t) / 1000;
    if (dtWall <= 0 || step === perf.step) return;
    const calls = sys.ff.pairThreads?.calls ?? 0;
    sys.io.emit({
      kind: 'perf', step, atoms: s.n, stepsPerSec: (step - perf.step) / dtWall,
      elapsed: (now - sys.run.t0) / 1000, threaded: calls > perf.calls,
    });
    perf.t = now; perf.step = step; perf.calls = calls;
  };
  const afterStep = (step: number) => {
    emitPerf(step);
    writeRestarts(sys, step);
    let due = th.due(step, first, first + n);
    if (th.everyVar && step >= nextThermoVar) { due = true; nextThermoVar = th.nextVariableStep(); }
    if (due) emitThermo();
    const dumped = writeDumps(sys, false);
    if (dumped || (frameEvery > 0 && step % frameEvery === 0)) emitFrame(sys);
  };
  // run_style respa runs on the CPU engine (the GPU paths step plain velocity Verlet only)
  const { accel, reason } = sys.respa ? { accel: null, reason: null }
    : reuse ? { accel: null, reason: 'run pre no continues from the CPU forces of the previous run' }
      : accelerator(sys, session.forceBackend);
  if (reason) sys.log(`${session.backendLabel}: this run uses the general fp64 CPU engine (${reason})`);
  const threadCalls0 = sys.ff.pairThreads?.calls ?? 0;
  const taken = accel
    ? await runAccelerated(sys, n, accel, {
      cancelled: () => session.isCancelled,
      afterSetup,
      afterStep,
      hostStep: (step) => th.due(step, first, first + n) || (th.everyVar !== null && step >= nextThermoVar)
        || sys.dumps.some((d) => step >= d.delay && (d.everyVar ? true : step % d.every === 0))
        || (frameEvery > 0 && step % frameEvery === 0) || restartDue(sys, step),
    })
    : sys.respa
      ? await runRespa(sys, n, { cancelled: () => session.isCancelled, afterSetup, afterStep, yieldMs: 30 })
      : await runVerlet(sys, n, {
        cancelled: () => session.isCancelled,
        afterSetup,
        afterStep,
        skipSetup: reuse,
        // keep the page responsive: yield about every 30 ms
        yieldMs: 30,
      });
  // a cancelled run still reports its last step; a run stopped by fix halt does not print a row for a step
  // that is not on the thermo interval (measured with native LAMMPS, black box)
  if (lastThermo !== s.step && !haltRequested(sys)) emitThermo();
  if (sys.logOpen) sys.logTable(th.logEnd());
  lastSetup.set(sys, setupSignature(sys));
  emitPerf(s.step, true);
  sys.run.inRun = false;
  emitFrame(sys);
  const seconds = (performance.now() - sys.run.t0) / 1000;
  // the general engine names the shared-memory threads only when its pair term ran on them
  const threaded = (sys.ff.pairThreads?.calls ?? 0) > threadCalls0;
  const label = accel || threaded ? session.backendLabel : 'CPU · fp64';
  sys.log(`Loop time of ${seconds.toFixed(3)} s for ${taken} steps with ${s.n} atoms (${label})`);
  sys.io.emit({ kind: 'done', steps: taken, seconds, backend: label });
  if (session.isCancelled) throw new RunCancelled();
  return taken;
};

/**
 * run N [upto] [start N1] [stop N2] [pre yes|no] [post yes|no] [every M c1 c2 ...] —
 * run.html: "A value of N = 0 is acceptable"; "upto ... perform a run starting
 * at the current timestep up to the specified timestep"; "The start and stop
 * keywords can be used in conjunction with the run command to ... ramp a
 * target temperature"; "If the every keyword is used ... the run is
 * performed in chunks of M timesteps and the specified commands are invoked
 * between chunks".
 */
const run: Handler = async (ctx, a) => {
  ctx.sys.frameMassSent = null; // the first frame of the run carries the masses again
  const { sys } = ctx;
  const s = sys.state;
  let n = int(a[0], 'number of steps');
  const opts: RunOpts = { start: null, stop: null, pre: true, every: 0, commands: [] };
  for (let k = 1; k < a.length;) {
    switch (a[k]) {
      case 'upto': n -= s.step; k++; break;
      case 'start': opts.start = int(a[k + 1], 'start'); k += 2; break;
      case 'stop': opts.stop = int(a[k + 1], 'stop'); k += 2; break;
      case 'pre': case 'post':
        if (a[k + 1] !== 'yes' && a[k + 1] !== 'no') throw new StyleError(`run ${a[k]} must be yes or no`);
        // run.html: "If post is specified as no, the full timing summary is skipped; only a one-line summary timing is
        // printed." The engine prints only the one-line summary, so post is accepted and has no further effect.
        if (a[k] === 'pre') opts.pre = a[k + 1] === 'yes';
        k += 2;
        break;
      case 'every':
        opts.every = int(a[k + 1], 'every');
        opts.commands = a.slice(k + 2);
        k = a.length;
        if (opts.commands.length === 1 && opts.commands[0] === 'NULL') opts.commands = [];
        break;
      default: throw new StyleError(`unknown run keyword '${a[k]}'`);
    }
  }
  if (n < 0) throw new StyleError('run needs N >= 0 (with upto: a timestep at or after the current one)');
  if (opts.every <= 0 || opts.every >= n) {
    await runSteps(ctx, n, opts);
    return;
  }
  // every M: chunks with commands in between, start/stop spanning the whole run
  const begin = s.step;
  if (opts.start === null) opts.start = begin;
  if (opts.stop === null) opts.stop = begin + n;
  // run.html: "If the pre and post options are set to no when used with the every keyword, then the first run will do
  // the full setup and ... these operations will be skipped for intermediate runs."
  let left = n;
  for (let chunk = 0; left > 0; chunk++) {
    const m = Math.min(opts.every, left);
    await runSteps(ctx, m, { ...opts, pre: opts.pre || chunk === 0 });
    left -= m;
    for (const c of opts.commands) await ctx.session.runCommand(c, 0);
  }
};

/** minimize etol ftol maxiter maxeval — minimize.html. */
const minimizeCmd: Handler = async (ctx, a) => {
  const { sys, session } = ctx;
  if (a.length !== 4) throw new StyleError('usage: minimize etol ftol maxiter maxeval');
  const etol = num(a[0], 'etol'), ftol = num(a[1], 'ftol');
  const maxiter = int(a[2], 'maxiter'), maxeval = int(a[3], 'maxeval');
  if (etol < 0 || ftol < 0) throw new StyleError('minimize tolerances must be >= 0');
  if (softHalted(sys)) return;
  clearHaltRequest(sys);
  ensureThermoComputes(sys);
  initRun(sys);
  const s = sys.state;
  const th = sys.thermo;
  sys.run = { inRun: true, firstStep: s.step, lastStep: s.step + maxiter, beginStep: s.step, endStep: s.step + maxiter, t0: performance.now(), ranOnce: true };
  sys.io.emit({ kind: 'thermo-header', keywords: [...th.keywords], labels: th.labels(), units: sys.units.style });
  if (sys.logOpen) sys.logTable(th.logHeader());
  const first = s.step;
  startRestarts(sys);
  let lastRestart = -1;
  // one thermo row per step: the final row of a minimization that stops on a step already printed
  // (maxiter 0, or a thermo step) is not repeated (measured with native LAMMPS: minimize ... 0 0
  // prints the step-0 row once)
  let lastThermo = -1;
  const result = await minimize(sys, { etol, ftol, maxiter, maxeval }, {
    cancelled: () => session.isCancelled || haltRequested(sys),
    thermo: (iterDone) => {
      if ((iterDone || th.due(s.step, first, Number.MAX_SAFE_INTEGER)) && s.step !== lastThermo) {
        lastThermo = emitThermoRow(sys);
      }
      writeDumps(sys, s.step === first);
      if (s.step !== first && s.step !== lastRestart && restartDue(sys, s.step)) { writeRestarts(sys, s.step); lastRestart = s.step; }
    },
  });
  // restart.html: "A restart file is written on the last timestep of a minimization if N > 0 and
  // the minimization converges."
  const converged = result.reason === 'energy tolerance' || result.reason === 'force tolerance' || result.reason === 'forces are zero';
  if (converged && sys.restartOut.length && s.step !== lastRestart) writeRestarts(sys, s.step, true);
  sys.run.inRun = false;
  emitFrame(sys);
  if (sys.logOpen) sys.logTable(th.logEnd());
  const seconds = (performance.now() - sys.run.t0) / 1000;
  sys.log([
    `Minimization stats:`,
    `  Stopping criterion = ${result.reason}`,
    `  Energy initial, next-to-last, final = ${result.e0} ${result.ePrev} ${result.e1}`,
    `  Force two-norm initial, final = ${result.fnorm0} ${result.fnorm1}`,
    `  Force max component initial, final = ${result.fmax0} ${result.fmax1}`,
    `  Iterations, force evaluations = ${result.iterations} ${result.evaluations}`,
  ].join('\n'));
  sys.io.emit({ kind: 'done', steps: result.iterations, seconds, backend: session.backendLabel });
  if (session.isCancelled) throw new RunCancelled();
};

/** min_style cg|sd|fire|quickmin|hftn — min_style.html. */
const minStyle: Handler = ({ sys }, a) => {
  if (a[0] && Object.prototype.hasOwnProperty.call(MINIMIZERS, a[0])) { sys.minStyle = a[0]; return; }
  if (!a[0] || !(MIN_STYLES as readonly string[]).includes(a[0])) {
    throw new StyleError(`min_style '${a[0] ?? ''}' is not supported by the browser engine; supported: ${MIN_STYLES.join(', ')}`);
  }
  sys.minStyle = a[0];
};

/** min_modify keyword value ... — min_modify.html. */
const minModify: Handler = ({ sys }, a) => {
  for (let k = 0; k < a.length;) {
    const key = a[k];
    const v = a[k + 1];
    switch (key) {
      case 'dmax': sys.minSettings.dmax = num(v, 'dmax'); k += 2; break;
      case 'line':
        // min_modify.html: spin_cubic and spin_none are line searches of the spin minimizers (min_style spin/cg, run/
        // minimizers.ts); they are kept in the minimizer settings and refused by the other minimizers in minimize().
        if (v === 'spin_cubic' || v === 'spin_none') {
          if (!sys.state?.sp) throw new StyleError(`min_modify line ${v} needs spin atoms (atom_style spin)`);
          sys.minSettings.fire.line = v;
          k += 2;
          break;
        }
        if (!['backtrack', 'quadratic', 'forcezero'].includes(v ?? '')) throw new StyleError('min_modify line must be backtrack, quadratic or forcezero');
        sys.minSettings.line = v as 'backtrack' | 'quadratic' | 'forcezero';
        k += 2;
        break;
      case 'norm':
        if (!['two', 'max', 'inf'].includes(v ?? '')) throw new StyleError('min_modify norm must be two, max or inf');
        sys.minSettings.norm = v as 'two' | 'max' | 'inf';
        k += 2;
        break;
      case 'integrator': case 'tmax': case 'tmin': case 'delaystep': case 'dtgrow': case 'dtshrink': case 'alpha0': case 'alphashrink':
      case 'halfstepback': case 'initialdelay': case 'vdfmax': case 'abcfire': case 'discrete_factor': case 'alpha_damp':
        sys.minSettings.fire[key] = v ?? '';
        k += 2;
        break;
      default: throw new StyleError(`unknown min_modify keyword '${key}'`);
    }
  }
};

export const RUN_COMMANDS: Record<string, Handler> = {
  fix, unfix, fix_modify: fixModify, compute, uncompute, compute_modify: computeModify,
  thermo, thermo_style: thermoStyle, thermo_modify: thermoModify, run_style: runStyle,
  dump, undump, dump_modify: dumpModify, write_dump: writeDump,
  run, minimize: minimizeCmd, min_style: minStyle, min_modify: minModify,
};
