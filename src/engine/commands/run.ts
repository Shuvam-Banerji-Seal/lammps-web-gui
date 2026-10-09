import type { Ctx, Handler } from './args';
import { int, num } from './args';
import { StyleError } from '../force/types';
import { FIX_STYLES, COMPUTE_STYLES } from '../styles';
import { Dump, type DumpStyle } from '../output/dump';
import { initRun, runVerlet } from '../run/verlet';
import { RunCancelled } from '../errors';
import { DYNAMIC_GROUP_FIXES, NO_DYNAMIC_GROUP_FIXES } from '../group';
import type { System } from '../system';
import { minimize, MIN_STYLES } from '../run/min';
import { accelerator, runAccelerated } from '../run/accel';
import { restartDue, startRestarts, writeRestarts } from './restart';

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
    throw new StyleError(`fix style '${style}' is not supported by the browser engine; supported: ${Object.keys(FIX_STYLES).sort().join(', ')}`);
  }
  // the style is checked first, so a fix the engine lacks is named even before the box exists;
  // fix_property_atom.html: "This fix is one of a small number that can be defined in an input
  // script before the simulation box is created or atoms are defined." fix_cmap.html: "To function
  // as expected this fix command must be issued *before* a read_data command".
  if (style !== 'property/atom' && style !== 'cmap') sys.state;
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
    throw new StyleError(`compute style '${style}' is not supported by the browser engine; supported: ${Object.keys(COMPUTE_STYLES).sort().join(', ')}`);
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
 * *respa* or *respa/omp*"; "*verlet* args = none". The engine integrates with velocity Verlet, so
 * run_style verlet changes nothing; the other styles split the force computation across levels or
 * partitions, which the engine does not do. Measured with native LAMMPS (black box): before a box
 * exists the command stops with Run_style command before simulation box is defined.
 */
const runStyle: Handler = ({ sys }, a) => {
  if (!sys.hasBox) throw new StyleError('Run_style command before simulation box is defined');
  const style = a[0];
  if (style === 'verlet') {
    if (a.length > 1) throw new StyleError('run_style verlet takes no arguments');
    return;
  }
  if (style === 'verlet/split' || style === 'respa' || style === 'respa/omp') {
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
  const d = new Dump(sys, '__write_dump', group, style as DumpStyle, 1, file, cols);
  if (mi >= 0) d.modify(rest.slice(mi + 1));
  d.write();
};

// ----------------------------------------------------------------- run

interface RunOpts {
  start: number | null;
  stop: number | null;
  every: number;
  commands: string[];
}

/** Emits a frame for the 3D view. */
export const emitFrame = (sys: System): void => {
  if (!sys.hasBox) return;
  const s = sys.state;
  sys.io.emit({
    kind: 'frame', step: s.step, x: Float64Array.from(s.x.subarray(0, 3 * s.n)), image: Int32Array.from(s.image.subarray(0, 3 * s.n)),
    type: Int32Array.from(s.type.subarray(0, s.n)), id: Int32Array.from(s.id.subarray(0, s.n)),
    box: { ...s.box, lo: [...s.box.lo], hi: [...s.box.hi], periodic: [...s.box.periodic], tilt: [...s.box.tilt], boundary: s.box.boundary.map((f) => [f[0], f[1]]) as typeof s.box.boundary, minLo: [...s.box.minLo], minHi: [...s.box.minHi] },
  });
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

/** One run of n steps with output; returns steps taken. */
const runSteps = async (ctx: Ctx, n: number, opts: RunOpts): Promise<number> => {
  const { sys, session } = ctx;
  const s = sys.state;
  ensureThermoComputes(sys);
  initRun(sys);
  const first = s.step;
  sys.run = {
    inRun: true, firstStep: first, lastStep: first + n,
    beginStep: opts.start ?? first, endStep: opts.stop ?? first + n, t0: performance.now(), ranOnce: true,
  };
  if (opts.start !== null && opts.start > first) throw new StyleError('run start cannot be after the current timestep');
  if (opts.stop !== null && opts.stop < first + n) throw new StyleError('run stop cannot be before the last timestep of the run');
  const th = sys.thermo;
  sys.io.emit({ kind: 'thermo-header', keywords: [...th.keywords], labels: th.labels(), units: sys.units.style });
  sys.io.emit({ kind: 'run', from: first, to: first + n, dt: s.dt, units: sys.units.style });
  const frameEvery = session.frameEvery;
  let nextThermoVar = th.everyVar ? th.nextVariableStep() : -1;
  let lastThermo = -1;
  const emitThermo = () => {
    sys.io.emit({ kind: 'thermo', row: th.row() });
    lastThermo = s.step;
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
  const { accel, reason } = accelerator(sys, session.forceBackend);
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
    : await runVerlet(sys, n, {
      cancelled: () => session.isCancelled,
      afterSetup,
      afterStep,
      // keep the page responsive: yield about every 30 ms
      yieldMs: 30,
    });
  // a cancelled run still reports its last step
  if (lastThermo !== s.step) emitThermo();
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
  const { sys } = ctx;
  const s = sys.state;
  let n = int(a[0], 'number of steps');
  const opts: RunOpts = { start: null, stop: null, every: 0, commands: [] };
  for (let k = 1; k < a.length;) {
    switch (a[k]) {
      case 'upto': n -= s.step; k++; break;
      case 'start': opts.start = int(a[k + 1], 'start'); k += 2; break;
      case 'stop': opts.stop = int(a[k + 1], 'stop'); k += 2; break;
      case 'pre': case 'post':
        if (a[k + 1] !== 'yes' && a[k + 1] !== 'no') throw new StyleError(`run ${a[k]} must be yes or no`);
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
  let left = n;
  while (left > 0) {
    const m = Math.min(opts.every, left);
    await runSteps(ctx, m, opts);
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
  ensureThermoComputes(sys);
  initRun(sys);
  const s = sys.state;
  const th = sys.thermo;
  sys.run = { inRun: true, firstStep: s.step, lastStep: s.step + maxiter, beginStep: s.step, endStep: s.step + maxiter, t0: performance.now(), ranOnce: true };
  sys.io.emit({ kind: 'thermo-header', keywords: [...th.keywords], labels: th.labels(), units: sys.units.style });
  const first = s.step;
  startRestarts(sys);
  let lastRestart = -1;
  // one thermo row per step: the final row of a minimization that stops on a step already printed
  // (maxiter 0, or a thermo step) is not repeated (measured with native LAMMPS: minimize ... 0 0
  // prints the step-0 row once)
  let lastThermo = -1;
  const result = await minimize(sys, { etol, ftol, maxiter, maxeval }, {
    cancelled: () => session.isCancelled,
    thermo: (iterDone) => {
      if ((iterDone || th.due(s.step, first, Number.MAX_SAFE_INTEGER)) && s.step !== lastThermo) {
        sys.io.emit({ kind: 'thermo', row: th.row() });
        lastThermo = s.step;
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
        if (!['backtrack', 'quadratic', 'forcezero', 'spin_cubic', 'spin_none'].includes(v ?? '')) throw new StyleError('min_modify line must be backtrack, quadratic or forcezero');
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
