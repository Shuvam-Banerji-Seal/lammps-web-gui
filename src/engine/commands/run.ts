import type { Ctx, Handler } from './args';
import { int, num, missingArgs, numeric, inumeric, bnumeric, logical } from './args';
import { errorUrl } from '../formula';
import { StyleError, type KSpace } from '../force/types';
import { removedStyleError } from '../deprecated';
import { loadNativeStyles, unknownStyleError } from '../nativeStyles';
import { gplHook } from '../gplHooks';
import { checkFixStateNotStored } from '../fix/restartRecord';
import { FIX_MODIFY_KEYWORDS } from '../fix/fix';
import { FIX_STYLES, COMPUTE_STYLES, styleNames } from '../styles';
import { Dump, type DumpStyle } from '../output/dump';
import { initRun, runVerlet } from '../run/verlet';
import { runVerletCoupled } from '../run/coupled';
import { parseRunStyleRespa, runRespa } from '../run/respa';
import { PairHybrid } from '../force/pair/hybrid';
import { RunCancelled } from '../errors';
import { FIX_COMMAND_STYLE } from '../group';
import type { System } from '../system';
import type { EngineEvent } from '../types';
import { hasCharge, massOf } from '../atoms';
import { defaultMinSettings, minimize, MIN_STYLES } from '../run/min';
import { MINIMIZERS } from '../run/minimizers';
import { accelerator, runAccelerated } from '../run/accel';
import { restartDue, startRestarts, writeRestarts } from './restart';
import { clearHaltRequest, haltRequested, softHalted } from '../run/halt';

/*
 * fix / compute / thermo / dump / run commands.
 */

const RESERVED_COMPUTES = new Set(['thermo_temp', 'thermo_press', 'thermo_pe']);

/** The fix styles native LAMMPS creates before the box exists (src/modify.cpp Modify::add_fix exceptions, at b332f28). */
const FIX_BEFORE_BOX = new Set(['property/atom', 'cmap', 'cmap3', 'rx', 'deprecated', 'amoeba/pitorsion', 'amoeba/bitorsion', 'DUMMY']);

/** The computes thermo output needs ("compute thermo_temp all temp" etc., thermo_style.html). */
export const ensureThermoComputes = (sys: System): void => {
  if (sys.computes.some((c) => c.id === 'thermo_temp')) return;
  sys.computes.push(COMPUTE_STYLES.temp(sys, 'thermo_temp', 'all', []));
  sys.computes.push(COMPUTE_STYLES.pressure(sys, 'thermo_press', 'all', ['thermo_temp']));
  sys.computes.push(COMPUTE_STYLES.pe(sys, 'thermo_pe', 'all', []));
};

/** fix ID group style args — fix.html: "If you specify a fix ID that already exists, it will replace the existing fix" when the style matches. */
const fix: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/modify.cpp Modify::add_fix and src/fix.cpp Fix::Fix at b332f28): the box (for all but a
  // few styles), the group, a replaced fix's style, the style itself and then the ID are checked, in that order.
  // fix_property_atom.html: "This fix is one of a small number that can be defined in an input script before the
  // simulation box is created or atoms are defined." fix_cmap.html: "To function as expected this fix command must be
  // issued *before* a read_data command". examples/amoeba/in.ubiquitin defines fix amoeba/pitorsion and fix
  // amoeba/bitorsion before read_data (their sections are read by that command, like fix cmap). fix rx
  // (src/DPD-REACT/fix_rx.cpp post_constructor) only adds its species vectors as fix property/atom fixes, so it may
  // precede create_box (examples/PACKAGES/dpd-react/dpdrx-shardlow does); its per-atom vectors are attached when the box exists
  if (a.length < 3) throw missingArgs('fix');
  const [id, group, style] = a;
  if (!sys.hasBox && !FIX_BEFORE_BOX.has(style)) throw new StyleError(`Fix ${style} command before simulation box is defined`);
  if (sys.groups.find(group) < 0) throw new StyleError(`Could not find fix group ID ${group}`);
  const k = sys.fixes.findIndex((f) => f.id === id);
  if (k >= 0 && sys.fixes[k].style !== style) throw new StyleError('Replacing a fix, but new style != old style');
  const make = FIX_STYLES[style];
  if (!make) {
    throw unknownStyleError('fix', style, () => removedStyleError('fix', style) ?? new StyleError(`fix style '${style}' is not supported by the browser engine; supported: ${styleNames().fix.join(', ')}`));
  }
  if (!/^[A-Za-z0-9_]+$/.test(id)) throw new StyleError('Fix ID must be alphanumeric or underscore characters');
  // native's constructor of the style checks the number of words first (most styles; gplHook('fix'))
  const argMsg = gplHook<(style: string, narg: number) => string | null>('fix')(style, a.length);
  if (argMsg) throw new StyleError(argMsg);
  checkFixStateNotStored(sys, id, style);
  ensureThermoComputes(sys);
  if (k >= 0) sys.fixes[k].destroy?.();
  const f = make(sys, id, group, a.slice(3));
  // a fix on a dynamic group is checked at the run's init (checkDynamicGroupFixes, src/modify.cpp Modify::init)
  FIX_COMMAND_STYLE.set(f, style);
  if (k >= 0) sys.fixes[k] = f; else sys.fixes.push(f);
  sys.refreshComputes();
};

const unfix: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/input.cpp Input::unfix at b332f28)
  if (a.length !== 1) throw new StyleError('Unfix command expects exactly one argument');
  const k = sys.fixes.findIndex((f) => f.id === a[0]);
  if (k < 0) throw new StyleError(`Could not find fix ID ${a[0]} to delete`);
  sys.fixes[k].destroy?.();
  sys.fixes.splice(k, 1);
  sys.refreshComputes();
};

/**
 * fix_modify fix-ID keyword value ... — fix_modify.html. As native LAMMPS does (src/modify.cpp Modify::modify_fix and
 * src/fix.cpp Fix::modify_params at b332f28): the fix reads its own keywords and then the common ones (Fix.modify), and
 * a keyword it does not take stops with Fix ID style does not support fix_modify keyword command.
 */
const fixModify: Handler = ({ sys }, a) => {
  if (a.length < 2) throw missingArgs('fix_modify');
  const f = sys.fixes.find((x) => x.id === a[0]);
  if (!f) throw new StyleError(`Could not find fix_modify ID ${a[0]}`);
  for (let k = 1; k < a.length;) {
    const key = a[k];
    if (FIX_MODIFY_KEYWORDS.has(key) && k + 2 > a.length) throw missingArgs(`fix_modify ${key}`);
    const used = f.modify(key, a.slice(k + 1));
    if (used <= 0) throw new StyleError(`Fix ${f.id} ${f.invokedStyle} does not support fix_modify ${key} command`);
    k += 1 + used;
  }
};

/** The keywords of native Fix::modify_params, each with one value. */

/** compute ID group style args — compute.html. */
const compute: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/modify.cpp Modify::add_compute and src/compute.cpp Compute::Compute at b332f28): a reused
  // ID, the style, the ID's characters and the group, in that order
  if (a.length < 3) throw missingArgs('compute');
  sys.state;
  const [id, group, style] = a;
  if (sys.computes.some((c) => c.id === id)) throw new StyleError(`Reuse of compute ID '${id}'`);
  const make = COMPUTE_STYLES[style];
  if (!make) {
    throw unknownStyleError('compute', style, () => removedStyleError('compute', style) ?? new StyleError(`compute style '${style}' is not supported by the browser engine; supported: ${styleNames().compute.join(', ')}`));
  }
  if (!/^[A-Za-z0-9_]+$/.test(id)) throw new StyleError(`Compute ID ${id} must only have alphanumeric or underscore characters`);
  if (sys.groups.find(group) < 0) throw new StyleError(`Group ID ${group} requested by compute ${style} does not exist`);
  // native's constructor of the style checks the number of words first (most styles; gplHook('compute'))
  const argMsg = gplHook<(style: string, narg: number) => string | null>('compute')(style, a.length);
  if (argMsg) throw new StyleError(argMsg);
  ensureThermoComputes(sys);
  if (style === 'msd' && sys.groups.isDynamic(sys.groupBit(group))) throw new StyleError('Compute msd is not compatible with dynamic groups');
  sys.computes.push(make(sys, id, group, a.slice(3)));
};

const uncompute: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/input.cpp Input::uncompute at b332f28). Native deletes the thermo computes too and stops
  // at the next run when they are still referenced; the engine refuses at once
  if (a.length !== 1) throw new StyleError('Uncompute command expects exactly one argument');
  const k = sys.computes.findIndex((c) => c.id === a[0]);
  if (k < 0) throw new StyleError(`Could not find compute ID ${a[0]} to delete`);
  if (RESERVED_COMPUTES.has(a[0])) throw new StyleError(`compute ${a[0]} is used by thermo output and cannot be deleted by the browser engine`);
  sys.computes.splice(k, 1);
};

/** compute_modify compute-ID keyword value ... — compute_modify.html. */
const computeModify: Handler = ({ sys }, a) => {
  ensureThermoComputes(sys);
  // As native LAMMPS does (src/modify.cpp Modify::modify_compute and src/compute.cpp Compute::modify_params at b332f28)
  if (a.length < 2) throw missingArgs('compute_modify');
  const c = sys.computes.find((x) => x.id === a[0]);
  if (!c) throw new StyleError(`Could not find compute_modify ID ${a[0]}`);
  for (let k = 1; k < a.length;) {
    const used = c.modify(a[k], a.slice(k + 1));
    if (used <= 0) throw new StyleError(`Compute ${c.id} ${c.style} does not support compute_modify ${a[k]} command`);
    k += 1 + used;
  }
  sys.refreshComputes();
};

/** thermo N | thermo v_name — thermo.html. */
const thermo: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/output.cpp Output::set_thermo at b332f28): a v_name is looked up at run setup
  if (a.length !== 1) throw new StyleError('Illegal thermo command');
  if (a[0].startsWith('v_')) {
    sys.thermo.everyVar = a[0].slice(2);
    sys.thermo.every = 0;
    return;
  }
  const n = inumeric(a[0]);
  if (n < 0) throw new StyleError(`Illegal thermo output frequency ${n}`);
  sys.thermo.every = n;
  sys.thermo.everyVar = null;
};

const thermoStyle: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/output.cpp Output::create_thermo at b332f28)
  if (a.length < 1) throw missingArgs('thermo_style');
  if (!sys.hasBox) throw new StyleError('Thermo_style command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  ensureThermoComputes(sys);
  sys.thermo.setStyle(a[0] ?? '', a.slice(1));
};

const thermoModify: Handler = ({ sys }, a) => {
  if (sys.hasBox) ensureThermoComputes(sys);
  sys.thermo.modify(a);
};

/**
 * run_style style args — docs.lammps.org/run_style.html: "*style* = *verlet* or *verlet/split* or
 * *verlet/split/rk* or *respa* or *respa/omp*"; "*verlet* args = none". run_style verlet selects the plain
 * velocity-Verlet path; run_style respa selects the rRESPA integrator (run/respa.ts); verlet/split and
 * verlet/split/rk split the force calculation between two partitions (runSplitKspace below); respa/omp is
 * refused. Measured with native LAMMPS (black box): before a box exists the command stops with
 * Run_style command before simulation box is defined.
 */
const runStyle: Handler = ({ sys }, a) => {
  if (!sys.hasBox) throw new StyleError('Run_style command before simulation box is defined');
  // As native LAMMPS does (src/update.cpp Update::create_integrate at b332f28)
  if (a.length < 1) throw missingArgs('run_style');
  const style = a[0];
  // As native LAMMPS does (src/update.cpp at b332f28, Update::new_integrate, and src/verlet.cpp Verlet::Verlet): the
  // verlet integrator ignores the words after its name, so run_style verlet bogus 1 is accepted
  if (style === 'verlet') {
    sys.respa = null;
    sys.splitRun = null;
    return;
  }
  if (style === 'respa') {
    const pair = sys.ff.pair;
    sys.respa = parseRunStyleRespa(a, { hybridStyles: pair instanceof PairHybrid ? pair.subs.length : null, log: (m) => sys.log(m), warn: (m) => sys.warn(m) });
    sys.splitRun = null;
    return;
  }
  // run_style.html: "The *verlet/split* style is also a velocity-Verlet integrator, but it splits the force calculation
  // within each timestep over 2 partitions of processors." As native LAMMPS does (src/REPLICA/verlet_split.cpp at
  // b332f28): verlet/split checks the partition count when the run style is set; verlet/split/rk does not (its
  // pppm/rk checks it when the run starts).
  if (style === 'verlet/split' || style === 'verlet/split/rk') {
    if (style === 'verlet/split' && sys.vars.nworlds !== 2) throw new StyleError('Verlet/split requires 2 partitions');
    sys.respa = null;
    sys.splitRun = style;
    return;
  }
  if (style === 'respa/omp') {
    throw new StyleError(`run_style ${style} is not supported by the browser engine (it runs velocity Verlet only)`);
  }
  return loadNativeStyles().then(() => {
    throw unknownStyleError('integrate', style ?? '', () => new StyleError(`run_style ${style} is not supported by the browser engine (verlet, verlet/split, verlet/split/rk, respa)`));
  });
};

// cfg is a lazily loaded helper (lazyStyles.ts group gpl_dump_cfg, loaded before the dump command by the session: its
// name is a word of the command)
const DUMP_STYLES: DumpStyle[] = ['atom', 'custom', 'cfg', 'xyz', 'extxyz', 'yaml', 'local', 'image'];
/** Dump styles of lazily loaded helpers (lazyStyles.ts). */
const LAZY_DUMP_STYLES: string[] = ['vtk', 'grid', 'grid/vtk'];
/** The COMPRESS package's gzip dump styles (dump.html: "atom/gz, cfg/gz, custom/gz, local/gz, xyz/gz"). */
const GZ_STYLES = ['atom/gz', 'cfg/gz', 'custom/gz', 'local/gz', 'xyz/gz'];

/**
 * A COMPRESS style name: its base style and the /gz name (null for an uncompressed style). The zstd styles need the
 * Zstandard compressor, which the browser engine does not have.
 */
const compressStyle = (style: string): { base: string; gzStyle: string | null } => {
  if (GZ_STYLES.includes(style)) return { base: style.slice(0, -3), gzStyle: style };
  if (/^(atom|cfg|custom|local|xyz)\/zstd$/.test(style)) {
    throw new StyleError(`dump ${style}: Zstandard compression is not available in the browser engine; use ${style.split('/')[0]}/gz or ${style.split('/')[0]}`);
  }
  return { base: style, gzStyle: null };
};

/** dump ID group style N file args — dump.html. */
const dump: Handler = ({ sys }, a) => {
  sys.state;
  const [id, group, style, every, file, ...cols] = a;
  // As native LAMMPS does (src/output.cpp Output::add_dump at b332f28): "if (narg < 5) utils::missing_cmd_args(FLERR,"dump", error)"
  if (a.length < 5) throw new StyleError('Illegal dump command: missing argument(s)');
  // then the ID, the group and the frequency (utils::inumeric, > 0; a v_name is a dump_modify every value), before the
  // style is looked up
  if (sys.dumps.some((d) => d.id === id)) throw new StyleError(`Reuse of dump ID: ${id}`);
  if (sys.groups.find(group) < 0) throw new StyleError(`Could not find dump group ID: ${group}`);
  const n = inumeric(every);
  if (n <= 0) throw new StyleError(`Invalid dump frequency ${every}`);
  const { base, gzStyle } = compressStyle(style);
  // vtk, grid and grid/vtk are lazily loaded styles (lazyStyles.ts groups gpl_dump_vtk and gpl_dump_grid, loaded before
  // this command by the session)
  if (!(DUMP_STYLES as string[]).includes(base) && !LAZY_DUMP_STYLES.includes(base)) {
    return loadNativeStyles().then(() => {
      throw unknownStyleError('dump', style, () => removedStyleError('dump', style) ?? new StyleError(`dump style '${style}' is not supported by the browser engine; supported: ${[...DUMP_STYLES, ...LAZY_DUMP_STYLES, ...GZ_STYLES].join(', ')}`));
    });
  }
  sys.dumps.push(new Dump(sys, id, group, base as DumpStyle, n, file, cols, false, gzStyle));
};

// As native LAMMPS does (src/input.cpp Input::undump and src/output.cpp Output::delete_dump, modify_dump at b332f28)
const undump: Handler = ({ sys }, a) => {
  if (a.length !== 1) throw new StyleError('Undump command expects exactly one argument');
  const k = sys.dumps.findIndex((d) => d.id === a[0]);
  if (k < 0) throw new StyleError(`Could not find undump ID: ${a[0]}`);
  sys.dumps.splice(k, 1);
};

const dumpModify: Handler = ({ sys }, a) => {
  if (a.length < 2) throw missingArgs('dump_modify');
  const d = sys.dumps.find((x) => x.id === a[0]);
  if (!d) throw new StyleError(`Could not find dump_modify ID: ${a[0]}`);
  d.modify(a.slice(1));
};

/** write_dump group style file args [modify ...] — write_dump.html: a single snapshot now. */
const writeDump: Handler = ({ sys }, a) => {
  const [group, style, file, ...rest] = a;
  // As native LAMMPS does (src/write_dump.cpp WriteDump::command at b332f28)
  if (!group || !style || !file) throw missingArgs('write_dump');
  const mi = rest.indexOf('modify');
  const cols = mi >= 0 ? rest.slice(0, mi) : rest;
  const { base, gzStyle } = compressStyle(style);
  if (!(DUMP_STYLES as string[]).includes(base) && base !== 'grid') throw new StyleError(`write_dump style '${style}' is not supported; supported: ${[...DUMP_STYLES, 'grid', ...GZ_STYLES].join(', ')}`);
  // write_dump: a dump image or cfg file needs no '*' (multifile_override, see dump_image.ts; As native LAMMPS does
  // (src/write_dump.cpp at b332f28): "if (strcmp(arg[1], "cfg") == 0) dump->multifile_override = 1")
  const d = new Dump(sys, '__write_dump', group, base as DumpStyle, 1, file, cols, true, gzStyle);
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
export const emitThermoRow = (sys: System): number => {
  const row = sys.thermo.row();
  sys.io.emit({ kind: 'thermo', row });
  if (sys.logOpen) sys.logTable(sys.thermo.logRow(row));
  return sys.state.step;
};

/** How often (wall-clock ms) a run reports its speed (EngineEvent 'perf'). */
const PERF_EVERY_MS = 500;

export const writeDumps = (sys: System, firstOfRun: boolean): boolean => {
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

/** Records the setup of a run that a session-level driver (the tempering commands, src/engine-gpl/replica) ran, so a later pre no run may continue it. */
export const markRunSetup = (sys: System): void => { lastSetup.set(sys, setupSignature(sys)); };

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
  if (opts.start !== null && opts.start > first) throw new StyleError(`Run command start value ${opts.start} is after start of run at step ${first}`);
  if (opts.stop !== null && opts.stop < first + n) throw new StyleError(`Run command stop value ${opts.stop} is before end of run at step ${first + n}`);
  const lastThermo = new Map<System, number>();
  for (const r of group) {
    ensureThermoComputes(r);
    // the run's steps before init, as in a single run
    r.run = {
      inRun: true, firstStep: first, lastStep: first + n,
      beginStep: opts.start ?? first, endStep: opts.stop ?? first + n, t0: performance.now(), ranOnce: true, eflagStep: r.run.eflagStep,
    };
    initRun(r, { verlet: true });
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
      if (th.everyVar && step >= nextThermoVar.get(r)!) { due = true; nextThermoVar.set(r, th.nextVariableStep(true)); }
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

/**
 * The checks a run makes of its run style and kspace style before it starts. As native LAMMPS does (VerletSplit::init and
 * VerletSplitRK::init in src/REPLICA/verlet_split.cpp and verlet_split_rk.cpp, Integrate::init in src/integrate.cpp and
 * PPPM_RK::setupRKBlock in src/KSPACE/pppm_rk.cpp at b332f28), in that order. A verlet/split run without a kspace style
 * is refused here (native warns and then stops with a segmentation fault).
 */
const checkSplitRun = (sys: System): void => {
  const k = sys.ff.kspace;
  const unsupportedSolver = (kk: KSpace): boolean => /tip4p|spin/.test(kk.name) || kk.dipole();
  if (sys.splitRun === 'verlet/split') {
    if (!k) throw new StyleError('A KSpace style must be defined with verlet/split');
    if (!hasCharge(sys.state) || unsupportedSolver(k)) throw new StyleError(`Verlet/split cannot (yet) be used with kpace style ${k.name}`);
  } else if (sys.splitRun === 'verlet/split/rk') {
    if (!k) throw new StyleError('A KSpace style must be defined with verlet/split/rk');
    if (!k.rkFlag) throw new StyleError(`The kspace style ${k.name} does not support an rk decomposition. Verlet/split/rk requires a kspace style (such as pppm/rk) that supports an rk decomposition.`);
    if (unsupportedSolver(k)) throw new StyleError(`Verlet/split/rk cannot (yet) be used with kspace style ${k.name}`);
  }
  if (k && k.computeOn && k.rkFlag && sys.splitRun !== 'verlet/split/rk') {
    throw new StyleError(`The Integrate type must support an rk decomposition (e.g., run_style verlet/split/rk) that is compatible with kspace style ${k.name}`);
  }
  if (k?.rkFlag && sys.vars.nworlds !== 2) throw new StyleError('pppm/rk requires 2 partitions');
};

/**
 * The run of the Kspace partition (partition 2) of run_style verlet/split or verlet/split/rk. As native LAMMPS does
 * (VerletSplit::setup and run in src/REPLICA/verlet_split.cpp at b332f28): that partition integrates nothing, invokes
 * no fix, and writes no thermo, dump or restart output; each step it computes the kspace forces of the Rspace
 * partition's atoms (their coordinates and charges, gathered in the Rspace partition's order) and returns them, and its
 * step count advances with the run. Partition 1 runs first here and computes its own kspace forces, which gives native's
 * forces: measured with native LAMMPS, a 64-ion pppm run prints the same thermo with run_style verlet/split on two
 * partitions as with run_style verlet, to all 15 printed digits (pppm/rk with verlet/split/rk too). After the run,
 * partition 2 holds the Rspace partition's coordinates and charges and keeps its own velocities and image flags
 * (measured with native LAMMPS for verlet/split). Native copies them in the Rspace partition's local order, which its
 * spatial sort at the run's setup has changed (measured with native LAMMPS: atom 53 of partition 2 then has another
 * atom's coordinates), and after verlet/split/rk partition 2's coordinates are not meaningful (measured: atom 1 at 0);
 * here each atom takes the coordinates and charge of the atom with its ID.
 */
const runSplitKspace = (sys: System, n: number): number => {
  const rspace = sys.replicaPeers()[0];
  const s = sys.state, r = rspace.state;
  if (r.n !== s.n) {
    throw new StyleError(`run_style ${sys.splitRun}: the Rspace partition has ${r.n} atoms and the Kspace partition ${s.n}; the browser engine keeps the two partitions' atoms one to one`);
  }
  for (let i = 0; i < s.n; i++) {
    const j = rspace.indexOfId(s.id[i]);
    if (j < 0) throw new StyleError(`run_style ${sys.splitRun}: atom ${s.id[i]} of the Kspace partition is not in the Rspace partition`);
    s.x[3 * i] = r.x[3 * j]; s.x[3 * i + 1] = r.x[3 * j + 1]; s.x[3 * i + 2] = r.x[3 * j + 2];
    s.q[i] = r.q[j];
  }
  s.step += n;
  sys.bump();
  return n;
};

/** One run of n steps with output; returns steps taken. */
const runSteps = async (ctx: Ctx, n: number, opts: RunOpts): Promise<number> => {
  const { sys, session } = ctx;
  // fix_halt.html: after a soft halt "additional run or minimize commands will be skipped" (no output)
  if (softHalted(sys)) return 0;
  clearHaltRequest(sys);
  checkSplitRun(sys);
  if (sys.splitRun && sys.vars.partition === 1) return runSplitKspace(sys, n);
  if (session.coupledSystems) return runCoupledSteps(ctx, n, opts);
  const s = sys.state;
  const first = s.step;
  // run.html: start and stop are checked before anything changes, so a rejected run leaves the system as it was
  // (measured with native LAMMPS, black box: the checks come before the setup)
  if (opts.start !== null && opts.start > first) throw new StyleError(`Run command start value ${opts.start} is after start of run at step ${first}`);
  if (opts.stop !== null && opts.stop < first + n) throw new StyleError(`Run command stop value ${opts.stop} is before end of run at step ${first + n}`);
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
  // As native LAMMPS does (src/run.cpp command at b332f28): the run's first, last, start and stop steps are set before
  // init, which fixes read (fix deform: the box rate over the run, 0 for a run of no steps).
  sys.run = {
    inRun: true, firstStep: first, lastStep: first + n,
    beginStep: opts.start ?? first, endStep: opts.stop ?? first + n, t0: performance.now(), ranOnce: true, eflagStep: sys.run.eflagStep,
  };
  if (!reuse) initRun(sys, { verlet: !sys.respa });
  if (reuse) sys.forcesCurrent();
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
    if (th.everyVar && step >= nextThermoVar) { due = true; nextThermoVar = th.nextVariableStep(true); }
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
  // As native LAMMPS does (src/run.cpp Run::command at b332f28): N is read first, then the keywords in order, then N
  // (or the upto value), start and stop are checked
  if (a.length < 1) throw missingArgs('run');
  if (!sys.hasBox) throw new StyleError('Run command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  const s = sys.state;
  const nIn = bnumeric(a[0]);
  let upto = false;
  const opts: RunOpts = { start: null, stop: null, pre: true, every: 0, commands: [] };
  for (let k = 1; k < a.length;) {
    const key = a[k];
    switch (key) {
      case 'upto': upto = true; k++; break;
      case 'start': if (k + 2 > a.length) throw missingArgs('run start'); opts.start = bnumeric(a[k + 1]); k += 2; break;
      case 'stop': if (k + 2 > a.length) throw missingArgs('run stop'); opts.stop = bnumeric(a[k + 1]); k += 2; break;
      case 'pre': case 'post': {
        if (k + 2 > a.length) throw missingArgs(`run ${key}`);
        // run.html: "If post is specified as no, the full timing summary is skipped; only a one-line summary timing is
        // printed." The engine prints only the one-line summary, so post is accepted and has no further effect.
        const on = logical(a[k + 1]);
        if (key === 'pre') opts.pre = on;
        k += 2;
        break;
      }
      case 'every':
        // "every M c1 c2 ...": all remaining words are the commands, a single NULL meaning none
        if (k + 3 > a.length) throw missingArgs('run every');
        opts.every = inumeric(a[k + 1]);
        if (opts.every <= 0) throw new StyleError(`Invalid run every argument: ${opts.every}`);
        opts.commands = a.slice(k + 2);
        k = a.length;
        if (opts.commands.length === 1 && opts.commands[0] === 'NULL') opts.commands = [];
        break;
      default: throw new StyleError(`Unknown run keyword: ${key}`);
    }
  }
  const MAXSMALLINT = 0x7fffffff;
  let n = nIn;
  if (!upto) {
    if (nIn < 0 || nIn > MAXSMALLINT) throw new StyleError(`Invalid run command N value: ${nIn}`);
  } else {
    n = nIn - s.step;
    if (n < 0 || n > MAXSMALLINT) throw new StyleError(`Invalid run command upto value: ${nIn}. Must be >= ${s.step}`);
  }
  if (opts.start !== null) {
    if (opts.start < 0) throw new StyleError(`Invalid run command start value: ${opts.start}`);
    if (opts.start > s.step) throw new StyleError(`Run command start value ${opts.start} is after start of run at step ${s.step}`);
  }
  if (opts.stop !== null) {
    if (opts.stop < 0) throw new StyleError(`Invalid run command stop value: ${opts.stop}`);
    if (opts.stop < s.step + n) throw new StyleError(`Run command stop value ${opts.stop} is before end of run at step ${s.step + n}`);
  }
  if (!opts.pre && sys.respa) throw new StyleError("Run flag 'pre no' not compatible with r-RESPA");
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
  // native LAMMPS splits the pppm/rk solve between the minimizations of the two partitions (src/KSPACE/pppm_rk.cpp at
  // b332f28, PPPM_RK::compute); measured with native LAMMPS: partition 1 then minimizes as with pppm, and partition 2
  // with a kspace energy of its own (0.0128 where pppm gives -2031.9). The browser engine refuses it by name.
  if (sys.ff.kspace?.rkFlag) throw new StyleError('minimize with kspace_style pppm/rk is not supported by the browser engine (native splits the kspace solve between the two partitions\' minimizations); use kspace_style pppm');
  ensureThermoComputes(sys);
  const s = sys.state;
  const th = sys.thermo;
  // the minimization's steps before init, as native Minimize::command (src/minimize.cpp at b332f28) sets them
  sys.run = { inRun: true, firstStep: s.step, lastStep: s.step + maxiter, beginStep: s.step, endStep: s.step + maxiter, t0: performance.now(), ranOnce: true, eflagStep: sys.run.eflagStep };
  initRun(sys);
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

/**
 * A new minimizer of a style, as min_style makes one. As native LAMMPS does (src/update.cpp Update::create_minimize at
 * b332f28): the new minimizer's constructor (src/min.cpp Min::Min) sets every min_modify setting back to its default.
 * Measured with native LAMMPS: min_modify dmax 0.01 followed by min_style quickmin minimizes as with the defaults.
 * The tad command makes its quench and NEB minimizers the same way (src/REPLICA/tad.cpp).
 */
export const createMinimize = (sys: System, style: string): void => {
  if (!Object.prototype.hasOwnProperty.call(MINIMIZERS, style) && !(MIN_STYLES as readonly string[]).includes(style)) {
    throw new StyleError(`Unknown minimize style ${style}`);
  }
  sys.minStyle = style;
  sys.minSettings = defaultMinSettings();
};

/** min_style cg|sd|fire|quickmin|hftn — min_style.html. */
const minStyle: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/input.cpp Input::min_style and src/update.cpp Update::create_minimize at b332f28)
  if (!sys.hasBox) throw new StyleError(`Min_style command before simulation box is defined${errorUrl(33)}`);
  if (a.length < 1) throw missingArgs('minimize_style');
  const known = a[0] !== undefined && (Object.prototype.hasOwnProperty.call(MINIMIZERS, a[0]) || (MIN_STYLES as readonly string[]).includes(a[0]));
  if (!known) {
    return loadNativeStyles().then(() => {
      throw unknownStyleError('minimize', a[0] ?? '', () => removedStyleError('min_style', a[0] ?? '') ?? new StyleError(`min_style '${a[0] ?? ''}' is not supported by the browser engine; supported: ${MIN_STYLES.join(', ')}`));
    });
  }
  createMinimize(sys, a[0]);
};

/**
 * min_modify keyword value ... — min_modify.html. As native LAMMPS does (src/min.cpp Min::modify_params at b332f28): a
 * missing value or an unknown line, norm or integrator name stops with Illegal min_modify command; a keyword min.cpp
 * does not read goes to the minimizer's own modify_param (alpha_damp and discrete_factor of min_style spin,
 * discrete_factor of spin/cg and spin/lbfgs; src/SPIN/min_spin*.cpp), and one it does not take stops with native's
 * Illegal fix_modify command.
 */
const minModify: Handler = ({ sys }, a) => {
  if (a.length === 0) throw new StyleError('Illegal min_modify command');
  const own: Record<string, string[]> = { spin: ['alpha_damp', 'discrete_factor'], 'spin/cg': ['discrete_factor'], 'spin/lbfgs': ['discrete_factor'] };
  for (let k = 0; k < a.length;) {
    const key = a[k];
    const v = a[k + 1];
    const minKey = MIN_MODIFY_KEYS.has(key);
    if (!minKey && !(own[sys.minStyle] ?? []).includes(key)) throw new StyleError('Illegal fix_modify command');
    // a missing value: Min::modify_params and the spin minimizers say "Illegal min_modify command", except min_style
    // spin/cg's own discrete_factor (src/SPIN/min_spin_cg.cpp modify_param at b332f28: "Illegal fix_modify command")
    if (v === undefined) throw new StyleError(!minKey && sys.minStyle === 'spin/cg' ? 'Illegal fix_modify command' : 'Illegal min_modify command');
    switch (key) {
      case 'dmax': sys.minSettings.dmax = numeric(v); k += 2; break;
      case 'line':
        // min_modify.html: spin_cubic and spin_none are line searches of the spin minimizers (min_style spin/cg, run/
        // minimizers.ts); native accepts them for every style, so they are kept in the minimizer settings and refused
        // by the other minimizers in minimize()
        if (v === 'spin_cubic' || v === 'spin_none') {
          sys.minSettings.fire.line = v;
          k += 2;
          break;
        }
        if (!['backtrack', 'quadratic', 'forcezero'].includes(v)) throw new StyleError('Illegal min_modify command');
        sys.minSettings.line = v as 'backtrack' | 'quadratic' | 'forcezero';
        k += 2;
        break;
      case 'norm':
        if (!['two', 'max', 'inf'].includes(v)) throw new StyleError('Illegal min_modify command');
        sys.minSettings.norm = v as 'two' | 'max' | 'inf';
        k += 2;
        break;
      default: {
        // each value is read when given: an integer (delaystep, vdfmax), a boolean (halfstepback, initialdelay,
        // abcfire), an integrator name or a number
        if (key === 'delaystep' || key === 'vdfmax') inumeric(v);
        else if (key === 'halfstepback' || key === 'initialdelay' || key === 'abcfire') logical(v);
        else if (key === 'integrator') {
          if (!['eulerimplicit', 'verlet', 'leapfrog', 'eulerexplicit'].includes(v)) throw new StyleError('Illegal min_modify command');
        } else numeric(v);
        sys.minSettings.fire[key] = v;
        k += 2;
        break;
      }
    }
  }
};

/** The keywords native Min::modify_params reads itself (src/min.cpp at b332f28). */
const MIN_MODIFY_KEYS = new Set(['dmax', 'delaystep', 'dtgrow', 'dtshrink', 'alpha0', 'alphashrink', 'tmax', 'tmin', 'halfstepback',
  'initialdelay', 'vdfmax', 'integrator', 'abcfire', 'line', 'norm']);

export const RUN_COMMANDS: Record<string, Handler> = {
  fix, unfix, fix_modify: fixModify, compute, uncompute, compute_modify: computeModify,
  thermo, thermo_style: thermoStyle, thermo_modify: thermoModify, run_style: runStyle,
  dump, undump, dump_modify: dumpModify, write_dump: writeDump,
  run, minimize: minimizeCmd, min_style: minStyle, min_modify: minModify,
};
