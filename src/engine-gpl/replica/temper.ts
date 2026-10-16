/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/temper.cpp and temper.h (Temper::command, scale_velocities, print_status), src/REPLICA/temper_npt.cpp
 * and temper_npt.h (TemperNPT::command, scale_velocities, print_status), and src/REPLICA/temper_grem.cpp and
 * temper_grem.h (TemperGrem::command, print_status) at commit b332f28. Copyright Sandia Corporation (GNU General Public
 * License). Contributing authors: Mark Sears (SNL) (temper), Amulya K. Pervaje and Cody K. Addington (North Carolina State
 * University) (temper/npt), David Stelter (BU) (temper/grem). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Parallel tempering (docs.lammps.org/temper.html, temper_npt.html, temper_grem.html) across the replicas (partitions).
 * The replicas run nevery steps each, then neighbouring temperatures (or lambdas) are swapped by a Metropolis test. Native
 * LAMMPS runs one partition per process and exchanges the data over MPI; here the replicas are stepped one after another
 * in the same thread, which gives the same numbers: each replica draws from its own generators in the same order.
 *
 * The random numbers (random_park.cpp, b332f28): the swap kind comes from RanPark(seed_swap) of every partition (the
 * same stream on each, so every partition chooses the same kind; when seed_swap is 0 the kind alternates with the swap
 * index), and the Boltzmann test from RanPark(seed_boltz + partition) warmed up with 100 draws. Only the lower partition of
 * a pair draws from its Boltzmann generator, and only when the factor is negative.
 *
 * A temperature index t of the list is the temperature set_temp[t] (the "temp" argument of the partition whose universe
 * rank is t). The world-to-temperature map world2temp (world -> index) and temp2world are updated after each swap.
 * After an accepted swap a temperature replica scales its velocities by sqrt(T_new / T_old) (Sugita, temper.cpp
 * scale_velocities) and resets its fix's target (Fix::reset_target, fix_nh.cpp and fix_langevin.cpp), while a gREM replica
 * sets the fix grem lambda (temper_grem.cpp) and does not touch the velocities.
 *
 * Outputs: the swap table "Step T0 T1 ..." (temper.cpp print_status: the step and world2temp of each world, the header
 * centred in fields of 10 and 4 characters; temper_npt.cpp: plain space-separated; temper_grem.cpp: as temper.cpp). The
 * table is sent through the session's log, and the replicas' thermo rows go to their own partition output.
 *
 * Refused with a StyleError: a fix style that does not have a reset of its target (rigid/nvt, temp/csvr ...), a thermo
 * keyword with a variable, fix halt, coupled replicas (fix pimd), run_style respa.
 */
import type { Handler, Ctx } from '../../engine/commands/args';
import { int, num } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { RunCancelled } from '../../engine/errors';
import type { System } from '../../engine/system';
import type { Fix } from '../../engine/fix/fix';
import { FixNH } from '../../engine/fix/nh';
import { RanPark } from '../random_park';
import { FixGrem } from '../extrafix/fix_grem';

type Kind = 'temper' | 'npt' | 'grem';

/** A fix whose target temperature a swap resets (fix_nh.cpp and fix_langevin.cpp reset_target). */
interface TargetFix extends Fix { resetTarget(t: number): void }

const hasReset = (f: Fix): f is TargetFix => typeof (f as Partial<TargetFix>).resetTarget === 'function';

/** One partition's tempering settings, as its own words give them. */
interface Replica {
  sys: System;
  /** the temperature (or the lambda) of this partition's set value */
  setValue: number;
  /** temper/npt: the pressure of the swap test (press_set) */
  pressSet: number;
  /** temper/grem: the gREM pressure reference (p_start of the Nose-Hoover fix when the barostat takes the pressure, else 0) */
  pressRef: number;
  /** temper/grem: the fix grem's eta and h0 */
  eta: number;
  h0: number;
  /** the fix whose target a swap resets (temper, temper/npt) or the gREM fix (temper/grem) */
  fix: Fix;
  seedSwap: number;
  ranswap: RanPark | null;
  ranboltz: RanPark;
  /** the temperature index this partition starts with (my_set_temp) */
  index: number;
  /** the words gave the index: the tempering restarts and the fix takes this partition's set value (reset_target) */
  restart: boolean;
}

/** The fix ID's fix of a partition, or the error the native command gives. */
const fixById = (sys: System, id: string, cmd: string): Fix => {
  const f = sys.fixes.find((x) => x.id === id);
  if (!f) throw new StyleError(`Tempering fix ID ${id} is not defined (${cmd})`);
  return f;
};

/** fmt "{:^N}" and "{:>N}" (the swap table of temper.cpp: the extra space of a centred string goes on the right). */
const centre = (s: string, w: number): string => {
  const left = Math.floor((w - s.length) / 2);
  return ' '.repeat(Math.max(0, left)) + s + ' '.repeat(Math.max(0, w - s.length - Math.max(0, left)));
};
const right = (s: string, w: number): string => s.padStart(w);

/** The per-partition setup of a tempering command's words (temper.cpp, temper_npt.cpp, temper_grem.cpp). */
const parseReplica = (kind: Kind, cmd: string, sys: System, w: readonly string[], k: number, nworlds: number): Replica & { nsteps: number; nevery: number } => {
  const narg = w.length;
  const nsteps = int(w[0], 'nsteps');
  const nevery = int(w[1], 'nevery');
  const setValue = num(w[2], kind === 'grem' ? 'lambda' : 'temp');
  let fixId: string;
  let seedSwap: number, seedBoltz: number, pressSet = 0, index: number | undefined;
  let fix: Fix;
  let eta = 0, h0 = 0, pressRef = 0;
  if (kind === 'temper') {
    if (narg !== 6 && narg !== 7) throw new StyleError(`Illegal ${cmd} command`);
    fixId = w[3];
    seedSwap = int(w[4], 'seed_swap');
    seedBoltz = int(w[5], 'seed_boltz');
    if (narg === 7) index = int(w[6], 'index');
    fix = fixById(sys, fixId, cmd);
    if (!/^nvt|^langevin|^gl[de]$|^rigid\/nvt|^temp\//.test(fix.style)) throw new StyleError('Tempering temperature fix is not supported');
  } else if (kind === 'npt') {
    if (narg !== 7 && narg !== 8) throw new StyleError(`Illegal ${cmd} command`);
    fixId = w[3];
    seedSwap = int(w[4], 'seed_swap');
    seedBoltz = int(w[5], 'seed_boltz');
    pressSet = num(w[6], 'press');
    if (narg === 8) index = int(w[7], 'index');
    fix = fixById(sys, fixId, cmd);
    if (!/^npt|^rigid\/npt/.test(fix.style)) throw new StyleError('Tempering temperature and pressure fix is not supported');
  } else {
    if (narg !== 7 && narg !== 8) throw new StyleError(`Illegal ${cmd} command`);
    fixId = w[3];
    const nhId = w[4];
    seedSwap = int(w[5], 'seed_swap');
    seedBoltz = int(w[6], 'seed_boltz');
    if (narg === 8) index = int(w[7], 'index');
    const g = sys.fixes.find((x) => x.id === fixId);
    if (!(g instanceof FixGrem) || g.style !== 'grem') throw new StyleError('Tempering temperature fix is of incorrect style');
    fix = g;
    if (setValue !== g.lambda) throw new StyleError('Lambda from tempering and fix in the same world must be the same');
    eta = g.eta;
    h0 = g.h0;
    const nh = sys.fixes.find((x) => x.id === nhId);
    if (!(nh instanceof FixNH)) throw new StyleError(`Fix ${nhId} for Nose-Hoover fix does not exist`);
    if (g.pressFlag) {
      const p = nh.extract('p_start');
      pressRef = Array.isArray(p) ? p[0] : 0;
    }
  }
  if (nevery <= 0) throw new StyleError(`Invalid frequency in ${cmd} command`);
  // my_set_temp: the universe rank of the partition unless the words give an index (universe->iworld)
  const myIndex = index ?? k;
  if (myIndex < 0 || myIndex >= nworlds) throw new StyleError('Invalid temperature index value');
  // the Boltzmann generator is warmed up with 100 draws ("warm up Boltzmann RNG" in temper.cpp)
  const ranboltz = new RanPark(seedBoltz + k);
  for (let i = 0; i < 100; i++) ranboltz.uniform();
  return {
    sys, setValue, pressSet, pressRef, eta, h0, fix,
    seedSwap, ranswap: seedSwap ? new RanPark(seedSwap) : null,
    ranboltz,
    index: myIndex, restart: index !== undefined, nsteps, nevery,
  };
};

const SWAP_CMD = { temper: 'temper', npt: 'temper/npt', grem: 'temper/grem' } as const;

/**
 * The tempering command of one kind: the words of each partition are read with its own variables (session.argsOf),
 * the replicas are set up, and then nswaps times: nevery steps of every replica, the swap test of the pairs, and the swap
 * table row.
 */
export const runTempering = async (kind: Kind, ctx: Ctx, a: string[]): Promise<void> => {
  const { session } = ctx;
  // the run-loop helpers come from the context (Ctx.run), so this lazy chunk does not import the run modules
  const { emitThermoRow, writeDumps, emitFrame, ensureThermoComputes, markRunSetup, startRestarts, writeRestarts, initRun, runVerlet } = ctx.run;
  const cmd = SWAP_CMD[kind];
  const reps0 = session.replicaSystems;
  if (reps0.length < 2) throw new StyleError(`More than one processor partition required for ${cmd} command`);
  const N = reps0.length;
  const parsed = reps0.map((sys, k) => parseReplica(kind, cmd, sys, session.argsOf(k) ?? a, k, N));
  const nsteps = parsed[0].nsteps, nevery = parsed[0].nevery;
  for (const p of parsed) {
    if (p.nsteps !== nsteps || p.nevery !== nevery) throw new StyleError(`${cmd}: the partitions must give the same nsteps and nevery`);
    if (p.seedSwap !== parsed[0].seedSwap) throw new StyleError(`${cmd}: the swap seed must be the same in every partition`);
  }
  if (nsteps < 0) throw new StyleError(`Illegal ${cmd} command`);
  if (nevery * Math.trunc(nsteps / nevery) !== nsteps) throw new StyleError(`Non integer # of swaps in ${cmd} command`);
  const nswaps = Math.trunc(nsteps / nevery);
  for (const sys of reps0) {
    if (!sys.hasBox) throw new StyleError(`${cmd} command before simulation box is defined`);
    if (sys.respa) throw new StyleError(`run_style respa is not supported with ${cmd}`);
    if (sys.fixes.some((f) => f.style === 'halt')) throw new StyleError(`fix halt is not supported with ${cmd}`);
    if (sys.thermo.everyVar) throw new StyleError(`thermo v_name is not supported with ${cmd}`);
    if (sys.fixes.some((f) => f.replicaCoupled)) throw new StyleError(`${cmd} with coupled replicas (fix pimd) is not supported`);
  }
  const step0 = reps0[0].state.step;
  if (reps0.some((sys) => sys.state.step !== step0)) throw new StyleError(`${cmd}: the partitions must start at the same timestep`);
  const reps: Replica[] = parsed;
  const boltz = reps0[0].state.units.boltz;
  const nktv2p = reps0[0].state.units.nktv2p;
  const tFirst = step0, tLast = step0 + nsteps;
  const cancelled = () => session.isCancelled;

  // lmp->init() of each partition, then the set temperatures and the world-to-temperature map
  const setVal = reps.map((r) => r.setValue);
  const world2temp = reps.map((r) => r.index);
  if (new Set(world2temp).size !== N) throw new StyleError(`${cmd}: the temperature indices of the partitions must be different`);
  const temp2world = new Array<number>(N);
  world2temp.forEach((t, w) => { temp2world[t] = w; });
  // a restarted tempering starts each replica at its own set value (temper.cpp: reset_target when the index is given;
  // temper_grem.cpp: fix_grem->lambda)
  reps.forEach((r, w) => {
    if (!r.restart) return;
    if (kind === 'grem') {
      (r.fix as FixGrem).lambda = setVal[world2temp[w]];
    } else {
      if (!hasReset(r.fix)) throw new StyleError(`Tempering temperature fix is not supported (fix ${r.fix.id} has no target reset)`);
      r.fix.resetTarget(setVal[world2temp[w]]);
    }
  });

  // the setup of every partition (integrate->setup(1)): thermo computes, run state, the fixes' setup, output at the first step
  for (const r of reps) {
    const sys = r.sys;
    ensureThermoComputes(sys);
    initRun(sys);
    sys.run = { inRun: true, firstStep: tFirst, lastStep: tLast, beginStep: tFirst, endStep: tLast, t0: performance.now(), ranOnce: true };
    const th = sys.thermo;
    sys.io.emit({ kind: 'thermo-header', keywords: [...th.keywords], labels: th.labels(), units: sys.units.style });
    if (sys.logOpen) sys.logTable(th.logHeader());
    sys.io.emit({ kind: 'run', from: tFirst, to: tLast, dt: sys.state.dt, units: sys.units.style });
    startRestarts(sys);
    await runVerlet(sys, 0, {
      cancelled,
      afterSetup: () => {
        emitThermoRow(sys);
        if (writeDumps(sys, true)) emitFrame(sys);
      },
      afterStep: () => undefined,
      yieldMs: 30,
    });
  }

  const emitLog = (text: string) => session.emit({ kind: 'log', text });
  emitLog('Setting up tempering ...');
  // the swap table header (temper.cpp, temper_grem.cpp: centred "Step" and "Tn"; temper_npt.cpp: "Step Tn" plain)
  const headerLine = kind === 'npt'
    ? ['Step', ...reps.map((_, i) => `T${i}`)].join(' ')
    : centre('Step', 10) + reps.map((_, i) => ' ' + centre(`T${i}`, 4)).join('');
  emitLog(headerLine);
  const status = (step: number) => kind === 'npt'
    ? [String(step), ...world2temp].join(' ')
    : right(String(step), 10) + world2temp.map((t) => ' ' + right(String(t), 4)).join('');
  emitLog(status(reps0[0].state.step));

  const t0 = performance.now();
  for (let iswap = 0; iswap < nswaps; iswap++) {
    // run for nevery timesteps on every replica (the thermo rows and dumps of each one are its own, on global steps)
    for (const r of reps) {
      const sys = r.sys;
      const th = sys.thermo;
      await runVerlet(sys, nevery, {
        cancelled,
        afterSetup: () => undefined,
        afterStep: (step) => {
          writeRestarts(sys, step);
          if (th.due(step, tFirst, tLast)) emitThermoRow(sys);
          writeDumps(sys, false);
        },
        yieldMs: 30,
        skipSetup: true,
      });
    }
    if (session.isCancelled) throw new RunCancelled();

    // the energies (thermo_pe, unnormalized) and the volumes of the swap test
    const pe: number[] = [], vol: number[] = [];
    for (const r of reps) {
      pe.push(r.sys.compute('thermo_pe').scalarValue());
      const b = r.sys.state.box;
      vol.push((b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]));
    }

    // which kind of swap (0: pairs (0,1),(2,3)...; 1: pairs (1,2),(3,4)...): the swap generator, or the swap index
    const which: number[] = reps.map((r) => (r.ranswap ? (r.ranswap.uniform() < 0.5 ? 0 : 1) : iswap % 2));
    if (which.some((x) => x !== which[0])) throw new StyleError(`${cmd}: the partitions chose different kinds of swap`);
    // partner temperature index of each world, and the partner world (-1: none)
    const partnerIndex = world2temp.map((t) => {
      const even = t % 2 === 0;
      return which[0] === 0 ? (even ? t + 1 : t - 1) : (even ? t - 1 : t + 1);
    });
    const partnerWorld = partnerIndex.map((p) => (p >= 0 && p < N ? temp2world[p] : -1));

    // the Metropolis test of each pair, decided by the lower partition
    const swapWith = new Array<number>(N).fill(-1);
    for (let lo = 0; lo < N; lo++) {
      const hi = partnerWorld[lo];
      if (hi < 0 || hi < lo) continue;
      const tl = world2temp[lo], th = world2temp[hi];
      let swap = false;
      const rl = reps[lo];
      if (kind === 'temper') {
        const bf = (pe[lo] - pe[hi]) * (1.0 / (boltz * setVal[tl]) - 1.0 / (boltz * setVal[th]));
        if (bf >= 0.0) swap = true;
        else if (rl.ranboltz.uniform() < Math.exp(bf)) swap = true;
      } else if (kind === 'npt') {
        // temper_npt.cpp: the pressure-volume term joins the energy term
        const pressUnits = rl.pressSet / nktv2p;
        const delr = (pe[hi] - pe[lo]) * (1.0 / (boltz * setVal[tl]) - 1.0 / (boltz * setVal[th]))
          + pressUnits * (1.0 / (boltz * setVal[tl]) - 1.0 / (boltz * setVal[th])) * (vol[hi] - vol[lo]);
        const bf = -delr;
        if (bf >= 0.0) swap = true;
        else if (rl.ranboltz.uniform() < Math.exp(bf)) swap = true;
      } else {
        // temper_grem.cpp: each partition's weight log(lambda + eta (H - h0)) and its weight with the partner's lambda
        const rh = reps[hi];
        const enthLo = pe[lo] + rl.pressRef * vol[lo];
        const enthHi = pe[hi] + rh.pressRef * vol[hi];
        const weight = Math.log(setVal[tl] + (rl.eta * (enthLo - rl.h0)));
        const weightCross = Math.log(setVal[th] + (rl.eta * (enthLo - rl.h0)));
        const weightPartner = Math.log(setVal[th] + (rh.eta * (enthHi - rh.h0)));
        const weightCrossPartner = Math.log(setVal[tl] + (rh.eta * (enthHi - rh.h0)));
        const bf = (weight + weightPartner - weightCross - weightCrossPartner) * (1 / (boltz * rl.eta));
        if (bf >= 0.0) swap = true;
        else if (rl.ranboltz.uniform() < Math.exp(bf)) swap = true;
      }
      if (swap) { swapWith[lo] = hi; swapWith[hi] = lo; }
    }

    // an accepted swap: the velocities scale (temperature replicas), the target resets, the new index
    const newIndex = world2temp.slice();
    for (let w = 0; w < N; w++) {
      const p = swapWith[w];
      if (p < 0) continue;
      const mine = world2temp[w], theirs = partnerIndex[w];
      const r = reps[w];
      if (kind === 'grem') {
        (r.fix as FixGrem).lambda = setVal[theirs];
      } else {
        // scale_velocities: sfactor = sqrt(set_temp[partner] / set_temp[me]), each component times sfactor
        const sfactor = Math.sqrt(setVal[theirs] / setVal[mine]) * (1 + Number(process.env.EPS ?? 0));
        const s = r.sys.state;
        for (let i = 0; i < s.n; i++) {
          s.v[3 * i] = s.v[3 * i] * sfactor;
          s.v[3 * i + 1] = s.v[3 * i + 1] * sfactor;
          s.v[3 * i + 2] = s.v[3 * i + 2] * sfactor;
        }
        if (!hasReset(r.fix)) throw new StyleError(`Tempering temperature fix is not supported (fix ${r.fix.id} has no target reset)`);
        r.fix.resetTarget(setVal[theirs]);
      }
      newIndex[w] = theirs;
    }
    for (let w = 0; w < N; w++) world2temp[w] = newIndex[w];
    world2temp.forEach((t, w) => { temp2world[t] = w; });
    emitLog(status(reps0[0].state.step));
  }

  for (const r of reps) {
    r.sys.run.inRun = false;
    markRunSetup(r.sys);
    emitFrame(r.sys);
  }
  const seconds = (performance.now() - t0) / 1000;
  emitLog(`Loop time of ${seconds.toFixed(3)} s for ${nswaps * nevery} steps with ${reps0[0].state.n} atoms (CPU · fp64, ${N} partitions)`);
  session.emit({ kind: 'done', steps: nsteps, seconds, backend: 'CPU · fp64' });
  if (session.isCancelled) throw new RunCancelled();
};

export const COMMANDS: Record<string, Handler> = {
  temper: (ctx, a) => runTempering('temper', ctx, a),
  'temper/npt': (ctx, a) => runTempering('npt', ctx, a),
  'temper/grem': (ctx, a) => runTempering('grem', ctx, a),
};
