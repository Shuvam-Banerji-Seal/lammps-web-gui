/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/prd.cpp (PRD::dephase, PRD::dynamics, PRD::quench, PRD::replicate), src/REPLICA/tad.cpp
 * (TAD::dynamics, TAD::quench, TAD::store_state, TAD::revert_state), src/REPLICA/fix_event.cpp (FixEvent::store_event,
 * restore_event, store_state_quench, restore_state_quench, store_state_dephase, restore_state_dephase), and the
 * velocity create of src/velocity.cpp (loop all and loop geom, mom yes) used by PRD::dephase, at commit b332f28.
 * Copyright Sandia Corporation (GNU General Public License). Contributing author: Mike Brown (SNL) (prd, tad,
 * fix event). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The shared helpers of the accelerated dynamics commands prd and tad (their logic is in prd.ts and tad.ts): the
 * state snapshots of fix event, the short dynamics run and the quench (minimize) with the thermo rows that native
 * LAMMPS prints for them, and the velocity create that PRD's dephasing uses.
 */
import type { RunApi } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';

/** The mass of atom i (atoms.ts massOf: the per-atom mass, else the type's). */
const massOf = (s: SimState, i: number): number => (s.rmass ? s.rmass[i] : s.massByType[s.type[i]]);
import { RanPark } from '../random_park';
import { nativeOrder } from '../../engine/atoms';

/** The per-atom state a fix event stores (FixEvent: x, v and the image flags). */
export interface Snap { x: Float64Array; v: Float64Array; image: Int32Array }

export const snapOf = (sys: System): Snap => {
  const s = sys.state;
  return { x: s.x.slice(0, 3 * s.n), v: s.v.slice(0, 3 * s.n), image: s.image.slice(0, 3 * s.n) };
};

/** FixEvent::restore_state_quench and friends: the stored positions, velocities and image flags come back. */
export const restoreSnap = (sys: System, k: Snap): void => {
  const s = sys.state;
  s.x.set(k.x);
  s.v.set(k.v);
  s.image.set(k.image);
  // the forces stay those of the last evaluation (FixEvent restores only x, v and the image flags; native writes them
  // as they were last computed), so the forces are not recomputed here
  sys.bump();
  sys.forcesCurrent();
};

/**
 * Copies positions and image flags of replica src to dst (PRD::replicate, one processor per replica: the atoms
 * are matched by index, which is the storage order of replicas that do not sort their atoms, atom_modify sort 0).
 */
export const copyPositions = (dst: System, src: System): void => {
  const d = dst.state, o = src.state;
  if (d.n !== o.n) throw new StyleError('prd/tad: the replicas must have the same number of atoms');
  d.x.set(o.x.subarray(0, 3 * o.n));
  d.image.set(o.image.subarray(0, 3 * o.n));
  dst.bump();
  dst.forcesCurrent();
};

export const emitHeader = (sys: System): void => {
  const th = sys.thermo;
  sys.io.emit({ kind: 'thermo-header', keywords: [...th.keywords], labels: th.labels(), units: sys.units.style });
};

/** laststep: the last step of the command (update->laststep, which native's thermo output tests as well as the interval) */
export interface DynamicsOpts { cancelled(): boolean; laststep: number }

/**
 * dynamics(nsteps): integrate->setup(1) and integrate->run(nsteps) on one replica, with lmp->init() first. Native
 * prints the setup row and the rows of the thermo interval (its last step included) for every run.
 */
export const dynamics = async (run: RunApi, sys: System, nsteps: number, o: DynamicsOpts): Promise<void> => {
  run.initRun(sys);
  emitHeader(sys);
  // a dynamics segment prints its setup row and the thermo interval; its last step is printed only when it is the
  // command's last step (native: the segment's laststep is the command's, see prd.cpp PRD::dynamics)
  await run.runVerlet(sys, nsteps, {
    cancelled: o.cancelled,
    afterSetup: () => { run.emitThermoRow(sys); },
    afterStep: (step) => { if (sys.thermo.due(step, Number.NaN, o.laststep)) run.emitThermoRow(sys); },
    yieldMs: 30,
  });
};

/** integrate->setup(1) alone (the header and the setup row, no steps). */
export const setupOnly = async (run: RunApi, sys: System, o: { cancelled(): boolean }): Promise<void> => {
  run.initRun(sys);
  emitHeader(sys);
  await run.runVerlet(sys, 0, { cancelled: o.cancelled, afterSetup: () => { run.emitThermoRow(sys); }, afterStep: () => undefined, yieldMs: 30 });
};

/** firststep: the first step of the command (update->firststep, which the quench's laststep is counted from) */
export interface QuenchOpts { etol: number; ftol: number; maxiter: number; maxeval: number; firststep: number }

/**
 * quench(): an energy minimization with the min_style of the script (update->minimize), then the timestep and the
 * end step are reset as if the quench did not happen (PRD::quench and TAD::quench). Its thermo rows follow Output
 * (output.cpp, measured with native LAMMPS: the setup row, then the rows at the thermo interval, both capped at the
 * quench's laststep = firststep + maxiter, and the final row unless the iterations ran out). The interval counter
 * next_thermo = min((step / every) * every + every, laststep) is set at the setup and advanced after each row.
 */
export const quench = async (run: RunApi, sys: System, p: QuenchOpts, cancelled: () => boolean): Promise<void> => {
  const s = sys.state;
  const th = sys.thermo;
  const hold = s.step;
  run.initRun(sys);
  emitHeader(sys);
  const laststep = p.firststep + p.maxiter;
  const first = s.step;
  const every = th.every;
  const next0 = (): number => (every > 0 ? Math.min(Math.trunc(s.step / every) * every + every, laststep) : laststep);
  let next = next0();
  let lastPrinted = -1;
  const printRow = (): void => { run.emitThermoRow(sys); lastPrinted = s.step; };
  printRow();
  const res = await run.minimize(sys, p, {
    cancelled,
    thermo: (iterDone) => {
      // the minimizer's call at its setup (no step taken yet) prints nothing: the setup row is printed above
      if (iterDone || s.step === first) return;
      if (s.step === next) {
        printRow();
        next = every > 0 ? Math.min(next + every, laststep) : laststep;
      }
    },
  });
  // Min::run: the final output when the minimization stopped before its maximum number of iterations
  if (res.reason !== 'max iterations' && lastPrinted !== s.step) printRow();
  s.step = hold;
};

/** Removes the centre-of-mass velocity of all atoms (Velocity::zero_momentum, group all). */
export const zeroMomentumAll = (sys: System): void => {
  const s = sys.state;
  const p = [0, 0, 0];
  let mt = 0;
  for (let i = 0; i < s.n; i++) {
    const m = massOf(s, i);
    mt += m;
    for (let d = 0; d < 3; d++) p[d] += m * s.v[3 * i + d];
  }
  if (!(mt > 0)) return;
  for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) s.v[3 * i + d] -= p[d] / mt;
};

/**
 * velocity all create T seed loop {all|local|geom} dist {uniform|gaussian} mom yes rot no, as PRD::dephase calls it
 * (velocity.cpp at b332f28). loop all draws the atoms in ID order from one RanPark(seed); loop local draws them in the
 * process's storage order (native's atom order, SimState.order) from RanPark(seed + me) after WARMUP = 100 deviates, me
 * = 0 with one process per partition; loop geom seeds each atom from its seed and coordinates (RanPark::reset(seed, x)). Each deviate is scaled by 1/sqrt(mass) and the velocities
 * are rescaled to T with the temperature compute (temperature(), the prd_temp of the command).
 */
export const createVelocity = (sys: System, t: number, seed: number, loop: 'all' | 'local' | 'geom', gaussian: boolean, temperature: () => number): void => {
  const s = sys.state;
  const n = s.n;
  const draw = (rng: RanPark, i: number) => {
    const c = 1 / Math.sqrt(massOf(s, i));
    for (let d = 0; d < 3; d++) {
      const r = gaussian ? rng.gaussian() : rng.uniform() - 0.5;
      s.v[3 * i + d] = d === 2 && s.dimension === 2 ? 0 : r * c;
    }
  };
  if (loop === 'all') {
    let maxId = 0;
    for (let i = 0; i < n; i++) if (s.id[i] > maxId) maxId = s.id[i];
    if (maxId !== n) throw new StyleError('Atom IDs must be consecutive for velocity create loop all');
    const slot = new Int32Array(maxId + 1).fill(-1);
    for (let i = 0; i < n; i++) slot[s.id[i]] = i;
    const rng = new RanPark(seed);
    for (let id = 1; id <= maxId; id++) draw(rng, slot[id]);
  } else if (loop === 'local') {
    const rng = new RanPark(seed);
    for (let k = 0; k < 100; k++) rng.uniform();
    for (const i of nativeOrder(s)) draw(rng, i);
  } else {
    const rp = new RanPark(1);
    for (let i = 0; i < n; i++) {
      rp.resetAtom(seed, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
      const r = [0, 0, 0];
      for (let d = 0; d < 3; d++) r[d] = gaussian ? rp.gaussian() : rp.uniform() - 0.5;
      const c = 1 / Math.sqrt(massOf(s, i));
      for (let d = 0; d < 3; d++) s.v[3 * i + d] = d === 2 && s.dimension === 2 ? 0 : r[d] * c;
    }
  }
  zeroMomentumAll(sys);
  sys.refreshComputes();
  const cur = temperature();
  if (cur > 0) {
    const f = Math.sqrt(t / cur);
    for (let k = 0; k < 3 * n; k++) s.v[k] *= f;
  } else if (t > 0) sys.warn('velocity: current temperature is 0, cannot rescale');
};
