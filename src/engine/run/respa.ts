import type { System } from '../system';
import type { Fix } from '../fix/fix';
import type { LevelSelect, PerAtomTerm } from '../force/forcefield';
import { PairHybrid } from '../force/pair/hybrid';
import { clearAccum, newAccum, StyleError, type Accum } from '../force/types';
import { yieldNow, type RunHooks } from './verlet';

/*
 * run_style respa (rRESPA, Tuckerman, Berne and Martyna, J Chem Phys 97, 1990 (1992)).
 *
 * docs.lammps.org/run_style.html: "The *respa* style implements the rRESPA multi-timescale integrator
 * :ref:`(Tuckerman) <Tuckerman3>` with N hierarchical levels, where level 1 is the innermost loop
 * (shortest timestep) and level N is the outermost loop (largest timestep). The loop factor arguments
 * specify what the looping factor is between levels. N1 specifies the number of iterations of level 1
 * for a single iteration of level 2, N2 is the iterations of level 2 per iteration of level 3, etc."
 * "The :doc:`timestep <timestep>` command sets the large timestep for the outermost rRESPA level."
 * Defaults: "bond forces = level 1 (innermost loop)", "angle forces = same level as bond forces",
 * "dihedral forces = same level as angle forces", "improper forces = same level as dihedral forces",
 * "pair forces = level N (outermost level)", "kspace forces = same level as pair forces".
 *
 * Integrator: with P_k(dt) the level-k propagator, P_1(d) = K_1(d/2) D(d) K_1(d/2) (kick with the
 * level-1 forces, drift, kick) and P_k(dt) = K_k(dt/2) [P_{k-1}(dt/n_{k-1})]^{n_{k-1}} K_k(dt/2), where
 * K_k is the half kick with the level-k forces only. The drift happens only at the innermost level.
 * A force evaluation of level k is needed only when the positions have moved since it was last computed.
 * Measured with native LAMMPS (black box): a 3-atom probe (bond on level 1, lj/cut on level 2, lj units)
 * reproduces the potential energy, kinetic energy and positions of this nesting to about 1e-12 relative.
 *
 * Not implemented (throw StyleError): the inner/middle/outer pair splitting, compute pe/atom and
 * stress/atom under respa, and fixes other than nve, nvt (without a barostat) and temp/rescale.
 */

export const RESPA_MAX_LEVELS = 4;

export interface RespaSettings {
  /** N, the number of rRESPA levels (2 to 4). */
  readonly levels: number;
  /** n1 .. n(N-1): loop factors; loops[k] is the number of level k+1 iterations per level k+2 iteration. */
  readonly loops: readonly number[];
  readonly bond: number;
  readonly angle: number;
  readonly dihedral: number;
  readonly improper: number;
  readonly pair: number;
  readonly kspace: number;
  /** hybrid keyword: the level (1-N) of each pair sub-style, in pair_style order; null without it. */
  readonly hybrid: readonly number[] | null;
}

const isInt = (w: string | undefined): boolean => w !== undefined && /^[0-9]+$/.test(w);

/** run_style respa N n1 ... nN-1 keyword values ... — parses the arguments after "respa". */
export const parseRunStyleRespa = (a: readonly string[]): RespaSettings => {
  // a[0] is "respa"
  if (!isInt(a[1])) throw new StyleError('run_style respa: usage run_style respa N n1 ... nN-1 [keyword values ...] (N = number of levels)');
  const levels = Number(a[1]);
  if (levels < 2 || levels > RESPA_MAX_LEVELS) {
    throw new StyleError(`run_style respa: the browser engine supports 2 to ${RESPA_MAX_LEVELS} levels (got ${levels})`);
  }
  const loops: number[] = [];
  for (let k = 0; k < levels - 1; k++) {
    const w = a[2 + k];
    if (!isInt(w) || Number(w) < 1) throw new StyleError(`run_style respa: loop factor ${k + 1} must be a positive integer (got '${w ?? ''}'; ${levels - 1} loop factors are needed)`);
    loops.push(Number(w));
  }
  const kw: Partial<Record<'bond' | 'angle' | 'dihedral' | 'improper' | 'pair' | 'kspace', number>> = {};
  let hybrid: number[] | null = null;
  const levelOf = (key: string, w: string | undefined): number => {
    if (!isInt(w) || Number(w) < 1 || Number(w) > levels) {
      throw new StyleError(`run_style respa ${key}: the level must be an integer 1 to ${levels} (got '${w ?? ''}')`);
    }
    return Number(w);
  };
  for (let k = 2 + levels - 1; k < a.length;) {
    const key = a[k];
    switch (key) {
      case 'inner': case 'middle': case 'outer':
        throw new StyleError(`run_style respa keyword '${key}' (pair splitting by distance) is not supported by the browser engine; use the pair keyword or the hybrid keyword`);
      case 'bond': case 'angle': case 'dihedral': case 'improper': case 'pair': case 'kspace': {
        if (kw[key] !== undefined) throw new StyleError(`run_style respa: keyword '${key}' is given more than once`);
        kw[key] = levelOf(key, a[k + 1]);
        k += 2;
        break;
      }
      case 'hybrid': {
        if (hybrid) throw new StyleError('run_style respa: keyword \'hybrid\' is given more than once');
        hybrid = [];
        k++;
        // one level per hybrid sub-style: the values run up to the next keyword
        while (k < a.length && isInt(a[k])) { hybrid.push(levelOf('hybrid', a[k])); k++; }
        if (hybrid.length === 0) throw new StyleError('run_style respa hybrid needs one level per pair_style hybrid sub-style');
        break;
      }
      default:
        throw new StyleError(`run_style respa: unknown keyword '${key ?? ''}' (bond, angle, dihedral, improper, pair, hybrid, kspace)`);
    }
  }
  if (hybrid && kw.pair !== undefined) throw new StyleError('run_style respa: the hybrid keyword and the pair keyword are mutually exclusive');
  // docs.lammps.org/run_style.html defaults (see the header comment)
  const bond = kw.bond ?? 1;
  const angle = kw.angle ?? bond;
  const dihedral = kw.dihedral ?? angle;
  const improper = kw.improper ?? dihedral;
  const pair = kw.pair ?? levels;
  const kspace = kw.kspace ?? pair;
  return { levels, loops, bond, angle, dihedral, improper, pair, kspace, hybrid };
};

/** The integrator-capable fixes (fix/nve.ts, fix/nh.ts). */
interface RespaFix {
  readonly respaOK: boolean;
  respaKick(f: Float64Array, h: number): void;
  respaDrift(dt: number): void;
  respaBegin?(): void;
  respaEnd?(): void;
}

const isRespaFix = (f: Fix): f is Fix & RespaFix => typeof (f as Partial<RespaFix>).respaKick === 'function';

const copyAccum = (dst: Accum, src: Accum): void => {
  dst.evdwl = src.evdwl; dst.ecoul = src.ecoul; dst.elong = src.elong; dst.ebond = src.ebond;
  dst.eangle = src.eangle; dst.edihed = src.edihed; dst.eimp = src.eimp;
  dst.virial.set(src.virial); dst.vbond.set(src.vbond); dst.vangle.set(src.vangle);
  dst.vdihed.set(src.vdihed); dst.vimp.set(src.vimp); dst.vlong.set(src.vlong);
};

const addAccum = (dst: Accum, src: Accum): void => {
  dst.evdwl += src.evdwl; dst.ecoul += src.ecoul; dst.elong += src.elong; dst.ebond += src.ebond;
  dst.eangle += src.eangle; dst.edihed += src.edihed; dst.eimp += src.eimp;
  for (const [d, v] of [[dst.virial, src.virial], [dst.vbond, src.vbond], [dst.vangle, src.vangle],
    [dst.vdihed, src.vdihed], [dst.vimp, src.vimp], [dst.vlong, src.vlong]] as const) {
    for (let c = 0; c < 6; c++) d[c] += v[c];
  }
};

/** The fixes a respa run accepts; anything else throws naming the fix. */
const checkFixes = (sys: System): RespaFix[] => {
  const movers: RespaFix[] = [];
  for (const f of sys.fixes) {
    if (f.style === 'temp/rescale') continue; // velocity rescaling at the end of each outer step
    if (isRespaFix(f) && f.respaOK) { movers.push(f); continue; }
    throw new StyleError(`fix ${f.style} (${f.id}) is not supported with run_style respa (the browser engine supports fix nve, fix nvt without a barostat, and fix temp/rescale)`);
  }
  if (movers.length > 1) throw new StyleError('run_style respa: more than one fix integrates the atoms (use one of fix nve or fix nvt)');
  return movers;
};

/**
 * Runs n outer steps with the rRESPA integrator; the same hook order as runVerlet (neighbor decision,
 * sorting and wrapping only on rebuild steps, fix endOfStep on multiples of nevery) with the force
 * evaluations split by level. thermo sees the sum of all levels (pe, ke, virial, forces).
 */
export const runRespa = async (sys: System, nsteps: number, hooks: RunHooks): Promise<number> => {
  const cfg = sys.respa;
  if (!cfg) throw new StyleError('run_style respa is not set');
  const flags = sys.computeFlags();
  if (flags.eatom || flags.vatom) throw new StyleError('compute pe/atom and stress/atom are not supported with run_style respa');
  const movers = checkFixes(sys);
  const s = sys.state, nb = sys.nb, geom = sys.geom, ff = sys.ff;
  const N = cfg.levels;
  // Pair styles that keep per-contact history (granular, peridynamics) would advance it once per level
  // evaluation here; their respa behaviour is not measured, so they are refused by name.
  const pairNames = ff.pair instanceof PairHybrid ? ff.pair.subs.map((sub) => sub.style.name) : ff.pair ? [ff.pair.name] : [];
  for (const name of pairNames) {
    if (name.startsWith('gran') || name.startsWith('peri')) {
      throw new StyleError(`pair_style ${name} is not supported with run_style respa (its contact history is not split by level in the browser engine)`);
    }
  }

  // hybrid: one level per sub-style; otherwise whole terms
  const pairHybrid = ff.pair instanceof PairHybrid ? ff.pair : null;
  let hyb: readonly number[] | null = null;
  if (cfg.hybrid) {
    if (!pairHybrid) throw new StyleError('run_style respa hybrid needs a hybrid pair_style (pair_style hybrid or hybrid/overlay)');
    if (cfg.hybrid.length !== pairHybrid.subs.length) {
      throw new StyleError(`run_style respa hybrid gives ${cfg.hybrid.length} levels but pair_style ${pairHybrid.name} has ${pairHybrid.subs.length} sub-styles`);
    }
    hyb = cfg.hybrid;
  }
  const levelOf: Record<PerAtomTerm, number> = {
    pair: cfg.pair, bond: cfg.bond, angle: cfg.angle, dihedral: cfg.dihedral, improper: cfg.improper, kspace: cfg.kspace,
  };
  // level index l (0-based) -> the terms it evaluates
  const sels: LevelSelect[] = [];
  for (let l = 0; l < N; l++) {
    sels.push({
      has: (t: PerAtomTerm) => (t === 'pair' && hyb ? hyb.includes(l + 1) : levelOf[t] === l + 1),
      subPair: hyb ? (k: number) => hyb![k] === l + 1 : undefined,
    });
  }

  const n3 = 3 * s.n;
  const F: Float64Array[] = [];
  const A: Accum[] = [];
  const valid: boolean[] = [];
  for (let l = 0; l < N; l++) { F.push(new Float64Array(n3)); A.push(newAccum()); valid.push(false); }
  const invalidateAll = (): void => { valid.fill(false); };

  /** Force evaluation of level l at the current positions (ghosts refreshed first). */
  const evalLevel = (l: number): void => {
    nb.forwardComm(s, geom);
    const acc = ff.compute(s, nb, geom, { step: true }, sels[l]);
    F[l].set(s.f.subarray(0, n3));
    copyAccum(A[l], acc);
    valid[l] = true;
  };
  const ensure = (l: number): void => { if (!valid[l]) evalLevel(l); };

  /** Total forces and energies (sum of the levels) into the state, for thermo and dumps. */
  const totals = (): void => {
    s.f.fill(0, 0, n3);
    for (let l = 0; l < N; l++) for (let k = 0; k < n3; k++) s.f[k] += F[l][k];
    const acc = ff.acc;
    clearAccum(acc);
    for (let l = 0; l < N; l++) addAccum(acc, A[l]);
    // the tail correction is added once, not per level (ForceField.compute skips it under a selector)
    if (ff.etailV !== 0) acc.evdwl += ff.etailV / geom.volume(s.dimension);
  };

  const kick = (l: number, h: number): void => {
    ensure(l);
    for (const m of movers) m.respaKick(F[l], h);
  };
  const drift = (dt: number): void => {
    for (const m of movers) m.respaDrift(dt);
    invalidateAll();
  };
  /** One level-l step of length dt (P_l(dt) above). */
  const level = (l: number, dt: number): void => {
    if (l === 0) {
      kick(0, dt / 2);
      drift(dt);
      kick(0, dt / 2);
      return;
    }
    kick(l, dt / 2);
    const nIn = cfg.loops[l - 1];
    for (let j = 0; j < nIn; j++) level(l - 1, dt / nIn);
    kick(l, dt / 2);
  };

  // setup: neighbor lists, all levels, totals (Verlet::setup() with the forces split by level)
  sys.setupNeighbors();
  sys.sortAtoms(true);
  invalidateAll();
  for (let l = 0; l < N; l++) ensure(l);
  totals();
  sys.forcesCurrent();
  sys.assignDynamicGroups(true);
  for (const f of sys.fixes) f.setup();
  sys.refreshComputes();
  hooks.afterSetup();

  let taken = 0;
  let lastYield = performance.now();
  for (let k = 0; k < nsteps; k++) {
    if (hooks.cancelled()) break;
    s.step++;
    sys.refreshComputes();
    for (const m of movers) m.respaBegin?.();
    sys.bump();
    // the neighbor decision and any sorting happen on the outer step only; cached level forces are
    // recomputed after a rebuild (the sort changes the atom order)
    if (nb.decide(s.step, s, geom)) {
      sys.pbc();
      sys.sortAtoms(false);
      nb.build(s, geom, s.step);
      invalidateAll();
    }
    level(N - 1, s.dt);
    for (const m of movers) m.respaEnd?.();
    totals();
    sys.forcesCurrent();
    sys.assignDynamicGroups(false);
    for (const f of sys.fixes) if (f.endOfStep && s.step % f.nevery === 0) f.endOfStep();
    sys.refreshComputes();
    taken++;
    hooks.afterStep(s.step);
    if (hooks.yieldMs >= 0 && performance.now() - lastYield > hooks.yieldMs) {
      await yieldNow();
      lastYield = performance.now();
    }
  }
  for (const f of sys.fixes) f.postRun?.();
  return taken;
};
