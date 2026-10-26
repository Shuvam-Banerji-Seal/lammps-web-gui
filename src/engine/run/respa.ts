import type { System } from '../system';
import type { NeighList } from '../neighbor';
import type { Fix } from '../fix/fix';
import type { LevelSelect, PerAtomTerm } from '../force/forcefield';
import { PairHybrid } from '../force/pair/hybrid';
import { clearAccum, newAccum, StyleError, type Accum } from '../force/types';
import { inumeric, missingArgs, numeric } from '../force/util';
import { computeSetup, yieldNow, type RunHooks } from './verlet';
import { restartDue } from '../commands/restart';
import { haltRequested } from './halt';
import { runLabelCheck } from '../labelcheck';

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
 * Per-atom energy and virial (compute pe/atom, stress/atom, centroid/stress/atom): each level's evaluation tallies its
 * own terms, and a per-atom compute sees the sum over the levels of their last evaluation. As native LAMMPS does
 * (src/respa.cpp at b332f28): every level's forces are computed with the outer step's energy and virial flags, the
 * last time at the positions that end the outer step, and each style keeps the per-atom values of its last call.
 *
 * Not implemented (throw StyleError): the inner/middle/outer pair splitting, and fixes other than nve, nvt (without a
 * barostat) and temp/rescale.
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

/** What the run_style respa parser needs besides its words. */
export interface RespaParseContext {
  /** The number of sub-styles of a pair_style hybrid or hybrid/overlay, or null for another pair style (or none). */
  hybridStyles?: number | null;
  /** The log text ("Respa levels:") and warnings, as native prints them when the run style is set. */
  log?: (text: string) => void;
  warn?: (text: string) => void;
}

/**
 * run_style respa N n1 ... nN-1 keyword values ... — parses the arguments after "respa". As native LAMMPS does
 * (src/respa.cpp Respa::Respa at b332f28): N and the loop factors (utils::inumeric) with their checks, each keyword with
 * its word count and values, the combination checks, the defaults, the "Respa levels:" listing, the order check and the
 * warning for a level without forces. The engine then refuses what it does not implement: more than 4 levels or a
 * single one, a level outside 1..N, and the inner/middle/outer pair splitting.
 */
export const parseRunStyleRespa = (a: readonly string[], ctx: RespaParseContext = {}): RespaSettings => {
  // a[0] is "respa"; native's narg counts the words after it
  const narg = a.length - 1;
  if (narg < 1) throw missingArgs('run_style respa');
  const nlevels = inumeric(a[1]);
  if (nlevels < 1) throw new StyleError('Number of rRESPA levels must be >= 1');
  if (narg < nlevels) throw new StyleError('Not enough rRESPA loop factors provided');
  const loops: number[] = [];
  for (let k = 1; k < nlevels; k++) {
    const v = inumeric(a[1 + k]);
    if (v <= 0) throw new StyleError(`Invalid rRESPA loop factor ${v} for level ${k}`);
    loops.push(v);
  }
  // the levels as native keeps them (0-based, -1 when not set)
  const lv = { bond: -1, angle: -1, dihedral: -1, improper: -1, pair: -1, kspace: -1, inner: -1, middle: -1, outer: -1 };
  const typed: Partial<Record<keyof typeof lv, string>> = {};
  const cutoff = [0, 0, 0, 0];
  let hybrid: number[] | null = null;
  const splitting: string[] = [];
  for (let k = 1 + nlevels; k < a.length;) {
    const key = a[k];
    switch (key) {
      case 'bond': case 'angle': case 'dihedral': case 'improper': case 'pair': case 'outer': case 'kspace':
        if (k + 2 > a.length) throw missingArgs(`run_style respa ${key}`);
        lv[key] = inumeric(a[k + 1]) - 1;
        typed[key] = a[k + 1];
        if (key === 'outer') splitting.push(key);
        k += 2;
        break;
      case 'inner': case 'middle':
        if (k + 4 > a.length) throw missingArgs(`run_style respa ${key}`);
        lv[key] = inumeric(a[k + 1]) - 1;
        typed[key] = a[k + 1];
        cutoff[key === 'inner' ? 0 : 2] = numeric(a[k + 2]);
        cutoff[key === 'inner' ? 1 : 3] = numeric(a[k + 3]);
        splitting.push(key);
        k += 4;
        break;
      case 'hybrid': {
        // the hybrid keyword requires a hybrid pair style, and takes one level per sub-style
        const n = ctx.hybridStyles;
        if (n === undefined || n === null) throw new StyleError('Cannot use hybrid keyword without hybrid pair style');
        if (k + n > a.length) throw missingArgs('run_style respa hybrid');
        hybrid = [];
        for (let i = 0; i < n; i++) hybrid.push(inumeric(a[++k]) - 1);
        k++;
        break;
      }
      default:
        throw new StyleError(`Unknown run_style respa keyword ${key}`);
    }
  }
  const nhybrid = hybrid ? hybrid.length : 0;
  if (lv.pair >= 0 && (lv.inner >= 0 || lv.middle >= 0 || lv.outer >= 0)) throw new StyleError('Cannot set both respa pair and inner/middle/outer');
  if ((lv.inner >= 0 && lv.outer === -1) || (lv.outer >= 0 && lv.inner === -1)) throw new StyleError('Must set both respa inner and outer');
  if (lv.middle >= 0 && lv.inner === -1) throw new StyleError('Cannot set respa middle without inner/outer');
  if (nhybrid > 0 && (lv.pair >= 0 || lv.inner >= 0 || lv.middle >= 0 || lv.outer >= 0)) throw new StyleError('Cannot set respa hybrid and any of pair/inner/middle/outer');
  // the defaults: bond innermost, angle as bond, dihedral as angle, improper as dihedral, pair outermost (without
  // inner/middle/outer or hybrid), kspace as pair, as outer or as the largest hybrid level
  if (lv.bond === -1) lv.bond = 0;
  if (lv.angle === -1) lv.angle = lv.bond;
  if (lv.dihedral === -1) lv.dihedral = lv.angle;
  if (lv.improper === -1) lv.improper = lv.dihedral;
  if (lv.pair === -1 && lv.inner === -1 && nhybrid < 1) lv.pair = nlevels - 1;
  if (lv.kspace === -1 && lv.pair >= 0) lv.kspace = lv.pair;
  if (lv.kspace === -1 && lv.pair === -1) {
    if (nhybrid < 1) lv.kspace = lv.outer;
    else for (const h of hybrid!) if (lv.kspace < h) lv.kspace = h;
  }
  let mesg = 'Respa levels:\n';
  for (let i = 0; i < nlevels; i++) {
    mesg += `  ${i + 1} =`;
    for (const [term, name] of [['bond', 'bond'], ['angle', 'angle'], ['dihedral', 'dihedral'], ['improper', 'improper'], ['pair', 'pair'], ['inner', 'pair-inner'], ['middle', 'pair-middle'], ['outer', 'pair-outer']] as const) {
      if (lv[term] === i) mesg += ` ${name}`;
    }
    for (let j = 0; j < nhybrid; j++) if (hybrid![j] === i) mesg += ` hybrid-${j + 1}`;
    if (lv.kspace === i) mesg += ' kspace';
    mesg += '\n';
  }
  ctx.log?.(mesg.slice(0, -1));
  const disorder = () => new StyleError('Invalid order of forces within respa levels');
  if (lv.angle < lv.bond || lv.dihedral < lv.angle || lv.improper < lv.dihedral) throw disorder();
  if (lv.pair >= 0 && (lv.pair < lv.improper || lv.kspace < lv.pair)) throw disorder();
  if (lv.pair === -1 && lv.middle === -1 && nhybrid < 1) {
    if (lv.inner < lv.improper || lv.outer < lv.inner || lv.kspace < lv.outer) throw disorder();
  }
  if (lv.pair === -1 && lv.middle >= 0) {
    if (lv.inner < lv.improper || lv.middle < lv.inner || lv.outer < lv.inner || lv.kspace < lv.outer) throw disorder();
  }
  let empty = false;
  for (let i = 0; i < nlevels; i++) {
    if (Object.values(lv).every((l) => l !== i)) empty = true;
  }
  if (empty) ctx.warn?.('One or more respa levels compute no forces');
  if (lv.inner >= 0 && cutoff[1] < cutoff[0]) throw new StyleError('Respa inner cutoffs are invalid');
  if (lv.middle >= 0 && (cutoff[3] < cutoff[2] || cutoff[2] < cutoff[1])) throw new StyleError('Respa middle cutoffs are invalid');

  // what the browser engine implements
  if (splitting.length) {
    throw new StyleError(`run_style respa keyword '${splitting[0]}' (pair splitting by distance) is not supported by the browser engine; use the pair keyword or the hybrid keyword`);
  }
  if (nlevels < 2 || nlevels > RESPA_MAX_LEVELS) {
    throw new StyleError(`run_style respa: the browser engine supports 2 to ${RESPA_MAX_LEVELS} levels (got ${nlevels})`);
  }
  const level = (key: string, l: number, w: string | undefined): number => {
    if (l < 0 || l >= nlevels) throw new StyleError(`run_style respa ${key}: the level must be an integer 1 to ${nlevels} (got '${w ?? l + 1}')`);
    return l + 1;
  };
  const one = (key: 'bond' | 'angle' | 'dihedral' | 'improper' | 'pair' | 'kspace'): number => level(key, lv[key], typed[key]);
  return {
    levels: nlevels, loops,
    bond: one('bond'), angle: one('angle'), dihedral: one('dihedral'), improper: one('improper'),
    // with hybrid native leaves the pair level unset (-1); the sub-style levels are used instead
    pair: nhybrid > 0 ? nlevels : one('pair'), kspace: one('kspace'),
    hybrid: hybrid ? hybrid.map((h) => level('hybrid', h, String(h + 1))) : null,
  };
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

/**
 * A fix that filters the evaluation of the outermost level (fix filter/corotate, docs.lammps.org/fix_filter_corotate.html:
 * "the corotational filter ... is applied to the outermost level"): respaFilterPositions gives the positions of the owned
 * atoms that level is evaluated at, respaFilterForces filters that level's forces in place, respaPreNeighbor runs at
 * setup and after every reneighboring.
 */
export interface RespaOuterFilter {
  readonly respaOuterFilter: true;
  respaPreNeighbor(): void;
  respaFilterPositions(): Float64Array;
  respaFilterForces(f: Float64Array): void;
  /**
   * The half list the filtered level is evaluated with, or null for the engine's own. With filtered owned atoms and real
   * ghosts, the energy of a pair that crosses a periodic boundary depends on which of its two atoms is the owned one,
   * so the fix hands the pairs from the side native LAMMPS's half list stores them.
   */
  respaHalfList?(): NeighList | null;
}

const isOuterFilter = (f: Fix): f is Fix & RespaOuterFilter => (f as Partial<RespaOuterFilter>).respaOuterFilter === true;

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
    if (isOuterFilter(f)) continue; // the outermost-level filter (evalLevel)
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
  const evalFlags = { ...flags, step: true };
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

  // As native LAMMPS does (src/MOLECULE/dihedral_charmm.cpp and dihedral_charmmfsw.cpp init_style at b332f28): the
  // dihedral computes the 1-4 pair terms, so "if (r->level_pair >= 0 && (r->level_pair != r->level_dihedral))
  // error->all(FLERR, "Dihedral style charmm must be set to same r-RESPA level as 'pair'")"
  const dihedralName = ff.dihedral?.name;
  if (ff.pair && !cfg.hybrid && (dihedralName === 'charmm' || dihedralName === 'charmmfsw') && cfg.dihedral !== cfg.pair) {
    throw new StyleError(`Dihedral style ${dihedralName} must be set to same r-RESPA level as 'pair'`);
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

  const filter = sys.fixes.find(isOuterFilter) ?? null;
  // native setup sums the level forces (sum_flevel_f) before the fix's setup filters the outermost one
  let deferFilter = false;
  const n3 = 3 * s.n;
  const F: Float64Array[] = [];
  const A: Accum[] = [];
  const valid: boolean[] = [];
  // per-atom terms of each level's last evaluation (empty without a per-atom compute)
  type Terms = Partial<Record<PerAtomTerm, Float64Array>>;
  const ET: Terms[] = [], VT: Terms[] = [], CT: Terms[] = [];
  for (let l = 0; l < N; l++) { F.push(new Float64Array(n3)); A.push(newAccum()); valid.push(false); ET.push({}); VT.push({}); CT.push({}); }
  const keepTerms = (l: number): void => { ET[l] = ff.eatomTerm; VT[l] = ff.vatomTerm; CT[l] = ff.cvatomTerm; };
  const invalidateAll = (): void => { valid.fill(false); };

  /** Force evaluation of level l at the current positions (ghosts refreshed first). */
  const evalLevel = (l: number): void => {
    nb.forwardComm(s, geom);
    if (filter && l === N - 1) {
      // the outermost level at the filtered positions of the owned atoms; the ghosts keep the real positions of the last
      // forward communication (native pre_force_respa points atom->x at the filtered copy without communicating it)
      const xf = filter.respaFilterPositions();
      const half = filter.respaHalfList?.() ?? null;
      const keep = Float64Array.from(s.x.subarray(0, n3));
      const keepHalf = nb.half;
      s.x.set(xf.subarray(0, n3));
      nb.xall.set(xf.subarray(0, n3));
      if (half) nb.half = half;
      let acc: Accum;
      try {
        acc = ff.compute(s, nb, geom, evalFlags, sels[l]);
      } finally {
        s.x.set(keep);
        nb.xall.set(keep);
        nb.half = keepHalf;
      }
      F[l].set(s.f.subarray(0, n3));
      if (!deferFilter) filter.respaFilterForces(F[l]);
      copyAccum(A[l], acc);
      keepTerms(l);
      valid[l] = true;
      return;
    }
    const acc = ff.compute(s, nb, geom, evalFlags, sels[l]);
    F[l].set(s.f.subarray(0, n3));
    copyAccum(A[l], acc);
    keepTerms(l);
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
    if (flags.eatom || flags.vatom || flags.cvatom) {
      // per-atom terms: a term belongs to one level (a hybrid pair's sub-styles may be on several, then they add)
      const merge = (levels: Terms[]): Terms => {
        const out: Terms = {};
        for (const t of levels) {
          for (const [k, arr] of Object.entries(t) as [PerAtomTerm, Float64Array][]) {
            const have = out[k];
            if (!have) { out[k] = Float64Array.from(arr); continue; }
            for (let q = 0; q < arr.length; q++) have[q] += arr[q];
          }
        }
        return out;
      };
      const total = (t: Terms, len: number): Float64Array => {
        const out = new Float64Array(len);
        for (const arr of Object.values(t)) for (let q = 0; q < len; q++) out[q] += arr![q];
        return out;
      };
      ff.eatomTerm = merge(ET);
      ff.vatomTerm = merge(VT);
      ff.cvatomTerm = merge(CT);
      ff.eatom = flags.eatom ? total(ff.eatomTerm, s.n) : null;
      ff.vatom = flags.vatom ? total(ff.vatomTerm, 6 * s.n) : null;
    }
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
  sys.setupNeighbors(true);
  filter?.respaPreNeighbor();
  for (const f of sys.fixes) f.setupPreForce?.();
  invalidateAll();
  deferFilter = true;
  for (let l = 0; l < N; l++) ensure(l);
  deferFilter = false;
  totals();
  // FixFilterCorotate::setup: the outermost level's setup forces are filtered after the sum
  if (filter) filter.respaFilterForces(F[N - 1]);
  sys.forcesCurrent(flags);
  sys.assignDynamicGroups(true);
  computeSetup(sys);
  for (const f of sys.fixes) {
    f.setup();
    // As native LAMMPS does (src/EXTRA-FIX/fix_filter_corotate.cpp FixFilterCorotate::setup at b332f28):
    // "copy_flevel_f(nlevels_respa-1)", the filter, "copy_f_flevel(nlevels_respa-1)": the atom forces are left holding
    // the filtered outermost level, which the fixes set up after it and the setup's output (fnorm, f dumps) read
    if (f === filter) s.f.set(F[N - 1].subarray(0, n3));
  }
  // labelmap check_labels: once, at the end of the setup (native Modify::setup)
  runLabelCheck(sys);
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
    nb.forceBuildStep = restartDue(sys, s.step) ? s.step : -1;
    if (nb.decide(s.step, s, geom)) {
      sys.pbc();
      sys.sortAtoms(false);
      filter?.respaPreNeighbor();
      nb.build(s, geom, s.step);
      invalidateAll();
    }
    level(N - 1, s.dt);
    for (const m of movers) m.respaEnd?.();
    totals();
    sys.forcesCurrent(flags);
    sys.assignDynamicGroups(false);
    for (const f of sys.fixes) if (f.endOfStep && s.step % f.nevery === 0) f.endOfStep();
    sys.refreshComputes();
    taken++;
    hooks.afterStep(s.step);
    if (haltRequested(sys)) break;
    if (hooks.yieldMs >= 0 && performance.now() - lastYield > hooks.yieldMs) {
      await yieldNow();
      lastYield = performance.now();
    }
  }
  for (const f of sys.fixes) f.postRun?.();
  return taken;
};
