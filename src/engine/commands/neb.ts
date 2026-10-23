import type { Handler } from './args';
import { int, num } from './args';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { EngineEvent } from '../types';
import { massOf } from '../atoms';
import { initRun } from '../run/verlet';
import { FixNeb } from '../fix/neb';
import { ensureThermoComputes } from './run';
import { fireSettings, type FireRunner, type LineminHook } from '../run/min';
import { gplHook } from '../gplHooks';
import { libm } from '../libm';

/*
 * neb etol ftol N1 N2 Nevery file-style [arg] [verbosity default] —
 * docs.lammps.org/neb.html. The replicas (partitions) run together: each
 * iteration moves every replica one damped-dynamics step, evaluates its
 * forces, and then the inter-replica nudging of fix neb is computed from all
 * replicas' positions and energies (the same input runs on every partition,
 * so the replicas share one set of fixes and settings, each in its own
 * System).
 *
 * Quotes from neb.rst that the driver relies on:
 *  - "Two or more replicas must be used; the first and last are the end
 *    points of the transition path."
 *  - "A NEB calculation proceeds in two stages, each of which is a
 *    minimization procedure, performed via damped dynamics."
 *  - "In the second stage of NEB, the replica with the highest energy is
 *    selected and the inter-replica forces on it are converted to a force that
 *    drives its atom coordinates to the top or saddle point of the barrier"
 *  - "The energies being compared to etol do not include any contribution
 *    from the inter-replica nudging forces, since these are non-conservative."
 *  - "The forces being compared to ftol include the inter-replica nudging
 *    forces."
 *  - "N1 and N2 must both be multiples of Nevery."
 *  - "The RD of the first replica RD1 = 0.0; the RD of the final replica RDN =
 *    RDT, the total reaction coordinate. The normalized RDs are divided by
 *    RDT, so that they form a monotonically increasing sequence from zero to
 *    one."
 *  - "The forward (reverse) energy barrier is the potential energy of the
 *    highest replica minus the energy of the first (last) replica."
 *  - "The "maximum force per replica" is the two-norm of the 3N-length force
 *    vector for the atoms in each replica, maximized across replicas"
 * fix_neb.rst gives the nudged force of an intermediate replica i as
 * -grad V + (grad V . T) T + F_par + F_perp, and of the climbing replica as
 * -grad V + 2 (grad V . T) T + F_perp, where T is the unit tangent (fix_neb.rst
 * cites Henkelman and Jonsson 2000 for it). The end replicas get no nudging
 * unless the end keyword is given.
 *
 * Supported here: file-style final, each and none; verbosity default, terse and
 * verbose; every fix neb keyword (parallel neigh, ideal and equal, perp, end),
 * whose forces are the GPL port of native FixNEB::min_post_force
 * (src/engine-gpl/replica/fix_neb.ts, through gplHook('neb')); min_style
 * quickmin and fire (the GPL port of native MinFire run over all replicas, as
 * native's multireplica minimization).
 */

/** What the neb command needs from its session (Session implements it). */
export interface NebHost {
  readonly replicaSystems: readonly System[];
  readonly backendLabel: string;
  emit(ev: EngineEvent): void;
  /** the command's arguments as replica r parsed them (undefined: the same words for every replica) */
  argsOf(r: number): readonly string[] | undefined;
  /**
   * The thermo rows of each replica's minimization (native: the minimize of every partition prints its own thermo
   * output, setup = true for the header and the setup row of a stage). Only tad asks for them (see tad.ts).
   */
  replicaThermo?(sys: System, setup: boolean): void;
}

const USAGE = 'usage: neb etol ftol N1 N2 Nevery file-style [arg] [verbosity default|terse|verbose]';

type Verbosity = 'default' | 'terse' | 'verbose';

/**
 * The NEB table columns: the step, the force maxima, the gradients and barriers, then RD and PE per replica (1-based),
 * and with verbosity verbose six more per replica. neb.rst: "If verbosity = terse the PE and RD of each replica are not
 * output; if verbosity = verbose the angles and gradients of each replica are output in addition".
 */
const tableKeywords = (n: number, verbosity: Verbosity): string[] => {
  const k = ['Step', 'MaxReplicaForce', 'MaxAtomForce', 'GradV0', 'GradV1', 'GradVc', 'EBF', 'EBR', 'RDT'];
  if (verbosity !== 'terse') for (let r = 1; r <= n; r++) k.push(`RD${r}`, `PE${r}`);
  if (verbosity === 'verbose') {
    for (let r = 1; r <= n; r++) k.push(`pathangle${r}`, `angletangrad${r}`, `anglegrad${r}`, `gradV${r}`, `RepForce${r}`, `MaxAtomForce${r}`);
  }
  return k;
};

/** One replica's FixNEB values after the nudging (src/engine-gpl/replica/fix_neb.ts). */
interface NebOutput { veng: number; plen: number; nlen: number; gradlen: number; dotpath: number; dottangrad: number; dotgrad: number }

/** The GPL port of FixNEB::min_post_force, reached through gplHook('neb') (lazy group gpl_linemin). */
interface NebForceHook {
  nebForces(
    reps: { X: Float64Array; F: Float64Array; E: number }[],
    p: {
      kspring: number; kspringPerp: number; mode: 'neigh' | 'ideal' | 'equal'; freeEndIni: boolean; freeEndFinal: boolean;
      freeEndFinalWithRespToEIni: boolean; finalAndInterWithRespToEIni: boolean; kspringIni: number; kspringFinal: number;
    },
    ends: { eIniIni: number; eFinalIni: number },
    rclimber: number,
    ntimestep: number,
    box: { prd: [number, number, number]; periodic: [boolean, boolean, boolean]; triclinic: boolean; tilt: [number, number, number] },
  ): NebOutput[];
}

/** Coordinates of one NEB file: "ID x y z" lines after the atom count (neb.rst). */
const parseCoords = (text: string): Map<number, [number, number, number]> => {
  const lines = text.split('\n').map((l) => l.trim()).filter((l) => l !== '' && !l.startsWith('#'));
  if (!lines.length) throw new StyleError('neb file has no atom count line');
  const count = Number(lines[0].split(/\s+/)[0]);
  if (!Number.isInteger(count) || count < 0) throw new StyleError(`neb file: first line must be the number of atoms, got '${lines[0]}'`);
  if (lines.length < count + 1) throw new StyleError(`neb file lists ${count} atoms but has ${lines.length - 1} lines`);
  const out = new Map<number, [number, number, number]>();
  for (let k = 1; k <= count; k++) {
    const w = lines[k].split(/\s+/);
    const x = [Number(w[1]), Number(w[2]), Number(w[3])] as [number, number, number];
    if (!Number.isInteger(Number(w[0])) || x.some((c) => !Number.isFinite(c))) throw new StyleError(`neb file: bad line '${lines[k]}'`);
    out.set(Number(w[0]), x);
  }
  return out;
};

/** Minimum-image difference a - b along periodic dimensions of an orthogonal box. */
const minImage = (d: number, len: number, periodic: boolean): number => {
  if (!periodic || !(len > 0)) return d;
  let x = d;
  while (x > len / 2) x -= len;
  while (x < -len / 2) x += len;
  return x;
};

/** One replica's state at the current iteration, restricted to the NEB atoms (sorted by id). */
interface Snap {
  /** local indices of the NEB atoms, in id order */
  idx: number[];
  /** their coordinates, 3 per atom */
  X: Float64Array;
  /** the force of every atom before the nudging (3 per atom, all atoms) */
  raw: Float64Array;
  /** potential energy without the nudging */
  E: number;
}

/** Atom-to-group membership and the NEB atoms of a replica (id order). */
const nebAtoms = (sys: System, bit: number): number[] => {
  const s = sys.state;
  const idx: number[] = [];
  for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) idx.push(i);
  return idx.sort((a, b) => s.id[a] - s.id[b]);
};

/** An extensive energy as thermo prints it: divided by the atom count with thermo norm on (lj units). */
const perAtom = (sys: System, e: number): number => (sys.thermo.norm && sys.state.n > 0 ? e / sys.state.n : e);

/** One force evaluation of a replica: the same steps as the minimizer's evaluation (run/min.ts), without the box DOF. */
const evaluate = (sys: System, first: boolean): number => {
  const s = sys.state;
  const nb = sys.nb;
  sys.bump();
  if (first) { sys.setupNeighbors(); sys.sortAtoms(true); }
  else if (nb.decide(s.step, s, sys.geom)) {
    sys.pbc();
    sys.sortAtoms(false);
    nb.build(s, sys.geom, s.step);
  } else nb.forwardComm(s, sys.geom);
  for (const f of sys.fixes) f.minPreForce?.();
  sys.computeForcesInRun(sys.computeFlags());
  sys.forcesCurrent();
  if (first) sys.assignDynamicGroups(true);
  for (const f of sys.fixes) f.minPostForce?.();
  sys.refreshComputes();
  const a = sys.ff.acc;
  return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp + sys.fixEnergy();
};

/**
 * One quickmin iteration of all replicas together. Measured with native LAMMPS (black box, a 3-atom lj/cut
 * replica pair, timestep 0.1, tests/oracle/w46neb_tiny.in with Nevery 1): the energy after iteration 1 is
 * unchanged; the replica energies follow one projection of v onto F summed over all replicas (a single scale
 * factor, sum P / sum F.F, then v = scale F), then the position update with the displacement limit, then the
 * velocity update. A per-replica projection gives different energies of the end replica from iteration 3 on.
 * neb.rst: "a NEB calculation performs a damped dynamics minimization across all the replicas".
 */
const quickminStep = (reps: readonly System[]): void => {
  let P = 0, FF = 0;
  for (const sys of reps) {
    const s = sys.state;
    for (let k = 0; k < 3 * s.n; k++) { P += s.v[k] * s.f[k]; FF += s.f[k] * s.f[k]; }
  }
  const scale = P > 0 && FF > 0 ? P / FF : 0;
  let vmax = 0;
  for (const sys of reps) {
    const s = sys.state;
    for (let k = 0; k < 3 * s.n; k++) {
      s.v[k] = scale * s.f[k];
      vmax = Math.max(vmax, Math.abs(s.v[k]));
    }
  }
  // the displacement limit uses the largest velocity component of all replicas (min_modify dmax)
  const dmax = reps[0].minSettings.dmax;
  const dt = reps[0].state.dt;
  const dtx = vmax * dt > dmax ? dmax / vmax : dt;
  for (const sys of reps) {
    const s = sys.state;
    const n3 = 3 * s.n;
    for (let k = 0; k < n3; k++) s.x[k] += dtx * s.v[k];
    const ftm2v = s.units.ftm2v;
    for (let i = 0; i < s.n; i++) {
      const c = dt * ftm2v / massOf(s, i);
      s.v[3 * i] += c * s.f[3 * i]; s.v[3 * i + 1] += c * s.f[3 * i + 1]; s.v[3 * i + 2] += c * s.f[3 * i + 2];
    }
    if (s.dimension === 2) for (let i = 0; i < s.n; i++) s.v[3 * i + 2] = 0;
  }
};

const norm = (a: Float64Array | number[], from = 0, to = a.length): number => {
  let t = 0;
  for (let k = from; k < to; k++) t += a[k] * a[k];
  return Math.sqrt(t);
};

/**
 * neb command: the two stages of the NEB calculation (neb.rst). Returns the energy barrier of the last table row (the
 * forward barrier EBF without thermo normalization: native neb->ebf, used by the tad command).
 */
export const runNeb = async (host: NebHost, a: string[]): Promise<number> => {
  const reps = [...host.replicaSystems];
  if (a.length < 6) throw new StyleError(USAGE);
  const etol = num(a[0], 'etol'), ftol = num(a[1], 'ftol');
  const N1 = int(a[2], 'N1'), N2 = int(a[3], 'N2'), nevery = int(a[4], 'Nevery');
  if (etol < 0 || ftol < 0) throw new StyleError('neb etol and ftol must be >= 0');
  if (N1 < 0 || N2 < 0) throw new StyleError('neb N1 and N2 must be >= 0');
  if (nevery < 1) throw new StyleError('neb Nevery must be >= 1');
  if (N1 % nevery !== 0 || N2 % nevery !== 0) throw new StyleError('neb N1 and N2 must be multiples of Nevery');
  // As native LAMMPS does (src/REPLICA/neb.cpp NEB::command at b332f28): the single-replica check comes after the numeric checks and before the file-style
  if (reps.length < 2) throw new StyleError('Cannot use NEB with a single replica');
  const fileStyle = a[5];
  let k = 6;
  let coordFile: string | null = null;
  /** file-style each: the file of every replica (replica 1 onward; the first keeps its coordinates) */
  const eachFile: (string | null)[] = reps.map(() => null);
  if (fileStyle === 'final') {
    if (a[6] === undefined) throw new StyleError('neb final needs a filename');
    coordFile = a[6];
    k = 7;
  } else if (fileStyle === 'each') {
    for (let r = 1; r < reps.length; r++) {
      const name = (host.argsOf(r) ?? a)[6];
      if (name === undefined) throw new StyleError('neb each needs a filename');
      eachFile[r] = name;
    }
    k = 7;
  } else if (fileStyle !== 'none') throw new StyleError(`neb file-style must be final, each or none (got '${fileStyle}')`);
  let verbosity: Verbosity = 'default';
  while (k < a.length) {
    const v = a[k + 1];
    if (a[k] === 'verbosity' && (v === 'default' || v === 'terse' || v === 'verbose')) { verbosity = v; k += 2; continue; }
    throw new StyleError(`illegal neb keyword '${a[k]}'`);
  }

  // every replica: the same settings, a damped minimizer, one fix neb
  const fixes: FixNeb[] = [];
  for (const sys of reps) {
    if (!sys.hasBox) throw new StyleError('neb needs a simulation box in every replica');
    // As native LAMMPS does (src/REPLICA/neb.cpp at b332f28): a line-search minimizer is refused
    if (sys.minStyle === 'cg' || sys.minStyle === 'sd' || sys.minStyle === 'hftn') throw new StyleError('NEB requires a damped dynamics minimizer');
    if (sys.minStyle !== 'quickmin' && sys.minStyle !== 'fire') throw new StyleError(`neb needs min_style quickmin or fire; min_style is ${sys.minStyle}`);
    const nebs = sys.fixes.filter((f): f is FixNeb => f instanceof FixNeb);
    if (nebs.length !== 1) throw new StyleError('neb needs exactly one fix neb in every replica');
    fixes.push(nebs[0]);
  }
  const fix0 = fixes[0];
  const nReps = reps.length;
  const n0 = reps[0].state.n;
  for (const sys of reps) if (sys.state.n !== n0) throw new StyleError('neb needs the same number of atoms in every replica');
  const dims = reps[0].state.dimension;
  const nebBit = fix0.groupBit;
  const nebHook = gplHook<LineminHook & NebForceHook>('neb');
  const settings = {
    kspring: fix0.kspring, kspringPerp: fix0.kspringPerp, mode: fix0.parallel, freeEndIni: fix0.freeEndIni, freeEndFinal: fix0.freeEndFinal,
    freeEndFinalWithRespToEIni: fix0.freeEndFinalWithRespToEIni, finalAndInterWithRespToEIni: fix0.finalAndInterWithRespToEIni,
    kspringIni: fix0.kspringIni, kspringFinal: fix0.kspringFinal,
  };
  // native: each partition's own fix keeps its target energy (EIniIni in the first replica's, EFinalIni in the last's)
  const firstFix = fixes[0], lastFix = fixes[fixes.length - 1];

  // the replicas start like a minimization: thermo computes, run state, fix setup, and the quickmin velocities at zero
  // (MinQuickMin::setup_style at b332f28, src/min_quickmin.cpp: v = 0 for every atom; a dynamics before the NEB leaves velocities)
  for (const sys of reps) {
    ensureThermoComputes(sys);
    initRun(sys);
    for (const f of sys.fixes) f.minSetup?.();
    sys.state.v.fill(0);
  }

  // initial configurations (neb.rst, file-style final: the intermediate replicas are interpolated from the
  // current positions to the file's positions; each: every replica reads its own file; none: no change)
  for (let r = 1; r < nReps; r++) {
    const name = eachFile[r];
    if (name === null) continue;
    const coords = parseCoords(reps[0].readFile(name));
    for (const [id, x] of coords) {
      const li = reps[r].indexOfId(id);
      if (li < 0) throw new StyleError(`neb each file ${name}: atom ${id} is not in replica ${r + 1}`);
      for (let d = 0; d < 3; d++) reps[r].state.x[3 * li + d] = x[d];
    }
    reps[r].pbc();
  }
  if (coordFile !== null) {
    const coords = parseCoords(reps[0].readFile(coordFile));
    const last = reps[nReps - 1];
    for (const [id, xf] of coords) {
      if (last.indexOfId(id) < 0) continue;
      for (let r = 1; r < nReps; r++) {
        const sys = reps[r];
        const li = sys.indexOfId(id);
        if (li < 0) throw new StyleError(`neb final file: atom ${id} is not in replica ${r + 1}`);
        const s = sys.state;
        if (r === nReps - 1) {
          for (let d = 0; d < 3; d++) s.x[3 * li + d] = xf[d];
        } else {
          // the fraction of the way from the replica's own position to the final one: r / (N - 1)
          const frac = r / (nReps - 1);
          for (let d = 0; d < 3; d++) {
            const len = s.box.hi[d] - s.box.lo[d];
            const dx = minImage(xf[d] - s.x[3 * li + d], len, s.box.periodic[d]);
            s.x[3 * li + d] += frac * dx;
          }
        }
      }
    }
    for (const sys of reps.slice(1)) sys.pbc();
  }

  // stage state
  const saved = reps.map((sys) => ({ every: sys.nb.every, delay: sys.nb.delay, check: sys.nb.check }));
  for (const sys of reps) { sys.nb.every = 1; sys.nb.delay = 0; sys.nb.check = true; }
  const stepBase = reps[0].state.step;
  const cols = tableKeywords(nReps, verbosity);
  const units = reps[0].units.style;
  let climb = -1;
  /** every replica's FixNEB values from the last nudging (what native print_status reads) */
  let nebOut: NebOutput[] = [];

  const snapshot = (sys: System, E: number): Snap => {
    const s = sys.state;
    const idx = nebAtoms(sys, nebBit);
    const X = new Float64Array(3 * idx.length);
    idx.forEach((li, j) => { for (let d = 0; d < 3; d++) X[3 * j + d] = s.x[3 * li + d]; });
    return { idx, X, raw: Float64Array.from(s.f.subarray(0, 3 * s.n)), E };
  };

  const energies: number[] = new Array(nReps).fill(0);
  let stepsDone = 0;
  let snaps: Snap[] = [];
  const evaluateAll = (first: boolean): void => {
    snaps = reps.map((sys, r) => {
      energies[r] = evaluate(sys, first);
      return snapshot(sys, energies[r]);
    });
  };

  // the nudging forces of every replica (fix_neb.rst), from the positions and raw forces of all replicas
  const applyNudging = (): void => {
    const g = reps[0].geom;
    const box = { prd: [g.lx, g.ly, g.lz] as [number, number, number], periodic: g.periodic, triclinic: g.triclinic, tilt: [g.xy, g.xz, g.yz] as [number, number, number] };
    const inputs = snaps.map((sn) => {
      const F = new Float64Array(3 * sn.idx.length);
      sn.idx.forEach((li, j) => { for (let d = 0; d < 3; d++) F[3 * j + d] = sn.raw[3 * li + d]; });
      return { X: sn.X, F, E: sn.E };
    });
    const ends = { eIniIni: firstFix.endEnergies.eIniIni, eFinalIni: lastFix.endEnergies.eFinalIni };
    nebOut = nebHook.nebForces(inputs, settings, ends, climb, reps[0].state.step, box);
    firstFix.endEnergies.eIniIni = ends.eIniIni;
    lastFix.endEnergies.eFinalIni = ends.eFinalIni;
    for (let r = 0; r < nReps; r++) {
      const s = reps[r].state;
      const sn = snaps[r];
      // raw forces first, then the NEB atoms get the nudged force
      s.f.set(sn.raw.subarray(0, 3 * s.n));
      const F = inputs[r].F;
      sn.idx.forEach((li, j) => { for (let d = 0; d < 3; d++) s.f[3 * li + d] = F[3 * j + d]; });
      if (dims === 2) for (let i = 0; i < s.n; i++) s.f[3 * i + 2] = 0;
    }
  };

  const printRow = (step: number): void => {
    // As native LAMMPS does (src/REPLICA/neb.cpp print_status at b332f28): the reaction coordinate sums each replica's
    // distance to the previous one, and the total ends with the next-to-last replica's distance to the last
    const rdist = [0];
    for (let r = 1; r < nReps; r++) rdist.push(rdist[r - 1] + nebOut[r].plen);
    const RDT = rdist[nReps - 1] = rdist[nReps - 2] + nebOut[nReps - 2].nlen;
    let maxRep = 0, maxAtom2 = 0;
    const repForce: number[] = [], repMax2: number[] = [];
    for (let r = 0; r < nReps; r++) {
      const s = reps[r].state;
      repForce.push(norm(s.f, 0, 3 * s.n));
      // printed as the square of the largest force component (Min::fnorm_inf is the largest f[i]*f[i])
      let m2 = 0;
      for (let i = 0; i < 3 * s.n; i++) m2 = Math.max(m2, s.f[i] * s.f[i]);
      repMax2.push(m2);
      maxAtom2 = Math.max(maxAtom2, m2);
      maxRep = Math.max(maxRep, repForce[r]);
    }
    // gradV is the norm of the force of the NEB atoms only, before the nudging (FixNEB gradlen)
    const gradV = nebOut.map((o) => o.gradlen);
    // in the climbing stage the barriers and GradVc are those of the climbing replica (irep = fneb->rclimber), even when
    // another replica is higher at that step; before it, of the highest replica
    let hi = 0;
    for (let r = 1; r < nReps; r++) if (energies[r] > energies[hi]) hi = r;
    const top = climb >= 0 ? climb : hi;
    ebfRaw = energies[top] - energies[0];
    // the energies are printed as thermo prints pe: divided by the atom count when thermo norm is on
    // (lj units; docs.lammps.org/thermo_modify.html "norm")
    const pe = reps.map((sys, r) => perAtom(sys, energies[r]));
    const row: Record<string, number> = {
      Step: step,
      MaxReplicaForce: maxRep,
      MaxAtomForce: maxAtom2,
      GradV0: gradV[0],
      GradV1: gradV[nReps - 1],
      GradVc: gradV[top],
      EBF: pe[top] - pe[0],
      EBR: pe[top] - pe[nReps - 1],
      RDT,
    };
    if (verbosity !== 'terse') {
      for (let r = 0; r < nReps; r++) {
        row[`RD${r + 1}`] = r > 0 ? rdist[r] / RDT : 0;
        row[`PE${r + 1}`] = pe[r];
      }
    }
    if (verbosity === 'verbose') {
      // the angles in degrees: 180 - acos(cosine); the end replicas have no path angle, the last no gradient angle
      const deg = (c: number): number => 180 - libm.acos(c) * (180 / Math.PI);
      for (let r = 0; r < nReps; r++) {
        const o = nebOut[r];
        row[`pathangle${r + 1}`] = r === 0 || r === nReps - 1 ? NaN : deg(o.dotpath);
        row[`angletangrad${r + 1}`] = deg(o.dottangrad);
        row[`anglegrad${r + 1}`] = r === nReps - 1 ? NaN : deg(o.dotgrad);
        row[`gradV${r + 1}`] = o.gradlen;
        row[`RepForce${r + 1}`] = repForce[r];
        row[`MaxAtomForce${r + 1}`] = repMax2[r];
      }
    }
    host.emit({ kind: 'thermo', row });
  };

  const header = () => host.emit({ kind: 'thermo-header', keywords: cols, labels: cols, units });

  const t0 = performance.now();
  let ebfRaw = 0;
  for (const f of fixes) f.active = true;
  try {
    // the initial replicas: forces, nudging, then the first row
    evaluateAll(true);
    applyNudging();
    header();
    printRow(stepBase);

    let prevE: number[] = energies.slice();
    /** etol: the energy of every replica settles; ftol: the largest replica force is below ftol (neb.rst) */
    const converged = (): boolean => {
      if (etol > 0) {
        let eConv = true;
        for (let r = 0; r < nReps && eConv; r++) {
          const e = energies[r], ep = prevE[r];
          eConv = Math.abs(e - ep) < etol * 0.5 * (Math.abs(e) + Math.abs(ep) + 1e-8);
        }
        if (eConv) return true;
      }
      return ftol > 0 && forceMax() < ftol;
    };
    /**
     * fire over all replicas (native multireplica minimization): FireRunner of the GPL port, initialized at the start of
     * each stage (Min::init and setup), its timestep put back at the end of the stage (Min::cleanup)
     */
    const fire = reps[0].minStyle === 'fire';
    let fireRunner: FireRunner | null = null;
    const fireStart = (): void => {
      if (!fire) return;
      if (!fireRunner) {
        fireRunner = nebHook.fire(fireSettings(reps[0], etol, ftol),
          reps.map((sys) => ({ s: sys.state, mass: (i: number) => massOf(sys.state, i) })),
          () => { evaluateAll(false); applyNudging(); return energies.slice(); }, true);
      }
      fireRunner.init(reps[0].state.dt, reps[0].state.step);
      fireRunner.setupStyle();
      fireRunner.setEnergies(energies.slice());
    };
    const dtinit = reps[0].state.dt;
    const fireEnd = (): void => { if (fire) for (const sys of reps) sys.state.dt = dtinit; };
    const ftm2v = reps[0].state.units.ftm2v;

    /** Up to n iterations of one stage; a converged step is printed and ends the stage (native prints it too). */
    const stage = async (n: number): Promise<void> => {
      // Nevery counts from the start of the stage: measured with native LAMMPS (black box, hop1 cut down: the
      // climbing stage that starts at step 87 prints at 107 and 124, its convergence step)
      const start = reps[0].state.step;
      // the replicas' own thermo rows (output.cpp Output::setup and Output::write): the next row at the thermo
      // interval, both capped at the stage's last step (update->laststep = firststep + N1 or N2), and the last row
      // when the stage converges
      const every = reps[0].thermo.every;
      const stageLast = start + n;
      let nextThermo = every > 0 ? Math.min(Math.trunc(start / every) * every + every, stageLast) : stageLast;
      let lastThermo = -1;
      const replicaRows = (setup: boolean): void => { for (const sys of reps) host.replicaThermo?.(sys, setup); };
      if (host.replicaThermo) { replicaRows(true); lastThermo = start; }
      if (fireRunner) {
        // native NEB::run: while (niter < N) { minimize->run(nevery); print_status(); if (stop_condition) break; }
        const runner: FireRunner = fireRunner;
        while (runner.niter < n) {
          const before = runner.niter;
          const stop = await runner.iterate(nevery, {
            advance: () => { for (const sys of reps) sys.state.step++; return reps[0].state.step; },
            afterIteration: async (step) => {
              if (host.replicaThermo && step === nextThermo) {
                replicaRows(false);
                lastThermo = step;
                nextThermo = every > 0 ? Math.min(nextThermo + every, stageLast) : stageLast;
              }
            },
            cancelled: () => false,
          }, ftm2v);
          stepsDone += runner.niter - before;
          const step = reps[0].state.step;
          printRow(step);
          if (stop !== 'max iterations') {
            if (host.replicaThermo && lastThermo !== step) { replicaRows(false); lastThermo = step; }
            return;
          }
        }
        return;
      }
      for (let it = 0; it < n; it++) {
        quickminStep(reps);
        for (const sys of reps) sys.state.step++;
        evaluateAll(false);
        applyNudging();
        stepsDone++;
        const step = reps[0].state.step;
        const conv = converged();
        if ((step - start) % nevery === 0 || conv) printRow(step);
        if (host.replicaThermo) {
          if (step === nextThermo || (conv && lastThermo !== step)) { replicaRows(false); lastThermo = step; }
          if (step === nextThermo) nextThermo = every > 0 ? Math.min(nextThermo + every, stageLast) : stageLast;
        }
        if (conv) return;
        prevE = energies.slice();
      }
    };
    const forceMax = (): number => {
      let mx = 0;
      for (const sys of reps) mx = Math.max(mx, norm(sys.state.f, 0, 3 * sys.state.n));
      return mx;
    };

    // stage 1 (N1 iterations or convergence), then stage 2 with the climbing replica (N2 more), as in neb.rst
    fireStart();
    await stage(N1);
    fireEnd();
    // the highest replica climbs (fix_neb.rst: the climbing replica's inter-replica force is converted)
    let hi = 0;
    for (let r = 1; r < nReps; r++) if (energies[r] > energies[hi]) hi = r;
    climb = hi;
    host.emit({ kind: 'log', text: `Climbing replica = ${climb + 1}` });
    for (const sys of reps) sys.state.v.fill(0);
    applyNudging();
    header();
    printRow(reps[0].state.step);
    prevE = energies.slice();
    fireStart();
    await stage(N2);
    fireEnd();
  } finally {
    for (const f of fixes) f.active = false;
    reps.forEach((sys, r) => { sys.nb.every = saved[r].every; sys.nb.delay = saved[r].delay; sys.nb.check = saved[r].check; });
    // the quickmin velocities stay when the NEB ends (min.cpp: only the start zeroes them; min_quickmin.cpp leaves v)
  }
  host.emit({ kind: 'done', steps: stepsDone, seconds: (performance.now() - t0) / 1000, backend: host.backendLabel });
  return ebfRaw;
};

export const NEB_COMMANDS: Record<string, Handler> = {
  neb: async ({ session }, a) => { await runNeb(session, a); },
};
