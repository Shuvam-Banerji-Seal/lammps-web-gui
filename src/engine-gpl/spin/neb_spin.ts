/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/neb_spin.cpp and src/SPIN/neb_spin.h
 * (NEBSpin, the neb/spin command: the file-style initial spins with Rodrigues' rotation between the end states,
 * the regular and climbing stages, and the print_status table), with the iteration of src/SPIN/min_spin.cpp
 * (MinSpin::iterate, evaluate_dt, advance_spins), src/SPIN/min_spin_cg.cpp (MinSpinCG::iterate, calc_gradient,
 * calc_search_direction, evaluate_dt) and src/SPIN/min_spin_lbfgs.cpp (MinSpinLBFGS::iterate, calc_gradient,
 * calc_search_direction, maximum_rotation) run for every replica in lockstep. Contributing authors (from neb_spin.cpp
 * and min_spin_cg.cpp / min_spin_lbfgs.cpp): Julien Tranchida (SNL), Aleksei Ivanov (University of Iceland). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Quotes, docs.lammps.org/neb_spin.html: "Perform a geodesic nudged elastic band (GNEB) calculation using multiple
 * replicas of a system." and "a GNEB calculation performs a minimization across all the replicas." Native runs the
 * replicas as separate processes that exchange their energies and spins each iteration (fix_neb_spin.cpp); here the
 * replicas are advanced together in one loop, so each iteration first advances every replica's spins with one shared
 * timestep (min_spin.cpp evaluate_dt takes the largest precession frequency over all replicas), then evaluates every
 * replica's forces and energies, then applies the nudging of every replica.
 *
 * Supported: file-style final, each and none; the verbose keyword (the extra per-replica columns); min_style spin and
 * spin/cg (its multireplica branch: the end replicas keep their spins, the sums of the conjugate direction run over all
 * replicas, and the timestep follows the largest torque of all replicas; with several replicas native never uses its
 * line search); fix neb/spin with its kspring. Refused with a StyleError: min_style spin/lbfgs (its multireplica
 * iteration is written below but does not yet match native, see the check), any other minimizer, atoms without spins,
 * units other than metal, a single replica, and other trailing words.
 */
import type { NebHost } from '../../engine/commands/neb';
import { num, int, type RunApi } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { FixNEBSpin, type NebSpinRound } from './fix_neb_spin';
import { advanceSpins, minSpinTorqueNorm, spinMinSettings } from './min_spin';
import { rodrigues, applyRot } from './min_spin_cg';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { acos as libmAcos } from '../libm/acos';

/** min_spin.cpp: DELAYSTEP (the energy test starts after it) and EPS_ENERGY (the energy test's normalization floor) */
const DELAYSTEP = 5;
const EPS_ENERGY = 1e-8;

/** One line of a neb/spin file: "ID musp x y z spx spy spz" (neb_spin.cpp ATTRIBUTE_PERLINE = 8). */
interface SpinRecord { id: number; musp: number; x: number[]; sp: number[] }

/**
 * The atoms of a neb/spin file: the first non-blank, non-comment line is the number of atoms, then that many lines of
 * eight values. Native reads the same layout (neb_spin.cpp readfile). Blank and comment lines are skipped here (native
 * counts a blank line as an atom line and fails on it).
 */
export const parseSpinFile = (text: string): SpinRecord[] => {
  const lines = text.split('\n').map((l) => l.replace(/#.*$/, '').trim()).filter((l) => l !== '');
  if (!lines.length) throw new StyleError('Unexpected end of neb/spin file');
  const count = Number(lines[0].split(/\s+/)[0]);
  if (!Number.isInteger(count) || count < 0) throw new StyleError(`Incorrectly formatted NEB file: first line must be the atom count, got '${lines[0]}'`);
  if (lines.length < count + 1) throw new StyleError(`Unexpected end of neb/spin file: ${count} atoms announced, ${lines.length - 1} lines found`);
  const out: SpinRecord[] = [];
  for (let k = 1; k <= count; k++) {
    const w = lines[k].split(/\s+/);
    if (w.length !== 8) throw new StyleError(`Incorrect atom format in neb/spin file: '${lines[k]}' (need ID musp x y z spx spy spz)`);
    const v = w.map(Number);
    if (v.some((x) => !Number.isFinite(x)) || !Number.isInteger(v[0])) throw new StyleError(`Incorrectly formatted NEB file: '${lines[k]}'`);
    out.push({ id: v[0], musp: v[1], x: [v[2], v[3], v[4]], sp: [v[5], v[6], v[7]] });
  }
  return out;
};

/**
 * Rodrigues' rotation of the unit spin spi toward spf by the fraction of the angle between them (neb_spin.cpp
 * initial_rotation). Returns the rotated spin; rot is true when the two spins are antiparallel and an arbitrary rotation
 * axis was chosen (native warns "arbitrary initial rotation of one or more spin(s)").
 */
export const initialRotation = (spi: readonly number[], spf: readonly number[], fraction: number): { sp: number[]; rot: boolean } => {
  if (fraction === 0 || fraction === 1) return { sp: [spf[0], spf[1], spf[2]], rot: false };
  const [spix, spiy, spiz] = spi;
  const [spfx, spfy, spfz] = spf;
  let kx = spiy * spfz - spiz * spfy;
  let ky = spiz * spfx - spix * spfz;
  let kz = spix * spfy - spiy * spfx;
  let knormsq = kx * kx + ky * ky + kz * kz;
  const sidotsf = spix * spfx + spiy * spfy + spiz * spfz;
  let rot = false;
  if (knormsq === 0) {
    if (sidotsf > 0) return { sp: [spf[0], spf[1], spf[2]], rot: false };
    if (sidotsf < 0) {
      // antiparallel: the rotation axis is spi x [100], or spi x [010] when spi lies along [100]
      if (spiy * spiy + spiz * spiz !== 0) {
        kx = 0; ky = spiz; kz = -spiy;
        knormsq = ky * ky + kz * kz;
      } else if (spix * spix + spiz * spiz !== 0) {
        kx = -spiz; ky = 0; kz = spix;
        knormsq = kx * kx + kz * kz;
      } else throw new StyleError('Incorrect initial rotation operation');
      rot = true;
    }
  }
  if (knormsq === 0) throw new StyleError('Incorrect initial rotation operation');
  const iknorm = 1 / Math.sqrt(knormsq);
  kx *= iknorm; ky *= iknorm; kz *= iknorm;
  const kcrossx = ky * spiz - kz * spiy;
  const kcrossy = kz * spix - kx * spiz;
  const kcrossz = kx * spiy - ky * spix;
  const kdots = kx * spix + ky * spiy + kz * spiz;
  const omega = libmAcos(sidotsf) * fraction;
  const c = libmCos(omega), s = libmSin(omega);
  let spkx = spix * c + kcrossx * s + kx * kdots * (1 - c);
  let spky = spiy * c + kcrossy * s + ky * kdots * (1 - c);
  let spkz = spiz * c + kcrossz * s + kz * kdots * (1 - c);
  const isnorm = 1 / Math.sqrt(spkx * spkx + spky * spky + spkz * spkz);
  if (isnorm === 0) throw new StyleError('Incorrect initial rotation operation');
  spkx *= isnorm; spky *= isnorm; spkz *= isnorm;
  return { sp: [spkx, spky, spkz], rot };
};

/** Forces and energies of one replica at its current positions and spins (the first part of Min::energy_force). */
const forcesOf = (sys: System, first: boolean): void => {
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
};

/** Torque norms of the printed table: the squared torque sum and the largest torque component over all atoms. */
const torqueMetrics = (sys: System): { sq: number; inf: number } => {
  const s = sys.state;
  const fm = s.fm!, sp = s.sp!;
  let sq = 0, inf = 0;
  for (let i = 0; i < s.n; i++) {
    const fx = fm[3 * i], fy = fm[3 * i + 1], fz = fm[3 * i + 2];
    const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
    const t = [fy * sz - fz * sy, fz * sx - fx * sz, fx * sy - fy * sx];
    sq += t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    for (const c of t) if (Math.abs(c) > inf) inf = Math.abs(c);
  }
  return { sq, inf };
};

/** The table columns of a neb/spin run (neb_spin.cpp print_status and its header; verbose adds the per-replica pairs). */
const tableKeywords = (n: number, verbose: boolean): string[] => {
  const k = ['Step', 'MaxReplicaTorque', 'MaxAtomTorque', 'GradV0', 'GradV1', 'GradVc', 'EBF', 'EBR', 'RDT'];
  for (let r = 1; r <= n; r++) k.push(`RD${r}`, `PE${r}`);
  if (verbose) for (let r = 0; r < n; r++) k.push(`GradV${r}dottan`, `DN${r}`);
  return k;
};

/** neb/spin command: the two stages of the geodesic NEB calculation (neb_spin.cpp NEBSpin::command and run). */
export const runNebSpin = async (host: NebHost, run: Pick<RunApi, 'ensureThermoComputes' | 'initRun'>, a: string[]): Promise<void> => {
  const reps = [...host.replicaSystems];
  if (a.length < 6) throw new StyleError('usage: neb/spin etol ttol N1 N2 Nevery file-style [arg] [verbose]');
  const etol = num(a[0], 'etol'), ttol = num(a[1], 'ttol');
  const N1 = int(a[2], 'N1'), N2 = int(a[3], 'N2'), nevery = int(a[4], 'Nevery');
  if (etol < 0 || ttol < 0) throw new StyleError('Illegal neb/spin command: etol and ttol must be >= 0');
  if (nevery <= 0) throw new StyleError('Illegal neb/spin command: Nevery must be > 0');
  if (N1 % nevery !== 0 || N2 % nevery !== 0) throw new StyleError('Illegal neb/spin command: N1 and N2 must be multiples of Nevery');
  if (reps.length < 2) throw new StyleError('Cannot use NEBSpin with a single replica (neb/spin needs two or more partitions)');
  const nRep = reps.length;

  // file-style: final (one file, the intermediate replicas interpolated), each (a file per replica) or none
  const style = a[5];
  let coordFile: string | null = null;
  const eachFile: (string | null)[] = reps.map(() => null);
  let k = 6;
  if (style === 'final') {
    if (a[6] === undefined) throw new StyleError('Illegal neb/spin command: final needs a file name');
    coordFile = a[6];
    k = 7;
  } else if (style === 'each') {
    for (let r = 1; r < nRep; r++) {
      const name = (host.argsOf(r) ?? a)[6];
      if (name === undefined) throw new StyleError('Illegal neb/spin command: each needs a file name');
      eachFile[r] = name;
    }
    k = 7;
  } else if (style !== 'none') throw new StyleError(`Illegal neb/spin command: file-style must be final, each or none (got '${style}')`);
  let verbose = false;
  for (; k < a.length; k++) {
    if (a[k] === 'verbose' && k === a.length - 1) verbose = true;
    else throw new StyleError(`illegal neb/spin keyword '${a[k]}' (the browser engine accepts only verbose, as the last word)`);
  }

  // every replica: metal units, atoms with spins, the damped spin minimizer (min_style spin), one fix neb/spin
  const fixes: FixNEBSpin[] = [];
  for (const sys of reps) {
    if (!sys.hasBox) throw new StyleError('NEBSpin command before simulation box is defined');
    if (!sys.state.sp || !sys.state.fm) throw new StyleError('neb/spin requires atom/spin style');
    if (sys.units.style !== 'metal') throw new StyleError('neb/spin simulation requires metal unit style');
    // neb_spin.cpp: a line-search minimizer, then a minimizer whose style does not start with spin, are refused
    if (sys.minStyle === 'cg' || sys.minStyle === 'sd' || sys.minStyle === 'hftn') throw new StyleError('NEBSpin requires damped dynamics minimizer');
    if (sys.minStyle !== 'spin' && sys.minStyle !== 'spin/cg' && sys.minStyle !== 'spin/lbfgs') throw new StyleError('NEBSpin requires a spin minimizer');
    // the multireplica L-BFGS below follows min_spin_lbfgs.cpp, but on tests/oracle/w194gneb_chain.in it leaves native's
    // table from iteration 59 on (1.4e-7 in RD2, 1e-4 by iteration 120), which neither the replica sum order nor rounding
    // explains (measured with native LAMMPS); until that is found, it is refused by name
    if (sys.minStyle === 'spin/lbfgs') {
      throw new StyleError('neb/spin with min_style spin/lbfgs is not supported by the browser engine yet (its multi-replica iteration does not yet match native LAMMPS); use min_style spin or spin/cg');
    }
    const nebs = sys.fixes.filter((f): f is FixNEBSpin => f instanceof FixNEBSpin);
    if (nebs.length !== 1) throw new StyleError('NEBSpin requires use of exactly one fix neb/spin instance');
    fixes.push(nebs[0]);
  }
  const n0 = reps[0].state.n;
  for (const sys of reps) {
    if (sys.state.n !== n0) throw new StyleError('neb/spin needs the same number of atoms in every replica');
    for (let i = 0; i < n0; i++) {
      if (sys.indexOfId(reps[0].state.id[i]) < 0) throw new StyleError(`neb/spin: atom ${reps[0].state.id[i]} is not in every replica`);
    }
  }

  // initial configurations (neb_spin.cpp readfile). final: the spins of intermediate replicas rotate from their own
  // spins toward the file's, by the fraction r / (N - 1) of the way; each: every replica after the first reads its own
  // file (positions and spins); none: nothing is read. The file's magnitudes are copied to the replicas.
  if (coordFile !== null) {
    const recs = parseSpinFile(reps[0].readFile(coordFile));
    let rotated = false;
    for (let r = 0; r < nRep; r++) {
      const s = reps[r].state;
      const fraction = r / (nRep - 1.0);
      for (const rec of recs) {
        const m = reps[r].indexOfId(rec.id);
        if (m < 0) throw new StyleError(`Invalid atom IDs in neb/spin file: atom ${rec.id} is not in replica ${r + 1}`);
        s.sp![4 * m + 3] = rec.musp;
        if (fraction === 0) continue;
        if (fraction === 1) {
          for (let d = 0; d < 3; d++) s.sp![4 * m + d] = rec.sp[d];
          continue;
        }
        const spinit = [s.sp![4 * m], s.sp![4 * m + 1], s.sp![4 * m + 2]];
        const res = initialRotation(spinit, rec.sp, fraction);
        rotated = rotated || res.rot;
        for (let d = 0; d < 3; d++) s.sp![4 * m + d] = res.sp[d];
      }
    }
    if (rotated) reps[0].warn('arbitrary initial rotation of one or more spin(s)');
  }
  for (let r = 1; r < nRep; r++) {
    const name = eachFile[r];
    if (name === null) continue;
    const s = reps[r].state;
    for (const rec of parseSpinFile(reps[0].readFile(name))) {
      const m = reps[r].indexOfId(rec.id);
      if (m < 0) throw new StyleError(`Invalid atom IDs in neb/spin file ${name}: atom ${rec.id} is not in replica ${r + 1}`);
      s.sp![4 * m + 3] = rec.musp;
      for (let d = 0; d < 3; d++) {
        s.x[3 * m + d] = rec.x[d];
        s.sp![4 * m + d] = rec.sp[d];
      }
    }
    reps[r].pbc();
  }

  for (const sys of reps) { run.ensureThermoComputes(sys); run.initRun(sys); }
  const { alpha, discrete } = spinMinSettings(reps[0]);
  const nb0 = reps.map((sys) => ({ every: sys.nb.every, delay: sys.nb.delay, check: sys.nb.check }));
  for (const sys of reps) {
    if (sys.nb.every !== 1 || sys.nb.delay !== 0) sys.log('Switching to \'neigh_modify every 1 delay 0 check yes\' setting during minimization');
    sys.nb.every = 1; sys.nb.delay = 0; sys.nb.check = true;
  }

  const neb0 = fixes[0];
  const units = reps[0].units.style;
  const cols = tableKeywords(nRep, verbose);
  /** energies of the replicas (raw, eV): the setup and every iteration's evaluation */
  const E: number[] = new Array(nRep).fill(0);
  const round: NebSpinRound = { systems: reps, energies: E };
  const before = (r: number): Fix[] => reps[r].fixes.slice(0, reps[r].fixes.indexOf(fixes[r]));
  const after = (r: number): Fix[] => reps[r].fixes.slice(reps[r].fixes.indexOf(fixes[r]) + 1);

  /**
   * One energy and force evaluation of every replica (Min::energy_force of each replica, with the nudging of fix neb/spin
   * between the fixes before it and the fixes after it). The energy of a replica is its pair, bond and kspace energy and
   * the energies of the fixes before fix neb/spin (native's pe compute at that point of the fix order).
   */
  const evaluate = (first: boolean): void => {
    for (const sys of reps) forcesOf(sys, first);
    if (first) for (const sys of reps) sys.assignDynamicGroups(true);
    for (let r = 0; r < nRep; r++) for (const f of before(r)) f.minPostForce?.();
    for (let r = 0; r < nRep; r++) {
      const acc = reps[r].ff.acc;
      let e = acc.evdwl + acc.ecoul + acc.elong + acc.ebond + acc.eangle + acc.edihed + acc.eimp;
      for (const f of before(r)) if (f.thermoEnergy) e += reps[r].fixEnergyOf(f);
      E[r] = e;
    }
    for (let r = 0; r < nRep; r++) fixes[r].postForceReplica(r, round);
    for (let r = 0; r < nRep; r++) for (const f of after(r)) f.minPostForce?.();
    for (const sys of reps) sys.refreshComputes();
  };

  /** The energies as print_status stores them: divided by the atom count when thermo norm is on. */
  const normalized = (): number[] => {
    const n = reps[0].state.n;
    return E.map((e) => (reps[0].thermo.norm && n > 0 ? e / n : e));
  };

  /** The replica with the highest energy (print_status: strict comparisons, the first of equal maxima). */
  const highest = (): number => {
    const pe = normalized();
    let top = 0;
    for (let r = 1; r < nRep; r++) if (pe[top] < pe[r]) top = r;
    return top;
  };

  /** Table row of print_status: the per-replica values of the fixes, the reaction distances and the energies. */
  const printRow = (step: number): void => {
    const pe = normalized();
    let maxSq = 0, maxAtom = 0;
    for (const sys of reps) {
      const t = torqueMetrics(sys);
      if (t.sq > maxSq) maxSq = t.sq;
      if (t.inf > maxAtom) maxAtom = t.inf;
    }
    const irep = neb0.rclimber > -1 ? neb0.rclimber : highest();
    // the cumulative reaction distances: RD_r = the path lengths of replicas 1..r, normalized by the total RDT
    const rdist: number[] = new Array(nRep).fill(0);
    for (let r = 1; r < nRep; r++) rdist[r] = rdist[r - 1] + fixes[r].plen;
    const endpt = (rdist[nRep - 1] = rdist[nRep - 2] + fixes[nRep - 2].nlen);
    for (let r = 1; r < nRep; r++) rdist[r] /= endpt;
    const row: Record<string, number> = {
      Step: step,
      MaxReplicaTorque: maxSq,
      MaxAtomTorque: maxAtom,
      GradV0: fixes[0].gradlen,
      GradV1: fixes[nRep - 1].gradlen,
      GradVc: fixes[irep].gradlen,
      EBF: pe[irep] - pe[0],
      EBR: pe[irep] - pe[nRep - 1],
      RDT: endpt,
    };
    for (let r = 0; r < nRep; r++) {
      row[`RD${r + 1}`] = rdist[r];
      row[`PE${r + 1}`] = pe[r];
    }
    if (verbose) {
      // native prints (nlen, dottangrad) of each replica, and (NaN, dottangrad) for the last one; the header's labels
      // name the pairs GradV<r>dottan and DN<r> in that printed order, so the value under GradV<r>dottan is the nlen
      for (let r = 0; r < nRep; r++) {
        row[`GradV${r}dottan`] = r < nRep - 1 ? fixes[r].nlen : NaN;
        row[`DN${r}`] = fixes[r].dottangrad;
      }
    }
    host.emit({ kind: 'thermo', row });
  };

  const header = (): void => host.emit({ kind: 'thermo-header', keywords: cols, labels: cols, units });

  /**
   * The minimizer of one stage (the multireplica branches of MinSpin, MinSpinCG and MinSpinLBFGS::iterate): stops on the
   * energy or torque criteria, which every replica must meet.
   */
  let niter = 0;
  let lastNegative = 0;
  const minStyle = reps[0].minStyle;
  const n = reps[0].state.n;
  const len = 3 * n;
  /** the end replicas keep their spins: their search direction is multiplied by 0 (calc_search_direction) */
  const keep = reps.map((_, r) => (r === 0 || r === nRep - 1 ? 0.0 : 1.0));
  const gCur = reps.map(() => new Float64Array(len));
  const gOld = reps.map(() => new Float64Array(len));
  const pS = reps.map(() => new Float64Array(len));
  /** min_spin_lbfgs.cpp: num_mem = 3 and the memory of each replica */
  const NUM_MEM = 3;
  const ds = reps.map(() => Array.from({ length: NUM_MEM }, () => new Float64Array(len)));
  const dy = reps.map(() => Array.from({ length: NUM_MEM }, () => new Float64Array(len)));
  const rho = new Float64Array(NUM_MEM);
  const hbar = reps[0].state.units.hplanck / (2 * Math.PI);
  // min_spin_lbfgs.cpp modify_param: maxepsrot = MY_2PI / (10 * discrete_factor)
  const maxepsrot = (2 * Math.PI) / (10 * discrete);
  const rot = new Float64Array(9);
  /** local_iter of MinSpinCG / MinSpinLBFGS: 0 at each stage's init, the same on every replica */
  let localIter = 0;

  /** evaluate_dt with multireplica: the spin timestep of the largest precession frequency over all replicas */
  const evaluateDt = (): number => {
    let fmaxsq = 0;
    for (const sys of reps) {
      const fm = sys.state.fm!;
      for (let i = 0; i < sys.state.n; i++) {
        const q = fm[3 * i] * fm[3 * i] + fm[3 * i + 1] * fm[3 * i + 1] + fm[3 * i + 2] * fm[3 * i + 2];
        if (q > fmaxsq) fmaxsq = q;
      }
    }
    if (fmaxsq === 0) throw new StyleError('Incorrect fmaxsqall calculation');
    return (2 * Math.PI) / (discrete * Math.sqrt(fmaxsq));
  };

  /** calc_gradient of every replica: g_i = (fm_i x s_i) times factor (the timestep for spin/cg, hbar for spin/lbfgs). */
  const calcGradient = (factor: number): void => {
    reps.forEach((sys, r) => {
      const fm = sys.state.fm!, sp = sys.state.sp!, g = gCur[r];
      for (let i = 0; i < n; i++) {
        const fx = fm[3 * i], fy = fm[3 * i + 1], fz = fm[3 * i + 2];
        const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
        g[3 * i] = (fx * sy - fy * sx) * factor;
        g[3 * i + 1] = -(fz * sx - fx * sz) * factor;
        g[3 * i + 2] = (fy * sz - fz * sy) * factor;
      }
    });
  };

  /** A sum over the replicas of each replica's value times its factor (the uworld Allreduce of the source). */
  const replicaSum = (f: (r: number) => number): number => {
    let t = 0;
    for (let r = 0; r < nRep; r++) t += f(r) * keep[r];
    return t;
  };

  /** MinSpinCG::calc_search_direction with nreplica > 1 (Fletcher-Reeves, steepest descent every fifth iteration). */
  const cgDirection = (): void => {
    if (localIter === 0 || localIter % 5 === 0) {
      for (let r = 0; r < nRep; r++) {
        for (let i = 0; i < len; i++) { pS[r][i] = -gCur[r][i] * keep[r]; gOld[r][i] = gCur[r][i] * keep[r]; }
      }
    } else {
      const dot = (u: Float64Array, v: Float64Array) => { let t = 0; for (let i = 0; i < len; i++) t += u[i] * v[i]; return t; };
      const g2 = replicaSum((r) => dot(gCur[r], gCur[r]));
      const g2old = replicaSum((r) => dot(gOld[r], gOld[r]));
      const beta = Math.abs(g2) < 1.0e-60 ? 0.0 : g2 / g2old;
      for (let r = 0; r < nRep; r++) {
        for (let i = 0; i < len; i++) { pS[r][i] = (beta * pS[r][i] - gCur[r][i]) * keep[r]; gOld[r][i] = gCur[r][i] * keep[r]; }
      }
    }
    localIter++;
  };

  /** maximum_rotation with nreplica > 1: the norm and the spin count over all replicas (the end replicas included). */
  const maximumRotation = (v: Float64Array[]): number => {
    let norm2 = 0;
    for (let r = 0; r < nRep; r++) { let t = 0; for (let i = 0; i < len; i++) t += v[r][i] * v[r][i]; norm2 += t; }
    const scaling = maxepsrot * Math.sqrt((n * nRep) / norm2);
    return scaling < 1.0 ? scaling : 1.0;
  };

  /** MinSpinLBFGS::calc_search_direction with nreplica > 1 (the two-loop recursion with sums over the replicas). */
  const lbfgsDirection = (): void => {
    if (localIter === 0) {
      const scaling = maximumRotation(gCur);
      for (let r = 0; r < nRep; r++) {
        for (let i = 0; i < len; i++) {
          pS[r][i] = -gCur[r][i] * keep[r] * scaling;
          gOld[r][i] = gCur[r][i] * keep[r];
          for (let k = 0; k < NUM_MEM; k++) { ds[r][k][i] = 0.0; dy[r][k][i] = 0.0; }
        }
      }
      rho.fill(0);
    } else {
      const m = localIter % NUM_MEM;
      const dyds = replicaSum((r) => {
        let t = 0;
        for (let i = 0; i < len; i++) {
          ds[r][m][i] = pS[r][i];
          dy[r][m][i] = gCur[r][i] - gOld[r][i];
          t += ds[r][m][i] * dy[r][m][i];
        }
        return t;
      });
      rho[m] = Math.abs(dyds) > 1.0e-60 ? 1.0 / dyds : 1.0e60;
      if (rho[m] < 0.0) {
        localIter = 0;
        lbfgsDirection();
        return;
      }
      const q = gCur.map((g) => Float64Array.from(g));
      const alpha = new Float64Array(NUM_MEM);
      for (let k = NUM_MEM - 1; k > -1; k--) {
        const c = (k + m + 1) % NUM_MEM;
        const sq = replicaSum((r) => { let t = 0; for (let i = 0; i < len; i++) t += ds[r][c][i] * q[r][i]; return t; });
        alpha[c] = rho[c] * sq;
        for (let r = 0; r < nRep; r++) for (let i = 0; i < len; i++) q[r][i] -= alpha[c] * dy[r][c][i];
      }
      const yy = replicaSum((r) => { let t = 0; for (let i = 0; i < len; i++) t += dy[r][m][i] * dy[r][m][i]; return t; });
      const devis = rho[m] * yy;
      for (let r = 0; r < nRep; r++) {
        if (Math.abs(devis) > 1.0e-60) for (let i = 0; i < len; i++) pS[r][i] = keep[r] * q[r][i] / devis;
        else for (let i = 0; i < len; i++) pS[r][i] = keep[r] * q[r][i] * 1.0e60;
      }
      for (let k = 0; k < NUM_MEM; k++) {
        const c = localIter < NUM_MEM ? k : (k + m + 1) % NUM_MEM;
        const yr = replicaSum((r) => { let t = 0; for (let i = 0; i < len; i++) t += dy[r][c][i] * pS[r][i]; return t; });
        const beta = rho[c] * yr;
        for (let r = 0; r < nRep; r++) for (let i = 0; i < len; i++) pS[r][i] += ds[r][c][i] * (alpha[c] - beta);
      }
      const scaling = maximumRotation(pS);
      for (let r = 0; r < nRep; r++) {
        for (let i = 0; i < len; i++) { pS[r][i] = -keep[r] * pS[r][i] * scaling; gOld[r][i] = gCur[r][i] * keep[r]; }
      }
    }
    localIter++;
  };

  /** advance_spins of spin/cg and spin/lbfgs: every spin rotates about its search direction (Rodrigues' formula). */
  const rotateSpins = (): void => {
    reps.forEach((sys, r) => {
      const sp = sys.state.sp!;
      for (let i = 0; i < n; i++) {
        rodrigues(pS[r], 3 * i, rot);
        applyRot(rot, sp, 4 * i);
      }
    });
  };

  const iterateStage = (maxiter: number): boolean => {
    for (let it = 0; it < maxiter; it++) {
      niter++;
      for (const sys of reps) sys.state.step++;
      const stepNow = reps[0].state.step;
      // min_spin.cpp compares ecurrent and eprevious, which Min::energy_force divides by the atom count under
      // thermo_modify norm yes: the energy test uses the normalized energies (as print_status does)
      const ePrev = normalized();
      if (minStyle === 'spin') {
        // the timestep follows the largest precession frequency over all replicas (evaluate_dt with multireplica)
        const dts = evaluateDt();
        for (const sys of reps) advanceSpins(sys, alpha, dts);
        evaluate(false);
      } else if (minStyle === 'spin/cg') {
        // MinSpinCG::iterate without a line search: gradient, direction, rotation, then energy and forces
        calcGradient(evaluateDt());
        cgDirection();
        rotateSpins();
        evaluate(false);
      } else {
        // MinSpinLBFGS::iterate without a line search: energy and forces first, then gradient, direction and rotation
        evaluate(false);
        calcGradient(hbar);
        lbfgsDirection();
        rotateSpins();
      }
      const eNow = normalized();
      // the energy test after DELAYSTEP iterations and the torque test: every replica must meet them
      if (etol > 0 && stepNow - lastNegative > DELAYSTEP) {
        let notConverged = 0;
        for (let r = 0; r < nRep; r++) {
          if (!(Math.abs(eNow[r] - ePrev[r]) < etol * 0.5 * (Math.abs(eNow[r]) + Math.abs(ePrev[r]) + EPS_ENERGY))) notConverged++;
        }
        if (notConverged === 0) return true;
      }
      if (ttol > 0) {
        let notConverged = 0;
        for (const sys of reps) {
          const fmsq = minSpinTorqueNorm(sys, sys.minSettings.norm);
          if (!(fmsq * fmsq < ttol * ttol)) notConverged++;
        }
        if (notConverged === 0) return true;
      }
    }
    return false;
  };

  const t0 = performance.now();
  let steps = 0;
  for (const f of fixes) f.active = true;
  try {
    // stage 1: the regular GNEB minimization of N1 iterations (or until converged); no climbing replica yet
    for (const sys of reps) if (sys.state.v) sys.state.v.fill(0);
    for (const f of fixes) f.rclimber = -1;
    evaluate(true);
    header();
    printRow(reps[0].state.step);
    niter = 0;
    lastNegative = reps[0].state.step;
    localIter = 0;
    while (niter < N1) {
      const stop = iterateStage(nevery);
      printRow(reps[0].state.step);
      if (stop) break;
    }
    steps += niter;

    // the replica with the highest energy becomes the climber (native: top over the last printed energies)
    const top = highest();
    host.emit({ kind: 'log', text: `Climbing replica = ${top + 1}` });
    for (const f of fixes) f.rclimber = top;

    // stage 2: the climbing minimization of N2 iterations; the minimizer's setup is evaluated again with the climber
    evaluate(true);
    header();
    printRow(reps[0].state.step);
    niter = 0;
    lastNegative = reps[0].state.step;
    localIter = 0;
    while (niter < N2) {
      const stop = iterateStage(nevery);
      printRow(reps[0].state.step);
      if (stop) break;
    }
    steps += niter;
  } finally {
    for (const f of fixes) { f.active = false; f.clearNudging(); }
    reps.forEach((sys, r) => { sys.nb.every = nb0[r].every; sys.nb.delay = nb0[r].delay; sys.nb.check = nb0[r].check; });
  }
  host.emit({ kind: 'done', steps, seconds: (performance.now() - t0) / 1000, backend: host.backendLabel });
};
