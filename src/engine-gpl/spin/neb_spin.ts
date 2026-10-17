/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/neb_spin.cpp and src/SPIN/neb_spin.h
 * (NEBSpin, the neb/spin command: the file-style initial spins with Rodrigues' rotation between the end states,
 * the regular and climbing stages, and the print_status table), with the iteration of src/SPIN/min_spin.cpp
 * (MinSpin::iterate, evaluate_dt, advance_spins) run for every replica in lockstep. Contributing author (from
 * neb_spin.cpp): Julien Tranchida (SNL). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Quotes, docs.lammps.org/neb_spin.html: "Perform a geodesic nudged elastic band (GNEB) calculation using multiple
 * replicas of a system." and "a GNEB calculation performs a minimization across all the replicas." Native runs the
 * replicas as separate processes that exchange their energies and spins each iteration (fix_neb_spin.cpp); here the
 * replicas are advanced together in one loop, so each iteration first advances every replica's spins with one shared
 * timestep (min_spin.cpp evaluate_dt takes the largest precession frequency over all replicas), then evaluates every
 * replica's forces and energies, then applies the nudging of every replica.
 *
 * Supported: file-style final, each and none; the verbose keyword (the extra per-replica columns); min_style spin;
 * fix neb/spin with its kspring. Refused with a StyleError: min_style spin/cg and spin/lbfgs (native accepts them, with
 * the multireplica branches of min_spin_cg.cpp and min_spin_lbfgs.cpp, which are not ported here), any other
 * minimizer, atoms without spins, units other than metal, a single replica, and other trailing words.
 */
import type { NebHost } from '../../engine/commands/neb';
import { num, int, type RunApi } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { FixNEBSpin, type NebSpinRound } from './fix_neb_spin';
import { advanceSpins, minSpinTorqueNorm, spinMinSettings } from './min_spin';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

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
  const omega = Math.acos(sidotsf) * fraction;
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
    if (sys.minStyle === 'spin/cg' || sys.minStyle === 'spin/lbfgs') {
      throw new StyleError(`neb/spin with min_style ${sys.minStyle} is not supported by the browser engine yet (its multi-replica iteration is not ported); use min_style spin`);
    }
    if (sys.minStyle !== 'spin') throw new StyleError(`NEBSpin requires a spin minimizer (min_style spin); min_style is ${sys.minStyle}`);
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

  /** The minimizer of one stage (MinSpin::iterate with the multireplica tests): stops on the energy or torque criteria. */
  let niter = 0;
  let lastNegative = 0;
  const iterateStage = (maxiter: number): boolean => {
    for (let it = 0; it < maxiter; it++) {
      niter++;
      for (const sys of reps) sys.state.step++;
      const stepNow = reps[0].state.step;
      // the timestep follows the largest precession frequency over all replicas (evaluate_dt with multireplica)
      let fmaxsq = 0;
      for (const sys of reps) {
        const fm = sys.state.fm!;
        for (let i = 0; i < sys.state.n; i++) {
          const q = fm[3 * i] * fm[3 * i] + fm[3 * i + 1] * fm[3 * i + 1] + fm[3 * i + 2] * fm[3 * i + 2];
          if (q > fmaxsq) fmaxsq = q;
        }
      }
      if (fmaxsq === 0) throw new StyleError('Incorrect fmaxsqall calculation');
      const dts = (2 * Math.PI) / (discrete * Math.sqrt(fmaxsq));
      for (const sys of reps) advanceSpins(sys, alpha, dts);
      // min_spin.cpp compares ecurrent and eprevious, which Min::energy_force divides by the atom count under
      // thermo_modify norm yes: the energy test uses the normalized energies (as print_status does)
      const ePrev = normalized();
      evaluate(false);
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
