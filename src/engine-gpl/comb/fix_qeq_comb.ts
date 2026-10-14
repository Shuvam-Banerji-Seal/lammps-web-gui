/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/fix_qeq_comb.cpp and src/MANYBODY/fix_qeq_comb.h (FixQEQComb: constructor,
 * init, setup, post_force, min_post_force, post_force_respa), at commit b332f28. Authors credited in
 * the C++ file headers: Ray Shan (Sandia). Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The file keyword writes the same diagnostic lines as the C++ fix (fopen "w" at creation, then the
 * per-step messages through the engine's file output). Refused (StyleError): respa. The charge force comes from the pair style's chargeForces (PairComb or PairComb3), which uses the neighbor data of the
 * last force evaluation of this step (the fix runs in postForce, after the forces, as in LAMMPS).
 */
import { Fix } from '../../engine/fix/fix';
import { formatNumber } from '../../engine/cformat';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
/** The pair styles that give the charge forces of fix qeq/comb: pair_style comb (PairComb) and comb3 (PairComb3). */
export interface ChargeForcePair {
  /** Charge forces of the group into qf (length nall); returns the group's total (enegtot). */
  chargeForces(qf: Float64Array, isGroup: (i: number) => boolean): number;
}

const isChargeForcePair = (p: unknown): p is ChargeForcePair =>
  typeof p === 'object' && p !== null && typeof (p as ChargeForcePair).chargeForces === 'function';

const QSUMSMALL = 0.00001;

/** fix qeq/comb (FixQEQComb): charge equilibration of pair_style comb. */
export class FixQeqComb extends Fix {
  readonly style = 'qeq/comb';

  private readonly nEvery: number;
  private readonly precision: number;
  /** firstflag: set during setup, where the iteration limit is 200 instead of 100. */
  private firstflag = false;
  /** file keyword: the diagnostic log (src/MANYBODY/fix_qeq_comb.cpp at b332f28), or null. */
  private readonly fileName: string | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal fix qeq/comb command: usage fix ID group qeq/comb Nevery precision');
    this.nEvery = Math.trunc(Number(args[0]));
    this.precision = Number(args[1]);
    if (!(this.nEvery > 0) || !(this.precision > 0)) throw new StyleError('Illegal fix qeq/comb command: Nevery and precision must be positive');
    // fix_qeq_comb.cpp (b332f28): "file" takes a file name; the file is opened with "w" when the fix is created
    for (let k = 2; k < args.length;) {
      if (args[k] !== 'file' || k + 1 >= args.length) throw new StyleError(`Illegal fix qeq/comb command: unknown keyword '${args[k]}'`);
      this.fileName = args[k + 1];
      k += 2;
    }
    if (this.fileName !== null) sys.writeFile(this.fileName, '', false);
  }

  /** The pair style: pair_style comb or comb3. */
  private pair(): ChargeForcePair {
    const pair = this.sys.ff.pair;
    if (!isChargeForcePair(pair)) throw new StyleError('Must use pair_style comb or comb3 with fix qeq/comb');
    return pair;
  }

  /** Charges are equilibrated before the first force computation of a run (setup_pre_force path). */
  setup(): void {
    this.pair();
    this.firstflag = true;
    try {
      this.postForce();
    } finally {
      this.firstflag = false;
    }
  }

  minPostForce(): void { this.postForce(); }

  /** post_force: the iterative charge solve every Nevery steps. */
  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    if (s.step % this.nEvery !== 0) return;
    const pair = this.pair();
    const nb = sys.nb;
    const n = s.n;
    const nall = nb.nall;
    const bit = this.groupBit;
    const inGroup = (i: number): boolean => (s.mask[i] & bit) !== 0;
    let ngroup = 0;
    let qsum = 0;
    for (let i = 0; i < n; i++) if (inGroup(i)) { ngroup++; qsum += s.q[i]; }
    if (ngroup === 0) throw new StyleError('Fix qeq/comb group has no atoms');
    if (this.firstflag && Math.abs(qsum) > QSUMSMALL) {
      sys.warn(`Fix ${this.style} group is not charge neutral, net charge = ${qsum.toPrecision(8)}`);
    }
    const loopmax = this.firstflag ? 200 : 100;
    const log = (text: string): void => { if (this.fileName !== null) sys.writeFile(this.fileName, text, true); };
    const g6 = (v: number): string => formatNumber(v, '%.6g');
    log(`Charge equilibration on step ${s.step}\n`);
    const heatpq = 0.05;
    const qmass = 0.016;
    const dtq = 0.01;
    const dtq2 = 0.5 * dtq * dtq / qmass;
    const q1 = new Float64Array(n);
    const q2 = new Float64Array(n);
    const qf = new Float64Array(nall);
    for (let i = 0; i < n; i++) if (inGroup(i)) { q1[i] = 0; q2[i] = 0; }
    let iloop = 0;
    let enegchkLast = 0;
    for (; iloop < loopmax; iloop++) {
      for (let i = 0; i < n; i++) {
        if (inGroup(i)) {
          q1[i] += qf[i] * dtq2 - heatpq * q1[i];
          s.q[i] += q1[i];
        }
      }
      nb.refreshCharges(s); // comm->forward_comm: ghost charges follow the owners
      let enegtot = pair.chargeForces(qf, inGroup);
      enegtot /= ngroup;
      let enegchk = 0.0, enegmax = 0.0;
      for (let i = 0; i < n; i++) {
        if (inGroup(i)) {
          q2[i] = enegtot - qf[i];
          enegmax = Math.max(enegmax, Math.abs(q2[i]));
          enegchk += Math.abs(q2[i]);
          qf[i] = q2[i];
        }
      }
      enegchk = enegchk / ngroup;
      enegchkLast = enegchk;
      if (enegchk <= this.precision && enegmax <= 100.0 * this.precision) break;
      log(`  iteration: ${iloop}, enegtot ${g6(enegtot)}, enegmax ${g6(enegmax)}, fq deviation: ${g6(enegchk)}\n`);
      for (let i = 0; i < n; i++) {
        if (inGroup(i)) q1[i] += qf[i] * dtq2 - heatpq * q1[i];
      }
    }
    if (iloop === loopmax) log(`Charges did not converge in ${iloop} iterations\n`);
    else log(`Charges converged in ${iloop} iterations to ${formatNumber(enegchkLast, '%.10f')} tolerance\n`);
  }
}
