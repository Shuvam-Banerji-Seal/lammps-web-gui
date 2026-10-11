/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/fix_qeq_comb.cpp and src/MANYBODY/fix_qeq_comb.h (FixQEQComb: constructor,
 * init, setup, post_force, min_post_force, post_force_respa), at commit b332f28. Authors credited in
 * the C++ file headers: Ray Shan (Sandia). Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Refused (StyleError): the file keyword (the diagnostic log is not written by the browser engine),
 * and respa. The charge force comes from PairComb.chargeForces, which uses the neighbor data of the
 * last force evaluation of this step (the fix runs in postForce, after the forces, as in LAMMPS).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairComb } from './comb';

const QSUMSMALL = 0.00001;

/** fix qeq/comb (FixQEQComb): charge equilibration of pair_style comb. */
export class FixQeqComb extends Fix {
  readonly style = 'qeq/comb';

  private readonly nEvery: number;
  private readonly precision: number;
  /** firstflag: set during setup, where the iteration limit is 200 instead of 100. */
  private firstflag = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal fix qeq/comb command: usage fix ID group qeq/comb Nevery precision');
    this.nEvery = Math.trunc(Number(args[0]));
    this.precision = Number(args[1]);
    if (!(this.nEvery > 0) || !(this.precision > 0)) throw new StyleError('Illegal fix qeq/comb command: Nevery and precision must be positive');
    for (let k = 2; k < args.length; k++) {
      if (args[k] === 'file') throw new StyleError("fix qeq/comb keyword 'file' is not supported by the browser engine (no diagnostic file is written)");
      throw new StyleError(`Illegal fix qeq/comb command: unknown keyword '${args[k]}' (the browser engine supports only Nevery and precision)`);
    }
  }

  /** The pair style: pair_style comb. */
  private pair(): PairComb {
    const pair = this.sys.ff.pair;
    if (!(pair instanceof PairComb)) throw new StyleError('Must use pair_style comb with fix qeq/comb (pair_style comb3 is not supported by the browser engine)');
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
    const heatpq = 0.05;
    const qmass = 0.016;
    const dtq = 0.01;
    const dtq2 = 0.5 * dtq * dtq / qmass;
    const q1 = new Float64Array(n);
    const q2 = new Float64Array(n);
    const qf = new Float64Array(nall);
    for (let i = 0; i < n; i++) if (inGroup(i)) { q1[i] = 0; q2[i] = 0; }
    for (let iloop = 0; iloop < loopmax; iloop++) {
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
      if (enegchk <= this.precision && enegmax <= 100.0 * this.precision) break;
      for (let i = 0; i < n; i++) {
        if (inGroup(i)) q1[i] += qf[i] * dtq2 - heatpq * q1[i];
      }
    }
  }
}
