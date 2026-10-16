/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/fix_alchemy.cpp and src/REPLICA/fix_alchemy.h (FixAlchemy: the constructor's checks, init,
 * setup, post_integrate, post_force, compute_scalar, compute_vector and extract) at commit b332f28. Authors of the
 * source: the LAMMPS REPLICA package (Axel Kohlmeyer, Temple University). Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Partitions: the two replicas of the session (System.replicaPeers) are the two partitions of native LAMMPS. Native
 * exchanges the positions, forces and energies with MPI between the partitions; here the replicas share a process, so
 * the fix of the first replica (the leader) does the exchange for both, in the phases where the exchange happens
 * (run/coupled.ts drives the replicas in lockstep). The fix of the second replica only reads the results.
 *
 * Native order of the exchange, kept here:
 *  - post_integrate: the owned atom count and ordering are checked, the positions of partition 0 are copied to
 *    partition 1 (MPI_Bcast from iworld 0) and the box is copied when it differs. Velocities are not exchanged
 *    (fix_alchemy.cpp broadcasts only atom->x).
 *  - post_force: every partition's forces are multiplied by its own lambda and summed over the partitions, so both
 *    partitions get the same force f = sum_r lambda_r f_r. The potential energies (epot[0], epot[1] per partition and
 *    epot[2] = sum_r lambda_r pe_r) and the pressure tensor (sum_r lambda_r P_r, the pressure compute of each
 *    partition) are summed the same way.
 *  - setup: post_integrate, then post_force (the forces of the setup step are mixed as well).
 *
 * fix ID group-ID alchemy v_lambda (docs.lammps.org/fix_alchemy.html, the variable is evaluated on each partition).
 * Not supported (StyleError): a group other than all, anything but one variable argument, more than one alchemy fix,
 * respa, and a load balancer.
 */
import { StyleError } from '../../engine/force/types';
import { Fix } from '../../engine/fix/fix';
import { ownCompute, removeCompute } from '../../engine/fix/util';
import type { System } from '../../engine/system';

interface Combined {
  /** lambda of each partition at the last post_force */
  lambda: number[];
  /** potential energy of each partition, then the mixed energy (epot[2]) */
  epot: Float64Array;
  /** mixed pressure tensor */
  pressure: Float64Array;
}

export class FixAlchemy extends Fix {
  readonly style: string = 'alchemy';
  replicaCoupled = true;
  scalarFlag = true;
  vectorFlag = true;
  sizeVector = 3;
  extvector = 1;

  private readonly varName: string;
  readonly idPe: string;
  readonly idTemp: string;
  readonly idPress: string;
  private readonly combined: Combined = { lambda: [0, 0], epot: new Float64Array(3), pressure: new Float64Array(6) };

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Incorrect number of arguments for fix alchemy (usage: fix ID group alchemy v_lambda)');
    if (!args[0].startsWith('v_')) throw new StyleError('Must use variable as lambda argument to fix alchemy');
    if (group !== 'all') throw new StyleError('fix alchemy must use group all');
    if (sys.replicaPeers().length !== 2) throw new StyleError('Must use exactly two partitions');
    this.varName = args[0].slice(2);
    this.idPe = `${id}_pe`;
    this.idTemp = `${id}_temp`;
    this.idPress = `${id}_press`;
    ownCompute(sys, this.idPe, 'all', 'pe', []);
    ownCompute(sys, this.idTemp, 'all', 'temp', []);
    ownCompute(sys, this.idPress, 'all', 'pressure', [this.idTemp]);
  }

  /** The fix of the leader partition (partition 0), which holds the exchanged results. */
  private get leader(): FixAlchemy {
    const peer0 = this.sys.replicaPeers()[0];
    if (peer0 === this.sys) return this;
    const f = peer0.fixes.find((x) => x.id === this.id);
    if (!(f instanceof FixAlchemy)) throw new StyleError(`fix ${this.id} (alchemy) is not defined on every partition`);
    return f;
  }

  private get iworld(): number { return this.sys.replicaPeers().indexOf(this.sys); }

  private get isLeader(): boolean { return this.leader === this; }

  init(): void {
    const alchemy = this.sys.fixes.filter((f) => f instanceof FixAlchemy);
    if (alchemy.length > 1) throw new StyleError('There may only be one fix alchemy at a time');
    if (this.sys.fixes.some((f) => f.style.startsWith('balance'))) throw new StyleError('Fix alchemy is not compatible with load balancing');
    if (this.sys.respa) throw new StyleError('Must not use run_style respa with fix alchemy');
    if (!this.isLeader) return;
    const peers = this.sys.replicaPeers();
    for (const p of peers) {
      if (!p.fixes.some((f) => f.id === this.id)) throw new StyleError(`fix ${this.id} (alchemy) is not defined on every partition`);
      if (p.state.n !== peers[0].state.n) throw new StyleError('Number of atoms and domain decomposition must be the same on all partitions');
    }
    // synchronize the box dimensions (native init: synchronize_box)
    this.syncBox();
  }

  setup(): void {
    if (!this.isLeader) return;
    this.postIntegrate();
    this.postForce();
  }

  /** native post_integrate: check the atoms, copy the positions (and the box) of partition 0 to the other partition. */
  postIntegrate(): void {
    if (!this.isLeader) return;
    this.checkConsistency();
    const peers = this.sys.replicaPeers();
    const x0 = peers[0].state.x;
    for (const p of peers.slice(1)) p.state.x.set(x0);
    this.syncBox();
  }

  /** native post_force: lambda of each partition, the mixed forces, energies and pressure. */
  postForce(): void {
    if (!this.isLeader) return;
    this.checkConsistency();
    const peers = this.sys.replicaPeers();
    const lambda = peers.map((p) => p.equalVariable(this.varName));
    // forces: f = sum_r lambda_r f_r, the same on both partitions (native: MPI_Allreduce of f * lambda)
    const n3 = peers[0].state.f.length;
    const fsum = new Float64Array(n3);
    for (let r = 0; r < peers.length; r++) {
      const f = peers[r].state.f;
      const lam = lambda[r];
      for (let i = 0; i < n3; i++) fsum[i] += f[i] * lam;
    }
    for (const p of peers) p.state.f.set(fsum);
    // energies and pressure of each partition, mixed with its own lambda
    const epot = this.combined.epot;
    epot.fill(0);
    const press = this.combined.pressure;
    press.fill(0);
    for (let r = 0; r < peers.length; r++) {
      const pe = peers[r].compute(this.idPe).scalarValue();
      epot[r] = pe;
      epot[2] += lambda[r] * pe;
    }
    for (let r = 0; r < peers.length; r++) {
      const pv = peers[r].compute(this.idPress).vectorValues();
      for (let c = 0; c < 6; c++) press[c] += lambda[r] * pv[c];
    }
    this.combined.lambda = lambda;
  }

  /** native compute_scalar: the lambda of this partition. */
  computeScalar(): number { return this.leader.combined.lambda[this.iworld] ?? 0; }

  /** native compute_vector / extract("pe") and ("pressure"): the energies (0 = partition 0, 1 = partition 1, 2 = mixed). */
  computeVector(i: number): number {
    if (i < 0 || i > 2) throw new StyleError(`fix ${this.id} (alchemy) vector index ${i + 1} is out of range`);
    return this.leader.combined.epot[i];
  }

  /** The mixed pressure tensor (compute pressure/alchemy). */
  pressureTensor(): Float64Array { return this.leader.combined.pressure; }

  destroy(): void {
    removeCompute(this.sys, this.idPe);
    removeCompute(this.sys, this.idTemp);
    removeCompute(this.sys, this.idPress);
  }

  /** native check_consistency_atoms: the same owned atom count and atom order on both partitions. */
  private checkConsistency(): void {
    const peers = this.sys.replicaPeers();
    const id0 = peers[0].state.id;
    for (const p of peers.slice(1)) {
      if (p.state.n !== peers[0].state.n) throw new StyleError('Fix alchemy local atom count is inconsistent');
      const id = p.state.id;
      for (let i = 0; i < id0.length; i++) {
        if (id[i] !== id0[i]) throw new StyleError('Fix alchemy local atom ordering is inconsistent');
      }
    }
  }

  /** native synchronize_box: the box of partition 0 on the others, when they differ. */
  private syncBox(): void {
    const peers = this.sys.replicaPeers();
    const b0 = peers[0].state.box;
    for (const p of peers.slice(1)) {
      const b = p.state.box;
      let differs = false;
      for (let d = 0; d < 3; d++) {
        if (b.lo[d] !== b0.lo[d] || b.hi[d] !== b0.hi[d] || b.tilt[d] !== b0.tilt[d]) differs = true;
      }
      if (differs) p.setBox([...b0.lo], [...b0.hi], [...b0.tilt], 0, this);
    }
  }
}
