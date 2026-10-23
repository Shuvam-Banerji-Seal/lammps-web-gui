/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_flow_gauss.cpp (FixFlowGauss: constructor, setup, post_force, compute_scalar,
 * compute_vector) at commit b332f28. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE). The method is Gaussian dynamics (Strong and Eaves, J Phys Chem B 121, 189, 2017).
 *
 * Adds a uniform acceleration to the group that conserves the group's momentum in the flagged directions
 * (docs.lammps.org/fix_flow_gauss.html).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { inumeric, logical } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';

export class FixFlowGauss extends Fix {
  readonly style = 'flow/gauss';
  private readonly flow: [boolean, boolean, boolean];
  /** fix_modify energy yes or the energy keyword: accumulate the work done by the added acceleration. */
  private workflag = false;
  private readonly dt: number;
  private peTot = 0;
  private readonly fTot = new Float64Array(3);
  private readonly aApp = new Float64Array(3);
  private mTot = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Not enough input arguments');
    // "a group which conserves momentum must also conserve particle number": dynamic groups are refused
    const flags: boolean[] = [];
    for (let k = 0; k < 3; k++) {
      // As native LAMMPS does (src/EXTRA-FIX/fix_flow_gauss.cpp FixFlowGauss::FixFlowGauss at b332f28): utils::inumeric
      // reads each flag before the 1-or-0 check.
      const t = inumeric(args[k]);
      if (!(t === 0 || t === 1)) throw new StyleError('Constraint flags must be 1 or 0');
      flags.push(t === 1);
    }
    this.flow = [flags[0], flags[1], flags[2]];
    // "by default, do not compute work done"
    this.workflag = false;
    // the timestep at declaration (update->dt in the constructor)
    this.dt = sys.state.dt;
    this.energyGlobal = true;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    for (let k = 3; k < args.length;) {
      if (args[k] !== 'energy') throw new StyleError('Illegal fix flow/gauss command');
      // fix_flow_gauss.cpp: utils::logical
      this.workflag = logical(args[k + 1]);
      k += 2;
    }
  }

  /** Total mass of the group (group->mass), at each run setup. */
  setup(): void {
    if (this.thermoEnergy) this.workflag = true;
    const s = this.sys.state;
    let m = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) m += massOf(s, i);
    this.mTot = m;
    if (this.mTot <= 0) throw new StyleError('Invalid group mass in fix flow/gauss');
    this.postForce();
  }

  /**
   * Sums the group forces in the flagged directions, applies a_app = -F_tot / M to every group atom (scaled by its
   * mass), and accumulates the work done when the energy term is on.
   */
  postForce(): void {
    const s = this.sys.state;
    const { f, v } = s;
    const bit = this.groupBit;
    const fThis = new Float64Array(3);
    // the sums follow native atom order (their roundoff depends on it)
    const order = nativeOrder(s);
    for (const i of order) {
      if (!(s.mask[i] & bit)) continue;
      for (let j = 0; j < 3; j++) if (this.flow[j]) fThis[j] += f[3 * i + j];
    }
    this.fTot.set(fThis);
    for (let j = 0; j < 3; j++) this.aApp[j] = -this.fTot[j] / this.mTot;
    let peAdded = 0;
    for (const i of order) {
      if (!(s.mask[i] & bit)) continue;
      const m = massOf(s, i);
      const fa0 = this.aApp[0] * m, fa1 = this.aApp[1] * m, fa2 = this.aApp[2] * m;
      f[3 * i] += fa0;
      f[3 * i + 1] += fa1;
      f[3 * i + 2] += fa2;
      if (this.workflag) peAdded += fa0 * v[3 * i] + fa1 * v[3 * i + 1] + fa2 * v[3 * i + 2];
    }
    if (this.workflag) this.peTot += peAdded;
  }

  /** -pe_tot * dt: the work done by the added acceleration (zero unless the energy term is on). */
  computeScalar(): number { return -this.peTot * this.dt; }
  computeVector(i: number): number { return -this.fTot[i]; }
  energy(): number { return this.computeScalar(); }
}
