import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * compute ID group temp — docs.lammps.org/compute_temp.html:
 *   T = 2 E_kin / (N_DOF k_B),  E_kin = sum 1/2 m_i v_i^2,
 *   N_DOF = n_dim N_atoms - n_dim - N_fixDOFs
 * "A symmetric tensor, stored as a six-element vector, is also calculated
 * ... the same as the above expression for E_kin, except that the 1/2 factor
 * is NOT included and the v_i^2 is replaced by v_i,x v_i,y for the xy
 * component". "This compute subtracts out degrees-of-freedom due to fixes
 * that constrain molecular motion, such as fix shake and fix rigid."
 */

export class ComputeTemp extends Compute {
  readonly style: string = 'temp';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`compute temp takes no arguments (got ${args.join(' ')})`);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extscalar = 0;
    this.extvector = 1;
    this.tempFlag = true;
    this.vector = new Float64Array(6);
  }

  init(): void {
    this.dofCompute();
  }

  /** Atoms of the group that count (subclasses restrict further). */
  protected counted(i: number): boolean {
    return (this.sys.state.mask[i] & this.groupBit) !== 0;
  }

  dofCompute(): void {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.counted(i)) n++;
    const d = this.sys.dimension;
    this.dof = d * n - this.extraDof - this.sys.dofRemoved(this.groupBit);
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const { v, massByType, type } = s;
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massByType[type[i]];
      t += m * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const { v, massByType, type } = s;
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massByType[type[i]];
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      t[0] += m * vx * vx; t[1] += m * vy * vy; t[2] += m * vz * vz;
      t[3] += m * vx * vy; t[4] += m * vx * vz; t[5] += m * vy * vz;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }
}
