import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * compute msd/nongauss — written from docs.lammps.org/compute_msd_nongauss.html
 * (plans/lammps-docs/compute_msd_nongauss.rst). Syntax: "compute ID group-ID
 * msd/nongauss keyword values ...", keyword "com" with value yes or no
 * (default no). The page says "A vector of three quantities is calculated by this
 * compute. The first element of the vector is the total squared displacement,
 * dr^2 = dx^2 + dy^2 + dz^2, of the atoms, and the second is the fourth power of
 * these displacements, dr^4 = (dx^2 + dy^2 + dz^2)^2, summed and averaged over
 * atoms in the group. The third component is the non-Gaussian diffusion
 * parameter NGP, NGP(t) = 3 <(r(t)-r(0))^4> / (5 <(r(t)-r(0))^2>^2) - 1."
 * "If the com option is set to yes then the effect of any drift in the
 * center-of-mass of the group of atoms is subtracted out before the
 * displacement of each atom is calculated." "Compute msd/nongauss cannot be
 * used with a dynamic group." Output: "The first vector value will be in" distance^2 units, the second in distance^4 units,
 * and the third is dimensionless.
 *
 * Displacements are from the unwrapped positions captured when the compute is
 * defined (compute_msd.rst: "Initial coordinates are stored in" unwrapped form, using the image flags).
 */
export class ComputeMSDNonGauss extends Compute {
  readonly style = 'msd/nongauss';
  private comFlag: boolean;
  private xorig: Float64Array;
  private cm0: [number, number, number] = [0, 0, 0];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.comFlag = false;
    if (args.length % 2) throw new StyleError('compute msd/nongauss: keyword values must come in pairs');
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k];
      if (key !== 'com') throw new StyleError(`unknown compute msd/nongauss keyword '${key}'`);
      const v = args[k + 1];
      if (v !== 'yes' && v !== 'no') throw new StyleError(`compute msd/nongauss com must be yes or no (got '${v ?? ''}')`);
      this.comFlag = v === 'yes';
    }
    if (sys.groups.isDynamic(this.groupBit)) throw new StyleError('compute msd/nongauss cannot be used with a dynamic group');
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extvector = 0;
    this.vector = new Float64Array(3);
    const s = sys.state;
    const u = [0, 0, 0];
    this.xorig = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) {
      sys.geom.unwrap(s.x, s.image, i, u);
      this.xorig[3 * i] = u[0]; this.xorig[3 * i + 1] = u[1]; this.xorig[3 * i + 2] = u[2];
    }
    if (this.comFlag) this.cm0 = this.groupCom();
  }

  /** Mass-weighted centre of mass of the group in unwrapped coordinates. */
  private groupCom(): [number, number, number] {
    const s = this.sys.state;
    const u = [0, 0, 0];
    let mx = 0, my = 0, mz = 0, msum = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      mx += m * u[0]; my += m * u[1]; mz += m * u[2];
      msum += m;
    }
    return [mx / msum, my / msum, mz / msum];
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const u = [0, 0, 0];
    const cmNow = this.comFlag ? this.groupCom() : null;
    let sum2 = 0, sum4 = 0, count = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      count++;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      let dx = u[0] - this.xorig[3 * i];
      let dy = u[1] - this.xorig[3 * i + 1];
      let dz = u[2] - this.xorig[3 * i + 2];
      if (cmNow) {
        dx -= cmNow[0] - this.cm0[0];
        dy -= cmNow[1] - this.cm0[1];
        dz -= cmNow[2] - this.cm0[2];
      }
      const r2 = dx * dx + dy * dy + dz * dz;
      sum2 += r2;
      sum4 += r2 * r2;
    }
    const v = this.vector;
    const m2 = sum2 / count, m4 = sum4 / count;
    v[0] = m2;
    v[1] = m4;
    v[2] = m2 > 0 ? (3 * m4) / (5 * m2 * m2) - 1 : 0;
  }
}
