import { Compute } from './compute';
import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { isEllipsoid, massOf } from '../atoms';
import type { SimState } from '../types';

/*
 * compute erotate/asphere and temp/asphere for atom_style ellipsoid, and the
 * rigid-body helpers shared with fix nve/asphere (quaternion to rotation
 * matrix, body-frame angular momentum and inertia).
 *
 * docs.lammps.org/compute_erotate_asphere.html: "All particles in the group
 * must be of finite size." "They cannot be point particles." The rotational
 * kinetic energy is one half I omega^2 (the page's formula), with I the inertia
 * tensor and omega the angular velocity, "computed from its angular momentum
 * if needed."
 *
 * docs.lammps.org/compute_temp_asphere.html: "For 3d finite-size particles,
 * each has six degrees of freedom (three translational, three rotational)."
 * "The translational kinetic energy is computed the same as is described by"
 * the temp compute. For the bias keyword, *bias-ID* refers to the ID of a
 * temperature compute that removes a bias velocity from each atom (the page
 * puts bias in quote marks). "a setting of *all* calculates a temperature that
 * includes both translational and rotational degrees of freedom." "A setting
 * of *rotate* calculates a temperature that includes only
 * rotational degrees of freedom."
 *
 * The moments of inertia of a solid ellipsoid with half-axes a, b, c and mass m
 * in its body frame (textbook): I = m/5 (b^2+c^2), m/5 (a^2+c^2), m/5 (a^2+b^2).
 * The body frame is the quaternion rotation R(q): space = R body. The body
 * angular velocity is I^-1 R^T L with L the space-frame angular momentum.
 *
 * Measured with native LAMMPS (black box; one ellipsoid with half-axes 1, 1.5,
 * 2, mass 2, several orientations and angular momenta):
 * - the erotate value equals 1/2 sum_k Lb_k^2 / I_k with Lb = R^T L (the
 *   energy in the space-frame quantities is the same);
 * - temp with dof all: N_DOF = 6 N - 3 (extra dof 3 subtracted), so that the
 *   temperature is 2 E / N_DOF; with dof rotate: N_DOF = 3 N, nothing subtracted;
 * - the vector (no 1/2 factor) has diagonal entries Lb_k w_k = I_k w_k^2 and
 *   off-diagonal entries I_0 w_0 w_1 (xy), I_1 w_0 w_2 (xz), I_2 w_1 w_2 (yz)
 *   with w the body-frame angular velocity: the rotational part is reported in
 *   the body frame, and the trace equals 2 E; the translational part m v_a v_b
 *   is in the space frame (oracle w19asphere_mix);
 * - bias tc (temp/com) removes the bias velocity from the translational part
 *   only: with equal velocities v = (1,0,0) on masses 2 and 1 and L = (0.5,-0.8,1.3)
 *   on the second atom, temp = 1.05 and with bias 0.71667 (= rotational part / 9);
 * - a point particle in the group stops the run for both computes (native
 *   messages, paraphrased: the temp/asphere one asks for all extended particles,
 *   the erotate/asphere one for extended particles).
 */

/** Row-major rotation matrix (9 entries) of the unit quaternion w i j k: space = R body. */
export const quatToMatrix = (w: number, x: number, y: number, z: number): number[] => [
  1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
];

/** R v for a row-major 3x3 matrix. */
export const rotateVec = (R: number[], x: number, y: number, z: number): [number, number, number] => [
  R[0] * x + R[1] * y + R[2] * z,
  R[3] * x + R[4] * y + R[5] * z,
  R[6] * x + R[7] * y + R[8] * z,
];

/** R^T v for a row-major 3x3 matrix. */
export const rotateTransposeVec = (R: number[], x: number, y: number, z: number): [number, number, number] => [
  R[0] * x + R[3] * y + R[6] * z,
  R[1] * x + R[4] * y + R[7] * z,
  R[2] * x + R[5] * y + R[8] * z,
];

/** Principal moments of inertia of ellipsoid i (body frame): m/5 (b^2+c^2), m/5 (a^2+c^2), m/5 (a^2+b^2). */
export const ellipsoidInertia = (s: SimState, i: number): [number, number, number] => {
  const m = massOf(s, i) / 5;
  const sh = s.shape!;
  const a = sh[3 * i], b = sh[3 * i + 1], c = sh[3 * i + 2];
  return [m * (b * b + c * c), m * (a * a + c * c), m * (a * a + b * b)];
};

/** Body-frame angular momentum Lb = R^T L and angular velocity wb = Lb / I of ellipsoid i. */
export const bodyMotion = (s: SimState, i: number): { I: [number, number, number]; Lb: [number, number, number]; wb: [number, number, number] } => {
  const q = s.quat!, L = s.angmom!;
  const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
  const Lb = rotateTransposeVec(R, L[3 * i], L[3 * i + 1], L[3 * i + 2]);
  const I = ellipsoidInertia(s, i);
  return { I, Lb, wb: [Lb[0] / I[0], Lb[1] / I[1], Lb[2] / I[2]] };
};

/** Every atom of the group must be a finite-size ellipsoid (all three half-axes positive). */
const allExtended = (s: SimState, groupBit: number): boolean => {
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & groupBit)) continue;
    if (!isEllipsoid(s, i) || !(s.shape![3 * i + 1] > 0) || !(s.shape![3 * i + 2] > 0)) return false;
  }
  return true;
};

export class ComputeERotateAsphere extends Compute {
  readonly style = 'erotate/asphere';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`Illegal compute erotate/asphere keyword ${args[0]}`);
    if (!sys.state.shape || !sys.state.quat || !sys.state.angmom) throw new StyleError('Compute erotate/asphere requires atom style ellipsoid');
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  init(): void {
    if (!allExtended(this.sys.state, this.groupBit)) throw new StyleError('Compute erotate/asphere requires extended particles');
  }

  protected computeScalar(): number {
    const s = this.sys.state;
    let e = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const { Lb, wb } = bodyMotion(s, i);
      e += Lb[0] * wb[0] + Lb[1] * wb[1] + Lb[2] * wb[2];
    }
    return 0.5 * e * s.units.mvv2e;
  }
}

export class ComputeTempAsphere extends ComputeTemp {
  readonly style: string = 'temp/asphere';
  private rotateOnly = false;
  private biasCompute: Compute | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (!sys.state.shape || !sys.state.quat || !sys.state.angmom) throw new StyleError('Compute temp/asphere requires atom style ellipsoid');
    if (sys.dimension !== 3) throw new StyleError('compute temp/asphere is only supported for 3d systems');
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k], val = args[k + 1];
      if (val === undefined) throw new StyleError(`Illegal compute temp/asphere keyword ${key}: missing value`);
      if (key === 'dof') {
        if (val !== 'all' && val !== 'rotate') throw new StyleError(`Illegal compute temp/asphere dof keyword ${val}`);
        this.rotateOnly = val === 'rotate';
      } else if (key === 'bias') {
        const bc = sys.compute(val);
        if (!bc.hasBias()) throw new StyleError(`Compute ${val} used as bias for temp/asphere does not remove a velocity bias`);
        this.biasCompute = bc;
      } else throw new StyleError(`Illegal compute temp/asphere keyword ${key}`);
    }
  }

  init(): void {
    if (!allExtended(this.sys.state, this.groupBit)) throw new StyleError('Compute temp/asphere requires all extended particles');
    this.dofCompute();
  }

  dofCompute(): void {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.counted(i)) n++;
    this.dof = this.rotateOnly ? 3 * n : 6 * n - this.extraDof - this.sys.dofRemoved(this.groupBit);
  }

  /**
   * Runs f with the bias velocity removed from the group (compute temp/com and the
   * other bias computes), restoring the velocities afterwards.
   */
  private withBias<T>(f: () => T): T {
    const bc = this.biasCompute;
    if (!bc) return f();
    bc.computeBias();
    bc.removeBiasAll();
    try { return f(); } finally { bc.restoreBiasAll(); }
  }

  /** Translational sum m v^2 (xx,yy,zz,xy,xz,yz) and body-frame rotational tensor (0 for dof rotate is handled by the caller). */
  private tensor(): { trans: number[]; rot: number[] } {
    const s = this.sys.state;
    const { v } = s;
    const trans = [0, 0, 0, 0, 0, 0], rot = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massOf(s, i);
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      trans[0] += m * vx * vx; trans[1] += m * vy * vy; trans[2] += m * vz * vz;
      trans[3] += m * vx * vy; trans[4] += m * vx * vz; trans[5] += m * vy * vz;
      const { I, Lb, wb } = bodyMotion(s, i);
      rot[0] += Lb[0] * wb[0]; rot[1] += Lb[1] * wb[1]; rot[2] += Lb[2] * wb[2];
      rot[3] += I[0] * wb[0] * wb[1]; rot[4] += I[1] * wb[0] * wb[2]; rot[5] += I[2] * wb[1] * wb[2];
    }
    return { trans, rot };
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    return this.withBias(() => {
      const t = this.tensor();
      const sum = (this.rotateOnly ? 0 : t.trans[0] + t.trans[1] + t.trans[2]) + t.rot[0] + t.rot[1] + t.rot[2];
      const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
      return sum * tfactor;
    });
  }

  protected computeVector(): void {
    const s = this.sys.state;
    this.withBias(() => {
      const t = this.tensor();
      for (let c = 0; c < 6; c++) this.vector[c] = ((this.rotateOnly ? 0 : t.trans[c]) + t.rot[c]) * s.units.mvv2e;
    });
  }
}
