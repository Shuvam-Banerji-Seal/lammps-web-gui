import { Compute } from './compute';
import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { isEllipsoid, isLineAtom, isLineStyle, isTriAtom, isTriStyle, LINE_BONUS, massOf, TRI_BONUS, TRI_BONUS_COLS } from '../atoms';
import type { SimState } from '../types';
import { SUPER_INERTIA } from '../superellipsoid';

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
 *
 * compute erotate/asphere also takes atom_style line and tri particles
 * (src/ASPHERE/compute_erotate_asphere.cpp at b332f28, compute_scalar; the
 * init message there is "Compute erotate/asphere requires atom style ellipsoid
 * or line or tri"):
 * - a line segment of length l and mass m contributes |omega|^2 l^2 m / 12
 *   (the omega of the atom, the same sum over all three components);
 * - a triangle contributes the body-frame sum I_k wb_k^2 with the principal
 *   moments I of its triangle bonus and wb = R^T L / I, the same form as the
 *   ellipsoid. The orientation is the atom's quaternion, which the engine keeps
 *   for tri particles (atoms.ts: the triangle bonus stores corners and moments only).
 */

/** Row-major rotation matrix (9 entries) of the unit quaternion w i j k: space = R body. */
export const quatToMatrix = (w: number, x: number, y: number, z: number): number[] => {
  // As native LAMMPS does (src/math_extra.cpp quat_to_mat at b332f28): the diagonal from the four squares, "mat[0][0] =
  // w2+i2-j2-k2", which equals 1 - 2 (y^2 + z^2) for a unit quaternion but keeps native's rounding, and is the zero matrix
  // for the zero quaternion a body created by create_atoms has (insertMolecules)
  const w2 = w * w, i2 = x * x, j2 = y * y, k2 = z * z;
  const twoij = 2 * x * y, twoik = 2 * x * z, twojk = 2 * y * z;
  const twoiw = 2 * x * w, twojw = 2 * y * w, twokw = 2 * z * w;
  return [
    w2 + i2 - j2 - k2, twoij - twokw, twojw + twoik,
    twoij + twokw, w2 - i2 + j2 - k2, twojk - twoiw,
    twoik - twojw, twojk + twoiw, w2 - i2 - j2 + k2,
  ];
};

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
  // atom_style ellipsoid superellipsoid: the moments stored when the shape or block was set (superellipsoid.ts)
  const stored = s.custom.get(SUPER_INERTIA);
  if (stored) return [stored.data[3 * i], stored.data[3 * i + 1], stored.data[3 * i + 2]];
  // I = m/5 (b^2 + c^2) etc., written as native LAMMPS does (src/ASPHERE/fix_nve_asphere.cpp at b332f28: "INERTIA*rmass[i] *
  // (shape[1]*shape[1]+shape[2]*shape[2])" with INERTIA = 0.2): 0.2 m and m / 5 can differ in the last bit
  const m = 0.2 * massOf(s, i);
  const sh = s.shape!;
  const a = sh[3 * i], b = sh[3 * i + 1], c = sh[3 * i + 2];
  return [m * (b * b + c * c), m * (a * a + c * c), m * (a * a + b * b)];
};

/** Body-frame angular momentum Lb = R^T L and angular velocity wb = Lb / I of particle i with principal moments I. */
export const bodyMotionWith = (s: SimState, i: number, I: [number, number, number]): { I: [number, number, number]; Lb: [number, number, number]; wb: [number, number, number] } => {
  const q = s.quat!, L = s.angmom!;
  const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
  const Lb = rotateTransposeVec(R, L[3 * i], L[3 * i + 1], L[3 * i + 2]);
  return { I, Lb, wb: [Lb[0] / I[0], Lb[1] / I[1], Lb[2] / I[2]] };
};

/** Body-frame angular momentum Lb = R^T L and angular velocity wb = Lb / I of ellipsoid i. */
export const bodyMotion = (s: SimState, i: number): { I: [number, number, number]; Lb: [number, number, number]; wb: [number, number, number] } =>
  bodyMotionWith(s, i, ellipsoidInertia(s, i));

/** Principal moments of triangle i (the triangle bonus, read_data Triangles; atoms.ts TRI_BONUS columns 9..11). */
const triMoments = (s: SimState, i: number): [number, number, number] => {
  const b = s.custom.get(TRI_BONUS)!.data;
  return [b[TRI_BONUS_COLS * i + 9], b[TRI_BONUS_COLS * i + 10], b[TRI_BONUS_COLS * i + 11]];
};

/** Every atom of the group must be a finite-size ellipsoid (all three half-axes positive). */
const allExtended = (s: SimState, groupBit: number): boolean => {
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & groupBit)) continue;
    if (!isEllipsoid(s, i) || !(s.shape![3 * i + 1] > 0) || !(s.shape![3 * i + 2] > 0)) return false;
  }
  return true;
};

/** A group atom must be a finite-size ellipsoid (all three half-axes positive), a line or a triangle. */
const extended = (s: SimState, i: number): boolean => {
  if (isEllipsoid(s, i)) return s.shape![3 * i + 1] > 0 && s.shape![3 * i + 2] > 0;
  return isLineAtom(s, i) || isTriAtom(s, i);
};

export class ComputeERotateAsphere extends Compute {
  readonly style = 'erotate/asphere';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute erotate/asphere command');
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  init(): void {
    // the atom style is checked at init (as native LAMMPS does, src/ASPHERE/compute_erotate_asphere.cpp at b332f28)
    const s = this.sys.state;
    const ellipsoidStyle = !!s.shape && !!s.quat && !!s.angmom;
    if (!ellipsoidStyle && !isLineStyle(s.atomStyle) && !isTriStyle(s.atomStyle)) {
      throw new StyleError('Compute erotate/asphere requires atom style ellipsoid or line or tri');
    }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (!extended(s, i)) throw new StyleError('Compute erotate/asphere requires extended particles');
    }
  }

  protected computeScalar(): number {
    const s = this.sys.state;
    let e = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (isEllipsoid(s, i)) {
        const { Lb, wb } = bodyMotion(s, i);
        e += Lb[0] * wb[0] + Lb[1] * wb[1] + Lb[2] * wb[2];
      } else if (isLineAtom(s, i)) {
        // native: (omega . omega) length^2 rmass / 12
        const w = s.omega!, len = s.custom.get(LINE_BONUS)!.data[2 * i];
        e += (w[3 * i] * w[3 * i] + w[3 * i + 1] * w[3 * i + 1] + w[3 * i + 2] * w[3 * i + 2]) * len * len * s.rmass![i] / 12;
      } else {
        const { Lb, wb } = bodyMotionWith(s, i, triMoments(s, i));
        e += Lb[0] * wb[0] + Lb[1] * wb[1] + Lb[2] * wb[2];
      }
    }
    return 0.5 * e * s.units.mvv2e;
  }
}

export class ComputeTempAsphere extends ComputeTemp {
  readonly style: string = 'temp/asphere';
  private rotateOnly = false;
  private biasId: string | null = null;
  private biasCompute: Compute | null = null;

  /**
   * The keywords are read here; the atom style and the bias compute are checked at init (as native LAMMPS does,
   * src/ASPHERE/compute_temp_asphere.cpp at b332f28).
   */
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k], val = args[k + 1];
      if (key === 'bias') {
        if (val === undefined) throw new StyleError('Illegal compute temp/asphere bias command: missing argument(s)');
        this.biasId = val;
      } else if (key === 'dof') {
        if (val === undefined) throw new StyleError('Illegal compute temp/asphere dof command: missing argument(s)');
        if (val !== 'all' && val !== 'rotate') throw new StyleError(`Unknown compute temp/asphere dof keyword ${val}`);
        this.rotateOnly = val === 'rotate';
      } else throw new StyleError(`Unknown compute temp/asphere keyword ${key}`);
    }
    // native: dof rotate counts no extra degrees of freedom unless compute_modify extra/dof sets them
    if (this.rotateOnly) this.extraDof = 0;
  }

  init(): void {
    const s = this.sys.state;
    if (!s.shape || !s.quat || !s.angmom) throw new StyleError('Compute temp/asphere requires atom style ellipsoid');
    if (!allExtended(s, this.groupBit)) throw new StyleError('Compute temp/asphere requires all extended particles');
    if (this.biasId !== null) {
      const bc = this.sys.computes.find((c) => c.id === this.biasId);
      if (!bc) throw new StyleError(`Could not find compute ID ${this.biasId} for temperature bias`);
      if (!bc.tempFlag) throw new StyleError(`Bias compute ${this.biasId} does not calculate temperature`);
      if (!bc.hasBias()) throw new StyleError(`Bias compute ${this.biasId} does not calculate a velocity bias`);
      if (bc.groupBit !== this.groupBit) throw new StyleError('Bias compute group does not match compute group');
      this.biasCompute = bc;
    }
    this.dofCompute();
  }

  dofCompute(): void {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.counted(i)) n++;
    // docs.lammps.org/compute_temp_asphere.html: "For 2d finite-size particles, each has three degrees of freedom (two translational, one rotational)." Measured with native LAMMPS (black box):
    // dof all in 2d is 3N - 2 (the default extra/dof is the dimension), dof rotate is N.
    const d = this.sys.dimension, rot = d === 3 ? 3 : 1;
    // As native LAMMPS does (src/ASPHERE/compute_temp_asphere.cpp dof_compute at b332f28): nper per atom, a bias's
    // dof_remove (per atom for temp/region), then extra/dof and the fixes' share, in both modes
    const nper = this.rotateOnly ? rot : d + rot;
    let dof = nper * n;
    const bc = this.biasCompute;
    if (bc && bc.style === 'temp/region') {
      for (let i = 0; i < s.n; i++) if (this.counted(i) && bc.dofRemove(i)) dof -= nper;
    } else if (bc && !this.rotateOnly) {
      dof -= bc.dofRemove(-1) * n;
    }
    this.dof = dof - this.extraDof - this.sys.dofRemoved(this.groupBit);
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
    // native recomputes the dof on every call with a temp/region bias (tempbias 2)
    if (this.dynamicDof || this.biasCompute?.style === 'temp/region') this.dofCompute();
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
