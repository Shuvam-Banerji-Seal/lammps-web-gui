import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';
import { checkEqualVariables, parseNumOrVar, valueOf, type NumOrVar } from './util';

/*
 * fix addtorque/group — written from docs.lammps.org/fix_addtorque_group.html
 * (plans/lammps-docs-live/fix_addtorque_group.rst). Syntax quoted there:
 *   "fix ID group-ID addtorque/group Tx Ty Tz"
 * "Tx,Ty,Tz = torque component values (torque units)"; "any of Tx,Ty,Tz can be a
 * variable (see below)". The doc: "Add a set of forces to each atom in the group
 * such that:" "the components of the total torque applied on the group (around
 * its center of mass) are T_x, T_y, and T_z" and "the group would move as a rigid
 * body in the absence of other forces."
 *
 * Per-atom force (rigid-body kinematics). With I the mass-weighted inertia tensor of the
 * group about its centre of mass, L the angular momentum about it and omega = I^-1 L:
 *   alpha = I^-1 (T - omega x L)        (Euler's equation: I alpha + omega x L = T)
 *   F_i   = m_i (alpha x r_i + omega x (omega x r_i))
 * Measured with native LAMMPS (black box): with zero pair forces the added forces match this
 * to 2e-15 at every step of a 3-step run, using omega and L from the velocities present at force
 * time (the half-step velocities of velocity Verlet, not the end-of-step ones; the end-of-step
 * velocities leave a 1e-4 residual). With T = 0 the group still receives the centripetal forces
 * (a rigidly spinning group needs them), and their torque is omega x L, cancelled by the Euler term.
 * Summed force is zero and the total torque of the added forces is exactly T.
 *
 * Singular inertia: for a two-atom group I is singular along the pair axis.
 * Measured with native LAMMPS (black box): T = (2, 0, 0) on that pair gives zero forces, and
 * T = (0, 0.5, 0.7) gives the forces of the pseudo-inverse solution. The engine uses the
 * pseudo-inverse (eigenvalues below 1e-10 of the largest are dropped).
 *
 * Global outputs. The doc: "The scalar is the potential energy discussed above.
 * The vector is the total torque on the group of atoms before the forces on
 * individual atoms are changed by the fix." The engine stores unnormalised values: the vector is
 * the torque of the forces present before the fix about the group centre of mass; the scalar is, as native LAMMPS
 * computes it (src/EXTRA-FIX/fix_addtorque_group.cpp post_force at b332f28: "potential energy = - x dot f"),
 * -sum x_i . f_i of the added forces with the stored, wrapped positions. With unwrapped positions that sum is
 * omega . I . omega (the added forces sum to zero and alpha x r is normal to r); an atom across a periodic boundary
 * adds the term of its image shift. Measured with native LAMMPS (black box): printed values are these divided by
 * the total atom count, because thermo_modify norm defaults to yes for lj units (the same division applies to pe).
 *
 * fix_modify energy: the doc says this fix supports the energy option, which adds the potential
 * energy inferred by the added torques to the global potential energy of the system. Measured with
 * native LAMMPS (black box): with fix_modify energy yes the pe rises by exactly the scalar (before
 * normalisation). Default no.
 *
 * Minimization: "The forces due to this fix are imposed during an energy minimization,
 * invoked by the minimize command." The engine applies the same forces in min_post_force.
 *
 * Units: the stored forces are LAMMPS forces, and the integrators turn them into velocity changes with ftm2v. The
 * kinematic torque is ftm2v T, the accelerations are a = ftm2v F / m, so a fix force is m a / ftm2v. In lj units
 * the factor is 1 (the oracle metal cases
 * w48fb_metal_addtorque cover the conversions).
 *
 * Variables: Tx, Ty, Tz may be v_name equal-style variables, evaluated each step.
 */

/** Jacobi eigen-decomposition of a symmetric 3x3 matrix (row-major). */
const symEigen3 = (a: number[]): { vals: number[]; vecs: number[][] } => {
  const m = [a.slice(0, 3), a.slice(3, 6), a.slice(6, 9)];
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  let scale2 = 0;
  for (let k = 0; k < 9; k++) scale2 = Math.max(scale2, m[Math.floor(k / 3)][k % 3] * m[Math.floor(k / 3)][k % 3]);
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) off += m[p][q] * m[p][q];
    // relative to the matrix scale (an SI inertia tensor is ~1e-43, so an absolute threshold would stop at once)
    if (off <= 1e-30 * scale2) break;
    for (let p = 0; p < 3; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(m[p][q]) < 1e-300) continue;
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 3; k++) {
          const mkp = m[k][p], mkq = m[k][q];
          m[k][p] = c * mkp - s * mkq;
          m[k][q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < 3; k++) {
          const mpk = m[p][k], mqk = m[q][k];
          m[p][k] = c * mpk - s * mqk;
          m[q][k] = s * mpk + c * mqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { vals: [m[0][0], m[1][1], m[2][2]], vecs: v };
};

/** Pseudo-inverse of a symmetric 3x3 matrix (row-major); eigenvalues below 1e-10 of the largest are dropped. */
export const pinvSym3 = (a: number[]): number[] => {
  const { vals, vecs } = symEigen3(a);
  const big = Math.max(Math.abs(vals[0]), Math.abs(vals[1]), Math.abs(vals[2]));
  const out = new Array(9).fill(0);
  for (let e = 0; e < 3; e++) {
    if (Math.abs(vals[e]) <= 1e-10 * big || big === 0) continue;
    const inv = 1 / vals[e];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[3 * i + j] += inv * vecs[i][e] * vecs[j][e];
  }
  return out;
};

/** fix addtorque and fix addtorque/group: one style (src/EXTRA-FIX/fix_addtorque_group.h at b332f28 registers both names). */
export class FixAddTorqueGroup extends Fix {
  readonly style: string;
  private readonly torque: [NumOrVar, NumOrVar, NumOrVar];
  private etot = 0;
  private vec = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[], style = 'addtorque/group') {
    super(sys, id, group, args);
    this.style = style;
    if (args.length !== 3) throw new StyleError(`fix ${style}: usage: fix ID group-ID ${style} Tx Ty Tz`);
    this.torque = [parseNumOrVar(args[0], 'Tx'), parseNumOrVar(args[1], 'Ty'), parseNumOrVar(args[2], 'Tz')];
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
  }

  /** fix_addtorque_group.cpp init: the variables exist and are equal-style (native names none of them). */
  init(): void {
    checkEqualVariables(this.sys, this.torque, () => 'Variable name for fix addtorque/group does not exist', () => 'Variable for fix addtorque/group is invalid style');
  }

  postForce(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const x = s.x, v = s.v, f = s.f;
    // the torque is in LAMMPS units (force times length); the kinematic torque is ftm2v times it
    const ftm2v = s.units.ftm2v;
    const T = [valueOf(this.sys, this.torque[0]) * ftm2v, valueOf(this.sys, this.torque[1]) * ftm2v, valueOf(this.sys, this.torque[2]) * ftm2v];
    // group members: mass, unwrapped position, velocity
    const idx: number[] = [];
    const mass: number[] = [];
    const pos: number[] = [];
    const vel: number[] = [];
    const p = [0, 0, 0];
    let mtot = 0;
    const com = [0, 0, 0], vcm = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const m = massOf(s, i);
      g.unwrap(x, s.image, i, p);
      idx.push(i); mass.push(m);
      pos.push(p[0], p[1], p[2]);
      vel.push(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
      mtot += m;
      for (let c = 0; c < 3; c++) { com[c] += m * p[c]; vcm[c] += m * v[3 * i + c]; }
    }
    this.vec.fill(0);
    this.etot = 0;
    if (idx.length === 0 || mtot === 0) return;
    for (let c = 0; c < 3; c++) { com[c] /= mtot; vcm[c] /= mtot; }
    // inertia tensor about the centre of mass, angular momentum about it, and the torque of the present forces
    const I = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    const L = [0, 0, 0];
    const tpre = [0, 0, 0];
    const r = [0, 0, 0];
    for (let k = 0; k < idx.length; k++) {
      const m = mass[k];
      for (let c = 0; c < 3; c++) r[c] = pos[3 * k + c] - com[c];
      const r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) I[3 * a + b] += m * ((a === b ? r2 : 0) - r[a] * r[b]);
      const dv = [vel[3 * k] - vcm[0], vel[3 * k + 1] - vcm[1], vel[3 * k + 2] - vcm[2]];
      L[0] += m * (r[1] * dv[2] - r[2] * dv[1]);
      L[1] += m * (r[2] * dv[0] - r[0] * dv[2]);
      L[2] += m * (r[0] * dv[1] - r[1] * dv[0]);
      const fi = idx[k];
      const fx = f[3 * fi], fy = f[3 * fi + 1], fz = f[3 * fi + 2];
      tpre[0] += r[1] * fz - r[2] * fy;
      tpre[1] += r[2] * fx - r[0] * fz;
      tpre[2] += r[0] * fy - r[1] * fx;
    }
    // vector: the torque of the forces present before this fix, in LAMMPS units (the stored forces are LAMMPS forces:
    // the integrators apply v += dt ftm2v F / m)
    for (let c = 0; c < 3; c++) this.vec[c] = tpre[c];
    const Ip = pinvSym3(I);
    // omega = I^+ L (rigid-body angular velocity at force time); alpha = I^+ (T - omega x L): the Euler equation
    const omega = [0, 0, 0];
    for (let a = 0; a < 3; a++) omega[a] = Ip[3 * a] * L[0] + Ip[3 * a + 1] * L[1] + Ip[3 * a + 2] * L[2];
    const wxL = [omega[1] * L[2] - omega[2] * L[1], omega[2] * L[0] - omega[0] * L[2], omega[0] * L[1] - omega[1] * L[0]];
    const rhs = [T[0] - wxL[0], T[1] - wxL[1], T[2] - wxL[2]];
    const alpha = [0, 0, 0];
    const invF = 1 / s.units.ftm2v;
    for (let a = 0; a < 3; a++) alpha[a] = Ip[3 * a] * rhs[0] + Ip[3 * a + 1] * rhs[1] + Ip[3 * a + 2] * rhs[2];
    // F_i = m_i (alpha x r_i + omega x (omega x r_i)): the rigid-body accelerations of the atoms
    for (let k = 0; k < idx.length; k++) {
      const m = mass[k];
      for (let c = 0; c < 3; c++) r[c] = pos[3 * k + c] - com[c];
      const fi = idx[k];
      const ar = [alpha[1] * r[2] - alpha[2] * r[1], alpha[2] * r[0] - alpha[0] * r[2], alpha[0] * r[1] - alpha[1] * r[0]];
      const wr = [omega[1] * r[2] - omega[2] * r[1], omega[2] * r[0] - omega[0] * r[2], omega[0] * r[1] - omega[1] * r[0]];
      const cen = [omega[1] * wr[2] - omega[2] * wr[1], omega[2] * wr[0] - omega[0] * wr[2], omega[0] * wr[1] - omega[1] * wr[0]];
      // acceleration a = ar + cen (kinematic units); the LAMMPS force is m a / ftm2v, since a = ftm2v F / m
      const fax = (m * (ar[0] + cen[0])) * invF;
      const fay = (m * (ar[1] + cen[1])) * invF;
      const faz = (m * (ar[2] + cen[2])) * invF;
      // the scalar: "potential energy = - x dot f" of the added forces with the stored (wrapped) positions
      this.etot -= fax * x[3 * fi] + fay * x[3 * fi + 1] + faz * x[3 * fi + 2];
      f[3 * fi] += fax;
      f[3 * fi + 1] += fay;
      f[3 * fi + 2] += faz;
    }
  }

  minPostForce(): void { this.postForce(); }

  energy(): number { return this.etot; }
  computeScalar(): number { return this.etot; }
  computeVector(i: number): number {
    if (i < 0 || i > 2) throw new StyleError(`fix ${this.id} addtorque/group vector index ${i + 1} out of range`);
    return this.vec[i];
  }
}
