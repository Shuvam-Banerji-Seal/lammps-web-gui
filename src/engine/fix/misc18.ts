import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Region } from '../region';
import { massOf } from '../atoms';
import { parseNumOrVar, valueOf, type NumOrVar } from './util';

/*
 * Wave-18 external / constraint fixes: addtorque, drag, oneway. Written from
 * the LAMMPS documentation (docs.lammps.org pages cited per class), never
 * from LAMMPS source. Forces are stored in internal units exactly like the
 * other force fixes (src/engine/fix/force_ext.ts): F_internal = F_lammps /
 * ftm2v is applied by the integrator, so a fix that wants a LAMMPS force F
 * adds F / ftm2v to state.f.
 */

/**
 * Solve I x = b for the symmetric 3x3 inertia tensor I (packed
 * xx,xy,xz,yy,yz,zz) by Gauss-Jordan with partial pivoting. A direction with
 * no usable pivot (a group whose atoms are collinear along that axis, so I is
 * singular there) gives x = 0 for it; measured with native LAMMPS (black
 * box): fix addtorque on two atoms on the x axis with a torque along x adds
 * exactly zero force, while a torque along z gives the expected +/- T/2.
 */
const solve3 = (I: Float64Array, b: Float64Array, out: Float64Array): void => {
  const a = [
    [I[0], I[1], I[2], b[0]],
    [I[1], I[3], I[4], b[1]],
    [I[2], I[4], I[5], b[2]],
  ];
  const used = [false, false, false];
  const row = [-1, -1, -1];
  for (let col = 0; col < 3; col++) {
    let best = -1;
    let bestVal = 1e-12;
    for (let r = 0; r < 3; r++) {
      if (!used[r] && Math.abs(a[r][col]) > bestVal) { bestVal = Math.abs(a[r][col]); best = r; }
    }
    if (best < 0) { out[col] = 0; continue; }
    used[best] = true;
    row[col] = best;
    const pv = a[best][col];
    for (let c = 0; c < 4; c++) a[best][c] /= pv;
    for (let r = 0; r < 3; r++) {
      if (r === best) continue;
      const fac = a[r][col];
      if (fac !== 0) for (let c = 0; c < 4; c++) a[r][c] -= fac * a[best][c];
    }
  }
  for (let col = 0; col < 3; col++) out[col] = row[col] < 0 ? 0 : a[row[col]][3];
};

/**
 * fix ID group-ID addtorque Tx Ty Tz — docs.lammps.org/fix_addtorque.html,
 * Syntax:
 *
 *   fix ID group-ID addtorque Tx Ty Tz
 *
 * "Tx,Ty,Tz = torque component values (torque units)" and "any of Tx,Ty,Tz
 * can be a variable". Description: "Add a set of forces to each atom in the
 * group such that:" "the components of the total torque applied on the group
 * (around its center of mass) are :math:`T_x`, :math:`T_y`, and :math:`T_z`"
 * "the group would move as a rigid body in the absence of other forces."
 * "Any of the three quantities defining the torque components can be
 * specified as an equal-style :doc:`variable <variable>`, namely *Tx*,
 * *Ty*, *Tz*". "the variable will be evaluated each timestep, and its value
 * used to determine the torque component."
 *
 * Output: "This fix computes a global scalar and a global 3-vector, which can
 * be accessed by various :doc:`output commands <Howto_output>`.  The scalar
 * is the potential energy discussed above.  The vector is the total torque on
 * the group of atoms before the forces on individual atoms are changed by the
 * fix.  The scalar and vector values calculated by this fix are "extensive"."
 * fix_modify: "The :doc:`fix_modify <fix_modify>` *energy* option is
 * supported by this fix to add the potential "energy" inferred by the added
 * torques to the global potential energy of the system as part of
 * :doc:`thermodynamic output <thermo_style>`.  The default setting for this
 * fix is :doc:`fix_modify energy no <fix_modify>`." Minimization: "The forces
 * due to this fix are imposed during an energy minimization, invoked by the
 * :doc:`minimize <minimize>` command."
 *
 * Force and energy, measured with native LAMMPS (black box) by dumping
 * per-atom forces and the f_1 scalar of a rotating free group to 1e-17:
 * with r_i = x_i - x_cm, the group inertia tensor I about its center of mass,
 * the group angular momentum L = sum_i m_i r_i x v_i, and the angular
 * velocity omega = I^-1 L (evaluated at post_force time, i.e. from the
 * half-step velocities), the drive angular acceleration is the rigid-body
 * value that makes dL/dt = T,
 *
 *   alpha = I^-1 (T - omega x (I omega)),
 *
 * and the added force is
 *
 *   F_i = m_i [ alpha x r_i + omega x (omega x r_i) ]
 *
 * (the second term is the rigid-body centripetal force; it is why a group
 * with a nonzero omega but T = 0 is still pushed inward and why the fix's
 * scalar is nonzero). The scalar is
 *
 *   U = sum_i m_i |omega x r_i|^2 = -sum_i F_i . r_i
 *
 * (LAMMPS computes it as -sum_i F_i . r_i). Default: "none".
 */
export class FixAddTorque extends Fix {
  readonly style = 'addtorque';
  private torque: [NumOrVar, NumOrVar, NumOrVar] = [0, 0, 0];
  /** Total torque on the group before the fix changed forces (LAMMPS torque units). */
  private vec = new Float64Array(3);
  private energyVal = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('usage: fix ID group-ID addtorque Tx Ty Tz');
    this.torque = [
      parseNumOrVar(args[0], 'addtorque Tx'),
      parseNumOrVar(args[1], 'addtorque Ty'),
      parseNumOrVar(args[2], 'addtorque Tz'),
    ];
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
  }

  postForce(): void {
    const s = this.sys.state;
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    // group center of mass
    let mtot = 0, cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const m = massOf(s, i);
      mtot += m;
      cx += m * x[3 * i]; cy += m * x[3 * i + 1]; cz += m * x[3 * i + 2];
    }
    this.vec.fill(0);
    this.energyVal = 0;
    if (!(mtot > 0)) return;
    cx /= mtot; cy /= mtot; cz /= mtot;
    // inertia tensor about the center of mass and angular momentum L = sum m r x v
    let Ixx = 0, Ixy = 0, Ixz = 0, Iyy = 0, Iyz = 0, Izz = 0;
    let Lx = 0, Ly = 0, Lz = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const m = massOf(s, i);
      const rx = x[3 * i] - cx, ry = x[3 * i + 1] - cy, rz = x[3 * i + 2] - cz;
      Ixx += m * (ry * ry + rz * rz);
      Iyy += m * (rx * rx + rz * rz);
      Izz += m * (rx * rx + ry * ry);
      Ixy -= m * rx * ry;
      Ixz -= m * rx * rz;
      Iyz -= m * ry * rz;
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      Lx += m * (ry * vz - rz * vy);
      Ly += m * (rz * vx - rx * vz);
      Lz += m * (rx * vy - ry * vx);
    }
    const I = new Float64Array([Ixx, Ixy, Ixz, Iyy, Iyz, Izz]);
    const T = new Float64Array([valueOf(this.sys, this.torque[0]), valueOf(this.sys, this.torque[1]), valueOf(this.sys, this.torque[2])]);
    const omega = new Float64Array(3);
    solve3(I, new Float64Array([Lx, Ly, Lz]), omega);
    // "the group would move as a rigid body": alpha = I^-1 (T - omega x (I omega))
    const iwx = Ixx * omega[0] + Ixy * omega[1] + Ixz * omega[2];
    const iwy = Ixy * omega[0] + Iyy * omega[1] + Iyz * omega[2];
    const iwz = Ixz * omega[0] + Iyz * omega[1] + Izz * omega[2];
    const gyro = new Float64Array([
      omega[1] * iwz - omega[2] * iwy,
      omega[2] * iwx - omega[0] * iwz,
      omega[0] * iwy - omega[1] * iwx,
    ]);
    const alpha = new Float64Array(3);
    solve3(I, new Float64Array([T[0] - gyro[0], T[1] - gyro[1], T[2] - gyro[2]]), alpha);
    const invF = 1 / s.units.ftm2v; // LAMMPS force -> internal
    const [ax, ay, az] = alpha;
    const [wx, wy, wz] = omega;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const m = massOf(s, i);
      const rx = x[3 * i] - cx, ry = x[3 * i + 1] - cy, rz = x[3 * i + 2] - cz;
      // "The vector is the total torque on the group of atoms before the forces on
      // individual atoms are changed by the fix." Existing internal forces -> LAMMPS.
      this.vec[0] += (ry * f[3 * i + 2] - rz * f[3 * i + 1]) * s.units.ftm2v;
      this.vec[1] += (rz * f[3 * i] - rx * f[3 * i + 2]) * s.units.ftm2v;
      this.vec[2] += (rx * f[3 * i + 1] - ry * f[3 * i]) * s.units.ftm2v;
      // alpha x r
      let gx = ay * rz - az * ry;
      let gy = az * rx - ax * rz;
      let gz = ax * ry - ay * rx;
      // omega x r, then omega x (omega x r)
      const bx = wy * rz - wz * ry;
      const by = wz * rx - wx * rz;
      const bz = wx * ry - wy * rx;
      gx += wy * bz - wz * by;
      gy += wz * bx - wx * bz;
      gz += wx * by - wy * bx;
      f[3 * i] += m * gx * invF;
      f[3 * i + 1] += m * gy * invF;
      f[3 * i + 2] += m * gz * invF;
      // energies are in LAMMPS energy units (m v^2 -> energy is mvv2e)
      this.energyVal += m * (bx * bx + by * by + bz * bz) * s.units.mvv2e;
    }
  }

  energy(): number { return this.energyVal; }
  computeScalar(): number { return this.energyVal; }
  computeVector(i: number): number { return this.vec[i]; }
  minPostForce(): void { this.postForce(); }
}

/**
 * fix ID group-ID drag x y z fmag delta — docs.lammps.org/fix_drag.html,
 * Syntax:
 *
 *   fix ID group-ID drag x y z fmag delta
 *
 * "x,y,z = coord to drag atoms towards", "fmag = magnitude of force to apply
 * to each atom (force units)", "delta = cutoff distance inside of which force
 * is not applied (distance units)". Description: "Apply a force to each atom
 * in a group to drag it towards the point (x,y,z).  The magnitude of the
 * force is specified by fmag.  If an atom is closer than a distance delta to
 * the point, then the force is not applied." "Any of the x,y,z values can be
 * specified as NULL which means do not include that dimension in the distance
 * calculation or force application."
 *
 * Output: "This fix computes a global 3-vector of forces, which can be
 * accessed by various :doc:`output commands <Howto_output>`.  This is the
 * total force on the group of atoms by the drag force.  The vector values
 * calculated by this fix are "extensive"." "This fix is not invoked during
 * :doc:`energy minimization <minimize>`." Default: "none".
 *
 * The force on an atom at distance r > delta is fmag * (target - x) / r using
 * only the non-NULL dimensions; at r == delta no force is applied (native
 * LAMMPS, black box: an atom at y = 1 with delta = 1 and a NULL x, z target
 * gets zero force, while y = 1.0000001 gets the full force).
 */
export class FixDrag extends Fix {
  readonly style = 'drag';
  private target: [number | null, number | null, number | null] = [null, null, null];
  private fmag = 0;
  private delta = 0;
  private vec = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 5) throw new StyleError('usage: fix ID group-ID drag x y z fmag delta');
    for (let d = 0; d < 3; d++) {
      if (args[d] === 'NULL') { this.target[d] = null; continue; }
      const val = Number(args[d]);
      if (args[d].trim() === '' || !Number.isFinite(val)) {
        throw new StyleError(`fix drag: ${'xyz'[d]} must be a number or NULL (coord to drag atoms towards), got '${args[d]}'`);
      }
      this.target[d] = val;
    }
    this.fmag = Number(args[3]);
    if (!Number.isFinite(this.fmag)) throw new StyleError(`fix drag: fmag must be a number (force units), got '${args[3]}'`);
    this.delta = Number(args[4]);
    if (!Number.isFinite(this.delta)) throw new StyleError(`fix drag: delta must be a number (distance units), got '${args[4]}'`);
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extvector = 1;
  }

  postForce(): void {
    const s = this.sys.state;
    const { x, f, mask } = s;
    const bit = this.groupBit;
    const d2 = this.delta * this.delta;
    const invF = 1 / s.units.ftm2v; // LAMMPS force -> internal
    this.vec.fill(0);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dx = this.target[0] === null ? 0 : this.target[0] - x[3 * i];
      const dy = this.target[1] === null ? 0 : this.target[1] - x[3 * i + 1];
      const dz = this.target[2] === null ? 0 : this.target[2] - x[3 * i + 2];
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= d2) continue;
      const c = this.fmag / Math.sqrt(r2);
      if (this.target[0] !== null) { const F = c * dx; f[3 * i] += F * invF; this.vec[0] += F; }
      if (this.target[1] !== null) { const F = c * dy; f[3 * i + 1] += F * invF; this.vec[1] += F; }
      if (this.target[2] !== null) { const F = c * dz; f[3 * i + 2] += F * invF; this.vec[2] += F; }
    }
  }

  /** The vector is "the total force on the group of atoms by the drag force" (LAMMPS force units). */
  computeVector(i: number): number { return this.vec[i]; }
}

/**
 * fix ID group-ID oneway N region-ID direction — docs.lammps.org/fix_oneway.html,
 * Syntax:
 *
 *   fix ID group-ID oneway N region-ID direction
 *
 * "N = apply this fix every this many timesteps", "region-ID = ID of region
 * where fix is active", "direction = *x* or *-x* or *y* or *-y* or *z* or
 * *-z* = coordinate and direction of the oneway constraint". Description:
 * "Enforce that particles in the group and in a given region can only move in
 * one direction.  This is done by reversing a particle's velocity component,
 * if it has the wrong sign in the specified dimension." "None of the
 * :doc:`fix_modify <fix_modify>` options are relevant to this fix.  No global
 * or per-atom quantities are stored by this fix for access by various
 * :doc:`output commands <Howto_output>`." "No parameter of this fix can be
 * used with the *start/stop* keywords of the :doc:`run <run>` command.  This
 * fix is not invoked during :doc:`energy minimization <minimize>`." Default:
 * "none".
 *
 * Measured with native LAMMPS (black box): the reversal is applied to the
 * full-step velocity, after the final velocity-Verlet kick and only on steps
 * where step % N == 0 (an atom pulled by fix addtorque plus oneway 5 ... x
 * has the sign of vx reversed only every fifth step, and the reversed value
 * equals -(v_half + dt/2 a), i.e. end_of_step, not post_integrate). Atoms
 * outside the region are untouched.
 */
export class FixOneway extends Fix {
  readonly style = 'oneway';
  private axis = 0;
  private sign = 1;
  private region: Region;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('usage: fix ID group-ID oneway N region-ID direction');
    const n = Number(args[0]);
    if (!Number.isInteger(n) || n < 1) throw new StyleError(`fix oneway: N must be a positive integer, got '${args[0]}'`);
    this.nevery = n;
    this.region = sys.region(args[1]);
    const dir = args[2];
    const map: Record<string, [number, number]> = {
      x: [0, 1], '-x': [0, -1], y: [1, 1], '-y': [1, -1], z: [2, 1], '-z': [2, -1],
    };
    const d = map[dir];
    if (!d) throw new StyleError(`fix oneway: direction must be x, -x, y, -y, z or -z, got '${dir}'`);
    this.axis = d[0];
    this.sign = d[1];
  }

  /** end_of_step runs only on steps that are a multiple of nevery (run loop). */
  endOfStep(): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    const bit = this.groupBit;
    const a = this.axis;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      if (!this.region.match(x[3 * i], x[3 * i + 1], x[3 * i + 2])) continue;
      if (v[3 * i + a] * this.sign < 0) v[3 * i + a] = -v[3 * i + a];
    }
  }
}
