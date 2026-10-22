import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Region } from '../region';
import { massOf } from '../atoms';
import { parseNumOrVar, valueOf, type NumOrVar } from './util';

/*
 * Wave-18 external / constraint fixes: addtorque, drag, oneway. Written from
 * the LAMMPS documentation (docs.lammps.org pages cited per class), never
 * from LAMMPS source. Forces are stored in LAMMPS force units exactly like the
 * other force fixes (src/engine/fix/force_ext.ts): the integrator applies
 * ftm2v, so a fix that wants a LAMMPS force F adds F to state.f.
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
  // the singularity test is relative to the tensor's scale: an inertia tensor in SI units is ~1e-43 kg m^2
  let scale = 0;
  for (let k = 0; k < 6; k++) scale = Math.max(scale, Math.abs(I[k]));
  for (let col = 0; col < 3; col++) {
    let best = -1;
    let bestVal = 1e-12 * scale;
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
    this.vec.fill(0);
    const g = this.sys.geom;
    const d = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      // the separation to the target uses the minimum image (measured with native LAMMPS, black box: an atom
      // at x = 8 in a periodic box of length 10 dragged to x = 0 gets a force along +x, dx = 2)
      d[0] = this.target[0] === null ? 0 : this.target[0] - x[3 * i];
      d[1] = this.target[1] === null ? 0 : this.target[1] - x[3 * i + 1];
      d[2] = this.target[2] === null ? 0 : this.target[2] - x[3 * i + 2];
      g.minimumImage(d);
      const [dx, dy, dz] = d;
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= d2) continue;
      const c = this.fmag / Math.sqrt(r2);
      if (this.target[0] !== null) { const F = c * dx; f[3 * i] += F; this.vec[0] += F; }
      if (this.target[1] !== null) { const F = c * dy; f[3 * i + 1] += F; this.vec[1] += F; }
      if (this.target[2] !== null) { const F = c * dz; f[3 * i + 2] += F; this.vec[2] += F; }
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
    // As native LAMMPS does (src/EXTRA-FIX/fix_oneway.cpp FixOneWay::FixOneWay at b332f28): it reads N, region and
    // direction and needs at least six words in all; words after the direction are not read, so they are accepted.
    if (args.length < 3) throw new StyleError('usage: fix ID group-ID oneway N region-ID direction');
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
