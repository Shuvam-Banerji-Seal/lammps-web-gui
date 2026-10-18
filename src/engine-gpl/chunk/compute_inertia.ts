/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/compute_inertia.cpp and
 * src/compute_inertia.h at commit b332f28 (compute inertia), with the group functions it calls (Group::mass, Group::xcm,
 * Group::inertia and Group::inertia_extended in src/group.cpp, and the add_extended_inertia helper). Copyright (2003)
 * Sandia Corporation (LAMMPS development team). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_inertia.html: "Define a computation that calculates the moment of inertia tensor of a group of
 * atoms. ... The six components of the symmetric inertia tensor ... are calculated in the order Ixx, Iyy, Izz, Ixy,
 * Iyz, Ixz." It is a global vector of length 6, intensive. The tensor is taken about the centre of mass of the group,
 * with unwrapped coordinates, and includes the moment of inertia of finite-size particles (spheres with radius > 0,
 * ellipsoids) as group->inertia_extended does.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { finiteSpin } from '../../engine/compute/finiteSpin';

export class ComputeInertia extends Compute {
  readonly style = 'inertia';
  /** masstotal of native: the group mass, set in init() and again at each evaluation of a dynamic group. */
  private masstotal = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 0) throw new StyleError('Illegal compute inertia command');
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extvector = 0;
  }

  /** Group::mass: the sum of the masses of the group atoms, in native order. */
  private groupMass(): number {
    const s = this.sys.state;
    let one = 0;
    for (const i of nativeOrder(s)) if (s.mask[i] & this.groupBit) one += massOf(s, i);
    return one;
  }

  init(): void {
    this.masstotal = this.groupMass();
  }

  protected computeVector(): void {
    const s = this.sys.state;
    if (this.sys.groups.isDynamic(this.groupBit)) this.masstotal = this.groupMass();
    const order = nativeOrder(s);
    // Group::xcm: unwrapped mass-weighted sums, divided by the group mass when it is positive
    const cm = [0, 0, 0];
    const u = [0, 0, 0];
    for (const i of order) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      cm[0] += u[0] * m; cm[1] += u[1] * m; cm[2] += u[2] * m;
    }
    if (this.masstotal > 0.0) {
      cm[0] /= this.masstotal; cm[1] /= this.masstotal; cm[2] /= this.masstotal;
    }
    // Group::inertia: point-mass tensor about cm
    const ione = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const i of order) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const dx = u[0] - cm[0], dy = u[1] - cm[1], dz = u[2] - cm[2];
      const m = massOf(s, i);
      ione[0][0] += m * (dy * dy + dz * dz);
      ione[1][1] += m * (dx * dx + dz * dz);
      ione[2][2] += m * (dx * dx + dy * dy);
      ione[0][1] -= m * dx * dy;
      ione[1][2] -= m * dy * dz;
      ione[0][2] -= m * dx * dz;
    }
    ione[1][0] = ione[0][1];
    ione[2][1] = ione[1][2];
    ione[2][0] = ione[0][2];
    // Group::inertia_extended: the finite-size particles' own tensors, added to the point-mass tensor
    const ext = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const i of order) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const f = finiteSpin(s, i);
      if (!f) continue;
      const T = f.T;
      ext[0][0] += T[0]; ext[1][1] += T[1]; ext[2][2] += T[2];
      ext[0][1] += T[3]; ext[1][2] += T[4]; ext[0][2] += T[5];
    }
    ext[1][0] = ext[0][1];
    ext[2][1] = ext[1][2];
    ext[2][0] = ext[0][2];
    const it = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) it[a][b] = ione[a][b] + ext[a][b];
    const out = new Float64Array(6);
    out[0] = it[0][0];
    out[1] = it[1][1];
    out[2] = it[2][2];
    out[3] = it[0][1];
    out[4] = it[1][2];
    out[5] = it[0][2];
    this.vector = out;
  }
}
