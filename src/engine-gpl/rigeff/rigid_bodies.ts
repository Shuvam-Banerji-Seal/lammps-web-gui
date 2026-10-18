/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/RIGID/compute_erotate_rigid.cpp,
 * compute_ke_rigid.cpp and compute_rigid_local.cpp (the rigid body reads of the three computes), with the quantities
 * of fix_rigid.cpp extract_ke / extract_erotational and fix_rigid_small.cpp extract_ke / extract_erotational
 * (the sums of the body state). Contributing authors (from the sources): Trung Dac Nguyen (ORNL) for the rigid
 * package, Mike Brown (ORNL) and Steve Plimpton (Sandia) for the computes. LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The engine's fix rigid styles keep their bodies privately. This module reads the body state of the core FixRigid
 * (snapshotBodies) and of the Nose-Hoover family (the protected bodies array of FixRigidBase) in the native
 * quantities: the wrapped centre of mass xcm, its image flags, quaternion, space-frame angular momentum, inertia.
 */
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';

/** One rigid body in the native quantities (fix_rigid_small.cpp Body): xcm is wrapped into the box, image counts periods. */
export interface RigidBodyView {
  atoms: number[];
  mass: number;
  xcm: number[];
  vcm: number[];
  fcm: number[];
  torque: number[];
  image: number[];
  inertia: number[];
  quat: number[];
  angmom: number[];
  omega: number[];
}

const isRigidStyle = (style: string): boolean => style.startsWith('rigid');

/**
 * The bodies of a rigid fix in native terms, or null if the fix holds none (the native dynamic_cast fails and the
 * compute returns 0). The core FixRigid keeps xcm unwrapped (its wrapped position is xcm - image L, as its own
 * computeArray reads it); the Nose-Hoover styles keep xcm wrapped with imagebody (domain->remap).
 */
export const rigidBodiesOf = (sys: System, fix: Fix): RigidBodyView[] | null => {
  if (!isRigidStyle(fix.style)) return null;
  const box = sys.state.box;
  const L = [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]];
  const core = fix as unknown as { snapshotBodies?: () => Array<Record<string, unknown>> };
  if (typeof core.snapshotBodies === 'function') {
    return core.snapshotBodies().map((b) => {
      const xu = b.xcm as number[];
      const image = b.image as number[];
      const xcm = [0, 1, 2].map((d) => xu[d] - image[d] * L[d]);
      return {
        atoms: b.atoms as number[], mass: b.mass as number, xcm, vcm: b.vcm as number[], fcm: b.fcm as number[],
        torque: b.torque as number[], image: image.slice(), inertia: b.inertia as number[], quat: b.q as number[],
        angmom: b.angmom as number[], omega: b.omega as number[],
      };
    });
  }
  const nh = fix as unknown as { bodies?: Array<Record<string, unknown>> };
  if (Array.isArray(nh.bodies)) {
    return nh.bodies.map((b) => ({
      atoms: b.atoms as number[], mass: b.mass as number, xcm: (b.xcm as number[]).slice(), vcm: (b.vcm as number[]).slice(),
      fcm: (b.fcm as number[]).slice(), torque: (b.torque as number[]).slice(), image: (b.imagebody as number[]).slice(),
      inertia: (b.inertia as number[]).slice(), quat: (b.quat as number[]).slice(), angmom: (b.angmom as number[]).slice(),
      omega: (b.omega as number[]).slice(),
    }));
  }
  return null;
};

/** math_extra.h quat_to_mat: the rotation matrix of a body quaternion (w, i, j, k), rot[row][col]. */
export const quatToMatNative = (q: readonly number[]): number[][] => {
  const w2 = q[0] * q[0], i2 = q[1] * q[1], j2 = q[2] * q[2], k2 = q[3] * q[3];
  const twoij = 2.0 * q[1] * q[2], twoik = 2.0 * q[1] * q[3], twojk = 2.0 * q[2] * q[3];
  const twoiw = 2.0 * q[1] * q[0], twojw = 2.0 * q[2] * q[0], twokw = 2.0 * q[3] * q[0];
  return [
    [w2 + i2 - j2 - k2, twoij - twokw, twojw + twoik],
    [twoij + twokw, w2 - i2 + j2 - k2, twojk - twoiw],
    [twoik - twojw, twojk + twoiw, w2 - i2 - j2 + k2],
  ];
};

/** math_extra.h transpose_matvec: ans = m^T v. */
export const transposeMatvecNative = (m: number[][], v: readonly number[]): number[] => [
  m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
  m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
  m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
];

/**
 * fix_rigid.cpp extract_ke / fix_rigid_small.cpp extract_ke: 0.5 sum_b M_b |vcm_b|^2 (the native loop order over bodies).
 */
export const extractKe = (bodies: RigidBodyView[]): number => {
  let ke = 0.0;
  for (const b of bodies) ke += b.mass * (b.vcm[0] * b.vcm[0] + b.vcm[1] * b.vcm[1] + b.vcm[2] * b.vcm[2]);
  return 0.5 * ke;
};

/**
 * fix_rigid.cpp extract_erotational / fix_rigid_small.cpp extract_erotational: 0.5 sum_b sum_k I_k w_k^2 with the body-frame
 * angular velocity w = R^T L / I (a zero principal moment gives w_k = 0).
 */
export const extractErotational = (bodies: RigidBodyView[]): number => {
  let erotate = 0.0;
  for (const b of bodies) {
    const rot = quatToMatNative(b.quat);
    const wbody = transposeMatvecNative(rot, b.angmom);
    for (let k = 0; k < 3; k++) {
      if (b.inertia[k] === 0.0) wbody[k] = 0.0;
      else wbody[k] /= b.inertia[k];
    }
    erotate += b.inertia[0] * wbody[0] * wbody[0] + b.inertia[1] * wbody[1] * wbody[1] + b.inertia[2] * wbody[2] * wbody[2];
  }
  return 0.5 * erotate;
};
