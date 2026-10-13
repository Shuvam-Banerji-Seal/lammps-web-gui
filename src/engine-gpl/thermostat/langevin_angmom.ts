/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_langevin.cpp (FixLangevin::angmom_thermostat) and src/math_extra.cpp (MathExtra::mq_to_omega), at
 * commit b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The rotational Langevin thermostat of fix langevin keyword angmom. The core FixLangevin (src/engine/fix/thermostats.ts)
 * keeps the keyword parsing, the ellipsoid checks and the call; it reaches this routine through the lazy HOOKS entry
 * 'angmom' (lazyStyles.ts group gpl_langevin_angmom, registry/langevin_angmom.ts).
 */
import type { System } from '../../engine/system';
import type { LangevinAngmomOptions } from '../../engine/fix/thermostats';
import { ellipsoidInertia, quatToMatrix, rotateTransposeVec, rotateVec } from '../../engine/compute/asphere';
import { nativeOrder } from '../../engine/atoms';

/**
 * Adds the angmom torques of the group atoms. wb = R^T L / I (I = 0.2 m (b^2 + c^2) etc., mq_to_omega) is damped with
 * gamma1 = -ascale / damp / ftm2v and the random torque sqrt(I) gamma2 (u - 0.5) in body axes, one draw per component
 * and atom, after the omega draws, in native atom order.
 */
export const angmomTorques = (s: System['state'], o: LangevinAngmomOptions): void => {
  const u = s.units;
  const { torque, angmom, quat, mask, type } = s;
  const ascale = o.ascale;
  for (const i of nativeOrder(s)) {
    if (!(mask[i] & o.groupBit)) continue;
    const I = ellipsoidInertia(s, i);
    const R = quatToMatrix(quat![4 * i], quat![4 * i + 1], quat![4 * i + 2], quat![4 * i + 3]);
    const Lb = rotateTransposeVec(R, angmom![3 * i], angmom![3 * i + 1], angmom![3 * i + 2]);
    const wbody = [0, 1, 2].map((d) => (I[d] === 0 ? 0 : Lb[d] / I[d]));
    const w = rotateVec(R, wbody[0], wbody[1], wbody[2]);
    const damp = o.damp * o.ratio[type[i]];
    const gamma1 = -ascale / damp / u.ftm2v;
    const gamma2 = Math.sqrt((ascale * 24 * u.boltz * o.tt) / (damp * s.dt * u.mvv2e)) / u.ftm2v;
    for (let d = 0; d < 3; d++) {
      const tran = Math.sqrt(I[d]) * gamma2 * (o.uniform() - 0.5);
      torque![3 * i + d] += I[d] * gamma1 * w[d] + tran;
    }
  }
};
