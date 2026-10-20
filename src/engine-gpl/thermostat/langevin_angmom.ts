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
import { ellipsoidInertia } from '../../engine/compute/asphere';
import { nativeOrder } from '../../engine/atoms';
import { mqToOmega } from '../asphere/math_extra';

/**
 * Adds the angmom torques of the group atoms. omega = mq_to_omega(angmom, quat, I) with I = 0.2 m (b^2 + c^2) etc.;
 * gamma1 = -ascale / damp / ftm2v and gamma2 = sqrt(ascale 24 kB / damp / dt / mvv2e) / ftm2v, then gamma1 times
 * 1 / ratio and gamma2 times sqrt(T) / sqrt(ratio); torque_k += I_k gamma1 omega_k + sqrt(I_k) gamma2 (u - 0.5). One
 * draw per component and atom, after the omega draws, in native atom order.
 */
export const angmomTorques = (s: System['state'], o: LangevinAngmomOptions): void => {
  const u = s.units;
  const { torque, angmom, quat, mask, type } = s;
  const ascale = o.ascale;
  const omega = new Float64Array(3);
  for (const i of nativeOrder(s)) {
    if (!(mask[i] & o.groupBit)) continue;
    const I = ellipsoidInertia(s, i);
    mqToOmega(angmom![3 * i], angmom![3 * i + 1], angmom![3 * i + 2], quat![4 * i], quat![4 * i + 1], quat![4 * i + 2], quat![4 * i + 3], I, omega);
    const tsqrt = o.tforce ? Math.sqrt(o.tforce[i]) : o.tsqrt;
    const ratio = o.ratio[type[i]];
    let gamma1 = -ascale / o.damp / u.ftm2v;
    let gamma2 = Math.sqrt(ascale * 24.0 * u.boltz / o.damp / s.dt / u.mvv2e) / u.ftm2v;
    gamma1 *= 1.0 / ratio;
    gamma2 *= 1.0 / Math.sqrt(ratio) * tsqrt;
    const t0 = Math.sqrt(I[0]) * gamma2 * (o.uniform() - 0.5);
    const t1 = Math.sqrt(I[1]) * gamma2 * (o.uniform() - 0.5);
    const t2 = Math.sqrt(I[2]) * gamma2 * (o.uniform() - 0.5);
    torque![3 * i] += I[0] * gamma1 * omega[0] + t0;
    torque![3 * i + 1] += I[1] * gamma1 * omega[1] + t1;
    torque![3 * i + 2] += I[2] * gamma1 * omega[2] + t2;
  }
};
