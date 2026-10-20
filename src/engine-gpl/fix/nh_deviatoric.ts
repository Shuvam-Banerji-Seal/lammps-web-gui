/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_nh.cpp (FixNH::compute_sigma, compute_deviatoric, compute_strain_energy) and src/domain.cpp
 * (Domain::set_global_box, the h_inv of the box) at commit b332f28. Copyright (2003) Sandia Corporation, distributed
 * under the GNU General Public License (the LAMMPS file headers). Contributing authors of fix_nh.cpp (from the source):
 * Mark Stevens (SNL), Aidan Thompson (SNL). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The deviatoric (non-hydrostatic) part of the Nose-Hoover barostat, used when one of the keywords x, y, z, xy, yz or
 * xz sets native's deviatoric_flag: sigma = vol0 h0_inv (p_target - p_hydro) h0_inv^t from the reference cell, the
 * barostat force fdev = h sigma h^t of the current cell, and the strain energy 0.5 Tr(sigma h h^t). Every tensor is
 * the upper triangle in LAMMPS' Voigt order (0 1 2 3 4 5 = xx yy zz yz xz xy), and the operation order of the native
 * code is kept. The core FixNH (src/engine/fix/nh.ts) reaches it through the lazy HOOKS words x, y, z, xy, yz, xz
 * (registry/nh_deviatoric.ts); fix npt/cauchy (../extrafix/fix_npt_cauchy.ts) reaches it through the core.
 */

/** Domain::set_global_box h_inv of the box h = (xprd, yprd, zprd, yz, xz, xy); the tilt entries are 0 for an orthogonal box. */
export const boxHInv = (h: ArrayLike<number>, triclinic: boolean): number[] => [
  1.0 / h[0],
  1.0 / h[1],
  1.0 / h[2],
  triclinic ? -h[3] / (h[1] * h[2]) : 0.0,
  triclinic ? (h[3] * h[5] - h[1] * h[4]) / (h[0] * h[1] * h[2]) : 0.0,
  triclinic ? -h[5] / (h[0] * h[1]) : 0.0,
];

/** FixNH::compute_sigma (without the nreset step, which the caller does): sigma = vol0 h0inv (p_target - p_hydro) h0inv^t. */
export const computeSigma = (vol0: number, h0Inv: ArrayLike<number>, p: ArrayLike<number>, pHydro: number): number[] => {
  const h = h0Inv;
  return [
    vol0 * (h[0] * ((p[0] - pHydro) * h[0] + p[5] * h[5] + p[4] * h[4]) +
      h[5] * (p[5] * h[0] + (p[1] - pHydro) * h[5] + p[3] * h[4]) +
      h[4] * (p[4] * h[0] + p[3] * h[5] + (p[2] - pHydro) * h[4])),
    vol0 * (h[1] * ((p[1] - pHydro) * h[1] + p[3] * h[3]) +
      h[3] * (p[3] * h[1] + (p[2] - pHydro) * h[3])),
    vol0 * (h[2] * ((p[2] - pHydro) * h[2])),
    vol0 * (h[1] * (p[3] * h[2]) + h[3] * ((p[2] - pHydro) * h[2])),
    vol0 * (h[0] * (p[4] * h[2]) + h[5] * (p[3] * h[2]) + h[4] * ((p[2] - pHydro) * h[2])),
    vol0 * (h[0] * (p[5] * h[1] + p[4] * h[3]) +
      h[5] * ((p[1] - pHydro) * h[1] + p[3] * h[3]) +
      h[4] * (p[3] * h[1] + (p[2] - pHydro) * h[3])),
  ];
};

/** FixNH::compute_deviatoric: the barostat force fdev = h sigma h^t (units of P V) of the current box h. */
export const computeDeviatoric = (h: ArrayLike<number>, s: ArrayLike<number>): number[] => [
  h[0] * (s[0] * h[0] + s[5] * h[5] + s[4] * h[4]) +
    h[5] * (s[5] * h[0] + s[1] * h[5] + s[3] * h[4]) +
    h[4] * (s[4] * h[0] + s[3] * h[5] + s[2] * h[4]),
  h[1] * (s[1] * h[1] + s[3] * h[3]) +
    h[3] * (s[3] * h[1] + s[2] * h[3]),
  h[2] * (s[2] * h[2]),
  h[1] * (s[3] * h[2]) +
    h[3] * (s[2] * h[2]),
  h[0] * (s[4] * h[2]) +
    h[5] * (s[3] * h[2]) +
    h[4] * (s[2] * h[2]),
  h[0] * (s[5] * h[1] + s[4] * h[3]) +
    h[5] * (s[1] * h[1] + s[3] * h[3]) +
    h[4] * (s[3] * h[1] + s[2] * h[3]),
];

/** FixNH::compute_strain_energy: 0.5 Tr(sigma h h^t) / nktv2p, in energy units. */
export const computeStrainEnergy = (h: ArrayLike<number>, s: ArrayLike<number>, nktv2p: number): number => {
  const d0 =
    s[0] * (h[0] * h[0] + h[5] * h[5] + h[4] * h[4]) +
    s[5] * (h[1] * h[5] + h[3] * h[4]) +
    s[4] * (h[2] * h[4]);
  const d1 =
    s[5] * (h[5] * h[1] + h[4] * h[3]) +
    s[1] * (h[1] * h[1] + h[3] * h[3]) +
    s[3] * (h[2] * h[3]);
  const d2 =
    s[4] * (h[4] * h[2]) +
    s[3] * (h[3] * h[2]) +
    s[2] * (h[2] * h[2]);
  return 0.5 * (d0 + d1 + d2) / nktv2p;
};
