import type { SimState } from './types';
import { libm } from './libm';

/*
 * atom_style ellipsoid superellipsoid: the per-atom blockiness exponents and the moments of inertia of a superellipsoid
 * (core, independent of the GPL ports; the contact model is in src/engine-gpl/superellipsoid).
 *
 * docs.lammps.org/atom_style.html: "By adding the flag *superellipsoid* to the *ellipsoid* atom_style command, the
 * particles can be superellipsoids, which are a generalization of ellipsoids with two additional blockiness parameters
 * that control the shape." "Superellipsoids also store the principal moments of inertia of the particle."
 *
 * docs.lammps.org/pair_granular_superellipsoid.html gives the surface as
 *   f(x) = ( |x/a|^n2 + |y/b|^n2 )^(n1/n2) + |z/c|^n1 - 1 = 0,
 * with a, b, c the half-diameters (the *shape* property) and n1, n2 the *block* property ("The half-diameters a, b, and c
 * correspond to the *shape* property, and the exponents n1 and n2 to the *block* property of the ellipsoid atom").
 * docs.lammps.org/set.html: "The *block1*, *block2* settings are the 2 exponents of the superellipsoid in the vertical
 * and horizontal directions." "If both parameters are set to a value of 2 (the default), the atom is a regular ellipsoid."
 *
 * The moments of inertia are integrals of the uniform solid bounded by that surface. Substituting x = a l X, y = b l Y,
 * z = c Z with l = (1 - |Z|^n1)^(1/n1) and |X|^n2 + |Y|^n2 <= 1 gives dV = a b c l^2 dX dY dZ, and the Dirichlet integrals
 *   M0 = int int dX dY = 4 G(1/p)^2 / (p^2 G(2/p+1)),   Q2 = int int X^2 dX dY = 4 G(3/p) G(1/p) / (p^2 G(4/p+1)),
 *   W(q,k) = int_-1^1 |Z|^q (1-|Z|^n1)^(k/n1) dZ = (2/n1) B((q+1)/n1, k/n1+1)   (p = n2, G the gamma function, B the beta
 * function). With V = a b c M0 W(0,2) and mass m:
 *   Ix = m ( b^2 Q2 W(0,4) / (M0 W(0,2)) + c^2 W(2,2) / W(0,2) ),   Iy likewise with a^2, and Iz = m (a^2+b^2) Q2 W(0,4) / (M0 W(0,2)).
 * Check: n1 = n2 = 2 gives m/5 (b^2+c^2) etc., the ellipsoid moments.
 *
 * Measured with native LAMMPS (black box): the inertia is stored when the shape or block of an atom is set, with the mass
 * the atom has at that time: "set type 1 shape ...", "set type 1 mass 3.0" (later) leaves the inertia of mass 1, and
 * "set type 1 block 3 4" after "set type 1 mass 3.0" uses mass 3 (the same for "set shape" after "set mass").
 * Measured with native LAMMPS (black box): shape 2 1 1 (diameters) with block 3 4 and mass 1 gives the inertia
 * 0.127667540830386 0.324596661116689 0.328215200477172, equal to the formula above to 1e-13.
 */

/** Per-atom blockiness exponents (block1, block2), 2 values per atom; the default is the regular ellipsoid (2, 2). */
export const SUPER_BLOCK = '#sblock';
/** Per-atom principal moments of inertia (body frame), 3 values per atom, stored when shape or block is set. */
export const SUPER_INERTIA = '#sinertia';
export const SUPER_DEFAULTS: Record<string, number> = { [SUPER_BLOCK]: 2, [SUPER_INERTIA]: 0 };

/**
 * Gamma function for x > 0: the reflection formula below 1/2, a shift up to x >= 15, then Stirling's series for ln G
 * (about 1e-15 relative accuracy for the arguments of the inertia formula).
 */
export const gammaFn = (x: number): number => {
  if (x < 0.5) return Math.PI / (libm.sin(Math.PI * x) * gammaFn(1 - x));
  let z = x, prod = 1;
  while (z < 15) {
    prod *= z;
    z += 1;
  }
  const z2 = 1 / (z * z);
  const series = (1 / z) * (1 / 12 - z2 * (1 / 360 - z2 * (1 / 1260 - z2 * (1 / 1680 - z2 * (1 / 1188 - z2 * (691 / 360360 - z2 / 156))))));
  const lnG = (z - 0.5) * libm.log(z) - z + 0.5 * libm.log(2 * Math.PI) + series;
  return Math.exp(lnG) / prod;
};

/** Beta function B(a, b) = G(a) G(b) / G(a + b) for a, b > 0. */
const betaFn = (a: number, b: number): number => (gammaFn(a) * gammaFn(b)) / gammaFn(a + b);

/**
 * Principal moments of inertia (body frame, x y z) of a uniform solid superellipsoid with half-diameters a, b, c, blockiness
 * n1 (vertical, z) and n2 (horizontal, x-y) and mass m. n1 = n2 = 2 gives the ellipsoid moments m/5 (b^2+c^2) etc.
 */
export const superellipsoidInertia = (a: number, b: number, c: number, n1: number, n2: number, m: number): [number, number, number] => {
  const p = n2;
  const M0 = (4 / (p * p)) * (gammaFn(1 / p) ** 2) / gammaFn(2 / p + 1);
  const Q2 = (4 / (p * p)) * gammaFn(3 / p) * gammaFn(1 / p) / gammaFn(4 / p + 1);
  const W = (q: number, k: number): number => (2 / n1) * betaFn((q + 1) / n1, k / n1 + 1);
  const W02 = W(0, 2), W22 = W(2, 2), W04 = W(0, 4);
  const xy = (Q2 * W04) / (M0 * W02);
  const zz = W22 / W02;
  return [m * (b * b * xy + c * c * zz), m * (a * a * xy + c * c * zz), m * (a * a + b * b) * xy];
};

/** Blockiness exponents (block1, block2) of atom i. */
export const superBlock = (s: SimState, i: number): [number, number] => {
  const d = s.custom.get(SUPER_BLOCK)!.data;
  return [d[2 * i], d[2 * i + 1]];
};

/** Recomputes the stored inertia of atom i from its current shape, block and mass (set shape and set block do this). */
export const storeSuperInertia = (s: SimState, i: number): void => {
  const sh = s.shape!;
  const [b1, b2] = superBlock(s, i);
  const I = superellipsoidInertia(sh[3 * i], sh[3 * i + 1], sh[3 * i + 2], b1, b2, s.rmass![i]);
  const out = s.custom.get(SUPER_INERTIA)!.data;
  out[3 * i] = I[0]; out[3 * i + 1] = I[1]; out[3 * i + 2] = I[2];
};
