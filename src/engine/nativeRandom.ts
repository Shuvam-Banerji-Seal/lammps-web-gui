/*
 * The per-atom random draws of commands whose documentation promises results that do not depend on the
 * processor count, so a given seed must give native LAMMPS's numbers:
 *
 * velocity.html: "For each atom a unique random number seed is created, based on the atom's xyz coordinates."
 * and "the velocity assigned to a particular atom will be the same, independent of how many MPI processes are
 * used."
 * displace_atoms.html: "Random numbers are used in such a way that the displacement of a particular atom is the
 * same, regardless of how many processors are being used."
 * set.html, type/fraction: "Random numbers are used in such a way that a particular atom is changed or not
 * changed, regardless of how many processors are being used." (dipole/random, spin/atom/random, quat/random
 * and theta/random say the same of the orientation.) type/ratio and type/subset: "Which atoms are selected will
 * change with the number of processors used." (the engine is one processor).
 *
 * The seeding from coordinates is not documented, so the implementation is ported from LAMMPS
 * (src/engine-gpl/, GPL-2.0) and loaded lazily: lazyStyles.ts lists it under the HOOKS kind with the command
 * words that need it, so Session.dispatch loads it before such a command runs. Core code reaches it only
 * through nativeRandom() in styles.ts.
 *
 * Every method takes the user seed and the atom's current coordinates (x, y, z, the exact stored values: one
 * bit of difference gives a different stream).
 */

export interface NativeRandom {
  /** velocity create ... loop geom: the three deviates of one atom (uniform() - 0.5 each, or gaussian() each), before the 1/sqrt(mass) scaling. */
  velocityGeom(seed: number, x: number, y: number, z: number, gaussian: boolean, out: Float64Array): void;
  /** displace_atoms random dx dy dz seed: the displacement of one atom (out[2] = 0 in 2d). */
  displaceRandom(seed: number, x: number, y: number, z: number, dx: number, dy: number, dz: number, dimension: number, out: Float64Array): void;
  /** delete_atoms random fraction F seed (eflag no): which of `count` eligible atoms (storage order) are deleted (1 = yes). */
  deleteFraction(seed: number, fraction: number, count: number): Uint8Array;
  /** set type/fraction: true when the atom takes the new type. */
  typeFraction(seed: number, x: number, y: number, z: number, fraction: number): boolean;
  /** set type/ratio and type/subset: which of `count` eligible atoms (storage order, one processor) take the new type (1 = yes). */
  selectSubset(seed: number, nsubset: number, count: number): Uint8Array;
  /** set dipole/random seed Dlen: the moment vector of one atom (out[0..2]; out[2] = 0 in 2d). */
  dipoleRandom(seed: number, x: number, y: number, z: number, dlen: number, dimension: number, out: Float64Array): void;
  /** set spin/atom/random (and spin/random) seed Dlen: the unit spin direction of one atom (out[0..2]; out[2] = 0 in 2d). */
  spinRandom(seed: number, x: number, y: number, z: number, dimension: number, out: Float64Array): void;
  /** set quat/random seed: the quaternion of one atom (out[0..3] = w, i, j, k). */
  quatRandom(seed: number, x: number, y: number, z: number, dimension: number, out: Float64Array): void;
  /** set theta/random seed: the angle of one line atom (radians). */
  thetaRandom(seed: number, x: number, y: number, z: number): number;
}

/** The command words that need the native streams (the lazy group's HOOKS names). */
export const NATIVE_RANDOM_WORDS = [
  'geom', 'random', 'type/fraction', 'type/ratio', 'type/subset', 'dipole/random', 'spin/random', 'spin/atom/random',
  'quat/random', 'theta/random',
] as const;
export type NativeRandomWord = (typeof NATIVE_RANDOM_WORDS)[number];
