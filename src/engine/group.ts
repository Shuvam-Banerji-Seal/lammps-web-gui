import { StyleError } from './force/types';

/*
 * Groups — docs.lammps.org/group.html: "Identify a collection of atoms as
 * belonging to a group." "A group with the ID all is predefined. All atoms
 * belong to this group." "There can be no more than 32 groups defined at
 * one time, including "all"." Membership is one bit per group in the per-atom
 * mask (bit 0 = all).
 */

export const MAX_GROUPS = 32;

/*
 * Dynamic groups — docs.lammps.org/group.html: "The *dynamic* style flags an
 * existing or new group as dynamic.  This means atoms will be (re)assigned to
 * the group periodically as a simulation runs." "Only atoms in the group
 * specified as the parent group via the parent-ID are assigned to the dynamic
 * group before the following conditions are applied." "The assignment of
 * atoms to a dynamic group is done at the beginning of each run and on every
 * timestep that is a multiple of *N*\ , which is the argument for the *every*
 * keyword (:math:`N = 1` is the default)." "The *static* style removes the
 * setting for a dynamic group, converting it to a static group (the default).
 * The atoms in the static group are those currently in the dynamic group."
 */
export interface DynamicGroup {
  parent: string;
  region: string | null;
  variable: string | null;
  property: string | null;
  every: number;
}

/*
 * Measured with native LAMMPS (black box, a dynamic group whose membership
 * changes during the run): given a dynamic group, the fix styles in
 * NO_DYNAMIC_GROUP_FIXES stop with the error Fix <style> does not allow use
 * with a dynamic group, and compute msd with Compute msd is not compatible
 * with dynamic groups. The fixes in
 * DYNAMIC_GROUP_FIXES accept one; every other compute style registered here
 * accepts one too. A fix style in neither list has not been measured.
 */
export const DYNAMIC_GROUP_FIXES = new Set([
  'addforce', 'ave/histo', 'ave/time', 'aveforce', 'efield', 'gravity', 'langevin', 'lineforce',
  'momentum', 'nph', 'nph/sphere', 'npt', 'npt/sphere', 'nve', 'nve/limit', 'nve/sphere', 'nvt',
  'nvt/sllod', 'nvt/sphere', 'planeforce', 'recenter', 'setforce', 'spring', 'temp/berendsen',
  'temp/csld', 'temp/csvr', 'temp/rescale', 'viscous', 'wall/harmonic', 'wall/lj1043', 'wall/lj126',
  'wall/lj93', 'wall/morse', 'wall/reflect', 'wall/reflect/stochastic', 'wall/region',
]);
export const NO_DYNAMIC_GROUP_FIXES = new Set([
  'ave/atom', 'ave/chunk', 'balance', 'deform', 'deposit', 'ehex', 'enforce2d', 'freeze', 'heat', 'indent',
  'nve/noforce', 'pour', 'print',
  'property/atom', 'rattle', 'rigid', 'rigid/nve', 'rigid/nve/small', 'rigid/nvt', 'rigid/nvt/small',
  'rigid/small', 'shake', 'spring/self', 'vector', 'wall/gran', 'wall/gran/region',
]);

export class Groups {
  /** Name per bit index (null = free); index 0 is 'all'. */
  names: (string | null)[] = ['all'];
  /** Dynamic groups by mask bit. */
  readonly dynamic = new Map<number, DynamicGroup>();

  isDynamic(bit: number): boolean {
    return this.dynamic.has(bit);
  }

  find(name: string): number {
    return this.names.indexOf(name);
  }

  /** Mask bit of an existing group, or a StyleError naming it. */
  bit(name: string): number {
    const k = this.find(name);
    if (k < 0) throw new StyleError(`unknown group '${name}'`);
    return k === 31 ? -0x80000000 : 1 << k;
  }

  /** Bit index of the group, creating it if needed. */
  create(name: string): number {
    const k = this.find(name);
    if (k >= 0) return k;
    // group.html sets no character rule for group IDs (unlike fix, compute and variable IDs), and native
    // LAMMPS accepts e.g. lo-fixed, a.b, a+b, a/b and 1abc, also inside count() (measured 2026-10-07)
    if (name === '') throw new StyleError('group ID must not be empty');
    let free = this.names.indexOf(null);
    if (free < 0) {
      if (this.names.length >= MAX_GROUPS) throw new StyleError('too many groups (at most 32, including all)');
      free = this.names.length;
      this.names.push(name);
    } else this.names[free] = name;
    return free;
  }

  delete(name: string): number {
    if (name === 'all') throw new StyleError('group all cannot be deleted');
    const k = this.find(name);
    if (k < 0) throw new StyleError(`unknown group '${name}'`);
    this.names[k] = null;
    this.dynamic.delete(bitOfIndex(k));
    return k;
  }

  list(): string[] {
    return this.names.filter((n): n is string => n !== null);
  }
}

export const bitOfIndex = (k: number): number => (k === 31 ? -0x80000000 : 1 << k);
