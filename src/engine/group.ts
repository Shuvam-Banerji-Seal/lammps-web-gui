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
 *
 * Keyword semantics (group.html, dynamic section). Measured with native LAMMPS (black box, a
 * per-atom probe of the members of each dynamic group after run 0): the region, var and property
 * keywords select the parent atoms first; within then adds every atom that is not in the parent
 * group and lies within the cutoff of one of those selected atoms (the test is inclusive, so an
 * atom at exactly the cutoff is added, and periodic images count); include molecule then adds
 * every atom that shares a molecule ID with a member; exclude removes last. With a region and
 * the parent group all, the atoms outside the region are not added by within even when they
 * are within the cutoff of a selected atom.
 */
export interface DynamicGroup {
  parent: string;
  region: string | null;
  variable: string | null;
  property: string | null;
  every: number;
  /** within keyword: cutoff distance (null = not set). */
  within: number | null;
  /** include molecule keyword. */
  includeMolecule: boolean;
  /** exclude keyword: group ID (null = not set). */
  exclude: string | null;
}

/*
 * Fixes on a dynamic group. As native LAMMPS does (src/modify.cpp Modify::init at b332f28): a fix whose
 * dynamic_group_allow is 0 (the Fix base class default; a style sets it to 1 in its constructor, or inherits it) stops
 * with Fix <style> does not allow use with a dynamic group. DYNAMIC_GROUP_FIXES are the styles that set it and whose
 * engine implementation re-reads the group at every step (checked against native on a dynamic group, e.g. the oracle
 * w435dyngroup_fixes); DYNAMIC_GROUP_UNVERIFIED are the styles that set it in native but whose engine implementation
 * has not been checked with a dynamic group, refused by name until it is. Measured with native LAMMPS (black box):
 * compute msd stops with Compute msd is not compatible with dynamic groups; every other compute style accepts one.
 */
// add/heat: src/GRANULAR/fix_add_heat.cpp (b332f28) sets dynamic_group_allow = 1 in its constructor.
// graphics/arrows, graphics/objects, graphics/periodic, graphics/replica, graphics/chunk, graphics/labels:
// src/GRAPHICS/fix_graphics_arrows.cpp, fix_graphics_objects.cpp, fix_graphics_periodic.cpp, fix_graphics_replica.cpp,
// fix_graphics_chunk.cpp and fix_graphics_labels.cpp set dynamic_group_allow = 1.
// npt/cauchy: src/EXTRA-FIX/fix_npt_cauchy.cpp (b332f28) sets dynamic_group_allow = 1 in its constructor.
export const DYNAMIC_GROUP_FIXES = new Set([
  'add/heat', 'addforce', 'addtorque/group', 'ave/histo', 'ave/histo/weight', 'ave/moments', 'ave/time', 'aveforce', 'dt/reset', 'efield', 'graphics/arrows', 'graphics/chunk', 'graphics/labels', 'graphics/objects', 'graphics/periodic', 'graphics/replica', 'gravity', 'langevin', 'lineforce',
  'baoab', 'momentum', 'nph', 'nph/sphere', 'npt', 'npt/cauchy', 'npt/sphere', 'nve', 'nve/limit', 'nve/sphere', 'nvt',
  'nvt/sllod', 'nvt/sphere', 'planeforce', 'recenter', 'setforce', 'spring', 'temp/berendsen',
  'temp/csld', 'temp/csvr', 'temp/rescale', 'viscous', 'viscous/sphere', 'wall/harmonic', 'wall/lj1043', 'wall/lj126',
  'wall/lj93', 'wall/morse', 'wall/reflect', 'wall/reflect/stochastic', 'wall/region',
  'wall/harmonic/outside', 'wall/harmonic/returned', 'wall/lepton', 'addtorque', 'spring/rg', 'oneway',
  'addtorque/atom', 'settorque/atom', 'viscous/nonlinear', 'damping/cundall', 'nve/bpm/sphere', 'heat/flow',
  'nve/asphere', 'nvt/asphere', 'nph/asphere', 'npt/asphere',
]);
export const DYNAMIC_GROUP_UNVERIFIED = new Set([
  'adapt', 'adapt/fep', 'atom/swap', 'bocs', 'bond/create',
  'bond/create/angle', 'efield/tip4p', 'gcmc', 'graphics/isosurface',
  'langevin/eff', 'mvv/dpd', 'mvv/edpd', 'mvv/tdpd', 'neighbor/swap',
  'nph/body', 'nph/eff', 'nphug', 'npt/body', 'npt/eff',
  'npt/uef', 'nve/body', 'nve/dot', 'nve/dotc/langevin',
  'nve/line', 'nve/manifold/rattle', 'nve/tri', 'nvk', 'nvt/body',
  'nvt/eff', 'nvt/manifold/rattle', 'nvt/sllod/eff', 'nvt/uef', 'precession/spin', 'reaxff/bonds',
  'reaxff/species', 'setforce/spin', 'smd/setvel', 'wall/flow',
  'wall/table', 'widom',
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
