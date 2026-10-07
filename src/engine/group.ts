import { StyleError } from './force/types';

/*
 * Groups — docs.lammps.org/group.html: "Identify a collection of atoms as
 * belonging to a group." "A group with the ID all is predefined. All atoms
 * belong to this group." "There can be no more than 32 groups defined at
 * one time, including "all"." Membership is one bit per group in the per-atom
 * mask (bit 0 = all).
 */

export const MAX_GROUPS = 32;

export class Groups {
  /** Name per bit index (null = free); index 0 is 'all'. */
  names: (string | null)[] = ['all'];

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
    if (!/^[A-Za-z0-9_]+$/.test(name)) throw new StyleError(`group ID '${name}' must be alphanumeric or underscore`);
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
    return k;
  }

  list(): string[] {
    return this.names.filter((n): n is string => n !== null);
  }
}

export const bitOfIndex = (k: number): number => (k === 31 ? -0x80000000 : 1 << k);
