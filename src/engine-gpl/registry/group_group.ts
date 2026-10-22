/*
 * Support code for the styles ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the
 * long-range part of compute group/group (../compute/group_group_kspace.ts), the helper of the style word group/group
 * (lazy group gpl_group_group, read by the core compute through gplHook('group/group')).
 */
import { GROUP_GROUP_KSPACE } from '../compute/group_group_kspace';

export const HOOKS = { 'group/group': GROUP_GROUP_KSPACE };
