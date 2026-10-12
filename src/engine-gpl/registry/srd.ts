/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registration of src/SRD/fix_srd.cpp (FixStyle(srd,FixSRD)) at commit b332f28; the port is
 * in ../srd/fix_srd.ts (copyright Sandia Corporation, LAMMPS development team, GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group srd); the names there must match these keys.
 */
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { FixSRD } from '../srd/fix_srd';

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  srd: (sys, id, group, args) => new FixSRD(sys, id, group, args),
};
