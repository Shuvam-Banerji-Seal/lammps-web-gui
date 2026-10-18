/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the styles of src/GRANULAR
 * (compute contact/atom, fix addtorque/atom, fix damping/cundall, fix viscous/nonlinear), src/SPH (pair sph/lj, compute
 * sph/t/atom) and src/TALLY (compute pe/mol/tally). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_gransph; the names listed there must match).
 */
import type { Pair } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { PairSPHLJ } from '../gransph/pair_sph_lj';
import { ComputeSPHTAtom } from '../gransph/compute_sph_t_atom';
import { ComputeContactAtom } from '../gransph/compute_contact_atom';
import { FixAddTorqueAtom } from '../gransph/fix_addtorque_atom';
import { FixDampingCundall } from '../gransph/fix_damping_cundall';
import { FixViscousNonlinear } from '../gransph/fix_viscous_nonlinear';
import { ComputePEMolTally } from '../gransph/compute_pe_mol_tally';
import { ComputeFabric } from '../gransph/compute_fabric';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'sph/lj': () => new PairSPHLJ(),
};

/** fix styles. */
export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'addtorque/atom': (sys, id, group, args) => new FixAddTorqueAtom(sys, id, group, args),
  'damping/cundall': (sys, id, group, args) => new FixDampingCundall(sys, id, group, args),
  'viscous/nonlinear': (sys, id, group, args) => new FixViscousNonlinear(sys, id, group, args),
};

/** compute styles. */
export const COMPUTES: Record<string, (sys: System, id: string, group: string, args: string[]) => Compute> = {
  'contact/atom': (sys, id, group, args) => new ComputeContactAtom(sys, id, group, args),
  'fabric': (sys, id, group, args) => new ComputeFabric(sys, id, group, args),
  'pe/mol/tally': (sys, id, group, args) => new ComputePEMolTally(sys, id, group, args),
  'sph/t/atom': (sys, id, group, args) => new ComputeSPHTAtom(sys, id, group, args),
};
