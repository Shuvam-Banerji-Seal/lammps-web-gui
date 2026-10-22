/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of src/REPLICA/fix_pimd_nvt.h (FixStyle(pimd,FixPIMDNVT), FixStyle(pimd/nvt,...)),
 * src/REPLICA/fix_pimd_nvt_bosonic.h (FixStyle(pimd/nvt/bosonic,FixPIMDBNVT)), src/REPLICA/fix_pimd_langevin.h
 * (FixStyle(pimd/langevin,FixPIMDLangevin)) and src/REPLICA/fix_pimd_langevin_bosonic.h (FixStyle(pimd/langevin/bosonic,
 * FixPIMDBLangevin)) at commit b332f28; the classes are in ../pimd/ (see each file's header).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_pimd); the core does not register these names.
 */
import type { FixFactory } from '../../engine/styles';
import { FixPIMDNVT } from '../pimd/fix_pimd_nvt';
import { FixPIMDNVTBosonic } from '../pimd/fix_pimd_nvt_bosonic';
import { FixPIMDLangevin } from '../pimd/fix_pimd_langevin';
import { FixPIMDLangevinBosonic } from '../pimd/fix_pimd_langevin_bosonic';

export const FIXES: Record<string, FixFactory> = {
  pimd: (sys, id, group, args) => new FixPIMDNVT(sys, id, group, args, 'pimd'),
  'pimd/nvt': (sys, id, group, args) => new FixPIMDNVT(sys, id, group, args),
  'pimd/nvt/bosonic': (sys, id, group, args) => new FixPIMDNVTBosonic(sys, id, group, args),
  'pimd/langevin': (sys, id, group, args) => new FixPIMDLangevin(sys, id, group, args),
  'pimd/langevin/bosonic': (sys, id, group, args) => new FixPIMDLangevinBosonic(sys, id, group, args),
};
