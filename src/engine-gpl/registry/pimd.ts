/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of src/REPLICA/fix_pimd_nvt.h (FixStyle(pimd,FixPIMDNVT), FixStyle(pimd/nvt,...)),
 * src/REPLICA/fix_pimd_nvt_bosonic.h (FixStyle(pimd/nvt/bosonic,FixPIMDBNVT)), src/REPLICA/fix_pimd_langevin.h and
 * src/REPLICA/fix_pimd_langevin_bosonic.h at commit b332f28; the classes are src/REPLICA/fix_pimd_nvt.cpp,
 * fix_pimd_nvt_bosonic.cpp (see ../pimd/). The two langevin styles are refused (their factories throw): their port is
 * not done (fix_pimd_langevin.cpp is 1945 lines, with the normal-mode unwrapping and the barostat).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_pimd); the core does not register these names.
 */
import { StyleError } from '../../engine/force/types';
import type { FixFactory } from '../../engine/styles';
import { FixPIMDNVT } from '../pimd/fix_pimd_nvt';
import { FixPIMDNVTBosonic } from '../pimd/fix_pimd_nvt_bosonic';

const refuseLangevin = (name: string): FixFactory => () => {
  throw new StyleError(`fix ${name} is not ported to the browser engine yet (LAMMPS src/REPLICA/fix_pimd_langevin.cpp: the Langevin/PILE integrator and the barostat are not implemented); fix pimd/nvt and pimd/nvt/bosonic are`);
};

export const FIXES: Record<string, FixFactory> = {
  pimd: (sys, id, group, args) => new FixPIMDNVT(sys, id, group, args),
  'pimd/nvt': (sys, id, group, args) => new FixPIMDNVT(sys, id, group, args),
  'pimd/nvt/bosonic': (sys, id, group, args) => new FixPIMDNVTBosonic(sys, id, group, args),
  'pimd/langevin': refuseLangevin('pimd/langevin'),
  'pimd/langevin/bosonic': refuseLangevin('pimd/langevin/bosonic'),
};
