/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the APIP package (src/APIP:
 * pair_eam_apip.cpp, pair_eam_fs_apip.cpp, pair_pace_apip.cpp, pair_pace_fast_apip.cpp, pair_pace_precise_apip.cpp)
 * at commit b332f28 (see ../apip/). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_apip; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import type { FixFactory } from '../../engine/styles';
import { FixAtomWeightAPIP } from '../apip/fix_atom_weight_apip';
import { FixLambdaAPIP } from '../apip/fix_lambda_apip';
import { FixLambdaLACSPAPIP } from '../apip/fix_lambda_la_csp_apip';
import { FixLambdaThermostatAPIP } from '../apip/fix_lambda_thermostat_apip';
import { PairLambdaInputAPIP, PairLambdaInputCSPAPIP } from '../apip/pair_lambda_input_apip';
import { PairLambdaZoneAPIP } from '../apip/pair_lambda_zone_apip';
import { PairEAMAPIP, PairEAMFSAPIP } from '../apip/pair_eam_apip';
import { PairPaceAPIP, PairPaceFastAPIP, PairPacePreciseAPIP } from '../apip/pair_pace_apip';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'eam/apip': () => new PairEAMAPIP(),
  'eam/fs/apip': () => new PairEAMFSAPIP(),
  'pace/apip': () => new PairPaceAPIP(),
  'pace/precise/apip': () => new PairPacePreciseAPIP(),
  'pace/fast/apip': () => new PairPaceFastAPIP(),
  'lambda/input/apip': () => new PairLambdaInputAPIP(),
  'lambda/input/csp/apip': () => new PairLambdaInputCSPAPIP(),
  'lambda/zone/apip': () => new PairLambdaZoneAPIP(),
};

/** fix names. */
export const FIXES: Record<string, FixFactory> = {
  'atom_weight/apip': (sys, id, group, args) => new FixAtomWeightAPIP(sys, id, group, args),
  'lambda/apip': (sys, id, group, args) => new FixLambdaAPIP(sys, id, group, args),
  'lambda_thermostat/apip': (sys, id, group, args) => new FixLambdaThermostatAPIP(sys, id, group, args),
  'lambda/la/csp/apip': (sys, id, group, args) => new FixLambdaLACSPAPIP(sys, id, group, args),
};
