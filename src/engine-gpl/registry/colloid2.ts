/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/COLLOID/pair_lubricate.cpp,
 * pair_lubricateU.cpp, pair_brownian.cpp and their _poly variants, fix_wall_colloid.cpp (Randy Schunk (SNL), Amit Kumar
 * and Michael Bybee (UIUC), Sandia Corporation, GNU General Public License) and src/SRD/fix_wall_srd.cpp. This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_colloid2); the names there must match these keys.
 */
import type { Pair } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { PairLubricate, PairLubricatePoly } from '../colloid2/pair_lubricate';
import { PairLubricateU } from '../colloid2/pair_lubricateU';
import { PairLubricateUPoly } from '../colloid2/pair_lubricateU_poly';
import { PairBrownian } from '../colloid2/pair_brownian';
import { PairBrownianPoly } from '../colloid2/pair_brownian_poly';
import { FixWallColloid } from '../colloid2/fix_wall_colloid';
import { FixWallSRD } from '../colloid2/fix_wall_srd';

export const PAIRS: Record<string, () => Pair> = {
  lubricate: () => new PairLubricate(),
  'lubricate/poly': () => new PairLubricatePoly(),
  lubricateU: () => new PairLubricateU(),
  'lubricateU/poly': () => new PairLubricateUPoly(),
  brownian: () => new PairBrownian(),
  'brownian/poly': () => new PairBrownianPoly(),
};

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'wall/colloid': (sys, id, group, args) => new FixWallColloid(sys, id, group, args),
  'wall/srd': (sys, id, group, args) => new FixWallSRD(sys, id, group, args),
};
