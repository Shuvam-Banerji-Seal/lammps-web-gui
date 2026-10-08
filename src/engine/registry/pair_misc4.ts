import type { Pair } from '../force/types';
import { PairLJSmoothLinear } from '../force/pair/lj_variants';
import { PairBornCoulLongCS, PairBuckCoulLongCS, PairCoulLongCS } from '../force/pair/coul_cs';

/** Pair styles of the Haiku wave 9 (lj/sf, core-shell born/buck/coul long); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  // lj/sf is not in the current docs. Measured with native LAMMPS (black box): pair_style lj/sf
  // gives the same energies, pressures and write_data output as lj/smooth/linear
  // (docs.lammps.org/pair_lj_smooth_linear.html) to the last printed digit, so it is that style.
  'lj/sf': () => Object.assign(new PairLJSmoothLinear(), { name: 'lj/sf' }),
  'born/coul/long/cs': () => new PairBornCoulLongCS(),
  'buck/coul/long/cs': () => new PairBuckCoulLongCS(),
  'coul/long/cs': () => new PairCoulLongCS(),
};
