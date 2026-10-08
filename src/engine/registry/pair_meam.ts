import type { Pair } from '../force/types';
import { PairMeam } from '../force/pair/meam';

/** pair_style meam (single element; docs.lammps.org/pair_meam.html). Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  meam: () => new PairMeam(),
};
