import type { Pair } from '../force/types';
import { PairThreebodyTable } from '../force/pair/threebody_table';
import { PairSWAngleTable } from '../force/pair/sw_angle_table';

/** Tabulated many-body pair styles (threebody/table, sw/angle/table); lazily loaded (lazyStyles.ts). Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'threebody/table': () => new PairThreebodyTable(),
  'sw/angle/table': () => new PairSWAngleTable(),
};
