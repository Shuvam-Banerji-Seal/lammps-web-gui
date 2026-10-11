import type { Pair } from '../force/types';
import { PairMeamSpline, PairMeamSWSpline } from '../force/pair/meam_spline';

/** Spline-based MEAM pair styles (meam/spline, meam/sw/spline); merged into styles.ts via lazyStyles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'meam/spline': () => new PairMeamSpline(),
  'meam/sw/spline': () => new PairMeamSWSpline(),
};
