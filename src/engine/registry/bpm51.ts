import type { Pair } from '../force/types';
import { PairBpmSpring } from '../force/pair/bpm';
import { FixNVEBpmSphere } from '../fix/nve_bpm_sphere';
import type { FixFactory } from '../styles';

/** BPM package styles (waves 51 and 53; the bond styles moved to engine-gpl); loaded lazily (lazyStyles.ts). Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'bpm/spring': () => new PairBpmSpring(),
};
export const FIXES: Record<string, FixFactory> = {
  'nve/bpm/sphere': (s, i, g, a) => new FixNVEBpmSphere(s, i, g, a),
};
