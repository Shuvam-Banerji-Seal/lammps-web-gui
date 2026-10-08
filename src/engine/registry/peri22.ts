import type { Pair } from '../force/types';
import type { ComputeFactory } from '../styles';
import { PairPeri } from '../force/pair/peri';
import { ComputeDamageAtom, ComputeDilatationAtom } from '../compute/peri';

/** Peridynamics (PERI package, wave 22): pair styles and per-atom computes; merged into styles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  'peri/pmb': () => new PairPeri('pmb'),
  'peri/lps': () => new PairPeri('lps'),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  'damage/atom': (sys, id, group, args) => new ComputeDamageAtom(sys, id, group, args),
  'dilatation/atom': (sys, id, group, args) => new ComputeDilatationAtom(sys, id, group, args),
};
