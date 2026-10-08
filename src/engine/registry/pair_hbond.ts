import type { Pair } from '../force/types';
import { PairHbondDreiding } from '../force/pair/hbond_dreiding';

/** pair_style hbond/dreiding/* (Haiku wave 14); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'hbond/dreiding/lj': () => new PairHbondDreiding('lj', false),
  'hbond/dreiding/morse': () => new PairHbondDreiding('morse', false),
  'hbond/dreiding/lj/angleoffset': () => new PairHbondDreiding('lj', true),
  'hbond/dreiding/morse/angleoffset': () => new PairHbondDreiding('morse', true),
};
