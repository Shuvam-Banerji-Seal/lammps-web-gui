import type { FixFactory } from '../styles';
import { FixRestrain } from '../fix/restrain';
import { FixSpringRG } from '../fix/spring_rg';
import { FixEvaporate } from '../fix/evaporate';

/** Fixes of the Haiku wave 11 (restrain, spring/rg, evaporate); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  restrain: (sys, id, group, args) => new FixRestrain(sys, id, group, args),
  'spring/rg': (sys, id, group, args) => new FixSpringRG(sys, id, group, args),
  evaporate: (sys, id, group, args) => new FixEvaporate(sys, id, group, args),
};
