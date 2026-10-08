import type { FixFactory } from '../styles';
import { FixAveCorrelate } from '../fix/gk';
import { FixViscosity, FixThermalConductivity } from '../fix/gk_swap';

/** Fixes of the Haiku wave 11 (fix_gk); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'ave/correlate': (sys, id, group, args) => new FixAveCorrelate(sys, id, group, args),
  viscosity: (sys, id, group, args) => new FixViscosity(sys, id, group, args),
  'thermal/conductivity': (sys, id, group, args) => new FixThermalConductivity(sys, id, group, args),
};
