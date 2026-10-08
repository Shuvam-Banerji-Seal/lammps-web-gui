import type { ComputeFactory } from '../styles';
import { ComputeEventDisplace } from '../compute/event_displace';

/** Computes of the Haiku wave 13; merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'event/displace': (sys, id, group, args) => new ComputeEventDisplace(sys, id, group, args),
};
