import type { ComputeFactory } from '../styles';
import { ComputeMSDNonGauss } from '../compute/msd_nongauss';

/**
 * Computes of the Haiku wave 11 restrain work (msd/nongauss). NOT YET WIRED into
 * styles.ts: the orchestrator must spread COMPUTES from this file into COMPUTE_STYLES.
 */
export const COMPUTES: Record<string, ComputeFactory> = {
  'msd/nongauss': (sys, id, group, args) => new ComputeMSDNonGauss(sys, id, group, args),
};
