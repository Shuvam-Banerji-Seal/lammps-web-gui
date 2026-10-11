import type { Pair } from '../force/types';
import type { ComputeFactory, FixFactory } from '../styles';
import { PairSpinExchange } from '../force/pair/spin_exchange';
import { ComputeSpin } from '../compute/spin';
import { FixNVESpin } from '../fix/nve_spin';
import { FixPrecessionSpin } from '../fix/precession_spin';

/*
 * SPIN package styles (atom_style spin): pair spin/exchange, compute spin, fix nve/spin and fix precession/spin.
 * Loaded lazily (src/engine/lazyStyles.ts).
 */
export const PAIRS: Record<string, () => Pair> = {
  'spin/exchange': () => new PairSpinExchange(),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  spin: (sys, id, group, args) => new ComputeSpin(sys, id, group, args),
};

export const FIXES: Record<string, FixFactory> = {
  'nve/spin': (s, i, g, a) => new FixNVESpin(s, i, g, a),
  'precession/spin': (s, i, g, a) => new FixPrecessionSpin(s, i, g, a),
};
