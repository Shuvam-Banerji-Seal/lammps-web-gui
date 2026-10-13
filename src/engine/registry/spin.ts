import type { Pair } from '../force/types';
import type { ComputeFactory, FixFactory } from '../styles';
import { PairSpinExchange } from '../force/pair/spin_exchange';
import { PairSpinDMI, PairSpinMagelec } from '../force/pair/spin_dmi';
import { ComputeSpin } from '../compute/spin';
import { FixNVESpin } from '../fix/nve_spin';
import { FixPrecessionSpin } from '../fix/precession_spin';
import { FixSetForceSpin } from '../fix/setforce_spin';

/*
 * SPIN package styles (atom_style spin): pair spin/exchange, spin/exchange/biquadratic, spin/dmi, spin/magelec,
 * compute spin, fix nve/spin, precession/spin and setforce/spin (langevin/spin: engine-gpl/registry/spin.ts). Loaded lazily (src/engine/lazyStyles.ts).
 */
export const PAIRS: Record<string, () => Pair> = {
  'spin/exchange': () => new PairSpinExchange(),
  'spin/exchange/biquadratic': () => new PairSpinExchange(true),
  'spin/dmi': () => new PairSpinDMI(),
  'spin/magelec': () => new PairSpinMagelec(),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  spin: (sys, id, group, args) => new ComputeSpin(sys, id, group, args),
};

export const FIXES: Record<string, FixFactory> = {
  'nve/spin': (s, i, g, a) => new FixNVESpin(s, i, g, a),
  'precession/spin': (s, i, g, a) => new FixPrecessionSpin(s, i, g, a),
  'setforce/spin': (s, i, g, a) => new FixSetForceSpin(s, i, g, a),
};
