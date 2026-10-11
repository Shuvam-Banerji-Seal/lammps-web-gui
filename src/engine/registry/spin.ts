import type { Pair } from '../force/types';
import { StyleError } from '../force/types';
import type { ComputeFactory, FixFactory } from '../styles';
import { PairSpinExchange } from '../force/pair/spin_exchange';
import { PairSpinDMI, PairSpinMagelec } from '../force/pair/spin_dmi';
import { ComputeSpin } from '../compute/spin';
import { FixNVESpin } from '../fix/nve_spin';
import { FixPrecessionSpin } from '../fix/precession_spin';
import { FixLangevinSpin } from '../fix/langevin_spin';
import { FixSetForceSpin } from '../fix/setforce_spin';

/*
 * SPIN package styles (atom_style spin): pair spin/exchange, spin/exchange/biquadratic, spin/dmi, spin/magelec,
 * compute spin, fix nve/spin, precession/spin, langevin/spin and setforce/spin. Loaded lazily (src/engine/lazyStyles.ts).
 */
export const PAIRS: Record<string, () => Pair> = {
  'spin/exchange': () => new PairSpinExchange(),
  'spin/exchange/biquadratic': () => new PairSpinExchange(true),
  'spin/dmi': () => new PairSpinDMI(),
  'spin/magelec': () => new PairSpinMagelec(),
  // docs.lammps.org/pair_spin_neel.html gives the Neel coupling in terms of g(r) and q(r), which are described only
  // as "defined and fitted according to the same Bethe-Slater function"; without their form the style is refused.
  'spin/neel': () => {
    throw new StyleError('pair_style spin/neel is not implemented in the browser engine: g(r) and q(r) are not given on pair_spin_neel.html');
  },
};

export const COMPUTES: Record<string, ComputeFactory> = {
  spin: (sys, id, group, args) => new ComputeSpin(sys, id, group, args),
};

export const FIXES: Record<string, FixFactory> = {
  'nve/spin': (s, i, g, a) => new FixNVESpin(s, i, g, a),
  'precession/spin': (s, i, g, a) => new FixPrecessionSpin(s, i, g, a),
  'langevin/spin': (s, i, g, a) => new FixLangevinSpin(s, i, g, a),
  'setforce/spin': (s, i, g, a) => new FixSetForceSpin(s, i, g, a),
};
