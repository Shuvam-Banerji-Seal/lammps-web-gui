import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * fix ID group-ID langevin/spin T Tdamp seed — docs.lammps.org/fix_langevin_spin.html: "T = desired temperature of
 * the bath (temperature units, K in metal units)", "Tdamp = transverse magnetic damping parameter (adim)" and
 * "seed = random number seed to use for white noise (positive integer)". The page gives the stochastic
 * equation "\frac{d \vec{s}_{i}}{dt} = \frac{1}{\left(1+\lambda^2 \right)} \left( \left( \vec{\omega}_{i} +\vec{\eta} \right) \times \vec{s}_{i} + \lambda\, \vec{s}_{i} \times\left( \vec{\omega}_{i} \times\vec{s}_{i} \right) \right)"
 * with "\lambda the transverse damping". The damping part (eta = 0) is applied in fix nve/spin
 * (fix/nve_spin.ts), where the precession vector of each spin is known.
 *
 * Measured with native LAMMPS (black box, one spin, temperature 0): the damping is the transverse term of the
 * equation above with lambda = Tdamp. The random torque eta for T > 0 uses a per-atom Gaussian stream whose
 * amplitude and draw order are not given on the page, so T > 0 is refused.
 */
export class FixLangevinSpin extends Fix {
  readonly style: string = 'langevin/spin';
  /** Transverse damping lambda (Tdamp). */
  readonly damp: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!sys.state.sp) throw new StyleError('fix langevin/spin requires atom_style spin');
    if (args.length !== 3) throw new StyleError('usage: fix ID group-ID langevin/spin T Tdamp seed');
    const T = Number(args[0]), damp = Number(args[1]), seed = Number(args[2]);
    if (!Number.isFinite(T) || T < 0) throw new StyleError(`fix langevin/spin: T must be a non-negative number, got '${args[0]}'`);
    if (!Number.isFinite(damp)) throw new StyleError(`fix langevin/spin: Tdamp must be a number, got '${args[1]}'`);
    if (!Number.isInteger(seed) || seed <= 0) throw new StyleError(`fix langevin/spin: seed must be a positive integer, got '${args[2]}'`);
    if (T > 0) throw new StyleError('fix langevin/spin with T > 0 is not implemented in the browser engine (the random torque stream is not documented)');
    this.damp = damp;
  }
}
