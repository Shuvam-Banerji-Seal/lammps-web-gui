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
 * equation above with lambda = Tdamp. The random torque for T > 0 (its Gaussian stream and amplitude are not on the
 * page) is in src/engine-gpl/spin/fix_langevin_spin.ts, a subclass that overrides addTemperature and
 * supportsTemperature; this base class refuses T > 0.
 */
export class FixLangevinSpin extends Fix {
  readonly style: string = 'langevin/spin';
  /** Transverse damping lambda (Tdamp). */
  readonly damp: number;
  /** Temperature T (0: no random torque) and the seed of the random stream. */
  readonly temp: number;
  readonly seed: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!sys.state.sp) throw new StyleError('fix langevin/spin requires atom_style spin');
    if (args.length !== 3) throw new StyleError('usage: fix ID group-ID langevin/spin T Tdamp seed');
    const T = Number(args[0]), damp = Number(args[1]), seed = Number(args[2]);
    if (!Number.isFinite(T) || T < 0) throw new StyleError(`fix langevin/spin: T must be a non-negative number, got '${args[0]}'`);
    if (!Number.isFinite(damp)) throw new StyleError(`fix langevin/spin: Tdamp must be a number, got '${args[1]}'`);
    if (!Number.isInteger(seed) || seed <= 0) throw new StyleError(`fix langevin/spin: seed must be a positive integer, got '${args[2]}'`);
    if (T > 0 && !this.supportsTemperature()) throw new StyleError('fix langevin/spin with T > 0 is not implemented in the browser engine (the random torque stream is not documented)');
    this.damp = damp;
    this.temp = T;
    this.seed = seed;
  }

  /** True for a subclass that adds the random torque (T > 0). */
  protected supportsTemperature(): boolean { return false; }

  /** The random torque of one update (added to w, the precession vector of the spin, which the subclass scales too). */
  addTemperature(_w: number[]): void {
    throw new StyleError('fix langevin/spin: the random torque is not available');
  }

  /**
   * compute_single_langevin of the source: for an atom of the group, the transverse damping of its precession vector
   * w (w' = w - lambda w x s when Tdamp > 0), then the random torque (T > 0). Called for each spin update.
   */
  torque(i: number, w: number[], sp: Float64Array): void {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    if (this.damp !== 0) {
      const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
      const cx = w[1] * sz - w[2] * sy, cy = w[2] * sx - w[0] * sz, cz = w[0] * sy - w[1] * sx;
      w[0] -= this.damp * cx;
      w[1] -= this.damp * cy;
      w[2] -= this.damp * cz;
    }
    if (this.temp > 0) this.addTemperature(w);
  }
}
