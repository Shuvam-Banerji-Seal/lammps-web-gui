import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { SPIN_HBAR } from '../force/pair/spin_exchange';
import { numeric } from '../force/util';
import { spinContribute } from '../force/spin_book';

/*
 * fix ID group-ID precession/spin style args — docs.lammps.org/fix_precession_spin.html: "This fix applies a
 * precession torque to each magnetic spin in the group." Styles zeeman (H x y z), anisotropy (K x y z) and
 * cubic (K1 K2c n1x n1y n1z n2x n2y n2z n3x n3y n3z), combinable in one command. "In all cases, the choice of
 * (x y z) only imposes the vector directions for the forces. Only the direction of the vector is important;
 * its length is ignored (the entered vectors are normalized)." The fix_modify energy option adds its energy to
 * the potential energy.
 *
 * Energies, measured with native LAMMPS (black box, one spin of magnitude 2.2 or 1.0, field 10 T along z):
 *   zeeman:      E = -2 mu_i mu_B H (s . n), mu_B = 5.78901e-5 eV/T as the measured value of the field term
 *   anisotropy:  E = -K (s . n)^2
 *   cubic:       E = -K1 [a^2 b^2 + b^2 c^2 + a^2 c^2] - K2c a^2 b^2 c^2, with a = s.n1, b = s.n2, c = s.n3
 * The page writes "+K2c" for the last term of the cubic Hamiltonian; the measured energy has the sign
 * given here (checked with s = (0.6, 0.48, 0.64): -0.00034175877 for K1 = 0.001, K2c = 0.0005).
 *
 * Precession vectors omega = -(1/hbar) dE/ds with hbar = SPIN_HBAR (measured: zeeman 2 mu_i mu_B H / hbar along n,
 * anisotropy 2 K (s.n) n / hbar, cubic as the derivative of the energy above, each matching native fmx fmy fmz).
 * Stt (spin-transfer torque) is not implemented and is refused.
 */

const MU_B_METAL = 5.78901e-5;

const unit = (v: number[], what: string): number[] => {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (!(n > 0)) throw new StyleError(`fix precession/spin: the ${what} direction vector must be non-zero`);
  return [v[0] / n, v[1] / n, v[2] / n];
};

export class FixPrecessionSpin extends Fix {
  readonly style: string = 'precession/spin';
  energyGlobal = true;
  // fix_precession_spin.cpp at b332f28: scalar_flag = 1, extscalar = 1; compute_scalar is the energy fix_modify energy
  // adds (eprec)
  scalarFlag = true;
  extscalar = 1;
  private zee: { h: number; n: number[] } | null = null;
  private aniso: { k: number; n: number[] } | null = null;
  private cubic: { k1: number; k2: number; n1: number[]; n2: number[]; n3: number[] } | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!sys.state.sp) throw new StyleError('fix precession/spin requires atom_style spin');
    // As native LAMMPS does (src/utils.cpp utils::numeric at b332f28): a value past the end of the line is NULL
    const num = (w: string | undefined, _what: string): number => numeric(w);
    // As native LAMMPS does (src/SPIN/fix_precession_spin.cpp FixPrecessionSpin::FixPrecessionSpin at b332f28): a style
    // whose first arguments are not all there is "Illegal fix precession/spin command"; the last values are read without
    // this check (past the end they are NULL, see num)
    const need = (n: number): void => {
      if (k + n > args.length) throw new StyleError('Illegal fix precession/spin command');
    };
    let k = 0;
    while (k < args.length) {
      const style = args[k];
      if (style === 'zeeman') {
        need(4);
        this.zee = { h: num(args[k + 1], 'zeeman H'), n: unit([num(args[k + 2], 'zeeman x'), num(args[k + 3], 'zeeman y'), num(args[k + 4], 'zeeman z')], 'zeeman') };
        k += 5;
      } else if (style === 'anisotropy') {
        need(4);
        this.aniso = { k: num(args[k + 1], 'anisotropy K'), n: unit([num(args[k + 2], 'anisotropy x'), num(args[k + 3], 'anisotropy y'), num(args[k + 4], 'anisotropy z')], 'anisotropy') };
        k += 5;
      } else if (style === 'cubic') {
        need(2);
        const v = [];
        for (let m = 1; m <= 11; m++) v.push(num(args[k + m], `cubic argument ${m}`));
        this.cubic = {
          k1: v[0], k2: v[1],
          n1: unit(v.slice(2, 5), 'cubic n1'), n2: unit(v.slice(5, 8), 'cubic n2'), n3: unit(v.slice(8, 11), 'cubic n3'),
        };
        k += 12;
      } else if (style === 'stt') {
        throw new StyleError('fix precession/spin stt is not implemented in the browser engine');
      } else throw new StyleError('Illegal precession/spin command'); // As native LAMMPS does (src/SPIN/fix_precession_spin.cpp FixPrecessionSpin::FixPrecessionSpin at b332f28)
    }
    if (!this.zee && !this.aniso && !this.cubic) throw new StyleError('fix precession/spin needs a style: zeeman, anisotropy or cubic');
  }

  /**
   * Energy of the current spins, evaluated when thermo asks for it. Measured with native LAMMPS (black box):
   * a thermo row of zeeman + cubic precession with nve/spin lattice frozen matches only when this fix energy
   * is taken from the spins after the step (fix energies are computed on demand at output time, while the pair
   * energies are those of the force evaluation).
   */
  energy(): number {
    const s = this.sys.state;
    const sp = s.sp!;
    let e = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) e += this.term(i, sp, null);
    return e;
  }

  /** Global scalar (thermo f_ID): the precession energy, as for fix_modify energy. */
  computeScalar(): number {
    return this.energy();
  }

  /** Adds this fix's precession vector of atom i (spin sp[4i..4i+3]) to out; returns the energy of atom i. */
  private term(i: number, sp: Float64Array, out: number[] | Float64Array | null): number {
    const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
    let e = 0;
    const add = (c: number, n: number[]) => {
      if (!out) return;
      out[0] += c * n[0]; out[1] += c * n[1]; out[2] += c * n[2];
    };
    if (this.zee) {
      const n = this.zee.n;
      const mu = sp[4 * i + 3];
      const sn = sx * n[0] + sy * n[1] + sz * n[2];
      e += -2 * mu * MU_B_METAL * this.zee.h * sn;
      add(2 * mu * MU_B_METAL * this.zee.h / SPIN_HBAR, n);
    }
    if (this.aniso) {
      const n = this.aniso.n;
      const sn = sx * n[0] + sy * n[1] + sz * n[2];
      e += -this.aniso.k * sn * sn;
      add(2 * this.aniso.k * sn / SPIN_HBAR, n);
    }
    if (this.cubic) {
      const { k1, k2, n1, n2, n3 } = this.cubic;
      const a = sx * n1[0] + sy * n1[1] + sz * n1[2];
      const b = sx * n2[0] + sy * n2[1] + sz * n2[2];
      const c = sx * n3[0] + sy * n3[1] + sz * n3[2];
      const a2 = a * a, b2 = b * b, c2 = c * c;
      e += -k1 * (a2 * b2 + b2 * c2 + a2 * c2) - k2 * a2 * b2 * c2;
      // omega = (1/hbar) sum_k (K1 df/da_k + K2 dg/da_k) n_k, with f = a2 b2 + b2 c2 + a2 c2 and g = a2 b2 c2
      const da = k1 * (2 * a * b2 + 2 * a * c2) + k2 * 2 * a * b2 * c2;
      const db = k1 * (2 * b * a2 + 2 * b * c2) + k2 * 2 * b * a2 * c2;
      const dc = k1 * (2 * c * b2 + 2 * c * a2) + k2 * 2 * c * a2 * b2;
      add(da / SPIN_HBAR, n1);
      add(db / SPIN_HBAR, n2);
      add(dc / SPIN_HBAR, n3);
    }
    return e;
  }

  /**
   * Precession vector of atom i from this fix (zero outside the group). As native LAMMPS does
   * (src/SPIN/fix_precession_spin.cpp compute_single_precession at b332f28): only an atom of the group gets the terms.
   */
  omegaOf(i: number, sp: Float64Array, out: Float64Array | number[]): void {
    out[0] = 0; out[1] = 0; out[2] = 0;
    if (this.sys.state.mask[i] & this.groupBit) this.term(i, sp, out);
  }

  /** The minimizers apply the fix after each force evaluation too (native min_post_force). */
  minPostForce(): void { this.postForce(); }

  /** Sets this fix's contribution to the precession vectors of the group (after each force evaluation). */
  postForce(): void {
    const s = this.sys.state;
    const sp = s.sp!;
    const bit = this.groupBit;
    const mask = s.mask;
    const vals = new Float64Array(3 * s.n);
    const w = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      w[0] = 0; w[1] = 0; w[2] = 0;
      this.term(i, sp, w);
      vals[3 * i] = w[0]; vals[3 * i + 1] = w[1]; vals[3 * i + 2] = w[2];
    }
    spinContribute(s, this, vals, 0);
  }
}
