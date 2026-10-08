import { SimpleBonded, atomIndex } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';
import { angleTerm } from './harmonic';
import { StyleError, type BondedCompute } from '../types';

/*
 * Angle styles added by wave 18: cosine/squared/restricted and mm3.
 * Each class cites its docs.lammps.org page and quotes the formula /
 * coefficient sentences it implements. theta is the angle at the middle atom
 * j of the triplet i-j-k, computed by angleTerm.
 */

const DEG = 180 / Math.PI;

/*
 * angle_style cosine/squared/restricted — docs.lammps.org/angle_cosine_squared_restricted.html:
 *   "E = K [\cos(\theta) - \cos(\theta_0)]^2 / \sin^2(\theta)"
 *   "which is commonly used in the MARTINI force field," "Note that the usual
 *   1/2 factor is included in :math:`K`." Coefficients: K (energy) and theta_0
 *   (degrees); ":math:`\theta_0` is specified in degrees, but LAMMPS converts it
 *   to radians internally."
 * Implemented with dE/dcos(theta) = K [2 u (1 - c^2) + 2 c u^2] / (1 - c^2)^2,
 * u = c - cos(theta0).
 *
 * Measured with native LAMMPS (black box), one triplet, angle_coeff 1 30.0
 * 109.5, atoms (5,5,5), (6,5,5), (6.5,5.9,5.1) (theta = 118.90561944 deg):
 * eangle 0.87564916563854811, forces (0, 11.1491332244501, 1.2387925804944),
 * (9.4935506168734, -16.359008562978, -1.8176676181087) and the negative of
 * those on atom 3. The energy and its gradient match those values, so this
 * style is conservative like the dihedral version.
 *
 * At theta exactly 0 or 180 degrees sin(theta) = 0 and native reports nan;
 * this style divides by 1 - c^2 directly and produces the same non-finite
 * values rather than inventing a cutoff.
 */
export class AngleCosineSquaredRestricted extends SimpleBonded {
  readonly name = 'cosine/squared/restricted';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'theta0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('angle_coeff cosine/squared/restricted needs K theta0');
    return [parseNum(args[0], 'K'), (parseNum(args[1], 'theta0') * Math.PI) / 180];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((th[t] * 180) / Math.PI)}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), th0 = this.params.p('theta0');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (_theta, c) => {
        const s2 = 1 - c * c;
        const u = c - Math.cos(th0[t]);
        const ea = (K[t] * u * u) / s2;
        const dEdc = (K[t] * (2 * u * s2 + 2 * c * u * u)) / (s2 * s2);
        e += ea;
        return { e: ea, dEdc };
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style mm3 — docs.lammps.org/angle_mm3.html:
 *   "E = K (\theta - \theta_0)^2 \left[ 1 - 0.014(\theta - \theta_0) +
 *   5.6(10)^{-5} (\theta - \theta_0)^2 - 7.0(10)^{-7} (\theta - \theta_0)^3 +
 *   9(10)^{-10} (\theta - \theta_0)^4 \right]"
 *   "The anharmonic prefactors have units :math:`\deg^{-n}`" and ":math:`\theta_0`
 *   is specified in degrees, but LAMMPS converts it to radians internally; hence
 *   :math:`K` is effectively energy per radian\^2."
 * So the leading (\theta-\theta_0)^2 is in radians while the anharmonic
 * bracket is a polynomial in (\theta-\theta_0) in degrees.
 *
 * Measured with native LAMMPS (black box) at 121 geometries (K = 1, theta0 = 90 deg, theta from 30 to 150 deg):
 * the reported eangle is, to 1e-12,
 *   E = K dR^2 [1 - 0.802141 dR + 0.183837 dR^2 - 0.131664 dR^3 + 0.23709 dR^4]
 * with dR = theta - theta0 in radians. These are the page's degree factors converted to radians and rounded
 * to 6 digits (0.014 * 180/pi = 0.80214091..., 5.6e-5 (180/pi)^2 = 0.18383716..., 7.0e-7 (180/pi)^3 =
 * 0.13166..., and 2.2e-8 (180/pi)^4 = 0.23709, not the page's 9e-10, which makes eangle 27 percent too small
 * at 50 deg); the unrounded degree factors miss native by 3e-8.
 * The native force is not the gradient of that energy: measured the same way, -dE/dtheta is, to 1e-12,
 *   K dR [2 - 2.406422 dR + 0.735348 dR^2 - 0.6478318 dR^3 + 1.42254 dR^4]
 * whose cubic factor differs from the exact derivative factor 5 x 0.131664 = 0.65832 (up to 0.8 percent of the
 * force at 50 deg). The style reports the energy and the force as native does.
 *
 * units: native lj divides the printed eangle by the atom count through
 * thermo normalization; the potential itself is unit independent.
 */
const E1 = -0.802141, E2 = 0.183837, E3 = -0.131664, E4 = 0.23709;
const F1 = -2.406422, F2 = 0.735348, F3 = -0.6478318, F4 = 1.42254;

export class AngleMm3 extends SimpleBonded {
  readonly name = 'mm3';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'theta0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('angle_coeff mm3 needs K theta0');
    return [parseNum(args[0], 'K'), (parseNum(args[1], 'theta0') * Math.PI) / 180];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((th[t] * 180) / Math.PI)}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), th0 = this.params.p('theta0');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const dR = theta - th0[t];
        const ea = K[t] * dR * dR * (1 + dR * (E1 + dR * (E2 + dR * (E3 + dR * E4))));
        const dEdtheta = K[t] * dR * (2 + dR * (F1 + dR * (F2 + dR * (F3 + dR * F4))));
        e += ea;
        return { e: ea, dEdtheta };
      });
    }
    bc.acc.eangle += e;
  }
}
