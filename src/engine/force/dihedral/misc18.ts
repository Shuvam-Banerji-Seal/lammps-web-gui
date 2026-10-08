import { SimpleBonded, atomIndex, dihedralGeometry, applyDihedral } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';
import { StyleError, type BondedCompute } from '../types';

/*
 * dihedral_style cosine/squared/restricted (wave 18).
 * docs.lammps.org/dihedral_cosine_squared_restricted.html: "The *cosine/squared/restricted* dihedral style
 * uses the potential", "E = K [\cos(\phi) - \cos(\phi_0)]^2 / \sin^2(\phi)", and ":math:`\phi_0` is
 * specified in degrees, but LAMMPS converts it to radians internally." That is the potential
 * E = K [cos(phi) - cos(phi_0)]^2 / sin^2(phi), with K in energy units and
 * phi_0 in degrees (the page says phi_0 is converted to radians internally).
 *
 * Measured with native LAMMPS (black box), one quadruplet with atoms at
 * (0.1,1.2,0.3), (0,0,0), (1,0,0.1), (1.4,0.9,0.8), dihedral_coeff 1 4.0 120,
 * bond and angle terms present: native edihed is 54.8771150012586, which equals
 * K [cos(phi) - cos(phi_0)]^2 / sin^2(phi) with phi_0 in degrees, so the
 * energy has no extra factor. The per-atom forces agree with the gradient of
 * that energy. An earlier version divided the energy by 4 and was wrong by
 * exactly that factor; the forces had always been correct.
 *
 * A 50-step nve run of the oracle case w18dihimp_cossqres agrees with native
 * in energy, forces and virial at every thermo row.
 *
 * Near planar quadruplets (sin(phi) = 0) the potential is singular. An earlier
 * probe along a cis-approaching quadruplet (not re-measured in this wave)
 * found native reports nan below |sin(phi)| of about 1e-8, so this engine
 * refuses below 1e-7 rather than produce nan forces.
 *
 * dE/dphi = -2 K u / sin(phi) - 2 K u^2 cos(phi) / sin^3(phi) with
 * u = cos(phi) - cos(phi_0). phi and its gradient come from bonded_util
 * dihedralGeometry; the potential is even in phi, so the sign branch of atan2
 * does not matter.
 */

const DEG2RAD = Math.PI / 180;
/** Below this |sin(phi)| the potential is singular and native is nan (carried over, see header). */
const SIN_MIN = 1e-7;

export class DihedralCosineSquaredRestricted extends SimpleBonded {
  readonly name = 'cosine/squared/restricted';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['K', 'phi0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('dihedral_coeff cosine/squared/restricted needs K phi0');
    return [parseNum(args[0], 'K'), parseNum(args[1], 'phi0') * DEG2RAD];
  }

  // write_data, measured with native LAMMPS: phi0 written back in degrees, e.g. the line 1 10 120
  dataCoeffs(): string[] {
    const K = this.params.p('K'), phi0 = this.params.p('phi0');
    return Array.from({ length: this.ntypes }, (_, k) => `${k + 1} ${fmtCoeff(K[k + 1])} ${fmtCoeff(phi0[k + 1] / DEG2RAD)}`);
  }

  compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const K = this.params.p('K'), phi0 = this.params.p('phi0');
    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < D.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, D.atoms[4 * q + w], 'dihedral'));
      const t = D.type[q];
      const phi = dihedralGeometry(bc, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
      const s = Math.sin(phi);
      const s2 = s * s;
      if (s2 < SIN_MIN * SIN_MIN) {
        throw new StyleError(`dihedral_style cosine/squared/restricted: |sin(phi)| = ${Math.abs(s).toExponential(3)} is below ${SIN_MIN}, the potential is singular at phi = 0 or 180 degrees`);
      }
      const u = Math.cos(phi) - Math.cos(phi0[t]);
      const c = Math.cos(phi);
      const ed = (K[t] * u * u) / s2;
      const dEdphi = (-2 * K[t] * u) / s - (2 * K[t] * u * u * c) / (s2 * s);
      e += ed;
      applyDihedral(bc, atoms, dEdphi, grad, rel, ed);
    }
    bc.acc.edihed += e;
  }
}
