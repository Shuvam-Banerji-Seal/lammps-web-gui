import { SimpleBonded, atomIndex, dihedralGeometry, applyDihedral } from '../bonded_util';
import { inumeric, parseNum } from '../util';
import { StyleError, type BondedCompute, bondedCoeffArgs } from '../types';
import { libm } from '../../libm';

/*
 * dihedral_style harmonic — docs.lammps.org/dihedral_harmonic.html:
 *   E = K [1 + d cos(n phi)]
 *   coefficients "K (energy)", "d (+1 or -1)", "n (integer >= 0)".
 */

export class DihedralHarmonic extends SimpleBonded {
  readonly name = 'harmonic';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['K', 'd', 'n'];

  protected parse(args: string[]): number[] {
    if (args.length !== 3) throw bondedCoeffArgs('dihedral', 'dihedral_coeff harmonic needs K d n');
    // As native LAMMPS does (src/MOLECULE/dihedral_harmonic.cpp coeff at b332f28): K utils::numeric, d and n
    // utils::inumeric, then the sign and multiplicity checks
    const K = parseNum(args[0], 'K'), d = inumeric(args[1]), n = inumeric(args[2]);
    if (d !== 1 && d !== -1) throw new StyleError('Incorrect sign arg for dihedral coefficients');
    if (n < 0) throw new StyleError('Incorrect multiplicity arg for dihedral coefficients');
    return [K, d, n];
  }

  compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const K = this.params.p('K'), dd = this.params.p('d'), nn = this.params.p('n');
    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < D.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, D.atoms[4 * q + w], 'dihedral'));
      const t = D.type[q];
      const phi = dihedralGeometry(bc, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
      const ed = K[t] * (1 + dd[t] * libm.cos(nn[t] * phi));
      const dEdphi = -K[t] * dd[t] * nn[t] * libm.sin(nn[t] * phi);
      e += ed;
      applyDihedral(bc, atoms, dEdphi, grad, rel, ed);
    }
    bc.acc.edihed += e;
  }
}
