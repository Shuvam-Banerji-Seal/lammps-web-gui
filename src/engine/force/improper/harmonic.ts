import { SimpleBonded, atomIndex, dihedralGeometry, applyDihedral } from '../bonded_util';
import { parseNum } from '../util';
import { StyleError, type BondedCompute } from '../types';

/*
 * improper_style harmonic — docs.lammps.org/improper_harmonic.html:
 *   E = K (chi - chi0)^2  "Note that the usual 1/2 factor is included in K."
 *   "If the 4 atoms in an improper quadruplet ... are ordered I,J,K,L then
 *   chi is the angle between the plane of I,J,K and the plane of J,K,L."
 *   coefficients "K (energy)", "chi0 (degrees)".
 * chi is taken as the unsigned angle between the planes, in [0, pi] (the
 * oracle case improper_harmonic checks this with chi0 != 0).
 */

export class ImproperHarmonic extends SimpleBonded {
  readonly name = 'harmonic';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K', 'chi0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('improper_coeff harmonic needs K chi0');
    return [parseNum(args[0], 'K'), (parseNum(args[1], 'chi0') * Math.PI) / 180];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), c0 = this.params.p('chi0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${K[t]} ${(c0[t] * 180) / Math.PI}`);
    return out;
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), c0 = this.params.p('chi0');
    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const phi = dihedralGeometry(bc, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
      // unsigned angle between the planes: chi = |phi|, dchi/dphi = sign(phi)
      const chi = Math.abs(phi);
      const sgn = phi < 0 ? -1 : 1;
      const dchi = chi - c0[t];
      const ei = K[t] * dchi * dchi;
      e += ei;
      applyDihedral(bc, atoms, 2 * K[t] * dchi * sgn, grad, rel, ei);
    }
    bc.acc.eimp += e;
  }
}
