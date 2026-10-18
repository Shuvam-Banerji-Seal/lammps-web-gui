/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MISC/bond_special.cpp and
 * bond_special.h (the LAMMPS developers; Copyright (2003) Sandia Corporation, GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * bond_style special (docs.lammps.org/bond_special.html): a bond whose only term is the pair interaction of its two
 * atoms, scaled by the factors of the bond type: bond_coeff N factor_lj factor_coul. It needs special_bonds that leave
 * the 1-2 pair in full (lj 0 and coul 0 for 1-2), so the pair style is the only term between bonded atoms.
 */
import { SimpleBonded, atomIndex, bondDelta } from '../../engine/force/bonded_util';
import { Pair, StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { evTallyBond } from '../xmolecule/ev';

/** The special_bonds weights as the force field passes them: lj/coul are [1-2, 1-3, 1-4]. */
interface SpecialLink {
  lj: readonly number[];
  coul: readonly number[];
  /** special_bonds angle yes (native special_angle = 1). */
  angle?: boolean;
  /** special_bonds dihedral yes (native special_dihedral = 1). */
  dihedral?: boolean;
}

export class BondSpecial extends SimpleBonded {
  readonly name = 'special';
  readonly kind = 'bond' as const;
  readonly paramNames = ['factor_lj', 'factor_coul'];
  private pair: Pair | null = null;
  private special: SpecialLink = { lj: [0, 0, 0], coul: [0, 0, 0] };

  linkForceField(link: { pair: Pair | null; special: SpecialLink }): void {
    this.pair = link.pair;
    this.special = link.special;
  }

  /** bond_special.cpp coeff(): narg must be 3 ("Incorrect args for bond coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for bond coefficients');
    super.coeff(args);
  }

  /** bond_special.cpp init_style(): pair style checks and the 1-2, 1-3, 1-4 special settings. */
  init(ctx?: StyleContext): void {
    const pair = this.pair;
    if (!pair) throw new StyleError('No pair style defined');
    if (typeof pair.single !== 'function' || pair.manybody) throw new StyleError('Pair style does not support bond style special');
    const { lj, coul } = this.special;
    if (lj[0] !== 0.0 || coul[0] !== 0) throw new StyleError('Invalid 1-2 setting for bond style special.');
    if (!this.special.angle && (lj[1] !== 1.0 || coul[1] !== 1.0)) {
      throw new StyleError('Invalid 1-3 setting for bond style special.');
    }
    if (!this.special.dihedral && (lj[2] !== 1.0 || coul[2] !== 1.0)) {
      throw new StyleError('Invalid 1-4 setting for bond style special.');
    }
    super.init(ctx);
  }

  /** bond_special.cpp equilibrium_distance(): 0. */
  equilibrium(_type: number): number {
    return 0;
  }

  compute(bc: BondedCompute): void {
    const s = bc.s;
    const bonds = s.topo.bonds;
    const pair = this.pair!;
    const factorLJ = this.params.p('factor_lj'), factorCoul = this.params.p('factor_coul');
    const f = bc.f;
    const d = [0, 0, 0];
    for (let n = 0; n < bonds.n; n++) {
      const type = bonds.type[n];
      const i1 = atomIndex(bc, bonds.atoms[2 * n], 'bond');
      const i2 = atomIndex(bc, bonds.atoms[2 * n + 1], 'bond');
      const itype = s.type[i1], jtype = s.type[i2];
      bondDelta(bc, i1, i2, d);
      const delx = d[0], dely = d[1], delz = d[2];
      const rsq = delx * delx + dely * dely + delz * delz;
      // native call: pair->single(i1, i2, itype, jtype, rsq, factor_lj[type], factor_coul[type], fbond). Its 6th
      // argument is the Coulomb factor, so the LJ factor of the bond type goes there (bond_special.cpp passes them in
      // this order; the two agree when factor_lj equals factor_coul, as in the docs' examples).
      const single = pair.single!(i1, i2, itype, jtype, rsq, factorLJ[type], factorCoul[type], s.q[i1], s.q[i2]);
      const ebond = single.eng;
      const fbond = single.fforce;
      f[3 * i1] += delx * fbond; f[3 * i1 + 1] += dely * fbond; f[3 * i1 + 2] += delz * fbond;
      f[3 * i2] -= delx * fbond; f[3 * i2 + 1] -= dely * fbond; f[3 * i2 + 2] -= delz * fbond;
      evTallyBond(bc, i1, i2, ebond, fbond, d);
    }
  }
}
