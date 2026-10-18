/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-MOLECULE/bond_quartic_exp.cpp and bond_quartic_exp.h at commit b332f28 (the LAMMPS developers;
 * Copyright (2003) Sandia Corporation, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * bond_style quartic/exp (docs.lammps.org/bond_quartic_exp.html): a quartic polynomial in (r - r0) plus an
 * exponential, E = k2 dr^2 + k3 dr^3 + k4 dr^4 + A exp(-r/B) with dr = r - r0; the exponential is left out when
 * A or B is 0. Coefficients: "r0 k2 k3 k4 A B" (bond_coeff N r0 k2 k3 k4 A B).
 * The expressions and their order are native's (bond_quartic_exp.cpp compute()).
 */
import { SimpleBonded, atomIndex, bondDelta } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { exp } from '../libm/exp';
import { evTallyBond } from './ev';

export class BondQuarticExp extends SimpleBonded {
  readonly name = 'quartic/exp';
  readonly kind = 'bond' as const;
  readonly paramNames = ['r0', 'k2', 'k3', 'k4', 'A', 'B'];

  /** bond_quartic_exp.cpp coeff(): narg must be 7 ("Incorrect args for bond coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 7) throw new StyleError('Incorrect args for bond coefficients');
    this.params.setRange(args[0], args.slice(1).map((w, k) => parseNum(w, this.paramNames[k])));
  }

  compute(bc: BondedCompute): void {
    const bonds = bc.s.topo.bonds;
    const r0 = this.params.p('r0'), k2 = this.params.p('k2'), k3 = this.params.p('k3'), k4 = this.params.p('k4');
    const A = this.params.p('A'), B = this.params.p('B');
    const f = bc.f;
    const d = [0, 0, 0];
    for (let n = 0; n < bonds.n; n++) {
      const i1 = atomIndex(bc, bonds.atoms[2 * n], 'bond');
      const i2 = atomIndex(bc, bonds.atoms[2 * n + 1], 'bond');
      const type = bonds.type[n];
      bondDelta(bc, i1, i2, d);
      const delx = d[0], dely = d[1], delz = d[2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const r = Math.sqrt(rsq);
      const dr = r - r0[type];
      const dr2 = dr * dr;
      const dr3 = dr2 * dr;
      const dr4 = dr3 * dr;
      const de_bond = 2.0 * k2[type] * dr + 3.0 * k3[type] * dr2 + 4.0 * k4[type] * dr3;
      let fbond = r > 0.0 ? -de_bond / r : 0.0;
      let ebond = k2[type] * dr2 + k3[type] * dr3 + k4[type] * dr4;
      if (A[type] !== 0.0 && B[type] !== 0.0) {
        const a = A[type];
        const b = B[type];
        const ebond_exp = a * exp(-r / b);
        ebond += ebond_exp;
        if (r > 0.0) fbond += ebond_exp / b / r;
      }
      f[3 * i1] += delx * fbond; f[3 * i1 + 1] += dely * fbond; f[3 * i1 + 2] += delz * fbond;
      f[3 * i2] -= delx * fbond; f[3 * i2 + 1] -= dely * fbond; f[3 * i2 + 2] -= delz * fbond;
      evTallyBond(bc, i1, i2, ebond, fbond, d);
    }
  }

  /** bond_quartic_exp.cpp equilibrium_distance(): r0. */
  equilibrium(type: number): number { return this.params.p('r0')[type]; }
}
