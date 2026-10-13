/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/YAFF/bond_mm3.cpp and src/YAFF/bond_mm3.h at commit b332f28 (YAFF package; contributing author
 * Steven Vandenbrande). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * bond_style mm3 — docs.lammps.org/bond_mm3.html:
 *   "E = K (r - r_0)^2 [ 1 - 2.55(r-r_0) + \frac{7}{12} 2.55^2(r-r_0)^2 ]"
 *   "The anharmonic prefactors have units \AA^{-n}: -2.55 \AA^{-1} and \frac{7}{12} 2.55^2 \AA^{-2}. The code
 *   takes care of the necessary unit conversion for these factors internally."
 *   Coefficients: "K (energy/distance^2)", "r_0 (distance)".
 * The prefactors are divided by force->angstrom (native), taken here from the units style (the table of
 * ANGSTROM_PER_UNIT, as pair zbl does). Forces follow native's analytic expression (the same as the gradient).
 */
import { SimpleBonded, atomIndex, bondDelta } from '../../engine/force/bonded_util';
import { type BondedCompute, type StyleContext } from '../../engine/force/types';
import { ANGSTROM_PER_UNIT } from '../../engine/lepton/units';

export class BondMM3 extends SimpleBonded {
  readonly name = 'mm3';
  readonly kind = 'bond' as const;
  readonly paramNames = ['K', 'r0'];
  /** -2.55 / angstrom and 7/12 2.55^2 / angstrom^2 (native K3 and K4), set at init. */
  private K3 = -2.55;
  private K4 = (7 / 12) * 2.55 * 2.55;

  init(ctx?: StyleContext): void {
    super.init(ctx as StyleContext);
    const angstrom = ANGSTROM_PER_UNIT[ctx?.s?.units.style ?? 'lj'] ?? 1;
    this.K3 = -2.55 / angstrom;
    this.K4 = ((7 / 12) * 2.55 * 2.55) / (angstrom * angstrom);
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }

  compute(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const K = this.params.p('K'), r0 = this.params.p('r0');
    const { K3, K4 } = this;
    const d = [0, 0, 0];
    const f = bc.f;
    const v = bc.virial;
    let e = 0;
    for (let k = 0; k < b.n; k++) {
      const i = atomIndex(bc, b.atoms[2 * k], 'bond');
      const j = atomIndex(bc, b.atoms[2 * k + 1], 'bond');
      const t = b.type[k];
      bondDelta(bc, i, j, d);   // r_i - r_j
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const dr = r - r0[t];
      const dr2 = dr * dr;
      // native bond_mm3.cpp: de_bond = 2 K dr (1 + 1.5 K3 dr + 2 K4 dr^2), energy K dr^2 (1 + K3 dr + K4 dr^2)
      const deBond = 2 * K[t] * dr * (1 + 1.5 * K3 * dr + 2 * K4 * dr2);
      const fbond = r > 0 ? -deBond / r : 0;
      const eb = K[t] * dr2 * (1 + K3 * dr + K4 * dr2);
      e += eb;
      f[3 * i] += d[0] * fbond; f[3 * i + 1] += d[1] * fbond; f[3 * i + 2] += d[2] * fbond;
      f[3 * j] -= d[0] * fbond; f[3 * j + 1] -= d[1] * fbond; f[3 * j + 2] -= d[2] * fbond;
      const w0 = d[0] * d[0] * fbond, w1 = d[1] * d[1] * fbond, w2 = d[2] * d[2] * fbond;
      const w3 = d[0] * d[1] * fbond, w4 = d[0] * d[2] * fbond, w5 = d[1] * d[2] * fbond;
      v[0] += w0; v[1] += w1; v[2] += w2; v[3] += w3; v[4] += w4; v[5] += w5;
      if (bc.eatom) { bc.eatom[i] += 0.5 * eb; bc.eatom[j] += 0.5 * eb; }
      if (bc.vatom) {
        const w = [w0, w1, w2, w3, w4, w5];
        for (let c = 0; c < 6; c++) { bc.vatom[6 * i + c] += 0.5 * w[c]; bc.vatom[6 * j + c] += 0.5 * w[c]; }
      }
    }
    bc.acc.ebond += e;
  }
}
