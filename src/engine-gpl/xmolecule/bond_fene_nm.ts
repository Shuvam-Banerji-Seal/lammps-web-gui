/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-MOLECULE/bond_fene_nm.cpp and bond_fene_nm.h at commit b332f28 (the LAMMPS developers; the
 * generalized Lennard-Jones tail of the fene bond is the nm style of pair_style nm/cut). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * bond_style fene/nm (docs.lammps.org/bond_fene.html, section "fene/nm"): the FENE attraction with the generalized
 * LJ term of n and m in place of the 12-6 LJ term:
 *   E = -0.5 K R0^2 ln[1 - (r/R0)^2] + E0/(n-m) [ m (sigma/r)^n - n (sigma/r)^m ]        for r < sigma,
 *   with E0 = epsilon, cut off at r = sigma (the doc: "E0/(n-m) [ m (r0/r)^n - n (r0/r)^m ]", r0 = sigma).
 *   Coefficients: "K R0 epsilon sigma n m" (bond_coeff N K R0 epsilon sigma n m).
 * The loop and the expressions are native's (fene/nm compute): the force is computed as a single fbond, not as
 * -dE/dr / r, so the summation matches native bit for bit. The "fene/nm bond too long" and "Bad FENE bond" limits
 * are native's (rlogarg < 0.02, rlogarg <= -0.21).
 */
import { SimpleBonded, atomIndex, bondDelta } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { pow } from '../libm/pow';
import { log } from '../libm/log';
import { evTallyBond } from './ev';

export class BondFeneNm extends SimpleBonded {
  readonly name = 'fene/nm';
  readonly kind = 'bond' as const;
  readonly paramNames = ['K', 'R0', 'epsilon', 'sigma', 'nn', 'mm'];
  private special: readonly number[] | null = null;

  /** bond_fene_nm.cpp coeff(): narg must be 7 ("Incorrect args for bond coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 7) throw new StyleError('Incorrect args for bond coefficients');
    this.params.setRange(args[0], args.slice(1).map((w, k) => parseNum(w, this.paramNames[k])));
  }

  /** bond_fene_nm.cpp init_style(): only the special_bonds warning. */
  linkForceField(link: { pair: unknown; special: { lj: readonly number[]; coul: readonly number[] } }): void {
    this.special = link.special.lj;
  }

  init(ctx: StyleContext): void {
    super.init(ctx);
    const sp = this.special;
    // native: force->special_lj[1] != 0.0 || special_lj[2] != 1.0 || special_lj[3] != 1.0 (the engine lists 1-2, 1-3, 1-4)
    if (sp && (sp[0] !== 0.0 || sp[1] !== 1.0 || sp[2] !== 1.0)) {
      ctx.log('WARNING: Use special bonds = 0,1,1 with bond style fene/nm');
    }
  }

  compute(bc: BondedCompute): void {
    const bonds = bc.s.topo.bonds;
    const K = this.params.p('K'), R0 = this.params.p('R0');
    const eps = this.params.p('epsilon'), sig = this.params.p('sigma');
    const nn = this.params.p('nn'), mm = this.params.p('mm');
    const f = bc.f;
    const d = [0, 0, 0];
    for (let n = 0; n < bonds.n; n++) {
      const i1 = atomIndex(bc, bonds.atoms[2 * n], 'bond');
      const i2 = atomIndex(bc, bonds.atoms[2 * n + 1], 'bond');
      const type = bonds.type[n];
      bondDelta(bc, i1, i2, d); // delx = x[i1] - x[i2] (the copy of i2 nearest to i1)
      const delx = d[0], dely = d[1], delz = d[2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const r0sq = R0[type] * R0[type];
      let rlogarg = 1.0 - rsq / r0sq;
      if (rlogarg < 0.02) {
        bc.warn?.(`fene/nm bond too long: ${bc.s.step} ${bonds.atoms[2 * n]} ${bonds.atoms[2 * n + 1]} ${Math.sqrt(rsq)}`);
        if (rlogarg <= -0.21) throw new StyleError('Bad FENE bond');
        rlogarg = 0.02;
      }
      let fbond = -K[type] / rlogarg;
      const sigsq = sig[type] * sig[type];
      let r = 0;
      if (rsq < sigsq) {
        r = Math.sqrt(rsq);
        fbond += eps[type] * (nn[type] * mm[type] / (nn[type] - mm[type])) *
          (pow(sig[type] / r, nn[type]) - pow(sig[type] / r, mm[type])) / rsq;
      }
      let ebond = -0.5 * K[type] * r0sq * log(rlogarg);
      if (rsq < sigsq) {
        ebond += (eps[type] / (nn[type] - mm[type])) *
          (mm[type] * pow(sig[type] / r, nn[type]) - nn[type] * pow(sig[type] / r, mm[type]));
      }
      f[3 * i1] += delx * fbond; f[3 * i1 + 1] += dely * fbond; f[3 * i1 + 2] += delz * fbond;
      f[3 * i2] -= delx * fbond; f[3 * i2 + 1] -= dely * fbond; f[3 * i2 + 2] -= delz * fbond;
      evTallyBond(bc, i1, i2, ebond, fbond, d);
    }
  }

  /** bond_fene_nm.cpp equilibrium_distance(): 0.97 sigma. */
  equilibrium(type: number): number { return 0.97 * this.params.p('sigma')[type]; }
}
