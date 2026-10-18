/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MOLECULE/bond_quartic.cpp and
 * bond_quartic.h (Contributing authors: Chris Lorenz and Mark Stevens, SNL; Copyright (2003) Sandia Corporation,
 * GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * bond_style quartic (docs.lammps.org/bond_quartic.html): breakable bonds. The bond energy is
 *   E = K (r - Rc)^2 (r - Rc - B1) (r - Rc - B2) + U0,
 * plus a LJ term cut at its minimum (sigma = epsilon = 1), which the bond subtracts through the pair style's single():
 * "subtracts the pairwise interaction as part of the bond computation" and so needs special_bonds 1,1,1. A bond longer
 * than Rc breaks: its type is set to 0 and the subtraction stops (the pair then acts on the two atoms normally).
 * Coefficients: "K B1 B2 Rc U0" (bond_coeff N K B1 B2 Rc U0).
 *
 * The loop, the expressions and their order are native's (bond_quartic.cpp compute()).
 */
import { SimpleBonded, atomIndex, bondDelta } from '../../engine/force/bonded_util';
import { Pair, StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { evTallyBond } from '../xmolecule/ev';

/** MY_CUBEROOT2 (math_const.h): 2^(1/3). */
const MY_CUBEROOT2 = 1.25992104989487316476;

/**
 * Pair::ev_tally (pair.cpp, newton on, both atoms owned) for the subtracted pair term: the energy goes to the pair's
 * van der Waals sum, the virial to the pair's virial bucket, and the per-atom split is the pair's (half to each atom).
 */
const pairTally = (bc: BondedCompute, i: number, j: number, evdwl: number, fpair: number, d: readonly number[]): void => {
  bc.acc.evdwl += evdwl;
  const dx = d[0], dy = d[1], dz = d[2];
  const v = bc.acc.virial;
  const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
  const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
  v[0] += v0; v[1] += v1; v[2] += v2; v[3] += v3; v[4] += v4; v[5] += v5;
  if (bc.eatom) { bc.eatom[i] += 0.5 * evdwl; bc.eatom[j] += 0.5 * evdwl; }
  if (bc.vatom) {
    for (const a of [i, j]) {
      bc.vatom[6 * a] += 0.5 * v0; bc.vatom[6 * a + 1] += 0.5 * v1; bc.vatom[6 * a + 2] += 0.5 * v2;
      bc.vatom[6 * a + 3] += 0.5 * v3; bc.vatom[6 * a + 4] += 0.5 * v4; bc.vatom[6 * a + 5] += 0.5 * v5;
    }
  }
};

export class BondQuartic extends SimpleBonded {
  readonly name = 'quartic';
  readonly kind = 'bond' as const;
  readonly paramNames = ['k', 'b1', 'b2', 'rc', 'u0'];
  private pair: Pair | null = null;
  private special: { lj: readonly number[]; coul: readonly number[] } = { lj: [0, 0, 0], coul: [0, 0, 0] };

  /** The pair style the bond subtracts through single(), and the special_bonds weights (checked at init). */
  linkForceField(link: { pair: Pair | null; special: { lj: readonly number[]; coul: readonly number[] } }): void {
    this.pair = link.pair;
    this.special = link.special;
  }

  /** bond_quartic.cpp coeff(): narg must be 6 ("Incorrect args for bond coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 6) throw new StyleError('Incorrect args for bond coefficients');
    super.coeff(args);
  }

  /** bond_quartic.cpp init_style(): the pair style must have single(), no 3- or 4-body terms, special_bonds 1,1,1. */
  init(ctx?: StyleContext): void {
    if (!this.pair || typeof this.pair.single !== 'function') throw new StyleError('Pair style does not support bond_style quartic');
    // native checks that no angle, dihedral or improper style is defined; the topology stands in for the style here
    const t = ctx?.s?.topo;
    if (t && (t.angles.n > 0 || t.dihedrals.n > 0 || t.impropers.n > 0)) {
      throw new StyleError('Bond style quartic cannot be used with 3,4-body interactions');
    }
    // special_lj[1..3] (1-2, 1-3, 1-4) are lj[0..2] here
    const lj = this.special.lj;
    if (lj[0] !== 1.0 || lj[1] !== 1.0 || lj[2] !== 1.0) throw new StyleError('Bond style quartic requires special_bonds = 1,1,1');
    super.init(ctx);
  }

  /** bond_quartic.cpp equilibrium_distance(): 0.97. */
  equilibrium(_type: number): number {
    return 0.97;
  }

  compute(bc: BondedCompute): void {
    const s = bc.s;
    const bonds = s.topo.bonds;
    const pair = this.pair!;
    const K = this.params.p('k'), B1 = this.params.p('b1'), B2 = this.params.p('b2');
    const RC = this.params.p('rc'), U0 = this.params.p('u0');
    const nt = pair.ntypes + 1;
    const f = bc.f;
    const d = [0, 0, 0];
    for (let n = 0; n < bonds.n; n++) {
      const type = bonds.type[n];
      // skip bond if already broken (native: bondlist[n][2] <= 0)
      if (type <= 0) continue;
      const i1 = atomIndex(bc, bonds.atoms[2 * n], 'bond');
      const i2 = atomIndex(bc, bonds.atoms[2 * n + 1], 'bond');
      bondDelta(bc, i1, i2, d);
      const delx = d[0], dely = d[1], delz = d[2];
      const rsq = delx * delx + dely * dely + delz * delz;
      // if the bond breaks, its type is set to 0 (native sets it in both atoms' bond lists, one entry here)
      if (rsq > RC[type] * RC[type]) {
        bonds.type[n] = 0;
        continue;
      }
      // quartic term, plus the LJ term cut at its minimum (eps = sigma = 1)
      const r = Math.sqrt(rsq);
      const dr = r - RC[type];
      const r2 = dr * dr;
      const ra = dr - B1[type];
      const rb = dr - B2[type];
      let fbond = -K[type] / r * (r2 * (ra + rb) + 2.0 * dr * ra * rb);
      const near = rsq < MY_CUBEROOT2;
      let sr6 = 0;
      if (near) {
        const sr2 = 1.0 / rsq;
        sr6 = sr2 * sr2 * sr2;
        fbond += 48.0 * sr6 * (sr6 - 0.5) / rsq;
      }
      let ebond = K[type] * r2 * ra * rb + U0[type];
      if (near) ebond += 4.0 * sr6 * (sr6 - 1.0) + 1.0;

      f[3 * i1] += delx * fbond; f[3 * i1 + 1] += dely * fbond; f[3 * i1 + 2] += delz * fbond;
      f[3 * i2] -= delx * fbond; f[3 * i2 + 1] -= dely * fbond; f[3 * i2 + 2] -= delz * fbond;
      evTallyBond(bc, i1, i2, ebond, fbond, d);

      // subtract the pairwise interaction of the two atoms (pair->single with factors 1, 1)
      const itype = s.type[i1], jtype = s.type[i2];
      if (rsq < pair.cutsq[itype * nt + jtype]) {
        const single = pair.single!(i1, i2, itype, jtype, rsq, 1.0, 1.0, s.q[i1], s.q[i2]);
        const evdwl = -single.eng;
        const fpair = -single.fforce;
        f[3 * i1] += delx * fpair; f[3 * i1 + 1] += dely * fpair; f[3 * i1 + 2] += delz * fpair;
        f[3 * i2] -= delx * fpair; f[3 * i2 + 1] -= dely * fpair; f[3 * i2 + 2] -= delz * fpair;
        pairTally(bc, i1, i2, evdwl, fpair, d);
      }
    }
  }
}
