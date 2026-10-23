/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MESONT/bond_mesocnt.cpp and src/MESONT/bond_mesocnt.h at commit b332f28 (contributing author Philipp
 * Kloza, University of Cambridge). Copyright (2003) Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * bond_style mesocnt: the harmonic bond E = K (r - r0)^2 (BondMesoCNT inherits BondHarmonic and only changes coeff()).
 * bond_coeff types "custom K r0" or "C n m r0": the K of the (n,m) tube is 0.5 (86.64 + 100.56 r) eunit / (ang r0)
 * with the tube radius r of chiralRadius (the C++ coeff() formula).
 */
import { BondHarmonic } from '../../engine/force/bond/harmonic';
import { SimpleBonded } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { inumeric, parseNum } from '../../engine/force/util';
import { ANGSTROM } from '../../engine/force/kspace/ewald';
import { chiralRadius, eunitOf } from './units';

export class BondMesoCNT extends SimpleBonded {
  readonly name = 'mesocnt';
  readonly kind = 'bond' as const;
  readonly paramNames = ['K', 'r0'];
  /** The harmonic term E = K (r - r0)^2 of the C++ base class BondHarmonic, computed with this style's coefficients. */
  private readonly harmonic = new BondHarmonic();

  compute(bc: BondedCompute): void {
    this.harmonic.params = this.params;
    this.harmonic.compute(bc);
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }

  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length < 2) throw new StyleError('Illegal bond_coeff command for bond_style mesocnt');
    const style = ctx?.s?.units.style;
    if (style === undefined) throw new StyleError('bond_style mesocnt needs the box and units before bond_coeff');
    const eunit = eunitOf(style, 'Bond style mesocnt');
    const ang = ANGSTROM[style];
    let k: number, r0: number;
    if (args[1] === 'custom') {
      if (args.length !== 4) throw new StyleError("Incorrect number of args for 'custom' bond coefficients");
      k = parseNum(args[2], 'K');
      r0 = parseNum(args[3], 'r0');
    } else if (args[1] === 'C') {
      if (args.length !== 5) throw new StyleError("Incorrect number of args for 'C' preset in bond coefficients");
      const n = inumeric(args[2]);
      const m = inumeric(args[3]);
      r0 = parseNum(args[4], 'r0');
      const rAng = chiralRadius(n, m);
      k = 0.5 * (86.64 + 100.56 * rAng) * eunit / (ang * r0);
    } else throw new StyleError(`Unknown ${args[1]} preset in bond coefficients`);
    this.params.setRange(args[0], [k, r0]);
  }
}
