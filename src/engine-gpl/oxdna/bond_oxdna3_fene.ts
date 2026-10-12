/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/bond_oxdna3_fene.cpp and
 * bond_oxdna3_fene.h (class BondOxdna3Fene, bond_style oxdna3/fene). It is the oxDNA2 bond (the oxDNA2 backbone
 * site, BondOxdna2Fene) with Delta and r0 looked up per tetramer: the 3' neighbour type, the bonded atom types and
 * the 5' neighbour type. coeff() reads them from the potential file ("oxdna3 potential", "fene" line: K, then the
 * n^4 Delta values, then the n^4 r0 values) and averages them over the neighbour types as the C++ does (see
 * tetramerTable in common.ts). Original author (from the headers): Oliver Henrich (University of Strathclyde,
 * Glasgow); described by Snodin et al. (oxDNA2) and Ouldridge, Louis and Doye (J Chem Phys 2011). LAMMPS is
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { StyleError, typeBounds, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { BondOxdnaFene, type Pair2 } from './bond_fene';
import { backboneOxdna2, tetIndex, tetramerTable, type OxdnaConstants, potentialLines } from './common';

/** The tables of one bond type: K, the four-index Delta and r0 (m = atom types + 1 entries per axis). */
interface Tables { k: number; m: number; delta: Float64Array; r0: Float64Array }

/** bond_style oxdna3/fene (BondOxdna3Fene). */
export class BondOxdna3Fene extends BondOxdnaFene {
  override readonly name = 'oxdna3/fene';
  protected override tetramer = true;
  private tabs: (Tables | undefined)[] = [];
  private scratch: Pair2 = { k: 0, delta: 0, r0: 0 };

  protected override backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxdna2(c, e1, e2, e3, out);
  }

  /**
   * bond_oxdna3_fene.cpp coeff(): "Incorrect args for bond coefficients in oxdna3/fene, use potential file" unless
   * narg is 2; "bond oxdna3/fene does not support more than 4 atom types for A, C, G and T" above 4 atom types.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 2) throw new StyleError('Incorrect args for bond coefficients in oxdna3/fene, use potential file');
    const n = ctx?.s?.ntypes ?? 0;
    if (n > 4) throw new StyleError('bond oxdna3/fene does not support more than 4 atom types for A, C, G and T');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const text = ctx?.readFile(args[1]) ?? '';
    const m4 = n ** 4;
    let found: number[] | null = null;
    for (const w of potentialLines(text, [args[0]], 'fene')) {
      if (w.length < 2) continue;
      if (w[1] === 'fene') {
        if (w.length < 3 + 2 * m4) throw new StyleError(`Problem parsing oxdna3 potential file: fene needs K and ${2 * m4} values in ${args[1]}`);
        found = w.slice(2, 3 + 2 * m4).map((x) => parseNum(x, 'fene'));
        break;
      }
    }
    if (!found) throw new StyleError(`No corresponding fene potential found in file ${args[1]} for bond type ${args[0]}`);
    const k = found[0];
    const tab: Tables = {
      k,
      m: n + 1,
      delta: tetramerTable(found.slice(1, 1 + m4), n),
      r0: tetramerTable(found.slice(1 + m4, 1 + 2 * m4), n),
    };
    for (let t = lo; t <= hi; t++) {
      this.tabs[t] = tab;
      this.par[t] = { k, delta: 0, r0: 0 };
    }
  }

  /** the Delta and r0 of the tetramer (a3p, atype, btype, b5p) of a bond of type `type` (fene in bond_oxdna3_fene.cpp). */
  protected override params(type: number, a3p: number, atype: number, btype: number, b5p: number): Pair2 {
    const t = this.tabs[type];
    if (!t) throw new StyleError(`All bond coeffs are not set (bond type ${type})`);
    const q = tetIndex(t.m, a3p, atype, btype, b5p);
    this.scratch.k = t.k;
    this.scratch.delta = t.delta[q];
    this.scratch.r0 = t.r0[q];
    return this.scratch;
  }
}
