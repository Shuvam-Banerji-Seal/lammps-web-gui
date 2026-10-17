/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna3_excv.cpp and
 * pair_oxdna3_excv.h (class PairOxdna3Excv, pair_style oxdna3/excv). The oxDNA3 excluded volume is the oxDNA2
 * one (the oxDNA2 backbone site, PairOxdna2Excv) with two changes: the base site depends on the nucleotide type
 * (NucleotideOxdna3::base_site<N>: purines and pyrimidines have their own offsets), and the base-base sigma and
 * cut_ast of the tetramer tables are read from the potential file ("oxdna3 potential", "excv" line: the nine
 * site values, then the n^4 sigma4 and the n^4 cut4 values, averaged as in tetramerTable). Original author (from
 * the headers): Oliver Henrich (University of Strathclyde, Glasgow); described by Ouldridge, Louis and Doye (J Chem
 * Phys 2011) and Snodin et al. LAMMPS is Copyright (2003) Sandia Corporation, distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { StyleError, typeBounds, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { derived } from './pair_excv';
import { PairOxdna2Excv } from './variants';
import { type OxdnaConstants, baseSiteOxdna3, tetIndex, tetramerTable, potentialLines } from './common';
import { pow as libmPow } from '../libm/pow';

/** pair_style oxdna3/excv (PairOxdna3Excv). */
export class PairOxdna3Excv extends PairOxdna2Excv {
  override readonly name: string = 'oxdna3/excv';
  protected override baseSiteOf: (c: OxdnaConstants, type: number, e1: ArrayLike<number>, out: number[]) => void = baseSiteOxdna3;

  /**
   * pair_oxdna3_excv.cpp coeff(): "pair_coeff I J file" only (narg 3), "Incorrect args for pair coefficients in
   * oxdna3/excv, use potential file"; more than 4 atom types is an error.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for pair coefficients in oxdna3/excv, use potential file');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    if (ilo !== jlo || ihi !== jhi) throw new StyleError('Incorrect args for pair coefficients in oxdna3/excv');
    if (ihi > 4) throw new StyleError('pair oxdna3/excv does not support more than 4 atom types for A, C, G and T');
    const n = ihi;
    const m4 = libmPow(n, 4);
    const text = ctx?.readFile(args[2]) ?? '';
    let found: number[] | null = null;
    for (const w of potentialLines(text, [args[0], args[1]], 'excv')) {
      if (w.length < 3) continue;
      if (w[2] === 'excv') {
        if (w.length < 3 + 9 + 2 * m4) throw new StyleError(`Problem parsing oxdna3 potential file: excv needs 9 and ${2 * m4} values in ${args[2]}`);
        found = w.slice(3, 3 + 9 + 2 * m4).map((x) => parseNum(x, 'excv'));
        break;
      }
    }
    if (!found) throw new StyleError(`No corresponding excv potential found in file ${args[2]} for pair type ${args[0]} ${args[1]}`);
    const v9 = found.slice(0, 9);
    const sig4 = tetramerTable(found.slice(9, 9 + m4), n);
    const cut4 = tetramerTable(found.slice(9 + m4, 9 + 2 * m4), n);
    this.fillSites(ilo, ihi, v9);
    // base-base parameters depending on the tetramer (i: 3' neighbour, j, k: the pair, l: 5' neighbour; 0 = none)
    const nt = this.nt;
    const t = this.t4;
    for (let i = 0; i <= n; i++) {
      for (let j = ilo; j <= ihi; j++) {
        for (let k = ilo; k <= ihi; k++) {
          for (let l = 0; l <= n; l++) {
            const q = ((i * nt + j) * nt + k) * nt + l;
            const sigma = sig4[tetIndex(n + 1, i, j, k, l)];
            const cutAst = cut4[tetIndex(n + 1, i, j, k, l)];
            const { b, cutC } = derived(sigma, cutAst);
            t.sigma[q] = sigma;
            t.cutAst[q] = cutAst;
            t.cutsqAst[q] = cutAst * cutAst;
            t.b[q] = b;
            t.cutC[q] = cutC;
            t.cutsqC[q] = cutC * cutC;
            t.lj14[q] = 4.0 * this.tEps[j * nt + k] * libmPow(sigma, 12.0);
            t.lj24[q] = 4.0 * this.tEps[j * nt + k] * libmPow(sigma, 6.0);
          }
        }
      }
    }
  }
}
