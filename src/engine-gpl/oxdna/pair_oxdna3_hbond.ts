/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna3_hbond.cpp and
 * pair_oxdna3_hbond.h (class PairOxdna3Hbond, pair_style oxdna3/hbond). The oxDNA3 hydrogen bonding is the oxDNA
 * one (PairOxdnaHbond) with the oxDNA3 base sites (NucleotideOxdna3::base_site<N>, the type-dependent offsets) and
 * with its own potential-file layout and base-pairing strengths: the epsilon and shift of every pair are scaled by
 * alpha_hb[i][j] (the oxDNA3 table of the constructor, always applied: the coeff line has no seqav/seqdep word),
 * and hydrogen-bond angle 4 has separate A-T (and T-A) and C-G (and G-C) parameters. Original author (from the
 * headers): Oliver Henrich (University of Strathclyde, Glasgow); described by Ouldridge, Louis and Doye (J Chem Phys
 * 2011). LAMMPS is Copyright (2003) Sandia Corporation, distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { StyleError, typeBounds, type StyleContext, potentialFile } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { PairOxdnaHbond } from './pair_hbond';
import { stackingDerived } from './pair_stk';
import { type OxdnaConstants, baseSiteOxdna3, potentialLines } from './common';

/** alpha_hb of pair_oxdna3_hbond.cpp (the constructor): A:0 C:1 G:2 T:3, indexed [i][j] (symmetric). */
const ALPHA_OXDNA3: number[][] = [
  [1.0, 1.0, 1.0, 0.6493620379646540],
  [1.0, 1.0, 1.1999420813642658, 1.0],
  [1.0, 1.1999420813642658, 1.0, 1.0],
  [0.6493620379646540, 1.0, 1.0, 1.0],
];

/** The 27 numbers of "pair_coeff I J file" for the line "I J hbond", in the order of pair_oxdna3_hbond.cpp coeff(). */
const HB3_NAMES = ['epsilon_hb', 'a_hb', 'cut_hb_0', 'cut_hb_c', 'cut_hb_lo', 'cut_hb_hi',
  'a_hb1', 'theta_hb1_0', 'dtheta_hb1_ast', 'a_hb2', 'theta_hb2_0', 'dtheta_hb2_ast',
  'a_hb3', 'theta_hb3_0', 'dtheta_hb3_ast', 'a_hb4_at', 'theta_hb4_0_at', 'dtheta_hb4_ast_at',
  'a_hb4_cg', 'theta_hb4_0_cg', 'dtheta_hb4_ast_cg', 'a_hb7', 'theta_hb7_0', 'dtheta_hb7_ast', 'a_hb8', 'theta_hb8_0', 'dtheta_hb8_ast'];

/** pair_style oxdna3/hbond (PairOxdna3Hbond). */
export class PairOxdna3Hbond extends PairOxdnaHbond {
  override readonly name: string = 'oxdna3/hbond';
  protected override alpha: number[][] = ALPHA_OXDNA3;
  protected override baseSiteOf: (c: OxdnaConstants, type: number, e1: ArrayLike<number>, out: number[]) => void = baseSiteOxdna3;

  /**
   * pair_oxdna3_hbond.cpp coeff(): "pair_coeff I J file" only (narg 3), "Incorrect args for pair coefficients in
   * oxdna3/hbond, use potential file"; more than 4 atom types is an error.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for pair coefficients in oxdna3/hbond, use potential file');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    if (ihi > 4 || jhi > 4) throw new StyleError('pair oxdna3/hbond does not support more than 4 atom types for A, C, G and T');
    const text = ctx?.readFile(args[2], potentialFile('oxdna3 potential')) ?? '';
    let found: number[] | null = null;
    for (const w of potentialLines(text, [args[0], args[1]], 'hbond')) {
      if (w.length < 3) continue;
      if (w[2] === 'hbond') {
        if (w.length < 3 + HB3_NAMES.length) throw new StyleError(`Problem parsing oxDNA3 potential file: hbond needs 27 values in ${args[2]}`);
        found = w.slice(3, 3 + HB3_NAMES.length).map((x, k) => parseNum(x, HB3_NAMES[k]));
        break;
      }
    }
    if (!found) throw new StyleError(`No corresponding hbond potential found in file ${args[2]} for pair type ${args[0]} ${args[1]}`);
    const v = found;
    const [eps, a, cut0, cutC, cutLo, cutHi, a1, th1, dth1, a2, th2, dth2, a3, th3, dth3,
      a4at, th4at, dth4at, a4cg, th4cg, dth4cg, a7, th7, dth7, a8, th8, dth8] = v;
    const { bLo, cLc, bHi, cHc, shiftOne } = stackingDerived(eps, a, cut0, cutC, cutLo, cutHi);
    const bang = (aa: number, ast: number) => aa * aa * ast * ast / (1 - aa * ast * ast);
    const cang = (aa: number, ast: number) => 1 / (aa * ast);
    const b1 = bang(a1, dth1), c1 = cang(a1, dth1);
    const b2 = bang(a2, dth2), c2 = cang(a2, dth2);
    const b3 = bang(a3, dth3), c3 = cang(a3, dth3);
    const b4at = bang(a4at, dth4at), c4at = cang(a4at, dth4at);
    const b4cg = bang(a4cg, dth4cg), c4cg = cang(a4cg, dth4cg);
    const b7 = bang(a7, dth7), c7 = cang(a7, dth7);
    const b8 = bang(a8, dth8), c8 = cang(a8, dth8);
    const nt = this.nt;
    const mod4 = (t: number) => (t % 4 === 0 ? 4 : t % 4);
    const T = this.t;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const im = mod4(i), jm = mod4(j);
        const q = i * nt + j;
        const alpha = this.alpha[im - 1][jm - 1];
        T.epsilon[q] = eps * alpha;
        T.a[q] = a; T.cut0[q] = cut0; T.cutc[q] = cutC; T.cutlo[q] = cutLo; T.cuthi[q] = cutHi;
        T.cutlc[q] = cLc; T.cuthc[q] = cHc; T.blo[q] = bLo; T.bhi[q] = bHi;
        T.shift[q] = shiftOne * alpha;
        T.a1[q] = a1; T.th1[q] = th1; T.dth1ast[q] = dth1; T.b1[q] = b1; T.dth1c[q] = c1;
        T.a2[q] = a2; T.th2[q] = th2; T.dth2ast[q] = dth2; T.b2[q] = b2; T.dth2c[q] = c2;
        T.a3[q] = a3; T.th3[q] = th3; T.dth3ast[q] = dth3; T.b3[q] = b3; T.dth3c[q] = c3;
        // hydrogen-bond angle 4: the A-T and C-G parameters for their pairs (other pairs keep zeros, as the C++
        // leaves them unset)
        if ((im === 1 && jm === 4) || (im === 4 && jm === 1)) {
          T.a4[q] = a4at; T.th4[q] = th4at; T.dth4ast[q] = dth4at; T.b4[q] = b4at; T.dth4c[q] = c4at;
        } else if ((im === 2 && jm === 3) || (im === 3 && jm === 2)) {
          T.a4[q] = a4cg; T.th4[q] = th4cg; T.dth4ast[q] = dth4cg; T.b4[q] = b4cg; T.dth4c[q] = c4cg;
        }
        T.a7[q] = a7; T.th7[q] = th7; T.dth7ast[q] = dth7; T.b7[q] = b7; T.dth7c[q] = c7;
        T.a8[q] = a8; T.th8[q] = th8; T.dth8ast[q] = dth8; T.b8[q] = b8; T.dth8c[q] = c8;
        this.set[q] = 1;
      }
    }
  }
}
