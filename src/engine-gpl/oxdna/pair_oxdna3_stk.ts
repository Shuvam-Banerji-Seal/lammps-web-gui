/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna3_stk.cpp and
 * pair_oxdna3_stk.h (class PairOxdna3Stk, pair_style oxdna3/stk). The oxDNA3 stacking is the oxDNA stacking
 * (PairOxdnaStk) with the oxDNA3 stacking site (NucleotideOxdna3::stacking_site, dx_cstk_oxdna3) and with the
 * sequence dependence of the potential file: the stacking cut-offs cut_0, cut_c, cut_lo, cut_hi, a4, and dtheta4
 * are per tetramer (tetramerTable in common.ts), and the strength is always scaled by eta_st[i][j] (the C++ has no
 * seqav/seqdep switch here). The coeff line is "pair_coeff I J T file": the temperature enters the strength as
 * xi + kappa T. Original author (from the headers): Oliver Henrich (University of Strathclyde, Glasgow); described
 * by Ouldridge, Louis and Doye (J Chem Phys 2011). LAMMPS is Copyright (2003) Sandia Corporation, distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { StyleError, typeBounds, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { PairOxdnaStk, stackingDerived, stackingStrength } from './pair_stk';
import { type StackRule, stackingSiteOxdna3, tetIndex, tetramerTable, potentialLines } from './common';

/**
 * eta_st[r][c] of pair_oxdna3_stk.cpp (the constructor): "sequence-specific stacking strength, A:0 C:1 G:2 T:3,
 * 3'- [i][j] -5'" (row = imod4 - 1, column = jmod4 - 1; the same numbers as the oxDNA3 coaxial table).
 */
const ETA_STK_OXDNA3: number[][] = [
  [1.1217958408368172, 1.1217958408368172, 1.1217958408368172, 0.9300223683636719],
  [1.0712851690057155, 0.7892685731520542, 0.9896542231533637, 0.7694592613578328],
  [1.1161603311902566, 1.1022201982984874, 1.1088392608169480, 1.0007533199170144],
  [1.0052361315065244, 0.8658975520778347, 1.1217958408368172, 0.8593983791552220],
];

/** pair_style oxdna3/stk (PairOxdna3Stk). */
export class PairOxdna3Stk extends PairOxdnaStk {
  override readonly name: string = 'oxdna3/stk';
  protected override stackSite: StackRule = stackingSiteOxdna3;
  protected override etaSt: number[][] = ETA_STK_OXDNA3;

  /**
   * pair_oxdna3_stk.cpp coeff(): "pair_coeff I J T file" (narg 4); the file line "I J stk" holds xi, kappa, a, the
   * four n^4 tables cut_0, cut_c, cut_lo, cut_hi, the n^4 table a4, theta4_0, the n^4 table dtheta4_ast, and then
   * a5 theta5_0 dtheta5_ast a6 theta6_0 dtheta6_ast a1 cosphi1_ast a2 cosphi2_ast.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 4) throw new StyleError('Incorrect args for pair coefficients in oxdna3/stk, use potential file');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    if (ilo !== jlo || ihi !== jhi) throw new StyleError('Incorrect args for pair coefficients in oxdna3/stk');
    if (ihi > 4) throw new StyleError('pair oxdna3/stk does not support more than 4 atom types for A, C, G and T');
    const T = parseNum(args[2], 'T');
    const n = ihi;
    const m4 = n ** 4;
    const text = ctx?.readFile(args[3]) ?? '';
    let found: number[] | null = null;
    for (const w of potentialLines(text, [args[0], args[1]], 'stk')) {
      if (w.length < 3) continue;
      if (w[2] === 'stk') {
        const need = 14 + 6 * m4;
        if (w.length < 3 + need) throw new StyleError(`Problem parsing oxDNA3 potential file: stk needs ${need} values in ${args[3]}`);
        found = w.slice(3, 3 + need).map((x) => parseNum(x, 'stk'));
        break;
      }
    }
    if (!found) throw new StyleError(`No corresponding stk potential found in file ${args[3]} for pair type ${args[0]} ${args[1]}`);
    const [xi, kappa, a] = found;
    let p = 3;
    const take = (count: number): number[] => { const r = found!.slice(p, p + count); p += count; return r; };
    const cut0T = tetramerTable(take(m4), n), cutCT = tetramerTable(take(m4), n);
    const cutLoT = tetramerTable(take(m4), n), cutHiT = tetramerTable(take(m4), n);
    const a4T = tetramerTable(take(m4), n);
    const [th4_0] = take(1);
    const dth4T = tetramerTable(take(m4), n);
    const [a5, th5, dth5, a6, th6, dth6, a1, cos1, a2, cos2] = take(10);

    const epsOne = stackingStrength(xi, kappa, T);
    const m = n + 1;
    const at = (i: number, j: number, k: number, l: number) => tetIndex(m, i, j, k, l);
    // smoothing: the strength and the lo and hi tails of the AAAA tetramer (1, 1, 1, 1) apply to all pairs
    const aaaa = at(1, 1, 1, 1);
    const { bLo: bLoOne, bHi: bHiOne } = stackingDerived(epsOne, a, cut0T[aaaa], cutCT[aaaa], cutLoT[aaaa], cutHiT[aaaa]);
    const b5 = a5 * a5 * dth5 * dth5 / (1 - a5 * dth5 * dth5), dth5c = 1 / (a5 * dth5);
    const b6 = a6 * a6 * dth6 * dth6 / (1 - a6 * dth6 * dth6), dth6c = 1 / (a6 * dth6);
    const b1 = a1 * a1 * cos1 * cos1 / (1 - a1 * cos1 * cos1), cos1c = 1 / (a1 * cos1);
    const b2 = a2 * a2 * cos2 * cos2 / (1 - a2 * cos2 * cos2), cos2c = 1 / (a2 * cos2);
    const nt = this.nt;
    const mod4 = (t: number) => (t % 4 === 0 ? 4 : t % 4);
    // parameters of the base-step pairs: epsilon_st[i][j] = epsilon_st_one eta_st[i-1][j-1]
    for (let i = ilo; i <= ihi; i++) {
      for (let j = jlo; j <= jhi; j++) {
        const q = i * nt + j;
        this.eps[q] = epsOne * this.etaSt[mod4(i) - 1][mod4(j) - 1];
        this.aSt[q] = a; this.bLo[q] = bLoOne; this.bHi[q] = bHiOne; this.th4[q] = th4_0;
        this.a5[q] = a5; this.th5[q] = th5; this.dth5ast[q] = dth5; this.b5[q] = b5; this.dth5c[q] = dth5c;
        this.a6[q] = a6; this.th6[q] = th6; this.dth6ast[q] = dth6; this.b6[q] = b6; this.dth6c[q] = dth6c;
        this.a1[q] = a1; this.cos1ast[q] = cos1; this.b1[q] = b1; this.cos1c[q] = cos1c;
        this.a2[q] = a2; this.cos2ast[q] = cos2; this.b2[q] = b2; this.cos2c[q] = cos2c;
        this.set[q] = 1;
      }
    }
    // tetramer parameters: [i][j][k][l] with i and l from the terminal type 0 (the tables of the file)
    for (let i = 0; i <= n; i++) {
      for (let j = ilo; j <= jhi; j++) {
        for (let k = ilo; k <= jhi; k++) {
          for (let l = 0; l <= n; l++) {
            const idx = at(i, j, k, l);
            const q = ((i * nt + j) * nt + k) * nt + l;
            const c0 = cut0T[idx], cc = cutCT[idx], clo = cutLoT[idx], chi = cutHiT[idx];
            const ulo = clo - c0, uhi = chi - c0;
            this.c0[q] = c0;
            this.clo[q] = clo;
            this.chi[q] = chi;
            this.clc[q] = clo - a * Math.exp(-a * ulo) * (1 - Math.exp(-a * ulo)) / bLoOne;
            this.chc[q] = chi - a * Math.exp(-a * uhi) * (1 - Math.exp(-a * uhi)) / bHiOne;
            const tmp = 1 - Math.exp(-(cc - c0) * a);
            this.shiftSt[q] = epsOne * this.etaSt[mod4(j) - 1][mod4(k) - 1] * tmp * tmp;
            const a4 = a4T[idx], dth4 = dth4T[idx];
            this.a4[q] = a4;
            this.dth4ast[q] = dth4;
            this.b4[q] = a4 * a4 * dth4 * dth4 / (1 - a4 * dth4 * dth4);
            this.dth4c[q] = 1 / (a4 * dth4);
          }
        }
      }
    }
  }
}
