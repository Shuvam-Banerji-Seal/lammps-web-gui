/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the extract() methods of src/CG-DNA/pair_oxdna_*.cpp, pair_oxdna2_*.cpp, pair_oxdna3_*.cpp and
 * pair_oxrna2_*.cpp at commit b332f28 (contributing authors Oliver Henrich, Ferdinando Randisi and Kaitlin Lu). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix adapt on the oxDNA and oxRNA pair styles: their extract() names (epsilon_hb, a_xst1, theta_cxst4_0,
 * cosphi_st1_ast, cut_hb_lc, b_dh, ...) name per-type-pair arrays (dim 2) that the engine keeps under short keys ('a1',
 * 'th4', 'cos1ast', 'cutlc', ...). the force loop reads the arrays, so a value set here acts at once (after init_one's mirroring); values coeff derives from others (b_*, the cutoff squares) are not re-derived, as in
 * native.
 */
import type { AdaptAccess } from '../../engine/force/types';

/** The engine key of an extract() name ('a_hb1' -> 'a1', 'theta_xst7_0_33' -> 'th7_33'), or null when there is none. */
export const oxdnaKey = (name: string): string | null => {
  let m: RegExpExecArray | null;
  if ((m = /^(epsilon|k|a)_[a-z]+$/.exec(name))) return m[1];
  if ((m = /^cut_[a-z]+_(0|c|lo|hi|lc|hc)(?:_(33|55))?$/.exec(name))) return `cut${m[1]}${m[2] ? `_${m[2]}` : ''}`;
  if ((m = /^b_[a-z]+_(lo|hi)$/.exec(name))) return `b${m[1]}`;
  if (/^shift_[a-z]+$/.test(name)) return 'shift';
  if ((m = /^a_[a-z]+?(\d+p?)$/.exec(name))) return `a${m[1]}`;
  if ((m = /^theta_[a-z]+?(\d+)_0(?:_(33|55))?$/.exec(name))) return `th${m[1]}${m[2] ? `_${m[2]}` : ''}`;
  if ((m = /^dtheta_[a-z]+?(\d+)_(ast|c)$/.exec(name))) return `dth${m[1]}${m[2]}`;
  if ((m = /^b_[a-z]+?(\d+p?)$/.exec(name))) return `b${m[1]}`;
  if ((m = /^cosphi_[a-z]+?(\d+p?)_(ast|c)$/.exec(name))) return `cos${m[1]}${m[2]}`;
  if ((m = /^(AA|BB)_[a-z]+\d$/.exec(name))) return m[1];
  return null;
};

/**
 * dim 2 access to a per-type-pair table (nt entries per axis): set writes [i][j] only. fix adapt sets the pairs I <= J
 * and then re-initializes the style (Pair::reinit), whose initOne mirrors [j][i] as native's init_one does: not at all
 * in oxdna/stk, all but k_cxst in the oxDNA2 coaxial stacking, epsilon_hb and shift_hb through the alpha ratio in hbond.
 */
export const oxdnaTableAccess = (tab: Float64Array, nt: number): AdaptAccess => ({
  dim: 2,
  get: (i, j) => tab[i * nt + j],
  set: (i, j, v) => { tab[i * nt + j] = v; },
});

/** The access of an extract() name in a record of per-type-pair tables keyed as oxdnaKey gives, or null. */
export const oxdnaRecordAccess = (name: string, tables: Record<string, Float64Array> | undefined, nt: number, rename: Record<string, string> = {}): AdaptAccess | null => {
  const key = oxdnaKey(name);
  if (key === null || !tables) return null;
  const tab = tables[rename[key] ?? key];
  return tab && tab.length === nt * nt ? oxdnaTableAccess(tab, nt) : null;
};
