import type { AdaptAccess } from '../types';

/*
 * fix adapt's scale factor by type pair, as the native styles that extract() a "scale" array keep it (e.g.
 * src/pair_coul_cut.cpp coeff and init_one at b332f28): pair_coeff sets it to 1 for the type pairs it covers, init_one
 * to 1 for a pair pair_coeff did not set and then mirrors it (scale[j][i] = scale[i][j]). Flat, index i * (ntypes + 1) + j.
 */

/** A table of ones for ntypes types. */
export const unitScaleTable = (ntypes: number): Float64Array => new Float64Array((ntypes + 1) * (ntypes + 1)).fill(1);

/** pair_coeff: scale 1 for the type pair (both orders). */
export const resetScale = (scale: Float64Array, ntypes: number, i: number, j: number): void => {
  const nt = ntypes + 1;
  scale[i * nt + j] = scale[j * nt + i] = 1;
};

/** init_one: 1 when pair_coeff did not set the pair, then the mirror entry. */
export const initScale = (scale: Float64Array, ntypes: number, i: number, j: number, set: boolean): void => {
  const nt = ntypes + 1;
  if (!set) scale[i * nt + j] = 1;
  scale[j * nt + i] = scale[i * nt + j];
};

/** fix adapt's access to a style's scale table (read through the style, which may reallocate it). */
export const scaleAccess = (style: { readonly scale: Float64Array; readonly ntypes: number }): AdaptAccess => ({
  dim: 2,
  get: (i, j) => style.scale[i * (style.ntypes + 1) + j],
  set: (i, j, v) => { style.scale[i * (style.ntypes + 1) + j] = v; },
});
