import { PairParams, StyleError, type AdaptAccess, type Pair } from '../force/types';
import { PAIR_EXTRACT } from './adaptPairsTable';

/**
 * Parameters whose native array init_one writes again at every init from the pair_coeff values, before the force
 * computation reads it, so a value fix adapt writes there has no effect (each from the init_one of the file named):
 * the engine keeps them only for fix adapt to write to.
 */
const INERT: Record<string, readonly string[]> = {};
const inert = (names: readonly string[], styles: readonly string[]): void => { for (const s of styles) INERT[s] = names; };
// KSPACE/pair_lj_charmm_coul_long.cpp, MOLECULE/pair_lj_charmm_coul_charmm.cpp and the charmmfsw files: lj14_1 .. lj14_4
inert(['lj14_1', 'lj14_2', 'lj14_3', 'lj14_4'], [
  'lj/charmm/coul/charmm', 'lj/charmm/coul/charmm/implicit', 'lj/charmm/coul/long', 'lj/charmm/coul/long/soft',
  'lj/charmm/coul/msm', 'lj/charmmfsw/coul/charmmfsh', 'lj/charmmfsw/coul/long',
]);
// CG-SPICA/pair_lj_spica.cpp and pair_lj_spica_coul_long.cpp (lj/sdk the old names): lj1 .. lj4, rminsq, emin
inert(['lj1', 'lj2', 'lj3', 'lj4', 'rminsq', 'emin'], [
  'lj/spica', 'lj/spica/coul/long', 'lj/spica/coul/msm', 'lj/sdk', 'lj/sdk/coul/long', 'lj/sdk/coul/msm',
]);
// MOFFF/pair_buck6d_coul_gauss_dsf.cpp and pair_buck6d_coul_gauss_long.cpp
inert(['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'cut_ljsq', 'offset', 'rsmooth_sq'], ['buck6d/coul/gauss/dsf', 'buck6d/coul/gauss/long']);
// KSPACE/pair_lj_long_coul_long.cpp (sigma, epsilon from sigma_read, epsilon_read; B is lj4) and the styles that use its
// init_one, and DIPOLE/pair_lj_long_dipole_long.cpp
inert(['B', 'epsilon', 'sigma'], ['lj/long/coul/long', 'lj/long/coul/long/dielectric', 'lj/long/tip4p/long', 'lj/long/dipole/long']);
// KSPACE/pair_buck_long_coul_long.cpp: B is buck_c, copied from buck_c_read
inert(['B'], ['buck/long/coul/long']);
// COLLOID/pair_colloid.cpp: diameter is written by coeff and the mixing of init_one but read by neither compute nor init_one
inert(['diameter'], ['colloid']);

/**
 * Tables this engine keeps under a name extract() also uses that do not hold native's array: coul/long's and
 * coul/slater/long's "scale" table records which pairs pair_coeff set (their compute does not apply a scale yet).
 */
const NOT_NATIVE: Record<string, readonly string[]> = {
  'coul/long': ['scale'], 'coul/long/cs': ['scale'], 'coul/long/dielectric': ['scale'], 'coul/esp': ['scale'],
  'coul/msm': ['scale'], 'coul/slater/long': ['scale'],
};

/** Whether the style's native extract() knows the name (PairHybrid::extract asks its sub-styles in order). */
export const pairExtractHas = (styleName: string, pparam: string): boolean => (PAIR_EXTRACT[styleName] ?? []).some((e) => e[0] === pparam);

/**
 * The dim of the parameter a fix adapt pair attribute names, as native's FixAdapt::init finds it through extract():
 * "not supported" when extract() returns nothing, "not compatible" for a dim other than 2 or 0.
 */
export const pairExtractDim = (styleName: string, pstyle: string, pparam: string): number => {
  const entry = (PAIR_EXTRACT[styleName] ?? []).find((e) => e[0] === pparam);
  if (!entry) throw new StyleError(`Fix adapt pair style ${pstyle} param ${pparam} not supported`);
  if (entry[1] !== 2 && entry[1] !== 0) throw new StyleError(`Pair style parameter ${pparam} is not compatible with fix adapt`);
  return entry[1];
};

/**
 * Where this engine keeps the parameter (pairExtractDim has accepted it): the style's own accessor (adaptParam), a
 * value native recomputes at every init (INERT), or the PairParams table of that name; an error naming the parameter
 * when the engine keeps no such value.
 */
export const pairAdaptAccess = (style: Pair, styleName: string, pstyle: string, pparam: string): AdaptAccess => {
  const entry = (PAIR_EXTRACT[styleName] ?? []).find((e) => e[0] === pparam)!;
  const [, dim, alias] = entry;
  const own = style.adaptParam(pparam);
  if (own) return own;
  if (INERT[styleName]?.includes(pparam)) {
    const nt = style.ntypes + 1;
    const arr = new Float64Array(nt * nt);
    if (dim === 2) return { dim: 2, get: (i, j) => arr[i * nt + j], set: (i, j, x) => { arr[i * nt + j] = x; } };
  }
  // a cutoff decides the neighbor lists and the force cutoff of the run's init: left to styles that state it themselves
  if (dim === 2 && !/cut/.test(pparam) && !NOT_NATIVE[styleName]?.includes(pparam)) {
    const want = (alias ?? pparam).toLowerCase();
    for (const v of Object.values(style)) {
      if (!(v instanceof PairParams)) continue;
      const name = v.names.find((n) => n.toLowerCase() === want);
      if (name === undefined) continue;
      const arr = v.p(name);
      return {
        dim: 2,
        get: (i, j) => arr[v.idx(i, j)],
        set: (i, j, x) => { arr[v.idx(i, j)] = x; arr[v.idx(j, i)] = x; },
      };
    }
  }
  throw new StyleError(
    `Fix adapt pair style ${pstyle} param ${pparam}: the browser engine cannot change this parameter of pair style ${styleName}`,
  );
};

/** The helpers fix adapt reads through gplHook('adapt') (registry/adapt_pairs.ts loads this module on first use). */
export interface AdaptPairsHook {
  pairExtractHas: typeof pairExtractHas;
  pairExtractDim: typeof pairExtractDim;
  pairAdaptAccess: typeof pairAdaptAccess;
}
