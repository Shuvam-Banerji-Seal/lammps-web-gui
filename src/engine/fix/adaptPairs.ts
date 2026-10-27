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
// KSPACE/pair_coul_streitz.cpp and DRUDE/pair_lj_cut_thole_long.cpp: scale is set by coeff and mirrored by init_one,
// but compute never reads it
inert(['scale'], ['coul/streitz', 'lj/cut/thole/long']);
// DRUDE/pair_thole.cpp: polar and thole are read by coeff (which derives ascreen from them) and mirrored by init_one
inert(['polar', 'thole'], ['thole']);

/**
 * Cutoff and geometry values of native extract() (dim 0) that this engine keeps as a number field of the style: the
 * field, and when native reads the value, from the files named (src/ at b332f28; a style without its own reads its
 * parent class's):
 * - 'init': only when a run initializes. init_style derives what the force loop uses (cut_coulsq, the tables, the
 *   switching constants), init_one at most returns the value as the cutoff (Pair::reinit ignores that), and the next
 *   pair_coeff and the kspace style's init read it. fix adapt's value is held and written to the field when the run
 *   ends (after reset yes restored the original), so the run keeps its init values and the next init reads the new
 *   one, as natively.
 * - 'now': the force loop or init_one reads the value itself; the style reads the field the same way (and keeps what
 *   native's init_style derives apart), so fix adapt writes the field at once.
 */
type CutTiming = 'init' | 'now';
const CUT_FIELDS: Record<string, Record<string, readonly [string, CutTiming]>> = {};
const cutFields = (names: Record<string, readonly [string, CutTiming]>, styles: readonly string[]): void => {
  for (const st of styles) CUT_FIELDS[st] = names;
};
// KSPACE/pair_born_coul_long.cpp, pair_buck_coul_long.cpp, pair_coul_long.cpp, pair_lj_cut_coul_long.cpp and
// pair_lj_charmm_coul_long.cpp, CLASS2/pair_lj_class2_coul_long.cpp, the CORESHELL and DIELECTRIC styles derived from
// them, MOFFF/pair_buck6d_coul_gauss_*.cpp, EXTRA-PAIR/pair_coul_slater_long.cpp, pair_lj_cut_coul_dsf.cpp,
// pair_lj_expand_coul_long.cpp and pair_nm_cut_coul_long.cpp, CG-SPICA/pair_lj_spica_coul_long.cpp,
// DPD-BASIC/pair_dpd_coul_slater_long.cpp, DRUDE/pair_lj_cut_thole_long.cpp, YAFF/pair_lj_switch3_coulgauss_long.cpp and
// pair_mm3_switch3_coulgauss_long.cpp: cut_coul in settings, init_style and init_one's returned cutoff
cutFields({ cut_coul: ['cutCoul', 'init'] }, [
  'born/coul/long', 'born/coul/long/cs', 'buck/coul/long', 'buck/coul/long/cs', 'buck6d/coul/gauss/dsf',
  'buck6d/coul/gauss/long', 'coul/long', 'coul/long/cs', 'coul/long/dielectric', 'coul/slater/long',
  'dpd/coul/slater/long', 'lj/charmm/coul/long', 'lj/class2/coul/long', 'lj/class2/coul/long/cs', 'lj/cut/coul/dsf',
  'lj/cut/coul/long', 'lj/cut/coul/long/cs', 'lj/cut/coul/long/dielectric', 'lj/cut/thole/long', 'lj/expand/coul/long',
  'lj/sdk/coul/long', 'lj/spica/coul/long', 'lj/switch3/coulgauss/long', 'mm3/switch3/coulgauss/long', 'nm/cut/coul/long',
]);
// pair_coul_dsf.cpp: the same, kept as the global cutoff
cutFields({ cut_coul: ['cutGlobal', 'init'] }, ['coul/dsf']);
// FEP/pair_coul_long_soft.cpp, pair_lj_charmm_coul_long_soft.cpp, pair_lj_class2_coul_long_soft.cpp,
// pair_lj_cut_coul_long_soft.cpp and
// DIPOLE/pair_lj_cut_dipole_long.cpp: the same, kept as the global Coulomb cutoff
cutFields({ cut_coul: ['cutCGlobal', 'init'] }, [
  'coul/long/soft', 'lj/charmm/coul/long/soft', 'lj/class2/coul/long/soft', 'lj/cut/coul/long/soft', 'lj/cut/dipole/long',
]);
// MOLECULE/pair_lj_charmmfsw_coul_charmmfsh.cpp and KSPACE/pair_lj_charmmfsw_coul_long.cpp: the cutoffs and the
// switching constants of init_style
cutFields({ cut_coul: ['cutCoul', 'init'], cut_lj_inner: ['inner', 'init'], cut_lj: ['outer', 'init'] }, [
  'lj/charmmfsw/coul/charmmfsh', 'lj/charmmfsw/coul/long',
]);
// EXTRA-PAIR/pair_coul_cut_global.cpp, pair_coul_exclude.cpp and pair_zero_coul.cpp: the global cutoff, read by
// settings, coeff and init_one's returned cutoff only
cutFields({ cut_coul: ['cutGlobal', 'init'] }, ['coul/cut/global', 'coul/exclude', 'zero/coul']);
// SPIN/pair_spin_*.cpp: the global cutoff of settings, the cutoff init_one returns
cutFields({ cut: ['cutGlobal', 'init'] }, ['spin/dmi', 'spin/exchange', 'spin/exchange/biquadratic', 'spin/magelec', 'spin/neel']);
cutFields({ cut: ['cutGlobal', 'init'], cut_coul: ['cutGlobal', 'init'] }, ['spin/dipole/cut', 'spin/dipole/long']);
// KSPACE/pair_*_coul_msm.cpp, DIELECTRIC/pair_lj_cut_coul_msm_dielectric.cpp, CG-SPICA/pair_lj_spica_coul_msm.cpp
// (cut_msm is its cut_coul): the force loop's gamma(r / cut_coul); KSPACE/pair_coul_esp.cpp and
// pair_lj_cut_coul_esp.cpp: its r_scal = 2 r / cut_coul - 1
cutFields({ cut_coul: ['cutCoul', 'now'] }, [
  'born/coul/msm', 'buck/coul/msm', 'coul/msm', 'lj/charmm/coul/msm', 'lj/cut/coul/msm', 'lj/cut/coul/msm/dielectric',
  'lj/sdk/coul/msm', 'coul/esp', 'lj/cut/coul/esp',
]);
cutFields({ cut_coul: ['cutCoul', 'now'], cut_msm: ['cutCoul', 'now'] }, ['lj/spica/coul/msm']);
// KSPACE/pair_coul_streitz.cpp: the Coulomb integrals and the Wolf and Fennell sums of the force loop
cutFields({ cut_coul: ['cutGlobal', 'now'] }, ['coul/streitz']);
// EXTRA-PAIR/pair_coul_ctip.cpp: t_cut and the shifted terms of the force loop (pair_coul_ctip.ts)
cutFields({ cut_coul: ['cutCoul', 'now'] }, ['coul/ctip']);
// the TIP4P styles: init_one's cutoff, the long styles' cut_coulsqplus of the force loop (pair_tip4p.ts)
cutFields({ cut_coul: ['cutCoul', 'now'], qdist: ['qdist', 'now'] }, [
  'lj/cut/tip4p/cut', 'lj/cut/tip4p/long', 'lj/cut/tip4p/long/soft', 'tip4p/cut', 'tip4p/long', 'tip4p/long/soft',
]);
// KSPACE/pair_lj_long_tip4p_long.cpp (its cut_coul and cut_vdwl are those of lj/long/coul/long, the style's own)
cutFields({ qdist: ['qdist', 'now'] }, ['lj/long/tip4p/long']);

/**
 * Per-type-pair cutoffs (dim 2) a PairParams table of the style holds: init_one re-derives what the force loop uses
 * from them (Pair::reinit), and the loop keeps to cutsq of the run's init first, as natively (pair_coul_cut.cpp,
 * pair_coul_debye.cpp, EXTRA-PAIR/pair_coul_slater_cut.cpp, pair_lj_cut_coul_cut.cpp, pair_lj_cut_coul_debye.cpp and
 * their DIELECTRIC variants, DIPOLE/pair_lj_cut_dipole_cut.cpp and pair_lj_sf_dipole_sf.cpp,
 * EXTRA-PAIR/pair_nm_cut_coul_cut.cpp and pair_harmonic_cut.cpp).
 */
const CUT_TABLES: Record<string, readonly string[]> = {};
for (const st of [
  'coul/cut', 'coul/cut/dielectric', 'coul/debye', 'coul/slater/cut', 'lj/cut/coul/cut', 'lj/cut/coul/cut/dielectric',
  'lj/cut/coul/debye', 'lj/cut/coul/debye/dielectric', 'lj/cut/dipole/cut', 'lj/sf/dipole/sf', 'nm/cut/coul/cut',
]) CUT_TABLES[st] = ['cut_coul'];
CUT_TABLES['harmonic/cut'] = ['cut'];

/** A number field of the style that fix adapt sets now, or holds until the run ends (CUT_FIELDS). */
const fieldAccess = (style: Pair, field: string, timing: CutTiming): AdaptAccess => {
  const o = style as unknown as Record<string, number>;
  if (timing === 'now') return { dim: 0, get: () => o[field], set: (v) => { o[field] = v; } };
  let held: number | null = null;
  return {
    dim: 0,
    get: () => held ?? o[field],
    set: (v) => { held = v; },
    commit: () => {
      const changed = held !== null && held !== o[field];
      if (held !== null) o[field] = held;
      held = null;
      return changed;
    },
  };
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
  const field = CUT_FIELDS[styleName]?.[pparam];
  if (field && typeof (style as unknown as Record<string, unknown>)[field[0]] === 'number') return fieldAccess(style, field[0], field[1]);
  if (INERT[styleName]?.includes(pparam)) {
    const nt = style.ntypes + 1;
    const arr = new Float64Array(nt * nt);
    if (dim === 2) return { dim: 2, get: (i, j) => arr[i * nt + j], set: (i, j, x) => { arr[i * nt + j] = x; } };
  }
  // a cutoff decides the neighbor lists and the force cutoff of the run's init: only those of CUT_TABLES, whose force
  // loops keep to the cutoffs of the run's init as native's do
  if (dim === 2 && (!/cut/.test(pparam) || CUT_TABLES[styleName]?.includes(pparam))) {
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
