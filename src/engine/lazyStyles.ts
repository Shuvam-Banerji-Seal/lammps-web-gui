/*
 * Lazily loaded style registries (generated from the registry modules; tests/engineLazyStyles.test.ts checks that
 * every name here is exactly what its module registers). These modules are not imported by styles.ts: the session
 * loads a group before a command that names one of its styles (Session.dispatch -> ensureStyles), so the engine
 * worker's first chunk stays small (npm run check:size, engine budget). Add a registry module here only if no
 * statically imported engine code needs it.
 */

export type StyleKind = 'PAIRS' | 'BONDS' | 'ANGLES' | 'DIHEDRALS' | 'IMPROPERS' | 'KSPACES' | 'FIXES' | 'COMPUTES';
export type StyleModule = Partial<Record<StyleKind, Record<string, unknown>>>;

export interface LazyGroup {
  /** registry module name (src/engine/registry/<id>.ts) */
  id: string;
  names: Partial<Record<StyleKind, readonly string[]>>;
  load: () => Promise<StyleModule>;
}

export const LAZY_GROUPS: readonly LazyGroup[] = [
  { id: 'pair_meam', names: { PAIRS: ['meam'] }, load: () => import('./registry/pair_meam') as Promise<StyleModule> },
  { id: 'pair_meam_spline', names: { PAIRS: ['meam/spline', 'meam/sw/spline'] }, load: () => import('./registry/pair_meam_spline') as Promise<StyleModule> },
  { id: 'pair_snap', names: { PAIRS: ['snap'] }, load: () => import('./registry/pair_snap') as Promise<StyleModule> },
  { id: 'compute_sna', names: { COMPUTES: ['sna/atom', 'snad/atom', 'snav/atom'] }, load: () => import('./registry/compute_sna') as Promise<StyleModule> },
  { id: 'compute_snap15', names: { COMPUTES: ['gaussian/grid/local', 'sna/grid', 'sna/grid/local', 'snap'] }, load: () => import('./registry/compute_snap15') as Promise<StyleModule> },
  { id: 'mliap', names: { COMPUTES: ['mliap'], PAIRS: ['mliap'] }, load: () => import('./registry/mliap') as Promise<StyleModule> },
  { id: 'pair_granular', names: { PAIRS: ['granular'] }, load: () => import('./registry/pair_granular') as Promise<StyleModule> },
  { id: 'pair_gran', names: { PAIRS: ['gran/hertz/history', 'gran/hooke', 'gran/hooke/history'] }, load: () => import('./registry/pair_gran') as Promise<StyleModule> },
  { id: 'fix_wallgran', names: { FIXES: ['freeze', 'heat/flow', 'wall/gran', 'wall/gran/region'] }, load: () => import('./registry/fix_wallgran') as Promise<StyleModule> },
  { id: 'fix_gcmc', names: { FIXES: ['gcmc'] }, load: () => import('./registry/fix_gcmc') as Promise<StyleModule> },
  { id: 'fix_widom', names: { FIXES: ['widom'] }, load: () => import('./registry/fix_widom') as Promise<StyleModule> },
  { id: 'fix_hmc', names: { FIXES: ['hmc'] }, load: () => import('./registry/fix_hmc') as Promise<StyleModule> },
  { id: 'fix_molswap', names: { FIXES: ['mol/swap'] }, load: () => import('./registry/fix_molswap') as Promise<StyleModule> },
  { id: 'compute_voro', names: { COMPUTES: ['voronoi/atom'] }, load: () => import('./registry/compute_voro') as Promise<StyleModule> },
  { id: 'peri22', names: { COMPUTES: ['damage/atom', 'dilatation/atom'], PAIRS: ['peri/lps', 'peri/pmb'] }, load: () => import('./registry/peri22') as Promise<StyleModule> },
  { id: 'pair_tip4p', names: { PAIRS: ['lj/cut/tip4p/cut', 'lj/cut/tip4p/long', 'tip4p/cut', 'tip4p/long'] }, load: () => import('./registry/pair_tip4p') as Promise<StyleModule> },
  { id: 'kspace_tip4p', names: { KSPACES: ['pppm/tip4p', 'zero'] }, load: () => import('./registry/kspace_tip4p') as Promise<StyleModule> },
  { id: 'kspace_disp', names: { KSPACES: ['ewald/disp'] }, load: () => import('./registry/kspace_disp') as Promise<StyleModule> },
  { id: 'kspace_dipole44', names: { KSPACES: ['ewald/dipole'] }, load: () => import('./registry/kspace_dipole44') as Promise<StyleModule> },
  { id: 'pair_dipole18', names: { PAIRS: ['lj/cut/dipole/cut', 'lj/cut/dipole/long', 'lj/long/dipole/long', 'lj/sf/dipole/sf'] }, load: () => import('./registry/pair_dipole18') as Promise<StyleModule> },
  { id: 'fix_ttm', names: { FIXES: ['ttm', 'ttm/grid', 'ttm/mod', 'ttm/thermal', 'ttm/cascade'] }, load: () => import('./registry/fix_ttm') as Promise<StyleModule> },
  { id: 'compute_tally47', names: { COMPUTES: ['force/tally', 'group/group', 'pe/tally', 'stress/tally'] }, load: () => import('./registry/compute_tally47') as Promise<StyleModule> },
  { id: 'spin', names: { PAIRS: ['spin/exchange', 'spin/exchange/biquadratic', 'spin/dmi', 'spin/magelec', 'spin/neel'], COMPUTES: ['spin'], FIXES: ['nve/spin', 'precession/spin', 'langevin/spin', 'setforce/spin'] }, load: () => import('./registry/spin') as Promise<StyleModule> },
  { id: 'fix_numdiff', names: { FIXES: ['numdiff', 'numdiff/virial'] }, load: () => import('./registry/fix_numdiff') as Promise<StyleModule> },
  { id: 'fix_qeq', names: { FIXES: ['qeq/dynamic', 'qeq/fire', 'qeq/point', 'qeq/shielded', 'qeq/slater'] }, load: () => import('./registry/fix_qeq') as Promise<StyleModule> },
  { id: 'fix_cmap', names: { FIXES: ['cmap'] }, load: () => import('./registry/fix_cmap') as Promise<StyleModule> },
  { id: 'lepton', names: { ANGLES: ['lepton'], BONDS: ['lepton'], DIHEDRALS: ['lepton'], FIXES: ['efield/lepton', 'wall/lepton'], PAIRS: ['lepton', 'lepton/coul', 'lepton/sphere'] }, load: () => import('./registry/lepton') as Promise<StyleModule> },
  { id: 'pair_eim', names: { PAIRS: ['eim'] }, load: () => import('./registry/pair_eim') as Promise<StyleModule> },
  { id: 'pair_tersoff2', names: { PAIRS: ['tersoff/mod', 'tersoff/mod/c', 'tersoff/zbl'] }, load: () => import('./registry/pair_tersoff2') as Promise<StyleModule> },
  { id: 'pair_vashishta', names: { PAIRS: ['vashishta'] }, load: () => import('./registry/pair_vashishta') as Promise<StyleModule> },
  { id: 'pair_colloid', names: { PAIRS: ['colloid'] }, load: () => import('./registry/pair_colloid') as Promise<StyleModule> },
  { id: 'pair_ycolloid', names: { PAIRS: ['yukawa/colloid'] }, load: () => import('./registry/pair_ycolloid') as Promise<StyleModule> },
  { id: 'pair_relres', names: { PAIRS: ['lj/relres'] }, load: () => import('./registry/pair_relres') as Promise<StyleModule> },
  { id: 'pair_hbond', names: { PAIRS: ['hbond/dreiding/lj', 'hbond/dreiding/lj/angleoffset', 'hbond/dreiding/morse', 'hbond/dreiding/morse/angleoffset'] }, load: () => import('./registry/pair_hbond') as Promise<StyleModule> },
  { id: 'body48', names: { FIXES: ['nve/body'], PAIRS: ['body/nparticle'] }, load: () => import('./registry/body48') as Promise<StyleModule> },
  { id: 'asphere19', names: { COMPUTES: ['erotate/asphere', 'temp/asphere'], FIXES: ['nph/asphere', 'npt/asphere', 'nve/asphere', 'nve/asphere/noforce', 'nvt/asphere'] }, load: () => import('./registry/asphere19') as Promise<StyleModule> },
  { id: 'pair_asphere19', names: { PAIRS: ['gayberne', 'resquared'] }, load: () => import('./registry/pair_asphere19') as Promise<StyleModule> },
  { id: 'compute_struct', names: { COMPUTES: ['aggregate/atom', 'centro/atom', 'cluster/atom', 'cna/atom', 'fragment/atom', 'hexorder/atom'] }, load: () => import('./registry/compute_struct') as Promise<StyleModule> },
  { id: 'compute_orient', names: { COMPUTES: ['orientorder/atom'] }, load: () => import('./registry/compute_orient') as Promise<StyleModule> },
  { id: 'class2_35', names: { ANGLES: ['class2'], DIHEDRALS: ['class2'], IMPROPERS: ['class2'] }, load: () => import('./registry/class2_35') as Promise<StyleModule> },
  { id: 'fix_ehex', names: { FIXES: ['ehex', 'heat'] }, load: () => import('./registry/fix_ehex') as Promise<StyleModule> },
  { id: 'fix_pkgsmall47', names: { FIXES: ['dt/reset', 'electron/stopping', 'ave/moments'] }, load: () => import('./registry/fix_pkgsmall47') as Promise<StyleModule> },
  { id: 'fix_gk', names: { FIXES: ['ave/correlate', 'thermal/conductivity', 'viscosity'] }, load: () => import('./registry/fix_gk') as Promise<StyleModule> },
  { id: 'compute_gk', names: { COMPUTES: ['heat/flux'] }, load: () => import('./registry/compute_gk') as Promise<StyleModule> },
  { id: 'fix_boxrelax', names: { FIXES: ['box/relax'] }, load: () => import('./registry/fix_boxrelax') as Promise<StyleModule> },
  { id: 'fix_restrain', names: { FIXES: ['evaporate', 'restrain', 'spring/rg'] }, load: () => import('./registry/fix_restrain') as Promise<StyleModule> },
  { id: 'compute_restrain', names: { COMPUTES: ['msd/nongauss'] }, load: () => import('./registry/compute_restrain') as Promise<StyleModule> },
  { id: 'fix_pour', names: { FIXES: ['deposit', 'pour'] }, load: () => import('./registry/fix_pour') as Promise<StyleModule> },
  { id: 'pair_3body', names: { PAIRS: ['tersoff'] }, load: () => import('./registry/pair_3body') as Promise<StyleModule> },
  { id: 'pair_batch49', names: { PAIRS: ['buck6d/coul/gauss/dsf', 'buck6d/coul/gauss/long', 'gauss/cut', 'coul/diel', 'momb'] }, load: () => import('./registry/pair_batch49') as Promise<StyleModule> },
  { id: 'pair_potbatch48', names: { PAIRS: ['pedone', 'smatb', 'smatb/single', 'edip', 'edip/multi'] }, load: () => import('./registry/pair_potbatch48') as Promise<StyleModule> },
  { id: 'pair_sw', names: { PAIRS: ['sw', 'sw/mod'] }, load: () => import('./registry/pair_sw') as Promise<StyleModule> },
  { id: 'pair_misc4', names: { PAIRS: ['born/coul/long/cs', 'buck/coul/long/cs', 'coul/long/cs', 'lj/sf'] }, load: () => import('./registry/pair_misc4') as Promise<StyleModule> },
  { id: 'pair_misc5', names: { PAIRS: ['atm', 'nb3b/harmonic', 'nb3b/screened'] }, load: () => import('./registry/pair_misc5') as Promise<StyleModule> },
  { id: 'pair_misc13', names: { PAIRS: ['vashishta/table'] }, load: () => import('./registry/pair_misc13') as Promise<StyleModule> },
  { id: 'fix_misc13', names: { FIXES: ['ave/grid', 'controller'] }, load: () => import('./registry/fix_misc13') as Promise<StyleModule> },
  { id: 'fix_misc14', names: { FIXES: ['accelerate/cos'] }, load: () => import('./registry/fix_misc14') as Promise<StyleModule> },
  { id: 'compute_misc13', names: { COMPUTES: ['event/displace'] }, load: () => import('./registry/compute_misc13') as Promise<StyleModule> },
  { id: 'compute_misc14', names: { COMPUTES: ['temp/chunk', 'temp/cs', 'temp/ramp'] }, load: () => import('./registry/compute_misc14') as Promise<StyleModule> },
  { id: 'compute_misc14b', names: { COMPUTES: ['viscosity/cos'] }, load: () => import('./registry/compute_misc14b') as Promise<StyleModule> },
  { id: 'compute_misc15', names: { COMPUTES: ['pair'] }, load: () => import('./registry/compute_misc15') as Promise<StyleModule> },
  { id: 'compute_batch48', names: { COMPUTES: ['cnp/atom', 'hma', 'temp/profile'] }, load: () => import('./registry/compute_batch48') as Promise<StyleModule> },
  { id: 'fix_batch48', names: { FIXES: ['addtorque/group', 'ti/spring', 'tmd', 'rhok'] }, load: () => import('./registry/fix_batch48') as Promise<StyleModule> },
  { id: 'fix_hyper49', names: { FIXES: ['halt', 'hyper/global'] }, load: () => import('./registry/fix_hyper49') as Promise<StyleModule> },
  { id: 'compute_stressprof', names: { COMPUTES: ['stress/cartesian', 'stress/cylinder', 'stress/mop', 'stress/mop/profile', 'stress/spherical'] }, load: () => import('./registry/compute_stressprof') as Promise<StyleModule> },
];
