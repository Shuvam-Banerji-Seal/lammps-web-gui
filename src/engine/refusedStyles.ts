import type { StyleKind } from './lazyStyles';

/*
 * Styles that are registered only so that using them raises an EngineError with a specific reason
 * (their factories throw). They are not supported, so styleNames() leaves them out of the lists the
 * notebook, the coverage table in docs/design/notebook.md and the "supported:" part of error messages
 * show. tests/engineRefusedStyles.test.ts checks that every name here is still registered.
 */
export const REFUSED_STYLES: Readonly<Partial<Record<StyleKind, readonly string[]>>> = {
  PAIRS: [
    'lj/long/dipole/long', // registry/pair_dipole18.ts
    'lj/spica/coul/msm', 'lj/sdk/coul/msm', // registry/pair_class2spica51.ts (lazy): msm kspace is not in the engine
    'spin/dipole/long', // engine-gpl/registry/spin.ts (lazy): the dipolar kspace solvers are not ported
  ],
  KSPACES: ['ewald/dipole/spin', 'pppm/dipole/spin'], // engine-gpl/registry/spin.ts (lazy): not ported
  MINIMIZERS: ['spin/lbfgs'], // engine-gpl/registry/spin.ts (lazy): min_spin_lbfgs.cpp is not ported
  FIXES: ['reaxff/bonds', 'reaxff/species', 'surface/global'], // engine-gpl/registry/reaxff.ts (stage 3, not ported); engine-gpl/registry/gransurf.ts
  COMPUTES: [
    'dipole/tip4p', 'dipole/tip4p/chunk', // registry/compute_chunk17.ts
    'stress/cylinder', 'stress/spherical', // registry/compute_stressprof.ts
    'reaxff/atom', // engine-gpl/registry/reaxff.ts (stage 3)
  ],
};

export const isRefusedStyle = (kind: StyleKind, name: string): boolean => (REFUSED_STYLES[kind] ?? []).includes(name);
