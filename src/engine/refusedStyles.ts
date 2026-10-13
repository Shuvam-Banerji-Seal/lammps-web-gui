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
    'smd/tri_surface', // engine-gpl/registry/machdyn.ts (lazy): the wall contact is not matched to native (contact order)
  ],
  FIXES: [
    'filter/corotate', // engine-gpl/registry/filtercorotate.ts (refused: respa fix hooks missing)
    'surface/global', // engine-gpl/registry/gransurf.ts
    'wall/piston', 'append/atoms', // engine-gpl/registry/shock.ts (SHOCK wall/piston and append/atoms are not ported)
  ],
  COMPUTES: [
    'dipole/tip4p', 'dipole/tip4p/chunk', // registry/compute_chunk17.ts
    'stress/cylinder', 'stress/spherical', // registry/compute_stressprof.ts
    'reaxff/atom', // engine-gpl/registry/reaxff.ts (stage 3)
  ],
};

export const isRefusedStyle = (kind: StyleKind, name: string): boolean => (REFUSED_STYLES[kind] ?? []).includes(name);
