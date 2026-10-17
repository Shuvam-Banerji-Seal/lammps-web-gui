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
    'srp/react', // engine-gpl/registry/srp.ts (lazy): srp/react with bond/break and bond/create (fix_srp_react.cpp), not ported
  ],
  FIXES: [
    'filter/corotate', // engine-gpl/registry/filtercorotate.ts (refused: respa fix hooks missing)
    'srp/react', // engine-gpl/registry/srp.ts (lazy, refused: see its header)
    'graphics/isosurface', // engine-gpl/registry/graphics.ts (refused: the reason is in the error)
  ],
  COMPUTES: [
    'dipole/tip4p', 'dipole/tip4p/chunk', // registry/compute_chunk17.ts
  ],
};

export const isRefusedStyle = (kind: StyleKind, name: string): boolean => (REFUSED_STYLES[kind] ?? []).includes(name);
