import type { StyleKind } from './lazyStyles';

/*
 * Styles that are registered only so that using them raises an EngineError with a specific reason
 * (their factories throw). They are not supported, so styleNames() leaves them out of the lists the
 * notebook, the coverage table in docs/design/notebook.md and the "supported:" part of error messages
 * show. tests/engineRefusedStyles.test.ts checks that every name here is still registered.
 */
export const REFUSED_STYLES: Readonly<Partial<Record<StyleKind, readonly string[]>>> = {
  PAIRS: [
    'pace/extrapolation', // engine-gpl/registry/pace.ts: ACE extrapolation grades are not ported yet
  ],
  FIXES: [
    'tune/kspace', // registry/fix_hyper49.ts: its choice comes from timed test runs on the host
  ],
  COMPUTES: [
  ],
  KSPACES: [
  ],
};

export const isRefusedStyle = (kind: StyleKind, name: string): boolean => (REFUSED_STYLES[kind] ?? []).includes(name);
