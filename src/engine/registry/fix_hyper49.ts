import type { FixFactory } from '../styles';
import { StyleError } from '../force/types';
import { FixHalt } from '../fix/halt';
import { FixHyperGlobal } from '../fix/hyper_global';

/** Fixes of the wave 49 (fix halt, fix hyper/global); lazily loaded (src/engine/lazyStyles.ts). Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  halt: (sys, id, group, args) => new FixHalt(sys, id, group, args),
  'hyper/global': (sys, id, group, args) => new FixHyperGlobal(sys, id, group, args),
  // fix_tune_kspace.html: "This fix tests each kspace style (Ewald, PPPM, and MSM), and automatically selects the
  // fastest style to use for the remainder of the run." The choice comes from timings on the host, so a run with it
  // is not reproducible; refused by name (listed in src/engine/refusedStyles.ts).
  'tune/kspace': (_sys, id) => {
    throw new StyleError(`fix ${id} tune/kspace is not supported by the browser engine: it picks the kspace style and cutoff from timed test runs on the host, so its choice depends on the machine (set kspace_style and the pair cutoff directly)`);
  },
};
