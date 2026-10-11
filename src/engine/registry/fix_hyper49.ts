import type { FixFactory } from '../styles';
import { FixHalt } from '../fix/halt';
import { FixHyperGlobal } from '../fix/hyper_global';

/** Fixes of the wave 49 (fix halt, fix hyper/global); lazily loaded (src/engine/lazyStyles.ts). Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  halt: (sys, id, group, args) => new FixHalt(sys, id, group, args),
  'hyper/global': (sys, id, group, args) => new FixHyperGlobal(sys, id, group, args),
};
