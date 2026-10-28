import type { FixFactory } from '../styles';
import { FixExternal } from '../fix/external';

/** fix external (library-interface forces; no caller in the notebook); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  external: (sys, id, group, args) => new FixExternal(sys, id, group, args),
};
