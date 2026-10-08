import type { FixFactory } from '../styles';
import { FixNVTSllod } from '../fix/nvt_sllod';

/** fix nvt/sllod (Haiku wave); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'nvt/sllod': (sys, id, group, args) => new FixNVTSllod(sys, id, group, args),
};
