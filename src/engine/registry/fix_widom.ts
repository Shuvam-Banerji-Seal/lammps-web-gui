import type { FixFactory } from '../styles';
import { FixWidom } from '../fix/widom';

/** fix widom (wave 30); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  widom: (sys, id, group, args) => new FixWidom(sys, id, group, args),
};
