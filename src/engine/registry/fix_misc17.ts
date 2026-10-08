import type { FixFactory } from '../styles';
import { FixAddTorque, FixDrag, FixOneway } from '../fix/misc18';

/** misc fixes (wave 18, GLM worker): addtorque, drag, oneway; merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  addtorque: (sys, id, group, args) => new FixAddTorque(sys, id, group, args),
  drag: (sys, id, group, args) => new FixDrag(sys, id, group, args),
  oneway: (sys, id, group, args) => new FixOneway(sys, id, group, args),
};
