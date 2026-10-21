import type { FixFactory } from '../styles';
import { FixDrag, FixOneway } from '../fix/misc18';
import { FixAddTorqueGroup } from '../fix/addtorque_group';

/** misc fixes (wave 18, GLM worker): addtorque, drag, oneway; merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  // native registers addtorque and addtorque/group as one class (src/EXTRA-FIX/fix_addtorque_group.h at b332f28)
  addtorque: (sys, id, group, args) => new FixAddTorqueGroup(sys, id, group, args, 'addtorque'),
  drag: (sys, id, group, args) => new FixDrag(sys, id, group, args),
  oneway: (sys, id, group, args) => new FixOneway(sys, id, group, args),
};
