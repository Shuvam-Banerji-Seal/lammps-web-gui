import type { FixFactory } from '../styles';
import { FixAddTorqueGroup } from '../fix/addtorque_group';
import { FixTiSpring } from '../fix/ti_spring';
import { FixTmd } from '../fix/tmd';
import { FixRhok } from '../fix/rhok';

/*
 * Fix styles of the EXTRA-FIX / PACKAGES examples (addtorque/group, ti/spring, tmd, rhok). Style name -> factory.
 */
export const FIXES: Record<string, FixFactory> = {
  'addtorque/group': (sys, id, group, args) => new FixAddTorqueGroup(sys, id, group, args),
  'ti/spring': (sys, id, group, args) => new FixTiSpring(sys, id, group, args),
  tmd: (sys, id, group, args) => new FixTmd(sys, id, group, args),
  rhok: (sys, id, group, args) => new FixRhok(sys, id, group, args),
};
