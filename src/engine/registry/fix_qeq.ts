import type { FixFactory } from '../styles';
import { FixQeq } from '../fix/qeq';

/**
 * fix qeq (wave 32, extended wave 36); merged into styles.ts.
 *
 * docs.lammps.org/fix_qeq.html: style = *qeq/point* or *qeq/shielded* or
 * *qeq/slater* or *qeq/ctip* or *qeq/dynamic* or *qeq/fire*. qeq/point,
 * qeq/shielded, qeq/dynamic and qeq/fire are implemented here; qeq/slater and
 * qeq/ctip are left unregistered so the fix command's generic error names them.
 */
export const FIXES: Record<string, FixFactory> = {
  'qeq/point': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'point'),
  'qeq/shielded': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'shielded'),
  'qeq/dynamic': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'dynamic'),
  'qeq/fire': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'fire'),
};
