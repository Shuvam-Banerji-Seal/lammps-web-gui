import type { FixFactory } from '../styles';
import { FixQeq } from '../fix/qeq';

/**
 * fix qeq (wave 32); merged into styles.ts.
 *
 * docs.lammps.org/fix_qeq.html: style = *qeq/point* or *qeq/shielded* or
 * *qeq/slater* or *qeq/ctip* or *qeq/dynamic* or *qeq/fire*. Only qeq/point and
 * qeq/shielded are implemented here; the other styles are left unregistered so
 * the fix command's generic error names them (the task says qeq/dynamic,
 * qeq/fire, qeq/slater and qeq/reaxff must be StyleErrors naming them).
 */
export const FIXES: Record<string, FixFactory> = {
  'qeq/point': (sys, id, group, args) => new FixQeq(sys, id, group, args, false),
  'qeq/shielded': (sys, id, group, args) => new FixQeq(sys, id, group, args, true),
};
