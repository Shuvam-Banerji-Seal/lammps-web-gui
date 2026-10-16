/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of src/SPIN/neb_spin.h (CommandStyle(neb/spin,NEBSpin)) and src/SPIN/fix_neb_spin.h
 * (FixStyle(neb/spin,FixNEBSpin)) at commit b332f28; the classes are in ../spin/ (see each file's header).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_neb_spin); the core does not register these names.
 */
import type { Handler } from '../../engine/commands/args';
import type { FixFactory } from '../../engine/styles';
import { FixNEBSpin } from '../spin/fix_neb_spin';
import { runNebSpin } from '../spin/neb_spin';

export const FIXES: Record<string, FixFactory> = {
  'neb/spin': (sys, id, group, args) => new FixNEBSpin(sys, id, group, args),
};

export const COMMANDS: Record<string, Handler> = {
  'neb/spin': ({ session, run }, a) => runNebSpin(session, run, a),
};
