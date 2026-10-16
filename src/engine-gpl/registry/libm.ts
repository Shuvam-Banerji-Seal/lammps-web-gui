/*
 * Support code for the styles ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: installs
 * the bit-exact glibc log, sin and cos of ../libm (see their headers) in the engine's libm table (src/engine/libm.ts)
 * when the session loads this module (lazy group gpl_libm, HOOKS word libm, loaded by Session.execute).
 */
import { libm } from '../../engine/libm';
import { log } from '../libm/log';
import { cos, sin } from '../libm/sincos';

libm.log = log;
libm.sin = sin;
libm.cos = cos;

export const HOOKS = { libm: { log, sin, cos } };
