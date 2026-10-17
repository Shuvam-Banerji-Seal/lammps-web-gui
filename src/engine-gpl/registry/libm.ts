/*
 * Support code for the styles ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: installs
 * the bit-exact glibc exp, log, sin, cos, erf and erfc of ../libm (see their headers) in the engine's libm table (src/engine/libm.ts)
 * when the session loads this module (lazy group gpl_libm, HOOKS word libm, loaded by Session.execute).
 */
import { libm } from '../../engine/libm';
import { exp } from '../libm/exp';
import { log } from '../libm/log';
import { cos, sin } from '../libm/sincos';
import { erf, erfc } from '../libm/erf';

libm.exp = exp;
libm.log = log;
libm.sin = sin;
libm.cos = cos;
libm.erf = erf;
libm.erfc = erfc;

export const HOOKS = { libm: { exp, log, sin, cos, erf, erfc } };
