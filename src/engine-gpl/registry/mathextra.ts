/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: installs the port of
 * MathExtra::richardson (src/math_extra.cpp at b332f28, ../asphere/math_extra.ts) as the orientation update of the
 * engine's rigid-body fixes (orientation table in src/engine/fix/nve_asphere.ts) when the session loads this module
 * (lazy group gpl_mathextra, HOOKS word mathextra, loaded by Session.execute).
 */
import { orientation } from '../../engine/fix/nve_asphere';
import { richardson } from '../asphere/math_extra';

orientation.richardson = richardson;

export const HOOKS = { mathextra: { richardson } };
