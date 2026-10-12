/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/fix_qeq_reaxff.cpp (init: the fix efield checks; get_chi_field: the field term
 * chi_field of the charge equilibration, on the unwrapped positions), src/REAXFF/fix_qtpie_reaxff.cpp
 * (init: the fix efield checks; calc_chi_eff: the field potential phi = -(x . E)/qe2f on the wrapped
 * positions), and src/REAXFF/fix_qeq_rel_reaxff.cpp (calc_chi_eff with the field), at commit b332f28.
 * Authors credited in the C++ file headers: Hasan Metin Aktulga (Purdue University, now Lawrence Berkeley
 * National Laboratory); Ray Shan (Sandia); Efstratios M Kritikos (Caltech); Navraj S Lalli (Imperial
 * College London). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Not ported (refused by name): the region keyword of fix efield, atom-style field variables (the
 * per-atom potential), and a field in a triclinic box (the unwrapped position needs the tilt factors).
 */
import { FixEfield } from '../../engine/fix/force_ext';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

/** The SMALL of fix_qeq_reaxff.cpp (the periodic-direction check). */
const SMALL = 1.0e-14;

/** The one fix efield of the system, or null; LAMMPS allows at most one. */
export const findEfield = (sys: System, label: string): FixEfield | null => {
  const found = sys.fixes.filter((f): f is FixEfield => f instanceof FixEfield);
  if (found.length > 1) throw new StyleError(`There may be only one fix efield instance used with fix ${label}`);
  return found.length === 1 ? found[0] : null;
};

/**
 * The checks of the init of each charge model (the errors of the C++ init) and the field components of
 * the current step (equal-style variables are evaluated here, as update_efield_variables does).
 */
export const efieldForModel = (
  sys: System, fix: FixEfield, label: string, opts: { allGroup: boolean; periodicCheck: boolean },
): [number, number, number] => {
  if (sys.state.units.style !== 'real') throw new StyleError(`Must use unit_style real with fix ${label} and external fields`);
  const st = fix.fieldState();
  if (st.region !== null) throw new StyleError(`Keyword region not supported for fix efield when using fix ${label}`);
  if (st.comp.some((c) => c.atomStyle)) {
    throw new StyleError(`Atom-style external electric field is not supported by fix ${label} in the browser engine (the per-atom potential term is not ported)`);
  }
  if (opts.allGroup && fix.group !== 'all') throw new StyleError(`Must use group id all for fix efield when using fix ${label}`);
  const E = fieldOf(sys, fix);
  if (opts.periodicCheck) {
    // fix_qeq_reaxff.cpp: no field component along a periodic direction (constant zero, or any variable), tested on
    // the stored components (qe2f times the input value, as fix_efield.cpp keeps them)
    const per = sys.state.box.periodic;
    for (let d = 0; d < 3; d++) {
      const c = st.comp[d];
      if (per[d] && (!c.constant || Math.abs(E[d]) > SMALL)) {
        throw new StyleError(`Must not have electric field component in direction of periodic boundary when using charge equilibration with ReaxFF.`);
      }
    }
  }
  return E;
};

/**
 * The field components (ex, ey, ez) of the fix for the current step, stored as fix_efield.cpp stores them:
 * qe2f times the input value (also for equal-style variables). The charge models divide by qe2f again, so
 * the conversion cancels in their field terms. No atom-style component (refused by efieldForModel).
 */
export const fieldOf = (sys: System, fix: FixEfield): [number, number, number] => {
  const st = fix.fieldState();
  const qe2f = sys.state.units.qe2f;
  return [qe2f * (st.comp[0].value as number), qe2f * (st.comp[1].value as number), qe2f * (st.comp[2].value as number)];
};

/**
 * chi_field of fix_qeq_reaxff.cpp get_chi_field: -(E . unwrapped x)/qe2f for the atoms of the fix's
 * group, zero for the others. The unwrapped position is x + image * box length (orthogonal boxes).
 */
export const qeqChiField = (sys: System, fix: FixEfield, E: [number, number, number]): Float64Array => {
  const s = sys.state;
  const n = s.n;
  const out = new Float64Array(n);
  const box = s.box;
  if (box.triclinic) throw new StyleError(`fix ${fix.style} with a field needs an orthogonal box in the browser engine (triclinic unmap is not ported)`);
  const prd = [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]];
  const factor = -1.0 / s.units.qe2f;
  const u = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    if (!(s.mask[i] & fix.groupBit)) continue;
    for (let d = 0; d < 3; d++) u[d] = s.x[3 * i + d] + s.image[3 * i + d] * prd[d];
    out[i] = factor * (E[0] * u[0] + E[1] * u[1] + E[2] * u[2]);
  }
  return out;
};

/** The field potential phi_i = -(x_i . E)/qe2f of the qtpie family (fix_qtpie_reaxff.cpp calc_chi_eff), on the given positions. */
export const fieldPotential = (x: Float64Array, a: number, E: [number, number, number], qe2f: number): number => {
  const factor = 1.0 / qe2f;
  return -factor * (x[3 * a] * E[0] + x[3 * a + 1] * E[1] + x[3 * a + 2] * E[2]);
};
