/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the per-atom field that the dipolar kspace
 * solvers of src/SPIN/ and src/KSPACE/ (ewald_dipole_spin.cpp, pppm_dipole_spin.cpp) store in atom->fm_long, read by
 * pair_spin_dipole_long.cpp (compute_single_pair adds fm_long[i] to the precession vector of atom i). LAMMPS is
 * GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The buffer is kept per simulation state and indexed by atom id (the atoms are permuted by the sorting), zeroed at
 * every kspace evaluation (native force_clear zeroes fm_long) and read by the pair's precession vector.
 */
import type { SimState } from '../../engine/types';

const buffers = new WeakMap<SimState, Float64Array>();

/** The fm_long buffer of a state, sized for its largest atom id (3 values per id), zero-filled when reset. */
export const fmLongOf = (s: SimState, reset: boolean): Float64Array => {
  let maxId = 0;
  for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
  const need = 3 * (maxId + 1);
  let buf = buffers.get(s);
  if (!buf || buf.length < need) {
    buf = new Float64Array(need);
    buffers.set(s, buf);
  } else if (reset) buf.fill(0);
  return buf;
};
