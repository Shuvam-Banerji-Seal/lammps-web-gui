/*
 * The bonded parameters fix adapt can name, as native LAMMPS does (src/fix_adapt.cpp FixAdapt::init at b332f28): the
 * attribute's style is found with Force::bond_match (angle_match, ...), a hybrid's sub-style included, and its bparam
 * passed to the style's extract(), which returns a per-type vector (dim 1) or nothing ("Fix adapt bond style parameter
 * X not supported"). BONDED_EXTRACT lists, for every bond, angle, dihedral and improper style of native that has an
 * extract(), the names it accepts, read from the extract() of the file named above each entry (src/ at b332f28; a
 * style without its own extract() inherits its parent class's).
 */

/** "<kind> <style>" -> native extract() names. Plain data (no imports): scripts/parity/run-parity.mjs reads it. */
export const BONDED_EXTRACT: Record<string, readonly string[]> = {
  // AMOEBA/angle_amoeba.cpp
  'angle amoeba': ['k2', 'k3', 'k4', 'k5', 'k6', 'theta0'],
  // MOLECULE/angle_charmm.cpp
  'angle charmm': ['k', 'theta0'],
  // CLASS2/angle_class2.cpp
  'angle class2': ['k2', 'k3', 'k4', 'theta0'],
  // MOFFF/angle_class2_p6.cpp
  'angle class2/p6': ['k2', 'k3', 'k4', 'k5', 'k6', 'theta0'],
  // CLASS2/angle_class2xe.cpp
  'angle class2xe': ['k2', 'k3', 'k4', 'theta0'],
  // MOLECULE/angle_cosine.cpp
  'angle cosine': ['k'],
  // MOFFF/angle_cosine_buck6d.cpp
  'angle cosine/buck6d': ['k', 'multiplicity', 'th0'],
  // EXTRA-MOLECULE/angle_cosine_delta.h (class AngleCosineDelta : AngleCosineSquared, whose extract() it keeps)
  'angle cosine/delta': ['k', 'theta0'],
  // EXTRA-MOLECULE/angle_cosine_periodic.cpp
  'angle cosine/periodic': ['k', 'b', 'multiplicity'],
  // MOLECULE/angle_cosine_squared.cpp
  'angle cosine/squared': ['k', 'theta0'],
  // EXTRA-MOLECULE/angle_cosine_squared_restricted.cpp
  'angle cosine/squared/restricted': ['k', 'theta0'],
  // YAFF/angle_cross.cpp
  'angle cross': ['r00', 'r01', 'kss', 'kbs0', 'kbs1', 'theta0'],
  // DIPOLE/angle_dipole.cpp
  'angle dipole': ['k', 'gamma0'],
  // EXTRA-MOLECULE/angle_fourier.cpp
  'angle fourier': ['k', 'C0', 'C1', 'C2'],
  // EXTRA-MOLECULE/angle_fourier_simple.cpp
  'angle fourier/simple': ['k', 'C', 'N'],
  // EXTRA-MOLECULE/angle_gaussian.cpp
  'angle gaussian': ['alpha', 'width', 'theta0'],
  // MOLECULE/angle_harmonic.cpp
  'angle harmonic': ['k', 'theta0'],
  // LEPTON/angle_lepton.cpp
  'angle lepton': ['theta0'],
  // MESONT/angle_mesocnt.cpp
  'angle mesocnt': ['theta0'],
  // YAFF/angle_mm3.cpp
  'angle mm3': ['k2', 'theta0'],
  // EXTRA-MOLECULE/angle_mwlc.cpp
  'angle mwlc': ['k1', 'k2', 'mu', 'temp'],
  // EXTRA-MOLECULE/angle_quartic.cpp
  'angle quartic': ['k2', 'k3', 'k4', 'theta0'],
  // CG-SPICA/angle_spica.cpp
  'angle sdk': ['k', 'theta0'],
  // CG-SPICA/angle_spica.cpp
  'angle spica': ['k', 'theta0'],
  // CLASS2/bond_class2.cpp
  'bond class2': ['k2', 'k3', 'k4', 'r0'],
  // MOLECULE/bond_fene.cpp
  'bond fene': ['k', 'r0'],
  // MOLECULE/bond_fene_expand.cpp
  'bond fene/expand': ['k', 'r0', 'epsilon', 'sigma', 'shift'],
  // EXTRA-MOLECULE/bond_fene_nm.cpp
  'bond fene/nm': ['k', 'r0'],
  // EXTRA-MOLECULE/bond_gaussian.cpp
  'bond gaussian': ['alpha', 'width', 'r0'],
  // MOLECULE/bond_gromos.cpp
  'bond gromos': ['k', 'r0'],
  // MOLECULE/bond_harmonic.cpp
  'bond harmonic': ['k', 'r0'],
  // EXTRA-MOLECULE/bond_harmonic_restrain.cpp
  'bond harmonic/restrain': ['k'],
  // EXTRA-MOLECULE/bond_harmonic_shift.cpp
  'bond harmonic/shift': ['k', 'r0', 'r1'],
  // EXTRA-MOLECULE/bond_harmonic_shift_cut.cpp
  'bond harmonic/shift/cut': ['k', 'r0', 'r1'],
  // LEPTON/bond_lepton.cpp
  'bond lepton': ['r0'],
  // MESONT/bond_mesocnt.h (class BondMesoCNT : BondHarmonic, whose extract() it keeps)
  'bond mesocnt': ['k', 'r0'],
  // YAFF/bond_mm3.cpp
  'bond mm3': ['k2', 'r0'],
  // MOLECULE/bond_morse.cpp
  'bond morse': ['d0', 'alpha', 'r0'],
  // EXTRA-MOLECULE/bond_nonlinear.cpp
  'bond nonlinear': ['lamda', 'epsilon', 'r0'],
  // EXTRA-MOLECULE/bond_quartic_exp.cpp
  'bond quartic/exp': ['r0', 'k2', 'k3', 'k4', 'A', 'B'],
  // bond_zero.cpp
  'bond zero': ['r0'],
  // MOLECULE/dihedral_charmm.cpp
  'dihedral charmm': ['k', 'n', 'd'],
  // MOLECULE/dihedral_charmmfsw.cpp
  'dihedral charmmfsw': ['k', 'n', 'd'],
  // CLASS2/dihedral_class2.cpp
  'dihedral class2': ['k1', 'k2', 'k3', 'phi1', 'phi2', 'phi3'],
  // CLASS2/dihedral_class2xe.cpp
  'dihedral class2xe': ['k1', 'k2', 'k3', 'phi1', 'phi2', 'phi3'],
  // EXTRA-MOLECULE/dihedral_cosine_squared_restricted.cpp
  'dihedral cosine/squared/restricted': ['k', 'phi0'],
  // EXTRA-MOLECULE/dihedral_helix.cpp
  'dihedral helix': ['a', 'b', 'c'],
  // MOLECULE/dihedral_multi_harmonic.cpp
  'dihedral multi/harmonic': ['a1', 'a2', 'a3', 'a4', 'a5'],
  // MOLECULE/dihedral_opls.cpp
  'dihedral opls': ['k1', 'k2', 'k3', 'k4'],
  // EXTRA-MOLECULE/dihedral_quadratic.cpp
  'dihedral quadratic': ['k', 'phi0'],
  // AMOEBA/improper_amoeba.cpp
  'improper amoeba': ['k'],
  // CLASS2/improper_class2.cpp
  'improper class2': ['k', 'chi0'],
  // EXTRA-MOLECULE/improper_cossq.cpp
  'improper cossq': ['k', 'chi0'],
  // MOLECULE/improper_cvff.cpp
  'improper cvff': ['k', 'd', 'n'],
  // EXTRA-MOLECULE/improper_distance.cpp
  'improper distance': ['k2', 'k4'],
  // YAFF/improper_distharm.cpp
  'improper distharm': ['k', 'd0'],
  // EXTRA-MOLECULE/improper_fourier.cpp
  'improper fourier': ['k', 'C0', 'C1', 'C2'],
  // MOLECULE/improper_harmonic.cpp
  'improper harmonic': ['k', 'chi0'],
  // MOFFF/improper_inversion_harmonic.cpp
  'improper inversion/harmonic': ['k', 'w0'],
  // EXTRA-MOLECULE/improper_ring.cpp
  'improper ring': ['k', 'theta0'],
  // YAFF/improper_sqdistharm.cpp
  'improper sqdistharm': ['k'],
  // MOLECULE/improper_umbrella.cpp
  'improper umbrella': ['k', 'w0'],
};
