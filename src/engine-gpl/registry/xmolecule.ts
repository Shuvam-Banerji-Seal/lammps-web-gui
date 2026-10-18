/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-MOLECULE/bond_fene_nm.cpp, bond_harmonic_restrain.cpp, bond_quartic_exp.cpp,
 * angle_cosine_shift_exp.cpp, angle_mwlc.cpp, dihedral_spherical.cpp and improper_ring.cpp at commit b332f28 (the
 * LAMMPS developers; Copyright (2003) Sandia Corporation, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * EXTRA-MOLECULE bonded styles, registered for the core through src/engine/lazyStyles.ts (group gpl_xmolecule; the
 * names listed there must match these keys).
 */
import type { Bonded } from '../../engine/force/types';
import { BondFeneNm } from '../xmolecule/bond_fene_nm';
import { BondHarmonicRestrain } from '../xmolecule/bond_harmonic_restrain';
import { BondQuarticExp } from '../xmolecule/bond_quartic_exp';
import { AngleCosineShiftExp } from '../xmolecule/angle_cosine_shift_exp';
import { AngleMWLC } from '../xmolecule/angle_mwlc';
import { DihedralSpherical } from '../xmolecule/dihedral_spherical';
import { ImproperRing } from '../xmolecule/improper_ring';

export const BONDS: Record<string, () => Bonded> = {
  'fene/nm': () => new BondFeneNm(),
  'harmonic/restrain': () => new BondHarmonicRestrain(),
  'quartic/exp': () => new BondQuarticExp(),
};
export const ANGLES: Record<string, () => Bonded> = {
  'cosine/shift/exp': () => new AngleCosineShiftExp(),
  mwlc: () => new AngleMWLC(),
};
export const DIHEDRALS: Record<string, () => Bonded> = {
  spherical: () => new DihedralSpherical(),
};
export const IMPROPERS: Record<string, () => Bonded> = {
  ring: () => new ImproperRing(),
};
