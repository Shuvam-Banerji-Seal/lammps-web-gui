/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the bond_style table, angle_style
 * table, dihedral_style table and dihedral_style table/cut styles (src/MOLECULE/bond_table.cpp, angle_table.cpp,
 * dihedral_table.cpp, src/EXTRA-MOLECULE/dihedral_table_cut.cpp). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Style name -> factory. The names listed in lazyStyles.ts (group gpl_table_bonded) must match these keys.
 */
import type { Bonded } from '../../engine/force/types';
import { BondTable } from '../table_bonded/bond_table';
import { AngleTable } from '../table_bonded/angle_table';
import { DihedralTable } from '../table_bonded/dihedral_table';
import { DihedralTableCut } from '../table_bonded/dihedral_table_cut';

export const BONDS: Record<string, () => Bonded> = { table: () => new BondTable() };
export const ANGLES: Record<string, () => Bonded> = { table: () => new AngleTable() };
export const DIHEDRALS: Record<string, () => Bonded> = {
  table: () => new DihedralTable(),
  'table/cut': () => new DihedralTableCut(),
};
