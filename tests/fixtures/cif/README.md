# Real CIF fixtures

Downloaded unmodified from the Crystallography Open Database
(<https://www.crystallography.net/cod/>), whose data is dedicated to the public
domain under CC0. They are real database entries, so — like almost every CIF
in the wild — they list only the **asymmetric unit** plus the symmetry
operations needed to generate the full cell.

| File | Compound | Space group | Sym. ops | Sites in file | Atoms in conventional cell |
|---|---|---|---|---|---|
| `cod-1000041.cif` | NaCl (rocksalt) | F m -3 m | 192 (`_symmetry_equiv_pos_as_xyz`) | Na 4a, Cl 4b | **8** |
| `cod-1011176.cif` | SiO2 (quartz) | P 32 2 1 | 6 (`_symmetry_equiv_pos_as_xyz`) | Si 3a, O 6c | **9** |
| `cod-9008800.cif` | AgZn (CsCl type) | P m -3 m | 48 (`_space_group_symop_operation_xyz`) | Ag 1a, Zn 1b | **2** |

The expected counts are the sums of the Wyckoff multiplicities. AgZn is the
deduplication control: all 48 operations map each site onto itself, so a
correct expansion must still produce exactly 2 atoms.
