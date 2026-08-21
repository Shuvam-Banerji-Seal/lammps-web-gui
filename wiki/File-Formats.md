# File Formats

Molecule3D auto-detects formats by extension **and** content sniffing, so
`.txt` files and pasted data usually work too.

---

## LAMMPS data files (`.data`, `.lmp`, `.lammps`)

Supported atom styles for the `Atoms` section (declared via `Atoms # style`,
or auto-detected per row):

| Style | Columns |
|---|---|
| `atomic` | `id type x y z` |
| `charge` | `id type q x y z` |
| `molecular` | `id molecule type x y z` |
| `full` | `id molecule type q x y z` |

Also parsed:

- **Box bounds**: `-10.0 4.9 xlo xhi` lines → simulation box rendering (`X`)
- **Triclinic tilt**: `0.0 1.0 0.5 xy xz yz` → true parallelepiped rendering
- **Masses section**: element identification by comment symbol ("C"),
  element name ("Carbon"), or nearest IUPAC mass (±0.5 u); type-id-as-atomic-number fallback
- Sections other than Masses/Atoms/Bonds (Velocities, coefficients…) are skipped

```text
# Example header
60 atoms
90 bonds
-10.0 4.9 xlo xhi          <- box bounds
Masses
1 12.011 # Carbon           <- element from comment/mass
Atoms # full                <- style declaration
   1 1 6 0.0  -1.18 -5.22 0.52
Bonds
   1 1 1 2
```

## XYZ (`.xyz`)

Standard two-line-header format. The **first frame** of a trajectory is shown.
Bonds are inferred automatically using an O(n) spatial hash over Cordero
covalent radii (tolerance ×1.2) — H₂ at 0.74 Å bonds, non-bonded contacts at
3 Å don't. Extended-XYZ extra columns are tolerated.

## PDB (`.pdb`, `.ent`)

- `ATOM` / `HETATM` coordinates
- Element resolution: columns 77–78 first, then the atom-name heuristic that
  respects the right-justified two-letter convention (`" CL "` is chlorine,
  `" CA "` starting at column 13 is alpha-carbon)
- `CONECT` bonds (deduplicated)
- `CRYST1` unit cell → box rendering

## CIF (`.cif`, `.mmcif`)

- `_cell_length_*` / `_cell_angle_*` → cell vectors
- `_atom_site` loop with fractional (`fract_x/y/z`) or Cartesian (`Cartn_x/y/z`) coords
- Element from `_atom_site_type_symbol` (charges like `Fe3+` stripped) or the
  alphabetic prefix of `_atom_site_label` (`Cl2` → Cl)
- Triclinic cells render as true parallelepipeds via LAMMPS-convention tilt factors
- First `data_` block only; symmetry expansion is **not** performed — use P1 files

## Element colors across all formats

Every parser maps atoms to atomic numbers, which index the built-in
CPK/Jmol palette for all 118 elements. Unknown types render magenta.
