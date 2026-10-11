# LAMMPS potential files (third-party, GPL-2.0)

`potentials/` is an unmodified copy of the `potentials/` directory of the LAMMPS
distribution:

- upstream: <https://github.com/lammps/lammps>
- commit: `b332f28` (the commit of the native binary used for the parity cases, `lmp -h`: "Git info (develop / b332f28)");
  every file of that commit's `potentials/` (274 files) is here, copied byte for byte (checked with `cmp`)
- nine further files from the earlier copy (commit `9792f6a9a32517780a8276c8ab201f17cae37d6b`) that the later commit
  no longer ships: `Al_Batra_2019.agni`, `Al_jpc.agni`, `Al_prb.agni`, `Cu_Huan_2019_fp1.agni`, `Cu_Huan_2019_fp2.agni`,
  `Pt_Chapman_2020.agni`, `MgAl.rann`, `Ti.rann`, `TiZr_2.rann`, `Zn.rann`, `Zr.rann` (kept unmodified)
- one further file from the `examples/fire/` directory of the same distribution, copied byte for byte:
  `FeTiClibrary.meam` (the MEAM library file of the `in.meam.fire` example; it is not in `potentials/`). It is read
  by the native-parity case `tests/oracle/w47mn_ti_crystal` and `tests/engineMeamNn2Hcp47.test.ts`.
- added for the SPIN oracle cases: `potentials/Fe_Mishin2006.eam.alloy`, copied byte for byte from `examples/SPIN/iron/` of the same LAMMPS tree (`cmp` checked)

These files are part of LAMMPS and are distributed under the **GNU General
Public License, version 2** (see `LICENSE` in this directory). Many of them
also carry their own author and citation headers, which are kept as they are.
Cite the original authors when you use a potential, as each file's header
asks.

They are **not** covered by this project's licence (see the repository's
`LICENSE`, `NOTICE` and `COMMERCIAL.md`): they are separate data files kept
side by side with the project and read at run time by the test suite (as
inputs of native-parity cases, see `tests/oracle/`). No LAMMPS source code is part of this project; the
in-browser engine is written from the documentation at docs.lammps.org and
textbook physics (AGENTS.md, rule 7).
