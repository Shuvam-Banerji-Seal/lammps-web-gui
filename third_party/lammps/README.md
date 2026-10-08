# LAMMPS potential files (third-party, GPL-2.0)

`potentials/` is an unmodified copy of the `potentials/` directory of the LAMMPS
distribution:

- upstream: <https://github.com/lammps/lammps>
- commit: `9792f6a9a32517780a8276c8ab201f17cae37d6b`
- 256 files, copied byte for byte (checked with `diff -rq`)

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
