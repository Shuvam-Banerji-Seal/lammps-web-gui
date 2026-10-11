# `src/engine-gpl/` — styles ported from LAMMPS (GPL-2.0)

The code in this directory is ported or translated from the LAMMPS C++ source
(<https://github.com/lammps/lammps>, commit `b332f28`, the commit of the native
binary the parity tests compare against). LAMMPS is distributed under the
**GNU General Public License, version 2**, so this directory is too: see
`LICENSE` here. It is **not** covered by the project's own licence (the root
`LICENSE`, `NOTICE` and `COMMERCIAL.md`), which applies to the rest of the
repository.

Rules (AGENTS.md, rule 7):

- Every source file here starts with a header naming the LAMMPS source
  file(s) it was ported from, the commit (`b332f28`), and that it is GPL-2.0.
  Keep the original authors' names from those files.
- The rest of `src/` stays independently written. The engine core reaches this
  directory only through the lazily loaded style registries listed in
  `src/engine/lazyStyles.ts` (a `registry/*.ts` module here exports `PAIRS`,
  `FIXES`, ... like the core registries). `tests/engineGplBoundary.test.ts`
  checks the import boundary and the headers.
- A ported style still needs native-parity oracle cases (`tests/oracle/`) before
  it is registered.

The built app loads these modules as separate engine chunks, on first use of a
style they provide.
