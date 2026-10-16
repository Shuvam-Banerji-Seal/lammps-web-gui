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

## Third-party code: voro++ (BSD-3-Clause)

`src/engine-gpl/voro/` also holds a port of the subset of **voro++ 0.4.6** (<http://math.lbl.gov/voro++/>) that LAMMPS
compiles into compute voronoi/atom: the cell construction (`cell.ts`, `container.ts`, `worklist.ts`), as bundled with
LAMMPS at commit `b332f28` (`lib/voronoi`, the LAMMPS build of voro++ 0.4.6, with its rad_option.hh patch).
voro++ is distributed under the **BSD three-clause licence**, copyright The Regents of the University of California
(through Lawrence Berkeley National Laboratory, 2008) and Chris H. Rycroft (2008-2011). Its licence text is kept
unchanged in `voro/LICENSE.voro++`; each ported file names the voro++ source it follows and carries the copyright
notice. The BSD licence is compatible with this directory's GPL-2.0 licence, and the code here stays under GPL-2.0 as
part of the LAMMPS port.

## Third-party code: PTM (MIT) and its voro++ cell

- `ptm/` (compute ptm/atom): the Polyhedral Template Matching library is MIT-licensed (Copyright (c) 2016 PM Larsen);
  its licence text is in `ptm/LICENSE.ptm`.
- `ptm/ptm_voronoi_cell.ts` and `ptm/ptm_voronoi_config.ts`: the voro++ (BSD-3, The Regents of the University of California,
  Copyright (c) 2008) Voronoi cell bundled in the PTM library, ported with the owner's approval (2026-10-16); the licence
  text is in `ptm/LICENSE.voro++`.
