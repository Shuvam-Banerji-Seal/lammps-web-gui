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

## LAMMPS packages with their own notice: MGPT (LLNL)

`src/engine-gpl/mgpt/` (pair_style mgpt) is ported from the MGPT package of LAMMPS (`src/MGPT`), written by Tomas
Oppelstrup and John Moriarty at Lawrence Livermore National Laboratory. It is Copyright (2015) Lawrence Livermore
National Security, LLC (LLNL-CODE-674031) and distributed under the GNU General Public License version 2, like the
rest of this directory. Its LLNL preamble notice, which the authors state "is required to be provided", is kept in
`mgpt/NOTICE.llnl`, and each ported file names it.

## Third-party code: ch5md (BSD-3-Clause)

`src/engine-gpl/h5md/` (dump h5md) follows the ch5md library that LAMMPS bundles for its H5MD package
(`lib/h5md/src/ch5md.c` at commit `b332f28`, Copyright (C) 2013-2014 Pierre de Buyl). ch5md is distributed under a
**BSD three-clause licence**, kept unchanged in `h5md/LICENSE.ch5md`; the ported files name it. The HDF5 files that
ch5md writes through the HDF5 C library are produced by `h5md/hdf5.ts`, written from the HDF5 File Format
Specification. The code here stays under GPL-2.0 as part of the LAMMPS port.

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

## Third-party code: the C and C++ runtime libraries

Native parity sometimes needs code that LAMMPS itself does not contain but calls or links. On 2026-10-16 the owner
decided that such code may be ported here when the port reproduces the native binary exactly. Each file names its
source and licence in its header:

- `libm/log.ts`: the `log()` of the C library the native binary runs with (Ubuntu glibc 2.35,
  `__ieee754_log_fma` from `sysdeps/ieee754/dbl-64/e_log.c` and `e_log_data.c`), bit for bit, with an emulated
  correctly rounded `fma`. glibc is LGPL-2.1-or-later; the file is used under LGPL-2.1 section 3, which allows
  distributing it under the GPL, version 2 or later.
- `libm/exp.ts`: the `exp()` of the same glibc (`__exp_fma` from `e_exp.c` and `e_exp_data.c`), bit for bit, with the
  fused steps of the compiled libm; LGPL-2.1-or-later, used the same way.
- `libm/pow.ts`: the `pow()` of the same glibc (`__ieee754_pow_fma` from `e_pow.c` and `e_pow_log_data.c`, with the
  exp tables), bit for bit, with the fused steps of the compiled libm; LGPL-2.1-or-later, used the same way. JavaScript
  engines disagree on `Math.pow` (Node 18 and 20 differ from glibc for about 10 percent of ordinary arguments), so the
  ported styles call this one.
- `libm/erf.ts`: the `erf()` and `erfc()` of the same glibc (`s_erf.c`, built without FMA; its `exp` calls go to
  `libm/exp.ts`), bit for bit. The file carries the Sun Microsystems notice of `s_erf.c` ("Permission to use, copy,
  modify, and distribute this software is freely granted, provided that this notice is preserved"); used here as part
  of the GPL-2.0 module.
- `libm/sincos.ts`: the `sin()` and `cos()` of the same glibc (the IBM accurate routines of `s_sin.c`,
  `sincostab.c`, `branred.c`, built without FMA contraction), bit for bit; LGPL-2.1-or-later, used the same way.
- `dpdrx/unordered_order.ts`: the iteration order of libstdc++'s `std::unordered_set<std::string>` (GCC 11),
  GPL-3.0-or-later with the GCC Runtime Library Exception (decided 2026-10-15).
- `apip/stdrandom.ts`: libstdc++'s `std::mt19937`, `std::shuffle` and `std::uniform_int_distribution` (GCC 11, the
  random stream of fix lambda_thermostat/apip), GPL-3.0-or-later with the GCC Runtime Library Exception, used the same
  way (written from the algorithms; checked against a GCC 11 build).
