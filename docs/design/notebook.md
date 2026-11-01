# In-browser MD notebook — design

Status: **engine v2, 2026-10-07.** Engine `src/engine/`, worker
`src/workers/engine.worker.ts`, UI `src/components/workbench/Notebook.tsx`.
The notebook runs real LAMMPS input scripts — data files, molecular force
fields, long-range electrostatics, minimization, variables and loops — and
every supported feature is checked against **native LAMMPS used as a
black-box oracle** (below). The CPU-threads and WebGPU paths accelerate the
plain Lennard-Jones case. The general engine's pair term can run on
shared-memory threads when the page is cross-origin isolated; the rest of
each step runs on the engine's thread.

## Goal

Let a student run a *small* molecular-dynamics simulation in the browser:
type LAMMPS input into notebook cells (or add the data and potential files a
script reads), press Run, watch the system move in the 3D viewer and read the
thermo output — no install, no server, no account.

## What it is not

- **It is not LAMMPS.** It is an independent engine that runs a documented,
  growing subset of LAMMPS input. Every unsupported command or style is an
  error that names it — never a silent no-op. Commands and styles that need
  a host process or an external library (`python`, `kim`, `mdi`, `plugin`,
  `fix plumed`, `fix colvars`, `pair_style quip`, `hdnnp`, `runner`, `mbx`,
  `kspace_style scafacos`) are errors that say why. `shell` runs only its
  built-ins (`cd`, `mkdir`, `mv`, `rm [-f]`, `rmdir`, `putenv`) on the
  in-browser file store; any other shell command is an error, since a
  browser has no operating-system shell (`commands/misc.ts`).
- **Its core contains no LAMMPS source code.** The core engine
  (`src/engine/`) is written from textbook physics (Allen & Tildesley;
  Frenkel & Smit; Hockney & Eastwood; Deserno & Holm) and from the
  *documented semantics* on docs.lammps.org, quoted in code comments. Where
  the documentation is silent, the behaviour is read from the LAMMPS source
  and checked against the native binary, and the code says which
  ("As native LAMMPS does (src/X.cpp at b332f28)", "Measured with native
  LAMMPS: ..."). Styles ported from the LAMMPS C++ source are kept in the
  separate GPL-2.0 module [`src/engine-gpl`](../../src/engine-gpl/README.md),
  which carries its own `LICENSE` and is reached only through the lazy style
  registries (`src/engine/lazyStyles.ts`).
- It is not for production science or large systems. The target is
  ≤ ~20 000 atoms, short runs, teaching.

## Verification — native LAMMPS as a black-box oracle

`tests/oracle/*.in` are deterministic inputs: positions from lattices plus
analytic displacements, velocities from atom-style variables, no random
seeds (the `w52rng_*` cases use native's per-atom random streams, see below).
`scripts/oracle/run-oracle.mjs` runs each one through a native `lmp`
binary (`LMP=/path/to/lmp node scripts/oracle/run-oracle.mjs [case ...]`) and
stores in `tests/fixtures/oracle/<case>.json`:
- every thermo row;
- the final per-atom positions, velocities and forces;
- any files the case writes.

`tests/engineOracle.test.ts` runs the same input in the engine and requires
agreement to **1e-8 relative** (thermo) and 1e-6 (per-atom state, written
files), unless a case documents a looser tolerance. Case directives:
`# oracle-inputs:` (files the input reads), `# oracle-files:` (files compared
token by token), `# oracle-compare:` (tolerances, rows, skipped keywords).

The binary is only ever *run*; nothing from its source or its `examples/` and
`potentials/` directories is copied into the repository. Oracle cases use
data and potential files written for this project.

The original acceptance check still holds: the documented `examples/melt` run
reproduces LAMMPS's published step-0 line (Temp 3, E_pair −6.7733681,
TotEng −2.2744931, Press −3.7033504) exactly. The per-atom streams of
`velocity create ... loop geom`, `displace_atoms random` and the random `set`
keywords are native's (`src/engine-gpl/`, reached through `src/engine/nativeRandom.ts`),
so the whole melt run matches native: `tests/oracle/w52rng_melt.in` runs its 250 steps
with thermo every 50 and agrees to about 1e-14 relative in thermo and 6e-10 in positions.
`velocity create ... loop all` and `loop local` keep the Park stream of `src/engine/rng.ts`.

## Supported subset

The authoritative list is what the engine reports: the notebook's help panel
shows the commands and styles, and an unsupported one is an error naming it.
The block below is checked against the engine's registries by
`tests/notebookDocs.test.ts`.

<!-- coverage:begin -->
| Kind | Supported |
|---|---|
| commands | `angle_coeff` `angle_style` `atom_modify` `atom_style` `balance` `bond_coeff` `bond_style` `boundary` `box` `change_box` `clear` `comm_modify` `comm_style` `compute` `compute_modify` `create_atoms` `create_bonds` `create_box` `delete_atoms` `delete_bonds` `dielectric` `dihedral_coeff` `dihedral_style` `dimension` `displace_atoms` `dump` `dump_modify` `echo` `fix` `fix_modify` `geturl` `group` `hyper` `if` `improper_coeff` `improper_style` `include` `info` `jump` `kspace_modify` `kspace_style` `label` `labelmap` `lattice` `log` `mass` `min_modify` `min_style` `minimize` `molecule` `neb` `neb/spin` `neigh_modify` `neighbor` `newton` `next` `package` `pair_coeff` `pair_modify` `pair_style` `partition` `print` `processors` `quit` `read_data` `read_dump` `read_restart` `region` `replicate` `rerun` `reset_atoms` `reset_timestep` `restart` `run` `run_style` `set` `shell` `special_bonds` `suffix` `thermo` `thermo_modify` `thermo_style` `timer` `timestep` `uncompute` `undump` `unfix` `units` `variable` `velocity` `write_data` `write_dump` `write_restart` |
| pair_style | `adp` `aip/water/2dm` `airebo` `airebo/bc` `airebo/morse` `amoeba` `atm` `beck` `body/nparticle` `body/rounded/polygon` `body/rounded/polyhedron` `bondval` `bondval/vec` `bop` `born` `born/coul/dsf` `born/coul/dsf/cs` `born/coul/long` `born/coul/long/cs` `born/coul/msm` `born/coul/wolf` `born/coul/wolf/cs` `born/gauss` `bpm/peri` `bpm/spring` `brownian` `brownian/poly` `buck` `buck/coul/cut` `buck/coul/long` `buck/coul/long/cs` `buck/coul/msm` `buck/long/coul/long` `buck/mdf` `buck6d/coul/gauss/dsf` `buck6d/coul/gauss/long` `colloid` `comb` `comb3` `cosine/squared` `coul/ctip` `coul/cut` `coul/cut/dielectric` `coul/cut/global` `coul/cut/soft` `coul/cut/soft/gapsys` `coul/debye` `coul/diel` `coul/dsf` `coul/esp` `coul/exclude` `coul/long` `coul/long/cs` `coul/long/dielectric` `coul/long/soft` `coul/msm` `coul/shield` `coul/slater/cut` `coul/slater/long` `coul/streitz` `coul/tt` `coul/wolf` `coul/wolf/cs` `dispersion/d3` `dpd` `dpd/coul/slater/long` `dpd/ext` `dpd/ext/tstat` `dpd/fdt` `dpd/fdt/energy` `dpd/tstat` `drip` `dsmc` `e3b` `eam` `eam/alloy` `eam/apip` `eam/cd` `eam/cd/old` `eam/fs` `eam/fs/apip` `eam/he` `edip` `edip/multi` `edpd` `eff/cut` `eim` `exp6/rx` `extep` `gauss` `gauss/cut` `gayberne` `gran/hertz/history` `gran/hooke` `gran/hooke/history` `granular` `granular/superellipsoid` `gw` `gw/zbl` `harmonic/cut` `hbond/dreiding/lj` `hbond/dreiding/lj/angleoffset` `hbond/dreiding/morse` `hbond/dreiding/morse/angleoffset` `hippo` `hybrid` `hybrid/molecular` `hybrid/overlay` `hybrid/scaled` `ilp/graphene/hbn` `ilp/tmd` `kolmogorov/crespi/full` `kolmogorov/crespi/z` `lambda/input/apip` `lambda/input/csp/apip` `lambda/zone/apip` `lcbop` `ldd` `lebedeva/z` `lennard/mdf` `lepton` `lepton/coul` `lepton/sphere` `line/lj` `list` `lj/charmm/coul/charmm` `lj/charmm/coul/charmm/implicit` `lj/charmm/coul/long` `lj/charmm/coul/long/soft` `lj/charmm/coul/msm` `lj/charmmfsw/coul/charmmfsh` `lj/charmmfsw/coul/long` `lj/class2` `lj/class2/coul/cut` `lj/class2/coul/cut/soft` `lj/class2/coul/long` `lj/class2/coul/long/cs` `lj/class2/coul/long/soft` `lj/class2/soft` `lj/cubic` `lj/cut` `lj/cut/coul/cut` `lj/cut/coul/cut/dielectric` `lj/cut/coul/cut/soft` `lj/cut/coul/debye` `lj/cut/coul/debye/dielectric` `lj/cut/coul/dsf` `lj/cut/coul/esp` `lj/cut/coul/long` `lj/cut/coul/long/cs` `lj/cut/coul/long/dielectric` `lj/cut/coul/long/soft` `lj/cut/coul/msm` `lj/cut/coul/msm/dielectric` `lj/cut/coul/wolf` `lj/cut/dipole/cut` `lj/cut/dipole/long` `lj/cut/soft` `lj/cut/soft/gapsys` `lj/cut/sphere` `lj/cut/thole/long` `lj/cut/tip4p/cut` `lj/cut/tip4p/long` `lj/cut/tip4p/long/soft` `lj/expand` `lj/expand/coul/long` `lj/expand/sphere` `lj/gromacs` `lj/gromacs/coul/gromacs` `lj/long/coul/long` `lj/long/coul/long/dielectric` `lj/long/dipole/long` `lj/long/tip4p/long` `lj/mdf` `lj/pirani` `lj/relres` `lj/sdk` `lj/sdk/coul/long` `lj/sdk/coul/msm` `lj/sf` `lj/sf/dipole/sf` `lj/smooth` `lj/smooth/linear` `lj/spica` `lj/spica/coul/long` `lj/spica/coul/msm` `lj/switch3/coulgauss/long` `lj96/cut` `local/density` `lubricate` `lubricate/poly` `lubricateU` `lubricateU/poly` `mdpd` `mdpd/rhosum` `meam` `meam/ms` `meam/spline` `meam/sw/spline` `mesocnt` `mesocnt/viscous` `mgpt` `mie/cut` `mliap` `mm3/switch3/coulgauss/long` `momb` `morse` `morse/smooth/linear` `morse/soft` `multi/lucy` `multi/lucy/rx` `nb3b/harmonic` `nb3b/screened` `nm/cut` `nm/cut/coul/cut` `nm/cut/coul/long` `nm/cut/split` `oxdna/coaxstk` `oxdna/excv` `oxdna/hbond` `oxdna/stk` `oxdna/xstk` `oxdna2/coaxstk` `oxdna2/dh` `oxdna2/excv` `oxdna2/hbond` `oxdna2/stk` `oxdna2/xstk` `oxdna3/coaxstk` `oxdna3/dh` `oxdna3/excv` `oxdna3/hbond` `oxdna3/stk` `oxdna3/xstk` `oxrna2/coaxstk` `oxrna2/dh` `oxrna2/excv` `oxrna2/hbond` `oxrna2/stk` `oxrna2/xstk` `pace` `pace/apip` `pace/extrapolation` `pace/fast/apip` `pace/precise/apip` `pedone` `peri/eps` `peri/lps` `peri/pmb` `peri/ves` `pod` `polymorphic` `reaxff` `rebo` `rebomos` `resquared` `rheo` `rheo/solid` `saip/metal` `saip/metal/tmd` `sdpd/taitwater/isothermal` `smatb` `smatb/single` `smd/hertz` `smd/tlsph` `smd/tri_surface` `smd/ulsph` `smtbq` `snap` `soft` `sph/heatconduction` `sph/idealgas` `sph/lj` `sph/rhosum` `sph/taitwater` `sph/taitwater/morris` `spin/dipole/cut` `spin/dipole/long` `spin/dmi` `spin/exchange` `spin/exchange/biquadratic` `spin/magelec` `spin/neel` `srp` `srp/react` `surf/granular` `sw` `sw/angle/table` `sw/mod` `table` `table/rx` `tdpd` `tersoff` `tersoff/mod` `tersoff/mod/c` `tersoff/table` `tersoff/zbl` `thole` `threebody/table` `tip4p/cut` `tip4p/long` `tip4p/long/soft` `tracker` `tri/lj` `uf3` `ufm` `vashishta` `vashishta/table` `wf/cut` `ylz` `yukawa` `yukawa/colloid` `zbl` `zero` `zero/coul` |
| bond_style | `bpm/peri` `bpm/rotational` `bpm/spring` `bpm/spring/plastic` `bpm/zero` `class2` `fene` `fene/expand` `fene/nm` `gaussian` `gromos` `harmonic` `harmonic/restrain` `harmonic/shift` `harmonic/shift/cut` `hybrid` `lepton` `mesocnt` `mm3` `morse` `nonlinear` `oxdna/fene` `oxdna2/fene` `oxdna3/fene` `oxrna2/fene` `quartic` `quartic/exp` `rheo/shell` `special` `table` `zero` |
| angle_style | `amoeba` `charmm` `class2` `class2/p6` `class2xe` `cosine` `cosine/buck6d` `cosine/delta` `cosine/periodic` `cosine/shift` `cosine/shift/exp` `cosine/squared` `cosine/squared/restricted` `cross` `dipole` `fourier` `fourier/simple` `gaussian` `harmonic` `hybrid` `lepton` `mesocnt` `mm3` `mwlc` `quartic` `sdk` `spica` `table` `zero` |
| dihedral_style | `charmm` `charmmfsw` `class2` `class2xe` `cosine/shift/exp` `cosine/squared/restricted` `fourier` `harmonic` `helix` `hybrid` `lepton` `multi/harmonic` `nharmonic` `opls` `quadratic` `spherical` `table` `table/cut` `zero` |
| improper_style | `amoeba` `class2` `cossq` `cvff` `distance` `distharm` `fourier` `harmonic` `hybrid` `inversion/harmonic` `ring` `sqdistharm` `umbrella` `zero` |
| kspace_style | `esp` `ewald` `ewald/dipole` `ewald/dipole/spin` `ewald/disp` `ewald/disp/dipole` `ewald/electrode` `msm` `msm/cg` `msm/dielectric` `pppm` `pppm/cg` `pppm/dielectric` `pppm/dipole` `pppm/dipole/spin` `pppm/disp` `pppm/disp/dielectric` `pppm/disp/tip4p` `pppm/electrode` `pppm/rk` `pppm/stagger` `pppm/tip4p` `zero` |
| fix | `SRP` `SRPREACT` `accelerate/cos` `acks2/reax` `acks2/reaxff` `adapt` `adapt/fep` `add/heat` `addforce` `addtorque` `addtorque/atom` `addtorque/group` `alchemy` `align/self` `amoeba/bitorsion` `amoeba/pitorsion` `append/atoms` `atom/swap` `atom_weight/apip` `ave/atom` `ave/chunk` `ave/correlate` `ave/correlate/long` `ave/grid` `ave/histo` `ave/histo/weight` `ave/moments` `ave/time` `aveforce` `balance` `baoab` `bocs` `bond/break` `bond/create` `bond/create/angle` `bond/react` `bond/swap` `box/relax` `brownian` `brownian/asphere` `brownian/sphere` `charge/regulation` `cmap` `controller` `damping/cundall` `deform` `deform/pressure` `deposit` `dpd/energy` `drag` `drude` `drude/transform/direct` `drude/transform/inverse` `dt/reset` `edpd/source` `efield` `efield/lepton` `efield/tip4p` `ehex` `electrode/conp` `electrode/conq` `electrode/thermo` `electron/stopping` `electron/stopping/fit` `enforce2d` `eos/cv` `eos/table` `eos/table/rx` `evaporate` `external` `ffl` `filter/corotate` `flow/gauss` `freeze` `gcmc` `gemc` `gjf` `gld` `gle` `graphics/arrows` `graphics/chunk` `graphics/isosurface` `graphics/labels` `graphics/lines` `graphics/objects` `graphics/periodic` `graphics/replica` `gravity` `grem` `halt` `heat` `heat/flow` `hmc` `hyper/global` `hyper/local` `ilves` `indent` `lambda/apip` `lambda/la/csp/apip` `lambda_thermostat/apip` `langevin` `langevin/drude` `langevin/eff` `langevin/spin` `lb/fluid` `lb/momentum` `lb/viscous` `lineforce` `manifoldforce` `meso/move` `mol/swap` `momentum` `momentum/chunk` `move` `msst` `mvv/dpd` `mvv/edpd` `mvv/tdpd` `neb` `neb/spin` `neighbor/swap` `nonaffine/displacement` `nph` `nph/asphere` `nph/body` `nph/eff` `nph/sphere` `nphug` `npt` `npt/asphere` `npt/body` `npt/cauchy` `npt/eff` `npt/sphere` `npt/uef` `numdiff` `numdiff/virial` `nve` `nve/asphere` `nve/asphere/noforce` `nve/body` `nve/bpm/sphere` `nve/dot` `nve/dotc/langevin` `nve/eff` `nve/limit` `nve/line` `nve/manifold/rattle` `nve/noforce` `nve/sphere` `nve/spin` `nve/tri` `nvk` `nvt` `nvt/asphere` `nvt/body` `nvt/eff` `nvt/manifold/rattle` `nvt/sllod` `nvt/sllod/eff` `nvt/sphere` `nvt/uef` `oneway` `orient/bcc` `orient/eco` `orient/fcc` `pafi` `pair` `phonon` `pimd` `pimd/langevin` `pimd/langevin/bosonic` `pimd/nvt` `pimd/nvt/bosonic` `planeforce` `polarize/bem/gmres` `polarize/bem/icc` `polarize/functional` `pour` `precession/spin` `press/berendsen` `press/langevin` `print` `propel/self` `property/atom` `qbmsst` `qeq/comb` `qeq/ctip` `qeq/dynamic` `qeq/fire` `qeq/point` `qeq/reax` `qeq/reaxff` `qeq/rel/reaxff` `qeq/shielded` `qeq/slater` `qtb` `qtpie/reaxff` `rattle` `reaxff/bonds` `reaxff/species` `recenter` `restrain` `rheo` `rheo/oxidation` `rheo/pressure` `rheo/thermal` `rheo/viscosity` `rhok` `rigid` `rigid/meso` `rigid/nph` `rigid/nph/small` `rigid/npt` `rigid/npt/small` `rigid/nve` `rigid/nve/small` `rigid/nvt` `rigid/nvt/small` `rigid/small` `rx` `saed/vtk` `set` `setforce` `setforce/spin` `settorque/atom` `sgcmc` `shake` `shardlow` `smd` `smd/adjust_dt` `smd/integrate_tlsph` `smd/integrate_ulsph` `smd/move_tri_surf` `smd/setvel` `smd/wall_surface` `sph` `sph/stationary` `spring` `spring/chunk` `spring/rg` `spring/self` `srd` `store/force` `store/state` `surface/global` `surface/local` `tdpd/source` `temp/berendsen` `temp/csld` `temp/csvr` `temp/rescale` `temp/rescale/eff` `tfmc` `tgnpt/drude` `tgnvt/drude` `thermal/conductivity` `ti/spring` `tmd` `ttm` `ttm/cascade` `ttm/grid` `ttm/mod` `ttm/thermal` `vector` `viscosity` `viscous` `viscous/nonlinear` `viscous/sphere` `wall/body/polygon` `wall/body/polyhedron` `wall/colloid` `wall/ees` `wall/flow` `wall/gran` `wall/gran/region` `wall/harmonic` `wall/harmonic/outside` `wall/harmonic/returned` `wall/lepton` `wall/lj1043` `wall/lj126` `wall/lj93` `wall/morse` `wall/piston` `wall/reflect` `wall/reflect/stochastic` `wall/region` `wall/region/ees` `wall/srd` `wall/table` `widom` |
| compute | `PRESSURE/BOCS` `ackland/atom` `adf` `aggregate/atom` `angle` `angle/local` `angmom/chunk` `ave/sphere/atom` `basal/atom` `body/local` `bond` `bond/local` `born/matrix` `centro/atom` `centroid/stress/atom` `chunk/atom` `chunk/spread/atom` `cluster/atom` `cna/atom` `cnp/atom` `com` `com/chunk` `composition/atom` `contact/atom` `coord/atom` `count/type` `damage/atom` `dihedral` `dihedral/local` `dilatation/atom` `dipole` `dipole/chunk` `dipole/tip4p` `dipole/tip4p/chunk` `displace/atom` `dpd` `dpd/atom` `edpd/temp/atom` `efield/atom` `efield/wolf/atom` `entropy/atom` `erotate/asphere` `erotate/rigid` `erotate/sphere` `erotate/sphere/atom` `event/displace` `fabric` `fep` `fep/ta` `force/tally` `fragment/atom` `frenkel` `gaussian/grid/local` `global/atom` `group/group` `gyration` `gyration/chunk` `gyration/shape` `gyration/shape/chunk` `hbond/local` `heat/flux` `heat/flux/tally` `heat/flux/virial/tally` `hexorder/atom` `hma` `improper` `improper/local` `inertia` `inertia/chunk` `ke` `ke/atom` `ke/atom/eff` `ke/eff` `ke/rigid` `mliap` `momentum` `msd` `msd/chunk` `msd/nongauss` `nbond/atom` `omega/chunk` `orientorder/atom` `pace` `pair` `pair/local` `pe` `pe/atom` `pe/mol/tally` `pe/tally` `plasticity/atom` `pod/atom` `pod/global` `pod/local` `podd/atom` `pressure` `pressure/alchemy` `pressure/cylinder` `pressure/uef` `property/atom` `property/chunk` `property/grid` `property/local` `ptm/atom` `rattlers/atom` `rdf` `reaxff/atom` `reduce` `reduce/chunk` `reduce/region` `rheo/property/atom` `rigid/local` `saed` `slcsa/atom` `slice` `smd/contact/radius` `smd/damage` `smd/hourglass/error` `smd/internal/energy` `smd/plastic/strain` `smd/plastic/strain/rate` `smd/rho` `smd/tlsph/defgrad` `smd/tlsph/dt` `smd/tlsph/num/neighs` `smd/tlsph/shape` `smd/tlsph/strain` `smd/tlsph/strain/rate` `smd/tlsph/stress` `smd/triangle/vertices` `smd/ulsph/effm` `smd/ulsph/num/neighs` `smd/ulsph/strain` `smd/ulsph/strain/rate` `smd/ulsph/stress` `smd/vol` `sna/atom` `sna/grid` `sna/grid/local` `snad/atom` `snap` `snav/atom` `sph/e/atom` `sph/rho/atom` `sph/t/atom` `spin` `stress/atom` `stress/cartesian` `stress/cylinder` `stress/mop` `stress/mop/profile` `stress/spherical` `stress/tally` `tdpd/cc/atom` `temp` `temp/asphere` `temp/body` `temp/chunk` `temp/com` `temp/cs` `temp/deform` `temp/deform/eff` `temp/drude` `temp/eff` `temp/partial` `temp/profile` `temp/ramp` `temp/region` `temp/region/eff` `temp/rotate` `temp/sphere` `temp/uef` `ti` `torque/chunk` `vacf` `vacf/chunk` `vcm/chunk` `viscosity/cos` `voronoi/atom` `xrd` |
| min_style | `cg` `sd` `fire` `quickmin` `hftn` `spin` `spin/cg` `spin/lbfgs` |
<!-- coverage:end -->

Native binary restart files (written by LAMMPS b332f28, serial, one file) are read by `read_restart` for the
styles named in `src/engine-gpl/restart/nativeRestart.ts`; any other style, a multi-file (`%`) restart, or fix state
the engine does not hold is an error that names it. `write_restart` writes the engine's own format only. Not supported
(an error names each): `atom_style hybrid` with a `body` or `line` sub-style or with `apip conservative`,
`fix tune/kspace` (it picks the kspace settings from timed runs on the host), and the styles that need an external
library or process (listed above).

## Architecture

```
src/engine/
  types.ts, units.ts, domain.ts     contracts, unit constants, box geometry (triclinic, boundaries)
  system.ts, session.ts             one simulation; command dispatch, loops, jump/label/if, include
  commands/                         setup, force field, run and misc command handlers
  script.ts, formula.ts, variables.ts, groupfn.ts, boolean.ts, refs.ts
                                    parsing, $-substitution, variables, c_/f_/v_ references
  atoms.ts, lattice.ts, region.ts, group.ts, neighbor.ts
                                    atoms and topology, lattices, regions, groups,
                                    ghost atoms + binned Verlet lists with special bonds
  force/                            force field: pair/, bond/, angle/, dihedral/, improper/,
                                    kspace/ (ewald, pppm), fft.ts, erfc.ts
  fix/, compute/                    fixes and computes (Developer_flow hook order)
  run/                              velocity Verlet run loop, minimizers, accelerators
  output/                           thermo, dump, write_data
  registry/ + styles.ts             style name -> implementation, one file per family
  cpu/, gpu/                        threaded CPU force paths (plain lj/cut; shared-memory
                                    pair threads for the general pair term), WebGPU forces
  workers/                          engine, force and pair Web Workers
  host.ts, client.ts, protocol.ts   worker plumbing; files added in the notebook
src/components/workbench/Notebook.tsx   the notebook module
```

### Accelerated backends (plain lj/cut)

For the one case they implement exactly — `pair_style lj/cut`, a fully
periodic orthogonal box, no bonds, charges or kspace, only `fix nve` /
`enforce2d` on all atoms, no pressure-tensor output — the notebook's
CPU-threads and WebGPU choices replace the force evaluation
(`src/engine/run/accel.ts`). Any other input runs on the fp64 engine, and
the run log says why. One interface, two implementations:

```ts
interface ForceBackend {
  readonly kind: 'cpu' | 'webgpu';
  compute(state: SimState): ForceResult;   // forces, pe, virial
}
```

- **CPU (fp64)** is the reference and the fallback.
- **WebGPU (fp32)**, v1: the CPU builds the cell list each step and uploads
  positions; a WGSL kernel scans the 27 neighbour cells per atom and writes
  forces, per-atom energy and virial; the CPU integrates. This keeps the GPU
  path small enough to verify against the CPU path force-by-force.
- **WebGPU, more pair styles** (`src/engine/gpu/pairSpec.ts`): besides lj/cut, the
  GPU runs `lj/cut/coul/cut`, `coul/cut`, `buck`, `buck/coul/cut`, `morse`, `soft`,
  `yukawa`, `lj96/cut`, `born`, `lj/class2` and `lj/class2/coul/cut` under
  the same conditions (charges allowed). The coefficients are the engine style's own
  after its init (the arrays its compute() reads), a second kernel switches on the
  style per pair, and the charges ride in the integer buffer as f32 bits (WebGPU
  compatibility mode allows four storage buffers). Van der Waals and Coulomb energies
  come back separately. `pairSpecEval` is the kernel's law in fp64:
  `tests/engineGpuPairSpec.test.ts` checks it against each CPU style force by force.
  Whole-step GPU runs (`resident.ts`) take lj/cut and these styles (the force laws are
  one WGSL module, `src/engine/gpu/styleLaws.ts`, shared by both kernels; the kernels switch
  on the short-range law, `SPEC_LAW`, so styles that share a law share its WGSL). For a charged
  style the force entry point has no fifth storage buffer to spare, so the charges ride
  after the coefficients in the coefficient buffer and the scatter pass records the atom
  of each slot in the cell-data buffer.
- **WebGPU with bonds** (`run/accel.ts`, `gpu/webgpuForces.ts`): a system with molecular
  topology or bond, angle, dihedral or improper styles (no kspace style) runs the pair term
  on the GPU with the styles above, lj/cut included (spec style 10, since the analytic
  lj/cut kernel has no special weights). Each step the bonded terms run on the CPU in fp64
  (`ForceField.compute` with only those terms, partners by the minimum image because the
  CPU builds no neighbor list on this path) and the GPU's pair forces are added to theirs.
  The special list goes up after the charges in the integer buffer: per slot a range of
  entries `(partner slot << 2) | order`, only for orders whose `special_bonds` weights are
  not both 1. The kernel scales the short-range part by the lj weight and the Coulomb part
  by the coul weight, and skips a pair whose weights are both 0, as the neighbor list does
  (`tests/engineGpuBonded.test.ts` runs chains through the loop with the kernel's arithmetic
  in fp64 against the CPU engine; in Chromium, chains of 1728 atoms with lj/cut and with
  lj/cut/coul/cut matched the CPU thermo to the 6 printed digits over 100 steps). These runs
  are not resident: every step comes back to the host.
- **WebGPU with kspace** (`gpu/pairSpec.ts` LONG_COUL_STYLES): `coul/long`, `lj/cut/coul/long`,
  `lj/charmm/coul/long`, `lj/class2/coul/long`, `buck/coul/long` and `born/coul/long` run their
  real-space part on the GPU and the kspace style (Ewald, PPPM,
  on the CPU threads) beside it, with the bonded terms. The kernel's Coulomb term is
  C q_i q_j erfc(g r)/r with erfc from the polynomial fit native uses without tables
  (`erfc.ts erfcPoly`; `pair_modify table` N > 0 interpolates the same function, below fp32
  rounding), G-ewald from the kspace style each step. A special pair stays a pair even with
  both weights 0 (special_bonds.html: with kspace the 1-2, 1-3 and 1-4 pairs are not excluded
  from the neighbor list), and its coul weight w removes (1 - w) of the bare C q_i q_j / r term.
  The GPU's pair energies are added to the accumulator, not stored over it, because dihedral
  charmm tallies its weighted 1-4 pairs into evdwl and ecoul. Along an edge of 2 to 3 cutoffs
  the kernel scans every cell once instead of the 27-cell neighbourhood; the minimum image keeps
  that exact (the whole-step kernel still needs 3 cutoffs). In Chromium the LAMMPS peptide
  example (27.4 A box, cutoff 10, PPPM, SHAKE, nvt) ran on this path and matched the CPU
  engine's thermo over 100 steps to fp32 (pe -6372.39 vs -6372.38 at step 0, the other energy
  terms to all printed digits). `fix adapt` keeps a run on the CPU, because the backends upload
  the pair coefficients once.
- **WebGPU pair_style table** (`gpu/pairSpec.ts` style 19, kernel law 20): the style's own
  interpolation tables after its init (`table.ts`: energy and F/r on a uniform r^2 grid, and the
  spline's second derivatives) follow the type pairs' coefficients in the coefficient buffer,
  which the kernel reads as flat floats. Lookup, linear and spline interpolation are table.ts's
  arithmetic (`pairSpecEval` checks it force by force). A pair inside a table's inner cutoff
  flags its atom; the CPU then finds the pair in fp64 and stops with the CPU style's error, so the
  GPU never integrates garbage. Per-step only: the whole-step kernel has the analytic laws.
  In Chromium a 4000-atom LJ melt through `table linear` and `table spline` matched the CPU
  engine's thermo over 100 steps.
- **WebGPU Stillinger-Weber** (`gpu/swSpec.ts`, `gpu/swKernels.ts`, `webgpuForces.ts computeSW`):
  `sw` (the GPL port) and `sw/mod` (core) hand their parameters to the GPU through a `gpuSW()`
  accessor (the two-body entry and leg of each type pair, the angular entry of each triplet).
  Two passes in one submission: each atom's neighbor list (written after the cell data in the
  integer buffer, which the first pass writes and the second reads), then a gather: each atom
  adds the forces on itself from its pairs, from the triplets it centers and from those
  centered on a neighbor in which it is a leg, so no atomics are needed; a triplet's energy
  and virial count at its center. Lists grow and the evaluation reruns when an atom has more
  neighbors than they hold. Asymmetric (I,J,K)/(I,K,J) entries are refused (native's result
  depends on its neighbor order there). In fp32 the two-body term must test `r - a sigma < 0`
  as well as `r^2 < cutoff^2`: sqrt can land an ulp past `a sigma`, and `exp(sigma / den)` of
  a tiny positive `den` is infinite (a 4096-atom sw/mod run went to inf at step 45 before
  that). `swGather` is the arithmetic in fp64 (`tests/engineGpuSw.test.ts`); in Chromium a
  4096-atom Si crystal matched the CPU engine's thermo over 100 steps with both styles.
- **WebGPU Tersoff** (`gpu/tersoffSpec.ts`, `gpu/tersoffKernels.ts`, `webgpuForces.ts computeTersoff`):
  `pair_style tersoff` (with or without `shift`; not the mod, mod/c or zbl variants, not as a
  hybrid sub-style, no molecular topology) hands its tables to the GPU through `gpuTersoff()`.
  Three passes in one submission: neighbor lists, then per bond i -> j its bond order b_ij and
  C_ij = 1/2 f_C f_A db/dzeta (zeta_ij sums over i's other neighbors), then a gather in which
  each atom adds the forces of its own bonds, of its neighbors' bonds toward it, and of its
  neighbors' other bonds in whose zeta it is the third atom. The gather needs symmetric lists,
  so a type pair's list cutoff is the larger of its two orders. fp32 traps: the angular term
  `1 + c^2/d^2 - c^2/(d^2 + q^2)` is evaluated as `1 + c^2 q^2 / (d^2 (d^2 + q^2))` (c^2/d^2 is
  ~4e7 for the Si and C sets), and the bond order is evaluated from `u = n ln(beta zeta)` in
  two overflow-free forms. Before that, `zeta^n` overflowed fp32 with Si(B)'s n = 22.956: a bond
  at the cutoff beside a short one gave zeta = 51 and `db/dzeta = 0 * inf = NaN` at step 43 of a
  4096-atom run. zeta's exponent is capped at 69.0776 as native does. `tersoffGather` is the
  arithmetic in fp64 (`tests/engineGpuTersoff.test.ts`). In Chromium a 4096-atom Si(B) crystal
  at 1600 K matched the CPU engine's thermo to all printed digits over 100 steps. A 4096-atom
  B/N/C system with `shift` gave forces within 5e-5 of the largest force at every 10th step.
- **WebGPU TIP4P** (`gpu/tip4pSpec.ts`, `gpu/tip4pKernels.ts`, `webgpuForces.ts computeTip4p`):
  `lj/cut/tip4p/long`, `lj/cut/tip4p/cut`, `tip4p/long` and `tip4p/cut` (not as hybrid sub-styles).
  The bonded terms and `pppm/tip4p` run on the CPU beside the GPU, as for the other molecular
  styles. Each step the host builds the M sites in fp64 (`tip4p_sites.ts buildOwnedSites`) and
  uploads each oxygen's offset M - x_O. One pass gathers per atom the Lennard-Jones force on the
  atom and the Coulomb force on its charge site. A site pair's separation is the atoms' minimum
  image plus the difference of their offsets, so the sites need no image of their own. As the
  CPU style does, a Coulomb pair is tested on the atoms against (cut_coul + 2 qdist)^2, then on
  the sites against cut_coul^2. Special pairs keep the kspace correction with their site
  separation. The host spreads each oxygen's site force (1 - alpha) to it and alpha/2 to each
  hydrogen. `tip4pGather` is the arithmetic in fp64: setup forces match the CPU styles to 1e-10
  and 40-step thermo to 1e-9 (`tests/engineGpuTip4p.test.ts`). In Chromium a 3000-atom TIP4P
  water box (8.5 A cutoff, pppm/tip4p) matched the CPU engine's temperature to 6 digits over
  100 steps, with and without SHAKE. The step-0 energy is within 0.02 kcal/mol (2e-7 of the
  Coulomb energy) of the CPU with `pair_modify table 0`; the CPU's default table adds another
  0.04. A box shorter than twice (cut_coul + 2 qdist) stays on the CPU, as do the gle water
  examples (12.4 A).
- **The virial tensor on the GPU path.** Every per-step kernel (lj/cut, the styles kernel, EAM, SW,
  Tersoff, TIP4P) accumulates the six virial components (xx yy zz xy xz yz) next to its scalar
  virial: half of 0.5 F/r d (x) d at each atom of a pair, d1 (x) F1 + d2 (x) F2 for a three-body
  term at its center. The host sums them in fp64 into `virialTensor`. Barostats (whose own
  pressure compute is a user pressure compute: `fix npt`, `nph`), user pressure computes and
  `pxx`..`pyz` therefore take the GPU path; only the whole-step kernel (resident.ts) and the
  lj/cut threads of cpu/parallel.ts still give the scalar virial, so such runs are not resident
  there and stay off those threads. A fix that changes the box (a barostat, fix deform) needs 3
  pair cutoffs per periodic edge at the start instead of 2, so a shrinking box keeps the 2 the
  kernels' minimum image needs. In Chromium, fix npt aniso on WebGPU tracked the CPU engine's
  pxx..pyz and volume for lj/cut, EAM, SW, Tersoff, TIP4P water and rhodopsin (32000 atoms,
  CHARMM + PPPM + SHAKE) to fp32 precision.
- **`pair_modify tail yes` on the accelerated paths.** The tail correction depends on the volume and the atom counts
  only, so the accelerated loop adds etail / V to the pair energy after each GPU (or lj/cut threads) evaluation, as
  `ForceField.compute` does when it computes every term; its pressure term comes from the pressure compute as on the
  CPU path (`tests/engineGpuTail.test.ts`, fix npt included). Dynamic groups are reassigned on the accelerated loop
  where the CPU loop does it (after the forces, before the fixes; group.html), so runs with them take the GPU path
  too, though not the whole-step one.
- **`neigh_modify exclude` and `include` on the GPU path** (`gpu/exclusions.ts`): runs with exclusions take the styles
  kernel (lj/cut as its style 10). Excluded type pairs get zero cutoffs in the packed coefficients; group,
  molecule/intra, molecule/inter rules (up to 16) and the include group are tested per pair in the kernel from each
  slot's group mask and molecule ID, before the special-bond lookup, so an excluded pair has no pair term, special
  weight or kspace correction, as on the CPU path. Not resident; the lj/cut threads, the EAM, SW, Tersoff and TIP4P
  kernels and rules on dynamic groups stay on the CPU. In Chromium, examples/rigid in.rigid (9 group rules) and
  in.rigid.cubes (molecule/intra) ran on WebGPU: in.rigid.cubes matched the CPU engine to all printed digits at step
  1000, in.rigid to 1e-5 through the clump collision (fp32), its later steps diverging chaotically.
- **Non-periodic boxes on the GPU path** (boundary `f`, `s`, `m`; `gpu/cellSort.ts`). The kernels' minimum image is
  `d - L round(d / L)`; along a non-periodic dimension the cell sort hands them a box length of 1e30 there, so it is
  the identity, and every edge of any length has at least one cell. Atoms past a non-periodic edge go into the edge
  cell, which keeps every partner within the cutoff in the cells the scan takes. The accelerated loop now makes the CPU
  loop's reneighboring decision each step (`Neighbor.decide`: every, delay, check, restart files, fixes). On those
  steps it does what the CPU loop does: the periodic wrap, the reset of a shrink-wrapped box and the lost atoms of a
  fixed boundary (`System.pbc`), and the sort. It records the build without the CPU lists (`Neighbor.markBuilt`), so
  `nbuild` and `ndanger` count as there. Between those steps the atoms move on unwrapped, as on the CPU path. The
  cell sorts (the GPU's, `cpu/cells.ts`, the lj/cut cell list of `cpu/forces.ts`) bin an atom past a periodic edge in
  its image's cell (`periodicCell`). lj/cut in such a box runs as the styles kernel's style 10; the plain lj/cut kernel,
  the whole-step kernel and the lj/cut threads keep fully periodic boxes. The 2-cutoff minimum applies to periodic edges
  only. Tests: `tests/engineGpuNonPeriodic.test.ts`. In Chromium (SwiftShader
  WebGPU), shortened LAMMPS examples on WebGPU matched the CPU engine:
  - shear/in.shear (s s p, EAM, both runs): every printed digit of temp, pe and etotal; press within 0.1 (values to 1.4e4);
  - indent/in.indent (p s p, fix indent), friction/in.friction (p s p) and wall/in.wall.specular (p f p) over 1500
    to 3000 steps: to the printed digits, save the sixth of a few;
  - crack/in.crack (s s p, 8141 atoms): matched through step 400.
- **Triclinic boxes on the GPU path** (`gpu/cellSort.ts`). The cell sort bins the atoms' fractional coordinates
  (lamda = h^-1 (x - lo), by back substitution in the upper-triangular h of the edge vectors), nc_d cells along d with
  the box's width perpendicular to the other two edges (`boxWidths`: volume over the opposite face) at least nc_d
  cutoffs, so two atoms within the cutoff are in the same or neighbouring cells. The positions stay Cartesian. The
  styles, EAM, SW, Tersoff and TIP4P kernels take the tilt in a `tilt` vec4 at the end of their uniform. Their minimum
  image (`WGSL_BOX_IMAGE`) rounds the separation's fractional coordinates and subtracts that lattice vector, which is
  the one copy of a partner within the cutoff while every periodic width is at least 2 cutoffs; the accelerator checks
  the widths. With tilt 0 it is the same arithmetic as before. lj/cut runs as the styles kernel's style 10; the plain
  lj/cut kernel, the whole-step kernel and the lj/cut threads take orthogonal boxes. Box flips (fix deform, the
  barostats with flip yes) come in pre_exchange, which keeps those runs on the CPU. Tests:
  `tests/engineGpuTriclinic.test.ts`. In Chromium, the triclinic examples on WebGPU matched the CPU engine:
  - rigid/in.rigid.cubes.tri (styles kernel, molecule/intra exclusions, 1000 steps): every printed digit;
  - threebody/in.mos2.sw.mod (sw/mod, p p f, xy tilt, 1000 steps): to the last printed digit, plus or minus 1;
  - PACKAGES/entropy/in.entropy (eam/fs, 4096 atoms, 1000 steps): every printed digit of pe and etotal, and of temp
    except a last-digit difference at step 1000; press within 0.011;
  - PACKAGES/phonon/4-Graphene/in.graphene (tersoff, 600 steps): to 5 or 6 digits;
  - TIP4P water (lj/cut/tip4p/cut, all three tilts): to 5 or 6 digits through 250 steps of a hot (1800 K) system;
  - orthogonal runs of the changed kernels (shear/in.shear EAM, tersoff/in.tersoff, TIP4P with pppm/tip4p): the
    printed numbers of shear are those before the change.
- **GPU validation errors stop the run.** Pipelines are created in a validation error scope
  (`checkedPipelines`), and each force evaluation runs in one (`validated`): WebGPU skips an
  invalid dispatch, which would otherwise leave zero forces (an entry point whose `auto`
  layout lacked a binding did exactly that).
- **WebGPU EAM** (`gpu/eamSpec.ts`, `gpu/eamKernels.ts`, `webgpuForces.ts computeEAM`): `eam`,
  `eam/alloy` and `eam/fs` (not as hybrid sub-styles, no molecular topology). The style's own
  density and r*phi tables after its init (the seven coefficients per grid interval its loops
  read, `eamKernel.ts`) go up once as f32, table offsets as f32 integers (as u32 bit patterns
  they would be denormals, which a GPU may flush). A step is two passes, each atom summing over
  its cell neighbourhood without atomics: the densities, read back; F'(rho) and the embedding
  energy on the CPU in fp64 with the style's own embedding spline (`eam.ts embedOwned`, the loop
  `embed()` runs); F' up as f32 bits after the cell data; then the forces, pair energies and
  virial. `eamDensity` and `eamPair` are the kernels' arithmetic in fp64
  (`tests/engineGpuEam.test.ts` checks them against the CPU styles and runs 30 steps through the
  loop). In Chromium a 4000-atom CuNi eam/alloy run matched the CPU engine's thermo over 100
  steps (pe -15872.5 both at step 0, press -9173.83 vs -9173.9). Two round trips per step; not
  resident.
- **WebGPU resident stepping**, v2 (`src/engine/gpu/resident.ts`): for runs
  whose only fixes are `nve` (plus `enforce2d`), whole velocity-Verlet steps
  stay on the GPU — half kick + drift + wrap, cell list rebuilt on the GPU
  (atomic counts, one-workgroup prefix sum, scatter into cell-sorted slots),
  the same 27-cell force kernel, second half kick. `md.run` hands the backend
  chunks of steps ending at the next step the host needs (thermo, dump, viewer
  frame, end of run; at most 200 so Stop stays responsive), and x, v, f,
  images, energy and virial are read back once per chunk. Thermostats and
  other fixes keep the per-step path. Every entry point fits compatibility
  mode (<= 4 storage buffers, <= 128 invocations per workgroup), and kernel
  validation errors are raised, not read back as zeros.
- The two must agree: GPU vs CPU forces within 1e-4 relative on random LJ
  configurations, checked in a real Chromium (SwiftShader adapter). The
  resident path additionally: one GPU step vs one fp64 CPU step (x within
  1e-5, v within 1e-4, F within 1e-4 relative), melt step-100 temperature
  within 0.02 of the CPU, NVE energy drift over 2000 steps below 2e-3 per atom
  (measured 1.2e-4 on the A100, 9e-5 on SwiftShader, fp32), and dumps on
  their steps. `tests/engineResident.test.ts` checks the chunking against an
  fp64 stand-in, which must reproduce the per-step loop exactly.

### Threading

The engine runs in a dedicated Web Worker (WebGPU is exposed to workers), so
the notebook and the 3D viewer never stall. The worker posts thermo rows and
frames (every 25 steps, repainted at most ~8x/s) that the existing
`MoleculeCanvas` renders.

CPU forces can use several threads (`src/engine/cpu/parallel.ts`): the engine
worker sorts atoms into cells, splits the cells into contiguous ranges of
about equal atom count, computes one range itself and sends the others to
force workers (`src/workers/force.worker.ts`). Each range uses the half
stencil, so every pair is computed once; reactions on atoms owned by another
range come back as a short ghost list and are summed by the engine worker.
No SharedArrayBuffer is needed (GitHub Pages cannot send the cross-origin
isolation headers it requires).

### Choosing the GPU

`createWebGpuBackend()` asks for a high-performance core adapter, then for a
compatibility-mode one (`featureLevel: 'compatibility'`), and takes the first
that is not a software fallback. On Linux + NVIDIA, Chromium exposes the GPU
only in compatibility mode (OpenGL ES through ANGLE on Vulkan), whose limit of
4 storage buffers per shader stage the kernel respects. A software adapter
(SwiftShader) is declined — it is slower than the CPU engine — unless a test
passes `allowFallback`.

### Device detection and Auto

`probeDevice()` (`src/engine/device.ts`) runs in the engine worker, so it
reports what the engine can use: logical cores (`navigator.hardwareConcurrency`),
memory (`navigator.deviceMemory`, Chromium), Web Worker support, shared memory
(SharedArrayBuffer and Atomics on a cross-origin isolated page) and the WebGPU
adapter (`webgpuAdapterInfo` in `gpu/webgpuForces.ts`: hardware, software or
none, with its name, compatibility mode and largest storage buffer; no device
is created).

The backend `auto` (the default) calls `autoPlan()`:

- a hardware GPU if there is one; the CPU threads still run the runs the GPU
  path cannot take;
- otherwise the CPU. A software-only adapter (SwiftShader) is not used, because
  it is slower than the CPU engine.

Auto's thread count (`autoThreads()`) is half the logical cores, at most 8,
and 1 without Web Workers. The rule comes from browser measurements, recorded
in `device.ts`: LJ melt, 24 logical cores, cross-origin isolated, steps/s at
1/2/4/8/12/16 threads:

| atoms | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| 2,048 | 373 | 437 | 491 | 499 | 384 | 306 |
| 6,912 | 117 | 153 | 149 | 152 | 134 | 119 |
| 16,384 | 46 | 60 | 69 | 71 | 71 | 53 |
| 42,592 | 18 | 20 | 27 | 28 | 26 | 27 |

Throughput peaks at 4–8 threads and falls beyond 8, so the cap is 8. Half the
cores leaves the rest to the page, the 3D view and the OS. The engine's
`ready` message carries the device profile, the plan and its reason (`why`),
and the threads in use; the notebook shows them in the resource monitor.

### Resource monitor and run events

The notebook's right panel (`src/components/workbench/ResourceMonitor.tsx`)
shows the device chips, the engine in use (with Auto's reason), the live speed
and page memory, and plain-language hints from `deviceHints()` (no shared
memory, few cores, little memory, software-only WebGPU). The engine reports
progress through three events (`src/engine/types.ts`):

- `run` `{ from, to, dt, units }` at the start of each run; the notebook's
  progress bar reads "step S of T (P%)" from it.
- `perf` `{ step, atoms, stepsPerSec, elapsed, threaded }`, about twice a
  second of wall time (`PERF_EVERY_MS = 500` in `commands/run.ts`) and once at
  the end of a run. `threaded` says whether the pair term ran on threads.
- `thermo` rows as before.

The monitor turns steps/s into ms per step, atom-steps/s (atoms times
steps/s) and simulated time per day, using `simulatedPerDay()` in
`lammps/thermoUnits.ts` with the run's `dt` and units style.

### Shared-memory pair threads

`src/engine/cpu/pairThreads.ts` runs the pair term of the general engine on
threads, for every style in `THREADED_PAIRS` (`cpu/threadedPairs.ts`), when:

- the page is cross-origin isolated (the COI service worker, `src/coi.ts` and
  `public/coi-sw.js`), so SharedArrayBuffer and Atomics exist;
- the system has at least `MIN_THREADED_ATOMS` (500) owned atoms, below which
  threads cost more than they save.

Each force evaluation the engine thread copies positions, types and charges into
shared buffers (the half neighbour list is copied after each rebuild). Each force worker
(`src/workers/pair.worker.ts`) holds its own copy of the pair style. The half
list is cut into chunks of about equal work (`CHUNKS_PER_THREAD` = 8 per thread,
`splitRanges` in `pairThreadsCore.ts`); every thread, the engine thread included,
claims chunks from a shared `Atomics.add` counter until none are left, each into
its own force array, so a slow thread (a busy core, an efficiency core) takes
fewer chunks. The kernels take an optional start atom (`NeighList.ilo`) so a
chunk runs on the shared list directly. The engine thread then waits on an
Atomics counter and adds the forces, energies and virials. Everything else in
the step stays on the engine thread.

Before the chunks (PR 35) each thread had one fixed range, and the engine thread
waited for the slowest: measured in Chromium at 16,384 atoms on 8 threads, 2.1 ms
of its own work then 2.4 ms waiting, and a whole-step speedup of about 1.5x. With
the chunks the same run went from about 75 to 92.8 steps/s (2.07x over one
thread). The force reduction is still serial on the engine thread. The plain lj/cut path
(`cpu/parallel.ts`) does not use shared memory and works without isolation.

With the GPU selected, runs the GPU path cannot take (a pair style it does not run, a kspace
style, ...) use the CPU threads instead of one core (PR 34).

GPL-ported pair styles run on the same threads when `lazyStyles.ts` lists them in `GPL_THREADED_PAIRS` (`sw` first).
The pair worker's bundle holds no GPL code: the engine thread loads `engine-gpl/registry/threaded_pairs.ts` through the
lazy registries (rule 7), and the 'pair' message carries that module's own URL, which each worker imports. Such a style
can take its neighbors in native LAMMPS's order (`Pair.threadNeighbors`, made once per neighbor build on the engine
thread and copied into the shared list) and the atom IDs (`Pair.threadTags`, a shared array filled on rebuilds), since a
worker has no `SimState` or `Neighbor`. Measured in Chromium (24-core Xeon), SW silicon with 4096 atoms, 200 steps:
loop 3.13 s on one thread, 1.33 s on 8 (pair term 2.89 s and 1.14 s), the same thermo.

The full-list many-body styles `snap` (chemflag 0 and 1), `meam/spline`, `meam/sw/spline` and `vashishta` run on the
threads too: each row adds its energy (SNAP's self term included) and forces to its own atoms only, so a thread runs
the rows `[ilo, inum)` of its chunk. SNAP keeps its lazily built Wigner tables and chemflag engine outside its fields,
so each worker's structured copy builds its own. meam/spline's file holds `CubicSpline` objects inside plain arrays,
which structured cloning returns as plain objects; `Pair.threadAdopt`, called on the copy rebuilt from the cloned
fields (`pairThreadsCore.ts pairFromFactory`), gives them their class again. vashishta reads the atom IDs on the threads
(`Pair.threadTags`) to count each two-body pair at one of its rows. Measured in Chromium (24-core Xeon, shared with
other jobs at a load of about 17), 1 vs 8 threads with the same thermo: SNAP Ta06A, 2000 atoms, 20 steps 8.39 s -> 2.17 s;
vashishta SiC, 4096 atoms, 100 steps 13.5 s -> 3.96 s; meam/sw/spline Si, 2744 atoms, 100 steps 5.15 s -> 1.26 s.

`pace` (the ML-PACE port) runs on the threads through `GPL_THREADED_PAIRS`. Its rows take their neighbors in native
full-list order (`Pair.threadNeighbors`), as the evaluator's sums do. The C-tilde basis and the evaluator stay outside
the fields a worker clones (a module `WeakMap` behind accessors); the style keeps the potential text, and a worker's
copy builds its basis and evaluator from it in `threadAdopt` (the last parsed text is reused, so a run's re-sent style
is not parsed again). While `fix pair` reads the per-atom `corerep` factor, `Pair.threadable()` keeps the style on the
engine thread for that evaluation, since each worker copy would hold its own factors. Measured in Chromium (load
about 16), Cu-PBE-core-rep.ace, 864 atoms, 20 steps: 24.6 s on 1 thread, 4.1 s on 8, the same thermo.

PPPM's mesh loops run on the same threads (`force/kspace/pppmKernel.ts`, `pairThreads.ts`), each giving the
one-thread result bit for bit (`tests/enginePppmThreads.test.ts`):
- the per-atom weights and the field interpolation, in chunks of atoms;
- the charge spread, in chunks of mesh z planes. Every thread walks all atoms in index order and adds only to its own
  planes, so each mesh point receives the same additions in the same order as on one thread;
- the FFTs, each pass (x, y, z lines) split into its independent lines, with a barrier between passes, when the mesh
  has at least 16384 points (`fftMinPoints`). A smaller mesh transforms faster on one thread than over three rounds
  of messages: about 4000 points (peptide) took 0.32 s threaded against 0.26 s serial per 300 steps.

The Green's function, rebuilt every step when the box changes (fix npt), evaluates each dimension's aliased
`sinc^2P` factor once per mesh index and image instead of once per (x, y, z) image of every mesh point (the same calls,
so the same values): 393 -> 93 ms per 20 steps on rhodopsin. Measured in Chromium, rhodopsin (32000 atoms, npt) on 8
threads, 20 steps: Kspace 1.14 -> 0.71 s with pppm 1e-4; with a 72^3 mesh 8.06 -> 2.43 s (loop 16.8 -> 10.5 s).
Still serial: the k-space loop (its energy and virial sums would change order) and the Green's function itself.

`pair_style hybrid`, `hybrid/overlay` and `hybrid/molecular` run each threaded sub-style on the threads in its own
pair slot (`pairThreadsCore.ts` MAX_PAIR_SLOTS: slot 0 is a plain pair style, slot 1 + k sub-style k), so the
workers keep one copy per sub-style instead of being sent a new style every step. The force field hands the hybrid a
runner (`PairHybrid.subRunner`) for ordinary evaluations: sub-styles on its direct path (each adds to the shared
forces and accumulators) go to the threads when they can, the rest run on the engine thread. Measured in Chromium,
`hybrid/overlay lj/cut coul/cut` with 4000 atoms, 100 steps: loop 1.10 s on one thread, 0.50 s on 8 (pair term 0.82 s
and 0.29 s), the same thermo.

### Script hand-over to the Script Builder

- **Notebook to Builder:** "Open in Script Builder" joins the non-empty cells
  (`cellsToScript` in `src/lammps/notebookBridge.ts`) and imports them with the
  Builder's own `parseScript` as a new tab named "From MD Notebook"
  (`src/App.tsx`). Open tabs are kept.
- **Builder to Notebook:** "Run in Notebook" sends the Builder's script; the
  notebook shows an incoming banner with Replace cells, Add as a new cell and
  Dismiss. The button reads "Run in Notebook" from the `xl` breakpoint (1280 px)
  up and is icon-only below it (PR 25).

### Measured (Chromium 153, 24-core Xeon Silver 4310, NVIDIA A100, shared machine)

Force step, ms (bench in real Chromium; the machine carried other jobs):

| atoms | 1 thread | 2 | 4 | 8 | 12 | 16 | WebGPU (A100, compat) |
|---|---|---|---|---|---|---|---|
| 4,000 | 11.0 | 6.3 | 3.5 | 2.4 | 2.5 | 3.1 | 2.2 |
| 16,384 | 42.6 | 22.3 | 12.2 | 8.9 | 7.4 | 8.7 | 3.7 |
| 55,296 | 126.8 | 66.0 | 39.5 | 23.3 | 21.0 | 20.5 | 3.7 |

Each force worker receives only the atoms its range can reach — its own
z-layers of cells, one layer up, and layer 0 when that wraps — not every
position (`sliceRange`); that moved the point where more threads stop
helping from ~8 to ~12-16 for large systems.

Whole notebook runs (`fix nve`, live view on, a frame every 25 steps):

| atoms | 1 thread | 4 | 8 | WebGPU per step (v1) | WebGPU resident (v2) |
|---|---|---|---|---|---|
| 16,384 | 22 steps/s | 51 | 58 (6.5 cores busy) | — | 817 steps/s |
| 55,296 | — | — | — | 91 steps/s | 745–760 steps/s |

Small systems stop gaining past ~8 threads (per-step messaging dominates).
With resident stepping the A100 runs at 30–45 % SM utilisation and the
browser uses ~1.5 cores (mostly the renderer drawing frames).

## Tests

1. **Oracle parity** (`tests/engineOracle.test.ts`): every `tests/oracle/*.in`
   against native LAMMPS, as above.
2. **Physics without LAMMPS**: forces equal minus the energy gradient (central
   differences) for every pair, bonded and kspace style; the rocksalt Madelung
   constant from Ewald; PPPM error falling with assignment order; Newton's
   third law; NVE energy drift; thermostats reaching their targets.
3. **Interpreter**: every command's error paths name the command and line;
   `$`-substitution, loops, `jump`/`label`/`if`, `include`.
4. **GPU parity**, in real Chromium via SwiftShader: WGSL forces vs CPU forces,
   resident GPU steps vs fp64 steps.
5. **UI**: the notebook at phone and desktop widths in a real browser,
   including adding files and reading them with `read_data`.
