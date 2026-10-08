# In-browser MD notebook — design

Status: **engine v2, 2026-10-07.** Engine `src/engine/`, worker
`src/workers/engine.worker.ts`, UI `src/components/workbench/Notebook.tsx`.
The notebook runs real LAMMPS input scripts — data files, molecular force
fields, long-range electrostatics, minimization, variables and loops — and
every supported feature is checked against **native LAMMPS used as a
black-box oracle** (below). The CPU-threads and WebGPU paths accelerate the
plain Lennard-Jones case; everything else runs on the fp64 engine.

## Goal

Let a student run a *small* molecular-dynamics simulation in the browser:
type LAMMPS input into notebook cells (or add the data and potential files a
script reads), press Run, watch the system move in the 3D viewer and read the
thermo output — no install, no server, no account.

## What it is not

- **It is not LAMMPS.** It is an independent engine that runs a documented,
  growing subset of LAMMPS input. Every unsupported command or style is an
  error that names it and lists what is supported — never a silent no-op.
  Commands that cannot exist in a browser (`shell`, `python`, `kim`, `mdi`,
  `plugin`, `geturl`, `package`) are errors that say why.
- **It contains no LAMMPS source code.** LAMMPS is GPL-2.0; this project is
  under its own source-available licence, so porting LAMMPS code would be a
  licence violation. The engine is written from textbook physics (Allen &
  Tildesley; Frenkel & Smit; Hockney & Eastwood; Deserno & Holm) and from the
  *documented semantics* on docs.lammps.org, quoted in code comments — never
  from the LAMMPS source tree. Where the documentation is silent, behaviour
  is **measured** by running the native binary on small inputs, and the
  measurement is written next to the code (e.g. the PPPM alias-image count,
  Ewald's cutoff sphere, `2/sqrt(pi)` to 8 digits in the real-space force).
- It is not for production science or large systems. The target is
  ≤ ~20 000 atoms, short runs, teaching.

## Verification — native LAMMPS as a black-box oracle

`tests/oracle/*.in` are deterministic inputs: positions from lattices plus
analytic displacements, velocities from atom-style variables, no random
seeds. `scripts/oracle/run-oracle.mjs` runs each one through a native `lmp`
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
TotEng −2.2744931, Press −3.7033504) exactly. Later steps differ only because
`velocity create` uses our own documented RNG.

## Supported subset

The authoritative list is what the engine reports: the notebook's help panel
shows the commands and styles, and an unsupported one is an error naming it.
The block below is checked against the engine's registries by
`tests/notebookDocs.test.ts`.

<!-- coverage:begin -->
| Kind | Supported |
|---|---|
| commands | `angle_coeff` `angle_style` `atom_modify` `atom_style` `balance` `bond_coeff` `bond_style` `boundary` `change_box` `clear` `comm_modify` `comm_style` `compute` `compute_modify` `create_atoms` `create_bonds` `create_box` `delete_atoms` `delete_bonds` `dielectric` `dihedral_coeff` `dihedral_style` `dimension` `displace_atoms` `dump` `dump_modify` `echo` `fix` `fix_modify` `group` `if` `improper_coeff` `improper_style` `include` `info` `jump` `kspace_modify` `kspace_style` `label` `lattice` `log` `mass` `min_modify` `min_style` `minimize` `molecule` `neigh_modify` `neighbor` `newton` `next` `pair_coeff` `pair_modify` `pair_style` `partition` `print` `processors` `quit` `read_data` `read_dump` `read_restart` `region` `replicate` `rerun` `reset_timestep` `restart` `run` `run_style` `set` `special_bonds` `suffix` `thermo` `thermo_modify` `thermo_style` `timer` `timestep` `uncompute` `undump` `unfix` `units` `variable` `velocity` `write_data` `write_dump` `write_restart` |
| pair_style | `atm` `born` `born/coul/long` `born/coul/long/cs` `buck` `buck/coul/cut` `buck/coul/long` `buck/coul/long/cs` `colloid` `coul/cut` `coul/debye` `coul/dsf` `coul/long` `coul/long/cs` `coul/wolf` `eam` `eam/alloy` `eam/fs` `eim` `gauss` `gran/hertz/history` `gran/hooke` `gran/hooke/history` `granular` `hbond/dreiding/lj` `hbond/dreiding/lj/angleoffset` `hbond/dreiding/morse` `hbond/dreiding/morse/angleoffset` `hybrid` `hybrid/molecular` `hybrid/overlay` `hybrid/scaled` `lj/charmm/coul/charmm` `lj/charmm/coul/charmm/implicit` `lj/charmm/coul/long` `lj/charmmfsw/coul/charmmfsh` `lj/charmmfsw/coul/long` `lj/class2` `lj/cubic` `lj/cut` `lj/cut/coul/cut` `lj/cut/coul/debye` `lj/cut/coul/dsf` `lj/cut/coul/long` `lj/cut/coul/wolf` `lj/cut/tip4p/cut` `lj/cut/tip4p/long` `lj/expand` `lj/gromacs` `lj/long/coul/long` `lj/relres` `lj/sf` `lj/smooth` `lj/smooth/linear` `lj96/cut` `meam` `mie/cut` `morse` `nb3b/harmonic` `nb3b/screened` `snap` `soft` `sw` `sw/mod` `table` `tersoff` `tersoff/mod` `tersoff/mod/c` `tersoff/zbl` `tip4p/cut` `tip4p/long` `vashishta` `vashishta/table` `yukawa` `yukawa/colloid` `zbl` `zero` |
| bond_style | `class2` `fene` `fene/expand` `gromos` `harmonic` `harmonic/shift` `harmonic/shift/cut` `morse` `nonlinear` `zero` |
| angle_style | `charmm` `cosine` `cosine/delta` `cosine/periodic` `cosine/shift` `cosine/squared` `fourier` `fourier/simple` `harmonic` `quartic` `zero` |
| dihedral_style | `charmm` `charmmfsw` `cosine/shift/exp` `fourier` `harmonic` `helix` `multi/harmonic` `nharmonic` `opls` `quadratic` `zero` |
| improper_style | `cossq` `cvff` `distance` `fourier` `harmonic` `umbrella` `zero` |
| kspace_style | `ewald` `ewald/disp` `pppm` `pppm/tip4p` |
| fix | `accelerate/cos` `adapt` `addforce` `ave/atom` `ave/chunk` `ave/correlate` `ave/grid` `ave/histo` `ave/time` `aveforce` `balance` `box/relax` `cmap` `controller` `deform` `deposit` `efield` `ehex` `enforce2d` `evaporate` `freeze` `gjf` `gravity` `heat` `indent` `langevin` `lineforce` `momentum` `nph` `nph/sphere` `npt` `npt/sphere` `nve` `nve/limit` `nve/noforce` `nve/sphere` `nvt` `nvt/sllod` `nvt/sphere` `planeforce` `pour` `print` `property/atom` `rattle` `recenter` `restrain` `rigid` `rigid/nve` `rigid/nve/small` `rigid/nvt` `rigid/nvt/small` `rigid/small` `setforce` `shake` `spring` `spring/rg` `spring/self` `temp/berendsen` `temp/csld` `temp/csvr` `temp/rescale` `thermal/conductivity` `ttm` `ttm/grid` `vector` `viscosity` `viscous` `wall/gran` `wall/gran/region` `wall/harmonic` `wall/lj1043` `wall/lj126` `wall/lj93` `wall/morse` `wall/reflect` `wall/reflect/stochastic` `wall/region` |
| compute | `aggregate/atom` `angle/local` `bond/local` `centro/atom` `chunk/atom` `cluster/atom` `cna/atom` `com` `com/chunk` `coord/atom` `dihedral/local` `displace/atom` `erotate/sphere` `event/displace` `fragment/atom` `gyration` `heat/flux` `hexorder/atom` `improper/local` `ke` `ke/atom` `msd` `msd/chunk` `msd/nongauss` `orientorder/atom` `pair` `pair/local` `pe` `pe/atom` `pressure` `property/atom` `property/local` `rdf` `reduce` `reduce/region` `sna/atom` `sna/grid` `sna/grid/local` `snad/atom` `snap` `snav/atom` `stress/atom` `temp` `temp/chunk` `temp/com` `temp/cs` `temp/deform` `temp/partial` `temp/ramp` `temp/region` `temp/sphere` `vacf` `viscosity/cos` `voronoi/atom` |
<!-- coverage:end -->

Not supported (an error names each): native binary restart files (`write_restart` / `read_restart`
use the engine's own format), `atom_style` other than
atomic / charge / bond / angle / molecular / full / sphere, `pppm/disp`, `msm`,
`kspace_modify diff ad`, `run_style`, atom-style `hybrid/scaled` factors.

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
  cpu/, gpu/                        threaded CPU and WebGPU force paths (plain lj/cut)
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
