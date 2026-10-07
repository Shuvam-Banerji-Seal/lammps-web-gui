# In-browser MD notebook (WebGPU) — design

Status: **implemented (v1), 2026-10-07.** Engine `src/engine/`, worker
`src/workers/engine.worker.ts`, UI `src/components/workbench/Notebook.tsx`.
Verified: the melt step-0 line below matches on the CPU (fp64) and WebGPU
paths; GPU forces match CPU within 1e-4 (real Chromium, SwiftShader);
the notebook UI passes a 16-check real-browser test at desktop and phone
widths.

## Goal

Let a student run a *small* molecular-dynamics simulation in the browser:
type LAMMPS-style input into notebook cells, press Run, watch the system move
in the existing 3D viewer and read the thermo output — no install, no server,
no account. Force evaluation runs on the GPU through **WebGPU** when the
browser offers it, and on the CPU otherwise.

## What it is not

- **It is not LAMMPS.** It is an independent engine that understands a
  documented subset of LAMMPS input syntax. Every unsupported command is a
  clear error that names the command and lists what is supported — never a
  silent no-op.
- **It contains no LAMMPS source code.** LAMMPS is GPL-2.0; this project is
  under its own source-available licence, so porting LAMMPS code would be a
  licence violation. The engine is written from the textbook physics (Allen &
  Tildesley, *Computer Simulation of Liquids*; Frenkel & Smit, *Understanding
  Molecular Simulation*) and from the *documented semantics* on
  docs.lammps.org — never from the LAMMPS source tree.
- It is not for production science or large systems. The target is
  ≤ ~20 000 atoms, short runs, teaching.

## Acceptance test — real LAMMPS output

The first thing the engine must do is reproduce the official `examples/melt`
run (`in.melt`: fcc LJ lattice, ρ* = 0.8442, 4000 atoms, `velocity create 3.0`,
`lj/cut 2.5`, `fix nve`, `run 250`). LAMMPS's own published log
(`log.8Apr21.melt.g++.1`) reports:

| Step | Temp | E_pair | TotEng | Press |
|---|---|---|---|---|
| 0 | 3 | −6.7733681 | −2.2744931 | −3.7033504 |
| 50 | 1.6842865 | −4.8082494 | −2.2824513 | 5.5666131 |
| 250 | 1.6645597 | −4.7774327 | −2.2812174 | 5.7526089 |

Step 0 depends only on the lattice positions and the exactly-rescaled
temperature, so it is **deterministic** and must match to the printed
precision (CPU fp64). Later steps diverge because our velocity RNG differs
from LAMMPS's; they are matched **statistically** (Temp 1.55–1.75,
Press 5.2–6.2 after step 50, TotEng drift < 0.01 per atom over 250 steps).

Consistency check of the reference itself: with 3N − 3 degrees of freedom the
per-atom kinetic energy at T = 3 is 1.5 · 3 · 3999/4000 = 4.49887, and
−6.7733681 + 4.49887 = −2.27450 ✓.

## Supported subset (v1)

| Command | Supported form | Notes |
|---|---|---|
| `units` | `lj`, `real`, `metal` | constants cited from docs.lammps.org/units.html |
| `dimension` | `2`, `3` | |
| `boundary` | `p p p` (2D: `p p p` with thin z) | non-periodic boundaries are a clear error in v1 |
| `atom_style` | `atomic` | |
| `lattice` | `sc bcc fcc hcp diamond sq sq2 hex` + scale | reduced density in lj, lattice constant otherwise |
| `region` | `block` (lattice or box units) | |
| `create_box` | `N region` | |
| `create_atoms` | `type box`, `type region ID`, `type random N seed region` | |
| `mass` | `type value`, `* value` | |
| `velocity` | `group create T seed [dist gaussian|uniform] [mom yes] [rot yes] [loop all|geom|local]`, `scale T`, `set` | `loop` accepted; values are reproducible from our own RNG, not LAMMPS's |
| `pair_style` | `lj/cut rc` | |
| `pair_coeff` | `i j eps sigma [rc]`, wildcards | |
| `pair_modify` | `shift yes|no` | default `no`, as in LAMMPS |
| `neighbor`, `neigh_modify` | accepted | the engine always evaluates the exact cutoff; documented |
| `timestep` | `dt` | defaults per units style |
| `fix` | `nve`, `langevin T0 T1 damp seed`, `temp/berendsen`, `temp/rescale`, `nvt temp T0 T1 damp` (Nosé–Hoover chain of 3, the documented `tchain` default), `enforce2d` | |
| `thermo` | `N` | |
| `thermo_style` | `custom step temp pe ke etotal press vol density` (+ `one`) | |
| `run` | `N` | |
| `dump` | `ID group atom|custom N file …` | streams frames to the viewer; downloadable as `.lammpstrj` |
| `write_data` | `file` | download |
| `print`, `variable … equal <expr>`, `${name}` / `$x` | | numeric expressions only, via a safe parser (no `eval`) |

## Architecture

```
src/engine/
  types.ts           contracts shared by every piece below      (orchestrator)
  units.ts           unit systems + constants
  rng.ts             seeded RNG (deterministic, documented)
  lattice.ts         lattice generation, create_atoms in regions
  cpu/forces.ts      LJ forces, energy, virial; cell list; minimum image
  integrate.ts       velocity Verlet + thermostats + enforce2d
  observables.ts     temp (dof-aware), ke, pe, press (virial), density
  interpreter.ts     LAMMPS-subset parsing → engine operations
  gpu/               WebGPU force backend (WGSL) + device lifecycle
  engine.worker.ts   runs a session off the main thread
src/components/workbench/Notebook.tsx   the 4th module
```

### Backends

One interface, two implementations:

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

## Verification plan

1. **Unit physics (CPU):** two-atom LJ force/energy vs the analytic formula;
   cell-list forces identical to brute force; Newton's third law (net force
   ≈ 0); NVE energy drift bounded; momentum conserved; each thermostat reaches
   its target within tolerance.
2. **Reference reproduction:** run the real `in.melt` and match the table above.
3. **GPU parity:** in real Chromium via SwiftShader — WGSL forces vs CPU forces.
4. **Interpreter:** every supported command parses; every unsupported one
   errors with its line number; `${var}` substitution; no `eval`.
5. **UI:** the notebook at 360 px and desktop in a real browser.
