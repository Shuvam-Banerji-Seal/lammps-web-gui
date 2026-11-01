<div align="center">

# LAMMPS Web GUI

**Run LAMMPS molecular dynamics in your browser — build input scripts visually, run them, plot thermo output and explore structures and trajectories in 3D.**

*No install, no account, nothing uploaded · free for education and non-commercial research · attribution required · [commercial use needs a licence](COMMERCIAL.md)*

[![Live Demo](https://img.shields.io/badge/▶_Open_the_app-shuvam--banerji--seal.github.io-blue?style=for-the-badge)](https://shuvam-banerji-seal.github.io/lammps-web-gui/)
[![Watch the film on YouTube](https://img.shields.io/badge/▶_Watch_the_film-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/bqGnbRRuwa8)
[![CI](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/actions/workflows/ci.yml/badge.svg)](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/actions/workflows/ci.yml)
[![Deploy](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/actions/workflows/deploy.yml/badge.svg)](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/actions/workflows/deploy.yml)
[![Native parity cases](https://img.shields.io/badge/native_LAMMPS_parity_cases-2027-brightgreen)](#-checked-against-native-lammps)
[![LAMMPS examples](https://img.shields.io/badge/LAMMPS_examples_that_run-650_of_882-brightgreen)](#-checked-against-native-lammps)
[![License: Educational / Non-Commercial](https://img.shields.io/badge/License-Educational%20%2F%20Non--Commercial-blue.svg)](LICENSE)
[![Commercial licence](https://img.shields.io/badge/Commercial_use-licence_required-orange.svg)](COMMERCIAL.md)

</div>

LAMMPS Web GUI is a browser-based workbench for [LAMMPS](https://www.lammps.org/), the molecular dynamics code from Sandia National Laboratories. It has four modules:

| Module | What it does |
|---|---|
| 🧮 **MD Notebook** | Type LAMMPS input into notebook cells and press Run. Thermo output fills a table and live charts while the atoms move in a 3D view. Forces run on the GPU (WebGPU) or on CPU threads. `dump image` frames, data files and restart files are written in the browser. |
| 🧩 **Script Builder** | Build LAMMPS input scripts as a flowchart from 185 commands, branch alternative ideas, import existing `in.*` files, and check the script against rules that each cite their docs.lammps.org page. |
| 🏗️ **Compiler Helper** | Pick packages and accelerators and get a ready-to-run CMake build script for native LAMMPS. |
| 🔬 **Structure Viewer** | GPU-instanced 3D viewer for LAMMPS data and dump files, XYZ, PDB and CIF, with trajectory playback, measurements and RDF/MSD analysis. |

### 🧪 Checked against native LAMMPS

The notebook's engine is written in TypeScript for the browser. Its results are compared with native LAMMPS (the same source commit, b332f28) on **2027 parity cases**: thermo output and per-atom positions, velocities and forces are compared at 1e-8 relative or tighter in most cases (each case states its tolerance and why when it is looser). Of the **894 input scripts in the LAMMPS `examples/` tree, 702 run** in the browser (runs shortened to a few steps; the rest need native libraries, Python, MPI partitions or features not ported yet, and stop with an error that names them). Styles ported from the LAMMPS C++ source live in a separate GPL-2.0 module, [`src/engine-gpl`](src/engine-gpl/README.md).

---

## 🎬 Demo

[![LAMMPS in your browser: watch the LAMMPS Web GUI film on YouTube](public/videos/lammps-web-gui-thumbnail.jpg)](https://youtu.be/bqGnbRRuwa8)

**▶ [Watch the film on YouTube (2 min 47 s, with chapters)](https://youtu.be/bqGnbRRuwa8)**, or the [MP4 with sound](https://shuvam-banerji-seal.github.io/lammps-web-gui/videos/lammps-web-gui-film.mp4) (1920×1080, H.264/AAC; also in the repo at [`public/videos/lammps-web-gui-film.mp4`](public/videos/lammps-web-gui-film.mp4)).

<details><summary>Animated preview</summary>

[![LAMMPS Web GUI film: a granular pour, an oxDNA ring, micelles, a running crack and a solvated peptide simulated by the app's engine, then how the app works, how it is checked against native LAMMPS, how it uses your own CPU and GPU, its themes and its four modules](docs/videos/lammps-web-gui-film-preview.webp)](https://youtu.be/bqGnbRRuwa8)

</details>

*In seven parts:*

- ***Cold open*** *— five simulations run by the app's own engine and rendered as 3D instanced atoms: a granular pour (2,814 spheres, `fix pour`, `gran/hooke/history`), a 74-base-pair oxDNA2 ring under Langevin dynamics, surfactants assembling into micelles, a crack running through a Lennard-Jones solid (8,141 atoms) and a CHARMM peptide in water (PPPM, SHAKE).*
- ***How it works*** *— the pipeline in one browser tab: the Script Builder writes the input, the interpreter reads it line by line as LAMMPS does, styles load on first use, forces run on your device, the integrator steps and the 3D viewer shows the atoms. No server, no install, no upload.*
- ***True to LAMMPS*** *— the script check that cites docs.lammps.org, the `examples/melt` thermo output printed digit for digit as native LAMMPS prints it, the native parity cases, the GPL-2.0 module of styles ported from the LAMMPS C++, and the C library math (`log`, `sin`, `cos`, `exp`, `pow`) ported from glibc, so the engine computes them as the native binary does rather than with the browser's `Math` functions, which differ between browser versions.*
- ***Your hardware*** *— worker threads sharing one copy of the atoms (`SharedArrayBuffer`), and WebGPU compute shaders where the browser has them.*
- ***Themes*** *— the five themes side by side.*
- ***Tour*** *— the Script Builder, the CMake compiler helper, the Structure Viewer and the MD Notebook.*
- ***End card*** *— a QR code that opens the app, and the links to the app, the source and the LAMMPS manual.*

*Cut with [Remotion](https://www.remotion.dev/) and three.js, scored with original synthesized music. The app footage is the production build captured in headless Chromium.*

## 📸 Screenshots

### MD Notebook — LAMMPS input running in the browser

| Lennard-Jones melt: cells, thermo table, 3D view | Notebook with markdown notes and several cells |
|---|---|
| ![MD Notebook running a Lennard-Jones melt: input cells, thermo output table, 3D view of 500 atoms and the device panel](docs/screenshots/notebook-run.png) | ![MD Notebook with a markdown cell and several LAMMPS input cells](docs/screenshots/notebook-script.png) |

<p align="center"><img src="docs/screenshots/notebook-monitor.png" width="360" alt="Device and performance panel: cores, memory, shared-memory threads, the engine Auto chose, steps per second and atom-steps per second"><br><em>Device and performance panel: what Auto chose and why, steps/s, ms/step and atom-steps/s.</em></p>

**Example notebooks** (notebook menu → *Open an example…*) show how to load external molecules, each with its input
files: an SPC/E water box from a LAMMPS data file, CO2 and n-butane placed from molecule templates with
`create_atoms ... mol`, and crystals imported from CIF files with **Import structure…**: alpha-quartz (BKS potential),
proton-ordered ice Ih (rigid SPC/E water) and urea, whose molecules cross the cell faces. The import infers bonds
through the periodic cell and writes image flags, so every molecule stays whole.

### Script Builder — LAMMPS input scripts as a flowchart

| Starter template with the generated script | Concept branching — one flowchart, several ideas |
|---|---|
| ![Script Builder with the LJ fluid template, generated LAMMPS input script and the script check](docs/screenshots/script-builder.png) | ![Script Builder flowchart with a concept branch forked after atom_style](docs/screenshots/flowchart-branching.png) |
| **Script check — every rule cites docs.lammps.org** | **Starter templates** |
| ![Script check panel showing an error with its docs.lammps.org link](docs/screenshots/script-check.png) | ![Starter pipeline templates menu: LJ fluid, EAM metal, ReaxFF, granular pouring, NEMD shear](docs/screenshots/templates-menu.png) |
| **Flowchart overview** | **Compiler Helper — CMake build script for native LAMMPS** |
| ![Script Builder overview with the command palette and flowchart](docs/screenshots/app-desktop.png) | ![Compiler Helper with the Materials / metals preset and the generated CMake build script](docs/screenshots/compiler-flags.png) |

### Structure Viewer — LAMMPS data and dump, XYZ, PDB, CIF

| C60 with the sidebar | LAMMPS dump trajectory playback |
|---|---|
| ![Structure Viewer showing a C60 fullerene](docs/screenshots/viewer-c60.png) | ![Structure Viewer playing back a Lennard-Jones melt dump trajectory](docs/screenshots/viewer-dump-playback.png) |
| **Trajectory analysis — RDF and MSD** | **Unit cell (NaCl, CIF)** |
| ![Analysis tab with the radial distribution function and mean squared displacement of a LAMMPS dump](docs/screenshots/analysis-desktop.png) | ![NaCl crystal from a CIF file with its unit cell box](docs/screenshots/simulation-box.png) |
| **Keyboard shortcuts** | **Warm paper theme** |
| ![Keyboard shortcut overlay of the Structure Viewer](docs/screenshots/shortcuts-overlay.png) | ![The app in the Warm paper theme](docs/screenshots/app-light.png) |

<p align="center"><img src="docs/screenshots/viewer-representations.png" alt="The solvated peptide example in the Lines, Cubes, Glass and Flat discs representations"><br><em>Four of the ten representations on the solvated peptide example: Lines, Cubes, Glass and Flat discs (<kbd>V</kbd> cycles through all of them).</em></p>

### On a phone

| MD Notebook | Trajectory analysis | Viewer dock |
|---|---|---|
| ![MD Notebook on a phone after a run](docs/screenshots/app-mobile.png) | ![Trajectory analysis on a phone](docs/screenshots/analysis-mobile.png) | ![Structure Viewer dock on a phone](docs/screenshots/viewer-dock-mobile.png) |

## 🎞️ Simulation videos — rendered in the browser

The viewer plays **LAMMPS dump** (`.lammpstrj`/`.dump`) natively and exports **MP4/WebM at 60 fps** — no Python needed. Load any `.lammpstrj` via drag-drop or **Paste data → DUMP → Visualize** → scrub with the playback bar (`P` / `,` / `.`, 2–30 fps) → **● Rec** to save MP4.

## ✨ Features

- 🧮 **MD Notebook — LAMMPS input in the browser, on the GPU or on CPU threads.** Type LAMMPS input into notebook cells, press Run, and watch the system move in the 3D view while thermo output fills a table and live temperature / energy charts. **Auto / CPU / GPU** chooses where forces run: WebGPU (WGSL kernels; hardware adapters are preferred and a software-only adapter is declined with a note) or an fp64 CPU engine split over Web Worker threads, with shared-memory threads when the page is cross-origin isolated. The GPU runs the pair term of the Lennard-Jones (12-6, 9-6 and class 2), Buckingham, Born-Mayer-Huggins, Morse, soft and Yukawa styles with cut or long-range Coulomb, `lj/charmm/coul/long`, tabulated `table` potentials, `eam`, `eam/alloy` and `eam/fs`, the three-body Stillinger-Weber `sw` and `sw/mod`, the Tersoff bond-order `tersoff`, and the TIP4P water styles (`lj/cut/tip4p/long`, `lj/cut/tip4p/cut`, `tip4p/long`, `tip4p/cut`); bonded terms and kspace styles (Ewald, PPPM, `pppm/tip4p`) run on the CPU threads beside it, so molecular and charged systems use the GPU too. Plain `fix nve` runs of the non-bonded styles stay resident on the GPU between output steps, and **Stop** cancels a run at any step. Measured in Chromium on a 24-core Xeon with an NVIDIA A100: a force step for 16,384 atoms takes 42.6 ms on 1 thread, 12.2 ms on 4 and 8.9 ms on 8; a live run of 55,296 atoms goes at ~750 steps/s on the GPU. The engine covers a large part of LAMMPS: pair, bond, angle, dihedral, improper and kspace styles (Ewald, PPPM and more), thermostats and barostats, rigid bodies, SHAKE/RATTLE, GCMC and other Monte Carlo fixes, `fix bond/react`, deposits and walls, computes, `minimize`, NEB with notebook partitions, `variable`/`if`/`label`/`jump`, `read_data`/`write_data`, binary restart files (its own and native LAMMPS's), `dump` (atom, custom, cfg, xyz, extxyz, local, yaml, vtk, grid, image, the binary dcd and xtc trajectories, and HDF5-based h5md) and `rerun`. A command that is not supported stops the run with an error that names the line and the command. See [Checked against native LAMMPS](#-checked-against-native-lammps) and the design notes in [`docs/design/notebook.md`](docs/design/notebook.md).

- 🌿 **Concept branching — divergent ideas in one flowchart.** Hover any step and hit the branch icon to *fork the pipeline from that point onward*. The new concept starts as a **copy of the rest of your pipeline**, so you edit a variant instead of rebuilding from scratch: swap the thermostat, try a different pair style, extend the production run. Switch between **Main line** and each concept with one click — the generated script, the warnings and the SVG export all follow whichever concept is taken. A concept either **replaces** everything after the fork (truly divergent) or **rejoins** the main line (a detour), and the steps it cut off stay visible, ghosted, under *"not in this concept"*. When one idea wins, **Promote** folds it into the main line and drops its rivals; **Tab** flattens any concept into its own flowchart tab for side-by-side comparison. Several forks can be open at once.
- ✅ **Script check — validation against the LAMMPS rules, not vibes.** A linter reads the *final* script (generated or hand-edited) and flags what LAMMPS would reject or silently do wrong: header-only commands used after the box (`units`, `dimension`, `boundary`, `atom_style`), commands used before it (`pair_coeff`, `mass`, `velocity`, `create_atoms`), `unfix`/`undump`/`uncompute` of an ID that was never defined, fix/dump/compute ID reuse with a different style, groups and regions referenced before definition, a `run` with no time-integration fix, `pair_style reaxff` without charge equilibration, and kspace/pair-style mismatches. **Every rule quotes the `docs.lammps.org` page it enforces and links to it**, and the whole rule set is verified silent on the official example scripts (`in.melt`, `in.heatflux`, `in.msd.2d`, `in.cos.1000SPCE`) — a linter that cries wolf on `in.melt` would be worse than none. Every shipped template is asserted to pass with zero errors *and* zero warnings.
- ⬇️ **Pipeline-order emission.** The script comes out in exactly the order the flowchart shows, because LAMMPS executes an input file top to bottom and the docs are explicit that the settings/run phases *"can be repeated as many times as desired"*. That makes multi-stage scripts work: `run` → `reset_timestep` → `run`, a second `dump` opened after a stage, and `write_data` **after** the run rather than before it. A **Sort** action still reorders into canonical section order on demand — it moves the steps, so the flowchart never disagrees with the script.
- 🧪 **Four integrated modules**: the **MD Notebook** above, a visual **Script Builder** (branching flowchart pipeline over 185 curated commands — the complete general-command surface, locked by a coverage test against docs.lammps.org), a **Compiler Helper** (package/accelerator presets → ready-to-run CMake build scripts) and the 3D **Structure Viewer**. The Script Builder and the MD Notebook pass scripts both ways: **Run in Notebook** sends the Builder's script to the notebook, and **Open in Script Builder** opens the notebook's cells as a new Builder tab (flowchart, script check, export).
- 📟 **Device detection, Auto and a resource monitor.** The engine probes the device (logical cores, memory, Web Workers, shared memory, and the WebGPU adapter: hardware, software or none). The notebook's right panel shows those chips, the engine in use and why Auto chose it, and the live steps/s, ms/step, atom-steps/s and simulated time per day (ns/day etc. from the units style), with a steps/s sparkline, page memory and hints about what limits the device.
- 🖼️ **`dump image` in the notebook.** PNG, JPEG and PPM frames written by `dump image` appear in a preview with a frame slider, and each file has a Download button. `dump movie` is not supported (no video encoder in the browser).
- 🧭 **Notebook usability.** A progress bar shows `step S of T` while a cell runs. Thermo tables carry a units caption and a tooltip per column. Reset asks for a second click when it would clear output, and a deleted cell can be restored with Undo. Unknown command errors suggest a "Did you mean …?" fix.
- 🗄️ **Browser storage panel.** The drive button in the header opens a drawer with every file kept in this browser (IndexedDB): search, preview, download, delete, and add files with the picker or by dropping them on the drawer. It shows the site's storage quota and asks the browser to keep the files (*Keep files*). Each file opens where it belongs: a LAMMPS script in the **Notebook** or the **Script Builder** (flowchart), a structure, data or dump file (also gzipped) in the **Viewer**. The notebook reads stored files by name in `read_data`, `include` and `pair_coeff`, and the Script Builder's file fields (`read_data`, `read_restart`, `molecule`, `read_dump`, `rerun`) suggest them.
- ⛶ **Full-screen simulation view and drag and drop.** The notebook's view has a full-screen button (Esc leaves it) that keeps the playback and render controls. Drop files anywhere on the notebook: a notebook or LAMMPS script opens, every other file (data, potentials, structures, gzip) is added as an input file. Drop scripts on the Script Builder to open them as new tabs (every dropped file is also kept in browser storage).
- 📈 **More analysis and exports in the notebook.** Besides MSD, g(r), density and speeds: the **coordination number** histogram (cutoff at the first minimum of g(r), editable), the **bond-angle distribution**, the **structure factor S(q)** from g(r), and per-column thermo statistics (**block-average error**, **autocorrelation**, **histogram**), each with SVG, PNG and CSV export, plus one JSON file with every series. Trajectories download as extended XYZ, LAMMPS dump, multi-model **PDB**, **CSV**, or the shown frame as a **LAMMPS data** file; dumps and `write_data` with a `.gz` name are written gzipped.
- 🧪 **Wider engine coverage.** Monte Carlo: `fix widom` (atoms or molecules), `fix mol/swap`, `fix gcmc` (every unit style) and `fix hmc`. Constraints: `fix ehex` with `constrain` and `constrain com`, `fix shake`/`rattle` with `mol`, `fix deposit` with `mol` (including `rigid` and `shake` on the inserted molecules), and `rigid/small` with `mol` (`fix rigid` itself refuses `mol`, as the docs say). `shell rm` and `shell mv` on the in-browser files. `voronoi/atom` in 2d, triclinic and non-periodic boxes (a non-periodic triclinic box is refused). MEAM: 2NN (`nn2 = 1`), `sc` and `hcp` reference lattices, `ialloy`, and multi-element `erose_form` with `dia` elements. `fix wall/region` with a `plane` region. Unsupported forms still stop with an error that names them. Coverage: 650 of the 882 input scripts in the LAMMPS `examples/` tree run in the engine (every `run` and `minimize` shortened to a few steps; measured 2026-10-15).
- 💾 **Everything persists** — your script pipeline, compiler options, active module and colour theme (Coffee & sage, Warm paper, Midnight, Solarized light or High contrast, from the header's theme picker) survive module switches *and* page reloads (local-only, nothing uploaded).
- 🧩 **Editable flowchart** — grab any card and drop it between two others (it locks into place with a live insertion indicator), pan by dragging the background, zoom with the mouse wheel (15–250%), press **F** to fit the whole pipeline (a long one becomes a bird's-eye overview), click any **connect** pill to insert a command at that exact spot. Prefer raw text? **Edit script** switches to a hand-editing mode that emits your text verbatim.
- 📥 **Import existing scripts** — load any `in.*` LAMMPS input and the flowchart builds itself: commands are matched back to editable parameter forms (user-chosen fix/dump/compute IDs handled), unknown lines are preserved verbatim as raw steps — nothing is lost.
- 🖼️ **Presentable flowchart export** — one click downloads the pipeline as a crisp **SVG** or 2× **PNG** (themed, titled, dated) for papers, slides and READMEs.
- 🏗️ **Compiler Helper with the full flag surface** — 46 CMake build options (FFT / Kokkos-FFT / heFFTe back ends, OpenMP, GPU API·precision·arch, Kokkos layout and precision, gzip/curl/JPEG/PNG/FFMPEG, sizes, memalign, exceptions, rpath, potential and KIM/Voro/QUIP/Eigen downloads, the LAMMPS GUI…), every one with a click-to-read description taken from the build docs, plus all 71 packages of `cmake/presets/most.cmake` (+ GPU/INTEL/KOKKOS) with descriptions checked against each package's own docs page.
- ↩️ **Undo / redo** — every builder action is undoable (`Ctrl+Z` / `Ctrl+Shift+Z`), including template loads; duplicate any step with its parameters; `Delete` removes the selected step.
- 📋 **Starter templates** — one-click pipelines for LJ fluid NVT, EAM metal relax+NPT, ReaxFF chemistry, granular pouring and NEMD shear — all generate warning-free scripts.
- 📚 **185-command library** — every general LAMMPS command in the official index, locked by a coverage test diffed against the *live* `docs.lammps.org/Commands_all.html` (which drifts both ways: `fenix` was added, `box` was removed in 22Dec2022 and is now hidden from the palette but still importable), rich editors for the popular fix/compute styles, and free-form any-style `fix`/`dump`/`compute` escape hatches covering the ~900 style surface — setup, system, interactions, output and run-control commands with per-parameter editors and links to the official docs; style enumerations verified against docs.lammps.org.
- 🔍 **CMake flag inspector** — the Compiler Helper lists *every* `-D` flag as an expandable chip set; click any flag to read what it does and where to change it.
- 🌰 **Warm light/dark themes** — a coffee-and-sage dark palette (no blue tint, no gradients) and a warm-paper light theme, switchable app-wide.
- 🧊 **True 3D rendering** — depth-correct perspective view of atoms, bonds and the simulation cell, powered by three.js / React Three Fiber.
- 📐 **Measurement tools** — click 2–4 atoms for live distance / angle / dihedral readouts with on-canvas overlays; in periodic cells the selection is joined through nearest periodic images.
- 🎞️ **Trajectory playback** — multi-frame XYZ **and LAMMPS dump** files get a scrubber with play/pause and 2–30 fps speeds (`P`, `,`, `.`).
- 📊 **Trajectory analysis** — the **Analysis** tab for dump trajectories: **RDF g(r)** (2D/3D PBC-aware), **MSD vs lag** (minimum-image, time-origin averaged, ID-following), **density profiles**, **speed histograms** (if `vx vy vz`), all as SVG charts with CSV export. Runs in a worker, once per structure. The physics is unit-tested, not just eyeballed: `g(r)→1` for a random gas, first and second shells of a simple-cubic lattice at `a` and `a√2`, MSD exactly `(v·t)²` under uniform drift, zero for a frozen structure, and correct across a periodic wrap.
- 🔬 **LAMMPS dump rendering** — all four documented coordinate forms (`x y z`, `xu yu zu`, `xs ys zs` **and** `xsu ysu zsu`), optional `vx vy vz`, thin 2D boxes (`-0.5→0.5`) with correct PBC, and **a per-frame cell** so an NPT box that breathes is drawn correctly instead of frozen at frame 0. `ix iy iz` image flags are read and kept separate from the rendered position: rendering stays wrapped (an unwrapped diffusing system scatters across box images), while **MSD unwraps with them** — without flags the minimum-image convention caps MSD near (L/2)², so it can never show linear diffusion, and the Analysis tab now says which mode it is using instead of quietly showing an artefact plateau.
- 🔗 **Shareable views** — the Share button copies a URL that restores your exact visualization settings.
- 📁 **Four structure formats**
  | Format | Extensions | Highlights |
  |---|---|---|
  | LAMMPS data | `.data` `.lmp` `.lammps` | all 23 documented `read_data` atom styles (with or without the `# style` hint), image flags, `Atom Type Labels`, box bounds incl. triclinic tilt (`xy xz yz`) |
  | XYZ | `.xyz` | multi-frame trajectories, **extended XYZ** `Lattice` / `Properties` (periodic cell per frame), chemistry-aware bond inference (never for noble gases) |
  | Protein Data Bank | `.pdb` `.ent` | NMR `MODEL`s as trajectory frames, alternate locations collapsed, elements per the wwPDB name alignment, fixed-width `CONECT` bonds, `CRYST1` unit cell |
  | CIF | `.cif` `.mmcif` | symmetry operations applied to the asymmetric unit, fractional ↔ Cartesian conversion, triclinic cells as true parallelepipeds |
- 🎨 **All 118 elements** resolved by symbol from *any* format and colored with standard **CPK/Jmol** palettes; per-type recoloring and sizing in the sidebar.
- 🌈 **Colour by quantity** — charge, speed |v|, atom ID, molecule or x/y/z position through viridis, plasma, inferno, turbo, coolwarm, rainbow or grayscale (reversible), with the range taken over every trajectory frame (or set by hand) and an on-canvas legend; bonds follow the atoms or take one colour.
- ⚡ **Large-system performance** — atoms *and* bonds render as single instanced draw calls with instance matrices written straight into the GPU buffer (no per-atom `Object3D` compose, no per-instance colour string re-parsing); bond inference uses an O(n) spatial hash grid; adaptive device-pixel-ratio and tessellation scale with system size; FPS-adaptive quality degradation; `frameloop="demand"` means zero idle GPU burn.
- 🧵 **Two Web Workers, so the UI never freezes** — parsing runs off the main thread, and so does trajectory analysis. **RDF uses a periodic cell list** instead of an all-pairs loop: measured **7× faster at 10 000 atoms and 36× at 30 000** (36.9 s → 1.0 s), with the accelerated histogram asserted bin-for-bin identical to a brute-force reference. MSD resolves a stable atom ordering once instead of rebuilding a lookup inside its inner loop.
- 🎬 **High-quality video export** — one-click recording of the live canvas at up to 60 fps and ~24 Mbps, saved as MP4 (H.264) where the browser supports it, WebM (VP9/VP8) otherwise.
- 💡 **Five lighting rigs** (studio · lab · outdoor · space · soft) plus four materials (realistic · plastic · metallic · toon), ten representations (ball & stick · space fill · wireframe · licorice · element balls sized by covalent radius · points: shaded sphere impostors in one draw call for very large systems · lines: bonds as one-pixel lines, for large molecules · cubes, for lattices · glass: see-through space fill · flat outlined discs, for 2D systems); every one is a single draw call for the atoms and one for the bonds.
- ✨ **Rendering quality** Low · Auto · High · Ultra (tessellation, supersampling, shadow resolution and reach), tone mapping (ACES filmic · AgX · Neutral) with exposure, environment reflections from a procedural room (nothing downloaded) and depth cueing; view settings are remembered in the browser and carried by share links.
- 📦 **Simulation box display** for LAMMPS bounds, PDB `CRYST1` and CIF cells — triclinic cells render correctly, never faked as cubes.
- ⌨️ **Full keyboard control** with an in-app shortcut overlay (`H`).
- 🖱️ **Drag & drop** files anywhere on the page (gzipped files such as `dump.lammpstrj.gz` are decompressed); hover any atom for element details.
- 💾 **Structure export** — the Export button writes the shown frame as a LAMMPS data file (for `read_data`), extended XYZ or PDB, and a trajectory's frames as one extended XYZ.
- ↔️ **Resizable sidebar** — drag its edge to any width between 280–560 px; the choice persists.
- 🌓 Dark / light themes, mobile-friendly layout, PNG screenshots.

## 🚀 Quick start

Open the [live demo](https://shuvam-banerji-seal.github.io/lammps-web-gui/) and:

1. Drag a `.data` / `.xyz` / `.pdb` / `.cif` file onto the page, or
2. Try the bundled examples (C60 fullerene, benzene, rocksalt NaCl, water cluster).

Your files never leave your machine — parsing happens locally in the browser.

## 🤖 MCP server for coding agents

Any coding agent that speaks the [Model Context Protocol](https://modelcontextprotocol.io) can use what the site knows
about writing LAMMPS input. That includes Claude Code, Codex, Cursor, VS Code, Windsurf, Cline, Gemini CLI and
opencode. The server gives them:

- the LAMMPS documentation: search, and every page at the engine's LAMMPS commit;
- the script checker;
- which commands and styles the browser engine runs;
- the starter scripts and example notebooks;
- CMake build scripts.

```bash
claude mcp add lammps -- npx -y --allow-remote=all https://shuvam-banerji-seal.github.io/lammps-web-gui/mcp/lammps-mcp.tgz
```

Setup for every client is on the [MCP page](https://shuvam-banerji-seal.github.io/lammps-web-gui/mcp/). The server is
one JavaScript file that runs locally over stdio. The site is static, so it cannot host an MCP HTTP endpoint. Agents
without MCP can read [`llms.txt`](https://shuvam-banerji-seal.github.io/lammps-web-gui/llms.txt) and
[`mcp/knowledge.json`](https://shuvam-banerji-seal.github.io/lammps-web-gui/mcp/knowledge.json). Sources:
[`mcp/`](mcp/README.md).

## ⌨️ Keyboard shortcuts

### Script Builder

| Key | Action |
|---|---|
| `/` | Focus the command palette search |
| `B` | **Fork a concept** at the selected step (or at the end) |
| `[` `]` | Cycle which concept is taken at that fork |
| `F` | Fit the whole pipeline in view |
| `0` | Back to 100%, centred |
| `S` | Toggle flowchart ⇄ generated script |
| `C` | Toggle the Script check panel |
| `Delete` | Remove the selected step |
| `Esc` | Deselect / close any open menu |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo (includes template loads and forks) |

Typing targets are always skipped, so the manual-script editor and every
parameter field keep their native behaviour.

### Structure Viewer

| Key | Action |
|---|---|
| `Space` | Play / pause auto-rotation |
| `R` | Reset & fit view |
| `1` `2` `3` `4` `6` `0` | Ball & Stick · Space Fill · Wireframe · Licorice · Element balls · Points |
| `V` | Next representation (every one, Lines · Cubes · Glass · Flat discs included) |
| `7` `8` `9` `5` | Camera: front · top · right · isometric |
| `←` `→` `↑` `↓` | Orbit camera |
| `+` / `−` | Zoom in / out |
| `B` | Toggle bonds |
| `X` | Toggle simulation box |
| `L` | Toggle element labels (≤400 atoms) |
| Click atoms | Measure distance → angle → dihedral (pick 2–4) |
| `C` | Clear measurement selection |
| `,` / `.` | Previous / next trajectory frame |
| `P` | Play / pause trajectory playback |
| `G` | Cycle lighting preset |
| `T` | Toggle dark / light theme |
| `O` | Toggle sidebar |
| `S` | Save PNG screenshot |
| `H` or `?` | Shortcut overlay |
| `Esc` | Close panels |

## 🛠️ Technology

- [React 19](https://react.dev/) + TypeScript
- [three.js](https://threejs.org/) via [@react-three/fiber](https://docs.pmnd.rs/react-three-fiber) & [@react-three/drei](https://github.com/pmndrs/drei)
- [Vite 8](https://vite.dev/) + [Tailwind CSS 4](https://tailwindcss.com/)
- [Vitest](https://vitest.dev/) — 3000+ tests: parsers, chemistry, catalog, import, and native-LAMMPS oracle cases for the MD Notebook engine
- Zero runtime third-party requests: fonts, colors, lighting — all local.

## 💻 Development

```bash
git clone https://github.com/Shuvam-Banerji-Seal/lammps-web-gui.git
cd lammps-web-gui
npm install        # Node >= 20
npm run dev        # http://localhost:5173
npm test           # vitest suite
npm run typecheck  # tsc --noEmit
npm run build      # production build → dist/
```

Project layout:

```
├── src/                     # application source
│   ├── main.tsx                   # entry point
│   ├── App.tsx                    # workbench shell, module switcher, global theme
│   ├── theme.ts                   # shared warm coffee-green light/dark tokens
│   ├── styles.css                 # Tailwind v4 entry + base styles
│   ├── constants.ts               # CPK colors + covalent radii (118 elements)
│   ├── types.ts
│   ├── components/          # React Three Fiber scene components
│   │   ├── MoleculeCanvas.tsx     # Canvas, adaptive DPR, hover picking
│   │   ├── InstancedAtomMesh.tsx  # one draw call for all atoms
│   │   ├── InstancedBondMesh.tsx  # one draw call for all bonds (half-bond coloring)
│   │   ├── BondLines.tsx          # lines representation: bonds as one-pixel lines, one draw call
│   │   ├── PointAtoms.tsx         # points, lines and flat-disc atoms: sphere or disc impostors, one draw call
│   │   ├── SceneEffects.tsx       # tone mapping, environment reflections, depth cueing
│   │   ├── SimulationBox.tsx      # box/triclinic cell wireframe
│   │   ├── LightingRig.tsx        # five lighting presets
│   │   ├── CameraRig.tsx          # fit/preset/zoom/orbit command handling
│   │   ├── AtomLabels.tsx         # canvas-texture element badges
│   │   ├── MeasurementOverlay.tsx # selection rings + measurement visuals
│   │   └── workbench/             # Script Builder · Compiler Helper · Viewer shell
│   ├── lammps/              # workbench logic (pure, fully tested)
│   │   ├── templates.ts           # one-click starter pipelines
│   │   ├── scriptParser.ts        # import in.* scripts → flowchart steps
│   ├── flowchartSvg.ts            # presentable SVG/PNG flowchart export
│   │   ├── catalog.ts             # 171-command declarative library
│   │   ├── generator.ts           # model → in.lammps (+ manual-edit mode)
│   │   ├── compiler.ts            # packages/presets → CMake script + flag docs
│   │   └── exporter.ts            # data-file writer + download helper
│   ├── services/            # parsers & geometry logic (pure functions, fully tested)
│   │   ├── parser.ts              # LAMMPS (4 atom styles + box + tilts)
│   │   ├── xyzParser.ts           # XYZ incl. multi-frame trajectories
│   │   ├── pdbParser.ts           # PDB (+CONECT, CRYST1)
│   │   ├── cifParser.ts           # CIF (fractional coords, triclinic)
│   │   ├── dumpParser.ts          # LAMMPS dump trajectories (.lammpstrj)
│   │   ├── bondInference.ts       # O(n) spatial-hash covalent bonding
│   │   ├── colorMaps.ts           # colour maps and per-atom colouring (type, quantity, molecule)
│   │   ├── atomStyle.ts           # atom radii per representation, quality tiers
│   │   ├── measure.ts             # distance / angle / dihedral math
│   │   ├── persistence.ts         # safe JSON localStorage layer
│   │   └── viewState.ts           # shareable ?s= view encoding
│   ├── hooks/               # keyboard · persistence · trajectory analysis
│   └── workers/             # parser + trajectory-analysis Web Workers
├── tests/                   # vitest suites (parsers, chemistry, catalog, persistence)
├── public/                  # static assets + example structures
├── docs/screenshots/
├── scripts/                 # wiki publisher, size budget check
├── wiki/                    # versioned GitHub wiki source
└── .github/workflows/       # CI (typecheck·test·audit·build) + Pages deploy
```

## 📦 Deployment

Every push to `main` runs CI (typecheck → tests → `npm audit` → build → bundle
budget, on Node 22 and 24) and deploys the static build to GitHub Pages via
GitHub Actions.

**There is no backend, by design.** Every parser, the script generator, the
validator and all the trajectory analysis run in your browser — your files
never leave your machine. A legacy Flask stub used to sit in `server/`; it was
unused by the deployed site and carried its own CVEs, so it was removed (it is
still in the git history if you ever want it back).

`main` is protected: direct pushes are blocked and every change lands through a
pull request that must pass CI on both Node lines, dependency review, CodeQL
and code-owner review. See [CONTRIBUTING.md](CONTRIBUTING.md#branch-protection-and-the-pr-flow).

## 🔍 Search indexing

The site ships `robots.txt`, an image `sitemap.xml`, a canonical URL, Open Graph and
Twitter cards with a 1200×630 preview (`public/og-image.png`), crawlable `<noscript>`
content, and JSON-LD structured data (`WebApplication`, `SoftwareSourceCode`, `Person`,
`VideoObject`) in [`index.html`](index.html). To get it indexed or re-indexed:

- **Google** — add `https://shuvam-banerji-seal.github.io/lammps-web-gui/` as a URL-prefix
  property in [Google Search Console](https://search.google.com/search-console), submit
  `sitemap.xml`, and use *URL inspection* → *Request indexing* after a release.
- **Bing, DuckDuckGo, Yahoo, Ecosia** (all use Bing's index) — add the site in
  [Bing Webmaster Tools](https://www.bing.com/webmasters) (it can import the Google
  Search Console property) and submit the same sitemap.

## 🤝 Contributing

PRs welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). By opening a PR you license your
contribution to the author under [LICENSE §6](LICENSE) so it can ship as part
of the project — you keep your copyright and get credited. Security issues go through
[private advisories](SECURITY.md) — please don't open public issues for them.

The project wiki has deeper documentation:
[format guides](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/wiki),
[performance notes](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/wiki/Performance)
and more.

## 📄 License

**[LAMMPS Web GUI Educational and Non-Commercial Source-Available Licence
v1.0](LICENSE)** © 2025–2026 Shuvam Banerji Seal. All rights reserved.

This is **source-available, not open source**. In plain terms:

| | |
|---|---|
| ✅ **Free** | Education, teaching, coursework, theses, academic research, personal study — clone it, modify it, fork it, share it at no charge. |
| ✅ **Free** | Universities, schools, research institutes, national labs, public libraries and museums, for their educational and academic mission. |
| ⚠️ **Required** | **Credit Shuvam Banerji Seal.** Keep the notices in the source, keep the visible credit in any build you deploy, and **cite the project** in any paper, thesis or poster it helped produce — see [`CITATION.cff`](CITATION.cff). Attribution is a licence condition (§3), not a courtesy. |
| ❌ **Not allowed** | Selling it, hosting it as a paid or ad-supported service, using it inside a for-profit company to build what you sell, paid consulting or training, or bundling it into a commercial product. |
| 📩 **Ask first** | Commercial use is available **by written permission**, with a revenue share (10% of gross by default, negotiable — reduced or waived terms are genuinely available for spin-offs, non-profits and small teams). Read [**COMMERCIAL.md**](COMMERCIAL.md) and email <shuvam@sycolex.com>. |

Redistributing it? Ship [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) unmodified,
keep every credit intact, and keep this same licence on your derivative work.

Releases made **before 22 September 2026** were MIT-licensed; that grant is not
revoked for those copies (LICENSE §11).

Third-party dependencies (React, three.js, Tailwind, …) keep their own licences
— see [`NOTICE`](NOTICE). **LAMMPS** itself is a separate GPL-2.0 project of
Sandia National Laboratories and the LAMMPS Developers; this tool is
independent and is not affiliated with or endorsed by it. Styles ported from
the LAMMPS C++ source are kept in [`src/engine-gpl`](src/engine-gpl/README.md), and
unmodified LAMMPS potential files and examples in [`third_party/lammps`](third_party/lammps/README.md);
both are GPL-2.0 and carry their own `LICENSE`.
