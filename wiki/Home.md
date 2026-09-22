# Home

**Molecule3D** is a free, GPU-accelerated LAMMPS workbench and 3D molecular
structure viewer that runs entirely in your browser.

It is **source-available**, not open source: free for education and
non-commercial research with attribution, commercial use by written licence.
See [LICENSE](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/blob/main/LICENSE)
and [COMMERCIAL.md](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/blob/main/COMMERCIAL.md).

▶ **Live app**: https://shuvam-banerji-seal.github.io/lammps-web-gui/
💻 **Source**: https://github.com/Shuvam-Banerji-Seal/lammps-web-gui

## Supported formats

- LAMMPS data files (`.data` `.lmp`) — atomic / charge / molecular / full styles
- LAMMPS dump trajectories (`.lammpstrj` `.dump`) — multi-frame playback,
  orthogonal & triclinic cells
- XYZ trajectories (`.xyz`)
- Protein Data Bank (`.pdb` `.ent`) — CONECT + CRYST1
- CIF crystals (`.cif`) — fractional coordinates, triclinic cells

## Wiki pages

| Page | Contents |
|---|---|
| [[Concept Branching]] | one flowchart, several divergent ideas |
| [[Script Check]] | the validator, every rule with its docs citation |
| [[File Formats]] | format guides with example snippets |
| [[Keyboard Shortcuts]] | every shortcut, printable |
| [[Performance]] | instancing, spatial hashing, quality ladder |
| [[Deployment]] | CI/CD pipeline and Pages setup |

## Highlights

- **LAMMPS Workbench** — Script Builder (editable flowchart, 140-command library,
  starter templates, undo/redo, manual script mode), Compiler Helper (presets →
  CMake scripts with a click-to-inspect flag list) and the 3D viewer, with
  state that persists across module switches and reloads
- Warm light/dark themes across the whole app (coffee-and-sage dark)
- All 118 elements with CPK/Jmol colors, resolved by symbol from any format
- Simulation box rendering including triclinic tilt
- One-draw-call instanced atoms *and* bonds; O(n) bond inference
- Five lighting presets, four materials, full keyboard control
- Drag & drop anywhere; files never leave your machine

## Contributing

PRs welcome — see [CONTRIBUTING](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/blob/main/CONTRIBUTING.md).
