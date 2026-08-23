# Home

**Molecule3D** is a free, open-source, GPU-accelerated 3D molecular structure
viewer that runs entirely in your browser.

▶ **Live app**: https://shuvam-banerji-seal.github.io/lammps_data_web_viewer/
💻 **Source**: https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer

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

PRs welcome — see [CONTRIBUTING](https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer/blob/main/CONTRIBUTING.md).
