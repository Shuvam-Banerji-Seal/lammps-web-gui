<div align="center">

# Molecule3D

**A free, open-source, GPU-accelerated 3D molecular structure viewer that runs entirely in your browser.**

Load **LAMMPS**, **XYZ**, **PDB** and **CIF** structures by drag & drop and explore them in real-time 3D — no install, no upload, no account.

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-shuvam--banerji--seal.github.io-blue?style=for-the-badge)](https://shuvam-banerji-seal.github.io/lammps_data_web_viewer/)
[![CI](https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer/actions/workflows/ci.yml)
[![Deploy](https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer/actions/workflows/deploy.yml/badge.svg)](https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer/actions/workflows/deploy.yml)
[![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen)](#development)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## ✨ Features

- 🧊 **True 3D rendering** — depth-correct perspective view of atoms, bonds and the simulation cell, powered by three.js / React Three Fiber.
- 📁 **Four structure formats**
  | Format | Extensions | Highlights |
  |---|---|---|
  | LAMMPS data | `.data` `.lmp` `.lammps` | `atomic` · `charge` · `molecular` · `full` atom styles, box bounds incl. triclinic tilt (`xy xz yz`) |
  | XYZ | `.xyz` | first frame of trajectories, chemistry-aware bond inference |
  | Protein Data Bank | `.pdb` `.ent` | `CONECT` bonds, `CRYST1` unit cell |
  | CIF | `.cif` `.mmcif` | fractional ↔ Cartesian conversion, triclinic cells as true parallelepipeds |
- 🎨 **All 118 elements** resolved by symbol from *any* format and colored with standard **CPK/Jmol** palettes; per-type recoloring in the sidebar.
- ⚡ **Large-system performance** — atoms *and* bonds render as single instanced draw calls; bond inference uses an O(n) spatial hash grid; adaptive device-pixel-ratio and tessellation scale with system size.
- 💡 **Five lighting rigs** (studio · lab · outdoor · space · soft) plus four materials (realistic · plastic · metallic · toon).
- 📦 **Simulation box display** for LAMMPS bounds, PDB `CRYST1` and CIF cells — triclinic cells render correctly, never faked as cubes.
- ⌨️ **Full keyboard control** with an in-app shortcut overlay (`H`).
- 🖱️ **Drag & drop** files anywhere on the page; hover any atom for element details.
- 🌓 Dark / light themes, mobile-friendly layout, PNG screenshots.

## 🚀 Quick start

Open the [live demo](https://shuvam-banerji-seal.github.io/lammps_data_web_viewer/) and:

1. Drag a `.data` / `.xyz` / `.pdb` / `.cif` file onto the page, or
2. Try the bundled examples (C60 fullerene, benzene, rocksalt NaCl, water cluster).

Your files never leave your machine — parsing happens locally in the browser.

## ⌨️ Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / pause auto-rotation |
| `R` | Reset & fit view |
| `1` `2` `3` `4` | Ball & Stick · Space Fill · Wireframe · Licorice |
| `7` `8` `9` `5` | Camera: front · top · right · isometric |
| `←` `→` `↑` `↓` | Orbit camera |
| `+` / `−` | Zoom in / out |
| `B` | Toggle bonds |
| `X` | Toggle simulation box |
| `L` | Toggle element labels (≤400 atoms) |
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
- [Vitest](https://vitest.dev/) — 67 parser/chemistry unit tests
- Zero runtime third-party requests: fonts, colors, lighting — all local.

## 💻 Development

```bash
git clone https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer.git
cd lammps_data_web_viewer
npm install        # Node >= 20
npm run dev        # http://localhost:5173
npm test           # vitest suite
npm run typecheck  # tsc --noEmit
npm run build      # production build → dist/
```

Project layout:

```
├── components/          # React Three Fiber scene + UI components
│   ├── MoleculeCanvas.tsx     # Canvas, adaptive DPR, hover picking
│   ├── InstancedAtomMesh.tsx  # one draw call for all atoms
│   ├── InstancedBondMesh.tsx  # one draw call for all bonds (half-bond coloring)
│   ├── SimulationBox.tsx      # box/triclinic cell wireframe
│   ├── LightingRig.tsx        # five lighting presets
│   ├── CameraRig.tsx          # fit/preset/zoom/orbit command handling
│   └── AtomLabels.tsx         # canvas-texture element badges
├── services/            # parsers & geometry logic (pure functions, fully tested)
│   ├── parser.ts              # LAMMPS (4 atom styles + box + tilts)
│   ├── xyzParser.ts           # XYZ
│   ├── pdbParser.ts           # PDB (+CONECT, CRYST1)
│   ├── cifParser.ts           # CIF (fractional coords, triclinic)
│   └── bondInference.ts       # O(n) spatial-hash covalent bonding
├── hooks/useKeyboardShortcuts.ts
├── constants.ts         # CPK colors + covalent radii for all 118 elements
└── .github/workflows/   # CI (typecheck·test·audit·build) + Pages deploy
```

## 📦 Deployment

Every push to `main` runs CI (typecheck → tests → `npm audit` → build) and deploys
the static build to GitHub Pages via GitHub Actions. No backend is required —
the optional Flask server (`server.py`) is legacy and not used by the deployed site.

## 🔍 Search indexing

The site ships `robots.txt`, `sitemap.xml`, Open Graph/Twitter meta and JSON-LD
(`WebApplication`) structured data for search engines. After major releases,
the URL can be (re-)submitted through
[Google Search Console](https://search.google.com/search-console) → *URL inspection* → *Request indexing*.

## 🤝 Contributing

PRs welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). Security issues go through
[private advisories](SECURITY.md) — please don't open public issues for them.

The project wiki has deeper documentation:
[format guides](https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer/wiki),
[performance notes](https://github.com/Shuvam-Banerji-Seal/lammps_data_web_viewer/wiki/Performance)
and more.

## 📄 License

[MIT](LICENSE) © Shuvam Banerji Seal
