# Example notebooks: how they are made

`public/notebooks/*.lmpnb` are the notebooks of **MD Notebook → notebook menu → Open an example…**; `index.json` lists
them. Each notebook carries its input files. They are built by `gen_notebooks.py` from one directory that holds:

| File | Made by |
|---|---|
| `water_spce.mol`, `co2_trappe.mol`, `butane_trappe.mol` | written by hand (this directory) |
| `water_spce.data` | `make_water.in` with native LAMMPS (216 SPC/E molecules on a lattice, 20 ps at 300 K, `write_data ... nocoeff`), title line edited |
| `quartz.cif`, `urea.cif` | written by hand from the published structures (`public/examples`) |
| `ice_ih.cif` | `gen_ice.py` (a proton-ordered ice Ih cell: ice rules from an Eulerian-circuit orientation of the O-O network, zero net dipole) (`public/examples`) |
| `quartz.data`, `ice_ih.data`, `urea.data` | the notebook's **Import structure…** dialog (`convertStructure` of `src/lammps/structureToLammps.ts`): quartz with atom style charge, ice and urea with atom style full and bonds |

`tests/notebookExamples.test.ts` checks that every notebook runs in the engine, that it carries every file it reads, and
that the three converted data files are what the import dialog writes from the CIFs today.

    python3 scripts/examples/gen_notebooks.py <dir with the files above> public/notebooks
