"""Builds the example notebooks (public/notebooks/*.lmpnb and index.json) of the LAMMPS Web GUI.

Usage: python3 scripts/examples/gen_notebooks.py SRC_DIR public/notebooks
SRC_DIR holds the input files the notebooks carry (see scripts/examples/README.md for how each one is made).
Every code cell below was run with native LAMMPS (b332f28) and with the browser engine before it was written here.
"""
import json
import os
import sys

SRC = sys.argv[1]
OUT = sys.argv[2]
STAMP = '2026-10-20T00:00:00.000Z'


def read(name):
    with open(os.path.join(SRC, name)) as fh:
        return fh.read()


def notebook(file, name, summary, cells, files):
    doc = {
        'format': 'lammps-web-notebook', 'version': 1, 'name': name, 'created': STAMP, 'modified': STAMP,
        'cells': [{'id': f'c{k + 1}', 'kind': kind, 'text': text.strip('\n')} for k, (kind, text) in enumerate(cells)],
        'inputFiles': {f: read(f) for f in files},
    }
    with open(os.path.join(OUT, file), 'w') as fh:
        json.dump(doc, fh, indent=1)
        fh.write('\n')
    return {'file': file, 'name': name, 'summary': summary, 'files': files}


md = lambda t: ('markdown', t)
code = lambda t: ('code', t)

index = []

index.append(notebook('water-spce-data.lmpnb', 'Water box from a data file (SPC/E)',
    'read_data of 216 SPC/E water molecules (atoms, bonds, angles, charges, velocities), SHAKE, PPPM, the O-O RDF and the MSD.',
    [md('''
# Water box from a LAMMPS data file

This notebook loads a complete molecular system from a **LAMMPS data file**, `water_spce.data` (listed under *Input
files*): 216 SPC/E water molecules that were equilibrated at 300 K and 0.9975 g/cm³. A data file carries everything
about the atoms that a script cannot easily create:

* the header: counts (`648 atoms`, `432 bonds`, `216 angles`), the number of types and the box bounds;
* `Masses`, then `Atoms # full`: one line per atom, `atom-ID molecule-ID atom-type charge x y z` and the image flags
  `ix iy iz`, which say in which periodic image each atom sits so that every molecule stays whole;
* `Velocities`, `Bonds` and `Angles`: the topology, by atom IDs.

The force-field coefficients are left out of the file (`write_data ... nocoeff`) and set in the script, so the same
file can be reused with another water model.

**Your own data file:** drop it on the notebook (or use *Add files*) and put its name in `read_data` below. The
`atom_style` must match the `Atoms # ...` line of the file. A PDB, CIF or XYZ structure becomes a data file with
**Import structure…**.
'''), code('''
units           real
atom_style      full
boundary        p p p
read_data       water_spce.data
'''), md('''
## The SPC/E force field

O-O Lennard-Jones (ε = 0.15535 kcal/mol, σ = 3.16557 Å), charges -0.8476 e and +0.4238 e (in the data file), long-range
electrostatics with PPPM, and rigid geometry: O-H 1.0 Å and H-O-H 109.47° are held by SHAKE, so the bond and angle
coefficients only give SHAKE its target values.
'''), code('''
pair_style      lj/cut/coul/long 10.0
pair_modify     mix arithmetic tail yes
kspace_style    pppm 1.0e-4
pair_coeff      1 1 0.15535 3.16557
pair_coeff      2 2 0.0 1.0
bond_style      harmonic
bond_coeff      1 1000.0 1.0
angle_style     harmonic
angle_coeff     1 1000.0 109.47
'''), md('''
## 1 ps of NVT dynamics with analysis

`compute rdf` gives the O-O and O-H radial distribution functions, averaged by `fix ave/time` into `water_rdf.txt`
(listed under *Files written*); the first O-O peak sits near 2.75 Å. `compute msd ... com yes` tracks the mean-squared
displacement (Å², the last thermo column). Increase `run` for smoother averages.
'''), code('''
fix             rigid all shake 0.0001 20 0 b 1 a 1
fix             nvt all nvt temp 300.0 300.0 100.0
timestep        2.0
compute         rdf all rdf 90 1 1 1 2 cutoff 9.0
fix             gr all ave/time 50 10 500 c_rdf[*] file water_rdf.txt mode vector
compute         msd all msd com yes
thermo_style    custom step temp pe press density c_msd[4]
thermo          50
run             500
''')], ['water_spce.data', 'water_spce.mol']))

index.append(notebook('co2-molecule-template.lmpnb', 'CO2 from a molecule template (TraPPE, rigid)',
    'molecule command + create_atoms random: 200 rigid TraPPE CO2 molecules placed without overlaps, fix rigid/small with a Langevin thermostat.',
    [md('''
# Filling a box with molecules from a molecule template

A **molecule template** (`co2_trappe.mol`, in *Input files*) describes one molecule: its atoms (`Coords`, `Types`,
`Charges`) and its topology (`Bonds`, `Angles`). The `molecule` command reads it, and `create_atoms ... mol` puts
copies into the box: here 200 molecules at random positions and orientations, each kept at least 3 Å from the others
(`overlap 3.0`). The molecule's bonds and angles need room in the box, so `create_box` reserves it
(`bond/types`, `extra/bond/per/atom`, ...).

**Your own molecule:** write a template by hand, or open a PDB, XYZ or CIF file with **Import structure…**: the dialog
converts it into a molecule template (*Output: Molecule template (molecule)*) and suggests the `molecule` and
`create_atoms` lines.
'''), code('''
units           real
atom_style      full
boundary        p p p
region          box block 0 40 0 40 0 40
create_box      2 box bond/types 1 angle/types 1 extra/bond/per/atom 2 extra/angle/per/atom 1 extra/special/per/atom 2
mass            1 12.011
mass            2 15.999
molecule        co2 co2_trappe.mol
create_atoms    0 random 200 37284 NULL mol co2 68213 overlap 3.0 maxtry 100
'''), md('''
## TraPPE CO2

Lennard-Jones sites on every atom (C: ε/k = 27 K, σ = 2.80 Å; O: ε/k = 79 K, σ = 3.05 Å, Lorentz-Berthelot mixing),
charges +0.70 e and -0.35 e, and a rigid linear molecule. `bond_style zero` and `angle_style zero` keep the topology
without forces, `special_bonds 0 0 0` removes the interactions inside a molecule, and `fix rigid/small molecule`
moves each molecule as a rigid body. At 300 K and 0.23 g/cm³ this is supercritical CO2.
'''), code('''
pair_style      lj/cut/coul/long 12.0
pair_modify     mix arithmetic tail yes
kspace_style    pppm 1.0e-4
pair_coeff      1 1 0.053654 2.80
pair_coeff      2 2 0.156989 3.05
bond_style      zero
bond_coeff      1 1.16
angle_style     zero
angle_coeff     1 180.0
special_bonds   lj/coul 0.0 0.0 0.0
'''), code('''
velocity        all create 300.0 87287 mom yes rot yes dist gaussian
fix             co2 all rigid/small molecule langevin 300.0 300.0 100.0 28473
timestep        2.0
thermo_style    custom step temp pe press density
thermo          100
run             1000
''')], ['co2_trappe.mol']))

index.append(notebook('butane-trappe.lmpnb', 'Liquid n-butane from a molecule template (TraPPE-UA)',
    'create_atoms box mol on a lattice with random orientations, a flexible united-atom model (bonds, angles, OPLS dihedral), minimize, NVT, and the dihedral angle distribution.',
    [md('''
# Liquid n-butane, flexible molecules from a template

`butane_trappe.mol` is a united-atom n-butane: four beads (CH3, CH2, CH2, CH3) with three bonds, two angles and one
dihedral. `create_atoms 0 box mol butane SEED` puts one molecule on every site of a simple cubic lattice, each with a
random orientation; 125 molecules at 0.61 g/cm³. Random orientations make some neighbours overlap, so a short
`minimize` comes before the dynamics.
'''), code('''
units           real
atom_style      molecular
boundary        p p p
lattice         sc 5.4
region          box block 0 5 0 5 0 5
create_box      2 box bond/types 1 angle/types 1 dihedral/types 1 extra/bond/per/atom 2 extra/angle/per/atom 2 extra/dihedral/per/atom 2 extra/special/per/atom 6
mass            1 15.035
mass            2 14.027
molecule        butane butane_trappe.mol
create_atoms    0 box mol butane 75421
'''), md('''
## TraPPE-UA force field

CH3: ε/k = 98 K, σ = 3.75 Å; CH2: ε/k = 46 K, σ = 3.95 Å; C-C 1.54 Å (a stiff harmonic bond instead of TraPPE's
fixed length), C-C-C 114° with k_θ/k = 62500 K/rad², and the TraPPE torsion in the OPLS form
(c1, c2, c3 = 355.03, -68.19, 791.32 K, so K_n = 2 c_n in kcal/mol). Beads of the same molecule do not interact
(`special_bonds lj 0 0 0`).
'''), code('''
pair_style      lj/cut 14.0
pair_modify     mix arithmetic tail yes
pair_coeff      1 1 0.194746 3.75
pair_coeff      2 2 0.091411 3.95
bond_style      harmonic
bond_coeff      1 452.9 1.54
angle_style     harmonic
angle_coeff     1 62.10 114.0
dihedral_style  opls
dihedral_coeff  1 1.41104 -0.27101 3.14505 0.0
special_bonds   lj 0.0 0.0 0.0
thermo_style    custom step temp pe press density
thermo          100
minimize        1.0e-4 1.0e-6 500 5000
'''), md('''
## Dynamics and the dihedral distribution

`compute dihedral/local` gives the C-C-C-C angle of every molecule, and `fix ave/histo` writes its histogram to
`butane_dihedrals.txt`: trans near ±180°, gauche near ±60°. The molecules start all-trans, so the gauche population
grows over the run; run longer to reach the equilibrium share.
'''), code('''
velocity        all create 270.0 59291 mom yes rot yes dist gaussian
fix             nvt all nvt temp 270.0 270.0 100.0
timestep        1.0
compute         phi all dihedral/local phi
fix             torsion all ave/histo 100 10 1000 -180.0 180.0 36 c_phi mode vector file butane_dihedrals.txt
run             3000
''')], ['butane_trappe.mol']))

index.append(notebook('quartz-cif.lmpnb', 'alpha-quartz from a CIF file (BKS)',
    'A CIF crystal structure (P3221, symmetry expanded) converted to a triclinic data file with the import dialog, replicated, charges set, and run with the BKS potential under NPT.',
    [md('''
# A crystal from a CIF file: alpha-quartz

`quartz.cif` is a crystallographic information file: the hexagonal cell, the space group P3₂21 with its six symmetry
operations, and the two atoms of the asymmetric unit (Si and O). LAMMPS does not read CIF files, so the notebook
converts it: **Import structure… → quartz.cif** opens the import dialog, which expands the symmetry operations (3 Si and
6 O per cell), makes the restricted triclinic LAMMPS box from the cell (`xy` tilt), and writes `quartz.data`
(*Output: LAMMPS data file*, *Atom style: charge*). That converted file is already in *Input files*; import the CIF
again to see the dialog.

The charges are not part of a CIF, so `set type ... charge` gives them; `replicate` builds a 3×3×3 supercell.
'''), code('''
units           metal
atom_style      charge
boundary        p p p
read_data       quartz.data
replicate       3 3 3
set             type 1 charge 2.4
set             type 2 charge -1.2
'''), md('''
## BKS potential and NPT dynamics

The van Beest-Kramer-van Santen potential: Buckingham terms for Si-O and O-O plus Coulomb with charges +2.4 e and
-1.2 e (PPPM). `npt ... tri` lets all six cell parameters relax at 300 K and 1 bar; the cell starts at the experimental
density, 2.649 g/cm³. BKS turns attractive at very short distances, which is harmless at room temperature.
'''), code('''
pair_style      buck/coul/long 10.0
pair_coeff      1 1 0.0 1.0 0.0
pair_coeff      1 2 18003.7572 0.205205 133.5381
pair_coeff      2 2 1388.7730 0.362319 175.0000
kspace_style    pppm 1.0e-4
velocity        all create 300.0 49104 mom yes rot yes dist gaussian
fix             npt all npt temp 300.0 300.0 0.1 tri 1.0 1.0 1.0
timestep        0.001
thermo_style    custom step temp pe press lx ly lz density
thermo          100
run             1000
''')], ['quartz.cif', 'quartz.data']))

index.append(notebook('ice-cif.lmpnb', 'Ice Ih from a CIF file (rigid SPC/E water)',
    'A molecular crystal from a CIF: bonds inferred through the periodic cell, molecules kept whole by image flags, rigid SPC/E water with fix rigid/nvt/small at 200 K.',
    [md('''
# A molecular crystal from a CIF file: ice Ih

`ice_ih.cif` holds 96 water molecules of proton-ordered ice Ih (every O donates two hydrogen bonds and accepts two,
zero net dipole). **Import structure…** with *Atom style: full* and *bonds* checked converts it into `ice_ih.data`:

* the O-H bonds are found from covalent radii **through the periodic cell**, so molecules cut by the cell faces keep
  their bonds;
* every atom gets image flags that keep its molecule whole, and the molecule IDs follow the bonds (one per water).

Whole molecules matter for rigid bodies: `fix rigid/small molecule` builds each body from the unwrapped coordinates.
'''), code('''
units           real
atom_style      full
boundary        p p p
read_data       ice_ih.data
set             type 1 charge -0.8476
set             type 2 charge 0.4238
'''), md('''
## Rigid SPC/E water

The SPC/E charges and O-O Lennard-Jones, with every molecule moved as a rigid body by `fix rigid/nvt/small` (the
bonds only define the molecules: `bond_style zero`, `special_bonds 0 0 0`). At 200 K, below the melting point of
SPC/E ice (about 215 K), the crystal keeps its order: the O-O RDF in `ice_rdf.txt` keeps sharp peaks at 2.75 Å and
4.5 Å.
'''), code('''
pair_style      lj/cut/coul/long 10.0
pair_modify     tail yes
kspace_style    pppm 1.0e-4
pair_coeff      1 1 0.15535 3.16557
pair_coeff      2 2 0.0 1.0
pair_coeff      1 2 0.0 1.0
bond_style      zero
bond_coeff      1 1.0
special_bonds   lj/coul 0.0 0.0 0.0
'''), code('''
velocity        all create 200.0 32108 mom yes rot yes dist gaussian
fix             ice all rigid/nvt/small molecule temp 200.0 200.0 100.0
timestep        2.0
compute         rdf all rdf 80 1 1 cutoff 7.0
fix             gr all ave/time 50 10 500 c_rdf[*] file ice_rdf.txt mode vector
thermo_style    custom step temp pe press
thermo          100
run             1000
''')], ['ice_ih.cif', 'ice_ih.data']))

index.append(notebook('urea-cif.lmpnb', 'Urea crystal from a CIF file (structure check)',
    'Symmetry expansion of a molecular CIF (P-421m) whose molecules cross the cell faces: the import keeps all 7 bonds of every molecule; replicate and count molecules and bond lengths.',
    [md('''
# Molecules that cross the cell: urea

`urea.cif` lists one urea molecule's asymmetric unit (C, O, N, two H) and the eight symmetry operations of P-421m. After
the symmetry expansion the two molecules of the cell lie across the cell faces, as in most molecular crystal files.
**Import structure…** (*Atom style: full*, *bonds* checked) infers the bonds through the periodic cell and writes image
flags, so
`urea.data` holds 2 whole molecules with 7 bonds each.

This notebook only checks the structure (no urea force field is included): after `replicate 3 3 3` there should be
54 molecules, 378 bonds, and every bond between 1.0 Å (N-H) and 1.35 Å (C-N). A molecule broken by the cell faces
would show up as a bond longer than the box.
'''), code('''
units           real
atom_style      full
boundary        p p p
read_data       urea.data
replicate       3 3 3
pair_style      zero 8.0
pair_coeff      * *
bond_style      zero
bond_coeff      *
compute         bonds all bond/local dist
compute         longest all reduce max c_bonds inputs local
compute         shortest all reduce min c_bonds inputs local
compute         molecules all chunk/atom molecule
thermo_style    custom step atoms bonds c_molecules c_shortest c_longest
run             0
''')], ['urea.cif', 'urea.data']))

with open(os.path.join(OUT, 'index.json'), 'w') as fh:
    json.dump({'notebooks': index}, fh, indent=1)
    fh.write('\n')
print('\n'.join(f"{e['file']}: {os.path.getsize(os.path.join(OUT, e['file']))} bytes" for e in index))
