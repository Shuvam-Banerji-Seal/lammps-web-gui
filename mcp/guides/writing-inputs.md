# Writing a LAMMPS input script

A LAMMPS run is driven by an input script: a text file of commands that LAMMPS reads and executes one at a time, in
order. This guide covers the structure of a script and the rules LAMMPS uses to read it. Look up the exact syntax of
every command you use (`get_lammps_doc`) instead of writing it from memory. The pages are the LAMMPS documentation at
the commit the browser engine is checked against.

## Structure

From [Input script structure](https://docs.lammps.org/Commands_structure.html): "A LAMMPS input script typically has 4
parts: Initialization, System definition, Simulation settings, Run a simulation", and "The last 2 parts can be repeated
as many times as desired. I.e. run a simulation, change some settings, run some more, etc."

1. **Initialization**: "Set parameters that need to be defined before atoms are created or read-in from a file": `units`,
   `dimension`, `newton`, `processors`, `boundary`, `atom_style`, `atom_modify`. When the data file holds force-field
   coefficients, the styles come here too (`pair_style`, `bond_style`, `angle_style`, `dihedral_style`,
   `improper_style`).
2. **System definition**: the box and the atoms. Read them from a data file (`read_data`) or a restart file
   (`read_restart`), or build them on a lattice with `lattice`, `region`, `create_box` and `create_atoms`. `replicate`
   duplicates the whole system.
3. **Simulation settings**: force-field coefficients (`pair_coeff`, `bond_coeff`, ..., `kspace_style`,
   `special_bonds`), simulation parameters (`neighbor`, `neigh_modify`, `group`, `timestep`, `run_style`, `min_style`),
   fixes (time integration, thermostats, constraints), computes and variables, and output (`thermo`, `dump`,
   `restart`).
4. **Run**: `run N` for molecular dynamics, `minimize` for energy minimization.

"Remember that almost all commands need only be used if a non-default value is desired."

## How LAMMPS reads a line

From [Parsing rules for input scripts](https://docs.lammps.org/Commands_parse.html):

- "Each non-blank line in the input script is treated as a command. LAMMPS commands are case sensitive."
- "If the last printable character on the line is a "&" character, the command is assumed to continue on the next
  line."
- "All characters from the first "#" character onward are treated as comment and discarded."
- `${name}` and `$x` are replaced by the value of a variable. `$(expression)` is an immediate variable, evaluated as an
  equal-style variable.

## A complete example

This is the notebook's starter, a small version of LAMMPS `examples/melt`: 500 atoms of a Lennard-Jones fcc solid,
given a 3.0 temperature in reduced units and run at constant energy.

```lammps
# 1. initialization
units           lj
atom_style      atomic

# 2. system definition
lattice         fcc 0.8442
region          box block 0 5 0 5 0 5
create_box      1 box
create_atoms    1 box
mass            1 1.0

# 3. settings
velocity        all create 3.0 87287
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
fix             1 all nve
thermo          50

# 4. run
run             250
```

## Checklist

Each item cites the docs page it comes from.

- Choose `units` first. It sets the meaning of every number that follows (time step, cutoffs, temperatures,
  pressures). The default is `lj` ([units](https://docs.lammps.org/units.html)).
- `atom_style` decides what each atom carries (charge, bonds, ...), and it must match the data file
  ([atom_style](https://docs.lammps.org/atom_style.html), [read_data](https://docs.lammps.org/read_data.html)).
- "All masses must be defined before a simulation is run. They must also all be defined before a velocity or fix
  shake command is used." ([mass](https://docs.lammps.org/mass.html))
- Time integration is done by fixes. "Not all thermostat fixes perform time integration ... Thus, they should be used
  in conjunction with a constant NVE integration fix" (for example `fix langevin` with `fix nve`)
  ([Howto thermostat](https://docs.lammps.org/Howto_thermostat.html)). With no integrating fix, atoms do not move.
- The `*/coul/long` pair styles damp the Coulomb term "so it can be used in conjunction with the kspace_style command
  and its ewald or pppm option", so they need a `kspace_style`
  ([pair lj/cut/coul](https://docs.lammps.org/pair_lj_cut_coul.html)).

## Tools on this server

- `search_lammps_docs`: find the command or style for a task, e.g. "langevin thermostat" or "radial distribution".
- `get_lammps_doc`: the full documentation page (syntax, keywords, defaults, restrictions).
- `validate_lammps_script`: the LAMMPS Web GUI script checker. Every rule cites its docs page.
- `check_browser_support`: whether the in-browser engine of the LAMMPS Web GUI can run the script.
- `list_templates`, `get_template`, `list_examples`, `get_example`: complete scripts to start from.
- `generate_cmake_build`: a build script for a local LAMMPS with the packages a script needs.
