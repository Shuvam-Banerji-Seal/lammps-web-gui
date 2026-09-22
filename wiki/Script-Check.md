# Script check

The Script Builder validates the script it produces against the LAMMPS
command rules, and links every finding to the documentation page that
defines it.

Open it with the shield/warning badge in the toolbar. The badge itself shows
the current error count (or warning count when there are no errors), so you can
see at a glance whether the pipeline is sound.

## What it checks

It reads the **final text** — generated *or* hand-edited in manual mode — so
what is validated is exactly what you will feed to `lmp -in`.

### Errors — LAMMPS will refuse, or silently do the wrong thing

| Rule | Requirement |
|---|---|
| `order/units-after-box` | `units` *"cannot be used after the simulation box is defined by a read_data or create_box command"* — [units](https://docs.lammps.org/units.html) |
| `order/dimension-after-box` | must come **before** the box — [dimension](https://docs.lammps.org/dimension.html) |
| `order/boundary-after-box` | cannot follow `read_data`/`create_box`/`read_restart`; use `change_box` — [boundary](https://docs.lammps.org/boundary.html) |
| `order/atom_style-after-box` | cannot follow the box definition — [atom_style](https://docs.lammps.org/atom_style.html) |
| `order/pair_coeff-before-box` | *"must come after the simulation box is defined"* — [pair_coeff](https://docs.lammps.org/pair_coeff.html) |
| `order/mass-before-box` | same requirement — [mass](https://docs.lammps.org/mass.html) |
| `order/pair_coeff-without-pair_style` | coefficients with no style declared |
| `order/create_atoms-without-atom_style` | *"An atom_style must be previously defined to use this command"* — [create_atoms](https://docs.lammps.org/create_atoms.html) |
| `order/velocity-without-atoms` | no `read_data`/`read_restart`/`create_atoms` yet |
| `ref/unknown-fix` · `ref/unknown-dump` · `ref/unknown-compute` | `unfix`/`undump`/`uncompute` of an ID that was never defined, or already removed |
| `ref/unknown-group` · `ref/unknown-region` | a group or region used before it exists |
| `id/fix-reuse` | re-issuing a fix ID with a **different** style (the same style is legal and is not flagged) |
| `id/dump-reuse` · `id/compute-reuse` | duplicate IDs |
| `run/no-atoms` | nothing to integrate (a `fix pour`/`deposit`/`gcmc` counts as an atom source) |
| `reaxff/missing-qeq` | *"LAMMPS requires that fix qeq/reaxff … is used with pair_style reaxff"* unless `checkqeq no` — [pair_reaxff](https://docs.lammps.org/pair_reaxff.html) |

### Warnings — it runs, but probably not as intended

| Rule | Why |
|---|---|
| `run/no-integrator` | no `fix nve`/`nvt`/`npt`/`rigid` active — LAMMPS runs and the atoms do not move |
| `run/no-pair-style` | atoms will not interact |
| `order/mass-after-velocity` | *"All masses must be defined before a velocity or fix shake command is used"* — [mass](https://docs.lammps.org/mass.html) |
| `order/create_atoms-without-lattice` | `create_atoms … box` fills from the current lattice; with no `lattice` command the default `lattice none 1.0` applies |
| `create_box-without-atom_style` | you get the default `atom_style atomic` — no charges, bonds or molecule IDs |
| `kspace/pair-mismatch` | a long-range solver needs a matching `coul/long`-family pair style |
| `script/no-run` | nothing is simulated |
| `script/pair-style-without-coeff` | a style with no coefficients and no data file to supply them |
| `script/no-mass` | no `mass` and no data file (not flagged for `atom_style sphere` and friends, which carry per-atom mass) |

## What it deliberately does not do

It is a **static ordering and reference checker**, not a physics referee. It
will not tell you that your timestep is too large, your thermostat damping is
unphysical or your force field is wrong for your system. It also does not
validate the style-specific argument grammar of all ~900 fix/compute/pair
styles.

Treat a clean check as "this script is structurally valid LAMMPS", not as
"this simulation is correct". The generated header says as much, and so does
the licence's warranty section.

## No false alarms

A linter that fires on the official examples would be worse than none, so the
whole rule set is asserted **silent** — zero errors *and* zero warnings — on
canonical scripts from the `lammps/lammps` examples tree: `in.melt`,
`in.heatflux`, `in.msd.2d` and `in.cos.1000SPCE`. Every shipped starter
template is held to the same bar.

If you hit a false positive, that is a bug worth reporting — use the
**"Incorrect LAMMPS output"** issue template and quote the docs sentence.

## See also

- [[Concept Branching]] — the check follows whichever concept is taken
- [`src/lammps/validate.ts`](https://github.com/Shuvam-Banerji-Seal/lammps-web-gui/blob/main/src/lammps/validate.ts) — the rules, each with its citation
