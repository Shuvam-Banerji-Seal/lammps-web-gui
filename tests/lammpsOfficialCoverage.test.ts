import { describe, it, expect } from 'vitest';
import { ALL_COMMANDS } from '../src/lammps/catalog';

/**
 * The authoritative alphabetical list of general LAMMPS commands, scraped from
 * docs.lammps.org/Commands_all.html and diffed against the catalog on
 * 2026-09-22. Includes the package-provided section. This test LOCKS coverage:
 * every new command LAMMPS adds fails this test until the catalog grows — by
 * design.
 *
 * The 2026-09-22 re-check against the LIVE index found two drifts:
 *  - `fenix` was added upstream (FENIX package, 2Sep2026) and was missing;
 *  - `box` is NOT in the index any more — it was removed in 22Dec2022. It
 *    stays in the catalog, marked `deprecated`, so old scripts still import,
 *    but it is hidden from the palette and the validator warns about it.
 */
const OFFICIAL_GENERAL_COMMANDS = [
  // 6.5 general commands
  'angle_coeff', 'angle_style', 'angle_write', 'atom_modify', 'atom_style',
  'balance', 'bond_coeff', 'bond_style', 'bond_write', 'boundary',
  'change_box', 'clear', 'comm_modify', 'comm_style', 'compute',
  'compute_modify', 'create_atoms', 'create_bonds', 'create_box',
  'delete_atoms', 'delete_bonds', 'dielectric', 'dihedral_coeff',
  'dihedral_style', 'dihedral_write', 'dimension', 'displace_atoms', 'dump',
  'dump_modify', 'echo', 'fenix', 'fix', 'fix_modify', 'geturl', 'group', 'if',
  'improper_coeff', 'improper_style', 'include', 'info', 'jump',
  'kspace_modify', 'kspace_style', 'label', 'labelmap', 'lattice', 'log',
  'mass', 'minimize', 'min_modify', 'min_style', 'molecule', 'neigh_modify',
  'neighbor', 'newton', 'next', 'package', 'pair_coeff', 'pair_modify',
  'pair_style', 'pair_write', 'partition', 'print', 'processors', 'quit',
  'read_data', 'read_dump', 'read_restart', 'region', 'replicate', 'rerun',
  'reset_atoms', 'reset_timestep', 'restart', 'run', 'run_style', 'set',
  'shell', 'special_bonds', 'suffix', 'thermo', 'thermo_modify',
  'thermo_style', 'timer', 'timestep', 'uncompute', 'undump', 'unfix',
  'units', 'variable', 'velocity', 'write_coeff', 'write_data',
  'write_dump', 'write_molecule', 'write_restart',
  // package-provided general commands
  'dynamical_matrix', 'group2ndx', 'hyper', 'kim', 'fitpod', 'mdi',
  'ndx2group', 'neb', 'neb/spin', 'plugin', 'prd', 'python', 'region2vmd',
  'tad', 'temper', 'temper/grem', 'temper/npt', 'third_order',
];

describe('official LAMMPS command coverage (docs.lammps.org Commands_all)', () => {
  const catalogKeywords = new Set(ALL_COMMANDS.map(d => d.command));

  it('covers EVERY general command in the official index', () => {
    const missing = OFFICIAL_GENERAL_COMMANDS.filter(c => !catalogKeywords.has(c));
    expect(missing, `missing general commands: ${missing.join(', ')}`).toEqual([]);
  });

  it('official index size sanity (guards against a truncated copy-paste)', () => {
    expect(OFFICIAL_GENERAL_COMMANDS.length).toBeGreaterThanOrEqual(110);
  });

  it('catalog has no command keyword outside the official list (except tooling)', () => {
    // Allowed extras: the import raw fallback, plus commands LAMMPS has
    // removed that we keep purely so old scripts still round-trip.
    const allowed = new Set(['raw', 'box']);
    const extras = [...catalogKeywords].filter(c => !OFFICIAL_GENERAL_COMMANDS.includes(c));
    const unexplained = extras.filter(c => !allowed.has(c));
    expect(unexplained, `unexplained keywords: ${unexplained.join(', ')}`).toEqual([]);
  });

  it('every command LAMMPS removed is marked deprecated, not offered as new', () => {
    // docs.lammps.org/Commands_removed.html — removed in 22Dec2022.
    const removed = ['box', 'reset_ids', 'reset_atom_ids', 'reset_mol_ids'];
    for (const cmd of ALL_COMMANDS) {
      if (removed.includes(cmd.command)) {
        expect(cmd.deprecated, `${cmd.id} must carry a deprecation note`).toBeTruthy();
        expect(cmd.deprecated).toMatch(/22Dec2022/);
      }
    }
    // and nothing removed may be reachable without a deprecation note
    const addable = ALL_COMMANDS.filter(d => !d.deprecated).map(d => d.command);
    for (const r of removed) expect(addable).not.toContain(r);
  });

  it('newly added commands are present (fenix, FENIX package 2Sep2026)', () => {
    expect(catalogKeywords.has('fenix')).toBe(true);
  });
});
