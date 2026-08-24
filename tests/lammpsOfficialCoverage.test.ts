import { describe, it, expect } from 'vitest';
import { ALL_COMMANDS } from '../src/lammps/catalog';

/**
 * The authoritative alphabetical list of general LAMMPS commands from
 * docs.lammps.org/Commands_all.html (git 4Jul2026), including the
 * package-provided section. This test LOCKS coverage: every new command
 * LAMMPS adds will fail this test until the catalog grows — by design.
 */
const OFFICIAL_GENERAL_COMMANDS = [
  // 6.5 general commands
  'angle_coeff', 'angle_style', 'angle_write', 'atom_modify', 'atom_style',
  'balance', 'bond_coeff', 'bond_style', 'bond_write', 'boundary', 'box',
  'change_box', 'clear', 'comm_modify', 'comm_style', 'compute',
  'compute_modify', 'create_atoms', 'create_bonds', 'create_box',
  'delete_atoms', 'delete_bonds', 'dielectric', 'dihedral_coeff',
  'dihedral_style', 'dihedral_write', 'dimension', 'displace_atoms', 'dump',
  'dump_modify', 'echo', 'fix', 'fix_modify', 'geturl', 'group', 'if',
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
    // Allowed extras: free-form escape hatches and the import raw fallback.
    const allowed = new Set(['raw']);
    const extras = [...catalogKeywords].filter(c => !OFFICIAL_GENERAL_COMMANDS.includes(c));
    const unexplained = extras.filter(c => !allowed.has(c));
    // Every extra must be a def-id-level alias of an official command family
    // (e.g. 'region' variants share the keyword; these are keyword-level so
    // extras should be none beyond tooling).
    expect(unexplained, `unexplained keywords: ${unexplained.join(', ')}`).toEqual([]);
  });
});
