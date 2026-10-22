import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix move as native LAMMPS reads it (src/fix_move.cpp at b332f28): the words of each style (linear, wiggle, rotate,
 * transrot, variable), the keywords, the 2d checks, the variables of the variable style at the run's init, the
 * timestep command after a run, and the per-atom array of original coordinates. Measured with native LAMMPS
 * (/home/roy/.local/bin/lmp, b332f28): each row is the 3d or 2d input below with its lines added, null when it runs,
 * else the first line of native's message. The motion itself is checked against native by the oracle cases
 * tests/oracle/move_variable_*.in and tests/oracle/meso_move_variable.in. The MESO rows are fix meso/move
 * (src/DPD-SMOOTH/fix_meso_move.cpp) on an atom_style sph system.
 */

const BASE3 = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
variable a equal 0.1*step
variable s string abc
variable va atom x*0.01
variable vv vector [1,2]
`;

const BASE2 = `units lj
dimension 2
atom_style atomic
lattice sq2 0.8442
region box block 0 2 0 2 -0.5 0.5
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
variable a equal 0.1*step
`;

/** an atom_style sph system for fix meso/move (src/DPD-SMOOTH/fix_meso_move.cpp, ported in src/engine-gpl) */
const BASE_SPH = `units micro
dimension 2
atom_style sph
region box block -0.5 0.5 -0.5 0.5 -0.05 0.05 units box
create_box 1 box
lattice sq 0.1
create_atoms 1 box
mass * 0.001
pair_style zero 0.45
pair_coeff * *
timestep 0.001
variable a equal 0.1*step
variable s string abc
variable va atom x*0.01
`;

const input = (dim: string, post: string): string => `${dim === '3' ? BASE3 : dim === '2' ? BASE2 : BASE_SPH}${post}\n`;

const errorOf = async (dim: string, post: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(input(dim, post)); } catch (e) { return (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return null;
};

const ROWS: [string, string, string | null][] = [
  ['3', 'fix 1 all move\nrun 1', 'Illegal fix move command: missing argument(s)'],
  ['3', 'fix 1 all move bogus 1 2 3\nrun 1', 'Unknown fix move mode bogus'],
  ['3', 'fix 1 all move linear 1 2\nrun 1', 'Illegal fix move linear command: missing argument(s)'],
  ['3', 'fix 1 all move linear a 0 0\nrun 1', 'Expected floating point parameter instead of \'a\' in input script or data file'],
  ['3', 'fix 1 all move linear NULL 0 0.5\nrun 1', null],
  ['3', 'fix 1 all move wiggle 1 0 0\nrun 1', 'Illegal fix move wiggle command: missing argument(s)'],
  ['3', 'fix 1 all move wiggle 1 0 0 0\nrun 1', 'Illegal fix move wiggle period 0'],
  ['3', 'fix 1 all move wiggle 1 0 0 -2\nrun 1', 'Illegal fix move wiggle period -2'],
  ['3', 'fix 1 all move rotate 0 0 0 0 0 0 1\nrun 1', 'Zero length rotation vector with fix move'],
  ['3', 'fix 1 all move rotate 0 0 0 0 0 1 0\nrun 1', 'Illegal fix move rotate period 0'],
  ['3', 'fix 1 all move rotate 0 0 0 0 0 1\nrun 1', 'Illegal fix move rotate command: missing argument(s)'],
  ['3', 'fix 1 all move transrot 1 0 0 0 0 0 0 0 1 1\nrun 1', null],
  ['3', 'fix 1 all move transrot 1 0 0 0 0 0 0 0 1 0\nrun 1', 'Illegal fix move rotate period 0'],
  ['3', 'fix 1 all move transrot NULL 0 0 0 0 0 0 0 1 1\nrun 1', 'Expected floating point parameter instead of \'NULL\' in input script or data file'],
  ['3', 'fix 1 all move variable v_a NULL NULL NULL NULL\nrun 1', 'Illegal fix move variable command: missing argument(s)'],
  ['3', 'fix 1 all move variable v_a NULL NULL NULL NULL NULL\nrun 1', null],
  ['3', 'fix 1 all move variable a NULL NULL NULL NULL NULL\nrun 1', 'Argument must be either NULL or variable reference'],
  ['3', 'fix 1 all move variable NULL NULL NULL NULL NULL v_q\nrun 1', 'Variable name q for fix move does not exist'],
  ['3', 'fix 1 all move variable v_s NULL NULL NULL NULL NULL\nrun 1', 'Variable name s for fix move is invalid style'],
  ['3', 'fix 1 all move variable v_vv NULL NULL NULL NULL NULL\nrun 1', 'Variable name vv for fix move is invalid style'],
  ['3', 'fix 1 all move variable v_va v_a NULL NULL NULL v_va\nrun 1', null],
  ['3', 'fix 1 all move variable NULL NULL NULL NULL NULL NULL\nrun 1', null],
  ['3', 'fix 1 all move linear 1 0 0 units\nrun 1', 'Illegal fix move units command: missing argument(s)'],
  ['3', 'fix 1 all move linear 1 0 0 units bogus\nrun 1', 'Unknown fix move units setting bogus'],
  ['3', 'fix 1 all move linear 1 0 0 units box\nrun 1', null],
  ['3', 'fix 1 all move linear 1 0 0 update\nrun 1', 'Illegal fix move update command: missing argument(s)'],
  ['3', 'fix 1 all move linear 1 0 0 update bogus\nrun 1', 'Unkown fix move update argument bogus'],
  ['3', 'fix 1 all move linear 1 0 0 update dipole\nrun 1', 'Fix move keyword update dipole requires style rotate or transrot'],
  ['3', 'fix 1 all move rotate 0 0 0 0 0 1 5 update dipole\nrun 1', 'Fix move keyword update dipole requires atom attribute mu'],
  ['3', 'fix 1 all move linear 1 0 0 bogus 1\nrun 1', 'Unknown fix move keyword bogus'],
  ['3', 'fix 1 all move variable v_a NULL NULL NULL NULL NULL units box\nrun 1', null],
  ['3', 'fix 1 all move linear 1 0 0\nrun 1\ntimestep 0.002\nrun 1', 'Resetting timestep size is not allowed with fix move'],
  ['3', 'fix 1 all move linear 1 0 0\ntimestep 0.002\nrun 1', null],
  ['3', 'fix 1 all move variable v_a NULL NULL NULL NULL NULL\nrun 1\ntimestep 0.002', 'Resetting timestep size is not allowed with fix move'],
  ['3', 'fix 1 all move linear 1 0 0\nvariable q equal f_1[1][4]\nrun 1\nprint "${q}"', 'Variable q: Fix \'1\' in variable formula does not compute the requested property. '],
  ['3', 'fix 1 all move linear 1 0 0\natom_modify map yes\nvariable q equal F_1[1][2]\nrun 1\nprint "VAL ${q}"', 'Atom_modify map command after simulation box is defined'],
  ['3', 'fix 1 all move linear 1 0 0\ncompute r all reduce max f_1[1]\nthermo_style custom step c_r\nrun 1', null],
  ['2', 'fix 1 all move linear 1 0 1\nrun 1', 'Fix move cannot set linear z motion for 2d problem'],
  ['2', 'fix 1 all move linear 1 0 0\nrun 1', null],
  ['2', 'fix 1 all move wiggle 1 0 1 2\nrun 1', 'Fix move cannot set wiggle z motion for 2d problem'],
  ['2', 'fix 1 all move rotate 0 0 0 1 0 0 2\nrun 1', 'Fix move cannot rotate around non z-axis for 2d problem'],
  ['2', 'fix 1 all move rotate 0 0 0 0 0 1 2\nrun 1', null],
  ['2', 'fix 1 all move transrot 0 0 1 0 0 0 0 0 1 2\nrun 1', 'Fix move cannot set linear z motion for 2d problem'],
  ['2', 'fix 1 all move variable NULL NULL v_a NULL NULL NULL\nrun 1', 'Fix move cannot define z or vz variable for 2d problem'],
  ['2', 'fix 1 all move variable NULL NULL NULL NULL NULL v_a\nrun 1', 'Fix move cannot define z or vz variable for 2d problem'],
  ['2', 'fix 1 all move variable v_a v_a NULL NULL NULL NULL\nrun 1', null],
  ['2', 'fix 1 all move linear 1 0 0\nfix 1 all move wiggle 1 0 0 1\nrun 1', null],
];

describe('fix move as native LAMMPS', () => {
  it.each(ROWS.map(([d, p, e]) => [`${d}d: ${p.replace(/\n/g, ' / ')}`, d, p, e] as const))('%s', async (_name, d, p, expected) => {
    expect(await errorOf(d, p)).toBe(expected);
  });
});

const MESO: [string, string | null][] = [
  ['fix 1 all meso/move\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move bogus 1 2 3\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move linear 1 2\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move linear a 0 0\nrun 1', 'Expected floating point parameter instead of \'a\' in input script or data file'],
  ['fix 1 all meso/move wiggle 1 0 0 0\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move rotate 0 0 0 0 0 0 1\nrun 1', 'Zero length rotation vector with fix meso/move'],
  ['fix 1 all meso/move rotate 0 0 0 1 0 0 1\nrun 1', 'Fix meso/move cannot rotate aroung non z-axis for 2d problem'],
  ['fix 1 all meso/move linear 1 0 0 units bogus\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move linear 1 0 0 bogus 1\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move linear 1 0 0 units\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move variable v_a NULL NULL NULL NULL\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move variable a NULL NULL NULL NULL NULL\nrun 1', 'Illegal fix meso/move command'],
  ['fix 1 all meso/move variable v_q NULL NULL NULL NULL NULL\nrun 1', 'Variable name for fix meso/move does not exist'],
  ['fix 1 all meso/move variable v_s NULL NULL NULL NULL NULL\nrun 1', 'Variable for fix meso/move is invalid style'],
  ['fix 1 all meso/move variable NULL NULL v_a NULL NULL NULL\nrun 1', 'Fix meso/move cannot define z or vz variable for 2d problem'],
  ['fix 1 all meso/move variable v_va v_a NULL v_a NULL NULL\nrun 2', null],
  ['fix 1 all meso/move linear 1 0 0\nrun 1\ntimestep 0.002', 'Resetting timestep size is not allowed with fix meso/move'],
  ['fix 1 all meso/move linear 1 0 0\ntimestep 0.002\nrun 1', null],
  ['fix 1 all meso/move variable v_a NULL NULL NULL NULL NULL units box\nrun 1', null],
];

describe('fix meso/move as native LAMMPS', () => {
  it.each(MESO.map(([p, e]) => [p.replace(/\n/g, ' / '), p, e] as const))('%s', async (_name, p, expected) => {
    expect(await errorOf('sph', p)).toBe(expected);
  });
});
