import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Reading the global outputs of averaging fixes in variables and thermo, as native LAMMPS does (src/variable.cpp
 * Variable::evaluate and special_function, src/thermo.cpp at b332f28): a fix is read on the steps its output is valid
 * (global_freq: Nfreq of ave/chunk, ave/histo, ave/correlate and ave/time, Nevery of vector), and a fix whose number of
 * rows varies (ave/chunk) reads 0 past its rows instead of an out-of-range error. Measured with native LAMMPS
 * (/home/roy/.local/bin/lmp, b332f28): each row is the input below with its lines added, null when it runs, else the
 * first line of native's message.
 */

const BASE = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass * 1.0
velocity all create 1.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
fix 1 all nve
compute ch all chunk/atom bin/1d z lower 0.25 units reduced
fix ac all ave/chunk 2 5 10 ch vx temp
compute myT all temp
fix vec all vector 5 c_myT
fix ah all ave/histo 1 5 10 0 2 4 vx mode vector
fix ac2 all ave/correlate 1 5 10 c_myT
thermo_style custom step temp
thermo_modify format float %.15g
`;

const errorOf = async (post: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(`${BASE}${post}\n`); } catch (e) { return (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return null;
};

const ROWS: [string, string | null][] = [
  ['variable a equal f_ac[1][4]\nthermo_style custom step v_a\nrun 0', null],
  ['variable a equal f_ac[99][4]\nthermo_style custom step v_a\nrun 0', null],
  ['variable a equal f_ac[1][5]\nthermo_style custom step v_a\nrun 0', 'Variable a: Variable formula fix array is accessed out-of-range'],
  ['variable a equal f_ac[1][4]\nthermo_style custom step v_a\nthermo 10\nrun 20', null],
  ['variable a equal f_ac[1][4]\nthermo_style custom step v_a\nthermo 5\nrun 20', 'Variable a: Fix in variable not computed at a compatible time'],
  ['variable a equal f_ac[99][4]\nthermo_style custom step v_a\nthermo 10\nrun 20', null],
  ['variable a equal slope(f_vec)\nthermo_style custom step v_a\nthermo 5\nrun 20', null],
  ['variable a equal slope(f_vec)\nthermo_style custom step v_a\nthermo 2\nrun 20', 'Fix with ID \'vec\' in variable formula not computed at compatible time'],
  ['variable a equal f_vec[1]\nthermo_style custom step v_a\nthermo 2\nrun 20', 'Variable a: Fix in variable not computed at a compatible time'],
  ['variable a equal f_ah[2]\nthermo_style custom step v_a\nthermo 5\nrun 20', 'Variable a: Fix in variable not computed at a compatible time'],
  ['variable a equal f_ac2[1][3]\nthermo_style custom step v_a\nthermo 5\nrun 20', 'Variable a: Fix in variable not computed at a compatible time'],
  ['thermo_style custom step f_ac2[1][3]\nthermo 5\nrun 20', 'Thermo and fix ac2 not computed at compatible times'],
  ['thermo_style custom step f_ah[2]\nthermo 10\nrun 20', null],
];

describe('fix outputs read by variables and thermo as native LAMMPS', () => {
  it.each(ROWS.map(([p, e]) => [p.replace(/\n/g, ' / '), p, e] as const))('%s', async (_name, p, expected) => {
    expect(await errorOf(p)).toBe(expected);
  });
});
