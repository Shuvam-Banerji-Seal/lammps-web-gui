import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * compute reduce and reduce/region as native LAMMPS reads them (src/compute_reduce.cpp ComputeReduce::ComputeReduce and
 * init, src/arg_info.cpp, utils::expand_args at b332f28): the mode, the inputs up to the first word that is none, the
 * replace and inputs keywords, each input against its compute, fix or variable (per-atom or local), and the inputs
 * again at the run's init. Two native messages print a literal "{}" (their format has no argument). Measured with
 * native LAMMPS (/home/roy/.local/bin/lmp, b332f28): each row is the input below with its lines added, null when it
 * runs, else the first line of native's message.
 */

const BASE = `units lj
atom_style bond
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
compute ke all ke/atom
compute st all stress/atom NULL
compute myT all temp
compute pl all pair/local dist eng
compute pl1 all pair/local dist
variable va atom x
variable ve equal 1
fix sp all store/state 0 x y
`;

const errorOf = async (post: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(`${BASE}${post}\n`); } catch (e) { return (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return null;
};

const ROWS: [string, string | null][] = [
  ['compute r all reduce sum\nrun 0', 'Illegal compute reduce command: missing argument(s)'],
  ['compute r all reduce bogus x\nrun 0', 'Unknown compute reduce mode: bogus'],
  ['compute r all reduce/region nope sum x\nrun 0', 'Region nope for compute reduce/region does not exist'],
  ['compute r all reduce/region box sum\nrun 0', 'Illegal compute reduce/region command: missing argument(s)'],
  ['compute r all reduce/region box sum x\nrun 0', null],
  ['compute r all reduce sum x y z vx vy vz fx fy fz\nrun 0', null],
  ['compute r all reduce sum c_nope\nrun 0', 'Compute ID nope for compute reduce does not exist'],
  ['compute r all reduce sum c_myT\nrun 0', 'Compute reduce compute myT does not calculate per-atom values'],
  ['compute r all reduce sum c_ke[1]\nrun 0', 'Compute reduce compute ke does not calculate a per-atom array'],
  ['compute r all reduce sum c_st\nrun 0', 'Compute reduce compute st does not calculate a per-atom vector'],
  ['compute r all reduce sum c_st[7]\nrun 0', 'Compute reduce compute st array is accessed out-of-range'],
  ['compute r all reduce sum c_st[1] c_st[*]\nrun 0', null],
  ['compute r all reduce sum c_st[1][2]\nrun 0', 'Illegal compute reduce argument: c_st[1][2]'],
  ['compute r all reduce sum f_nope\nrun 0', 'Fix ID nope for compute reduce does not exist'],
  ['compute r all reduce sum f_sp\nrun 0', 'Compute reduce fix sp does not calculate a per-atom vector'],
  ['compute r all reduce sum f_sp[3]\nrun 0', 'Compute reduce fix sp array is accessed out-of-range'],
  ['compute r all reduce sum v_nope\nrun 0', 'Variable name nope for compute reduce does not exist'],
  ['compute r all reduce sum v_ve\nrun 0', 'Compute reduce variable ve is not atom-style variable'],
  ['compute r all reduce sum v_va c_ke\nrun 0', null],
  ['compute r all reduce sum x bogus\nrun 0', 'Unknown compute reduce keyword: bogus'],
  ['compute r all reduce sum x replace 1 2\nrun 0', 'Compute reduce replace requires min or max mode'],
  ['compute r all reduce max x y replace 1 2\nrun 0', null],
  ['compute r all reduce max x y replace 1 1\nrun 0', 'Compute {} replace columns must be different'],
  ['compute r all reduce max x y replace 1 3\nrun 0', 'Invalid compute reduce replace second column index 2'],
  ['compute r all reduce max x y replace 0 1\nrun 0', 'Invalid compute reduce replace first column index -1'],
  ['compute r all reduce max x y z replace 1 2 replace 2 3\nrun 0', null],
  ['compute r all reduce max x y replace 1\nrun 0', 'Illegal compute reduce replace command: missing argument(s)'],
  ['compute r all reduce max x y replace a 2\nrun 0', 'Expected integer parameter instead of \'a\' in input script or data file'],
  ['compute r all reduce sum c_pl inputs local\nrun 0', 'Compute reduce compute pl does not calculate a local vector'],
  ['compute r all reduce sum c_pl[1] inputs local\nrun 0', null],
  ['compute r all reduce sum c_pl1 inputs local\nrun 0', null],
  ['compute r all reduce sum c_pl1[1] inputs local\nrun 0', 'Compute reduce compute pl1 does not calculate a local array'],
  ['compute r all reduce sum c_pl[3] inputs local\nrun 0', 'Compute reduce compute pl array is accessed out-of-range'],
  ['compute r all reduce sum x inputs local\nrun 0', 'Compute {} inputs must be all local'],
  ['compute r all reduce sum v_va inputs local\nrun 0', 'Compute {} inputs must be all local'],
  ['compute r all reduce sum c_ke inputs local\nrun 0', 'Compute reduce compute ke does not calculate local values'],
  ['compute r all reduce sum c_pl1 inputs bogus\nrun 0', 'Unknown compute reduce inputs argument: bogus'],
  ['compute r all reduce sum c_pl1 inputs\nrun 0', 'Illegal compute reduce inputs command: missing argument(s)'],
  ['compute r all reduce sumsq x\nrun 0', null],
  ['compute r all reduce minabs x\nrun 0', null],
  ['compute r all reduce aveabs vx\nrun 0', null],
  ['compute r all reduce sum x\nuncompute ke\nrun 0', null],
  ['compute r all reduce sum c_ke\nuncompute ke\nrun 0', 'Compute ID ke for compute reduce does not exist'],
  ['compute r all reduce sum 3\nrun 0', 'Unknown compute reduce keyword: 3'],
  ['compute r all reduce sum c_\nrun 0', 'Unknown compute reduce keyword: c_'],
];

describe('compute reduce as native LAMMPS', () => {
  it.each(ROWS.map(([p, e]) => [p.replace(/\n/g, ' / '), p, e] as const))('%s', async (_name, p, expected) => {
    expect(await errorOf(p)).toBe(expected);
  });
});
