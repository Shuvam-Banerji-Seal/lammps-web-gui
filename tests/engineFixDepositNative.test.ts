import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix deposit as native LAMMPS reads it (src/fix_deposit.cpp FixDeposit::FixDeposit, options and init at b332f28): the
 * required values, every keyword (var and set, gaussian, global and local, molfrac, orient, rigid, shake, units, id,
 * attempt, near, rate, target), the variables, the type, the region's existence, bounding box and extent, and the
 * region at the run's init. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28): each row is the input
 * below with its lines added, null when it runs, else the first line of native's message. The insertions themselves
 * are checked against native by the oracle cases (tests/oracle/deposit_*.in and the w*dep* cases).
 */

const BASE = `units lj
atom_style atomic
boundary p p f
lattice fcc 0.8442
region box block 0 5 0 5 0 10
create_box 2 box
region slab block 0 5 0 5 0 1
create_atoms 1 region slab
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
region top block 0 5 0 5 8 9
region far block 0 5 0 5 8 12
region sph sphere 2.5 2.5 8.5 1.0
region out block 0 5 0 5 8 9 side out
variable v equal 1
variable s string abc
`;

const errorOf = async (post: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(`${BASE}${post}\n`); } catch (e) { return (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return null;
};

const ROWS: [string, string | null][] = [
  ['fix 1 all deposit 5 1 10\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 0 region top\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 -3 region top\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit x 1 10 12345 region top\nrun 1', 'Expected integer parameter instead of \'x\' in input script or data file'],
  ['fix 1 all deposit 5 1 x 12345 region top\nrun 1', 'Expected integer parameter instead of \'x\' in input script or data file'],
  ['fix 1 all deposit 5 x 10 12345 region top\nrun 1', 'Expected integer parameter instead of \'x\' in input script or data file'],
  ['fix 1 all deposit 5 3 10 12345 region top\nrun 1', 'Invalid atom type in fix deposit command'],
  ['fix 1 all deposit 5 0 10 12345 region top\nrun 1', 'Invalid atom type in fix deposit command'],
  ['fix 1 all deposit 0 1 10 12345 region top\nrun 20', null],
  ['fix 1 all deposit -1 1 10 12345 region top\nrun 20', null],
  ['fix 1 all deposit 5 1 10 12345\nrun 1', 'Must specify a region in fix deposit'],
  ['fix 1 all deposit 5 1 10 12345 region nope\nrun 1', 'Region ID nope for fix deposit does not exist'],
  ['fix 1 all deposit 5 1 10 12345 region\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region far\nrun 1', 'Deposition region extends outside simulation box'],
  ['fix 1 all deposit 5 1 10 12345 region out\nrun 1', 'Fix deposit region does not support a bounding box'],
  ['fix 1 all deposit 5 1 10 12345 region top bogus 1\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region top var\nrun 1', 'Illegal fix deposit var command: missing argument(s)'],
  ['fix 1 all deposit 5 1 10 12345 region top var v\nrun 1', 'Incomplete use of variables in fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region top set x xx\nrun 1', 'Incomplete use of variables in fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region top var nope set x xx\nrun 1', 'Variable nope for fix deposit does not exist'],
  ['fix 1 all deposit 5 1 10 12345 region top var s set x xx\nrun 1', 'Variable for fix deposit is invalid style'],
  ['fix 1 all deposit 5 1 10 12345 region top var v set q xx\nrun 1', 'Unknown fix deposit set option xx'],
  ['fix 1 all deposit 5 1 10 12345 region top var v set x\nrun 1', 'Illegal fix deposit set command: missing argument(s)'],
  ['fix 1 all deposit 5 1 10 12345 region top var v set x s\nrun 1', 'Variable s for fix deposit is invalid style'],
  ['fix 1 all deposit 5 1 10 12345 region top var v set x v\nrun 1', 'Variable v for fix deposit is invalid style'],
  ['fix 1 all deposit 5 1 10 12345 region top var v set x xx\nrun 1', null],
  ['fix 1 all deposit 5 1 10 12345 region top gaussian 1 2 3\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region top gaussian 2.5 2.5 8.5 0.3\nrun 1', null],
  ['fix 1 all deposit 5 1 10 12345 region top gaussian a 2.5 8.5 0.3\nrun 1', 'Expected floating point parameter instead of \'a\' in input script or data file'],
  ['fix 1 all deposit 5 1 10 12345 region top global 0 1 local 0 1 2\nrun 1', null],
  ['fix 1 all deposit 5 1 10 12345 region top local 0 1 2 global 0 1\nrun 1', null],
  ['fix 1 all deposit 5 1 10 12345 region top global 1 0\nrun 1', null],
  ['fix 1 all deposit 5 1 10 12345 region top vx 1 0\nrun 1', null],
  ['fix 1 all deposit 5 1 10 12345 region top orient 0 0 1\nrun 1', null],
  ['fix 1 all deposit 5 1 10 12345 region top orient 0 0 0\nrun 1', 'Illegal fix deposit orient settings'],
  ['fix 1 all deposit 5 1 10 12345 region top rigid r\nrun 1', 'Cannot use fix deposit rigid and not molecule'],
  ['fix 1 all deposit 5 1 10 12345 region top shake s\nrun 1', 'Cannot use fix deposit shake and not molecule'],
  ['fix 1 all deposit 5 1 10 12345 region top molfrac 0.5 0.5\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region top mol nope\nrun 1', 'Molecule template ID for fix deposit does not exist'],
  ['fix 1 all deposit 5 1 10 12345 region top units bogus\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region top id bogus\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region top attempt x\nrun 1', 'Expected integer parameter instead of \'x\' in input script or data file'],
  ['fix 1 all deposit 5 1 10 12345 region top near x\nrun 1', 'Expected floating point parameter instead of \'x\' in input script or data file'],
  ['fix 1 all deposit 5 1 10 12345 region top rate x\nrun 1', 'Expected floating point parameter instead of \'x\' in input script or data file'],
  ['fix 1 all deposit 5 1 10 12345 region top target 1 2\nrun 1', 'Illegal fix deposit command'],
  ['fix 1 all deposit 5 1 10 12345 region top\nregion top delete\nrun 1', 'Region ID top for fix deposit does not exist'],
];

describe('fix deposit as native LAMMPS', () => {
  it.each(ROWS.map(([p, e]) => [p.replace(/\n/g, ' / '), p, e] as const))('%s', async (_name, p, expected) => {
    expect(await errorOf(p)).toBe(expected);
  });
});

describe('fix deposit inputs native LAMMPS does not finish', () => {
  // M = 0: native divides by M and stops with a floating-point exception; the engine refuses it
  it.each(['fix 1 all deposit 5 1 0 12345 region top\nrun 3'])('%s', async (p) => {
    expect(await errorOf(p)).toBe('fix deposit with M = 0 is refused: native LAMMPS stops with a floating-point exception for it');
  });
});
