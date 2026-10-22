import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * compute property/local as native LAMMPS reads it (src/compute_property_local.cpp at b332f28): the attributes of one
 * kind (neighbor, pair, bond, angle, dihedral, improper), the cutoff keyword, the topology the atom style stores, the
 * radius of the radius cutoff and the pair style of the pair attributes. Measured with native LAMMPS
 * (/home/roy/.local/bin/lmp, b332f28): each row is the input below for the atom style with its lines added, null when
 * it runs, else the first line of native's message. The pair rows themselves are compared with native by the oracle
 * cases tests/oracle/propl_*.in.
 */

const input = (style: string, post: string): string => {
  const box = style === 'bond' || style === 'full' ? ' bond/types 1 extra/bond/per/atom 2' : '';
  return `units lj
atom_style ${style}
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 2 box${box}
create_atoms 1 box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
${post}
`;
};

const errorOf = async (style: string, post: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(input(style, post)); } catch (e) { return (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return null;
};

const ROWS: [string, string, string | null][] = [
  ['atomic', 'compute p all property/local\nrun 0', 'Illegal compute property/local command: missing argument(s)'],
  ['atomic', 'compute p all property/local bogus\nrun 0', 'Unknown compute property/local keyword: bogus'],
  ['atomic', 'compute p all property/local patom1 batom1\nrun 0', 'Compute property/local cannot use these inputs together'],
  ['atomic', 'compute p all property/local patom1 natom1\nrun 0', 'Compute property/local cannot use these inputs together'],
  ['atomic', 'compute p all property/local batom1\nrun 0', 'Compute property/local for property that isn\'t allocated'],
  ['bond', 'compute p all property/local atype\nrun 0', 'Compute property/local for property that isn\'t allocated'],
  ['bond', 'compute p all property/local batom1 batom2 btype\nrun 0', null],
  ['full', 'compute p all property/local datom1 dtype\nrun 0', null],
  ['atomic', 'compute p all property/local patom1 cutoff\nrun 0', 'Illegal compute property/local cutoff command: missing argument(s)'],
  ['atomic', 'compute p all property/local patom1 cutoff bogus\nrun 0', 'Unknown compute property/local cutoff keyword: bogus'],
  ['atomic', 'compute p all property/local patom1 cutoff radius\nrun 0', 'Compute property/local with ID p requires atom attribute radius'],
  ['atomic', 'compute p all property/local patom1 cutoff type\nrun 0', null],
  ['atomic', 'compute p all property/local cutoff type\nrun 0', null],
  ['atomic', 'compute p all property/local patom1 patom2 bogus\nrun 0', 'Unknown compute property/local keyword: bogus'],
  ['atomic', 'pair_style none\ncompute p all property/local patom1\nrun 0', 'No pair style is defined for compute property/local'],
  ['atomic', 'pair_style none\ncompute p all property/local natom1 ntype2\nrun 0', 'No pair style is defined for compute property/local'],
  ['sphere', 'compute p all property/local patom1 cutoff radius\nrun 0', 'Cannot set per-type atom mass for atom style sphere'],
  ['atomic', 'compute p all property/local natom1 cutoff radius\nrun 0', 'Compute property/local with ID p requires atom attribute radius'],
];

describe('compute property/local as native LAMMPS', () => {
  it.each(ROWS.map(([st, p, e]) => [`${st}: ${p.replace(/\n/g, ' / ')}`, st, p, e] as const))('%s', async (_name, st, p, expected) => {
    expect(await errorOf(st, p)).toBe(expected);
  });
});
