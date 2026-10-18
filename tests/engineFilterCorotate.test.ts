import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix filter/corotate (GPL port of src/EXTRA-FIX/fix_filter_corotate.cpp, src/engine-gpl/filtercorotate): the argument
 * and init checks, the cluster finder's errors and its counts. Native agreement of the filtered rRESPA runs is in
 * tests/oracle/w214fc_*.in.
 */

const DATA = readFileSync(join(__dirname, 'oracle', 'w214fc_clusters.data'), 'utf8');

const run = async (lines: string[], files: Record<string, string> = { 'c.data': DATA }): Promise<{ logs: string[]; error?: string }> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) s.addFile(n, t);
  let error: string | undefined;
  try {
    await s.execute(`${lines.join('\n')}\n`);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) error = String(err.message);
  return { logs: events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text), error };
};

const SETUP = [
  'units real', 'atom_style full', 'bond_style harmonic', 'angle_style harmonic', 'pair_style lj/cut/coul/cut 8.0',
  'read_data c.data', 'pair_coeff * * 0.05 3.0', 'bond_coeff 1 340.0 1.09', 'bond_coeff 2 268.0 1.53', 'bond_coeff 3 450.0 0.9572',
  'angle_coeff 1 33.0 109.5', 'angle_coeff 2 37.5 110.7', 'angle_coeff 3 55.0 104.52', 'timestep 2',
];
const RESPA = 'run_style respa 2 4 bond 1 angle 1 pair 2';

/** A chain of `n` carbon atoms along x, 1.53 A apart, with the given extra bonds; bond type 1 everywhere. */
const chain = (n: number, extra: Array<[number, number]> = [], ntot = n): string => {
  const lines = ['chain', '', `${ntot} atoms`, `${n - 1 + extra.length} bonds`, '', '1 atom types', '1 bond types', '',
    '0 40 xlo xhi', '0 40 ylo yhi', '0 40 zlo zhi', '', 'Masses', '', '1 12.011', '', 'Atoms # full', ''];
  for (let i = 1; i <= ntot; i++) lines.push(`${i} 1 1 0.0 ${5 + 1.53 * i} ${10 + (i > n ? 1.2 * (i - n) : 0)} 10`);
  lines.push('', 'Bonds', '');
  let k = 0;
  for (let i = 1; i < n; i++) lines.push(`${++k} 1 ${i} ${i + 1}`);
  for (const [a, b] of extra) lines.push(`${++k} 1 ${a} ${b}`);
  return `${lines.join('\n')}\n`;
};
const CHAIN_SETUP = ['units real', 'atom_style full', 'bond_style harmonic', 'pair_style lj/cut 8.0', 'read_data ch.data',
  'pair_coeff * * 0.05 3.0', 'bond_coeff 1 268.0 1.53'];

describe('fix filter/corotate', () => {
  it('logs the clusters it found, as native find_clusters does', async () => {
    const r = await run([...SETUP, RESPA, 'fix cor all filter/corotate m 1.008']);
    expect(r.error).toBeUndefined();
    // Measured with native LAMMPS on this data file (tests/oracle/w214fc_clusters_m.in): 1, 0, 2, 1 clusters of size
    // 2..5 and 2 frozen angles
    expect(r.logs).toEqual(expect.arrayContaining(['  1 = # of size 2 clusters', '  0 = # of size 3 clusters',
      '  2 = # of size 4 clusters', '  1 = # of size 5 clusters', '  2 = # of frozen angles']));
  });

  it('requires run_style respa and refuses fix shake beside it, as native init does', async () => {
    expect((await run([...SETUP, 'fix 1 all nve', 'fix cor all filter/corotate m 1.008', 'run 0'])).error).toMatch(/Fix filter\/corotate requires rRESPA!/);
    expect((await run([...SETUP, RESPA, 'fix s all shake 1e-4 20 0 b 3', 'fix cor all filter/corotate m 1.008', 'run 0'])).error)
      .toMatch(/Both fix shake and fix filter\/corotate detected\./);
    expect((await run([...SETUP, RESPA, 'fix a all filter/corotate m 1.008', 'fix b all filter/corotate b 3', 'run 0'])).error)
      .toMatch(/More than one fix filter\/corotate/);
  });

  it('refuses arguments as the native constructor does', async () => {
    expect((await run([...SETUP, 'fix cor all filter/corotate b 7'])).error).toMatch(/Invalid bond type index for fix filter\/corotate/);
    expect((await run([...SETUP, 'fix cor all filter/corotate a 9'])).error).toMatch(/Invalid angle type index for fix filter\/corotate/);
    expect((await run([...SETUP, 'fix cor all filter/corotate t 4'])).error).toMatch(/Invalid atom type index for fix filter\/corotate/);
    expect((await run([...SETUP, 'fix cor all filter/corotate m 0.0'])).error).toMatch(/Invalid atom mass for fix filter\/corotate/);
    expect((await run([...SETUP, 'fix cor all filter/corotate 3'])).error).toMatch(/Illegal fix filter\/corotate command/);
    const atomic = await run(['units real', 'atom_style atomic', 'region b block 0 10 0 10 0 10', 'create_box 1 b', 'fix cor all filter/corotate m 1.0']);
    expect(atomic.error).toMatch(/Cannot use fix filter\/corotate with non-molecular system/);
  });

  it('refuses connected clusters and clusters of more than five atoms (find_clusters checks)', async () => {
    // a chain of four carbons with every C-C bond selected: atoms 2 and 3 are both centres, and bonded
    expect((await run([...CHAIN_SETUP, 'fix cor all filter/corotate b 1'], { 'ch.data': chain(4) })).error).toMatch(/Clusters are connected/);
    // atom 1 with five selected partners
    const star = chain(2, [[1, 3], [1, 4], [1, 5], [1, 6]], 6);
    expect((await run([...CHAIN_SETUP, 'fix cor all filter/corotate b 1'], { 'ch.data': star })).error).toMatch(/Cluster of more than 5 atoms/);
  });

  it('keeps the charmm dihedrals on the pair level, as native dihedral charmm init_style does', async () => {
    const r = await run(['units real', 'atom_style full', 'bond_style harmonic', 'dihedral_style charmm', 'pair_style lj/charmm/coul/charmm 8.0 10.0',
      'read_data ch.data', 'pair_coeff * * 0.05 3.0', 'bond_coeff 1 268.0 1.53', 'dihedral_coeff 1 0.2 3 0 0.5',
      'run_style respa 2 2 bond 1 dihedral 1 pair 2', 'run 0'], {
      'ch.data': chain(4).replace('1 bond types', '1 bond types\n1 dihedral types').replace('3 bonds', '3 bonds\n1 dihedrals') + '\nDihedrals\n\n1 1 1 2 3 4\n',
    });
    expect(r.error).toMatch(/Dihedral style charmm must be set to same r-RESPA level as 'pair'/);
  });
});
