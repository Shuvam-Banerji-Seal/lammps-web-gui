import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * The timing breakdown after a run (src/engine/timer.ts): native's "MPI task timing breakdown" table of src/finish.cpp
 * (Pair, Bond for molecular systems, Kspace with a kspace style, Neigh, Comm, Output, Modify, Other), printed for
 * timer normal (the default) and full, not for timer loop or off (timer.html).
 */

const LJ = 'units lj\natom_style atomic\nlattice fcc 0.8442\nregion box block 0 4 0 4 0 4\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nvelocity all create 3.0 87287\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix 1 all nve\n';

const logsOf = async (text: string): Promise<string[]> => {
  const logs: string[] = [];
  const s = new Session({ emit: (e: EngineEvent) => { if (e.kind === 'log') logs.push(e.text); }, writeFile: () => {} });
  await s.execute(text);
  return logs;
};

const ROW = /^(Pair|Bond|Kspace|Neigh|Comm|Output|Modify) {2,4}\| [0-9.e+-]+ +\| [0-9.e+-]+ +\| [0-9.e+-]+ +\| {3}0\.0 \| *[0-9.]+$/;

describe('timing breakdown', () => {
  it('follows the loop time in native\'s format, with the sections of an atomic system', async () => {
    const logs = await logsOf(`${LJ}run 50\n`);
    const k = logs.findIndex((l) => l === 'MPI task timing breakdown:');
    expect(logs[k - 2]).toMatch(/^Loop time of /);
    expect(logs[k - 1]).toBe('');
    expect(logs[k + 1]).toBe('Section |  min time  |  avg time  |  max time  |%varavg| %total');
    expect(logs[k + 2]).toBe('-'.repeat(63));
    const rows = logs.slice(k + 3, k + 9);
    expect(rows.map((r) => r.split('|')[0].trim())).toEqual(['Pair', 'Neigh', 'Comm', 'Output', 'Modify', 'Other']);
    for (const r of rows.slice(0, 5)) expect(r).toMatch(ROW);
    expect(rows[5]).toMatch(/^Other {3}\| {12}\| [0-9.e+-]+ +\| {12}\| {7}\| *[0-9.]+$/);
    // the sections and Other add up to the loop
    const pct = rows.map((r) => Number(r.split('|').pop()));
    expect(pct.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 0);
  });

  it('adds Bond for a molecular system and Kspace with a kspace style', async () => {
    const logs = await logsOf(`units real
atom_style full
region box block -6 6 -6 6 -6 6
create_box 1 box bond/types 1 extra/bond/per/atom 1
mass 1 12.0
create_atoms 1 single 0 0 0
create_atoms 1 single 1.5 0 0
create_bonds single/bond 1 1 2
set atom 1 charge 0.5
set atom 2 charge -0.5
pair_style lj/cut/coul/long 4.0
pair_coeff 1 1 0.1 3.0
bond_style harmonic
bond_coeff 1 100 1.5
special_bonds lj/coul 0 1 1
kspace_style ewald 1e-4
fix 1 all nve
run 10
`);
    const k = logs.findIndex((l) => l === 'MPI task timing breakdown:');
    expect(logs.slice(k + 3, k + 11).map((r) => r.split('|')[0].trim())).toEqual(['Pair', 'Bond', 'Kspace', 'Neigh', 'Comm', 'Output', 'Modify', 'Other']);
  });

  it('timer loop and timer off print no breakdown; timer full and normal do', async () => {
    for (const [level, shown] of [['loop', false], ['off', false], ['full', true], ['normal', true]] as const) {
      const logs = await logsOf(`${LJ}timer ${level}\nrun 10\n`);
      expect(logs.includes('MPI task timing breakdown:'), level).toBe(shown);
    }
  });
});
