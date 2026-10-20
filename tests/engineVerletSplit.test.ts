import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * run_style verlet/split and verlet/split/rk (docs.lammps.org/run_style.html) and kspace_style pppm/rk on two
 * partitions: the Rspace partition's run is the run of run_style verlet with kspace_style pppm, the Kspace partition
 * writes no output and ends the run with the Rspace partition's coordinates. Native parity: tests/oracle/w303vsplit_*.in.
 */

const SETUP = `units real
atom_style charge
atom_modify map array
lattice sc 2.8
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 box
mass * 22.99
set type 1 charge 1.0
set group all type/fraction 2 0.5 12345
set type 2 charge -1.0
pair_style lj/cut/coul/long 8.0
pair_coeff * * 0.1 2.5
kspace_style pppm 1.0e-5
velocity all create 300 4928459
fix 1 all nve
thermo 10
thermo_style custom step temp pe ecoul elong press
`;

const run = async (text: string, partitions: number) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, partitions);
  session.recordPartitionOutput = true;
  let error: Error | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as Error;
  }
  const rows = (k: number) => events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo' && (e.partition ?? 0) === k).map((e) => e.row);
  return { session, error, rows };
};

describe('run_style verlet/split', () => {
  it('needs two partitions (native: Verlet/split requires 2 partitions)', async () => {
    expect((await run(`${SETUP}run_style verlet/split\n`, 1)).error?.message).toMatch(/Verlet\/split requires 2 partitions/);
    expect((await run(`${SETUP}run_style verlet/split\n`, 3)).error?.message).toMatch(/Verlet\/split requires 2 partitions/);
  });

  it('runs the Rspace partition as run_style verlet; the Kspace partition prints nothing and takes its coordinates', async () => {
    const plain = await run(`${SETUP}run 50\n`, 1);
    const split = await run(`${SETUP}run_style verlet/split\nrun 50\n`, 2);
    expect(plain.error).toBeNull();
    expect(split.error).toBeNull();
    expect(split.rows(0)).toEqual(plain.rows(0));
    expect(split.rows(0).length).toBe(6);
    expect(split.rows(1).length).toBe(0);
    const [r, k] = split.session.replicaSystems;
    expect(k.state.step).toBe(50);
    for (let i = 0; i < k.state.n; i++) { const j = r.indexOfId(k.state.id[i]); expect([k.state.x[3 * i], k.state.x[3 * i + 1], k.state.x[3 * i + 2]]).toEqual([r.state.x[3 * j], r.state.x[3 * j + 1], r.state.x[3 * j + 2]]); }
    // its own velocities stay (measured with native LAMMPS: after the run partition 2 has its initial velocities)
    expect(Array.from(k.state.v.subarray(0, 3))).not.toEqual(Array.from(r.state.v.subarray(0, 3)));
  });

  it('runs pppm/rk with verlet/split/rk as pppm with verlet, also with kspace_modify diff ad', async () => {
    const plain = await run(`${SETUP}run 30\n`, 1);
    const rk = await run(`${SETUP.replace('kspace_style pppm 1.0e-5', 'kspace_style pppm/rk 1.0e-5')}run_style verlet/split/rk\nrun 30\n`, 2);
    expect(rk.error).toBeNull();
    expect(rk.rows(0)).toEqual(plain.rows(0));
    const plainAd = await run(`${SETUP}kspace_modify diff ad\nrun 30\n`, 1);
    const rkAd = await run(`${SETUP.replace('kspace_style pppm 1.0e-5', 'kspace_style pppm/rk 1.0e-5')}kspace_modify diff ad\nrun_style verlet/split/rk\nrun 30\n`, 2);
    expect(rkAd.error).toBeNull();
    expect(rkAd.rows(0)).toEqual(plainAd.rows(0));
    expect(rkAd.session.replicaSystems[0].ff.kspace?.rkFlag).toBe(true);
  });

  it('refuses the pairings native refuses, with native text', async () => {
    const rkStyle = SETUP.replace('kspace_style pppm 1.0e-5', 'kspace_style pppm/rk 1.0e-5');
    expect((await run(`${rkStyle}run 0\n`, 2)).error?.message)
      .toMatch(/The Integrate type must support an rk decomposition \(e\.g\., run_style verlet\/split\/rk\) that is compatible with kspace style pppm\/rk/);
    expect((await run(`${rkStyle}run_style verlet/split\nrun 0\n`, 2)).error?.message).toMatch(/The Integrate type must support an rk decomposition/);
    expect((await run(`${SETUP}run_style verlet/split/rk\nrun 0\n`, 2)).error?.message)
      .toMatch(/The kspace style pppm does not support an rk decomposition\. Verlet\/split\/rk requires a kspace style \(such as pppm\/rk\) that supports an rk decomposition\./);
    expect((await run(`${SETUP}kspace_style none\nrun_style verlet/split\nrun 0\n`, 2)).error?.message).toMatch(/A KSpace style must be defined with verlet\/split/);
    expect((await run(`${SETUP}kspace_style none\nrun_style verlet/split/rk\nrun 0\n`, 2)).error?.message).toMatch(/A KSpace style must be defined with verlet\/split\/rk/);
    expect((await run(`${rkStyle}run_style verlet/split/rk\nminimize 0 1e-6 10 100\n`, 2)).error?.message).toMatch(/minimize with kspace_style pppm\/rk is not supported/);
  });
});
