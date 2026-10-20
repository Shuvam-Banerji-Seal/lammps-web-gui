import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix gemc (src/engine-gpl/mc/fix_gemc.ts, ported from src/MC/fix_gemc.cpp at b332f28): refusals with native's text and
 * the invariants of Gibbs ensemble MC (the atoms of the two boxes and their total volume are conserved). The native
 * agreement is checked by the oracle cases tests/oracle/w246gm_*.in.
 */

const setup = (opts: { style?: string; extra?: string; fix?: string } = {}) => `
units lj
atom_style ${opts.style ?? 'atomic'}
variable rho world 0.2 0.5
variable ncell world 1 2
lattice fcc \${rho}
region box block 0 \${ncell} 0 \${ncell} 0 \${ncell}
create_box 1 box ${opts.style === 'bond' ? 'bond/types 1 extra/bond/per/atom 1' : ''}
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify every 1 delay 5 check yes
${opts.extra ?? ''}
${opts.fix ?? 'fix 2 all gemc 1 5 10 5 0.9 0.5 0.5 31'}
thermo 1
thermo_style custom step pe vol atoms
thermo_modify format float %.15g
`;

const run = async (text: string, partitions = 2, steps = 0) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, partitions);
  // the other partitions' thermo rows are passed on (as the oracle harness does for partitioned cases)
  session.recordPartitionOutput = true;
  let error: Error | null = null;
  try {
    await session.execute(`${text}run ${steps}\n`);
  } catch (e) {
    error = e as Error;
  }
  return { events, error };
};

describe('fix gemc refusals', () => {
  it('needs exactly two partitions (native: Must use exactly two partitions with fix gemc)', async () => {
    // without world variables: the fix refuses the partition count (measured with native LAMMPS on one partition)
    const plain = setup().replace('variable rho world 0.2 0.5\nvariable ncell world 1 2\n', '')
      .replace('lattice fcc ${rho}', 'lattice fcc 0.5').replace('block 0 ${ncell} 0 ${ncell} 0 ${ncell}', 'block 0 1 0 1 0 1');
    expect((await run(plain, 1)).error?.message).toMatch(/Must use exactly two partitions with fix gemc/);
    expect((await run(plain, 3)).error?.message).toMatch(/Must use exactly two partitions with fix gemc/);
    // with the two-value world variables, the variable command refuses first (measured with native LAMMPS:
    // World variable count 2 doesn't match # of partitions 1)
    expect((await run(setup(), 1)).error?.message).toMatch(/World variable count 2 doesn't match # of partitions 1/);
    expect((await run(setup(), 3)).error?.message).toMatch(/World variable count 2 doesn't match # of partitions 3/);
  });

  it('takes its eight arguments (native: Illegal fix gemc command: missing argument(s))', async () => {
    const text = setup({ fix: 'fix 2 all gemc 1 5 10 5 0.9 0.5 0.5' });
    expect((await run(text)).error?.message).toMatch(/Illegal fix gemc command: missing argument\(s\)/);
  });

  it('refuses molecular atom styles (native: Using fix gemc with molecules not (yet) supported)', async () => {
    expect((await run(setup({ style: 'bond' }))).error?.message).toMatch(/Using fix gemc with molecules not \(yet\) supported/);
  });

  it('refuses nevery below 1, naming it', async () => {
    expect((await run(setup({ fix: 'fix 2 all gemc 0 5 10 5 0.9 0.5 0.5 31' }))).error?.message).toMatch(/N must be a positive integer/);
  });

  it('runs with pair_modify tail yes and holds the tail correction until the next run (native compared in w313gm_lj_tail)', async () => {
    const { error } = await run(setup({ extra: 'pair_modify tail yes\n' }), 2, 20);
    expect(error).toBeNull();
    // the next run's init recomputes the tail correction from the current atom counts (Pair::init) and holds it
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, 2);
    await session.execute(`${setup({ extra: 'pair_modify tail yes\n' })}run 20\nunfix 2\nrun 0\n`);
    for (const sys of session.replicaSystems) {
      const held = sys.ff.tailHold!;
      expect(held).not.toBeNull();
      sys.ff.tailHold = null;
      sys.ff.updateTail(sys.state);
      expect(held.etail).toBe(sys.ff.etailV);
      expect(held.ptail).toBe(sys.ff.ptailV);
    }
  });

  it('refuses a seed that RanPark cannot use', async () => {
    expect((await run(setup({ fix: 'fix 2 all gemc 1 5 10 5 0.9 0.5 0.5 0' }))).error?.message).toMatch(/Invalid seed for Park random # generator/);
  });
});

describe('fix gemc invariants', () => {
  it('conserves the atoms and the total volume of the two boxes, and moves atoms and volumes', async () => {
    const { events, error } = await run(setup(), 2, 60);
    expect(error).toBeNull();
    const rows: Record<number, { step: number; atoms: number; vol: number }[]> = { 0: [], 1: [] };
    for (const e of events) {
      // replica 0's rows are the session's own output (untagged); the others carry their partition
      if (e.kind !== 'thermo') continue;
      rows[e.partition ?? 0].push({ step: e.row.step, atoms: e.row.atoms, vol: e.row.vol });
    }
    expect(rows[0].length).toBe(61);
    expect(rows[1].length).toBe(61);
    const totalAtoms = rows[0][0].atoms + rows[1][0].atoms;
    const totalVol = rows[0][0].vol + rows[1][0].vol;
    let transfers = 0;
    let volumeChanges = 0;
    for (let k = 0; k < rows[0].length; k++) {
      const a = rows[0][k], b = rows[1][k];
      expect(a.atoms + b.atoms).toBe(totalAtoms);
      expect(Math.abs(a.vol + b.vol - totalVol)).toBeLessThan(1e-9 * totalVol);
      if (k > 0) {
        if (a.atoms !== rows[0][k - 1].atoms) transfers++;
        if (a.vol !== rows[0][k - 1].vol) volumeChanges++;
      }
    }
    expect(transfers).toBeGreaterThan(0);
    expect(volumeChanges).toBeGreaterThan(0);
  });
});
