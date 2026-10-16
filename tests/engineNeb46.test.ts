import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Partitions and the neb command (docs.lammps.org/neb.html, fix_neb.html,
 * Run_options.html -partition): parsing, refusals, and the unchanged
 * single-partition behaviour. Numeric parity with native LAMMPS is in
 * tests/oracle/w46neb_*.in (engineOracle.test.ts).
 */

/** A 3-atom 2d lj/cut setup like tests/oracle/w46neb_tiny.in, with the final coordinates as a file. */
const SETUP = `dimension 2
boundary p s p
units lj
atom_style atomic
atom_modify map array
region box block 0 6 0 6 -0.5 0.5
create_box 1 box
create_atoms 1 single 1.5 1.5 0
create_atoms 1 single 3.0 1.7 0
create_atoms 1 single 4.5 1.5 0
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
pair_modify shift yes
variable u uloop 3
fix 1 all enforce2d
fix 2 all neb 1.0 parallel ideal
timestep 0.1
min_style quickmin
`;
const FINAL = '1\n2 2.0 2.6 0\n';

/** Runs the setup with extra input after it; returns the events and the error (if any). */
const run = async (extra: string, partitions = 3, setup = SETUP) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, partitions);
  session.addFile('final.tiny', FINAL);
  let error: Error | null = null;
  try {
    await session.execute(setup + extra);
  } catch (e) {
    error = e as Error;
  }
  return { events, error, session };
};

describe('partitions', () => {
  it('rejects a partition count below 1', () => {
    expect(() => new Session({ emit: () => undefined }, undefined, 0, 0)).toThrow(/partitions/);
    expect(() => new Session({ emit: () => undefined }, undefined, 0, 1.5)).toThrow(/partitions/);
  });

  it('keeps one system per replica and gives uloop variables the replica value', async () => {
    const { session, error } = await run('', 3);
    expect(error).toBeNull();
    expect(session.replicaSystems.length).toBe(3);
    for (let k = 0; k < 3; k++) {
      expect(session.replicaSystems[k].vars.get('u')?.which).toBe(k);
      expect(session.replicaSystems[k].state.n).toBe(3);
    }
  });

  it('runs an ordinary command on every replica', async () => {
    const { session, error } = await run('', 2);
    expect(error).toBeNull();
    // the same setup in both replicas, each with its own state
    expect(session.replicaSystems[1].state.n).toBe(session.replicaSystems[0].state.n);
  });

  it('keeps the single-partition session as before (no replica loop, one system)', async () => {
    const plain = SETUP.replace('fix 2 all neb 1.0 parallel ideal\n', '');
    const { session, error, events } = await run('thermo 1\nrun 0\n', 1, plain);
    expect(error).toBeNull();
    expect(session.replicaSystems.length).toBe(1);
    expect(events.filter((e) => e.kind === 'thermo').length).toBe(1);
  });

  it('runs a next/jump loop on every partition (one input stream, each replica advances its own loop variable)', async () => {
    const { session, error } = await run('variable i loop 3\nlabel top\nnext i\njump SELF top\n', 2);
    expect(error).toBeNull();
    // the loop ran out on both replicas, so next deleted the loop variable on each
    for (const sys of session.replicaSystems) expect(sys.vars.get('i')).toBeUndefined();
  });

  it('refuses a loop whose variable runs out at a different pass on different partitions', async () => {
    const { error } = await run('variable n world 2 3\nvariable i loop ${n}\nlabel top\nnext i\njump SELF top\n', 2);
    expect(error?.message).toMatch(/run out at different times/);
  });
});

describe('neb command', () => {
  it('needs two or more partitions', async () => {
    const { error } = await run('neb 0.0 0.01 10 10 10 none\n', 1);
    expect(error?.message).toMatch(/two or more partitions/);
  });

  it('checks its argument form', async () => {
    expect((await run('neb 0.0 0.01 10\n')).error?.message).toMatch(/usage: neb/);
    expect((await run('neb 0.0 0.01 10 10 7 none\n')).error?.message).toMatch(/multiples of Nevery/);
    expect((await run('neb 0.0 0.01 10 10 10 sideways\n')).error?.message).toMatch(/file-style must be final, each or none/);
    expect((await run('neb 0.0 0.01 10 10 10 final\n')).error?.message).toMatch(/needs a filename/);
  });

  it('needs min_style quickmin', async () => {
    const { error } = await run('min_style cg\nneb 0.0 0.01 10 10 10 final final.tiny\n');
    expect(error?.message).toMatch(/min_style quickmin/);
  });

  it('needs fix neb in every replica', async () => {
    const setup = SETUP.replace('fix 2 all neb 1.0 parallel ideal\n', '');
    const { error } = await run('neb 0.0 0.01 10 10 10 none\n', 3, setup);
    expect(error?.message).toMatch(/fix neb/);
  });

  it('refuses the settings the browser engine does not run yet, by name', async () => {
    const cases: [string, RegExp][] = [
      ['fix 2 all neb 1.0 perp 1.0\n', /perp/],
      ['fix 2 all neb 1.0 end first 1.0\n', /end/],
      ['fix 2 all neb 1.0 parallel equal\n', /equal/],
    ];
    for (const [fix, want] of cases) {
      const setup = SETUP.replace('fix 2 all neb 1.0 parallel ideal\n', fix);
      const { error } = await run('neb 0.0 0.01 10 10 10 none\n', 3, setup);
      expect(error?.message, fix).toMatch(want);
    }
    const verbose = await run('neb 0.0 0.01 10 10 10 none verbosity verbose\n');
    expect(verbose.error?.message).toMatch(/verbosity verbose/);
  });

  it('refuses fix neb outside the neb command instead of ignoring its forces', async () => {
    expect((await run('minimize 0 0.01 10 10\n')).error?.message).toMatch(/acts only inside a neb command/);
    expect((await run('run 0\n', 1)).error?.message).toMatch(/acts only inside a neb command/);
  });

  it('refuses fix neb keywords it does not know', async () => {
    const setup = SETUP.replace('fix 2 all neb 1.0 parallel ideal\n', 'fix 2 all neb 1.0 sideways 2\n');
    const { error } = await run('', 3, setup);
    expect(error?.message).toMatch(/illegal fix neb keyword/);
  });

  it('reads a malformed final file as an error, not as coordinates', async () => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, 3);
    session.addFile('final.tiny', '2\n2 2.0 2.6 0\n');
    await expect(session.execute(SETUP + 'neb 0.0 0.01 10 10 10 final final.tiny\n')).rejects.toThrow(/lists 2 atoms/);
  });

  it('emits the NEB table as thermo rows and the climbing line', async () => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, 3);
    session.addFile('final.tiny', FINAL);
    await session.execute(SETUP + 'neb 0.0 0.01 10 10 10 final final.tiny\n');
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo' && 'RD1' in e.row).map((e) => e.row);
    // the table has a row at step 0, at each Nevery multiple of stage 1, the stage-2 start row and its own rows
    expect(rows[0].Step).toBe(0);
    expect(Object.keys(rows[0])).toEqual(expect.arrayContaining(['MaxReplicaForce', 'MaxAtomForce', 'GradV0', 'GradV1', 'GradVc', 'EBF', 'EBR', 'RDT', 'RD1', 'PE1', 'RD2', 'PE2', 'RD3', 'PE3']));
    expect(rows[0].RD1).toBe(0);
    expect(rows[0].RD3).toBe(1);
    expect(events.some((e) => e.kind === 'log' && e.text === 'Climbing replica = 3')).toBe(true);
    expect(events.some((e) => e.kind === 'thermo-header')).toBe(true);
  });
});
