import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Argument and refusal checks of the styles ported in this change (src/engine-gpl): fix settorque/atom,
 * compute frenkel, pair_style bondval and bondval/vec. The numbers are checked against native LAMMPS by the oracle
 * cases tests/oracle/w166px_* (tests/engineOracle.test.ts); these tests only cover what the input is refused for.
 */

const runScript = async (text: string): Promise<{ error: Error | null }> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: Error | null = null;
  try {
    await s.execute(text);
  } catch (e) {
    error = e instanceof Error ? e : new Error(String(e));
  }
  const err = events.find((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error');
  return { error: error ?? (err ? new Error(err.message) : null) };
};

const SPHERE = `units lj
atom_style sphere
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 1.0 1.0 1.0
create_atoms 1 single 2.0 1.0 1.0
set atom * diameter 1.0
set atom * mass 1.0
pair_style zero 1.0
pair_coeff * *
`;

const ATOMIC = `units lj
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 1.0 1.0 1.0
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
`;

describe('fix settorque/atom', () => {
  it('does not support fix_modify energy', async () => {
    const { error } = await runScript(`${SPHERE}fix 1 all settorque/atom 0.0 0.0 0.0\nfix_modify 1 energy yes\n`);
    expect(error?.message ?? '').toMatch(/does not support fix_modify energy/);
  });

  it('names an unknown keyword', async () => {
    const { error } = await runScript(`${SPHERE}fix 1 all settorque/atom 0.0 0.0 0.0 bogus 1\n`);
    expect(error?.message ?? '').toMatch(/Unknown fix settorque\/atom keyword: bogus/);
  });

  it('needs a torque atom style', async () => {
    const { error } = await runScript(`${ATOMIC}fix 1 all settorque/atom 0.0 0.0 0.0\nrun 0\n`);
    expect(error?.message ?? '').toMatch(/requires atom attribute torque/);
  });
});

describe('compute frenkel', () => {
  it('refuses rescale yes', async () => {
    const { error } = await runScript(`${ATOMIC}lattice fcc 0.8442\ncompute fr all frenkel rescale yes\nrun 0\n`);
    expect(error?.message ?? '').toMatch(/rescale yes is not supported/);
  });

  it('refuses site_file', async () => {
    const { error } = await runScript(`${ATOMIC}lattice fcc 0.8442\ncompute fr all frenkel site_file sites.txt\nrun 0\n`);
    expect(error?.message ?? '').toMatch(/site_file is not supported/);
  });

  it('needs a positive drvac', async () => {
    const { error } = await runScript(`${ATOMIC}lattice fcc 0.8442\ncompute fr all frenkel drvac 0.0\n`);
    expect(error?.message ?? '').toMatch(/drvac value must be > 0.0/);
  });

  it('needs a lattice', async () => {
    const { error } = await runScript(`${ATOMIC}compute fr all frenkel\nrun 0\n`);
    expect(error?.message ?? '').toMatch(/undefined lattice/);
  });
});

describe('pair_style bondval', () => {
  it('takes power and cutoff', async () => {
    const { error } = await runScript(`${ATOMIC}pair_style bondval 2.0\n`);
    expect(error?.message ?? '').toMatch(/Illegal pair_style command/);
  });

  it('takes r0 alpha S V0 and an optional cutoff', async () => {
    const { error } = await runScript(`units metal
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 1.0 1.0 1.0
pair_style bondval 2.0 8.0
pair_coeff 1 1 0.0 5.0 0.5
`);
    // pair_bondval.cpp coeff(): 4 to 7 words pass the count, then V0 (past the line) is read as NULL
    expect(error?.message ?? '').toMatch(/Expected floating point parameter instead of NULL or empty string/);
  });

  it('requires every I,I pair (native Pair::init)', async () => {
    const { error } = await runScript(`units metal
atom_style atomic
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 1.0 1.0 1.0
create_atoms 2 single 2.0 1.0 1.0
mass * 1.0
pair_style bondval 2.0 8.0
pair_coeff 1 1 0.0 5.0 0.5 2.0
run 0
`);
    expect(error?.message ?? '').toMatch(/All pair coeffs are not set\. Status:/);
  });
});
