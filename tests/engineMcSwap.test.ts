import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * The MC-package fixes ported into src/engine-gpl/mcswap (fix atom/swap, fix tfmc, fix bond/create/angle and fix
 * bond/swap). The Monte Carlo itself (the random streams, the accept counts, the energies and the displacements) is
 * pinned by the w224mc_* oracle cases; these tests cover the argument errors, with native's messages, and the
 * refusals of the options this browser engine does not evaluate.
 */

const run = async (script: string): Promise<{ error: string | null; events: EngineEvent[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  return { error, events };
};

const LJ2 = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 4242
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
velocity all create 1.0 87287 loop geom
fix integ all nve
`;

describe('fix atom/swap argument errors', () => {
  it('rejects nevery, ncycles, seed and temperature values native rejects', async () => {
    expect((await run(LJ2 + 'fix sw all atom/swap 0 1 12345 1.0 types 1 2\nrun 1\n')).error).toContain('Illegal fix atom/swap command nevery value');
    expect((await run(LJ2 + 'fix sw all atom/swap 1 -1 12345 1.0 types 1 2\nrun 1\n')).error).toContain('Illegal fix atom/swap command ncycles value');
    expect((await run(LJ2 + 'fix sw all atom/swap 1 1 0 1.0 types 1 2\nrun 1\n')).error).toContain('Illegal fix atom/swap command random seed');
    expect((await run(LJ2 + 'fix sw all atom/swap 1 1 12345 0.0 types 1 2\nrun 1\n')).error).toContain('Illegal fix atom/swap command temperature value');
  });

  it('requires two swap types, and mu values only with semi-grand yes', async () => {
    const one = await run(LJ2 + 'fix sw all atom/swap 1 1 12345 1.0 types 1 ke yes\nrun 1\n');
    expect(one.error).toContain('Must specify at least 2 atom types in fix atom/swap command');
    const mu = await run(LJ2 + 'fix sw all atom/swap 1 1 12345 1.0 types 1 2 mu 0.1 0.2\nrun 1\n');
    expect(mu.error).toContain('Mu not allowed when not using semi-grand in fix atom/swap command');
    const semi = await run(LJ2 + 'fix sw all atom/swap 1 1 12345 1.0 types 1 2 semi-grand yes\nrun 1\n');
    expect(semi.error).toContain('Need nswaptypes mu values in fix atom/swap command');
  });

  it('names an unknown keyword and a missing region', async () => {
    expect((await run(LJ2 + 'fix sw all atom/swap 1 1 12345 1.0 types 1 2 bogus yes\n')).error).toContain('Illegal fix atom/swap command');
    expect((await run(LJ2 + 'fix sw all atom/swap 1 1 12345 1.0 types 1 2 region nope\n')).error).toContain('Region nope for fix atom/swap does not exist');
  });
});

describe('fix tfmc argument errors', () => {
  it('rejects a non-positive displacement, temperature or seed', async () => {
    expect((await run(LJ2 + 'fix mc all tfmc 0.0 1.0 12345\n')).error).toContain('Fix tfmc displacement length must be > 0');
    expect((await run(LJ2 + 'fix mc all tfmc 0.1 0.0 12345\n')).error).toContain('Fix tfmc temperature must be > 0');
    expect((await run(LJ2 + 'fix mc all tfmc 0.1 1.0 0\n')).error).toContain('Illegal fix tfmc random seed');
  });

  it('rejects an unknown keyword', async () => {
    expect((await run(LJ2 + 'fix mc all tfmc 0.1 1.0 12345 bogus\n')).error).toContain('Illegal fix tfmc command');
  });
});

describe('fix bond/create/angle aconstrain errors', () => {
  it('checks the angle range with native messages', async () => {
    const BOND = LJ2.replace('atom_style atomic', 'atom_style bond').replace('create_box 2 box', 'create_box 2 box bond/types 1 extra/bond/per/atom 2');
    const base = 'fix bc all bond/create/angle 1 1 1 2.0 1';
    expect((await run(BOND + `${base} aconstrain 120 90\n`)).error).toContain('Illegal fix bond/create/angle aconstrain command');
    expect((await run(BOND + `${base} aconstrain -5 90\n`)).error).toContain('Invalid fix bond/create/angle aconstrain amin value -5');
    expect((await run(BOND + `${base} aconstrain 10 190\n`)).error).toContain('Invalid fix bond/create/angle aconstrain amax value 190');
  });
});

describe('fix bond/swap argument errors and refusals', () => {
  it('needs exactly four arguments', async () => {
    expect((await run(LJ2 + 'fix bs all bond/swap 1 1.0 2.0\n')).error).toContain('Illegal fix bond/swap command');
  });

  it('needs molecule IDs', async () => {
    expect((await run(LJ2 + 'fix bs all bond/swap 1 1.0 2.0 4321\nrun 1\n')).error).toContain('Cannot use fix bond/swap with non-molecular systems');
  });
});
