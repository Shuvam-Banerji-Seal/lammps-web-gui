import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// fix reaxff/species allows a dynamic group only with Nrepeat 1 (src/REAXFF/fix_reaxff_species.cpp at b332f28:
// "if (nrepeat == 1) dynamic_group_allow = 1;"); the check is the run's (src/modify.cpp Modify::init)
const input = (nrepeat: number) => `units real
atom_style charge
read_data w55reax_rdx.data
pair_style reaxff w55reax_rdx.control checkqeq no
pair_coeff * * ffield.reax.rdx H C O N
region half block INF INF INF INF INF 12.5
group dyn dynamic all region half
fix 1 all nve
fix 3 dyn reaxff/species 1 ${nrepeat} 5 species.dyn element H C O N
run 0
`;

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  session.addFile('w55reax_rdx.data', readFileSync('tests/oracle/w55reax_rdx.data', 'utf8'));
  session.addFile('w55reax_rdx.control', readFileSync('tests/oracle/w55reax_rdx.control', 'utf8'));
  session.addFile('ffield.reax.rdx', readFileSync('third_party/lammps/potentials/ffield.reax.rdx', 'utf8'));
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error');
  return err?.message ?? error;
};

describe('fix reaxff/species on a dynamic group', () => {
  it('runs with Nrepeat 1', async () => {
    expect(await run(input(1))).toBeNull();
  });
  it('stops with Nrepeat above 1 (native text)', async () => {
    expect(await run(input(5))).toContain('Fix reaxff/species does not allow use with a dynamic group');
  });
});
