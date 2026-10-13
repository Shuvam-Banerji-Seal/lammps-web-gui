import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix shardlow (src/engine-gpl/dpdreact/fix_shardlow.ts) checks of fix_shardlow.cpp: the integrator order and the
 * constant temperature refusal. The messages are the native ones (docs.lammps.org/fix_shardlow.html requires a
 * deterministic integrator after it; the DPD-REACT constant temperature check is in fix_shardlow.cpp setup()).
 */
const DATA = join(__dirname, 'oracle', 'w95dreact_shardlow_dpd_1000.data');

const runInput = async (fixes: string): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, undefined, 0, 1);
  session.addFile('w95dreact_shardlow_dpd_1000.data', readFileSync(DATA, 'utf8'));
  const text = `boundary p p p
units metal
atom_style atomic
read_data w95dreact_shardlow_dpd_1000.data
comm_modify mode single vel yes
mass 1 100.0
pair_style dpd/fdt 300.0 10.0 234324
pair_coeff 1 1 0.075 0.022 10.0
neighbor 2.0 bin
timestep 0.001
${fixes}
run 1
`;
  try {
    await session.execute(text);
  } catch (e) {
    return String(e);
  }
  const err = events.find((ev) => ev.kind === 'error');
  return err && 'message' in err ? err.message : null;
};

describe('fix shardlow', () => {
  it('needs a deterministic integrator after it', async () => {
    const msg = await runInput('fix 1 all nve\nfix 2 all shardlow');
    expect(msg).toContain('The deterministic integrator must follow fix shardlow in the input file.');
  });

  it('needs a deterministic integrator at all', async () => {
    const msg = await runInput('fix 1 all shardlow');
    expect(msg).toContain('A deterministic integrator (e.g. fix nve or fix nph) is required when using fix shardlow.');
  });

  it('refuses constant temperature fixes', async () => {
    const msg = await runInput('fix 1 all shardlow\nfix 2 all nvt temp 300 300 0.1');
    expect(msg).toContain('Cannot use constant temperature integration routines with DPD-REACT.');
  });

  it('runs after the deterministic integrator order is right', async () => {
    expect(await runInput('fix 1 all shardlow\nfix 2 all nve')).toBeNull();
  });
});
