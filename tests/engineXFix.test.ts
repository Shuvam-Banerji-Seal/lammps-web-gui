import { beforeAll, describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { loadAllStyles } from '../src/engine/styles';

/* Unit tests of the xfix ports (src/engine-gpl/xfix): native error texts for bad arguments, and the refusals. */

const HEAD = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 87287 loop all
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
timestep 0.002
`;

const runText = async (text: string): Promise<{ error: string | null; thermo: number }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  let error: string | null = null;
  try {
    await session.execute(text + '\n');
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (!error && err && 'message' in err) error = err.message;
  return { error, thermo: events.filter((e) => e.kind === 'thermo').length };
};

describe('xfix ports: argument errors and refusals', () => {
  beforeAll(async () => { await loadAllStyles(); });

  it('fix nvk refuses groups other than all', async () => {
    const r = await runText(`${HEAD}group g id 1:4\nfix 1 g nvk\nrun 1`);
    expect(r.error).toContain('Fix nvk only supports group all');
  });

  it('fix ffl rejects a non-positive tau and a bad flip type', async () => {
    expect((await runText(`${HEAD}fix 1 all ffl 0.0 1.0 1.0 5\nrun 1`)).error).toContain('Illegal fix ffl tau value, should be greater than 0');
    expect((await runText(`${HEAD}fix 1 all ffl 0.5 1.0 1.0 5 bogus\nrun 1`)).error).toContain('Illegal fix ffl flip type');
  });

  it('fix gld needs the pprony series and enough coefficients', async () => {
    expect((await runText(`${HEAD}fix 1 all gld 1.0 1.0 1 5 kpp 0.5 0.5\nrun 1`)).error).toContain('Fix gld series type must be pprony for now');
    expect((await runText(`${HEAD}fix 1 all gld 1.0 1.0 2 5 pprony 0.5 0.5\nrun 1`)).error).toContain('Fix gld needs more prony series coefficients');
  });

  it('fix smd rejects an unknown pulling style', async () => {
    expect((await runText(`${HEAD}fix 1 all smd pull 1.0 tether NULL NULL 0 1\nrun 1`)).error).toContain('Illegal fix smd command');
  });

  it('fix set rejects a keyword that does not use a per-atom variable', async () => {
    const r = await runText(`${HEAD}fix 1 all set 5 0 atom 1*8 vx 0.5\nrun 1`);
    expect(r.error).toContain('Fix set command keyword vx does not invoke a per-atom variable');
  });

  it('fix ave/correlate/long requires nlen divisible by ncount', async () => {
    const r = await runText(`${HEAD}compute p all pressure thermo_temp\nfix 1 all ave/correlate/long 1 10 c_p[4] nlen 5 ncount 2\nrun 1`);
    expect(r.error).toContain('Fix ave/correlate/long: nlen must be divisible by ncount');
  });

  it('fix nonaffine/displacement refuses the d2min style by name', async () => {
    const r = await runText(`${HEAD}fix 1 all nonaffine/displacement 5 d2min type fixed 0\nrun 1`);
    expect(r.error).toContain('d2min');
  });

  it('fix nonaffine/displacement rejects a non-positive nevery', async () => {
    const r = await runText(`${HEAD}fix 1 all nonaffine/displacement 0 integrated fixed 0\nrun 1`);
    expect(r.error).toContain('Illegal nevery value 0 in fix nonaffine/displacement');
  });

  it('a valid nvk run produces thermo output', async () => {
    const r = await runText(`${HEAD}fix 1 all nvk\nthermo 5\nrun 10`);
    expect(r.error).toBeNull();
    expect(r.thermo).toBeGreaterThan(0);
  });
});
