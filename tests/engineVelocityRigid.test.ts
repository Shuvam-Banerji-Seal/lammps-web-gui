import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/session';
import type { EngineEvent } from '../src/engine/types';

/*
 * velocity ... rigid fix-ID (src/velocity.cpp, FixRigid zero_momentum / zero_rotation): native's errors and the
 * warning for changing velocities of atoms in rigid bodies. Native parity of the zeroing is the oracle case
 * w432velocity_rigid.
 */

const run = async (text: string): Promise<{ s: Session; log: string }> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  await s.execute(`${text}\n`);
  const log = events.map((e) => ('text' in e ? String((e as { text: unknown }).text) : '')).join('\n');
  return { s, log };
};

const BASE = `units lj
lattice sc 0.6
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
region r1 block 0 1.9 0 1.9 0 1.9
group a1 region r1
velocity all create 1.0 4211 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 a1 rigid single
fix 2 all nve
`;

describe('velocity rigid', () => {
  it('names a missing fix and a fix that is not a rigid-body fix', async () => {
    await expect(run(`${BASE}velocity a1 zero linear rigid nope`)).rejects.toThrow('Fix ID nope for velocity does not exist');
    await expect(run(`${BASE}velocity a1 zero linear rigid 2`)).rejects.toThrow('Velocity rigid used with non-rigid fix-ID 2');
  });

  it('zeroes the body momentum before the first run', async () => {
    const { s } = await run(`${BASE}velocity a1 zero linear rigid 1`);
    const st = s.sys.state;
    const p = [0, 0, 0];
    for (let i = 0; i < st.n; i++) if (st.mask[i] & s.sys.groupBit('a1')) for (let d = 0; d < 3; d++) p[d] += st.v[3 * i + d];
    for (let d = 0; d < 3; d++) expect(p[d]).toBeCloseTo(0, 12);
  });

  it('warns when velocities of atoms of set-up bodies change without the rigid keyword', async () => {
    const { log } = await run(`${BASE}run 0\nvelocity a1 scale 2.0`);
    expect(log).toContain('Changing velocities of atoms in rigid bodies. This has no effect unless rigid bodies are rebuilt');
    const quiet = await run(`${BASE}velocity a1 scale 2.0`);
    expect(quiet.log).not.toContain('Changing velocities of atoms in rigid bodies');
  });
});
