import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix lb/fluid (src/engine-gpl/latboltz): the keywords and lattices not ported to the browser engine are refused by
 * name (StyleError), never ignored. The oracle cases tests/oracle/w132lb_* check the ported behaviour against native.
 */
const BASE = `units lj
dimension 3
boundary p p p
atom_style atomic
region box block 0 12 0 12 0 12
create_box 1 box
create_atoms 1 single 6 6 6
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
timestep 0.001
`;

const run = async (fixLine: string): Promise<string> => {
  const session = new Session({ emit: () => {}, writeFile: () => {} } as never, undefined, 0, 1);
  try {
    await session.execute(`${BASE}${fixLine}\nrun 1\n`);
  } catch (e) {
    return String(e instanceof Error ? e.message : e);
  }
  return '';
};

describe('fix lb/fluid refusals and checks', () => {
  it('refuses the D3Q19 lattice by name', async () => {
    expect(await run('fix FL all lb/fluid 1 1.0 1.0 dx 2.0 D3Q19')).toMatch(/D3Q19/);
  });

  it('refuses pit geometry and the Keys stencil by name', async () => {
    expect(await run('fix FL all lb/fluid 1 1.0 1.0 dx 2.0 npits 1 4 4 4 0')).toMatch(/npits/);
    expect(await run('fix FL all lb/fluid 1 1.0 1.0 dx 2.0 stencil 4')).toMatch(/stencil 4/);
  });

  it('refuses the file output and restart keywords by name', async () => {
    expect(await run('fix FL all lb/fluid 1 1.0 1.0 dx 2.0 dumpxdmf 10 fflow 0')).toMatch(/dumpxdmf/);
    expect(await run('fix FL all lb/fluid 1 1.0 1.0 dx 2.0 write_restart 10')).toMatch(/write_restart/);
  });

  it('rejects an unknown keyword and a z wall velocity with periodic z', async () => {
    expect(await run('fix FL all lb/fluid 1 1.0 1.0 dx 2.0 bogus')).toMatch(/Illegal fix lb\/fluid/);
    expect(await run('fix FL all lb/fluid 1 1.0 1.0 dx 2.0 zwall_velocity 0 1')).toMatch(/z wall velocity/);
  });

  it('requires fix lb/fluid for fix lb/viscous and fix lb/momentum', async () => {
    expect(await run('fix V all lb/viscous')).toMatch(/lb\/fluid fix must also be used/);
    expect(await run('fix M all lb/momentum 10 linear 1 1 1')).toMatch(/requires a fix lb\/fluid/);
  });
});
