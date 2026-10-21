import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/session';

/*
 * read_dump into boxes with fixed and shrink-wrapped boundaries (oracle cases w427readdump_shrink, w428readdump_fixed)
 * and the timestep reset of reset_timestep and read_dump with an active dump (src/output.cpp Output::reset_timestep).
 */

const run = async (text: string): Promise<Session> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  await s.execute(`${text}\n`);
  return s;
};

const BASE = `units lj
boundary p s f
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 51223 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 all nve
dump 1 all custom 5 rd.dump id type x y z
run 10
`;

describe('read_dump and the timestep reset', () => {
  it('stops while a single-file dump that has written is defined, as native does', async () => {
    await expect(run(`${BASE}reset_timestep 0`)).rejects.toThrow('Cannot reset timestep with active dump - must undump first');
    await expect(run(`${BASE}read_dump rd.dump 5 x y z`)).rejects.toThrow('Cannot reset timestep with active dump - must undump first');
    await expect(run(`${BASE}read_dump rd.dump 5 x y z timestep no`)).resolves.toBeDefined();
    await expect(run(`${BASE}undump 1\nread_dump rd.dump 5 x y z`)).resolves.toBeDefined();
  });

  it('re-fits shrink-wrapped faces to the atoms read with box yes', async () => {
    const s = await run(`${BASE}undump 1\nread_dump rd.dump 5 x y z box yes`);
    const st = s.sys.state;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < st.n; i++) { mn = Math.min(mn, st.x[3 * i + 1]); mx = Math.max(mx, st.x[3 * i + 1]); }
    expect(st.box.lo[1]).toBeLessThan(mn);
    expect(st.box.hi[1]).toBeGreaterThan(mx);
    expect(mn - st.box.lo[1]).toBeLessThan(1e-3);
  });
});
