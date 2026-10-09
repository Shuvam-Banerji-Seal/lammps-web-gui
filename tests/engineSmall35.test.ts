import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Small wave-35 fixes: the removed box command (commands/misc.ts), fix box/relax iso/aniso/tri in 2d
 * (fix/box_relax.ts; parity in tests/oracle/w35boxrelax_2d.in) and write_restart after the erfc
 * tables of the coul/long family were built (output/restart.ts; parity in w35restart_buck_table.in).
 */

const newSession = () => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  return { session, events };
};

const textOf = (events: EngineEvent[]): string[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);

const LJ2D = `
units           lj
atom_style      atomic
dimension       2
lattice         hex 0.8
region          box block 0 4 0 3 -0.5 0.5
create_box      1 box
create_atoms    1 box
mass            1 1.0
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
min_modify      line backtrack
`;

describe('box (removed command)', () => {
  it('prints the native warning and carries on', async () => {
    const { session, events } = newSession();
    await session.execute('units lj\nbox tilt large\nregion b block 0 1 0 1 0 1\ncreate_box 1 b\n');
    expect(textOf(events)).toContain("WARNING: The 'box' command has been removed and will be ignored");
    expect(session.sys.state.box.hi[0]).toBe(1);
  });
});

describe('fix box/relax in 2d', () => {
  for (const kw of ['iso 1.0', 'aniso 1.0', 'x 1.0 y 1.0 couple xy']) {
    it(`accepts ${kw} (z is ignored)`, async () => {
      const { session } = newSession();
      await session.execute(`${LJ2D}fix 3 all box/relax ${kw} vmax 1.0e-3\nminimize 0.0 1.0e-6 3 10\n`);
      expect(session.sys.state.step).toBeGreaterThan(0);
    });
  }
  it('still refuses an explicit z target', async () => {
    const { session } = newSession();
    await expect(session.execute(`${LJ2D}fix 3 all box/relax z 1.0\nminimize 0.0 1.0e-6 3 10\n`)).rejects.toThrow(/z is not available for 2d/);
  });
});

describe('write_restart with the erfc tables built', () => {
  it('stores buck/coul/long without its erfc table cache', async () => {
    const files = new Map<string, string>();
    const session = new Session({ emit: () => {}, writeFile: (n, t, append) => files.set(n, (append ? files.get(n) ?? '' : '') + t) });
    await session.execute(`units real
atom_style charge
lattice fcc 5.4
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 39.948
set group all charge 0.0
pair_style buck/coul/long 8.0
pair_coeff * * 50000.0 0.27 900.0
kspace_style ewald 1.0e-5
run 0
write_restart b.restart
`);
    const text = files.get('b.restart') ?? '';
    expect(text).toContain('"name":"buck/coul/long"');
    expect(text).not.toContain('erfcTables');
  });
});
