import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import type { SpecialList } from '../src/engine/atoms';

/** The special list the neighbor list builds its pair exclusions from (white box: no setup between the changes). */
const nbSpecial = (sys: { nb: unknown }): SpecialList | null => (sys.nb as { needs: { special: SpecialList | null } | null }).needs?.special ?? null;

/*
 * The special (1-2, 1-3, 1-4) list that the force field builds from the bonds must reach the neighbor list at
 * every topology change, so the pair exclusions follow the bonds between runs and during a run (bpm bond breaking).
 * Two atoms at r = 1.2 with lj/cut 2.5 (sigma = epsilon = 1): the pair energy is 0 while the bond excludes the pair
 * (special_bonds lj 0 0 0, the default) and the pair energy below once it does not.
 */

const RB = 1.2;
// Measured with native LAMMPS (black box, this two-atom input with run 0): E_pair -0.445482643792 without the bond
const LJ = -0.445482643792;

const epairRows = async (cmds: string): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  await session.execute(`units lj
atom_style bond
region box block 0 10 0 10 0 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0 0 0
create_atoms 1 single ${RB} 0 0
mass 1 1
pair_style lj/cut 2.5
pair_coeff 1 1 1 1
bond_style harmonic
bond_coeff 1 1 ${RB}
create_bonds single/bond 1 1 2
thermo_style custom step epair
thermo_modify format float %.12g
${cmds}`);
  return events.flatMap((e) => (e.kind === 'thermo' ? [e.row] : []));
};

const expectEpair = (rows: ThermoRow[], want: number[]): void => {
  expect(rows.length).toBe(want.length);
  rows.forEach((r, k) => expect(r.epair).toBeCloseTo(want[k], 9));
};

describe('special list sync with the neighbor list', () => {
  it('topology commands update the neighbor list special copy without a run (no setup in between)', async () => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
    await session.execute(`units lj
atom_style bond
region box block 0 10 0 10 0 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0 0 0
create_atoms 1 single ${RB} 0 0
mass 1 1
pair_style lj/cut 2.5
pair_coeff 1 1 1 1
bond_style harmonic
bond_coeff 1 1 ${RB}
run 0
`);
    const partners = (sp: SpecialList | null, i: number): number[] => (sp ? Array.from(sp.partner.slice(sp.offset[i], sp.offset[i + 1])) : []);
    const sys = session.sys;
    expect(partners(nbSpecial(sys), 0)).toEqual([]);
    await session.execute('create_bonds single/bond 1 1 2\n');
    expect(partners(nbSpecial(sys), 0)).toEqual([2]);
    expect(partners(nbSpecial(sys), 1)).toEqual([1]);
    await session.execute('delete_bonds all bond 1 remove special\n');
    expect(partners(nbSpecial(sys), 0)).toEqual([]);
    await session.execute('create_bonds single/bond 1 1 2\n');
    expect(partners(nbSpecial(sys), 0)).toEqual([2]);
  });

  it('the bond excludes the pair until it is deleted, and a new bond excludes it again (between runs)', async () => {
    const rows = await epairRows(`run 0
delete_bonds all bond 1 remove special
run 0
create_bonds single/bond 1 1 2
run 0
`);
    expectEpair(rows, [0, LJ, 0]);
  });

  it('a bond created between runs is excluded from the next run, not only after a full setup', async () => {
    const rows = await epairRows(`delete_bonds all bond 1 remove special
run 0
run 0
create_bonds single/bond 1 1 2
run 0
`);
    // the first run sees no bond (pair included); the bond created afterwards is excluded in the next run
    expectEpair(rows, [LJ, LJ, 0]);
  });

  it('a bpm bond that breaks during the run lets the pair interact from the next neighbor build on', async () => {
    const rows = await epairRows(`bond_style bpm/spring
bond_coeff 1 1.0 0.1 0
group a id 1
group b id 2
velocity a set -1.0 0 0 units box
velocity b set 1.0 0 0 units box
fix 1 all nve
run 0
run 120
`);
    const last = rows[rows.length - 1].epair;
    // the bond stretched past strain 0.1 and broke, so the final pair energy is nonzero (the stale list would give 0)
    expect(last).not.toBe(0);
    expect(last).toBeLessThan(0);
  });
});
