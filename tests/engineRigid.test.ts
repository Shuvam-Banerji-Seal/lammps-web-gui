import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError } from '../src/engine/types';

/* fix rigid family (src/engine/fix/rigid.ts); dynamics vs native LAMMPS: oracle cases rigid_*. */

const run = async (fix: string, extra = '') => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  try {
    await s.execute(`units lj
atom_style molecular
lattice sc 1.5
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
set atom 1*14 mol 1
set atom 15*27 mol 2
${extra}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
${fix}
run 5`);
    return null;
  } catch (e) { return e as EngineError; }
};

describe('fix rigid', () => {
  it('integrates molecule, single and group bodies', async () => {
    expect(await run('fix 1 all rigid molecule')).toBeNull();
    expect(await run('fix 1 all rigid single')).toBeNull();
    expect(await run('fix 1 all rigid group 2 a b', 'group a id 1:10\ngroup b id 11:27')).toBeNull();
    expect(await run('fix 1 all rigid/small molecule')).toBeNull();
  });

  it('names what it rejects', async () => {
    expect((await run('fix 1 all rigid/small single'))?.message).toMatch(/single is only allowed for the rigid styles/);
    expect((await run('fix 1 all rigid molecule langevin 1 1 1 4567'))?.message).toMatch(/keyword 'langevin' is not supported/);
    expect((await run('fix 1 all rigid custom v_b'))?.message).toMatch(/bodystyle custom is not supported/);
    expect((await run('fix 1 all rigid/small molecule force * off off off'))?.message).toMatch(/only allowed for the rigid styles/);
    expect((await run('fix 1 all rigid molecule', 'set atom 27 mol 3'))?.message).toMatch(/two or more atoms/);
  });
});
