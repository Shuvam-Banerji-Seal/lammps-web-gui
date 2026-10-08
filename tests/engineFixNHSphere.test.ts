import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Argument and rule checks for fix nvt/sphere, npt/sphere, nph/sphere
 * (src/engine/fix/nh_sphere.ts; docs.lammps.org/fix_nvt_sphere.html,
 * docs.lammps.org/fix_npt_sphere.html, docs.lammps.org/fix_nph_sphere.html).
 * The docs say: "Use of the *disc* keyword is only allowed for 2d simulations,
 * as defined by the dimension keyword." and "This fix requires that atoms store
 * torque and angular velocity (omega) and a radius as defined by the atom_style
 * sphere command." Messages were checked against native LAMMPS (black box):
 * disc in 3d, update dipole without a mu attribute, point particles in the
 * group, and an unknown keyword.
 */

const head = (dim: 2 | 3) => `
units           lj
dimension       ${dim}
atom_style      sphere
lattice         ${dim === 3 ? 'fcc 0.8442' : 'sq 0.8'}
region          box block 0 2 0 2 ${dim === 3 ? '0 2' : '-0.5 0.5'}
create_box      1 box
create_atoms    1 box
set             group all diameter 1.0
set             group all density 1.0
pair_style      lj/cut 2.5
pair_coeff      * * 1.0 1.0
`;

const run = async (text: string): Promise<string | null> => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  try {
    await session.execute(text);
    return null;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
};

describe('fix nvt/npt/nph sphere argument rules', () => {
  it('rejects the disc keyword in 3d, naming it', async () => {
    const err = await run(`${head(3)}\nfix 1 all nvt/sphere temp 1 1 0.5 disc\nrun 1\n`);
    expect(err).toMatch(/disc/);
    expect(err).toMatch(/2d/);
  });

  it('accepts disc in 2d', async () => {
    const err = await run(`${head(2)}\nfix 1 all nvt/sphere temp 1 1 0.5 disc\nrun 1\n`);
    expect(err).toBeNull();
  });

  it('rejects update dipole and dipole/dlm (no dipole in atom_style sphere), naming them', async () => {
    const e1 = await run(`${head(3)}\nfix 1 all nvt/sphere temp 1 1 0.5 update dipole\nrun 1\n`);
    expect(e1).toMatch(/update dipole/);
    expect(e1).toMatch(/mu/);
    const e2 = await run(`${head(3)}\nfix 1 all npt/sphere temp 1 1 0.5 iso 1 1 1 update dipole/dlm\nrun 1\n`);
    expect(e2).toMatch(/dipole\/dlm/);
  });

  it('rejects unknown keywords by name', async () => {
    const err = await run(`${head(3)}\nfix 1 all nvt/sphere temp 1 1 0.5 bogus\nrun 1\n`);
    expect(err).toMatch(/bogus/);
  });

  it('nvt/sphere needs temp and no pressure keyword; npt/sphere needs both; nph/sphere needs pressure only', async () => {
    expect(await run(`${head(3)}\nfix 1 all nvt/sphere iso 1 1 1\nrun 1\n`)).not.toBeNull();
    expect(await run(`${head(3)}\nfix 1 all npt/sphere temp 1 1 0.5\nrun 1\n`)).not.toBeNull();
    expect(await run(`${head(3)}\nfix 1 all nph/sphere temp 1 1 0.5 iso 1 1 1\nrun 1\n`)).not.toBeNull();
    expect(await run(`${head(3)}\nfix 1 all nph/sphere iso 1 1 1 ptemp 1.0\nrun 1\n`)).toBeNull();
  });

  it('rejects point particles in the group', async () => {
    const text = `${head(3)}\nset group all diameter 0.0\nfix 1 all nvt/sphere temp 1 1 0.5\nrun 1\n`;
    expect(await run(text)).toMatch(/extended particles/);
  });

  it('a 2d disc fix sets up the temperature compute with the group rule of the variant', async () => {
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    await session.execute(`${head(2)}\nfix 1 all nvt/sphere temp 1 1 0.5 disc\n`);
    const c = session.sys.compute('1_temp');
    expect(c.style).toBe('temp/sphere');
  });
});
