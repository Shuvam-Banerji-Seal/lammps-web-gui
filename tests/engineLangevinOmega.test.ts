import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix langevin omega (fix_langevin.html: "The keyword angmom and omega keywords enable thermostatting of rotational
 * degrees of freedom ... This can only be done for finite-size particles."). The numbers are checked against native
 * LAMMPS by the w84lang_omega_sphere and w330lang_omega_tally_zero oracle cases.
 */

const run = async (text: string): Promise<string[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  try {
    await session.execute(text);
  } catch (e) {
    return [String(e)];
  }
  return events.flatMap((e) => (e.kind === 'error' ? [e.message] : []));
};

const sphereSetup = `units lj
atom_style sphere
atom_modify map array sort 0 0.0
lattice sc 0.6
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
set group all diameter 1.0 density 1.0
pair_style lj/cut 2.5
pair_coeff * * 0.2 0.8
fix 1 all nve/sphere
`;

describe('fix langevin omega and angmom', () => {
  it('runs omega yes on spheres', async () => {
    expect(await run(`${sphereSetup}fix 2 all langevin 1.0 1.0 1.0 9182 omega yes\nrun 2\n`)).toEqual([]);
  });

  it('refuses omega yes on point particles', async () => {
    const errs = await run(`units lj\natom_style atomic\nlattice sc 0.6\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut 2.5\npair_coeff * * 0.2 0.8\nfix 2 all langevin 1.0 1.0 1.0 9182 omega yes\nrun 0\n`);
    expect(errs.join('\n')).toMatch(/Fix langevin omega requires atom attribute omega/);
  });

  it('runs omega yes with tally yes and zero yes (the w330lang_omega_tally_zero oracle case checks the numbers)', async () => {
    expect(await run(`${sphereSetup}fix 2 all langevin 1.0 1.0 1.0 9182 omega yes tally yes zero yes\nthermo_style custom step f_2 ecouple\nrun 2\n`)).toEqual([]);
  });

  it('refuses omega yes on a sphere of radius 0, as native init does', async () => {
    const errs = await run(`${sphereSetup}set atom 3 diameter 0.0\nfix 2 all langevin 1.0 1.0 1.0 9182 omega yes\nrun 0\n`);
    expect(errs.join('\n')).toMatch(/Fix langevin omega requires extended particles/);
  });

  it('takes an atom-style temperature variable and refuses a vector-style one', async () => {
    expect(await run(`${sphereSetup}variable tt atom 0.5+0.1*x\nfix 2 all langevin v_tt 1.0 1.0 9182\nrun 2\n`)).toEqual([]);
    const errs = await run(`${sphereSetup}variable tt vector [1,2]\nfix 2 all langevin v_tt 1.0 1.0 9182\nrun 0\n`);
    expect(errs.join('\n')).toMatch(/Variable tt for fix langevin is invalid style/);
  });

  it('has a scalar of 0 without tally, as native does', async () => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
    await session.execute(`${sphereSetup}fix 2 all langevin 1.0 1.0 1.0 9182\nvariable e equal f_2\nrun 2\nprint "E=\${e}"\n`);
    expect(events.flatMap((e) => (e.kind === 'error' ? [e.message] : []))).toEqual([]);
    expect(JSON.stringify(events)).toContain('E=0');
  });

  it('refuses angmom on particles that are not ellipsoids (fix_langevin.cpp)', async () => {
    const errs = await run(`${sphereSetup}fix 2 all langevin 1.0 1.0 1.0 9182 angmom 10\nrun 0\n`);
    expect(errs.join('\n')).toMatch(/angmom requires atom style ellipsoid/);
  });
});
