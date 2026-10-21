import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * dump_modify region and thresh belong to native's custom dump family (DumpCustom and its subclasses: custom, cfg,
 * image, yaml and their compressed forms; src/dump_custom.cpp at b332f28, modify_param). The other styles hand them to
 * Dump::modify_params, which does not know them. A threshold names one of the per-atom attributes of the thresh list or
 * a compute, fix, variable or custom property reference.
 */

const BASE = `units lj
atom_style bond
lattice sc 1.0
region box block 0 3 0 3 0 3
create_box 1 box bond/types 1 extra/bond/per/atom 6
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 1.5
pair_coeff 1 1 1.0 1.0
bond_style harmonic
bond_coeff 1 1.0 1.0
create_bonds many all all 1 0.9 1.1
compute bl all bond/local dist
region r block 0 1 0 1 0 1
`;

const run = async (dump: string, modify: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  try {
    await s.execute(`${BASE}${dump}\ndump_modify d ${modify}\nrun 0\n`);
    return null;
  } catch (e) {
    return (e as Error).message;
  }
};

describe('dump_modify region and thresh by dump style, as native LAMMPS', () => {
  // Measured with native LAMMPS: ERROR: Unknown dump_modify keyword: thresh (region) (src/dump.cpp:1364) for atom, xyz,
  // extxyz, local, atom/gz and xyz/gz; custom, cfg, image and custom/gz accept both
  it.each([
    ['dump d all atom 1 d.dump'], ['dump d all xyz 1 d.xyz'], ['dump d all extxyz 1 d.xyz'],
    ['dump d all local 1 d.dump c_bl'], ['dump d all atom/gz 1 d.dump.gz'], ['dump d all xyz/gz 1 d.xyz.gz'],
  ])('%s refuses them', async (dump) => {
    expect(await run(dump, 'thresh id > 5')).toContain('Unknown dump_modify keyword: thresh');
    expect(await run(dump, 'region r')).toContain('Unknown dump_modify keyword: region');
  });

  it.each([
    ['dump d all custom 1 d.dump id x'], ['dump d all cfg 1 d.*.cfg mass type xs ys zs'],
    ['dump d all image 1 img.*.ppm type type'], ['dump d all custom/gz 1 d.dump.gz id x'],
  ])('%s takes them', async (dump) => {
    expect(await run(dump, 'thresh id > 5')).toBeNull();
    expect(await run(dump, 'region r')).toBeNull();
  });

  // Measured with native LAMMPS (src/dump_custom.cpp:2265, 2029, 2251 and 1920):
  // ERROR: Invalid dump_modify thresh operator
  // ERROR: Illegal dump_modify thresh command: missing argument(s)
  // ERROR: Invalid dump_modify thresh attribute: element
  // ERROR: Dump_modify region nope does not exist
  it.each([
    ['thresh id ~ 5', 'Invalid dump_modify thresh operator'],
    ['thresh id >', 'Illegal dump_modify thresh command: missing argument(s)'],
    ['thresh element == 1', 'Invalid dump_modify thresh attribute: element'],
    ['region nope', 'Dump_modify region nope does not exist'],
  ])('dump custom: %s stops as native does', async (modify, text) => {
    expect(await run('dump d all custom 1 d.dump id x', modify)).toContain(text);
  });
});
