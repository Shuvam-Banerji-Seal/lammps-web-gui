import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Which dump_modify keywords each dump style takes, as native LAMMPS splits them between Dump::modify_params (the
 * base list, src/dump.cpp at b332f28) and the styles' modify_param (DumpAtom, DumpCustom and DumpYAML, DumpXYZ and
 * DumpExtXYZ, DumpLocal), and the init checks that depend on them (extxyz needs element names or type labels, dump
 * local cannot sort by ID, the sort column index).
 *
 * Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28): each row is one input (the 27-atom bonded cube
 * below, one dump of the style, one dump_modify, run 0) and native's outcome: null when it runs, else the start of
 * its error message. The rows are the pairs where the engine differed before this change.
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
variable vx atom vx
variable ev equal 2
`;

const DUMP: Record<string, string> = {
  atom: 'atom 1 d.dump', custom: 'custom 1 d.dump id x vx', xyz: 'xyz 1 d.xyz', cfg: 'cfg 1 d.*.cfg mass type xs ys zs',
  local: 'local 1 d.dump c_bl', yaml: 'yaml 1 d.yaml id x', extxyz: 'extxyz 1 d.xyz',
};

const ROWS: [string, string, string | null][] = [
  ['atom', 'compression_level 3', 'Unknown dump_modify keyword: compression_level'],
  ['atom', 'element C', 'Unknown dump_modify keyword: element'],
  ['atom', 'format 1 %d', 'Unknown dump_modify format keyword: 1'],
  ['atom', 'format float %g', 'Unknown dump_modify format keyword: float'],
  ['atom', 'label foo', 'Unknown dump_modify keyword: label'],
  ['atom', 'refresh c_bl', 'Unknown dump_modify keyword: refresh'],
  ['atom', 'skip v_ev', null],
  ['atom', 'thermo yes', 'Unknown dump_modify keyword: thermo'],
  ['atom', 'types labels', 'Unknown dump_modify keyword: types'],
  ['atom', 'unwrap yes', 'Unknown dump_modify keyword: unwrap'],
  ['cfg', 'compression_level 3', 'Unknown dump_modify keyword: compression_level'],
  ['cfg', 'image yes', 'Unknown dump_modify keyword: image'],
  ['cfg', 'label foo', 'Unknown dump_modify keyword: label'],
  ['cfg', 'refresh c_bl', null],
  ['cfg', 'scale no', 'Unknown dump_modify keyword: scale'],
  ['cfg', 'skip v_ev', null],
  ['cfg', 'thermo yes', 'Unknown dump_modify keyword: thermo'],
  ['cfg', 'types labels', 'Unknown dump_modify keyword: types'],
  ['cfg', 'unwrap yes', 'Unknown dump_modify keyword: unwrap'],
  ['custom', 'compression_level 3', 'Unknown dump_modify keyword: compression_level'],
  ['custom', 'image yes', 'Unknown dump_modify keyword: image'],
  ['custom', 'label foo', 'Unknown dump_modify keyword: label'],
  ['custom', 'refresh c_bl', null],
  ['custom', 'scale no', 'Unknown dump_modify keyword: scale'],
  ['custom', 'skip v_ev', null],
  ['custom', 'thermo yes', 'Unknown dump_modify keyword: thermo'],
  ['custom', 'types labels', 'Unknown dump_modify keyword: types'],
  ['custom', 'unwrap yes', 'Unknown dump_modify keyword: unwrap'],
  ['extxyz', 'append yes', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'balance no', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'buffer no', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'colname 1 aa', 'Incorrect dump_modify arguments: colname 1 aa'],
  ['extxyz', 'compression_level 3', 'Unknown dump_modify keyword: compression_level'],
  ['extxyz', 'delay 1', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'every 2', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'every v_ev', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'first yes', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'flush yes', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'format 1 %d', 'Unknown dump_modify format keyword: 1'],
  ['extxyz', 'format float %g', 'Unknown dump_modify format keyword: float'],
  ['extxyz', 'header no', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'image yes', 'Unknown dump_modify keyword: image'],
  ['extxyz', 'label foo', 'Unknown dump_modify keyword: label'],
  ['extxyz', 'pad 3', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'pbc yes', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'refresh c_bl', 'Unknown dump_modify keyword: refresh'],
  ['extxyz', 'scale no', 'Unknown dump_modify keyword: scale'],
  ['extxyz', 'skip v_ev', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'sort 2', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'sort id', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'thermo yes', 'Unknown dump_modify keyword: thermo'],
  ['extxyz', 'time yes', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'triclinic/general no', 'Unknown dump_modify keyword: triclinic/general'],
  ['extxyz', 'types labels', 'Label map must be defined when using \'types labels\''],
  ['extxyz', 'units yes', 'Must use either type lables or dump_modify element with dump style extxyz'],
  ['extxyz', 'unwrap yes', 'Unknown dump_modify keyword: unwrap'],
  ['local', 'compression_level 3', 'Unknown dump_modify keyword: compression_level'],
  ['local', 'element C', 'Unknown dump_modify keyword: element'],
  ['local', 'image yes', 'Unknown dump_modify keyword: image'],
  ['local', 'refresh c_bl', 'Unknown dump_modify keyword: refresh'],
  ['local', 'scale no', 'Unknown dump_modify keyword: scale'],
  ['local', 'skip v_ev', null],
  ['local', 'sort 2', 'Dump sort column index 2 is invalid'],
  ['local', 'sort id', 'Dump local cannot sort by atom ID'],
  ['local', 'thermo yes', 'Unknown dump_modify keyword: thermo'],
  ['local', 'triclinic/general no', 'Unknown dump_modify keyword: triclinic/general'],
  ['local', 'types labels', 'Unknown dump_modify keyword: types'],
  ['local', 'unwrap yes', 'Unknown dump_modify keyword: unwrap'],
  ['xyz', 'colname 1 aa', 'Incorrect dump_modify arguments: colname 1 aa'],
  ['xyz', 'compression_level 3', 'Unknown dump_modify keyword: compression_level'],
  ['xyz', 'format 1 %d', 'Unknown dump_modify format keyword: 1'],
  ['xyz', 'format float %g', 'Unknown dump_modify format keyword: float'],
  ['xyz', 'image yes', 'Unknown dump_modify keyword: image'],
  ['xyz', 'label foo', 'Unknown dump_modify keyword: label'],
  ['xyz', 'refresh c_bl', 'Unknown dump_modify keyword: refresh'],
  ['xyz', 'scale no', 'Unknown dump_modify keyword: scale'],
  ['xyz', 'skip v_ev', null],
  ['xyz', 'thermo yes', 'Unknown dump_modify keyword: thermo'],
  ['xyz', 'triclinic/general no', 'Unknown dump_modify keyword: triclinic/general'],
  ['xyz', 'types labels', 'Label map must be defined when using \'types labels\''],
  ['xyz', 'unwrap yes', 'Unknown dump_modify keyword: unwrap'],
  ['yaml', 'compression_level 3', 'Unknown dump_modify keyword: compression_level'],
  ['yaml', 'image yes', 'Unknown dump_modify keyword: image'],
  ['yaml', 'label foo', 'Unknown dump_modify keyword: label'],
  ['yaml', 'refresh c_bl', null],
  ['yaml', 'scale no', 'Unknown dump_modify keyword: scale'],
  ['yaml', 'skip v_ev', null],
  ['yaml', 'types labels', 'Unknown dump_modify keyword: types'],
  ['yaml', 'unwrap yes', 'Unknown dump_modify keyword: unwrap'],
];

const outcome = async (style: string, kw: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  try {
    await s.execute(`${BASE}dump d all ${DUMP[style]}
dump_modify d ${kw}
run 0
`);
    return null;
  } catch (e) {
    return (e as Error).message;
  }
};

describe('dump_modify keywords by dump style, as native LAMMPS', () => {
  it.each(ROWS)('%s | %s', async (style, kw, expected) => {
    const got = await outcome(style, kw);
    if (expected === null) expect(got).toBeNull();
    else expect(got).toContain(expected);
  });

  it('dump_modify skip v_name skips the snapshots where the variable is not 0', async () => {
    // Measured with native LAMMPS: the same input writes the snapshots of steps 0, 2 and 4
    const files = new Map<string, string>();
    const s = new Session({ emit: () => {}, writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
    await s.execute(`${BASE}fix 1 all nve\nvariable sk equal "step % 2"\ndump d all custom 1 d.dump id\ndump_modify d skip v_sk\nrun 4\n`);
    const steps = [...(files.get('d.dump') ?? '').matchAll(/ITEM: TIMESTEP\n(\d+)/g)].map((m) => Number(m[1]));
    expect(steps).toEqual([0, 2, 4]);
  });
});
