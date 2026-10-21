import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * thermo_modify keywords and their stops as native LAMMPS has them (src/thermo.cpp at b332f28, modify_params; the
 * warning limit of src/error.cpp). Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28): each row is the
 * input below with one thermo_modify line and run 2; null when it runs, else the start of native's error message.
 */

const BASE = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
compute t2 all temp
compute p2 all pressure t2
thermo_style custom step temp pe press
`;

const ROWS: [string, string | null][] = [
  ['lost ignore', null],
  ['lost warn', null],
  ['lost error', null],
  ['lost bogus', 'Unknown thermo_modify lost argument: bogus'],
  ['lost/bond ignore', null],
  ['lost/bond warn', null],
  ['lost/bond error', null],
  ['warn 10', null],
  ['warn ignore', null],
  ['warn reset', null],
  ['warn default', null],
  ['warn -1', null],
  ['norm yes', null],
  ['norm no', null],
  ['norm maybe', 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  ['flush yes', null],
  ['line one', null],
  ['line multi', null],
  ['line yaml', null],
  ['line multiline', 'Unknown thermo_modify line argument: multiline'],
  ['line bogus', 'Unknown thermo_modify line argument: bogus'],
  ['format line "%d %g %g %g"', null],
  ['format float %10.4f', null],
  ['format int %8d', null],
  ['format 2 %12.6f', null],
  ['format none', null],
  ['format 9 %g', 'Invalid thermo_modify format argument: 9'],
  ['format float %d', 'Invalid thermo_modify float format: conversion 1 of \'%d\' formats integer values, but floating-point values are provided'],
  ['temp t2', null],
  ['press p2', null],
  ['temp nope', 'Could not find thermo_modify temperature compute nope'],
  ['press nope', 'Could not find thermo_modify pressure compute nope'],
  ['triclinic/general yes', 'Thermo_modify triclinic/general cannot be used if simulation box is not general triclinic'],
  ['triclinic/general no', null],
  ['colname 2 T', null],
  ['colname temp T', null],
  ['colname default', null],
  ['colname 9 X', 'Invalid thermo_modify colname argument: 9'],
  ['universe yes', 'Unknown thermo_modify keyword: universe'],
  ['every 2', 'Unknown thermo_modify keyword: every'],
  ['bogus 1', 'Unknown thermo_modify keyword: bogus'],
  ['temp', 'Illegal thermo_modify temp command: missing argument(s)'],
  ['lost', 'Illegal thermo_modify lost command: missing argument(s)'],
];

describe('thermo_modify as native LAMMPS', () => {
  it.each(ROWS)('thermo_modify %s', async (kw, expected) => {
    const s = new Session({ emit: () => {}, writeFile: () => {} });
    let got: string | null = null;
    try { await s.execute(`${BASE}thermo_modify ${kw}\nrun 2\n`); } catch (e) { got = (e as Error).message; }
    if (expected === null) expect(got).toBeNull();
    else expect(got).toContain(expected);
  });
});
