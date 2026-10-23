/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/dihedral_write.cpp and
 * src/dihedral_write.h (DihedralWrite::command; LAMMPS development team, Sandia National Laboratories). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * dihedral_write dtype N file keyword — dihedral_write.html: "Write a table file of the energy and force of a dihedral
 * style for the dihedral type dtype, at N equally spaced dihedral angles from 0 to 360 degrees, in the format of
 * dihedral_style table". The native command runs a second LAMMPS instance with four atoms; the engine runs the same
 * setup in a dummy session (table_write.ts) and moves the fourth atom around the axis of the second and third.
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { cos, sin } from '../libm/sincos';
import { minEnergyForce } from '../../engine/run/minimizers';
import { errorUrl } from './common';
import { dummySession, finite, startTableFile, tableRow } from './table_write';
import { inumeric } from '../../engine/force/util';

const DEG2RAD = Math.PI / 180;
const EPSILON = 6.5e-6;

const dihedralWrite: Handler = async ({ sys, session }, a) => {
  if (!sys.hasBox) throw new StyleError(`Dihedral_write command before simulation box is defined${errorUrl(33)}`);
  const dihedral = sys.ff.dihedral;
  if (!dihedral) throw new StyleError(`Dihedral_write command before an dihedral_style is defined${errorUrl(33)}`);
  if (dihedral.dataCoeffs() === null) throw new StyleError('Dihedral style must support writing coeffs to data file for dihedral_write');
  if (/^(charmm|class2)/.test(dihedral.name)) throw new StyleError(`Dihedral_write command is not compatible with dihedral style ${dihedral.name}`);
  if (a.length !== 4) throw new StyleError('Illegal dihedral_write command');
  const dtype = inumeric(a[0]);
  const ndt = sys.state.topo.ndihedraltypes;
  if (dtype <= 0 || dtype > ndt) throw new StyleError(`Invalid dihedral type ${dtype} in dihedral_write command`);
  const n = inumeric(a[1]);
  const tableFile = a[2];
  const keyword = a[3];
  if (n < 2) throw new StyleError('Must have at least 2 table values');
  const coeffLines = dihedral.dataCoeffs() ?? [];
  const { head, append } = startTableFile(sys, { file: tableFile, cmd: 'dihedral_write' });

  // the dummy system: atoms 1..4 in a zig-zag, the bond 1-2-3-4 of type dtype; atom 4 moves on a circle around the 2-3 axis
  const lines = [
    `units ${sys.units.style}`,
    'atom_style molecular',
    'atom_modify map array',
    'boundary f f f',
    'region box block -2 2 -2 2 -2 2',
    `create_box ${sys.state.ntypes} box dihedral/types ${ndt} extra/dihedral/per/atom 1 extra/special/per/atom 4`,
    'create_atoms 1 single 1.0 0.0 -1.0',
    'create_atoms 1 single 0.0 0.0 -1.0',
    'create_atoms 1 single 0.0 0.0 1.0',
    'create_atoms 1 single 1.0 0.0 1.0',
    `create_bonds single/dihedral ${dtype} 1 2 3 4`,
    'pair_style zero 10.0',
    'pair_coeff * *',
    'mass * 1.0',
    `dihedral_style ${dihedral.name}`,
    ...coeffLines.slice(0, ndt).map((l) => `dihedral_coeff ${l}`),
    'run 0 post no',
  ];
  const w = await dummySession(session, lines);
  // energy at the dihedral angle theta (native GET_ENERGY: the x and y of atom 4, its z kept)
  const energyAt = (theta: number): number => {
    const k4 = w.indexOfId(4);
    w.state.x[3 * k4] = cos(theta * DEG2RAD);
    w.state.x[3 * k4 + 1] = sin(theta * DEG2RAD);
    return minEnergyForce(w, false);
  };

  let out = head;
  out += `# Dihedral potential ${dihedral.name} for dihedral type ${dtype}: i,theta,energy,force\n`;
  out += `\n${keyword}\nN ${n} DEGREES\n\n`;
  const dtheta = 360.0 / n;
  for (let i = 0; i < n; i++) {
    const phi1 = energyAt(dtheta * i - EPSILON);
    const phi2 = energyAt(dtheta * i + EPSILON);
    let phi = energyAt(dtheta * i);
    if (!finite(phi)) phi = 0.0;
    // the force from numerical differentiation
    let f = (-0.5 * (phi2 - phi1)) / EPSILON;
    if (!finite(f)) f = 0.0;
    out += tableRow(i + 1, dtheta * i, phi, f);
  }
  sys.writeFile(tableFile, out, append);
};

export const COMMANDS = { dihedral_write: dihedralWrite };
