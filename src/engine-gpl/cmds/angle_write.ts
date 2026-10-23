/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/angle_write.cpp and
 * src/angle_write.h (AngleWrite::command; LAMMPS development team, Sandia National Laboratories). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * angle_write atype N file keyword — angle_write.html: "Write a table file of the energy and force of an angle style
 * for the angle type atype, at N angles from 0 to 180 degrees, in the format of angle_style table". The native command
 * runs a second LAMMPS instance with three atoms in a line; the engine runs the same setup in a dummy session
 * (table_write.ts) and moves the third atom around the central one.
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { cos, sin } from '../libm/sincos';
import { minEnergyForce } from '../../engine/run/minimizers';
import { formatNumber } from '../../engine/cformat';
import { errorUrl } from './common';
import { dummySession, finite, startTableFile, tableRow } from './table_write';
import { inumeric } from '../../engine/force/util';

/** math_const.h: DEG2RAD = MY_PI/180, RAD2DEG = 180/MY_PI, and the epsilon of the numerical derivative. */
const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;
const EPSILON = 6.5e-6;

const angleWrite: Handler = async ({ sys, session }, a) => {
  if (!sys.hasBox) throw new StyleError(`Angle_write command before simulation box is defined${errorUrl(33)}`);
  const angle = sys.ff.angle;
  if (!angle) throw new StyleError(`Angle_write command before an angle_style is defined${errorUrl(33)}`);
  if (angle.dataCoeffs() === null) throw new StyleError('Angle style must support writing coeffs to data file for angle_write');
  if (/^class2/.test(angle.name)) throw new StyleError(`Angle_write command is not compatible with angle style ${angle.name}`);
  if (a.length !== 4) throw new StyleError('Illegal angle_write command');
  const atype = inumeric(a[0]);
  if (atype <= 0 || atype > sys.state.topo.nangletypes) throw new StyleError(`Invalid angle type ${atype} in angle_write command`);
  const n = inumeric(a[1]);
  const tableFile = a[2];
  const keyword = a[3];
  if (n < 2) throw new StyleError('Must have at least 2 table values');
  // the equilibrium angle in degrees (native equilibrium_angle(atype) * RAD2DEG)
  const theta0 = angle.equilibriumRadians(atype) * RAD2DEG;
  const coeffLines = angle.dataCoeffs() ?? [];
  const { head, append } = startTableFile(sys, { file: tableFile, cmd: 'angle_write' });

  // the dummy system: three atoms, 1 at the origin (the centre), 2 at (1,0,0), 3 at (-1,0,0); one angle 2-1-3
  const ntypes = sys.state.ntypes;
  const nat = sys.state.topo.nangletypes;
  const lines = [
    `units ${sys.units.style}`,
    'atom_style angle',
    'atom_modify map array',
    'boundary f f f',
    'region box block -2 2 -2 2 -1 1',
    `create_box ${ntypes} box angle/types ${nat} extra/angle/per/atom 1 extra/special/per/atom 4`,
    'create_atoms 1 single 0.0 0.0 0.0',
    'create_atoms 1 single 1.0 0.0 0.0',
    'create_atoms 1 single -1.0 0.0 0.0',
    `create_bonds single/angle ${atype} 2 1 3`,
    'pair_style zero 10.0',
    'pair_coeff * *',
    'mass * 1.0',
    `angle_style ${angle.name}`,
    ...coeffLines.slice(0, nat).map((l) => `angle_coeff ${l}`),
    'run 0 post no',
  ];
  const w = await dummySession(session, lines);
  const ws = w.state;
  // energy of the angle 2-1-3 with atom 3 at the angle theta (native GET_ENERGY: x and y of atom 3, z kept)
  const energyAt = (theta: number): number => {
    const k3 = w.indexOfId(3);
    ws.x[3 * k3] = cos(theta * DEG2RAD);
    ws.x[3 * k3 + 1] = sin(theta * DEG2RAD);
    return minEnergyForce(w, false);
  };

  let out = head;
  // the header, then the N rows (native fprintf calls, in the same order)
  out += `# Angle potential ${angle.name} for angle type ${atype}: i,theta,energy,force\n`;
  out += `\n${keyword}\nN ${n} EQ ${formatNumber(theta0, '%.15g')}\n\n`;
  const dtheta = 180.0 / (n - 1);
  // the force at 0 degrees is divergent: it comes from the interpolation to the right
  let phi = energyAt(0.0);
  let phi1 = energyAt(EPSILON);
  let phi2 = energyAt(2.0 * EPSILON);
  let f = (1.5 * phi - 2.0 * phi1 + 0.5 * phi2) / EPSILON;
  if (!finite(f)) f = 0.0;
  if (!finite(phi)) phi = 0.0;
  out += tableRow(1, 0.0, phi, f);
  for (let i = 1; i < n - 1; i++) {
    phi1 = energyAt(dtheta * i - EPSILON);
    phi2 = energyAt(dtheta * i + EPSILON);
    phi = energyAt(dtheta * i);
    if (!finite(phi)) phi = 0.0;
    // the force from numerical differentiation
    f = (-0.5 * (phi2 - phi1)) / EPSILON;
    if (!finite(f)) f = 0.0;
    out += tableRow(i + 1, dtheta * i, phi, f);
  }
  // the force at 180 degrees: from the interpolation to the left
  phi = energyAt(180.0);
  phi1 = energyAt(180.0 - EPSILON);
  phi2 = energyAt(180.0 - 2.0 * EPSILON);
  f = (2.0 * phi1 - 1.5 * phi - 0.5 * phi2) / EPSILON;
  if (!finite(f)) f = 0.0;
  if (!finite(phi)) phi = 0.0;
  out += tableRow(1, 180.0, phi, f);
  sys.writeFile(tableFile, out, append);
};

export const COMMANDS = { angle_write: angleWrite };
