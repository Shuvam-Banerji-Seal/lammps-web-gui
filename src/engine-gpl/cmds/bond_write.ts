/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/bond.cpp (Bond::write_file) and the
 * command dispatch of src/input.cpp (Input::bond_write). LAMMPS development team, Sandia National Laboratories. This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * bond_write btype N inner outer file keyword [itype jtype] — bond_write.html: "Write energy and force values to a file as
 * a function of distance for the currently defined bond style for a selected bond type." The optional itype and jtype
 * are two atom types, checked as native does. The native command evaluates Bond::single() at each length; the engine has
 * no single() for bond styles, so the table is taken from a dummy system of two bonded atoms (as angle_write does): the
 * energy is the bond energy and the force is the x force on atom 2, placed at +r from atom 1.
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { formatNumber } from '../../engine/cformat';
import { minEnergyForce } from '../../engine/run/minimizers';
import { nativeFloat, nativeInt } from './common';
import { dummySession, finite, startTableFile, tableRow } from './table_write';

/** Atom styles that allow bonds (native atom_vec_bond, atom_vec_angle/molecular/full, bpm/sphere, template, oxdna). */
const BONDS_ALLOWED = /^(bond|angle|molecular|full|bpm\/sphere|template:|oxdna)/;

const bondWrite: Handler = async ({ sys, session }, a) => {
  if (!BONDS_ALLOWED.test(sys.atomStyle) && !/^hybrid /.test(sys.atomStyle)) {
    throw new StyleError('Bond_write command when no bonds allowed');
  }
  const bond = sys.ff.bond;
  if (!bond) throw new StyleError('Bond_write command before bond_style is defined');
  // Bond::write_file
  if (a.length !== 6 && a.length !== 8) throw new StyleError('Illegal bond_write command');
  const ntypes = sys.state.ntypes;
  if (a.length === 8) {
    const itype = nativeInt(a[6]);
    const jtype = nativeInt(a[7]);
    if (itype < 1 || itype > ntypes) throw new StyleError(`Invalid atom type ${itype} in bond_write command`);
    if (jtype < 1 || jtype > ntypes) throw new StyleError(`Invalid atom type ${jtype} in bond_write command`);
  }
  const btype = nativeInt(a[0]);
  const n = nativeInt(a[1]);
  const inner = nativeFloat(a[2]);
  const outer = nativeFloat(a[3]);
  if (inner <= 0 || inner >= outer) throw new StyleError(`Invalid rlo=${inner} / rhi=${outer} values in bond_write command.`);
  if (n < 2) throw new StyleError('Must have at least 2 table values');
  const nbt = sys.state.topo.nbondtypes;
  if (btype < 1 || btype > nbt) throw new StyleError(`Invalid bond type ${btype} in bond_write command`);
  const tableFile = a[4];
  const keyword = a[5];
  // Bond::equilibrium_distance of the style (the EQ value of the header)
  const r0raw = bond.equilibrium(btype);
  const r0 = finite(r0raw) ? r0raw : 0;
  const coeffLines = bond.dataCoeffs() ?? [];
  const { head, append } = startTableFile(sys, { file: tableFile, cmd: 'bond_write' });
  sys.writeFile(tableFile, `${head}# Bond potential ${bond.name} for bond type ${btype}: i,r,energy,force\n\n${keyword}\nN ${n} EQ ${formatNumber(r0, '%.15g')}\n\n`, append);

  // the dummy system: two atoms bonded with type btype; atom 2 is moved along +x to the length r of each row
  const L = Math.ceil(outer) + 10;
  const w = await dummySession(session, [
    `units ${sys.units.style}`,
    'atom_style bond',
    'atom_modify map array',
    'boundary f f f',
    `region box block ${-L} ${L} ${-L} ${L} ${-L} ${L}`,
    `create_box ${ntypes} box bond/types ${nbt} extra/bond/per/atom 2 extra/special/per/atom 2`,
    'create_atoms 1 single 0.0 0.0 0.0',
    'create_atoms 1 single 1.0 0.0 0.0',
    `create_bonds single/bond ${btype} 1 2`,
    'pair_style zero 10.0',
    'pair_coeff * *',
    'mass * 1.0',
    `bond_style ${bond.name}`,
    ...coeffLines.map((l) => `bond_coeff ${l}`),
    'run 0 post no',
  ]);
  // energy and the force on atom 2 (native Bond::single gives fforce, and the table holds fforce * r = -dE/dr)
  const k2 = w.indexOfId(2);
  let rows = '';
  const dr = (outer - inner) / (n - 1);
  for (let i = 0; i < n; i++) {
    const r = inner + dr * i;
    w.state.x[3 * k2] = r;
    const e = minEnergyForce(w, false);
    rows += tableRow(i + 1, r, e, w.state.f[3 * k2]);
  }
  sys.writeFile(tableFile, rows, true);
};

export const COMMANDS = { bond_write: bondWrite };
