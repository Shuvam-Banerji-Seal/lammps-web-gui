/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/write_coeff.cpp and
 * src/write_coeff.h (WriteCoeff::command; LAMMPS development team, Sandia National Laboratories). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * write_coeff file — write_coeff.html: the force field coefficients of the pair, bond, angle, dihedral and improper
 * styles are written as pair_coeff, bond_coeff, ... lines.
 *
 * The native command writes each style's data-file section into a temporary file (Pair::write_data_all for the pair,
 * write_data for the bonded styles) and then rewrites it line by line: each coefficient line gets its *_coeff command
 * in front, and the class 2 styles get their extra sections (BondBond, BondAngle, ...) as a type and a section tag.
 * The engine's section writers (dataCoeffsIJ, dataCoeffs, dataCrossSections) play the part of write_data_all and
 * write_data here, and the rewrite below is the same loop.
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { LAMMPS_VERSION, errorUrl } from './common';

const REGULAR_MODE = 0;
const CLASS2_MODE = 1;

/** The section tags of the class 2 extra sections (write_coeff.cpp, the strcmp chain; UreyBradley is not listed there). */
const CLASS2_SECTIONS: Record<string, string> = {
  'BondBond Coeffs\n': 'bb',
  'BondAngle Coeffs\n': 'ba',
  'MiddleBondTorsion Coeffs\n': 'mbt',
  'EndBondTorsion Coeffs\n': 'ebt',
  'AngleTorsion Coeffs\n': 'at',
  'AngleAngleTorsion Coeffs\n': 'aat',
  'BondBond13 Coeffs\n': 'bb13',
  'AngleAngle Coeffs\n': 'aa',
};

/**
 * The temporary file of native write_coeff before its rewrite: one block per style ("# <kind>_style <name>",
 * "<kind>_coeff", the section lines, "end"). Every entry ends with a newline, as fgets returns its line.
 */
const styleBlocks = (sys: System): string[] => {
  const out: string[] = [];
  const add = (kind: string, name: string, lines: string[], cross: { title: string; lines: string[] }[] = []) => {
    out.push(`# ${kind}_style ${name}\n`, `${kind}_coeff\n`);
    for (const l of lines) out.push(`${l}\n`);
    for (const sec of cross) {
      out.push('\n', `${sec.title}\n`, '\n');
      for (const l of sec.lines) out.push(`${l}\n`);
    }
    out.push('end\n');
  };
  const p = sys.ff.pair;
  if (p) {
    // Pair::write_data_all (the i <= j list) is what the PairIJ Coeffs data section writes
    const lines = p.dataCoeffsIJ();
    if (lines) add('pair', p.name, lines);
  }
  for (const kind of ['bond', 'angle', 'dihedral', 'improper'] as const) {
    const st = sys.ff[kind];
    if (!st) continue;
    const lines = st.dataCoeffs();
    if (lines) add(kind, st.name, lines, st.dataCrossSections());
  }
  return out;
};

/** write_coeff file: the native temporary file rewritten line by line into the coefficient file. */
const writeCoeff: Handler = ({ sys }, a) => {
  if (!sys.hasBox) throw new StyleError(`Write_coeff command before simulation box is defined${errorUrl(33)}`);
  if (a.length !== 1) throw new StyleError('Illegal write_coeff command: missing argument(s)');
  const blocks = styleBlocks(sys);
  let pos = 0;
  const need = (): string => {
    const s = blocks[pos++];
    if (s === undefined) throw new StyleError('write_coeff: unexpected end of the coefficient lines');
    return s;
  };
  let out = `# LAMMPS coeff file via write_coeff, version ${LAMMPS_VERSION}\n`;
  while (pos < blocks.length) {
    let str = need();
    let mode = REGULAR_MODE;
    // some coeffs need special treatment: the class 2 angle, dihedral and improper styles have several sections
    if (str.includes('class2') && (str.includes('angle_style') || str.includes('dihedral_style') || str.includes('improper_style'))) {
      mode = CLASS2_MODE;
    }
    let section = '';
    out += str;                                  // the style line
    const coeff = need().replace(/\n$/, '');     // the coeff command line, without its newline
    str = need();
    while (str !== 'end\n') {
      if (mode === REGULAR_MODE) {
        out += `${coeff} ${str}`;
        str = need();
        continue;
      }
      // class 2 sections: all but the last section end with an empty line; skip it, read the section title and
      // gobble up one more empty line
      if (str === '\n') {
        str = need();
        if (Object.hasOwn(CLASS2_SECTIONS, str)) section = CLASS2_SECTIONS[str];
        need();
        str = need();
      }
      // the type number, then the rest of the line as it is (native: std::stoi, skip the spaces and the digits)
      const m = /^ *(\d+)/.exec(str);
      if (!m) throw new StyleError(`write_coeff: invalid coefficient line '${str.replace(/\n$/, '')}'`);
      out += `${coeff} ${Number(m[1])} ${section} ${str.slice(m[0].length)}`;
      str = need();
    }
    out += '\n';
  }
  sys.writeFile(a[0], out, false);
};

export const COMMANDS = { write_coeff: writeCoeff };
