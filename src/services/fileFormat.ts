import { FileFormat } from '../types';

/*
 * File-format detection by name and by content, kept apart from the parsers (fileParser.ts) so that the viewer can
 * detect a format without loading every parser into the initial bundle (they run in the parser worker).
 */

/**
 * Detects file format from filename extension.
 */
export const detectFileFormat = (filename: string): FileFormat => {
  const ext = filename.toLowerCase().split('.').pop() || '';
  switch (ext) {
    case 'xyz':
      return 'xyz';
    case 'pdb':
    case 'ent':
      return 'pdb';
    case 'cif':
    case 'mmcif':
      return 'cif';
    case 'lammpstrj':
    case 'dump':
      return 'lammpsdump';
    case 'data':
    case 'lammps':
    case 'lmp':
    case 'txt':
    default:
      return 'lammps';
  }
};

/**
 * Detects file format from content heuristics (used when the user pastes
 * data or uploads a generic extension like .txt).
 */
export const detectFormatFromContent = (content: string): FileFormat => {
  const lines = content.trim().split('\n');
  if (lines.length === 0) return 'lammps';

  // CIF: data_ block or _atom_site / _cell tags
  for (const line of lines.slice(0, 80)) {
    const t = line.trim();
    if (/^data_/i.test(t) || /^_atom_site/i.test(t) || /^_cell_length/i.test(t)) {
      return 'cif';
    }
  }

  // PDB: look for ATOM or HETATM records
  for (const line of lines.slice(0, 50)) {
    if (/^(ATOM|HETATM|HEADER|CRYST1|COMPND)/.test(line)) {
      return 'pdb';
    }
  }

  // XYZ: first line is a number (atom count), third line "Sym X Y Z"
  const firstLine = lines[0].trim();
  if (/^\d+$/.test(firstLine) && lines.length >= 3) {
    const thirdLine = lines[2].trim();
    if (/^[A-Za-z]{1,2}\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+/.test(thirdLine)) {
      return 'xyz';
    }
  }

  // LAMMPS dump: ITEM: blocks
  for (const line of lines.slice(0, 20)) {
    if (/^ITEM:\s+TIMESTEP/.test(line.trim())) return 'lammpsdump';
  }

  // LAMMPS: header keywords
  for (const line of lines.slice(0, 40)) {
    if (/\b(atoms|bonds|atom types|bond types|xlo xhi|Atoms\s*#)\b/i.test(line)) {
      return 'lammps';
    }
  }

  return 'lammps';
};
