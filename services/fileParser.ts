import { MoleculeData, FileFormat } from '../types';
import { parseDataFile } from './parser';
import { parseXYZFile } from './xyzParser';
import { parsePDBFile } from './pdbParser';

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
    case 'data':
    case 'lammps':
    case 'lmp':
    case 'txt':
    default:
      return 'lammps';
  }
};

/**
 * Detects file format from content heuristics.
 */
export const detectFormatFromContent = (content: string): FileFormat => {
  const lines = content.trim().split('\n');
  if (lines.length === 0) return 'lammps';

  // XYZ: first line is a number (atom count)
  const firstLine = lines[0].trim();
  if (/^\d+$/.test(firstLine) && lines.length >= 3) {
    // Check if line 3+ matches "Symbol X Y Z"
    const thirdLine = lines[2].trim();
    if (/^[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+/.test(thirdLine)) {
      return 'xyz';
    }
  }

  // PDB: look for ATOM or HETATM records
  for (const line of lines.slice(0, 50)) {
    if (line.startsWith('ATOM') || line.startsWith('HETATM') || line.startsWith('HEADER')) {
      return 'pdb';
    }
  }

  return 'lammps';
};

/**
 * Parses molecular data from a string using the specified or auto-detected format.
 */
export const parseFile = (content: string, format?: FileFormat): MoleculeData => {
  const detectedFormat = format || detectFormatFromContent(content);

  switch (detectedFormat) {
    case 'xyz':
      return parseXYZFile(content);
    case 'pdb':
      return parsePDBFile(content);
    case 'lammps':
    default:
      return parseDataFile(content);
  }
};
