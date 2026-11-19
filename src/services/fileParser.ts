import { MoleculeData, FileFormat } from '../types';
import { parseDataFile } from './parser';
import { parseXYZFile } from './xyzParser';
import { parsePDBFile } from './pdbParser';
import { parseCIFFile } from './cifParser';
import { parseDumpFile } from './dumpParser';

import { detectFormatFromContent } from './fileFormat';
export { detectFileFormat, detectFormatFromContent } from './fileFormat';

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
    case 'cif':
      return parseCIFFile(content);
    case 'lammpsdump':
      return parseDumpFile(content);
    case 'lammps':
    default:
      return parseDataFile(content);
  }
};
