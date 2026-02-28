import { Atom, Bond, MoleculeData, AtomTypeInfo } from '../types';
import { ELEMENT_DATA } from '../constants';

/**
 * Parses PDB (Protein Data Bank) file format.
 * Supports ATOM/HETATM records and CONECT records for bonds.
 */
export const parsePDBFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  const elementTypeMap: Record<string, number> = {};
  // PDB serial -> our atom id mapping
  const serialToId: Record<number, number> = {};
  const bondSet = new Set<string>();
  let bondId = 1;

  for (const line of lines) {
    const recordType = line.substring(0, 6).trim();

    if (recordType === 'ATOM' || recordType === 'HETATM') {
      const serial = parseInt(line.substring(6, 11).trim(), 10);
      const x = parseFloat(line.substring(30, 38).trim());
      const y = parseFloat(line.substring(38, 46).trim());
      const z = parseFloat(line.substring(46, 54).trim());

      // Element symbol from columns 77-78, or infer from atom name
      let symbol = line.length >= 78 ? line.substring(76, 78).trim() : '';
      if (!symbol) {
        // Infer from atom name (columns 13-16)
        const atomName = line.substring(12, 16).trim();
        symbol = atomName.replace(/[0-9]/g, '').substring(0, 2).trim();
        if (symbol.length > 1) {
          symbol = symbol[0].toUpperCase() + symbol[1].toLowerCase();
        }
      }

      if (isNaN(x) || isNaN(y) || isNaN(z)) continue;

      // Resolve element type
      const elem = ELEMENT_DATA.find(e => e.symbol === symbol);
      if (!(symbol in elementTypeMap)) {
        elementTypeMap[symbol] = elem ? elem.number : Object.keys(elementTypeMap).length + 200;
      }

      const type = elementTypeMap[symbol];
      const id = atoms.length + 1;
      serialToId[serial] = id;

      atoms.push({ id, molId: 1, type, charge: 0, x, y, z });

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    } else if (recordType === 'CONECT') {
      const tokens = line.substring(6).trim().split(/\s+/).map(Number);
      if (tokens.length >= 2) {
        const fromSerial = tokens[0];
        const fromId = serialToId[fromSerial];
        if (fromId) {
          for (let i = 1; i < tokens.length; i++) {
            const toId = serialToId[tokens[i]];
            if (toId) {
              // Deduplicate bonds
              const key = fromId < toId ? `${fromId}-${toId}` : `${toId}-${fromId}`;
              if (!bondSet.has(key)) {
                bondSet.add(key);
                bonds.push({ id: bondId++, type: 1, atom1Id: fromId, atom2Id: toId });
              }
            }
          }
        }
      }
    }
  }

  // Generate atom type metadata
  const atomTypes: Record<number, AtomTypeInfo> = {};
  const usedTypes = new Set<number>();
  atoms.forEach(a => usedTypes.add(a.type));

  usedTypes.forEach(type => {
    const elem = ELEMENT_DATA.find(e => e.number === type);
    atomTypes[type] = {
      id: type,
      mass: elem?.mass ?? 0,
      element: elem?.symbol ?? 'X',
      label: elem ? `${elem.name} (${elem.symbol})` : `Type ${type}`,
      count: atoms.filter(a => a.type === type).length
    };
  });

  const safeCenter = atoms.length > 0
    ? { x: (minX + maxX) / 2, y: (minY + maxY) / 2, z: (minZ + maxZ) / 2 }
    : { x: 0, y: 0, z: 0 };

  return {
    atoms,
    bonds,
    atomTypes,
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center: safeCenter
  };
};
