import { Atom, Bond, MoleculeData, AtomTypeInfo } from '../types';
import { ELEMENT_DATA } from '../constants';

/**
 * Parses XYZ file format.
 * Format:
 *   Line 1: Number of atoms
 *   Line 2: Comment line
 *   Lines 3+: Element X Y Z
 */
export const parseXYZFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];

  if (lines.length < 3) {
    throw new Error('Invalid XYZ file: too few lines');
  }

  const numAtoms = parseInt(lines[0].trim(), 10);
  if (isNaN(numAtoms) || numAtoms <= 0) {
    throw new Error('Invalid XYZ file: first line must be atom count');
  }

  // Line 2 is comment, skip it

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  // Track element symbol -> type ID mapping
  const elementTypeMap: Record<string, number> = {};
  let nextTypeId = 1;

  for (let i = 2; i < lines.length && atoms.length < numAtoms; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const tokens = line.split(/\s+/);
    if (tokens.length < 4) continue;

    const symbol = tokens[0];
    const x = parseFloat(tokens[1]);
    const y = parseFloat(tokens[2]);
    const z = parseFloat(tokens[3]);

    if (isNaN(x) || isNaN(y) || isNaN(z)) continue;

    // Assign type ID based on element symbol
    if (!(symbol in elementTypeMap)) {
      // Try to find matching element
      const elem = ELEMENT_DATA.find(e => e.symbol === symbol);
      elementTypeMap[symbol] = elem ? elem.number : nextTypeId++;
    }

    const type = elementTypeMap[symbol];
    const id = atoms.length + 1;

    atoms.push({ id, molId: 1, type, charge: 0, x, y, z });

    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
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

  // Auto-generate bonds based on distance (simple heuristic)
  const bondDistanceThreshold = 1.8; // Angstroms - typical covalent bond
  let bondId = 1;
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dx = atoms[i].x - atoms[j].x;
      const dy = atoms[i].y - atoms[j].y;
      const dz = atoms[i].z - atoms[j].z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < bondDistanceThreshold) {
        bonds.push({ id: bondId++, type: 1, atom1Id: atoms[i].id, atom2Id: atoms[j].id });
      }
    }
  }

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
