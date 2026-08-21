import { Atom, Bond, MoleculeData, AtomTypeInfo } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';
import { inferBonds } from './bondInference';

/**
 * Parses XYZ file format.
 *
 *   Line 1: Number of atoms
 *   Line 2: Comment line
 *   Lines 3+: ElementSymbol X Y Z [extra columns ignored]
 *
 * ExtendedXYZ extras after the coordinates are tolerated. Multi-trajectory
 * files (repeated frame blocks): only the FIRST frame is visualized.
 *
 * Bonds are inferred with an O(n) spatial-hash pass using covalent radii
 * (Cordero 2008) — replaces the historical O(n^2) fixed-threshold scan.
 */
export const parseXYZFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  const atoms: Atom[] = [];

  if (lines.length < 3) {
    throw new Error('Invalid XYZ file: too few lines');
  }

  const numAtoms = parseInt(lines[0].trim(), 10);
  if (!Number.isInteger(numAtoms) || numAtoms <= 0) {
    throw new Error('Invalid XYZ file: first line must be atom count');
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  // element token -> type id (atomic number when known)
  const elementTypeMap: Record<string, number> = {};
  let nextSyntheticTypeId = 1000; // above real atomic numbers

  for (let i = 2; i < lines.length && atoms.length < numAtoms; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const tokens = line.split(/\s+/);
    if (tokens.length < 4) continue;

    const rawSymbol = tokens[0];
    const x = parseFloat(tokens[1]);
    const y = parseFloat(tokens[2]);
    const z = parseFloat(tokens[3]);
    if (![x, y, z].every(Number.isFinite)) continue;

    // Normalize casing for lookup ("c" -> "C", "cl" -> "CL" matches map key)
    const atomicNumber = getAtomicNumberFromSymbol(rawSymbol);
    const lookupKey = rawSymbol.trim().toUpperCase();
    if (!(lookupKey in elementTypeMap)) {
      elementTypeMap[lookupKey] =
        atomicNumber !== undefined ? atomicNumber : nextSyntheticTypeId++;
    }
    const type = elementTypeMap[lookupKey];

    atoms.push({ id: atoms.length + 1, molId: 1, type, charge: 0, x, y, z });

    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
  }

  if (atoms.length < numAtoms) {
    throw new Error(
      `Invalid XYZ file: expected ${numAtoms} atoms but found ${atoms.length}`
    );
  }

  // --- Type metadata ---
  const atomTypes: Record<number, AtomTypeInfo> = {};
  const usedTypes = Array.from(new Set(atoms.map(a => a.type)));

  for (const type of usedTypes) {
    const elem = ELEMENT_DATA.find(e => e.number === type);
    let count = 0;
    for (const a of atoms) if (a.type === type) count++;
    atomTypes[type] = {
      id: type,
      mass: elem?.mass ?? 0,
      element: elem?.symbol ?? 'X',
      label: elem ? `${elem.name} (${elem.symbol})` : `Type ${type}`,
      count,
    };
  }

  // --- Bonds via spatial hash (covalent radii) ---
  const bonds: Bond[] = inferBonds(atoms);

  const safeCenter = atoms.length > 0
    ? { x: (minX + maxX) / 2, y: (minY + maxY) / 2, z: (minZ + maxZ) / 2 }
    : { x: 0, y: 0, z: 0 };

  return {
    atoms,
    bonds,
    atomTypes,
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center: safeCenter,
  };
};
