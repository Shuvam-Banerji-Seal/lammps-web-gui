import { Atom, Bond, MoleculeData, AtomTypeInfo, BoxBounds } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';

/**
 * Parses PDB (Protein Data Bank) format.
 *
 * Supported records:
 *  - ATOM / HETATM : coordinates + element
 *  - CONECT        : explicit bonds (deduplicated)
 *  - CRYST1        : unit cell -> simulation box (a b c alpha beta gamma)
 *
 * Element resolution order per atom:
 *  1. Columns 77-78 (element right-justified) — the authoritative field
 *  2. Atom-name heuristic on columns 13-16: strip digits/charges, then try
 *     the two-letter interpretation ONLY when column 13 is a space (PDB
 *     convention: two-letter elements are right-justified in the name field),
 *     else single-letter.
 */
export const parsePDBFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let box: BoxBounds | undefined;

  const elementTypeMap: Record<string, number> = {};
  const serialToId: Record<number, number> = {};
  const bondSet = new Set<string>();
  let bondId = 1;

  const inferElementFromAtomName = (atomNameRaw: string): string => {
    const atomName = atomNameRaw.trim();
    if (!atomName) return '';
    // PDB convention: element field of the name is left-padded for 2-letter elements.
    // e.g. " CL ", "CA  " (alpha carbon, starts at col 13), "FE"
    const stripped = atomName.replace(/[^A-Za-z]/g, '');
    if (!stripped) return '';

    if (/^[A-Z][a-z]$/.test(stripped)) return stripped;           // already "Cl" style
    if (atomName.startsWith(' ')) {                                // right-justified two-letter
      const two = stripped.slice(0, 2);
      const norm = two[0].toUpperCase() + two[1].toLowerCase();
      if (getAtomicNumberFromSymbol(norm) !== undefined) return norm;
    }
    const one = stripped[0].toUpperCase();
    if (getAtomicNumberFromSymbol(one) !== undefined) return one;
    // last resort: two-letter from a name like "CL1" written without padding
    if (stripped.length >= 2) {
      const two = stripped[0].toUpperCase() + stripped[1].toLowerCase();
      if (getAtomicNumberFromSymbol(two) !== undefined) return two;
    }
    return one;
  };

  for (const line of lines) {
    const recordType = line.substring(0, 6).trim();

    if (recordType === 'ATOM' || recordType === 'HETATM') {
      const serial = parseInt(line.substring(6, 11).trim(), 10);
      const x = parseFloat(line.substring(30, 38).trim());
      const y = parseFloat(line.substring(38, 46).trim());
      const z = parseFloat(line.substring(46, 54).trim());
      if (![x, y, z].every(Number.isFinite)) continue;

      // 1. Authoritative element columns 77-78
      let symbol = line.length >= 78 ? line.substring(76, 78).trim() : '';
      let atomicNumber = getAtomicNumberFromSymbol(symbol);

      // 2. Heuristic from the atom name
      if (!atomicNumber) {
        symbol = inferElementFromAtomName(line.substring(12, 16));
        atomicNumber = getAtomicNumberFromSymbol(symbol);
      }

      const lookupKey = (symbol || 'X').toUpperCase();
      if (!(lookupKey in elementTypeMap)) {
        elementTypeMap[lookupKey] =
          atomicNumber !== undefined ? atomicNumber : 900 + Object.keys(elementTypeMap).length;
      }
      const type = elementTypeMap[lookupKey];

      const id = atoms.length + 1;
      if (Number.isFinite(serial)) serialToId[serial] = id;

      atoms.push({ id, molId: 1, type, charge: 0, x, y, z });

      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
    } else if (recordType === 'CONECT') {
      const tokens = line.substring(6).trim().split(/\s+/).map(Number);
      if (tokens.length >= 2 && Number.isFinite(tokens[0])) {
        const fromId = serialToId[tokens[0]];
        if (fromId) {
          for (let i = 1; i < tokens.length; i++) {
            const toId = serialToId[tokens[i]];
            if (toId && toId !== fromId) {
              const key = fromId < toId ? `${fromId}-${toId}` : `${toId}-${fromId}`;
              if (!bondSet.has(key)) {
                bondSet.add(key);
                bonds.push({ id: bondId++, type: 1, atom1Id: fromId, atom2Id: toId });
              }
            }
          }
        }
      }
    } else if (recordType === 'CRYST1') {
      // CRYST1: cols 7-15 a, 16-24 b, 25-33 c (Angstroms); angles ignored for box render
      const a = parseFloat(line.substring(6, 15).trim());
      const b = parseFloat(line.substring(15, 24).trim());
      const c = parseFloat(line.substring(24, 33).trim());
      if ([a, b, c].every(Number.isFinite) && a > 0 && b > 0 && c > 0) {
        box = { xlo: 0, xhi: a, ylo: 0, yhi: b, zlo: 0, zhi: c };
      }
    }
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

  const safeCenter = (() => {
    if (box) return { x: box.xhi / 2, y: box.yhi / 2, z: box.zhi / 2 };
    if (atoms.length > 0) {
      return { x: (minX + maxX) / 2, y: (minY + maxY) / 2, z: (minZ + maxZ) / 2 };
    }
    return { x: 0, y: 0, z: 0 };
  })();

  return {
    atoms,
    bonds,
    atomTypes,
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center: safeCenter,
    ...(box ? { box } : {}),
  };
};
