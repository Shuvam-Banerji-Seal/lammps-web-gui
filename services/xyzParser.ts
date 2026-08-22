import { Atom, Bond, MoleculeData, AtomTypeInfo, TrajectoryFrame } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';
import { inferBonds } from './bondInference';

/**
 * Parses XYZ file format — including multi-frame trajectories.
 *
 *   Line 1: Number of atoms
 *   Line 2: Comment line
 *   Lines 3+: ElementSymbol X Y Z [extra columns ignored]
 *
 * Every frame block is captured; the FIRST frame defines `atoms`, `bonds`
 * (topology assumed stable across frames — standard for MD output) and the
 * element->type mapping, so colors stay consistent during playback.
 * min/max/center span ALL frames so the camera framing never jumps.
 *
 * Bonds are inferred with an O(n) spatial-hash pass using covalent radii.
 */
export const parseXYZFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  if (lines.length < 3) {
    throw new Error('Invalid XYZ file: too few lines');
  }

  // Symbol token -> type id, shared across frames for stable coloring.
  const elementTypeMap: Record<string, number> = {};
  let nextSyntheticTypeId = 1000;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  const frames: TrajectoryFrame[] = [];
  let cursor = 0;

  while (cursor < lines.length) {
    // skip blank separators between frames
    while (cursor < lines.length && !lines[cursor].trim()) cursor++;
    if (cursor >= lines.length) break;

    const numAtoms = parseInt(lines[cursor].trim(), 10);
    if (!Number.isInteger(numAtoms) || numAtoms <= 0) {
      if (frames.length === 0) {
        throw new Error('Invalid XYZ file: first line must be atom count');
      }
      break; // trailing junk after valid frames — stop gracefully
    }
    cursor++;

    const comment = cursor < lines.length ? lines[cursor].trim() : undefined;
    cursor++;

    const atoms: Atom[] = [];
    while (atoms.length < numAtoms && cursor < lines.length) {
      const line = lines[cursor++].trim();
      if (!line) continue;
      const tokens = line.split(/\s+/);
      if (tokens.length < 4) continue;
      const x = parseFloat(tokens[1]);
      const y = parseFloat(tokens[2]);
      const z = parseFloat(tokens[3]);
      if (![x, y, z].every(Number.isFinite)) continue;

      const atomicNumber = getAtomicNumberFromSymbol(tokens[0]);
      const lookupKey = tokens[0].trim().toUpperCase();
      if (!(lookupKey in elementTypeMap)) {
        elementTypeMap[lookupKey] =
          atomicNumber !== undefined ? atomicNumber : nextSyntheticTypeId++;
      }

      atoms.push({
        id: atoms.length + 1,
        molId: 1,
        type: elementTypeMap[lookupKey],
        charge: 0,
        x, y, z,
      });

      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
    }

    if (atoms.length < numAtoms) {
      if (frames.length === 0) {
        throw new Error(
          `Invalid XYZ file: expected ${numAtoms} atoms but found ${atoms.length}`
        );
      }
      break; // truncated trailing frame — drop it silently
    }

    frames.push({ comment: comment || undefined, atoms });
  }

  const firstFrame = frames[0];
  const atoms = firstFrame.atoms;

  // --- Type metadata from the reference frame ---
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

  // --- Bonds from the reference frame's topology ---
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
    ...(frames.length > 1 ? { frames } : {}),
  };
};
