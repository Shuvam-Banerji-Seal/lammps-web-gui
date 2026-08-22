import { Atom, Bond } from '../types';
import { COVALENT_RADII, DEFAULT_COVALENT_RADIUS } from '../constants';

export interface BondInferenceOptions {
  /**
   * Multiplier on sum-of-covalent-radii cutoff. Default 1.2: chosen so the
   * real H2 bond (0.74 A) is captured against Cordero H radius 0.31
   * (0.62 * 1.194 = 0.74), while staying tight enough to reject most
   * non-bonded contacts.
   */
  tolerance?: number;
  /** Hard cap on bond length in Angstroms regardless of radii (guards huge cells). */
  maxBondLength?: number;
}

/**
 * Uniform spatial hash grid for O(n) neighbor queries.
 * Atoms are bucketed into cubic cells of edge `cellSize`; a pair can only
 * bond if it lies within adjacent cells, so each atom checks <=27 buckets.
 */
class SpatialGrid {
  private cells = new Map<string, number[]>();

  constructor(private cellSize: number) {}

  private key(cx: number, cy: number, cz: number): string {
    return `${cx},${cy},${cz}`;
  }

  insert(atom: Atom, index: number): void {
    const cx = Math.floor(atom.x / this.cellSize);
    const cy = Math.floor(atom.y / this.cellSize);
    const cz = Math.floor(atom.z / this.cellSize);
    const k = this.key(cx, cy, cz);
    let bucket = this.cells.get(k);
    if (!bucket) {
      bucket = [];
      this.cells.set(k, bucket);
    }
    bucket.push(index);
  }

  /** Iterate indices in the 3x3x3 cell neighborhood around the given point. */
  neighbors(x: number, y: number, z: number, visit: (idx: number) => void): void {
    const cx = Math.floor(x / this.cellSize);
    const cy = Math.floor(y / this.cellSize);
    const cz = Math.floor(z / this.cellSize);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = this.cells.get(this.key(cx + dx, cy + dy, cz + dz));
          if (bucket) {
            for (const idx of bucket) visit(idx);
          }
        }
      }
    }
  }
}

const covalentRadius = (atomicNumber: number): number =>
  COVALENT_RADII[atomicNumber] ?? DEFAULT_COVALENT_RADIUS;

/**
 * Infer bonds from coordinates using covalent radii and a spatial hash grid.
 * Complexity: O(n * k) where k = average neighbors per cell neighborhood,
 * versus O(n^2) for naive all-pairs scanning. Deduplicates pairs by atom id.
 *
 * `typeOf` maps an Atom to its atomic number when Atom.type is not already
 * the atomic number (callers with symbol-derived types pass identity).
 */
export const inferBonds = (
  atoms: Atom[],
  options: BondInferenceOptions = {}
): Bond[] => {
  const tolerance = options.tolerance ?? 1.2;
  const maxBondLength = options.maxBondLength ?? Infinity;
  const bonds: Bond[] = [];
  if (atoms.length < 2) return bonds;

  // Correctness invariant of the 3x3x3 neighborhood: cellSize must be >= the
  // maximum accepted bond distance, else a valid pair could straddle
  // non-adjacent cells. Compute over ALL atoms (O(n), cheap).
  let largestRadius = DEFAULT_COVALENT_RADIUS;
  for (const a of atoms) {
    const r = covalentRadius(a.type);
    if (r > largestRadius) largestRadius = r;
  }
  const maxPairDistance = Math.min(2 * largestRadius * tolerance, maxBondLength);
  const cellSize = maxPairDistance;

  const grid = new SpatialGrid(cellSize);
  atoms.forEach((atom, i) => grid.insert(atom, i));

  const seen = new Set<string>();
  let bondId = 1;

  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i];
    const ra = covalentRadius(a.type);
    grid.neighbors(a.x, a.y, a.z, j => {
      if (j <= i) return; // each unordered pair once
      const b = atoms[j];
      const rb = covalentRadius(b.type);
      const threshold = Math.min((ra + rb) * tolerance, maxBondLength);
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > threshold * threshold) return;

      const key = a.id < b.id ? `${a.id}-${b.id}` : `${b.id}-${a.id}`;
      if (seen.has(key)) return;
      seen.add(key);
      bonds.push({ id: bondId++, type: 1, atom1Id: a.id, atom2Id: b.id });
    });
  }

  return bonds;
};
