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
  /**
   * The atomic number of an atom, from its type's element (atomicNumberByType in elementTypes.ts); undefined for an
   * atom whose element is unknown, which then takes part in no bond (its covalent radius is unknown). Without this
   * option Atom.type is taken as the atomic number, which holds for the parsers that number types by element (CIF,
   * PDB).
   */
  atomicNumberOf?: (atom: Atom) => number | undefined;
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

const typeAsAtomicNumber = (a: Atom): number | undefined => a.type;

/** The longest bond inferBonds can accept among these atoms: twice the largest covalent radius times the tolerance. */
export const maxBondDistance = (
  atoms: readonly Atom[],
  tolerance = 1.2,
  atomicNumberOf: (atom: Atom) => number | undefined = typeAsAtomicNumber,
): number => {
  let largest = DEFAULT_COVALENT_RADIUS;
  for (const a of atoms) {
    const z = atomicNumberOf(a);
    if (z === undefined) continue;
    const r = covalentRadius(z);
    if (r > largest) largest = r;
  }
  return 2 * largest * tolerance;
};

/**
 * Noble gases (He, Ne, Ar, Kr, Xe, Rn, Og). Closed-shell atoms do not form
 * covalent bonds in any structure this viewer renders — they interact only
 * through dispersion, so the covalent-radius criterion must not be applied
 * to them. Without this guard the bundled LJ-liquid demo (108 Ar atoms,
 * public/examples/lj-melt.lammpstrj) gained ~1554 phantom sticks.
 */
const NOBLE_GASES = new Set([2, 10, 18, 36, 54, 86, 118]);

/** Two atoms closer than this fraction of the sum of their covalent radii
 * are an overlap artefact (duplicate/pasted coordinates), not a bond. */
const OVERLAP_FRACTION = 0.1;

/**
 * Infer bonds from coordinates using covalent radii and a spatial hash grid.
 * Complexity: O(n * k) where k = average neighbors per cell neighborhood,
 * versus O(n^2) for naive all-pairs scanning. Deduplicates pairs by atom id.
 *
 * `options.atomicNumberOf` gives an atom's atomic number from its type's element; an atom of unknown element is
 * left unbonded.
 */
export const inferBonds = (
  atoms: Atom[],
  options: BondInferenceOptions = {}
): Bond[] => {
  const tolerance = options.tolerance ?? 1.2;
  const maxBondLength = options.maxBondLength ?? Infinity;
  const zOf = options.atomicNumberOf ?? typeAsAtomicNumber;
  const bonds: Bond[] = [];
  if (atoms.length < 2) return bonds;

  // The atomic number of every atom once (NaN: unknown element, never bonded).
  const zs = new Float64Array(atoms.length);
  for (let i = 0; i < atoms.length; i++) zs[i] = zOf(atoms[i]) ?? NaN;

  // Correctness invariant of the 3x3x3 neighborhood: cellSize must be >= the
  // maximum accepted bond distance, else a valid pair could straddle
  // non-adjacent cells. Compute over ALL atoms (O(n), cheap).
  let largestRadius = DEFAULT_COVALENT_RADIUS;
  for (let i = 0; i < atoms.length; i++) {
    if (Number.isNaN(zs[i])) continue;
    const r = covalentRadius(zs[i]);
    if (r > largestRadius) largestRadius = r;
  }
  const maxPairDistance = Math.min(2 * largestRadius * tolerance, maxBondLength);
  const cellSize = maxPairDistance;

  const grid = new SpatialGrid(cellSize);
  atoms.forEach((atom, i) => { if (!Number.isNaN(zs[i])) grid.insert(atom, i); });

  const seen = new Set<string>();
  let bondId = 1;

  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i];
    const za = zs[i];
    if (Number.isNaN(za)) continue;
    const ra = covalentRadius(za);
    grid.neighbors(a.x, a.y, a.z, j => {
      if (j <= i) return; // each unordered pair once
      const b = atoms[j];
      const zb = zs[j];
      const rb = covalentRadius(zb);
      const threshold = Math.min((ra + rb) * tolerance, maxBondLength);
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > threshold * threshold) return;
      // Overlap artefact, not a bond: nearer than 10% of ra+rb means the two
      // sites coincide (bad parse / duplicated row), and drawing a stick
      // there just paints a blob.
      if (d2 < (OVERLAP_FRACTION * (ra + rb)) * (OVERLAP_FRACTION * (ra + rb))) return;
      // Noble gases never bond (see NOBLE_GASES above).
      if (NOBLE_GASES.has(za) || NOBLE_GASES.has(zb)) return;

      const key = a.id < b.id ? `${a.id}-${b.id}` : `${b.id}-${a.id}`;
      if (seen.has(key)) return;
      seen.add(key);
      bonds.push({ id: bondId++, type: 1, atom1Id: a.id, atom2Id: b.id });
    });
  }

  return bonds;
};
