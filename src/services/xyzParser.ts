import { Atom, Bond, MoleculeData, AtomTypeInfo, TrajectoryFrame, BoxBounds } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';
import { inferBonds } from './bondInference';

/**
 * Parses XYZ file format — including multi-frame trajectories AND the
 * extended-XYZ (extxyz) comment-line metadata.
 *
 *   Line 1: Number of atoms
 *   Line 2: Comment line (plain text, or extxyz key=value pairs)
 *   Lines 3+: ElementSymbol X Y Z [extra columns ignored]
 *
 * extxyz support [VERIFIED 2026-10-07, github.com/libAtoms/extxyz]:
 *   - The comment line is a list of `key=value` pairs; a value containing
 *     whitespace is double-quoted ("values containing whitespace are
 *     double-quoted"), with backslash escapes. Unknown keys are ignored.
 *   - `Lattice="ax ay az bx by bz cx cy cz"`: "3x3 matrix - rows are cell
 *     vectors" a, b, c. A 9-vector (concatenated) and a 3-vector (diagonal
 *     entries) are also accepted. Converted to LAMMPS restricted triclinic
 *     per docs.lammps.org/Howto_triclinic.html (see latticeToCell below).
 *   - `Origin="ox oy oz"` shifts the cell origin (default 0 0 0).
 *   - `Properties=species:S:1:pos:R:3` — "a series of triplets, separated by
 *     :, each triplet having the format <name>:<T>:<m>" with T in S, I, R, L.
 *     "If after full parsing the key Properties is missing, the format is
 *     retroactively assumed to be plain xyz" — i.e. species:S:1:pos:R:3.
 *   - `pbc="T T T"` and any other keys are accepted and ignored (there is no
 *     pbc field on TrajectoryFrame; the box itself carries the cell).
 *
 * Every frame block is captured; the FIRST frame defines `atoms`, `bonds`
 * (topology assumed stable across frames — standard for MD output) and the
 * element->type mapping, so colors stay consistent during playback.
 * min/max/center span ALL frames so the camera framing never jumps.
 * Every frame keeps its OWN box (lattices may change per frame, like a
 * breathing NPT cell in a LAMMPS dump).
 *
 * Bonds are inferred with an O(n) spatial-hash pass using covalent radii.
 */

type V3 = [number, number, number];

const isNumericToken = (s: string): boolean =>
  /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s);

/**
 * Parse an extxyz comment line into key=value pairs.
 * Values are double-quoted (with backslash escapes) or bare (up to
 * whitespace). A line without any `key=` form yields an empty map, so plain
 * comments ("Water molecule") behave exactly as before.
 */
export const parseExtxyzComment = (comment: string): Record<string, string> => {
  // keys are extxyz comment keys from the file: a null-prototype map (no prototype keys)
  const meta: Record<string, string> = Object.create(null);
  let i = 0;
  const n = comment.length;
  while (i < n) {
    while (i < n && /\s/.test(comment[i])) i++;
    if (i >= n) break;
    let key = '';
    while (i < n && comment[i] !== '=' && !/\s/.test(comment[i])) key += comment[i++];
    if (i >= n || comment[i] !== '=') continue; // bare word — plain comment fragment
    i++; // consume '='
    let value = '';
    if (comment[i] === '"') {
      i++;
      while (i < n && comment[i] !== '"') {
        if (comment[i] === '\\' && i + 1 < n) {
          value += comment[i + 1]; // backslash escape: \" \\ etc.
          i += 2;
        } else {
          value += comment[i++];
        }
      }
      i++; // closing quote
    } else {
      while (i < n && !/\s/.test(comment[i])) value += comment[i++];
      // Sloppy writers may emit Lattice=/Origin= unquoted despite the spec.
      // Absorb following whitespace-separated numeric tokens (stop at the
      // next `key=` pair) so such files still parse.
      if ((key === 'Lattice' || key === 'Origin') && isNumericToken(value)) {
        while (i < n) {
          const save = i;
          while (i < n && /\s/.test(comment[i])) i++;
          let tok = '';
          while (i < n && !/\s/.test(comment[i])) tok += comment[i++];
          if (tok && !tok.includes('=') && isNumericToken(tok)) {
            value += ` ${tok}`;
          } else {
            i = save;
            break;
          }
        }
      }
    }
    meta[key] = value;
  }
  return meta;
};

/** Column layout resolved from a Properties string. Offsets are 0-based. */
interface ColumnPlan {
  /** First S column named `species` (element symbols), if any. */
  speciesCol?: number;
  /** First I column named `Z` (atomic numbers), if any. */
  atomicNumberCol?: number;
  /** The R:3 column named `pos` — always present (throws otherwise). */
  posCol: number;
}

const DEFAULT_PROPERTIES = 'species:S:1:pos:R:3';

/**
 * Walk `<name>:<T>:<m>` triplets accumulating column offsets. extxyz:
 * "a series of triplets, separated by :, each triplet having the format
 * <name>:<T>:<m>" with T in S, I, R, L; 'Z -> numbers', 'pos -> positions'.
 * Missing Properties = plain xyz = species:S:1:pos:R:3.
 */
export const resolveColumnPlan = (properties?: string): ColumnPlan => {
  const spec =
    properties !== undefined && properties.trim() !== '' ? properties.trim() : DEFAULT_PROPERTIES;
  const parts = spec.split(':');
  let speciesCol: number | undefined;
  let atomicNumberCol: number | undefined;
  let posCol: number | undefined;
  let col = 0;
  for (let i = 0; i + 2 < parts.length; i += 3) {
    const name = parts[i].trim();
    const t = parts[i + 1].trim().toUpperCase();
    const m = parseInt(parts[i + 2], 10);
    if (!['S', 'I', 'R', 'L'].includes(t) || !Number.isInteger(m) || m < 1) break;
    if (name === 'species' && t === 'S' && speciesCol === undefined) speciesCol = col;
    if (name === 'Z' && t === 'I' && atomicNumberCol === undefined) atomicNumberCol = col;
    if (name === 'pos' && t === 'R' && m === 3 && posCol === undefined) posCol = col;
    col += m;
  }
  if (posCol === undefined) {
    throw new Error(
      `Invalid XYZ: Properties "${spec}" has no position column (expected a pos:R:3 triplet)`,
    );
  }
  return { speciesCol, atomicNumberCol, posCol };
};

const parseOrigin = (value?: string): V3 => {
  if (value === undefined) return [0, 0, 0];
  const nums = value.trim().split(/\s+/).map(Number);
  return nums.length === 3 && nums.every(Number.isFinite)
    ? [nums[0], nums[1], nums[2]]
    : [0, 0, 0];
};

/** Lattice value: "3x3 matrix - rows are cell vectors", a 9-vector, or a
 * 3-vector of diagonal entries (libAtoms/extxyz). */
const latticeNumbers = (value: string): number[] | undefined => {
  const nums = value.trim().split(/\s+/).map(Number);
  if ((nums.length === 9 || nums.length === 3) && nums.every(Number.isFinite)) return nums;
  return undefined;
};

/** A cell parsed from a Lattice value: its box plus the rotation that maps
 * file coordinates into the LAMMPS-aligned frame. */
interface CellFrame {
  box: BoxBounds;
  /** Basis vectors (e1 along A, e2 in the AB plane, e3 completes RH triad);
   * undefined when the lattice is already aligned (positions untouched). */
  rotation?: { e1: V3; e2: V3; e3: V3 };
  origin: V3;
}

const cross = (u: V3, v: V3): V3 => [
  u[1] * v[2] - u[2] * v[1],
  u[2] * v[0] - u[0] * v[2],
  u[0] * v[1] - u[1] * v[0],
];

/**
 * Convert extxyz cell vectors to a LAMMPS restricted-triclinic box and the
 * rotation into that frame.
 *
 * docs.lammps.org/Howto_triclinic.html [VERIFIED 2026-10-07], "Transforming
 * to/from the restricted triclinic box": with A along x and B in the
 * xy-plane,
 *   lx = |A| ; xy = B·Â ; ly = sqrt(|B|^2 - xy^2)
 *   xz = C·Â ; yz = (B·C - xy*xz)/ly ; lz = sqrt(|C|^2 - xz^2 - yz^2)
 * which is exactly how LAMMPS defines the restricted triclinic parameters.
 */
const latticeToCell = (nums: number[], origin: V3): CellFrame | undefined => {
  let A: V3, B: V3, C: V3;
  if (nums.length === 9) {
    A = [nums[0], nums[1], nums[2]];
    B = [nums[3], nums[4], nums[5]];
    C = [nums[6], nums[7], nums[8]];
  } else {
    const [a, b, c] = nums;
    A = [a, 0, 0];
    B = [0, b, 0];
    C = [0, 0, c];
  }

  const normA = Math.hypot(A[0], A[1], A[2]);
  const normB = Math.hypot(B[0], B[1], B[2]);
  const normC = Math.hypot(C[0], C[1], C[2]);
  if (!(normA > 0) || !(normB > 0) || !(normC > 0)) return undefined; // degenerate cell

  // Handedness: (A x B)·C < 0 means the triad is left-handed; LAMMPS
  // restricted triclinic cannot represent it (it would need a mirror).
  const triple = cross(A, B).reduce((s, v, k) => s + v * C[k], 0);
  if (triple < -1e-9 * normA * normB * normC) {
    throw new Error('left-handed lattice is not supported');
  }

  // Restricted-triclinic parameters (see doc citation above).
  const lx = normA;
  const e1: V3 = [A[0] / normA, A[1] / normA, A[2] / normA];
  const xy = B[0] * e1[0] + B[1] * e1[1] + B[2] * e1[2];
  const ly = Math.sqrt(Math.max(0, normB * normB - xy * xy));
  const xz = C[0] * e1[0] + C[1] * e1[1] + C[2] * e1[2];
  const yz = ly > 0 ? (B[0] * C[0] + B[1] * C[1] + B[2] * C[2] - xy * xz) / ly : 0;
  const lz = Math.sqrt(Math.max(0, normC * normC - xz * xz - yz * yz));

  // Rotation into the aligned frame: e1 = Â, e2 = normalize(B - (B·e1)e1),
  // e3 = e1 x e2. Positions map to (r·e1, r·e2, r·e3) relative to the origin.
  let e2: V3 = [B[0] - xy * e1[0], B[1] - xy * e1[1], B[2] - xy * e1[2]];
  const e2len = Math.hypot(e2[0], e2[1], e2[2]);
  e2 = e2len > 0 ? [e2[0] / e2len, e2[1] / e2len, e2[2] / e2len] : [0, 1, 0];
  const e3 = cross(e1, e2);

  const isAligned =
    Math.abs(e1[0] - 1) < 1e-12 && Math.abs(e1[1]) < 1e-12 && Math.abs(e1[2]) < 1e-12 &&
    Math.abs(e2[0]) < 1e-12 && Math.abs(e2[1] - 1) < 1e-12 && Math.abs(e2[2]) < 1e-12 &&
    Math.abs(e3[0]) < 1e-12 && Math.abs(e3[1]) < 1e-12 && Math.abs(e3[2] - 1) < 1e-12;

  const box: BoxBounds = {
    xlo: origin[0], xhi: origin[0] + lx,
    ylo: origin[1], yhi: origin[1] + ly,
    zlo: origin[2], zhi: origin[2] + lz,
  };
  // Omit tilt factors that are zero to within fp dust — mirrors the dump
  // parser, where an orthogonal box carries no xy/xz/yz keys at all.
  if (Math.abs(xy) > 1e-9) box.xy = xy;
  if (Math.abs(xz) > 1e-9) box.xz = xz;
  if (Math.abs(yz) > 1e-9) box.yz = yz;

  return {
    box,
    ...(isAligned ? {} : { rotation: { e1, e2, e3 } }),
    origin,
  };
};

/** Map a file-coordinate position into the aligned frame (no-op without a
 * cell or when the lattice is already aligned — bit-identical passthrough). */
const toAlignedFrame = (
  cell: CellFrame | undefined,
  x: number,
  y: number,
  z: number,
): [number, number, number] => {
  if (cell === undefined || cell.rotation === undefined) return [x, y, z];
  const { e1, e2, e3 } = cell.rotation;
  const rx = x - cell.origin[0];
  const ry = y - cell.origin[1];
  const rz = z - cell.origin[2];
  return [
    rx * e1[0] + ry * e1[1] + rz * e1[2] + cell.origin[0],
    rx * e2[0] + ry * e2[1] + rz * e2[2] + cell.origin[1],
    rx * e3[0] + ry * e3[1] + rz * e3[2] + cell.origin[2],
  ];
};

/** Total number of leading tokens a row must carry for the plan to apply. */
const requiredColumns = (plan: ColumnPlan): number =>
  Math.max(
    plan.posCol + 3,
    plan.speciesCol !== undefined ? plan.speciesCol + 1 : 0,
    plan.atomicNumberCol !== undefined ? plan.atomicNumberCol + 1 : 0,
  );

export const parseXYZFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  if (lines.length < 3) {
    throw new Error('Invalid XYZ file: too few lines');
  }

  // Symbol token -> type id, shared across frames for stable coloring.
  // keys are element symbols from the file: a null-prototype map (no prototype keys)
  const elementTypeMap: Record<string, number> = Object.create(null);
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

    // --- extxyz metadata for THIS frame (lattices may change per frame) ---
    const meta = parseExtxyzComment(comment ?? '');
    const plan = resolveColumnPlan(meta.Properties);
    const latNums = meta.Lattice !== undefined ? latticeNumbers(meta.Lattice) : undefined;
    const cell = latNums ? latticeToCell(latNums, parseOrigin(meta.Origin)) : undefined;
    const minCols = requiredColumns(plan);

    const atoms: Atom[] = [];
    while (atoms.length < numAtoms && cursor < lines.length) {
      const line = lines[cursor++].trim();
      if (!line) continue;
      const tokens = line.split(/\s+/);
      if (tokens.length < minCols) continue;
      const fx = parseFloat(tokens[plan.posCol]);
      const fy = parseFloat(tokens[plan.posCol + 1]);
      const fz = parseFloat(tokens[plan.posCol + 2]);
      if (![fx, fy, fz].every(Number.isFinite)) continue;
      const [x, y, z] = toAlignedFrame(cell, fx, fy, fz);

      // Species: first S column named `species`, else the I column named `Z`.
      let mapKey: string;
      let knownNumber: number | undefined;
      if (plan.speciesCol !== undefined) {
        const symbol = tokens[plan.speciesCol];
        mapKey = symbol.trim().toUpperCase();
        knownNumber = getAtomicNumberFromSymbol(symbol);
      } else if (plan.atomicNumberCol !== undefined) {
        const zNum = parseInt(tokens[plan.atomicNumberCol], 10);
        if (Number.isInteger(zNum) && zNum >= 1) {
          mapKey = `Z${zNum}`;
          knownNumber = zNum <= 118 ? zNum : undefined;
        } else {
          mapKey = ''; // malformed Z — fall into the shared synthetic type
        }
      } else {
        mapKey = ''; // Properties carries no species column at all
      }
      if (!(mapKey in elementTypeMap)) {
        elementTypeMap[mapKey] =
          knownNumber !== undefined ? knownNumber : nextSyntheticTypeId++;
      }

      atoms.push({
        id: atoms.length + 1,
        molId: 1,
        type: elementTypeMap[mapKey],
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

    frames.push({
      comment: comment || undefined,
      atoms,
      ...(cell ? { box: cell.box } : {}),
    });
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
    ...(firstFrame.box ? { box: firstFrame.box } : {}),
    ...(frames.length > 1 ? { frames } : {}),
  };
};
