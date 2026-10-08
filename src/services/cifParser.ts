import { Atom, Bond, MoleculeData, AtomTypeInfo, BoxBounds } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';
import { inferBonds } from './bondInference';

/**
 * Parses CIF (Crystallographic Information Framework) files — the standard
 * format for crystal structures from databases such as the COD, ICSD exports,
 * and materials-project style output.
 *
 * Supported subset (covers the overwhelming majority of structural CIFs):
 *   - _cell_length_a/b/c, _cell_angle_alpha/beta/gamma
 *   - loop_ _atom_site_* with fractional (fract_x/y/z) or Cartesian
 *     (Cartn_x/y/z) coordinates
 *   - element from _atom_site_type_symbol (charges like "Fe3+" tolerated)
 *     falling back to the alphabetic prefix of _atom_site_label ("Cl2" -> Cl)
 *
 * The first data_ block is visualized. If the file carries a symmetry
 * operation loop (`_space_group_symop_operation_xyz` or the older
 * `_symmetry_equiv_pos_as_xyz`), the listed asymmetric unit is expanded:
 * every operation (rotation R + translation t parsed from its "x,y,z" string)
 * is applied to every site, each generated position is wrapped into [0,1)
 * and deduplicated against all accepted positions, so special positions,
 * lattice centring and duplicate disordered sites collapse to one atom each.
 * Files without such a loop are treated as P1 and render exactly the listed
 * sites; Cartesian (`_atom_site_Cartn_*`) files are never expanded.
 */

export interface CifCell {
  a: number; b: number; c: number;
  alphaDeg: number; betaDeg: number; gammaDeg: number;
}

const DEG2RAD = Math.PI / 180;

/** Strip value uncertainty notation: "10.5000(3)" -> 10.5 */
const parseCifNumber = (raw: string): number => parseFloat(raw.replace(/\(\d+\)\s*$/, '').trim());

/** Extract a scalar datum `tag value` (value may also be quoted). */
const findTag = (lines: string[], tag: string): string | undefined => {
  const re = new RegExp(`^${tag}\\s+(.+)$`, 'i');
  for (const line of lines) {
    const m = line.trim().match(re);
    if (m) return m[1].replace(/^['"]|['"]$/g, '').trim();
  }
  return undefined;
};

/**
 * Locate the first `loop_` whose header contains any of the wanted tags.
 * Returns column tag list and the row lines following the header.
 */
export const findLoopBlock = (
  lines: string[],
  requiredTagPrefixes: string[]
): { tags: string[]; rows: string[] } | null => {
  for (let i = 0; i < lines.length; i++) {
    if (!/^loop_\s*$/i.test(lines[i].trim())) continue;

    const tags: string[] = [];
    let j = i + 1;
    while (j < lines.length && /^_\S+/.test(lines[j].trim())) {
      tags.push(lines[j].trim().split(/\s+/)[0].toLowerCase());
      j++;
    }
    if (!tags.some(t => requiredTagPrefixes.some(p => t.startsWith(p)))) continue;

    const rows: string[] = [];
    while (j < lines.length) {
      const t = lines[j].trim();
      if (!t) { j++; continue; }                    // blank lines inside rows are legal separators
      if (/^loop_/i.test(t) || /^_\S/.test(t) || /^data_/i.test(t) || /^#/ .test(t)) break;
      rows.push(t);
      j++;
    }
    return { tags, rows };
  }
  return null;
};

/** Element symbol from a type_symbol token like "O2-", "Fe3+", "Cl". */
export const elementFromTypeSymbol = (token: string): string | undefined => {
  const letters = token.replace(/[^A-Za-z]/g, '');
  if (!letters) return undefined;
  const norm = letters[0].toUpperCase() + letters.slice(1).toLowerCase();
  return getAtomicNumberFromSymbol(norm) !== undefined ? norm : undefined;
};

/** Element symbol from an atom-site label like "C1", "Cl2", "OW32". */
export const elementFromLabel = (label: string): string | undefined => {
  const letters = label.replace(/[^A-Za-z]/g, '');
  if (!letters) return undefined;
  if (letters.length >= 2) {
    const two = letters[0].toUpperCase() + letters[1].toLowerCase();
    if (getAtomicNumberFromSymbol(two) !== undefined) return two;
  }
  const one = letters[0].toUpperCase();
  return getAtomicNumberFromSymbol(one) !== undefined ? one : undefined;
};

/** Split a CIF data row into values, honouring single/double quotes. */
const tokenizeCifRow = (row: string): string[] =>
  row.match(/'[^']*'|"[^"]*"|\S+/g)?.map(t => t.replace(/^['"]|['"]$/g, '')) ?? [];

/**
 * A crystallographic symmetry operation in fractional coordinates:
 *   x' = R·x + t
 * with an integer rotation matrix R (R[row][col], rows are output x,y/z,
 * columns are input x/y/z) and a rational translation vector t.
 */
export interface SymmetryOperation {
  r: number[][];
  t: [number, number, number];
}

/**
 * Parse a symmetry operation string as found in `_space_group_symop_operation_xyz`
 * / `_symmetry_equiv_pos_as_xyz` loops, e.g. "-x+1/2,y,-z", "1/2+z,y,1/2-x",
 * "x-y,x,z+1/6". Components are signed sums of the variables x/y/z
 * (case-insensitive) and constants written as fractions (1/2, 2/3, ...) or
 * decimals (0.5); term order is arbitrary and whitespace/quotes are ignored.
 * Throws naming the operation when a component cannot be parsed.
 */
export const parseSymmetryOperation = (opRaw: string): SymmetryOperation => {
  const op = opRaw.trim();
  const components = op.split(',').map(c => c.trim().toLowerCase());
  if (components.length !== 3 || components.some(c => c === '')) {
    throw new Error(
      `Invalid CIF symmetry operation "${opRaw}": expected three comma-separated components`
    );
  }

  const r: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const t: [number, number, number] = [0, 0, 0];

  components.forEach((component, row) => {
    const compact = component.replace(/\s+/g, '');
    // Split into signed terms: a new term starts at each + / - after the first
    // character ("1/2-x" -> ["1/2", "-x"], "x-y" -> ["x", "-y"], "+y" -> ["+y"]).
    const terms: string[] = [];
    let start = 0;
    for (let i = 1; i < compact.length; i++) {
      if (compact[i] === '+' || compact[i] === '-') {
        terms.push(compact.slice(start, i));
        start = i;
      }
    }
    terms.push(compact.slice(start));

    for (const term of terms) {
      const varMatch = term.match(/^([+-]?)([xyz])$/);
      if (varMatch) {
        const col = 'xyz'.indexOf(varMatch[2]);
        r[row][col] += varMatch[1] === '-' ? -1 : 1;
        continue;
      }
      const numMatch = term.match(/^([+-]?)(\d+(?:\.\d+)?|\.\d+)(?:\/(\d+(?:\.\d+)?|\.\d+))?$/);
      if (numMatch) {
        const denominator = numMatch[3] !== undefined ? parseFloat(numMatch[3]) : 1;
        if (denominator === 0) {
          throw new Error(
            `Invalid CIF symmetry operation "${opRaw}": cannot parse component "${component}"`
          );
        }
        const value = parseFloat(numMatch[2]) / denominator;
        t[row] += numMatch[1] === '-' ? -value : value;
        continue;
      }
      throw new Error(
        `Invalid CIF symmetry operation "${opRaw}": cannot parse component "${component}"`
      );
    }
  });

  return { r, t };
};

/** Wrap a fractional coordinate component into [0, 1), snapping ~1 to 0. */
const wrap01 = (v: number): number => {
  const w = v - Math.floor(v);
  return 1 - w < 1e-6 ? 0 : w;
};

/**
 * Fractional -> Cartesian using the standard crystallographic convention:
 *   a along +x, b in the xy plane, c completing right-handed frame.
 */
export const fractionalToCartesian = (
  cell: CifCell,
  fx: number, fy: number, fz: number
): { x: number; y: number; z: number } => {
  const { a, b, c } = cell;
  const ca = Math.cos(cell.alphaDeg * DEG2RAD);
  const cb = Math.cos(cell.betaDeg * DEG2RAD);
  const cg = Math.cos(cell.gammaDeg * DEG2RAD);
  const sg = Math.sin(cell.gammaDeg * DEG2RAD);

  // volume factor for the c vector's z component
  const czFactor = Math.sqrt(Math.max(0, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)) / sg;

  // cell vectors as rows
  const ax = a, ay = 0, az = 0;
  const bx = b * cg, by = b * sg, bz = 0;
  const cx = c * cb, cy = c * (ca - cb * cg) / sg, cz = c * czFactor;

  return {
    x: fx * ax + fy * bx + fz * cx,
    y: fx * ay + fy * by + fz * cy,
    z: fx * az + fy * bz + fz * cz,
  };
};

/**
 * Convert a CIF cell to the LAMMPS-convention BoxBounds used by the renderer.
 *
 * LAMMPS maps an arbitrary triclinic cell to:
 *   lx = a,            xy = b*cos(gamma)
 *   ly = b*sin(gamma), xz = c*cos(beta)
 *   yz = c*(cos(alpha) - cos(beta)*cos(gamma)) / sin(gamma)
 *   lz = c*sqrt(1 - ca^2 - cb^2 - cg^2 + 2*ca*cb*cg) / sin(gamma)
 *
 * The renderer reconstructs the 8 corners from these six numbers, so
 * triclinic cells render as true parallelepipeds, never as fake boxes.
 */
export const cifCellToBoxBounds = (cell: CifCell): BoxBounds => {
  const ca = Math.cos(cell.alphaDeg * DEG2RAD);
  const cb = Math.cos(cell.betaDeg * DEG2RAD);
  const cg = Math.cos(cell.gammaDeg * DEG2RAD);
  const sg = Math.sin(cell.gammaDeg * DEG2RAD);

  const lz = cell.c * Math.sqrt(Math.max(0, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)) / sg;

  return {
    xlo: 0,
    xhi: cell.a,
    ylo: 0,
    yhi: cell.b * sg,
    zlo: 0,
    zhi: lz,
    xy: cell.b * cg,
    xz: cell.c * cb,
    yz: cell.c * (ca - cb * cg) / sg,
  };
};

/**
 * Exact cell corner vectors (Å) for triclinic rendering:
 * origin O, A = a_vec, B = b_vec, C = c_vec.
 */
export const cifCellVectors = (cell: CifCell) => {
  const p = fractionalToCartesian(cell, 1, 0, 0);
  const q = fractionalToCartesian(cell, 0, 1, 0);
  const r = fractionalToCartesian(cell, 0, 0, 1);
  return { A: p, B: q, C: r };
};

export const parseCIFFile = (data: string): MoleculeData => {
  const lines = data.split('\n');

  // --- Cell parameters (first block only) ---
  const num = (tag: string): number | undefined => {
    const v = findTag(lines, tag);
    if (v === undefined) return undefined;
    const n = parseCifNumber(v);
    return Number.isFinite(n) ? n : undefined;
  };

  let cell: CifCell | undefined;
  const a = num('_cell_length_a');
  const b = num('_cell_length_b');
  const c = num('_cell_length_c');
  if (a && b && c) {
    cell = {
      a, b, c,
      alphaDeg: num('_cell_angle_alpha') ?? 90,
      betaDeg: num('_cell_angle_beta') ?? 90,
      gammaDeg: num('_cell_angle_gamma') ?? 90,
    };
  }

  // --- Atom site loop ---
  const loop = findLoopBlock(lines, ['_atom_site_fract_', '_atom_site_cartn_', '_atom_site_label']);
  if (!loop) throw new Error('Invalid CIF file: no _atom_site loop found');

  const colIndex = (name: string) => loop.tags.indexOf(name.toLowerCase());
  const iLabel = colIndex('_atom_site_label');
  const iType = colIndex('_atom_site_type_symbol');
  const iFx = colIndex('_atom_site_fract_x');
  const iFy = colIndex('_atom_site_fract_y');
  const iFz = colIndex('_atom_site_fract_z');
  const iCx = colIndex('_atom_site_cartn_x');
  const iCy = colIndex('_atom_site_cartn_y');
  const iCz = colIndex('_atom_site_cartn_z');

  const useFractional = iFx >= 0 && iFy >= 0 && iFz >= 0;
  const useCartesian = iCx >= 0 && iCy >= 0 && iCz >= 0;
  if (!useFractional && !useCartesian) {
    throw new Error('Invalid CIF file: atom sites lack fract_* and Cartn_* coordinates');
  }
  if (useFractional && !cell) {
    throw new Error(
      'Invalid CIF file: fractional coordinates require _cell_length_* and _cell_angle_* parameters'
    );
  }

  // --- Symmetry operations loop (optional; its absence means P1) ---
  // Read from a loop containing `_space_group_symop_operation_xyz` (modern
  // tag) or `_symmetry_equiv_pos_as_xyz` (legacy tag); other columns such as
  // `_space_group_symop_id` / `_symmetry_equiv_pos_site_id` may precede it,
  // so the xyz column is picked by its header name.
  const symLoop = findLoopBlock(lines, [
    '_space_group_symop_operation_xyz',
    '_symmetry_equiv_pos_as_xyz',
  ]);
  const symOps: SymmetryOperation[] = [];
  if (symLoop) {
    const opCol =
      symLoop.tags.indexOf('_space_group_symop_operation_xyz') >= 0
        ? symLoop.tags.indexOf('_space_group_symop_operation_xyz')
        : symLoop.tags.indexOf('_symmetry_equiv_pos_as_xyz');
    if (opCol >= 0) {
      for (const row of symLoop.rows) {
        const raw = tokenizeCifRow(row)[opCol];
        if (raw === undefined || raw === '.' || raw === '?') continue; // inapplicable value
        symOps.push(parseSymmetryOperation(raw));
      }
    }
  }
  const expandSymmetry = useFractional && symOps.length > 0;

  // --- Deduplication of generated positions (fractional space) ---
  // O(n): positions are hashed into buckets quantized to 0.01 fractional
  // units; a candidate only needs its 3x3x3 neighbour buckets, since two
  // positions whose minimum-image difference is < FRAC_TOL (1e-3) per
  // component can never land in buckets further apart than one step
  // (mod 100, so the 0/1 wrap is covered too).
  const FRAC_TOL = 1e-3;
  const BUCKET_STEPS = 100; // 1 / 0.01
  const expandedFrac: Array<[number, number, number]> = [];
  const fracBuckets = new Map<string, number[]>();
  const bucketOf = (f: [number, number, number]): [number, number, number] => {
    const q = (v: number) => ((Math.round(v * BUCKET_STEPS)) % BUCKET_STEPS + BUCKET_STEPS) % BUCKET_STEPS;
    return [q(f[0]), q(f[1]), q(f[2])];
  };
  const isDuplicatePosition = (f: [number, number, number]): boolean => {
    const [q0, q1, q2] = bucketOf(f);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const key =
            `${((q0 + dx) % BUCKET_STEPS + BUCKET_STEPS) % BUCKET_STEPS},` +
            `${((q1 + dy) % BUCKET_STEPS + BUCKET_STEPS) % BUCKET_STEPS},` +
            `${((q2 + dz) % BUCKET_STEPS + BUCKET_STEPS) % BUCKET_STEPS}`;
          const hits = fracBuckets.get(key);
          if (!hits) continue;
          for (const idx of hits) {
            const g = expandedFrac[idx];
            const d0 = f[0] - g[0] - Math.round(f[0] - g[0]);
            const d1 = f[1] - g[1] - Math.round(f[1] - g[1]);
            const d2 = f[2] - g[2] - Math.round(f[2] - g[2]);
            if (Math.abs(d0) < FRAC_TOL && Math.abs(d1) < FRAC_TOL && Math.abs(d2) < FRAC_TOL) {
              return true;
            }
          }
        }
      }
    }
    return false;
  };
  const rememberPosition = (f: [number, number, number]): void => {
    const key = bucketOf(f).join(',');
    const hits = fracBuckets.get(key);
    if (hits) hits.push(expandedFrac.push(f) - 1);
    else fracBuckets.set(key, [expandedFrac.push(f) - 1]);
  };

  const atoms: Atom[] = [];
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  // keys are element symbols from the file: a null-prototype map (no prototype keys)
  const elementTypeMap: Record<string, number> = Object.create(null);
  let nextSyntheticTypeId = 1000;

  const addAtom = (symbol: string | undefined, cx: number, cy: number, cz: number): void => {
    const lookupKey = (symbol ?? 'X').toUpperCase();
    if (!(lookupKey in elementTypeMap)) {
      const atomicNumber = symbol ? getAtomicNumberFromSymbol(symbol) : undefined;
      elementTypeMap[lookupKey] =
        atomicNumber !== undefined ? atomicNumber : nextSyntheticTypeId++;
    }
    const type = elementTypeMap[lookupKey];

    atoms.push({ id: atoms.length + 1, molId: 1, type, charge: 0, x: cx, y: cy, z: cz });

    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
    minZ = Math.min(minZ, cz); maxZ = Math.max(maxZ, cz);
  };

  for (const row of loop.rows) {
    // split respecting simple quotes
    const tokens = tokenizeCifRow(row);
    if (tokens.length < loop.tags.length) continue;

    let symbol: string | undefined;
    if (iType >= 0) symbol = elementFromTypeSymbol(tokens[iType]);
    if (!symbol && iLabel >= 0) symbol = elementFromLabel(tokens[iLabel]);

    if (useFractional) {
      if (!cell) continue; // fractional coords require a cell
      const fx = parseCifNumber(tokens[iFx]);
      const fy = parseCifNumber(tokens[iFy]);
      const fz = parseCifNumber(tokens[iFz]);
      if (![fx, fy, fz].every(Number.isFinite)) continue;

      if (expandSymmetry) {
        // Generate R·f + t for every operation, wrap into [0,1), dedup.
        for (const op of symOps) {
          const gf: [number, number, number] = [
            wrap01(op.r[0][0] * fx + op.r[0][1] * fy + op.r[0][2] * fz + op.t[0]),
            wrap01(op.r[1][0] * fx + op.r[1][1] * fy + op.r[1][2] * fz + op.t[1]),
            wrap01(op.r[2][0] * fx + op.r[2][1] * fy + op.r[2][2] * fz + op.t[2]),
          ];
          if (isDuplicatePosition(gf)) continue;
          rememberPosition(gf);
          const cart = fractionalToCartesian(cell, gf[0], gf[1], gf[2]);
          addAtom(symbol, cart.x, cart.y, cart.z);
        }
      } else {
        const cart = fractionalToCartesian(cell, fx, fy, fz);
        addAtom(symbol, cart.x, cart.y, cart.z);
      }
    } else {
      const cx = parseCifNumber(tokens[iCx]);
      const cy = parseCifNumber(tokens[iCy]);
      const cz = parseCifNumber(tokens[iCz]);
      if (![cx, cy, cz].every(Number.isFinite)) continue;
      addAtom(symbol, cx, cy, cz);
    }
  }

  if (atoms.length === 0) throw new Error('Invalid CIF file: zero atom positions parsed');

  // --- Type metadata ---
  const atomTypes: Record<number, AtomTypeInfo> = {};
  const usedTypes = Array.from(new Set(atoms.map(at => at.type)));
  for (const type of usedTypes) {
    const elem = ELEMENT_DATA.find(e => e.number === type);
    let count = 0;
    for (const at of atoms) if (at.type === type) count++;
    atomTypes[type] = {
      id: type,
      mass: elem?.mass ?? 0,
      element: elem?.symbol ?? 'X',
      label: elem ? `${elem.name} (${elem.symbol})` : `Type ${type}`,
      count,
    };
  }

  // --- Bonds (molecular CIFs only make sense; capped for huge crystals) ---
  const bonds: Bond[] = atoms.length <= 30000 ? inferBonds(atoms) : [];

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
    ...(cell ? { box: cifCellToBoxBounds(cell) } : {}),
  };
};
