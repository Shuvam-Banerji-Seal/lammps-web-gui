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
 * The first data_ block is visualized. Symmetry operations are NOT expanded;
 * files relying purely on symmetry with an asymmetric unit will render that
 * asymmetric unit only. P1-style files render completely. [documented limit]
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

  const atoms: Atom[] = [];
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  const elementTypeMap: Record<string, number> = {};
  let nextSyntheticTypeId = 1000;

  for (const row of loop.rows) {
    // split respecting simple quotes
    const tokens = row.match(/'[^']*'|"[^"]*"|\S+/g)?.map(t => t.replace(/^['"]|['"]$/g, '')) ?? [];
    if (tokens.length < loop.tags.length) continue;

    let symbol: string | undefined;
    if (iType >= 0) symbol = elementFromTypeSymbol(tokens[iType]);
    if (!symbol && iLabel >= 0) symbol = elementFromLabel(tokens[iLabel]);

    let cx: number, cy: number, cz: number;
    if (useFractional) {
      if (!cell) continue; // fractional coords require a cell
      const fx = parseCifNumber(tokens[iFx]);
      const fy = parseCifNumber(tokens[iFy]);
      const fz = parseCifNumber(tokens[iFz]);
      if (![fx, fy, fz].every(Number.isFinite)) continue;
      ({ x: cx, y: cy, z: cz } = fractionalToCartesian(cell, fx, fy, fz));
    } else {
      cx = parseCifNumber(tokens[iCx]);
      cy = parseCifNumber(tokens[iCy]);
      cz = parseCifNumber(tokens[iCz]);
    }
    if (![cx, cy, cz].every(Number.isFinite)) continue;

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
