import { Atom, Bond, MoleculeData, AtomTypeInfo, BoxBounds } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';

/** Sections whose contents we consume. */
type Section = 'none' | 'masses' | 'atoms' | 'bonds';

/** Declared LAMMPS atom styles we can map to column layouts. */
export type LammpsAtomStyle =
  | 'atomic' | 'charge' | 'molecular' | 'full' | 'auto';

const INT_RE = /^-?\d+$/;
const FLOAT_RE = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

const isInt = (s: string) => INT_RE.test(s);
const isFloat = (s: string) => FLOAT_RE.test(s);

/**
 * Parse one Atoms-section row given a resolved style.
 * Returns null when the row does not fit the style's layout.
 */
const parseAtomRow = (
  tokens: string[],
  style: LammpsAtomStyle
): Omit<Atom, 'id'> & { id: number } | null => {
  const n = tokens.length;
  const t = (i: number) => tokens[i];

  const num = (s: string) => parseFloat(s);

  switch (style) {
    case 'atomic': {
      if (n < 5 || !isInt(t(1))) return null;
      const x = num(t(2)), y = num(t(3)), z = num(t(4));
      if (![x, y, z].every(Number.isFinite)) return null;
      return { id: parseInt(t(0), 10), molId: 1, type: parseInt(t(1), 10), charge: 0, x, y, z };
    }
    case 'charge': {
      if (n < 6 || !isInt(t(1))) return null;
      const x = num(t(3)), y = num(t(4)), z = num(t(5));
      if (![x, y, z].every(Number.isFinite)) return null;
      return { id: parseInt(t(0), 10), molId: 1, type: parseInt(t(1), 10), charge: num(t(2)), x, y, z };
    }
    case 'molecular': {
      if (n < 6 || !isInt(t(1)) || !isInt(t(2))) return null;
      const x = num(t(3)), y = num(t(4)), z = num(t(5));
      if (![x, y, z].every(Number.isFinite)) return null;
      return { id: parseInt(t(0), 10), molId: parseInt(t(1), 10), type: parseInt(t(2), 10), charge: 0, x, y, z };
    }
    case 'full':
    default: {
      if (n < 7 || !isInt(t(1)) || !isInt(t(2))) return null;
      const x = num(t(4)), y = num(t(5)), z = num(t(6));
      if (![x, y, z].every(Number.isFinite)) return null;
      return { id: parseInt(t(0), 10), molId: parseInt(t(1), 10), type: parseInt(t(2), 10), charge: num(t(3)), x, y, z };
    }
  }
};

/**
 * Guess the atom style of a row when the file did not declare one.
 * Column layouts (minimum):
 *   atomic    -> ID type x y z                (5)
 *   molecular -> ID mol type x y z            (6)
 *   charge    -> ID type q x y z              (6)
 *   full      -> ID mol type q x y z          (7)
 */
const guessStyleForRow = (tokens: string[]): LammpsAtomStyle => {
  const n = tokens.length;
  if (n >= 7 && isInt(tokens[1]) && isInt(tokens[2]) &&
      isFloat(tokens[3]) && isFloat(tokens[4]) && isFloat(tokens[5]) && isFloat(tokens[6])) {
    return 'full';
  }
  if (n === 6 || n >= 6) {
    if (isInt(tokens[1]) && isInt(tokens[2]) && isFloat(tokens[3]) && isFloat(tokens[4]) && isFloat(tokens[5])) {
      return 'molecular';
    }
    if (isInt(tokens[1]) && isFloat(tokens[2]) && isFloat(tokens[3]) && isFloat(tokens[4]) && isFloat(tokens[5])) {
      return 'charge';
    }
  }
  if (n >= 5 && isInt(tokens[1]) && isFloat(tokens[2]) && isFloat(tokens[3]) && isFloat(tokens[4])) {
    return 'atomic';
  }
  return 'full';
};

/**
 * Resolve an element for a LAMMPS type from its Masses entry:
 * 1. exact symbol match on the comment ("C", "Cl")
 * 2. exact element-name match on the comment ("Carbon")
 * 3. nearest IUPAC mass within 0.5 amu
 */
const resolveElementFromMass = (
  mass: number, comment?: string
): { symbol: string; label?: string } => {
  if (comment) {
    const c = comment.trim();
    const bySymbol = getAtomicNumberFromSymbol(c);
    if (bySymbol) return { symbol: ELEMENT_DATA[bySymbol - 1].symbol, label: c };
    const byName = ELEMENT_DATA.find(e => e.name.toLowerCase() === c.toLowerCase());
    if (byName) return { symbol: byName.symbol, label: byName.name };
    if (c.length <= 2) return { symbol: c, label: c }; // opaque short label, keep verbatim
  }
  if (mass > 0) {
    let best = ELEMENT_DATA[0];
    let bestDiff = Math.abs(best.mass - mass);
    for (const e of ELEMENT_DATA) {
      const d = Math.abs(e.mass - mass);
      if (d < bestDiff) { bestDiff = d; best = e; }
    }
    if (bestDiff < 0.5) return { symbol: best.symbol };
  }
  return { symbol: 'X' };
};

/**
 * Parses a LAMMPS data file: box bounds (incl. triclinic tilt), Masses,
 * Atoms (styles: atomic/charge/molecular/full), Bonds.
 */
export const parseDataFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];
  const masses: Record<number, { mass: number; comment?: string }> = {};

  let currentSection: Section = 'none';
  let declaredStyle: LammpsAtomStyle | undefined;

  let box: BoxBounds | undefined;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const line = rawLine.trim();
    if (!line) continue;

    const hashIdx = line.indexOf('#');
    const content = (hashIdx >= 0 ? line.slice(0, hashIdx) : line).trim();
    const comment = hashIdx >= 0 ? line.slice(hashIdx + 1).trim() : undefined;
    if (!content) continue;

    // --- Box bounds (header region, before any section) ---
    const boxMatch = content.match(/^([-+\d.eE]+)\s+([-+\d.eE]+)\s+(xlo|xhi|ylo|yhi|zlo|zhi)\s+(xlo|xhi|ylo|yhi|zlo|zhi)$/);
    if (boxMatch) {
      const lo = parseFloat(boxMatch[1]);
      const hi = parseFloat(boxMatch[2]);
      const axis = boxMatch[3];
      if (Number.isFinite(lo) && Number.isFinite(hi)) {
        box = box ?? { xlo: 0, xhi: 0, ylo: 0, yhi: 0, zlo: 0, zhi: 0 };
        if (axis === 'xlo') { box.xlo = lo; box.xhi = hi; }
        else if (axis === 'ylo') { box.ylo = lo; box.yhi = hi; }
        else if (axis === 'zlo') { box.zlo = lo; box.zhi = hi; }
      }
      continue;
    }

    // --- Triclinic tilt factors ---
    const tiltMatch = content.match(/^([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+xy\s+xz\s+yz$/);
    if (tiltMatch) {
      box = box ?? { xlo: 0, xhi: 0, ylo: 0, yhi: 0, zlo: 0, zhi: 0 };
      box.xy = parseFloat(tiltMatch[1]);
      box.xz = parseFloat(tiltMatch[2]);
      box.yz = parseFloat(tiltMatch[3]);
      continue;
    }

    // --- Section headers: any line beginning with a letter ---
    if (/^[A-Za-z]/.test(content)) {
      if (/^Masses\b/i.test(content)) { currentSection = 'masses'; continue; }
      if (/^Atoms?\b/i.test(content)) {
        currentSection = 'atoms';
        const styleComment = comment?.split(/\s+/)[0]?.toLowerCase() as LammpsAtomStyle | undefined;
        declaredStyle =
          styleComment && ['atomic', 'charge', 'molecular', 'full'].includes(styleComment)
            ? styleComment
            : undefined;
        continue;
      }
      if (/^Bonds\b/i.test(content)) { currentSection = 'bonds'; continue; }
      currentSection = 'none'; // Velocities, Angles, coeffs, etc. — skip wholesale
      continue;
    }

    // --- Section content ---
    const tokens = content.split(/\s+/);

    if (currentSection === 'masses') {
      if (tokens.length >= 2 && isInt(tokens[0]) && isFloat(tokens[1])) {
        const id = parseInt(tokens[0], 10);
        masses[id] = { mass: parseFloat(tokens[1]), comment };
      }
    } else if (currentSection === 'atoms') {
      const style = declaredStyle ?? guessStyleForRow(tokens);
      const atom = parseAtomRow(tokens, style);
      if (!atom) {
        // retry remaining styles before giving up on the row
        const alternatives: LammpsAtomStyle[] = ['full', 'charge', 'molecular', 'atomic'];
        for (const alt of alternatives) {
          if (alt === style) continue;
          const retry = parseAtomRow(tokens, alt);
          if (retry) { atoms.push(retry); break; }
        }
        continue;
      }
      atoms.push(atom);
      minX = Math.min(minX, atom.x); maxX = Math.max(maxX, atom.x);
      minY = Math.min(minY, atom.y); maxY = Math.max(maxY, atom.y);
      minZ = Math.min(minZ, atom.z); maxZ = Math.max(maxZ, atom.z);
    } else if (currentSection === 'bonds') {
      if (tokens.length >= 4 && isInt(tokens[0]) && isInt(tokens[1]) && isInt(tokens[2]) && isInt(tokens[3])) {
        bonds.push({
          id: parseInt(tokens[0], 10),
          type: parseInt(tokens[1], 10),
          atom1Id: parseInt(tokens[2], 10),
          atom2Id: parseInt(tokens[3], 10),
        });
      }
    }
  }

  // --- Atom type metadata ---
  const atomTypes: Record<number, AtomTypeInfo> = {};
  const usedTypes = Array.from(new Set(atoms.map(a => a.type)));

  for (const type of usedTypes) {
    let mass = 0;
    let label = `Type ${type}`;
    let element = 'X';

    const m = masses[type];
    if (m) {
      mass = m.mass;
      const resolved = resolveElementFromMass(m.mass, m.comment);
      element = resolved.symbol;
      if (resolved.label) label = resolved.label;
    }

    if (element === 'X') {
      // Type ID doubles as atomic number in many generated files (e.g. type 6 = Carbon)
      if (type >= 1 && type <= 118) {
        element = ELEMENT_DATA[type - 1].symbol;
        if (label === `Type ${type}`) label = `${ELEMENT_DATA[type - 1].name} (Type ${type})`;
      }
    } else if (label === `Type ${type}`) {
      const meta = ELEMENT_DATA.find(e => e.symbol === element);
      if (meta) label = `${meta.name} (${meta.symbol})`;
    }

    let count = 0;
    for (const a of atoms) if (a.type === type) count++;

    atomTypes[type] = { id: type, mass, element, label, count };
  }

  // --- Centering: prefer simulation box center when available ---
  const safeCenter = (() => {
    if (box) {
      return {
        x: (box.xlo + box.xhi) / 2,
        y: (box.ylo + box.yhi) / 2,
        z: (box.zlo + box.zhi) / 2,
      };
    }
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
