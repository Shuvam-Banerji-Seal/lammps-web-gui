import { Atom, Bond, MoleculeData, AtomTypeInfo, BoxBounds } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';

/** Sections whose contents we consume. */
type Section = 'none' | 'masses' | 'labels' | 'atoms' | 'bonds';

/**
 * Declared LAMMPS atom styles we can map to column layouts.
 *
 * Every name is an atom_style accepted by read_data
 * (docs.lammps.org/read_data.html, verified 2026-10-07): "atom-style =
 * angle or atomic or body or bond or bpm/sphere or charge or dielectric or
 * dipole or dpd or edpd or electron or ellipsoid or full or line or mdpd or
 * molecular or peri or rheo or sphere or spin or template or tri or hybrid".
 * 'auto' is kept for backward compatibility of the exported union.
 */
export type LammpsAtomStyle =
  | 'angle' | 'atomic' | 'body' | 'bond' | 'bpm/sphere' | 'charge'
  | 'dielectric' | 'dipole' | 'dpd' | 'edpd' | 'electron' | 'ellipsoid'
  | 'full' | 'line' | 'mdpd' | 'molecular' | 'peri' | 'rheo' | 'sphere'
  | 'spin' | 'template' | 'tri' | 'hybrid' | 'auto';

/** A hint style is any of the documented styles (everything but 'auto'). */
type HintStyle = Exclude<LammpsAtomStyle, 'auto'>;

/**
 * Column layout of one atom style: column indices of the atom-type, the
 * molecule-ID (null when the style has none), the charge (null when the
 * style has none) and of x. y = x+1 and z = x+2 for every style.
 */
interface StyleLayout {
  type: number;
  mol: number | null;
  q: number | null;
  x: number;
  /**
   * Number of documented columns AFTER z (e.g. spin's "spx spy spz sp").
   * Image flags, when present, follow those.
   */
  trailing?: number;
  /** hybrid rows continue with sub-style values after z, so their length is open-ended */
  openEnded?: boolean;
}

/**
 * Style -> column layout, transcribed from the per-style "Atoms" line
 * formats on docs.lammps.org/read_data.html (verified 2026-10-07):
 *   angle/bond/molecular = atom-ID molecule-ID atom-type x y z
 *   atomic = atom-ID atom-type x y z
 *   body = atom-ID atom-type bodyflag mass x y z
 *   bpm/sphere = atom-ID molecule-ID atom-type diameter density x y z
 *   charge = atom-ID atom-type q x y z
 *   dielectric = atom-ID atom-type q x y z mux muy muz area ed em epsilon curvature
 *   dipole = atom-ID atom-type q x y z mux muy muz
 *   dpd = atom-ID atom-type theta x y z
 *   edpd = atom-ID atom-type edpd_temp edpd_cv x y z
 *   electron = atom-ID atom-type q espin eradius x y z
 *   ellipsoid = atom-ID atom-type ellipsoidflag density x y z
 *   full = atom-ID molecule-ID atom-type q x y z
 *   line = atom-ID molecule-ID atom-type lineflag density x y z
 *   mdpd = atom-ID atom-type rho x y z
 *   peri = atom-ID atom-type volume density x y z
 *   rheo = atom-ID atom-type status rho x y z
 *   sphere = atom-ID atom-type diameter density x y z
 *   spin = atom-ID atom-type x y z spx spy spz sp
 *   template = atom-ID atom-type molecule-ID template-index template-atom x y z
 *   tri = atom-ID molecule-ID atom-type triangleflag density x y z
 *   hybrid = atom-ID atom-type x y z sub-style-values...
 *
 * docs.lammps.org/read_data.html on image flags: "atom lines (all lines or
 * none of them) can optionally list 3 trailing integer values (nx,ny,nz),
 * which are used to initialize the atom's image flags". On type labels:
 * "atom-type = type of atom (1-Ntype, or type label)".
 */
const STYLE_LAYOUTS: Record<HintStyle, StyleLayout> = {
  angle:        { type: 2, mol: 1,    q: null, x: 3 },
  atomic:       { type: 1, mol: null, q: null, x: 2 },
  body:         { type: 1, mol: null, q: null, x: 4 },
  bond:         { type: 2, mol: 1,    q: null, x: 3 },
  'bpm/sphere': { type: 2, mol: 1,    q: null, x: 5 },
  charge:       { type: 1, mol: null, q: 2,    x: 3 },
  dielectric:   { type: 1, mol: null, q: 2,    x: 3, trailing: 8 },
  dipole:       { type: 1, mol: null, q: 2,    x: 3, trailing: 3 },
  dpd:          { type: 1, mol: null, q: null, x: 3 },
  edpd:         { type: 1, mol: null, q: null, x: 4 },
  electron:     { type: 1, mol: null, q: 2,    x: 5 },
  ellipsoid:    { type: 1, mol: null, q: null, x: 4 },
  full:         { type: 2, mol: 1,    q: 3,    x: 4 },
  line:         { type: 2, mol: 1,    q: null, x: 5 },
  mdpd:         { type: 1, mol: null, q: null, x: 3 },
  molecular:    { type: 2, mol: 1,    q: null, x: 3 },
  peri:         { type: 1, mol: null, q: null, x: 4 },
  rheo:         { type: 1, mol: null, q: null, x: 4 },
  sphere:       { type: 1, mol: null, q: null, x: 4 },
  spin:         { type: 1, mol: null, q: null, x: 2, trailing: 4 },
  template:     { type: 1, mol: 2,    q: null, x: 5 },
  tri:          { type: 2, mol: 1,    q: null, x: 5 },
  hybrid:       { type: 1, mol: null, q: null, x: 2, openEnded: true },
};

const INT_RE = /^-?\d+$/;
const FLOAT_RE = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

const isInt = (s: string) => INT_RE.test(s);
const isFloat = (s: string) => FLOAT_RE.test(s);

/**
 * Parse one Atoms-section row given a column layout.
 * Returns null when the row does not fit the layout — it is then skipped,
 * never re-read under another layout.
 */
const parseAtomRow = (
  tokens: string[],
  layout: StyleLayout,
  ntypes: number | null,
  typeLabels: Map<string, number>
): (Omit<Atom, 'id'> & { id: number }) | null => {
  const n = tokens.length;
  const base = layout.x + 3 + (layout.trailing ?? 0); // tokens up to and including z + documented trailing columns

  let ix: number | undefined, iy: number | undefined, iz: number | undefined;
  if (layout.openEnded) {
    // hybrid: id type x y z sub-style-values... — length is open-ended and
    // trailing image flags cannot be distinguished from sub-style values.
    if (n < base) return null;
  } else if (n === base + 3) {
    // trailing nx ny nz image flags (docs: "3 trailing integer values")
    if (!isInt(tokens[n - 3]) || !isInt(tokens[n - 2]) || !isInt(tokens[n - 1])) return null;
    ix = parseInt(tokens[n - 3], 10);
    iy = parseInt(tokens[n - 2], 10);
    iz = parseInt(tokens[n - 1], 10);
  } else if (n !== base) {
    return null;
  }

  // atom-ID is an integer
  if (!isInt(tokens[0])) return null;

  // atom-type: "1-Ntype, or type label"
  let type: number;
  if (isInt(tokens[layout.type])) {
    type = parseInt(tokens[layout.type], 10);
    if (type < 1 || (ntypes !== null && type > ntypes)) return null;
  } else if (typeLabels.has(tokens[layout.type])) {
    type = typeLabels.get(tokens[layout.type])!;
  } else {
    return null;
  }

  let molId = 1;
  if (layout.mol !== null) {
    if (!isInt(tokens[layout.mol])) return null;
    molId = parseInt(tokens[layout.mol], 10);
  }

  let charge = 0;
  if (layout.q !== null) {
    charge = parseFloat(tokens[layout.q]);
    if (!Number.isFinite(charge)) return null;
  }

  const x = parseFloat(tokens[layout.x]);
  const y = parseFloat(tokens[layout.x + 1]);
  const z = parseFloat(tokens[layout.x + 2]);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;

  const atom: Omit<Atom, 'id'> & { id: number } = { id: parseInt(tokens[0], 10), molId, type, charge, x, y, z };
  if (ix !== undefined) { atom.ix = ix; atom.iy = iy; atom.iz = iz; }
  return atom;
};

/**
 * Candidate layouts for a section whose Atoms header carried no recognised
 * style hint, in priority order. One layout is chosen for the WHOLE section.
 */
const AUTO_CANDIDATES: HintStyle[] = ['full', 'molecular', 'charge', 'atomic'];

/**
 * Choose ONE layout for the whole Atoms section: the first candidate that
 * fits every row; otherwise the candidate that fits the most rows.
 */
const chooseSectionLayout = (
  rows: string[][],
  ntypes: number | null,
  typeLabels: Map<string, number>
): HintStyle => {
  let best = AUTO_CANDIDATES[0];
  let bestFit = -1;
  for (const cand of AUTO_CANDIDATES) {
    const layout = STYLE_LAYOUTS[cand];
    let fit = 0;
    for (const tokens of rows) {
      if (parseAtomRow(tokens, layout, ntypes, typeLabels)) fit++;
    }
    if (fit === rows.length) return cand;
    if (fit > bestFit) { bestFit = fit; best = cand; }
  }
  return best;
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
 * Parses a LAMMPS data file: box bounds (incl. triclinic tilt), Atom Type
 * Labels, Masses, Atoms (every documented atom style, with or without the
 * style hint, with or without trailing image flags), Bonds.
 */
export const parseDataFile = (data: string): MoleculeData => {
  const lines = data.split('\n');
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];
  const masses: Record<number, { mass: number; comment?: string }> = {};
  // Masses rows whose first column is a type label, merged once labels are known
  const labelMasses: Record<string, { mass: number; comment?: string }> = {};
  // "Atom Type Labels" section: label -> numeric type
  const typeLabels = new Map<string, number>();

  let currentSection: Section = 'none';
  let atomLayoutHint: HintStyle | undefined;
  let ntypes: number | null = null; // from the header line "N atom types"

  let box: BoxBounds | undefined;

  // Rows of the current Atoms section, parsed once a single layout for the
  // whole section has been chosen (at the next section header or at EOF).
  const pendingAtomRows: string[][] = [];

  const finalizeAtoms = () => {
    if (pendingAtomRows.length === 0) return;
    const style = atomLayoutHint ?? chooseSectionLayout(pendingAtomRows, ntypes, typeLabels);
    const layout = STYLE_LAYOUTS[style];
    // "N atom types" helps CHOOSE a layout above, but is not enforced on the
    // rows: generated files often declare 1 type and use the atomic number
    // as the type (the bundled C60 example: "1 atom types", type 6), and a
    // viewer should show them rather than drop every atom.
    for (const tokens of pendingAtomRows) {
      const atom = parseAtomRow(tokens, layout, null, typeLabels);
      if (atom) atoms.push(atom); // rows that do not fit are skipped
    }
    pendingAtomRows.length = 0;
  };

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

    // --- Header line "N atom types" (needed to validate atom-type columns) ---
    if (currentSection === 'none') {
      const ntypesMatch = content.match(/^(\d+)\s+atom\s+types\b/i);
      if (ntypesMatch) {
        ntypes = parseInt(ntypesMatch[1], 10);
        continue;
      }
    }

    // --- Section headers: any line beginning with a letter ---
    // EXCEPT a Masses row whose first column is a type label ("O 15.999"):
    // it also starts with a letter, but its "word number" shape distinguishes
    // it from any section header.
    const massesLabelRow =
      currentSection === 'masses' && /^[A-Za-z]\S*\s+[-+\d.eE]+$/.test(content);
    if (/^[A-Za-z]/.test(content) && !massesLabelRow) {
      finalizeAtoms(); // close out a preceding Atoms section, if any
      if (/^Masses\b/i.test(content)) { currentSection = 'masses'; continue; }
      // Must be tested before "Atoms": "Atom Type Labels" is its own section
      if (/^Atom Type Labels\b/i.test(content)) { currentSection = 'labels'; continue; }
      if (/^Atoms\b/i.test(content)) {
        currentSection = 'atoms';
        // The style hint is the first word after '#' on the Atoms header
        const hint = comment?.split(/\s+/)[0]?.toLowerCase();
        atomLayoutHint = hint && hint in STYLE_LAYOUTS ? (hint as HintStyle) : undefined;
        continue;
      }
      if (/^Bonds\b/i.test(content)) { currentSection = 'bonds'; continue; }
      currentSection = 'none'; // Velocities, Angles, coeffs, etc. — skip wholesale
      continue;
    }

    // --- Section content ---
    const tokens = content.split(/\s+/);

    if (currentSection === 'masses') {
      if (tokens.length >= 2 && isFloat(tokens[1])) {
        const entry = { mass: parseFloat(tokens[1]), comment };
        if (isInt(tokens[0])) {
          masses[parseInt(tokens[0], 10)] = entry;
        } else {
          // first column may be a type label ("C 12.011")
          labelMasses[tokens[0]] = entry;
        }
      }
    } else if (currentSection === 'labels') {
      // "Atom Type Labels" rows: "N label"
      if (tokens.length >= 2 && isInt(tokens[0]) && tokens[1]) {
        typeLabels.set(tokens[1], parseInt(tokens[0], 10));
      }
    } else if (currentSection === 'atoms') {
      pendingAtomRows.push(tokens);
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
  finalizeAtoms(); // Atoms section running to end of file

  // Merge label-keyed Masses rows now that every label is known
  for (const label of Object.keys(labelMasses)) {
    const t = typeLabels.get(label);
    if (t !== undefined && masses[t] === undefined) masses[t] = labelMasses[label];
  }

  // --- Extents from the final atoms array (every accepted atom counts) ---
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const a of atoms) {
    if (a.x < minX) minX = a.x; if (a.x > maxX) maxX = a.x;
    if (a.y < minY) minY = a.y; if (a.y > maxY) maxY = a.y;
    if (a.z < minZ) minZ = a.z; if (a.z > maxZ) maxZ = a.z;
  }

  // --- Atom type metadata ---
  const atomTypes: Record<number, AtomTypeInfo> = {};
  const usedTypes = Array.from(new Set(atoms.map(a => a.type)));
  // inverse of the label map, for element resolution
  const labelByType = new Map<number, string>();
  for (const [label, type] of typeLabels) labelByType.set(type, label);

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
      // A type label that is an element symbol names this type's element
      // (docs.lammps.org/read_data.html: "type label ... e.g. the LAMMPS
      // input can use type labels to refer to atom types")
      const typeLabel = labelByType.get(type);
      if (typeLabel) {
        const bySymbol = getAtomicNumberFromSymbol(typeLabel);
        if (bySymbol) element = ELEMENT_DATA[bySymbol - 1].symbol;
      }
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
