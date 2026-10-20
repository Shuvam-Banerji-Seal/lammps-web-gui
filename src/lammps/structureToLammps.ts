/**
 * Converts a parsed structure (PDB, CIF, XYZ, or a LAMMPS data file) into LAMMPS input
 * files: a read_data file, or a molecule template for the molecule command. Pure: no DOM,
 * no engine, no storage. The structure parsers (src/services) supply the MoleculeData.
 *
 * LAMMPS docs (docs.lammps.org), read_data.html:
 * - "first line of the header and thus of the data file is always skipped", so the title line comes first.
 * - "the keyword xy xz yz should be in a line like" the three tilt values, then the keyword.
 * - "Thus the keyword atoms should be in a line like" a count, the same form for each header keyword.
 * - "The Bonds section must appear after the Atoms section."
 * - "These are the line formats for each atom style in LAMMPS", and a trailing comment is allowed:
 *   "Any individual line in the various sections can have a trailing comment starting with".
 * molecule.html:
 * - The native molecule file "individual molecule file looks similar but is" the data-file layout:
 *   sections Coords, Types, Charges and Bonds with the counts in the header.
 * create_atoms.html: "coordinates of a single particle (distance units)" for the single style.
 */
import type { Atom, Bond, BoxBounds, MoleculeData } from '../types';
import { ELEMENT_DATA } from '../constants';
import { inferBonds, maxBondDistance } from '../services/bondInference';
import { cifCellToBoxBounds, type CifCell } from '../services/cifParser';

/** Above this many atoms the notebook warns: the engine runs in the browser. */
export const WARN_ATOMS = 200_000;
/** Above this many atoms the conversion is refused. */
export const MAX_ATOMS = 1_000_000;

export type AtomStyleChoice = 'atomic' | 'charge' | 'full';

export interface ConvertOptions {
  /** data = read_data file; molecule = molecule template file. */
  target: 'data' | 'molecule';
  /**
   * elements: the atom types of the MoleculeData are atomic numbers (PDB, CIF, XYZ parsers).
   * lammps: the atom types are LAMMPS types of a data file, kept as they are.
   */
  source: 'elements' | 'lammps';
  atomStyle: AtomStyleChoice;
  /** One LAMMPS type per element, or per element and PDB atom name (needs atomNames). */
  typeBy: 'element' | 'pdbName';
  /** Write bonds (atom_style full for data files; always for molecule templates). */
  bonds: boolean;
  /** Padding (Angstrom) added around the atoms when the structure has no cell. */
  padding: number;
  /** Output file name, for example "protein.data" or "ligand.mol". */
  fileName: string;
  /** First line of the file (read_data skips it; molecule files keep it as a comment). */
  title: string;
  /** Cell as a LAMMPS box (restricted triclinic). Atoms are wrapped into it for source elements. */
  box?: BoxBounds;
  /** PDB atom names aligned with the atoms (see pdbAtomNames). */
  atomNames?: string[];
}

export interface TypeRow {
  id: number;
  label: string;
  element: string;
  mass: number;
  count: number;
}

export interface ConvertResult {
  fileName: string;
  text: string;
  natoms: number;
  nbonds: number;
  ntypes: number;
  nbondTypes: number;
  types: TypeRow[];
  /** Box written to the data file (undefined for molecule templates). */
  box?: BoxBounds;
  triclinic: boolean;
  /** Non-fatal notes shown in the dialog. */
  warnings: string[];
  /** Code-cell text that loads the file (into the notebook). */
  suggestion: string;
}

export class StructureConvertError extends Error {}

const SYMBOL_BY_NUMBER = new Map(ELEMENT_DATA.map((e) => [e.number, e] as const));
const TILT_EPS = 1e-9;

const fmt = (n: number): string => {
  const s = n.toFixed(6);
  return s === '-0.000000' ? '0.000000' : s;
};

/** Cell lengths and angles of a PDB CRYST1 record (wwPDB format 3.3, sect8). Null without a cell. */
export const pdbCell = (text: string): CifCell | null => {
  const line = text.split('\n').find((l) => l.substring(0, 6).trim() === 'CRYST1');
  if (!line) return null;
  const num = (s: number, e: number, dflt: number) => {
    const t = line.substring(s, e).trim();
    if (!t) return dflt;
    const v = parseFloat(t);
    return Number.isFinite(v) ? v : NaN;
  };
  const a = num(6, 15, NaN), b = num(15, 24, NaN), c = num(24, 33, NaN);
  const alpha = num(33, 40, 90), beta = num(40, 47, 90), gamma = num(47, 54, 90);
  if (![a, b, c, alpha, beta, gamma].every(Number.isFinite) || a <= 0 || b <= 0 || c <= 0) return null;
  return { a, b, c, alphaDeg: alpha, betaDeg: beta, gammaDeg: gamma };
};

/** The LAMMPS box of a PDB CRYST1 record (angles included), or undefined. */
export const pdbBox = (text: string): BoxBounds | undefined => {
  const cell = pdbCell(text);
  return cell ? cifCellToBoxBounds(cell) : undefined;
};

/**
 * PDB atom names aligned with `atoms` (the first model of the parsed structure). The parser keeps
 * no names, so each atom is matched by its coordinates, which the parser reads from the same
 * columns. Returns undefined when an atom cannot be matched.
 */
export const pdbAtomNames = (text: string, atoms: Atom[]): string[] | undefined => {
  const byPos = new Map<string, string>();
  for (const line of text.split('\n')) {
    const rec = line.substring(0, 6).trim();
    if (rec === 'ENDMDL') break;
    if (rec !== 'ATOM' && rec !== 'HETATM') continue;
    const x = parseFloat(line.substring(30, 38).trim());
    const y = parseFloat(line.substring(38, 46).trim());
    const z = parseFloat(line.substring(46, 54).trim());
    if (![x, y, z].every(Number.isFinite)) continue;
    const key = `${x}|${y}|${z}`;
    if (!byPos.has(key)) byPos.set(key, line.substring(12, 16).trim());
  }
  const names: string[] = [];
  for (const a of atoms) {
    const name = byPos.get(`${a.x}|${a.y}|${a.z}`);
    if (name === undefined) return undefined;
    names.push(name || '?');
  }
  return names;
};

type Vec3 = [number, number, number];

/** Fractional coordinates of a point in the restricted triclinic box (read_data.html: a = (lx,0,0), b = (xy,ly,0), c = (xz,yz,lz)). */
const toFrac = (x: number, y: number, z: number, box: BoxBounds): Vec3 => {
  const lx = box.xhi - box.xlo, ly = box.yhi - box.ylo, lz = box.zhi - box.zlo;
  const xy = box.xy ?? 0, xz = box.xz ?? 0, yz = box.yz ?? 0;
  const u = (z - box.zlo) / lz;
  const t = (y - box.ylo - u * yz) / ly;
  const s = (x - box.xlo - t * xy - u * xz) / lx;
  return [s, t, u];
};

/** The point of fractional coordinates f in the box. */
const fromFrac = (f: Vec3, box: BoxBounds): Vec3 => {
  const lx = box.xhi - box.xlo, ly = box.yhi - box.ylo, lz = box.zhi - box.zlo;
  const xy = box.xy ?? 0, xz = box.xz ?? 0, yz = box.yz ?? 0;
  return [box.xlo + f[0] * lx + f[1] * xy + f[2] * xz, box.ylo + f[1] * ly + f[2] * yz, box.zlo + f[2] * lz];
};

/** A bond between atoms a and b (0-based) where b's partner is its periodic image shifted by `shift` cells. */
interface PeriodicBond { a: number; b: number; shift: Vec3 }

/**
 * Bonds through the periodic cell, from covalent radii: the atoms (wrapped into the cell, fractional coordinates `frac`)
 * and their periodic images that lie within the longest possible bond of a cell face go through inferBonds; a bond to
 * an image keeps that image's cell shift. Each bond is listed once.
 */
const inferPeriodicBonds = (atoms: readonly Atom[], frac: readonly Vec3[], box: BoxBounds): PeriodicBond[] => {
  const lx = box.xhi - box.xlo, ly = box.yhi - box.ylo, lz = box.zhi - box.zlo;
  const xy = box.xy ?? 0, xz = box.xz ?? 0, yz = box.yz ?? 0;
  // the perpendicular widths of the cell: volume over the area of each pair of faces
  const vol = lx * ly * lz;
  const width: Vec3 = [vol / Math.hypot(ly * lz, xy * lz, xy * yz - ly * xz), vol / (lx * Math.hypot(lz, yz)), lz];
  const dmax = maxBondDistance(atoms);
  const delta = width.map((w) => dmax / w);
  const ext: Atom[] = [];
  const origin: { i: number; s: Vec3 }[] = [];
  const add = (i: number, s: Vec3): void => {
    const f = frac[i];
    const p = fromFrac([f[0] + s[0], f[1] + s[1], f[2] + s[2]], box);
    ext.push({ ...atoms[i], id: ext.length + 1, x: p[0], y: p[1], z: p[2] });
    origin.push({ i, s });
  };
  for (let i = 0; i < atoms.length; i++) add(i, [0, 0, 0]);
  for (let i = 0; i < atoms.length; i++) {
    const f = frac[i];
    for (let sx = -1; sx <= 1; sx++) for (let sy = -1; sy <= 1; sy++) for (let sz = -1; sz <= 1; sz++) {
      if (sx === 0 && sy === 0 && sz === 0) continue;
      const g = [f[0] + sx, f[1] + sy, f[2] + sz];
      if (g.every((v, k) => v >= -delta[k] && v <= 1 + delta[k])) add(i, [sx, sy, sz]);
    }
  }
  const out: PeriodicBond[] = [];
  const seen = new Set<string>();
  for (const bd of inferBonds(ext)) {
    const p = origin[bd.atom1Id - 1], q = origin[bd.atom2Id - 1];
    let a = p.i, b = q.i;
    let shift: Vec3 = [q.s[0] - p.s[0], q.s[1] - p.s[1], q.s[2] - p.s[2]];
    // one listing per bond: the lower atom first (for a bond to the atom's own image, the shift that is positive first)
    const flip = a > b || (a === b && (shift[0] < 0 || (shift[0] === 0 && (shift[1] < 0 || (shift[1] === 0 && shift[2] < 0)))));
    if (flip) { [a, b] = [b, a]; shift = [-shift[0], -shift[1], -shift[2]]; }
    const key = `${a},${b},${shift.join(',')}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ a, b, shift });
  }
  return out;
};

/**
 * The image flags that keep every bonded molecule whole: a walk over the bonds from the first atom of each molecule
 * gives each atom the image of its bonded neighbour plus the bond's shift. A molecule whose walk comes back to an atom
 * with another image (a framework or covalent crystal bonded through the cell) keeps zero flags; `spanning` counts them.
 */
const wholeMoleculeImages = (n: number, bonds: readonly PeriodicBond[]): { img: Int32Array; spanning: number } => {
  const adj: { to: number; s: Vec3 }[][] = Array.from({ length: n }, () => []);
  for (const b of bonds) {
    adj[b.a].push({ to: b.b, s: b.shift });
    adj[b.b].push({ to: b.a, s: [-b.shift[0], -b.shift[1], -b.shift[2]] });
  }
  const img = new Int32Array(3 * n);
  const done = new Uint8Array(n);
  let spanning = 0;
  for (let root = 0; root < n; root++) {
    if (done[root]) continue;
    done[root] = 1;
    const members = [root];
    let whole = true;
    for (let k = 0; k < members.length; k++) {
      const u = members[k];
      for (const { to, s } of adj[u]) {
        const want = [img[3 * u] + s[0], img[3 * u + 1] + s[1], img[3 * u + 2] + s[2]];
        if (!done[to]) {
          done[to] = 1;
          img[3 * to] = want[0]; img[3 * to + 1] = want[1]; img[3 * to + 2] = want[2];
          members.push(to);
        } else if (img[3 * to] !== want[0] || img[3 * to + 1] !== want[1] || img[3 * to + 2] !== want[2]) {
          whole = false;
        }
      }
    }
    if (!whole) {
      spanning++;
      for (const m of members) img[3 * m] = img[3 * m + 1] = img[3 * m + 2] = 0;
    }
  }
  return { img, spanning };
};

/** Union-find over atoms (1-based index) for molecule IDs from bonds. */
const componentIds = (n: number, pairs: [number, number][]): number[] => {
  const parent = Array.from({ length: n + 1 }, (_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  for (const [a, b] of pairs) {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  }
  const ids = new Array<number>(n + 1).fill(0);
  const seen = new Map<number, number>();
  for (let i = 1; i <= n; i++) {
    const r = find(i);
    if (!seen.has(r)) seen.set(r, seen.size + 1);
    ids[i] = seen.get(r)!;
  }
  return ids;
};

/** A file name for a generated file: the base name of the input with the given extension. */
export const outputName = (inputName: string, ext: 'data' | 'mol'): string => {
  const base = inputName.replace(/\.[^.]*$/, '').replace(/[^A-Za-z0-9_.-]+/g, '_') || 'structure';
  return `${base}.${ext}`;
};

/**
 * Converts the structure. Throws StructureConvertError for inputs that cannot be converted
 * (no atoms, too many atoms, an element without a mass in the element table, a data-file type
 * without a mass).
 */
export const convertStructure = (mol: MoleculeData, opts: ConvertOptions): ConvertResult => {
  const n = mol.atoms.length;
  if (n === 0) throw new StructureConvertError('The file has no atoms.');
  if (n > MAX_ATOMS) {
    throw new StructureConvertError(
      `${n.toLocaleString('en-US')} atoms is above the limit of ${MAX_ATOMS.toLocaleString('en-US')} for the in-browser engine. Cut the structure down first.`,
    );
  }
  const warnings: string[] = [];
  if (n > WARN_ATOMS) {
    warnings.push(`${n.toLocaleString('en-US')} atoms: the in-browser engine will be slow to read and run this file.`);
  }

  // Atom types. Rows keep the order of first appearance (elements) or the file's type numbers (lammps).
  const typeOf = new Array<number>(n);
  const labelOf = new Array<string>(n);
  const elementOf = new Array<string>(n);
  const rows: TypeRow[] = [];
  if (opts.source === 'elements') {
    const rowByKey = new Map<string, number>();
    for (let i = 0; i < n; i++) {
      const a = mol.atoms[i];
      const el = SYMBOL_BY_NUMBER.get(a.type);
      if (!el) {
        throw new StructureConvertError(
          `Atom ${i + 1} has an element that is not in the element table (type ${a.type}). Its mass is unknown, so the file is not converted.`,
        );
      }
      const name = opts.typeBy === 'pdbName' ? opts.atomNames?.[i] : undefined;
      const key = name ? `${el.symbol}:${name}` : el.symbol;
      let id = rowByKey.get(key);
      if (id === undefined) {
        id = rows.length + 1;
        rowByKey.set(key, id);
        rows.push({ id, label: name ? `${name} (${el.symbol})` : el.symbol, element: el.symbol, mass: el.mass, count: 0 });
      }
      rows[id - 1].count++;
      typeOf[i] = id;
      labelOf[i] = rows[id - 1].label;
      elementOf[i] = el.symbol;
    }
  } else {
    const maxType = mol.atoms.reduce((m, a) => Math.max(m, a.type), 0);
    const counts = new Map<number, number>();
    for (const a of mol.atoms) counts.set(a.type, (counts.get(a.type) ?? 0) + 1);
    for (let t = 1; t <= maxType; t++) {
      const mass = mol.atomTypes[t]?.mass;
      if (!(mass && mass > 0)) {
        throw new StructureConvertError(`Atom type ${t} has no mass in the data file, so the file is not converted.`);
      }
      rows.push({ id: t, label: `type ${t}`, element: mol.atomTypes[t]?.element ?? 'X', mass, count: counts.get(t) ?? 0 });
    }
    for (let i = 0; i < n; i++) {
      typeOf[i] = mol.atoms[i].type;
      labelOf[i] = `type ${mol.atoms[i].type}`;
      elementOf[i] = rows[mol.atoms[i].type - 1].element;
    }
  }
  const ntypes = rows.length;

  // Bonds, numbered 1..n by the renumbered atom IDs.
  const index = new Map<number, number>();
  mol.atoms.forEach((a, i) => index.set(a.id, i + 1));
  const writeBonds = opts.bonds && (opts.target === 'molecule' || opts.atomStyle === 'full');
  let rawBonds: Bond[] = [];
  let nbondTypes = 0;
  const bondTypeRows = new Map<string, number>();
  // A periodic structure (a cell, element types): the atoms are wrapped into the cell, the guessed bonds are inferred
  // again through the periodic cell, and molecules cut by the cell faces are kept whole by image flags (read_data.html:
  // "the image flags ... which indicate which image of the periodic box the atom is in"; molecule templates get whole
  // coordinates).
  const periodicCell = opts.source === 'elements' && opts.box ? opts.box : undefined;
  const frac: Vec3[] = [];
  const shiftOf = new Map<Bond, Vec3>();
  if (periodicCell) {
    for (const a of mol.atoms) {
      const f = toFrac(a.x, a.y, a.z, periodicCell);
      frac.push([f[0] - Math.floor(f[0]), f[1] - Math.floor(f[1]), f[2] - Math.floor(f[2])]);
    }
  }
  if (writeBonds) {
    const guessed = mol.bonds.length === 0 || mol.bonds.length > 0 && mol.bondsInferred === true;
    if (periodicCell && guessed && opts.source === 'elements') {
      for (const pb of inferPeriodicBonds(mol.atoms, frac, periodicCell)) {
        const bd: Bond = { id: rawBonds.length + 1, type: 1, atom1Id: mol.atoms[pb.a].id, atom2Id: mol.atoms[pb.b].id };
        rawBonds.push(bd);
        shiftOf.set(bd, pb.shift);
      }
      if (rawBonds.length > 0) warnings.push('Bonds are inferred from covalent radii through the periodic cell (the file lists none); molecules cut by the cell faces are kept whole.');
    } else {
      rawBonds = mol.bonds.length > 0 ? mol.bonds : (opts.source === 'elements' ? inferBonds(mol.atoms) : []);
      if (opts.source === 'elements' && mol.bonds.length === 0 && rawBonds.length > 0) {
        warnings.push('Bonds are inferred from covalent radii (the file has no CONECT records).');
      }
    }
  }
  const bondPairs: [number, number][] = [];
  const bondLines: { type: number; a: number; b: number }[] = [];
  /** the bonds with their cell shifts, for the whole-molecule image flags of a periodic structure */
  const shifted: PeriodicBond[] = [];
  for (const b of rawBonds) {
    const ia = index.get(b.atom1Id), ib = index.get(b.atom2Id);
    if (ia === undefined || ib === undefined) continue;
    if (periodicCell) {
      // a bond read from the file joins the nearest images of its atoms
      const fa = frac[ia - 1], fb = frac[ib - 1];
      const shift = shiftOf.get(b) ?? [Math.round(fa[0] - fb[0]), Math.round(fa[1] - fb[1]), Math.round(fa[2] - fb[2])];
      shifted.push({ a: ia - 1, b: ib - 1, shift });
    }
    if (ia === ib) continue;
    bondPairs.push([ia, ib]);
    let type: number;
    if (opts.source === 'lammps') {
      type = b.type;
    } else {
      const key = [elementOf[ia - 1], elementOf[ib - 1]].sort().join('-');
      let t = bondTypeRows.get(key);
      if (t === undefined) {
        t = bondTypeRows.size + 1;
        bondTypeRows.set(key, t);
      }
      type = t;
    }
    bondLines.push({ type, a: ia, b: ib });
  }
  // image flags that keep the molecules whole; a bonded network that runs through the cell keeps zero flags
  let img: Int32Array | null = null;
  if (periodicCell) {
    const whole = wholeMoleculeImages(n, shifted);
    img = whole.img;
    if (whole.spanning > 0) {
      warnings.push(`${whole.spanning} bonded network${whole.spanning === 1 ? ' runs' : 's run'} through the periodic cell (a framework or covalent crystal): ${whole.spanning === 1 ? 'its' : 'their'} atoms are written wrapped into the cell.`);
    }
  }
  if (opts.source === 'lammps') {
    nbondTypes = bondLines.reduce((m, b) => Math.max(m, b.type), 0);
  } else {
    nbondTypes = bondTypeRows.size;
  }
  const nbonds = bondLines.length;

  // Coordinates: wrapped into the cell when there is one (elements); raw otherwise.
  const pts: [number, number, number][] = mol.atoms.map((a) => [a.x, a.y, a.z]);
  let box: BoxBounds | undefined;
  let triclinic = false;
  if (opts.target === 'data') {
    if (opts.source === 'lammps') {
      box = opts.box ?? mol.box;
      if (!box) throw new StructureConvertError('The data file has no box bounds.');
    } else if (opts.box) {
      box = opts.box;
      for (let i = 0; i < n; i++) pts[i] = fromFrac(frac[i], box);
    } else {
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const p of pts) {
        for (let k = 0; k < 3; k++) {
          if (p[k] < lo[k]) lo[k] = p[k];
          if (p[k] > hi[k]) hi[k] = p[k];
        }
      }
      box = {
        xlo: lo[0] - opts.padding, xhi: hi[0] + opts.padding,
        ylo: lo[1] - opts.padding, yhi: hi[1] + opts.padding,
        zlo: lo[2] - opts.padding, zhi: hi[2] + opts.padding,
      };
      warnings.push(`No cell in the file: the box is the atom extent plus ${opts.padding} Angstrom on each side. Use boundary f f f.`);
    }
    triclinic = Math.abs(box.xy ?? 0) > TILT_EPS || Math.abs(box.xz ?? 0) > TILT_EPS || Math.abs(box.yz ?? 0) > TILT_EPS;
  }

  const periodic = opts.target === 'data' && !!opts.box && opts.source === 'elements';
  // a molecule template from a periodic structure: the whole molecules (the cell's wrapped atoms moved by their images)
  if (opts.target === 'molecule' && periodicCell && img) {
    for (let i = 0; i < n; i++) {
      const f = frac[i];
      pts[i] = fromFrac([f[0] + img[3 * i], f[1] + img[3 * i + 1], f[2] + img[3 * i + 2]], periodicCell);
    }
  }

  let text: string;
  let suggestion: string;
  if (opts.target === 'data') {
    const style = opts.atomStyle;
    const molIds = style === 'full' ? componentIds(n, writeBonds ? bondPairs : []) : null;
    const out: string[] = [];
    out.push(opts.title, '');
    out.push(`${n} atoms`);
    if (nbonds > 0 && writeBonds) out.push(`${nbonds} bonds`);
    out.push(`${ntypes} atom types`);
    if (nbonds > 0 && writeBonds) out.push(`${nbondTypes} bond types`);
    out.push('');
    const b = box as BoxBounds;
    out.push(`${fmt(b.xlo)} ${fmt(b.xhi)} xlo xhi`);
    out.push(`${fmt(b.ylo)} ${fmt(b.yhi)} ylo yhi`);
    out.push(`${fmt(b.zlo)} ${fmt(b.zhi)} zlo zhi`);
    if (triclinic) out.push(`${fmt(b.xy ?? 0)} ${fmt(b.xz ?? 0)} ${fmt(b.yz ?? 0)} xy xz yz`);
    out.push('', 'Masses', '');
    for (const r of rows) out.push(`${r.id} ${r.mass} # ${r.label}`);
    out.push('', `Atoms # ${style}`, '');
    for (let i = 0; i < n; i++) {
      const cols = [String(i + 1)];
      if (molIds) cols.push(String(molIds[i + 1]));
      cols.push(String(typeOf[i]));
      if (style !== 'atomic') cols.push(String(mol.atoms[i].charge));
      cols.push(fmt(pts[i][0]), fmt(pts[i][1]), fmt(pts[i][2]));
      if (img) cols.push(String(img[3 * i]), String(img[3 * i + 1]), String(img[3 * i + 2]));
      out.push(`${cols.join(' ')} # ${labelOf[i]}`);
    }
    if (nbonds > 0 && writeBonds) {
      out.push('', 'Bonds', '');
      bondLines.forEach((bl, k) => out.push(`${k + 1} ${bl.type} ${bl.a} ${bl.b}`));
    }
    out.push('');
    text = out.join('\n');
    suggestion = [
      'units real',
      `atom_style ${style}`,
      `boundary ${periodic ? 'p p p' : 'f f f'}`,
      `read_data ${opts.fileName}`,
      'run 0',
    ].join('\n');
  } else {
    // Molecule template: raw coordinates, types, charges, bonds (molecule.html sections).
    const out: string[] = [];
    out.push(opts.title);
    out.push(`${n} atoms`);
    if (nbonds > 0 && writeBonds) out.push(`${nbonds} bonds`);
    out.push('', 'Coords', '');
    for (let i = 0; i < n; i++) out.push(`${i + 1} ${fmt(pts[i][0])} ${fmt(pts[i][1])} ${fmt(pts[i][2])}`);
    out.push('', 'Types', '');
    for (let i = 0; i < n; i++) out.push(`${i + 1} ${typeOf[i]} # ${labelOf[i]}`);
    if (opts.atomStyle !== 'atomic') {
      out.push('', 'Charges', '');
      for (let i = 0; i < n; i++) out.push(`${i + 1} ${mol.atoms[i].charge}`);
    }
    if (nbonds > 0 && writeBonds) {
      out.push('', 'Bonds', '');
      bondLines.forEach((bl, k) => out.push(`${k + 1} ${bl.type} ${bl.a} ${bl.b}`));
    }
    out.push('');
    text = out.join('\n');
    // The molecule goes into a box sized to the atoms plus padding.
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) {
      for (let k = 0; k < 3; k++) {
        if (p[k] < lo[k]) lo[k] = p[k];
        if (p[k] > hi[k]) hi[k] = p[k];
      }
    }
    const pad = opts.padding;
    const regionLo = lo.map((v) => v - pad), regionHi = hi.map((v) => v + pad);
    const centre = [0, 1, 2].map((k) => fmt((lo[k] + hi[k]) / 2));
    const maxDeg = new Array<number>(n + 1).fill(0);
    for (const [a, c] of bondPairs) {
      maxDeg[a]++;
      maxDeg[c]++;
    }
    let extra = 1;
    for (const d of maxDeg) if (d > extra) extra = d;
    const bondKw = nbonds > 0 && writeBonds ? ` bond/types ${nbondTypes} extra/bond/per/atom ${extra}` : '';
    suggestion = [
      'units real',
      'atom_style full',
      'boundary f f f',
      `region box block ${[0, 1, 2].flatMap((k) => [fmt(regionLo[k]), fmt(regionHi[k])]).join(' ')} units box`,
      `create_box ${ntypes} box${bondKw}`,
      ...rows.map((r) => `mass ${r.id} ${r.mass}`),
      `molecule m ${opts.fileName}`,
      `create_atoms 0 single ${centre.join(' ')} mol m 12345 units box`,
    ].join('\n');
  }

  return {
    fileName: opts.fileName,
    text,
    natoms: n,
    nbonds: writeBonds ? nbonds : 0,
    ntypes,
    nbondTypes: writeBonds ? nbondTypes : 0,
    types: rows,
    box: opts.target === 'data' ? box : undefined,
    triclinic,
    warnings,
    suggestion,
  };
};
