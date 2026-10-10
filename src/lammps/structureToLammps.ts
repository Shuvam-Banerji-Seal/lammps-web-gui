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
import { inferBonds } from '../services/bondInference';
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

/** Wraps a point into the periodic restricted triclinic box (fractional coordinates in [0, 1)). */
const wrapInto = (x: number, y: number, z: number, box: BoxBounds): [number, number, number] => {
  const lx = box.xhi - box.xlo, ly = box.yhi - box.ylo, lz = box.zhi - box.zlo;
  const xy = box.xy ?? 0, xz = box.xz ?? 0, yz = box.yz ?? 0;
  const dz = z - box.zlo;
  const u = dz / lz;
  const t = (y - box.ylo - u * yz) / ly;
  const s = (x - box.xlo - t * xy - u * xz) / lx;
  const fs = s - Math.floor(s), ft = t - Math.floor(t), fu = u - Math.floor(u);
  return [
    box.xlo + fs * lx + ft * xy + fu * xz,
    box.ylo + ft * ly + fu * yz,
    box.zlo + fu * lz,
  ];
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
  if (writeBonds) {
    rawBonds = mol.bonds.length > 0 ? mol.bonds : (opts.source === 'elements' ? inferBonds(mol.atoms) : []);
    if (opts.source === 'elements' && mol.bonds.length === 0 && rawBonds.length > 0 && opts.box) {
      warnings.push('Bonds are inferred from covalent radii (the file has no CONECT records). Bonds across the periodic boundary are not added.');
    }
  }
  const bondPairs: [number, number][] = [];
  const bondLines: { type: number; a: number; b: number }[] = [];
  for (const b of rawBonds) {
    const ia = index.get(b.atom1Id), ib = index.get(b.atom2Id);
    if (ia === undefined || ib === undefined || ia === ib) continue;
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
      for (let i = 0; i < n; i++) pts[i] = wrapInto(pts[i][0], pts[i][1], pts[i][2], box);
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
    const extra = Math.max(1, ...maxDeg);
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
