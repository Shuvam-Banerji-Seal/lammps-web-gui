import { StyleError } from './force/types';
import { LabelMap, inferAngleType, inferDihedralType, inferImproperType, type LabelKind } from './labelmap';

/*
 * Molecule templates — docs.lammps.org/molecule.html: "Define a molecule
 * template that can be used as part of other LAMMPS commands, typically to
 * define a collection of particles as a bonded molecule or a rigid body."
 *
 * Native file format: "A molecule file has a header and a body. The header
 * appears first. The first line of the header and thus of the molecule file
 * is always skipped"; "Lines can have a trailing comment starting with '#'
 * that is ignored"; "If the line contains a header keyword, the
 * corresponding value(s) is/are read from the line. A line that is not
 * blank and does not contains a header keyword begins the body of the
 * file." "The first line of a section has only a keyword. The next line is
 * skipped. The remaining lines of the section contain values."
 * Sections read here: Coords ("line syntax: ID x y z"), Types ("ID type"),
 * Molecules ("ID molecule-ID"), Charges ("ID q"), Bonds ("ID type atom1
 * atom2"), Angles, Dihedrals, Impropers, Special Bond Counts ("ID N1 N2
 * N3"), Special Bonds ("ID a b c d ..."), Shake Flags, Shake Atoms, Shake
 * Bond Types (kept for fix shake). Diameters and Masses are read as per-atom
 * arrays and carried into created atoms by the insertion paths (create_atoms
 * mol, fix pour mol, fix deposit mol). "This section is only allowed for
 * atom styles that support finite-size spherical particles, e.g. atom_style
 * sphere. If not listed, the default diameter of each atom in the molecule
 * is 1.0." (Diameters) and "This section is only allowed for atom styles
 * that support per-atom mass, as opposed to per-type mass. See the mass
 * command for details. If this section is not included, the default mass for
 * each atom is derived from its volume (see Diameters section) and a default
 * density of 1.0, in units of mass/volume." (Masses). The scale keyword
 * applies to both: "The scale factor is applied to each of these properties
 * in the molecule file, if they are defined: the individual particle
 * coordinates (...), the individual mass of each particle (Masses ...), the
 * individual diameters of each particle (Diameters ...)". Dipoles: "This
 * section is only allowed for atom styles that support particles with point
 * dipoles, e.g. atom_style dipole." (checked where the atoms are inserted,
 * see templateDipoles). Fragments ("name atom-IDs" lines, read for the fix bond/react modify_create fit) are read. Body molecules (a "1 atoms" template with
 * a "N M body" header, Body Integers and Body Doubles, and Masses) are read into MoleculeTemplate.body; the body radius is
 * set by the inserting fix (fix pour), which knows the atom style.
 * Keywords: offset (all five type offsets), toff / boff / aoff / doff / ioff
 * (one each), and scale: "The scale factor is applied to each of these
 * properties in the molecule file, if they are defined: the individual
 * particle coordinates".
 */

export interface MoleculeTemplate {
  id: string;
  natoms: number;
  /** 3N coordinates as read (scaled). */
  x: Float64Array;
  type: Int32Array;
  q: Float64Array | null;
  /** Molecule IDs within the template (Molecules section), or null. */
  mol: Int32Array | null;
  /** Topology with template-local atom indices 1..N. */
  bonds: number[][];
  angles: number[][];
  dihedrals: number[][];
  impropers: number[][];
  /** Explicit special lists (template-local IDs per atom: [1-2], [1-3], [1-4]), or null. */
  special: number[][][] | null;
  shake: { flags: Int32Array; atoms: number[][]; types: number[][] } | null;
  /** Per-atom diameters (Diameters section, scaled), or null (default 1.0). */
  diam: Float64Array | null;
  /** Per-atom masses (Masses section, scaled), or null (default from the diameter and density 1.0). */
  mass: Float64Array | null;
  /**
   * Body particle entry (molecule.html, "Body Integers" and "Body Doubles"): the integers and doubles of the body style,
   * or absent for a point molecule. Its radius is not set here (bodies.ts bodyRadiusBody, from the atom style).
   */
  body?: { ints: number[]; dbls: number[] } | null;
  /** Per-atom point dipoles (Dipoles section: "ID mux muy muz", scaled), flat 3N, or null (default 0). */
  mu: Float64Array | null;
  /**
   * Fragments (molecule.html, Fragments section: "one line per fragment"; "a,b,c,d,... = IDs of atoms in fragment"):
   * the name and the 1-based atom IDs of each fragment, in file order. Empty when the file has none.
   */
  fragments: { name: string; atoms: number[] }[];
  /**
   * Surface templates (no atoms): the Lines section of a "lines" molecule (2d) or the Triangles section of a
   * "triangles" molecule (3d). p holds the end points x1 y1 x2 y2 (lines) or the corners x1 y1 z1 ... (triangles),
   * scaled. null for atom templates.
   */
  lines: SurfaceEntry[] | null;
  tris: SurfaceEntry[] | null;
}

/** One line segment or triangle of a surface template: molecule-ID, type (offset applied) and corner coordinates. */
export interface SurfaceEntry {
  mol: number;
  type: number;
  p: number[];
}

export interface MoleculeOptions {
  toff: number; boff: number; aoff: number; doff: number; ioff: number; scale: number;
  /** auto keyword: the letters 'a', 'd', 'i' (generate angles, dihedrals, impropers), or '' */
  auto?: string;
  /** the system's unit style, checked against a JSON file's "units" entry */
  units?: string;
  /**
   * The symmetry of the system's improper style (symmatoms of the style, see inferImproperType), or null when
   * the style does not provide one. Undefined means no improper style is defined.
   */
  improperSymmatoms?: readonly number[] | null;
}

const HEADER: Record<string, number> = { atoms: 1, bonds: 1, angles: 1, dihedrals: 1, impropers: 1, fragments: 1, body: 2, mass: 1, com: 3, inertia: 6, lines: 1, triangles: 1 };

const strip = (line: string): string => {
  // "There must be at least one blank between any valid content and the comment."
  const k = line.search(/(^|\s)#/);
  return (k < 0 ? line : line.slice(0, k)).trim();
};

const parseMoleculeText = (id: string, name: string, text: string, o: MoleculeOptions, labels?: LabelMap): MoleculeTemplate => {
  const lines = text.split('\n');
  const head: Record<string, number[]> = {};
  let k = 1;
  // header
  for (; k < lines.length; k++) {
    const l = strip(lines[k]);
    if (!l) continue;
    const w = l.split(/\s+/);
    const key = Object.keys(HEADER).find((h) => w[w.length - 1] === h || (h === 'body' && w[w.length - 1] === 'body'));
    if (!key) break;
    const vals = w.slice(0, w.length - 1).map(Number);
    if (vals.length !== HEADER[key] || vals.some((v) => !Number.isFinite(v))) throw new StyleError(`molecule ${id}: bad header line '${l}' in ${name}`);
    head[key] = vals;
  }
  // molecule.html, header keywords: "A molecule file can contain only either the *atoms*, *lines*, or *triangles*
  // keyword." The surface templates (lines, triangles) have no atoms, so their atom count is 0.
  const hasAtoms = head.atoms !== undefined;
  if (!hasAtoms && !head.lines && !head.triangles) throw new StyleError(`molecule ${id}: ${name} needs an "N atoms" header line`);
  if (hasAtoms && (head.lines || head.triangles)) throw new StyleError(`molecule ${id}: ${name} has both an atoms and a lines or triangles header keyword; a molecule file can contain only one of them`);
  if (head.lines && head.triangles) throw new StyleError(`molecule ${id}: ${name} has both lines and triangles header keywords; a molecule file can contain only one of them`);
  const natoms = hasAtoms ? head.atoms[0] : 0;
  if (hasAtoms && (natoms < 1 || !Number.isInteger(natoms))) throw new StyleError(`molecule ${id}: ${name} needs an "N atoms" header line`);
  if (head.lines && (!Number.isInteger(head.lines[0]) || head.lines[0] < 1)) throw new StyleError(`molecule ${id}: ${name} has no lines or an invalid line count`);
  if (head.triangles && (!Number.isInteger(head.triangles[0]) || head.triangles[0] < 1)) throw new StyleError(`molecule ${id}: ${name} has no triangles or an invalid triangle count`);
  // molecule.html: "defining a *body* particle, which requires setting the number of *atoms* to 1, and setting the *inertia*
  // in a specific section". The Body Integers and Body Doubles counts come from the header (Ninteger, Ndouble).
  const bodyHeader = head.body !== undefined && (head.body[0] !== 0 || head.body[1] !== 0);
  if (bodyHeader && (!hasAtoms || natoms !== 1)) throw new StyleError(`molecule ${id}: Molecule natoms must be 1 for body particle`);
  if (bodyHeader && (head.body![0] < 1 || head.body![1] < 1)) throw new StyleError(`molecule ${id}: a body molecule needs Ninteger and Ndouble values of at least 1 in ${name}`);
  if (bodyHeader && o.scale !== 1) throw new StyleError(`molecule ${id}: Molecule sizescale must be 1.0 for body particle`);
  let bodyInts: number[] | null = null, bodyDbls: number[] | null = null;
  const count = (h: string) => head[h]?.[0] ?? 0;
  const t: MoleculeTemplate = {
    id, natoms, x: new Float64Array(3 * natoms), type: new Int32Array(natoms), q: null, mol: null,
    bonds: [], angles: [], dihedrals: [], impropers: [], special: null, shake: null, diam: null, mass: null, mu: null,
    fragments: [],
    lines: null, tris: null,
  };
  const seen = new Set<string>();
  let specialCounts: number[][] | null = null;
  // body
  while (k < lines.length) {
    const sec = strip(lines[k]);
    if (!sec) { k++; continue; }
    k += 2; // the keyword line and the skipped line after it
    const wordRows = (n: number, width: number): string[][] => {
      const out: string[][] = [];
      for (; out.length < n && k < lines.length; k++) {
        const l = strip(lines[k]);
        if (!l) continue;
        const w = l.split(/\s+/);
        if (w.length < width) throw new StyleError(`molecule ${id}: bad line '${l}' in the ${sec} section of ${name}`);
        out.push(w);
      }
      if (out.length < n) throw new StyleError(`molecule ${id}: the ${sec} section of ${name} has ${out.length} of ${n} lines`);
      return out;
    };
    const rows = (n: number, width: number): number[][] => {
      const out: number[][] = [];
      for (const w of wordRows(n, width)) {
        const v = w.map(Number);
        if (v.slice(0, width).some((x) => !Number.isFinite(x))) throw new StyleError(`molecule ${id}: bad line '${w.join(' ')}' in the ${sec} section of ${name}`);
        out.push(v);
      }
      return out;
    };
    /** The numeric type of a type word of this file: a number (plus the offset) or a label of the kind (labels take no offset). */
    const typeOf = (word: string, kind: LabelKind, off: number): number => {
      if (/^[+-]?\d/.test(word)) {
        const v = Number(word);
        if (!Number.isInteger(v)) throw new StyleError(`molecule ${id}: bad type '${word}' in the ${sec} section of ${name}`);
        return v + off;
      }
      const v = labels?.typeOf(kind, word);
      if (v === undefined) throw new StyleError(`molecule ${id}: ${kind} type label ${word} is not defined in the ${sec} section of ${name}`);
      return v;
    };
    const atomIndex = (v: number, what: string) => {
      if (!Number.isInteger(v) || v < 1 || v > natoms) throw new StyleError(`molecule ${id}: ${what} ${v} is outside 1..${natoms} in ${name}`);
      return v;
    };
    seen.add(sec);
    switch (sec) {
      case 'Coords':
        for (const r of rows(natoms, 4)) { const i = atomIndex(r[0], 'atom ID') - 1; for (let d = 0; d < 3; d++) t.x[3 * i + d] = r[1 + d] * o.scale; }
        break;
      case 'Types':
        for (const w of wordRows(natoms, 2)) t.type[atomIndex(Number(w[0]), 'atom ID') - 1] = typeOf(w[1], 'atom', o.toff);
        break;
      case 'Molecules':
        t.mol = new Int32Array(natoms);
        for (const r of rows(natoms, 2)) t.mol[atomIndex(r[0], 'atom ID') - 1] = r[1];
        break;
      case 'Charges':
        t.q = new Float64Array(natoms);
        for (const r of rows(natoms, 2)) t.q[atomIndex(r[0], 'atom ID') - 1] = r[1];
        break;
      case 'Bonds': case 'Angles': case 'Dihedrals': case 'Impropers': {
        const width = { Bonds: 2, Angles: 3, Dihedrals: 4, Impropers: 4 }[sec];
        const off = { Bonds: o.boff, Angles: o.aoff, Dihedrals: o.doff, Impropers: o.ioff }[sec];
        const list = { Bonds: t.bonds, Angles: t.angles, Dihedrals: t.dihedrals, Impropers: t.impropers }[sec];
        const kind: LabelKind = { Bonds: 'bond', Angles: 'angle', Dihedrals: 'dihedral', Impropers: 'improper' }[sec] as LabelKind;
        for (const w of wordRows(count(sec.toLowerCase()), 2 + width)) {
          // native refuses a bonded entry that names one atom twice (measured with native LAMMPS, black box:
          // "1 1 1 2 2" in Angles, "1 1 1 1" in Bonds and "1 1 1 2 3 1" in Dihedrals all stop with
          // "Invalid atom ID in <section> section of molecule file")
          const raw = w.slice(2, 2 + width).map(Number);
          if (new Set(raw).size !== raw.length) throw new StyleError(`molecule ${id}: Invalid atom ID in ${sec} section of ${name}: ${w.join(' ')}`);
          const ids = raw.map((v) => atomIndex(v, 'atom ID'));
          list.push([typeOf(w[1], kind, off), ...ids]);
        }
        break;
      }
      case 'Fragments':
        // molecule.html: "The fragment-ID is a name; the atom IDs are the atoms of the fragment"
        for (const w of wordRows(count('fragments'), 1)) {
          t.fragments.push({ name: w[0], atoms: w.slice(1).map((v) => atomIndex(Number(v), 'atom ID in fragment')) });
        }
        break;
      case 'Special Bond Counts':
        specialCounts = rows(natoms, 4).map((r) => [atomIndex(r[0], 'atom ID'), r[1], r[2], r[3]]);
        break;
      case 'Special Bonds': {
        if (!specialCounts) throw new StyleError(`molecule ${id}: Special Bonds must follow Special Bond Counts in ${name}`);
        const byAtom = new Map(specialCounts.map((c) => [c[0], c]));
        t.special = Array.from({ length: natoms }, () => [[], [], []] as number[][]);
        for (const r of rows(natoms, 1)) {
          const i = atomIndex(r[0], 'atom ID');
          const c = byAtom.get(i)!;
          const ids = r.slice(1).map((v) => atomIndex(v, 'special neighbor'));
          if (ids.length !== c[1] + c[2] + c[3]) throw new StyleError(`molecule ${id}: atom ${i} lists ${ids.length} special neighbors, Special Bond Counts says ${c[1] + c[2] + c[3]}`);
          t.special[i - 1] = [ids.slice(0, c[1]), ids.slice(c[1], c[1] + c[2]), ids.slice(c[1] + c[2])];
        }
        break;
      }
      case 'Shake Flags': {
        t.shake ??= { flags: new Int32Array(natoms), atoms: Array.from({ length: natoms }, () => []), types: Array.from({ length: natoms }, () => []) };
        for (const r of rows(natoms, 2)) t.shake.flags[atomIndex(r[0], 'atom ID') - 1] = r[1];
        break;
      }
      case 'Shake Atoms': case 'Shake Bond Types': {
        t.shake ??= { flags: new Int32Array(natoms), atoms: Array.from({ length: natoms }, () => []), types: Array.from({ length: natoms }, () => []) };
        for (const r of rows(natoms, 1)) (sec === 'Shake Atoms' ? t.shake.atoms : t.shake.types)[atomIndex(r[0], 'atom ID') - 1] = r.slice(1);
        break;
      }
      case 'Masses': {
        t.mass = new Float64Array(natoms);
        for (const r of rows(natoms, 2)) t.mass[atomIndex(r[0], 'atom ID') - 1] = r[1] * o.scale;
        break;
      }
      case 'Body Integers': case 'Body Doubles': {
        // molecule.html: the values "must be equal to the Ninteger value given in the *body* header" (Ndouble for doubles)
        const want = head.body?.[sec === 'Body Integers' ? 0 : 1] ?? 0;
        if (!bodyHeader) throw new StyleError(`molecule ${id}: found the ${sec} section of ${name} but no body header line`);
        const vals: number[] = [];
        while (vals.length < want && k < lines.length) {
          const l = strip(lines[k]);
          k++;
          if (!l) continue;
          for (const w of l.split(/\s+/)) {
            const v = Number(w);
            if (!Number.isFinite(v)) throw new StyleError(`molecule ${id}: bad value '${w}' in the ${sec} section of ${name}`);
            vals.push(v);
          }
          if (vals.length > want) throw new StyleError(`molecule ${id}: too many values in the ${sec} section of ${name}`);
        }
        if (vals.length !== want) throw new StyleError(`molecule ${id}: the ${sec} section of ${name} has ${vals.length} of ${want} values`);
        if (sec === 'Body Integers') bodyInts = vals; else bodyDbls = vals;
        break;
      }
      case 'Diameters': {
        t.diam = new Float64Array(natoms);
        for (const r of rows(natoms, 2)) t.diam[atomIndex(r[0], 'atom ID') - 1] = r[1] * o.scale;
        break;
      }
      case 'Lines': case 'Triangles': {
        // molecule.html, Lines: "line syntax: ID molecule-ID type x1 y1 x2 y2"; Triangles: "line syntax: ID molecule-ID
        // type x1 y1 z1 x2 y2 z2 x3 y3 z3". Each entry is listed once and the IDs run 1..N.
        const surf = sec === 'Lines' ? 'lines' : 'triangles';
        if (!head[surf]) throw new StyleError(`molecule ${id}: the ${sec} section of ${name} needs the "${surf}" header keyword`);
        const n = count(surf), width = sec === 'Lines' ? 4 : 9;
        const out: (SurfaceEntry | null)[] = new Array(n).fill(null);
        for (const w of wordRows(n, 3 + width)) {
          if (w.length !== 3 + width) throw new StyleError(`molecule ${id}: bad line '${w.join(' ')}' in the ${sec} section of ${name}`);
          const v = w.map(Number);
          if (v.some((x) => !Number.isFinite(x))) throw new StyleError(`molecule ${id}: bad line '${w.join(' ')}' in the ${sec} section of ${name}`);
          const i = v[0];
          if (!Number.isInteger(i) || i < 1 || i > n) throw new StyleError(`molecule ${id}: invalid index ${i} in the ${sec} section of ${name}`);
          // molecule.cpp (line_segments, triangles): "Invalid line molecule ID" for a molecule-ID < 0 and
          // "Invalid line type" for a type <= 0; the type offset is added after the check
          if (v[1] < 0) throw new StyleError(`molecule ${id}: invalid molecule ID ${v[1]} in the ${sec} section of ${name}`);
          if (v[2] <= 0) throw new StyleError(`molecule ${id}: invalid type ${v[2]} in the ${sec} section of ${name}`);
          out[i - 1] = { mol: v[1], type: v[2] + o.toff, p: v.slice(3).map((x) => x * o.scale) };
        }
        const missing = out.findIndex((e) => e === null);
        if (missing >= 0) throw new StyleError(`molecule ${id}: ${sec === 'Lines' ? 'line' : 'triangle'} ${missing + 1} missing in the ${sec} section of ${name}`);
        if (sec === 'Lines') t.lines = out as SurfaceEntry[]; else t.tris = out as SurfaceEntry[];
        break;
      }
      case 'Dipoles': {
        // molecule.html, Dipoles section: "one line per atom"; "line syntax: ID mux muy muz";
        // "mux,muy,muz = x-, y-, and z-component of point dipole vector of atom"
        t.mu = new Float64Array(3 * natoms);
        for (const r of rows(natoms, 4)) { const i = atomIndex(r[0], 'atom ID') - 1; for (let d = 0; d < 3; d++) t.mu[3 * i + d] = r[1 + d] * o.scale; }
        break;
      }
      default:
        throw new StyleError(`molecule ${id}: section '${sec}' in ${name} is not supported by the browser engine`);
    }
  }
  if (bodyHeader) {
    // molecule.html: a body molecule has Masses and both Body sections; the native reader also requires them (molecule.cpp)
    if (!seen.has('Body Integers') || !seen.has('Body Doubles') || !bodyInts || !bodyDbls) throw new StyleError(`molecule ${id}: ${name} needs Body Integers and Body Doubles sections for a body particle`);
    if (!t.mass) throw new StyleError(`molecule ${id}: Molecule file must have Masses section for body particle`);
    t.body = { ints: bodyInts, dbls: bodyDbls };
  }
  if (hasAtoms && !seen.has('Coords')) throw new StyleError(`molecule ${id}: ${name} has no Coords section`);
  if (hasAtoms && !seen.has('Types')) throw new StyleError(`molecule ${id}: ${name} has no Types section`);
  if (head.lines && !seen.has('Lines')) throw new StyleError(`molecule ${id}: ${name} has no Lines section`);
  if (head.triangles && !seen.has('Triangles')) throw new StyleError(`molecule ${id}: ${name} has no Triangles section`);
  for (const [h, list] of [['bonds', t.bonds], ['angles', t.angles], ['dihedrals', t.dihedrals], ['impropers', t.impropers]] as const) {
    if (count(h) !== list.length) throw new StyleError(`molecule ${id}: the header says ${count(h)} ${h} but no ${h[0].toUpperCase()}${h.slice(1)} section was read`);
  }
  if (t.special) checkSpecial(t);
  return t;
};

/*
 * JSON molecule files — docs.lammps.org/molecule.html, "Format of a JSON molecule file": "The format of a JSON
 * format individual molecule file must follow the JSON format, ..."; "Unlike with the native molecule file
 * format, there are no header or body sections, just a list of keywords with associated data."; "Data for
 * keywords is either provided directly following the keyword or as a *data block*. A *data block* is a list
 * that has to include two keywords, "format" and "data", where the former lists keywords of the properties
 * that are stored in the columns of the "data" lists. The names and order of entries in the "format" list
 * (and thus how the data is interpreted) are currently fixed."
 *
 * The keywords map onto the same template fields as the native sections (the table of molecule.html): coords,
 * types, molecules, charges, diameters, masses, dipoles, bonds, angles, dihedrals, impropers, special (counts,
 * bonds) and shake (flags, atoms, bonds). A type given as a number is offset like the native type words; a
 * type given as a string is a type label (molecule.html: "Offsets are ignored on lines using type labels").
 * Keywords that describe the template as a whole (title, schema, units, com, masstotal, inertia) do not change
 * the template as the native text reader does not use them either.
 */
const JSON_IGNORED = new Set(['application', 'format', 'revision', 'title', 'schema', 'units', 'com', 'masstotal', 'inertia']);
const JSON_SECTIONS = new Set(['coords', 'types', 'molecules', 'fragments', 'charges', 'diameters', 'masses', 'dipoles', 'bonds', 'angles', 'dihedrals', 'impropers', 'special', 'shake']);

type JsonRow = unknown[];
interface JsonBlock { format: string[]; data: JsonRow[] }

/** The units a JSON file may name for a system: "real" and "metal" are interchangeable (molecule.html "units"). */
const unitsCompatible = (json: string, system: string): boolean => {
  if (json === 'real' || json === 'metal') return system === 'real' || system === 'metal';
  return json === system;
};

export const parseMoleculeJson = (id: string, name: string, text: string, o: MoleculeOptions, labels?: LabelMap): MoleculeTemplate => {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch (e) {
    throw new StyleError(`molecule ${id}: Error parsing JSON file ${name}: ${(e as Error).message}`);
  }
  if (!doc || typeof doc !== 'object' || Array.isArray(doc)) throw new StyleError(`molecule ${id}: JSON molecule data in ${name} is not an object`);
  const top = doc as Record<string, unknown>;
  const need = (ok: boolean, msg: string) => { if (!ok) throw new StyleError(`molecule ${id}: ${msg}`); };
  need('application' in top, 'JSON data does not contain required "application" field');
  need(top.application === 'LAMMPS', `JSON data is for incompatible application: ${String(top.application)}`);
  need('format' in top, 'JSON data does not contain required "format" field');
  need(top.format === 'molecule', `JSON data is not for a molecule: ${String(top.format)}`);
  need('revision' in top, 'JSON data does not contain required "revision" field');
  need(top.revision === 1, `JSON molecule data with unsupported revision ${String(top.revision)}`);
  if ('units' in top && o.units !== undefined) {
    need(unitsCompatible(String(top.units), o.units), `Incompatible units in JSON molecule data: current = ${o.units}, JSON = ${String(top.units)}`);
  }
  for (const key of Object.keys(top)) {
    if (JSON_IGNORED.has(key) || JSON_SECTIONS.has(key)) continue;
    if (key === 'body') throw new StyleError(`molecule ${id}: '${key}' molecules are not supported by the browser engine`);
    throw new StyleError(`molecule ${id}: JSON section '${key}' in ${name} is not supported by the browser engine`);
  }

  /** A data block: {format, data} with the expected format list (native checks the entries of the format list). */
  const block = (key: string, expect: string[] | null): JsonBlock | null => {
    const b = top[key];
    if (b === undefined) return null;
    need(!!b && typeof b === 'object' && !Array.isArray(b), `JSON data for '${key}' must be a data block`);
    const blk = b as Record<string, unknown>;
    need(Array.isArray(blk.format), `JSON molecule data does not contain required "format" field for '${key}'`);
    need(Array.isArray(blk.data), `JSON molecule data does not contain required "data" field for '${key}'`);
    const format = (blk.format as unknown[]).map(String);
    if (expect) need(format.length === expect.length && format.every((f, i) => f === expect[i]),
      `Expected "${key}" format [${expect.map((x) => `"${x}"`).join(',')}] but found [${format.map((x) => `"${x}"`).join(',')}]`);
    return { format, data: blk.data as JsonRow[] };
  };
  const typesBlock = block('types', ['atom-id', 'type']);
  need(typesBlock !== null, 'JSON data does not contain required "types" field');
  const natoms = typesBlock!.data.length;
  need(natoms > 0, 'No types entries in JSON data for molecule');
  const t: MoleculeTemplate = {
    id, natoms, x: new Float64Array(3 * natoms), type: new Int32Array(natoms), q: null, mol: null,
    bonds: [], angles: [], dihedrals: [], impropers: [], special: null, shake: null, diam: null, mass: null, mu: null,
    lines: null, tris: null, fragments: [],
  };
  const atomIndex = (v: unknown, what: string): number => {
    need(typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= natoms, `invalid ${what} ${String(v)} in molecule JSON data (1..${natoms})`);
    return v as number;
  };
  const num = (v: unknown, what: string): number => {
    need(typeof v === 'number' && Number.isFinite(v), `non-numeric ${what} ${String(v)} in molecule JSON data`);
    return v as number;
  };
  const scale = o.scale;

  // types: a number is offset by toff, a string is a type label of the atom kind
  const seenType = new Set<number>();
  for (const r of typesBlock!.data) {
    need(Array.isArray(r) && r.length >= 2, `missing data in "types" section of molecule JSON data: ${JSON.stringify(r)}`);
    const i = atomIndex(r[0], 'atom-id') - 1;
    need(!seenType.has(i), `atom ${i + 1} has more than one entry in "types"`);
    seenType.add(i);
    t.type[i] = typeOfJson(id, r[1], 'atom', o.toff, labels, 'types');
  }
  need(seenType.size === natoms, 'Molecule template: some atoms have no type');

  const coords = block('coords', ['atom-id', 'x', 'y', 'z']);
  for (const r of coords?.data ?? []) {
    const i = atomIndex(r[0], 'atom-id') - 1;
    for (let d = 0; d < 3; d++) t.x[3 * i + d] = num(r[1 + d], 'coordinate') * scale;
  }
  const mols = block('molecules', ['atom-id', 'molecule-id']);
  if (mols) {
    t.mol = new Int32Array(natoms);
    for (const r of mols.data) t.mol[atomIndex(r[0], 'atom-id') - 1] = num(r[1], 'molecule-id');
  }
  const charges = block('charges', ['atom-id', 'charge']);
  if (charges) {
    t.q = new Float64Array(natoms);
    for (const r of charges.data) t.q[atomIndex(r[0], 'atom-id') - 1] = num(r[1], 'charge');
  }
  const masses = block('masses', ['atom-id', 'mass']);
  if (masses) {
    t.mass = new Float64Array(natoms);
    for (const r of masses.data) t.mass[atomIndex(r[0], 'atom-id') - 1] = num(r[1], 'mass') * scale;
  }
  const frags = block('fragments', ['fragment-id', 'atom-id-list']);
  for (const r of frags?.data ?? []) {
    need(Array.isArray(r) && r.length === 2 && Array.isArray(r[1]), 'invalid format of JSON data for fragments');
    t.fragments.push({ name: String(r[0]), atoms: (r[1] as unknown[]).map((v) => atomIndex(v, 'atom ID in fragment')) });
  }
  const diams = block('diameters', ['atom-id', 'diameter']);
  if (diams) {
    t.diam = new Float64Array(natoms);
    for (const r of diams.data) t.diam[atomIndex(r[0], 'atom-id') - 1] = num(r[1], 'diameter') * scale;
  }
  const dips = block('dipoles', ['atom-id', 'mux', 'muy', 'muz']);
  if (dips) {
    t.mu = new Float64Array(3 * natoms);
    for (const r of dips.data) {
      const i = atomIndex(r[0], 'atom-id') - 1;
      for (let d = 0; d < 3; d++) t.mu[3 * i + d] = num(r[1 + d], 'dipole') * scale;
    }
  }

  // topology: the same entries and checks as the native sections (no ID repeated within an entry)
  const topo: [string, 'bonds' | 'angles' | 'dihedrals' | 'impropers', number, number, LabelKind, string[]][] = [
    ['bonds', 'bonds', 2, o.boff, 'bond', ['bond-type', 'atom1', 'atom2']],
    ['angles', 'angles', 3, o.aoff, 'angle', ['angle-type', 'atom1', 'atom2', 'atom3']],
    ['dihedrals', 'dihedrals', 4, o.doff, 'dihedral', ['dihedral-type', 'atom1', 'atom2', 'atom3', 'atom4']],
    ['impropers', 'impropers', 4, o.ioff, 'improper', ['improper-type', 'atom1', 'atom2', 'atom3', 'atom4']],
  ];
  for (const [key, field, width, off, kind, format] of topo) {
    const b = block(key, format);
    if (!b) continue;
    for (const r of b.data) {
      need(Array.isArray(r) && r.length >= 1 + width, `invalid format of JSON data for ${kind} ${JSON.stringify(r)}`);
      const ids = r.slice(1, 1 + width).map((v) => atomIndex(v, 'atom ID'));
      if (new Set(ids).size !== ids.length) throw new StyleError(`molecule ${id}: Invalid atom ID in ${key} section of ${name}: ${JSON.stringify(r)}`);
      t[field].push([typeOfJson(id, r[0], kind, off, labels, key), ...ids]);
    }
  }

  // special: counts (n12, n13, n14 per atom) and the neighbour lists, as the Special Bond sections
  const special = top.special;
  if (special !== undefined) {
    need(!!special && typeof special === 'object' && !Array.isArray(special), 'JSON data for \'special\' must be an object');
    const sp = special as Record<string, unknown>;
    need('counts' in sp, 'JSON molecule data does not contain required \'counts\' field for "special"');
    need('bonds' in sp, 'JSON molecule data does not contain required "bonds" field for "special"');
    const cblk = { format: ['atom-id', 'n12', 'n13', 'n14'], data: (sp.counts as { data?: JsonRow[] } | undefined)?.data ?? [] };
    need(Array.isArray((sp.counts as { data?: unknown })?.data), 'JSON molecule data does not contain required "data" field for \'special:counts\'');
    need(cblk.data.length === natoms, `Found ${cblk.data.length} instead of ${natoms} data entries for 'special:counts'`);
    const counts = cblk.data.map((r) => {
      need(Array.isArray(r) && r.length === 4, `Found ${Array.isArray(r) ? r.length : 0} instead of 4 data entries for 'special:counts'`);
      return [atomIndex(r[0], 'atom-id'), num(r[1], 'count'), num(r[2], 'count'), num(r[3], 'count')];
    });
    const sb = (sp.bonds as { data?: unknown[] } | undefined)?.data;
    need(Array.isArray(sb), 'JSON molecule data does not contain required "data" field for "special:bonds"');
    need(sb!.length === natoms, `Found ${sb!.length} instead of ${natoms} data entries for "special:bonds"`);
    const byAtom = new Map(counts.map((c) => [c[0], c]));
    t.special = Array.from({ length: natoms }, () => [[], [], []] as number[][]);
    for (const r of sb as unknown[][]) {
      need(Array.isArray(r) && r.length === 2 && Array.isArray(r[1]), '"special:bonds" is incorrectly formatted');
      const i = atomIndex(r[0], 'atom-id');
      const c = byAtom.get(i)!;
      const ids = (r[1] as unknown[]).map((v) => atomIndex(v, 'special neighbor'));
      need(ids.length === c[1] + c[2] + c[3], `atom ${i} lists ${ids.length} special neighbors, the counts say ${c[1] + c[2] + c[3]}`);
      t.special[i - 1] = [ids.slice(0, c[1]), ids.slice(c[1], c[1] + c[2]), ids.slice(c[1] + c[2])];
    }
  }

  // shake: flags, atom lists and type lists (Shake Flags, Shake Atoms, Shake Bond Types)
  const shake = top.shake;
  if (shake !== undefined) {
    need(!!shake && typeof shake === 'object', 'JSON data for \'shake\' must be an object');
    const sk = shake as Record<string, unknown>;
    const sub = (k: string, fmt: string[]): JsonRow[] => {
      const b = sk[k] as { format?: unknown; data?: unknown } | undefined;
      need(!!b && Array.isArray(b.format) && Array.isArray(b.data), `JSON molecule data does not contain required "data" field for "shake:${k}"`);
      need(JSON.stringify((b!.format as unknown[]).map(String)) === JSON.stringify(fmt), `Expected "shake:${k}" format [${fmt.join(',')}]`);
      need((b!.data as unknown[]).length === natoms, `Found ${(b!.data as unknown[]).length} instead of ${natoms} data entries for "shake:${k}"`);
      return b!.data as JsonRow[];
    };
    const fl = sub('flags', ['atom-id', 'flag']), at = sub('atoms', ['atom-id', 'atom-id-list']), ty = sub('bonds', ['atom-id', 'type-list']);
    t.shake = { flags: new Int32Array(natoms), atoms: Array.from({ length: natoms }, () => []), types: Array.from({ length: natoms }, () => []) };
    for (const r of fl) t.shake.flags[atomIndex(r[0], 'atom-id') - 1] = num(r[1], 'shake flag');
    for (const r of at) t.shake.atoms[atomIndex(r[0], 'atom-id') - 1] = (r[1] as unknown[]).map((v) => num(v, 'shake atom'));
    for (const r of ty) t.shake.types[atomIndex(r[0], 'atom-id') - 1] = (r[1] as unknown[]).map((v) => num(v, 'shake type'));
  }
  if (t.special) checkSpecial(t);
  return t;
};

/** A type of a JSON entry: a number (offset) or a label of the kind (no offset). */
const typeOfJson = (id: string, v: unknown, kind: LabelKind, off: number, labels: LabelMap | undefined, where: string): number => {
  if (typeof v === 'number') {
    if (!Number.isInteger(v)) throw new StyleError(`molecule ${id}: bad ${kind} type ${String(v)} in the ${where} section of a JSON molecule file`);
    return v + off;
  }
  if (typeof v === 'string') {
    const n = labels?.typeOf(kind, v);
    if (n === undefined) throw new StyleError(`molecule ${id}: ${kind} type label ${v} is not defined (${where} section of a JSON molecule file)`);
    return n;
  }
  throw new StyleError(`molecule ${id}: invalid ${kind} type ${JSON.stringify(v)} in the ${where} section of a JSON molecule file`);
};

/**
 * The auto keyword (molecule.html): "The *auto* keyword allows the molecule command to generate new angles,
 * dihedrals, and/or impropers, and assign their angle types, dihedral types, and/or improper types. New
 * interactions are discovered by traversing the bond graph defined in the *Bonds* section, and new types are
 * inferred using type label". "Angles are generated from all unique 1-2-3 paths through the bond graph.
 * Dihedrals are generated from all unique 1-2-3-4 paths through the bond graph, as long as there are no
 * duplicate atoms in the dihedral. Impropers are generated from all atoms bonded to exactly three neighbors."
 * The generated entries replace the ones in the file (native: "Generating angles will overwrite angle data from
 * molecule template.").
 */
const autoGenerate = (t: MoleculeTemplate, auto: string, labels: LabelMap | undefined, impSym: readonly number[] | null | undefined): void => {
  if (t.bonds.length === 0) throw new StyleError(`molecule ${t.id}: Unable to generate angles without special bonds.`);
  const map = labels ?? new LabelMap();
  const sp = specialFromBonds(t).map((s) => s[0]);
  const n = t.natoms;
  const type = (i: number) => t.type[i - 1];
  if (auto.includes('a')) {
    t.angles = [];
    for (let b = 1; b <= n; b++) {
      const nb = sp[b - 1];
      for (let i = 0; i < nb.length; i++) {
        for (let j = i + 1; j < nb.length; j++) {
          let a1 = nb[i], a3 = nb[j];
          const s = inferAngleType(map, type(a1), type(b), type(a3));
          if (!s) throw new StyleError(`molecule ${t.id}: Unable to infer angle type from bonds.`);
          if (s < 0) [a1, a3] = [a3, a1];
          t.angles.push([Math.abs(s), a1, b, a3]);
        }
      }
    }
  }
  if (auto.includes('d')) {
    t.dihedrals = [];
    for (let b = 1; b <= n; b++) {
      for (const c of sp[b - 1]) {
        if (c <= b) continue;
        for (const a of sp[b - 1]) {
          if (a === c || a === b) continue;
          for (const d of sp[c - 1]) {
            if (d === c || d === b || d === a) continue;
            const s = inferDihedralType(map, type(a), type(b), type(c), type(d));
            if (!s) throw new StyleError(`molecule ${t.id}: Unable to infer dihedral type from bonds.`);
            t.dihedrals.push(s < 0 ? [-s, d, c, b, a] : [s, a, b, c, d]);
          }
        }
      }
    }
  }
  if (auto.includes('i')) {
    t.impropers = [];
    if (impSym === null) throw new StyleError(`molecule ${t.id}: auto impropers need the symmetry of the improper style, which the browser engine does not provide for this style`);
    const sym = impSym ?? [0, 0, 0, 0];
    for (let b = 1; b <= n; b++) {
      const nb = sp[b - 1];
      if (nb.length !== 3) continue;
      const [a1, a3, a4] = nb;
      const r = inferImproperType(map, [type(a1), type(b), type(a3), type(a4)], sym);
      if (!r.type) throw new StyleError(`molecule ${t.id}: Unable to infer improper type from bonds.`);
      const tags = [a1, b, a3, a4];
      const ord = r.type < 0 ? r.iorder.map((k) => tags[k]) : tags;
      t.impropers.push([Math.abs(r.type), ord[0], ord[1], ord[2], ord[3]]);
    }
  }
};

/**
 * A molecule file, native or JSON (by its .json name), with its auto keyword applied. The reader is chosen by
 * the file name: "JSON format filenames must have the extension .json".
 */
export const parseMoleculeFile = (id: string, name: string, text: string, o: MoleculeOptions, labels?: LabelMap): MoleculeTemplate => {
  const t = /\.json$/i.test(name) ? parseMoleculeJson(id, name, text, o, labels) : parseMoleculeText(id, name, text, o, labels);
  if (o.auto) autoGenerate(t, o.auto, labels, o.improperSymmatoms);
  return t;
};

/**
 * The dipoles of a template turned by the rotation `rot` (3x3, about the molecule center; the dipole
 * vectors turn with the molecule, see create_atoms in setup.ts), flat 3N, or undefined without a Dipoles section.
 */
export const rotateDipoles = (t: MoleculeTemplate, rot: number[][]): number[] | undefined => {
  if (!t.mu) return undefined;
  const out: number[] = [];
  for (let i = 0; i < t.natoms; i++) {
    const m = [t.mu[3 * i], t.mu[3 * i + 1], t.mu[3 * i + 2]];
    for (let d = 0; d < 3; d++) out.push(rot[d][0] * m[0] + rot[d][1] * m[1] + rot[d][2] * m[2]);
  }
  return out;
};

/**
 * Point dipoles of `copies` copies of a template as the atom_style dipole storage (4 per atom:
 * mux, muy, muz and the length |mu|), or undefined without a Dipoles section. `rotated` (optional) holds
 * the dipoles of the copies after their rotation (3 per atom, copy-major). The caller checks that
 * the atom style has dipoles (molecule.html: "This section is only allowed for atom styles that
 * support particles with point dipoles, e.g. atom_style dipole.").
 */
export const templateDipoles = (t: MoleculeTemplate, copies: number, rotated?: number[]): Float64Array | undefined => {
  if (!t.mu) return undefined;
  const out = new Float64Array(4 * copies * t.natoms);
  for (let c = 0; c < copies; c++) {
    for (let i = 0; i < t.natoms; i++) {
      const k = 4 * (c * t.natoms + i), j = 3 * i;
      // rotated: the per-copy dipoles already turned with the molecule (3 per atom of every copy)
      const src = rotated ? 3 * (c * t.natoms + i) : j;
      const mu = rotated ?? t.mu;
      const mx = mu[src], my = mu[src + 1], mz = mu[src + 2];
      out[k] = mx; out[k + 1] = my; out[k + 2] = mz; out[k + 3] = Math.hypot(mx, my, mz);
    }
  }
  return out;
};

/** 1-2, 1-3, 1-4 neighbors from the bonds (sorted template-local IDs). */
export const specialFromBonds = (t: MoleculeTemplate): number[][][] => {
  const nb: Set<number>[] = Array.from({ length: t.natoms + 1 }, () => new Set());
  for (const [, a, b] of t.bonds) { nb[a].add(b); nb[b].add(a); }
  const out: number[][][] = [];
  for (let i = 1; i <= t.natoms; i++) {
    const s12 = [...nb[i]];
    const s13 = new Set<number>();
    for (const j of s12) for (const k of nb[j]) if (k !== i && !nb[i].has(k)) s13.add(k);
    const s14 = new Set<number>();
    for (const j of s13) for (const k of nb[j]) if (k !== i && !nb[i].has(k) && !s13.has(k)) s14.add(k);
    out.push([s12.sort((p, q) => p - q), [...s13].sort((p, q) => p - q), [...s14].sort((p, q) => p - q)]);
  }
  return out;
};

/** The engine derives special neighbors from bonds; explicit lists must agree with that. */
const checkSpecial = (t: MoleculeTemplate): void => {
  const want = specialFromBonds(t);
  for (let i = 0; i < t.natoms; i++) {
    for (let o = 0; o < 3; o++) {
      const a = [...t.special![i][o]].sort((p, q) => p - q).join(' '), b = want[i][o].join(' ');
      if (a !== b) {
        throw new StyleError(`molecule ${t.id}: the Special Bonds of atom ${i + 1} (1-${o + 2}: ${a || 'none'}) differ from those its bonds imply (${b || 'none'}); the browser engine derives special neighbors from bonds`);
      }
    }
  }
};

/** Geometric center (unweighted mean of the coordinates). */
export const geometricCenter = (t: MoleculeTemplate): [number, number, number] => {
  const c: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < t.natoms; i++) for (let d = 0; d < 3; d++) c[d] += t.x[3 * i + d];
  return [c[0] / t.natoms, c[1] / t.natoms, c[2] / t.natoms];
};

/** Rotation matrix for angle theta (radians) about the axis (rx, ry, rz) (right-hand rule). */
export const rotationMatrix = (theta: number, rx: number, ry: number, rz: number): number[][] => {
  const n = Math.hypot(rx, ry, rz);
  if (!(n > 0)) throw new StyleError('rotation vector must be non-zero');
  const [x, y, z] = [rx / n, ry / n, rz / n];
  const c = Math.cos(theta), s = Math.sin(theta), C = 1 - c;
  return [
    [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
  ];
};
