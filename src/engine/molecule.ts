import { StyleError } from './force/types';

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
 * individual diameters of each particle (Diameters ...)". Fragments, Dipoles
 * and Body sections are rejected.
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
}

export interface MoleculeOptions {
  toff: number; boff: number; aoff: number; doff: number; ioff: number; scale: number;
}

const HEADER: Record<string, number> = { atoms: 1, bonds: 1, angles: 1, dihedrals: 1, impropers: 1, fragments: 1, body: 2, mass: 1, com: 3, inertia: 6 };

const strip = (line: string): string => {
  // "There must be at least one blank between any valid content and the comment."
  const k = line.search(/(^|\s)#/);
  return (k < 0 ? line : line.slice(0, k)).trim();
};

export const parseMoleculeFile = (id: string, name: string, text: string, o: MoleculeOptions): MoleculeTemplate => {
  if (/\.json$/i.test(name)) throw new StyleError(`molecule ${id}: JSON molecule files (${name}) are not supported by the browser engine; use the native format`);
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
  const natoms = head.atoms?.[0];
  if (!natoms || natoms < 1 || !Number.isInteger(natoms)) throw new StyleError(`molecule ${id}: ${name} needs an "N atoms" header line`);
  for (const h of ['fragments', 'body']) if (head[h] && head[h].some((v) => v !== 0)) throw new StyleError(`molecule ${id}: '${h}' molecules are not supported by the browser engine`);
  const count = (h: string) => head[h]?.[0] ?? 0;
  const t: MoleculeTemplate = {
    id, natoms, x: new Float64Array(3 * natoms), type: new Int32Array(natoms), q: null, mol: null,
    bonds: [], angles: [], dihedrals: [], impropers: [], special: null, shake: null, diam: null, mass: null,
  };
  const seen = new Set<string>();
  let specialCounts: number[][] | null = null;
  // body
  while (k < lines.length) {
    const sec = strip(lines[k]);
    if (!sec) { k++; continue; }
    k += 2; // the keyword line and the skipped line after it
    const rows = (n: number, width: number): number[][] => {
      const out: number[][] = [];
      for (; out.length < n && k < lines.length; k++) {
        const l = strip(lines[k]);
        if (!l) continue;
        const w = l.split(/\s+/).map(Number);
        if (w.length < width || w.slice(0, width).some((v) => !Number.isFinite(v))) throw new StyleError(`molecule ${id}: bad line '${l}' in the ${sec} section of ${name}`);
        out.push(w);
      }
      if (out.length < n) throw new StyleError(`molecule ${id}: the ${sec} section of ${name} has ${out.length} of ${n} lines`);
      return out;
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
        for (const r of rows(natoms, 2)) t.type[atomIndex(r[0], 'atom ID') - 1] = r[1] + o.toff;
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
        for (const r of rows(count(sec.toLowerCase()), 2 + width)) list.push([r[1] + off, ...r.slice(2, 2 + width).map((v) => atomIndex(v, 'atom ID'))]);
        break;
      }
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
      case 'Diameters': {
        t.diam = new Float64Array(natoms);
        for (const r of rows(natoms, 2)) t.diam[atomIndex(r[0], 'atom ID') - 1] = r[1] * o.scale;
        break;
      }
      default:
        throw new StyleError(`molecule ${id}: section '${sec}' in ${name} is not supported by the browser engine`);
    }
  }
  if (!seen.has('Coords')) throw new StyleError(`molecule ${id}: ${name} has no Coords section`);
  if (!seen.has('Types')) throw new StyleError(`molecule ${id}: ${name} has no Types section`);
  for (const [h, list] of [['bonds', t.bonds], ['angles', t.angles], ['dihedrals', t.dihedrals], ['impropers', t.impropers]] as const) {
    if (count(h) !== list.length) throw new StyleError(`molecule ${id}: the header says ${count(h)} ${h} but no ${h[0].toUpperCase()}${h.slice(1)} section was read`);
  }
  if (t.special) checkSpecial(t);
  return t;
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
