/**
 * Structure export of the Structure Viewer (loaded when an export is chosen): the shown frame as a LAMMPS data file,
 * extended XYZ or PDB, and every frame as extended XYZ.
 */
import type { BoxBounds, MoleculeData } from '../types';

// the LAMMPS data writer stays in exporter.ts, with its signature line
export { toLammpsDataFile } from './exporter';

/** The element symbol of an atom type, or null when the file did not name one ('X' or a non-element label). */
const elementOf = (data: MoleculeData, type: number): string | null => {
  const e = data.atomTypes[type]?.element;
  return e && e !== 'X' && /^[A-Z][a-z]?$/.test(e) ? e : null;
};

const fmt = (v: number): string => String(Number(v.toPrecision(9)));

/**
 * Extended XYZ of the atoms of `data` (one frame): the atom count, a comment line with Lattice (the cell vectors, tilt
 * included) and Origin when a box is known, Properties=species:S:1:pos:R:3, then one line per atom. The species is
 * the element of the atom's type, or T<type> when the file named none.
 */
export const toExtXyz = (data: MoleculeData, box?: BoxBounds, comment = ''): string => {
  const b = box ?? data.box;
  let head = 'Properties=species:S:1:pos:R:3';
  if (b) {
    const lx = b.xhi - b.xlo, ly = b.yhi - b.ylo, lz = b.zhi - b.zlo;
    head = `Lattice="${fmt(lx)} 0 0 ${fmt(b.xy ?? 0)} ${fmt(ly)} 0 ${fmt(b.xz ?? 0)} ${fmt(b.yz ?? 0)} ${fmt(lz)}" ` +
      `Origin="${fmt(b.xlo)} ${fmt(b.ylo)} ${fmt(b.zlo)}" ${head}`;
  }
  const lines = [String(data.atoms.length), comment ? `${head} ${comment}` : head];
  for (const a of data.atoms) lines.push(`${elementOf(data, a.type) ?? `T${a.type}`} ${fmt(a.x)} ${fmt(a.y)} ${fmt(a.z)}`);
  return lines.join('\n') + '\n';
};

/** Every frame of a trajectory as extended XYZ (each frame with its own cell when it has one). */
export const framesToExtXyz = (data: MoleculeData): string => {
  const frames = data.frames && data.frames.length > 1 ? data.frames : [{ atoms: data.atoms, box: data.box }];
  return frames.map((f, k) => toExtXyz({ ...data, atoms: f.atoms }, f.box ?? data.box, `Frame=${k}`)).join('');
};

/**
 * PDB of the atoms of `data` (one model). CRYST1 (cell lengths, 90-degree angles, P 1) is written when the box is
 * orthogonal; HETATM records carry the element of the type (columns 77-78, and the atom name) or, when the file
 * named none, the name T<type> with blank element columns. Serials wrap after 99999, the PDB limit. Columns follow
 * the wwPDB format guide (CRYST1 1-70, HETATM 1-78).
 */
export const toPdb = (data: MoleculeData, box?: BoxBounds): string => {
  const b = box ?? data.box;
  const col = (v: number, w: number, d: number) => v.toFixed(d).padStart(w);
  const lines: string[] = [];
  if (b && !b.xy && !b.xz && !b.yz) {
    lines.push('CRYST1' + col(b.xhi - b.xlo, 9, 3) + col(b.yhi - b.ylo, 9, 3) + col(b.zhi - b.zlo, 9, 3) +
      col(90, 7, 2) + col(90, 7, 2) + col(90, 7, 2) + ' ' + 'P 1'.padEnd(11) + '1'.padStart(4));
  }
  data.atoms.forEach((a, i) => {
    const el = elementOf(data, a.type);
    const name = el ? (el.length === 1 ? ` ${el}` : el).padEnd(4) : `T${a.type}`.slice(0, 4).padEnd(4);
    lines.push('HETATM' + String((i % 99999) + 1).padStart(5) + ' ' + name + ' MOL A' + String(a.molId || 1).slice(-4).padStart(4) +
      '    ' + col(a.x, 8, 3) + col(a.y, 8, 3) + col(a.z, 8, 3) + '  1.00  0.00' + ' '.repeat(10) + (el ?? '').padStart(2));
  });
  lines.push('END');
  return lines.join('\n') + '\n';
};
