import { BoxBounds, MoleculeData } from '../types';

/**
 * Convert parsed molecular data into a LAMMPS data file
 * (atom_style full layout: id mol type q x y z).
 *
 * The bounding box of the atoms becomes the simulation box (with a small
 * margin), unless the structure already carries a box.
 *
 * Atom and bond types are renumbered 1..N in increasing order of the
 * structure's own type ids (a PDB or CIF import numbers types by atomic
 * number). docs.lammps.org/read_data.html: "atom types = # of atom types in
 * system". Measured with native LAMMPS: "2 atom types" with a Masses line for
 * type 8 is ERROR: Invalid atom type 8 in Masses section of data file.
 */
export const toLammpsDataFile = (
  data: MoleculeData,
  opts: { title?: string; margin?: number; box?: BoxBounds } = {}
): string => {
  const margin = opts.margin ?? 1.0;
  const box: BoxBounds = opts.box ?? data.box ?? {
    xlo: data.min.x - margin,
    xhi: data.max.x + margin,
    ylo: data.min.y - margin,
    yhi: data.max.y + margin,
    zlo: data.min.z - margin,
    zhi: data.max.z + margin,
  };

  const types = Object.values(data.atomTypes).sort((a, b) => a.id - b.id);
  const typeOf = new Map(types.map((t, k) => [t.id, k + 1]));
  const hasBonds = data.bonds.length > 0;
  const bondTypeOf = new Map([...new Set(data.bonds.map(b => b.type))].sort((a, b) => a - b).map((t, k) => [t, k + 1]));

  const lines: string[] = [];
  const signature = 'written by LAMMPS Web GUI — Shuvam Banerji Seal';
  lines.push(opts.title ? `${opts.title} (${signature})` : `LAMMPS data file (${signature})`);
  lines.push('');
  lines.push(`${data.atoms.length} atoms`);
  if (hasBonds) lines.push(`${data.bonds.length} bonds`);
  lines.push('');
  lines.push(`${types.length} atom types`);
  if (hasBonds) lines.push(`${bondTypeOf.size} bond types`);
  lines.push('');
  lines.push(`${box.xlo.toFixed(6)} ${box.xhi.toFixed(6)} xlo xhi`);
  lines.push(`${box.ylo.toFixed(6)} ${box.yhi.toFixed(6)} ylo yhi`);
  lines.push(`${box.zlo.toFixed(6)} ${box.zhi.toFixed(6)} zlo zhi`);
  // docs.lammps.org/read_data.html: "xy xz yz = simulation box tilt factors for triclinic system (3 values)",
  // in "a line like \"0.0 5.0 6.0 xy xz yz\""
  if (box.xy || box.xz || box.yz) lines.push(`${(box.xy ?? 0).toFixed(6)} ${(box.xz ?? 0).toFixed(6)} ${(box.yz ?? 0).toFixed(6)} xy xz yz`);
  lines.push('');
  lines.push('Masses');
  lines.push('');
  for (const t of types) {
    const mass = t.mass > 0 ? t.mass : 12.011;
    lines.push(`${typeOf.get(t.id)} ${mass.toFixed(4)} # ${t.element}`);
  }
  lines.push('');
  lines.push('Atoms # full');
  lines.push('');
  for (const a of data.atoms) {
    lines.push(
      `${a.id} ${a.molId || 1} ${typeOf.get(a.type) ?? a.type} ${a.charge ?? 0} ` +
      `${a.x.toFixed(6)} ${a.y.toFixed(6)} ${a.z.toFixed(6)}`
    );
  }
  if (hasBonds) {
    lines.push('');
    lines.push('Bonds');
    lines.push('');
    for (const b of data.bonds) {
      lines.push(`${b.id} ${bondTypeOf.get(b.type)} ${b.atom1Id} ${b.atom2Id}`);
    }
  }
  lines.push('');
  return lines.join('\n');
};

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

/** Download helper shared by script/data exporters. */
/**
 * A file the engine stored as a base64 data URL (dump image writes its PNG / JPEG / PPM frames this way,
 * because the engine's file store holds text): its MIME type and bytes, or null for an ordinary text file.
 */
export const dataUrlBytes = (text: string): { mime: string; bytes: Uint8Array<ArrayBuffer> } | null => {
  const m = /^data:([\w.+/-]+);base64,/.exec(text);
  if (!m) return null;
  const bin = atob(text.slice(m[0].length));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return { mime: m[1], bytes };
};

/** True for an image stored as a data URL (previewable in an <img>). */
export const isImageDataUrl = (text: string): boolean => /^data:image\/[\w.+-]+;base64,/.test(text);

/** Download an engine file: data URLs as their binary content, everything else as text. */
export const downloadFile = (filename: string, content: string): void => {
  // A .gz file from the engine is a string with one character per byte (gzip.ts); save its bytes, not UTF-8 text.
  if (filename.endsWith('.gz')) {
    const bytes = new Uint8Array(content.length);
    for (let k = 0; k < content.length; k++) bytes[k] = content.charCodeAt(k) & 0xff;
    const url = URL.createObjectURL(new Blob([bytes], { type: 'application/gzip' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 4000);
    return;
  }
  const bin = dataUrlBytes(content);
  if (!bin) { downloadTextFile(filename, content); return; }
  const url = URL.createObjectURL(new Blob([bin.bytes], { type: bin.mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};

export const downloadTextFile = (filename: string, text: string): void => {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};
