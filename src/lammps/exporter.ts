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
  // A .gz, .dcd or .xtc file from the engine is a string with one character per byte (gzip.ts, dump dcd and xtc); save
  // its bytes, not UTF-8 text.
  if (/\.(gz|dcd|xtc)$/.test(filename)) {
    const bytes = new Uint8Array(content.length);
    for (let k = 0; k < content.length; k++) bytes[k] = content.charCodeAt(k) & 0xff;
    const url = URL.createObjectURL(new Blob([bytes], { type: filename.endsWith('.gz') ? 'application/gzip' : 'application/octet-stream' }));
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
