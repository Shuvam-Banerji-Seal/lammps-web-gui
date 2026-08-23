import { MoleculeData } from '../types';

/**
 * Convert parsed molecular data into a LAMMPS data file
 * (atom_style full layout: id mol type q x y z).
 *
 * The bounding box of the atoms becomes the simulation box (with a small
 * margin), unless the structure already carries a box.
 */
export const toLammpsDataFile = (
  data: MoleculeData,
  opts: { title?: string; margin?: number } = {}
): string => {
  const margin = opts.margin ?? 1.0;
  const box = data.box ?? {
    xlo: data.min.x - margin,
    xhi: data.max.x + margin,
    ylo: data.min.y - margin,
    yhi: data.max.y + margin,
    zlo: data.min.z - margin,
    zhi: data.max.z + margin,
  };

  const types = Object.values(data.atomTypes).sort((a, b) => a.id - b.id);
  const hasBonds = data.bonds.length > 0;

  const lines: string[] = [];
  const signature = 'written by Molecule3D — Shuvam Banerji Seal';
  lines.push(opts.title ? `${opts.title} (${signature})` : `LAMMPS data file (${signature})`);
  lines.push('');
  lines.push(`${data.atoms.length} atoms`);
  if (hasBonds) lines.push(`${data.bonds.length} bonds`);
  lines.push('');
  lines.push(`${types.length} atom types`);
  if (hasBonds) lines.push(`${new Set(data.bonds.map(b => b.type)).size} bond types`);
  lines.push('');
  lines.push(`${box.xlo.toFixed(6)} ${box.xhi.toFixed(6)} xlo xhi`);
  lines.push(`${box.ylo.toFixed(6)} ${box.yhi.toFixed(6)} ylo yhi`);
  lines.push(`${box.zlo.toFixed(6)} ${box.zhi.toFixed(6)} zlo zhi`);
  lines.push('');
  lines.push('Masses');
  lines.push('');
  for (const t of types) {
    const mass = t.mass > 0 ? t.mass : 12.011;
    lines.push(`${t.id} ${mass.toFixed(4)} # ${t.element}`);
  }
  lines.push('');
  lines.push('Atoms # full');
  lines.push('');
  for (const a of data.atoms) {
    lines.push(
      `${a.id} ${a.molId || 1} ${a.type} ${a.charge ?? 0} ` +
      `${a.x.toFixed(6)} ${a.y.toFixed(6)} ${a.z.toFixed(6)}`
    );
  }
  if (hasBonds) {
    lines.push('');
    lines.push('Bonds');
    lines.push('');
    for (const b of data.bonds) {
      lines.push(`${b.id} ${b.type} ${b.atom1Id} ${b.atom2Id}`);
    }
  }
  lines.push('');
  return lines.join('\n');
};

/** Download helper shared by script/data exporters. */
export const downloadTextFile = (filename: string, text: string): void => {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};
