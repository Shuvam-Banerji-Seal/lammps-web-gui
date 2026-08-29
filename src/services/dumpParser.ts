import { Atom, MoleculeData, AtomTypeInfo, BoxBounds, TrajectoryFrame } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';
import { inferBonds } from './bondInference';

/**
 * Parser for native LAMMPS dump trajectories (dump custom / dump atom text
 * format, .lammpstrj / .dump).
 *
 * Format reference [VERIFIED 2026-08-23, docs.lammps.org git 4Jul2026]:
 *   ITEM: TIMESTEP / NUMBER OF ATOMS / BOX BOUNDS / ATOMS <columns...>
 *
 * Handled:
 *  - Orthogonal boxes ("ITEM: BOX BOUNDS pp pp pp") → lo/hi as-is.
 *  - Restricted triclinic boxes ("ITEM: BOX BOUNDS xy xz yz ...") → the file
 *    stores the orthogonal BOUNDING box; converted back to the true box via
 *    the documented inverse:
 *      xlo = xlo_bound - MIN(0, xy, xz, xy+xz)
 *      xhi = xhi_bound - MAX(0, xy, xz, xy+xz)
 *      ylo = ylo_bound - MIN(0, yz) ; yhi = yhi_bound - MAX(0, yz)
 *  - Coordinate columns: prefer x/y/z, then xu/yu/zu (unwrapped), then
 *    xs/ys/zs (fractional). Fractional → Cartesian uses the restricted
 *    triclinic basis: x = xlo + xs·lx + ys·xy + zs·xz (and cyclic), which
 *    also degenerates correctly for orthogonal boxes.
 *  - `element` column → symbol lookup; else type-as-atomic-number heuristic
 *    (types 1..118), matching the .data parser's fallback.
 *  - q / mol columns honored when present; arbitrary extra columns ignored.
 *
 * Multi-frame: every block becomes a TrajectoryFrame; the FIRST frame
 * defines atoms/box/type metadata (topology assumed stable), min/max/center
 * span ALL frames so camera framing never jumps during playback.
 */

const isInt = (s: string) => /^-?\d+$/.test(s);
const isFloat = (s: string) => /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s);

/**
 * Convert a dump's triclinic BOUNDING box + tilts into the true box used by
 * the renderer (LAMMPS data-file convention). Inverse of the formulas on
 * docs.lammps.org/Howto_triclinic.html.
 */
export const boundBoxToTriclinic = (
  xBound: [number, number],
  yBound: [number, number],
  zBound: [number, number],
  xy: number,
  xz: number,
  yz: number,
): BoxBounds => {
  const xShiftLo = Math.min(0, xy, xz, xy + xz);
  const xShiftHi = Math.max(0, xy, xz, xy + xz);
  const yShiftLo = Math.min(0, yz);
  const yShiftHi = Math.max(0, yz);
  return {
    xlo: xBound[0] - xShiftLo,
    xhi: xBound[1] - xShiftHi,
    ylo: yBound[0] - yShiftLo,
    yhi: yBound[1] - yShiftHi,
    zlo: zBound[0],
    zhi: zBound[1],
    xy, xz, yz,
  };
};

interface ParsedFrame {
  timestep: number;
  atoms: Atom[];
  box?: BoxBounds;
}

/** Resolve one atom row given the column index map. */
const rowToAtom = (
  tokens: string[],
  col: Map<string, number>,
  frameBox: BoxBounds | undefined,
  fallbackId: number,
): Atom | null => {
  const num = (name: string): number | undefined => {
    const i = col.get(name);
    if (i === undefined || i >= tokens.length) return undefined;
    const v = parseFloat(tokens[i]);
    return Number.isFinite(v) ? v : undefined;
  };

  let x = num('x') ?? num('xu');
  let y = num('y') ?? num('yu');
  let z = num('z') ?? num('zu');

  const xs = num('xs'), ys = num('ys'), zs = num('zs');
  if (x === undefined && xs !== undefined && ys !== undefined && zs !== undefined && frameBox) {
    const lx = frameBox.xhi - frameBox.xlo;
    const ly = frameBox.yhi - frameBox.ylo;
    const lz = frameBox.zhi - frameBox.zlo;
    const xy = frameBox.xy ?? 0, xz = frameBox.xz ?? 0, yz = frameBox.yz ?? 0;
    x = frameBox.xlo + xs * lx + ys * xy + zs * xz;
    y = frameBox.ylo + ys * ly + zs * yz;
    z = frameBox.zlo + zs * lz;
  }
  if (x === undefined || y === undefined || z === undefined) return null;

  const idRaw = num('id');
  const typeRaw = num('type');
  const type = typeRaw !== undefined && typeRaw >= 1 ? Math.round(typeRaw) : 1;
  const id = idRaw !== undefined ? Math.round(idRaw) : fallbackId;
  const mol = num('mol');
  const q = num('q');
  const vx = num('vx');
  const vy = num('vy');
  const vz = num('vz');
  let symbol: string | undefined;
  const elemIdx = col.get('element');
  if (elemIdx !== undefined && elemIdx < tokens.length) {
    const norm = tokens[elemIdx].trim();
    const z = getAtomicNumberFromSymbol(norm);
    if (z !== undefined) symbol = ELEMENT_DATA[z - 1].symbol;
  }
  if (!symbol && type >= 1 && type <= 118) symbol = ELEMENT_DATA[type - 1].symbol;

  return {
    id,
    molId: mol !== undefined ? Math.round(mol) : 1,
    type: symbol ? (getAtomicNumberFromSymbol(symbol) ?? type) : type,
    charge: q ?? 0,
    x, y, z,
    ...(vx !== undefined ? { vx } : {}),
    ...(vy !== undefined ? { vy } : {}),
    ...(vz !== undefined ? { vz } : {}),
  };
};

export const parseDumpFile = (data: string): MoleculeData => {
  if (!data.trim()) throw new Error('Invalid LAMMPS dump: file is empty');

  const lines = data.split('\n');
  const frames: ParsedFrame[] = [];

  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trim();
    if (!line.startsWith('ITEM: TIMESTEP')) { i++; continue; }

    // TIMESTEP
    i++;
    const tsLine = lines[i]?.trim();
    const timestep = tsLine !== undefined && isInt(tsLine) ? parseInt(tsLine, 10) : NaN;
    i++;

    // NUMBER OF ATOMS
    while (i < lines.length && !lines[i].trim().startsWith('ITEM:')) i++;
    if (i >= lines.length) break;
    if (!lines[i].trim().startsWith('ITEM: NUMBER OF ATOMS')) continue;
    i++;
    const countLine = lines[i]?.trim();
    const numAtoms = countLine !== undefined && isInt(countLine) ? parseInt(countLine, 10) : NaN;
    i++;
    if (!Number.isInteger(numAtoms) || numAtoms <= 0) continue;

    // BOX BOUNDS (header may carry boundary flags and/or xy xz yz)
    while (i < lines.length && !lines[i].trim().startsWith('ITEM:')) i++;
    if (i >= lines.length) break;
    let frameBox: BoxBounds | undefined;
    if (lines[i].trim().startsWith('ITEM: BOX BOUNDS')) {
      const hasTilt = /\bxy\b/.test(lines[i]);
      i++;
      const readBound = (): [number, number, number] | null => {
        const t = lines[i]?.trim().split(/\s+/).map(Number) ?? [];
        i++;
        return t.length >= 2 && t.slice(0, 3).every(Number.isFinite)
          ? [t[0], t[1], t[2] ?? 0]
          : null;
      };
      const xb = readBound(), yb = readBound(), zb = readBound();
      if (xb && yb && zb) {
        frameBox = hasTilt
          ? boundBoxToTriclinic(
              [xb[0], xb[1]], [yb[0], yb[1]], [zb[0], zb[1]],
              xb[2], yb[2], zb[2],
            )
          : { xlo: xb[0], xhi: xb[1], ylo: yb[0], yhi: yb[1], zlo: zb[0], zhi: zb[1] };
      }
    }

    // ATOMS header with column names
    while (i < lines.length && !lines[i].trim().startsWith('ITEM:')) i++;
    if (i >= lines.length || !lines[i].trim().startsWith('ITEM: ATOMS')) continue;
    const columns = lines[i].trim().split(/\s+/).slice(2);
    const col = new Map<string, number>();
    columns.forEach((name, idx) => { if (!col.has(name)) col.set(name, idx); });
    const hasCoords = ['x', 'y', 'z', 'xu', 'yu', 'zu', 'xs', 'ys', 'zs']
      .some(c => col.has(c));
    if (!hasCoords) {
      throw new Error(
        'Invalid LAMMPS dump: no coordinate columns (x/y/z, xu/yu/zu or xs/ys/zs) in ITEM: ATOMS header',
      );
    }
    i++;

    const atoms: Atom[] = [];
    while (i < lines.length && atoms.length < numAtoms) {
      const row = lines[i].trim();
      if (!row) { i++; continue; }
      if (row.startsWith('ITEM:')) break; // truncated frame
      const tokens = row.split(/\s+/);
      const atom = rowToAtom(tokens, col, frameBox, atoms.length + 1);
      if (atom) atoms.push(atom);
      i++;
    }

    if (atoms.length < numAtoms) {
      if (frames.length === 0) {
        throw new Error(
          `Invalid LAMMPS dump: expected ${numAtoms} atoms in the first frame but found ${atoms.length}`,
        );
      }
      break; // truncated trailing frame — drop silently (xyz-parser policy)
    }

    frames.push({ timestep, atoms, box: frameBox });
  }

  const first = frames[0];
  if (!first) throw new Error('Invalid LAMMPS dump: no ITEM: TIMESTEP frames found');
  if (!first.box) throw new Error('Invalid LAMMPS dump: first frame lacks ITEM: BOX BOUNDS');

  // --- Type metadata from the reference frame ---
  const atomTypes: Record<number, AtomTypeInfo> = {};
  const usedTypes = Array.from(new Set(first.atoms.map(a => a.type)));
  for (const type of usedTypes) {
    const elem = ELEMENT_DATA.find(e => e.number === type);
    let count = 0;
    for (const a of first.atoms) if (a.type === type) count++;
    atomTypes[type] = {
      id: type,
      mass: elem?.mass ?? 0,
      element: elem?.symbol ?? 'X',
      label: elem ? `${elem.name} (${elem.symbol})` : `Type ${type}`,
      count,
    };
  }

  // --- Trajectory frames (comment carries the timestep for the scrubber) ---
  const trajFrames: TrajectoryFrame[] = frames.map(f => ({
    comment: Number.isFinite(f.timestep) ? `timestep ${f.timestep}` : undefined,
    atoms: f.atoms,
  }));

  // --- Bonds from the reference frame (dumps carry no topology) ---
  const bonds = first.atoms.length <= 30000 ? inferBonds(first.atoms) : [];

  // --- Extents across ALL frames (atoms only, consistent with the other
  // parsers — the canvas factors the box in separately for framing) ---
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const f of frames) {
    for (const a of f.atoms) {
      minX = Math.min(minX, a.x); maxX = Math.max(maxX, a.x);
      minY = Math.min(minY, a.y); maxY = Math.max(maxY, a.y);
      minZ = Math.min(minZ, a.z); maxZ = Math.max(maxZ, a.z);
    }
  }

  const center = {
    x: (minX + maxX) / 2,
    y: (minY + maxY) / 2,
    z: (minZ + maxZ) / 2,
  };

  return {
    atoms: first.atoms,
    bonds,
    atomTypes,
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center,
    box: first.box,
    ...(trajFrames.length > 1 ? { frames: trajFrames } : {}),
  };
};
