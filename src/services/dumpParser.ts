import { Atom, MoleculeData, AtomTypeInfo, BoxBounds, TrajectoryFrame } from '../types';
import { ELEMENT_DATA, getAtomicNumberFromSymbol } from '../constants';
import { inferBonds } from './bondInference';
import { atomicNumberByType } from './elementTypes';
import { resolveElementFromMass } from './parser';

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
 *  - Coordinate columns [VERIFIED 2026-09-22, docs.lammps.org/dump.html]:
 *    x/y/z (wrapped), xu/yu/zu (unwrapped), xs/ys/zs (scaled to 0..1) and
 *    xsu/ysu/zsu ("unwrapped coordinates scaled by the box size"). Scaled →
 *    Cartesian uses the restricted triclinic basis
 *      x = xlo + xs·lx + ys·xy + zs·xz   (and cyclic),
 *    which degenerates correctly for orthogonal boxes.
 *
 *    x/y/z is preferred for RENDERING because it is wrapped: an unwrapped
 *    diffusing system scatters across box images and looks broken.
 *  - Image flags ix/iy/iz are captured separately when present, and ONLY
 *    alongside a wrapped position. Rendering keeps the wrapped coordinate;
 *    MSD unwraps with the flags, which is what lets it exceed (L/2)² and show
 *    real linear diffusion. A dump carrying both `xu yu zu` and `ix iy iz`
 *    must NOT pass the flags on — the coordinate is already absolute, so
 *    applying them again double-counts the box offset.
 *  - Elements. A dump names an atom's element only through an `element`
 *    column (dump_modify element), a type label that is an element symbol
 *    (dump_modify types labels) or a `mass` column (nearest element within
 *    0.5 u, the .data parser's rule). With an `element` column the atom's
 *    type is that element's atomic number (one type per element, as before).
 *    Otherwise the atom keeps its LAMMPS type and the type's element is
 *    unknown ('X'): a type NUMBER is never read as an atomic number (types
 *    are numbered in the input's own order, so type 2 is as likely H as He;
 *    reading it as Z drew hydrogen as helium and nitrogen as lithium). The
 *    viewer's Elements tab assigns an element to such a type.
 *  - Type labels (a non-numeric `type` column) number the types in the order
 *    of first appearance; the label names the type.
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
  /** False when the coordinate columns were the unwrapped ones. */
  wrapped: boolean;
}

/**
 * The element symbol a token names. An element column (dump_modify element) holds element names in any case. A type
 * label names an element only when it is a symbol in its own capitalization ("H", "Na", "Cl"): a label such as "CA"
 * (an alpha carbon in a protein input) or "ca" (an aromatic carbon type) is not calcium.
 */
const symbolOfToken = (token: string, labelCase: boolean): string | undefined => {
  const t = token.trim();
  const z = getAtomicNumberFromSymbol(t);
  if (z === undefined) return undefined;
  const sym = ELEMENT_DATA[z - 1].symbol;
  return !labelCase || t === sym ? sym : undefined;
};

/** What the parser learns about each atom type over the rows (the element and mass of the type, its label). */
interface TypeTable {
  /** type label (non-numeric `type` token) -> type id, numbered in order of first appearance */
  labelIds: Map<string, number>;
  /** type id -> the label that named it */
  labels: Map<number, string>;
  /** type id -> element symbol, when the file names it */
  elements: Map<number, string>;
  /** type id -> the mass column's value for its first atom */
  masses: Map<number, number>;
  /** element-column tokens that are not element symbols -> their synthetic type ids */
  unknownElementIds: Map<string, number>;
}

const newTypeTable = (): TypeTable => ({ labelIds: new Map(), labels: new Map(), elements: new Map(), masses: new Map(), unknownElementIds: new Map() });

/** Synthetic type ids (an element-column token that is no element) start here, clear of atomic numbers. */
const SYNTHETIC_TYPE_BASE = 1000;

/** Resolve one atom row given the column index map. */
const rowToAtom = (
  tokens: string[],
  col: Map<string, number>,
  frameBox: BoxBounds | undefined,
  fallbackId: number,
  types: TypeTable,
): { atom: Atom; wrapped: boolean } | null => {
  const num = (name: string): number | undefined => {
    const i = col.get(name);
    if (i === undefined || i >= tokens.length) return undefined;
    const v = parseFloat(tokens[i]);
    return Number.isFinite(v) ? v : undefined;
  };

  /*
   * Resolve the position, tracking whether what we store is WRAPPED.
   *
   * That flag decides whether the image flags may be attached at all. A dump
   * can legitimately carry `xu yu zu` AND `ix iy iz`; in that case the
   * coordinate is already absolute, so handing the flags downstream would
   * make MSD add the same box offset a second time — a 4x overestimate on a
   * steady drift. The invariant this function guarantees is therefore:
   *
   *   ix/iy/iz are present on an Atom ONLY IF x/y/z are wrapped.
   */
  let x: number | undefined;
  let y: number | undefined;
  let z: number | undefined;
  let wrapped = false;

  const xw = num('x'), yw = num('y'), zw = num('z');
  if (xw !== undefined && yw !== undefined && zw !== undefined) {
    x = xw; y = yw; z = zw;
    wrapped = true;
  } else {
    const xu = num('xu'), yu = num('yu'), zu = num('zu');
    if (xu !== undefined && yu !== undefined && zu !== undefined) {
      x = xu; y = yu; z = zu;          // already unwrapped
    } else if (frameBox) {
      // Scaled columns, wrapped (xs) or unwrapped (xsu). Both map through the
      // same restricted-triclinic basis; xsu values simply run outside 0..1.
      const xs = num('xs'), ys = num('ys'), zs = num('zs');
      const scaledWrapped = xs !== undefined && ys !== undefined && zs !== undefined;
      const sx = scaledWrapped ? xs : num('xsu');
      const sy = scaledWrapped ? ys : num('ysu');
      const sz = scaledWrapped ? zs : num('zsu');
      if (sx !== undefined && sy !== undefined && sz !== undefined) {
        const lx = frameBox.xhi - frameBox.xlo;
        const ly = frameBox.yhi - frameBox.ylo;
        const lz = frameBox.zhi - frameBox.zlo;
        const xy = frameBox.xy ?? 0, xz = frameBox.xz ?? 0, yz = frameBox.yz ?? 0;
        x = frameBox.xlo + sx * lx + sy * xy + sz * xz;
        y = frameBox.ylo + sy * ly + sz * yz;
        z = frameBox.zlo + sz * lz;
        wrapped = scaledWrapped;
      }
    }
  }
  if (x === undefined || y === undefined || z === undefined) return null;

  // Only meaningful alongside a wrapped position — see the note above.
  const ix = wrapped ? num('ix') : undefined;
  const iy = wrapped ? num('iy') : undefined;
  const iz = wrapped ? num('iz') : undefined;

  const idRaw = num('id');
  const id = idRaw !== undefined ? Math.round(idRaw) : fallbackId;
  const mol = num('mol');
  const q = num('q');
  const vx = num('vx');
  const vy = num('vy');
  const vz = num('vz');

  // The LAMMPS type: a number, or a type label (dump_modify types labels) numbered in order of first appearance.
  let type = 1;
  const typeIdx = col.get('type');
  if (typeIdx !== undefined && typeIdx < tokens.length) {
    const tok = tokens[typeIdx];
    if (isInt(tok)) {
      const t = parseInt(tok, 10);
      if (t >= 1) type = t;
    } else if (isFloat(tok)) {
      const t = Math.round(parseFloat(tok));
      if (t >= 1) type = t;
    } else {
      let t = types.labelIds.get(tok);
      if (t === undefined) {
        t = types.labelIds.size + 1;
        types.labelIds.set(tok, t);
        types.labels.set(t, tok);
        const sym = symbolOfToken(tok, true);
        if (sym) types.elements.set(t, sym);
      }
      type = t;
    }
  }

  // An element column names the element: the type becomes that element's atomic number (one type per element). A
  // token that is no element symbol gets a synthetic type of its own, named by the token.
  const elemIdx = col.get('element');
  if (elemIdx !== undefined && elemIdx < tokens.length) {
    const tok = tokens[elemIdx].trim();
    const sym = symbolOfToken(tok, false);
    if (sym) {
      // `atomicNumber`, not `z` — the outer `z` here is a COORDINATE.
      const atomicNumber = getAtomicNumberFromSymbol(sym)!;
      type = atomicNumber;
      types.elements.set(type, sym);
    } else {
      let t = types.unknownElementIds.get(tok);
      if (t === undefined) {
        t = SYNTHETIC_TYPE_BASE + types.unknownElementIds.size;
        types.unknownElementIds.set(tok, t);
        types.labels.set(t, tok);
      }
      type = t;
    }
  }
  const mass = num('mass');
  if (mass !== undefined && !types.masses.has(type)) types.masses.set(type, mass);

  const atom: Atom = {
    id,
    molId: mol !== undefined ? Math.round(mol) : 1,
    type,
    charge: q ?? 0,
    x, y, z,
    ...(vx !== undefined ? { vx } : {}),
    ...(vy !== undefined ? { vy } : {}),
    ...(vz !== undefined ? { vz } : {}),
    ...(ix !== undefined ? { ix: Math.round(ix) } : {}),
    ...(iy !== undefined ? { iy: Math.round(iy) } : {}),
    ...(iz !== undefined ? { iz: Math.round(iz) } : {}),
  };
  return { atom, wrapped };
};

export const parseDumpFile = (data: string): MoleculeData => {
  if (!data.trim()) throw new Error('Invalid LAMMPS dump: file is empty');

  const lines = data.split('\n');
  const frames: ParsedFrame[] = [];
  const types = newTypeTable();

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
    const hasCoords = [
      'x', 'y', 'z', 'xu', 'yu', 'zu',
      'xs', 'ys', 'zs', 'xsu', 'ysu', 'zsu',
    ].some(c => col.has(c));
    if (!hasCoords) {
      throw new Error(
        'Invalid LAMMPS dump: no coordinate columns (x/y/z, xu/yu/zu, ' +
        'xs/ys/zs or xsu/ysu/zsu) in ITEM: ATOMS header',
      );
    }
    i++;

    const atoms: Atom[] = [];
    let frameWrapped = true;
    while (i < lines.length && atoms.length < numAtoms) {
      const row = lines[i].trim();
      if (!row) { i++; continue; }
      if (row.startsWith('ITEM:')) break; // truncated frame
      const tokens = row.split(/\s+/);
      const parsed = rowToAtom(tokens, col, frameBox, atoms.length + 1, types);
      if (parsed) {
        atoms.push(parsed.atom);
        frameWrapped = parsed.wrapped;
      }
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

    frames.push({ timestep, atoms, box: frameBox, wrapped: frameWrapped });
  }

  const first = frames[0];
  if (!first) throw new Error('Invalid LAMMPS dump: no ITEM: TIMESTEP frames found');
  if (!first.box) throw new Error('Invalid LAMMPS dump: first frame lacks ITEM: BOX BOUNDS');

  // --- Type metadata from the reference frame ---
  // The element of a type: the element column or an element type label, else the mass column (nearest element
  // within 0.5 u), else unknown ('X'). The type number itself says nothing about the element.
  const atomTypes: Record<number, AtomTypeInfo> = {};
  const counts = new Map<number, number>();
  for (const a of first.atoms) counts.set(a.type, (counts.get(a.type) ?? 0) + 1);
  for (const type of [...counts.keys()].sort((a, b) => a - b)) {
    const fileMass = types.masses.get(type);
    let element = types.elements.get(type);
    if (!element && fileMass !== undefined && fileMass > 0) {
      const byMass = resolveElementFromMass(fileMass).symbol;
      if (byMass !== 'X') element = byMass;
    }
    const elem = element ? ELEMENT_DATA.find(e => e.symbol === element) : undefined;
    const name = types.labels.get(type) ?? `Type ${type}`;
    atomTypes[type] = {
      id: type,
      mass: fileMass ?? elem?.mass ?? 0,
      element: elem?.symbol ?? 'X',
      label: elem ? `${elem.name} (${elem.symbol})` : name,
      count: counts.get(type)!,
    };
  }

  // --- Trajectory frames (comment carries the timestep for the scrubber) ---
  const trajFrames: TrajectoryFrame[] = frames.map(f => ({
    comment: Number.isFinite(f.timestep) ? `timestep ${f.timestep}` : undefined,
    atoms: f.atoms,
    // Each frame keeps its own cell: under NPT the box breathes, so framing,
    // the rendered cell and the RDF normalisation must not all be pinned to
    // frame 0.
    ...(f.box ? { box: f.box } : {}),
    ...(f.wrapped ? {} : { coordsUnwrapped: true }),
  }));

  // --- Bonds from the reference frame (dumps carry no topology) ---
  // (only between atoms whose element is known: an unknown element has no covalent radius)
  const bonds = first.atoms.length <= 30000 ? inferBonds(first.atoms, { atomicNumberOf: atomicNumberByType(atomTypes) }) : [];

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
    bondsInferred: true,
    atomTypes,
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center,
    box: first.box,
    ...(trajFrames.length > 1 ? { frames: trajFrames } : {}),
  };
};
