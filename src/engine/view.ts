import type { Atom, AtomTypeInfo, Bond, MoleculeData } from '../types';
import type { EngineEvent } from './types';

/** Distinct colours for engine atom types (index = type - 1, cycled). */
export const TYPE_COLORS = ['#7aa6d8', '#e07a5f', '#81b29a', '#f2cc8f', '#b28dff', '#ef8fb3', '#5fc9c4', '#c9a26b'];

export type FrameEvent = Extract<EngineEvent, { kind: 'frame' }>;

/**
 * Median length of the bonds whose atoms are less than half a box length apart in every direction (a bond across a
 * periodic boundary joins wrapped positions and is skipped), from at most `sample` bonds; null without such bonds.
 */
export const typicalBondLength = (frame: FrameEvent, bonds: Int32Array, sample = 2000): number | null => {
  const at = new Map<number, number>();
  for (let i = 0; i < frame.id.length; i++) at.set(frame.id[i], i);
  const half = [0, 1, 2].map((d) => (frame.box.hi[d] - frame.box.lo[d]) / 2);
  const lens: number[] = [];
  const step = Math.max(1, Math.floor(bonds.length / 2 / sample));
  for (let k = 0; k + 1 < bonds.length; k += 2 * step) {
    const i = at.get(bonds[k]), j = at.get(bonds[k + 1]);
    if (i === undefined || j === undefined) continue;
    let r2 = 0, wraps = false;
    for (let d = 0; d < 3; d++) {
      const dx = frame.x[3 * j + d] - frame.x[3 * i + d];
      if (Math.abs(dx) > half[d]) wraps = true;
      r2 += dx * dx;
    }
    if (!wraps && r2 > 0) lens.push(Math.sqrt(r2));
  }
  if (!lens.length) return null;
  lens.sort((a, b) => a - b);
  return lens[lens.length >> 1];
};

/** customColors for the viewer config: one colour per engine atom type. */
export const typeColors = (ntypes: number): Record<number, string> => {
  const out: Record<number, string> = {};
  for (let t = 1; t <= ntypes; t++) out[t] = TYPE_COLORS[(t - 1) % TYPE_COLORS.length];
  return out;
};

/**
 * Converts an engine frame into the viewer's MoleculeData: wrapped
 * positions (what the viewer renders), image flags kept, the box as the
 * scene extent, and the bonds of `bonds` (atom ID pairs, the frame's own or
 * the last ones the engine sent; none for an atomic system).
 */
export const frameToMoleculeData = (frame: FrameEvent, bonds: Int32Array | null = frame.bonds ?? null): MoleculeData => {
  const n = frame.id.length;
  const atoms: Atom[] = new Array(n);
  const counts = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const type = frame.type[i];
    counts.set(type, (counts.get(type) ?? 0) + 1);
    atoms[i] = {
      id: frame.id[i], molId: 0, type, charge: 0,
      x: frame.x[3 * i], y: frame.x[3 * i + 1], z: frame.x[3 * i + 2],
      ix: frame.image[3 * i], iy: frame.image[3 * i + 1], iz: frame.image[3 * i + 2],
    };
  }
  const atomTypes: Record<number, AtomTypeInfo> = {};
  for (const [type, count] of [...counts.entries()].sort((a, b) => a[0] - b[0])) {
    atomTypes[type] = { id: type, mass: 1, element: 'X', label: `type ${type}`, count };
  }
  const { lo, hi } = frame.box;
  const bondList: Bond[] = [];
  if (bonds) {
    for (let k = 0; k + 1 < bonds.length; k += 2) bondList.push({ id: k / 2 + 1, type: 1, atom1Id: bonds[k], atom2Id: bonds[k + 1] });
  }
  return {
    atoms,
    bonds: bondList,
    atomTypes,
    min: { x: lo[0], y: lo[1], z: lo[2] },
    max: { x: hi[0], y: hi[1], z: hi[2] },
    center: { x: (lo[0] + hi[0]) / 2, y: (lo[1] + hi[1]) / 2, z: (lo[2] + hi[2]) / 2 },
    box: { xlo: lo[0], xhi: hi[0], ylo: lo[1], yhi: hi[1], zlo: lo[2], zhi: hi[2] },
  };
};
