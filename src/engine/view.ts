import type { Atom, AtomTypeInfo, MoleculeData } from '../types';
import type { EngineEvent } from './types';

/** Distinct colours for engine atom types (index = type - 1, cycled). */
export const TYPE_COLORS = ['#7aa6d8', '#e07a5f', '#81b29a', '#f2cc8f', '#b28dff', '#ef8fb3', '#5fc9c4', '#c9a26b'];

export type FrameEvent = Extract<EngineEvent, { kind: 'frame' }>;

/** customColors for the viewer config: one colour per engine atom type. */
export const typeColors = (ntypes: number): Record<number, string> => {
  const out: Record<number, string> = {};
  for (let t = 1; t <= ntypes; t++) out[t] = TYPE_COLORS[(t - 1) % TYPE_COLORS.length];
  return out;
};

/**
 * Converts an engine frame into the viewer's MoleculeData: wrapped
 * positions (what the viewer renders), image flags kept, the box as the
 * scene extent, no bonds (atomic style).
 */
export const frameToMoleculeData = (frame: FrameEvent): MoleculeData => {
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
  return {
    atoms,
    bonds: [],
    atomTypes,
    min: { x: lo[0], y: lo[1], z: lo[2] },
    max: { x: hi[0], y: hi[1], z: hi[2] },
    center: { x: (lo[0] + hi[0]) / 2, y: (lo[1] + hi[1]) / 2, z: (lo[2] + hi[2]) / 2 },
    box: { xlo: lo[0], xhi: hi[0], ylo: lo[1], yhi: hi[1], zlo: lo[2], zhi: hi[2] },
  };
};
