import type { UnitSystem } from './types';

/*
 * Lattices, from docs.lammps.org/lattice.html:
 *   styles "none, sc, bcc, fcc, hcp, diamond, sq, sq2, hex, or custom";
 *   hcp "a2 = 0 sqrt(3) 0, a3 = 0 0 sqrt(8/3)", 4 basis atoms "two in the
 *   z = 0 plane and 2 in the z = 0.5 plane"; hex "a2 = 0 sqrt(3) 0", 2 basis
 *   atoms "one at the corner and one at the center of the rectangle".
 *   "For unit style lj, the scale argument is the Lennard-Jones reduced
 *   density ... factor^dim = rho/rho*, where rho = N/V with V = the volume of
 *   the lattice unit cell and N = the number of basis atoms in the unit cell".
 *   "For all unit styles except lj, the scale argument is specified in the
 *   distance units defined by the unit style."
 * Basis coordinates are the standard crystallographic ones (diamond = fcc
 * plus the fcc sublattice shifted by (1/4, 1/4, 1/4)).
 * v1 supports the default origin and orientation only.
 */

export type LatticeStyle = 'sc' | 'bcc' | 'fcc' | 'hcp' | 'diamond' | 'sq' | 'sq2' | 'hex';

interface LatticeDef {
  dimension: 2 | 3;
  /** Edge lengths of the (orthogonal) unit cell before scaling. */
  cell: [number, number, number];
  /** Basis atoms in fractional coordinates of the unit cell. */
  basis: [number, number, number][];
}

const SQRT3 = Math.sqrt(3);

const DEFS: Record<LatticeStyle, LatticeDef> = {
  sc: { dimension: 3, cell: [1, 1, 1], basis: [[0, 0, 0]] },
  bcc: { dimension: 3, cell: [1, 1, 1], basis: [[0, 0, 0], [0.5, 0.5, 0.5]] },
  fcc: { dimension: 3, cell: [1, 1, 1], basis: [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]] },
  hcp: {
    dimension: 3,
    cell: [1, SQRT3, Math.sqrt(8 / 3)],
    basis: [[0, 0, 0], [0.5, 0.5, 0], [0.5, 5 / 6, 0.5], [0, 1 / 3, 0.5]],
  },
  diamond: {
    dimension: 3,
    cell: [1, 1, 1],
    basis: [
      [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
      [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
    ],
  },
  sq: { dimension: 2, cell: [1, 1, 1], basis: [[0, 0, 0]] },
  sq2: { dimension: 2, cell: [1, 1, 1], basis: [[0, 0, 0], [0.5, 0.5, 0]] },
  hex: { dimension: 2, cell: [1, SQRT3, 1], basis: [[0, 0, 0], [0.5, 0.5, 0]] },
};

export const LATTICE_STYLES = Object.keys(DEFS) as LatticeStyle[];
export const isLatticeStyle = (s: string): s is LatticeStyle => s in DEFS;

export interface Lattice {
  style: LatticeStyle;
  /** Multiplicative factor applied to the unit cell. */
  factor: number;
  /** Lattice spacings along x, y, z in distance units (region "lattice" units). */
  spacing: [number, number, number];
  basis: [number, number, number][];
}

/** Builds a lattice; throws a plain Error the interpreter turns into an EngineError. */
export const makeLattice = (
  style: LatticeStyle, scale: number, units: UnitSystem, dimension: 2 | 3,
): Lattice => {
  const def = DEFS[style];
  if (def.dimension !== dimension) {
    throw new Error(`lattice ${style} is ${def.dimension}d but the simulation is ${dimension}d`);
  }
  if (!(scale > 0)) throw new Error(`lattice scale must be > 0, got ${scale}`);
  let factor = scale;
  if (units.style === 'lj') {
    const volume = dimension === 3 ? def.cell[0] * def.cell[1] * def.cell[2] : def.cell[0] * def.cell[1];
    const rho = def.basis.length / volume;
    factor = Math.pow(rho / scale, 1 / dimension);
  }
  return {
    style,
    factor,
    spacing: [def.cell[0] * factor, def.cell[1] * factor, def.cell[2] * factor],
    basis: def.basis,
  };
};

/**
 * Lattice points p with lo <= p < hi in every dimension (half-open, so a
 * periodic box filled with whole cells gets no duplicate on its hi faces),
 * further restricted to the closed region [rlo, rhi] when one is given.
 * 2D lattices have z = 0 only. Returns a flat 3N array, z-major then y, x,
 * basis — the order sets the atom IDs.
 */
export const latticePoints = (
  lat: Lattice,
  lo: readonly number[], hi: readonly number[],
  dimension: 2 | 3,
  region?: { lo: readonly number[]; hi: readonly number[] },
): Float64Array => {
  const [sx, sy, sz] = lat.spacing;
  const tol = (d: number) => 1e-9 * Math.max(1, Math.abs(hi[d] - lo[d]));
  const range = (d: number, s: number): [number, number] =>
    [Math.floor(lo[d] / s) - 1, Math.ceil(hi[d] / s) + 1];
  const [i0, i1] = range(0, sx);
  const [j0, j1] = range(1, sy);
  const [k0, k1] = dimension === 3 ? range(2, sz) : [0, 0];
  const inside = (p: number, d: number) => {
    const t = tol(d);
    if (p < lo[d] - t || p >= hi[d] - t) return false;
    if (region && (p < region.lo[d] - t || p > region.hi[d] + t)) return false;
    return true;
  };
  const out: number[] = [];
  for (let k = k0; k <= k1; k++) {
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        for (const b of lat.basis) {
          const x = (i + b[0]) * sx;
          const y = (j + b[1]) * sy;
          const z = dimension === 3 ? (k + b[2]) * sz : 0;
          if (inside(x, 0) && inside(y, 1) && inside(z, 2)) out.push(x, y, z);
        }
      }
    }
  }
  return Float64Array.from(out);
};
