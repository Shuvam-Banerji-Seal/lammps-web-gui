import { describe, expect, it } from 'vitest';
import { dihedralGeometry } from '../src/engine/force/bonded_util';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import type { BondedCompute } from '../src/engine/force/types';
import { newAccum } from '../src/engine/force/types';

/*
 * Bonded geometry: the analytic dihedral gradient must match finite
 * differences of the dihedral angle, for generic and near-degenerate
 * configurations, including across a periodic boundary.
 */

const setup = (pts: number[][], L = 20) => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [L, L, L] }, 1);
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [L, L, L] }));
  const bc: BondedCompute = { s, geom, map: new Int32Array(0), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
  return { s, bc };
};

const check = (pts: number[][], L = 20) => {
  const { s, bc } = setup(pts, L);
  const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
  dihedralGeometry(bc, 0, 1, 2, 3, grad, rel);
  const h = 1e-6;
  for (let a = 0; a < 4; a++) {
    for (let d = 0; d < 3; d++) {
      const x0 = s.x[3 * a + d];
      s.x[3 * a + d] = x0 + h;
      const fp = dihedralGeometry(bc, 0, 1, 2, 3, new Array(12).fill(0), new Array(12).fill(0));
      s.x[3 * a + d] = x0 - h;
      const fm = dihedralGeometry(bc, 0, 1, 2, 3, new Array(12).fill(0), new Array(12).fill(0));
      s.x[3 * a + d] = x0;
      expect(grad[3 * a + d]).toBeCloseTo((fp - fm) / (2 * h), 6);
    }
  }
};

describe('dihedral geometry', () => {
  it('analytic gradient matches finite differences', () => {
    check([[5.1, 5.2, 4.7], [6.0, 5.1, 5.05], [6.6, 6.1, 4.8], [7.4, 6.3, 5.7]]);
    check([[5, 5, 5], [6, 5, 5], [6.5, 5.9, 5.1], [7.5, 6.0, 4.0]]);
  });
  it('works across a periodic boundary (minimum image)', () => {
    check([[19.6, 5.2, 4.7], [0.5, 5.1, 5.05], [1.1, 6.1, 4.8], [1.9, 6.3, 5.7]]);
  });
  it('trans is 180 degrees', () => {
    const { bc } = setup([[5, 6, 5], [5, 5, 5], [6, 5, 5], [6, 4, 5]]);
    const phi = dihedralGeometry(bc, 0, 1, 2, 3, new Array(12).fill(0), new Array(12).fill(0));
    expect(Math.abs(phi)).toBeCloseTo(Math.PI, 12);
  });
});
