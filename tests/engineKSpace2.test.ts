import { describe, expect, it } from 'vitest';
import { makeErfcTable, erfcPoly, EWALD_F } from '../src/engine/force/erfc';

/*
 * pair_modify table N emulation (erfc.ts makeErfcTable) checked against native
 * LAMMPS, black box. Input for each value: two charges +1 at separation r,
 * units real, pair_style coul/long 8.0, kspace_style ewald 1e-3,
 * kspace_modify gewald 0.3, pair_modify table 12, thermo ecoul (%.15g).
 * Native ecoul = qqrd2e erfc(g r)/r with qqrd2e = 332.06371.
 */
const C = 332.06371, G = 0.3;
const NATIVE_TABLE12: Array<[number, number]> = [
  [1.5, 116.115657393582],
  [3.0, 22.4798041511354],
];

describe('pair_modify table emulation (erfc.ts)', () => {
  it('energy table reproduces native table-12 ecoul at node and off-node distances', () => {
    const T = makeErfcTable(12, G, 100);
    for (const [r, e] of NATIVE_TABLE12) {
      expect(Math.abs(C * T.energy(r * r) - e) / e).toBeLessThan(1e-12);
    }
  });

  it('table 12 error is ~1e-6 relative to the polynomial no-table value at r = 5.1', () => {
    const T = makeErfcTable(12, G, 100);
    const r = 5.1, x = G * r, ex = Math.exp(-x * x);
    const rel = (T.energy(r * r) * r) / erfcPoly(x, ex) - 1;
    expect(Math.abs(rel)).toBeLessThan(1e-5);
    expect(Math.abs(rel)).toBeGreaterThan(1e-7);
  });

  it('force kernel is the same node structure with EWALD_F, exact at a node', () => {
    const T = makeErfcTable(12, G, 100);
    const r = 3.0, x = G * r, ex = Math.exp(-x * x);
    expect(T.force(r * r)).toBeCloseTo((erfcPoly(x, ex) + EWALD_F * x * ex) / r, 6);
  });
});
