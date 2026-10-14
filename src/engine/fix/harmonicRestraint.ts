import type { Geometry } from '../domain';
import type { SimState } from '../types';

/**
 * One harmonic restraint E = k (r - d)^2 between atoms ia and ib (indices into the state), the substitute that a
 * constraint fix uses during minimization (fix_shake.cpp, bond_force(), at b332f28: "apply bond force for
 * minimization between atom indices i1 and i2"). The separation is the minimum-image vector x_ia - x_ib. Forces
 * are added to s.f: f_ia += delx * fbond, f_ib -= delx * fbond with fbond = -2 k (r - d) / r (0 when r = 0).
 * The energy k (r - d)^2 is returned even for r = 0, as native does.
 *
 * skipBelow: a separation below this value gives neither force nor energy (fix_ilves.cpp, min_harmonic_bond(),
 * "if (r < 1.0e-10) return 0.0"); fix shake passes 0.
 * store: when given (fix shake store yes), the same force is added to it per atom (3 values per atom).
 */
export const harmonicRestraint = (
  s: SimState, g: Geometry, ia: number, ib: number, d: number, k: number, skipBelow: number, store: Float64Array | null,
): number => {
  const del = [s.x[3 * ia] - s.x[3 * ib], s.x[3 * ia + 1] - s.x[3 * ib + 1], s.x[3 * ia + 2] - s.x[3 * ib + 2]];
  g.minimumImage(del);
  const r = Math.hypot(del[0], del[1], del[2]);
  if (r < skipBelow) return 0;
  const dr = r - d;
  const rk = k * dr;
  const fbond = r > 0 ? (-2 * rk) / r : 0;
  for (let q = 0; q < 3; q++) {
    const f = del[q] * fbond;
    s.f[3 * ia + q] += f;
    s.f[3 * ib + q] -= f;
    if (store) { store[3 * ia + q] += f; store[3 * ib + q] -= f; }
  }
  return k * dr * dr;
};
