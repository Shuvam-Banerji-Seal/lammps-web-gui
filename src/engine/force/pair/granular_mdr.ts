import { StyleError } from '../types';

/*
 * Normal contact model mdr of pair_style granular (docs.lammps.org/pair_granular.html).
 * The doc says: "The *mdr* model is a mechanically-derived contact model designed to capture the
 * contact response between adhesive elastic-plastic particles into large deformation."
 * The doc points to its theory in the two-part series by Zunker and Kamrin (2024, J. Mech. Phys. Solids 183,
 * arXiv 2309.07300 and 2309.07317). Their method of dimensionality reduction
 * gives the elastic force of a sphere pressed on a rigid plane, F = E_c' A B/4 [arccos(1 - 2 d/A) - (1 - 2 d/A)
 * sqrt(4 d/A - 4 d^2/A^2)] with A = 4R, B = 2R and E_c' = E/(1 - nu^2) (paper eqs 1, 2, 4), and the paper's
 * yield displacement d_Y solves p_H(d) = p_Y(d) with p_H = 4 E_c' sqrt(d)/(3 pi sqrt(R)) and
 * p_Y = Y (1.75 exp(-4.4 d/R) + 1) (paper eqs 5, 6, 13).
 *
 * Measured with native LAMMPS (black box; two equal spheres, head-on and dynamic runs, nu = 0, 0.3, 0.5, radii
 * 0.5 and 1):
 * - the elastic force of two equal spheres of radius R at total overlap delta is the sphere-on-plane force above
 *   with R, E_c' = E/(1-nu^2) and the per-sphere overlap d = delta/2 (F = E_c' 2 R^2 br(d/(2R)));
 * - the normal damping with damping mdr 1 is eta_n = eta_n0 sqrt(m_eff k_mdr), k_mdr = 2 E_c' a with
 *   a = sqrt(4 R d - d^2) (d = delta/2), and with damping mdr 2 it is eta_n = eta_n0;
 * - the first deviation from the elastic force (the yield point) is at d = d_Y of the paper with the exponent 4.4.
 * Not implemented (StyleError, see granular.ts): plastic contact (d >= d_Y, which changes the particle radius
 * through the incompressible apparent-radius update), adhesion (surface energy > 0), unequal radii, the bulk
 * response, the mindlin tangential models and fix wall/gran.
 */

/** br(x) = arccos(1 - x) - (1 - x) sqrt(2x - x^2): the MDR elastic kernel for x = d/(2R) in [0, 1]. */
export const mdrKernel = (x: number): number => Math.acos(1 - x) - (1 - x) * Math.sqrt(2 * x - x * x);

const yieldCache = new Map<string, number>();

/**
 * Per-sphere yield displacement d_Y: the root of 4 E' sqrt(d)/(3 pi sqrt(R)) = Y (1.75 exp(-4.4 d/R) + 1).
 * The left side rises from 0 and the right side falls to Y, so the root is unique for Y > 0; for Y = 0
 * every positive overlap is plastic (d_Y = 0).
 */
export function mdrYieldDisplacement(Ep: number, R: number, Y: number): number {
  if (!(Y > 0)) return 0;
  const key = `${Ep}|${R}|${Y}`;
  const hit = yieldCache.get(key);
  if (hit !== undefined) return hit;
  const c = (4 * Ep) / (3 * Math.PI * Math.sqrt(R));
  const g = (d: number) => c * Math.sqrt(d) - Y * (1.75 * Math.exp((-4.4 * d) / R) + 1);
  let lo = 0, hi = Y;
  while (g(hi) < 0) hi *= 2;
  for (let k = 0; k < 200; k++) {
    const mid = 0.5 * (lo + hi);
    if (g(mid) < 0) lo = mid; else hi = mid;
  }
  const d = 0.5 * (lo + hi);
  if (yieldCache.size > 512) yieldCache.clear();
  yieldCache.set(key, d);
  return d;
}

/** Elastic MDR normal force (positive repulsive) and the contact radius a = sqrt(4 R d - d^2) of the two-sphere contact. */
export function mdrElastic(delta: number, R: number, Ep: number, Y: number): { fne: number; a: number } {
  const d = 0.5 * delta;
  const dY = mdrYieldDisplacement(Ep, R, Y);
  if (d >= dY) {
    throw new StyleError(
      `pair_style granular: normal model 'mdr' reached the yield displacement (per-sphere overlap ${d.toPrecision(6)} >= ${dY.toPrecision(6)}); plastic mdr contact (apparent radius growth) is not implemented in this engine`,
    );
  }
  const x = d / (2 * R);
  return { fne: Ep * 2 * R * R * mdrKernel(x), a: Math.sqrt(4 * R * d - d * d) };
}

/** Normal damping coefficient of damping mdr d_type (1 or 2): eta_n0 sqrt(m_eff 2 E' a) for 1, eta_n0 for 2. */
export function mdrDampCoeff(dtype: number, eta0: number, meff: number, delta: number, R: number, Ep: number): number {
  if (dtype === 2) return eta0;
  const d = 0.5 * delta;
  const a = Math.sqrt(4 * R * d - d * d);
  return eta0 * Math.sqrt(meff * 2 * Ep * a);
}

/**
 * Closed triangles of touching particles. Measured with native LAMMPS (black box): a three-sphere triangle (all
 * three pairs in contact, newton off) gives forces that differ from the pairwise elastic law by about 2e-6 relative
 * once the spheres move, while a chain of three spheres and a cube of eight spheres (no triangle) match the law
 * exactly. The native rule (the doc's "topological penalty" for "obstructing particles") is not identified, so a
 * contact that closes a triangle is a StyleError here.
 */
export function mdrCheckTriangles(pairs: Array<[number, number]>): void {
  if (pairs.length < 3) return;
  const adj = new Map<number, Set<number>>();
  for (const [a, b] of pairs) {
    if (!adj.has(a)) adj.set(a, new Set());
    if (!adj.has(b)) adj.set(b, new Set());
    adj.get(a)!.add(b);
    adj.get(b)!.add(a);
  }
  for (const [a, b] of pairs) {
    for (const c of adj.get(a)!) {
      if (c !== b && adj.get(b)!.has(c)) {
        throw new StyleError(`pair_style granular: normal model 'mdr' contacts that close a triangle of touching particles (atoms ${a}, ${b}, ${c}) are not implemented in this engine`);
      }
    }
  }
}
