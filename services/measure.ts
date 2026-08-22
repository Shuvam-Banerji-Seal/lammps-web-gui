import { Atom } from '../types';

/**
 * Geometric measurements over atom selections.
 * Pure functions — no three.js dependency so they stay unit-testable.
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const len = (a: Vec3): number => Math.sqrt(dot(a, a));

/** Euclidean distance in whatever units the coordinates use (Å here). */
export const distance = (a: Atom, b: Atom): number =>
  len(sub(a, b));

/** Angle a-b-c in degrees, vertex at b. */
export const angleDeg = (a: Atom, b: Atom, c: Atom): number => {
  const u = sub(a, b);
  const v = sub(c, b);
  const lu = len(u);
  const lv = len(v);
  if (lu < 1e-9 || lv < 1e-9) return 0;
  const cos = Math.min(1, Math.max(-1, dot(u, v) / (lu * lv)));
  return (Math.acos(cos) * 180) / Math.PI;
};

/**
 * Torsion angle a-b-c-d in degrees.
 *
 * Convention: the widely-used praxeolitic / RDKit / MDAnalysis formulation —
 *   view down the b→c axis; θ = rotation of (c→d) relative to (b→a),
 *   positive = counter-clockwise per right-hand rule about b→c.
 * Anti/trans arrangements read ≈ ±180°, eclipsed/cis ≈ 0°, gauche ≈ ±60°.
 *
 * Algorithm: project (a−b) and (d−c) onto the plane normal to the b→c axis,
 * then take the signed angle between the projections:
 *   x = v·w (cos part), y = (b1×v)·w (sin part), θ = atan2(y, x).
 */
export const dihedralDeg = (a: Atom, b: Atom, c: Atom, d: Atom): number => {
  let bx = c.x - b.x, by = c.y - b.y, bz = c.z - b.z;
  const bn = Math.sqrt(bx * bx + by * by + bz * bz);
  if (bn < 1e-9) return 0;
  bx /= bn; by /= bn; bz /= bn;

  // (a−b) and (d−c) projected perpendicular to the central axis
  const ax = a.x - b.x, ay = a.y - b.y, az = a.z - b.z;
  const dx = d.x - c.x, dy = d.y - c.y, dz = d.z - c.z;
  const pa = ax * bx + ay * by + az * bz;
  const pd = dx * bx + dy * by + dz * bz;
  const vx = ax - pa * bx, vy = ay - pa * by, vz = az - pa * bz;
  const wx = dx - pd * bx, wy = dy - pd * by, wz = dz - pd * bz;

  const vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
  const wl = Math.sqrt(wx * wx + wy * wy + wz * wz);
  if (vl < 1e-9 || wl < 1e-9) return 0;

  // Both components share the same normalization (vl*wl) — atan2 requires
  // a consistent (sin, cos) pair or the angle skews when |v| != 1.
  const denom = vl * wl;
  const cosPart = (vx * wx + vy * wy + vz * wz) / denom;
  // cross(b1, v) · w — sin component carrying the sign
  const sx = by * vz - bz * vy;
  const sy = bz * vx - bx * vz;
  const sz = bx * vy - by * vx;
  const sinPart = (sx * wx + sy * wy + sz * wz) / denom;

  return (Math.atan2(sinPart, cosPart) * 180) / Math.PI;
};

export type MeasurementKind = 'distance' | 'angle' | 'dihedral';

export interface MeasurementResult {
  kind: MeasurementKind;
  /** Human-readable value, e.g. "1.42 Å" or "109.5°". */
  label: string;
}

/** Measure the current selection (2 → distance, 3 → angle, 4 → dihedral). */
export const measureSelection = (atoms: Atom[]): MeasurementResult | null => {
  if (atoms.length === 2) {
    return { kind: 'distance', label: `${distance(atoms[0], atoms[1]).toFixed(3)} Å` };
  }
  if (atoms.length === 3) {
    return { kind: 'angle', label: `${angleDeg(atoms[0], atoms[1], atoms[2]).toFixed(1)}°` };
  }
  if (atoms.length === 4) {
    return {
      kind: 'dihedral',
      label: `${dihedralDeg(atoms[0], atoms[1], atoms[2], atoms[3]).toFixed(1)}°`,
    };
  }
  return null;
};

/** Short descriptor for the UI panel, e.g. "∠ O–H–O". */
export const measurementGlyph = (kind: MeasurementKind): string =>
  kind === 'distance' ? 'd(A–B)' : kind === 'angle' ? '∠ A–B–C' : 'torsion A–B–C–D';
