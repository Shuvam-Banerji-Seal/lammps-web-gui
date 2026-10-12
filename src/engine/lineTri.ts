/*
 * Data layout of atom_style line and tri particles (core, independent of the GPL ports): how a line segment or a
 * triangle given by its end or corner points becomes the per-atom centre, length or corners, orientation and
 * principal moments that the engine stores (atoms.ts LINE_BONUS, TRI_BONUS).
 *
 * docs.lammps.org/read_data.html, Lines section: "The 2 end points are the end points of the line segment. They
 * should be values close to the center point of the line segment specified in the Atoms section of the data file".
 * Triangles section: "The 3 corner points are the corner points of the triangle. They should be values close to the
 * center point of the triangle specified in the Atoms section of the data file".
 * docs.lammps.org/atom_style.html: "For the *line* style, particles can be are idealized line segments which store a
 * per-particle mass and length and orientation (i.e. the end points of the line segment)" and "For the *tri* style,
 * particles can be planar triangles which each stores a per-particle mass and size and orientation (i.e. the corner
 * points of the triangle)".
 * read_data.html: "lineflag = 1 for line segment particles, 0 for point or spherical particles" and "triangleflag = 1
 * for triangular particles, 0 for point or spherical particles".
 *
 */
import { quatToMatrix } from './compute/asphere';

type V3 = [number, number, number];

/** Line segment bonus: centre-to-end half length is length / 2; theta is the angle of c2 - c1 from +x in (-pi, pi]. */
export interface LineBonus { length: number; theta: number }

/**
 * Line segment from its end points (2d). read_data.html fixes the end-point order: the normal (c2 - c1) x (0, 0, 1)
 * points outward; the orientation theta is the direction of c2 - c1 (atom_vec_line.cpp data_atom_bonus: theta is
 * acos(dx / length) for dy >= 0 and minus that otherwise, which is atan2(dy, dx)).
 */
export const lineFromEnds = (x1: number, y1: number, x2: number, y2: number): LineBonus => {
  const dx = x2 - x1, dy = y2 - y1;
  return { length: Math.hypot(dx, dy), theta: Math.atan2(dy, dx) };
};


/** End points of a line segment centred at (cx, cy) (write_data: Lines section). */
export const lineEnds = (cx: number, cy: number, length: number, theta: number): [number, number, number, number] => {
  const h = 0.5 * length, c = Math.cos(theta), s = Math.sin(theta);
  return [cx - h * c, cy - h * s, cx + h * c, cy + h * s];
};

/** Area of a triangle: half the length of the cross product of two edges. */
export const triangleArea = (c: [V3, V3, V3]): number => {
  const a = [c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]];
  const b = [c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]];
  const n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  return 0.5 * Math.hypot(n[0], n[1], n[2]);
};

/** Corner point k of a triangle bonus in space, given the centroid and the orientation quaternion. */
export const triangleCorner = (centroid: ArrayLike<number>, quat: ArrayLike<number>, corners: ArrayLike<number>, k: number): V3 => {
  const R = quatToMatrix(quat[0], quat[1], quat[2], quat[3]);
  const b = [corners[3 * k], corners[3 * k + 1], corners[3 * k + 2]];
  return [0, 1, 2].map((r) => centroid[r] + R[3 * r] * b[0] + R[3 * r + 1] * b[1] + R[3 * r + 2] * b[2]) as V3;
};

/** Area of a triangle from its 9 body-frame corner values (the area does not depend on the orientation). */
export const triangleAreaOfBody = (c: ArrayLike<number>): number => triangleArea([[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]]);
