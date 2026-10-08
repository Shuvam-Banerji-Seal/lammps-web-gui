/**
 * Direct instance-matrix writes.
 *
 * Every atom instance is a translation plus a UNIFORM scale, so its 4x4 is
 * known in closed form and can be written straight into the
 * InstancedBufferAttribute. That avoids Object3D.updateMatrix()'s
 * position/quaternion/scale compose and the setMatrixAt copy for every atom,
 * which is the hot loop when a trajectory rewrites tens of thousands of
 * instances per frame.
 *
 * Layout is three.js's column-major Matrix4.elements order, i.e. exactly what
 * `Matrix4.toArray(target, offset)` writes — see the unit tests, which assert
 * equality against THREE.Object3D for random inputs.
 */

/** Write the instance `i` transform for a uniformly scaled translation. */
export const writeInstanceTransform = (
  out: Float32Array,
  i: number,
  x: number,
  y: number,
  z: number,
  scale: number,
): void => {
  const o = i * 16;
  out[o] = scale;   out[o + 1] = 0;       out[o + 2] = 0;       out[o + 3] = 0;
  out[o + 4] = 0;   out[o + 5] = scale;   out[o + 6] = 0;       out[o + 7] = 0;
  out[o + 8] = 0;   out[o + 9] = 0;       out[o + 10] = scale;  out[o + 11] = 0;
  out[o + 12] = x;  out[o + 13] = y;      out[o + 14] = z;      out[o + 15] = 1;
};

/**
 * Write instance `i` for a segment between two points: a unit cylinder along
 * +Y is scaled to `radius x length x radius`, rotated onto the segment and
 * translated to its midpoint. Used for bonds.
 *
 * `basis` is scratch space (9 floats) so the hot loop allocates nothing.
 */
export const writeInstanceSegment = (
  out: Float32Array,
  i: number,
  x1: number, y1: number, z1: number,
  x2: number, y2: number, z2: number,
  radius: number,
  basis: Float32Array,
): void => {
  let dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
  const len = Math.hypot(dx, dy, dz);
  const o = i * 16;
  if (len === 0) {
    // Degenerate segment: collapse the instance so it renders nothing.
    for (let k = 0; k < 16; k++) out[o + k] = 0;
    out[o + 15] = 1;
    return;
  }

  // Y axis = the segment direction.
  dx /= len; dy /= len; dz /= len;

  // Pick any vector not parallel to the direction, then Gram-Schmidt.
  let ax = 0, ay = 0, az = 1;
  if (Math.abs(dz) > 0.9) { ax = 1; ay = 0; az = 0; }

  // X axis = normalize(a x dir)
  let xx = ay * dz - az * dy;
  let xy = az * dx - ax * dz;
  let xz = ax * dy - ay * dx;
  const xl = Math.hypot(xx, xy, xz) || 1;
  xx /= xl; xy /= xl; xz /= xl;

  // Z axis = X x Y. Using Y x X instead yields a LEFT-handed basis whose
  // determinant is negative, which flips the cylinder's normals and lights
  // every bond from the wrong side.
  const zx = xy * dz - xz * dy;
  const zy = xz * dx - xx * dz;
  const zz = xx * dy - xy * dx;

  basis[0] = xx; basis[1] = xy; basis[2] = xz;
  basis[3] = dx; basis[4] = dy; basis[5] = dz;
  basis[6] = zx; basis[7] = zy; basis[8] = zz;

  out[o] = xx * radius;     out[o + 1] = xy * radius;  out[o + 2] = xz * radius;  out[o + 3] = 0;
  out[o + 4] = dx * len;    out[o + 5] = dy * len;     out[o + 6] = dz * len;     out[o + 7] = 0;
  out[o + 8] = zx * radius; out[o + 9] = zy * radius;  out[o + 10] = zz * radius; out[o + 11] = 0;
  out[o + 12] = (x1 + x2) / 2;
  out[o + 13] = (y1 + y2) / 2;
  out[o + 14] = (z1 + z2) / 2;
  out[o + 15] = 1;
};
