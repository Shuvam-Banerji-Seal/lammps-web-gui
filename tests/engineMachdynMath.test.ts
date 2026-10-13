import { describe, expect, it } from 'vitest';
import { det3, identity3, inverse3, mul3, polDec, pseudoInverseSVD, svd3, transpose3, type Mat3 } from '../src/engine-gpl/machdyn/smd_math';
import { quatFromRotation } from '../src/engine-gpl/machdyn/compute_smd';

// The 3x3 helpers of the MACHDYN port (smd_math.ts) against their defining identities.
const close = (a: Mat3, b: Mat3, tol = 1e-12) => a.every((x, k) => Math.abs(x - b[k]) <= tol * (1 + Math.abs(b[k])));
const rot = (deg: number, axis: 0 | 1 | 2): Mat3 => {
  const c = Math.cos((deg * Math.PI) / 180), s = Math.sin((deg * Math.PI) / 180);
  const m = identity3();
  if (axis === 2) { m[0] = c; m[1] = -s; m[3] = s; m[4] = c; }
  if (axis === 0) { m[4] = c; m[5] = -s; m[7] = s; m[8] = c; }
  if (axis === 1) { m[0] = c; m[2] = s; m[6] = -s; m[8] = c; }
  return m;
};

describe('SMD 3x3 math (MACHDYN port helpers)', () => {
  it('SVD reconstructs the matrix with sorted non-negative singular values', () => {
    const M: Mat3 = [2, 0.3, -0.1, 0.2, 1.5, 0.4, -0.3, 0.1, 0.8];
    const { U, s, V } = svd3(M);
    const S: Mat3 = [s[0], 0, 0, 0, s[1], 0, 0, 0, s[2]];
    expect(close(mul3(mul3(U, S), transpose3(V)), M)).toBe(true);
    expect(s[0] >= s[1] && s[1] >= s[2] && s[2] >= 0).toBe(true);
  });

  it('polar decomposition of R T gives back the rotation R with det 1', () => {
    const R = mul3(rot(30, 2), rot(-20, 0));
    const T: Mat3 = [1.1, 0.05, 0, 0.05, 0.95, 0, 0, 0, 1.02];
    const F = mul3(R, T);
    const Rout = mat3Zero();
    const Tout = mat3Zero();
    expect(polDec(F, Rout, Tout)).toBe(true);
    expect(close(Rout, R, 1e-10)).toBe(true);
    expect(Math.abs(det3(Rout) - 1)).toBeLessThan(1e-12);
  });

  it('pseudo-inverse of an invertible symmetric matrix is its inverse', () => {
    const K: Mat3 = [4, 1, 0, 1, 3, 0.5, 0, 0.5, 2];
    const inv = inverse3(K);
    const P = Array.from(K);
    pseudoInverseSVD(P);
    expect(close(P, inv, 1e-10)).toBe(true);
  });

  it('pseudo-inverse of a rank-deficient matrix ignores the zero singular value', () => {
    const K: Mat3 = [2, 0, 0, 0, 3, 0, 0, 0, 0];
    const P = Array.from(K);
    pseudoInverseSVD(P);
    expect(close(P, [0.5, 0, 0, 0, 1 / 3, 0, 0, 0, 0])).toBe(true);
  });

  it('quaternion of a rotation matrix (Eigen branch rule) reproduces the rotation angle', () => {
    const R = rot(70, 1);
    const [x, y, z, w] = quatFromRotation(R);
    expect(Math.abs(w * w + x * x + y * y + z * z - 1)).toBeLessThan(1e-12);
    expect(Math.abs(Math.abs(w) - Math.cos((70 * Math.PI) / 360))).toBeLessThan(1e-12);
  });
});

const mat3Zero = (): Mat3 => [0, 0, 0, 0, 0, 0, 0, 0, 0];
