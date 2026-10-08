/*
 * Small dense helpers for the chunk17 computes (symmetric 3x3 tensors).
 * Textbook linear algebra only: cyclic Jacobi eigen-decomposition and a
 * minimum-norm (pseudo-inverse) solve of I w = L.
 */

/** Eigen-decomposition of a symmetric 3x3 matrix (row-major, length 9). Values sorted descending. */
export const symEigen3 = (m: ArrayLike<number>): { values: number[]; vectors: number[] } => {
  const a = [
    [m[0], m[1], m[2]],
    [m[3], m[4], m[5]],
    [m[6], m[7], m[8]],
  ];
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    const off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    const scale = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2] + off;
    if (off <= 1e-300 || off <= 1e-32 * scale) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        const apq = a[p][q];
        if (apq === 0) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [0, 1, 2].sort((x, y) => a[y][y] - a[x][x]);
  const values = order.map((k) => a[k][k]);
  const vectors: number[] = [];
  for (let r = 0; r < 3; r++) for (const k of order) vectors.push(v[r][k]);
  return { values, vectors };
};

/**
 * Minimum-norm solution w of I w = L for a symmetric 3x3 inertia tensor
 * (row-major). Eigen-directions with |lambda| <= 1e-10 max|lambda| are
 * dropped, so a singular tensor (a linear or single-atom chunk) gives the
 * minimum-norm solution, and a zero tensor gives zero.
 */
export const pinvSolve3 = (I: ArrayLike<number>, L: ArrayLike<number>): number[] => {
  const { values, vectors } = symEigen3(I);
  const maxAbs = Math.max(Math.abs(values[0]), Math.abs(values[1]), Math.abs(values[2]));
  const w = [0, 0, 0];
  if (maxAbs === 0) return w;
  const thr = 1e-10 * maxAbs;
  for (let k = 0; k < 3; k++) {
    const lam = values[k];
    if (Math.abs(lam) <= thr) continue;
    const ek = [vectors[k], vectors[3 + k], vectors[6 + k]];
    const proj = ek[0] * L[0] + ek[1] * L[1] + ek[2] * L[2];
    for (let d = 0; d < 3; d++) w[d] += (proj / lam) * ek[d];
  }
  return w;
};

/**
 * Shape parameters of a gyration tensor (xx, yy, zz, xy, xz, yz). Returns the
 * eigenvalues in descending order, then asphericity b, acylindricity c and
 * relative shape anisotropy k, as compute_gyration_shape.rst defines them with
 * l_z the largest eigenvalue.
 */
export const shapeOf = (t: ArrayLike<number>): number[] => {
  const { values } = symEigen3([t[0], t[3], t[4], t[3], t[1], t[5], t[4], t[5], t[2]]);
  const [l1, l2, l3] = values;
  const b = l1 - 0.5 * (l2 + l3);
  const c = l2 - l3;
  const sum = l1 + l2 + l3;
  const k = (1.5 * (l1 * l1 + l2 * l2 + l3 * l3)) / (sum * sum) - 0.5;
  return [l1, l2, l3, b, c, k];
};
