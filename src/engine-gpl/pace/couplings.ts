/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the ML-PACE library v.2025.12.4.p1 it builds
 * for src/ML-PACE, GPL-2.0 use: ace/ace_couplings.cpp (generate_coupling_tree, get_ms_sign,
 * get_ms_space_size_and_strides, unpack_ms_space_point, generate_equivariant_ms_cg_list, generate_ms_cg_list,
 * generate_basis_function_n_body) and ace/ace_clebsch_gordan.cpp (wigner3j, anotherClebschGordan). Library authors (from
 * its headers): Yury Lysogorskiy, Cas van der Oord, Anton Bochkarev, Sarath Menon, Matteo Rinaldi, Thomas Hammerschmidt,
 * Matous Mrovec, Aidan Thompson, Gabor Csanyi, Christoph Ortner, Ralf Drautz. The Wigner 3j series is ported from the
 * wigner-cpp library bundled with it (wigner-cpp/include/wigner/wigner_3nj.hpp: rec_luscombe_luban,
 * funcs_xyz_wigner3j_rec_j, funcs_xyz_wigner3j_rec_j_jmin0, wigner3j_rec_j_impl; MIT licence, Copyright (c) 2016
 * Hidekazu Ikeno). This file is GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * The generalized Clebsch-Gordan coefficients of the B-basis functions: the m-combinations of a rank-r function coupled
 * through the binary coupling tree to total L = 0, each with the product of the Clebsch-Gordan coefficients along the
 * tree. The 3j symbols come from the same Schulten-Gordon recursion as the C++, so the coefficients match it to the bit.
 */

/** sqrt(sqrt(DBL_MIN)): rec_luscombe_luban's "tiny". */
const TINY = Math.sqrt(Math.sqrt(2.2250738585072014e-308));

interface XYZ {
  X(j: number): number;
  Y(j: number): number;
  Z(j: number): number;
}

/** funcs_xyz_wigner3j_rec_j: f(j) = (j1 j2 j; m1 m2 -m1-m2) for jmin > 0. */
const xyzRecJ = (j1: number, j2: number, m1: number, m2: number): XYZ => {
  const a1 = j1 - j2, a2 = j1 + j2 + 1, a3 = -m1 - m2, b1 = m2 - m1;
  const b2 = a3 * (j1 * (j1 + 1) - j2 * (j2 + 1));
  const aux = (j: number) => Math.sqrt((a1 + j) * (a1 - j) * (a2 + j) * (a2 - j) * (a3 - j) * (a3 + j));
  return {
    X: (j) => j * aux(j + 1),
    Y: (j) => (2 * j + 1) * (j * (j + 1) * b1 - b2),
    Z: (j) => (j + 1) * aux(j),
  };
};

/** funcs_xyz_wigner3j_rec_j_jmin0: the same with j(j+1) factored out (j1 = j2, m1 = -m2). */
const xyzRecJmin0 = (j1: number, j2: number, m1: number, m2: number): XYZ => {
  const a2 = j1 + j2 + 1, b1 = m2 - m1;
  const Z = (j: number) => j * Math.sqrt((a2 + j) * (a2 - j));
  return { X: (j) => Z(j + 1), Y: (j) => (2 * j + 1) * b1, Z };
};

/**
 * rec_luscombe_luban: the solution of X(n) f(n+1) + Y(n) f(n) + Z(n) f(n-1) = 0 for nmin <= n <= nmax, unnormalized,
 * by the two-term ratio recursions in the non-classical regions and the three-term recursion between them.
 */
const recLuscombeLuban = (nmin: number, nmax: number, dest: Float64Array, fn: XYZ): void => {
  const siz = nmax - nmin + 1;
  // backward: r(n) = f(n)/f(n-1) from nmax down to where |r| first exceeds 1
  let i2 = siz - 1;
  let n2 = nmax;
  const ymax = fn.Y(nmax);
  if (Math.abs(ymax) < TINY) {
    dest[i2] = 1;
  } else {
    let rn = -fn.Z(nmax) / ymax;
    dest[i2] = rn;
    while (--i2 > 0) {
      --n2;
      rn = -fn.Z(n2) / (fn.Y(n2) + fn.X(n2) * rn);
      if (Math.abs(rn) > 1) break;
      dest[i2] = rn;
    }
    let k = i2;
    dest[k] = 1;
    while (++k < siz - 1) dest[k + 1] *= dest[k];
  }
  if (i2 === 0) return;
  // forward: s(n) = f(n)/f(n+1) from nmin up to where |s| first exceeds 1 (or i2)
  let i1 = 0;
  let n1 = nmin;
  const ymin = fn.Y(nmin);
  if (Math.abs(ymin) < TINY) {
    dest[i1] = 1;
  } else {
    let sn = -fn.X(nmin) / ymin;
    dest[i1] = sn;
    while (++i1 < i2) {
      ++n1;
      sn = -fn.X(n1) / (fn.Y(n1) + fn.Z(n1) * sn);
      if (Math.abs(sn) > 1) break;
      dest[i1] = sn;
    }
    let k = i1;
    dest[k] = 1;
    while (--k > 0) dest[k - 1] *= dest[k];
  }
  if (i1 < i2) {
    // classical region: the three-term recursion from n1 to n2, then scaled to f(n2) = 1
    let n = n1;
    if (i1 === 0) {
      dest[i1 + 1] = (-fn.Y(n) * dest[i1]) / fn.X(n);
      ++n;
      ++i1;
    }
    for (let k = i1; k < i2; ++k) {
      const numer = -fn.Y(n) * dest[k] - fn.Z(n) * dest[k - 1];
      dest[k + 1] = numer / fn.X(n);
      ++n;
    }
    const scale = 1 / dest[i2];
    for (let k = 0; k <= i2; ++k) dest[k] *= scale;
  }
};

/**
 * Wigner3jSeriesJ::compute(j1, j2, m1, m2): (j1 j2 j; m1 m2 -m1-m2) for j from jmin = max(|j1-j2|, |m1+m2|) to
 * jmax = j1+j2, normalized by sum (2j+1) f^2 = 1 with sgn f(jmax) = (-1)^(j1-j2+m1+m2). Empty when |m1| > j1 or
 * |m2| > j2.
 */
const wigner3jSeriesJ = (j1: number, j2: number, m1: number, m2: number): { jmin: number; values: Float64Array } => {
  if (!(Math.abs(m1) <= j1 && Math.abs(m2) <= j2)) return { jmin: -1, values: new Float64Array(0) };
  const jmin = Math.max(Math.abs(j1 - j2), Math.abs(m1 + m2));
  const jmax = j1 + j2;
  const out = new Float64Array(jmax - jmin + 1);
  let parity = j1 - j2 + m1 + m2;
  let jj1 = 2 * jmin + 1;
  if (jmin === jmax) {
    out[0] = ((parity & 1) ? -1 : 1) / Math.sqrt(jj1);
    return { jmin, values: out };
  }
  recLuscombeLuban(jmin, jmax, out, jmin === 0 ? xyzRecJmin0(j1, j2, m1, m2) : xyzRecJ(j1, j2, m1, m2));
  let sum = 0;
  for (let k = 0; k < out.length; k++) {
    sum += jj1 * out[k] * out[k];
    jj1 += 2;
  }
  parity += out[out.length - 1] < 0 ? 1 : 0;
  const scale = ((parity & 1) ? -1 : 1) / Math.sqrt(sum);
  for (let k = 0; k < out.length; k++) out[k] *= scale;
  return { jmin, values: out };
};

/** ace_clebsch_gordan.cpp wigner3j: (j1 j2 J; m1 m2 M), zero unless m1+m2+M = 0 and J is in the series range. */
export const wigner3j = (j1: number, m1: number, j2: number, m2: number, J: number, M: number): number => {
  if (m1 + m2 + M !== 0) return 0;
  const s = wigner3jSeriesJ(j1, j2, m1, m2);
  const jmax = s.jmin + s.values.length - 1;
  if (J < s.jmin || J > jmax) return 0;
  return s.values[J - s.jmin];
};

/** anotherClebschGordan: C(j1 m1, j2 m2 | J M) = (-1)^(-j1+j2-M) sqrt(2J+1) (j1 j2 J; m1 m2 -M). */
export const clebschGordan = (j1: number, m1: number, j2: number, m2: number, J: number, M: number): number => {
  if (m1 + m2 !== M) return 0;
  const jmin = Math.abs(j1 - j2), jmax = Math.abs(j1 + j2);
  if (J > jmax || J < jmin) return 0;
  const head = `C_L(${J}|${j1},${j2})_M(${M}|${m1},${m2}): `;
  if (Math.abs(m1) > j1) throw new Error(`${head}Non-sense coefficient C_L: |m1|>l1`);
  if (Math.abs(m2) > j2) throw new Error(`${head}Non-sense coefficient: |m2|>l2`);
  if (Math.abs(M) > J) throw new Error(`${head}Non-sense coefficient: |M|>L`);
  const factor = ((-j1 + j2 - M) & 1) ? -1 : 1;
  return factor * Math.sqrt(2.0 * J + 1.0) * wigner3j(j1, m1, j2, m2, J, -M);
};

/**
 * generate_coupling_tree: the binary coupling tree of rank r as triples (i1, i2, i_coupled), the leaves 0..r-1, the
 * intermediate nodes r..2r-3 and -1 for the root.
 */
export const generateCouplingTree = (r: number): Int32Array => {
  if (r < 2) return new Int32Array(0);
  const out: number[] = [];
  const msQueue: number[] = [];
  const MSQueue: number[] = [];
  const interm: number[] = [];
  for (let i = 0; i < r; i++) msQueue.push(i);
  for (let i = 0; i < r - 2; i++) MSQueue.push(i + r);
  let iInt = 0;
  for (let i = 0; i < Math.trunc(r / 2); i++) {
    const i1 = msQueue.shift()!, i2 = msQueue.shift()!;
    iInt = MSQueue.length > 0 ? MSQueue.shift()! : -1;
    out.push(i1, i2, iInt);
    interm.push(iInt);
  }
  while (interm.length > 1 && iInt !== -1) {
    const i1 = interm.shift()!, i2 = interm.shift()!;
    iInt = MSQueue.length > 0 ? MSQueue.shift()! : -1;
    out.push(i1, i2, iInt);
    interm.push(iInt);
  }
  if (iInt !== -1) {
    // last node of the tree: the remaining intermediate with the remaining leaf
    const i1 = interm.shift()!, i2 = msQueue.shift()!;
    out.push(i1, i2, -1);
    interm.push(-1);
  }
  if (out.length !== 3 * (r - 1)) {
    throw new Error(`Unable to build coupling tree for rank = ${r}, expected size = ${3 * (r - 1)} actual size = ${out.length}`);
  }
  return Int32Array.from(out);
};

/** get_ms_sign: the sign of the first non-zero m (0 if all are zero). */
export const getMsSign = (rank: number, ms: ArrayLike<number>): number => {
  for (let j = 0; j < rank; ++j) {
    if (ms[j] < 0) return -1;
    if (ms[j] > 0) return 1;
  }
  return 0;
};

/** One m-combination with its generalized Clebsch-Gordan coefficient (ms_cg_pair). */
export interface MsCgPair {
  ms: Int32Array;
  c: number;
  sign: number;
}

/**
 * generate_equivariant_ms_cg_list: every (m1..mr) with sum M whose tree couplings are valid and whose coefficient is
 * non-zero, in ms-space order (m of the first particle varies fastest). With half_basis the combinations whose first
 * non-zero m is negative are dropped and the positive ones count twice.
 */
export const generateEquivariantMsCgList = (
  rank: number, ls: ArrayLike<number>, LS: ArrayLike<number>, L: number, M: number, halfBasis: boolean, checkIsEven: boolean,
): MsCgPair[] => {
  if (Math.abs(M) > L) throw new Error(`Should be |M|<L, but get L=${L}, M=${M}`);
  const lsLS = new Int32Array(2 * rank - 1);
  let sumOfLs = 0;
  for (let i = 0; i < rank; i++) {
    lsLS[i] = ls[i];
    sumOfLs += ls[i];
  }
  if (checkIsEven && sumOfLs % 2 !== 0) {
    let s = 'sum of ls is not even ls = (';
    for (let i = 0; i < rank; i++) s += `${ls[i]} `;
    throw new Error(`${s})`);
  }
  for (let i = rank; i < 2 * rank - 2; i++) lsLS[i] = LS[i - rank];
  lsLS[2 * rank - 2] = L;
  const tree = generateCouplingTree(rank);
  // get_ms_space_size_and_strides
  const strides = new Float64Array(rank);
  strides[0] = 1;
  let size = 2 * ls[0] + 1;
  for (let i = 1; i < rank; i++) {
    strides[i] = strides[i - 1] * (2 * ls[i - 1] + 1);
    size *= 2 * ls[i] + 1;
  }
  const out: MsCgPair[] = [];
  const ms = new Int32Array(rank);
  const msMS = new Int32Array(2 * rank - 1);
  for (let ind = 0; ind < size; ind++) {
    // unpack_ms_space_point
    let p = ind;
    for (let j = rank - 1; j >= 0; j--) {
      ms[j] = Math.floor(p / strides[j]) - ls[j];
      p = p % strides[j];
    }
    let sumOfMs = 0;
    for (let j = 0; j < rank; j++) sumOfMs += ms[j];
    if (sumOfMs !== M) continue;
    const sign = getMsSign(rank, ms);
    if (halfBasis && sign < 0) continue;
    msMS.fill(0);
    for (let i = 0; i < rank; i++) msMS[i] = ms[i];
    let gen = 1.0;
    let invalid = false;
    for (let t = 0; t < rank - 1; t++) {
      const i1 = tree[3 * t], i2 = tree[3 * t + 1];
      let ic = tree[3 * t + 2];
      if (ic === -1) ic = 2 * rank - 2;
      msMS[ic] = msMS[i1] + msMS[i2];
      if (Math.abs(msMS[i1]) > lsLS[i1] || Math.abs(msMS[i2]) > lsLS[i2] || Math.abs(msMS[ic]) > lsLS[ic]) {
        invalid = true;
        break;
      }
      gen *= clebschGordan(lsLS[i1], msMS[i1], lsLS[i2], msMS[i2], lsLS[ic], msMS[ic]);
      if (gen === 0.0) break;
    }
    if (invalid || gen === 0.0) continue;
    if (halfBasis && sign > 0) gen *= 2;
    out.push({ ms: ms.slice(), c: gen, sign });
  }
  return out;
};

/** generate_ms_cg_list: the rotation-invariant (L = M = 0) list, with the even-sum check. */
export const generateMsCgList = (rank: number, ls: ArrayLike<number>, LS: ArrayLike<number>, halfBasis: boolean): MsCgPair[] =>
  generateEquivariantMsCgList(rank, ls, LS, 0, 0, halfBasis, true);
