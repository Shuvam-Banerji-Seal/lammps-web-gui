/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28) and the ML-PACE library (ACE evaluator), GPL-2.0 use: ML-PACE/ace-evaluator/ace_recursive.cpp (ACEDAG::init,
 * ACEDAG::insert_node, ACEDAG::find_2partitions and ACERecursiveEvaluator::acejlformat) of lammps-user-pace
 * v.2025.12.4.p1 (checked against LAMMPS's CMake SHA256). The LAMMPS package that calls it is src/ML-PACE/pair_pace.cpp at
 * commit b332f28. Library authors (from its headers): Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz and
 * co-authors (ICAMS, Ruhr-University Bochum). Used here under GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * The product (Julia-style) form of the rank > 1 basis: the one-particle A's, the products AA with their coefficients
 * ("acejlformat"), and the computational graph (DAG) of the recursive evaluator, built with the same heuristic 0 and the
 * same partition order as the C++ code.
 */
import type { CTildeBasis, CTildeFunc } from './cbasis';

/** The product-form data of one species (the jl_* arrays of ACERecursiveEvaluator). */
export interface ProductForm {
  /** Aspec(iA, 0..3) = (mu, n, l, m) of each one-particle A, num1 rows */
  Aspec: Int32Array;
  num1: number;
  /** AAspec(iAA, t) of the num2 products, maxorder columns (-1 padded) */
  AAspec: Int32Array;
  /** orders(iAA) */
  orders: Int32Array;
  /** coeffs(iAA, p), ndensity columns */
  coeffs: Float64Array;
  num2: number;
  /** jl_AAspec_flat: the A indices of the products, concatenated */
  AAspecFlat: Int32Array;
  maxorder: number;
  ndensity: number;
}

/** A computational graph of the recursive evaluator (ACEDAG). Indices: A's first (num1), then the internal products. */
export class ACEDAG {
  num1 = 0;
  /** nodes(k, 0..1): children of node num1 + k (internal and leaf nodes), 2 per node */
  nodes = new Int32Array(0);
  /** coeffs(k, p): ndensity per node */
  coeffs = new Float64Array(0);
  numNodes = 0;
  num2Int = 0;
  num2Leaf = 0;
  ndensity = 0;

  /** the AA buffer: one complex value per A and internal node (re, im) */
  AAbufRe = new Float64Array(0);
  AAbufIm = new Float64Array(0);
  /** the weights w(idx) of all nodes (re, im) */
  wRe = new Float64Array(0);
  wIm = new Float64Array(0);

  static build(Aspec: Int32Array, num1: number, AAspec: Int32Array, orders: Int32Array,
    coeffs: Float64Array, num2: number, maxorder: number, ndensity: number): ACEDAG {
    const dag = new ACEDAG();
    dag.num1 = num1;
    dag.ndensity = ndensity;
    const dagmap = new Map<string, number>();
    // stage one: the one-particle basis
    for (let iA = 0; iA < num1; iA++) dagmap.set(String(iA), iA);
    const st = { dagIdx: num1, nodesPre: [] as number[], coeffsPre: [] as number[] };
    // main loop over the AA basis
    for (let iAA = 0; iAA < num2; iAA++) {
      const ord = orders[iAA];
      const aa: number[] = [];
      for (let t = 0; t < ord; t++) aa.push(AAspec[iAA * maxorder + t]);
      const c: number[] = [];
      for (let p = 0; p < ndensity; p++) c.push(coeffs[iAA * ndensity + p]);
      insertNode(dagmap, st, num1, ndensity, aa, c);
    }
    const numNodes = st.dagIdx;
    dag.numNodes = numNodes;
    const nk = numNodes - num1;
    const nodesPre = st.nodesPre, coeffsPre = st.coeffsPre;
    const haschild = new Uint8Array(nk);
    for (let k = 0; k < nk; k++) {
      if (nodesPre[2 * k] >= num1) haschild[nodesPre[2 * k] - num1] = 1;
      if (nodesPre[2 * k + 1] >= num1) haschild[nodesPre[2 * k + 1] - num1] = 1;
    }
    const neworder = new Map<number, number>();
    for (let iA = 0; iA < num1; iA++) neworder.set(iA, iA);
    dag.nodes = new Int32Array(2 * nk);
    dag.coeffs = new Float64Array(nk * ndensity);
    let num2Int = 0, num2Leaf = 0;
    let dagIdx = num1;
    for (let iAA = 0; iAA < nk; iAA++) {
      if (haschild[iAA]) {
        num2Int += 1;
        const i1pre = nodesPre[2 * iAA], i2pre = nodesPre[2 * iAA + 1];
        const i1 = neworder.get(i1pre) as number, i2 = neworder.get(i2pre) as number;
        neworder.set(num1 + iAA, dagIdx);
        dag.nodes[2 * (dagIdx - num1)] = i1;
        dag.nodes[2 * (dagIdx - num1) + 1] = i2;
        for (let t = 0; t < ndensity; t++) dag.coeffs[(dagIdx - num1) * ndensity + t] = coeffsPre[iAA * ndensity + t];
        dagIdx++;
      }
    }
    for (let iAA = 0; iAA < nk; iAA++) {
      if (!haschild[iAA]) {
        num2Leaf += 1;
        const i1pre = nodesPre[2 * iAA], i2pre = nodesPre[2 * iAA + 1];
        dag.nodes[2 * (dagIdx - num1)] = neworder.get(i1pre) as number;
        dag.nodes[2 * (dagIdx - num1) + 1] = neworder.get(i2pre) as number;
        for (let t = 0; t < ndensity; t++) dag.coeffs[(dagIdx - num1) * ndensity + t] = coeffsPre[iAA * ndensity + t];
        dagIdx++;
      }
    }
    dag.num2Int = num2Int;
    dag.num2Leaf = num2Leaf;
    dag.AAbufRe = new Float64Array(num1 + num2Int);
    dag.AAbufIm = new Float64Array(num1 + num2Int);
    dag.wRe = new Float64Array(numNodes);
    dag.wIm = new Float64Array(numNodes);
    return dag;
  }
}

/** ACEDAG::find_2partitions: all splits of v into two non-empty groups, in the order of the C++ bit loop. */
const find2partitions = (v: number[]): Array<[number[], number[]]> => {
  const N = v.length;
  const parts: Array<[number[], number[]]> = [];
  for (let i = 1; i < (1 << N) / 2; i++) {
    const v1: number[] = [], v2: number[] = [];
    let p = 1;
    for (let n = 0; n < N; n++) {
      const zo = Math.floor(i / p) % 2;
      if (zo === 1) v1.push(v[n]);
      else v2.push(v[n]);
      p *= 2;
    }
    parts.push([v1, v2]);
  }
  return parts;
};

const keyOf = (v: number[]): string => v.join(',');

/** ACEDAG::insert_node with the heuristic 0 (the default of set_basis). */
const insertNode = (dagmap: Map<string, number>, st: { dagIdx: number; nodesPre: number[]; coeffsPre: number[] },
  num1: number, ndensity: number, a: number[], c: number[]): void => {
  const partitions = find2partitions(a);
  for (const p of partitions) {
    const k1 = keyOf(p[0]), k2 = keyOf(p[1]);
    if (dagmap.has(k1) && dagmap.has(k2)) {
      const i1 = dagmap.get(k1) as number;
      const i2 = dagmap.get(k2) as number;
      st.nodesPre.push(i1, i2);
      dagmap.set(keyOf(a), st.dagIdx);
      for (let q = 0; q < ndensity; q++) st.coeffsPre.push(c[q]);
      st.dagIdx += 1;
      return;
    }
  }
  // no partition has both parts in the graph: find the partition with the longest part already in it
  let longest: [number[], number[]] | null = null;
  let longestLength = 0;
  for (const p of partitions) {
    let len = 0;
    if (dagmap.has(keyOf(p[0]))) len = p[0].length;
    else if (dagmap.has(keyOf(p[1]))) len = p[1].length;
    if (len > 0 && len > longestLength) {
      longestLength = len;
      longest = p;
    }
  }
  if (longestLength === 0 || longest === null) {
    throw new Error('WARNING : something has gone horribly wrong! `longest_length == 0`!');
  }
  const cz = new Array<number>(ndensity).fill(0);
  if (dagmap.has(keyOf(longest[0]))) insertNode(dagmap, st, num1, ndensity, longest[1], cz);
  else insertNode(dagmap, st, num1, ndensity, longest[0], cz);
  insertNode(dagmap, st, num1, ndensity, a, c);
};

/**
 * ACERecursiveEvaluator::acejlformat (ace_recursive.cpp): the one-particle A list of species mu0, the AA products with
 * their (non-zero) coefficients, and the flat product index list. Like the C++ code it zeroes the coefficients of
 * m-combinations whose opposite is present (the half-basis trick), in place on the basis.
 */
export const acejlformat = (basis: CTildeBasis, mu0: number): ProductForm => {
  const basisFuncs: CTildeFunc[] = basis.basis[mu0];
  const totalBasisSize = basis.totalBasisSize[mu0];
  const maxmu = basis.nelements - 1;
  const maxn = basis.nradmax;
  const maxl = basis.lmax;
  const maxorder = basis.rankmax;
  const ndensity = basis.ndensitymax;
  const nl = maxl + 1, nm = 2 * maxl + 1;
  // lookup of 1-particle A's: index ((mu * maxn + (n - 1)) * nl + l) * nm + (m + l)
  const lk = (mu: number, n: number, l: number, m: number) => ((mu * maxn + (n - 1)) * nl + l) * nm + m;
  const A_lookup = new Int32Array((maxmu + 1) * maxn * nl * nm).fill(-1);
  let A_idx = 0;
  for (let fi = 0; fi < totalBasisSize; fi++) {
    const func = basisFuncs[fi];
    const order = func.rank;
    for (let ms_ind = 0; ms_ind < func.numMsCombs; ms_ind++) {
      for (let t = 0; t < order; t++) {
        const mus = func.mus[t], ns = func.ns[t], ls = func.ls[t], m = func.msCombs[ms_ind * order + t];
        const idx = lk(mus, ns, ls, m + ls);
        if (A_lookup[idx] === -1) {
          A_lookup[idx] = A_idx;
          A_idx += 1;
        }
      }
    }
  }
  const num1 = A_idx;
  const Aspec = new Int32Array(num1 * 4);
  for (let mu = 0; mu <= maxmu; mu++) {
    for (let n = 1; n <= maxn; n++) {
      for (let l = 0; l <= maxl; l++) {
        for (let m = -l; m <= l; m++) {
          const iA = A_lookup[lk(mu, n, l, l + m)];
          if (iA !== -1) {
            Aspec[iA * 4] = mu;
            Aspec[iA * 4 + 1] = n;
            Aspec[iA * 4 + 2] = l;
            Aspec[iA * 4 + 3] = m;
          }
        }
      }
    }
  }
  // half-basis trick
  for (let fi = 0; fi < totalBasisSize; fi++) {
    const func = basisFuncs[fi];
    const order = func.rank;
    if (!(func.mus[0] <= maxmu && func.ns[0] <= maxn && func.ls[0] <= maxl)) continue;
    for (let ms_ind = 0; ms_ind < func.numMsCombs; ms_ind++) {
      const ms = (t: number) => func.msCombs[ms_ind * order + t];
      let pos_idx = order + 1;
      let neg_idx = order + 1;
      for (let t = order - 1; t >= 0; t--) {
        if (ms(t) > 0) pos_idx = t;
        else if (ms(t) < 0) neg_idx = t;
      }
      if (neg_idx < pos_idx) {
        let ms_ind2 = 0;
        let found = false;
        for (ms_ind2 = 0; ms_ind2 < func.numMsCombs; ms_ind2++) {
          let isopposite = true;
          for (let t = 0; t < order; t++) {
            if (ms(t) !== -func.msCombs[ms_ind2 * order + t]) {
              isopposite = false;
              break;
            }
          }
          if (isopposite) {
            found = true;
            break;
          }
        }
        if (found) {
          for (let p = 0; p < ndensity; p++) {
            func.ctildes[ms_ind2 * ndensity + p] += func.ctildes[ms_ind * ndensity + p];
            func.ctildes[ms_ind * ndensity + p] = 0.0;
          }
        }
      }
    }
  }
  // count the non-zero products
  let num2 = 0;
  for (let fi = 0; fi < totalBasisSize; fi++) {
    const func = basisFuncs[fi];
    for (let ms_ind = 0; ms_ind < func.numMsCombs; ms_ind++) {
      let isnonzero = false;
      for (let p = 0; p < ndensity; p++) if (func.ctildes[ms_ind * ndensity + p] !== 0.0) isnonzero = true;
      if (isnonzero) num2++;
    }
  }
  const orders = new Int32Array(num2);
  const AAspec = new Int32Array(num2 * maxorder).fill(-1);
  const coeffs = new Float64Array(num2 * ndensity);
  let AAidx = 0;
  let len_flat = 0;
  for (let fi = 0; fi < totalBasisSize; fi++) {
    const func = basisFuncs[fi];
    const order = func.rank;
    if (!(func.mus[0] <= maxmu && func.ns[0] <= maxn && func.ls[0] <= maxl)) continue;
    for (let ms_ind = 0; ms_ind < func.numMsCombs; ms_ind++) {
      let iszero = true;
      for (let p = 0; p < ndensity; p++) if (func.ctildes[ms_ind * ndensity + p] !== 0.0) iszero = false;
      if (iszero) continue;
      orders[AAidx] = order;
      for (let t = 0; t < order; t++) {
        const Ait = A_lookup[lk(func.mus[t], func.ns[t], func.ls[t], func.msCombs[ms_ind * order + t] + func.ls[t])];
        AAspec[AAidx * maxorder + t] = Ait;
        len_flat += 1;
      }
      for (let t = order; t < maxorder; t++) AAspec[AAidx * maxorder + t] = -1;
      for (let p = 0; p < ndensity; p++) coeffs[AAidx * ndensity + p] = func.ctildes[ms_ind * ndensity + p];
      AAidx += 1;
    }
  }
  const AAspecFlat = new Int32Array(len_flat);
  let idx = 0;
  for (let a = 0; a < num2; a++) {
    for (let p = 0; p < orders[a]; p++) AAspecFlat[idx++] = AAspec[a * maxorder + p];
  }
  return { Aspec, num1, AAspec, orders, coeffs, num2, AAspecFlat, maxorder, ndensity };
};
