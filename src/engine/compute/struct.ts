import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNum, parseInt_ } from '../force/util';

/*
 * Wave-11 per-atom structure computes (compute_struct):
 *   centro/atom, cna/atom, cluster/atom, fragment/atom, aggregate/atom.
 *
 * Sources: docs.lammps.org/compute_centro_atom.html,
 * docs.lammps.org/compute_cna_atom.html, docs.lammps.org/compute_cluster_atom.html
 * (the three pages are the same .rst file group). Algorithms are the published
 * methods: Kelchner, Plimpton, Hamilton, PRB 58, 11085 (1998) for the
 * centro-symmetry parameter; Faken & Jonsson, Comput Mater Sci 2, 279 (1994)
 * for common neighbor analysis.
 *
 * Neighbours are taken directly from the engine's periodic environment
 * (nb.xall: owned + ghost images), so a neighbour is an image position and
 * distances between two neighbours of one atom are plain differences of
 * xall positions. No LAMMPS source was read for this file.
 */

/** Tolerance on cutoff comparisons (ghost shell). */
const EPS = 1e-9;

/** Resolves the cutoff: keyword/argument value, else the largest pair style cutoff. */
const pairMaxCut = (sys: System): number => {
  let maxCut = 0;
  const pair = sys.ff.pair;
  if (pair) {
    const cut = pair.cut;
    for (let k = 0; k < cut.length; k++) if (cut[k] > maxCut) maxCut = cut[k];
  }
  return maxCut;
};

/** The neighbour cutoff must lie inside the ghost shell (force cutoff + skin), as in orientorder/atom. */
const checkGhostShell = (sys: System, id: string, style: string, cut: number): void => {
  const nb = sys.nb;
  if (nb.cutghost > 0 && nb.cutghost < cut + nb.skin - EPS) {
    throw new StyleError(`compute ${id} (${style}): cutoff ${cut} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
  }
};

/** Positive-cutoff argument of the cutoff-based styles. */
const parseCutoff = (w: string | undefined, id: string, style: string): number => {
  if (w === undefined) throw new StyleError(`compute ${id} (${style}) needs a cutoff distance`);
  const v = parseNum(w, `compute ${id} (${style}) cutoff`);
  if (!(v > 0)) throw new StyleError(`compute ${id} (${style}): cutoff must be > 0 (got ${w})`);
  return v;
};

/** Owned + ghost neighbours of owned atom i within cut (self excluded), in xall order. */
class NeighbourScratch {
  idx = new Int32Array(0);
  r2 = new Float64Array(0);
  dx = new Float64Array(0);
  dy = new Float64Array(0);
  dz = new Float64Array(0);
  count = 0;

  ensure(cap: number): void {
    if (this.idx.length >= cap) return;
    const c = cap + 64;
    this.idx = new Int32Array(c);
    this.r2 = new Float64Array(c);
    this.dx = new Float64Array(c);
    this.dy = new Float64Array(c);
    this.dz = new Float64Array(c);
  }

  gather(xa: Float64Array, nall: number, i: number, cut2: number): void {
    this.ensure(nall);
    const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
    let c = 0;
    for (let k = 0; k < nall; k++) {
      if (k === i) continue;
      const dx = xa[3 * k] - xi, dy = xa[3 * k + 1] - yi, dz = xa[3 * k + 2] - zi;
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= cut2) continue;
      this.idx[c] = k; this.r2[c] = r2; this.dx[c] = dx; this.dy[c] = dy; this.dz[c] = dz;
      c++;
    }
    this.count = c;
  }

  /** Reorders the first `count` entries so the `sel` nearest come first (ascending r2). */
  nearestFirst(sel: number): void {
    const { idx, r2, dx, dy, dz } = this;
    const c = this.count;
    for (let t = 0; t < sel; t++) {
      let best = t;
      for (let u = t + 1; u < c; u++) if (r2[u] < r2[best]) best = u;
      if (best === t) continue;
      const swap = (a: Int32Array | Float64Array) => { const tmp = a[t]; a[t] = a[best]; a[best] = tmp; };
      swap(idx); swap(r2); swap(dx); swap(dy); swap(dz);
    }
  }
}

/** Union-find over owned atom indices (path halving, union by index). */
class Forest {
  parent: Int32Array;
  constructor(n: number) {
    this.parent = new Int32Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }
  find(i: number): number {
    const p = this.parent;
    while (p[i] !== i) { p[i] = p[p[i]]; i = p[i]; }
    return i;
  }
  union(a: number, b: number): void {
    const ra = this.find(a), rb = this.find(b);
    if (ra !== rb) this.parent[Math.max(ra, rb)] = Math.min(ra, rb);
  }
}

/** Common setup of the per-atom structure computes: forces (ghosts) and the cutoff check. */
const prepare = (sys: System, id: string, style: string, cut: number) => {
  sys.forces();
  checkGhostShell(sys, id, style, cut);
  return { s: sys.state, nb: sys.nb };
};

// ---------------------------------------------------------------------------
// compute centro/atom
// ---------------------------------------------------------------------------

/**
 * "CS = \sum_{i = 1}^{N/2} | \vec{R}_i + \vec{R}_{i+N/2} |^2 ... There are
 * N (N-1)/2 possible neighbor pairs that can contribute to this formula. The
 * quantity in the sum is computed for each, and the N/2 smallest are used."
 * "If the atom does not have N neighbors (within the potential cutoff), then
 * its centro-symmetry parameter is set to 0.0." (compute_centro_atom.html)
 */
export class ComputeCentroAtom extends Compute {
  readonly style = 'centro/atom';
  peratomFlag = true;
  private readonly nNeigh: number;
  private readonly axes: boolean;
  private readonly scratch = new NeighbourScratch();
  private pairVal = new Float64Array(0);
  private pairA = new Int32Array(0);
  private pairB = new Int32Array(0);
  private pairOrder = new Int32Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError(`compute ${id} (centro/atom): usage: compute ID group-ID centro/atom lattice [axes yes|no]`);
    const lat = args[0];
    let n: number;
    if (lat === 'fcc') n = 12;
    else if (lat === 'bcc') n = 8;
    else {
      if (!/^\d+$/.test(lat)) throw new StyleError(`compute ${id} (centro/atom): lattice must be fcc, bcc or N (got '${lat}')`);
      n = parseInt_(lat, `compute ${id} (centro/atom) lattice`);
      if (n < 2 || n % 2 !== 0) throw new StyleError(`compute ${id} (centro/atom): N must be a positive even integer or fcc/bcc (got ${lat})`);
    }
    this.nNeigh = n;
    let axes = false;
    for (let k = 1; k < args.length; k++) {
      if (args[k] === 'axes') {
        const w = args[++k];
        if (w !== 'yes' && w !== 'no') throw new StyleError(`compute ${id} (centro/atom): axes must be yes or no (got '${w}')`);
        axes = w === 'yes';
      } else {
        throw new StyleError(`compute ${id} (centro/atom): unknown keyword '${args[k]}' (use axes)`);
      }
    }
    this.axes = axes;
    if (axes) this.sizePeratomCols = 10;
    this.pairVal = new Float64Array(n * (n - 1) / 2);
    this.pairA = new Int32Array(n * (n - 1) / 2);
    this.pairB = new Int32Array(n * (n - 1) / 2);
    this.pairOrder = new Int32Array(n * (n - 1) / 2);
  }

  protected computePeratom(): void {
    const cut = pairMaxCut(this.sys);
    if (!(cut > 0)) throw new StyleError(`compute ${this.id} (centro/atom): no pair style cutoff is defined`);
    const { s, nb } = prepare(this.sys, this.id, 'centro/atom', cut);
    const n = s.n;
    const N = this.nNeigh;
    const half = N / 2;
    const out = this.axes ? (this.arrayAtom = new Float64Array(10 * n)) : (this.vectorAtom = new Float64Array(n));
    const sc = this.scratch;
    const xa = nb.xall;
    const cut2 = cut * cut;
    const P = this.pairVal.length;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      sc.gather(xa, nb.nall, i, cut2);
      if (sc.count < N) continue; // not enough neighbours: centro-symmetry 0, axes 0
      sc.nearestFirst(N);
      // all pairs among the N nearest: |R_a + R_b|^2, then sort ascending (stable on pair index)
      let p = 0;
      for (let a = 0; a < N; a++) {
        for (let b = a + 1; b < N; b++) {
          const sx = sc.dx[a] + sc.dx[b], sy = sc.dy[a] + sc.dy[b], sz = sc.dz[a] + sc.dz[b];
          this.pairVal[p] = sx * sx + sy * sy + sz * sz;
          this.pairA[p] = a;
          this.pairB[p] = b;
          this.pairOrder[p] = p;
          p++;
        }
      }
      const ord = this.pairOrder;
      // insertion sort of pair indices by value (P <= 91 here)
      for (let u = 1; u < P; u++) {
        const v = ord[u];
        let w = u - 1;
        while (w >= 0 && this.pairVal[ord[w]] > this.pairVal[v]) { ord[w + 1] = ord[w]; w--; }
        ord[w + 1] = v;
      }
      let cs = 0;
      for (let t = 0; t < half; t++) cs += this.pairVal[ord[t]];
      if (!this.axes) {
        out[i] = cs;
        continue;
      }
      out[10 * i] = cs;
      // axes: unit vectors R_a - R_b of the two most symmetric pairs, third = right-hand normal
      const ax = [0, 0, 0, 0, 0, 0];
      for (let t = 0; t < 2 && t < half; t++) {
        const pidx = ord[t];
        const a = this.pairA[pidx], b = this.pairB[pidx];
        const vx = sc.dx[a] - sc.dx[b], vy = sc.dy[a] - sc.dy[b], vz = sc.dz[a] - sc.dz[b];
        const L = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;
        ax[3 * t] = vx / L; ax[3 * t + 1] = vy / L; ax[3 * t + 2] = vz / L;
      }
      // right-hand normal a x b of the two axes a = ax[0..2], b = ax[3..5]
      const c0 = ax[1] * ax[5] - ax[2] * ax[4];
      const c1 = ax[2] * ax[3] - ax[0] * ax[5];
      const c2 = ax[0] * ax[4] - ax[1] * ax[3];
      const CL = Math.sqrt(c0 * c0 + c1 * c1 + c2 * c2) || 1;
      for (let t = 0; t < 6; t++) out[10 * i + 1 + t] = ax[t];
      out[10 * i + 7] = c0 / CL; out[10 * i + 8] = c1 / CL; out[10 * i + 9] = c2 / CL;
    }
  }
}

// ---------------------------------------------------------------------------
// compute cna/atom
// ---------------------------------------------------------------------------

/**
 * Common neighbor analysis (Faken & Jonsson 1994): for each bonded neighbour
 * pair (i, j) of an atom, the signature is (nc, nb, nc_chain): nc common
 * neighbours (neighbours of both i and j), nb bonds among them, and the
 * length of the longest chain of those bonds. Classes used here follow the
 * documented codes fcc = 1, hcp = 2, bcc = 3, icosahedral = 4, unknown = 5
 * (compute_cna_atom.html). The signature sets for 12 neighbours (fcc 12x421;
 * hcp 6x421 + 6x422; icosahedral 12x555) and for 14 neighbours (bcc, see
 * classifyCna) were checked against native LAMMPS (black box) on the oracle cases.
 */
const CNA_UNKNOWN = 5;

/** Longest trail (bonds used at most once) in a small graph given as an adjacency matrix. */
const longestTrail = (m: number, adj: Uint8Array): number => {
  // edges of the graph on m nodes
  const eu: number[] = [];
  const ev: number[] = [];
  for (let a = 0; a < m; a++) for (let b = a + 1; b < m; b++) if (adj[a * m + b]) { eu.push(a); ev.push(b); }
  const E = eu.length;
  if (E === 0) return 0;
  let best = 0;
  const used = new Uint8Array(E);
  const walk = (node: number, len: number): void => {
    if (len > best) best = len;
    for (let e = 0; e < E; e++) {
      if (used[e]) continue;
      let next = -1;
      if (eu[e] === node) next = ev[e];
      else if (ev[e] === node) next = eu[e];
      if (next < 0) continue;
      used[e] = 1;
      walk(next, len + 1);
      used[e] = 0;
    }
  };
  for (let s = 0; s < m; s++) walk(s, 0);
  return best;
};

export class ComputeCnaAtom extends Compute {
  readonly style = 'cna/atom';
  peratomFlag = true;
  private readonly cutoffArg: number;
  private readonly scratch = new NeighbourScratch();
  private adjA = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`compute ${id} (cna/atom): usage: compute ID group-ID cna/atom cutoff`);
    this.cutoffArg = parseCutoff(args[0], id, 'cna/atom');
  }

  protected computePeratom(): void {
    const cut = this.cutoffArg;
    const { s, nb } = prepare(this.sys, this.id, 'cna/atom', cut);
    const n = s.n;
    const out = (this.vectorAtom = new Float64Array(n));
    const sc = this.scratch;
    const xa = nb.xall;
    const cut2 = cut * cut;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      sc.gather(xa, nb.nall, i, cut2);
      const M = sc.count;
      if (M === 0) { out[i] = CNA_UNKNOWN; continue; }
      // bond matrix among the neighbours of i (positions are image positions in xall)
      if (this.adjA.length < M * M) this.adjA = new Uint8Array(M * M);
      const adj = this.adjA;
      for (let a = 0; a < M; a++) {
        const ka = sc.idx[a];
        for (let b = 0; b < M; b++) {
          if (a === b) { adj[a * M + b] = 0; continue; }
          const kb = sc.idx[b];
          const dx = xa[3 * ka] - xa[3 * kb], dy = xa[3 * ka + 1] - xa[3 * kb + 1], dz = xa[3 * ka + 2] - xa[3 * kb + 2];
          adj[a * M + b] = dx * dx + dy * dy + dz * dz < cut2 ? 1 : 0;
        }
      }
      // signature of every bonded pair (i, a): common neighbours b with adj[a][b]
      const sigs: [number, number, number][] = [];
      for (let a = 0; a < M; a++) {
        const common: number[] = [];
        for (let b = 0; b < M; b++) if (adj[a * M + b]) common.push(b);
        const nc = common.length;
        const m = nc;
        const sub = new Uint8Array(m * m);
        let nbonds = 0;
        for (let p = 0; p < m; p++) for (let q = 0; q < m; q++) {
          if (p === q) continue;
          const bit = adj[common[p] * M + common[q]];
          sub[p * m + q] = bit;
          if (bit && p < q) nbonds++;
        }
        sigs.push([nc, nbonds, longestTrail(m, sub)]);
      }
      out[i] = classifyCna(sigs, M);
    }
  }
}

/** Maps the per-pair signatures of an atom with M neighbours to the CNA class (1..5). */
const classifyCna = (sigs: [number, number, number][], M: number): number => {
  let s421 = 0, s422 = 0, s555 = 0, s666 = 0, s444 = 0;
  for (const [nc, nb, ch] of sigs) {
    if (nc === 4 && nb === 2 && ch === 1) s421++;
    else if (nc === 4 && nb === 2 && ch === 2) s422++;
    else if (nc === 5 && nb === 5 && ch === 5) s555++;
    else if (nc === 6 && nb === 6 && ch === 6) s666++;
    else if (nc === 4 && nb === 4 && ch === 4) s444++;
  }
  if (M === 12) {
    if (s421 === 12) return 1;
    if (s421 === 6 && s422 === 6) return 2;
    if (s555 === 12) return 4;
    return CNA_UNKNOWN;
  }
  if (M === 14) {
    // Measured with native LAMMPS (black box, bccvac sweep, 14-neighbour atoms): class 3 for
    // 8 x 666 + 6 x 444 per-neighbour signatures (87 of 87 atoms). Counts are per neighbour here.
    if (s666 === 8 && s444 === 6) return 3;
    return CNA_UNKNOWN;
  }
  return CNA_UNKNOWN;
};

// ---------------------------------------------------------------------------
// compute cluster/atom, fragment/atom, aggregate/atom
// ---------------------------------------------------------------------------

/**
 * "A cluster is defined as a set of atoms, each of which is within the cutoff
 * distance from one or more other atoms in the cluster. If an atom has no
 * neighbors within the cutoff distance, then it is a 1-atom cluster."
 * "The cluster ID or fragment ID of every atom in the cluster will be set to
 * the smallest atom ID of any atom in the cluster or fragment, respectively."
 * "Only atoms in the compute group are clustered and assigned cluster IDs.
 * Atoms not in the compute group are assigned an ID = 0." (compute_cluster_atom.html)
 */
export class ComputeClusterAtom extends Compute {
  readonly style = 'cluster/atom';
  peratomFlag = true;
  private readonly cutoffArg: number;
  private readonly scratch = new NeighbourScratch();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`compute ${id} (cluster/atom): usage: compute ID group-ID cluster/atom cutoff`);
    this.cutoffArg = parseCutoff(args[0], id, 'cluster/atom');
  }

  protected computePeratom(): void {
    const cut = this.cutoffArg;
    const { s, nb } = prepare(this.sys, this.id, 'cluster/atom', cut);
    const n = s.n;
    const out = (this.vectorAtom = new Float64Array(n));
    const forest = new Forest(n);
    const sc = this.scratch;
    const cut2 = cut * cut;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      sc.gather(nb.xall, nb.nall, i, cut2);
      for (let c = 0; c < sc.count; c++) {
        const o = nb.owner[sc.idx[c]];
        if (s.mask[o] & this.groupBit) forest.union(i, o);
      }
    }
    fillMinIds(out, s.id, s.mask, this.groupBit, n, forest);
  }
}

/** Sets out[i] = smallest atom ID of i's component (group atoms), 0 for atoms outside the group. */
const fillMinIds = (out: Float64Array, id: Int32Array, mask: Int32Array | Uint32Array, groupBit: number, n: number, forest: Forest): void => {
  const minId = new Float64Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) {
    if (!(mask[i] & groupBit)) continue;
    const r = forest.find(i);
    if (id[i] < minId[r]) minId[r] = id[i];
  }
  for (let i = 0; i < n; i++) {
    if (!(mask[i] & groupBit)) { out[i] = 0; continue; }
    out[i] = minId[forest.find(i)];
  }
};

export class ComputeFragmentAtom extends Compute {
  readonly style = 'fragment/atom';
  peratomFlag = true;
  private readonly single: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    let single = false;
    for (let k = 0; k < args.length; k++) {
      if (args[k] === 'single') {
        const w = args[++k];
        if (w !== 'yes' && w !== 'no') throw new StyleError(`compute ${id} (fragment/atom): single must be yes or no (got '${w}')`);
        single = w === 'yes';
      } else {
        throw new StyleError(`compute ${id} (fragment/atom): unknown keyword '${args[k]}' (use single)`);
      }
    }
    this.single = single;
  }

  protected computePeratom(): void {
    this.sys.forces();
    const s = this.sys.state;
    const n = s.n;
    const out = (this.vectorAtom = new Float64Array(n));
    const forest = new Forest(n);
    const bonded = new Uint8Array(n);
    const map = bondMap(s.id, n);
    const b = this.sys.state.topo.bonds;
    for (let k = 0; k < b.n; k++) {
      const ia = map(b.atoms[2 * k]), ib = map(b.atoms[2 * k + 1]);
      if (ia < 0 || ib < 0) continue;
      if (!(s.mask[ia] & this.groupBit) || !(s.mask[ib] & this.groupBit)) continue;
      forest.union(ia, ib);
      bonded[ia] = 1; bonded[ib] = 1;
    }
    fillMinIds(out, s.id, s.mask, this.groupBit, n, forest);
    // atoms without any bond: 0, or their own ID with single yes
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (!bonded[i]) out[i] = this.single ? s.id[i] : 0;
    }
  }
}

/** Atom ID -> owned index (-1 when the ID is not present). */
const bondMap = (ids: Int32Array, n: number) => {
  let maxId = 0;
  for (let i = 0; i < n; i++) if (ids[i] > maxId) maxId = ids[i];
  const table = new Int32Array(maxId + 1).fill(-1);
  for (let i = 0; i < n; i++) table[ids[i]] = i;
  return (id: number): number => (id >= 0 && id < table.length ? table[id] : -1);
};

/**
 * "An aggregate is defined by combining the rules for clusters and fragments
 * (i.e., a set of atoms, where each of them is within the cutoff distance from
 * one or more atoms within a fragment that is part of the same cluster)"
 * (compute_cluster_atom.html).
 * Implemented as the union of bond connectivity (fragments) and cutoff
 * proximity among group atoms.
 */
export class ComputeAggregateAtom extends Compute {
  readonly style = 'aggregate/atom';
  peratomFlag = true;
  private readonly cutoffArg: number;
  private readonly scratch = new NeighbourScratch();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`compute ${id} (aggregate/atom): usage: compute ID group-ID aggregate/atom cutoff`);
    this.cutoffArg = parseCutoff(args[0], id, 'aggregate/atom');
    // Measured with native LAMMPS (black box): aggregate/atom under atom_style atomic fails with
    // native error text: Compute aggregate/atom used when bonds are not allowed
    const st = sys.atomStyle;
    if (st === 'atomic' || st === 'charge' || st === 'sphere') {
      throw new StyleError(`compute ${id} (aggregate/atom): bonds are not allowed by atom_style ${st}; use a bonded atom_style`);
    }
  }

  protected computePeratom(): void {
    const cut = this.cutoffArg;
    const { s, nb } = prepare(this.sys, this.id, 'aggregate/atom', cut);
    const n = s.n;
    const out = (this.vectorAtom = new Float64Array(n));
    const forest = new Forest(n);
    const sc = this.scratch;
    const cut2 = cut * cut;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      sc.gather(nb.xall, nb.nall, i, cut2);
      for (let c = 0; c < sc.count; c++) {
        const o = nb.owner[sc.idx[c]];
        if (s.mask[o] & this.groupBit) forest.union(i, o);
      }
    }
    const map = bondMap(s.id, n);
    const b = this.sys.state.topo.bonds;
    for (let k = 0; k < b.n; k++) {
      const ia = map(b.atoms[2 * k]), ib = map(b.atoms[2 * k + 1]);
      if (ia < 0 || ib < 0) continue;
      if (!(s.mask[ia] & this.groupBit) || !(s.mask[ib] & this.groupBit)) continue;
      forest.union(ia, ib);
    }
    fillMinIds(out, s.id, s.mask, this.groupBit, n, forest);
  }
}

// ---------------------------------------------------------------------------
// compute hexorder/atom
// ---------------------------------------------------------------------------

/**
 * Bond-orientational order of a 2d system (compute_hexorder_atom.rst):
 * "q_n = \frac{1}{nnn}\sum_{j = 1}^{nnn} e^{n i \theta({\textbf{r}}_{ij})}"
 * "where the sum is over the *nnn* nearest neighbors of the central atom. The
 * angle :math:`\theta` is formed by the bond vector :math:`r_{ij}` and the *x* axis."
 * "\theta is calculated only using the x and y components, whereas the
 * distance from the central atom is calculated using all three x, y, and z
 * components of the bond vector." Output: 2 columns, Re and Im of q_n.
 * "The value of :math:`q_n` is set to zero for atoms not in the specified compute
 * group, as well as for atoms that have less than *nnn* neighbors within the
 * distance cutoff." Defaults: cutoff = pair style cutoff, nnn = 6, degree = 6.
 * "If the value is NULL, then all neighbors up to the distance cutoff are used."
 * Neighbours outside the group are included, as the docs state.
 */
export class ComputeHexorderAtom extends Compute {
  readonly style = 'hexorder/atom';
  peratomFlag = true;
  sizePeratomCols = 2;
  private readonly degree: number;
  /** null = all neighbours within the cutoff (NULL keyword). */
  private readonly nnn: number | null;
  private readonly cutoffArg: number | null;
  private readonly scratch = new NeighbourScratch();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    let degree = 6;
    let nnn: number | null = 6;
    let cutoffArg: number | null = null;
    if (sys.dimension !== 2) throw new StyleError(`compute ${id} (hexorder/atom): requires a 2d system (dimension 2)`);
    for (let k = 0; k < args.length; k++) {
      const kw = args[k];
      const w = args[k + 1];
      if (kw === 'degree') {
        degree = parseInt_(w, `compute ${id} (hexorder/atom) degree`);
        if (degree < 0) throw new StyleError(`compute ${id} (hexorder/atom): degree must be a non-negative integer (got ${w})`);
      } else if (kw === 'nnn') {
        if (w === 'NULL') nnn = null;
        else {
          nnn = parseInt_(w, `compute ${id} (hexorder/atom) nnn`);
          if (nnn < 1) throw new StyleError(`compute ${id} (hexorder/atom): nnn must be a positive integer or NULL (got ${w})`);
        }
      } else if (kw === 'cutoff') {
        cutoffArg = parseCutoff(w, id, 'hexorder/atom');
      } else {
        throw new StyleError(`compute ${id} (hexorder/atom): unknown keyword '${kw}' (use degree, nnn, cutoff)`);
      }
      k++;
    }
    this.degree = degree;
    this.nnn = nnn;
    this.cutoffArg = cutoffArg;
  }

  protected computePeratom(): void {
    const pairCut = pairMaxCut(this.sys);
    const cut = this.cutoffArg ?? pairCut;
    if (!(cut > 0)) throw new StyleError(`compute ${this.id} (hexorder/atom): no pair style cutoff is defined; define a pair style or use the cutoff keyword`);
    if (this.cutoffArg !== null && this.cutoffArg > pairCut + EPS) {
      throw new StyleError(`compute ${this.id} (hexorder/atom): cutoff ${this.cutoffArg} exceeds the pair style cutoff ${pairCut} (the maximum allowable value)`);
    }
    const { s, nb } = prepare(this.sys, this.id, 'hexorder/atom', cut);
    const n = s.n;
    const out = (this.arrayAtom = new Float64Array(2 * n));
    const sc = this.scratch;
    const xa = nb.xall;
    const cut2 = cut * cut;
    const sel = this.nnn ?? 0;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      sc.gather(xa, nb.nall, i, cut2);
      if (sel > 0) {
        if (sc.count < sel) continue; // fewer than nnn neighbours: q_n = 0
        sc.nearestFirst(sel);
      }
      const m = sel > 0 ? sel : sc.count;
      if (m === 0) continue;
      let re = 0, im = 0;
      for (let t = 0; t < m; t++) {
        const theta = Math.atan2(sc.dy[t], sc.dx[t]);
        re += Math.cos(this.degree * theta);
        im += Math.sin(this.degree * theta);
      }
      out[2 * i] = re / m;
      out[2 * i + 1] = im / m;
    }
  }
}
