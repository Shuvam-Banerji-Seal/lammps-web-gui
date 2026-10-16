/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0, which bundles voro++ 0.4.6.
 * Source: voro++ 0.4.6 src/cell.cc (voronoicell_base::nplane, collapse_order1/2, delete_connection, add_memory*,
 * init_base, init (voronoicell_neighbor), neighbors, m_test, check_marginal, max_radius_squared,
 * plane_intersects, plane_intersects_guess, plane_intersects_track) and src/cell.hh, as built in LAMMPS
 * lib/voronoi (the LAMMPS patch of rad_option.hh is the mono radius policy used in worklist.ts/compute.ts).
 *
 * voro++ is Copyright (c) 2008, The Regents of the University of California, through Lawrence Berkeley National
 * Laboratory, and Copyright (c) 2008-2011 Chris H. Rycroft (chr@alum.mit.edu), distributed under the BSD
 * three-clause licence reproduced in LICENSE.voro++ next to this file. Ported with the same data structures, the
 * same loops and the same floating-point operation order; the memory blocks, slot strides and pointer
 * relocations of the C code are kept as typed-array views, so the vertex and face order is the same.
 *
 * Only the neighbour-tracking cell (voronoicell_neighbor) is ported: volume, face areas and the other output
 * routines of cell.cc are not needed for the neighbour order.
 */

/** config.hh */
export const TOLERANCE = 1e-11;
export const TOLERANCE2 = 2e-11;
export const LARGE_NUMBER = 1e30;
const INIT_VERTICES = 256;
const INIT_VERTEX_ORDER = 64;
const INIT_3_VERTICES = 256;
const INIT_N_VERTICES = 8;
const INIT_MARGINAL = 64;
const INIT_DELETE_SIZE = 256;
const INIT_DELETE2_SIZE = 256;
const MAX_VERTICES = 16777216;
const MAX_VERTEX_ORDER = 2048;
const MAX_N_VERTICES = 16777216;
const MAX_MARGINAL = 16777216;
const MAX_DELETE_SIZE = 16777216;
const MAX_DELETE2_SIZE = 16777216;

/** Two row views start at the same storage position (C pointer equality). */
const samePtrView = (a: Int32Array, b: Int32Array): boolean => a.buffer === b.buffer && a.byteOffset === b.byteOffset;

/** A row view sits at slot offset `offset` of a block (C pointer equality against mep[i]+j). */
const samePtr = (a: Int32Array | undefined, block: Int32Array, offset: number): boolean =>
  a !== undefined && a.buffer === block.buffer && a.byteOffset === block.byteOffset + 4 * offset;

const fatal = (msg: string): never => {
  throw new Error(`voro++ internal error: ${msg}`);
};

/** voronoicell_neighbor: a Voronoi cell that records the plane (neighbour) id of every face. */
export class VoroCellNeighbor {
  /** number of vertices */
  p = 0;
  up = 0;
  private curVertices = INIT_VERTICES;
  private curOrder = INIT_VERTEX_ORDER;
  private curDel = INIT_DELETE_SIZE;
  private curDel2 = INIT_DELETE2_SIZE;
  /** edge rows: ed[v] = [neighbour vertices (nu) | relation indices (nu) | vertex id] */
  ed: (Int32Array | undefined)[];
  /** neighbour (plane id) rows, one per vertex, parallel to ed */
  ne: (Int32Array | undefined)[];
  nu: Int32Array;
  pts: Float64Array;
  /** per vertex order: slot count, capacity, slot block (stride 2i+1) and neighbour block (stride i) */
  mem: Int32Array;
  mec: Int32Array;
  mep: (Int32Array | undefined)[];
  mne: (Int32Array | undefined)[];
  private ds: Int32Array;
  private ds2: Int32Array;
  private marg: Int32Array;
  private curMarg = INIT_MARGINAL;
  private nMarg = 0;
  private px = 0;
  private py = 0;
  private pz = 0;
  private prsq = 0;
  private mAns = 0;
  private paux1: Int32Array = new Int32Array(0);
  private paux2: Int32Array = new Int32Array(0);

  constructor() {
    this.ed = new Array(this.curVertices);
    this.ne = new Array(this.curVertices);
    this.nu = new Int32Array(this.curVertices);
    this.pts = new Float64Array(3 * this.curVertices);
    this.mem = new Int32Array(this.curOrder);
    this.mec = new Int32Array(this.curOrder);
    this.mep = new Array(this.curOrder);
    this.mne = new Array(this.curOrder);
    for (let i = 0; i < 3; i++) {
      this.mem[i] = INIT_N_VERTICES;
      this.mec[i] = 0;
      this.mep[i] = new Int32Array(INIT_N_VERTICES * ((i << 1) + 1));
      this.mne[i] = new Int32Array(INIT_N_VERTICES * i);
    }
    this.mem[3] = INIT_3_VERTICES;
    this.mec[3] = 0;
    this.mep[3] = new Int32Array(INIT_3_VERTICES * 7);
    this.mne[3] = new Int32Array(INIT_3_VERTICES * 3);
    for (let i = 4; i < this.curOrder; i++) {
      this.mem[i] = INIT_N_VERTICES;
      this.mec[i] = 0;
      this.mep[i] = new Int32Array(INIT_N_VERTICES * ((i << 1) + 1));
      this.mne[i] = new Int32Array(INIT_N_VERTICES * i);
    }
    this.ds = new Int32Array(this.curDel);
    this.ds2 = new Int32Array(this.curDel2);
    this.marg = new Int32Array(this.curMarg);
  }

  /** Cell volume bounds: a box of half-widths given relative to the particle (voronoicell_neighbor::init). */
  init(xmin: number, xmax: number, ymin: number, ymax: number, zmin: number, zmax: number): void {
    this.initBase(xmin, xmax, ymin, ymax, zmin, zmax);
    const q = this.mne[3] as Int32Array;
    q[0] = -5; q[1] = -3; q[2] = -1;
    q[3] = -5; q[4] = -2; q[5] = -3;
    q[6] = -5; q[7] = -1; q[8] = -4;
    q[9] = -5; q[10] = -4; q[11] = -2;
    q[12] = -6; q[13] = -1; q[14] = -3;
    q[15] = -6; q[16] = -3; q[17] = -2;
    q[18] = -6; q[19] = -4; q[20] = -1;
    q[21] = -6; q[22] = -2; q[23] = -4;
    for (let v = 0; v < 8; v++) this.ne[v] = q.subarray(3 * v, 3 * v + 3);
  }

  private initBase(xmin0: number, xmax0: number, ymin0: number, ymax0: number, zmin0: number, zmax0: number): void {
    for (let i = 0; i < this.curOrder; i++) this.mec[i] = 0;
    this.up = 0;
    this.mec[3] = 8;
    this.p = 8;
    const xmin = xmin0 * 2, xmax = xmax0 * 2, ymin = ymin0 * 2, ymax = ymax0 * 2, zmin = zmin0 * 2, zmax = zmax0 * 2;
    const pts = this.pts;
    pts[0] = xmin; pts[1] = ymin; pts[2] = zmin;
    pts[3] = xmax; pts[4] = ymin; pts[5] = zmin;
    pts[6] = xmin; pts[7] = ymax; pts[8] = zmin;
    pts[9] = xmax; pts[10] = ymax; pts[11] = zmin;
    pts[12] = xmin; pts[13] = ymin; pts[14] = zmax;
    pts[15] = xmax; pts[16] = ymin; pts[17] = zmax;
    pts[18] = xmin; pts[19] = ymax; pts[20] = zmax;
    pts[21] = xmax; pts[22] = ymax; pts[23] = zmax;
    const q = this.mep[3] as Int32Array;
    const v = [1, 4, 2, 2, 1, 0, 0, 3, 5, 0, 2, 1, 0, 1, 0, 6, 3, 2, 1, 0, 2, 2, 7, 1, 2, 1, 0, 3, 6, 0, 5, 2, 1, 0, 4, 4, 1, 7, 2, 1, 0, 5, 7, 2, 4, 2, 1, 0, 6, 5, 3, 6, 2, 1, 0, 7];
    for (let t = 0; t < 56; t++) q[t] = v[t];
    for (let k = 0; k < 8; k++) this.ed[k] = q.subarray(7 * k, 7 * k + 7);
    for (let k = 0; k < 8; k++) this.nu[k] = 3;
  }

  /** Maximum squared distance from the cell centre to a vertex (voronoicell_base::max_radius_squared). */
  maxRadiusSquared(): number {
    const pts = this.pts;
    let r = pts[0] * pts[0] + pts[1] * pts[1] + pts[2] * pts[2];
    for (let t = 3; t < 3 * this.p; t += 3) {
      let s = pts[t] * pts[t];
      s += pts[t + 1] * pts[t + 1];
      s += pts[t + 2] * pts[t + 2];
      if (s > r) r = s;
    }
    return r;
  }

  /** Neighbour (plane) ids of the faces, in voro++ face order (voronoicell_neighbor::neighbors). */
  neighbors(): number[] {
    const v: number[] = [];
    const nu = this.nu, ed = this.ed, ne = this.ne;
    for (let i = 1; i < this.p; i++) {
      for (let j = 0; j < nu[i]; j++) {
        let k = ed[i]![j];
        if (k >= 0) {
          v.push(ne[i]![j]);
          ed[i]![j] = -1 - k;
          let l = this.cycleUp(ed[i]![nu[i] + j], k);
          do {
            const m = ed[k]![l];
            ed[k]![l] = -1 - m;
            l = this.cycleUp(ed[k]![nu[k] + l], m);
            k = m;
          } while (k !== i);
        }
      }
    }
    this.resetEdges();
    return v;
  }

  /** Face areas in the same face order as neighbors() (voronoicell_base::face_areas, 1/8 of the Newell sum). */
  faceAreas(): number[] {
    const v: number[] = [];
    const pts = this.pts, nu = this.nu, ed = this.ed;
    for (let i = 1; i < this.p; i++) {
      for (let j = 0; j < nu[i]; j++) {
        let k = ed[i]![j];
        if (k >= 0) {
          let area = 0;
          ed[i]![j] = -1 - k;
          let l = this.cycleUp(ed[i]![nu[i] + j], k);
          let m = ed[k]![l];
          ed[k]![l] = -1 - m;
          while (m !== i) {
            const n = this.cycleUp(ed[k]![nu[k] + l], m);
            const ux = pts[3 * k] - pts[3 * i];
            const uy = pts[3 * k + 1] - pts[3 * i + 1];
            const uz = pts[3 * k + 2] - pts[3 * i + 2];
            const vx = pts[3 * m] - pts[3 * i];
            const vy = pts[3 * m + 1] - pts[3 * i + 1];
            const vz = pts[3 * m + 2] - pts[3 * i + 2];
            const wx = uy * vz - uz * vy;
            const wy = uz * vx - ux * vz;
            const wz = ux * vy - uy * vx;
            area += Math.sqrt(wx * wx + wy * wy + wz * wz);
            k = m;
            l = n;
            m = ed[k]![l];
            ed[k]![l] = -1 - m;
          }
          v.push(0.125 * area);
        }
      }
    }
    this.resetEdges();
    return v;
  }

  private cycleUp(a: number, p: number): number {
    return a === this.nu[p] - 1 ? 0 : a + 1;
  }

  private cycleDown(a: number, p: number): number {
    return a === 0 ? this.nu[p] - 1 : a - 1;
  }

  private resetEdges(): void {
    for (let i = 0; i < this.p; i++) {
      for (let j = 0; j < this.nu[i]; j++) {
        if (this.ed[i]![j] >= 0) fatal('Edge reset routine found a previously untested edge');
        this.ed[i]![j] = -1 - this.ed[i]![j];
      }
    }
  }

  /** Sign of the plane test against vertex n; the value is left in mAns (voronoicell_base::m_test). */
  private mTest(n: number): number {
    const pts = this.pts;
    const base = 3 * n;
    let ans = pts[base] * this.px;
    ans += pts[base + 1] * this.py;
    ans += pts[base + 2] * this.pz - this.prsq;
    this.mAns = ans;
    if (ans < -TOLERANCE2) return -1;
    else if (ans > TOLERANCE2) return 1;
    return this.checkMarginal(n, ans);
  }

  private checkMarginal(n: number, ans: number): number {
    for (let i = 0; i < this.nMarg; i += 2) if (this.marg[i] === n) return this.marg[i + 1];
    if (this.nMarg === this.curMarg) {
      this.curMarg <<= 1;
      if (this.curMarg > MAX_MARGINAL) fatal('Marginal case buffer allocation exceeded absolute maximum');
      const pm = new Int32Array(this.curMarg);
      for (let j = 0; j < this.nMarg; j++) pm[j] = this.marg[j];
      this.marg = pm;
    }
    this.marg[this.nMarg++] = n;
    this.marg[this.nMarg++] = ans > TOLERANCE ? 1 : ans < -TOLERANCE ? -1 : 0;
    return this.marg[this.nMarg - 1];
  }

  private addMemoryVertices(): void {
    const i = this.curVertices << 1;
    if (i > MAX_VERTICES) fatal('Vertex memory allocation exceeded absolute maximum');
    const ed = new Array<Int32Array | undefined>(i), ne = new Array<Int32Array | undefined>(i);
    for (let j = 0; j < this.curVertices; j++) { ed[j] = this.ed[j]; ne[j] = this.ne[j]; }
    this.ed = ed;
    this.ne = ne;
    const nu = new Int32Array(i);
    for (let j = 0; j < this.curVertices; j++) nu[j] = this.nu[j];
    this.nu = nu;
    const pts = new Float64Array(3 * i);
    for (let j = 0; j < 3 * this.curVertices; j++) pts[j] = this.pts[j];
    this.pts = pts;
    this.curVertices = i;
  }

  private addMemoryVorder(): void {
    const i = this.curOrder << 1;
    if (i > MAX_VERTEX_ORDER) fatal('Vertex order memory allocation exceeded absolute maximum');
    const mem = new Int32Array(i), mec = new Int32Array(i);
    for (let j = 0; j < this.curOrder; j++) { mem[j] = this.mem[j]; mec[j] = this.mec[j]; }
    this.mem = mem;
    this.mec = mec;
    const mep = new Array<Int32Array | undefined>(i), mne = new Array<Int32Array | undefined>(i);
    for (let j = 0; j < this.curOrder; j++) { mep[j] = this.mep[j]; mne[j] = this.mne[j]; }
    this.mep = mep;
    this.mne = mne;
    this.curOrder = i;
  }

  private addMemoryDs(sp: number): number {
    this.curDel <<= 1;
    if (this.curDel > MAX_DELETE_SIZE) fatal('Delete stack 1 memory allocation exceeded absolute maximum');
    const dsn = new Int32Array(this.curDel);
    for (let t = 0; t < sp; t++) dsn[t] = this.ds[t];
    this.ds = dsn;
    return sp;
  }

  private addMemoryDs2(sp2: number): number {
    this.curDel2 <<= 1;
    if (this.curDel2 > MAX_DELETE2_SIZE) fatal('Delete stack 2 memory allocation exceeded absolute maximum');
    const dsn = new Int32Array(this.curDel2);
    for (let t = 0; t < sp2; t++) dsn[t] = this.ds2[t];
    this.ds2 = dsn;
    return sp2;
  }

  /** Doubles the slot block of order i and re-points every row that lives in it (voronoicell_base::add_memory). */
  private addMemory(i: number, stackp2: number): void {
    const s = (i << 1) + 1;
    if (this.mem[i] === 0) {
      this.mne[i] = new Int32Array(INIT_N_VERTICES * i);
      this.mep[i] = new Int32Array(INIT_N_VERTICES * s);
      this.mem[i] = INIT_N_VERTICES;
      return;
    }
    let j = 0, k: number, m = 0;
    this.mem[i] <<= 1;
    if (this.mem[i] > MAX_N_VERTICES) fatal('Point memory allocation exceeded absolute maximum');
    const l = new Int32Array(s * this.mem[i]);
    const newNe = new Int32Array(i * this.mem[i]);
    this.paux1 = newNe;
    const oldMep = this.mep[i] as Int32Array;
    const oldNe = this.mne[i] as Int32Array;
    while (j < s * this.mec[i]) {
      k = oldMep[j + (i << 1)];
      if (k >= 0) {
        this.ed[k] = l.subarray(j, j + s);
        this.ne[k] = newNe.subarray(m, m + i);
      } else {
        let found = false;
        for (let dsp = 0; dsp < stackp2; dsp++) {
          const vtx = this.ds2[dsp];
          if (samePtr(this.ed[vtx], oldMep, j)) {
            this.ed[vtx] = l.subarray(j, j + s);
            this.ne[vtx] = newNe.subarray(m, m + i);
            found = true;
            break;
          }
        }
        if (!found) fatal("Couldn't relocate dangling pointer");
      }
      for (k = 0; k < s; k++, j++) l[j] = oldMep[j];
      for (k = 0; k < i; k++, m++) newNe[m] = oldNe[m];
    }
    this.mep[i] = l;
    this.mne[i] = newNe;
  }

  /** Cut the cell by the plane at (x,y,z) with squared distance rsq, plane id pId (voronoicell_base::nplane). */
  nplane(x: number, y: number, z: number, rsq: number, pId: number): boolean {
    let count = 0, i = 0, j = 0, k = 0, lp = this.up, cp = 0, qp = 0, rp = 0;
    let stackp = 0, stackp2 = 0, dsp = 0;
    let us = 0, ls = 0, qs = 0, iqs = 0, cs = 0, uw = 0, qw = 0, lw = 0;
    let edp: Int32Array, edd: Int32Array;
    let u = 0, l = 0, r = 0, q = 0;
    let complicatedSetup = false, newDoubleEdge = false, doubleEdge = false;
    this.nMarg = 0;
    this.px = x; this.py = y; this.pz = z; this.prsq = rsq;
    uw = this.mTest(this.up);
    u = this.mAns;
    try {
      if (uw === 1) {
        us = 0;
        do {
          lp = this.ed[this.up]![us];
          lw = this.mTest(lp);
          l = this.mAns;
          if (l < u) break;
          us++;
        } while (us < this.nu[this.up]);
        if (us === this.nu[this.up]) return false;
        ls = this.ed[this.up]![this.nu[this.up] + us];
        while (lw === 1) {
          if (++count >= this.p) throw true;
          u = l;
          this.up = lp;
          for (us = 0; us < ls; us++) {
            lp = this.ed[this.up]![us];
            lw = this.mTest(lp);
            l = this.mAns;
            if (l < u) break;
          }
          if (us === ls) {
            us++;
            while (us < this.nu[this.up]) {
              lp = this.ed[this.up]![us];
              lw = this.mTest(lp);
              l = this.mAns;
              if (l < u) break;
              us++;
            }
            if (us === this.nu[this.up]) return false;
          }
          ls = this.ed[this.up]![this.nu[this.up] + us];
        }
        if (lw === 0) {
          this.up = lp;
          complicatedSetup = true;
        } else complicatedSetup = false;
      } else if (uw === -1) {
        us = 0;
        do {
          qp = this.ed[this.up]![us];
          qw = this.mTest(qp);
          q = this.mAns;
          if (u < q) break;
          us++;
        } while (us < this.nu[this.up]);
        if (us === this.nu[this.up]) return true;
        while (qw === -1) {
          qs = this.ed[this.up]![this.nu[this.up] + us];
          if (++count >= this.p) throw true;
          u = q;
          this.up = qp;
          for (us = 0; us < qs; us++) {
            qp = this.ed[this.up]![us];
            qw = this.mTest(qp);
            q = this.mAns;
            if (u < q) break;
          }
          if (us === qs) {
            us++;
            while (us < this.nu[this.up]) {
              qp = this.ed[this.up]![us];
              qw = this.mTest(qp);
              q = this.mAns;
              if (u < q) break;
              us++;
            }
            if (us === this.nu[this.up]) return true;
          }
        }
        if (qw === 1) {
          lp = this.up;
          ls = us;
          l = u;
          this.up = qp;
          us = this.ed[lp]![this.nu[lp] + ls];
          u = q;
          complicatedSetup = false;
        } else {
          this.up = qp;
          complicatedSetup = true;
        }
      } else {
        complicatedSetup = true;
      }
    } catch (e) {
      if (e !== true) throw e;
      qw = 1;
      lw = 0;
      for (qp = 0; qp < this.p; qp++) {
        qw = this.mTest(qp);
        q = this.mAns;
        if (qw === 1) {
          for (us = 0; us < this.nu[qp]; us++) {
            lp = this.ed[qp]![us];
            if (lp < qp) {
              lw = this.mTest(lp);
              l = this.mAns;
              if (lw !== 1) break;
            }
          }
          if (us < this.nu[qp]) {
            this.up = qp;
            if (lw === 0) {
              complicatedSetup = true;
            } else {
              complicatedSetup = false;
              u = q;
              ls = this.ed[this.up]![this.nu[this.up] + us];
            }
            break;
          }
        } else if (qw === -1) {
          for (ls = 0; ls < this.nu[qp]; ls++) {
            this.up = this.ed[qp]![ls];
            if (this.up < qp) {
              uw = this.mTest(this.up);
              u = this.mAns;
              if (uw !== -1) break;
            }
          }
          if (ls < this.nu[qp]) {
            if (uw === 0) {
              this.up = qp;
              complicatedSetup = true;
            } else {
              complicatedSetup = false;
              lp = qp;
              l = q;
              us = this.ed[lp]![this.nu[lp] + ls];
            }
            break;
          }
        } else {
          this.up = qp;
          complicatedSetup = true;
          break;
        }
      }
      if (qp === this.p) return qw === -1;
    }
    if (this.p === this.curVertices) this.addMemoryVertices();
    if (complicatedSetup) {
      if (!this.searchForOutsideEdge()) return false;
      const up0 = this.up;
      const pts = this.pts;
      pts[3 * this.p] = pts[3 * up0];
      pts[3 * this.p + 1] = pts[3 * up0 + 1];
      pts[3 * this.p + 2] = pts[3 * up0 + 2];
      i = 0;
      const up = this.up;
      lp = this.ed[up]![0];
      lw = this.mTest(lp);
      l = this.mAns;
      if (lw !== -1) {
        rp = lw;
        do {
          i++;
          if (i === this.nu[up]) return false;
          lp = this.ed[up]![i];
          lw = this.mTest(lp);
          l = this.mAns;
        } while (lw !== -1);
        j = i + 1;
        while (j < this.nu[up]) {
          lp = this.ed[up]![j];
          lw = this.mTest(lp);
          l = this.mAns;
          if (lw !== -1) break;
          j++;
        }
        if (j === this.nu[up] && i === 1 && rp === 0) {
          this.nu[this.p] = this.nu[up];
          doubleEdge = true;
        } else this.nu[this.p] = j - i + 2;
        k = 1;
        while (this.nu[this.p] >= this.curOrder) this.addMemoryVorder();
        if (this.mec[this.nu[this.p]] === this.mem[this.nu[this.p]]) stackp2 = this.addMemoryWrap(this.nu[this.p], stackp2);
        this.setPointer(this.p, this.nu[this.p]);
        const o = this.nu[this.p];
        this.ed[this.p] = this.slotEd(o);
        this.ed[this.p]![o << 1] = this.p;
        us = this.cycleDown(i, up);
        while (i < j) {
          qp = this.ed[up]![i];
          qs = this.ed[up]![this.nu[up] + i];
          this.copyNe(this.p, k, up, i);
          this.ed[this.p]![k] = qp;
          this.ed[this.p]![this.nu[this.p] + k] = qs;
          this.ed[qp]![qs] = this.p;
          this.ed[qp]![this.nu[qp] + qs] = k;
          this.ed[up]![i] = -1;
          i++;
          k++;
        }
        qs = i === this.nu[up] ? 0 : i;
      } else {
        i = this.nu[up] - 1;
        lp = this.ed[up]![i];
        lw = this.mTest(lp);
        l = this.mAns;
        while (lw === -1) {
          i--;
          if (i === 0) return true;
          lp = this.ed[up]![i];
          lw = this.mTest(lp);
          l = this.mAns;
        }
        j = 1;
        qp = this.ed[up]![j];
        qw = this.mTest(qp);
        q = this.mAns;
        while (qw === -1) {
          j++;
          qp = this.ed[up]![j];
          qw = this.mTest(qp);
          // voro++ 0.4.6 evaluates this test at l, not q (kept as in the source)
          l = this.mAns;
        }
        if (i === j && qw === 0) {
          doubleEdge = true;
          this.nu[this.p] = this.nu[up];
        } else {
          this.nu[this.p] = this.nu[up] - i + j + 1;
        }
        k = 1;
        while (this.nu[this.p] >= this.curOrder) this.addMemoryVorder();
        if (this.mec[this.nu[this.p]] === this.mem[this.nu[this.p]]) stackp2 = this.addMemoryWrap(this.nu[this.p], stackp2);
        this.setPointer(this.p, this.nu[this.p]);
        const o = this.nu[this.p];
        this.ed[this.p] = this.slotEd(o);
        this.ed[this.p]![o << 1] = this.p;
        us = i++;
        while (i < this.nu[up]) {
          qp = this.ed[up]![i];
          qs = this.ed[up]![this.nu[up] + i];
          this.copyNe(this.p, k, up, i);
          this.ed[this.p]![k] = qp;
          this.ed[this.p]![this.nu[this.p] + k] = qs;
          this.ed[qp]![qs] = this.p;
          this.ed[qp]![this.nu[qp] + qs] = k;
          this.ed[up]![i] = -1;
          i++;
          k++;
        }
        i = 0;
        while (i < j) {
          qp = this.ed[up]![i];
          qs = this.ed[up]![this.nu[up] + i];
          this.copyNe(this.p, k, up, i);
          this.ed[this.p]![k] = qp;
          this.ed[this.p]![this.nu[this.p] + k] = qs;
          this.ed[qp]![qs] = this.p;
          this.ed[qp]![this.nu[qp] + qs] = k;
          this.ed[up]![i] = -1;
          i++;
          k++;
        }
        qs = j;
      }
      if (!doubleEdge) {
        this.copyNe(this.p, k, up, qs);
        this.ne[this.p]![0] = pId;
      } else this.copyNe(this.p, 0, up, qs);
      stackp2 = this.pushDs2(stackp2, up);
      cs = k;
      qp = up;
      q = u;
      i = this.ed[up]![us];
      us = this.ed[up]![this.nu[up] + us];
      this.up = i;
      this.ed[qp]![this.nu[qp] << 1] = -this.p;
    } else {
      stackp = this.pushDs(stackp, this.up);
      r = u / (u - l);
      l = 1 - r;
      const pts = this.pts;
      pts[3 * this.p] = pts[3 * lp] * r + pts[3 * this.up] * l;
      pts[3 * this.p + 1] = pts[3 * lp + 1] * r + pts[3 * this.up + 1] * l;
      pts[3 * this.p + 2] = pts[3 * lp + 2] * r + pts[3 * this.up + 2] * l;
      this.nu[this.p] = 3;
      if (this.mec[3] === this.mem[3]) stackp2 = this.addMemoryWrap(3, stackp2);
      this.setPointer(this.p, 3);
      this.ne[this.p]![0] = pId;
      this.copyNe(this.p, 1, this.up, us);
      this.copyNe(this.p, 2, lp, ls);
      this.ed[this.p] = this.slotEd(3);
      this.ed[this.p]![6] = this.p;
      this.ed[this.up]![us] = -1;
      this.ed[lp]![ls] = this.p;
      this.ed[lp]![this.nu[lp] + ls] = 1;
      this.ed[this.p]![1] = lp;
      this.ed[this.p]![this.nu[this.p] + 1] = ls;
      cs = 2;
      qs = this.cycleUp(us, this.up);
      qp = this.up;
      q = u;
    }
    cp = this.p;
    rp = this.p;
    this.p++;
    while (qp !== this.up || qs !== us) {
      lp = this.ed[qp]![qs];
      lw = this.mTest(lp);
      l = this.mAns;
      if (lw === 1) {
        qs = this.cycleUp(this.ed[qp]![this.nu[qp] + qs], lp);
        qp = lp;
        q = l;
        stackp = this.pushDs(stackp, qp);
      } else if (lw === -1) {
        if (this.p === this.curVertices) this.addMemoryVertices();
        r = q / (q - l);
        l = 1 - r;
        const pts = this.pts;
        pts[3 * this.p] = pts[3 * lp] * r + pts[3 * qp] * l;
        pts[3 * this.p + 1] = pts[3 * lp + 1] * r + pts[3 * qp + 1] * l;
        pts[3 * this.p + 2] = pts[3 * lp + 2] * r + pts[3 * qp + 2] * l;
        this.nu[this.p] = 3;
        if (this.mec[3] === this.mem[3]) stackp2 = this.addMemoryWrap(3, stackp2);
        ls = this.ed[qp]![qs + this.nu[qp]];
        this.setPointer(this.p, 3);
        this.ne[this.p]![0] = pId;
        this.copyNe(this.p, 1, qp, qs);
        this.copyNe(this.p, 2, lp, ls);
        this.ed[this.p] = this.slotEd(3);
        const e = this.ed[this.p]!;
        e[0] = cp;
        e[1] = lp;
        e[3] = cs;
        e[4] = ls;
        e[6] = this.p;
        this.ed[lp]![ls] = this.p;
        this.ed[lp]![this.nu[lp] + ls] = 1;
        this.ed[cp]![cs] = this.p;
        this.ed[cp]![this.nu[cp] + cs] = 0;
        this.ed[qp]![qs] = -1;
        qs = this.cycleUp(qs, qp);
        cp = this.p++;
        cs = 2;
      } else {
        if (this.p === this.curVertices) this.addMemoryVertices();
        k = doubleEdge ? 0 : 1;
        qs = this.ed[qp]![this.nu[qp] + qs];
        qp = lp;
        iqs = qs;
        do {
          k++;
          qs = this.cycleUp(qs, qp);
          lp = this.ed[qp]![qs];
          lw = this.mTest(lp);
          l = this.mAns;
        } while (lw === -1);
        j = -this.ed[qp]![this.nu[qp] << 1];
        if (qp === this.up && qs === us) {
          newDoubleEdge = false;
          if (j > 0) k += this.nu[j];
        } else {
          if (j > 0) {
            k += this.nu[j];
            if (lw === 0) {
              i = -this.ed[lp]![this.nu[lp] << 1];
              if (i > 0) {
                if (this.ed[i]![this.nu[i] - 1] === j) {
                  newDoubleEdge = true;
                  k -= 1;
                } else newDoubleEdge = false;
              } else {
                if (j === rp && lp === this.up && this.ed[qp]![this.nu[qp] + qs] === us) {
                  newDoubleEdge = true;
                  k -= 1;
                } else newDoubleEdge = false;
              }
            } else newDoubleEdge = false;
          } else {
            if (lw === 0) {
              i = -this.ed[lp]![this.nu[lp] << 1];
              if (i === cp) {
                newDoubleEdge = true;
                k -= 1;
              } else newDoubleEdge = false;
            } else newDoubleEdge = false;
          }
        }
        while (k >= this.curOrder) this.addMemoryVorder();
        if (this.mec[k] === this.mem[k]) stackp2 = this.addMemoryWrap(k, stackp2);
        if (j > 0) {
          if (this.nu[j] !== k) {
            this.setAux1(k);
            const sk = (k << 1) + 1;
            edp = (this.mep[k] as Int32Array).subarray(sk * this.mec[k], sk * this.mec[k] + sk);
            this.mec[k]++;
            i = 0;
            while (i < this.nu[j]) {
              this.paux1[i] = this.ne[j]![i];
              edp[i] = this.ed[j]![i];
              edp[k + i] = this.ed[j]![this.nu[j] + i];
              i++;
            }
            edp[k << 1] = j;
            this.mec[this.nu[j]]--;
            const sj = (this.nu[j] << 1) + 1;
            edd = (this.mep[this.nu[j]] as Int32Array).subarray(sj * this.mec[this.nu[j]], sj * this.mec[this.nu[j]] + sj);
            if (!samePtrView(this.ed[j]!, edd)) {
              for (lw = 0; lw <= (this.nu[j] << 1); lw++) this.ed[j]![lw] = edd[lw];
              this.setAux2Copy(j, this.nu[j]);
              this.ne[edd[this.nu[j] << 1]] = this.ne[j];
              this.ed[edd[this.nu[j] << 1]] = this.ed[j];
            }
            this.ne[j] = this.paux1;
            this.ed[j] = edp;
          } else i = this.nu[j];
        } else {
          this.setPointer(this.p, k);
          this.ed[this.p] = this.slotEd(k);
          this.ed[this.p]![k << 1] = this.p;
          stackp2 = this.pushDs2(stackp2, qp);
          const pts = this.pts;
          pts[3 * this.p] = pts[3 * qp];
          pts[3 * this.p + 1] = pts[3 * qp + 1];
          pts[3 * this.p + 2] = pts[3 * qp + 2];
          this.ed[qp]![this.nu[qp] << 1] = -this.p;
          j = this.p++;
          i = 0;
        }
        this.nu[j] = k;
        if (!doubleEdge) {
          this.ed[j]![i] = cp;
          this.ed[j]![this.nu[j] + i] = cs;
          this.ne[j]![i] = pId;
          this.ed[cp]![cs] = j;
          this.ed[cp]![this.nu[cp] + cs] = i;
          i++;
        }
        qs = iqs;
        while (i < (newDoubleEdge ? k : k - 1)) {
          qs = this.cycleUp(qs, qp);
          lp = this.ed[qp]![qs];
          ls = this.ed[qp]![this.nu[qp] + qs];
          this.copyNe(j, i, qp, qs);
          this.ed[j]![i] = lp;
          this.ed[j]![this.nu[j] + i] = ls;
          this.ed[lp]![ls] = j;
          this.ed[lp]![this.nu[lp] + ls] = i;
          this.ed[qp]![qs] = -1;
          i++;
        }
        qs = this.cycleUp(qs, qp);
        cs = i;
        cp = j;
        this.copyNe(j, newDoubleEdge ? 0 : cs, qp, qs);
        doubleEdge = newDoubleEdge;
      }
    }
    this.ed[cp]![cs] = rp;
    this.ed[rp]![0] = cp;
    this.ed[cp]![this.nu[cp] + cs] = 0;
    this.ed[rp]![this.nu[rp]] = cs;
    dsp = 0;
    while (dsp < stackp) {
      j = this.ds[dsp];
      if (this.ed[j]![this.nu[j]] !== -1) {
        this.ed[j]![this.nu[j]] = -1;
        dsp++;
      } else this.ds[dsp] = this.ds[--stackp];
    }
    for (dsp = 0; dsp < stackp2; dsp++) {
      j = this.ds2[dsp];
      this.ed[j]![this.nu[j] << 1] = j;
      if (this.ed[j]![this.nu[j]] !== -1) {
        this.ed[j]![this.nu[j]] = -1;
        stackp = this.pushDs(stackp, j);
      }
    }
    for (dsp = 0; dsp < stackp; dsp++) {
      cp = this.ds[dsp];
      const row = this.ed[cp]!;
      for (let e = 0; e < this.nu[cp]; e++) {
        qp = row[e];
        if (qp !== -1 && this.ed[qp]![this.nu[qp]] !== -1) {
          stackp = this.pushDs(stackp, qp);
          this.ed[qp]![this.nu[qp]] = -1;
        }
      }
    }
    this.up = 0;
    while (stackp > 0) {
      --this.p;
      while (this.ed[this.p]![this.nu[this.p]] === -1) {
        j = this.nu[this.p];
        const sj = (j << 1) + 1;
        this.mec[j]--;
        edd = (this.mep[j] as Int32Array).subarray(sj * this.mec[j], sj * this.mec[j] + sj);
        const row = this.ed[this.p]!;
        for (let t = 0; t <= (j << 1); t++) row[t] = edd[t];
        this.setAux2Copy(this.p, j);
        const vid = row[j << 1];
        this.ne[vid] = this.ne[this.p];
        this.ed[vid] = row;
        --this.p;
      }
      stackp--;
      this.up = this.ds[stackp];
      if (this.up < this.p) {
        const pts = this.pts;
        pts[3 * this.up] = pts[3 * this.p];
        pts[3 * this.up + 1] = pts[3 * this.p + 1];
        pts[3 * this.up + 2] = pts[3 * this.p + 2];
        j = this.nu[this.up];
        const sj = (j << 1) + 1;
        this.mec[j]--;
        edd = (this.mep[j] as Int32Array).subarray(sj * this.mec[j], sj * this.mec[j] + sj);
        const row = this.ed[this.up]!;
        for (let t = 0; t <= (j << 1); t++) row[t] = edd[t];
        this.setAux2Copy(this.up, j);
        const vid = row[j << 1];
        this.ne[vid] = this.ne[this.up];
        this.ne[this.up] = this.ne[this.p];
        this.ed[vid] = row;
        this.ed[this.up] = this.ed[this.p];
        this.nu[this.up] = this.nu[this.p];
        for (i = 0; i < this.nu[this.up]; i++) this.ed[this.ed[this.up]![i]]![this.ed[this.up]![this.nu[this.up] + i]] = this.up;
        this.ed[this.up]![this.nu[this.up] << 1] = this.up;
      } else this.up = this.p++;
    }
    if (this.mec[0] > 0) fatal('Zero order vertex formed');
    return this.collapseOrder2();
  }

  /** Pushes vertex v on the delete stack (ds) and returns the new top. */
  private pushDs(sp: number, v: number): number {
    if (sp === this.ds.length) sp = this.addMemoryDs(sp);
    this.ds[sp++] = v;
    return sp;
  }

  private pushDs2(sp2: number, v: number): number {
    if (sp2 === this.ds2.length) sp2 = this.addMemoryDs2(sp2);
    this.ds2[sp2++] = v;
    return sp2;
  }

  /**
   * voronoicell_base::search_for_outside_edge: the search keeps its own stack in ds2 from index 0, so the
   * caller's auxiliary stack top (nplane's stackp2) is not changed.
   */
  private searchForOutsideEdge(): boolean {
    let j = 0;
    let stackp2 = 0;
    stackp2 = this.pushDs2(stackp2, this.up);
    while (j < stackp2) {
      this.up = this.ds2[j++];
      for (let i = 0; i < this.nu[this.up]; i++) {
        const lp = this.ed[this.up]![i];
        const lw = this.mTest(lp);
        if (lw === -1) return true;
        else if (lw === 0) stackp2 = this.addToStack(lp, stackp2);
      }
    }
    return false;
  }

  private addToStack(lp: number, stackp2: number): number {
    for (let k = 0; k < stackp2; k++) if (this.ds2[k] === lp) return stackp2;
    return this.pushDs2(stackp2, lp);
  }

  /** Slot row of order o from the current block (the next free slot, stride 2o+1). */
  private slotEd(o: number): Int32Array {
    const s = (o << 1) + 1;
    const block = this.mep[o] as Int32Array;
    const start = s * this.mec[o];
    this.mec[o]++;
    return block.subarray(start, start + s);
  }

  /** ne[p] = row of order n in the neighbour block (voronoicell_neighbor::n_set_pointer). */
  private setPointer(p: number, n: number): void {
    const block = this.mne[n] as Int32Array;
    const start = n * this.mec[n];
    this.ne[p] = block.subarray(start, start + n);
  }

  /** ne[a][b] = ne[c][d] */
  private copyNe(a: number, b: number, c: number, d: number): void {
    this.ne[a]![b] = this.ne[c]![d];
  }

  /** The order-o block grows and the slots of this order are relocated (add_memory with the neighbour block). */
  private addMemoryWrap(i: number, stackp2: number): number {
    this.addMemory(i, stackp2);
    return stackp2;
  }

  private setAux1(k: number): void {
    const block = this.mne[k] as Int32Array;
    this.paux1 = block.subarray(k * this.mec[k], k * this.mec[k] + k);
  }

  /** paux2 = last neighbour slot of order b; ne[a][0..b) = paux2 (n_set_aux2_copy). */
  private setAux2Copy(a: number, b: number): void {
    const block = this.mne[b] as Int32Array;
    this.paux2 = block.subarray(b * this.mec[b], b * this.mec[b] + b);
    for (let i = 0; i < b; i++) this.ne[a]![i] = this.paux2[i];
  }

  /** Removes the connection of vertex j at position k (delete_connection); hand selects the cycle side. */
  private deleteConnection(j: number, k0: number, hand: boolean): boolean {
    let k = k0;
    const q = hand ? k : this.cycleUp(k, j);
    const i = this.nu[j] - 1;
    let l: number, m: number;
    if (this.mec[i] === this.mem[i]) this.addMemory(i, 0);
    this.setAux1(i);
    const paux1 = this.paux1;
    for (l = 0; l < q; l++) paux1[l] = this.ne[j]![l];
    while (l < i) {
      paux1[l] = this.ne[j]![l + 1];
      l++;
    }
    const s = (i << 1) + 1;
    const block = this.mep[i] as Int32Array;
    const start = s * this.mec[i];
    const edp = block.subarray(start, start + s);
    this.mec[i]++;
    edp[i << 1] = j;
    for (l = 0; l < k; l++) {
      edp[l] = this.ed[j]![l];
      edp[l + i] = this.ed[j]![l + this.nu[j]];
    }
    while (l < i) {
      m = this.ed[j]![l + 1];
      edp[l] = m;
      k = this.ed[j]![l + this.nu[j] + 1];
      edp[l + i] = k;
      this.ed[m]![this.nu[m] + k]--;
      l++;
    }
    const sj = (this.nu[j] << 1) + 1;
    this.mec[this.nu[j]]--;
    const edd = (this.mep[this.nu[j]] as Int32Array).subarray(sj * this.mec[this.nu[j]], sj * this.mec[this.nu[j]] + sj);
    for (l = 0; l <= (this.nu[j] << 1); l++) this.ed[j]![l] = edd[l];
    this.setAux2Copy(j, this.nu[j]);
    const vid = edd[this.nu[j] << 1];
    this.ne[vid] = this.paux2;
    this.ne[j] = this.paux1;
    this.ed[vid] = edd;
    this.ed[j] = edp;
    this.nu[j] = i;
    return true;
  }

  private collapseOrder1(): boolean {
    while (this.mec[1] > 0) {
      this.up = 0;
      this.mec[1]--;
      const idx = this.mec[1];
      const blk = this.mep[1] as Int32Array;
      const j = blk[3 * idx];
      let k = blk[3 * idx + 1];
      let i = blk[3 * idx + 2];
      if (!this.deleteConnection(j, k, false)) return false;
      --this.p;
      if (this.up === i) this.up = 0;
      if (this.p !== i) {
        if (this.up === this.p) this.up = i;
        this.pts[3 * i] = this.pts[3 * this.p];
        this.pts[3 * i + 1] = this.pts[3 * this.p + 1];
        this.pts[3 * i + 2] = this.pts[3 * this.p + 2];
        for (k = 0; k < this.nu[this.p]; k++) this.ed[this.ed[this.p]![k]]![this.ed[this.p]![this.nu[this.p] + k]] = i;
        this.ne[i] = this.ne[this.p];
        this.ed[i] = this.ed[this.p];
        this.nu[i] = this.nu[this.p];
        this.ed[i]![this.nu[i] << 1] = i;
      }
    }
    return true;
  }

  private collapseOrder2(): boolean {
    if (!this.collapseOrder1()) return false;
    let a: number, b: number, i: number, j: number, k: number, l: number;
    while (this.mec[2] > 0) {
      this.mec[2]--;
      i = this.mec[2];
      const blk = this.mep[2] as Int32Array;
      j = blk[5 * i];
      k = blk[5 * i + 1];
      if (j === k) return false;
      for (l = 0; l < this.nu[j]; l++) if (this.ed[j]![l] === k) break;
      a = blk[5 * i + 2];
      b = blk[5 * i + 3];
      i = blk[5 * i + 4];
      if (l === this.nu[j]) {
        this.ed[j]![a] = k;
        this.ed[k]![b] = j;
        this.ed[j]![this.nu[j] + a] = b;
        this.ed[k]![this.nu[k] + b] = a;
      } else {
        if (!this.deleteConnection(j, a, false)) return false;
        if (!this.deleteConnection(k, b, true)) return false;
      }
      --this.p;
      if (this.up === i) this.up = 0;
      if (this.p !== i) {
        if (this.up === this.p) this.up = i;
        this.pts[3 * i] = this.pts[3 * this.p];
        this.pts[3 * i + 1] = this.pts[3 * this.p + 1];
        this.pts[3 * i + 2] = this.pts[3 * this.p + 2];
        for (k = 0; k < this.nu[this.p]; k++) this.ed[this.ed[this.p]![k]]![this.ed[this.p]![this.nu[this.p] + k]] = i;
        this.ne[i] = this.ne[this.p];
        this.ed[i] = this.ed[this.p];
        this.nu[i] = this.nu[this.p];
        this.ed[i]![this.nu[i] << 1] = i;
      }
      if (!this.collapseOrder1()) return false;
    }
    return true;
  }

  /** Whether the plane (x,y,z,rsq) cuts the cell (voronoicell_base::plane_intersects). */
  planeIntersects(x: number, y: number, z: number, rsq: number): boolean {
    const pts = this.pts;
    const g = x * pts[3 * this.up] + y * pts[3 * this.up + 1] + z * pts[3 * this.up + 2];
    if (g < rsq) return this.planeIntersectsTrack(x, y, z, rsq, g);
    return true;
  }

  /** voronoicell_base::plane_intersects_guess: starts the search from the best vertex of a sample of vertices. */
  planeIntersectsGuess(x: number, y: number, z: number, rsq: number): boolean {
    this.up = 0;
    const pts = this.pts;
    let g = x * pts[0] + y * pts[1] + z * pts[2];
    if (g < rsq) {
      let ca = 1;
      const cc = this.p >> 3;
      let mp = 1;
      let m: number;
      while (ca < cc) {
        m = x * pts[3 * mp] + y * pts[3 * mp + 1] + z * pts[3 * mp + 2];
        if (m > g) {
          if (m > rsq) return true;
          g = m;
          this.up = mp;
        }
        ca += mp++;
      }
      return this.planeIntersectsTrack(x, y, z, rsq, g);
    }
    return true;
  }

  private planeIntersectsTrack(x: number, y: number, z: number, rsq: number, g0: number): boolean {
    const pts = this.pts;
    let g = g0;
    let count = 0, ls: number, us: number, tp: number;
    let t: number;
    for (us = 0; us < this.nu[this.up]; us++) {
      tp = this.ed[this.up]![us];
      t = x * pts[3 * tp] + y * pts[3 * tp + 1] + z * pts[3 * tp + 2];
      if (t > g) {
        ls = this.ed[this.up]![this.nu[this.up] + us];
        this.up = tp;
        while (t < rsq) {
          if (++count >= this.p) {
            for (tp = 0; tp < this.p; tp++) if (x * pts[3 * tp] + y * pts[3 * tp + 1] + z * pts[3 * tp + 2] > rsq) return true;
            return false;
          }
          for (us = 0; us < ls; us++) {
            tp = this.ed[this.up]![us];
            g = x * pts[3 * tp] + y * pts[3 * tp + 1] + z * pts[3 * tp + 2];
            if (g > t) break;
          }
          if (us === ls) {
            us++;
            while (us < this.nu[this.up]) {
              tp = this.ed[this.up]![us];
              g = x * pts[3 * tp] + y * pts[3 * tp + 1] + z * pts[3 * tp + 2];
              if (g > t) break;
              us++;
            }
            if (us === this.nu[this.up]) return false;
          }
          ls = this.ed[this.up]![this.nu[this.up] + us];
          this.up = tp;
          t = g;
        }
        return true;
      }
    }
    return false;
  }
}
