/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the voro++-derived Voronoi cell of the PTM library (src/PTM/ptm_voronoi_cell.cpp, ptm_voronoi_cell.h, ptm_voronoi_config.h), modified by PM Larsen for Polyhedral Template Matching. voro++ is Copyright (c) 2008, The Regents of the University of California (BSD-3, see LICENSE.voro++ next to this file). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
/*
Voro++ Copyright (c) 2008, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a  non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
*/


// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011
//
// Modified by PM Larsen for use in Polyhedral Template Matching

/*
 * Translation notes (C -> TypeScript):
 * - C pointers into the mep/mne memory pools are Int32Array views (subarray) sharing the
 *   pool's ArrayBuffer, so ed[i][j] and ne[i][j] keep the C indexing. Pointer equality is
 *   tested with sameRecord()/sameRecordAt().
 * - Pointers into the delete stacks (ds, ds2) are indices into number[] arrays. Growing a
 *   stack keeps every index valid, so add_memory_ds()/add_memory_ds2() leave the stack
 *   pointer unchanged and the C "stackp=dsnp" assignment is implicit.
 * - Pass-by-reference scalars: m_test() returns the int and stores its double output in
 *   m_test_ans; add_to_stack() returns the updated stackp2; search_for_outside_edge() works
 *   on this.up, which is the only argument it is ever called with.
 * - The template nplane(vc_class&,...) body is nplane_impl(vc,...); the public
 *   nplane(x,y,z,rsq,p_id) overloads live on voronoicell_neighbor.
 * - Error paths that exit() or fprintf(stderr) throw an Error with the same message text.
 */

import {
  init_vertices,
  init_vertex_order,
  init_3_vertices,
  init_n_vertices,
  init_marginal,
  init_delete_size,
  init_delete2_size,
  max_vertices,
  max_vertex_order,
  max_n_vertices,
  max_marginal,
  max_delete_size,
  max_delete2_size,
  tolerance,
  tolerance2,
  VOROPP_MEMORY_ERROR,
  VOROPP_INTERNAL_ERROR,
} from './ptm_voronoi_config';

/** Error raised where the C code calls voro_fatal_error (which exits). */
class VoroFatalError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'VoroFatalError';
    this.status = status;
  }
}

function voro_fatal_error(p: string, status: number): never {
  throw new VoroFatalError(`voro++: ${p}`, status);
}

/** Shared zero-length record used as the "null" pointer value of ed/ne/mep/mne. */
const EMPTY_INTS = new Int32Array(0);

function zeros(n: number): number[] {
  return new Array<number>(n).fill(0);
}

function emptyRecords(n: number): Int32Array[] {
  return new Array<Int32Array>(n).fill(EMPTY_INTS);
}

/** C pointer equality for two int* that point into pool arrays: a == b. */
function sameRecord(a: Int32Array, b: Int32Array): boolean {
  return a.buffer === b.buffer && a.byteOffset === b.byteOffset;
}

/** C pointer equality: a == base + j. */
function sameRecordAt(a: Int32Array, base: Int32Array, j: number): boolean {
  return a.buffer === base.buffer && a.byteOffset === base.byteOffset + 4 * j;
}

/** Base class of a single Voronoi cell (C voronoicell_base). The neighbour hooks it calls
 * through its vc argument are the n_* members of voronoicell_neighbor. */
export class voronoicell_base {
  current_vertices: number;
  current_vertex_order: number;
  current_delete_size: number;
  current_delete2_size: number;
  p: number;
  up: number;
  ed: Int32Array[];
  nu: number[];
  pts: Float64Array;

  mem: number[];
  mec: number[];
  mep: Int32Array[];

  ds: number[];
  stacke: number;
  ds2: number[];
  stacke2: number;

  current_marginal: number;
  n_marg: number;
  marg: number[];

  px: number;
  py: number;
  pz: number;
  prsq: number;

  /** Scratch output of m_test(): the double that the C code returns through its reference argument. */
  m_test_ans: number;

  constructor() {
    this.current_vertices = init_vertices;
    this.current_vertex_order = init_vertex_order;
    this.current_delete_size = init_delete_size;
    this.current_delete2_size = init_delete2_size;
    this.p = 0;
    this.up = 0;
    this.ed = emptyRecords(this.current_vertices);
    this.nu = zeros(this.current_vertices);
    this.pts = new Float64Array(3 * this.current_vertices);
    this.mem = zeros(this.current_vertex_order);
    this.mec = zeros(this.current_vertex_order);
    this.mep = emptyRecords(this.current_vertex_order);
    this.ds = zeros(this.current_delete_size);
    this.stacke = this.current_delete_size;
    this.ds2 = zeros(this.current_delete2_size);
    // As in the C constructor, stacke2 is initialised from current_delete_size.
    this.stacke2 = this.current_delete_size;
    this.current_marginal = init_marginal;
    this.marg = zeros(this.current_marginal);
    this.n_marg = 0;
    this.px = 0;
    this.py = 0;
    this.pz = 0;
    this.prsq = 0;
    this.m_test_ans = 0;

    let i: number;
    for (i = 0; i < 3; i++) {
      this.mem[i] = init_n_vertices;
      this.mec[i] = 0;
      this.mep[i] = new Int32Array(init_n_vertices * ((i << 1) + 1));
    }
    this.mem[3] = init_3_vertices;
    this.mec[3] = 0;
    this.mep[3] = new Int32Array(init_3_vertices * 7);
    for (i = 4; i < this.current_vertex_order; i++) {
      this.mem[i] = init_n_vertices;
      this.mec[i] = 0;
      this.mep[i] = new Int32Array(init_n_vertices * ((i << 1) + 1));
    }
  }

  /** Takes the next record of order `order` from mep[order] (the C idiom
   * mep[order]+((order<<1)+1)*mec[order]++). */
  mep_new_record(order: number): Int32Array {
    const off = ((order << 1) + 1) * this.mec[order];
    this.mec[order]++;
    return this.mep[order].subarray(off);
  }

  /** Gives back the last record of order `order` (the C idiom
   * mep[order]+((order<<1)+1)*--mec[order]). */
  mep_pop_record(order: number): Int32Array {
    this.mec[order]--;
    return this.mep[order].subarray(((order << 1) + 1) * this.mec[order]);
  }

  /** Ensures that enough memory is allocated prior to carrying out a copy. */
  check_memory_for_copy(vc: voronoicell_neighbor, vb: voronoicell_base): void {
    while (this.current_vertex_order < vb.current_vertex_order) this.add_memory_vorder(vc);
    for (let i = 0; i < this.current_vertex_order; i++) while (this.mem[i] < vb.mec[i]) this.add_memory(vc, i, 0);
    while (this.current_vertices < vb.p) this.add_memory_vertices(vc);
  }

  /** Increases the memory storage for a particular vertex order (the mep[i] array). */
  add_memory(vc: voronoicell_neighbor, i: number, stackp2: number): void {
    const s = (i << 1) + 1;
    if (this.mem[i] === 0) {
      vc.n_allocate(i, init_n_vertices);
      this.mep[i] = new Int32Array(init_n_vertices * s);
      this.mem[i] = init_n_vertices;
    } else {
      let j = 0;
      let k: number;
      this.mem[i] <<= 1;
      if (this.mem[i] > max_n_vertices) voro_fatal_error('Point memory allocation exceeded absolute maximum', VOROPP_MEMORY_ERROR);
      const l = new Int32Array(s * this.mem[i]);
      let m = 0;
      vc.n_allocate_aux1(i);
      while (j < s * this.mec[i]) {
        k = this.mep[i][j + (i << 1)];
        if (k >= 0) {
          this.ed[k] = l.subarray(j);
          vc.n_set_to_aux1_offset(k, m);
        } else {
          let dsp = 0;
          for (dsp = 0; dsp < stackp2; dsp++) {
            if (sameRecordAt(this.ed[this.ds2[dsp]], this.mep[i], j)) {
              this.ed[this.ds2[dsp]] = l.subarray(j);
              vc.n_set_to_aux1_offset(this.ds2[dsp], m);
              break;
            }
          }
          if (dsp === stackp2) voro_fatal_error('Couldn\'t relocate dangling pointer', VOROPP_INTERNAL_ERROR);
        }
        for (k = 0; k < s; k++, j++) l[j] = this.mep[i][j];
        for (k = 0; k < i; k++, m++) vc.n_copy_to_aux1(i, m);
      }
      this.mep[i] = l;
      vc.n_switch_to_aux1(i);
    }
  }

  /** Doubles the maximum number of vertices allowed (ed, nu and pts arrays).
   * In this build the C routine is reached only at the cell limit, where the C
   * code prints "nope" and exits with status 3. */
  add_memory_vertices(vc: voronoicell_neighbor): void {
    throw new Error(`nope: ${this.current_vertices}`);

    // Unreachable: kept from the C code (the routine above exits in C).
    const i = this.current_vertices << 1;
    if (i > max_vertices) voro_fatal_error('Vertex memory allocation exceeded absolute maximum', VOROPP_MEMORY_ERROR);
    const pp = emptyRecords(i);
    let j: number;
    for (j = 0; j < this.current_vertices; j++) pp[j] = this.ed[j];
    this.ed = pp;
    vc.n_add_memory_vertices(i);
    const pnu = zeros(i);
    for (j = 0; j < this.current_vertices; j++) pnu[j] = this.nu[j];
    this.nu = pnu;
    const ppts = new Float64Array(3 * i);
    for (j = 0; j < 3 * this.current_vertices; j++) ppts[j] = this.pts[j];
    this.pts = ppts;
    this.current_vertices = i;
  }

  /** Doubles the maximum allowed vertex order, by reallocating mem, mep and mec. */
  add_memory_vorder(vc: voronoicell_neighbor): void {
    const i = this.current_vertex_order << 1;
    let j: number;
    if (i > max_vertex_order) voro_fatal_error('Vertex order memory allocation exceeded absolute maximum', VOROPP_MEMORY_ERROR);
    const p1 = zeros(i);
    for (j = 0; j < this.current_vertex_order; j++) p1[j] = this.mem[j];
    while (j < i) p1[j++] = 0;
    this.mem = p1;
    const p2 = emptyRecords(i);
    for (j = 0; j < this.current_vertex_order; j++) p2[j] = this.mep[j];
    this.mep = p2;
    const p3 = zeros(i);
    for (j = 0; j < this.current_vertex_order; j++) p3[j] = this.mec[j];
    while (j < i) p3[j++] = 0;
    this.mec = p3;
    vc.n_add_memory_vorder(i);
    this.current_vertex_order = i;
  }

  /** Doubles the size allocation of the main delete stack. The stack pointer is an
   * index, so it is unchanged by the copy. */
  add_memory_ds(stackp: number): void {
    this.current_delete_size <<= 1;
    if (this.current_delete_size > max_delete_size) voro_fatal_error('Delete stack 1 memory allocation exceeded absolute maximum', VOROPP_MEMORY_ERROR);
    const dsn = zeros(this.current_delete_size);
    let dsnp = 0;
    let dsp = 0;
    while (dsp < stackp) dsn[dsnp++] = this.ds[dsp++];
    this.ds = dsn;
    this.stacke = this.current_delete_size;
  }

  /** Doubles the size allocation of the auxiliary delete stack. The stack pointer is an
   * index, so it is unchanged by the copy. */
  add_memory_ds2(stackp2: number): void {
    this.current_delete2_size <<= 1;
    if (this.current_delete2_size > max_delete2_size) voro_fatal_error('Delete stack 2 memory allocation exceeded absolute maximum', VOROPP_MEMORY_ERROR);
    const dsn = zeros(this.current_delete2_size);
    let dsnp = 0;
    let dsp = 0;
    while (dsp < stackp2) dsn[dsnp++] = this.ds2[dsp++];
    this.ds2 = dsn;
    this.stacke2 = this.current_delete2_size;
  }

  /** Initializes a Voronoi cell as a rectangular box with the given dimensions. */
  init_base(xmin: number, xmax: number, ymin: number, ymax: number, zmin: number, zmax: number): void {
    for (let i = 0; i < this.current_vertex_order; i++) this.mec[i] = 0;
    this.up = 0;
    this.mec[3] = 8;
    this.p = 8;
    xmin *= 2;
    xmax *= 2;
    ymin *= 2;
    ymax *= 2;
    zmin *= 2;
    zmax *= 2;
    const pts = this.pts;
    pts[0] = xmin;
    pts[1] = ymin;
    pts[2] = zmin;
    pts[3] = xmax;
    pts[4] = ymin;
    pts[5] = zmin;
    pts[6] = xmin;
    pts[7] = ymax;
    pts[8] = zmin;
    pts[9] = xmax;
    pts[10] = ymax;
    pts[11] = zmin;
    pts[12] = xmin;
    pts[13] = ymin;
    pts[14] = zmax;
    pts[15] = xmax;
    pts[16] = ymin;
    pts[17] = zmax;
    pts[18] = xmin;
    pts[19] = ymax;
    pts[20] = zmax;
    pts[21] = xmax;
    pts[22] = ymax;
    pts[23] = zmax;
    const q = this.mep[3];
    q[0] = 1; q[1] = 4; q[2] = 2; q[3] = 2; q[4] = 1; q[5] = 0; q[6] = 0;
    q[7] = 3; q[8] = 5; q[9] = 0; q[10] = 2; q[11] = 1; q[12] = 0; q[13] = 1;
    q[14] = 0; q[15] = 6; q[16] = 3; q[17] = 2; q[18] = 1; q[19] = 0; q[20] = 2;
    q[21] = 2; q[22] = 7; q[23] = 1; q[24] = 2; q[25] = 1; q[26] = 0; q[27] = 3;
    q[28] = 6; q[29] = 0; q[30] = 5; q[31] = 2; q[32] = 1; q[33] = 0; q[34] = 4;
    q[35] = 4; q[36] = 1; q[37] = 7; q[38] = 2; q[39] = 1; q[40] = 0; q[41] = 5;
    q[42] = 7; q[43] = 2; q[44] = 4; q[45] = 2; q[46] = 1; q[47] = 0; q[48] = 6;
    q[49] = 5; q[50] = 3; q[51] = 6; q[52] = 2; q[53] = 1; q[54] = 0; q[55] = 7;
    this.ed[0] = q;
    this.ed[1] = q.subarray(7);
    this.ed[2] = q.subarray(14);
    this.ed[3] = q.subarray(21);
    this.ed[4] = q.subarray(28);
    this.ed[5] = q.subarray(35);
    this.ed[6] = q.subarray(42);
    this.ed[7] = q.subarray(49);
    this.nu[0] = 3;
    this.nu[1] = 3;
    this.nu[2] = 3;
    this.nu[3] = 3;
    this.nu[4] = 3;
    this.nu[5] = 3;
    this.nu[6] = 3;
    this.nu[7] = 3;
  }

  /** Starting from a point within the current cutting plane, attempts to find an edge
   * to a point outside the cutting plane. The C routine works on the reference `up`,
   * which is always the member up here. */
  search_for_outside_edge(vc: voronoicell_neighbor): boolean {
    let i: number;
    let lp: number;
    let lw: number;
    let j = 0;
    let stackp2 = 0;
    this.ds2[stackp2++] = this.up;
    while (j < stackp2) {
      this.up = this.ds2[j++];
      for (i = 0; i < this.nu[this.up]; i++) {
        lp = this.ed[this.up][i];
        lw = this.m_test(lp);
        if (lw === -1) return true;
        else if (lw === 0) stackp2 = this.add_to_stack(vc, lp, stackp2);
      }
    }
    return false;
  }

  /** Adds a point to the auxiliary delete stack if it is not already there.
   * Returns the updated stack pointer. */
  add_to_stack(vc: voronoicell_neighbor, lp: number, stackp2: number): number {
    void vc;
    for (let k = 0; k < stackp2; k++) if (this.ds2[k] === lp) return stackp2;
    if (stackp2 === this.stacke2) this.add_memory_ds2(stackp2);
    this.ds2[stackp2++] = lp;
    return stackp2;
  }

  /** Cuts the Voronoi cell by a particle at a separation of (x,y,z) from the cell
   * centre (the C template nplane(vc,...)). The public entry points are on
   * voronoicell_neighbor. */
  nplane_impl(vc: voronoicell_neighbor, x: number, y: number, z: number, rsq: number, p_id: number): boolean {
    let count = 0;
    let i = 0;
    let j = 0;
    let k = 0;
    let lp = this.up;
    let cp = 0;
    let qp = 0;
    let rp = 0;
    let stackp = 0;
    let stackp2 = 0;
    let dsp = 0;
    let us = 0;
    let ls = 0;
    let qs = 0;
    let iqs = 0;
    let cs = 0;
    let uw = 0;
    let qw = 0;
    let lw = 0;
    let edp: Int32Array = EMPTY_INTS;
    let edd: Int32Array = EMPTY_INTS;
    let u = 0;
    let l = 0;
    let r = 0;
    let q = 0;
    let complicated_setup = false;
    let new_double_edge = false;
    let double_edge = false;

    // Initialize the safe testing routine
    this.n_marg = 0;
    this.px = x;
    this.py = y;
    this.pz = z;
    this.prsq = rsq;

    // Test approximately sqrt(n)/4 points for their proximity to the plane
    // and keep the one which is closest
    uw = this.m_test(this.up);
    u = this.m_test_ans;

    // Starting from an initial guess, we now move from vertex to vertex,
    // to try and find an edge which intersects the cutting plane,
    // or a vertex which is on the plane
    try {
      if (uw === 1) {
        // The test point is inside the cutting plane.
        us = 0;
        do {
          lp = this.ed[this.up][us];
          lw = this.m_test(lp);
          l = this.m_test_ans;
          if (l < u) break;
          us++;
        } while (us < this.nu[this.up]);

        if (us === this.nu[this.up]) {
          return false;
        }

        ls = this.ed[this.up][this.nu[this.up] + us];
        while (lw === 1) {
          if (++count >= this.p) throw true;
          u = l;
          this.up = lp;
          for (us = 0; us < ls; us++) {
            lp = this.ed[this.up][us];
            lw = this.m_test(lp);
            l = this.m_test_ans;
            if (l < u) break;
          }
          if (us === ls) {
            us++;
            while (us < this.nu[this.up]) {
              lp = this.ed[this.up][us];
              lw = this.m_test(lp);
              l = this.m_test_ans;
              if (l < u) break;
              us++;
            }
            if (us === this.nu[this.up]) {
              return false;
            }
          }
          ls = this.ed[this.up][this.nu[this.up] + us];
        }

        // If the last point in the iteration is within the
        // plane, we need to do the complicated setup
        // routine. Otherwise, we use the regular iteration.
        if (lw === 0) {
          this.up = lp;
          complicated_setup = true;
        } else complicated_setup = false;
      } else if (uw === -1) {
        us = 0;
        do {
          qp = this.ed[this.up][us];
          qw = this.m_test(qp);
          q = this.m_test_ans;
          if (u < q) break;
          us++;
        } while (us < this.nu[this.up]);
        if (us === this.nu[this.up]) return true;

        while (qw === -1) {
          qs = this.ed[this.up][this.nu[this.up] + us];
          if (++count >= this.p) throw true;
          u = q;
          this.up = qp;
          for (us = 0; us < qs; us++) {
            qp = this.ed[this.up][us];
            qw = this.m_test(qp);
            q = this.m_test_ans;
            if (u < q) break;
          }
          if (us === qs) {
            us++;
            while (us < this.nu[this.up]) {
              qp = this.ed[this.up][us];
              qw = this.m_test(qp);
              q = this.m_test_ans;
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
          us = this.ed[lp][this.nu[lp] + ls];
          u = q;
          complicated_setup = false;
        } else {
          this.up = qp;
          complicated_setup = true;
        }
      } else {
        // Our original test point was on the plane, so we
        // automatically head for the complicated setup
        // routine
        complicated_setup = true;
      }
    } catch (except: unknown) {
      // This routine is a fall-back, in case floating point errors
      // cause the usual search routine to fail. In the fall-back
      // routine, we just test every edge to find one straddling
      // the plane.
      if (except !== true) throw except;
      qw = 1;
      lw = 0;
      for (qp = 0; qp < this.p; qp++) {
        qw = this.m_test(qp);
        q = this.m_test_ans;
        if (qw === 1) {
          // The point is inside the cutting space. Now
          // see if we can find a neighbor which isn't.
          for (us = 0; us < this.nu[qp]; us++) {
            lp = this.ed[qp][us];
            if (lp < qp) {
              lw = this.m_test(lp);
              l = this.m_test_ans;
              if (lw !== 1) break;
            }
          }
          if (us < this.nu[qp]) {
            this.up = qp;
            if (lw === 0) {
              complicated_setup = true;
            } else {
              complicated_setup = false;
              u = q;
              ls = this.ed[this.up][this.nu[this.up] + us];
            }
            break;
          }
        } else if (qw === -1) {
          // The point is outside the cutting space. See
          // if we can find a neighbor which isn't.
          for (ls = 0; ls < this.nu[qp]; ls++) {
            this.up = this.ed[qp][ls];
            if (this.up < qp) {
              uw = this.m_test(this.up);
              u = this.m_test_ans;
              if (uw !== -1) break;
            }
          }
          if (ls < this.nu[qp]) {
            if (uw === 0) {
              this.up = qp;
              complicated_setup = true;
            } else {
              complicated_setup = false;
              lp = qp;
              l = q;
              us = this.ed[lp][this.nu[lp] + ls];
            }
            break;
          }
        } else {
          // The point is in the plane, so we just
          // proceed with the complicated setup routine
          this.up = qp;
          complicated_setup = true;
          break;
        }
      }
      if (qp === this.p) return qw === -1;
    }

    // We're about to add the first point of the new facet. In either
    // routine, we have to add a point, so first check there's space for
    // it.
    if (this.p === this.current_vertices) this.add_memory_vertices(vc);

    if (complicated_setup) {
      // We want to be strict about reaching the conclusion that the
      // cell is entirely within the cutting plane. It's not enough
      // to find a vertex that has edges which are all inside or on
      // the plane. If the vertex has neighbors that are also on the
      // plane, we should check those too.
      if (!this.search_for_outside_edge(vc)) return false;

      // The search algorithm found a point which is on the cutting
      // plane. We leave that point in place, and create a new one at
      // the same location.
      this.pts[3 * this.p] = this.pts[3 * this.up];
      this.pts[3 * this.p + 1] = this.pts[3 * this.up + 1];
      this.pts[3 * this.p + 2] = this.pts[3 * this.up + 2];

      // Search for a collection of edges of the test vertex which
      // are outside of the cutting space. Begin by testing the
      // zeroth edge.
      i = 0;
      lp = this.ed[this.up][0];
      lw = this.m_test(lp);
      l = this.m_test_ans;
      if (lw !== -1) {
        // The first edge is either inside the cutting space,
        // or lies within the cutting plane. Test the edges
        // sequentially until we find one that is outside.
        rp = lw;
        do {
          i++;

          // If we reached the last edge with no luck
          // then all of the vertices are inside
          // or on the plane, so the cell is completely
          // deleted
          if (i === this.nu[this.up]) return false;
          lp = this.ed[this.up][i];
          lw = this.m_test(lp);
          l = this.m_test_ans;
        } while (lw !== -1);
        j = i + 1;

        // We found an edge outside the cutting space. Keep
        // moving through these edges until we find one that's
        // inside or on the plane.
        while (j < this.nu[this.up]) {
          lp = this.ed[this.up][j];
          lw = this.m_test(lp);
          l = this.m_test_ans;
          if (lw !== -1) break;
          j++;
        }

        // Compute the number of edges for the new vertex. In
        // general it will be the number of outside edges
        // found, plus two. But we need to recognize the
        // special case when all but one edge is outside, and
        // the remaining one is on the plane. For that case we
        // have to reduce the edge count by one to prevent
        // doubling up.
        if (j === this.nu[this.up] && i === 1 && rp === 0) {
          this.nu[this.p] = this.nu[this.up];
          double_edge = true;
        } else this.nu[this.p] = j - i + 2;
        k = 1;

        // Add memory for the new vertex if needed, and
        // initialize
        while (this.nu[this.p] >= this.current_vertex_order) this.add_memory_vorder(vc);
        if (this.mec[this.nu[this.p]] === this.mem[this.nu[this.p]]) this.add_memory(vc, this.nu[this.p], stackp2);
        vc.n_set_pointer(this.p, this.nu[this.p]);
        this.ed[this.p] = this.mep_new_record(this.nu[this.p]);
        this.ed[this.p][this.nu[this.p] << 1] = this.p;

        // Copy the edges of the original vertex into the new
        // one. Delete the edges of the original vertex, and
        // update the relational table.
        us = this.cycle_down(i, this.up);
        while (i < j) {
          qp = this.ed[this.up][i];
          qs = this.ed[this.up][this.nu[this.up] + i];
          vc.n_copy(this.p, k, this.up, i);
          this.ed[this.p][k] = qp;
          this.ed[this.p][this.nu[this.p] + k] = qs;
          this.ed[qp][qs] = this.p;
          this.ed[qp][this.nu[qp] + qs] = k;
          this.ed[this.up][i] = -1;
          i++;
          k++;
        }
        qs = i === this.nu[this.up] ? 0 : i;
      } else {
        // In this case, the zeroth edge is outside the cutting
        // plane. Begin by searching backwards from the last
        // edge until we find an edge which isn't outside.
        i = this.nu[this.up] - 1;
        lp = this.ed[this.up][i];
        lw = this.m_test(lp);
        l = this.m_test_ans;
        while (lw === -1) {
          i--;

          // If i reaches zero, then we have a point in
          // the plane all of whose edges are outside
          // the cutting space, so we just exit
          if (i === 0) return true;
          lp = this.ed[this.up][i];
          lw = this.m_test(lp);
          l = this.m_test_ans;
        }

        // Now search forwards from zero
        j = 1;
        qp = this.ed[this.up][j];
        qw = this.m_test(qp);
        // Note: the C code stores this second test's output in l, not q (kept as is).
        l = this.m_test_ans;
        while (qw === -1) {
          j++;
          qp = this.ed[this.up][j];
          qw = this.m_test(qp);
          l = this.m_test_ans;
        }

        // Compute the number of edges for the new vertex. In
        // general it will be the number of outside edges
        // found, plus two. But we need to recognize the
        // special case when all but one edge is outside, and
        // the remaining one is on the plane. For that case we
        // have to reduce the edge count by one to prevent
        // doubling up.
        if (i === j && qw === 0) {
          double_edge = true;
          this.nu[this.p] = this.nu[this.up];
        } else {
          this.nu[this.p] = this.nu[this.up] - i + j + 1;
        }

        // Add memory to store the vertex if it doesn't exist
        // already
        k = 1;
        while (this.nu[this.p] >= this.current_vertex_order) this.add_memory_vorder(vc);
        if (this.mec[this.nu[this.p]] === this.mem[this.nu[this.p]]) this.add_memory(vc, this.nu[this.p], stackp2);

        // Copy the edges of the original vertex into the new
        // one. Delete the edges of the original vertex, and
        // update the relational table.
        vc.n_set_pointer(this.p, this.nu[this.p]);
        this.ed[this.p] = this.mep_new_record(this.nu[this.p]);
        this.ed[this.p][this.nu[this.p] << 1] = this.p;
        us = i++;
        while (i < this.nu[this.up]) {
          qp = this.ed[this.up][i];
          qs = this.ed[this.up][this.nu[this.up] + i];
          vc.n_copy(this.p, k, this.up, i);
          this.ed[this.p][k] = qp;
          this.ed[this.p][this.nu[this.p] + k] = qs;
          this.ed[qp][qs] = this.p;
          this.ed[qp][this.nu[qp] + qs] = k;
          this.ed[this.up][i] = -1;
          i++;
          k++;
        }
        i = 0;
        while (i < j) {
          qp = this.ed[this.up][i];
          qs = this.ed[this.up][this.nu[this.up] + i];
          vc.n_copy(this.p, k, this.up, i);
          this.ed[this.p][k] = qp;
          this.ed[this.p][this.nu[this.p] + k] = qs;
          this.ed[qp][qs] = this.p;
          this.ed[qp][this.nu[qp] + qs] = k;
          this.ed[this.up][i] = -1;
          i++;
          k++;
        }
        qs = j;
      }
      if (!double_edge) {
        vc.n_copy(this.p, k, this.up, qs);
        vc.n_set(this.p, 0, p_id);
      } else vc.n_copy(this.p, 0, this.up, qs);

      // Add this point to the auxiliary delete stack
      if (stackp2 === this.stacke2) this.add_memory_ds2(stackp2);
      this.ds2[stackp2++] = this.up;

      // Look at the edges on either side of the group that was
      // detected. We're going to commence facet computation by
      // moving along one of them. We are going to end up coming back
      // along the other one.
      cs = k;
      qp = this.up;
      q = u;
      i = this.ed[this.up][us];
      us = this.ed[this.up][this.nu[this.up] + us];
      this.up = i;
      this.ed[qp][this.nu[qp] << 1] = -this.p;
    } else {
      // The search algorithm found an intersected edge between the
      // points lp and up. Create a new vertex between them which
      // lies on the cutting plane. Since u and l differ by at least
      // the tolerance, this division should never screw up.
      if (stackp === this.stacke) this.add_memory_ds(stackp);
      this.ds[stackp++] = this.up;
      r = u / (u - l);
      l = 1 - r;
      this.pts[3 * this.p] = this.pts[3 * lp] * r + this.pts[3 * this.up] * l;
      this.pts[3 * this.p + 1] = this.pts[3 * lp + 1] * r + this.pts[3 * this.up + 1] * l;
      this.pts[3 * this.p + 2] = this.pts[3 * lp + 2] * r + this.pts[3 * this.up + 2] * l;

      // This point will always have three edges. Connect one of them
      // to lp.
      this.nu[this.p] = 3;
      if (this.mec[3] === this.mem[3]) this.add_memory(vc, 3, stackp2);
      vc.n_set_pointer(this.p, 3);
      vc.n_set(this.p, 0, p_id);
      vc.n_copy(this.p, 1, this.up, us);
      vc.n_copy(this.p, 2, lp, ls);
      this.ed[this.p] = this.mep_new_record(3);
      this.ed[this.p][6] = this.p;
      this.ed[this.up][us] = -1;
      this.ed[lp][ls] = this.p;
      this.ed[lp][this.nu[lp] + ls] = 1;
      this.ed[this.p][1] = lp;
      this.ed[this.p][this.nu[this.p] + 1] = ls;
      cs = 2;

      // Set the direction to move in
      qs = this.cycle_up(us, this.up);
      qp = this.up;
      q = u;
    }

    // When the code reaches here, we have initialized the first point, and
    // we have a direction for moving it to construct the rest of the facet
    cp = this.p;
    rp = this.p;
    this.p++;
    while (qp !== this.up || qs !== us) {
      // We're currently tracing round an intersected facet. Keep
      // moving around it until we find a point or edge which
      // intersects the plane.
      lp = this.ed[qp][qs];
      lw = this.m_test(lp);
      l = this.m_test_ans;

      if (lw === 1) {
        // The point is still in the cutting space. Just add it
        // to the delete stack and keep moving.
        qs = this.cycle_up(this.ed[qp][this.nu[qp] + qs], lp);
        qp = lp;
        q = l;
        if (stackp === this.stacke) this.add_memory_ds(stackp);
        this.ds[stackp++] = qp;
      } else if (lw === -1) {
        // The point is outside of the cutting space, so we've
        // found an intersected edge. Introduce a regular point
        // at the point of intersection. Connect it to the
        // point we just tested. Also connect it to the previous
        // new point in the facet we're constructing.
        if (this.p === this.current_vertices) this.add_memory_vertices(vc);
        r = q / (q - l);
        l = 1 - r;
        this.pts[3 * this.p] = this.pts[3 * lp] * r + this.pts[3 * qp] * l;
        this.pts[3 * this.p + 1] = this.pts[3 * lp + 1] * r + this.pts[3 * qp + 1] * l;
        this.pts[3 * this.p + 2] = this.pts[3 * lp + 2] * r + this.pts[3 * qp + 2] * l;
        this.nu[this.p] = 3;
        if (this.mec[3] === this.mem[3]) this.add_memory(vc, 3, stackp2);
        ls = this.ed[qp][qs + this.nu[qp]];
        vc.n_set_pointer(this.p, 3);
        vc.n_set(this.p, 0, p_id);
        vc.n_copy(this.p, 1, qp, qs);
        vc.n_copy(this.p, 2, lp, ls);
        this.ed[this.p] = this.mep_new_record(3);
        this.ed[this.p][0] = cp;
        this.ed[this.p][1] = lp;
        this.ed[this.p][3] = cs;
        this.ed[this.p][4] = ls;
        this.ed[this.p][6] = this.p;
        this.ed[lp][ls] = this.p;
        this.ed[lp][this.nu[lp] + ls] = 1;
        this.ed[cp][cs] = this.p;
        this.ed[cp][this.nu[cp] + cs] = 0;
        this.ed[qp][qs] = -1;
        qs = this.cycle_up(qs, qp);
        cp = this.p++;
        cs = 2;
      } else {
        // We've found a point which is on the cutting plane.
        // We're going to introduce a new point right here, but
        // first we need to figure out the number of edges it
        // has.
        if (this.p === this.current_vertices) this.add_memory_vertices(vc);

        // If the previous vertex detected a double edge, our
        // new vertex will have one less edge.
        k = double_edge ? 0 : 1;
        qs = this.ed[qp][this.nu[qp] + qs];
        qp = lp;
        iqs = qs;

        // Start testing the edges of the current point until
        // we find one which isn't outside the cutting space
        do {
          k++;
          qs = this.cycle_up(qs, qp);
          lp = this.ed[qp][qs];
          lw = this.m_test(lp);
          l = this.m_test_ans;
        } while (lw === -1);

        // Now we need to find out whether this marginal vertex
        // we are on has been visited before, because if that's
        // the case, we need to add vertices to the existing
        // new vertex, rather than creating a fresh one. We also
        // need to figure out whether we're in a case where we
        // might be creating a duplicate edge.
        j = -this.ed[qp][this.nu[qp] << 1];
        if (qp === this.up && qs === us) {
          // If we're heading into the final part of the
          // new facet, then we never worry about the
          // duplicate edge calculation.
          new_double_edge = false;
          if (j > 0) k += this.nu[j];
        } else {
          if (j > 0) {
            // This vertex was visited before, so
            // count those vertices to the ones we
            // already have.
            k += this.nu[j];

            // The only time when we might make a
            // duplicate edge is if the point we're
            // going to move to next is also a
            // marginal point, so test for that
            // first.
            if (lw === 0) {
              // Now see whether this marginal point
              // has been visited before.
              i = -this.ed[lp][this.nu[lp] << 1];
              if (i > 0) {
                // Now see if the last edge of that other
                // marginal point actually ends up here.
                if (this.ed[i][this.nu[i] - 1] === j) {
                  new_double_edge = true;
                  k -= 1;
                } else new_double_edge = false;
              } else {
                // That marginal point hasn't been visited
                // before, so we probably don't have to worry
                // about duplicate edges, except in the
                // case when that's the way into the end
                // of the facet, because that way always creates
                // an edge.
                if (j === rp && lp === this.up && this.ed[qp][this.nu[qp] + qs] === us) {
                  new_double_edge = true;
                  k -= 1;
                } else new_double_edge = false;
              }
            } else new_double_edge = false;
          } else {
            // The vertex hasn't been visited
            // before, but let's see if it's
            // marginal
            if (lw === 0) {
              // If it is, we need to check
              // for the case that it's a
              // small branch, and that we're
              // heading right back to where
              // we came from
              i = -this.ed[lp][this.nu[lp] << 1];
              if (i === cp) {
                new_double_edge = true;
                k -= 1;
              } else new_double_edge = false;
            } else new_double_edge = false;
          }
        }

        // k now holds the number of edges of the new vertex
        // we are forming. Add memory for it if it doesn't exist
        // already.
        while (k >= this.current_vertex_order) this.add_memory_vorder(vc);
        if (this.mec[k] === this.mem[k]) this.add_memory(vc, k, stackp2);

        // Now create a new vertex with order k, or augment
        // the existing one
        if (j > 0) {
          // If we're augmenting a vertex but we don't
          // actually need any more edges, just skip this
          // routine to avoid memory confusion
          if (this.nu[j] !== k) {
            // Allocate memory and copy the edges
            // of the previous instance into it
            vc.n_set_aux1(k);
            edp = this.mep_new_record(k);
            i = 0;
            while (i < this.nu[j]) {
              vc.n_copy_aux1(j, i);
              edp[i] = this.ed[j][i];
              edp[k + i] = this.ed[j][this.nu[j] + i];
              i++;
            }
            edp[k << 1] = j;

            // Remove the previous instance with
            // fewer vertices from the memory
            // structure
            edd = this.mep_pop_record(this.nu[j]);
            if (!sameRecord(edd, this.ed[j])) {
              for (lw = 0; lw <= (this.nu[j] << 1); lw++) this.ed[j][lw] = edd[lw];
              vc.n_set_aux2_copy(j, this.nu[j]);
              vc.n_copy_pointer(edd[this.nu[j] << 1], j);
              this.ed[edd[this.nu[j] << 1]] = this.ed[j];
            }
            vc.n_set_to_aux1(j);
            this.ed[j] = edp;
          } else i = this.nu[j];
        } else {
          // Allocate a new vertex of order k
          vc.n_set_pointer(this.p, k);
          this.ed[this.p] = this.mep_new_record(k);
          this.ed[this.p][k << 1] = this.p;
          if (stackp2 === this.stacke2) this.add_memory_ds2(stackp2);
          this.ds2[stackp2++] = qp;
          this.pts[3 * this.p] = this.pts[3 * qp];
          this.pts[3 * this.p + 1] = this.pts[3 * qp + 1];
          this.pts[3 * this.p + 2] = this.pts[3 * qp + 2];
          this.ed[qp][this.nu[qp] << 1] = -this.p;
          j = this.p++;
          i = 0;
        }
        this.nu[j] = k;

        // Unless the previous case was a double edge, connect
        // the first available edge of the new vertex to the
        // last one in the facet
        if (!double_edge) {
          this.ed[j][i] = cp;
          this.ed[j][this.nu[j] + i] = cs;
          vc.n_set(j, i, p_id);
          this.ed[cp][cs] = j;
          this.ed[cp][this.nu[cp] + cs] = i;
          i++;
        }

        // Copy in the edges of the underlying vertex,
        // and do one less if this was a double edge
        qs = iqs;
        while (i < (new_double_edge ? k : k - 1)) {
          qs = this.cycle_up(qs, qp);
          lp = this.ed[qp][qs];
          ls = this.ed[qp][this.nu[qp] + qs];
          vc.n_copy(j, i, qp, qs);
          this.ed[j][i] = lp;
          this.ed[j][this.nu[j] + i] = ls;
          this.ed[lp][ls] = j;
          this.ed[lp][this.nu[lp] + ls] = i;
          this.ed[qp][qs] = -1;
          i++;
        }
        qs = this.cycle_up(qs, qp);
        cs = i;
        cp = j;
        vc.n_copy(j, new_double_edge ? 0 : cs, qp, qs);

        // Update the double_edge flag, to pass it
        // to the next instance of this routine
        double_edge = new_double_edge;
      }
    }

    // Connect the final created vertex to the initial one
    this.ed[cp][cs] = rp;
    this.ed[rp][0] = cp;
    this.ed[cp][this.nu[cp] + cs] = 0;
    this.ed[rp][this.nu[rp]] = cs;

    // Delete points: first, remove any duplicates
    dsp = 0;
    while (dsp < stackp) {
      j = this.ds[dsp];
      if (this.ed[j][this.nu[j]] !== -1) {
        this.ed[j][this.nu[j]] = -1;
        dsp++;
      } else {
        stackp--;
        this.ds[dsp] = this.ds[stackp];
      }
    }

    // Add the points in the auxiliary delete stack,
    // and reset their back pointers
    for (dsp = 0; dsp < stackp2; dsp++) {
      j = this.ds2[dsp];
      this.ed[j][this.nu[j] << 1] = j;
      if (this.ed[j][this.nu[j]] !== -1) {
        this.ed[j][this.nu[j]] = -1;
        if (stackp === this.stacke) this.add_memory_ds(stackp);
        this.ds[stackp++] = j;
      }
    }

    // Scan connections and add in extras
    for (dsp = 0; dsp < stackp; dsp++) {
      cp = this.ds[dsp];
      for (let e = 0; e < this.nu[cp]; e++) {
        qp = this.ed[cp][e];
        if (qp !== -1 && this.ed[qp][this.nu[qp]] !== -1) {
          // (The C code re-bases dsp after add_memory_ds(); with index stacks dsp is unchanged.)
          if (stackp === this.stacke) this.add_memory_ds(stackp);
          this.ds[stackp++] = qp;
          this.ed[qp][this.nu[qp]] = -1;
        }
      }
    }
    this.up = 0;

    // Delete them from the array structure
    while (stackp > 0) {
      --this.p;
      while (this.ed[this.p][this.nu[this.p]] === -1) {
        j = this.nu[this.p];
        edp = this.ed[this.p];
        edd = this.mep_pop_record(j);
        for (let t = 0; t < (j << 1) + 1; t++) edp[t] = edd[t];
        vc.n_set_aux2_copy(this.p, j);
        vc.n_copy_pointer(this.ed[this.p][j << 1], this.p);
        this.ed[this.ed[this.p][j << 1]] = this.ed[this.p];
        --this.p;
      }
      this.up = this.ds[--stackp];
      if (this.up < this.p) {
        // Vertex management
        this.pts[3 * this.up] = this.pts[3 * this.p];
        this.pts[3 * this.up + 1] = this.pts[3 * this.p + 1];
        this.pts[3 * this.up + 2] = this.pts[3 * this.p + 2];

        // Memory management
        j = this.nu[this.up];
        edp = this.ed[this.up];
        edd = this.mep_pop_record(j);
        for (let t = 0; t < (j << 1) + 1; t++) edp[t] = edd[t];
        vc.n_set_aux2_copy(this.up, j);
        vc.n_copy_pointer(this.ed[this.up][j << 1], this.up);
        vc.n_copy_pointer(this.up, this.p);
        this.ed[this.ed[this.up][j << 1]] = this.ed[this.up];

        // Edge management
        this.ed[this.up] = this.ed[this.p];
        this.nu[this.up] = this.nu[this.p];
        for (i = 0; i < this.nu[this.up]; i++) this.ed[this.ed[this.up][i]][this.ed[this.up][this.nu[this.up] + i]] = this.up;
        this.ed[this.up][this.nu[this.up] << 1] = this.up;
      } else {
        this.up = this.p++;
      }
    }

    // Check for any vertices of zero order
    if (this.mec[0] > 0) voro_fatal_error('Zero order vertex formed', VOROPP_INTERNAL_ERROR);

    // Collapse any order 2 vertices and exit
    return this.collapse_order2(vc);
  }

  /** During the creation of a new facet in the plane routine, some order two vertices
   * may arise. This routine removes them. Returns false if the cell disappears. */
  collapse_order2(vc: voronoicell_neighbor): boolean {
    if (!this.collapse_order1(vc)) return false;
    let a: number;
    let b: number;
    let i: number;
    let j: number;
    let k: number;
    let l: number;
    while (this.mec[2] > 0) {
      // Pick a order 2 vertex and read in its edges
      this.mec[2]--;
      i = this.mec[2];
      j = this.mep[2][5 * i];
      k = this.mep[2][5 * i + 1];
      if (j === k) {
        // (VOROPP_VERBOSE >= 1 message omitted: verbosity is 0.)
        return false;
      }

      // Scan the edges of j to see if joins k
      for (l = 0; l < this.nu[j]; l++) {
        if (this.ed[j][l] === k) break;
      }

      // If j doesn't already join k, join them together.
      // Otherwise delete the connection to the current
      // vertex from j and k.
      a = this.mep[2][5 * i + 2];
      b = this.mep[2][5 * i + 3];
      i = this.mep[2][5 * i + 4];
      if (l === this.nu[j]) {
        this.ed[j][a] = k;
        this.ed[k][b] = j;
        this.ed[j][this.nu[j] + a] = b;
        this.ed[k][this.nu[k] + b] = a;
      } else {
        if (!this.delete_connection(vc, j, a, false)) return false;
        if (!this.delete_connection(vc, k, b, true)) return false;
      }

      // Compact the memory
      --this.p;
      if (this.up === i) this.up = 0;
      if (this.p !== i) {
        if (this.up === this.p) this.up = i;
        this.pts[3 * i] = this.pts[3 * this.p];
        this.pts[3 * i + 1] = this.pts[3 * this.p + 1];
        this.pts[3 * i + 2] = this.pts[3 * this.p + 2];
        for (k = 0; k < this.nu[this.p]; k++) this.ed[this.ed[this.p][k]][this.ed[this.p][this.nu[this.p] + k]] = i;
        vc.n_copy_pointer(i, this.p);
        this.ed[i] = this.ed[this.p];
        this.nu[i] = this.nu[this.p];
        this.ed[i][this.nu[i] << 1] = i;
      }

      // Collapse any order 1 vertices if they were created
      if (!this.collapse_order1(vc)) return false;
    }
    return true;
  }

  /** Order one vertices can potentially be created during the order two collapse
   * routine. This routine keeps removing them until there are none left. */
  collapse_order1(vc: voronoicell_neighbor): boolean {
    let i: number;
    let j: number;
    let k: number;
    while (this.mec[1] > 0) {
      this.up = 0;
      // (VOROPP_VERBOSE >= 1 message omitted: verbosity is 0.)
      this.mec[1]--;
      i = this.mec[1];
      j = this.mep[1][3 * i];
      k = this.mep[1][3 * i + 1];
      i = this.mep[1][3 * i + 2];
      if (!this.delete_connection(vc, j, k, false)) return false;
      --this.p;
      if (this.up === i) this.up = 0;
      if (this.p !== i) {
        if (this.up === this.p) this.up = i;
        this.pts[3 * i] = this.pts[3 * this.p];
        this.pts[3 * i + 1] = this.pts[3 * this.p + 1];
        this.pts[3 * i + 2] = this.pts[3 * this.p + 2];
        for (k = 0; k < this.nu[this.p]; k++) this.ed[this.ed[this.p][k]][this.ed[this.p][this.nu[this.p] + k]] = i;
        vc.n_copy_pointer(i, this.p);
        this.ed[i] = this.ed[this.p];
        this.nu[i] = this.nu[this.p];
        this.ed[i][this.nu[i] << 1] = i;
      }
    }
    return true;
  }

  /** Deletes the kth edge of vertex j and reorganizes the memory. The hand flag
   * decides whether to preserve the plane on the left or right of the connection.
   * Returns false if a zero order vertex was formed. */
  delete_connection(vc: voronoicell_neighbor, j: number, k: number, hand: boolean): boolean {
    const q = hand ? k : this.cycle_up(k, j);
    let i = this.nu[j] - 1;
    let l: number;
    let m: number;
    // (VOROPP_VERBOSE >= 1 zero-order message omitted: verbosity is 0.)
    if (this.mec[i] === this.mem[i]) this.add_memory(vc, i, 0);
    vc.n_set_aux1(i);
    for (l = 0; l < q; l++) vc.n_copy_aux1(j, l);
    while (l < i) {
      vc.n_copy_aux1_shift(j, l);
      l++;
    }
    const edp = this.mep_new_record(i);
    edp[i << 1] = j;
    for (l = 0; l < k; l++) {
      edp[l] = this.ed[j][l];
      edp[l + i] = this.ed[j][l + this.nu[j]];
    }
    while (l < i) {
      m = this.ed[j][l + 1];
      edp[l] = m;
      k = this.ed[j][l + this.nu[j] + 1];
      edp[l + i] = k;
      this.ed[m][this.nu[m] + k]--;
      l++;
    }

    const edd = this.mep_pop_record(this.nu[j]);
    for (l = 0; l <= (this.nu[j] << 1); l++) this.ed[j][l] = edd[l];
    vc.n_set_aux2_copy(j, this.nu[j]);
    vc.n_set_to_aux2(edd[this.nu[j] << 1]);
    vc.n_set_to_aux1(j);
    this.ed[edd[this.nu[j] << 1]] = edd;
    this.ed[j] = edp;
    this.nu[j] = i;
    return true;
  }

  /** Calculates the areas of each face of the Voronoi cell. */
  face_areas(face_areas: number[]): void {
    let area: number;
    face_areas.length = 0;
    let i: number;
    let j: number;
    let k: number;
    let l: number;
    let m: number;
    let n: number;
    let ux: number;
    let uy: number;
    let uz: number;
    let vx: number;
    let vy: number;
    let vz: number;
    let wx: number;
    let wy: number;
    let wz: number;
    for (i = 1; i < this.p; i++) {
      for (j = 0; j < this.nu[i]; j++) {
        k = this.ed[i][j];
        if (k >= 0) {
          area = 0;
          this.ed[i][j] = -1 - k;
          l = this.cycle_up(this.ed[i][this.nu[i] + j], k);
          m = this.ed[k][l];
          this.ed[k][l] = -1 - m;
          while (m !== i) {
            n = this.cycle_up(this.ed[k][this.nu[k] + l], m);
            ux = this.pts[3 * k] - this.pts[3 * i];
            uy = this.pts[3 * k + 1] - this.pts[3 * i + 1];
            uz = this.pts[3 * k + 2] - this.pts[3 * i + 2];
            vx = this.pts[3 * m] - this.pts[3 * i];
            vy = this.pts[3 * m + 1] - this.pts[3 * i + 1];
            vz = this.pts[3 * m + 2] - this.pts[3 * i + 2];
            wx = uy * vz - uz * vy;
            wy = uz * vx - ux * vz;
            wz = ux * vy - uy * vx;
            area += Math.sqrt(wx * wx + wy * wy + wz * wz);
            k = m;
            l = n;
            m = this.ed[k][l];
            this.ed[k][l] = -1 - m;
          }
          face_areas.push(0.125 * area);
        }
      }
    }
    this.reset_edges();
  }

  /** Several routines track their progress by flipping edges to negative. This
   * resets them back to positive, and bails out if a positive edge is found. */
  reset_edges(): void {
    let i: number;
    let j: number;
    for (i = 0; i < this.p; i++) {
      for (j = 0; j < this.nu[i]; j++) {
        if (this.ed[i][j] >= 0) voro_fatal_error('Edge reset routine found a previously untested edge', VOROPP_INTERNAL_ERROR);
        this.ed[i][j] = -1 - this.ed[i][j];
      }
    }
  }

  /** Checks whether vertex n is inside (-1), outside (1) or within (0) the test plane.
   * The double output of the C code is stored in m_test_ans. */
  m_test(n: number): number {
    let pp = 3 * n;
    let ans = this.pts[pp] * this.px;
    pp++;
    ans += this.pts[pp] * this.py;
    pp++;
    ans += this.pts[pp] * this.pz - this.prsq;
    this.m_test_ans = ans;
    if (ans < -tolerance2) {
      return -1;
    } else if (ans > tolerance2) {
      return 1;
    }
    return this.check_marginal(n, ans);
  }

  /** Checks a vertex that is very close to the plane, keeping a table of marginal
   * results so that repeated tests are consistent. */
  check_marginal(n: number, ans: number): number {
    let i: number;
    for (i = 0; i < this.n_marg; i += 2) if (this.marg[i] === n) return this.marg[i + 1];
    if (this.n_marg === this.current_marginal) {
      this.current_marginal <<= 1;
      if (this.current_marginal > max_marginal) {
        voro_fatal_error('Marginal case buffer allocation exceeded absolute maximum', VOROPP_MEMORY_ERROR);
      }
      const pmarg = zeros(this.current_marginal);
      for (let j = 0; j < this.n_marg; j++) pmarg[j] = this.marg[j];
      this.marg = pmarg;
    }
    this.marg[this.n_marg++] = n;
    this.marg[this.n_marg++] = ans > tolerance ? 1 : ans < -tolerance ? -1 : 0;
    return this.marg[this.n_marg - 1];
  }

  /** Counts the faces of the computed Voronoi cell. */
  number_of_faces(): number {
    let i: number;
    let j: number;
    let k: number;
    let l: number;
    let m: number;
    let s = 0;
    for (i = 1; i < this.p; i++) {
      for (j = 0; j < this.nu[i]; j++) {
        k = this.ed[i][j];
        if (k >= 0) {
          s++;
          this.ed[i][j] = -1 - k;
          l = this.cycle_up(this.ed[i][this.nu[i] + j], k);
          do {
            m = this.ed[k][l];
            this.ed[k][l] = -1 - m;
            l = this.cycle_up(this.ed[k][this.nu[k] + l], m);
            k = m;
          } while (k !== i);
        }
      }
    }
    this.reset_edges();
    return s;
  }

  /** Returns the vertex vectors in the global coordinate system (the C
   * vertices(x,y,z,v)). */
  vertices(x: number, y: number, z: number, vertices: number[]): void {
    vertices.length = 3 * this.p;
    for (let i = 0; i < 3 * this.p; i += 3) {
      vertices[i] = x + this.pts[i] * 0.5;
      vertices[i + 1] = y + this.pts[i + 1] * 0.5;
      vertices[i + 2] = z + this.pts[i + 2] * 0.5;
    }
  }

  /** For each face, outputs a bracketed sequence of numbers containing a list of all
   * the vertices that make up that face. */
  face_vertices(face_vertices: number[]): void {
    let i: number;
    let j: number;
    let k: number;
    let l: number;
    let m: number;
    let vp = 0;
    let vn: number;
    face_vertices.length = 0;
    for (i = 1; i < this.p; i++) {
      for (j = 0; j < this.nu[i]; j++) {
        k = this.ed[i][j];
        if (k >= 0) {
          face_vertices.push(0);
          face_vertices.push(i);
          this.ed[i][j] = -1 - k;
          l = this.cycle_up(this.ed[i][this.nu[i] + j], k);
          do {
            face_vertices.push(k);
            m = this.ed[k][l];
            this.ed[k][l] = -1 - m;
            l = this.cycle_up(this.ed[k][this.nu[k] + l], m);
            k = m;
          } while (k !== i);
          vn = face_vertices.length;
          face_vertices[vp] = vn - vp - 1;
          vp = vn;
        }
      }
    }
    this.reset_edges();
  }

  /** Base-class neighbour list: no neighbour information, so the list is empty. */
  neighbors(v: number[]): void {
    v.length = 0;
  }

  cycle_up(a: number, p: number): number {
    return a === this.nu[p] - 1 ? 0 : a + 1;
  }

  cycle_down(a: number, p: number): number {
    return a === 0 ? this.nu[p] - 1 : a - 1;
  }
}

/** Extension of voronoicell_base that tracks the neighbouring particle ID of each face. */
export class voronoicell_neighbor extends voronoicell_base {
  mne: Int32Array[];
  ne: Int32Array[];
  paux1: Int32Array;
  paux2: Int32Array;

  constructor() {
    super();
    this.paux1 = EMPTY_INTS;
    this.paux2 = EMPTY_INTS;
    this.mne = emptyRecords(this.current_vertex_order);
    this.ne = emptyRecords(this.current_vertices);
    let i: number;
    for (i = 0; i < 3; i++) this.mne[i] = new Int32Array(init_n_vertices * i);
    this.mne[3] = new Int32Array(init_3_vertices * 3);
    for (i = 4; i < this.current_vertex_order; i++) this.mne[i] = new Int32Array(init_n_vertices * i);
  }

  /** Cuts the Voronoi cell by a particle whose centre is at (x,y,z) from the cell
   * centre; rsq is x^2+y^2+z^2. Returns false if the cell was deleted entirely. */
  nplane(x: number, y: number, z: number, rsq: number, p_id: number): boolean;
  /** Cuts the cell by the vector (x,y,z), computing the modulus squared first. */
  nplane(x: number, y: number, z: number, p_id: number): boolean;
  nplane(x: number, y: number, z: number, a: number, b?: number): boolean {
    if (b === undefined) {
      const rsq = x * x + y * y + z * z;
      return this.nplane_impl(this, x, y, z, rsq, a);
    }
    return this.nplane_impl(this, x, y, z, a, b);
  }

  /** Cuts the cell by a plane with plane ID 0 (the C plane()). */
  plane(x: number, y: number, z: number, rsq: number): boolean;
  plane(x: number, y: number, z: number): boolean;
  plane(x: number, y: number, z: number, rsq?: number): boolean {
    if (rsq === undefined) {
      const r2 = x * x + y * y + z * z;
      return this.nplane_impl(this, x, y, z, r2, 0);
    }
    return this.nplane_impl(this, x, y, z, rsq, 0);
  }

  /** Initializes the class to be a rectangular box with neighbour IDs -1 to -6. */
  init(x_min: number, x_max: number, y_min: number, y_max: number, z_min: number, z_max: number): void {
    this.init_base(x_min, x_max, y_min, y_max, z_min, z_max);
    const q = this.mne[3];
    q[0] = -5; q[1] = -3; q[2] = -1;
    q[3] = -5; q[4] = -2; q[5] = -3;
    q[6] = -5; q[7] = -1; q[8] = -4;
    q[9] = -5; q[10] = -4; q[11] = -2;
    q[12] = -6; q[13] = -1; q[14] = -3;
    q[15] = -6; q[16] = -3; q[17] = -2;
    q[18] = -6; q[19] = -4; q[20] = -1;
    q[21] = -6; q[22] = -2; q[23] = -4;
    this.ne[0] = q;
    this.ne[1] = q.subarray(3);
    this.ne[2] = q.subarray(6);
    this.ne[3] = q.subarray(9);
    this.ne[4] = q.subarray(12);
    this.ne[5] = q.subarray(15);
    this.ne[6] = q.subarray(18);
    this.ne[7] = q.subarray(21);
  }

  /** Checks that the neighbour information of each face is consistent. */
  check_facets(): void {
    let i: number;
    let j: number;
    let k: number;
    let l: number;
    let m: number;
    let q: number;
    for (i = 1; i < this.p; i++) {
      for (j = 0; j < this.nu[i]; j++) {
        k = this.ed[i][j];
        if (k >= 0) {
          this.ed[i][j] = -1 - k;
          q = this.ne[i][j];
          l = this.cycle_up(this.ed[i][this.nu[i] + j], k);
          do {
            m = this.ed[k][l];
            this.ed[k][l] = -1 - m;
            if (this.ne[k][l] !== q) {
              throw new Error(`Facet error at (${k},${l})=${this.ne[k][l]}, started from (${i},${j})=${q}`);
            }
            l = this.cycle_up(this.ed[k][this.nu[k] + l], m);
            k = m;
          } while (k !== i);
        }
      }
    }
    this.reset_edges();
  }

  /** Computes a list of the neighbour IDs of the faces. */
  override neighbors(nbr_indices: number[]): void {
    nbr_indices.length = 0;
    let i: number;
    let j: number;
    let k: number;
    let l: number;
    let m: number;
    for (i = 1; i < this.p; i++) {
      for (j = 0; j < this.nu[i]; j++) {
        k = this.ed[i][j];
        if (k >= 0) {
          nbr_indices.push(this.ne[i][j]);
          this.ed[i][j] = -1 - k;
          l = this.cycle_up(this.ed[i][this.nu[i] + j], k);
          do {
            m = this.ed[k][l];
            this.ed[k][l] = -1 - m;
            l = this.cycle_up(this.ed[k][this.nu[k] + l], m);
            k = m;
          } while (k !== i);
        }
      }
    }
    this.reset_edges();
  }

  // ---- neighbour memory hooks (the C n_* inline members) ----

  n_allocate(i: number, m: number): void {
    this.mne[i] = new Int32Array(m * i);
  }

  n_add_memory_vertices(i: number): void {
    const pp = emptyRecords(i);
    for (let j = 0; j < this.current_vertices; j++) pp[j] = this.ne[j];
    this.ne = pp;
  }

  n_add_memory_vorder(i: number): void {
    const p2 = emptyRecords(i);
    for (let j = 0; j < this.current_vertex_order; j++) p2[j] = this.mne[j];
    this.mne = p2;
  }

  n_set_pointer(p: number, n: number): void {
    this.ne[p] = this.mne[n].subarray(n * this.mec[n]);
  }

  n_copy(a: number, b: number, c: number, d: number): void {
    this.ne[a][b] = this.ne[c][d];
  }

  n_set(a: number, b: number, c: number): void {
    this.ne[a][b] = c;
  }

  n_set_aux1(k: number): void {
    this.paux1 = this.mne[k].subarray(k * this.mec[k]);
  }

  n_copy_aux1(a: number, b: number): void {
    this.paux1[b] = this.ne[a][b];
  }

  n_copy_aux1_shift(a: number, b: number): void {
    this.paux1[b] = this.ne[a][b + 1];
  }

  n_set_aux2_copy(a: number, b: number): void {
    this.paux2 = this.mne[b].subarray(b * this.mec[b]);
    for (let i = 0; i < b; i++) this.ne[a][i] = this.paux2[i];
  }

  n_copy_pointer(a: number, b: number): void {
    this.ne[a] = this.ne[b];
  }

  n_set_to_aux1(j: number): void {
    this.ne[j] = this.paux1;
  }

  n_set_to_aux2(j: number): void {
    this.ne[j] = this.paux2;
  }

  n_allocate_aux1(i: number): void {
    this.paux1 = new Int32Array(i * this.mem[i]);
  }

  n_switch_to_aux1(i: number): void {
    this.mne[i] = this.paux1;
  }

  n_copy_to_aux1(i: number, m: number): void {
    this.paux1[m] = this.mne[i][m];
  }

  n_set_to_aux1_offset(k: number, m: number): void {
    this.ne[k] = this.paux1.subarray(m);
  }
}
