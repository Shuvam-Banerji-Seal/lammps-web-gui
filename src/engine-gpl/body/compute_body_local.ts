/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/compute_body_local.cpp and
 * compute_body_local.h (ComputeBodyLocal: compute body/local, the argument parsing, compute_body (the row count and
 * the fill of the local vector or array) and the per-row values). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_body_local.html: "compute ID group-ID body/local input1 input2 ..." with the keywords "id"
 * (atom ID of the body particle), "type" (atom type of the body particle) and "integer" (1,2,3,etc = index of fields
 * defined by body style). "Define a computation that calculates properties of individual body sub-particles." and
 * "For both body particles and non-body particles, the *id* keyword will store the ID of the particle."
 *
 * Rows: a body atom in the group contributes one row per sub-particle (its vertices for rounded/polygon and
 * rounded/polyhedron, its sub-particles for nparticle), in atom order; a non-body atom contributes one row. One
 * keyword gives a local vector, two or more a local array with one column per keyword.
 *
 * The integer fields of a body sub-particle are the space coordinates x, y, z of the sub-particle, which is what
 * the native body styles output (BodyNparticle::output, BodyRoundedPolygon::output and BodyRoundedPolyhedron::output
 * in src/BODY at b332f28: the body-frame displacement rotated by the body quaternion, plus the body's position);
 * noutcol() is 3 for all three styles, so the integer must be 1, 2 or 3. For a non-body atom the integer is its own
 * x, y or z (the native "x[i][index]"). The native init check for non-body particles (index above 2 is refused
 * when a non-body atom is in the group) cannot fire here: every integer is at most 3.
 *
 * Output units: the body style's (the vertex coordinates are in the box length units).
 */
import { Compute } from '../../engine/compute/compute';
import { quatToMatrix, rotateVec } from '../../engine/compute/asphere';
import { BODY_RECORD, bodyRecordOffset, parseBodyStyle } from '../../engine/bodies';
import { StyleError } from '../../engine/force/types';
import { inumeric } from '../../engine/force/util';
import type { System } from '../../engine/system';

/** One keyword: 'id', 'type', or the 0-based index of a space coordinate (integer 1..3 in the command). */
type Field = 'id' | 'type' | number;

export class ComputeBodyLocal extends Compute {
  readonly style: string = 'body/local';
  private readonly which: Field[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal compute body/local command');
    // compute_body_local.cpp: the words (id, type or an index read with utils::inumeric), then the atom style, then the
    // indices, which must be < noutcol() (3 for every body style)
    for (const w of args) {
      if (w === 'id' || w === 'type') { this.which.push(w); continue; }
      this.which.push(inumeric(w) - 1);
    }
    if (!sys.state.custom.has(BODY_RECORD) || !sys.state.quat) throw new StyleError('Compute body/local requires atom style body');
    for (const w of this.which) {
      if (typeof w === 'number' && (w < 0 || w >= 3)) throw new StyleError('Invalid index in compute body/local command');
    }
    this.localFlag = true;
    this.sizeLocalCols = this.which.length === 1 ? 0 : this.which.length;
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    const s = this.sys.state;
    const rec = s.custom.get(BODY_RECORD)!;
    const B = rec.cols, bd = rec.data;
    const bstyle = parseBodyStyle(s.atomStyle);
    if (!bstyle) throw new StyleError('Compute body/local requires atom style body');
    // the first sub-particle (vertex) displacement of the record of this body style (bodies.ts)
    const off = bodyRecordOffset(bstyle.bstyle);
    const q = s.quat!, x = s.x;
    const m = this.which.length;
    const cols = Math.max(1, m);

    // count the rows: one per sub-particle of a body atom, one per non-body atom (native compute_body(0))
    let rows = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      rows += bd[B * i] === 0 ? 1 : bd[B * i];
    }
    const out = new Float64Array(rows * cols);
    const vals = [0, 0, 0];
    const put = (r: number, i: number): void => {
      for (let c = 0; c < m; c++) {
        const w = this.which[c];
        out[r * cols + c] = w === 'id' ? s.id[i] : w === 'type' ? s.type[i] : vals[w];
      }
    };

    let r = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const nsub = bd[B * i];
      if (nsub === 0) {
        // non-body particle: its own coordinates
        vals[0] = x[3 * i]; vals[1] = x[3 * i + 1]; vals[2] = x[3 * i + 2];
        put(r++, i);
        continue;
      }
      // body particle: each sub-particle's space position = R d_body + x (native Body*::output with the bonus quaternion)
      const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
      for (let k = 0; k < nsub; k++) {
        const o = B * i + off + 3 * k;
        const [dx, dy, dz] = rotateVec(R, bd[o], bd[o + 1], bd[o + 2]);
        vals[0] = dx + x[3 * i]; vals[1] = dy + x[3 * i + 1]; vals[2] = dz + x[3 * i + 2];
        put(r++, i);
      }
    }
    this.localRows = rows;
    return out;
  }
}
