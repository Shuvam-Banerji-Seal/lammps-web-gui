/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the Bodies section of write_data,
 * src/BODY/body_nparticle.cpp, body_rounded_polygon.cpp and body_rounded_polyhedron.cpp (pack_data_body and
 * write_data_body of each body style), with src/atom_vec_body.cpp (pack_data_bonus: the bodies in local order) and the
 * src/math_extra.h helpers they use (quat_to_mat, times3_diag, times3_transpose, matvec).
 * LAMMPS is Copyright Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (kind HOOKS, word 'body', loaded with atom_style body):
 * write_data (src/engine/output/data.ts) reads the lines through gplHook('body').
 */
import type { SimState } from '../../engine/types';
import { BODY_RECORD, POLY_OFF, bodyRecordOffset } from '../../engine/bodies';
import { shortest, type BodyWriteHook } from '../../engine/output/data';

/** MathExtra::quat_to_mat (row-major). */
const quatToMat = (q: ArrayLike<number>, o: number): number[] => {
  const w2 = q[o] * q[o], i2 = q[o + 1] * q[o + 1], j2 = q[o + 2] * q[o + 2], k2 = q[o + 3] * q[o + 3];
  const twoij = 2.0 * q[o + 1] * q[o + 2], twoik = 2.0 * q[o + 1] * q[o + 3], twojk = 2.0 * q[o + 2] * q[o + 3];
  const twoiw = 2.0 * q[o + 1] * q[o], twojw = 2.0 * q[o + 2] * q[o], twokw = 2.0 * q[o + 3] * q[o];
  return [
    w2 + i2 - j2 - k2, twoij - twokw, twojw + twoik,
    twoij + twokw, w2 - i2 + j2 - k2, twojk - twoiw,
    twoik - twojw, twojk + twoiw, w2 - i2 - j2 + k2,
  ];
};

/**
 * "MathExtra::quat_to_mat(quat,p); MathExtra::times3_diag(p,inertia,pdiag); MathExtra::times3_transpose(pdiag,p,ispace);"
 * and the six values ixx iyy izz ixy ixz iyz.
 */
const spaceInertia = (p: number[], inertia: number[]): number[] => {
  const pd = [0, 1, 2].flatMap((i) => [p[3 * i] * inertia[0], p[3 * i + 1] * inertia[1], p[3 * i + 2] * inertia[2]]);
  const t = (i: number, j: number): number => pd[3 * i] * p[3 * j] + pd[3 * i + 1] * p[3 * j + 1] + pd[3 * i + 2] * p[3 * j + 2];
  return [t(0, 0), t(1, 1), t(2, 2), t(0, 1), t(0, 2), t(1, 2)];
};

/** MathExtra::matvec: the displacement of a sub-particle in the box frame. */
const matvec = (p: number[], d0: number, d1: number, d2: number): number[] =>
  [p[0] * d0 + p[1] * d1 + p[2] * d2, p[3] * d0 + p[4] * d1 + p[5] * d2, p[6] * d0 + p[7] * d1 + p[8] * d2];

const bodiesSection = (s: SimState, order: ArrayLike<number>, bstyle: string): string[] => {
  const rec = s.custom.get(BODY_RECORD)!;
  const B = rec.cols, d = rec.data;
  const q = s.quat!;
  const out: string[] = [];
  for (let k = 0; k < order.length; k++) {
    const i = order[k];
    const base = B * i;
    const nsub = d[base];
    if (!(nsub > 0)) continue;
    const p = quatToMat(q, 4 * i);
    const ispace = spaceInertia(p, [d[base + 1], d[base + 2], d[base + 3]]);
    const o0 = bodyRecordOffset(bstyle);
    const verts: string[] = [];
    for (let v = 0; v < nsub; v++) {
      const at = base + o0 + 3 * v;
      verts.push(matvec(p, d[at], d[at + 1], d[at + 2]).map(shortest).join(' '));
    }
    const id = s.id[i];
    if (bstyle === 'rounded/polyhedron') {
      // "buf[m++] = ubuf(3).d;" and "if (nsub < 3) ... 6 + 3*nsub + 1 else 6 + 3*nsub + 2*nedge + MAX_FACE_SIZE*nface + 1"
      const nedge = d[base + 4], nface = d[base + 5];
      const ndouble = nsub < 3 ? 6 + 3 * nsub + 1 : 6 + 3 * nsub + 2 * nedge + 4 * nface + 1;
      out.push(`${id} 3 ${ndouble}`, `${nsub} ${nedge} ${nface}`, ispace.map(shortest).join(' '), ...verts);
      if (nsub > 2) {
        const eo = base + POLY_OFF + 3 * nsub;
        for (let e = 0; e < nedge; e++) out.push(`${Math.trunc(d[eo + 2 * e])} ${Math.trunc(d[eo + 2 * e + 1])}`);
        const fo = eo + 2 * nedge;
        for (let f = 0; f < nface; f++) out.push([0, 1, 2, 3].map((m) => String(Math.trunc(d[fo + 4 * f + m]))).join(' '));
      }
      // "buf[m++] = 2.0 * dvalue[j+1];": the rounded diameter, twice the rounded radius
      out.push(shortest(2.0 * d[base + 7]));
    } else if (bstyle === 'rounded/polygon') {
      out.push(`${id} 1 ${6 + 3 * nsub + 1}`, String(nsub), ispace.map(shortest).join(' '), ...verts);
      // "buf[m++] = 2.0 * dvalue[ilast];": the rounded diameter (the record keeps the diameter itself)
      out.push(shortest(2.0 * (0.5 * d[base + 4])));
    } else {
      out.push(`${id} 1 ${6 + 3 * nsub}`, String(nsub), ispace.map(shortest).join(' '), ...verts);
    }
  }
  return out;
};

export const BODY_WRITE_HOOK: BodyWriteHook = { bodiesSection };
