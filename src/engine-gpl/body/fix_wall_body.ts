/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/fix_wall_body_polygon.cpp,
 * fix_wall_body_polygon.h, fix_wall_body_polyhedron.cpp and fix_wall_body_polyhedron.h (FixWallBodyPolygon and
 * FixWallBodyPolyhedron: the wall force on the vertices of rounded polygon and rounded polyhedron bodies; init,
 * post_force, wall_force, sum_torque, total_velocity). LAMMPS is Copyright (2003) Sandia Corporation and distributed
 * under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Contributing author (from the source): Trung Dac Nguyen (ndactrung@gmail.com).
 *
 * docs.lammps.org/fix_wall_body_polygon.html and fix_wall_body_polyhedron.html: "fix ID group-ID wall/body/polygon
 * k_n c_n c_t wallstyle args keyword values" with "wallstyle = xplane or yplane" (polygon) or zplane (polyhedron),
 * and "args = lo hi" (NULL for no wall on that side). The wall acts on the vertices of each body in the group: a
 * vertex overlapping the wall by the rounded radius less its signed distance feels a spring force k_n and a damping
 * force c_n (normal) and c_t (tangential) from the velocity of the body at the contact point, and the torque of that
 * force acts on the body. The polyhedron wall is the same with the polyhedron record (bodies.ts POLY_OFF layout).
 *
 * Supported: the elastic and damping wall force, and the walls at lo and hi. Refused by name (StyleError): the wiggle
 * keyword and the history keyword (the tangential contact history of the wall is not ported); the pair style must
 * be pair body/rounded/polygon or body/rounded/polyhedron (native init: "... is incompatible with Pair style").
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { BODY_RECORD, bodyRecordOffset, POLY_OFF } from '../../engine/bodies';
import { quatToMatrix } from '../../engine/compute/asphere';
import { atomRadius, bodyGeom, type BodyGeom, type V3 } from './rounded_polygon_law';

/** Space-frame angular velocity of body i from its angular momentum and principal moments (native angmom_to_omega). */
const spaceOmega = (q: Float64Array, L: Float64Array, i: number, I: [number, number, number]): V3 => {
  const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
  const m: V3 = [L[3 * i], L[3 * i + 1], L[3 * i + 2]];
  const w: V3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    // the principal axis k is column k of R
    const e: V3 = [R[k], R[3 + k], R[6 + k]];
    const wb = I[k] === 0 ? 0 : (m[0] * e[0] + m[1] * e[1] + m[2] * e[2]) / I[k];
    w[0] += wb * e[0]; w[1] += wb * e[1]; w[2] += wb * e[2];
  }
  return w;
};

export class FixWallBody extends Fix {
  readonly style: string;
  private readonly kn: number;
  private readonly cn: number;
  private readonly ct: number;
  private readonly dim: 0 | 1 | 2;
  private readonly lo: number;
  private readonly hi: number;
  private readonly polyhedron: boolean;

  constructor(sys: System, id: string, group: string, args: string[], variant: 'polygon' | 'polyhedron') {
    super(sys, id, group, args);
    this.polyhedron = variant === 'polyhedron';
    this.style = `wall/body/${variant}`;
    if (args.length < 6) throw new StyleError(`Illegal fix ${this.style} command`);
    const num = (w: string, what: string): number => {
      const v = Number(w);
      if (!Number.isFinite(v)) throw new StyleError(`Illegal fix ${this.style} ${what}: ${w}`);
      return v;
    };
    this.kn = num(args[0], 'k_n');
    this.cn = num(args[1], 'c_n');
    // native: c_t NULL is 0.5 c_n
    this.ct = args[2] === 'NULL' ? 0.5 * this.cn : num(args[2], 'c_t');
    
    if (args[3] === 'xplane') this.dim = 0;
    else if (args[3] === 'yplane') this.dim = 1;
    else if (args[3] === 'zplane' && this.polyhedron) this.dim = 2;
    else throw new StyleError(`Illegal fix ${this.style} wallstyle ${args[3]}`);
    // native: a NULL wall is at infinity (-BIG for lo, BIG for hi)
    this.lo = args[4] === 'NULL' ? -Infinity : num(args[4], 'lo');
    this.hi = args[5] === 'NULL' ? Infinity : num(args[5], 'hi');
    for (let k = 6; k < args.length; k++) {
      if (args[k] === 'wiggle') throw new StyleError(`Fix ${this.style} wiggle is not supported by the browser engine`);
      if (args[k] === 'history') throw new StyleError(`Fix ${this.style} history is not implemented yet by the browser engine`);
      throw new StyleError(`Unknown fix ${this.style} keyword ${args[k]}`);
    }
    const layout = this.polyhedron ? 'rounded/polyhedron' : 'rounded/polygon';
    if (!sys.state.quat || !sys.state.angmom || !sys.state.torque || !sys.state.custom.has(BODY_RECORD)) {
      throw new StyleError(`Fix ${this.style} requires atom style body`);
    }
    if (!sys.state.atomStyle.startsWith(`body ${layout}`)) {
      throw new StyleError(`Fix ${this.style} requires body style ${layout}`);
    }
  }

  init(): void {
    const want = this.polyhedron ? 'body/rounded/polyhedron' : 'body/rounded/polygon';
    if (this.sys.ff.pair?.name !== want) throw new StyleError(`Fix ${this.style} is incompatible with Pair style`);
  }

  /**
   * Space-frame geometry of body i: the vertices (disc), the rounded and enclosing radii of its record. The polygon
   * record keeps the diameter (rounded radius = half of it); the polyhedron record keeps both radii (bodies.ts).
   */
  private geometry(i: number, B: number, rec: Float64Array): BodyGeom {
    const s = this.sys.state;
    const nv = rec[B * i];
    const off = this.polyhedron ? POLY_OFF : bodyRecordOffset('rounded/polygon');
    const body: V3[] = [];
    for (let k = 0; k < nv; k++) body.push([rec[B * i + off + 3 * k], rec[B * i + off + 3 * k + 1], rec[B * i + off + 3 * k + 2]]);
    const I: [number, number, number] = [rec[B * i + 1], rec[B * i + 2], rec[B * i + 3]];
    const q = s.quat!;
    const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
    const xc: V3 = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
    const vcm: V3 = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
    const geo = bodyGeom(xc, vcm, spaceOmega(q, s.angmom!, i, I), body, R, this.polyhedron ? 0 : rec[B * i + 4]);
    if (this.polyhedron) {
      geo.erad = rec[B * i + 6];
      geo.rrad = rec[B * i + 7];
    }
    return geo;
  }

  postForce(): void {
    const s = this.sys.state;
    const rec = s.custom.get(BODY_RECORD)!.data;
    const B = s.custom.get(BODY_RECORD)!.cols;
    const x = s.x, f = s.f, tq = s.torque!, d = this.dim;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (rec[B * i] < 1) continue;
      const geo = this.geometry(i, B, rec);
      const xc: V3 = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
      const vcm = geo.vcm, om = geo.omega;
      const radius = atomRadius(geo), rrad = geo.rrad;
      // lo wall first (inward normal +), then hi (inward normal -), as native post_force
      for (const nside of [1, -1]) {
        const wall = nside === 1 ? this.lo : this.hi;
        if (!Number.isFinite(wall)) continue;
        const nw: V3 = [0, 0, 0];
        nw[d] = nside;
        const scom = (xc[d] - wall) * nw[d];
        if (scom > radius) continue;
        for (const p of geo.disc) {
          const xp: V3 = [xc[0] + p[0], xc[1] + p[1], xc[2] + p[2]];
          const sd = (xp[d] - wall) * nw[d];
          const Rv = sd - rrad;
          if (Rv >= 0) continue;
          // contact point halfway between the rounded surface and the wall
          const pc: V3 = [xp[0] - 0.5 * (sd + rrad) * nw[0], xp[1] - 0.5 * (sd + rrad) * nw[1], xp[2] - 0.5 * (sd + rrad) * nw[2]];
          // velocity of the body at the contact point: vcm + omega x (pc - xcm)
          const r: V3 = [pc[0] - xc[0], pc[1] - xc[1], pc[2] - xc[2]];
          const vi: V3 = [om[1] * r[2] - om[2] * r[1] + vcm[0], om[2] * r[0] - om[0] * r[2] + vcm[1], om[0] * r[1] - om[1] * r[0] + vcm[2]];
          const vnn = vi[0] * nw[0] + vi[1] * nw[1] + vi[2] * nw[2];
          const vn: V3 = [vnn * nw[0], vnn * nw[1], vnn * nw[2]];
          const vt: V3 = [vi[0] - vn[0], vi[1] - vn[1], vi[2] - vn[2]];
          const fw: V3 = [0, 1, 2].map((k) => -this.kn * Rv * nw[k] - this.cn * vn[k] - this.ct * vt[k]) as V3;
          f[3 * i] += fw[0]; f[3 * i + 1] += fw[1]; f[3 * i + 2] += fw[2];
          // torque of the force at pc about the centre of mass (native sum_torque)
          tq[3 * i] += r[1] * fw[2] - r[2] * fw[1];
          tq[3 * i + 1] += r[2] * fw[0] - r[0] * fw[2];
          tq[3 * i + 2] += r[0] * fw[1] - r[1] * fw[0];
        }
      }
    }
  }
}
