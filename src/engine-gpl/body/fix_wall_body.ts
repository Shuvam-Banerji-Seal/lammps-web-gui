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
 * Supported: the elastic and damping wall force, the walls at lo and hi, and the keyword history (a tangential spring
 * per body and wall, its deformation kept per atom ID; see contact_history.ts for the shared spring). Refused by name
 * (StyleError): the wiggle keyword; the pair style must be pair body/rounded/polygon or body/rounded/polyhedron
 * (native init: "... is incompatible with Pair style").
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { numeric } from '../../engine/force/util';
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
  /** keyword history: friction coefficient mu and tangential stiffness k_t (native defaults 2/7 k_n). */
  private history = false;
  private mu = 0;
  private kt = 0;
  /** the tangential deformation of each body at the walls [lo, hi], keyed by atom ID (zero when not touching). */
  private shear = new Map<number, [V3, V3]>();
  /** true while the run setup calls postForce (native update->setupflag): the shear does not advance then. */
  private setupCall = false;

  constructor(sys: System, id: string, group: string, args: string[], variant: 'polygon' | 'polyhedron') {
    super(sys, id, group, args);
    this.polyhedron = variant === 'polyhedron';
    this.style = `wall/body/${variant}`;
    if (args.length < 6) throw new StyleError(`Illegal fix ${this.style} command`);
    // As native LAMMPS does (src/BODY/fix_wall_body_polygon.cpp at b332f28): utils::numeric for the numbers
    const num = (w: string, _what: string): number => numeric(w);
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
      if (args[k] !== 'history') throw new StyleError(`Unknown fix ${this.style} keyword ${args[k]}`);
      // native: history mu kt, kt NULL is 2/7 k_n (as in pair body/rounded/polygon)
      if (k + 3 > args.length) throw new StyleError(`Illegal fix ${this.style} history: needs mu and kt`);
      this.mu = num(args[k + 1], 'history mu');
      if (this.mu < 0) throw new StyleError(`Illegal fix ${this.style} history argument ${this.mu}`);
      this.kt = args[k + 2] === 'NULL' ? (2.0 / 7.0) * this.kn : num(args[k + 2], 'history kt');
      if (this.kt < 0) throw new StyleError(`Illegal fix ${this.style} history argument ${this.kt}`);
      this.history = true;
      k += 2;
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

  setup(): void {
    this.setupCall = true;
    try {
      super.setup();
    } finally {
      this.setupCall = false;
    }
  }

  postForce(): void {
    const s = this.sys.state;
    const rec = s.custom.get(BODY_RECORD)!.data;
    const B = s.custom.get(BODY_RECORD)!.cols;
    const x = s.x, f = s.f, tq = s.torque!, d = this.dim;
    // the shear of the bodies touching a wall this step (native: the history of the others is reset)
    const nextShear = new Map<number, [V3, V3]>();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (rec[B * i] < 1) continue;
      const geo = this.geometry(i, B, rec);
      const xc: V3 = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
      const vcm = geo.vcm, om = geo.omega;
      const radius = atomRadius(geo), rrad = geo.rrad;
      const prev = this.history ? this.shear.get(s.id[i]) : undefined;
      const xiNext: [V3, V3] = [[0, 0, 0], [0, 0, 0]];
      let touched = false;
      // lo wall first (inward normal +), then hi (inward normal -), as native post_force
      for (const nside of [1, -1]) {
        const wall = nside === 1 ? this.lo : this.hi;
        if (!Number.isFinite(wall)) continue;
        const nw: V3 = [0, 0, 0];
        nw[d] = nside;
        const scom = (xc[d] - wall) * nw[d];
        if (scom > radius) continue;
        const iwall = nside === 1 ? 0 : 1;
        // with the history, one friction force per wall acts on the body, at the contact points weighted by their
        // elastic normal forces (native fnsum, pcsum, vtsum)
        let fnsum = 0;
        const pcsum: V3 = [0, 0, 0], vtsum: V3 = [0, 0, 0];
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
          if (this.history && -this.kn * Rv > 0) {
            const fne = -this.kn * Rv;
            fnsum += fne;
            for (let k = 0; k < 3; k++) {
              pcsum[k] += fne * pc[k];
              vtsum[k] += fne * vt[k];
            }
          }
        }
        if (this.history && fnsum > 0) {
          // native: pcsum and vtsum are scaled by 1/fnsum (scale3), then tangential_spring()
          const inv = 1.0 / fnsum;
          const pcm: V3 = [inv * pcsum[0], inv * pcsum[1], inv * pcsum[2]];
          const vtm: V3 = [inv * vtsum[0], inv * vtsum[1], inv * vtsum[2]];
          const xi0: V3 = prev ? [...prev[iwall]] as V3 : [0, 0, 0];
          const sp = wallSpring(nw, vtm, fnsum, xi0, this.kt, this.mu, s.dt, !this.setupCall);
          f[3 * i] += sp.ft[0]; f[3 * i + 1] += sp.ft[1]; f[3 * i + 2] += sp.ft[2];
          const r: V3 = [pcm[0] - xc[0], pcm[1] - xc[1], pcm[2] - xc[2]];
          tq[3 * i] += r[1] * sp.ft[2] - r[2] * sp.ft[1];
          tq[3 * i + 1] += r[2] * sp.ft[0] - r[0] * sp.ft[2];
          tq[3 * i + 2] += r[0] * sp.ft[1] - r[1] * sp.ft[0];
          xiNext[iwall] = sp.xi;
          touched = true;
        }
      }
      if (touched) nextShear.set(s.id[i], xiNext);
    }
    if (this.history) this.shear = nextShear;
  }
}

/**
 * tangential_spring (native fix wall/body/polygon and polyhedron): xi is rotated into the tangent plane of the wall
 * normal n keeping its magnitude, advanced by vt dt unless in the setup, the spring force -kt xi is limited to mu fne
 * and xi is reduced with it. Returns the force and the new deformation.
 */
const wallSpring = (n: V3, vt: V3, fne: number, xi0: V3, kt: number, mu: number, dt: number, update: boolean): { ft: V3; xi: V3 } => {
  const dotv = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const lenv = (a: V3): number => Math.sqrt(dotv(a, a));
  let xi: V3 = [xi0[0], xi0[1], xi0[2]];
  const xin = dotv(xi, n);
  const mag = lenv(xi);
  xi = [xi[0] - xin * n[0], xi[1] - xin * n[1], xi[2] - xin * n[2]];
  const magt = lenv(xi);
  if (magt > 0.0) {
    const sc = mag / magt;
    xi = [sc * xi[0], sc * xi[1], sc * xi[2]];
  }
  if (update) xi = [xi[0] + vt[0] * dt, xi[1] + vt[1] * dt, xi[2] + vt[2] * dt];
  let ft: V3 = [-kt * xi[0], -kt * xi[1], -kt * xi[2]];
  const ftmag = lenv(ft);
  const ftmax = mu * fne;
  if (ftmag > ftmax) {
    const sc = ftmax / ftmag;
    ft = [sc * ft[0], sc * ft[1], sc * ft[2]];
    xi = [sc * xi[0], sc * xi[1], sc * xi[2]];
  }
  return { ft, xi };
};
