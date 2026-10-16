/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/pair_body_rounded_polyhedron.cpp
 * (the contact search, the contact list, the weights, the friction and the damping) and
 * src/BODY/body_rounded_polyhedron.cpp (the body geometry, body2space). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Original authors (from the sources): Trung Dac Nguyen (ndactrung@gmail.com) and the
 * contributors named in the native header. Reference: Wang, Nguyen, Dodds et al., rounded polyhedron contact model.
 *
 * docs.lammps.org/pair_body_rounded_polyhedron.html: "pair_style body/rounded/polyhedron c_n c_t mu A_ua cutoff
 * keyword" and "pair_coeff * * k_n k_na". The contact law is that of the native code, transcribed in the order of
 * its loops, so that the contact list and the sums of the forces follow the native sequence.
 *
 * The keyword history (the tangential spring of the contact history, see contact_history.ts) is applied per pair by
 * applyHistory(), called by the style after interact(). Not ported: the work done by the forces that do not derive
 * from the energy (only pvector), and the debug printouts of the native code.
 *
 * The per-atom quantities of native (discrete, facnorm, facplane, reach, vertex_done) live on the PBody objects of
 * one pair compute; the contacts of a pair of bodies are the Contact records of the native contact list.
 */
import { normalForce } from './rounded_polygon_law';
import { accumulateContact, historyFriction, newContactHistory, type ContactHistory } from './contact_history';
import { cos as libmCos } from '../libm/sincos';

export type V3 = [number, number, number];

const EPSILON = 1.0e-3;
const PARALLEL_TOL = 1.0e-12;
const CONE_TOL = 1.0e-10;
const BIG = 1.0e20;
const MAX_FACE_SIZE = 4;
const MY_2PI = 2 * Math.PI;
const DEG2RAD = Math.PI / 180;
const PATCH_ANGLE = 5.0 * DEG2RAD;
const LINE_ANGLE = 5.0 * DEG2RAD;
const COS_PATCH_ANGLE = libmCos(PATCH_ANGLE);
const COS_HALF_PATCH_ANGLE = libmCos(0.5 * PATCH_ANGLE);
const COS_LINE_ANGLE = libmCos(LINE_ANGLE);

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl = (s: number, a: V3): V3 => [s * a[0], s * a[1], s * a[2]];
const neg = (a: V3): V3 => [-a[0], -a[1], -a[2]];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.sqrt(dot(a, a));
const distsq = (a: V3, b: V3): number => dot(sub(a, b), sub(a, b));
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
/** MathExtra::norm3: multiply by the reciprocal length. */
const unit = (a: V3): V3 => scl(1.0 / len(a), a);
const clamp01 = (x: number): number => Math.max(0.0, Math.min(1.0, x));
/** sum_torque (native): the torque about xm of the force f acting at x. */
const torqueOf = (xm: V3, x: V3, f: V3): V3 => {
  const r = sub(x, xm);
  return [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
};

/** Space-frame body: one atom of a rounded polyhedron (native discrete, facnorm, facplane, edges and faces). */
export interface PBody {
  /** index of the atom in the force arrays (owned or ghost) */
  idx: number;
  /** centre of mass (the atom position) */
  c: V3;
  /** centre-of-mass velocity and angular velocity (space frame) */
  vcm: V3;
  omega: V3;
  erad: number;
  rrad: number;
  /** sub-particle displacements from the centre of mass, space frame (native discrete) */
  disc: V3[];
  /** edges: pairs of vertex indices */
  edges: [number, number][];
  /** faces: vertex indices (3 or 4 per face) */
  faces: number[][];
  /** outward unit normal of each face, space frame (native facnorm) */
  fnorm: V3[];
  /** first vertex and outward unit normal of each face in space (native facplane) */
  fplane: { p: V3; n: V3 }[];
  /** per pair: vertices already interacted with, reach towards the other body (native vertex_done, reach) */
  vdone: Uint8Array;
  reach: Float64Array;
}

/**
 * Build a space-frame body (native body2space): disc = R * body-frame vertices, facnorm from the first three vertices
 * of each face (flipped to point away from the centre of mass), facplane from the atom position c.
 * R is the row-major rotation matrix of the body quaternion.
 */
export const makePBody = (idx: number, c: V3, vcm: V3, omega: V3, R: ArrayLike<number>, bodyVerts: V3[],
  edges: [number, number][], faces: number[][], erad: number, rrad: number): PBody => {
  const disc: V3[] = bodyVerts.map((d) => [
    R[0] * d[0] + R[1] * d[1] + R[2] * d[2],
    R[3] * d[0] + R[4] * d[1] + R[5] * d[2],
    R[6] * d[0] + R[7] * d[1] + R[8] * d[2],
  ]);
  const fnorm: V3[] = [];
  const fplane: { p: V3; n: V3 }[] = [];
  for (const fv of faces) {
    const x1 = disc[fv[0]], x2 = disc[fv[1]], x3 = disc[fv[2]];
    let n = unit(cross(sub(x2, x1), sub(x3, x1)));
    if (dot(x1, n) < 0.0) n = neg(n);
    fnorm.push(n);
    const xi1 = add(c, x1), xi2 = add(c, x2), xi3 = add(c, x3);
    let nf = unit(cross(sub(xi2, xi1), sub(xi3, xi1)));
    const xc: V3 = [(xi1[0] + xi2[0] + xi3[0]) / 3.0, (xi1[1] + xi2[1] + xi3[1]) / 3.0, (xi1[2] + xi2[2] + xi3[2]) / 3.0];
    if (dot(sub(xc, c), nf) < 0) nf = neg(nf);
    fplane.push({ p: xi1, n: nf });
  }
  return {
    idx, c, vcm, omega, erad, rrad, disc, edges, faces, fnorm, fplane,
    vdone: new Uint8Array(disc.length), reach: new Float64Array(disc.length),
  };
};

/** A contact of the pair (native struct Contact): the points on each body, the separation and the weight. */
export interface Contact {
  ib: PBody;
  jb: PBody;
  xi: V3;
  xj: V3;
  separation: number;
  r: number;
  unique: boolean;
  w: number;
  patch: boolean;
}

/** Parameters of one pair of types and of the style (native k_n, k_na, c_n, c_t, mu, A_ua, cut_inner). */
export interface PolyParams {
  kn: number;
  kna: number;
  cn: number;
  ct: number;
  mu: number;
  Aua: number;
  cut: number;
  /** keyword history: the friction is that of a tangential spring with the shear of the pair (see contact_history.ts). */
  history?: boolean;
  /** tangential stiffness k_t of the pair of types, the time step, the setup flag and the shear (as seen from body i). */
  kt?: number;
  dt?: number;
  update?: boolean;
  shear?: V3;
}

/** A corner of the clipped polygon of a region of overlap (native PatchVertex). */
interface PV {
  u: number;
  v: number;
  type: number;
  a: number;
  b: number;
  src: number;
  srci: number;
}

const MAXP = 2 * MAX_FACE_SIZE + 2;

const smoothstep = (x: number): number => {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  return x * x * (3.0 - 2.0 * x);
};

/** patch_factor (native): one up to half of PATCH_ANGLE between the faces, zero from PATCH_ANGLE on. */
const patchFactor = (n1: V3, n2: V3): number => {
  const c = -dot(n1, n2);
  if (c <= COS_PATCH_ANGLE) return 0.0;
  if (c >= COS_HALF_PATCH_ANGLE) return 1.0;
  const phi = Math.acos(Math.min(1.0, c));
  return 1.0 - smoothstep(2.0 * phi / PATCH_ANGLE - 1.0);
};

/** clip_patch (native): clip the convex polygon by the half-plane hu u + hv v + h0 >= 0; merge vertices closer than tol. */
const clipPatch = (poly: PV[], hu: number, hv: number, h0: number, line: number, nvi: number, tol: number): PV[] => {
  const np = poly.length;
  const work: PV[] = [];
  for (let m = 0; m < np; m++) {
    const pp = poly[m], pq = poly[(m + 1) % np];
    const sp = hu * pp.u + hv * pp.v + h0;
    const sq = hu * pq.u + hv * pq.v + h0;
    if (sp >= 0.0) {
      if (work.length >= MAXP) break;
      work.push({ ...pp });
    }
    if ((sp >= 0.0) !== (sq >= 0.0)) {
      if (work.length >= MAXP) break;
      const t = sp / (sp - sq);
      const pn: PV = { u: pp.u + t * (pq.u - pp.u), v: pp.v + t * (pq.v - pp.v), type: 3, a: -1, b: -1, src: 0, srci: 0 };
      if (line >= 0) {
        if (pp.src === 0) {
          pn.type = 2;
          pn.a = pp.srci;
          pn.b = line;
        } else if (pp.src === 1) {
          if (line === (pp.srci + 1) % nvi) {
            pn.type = 1;
            pn.a = line;
          } else if (pp.srci === (line + 1) % nvi) {
            pn.type = 1;
            pn.a = pp.srci;
          }
        }
      }
      if (sp >= 0.0) {
        pn.src = line >= 0 ? 1 : 2;
        pn.srci = line;
      } else {
        pn.src = pp.src;
        pn.srci = pp.srci;
      }
      work.push(pn);
    }
  }
  const out: PV[] = [];
  for (const w of work) {
    const last = out[out.length - 1];
    if (out.length > 0 && Math.abs(w.u - last.u) < tol && Math.abs(w.v - last.v) < tol) {
      last.src = w.src;
      last.srci = w.srci;
      continue;
    }
    out.push({ ...w });
  }
  if (out.length > 1 && Math.abs(out[out.length - 1].u - out[0].u) < tol && Math.abs(out[out.length - 1].v - out[0].v) < tol) out.pop();
  return out;
};

/** closest_segments (native, Ericson): c1 on p1-q1, c2 on p2-q2, and their distance. */
const closestSegments = (p1: V3, q1: V3, p2: V3, q2: V3): { c1: V3; c2: V3; d: number } => {
  const d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
  const a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
  let s: number, t: number;
  if (a === 0.0 && e === 0.0) {
    s = t = 0.0;
  } else if (a === 0.0) {
    s = 0.0;
    t = clamp01(f / e);
  } else {
    const cc = dot(d1, r);
    if (e === 0.0) {
      t = 0.0;
      s = clamp01(-cc / a);
    } else {
      const b = dot(d1, d2);
      const denom = a * e - b * b;
      s = denom > 0.0 ? clamp01((b * f - cc * e) / denom) : 0.0;
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = clamp01(-cc / a);
      } else if (t > 1.0) {
        t = 1.0;
        s = clamp01((b - cc) / a);
      }
    }
  }
  const c1: V3 = [p1[0] + s * d1[0], p1[1] + s * d1[1], p1[2] + s * d1[2]];
  const c2: V3 = [p2[0] + t * d2[0], p2[1] + t * d2[1], p2[2] + t * d2[2]];
  return { c1, c2, d: Math.sqrt(distsq(c1, c2)) };
};

/** Inverse of a 3x3 matrix (row-major m[3*r+c]) times the vector p. */
const solve3 = (m: number[], p: V3): V3 => {
  const det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  const inv = [
    (m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
    (m[5] * m[6] - m[3] * m[8]) / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
    (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det, (m[0] * m[4] - m[1] * m[3]) / det,
  ];
  return [
    inv[0] * p[0] + inv[1] * p[1] + inv[2] * p[2],
    inv[3] * p[0] + inv[4] * p[1] + inv[5] * p[2],
    inv[6] * p[0] + inv[7] * p[1] + inv[8] * p[2],
  ];
};

/** project_pt_plane (native, plane given by a point and a unit normal): the projection of q and its distance. */
const projectPlane = (q: V3, p: V3, n: V3): { qp: V3; d: number } => {
  const dt = dot(sub(q, p), n);
  const npv = scl(dt, n);
  const qp = sub(q, npv);
  return { qp, d: len(sub(q, qp)) };
};

/** project_pt_plane (native, through three points): the projection, its distance and whether it is inside the triangle. */
const projectTriangle = (q: V3, x1: V3, x2: V3, x3: V3): { qp: V3; d: number; inside: boolean } => {
  const u = sub(x2, x1), v = sub(x3, x1);
  const n = unit(cross(u, v));
  const p = sub(q, x1);
  const ans = solve3([-n[0], u[0], v[0], -n[1], u[1], v[1], -n[2], u[2], v[2]], p);
  const t = ans[0];
  const qp: V3 = [q[0] + n[0] * t, q[1] + n[1] * t, q[2] + n[2] * t];
  const inside = ans[1] > 0 && ans[2] > 0 && ans[1] + ans[2] < 1;
  return { qp, d: len(sub(q, qp)), inside };
};

/** project_pt_line (native): the foot h of q on the line xi1-xi2, its distance d and the fraction t along the segment. */
const projectLine = (q: V3, xi1: V3, xi2: V3): { h: V3; d: number; t: number } => {
  const u = unit(sub(xi2, xi1));
  const v = sub(q, xi1);
  const s = dot(u, v);
  const h: V3 = [xi1[0] + s * u[0], xi1[1] + s * u[1], xi1[2] + s * u[2]];
  const d = len(sub(q, h));
  const t = s / Math.sqrt(distsq(xi1, xi2));
  return { h, d, t };
};

/** distance_bt_edges (native): the closest points of the segments x1-x2 (h1, t1) and x3-x4 (h2, t2), and their distance r. */
const distanceEdges = (x1: V3, x2: V3, x3: V3, x4: V3): { h1: V3; h2: V3; t1: number; t2: number; r: number } => {
  let t1 = -2, t2 = 2;
  let h1: V3 = [0, 0, 0], h2: V3 = [0, 0, 0];
  const uu = unit(sub(x2, x1));
  const vv = unit(sub(x4, x3));
  const dt = Math.abs(dot(uu, vv));
  if (1.0 - dt < PARALLEL_TOL) {
    const x13 = sub(x1, x3);
    const s1 = dot(x13, vv);
    const x13b: V3 = [x13[0] - s1 * vv[0], x13[1] - s1 * vv[1], x13[2] - s1 * vv[2]];
    return { h1, h2, t1, t2, r: len(x13b) };
  }
  const n = unit(cross(uu, vv));
  const s = sub(x4, x3);
  const u = sub(x2, x1);
  const p = sub(x3, x1);
  const ans = solve3([-s[0], u[0], n[0], -s[1], u[1], n[1], -s[2], u[2], n[2]], p);
  t2 = ans[0];
  h2 = [x3[0] + s[0] * t2, x3[1] + s[1] * t2, x3[2] + s[2] * t2];
  const pl = projectPlane(h2, x1, n);
  h1 = pl.qp;
  const h1x1 = sub(h1, x1);
  t1 = dot(h1x1, u) / dot(u, u);
  return { h1, h2, t1, t2, r: pl.d };
};

/** opposite_sides (native): 1 if a and b are on opposite sides of the plane through x0 with normal n. */
const oppositeSides = (n: V3, x0: V3, a: V3, b: V3): boolean => {
  const ma = dot(n, sub(a, x0));
  const mb = dot(n, sub(b, x0));
  return ma * mb <= 0;
};

/** The rounded polyhedron interaction of one pair of bodies (native PairBodyRoundedPolyhedron::pair_interaction). */
export class PolyhedronPairLaw {
  /** forces (3 per atom, nall) and torques (3 per atom) accumulated in the compute, and the energy */
  f!: Float64Array;
  tq!: Float64Array;
  energy = 0;
  P!: PolyParams;
  contacts: Contact[] = [];
  private bi!: PBody;
  private bj!: PBody;
  /** the contact history of the pair (keyword history), its first body (native shear_i) and the two bodies. */
  private hist: ContactHistory | null = null;
  private histI = -1;
  private histBodies: [PBody, PBody] | null = null;
  private reachMinI = 0;
  private reachMinJ = 0;
  private patchAny = false;
  private patch = new Float64Array(0);

  /** the two-body interaction (native pair_interaction) for bodies i and j with the parameters P */
  interact(bi: PBody, bj: PBody): void {
    this.hist = this.P.history ? newContactHistory() : null;
    this.histI = bi.idx;
    this.histBodies = [bi, bj];
    const npi = bi.disc.length, npj = bj.disc.length;
    if (npi === 1 && npj === 1) {
      this.sphereSphere(bi, bj);
      return;
    }
    bi.vdone.fill(0);
    bj.vdone.fill(0);
    if (npi === 1 || npj === 1) {
      this.contacts = [];
      if (npj === 1) this.sphereAgainstPolyhedron(bi, bj);
      else this.sphereAgainstPolyhedron(bj, bi);
      if (this.contacts.length > 0) this.frictionForces();
      return;
    }
    const r = len(sub(bi.c, bj.c));
    const cut = (bi.rrad + bj.rrad + this.P.cut) * (1.0 + EPSILON);
    this.bi = bi;
    this.bj = bj;
    if (r > 0.0) {
      const del = sub(bi.c, bj.c);
      const uu: V3 = [-del[0] / r, -del[1] / r, -del[2] / r];
      let reachi = dot(bi.disc[0], uu);
      for (let ni = 0; ni < npi; ni++) {
        bi.reach[ni] = dot(bi.disc[ni], uu);
        reachi = Math.max(reachi, bi.reach[ni]);
      }
      let reachj = -dot(bj.disc[0], uu);
      for (let nj = 0; nj < npj; nj++) {
        bj.reach[nj] = -dot(bj.disc[nj], uu);
        reachj = Math.max(reachj, bj.reach[nj]);
      }
      if (r - reachi - reachj > cut) return;
      this.reachMinI = r - reachj - cut;
      this.reachMinJ = r - reachi - cut;
    } else {
      bi.reach.fill(0);
      bj.reach.fill(0);
      this.reachMinI = this.reachMinJ = -cut;
    }
    this.contacts = [];
    this.faceFacePatches(bi, bj);
    this.vertexAgainstCore(bi, bj);
    this.vertexAgainstCore(bj, bi);
    this.edgeAgainstFace(bi, bj);
    this.edgeAgainstFace(bj, bi);
    this.vertexAgainstEdge(bi, bj);
    this.vertexAgainstEdge(bj, bi);
    this.vertexAgainstVertex(bi, bj);
    this.edgeAgainstEdge(bi, bj);
    if (this.contacts.length > 0) {
      this.rescaleCohesiveForces();
      this.frictionForces();
    }
  }

  // ---- forces and torques -------------------------------------------------------------------------------

  private addForce(idx: number, F: V3): void {
    this.f[3 * idx] += F[0];
    this.f[3 * idx + 1] += F[1];
    this.f[3 * idx + 2] += F[2];
  }

  private addTorque(idx: number, xm: V3, x: V3, F: V3): void {
    const t = torqueOf(xm, x, F);
    this.tq[3 * idx] += t[0];
    this.tq[3 * idx + 1] += t[1];
    this.tq[3 * idx + 2] += t[2];
  }

  /** total_velocity (native): the velocity of the space point p of body b. */
  private totalVelocity(b: PBody, p: V3): V3 {
    const r = sub(p, b.c);
    const w = b.omega;
    return [w[1] * r[2] - w[2] * r[1] + b.vcm[0], w[2] * r[0] - w[0] * r[2] + b.vcm[1], w[0] * r[1] - w[1] * r[0] + b.vcm[2]];
  }

  /**
   * damping_friction (native): the damping, and the friction limited by mu fne. With the contact history the friction
   * of the contact is added to the history of the pair instead (and no force acts here), see applyHistory().
   */
  private dampingFriction(ib: PBody, jb: PBody, pc: V3, n: V3, fne: number, damping: boolean, friction: boolean, w: number): void {
    const vi = this.totalVelocity(ib, pc), vj = this.totalVelocity(jb, pc);
    const vr = sub(vi, vj);
    const vnnr = dot(vr, n);
    const vn = scl(vnnr, n);
    const vt: V3 = [vr[0] - vn[0], vr[1] - vn[1], vr[2] - vn[2]];
    const fdiss: V3 = [0, 0, 0];
    const { cn, ct, mu } = this.P;
    if (damping) for (let k = 0; k < 3; k++) fdiss[k] = -w * (cn * vn[k] + ct * vt[k]);
    const vtmag = len(vt);
    if (friction && this.hist) {
      if (fne > 0.0) accumulateContact(this.hist, ib.idx === this.histI ? 1 : -1, fne, pc, vt, n);
      if (!damping) return;
    } else if (friction && fne > 0.0 && vtmag > 0.0) {
      const scale = Math.min(mu * fne, ct * vtmag) / vtmag;
      for (let k = 0; k < 3; k++) fdiss[k] -= scale * vt[k];
    }
    this.addForce(ib.idx, fdiss);
    this.addTorque(ib.idx, ib.c, pc, fdiss);
    const mf = neg(fdiss);
    this.addForce(jb.idx, mf);
    this.addTorque(jb.idx, jb.c, pc, mf);
  }

  /** contact_point (native): halfway between the surface points on the line through pi and pj. */
  private contactPoint(pi: V3, pj: V3, n: V3, rradi: number, rradj: number): V3 {
    return [
      0.5 * ((pi[0] - rradi * n[0]) + (pj[0] + rradj * n[0])),
      0.5 * ((pi[1] - rradi * n[1]) + (pj[1] + rradj * n[1])),
      0.5 * ((pi[2] - rradi * n[2]) + (pj[2] + rradj * n[2])),
    ];
  }

  /** kernel_force (native): the normal force and energy at the separation R of the pair. */
  private kernel(R: number): { fpair: number; energy: number } {
    const { fe, fc, energy } = normalForce(R, this.P.kn, this.P.kna, this.P.cut);
    return { fpair: fe + fc, energy };
  }

  /** pair_force_and_torque (native): the cohesive force of a pair of points beyond the contact distance (R > 0). */
  private pairForceTorque(ib: PBody, jb: PBody, pi: V3, pj: V3, r: number, cdist: number, w: number): void {
    const del = sub(pi, pj);
    const R = r - cdist;
    const { fpair, energy: e } = this.kernel(R);
    const fx: V3 = [w * del[0] * fpair / r, w * del[1] * fpair / r, w * del[2] * fpair / r];
    if (R > 0) {
      this.energy += w * e;
      this.addForce(ib.idx, fx);
      this.addTorque(ib.idx, ib.c, pi, fx);
      this.addForce(jb.idx, neg(fx));
      this.addTorque(jb.idx, jb.c, pj, neg(fx));
    }
  }

  // ---- spheres -----------------------------------------------------------------------------------------

  /** sphere_against_sphere (native): both bodies are single sub-particles, friction acts on the pair directly. */
  private sphereSphere(ib: PBody, jb: PBody): void {
    const cdist = ib.rrad + jb.rrad;
    const del = sub(ib.c, jb.c);
    const rij = len(del);
    const R = rij - cdist;
    const { fpair, energy } = this.kernel(R);
    const fx: V3 = [del[0] * fpair / rij, del[1] * fpair / rij, del[2] * fpair / rij];
    this.addForce(ib.idx, fx);
    this.addForce(jb.idx, neg(fx));
    this.energy += energy;
    if (R <= 0) {
      const n: V3 = [del[0] / rij, del[1] / rij, del[2] / rij];
      const pc = this.contactPoint(ib.c, jb.c, n, ib.rrad, jb.rrad);
      this.dampingFriction(ib, jb, pc, n, -this.P.kn * R, true, true, 1.0);
    }
  }

  /** sphere_against_polyhedron (native): the point of the polyhedron b nearest to the sphere s. */
  private sphereAgainstPolyhedron(b: PBody, s: PBody): void {
    const cdist = b.rrad + s.rrad;
    if (b.faces.length > 0) {
      const nearest = this.nearestFace(b, s.c);
      if (nearest.s <= 0.0) {
        const sd = Math.min(nearest.s, -EPSILON * EPSILON * cdist);
        const h = sub(s.c, scl(sd, nearest.n));
        this.spherePointContact(b, s, h, true);
        return;
      }
    }
    let dmin = -1.0;
    let hmin: V3 = [0, 0, 0];
    for (let nf = 0; nf < b.faces.length; nf++) {
      const fv = b.faces[nf];
      const xi1 = add(b.c, b.disc[fv[0]]), xi2 = add(b.c, b.disc[fv[1]]), xi3 = add(b.c, b.disc[fv[2]]);
      const n = unit(cross(sub(xi2, xi1), sub(xi3, xi1)));
      if (!oppositeSides(n, xi1, b.c, s.c)) continue;
      const pt = projectTriangle(s.c, xi1, xi2, xi3);
      const inside = this.insidePolygon(b, nf, b.c, pt.qp, null).in1;
      if (!inside) continue;
      if (dmin < 0.0 || pt.d < dmin) {
        dmin = pt.d;
        hmin = pt.qp;
      }
    }
    for (let ne = 0; ne < b.edges.length; ne++) {
      const xi1 = add(b.c, b.disc[b.edges[ne][0]]), xi2 = add(b.c, b.disc[b.edges[ne][1]]);
      const pl = projectLine(s.c, xi1, xi2);
      let h = pl.h, d = pl.d;
      if (pl.t < 0.0) {
        h = xi1;
        d = len(sub(s.c, xi1));
      } else if (pl.t > 1.0) {
        h = xi2;
        d = len(sub(s.c, xi2));
      }
      if (dmin < 0.0 || d < dmin) {
        dmin = d;
        hmin = h;
      }
    }
    if (dmin < 0.0 || dmin > cdist + this.P.cut) return;
    this.spherePointContact(b, s, hmin, false);
  }

  /** sphere_point_contact (native): the force between the point h of body b and the sphere s; contacts are recorded. */
  private spherePointContact(b: PBody, s: PBody, h: V3, inside: boolean): void {
    const del = sub(h, s.c);
    const rsq = dot(del, del);
    if (rsq === 0.0) return;
    const cdist = b.rrad + s.rrad;
    const rij = inside ? -Math.sqrt(rsq) : Math.sqrt(rsq);
    const R = rij - cdist;
    const { fpair, energy } = this.kernel(R);
    const fx: V3 = [del[0] * fpair / rij, del[1] * fpair / rij, del[2] * fpair / rij];
    if (R <= 0) {
      const nrm: V3 = [del[0] / rij, del[1] / rij, del[2] / rij];
      const pc = this.contactPoint(h, s.c, nrm, b.rrad, s.rrad);
      this.dampingFriction(b, s, pc, nrm, 0.0, true, false, 1.0);
      this.contacts.push({ ib: b, jb: s, xi: h, xj: s.c, separation: R, r: rij, unique: true, w: 1.0, patch: false });
    }
    this.addForce(b.idx, fx);
    this.addTorque(b.idx, b.c, h, fx);
    this.addForce(s.idx, neg(fx));
    this.energy += energy;
  }

  // ---- reach, cones and weights ---------------------------------------------------------------------------

  private vertexNear(b: PBody, ni: number): boolean {
    const reachMin = b === this.bi ? this.reachMinI : this.reachMinJ;
    return b.reach[ni] >= reachMin;
  }

  private edgeNear(b: PBody, ne: number): boolean {
    return this.vertexNear(b, b.edges[ne][0]) || this.vertexNear(b, b.edges[ne][1]);
  }

  private faceNear(b: PBody, nf: number): boolean {
    for (const np of b.faces[nf]) if (this.vertexNear(b, np)) return true;
    return false;
  }

  /** vertex_cone (native): the direction d from vertex nv is in its normal cone (no incident edge is closer). */
  private vertexCone(b: PBody, nv: number, d: V3, ul: V3 | null = null): boolean {
    const dlen = len(d);
    for (const [na, nb] of b.edges) {
      let nw: number;
      if (na === nv) nw = nb;
      else if (nb === nv) nw = na;
      else continue;
      const e = sub(b.disc[nw], b.disc[nv]);
      const elen = len(e);
      if (ul && Math.abs(dot(e, ul)) > COS_LINE_ANGLE * elen * len(ul)) continue;
      if (dot(d, e) > CONE_TOL * dlen * elen) return false;
    }
    return true;
  }

  /** edge_cone (native): the direction d from edge ne is in its normal cone (the faces next to the edge). */
  private edgeCone(b: PBody, ne: number, d: V3): boolean {
    const [na, nb] = b.edges[ne];
    const dlen = len(d);
    const u = sub(b.disc[nb], b.disc[na]);
    const uu = dot(u, u);
    if (uu === 0.0) return true;
    for (const fv of b.faces) {
      const hasa = fv.includes(na), hasb = fv.includes(nb);
      if (!hasa || !hasb) continue;
      let xc: V3 = [0, 0, 0];
      for (const np of fv) xc = add(xc, b.disc[np]);
      xc = scl(1.0 / fv.length, xc);
      const tf0 = sub(xc, b.disc[na]);
      const s = dot(tf0, u) / uu;
      const tf: V3 = [tf0[0] - s * u[0], tf0[1] - s * u[1], tf0[2] - s * u[2]];
      if (dot(d, tf) > CONE_TOL * dlen * len(tf)) return false;
    }
    return true;
  }

  /** face_has (native): 1 if face nf of body b contains vertex nv (if nv >= 0) or edge ne (if ne >= 0). */
  private faceHas(b: PBody, nf: number, nv: number, ne: number): boolean {
    let na = nv, nb = nv;
    if (ne >= 0) [na, nb] = b.edges[ne];
    const fv = b.faces[nf];
    return fv.includes(na) && fv.includes(nb);
  }

  /** feature_patch_factor (native): the largest factor of a region of overlap that contains the feature. */
  private featurePatchFactor(ib: PBody, iv: number, ie: number, jb: PBody, jv: number, je: number, jf: number): number {
    if (!this.patchAny || this.patch.length === 0) return 0.0;
    const nfj = this.bj.faces.length;
    let smax = 0.0;
    for (let mf = 0; mf < ib.faces.length; mf++) {
      if (!this.faceHas(ib, mf, iv, ie)) continue;
      for (let nf = 0; nf < jb.faces.length; nf++) {
        if (jf >= 0) {
          if (nf !== jf) continue;
        } else if (!this.faceHas(jb, nf, jv, je)) continue;
        const sp = ib === this.bi ? this.patch[mf * nfj + nf] : this.patch[nf * nfj + mf];
        smax = Math.max(smax, sp);
      }
    }
    return smax;
  }

  /** vertex_face_weight (native): the weight of the vertex nv of body b against the face nf of body o. */
  private vertexFaceWeight(b: PBody, nv: number, o: PBody, nf: number): number {
    return 1.0 - this.featurePatchFactor(b, nv, -1, o, -1, -1, nf);
  }

  /** edge_edge_weight (native): the weight of the crossing of edge ei of body b and edge ej of body o. */
  private edgeEdgeWeight(b: PBody, ei: number, o: PBody, ej: number): number {
    const ui = sub(b.disc[b.edges[ei][1]], b.disc[b.edges[ei][0]]);
    const uj = sub(o.disc[o.edges[ej][1]], o.disc[o.edges[ej][0]]);
    const c = Math.abs(dot(ui, uj)) / (len(ui) * len(uj));
    const wline = c <= COS_LINE_ANGLE ? 1.0 : smoothstep(Math.acos(Math.min(1.0, c)) / LINE_ANGLE);
    return (1.0 - this.featurePatchFactor(b, -1, ei, o, -1, ej, -1)) * wline;
  }

  /** edge_edge_nearest (native): the nearest points of two edges, whether they interact, and whether they crossed. */
  private edgeEdgeNearest(ib: PBody, ei: number, jb: PBody, ej: number):
    { ok: boolean; hi: V3; hj: V3; r: number; crossed: boolean } {
    const xi1 = add(ib.c, ib.disc[ib.edges[ei][0]]), xi2 = add(ib.c, ib.disc[ib.edges[ei][1]]);
    const xj1 = add(jb.c, jb.disc[jb.edges[ej][0]]), xj2 = add(jb.c, jb.disc[jb.edges[ej][1]]);
    const de = distanceEdges(xj1, xj2, xi1, xi2);
    let hj = de.h1;
    const hi = de.h2;
    let r = de.r;
    const ti = de.t2, tj = de.t1;
    const cdist = ib.rrad + jb.rrad;
    const rmin = Math.min(ib.rrad, jb.rrad);
    if (ti < 0 || ti > 1 || tj < 0 || tj > 1 || r >= cdist + this.P.cut) return { ok: false, hi, hj, r, crossed: false };
    const ui = sub(xi2, xi1), uj = sub(xj2, xj1);
    let n = cross(ui, uj);
    const d = sub(hj, hi);
    const nlen = len(n);
    if (nlen <= PARALLEL_TOL * Math.sqrt(dot(ui, ui) * dot(uj, uj))) {
      if (r < EPSILON * rmin) return { ok: false, hi, hj, r, crossed: false };
      const nc = this.nearestFace(ib, hj).s;
      return { ok: true, hi, hj, r, crossed: nc < 0.0 };
    }
    n = scl(1.0 / nlen, n);
    let score = 0.0;
    for (let mf = 0; mf < ib.faces.length; mf++) {
      if (!this.faceHas(ib, mf, -1, ei)) continue;
      score += dot(n, ib.fnorm[mf]);
    }
    for (let mf = 0; mf < jb.faces.length; mf++) {
      if (!this.faceHas(jb, mf, -1, ej)) continue;
      score -= dot(n, jb.fnorm[mf]);
    }
    if (score === 0.0) score = dot(n, sub(jb.c, ib.c));
    if (score < 0.0) n = neg(n);
    let sep = dot(d, n);
    if (sep === 0.0) sep = EPSILON * EPSILON * rmin;
    hj = [hi[0] + sep * n[0], hi[1] + sep * n[1], hi[2] + sep * n[2]];
    r = Math.abs(sep);
    return { ok: true, hi, hj, r, crossed: sep < 0.0 };
  }

  /** vertex_edges_weight (native): the largest edge-edge weight of the edges at vertex ni (or at vertex nj). */
  private vertexEdgesWeight(ib: PBody, ni: number, jb: PBody, ej: number, nj: number): number {
    let wmax = 0.0;
    for (let ei = 0; ei < ib.edges.length; ei++) {
      if (ib.edges[ei][0] !== ni && ib.edges[ei][1] !== ni) continue;
      for (let e = 0; e < jb.edges.length; e++) {
        if (ej >= 0) {
          if (e !== ej) continue;
        } else if (jb.edges[e][0] !== nj && jb.edges[e][1] !== nj) continue;
        const w = this.edgeEdgeWeight(ib, ei, jb, e);
        if (w > wmax && this.edgeEdgeNearest(ib, ei, jb, e).ok) {
          wmax = w;
          if (wmax >= 1.0) return 1.0;
        }
      }
    }
    return wmax;
  }

  // ---- nearest face and the core ----------------------------------------------------------------------

  /** nearest_face (native): the face of b whose plane has the largest signed distance to q, its normal and index. */
  private nearestFace(b: PBody, q: V3): { s: number; n: V3; nf: number } {
    let smax = 0.0;
    let n: V3 = [0, 0, 0];
    let nf = -1;
    for (let k = 0; k < b.fplane.length; k++) {
      const pl = b.fplane[k];
      const s = dot(sub(q, pl.p), pl.n);
      if (k === 0 || s > smax) {
        smax = s;
        n = pl.n;
        nf = k;
      }
    }
    return { s: smax, n, nf };
  }

  /** inside_polygon (native): whether q1 (and q2) are inside the face fi of body b, by the angle sum about the point. */
  private insidePolygon(b: PBody, fi: number, xmi: V3, q1: V3, q2: V3 | null): { in1: boolean; in2: boolean } {
    const fv = b.faces[fi];
    const nv = fv.length;
    const rradsq = b.rrad * b.rrad;
    let anglesum1 = 0, anglesum2 = 0;
    let atvertex1 = false, atvertex2 = false;
    for (let i = 0; i < nv; i++) {
      const xi1 = add(xmi, b.disc[fv[i]]);
      const xi2 = add(xmi, b.disc[fv[(i + 1) % nv]]);
      let u = sub(xi1, q1), v = sub(xi2, q1);
      let magu = len(u), magv = len(v);
      if (magu * magv < EPSILON * rradsq) atvertex1 = true;
      else anglesum1 += Math.acos(Math.max(-1.0, Math.min(1.0, dot(u, v) / (magu * magv))));
      if (q2 !== null) {
        u = sub(xi1, q2);
        v = sub(xi2, q2);
        magu = len(u);
        magv = len(v);
        if (magu * magv < EPSILON * rradsq) atvertex2 = true;
        else anglesum2 += Math.acos(Math.max(-1.0, Math.min(1.0, dot(u, v) / (magu * magv))));
      }
    }
    const in1 = atvertex1 || Math.abs(anglesum1 - MY_2PI) < EPSILON;
    const in2 = q2 === null ? false : atvertex2 || Math.abs(anglesum2 - MY_2PI) < EPSILON;
    return { in1, in2 };
  }

  /** vertex_against_core (native): vertices of ib inside the core of jb, pushed out through the nearest face. */
  private vertexAgainstCore(ib: PBody, jb: PBody): void {
    if (jb.faces.length === 0) return;
    const cdist = ib.rrad + jb.rrad;
    for (let ni = 0; ni < ib.disc.length; ni++) {
      if (ib.vdone[ni] || !this.vertexNear(ib, ni)) continue;
      const xp = add(ib.c, ib.disc[ni]);
      const near = this.nearestFace(jb, xp);
      let sd = near.s;
      if (sd > 0.0) continue;
      sd = Math.min(sd, -EPSILON * EPSILON * cdist);
      const hp = sub(xp, scl(sd, near.n));
      const w = this.vertexFaceWeight(ib, ni, jb, near.nf);
      this.pairForceTorque(ib, jb, xp, hp, sd, cdist, w);
      this.contacts.push({ ib: jb, jb: ib, xi: hp, xj: xp, separation: sd - cdist, r: sd, unique: true, w, patch: false });
      ib.vdone[ni] = 1;
    }
  }

  /** vertex_against_edge (native): vertices of ib with no face interaction interact with the edges of jb. */
  private vertexAgainstEdge(ib: PBody, jb: PBody): void {
    const cdist = ib.rrad + jb.rrad;
    for (let ni = 0; ni < ib.disc.length; ni++) {
      if (ib.vdone[ni] || !this.vertexNear(ib, ni)) continue;
      const xpi = add(ib.c, ib.disc[ni]);
      if (len(sub(xpi, jb.c)) > jb.erad + jb.rrad + ib.rrad + this.P.cut) continue;
      let dmin = -1.0, wmin = 0.0;
      let hmin: V3 = [0, 0, 0];
      for (let ne = 0; ne < jb.edges.length; ne++) {
        if (!this.edgeNear(jb, ne)) continue;
        const xj1 = add(jb.c, jb.disc[jb.edges[ne][0]]), xj2 = add(jb.c, jb.disc[jb.edges[ne][1]]);
        const u = sub(xj2, xj1);
        const w = sub(xpi, xj1);
        const uu = dot(u, u);
        if (uu === 0.0) continue;
        const t = dot(w, u) / uu;
        if (t <= 0.0 || t >= 1.0) continue;
        const h: V3 = [xj1[0] + t * u[0], xj1[1] + t * u[1], xj1[2] + t * u[2]];
        const d = len(sub(xpi, h));
        if (d > cdist + this.P.cut) continue;
        const dv = sub(h, xpi);
        const dm = neg(dv);
        if (!this.vertexCone(ib, ni, dv, u) || !this.edgeCone(jb, ne, dm)) continue;
        if (dmin < 0.0 || d < dmin) {
          const we = (1.0 - this.vertexEdgesWeight(ib, ni, jb, ne, -1)) * (1.0 - this.featurePatchFactor(ib, ni, -1, jb, -1, ne, -1));
          if (we <= 0.0) continue;
          dmin = d;
          wmin = we;
          hmin = h;
        }
      }
      if (dmin <= 0.0) continue;
      if (this.nearestFace(jb, xpi).s < 0.0) continue;
      this.pairForceTorque(ib, jb, xpi, hmin, dmin, cdist, wmin);
      if (dmin <= cdist) this.contacts.push({ ib, jb, xi: xpi, xj: hmin, separation: dmin - cdist, r: dmin, unique: true, w: wmin, patch: false });
      ib.vdone[ni] = 1;
    }
  }

  /** vertex_against_vertex (native): vertices of ib with no interaction interact with the nearest free vertex of jb. */
  private vertexAgainstVertex(ib: PBody, jb: PBody): void {
    const cdist = ib.rrad + jb.rrad;
    for (let ni = 0; ni < ib.disc.length; ni++) {
      if (ib.vdone[ni] || !this.vertexNear(ib, ni)) continue;
      const xpi = add(ib.c, ib.disc[ni]);
      let dmin = -1.0, wmin = 0.0;
      let nmin = -1;
      let xmin: V3 = [0, 0, 0];
      for (let nj = 0; nj < jb.disc.length; nj++) {
        if (jb.vdone[nj] || !this.vertexNear(jb, nj)) continue;
        const xpj = add(jb.c, jb.disc[nj]);
        const d = len(sub(xpj, xpi));
        if (d > cdist + this.P.cut) continue;
        if (dmin < 0.0 || d < dmin) {
          const dv = sub(xpj, xpi);
          const dm = neg(dv);
          if ((!this.vertexCone(ib, ni, dv) || !this.vertexCone(jb, nj, dm)) &&
              this.nearestFace(jb, xpi).s >= 0.0 && this.nearestFace(ib, xpj).s >= 0.0) continue;
          const we = (1.0 - this.vertexEdgesWeight(ib, ni, jb, -1, nj)) * (1.0 - this.featurePatchFactor(ib, ni, -1, jb, nj, -1, -1));
          if (we <= 0.0) continue;
          dmin = d;
          wmin = we;
          nmin = nj;
          xmin = xpj;
        }
      }
      if (nmin < 0 || dmin <= 0.0) continue;
      this.pairForceTorque(ib, jb, xpi, xmin, dmin, cdist, wmin);
      if (dmin <= cdist) this.contacts.push({ ib, jb, xi: xpi, xj: xmin, separation: dmin - cdist, r: dmin, unique: true, w: wmin, patch: false });
      ib.vdone[ni] = 1;
      jb.vdone[nmin] = 1;
    }
  }

  /** edge_against_edge (native): every near edge of ib against every near edge of jb. */
  private edgeAgainstEdge(ib: PBody, jb: PBody): void {
    for (let ni = 0; ni < ib.edges.length; ni++) {
      if (!this.edgeNear(ib, ni)) continue;
      for (let nj = 0; nj < jb.edges.length; nj++) {
        if (!this.edgeNear(jb, nj)) continue;
        this.interactionEdgeToEdge(ib, ni, jb, nj);
      }
    }
  }

  /** edge_against_face (native): every near edge of ib against every near face of jb. */
  private edgeAgainstFace(ib: PBody, jb: PBody): void {
    for (let ni = 0; ni < ib.edges.length; ni++) {
      if (!this.edgeNear(ib, ni)) continue;
      for (let nj = 0; nj < jb.faces.length; nj++) {
        if (!this.faceNear(jb, nj)) continue;
        this.interactionFaceToEdge(jb, nj, ib, ni);
      }
    }
  }

  /** interaction_edge_to_edge (native). */
  private interactionEdgeToEdge(ib: PBody, ei: number, jb: PBody, ej: number): void {
    const res = this.edgeEdgeNearest(ib, ei, jb, ej);
    if (!res.ok) return;
    let r = res.r;
    if (res.crossed) r = -r;
    const w = this.edgeEdgeWeight(ib, ei, jb, ej);
    if (w <= 0.0) return;
    const cdist = ib.rrad + jb.rrad;
    this.pairForceTorque(jb, ib, res.hj, res.hi, r, cdist, w);
    if (r <= cdist) this.contacts.push({ ib, jb, xi: res.hi, xj: res.hj, separation: r - cdist, r, unique: true, w, patch: false });
  }

  /** interaction_face_to_edge (native): the edge eb of body eb against the face nf of body fb. */
  private interactionFaceToEdge(fb: PBody, nf: number, eb: PBody, ne: number): void {
    const fv = fb.faces[nf];
    const xi1 = add(fb.c, fb.disc[fv[0]]), xi2 = add(fb.c, fb.disc[fv[1]]), xi3 = add(fb.c, fb.disc[fv[2]]);
    let n = unit(cross(sub(xi2, xi1), sub(xi3, xi1)));
    const xc: V3 = [(xi1[0] + xi2[0] + xi3[0]) / 3.0, (xi1[1] + xi2[1] + xi3[1]) / 3.0, (xi1[2] + xi2[2] + xi3[2]) / 3.0];
    if (dot(sub(xc, fb.c), n) < 0) n = neg(n);
    const [a, b] = eb.edges[ne];
    const xpj1 = add(eb.c, eb.disc[a]), xpj2 = add(eb.c, eb.disc[b]);
    const cdist = fb.rrad + eb.rrad;
    const cut = cdist + this.P.cut;
    const d1 = dot(sub(xpj1, xi1), n);
    const d2 = dot(sub(xpj2, xi1), n);
    let in1 = d1 > 0.0 && d1 <= cut && eb.vdone[a] === 0;
    let in2 = d2 > 0.0 && d2 <= cut && eb.vdone[b] === 0;
    if (!in1 && !in2) return;
    const hi1: V3 = [xpj1[0] - d1 * n[0], xpj1[1] - d1 * n[1], xpj1[2] - d1 * n[2]];
    const hi2: V3 = [xpj2[0] - d2 * n[0], xpj2[1] - d2 * n[1], xpj2[2] - d2 * n[2]];
    const ins = this.insidePolygon(fb, nf, fb.c, hi1, hi2);
    const stol = EPSILON * EPSILON * cdist;
    if (in1 && ins.in1 && d1 < this.nearestFace(fb, xpj1).s - stol) in1 = false;
    if (in2 && ins.in2 && d2 < this.nearestFace(fb, xpj2).s - stol) in2 = false;
    for (let k = 0; k < 2; k++) {
      if (!(k ? in2 && ins.in2 : in1 && ins.in1)) continue;
      const npj = k ? b : a;
      const xpj = k ? xpj2 : xpj1;
      const hi = k ? hi2 : hi1;
      const d = k ? d2 : d1;
      const w = this.vertexFaceWeight(eb, npj, fb, nf);
      this.pairForceTorque(eb, fb, xpj, hi, d, cdist, w);
      if (d <= cdist) this.contacts.push({ ib: fb, jb: eb, xi: hi, xj: xpj, separation: d - cdist, r: d, unique: true, w, patch: false });
      eb.vdone[npj] = 1;
    }
  }

  // ---- regions of overlap of nearly parallel faces ----------------------------------------------------

  /** face_face_patches (native): the corners of the regions of overlap of nearly parallel faces of ib and jb. */
  private faceFacePatches(ib: PBody, jb: PBody): void {
    const cdist = ib.rrad + jb.rrad;
    const tol = EPSILON * EPSILON * EPSILON * Math.max(ib.erad, jb.erad);
    const cut = this.P.cut;
    const nfi = ib.faces.length, nfj = jb.faces.length;
    this.patchAny = false;
    const nearJ: number[] = new Array(nfj).fill(-1);
    for (let fi = 0; fi < nfi; fi++) {
      const ni = ib.fnorm[fi];
      let nearI = -1;
      for (let fj = 0; fj < nfj; fj++) {
        const nj = jb.fnorm[fj];
        if (-dot(ni, nj) <= COS_PATCH_ANGLE) continue;
        let sp = patchFactor(ni, nj);
        if (sp <= 0.0) continue;
        if (nearI < 0) nearI = this.faceNear(ib, fi) ? 1 : 0;
        if (!nearI) break;
        if (nearJ[fj] < 0) nearJ[fj] = this.faceNear(jb, fj) ? 1 : 0;
        if (!nearJ[fj]) continue;
        const nm = unit(sub(ni, nj));
        const ax: V3 = Math.abs(nm[0]) > 0.9 ? [0.0, 1.0, 0.0] : [1.0, 0.0, 0.0];
        const e1 = unit(cross(nm, ax));
        const e2 = cross(nm, e1);
        const nvi = ib.faces[fi].length, nvj = jb.faces[fj].length;
        const vi: V3[] = [];
        const vj: V3[] = [];
        let o: V3 = [0, 0, 0];
        for (let k = 0; k < nvi; k++) {
          vi[k] = add(ib.c, ib.disc[ib.faces[fi][k]]);
          o = add(o, vi[k]);
        }
        o = scl(1.0 / nvi, o);
        for (let k = 0; k < nvj; k++) vj[k] = add(jb.c, jb.disc[jb.faces[fj][k]]);
        // the positions xa, xb are those before the reversals below, as native does
        const xa = vi[0];
        const xb = vj[0];
        const pi2: [number, number][] = [];
        for (let k = 0; k < nvi; k++) {
          const q = sub(vi[k], o);
          pi2[k] = [dot(q, e1), dot(q, e2)];
        }
        const poly: PV[] = [];
        for (let k = 0; k < nvj; k++) {
          const q = sub(vj[k], o);
          poly[k] = { u: dot(q, e1), v: dot(q, e2), type: 0, a: k, b: -1, src: 0, srci: k };
        }
        let areai = 0.0, areaj = 0.0;
        for (let k = 0; k < nvi; k++) {
          const kn = (k + 1) % nvi;
          areai += pi2[k][0] * pi2[kn][1] - pi2[kn][0] * pi2[k][1];
        }
        for (let k = 0; k < nvj; k++) {
          const kn = (k + 1) % nvj;
          areaj += poly[k].u * poly[kn].v - poly[kn].u * poly[k].v;
        }
        if (areai < 0.0) {
          for (let k = 0; k < Math.floor(nvi / 2); k++) {
            const t2 = pi2[k];
            pi2[k] = pi2[nvi - 1 - k];
            pi2[nvi - 1 - k] = t2;
            const tv = vi[k];
            vi[k] = vi[nvi - 1 - k];
            vi[nvi - 1 - k] = tv;
          }
        }
        if (areaj < 0.0) {
          for (let k = 0; k < Math.floor(nvj / 2); k++) {
            const tp = poly[k];
            poly[k] = poly[nvj - 1 - k];
            poly[nvj - 1 - k] = tp;
            const tv = vj[k];
            vj[k] = vj[nvj - 1 - k];
            vj[nvj - 1 - k] = tv;
          }
        }
        for (let k = 0; k < nvj; k++) {
          poly[k].type = 0;
          poly[k].a = k;
          poly[k].b = -1;
          poly[k].src = 0;
          poly[k].srci = k;
        }
        const cni = dot(nm, ni), cnj = dot(nm, nj);
        const g0 = dot(sub(xb, o), nj) / cnj - dot(sub(xa, o), ni) / cni;
        const g1 = dot(e1, ni) / cni - dot(e1, nj) / cnj;
        const g2 = dot(e2, ni) / cni - dot(e2, nj) / cnj;
        let pl = poly;
        for (let k = 0; k <= nvi && pl.length > 0; k++) {
          let hu: number, hv: number, h0: number;
          if (k < nvi) {
            const pa2 = pi2[k], pb2 = pi2[(k + 1) % nvi];
            hu = -(pb2[1] - pa2[1]);
            hv = pb2[0] - pa2[0];
            h0 = -(hu * pa2[0] + hv * pa2[1]);
          } else {
            hu = -g1;
            hv = -g2;
            h0 = cdist + cut - g0;
          }
          pl = clipPatch(pl, hu, hv, h0, k < nvi ? k : -1, nvi, tol);
        }
        if (pl.length < 3) continue;
        const np = pl.length;
        let gmin = BIG;
        for (let k = 0; k < np; k++) gmin = Math.min(gmin, g0 + g1 * pl[k].u + g2 * pl[k].v);
        sp *= smoothstep(1.0 + gmin / cdist);
        let width = BIG;
        for (let k = 0; k < np; k++) {
          const pk = pl[k], pn = pl[(k + 1) % np];
          const ex = pn.u - pk.u, ey = pn.v - pk.v;
          const elen = Math.sqrt(ex * ex + ey * ey);
          if (elen === 0.0) continue;
          let dmax = 0.0;
          for (let m = 0; m < np; m++) dmax = Math.max(dmax, (ex * (pl[m].v - pk.v) - ey * (pl[m].u - pk.u)) / elen);
          width = Math.min(width, dmax);
        }
        sp *= smoothstep(width / cdist);
        if (sp <= 0.0) continue;
        if (!this.patchAny) {
          this.patch = new Float64Array(nfi * nfj);
          this.patchAny = true;
        }
        this.patch[fi * nfj + fj] = sp;
        const kappa = MY_2PI / Math.max(nvi, nvj);
        for (let k = 0; k < np; k++) {
          const pp = pl[(k + np - 1) % np], pc = pl[k], pn = pl[(k + 1) % np];
          if (pc.type === 3) continue;
          const d1x = pc.u - pp.u, d1y = pc.v - pp.v;
          const d2x = pn.u - pc.u, d2y = pn.v - pc.v;
          const ext = Math.atan2(d1x * d2y - d1y * d2x, d1x * d2x + d1y * d2y);
          if (ext <= 0.0) continue;
          const w = sp * ext / kappa;
          let r: number;
          let pa: V3, pb: V3;
          if (pc.type === 0) {
            const q = sub(vj[pc.a], xa);
            r = dot(q, ni);
            pb = vj[pc.a];
            pa = [pb[0] - r * ni[0], pb[1] - r * ni[1], pb[2] - r * ni[2]];
          } else if (pc.type === 1) {
            const q = sub(vi[pc.a], xb);
            r = dot(q, nj);
            pa = vi[pc.a];
            pb = [pa[0] - r * nj[0], pa[1] - r * nj[1], pa[2] - r * nj[2]];
          } else {
            const cs = closestSegments(vj[pc.a], vj[(pc.a + 1) % nvj], vi[pc.b], vi[(pc.b + 1) % nvi]);
            pb = cs.c1;
            pa = cs.c2;
            r = cs.d;
            if (dot(sub(pb, pa), nm) < 0.0) r = -r;
          }
          if (r > cdist + cut) continue;
          if (Math.abs(r) < tol) {
            if (r === 0.0) r = EPSILON * tol;
            if (pc.type === 0) pa = [pb[0] - r * ni[0], pb[1] - r * ni[1], pb[2] - r * ni[2]];
            else if (pc.type === 1) pb = [pa[0] - r * nj[0], pa[1] - r * nj[1], pa[2] - r * nj[2]];
            else pb = [pa[0] + r * nm[0], pa[1] + r * nm[1], pa[2] + r * nm[2]];
          }
          this.pairForceTorque(ib, jb, pa, pb, r, cdist, w);
          if (r <= cdist) this.contacts.push({ ib, jb, xi: pa, xj: pb, separation: r - cdist, r, unique: true, w, patch: true });
        }
      }
    }
  }

  // ---- contacts: unique contacts, rescaling, friction -------------------------------------------------

  /** contact_separation (native): the squared distance of the midpoints of two contacts. */
  private contactSeparation(c1: Contact, c2: Contact): number {
    const m1: V3 = [0.5 * (c1.xi[0] + c1.xj[0]), 0.5 * (c1.xi[1] + c1.xj[1]), 0.5 * (c1.xi[2] + c1.xj[2])];
    const m2: V3 = [0.5 * (c2.xi[0] + c2.xj[0]), 0.5 * (c2.xi[1] + c2.xj[1]), 0.5 * (c2.xi[2] + c2.xj[2])];
    return distsq(m2, m1);
  }

  /** find_unique_contacts (native): contacts of the same point are counted once. */
  private findUniqueContacts(contacts: Contact[]): void {
    const n = contacts.length;
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!contacts[i].unique) continue;
        if (contacts[i].patch || contacts[j].patch) continue;
        const d = this.contactSeparation(contacts[i], contacts[j]);
        const rmin = Math.min(contacts[i].ib.rrad, contacts[i].jb.rrad);
        if (d < EPSILON * EPSILON * rmin * rmin) contacts[j].unique = false;
      }
    }
  }

  /** rescale_cohesive_forces (native): the forces of the unique contacts scaled by the contact area factor j_a. */
  private rescaleCohesiveForces(): void {
    const contacts = this.contacts;
    if (contacts.length > 1) this.findUniqueContacts(contacts);
    let xc: V3 = [0, 0, 0];
    let wsum = 0.0;
    for (const c of contacts) {
      if (!c.unique) continue;
      xc = [xc[0] + c.w * c.xi[0], xc[1] + c.w * c.xi[1], xc[2] + c.w * c.xi[2]];
      wsum += c.w;
    }
    if (wsum <= 0.0) return;
    xc = scl(1.0 / wsum, xc);
    let area = 0.0;
    for (const c of contacts) {
      if (!c.unique) continue;
      area += c.w * distsq(c.xi, xc);
    }
    area *= Math.PI / wsum;
    let ja = area / (wsum * this.P.Aua);
    if (ja < 1.0) ja = 1.0;
    const { kn, kna, cut } = this.P;
    for (const c of contacts) {
      if (!c.unique) continue;
      const w = c.w;
      const del = sub(c.xi, c.xj);
      const r = c.r;
      const R = c.separation;
      const { fe, fc, energy } = normalForce(R, kn, kna, cut);
      this.energy += w * energy;
      const fpair = w * (fe + ja * fc);
      const fx: V3 = [del[0] * fpair / r, del[1] * fpair / r, del[2] * fpair / r];
      this.addForce(c.ib.idx, fx);
      this.addTorque(c.ib.idx, c.ib.c, c.xi, fx);
      this.addForce(c.jb.idx, neg(fx));
      this.addTorque(c.jb.idx, c.jb.c, c.xj, neg(fx));
      if (r !== 0.0) {
        const n: V3 = [del[0] / r, del[1] / r, del[2] / r];
        const pc = this.contactPoint(c.xi, c.xj, n, c.ib.rrad, c.jb.rrad);
        this.dampingFriction(c.ib, c.jb, pc, n, 0.0, true, false, w);
      }
    }
  }

  /** history_friction (native): the friction of the pair from its contact history and tangential spring; returns the new shear. */
  applyHistory(): V3 | null {
    const h = this.hist, bodies = this.histBodies;
    if (!h || !bodies) return null;
    const P = this.P;
    const hf = historyFriction(h, P.shear ?? [0, 0, 0], P.kt ?? 0, P.mu, P.dt ?? 0, P.update !== false);
    if (!hf) return null;
    const [bi, bj] = bodies;
    this.addForce(bi.idx, hf.ft);
    this.addTorque(bi.idx, bi.c, hf.pc, hf.ft);
    const mf = neg(hf.ft);
    this.addForce(bj.idx, mf);
    this.addTorque(bj.idx, bj.c, hf.pc, mf);
    return hf.shear;
  }

  /**
   * friction_forces (native): without the contact history the friction at the unique contact with the largest overlap;
   * with it, the friction of every unique contact (added to the history of the pair).
   */
  private frictionForces(): void {
    const contacts = this.contacts;
    if (contacts.length === 0) return;
    if (this.hist) {
      for (const c of contacts) if (c.unique) this.frictionForce(c);
      return;
    }
    let mmax = -1;
    let fmax = 0.0;
    for (let m = 0; m < contacts.length; m++) {
      if (!contacts[m].unique) continue;
      const fm = -contacts[m].separation * contacts[m].w;
      if (mmax < 0 || fm > fmax) {
        mmax = m;
        fmax = fm;
      }
    }
    if (mmax >= 0) this.frictionForce(contacts[mmax]);
  }

  /** friction_force (native): magnitude mu fne, opposite to the tangential velocity, at the contact point. */
  private frictionForce(c: Contact): void {
    if (c.separation >= 0.0 || c.r === 0.0) return;
    const n = scl(1.0 / c.r, sub(c.xi, c.xj));
    const pc = this.contactPoint(c.xi, c.xj, n, c.ib.rrad, c.jb.rrad);
    const fne = -this.P.kn * c.separation * c.w;
    this.dampingFriction(c.ib, c.jb, pc, n, fne, false, true, 1.0);
  }
}

/** Radius of an atom (native atom->radius): the rounded radius for one sub-particle, else enclosing plus rounded. */
export const polyAtomRadius = (b: PBody): number => (b.disc.length === 1 ? b.rrad : b.erad + b.rrad);
