/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/pair_body_rounded_polygon.cpp (the
 * contact law, the contact search, the damping and friction) and src/BODY/body_rounded_polygon.cpp (the body geometry:
 * the space-frame vertices, the edge normals, the enclosing and rounded radii). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Original authors (from the sources): Trung Dac Nguyen (ndactrung@gmail.com). Reference:
 * Fraige, Langston, Matchett and Dodds, Particuology 2008, 6:455-466.
 *
 * This module holds the pure functions of the pair style: one pair of bodies in, the forces, the torques and the
 * energy out. The style itself (neighbour loop, body records) is pair_body_rounded_polygon.ts. The friction term
 * is the one without contact history; the keyword history is in contact_history.ts (see the style).
 *
 * Scope: two-dimensional bodies (z = 0). The three-component vectors keep the native arithmetic.
 */

import { accumulateContact, historyFriction, newContactHistory, type ContactHistory, type V3 } from './contact_history';

export type { V3 };

/** Constants of the native source (pair_body_rounded_polygon.cpp). */
const EPSILON = 1.0e-3; // dimensionless threshold (end points, contact separations)
const EFF_CONTACTS = 2; // effective contacts for 2d models
const BIG = 1.0e20;

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scl = (s: number, a: V3): V3 => [s * a[0], s * a[1], s * a[2]];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.sqrt(dot(a, a));
/** sum_torque (native): torque on a body with centre xm from the force F acting at x. */
export const torqueOf = (xm: V3, x: V3, F: V3): V3 => {
  const r = sub(x, xm);
  return [r[1] * F[2] - r[2] * F[1], r[2] * F[0] - r[0] * F[2], r[0] * F[1] - r[1] * F[0]];
};

/** One edge of a body's core polygon: vertex indices, outward unit normal (in the xy plane), squared length. */
export interface BodyEdge {
  a: number;
  b: number;
  en: V3;
  vsq: number;
}

/**
 * The geometry of one body in the space frame (native body2space and BodyRoundedPolygon): disc holds the vertex
 * displacements from the centre of mass, rrad the rounded radius (half the diameter), erad the enclosing radius.
 */
export interface BodyGeom {
  c: V3;
  /** Centre-of-mass velocity and angular velocity (space frame). */
  vcm: V3;
  omega: V3;
  disc: V3[];
  edges: BodyEdge[];
  rrad: number;
  erad: number;
}

/**
 * Build the space-frame geometry of a body from its body-frame vertex displacements, the rotation R (row-major 3x3)
 * and the diameter. body2space() in pair_body_rounded_polygon.cpp: the edges join vertex m to m+1 (a rod has one
 * edge, a single sub-particle none); the normal of an edge of a polygon is (v_y, -v_x) normalised, flipped when it
 * points towards the origin from the first vertex.
 */
export const bodyGeom = (c: V3, vcm: V3, omega: V3, body: V3[], R: number[], diameter: number): BodyGeom => {
  const nsub = body.length;
  const disc: V3[] = body.map((d) => [
    R[0] * d[0] + R[1] * d[1] + R[2] * d[2],
    R[3] * d[0] + R[4] * d[1] + R[5] * d[2],
    R[6] * d[0] + R[7] * d[1] + R[8] * d[2],
  ]);
  const rrad = 0.5 * diameter;
  // the enclosing radius: the largest vertex distance (a single sub-particle has the rounded radius)
  let erad2 = 0;
  for (const d of body) erad2 = Math.max(erad2, d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  const erad = nsub === 1 ? rrad : Math.sqrt(erad2);
  const edges: BodyEdge[] = [];
  const ne = nsub === 1 ? 0 : nsub === 2 ? 1 : nsub;
  for (let m = 0; m < ne; m++) {
    const a = disc[m];
    const b = disc[nsub === 2 ? 1 : (m + 1) % nsub];
    const ab = sub(b, a);
    const vsq = dot(ab, ab);
    let en: V3 = [0, 0, 0];
    if (nsub > 2 && vsq > 0) {
      en = [ab[1], -ab[0], 0];
      if (en[0] * a[0] + en[1] * a[1] < 0) en = scl(-1, en);
      const l = len(en);
      en = scl(1 / l, en);
    }
    edges.push({ a: m, b: nsub === 2 ? 1 : (m + 1) % nsub, en, vsq });
  }
  return { c, vcm, omega, disc, edges, rrad, erad };
};

/** Radius of the atom (atom->radius in native): the enclosing radius plus the rounded radius, or the rounded one for a disc. */
export const atomRadius = (g: BodyGeom): number => (g.disc.length === 1 ? g.rrad : g.erad + g.rrad);

/** Normal force at the surface separation R (native normal_force): fe elastic, fc cohesive; returns the energy. */
export const normalForce = (R: number, kn: number, kna: number, cut: number): { fe: number; fc: number; energy: number } => {
  if (R > cut) return { fe: 0, fc: 0, energy: 0 };
  const dna = cut - R;
  const fc = -kna * dna;
  let energy = -0.5 * kna * dna * dna;
  let fe = 0;
  if (R < 0) {
    fe = -kn * R;
    energy += 0.5 * kn * R * R;
  }
  return { fe, fc, energy };
};

/**
 * nearest_point (native): the point h of the core of body Y nearest to xp, the signed distance d (negative inside the
 * core), the unit normal n from h to xp and the vertex index nv of h (-1 when h is inside an edge or when only d is
 * returned for a distance above dcut, with h and n undefined).
 */
export const nearestPoint = (Y: BodyGeom, xp: V3, dcut = BIG): { d: number; h: V3; n: V3; nv: number } => {
  // the same operations as add/sub/scl/dot/distsq in the same order, on scalars (no vector per step)
  const xm = Y.c;
  const disc = Y.disc;
  const nsub = disc.length;
  const nedges = Y.edges.length;
  const px = xp[0], py = xp[1], pz = xp[2];
  const cx = xm[0], cy = xm[1], cz = xm[2];
  let nv = 0;
  let hx = cx + disc[0][0], hy = cy + disc[0][1], hz = cz + disc[0][2];
  let dx = px - hx, dy = py - hy, dz = pz - hz;
  let dmin = dx * dx + dy * dy + dz * dz;
  if (nedges === 0) {
    for (let m = 1; m < nsub; m++) {
      const qx = cx + disc[m][0], qy = cy + disc[m][1], qz = cz + disc[m][2];
      dx = px - qx; dy = py - qy; dz = pz - qz;
      const dsq = dx * dx + dy * dy + dz * dz;
      if (dsq < dmin) {
        dmin = dsq;
        nv = m;
        hx = qx; hy = qy; hz = qz;
      }
    }
  }
  // signed distances of xp from the edge lines (outward normals): inside the core if all are negative
  let smax = -BIG;
  let emax = -1;
  for (let ne = 0; ne < nedges; ne++) {
    const eg = Y.edges[ne];
    if (eg.vsq === 0) continue;
    const da = disc[eg.a], en = eg.en;
    const ux = px - (cx + da[0]), uy = py - (cy + da[1]), uz = pz - (cz + da[2]);
    const sd = ux * en[0] + uy * en[1] + uz * en[2];
    if (emax < 0 || sd > smax) {
      smax = sd;
      emax = ne;
    }
  }
  if (nsub > 2 && emax >= 0 && smax > dcut) return { d: smax, h: [0, 0, 0], n: [0, 0, 0], nv: -1 };
  if (nsub > 2 && emax >= 0 && smax < 0) {
    const en = Y.edges[emax].en;
    return { d: smax, h: [px - smax * en[0], py - smax * en[1], pz - smax * en[2]], n: en, nv: -1 };
  }
  // the nearest point over the edges that xp is in front of (non-negative signed distance)
  let first = true;
  for (let ne = 0; ne < nedges; ne++) {
    const eg = Y.edges[ne];
    const da = disc[eg.a], en = eg.en;
    const ax = cx + da[0], ay = cy + da[1], az = cz + da[2];
    const ux = px - ax, uy = py - ay, uz = pz - az;
    if (ux * en[0] + uy * en[1] + uz * en[2] < 0) continue;
    const db = disc[eg.b];
    const bx = cx + db[0], by = cy + db[1], bz = cz + db[2];
    const vx = bx - ax, vy = by - ay, vz = bz - az;
    const t = eg.vsq > 0 ? (ux * vx + uy * vy + uz * vz) / eg.vsq : 0;
    let qx: number, qy: number, qz: number;
    let nvp = -1;
    if (t <= 0) {
      qx = ax; qy = ay; qz = az;
      nvp = eg.a;
    } else if (t >= 1) {
      qx = bx; qy = by; qz = bz;
      nvp = eg.b;
    } else {
      qx = ax + t * vx; qy = ay + t * vy; qz = az + t * vz;
    }
    dx = px - qx; dy = py - qy; dz = pz - qz;
    const dsq = dx * dx + dy * dy + dz * dz;
    if (first || dsq < dmin) {
      first = false;
      dmin = dsq;
      nv = nvp;
      hx = qx; hy = qy; hz = qz;
    }
  }
  const d = Math.sqrt(dmin);
  if (d === 0) throw new Error('A vertex of a body touches the core of another body');
  const inv = 1 / d;
  return { d, h: [hx, hy, hz], n: [inv * (px - hx), inv * (py - hy), inv * (pz - hz)], nv };
};

/** One contact between a vertex of body `ib` and the core of body `jb` (roles 0 or 1 in the pair). */
export interface RPContact {
  ib: 0 | 1;
  jb: 0 | 1;
  xv: V3;
  xe: V3;
  n: V3;
  fe: V3;
  fc: V3;
  separation: number;
  energy: number;
}

/** Parameters of pair_style body/rounded/polygon: c_n, c_t, mu, delta_ua and the cutoff cut_inner. */
export interface RPParams {
  kn: number;
  kna: number;
  cn: number;
  ct: number;
  mu: number;
  dua: number;
  cut: number;
  /** keyword history: the friction is that of a tangential spring with the shear of the pair (native history). */
  history?: boolean;
  /** tangential stiffness k_t of the pair of types (native k_t) and the time step (native dt). */
  kt?: number;
  dt?: number;
  /** false in the setup of a run: the tangential velocity does not advance the shear (native setupflag). */
  update?: boolean;
  /** the shear of the pair as seen from body 0 (the stored value, zero when new). */
  shear?: V3;
}

/** The pair accumulator: forces and torques on the two bodies of the pair (role 0 = i, role 1 = j), energy, contacts. */
export interface RPAcc {
  f: [V3, V3];
  t: [V3, V3];
  energy: number;
  contacts: RPContact[];
  /** j_a applied to the cohesion of the two contacts that carry forces (1 when not scaled). */
  ja: number;
  /** the contact history of the pair (keyword history), and the new shear when the spring acts (null when not). */
  hist: ContactHistory | null;
  shear: V3 | null;
}

/** Force F on body `role` acting at the point at, with its torque about the centre of that body. */
const addForce = (acc: RPAcc, role: 0 | 1, F: V3, at: V3, bodies: [BodyGeom, BodyGeom]): void => {
  acc.f[role] = add(acc.f[role], F);
  acc.t[role] = add(acc.t[role], torqueOf(bodies[role].c, at, F));
};

/**
 * vertex_against_edge (native): the vertices of body X (role ib) against the core of body Y (role jb). A contact
 * (R < EPSILON rmin) is stored for the later contact_forces(); a cohesion-only pair (R in [EPSILON rmin, cut]) acts
 * at once. A vertex-vertex interaction is counted in the first call only (first = true), once both vertices are the
 * nearest features of their bodies to each other.
 */
export const vertexAgainstEdge = (bodies: [BodyGeom, BodyGeom], ib: 0 | 1, jb: 0 | 1, first: boolean, p: RPParams, acc: RPAcc): void => {
  const X = bodies[ib], Y = bodies[jb];
  const rmin = Math.min(X.rrad, Y.rrad);
  for (let ni = 0; ni < X.disc.length; ni++) {
    // the vertex position and its distance from Y's centre on scalars (add and distsq), the vector only when it is near
    const dv = X.disc[ni];
    const xx = X.c[0] + dv[0], xy = X.c[1] + dv[1], xz = X.c[2] + dv[2];
    const ex = xx - Y.c[0], ey = xy - Y.c[1], ez = xz - Y.c[2];
    const dist = Math.sqrt(ex * ex + ey * ey + ez * ez);
    if (dist > Y.erad + Y.rrad + X.rrad + p.cut) continue;
    const xpi: V3 = [xx, xy, xz];
    const { d, h, n, nv } = nearestPoint(Y, xpi, X.rrad + Y.rrad + p.cut);
    const R = d - (X.rrad + Y.rrad);
    if (R > p.cut) continue;
    if (nv >= 0) {
      if (!first) continue;
      if (X.disc.length > 1) {
        const nvv = nearestPoint(X, h).nv;
        if (nvv !== ni) continue;
      }
    }
    const { fe, fc, energy } = normalForce(R, p.kn, p.kna, p.cut);
    if (R < EPSILON * rmin) {
      acc.contacts.push({
        ib,
        jb,
        xv: xpi,
        xe: h,
        n,
        fe: scl(fe, n),
        fc: scl(fc, n),
        separation: R,
        energy,
      });
    } else {
      acc.energy += energy;
      const F = scl(fc, n);
      addForce(acc, ib, F, xpi, bodies);
      addForce(acc, jb, scl(-1, F), h, bodies);
    }
  }
};

/** contact_point (native): the midpoint of the two rounded surfaces along the normal n (from body jb to ib). */
export const contactPoint = (pi: V3, pj: V3, n: V3, rradi: number, rradj: number): V3 => [
  0.5 * (pi[0] - rradi * n[0] + pj[0] + rradj * n[0]),
  0.5 * (pi[1] - rradi * n[1] + pj[1] + rradj * n[1]),
  0.5 * (pi[2] - rradi * n[2] + pj[2] + rradj * n[2]),
];

/** Total velocity of the body at the point x (native total_velocity): vcm + omega x (x - c). */
const pointVelocity = (g: BodyGeom, x: V3): V3 => {
  const r = sub(x, g.c);
  const w = g.omega;
  return [w[1] * r[2] - w[2] * r[1] + g.vcm[0], w[2] * r[0] - w[0] * r[2] + g.vcm[1], w[0] * r[1] - w[1] * r[0] + g.vcm[2]];
};

/**
 * damping_friction (native): the damping -c_n v_n - c_t v_t at the contact point, and, when friction is on, the friction
 * -min(mu fne, c_t |v_t|) v_t/|v_t| with the elastic normal force fne. With the contact history (acc.hist), the friction
 * is added to the history of the pair instead, see historyFriction().
 * The forces act at the contact point pc on the two bodies (role ib receives +fd).
 */
export const dampingFriction = (bodies: [BodyGeom, BodyGeom], ib: 0 | 1, jb: 0 | 1, pc: V3, n: V3, fne: number, friction: boolean, p: RPParams, acc: RPAcc): void => {
  const vi = pointVelocity(bodies[ib], pc);
  const vj = pointVelocity(bodies[jb], pc);
  const vr = sub(vi, vj);
  const vnnr = dot(vr, n);
  const vn = scl(vnnr, n);
  const vt = sub(vr, vn);
  let fd: V3 = add(scl(-p.cn, vn), scl(-p.ct, vt));
  const vtmag = len(vt);
  if (friction && acc.hist) {
    // with the contact history, the friction of the pair is that of one tangential spring, see historyFriction()
    if (fne > 0) accumulateContact(acc.hist, ib === 0 ? 1 : -1, fne, pc, vt, n);
  } else if (friction && fne > 0 && vtmag > 0) {
    const s = Math.min(p.mu * fne, p.ct * vtmag) / vtmag;
    fd = sub(fd, scl(s, vt));
  }
  addForce(acc, ib, fd, pc, bodies);
  addForce(acc, jb, scl(-1, fd), pc, bodies);
};

/** contact_separation (native): the separation of two contacts along the tangent of the first one. */
const contactSeparation = (c1: RPContact, c2: RPContact): number =>
  Math.abs(-c1.n[1] * (c2.xv[0] - c1.xv[0]) + c1.n[0] * (c2.xv[1] - c1.xv[1]));

/** contact_forces (native, without history): the elastic force and the cohesion scaled by j_a, then damping and friction. */
const contactForces = (c: RPContact, ja: number, friction: boolean, bodies: [BodyGeom, BodyGeom], p: RPParams, acc: RPAcc): void => {
  const rI = bodies[c.ib].rrad, rJ = bodies[c.jb].rrad;
  const pc = contactPoint(c.xv, c.xe, c.n, rI, rJ);
  acc.energy += c.energy;
  const F = add(c.fe, scl(ja, c.fc));
  addForce(acc, c.ib, F, pc, bodies);
  addForce(acc, c.jb, scl(-1, F), pc, bodies);
  dampingFriction(bodies, c.ib, c.jb, pc, c.n, len(c.fe), friction, p, acc);
};

/**
 * sphere_against_sphere (native): two single sub-particles (discs). The elastic and cohesive normal force, and at
 * R <= EPSILON rmin the damping and friction at the contact point.
 */
const sphereAgainstSphere = (bodies: [BodyGeom, BodyGeom], p: RPParams, acc: RPAcc): void => {
  const [A, B] = bodies;
  const del = sub(A.c, B.c);
  const rij = Math.sqrt(dot(del, del));
  const R = rij - (A.rrad + B.rrad);
  const { fe, fc, energy } = normalForce(R, p.kn, p.kna, p.cut);
  const fpair = fe + fc;
  const F = scl(fpair / rij, del);
  // the normal force of two discs acts through the centres: no torque
  acc.f[0] = add(acc.f[0], F);
  acc.f[1] = sub(acc.f[1], F);
  acc.energy += energy;
  const rmin = Math.min(A.rrad, B.rrad);
  if (R <= EPSILON * rmin) {
    const n = scl(1 / rij, del);
    const pc = contactPoint(A.c, B.c, n, A.rrad, B.rrad);
    dampingFriction(bodies, 0, 1, pc, n, fe, true, p, acc);
  }
};

/**
 * The pair interaction of two bodies (native pair_interaction without history): the contacts of both vertex sets, the
 * two contacts that carry forces (the one of least separation, and the one farthest from it along the contact
 * segment), and j_a = max(delta / (2 delta_ua), 1) of the segment. Returns the accumulated forces, torques and energy
 * on bodies 0 and 1 (the pair's i and j).
 */
export const roundedPolygonPair = (bodies: [BodyGeom, BodyGeom], p: RPParams): RPAcc => {
  const acc: RPAcc = {
    f: [[0, 0, 0], [0, 0, 0]], t: [[0, 0, 0], [0, 0, 0]], energy: 0, contacts: [], ja: 1,
    hist: p.history ? newContactHistory() : null, shear: null,
  };
  pairContacts(bodies, p, acc);
  // the friction of the contact history acts once per pair, after all contacts (native history_friction)
  if (acc.hist && p.kt !== undefined) {
    const hf = historyFriction(acc.hist, p.shear ?? [0, 0, 0], p.kt, p.mu, p.dt ?? 0, p.update !== false);
    if (hf) {
      addForce(acc, 0, hf.ft, hf.pc, bodies);
      addForce(acc, 1, scl(-1, hf.ft), hf.pc, bodies);
      acc.shear = hf.shear;
    }
  }
  return acc;
};

/** The contacts of the pair of bodies (native pair_interaction without the friction of the history). */
const pairContacts = (bodies: [BodyGeom, BodyGeom], p: RPParams, acc: RPAcc): void => {
  const [A, B] = bodies;
  if (A.disc.length === 1 && B.disc.length === 1) {
    sphereAgainstSphere(bodies, p, acc);
    return;
  }
  vertexAgainstEdge(bodies, 0, 1, true, p, acc);
  vertexAgainstEdge(bodies, 1, 0, false, p, acc);
  const cs = acc.contacts;
  const nc = cs.length;
  if (nc === 0) return;
  let m0 = 0, n0 = -1;
  for (let m = 1; m < nc; m++) if (cs[m].separation < cs[m0].separation) m0 = m;
  const rmin = Math.min(A.rrad, B.rrad);
  let deltaMax = EPSILON * rmin;
  for (let m = 0; m < nc - 1; m++) {
    for (let n = m + 1; n < nc; n++) {
      const da = contactSeparation(cs[m], cs[n]);
      if (da > deltaMax) {
        deltaMax = da;
        m0 = m;
        n0 = n;
      }
    }
  }
  let ja = 1;
  if (n0 >= 0) ja = Math.max(deltaMax / (EFF_CONTACTS * p.dua), 1);
  acc.ja = ja;
  // the friction at the contact of largest overlap, or at both contacts with the contact history
  let frictionM = true;
  if (n0 >= 0 && cs[n0].separation < cs[m0].separation) frictionM = false;
  if (acc.hist) frictionM = true;
  const frictionN = acc.hist ? true : !frictionM;
  const c0 = cs[m0];
  contactForces(c0, ja, frictionM, bodies, p, acc);
  if (n0 >= 0) contactForces(cs[n0], ja, frictionN, bodies, p, acc);
};
