import { StyleError } from './force/types';

/*
 * General triclinic boxes — docs.lammps.org/Howto_triclinic.html.
 *
 * A general triclinic box has an origin and arbitrary edge vectors A, B, C.
 * "However internally, LAMMPS only uses restricted triclinic simulation
 * boxes." "Input of a general triclinic system is immediately converted to a
 * restricted triclinic system." "The conversion of the simulation box and
 * per-atom data from general triclinic to restricted triclinic (and vice
 * versa) is a 3d rotation operation around an origin, which is the lower left
 * corner of the simulation box."
 *
 * The restricted edge vectors are (Howto_triclinic, "Transformation from
 * general to restricted triclinic boxes"): a_x = A; b_x = B . Ahat = B cos g;
 * b_y = |Ahat x B| = B sin g; c_x = C . Ahat; c_y = (B . C - b_x c_x) / b_y;
 * c_z = |C . (A x B)^hat| — with a = (a_x, 0, 0), b = (b_x, b_y, 0) and
 * c = (c_x, c_y, c_z). The same rotation Q, written here as orthonormal rows
 * e1 = Ahat, e2 = (B - b_x Ahat) / b_y, e3 = e1 x e2, maps A, B, C onto a, b, c
 * (Q [A B C] = [a b c]) and is the rotation the page describes.
 *
 * "For consistency, the same rotation applied to the triclinic box edge
 * vectors can also be applied to atom positions, velocities, and other vector
 * quantities." Positions rotate about the box origin (kept fixed); velocities
 * rotate without an origin.
 *
 * Restricted box header (read_data.html): the xlo xhi, ylo yhi, zlo zhi and xy xz yz lines,
 * with lo = origin, hi = origin + (a_x, b_y, c_z), tilt = (b_x, c_x, c_y).
 *
 * Measured with native LAMMPS (black box): with A = (1 1 0), B = (0 1 0.5),
 * C = (0.2 0.1 1) and abc origin (0.5 0.2 0.1), the restricted header is
 * xlo xhi = 0.5 1.914213562373095, tilt xy xz yz = 0.7071067811865476
 * 0.2121320343559643 0.5196152422706631; the origin is not rotated; atom
 * (0.65 0.825 0.875) in general coordinates is at (1.5429825 0.8062178
 * 0.7429911) = origin + Q (x - origin); velocity (1 2 3) becomes
 * (2.1213203 2.3094011 2.0412415) = Q v.
 *
 * Measured with native LAMMPS (black box): create_box N NULL ... and the
 * lattice triclinic/general option reject a left-handed set with the error
 * Lattice triclinic/general a1,a2,a3 must be right-handed, and a collinear set
 * with Lattice primitive vectors are collinear.
 */

export type V3 = [number, number, number];
export type Mat3 = [V3, V3, V3];

/** Edge vectors and origin of a general triclinic box (general coordinates). */
export interface GeneralBox {
  origin: V3;
  A: V3;
  B: V3;
  C: V3;
}

/** The rotation Q (rows orthonormal, det +1) and the restricted box it gives. */
export interface GeneralFrame {
  Q: Mat3;
  /** Restricted box: lo = origin, hi = origin + (ax, by, cz); tilt = (xy, xz, yz) = (bx, cx, cy). */
  lo: V3;
  hi: V3;
  tilt: V3;
}

const dot = (p: V3, q: V3): number => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
const cross = (p: V3, q: V3): V3 => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
const norm = (p: V3): number => Math.sqrt(dot(p, p));
const scale = (p: V3, s: number): V3 => [p[0] * s, p[1] * s, p[2] * s];
const sub = (p: V3, q: V3): V3 => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
const add = (p: V3, q: V3): V3 => [p[0] + q[0], p[1] + q[1], p[2] + q[2]];

/**
 * Rotation for edge vectors A, B, C (general triclinic, 3d). Throws StyleError
 * when the vectors are zero, coplanar, or left-handed.
 */
export const rotationFromEdges = (A: V3, B: V3, C: V3): { Q: Mat3; a: number; b: number; c: [number, number, number] } => {
  const la = norm(A), lb = norm(B), lc = norm(C);
  if (!(la > 0) || !(lb > 0) || !(lc > 0)) throw new StyleError('general triclinic edge vectors A, B, C must be non-zero');
  const triple = dot(cross(A, B), C);
  if (Math.abs(triple) <= 1e-12 * la * lb * lc) {
    throw new StyleError('general triclinic edge vectors A, B, C must not be co-planar');
  }
  if (triple < 0) {
    throw new StyleError('general triclinic edge vectors A, B, C must be right-handed (A x B points along C); swap two of them');
  }
  const e1 = scale(A, 1 / la);
  const bx = dot(B, e1);
  const perpB = sub(B, scale(e1, bx));
  const by = norm(perpB);
  const e2 = scale(perpB, 1 / by);
  const e3 = cross(e1, e2);
  const cx = dot(C, e1);
  const cy = dot(C, e2);
  const cz = dot(C, e3);
  return { Q: [e1, e2, e3], a: la, b: by, c: [cx, cy, cz] };
};

/** Restricted frame (rotation and restricted box) of a general triclinic box. */
export const generalFrame = (box: GeneralBox): GeneralFrame => {
  const r = rotationFromEdges(box.A, box.B, box.C);
  const o = box.origin;
  return {
    Q: r.Q,
    lo: [...o] as V3,
    hi: [o[0] + r.a, o[1] + r.b, o[2] + r.c[2]],
    // b_x = B . Ahat is the xy tilt; c_x and c_y are the xz and yz tilts
    tilt: [dot(box.B, r.Q[0]), r.c[0], r.c[1]],
  };
};

/** Q v (no origin): velocities, and any other direction-type vector. */
export const rotateVector = (Q: Mat3, v: V3): V3 => [dot(Q[0], v), dot(Q[1], v), dot(Q[2], v)];

/** Q^T v: inverse of rotateVector. */
export const unrotateVector = (Q: Mat3, v: V3): V3 => [
  Q[0][0] * v[0] + Q[1][0] * v[1] + Q[2][0] * v[2],
  Q[0][1] * v[0] + Q[1][1] * v[1] + Q[2][1] * v[2],
  Q[0][2] * v[0] + Q[1][2] * v[1] + Q[2][2] * v[2],
];

/** General -> restricted position: origin + Q (x - origin). */
export const toRestrictedPoint = (Q: Mat3, origin: V3, x: V3): V3 => add(origin, rotateVector(Q, sub(x, origin)));

/** Restricted -> general position: origin + Q^T (x - origin). */
export const toGeneralPoint = (Q: Mat3, origin: V3, x: V3): V3 => add(origin, unrotateVector(Q, sub(x, origin)));

/**
 * Lattice sites of a general triclinic box, in restricted coordinates. The
 * lattice (general coordinates, lattice origin 0) is the one `generalLattice`
 * describes; a site is kept when its fractional position in the box
 * (general edge vectors from the box origin) lies in [0, 1), the same
 * half-open rule create_atoms uses for orthogonal boxes. Sites come in the
 * (k, j, i, basis) order of latticeSites so atom IDs match that order.
 */
export interface GeneralLattice {
  basis: V3[];
  /** Scaled primitive vectors a1, a2, a3 (lattice scale applied): the box edges are multiples of them. */
  cell: [V3, V3, V3];
  /** Rotation general -> restricted of the cell (its rows are e1, e2, e3). */
  Q: Mat3;
  /** General coordinates of lattice point (i + b0, j + b1, k + b2) in unit-cell fractions. */
  toGeneral(f: V3): V3;
  /** Inverse of toGeneral. */
  fromGeneral(x: V3): V3;
}

/**
 * The general box that a restricted box came from: the origin is unchanged by
 * the rotation, and A, B, C = Q^T (a, 0, 0), Q^T (b_x, b_y, 0), Q^T (c_x, c_y, c_z)
 * with (a, b, c) the restricted edge vectors of lo/hi/tilt.
 */
export const generalBoxFromRestricted = (Q: Mat3, lo: V3, hi: V3, tilt: V3): GeneralBox => ({
  origin: [...lo] as V3,
  A: unrotateVector(Q, [hi[0] - lo[0], 0, 0]),
  B: unrotateVector(Q, [tilt[0], hi[1] - lo[1], 0]),
  C: unrotateVector(Q, [tilt[1], tilt[2], hi[2] - lo[2]]),
});

export const generalAtomSites = (
  lat: GeneralLattice, box: GeneralBox, accept: (x: number, y: number, z: number) => boolean = () => true,
): { x: number[]; basis: number[] } => {
  const Q = lat.Q;
  const origin = box.origin;
  // inverse of [A B C] (columns) gives the box fraction of a general point
  const M: Mat3 = [
    [box.A[0], box.B[0], box.C[0]],
    [box.A[1], box.B[1], box.C[1]],
    [box.A[2], box.B[2], box.C[2]],
  ];
  const inv = invert3(M);
  const boxFrac = (p: V3): V3 => {
    const d = sub(p, origin);
    return [0, 1, 2].map((r) => inv[r][0] * d[0] + inv[r][1] * d[1] + inv[r][2] * d[2]) as V3;
  };
  // range of lattice cells covering the box corners
  const lmin: V3 = [Infinity, Infinity, Infinity], lmax: V3 = [-Infinity, -Infinity, -Infinity];
  for (const s of [0, 1]) for (const t of [0, 1]) for (const u of [0, 1]) {
    const corner = add(origin, add(add(scale(box.A, s), scale(box.B, t)), scale(box.C, u)));
    const f = lat.fromGeneral(corner);
    for (let d = 0; d < 3; d++) { lmin[d] = Math.min(lmin[d], f[d]); lmax[d] = Math.max(lmax[d], f[d]); }
  }
  const i0 = Math.floor(lmin[0]) - 1, i1 = Math.ceil(lmax[0]) + 1;
  const j0 = Math.floor(lmin[1]) - 1, j1 = Math.ceil(lmax[1]) + 1;
  const k0 = Math.floor(lmin[2]) - 1, k1 = Math.ceil(lmax[2]) + 1;
  const tol = 1e-9;
  const x: number[] = [];
  const basis: number[] = [];
  for (let k = k0; k <= k1; k++) {
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        lat.basis.forEach((b, m) => {
          const g = lat.toGeneral([i + b[0], j + b[1], k + b[2]]);
          const fr = boxFrac(g);
          if (fr.some((v) => v < -tol || v >= 1 - tol)) return;
          const r = toRestrictedPoint(Q, origin, g);
          if (accept(r[0], r[1], r[2])) { x.push(r[0], r[1], r[2]); basis.push(m); }
        });
      }
    }
  }
  return { x, basis };
};

/**
 * `create_box N NULL alo ahi blo bhi clo chi` (create_box.html): the box is
 * A = (ahi-alo) a1, B = (bhi-blo) a2, C = (chi-clo) a3, origin = alo a1 + blo a2
 * + clo a3, with a1..a3 the scaled cell of a triclinic/general lattice. Returns
 * the restricted box (lo, hi, tilt) that LAMMPS stores.
 */
export const generalCreateBox = (
  lat: { general?: GeneralLattice } | null | undefined, bounds: readonly string[],
): { lo: V3; hi: V3; tilt: V3 } => {
  if (!lat?.general) throw new StyleError('create_box N NULL needs a lattice with the triclinic/general option');
  if (bounds.length < 6) throw new StyleError('create_box N NULL needs alo ahi blo bhi clo chi');
  const v = bounds.slice(0, 6).map((w) => {
    const n = Number(w);
    if (!Number.isFinite(n)) throw new StyleError(`create_box NULL: expected a number, got '${w}'`);
    return n;
  });
  if (!(v[1] > v[0]) || !(v[3] > v[2]) || !(v[5] > v[4])) throw new StyleError('create_box NULL: each hi bound must exceed its lo bound');
  const [a1, a2, a3] = lat.general.cell;
  const o: V3 = [
    v[0] * a1[0] + v[2] * a2[0] + v[4] * a3[0],
    v[0] * a1[1] + v[2] * a2[1] + v[4] * a3[1],
    v[0] * a1[2] + v[2] * a2[2] + v[4] * a3[2],
  ];
  const gb: GeneralBox = {
    origin: o,
    A: scale(a1, v[1] - v[0]),
    B: scale(a2, v[3] - v[2]),
    C: scale(a3, v[5] - v[4]),
  };
  const f = generalFrame(gb);
  return { lo: f.lo, hi: f.hi, tilt: f.tilt };
};

const invert3 = (m: Mat3): Mat3 => {
  const [[a, b, c], [d, e, f], [g, h, k]] = m;
  const A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-300) throw new StyleError('general triclinic box is degenerate');
  const s = 1 / det;
  return [
    [A * s, -(b * k - c * h) * s, (b * f - c * e) * s],
    [B * s, (a * k - c * g) * s, -(a * f - c * d) * s],
    [C * s, -(a * h - b * g) * s, (a * e - b * d) * s],
  ];
};
