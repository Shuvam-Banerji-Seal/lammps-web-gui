import { StyleError } from './force/types';
import type { SimState } from './types';

/*
 * atom_style body nparticle: per-body storage shared by the atom code (read_data, the atom arrays), the
 * body pair style and fix nve/body.
 *
 * Howto_body.rst (nparticle): "The 6 moments of inertia (ixx,iyy,izz,ixy,ixz,iyz) should be the values
 * consistent with the current orientation of the rigid body around its center of mass. The values are with
 * respect to the simulation box XYZ axes, not with respect to the principal axes of the rigid body itself.
 * LAMMPS performs the latter calculation internally." and "The coordinates of each sub-particle are specified
 * as its x,y,z displacement from the center-of-mass of the body particle."
 *
 * The inertia tensor is the symmetric matrix [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]. Measured with
 * native LAMMPS (black box): a body with ixy 0.5 and angular momentum (1,1,0) has the kinetic energy 0.5 L.I^-1.L of
 * that matrix, not of ixy = -0.5 (temp/body with the rotational dof gives the same numbers). Its eigen-decomposition
 * gives the principal moments and the orientation quaternion of the principal axes. The engine keeps every body's
 * sub-particle displacements in the body frame (d_body = E^T d_space), so that the space displacement is R d_body
 * at every step.
 */

/** Name of the per-atom body record in SimState.custom (the '#' keeps it out of user property names). */
export const BODY_RECORD = '#body';

/**
 * Record layout per atom. nparticle, stride 4 + 3 Nmax: [nsub, I1, I2, I3, then per sub-particle k the body-frame
 * displacement (dx, dy, dz) at 4 + 3k]. rounded/polygon, stride 5 + 3 Nmax: [nvert, I1, I2, I3, diameter, then per
 * vertex k the body-frame displacement (dx, dy, dz) at 5 + 3k]. Point particles (bodyflag 0) have a zero count.
 */
export const bodyRecordStride = (nmax: number, bstyle: string = 'nparticle'): number =>
  bstyle === 'rounded/polyhedron' ? polyRecordLength(nmax, 3 * nmax, 2 * nmax) : bodyRecordOffset(bstyle) + 3 * nmax;

/** Index of the first sub-particle (vertex) displacement in the record of a body style. */
export const bodyRecordOffset = (bstyle: string): number => (bstyle === 'rounded/polygon' ? 5 : bstyle === 'rounded/polyhedron' ? POLY_OFF : 4);

/**
 * atom_style body rounded/polyhedron record (variable length per body, as native body_rounded_polyhedron.cpp keeps it):
 * [nsub, I1, I2, I3, nedge, nface, enclosing radius, rounded radius] (POLY_OFF = 8 values; the moments at 1..3 as in
 * the other body styles, which fix nve/body reads), then the nsub vertex
 * displacements in the body frame (3 per vertex), the nedge edge ends (2 per edge, vertex indices as doubles), and
 * the nface faces (4 per face, the fourth index -1 for a triangle). nedge is 0 for one sub-particle, 1 for two (the
 * rod edge 0-1, as native), and nface is 0 below three sub-particles. Native: "3 integers: nvertices, nedges, nfaces".
 */
export const POLY_OFF = 8;
/** Number of record values of one polyhedron body with nsub vertices, nedge edges and nface faces. */
export const polyRecordLength = (nsub: number, nedge: number, nface: number): number => POLY_OFF + 3 * nsub + 2 * nedge + 4 * nface;
/** Edges and faces stored for a polyhedron body with the data-file counts (native nedges() and nfaces()). */
export const polyStoredCounts = (nsub: number, ned: number, nfac: number): { nedge: number; nface: number } =>
  nsub === 1 ? { nedge: 0, nface: 0 } : nsub === 2 ? { nedge: 1, nface: 0 } : { nedge: ned, nface: nfac };

/**
 * atom_style body nparticle Nmin Nmax | rounded/polygon Nmin Nmax: the body style and its bounds, or null for
 * other styles. Howto_body.rst: "atom_style body rounded/polygon Nmin Nmax" with "Nmin = minimum # of vertices in
 * any body in the system".
 */
export const parseBodyStyle = (st: string): { bstyle: string; nmin: number; nmax: number } | null => {
  if (!st.startsWith('body ')) return null;
  const w = st.split(/\s+/);
  return { bstyle: w[1], nmin: Number(w[2]), nmax: Number(w[3]) };
};

/** Eigen-decomposition of a symmetric 3x3 matrix (row-major, 9 entries) by cyclic Jacobi rotations. */
export const eigenSym3 = (m: ArrayLike<number>): { values: [number, number, number]; vectors: number[] } => {
  const A = Array.from(m);
  let V = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const mul = (a: number[], b: number[]): number[] => {
    const o = new Array<number>(9).fill(0);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
    return o;
  };
  const tr = (a: number[]): number[] => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
  for (let sweep = 0; sweep < 100; sweep++) {
    const off = A[1] * A[1] + A[2] * A[2] + A[5] * A[5];
    const scale = A[0] * A[0] + A[4] * A[4] + A[8] * A[8] + 2 * off;
    if (off <= 1e-34 * scale || off === 0) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = A[3 * p + q];
      if (apq === 0) continue;
      const theta = (A[3 * q + q] - A[3 * p + p]) / (2 * apq);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      const J = [1, 0, 0, 0, 1, 0, 0, 0, 1];
      J[3 * p + p] = c; J[3 * q + q] = c; J[3 * p + q] = s; J[3 * q + p] = -s;
      const JA = mul(tr(J), A);
      A.splice(0, 9, ...mul(JA, J));
      V = mul(V, J);
    }
  }
  const values: [number, number, number] = [A[0], A[4], A[8]];
  // a proper rotation: det(V) = +1 (flip the last axis otherwise)
  const det = V[0] * (V[4] * V[8] - V[5] * V[7]) - V[1] * (V[3] * V[8] - V[5] * V[6]) + V[2] * (V[3] * V[7] - V[4] * V[6]);
  if (det < 0) { V[2] = -V[2]; V[5] = -V[5]; V[8] = -V[8]; }
  return { values, vectors: V };
};

/** Unit quaternion (w, i, j, k) of a proper rotation matrix (row-major): the inverse of quatToMatrix. */
export const rotationToQuat = (R: ArrayLike<number>): [number, number, number, number] => {
  const tr = R[0] + R[4] + R[8];
  let w: number, x: number, y: number, z: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    w = (R[7] - R[5]) / s; x = 0.25 * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25 * s; z = (R[5] + R[7]) / s;
  } else {
    const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z);
  if (w < 0) return [-w / n, -x / n, -y / n, -z / n];
  return [w / n, x / n, y / n, z / n];
};

/** Principal frame of a body from its 6 inertia values (ixx iyy izz ixy ixz iyz). */
export const principalFrame = (I6: ArrayLike<number>): { moments: [number, number, number]; E: number[]; quat: [number, number, number, number] } => {
  const M = [I6[0], I6[3], I6[4], I6[3], I6[1], I6[5], I6[4], I6[5], I6[2]];
  const { values, vectors } = eigenSym3(M);
  // A rigid body's principal moments are sums of m r^2, so they are not negative. A rod (two sub-particles on one
  // line) has a zero moment about its axis, which the data file rounds to a value of order 1e-7 of either sign
  // (examples/body/data.body): only a clearly negative moment is refused.
  const big = Math.max(Math.abs(values[0]), Math.abs(values[1]), Math.abs(values[2]));
  for (const v of values) {
    if (v < -1e-6 * big) throw new StyleError('read_data Bodies: the inertia tensor of a body must not have a negative principal moment');
  }
  return { moments: values, E: vectors, quat: rotationToQuat(vectors) };
};

/**
 * Quaternion of a right-handed frame given by its axes ex, ey, ez (unit vectors, the columns of the rotation matrix),
 * by the largest-component branch of the native conversion (math_extra.cpp exyz_to_q at b332f28: the squared
 * components q0^2 .. q3^2 from the trace, the branch with the largest one is taken, then normalised). The sign is the
 * one of that branch, so it is not forced to be positive as rotationToQuat does.
 */
export const exyzToQuat = (ex: ArrayLike<number>, ey: ArrayLike<number>, ez: ArrayLike<number>): [number, number, number, number] => {
  const q0sq = 0.25 * (ex[0] + ey[1] + ez[2] + 1.0);
  const q1sq = q0sq - 0.5 * (ey[1] + ez[2]);
  const q2sq = q0sq - 0.5 * (ex[0] + ez[2]);
  const q3sq = q0sq - 0.5 * (ex[0] + ey[1]);
  const q = [0, 0, 0, 0];
  if (q0sq >= 0.25) {
    q[0] = Math.sqrt(q0sq);
    q[1] = (ey[2] - ez[1]) / (4.0 * q[0]);
    q[2] = (ez[0] - ex[2]) / (4.0 * q[0]);
    q[3] = (ex[1] - ey[0]) / (4.0 * q[0]);
  } else if (q1sq >= 0.25) {
    q[1] = Math.sqrt(q1sq);
    q[0] = (ey[2] - ez[1]) / (4.0 * q[1]);
    q[2] = (ey[0] + ex[1]) / (4.0 * q[1]);
    q[3] = (ex[2] + ez[0]) / (4.0 * q[1]);
  } else if (q2sq >= 0.25) {
    q[2] = Math.sqrt(q2sq);
    q[0] = (ez[0] - ex[2]) / (4.0 * q[2]);
    q[1] = (ey[0] + ex[1]) / (4.0 * q[2]);
    q[3] = (ez[1] + ey[2]) / (4.0 * q[2]);
  } else if (q3sq >= 0.25) {
    q[3] = Math.sqrt(q3sq);
    q[0] = (ex[1] - ey[0]) / (4.0 * q[3]);
    q[1] = (ez[0] + ex[2]) / (4.0 * q[3]);
    q[2] = (ez[1] + ey[2]) / (4.0 * q[3]);
  }
  const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
};

/**
 * Principal frame of a polyhedron body (native body_rounded_polyhedron.cpp data_body): the eigen-decomposition of the
 * 6 inertia values with the principal moments in decreasing order (native jacobi3 default), each moment below
 * 1e-7 of the largest set to zero, the axes as the columns of E (E^T maps a space displacement to the body frame), and
 * the quaternion of the right-handed frame (exyzToQuat).
 */
export const polyhedronFrame = (I6: ArrayLike<number>): { moments: [number, number, number]; E: number[]; quat: [number, number, number, number] } => {
  const { values, vectors } = eigenSym3([I6[0], I6[3], I6[4], I6[3], I6[1], I6[5], I6[4], I6[5], I6[2]]);
  const order = [0, 1, 2].sort((a, b) => values[b] - values[a]);
  const E = new Array<number>(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) E[3 * r + c] = vectors[3 * r + order[c]];
  const big = Math.max(values[0], values[1], values[2]);
  const moments = order.map((k) => (values[k] < 1e-7 * big ? 0 : values[k])) as [number, number, number];
  // enforce a right-handed frame: the reordering of the columns may flip it, and native then negates ez
  const ex = [E[0], E[3], E[6]], ey = [E[1], E[4], E[7]];
  const c3 = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
  if (c3[0] * E[2] + c3[1] * E[5] + c3[2] * E[8] < 0) { E[2] = -E[2]; E[5] = -E[5]; E[8] = -E[8]; }
  const ez = [E[2], E[5], E[8]];
  return { moments, E, quat: exyzToQuat(ex, ey, ez) };
};

/*
 * Body molecules (docs.lammps.org/molecule.html, "Body Integers" and "Body Doubles"): "The total number of values that
 * must appear is determined by the body style, and must be equal to the Ninteger value given in the *body* header."
 * and "For *rounded/polyhedron*, the number of edges E and faces F is required, and Ninteger should have a value of 3."
 * The integers and doubles are the Bodies entry of a data file, so the same record is filled (As native LAMMPS does,
 * src/molecule.cpp and src/atom.cpp add_molecule_atom at b332f28: data_body on the body class, then set_quat with the
 * molecule's quat_external, which replaces the principal-axes quaternion). The radius of the particle is the body class
 * radius_body (BODY/body_rounded_polyhedron.cpp, body_rounded_polygon.cpp, body_nparticle.cpp at b332f28), written here
 * in engine terms: the largest sub-particle distance from the centre, plus the rounded corner radius (half the rounded
 * diameter) for the rounded styles.
 */

/** Stored counts of a body: sub-particles, and edges and faces kept in the record (polyStoredCounts). */
export interface BodyCounts { nsub: number; nedge: number; nface: number }

/** Checks the integers and doubles of a body entry for a body style (native data_body error checks) and returns its counts. */
export const checkBodyParams = (bstyle: string, ints: readonly number[], dbls: readonly number[]): BodyCounts => {
  if (bstyle === 'rounded/polyhedron') {
    if (ints.length !== 3) throw new StyleError('Incorrect # of integer values in Bodies section of data file (body rounded/polyhedron has nvertices nedges nfaces)');
    const [nsub, ned, nfac] = ints;
    if (nsub < 1) throw new StyleError('Incorrect integer value in Bodies section of data file');
    const nentries = nsub < 3 ? 6 + 3 * nsub + 1 : 6 + 3 * nsub + 2 * ned + 4 * nfac + 1;
    if (dbls.length !== nentries) throw new StyleError(`Incorrect # of floating-point values in Bodies section of data file (needs ${nentries}, got ${dbls.length})`);
    const st = polyStoredCounts(nsub, ned, nfac);
    if (nsub === 2 && (dbls[6] - dbls[9]) ** 2 + (dbls[7] - dbls[10]) ** 2 + (dbls[8] - dbls[11]) ** 2 === 0) throw new StyleError('The two vertices of a rod body coincide');
    if (nsub >= 3) {
      for (let e = 0; e < st.nedge; e++) {
        const a = dbls[6 + 3 * nsub + 2 * e], b = dbls[7 + 3 * nsub + 2 * e];
        if (!(a >= 0 && a < nsub && b >= 0 && b < nsub && a !== b)) throw new StyleError(`Edge ${e} of a body has invalid vertices`);
      }
      for (let f = 0; f < st.nface; f++) {
        for (let m = 0; m < 3; m++) {
          const v = dbls[6 + 3 * nsub + 2 * ned + 4 * f + m];
          if (!(v >= 0 && v < nsub)) throw new StyleError(`Face ${f} of a body has invalid vertices`);
        }
      }
    }
    return { nsub, ...st };
  }
  if (bstyle !== 'nparticle' && bstyle !== 'rounded/polygon') throw new StyleError(`body style ${bstyle} is not supported for body molecules by the browser engine`);
  if (ints.length !== 1) throw new StyleError(`body style ${bstyle} has one integer (the number of ${bstyle === 'rounded/polygon' ? 'vertices' : 'sub-particles'})`);
  const nsub = ints[0];
  if (nsub < 1) throw new StyleError('Incorrect integer value in Bodies section of data file');
  const extra = bstyle === 'rounded/polygon' ? 1 : 0;
  if (dbls.length !== 6 + 3 * nsub + extra) throw new StyleError(`Incorrect # of floating-point values in Bodies section of data file (needs ${6 + 3 * nsub + extra}, got ${dbls.length})`);
  if (extra) {
    for (let q = 0; q < nsub; q++) {
      if (dbls[8 + 3 * q] !== 0) throw new StyleError('body rounded/polygon vertices must have z = 0 (2d bodies)');
    }
  }
  return { nsub, nedge: 0, nface: 0 };
};

/** The radius of a body particle with the given body style and integers and doubles (the molecule's radius). */
export const bodyRadiusBody = (bstyle: string, ints: readonly number[], dbls: readonly number[]): number => {
  const { nsub } = checkBodyParams(bstyle, ints, dbls);
  let maxrad = 0;
  for (let q = 0; q < nsub; q++) {
    const dx = dbls[6 + 3 * q], dy = dbls[7 + 3 * q], dz = dbls[8 + 3 * q];
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (r > maxrad) maxrad = r;
  }
  if (bstyle === 'nparticle') return maxrad;
  // the rounded diameter follows the vertices (after the 2*E + 4*F edge and face values of a polyhedron with more than two vertices)
  let off = 6 + 3 * nsub;
  if (bstyle === 'rounded/polyhedron' && nsub > 2) off += 2 * ints[1] + 4 * ints[2];
  return maxrad + 0.5 * dbls[off];
};

/**
 * The radius of body particle i of the system (the radius the body atom takes in native LAMMPS: the enclosing radius
 * plus the rounded radius, the rounded radius alone for one sub-particle; for nparticle the enclosing radius). Read from
 * the record, so it follows the body's own displacements.
 */
export const bodyAtomRadius = (s: SimState, i: number): number => {
  const bstyle = parseBodyStyle(s.atomStyle)!.bstyle;
  const rec = s.custom.get(BODY_RECORD)!;
  const base = rec.cols * i, nsub = rec.data[base];
  if (bstyle === 'rounded/polyhedron') return nsub === 1 ? rec.data[base + 6] : rec.data[base + 6] + rec.data[base + 7];
  const o0 = bodyRecordOffset(bstyle);
  let erad2 = 0;
  for (let q = 0; q < nsub; q++) {
    const o = base + o0 + 3 * q;
    erad2 = Math.max(erad2, rec.data[o] * rec.data[o] + rec.data[o + 1] * rec.data[o + 1] + rec.data[o + 2] * rec.data[o + 2]);
  }
  const erad = Math.sqrt(erad2);
  if (bstyle === 'nparticle') return erad;
  const rrad = 0.5 * rec.data[base + 4];
  return nsub === 1 ? rrad : erad + rrad;
};

/**
 * The principal axes of a body molecule's inertia in the order native LAMMPS takes them (MathEigen::jacobi3 with its
 * default decreasing sort, math_eigen_impl.h SortRows at b332f28: a selection sort that swaps each eigenvalue with the
 * first largest one after it and swaps the eigenvectors with it, so ties keep the lower index, then the right-handed
 * flip of the third axis as data_body does). A body molecule's orientation is replaced by the random quaternion of the
 * insertion (set_quat), so the axes order and signs decide the vertices in space. Only a diagonal inertia tensor is
 * accepted here: its eigenvectors are the coordinate axes (no Jacobi rotation), so no other eigenvector convention enters.
 * The example molecules (examples/body) all have diagonal tensors.
 */
export const nativeFrame = (I6: ArrayLike<number>): { moments: [number, number, number]; E: number[] } => {
  if (I6[3] !== 0 || I6[4] !== 0 || I6[5] !== 0) {
    throw new StyleError('body molecule with off-diagonal inertia (Ixy, Ixz, Iyz nonzero) is not supported by the browser engine: its principal axes need the native Jacobi eigenvectors');
  }
  const val = [I6[0], I6[1], I6[2]];
  // E holds the axes as columns (row-major); the sort swaps columns, as it swaps the eigenvector rows of native
  const E = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let i = 0; i < 2; i++) {
    let imax = i;
    for (let j = i + 1; j < 3; j++) if (val[j] > val[imax]) imax = j;
    if (imax === i) continue;
    [val[i], val[imax]] = [val[imax], val[i]];
    for (let r = 0; r < 3; r++) [E[3 * r + i], E[3 * r + imax]] = [E[3 * r + imax], E[3 * r + i]];
  }
  const big = Math.max(val[0], val[1], val[2]);
  const moments = val.map((v) => (v < 1e-7 * big ? 0 : v)) as [number, number, number];
  // enforce a right-handed frame: the third axis flips when ex x ey points against it
  const ex = [E[0], E[3], E[6]], ey = [E[1], E[4], E[7]];
  const c3 = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
  if (c3[0] * E[2] + c3[1] * E[5] + c3[2] * E[8] < 0) { E[2] = -E[2]; E[5] = -E[5]; E[8] = -E[8]; }
  return { moments, E };
};

/**
 * Fills the body record of atom i (the last atom of a body molecule insertion, or any atom with a body entry) from its
 * integers and doubles, and sets its orientation to quat (w, x, y, z), replacing the principal-axes quaternion of the
 * entry as native set_quat does. The record grows to the body's size when it does not fit the atom style's Nmax.
 */
export const setBodyAtom = (s: SimState, i: number, bstyle: string, ints: readonly number[], dbls: readonly number[], quat: readonly number[]): void => {
  if (!s.quat) throw new StyleError('body atoms need the quaternion arrays of atom_style body');
  const c = checkBodyParams(bstyle, ints, dbls);
  const rec = s.custom.get(BODY_RECORD)!;
  const need = bstyle === 'rounded/polyhedron' ? polyRecordLength(c.nsub, c.nedge, c.nface) : bodyRecordOffset(bstyle) + 3 * c.nsub;
  if (need > rec.cols) {
    const data = new Float64Array(s.n * need);
    for (let k = 0; k < s.n; k++) data.set(rec.data.subarray(rec.cols * k, rec.cols * (k + 1)), need * k);
    rec.data = data;
    rec.cols = need;
  }
  const B = rec.cols, base = B * i, nsub = c.nsub;
  if (bstyle === 'rounded/polyhedron') {
    const ned = ints[1], nfac = ints[2];
    const pf = nativeFrame(dbls.slice(0, 6));
    rec.data[base] = nsub;
    for (let d = 0; d < 3; d++) rec.data[base + 1 + d] = pf.moments[d];
    rec.data[base + 4] = c.nedge;
    rec.data[base + 5] = c.nface;
    const diam = nsub < 3 ? dbls[6 + 3 * nsub] : dbls[6 + 3 * nsub + 2 * ned + 4 * nfac];
    let erad2 = 0;
    for (let q = 0; q < nsub; q++) {
      const d0 = dbls[6 + 3 * q], d1 = dbls[7 + 3 * q], d2 = dbls[8 + 3 * q];
      erad2 = Math.max(erad2, d0 * d0 + d1 * d1 + d2 * d2);
      for (let d = 0; d < 3; d++) rec.data[base + POLY_OFF + 3 * q + d] = pf.E[d] * d0 + pf.E[3 + d] * d1 + pf.E[6 + d] * d2;
    }
    const rrad = 0.5 * diam;
    rec.data[base + 7] = rrad;
    rec.data[base + 6] = nsub === 1 ? rrad : Math.sqrt(erad2);
    const eo = base + POLY_OFF + 3 * nsub;
    if (nsub === 2) {
      rec.data[eo] = 0;
      rec.data[eo + 1] = 1;
    } else {
      for (let e = 0; e < c.nedge; e++) {
        rec.data[eo + 2 * e] = dbls[6 + 3 * nsub + 2 * e];
        rec.data[eo + 2 * e + 1] = dbls[7 + 3 * nsub + 2 * e];
      }
    }
    const fo = eo + 2 * c.nedge;
    for (let f = 0; f < c.nface; f++) {
      for (let m = 0; m < 4; m++) rec.data[fo + 4 * f + m] = dbls[6 + 3 * nsub + 2 * ned + 4 * f + m];
    }
  } else {
    const pf = nativeFrame(dbls.slice(0, 6));
    rec.data[base] = nsub;
    for (let d = 0; d < 3; d++) rec.data[base + 1 + d] = pf.moments[d];
    if (bstyle === 'rounded/polygon') rec.data[base + 4] = dbls[6 + 3 * nsub];
    const o0 = bodyRecordOffset(bstyle);
    for (let q = 0; q < nsub; q++) {
      const d0 = dbls[6 + 3 * q], d1 = dbls[7 + 3 * q], d2 = dbls[8 + 3 * q];
      for (let d = 0; d < 3; d++) rec.data[base + o0 + 3 * q + d] = pf.E[d] * d0 + pf.E[3 + d] * d1 + pf.E[6 + d] * d2;
    }
  }
  for (let d = 0; d < 4; d++) s.quat[4 * i + d] = quat[d];
};
