/*
 * Contract of the triangle body frame (atom_style tri). The derivation is LAMMPS-specific and lives in
 * src/engine-gpl/tri/tri_bonus.ts; the core reaches it through the lazy HOOKS entry 'tri' (lazyStyles.ts group
 * gpl_tri; styles.ts triBodyHook). The core keeps only the data layout: the per-atom centroid, orientation
 * quaternion, principal moments and corner displacements that the engine stores (atoms.ts TRI_BONUS).
 *
 * docs.lammps.org/read_data.html, Triangles section: "The 3 corner points are the corner points of the triangle.
 * They should be values close to the center point of the triangle specified in the Atoms section of the data file".
 */
export type V3 = [number, number, number];

/** The body frame of one triangle, from its corner points and density (read_data.html: density is per unit area). */
export interface TriBody {
  /** Area of the triangle (half the length of the cross product of two edges). */
  area: number;
  /** Mass = density * area. */
  mass: number;
  /** Centre of the three corner points: the atom's position. */
  centroid: V3;
  /** Largest distance from the centroid to a corner. */
  radius: number;
  /** Unit quaternion (w, i, j, k) of the principal frame: space = R(quat) body. */
  quat: [number, number, number, number];
  /** Principal moments of inertia. */
  moments: V3;
  /** Corner displacements from the centroid in the principal frame (9 values, corner-major). */
  corners: Float64Array;
}

/** The hook registered as HOOKS 'tri' (src/engine-gpl/registry/tri_hook.ts). */
export interface TriBonusHook {
  triBody(corners: [V3, V3, V3], density: number): TriBody;
}
