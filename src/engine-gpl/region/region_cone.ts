/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/region_cone.cpp and src/region_cone.h (RegCone: the constructor's values and extent, inside,
 * surface_interior, surface_exterior, shape_update and closest) at commit b332f28. Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Only the region-surface parts are ported: argument parsing, bbox_update, variable_check and the error messages are
 * the engine's job. The native axis char is `caxis` here (0 = 'x', 1 = 'y', 2 = 'z'), since NativeRegion.axis is the
 * rotation axis.
 */
import { NativeRegion, type NativeRegionEnv } from './native_region';
import type { Param } from '../../engine/region';

/** region_cone.cpp: "static constexpr double BIG = 1.0e200;" */
const BIG = 1.0e200;

export class NativeCone extends NativeRegion {
  /** c1, c2: the centre coordinates perpendicular to the axis (native's c1, c2). */
  c1 = 0;
  c2 = 0;
  /** radiuslo, radiushi: the radii at lo and hi (native's radiuslo, radiushi). */
  radiuslo = 0;
  radiushi = 0;
  /** lo, hi: the bounds along the axis (native's lo, hi). */
  lo = 0;
  hi = 0;
  /**
   * native's maxradius. Set once in the constructor (after its shape_update) and never updated by shape_update, so it
   * keeps the value of the constructor when the radii are variables (quirk reproduced on purpose).
   */
  maxradius = 0;

  // scratch arrays of native's local arrays (surface_interior: surflo, surfhi, xs; surface_exterior: corner1..4, xp,
  // nearest)
  private readonly surflo = new Float64Array(3);
  private readonly surfhi = new Float64Array(3);
  private readonly xs = new Float64Array(3);
  private readonly corner1 = new Float64Array(3);
  private readonly corner2 = new Float64Array(3);
  private readonly corner3 = new Float64Array(3);
  private readonly corner4 = new Float64Array(3);
  private readonly xp = new Float64Array(3);
  /** closest()'s output: native leaves it unset when closest() keeps the old distance (the scratch keeps its last value). */
  private readonly nearest = new Float64Array(3);

  /**
   * caxis: 0 = 'x', 1 = 'y', 2 = 'z'. c1, c2, radiuslo, radiushi, lo, hi: as the engine parsed them (numbers scaled by
   * the lattice, or variables with scale). Native lets every one of the six be a variable (c1, c2, radiuslo, radiushi,
   * lo, hi), so varshape is 1 when any of them is one.
   */
  constructor(
    id: string, env: NativeRegionEnv, readonly caxis: 0 | 1 | 2,
    private readonly pc1: Param, private readonly pc2: Param, private readonly prlo: Param, private readonly prhi: Param,
    private readonly plo: Param, private readonly phi: Param,
  ) {
    super(id, 'cone', env);
    if ([pc1, pc2, prlo, prhi, plo, phi].some((v) => typeof v !== 'number')) this.varshape = 1;
    // cmax = 3: the two contacts of a particle near the curved surface and one end, or one end plus the surface
    this.allocContacts(3);
  }

  /** The constructor's values, native's tmax (interior 2, exterior 1), and maxradius after the variable shape update. */
  setup(): void {
    if (typeof this.pc1 === 'number') this.c1 = this.pc1;
    if (typeof this.pc2 === 'number') this.c2 = this.pc2;
    if (typeof this.prlo === 'number') this.radiuslo = this.prlo;
    if (typeof this.prhi === 'number') this.radiushi = this.prhi;
    if (typeof this.plo === 'number') this.lo = this.plo;
    if (typeof this.phi === 'number') this.hi = this.phi;
    this.tmax = this.interior ? 2 : 1;
    if (this.varshape) this.shape_update();
    // maxradius = std::max(radiuslo, radiushi)
    this.maxradius = this.radiuslo < this.radiushi ? this.radiushi : this.radiuslo;
  }

  private val(p: Param): number {
    return typeof p === 'number' ? p : p.scale * this.env.variable(p.variable);
  }

  /** RegCone::shape_update: the variable values in native's order (the error checks are the engine's). */
  shape_update(): void {
    if (typeof this.pc1 !== 'number') this.c1 = this.val(this.pc1);
    if (typeof this.pc2 !== 'number') this.c2 = this.val(this.pc2);
    if (typeof this.prlo !== 'number') this.radiuslo = this.val(this.prlo);
    if (typeof this.prhi !== 'number') this.radiushi = this.val(this.prhi);
    if (typeof this.plo !== 'number') this.lo = this.val(this.plo);
    if (typeof this.phi !== 'number') this.hi = this.val(this.phi);
  }

  /** RegCone::inside (region_cone.cpp lines inside, 1 if inside or on the surface). */
  inside(x: number, y: number, z: number): number {
    let del1: number, del2: number, dist: number, currentradius: number;
    if (this.caxis === 0) {
      del1 = y - this.c1;
      del2 = z - this.c2;
      dist = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = this.radiuslo + (x - this.lo) * (this.radiushi - this.radiuslo) / (this.hi - this.lo);
      if (dist <= currentradius && x >= this.lo && x <= this.hi) return 1;
      return 0;
    } else if (this.caxis === 1) {
      del1 = x - this.c1;
      del2 = z - this.c2;
      dist = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = this.radiuslo + (y - this.lo) * (this.radiushi - this.radiuslo) / (this.hi - this.lo);
      if (dist <= currentradius && y >= this.lo && y <= this.hi) return 1;
      return 0;
    } else {
      del1 = x - this.c1;
      del2 = y - this.c2;
      dist = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = this.radiuslo + (z - this.lo) * (this.radiushi - this.radiuslo) / (this.hi - this.lo);
      if (dist <= currentradius && z >= this.lo && z <= this.hi) return 1;
      return 0;
    }
  }

  /**
   * RegCone::surface_interior (region_cone.cpp, surface_interior): contacts with the three inner surfaces (lateral
   * face = iwall 2, the lo end = iwall 0, the hi end = iwall 1). The lateral contact radius is -2 times the radius of
   * the surface point, as native writes it. Open faces: lateral = open_faces[2], lo = open_faces[0], hi = open_faces[1].
   */
  surface_interior(x: Float64Array, cutoff: number): number {
    let del1: number, del2: number, r: number, currentradius: number;
    let delx: number, dely: number, delz: number, dist: number, delta: number;
    const cR = this.cR, cDelx = this.cDelx, cDely = this.cDely, cDelz = this.cDelz, cRadius = this.cRadius, cIwall = this.cIwall;
    const of = this.open_faces;
    const surflo = this.surflo, surfhi = this.surfhi, xs = this.xs;
    const { c1, c2, radiuslo, radiushi, lo, hi } = this;

    let n = 0;

    if (this.caxis === 0) {
      del1 = x[1] - c1;
      del2 = x[2] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = radiuslo + (x[0] - lo) * (radiushi - radiuslo) / (hi - lo);

      // x is exterior to cone
      if (r > currentradius || x[0] < lo || x[0] > hi) return 0;

      // x is interior to cone or on its surface
      // surflo = pt on outer circle of bottom end plane, same dir as x vs axis
      // surfhi = pt on outer circle of top end plane, same dir as x vs axis
      if (r > 0.0 && !of[2]) {
        surflo[0] = lo;
        surflo[1] = c1 + del1 * radiuslo / r;
        surflo[2] = c2 + del2 * radiuslo / r;
        surfhi[0] = hi;
        surfhi[1] = c1 + del1 * radiushi / r;
        surfhi[2] = c2 + del2 * radiushi / r;
        this.point_on_line_segment(surflo, surfhi, x, xs);
        delx = x[0] - xs[0];
        dely = x[1] - xs[1];
        delz = x[2] - xs[2];
        dist = Math.sqrt(delx * delx + dely * dely + delz * delz);
        if (dist < cutoff) {
          cR[n] = dist;
          cDelx[n] = delx;
          cDely[n] = dely;
          cDelz[n] = delz;
          cRadius[n] = -2.0 * (radiuslo + (xs[0] - lo) * (radiushi - radiuslo) / (hi - lo));
          cIwall[n] = 2;
          n++;
        }
      }

      delta = x[0] - lo;
      if (delta < cutoff && !of[0]) {
        cR[n] = delta;
        cDelx[n] = delta;
        cDely[n] = 0.0;
        cDelz[n] = 0.0;
        cRadius[n] = 0;
        cIwall[n] = 0;
        n++;
      }
      delta = hi - x[0];
      if (delta < cutoff && !of[1]) {
        cR[n] = delta;
        cDelx[n] = -delta;
        cDely[n] = 0.0;
        cDelz[n] = 0.0;
        cRadius[n] = 0;
        cIwall[n] = 1;
        n++;
      }
    } else if (this.caxis === 1) {
      del1 = x[0] - c1;
      del2 = x[2] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = radiuslo + (x[1] - lo) * (radiushi - radiuslo) / (hi - lo);

      // y is exterior to cone
      if (r > currentradius || x[1] < lo || x[1] > hi) return 0;

      // y is interior to cone or on its surface
      // surflo = pt on outer circle of bottom end plane, same dir as y vs axis
      // surfhi = pt on outer circle of top end plane, same dir as y vs axis
      if (r > 0.0 && !of[2]) {
        surflo[0] = c1 + del1 * radiuslo / r;
        surflo[1] = lo;
        surflo[2] = c2 + del2 * radiuslo / r;
        surfhi[0] = c1 + del1 * radiushi / r;
        surfhi[1] = hi;
        surfhi[2] = c2 + del2 * radiushi / r;
        this.point_on_line_segment(surflo, surfhi, x, xs);
        delx = x[0] - xs[0];
        dely = x[1] - xs[1];
        delz = x[2] - xs[2];
        dist = Math.sqrt(delx * delx + dely * dely + delz * delz);
        if (dist < cutoff) {
          cR[n] = dist;
          cDelx[n] = delx;
          cDely[n] = dely;
          cDelz[n] = delz;
          cIwall[n] = 2;
          cRadius[n] = -2.0 * (radiuslo + (xs[1] - lo) * (radiushi - radiuslo) / (hi - lo));
          n++;
        }
      }

      delta = x[1] - lo;
      if (delta < cutoff && !of[0]) {
        cR[n] = delta;
        cDelz[n] = delta;
        cDelx[n] = 0.0;
        cDely[n] = 0.0;
        cIwall[n] = 0;
        cRadius[n] = 0;
        n++;
      }
      delta = hi - x[1];
      if (delta < cutoff && !of[1]) {
        cR[n] = delta;
        cDelz[n] = -delta;
        cDelx[n] = 0.0;
        cDely[n] = 0.0;
        cIwall[n] = 1;
        cRadius[n] = 0;
        n++;
      }
    } else {
      del1 = x[0] - c1;
      del2 = x[1] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = radiuslo + (x[2] - lo) * (radiushi - radiuslo) / (hi - lo);

      // z is exterior to cone
      if (r > currentradius || x[2] < lo || x[2] > hi) return 0;

      // z is interior to cone or on its surface
      // surflo = pt on outer circle of bottom end plane, same dir as z vs axis
      // surfhi = pt on outer circle of top end plane, same dir as z vs axis
      if (r > 0.0 && !of[2]) {
        surflo[0] = c1 + del1 * radiuslo / r;
        surflo[1] = c2 + del2 * radiuslo / r;
        surflo[2] = lo;
        surfhi[0] = c1 + del1 * radiushi / r;
        surfhi[1] = c2 + del2 * radiushi / r;
        surfhi[2] = hi;
        this.point_on_line_segment(surflo, surfhi, x, xs);
        delx = x[0] - xs[0];
        dely = x[1] - xs[1];
        delz = x[2] - xs[2];
        dist = Math.sqrt(delx * delx + dely * dely + delz * delz);
        if (dist < cutoff) {
          cR[n] = dist;
          cDelx[n] = delx;
          cDely[n] = dely;
          cDelz[n] = delz;
          cIwall[n] = 2;
          cRadius[n] = -2.0 * (radiuslo + (xs[2] - lo) * (radiushi - radiuslo) / (hi - lo));
          n++;
        }
      }

      delta = x[2] - lo;
      if (delta < cutoff && !of[0]) {
        cR[n] = delta;
        cDelz[n] = delta;
        cDelx[n] = 0.0;
        cDely[n] = 0.0;
        cIwall[n] = 0;
        cRadius[n] = 0;
        n++;
      }
      delta = hi - x[2];
      if (delta < cutoff && !of[1]) {
        cR[n] = delta;
        cDelz[n] = -delta;
        cDelx[n] = 0.0;
        cDely[n] = 0.0;
        cIwall[n] = 1;
        cRadius[n] = 0;
        n++;
      }
    }

    return n;
  }

  /**
   * RegCone::surface_exterior (region_cone.cpp, surface_exterior): one contact with the outer surface, the nearest of
   * the lateral trapezoid edges (corner1-2), the lo cap (corner1-3) and the hi cap (corner2-4), each only when its face
   * is not open. crad is the curvature radius (-2 times the radius at the nearest point) on the lateral face and 0 on
   * the caps. Native's closest() keeps the nearest point of the previous closest() call when a later call does not beat
   * it, so this.nearest persists across calls as native's local array does for the value it reads.
   */
  surface_exterior(x: Float64Array, cutoff: number): number {
    let del1: number, del2: number, r: number, currentradius: number, distsq: number, distsqprev: number, crad: number;
    const { c1, c2, radiuslo, radiushi, lo, hi, maxradius } = this;
    const of = this.open_faces;
    const corner1 = this.corner1, corner2 = this.corner2, corner3 = this.corner3, corner4 = this.corner4;
    const xp = this.xp, nearest = this.nearest;

    if (this.caxis === 0) {
      del1 = x[1] - c1;
      del2 = x[2] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = radiuslo + (x[0] - lo) * (radiushi - radiuslo) / (hi - lo);

      // radius of curvature, only used for granular walls
      crad = 0.0;

      // x is far enough from cone that there is no contact
      // x is interior to cone
      if (r >= maxradius + cutoff || x[0] <= lo - cutoff || x[0] >= hi + cutoff) return 0;
      if (r < currentradius && x[0] > lo && x[0] < hi) return 0;

      // x is exterior to cone or on its surface
      // corner1234 = 4 corner pts of half trapezoid = cone surf in plane of x
      // project x to 3 line segments in half trapezoid (4th is axis of cone)
      // nearest = point on surface of cone that x is closest to
      //           could be edge of cone
      // do not add contact point if r >= cutoff
      corner1[0] = lo;
      corner1[1] = c1 + del1 * radiuslo / r;
      corner1[2] = c2 + del2 * radiuslo / r;
      corner2[0] = hi;
      corner2[1] = c1 + del1 * radiushi / r;
      corner2[2] = c2 + del2 * radiushi / r;
      corner3[0] = lo;
      corner3[1] = c1;
      corner3[2] = c2;
      corner4[0] = hi;
      corner4[1] = c1;
      corner4[2] = c2;

      distsq = BIG;

      if (!of[2]) {
        this.point_on_line_segment(corner1, corner2, x, xp);
        distsq = this.closest(x, xp, distsq);
        crad = -2.0 * (radiuslo + (nearest[0] - lo) * (radiushi - radiuslo) / (hi - lo));
      }

      if (!of[0]) {
        this.point_on_line_segment(corner1, corner3, x, xp);
        distsqprev = distsq;
        distsq = this.closest(x, xp, distsq);
        if (distsq < distsqprev) crad = 0.0;
      }

      if (!of[1]) {
        this.point_on_line_segment(corner2, corner4, x, xp);
        distsqprev = distsq;
        distsq = this.closest(x, xp, distsq);
        if (distsq < distsqprev) crad = 0.0;
      }

      if (distsq === BIG) return 0;
      this.add_contact(0, x, nearest[0], nearest[1], nearest[2]);
      this.cRadius[0] = crad;
      this.cIwall[0] = 0;
      if (this.cR[0] < cutoff) return 1;
      return 0;
    } else if (this.caxis === 1) {
      del1 = x[0] - c1;
      del2 = x[2] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = radiuslo + (x[1] - lo) * (radiushi - radiuslo) / (hi - lo);

      // radius of curvature, only used for granular walls
      crad = 0.0;

      // y is far enough from cone that there is no contact
      // y is interior to cone
      if (r >= maxradius + cutoff || x[1] <= lo - cutoff || x[1] >= hi + cutoff) return 0;
      if (r < currentradius && x[1] > lo && x[1] < hi) return 0;

      // y is exterior to cone or on its surface
      // corner1234 = 4 corner pts of half trapezoid = cone surf in plane of y
      // project x to 3 line segments in half trapezoid (4th is axis of cone)
      // nearest = point on surface of cone that y is closest to
      //           could be edge of cone
      // do not add contact point if r >= cutoff
      corner1[0] = c1 + del1 * radiuslo / r;
      corner1[1] = lo;
      corner1[2] = c2 + del2 * radiuslo / r;
      corner2[0] = c1 + del1 * radiushi / r;
      corner2[1] = hi;
      corner2[2] = c2 + del2 * radiushi / r;
      corner3[0] = c1;
      corner3[1] = lo;
      corner3[2] = c2;
      corner4[0] = c1;
      corner4[1] = hi;
      corner4[2] = c2;

      distsq = BIG;

      if (!of[2]) {
        this.point_on_line_segment(corner1, corner2, x, xp);
        distsq = this.closest(x, xp, distsq);
        crad = -2.0 * (radiuslo + (nearest[1] - lo) * (radiushi - radiuslo) / (hi - lo));
      }

      if (!of[0]) {
        this.point_on_line_segment(corner1, corner3, x, xp);
        distsqprev = distsq;
        distsq = this.closest(x, xp, distsq);
        if (distsq < distsqprev) crad = 0;
      }

      if (!of[1]) {
        this.point_on_line_segment(corner2, corner4, x, xp);
        distsqprev = distsq;
        distsq = this.closest(x, xp, distsq);
        if (distsq < distsqprev) crad = 0;
      }

      if (distsq === BIG) return 0;
      this.add_contact(0, x, nearest[0], nearest[1], nearest[2]);
      this.cRadius[0] = crad;
      this.cIwall[0] = 0;
      if (this.cR[0] < cutoff) return 1;
      return 0;
    } else {
      del1 = x[0] - c1;
      del2 = x[1] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);
      currentradius = radiuslo + (x[2] - lo) * (radiushi - radiuslo) / (hi - lo);

      // radius of curvature, only used for granular walls
      crad = 0.0;

      // z is far enough from cone that there is no contact
      // z is interior to cone
      if (r >= maxradius + cutoff || x[2] <= lo - cutoff || x[2] >= hi + cutoff) return 0;
      if (r < currentradius && x[2] > lo && x[2] < hi) return 0;

      // z is exterior to cone or on its surface
      // corner1234 = 4 corner pts of half trapezoid = cone surf in plane of z
      // project x to 3 line segments in half trapezoid (4th is axis of cone)
      // nearest = point on surface of cone that z is closest to
      //           could be edge of cone
      // do not add contact point if r >= cutoff
      corner1[0] = c1 + del1 * radiuslo / r;
      corner1[1] = c2 + del2 * radiuslo / r;
      corner1[2] = lo;
      corner2[0] = c1 + del1 * radiushi / r;
      corner2[1] = c2 + del2 * radiushi / r;
      corner2[2] = hi;
      corner3[0] = c1;
      corner3[1] = c2;
      corner3[2] = lo;
      corner4[0] = c1;
      corner4[1] = c2;
      corner4[2] = hi;

      distsq = BIG;

      if (!of[2]) {
        this.point_on_line_segment(corner1, corner2, x, xp);
        distsq = this.closest(x, xp, distsq);
        crad = -2.0 * (radiuslo + (nearest[2] - lo) * (radiushi - radiuslo) / (hi - lo));
      }

      if (!of[0]) {
        this.point_on_line_segment(corner1, corner3, x, xp);
        distsqprev = distsq;
        distsq = this.closest(x, xp, distsq);
        if (distsq < distsqprev) crad = 0;
      }

      if (!of[1]) {
        this.point_on_line_segment(corner2, corner4, x, xp);
        distsqprev = distsq;
        distsq = this.closest(x, xp, distsq);
        if (distsq < distsqprev) crad = 0;
      }

      if (distsq === BIG) return 0;
      this.add_contact(0, x, nearest[0], nearest[1], nearest[2]);
      this.cRadius[0] = crad;
      this.cIwall[0] = 0;
      if (this.cR[0] < cutoff) return 1;
      return 0;
    }
  }

  /**
   * RegCone::closest (region_cone.cpp, closest): the squared distance from x to near when it is below dsq (and near
   * becomes the nearest point), else dsq. Writes this.nearest only in the first case, as native does.
   */
  private closest(x: Float64Array, near: Float64Array, dsq: number): number {
    const delx = x[0] - near[0];
    const dely = x[1] - near[1];
    const delz = x[2] - near[2];
    const rsq = delx * delx + dely * dely + delz * delz;
    if (rsq >= dsq) return dsq;

    const nearest = this.nearest;
    nearest[0] = near[0];
    nearest[1] = near[1];
    nearest[2] = near[2];
    return rsq;
  }
}
