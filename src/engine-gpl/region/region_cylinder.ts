/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/region_cylinder.cpp and src/region_cylinder.h (RegCylinder: the constructor's variable handling and cmax,
 * inside, surface_interior, surface_exterior, shape_update, set_velocity_shape and velocity_contact_shape) at commit
 * b332f28. Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 */
import { NativeRegion, type NativeRegionEnv } from './native_region';
import type { Param } from '../../engine/region';

/** region_cylinder.cpp: "static constexpr double BIG = 1.0e200;" */
const BIG = 1.0e200;

/** A parameter's value now: a constant, or an equal-style variable times its lattice scale (as shape_update does). */
const val = (env: NativeRegionEnv, p: Param): number => (typeof p === 'number' ? p : p.scale * env.variable(p.variable));

export class NativeCylinder extends NativeRegion {
  /** The centre coordinates perpendicular to the axis (c1, c2), the radius, and the bounds along the axis (lo, hi). */
  c1 = 0;
  c2 = 0;
  radius = 0;
  lo = 0;
  hi = 0;
  /** native's char axis: 0 = 'x', 1 = 'y', 2 = 'z'. */
  readonly caxis: 0 | 1 | 2;

  /**
   * c1, c2 and radius may be variables in native (arguments 3 to 5 with a "v_" prefix). lo and hi cannot be (native reads
   * them as numbers only); they are resolved once in setup() so that a Param of either kind still gives a value.
   */
  constructor(
    id: string,
    env: NativeRegionEnv,
    caxis: 0 | 1 | 2,
    private readonly p1: Param,
    private readonly p2: Param,
    private readonly pr: Param,
    private readonly plo: Param,
    private readonly phi: Param,
  ) {
    super(id, 'cylinder', env);
    this.caxis = caxis;
    // RegCylinder constructor: "varshape = 1" for each of c1, c2 and radius given as "v_name"
    if (typeof p1 !== 'number' || typeof p2 !== 'number' || typeof pr !== 'number') this.varshape = 1;
    // "cmax = 3;" (a particle can touch the curved surface and one end; or only one surface and one end)
    this.allocContacts(3);
  }

  /**
   * The constructor's constant values, then shape_update when a variable is present (the constructor calls
   * RegCylinder::shape_update after the variable checks); tmax is 2 inside and 1 outside.
   */
  setup(): void {
    if (typeof this.p1 === 'number') this.c1 = this.p1;
    if (typeof this.p2 === 'number') this.c2 = this.p2;
    if (typeof this.pr === 'number') this.radius = this.pr;
    this.lo = val(this.env, this.plo);
    this.hi = val(this.env, this.phi);
    this.tmax = this.interior ? 2 : 1;
    if (this.varshape) this.shape_update();
  }

  /** RegCylinder::inside: 1 if the point is within the radius and between the bounds along the axis (or on them). */
  inside(x: number, y: number, z: number): number {
    let del1: number, del2: number, dist: number;
    let inside: number;
    if (this.caxis === 0) {
      del1 = y - this.c1;
      del2 = z - this.c2;
      dist = Math.sqrt(del1 * del1 + del2 * del2);
      if (dist <= this.radius && x >= this.lo && x <= this.hi) inside = 1;
      else inside = 0;
    } else if (this.caxis === 1) {
      del1 = x - this.c1;
      del2 = z - this.c2;
      dist = Math.sqrt(del1 * del1 + del2 * del2);
      if (dist <= this.radius && y >= this.lo && y <= this.hi) inside = 1;
      else inside = 0;
    } else {
      del1 = x - this.c1;
      del2 = y - this.c2;
      dist = Math.sqrt(del1 * del1 + del2 * del2);
      if (dist <= this.radius && z >= this.lo && z <= this.hi) inside = 1;
      else inside = 0;
    }
    return inside;
  }

  /**
   * RegCylinder::surface_interior: contacts of a point inside the cylinder (or on its surface) within cutoff, from the
   * curved surface (iwall 2, radius -2 R, varflag 1) and the two ends (iwall 0 at lo, 1 at hi, varflag 0). Open faces:
   * open_faces[2] is the curved surface, [0] the lo end and [1] the hi end. No curved contact on the axis (r = 0).
   */
  surface_interior(x: Float64Array, cutoff: number): number {
    const of = this.open_faces;
    const cR = this.cR, cDelx = this.cDelx, cDely = this.cDely, cDelz = this.cDelz;
    const cRadius = this.cRadius, cIwall = this.cIwall, cVarflag = this.cVarflag;
    const radius = this.radius, lo = this.lo, hi = this.hi;
    let del1: number, del2: number, r: number, delta: number;
    let n = 0;

    if (this.caxis === 0) {
      del1 = x[1] - this.c1;
      del2 = x[2] - this.c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);

      // x is exterior to cylinder
      if (r > radius || x[0] < lo || x[0] > hi) return 0;

      // x is interior to cylinder or on its surface
      delta = radius - r;
      if (delta < cutoff && r > 0.0 && !of[2]) {
        cR[n] = delta;
        cDelx[n] = 0.0;
        cDely[n] = del1 * (1.0 - radius / r);
        cDelz[n] = del2 * (1.0 - radius / r);
        cRadius[n] = -2.0 * radius;
        cIwall[n] = 2;
        cVarflag[n] = 1;
        n++;
      }
      delta = x[0] - lo;
      if (delta < cutoff && !of[0]) {
        cR[n] = delta;
        cDelx[n] = delta;
        cDely[n] = 0.0;
        cDelz[n] = 0.0;
        cRadius[n] = 0;
        cIwall[n] = 0;
        cVarflag[n] = 0;
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
        cVarflag[n] = 0;
        n++;
      }
    } else if (this.caxis === 1) {
      del1 = x[0] - this.c1;
      del2 = x[2] - this.c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);

      // y is exterior to cylinder
      if (r > radius || x[1] < lo || x[1] > hi) return 0;

      // y is interior to cylinder or on its surface
      delta = radius - r;
      if (delta < cutoff && r > 0.0 && !of[2]) {
        cR[n] = delta;
        cDelx[n] = del1 * (1.0 - radius / r);
        cDely[n] = 0.0;
        cDelz[n] = del2 * (1.0 - radius / r);
        cRadius[n] = -2.0 * radius;
        cIwall[n] = 2;
        cVarflag[n] = 1;
        n++;
      }
      delta = x[1] - lo;
      if (delta < cutoff && !of[0]) {
        cR[n] = delta;
        cDely[n] = delta;
        cDelx[n] = 0.0;
        cDelz[n] = 0.0;
        cRadius[n] = 0;
        cIwall[n] = 0;
        cVarflag[n] = 0;
        n++;
      }
      delta = hi - x[1];
      if (delta < cutoff && !of[1]) {
        cR[n] = delta;
        cDely[n] = -delta;
        cDelx[n] = 0.0;
        cDelz[n] = 0.0;
        cRadius[n] = 0;
        cIwall[n] = 1;
        cVarflag[n] = 0;
        n++;
      }
    } else {
      del1 = x[0] - this.c1;
      del2 = x[1] - this.c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);

      // z is exterior to cylinder
      if (r > radius || x[2] < lo || x[2] > hi) return 0;

      // z is interior to cylinder or on its surface
      delta = radius - r;
      if (delta < cutoff && r > 0.0 && !of[2]) {
        cR[n] = delta;
        cDelx[n] = del1 * (1.0 - radius / r);
        cDely[n] = del2 * (1.0 - radius / r);
        cDelz[n] = 0.0;
        cRadius[n] = -2.0 * radius;
        cIwall[n] = 2;
        cVarflag[n] = 1;
        n++;
      }
      delta = x[2] - lo;
      if (delta < cutoff && !of[0]) {
        cR[n] = delta;
        cDelz[n] = delta;
        cDelx[n] = 0.0;
        cDely[n] = 0.0;
        cRadius[n] = 0;
        cIwall[n] = 0;
        cVarflag[n] = 0;
        n++;
      }
      delta = hi - x[2];
      if (delta < cutoff && !of[1]) {
        cR[n] = delta;
        cDelz[n] = -delta;
        cDelx[n] = 0.0;
        cDely[n] = 0.0;
        cRadius[n] = 0;
        cIwall[n] = 1;
        cVarflag[n] = 0;
        n++;
      }
    }

    return n;
  }

  /**
   * RegCylinder::surface_exterior: one contact (index 0) from the outer surface of the cylinder, its closest point on the
   * curved surface, the lo end or the hi end (the open faces are skipped in the openflag branch). Curved contacts carry
   * radius 2 R and varflag 1; flat contacts radius 0 and varflag 0. Native's xp, yp, zp are not initialised; a face is
   * always selected here, so they are set in every reachable path, and they are initialised to 0 for JS.
   */
  surface_exterior(x: Float64Array, cutoff: number): number {
    const of = this.open_faces;
    // with all three faces open there is no surface left to contact
    if (of[0] && of[1] && of[2]) return 0;

    const radius = this.radius, lo = this.lo, hi = this.hi, c1 = this.c1, c2 = this.c2;
    // radius of curvature for granular: 0 for flat surfaces (infinite case), 2*radius for curved portion
    let crad = 0.0;
    let varflag = 0;
    let del1: number, del2: number, r: number;
    let xp = 0, yp = 0, zp = 0;
    let dx = 0, dr = 0, dr2 = 0, d2 = 0, d2prev = 0;

    if (this.caxis === 0) {
      del1 = x[1] - c1;
      del2 = x[2] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);

      // x is far enough from cylinder that there is no contact
      // x is interior to cylinder
      if (r >= radius + cutoff || x[0] <= lo - cutoff || x[0] >= hi + cutoff) return 0;
      if (r < radius && x[0] > lo && x[0] < hi) return 0;

      // x is exterior to cylinder or on its surface
      // xp,yp,zp = point on surface of cylinder that x is closest to, could be edge of cylinder
      // do not add contact point if r >= cutoff
      d2prev = BIG;
      if (!this.openflag) {
        if (r > radius) {
          yp = c1 + del1 * radius / r;
          zp = c2 + del2 * radius / r;
          crad = 2.0 * radius;
          varflag = 1;
        } else {
          yp = x[1];
          zp = x[2];
        }
        if (x[0] < lo) xp = lo;
        else if (x[0] > hi) xp = hi;
        else xp = x[0];
      } else {
        // closest point on curved surface
        dr = r - radius;
        dr2 = dr * dr;
        if (!of[2]) {
          yp = c1 + del1 * radius / r;
          zp = c2 + del2 * radius / r;
          if (x[0] < lo) {
            dx = lo - x[0];
            xp = lo;
          } else if (x[0] > hi) {
            dx = x[0] - hi;
            xp = hi;
          } else {
            dx = 0;
            xp = x[0];
          }
          d2prev = dr2 + dx * dx;
          if (r > radius) {
            crad = 2.0 * radius;
            varflag = 1;
          }
        }

        // closest point on bottom cap
        if (!of[0]) {
          dx = lo - x[0];
          if (r < radius) d2 = dx * dx;
          else d2 = dr2 + dx * dx;
          if (d2 < d2prev) {
            xp = lo;
            if (r < radius) {
              yp = x[1];
              zp = x[2];
            } else {
              yp = c1 + del1 * radius / r;
              zp = c2 + del2 * radius / r;
            }
            d2prev = d2;
          }
        }

        // closest point on top cap (native does not update d2prev here; it is the last face)
        if (!of[1]) {
          dx = hi - x[0];
          if (r < radius) d2 = dx * dx;
          else d2 = dr2 + dx * dx;
          if (d2 < d2prev) {
            xp = hi;
            if (r < radius) {
              yp = x[1];
              zp = x[2];
            } else {
              yp = c1 + del1 * radius / r;
              zp = c2 + del2 * radius / r;
            }
          }
        }
      }

      this.add_contact(0, x, xp, yp, zp);
      this.cRadius[0] = crad;
      this.cVarflag[0] = varflag;
      this.cIwall[0] = 0;
      if (this.cR[0] < cutoff) return 1;
      return 0;
    } else if (this.caxis === 1) {
      del1 = x[0] - c1;
      del2 = x[2] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);

      // y is far enough from cylinder that there is no contact
      // y is interior to cylinder
      if (r >= radius + cutoff || x[1] <= lo - cutoff || x[1] >= hi + cutoff) return 0;
      if (r < radius && x[1] > lo && x[1] < hi) return 0;

      // y is exterior to cylinder or on its surface
      d2prev = BIG;
      if (!this.openflag) {
        if (r > radius) {
          xp = c1 + del1 * radius / r;
          zp = c2 + del2 * radius / r;
          crad = 2.0 * radius;
          varflag = 1;
        } else {
          xp = x[0];
          zp = x[2];
        }
        if (x[1] < lo) yp = lo;
        else if (x[1] > hi) yp = hi;
        else yp = x[1];
      } else {
        // closest point on curved surface
        dr = r - radius;
        dr2 = dr * dr;
        if (!of[2]) {
          xp = c1 + del1 * radius / r;
          zp = c2 + del2 * radius / r;
          if (x[1] < lo) {
            dx = lo - x[1];
            yp = lo;
          } else if (x[1] > hi) {
            dx = x[1] - hi;
            yp = hi;
          } else {
            dx = 0;
            yp = x[1];
          }
          d2prev = dr2 + dx * dx;
          if (r > radius) {
            crad = 2.0 * radius;
            varflag = 1;
          }
        }

        // closest point on bottom cap
        if (!of[0]) {
          dx = lo - x[1];
          if (r < radius) d2 = dx * dx;
          else d2 = dr2 + dx * dx;
          if (d2 < d2prev) {
            yp = lo;
            if (r < radius) {
              xp = x[0];
              zp = x[2];
            } else {
              xp = c1 + del1 * radius / r;
              zp = c2 + del2 * radius / r;
            }
            d2prev = d2;
          }
        }

        // closest point on top cap (native does not update d2prev here; it is the last face)
        if (!of[1]) {
          dx = hi - x[1];
          if (r < radius) d2 = dx * dx;
          else d2 = dr2 + dx * dx;
          if (d2 < d2prev) {
            yp = hi;
            if (r < radius) {
              xp = x[0];
              zp = x[2];
            } else {
              xp = c1 + del1 * radius / r;
              zp = c2 + del2 * radius / r;
            }
          }
        }
      }

      this.add_contact(0, x, xp, yp, zp);
      this.cRadius[0] = crad;
      this.cVarflag[0] = varflag;
      this.cIwall[0] = 0;
      if (this.cR[0] < cutoff) return 1;
      return 0;
    } else {
      del1 = x[0] - c1;
      del2 = x[1] - c2;
      r = Math.sqrt(del1 * del1 + del2 * del2);

      // z is far enough from cylinder that there is no contact
      // z is interior to cylinder
      if (r >= radius + cutoff || x[2] <= lo - cutoff || x[2] >= hi + cutoff) return 0;
      if (r < radius && x[2] > lo && x[2] < hi) return 0;

      // z is exterior to cylinder or on its surface
      d2prev = BIG;
      if (!this.openflag) {
        if (r > radius) {
          xp = c1 + del1 * radius / r;
          yp = c2 + del2 * radius / r;
          crad = 2.0 * radius;
          varflag = 1;
        } else {
          xp = x[0];
          yp = x[1];
        }
        if (x[2] < lo) zp = lo;
        else if (x[2] > hi) zp = hi;
        else zp = x[2];
      } else {
        // closest point on curved surface
        dr = r - radius;
        dr2 = dr * dr;
        if (!of[2]) {
          xp = c1 + del1 * radius / r;
          yp = c2 + del2 * radius / r;
          if (x[2] < lo) {
            dx = lo - x[2];
            zp = lo;
          } else if (x[2] > hi) {
            dx = x[2] - hi;
            zp = hi;
          } else {
            dx = 0;
            zp = x[2];
          }
          d2prev = dr2 + dx * dx;
          if (r > radius) {
            crad = 2.0 * radius;
            varflag = 1;
          }
        }

        // closest point on bottom cap
        if (!of[0]) {
          dx = lo - x[2];
          if (r < radius) d2 = dx * dx;
          else d2 = dr2 + dx * dx;
          if (d2 < d2prev) {
            zp = lo;
            if (r < radius) {
              xp = x[0];
              yp = x[1];
            } else {
              xp = c1 + del1 * radius / r;
              yp = c2 + del2 * radius / r;
            }
            d2prev = d2;
          }
        }

        // closest point on top cap (native does not update d2prev here; it is the last face)
        if (!of[1]) {
          dx = hi - x[2];
          if (r < radius) d2 = dx * dx;
          else d2 = dr2 + dx * dx;
          if (d2 < d2prev) {
            zp = hi;
            if (r < radius) {
              xp = x[0];
              yp = x[1];
            } else {
              xp = c1 + del1 * radius / r;
              yp = c2 + del2 * radius / r;
            }
          }
        }
      }

      this.add_contact(0, x, xp, yp, zp);
      this.cRadius[0] = crad;
      this.cVarflag[0] = varflag;
      this.cIwall[0] = 0;
      if (this.cR[0] < cutoff) return 1;
      return 0;
    }
  }

  /**
   * RegCylinder::shape_update: the variable values of c1, c2 and radius, each times its lattice scale (the scale of the
   * dimension it lies in, as the engine parsed it). Constant values are left as they are.
   */
  shape_update(): void {
    if (typeof this.p1 !== 'number') this.c1 = val(this.env, this.p1);
    if (typeof this.p2 !== 'number') this.c2 = val(this.env, this.p2);
    if (typeof this.pr !== 'number') this.radius = val(this.env, this.pr);
  }

  /** RegCylinder::set_velocity_shape: the centre of the cylinder's end (the point on the axis) and the radius history. */
  set_velocity_shape(step: number): void {
    const xcenter = this.xcenter;
    if (this.caxis === 0) {
      xcenter[0] = 0;
      xcenter[1] = this.c1;
      xcenter[2] = this.c2;
    } else if (this.caxis === 1) {
      xcenter[0] = this.c1;
      xcenter[1] = 0;
      xcenter[2] = this.c2;
    } else {
      xcenter[0] = this.c1;
      xcenter[1] = this.c2;
      xcenter[2] = 0;
    }
    this.forward_transform(xcenter);
    if (step > 0) this.rprev = this.prev[4];
    else this.rprev = this.radius;
    this.prev[4] = this.radius;
  }

  /** RegCylinder::velocity_contact_shape: add the wall velocity due to the radius change at contact point xc. */
  velocity_contact_shape(vwall: Float64Array, xc: ArrayLike<number>, dt: number): void {
    const xcenter = this.xcenter;
    const rprev = this.rprev, radius = this.radius;
    // displacement of contact point in x, y, z
    let delx: number, dely: number, delz: number;
    if (this.caxis === 0) {
      delx = 0;
      dely = (xc[1] - xcenter[1]) * (1 - rprev / radius);
      delz = (xc[2] - xcenter[2]) * (1 - rprev / radius);
    } else if (this.caxis === 1) {
      delx = (xc[0] - xcenter[0]) * (1 - rprev / radius);
      dely = 0;
      delz = (xc[2] - xcenter[2]) * (1 - rprev / radius);
    } else {
      delx = (xc[0] - xcenter[0]) * (1 - rprev / radius);
      dely = (xc[1] - xcenter[1]) * (1 - rprev / radius);
      delz = 0;
    }
    vwall[0] += delx / dt;
    vwall[1] += dely / dt;
    vwall[2] += delz / dt;
  }
}
