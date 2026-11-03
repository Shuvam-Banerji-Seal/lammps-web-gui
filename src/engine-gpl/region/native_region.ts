/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/region.cpp and src/region.h (Region::dynamic_check, prematch, match, surface, add_contact, pretransform,
 * forward_transform, inverse_transform, rotate, point_on_line_segment, set_velocity, velocity_contact, reset_vel and the
 * rotation-axis setup of Region::options) at commit b332f28. Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The native region surface for fix wall/gran/region: each region style of this directory mirrors a region of the
 * engine (src/engine/region.ts, whose parsed values it reads; see build.ts) with native's geometry in native's operation
 * order. The contact list is native's Region::contact array as parallel typed arrays (cR, cDelx, cDely, cDelz, cRadius,
 * cIwall, cVarflag) of length cmax, written in place and never cleared between calls, as native's array is.
 */
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** What a native region reads from the session. */
export interface NativeRegionEnv {
  /** input->variable->compute_equal of an equal-style variable. */
  variable(name: string): number;
  /** domain->remap of a point (periodic dimensions mapped into the box), in place. */
  remap(p: number[]): void;
}

export abstract class NativeRegion {
  interior = 1;
  openflag = 0;
  readonly open_faces = new Int8Array(6);
  varshape = 0;
  dynamic = 0;
  moveflag = 0;
  rotateflag = 0;
  xscale = 1;
  yscale = 1;
  zscale = 1;
  xstr: string | null = null;
  ystr: string | null = null;
  zstr: string | null = null;
  tstr: string | null = null;
  // Region::Region: "dx = dy = dz = 0.0; theta = 0.0;"
  dx = 0;
  dy = 0;
  dz = 0;
  theta = 0;
  readonly point = new Float64Array(3);
  readonly runit = new Float64Array(3);
  readonly axis = new Float64Array(3);
  readonly v = new Float64Array(3);
  readonly rpoint = new Float64Array(3);
  readonly omega = new Float64Array(3);
  rprev = 0;
  readonly xcenter = new Float64Array(3);
  readonly prev = new Float64Array(5);
  vel_timestep = -1;
  nregion = 1;
  cmax = 0;
  tmax = 0;
  cR = new Float64Array(0);
  cDelx = new Float64Array(0);
  cDely = new Float64Array(0);
  cDelz = new Float64Array(0);
  cRadius = new Float64Array(0);
  cIwall = new Int32Array(0);
  cVarflag = new Int32Array(0);
  /** domain->remap's copy of the point (match and surface), the original point and the body-frame point (surface). */
  protected readonly xremap = [0.5, 0.5, 0.5];
  protected readonly xorig = new Float64Array(3);
  protected readonly xnear = new Float64Array(3);
  /** forward_transform / inverse_transform operand (x, y, z in place). */
  readonly T = new Float64Array(3);
  /** rotate()'s sin and cos of the last two angles (+theta and -theta alternate; the same values as the calls). */
  private angA = NaN;
  private sinA = 0;
  private cosA = 1;
  private angB = NaN;
  private sinB = 0;
  private cosB = 1;

  constructor(readonly id: string, readonly style: string, protected readonly env: NativeRegionEnv) {}

  /** contact = new Contact[cmax] (the varflag of a Contact starts at 0). */
  protected allocContacts(cmax: number): void {
    this.cmax = cmax;
    this.cR = new Float64Array(cmax);
    this.cDelx = new Float64Array(cmax);
    this.cDely = new Float64Array(cmax);
    this.cDelz = new Float64Array(cmax);
    this.cRadius = new Float64Array(cmax);
    this.cIwall = new Int32Array(cmax);
    this.cVarflag = new Int32Array(cmax);
  }

  /** The rotation setup of Region::options: point scaled by the lattice, runit = axis / |axis| (sqrt of the squares). */
  setRotate(tstr: string, px: number, py: number, pz: number, ax: number, ay: number, az: number): void {
    this.tstr = tstr;
    this.rotateflag = 1;
    this.point[0] = px * this.xscale;
    this.point[1] = py * this.yscale;
    this.point[2] = pz * this.zscale;
    this.axis[0] = ax; this.axis[1] = ay; this.axis[2] = az;
    const len = Math.sqrt(ax * ax + ay * ay + az * az);
    this.runit[0] = ax / len;
    this.runit[1] = ay / len;
    this.runit[2] = az / len;
  }

  abstract inside(x: number, y: number, z: number): number;
  abstract surface_interior(x: Float64Array, cutoff: number): number;
  abstract surface_exterior(x: Float64Array, cutoff: number): number;
  shape_update(): void { /* constant shape */ }
  bbox_update(): void { /* the wall does not read the bounding box */ }
  set_velocity_shape(_step: number): void { /* constant shape */ }
  velocity_contact_shape(_vwall: Float64Array, _xc: ArrayLike<number>, _dt: number): void { /* constant shape */ }

  dynamic_check(): number {
    return this.dynamic || this.varshape ? 1 : 0;
  }

  prematch(): void {
    if (this.varshape) this.shape_update();
    if (this.dynamic) this.pretransform();
  }

  /** Region::match: remap into the box, unmove and unrotate, then !(inside ^ interior) (1 always for an open region). */
  match(x: number, y: number, z: number): number {
    const p = this.xremap;
    p[0] = x; p[1] = y; p[2] = z;
    this.env.remap(p);
    if (this.dynamic) {
      const t = this.T;
      t[0] = p[0]; t[1] = p[1]; t[2] = p[2];
      this.inverse_transform(t);
      p[0] = t[0]; p[1] = t[1]; p[2] = t[2];
    }
    if (this.openflag) return 1;
    return this.inside(p[0], p[1], p[2]) === this.interior ? 1 : 0;
  }

  /**
   * Region::surface: the contacts of the point within cutoff, delx/dely/delz from the surface point to the particle.
   * An open region tries the exterior first and the interior only without an exterior contact. With rotation the contact
   * points are rotated back and the deltas recomputed from the original point.
   */
  surface(x: number, y: number, z: number, cutoff: number): number {
    const p = this.xremap;
    p[0] = x; p[1] = y; p[2] = z;
    this.env.remap(p);
    const xorig = this.xorig;
    const xnear = this.xnear;
    if (this.dynamic) {
      xorig[0] = p[0]; xorig[1] = p[1]; xorig[2] = p[2];
      const t = this.T;
      t[0] = p[0]; t[1] = p[1]; t[2] = p[2];
      this.inverse_transform(t);
      p[0] = t[0]; p[1] = t[1]; p[2] = t[2];
    }
    xnear[0] = p[0]; xnear[1] = p[1]; xnear[2] = p[2];
    let ncontact: number;
    if (!this.openflag) {
      ncontact = this.interior ? this.surface_interior(xnear, cutoff) : this.surface_exterior(xnear, cutoff);
    } else {
      ncontact = this.surface_exterior(xnear, cutoff);
      if (ncontact === 0) ncontact = this.surface_interior(xnear, cutoff);
    }
    if (this.rotateflag && ncontact) {
      const t = this.T;
      for (let i = 0; i < ncontact; i++) {
        t[0] = xnear[0] - this.cDelx[i];
        t[1] = xnear[1] - this.cDely[i];
        t[2] = xnear[2] - this.cDelz[i];
        this.forward_transform(t);
        this.cDelx[i] = xorig[0] - t[0];
        this.cDely[i] = xorig[1] - t[1];
        this.cDelz[i] = xorig[2] - t[2];
      }
    }
    return ncontact;
  }

  /** Region::add_contact: contact n between the particle at x and the surface point (xp, yp, zp). */
  add_contact(n: number, x: Float64Array, xp: number, yp: number, zp: number): void {
    const delx = x[0] - xp;
    const dely = x[1] - yp;
    const delz = x[2] - zp;
    this.cR[n] = Math.sqrt(delx * delx + dely * dely + delz * delz);
    this.cRadius[n] = 0;
    this.cDelx[n] = delx;
    this.cDely[n] = dely;
    this.cDelz[n] = delz;
  }

  pretransform(): void {
    if (this.moveflag) {
      if (this.xstr) this.dx = this.env.variable(this.xstr);
      if (this.ystr) this.dy = this.env.variable(this.ystr);
      if (this.zstr) this.dz = this.env.variable(this.zstr);
    }
    if (this.rotateflag) this.theta = this.env.variable(this.tstr!);
    this.bbox_update();
  }

  /** Rotate first (around the original point P), then displace. */
  forward_transform(t: Float64Array): void {
    if (this.rotateflag) this.rotate(t, this.theta);
    if (this.moveflag) {
      t[0] += this.dx;
      t[1] += this.dy;
      t[2] += this.dz;
    }
  }

  /** Undisplace first, then unrotate (around the original point P). */
  inverse_transform(t: Float64Array): void {
    if (this.moveflag) {
      t[0] -= this.dx;
      t[1] -= this.dy;
      t[2] -= this.dz;
    }
    if (this.rotateflag) this.rotate(t, -this.theta);
  }

  /** Region::rotate: new x = P + C + A cos(angle) + B sin(angle) (C = (D.R0) R0, A = D - C, B = R0 x A, D = X0 - P). */
  rotate(t: Float64Array, angle: number): void {
    let sine: number, cosine: number;
    if (Object.is(angle, this.angA)) { sine = this.sinA; cosine = this.cosA; }
    else if (Object.is(angle, this.angB)) { sine = this.sinB; cosine = this.cosB; }
    else {
      sine = libmSin(angle);
      cosine = libmCos(angle);
      this.angB = this.angA; this.sinB = this.sinA; this.cosB = this.cosA;
      this.angA = angle; this.sinA = sine; this.cosA = cosine;
    }
    const point = this.point, runit = this.runit;
    const d0 = t[0] - point[0];
    const d1 = t[1] - point[1];
    const d2 = t[2] - point[2];
    const x0dotr = d0 * runit[0] + d1 * runit[1] + d2 * runit[2];
    const c0 = x0dotr * runit[0];
    const c1 = x0dotr * runit[1];
    const c2 = x0dotr * runit[2];
    const a0 = d0 - c0;
    const a1 = d1 - c1;
    const a2 = d2 - c2;
    const b0 = runit[1] * a2 - runit[2] * a1;
    const b1 = runit[2] * a0 - runit[0] * a2;
    const b2 = runit[0] * a1 - runit[1] * a0;
    const disp0 = a0 * cosine + b0 * sine;
    const disp1 = a1 * cosine + b1 * sine;
    const disp2 = a2 * cosine + b2 * sine;
    t[0] = point[0] + c0 + disp0;
    t[1] = point[1] + c1 + disp1;
    t[2] = point[2] + c2 + disp2;
  }

  /** Region::point_on_line_segment: the point d of segment a-b closest to c. */
  point_on_line_segment(a: ArrayLike<number>, b: ArrayLike<number>, c: ArrayLike<number>, d: Float64Array): void {
    const ba0 = b[0] - a[0], ba1 = b[1] - a[1], ba2 = b[2] - a[2];
    const ca0 = c[0] - a[0], ca1 = c[1] - a[1], ca2 = c[2] - a[2];
    const t = (ca0 * ba0 + ca1 * ba1 + ca2 * ba2) / (ba0 * ba0 + ba1 * ba1 + ba2 * ba2);
    if (t <= 0.0) {
      d[0] = a[0]; d[1] = a[1]; d[2] = a[2];
    } else if (t >= 1.0) {
      d[0] = b[0]; d[1] = b[1]; d[2] = b[2];
    } else {
      d[0] = a[0] + t * ba0;
      d[1] = a[1] + t * ba1;
      d[2] = a[2] + t * ba2;
    }
  }

  /**
   * Region::set_velocity: the translational and angular velocity as differences of the motion over one timestep
   * (zero on step 0), once per step; prev keeps this step's displacement and angle. omega is along the axis as given.
   */
  set_velocity(step: number, dt: number): void {
    if (this.vel_timestep === step) return;
    this.vel_timestep = step;
    const prev = this.prev;
    if (this.moveflag) {
      if (step > 0) {
        this.v[0] = (this.dx - prev[0]) / dt;
        this.v[1] = (this.dy - prev[1]) / dt;
        this.v[2] = (this.dz - prev[2]) / dt;
      } else this.v[0] = this.v[1] = this.v[2] = 0.0;
      prev[0] = this.dx;
      prev[1] = this.dy;
      prev[2] = this.dz;
    }
    if (this.rotateflag) {
      this.rpoint[0] = this.point[0] + this.dx;
      this.rpoint[1] = this.point[1] + this.dy;
      this.rpoint[2] = this.point[2] + this.dz;
      if (step > 0) {
        const angvel = (this.theta - prev[3]) / dt;
        this.omega[0] = angvel * this.axis[0];
        this.omega[1] = angvel * this.axis[1];
        this.omega[2] = angvel * this.axis[2];
      } else this.omega[0] = this.omega[1] = this.omega[2] = 0.0;
      prev[3] = this.theta;
    }
    if (this.varshape) this.set_velocity_shape(step);
  }

  /** The contact point's xc of velocity_contact (native leaves it unset without rotation; it is then not read). */
  private readonly xcv = new Float64Array(3);

  /** Region::velocity_contact: the wall velocity at contact ic of the particle at x. */
  velocity_contact(vwall: Float64Array, x: ArrayLike<number>, ic: number, dt: number): void {
    vwall[0] = vwall[1] = vwall[2] = 0.0;
    if (this.moveflag) {
      vwall[0] = this.v[0];
      vwall[1] = this.v[1];
      vwall[2] = this.v[2];
    }
    const xc = this.xcv;
    if (this.rotateflag) {
      xc[0] = x[0] - this.cDelx[ic];
      xc[1] = x[1] - this.cDely[ic];
      xc[2] = x[2] - this.cDelz[ic];
      const omega = this.omega, rpoint = this.rpoint;
      vwall[0] += omega[1] * (xc[2] - rpoint[2]) - omega[2] * (xc[1] - rpoint[1]);
      vwall[1] += omega[2] * (xc[0] - rpoint[0]) - omega[0] * (xc[2] - rpoint[2]);
      vwall[2] += omega[0] * (xc[1] - rpoint[1]) - omega[1] * (xc[0] - rpoint[0]);
    }
    if (this.varshape && this.cVarflag[ic]) this.velocity_contact_shape(vwall, xc, dt);
  }

  reset_vel(): void {
    this.prev.fill(0);
  }
}
