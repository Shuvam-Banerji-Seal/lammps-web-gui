import { StyleError } from './force/types';

/*
 * Geometric regions — docs.lammps.org/region.html.
 *
 * "coordinates exactly on the region boundary are considered to be interior
 * to the region. This means, for example, for a spherical region, an atom on
 * the sphere surface would be part of the region if the sphere were defined
 * with the side in keyword, but would not be part of the region if it were
 * defined using the side out keyword."
 * "INF means a large negative or positive number (1.0e20)"; "EDGE means they
 * extend all the way to the global simulation box boundary".
 * "Regions in LAMMPS do not get wrapped across periodic boundaries".
 * Styles: block, sphere, cylinder, cone, ellipsoid, plane, prism, union,
 * intersect. Parameters may be equal-style variables (v_name), evaluated each
 * time the region is used. move/rotate: "The move keyword allows one or more
 * equal-style variables to be used to specify the x,y,z displacement of the
 * region"; "The rotate keyword rotates the region around a rotation axis
 * R = (Rx,Ry,Rz) that goes through a point P = (Px,Py,Pz) ... in radians ...
 * consistent with the right-hand rule"; "the displacement specified by the
 * move keyword is applied to the P point of the rotate keyword".
 */

export const BIG = 1.0e20;

/** A parameter: a constant or an equal-style variable (with a scale factor). */
export type Param = number | { variable: string; scale: number };

/**
 * One wall/particle contact point of a region surface (fix wall/region and
 * fix wall/gran/region).  "The distance between a particle and the region
 * boundary is the distance to the nearest point on the region surface.  The
 * force the wall exerts on the particle is along the direction between that
 * point and the particle center, which is the direction normal to the surface
 * at that point." (docs.lammps.org/fix_wall_gran_region.html)
 * For a compound region the contacts are those of its primitive sub-regions,
 * filtered so that internal faces are dropped: "LAMMPS discards points that
 * are part of multiple sub-regions when calculating wall/particle
 * interactions, to avoid double-counting the interaction." (region.html)
 */
export interface SurfaceContact {
  /** Stable identity of the face (granular shear-history key). */
  key: string;
  /** Distance from the particle center to the surface point (>= 0 inside). */
  dist: number;
  /** Unit normal from the surface point toward the particle (lab frame). */
  nx: number; ny: number; nz: number;
  /** Signed radius of curvature of the wall at the contact (0 = flat). */
  curvature: number;
  /** The primitive sub-region that produced this contact (for its motion). */
  source: Region;
}

export interface RegionEnv {
  /** Evaluates an equal-style variable. */
  variable(name: string): number;
  /** Looks up another region (union / intersect). */
  region(id: string): Region | undefined;
  /** Maps a point into the periodic box (absent before the box exists). */
  remap?(p: number[]): void;
}

export abstract class Region {
  interior = true;
  /** move: displacement variables (NULL = 0). */
  move: [string | null, string | null, string | null] | null = null;
  /** rotate: theta variable, point P, unit axis R. */
  rotate: { theta: string; p: [number, number, number]; r: [number, number, number] } | null = null;
  openFaces: number[] = [];
  private readonly remapBuf = [0, 0, 0];

  constructor(readonly id: string, readonly style: string, protected env: RegionEnv) {}

  /** True if the point is inside the geometry (boundary included). */
  protected abstract inside(x: number, y: number, z: number): boolean;

  get dynamic(): boolean { return this.move !== null || this.rotate !== null; }

  /**
   * True if (x, y, z) belongs to the region, honouring side and move/rotate. Measured with native
   * LAMMPS (black box): the point is first mapped into the periodic box — an atom that has
   * crossed a periodic face since the last reneighboring (stored x = -0.0033 in a box 0..6.72)
   * is tested at x + L by count(group,region), compute reduce/region and dynamic groups.
   */
  match(x: number, y: number, z: number): boolean {
    if (this.env.remap) {
      const p = this.remapBuf;
      p[0] = x; p[1] = y; p[2] = z;
      this.env.remap(p);
      [x, y, z] = p;
    }
    if (this.dynamic) [x, y, z] = this.toBodyFrame(x, y, z);
    const ins = this.inside(x, y, z);
    return this.interior ? ins : !ins;
  }

  /** Inverse of the region's current displacement and rotation. */
  private toBodyFrame(x: number, y: number, z: number): [number, number, number] {
    let d: [number, number, number] = [0, 0, 0];
    if (this.move) d = this.move.map((v) => (v ? this.env.variable(v) : 0)) as [number, number, number];
    let px = x - d[0], py = y - d[1], pz = z - d[2];
    if (this.rotate) {
      const th = -this.env.variable(this.rotate.theta);
      const [ox, oy, oz] = this.rotate.p;
      const [ux, uy, uz] = this.rotate.r;
      // rotation of (p - P) by -theta about R (Rodrigues)
      const vx = px - ox, vy = py - oy, vz = pz - oz;
      const c = Math.cos(th), s = Math.sin(th);
      const dot = ux * vx + uy * vy + uz * vz;
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      px = ox + vx * c + cx * s + ux * dot * (1 - c);
      py = oy + vy * c + cy * s + uy * dot * (1 - c);
      pz = oz + vz * c + cz * s + uz * dot * (1 - c);
    }
    return [px, py, pz];
  }

  /** A parameter's current value. */
  protected val(p: Param): number {
    return typeof p === 'number' ? p : this.env.variable(p.variable) * p.scale;
  }

  /**
   * Current rigid motion of the region: the move displacement and the rotate
   * angle.  "If the move or rotate keywords are used, the region is dynamic,
   * meaning its location or orientation changes with time." (region.html)  The
   * wall fixes difference this over a timestep to get the surface velocity
   * that enters the granular contact.
   */
  transformState(): { d: [number, number, number]; theta: number } {
    const d = this.move
      ? (this.move.map((v) => (v ? this.env.variable(v) : 0)) as [number, number, number])
      : ([0, 0, 0] as [number, number, number]);
    const theta = this.rotate ? this.env.variable(this.rotate.theta) : 0;
    return { d, theta };
  }

  /** Rotate a body-frame vector by +theta about the region axis (Rodrigues). */
  protected toLabVector(vx: number, vy: number, vz: number): [number, number, number] {
    if (!this.rotate) return [vx, vy, vz];
    const th = this.env.variable(this.rotate.theta);
    const [ux, uy, uz] = this.rotate.r;
    const c = Math.cos(th), s = Math.sin(th);
    const dot = ux * vx + uy * vy + uz * vz;
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    return [vx * c + cx * s + ux * dot * (1 - c), vy * c + cy * s + uy * dot * (1 - c), vz * c + cz * s + uz * dot * (1 - c)];
  }

  /** Maps a lab-frame point into the region's body frame (identity when static). */
  bodyPoint(x: number, y: number, z: number): [number, number, number] {
    return this.dynamic ? this.toBodyFrame(x, y, z) : [x, y, z];
  }

  /** Maps a body-frame point back to the lab frame (move applied after rotate). */
  labPoint(bx: number, by: number, bz: number): [number, number, number] {
    if (!this.dynamic) return [bx, by, bz];
    let px = bx, py = by, pz = bz;
    if (this.rotate) {
      const th = this.env.variable(this.rotate.theta);
      const [ox, oy, oz] = this.rotate.p;
      const [ux, uy, uz] = this.rotate.r;
      const vx = bx - ox, vy = by - oy, vz = bz - oz;
      const c = Math.cos(th), s = Math.sin(th);
      const dot = ux * vx + uy * vy + uz * vz;
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      px = ox + vx * c + cx * s + ux * dot * (1 - c);
      py = oy + vy * c + cy * s + uy * dot * (1 - c);
      pz = oz + vz * c + cz * s + uz * dot * (1 - c);
    }
    const d = this.move ? this.move.map((v) => (v ? this.env.variable(v) : 0)) : [0, 0, 0];
    return [px + d[0], py + d[1], pz + d[2]];
  }

  /**
   * Surface contacts the region's interior surface exerts on a particle at the
   * lab-frame point (x, y, z).  A primitive returns its faces (the wall fixes
   * apply each one: "if the region surface is comprised of multiple faces,
   * then each face can exert a force on the particle if it is close enough").
   * A point outside the region yields no contact.
   */
  contacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    let px = x, py = y, pz = z;
    if (this.env.remap) {
      const p = [px, py, pz];
      this.env.remap(p);
      px = p[0]; py = p[1]; pz = p[2];
    }
    const [bx, by, bz] = this.dynamic ? this.toBodyFrame(px, py, pz) : [px, py, pz];
    const body: SurfaceContact[] = [];
    this.primitiveContacts(bx, by, bz, body);
    if (!this.dynamic) { for (const c of body) out.push(c); return; }
    for (const c of body) {
      const [nx, ny, nz] = this.toLabVector(c.nx, c.ny, c.nz);
      out.push({ key: c.key, dist: c.dist, nx, ny, nz, curvature: c.curvature, source: c.source });
    }
  }

  /** The interior faces of a primitive region in its body frame (no motion). */
  protected primitiveContacts(_x: number, _y: number, _z: number, _out: SurfaceContact[]): void {
    throw new StyleError(`region style ${this.style} cannot be used as a wall`);
  }

  /**
   * Contact candidates of the region surface for fix wall/gran/region.  A
   * side-in region returns its interior faces (contacts); a side-out region
   * returns the nearest point of the solid (its exterior surface).  A
   * compound region overrides this to filter the sub-region contacts as
   * region.html describes.  The force direction is "along the direction
   * between that point and the particle center, which is the direction normal
   * to the surface at that point" (fix_wall_gran_region.rst).
   */
  surfaceContacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    if (this.interior) this.contacts(x, y, z, out);
    else this.outContacts(x, y, z, out);
  }

  /**
   * Side-out surface contacts (nearest point of the solid) in the lab frame.
   * Measured with native LAMMPS (black box, fix wall/gran/region, single
   * sphere radius 0.5 against a side-out block 0..2^3, hooke and
   * hertz/history): a particle outside a face, an edge, or a corner feels a
   * single contact at the nearest point of the solid, with the normal along
   * (particle - nearest point) and the block faces, edges and corners all
   * flat (R_eff = R, the edge and corner forces are the face-like
   * kn * delta * sqrt(delta R)); a side-out sphere of radius 1 gives R_eff =
   * R * Rw / (R + Rw) with Rw = +1 (the convex outer surface); a side-out
   * cylinder (radius 1) gives Rw = +2 at the lateral surface and at the rim,
   * and Rw = 0 at the flat caps.
   */
  outContacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    let px = x, py = y, pz = z;
    if (this.env.remap) {
      const p = [px, py, pz];
      this.env.remap(p);
      px = p[0]; py = p[1]; pz = p[2];
    }
    const [bx, by, bz] = this.dynamic ? this.toBodyFrame(px, py, pz) : [px, py, pz];
    const body: SurfaceContact[] = [];
    this.primitiveSideOutContacts(bx, by, bz, body);
    if (!this.dynamic) { for (const c of body) out.push(c); return; }
    for (const c of body) {
      const [nx, ny, nz] = this.toLabVector(c.nx, c.ny, c.nz);
      out.push({ key: c.key, dist: c.dist, nx, ny, nz, curvature: c.curvature, source: c.source });
    }
  }

  /**
   * Nearest-point (side-out) contacts of a primitive region in its body frame.
   * Measured with native LAMMPS (black box): block, sphere, cylinder (radlo =
   * radhi) and cone solids use the nearest point of the solid; a particle on
   * or inside the solid yields no exterior contact.
   */
  protected primitiveSideOutContacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    if (this instanceof BlockRegion) {
      const b = this.b.map((p) => this.val(p));
      const lo = [b[0], b[2], b[4]], hi = [b[1], b[3], b[5]];
      const p = [x, y, z];
      let inside = true;
      for (let a = 0; a < 3; a++) if (p[a] < lo[a] || p[a] > hi[a]) inside = false;
      if (inside) return;
      const s = [0, 0, 0];
      const act: string[] = [];
      for (let a = 0; a < 3; a++) {
        if (p[a] < lo[a]) { s[a] = lo[a]; act.push(`${a}lo`); }
        else if (p[a] > hi[a]) { s[a] = hi[a]; act.push(`${a}hi`); }
        else s[a] = p[a];
      }
      const dx = x - s[0], dy = y - s[1], dz = z - s[2];
      const d = Math.hypot(dx, dy, dz);
      if (!(d > 0)) return;
      out.push({ key: `out:${act.join('')}`, dist: d, nx: dx / d, ny: dy / d, nz: dz / d, curvature: 0, source: this });
      return;
    }
    if (this instanceof SphereRegion) {
      const c = this.c.map((p) => this.val(p));
      const R = this.val(this.r);
      const dx = x - c[0], dy = y - c[1], dz = z - c[2];
      const rho = Math.hypot(dx, dy, dz);
      if (rho <= R) return;
      out.push({ key: 'out', dist: rho - R, nx: dx / rho, ny: dy / rho, nz: dz / rho, curvature: R, source: this });
      return;
    }
    if (this instanceof ConeRegion) {
      const axis = this.axis;
      const c1 = this.val(this.c1), c2 = this.val(this.c2);
      const rl = this.val(this.radlo), rh = this.val(this.radhi);
      const lo = this.val(this.lo), hi = this.val(this.hi);
      const p = [x, y, z];
      const [d1, d2] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
      const e1 = p[d1] - c1, e2 = p[d2] - c2;
      const rho = Math.hypot(e1, e2);
      const ac = Math.min(Math.max(p[axis], lo), hi);
      const slope = hi > lo ? (rh - rl) / (hi - lo) : 0;
      const rad = rl + slope * (ac - lo);
      const curved = rho > rad;
      const rc = Math.min(rho, rad);
      const s = [0, 0, 0];
      s[axis] = ac;
      s[d1] = c1 + (rho > 0 ? (e1 * rc) / rho : 0);
      s[d2] = c2 + (rho > 0 ? (e2 * rc) / rho : 0);
      const dx = x - s[0], dy = y - s[1], dz = z - s[2];
      const d = Math.hypot(dx, dy, dz);
      if (!(d > 0)) return;
      out.push({
        key: curved ? 'outlat' : 'outcap', dist: d, nx: dx / d, ny: dy / d, nz: dz / d,
        curvature: curved ? 2 * rc : 0, source: this,
      });
      return;
    }
    throw new StyleError(`region style ${this.style} cannot be used as a side-out wall`);
  }

  /** The primitive sub-regions of a compound region (empty for a primitive). */
  subRegions(): Region[] { return []; }

  /**
   * True if the region or any sub-region is side out.  The interior-surface
   * helper `contacts` models side-in regions only, so fix wall/region rejects
   * a compound region with a side-out sub-region.  fix wall/gran/region uses
   * `surfaceContacts`, which handles side-out sub-regions (region.ts).
   */
  hasSideOutSubRegion(): boolean {
    return this.subRegions().some((r) => !r.interior || r.hasSideOutSubRegion());
  }

  /** Axis-aligned bounding box of the interior, if finite (for create_atoms). */
  bbox(): { lo: number[]; hi: number[] } | null { return null; }
}

export class BlockRegion extends Region {
  constructor(id: string, env: RegionEnv, readonly b: Param[]) { super(id, 'block', env); }
  protected inside(x: number, y: number, z: number): boolean {
    const b = this.b.map((p) => this.val(p));
    return x >= b[0] && x <= b[1] && y >= b[2] && y <= b[3] && z >= b[4] && z <= b[5];
  }
  protected primitiveContacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    const b = this.b.map((p) => this.val(p));
    const lo = [b[0], b[2], b[4]], hi = [b[1], b[3], b[5]];
    const p = [x, y, z];
    for (let a = 0; a < 3; a++) if (p[a] < lo[a] || p[a] > hi[a]) return;
    for (let a = 0; a < 3; a++) {
      const e = [0, 0, 0];
      e[a] = 1;
      out.push({ key: `${a}lo`, dist: p[a] - lo[a], nx: e[0], ny: e[1], nz: e[2], curvature: 0, source: this });
      out.push({ key: `${a}hi`, dist: hi[a] - p[a], nx: -e[0], ny: -e[1], nz: -e[2], curvature: 0, source: this });
    }
  }
  bbox() {
    if (!this.interior || this.dynamic) return null;
    const b = this.b.map((p) => this.val(p));
    return { lo: [b[0], b[2], b[4]], hi: [b[1], b[3], b[5]] };
  }
}

export class SphereRegion extends Region {
  constructor(id: string, env: RegionEnv, readonly c: Param[], readonly r: Param) { super(id, 'sphere', env); }
  protected inside(x: number, y: number, z: number): boolean {
    const [cx, cy, cz] = this.c.map((p) => this.val(p));
    const r = this.val(this.r);
    const dx = x - cx, dy = y - cy, dz = z - cz;
    return dx * dx + dy * dy + dz * dz <= r * r;
  }
  protected primitiveContacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    const [cx, cy, cz] = this.c.map((p) => this.val(p));
    const R = this.val(this.r);
    const dx = x - cx, dy = y - cy, dz = z - cz;
    const rho = Math.hypot(dx, dy, dz);
    if (rho > R) return;
    // concave wall as seen from the interior: radius of curvature -R (measured
    // with native LAMMPS: R_eff = R Rw / (R + Rw) with Rw = -R)
    if (rho === 0) { out.push({ key: 'sphere', dist: R, nx: 0, ny: 0, nz: 0, curvature: -R, source: this }); return; }
    out.push({ key: 'sphere', dist: R - rho, nx: -dx / rho, ny: -dy / rho, nz: -dz / rho, curvature: -R, source: this });
  }
}

export class EllipsoidRegion extends Region {
  constructor(id: string, env: RegionEnv, readonly c: Param[], readonly ax: Param[]) { super(id, 'ellipsoid', env); }
  protected inside(x: number, y: number, z: number): boolean {
    const [cx, cy, cz] = this.c.map((p) => this.val(p));
    const [a, b, c] = this.ax.map((p) => this.val(p));
    const dx = (x - cx) / a, dy = (y - cy) / b, dz = (z - cz) / c;
    return dx * dx + dy * dy + dz * dz <= 1;
  }
}

/** cylinder (radlo = radhi) and cone. */
export class ConeRegion extends Region {
  constructor(
    id: string, style: 'cylinder' | 'cone', env: RegionEnv, readonly axis: 0 | 1 | 2,
    readonly c1: Param, readonly c2: Param, readonly radlo: Param, readonly radhi: Param, readonly lo: Param, readonly hi: Param,
  ) { super(id, style, env); }
  protected inside(x: number, y: number, z: number): boolean {
    const p = [x, y, z];
    const a = p[this.axis];
    const [d1, d2] = this.axis === 0 ? [1, 2] : this.axis === 1 ? [0, 2] : [0, 1];
    const lo = this.val(this.lo), hi = this.val(this.hi);
    if (a < lo || a > hi) return false;
    const r1 = this.val(this.radlo), r2 = this.val(this.radhi);
    const r = hi > lo ? r1 + (r2 - r1) * (a - lo) / (hi - lo) : Math.max(r1, r2);
    const e1 = p[d1] - this.val(this.c1), e2 = p[d2] - this.val(this.c2);
    return e1 * e1 + e2 * e2 <= r * r;
  }
  protected primitiveContacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    // lateral surface: the generator radius rho(a) = rl + slope (a - lo) (a
    // cylinder has slope 0).  Measured with native LAMMPS (black box, single
    // sphere, Hertz k_n): the overlap is the distance to the generator, the
    // normal is the gradient of the generator, and the curvature radius of the
    // wall at the contact point is Rw = -2 rho_s, rho_s = radial distance of
    // the surface point (cylinder: rho_s = Rc; cones: 2.94 = 2 x 1.47 and
    // 3.44 = 2 x 1.72 at two points).
    const rl = this.val(this.radlo), rh = this.val(this.radhi);
    const axis = this.axis;
    const c1 = this.val(this.c1), c2 = this.val(this.c2), lo = this.val(this.lo), hi = this.val(this.hi);
    const p = [x, y, z];
    const a = p[axis];
    if (a < lo || a > hi) return;
    const [d1, d2] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
    const e1 = p[d1] - c1, e2 = p[d2] - c2;
    const rho = Math.hypot(e1, e2);
    const slope = hi > lo ? (rh - rl) / (hi - lo) : 0;
    const rad = rl + slope * (a - lo);
    if (rho > rad) return;
    const f = rho - rad;
    const q = 1 + slope * slope;
    if (rho > 0) {
      const u = [0, 0, 0];
      u[d1] = (-e1 / rho) / Math.sqrt(q);
      u[d2] = (-e2 / rho) / Math.sqrt(q);
      u[axis] = slope / Math.sqrt(q);
      const rhoS = rho - f / q;
      out.push({ key: 'lat', dist: -f / Math.sqrt(q), nx: u[0], ny: u[1], nz: u[2], curvature: -2 * rhoS, source: this });
    }
    const ax = [0, 0, 0];
    ax[axis] = 1;
    out.push({ key: 'lo', dist: a - lo, nx: ax[0], ny: ax[1], nz: ax[2], curvature: 0, source: this });
    out.push({ key: 'hi', dist: hi - a, nx: -ax[0], ny: -ax[1], nz: -ax[2], curvature: 0, source: this });
  }
  /**
   * Axis-aligned bounding box, as create_box.html describes for regions other than prism. Measured with
   * native LAMMPS (black box): region cylinder y 0 0 0.005 -0.005 0 gives the box (-0.005 -0.005 -0.005)
   * to (0.005 0 0.005).
   */
  bbox(): { lo: number[]; hi: number[] } | null {
    if (!this.interior || this.dynamic) return null;
    const lo = [0, 0, 0], hi = [0, 0, 0];
    const [d1, d2] = this.axis === 0 ? [1, 2] : this.axis === 1 ? [0, 2] : [0, 1];
    const a0 = this.val(this.lo), a1 = this.val(this.hi);
    const r = Math.max(Math.abs(this.val(this.radlo)), Math.abs(this.val(this.radhi)));
    const c1 = this.val(this.c1), c2 = this.val(this.c2);
    lo[this.axis] = Math.min(a0, a1); hi[this.axis] = Math.max(a0, a1);
    lo[d1] = c1 - r; hi[d1] = c1 + r;
    lo[d2] = c2 - r; hi[d2] = c2 + r;
    return { lo, hi };
  }
}

export class PlaneRegion extends Region {
  constructor(id: string, env: RegionEnv, readonly pt: Param[], readonly nrm: Param[]) { super(id, 'plane', env); }
  /** "The inside of the plane is the half-space in the direction of the normal vector". */
  protected inside(x: number, y: number, z: number): boolean {
    const [px, py, pz] = this.pt.map((p) => this.val(p));
    const [nx, ny, nz] = this.nrm.map((p) => this.val(p));
    return (x - px) * nx + (y - py) * ny + (z - pz) * nz >= 0;
  }
  /**
   * The single flat face of the plane.  "For style *plane*, a plane is defined
   * which contain the point (px,py,pz) and has a normal vector (nx,ny,nz).  The
   * normal vector does not have to be of unit length.  The "inside" of the
   * plane is the half-space in the direction of the normal vector"
   * (docs.lammps.org/region.html).  The distance to the surface is the signed
   * distance of the atom from the plane, (x-p) . n / |n|, and the normal
   * pointing into the interior is n / |n|.  The wall is flat, so its radius of
   * curvature is 0: "For a flat wall, delta = radius - r = overlap of particle
   * with wall, m_eff = mass of particle, and the effective radius of contact is
   * just the radius of the particle." (docs.lammps.org/fix_wall_gran_region.html)
   * Measured with native LAMMPS (black box, single atom at (1.6,1.5,1.7) and
   * region plane 1 1 1 1 1 1 side in, fix wall/region harmonic 1.0 0.0 2.5):
   * r = 1.8/sqrt(3) = 1.0392304845413265 and the force is
   * 2(2.5 - r) n_hat = (1.68675134594813, 1.68675134594813, 1.68675134594813),
   * i.e. the projection onto the unit normal.
   */
  protected primitiveContacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    const [px, py, pz] = this.pt.map((p) => this.val(p));
    const [nx, ny, nz] = this.nrm.map((p) => this.val(p));
    const n = Math.hypot(nx, ny, nz);
    const dist = ((x - px) * nx + (y - py) * ny + (z - pz) * nz) / n;
    if (dist < 0) return;
    out.push({ key: 'plane', dist, nx: nx / n, ny: ny / n, nz: nz / n, curvature: 0, source: this });
  }
}

/** prism: origin (xlo, ylo, zlo), edges A = (xhi-xlo,0,0), B = (xy,yhi-ylo,0), C = (xz,yz,zhi-zlo). */
export class PrismRegion extends Region {
  constructor(id: string, env: RegionEnv, readonly b: Param[]) { super(id, 'prism', env); }
  protected inside(x: number, y: number, z: number): boolean {
    const [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz] = this.b.map((p) => this.val(p));
    const lx = xhi - xlo, ly = yhi - ylo, lz = zhi - zlo;
    const c = (z - zlo) / lz;
    const b = (y - ylo - c * yz) / ly;
    const a = (x - xlo - b * xy - c * xz) / lx;
    return a >= 0 && a <= 1 && b >= 0 && b <= 1 && c >= 0 && c <= 1;
  }
  /** The prism's parameters (create_box with a prism makes a triclinic box). */
  values(): number[] { return this.b.map((p) => this.val(p)); }
}

export class CompoundRegion extends Region {
  constructor(id: string, style: 'union' | 'intersect', env: RegionEnv, readonly members: string[]) { super(id, style, env); }
  subRegions(): Region[] {
    return this.members.map((m) => {
      const r = this.env.region(m);
      if (!r) throw new StyleError(`region ${this.id}: sub-region ${m} no longer exists`);
      return r;
    });
  }
  protected inside(x: number, y: number, z: number): boolean {
    const regs = this.members.map((m) => {
      const r = this.env.region(m);
      if (!r) throw new StyleError(`region ${this.id}: sub-region ${m} no longer exists`);
      return r;
    });
    return this.style === 'union' ? regs.some((r) => r.match(x, y, z)) : regs.every((r) => r.match(x, y, z));
  }
  /**
   * Contacts of the union/intersect surface.  Measured with native LAMMPS
   * (black box, fix wall/gran/region granular hooke, blocks A = [0,2]^3 and
   * B = [1,3]x[0,2]x[0,2], particle radius 0.5):
   * - a sub-region contributes only when the particle is inside it;
   * - intersect keeps a face contact only when its contact point lies inside
   *   every sub-region (A's xlo and B's xhi faces were dropped, the shared
   *   ylo/yhi/zlo/zhi faces were counted once per sub-region);
   * - union keeps a face contact only when its contact point lies inside no
   *   other sub-region (the shared faces became invisible, matching the note
   *   "Having two coincident faces could cause the face to become invisible
   *   to the particles." on region.html).
   */
  contacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    let px = x, py = y, pz = z;
    if (this.env.remap) {
      const p = [px, py, pz];
      this.env.remap(p);
      px = p[0]; py = p[1]; pz = p[2];
    }
    const members = this.members.map((m) => {
      const r = this.env.region(m);
      if (!r) throw new StyleError(`region ${this.id}: sub-region ${m} no longer exists`);
      return r;
    });
    for (const m of members) {
      if (!m.match(px, py, pz)) continue;
      const sub: SurfaceContact[] = [];
      m.contacts(px, py, pz, sub);
      for (const c of sub) {
        const qx = px - c.dist * c.nx, qy = py - c.dist * c.ny, qz = pz - c.dist * c.nz;
        const keep = this.style === 'intersect'
          ? members.every((mm) => mm.match(qx, qy, qz))
          : members.every((mm) => mm === m || !mm.match(qx, qy, qz));
        if (keep) out.push(c);
      }
    }
  }

  /**
   * Compound contacts including side-out sub-regions.  The sub-region faces are
   * filtered as docs.lammps.org/region.html describes: "LAMMPS discards points
   * that are part of multiple sub-regions when calculating wall/particle
   * interactions, to avoid double-counting the interaction."  A contact's own
   * sub-region is not re-tested: a side-out sub-region's nearest point lies on
   * its boundary, which is not "part of" a side-out region ("coordinates
   * exactly on the region boundary are considered to be interior to the
   * region ... would not be part of the region if it were defined using the
   * side out keyword", region.html).  For an intersect region the contact
   * point must lie in every other sub-region; for a union it must lie in no
   * other sub-region.
   */
  surfaceContacts(x: number, y: number, z: number, out: SurfaceContact[]): void {
    let px = x, py = y, pz = z;
    if (this.env.remap) {
      const p = [px, py, pz];
      this.env.remap(p);
      px = p[0]; py = p[1]; pz = p[2];
    }
    const members = this.members.map((m) => {
      const r = this.env.region(m);
      if (!r) throw new StyleError(`region ${this.id}: sub-region ${m} no longer exists`);
      return r;
    });
    for (const m of members) {
      if (!m.match(px, py, pz)) continue;
      const sub: SurfaceContact[] = [];
      m.surfaceContacts(px, py, pz, sub);
      for (const c of sub) {
        const qx = px - c.dist * c.nx, qy = py - c.dist * c.ny, qz = pz - c.dist * c.nz;
        const keep = this.style === 'intersect'
          ? members.every((mm) => mm === m || mm.match(qx, qy, qz))
          : members.every((mm) => mm === m || !mm.match(qx, qy, qz));
        if (keep) out.push(c);
      }
    }
  }
}
