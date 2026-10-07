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

export interface RegionEnv {
  /** Evaluates an equal-style variable. */
  variable(name: string): number;
  /** Looks up another region (union / intersect). */
  region(id: string): Region | undefined;
}

export abstract class Region {
  interior = true;
  /** move: displacement variables (NULL = 0). */
  move: [string | null, string | null, string | null] | null = null;
  /** rotate: theta variable, point P, unit axis R. */
  rotate: { theta: string; p: [number, number, number]; r: [number, number, number] } | null = null;
  openFaces: number[] = [];

  constructor(readonly id: string, readonly style: string, protected env: RegionEnv) {}

  /** True if the point is inside the geometry (boundary included). */
  protected abstract inside(x: number, y: number, z: number): boolean;

  get dynamic(): boolean { return this.move !== null || this.rotate !== null; }

  /** True if (x, y, z) belongs to the region, honouring side and move/rotate. */
  match(x: number, y: number, z: number): boolean {
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

  /** Axis-aligned bounding box of the interior, if finite (for create_atoms). */
  bbox(): { lo: number[]; hi: number[] } | null { return null; }
}

export class BlockRegion extends Region {
  constructor(id: string, env: RegionEnv, readonly b: Param[]) { super(id, 'block', env); }
  protected inside(x: number, y: number, z: number): boolean {
    const b = this.b.map((p) => this.val(p));
    return x >= b[0] && x <= b[1] && y >= b[2] && y <= b[3] && z >= b[4] && z <= b[5];
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
}

export class PlaneRegion extends Region {
  constructor(id: string, env: RegionEnv, readonly pt: Param[], readonly nrm: Param[]) { super(id, 'plane', env); }
  /** "The inside of the plane is the half-space in the direction of the normal vector". */
  protected inside(x: number, y: number, z: number): boolean {
    const [px, py, pz] = this.pt.map((p) => this.val(p));
    const [nx, ny, nz] = this.nrm.map((p) => this.val(p));
    return (x - px) * nx + (y - py) * ny + (z - pz) * nz >= 0;
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
  protected inside(x: number, y: number, z: number): boolean {
    const regs = this.members.map((m) => {
      const r = this.env.region(m);
      if (!r) throw new StyleError(`region ${this.id}: sub-region ${m} no longer exists`);
      return r;
    });
    return this.style === 'union' ? regs.some((r) => r.match(x, y, z)) : regs.every((r) => r.match(x, y, z));
  }
}
