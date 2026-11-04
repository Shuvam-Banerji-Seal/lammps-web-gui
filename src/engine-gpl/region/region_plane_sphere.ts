/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/region_plane.cpp, src/region_plane.h, src/region_sphere.cpp and src/region_sphere.h (RegPlane and RegSphere:
 * the unit normal of the constructor, inside, surface_interior, surface_exterior, shape_update, set_velocity_shape and
 * velocity_contact_shape) at commit b332f28. Copyright (2003) Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { NativeRegion, type NativeRegionEnv } from './native_region';
import type { Param } from '../../engine/region';

const val = (env: NativeRegionEnv, p: Param): number => (typeof p === 'number' ? p : p.scale * env.variable(p.variable));

export class NativePlane extends NativeRegion {
  xp = 0; yp = 0; zp = 0;
  readonly normal = new Float64Array(3);
  private readonly nVariable: boolean;

  /** pt: the point, nrm: the normal vector as the engine parsed them (numbers scaled by the lattice, or variables). */
  constructor(id: string, env: NativeRegionEnv, private readonly pt: readonly Param[], private readonly nrm: readonly Param[]) {
    super(id, 'plane', env);
    this.nVariable = typeof nrm[0] !== 'number';
    if (pt.some((v) => typeof v !== 'number') || this.nVariable) this.varshape = 1;
    this.allocContacts(1);
  }

  setup(): void {
    if (typeof this.pt[0] === 'number') this.xp = this.pt[0];
    if (typeof this.pt[1] === 'number') this.yp = this.pt[1];
    if (typeof this.pt[2] === 'number') this.zp = this.pt[2];
    if (!this.nVariable) for (let d = 0; d < 3; d++) this.normal[d] = this.nrm[d] as number;
    if (this.varshape) this.shape_update();
    // enforce unit normal: the constructor normalizes after its shape_update whatever the normal's style (a variable
    // normal, already normalized by shape_update, is normalized again until the next shape_update)
    this.unitNormal();
    this.tmax = 1;
  }

  private unitNormal(): void {
    const n = this.normal;
    const rsq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    n[0] /= Math.sqrt(rsq);
    n[1] /= Math.sqrt(rsq);
    n[2] /= Math.sqrt(rsq);
  }

  inside(x: number, y: number, z: number): number {
    const n = this.normal;
    const dot = (x - this.xp) * n[0] + (y - this.yp) * n[1] + (z - this.zp) * n[2];
    if (dot >= 0.0) return 1;
    return 0;
  }

  surface_interior(x: Float64Array, cutoff: number): number {
    const n = this.normal;
    const dot = (x[0] - this.xp) * n[0] + (x[1] - this.yp) * n[1] + (x[2] - this.zp) * n[2];
    if (dot < cutoff && dot >= 0.0) {
      this.cR[0] = dot;
      this.cDelx[0] = dot * n[0];
      this.cDely[0] = dot * n[1];
      this.cDelz[0] = dot * n[2];
      this.cRadius[0] = 0;
      this.cIwall[0] = 0;
      return 1;
    }
    return 0;
  }

  surface_exterior(x: Float64Array, cutoff: number): number {
    const n = this.normal;
    let dot = (x[0] - this.xp) * n[0] + (x[1] - this.yp) * n[1] + (x[2] - this.zp) * n[2];
    dot = -dot;
    if (dot < cutoff && dot >= 0.0) {
      this.cR[0] = dot;
      this.cDelx[0] = -dot * n[0];
      this.cDely[0] = -dot * n[1];
      this.cDelz[0] = -dot * n[2];
      this.cRadius[0] = 0;
      this.cIwall[0] = 0;
      return 1;
    }
    return 0;
  }

  shape_update(): void {
    if (typeof this.pt[0] !== 'number') this.xp = val(this.env, this.pt[0]);
    if (typeof this.pt[1] !== 'number') this.yp = val(this.env, this.pt[1]);
    if (typeof this.pt[2] !== 'number') this.zp = val(this.env, this.pt[2]);
    if (this.nVariable) {
      for (let d = 0; d < 3; d++) this.normal[d] = val(this.env, this.nrm[d]);
      this.unitNormal();
    }
  }
}

export class NativeSphere extends NativeRegion {
  xc = 0; yc = 0; zc = 0; radius = 0;

  /** c: the centre, r: the radius as the engine parsed them (numbers scaled by the lattice, or variables). */
  constructor(id: string, env: NativeRegionEnv, private readonly c: readonly Param[], private readonly r: Param) {
    super(id, 'sphere', env);
    if (c.some((v) => typeof v !== 'number') || typeof r !== 'number') this.varshape = 1;
    this.allocContacts(1);
  }

  setup(): void {
    if (typeof this.c[0] === 'number') this.xc = this.c[0];
    if (typeof this.c[1] === 'number') this.yc = this.c[1];
    if (typeof this.c[2] === 'number') this.zc = this.c[2];
    if (typeof this.r === 'number') this.radius = this.r;
    if (this.varshape) this.shape_update();
    this.tmax = 1;
  }

  inside(x: number, y: number, z: number): number {
    const delx = x - this.xc;
    const dely = y - this.yc;
    const delz = z - this.zc;
    const r = Math.sqrt(delx * delx + dely * dely + delz * delz);
    if (r <= this.radius) return 1;
    return 0;
  }

  surface_interior(x: Float64Array, cutoff: number): number {
    const radius = this.radius;
    const delx = x[0] - this.xc;
    const dely = x[1] - this.yc;
    const delz = x[2] - this.zc;
    const r = Math.sqrt(delx * delx + dely * dely + delz * delz);
    if (r > radius || r === 0.0) return 0;
    const delta = radius - r;
    if (delta < cutoff) {
      this.cR[0] = delta;
      this.cDelx[0] = delx * (1.0 - radius / r);
      this.cDely[0] = dely * (1.0 - radius / r);
      this.cDelz[0] = delz * (1.0 - radius / r);
      this.cRadius[0] = -radius;
      this.cIwall[0] = 0;
      this.cVarflag[0] = 1;
      return 1;
    }
    return 0;
  }

  surface_exterior(x: Float64Array, cutoff: number): number {
    const radius = this.radius;
    const delx = x[0] - this.xc;
    const dely = x[1] - this.yc;
    const delz = x[2] - this.zc;
    const r = Math.sqrt(delx * delx + dely * dely + delz * delz);
    if (r < radius) return 0;
    const delta = r - radius;
    if (delta < cutoff) {
      this.cR[0] = delta;
      this.cDelx[0] = delx * (1.0 - radius / r);
      this.cDely[0] = dely * (1.0 - radius / r);
      this.cDelz[0] = delz * (1.0 - radius / r);
      this.cRadius[0] = radius;
      this.cIwall[0] = 0;
      this.cVarflag[0] = 1;
      return 1;
    }
    return 0;
  }

  shape_update(): void {
    if (typeof this.c[0] !== 'number') this.xc = val(this.env, this.c[0]);
    if (typeof this.c[1] !== 'number') this.yc = val(this.env, this.c[1]);
    if (typeof this.c[2] !== 'number') this.zc = val(this.env, this.c[2]);
    if (typeof this.r !== 'number') this.radius = val(this.env, this.r);
  }

  set_velocity_shape(step: number): void {
    const xcenter = this.xcenter;
    const t = this.T;
    t[0] = this.xc; t[1] = this.yc; t[2] = this.zc;
    this.forward_transform(t);
    xcenter[0] = t[0]; xcenter[1] = t[1]; xcenter[2] = t[2];
    if (step > 0) this.rprev = this.prev[4];
    else this.rprev = this.radius;
    this.prev[4] = this.radius;
  }

  velocity_contact_shape(vwall: Float64Array, xc: ArrayLike<number>, dt: number): void {
    const xcenter = this.xcenter;
    const delx = (xc[0] - xcenter[0]) * (1 - this.rprev / this.radius);
    const dely = (xc[1] - xcenter[1]) * (1 - this.rprev / this.radius);
    const delz = (xc[2] - xcenter[2]) * (1 - this.rprev / this.radius);
    vwall[0] += delx / dt;
    vwall[1] += dely / dt;
    vwall[2] += delz / dt;
  }
}
