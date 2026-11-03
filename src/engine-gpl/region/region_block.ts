/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/region_block.cpp and src/region_block.h (RegBlock: the face and corner setup of the constructor, inside,
 * surface_interior, surface_exterior, shape_update, find_closest_point, inside_face) at commit b332f28. Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { NativeRegion, type NativeRegionEnv } from './native_region';
import type { Param } from '../../engine/region';

/** region_block.cpp: "static constexpr double BIG = 1.0e200;" */
const BIG = 1.0e200;

export class NativeBlock extends NativeRegion {
  xlo = 0; xhi = 0; ylo = 0; yhi = 0; zlo = 0; zhi = 0;
  /** corners[6][4][3] and face[6][3] of native, flat: corners[i][j][k] at 12 i + 3 j + k, face[i][k] at 3 i + k. */
  readonly corners = new Float64Array(72);
  readonly face = new Float64Array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0]);
  private readonly cornerView: Float64Array[];
  private readonly xr = new Float64Array(3);
  private readonly xproj = new Float64Array(3);
  private readonly p = new Float64Array(3);

  /** b: xlo, xhi, ylo, yhi, zlo, zhi as the engine parsed them (numbers scaled by the lattice, or variables with scale). */
  constructor(id: string, env: NativeRegionEnv, private readonly b: readonly Param[]) {
    super(id, 'block', env);
    if (b.some((v) => typeof v !== 'number')) this.varshape = 1;
    this.cornerView = Array.from({ length: 24 }, (_, k) => this.corners.subarray(3 * k, 3 * k + 3));
    this.allocContacts(6);
  }

  /** The constructor's values and shape_update (with variables), then the corners; cmax 6, tmax 3 inside, 1 outside. */
  setup(): void {
    const b = this.b;
    if (typeof b[0] === 'number') this.xlo = b[0];
    if (typeof b[1] === 'number') this.xhi = b[1];
    if (typeof b[2] === 'number') this.ylo = b[2];
    if (typeof b[3] === 'number') this.yhi = b[3];
    if (typeof b[4] === 'number') this.zlo = b[4];
    if (typeof b[5] === 'number') this.zhi = b[5];
    this.tmax = this.interior ? 3 : 1;
    if (this.varshape) this.shape_update();
    else this.setCorners();
  }

  private val(k: number): number {
    const p = this.b[k];
    return typeof p === 'number' ? p : p.scale * this.env.variable(p.variable);
  }

  inside(x: number, y: number, z: number): number {
    if (x >= this.xlo && x <= this.xhi && y >= this.ylo && y <= this.yhi && z >= this.zlo && z <= this.zhi) return 1;
    return 0;
  }

  surface_interior(x: Float64Array, cutoff: number): number {
    const { xlo, xhi, ylo, yhi, zlo, zhi } = this;
    // x is exterior to block
    if (x[0] < xlo || x[0] > xhi || x[1] < ylo || x[1] > yhi || x[2] < zlo || x[2] > zhi) return 0;
    // x is interior to block or on its surface
    const of = this.open_faces;
    const cR = this.cR, cDelx = this.cDelx, cDely = this.cDely, cDelz = this.cDelz, cRadius = this.cRadius, cIwall = this.cIwall;
    let n = 0;
    let delta = x[0] - xlo;
    if (delta < cutoff && !of[0]) {
      cR[n] = delta; cDelx[n] = delta; cDely[n] = cDelz[n] = 0.0; cRadius[n] = 0; cIwall[n] = 0;
      n++;
    }
    delta = xhi - x[0];
    if (delta < cutoff && !of[1]) {
      cR[n] = delta; cDelx[n] = -delta; cDely[n] = cDelz[n] = 0.0; cRadius[n] = 0; cIwall[n] = 1;
      n++;
    }
    delta = x[1] - ylo;
    if (delta < cutoff && !of[2]) {
      cR[n] = delta; cDely[n] = delta; cDelx[n] = cDelz[n] = 0.0; cRadius[n] = 0; cIwall[n] = 2;
      n++;
    }
    delta = yhi - x[1];
    if (delta < cutoff && !of[3]) {
      cR[n] = delta; cDely[n] = -delta; cDelx[n] = cDelz[n] = 0.0; cRadius[n] = 0; cIwall[n] = 3;
      n++;
    }
    delta = x[2] - zlo;
    if (delta < cutoff && !of[4]) {
      cR[n] = delta; cDelz[n] = delta; cDelx[n] = cDely[n] = 0.0; cRadius[n] = 0; cIwall[n] = 4;
      n++;
    }
    delta = zhi - x[2];
    if (delta < cutoff && !of[5]) {
      cR[n] = delta; cDelz[n] = -delta; cDelx[n] = cDely[n] = 0.0; cRadius[n] = 0; cIwall[n] = 5;
      n++;
    }
    return n;
  }

  surface_exterior(x: Float64Array, cutoff: number): number {
    const { xlo, xhi, ylo, yhi, zlo, zhi } = this;
    // x is far enough from block that there is no contact; x is interior to block
    if (x[0] <= xlo - cutoff || x[0] >= xhi + cutoff || x[1] <= ylo - cutoff || x[1] >= yhi + cutoff
      || x[2] <= zlo - cutoff || x[2] >= zhi + cutoff) return 0;
    if (x[0] > xlo && x[0] < xhi && x[1] > ylo && x[1] < yhi && x[2] > zlo && x[2] < zhi) return 0;
    // xp,yp,zp = point on surface of block that x is closest to (could be edge or corner pt of block)
    let xp = 0, yp = 0, zp = 0;
    if (!this.openflag) {
      if (x[0] < xlo) xp = xlo;
      else if (x[0] > xhi) xp = xhi;
      else xp = x[0];
      if (x[1] < ylo) yp = ylo;
      else if (x[1] > yhi) yp = yhi;
      else yp = x[1];
      if (x[2] < zlo) zp = zlo;
      else if (x[2] > zhi) zp = zhi;
      else zp = x[2];
    } else {
      let mindist = BIG;
      for (let i = 0; i < 6; i++) {
        if (this.open_faces[i]) continue;
        const dist = this.find_closest_point(i, x);
        if (dist < mindist) {
          xp = this.fcX; yp = this.fcY; zp = this.fcZ;
          mindist = dist;
        }
      }
      // no contact possible if all faces are open
      if (mindist === BIG) return 0;
    }
    this.add_contact(0, x, xp, yp, zp);
    this.cIwall[0] = 0;
    if (this.cR[0] < cutoff) return 1;
    return 0;
  }

  shape_update(): void {
    const b = this.b;
    if (typeof b[0] !== 'number') this.xlo = this.val(0);
    if (typeof b[1] !== 'number') this.xhi = this.val(1);
    if (typeof b[2] !== 'number') this.ylo = this.val(2);
    if (typeof b[3] !== 'number') this.yhi = this.val(3);
    if (typeof b[4] !== 'number') this.zlo = this.val(4);
    if (typeof b[5] !== 'number') this.zhi = this.val(5);
    this.setCorners();
  }

  /** The corners of the six faces (constructor and shape_update). */
  private setCorners(): void {
    const c = this.corners;
    const set = (i: number, j: number, x: number, y: number, z: number) => { const o = 12 * i + 3 * j; c[o] = x; c[o + 1] = y; c[o + 2] = z; };
    const copy = (fi: number, fj: number, ti: number, tj: number) => { const s = 12 * fi + 3 * fj, d = 12 * ti + 3 * tj; c[d] = c[s]; c[d + 1] = c[s + 1]; c[d + 2] = c[s + 2]; };
    const { xlo, xhi, ylo, yhi, zlo, zhi } = this;
    // face[0]
    set(0, 0, xlo, ylo, zlo); set(0, 1, xlo, ylo, zhi); set(0, 2, xlo, yhi, zhi); set(0, 3, xlo, yhi, zlo);
    // face[1]
    set(1, 0, xhi, ylo, zlo); set(1, 1, xhi, ylo, zhi); set(1, 2, xhi, yhi, zhi); set(1, 3, xhi, yhi, zlo);
    // face[2]
    copy(0, 0, 2, 0); copy(1, 0, 2, 1); copy(1, 1, 2, 2); copy(0, 1, 2, 3);
    // face[3]
    copy(0, 3, 3, 0); copy(0, 2, 3, 1); copy(1, 2, 3, 2); copy(1, 3, 3, 3);
    // face[4]
    copy(0, 0, 4, 0); copy(0, 3, 4, 1); copy(1, 3, 4, 2); copy(1, 0, 4, 3);
    // face[5]
    copy(0, 1, 5, 0); copy(1, 1, 5, 1); copy(1, 2, 5, 2); copy(0, 2, 5, 3);
  }

  /** find_closest_point's xc, yc, zc (references in the C++). */
  private fcX = 0;
  private fcY = 0;
  private fcZ = 0;

  /** RegBlock::find_closest_point: squared distance from x to face i, the point into fcX, fcY, fcZ. */
  private find_closest_point(i: number, x: Float64Array): number {
    const c = this.corners, f = this.face, o = 12 * i;
    const xr = this.xr, xproj = this.xproj, p = this.p;
    xr[0] = x[0] - c[o];
    xr[1] = x[1] - c[o + 1];
    xr[2] = x[2] - c[o + 2];
    const dot = f[3 * i] * xr[0] + f[3 * i + 1] * xr[1] + f[3 * i + 2] * xr[2];
    xproj[0] = xr[0] - dot * f[3 * i];
    xproj[1] = xr[1] - dot * f[3 * i + 1];
    xproj[2] = xr[2] - dot * f[3 * i + 2];
    let d2min = BIG;
    // check if point projects inside of face
    if (this.inside_face(xproj, i)) {
      d2min = dot * dot;
      this.fcX = xproj[0] + c[o];
      this.fcY = xproj[1] + c[o + 1];
      this.fcZ = xproj[2] + c[o + 2];
    } else {
      // check each edge
      const cv = this.cornerView;
      for (let e = 0; e < 4; e++) {
        this.point_on_line_segment(cv[4 * i + e], cv[4 * i + ((e + 1) & 3)], x, p);
        const d2 = (p[0] - x[0]) * (p[0] - x[0]) + (p[1] - x[1]) * (p[1] - x[1]) + (p[2] - x[2]) * (p[2] - x[2]);
        if (d2 < d2min) {
          d2min = d2;
          this.fcX = p[0]; this.fcY = p[1]; this.fcZ = p[2];
        }
      }
    }
    return d2min;
  }

  private inside_face(xproj: Float64Array, iface: number): number {
    const { xlo, xhi, ylo, yhi, zlo, zhi } = this;
    if (iface < 2) {
      if (xproj[1] > 0 && (xproj[1] < yhi - ylo) && xproj[2] > 0 && (xproj[2] < zhi - zlo)) return 1;
    } else if (iface < 4) {
      if (xproj[0] > 0 && (xproj[0] < (xhi - xlo)) && xproj[2] > 0 && (xproj[2] < (zhi - zlo))) return 1;
    } else {
      if (xproj[0] > 0 && xproj[0] < (xhi - xlo) && xproj[1] > 0 && xproj[1] < (yhi - ylo)) return 1;
    }
    return 0;
  }
}
