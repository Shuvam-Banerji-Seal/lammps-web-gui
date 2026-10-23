/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MACHDYN/fix_smd_move_triangulated_surface.cpp
 * and .h (class FixSMDMoveTriSurf, fix smd/move_tri_surf): moves a triangulated surface of atom_style smd in a linear
 * translation (*LINEAR vx vy vz), a back-and-forth wiggle (*WIGGLE vx vy vz max_travel) or a rotation about an axis
 * (*ROTATE ox oy oz ax ay az period). Triangle particles have molecule IDs of at least 65535 and keep their vertices in
 * the 9 per-atom values of SMD_DEFGRAD (the smd_data_9 of the source). MACHDYN author (from the source): Georg C.
 * Ganzenmueller, Fraunhofer EMI. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The forward communication of x0 and the vertices is a no-op on one process. Native prints its banner to the screen;
 * this port does not.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { isSmdStyle, SMD_DEFGRAD, SMD_VEST } from '../../engine/smd';
import { cos as libmCos, sin as libmSin } from '../libm/sincos';

/** math_const.h MY_2PI */
const MY_2PI = 6.28318530717958647692;
/** atom molecule ID from which an atom is a triangle vertex carrier (fix_smd_move_triangulated_surface.cpp: mol[i] >= 65535) */
const TRI_MOL_MIN = 65535;

type Vec3 = [number, number, number];

export class FixSMDMoveTriSurf extends Fix {
  readonly style = 'smd/move_tri_surf';
  private readonly linear: boolean;
  private readonly wiggle: boolean;
  private readonly rotate: boolean;
  private vx = 0;
  private vy = 0;
  private vz = 0;
  private wiggleMaxTravel = 0;
  private wiggleTravel = 0;
  private wiggleDirection = 1.0;
  private origin: Vec3 = [0, 0, 0];
  private axis: Vec3 = [0, 0, 0];
  private rotationPeriod = 0;
  private dtv = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!isSmdStyle(sys.state.atomStyle)) throw new StyleError('fix fix smd/move_tri_surf command requires atom_style smd');
    if (args.length < 1) throw new StyleError('Illegal number of arguments for fix fix smd/move_tri_surf command');
    let linear = false, wiggle = false, rotate = false;
    // As native LAMMPS does (src/MACHDYN/fix_smd_move_triangulated_surface.cpp at b332f28): a missing value gives the
    // keyword's message, a present one is read by utils::numeric.
    const num = (w: string | undefined, msg: string): number => {
      if (w === undefined) throw new StyleError(msg);
      return numeric(w);
    };
    for (let k = 0; k < args.length;) {
      const w = args[k];
      if (w === '*LINEAR') {
        linear = true;
        const m = 'expected three floats for velocity following *LINEAR';
        this.vx = num(args[k + 1], m); this.vy = num(args[k + 2], m); this.vz = num(args[k + 3], m);
        k += 4;
      } else if (w === '*WIGGLE') {
        wiggle = true;
        const m = 'expected 4 floats following *WIGGLE : vx vy vz max_travel';
        this.vx = num(args[k + 1], m); this.vy = num(args[k + 2], m); this.vz = num(args[k + 3], m);
        this.wiggleMaxTravel = num(args[k + 4], m);
        k += 5;
      } else if (w === '*ROTATE') {
        rotate = true;
        const m = 'expected 7 floats following *ROTATE: origin, rotation axis, and rotation period';
        this.origin = [num(args[k + 1], m), num(args[k + 2], m), num(args[k + 3], m)];
        this.axis = [num(args[k + 4], m), num(args[k + 5], m), num(args[k + 6], m)];
        this.rotationPeriod = num(args[k + 7], m);
        k += 8;
      } else {
        throw new StyleError(`Unknown fix smd/move_tri_surf keyword: ${w}`);
      }
    }
    this.linear = linear;
    this.wiggle = wiggle;
    this.rotate = rotate;
    this.timeIntegrate = true;
  }

  init(): void {
    this.dtv = this.sys.state.dt;
  }

  resetDt(): void {
    this.dtv = this.sys.state.dt;
  }

  /** The velocity written to the atoms and the vertices (vest) of the surface at each step. */
  private setVelocity(i: number, vel: Vec3): void {
    const s = this.sys.state;
    const vest = s.custom.get(SMD_VEST)!.data;
    s.v[3 * i] = vel[0]; s.v[3 * i + 1] = vel[1]; s.v[3 * i + 2] = vel[2];
    vest[3 * i] = vel[0]; vest[3 * i + 1] = vel[1]; vest[3 * i + 2] = vel[2];
  }

  /** fix_smd_move_triangulated_surface.cpp initial_integrate: linear, wiggle and rotation of the group. */
  initialIntegrate(): void {
    const s = this.sys.state;
    const { x, mask, molecule } = s;
    const sm = s.custom.get(SMD_DEFGRAD)!.data;
    const dtv = this.dtv;
    const vx = this.vx, vy = this.vy, vz = this.vz;
    if (this.linear) {
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        this.setVelocity(i, [vx, vy, vz]);
        x[3 * i] += dtv * vx;
        x[3 * i + 1] += dtv * vy;
        x[3 * i + 2] += dtv * vz;
        if (molecule[i] >= TRI_MOL_MIN) {
          for (let k = 0; k < 3; k++) {
            sm[9 * i + k] += dtv * vx;
            sm[9 * i + 3 + k] += dtv * vy;
            sm[9 * i + 6 + k] += dtv * vz;
          }
        }
      }
    }
    if (this.wiggle) {
      this.wiggleTravel += Math.sqrt(vx * vx + vy * vy + vz * vz) * dtv;
      const wvx = this.wiggleDirection * vx, wvy = this.wiggleDirection * vy, wvz = this.wiggleDirection * vz;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        this.setVelocity(i, [wvx, wvy, wvz]);
        x[3 * i] += dtv * wvx;
        x[3 * i + 1] += dtv * wvy;
        x[3 * i + 2] += dtv * wvz;
        if (molecule[i] >= TRI_MOL_MIN) {
          for (let k = 0; k < 3; k++) {
            sm[9 * i + k] += dtv * wvx;
            sm[9 * i + 3 + k] += dtv * wvy;
            sm[9 * i + 6 + k] += dtv * wvz;
          }
        }
      }
      if (this.wiggleTravel >= this.wiggleMaxTravel) {
        this.wiggleDirection *= -1.0;
        this.wiggleTravel = 0.0;
      }
    }
    if (this.rotate) this.rotateSurface(dtv);
  }

  /** The rotation matrix Rot = cos eye + sin u_cross + (1 - cos) axis axis^T (Eigen element order). */
  private rotationMatrix(phi: number): number[][] {
    const a = this.axis;
    const c = libmCos(phi), sn = libmSin(phi);
    const ucross = [
      [0.0, -a[2], a[1]],
      [a[2], 0.0, -a[0]],
      [-a[1], a[0], 0.0],
    ];
    const uxu = [
      [a[0] * a[0], a[0] * a[1], a[0] * a[2]],
      [a[1] * a[0], a[1] * a[1], a[1] * a[2]],
      [a[2] * a[0], a[2] * a[1], a[2] * a[2]],
    ];
    const R: number[][] = [];
    for (let r = 0; r < 3; r++) {
      R.push([0, 1, 2].map((q) => (c * (r === q ? 1.0 : 0.0) + sn * ucross[r][q]) + (1.0 - c) * uxu[r][q]));
    }
    return R;
  }

  /** Rot * (p - origin) + origin for one point. */
  private rotatePoint(Rot: number[][], p: Vec3): Vec3 {
    const o = this.origin;
    const R: Vec3 = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    const out: Vec3 = [0, 0, 0];
    for (let r = 0; r < 3; r++) out[r] = (Rot[r][0] * R[0] + Rot[r][1] * R[1] + Rot[r][2] * R[2]) + o[r];
    return out;
  }

  /** The rotation of the group: positions, velocities; for triangle vertices also their normal (x0). */
  private rotateSurface(dtv: number): void {
    const s = this.sys.state;
    const { x, mask, molecule } = s;
    const sm = s.custom.get(SMD_DEFGRAD)!.data;
    const x0 = s.x0;
    const dt = s.dt;
    const phi = MY_2PI * dtv / this.rotationPeriod;
    const Rot = this.rotationMatrix(phi);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const point: Vec3 = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
      const rp = this.rotatePoint(Rot, point);
      this.setVelocity(i, [(rp[0] - point[0]) / dt, (rp[1] - point[1]) / dt, (rp[2] - point[2]) / dt]);
      x[3 * i] = rp[0]; x[3 * i + 1] = rp[1]; x[3 * i + 2] = rp[2];
      if (molecule[i] >= TRI_MOL_MIN) {
        const v: Vec3[] = [0, 1, 2].map((k) => [sm[9 * i + 3 * k], sm[9 * i + 3 * k + 1], sm[9 * i + 3 * k + 2]] as Vec3);
        const rv = v.map((p) => this.rotatePoint(Rot, p));
        for (let k = 0; k < 3; k++) for (let d = 0; d < 3; d++) sm[9 * i + 3 * k + d] = rv[k][d];
        const v1 = rv[0], v2 = rv[1], v3 = rv[2];
        // n = (v2 - v1) x (v2 - v3), normalized (fix_smd_move_triangulated_surface.cpp)
        const a: Vec3 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
        const b: Vec3 = [v2[0] - v3[0], v2[1] - v3[1], v2[2] - v3[2]];
        const n: Vec3 = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        const norm = Math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (x0) { x0[3 * i] = n[0] / norm; x0[3 * i + 1] = n[1] / norm; x0[3 * i + 2] = n[2] / norm; }
      }
    }
  }
}
