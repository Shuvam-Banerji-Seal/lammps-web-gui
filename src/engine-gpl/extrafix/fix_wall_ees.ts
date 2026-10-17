/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_wall_ees.cpp and fix_wall_ees.h (FixWallEES: constructor, precompute, init, wall_particle)
 * at commit b332f28, with MathExtra::quat_to_mat, transpose_matvec and matvec (src/math_extra.cpp, src/math_extra.h) and
 * MathSpecial::powint (src/math_special.h). Copyright Sandia Corporation (GNU General Public License); the EES wall
 * (ellipsoid-wall interaction) is by Ray Shan (Sandia). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix wall/ees: the wall-ellipsoid interaction integrated over the ellipsoid's surface (docs.lammps.org/fix_wall_ees.html).
 * The force on the centre and the torque on the particle come from the signed distance to the wall and the ellipsoid
 * half-length along the wall normal. Energies are not shifted at the cutoff (as the source: ewall[0] sums the raw terms).
 */
import { FixWallBase } from '../../engine/fix/walls';
import { StyleError } from '../../engine/force/types';
import { valueOf } from '../../engine/fix/util';
import { nativeOrder } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { log as libmLog } from '../libm/log';

/** MathSpecial::powint (src/math_special.h): x^n by repeated squaring, the source's operation order. */
const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

/** MathExtra::quat_to_mat (src/math_extra.cpp): rotation matrix of quaternion (w, i, j, k). */
const quatToMat = (q: Float64Array, o: number, a: number[][]): void => {
  const w2 = q[o] * q[o], i2 = q[o + 1] * q[o + 1], j2 = q[o + 2] * q[o + 2], k2 = q[o + 3] * q[o + 3];
  const twoij = 2.0 * q[o + 1] * q[o + 2];
  const twoik = 2.0 * q[o + 1] * q[o + 3];
  const twojk = 2.0 * q[o + 2] * q[o + 3];
  const twoiw = 2.0 * q[o + 1] * q[o];
  const twojw = 2.0 * q[o + 2] * q[o];
  const twokw = 2.0 * q[o + 3] * q[o];
  a[0][0] = w2 + i2 - j2 - k2;
  a[0][1] = twoij - twokw;
  a[0][2] = twojw + twoik;
  a[1][0] = twoij + twokw;
  a[1][1] = w2 - i2 + j2 - k2;
  a[1][2] = twojk - twoiw;
  a[2][0] = twoik - twojw;
  a[2][1] = twojk + twoiw;
  a[2][2] = w2 - i2 - j2 + k2;
};

/** MathExtra::matvec: m v. */
const matvec = (m: number[][], v: number[]): number[] => [
  m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
  m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
  m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
];

/** MathExtra::transpose_matvec: m^T v. */
const transposeMatvec = (m: number[][], v: number[]): number[] => [
  m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
  m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
  m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
];

/** The three Lx, Ly, Lz generators of wall_particle (fix_wall_ees.cpp). */
const LX = [[0, 0, 0], [0, 0, -1], [0, 1, 0]];
const LY = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]];
const LZ = [[0, -1, 0], [1, 0, 0], [0, 0, 0]];

export class FixWallEES extends FixWallBase {
  readonly style = 'wall/ees';
  /** Energy sum (ewall[0]) and per-wall force sums (ewall[m+1]) of the last evaluation. */
  private ewall0 = 0;
  private ewallF = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'wall/ees', 4);
    this.ewallF = new Float64Array(this.walls.length);
  }

  /** The ees energy depends on the particle orientation, so it has no per-distance form: apply() is overridden. */
  protected energyAt(): number { throw new StyleError('fix wall/ees: the energy depends on the ellipsoid orientation'); }
  protected forceAt(): number { throw new StyleError('fix wall/ees: the force depends on the ellipsoid orientation'); }

  init(): void {
    const s = this.sys.state;
    if (!s.shape || !s.quat) throw new StyleError('Fix wall/ees requires atom style ellipsoid');
    if (!s.torque) throw new StyleError('Fix wall/ees requires atom style ellipsoid');
    // "no point particles allowed, spherical is OK": a point particle has all half-axes 0
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (s.shape[3 * i] === 0 && s.shape[3 * i + 1] === 0 && s.shape[3 * i + 2] === 0) {
        throw new StyleError('Fix wall/ees requires extended particles');
      }
    }
    super.init();
  }

  /** Per-wall coefficients (FixWallEES::precompute) from the current epsilon and sigma of wall m. */
  private coeffs(k: number): number[] {
    const w = this.walls[k];
    const eps = valueOf(this.sys, w.params[0]);
    const sig = valueOf(this.sys, w.params[1]);
    return [
      (2.0 / 4725.0) * eps * powint(sig, 12),
      (1.0 / 24.0) * eps * powint(sig, 6),
      (2.0 / 315.0) * eps * powint(sig, 12),
      (1.0 / 3.0) * eps * powint(sig, 6),
      (4.0 / 315.0) * eps * powint(sig, 12),
      (1.0 / 12.0) * eps * powint(sig, 6),
    ];
  }

  /** wall_particle for every wall, in the source's order (walls outer, atoms inner). */
  protected apply(): void {
    const s = this.sys.state;
    const shape = s.shape!, quat = s.quat!, torque = s.torque!, f = s.f, x = s.x;
    const coord = this.coords();
    this.ewall0 = 0;
    this.ewallF.fill(0);
    let onflag = false;
    const A: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let k = 0; k < this.walls.length; k++) {
      const w = this.walls[k];
      const [c1, c2, c3, c4, c5, c6] = this.coeffs(k);
      const dim = w.dim;
      // side = -1 for a lo wall, +1 for a hi wall (fix_wall_ees.cpp: which % 2)
      const side = w.lo ? -1 : 1;
      const cutoff = w.cutoff;
      for (const i of nativeOrder(s)) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const delta = side < 0 ? x[3 * i + dim] - coord[k] : coord[k] - x[3 * i + dim];
        if (delta >= cutoff) continue;
        const nhat = [0, 0, 0];
        nhat[dim] = -1 * side;
        const sh = [shape[3 * i], shape[3 * i + 1], shape[3 * i + 2]];
        quatToMat(quat, 4 * i, A);
        const tempvec = transposeMatvec(A, nhat);
        let sigman2 = 0.0;
        for (let q = 0; q < 3; q++) {
          tempvec[q] *= sh[q];
          sigman2 += tempvec[q] * tempvec[q];
        }
        const sigman = Math.sqrt(sigman2);
        if (delta <= sigman) {
          onflag = true;
          continue;
        }
        const sigman3 = sigman2 * sigman;
        const sigman4 = sigman2 * sigman2;
        const sigman5 = sigman4 * sigman;
        const sigman6 = sigman3 * sigman3;
        const delta2 = delta * delta;
        const delta3 = delta2 * delta;
        const delta4 = delta2 * delta2;
        const delta5 = delta3 * delta2;
        const delta6 = delta3 * delta3;
        const hhss = delta2 - sigman2;
        const hhss2 = hhss * hhss;
        const hhss4 = hhss2 * hhss2;
        const hhss8 = hhss4 * hhss4;
        const hhss7 = hhss4 * hhss2 * hhss;
        const hps = delta + sigman;
        const hms = delta - sigman;
        const fwall = side * (
          -1 * c4 / hhss2 +
          c3 * (21 * delta6 + 63 * delta4 * sigman2 + 27 * delta2 * sigman4 + sigman6) / hhss8
        );
        f[3 * i + dim] -= fwall;
        this.ewall0 += -1 * c2 * (4 * delta / sigman2 / hhss + 2 * libmLog(hms / hps) / sigman3) +
          c1 * (35 * delta5 + 70 * delta3 * sigman2 + 15 * delta * sigman4) / hhss7;
        this.ewallF[k] += fwall;
        const twall = c6 * (6. * delta3 / sigman4 / hhss2 - 10. * delta / sigman2 / hhss2 + 3. * libmLog(hms / hps) / sigman5) +
          c5 * (21. * delta5 + 30. * delta3 * sigman2 + 5. * delta * sigman4) / hhss8;
        // SAn = S A^T nhat (tempvec after scaling); that[j] = SAn . S A^T (L_j nhat)
        const SAn = tempvec;
        const that = [0, 0, 0];
        for (const L of [LX, LY, LZ]) {
          const t1 = matvec(L, nhat);
          const t2 = transposeMatvec(A, t1);
          for (let q = 0; q < 3; q++) t2[q] *= sh[q];
          that[L === LX ? 0 : L === LY ? 1 : 2] = SAn[0] * t2[0] + SAn[1] * t2[1] + SAn[2] * t2[2];
        }
        for (let j = 0; j < 3; j++) torque[3 * i + j] += twall * that[j];
      }
    }
    if (onflag) throw new StyleError('Particle on or inside fix wall surface');
  }

  energy(): number { return this.ewall0; }
  computeScalar(): number { return this.ewall0; }
  computeVector(i: number): number { return this.ewallF[i]; }
}
