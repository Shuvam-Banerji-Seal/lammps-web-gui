/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_wall_region_ees.cpp and fix_wall_region_ees.h (FixWallRegionEES: constructor, init, min_setup,
 * post_force, ees, compute_scalar, compute_vector) at commit b332f28, with MathExtra::quat_to_mat, transpose_matvec and
 * matvec, MathSpecial::powint (src/math_extra.cpp, src/math_special.h). Copyright Sandia Corporation (GNU General Public
 * License); the EES wall is by Ray Shan (Sandia). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix wall/region/ees (docs.lammps.org/fix_wall_ees.html, "args for style wall/region/ees"): the boundary of a region as
 * an ellipsoid wall. Each surface contact within the cutoff gives the same ellipsoid-wall energy, force and torque as
 * fix wall/ees (src/engine-gpl/extrafix/fix_wall_ees.ts); the contact normal is the surface point to particle direction.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { nativeOrder } from '../../engine/atoms';
import type { System } from '../../engine/system';
import type { SurfaceContact } from '../../engine/region';

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

/** MathExtra::quat_to_mat (src/math_extra.cpp): rotation matrix of quaternion (w, i, j, k) at offset o. */
const quatToMat = (q: Float64Array, o: number): number[][] => {
  const w2 = q[o] * q[o], i2 = q[o + 1] * q[o + 1], j2 = q[o + 2] * q[o + 2], k2 = q[o + 3] * q[o + 3];
  const twoij = 2.0 * q[o + 1] * q[o + 2];
  const twoik = 2.0 * q[o + 1] * q[o + 3];
  const twojk = 2.0 * q[o + 2] * q[o + 3];
  const twoiw = 2.0 * q[o + 1] * q[o];
  const twojw = 2.0 * q[o + 2] * q[o];
  const twokw = 2.0 * q[o + 3] * q[o];
  return [
    [w2 + i2 - j2 - k2, twoij - twokw, twojw + twoik],
    [twoij + twokw, w2 - i2 + j2 - k2, twojk - twoiw],
    [twoik - twojw, twojk + twoiw, w2 - i2 - j2 + k2],
  ];
};

const matvec = (m: number[][], v: number[]): number[] => [
  m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
  m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
  m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
];

const transposeMatvec = (m: number[][], v: number[]): number[] => [
  m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
  m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
  m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
];

const LX = [[0, 0, 0], [0, 0, -1], [0, 1, 0]];
const LY = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]];
const LZ = [[0, -1, 0], [1, 0, 0], [0, 0, 0]];

export class FixWallRegionEES extends Fix {
  readonly style = 'wall/region/ees';
  private readonly regionId: string;
  private readonly epsilon: number;
  private readonly sigma: number;
  private readonly cutoff: number;
  /** Energy sum (ewall[0]) and the three wall force sums (ewall[1..3]). */
  private readonly ewall = [0, 0, 0, 0];
  private readonly contacts: SurfaceContact[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4) throw new StyleError('Illegal fix wall/region/ees command: usage fix ID group wall/region/ees region-ID epsilon sigma cutoff');
    this.regionId = args[0];
    const nums = args.slice(1).map((w) => {
      const v = Number(w);
      if (w.startsWith('v_') || !Number.isFinite(v)) throw new StyleError(`fix wall/region/ees: '${w}' must be a number`);
      return v;
    });
    this.epsilon = nums[0];
    this.sigma = nums[1];
    this.cutoff = nums[2];
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
  }

  init(): void {
    this.sys.region(this.regionId);
    const s = this.sys.state;
    if (!s.shape || !s.quat || !s.torque) throw new StyleError('Fix wall/region/ees requires atom style ellipsoid');
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (s.shape[3 * i] === 0 && s.shape[3 * i + 1] === 0 && s.shape[3 * i + 2] === 0) {
        throw new StyleError('Fix wall/region/ees requires only extended particles');
      }
    }
  }

  /** The source's constants (init): coeff1..coeff6 from epsilon and sigma. */
  private coeffs(): number[] {
    const eps = this.epsilon, sig = this.sigma;
    return [
      (2.0 / 4725.0) * eps * powint(sig, 12),
      (1.0 / 24.0) * eps * powint(sig, 6),
      (2.0 / 315.0) * eps * powint(sig, 12),
      (1.0 / 3.0) * eps * powint(sig, 6),
      (4.0 / 315.0) * eps * powint(sig, 12),
      (1.0 / 12.0) * eps * powint(sig, 6),
    ];
  }

  postForce(): void {
    const s = this.sys.state;
    const region = this.sys.region(this.regionId);
    const shape = s.shape!, quat = s.quat!, torque = s.torque!, f = s.f, x = s.x;
    this.ewall.fill(0);
    const [c1, c2, c3, c4, c5, c6] = this.coeffs();
    let onflag = false;
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (!region.match(x[3 * i], x[3 * i + 1], x[3 * i + 2])) {
        onflag = true;
        continue;
      }
      // the contact-distance limit of each direction: the ellipsoid half-length along the axis
      const A = quatToMat(quat, 4 * i);
      const sh = [shape[3 * i], shape[3 * i + 1], shape[3 * i + 2]];
      const tooclose = [0, 0, 0];
      for (let which = 0; which < 3; which++) {
        const nh = [0, 0, 0];
        nh[which] = 1;
        const tv = transposeMatvec(A, nh);
        let sn2 = 0.0;
        for (let k = 0; k < 3; k++) {
          tv[k] *= sh[k];
          sn2 += tv[k] * tv[k];
        }
        tooclose[which] = Math.sqrt(sn2);
      }
      this.contacts.length = 0;
      region.surfaceContacts(x[3 * i], x[3 * i + 1], x[3 * i + 2], this.contacts);
      for (const c of this.contacts) {
        if (!(c.dist < this.cutoff)) continue;
        // the source's del components: nonzero components of the contact vector (del = n r)
        const delx = c.nx * c.dist, dely = c.ny * c.dist, delz = c.nz * c.dist;
        if (delx !== 0 && c.dist <= tooclose[0]) { onflag = true; continue; }
        if (dely !== 0 && c.dist <= tooclose[1]) { onflag = true; continue; }
        if (delz !== 0 && c.dist <= tooclose[2]) { onflag = true; continue; }
        const rinv = 1.0 / c.dist;
        const { fwall, eng, torque: tq } = this.ees(c, c1, c2, c3, c4, c5, c6, shape, i, A);
        const fx = fwall * delx * rinv, fy = fwall * dely * rinv, fz = fwall * delz * rinv;
        f[3 * i] += fx;
        f[3 * i + 1] += fy;
        f[3 * i + 2] += fz;
        this.ewall[0] += eng;
        this.ewall[1] -= fx;
        this.ewall[2] -= fy;
        this.ewall[3] -= fz;
        torque[3 * i] += tq[0];
        torque[3 * i + 1] += tq[1];
        torque[3 * i + 2] += tq[2];
      }
    }
    if (onflag) throw new StyleError('Particle on or inside surface of region used in fix wall/region/ees');
  }

  /** FixWallRegionEES::ees: the force, energy and torque of one contact (distance r along the unit normal n). */
  private ees(c: SurfaceContact, c1: number, c2: number, c3: number, c4: number, c5: number, c6: number,
    shape: Float64Array, i: number, A: number[][]): { fwall: number; eng: number; torque: number[] } {
    const nhat = [c.nx, c.ny, c.nz];
    const sh = [shape[3 * i], shape[3 * i + 1], shape[3 * i + 2]];
    const tempvec = transposeMatvec(A, nhat);
    let sigman2 = 0.0;
    for (let k = 0; k < 3; k++) {
      tempvec[k] *= sh[k];
      sigman2 += tempvec[k] * tempvec[k];
    }
    const SAn = tempvec.slice();
    const sigman = Math.sqrt(sigman2);
    const delta = Math.abs(c.dist);
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
    const fwall = -1 * c4 / hhss2 +
      c3 * (21 * delta6 + 63 * delta4 * sigman2 + 27 * delta2 * sigman4 + sigman6) / hhss8;
    const eng = -1 * c2 * (4 * delta / sigman2 / hhss + 2 * Math.log(hms / hps) / sigman3) +
      c1 * (35 * delta5 + 70 * delta3 * sigman2 + 15 * delta * sigman4) / hhss7;
    const twall = c6 * (6 * delta3 / sigman4 / hhss2 - 10 * delta / sigman2 / hhss2 + 3 * Math.log(hms / hps) / sigman5) +
      c5 * (21. * delta5 + 30. * delta3 * sigman2 + 5. * delta * sigman4) / hhss8;
    const that = [0, 0, 0];
    for (let L = 0; L < 3; L++) {
      const Lm = L === 0 ? LX : L === 1 ? LY : LZ;
      const t1 = matvec(Lm, nhat);
      const t2 = transposeMatvec(A, t1);
      for (let k = 0; k < 3; k++) t2[k] *= sh[k];
      that[L] = SAn[0] * t2[0] + SAn[1] * t2[1] + SAn[2] * t2[2];
    }
    return { fwall, eng, torque: [twall * that[0], twall * that[1], twall * that[2]] };
  }

  minSetup(): void { this.postForce(); }
  minPostForce(): void { this.postForce(); }

  energy(): number { return this.ewall[0]; }
  computeScalar(): number { return this.ewall[0]; }
  computeVector(n: number): number { return this.ewall[n + 1]; }
}
