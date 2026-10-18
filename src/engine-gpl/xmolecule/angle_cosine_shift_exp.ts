/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-MOLECULE/angle_cosine_shift_exp.cpp and angle_cosine_shift_exp.h at commit b332f28 (the LAMMPS
 * developers; Copyright (2003) Sandia Corporation, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * angle_style cosine/shift/exp (docs.lammps.org/angle_cosine_shift_exp.html):
 *   E = -Umin (exp(-a U) - 1) / (exp(a) - 1),   U(theta, theta0) = -0.5 (1 + cos(theta - theta0))
 * Coefficients: "Umin theta0 a" (angle_coeff N Umin theta0 a, theta0 in degrees). For |a| < 0.001 native uses the
 * limit form (doExpansion) instead of the exponential. The trigonometric pieces use the cos and sin of theta0, the
 * clamp s >= 0.001 and the expressions of angle_cosine_shift_exp.cpp compute(), in native order.
 */
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { cos, sin } from '../libm/sincos';
import { exp } from '../libm/exp';
import { evTallyAngle } from './ev';

// LAMMPS: static constexpr double SMALL = 0.001 (angle_cosine_shift_exp.cpp)
const SMALL = 0.001;
// MathConst::MY_PI (math_const.h): 3.14159265358979323846 (the same double as Math.PI)
const MY_PI = 3.14159265358979323846;

export class AngleCosineShiftExp extends Bonded {
  readonly name = 'cosine/shift/exp';
  readonly kind = 'angle' as const;
  private doExpansion!: Uint8Array;
  private umin!: Float64Array;
  private a!: Float64Array;
  private opt1!: Float64Array;
  private theta0!: Float64Array;
  private sint!: Float64Array;
  private cost!: Float64Array;
  private set!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`${this.kind}_style ${this.name} takes no arguments`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    this.doExpansion = new Uint8Array(n);
    this.umin = new Float64Array(n);
    this.a = new Float64Array(n);
    this.opt1 = new Float64Array(n);
    this.theta0 = new Float64Array(n);
    this.sint = new Float64Array(n);
    this.cost = new Float64Array(n);
    this.set = new Uint8Array(n);
  }

  /** angle_cosine_shift_exp.cpp coeff(): narg must be 4 ("Incorrect args for angle coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 4) throw new StyleError('Incorrect args for angle coefficients');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const umin_ = parseNum(args[1], 'Umin');
    const theta0_ = parseNum(args[2], 'theta0');
    const a_ = parseNum(args[3], 'a');
    for (let i = lo; i <= hi; i++) {
      this.doExpansion[i] = Math.abs(a_) < 0.001 ? 1 : 0;
      this.umin[i] = umin_;
      this.a[i] = a_;
      this.cost[i] = cos((theta0_ * MY_PI) / 180.0);
      this.sint[i] = sin((theta0_ * MY_PI) / 180.0);
      this.theta0[i] = (theta0_ * MY_PI) / 180.0;
      if (!this.doExpansion[i]) this.opt1[i] = umin_ / (exp(a_) - 1);
      this.set[i] = 1;
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`All angle coeffs are not set (type ${t})`);
    }
  }

  /** angle_cosine_shift_exp.cpp equilibrium_angle(): theta0 (radians); degrees here, as the other angle styles. */
  equilibrium(type: number): number { return (this.theta0[type] * 180) / Math.PI; }
  equilibriumRadians(type: number): number { return this.theta0[type]; }

  /** write_data: "%d %g %g %g" with theta0 converted back to degrees (theta0[i]/MY_PI*180.0). */
  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${fmtCoeff(this.umin[t])} ${fmtCoeff((this.theta0[t] / MY_PI) * 180.0)} ${fmtCoeff(this.a[t])}`);
    }
    return out;
  }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    const f = bc.f;
    const f1 = [0, 0, 0], f3 = [0, 0, 0];
    for (let n = 0; n < A.n; n++) {
      const i1 = atomIndex(bc, A.atoms[3 * n], 'angle');
      const i2 = atomIndex(bc, A.atoms[3 * n + 1], 'angle');
      const i3 = atomIndex(bc, A.atoms[3 * n + 2], 'angle');
      const type = A.type[n];
      delta(bc, i2, i1, d1); // delx1 = x[i1] - x[i2]
      const rsq1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
      const r1 = Math.sqrt(rsq1);
      delta(bc, i2, i3, d2); // delx2 = x[i3] - x[i2]
      const rsq2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
      const r2 = Math.sqrt(rsq2);
      let c = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];
      c /= r1 * r2;
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      let s = Math.sqrt(1.0 - c * c);
      if (s < SMALL) s = SMALL;
      const aa = this.a[type];
      const uumin = this.umin[type];
      const cccpsss = c * this.cost[type] + s * this.sint[type];
      const cssmscc = c * this.sint[type] - s * this.cost[type];
      let eangle: number;
      let ff: number;
      if (this.doExpansion[type]) {
        eangle = -0.125 * (1 + cccpsss) * (4 + aa * (cccpsss - 1)) * uumin;
        ff = 0.25 * uumin * cssmscc * (2 + aa * cccpsss) / s;
      } else {
        const exp2 = exp(0.5 * aa * (1 + cccpsss));
        eangle = this.opt1[type] * (1 - exp2);
        ff = 0.5 * this.a[type] * this.opt1[type] * exp2 * cssmscc / s;
      }
      const a11 = ff * c / rsq1;
      const a12 = -ff / (r1 * r2);
      const a22 = ff * c / rsq2;
      for (let q = 0; q < 3; q++) {
        f1[q] = a11 * d1[q] + a12 * d2[q];
        f3[q] = a22 * d2[q] + a12 * d1[q];
      }
      for (let q = 0; q < 3; q++) {
        f[3 * i1 + q] += f1[q];
        f[3 * i2 + q] -= f1[q] + f3[q];
        f[3 * i3 + q] += f3[q];
      }
      evTallyAngle(bc, i1, i2, i3, eangle, f1, f3, d1, d2);
    }
  }
}
