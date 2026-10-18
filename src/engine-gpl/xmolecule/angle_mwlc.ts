/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-MOLECULE/angle_mwlc.cpp and angle_mwlc.h at commit b332f28 (the LAMMPS developers; Copyright
 * (2003) Sandia Corporation, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * angle_style mwlc (docs.lammps.org/angle_mwlc.html), the meltable wormlike chain:
 *   E = -kB T log[q + q^m] - v_min,   q = exp[-k1 (1 + cos theta)/kB T],   q^m = exp[-(mu + k2 (1 + cos theta))/kB T]
 *   v_min = -kB T log(1 + exp(-mu/kB T))   (native subtracts it so that E = 0 at the minimum of the expression)
 * Coefficients: "k1 k2 mu T" (angle_coeff N k1 k2 mu T). kB is force->boltz of the units style, T the temperature
 * of the coefficient line. The equilibrium angle is pi (native equilibrium_angle returns MY_PI).
 */
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { exp } from '../libm/exp';
import { log } from '../libm/log';
import { evTallyAngle } from './ev';

export class AngleMWLC extends Bonded {
  readonly name = 'mwlc';
  readonly kind = 'angle' as const;
  private k1!: Float64Array;
  private k2!: Float64Array;
  private mu!: Float64Array;
  private temp!: Float64Array;
  private set!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`${this.kind}_style ${this.name} takes no arguments`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    this.k1 = new Float64Array(n);
    this.k2 = new Float64Array(n);
    this.mu = new Float64Array(n);
    this.temp = new Float64Array(n);
    this.set = new Uint8Array(n);
  }

  /** angle_mwlc.cpp coeff(): narg must be 5 ("Incorrect args for angle coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 5) throw new StyleError('Incorrect args for angle coefficients');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const k1_one = parseNum(args[1], 'k1');
    const k2_one = parseNum(args[2], 'k2');
    const mu_one = parseNum(args[3], 'mu');
    const temp_one = parseNum(args[4], 'temp');
    for (let i = lo; i <= hi; i++) {
      this.k1[i] = k1_one;
      this.k2[i] = k2_one;
      this.mu[i] = mu_one;
      this.temp[i] = temp_one;
      this.set[i] = 1;
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`All angle coeffs are not set (type ${t})`);
    }
  }

  /** angle_mwlc.cpp equilibrium_angle(): MY_PI (radians); 180 degrees here. */
  equilibrium(_type: number): number { return 180; }
  equilibriumRadians(_type: number): number { return Math.PI; }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${fmtCoeff(this.k1[t])} ${fmtCoeff(this.k2[t])} ${fmtCoeff(this.mu[t])} ${fmtCoeff(this.temp[t])}`);
    }
    return out;
  }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const boltz = bc.s.units.boltz;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    const f = bc.f;
    const f1 = [0, 0, 0], f3 = [0, 0, 0];
    for (let n = 0; n < A.n; n++) {
      const i1 = atomIndex(bc, A.atoms[3 * n], 'angle');
      const i2 = atomIndex(bc, A.atoms[3 * n + 1], 'angle');
      const i3 = atomIndex(bc, A.atoms[3 * n + 2], 'angle');
      const type = A.type[n];
      const kbt = this.temp[type] * boltz;
      const v_min = -kbt * log(1 + exp(-this.mu[type] / kbt));
      delta(bc, i2, i1, d1);
      const rsq1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
      const r1 = Math.sqrt(rsq1);
      delta(bc, i2, i3, d2);
      const rsq2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
      const r2 = Math.sqrt(rsq2);
      let c = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];
      c /= r1 * r2;
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      const q = exp(-this.k1[type] * (1.0 + c) / kbt);
      const qm = exp((-this.k2[type] * (1.0 + c) - this.mu[type]) / kbt);
      const Q = q + qm;
      const eangle = -kbt * log(Q) - v_min;
      const a = (this.k1[type] * q + this.k2[type] * qm) / Q;
      const a11 = a * c / rsq1;
      const a12 = -a / (r1 * r2);
      const a22 = a * c / rsq2;
      for (let k = 0; k < 3; k++) {
        f1[k] = a11 * d1[k] + a12 * d2[k];
        f3[k] = a22 * d2[k] + a12 * d1[k];
      }
      for (let k = 0; k < 3; k++) {
        f[3 * i1 + k] += f1[k];
        f[3 * i2 + k] -= f1[k] + f3[k];
        f[3 * i3 + k] += f3[k];
      }
      evTallyAngle(bc, i1, i2, i3, eangle, f1, f3, d1, d2);
    }
  }
}
