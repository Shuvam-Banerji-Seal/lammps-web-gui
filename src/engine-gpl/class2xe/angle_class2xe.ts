/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/angle_class2xe.cpp and src/CLASS2/angle_class2xe.h at commit b332f28 (contributing author
 * Josh Kemppainen, Michigan Technological University). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * angle_style class2xe — the ClassII-xe angle: the class2 angle polynomial plus a bond-bond term and a bond-angle
 * term whose bond-stretch factors are Morse-like, so the terms stay finite as a bond dissociates (the reference
 * the native style prints: doi 10.1038/s41524-025-01838-5).
 *   E_a  = K2 dt^2 + K3 dt^3 + K4 dt^4,             dt = theta - theta0 (theta0 given in degrees)
 *   E_bb = D0 (1 - e^{-a (r1 - R1)}) (1 - e^{-a (r2 - R2)})          (coeffs: D0, alpha, R1, R2)
 *   E_ba = D1 (1 - e^{-a1 (r1 - R1)}) dt + D2 (1 - e^{-a2 (r2 - R2)}) dt
 *                                                                    (coeffs: D1, D2, alpha1, alpha2, R1, R2)
 * Coefficient lines (angle_coeff): "type theta0 K2 K3 K4", "type bb D0 alpha R1 R2", "type ba D1 D2 alpha1 alpha2
 * R1 R2"; all three must be set for every type. The forces are the native expressions of angle_class2xe.cpp,
 * including the native reuse of rsq1 in the bond-angle derivative of the second bond (aa21 in the first block),
 * which is kept as native has it. The angle clamp s >= 0.001 (SMALL) is native's.
 * Geometry: every atom is placed relative to the vertex (delta from the owner), as the other angle styles do.
 */
import { Bonded, StyleError, bondedVirial, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { exp as libmExp } from '../libm/exp';
import { acos as libmAcos } from '../libm/acos';

// LAMMPS: static constexpr double SMALL = 0.001 (angle_class2xe.cpp)
const SMALL = 0.001;

export class AngleClass2xe extends Bonded {
  readonly name = 'class2xe';
  readonly kind = 'angle' as const;
  private theta0!: Float64Array;
  private k2!: Float64Array;
  private k3!: Float64Array;
  private k4!: Float64Array;
  private bbD0!: Float64Array;
  private bbAlpha!: Float64Array;
  private bbR1!: Float64Array;
  private bbR2!: Float64Array;
  private baD1!: Float64Array;
  private baD2!: Float64Array;
  private baAlpha1!: Float64Array;
  private baAlpha2!: Float64Array;
  private baR1!: Float64Array;
  private baR2!: Float64Array;
  private setA!: Uint8Array;
  private setBb!: Uint8Array;
  private setBa!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`${this.kind}_style ${this.name} takes no arguments`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    for (const k of ['theta0', 'k2', 'k3', 'k4', 'bbD0', 'bbAlpha', 'bbR1', 'bbR2', 'baD1', 'baD2', 'baAlpha1', 'baAlpha2', 'baR1', 'baR2'] as const) {
      this[k] = new Float64Array(n);
    }
    this.setA = new Uint8Array(n);
    this.setBb = new Uint8Array(n);
    this.setBa = new Uint8Array(n);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 2) throw new StyleError('Incorrect args for angle coefficients');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const kw = args[1];
    if (kw === 'bb') {
      if (args.length !== 6) throw new StyleError('Incorrect args for angle coefficients');
      const d0 = parseNum(args[2], 'bb_d0'), al = parseNum(args[3], 'bb_alpha');
      const r1 = parseNum(args[4], 'bb_r1'), r2 = parseNum(args[5], 'bb_r2');
      for (let t = lo; t <= hi; t++) {
        this.bbD0[t] = d0; this.bbAlpha[t] = al; this.bbR1[t] = r1; this.bbR2[t] = r2; this.setBb[t] = 1;
      }
    } else if (kw === 'ba') {
      if (args.length !== 8) throw new StyleError('Incorrect args for angle coefficients');
      const d1 = parseNum(args[2], 'ba_d1'), d2 = parseNum(args[3], 'ba_d2');
      const a1 = parseNum(args[4], 'ba_alpha1'), a2 = parseNum(args[5], 'ba_alpha2');
      const r1 = parseNum(args[6], 'ba_r1'), r2 = parseNum(args[7], 'ba_r2');
      for (let t = lo; t <= hi; t++) {
        this.baD1[t] = d1; this.baD2[t] = d2; this.baAlpha1[t] = a1; this.baAlpha2[t] = a2;
        this.baR1[t] = r1; this.baR2[t] = r2; this.setBa[t] = 1;
      }
    } else {
      if (args.length !== 5) throw new StyleError('Incorrect args for angle coefficients');
      const th = parseNum(args[1], 'theta0'), K2 = parseNum(args[2], 'K2');
      const K3 = parseNum(args[3], 'K3'), K4 = parseNum(args[4], 'K4');
      // theta0 converted from degrees to radians (angle_class2xe.cpp coeff: theta0/180.0 * MY_PI)
      for (let t = lo; t <= hi; t++) {
        this.theta0[t] = (th / 180) * Math.PI; this.k2[t] = K2; this.k3[t] = K3; this.k4[t] = K4; this.setA[t] = 1;
      }
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!(this.setA[t] && this.setBb[t] && this.setBa[t])) {
        throw new StyleError(`All angle coeffs are not set (type ${t})`);
      }
    }
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${[fmtCoeff((this.theta0[t] * 180) / Math.PI), fmtCoeff(this.k2[t]), fmtCoeff(this.k3[t]), fmtCoeff(this.k4[t])].join(' ')}`);
    }
    return out;
  }

  /** write_data cross terms, in native's order: BondBond (D0 alpha R1 R2), BondAngle (D1 D2 alpha1 alpha2 R1 R2). */
  dataCrossSections(): { title: string; lines: string[] }[] {
    const bb: string[] = [], ba: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      bb.push(`${t} ${fmtCoeff(this.bbD0[t])} ${fmtCoeff(this.bbAlpha[t])} ${fmtCoeff(this.bbR1[t])} ${fmtCoeff(this.bbR2[t])}`);
      ba.push(`${t} ${[this.baD1[t], this.baD2[t], this.baAlpha1[t], this.baAlpha2[t], this.baR1[t], this.baR2[t]].map((v) => fmtCoeff(v)).join(' ')}`);
    }
    return [{ title: 'BondBond Coeffs', lines: bb }, { title: 'BondAngle Coeffs', lines: ba }];
  }

  equilibrium(type: number): number { return (this.theta0[type] * 180) / Math.PI; }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    const f = bc.f;
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      delta(bc, j, i, d1); // r_i - r_j (native delx1)
      delta(bc, j, k, d2); // r_k - r_j (native delx2)
      const rsq1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
      const rsq2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
      const r1 = Math.sqrt(rsq1), r2 = Math.sqrt(rsq2);
      let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
      if (c > 1) c = 1;
      if (c < -1) c = -1;
      let s = Math.sqrt(1 - c * c);
      if (s < SMALL) s = SMALL;
      s = 1 / s;

      // angle term
      const dtheta = libmAcos(c) - this.theta0[t];
      const dtheta2 = dtheta * dtheta, dtheta3 = dtheta2 * dtheta, dtheta4 = dtheta3 * dtheta;
      const deAngle = 2 * this.k2[t] * dtheta + 3 * this.k3[t] * dtheta2 + 4 * this.k4[t] * dtheta3;
      const aA = -deAngle * s;
      const a11 = (aA * c) / rsq1;
      const a12 = -aA / (r1 * r2);
      const a22 = (aA * c) / rsq2;
      const f1 = [0, 0, 0], f3 = [0, 0, 0];
      for (let q = 0; q < 3; q++) {
        f1[q] = a11 * d1[q] + a12 * d2[q];
        f3[q] = a22 * d2[q] + a12 * d1[q];
      }
      let eangle = this.k2[t] * dtheta2 + this.k3[t] * dtheta3 + this.k4[t] * dtheta4;

      // bond-bond term
      const bbD0 = this.bbD0[t], bbAl = this.bbAlpha[t];
      const bbRa1 = libmExp(-bbAl * (r1 - this.bbR1[t]));
      const bbRa2 = libmExp(-bbAl * (r2 - this.bbR2[t]));
      const tk1 = bbD0 * bbAl * bbRa2 * (1 - bbRa1);
      const tk2 = bbD0 * bbAl * bbRa1 * (1 - bbRa2);
      for (let q = 0; q < 3; q++) {
        f1[q] -= (d1[q] * tk2) / r1;
        f3[q] -= (d2[q] * tk1) / r2;
      }
      eangle += bbD0 * (1 - bbRa1) * (1 - bbRa2);

      // bond-angle term
      const baRa1 = libmExp(-this.baAlpha1[t] * (r1 - this.baR1[t]));
      const baRa2 = libmExp(-this.baAlpha2[t] * (r2 - this.baR2[t]));
      const aa1 = this.baD1[t] * (1 - baRa1) * s;
      const aa2 = this.baD2[t] * (1 - baRa2) * s;
      // native: aa21 uses rsq1 in the first block (angle_class2xe.cpp, bond-angle term)
      let aa11 = (aa1 * c) / rsq1;
      const aa12 = -aa1 / (r1 * r2);
      let aa21 = (aa2 * c) / rsq1;
      const aa22 = -aa2 / (r1 * r2);
      const vx11 = [0, 0, 0], vx12 = [0, 0, 0];
      for (let q = 0; q < 3; q++) {
        vx11[q] = aa11 * d1[q] + aa12 * d2[q];
        vx12[q] = aa21 * d1[q] + aa22 * d2[q];
      }
      aa11 = (aa1 * c) / rsq2;
      aa21 = (aa2 * c) / rsq2;
      const vx21 = [0, 0, 0], vx22 = [0, 0, 0];
      for (let q = 0; q < 3; q++) {
        vx21[q] = aa11 * d2[q] + aa12 * d1[q];
        vx22[q] = aa21 * d2[q] + aa22 * d1[q];
      }
      const b1 = (this.baD1[t] * this.baAlpha1[t] * baRa1 * dtheta) / r1;
      const b2 = (this.baD2[t] * this.baAlpha2[t] * baRa2 * dtheta) / r2;
      for (let q = 0; q < 3; q++) {
        f1[q] -= vx11[q] + b1 * d1[q] + vx12[q];
        f3[q] -= vx21[q] + b2 * d2[q] + vx22[q];
      }
      eangle += this.baD1[t] * (1 - baRa1) * dtheta + this.baD2[t] * (1 - baRa2) * dtheta;
      e += eangle;

      for (let q = 0; q < 3; q++) {
        f[3 * i + q] += f1[q];
        f[3 * j + q] -= f1[q] + f3[q];
        f[3 * k + q] += f3[q];
      }
      // native ev_tally (angle_class2xe.cpp, line 261): virial, per-atom shares and the centroid virial from d1, d2, f1, f3
      bondedVirial(bc, [i, j, k], [d1[0], d1[1], d1[2], 0, 0, 0, d2[0], d2[1], d2[2]],
        [f1[0], f1[1], f1[2], -(f1[0] + f3[0]), -(f1[1] + f3[1]), -(f1[2] + f3[2]), f3[0], f3[1], f3[2]], eangle);
    }
    bc.acc.eangle += e;
  }
}
