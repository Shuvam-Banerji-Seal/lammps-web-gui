/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/angle_amoeba.cpp (coeff, compute, tinker_angle, tinker_bondangle, tinker_urey_bradley) and
 * src/AMOEBA/angle_amoeba.h at commit b332f28. AMOEBA package authors: the LAMMPS developers (Sandia Corporation,
 * GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * angle_style amoeba: the Tinker angle (sextic in the angle deviation), the bond-stretch/angle cross term (bondangle,
 * only for types with ba_k1 != 0) and the Urey-Bradley 1-3 term (for types with the ub flag). The in-plane "anglep"
 * variant (pflag = 1) is not ported yet and is refused.
 * docs.lammps.org/angle_amoeba.html: "angle_coeff N ba ..." (BondAngle) and "angle_coeff N ub ..." (Urey-Bradley)
 * and the plain line "pflag ubflag theta0 K2 K3 K4 K5 K6".
 */
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { parseNum } from '../../engine/force/util';

/** SMALL in angle_amoeba.cpp: the lower bound on sin(theta). */
const SMALL = 0.001;

export class AngleAmoeba extends Bonded {
  readonly name: string = 'amoeba';
  readonly kind = 'angle' as const;
  private pflag!: Uint8Array;
  private ubflag!: Uint8Array;
  private theta0!: Float64Array;
  private k2!: Float64Array;
  private k3!: Float64Array;
  private k4!: Float64Array;
  private k5!: Float64Array;
  private k6!: Float64Array;
  private baK1!: Float64Array;
  private baK2!: Float64Array;
  private baR1!: Float64Array;
  private baR2!: Float64Array;
  private ubK!: Float64Array;
  private ubR0!: Float64Array;
  private setA!: Uint8Array;
  private setBa!: Uint8Array;
  private setUb!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`${this.kind}_style ${this.name} takes no arguments`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    this.pflag = new Uint8Array(n);
    this.ubflag = new Uint8Array(n);
    for (const k of ['theta0', 'k2', 'k3', 'k4', 'k5', 'k6', 'baK1', 'baK2', 'baR1', 'baR2', 'ubK', 'ubR0'] as const) this[k] = new Float64Array(n);
    this.setA = new Uint8Array(n);
    this.setBa = new Uint8Array(n);
    this.setUb = new Uint8Array(n);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 2) throw new StyleError('Incorrect args for angle coefficients');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    if (args[1] === 'ba') {
      if (args.length !== 6) throw new StyleError('Incorrect args for angle coefficients (angle_coeff N ba k1 k2 r1 r2)');
      const k1 = parseNum(args[2], 'ba_k1'), k2 = parseNum(args[3], 'ba_k2');
      const r1 = parseNum(args[4], 'ba_r1'), r2 = parseNum(args[5], 'ba_r2');
      for (let t = lo; t <= hi; t++) { this.baK1[t] = k1; this.baK2[t] = k2; this.baR1[t] = r1; this.baR2[t] = r2; this.setBa[t] = 1; }
    } else if (args[1] === 'ub') {
      if (args.length !== 4) throw new StyleError('Incorrect args for angle coefficients (angle_coeff N ub k r0)');
      const k = parseNum(args[2], 'ub_k'), r0 = parseNum(args[3], 'ub_r0');
      for (let t = lo; t <= hi; t++) { this.ubK[t] = k; this.ubR0[t] = r0; this.setUb[t] = 1; }
    } else {
      if (args.length !== 9) throw new StyleError('Incorrect args for angle coefficients (angle_coeff N pflag ubflag theta0 K2 K3 K4 K5 K6)');
      const pflag = Math.trunc(parseNum(args[1], 'pflag')), ubflag = Math.trunc(parseNum(args[2], 'ubflag'));
      const th = parseNum(args[3], 'theta0');
      const K = args.slice(4).map((v, q) => parseNum(v, `K${q + 2}`));
      for (let t = lo; t <= hi; t++) {
        this.pflag[t] = pflag ? 1 : 0;
        this.ubflag[t] = ubflag ? 1 : 0;
        this.theta0[t] = (th / 180.0) * Math.PI;
        this.k2[t] = K[0]; this.k3[t] = K[1]; this.k4[t] = K[2]; this.k5[t] = K[3]; this.k6[t] = K[4];
        this.setA[t] = 1;
      }
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!(this.setA[t] && this.setBa[t] && this.setUb[t])) throw new StyleError(`All angle coeffs are not set (type ${t})`);
      if (this.pflag[t]) throw new StyleError('angle amoeba: the in-plane anglep term (pflag = 1) is not ported yet');
    }
  }

  equilibrium(type: number): number { return (this.theta0[type] * 180) / Math.PI; }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const f = bc.f;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i1 = atomIndex(bc, A.atoms[3 * a], 'angle');
      const i2 = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const i3 = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      // del1 = x_i1 - x_i2, del2 = x_i3 - x_i2 (angle_amoeba.cpp tinker_angle)
      delta(bc, i2, i1, d1);
      delta(bc, i2, i3, d2);
      const r1 = Math.sqrt(d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]);
      const r2 = Math.sqrt(d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]);
      const rsq1 = r1 * r1, rsq2 = r2 * r2;
      let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]);
      c /= r1 * r2;
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      let s = Math.sqrt(1.0 - c * c);
      if (s < SMALL) s = SMALL;
      s = 1.0 / s;
      const dtheta = Math.acos(c) - this.theta0[t];
      // Tinker angle (sextic)
      const dt2 = dtheta * dtheta, dt3 = dt2 * dtheta, dt4 = dt3 * dtheta, dt5 = dt4 * dtheta, dt6 = dt5 * dtheta;
      const deAngle = 2.0 * this.k2[t] * dtheta + 3.0 * this.k3[t] * dt2 + 4.0 * this.k4[t] * dt3
        + 5.0 * this.k5[t] * dt4 + 6.0 * this.k6[t] * dt5;
      const aa = -deAngle * s;
      const a11 = (aa * c) / rsq1;
      const a12 = -aa / (r1 * r2);
      const a22 = (aa * c) / rsq2;
      const f1 = [0, 0, 0], f3 = [0, 0, 0];
      for (let q = 0; q < 3; q++) {
        f1[q] = a11 * d1[q] + a12 * d2[q];
        f3[q] = a22 * d2[q] + a12 * d1[q];
      }
      const eAngle = this.k2[t] * dt2 + this.k3[t] * dt3 + this.k4[t] * dt4 + this.k5[t] * dt5 + this.k6[t] * dt6;
      e += eAngle;
      this.applyAngleForces(f, i1, i2, i3, f1, f3);
      const w0 = [
        d1[0] * f1[0] + d2[0] * f3[0], d1[1] * f1[1] + d2[1] * f3[1], d1[2] * f1[2] + d2[2] * f3[2],
        d1[0] * f1[1] + d2[0] * f3[1], d1[0] * f1[2] + d2[0] * f3[2], d1[1] * f1[2] + d2[1] * f3[2],
      ];
      for (let q = 0; q < 6; q++) bc.virial[q] += w0[q];
      // bond-angle cross term (tinker_bondangle), only when k1 != 0
      if (this.baK1[t] !== 0.0) {
        const dr1 = r1 - this.baR1[t], dr2 = r2 - this.baR2[t];
        const aa1 = s * dr1 * this.baK1[t];
        const aa2 = s * dr2 * this.baK2[t];
        let aa11 = (aa1 * c) / rsq1;
        let aa12 = -aa1 / (r1 * r2);
        let aa21 = (aa2 * c) / rsq1;
        let aa22 = -aa2 / (r1 * r2);
        const vx11 = aa11 * d1[0] + aa12 * d2[0], vx12 = aa21 * d1[0] + aa22 * d2[0];
        const vy11 = aa11 * d1[1] + aa12 * d2[1], vy12 = aa21 * d1[1] + aa22 * d2[1];
        const vz11 = aa11 * d1[2] + aa12 * d2[2], vz12 = aa21 * d1[2] + aa22 * d2[2];
        aa11 = (aa1 * c) / rsq2;
        aa21 = (aa2 * c) / rsq2;
        const vx21 = aa11 * d2[0] + aa12 * d1[0], vx22 = aa21 * d2[0] + aa22 * d1[0];
        const vy21 = aa11 * d2[1] + aa12 * d1[1], vy22 = aa21 * d2[1] + aa22 * d1[1];
        const vz21 = aa11 * d2[2] + aa12 * d1[2], vz22 = aa21 * d2[2] + aa22 * d1[2];
        const b1 = this.baK1[t] * dtheta / r1;
        const b2 = this.baK2[t] * dtheta / r2;
        const g1 = [-(vx11 + b1 * d1[0] + vx12), -(vy11 + b1 * d1[1] + vy12), -(vz11 + b1 * d1[2] + vz12)];
        const g3 = [-(vx21 + b2 * d2[0] + vx22), -(vy21 + b2 * d2[1] + vy22), -(vz21 + b2 * d2[2] + vz22)];
        const eBa = this.baK1[t] * dr1 * dtheta + this.baK2[t] * dr2 * dtheta;
        e += eBa;
        this.applyAngleForces(f, i1, i2, i3, g1, g3);
        const wb = [
          d1[0] * g1[0] + d2[0] * g3[0], d1[1] * g1[1] + d2[1] * g3[1], d1[2] * g1[2] + d2[2] * g3[2],
          d1[0] * g1[1] + d2[0] * g3[1], d1[0] * g1[2] + d2[0] * g3[2], d1[1] * g1[2] + d2[1] * g3[2],
        ];
        for (let q = 0; q < 6; q++) bc.virial[q] += wb[q];
      }
      // Urey-Bradley 1-3 term (tinker_urey_bradley on atoms i1, i3)
      if (this.ubflag[t]) {
        const dl = [0, 0, 0];
        delta(bc, i3, i1, dl);
        const r = Math.sqrt(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]);
        const dr = r - this.ubR0[t];
        const rk = this.ubK[t] * dr;
        const fbond = r > 0.0 ? -2.0 * rk / r : 0.0;
        e += rk * dr;
        f[3 * i1] += dl[0] * fbond; f[3 * i1 + 1] += dl[1] * fbond; f[3 * i1 + 2] += dl[2] * fbond;
        f[3 * i3] -= dl[0] * fbond; f[3 * i3 + 1] -= dl[1] * fbond; f[3 * i3 + 2] -= dl[2] * fbond;
        bc.virial[0] += dl[0] * dl[0] * fbond;
        bc.virial[1] += dl[1] * dl[1] * fbond;
        bc.virial[2] += dl[2] * dl[2] * fbond;
        bc.virial[3] += dl[0] * dl[1] * fbond;
        bc.virial[4] += dl[0] * dl[2] * fbond;
        bc.virial[5] += dl[1] * dl[2] * fbond;
      }
    }
    bc.acc.eangle += e;
  }

  /** Angle forces as angle_amoeba.cpp: f1 on atom 1, -(f1+f3) on atom 2, f3 on atom 3. */
  private applyAngleForces(f: Float64Array, i1: number, i2: number, i3: number, f1: number[], f3: number[]): void {
    for (let q = 0; q < 3; q++) {
      f[3 * i1 + q] += f1[q];
      f[3 * i2 + q] -= f1[q] + f3[q];
      f[3 * i3 + q] += f3[q];
    }
  }
}
