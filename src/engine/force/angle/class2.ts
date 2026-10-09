import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';
import { atomIndex, delta } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';

/*
 * angle_style class2 — docs.lammps.org/angle_class2.html:
 *
 *   E      = E_a + E_bb + E_ba
 *   E_a    = K2 (theta - theta0)^2 + K3 (theta - theta0)^3 + K4 (theta - theta0)^4
 *   E_bb   = M (r_ij - r1)(r_jk - r2)
 *   E_ba   = N1 (r_ij - r1)(theta - theta0) + N2 (r_jk - r2)(theta - theta0)
 *
 * theta_0 "is specified in degrees, but LAMMPS converts it to radians
 * internally". The plain angle_coeff line lists theta0 K2 K3 K4; the bb line
 * M (energy/distance^2), r_1, r_2 (distance); the ba line N_1, N_2
 * (energy/distance), r_1, r_2 (distance). The theta_0 "value in the" E_ba
 * "formula is not specified, since it is the same value from the" E_a
 * "formula." The note says the terms in E_bb and E_ba "will use the I,J atoms
 * to compute" r_ij "and the J,K atoms to compute" r_jk: r1 is the equilibrium of
 * r_ij (I-J) and r2 of r_jk (J-K).
 *
 * Measured with native LAMMPS (black box): the three coefficient lines
 * (plain, bb, ba) must all be set for every angle type or native refuses to
 * run (All angle coeffs are not set). write_data prints the plain line under Angle Coeffs # class2,
 * the bb line under BondBond Coeffs and the ba line under BondAngle Coeffs (dataCrossSections).
 */
export class AngleClass2 extends Bonded {
  readonly name = 'class2';
  readonly kind = 'angle' as const;
  private th0!: Float64Array;
  private K2!: Float64Array;
  private K3!: Float64Array;
  private K4!: Float64Array;
  private M!: Float64Array;
  private r1bb!: Float64Array;
  private r2bb!: Float64Array;
  private N1!: Float64Array;
  private N2!: Float64Array;
  private r1ba!: Float64Array;
  private r2ba!: Float64Array;
  private setA!: Uint8Array;
  private setBb!: Uint8Array;
  private setBa!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`${this.kind}_style ${this.name} takes no arguments`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    this.th0 = new Float64Array(n);
    this.K2 = new Float64Array(n);
    this.K3 = new Float64Array(n);
    this.K4 = new Float64Array(n);
    this.M = new Float64Array(n);
    this.r1bb = new Float64Array(n);
    this.r2bb = new Float64Array(n);
    this.N1 = new Float64Array(n);
    this.N2 = new Float64Array(n);
    this.r1ba = new Float64Array(n);
    this.r2ba = new Float64Array(n);
    this.setA = new Uint8Array(n);
    this.setBb = new Uint8Array(n);
    this.setBa = new Uint8Array(n);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (!args.length) throw new StyleError(`usage: ${this.kind}_coeff N coefficients`);
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const kw = args[1];
    if (kw === 'bb') {
      if (args.length !== 5) throw new StyleError('angle_coeff class2 bb needs M r1 r2');
      const M = parseNum(args[2], 'M'), r1 = parseNum(args[3], 'r1'), r2 = parseNum(args[4], 'r2');
      for (let t = lo; t <= hi; t++) { this.M[t] = M; this.r1bb[t] = r1; this.r2bb[t] = r2; this.setBb[t] = 1; }
    } else if (kw === 'ba') {
      if (args.length !== 6) throw new StyleError('angle_coeff class2 ba needs N1 N2 r1 r2');
      const N1 = parseNum(args[2], 'N1'), N2 = parseNum(args[3], 'N2');
      const r1 = parseNum(args[4], 'r1'), r2 = parseNum(args[5], 'r2');
      for (let t = lo; t <= hi; t++) { this.N1[t] = N1; this.N2[t] = N2; this.r1ba[t] = r1; this.r2ba[t] = r2; this.setBa[t] = 1; }
    } else {
      if (args.length !== 5) throw new StyleError('angle_coeff class2 needs theta0 K2 K3 K4');
      const th = (parseNum(args[1], 'theta0') * Math.PI) / 180;
      const K2 = parseNum(args[2], 'K2'), K3 = parseNum(args[3], 'K3'), K4 = parseNum(args[4], 'K4');
      for (let t = lo; t <= hi; t++) { this.th0[t] = th; this.K2[t] = K2; this.K3[t] = K3; this.K4[t] = K4; this.setA[t] = 1; }
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!(this.setA[t] && this.setBb[t] && this.setBa[t])) {
        throw new StyleError(`all angle coeffs are not set (type ${t})`);
      }
    }
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${fmtCoeff((this.th0[t] * 180) / Math.PI)} ${fmtCoeff(this.K2[t])} ${fmtCoeff(this.K3[t])} ${fmtCoeff(this.K4[t])}`);
    }
    return out;
  }

  /** write_data cross terms (order measured, see output/data.ts): BondBond M r1 r2, BondAngle N1 N2 r1 r2. */
  dataCrossSections(): { title: string; lines: string[] }[] {
    const bb: string[] = [], ba: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      bb.push(`${t} ${fmtCoeff(this.M[t])} ${fmtCoeff(this.r1bb[t])} ${fmtCoeff(this.r2bb[t])}`);
      ba.push(`${t} ${fmtCoeff(this.N1[t])} ${fmtCoeff(this.N2[t])} ${fmtCoeff(this.r1ba[t])} ${fmtCoeff(this.r2ba[t])}`);
    }
    return [{ title: 'BondBond Coeffs', lines: bb }, { title: 'BondAngle Coeffs', lines: ba }];
  }

  equilibrium(type: number): number { return (this.th0[type] * 180) / Math.PI; }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      delta(bc, j, i, d1); // r_i - r_j
      delta(bc, j, k, d2); // r_k - r_j
      const r1n = Math.hypot(d1[0], d1[1], d1[2]);
      const r2n = Math.hypot(d2[0], d2[1], d2[2]);
      let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1n * r2n);
      if (c > 1) c = 1;
      if (c < -1) c = -1;
      const theta = Math.acos(c);
      const dth = theta - this.th0[t];
      const ea = this.K2[t] * dth * dth + this.K3[t] * dth ** 3 + this.K4[t] * dth ** 4;
      const ebb = this.M[t] * (r1n - this.r1bb[t]) * (r2n - this.r2bb[t]);
      const eba = (this.N1[t] * (r1n - this.r1ba[t]) + this.N2[t] * (r2n - this.r2ba[t])) * dth;
      const e1 = ea + ebb + eba;
      e += e1;
      // dE/dtheta and dE/dr
      const dEdth = 2 * this.K2[t] * dth + 3 * this.K3[t] * dth * dth + 4 * this.K4[t] * dth ** 3
        + this.N1[t] * (r1n - this.r1ba[t]) + this.N2[t] * (r2n - this.r2ba[t]);
      const dEdr1 = this.M[t] * (r2n - this.r2bb[t]) + this.N1[t] * dth;
      const dEdr2 = this.M[t] * (r1n - this.r1bb[t]) + this.N2[t] * dth;
      let s = Math.sqrt(1 - c * c);
      if (s < 0.001) s = 0.001;
      const D = dEdth / s;
      const f1 = [0, 0, 0], f3 = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        f1[d] = D * (d2[d] / (r1n * r2n) - (c * d1[d]) / (r1n * r1n)) - (dEdr1 * d1[d]) / r1n;
        f3[d] = D * (d1[d] / (r1n * r2n) - (c * d2[d]) / (r2n * r2n)) - (dEdr2 * d2[d]) / r2n;
      }
      const f = bc.f;
      for (let d = 0; d < 3; d++) {
        f[3 * i + d] += f1[d];
        f[3 * j + d] -= f1[d] + f3[d];
        f[3 * k + d] += f3[d];
      }
      const w = [
        d1[0] * f1[0] + d2[0] * f3[0], d1[1] * f1[1] + d2[1] * f3[1], d1[2] * f1[2] + d2[2] * f3[2],
        d1[0] * f1[1] + d2[0] * f3[1], d1[0] * f1[2] + d2[0] * f3[2], d1[1] * f1[2] + d2[1] * f3[2],
      ];
      for (let q = 0; q < 6; q++) bc.virial[q] += w[q];
      if (bc.eatom) { const third = e1 / 3; bc.eatom[i] += third; bc.eatom[j] += third; bc.eatom[k] += third; }
      if (bc.vatom) for (const at of [i, j, k]) for (let q = 0; q < 6; q++) bc.vatom[6 * at + q] += w[q] / 3;
    }
    bc.acc.eangle += e;
  }
}
