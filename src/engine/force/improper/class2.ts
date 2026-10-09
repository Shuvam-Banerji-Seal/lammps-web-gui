import { Bonded, StyleError, typeBounds, bondedVirial, type BondedCompute, type StyleContext } from '../types';
import { atomIndex, delta } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';

/*
 * improper_style class2 — docs.lammps.org/improper_class2.html:
 *
 *   E      = E_i + E_aa
 *   E_i    = K [ (chi_ijkl + chi_kjli + chi_ljik)/3 - chi0 ]^2
 *   E_aa   = M1 (theta_ijk - theta1)(theta_kjl - theta3)
 *          + M2 (theta_ijk - theta1)(theta_ijl - theta2)
 *          + M3 (theta_ijl - theta2)(theta_kjl - theta3)
 *
 * "The 4 atoms in an improper quadruplet ... are ordered I,J,K,L." chi_ijkl
 * "refers to the angle between the plane of I,J,K and the plane of J,K,L, and
 * the bond JK lies in both planes." "Note that atom J appears in the common
 * bonds (JI, JK, JL) of all 3 X terms." theta_ijl "is the angle formed by atoms
 * I,J,L with J in the middle." theta_1, theta_2, theta_3 "are the equilibrium
 * positions of those angles." Plain line: K (energy), chi0 (degrees); aa line:
 * M1, M2, M3 (energy), theta1, theta2, theta3 (degrees). The theta values "are
 * specified in degrees, but LAMMPS converts them to radians internally".
 *
 * Measured with native LAMMPS (black box, isolated impropers): the three
 * chi_ijkl in the doc are the signed out-of-plane angles of atom I (resp. K,
 * L) above the plane JKL (resp. JLI, JIK), and the energies are exactly the
 * doc expressions above (chi = asin of the normalized triple product). Both
 * plain and aa coefficient lines are
 * required for every type; with K = 0 the chi block is skipped (chi0 is
 * irrelevant) and only E_aa acts. write_data prints the plain line under
 * Improper Coeffs # class2 and the aa line under AngleAngle Coeffs
 * (dataCrossSections).
 */
export class ImproperClass2 extends Bonded {
  readonly name = 'class2';
  readonly kind = 'improper' as const;
  private K!: Float64Array;
  private chi0!: Float64Array;
  private M1!: Float64Array;
  private M2!: Float64Array;
  private M3!: Float64Array;
  private t1!: Float64Array;
  private t2!: Float64Array;
  private t3!: Float64Array;
  private setPlain!: Uint8Array;
  private setAa!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`${this.kind}_style ${this.name} takes no arguments`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    this.K = new Float64Array(n);
    this.chi0 = new Float64Array(n);
    this.M1 = new Float64Array(n);
    this.M2 = new Float64Array(n);
    this.M3 = new Float64Array(n);
    this.t1 = new Float64Array(n);
    this.t2 = new Float64Array(n);
    this.t3 = new Float64Array(n);
    this.setPlain = new Uint8Array(n);
    this.setAa = new Uint8Array(n);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (!args.length) throw new StyleError(`usage: ${this.kind}_coeff N coefficients`);
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    if (args[1] === 'aa') {
      if (args.length !== 8) throw new StyleError('improper_coeff class2 aa needs M1 M2 M3 theta1 theta2 theta3');
      const M1 = parseNum(args[2], 'M1'), M2 = parseNum(args[3], 'M2'), M3 = parseNum(args[4], 'M3');
      const t1 = (parseNum(args[5], 'theta1') * Math.PI) / 180;
      const t2 = (parseNum(args[6], 'theta2') * Math.PI) / 180;
      const t3 = (parseNum(args[7], 'theta3') * Math.PI) / 180;
      for (let t = lo; t <= hi; t++) {
        this.M1[t] = M1; this.M2[t] = M2; this.M3[t] = M3;
        this.t1[t] = t1; this.t2[t] = t2; this.t3[t] = t3;
        this.setAa[t] = 1;
      }
    } else {
      if (args.length !== 3) throw new StyleError('improper_coeff class2 needs K chi0');
      const K = parseNum(args[1], 'K');
      const c0 = (parseNum(args[2], 'chi0') * Math.PI) / 180;
      for (let t = lo; t <= hi; t++) { this.K[t] = K; this.chi0[t] = c0; this.setPlain[t] = 1; }
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!(this.setPlain[t] && this.setAa[t])) {
        throw new StyleError(`all improper coeffs are not set (type ${t})`);
      }
    }
  }

  /** write_data cross terms (see output/data.ts): AngleAngle M1 M2 M3 theta1 theta2 theta3 (degrees). */
  dataCrossSections(): { title: string; lines: string[] }[] {
    const deg = (r: number) => fmtCoeff((r * 180) / Math.PI);
    const aa: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      aa.push(`${t} ${fmtCoeff(this.M1[t])} ${fmtCoeff(this.M2[t])} ${fmtCoeff(this.M3[t])} ${deg(this.t1[t])} ${deg(this.t2[t])} ${deg(this.t3[t])}`);
    }
    return [{ title: 'AngleAngle Coeffs', lines: aa }];
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${fmtCoeff(this.K[t])} ${fmtCoeff((this.chi0[t] * 180) / Math.PI)}`);
    }
    return out;
  }

  /**
   * Out-of-plane angle of `apex` above the plane (centre, p1, p2), signed by
   * the triple product. Fills grad[3*k + d] with d(angle)/d(coord d of
   * atoms[k]) for k = ci, p1i, p2i, ai. From
   *   u = (v . n) / (|v| |n|),  v = r_apex - r_centre,  n = (r_p1 - r_centre) x (r_p2 - r_centre),
   *   du = w da - (u/2) (dv^2/v^2 + dn^2/n^2),  w = 1/sqrt(v^2 n^2),
   *   d(angle) = du / sqrt(1 - u^2)   (angle = asin(u)).
   * The scalar gradients da, dv^2, dn^2 are those of umbrellaC in
   * improper/styles.ts (checked there against central finite differences).
   */
  private oopOne(bc: BondedCompute, atoms: readonly number[], ci: number, ai: number, p1i: number, p2i: number, grad: number[]): number {
    const C = atoms[ci], Ap = atoms[ai], P1 = atoms[p1i], P2 = atoms[p2i];
    const v = [0, 0, 0], b1 = [0, 0, 0], b2 = [0, 0, 0];
    delta(bc, C, Ap, v);
    delta(bc, C, P1, b1);
    delta(bc, C, P2, b2);
    const nx = b1[1] * b2[2] - b1[2] * b2[1], ny = b1[2] * b2[0] - b1[0] * b2[2], nz = b1[0] * b2[1] - b1[1] * b2[0];
    const a = v[0] * nx + v[1] * ny + v[2] * nz;
    const v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const nv2 = nx * nx + ny * ny + nz * nz;
    const den = v2 * nv2;
    if (den <= 0) { grad.fill(0); return 0; }
    const w = 1 / Math.sqrt(den);
    const u = a * w;
    const gjx = b2[1] * v[2] - b2[2] * v[1], gjy = b2[2] * v[0] - b2[0] * v[2], gjz = b2[0] * v[1] - b2[1] * v[0]; // b2 x v
    const gkx = v[1] * b1[2] - v[2] * b1[1], gky = v[2] * b1[0] - v[0] * b1[2], gkz = v[0] * b1[1] - v[1] * b1[0]; // v x b1
    const ga = [
      [-(nx + gjx + gkx), -(ny + gjy + gky), -(nz + gjz + gkz)],
      [gjx, gjy, gjz],
      [gkx, gky, gkz],
      [nx, ny, nz],
    ];
    const dbx = b2[0] - b1[0], dby = b2[1] - b1[1], dbz = b2[2] - b1[2];
    const gn = [
      [2 * (ny * dbz - nz * dby), 2 * (nz * dbx - nx * dbz), 2 * (nx * dby - ny * dbx)],
      [2 * (b2[1] * nz - b2[2] * ny), 2 * (b2[2] * nx - b2[0] * nz), 2 * (b2[0] * ny - b2[1] * nx)],
      [2 * (ny * b1[2] - nz * b1[1]), 2 * (nz * b1[0] - nx * b1[2]), 2 * (nx * b1[1] - ny * b1[0])],
      [0, 0, 0],
    ];
    const gv2 = [[-2 * v[0], -2 * v[1], -2 * v[2]], [0, 0, 0], [0, 0, 0], [2 * v[0], 2 * v[1], 2 * v[2]]];
    const invSin = 1 / Math.sqrt(Math.max(1e-12, 1 - u * u));
    const localIdx = [ci, p1i, p2i, ai];
    for (let p = 0; p < 4; p++) {
      const gvi = localIdx[p];
      for (let d = 0; d < 3; d++) {
        const du = w * ga[p][d] - (u / 2) * (gv2[p][d] / v2 + gn[p][d] / nv2);
        grad[3 * gvi + d] += du * invSin;
      }
    }
    return Math.asin(Math.max(-1, Math.min(1, u)));
  }

  /** Angle at J of the triplet (p, J, q) in radians. */
  private angleAt(bc: BondedCompute, atoms: readonly number[], pi: number, ji: number, qi: number): number {
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    delta(bc, atoms[ji], atoms[pi], d1);
    delta(bc, atoms[ji], atoms[qi], d2);
    const r1 = Math.hypot(d1[0], d1[1], d1[2]), r2 = Math.hypot(d2[0], d2[1], d2[2]);
    let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
    if (c > 1) c = 1;
    if (c < -1) c = -1;
    return Math.acos(c);
  }

  /** Adds f = -coef * d(theta)/dx for the triplet to fk (indexed by the positions pi, ji, qi). */
  private addAngleForce(bc: BondedCompute, atoms: readonly number[], pi: number, ji: number, qi: number, coef: number, fk: number[]): void {
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    delta(bc, atoms[ji], atoms[pi], d1);
    delta(bc, atoms[ji], atoms[qi], d2);
    const r1 = Math.hypot(d1[0], d1[1], d1[2]), r2 = Math.hypot(d2[0], d2[1], d2[2]);
    let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
    if (c > 1) c = 1;
    if (c < -1) c = -1;
    let s = Math.sqrt(1 - c * c);
    if (s < 0.001) s = 0.001;
    const inv = 1 / s;
    for (let d = 0; d < 3; d++) {
      const dp = -inv * (d2[d] / (r1 * r2) - (c * d1[d]) / (r1 * r1));
      const dq = -inv * (d1[d] / (r1 * r2) - (c * d2[d]) / (r2 * r2));
      const dj = -(dp + dq);
      fk[3 * pi + d] += -coef * dp;
      fk[3 * qi + d] += -coef * dq;
      fk[3 * ji + d] += -coef * dj;
    }
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const fk = new Array(12).fill(0);
    const rel = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      fk.fill(0);
      let ei = 0;
      if (this.K[t] !== 0) {
        const g1 = new Array(12).fill(0), g2 = new Array(12).fill(0), g3 = new Array(12).fill(0);
        const o1 = this.oopOne(bc, atoms, 1, 0, 2, 3, g1);
        const o2 = this.oopOne(bc, atoms, 1, 2, 3, 0, g2);
        const o3 = this.oopOne(bc, atoms, 1, 3, 0, 2, g3);
        const P = (o1 + o2 + o3) / 3 - this.chi0[t];
        ei = this.K[t] * P * P;
        const coef = (2 * this.K[t] * P) / 3;
        for (let p = 0; p < 12; p++) fk[p] += -coef * (g1[p] + g2[p] + g3[p]);
      }
      const th1 = this.t1[t], th2 = this.t2[t], th3 = this.t3[t];
      const a = this.angleAt(bc, atoms, 0, 1, 2);
      const b = this.angleAt(bc, atoms, 0, 1, 3);
      const c = this.angleAt(bc, atoms, 2, 1, 3);
      const da = a - th1, db = b - th2, dc = c - th3;
      const eaa = this.M1[t] * da * dc + this.M2[t] * da * db + this.M3[t] * db * dc;
      this.addAngleForce(bc, atoms, 0, 1, 2, this.M1[t] * dc + this.M2[t] * db, fk);
      this.addAngleForce(bc, atoms, 0, 1, 3, this.M2[t] * da + this.M3[t] * dc, fk);
      this.addAngleForce(bc, atoms, 2, 1, 3, this.M1[t] * da + this.M3[t] * db, fk);
      const et = ei + eaa;
      e += et;
      const f = bc.f;
      for (let p = 0; p < 4; p++) for (let d = 0; d < 3; d++) f[3 * atoms[p] + d] += fk[3 * p + d];
      for (let p = 0; p < 4; p++) delta(bc, atoms[0], atoms[p], rel, 3 * p);
      bondedVirial(bc, atoms, rel, fk, et);
    }
    bc.acc.eimp += e;
  }
}
