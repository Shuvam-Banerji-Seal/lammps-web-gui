/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/angle_amoeba.cpp (coeff, compute, tinker_angle, tinker_bondangle, tinker_urey_bradley) and
 * src/AMOEBA/angle_amoeba.h at commit b332f28. AMOEBA package authors: the LAMMPS developers (Sandia Corporation,
 * GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * angle_style amoeba: the Tinker angle (sextic in the angle deviation), the bond-stretch/angle cross term (bondangle,
 * only for types with ba_k1 != 0) and the Urey-Bradley 1-3 term (for types with the ub flag). The in-plane "anglep"
 * variant (pflag = 1, tinker_anglep) applies when atom 2 has exactly three 1-2 partners; the fourth atom is that partner
 * other than atoms 1 and 3 (angle_amoeba.cpp compute).
 * docs.lammps.org/angle_amoeba.html: "angle_coeff N ba ..." (BondAngle) and "angle_coeff N ub ..." (Urey-Bradley)
 * and the plain line "pflag ubflag theta0 K2 K3 K4 K5 K6".
 */
import { Bonded, StyleError, bondedVirial, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { parseNum } from '../../engine/force/util';
import { acos as libmAcos } from '../libm/acos';

/** SMALL in angle_amoeba.cpp: the lower bound on sin(theta). */
const SMALL = 0.001;

/** 1-2 partner IDs of every atom (the bonds of the topology; bond type 0 is turned off and not counted). */
export const partnerMap = (bc: BondedCompute): Map<number, number[]> => {
  const out = new Map<number, number[]>();
  const B = bc.s.topo.bonds;
  for (let e = 0; e < B.n; e++) {
    if (B.type[e] === 0) continue;
    const a = B.atoms[2 * e], b = B.atoms[2 * e + 1];
    if (!out.has(a)) out.set(a, []);
    if (!out.has(b)) out.set(b, []);
    out.get(a)!.push(b);
    out.get(b)!.push(a);
  }
  return out;
};

/** Angle::ev_tally for the Tinker angle i1-i2-i3 (d1 = r1 - r2, d2 = r3 - r2, forces f1 and f3): bondedVirial. */
const tallyAngleEv = (
  bc: BondedCompute, i1: number, i2: number, i3: number, e: number,
  d1: number[], d2: number[], f1: number[], f3: number[],
): void => {
  bondedVirial(bc, [i1, i2, i3], [d1[0], d1[1], d1[2], 0, 0, 0, d2[0], d2[1], d2[2]],
    [f1[0], f1[1], f1[2], -(f1[0] + f3[0]), -(f1[1] + f3[1]), -(f1[2] + f3[2]), f3[0], f3[1], f3[2]], e);
};

/**
 * Angle::ev_tally4 and Angle::ev_tally2 (angle.cpp at b332f28): the energy and the virial are shared equally by the atoms
 * (the four of the in-plane term, the two of the 1-3 bond), and the per-atom virial takes the same shares. The centroid
 * virial is not tallied here (cvatom is left to the Angle::ev_tally terms).
 */
const tallyNoCentroid = (bc: BondedCompute, atoms: number[], e: number, vir: number[]): void => {
  for (let q = 0; q < 6; q++) bc.virial[q] += vir[q];
  const m = atoms.length;
  if (bc.eatom) for (const a of atoms) bc.eatom[a] += e / m;
  if (bc.vatom) for (const a of atoms) for (let q = 0; q < 6; q++) bc.vatom[6 * a + q] += vir[q] / m;
};

export class AngleAmoeba extends Bonded {
  readonly name: string = 'amoeba';
  readonly kind = 'angle' as const;
  /**
   * The per-atom centroid virial is native's, not the virial split: native angle_amoeba.cpp tallies the sextic angle and
   * the bond-angle terms with Angle::ev_tally (centroid virial included), but the in-plane anglep term with ev_tally4 and
   * the Urey-Bradley term with ev_tally2, and neither of those touches cvatom (angle.cpp, ev_tally4 and ev_tally2 at
   * b332f28). So the centroid sum of this style differs from its virial by those terms, as in native.
   */
  readonly centroidNative = true;
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
    }
  }

  /** angle_amoeba.cpp equilibrium_angle(): theta0 (radians) */
  equilibriumRadians(type: number): number { return this.theta0[type]; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const f = bc.f;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    let partners: Map<number, number[]> | null = null;
    for (let a = 0; a < A.n; a++) {
      const id1 = A.atoms[3 * a], id2 = A.atoms[3 * a + 1], id3 = A.atoms[3 * a + 2];
      const i1 = atomIndex(bc, id1, 'angle');
      const i2 = atomIndex(bc, id2, 'angle');
      const i3 = atomIndex(bc, id3, 'angle');
      const t = A.type[a];
      // del1 = x_i1 - x_i2, del2 = x_i3 - x_i2 (angle_amoeba.cpp tinker_angle)
      delta(bc, i2, i1, d1);
      delta(bc, i2, i3, d2);
      // tinker_anglep (in-plane) when atom 2 has exactly three 1-2 partners (nspecial[i2][0] == 3), else tinker_angle
      let inPlane = false;
      if (this.pflag[t]) {
        if (partners === null) partners = partnerMap(bc);
        const p2 = partners.get(id2) ?? [];
        if (p2.length === 3) {
          inPlane = true;
          const id4 = p2.find((q) => q !== id1 && q !== id3);
          if (id4 === undefined) throw new StyleError('angle amoeba: no fourth atom for the in-plane angle');
          const i4 = atomIndex(bc, id4, 'angle');
          const d4 = [0, 0, 0];
          delta(bc, i2, i4, d4);
          e += this.anglepTerm(bc, f, i1, i2, i3, i4, t, d1, d2, d4);
        }
      }
      if (!inPlane) e += this.angleTerm(bc, f, i1, i2, i3, t, d1, d2);
      // bond-angle cross term (tinker_bondangle), only when k1 != 0
      if (this.baK1[t] !== 0.0) e += this.bondAngleTerm(bc, f, i1, i2, i3, t, d1, d2);
      // Urey-Bradley 1-3 term (tinker_urey_bradley on atoms i1, i3)
      if (this.ubflag[t]) e += this.ureyBradleyTerm(bc, f, i1, i3, t);
    }
    bc.acc.eangle += e;
  }

  /** tinker_angle: the sextic angle term of the angle i1-i2-i3; returns its energy and tallies the virial. */
  private angleTerm(bc: BondedCompute, f: Float64Array, i1: number, i2: number, i3: number, t: number, d1: number[], d2: number[]): number {
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
    const dtheta = libmAcos(c) - this.theta0[t];
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
    this.applyAngleForces(f, i1, i2, i3, f1, f3);
    // Angle::ev_tally: virial, per-atom shares and the centroid virial (r_a - r_0) (x) f_a of the three atoms
    tallyAngleEv(bc, i1, i2, i3, eAngle, d1, d2, f1, f3);
    return eAngle;
  }

  /**
   * tinker_anglep (angle_amoeba.cpp): the in-plane angle of atom 2 with its three partners 1, 3 and 4. The projection of
   * atom 1 on the plane of 2, 3 and 4 gives the angle; the forces are the derivatives of the sextic with respect to the
   * four atoms (relative positions with atom 2 at the origin, which leaves every difference unchanged). Returns the
   * energy; a degenerate geometry (rap2 or rcp2 zero) gives no energy and no force, as native.
   */
  private anglepTerm(
    bc: BondedCompute, f: Float64Array, i1: number, i2: number, i3: number, i4: number, t: number,
    d1: number[], d2: number[], d4: number[],
  ): number {
    // positions relative to atom 2: a = atom 1, b = atom 2 (origin), c = atom 3, d = atom 4
    const xia = d1[0], yia = d1[1], zia = d1[2];
    const xib = 0, yib = 0, zib = 0;
    const xic = d2[0], yic = d2[1], zic = d2[2];
    const xid = d4[0], yid = d4[1], zid = d4[2];
    const xad = xia - xid, yad = yia - yid, zad = zia - zid;
    const xbd = xib - xid, ybd = yib - yid, zbd = zib - zid;
    const xcd = xic - xid, ycd = yic - yid, zcd = zic - zid;
    const xt = yad * zcd - zad * ycd, yt = zad * xcd - xad * zcd, zt = xad * ycd - yad * xcd;
    const rt2 = xt * xt + yt * yt + zt * zt;
    const delta0 = -(xt * xbd + yt * ybd + zt * zbd) / rt2;
    const xip = xib + xt * delta0, yip = yib + yt * delta0, zip = zib + zt * delta0;
    const xap = xia - xip, yap = yia - yip, zap = zia - zip;
    const xcp = xic - xip, ycp = yic - yip, zcp = zic - zip;
    const rap2 = xap * xap + yap * yap + zap * zap;
    const rcp2 = xcp * xcp + ycp * ycp + zcp * zcp;
    if (rap2 === 0.0 || rcp2 === 0.0) return 0.0;
    const xm = ycp * zap - zcp * yap, ym = zcp * xap - xcp * zap, zm = xcp * yap - ycp * xap;
    const rm = Math.max(Math.sqrt(xm * xm + ym * ym + zm * zm), 0.0001);
    const dot = xap * xcp + yap * ycp + zap * zcp;
    let cosine = dot / Math.sqrt(rap2 * rcp2);
    cosine = Math.min(1.0, Math.max(-1.0, cosine));
    const dtheta = libmAcos(cosine) - this.theta0[t];
    const dtheta2 = dtheta * dtheta, dtheta3 = dtheta2 * dtheta, dtheta4 = dtheta3 * dtheta;
    const dtheta5 = dtheta4 * dtheta, dtheta6 = dtheta5 * dtheta;
    const deddt = 2.0 * this.k2[t] * dtheta + 3.0 * this.k3[t] * dtheta2 + 4.0 * this.k4[t] * dtheta3 +
      5.0 * this.k5[t] * dtheta4 + 6.0 * this.k6[t] * dtheta5;
    const eangle = this.k2[t] * dtheta2 + this.k3[t] * dtheta3 + this.k4[t] * dtheta4 + this.k5[t] * dtheta5 + this.k6[t] * dtheta6;
    const terma = -deddt / (rap2 * rm);
    const termc = deddt / (rcp2 * rm);
    const f1 = [terma * (yap * zm - zap * ym), terma * (zap * xm - xap * zm), terma * (xap * ym - yap * xm)];
    const f3 = [termc * (ycp * zm - zcp * ym), termc * (zcp * xm - xcp * zm), termc * (xcp * ym - ycp * xm)];
    const dedxip = -f1[0] - f3[0], dedyip = -f1[1] - f3[1], dedzip = -f1[2] - f3[2];
    const delta2 = 2.0 * delta0;
    const ptrt2 = (dedxip * xt + dedyip * yt + dedzip * zt) / rt2;
    let term = (zcd * ybd - ycd * zbd) + delta2 * (yt * zcd - zt * ycd);
    const dpdxia = delta0 * (ycd * dedzip - zcd * dedyip) + term * ptrt2;
    term = (xcd * zbd - zcd * xbd) + delta2 * (zt * xcd - xt * zcd);
    const dpdyia = delta0 * (zcd * dedxip - xcd * dedzip) + term * ptrt2;
    term = (ycd * xbd - xcd * ybd) + delta2 * (xt * ycd - yt * xcd);
    const dpdzia = delta0 * (xcd * dedyip - ycd * dedxip) + term * ptrt2;
    term = (yad * zbd - zad * ybd) + delta2 * (zt * yad - yt * zad);
    const dpdxic = delta0 * (zad * dedyip - yad * dedzip) + term * ptrt2;
    term = (zad * xbd - xad * zbd) + delta2 * (xt * zad - zt * xad);
    const dpdyic = delta0 * (xad * dedzip - zad * dedxip) + term * ptrt2;
    term = (xad * ybd - yad * xbd) + delta2 * (yt * xad - xt * yad);
    const dpdzic = delta0 * (yad * dedxip - xad * dedyip) + term * ptrt2;
    f1[0] += dpdxia; f1[1] += dpdyia; f1[2] += dpdzia;
    const f2 = [dedxip, dedyip, dedzip];
    f3[0] += dpdxic; f3[1] += dpdyic; f3[2] += dpdzic;
    const f4 = [-f1[0] - f2[0] - f3[0], -f1[1] - f2[1] - f3[1], -f1[2] - f2[2] - f3[2]];
    // forces as native: f_k -= f_k (native signs); the virial is sum over atoms 1, 3, 4 of (x_k - x_2) times the force
    const fk: [number, number, number, number][] = [[i1, f1[0], f1[1], f1[2]], [i2, f2[0], f2[1], f2[2]], [i3, f3[0], f3[1], f3[2]], [i4, f4[0], f4[1], f4[2]]];
    for (const [idx, a, b, c] of fk) {
      f[3 * idx] -= a;
      f[3 * idx + 1] -= b;
      f[3 * idx + 2] -= c;
    }
    const rel: [number[], number[], number[]] = [d1, d2, d4];
    const forces: [number[], number[], number[]] = [f1, f3, f4];
    const vir = [0, 0, 0, 0, 0, 0];
    for (let q = 0; q < 3; q++) {
      const r = rel[q], g = forces[q];
      // actual force = -native f
      vir[0] -= r[0] * g[0];
      vir[1] -= r[1] * g[1];
      vir[2] -= r[2] * g[2];
      vir[3] -= 0.5 * (r[0] * g[1] + r[1] * g[0]);
      vir[4] -= 0.5 * (r[0] * g[2] + r[2] * g[0]);
      vir[5] -= 0.5 * (r[1] * g[2] + r[2] * g[1]);
    }
    tallyNoCentroid(bc, [i1, i2, i3, i4], eangle, vir);
    return eangle;
  }

  /** tinker_bondangle: the bond-stretch and angle cross term of i1-i2-i3 (only when ba_k1 != 0); returns its energy. */
  private bondAngleTerm(bc: BondedCompute, f: Float64Array, i1: number, i2: number, i3: number, t: number, d1: number[], d2: number[]): number {
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
    const dtheta = libmAcos(c) - this.theta0[t];
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
    this.applyAngleForces(f, i1, i2, i3, g1, g3);
    tallyAngleEv(bc, i1, i2, i3, eBa, d1, d2, g1, g3);
    return eBa;
  }

  /** Urey-Bradley 1-3 term between atoms i1 and i3 (tinker_urey_bradley); returns its energy. */
  private ureyBradleyTerm(bc: BondedCompute, f: Float64Array, i1: number, i3: number, t: number): number {
    const dl = [0, 0, 0];
    delta(bc, i3, i1, dl);
    const r = Math.sqrt(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]);
    const dr = r - this.ubR0[t];
    const rk = this.ubK[t] * dr;
    const fbond = r > 0.0 ? -2.0 * rk / r : 0.0;
    f[3 * i1] += dl[0] * fbond; f[3 * i1 + 1] += dl[1] * fbond; f[3 * i1 + 2] += dl[2] * fbond;
    f[3 * i3] -= dl[0] * fbond; f[3 * i3 + 1] -= dl[1] * fbond; f[3 * i3 + 2] -= dl[2] * fbond;
    // Angle::ev_tally2 (bond-style tally): the virial of the 1-3 bond, shared by its two atoms, no centroid virial
    const vir = [dl[0] * dl[0] * fbond, dl[1] * dl[1] * fbond, dl[2] * dl[2] * fbond, dl[0] * dl[1] * fbond, dl[0] * dl[2] * fbond, dl[1] * dl[2] * fbond];
    tallyNoCentroid(bc, [i1, i3], rk * dr, vir);
    return rk * dr;
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
