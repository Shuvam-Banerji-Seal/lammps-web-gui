import { SimpleBonded, atomIndex, termDelta } from '../bonded_util';
import { angleTerm } from './harmonic';
import { fmtCoeff, parseInt_, parseNum } from '../util';
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';
import { libm } from '../../libm';

/*
 * Angle styles added by wave 1. Each class cites its docs.lammps.org page and
 * quotes the formula / coefficient sentences it implements. theta is the
 * angle at the middle atom j of the triplet i-j-k, computed by angleTerm.
 */

/**
 * Degree-to-radian forms, each the expression native LAMMPS uses in that style's coeff():
 * 'DEG2RAD' is DEG2RAD * v, 'div180-pi' is v / 180.0 * MY_PI, 'pi-div180' is v * MY_PI / 180.0.
 */
type DegForm = 'DEG2RAD' | 'div180-pi' | 'pi-div180';
const degToRad = (form: DegForm, v: number): number =>
  form === 'DEG2RAD' ? (Math.PI / 180) * v : form === 'div180-pi' ? (v / 180) * Math.PI : (v * Math.PI) / 180;

/** Parses angle_coeff values, converting the named angle coefficients from degrees to radians with the given native form. */
const parseDeg = (names: readonly string[], degIdx: readonly number[], form: DegForm) => (args: string[]): number[] => {
  if (args.length !== names.length) throw new StyleError(`angle_coeff: expected ${names.length} coefficients: ${names.join(' ')}`);
  return args.map((w, k) => {
    const v = parseNum(w, names[k]);
    return degIdx.includes(k) ? degToRad(form, v) : v;
  });
};

/*
 * angle_style cosine — docs.lammps.org/angle_cosine.html:
 *   E = K [1 + cos(theta)]
 *   "K is defined for each angle type." Coefficient: "K (energy)".
 * Implemented via dE/dcostheta = K directly.
 */
export class AngleCosine extends SimpleBonded {
  readonly name = 'cosine';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K'];

  /** angle_cosine.cpp equilibrium_angle(): MY_PI */
  equilibriumRadians(_type: number): number { return Math.PI; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (_theta, c) => {
        const ea = K[t] * (1 + c);
        e += ea;
        return { e: ea, dEdc: K[t] };
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style cosine/squared — docs.lammps.org/angle_cosine_squared.html:
 *   E = K [cos(theta) - cos(theta0)]^2
 *   "commonly used in the DREIDING force field ... Note that the usual 1/2
 *   factor is included in K." coefficients: "K (energy)", "theta0 (degrees)";
 *   "theta0 is specified in degrees, but LAMMPS converts it to radians
 *   internally." Implemented via dE/dcostheta = 2 K (c - cos theta0).
 */
export class AngleCosineSquared extends SimpleBonded {
  readonly name = 'cosine/squared';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'theta0'];

  // As native LAMMPS does (src/MOLECULE/angle_cosine_squared.cpp at b332f28): theta0[i] = DEG2RAD * theta0_one
  protected parse = parseDeg(this.paramNames, [1], 'DEG2RAD');

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    // As native LAMMPS does (src/MOLECULE/angle_cosine_squared.cpp at b332f28): write_data: RAD2DEG * theta0[i]
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((180 / Math.PI) * th[t])}`);
    return out;
  }

  // As native LAMMPS does (src/angle_write.cpp at b332f28, double theta0 = equilibrium_angle(atype) * RAD2DEG
  /** native equilibrium_angle() (radians, the stored parameter) */
  equilibriumRadians(type: number): number { return this.params.p('theta0')[type]; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), th0 = this.params.p('theta0');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta, c) => {
        const dc = c - libm.cos(th0[t]);
        const ea = K[t] * dc * dc;
        e += ea;
        return { e: ea, dEdc: 2 * K[t] * dc };
      });
    }
    bc.acc.eangle += e;
  }
}

/** Chebyshev polynomial of the second kind U_m(c) = sin((m+1) theta) / sin(theta), by recurrence. */
const chebyshevU = (m: number, c: number): number => {
  let u0 = 1, u1 = 2 * c;
  if (m === 0) return u0;
  for (let q = 1; q < m; q++) { const u2 = 2 * c * u1 - u0; u0 = u1; u1 = u2; }
  return u1;
};

/*
 * angle_style cosine/periodic — docs.lammps.org/angle_cosine_periodic.html:
 *   E = (2.0/n^2) * C [ 1 - B (-1)^n cos( n theta ) ]
 *   coefficients: "C (energy)", "B = 1 or -1", "n = 1, 2, 3, 4, 5 or 6 for
 *   periodicity". "Note that the prefactor C is specified as coefficient and
 *   not the overall force constant", K = 2 C / n^2.
 */
export class AngleCosinePeriodic extends SimpleBonded {
  readonly name = 'cosine/periodic';
  readonly kind = 'angle' as const;
  readonly paramNames = ['C', 'B', 'n'];

  protected parse(args: string[]): number[] {
    if (args.length !== 3) throw new StyleError('angle_coeff cosine/periodic needs C B n');
    return [parseNum(args[0], 'C'), parseNum(args[1], 'B'), parseInt_(args[2], 'n')];
  }

  /** angle_cosine_periodic.cpp equilibrium_angle(): MY_PI*(1.0 - ((b[i]>0) ? 0.0 : (1.0/multiplicity[i]))) */
  equilibriumRadians(type: number): number {
    const B = this.params.p('B')[type], n = this.params.p('n')[type];
    return Math.PI * (1.0 - (B > 0 ? 0.0 : 1.0 / n));
  }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const C = this.params.p('C'), B = this.params.p('B'), n = this.params.p('n');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      const nt = n[t], sign = nt % 2 === 0 ? 1 : -1;
      angleTerm(bc, i, j, k, d1, d2, (theta, c) => {
        const ea = ((2 / (nt * nt)) * C[t]) * (1 - B[t] * sign * libm.cos(nt * theta));
        e += ea;
        // dE/dc = -(2C/n) B (-1)^n U_{n-1}(c), with U_{n-1}(c) = sin(n theta)/sin(theta) (a polynomial, no 1/sin)
        return { e: ea, dEdc: -((2 * C[t]) / nt) * B[t] * sign * chebyshevU(nt - 1, c) };
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style cosine/shift — docs.lammps.org/angle_cosine_shift.html:
 *   E = -Umin/2 [ 1 + cos(theta - theta0) ]
 *   "theta0 is the equilibrium angle. The potential is bounded between
 *   -Umin and zero." coefficients: "Umin (energy)", "theta (angle)".
 * theta0 is given in degrees (verified against the native binary:
 * theta = 120 deg, theta0 = 45 deg gives E = -5(1 + cos 75 deg) per angle).
 * Measured with native LAMMPS (black box, Umin = 20, theta0 = 110 deg, one angle
 * from exactly 180 to 179.9998 deg): the energy is evaluated with the expansion
 * cos(theta - theta0) = c cos(theta0) + s sin(theta0), where s = sqrt(1 - c^2) is
 * clamped below at 0.001 (the same clamp as the harmonic angle). Exactly linear,
 * native gives -13.4295983594645 and the unclamped formula gives -13.4208278874.
 * The force uses dE/dc = -Umin/2 [cos(theta0) - (c/s) sin(theta0)] with the same
 * clamped s (measured: agrees with native to 1e-10 relative at exactly 180 degrees
 * and at near-linear geometries with s on both sides of the clamp).
 */
export class AngleCosineShift extends SimpleBonded {
  readonly name = 'cosine/shift';
  readonly kind = 'angle' as const;
  readonly paramNames = ['Umin', 'theta0'];

  // As native LAMMPS does (src/EXTRA-MOLECULE/angle_cosine_shift.cpp at b332f28): theta[i] = theta0*MY_PI / 180.0
  protected parse = parseDeg(this.paramNames, [1], 'pi-div180');

  dataCoeffs(): string[] {
    const U = this.params.p('Umin'), th = this.params.p('theta0');
    const out: string[] = [];
    // As native LAMMPS does (src/EXTRA-MOLECULE/angle_cosine_shift.cpp at b332f28): write_data: theta[i]/MY_PI*180.0
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(U[t])} ${fmtCoeff((th[t] / Math.PI) * 180)}`);
    return out;
  }

  // As native LAMMPS does (src/angle_write.cpp at b332f28, double theta0 = equilibrium_angle(atype) * RAD2DEG
  /** native equilibrium_angle() (radians, the stored parameter) */
  equilibriumRadians(type: number): number { return this.params.p('theta0')[type]; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const U = this.params.p('Umin'), th0 = this.params.p('theta0');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (_theta, c) => {
        const s = Math.max(Math.sqrt(1 - c * c), 0.001);
        const ea = -0.5 * U[t] * (1 + c * libm.cos(th0[t]) + s * libm.sin(th0[t]));
        e += ea;
        return { e: ea, dEdc: -0.5 * U[t] * (libm.cos(th0[t]) - (c / s) * libm.sin(th0[t])) };
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style cosine/delta — docs.lammps.org/angle_cosine_delta.html:
 *   E = K [1 - cos(theta - theta0)]
 *   "Note that the usual 1/2 factor is included in K." coefficients:
 *   "K (energy)", "theta0 (degrees)"; "theta0 is specified in degrees, but
 *   LAMMPS converts it to radians internally."
 */
export class AngleCosineDelta extends SimpleBonded {
  readonly name = 'cosine/delta';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'theta0'];

  // As native LAMMPS does (src/EXTRA-MOLECULE/angle_cosine_delta.cpp at b332f28, coeff inherited from src/MOLECULE/angle_cosine_squared.cpp at b332f28): theta0[i] = DEG2RAD * theta0_one
  protected parse = parseDeg(this.paramNames, [1], 'DEG2RAD');

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    // As native LAMMPS does (src/EXTRA-MOLECULE/angle_cosine_delta.cpp at b332f28, write_data inherited from src/MOLECULE/angle_cosine_squared.cpp at b332f28): RAD2DEG * theta0[i]
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((180 / Math.PI) * th[t])}`);
    return out;
  }

  // As native LAMMPS does (src/angle_write.cpp at b332f28, double theta0 = equilibrium_angle(atype) * RAD2DEG
  /** native equilibrium_angle() (radians, the stored parameter) */
  equilibriumRadians(type: number): number { return this.params.p('theta0')[type]; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), th0 = this.params.p('theta0');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta, c) => {
        const ea = K[t] * (1 - libm.cos(theta - th0[t]));
        e += ea;
        const s = Math.max(Math.sqrt(1 - c * c), 0.001);
        return { e: ea, dEdc: K[t] * (-libm.cos(th0[t]) + (c / s) * libm.sin(th0[t])) };
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style charmm — docs.lammps.org/angle_charmm.html:
 *   E = K (theta - theta0)^2 + Kub (r - rub)^2
 *   "with an additional Urey_Bradley term based on the distance r between the
 *   first and third atoms in the angle." coefficients (in this order):
 *   "K (energy)", "theta0 (degrees)", "Kub (energy/distance^2)",
 *   "rub (distance)"; "theta0 is specified in degrees, but LAMMPS converts it
 *   to radians internally; hence K is effectively energy per radian^2."
 */
export class AngleCharmm extends SimpleBonded {
  readonly name = 'charmm';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'theta0', 'Kub', 'rub'];

  // As native LAMMPS does (src/MOLECULE/angle_charmm.cpp at b332f28): theta0[i] = DEG2RAD * theta0_one
  protected parse = parseDeg(this.paramNames, [1], 'DEG2RAD');

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const Kub = this.params.p('Kub'), rub = this.params.p('rub');
    const out: string[] = [];
    // As native LAMMPS does (src/MOLECULE/angle_charmm.cpp at b332f28): write_data: RAD2DEG * theta0[i]
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((180 / Math.PI) * th[t])} ${fmtCoeff(Kub[t])} ${fmtCoeff(rub[t])}`);
    return out;
  }

  // As native LAMMPS does (src/angle_write.cpp at b332f28, double theta0 = equilibrium_angle(atype) * RAD2DEG
  /** native equilibrium_angle() (radians, the stored parameter) */
  equilibriumRadians(type: number): number { return this.params.p('theta0')[type]; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), th0 = this.params.p('theta0');
    const Kub = this.params.p('Kub'), rub = this.params.p('rub');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0], dk = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      // the Urey-Bradley term on the i-k distance (atoms 1 and 3) joins the angle's f1, f3 and energy, as native
      // angle_charmm.cpp does before its one Angle::ev_tally: the per-atom energy and virial go a third to each atom
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const dth = theta - th0[t];
        const ea = K[t] * dth * dth;
        e += ea;
        return { e: ea, dEdtheta: 2 * K[t] * dth };
      }, Kub[t] === 0 ? undefined : (f1, f3) => {
        termDelta(bc, j, k, i, dk); // r_i - r_k, both copies nearest to the centre j
        const r = Math.sqrt(dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]);
        const dr = r - rub[t];
        const eub = Kub[t] * dr * dr;
        e += eub;
        const fmag = r > 0 ? (-2 * Kub[t] * dr) / r : 0;
        for (let c = 0; c < 3; c++) { f1[c] += dk[c] * fmag; f3[c] -= dk[c] * fmag; }
        return eub;
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style quartic — docs.lammps.org/angle_quartic.html:
 *   E = K2 (theta - theta0)^2 + K3 (theta - theta0)^3 + K4 (theta - theta0)^4
 *   coefficients (in this order): "theta0 (degrees)", "K2 (energy)",
 *   "K3 (energy)", "K4 (energy)"; "theta0 is specified in degrees, but LAMMPS
 *   converts it to radians internally; hence the various K are effectively
 *   energy per radian^2 or radian^3 or radian^4."
 */
export class AngleQuartic extends SimpleBonded {
  readonly name = 'quartic';
  readonly kind = 'angle' as const;
  readonly paramNames = ['theta0', 'K2', 'K3', 'K4'];

  // As native LAMMPS does (src/EXTRA-MOLECULE/angle_quartic.cpp at b332f28): theta0[i] = theta0_one/180.0 * MY_PI
  protected parse = parseDeg(this.paramNames, [0], 'div180-pi');

  dataCoeffs(): string[] {
    const th = this.params.p('theta0'), K2 = this.params.p('K2'), K3 = this.params.p('K3'), K4 = this.params.p('K4');
    const out: string[] = [];
    // As native LAMMPS does (src/EXTRA-MOLECULE/angle_quartic.cpp at b332f28): write_data: theta0[i]/MY_PI*180.0
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff((th[t] / Math.PI) * 180)} ${fmtCoeff(K2[t])} ${fmtCoeff(K3[t])} ${fmtCoeff(K4[t])}`);
    return out;
  }

  // As native LAMMPS does (src/angle_write.cpp at b332f28, double theta0 = equilibrium_angle(atype) * RAD2DEG
  /** native equilibrium_angle() (radians, the stored parameter) */
  equilibriumRadians(type: number): number { return this.params.p('theta0')[type]; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const th0 = this.params.p('theta0'), K2 = this.params.p('K2'), K3 = this.params.p('K3'), K4 = this.params.p('K4');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const dth = theta - th0[t], dth2 = dth * dth;
        const ea = K2[t] * dth2 + K3[t] * dth2 * dth + K4[t] * dth2 * dth2;
        e += ea;
        return { e: ea, dEdtheta: 2 * K2[t] * dth + 3 * K3[t] * dth2 + 4 * K4[t] * dth2 * dth };
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style fourier — docs.lammps.org/angle_fourier.html:
 *   E = K [ C0 + C1 cos(theta) + C2 cos(2 theta) ]
 *   coefficients: "K (energy)", "C0 (real)", "C1 (real)", "C2 (real)".
 */
export class AngleFourier extends SimpleBonded {
  readonly name = 'fourier';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'C0', 'C1', 'C2'];

  /** angle_fourier.cpp equilibrium_angle(): acos(-C1/4/C2) when |C1/4/C2| <= 1, else C1/4/C2; MY_PI when C2 is 0. */
  equilibriumRadians(type: number): number {
    const C1 = this.params.p('C1')[type], C2 = this.params.p('C2')[type];
    let ret = Math.PI;
    if (C2 !== 0.0) {
      ret = C1 / 4.0 / C2;
      if (Math.abs(ret) <= 1.0) ret = libm.acos(-ret);
    }
    return ret;
  }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), C0 = this.params.p('C0'), C1 = this.params.p('C1'), C2 = this.params.p('C2');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta, c) => {
        const ea = K[t] * (C0[t] + C1[t] * c + C2[t] * libm.cos(2 * theta));
        e += ea;
        // cos(2 theta) = 2c^2 - 1 is a polynomial in c, so dE/dc is exact
        return { e: ea, dEdc: K[t] * (C1[t] + 4 * C2[t] * c) };
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style fourier/simple — docs.lammps.org/angle_fourier_simple.html:
 *   E = K [ 1.0 + c cos( n theta ) ]
 *   coefficients: "K (energy)", "c (real)", "n (real)".
 * The force is the exact gradient of this energy. Measured with native
 * LAMMPS (2 Sep 2026): its forces equal the gradient for n = 1 and 2, but not
 * for n >= 3 (for n = 3 at theta = 66.4 degrees they are 5.0 times larger),
 * and a three-atom NVE run with n = 3 does not conserve energy (etotal 0.85 ->
 * 3.5) while n = 2 does (to 1e-5). The engine keeps the documented energy
 * and its gradient; oracle case w1ba_angle_fourier_simple uses n = 1 and 2.
 */
export class AngleFourierSimple extends SimpleBonded {
  readonly name = 'fourier/simple';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'c', 'n'];

  /** angle_fourier_simple.cpp equilibrium_angle(): MY_PI / N[i] */
  equilibriumRadians(type: number): number { return Math.PI / this.params.p('n')[type]; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), c = this.params.p('c'), n = this.params.p('n');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const nt = n[t];
        const ea = K[t] * (1 + c[t] * libm.cos(nt * theta));
        e += ea;
        return { e: ea, dEdtheta: -K[t] * c[t] * nt * libm.sin(nt * theta) };
      });
    }
    bc.acc.eangle += e;
  }
}

/*
 * angle_style zero — docs.lammps.org/angle_zero.html:
 *   "Using an angle style of zero means angle forces and energies are not
 *   computed, but the geometry of angle triplets is still accessible to other
 *   commands." "The optional nocoeff flag allows to read data files with
 *   AngleCoeff section for any angle style.  Similarly, any angle_coeff
 *   commands will only be checked for the angle type number and the rest
 *   ignored." "Note that the angle_coeff command must be used for all angle
 *   types.  If specified, there can be only one value, which is going to be
 *   used to assign an equilibrium angle, e.g. for use with fix shake."
 */
export class AngleZero extends Bonded {
  readonly name = 'zero';
  readonly kind = 'angle' as const;
  private nocoeff = false;
  private eq = new Float64Array(0);
  private set = new Uint8Array(0);
  ntypes = 0;

  settings(args: string[], _ctx?: StyleContext): void {
    for (const a of args) if (a !== 'nocoeff') throw new StyleError(`angle_style zero: unknown keyword '${a}'`);
    this.nocoeff = args.length > 0;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.eq = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    const [lo, hi] = typeBounds(args[0] ?? '', this.ntypes);
    // "If specified, there can be only one value" (angle_zero.html); with nocoeff the values are not checked and
    // not read, as native angle_zero.cpp does ("if ((narg < 1) || (coeffflag && narg > 2))"), so read_data can pass
    // another style's coefficients
    if (args.length > 2 && !this.nocoeff) throw new StyleError('angle_coeff zero: incorrect args (only one value allowed)');
    if (args.length === 2 && !this.nocoeff) {
      // theta0 in degrees -> radians, DEG2RAD * theta0_one (angle_zero.cpp coeff)
      const v = (Math.PI / 180) * parseNum(args[1], 'theta0');
      for (let t = lo; t <= hi; t++) this.eq[t] = v;
    }
    for (let t = lo; t <= hi; t++) this.set[t] = 1;
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`angle_coeff must be used for all angle types with angle_style zero (type ${t} not set)`);
    }
  }

  compute(_bc: BondedCompute): void {}

  dataCoeffs(): string[] | null {
    if (this.nocoeff) return null;
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff((180 / Math.PI) * this.eq[t])}`);
    return out;
  }

  /** angle_zero.cpp equilibrium_angle(): theta0 (radians) */
  equilibriumRadians(type: number): number { return this.eq[type]; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }
}
