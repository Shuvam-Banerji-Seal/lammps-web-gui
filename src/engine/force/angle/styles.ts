import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import { angleTerm } from './harmonic';
import { fmtCoeff, parseInt_, parseNum } from '../util';
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';

/*
 * Angle styles added by wave 1. Each class cites its docs.lammps.org page and
 * quotes the formula / coefficient sentences it implements. theta is the
 * angle at the middle atom j of the triplet i-j-k, computed by angleTerm.
 */

/** Parses angle_coeff values, converting the named angle coefficients from degrees to radians. */
const parseDeg = (names: readonly string[], degIdx: readonly number[]) => (args: string[]): number[] => {
  if (args.length !== names.length) throw new StyleError(`angle_coeff: expected ${names.length} coefficients: ${names.join(' ')}`);
  return args.map((w, k) => {
    const v = parseNum(w, names[k]);
    return degIdx.includes(k) ? (v * Math.PI) / 180 : v;
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

  protected parse = parseDeg(this.paramNames, [1]);

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((th[t] * 180) / Math.PI)}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

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
        const dc = c - Math.cos(th0[t]);
        const ea = K[t] * dc * dc;
        e += ea;
        return { e: ea, dEdc: 2 * K[t] * dc };
      });
    }
    bc.acc.eangle += e;
  }
}

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
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const ea = ((2 / (nt * nt)) * C[t]) * (1 - B[t] * sign * Math.cos(nt * theta));
        e += ea;
        // dE/dtheta = (2C/n) B (-1)^n sin(n theta)
        return { e: ea, dEdtheta: ((2 * C[t]) / nt) * B[t] * sign * Math.sin(nt * theta) };
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
 */
export class AngleCosineShift extends SimpleBonded {
  readonly name = 'cosine/shift';
  readonly kind = 'angle' as const;
  readonly paramNames = ['Umin', 'theta0'];

  protected parse = parseDeg(this.paramNames, [1]);

  dataCoeffs(): string[] {
    const U = this.params.p('Umin'), th = this.params.p('theta0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(U[t])} ${fmtCoeff((th[t] * 180) / Math.PI)}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

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
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const ea = -0.5 * U[t] * (1 + Math.cos(theta - th0[t]));
        e += ea;
        return { e: ea, dEdtheta: 0.5 * U[t] * Math.sin(theta - th0[t]) };
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

  protected parse = parseDeg(this.paramNames, [1]);

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((th[t] * 180) / Math.PI)}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

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
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const ea = K[t] * (1 - Math.cos(theta - th0[t]));
        e += ea;
        return { e: ea, dEdtheta: K[t] * Math.sin(theta - th0[t]) };
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

  protected parse = parseDeg(this.paramNames, [1]);

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const Kub = this.params.p('Kub'), rub = this.params.p('rub');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((th[t] * 180) / Math.PI)} ${fmtCoeff(Kub[t])} ${fmtCoeff(rub[t])}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

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
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const dth = theta - th0[t];
        const ea = K[t] * dth * dth;
        e += ea;
        return { e: ea, dEdtheta: 2 * K[t] * dth };
      });
      // Urey-Bradley term on the i-k distance (atoms 1 and 3 of the angle)
      if (Kub[t] !== 0) {
        delta(bc, k, i, dk); // r_i - r_k
        const r = Math.sqrt(dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]);
        const dr = r - rub[t];
        const eub = Kub[t] * dr * dr;
        e += eub;
        const fmag = r > 0 ? (-2 * Kub[t] * dr) / r : 0;
        const f = bc.f;
        f[3 * i] += dk[0] * fmag; f[3 * i + 1] += dk[1] * fmag; f[3 * i + 2] += dk[2] * fmag;
        f[3 * k] -= dk[0] * fmag; f[3 * k + 1] -= dk[1] * fmag; f[3 * k + 2] -= dk[2] * fmag;
        const w0 = dk[0] * dk[0] * fmag, w1 = dk[1] * dk[1] * fmag, w2 = dk[2] * dk[2] * fmag;
        const w3 = dk[0] * dk[1] * fmag, w4 = dk[0] * dk[2] * fmag, w5 = dk[1] * dk[2] * fmag;
        const v = bc.virial;
        v[0] += w0; v[1] += w1; v[2] += w2; v[3] += w3; v[4] += w4; v[5] += w5;
        if (bc.eatom) { bc.eatom[i] += 0.5 * eub; bc.eatom[k] += 0.5 * eub; }
        if (bc.vatom) {
          const w = [w0, w1, w2, w3, w4, w5];
          for (let c = 0; c < 6; c++) { bc.vatom[6 * i + c] += 0.5 * w[c]; bc.vatom[6 * k + c] += 0.5 * w[c]; }
        }
      }
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

  protected parse = parseDeg(this.paramNames, [0]);

  dataCoeffs(): string[] {
    const th = this.params.p('theta0'), K2 = this.params.p('K2'), K3 = this.params.p('K3'), K4 = this.params.p('K4');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff((th[t] * 180) / Math.PI)} ${fmtCoeff(K2[t])} ${fmtCoeff(K3[t])} ${fmtCoeff(K4[t])}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

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
        const ea = K[t] * (C0[t] + C1[t] * c + C2[t] * Math.cos(2 * theta));
        e += ea;
        return { e: ea, dEdtheta: -K[t] * (C1[t] + 4 * C2[t] * c) * Math.sin(theta) };
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
        const ea = K[t] * (1 + c[t] * Math.cos(nt * theta));
        e += ea;
        return { e: ea, dEdtheta: -K[t] * c[t] * nt * Math.sin(nt * theta) };
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
    // "If specified, there can be only one value" (angle_zero.html)
    if (args.length > 2) throw new StyleError('angle_coeff zero: incorrect args (only one value allowed)');
    if (args.length === 2) {
      const v = parseNum(args[1], 'theta0');
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
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(this.eq[t])}`);
    return out;
  }

  equilibrium(type: number): number { return this.eq[type]; }
}
