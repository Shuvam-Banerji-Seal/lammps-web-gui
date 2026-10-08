import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';
import { atomIndex } from '../bonded_util';
import { angleTerm } from './harmonic';
import { fmtCoeff, parseInt_, parseNum } from '../util';

/*
 * angle_style gaussian — docs.lammps.org/angle_gaussian.html:
 *   E = -k_B T ln( sum_{i=1}^{n} A_i/(w_i sqrt(pi/2))
 *                  exp( -2 (theta - theta_i)^2 / w_i^2 ) )
 *   coefficients, in this order: "T temperature at which the potential was
 *   derived", "n (integer >=1)", then for each i "A_i (> 0, radians)",
 *   "w_i (> 0, radians)", "theta_i (degrees)". theta_i is converted to radians
 *   internally (measured with native LAMMPS (black box): T = 1, n = 1,
 *   A = 1, w = 0.3 rad, theta_i = 100 deg at theta = 100 deg gives
 *   -ln(1/(0.3 sqrt(pi/2))) = -0.978181451681209).
 * The force is the exact derivative of this energy:
 *   dE/dtheta = -k_B T/S * dS/dtheta,
 *   dS/dtheta = -4 sum_i g_i (theta - theta_i) / w_i^2.
 */

const SQRT_PI_2 = Math.sqrt(Math.PI / 2);

interface GaussTerm {
  a: number;
  w: number;
  /** Center angle in radians. */
  th: number;
}

export class AngleGaussian extends Bonded {
  readonly name = 'gaussian';
  readonly kind = 'angle' as const;
  private T = new Float64Array(0);
  private set = new Uint8Array(0);
  private terms: GaussTerm[][] = [];

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`angle_style gaussian takes no arguments (got '${args[0]}')`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.T = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
    this.terms = Array.from({ length: ntypes + 1 }, () => [] as GaussTerm[]);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 3) throw new StyleError('usage: angle_coeff N T n A1 w1 theta1 ...');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const T = parseNum(args[1], 'T');
    const n = parseInt_(args[2], 'n');
    if (n < 1) throw new StyleError(`angle_coeff gaussian: n must be an integer >= 1, got ${n}`);
    const need = 3 + 3 * n;
    if (args.length !== need) {
      throw new StyleError(`angle_coeff gaussian: expected T n and ${n} (A w theta) term(s) = ${need - 1} values after the type, got ${args.length - 1}`);
    }
    const terms: GaussTerm[] = [];
    for (let i = 0; i < n; i++) {
      terms.push({
        a: parseNum(args[3 + 3 * i], `A${i + 1}`),
        w: parseNum(args[4 + 3 * i], `w${i + 1}`),
        th: (parseNum(args[5 + 3 * i], `theta${i + 1}`) * Math.PI) / 180,
      });
    }
    for (let t = lo; t <= hi; t++) {
      this.T[t] = T;
      this.terms[t] = terms.map((x) => ({ ...x }));
      this.set[t] = 1;
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) if (!this.set[t]) throw new StyleError(`all angle coeffs are not set (type ${t})`);
  }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const kb = bc.s.units.boltz;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const kT = kb * this.T[t];
        const terms = this.terms[t];
        let S = 0;
        for (const g of terms) {
          const d = theta - g.th;
          S += (g.a / (g.w * SQRT_PI_2)) * Math.exp((-2 * d * d) / (g.w * g.w));
        }
        let dS = 0;
        for (const g of terms) {
          const d = theta - g.th;
          dS += (g.a / (g.w * SQRT_PI_2)) * Math.exp((-2 * d * d) / (g.w * g.w)) * ((-4 * d) / (g.w * g.w));
        }
        const eb = -kT * Math.log(S);
        e += eb;
        return { e: eb, dEdtheta: (-kT * dS) / S };
      });
    }
    bc.acc.eangle += e;
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      const w = [String(t), fmtCoeff(this.T[t]), String(this.terms[t].length)];
      for (const g of this.terms[t]) w.push(fmtCoeff(g.a), fmtCoeff(g.w), fmtCoeff((g.th * 180) / Math.PI));
      out.push(w.join(' '));
    }
    return out;
  }
}
