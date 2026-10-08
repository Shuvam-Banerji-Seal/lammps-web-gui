import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';
import { bondLoop } from './styles';
import { fmtCoeff, parseInt_, parseNum } from '../util';

/*
 * bond_style gaussian — docs.lammps.org/bond_gaussian.html:
 *   E = -k_B T ln( sum_{i=1}^{n} A_i/(w_i sqrt(pi/2))
 *                  exp( -2 (r - r_i)^2 / w_i^2 ) )
 *   coefficients, in this order: "T temperature at which the potential was
 *   derived", "n (integer >=1)", then for each i "A_i (> 0, distance)",
 *   "w_i (> 0, distance)", "r_i (>= 0, distance)".
 *   "Note that the usual ..." does not apply here; k_B is the unit system's
 *   Boltzmann constant (boltz), so T is a temperature (measured with native
 *   LAMMPS (black box): units real, T = 300 gives the full boltz * 300 factor,
 *   units lj the full T factor once thermo normalisation is disabled).
 * The force is the exact derivative of this energy:
 *   dE/dr = -k_B T/S * dS/dr,  dS/dr = -4 sum_i g_i (r - r_i) / w_i^2.
 */

const SQRT_PI_2 = Math.sqrt(Math.PI / 2);

interface GaussTerm {
  a: number;
  w: number;
  r0: number;
}

export class BondGaussian extends Bonded {
  readonly name = 'gaussian';
  readonly kind = 'bond' as const;
  private T = new Float64Array(0);
  private set = new Uint8Array(0);
  private terms: GaussTerm[][] = [];

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`bond_style gaussian takes no arguments (got '${args[0]}')`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.T = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
    this.terms = Array.from({ length: ntypes + 1 }, () => [] as GaussTerm[]);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 3) throw new StyleError('usage: bond_coeff N T n A1 w1 r1 ...');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const T = parseNum(args[1], 'T');
    const n = parseInt_(args[2], 'n');
    if (n < 1) throw new StyleError(`bond_coeff gaussian: n must be an integer >= 1, got ${n}`);
    const need = 3 + 3 * n;
    if (args.length !== need) {
      throw new StyleError(`bond_coeff gaussian: expected T n and ${n} (A w r) term(s) = ${need - 1} values after the type, got ${args.length - 1}`);
    }
    const terms: GaussTerm[] = [];
    for (let i = 0; i < n; i++) {
      terms.push({
        a: parseNum(args[3 + 3 * i], `A${i + 1}`),
        w: parseNum(args[4 + 3 * i], `w${i + 1}`),
        r0: parseNum(args[5 + 3 * i], `r${i + 1}`),
      });
    }
    for (let t = lo; t <= hi; t++) {
      this.T[t] = T;
      this.terms[t] = terms.map((x) => ({ ...x }));
      this.set[t] = 1;
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) if (!this.set[t]) throw new StyleError(`all bond coeffs are not set (type ${t})`);
  }

  compute(bc: BondedCompute): void {
    const kb = bc.s.units.boltz;
    bondLoop(bc, (r, t) => {
      const kT = kb * this.T[t];
      const terms = this.terms[t];
      let S = 0;
      for (const g of terms) {
        const d = r - g.r0;
        S += (g.a / (g.w * SQRT_PI_2)) * Math.exp((-2 * d * d) / (g.w * g.w));
      }
      let dS = 0;
      for (const g of terms) {
        const d = r - g.r0;
        dS += (g.a / (g.w * SQRT_PI_2)) * Math.exp((-2 * d * d) / (g.w * g.w)) * ((-4 * d) / (g.w * g.w));
      }
      return { e: -kT * Math.log(S), dEdr: (-kT * dS) / S };
    });
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      const w = [String(t), fmtCoeff(this.T[t]), String(this.terms[t].length)];
      for (const g of this.terms[t]) w.push(fmtCoeff(g.a), fmtCoeff(g.w), fmtCoeff(g.r0));
      out.push(w.join(' '));
    }
    return out;
  }
}
