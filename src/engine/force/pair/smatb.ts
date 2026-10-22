import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';
import { libm } from '../../libm';

/*
 * pair_style smatb and smatb/single, docs.lammps.org/pair_smatb.html (text: plans/lammps-docs-live/pair_smatb.rst).
 * Second Moment Approximation to the Tight Binding (Cyrot, Gupta, Rosato references on the doc page).
 *
 * Energy (the doc formula): E_i = sum_j alpha(R_ij) - sqrt( sum_j Xi^2(R_ij) ), summed over j with R_ij <= R_c, with
 *   alpha(r) = A exp(-p (r/R0 - 1))                              r < R_sc
 *   alpha(r) = a3 (r-Rc)^3 + a4 (r-Rc)^4 + a5 (r-Rc)^5          R_sc < r < R_c
 *   Xi(r)    = xi exp(-q (r/R0 - 1))                              r < R_sc
 *   Xi(r)    = x3 (r-Rc)^3 + x4 (r-Rc)^4 + x5 (r-Rc)^5          R_sc < r < R_c
 *
 * The doc says that the polynomial coefficients "are computed by LAMMPS: the two exponential terms and
 * their first and second derivatives are smoothly" reduced to zero between the inner and the outer cutoff.
 * The engine takes this as three conditions at R_sc (value, first and second derivative of the polynomial
 * equal those of the exponential), the polynomial vanishing to second order at R_c.
 *
 * Native agreement (examples/PACKAGES/smtbq logs, which the native binary used here cannot run because it
 * lacks the SMTBQ package): the perfect fcc Au crystal of in.smatbBulkFCC (2048 atoms, the 2nd neighbours
 * lie in the polynomial zone) has step-0 E_pair -7800.9629 in its log; this construction gives -7800.96285.
 * The AgCu pancake (AgCuPancake.data, mixed types, periodic) has step-0 E_pair -90.165962; this gives
 * -90.16596205. The virial pressure of a homogeneous scaling agrees with -dE/dV (unit test).
 *
 * Coefficients: the doc example "pair_coeff 1 1 2.88 10.35 4.178 0.210 1.818 4.07293506 4.9883063257983666"
 * gives the order R0, p, q, A, xi, R_cs, R_c.
 *
 * Mixing: the doc says "For atom type pairs I,J and I != J the coefficients are not automatically mixed."
 * An unset I,J pair is a StyleError.
 *
 * smatb/single: the doc says "The *smatb/single* style is an optimization when using only a single atom type."
 * The engine computes both styles with the same code and claims nothing for several types with smatb/single.
 *
 * Summation: E = sum over owned i of E_i. Each E_i uses the full neighbour list; the force on j from E_i is
 * -g d / r and on i +g d / r, with d = x_j - x_i and g = alpha' - Xi Xi' / sqrt(sum Xi^2). Per-atom energy
 * goes to atom i; the per-atom virial of a pair term is shared half to each atom; the global virial is the
 * x.f dot product.
 */

/** Per type pair parameters and the polynomial coefficients (value, derivative and curvature matched at R_sc). */
interface Pot {
  R0: number; p: number; q: number; A: number; xi: number; Rsc: number; Rc: number;
  a: [number, number, number];
  x: [number, number, number];
}

/** Solves the 3x3 system M c = b by Gaussian elimination with partial pivoting. */
const solve3 = (M: number[][], b: number[]): [number, number, number] => {
  const m = M.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < 3; c++) {
    let piv = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(m[r][c]) > Math.abs(m[piv][c])) piv = r;
    [m[c], m[piv]] = [m[piv], m[c]];
    for (let r = c + 1; r < 3; r++) {
      const f = m[r][c] / m[c][c];
      for (let k = c; k < 4; k++) m[r][k] -= f * m[c][k];
    }
  }
  const out = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let s = m[r][3];
    for (let k = r + 1; k < 3; k++) s -= m[r][k] * out[k];
    out[r] = s / m[r][r];
  }
  return [out[0], out[1], out[2]];
};

/** Polynomial c3 d^3 + c4 d^4 + c5 d^5 (d = r - Rc) matching value f0, slope f1 and curvature f2 at d0 = Rsc - Rc. */
const polyFor = (d0: number, f0: number, f1: number, f2: number): [number, number, number] => {
  const M = [
    [d0 ** 3, d0 ** 4, d0 ** 5],
    [3 * d0 ** 2, 4 * d0 ** 3, 5 * d0 ** 4],
    [6 * d0, 12 * d0 ** 2, 20 * d0 ** 3],
  ];
  return solve3(M, [f0, f1, f2]);
};

const makePot = (R0: number, p: number, q: number, A: number, xi: number, Rsc: number, Rc: number): Pot => {
  const d0 = Rsc - Rc;
  // exponential branch at R_sc: f = C exp(-k (r/R0 - 1)), f' = -(k/R0) f, f'' = (k/R0)^2 f
  const ea = A * libm.exp(-p * (Rsc / R0 - 1));
  const ex = xi * libm.exp(-q * (Rsc / R0 - 1));
  const a = polyFor(d0, ea, (-p / R0) * ea, ((p / R0) ** 2) * ea);
  const x = polyFor(d0, ex, (-q / R0) * ex, ((q / R0) ** 2) * ex);
  return { R0, p, q, A, xi, Rsc, Rc, a, x };
};

/** alpha(r) and Xi(r) with their radial derivatives (r < Rc). */
const evalPot = (P: Pot, r: number): { al: number; dal: number; xi: number; dxi: number } => {
  if (r < P.Rsc) {
    const ea = P.A * libm.exp(-P.p * (r / P.R0 - 1));
    const ex = P.xi * libm.exp(-P.q * (r / P.R0 - 1));
    return { al: ea, dal: (-P.p / P.R0) * ea, xi: ex, dxi: (-P.q / P.R0) * ex };
  }
  const d = r - P.Rc, d2 = d * d;
  const [a3, a4, a5] = P.a, [x3, x4, x5] = P.x;
  return {
    al: a3 * d2 * d + a4 * d2 * d2 + a5 * d2 * d2 * d,
    dal: 3 * a3 * d2 + 4 * a4 * d2 * d + 5 * a5 * d2 * d2,
    xi: x3 * d2 * d + x4 * d2 * d2 + x5 * d2 * d2 * d,
    dxi: 3 * x3 * d2 + 4 * x4 * d2 * d + 5 * x5 * d2 * d2,
  };
};

export class PairSMATB extends Pair {
  readonly name: string = 'smatb';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;
  p!: PairParams;
  private pots: (Pot | null)[] = [];

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError(`usage: pair_style ${this.name} (no arguments)`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['R0', 'p', 'q', 'A', 'xi', 'Rsc', 'Rc']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length !== 9) throw pairCoeffArgs(`usage: pair_coeff I J R0 p q A xi Rcs Rc (style ${this.name})`);
    const v = args.slice(2).map((w, k) => parseNum(w, ['R0', 'p', 'q', 'A', 'xi', 'Rcs', 'Rc'][k]));
    if (!(v[0] > 0)) throw new StyleError('pair_coeff smatb: R0 must be > 0');
    if (!(v[5] > 0 && v[6] > v[5])) throw new StyleError('pair_coeff smatb: the cutoffs must satisfy 0 < Rcs < Rc');
    this.p.setRange(args[0], args[1], v);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j}; smatb does not mix)`);
    const nt = this.ntypes + 1;
    if (this.pots.length !== nt * nt) this.pots = new Array(nt * nt).fill(null);
    const g = (n: string) => this.p.get(n, i, j);
    const pot = makePot(g('R0'), g('p'), g('q'), g('A'), g('xi'), g('Rsc'), g('Rc'));
    this.pots[i * nt + j] = this.pots[j * nt + i] = pot;
    return pot.Rc;
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const pots = this.pots;
    const nlocal = pc.nlocal;
    const eatom = pc.eatom, vatom = pc.vatom;
    const nb = list.neighbors;
    let evdwl = 0;
    // pass 1: sums over the neighbours of each owned atom
    const sumA = new Float64Array(nlocal), sumX = new Float64Array(nlocal);
    for (let i = 0; i < nlocal; i++) {
      const ti = type[i] * nt;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const e = evalPot(pots[t]!, Math.sqrt(rsq));
        sumA[i] += e.al;
        sumX[i] += e.xi * e.xi;
      }
    }
    // pass 2: energy E_i = sum alpha - sqrt(sum Xi^2) and the pair forces of E_i
    for (let i = 0; i < nlocal; i++) {
      const ti = type[i] * nt;
      const X = sumX[i];
      const sq = Math.sqrt(X);
      const ei = sumA[i] - sq;
      evdwl += ei;
      if (eatom) eatom[i] += ei;
      if (sq === 0) continue;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        const e = evalPot(pots[t]!, r);
        // dE_i/d(x_j - x_i) = g d / r with g = alpha' - Xi Xi' / sqrt(sum Xi^2)
        const gr = (e.dal - (e.xi * e.dxi) / sq) / r;
        const hx = gr * dx, hy = gr * dy, hz = gr * dz;
        fxi += hx; fyi += hy; fzi += hz;
        f[3 * j] -= hx; f[3 * j + 1] -= hy; f[3 * j + 2] -= hz;
        if (vatom) {
          // the pair virial -d (x) g d / r is shared half to each atom
          const w = [dx * dx, dy * dy, dz * dz, dx * dy, dx * dz, dy * dz];
          for (let c = 0; c < 6; c++) {
            const h = -0.5 * gr * w[c];
            vatom[6 * i + c] += h;
            vatom[6 * j + c] += h;
          }
        }
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }
}

/** pair_style smatb/single: the same computation (the doc calls it an optimization for one atom type). */
export class PairSMATBSingle extends PairSMATB {
  override readonly name: string = 'smatb/single';
}
