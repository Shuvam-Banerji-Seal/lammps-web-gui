import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';
import { fmtCoeff, parseNum } from '../util';

/*
 * Bond styles added by wave 1. Each class cites its docs.lammps.org page and
 * quotes the formula / coefficient sentences it implements.
 */

export interface BondTerm {
  e: number;
  dEdr: number;
}

/**
 * Runs the bond list once, asking `pot` for the energy and dE/dr of each bond
 * (t = bond type, r = length). Adds forces F_i = -(dE/dr) d/r with
 * d = r_i - r_j, the virial d (x) F_i, per-atom energy halves and the total
 * to bc.acc.ebond (same bookkeeping as bond_style harmonic).
 */
export const bondLoop = (
  bc: BondedCompute,
  pot: (r: number, t: number, k: number) => BondTerm,
): void => {
  const b = bc.s.topo.bonds;
  const d = [0, 0, 0];
  const f = bc.f;
  const v = bc.virial;
  let e = 0;
  for (let k = 0; k < b.n; k++) {
    const i = atomIndex(bc, b.atoms[2 * k], 'bond');
    const j = atomIndex(bc, b.atoms[2 * k + 1], 'bond');
    const t = b.type[k];
    delta(bc, j, i, d); // r_i - r_j
    const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const term = r > 0 ? pot(r, t, k) : { e: 0, dEdr: 0 };
    const fmag = r > 0 ? -term.dEdr / r : 0;
    e += term.e;
    f[3 * i] += d[0] * fmag; f[3 * i + 1] += d[1] * fmag; f[3 * i + 2] += d[2] * fmag;
    f[3 * j] -= d[0] * fmag; f[3 * j + 1] -= d[1] * fmag; f[3 * j + 2] -= d[2] * fmag;
    const w0 = d[0] * d[0] * fmag, w1 = d[1] * d[1] * fmag, w2 = d[2] * d[2] * fmag;
    const w3 = d[0] * d[1] * fmag, w4 = d[0] * d[2] * fmag, w5 = d[1] * d[2] * fmag;
    v[0] += w0; v[1] += w1; v[2] += w2; v[3] += w3; v[4] += w4; v[5] += w5;
    if (bc.eatom) { bc.eatom[i] += 0.5 * term.e; bc.eatom[j] += 0.5 * term.e; }
    if (bc.vatom) {
      const w = [w0, w1, w2, w3, w4, w5];
      for (let c = 0; c < 6; c++) { bc.vatom[6 * i + c] += 0.5 * w[c]; bc.vatom[6 * j + c] += 0.5 * w[c]; }
    }
  }
  bc.acc.ebond += e;
};

/**
 * The FENE logarithm's argument 1 - (r/R0)^2 for bond k, as native LAMMPS
 * guards it (measured 2026-10-07, K 30 R0 1.5): below 0.1 it prints
 * "WARNING: FENE bond too long: <step> <id1> <id2> <r>" and uses 0.1 in both
 * the energy and the force; at -3 or below it stops with "Bad FENE bond".
 */
const feneArg = (bc: BondedCompute, arg: number, k: number, r: number): number => {
  if (arg >= 0.1) return arg;
  const b = bc.s.topo.bonds;
  bc.warn?.(`FENE bond too long: ${bc.s.step} ${b.atoms[2 * k]} ${b.atoms[2 * k + 1]} ${r}`);
  if (arg <= -3) throw new StyleError('Bad FENE bond');
  return 0.1;
};

/*
 * bond_style fene — docs.lammps.org/bond_fene.html:
 *   E = -0.5 K R0^2 ln[1 - (r/R0)^2] + 4 eps [ (sigma/r)^12 - (sigma/r)^6 ] + eps
 *   "The first term extends to R0, the maximum extent of the bond.  The second
 *   term is cutoff at 2^(1/6) sigma, the minimum of the LJ potential."
 *   coefficients: "K (energy/distance^2)", "R0 (distance)", "epsilon (energy)",
 *   "sigma (distance)".
 */
export class BondFene extends SimpleBonded {
  readonly name = 'fene';
  readonly kind = 'bond' as const;
  readonly paramNames = ['K', 'R0', 'epsilon', 'sigma'];

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), R0 = this.params.p('R0');
    const eps = this.params.p('epsilon'), sig = this.params.p('sigma');
    const rmin = Math.pow(2, 1 / 6);
    bondLoop(bc, (r, t, k) => {
      const arg = feneArg(bc, 1 - (r * r) / (R0[t] * R0[t]), k, r);
      const e = -0.5 * K[t] * R0[t] * R0[t] * Math.log(arg);
      const dEdr = (K[t] * r) / arg;
      if (r < rmin * sig[t]) {
        const s = sig[t] / r, s6 = s ** 6, s12 = s6 * s6;
        return { e: e + 4 * eps[t] * (s12 - s6) + eps[t], dEdr: dEdr + 4 * eps[t] * (-12 * s12 / r + 6 * s6 / r) };
      }
      return { e, dEdr };
    });
  }

  equilibrium(type: number): number { return this.params.p('R0')[type]; }
}

/*
 * bond_style fene/expand — docs.lammps.org/bond_fene_expand.html:
 *   E = -0.5 K R0^2 ln[1 - ((r-Delta)/R0)^2]
 *       + 4 eps [ (sigma/(r-Delta))^12 - (sigma/(r-Delta))^6 ] + eps
 *   "an extra shift factor of Delta (positive or negative) is added to r ...
 *   The first term now extends to R0 + Delta and the second term is cutoff at
 *   2^(1/6) sigma + Delta."
 *   coefficients: "K (energy/distance^2)", "R0 (distance)", "epsilon (energy)",
 *   "sigma (distance)", "Delta (distance)".
 */
export class BondFeneExpand extends SimpleBonded {
  readonly name = 'fene/expand';
  readonly kind = 'bond' as const;
  readonly paramNames = ['K', 'R0', 'epsilon', 'sigma', 'delta'];

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), R0 = this.params.p('R0');
    const eps = this.params.p('epsilon'), sig = this.params.p('sigma'), del = this.params.p('delta');
    const rmin = Math.pow(2, 1 / 6);
    bondLoop(bc, (r, t, k) => {
      const r0 = r - del[t];
      const arg = feneArg(bc, 1 - (r0 * r0) / (R0[t] * R0[t]), k, r);
      const e = -0.5 * K[t] * R0[t] * R0[t] * Math.log(arg);
      const dEdr = (K[t] * r0) / arg;
      if (r0 > 0 && r < rmin * sig[t] + del[t]) {
        const s = sig[t] / r0, s6 = s ** 6, s12 = s6 * s6;
        return { e: e + 4 * eps[t] * (s12 - s6) + eps[t], dEdr: dEdr + 4 * eps[t] * (-12 * s12 / r0 + 6 * s6 / r0) };
      }
      return { e, dEdr };
    });
  }

  equilibrium(type: number): number { return this.params.p('R0')[type] + this.params.p('delta')[type]; }
}

/*
 * bond_style morse — docs.lammps.org/bond_morse.html:
 *   E = D [ 1 - e^(-alpha (r - r0)) ]^2
 *   "r0 is the equilibrium bond distance, alpha is a stiffness parameter, and
 *   D determines the depth of the potential well."
 *   coefficients: "D (energy)", "alpha (inverse distance)", "r0 (distance)".
 */
export class BondMorse extends SimpleBonded {
  readonly name = 'morse';
  readonly kind = 'bond' as const;
  readonly paramNames = ['D', 'alpha', 'r0'];

  compute(bc: BondedCompute): void {
    const D = this.params.p('D'), al = this.params.p('alpha'), r0 = this.params.p('r0');
    bondLoop(bc, (r, t) => {
      const dr = r - r0[t];
      const ex = Math.exp(-al[t] * dr);
      return { e: D[t] * (1 - ex) * (1 - ex), dEdr: 2 * D[t] * (1 - ex) * al[t] * ex };
    });
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }
}

/*
 * bond_style nonlinear — docs.lammps.org/bond_nonlinear.html:
 *   E = epsilon (r - r0)^2 / [ lambda^2 - (r - r0)^2 ]
 *   "an anharmonic spring (Rector) of equilibrium length r0 and maximum
 *   extension lamda."
 *   coefficients: "epsilon (energy)", "r0 (distance)", "lambda (distance)".
 */
export class BondNonlinear extends SimpleBonded {
  readonly name = 'nonlinear';
  readonly kind = 'bond' as const;
  readonly paramNames = ['epsilon', 'r0', 'lam'];

  compute(bc: BondedCompute): void {
    const eps = this.params.p('epsilon'), r0 = this.params.p('r0'), lam = this.params.p('lam');
    bondLoop(bc, (r, t) => {
      const u = r - r0[t], u2 = u * u;
      const denom = lam[t] * lam[t] - u2;
      return { e: (eps[t] * u2) / denom, dEdr: (2 * eps[t] * u * lam[t] * lam[t]) / (denom * denom) };
    });
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }
}

/*
 * bond_style class2 — docs.lammps.org/bond_class2.html:
 *   E = K2 (r - r0)^2 + K3 (r - r0)^3 + K4 (r - r0)^4
 *   coefficients (in this order): "r0 (distance)", "K2 (energy/distance^2)",
 *   "K3 (energy/distance^3)", "K4 (energy/distance^4)".
 */
export class BondClass2 extends SimpleBonded {
  readonly name = 'class2';
  readonly kind = 'bond' as const;
  readonly paramNames = ['r0', 'K2', 'K3', 'K4'];

  compute(bc: BondedCompute): void {
    const r0 = this.params.p('r0'), K2 = this.params.p('K2'), K3 = this.params.p('K3'), K4 = this.params.p('K4');
    bondLoop(bc, (r, t) => {
      const u = r - r0[t], u2 = u * u;
      return {
        e: K2[t] * u2 + K3[t] * u2 * u + K4[t] * u2 * u2,
        dEdr: 2 * K2[t] * u + 3 * K3[t] * u2 + 4 * K4[t] * u2 * u,
      };
    });
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }
}

/*
 * bond_style gromos — docs.lammps.org/bond_gromos.html:
 *   E = K (r^2 - r0^2)^2
 *   "Note that the usual 1/4 factor is included in K."
 *   coefficients: "K (energy/distance^4)", "r0 (distance)".
 */
export class BondGromos extends SimpleBonded {
  readonly name = 'gromos';
  readonly kind = 'bond' as const;
  readonly paramNames = ['K', 'r0'];

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), r0 = this.params.p('r0');
    bondLoop(bc, (r, t) => {
      const dr2 = r * r - r0[t] * r0[t];
      return { e: K[t] * dr2 * dr2, dEdr: 4 * K[t] * r * dr2 };
    });
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }
}

/*
 * bond_style harmonic/shift — docs.lammps.org/bond_harmonic_shift.html:
 *   E = Umin / (r0 - rc)^2 * [ (r - r0)^2 - (rc - r0)^2 ]
 *   "The potential energy has the value -Umin at r0 and zero at rc."
 *   coefficients: "Umin (energy)", "r0 (distance)", "rc (distance)".
 */
export class BondHarmonicShift extends SimpleBonded {
  readonly name = 'harmonic/shift';
  readonly kind = 'bond' as const;
  readonly paramNames = ['Umin', 'r0', 'rc'];

  compute(bc: BondedCompute): void {
    const U = this.params.p('Umin'), r0 = this.params.p('r0'), rc = this.params.p('rc');
    bondLoop(bc, (r, t) => {
      const dr = r - r0[t], drc = rc[t] - r0[t];
      const k = U[t] / (drc * drc);
      return { e: k * (dr * dr - drc * drc), dEdr: 2 * k * dr };
    });
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }
}

/*
 * bond_style harmonic/shift/cut — docs.lammps.org/bond_harmonic_shift_cut.html:
 *   E = Umin / (r0 - rc)^2 * [ (r - r0)^2 - (rc - r0)^2 ]
 *   "The bond potential is zero and thus its force also zero for distances
 *   r > rc.  The potential energy has the value -Umin at r0 and zero at rc."
 *   coefficients: "Umin (energy)", "r0 (distance)", "rc (distance)".
 */
export class BondHarmonicShiftCut extends SimpleBonded {
  readonly name = 'harmonic/shift/cut';
  readonly kind = 'bond' as const;
  readonly paramNames = ['Umin', 'r0', 'rc'];

  compute(bc: BondedCompute): void {
    const U = this.params.p('Umin'), r0 = this.params.p('r0'), rc = this.params.p('rc');
    bondLoop(bc, (r, t) => {
      if (r > rc[t]) return { e: 0, dEdr: 0 };
      const dr = r - r0[t], drc = rc[t] - r0[t];
      const k = U[t] / (drc * drc);
      return { e: k * (dr * dr - drc * drc), dEdr: 2 * k * dr };
    });
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }
}

/*
 * bond_style zero — docs.lammps.org/bond_zero.html:
 *   "Using an bond style of zero means bond forces and energies are not
 *   computed, but the geometry of bond pairs is still accessible to other
 *   commands." "The optional nocoeff flag allows to read data files with a
 *   BondCoeff section for any bond style.  Similarly, any bond_coeff commands
 *   will only be checked for the bond type number and the rest ignored."
 *   "Note that the bond_coeff command must be used for all bond types.  If
 *   specified, there can be only one value, which is going to be used to
 *   assign an equilibrium distance, e.g. for use with fix shake."
 */
export class BondZero extends Bonded {
  readonly name = 'zero';
  readonly kind = 'bond' as const;
  private nocoeff = false;
  private eq = new Float64Array(0);
  private set = new Uint8Array(0);
  ntypes = 0;

  settings(args: string[], _ctx?: StyleContext): void {
    for (const a of args) if (a !== 'nocoeff') throw new StyleError(`bond_style zero: unknown keyword '${a}'`);
    this.nocoeff = args.length > 0;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.eq = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    const [lo, hi] = typeBounds(args[0] ?? '', this.ntypes);
    // "If specified, there can be only one value" (bond_zero.html)
    if (args.length > 2) throw new StyleError('bond_coeff zero: incorrect args (only one value allowed)');
    if (args.length === 2) {
      const v = parseNum(args[1], 'r0');
      for (let t = lo; t <= hi; t++) this.eq[t] = v;
    }
    for (let t = lo; t <= hi; t++) this.set[t] = 1;
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`bond_coeff must be used for all bond types with bond_style zero (type ${t} not set)`);
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
