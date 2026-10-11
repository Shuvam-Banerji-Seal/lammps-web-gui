import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';
import { StyleError, type BondedCompute, type Pair, type StyleContext } from '../types';
import { angleTerm } from './harmonic';
import { PairLJSpica } from '../pair/class2_spica51';

/*
 * angle_style spica (and its original name sdk) — docs.lammps.org/angle_spica.html:
 * "The *spica* angle style is a combination of the harmonic angle potential,"
 *   E = K (theta - theta_0)^2
 * "with the *repulsive* part of the non-bonded *lj/spica* pair style between the atoms 1 and 3."
 * "Relative to the pair_style *lj/spica*, however, the energy is shifted by epsilon, to avoid sudden jumps."
 * Coefficients: "K (energy)" and "theta_0 (degrees)" as harmonic; "The required *lj/spica* parameters are
 * extracted automatically from the pair_style." "Style *sdk*, the original implementation of style *spica*, is
 * available for backward compatibility."
 * The harmonic part is the angle_style harmonic term (force/angle/harmonic.ts angleTerm). The 1-3 repulsion is
 * computed by the linked pair style (PairLJSpica.angleRepulsion, measured with native LAMMPS, black box): the
 * unshifted LJ term plus epsilon below the LJ minimum, zero beyond it; the pair cutoff and pair_modify shift
 * do not change it. Per-atom energy of the repulsion is split evenly between atoms 1 and 3 (not measured).
 */
export class AngleSpica extends SimpleBonded {
  readonly name: string;
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'theta0'];
  private pair: PairLJSpica | null = null;

  constructor(name: 'spica' | 'sdk') {
    super();
    this.name = name;
  }

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError(`angle_coeff ${this.name} needs K theta0`);
    return [parseNum(args[0], 'K'), (parseNum(args[1], 'theta0') * Math.PI) / 180];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((th[t] * 180) / Math.PI)}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

  linkForceField(link: { pair: Pair | null }): void {
    this.pair = link.pair instanceof PairLJSpica ? link.pair : null;
  }

  init(ctx: StyleContext): void {
    super.init(ctx);
    if (!this.pair) throw new StyleError(`angle_style ${this.name} needs pair_style lj/spica (or lj/sdk) for its 1-3 repulsion`);
  }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const K = this.params.p('K'), th0 = this.params.p('theta0');
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
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
    }
    bc.acc.eangle += e;
    const pair = this.pair;
    if (!pair) throw new StyleError(`angle_style ${this.name} needs pair_style lj/spica (or lj/sdk)`);
    const f = bc.f;
    const d = [0, 0, 0];
    let er = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const rep = pair.angleRepulsion(bc.s.type[i], bc.s.type[k]);
      if (!rep) continue;
      delta(bc, k, i, d);  // r_i - r_k (minimum image)
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (!(r < rep.rmin)) continue;
      const rinv = 1 / r;
      const rn = Math.pow(rinv, rep.n), rm = Math.pow(rinv, rep.m);
      const eRep = rep.a * rn - rep.b * rm + rep.eps;
      // F/r of the unshifted LJ term; the force on atom 1 is fpair * (r_1 - r_3)
      const fpair = (rep.n * rep.a * rn - rep.m * rep.b * rm) * rinv * rinv;
      for (let q = 0; q < 3; q++) {
        f[3 * i + q] += d[q] * fpair;
        f[3 * k + q] -= d[q] * fpair;
      }
      const w = [d[0] * d[0] * fpair, d[1] * d[1] * fpair, d[2] * d[2] * fpair, d[0] * d[1] * fpair, d[0] * d[2] * fpair, d[1] * d[2] * fpair];
      for (let q = 0; q < 6; q++) bc.virial[q] += w[q];
      if (bc.eatom) { bc.eatom[i] += 0.5 * eRep; bc.eatom[k] += 0.5 * eRep; }
      er += eRep;
    }
    bc.acc.eangle += er;
  }
}
