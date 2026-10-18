import { SimpleBonded, atomIndex, termDelta } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';
import { StyleError, type BondedCompute, type Pair, type StyleContext } from '../types';
import { angleTerm } from './harmonic';
import { PairLJSpica } from '../pair/class2_spica51';
import { PairHybrid } from '../pair/hybrid';

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
/*
 * The lj/spica (or lj/sdk) pair style whose coefficients give the 1-3 repulsion: the pair style itself, or the
 * first lj/spica sub-style of a hybrid pair style. docs.lammps.org/angle_spica.html: "The required lj/spica
 * parameters are extracted automatically from the pair_style." The pair_style page for hybrid says that the
 * sub-style is named in the pair_style command (pair_hybrid.html). Measured with native LAMMPS (black box):
 * pair_style hybrid/overlay lj/sdk 15.0 coul/long 26.5 with angle_style sdk gives the 1-3 repulsion of the
 * lj/sdk sub-style (w104cf_sds_overlay); a type pair without lj/sdk coefficients gets no repulsion
 * (w104cf_sds_overlay_unset).
 */
function spicaSubStyle(pair: Pair | null): PairLJSpica | null {
  if (pair instanceof PairLJSpica) return pair;
  if (pair instanceof PairHybrid) {
    for (const sub of pair.subs) if (sub.style instanceof PairLJSpica) return sub.style;
  }
  return null;
}

export class AngleSpica extends SimpleBonded {
  readonly name: string;
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'theta0'];
  private pair: PairLJSpica | null = null;
  /** The hybrid pair style that holds the lj/spica sub-style, and that sub-style's index in it (null when not hybrid). */
  private hybrid: { pair: PairHybrid; index: number } | null = null;

  constructor(name: 'spica' | 'sdk') {
    super();
    this.name = name;
  }

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError(`angle_coeff ${this.name} needs K theta0`);
    // As native LAMMPS does (src/CG-SPICA/angle_spica.cpp at b332f28): theta0[i] = theta0_one/180.0 * MY_PI
    return [parseNum(args[0], 'K'), (parseNum(args[1], 'theta0') / 180) * Math.PI];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    // As native LAMMPS does (src/CG-SPICA/angle_spica.cpp at b332f28): write_data: theta0[i]/MY_PI*180.0
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((th[t] / Math.PI) * 180)}`);
    return out;
  }

  // As native LAMMPS does (src/angle_write.cpp at b332f28, double theta0 = equilibrium_angle(atype) * RAD2DEG
  equilibrium(type: number): number { return (180 / Math.PI) * this.params.p('theta0')[type]; }

  linkForceField(link: { pair: Pair | null }): void {
    this.pair = spicaSubStyle(link.pair);
    this.hybrid = null;
    const p = link.pair;
    if (this.pair && p instanceof PairHybrid) {
      const index = p.subs.findIndex((sub) => sub.style === this.pair);
      this.hybrid = { pair: p, index };
    }
  }

  /**
   * The 1-3 repulsion of a type pair. Inside a hybrid pair style the lj/spica sub-style gives it only for the
   * type pairs the hybrid assigns to that sub-style. Measured with native LAMMPS (black box): after pair_coeff 3 3
   * lj/sdk and then pair_coeff 3 3 coul/long in plain hybrid, the 3-3 pair has no repulsion
   * (w104cf_sds_hybrid_reassign, the same eangle as w104cf_sds_hybrid).
   */
  private repulsion(itype: number, jtype: number): ReturnType<PairLJSpica['angleRepulsion']> {
    const pair = this.pair;
    if (!pair) return null;
    if (this.hybrid && !this.hybrid.pair.subsOf(itype, jtype).includes(this.hybrid.index)) return null;
    return pair.angleRepulsion(itype, jtype);
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
    if (!this.pair) throw new StyleError(`angle_style ${this.name} needs pair_style lj/spica (or lj/sdk)`);
    const f = bc.f;
    const d = [0, 0, 0];
    let er = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const rep = this.repulsion(bc.s.type[i], bc.s.type[k]);
      if (!rep) continue;
      termDelta(bc, j, k, i, d);  // r_i - r_k, both copies nearest to the centre j
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
