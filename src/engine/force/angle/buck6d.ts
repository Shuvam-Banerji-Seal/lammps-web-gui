import { SimpleBonded, atomIndex, termDelta } from '../bonded_util';
import { angleTerm } from './harmonic';
import { PairBuck6dCoulGauss } from '../pair/buck6d_gauss';
import { fmtCoeff, parseInt_, parseNum } from '../util';
import { StyleError, addPairCentroid, type BondedCompute, type StyleContext, bondedCoeffArgs } from '../types';
import type { CharmmLink } from '../dihedral/charmm';
import { libm } from '../../libm';

/*
 * angle_style cosine/buck6d — docs.lammps.org/angle_cosine_buck6d.html (text in
 * plans/lammps-docs-live/angle_cosine_buck6d.rst):
 *   "E = K \left[ 1 + \cos(n\theta - \theta_0)\right]"; coefficients "K (energy)", "n", "\theta_0 (degrees)".
 *   "Additional to the cosine term the *cosine/buck6d* angle style computes the short range (vdW) interaction
 *   belonging to the pair_style buck6d between the end atoms of the angle."
 *   "*cosine/buck6d* can only be used in combination with the pair_style buck6d style and with a special_bonds
 *   0.0 weighting of 1-3 interactions."
 *
 * Cosine term. Measured with native LAMMPS (black box): the eangle of a 4-atom chain equals K [1 + cos(n theta - theta_0)]
 * summed over the angles, with theta_0 in degrees converted to radians (K 1.97835, n 4, theta_0 180 and 120).
 *
 * vdW term. Measured with native LAMMPS (black box): the buck6d vdW of the two end atoms of each angle (types i and k,
 * distance r_ik, with the pair's own coefficients, smoothing and cutoff) is added to evdwl, not to eangle, and it is
 * not scaled by special_bonds. The added amount equals the sum of the two end-atom pair energies to 1e-10 (chain 1-2-3-4
 * with distinct per-type buck6d coefficients: -20.6374 + -48.8165 = -69.4539 change in evdwl, pair-only evdwl 82.4998).
 * Both buck6d/coul/gauss/dsf and buck6d/coul/gauss/long gave the same change. Charges do not enter this term.
 *
 * Restrictions. Measured with native LAMMPS (black box): with special_bonds lj 1-3 not 0 the run aborts (the angle
 * style requires the 1-3 lj weight to be 0, since the buck6d 1-3 interaction would otherwise be counted twice); this
 * engine refuses it likewise. With a non-buck6d pair style the angle has no vdW partner, so this style refuses to run.
 */
export class AngleCosineBuck6d extends SimpleBonded {
  readonly name = 'cosine/buck6d';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'n', 'theta0'];
  private link: CharmmLink | null = null;
  private pair: PairBuck6dCoulGauss | null = null;

  /** The force field passes its pair style and special_bonds before init (see Bonded.linkForceField). */
  linkForceField(link: CharmmLink): void {
    this.link = link;
  }

  protected parse(args: string[]): number[] {
    if (args.length !== 3) throw bondedCoeffArgs('angle', 'angle_coeff cosine/buck6d needs K n theta0');
    const K = parseNum(args[0], 'K');
    const n = parseInt_(args[1], 'n');
    if (n < 0) throw new StyleError('angle cosine/buck6d n must be an integer >= 0');
    // As native LAMMPS does (src/MOFFF/angle_cosine_buck6d.cpp at b332f28): th0[i] = th0_one/180.0 * MY_PI
    return [K, n, (parseNum(args[2], 'theta0') / 180) * Math.PI];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), n = this.params.p('n'), th = this.params.p('theta0');
    const out: string[] = [];
    // As native LAMMPS does (src/MOFFF/angle_cosine_buck6d.cpp at b332f28): write_data prints the stored radians,
    // fprintf(fp,"%d %g %d %g\n",i,k[i],multiplicity[i],th0[i]) (read back, the value is taken as degrees)
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff(n[t])} ${fmtCoeff(th[t])}`);
    return out;
  }

  init(ctx?: StyleContext): void {
    super.init(ctx);
    const link = this.link;
    const pair = link?.pair ?? null;
    if (!(pair instanceof PairBuck6dCoulGauss)) {
      throw new StyleError(`angle_style cosine/buck6d needs pair_style buck6d/coul/gauss/dsf or buck6d/coul/gauss/long (got ${pair?.name ?? 'none'})`);
    }
    if (!link || link.special.lj[1] !== 0) {
      throw new StyleError('angle_style cosine/buck6d requires special_bonds lj 1-3 weight 0.0 (the buck6d 1-3 interaction would be counted twice)');
    }
    this.pair = pair;
  }

  /** angle_cosine_buck6d.cpp equilibrium_angle(): MY_PI */
  equilibriumRadians(_type: number): number { return Math.PI; }
  equilibrium(type: number): number { return (180 / Math.PI) * this.equilibriumRadians(type); }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const K = this.params.p('K'), n = this.params.p('n'), th0 = this.params.p('theta0');
    const pair = this.pair;
    if (!pair) throw new StyleError('angle_style cosine/buck6d: no pair style linked');
    const d1 = [0, 0, 0], d2 = [0, 0, 0], dk = [0, 0, 0];
    let e = 0, evdw = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const arg = n[t] * theta - th0[t];
        const ea = K[t] * (1 + libm.cos(arg));
        e += ea;
        return { e: ea, dEdtheta: -K[t] * n[t] * libm.sin(arg) };
      });
      // buck6d vdW between the end atoms i and k: dk = r_k - r_i (minimum image)
      termDelta(bc, j, i, k, dk);
      const r = Math.sqrt(dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]);
      const v = r > 0 ? pair.vdwEnergy(bc.s.type[i], bc.s.type[k], r) : null;
      if (!v) continue;
      evdw += v.e;
      const g = -v.de / r;
      const fk = [g * dk[0], g * dk[1], g * dk[2]];
      const f = bc.f;
      for (let c = 0; c < 3; c++) {
        f[3 * k + c] += fk[c];
        f[3 * i + c] -= fk[c];
      }
      const w = [dk[0] * fk[0], dk[1] * fk[1], dk[2] * fk[2], dk[0] * fk[1], dk[0] * fk[2], dk[1] * fk[2]];
      for (let q = 0; q < 6; q++) bc.virial[q] += w[q];
      // native tallies this 1-3 pair through force->pair->ev_tally (angle_cosine_buck6d.cpp): the pair term's per-atom
      // energy and virial, half to each atom
      const ea = bc.pairEatom ?? bc.eatom, va = bc.pairVatom ?? bc.vatom;
      if (ea) { ea[i] += v.e / 2; ea[k] += v.e / 2; }
      if (va) for (const at of [i, k]) for (let q = 0; q < 6; q++) va[6 * at + q] += w[q] / 2;
      if (bc.cvatom && !bc.pairVatom) addPairCentroid(bc.cvatom, i, k, w);
    }
    bc.acc.eangle += e;
    bc.acc.evdwl += evdw;
  }
}
