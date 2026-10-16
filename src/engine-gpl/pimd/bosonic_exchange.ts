/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/bosonic_exchange.cpp and src/REPLICA/bosonic_exchange.h (BosonicExchange: cycle energies,
 * the V and V_backwards recursions, the connection probabilities, the spring forces of the exterior and interior beads,
 * the bead spring energy and the primitive kinetic estimator) at commit b332f28. Authors of the source: Ofir Blumer,
 * Jacob Higer, Yotam Feldman and Barak Hirshberg (Hirshberg lab, Tel Aviv University). Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The bosons are the atoms of a bead, indexed by the bead's local atom index (native: the same index in every bead's
 * position buffer). The exchange has no random numbers: the same positions give the same estimators.
 *
 * Native Hirshberg et al., doi:10.1073/pnas.1913365116 (Supporting Information, Eqs. 4-5), and the convention notes of
 * prim_estimator are kept as comments in the code below.
 */
import { StyleError } from '../../engine/force/types';
import { log as libmLog } from '../libm/log';

export class BosonicExchange {
  /** scratch: temp_nbosons_array (n + 1) */
  private readonly tmp: Float64Array;
  /** E_kn, indexed by m(m+1)/2 - k */
  private readonly Ekn: Float64Array;
  /** V and V_backwards (n + 1 each) */
  private readonly V: Float64Array;
  private readonly Vb: Float64Array;
  /** connection_probabilities, n x n row-major: [n * l + u] */
  private readonly conn: Float64Array;
  private x: Float64Array = new Float64Array(0);
  private xPrev: Float64Array = new Float64Array(0);
  private xNext: Float64Array = new Float64Array(0);
  private beta = 0;
  private springConstant = 0;

  /**
   * @param nbosons atoms per bead
   * @param np beads
   * @param beadNum this bead (0..np-1)
   * @param minImage the minimum-image of a separation along a dimension (domain->minimum_image)
   * @param dimension the simulation dimension (domain->dimension)
   * @param physicalBeta true: the physical-beta convention (native fix pimd/nvt/bosonic), false: the reduced one
   */
  constructor(
    private readonly nbosons: number,
    private readonly np: number,
    private readonly beadNum: number,
    private readonly minImage: (d: number, dim: number) => number,
    private readonly dimension: number,
    private readonly physicalBeta: boolean,
  ) {
    if (!(nbosons >= 1)) throw new StyleError('bosonic exchange needs at least one boson');
    this.tmp = new Float64Array(nbosons + 1);
    this.Ekn = new Float64Array((nbosons * (nbosons + 1)) / 2 + 1);
    this.V = new Float64Array(nbosons + 1);
    this.Vb = new Float64Array(nbosons + 1);
    this.conn = new Float64Array(nbosons * nbosons);
  }

  /** prepare_with_coordinates: the bead's positions, its neighbours' and the constants; exterior beads evaluate the exchange. */
  prepare(x: Float64Array, xPrev: Float64Array, xNext: Float64Array, beta: number, springConstant: number): void {
    this.x = x;
    this.xPrev = xPrev;
    this.xNext = xNext;
    this.beta = beta;
    this.springConstant = springConstant;
    if (this.beadNum === 0 || this.beadNum === this.np - 1) {
      this.evaluateCycleEnergies();
      this.evaluateVBn();
      this.evaluateVBackwards();
      this.evaluateConnectionProbabilities();
    }
  }

  /** diff_two_beads: the minimum-image separation x2[l2] - x1[l1] (boson indices taken modulo nbosons). */
  private diff(x1: Float64Array, l1: number, x2: Float64Array, l2: number): [number, number, number] {
    const a = l1 % this.nbosons, b = l2 % this.nbosons;
    const dx = x2[3 * b] - x1[3 * a];
    const dy = x2[3 * b + 1] - x1[3 * a + 1];
    const dz = x2[3 * b + 2] - x1[3 * a + 2];
    return [this.minImage(dx, 0), this.minImage(dy, 1), this.minImage(dz, 2)];
  }

  private distSq(x1: Float64Array, l1: number, x2: Float64Array, l2: number): number {
    const d = this.diff(x1, l1, x2, l2);
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  }

  private getEnk(m: number, k: number): number {
    return this.Ekn[(m * (m + 1)) / 2 - k];
  }

  private setEnk(m: number, k: number, val: number): void {
    this.Ekn[(m * (m + 1)) / 2 - k] = val;
  }

  /** evaluate_cycle_energies: the energies of the cycles that join the bosons u..v (E_kn). */
  private evaluateCycleEnergies(): void {
    const n = this.nbosons;
    const k = this.springConstant;
    let xFirst: Float64Array, xLast: Float64Array;
    if (this.beadNum === 0) { xFirst = this.x; xLast = this.xPrev; } else { xFirst = this.xNext; xLast = this.x; }
    for (let i = 0; i < n; i++) this.tmp[i] = this.distSq(xFirst, i, xLast, i);
    for (let v = 0; v < n; v++) {
      this.setEnk(v + 1, 1, 0.5 * k * this.tmp[v]);
      for (let u = v - 1; u >= 0; u--) {
        const val = this.getEnk(v + 1, v - u)
          + 0.5 * k * (
            // connect u to u+1
            +this.distSq(xLast, u, xFirst, u + 1)
            // break cycle [u+1,v]
            - this.distSq(xFirst, u + 1, xLast, v)
            // close cycle from v to u
            + this.distSq(xFirst, u, xLast, v));
        this.setEnk(v + 1, v - u + 1, val);
      }
    }
  }

  /** Evaluate_VBn: the partition-function potential V of the first m bosons (V[n] is the total). */
  private evaluateVBn(): void {
    const n = this.nbosons;
    const beta = this.beta;
    this.V[0] = 0.0;
    for (let m = 1; m < n + 1; m++) {
      let Elongest = Number.MAX_VALUE;
      for (let k = m; k > 0; k--) {
        const val = this.getEnk(m, k) + this.V[m - k];
        // numerical stability (Xiong & Xiong, doi:10.1103/PhysRevE.106.025309)
        Elongest = Math.min(Elongest, val);
        this.tmp[k - 1] = val;
      }
      let sigDenom = 0.0;
      for (let k = m; k > 0; k--) sigDenom += Math.exp(-beta * (this.tmp[k - 1] - Elongest));
      this.V[m] = Elongest - (1.0 / beta) * libmLog(sigDenom / m);
      if (!Number.isFinite(this.V[m])) throw new StyleError(`Invalid sig_denom ${sigDenom} with Elongest ${Elongest} in bosonic exchange potential`);
    }
  }

  /** Evaluate_V_backwards: the potential of the bosons l..n-1 (V_backwards); V_backwards[0] = V[n]. */
  private evaluateVBackwards(): void {
    const n = this.nbosons;
    const beta = this.beta;
    this.Vb[n] = 0.0;
    for (let l = n - 1; l > 0; l--) {
      let Elongest = Number.MAX_VALUE;
      for (let p = l; p < n; p++) {
        const val = this.getEnk(p + 1, p - l + 1) + this.Vb[p + 1];
        Elongest = Math.min(Elongest, val);
        this.tmp[p] = val;
      }
      let sigDenom = 0.0;
      for (let p = l; p < n; p++) sigDenom += (1.0 / (p + 1)) * Math.exp(-beta * (this.tmp[p] - Elongest));
      this.Vb[l] = Elongest - libmLog(sigDenom) / beta;
      if (!Number.isFinite(this.Vb[l])) throw new StyleError(`Invalid sig_denom ${sigDenom} with Elongest ${Elongest} in bosonic exchange potential backwards`);
    }
    this.Vb[0] = this.V[n];
  }

  /** evaluate_connection_probabilities: the direct link of l and l+1, and the closing of a cycle from u to l. */
  private evaluateConnectionProbabilities(): void {
    const n = this.nbosons;
    const beta = this.beta;
    for (let l = 0; l < n - 1; l++) {
      this.conn[n * l + (l + 1)] = 1.0 - Math.exp(-beta * (this.V[l + 1] + this.Vb[l + 1] - this.V[n]));
    }
    for (let u = 0; u < n; u++) {
      for (let l = u; l < n; l++) {
        this.conn[n * l + u] = (1.0 / (l + 1)) * Math.exp(-beta * (this.V[u] + this.getEnk(l + 1, l - u + 1) + this.Vb[l + 1] - this.V[n]));
      }
    }
  }

  /** get_potential: the exchange potential of the whole bead set (V[n]). */
  potential(): number { return this.V[this.nbosons]; }

  /** get_interior_bead_spring_energy: the springs of this bead to the previous bead. */
  private interiorSpringEnergy(): number {
    let e = 0;
    for (let i = 0; i < this.nbosons; i++) e += 0.5 * this.springConstant * this.distSq(this.x, i, this.xPrev, i);
    return e;
  }

  /** get_bead_spring_energy: the exchange potential for the first bead, the interior springs otherwise. */
  beadSpringEnergy(): number {
    return this.beadNum === 0 ? this.potential() : this.interiorSpringEnergy();
  }

  /** spring_force: adds this bead's exchange-aware spring force to f (the local array of the bead). */
  springForce(f: Float64Array): void {
    if (this.beadNum === this.np - 1) this.springForceLast(f);
    else if (this.beadNum === 0) this.springForceFirst(f);
    else this.springForceInterior(f);
  }

  private springForceLast(f: Float64Array): void {
    const n = this.nbosons;
    const xFirst = this.xNext, xLast = this.x;
    for (let l = 0; l < n; l++) {
      let sx = 0.0, sy = 0.0, sz = 0.0;
      for (let nl = 0; nl <= l + 1 && nl < n; nl++) {
        const d = this.diff(xLast, l, xFirst, nl);
        const prob = this.conn[n * l + nl];
        sx += prob * d[0]; sy += prob * d[1]; sz += prob * d[2];
      }
      const dp = this.diff(xLast, l, this.xPrev, l);
      sx += dp[0]; sy += dp[1]; sz += dp[2];
      f[3 * l] += sx * this.springConstant;
      f[3 * l + 1] += sy * this.springConstant;
      f[3 * l + 2] += sz * this.springConstant;
    }
  }

  private springForceFirst(f: Float64Array): void {
    const n = this.nbosons;
    const xFirst = this.x, xLast = this.xPrev;
    for (let l = 0; l < n; l++) {
      let sx = 0.0, sy = 0.0, sz = 0.0;
      for (let pl = Math.max(0, l - 1); pl < n; pl++) {
        const d = this.diff(xFirst, l, xLast, pl);
        const prob = this.conn[n * pl + l];
        sx += prob * d[0]; sy += prob * d[1]; sz += prob * d[2];
      }
      const dn = this.diff(xFirst, l, this.xNext, l);
      sx += dn[0]; sy += dn[1]; sz += dn[2];
      f[3 * l] += sx * this.springConstant;
      f[3 * l + 1] += sy * this.springConstant;
      f[3 * l + 2] += sz * this.springConstant;
    }
  }

  private springForceInterior(f: Float64Array): void {
    for (let l = 0; l < this.nbosons; l++) {
      const dp = this.diff(this.x, l, this.xPrev, l);
      const dn = this.diff(this.x, l, this.xNext, l);
      const sx = dp[0] + dn[0], sy = dp[1] + dn[1], sz = dp[2] + dn[2];
      f[3 * l] += sx * this.springConstant;
      f[3 * l + 1] += sy * this.springConstant;
      f[3 * l + 2] += sz * this.springConstant;
    }
  }

  /**
   * prim_estimator: the primitive kinetic energy estimator of this bead. Native comments kept in short: the reduced-beta
   * convention (exp[-(beta/P) H]) and the physical-beta one (exp[-beta H]) differ by the factor 1/P (convention_correction);
   * the choice only changes this estimator.
   */
  primEstimator(): number {
    const n = this.nbosons;
    const beta = this.beta;
    const convention = this.physicalBeta ? 1 : 1.0 / this.np;
    if (this.beadNum !== 0) {
      return convention * (0.5 * this.dimension * n / beta - this.interiorSpringEnergy());
    }
    // first bead: the exterior bead's contribution (Eqs. 4-5 of the Supporting Information of Hirshberg et al.)
    this.tmp[0] = 0.0;
    for (let m = 1; m < n + 1; m++) {
      let sig = 0.0;
      this.tmp[m] = 0.0;
      let Elongest = Number.MAX_VALUE;
      for (let k = m; k > 0; k--) Elongest = Math.min(Elongest, this.getEnk(m, k) + this.V[m - k]);
      for (let k = m; k > 0; k--) {
        const Ekn = this.getEnk(m, k);
        sig += (this.tmp[m - k] - Ekn) * Math.exp(-beta * (Ekn + this.V[m - k] - Elongest));
      }
      const sigDenomM = m * Math.exp(-beta * (this.V[m] - Elongest));
      this.tmp[m] = sig / sigDenomM;
    }
    return convention * (0.5 * this.dimension * n / beta + this.tmp[n]);
  }
}
