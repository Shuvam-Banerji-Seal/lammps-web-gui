import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style colloid — docs.lammps.org/pair_colloid.html
 * (source: plans/lammps-docs/pair_colloid.rst)
 *
 * Syntax (verbatim):
 *
 *    pair_style colloid cutoff
 *
 * * cutoff = global cutoff for colloidal interactions (distance units)
 *
 * "Style *colloid* computes pairwise interactions between large colloidal
 * particles and small solvent particles using 3 formulas.  A colloidal
 * particle has a size > sigma; a solvent particle is the usual
 * Lennard-Jones particle of size sigma."
 *
 * Colloid-colloid energy (verbatim LaTeX from the doc page):
 *
 *    U_A = & - \frac{A_{cc}}{6} \left[
 *    \frac{2 a_1 a_2}{r^2-\left(a_1+a_2\right)^2}
 *    + \frac{2 a_1 a_2}{r^2 - \left(a_1 - a_2\right)^2}
 *      + \mathrm{ln}
 *        \left(
 *   \frac{r^2-\left(a_1+a_2\right)^2}{r^2-\left(a_1-a_2\right)^2}
 *    \right)
 *   \right] \\
 *     & \\
 *     U_R = & \frac{A_{cc}}{37800}  \frac{\sigma^6}{r}
 *     \biggl[ \frac{r^2-7r\left(a_1+a_2\right)+6\left(a_1^2+7a_1a_2+a_2^2\right)}
 *   {\left(r-a_1-a_2\right)^7} \\
 *    &\qquad              +\frac{r^2+7r\left(a_1+a_2\right)+6\left(a_1^2+7a_1a_2+a_2^2\right)}
 *   {\left(r+a_1+a_2\right)^7}  \\
 *   &\qquad               -\frac{r^2+7r\left(a_1-a_2\right)+6\left(a_1^2-7a_1a_2+a_2^2\right)}
 *   {\left(r+a_1-a_2\right)^7} \\
 *   &\qquad       \left.  -\frac{r^2-7r\left(a_1-a_2\right)+6\left(a_1^2-7a_1a_2+a_2^2\right)}
 *   {\left(r-a_1+a_2\right)^7}
 *   \right]  \\
 *   & \\
 *   U = & U_A + U_R, \qquad r < r_c
 *
 * "where :math:`A_{cc}` is the Hamaker constant, :math:`a_1` and :math:`a_2` are the
 * radii of the two colloidal particles, and :math:`r_c` is the cutoff."
 *
 * Colloid-solvent energy (verbatim):
 *
 *    U = \frac{2 ~ a^3 ~ \sigma^3 ~ A_{cs}}{9 \left( a^2 - r^2 \right)^3}
 *    \left[ 1 - \frac{\left(5 ~ a^6+45~a^4~r^2+63~a^2~r^4+15~r^6\right) \sigma^6}
 *    {15 \left(a-r\right)^6 \left( a+r \right)^6} \right], \quad r < r_c
 *
 * "where :math:`A_{cs}` is the Hamaker constant, *a* is the radius of the colloidal
 * particle, and :math:`r_c` is the cutoff."
 *
 * Solvent-solvent energy (verbatim):
 *
 *    U = \frac{A_{ss}}{36} \left[ \left( \frac{\sigma}{r}
 *         \right)^{12} - \left( \frac{ \sigma}{r} \right)^6 \right], \quad
 *         r < r_c
 *
 * Coefficients (verbatim):
 *
 * * A (energy units)
 * * :math:`\sigma` (distance units)
 * * d1 (distance units)
 * * d2 (distance units)
 * * cutoff (distance units)
 *
 * "D1 and d2 are particle diameters, so that d1 = 2\*a1 and d2 = 2\*a2 in
 * the formulas above.  Both d1 and d2 must be values >= 0.  If d1 > 0
 * and d2 > 0, then the pair interacts via the colloid-colloid formula
 * above.  If d1 = 0 and d2 = 0, then the pair interacts via the
 * solvent-solvent formula.  I.e. a d value of 0 is a Lennard-Jones
 * particle of size :math:`\sigma`.  If either d1 = 0 or d2 = 0 and the other is
 * larger, then the pair interacts via the colloid-solvent formula."
 * "The last coefficient is optional.  If not specified, the global cutoff
 * specified in the pair_style command is used."
 *
 * Mixing, shift, tail, table (verbatim):
 *   "For atom type pairs I,J and I != J, the A, sigma, d1, and d2
 *   coefficients and cutoff distance for this pair style can be mixed.  A
 *   is an energy value mixed like a LJ epsilon.  D1 and d2 are distance
 *   values and are mixed like sigma.  The default mix value is
 *   *geometric*\ .  See the "pair_modify" command for details."
 *   "This pair style supports the :doc:`pair_modify <pair_modify>` shift
 *   option for the energy of the pair interaction."
 *   "The :doc:`pair_modify <pair_modify>` table option is not relevant
 *   for this pair style."
 *   "This pair style does not support the :doc:`pair_modify <pair_modify>`
 *   tail option for adding long-range tail corrections to energy and
 *   pressure."
 *
 * Default (verbatim):
 *   none
 *
 * Forces are the analytic derivatives of the three documented formulas:
 * - colloid-colloid: with q1 = r^2-(a1+a2)^2, q2 = r^2-(a1-a2)^2,
 *   dU_A/dr = -(A/3) r [1/q1 - 1/q2 - 2a1a2(1/q1^2 + 1/q2^2)]; each U_R
 *   term has the form T(r) = (r^2 + p r + q)/(r+s)^7 with
 *   T'(r) = [-5 r^2 + (2s - 6p) r + p s - 7 q]/(r+s)^8, and
 *   U_R = K S(r)/r with K = A sigma^6/37800 and S = T1+T2-T3-T4, so
 *   fpair_R = -dU_R/dr/r = K (S/r^3 - S'/r^2).
 * - colloid-solvent: with D = a^2 - r^2 (so (a-r)^6(a+r)^6 = D^6),
 *   S = 5a^6+45a^4r^2+63a^2r^4+15r^6 (S' = 90a^4r+252a^2r^3+90r^5),
 *   C = 2a^3 sigma^3 A/9:  U = C D^-3 (1 - S sigma^6/(15 D^6))
 *   = C D^-3 - (C sigma^6/15) S D^-9, and
 *   fpair = -dU/dr/r = -6C D^-4 + (C sigma^6/15)(S'/r D^-9 + 18 S D^-10).
 * - solvent-solvent: U = (A/36)[(sigma/r)^12 - (sigma/r)^6],
 *   fpair = (A/36)(12 sigma^12/r^14 - 6 sigma^6/r^8).
 *
 * Neighbor scheme: the doc's formula is pairwise, but this engine runs the
 * style on a full neighbor list with newton pairing (manybody contract):
 * every physical pair is visited once from each end and each visit adds
 * half the energy and half the force, like the two-body term of pair sw.
 */

/** (r^2 + p*r + q)/(r+s)^7 and its derivative [-5r^2+(2s-6p)r+ps-7q]/(r+s)^8. */
const uRTerm = (r: number, p: number, q: number, s: number): [number, number] => {
  const u = r + s;
  const u7 = u * u * u * u * u * u * u;
  const t = (r * r + p * r + q) / u7;
  const dt = (-5 * r * r + (2 * s - 6 * p) * r + p * s - 7 * q) / (u7 * u);
  return [t, dt];
};

export class PairColloid extends Pair {
  readonly name = 'colloid';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  /** Per type pair (both orders): A, sigma, d1, d2 and the shift offset. */
  private pA = new Float64Array(0);
  private pSig = new Float64Array(0);
  private pD1 = new Float64Array(0);
  private pD2 = new Float64Array(0);
  private pOff = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style colloid cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['a', 'sigma', 'd1', 'd2', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 6 || args.length > 7) throw new StyleError('usage: pair_coeff I J A sigma d1 d2 [cutoff]');
    const a = parseNum(args[2], 'A');
    const sig = parseNum(args[3], 'sigma');
    const d1 = parseNum(args[4], 'd1');
    const d2 = parseNum(args[5], 'd2');
    // "Both d1 and d2 must be values >= 0"
    if (!(d1 >= 0) || !(d2 >= 0)) throw new StyleError('d1 and d2 must be values >= 0');
    const cut = args[6] !== undefined ? parseNum(args[6], 'cutoff') : this.cutGlobal;
    if (!(cut > 0)) throw new StyleError('cutoff must be > 0');
    this.p.setRange(args[0], args[1], [a, sig, d1, d2, cut]);
  }

  initStyle(_ctx: StyleContext): void {
    // "This pair style does not support the pair_modify tail option for adding
    // long-range tail corrections to energy and pressure."
    if (this.tail) throw new StyleError('pair_modify tail is not supported for pair style colloid');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      // "A is an energy value mixed like a LJ epsilon.  D1 and d2 are distance
      // values and are mixed like sigma." The cutoff "is mixed the same way as
      // sigma" (pair_modify.html), like sigma itself.
      const a = mixEpsilon(this.mix, p.get('a', i, i), p.get('a', j, j), p.get('sigma', i, i), p.get('sigma', j, j));
      const sig = mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j));
      const d1 = mixDistance(this.mix, p.get('d1', i, i), p.get('d1', j, j));
      const d2 = mixDistance(this.mix, p.get('d2', i, i), p.get('d2', j, j));
      const cut = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'a', a);
      p.setMixed(i, j, 'sigma', sig);
      p.setMixed(i, j, 'd1', d1);
      p.setMixed(i, j, 'd2', d2);
      p.setMixed(i, j, 'cut', cut);
    }
    const nt = this.ntypes + 1;
    if (this.pA.length !== nt * nt) {
      this.pA = new Float64Array(nt * nt);
      this.pSig = new Float64Array(nt * nt);
      this.pD1 = new Float64Array(nt * nt);
      this.pD2 = new Float64Array(nt * nt);
      this.pOff = new Float64Array(nt * nt);
    }
    const a = p.get('a', i, j), sig = p.get('sigma', i, j), d1 = p.get('d1', i, j), d2 = p.get('d2', i, j);
    const cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    this.pA[k1] = this.pA[k2] = a;
    this.pSig[k1] = this.pSig[k2] = sig;
    this.pD1[k1] = this.pD1[k2] = d1;
    this.pD2[k1] = this.pD2[k2] = d2;
    // pair_modify shift: "adds an energy term to each pairwise interaction
    // which will be included in the thermodynamic output, but does not affect
    // pair forces" (pair_modify.html)
    this.pOff[k1] = this.pOff[k2] = this.shift ? this.pairEnergy(a, sig, d1, d2, cut, cut * cut) : 0;
    return cut;
  }

  /** Energy of one pair of the given type pair at separation r (rsq = r^2). */
  private pairEnergy(a: number, sig: number, d1: number, d2: number, r: number, rsq: number): number {
    if (d1 > 0 && d2 > 0) return this.energyCC(a, sig, 0.5 * d1, 0.5 * d2, r, rsq);
    if (d1 > 0 || d2 > 0) return this.energyCS(a, sig, 0.5 * Math.max(d1, d2), r, rsq);
    return this.energySS(a, sig, rsq);
  }

  /** Radial force coefficient fpair = -dU/dr / r (force on i = fpair * (x_i - x_j)). */
  private pairForce(a: number, sig: number, d1: number, d2: number, r: number, rsq: number): number {
    if (d1 > 0 && d2 > 0) return this.forceCC(a, sig, 0.5 * d1, 0.5 * d2, r, rsq);
    if (d1 > 0 || d2 > 0) return this.forceCS(a, sig, 0.5 * Math.max(d1, d2), r, rsq);
    return this.forceSS(a, sig, rsq);
  }

  /** Colloid-colloid: U_A + U_R with radii a1 = d1/2, a2 = d2/2. */
  private energyCC(A: number, sig: number, a1: number, a2: number, r: number, rsq: number): number {
    const dp = a1 + a2, dm = a1 - a2;
    const q1 = rsq - dp * dp, q2 = rsq - dm * dm;
    const twoA1A2 = 2 * a1 * a2;
    const ua = -(A / 6) * (twoA1A2 / q1 + twoA1A2 / q2 + Math.log(q1 / q2));
    const kp = a1 * a1 + 7 * a1 * a2 + a2 * a2;
    const km = a1 * a1 - 7 * a1 * a2 + a2 * a2;
    const T1 = uRTerm(r, -7 * dp, 6 * kp, -dp)[0];
    const T2 = uRTerm(r, 7 * dp, 6 * kp, dp)[0];
    const T3 = uRTerm(r, 7 * dm, 6 * km, dm)[0];
    const T4 = uRTerm(r, -7 * dm, 6 * km, -dm)[0];
    const s6 = sig * sig * sig * sig * sig * sig;
    const ur = ((A / 37800) * s6 / r) * (T1 + T2 - T3 - T4);
    return ua + ur;
  }

  /** Colloid-colloid radial force coefficient. */
  private forceCC(A: number, sig: number, a1: number, a2: number, r: number, rsq: number): number {
    const dp = a1 + a2, dm = a1 - a2;
    const q1 = rsq - dp * dp, q2 = rsq - dm * dm;
    const twoA1A2 = 2 * a1 * a2;
    const fa = (A / 3) * (1 / q1 - 1 / q2 - twoA1A2 * (1 / (q1 * q1) + 1 / (q2 * q2)));
    const kp = a1 * a1 + 7 * a1 * a2 + a2 * a2;
    const km = a1 * a1 - 7 * a1 * a2 + a2 * a2;
    const T1 = uRTerm(r, -7 * dp, 6 * kp, -dp);
    const T2 = uRTerm(r, 7 * dp, 6 * kp, dp);
    const T3 = uRTerm(r, 7 * dm, 6 * km, dm);
    const T4 = uRTerm(r, -7 * dm, 6 * km, -dm);
    const s = T1[0] + T2[0] - T3[0] - T4[0];
    const sp = T1[1] + T2[1] - T3[1] - T4[1];
    const s6 = sig * sig * sig * sig * sig * sig;
    const k = (A / 37800) * s6;
    // U_R = k*S/r  =>  fpair_R = -dU_R/dr/r = k*(S/r^3 - S'/r^2)
    return fa + (k / (r * r * r)) * (s - r * sp);
  }

  /** Colloid-solvent: one diameter zero, a = half the other diameter. */
  private energyCS(A: number, sig: number, a: number, r: number, rsq: number): number {
    const d = a * a - rsq;
    const d3 = d * d * d;
    const d6 = d3 * d3;
    const s3 = sig * sig * sig;
    const c = (2 * a * a * a * s3 * A) / 9;
    const a2 = a * a;
    const rsq2 = rsq * rsq;
    const sSum = 5 * a2 * a2 * a2 + 45 * a2 * a2 * rsq + 63 * a2 * rsq2 + 15 * rsq2 * rsq;
    const inner = 1 - (sSum * s3 * s3) / (15 * d6);
    return (c / d3) * inner;
  }

  /** Colloid-solvent radial force coefficient. */
  private forceCS(A: number, sig: number, a: number, r: number, rsq: number): number {
    const d = a * a - rsq;
    const d3 = d * d * d;
    const d4 = d3 * d;
    const d6 = d3 * d3;
    const d9 = d6 * d3;
    const d10 = d9 * d;
    const s3 = sig * sig * sig;
    const s6 = s3 * s3;
    const c = (2 * a * a * a * s3 * A) / 9;
    const a2 = a * a;
    const rsq2 = rsq * rsq;
    const s = 5 * a2 * a2 * a2 + 45 * a2 * a2 * rsq + 63 * a2 * rsq2 + 15 * rsq2 * rsq;
    const spOverR = 90 * a2 * a2 + 252 * a2 * rsq + 90 * rsq2;
    // fpair = -dU/dr/r = -6C D^-4 + (C sigma^6/15)(S'/r D^-9 + 18 S D^-10)
    return -6 * c / d4 + (c * s6 / 15) * (spOverR / d9 + 18 * s / d10);
  }

  /** Solvent-solvent: the Lennard-Jones form scaled by A/36. */
  private energySS(A: number, sig: number, rsq: number): number {
    const r2inv = 1 / rsq;
    const s3 = sig * sig * sig;
    const s6 = s3 * s3;
    const s12 = s6 * s6;
    return (A / 36) * (s12 * r2inv * r2inv * r2inv * r2inv * r2inv * r2inv - s6 * r2inv * r2inv * r2inv);
  }

  /** Solvent-solvent radial force coefficient. */
  private forceSS(A: number, sig: number, rsq: number): number {
    const r2inv = 1 / rsq;
    const s3 = sig * sig * sig;
    const s6 = s3 * s3;
    const s12 = s6 * s6;
    const r2inv2 = r2inv * r2inv;
    const r2inv4 = r2inv2 * r2inv2;
    return (A / 36) * r2inv4 * (12 * s12 * r2inv * r2inv * r2inv - 6 * s6);
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, pA, pSig, pD1, pD2, pOff } = this;
    const eatom = pc.eatom;
    const vatom = pc.vatom;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        const A = pA[t], sig = pSig[t], d1 = pD1[t], d2 = pD2[t];
        // every physical pair is visited from both ends: half per visit
        const e = 0.5 * (this.pairEnergy(A, sig, d1, d2, r, rsq) - pOff[t]);
        const fpair = 0.5 * this.pairForce(A, sig, d1, d2, r, rsq);
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        evdwl += e;
        if (eatom) {
          const e4 = 0.5 * e;
          eatom[i] += e4;
          eatom[j] += e4;
        }
        if (vatom) {
          const va = vatom;
          const vc = 0.5 * fpair;
          const v0 = vc * dx * dx, v1 = vc * dy * dy, v2 = vc * dz * dz;
          const v3 = vc * dx * dy, v4 = vc * dx * dz, v5 = vc * dy * dz;
          va[6 * i] += v0; va[6 * i + 1] += v1; va[6 * i + 2] += v2;
          va[6 * i + 3] += v3; va[6 * i + 4] += v4; va[6 * i + 5] += v5;
          va[6 * j] += v0; va[6 * j + 1] += v1; va[6 * j + 2] += v2;
          va[6 * j + 3] += v3; va[6 * j + 4] += v4; va[6 * j + 5] += v5;
        }
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number): { eng: number; fforce: number } {
    const nt = this.ntypes + 1;
    const t = itype * nt + jtype;
    if (rsq >= this.cutsq[t]) return { eng: 0, fforce: 0 };
    const r = Math.sqrt(rsq);
    const A = this.pA[t], sig = this.pSig[t], d1 = this.pD1[t], d2 = this.pD2[t];
    return {
      eng: this.pairEnergy(A, sig, d1, d2, r, rsq) - this.pOff[t],
      fforce: this.pairForce(A, sig, d1, d2, r, rsq),
    };
  }
}
