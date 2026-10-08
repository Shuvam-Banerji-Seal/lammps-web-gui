import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { fmtCoeff, parseNum } from '../util';
import { tallyAtom } from './lj_cut';

/*
 * pair_style harmonic/cut — docs.lammps.org/pair_harmonic_cut.html:
 *   E = k (r_c - r)^2   \qquad r < r_c
 * "where :math:`r_c` is the cutoff.  Note that the usual 1/2 factor is included in :math:`k`."
 * The pair_coeff arguments are k (energy/distance^2 units) and r_c (distance
 * units); the pair_style command takes no arguments. The style is repulsive
 * only: for r < r_c the force is 2 k (r_c - r), the negative derivative of E.
 * The k and r_c coefficients of I != J pairs "can be mixed. The default mix value
 * is *geometric*." "Since the potential is zero at
 * and beyond the cutoff parameter by construction, there is no need to support
 * the pair_modify shift or tail options for the energy and pressure of the
 * pair interaction." — native accepts shift/tail and ignores them (only the
 * generic shift+tail and 2d+tail errors apply), and this style never reads
 * this.shift/this.tail.
 * Measured with native LAMMPS (black box): pair_style takes zero arguments
 * (an extra argument is an illegal pair_style command), pair_coeff needs
 * exactly I J k r_c, negative k and negative r_c are accepted, and the
 * Pair Coeffs section of write_data holds the diagonal values. Measured with native
 * LAMMPS (black box): with pair_modify mix arithmetic the cross k is the
 * geometric mean sqrt(k_i k_j) while r_c is the arithmetic mean; with mix
 * sixthpower k follows the sixthpower energy rule and r_c the sixthpower
 * distance rule — exactly mixEpsilon/mixDistance, which are used here.
 */
export class PairHarmonicCut extends Pair {
  readonly name: string = 'harmonic/cut';
  virialFdotr = true;
  p!: PairParams;
  /** Per type pair (both orders) after mixing/init. */
  hk = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('usage: pair_style harmonic/cut');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['k', 'cut']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length !== 4) throw new StyleError('usage: pair_coeff I J k rc');
    const k = parseNum(args[2], 'k');
    const rc = parseNum(args[3], 'rc');
    this.p.setRange(args[0], args[1], [k, rc]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      const k = mixEpsilon(this.mix, p.get('k', i, i), p.get('k', j, j), p.get('cut', i, i), p.get('cut', j, j));
      const c = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'k', k);
      p.setMixed(i, j, 'cut', c);
    }
    const nt = this.ntypes + 1;
    if (this.hk.length !== nt * nt) this.hk = new Float64Array(nt * nt);
    const k = p.get('k', i, j);
    this.hk[i * nt + j] = this.hk[j * nt + i] = k;
    return p.get('cut', i, j);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, cut, hk } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factor = sLJ[jj >>> SBBITS];
        const r = Math.sqrt(rsq);
        const dr = cut[t] - r;
        // F = 2 k (r_c - r) * rhat = fpair * (ri - rj), fpair = 2 k dr / r.
        const fpair = factor * 2 * hk[t] * dr / r;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * hk[t] * dr * dr;
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('k', i, i))} ${fmtCoeff(this.p.get('cut', i, i))}`);
    }
    return out;
  }
}
