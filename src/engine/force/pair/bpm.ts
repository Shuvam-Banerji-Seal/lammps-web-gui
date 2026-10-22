import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum, fmtCoeff } from '../util';
import { tallyAtom } from './lj_cut';

/*
 * pair_style bpm/spring — docs.lammps.org/pair_bpm_spring.html (texts: plans/lammps-docs-live/pair_bpm_spring.rst).
 * The doc says: "computes pairwise forces with the formula" F = k (r - r_c) + k_a (r - r_c)^3
 * and "An additional damping force is also applied to interacting particles", F_D = -gamma w (r_hat . v) with
 * w = 1 - (r/r_c)^8. Its energy proxy is "k (r - r_c)^2 / 2 + k_a (r - r_c)^4 / 4".
 *
 * Measured with native LAMMPS (black box), two unbonded atoms in lj units with thermo norm no: at r = 0.8, 0.4
 * (r_c = 1, k = 1, gamma = 0.5, k_a = 2) the energies are 0.0208 and 0.2448 and the force magnitudes 0.216 and
 * 1.032, i.e. E = k (r_c - r)^2/2 + k_a (r_c - r)^4/4 and F = k (r_c - r) + k_a (r_c - r)^3 with no smoothing on
 * the spring term; beyond r_c the pair has no interaction. Bonded pairs with special_bonds lj 0 are excluded
 * (the special factor multiplies both energy and force): a bonded pair at r = 0.8 gives zero pair energy.
 * Mixing: the doc says "A is always mixed via a geometric rule" and the cutoff follows pair_modify mix. The
 * gamma and k_a mixing rules are not in the doc, so an unset I,J pair is refused.
 * Requires comm_modify vel yes (ghost velocities for the damping term), as native does.
 */

export class PairBpmSpring extends Pair {
  readonly name = 'bpm/spring';
  virialFdotr = true;
  anharmonic = false;
  p!: PairParams;
  private kk = new Float64Array(0);
  private rc = new Float64Array(0);
  private gg = new Float64Array(0);
  private ka = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    for (let k = 0; k < args.length; k += 2) {
      if (args[k] !== 'anharmonic') throw new StyleError(`illegal pair_style bpm/spring keyword '${args[k]}'`);
      if (args[k + 1] === 'yes') this.anharmonic = true;
      else if (args[k + 1] === 'no') this.anharmonic = false;
      else throw new StyleError(`pair_style bpm/spring anharmonic must be yes or no (got '${args[k + 1]}')`);
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['k', 'rc', 'gamma', 'ka']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    const need = this.anharmonic ? 6 : 5;
    if (args.length !== need) throw pairCoeffArgs(`usage: pair_coeff I J k r_c gamma${this.anharmonic ? ' k_a' : ''}`);
    const vals = args.slice(2).map((w, q) => parseNum(w, ['k', 'r_c', 'gamma', 'k_a'][q]));
    if (!(vals[1] > 0)) throw new StyleError('pair bpm/spring: r_c must be > 0');
    const kv = this.anharmonic ? vals : [vals[0], vals[1], vals[2], 0];
    this.p.setRange(args[0], args[1], kv);
  }

  initStyle(ctx: StyleContext): void {
    if (this.tail) throw new StyleError('pair_modify tail is not supported by pair style bpm/spring');
    if (ctx.ghostVelocity !== true) throw new StyleError('Pair bpm/spring requires ghost atoms store velocity (use comm_modify vel yes)');
    const nt = this.ntypes + 1;
    this.kk = new Float64Array(nt * nt); this.rc = new Float64Array(nt * nt);
    this.gg = new Float64Array(nt * nt); this.ka = new Float64Array(nt * nt);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
      throw new StyleError(`pair bpm/spring: pair ${i} ${j} is not set and its gamma and k_a cannot be mixed (not in the doc); set it with pair_coeff`);
    }
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.kk[k1] = this.kk[k2] = p.get('k', i, j);
    this.rc[k1] = this.rc[k2] = p.get('rc', i, j);
    this.gg[k1] = this.gg[k2] = p.get('gamma', i, j);
    this.ka[k1] = this.ka[k2] = p.get('ka', i, j);
    return p.get('rc', i, j);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const s = pc.s;
    const vel = s.v;
    const owner = pc.nb.owner;
    const nt = this.ntypes + 1;
    const { cutsq, kk, rc, gg, ka } = this;
    const sLJ = pc.specialLJ;
    const anh = this.anharmonic;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const oi = owner[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factor = sLJ[jj >>> SBBITS];
        const r = Math.sqrt(rsq);
        const dr = r - rc[t];
        // conservative part: -dE/dr = -(k dr + k_a dr^3), along r_hat = d / r
        const kdr = kk[t] * dr;
        const cube = anh ? ka[t] * dr * dr * dr : 0;
        const fcons = (-(kdr + cube) * factor) / r;
        // damping on i: -gamma w (r_hat . v_ij) r_hat with w = 1 - (r/r_c)^8
        const oj = owner[j];
        const dvx = vel[3 * oi] - vel[3 * oj], dvy = vel[3 * oi + 1] - vel[3 * oj + 1], dvz = vel[3 * oi + 2] - vel[3 * oj + 2];
        const w = 1 - Math.pow(r / rc[t], 8);
        const fdamp = (-gg[t] * w * (dx * dvx + dy * dvy + dz * dvz) * factor) / rsq;
        const fpair = fcons + fdamp;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * (0.5 * kk[t] * dr * dr + 0.25 * (anh ? ka[t] * dr ** 4 : 0));
        evdwl += e;
        // per-atom energy and virial, as native LAMMPS does (src/BPM/pair_bpm_spring.cpp compute at b332f28: ev_tally)
        if (pc.eatom || pc.vatom) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('k', i, i))} ${fmtCoeff(this.p.get('rc', i, i))} ${fmtCoeff(this.p.get('gamma', i, i))}`);
    }
    return out;
  }
}

