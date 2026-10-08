import { StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { PairBorn, PairBuck, } from './simple';
import { tallyAtom } from './lj_cut';
import { erfcFast, erfcPoly, EWALD_F } from '../erfc';
import { fmtCoeff, parseNum } from '../util';

/*
 * born/coul/long and buck/coul/long — the Born-Mayer-Huggins and Buckingham
 * potentials plus the Ewald/PPPM real-space Coulomb sum, written from the
 * LAMMPS documentation only. The shared loop shape is PairLJCutCoulLong
 * (docs.lammps.org/pair_lj_cut_coul.html): non-Coulomb term with its own
 * per-pair cutoff, damped Coulomb term coulLongPair with the global Coulomb
 * cutoff, special weights sLJ for the short-range part and sC for Coulomb.
 *
 * docs.lammps.org/pair_born.html — the potential:
 *   E = A \exp \left(\frac{\sigma - r}{\rho} \right) -
 *   \frac{C}{r^6} + \frac{D}{r^8} \qquad r < r_c
 * "where :math:`\sigma` is an interaction-dependent length parameter,
 * :math:`\rho` is an ionic-pair dependent length parameter, and
 * :math:`r_c` is the cutoff." pair_style arguments (born/coul/long):
 *      *born/coul/long* args = cutoff (cutoff2)
 *        cutoff = global cutoff for non-Coulombic (and Coulombic if only 1 arg) (distance units)
 *        cutoff2 = global cutoff for Coulombic (optional) (distance units)
 * coefficients:
 * * A (energy units)
 * * :math:`\rho` (distance units)
 * * :math:`\sigma` (distance units)
 * * C (energy units \* distance units\^6)
 * * D (energy units \* distance units\^8)
 * * cutoff (distance units)
 * "The second coefficient, rho, must be greater than zero." "The last
 * coefficient is optional.  If not specified, the global A,C,D cutoff
 * specified in the pair_style command is used." "For *born/coul/long*,
 * *born/coul/wolf* and *born/coul/dsf* no Coulombic cutoff can be specified
 * for an individual I,J type pair. All type pairs use the same global
 * Coulombic cutoff specified in the pair_style command."
 *
 * docs.lammps.org/pair_buck.html — the potential:
 *   E = A e^{-r / \rho} - \frac{C}{r^6} \qquad r < r_c
 * "where :math:`\rho` is an ionic-pair dependent length parameter, and
 * :math:`r_c` is the cutoff on both terms." pair_style arguments
 * (buck/coul/long):
 *      *buck/coul/long* args = cutoff (cutoff2)
 *        cutoff = global cutoff for Buckingham (and Coulombic if only 1 arg) (distance units)
 *        cutoff2 = global cutoff for Coulombic (optional) (distance units)
 * coefficients:
 * * A (energy units)
 * * :math:`\rho` (distance units)
 * * C (energy-distance\^6 units)
 * * cutoff (distance units)
 * * cutoff2 (distance units)
 * "The second coefficient, :math:`\rho`, must be greater than zero." "The
 * latter 2 coefficients are optional.  If not specified, the global A,C and
 * Coulombic cutoffs are used." (the optional per-pair cutoff2 exists only for
 * buck/coul/cut) "For *buck/coul/long* only the LJ cutoff can be specified
 * since a Coulombic cutoff cannot be specified for an individual I,J type
 * pair. All type pairs use the same global Coulombic cutoff specified in the
 * pair_style command."
 *
 * Both pages: "These pair styles do not support mixing.  Thus, coefficients
 * for all I,J pairs must be specified explicitly." Born: "These styles
 * support the :doc:`pair_modify <pair_modify>` shift option for the energy
 * of the exp(), 1/r\^6, and 1/r\^8 portion of the pair interaction."
 * Buckingham: "These styles support the :doc:`pair_modify <pair_modify>`
 * shift option for the energy of the exp() and 1/r\^6 portion of the pair
 * interaction." Both: "These styles support the pair_modify tail option for
 * adding long-range tail corrections to energy and pressure for the A,C
 * terms in the pair interaction" and the coul/long table option. The tail
 * integrals are the Sun form of docs.lammps.org/pair_modify.html, evaluated
 * by the base classes — with the A,C cutoff, which tailSums below keeps
 * distinct from the combined pair cutoff max(A,C, Coulomb).
 * The damped Coulomb term is documented on docs.lammps.org/pair_coul.html
 * ("an additional damping factor is applied so it can be used in conjunction
 * with the kspace_style command") as E = C q_i q_j erfc(g r)/r with
 * C = qqr2e/dielectric and g the kspace G-ewald parameter; see coul_long.ts
 * for the erfc variants and the special_bonds subtraction.
 */

/** Per-pair non-Coulomb cutoff: NaN means "use the global pair_style cutoff". */
const nonCoulCut = (raw: number, cutGlobal: number): number => (Number.isNaN(raw) ? cutGlobal : raw);

/**
 * Damped real-space Coulomb pair term, identical to the coulLongPair helper of
 * coul_long.ts (which is module-local there): E = C q_i q_j erfc(g r)/r with
 * C = qqrd2e, and force magnitude -dE/dr/r = C q_i q_j (erfc(g r)
 * + 2/sqrt(pi) g r e^{-g^2 r^2}) / r^2 (docs.lammps.org/pair_coul.html).
 * The special-bond weight fc < 1 removes (1 - fc) of the bare C q_i q_j / r
 * term, which the reciprocal sum includes (docs.lammps.org/special_bonds.html).
 */
const coulLongPair = (
  rsq: number, qi: number, qj: number, g: number, qqrd2e: number, fc: number, poly: boolean,
): { e: number; f: number } => {
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const ex = Math.exp(-grij * grij);
  const erfc = poly ? erfcPoly(grij, ex) : erfcFast(grij, ex);
  const pre = qqrd2e * qi * qj / r;
  let forcecoul = pre * (erfc + EWALD_F * grij * ex);
  let e = pre * erfc;
  if (fc < 1) {
    forcecoul -= (1 - fc) * pre;
    e -= (1 - fc) * pre;
  }
  return { e, f: forcecoul / rsq };
};

export class PairBornCoulLong extends PairBorn {
  readonly name: string = 'born/coul/long';
  coulLong = true;
  cutCoul = 0;
  /** Coulomb prefactor of the current run (pc.qqrd2e); single() uses the last-seen value. */
  qqrd2e = 1;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1 && args.length !== 2) throw new StyleError('usage: pair_style born/coul/long cutoff (cutoff2)');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = args.length === 2 ? parseNum(args[1], 'Coulomb cutoff') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  initOne(i: number, j: number): number {
    const cutAC = super.initOne(i, j);
    return Math.max(cutAC, this.cutCoul);
  }

  compute(pc: PairCompute): void {
    this.qqrd2e = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { bornA, bornIR, bornSig, bornC, bornD, offset } = this;
    const bornCutSq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) bornCutSq[i * nt + j] = nonCoulCut(this.p.get('cut', i, j), this.cutGlobal) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        let fpair = 0, e = 0;
        if (rsq < bornCutSq[t]) {
          const factor = sLJ[sb];
          const r = Math.sqrt(rsq);
          const rinv = 1 / r;
          const ex = Math.exp((bornSig[t] - r) * bornIR[t]);
          const r2inv = 1 / rsq;
          const r6inv = r2inv * r2inv * r2inv;
          const r8inv = r6inv * r2inv;
          // F_vec = [(A/rho) e^{(sigma-r)/rho} - 6 C/r^7 + 8 D/r^9] * rhat
          //       = fpair * (ri - rj) with fpair = (A/rho) e/r - 6 C/r^8 + 8 D/r^10
          fpair += factor * (bornA[t] * bornIR[t] * ex * rinv - 6 * bornC[t] * r8inv + 8 * bornD[t] * r8inv * r2inv);
          const ev = factor * (bornA[t] * ex - bornC[t] * r6inv + bornD[t] * r8inv - offset[t]);
          evdwl += ev;
          e += ev;
        }
        if (rsq < cutcsq && qi !== 0 && q[j] !== 0) {
          const r = coulLongPair(rsq, qi, q[j], g, pc.qqrd2e, sC[sb], this.table === 0);
          fpair += r.f;
          ecoul += r.e;
          e += r.e;
        }
        if (fpair === 0 && e === 0) continue;
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number) {
    const b = super.single(_i, _j, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
    let eng = b.eng, fforce = b.fforce;
    if (rsq < this.cutCoul * this.cutCoul && qi !== 0 && qj !== 0) {
      const r = coulLongPair(rsq, qi, qj, this.gEwald, this.qqrd2e, factorCoul, this.table === 0);
      fforce += r.f;
      eng += r.e;
    }
    return { eng, fforce };
  }

  /** PairIJ Coeffs lines carry the A,C,D cutoff (the Coulomb cutoff is global only). */
  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const cut = nonCoulCut(this.p.get('cut', i, j), this.cutGlobal);
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('A', i, j))} ${fmtCoeff(this.p.get('rho', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('C', i, j))} ${fmtCoeff(this.p.get('D', i, j))} ${fmtCoeff(cut)}`);
      }
    }
    return out;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    // Same Sun-form integrals as PairBorn (pair_modify.html) but with the
    // A,C,D cutoff, not the combined pair cutoff max(A,C,D, Coulomb).
    let e = 0, pr = 0;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = 1; j <= this.ntypes; j++) {
        const a = this.p.get('A', i, j), rho = this.p.get('rho', i, j), sig = this.p.get('sigma', i, j);
        const c = this.p.get('C', i, j), d = this.p.get('D', i, j);
        const rc = nonCoulCut(this.p.get('cut', i, j), this.cutGlobal);
        if (!(rc > 0)) continue;
        const nn = count[i] * count[j], ex = Math.exp((sig - rc) / rho);
        e += nn * (a * rho * ex * (rc * rc + 2 * rho * rc + 2 * rho * rho) - c / (3 * rc ** 3) + d / (5 * rc ** 5));
        pr += nn * (-(a / rho) * ex * (rc ** 3 * rho + 3 * rc * rc * rho * rho + 6 * rc * rho ** 3 + 6 * rho ** 4) + 2 * c / rc ** 3 - 8 * d / (5 * rc ** 5));
      }
    }
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }
}

export class PairBuckCoulLong extends PairBuck {
  readonly name: string = 'buck/coul/long';
  coulLong = true;
  cutCoul = 0;
  /** Coulomb prefactor of the current run (pc.qqrd2e); single() uses the last-seen value. */
  qqrd2e = 1;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1 && args.length !== 2) throw new StyleError('usage: pair_style buck/coul/long cutoff (cutoff2)');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = args.length === 2 ? parseNum(args[1], 'Coulomb cutoff') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  initOne(i: number, j: number): number {
    const cutAC = super.initOne(i, j);
    return Math.max(cutAC, this.cutCoul);
  }

  compute(pc: PairCompute): void {
    this.qqrd2e = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { buckA, buckIR, buckC, offset } = this;
    const buckCutSq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) buckCutSq[i * nt + j] = nonCoulCut(this.p.get('cut', i, j), this.cutGlobal) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        let fpair = 0, e = 0;
        if (rsq < buckCutSq[t]) {
          const factor = sLJ[sb];
          const r = Math.sqrt(rsq);
          const ex = Math.exp(-r * buckIR[t]);
          const r2inv = 1 / rsq;
          // F_vec = [(A/rho) e^{-r/rho} - 6 C/r^7] * rhat = fpair * (ri - rj)
          fpair += factor * (buckA[t] * buckIR[t] * ex / r - 6 * buckC[t] * r2inv * r2inv * r2inv * r2inv);
          const ev = factor * (buckA[t] * ex - buckC[t] * (r2inv * r2inv * r2inv) - offset[t]);
          evdwl += ev;
          e += ev;
        }
        if (rsq < cutcsq && qi !== 0 && q[j] !== 0) {
          const r = coulLongPair(rsq, qi, q[j], g, pc.qqrd2e, sC[sb], this.table === 0);
          fpair += r.f;
          ecoul += r.e;
          e += r.e;
        }
        if (fpair === 0 && e === 0) continue;
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number) {
    const b = super.single(_i, _j, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
    let eng = b.eng, fforce = b.fforce;
    if (rsq < this.cutCoul * this.cutCoul && qi !== 0 && qj !== 0) {
      const r = coulLongPair(rsq, qi, qj, this.gEwald, this.qqrd2e, factorCoul, this.table === 0);
      fforce += r.f;
      eng += r.e;
    }
    return { eng, fforce };
  }

  /** PairIJ Coeffs lines carry the A,C cutoff (the Coulomb cutoff is global only). */
  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const cut = nonCoulCut(this.p.get('cut', i, j), this.cutGlobal);
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('A', i, j))} ${fmtCoeff(this.p.get('rho', i, j))} ${fmtCoeff(this.p.get('C', i, j))} ${fmtCoeff(cut)}`);
      }
    }
    return out;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    // Same Sun-form integrals as PairBuck (pair_modify.html) but with the
    // A,C cutoff, not the combined pair cutoff max(A,C, Coulomb).
    let e = 0, pr = 0;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = 1; j <= this.ntypes; j++) {
        const a = this.p.get('A', i, j), rho = this.p.get('rho', i, j), c = this.p.get('C', i, j);
        const rc = nonCoulCut(this.p.get('cut', i, j), this.cutGlobal);
        if (!(rc > 0)) continue;
        const nn = count[i] * count[j], ex = Math.exp(-rc / rho);
        e += nn * (a * rho * ex * (rc * rc + 2 * rho * rc + 2 * rho * rho) - c / (3 * rc ** 3));
        pr += nn * (-(a / rho) * ex * (rc ** 3 * rho + 3 * rc * rc * rho * rho + 6 * rc * rho ** 3 + 6 * rho ** 4) + 2 * c / rc ** 3);
      }
    }
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }
}
