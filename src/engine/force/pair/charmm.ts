import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type MixRule, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { parseNum } from '../util';
import { erfcPoly, EWALD_F, ErfcTableCache, TABLE_INNER_RSQ } from '../erfc';

/*
 * CHARMM LJ + Coulomb pair styles with the energy switching function:
 * lj/charmm/coul/charmm, lj/charmm/coul/charmm/implicit, lj/charmm/coul/long.
 *
 * docs.lammps.org/pair_charmm.html (Syntax):
 *   "*lj/charmm/coul/charmm* args = inner outer (inner2) (outer2)"
 *     "inner, outer = global switching cutoffs for Lennard Jones (and Coulombic if only 2 args)"
 *     "inner2, outer2 = global switching cutoffs for Coulombic (optional)"
 *   "*lj/charmm/coul/charmm/implicit* args = inner outer (inner2) (outer2)"
 *   "*lj/charmm/coul/long* args = inner outer (cutoff)"
 *     "cutoff = global cutoff for Coulombic (optional, outer is Coulombic cutoff if only 2 args)"
 *   "In all cases where an inner and outer cutoff are specified, the inner
 *   cutoff distance must be less than the outer cutoff."
 * docs.lammps.org/pair_charmm.html (Description):
 *   "Style *lj/charmm/coul/charmm/implicit* computes the same formulas as
 *   style *lj/charmm/coul/charmm* except that an additional 1/r term is
 *   included in the Coulombic formula.  The Coulombic energy thus varies
 *   as 1/r\^2."
 *   "Styles *lj/charmm/coul/long* and *lj/charmm/coul/msm* compute the same
 *   formulas as style *lj/charmm/coul/charmm* ... except that an additional
 *   damping factor is applied to the Coulombic term, so it can be used in
 *   conjunction with the kspace_style command and its ewald or pppm or msm
 *   option.  Only one Coulombic cutoff is specified for these styles; if
 *   only 2 arguments are used in the pair_style command, then the outer LJ
 *   cutoff is used as the single Coulombic cutoff."
 *   "The latter 2 coefficients are optional.  If they are specified, they
 *   are used in the LJ formula between two atoms of these types which are
 *   also first and fourth atoms in any dihedral.  No cutoffs are specified
 *   because the CHARMM force field does not allow varying cutoffs for
 *   individual atom pairs; all pairs use the global cutoff(s) specified in
 *   the pair_style command."  (The 1-4 interaction itself is computed by
 *   dihedral_style charmm — docs.lammps.org/dihedral_charmm.html: "interactions
 *   between the first and fourth atoms in a dihedral are skipped during the
 *   normal non-bonded force computation and instead evaluated as part of the
 *   dihedral using special epsilon and sigma values specified with the
 *   pair_coeff command" — so the pair style only stores epsilon14/sigma14.)
 * docs.lammps.org/pair_charmm.html (Mixing ... info):
 *   "For atom type pairs I,J and I != J, the epsilon, sigma, epsilon_14,
 *   and sigma_14 coefficients for all of the lj/charmm pair styles can be
 *   mixed.  The default mix value is *arithmetic* to coincide with the
 *   usual settings for the CHARMM force field."
 *   "None of the *lj/charmm* or *lj/charmmfsw* pair styles support the
 *   :doc:`pair_modify <pair_modify>` shift option, since the Lennard-Jones
 *   portion of the pair interaction is smoothed to 0.0 at the cutoff."
 *   "None of the *lj/charmm* or *lj/charmmfsw* pair styles support the
 *   :doc:`pair_modify <pair_modify>` tail option for adding long-range tail
 *   corrections to energy and pressure, since the Lennard-Jones portion of
 *   the pair interaction is smoothed to 0.0 at the cutoff."
 *   "The *lj/charmm/coul/long* and *lj/charmmfsw/coul/long* styles
 *   support the :doc:`pair_modify <pair_modify>` table option since they can
 *   tabulate the short-range portion of the long-range Coulombic
 *   interaction."
 *   The Coulomb conversion factor stays the LAMMPS one (pc.qqrd2e, real units
 *   qqr2e = 332.06371): the CHARMM value 332.0716 applies to "The newest
 *   CHARMM pair styles ... when using one of these two CHARMM pair styles",
 *   i.e. the charmmfsw styles, not these older ones.
 *
 * The switching function and the switched energies, docs.lammps.org/Howto_bioFF.html
 * ("The older styles with *charmm* (not *charmmfsw* or *charmmfsh*\ ) in their
 * name compute the LJ and Coulombic interactions with an energy switching
 * function (esw) :math:`S(r)` which ramps the energy smoothly to zero between
 * the inner and outer cutoff"):
 *
 *   LJ(r) &= 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
 *            \left(\frac{\sigma}{r}\right)^6 \right]
 *   C(r) &= \frac{C q_i q_j}{ \epsilon r}
 *   S(r) &=  \frac{ \left(b^2 - r^2\right)^2 \left(b^2 + 2r^2 - 3{a^2}\right)}
 *           { \left(b^2 - a^2\right)^3 }
 *   E_{LJ}(r) &=  \begin{cases}
 *     LJ(r), & r \leq a \\
 *     LJ(r) S(r), & a < r \leq b \\
 *     0, &r > b
 *   \end{cases}
 *   E_{coul}(r) &=  \begin{cases}
 *     C(r), & r \leq a \\
 *     C(r) S(r), & a < r \leq b \\
 *     0, & r > b
 *   \end{cases}
 *
 * with a the inner and b the outer cutoff.
 *
 * Force treatment (the pages above do not spell it out; measured with native
 * LAMMPS on two-atom inputs, /tmp/opencode/w3charmm, 2026-10-08):
 *  - The LJ force is the analytic derivative of the switched energy,
 *    F = -d(LJ S)/dr: with F_std = -LJ'(r)/r and
 *    S'(r) = 12 r (b^2 - r^2) (a^2 - r^2) / (b^2 - a^2)^3 (S(a) = 1, S(b) = 0),
 *    F = S F_std + 12 LJ(r) (b^2 - r^2) (r^2 - a^2) / (b^2 - a^2)^3
 *    (agrees with native to machine precision at r = 8.1 ... 9.9 Angstrom).
 *  - lj/charmm/coul/charmm applies S to the Coulomb energy AND to the
 *    standard Coulomb force: E = C(r) S(r), F = S(r) F_std with
 *    F_std = -C'(r)/r — NOT the derivative of C S (native fpair = S F_std to
 *    1e-12 at r = 8.1 ... 9.9; the analytic derivative would be ~13x larger
 *    at r = 9).
 *  - lj/charmm/coul/charmm/implicit uses the analytic derivative for the
 *    Coulomb too (native fpair = -d(C_imp S)/dr to 1e-15, E = C_imp S with
 *    C_imp = C q_i q_j / r^2).
 *  - lj/charmm/coul/long does not switch the Coulombic term at all: with
 *    kspace gewald forced to ~0 the native ecoul is the plain damped
 *    erfc(g r) C q_i q_j / r and the force the plain damped F_std, i.e. the
 *    Coulomb is treated exactly like lj/cut/coul/long (docs.lammps.org/
 *    pair_lj_cut_coul.html, coul_long.ts) with its single Coulombic cutoff.
 *    The special-bonds subtraction follows docs.lammps.org/special_bonds.html
 *    as cited in coul_long.ts: excluded pairs stay in the neighbor list and
 *    their Coulomb weight w removes (1 - w) of the bare C q_i q_j / r term,
 *    which the reciprocal sum includes.
 * pair_modify table 0 evaluates erfc with the polynomial fit (erfc.ts
 * erfcPoly), any other value uses the ~1e-12 direct evaluation, matching the
 * coul/long styles.
 */

/** S(r) = (b^2 - r^2)^2 (b^2 + 2 r^2 - 3 a^2) / (b^2 - a^2)^3 for r^2 in the switch region. */
const esw = (r2: number, a2: number, b2: number, invD3: number): number => {
  const t = b2 - r2;
  return t * t * (b2 + 2 * r2 - 3 * a2) * invD3;
};

/** Extra force coefficient from switching E = U S: -U S'(r)/r = 12 U (b^2-r^2)(r^2-a^2) / (b^2-a^2)^3. */
const eswForce = (u: number, r2: number, a2: number, b2: number, invD3: number): number =>
  12 * u * (b2 - r2) * (r2 - a2) * invD3;

export class PairLJCharmmCoulCharmm extends Pair {
  readonly name: string = 'lj/charmm/coul/charmm';
  virialFdotr = true;
  /** docs.lammps.org/pair_charmm.html: "The default mix value is *arithmetic*". */
  mix: MixRule = 'arithmetic';
  protected inner = 0;
  protected outer = 0;
  protected innerCoul = 0;
  protected outerCoul = 0;
  /** Coulomb cutoff (outer2, or outer with 2 args; the separate cutoff arg for coul/long). */
  protected cutCoul = 0;
  /** LJ switch bounds as squares, with 1/(b^2-a^2)^3. */
  protected aL2 = 0;
  protected bL2 = 0;
  protected invDL3 = 0;
  /** Coulomb switch bounds (lj/charmm/coul/charmm and /implicit only). */
  protected aC2 = 0;
  protected bC2 = 0;
  protected invDC3 = 0;
  /** lj/charmm/coul/long supports pair_modify table; the others do not. */
  protected tableSupported = false;
  p!: PairParams;
  protected lj1 = new Float64Array(0);
  protected lj2 = new Float64Array(0);
  protected lj3 = new Float64Array(0);
  protected lj4 = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 4) throw new StyleError(`usage: pair_style ${this.name} inner outer [inner2 outer2]`);
    this.inner = parseNum(args[0], 'inner cutoff');
    this.outer = parseNum(args[1], 'outer cutoff');
    if (args.length === 4) {
      this.innerCoul = parseNum(args[2], 'inner Coulomb cutoff');
      this.outerCoul = parseNum(args[3], 'outer Coulomb cutoff');
    } else {
      this.innerCoul = this.inner;
      this.outerCoul = this.outer;
    }
    this.checkCutoffs();
    this.cutCoul = this.outerCoul;
    this.aL2 = this.inner * this.inner;
    this.bL2 = this.outer * this.outer;
    this.invDL3 = 1 / (this.bL2 - this.aL2) ** 3;
    this.aC2 = this.innerCoul * this.innerCoul;
    this.bC2 = this.outerCoul * this.outerCoul;
    this.invDC3 = 1 / (this.bC2 - this.aC2) ** 3;
  }

  protected checkCutoffs(): void {
    // "In all cases where an inner and outer cutoff are specified, the inner
    // cutoff distance must be less than the outer cutoff." (pair_charmm.html)
    if (!(this.inner > 0) || !(this.outer > 0)) throw new StyleError(`${this.name}: cutoffs must be > 0`);
    if (!(this.outer > this.inner)) {
      throw new StyleError(`${this.name}: the inner cutoff (${this.inner}) must be less than the outer cutoff (${this.outer})`);
    }
    if (!(this.innerCoul > 0) || !(this.outerCoul > 0)) throw new StyleError(`${this.name}: Coulomb cutoffs must be > 0`);
    if (this.innerCoul !== this.inner || this.outerCoul !== this.outer) {
      if (!(this.outerCoul > this.innerCoul)) {
        throw new StyleError(`${this.name}: the inner Coulomb cutoff (${this.innerCoul}) must be less than the outer Coulomb cutoff (${this.outerCoul})`);
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'epsilon14', 'sigma14']);
  }

  coeff(args: string[]): void {
    // Coefficients epsilon, sigma, epsilon_14, sigma_14 (pair_charmm.html);
    // "The latter 2 coefficients are optional.";
    // "No cutoffs are specified" (pair_charmm.html).
    if (args.length !== 4 && args.length !== 6) {
      throw new StyleError(`usage: pair_coeff I J epsilon sigma [epsilon14 sigma14] for pair style ${this.name}`);
    }
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const eps14 = args.length === 6 ? parseNum(args[4], 'epsilon14') : 0;
    const sig14 = args.length === 6 ? parseNum(args[5], 'sigma14') : 0;
    this.p.setRange(args[0], args[1], [eps, sig, eps14, sig14]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      const e = mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j));
      const s = mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j));
      const e14 = mixEpsilon(this.mix, p.get('epsilon14', i, i), p.get('epsilon14', j, j), p.get('sigma14', i, i), p.get('sigma14', j, j));
      const s14 = mixDistance(this.mix, p.get('sigma14', i, i), p.get('sigma14', j, j));
      p.setMixed(i, j, 'epsilon', e);
      p.setMixed(i, j, 'sigma', s);
      p.setMixed(i, j, 'epsilon14', e14);
      p.setMixed(i, j, 'sigma14', s14);
    }
    const nt = this.ntypes + 1;
    if (this.lj1.length !== nt * nt) {
      this.lj1 = new Float64Array(nt * nt); this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt); this.lj4 = new Float64Array(nt * nt);
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    const s6 = sig ** 6, s12 = s6 * s6;
    // LJ(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] = r6inv (lj3 r6inv - lj4);
    // F_std = -LJ'(r)/r = r6inv (lj1 r6inv - lj2) r2inv.
    this.lj1[k1] = this.lj1[k2] = 48 * eps * s12;
    this.lj2[k1] = this.lj2[k2] = 24 * eps * s6;
    this.lj3[k1] = this.lj3[k2] = 4 * eps * s12;
    this.lj4[k1] = this.lj4[k2] = 4 * eps * s6;
    return this.outer > this.cutCoul ? this.outer : this.cutCoul;
  }

  init(ctx: StyleContext): void {
    if (this.shift) {
      throw new StyleError(`pair_modify shift yes is not supported for pair style ${this.name} (the Lennard-Jones portion of the pair interaction is smoothed to 0.0 at the cutoff)`);
    }
    if (this.tail) {
      throw new StyleError(`pair_modify tail yes is not supported for pair style ${this.name} (the Lennard-Jones portion of the pair interaction is smoothed to 0.0 at the cutoff)`);
    }
    if (!this.tableSupported && this.table !== 12) {
      throw new StyleError(`pair_modify table is not supported for pair style ${this.name}`);
    }
    super.init(ctx);
  }

  /** The unswitched Coulomb pair term: energy U(r) and force coefficient -U'(r)/r. */
  protected coulombPair(rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    // C(r) = C q_i q_j / r with C = qqrd2e (dielectric folded in); F_std = C q_i q_j / r^3.
    const pre = qqrd2e * qi * qj / Math.sqrt(rsq);
    return { e: pre, f: pre / rsq };
  }

  /**
   * One pair's Coulomb energy/force with the switch and the special_bonds
   * weight applied. lj/charmm/coul/charmm (measured, see the file header):
   * the switch multiplies the energy and the standard force,
   * E = C(r) S(r), F = S(r) F_std.
   */
  protected coulTerm(pc: PairCompute, rsq: number, sb: number, qi: number, qj: number): { e: number; f: number } {
    const c = this.coulombPair(rsq, qi, qj, pc.qqrd2e);
    const fc = pc.specialCoul[sb];
    if (rsq > this.aC2) {
      const s = esw(rsq, this.aC2, this.bC2, this.invDC3);
      return { e: fc * c.e * s, f: fc * c.f * s };
    }
    return { e: fc * c.e, f: fc * c.f };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, aL2, bL2, invDL3 } = this;
    const cutCoulSq = this.cutCoul * this.cutCoul;
    const sLJ = pc.specialLJ;
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
        if (rsq < bL2) {
          const factor = sLJ[sb];
          const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
          const u = r6inv * (lj3[t] * r6inv - lj4[t]);
          const fstd = r6inv * (lj1[t] * r6inv - lj2[t]) * r2inv;
          if (rsq <= aL2) {
            e = u; fpair = fstd;
          } else {
            const s = esw(rsq, aL2, bL2, invDL3);
            e = u * s;
            // LJ force = analytic derivative of the switched energy (measured).
            fpair = fstd * s + eswForce(u, rsq, aL2, bL2, invDL3);
          }
          e *= factor;
          fpair *= factor;
          evdwl += e;
        }
        if (rsq < cutCoulSq && qi !== 0 && q[j] !== 0) {
          const c = this.coulTerm(pc, rsq, sb, qi, q[j]);
          ecoul += c.e;
          e += c.e;
          fpair += c.f;
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

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${this.p.get('epsilon', i, i)} ${this.p.get('sigma', i, i)} ${this.p.get('epsilon14', i, i)} ${this.p.get('sigma14', i, i)}`);
    }
    return out;
  }

  extract(name: string): unknown {
    return name === 'cut_coul' ? this.cutCoul : undefined;
  }
}

/**
 * lj/charmm/coul/charmm/implicit — pair_charmm.html: "computes the same
 * formulas as style *lj/charmm/coul/charmm* except that an additional 1/r
 * term is included in the Coulombic formula.  The Coulombic energy thus
 * varies as 1/r\^2."  The Coulomb force is the analytic derivative of the
 * switched energy (measured, see the file header).
 */
export class PairLJCharmmCoulCharmmImplicit extends PairLJCharmmCoulCharmm {
  readonly name: string = 'lj/charmm/coul/charmm/implicit';

  protected override coulombPair(rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    // C(r) = C q_i q_j / r^2; F_std = -C'(r)/r = 2 C q_i q_j / r^4.
    const c = qqrd2e * qi * qj / rsq;
    return { e: c, f: 2 * c / rsq };
  }

  protected override coulTerm(pc: PairCompute, rsq: number, sb: number, qi: number, qj: number): { e: number; f: number } {
    const c = this.coulombPair(rsq, qi, qj, pc.qqrd2e);
    const fc = pc.specialCoul[sb];
    if (rsq > this.aC2) {
      const s = esw(rsq, this.aC2, this.bC2, this.invDC3);
      return { e: fc * c.e * s, f: fc * (c.f * s + eswForce(c.e, rsq, this.aC2, this.bC2, this.invDC3)) };
    }
    return { e: fc * c.e, f: fc * c.f };
  }
}

/**
 * lj/charmm/coul/long — pair_charmm.html: "compute the same formulas as
 * style *lj/charmm/coul/charmm*" ... "except that an additional damping
 * factor is applied to the Coulombic term, so it can be used in conjunction
 * with the :doc:`kspace_style <kspace_style>` command and its *ewald* or
 * *pppm* or *msm* option.  Only one Coulombic cutoff is specified for these
 * styles; if only 2 arguments are used in the pair_style command, then the
 * outer LJ cutoff is used as the single Coulombic cutoff."  Measured (see
 * the file header): the Coulombic term is the plain damped erfc(g r) term cut
 * off at the Coulombic cutoff with no switching (as in lj/cut/coul/long);
 * the switching function applies to the LJ term only.
 */
export class PairLJCharmmCoulLong extends PairLJCharmmCoulCharmm {
  readonly name: string = 'lj/charmm/coul/long';
  coulLong = true;
  protected override tableSupported = true;

  override settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 3) throw new StyleError('usage: pair_style lj/charmm/coul/long inner outer [Coulomb cutoff]');
    this.inner = parseNum(args[0], 'inner cutoff');
    this.outer = parseNum(args[1], 'outer cutoff');
    if (!(this.inner > 0) || !(this.outer > 0)) throw new StyleError('lj/charmm/coul/long: cutoffs must be > 0');
    if (!(this.outer > this.inner)) {
      throw new StyleError(`lj/charmm/coul/long: the inner cutoff (${this.inner}) must be less than the outer cutoff (${this.outer})`);
    }
    // "outer is Coulombic cutoff if only 2 args"; the Coulomb has no switching
    // region of its own, so innerCoul/outerCoul stay unused.
    this.cutCoul = args.length === 3 ? parseNum(args[2], 'Coulomb cutoff') : this.outer;
    if (!(this.cutCoul > 0)) throw new StyleError('lj/charmm/coul/long: Coulomb cutoff must be > 0');
    this.innerCoul = this.outerCoul = this.cutCoul;
    this.aL2 = this.inner * this.inner;
    this.bL2 = this.outer * this.outer;
    this.invDL3 = 1 / (this.bL2 - this.aL2) ** 3;
  }

  private readonly erfcTables = new ErfcTableCache();

  protected override coulombPair(rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    // Ewald real-space damped term: U = C q_i q_j erfc(g r)/r,
    // F_std = C q_i q_j (erfc(g r) + EWALD_F g r exp(-g^2 r^2)) / r^3 (coul_long.ts). pair_modify table N
    // > 0: native's tables above r^2 = 2 (erfc.ts makeErfcTable), the polynomial below; measured with
    // native LAMMPS (black box): table 0 and 12 agree at r = 1.2 and differ at 3.1 and 4.7
    const tab = this.erfcTables.get(this.table, this.gEwald, this.cutCoul * this.cutCoul);
    if (tab && rsq >= TABLE_INNER_RSQ) {
      const qq = qqrd2e * qi * qj;
      return { e: qq * tab.energy(rsq), f: qq * tab.force(rsq) / rsq };
    }
    const r = Math.sqrt(rsq);
    const grij = this.gEwald * r;
    const ex = Math.exp(-grij * grij);
    const erfc = erfcPoly(grij, ex);
    const pre = qqrd2e * qi * qj / r;
    return { e: pre * erfc, f: pre * (erfc + EWALD_F * grij * ex) / rsq };
  }

  protected override coulTerm(pc: PairCompute, rsq: number, sb: number, qi: number, qj: number): { e: number; f: number } {
    const c = this.coulombPair(rsq, qi, qj, pc.qqrd2e);
    // special_bonds.html, as cited in coul_long.ts: excluded pairs stay in the
    // neighbor list for kspace styles; their Coulomb weight w removes (1 - w)
    // of the bare C q_i q_j / r term, which the reciprocal sum includes (from the 1/r table when
    // the table is in use, as native does).
    const fc = pc.specialCoul[sb];
    if (fc < 1) {
      const tab = this.erfcTables.get(this.table, this.gEwald, this.cutCoul * this.cutCoul);
      const inv = tab && rsq >= TABLE_INNER_RSQ ? tab.coul(rsq) : 1 / Math.sqrt(rsq);
      const bare = pc.qqrd2e * qi * qj * inv;
      return { e: c.e - (1 - fc) * bare, f: c.f - (1 - fc) * bare / rsq };
    }
    return c;
  }
}

/**
 * Force-switched CHARMM styles lj/charmmfsw/coul/charmmfsh and
 * lj/charmmfsw/coul/long (docs.lammps.org/pair_charmm.html; the energy forms
 * are in docs.lammps.org/Howto_bioFF.html).
 *
 * pair_charmm.html (Description): "The newer styles with *charmmfsw* or
 * *charmmfsh* in their name replace the energy switching with force switching
 * (fsw) and force shifting (fsh) functions, for LJ and Coulombic interactions
 * respectively."
 * pair_charmm.html (Description): "For the *lj/charmmfsw/coul/charmmfsh* style,
 * the LJ term requires both an inner and outer cutoff, while the Coulombic term
 * requires only one cutoff.  If the Coulombic cutoff is not specified (2
 * instead of 3 arguments), the LJ outer cutoff is used for the Coulombic cutoff."
 * pair_charmm.html (Description): "*lj/charmmfsw/coul/long* computes the same
 * formulas as style *lj/charmmfsw/coul/charmmfsh*, except that an additional
 * damping factor is applied to the Coulombic term".
 * pair_charmm.html (Description): "The newest CHARMM pair styles reset the
 * Coulombic energy conversion factor used internally in the code, from the
 * LAMMPS value to the CHARMM value ... CHARMM = 332.0716, LAMMPS = 332.06371."
 * Howto_bioFF.rst gives the force-switched LJ energy as
 *   E_LJ(r) = 4 eps sigma^6 ( (sigma^6-r^6)/r^12 - sigma^6/(a^6 b^6) + 1/(a^3 b^3) ), r <= a
 *   E_LJ(r) = 4 eps sigma^6 ( sigma^6 (b^6-r^6)^2 - b^3 r^6 (a^3+b^3) (b^3-r^3)^2 )
 *             / ( b^6 r^12 (b^6-a^6) ),                                           a < r <= b
 * and the Coulombic energy as E_coul(r) = C(r) (b-r)^2 / (r b^2), r <= b (its
 * "C(r) \frac{\displaystyle (b-r)^2}{\displaystyle r b^2}, &  r \leq b").
 *
 * Measured with native LAMMPS (black box, pair_write on two-atom inputs,
 * units real, a=8, b=10, eps=0.1, sigma=3):
 *  - the LJ energy equals the Howto_bioFF forms above to 1e-9 absolute;
 *  - the LJ force is NOT the derivative of that energy: for a < r <= b the
 *    native fpair is the unswitched LJ fpair times S(r) with the energy
 *    switching polynomial S(r) = (b^2-r^2)^2 (b^2+2r^2-3a^2)/(b^2-a^2)^3
 *    (agreement to 8 digits at r = 8.02 ... 9.90);
 *  - the Coulomb (charmmfsh) energy and radial force are
 *    E = C q_i q_j (b-r)^2/(r b^2) and F = C q_i q_j (1/r^2 - 1/b^2), agreeing to
 *    1e-12 with C = 332.0716 in units real (C = 332.06371 is off by 6e-3);
 *  - in units lj the Coulomb conversion factor is the LAMMPS one (C = 1);
 *    the CHARMM value is applied only in units real (the docs state real
 *    units; other unit styles are not measured and keep qqrd2e);
 *  - the charmmfsw/coul/long Coulombic energy is the plain damped
 *    C q_i q_j erfc(g r)/r with C = 332.0716 (agreement 1e-9 at r = 6 and 8.5
 *    for g = 0.3), i.e. no force shifting, and the cutoff at the Coulombic
 *    cutoff.
 */
const CHARMM_QQR2E_REAL = 332.0716;
const LAMMPS_QQR2E_REAL = 332.06371;

abstract class PairLJCharmmfswBase extends PairLJCharmmCoulCharmm {
  /** Per type-pair sigma^6 and the energy constant added for r <= a (Howto_bioFF.rst). */
  protected fswSig6 = new Float64Array(0);
  protected fswAdd = new Float64Array(0);
  /** a^3, b^3, a^6, b^6. */
  protected fA3 = 0;
  protected fB3 = 0;
  protected fA6 = 0;
  protected fB6 = 0;
  // coulConstScale (Pair base): CHARMM / LAMMPS Coulomb conversion in units real, 1 otherwise;
  // the force field multiplies the qqrd2e of the pair and of kspace by it (measured with native
  // LAMMPS: the kspace energy of lj/charmmfsw/coul/long is scaled by 332.0716/332.06371).

  override settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 3) throw new StyleError(`usage: pair_style ${this.name} inner outer [Coulomb cutoff]`);
    this.inner = parseNum(args[0], 'inner cutoff');
    this.outer = parseNum(args[1], 'outer cutoff');
    if (!(this.inner > 0) || !(this.outer > 0)) throw new StyleError(`${this.name}: cutoffs must be > 0`);
    // pair_charmm.html: "the inner cutoff distance must be less than the outer cutoff"
    if (!(this.outer > this.inner)) {
      throw new StyleError(`${this.name}: the inner cutoff (${this.inner}) must be less than the outer cutoff (${this.outer})`);
    }
    this.cutCoul = args.length === 3 ? parseNum(args[2], 'Coulomb cutoff') : this.outer;
    if (!(this.cutCoul > 0)) throw new StyleError(`${this.name}: Coulomb cutoff must be > 0`);
    this.innerCoul = this.outerCoul = this.cutCoul;
    this.aL2 = this.inner * this.inner;
    this.bL2 = this.outer * this.outer;
    this.invDL3 = 1 / (this.bL2 - this.aL2) ** 3;
    this.fA3 = this.inner ** 3;
    this.fB3 = this.outer ** 3;
    this.fA6 = this.aL2 ** 3;
    this.fB6 = this.bL2 ** 3;
  }

  override initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    const nt = this.ntypes + 1;
    if (this.fswSig6.length !== nt * nt) {
      this.fswSig6 = new Float64Array(nt * nt);
      this.fswAdd = new Float64Array(nt * nt);
    }
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    const s6 = sig ** 6;
    // Howto_bioFF.rst, r <= a branch: 4 eps s6 ((s6-r^6)/r^12 - s6/(a^6 b^6) + 1/(a^3 b^3)),
    // i.e. the LJ energy plus the constant 4 eps s6 (1/(a^3 b^3) - s6/(a^6 b^6)).
    const add = 4 * eps * s6 * (1 / (this.fA3 * this.fB3) - s6 / (this.fA6 * this.fB6));
    const k1 = i * nt + j, k2 = j * nt + i;
    this.fswSig6[k1] = this.fswSig6[k2] = s6;
    this.fswAdd[k1] = this.fswAdd[k2] = add;
    return cut;
  }

  override init(ctx: StyleContext): void {
    this.coulConstScale = ctx.s?.units.style === 'real' ? CHARMM_QQR2E_REAL / LAMMPS_QQR2E_REAL : 1;
    super.init(ctx);
  }

  /** Force-switched LJ for rsq < b^2: energy and fpair (before special_lj). */
  protected ljForceSwitched(rsq: number, t: number): { e: number; fpair: number } {
    const { lj1, lj2, lj3, lj4 } = this;
    const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
    const fstd = r6inv * (lj1[t] * r6inv - lj2[t]) * r2inv;
    if (rsq <= this.aL2) {
      return { e: r6inv * (lj3[t] * r6inv - lj4[t]) + this.fswAdd[t], fpair: fstd };
    }
    // Howto_bioFF.rst, a < r <= b: E = 4 eps s6 (s6 (b^6-r^6)^2 - b^3 r^6 (a^3+b^3) (b^3-r^3)^2) / (b^6 r^12 (b^6-a^6))
    const r = Math.sqrt(rsq), r3 = rsq * r, r6 = r3 * r3;
    const b6 = this.fB6, a6 = this.fA6, b3 = this.fB3, a3 = this.fA3;
    const s6 = this.fswSig6[t];
    const d6 = b6 - r6, d3 = b3 - r3;
    const e = lj4[t] * (s6 * d6 * d6 - b3 * r6 * (a3 + b3) * d3 * d3) / (b6 * r6 * r6 * (b6 - a6));
    // measured with native LAMMPS: fpair = unswitched LJ fpair times the energy-switch polynomial S(r)
    return { e, fpair: fstd * esw(rsq, this.aL2, this.bL2, this.invDL3) };
  }

  /** The Coulomb term of one pair with the special_coul weight applied (energy, fpair). */
  protected abstract coulFswPair(rsq: number, sb: number, qi: number, qj: number, qqr: number, pc: PairCompute): { e: number; f: number };

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { bL2 } = this;
    const cutCoulSq = this.cutCoul * this.cutCoul;
    const qqr = pc.qqrd2e;
    const sLJ = pc.specialLJ;
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
        if (rsq < bL2) {
          const lj = this.ljForceSwitched(rsq, t);
          const factor = sLJ[sb];
          e = lj.e * factor;
          fpair = lj.fpair * factor;
          evdwl += e;
        }
        if (rsq < cutCoulSq && qi !== 0 && q[j] !== 0) {
          const c = this.coulFswPair(rsq, sb, qi, q[j], qqr, pc);
          ecoul += c.e;
          e += c.e;
          fpair += c.f;
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
}

/** lj/charmmfsw/coul/charmmfsh: force-shifted Coulomb, E = C q q (b-r)^2 / (r b^2). */
export class PairLJCharmmfswCoulCharmmfsh extends PairLJCharmmfswBase {
  readonly name: string = 'lj/charmmfsw/coul/charmmfsh';

  protected coulFswPair(rsq: number, sb: number, qi: number, qj: number, qqr: number, pc: PairCompute): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const b2 = this.cutCoul * this.cutCoul;
    const pre = qqr * qi * qj * pc.specialCoul[sb];
    // E = pre (b-r)^2/(r b^2); radial F = pre (1/r^2 - 1/b^2); fpair = F/r (measured, see the header)
    return { e: pre * (this.cutCoul - r) * (this.cutCoul - r) / (r * b2), f: pre * (1 / rsq - 1 / b2) / r };
  }
}

/** lj/charmmfsw/coul/long: the damped Ewald real-space Coulomb term, as lj/charmm/coul/long. */
export class PairLJCharmmfswCoulLong extends PairLJCharmmfswBase {
  readonly name: string = 'lj/charmmfsw/coul/long';
  coulLong = true;
  protected override tableSupported = true;

  private readonly erfcTables = new ErfcTableCache();

  protected coulFswPair(rsq: number, sb: number, qi: number, qj: number, qqr: number, pc: PairCompute): { e: number; f: number } {
    // pair_modify table N as in PairLJCharmmCoulLong (measured the same way for this style)
    const tab = this.erfcTables.get(this.table, this.gEwald, this.cutCoul * this.cutCoul);
    const useTab = tab !== null && rsq >= TABLE_INNER_RSQ;
    let e: number, f: number;
    const qq = qqr * qi * qj;
    if (useTab) {
      e = qq * tab!.energy(rsq);
      f = qq * tab!.force(rsq) / rsq;
    } else {
      const r = Math.sqrt(rsq);
      const grij = this.gEwald * r;
      const ex = Math.exp(-grij * grij);
      const erfc = erfcPoly(grij, ex);
      e = qq * erfc / r;
      f = qq * (erfc + EWALD_F * grij * ex) / (r * rsq);
    }
    const fc = pc.specialCoul[sb];
    if (fc < 1) {
      // special_bonds: removes (1 - w) of the bare term, as in PairLJCharmmCoulLong
      const bare = qq * (useTab ? tab!.coul(rsq) : 1 / Math.sqrt(rsq));
      e -= (1 - fc) * bare;
      f -= (1 - fc) * bare / rsq;
    }
    return { e, f };
  }
}
