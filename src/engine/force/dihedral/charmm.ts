import { SimpleBonded, atomIndex, dihedralGeometry, applyDihedral, delta } from '../bonded_util';
import { parseNum, parseInt_ } from '../util';
import { StyleError, type BondedCompute, type Pair, type StyleContext } from '../types';

/*
 * dihedral_style charmm and charmmfsw (wave 13).
 *
 * Dihedral term — docs.lammps.org/dihedral_charmm.html: "E = K [ 1 + \cos (n \phi - d) ]",
 * with the coefficients "K (energy)", "n (integer >= 0)", "d (integer value of degrees)" and
 * "weighting factor (1.0, 0.5, or 0.0)". The weighting factor does not touch K; it scales the
 * 1-4 non-bonded terms of the quadruplet's first and fourth atoms.
 *
 * 1-4 non-bonded terms — docs.lammps.org/dihedral_charmm.html: "interactions between the first
 * and fourth atoms in a dihedral are skipped during the normal non-bonded force computation and
 * instead evaluated as part of the dihedral using special epsilon and sigma values specified with
 * the" pair_coeff command of the lj/charmm pair styles. "For CHARMM force fields, the special_bonds
 * 1-4 interaction scaling factor should be set to 0.0. Since the corresponding 1-4 non-bonded
 * interactions are computed with the dihedral." "In 6-membered rings, ... the weighting factor has
 * to be 0.5 in this case." The epsilon_14 and sigma_14 values are read from the linked pair style
 * (docs.lammps.org/pair_charmm.html: "the epsilon, sigma, epsilon_14, and sigma_14 coefficients for
 * all of the lj/charmm pair styles can be mixed").
 *
 * Measured with native LAMMPS (black box; 4-atom chains and an 16-atom two-chain file, pair styles
 * lj/charmm/coul/charmm, lj/charmm/coul/long, lj/charmmfsw/coul/charmmfsh and lj/charmmfsw/coul/long,
 * 2026-10-08, scratch directory /tmp/haiku-dihcharmm/):
 *  - 1-4 Lennard-Jones with dihedral charmm: the plain 4 eps14 [(sigma14/r)^12 - (sigma14/r)^6] at
 *    every separation (checked from 2.6 to 10.5 A, no switching, no cutoff). Tallied in evdwl.
 *  - 1-4 Coulomb with dihedral charmm: w C q_i q_j / r, C = qqr2e times the pair's conversion scale
 *    (real: 332.06371 for lj/charmm pairs, 332.0716 for lj/charmmfsw pairs). Tallied in ecoul; no
 *    cutoff and no damping, also with lj/charmm/coul/long and lj/charmmfsw/coul/long. The pppm
 *    part and the pair's own special_bonds handling do not depend on w.
 *  - Each dihedral line adds its own 1-4 term: two lines on one quadruplet with w = 1 gave exactly
 *    twice the single-line evdwl and ecoul. K enters edihed only (independent of w).
 *  - dihedral charmmfsw with lj/charmmfsw/coul/charmmfsh: the 1-4 LJ gets the constant
 *    4 eps14 sigma14^6 [1/(a^3 b^3) - sigma14^6/(a^6 b^6)] (a, b the pair's LJ inner and outer
 *    cutoffs) at every r. Its LJ then equals the first branch of the force-switched expression in
 *    docs.lammps.org/Howto_bioFF.html, "4  \epsilon \sigma^6  \left(\frac{\displaystyle\sigma"
 *    (r <= a), evaluated for all r (measured at r = 8.5 and 10.5, relative error 1e-9).
 *  - dihedral charmmfsw with lj/charmmfsw/coul/charmmfsh: the 1-4 Coulomb is
 *    C q_i q_j (1/r - 2/b + r/b^2) = C q_i q_j (b-r)^2/(r b^2), b the Coulomb cutoff, for every r
 *    (zero at r = b, not cut off beyond it). Docs: "C(r) \frac{\displaystyle (b-r)^2}{\displaystyle r b^2}"
 *    in docs.lammps.org/Howto_bioFF.html.
 *  - dihedral charmmfsw with lj/charmmfsw/coul/long: the 1-4 Coulomb is the plain term (C = 332.0716).
 *  - dihedral charmmfsw with an lj/charmm pair: native aborts with the message Dihedral charmmfsw is
 *    incompatible with Pair style, so this style throws likewise. dihedral charmm with
 *    lj/charmm/coul/charmm/implicit was not measured and throws.
 *  - With a nonzero weight, native aborts unless the 1-4 special_bonds weights are 0 (its message
 *    begins Must use special_bonds charmm); measured: special_bonds lj/coul 1-4 = 0.5 with w = 1
 *    aborts, w = 0 runs. This is enforced here for lj and coul 1-4 weights.
 *
 * Coefficient example from docs.lammps.org/dihedral_charmm.html: "dihedral_coeff  1 0.2 1 180 1.0".
 */

/** Pair-style link the ForceField must provide before init (see registry/dihedral_charmm.ts). */
export interface CharmmLink {
  pair: Pair | null;
  /** special_bonds weights as set by the input: [1-2, 1-3, 1-4] for lj and coul. */
  special: { lj: readonly number[]; coul: readonly number[] };
}

/** Coulomb form used for the 1-4 term; fixed by the linked pair style. */
type Coul14 = 'plain' | 'fsh';

const LJ_PLAIN_PAIRS = new Set([
  'lj/charmm/coul/charmm', 'lj/charmm/coul/long',
  'lj/charmmfsw/coul/charmmfsh', 'lj/charmmfsw/coul/long',
]);

/** Pair-style cut data the 1-4 terms need; the pair keeps these as protected fields. */
interface PairCuts { inner: number; outer: number; cutCoul: number; coulConstScale: number; p: { get(name: string, i: number, j: number): number } }

export class DihedralCharmm extends SimpleBonded {
  readonly name: string = 'charmm';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['K', 'n', 'd', 'w'];
  /** True for dihedral_style charmmfsw. */
  protected readonly forceSwitched: boolean = false;
  protected link: CharmmLink | null = null;

  /** Called by the force field with its pair style and special_bonds before init. */
  linkForceField(link: CharmmLink): void {
    this.link = link;
  }

  protected parse(args: string[]): number[] {
    if (args.length !== 4) throw new StyleError(`dihedral_coeff ${this.name} needs K n d w`);
    const K = parseNum(args[0], 'K');
    const n = parseInt_(args[1], 'n');
    const d = parseInt_(args[2], 'd');
    const w = parseNum(args[3], 'weighting factor');
    if (n < 0) throw new StyleError(`dihedral ${this.name} n must be an integer >= 0`);
    if (!(w >= 0 && w <= 1)) throw new StyleError(`dihedral ${this.name} weighting factor must be between 0 and 1 (documented values 1.0, 0.5, 0.0)`);
    return [K, n, d, w];
  }

  init(ctx?: StyleContext): void {
    super.init(ctx);
    const link = this.link;
    if (!link) {
      throw new StyleError(`dihedral_style ${this.name} needs its pair style linked (the force field must call linkForceField)`);
    }
    const pair = link.pair;
    if (!pair) throw new StyleError(`dihedral_style ${this.name} needs a pair style of the lj/charmm family`);
    const pn = pair.name;
    if (this.forceSwitched) {
      if (!pn.startsWith('lj/charmmfsw/')) throw new StyleError(`Dihedral charmmfsw is incompatible with Pair style ${pn}`);
    } else if (!LJ_PLAIN_PAIRS.has(pn)) {
      throw new StyleError(`dihedral_style ${this.name} is not measured with pair style ${pn} (supported: ${[...LJ_PLAIN_PAIRS].join(', ')})`);
    }
    const anyWeight = this.params.p('w').some((w, t) => t > 0 && this.params.set[t] === 1 && w !== 0);
    if (anyWeight && (link.special.lj[2] !== 0 || link.special.coul[2] !== 0)) {
      throw new StyleError(`Must use 'special_bonds charmm' (1-4 lj and coul = 0) with dihedral style ${this.name} for use with CHARMM pair styles`);
    }
  }

  /** Coulomb 1-4 form: fsh for lj/charmmfsw/coul/charmmfsh with charmmfsw, plain otherwise. */
  private coul14Form(pn: string): Coul14 {
    return this.forceSwitched && pn.endsWith('/coul/charmmfsh') ? 'fsh' : 'plain';
  }

  compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const K = this.params.p('K'), nn = this.params.p('n'), dd = this.params.p('d'), ww = this.params.p('w');
    const pair = this.link?.pair as unknown as PairCuts | null;
    if (!pair || !this.link?.pair) throw new StyleError(`dihedral_style ${this.name}: no pair style`);
    const pn = this.link.pair.name;
    const form = this.coul14Form(pn);
    const C = bc.s.units.qqr2e * pair.coulConstScale;
    const a = pair.inner, b = pair.outer, bCoul = pair.cutCoul;
    const a3b3 = (a ** 3) * (b ** 3), a6b6 = a3b3 * a3b3;
    const DEG2RAD = Math.PI / 180;
    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    const rvec = [0, 0, 0];
    let e = 0;
    for (let q = 0; q < D.n; q++) {
      const atoms = [0, 1, 2, 3].map((k) => atomIndex(bc, D.atoms[4 * q + k], 'dihedral'));
      const t = D.type[q];
      const phi = dihedralGeometry(bc, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
      const arg = nn[t] * phi - dd[t] * DEG2RAD;
      const ed = K[t] * (1 + Math.cos(arg));
      const dEdphi = -K[t] * nn[t] * Math.sin(arg);
      e += ed;
      applyDihedral(bc, atoms, dEdphi, grad, rel, ed);
      const w = ww[t];
      if (w !== 0) this.add14(bc, atoms[0], atoms[3], w, { pair, form, C, a, b, bCoul, a3b3, a6b6, rvec });
    }
    bc.acc.edihed += e;
  }

  /** 1-4 LJ and Coulomb of the pair (i, l) with weight w; energies to evdwl/ecoul, forces applied. */
  private add14(
    bc: BondedCompute, i: number, l: number, w: number,
    c: { pair: PairCuts; form: Coul14; C: number; a: number; b: number; bCoul: number; a3b3: number; a6b6: number; rvec: number[] },
  ): void {
    const { pair, form, C, a, b, bCoul, a3b3, a6b6, rvec } = c;
    const ti = bc.s.type[i], tl = bc.s.type[l];
    const eps = pair.p.get('epsilon14', ti, tl), sig = pair.p.get('sigma14', ti, tl);
    const qq = bc.s.q[i] * bc.s.q[l];
    delta(bc, i, l, rvec);
    const r2 = rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2];
    const r = Math.sqrt(r2);
    const sig6 = sig ** 6;
    const s2 = (sig * sig) / r2;
    const s6 = s2 * s2 * s2, s12 = s6 * s6;
    let eLJ = 4 * eps * (s12 - s6);
    let dEdr = (4 * eps * (-12 * s12 + 6 * s6)) / r;
    if (this.forceSwitched) eLJ += 4 * eps * sig6 * (1 / a3b3 - sig6 / a6b6);
    let eC: number, dCdr: number;
    if (form === 'fsh') {
      eC = C * qq * (1 / r - 2 / bCoul + r / (bCoul * bCoul));
      dCdr = C * qq * (-1 / r2 + 1 / (bCoul * bCoul));
    } else {
      eC = (C * qq) / r;
      dCdr = -(C * qq) / r2;
    }
    const eTot = w * (eLJ + eC);
    bc.acc.evdwl += w * eLJ;
    bc.acc.ecoul += w * eC;
    // force on l is -dE/dr * rvec/r (rvec = r_l - r_i); equal and opposite on i
    const g = -w * (dEdr + dCdr) / r;
    const fx = g * rvec[0], fy = g * rvec[1], fz = g * rvec[2];
    bc.f[3 * l] += fx; bc.f[3 * l + 1] += fy; bc.f[3 * l + 2] += fz;
    bc.f[3 * i] -= fx; bc.f[3 * i + 1] -= fy; bc.f[3 * i + 2] -= fz;
    const v = bc.virial;
    v[0] += rvec[0] * fx; v[1] += rvec[1] * fy; v[2] += rvec[2] * fz;
    v[3] += rvec[0] * fy; v[4] += rvec[0] * fz; v[5] += rvec[1] * fz;
    if (bc.eatom) { bc.eatom[i] += eTot / 2; bc.eatom[l] += eTot / 2; }
  }
}

/** dihedral_style charmmfsw — see the header; requires an lj/charmmfsw pair. */
export class DihedralCharmmfsw extends DihedralCharmm {
  override readonly name: string = 'charmmfsw';
  protected override readonly forceSwitched: boolean = true;
}
