import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';
import type { UnitStyle } from '../../types';

/*
 * pair_style zbl inner outer — docs.lammps.org/pair_zbl.html:
 *   pair_style zbl inner outer
 * "inner = distance where switching function begins", "outer = global cutoff
 * for ZBL interaction".
 * "Style *zbl* computes the Ziegler-Biersack-Littmark (ZBL) screened nuclear
 * repulsion for describing high-energy collisions between atoms." (Ziegler)
 * "It includes an additional switching function that ramps the energy, force, and
 * curvature smoothly to zero between an inner and outer cutoff. The potential
 * energy due to a pair of atoms at a distance r_ij is given by:"
 *
 *   E^{ZBL}_{ij} & = \frac{1}{4\pi\epsilon_0} \frac{Z_i Z_j \,e^2}{r_{ij}} \phi(r_{ij}/a)+ S(r_{ij})\\
 *   a & =  \frac{0.46850}{Z_{i}^{0.23} + Z_{j}^{0.23}}\\
 *   \phi(x) & =  0.18175e^{-3.19980x} + 0.50986e^{-0.94229x} + 0.28022e^{-0.40290x} + 0.02817e^{-0.20162x}\\
 *
 * "where *e* is the electron charge, :math:`\epsilon_0` is the electrical
 * permittivity of vacuum, and :math:`Z_i` and :math:`Z_j` are the nuclear
 * charges of the two atoms.  The switching function :math:`S(r)` is
 * identical to that used by :doc:`pair_style lj/gromacs <pair_gromacs>`.
 * Here, the inner and outer cutoff are the same for all pairs of atom
 * types."
 * Switching function, docs.lammps.org/pair_gromacs.html (there applied to the
 * unswitched pair energy E; here E = E^ZBL before switching):
 *
 *   S(r) = & C \qquad r < r_1 \\
 *   S(r) = & \frac{A}{3} (r - r_1)^3 + \frac{B}{4} (r - r_1)^4 + C \qquad  r_1 < r < r_c \\
 *   A = & (-3 E'(r_c) + (r_c - r_1) E''(r_c))/(r_c - r_1)^2 \\
 *   B = & (2 E'(r_c) - (r_c - r_1) E''(r_c))/(r_c - r_1)^3 \\
 *   C = & -E(r_c) + \frac{1}{2} (r_c - r_1) E'(r_c) - \frac{1}{12} (r_c - r_1)^2 E''(r_c)
 *
 * so E + S and its first two derivatives vanish at r_c.
 * Coefficients ("The following coefficients must be defined for each pair of
 * atom types via the pair_coeff command"):
 * * Z_i (atomic number for first atom type, e.g. 13.0 for aluminum)
 * * Z_j (ditto for second atom type)
 * "when :math:`i==j` it is required that :math:`Z_i == Z_j`".
 * Mixing: "For atom type pairs *i,j* and :math:`i \neq j`, the :math:`Z_i` and
 * :math:`Z_j` coefficients can be mixed by taking :math:`Z_a` and
 * :math:`Z_b` from the values specified for the two cases where :math:`i
 * == j`" and "The pair_modify mix option has no effect on the
 * mixing behavior".
 * "The ZBL pair style does not support the pair_modify shift option, since
 * the ZBL interaction is already smoothed to 0.0 at the cutoff." / "This
 * pair style does not support the pair_modify tail option for adding
 * long-range tail corrections to energy and pressure, since there are no
 * corrections for a potential that goes to 0.0 at the cutoff."
 * Units: "The numerical values of the exponential decay constants in the
 * screening function depend on the unit of distance. In the above equation
 * they are given for units of Angstroms. LAMMPS will automatically convert
 * these values to the distance unit of the specified LAMMPS units setting."
 * The conversion multiplies the four decay constants by the number of
 * Angstroms in one distance unit of the active unit style
 * (docs.lammps.org/units.html: distance = Angstroms for real and metal,
 * meters for si, centimeters for cgs, Bohr for electron, micrometers for
 * micro, nanometers for nano, sigma for lj — used as-is there). The factor
 * for electron units is the CODATA Bohr radius in Angstroms (units.html does
 * not list it; the other engine constants are 2006-CODATA era, see
 * units.ts). Equivalently the screening length 0.46850/(...)/ could be
 * converted instead; the product decay*r/a is the same.
 * (1/4\pi\epsilon_0) e^2 is the Coulomb prefactor pc.qqrd2e — the same
 * constant pair coul/cut uses for C q_i q_j / r (docs.lammps.org/
 * pair_coul.html, see units.ts for its per-unit values).
 * Special weights: the pair_zbl page is silent on special_bonds; the ZBL
 * interaction is purely Coulombic between nuclei, so the Coulombic weight
 * (pc.specialCoul) is applied.
 */

/** Screening amplitudes and decay constants for phi(x), in Angstrom-based units (pair_zbl.html). */
const SCREEN = [0.18175, 0.50986, 0.28022, 0.02817];
const DECAY = [3.19980, 0.94229, 0.40290, 0.20162];

/** Angstroms per distance unit of each unit style (docs.lammps.org/units.html). */
const ANGSTROM_PER_UNIT: Record<UnitStyle, number> = {
  lj: 1, real: 1, metal: 1, si: 1e10, cgs: 1e8,
  electron: 0.52917721092, micro: 1e4, nano: 10,
};

export class PairZBL extends Pair {
  readonly name: string = 'zbl';
  virialFdotr = true;
  inner = 0;
  outer = 0;
  p!: PairParams;
  /** z_i * z_j per type pair. */
  zz = new Float64Array(0);
  /** 1/a with a = 0.46850/(Z_i^0.23 + Z_j^0.23) in distance units. */
  ainv = new Float64Array(0);
  /** inner cutoff r_1 (global). */
  rinTab = new Float64Array(0);
  /** Switching polynomial A, B, C for a unit prefactor (pair_gromacs.html). */
  aTab = new Float64Array(0);
  bTab = new Float64Array(0);
  cTab = new Float64Array(0);
  /** Decay constants converted to the current distance unit (set at init). */
  dScaled: number[] = [...DECAY];
  /** Coulomb prefactor of the current run (pc.qqrd2e); single() uses the last-seen value. */
  qqrd2e = 1;

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('usage: pair_style zbl inner outer');
    this.inner = parseNum(args[0], 'inner');
    this.outer = parseNum(args[1], 'outer');
    if (!(this.inner > 0)) throw new StyleError('zbl: the inner cutoff must be > 0');
    if (!(this.outer > this.inner)) throw new StyleError(`zbl: inner cutoff ${this.inner} must be less than outer cutoff ${this.outer}`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['z1', 'z2']);
  }

  coeff(args: string[]): void {
    if (args.length !== 4) throw new StyleError('usage: pair_coeff I J z_i z_j');
    const zi = parseNum(args[2], 'z_i');
    const zj = parseNum(args[3], 'z_j');
    if (!(zi > 0) || !(zj > 0)) throw new StyleError('zbl: Z coefficients must be > 0 (multiples of a proton charge)');
    this.p.setRange(args[0], args[1], [zi, zj]);
  }

  initStyle(ctx: StyleContext): void {
    // pair_zbl.html: "The ZBL pair style does not support the pair_modify shift
    // option, since the ZBL interaction is already smoothed to 0.0 at the
    // cutoff." / "This pair style does not support the pair_modify tail option
    // for adding long-range tail corrections to energy and pressure, since
    // there are no corrections for a potential that goes to 0.0 at the cutoff."
    if (this.shift) throw new StyleError('pair_modify shift yes is not supported for pair style zbl (the ZBL interaction is already smoothed to 0.0 at the cutoff)');
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style zbl (there are no corrections for a potential that goes to 0.0 at the cutoff)');
    // "LAMMPS will automatically convert these values to the distance unit of
    // the specified LAMMPS units setting" — the decay constants scale with
    // the Angstrom content of the distance unit.
    const style = ctx.s?.units.style ?? 'metal';
    const ang = ANGSTROM_PER_UNIT[style];
    this.dScaled = DECAY.map((d) => d * ang);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      // the Z_i, Z_j "coefficients can be mixed by taking :math:`Z_a` and
      // :math:`Z_b` from the values specified for the two cases where
      // :math:`i == j`"; "The pair_modify
      // mix option has no effect on the mixing behavior".
      p.setMixed(i, j, 'z1', p.get('z1', i, i));
      p.setMixed(i, j, 'z2', p.get('z2', j, j));
    }
    if (i === j && p.get('z1', i, j) !== p.get('z2', i, j)) {
      throw new StyleError(`zbl: Z_i must equal Z_j for the i==j pair ${i} ${j}`);
    }
    const nt = this.ntypes + 1;
    if (this.zz.length !== nt * nt) {
      this.zz = new Float64Array(nt * nt);
      this.ainv = new Float64Array(nt * nt);
      this.rinTab = new Float64Array(nt * nt);
      this.aTab = new Float64Array(nt * nt);
      this.bTab = new Float64Array(nt * nt);
      this.cTab = new Float64Array(nt * nt);
    }
    const z1 = p.get('z1', i, j), z2 = p.get('z2', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    this.zz[k1] = this.zz[k2] = z1 * z2;
    const ai = (z1 ** 0.23 + z2 ** 0.23) / 0.46850;
    this.ainv[k1] = this.ainv[k2] = ai;
    this.rinTab[k1] = this.rinTab[k2] = this.inner;
    // A, B, C of the switching polynomial from E, E', E'' of the unswitched
    // energy (prefactor 1; the prefactor qqrd2e*zz scales all of it) at r_c:
    // g = phi/r, g' = -(sum ek*lambda*ainv + phi/r)/r,
    // g'' = sum ek*lambda^2*ainv^2/r + 2 sum ek*lambda*ainv/r^2 + 2 phi/r^3.
    const rc = this.outer, uc = rc * ai;
    const [d1, d2, d3, d4] = this.dScaled;
    const e1 = SCREEN[0] * Math.exp(-d1 * uc), e2 = SCREEN[1] * Math.exp(-d2 * uc);
    const e3 = SCREEN[2] * Math.exp(-d3 * uc), e4 = SCREEN[3] * Math.exp(-d4 * uc);
    const phi = e1 + e2 + e3 + e4;
    const dl = e1 * d1 + e2 * d2 + e3 * d3 + e4 * d4;
    const ddl = e1 * d1 * d1 + e2 * d2 * d2 + e3 * d3 * d3 + e4 * d4 * d4;
    const gp = -(dl * ai + phi / rc) / rc;
    const gpp = ddl * ai * ai / rc + 2 * dl * ai / (rc * rc) + 2 * phi / (rc * rc * rc);
    const dd = rc - this.inner;
    this.aTab[k1] = this.aTab[k2] = (-3 * gp + dd * gpp) / (dd * dd);
    this.bTab[k1] = this.bTab[k2] = (2 * gp - dd * gpp) / (dd * dd * dd);
    this.cTab[k1] = this.cTab[k2] = -phi / rc + 0.5 * dd * gp - dd * dd * gpp / 12;
    return rc;
  }

  /** Energy and radial force (positive = repulsive) of one pair, special factor excluded. */
  private kernel(t: number, r: number, rsq: number, pref: number): { eng: number; fforce: number } {
    const ai = this.ainv[t];
    const u = r * ai;
    const [d1, d2, d3, d4] = this.dScaled;
    const e1 = SCREEN[0] * Math.exp(-d1 * u), e2 = SCREEN[1] * Math.exp(-d2 * u);
    const e3 = SCREEN[2] * Math.exp(-d3 * u), e4 = SCREEN[3] * Math.exp(-d4 * u);
    const phi = e1 + e2 + e3 + e4;
    const dl = e1 * d1 + e2 * d2 + e3 * d3 + e4 * d4;
    const k = pref * this.zz[t];
    // E = k*phi/r + S ; -dE/dr = k*(-g') with -g' = dl*ainv/r + phi/r^2;
    // fforce is the fpair coefficient (force vector = fpair * dx), i.e.
    // (-dE/dr)/r, and the switching term adds -k*(A s^2 + B s^3)/r.
    const fr = k * (dl * ai / r + phi / rsq);
    let eng = k * (phi / r);
    let fforce = fr / r;
    if (r > this.rinTab[t]) {
      const s = r - this.rinTab[t], s2 = s * s, s3 = s2 * s;
      const a = this.aTab[t], b = this.bTab[t], c = this.cTab[t];
      eng += k * ((a / 3) * s3 + (b / 4) * s3 * s + c);
      fforce -= k * (a * s2 + b * s3) / r;
    } else {
      eng += k * this.cTab[t];
    }
    return { eng, fforce };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const pref = pc.qqrd2e;
    this.qqrd2e = pref;
    const sC = pc.specialCoul;
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
        const factor = sC[jj >>> SBBITS];
        const r = Math.sqrt(rsq);
        const { eng, fforce } = this.kernel(t, r, rsq, pref);
        const fpair = factor * fforce;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * eng;
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, _fl: number, _qi: number, _qj: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const r = Math.sqrt(rsq);
    const { eng, fforce } = this.kernel(t, r, rsq, this.qqrd2e);
    return { fforce: factorCoul * fforce, eng: factorCoul * eng };
  }

  dataCoeffs(): string[] | null {
    // The per-type "Pair Coeffs" form cannot express the two Z values pair_coeff
    // takes, and pair_zbl.html documents no data-file section; only the
    // PairIJ form (dataCoeffsIJ) round-trips through read_data.
    return null;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('z1', i, j))} ${fmtCoeff(this.p.get('z2', i, j))}`);
      }
    }
    return out;
  }
}
