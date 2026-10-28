import { Pair, StyleError, type AdaptAccess, type PairCompute, type StyleContext, potentialFile } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { parseNum } from '../util';
import { libm, SQRT_PI } from '../../libm';

/*
 * pair_style coul/streitz — Streitz-Mintmire Coulomb part with fixed charges.
 *
 * docs.lammps.org/pair_coul.html:
 *   pair_style coul/streitz cutoff keyword alpha
 * "Style *coul/streitz* is the Coulomb pair interaction defined as part of
 * the Streitz-Mintmire potential, as described in :ref:`this paper
 * <Streitz2>`, in which charge distribution about an atom is modeled as a
 * Slater 1\ *s* orbital."
 * "The keyword *wolf* in the coul/streitz command denotes computing
 * Coulombic interactions via Wolf summation. An additional damping
 * parameter is required for the Wolf summation, as described for the
 * coul/wolf potential above. Alternatively, Coulombic interactions can
 * be computed via an Ewald summation."
 * "Keyword *ewald* does not need a damping parameter, but a
 * :doc:`kspace_style <kspace_style>` must be defined, which can be style
 * *ewald* or *pppm*\ ."
 * The page and the file header say the style is used with pair_style eam/alloy
 * through hybrid/overlay; the file header also says it must be used with
 * units metal.
 *
 * Model (Streitz and Mintmire, Phys. Rev. B 50, 11996 (1994)): atom i carries a point charge Z_i at
 * its nucleus and a Slater 1s cloud f_i(r) = (zeta_i^3/pi) exp(-2 zeta_i r)
 * with total charge q_i - Z_i. The Coulomb energy of two such densities
 * (C = qqrd2e) splits into the bare 1/r term q_i q_j / r and a short-range
 * part G_ij(r), where, with S_ij(r) = [f_i|f_j](r) - 1/r and
 * U_b(r) = [point|f_b](r) - 1/r = -(zeta_b + 1/r) exp(-2 zeta_b r):
 *   G_ij(r) = (q_i q_j - Z_i q_j - Z_j q_i) S_ij(r) + Z_i q_j U_j(r) + Z_j q_i U_i(r).
 * Self energy per owned atom: chi_i q_i + J_i q_i^2 / 2 (chi, J from the file).
 *
 * Measured with native LAMMPS (black box), two-atom and one-atom systems
 * (units metal, atom_style charge, periodic box 100 A, AlO.streitz, Al and O):
 * - one isolated atom: ecoul = chi q + J q^2 / 2 + wolf self term, with the
 *   wolf self term -C (erfc(alpha rc)/(2 rc) + alpha/sqrt(pi)) q^2 (C = 14.399645);
 *   the pe is ecoul (evdwl 0, elong 0). Al (chi 0) gives a pure quadratic in q,
 *   O (chi 5.484763) gives the linear chi term as well;
 * - the pair energy (two-atom energy minus the two self energies) is
 *   C q_i q_j (erfc(alpha r)/r - erfc(alpha rc)/rc) + C G^sf_ij(r), where
 *   G^sf_ij(r) = G(r) - G(rc) - (r - rc) G'(rc): a shifted-force
 *   cut of the short-range part (the Al-Al and Al-O cross terms at r = 11.9 A
 *   agree with this to 5e-11 eV, while the unshifted value is off by 3e-8 eV);
 * - a constant Z_i Z_j term is absent: for q = 0 on both atoms the pair energy is 0;
 * - the 1/r part of the force carries the coul/wolf force shift (the force
 *   on the O atom at 11 A differs from -dE/dr by the coul/wolf f_shift term),
 *   and the short-range part is consistent with -dE/dr of the shifted-force energy;
 * - ewald: the reciprocal energy is in elong and identical to coul/long with the
 *   same kspace style (only the charges q enter the kspace sum); ecoul = sum over
 *   pairs C q_i q_j erfc(g r)/r + C G_ij(r) (G unshifted) + sum over atoms of
 *   chi q + J q^2/2, with no wolf self term (checked at 2, 3, 5, 11.5, 11.9 A).
 *
 * Slater integral. For two 1s densities with exponents za, zb,
 *   [f_a|f_b](R) = int f_b(x) V_a(|x - R|) d^3x,  V_a(s) = 1/s - (1/s + za) exp(-2 za s),
 * and the spherical average of V_a over a sphere of radius r about the origin is
 * (1/(2 r R)) int_{|r-R|}^{r+R} s V_a(s) ds, with s V_a(s) = 1 - (1 + za s) exp(-2 za s).
 * Doing the radial integral with the weight 2 zb^3 r exp(-2 zb r) / R gives
 *   [f_a|f_b](R) = 1/R + [ XA cA + XB cB ] / (R D),
 *   XA = exp(-2 za R), XB = exp(-2 zb R), D = (za^2 - zb^2)^3,
 *   cA = -zb^4 [ za R (za^2 - zb^2) + (3 za^2 - zb^2) ],
 *   cB = -za^4 [ zb R (za^2 - zb^2) + (za^2 - 3 zb^2) ],
 * which is symmetric in (za, zb) and reduces to the equal-zeta form
 *   1/R - (1/R + 11 z/8 + 3 z^2 R/4 + z^3 R^2/6) exp(-2 z R)
 * as za -> zb (checked in tests/engineCoulStreitz38.test.ts against 2D quadrature).
 * As native LAMMPS does (src/KSPACE/pair_coul_streitz.cpp coulomb_integral_wolf and _ewald, and
 * src/QEQ/fix_qeq_slater.cpp, at b332f28: "if (zei == zej)"), the closed form is used whenever
 * za != zb, its cancellation near za = zb included, and the equal-zeta form when za == zb.
 *
 * The 1/r real-space part follows the coul/wolf (pair_coul.ts) and Ewald real-space
 * conventions, with the C library's erfc (libm.erfc, as native calls erfc()), not from the
 * tables coul/long uses; the w38streitz_ewald_* oracle cases agree with native at 1e-9.
 * pair_modify table other than its default, shift and tail are refused.
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/**
 * The taper of the shifted-force Slater part with taper width w (dsf only): f = 1 for r <= rc - w, 0 for r >= rc, and in
 * between with x = (r - (rc - w)) / w: f = (1 - x)^3 (1 + 3x + 6x^2), df/dr its derivative. As native LAMMPS does
 * (MathSpecial::mdftaper, src/MATH/math_special.cpp at b332f28). The pair's erfc part is not tapered.
 */
export const mdfTaper = (r: number, rc: number, width: number, out: { f: number; df: number }): void => {
  const rmin = rc - width;
  if (r <= rmin) { out.f = 1; out.df = 0; return; }
  if (r >= rc) { out.f = 0; out.df = 0; return; }
  const x = (r - rmin) / (rc - rmin);
  const dx = 1 / (rc - rmin);
  const t1 = 1 - x;
  const t2 = 1 + 3 * x + 6 * x * x;
  const cube = t1 * t1 * t1;
  out.f = cube * t2;
  out.df = (-3 * t1 * t1 * t2 + cube * (3 + 12 * x)) * dx;
};
const TAPER_OUT = { f: 1, df: 0 };
/** Slater integral S(R) = [f_a|f_b](R) - 1/R and dS/dR for exponents za, zb (R > 0). */
export const slaterShort = (R: number, za: number, zb: number): { s: number; ds: number } => {
  if (za === zb) return equalS(R, za);
  const D1 = za * za - zb * zb;
  const XA = libm.exp(-2 * za * R), XB = libm.exp(-2 * zb * R);
  const zb4 = zb ** 4, za4 = za ** 4;
  const cA = -zb4 * (za * R * D1 + (3 * za * za - zb * zb));
  const cB = -za4 * (zb * R * D1 + (za * za - 3 * zb * zb));
  const N = XA * cA + XB * cB;
  const dcA = -zb4 * za * D1, dcB = -za4 * zb * D1;
  const dN = XA * (-2 * za * cA + dcA) + XB * (-2 * zb * cB + dcB);
  const D = D1 * D1 * D1;
  return { s: N / (R * D), ds: (dN * R - N) / (R * R * D) };
};

/** S and dS/dR for equal exponents: -(1/R + 11 z/8 + 3 z^2 R/4 + z^3 R^2/6) exp(-2 z R), see the header comment. */
const equalS = (R: number, z: number): { s: number; ds: number } => {
  const E = libm.exp(-2 * z * R);
  const z2 = z * z, z3 = z2 * z, z4 = z3 * z;
  const R2 = R * R, R3 = R2 * R;
  const s = (-E * (4 * R3 * z3 + 18 * R2 * z2 + 33 * R * z + 24)) / (24 * R);
  const ds = E * (R2 * z4 / 3 + 7 * R * z3 / 6 + 2 * z2 + 2 * z / R + 1 / R2);
  return { s, ds };
};

/** Slater integral S(R) alone (no derivative, no allocation): the same value as slaterShort(R, za, zb).s. */
export const slaterS = (R: number, za: number, zb: number): number => {
  if (za === zb) return equalS(R, za).s;
  const D1 = za * za - zb * zb;
  const XA = libm.exp(-2 * za * R), XB = libm.exp(-2 * zb * R);
  const zb2 = zb * zb, za2 = za * za;
  const cA = -(zb2 * zb2) * (za * R * D1 + (3 * za2 - zb2));
  const cB = -(za2 * za2) * (zb * R * D1 + (za2 - 3 * zb2));
  return (XA * cA + XB * cB) / (R * D1 * D1 * D1);
};

/** Point-charge short-range potential U_b(R) alone (no allocation): the value of pointU(R, z).u. */
export const pointUval = (R: number, z: number): number => -(z + 1 / R) * libm.exp(-2 * z * R);

/** Point-charge short-range potential U_b(R) = -(zeta_b + 1/R) exp(-2 zeta_b R) and dU/dR. */
export const pointU = (R: number, z: number): { u: number; du: number } => {
  const E = libm.exp(-2 * z * R);
  return { u: -(z + 1 / R) * E, du: E * (1 / (R * R) + 2 * z * (z + 1 / R)) };
};

export interface StreitzParams {
  /** Electronegativity X (energy units). */
  chi: number;
  /** Self-Coulomb J (energy units). */
  J: number;
  /** gamma column of the file (used by fix qeq styles). */
  gamma: number;
  /** Slater exponent zeta (1/distance). */
  zeta: number;
  /** Nucleus charge Z (charge units). */
  Z: number;
}

/**
 * Parses the AlO.streitz-style file: "element X J gamma zeta Z" per line, '#' comments.
 * The column names are those of the potential file header (the task's fix_qeq parameter list).
 */
export const parseStreitzFile = (text: string, fileName: string): Map<string, StreitzParams> => {
  const out = new Map<string, StreitzParams>();
  text.split(/\r?\n/).forEach((raw, k) => {
    const h = raw.indexOf('#');
    const line = (h >= 0 ? raw.slice(0, h) : raw).trim();
    if (!line) return;
    const w = line.split(/\s+/);
    if (w.length < 6) throw new StyleError(`coul/streitz potential file ${fileName} line ${k + 1}: expected 6 words (element X J gamma zeta Z), got ${w.length}`);
    const v = w.slice(1, 6).map((x, c) => parseNum(x, `${['X', 'J', 'gamma', 'zeta', 'Z'][c]} for ${w[0]} in ${fileName}`));
    if (out.has(w[0])) throw new StyleError(`coul/streitz potential file ${fileName} lists element ${w[0]} twice`);
    if (!(v[3] > 0)) throw new StyleError(`coul/streitz zeta for ${w[0]} in ${fileName} must be > 0`);
    out.set(w[0], { chi: v[0], J: v[1], gamma: v[2], zeta: v[3], Z: v[4] });
  });
  if (out.size === 0) throw new StyleError(`coul/streitz potential file ${fileName} has no entries`);
  return out;
};

/** coul/wolf's force shift at cutoff rc: -(erfc(alpha rc) / rc + 2 alpha / sqrt(pi) exp(-(alpha rc)^2)) / rc. */
const wolfForceShift = (alpha: number, rc: number): number => {
  const arc = alpha * rc;
  return -(libm.erfc(arc) / rc + TWO_OVER_SQRTPI * alpha * libm.exp(-arc * arc)) / rc;
};

export class PairCoulStreitz extends Pair {
  readonly name: string = 'coul/streitz';
  virialFdotr = true;
  /** Excluded pairs stay in the list so a special weight can be rejected (see compute). */
  keepExcluded = true;
  mode: 'wolf' | 'dsf' | 'ewald' = 'wolf';
  /** dsf taper width (the optional value after the alpha of pair_style coul/streitz dsf); 0 when none. */
  taper = 0;
  cutGlobal = 0;
  alpha = 0;
  private fileName = '';
  private fileRead = false;
  private entries = new Map<string, StreitzParams>();
  /** Per atom type (1..ntypes) parameters; null until pair_coeff sets them. */
  private tp: (StreitzParams | null)[] = [];
  /**
   * The Wolf force shift of the 1/r part (coul/wolf's f_shift) at the cutoff and alpha of the last pair_coeff, and that
   * cutoff. As native LAMMPS does (src/KSPACE/pair_coul_streitz.cpp at b332f28): setup_params, which pair_coeff runs,
   * sets the Wolf self terms woself and dwoself from cut_coul and g_wolf; init_style sets cut_coulsq; the force loop
   * (coulomb_integral_wolf, wolf_sum, fennell_sum) reads cut_coul and g_wolf as they are then. fix adapt can change
   * either between them.
   */
  private fShiftW = 0;
  private rcCoeff = 0;
  /** alpha at the last pair_coeff: setup_params derives woself and dwoself from g_wolf as it is then. */
  private alphaCoeff = 0;
  /** Shifted-force constants of the short-range part, per type pair (ntypes+1)^2: S, S', U_j, U_j', U_i, U_i' at rc. */
  private rcS = new Float64Array(0);
  private rcDS = new Float64Array(0);
  private rcUj = new Float64Array(0);
  private rcDUj = new Float64Array(0);
  private rcUi = new Float64Array(0);
  private rcDUi = new Float64Array(0);

  /** Typed accessor for the per-type parameters (for fix qeq/slater later). */
  typeParams(type: number): StreitzParams {
    const p = this.tp[type];
    if (!p) throw new StyleError(`coul/streitz has no parameters for atom type ${type} (pair_coeff not set)`);
    return p;
  }

  settings(args: string[]): void {
    if (args.length < 2) throw new StyleError('usage: pair_style coul/streitz cutoff keyword alpha (keyword wolf or ewald)');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style coul/streitz cutoff must be > 0');
    const kw = args[1];
    if (kw === 'wolf' || kw === 'dsf') {
      // As native LAMMPS does (src/KSPACE/pair_coul_streitz.cpp PairCoulStreitz::settings at b332f28): "if (narg < 3)
      // utils::missing_cmd_args(FLERR, "pair_style coul/streitz dsf", error)"
      if (args.length < 3) throw new StyleError(`Illegal pair_style coul/streitz ${kw} command: missing argument(s)`);
      this.alpha = parseNum(args[2], 'alpha');
      // docs.lammps.org/pair_coul.html: "the optional arguments *wolf* and *dsf* with the value representing the width of
      // taper that smoothly terminates the Coulomb integrals. ... For backward compatibility, the default taper width is zero."
      // As native LAMMPS does (src/KSPACE/pair_coul_streitz.cpp PairCoulStreitz::settings at b332f28): only the word after
      // alpha is read as the taper, and the words after it are ignored
      const taper = args.length >= 4 ? parseNum(args[3], 'taper width') : 0;
      if (kw === 'wolf' && taper !== 0) throw new StyleError(`pair_style coul/streitz wolf taper width ${taper} is not reproduced by the browser engine (only width 0)`);
      if (taper < 0) throw new StyleError('pair_style coul/streitz taper width must be >= 0');
      this.taper = taper;
      this.mode = kw;
      this.coulLong = false;
    } else if (kw === 'ewald') {
      if (args.length !== 2) throw new StyleError('pair_style coul/streitz with keyword ewald takes no damping parameter (it needs a kspace_style ewald or pppm)');
      this.mode = 'ewald';
      this.coulLong = true;
    } else {
      throw new StyleError(`unknown keyword '${kw}' in pair_style coul/streitz (expected wolf or ewald)`);
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.tp = new Array(ntypes + 1).fill(null);
  }

  /** The style has no per-pair cutoff: the global cutoff applies to every type pair. */
  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError('usage: pair_coeff * * AlO.streitz elem1 ... elemN (style coul/streitz)');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style coul/streitz must be * *');
    const filename = args[2];
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    if (!this.fileRead) {
      this.entries = parseStreitzFile(ctx.readFile(filename, potentialFile('coul/streitz')), filename);
      this.fileName = filename;
      this.fileRead = true;
    } else if (filename !== this.fileName) {
      throw new StyleError(`only one pair_coeff command is allowed for style coul/streitz (potential file already set to ${this.fileName})`);
    }
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      const e = this.entries.get(name);
      if (name === 'NULL') throw new StyleError('pair_coeff for style coul/streitz does not accept NULL types');
      if (!e) throw new StyleError(`element '${name}' is not in coul/streitz potential file ${this.fileName} (elements: ${[...this.entries.keys()].sort().join(' ')})`);
      this.tp[t] = e;
    }
    this.rcCoeff = this.cutGlobal;
    this.alphaCoeff = this.alpha;
  }

  initStyle(ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError('pair_style coul/streitz does not support the pair_modify shift and tail options');
    if (this.table !== 12) throw new StyleError('pair_style coul/streitz does not support the pair_modify table option');
    if (ctx.s && ctx.s.units.style !== 'metal') {
      throw new StyleError(`pair_style coul/streitz needs units metal (the potential file is in eV and Angstrom), got units ${ctx.s.units.style}`);
    }
    if (!this.fileRead) throw new StyleError('pair_style coul/streitz needs a pair_coeff command with the potential file');
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.tp[t]) throw new StyleError(`pair_coeff for style coul/streitz does not set atom type ${t}`);
    }
    if (this.mode !== 'ewald') this.fShiftW = wolfForceShift(this.alphaCoeff, this.rcCoeff);
    const nt = this.ntypes + 1;
    this.rcS = new Float64Array(nt * nt);
    this.rcDS = new Float64Array(nt * nt);
    this.rcUj = new Float64Array(nt * nt);
    this.rcDUj = new Float64Array(nt * nt);
    this.rcUi = new Float64Array(nt * nt);
    this.rcDUi = new Float64Array(nt * nt);
  }

  initOne(i: number, j: number): number {
    const rc = this.cutGlobal;
    const nt = this.ntypes + 1;
    // t = a * nt + b holds the constants for an i of type a and a j of type b, so the two orders differ in U_i / U_j
    for (const [a, b] of [[i, j], [j, i]]) {
      const pa = this.typeParams(a), pb = this.typeParams(b);
      const t = a * nt + b;
      const { s, ds } = slaterShort(rc, pa.zeta, pb.zeta);
      const ua = pointU(rc, pa.zeta), ub = pointU(rc, pb.zeta);
      this.rcS[t] = s;
      this.rcDS[t] = ds;
      this.rcUi[t] = ua.u;
      this.rcDUi[t] = ua.du;
      this.rcUj[t] = ub.u;
      this.rcDUj[t] = ub.du;
    }
    return rc;
  }

  /**
   * fix adapt's alpha. As native LAMMPS does (src/KSPACE/pair_coul_streitz.cpp extract at b332f28): "alpha" is g_wolf
   * (wolf and dsf) or g_ewald (ewald), which the force loop reads as it is then (wolf_sum, fennell_sum, ewald_sum);
   * woself and dwoself keep the g_wolf of the last pair_coeff, and init_style sets g_ewald from the kspace style again.
   */
  override adaptParam(name: string): AdaptAccess | null {
    if (name !== 'alpha') return null;
    if (this.mode === 'ewald') return { dim: 0, get: () => this.gEwald, set: (v) => { this.gEwald = v; } };
    return { dim: 0, get: () => this.alpha, set: (v) => { this.alpha = v; } };
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.mode === 'ewald' ? this.cutGlobal : undefined;
    return undefined;
  }

  compute(pc: PairCompute): void {
    const C = pc.qqrd2e;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const rc = this.cutGlobal;
    const wolf = this.mode !== 'ewald';
    const dsf = this.mode === 'dsf';
    const alpha = this.alpha, g = this.gEwald;
    // the energy shift erfc(alpha rc) / rc at the cutoff as it is now; the force shift of the pair_coeff cutoff (dwoself)
    const eShiftW = wolf ? libm.erfc(alpha * rc) / rc : 0;
    // dsf: the Fennell-Gezelter energy adds K (r - rc) to the wolf energy, K = -f_shift at the cutoff as it is now; its
    // force carries both K and the pair_coeff force shift (src/KSPACE/pair_coul_streitz.cpp fennell_sum at b332f28)
    const fShiftNow = dsf ? wolfForceShift(alpha, rc) : 0;
    const dsfK = -fShiftNow;
    const fShift = this.fShiftW + fShiftNow;
    let ecoul = 0;
    // per owned atom: chi q + J q^2 / 2 (and the wolf self term, woself of the pair_coeff cutoff and alpha)
    const rc0 = this.rcCoeff;
    const a0 = this.alphaCoeff;
    const selfWolf = wolf ? -(libm.erfc(a0 * rc0) / (2 * rc0) + a0 / SQRT_PI) : 0;
    for (let i = 0; i < pc.nlocal; i++) {
      const p = this.typeParams(type[i]);
      const qi = q[i];
      const eself = p.chi * qi + 0.5 * p.J * qi * qi + (wolf ? C * selfWolf * qi * qi : 0);
      if (eself !== 0) {
        ecoul += eself;
        if (pc.eatom) pc.eatom[i] += eself;
      }
    }
    const list = pc.half!;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = q[i];
      const ti = type[i];
      const pI = this.typeParams(ti);
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        if (sC[jj >>> SBBITS] !== 1) {
          throw new StyleError('pair_style coul/streitz does not support special_bonds weights other than 1 (bonded pairs in this system)');
        }
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti * nt + type[j];
        if (rsq >= cutsq[t]) continue;
        const qj = q[j];
        const pJ = this.typeParams(type[j]);
        const r = Math.sqrt(rsq);
        // fennell_sum: "if (r >= rc)" the pair adds nothing (rc as it is now; cutsq is that of the run's init)
        if (dsf && r >= rc) continue;
        const qq = qi * qj;
        // short-range part: G(r) and G'(r) with the type-pair Slater terms
        const { s, ds } = slaterShort(r, pI.zeta, pJ.zeta);
        const uj = pointU(r, pJ.zeta), ui = pointU(r, pI.zeta);
        const cS = qq - pI.Z * qj - pJ.Z * qi;
        const G = cS * s + pI.Z * qj * uj.u + pJ.Z * qi * ui.u;
        const dG = cS * ds + pI.Z * qj * uj.du + pJ.Z * qi * ui.du;
        let e: number, fpair: number;
        if (wolf) {
          const grij = alpha * r;
          const ex = libm.exp(-grij * grij);
          const erfcc = libm.erfc(grij);
          const pref = C * qq;
          const Gc = cS * this.rcS[t] + pI.Z * qj * this.rcUj[t] + pJ.Z * qi * this.rcUi[t];
          const dGc = cS * this.rcDS[t] + pI.Z * qj * this.rcDUj[t] + pJ.Z * qi * this.rcDUi[t];
          const Gsf = G - Gc - (r - rc) * dGc;
          // the Slater part is multiplied by the taper (native wolf_sum and fennell_sum: ftap on the ci terms, not on the
          // erfc part); with no taper width it is 0 past rc as it is now (a pair inside the cutoff of the run's init)
          let tf = 1, dtf = 0;
          if (this.taper > 0 || r > rc) { mdfTaper(r, rc, this.taper, TAPER_OUT); tf = TAPER_OUT.f; dtf = TAPER_OUT.df; }
          e = pref * (erfcc / r - eShiftW + dsfK * (r - rc)) + C * tf * Gsf;
          fpair = (pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * alpha * ex) / rsq + fShift / r)
            - C * (tf * (dG - dGc) + dtf * Gsf) / r);
        } else {
          const grij = g * r;
          const ex = libm.exp(-grij * grij);
          const erfcc = libm.erfc(grij);
          const pref = C * qq;
          e = pref * erfcc / r + C * G;
          fpair = (pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * g * ex) / rsq) - C * dG / r);
        }
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        ecoul += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.ecoul += ecoul;
  }

}

