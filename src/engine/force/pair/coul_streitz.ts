import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { parseNum } from '../util';
import { erfcExact } from '../erfc';

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
 * Near za = zb the closed form cancels, so for |za - zb| R < 0.2 the code uses the
 * Taylor series in d = (za - zb)/2 about m = (za + zb)/2 (coefficients generated
 * symbolically; the d^8 truncation error is below 1e-11 relative at that bound).
 *
 * The 1/r real-space part follows the coul/wolf (pair_coul.ts) and Ewald real-space
 * conventions, with the erfc evaluated exactly (erfc.ts erfcExact), not from the
 * tables coul/long uses; the w38streitz_ewald_* oracle cases agree with native at 1e-9.
 * pair_modify table other than its default, shift and tail are refused.
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);
/** Series branch: |za - zb| R below this uses the Taylor expansion in d = (za - zb)/2. */
const SERIES_DR = 0.2;

/** Slater integral S(R) = [f_a|f_b](R) - 1/R and dS/dR for exponents za, zb (R > 0). */
export const slaterShort = (R: number, za: number, zb: number): { s: number; ds: number } => {
  const m = 0.5 * (za + zb), d = 0.5 * (za - zb);
  if (Math.abs(d) * R < SERIES_DR) return seriesS(R, m, d);
  const D1 = za * za - zb * zb;
  const XA = Math.exp(-2 * za * R), XB = Math.exp(-2 * zb * R);
  const zb4 = zb ** 4, za4 = za ** 4;
  const cA = -zb4 * (za * R * D1 + (3 * za * za - zb * zb));
  const cB = -za4 * (zb * R * D1 + (za * za - 3 * zb * zb));
  const N = XA * cA + XB * cB;
  const dcA = -zb4 * za * D1, dcB = -za4 * zb * D1;
  const dN = XA * (-2 * za * cA + dcA) + XB * (-2 * zb * cB + dcB);
  const D = D1 * D1 * D1;
  return { s: N / (R * D), ds: (dN * R - N) / (R * R * D) };
};

/** Taylor series of S in d about m (even in d), see the header comment. */
const seriesS = (R: number, m: number, d: number): { s: number; ds: number } => {
  const E = Math.exp(-2 * m * R);
  const m2 = m * m, m3 = m2 * m, m4 = m3 * m, m5 = m4 * m, m6 = m5 * m, m7 = m6 * m;
  const R2 = R * R, R3 = R2 * R, R4 = R3 * R, R5 = R4 * R, R6 = R5 * R, R7 = R6 * R;
  const s0 = (-E * (4 * R3 * m3 + 18 * R2 * m2 + 33 * R * m + 24)) / (24 * R);
  const ds0 = E * (R2 * m4 / 3 + 7 * R * m3 / 6 + 2 * m2 + 2 * m / R + 1 / R2);
  const c2 = (-E * (4 * R4 * m4 + 30 * R3 * m3 + 75 * R2 * m2 + 90 * R * m + 45)) / (60 * m);
  const c4 = (E * (-8 * R6 * m6 - 84 * R5 * m5 - 266 * R4 * m4 - 280 * R3 * m3 + 210 * R * m + 105)) / (840 * m3);
  const c6 = (E * R2 * (-8 * R6 * m6 - 108 * R5 * m5 - 414 * R4 * m4 - 252 * R3 * m3 + 1134 * R2 * m2 + 1890 * R * m + 945)) / (11340 * m3);
  const c8 = (E * R4 * (-4 * R6 * m6 - 66 * R5 * m5 - 297 * R4 * m4 + 2376 * R2 * m2 + 4158 * R * m + 2079)) / (124740 * m3);
  const k2 = (E * R * m * (4 * R3 * m3 + 22 * R2 * m2 + 30 * R * m + 15)) / 30;
  const k4 = (E * R * (4 * R5 * m5 + 30 * R4 * m4 + 28 * R3 * m3 - 126 * R2 * m2 - 210 * R * m - 105)) / (210 * m);
  const k6 = (E * R * (8 * R7 * m7 + 76 * R6 * m6 + 36 * R5 * m5 - 990 * R4 * m4 - 1764 * R3 * m3 + 378 * R2 * m2 + 1890 * R * m + 945)) / (5670 * m3);
  const k8 = (E * R3 * (2 * R7 * m7 + 23 * R6 * m6 - 594 * R4 * m4 - 1188 * R3 * m3 + 1485 * R2 * m2 + 4158 * R * m + 2079)) / (31185 * m3);
  const d2 = d * d;
  const s = s0 + d2 * (c2 + d2 * (c4 + d2 * (c6 + d2 * c8)));
  const ds = ds0 + d2 * (k2 + d2 * (k4 + d2 * (k6 + d2 * k8)));
  return { s, ds };
};

/** Point-charge short-range potential U_b(R) = -(zeta_b + 1/R) exp(-2 zeta_b R) and dU/dR. */
export const pointU = (R: number, z: number): { u: number; du: number } => {
  const E = Math.exp(-2 * z * R);
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

export class PairCoulStreitz extends Pair {
  readonly name: string = 'coul/streitz';
  virialFdotr = true;
  /** Excluded pairs stay in the list so a special weight can be rejected (see compute). */
  keepExcluded = true;
  mode: 'wolf' | 'ewald' = 'wolf';
  cutGlobal = 0;
  alpha = 0;
  private fileName = '';
  private fileRead = false;
  private entries = new Map<string, StreitzParams>();
  /** Per atom type (1..ntypes) parameters; null until pair_coeff sets them. */
  private tp: (StreitzParams | null)[] = [];
  /** Wolf shifts of the 1/r part (coul/wolf): energy shift and force shift. */
  private eShiftW = 0;
  private fShiftW = 0;
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
    if (kw === 'wolf') {
      if (args.length !== 3) throw new StyleError('pair_style coul/streitz with keyword wolf needs a damping parameter: pair_style coul/streitz cutoff wolf alpha');
      this.alpha = parseNum(args[2], 'alpha');
      this.mode = 'wolf';
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
      throw new StyleError(`pair_coeff for style coul/streitz needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    }
    if (!this.fileRead) {
      this.entries = parseStreitzFile(ctx.readFile(filename), filename);
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
    const rc = this.cutGlobal;
    if (this.mode === 'wolf') {
      const arc = this.alpha * rc;
      this.eShiftW = erfcExact(arc) / rc;
      this.fShiftW = -(this.eShiftW + TWO_OVER_SQRTPI * this.alpha * Math.exp(-arc * arc)) / rc;
    }
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
    const wolf = this.mode === 'wolf';
    const alpha = this.alpha, g = this.gEwald;
    let ecoul = 0;
    // per owned atom: chi q + J q^2 / 2 (and the wolf self term)
    const selfWolf = wolf ? -(erfcExact(alpha * rc) / (2 * rc) + alpha / Math.sqrt(Math.PI)) : 0;
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
          const ex = Math.exp(-grij * grij);
          const erfcc = erfcExact(grij);
          const pref = C * qq;
          const Gc = cS * this.rcS[t] + pI.Z * qj * this.rcUj[t] + pJ.Z * qi * this.rcUi[t];
          const dGc = cS * this.rcDS[t] + pI.Z * qj * this.rcDUj[t] + pJ.Z * qi * this.rcDUi[t];
          const Gsf = G - Gc - (r - rc) * dGc;
          e = pref * (erfcc / r - this.eShiftW) + C * Gsf;
          fpair = (pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * alpha * ex) / rsq + this.fShiftW / r)
            - C * (dG - dGc) / r);
        } else {
          const grij = g * r;
          const ex = Math.exp(-grij * grij);
          const erfcc = erfcExact(grij);
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

