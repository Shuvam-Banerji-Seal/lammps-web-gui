/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/fix_qtpie_reaxff.cpp and src/REAXFF/fix_qtpie_reaxff.h (FixQtpieReaxFF:
 * constructor, pertype_parameters, init, init_shielding, init_taper, init_olap, setup_pre_force,
 * init_storage, pre_force, calc_chi_eff, find_min_exp, calculate_Q), and src/REAXFF/
 * fix_qeq_rel_reaxff.cpp (FixQEqRelReaxFF::calc_chi_eff) and fix_qeq_rel_reaxff.h, at commit b332f28.
 * Authors credited in the C++ file headers: Efstratios M Kritikos (California Institute of Technology,
 * the original QTPIE fix); Navraj S Lalli (Imperial College London, the reimplementation and extension,
 * and the qeq/rel variant). Copyright (2003) Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The overlap integrals are those of two normalised 1s Gaussian type orbitals, with the orbital
 * exponents of the gauss file (in 1/bohr^2, converted to 1/Angstrom^2) and a distance cutoff from the
 * smallest exponent (olap_cut = 10, see fix_qtpie_reaxff.cpp). The solve itself is the fictitious-charge
 * step of charge.ts. Refused: the fix efield coupling (chi_eff with the field potential) and the
 * matrix-free and dual variants.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { type HistoryMap, solveFictitious, taperFix } from './charge';
import { parsePertypeFile, type QeqParams } from './qeq';
import type { PairReaxFF } from './pair';
import type { FixEfield } from '../../engine/fix/force_ext';
import { efieldForModel, fieldOf, findEfield, fieldPotential } from './efield';

/** ANGSTROM_TO_BOHRRADIUS_SQ of fix_qtpie_reaxff.cpp: an exponent in 1/bohr^2 is this many 1/Angstrom^2. */
const ANGSTROM_TO_BOHRRADIUS_SQ = 3.571064831;
const QSUMSMALL = 0.00001;
const OLAP_CUT = 10;

/**
 * fix qtpie/reaxff (FixQtpieReaxFF). The electronegativity of each owned atom is the overlap-weighted
 * mean of the differences chi_i - chi_j (calc_chi_eff); fix qeq/rel/reaxff (the subclass below)
 * uses the plain chi of the type instead.
 */
export class FixQtpieReaxFF extends Fix {
  readonly style: string;
  nevery: number;
  protected swa: number;
  protected swb: number;
  protected tolerance: number;
  protected imax = 200;
  protected maxwarn = true;
  protected scale = 1.0;
  protected params: QeqParams | null = null;
  protected pertype: string;
  protected gaussFile: string;
  /** gauss_exp per type (1-based), in 1/Angstrom^2. */
  protected gaussExp = new Float64Array(0);
  /** prefactor and expfactor of the overlap integrals, flattened (ntypes + 1)^2. */
  protected prefactor = new Float64Array(0);
  protected expfactor = new Float64Array(0);
  protected distCutSq = 0;
  protected Tap: Float64Array<ArrayBufferLike> = new Float64Array(8);
  protected shld = new Float64Array(0);
  protected ntypes = 0;
  protected hist: HistoryMap = new Map();
  protected matvecs = 0;
  /** fix efield (the field term of the electronegativity), or null. */
  protected efieldFix: FixEfield | null = null;
  protected matvecsS = 0;
  protected matvecsT = 0;

  constructor(sys: System, id: string, group: string, args: string[], protected pairRef: () => PairReaxFF | null, style = 'qtpie/reaxff') {
    super(sys, id, group, args);
    this.style = style;
    this.scalarFlag = true;
    this.extscalar = 0;
    // narg of LAMMPS is 9..14 with "fix ID group style" in front: 6..11 arguments here
    if (args.length < 6 || args.length > 11) throw new StyleError(`Illegal fix ${style} command: incorrect number of parameters`);
    this.pairRef()?.attachQeqFix();
    this.nevery = intArg(args[0], 'Nevery', style);
    if (this.nevery <= 0) throw new StyleError(`Illegal fix ${style} command`);
    this.swa = numArg(args[1], 'swa', style);
    this.swb = numArg(args[2], 'swb', style);
    this.tolerance = numArg(args[3], 'tolerance', style);
    this.pertype = args[4];
    this.gaussFile = args[5];
    let iarg = 6;
    while (iarg < args.length) {
      const k = args[iarg];
      if (k === 'nowarn') this.maxwarn = false;
      else if (k === 'maxiter' || k === 'scale') {
        if (iarg + 1 > args.length - 1) throw new StyleError(`Illegal fix ${style} command: ${k} needs a value`);
        if (k === 'maxiter') this.imax = intArg(args[iarg + 1], 'maxiter', style);
        else this.scale = numArg(args[iarg + 1], 'scale', style);
        iarg++;
      } else throw new StyleError(`Illegal fix ${style} command: unknown keyword ${k}`);
      iarg++;
    }
  }

  override init(): void {
    const st = this.sys.state;
    let count = 0;
    for (let i = 0; i < st.n; i++) if (st.mask[i] & this.groupBit) count++;
    if (count === 0) throw new StyleError(`Fix ${this.style} group has no atoms`);
    if (this.swb < 0) throw new StyleError(`Fix ${this.style} has negative upper Taper radius cutoff`);
    this.efieldFix = findEfield(this.sys, this.style);
    if (this.efieldFix) efieldForModel(this.sys, this.efieldFix, this.style, { allGroup: true, periodicCheck: false });
    this.ntypes = st.ntypes;
    this.Tap = taperFix(this.swa, this.swb);
    this.pertypeParameters();
    this.initShielding();
    this.initOlap();
  }

  /** pertype_parameters: the gauss exponents (distance cutoff) and the chi, eta, gamma of the pair or of the file. */
  protected pertypeParameters(): void {
    if (this.params) return;
    const nt = this.ntypes;
    this.readGaussExponents(nt);
    if (this.pertype.startsWith('reaxff')) {
      const pair = this.pairRef();
      if (!pair) throw new StyleError(`No reaxff pair style for fix ${this.style}`);
      this.params = pair.qeqParameters(nt);
      // the types of the group must have parameters in the pair style (chi, eta and gamma not all zero)
      const st = this.sys.state, p = this.params;
      for (let i = 0; i < st.n; i++) {
        if (!(st.mask[i] & this.groupBit)) continue;
        const t = st.type[i];
        if (p.chi[t] === 0.0 && p.eta[t] === 0.0 && p.gamma[t] === 0.0) {
          throw new StyleError(`No qtpie parameters for atom type ${t} provided by pair reaxff`);
        }
      }
    } else if (this.pertype.startsWith('reax/c')) {
      throw new StyleError(`Fix ${this.style} keyword 'reax/c' is obsolete; please use 'reaxff'`);
    } else {
      this.params = parsePertypeFile(this.sys.readFile(this.pertype), nt, this.style);
    }
  }

  /** The gauss file: "itype exponent" per type in 1/bohr^2 (comments after '#'), and the overlap distance cutoff. */
  protected readGaussExponents(nt: number): void {
    const gaussExp = new Float64Array(nt + 1);
    const lines = this.sys.readFile(this.gaussFile).split(/\r?\n/).map((l) => l.replace(/#.*$/, '').trim()).filter((l) => l !== '');
    for (let i = 1; i <= nt; i++) {
      const line = lines[i - 1];
      if (line === undefined) throw new StyleError('Fix qtpie/reaxff: Incorrect number of atom types in gauss file');
      const w = line.split(/\s+/);
      if (w.length !== 2) throw new StyleError(`Fix qtpie/reaxff: Incorrect number of values per line in gauss file`);
      const itype = Number(w[0]);
      if (!Number.isInteger(itype) || itype < 1 || itype > nt) throw new StyleError(`Fix qtpie/reaxff: Invalid atom type in gauss file ${w[0]}`);
      const exp = Number(w[1]);
      if (!(exp >= 0)) throw new StyleError(`Fix qtpie/reaxff: Invalid orbital exponent in gauss file ${w[1]}`);
      gaussExp[itype] = exp * ANGSTROM_TO_BOHRRADIUS_SQ;
    }
    this.gaussExp = gaussExp;
    // find_min_exp: the smallest exponent over the types 1..ntypes
    let expMin = gaussExp[1];
    for (let i = 2; i <= nt; i++) if (gaussExp[i] < expMin) expMin = gaussExp[i];
    this.distCutSq = (2 * OLAP_CUT * Math.log(10.0)) / expMin;
  }

  /** init_shielding: shld[i][j] = (gamma_i gamma_j)^(-1.5). */
  protected initShielding(): void {
    const nt = this.ntypes, p = this.params!;
    this.shld = new Float64Array((nt + 1) * (nt + 1));
    for (let i = 1; i <= nt; i++) {
      for (let j = 1; j <= nt; j++) this.shld[i * (nt + 1) + j] = Math.pow(p.gamma[i] * p.gamma[j], -1.5);
    }
  }

  /** init_olap: the Gaussian overlap prefactor and exponent of each type pair (normalised 1s orbitals). */
  protected initOlap(): void {
    const nt = this.ntypes;
    this.prefactor = new Float64Array((nt + 1) * (nt + 1));
    this.expfactor = new Float64Array((nt + 1) * (nt + 1));
    for (let i = 1; i <= nt; i++) {
      for (let j = 1; j <= nt; j++) {
        const expa = this.gaussExp[i], expb = this.gaussExp[j];
        const expsum = expa + expb;
        const expnorm = (expa * expb) / expsum;
        this.prefactor[i * (nt + 1) + j] = Math.pow(4.0 * expnorm / expsum, 0.75);
        this.expfactor[i * (nt + 1) + j] = expnorm;
      }
    }
  }

  /** setup_pre_force: the storage of the solve, then one pre_force (same step rule as fix qeq/reaxff). */
  override setup(): void { this.preForce(); }

  override preForce(): void {
    if (this.sys.state.step % this.nevery !== 0) return;
    this.solve();
  }

  override computeScalar(): number { return this.matvecs / 2.0; }

  /** pre_force: calc_chi_eff, init_matvec, the two CG solves and calculate_Q. */
  protected solve(): void {
    if (!this.params) this.init();
    const s = this.sys.state;
    const chiEff = this.calcChiEff(this.efieldFix ? fieldOf(this.sys, this.efieldFix) : null);
    const r = solveFictitious({
      sys: this.sys, groupBit: this.groupBit, swb: this.swb, ntypes: this.ntypes, shld: this.shld, Tap: this.Tap,
      eta: this.params!.eta, chiEff, hist: this.hist,
      opts: { tolerance: this.tolerance, imax: this.imax, maxwarn: this.maxwarn, warn: (t) => this.sys.warn(t), step: s.step, name: this.style },
    });
    this.matvecsS = r.matvecsS;
    this.matvecsT = r.matvecsT;
    this.matvecs = r.matvecsS + r.matvecsT;
  }

  /**
   * calc_chi_eff: for each owned atom, the overlap-weighted mean of chi_i - chi_j over the atoms j (owned
   * or ghost) within the distance cutoff; with the field E, the difference includes scale (phi_i - phi_j).
   * The self overlap is 1 so the sum is never empty.
   */
  protected calcChiEff(E: [number, number, number] | null): Float64Array {
    const nb = this.sys.nb;
    const n = nb.nlocal, N = nb.nall;
    const x = nb.xall;
    const typeAll = nb.typeall;
    const p = this.params!;
    const nt1 = this.ntypes + 1;
    const qe2f = this.sys.state.units.qe2f;
    const chiEff = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const chia = p.chi[typeAll[i]];
      const phia = E ? fieldPotential(x, i, E, qe2f) : 0;
      let sumN = 0, sumD = 0;
      for (let j = 0; j < N; j++) {
        const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
        const distSq = dx * dx + dy * dy + dz * dz;
        if (distSq < this.distCutSq) {
          const chib = p.chi[typeAll[j]];
          const overlap = this.prefactor[typeAll[i] * nt1 + typeAll[j]] * Math.exp(-this.expfactor[typeAll[i] * nt1 + typeAll[j]] * distSq);
          if (E) {
            const phib = fieldPotential(x, j, E, qe2f);
            sumN += (chia - chib + this.scale * (phia - phib)) * overlap;
          } else {
            sumN += (chia - chib) * overlap;
          }
          sumD += overlap;
        }
      }
      chiEff[i] = sumN / sumD;
    }
    return chiEff;
  }
}

/**
 * fix qeq/rel/reaxff (FixQEqRelReaxFF): the QTPIE solve with the plain electronegativity chi of each
 * type. Without fix efield (the only case ported) LAMMPS' calc_chi_eff is chi_eff = chi.
 */
export class FixQEqRelReaxFF extends FixQtpieReaxFF {
  constructor(sys: System, id: string, group: string, args: string[], pairRef: () => PairReaxFF | null) {
    super(sys, id, group, args, pairRef, 'qeq/rel/reaxff');
  }

  /**
   * calc_chi_eff of fix qeq/rel/reaxff: without the field, chi_eff = chi of the type. With the field,
   * chi_eff = chi_i + scale (phi_i - sum_j phi_j S_ij / sum_j S_ij), the sums over the atoms j within the cutoff.
   */
  protected override calcChiEff(E: [number, number, number] | null): Float64Array {
    const nb = this.sys.nb;
    const n = nb.nlocal, N = nb.nall;
    const p = this.params!;
    const chiEff = new Float64Array(n);
    if (!E) {
      for (let i = 0; i < n; i++) chiEff[i] = p.chi[nb.typeall[i]];
      return chiEff;
    }
    const x = nb.xall, typeAll = nb.typeall, nt1 = this.ntypes + 1, qe2f = this.sys.state.units.qe2f;
    for (let i = 0; i < n; i++) {
      const chia = p.chi[typeAll[i]];
      const phia = fieldPotential(x, i, E, qe2f);
      let sumN = 0, sumD = 0;
      for (let j = 0; j < N; j++) {
        const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
        const distSq = dx * dx + dy * dy + dz * dz;
        if (distSq < this.distCutSq) {
          const overlap = this.prefactor[typeAll[i] * nt1 + typeAll[j]] * Math.exp(-this.expfactor[typeAll[i] * nt1 + typeAll[j]] * distSq);
          sumN += fieldPotential(x, j, E, qe2f) * overlap;
          sumD += overlap;
        }
      }
      chiEff[i] = sumD !== 0.0 ? chia + this.scale * (phia - sumN / sumD) : chia;
    }
    return chiEff;
  }
}

const intArg = (w: string, what: string, style: string): number => {
  const v = Number(w);
  if (!Number.isInteger(v)) throw new StyleError(`Illegal fix ${style} command: expected an integer for ${what}, got '${w}'`);
  return v;
};
const numArg = (w: string, what: string, style: string): number => {
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Illegal fix ${style} command: expected a number for ${what}, got '${w}'`);
  return v;
};
