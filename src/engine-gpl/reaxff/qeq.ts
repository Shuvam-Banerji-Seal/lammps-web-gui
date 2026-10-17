/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/fix_qeq_reaxff.cpp and src/REAXFF/fix_qeq_reaxff.h (FixQEqReaxFF: constructor,
 * pertype_parameters, init, init_shielding, init_taper, setup_pre_force, init_storage, pre_force,
 * init_matvec, compute_H, calculate_H, CG, sparse_matvec, calculate_Q), at commit b332f28. Authors
 * credited in the C++ file headers: Hasan Metin Aktulga (Purdue University, now Lawrence Berkeley
 * National Laboratory); hybrid and sub-group capabilities: Ray Shan (Sandia). Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Refused (StyleError): the dual and matfree variants. The fix efield term is in efield.ts. The fictitious-charge histories
 * are keyed by atom id (the engine sorts atoms at reneighboring; LAMMPS keeps them with the atoms
 * through the Atom callbacks). The H matrix is built from the owned and ghost positions of the
 * engine (a binned-free O(n N) scan over the owned i and the j after i), with the same flag rule
 * as LAMMPS compute_H, instead of the pair style's neighbor list.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairReaxFF } from './pair';
import { type HistoryMap, solveFictitious, taperFix } from './charge';
import { efieldForModel, fieldOf, findEfield, qeqChiField } from './efield';
import type { FixEfield } from '../../engine/fix/force_ext';
import { pow as libmPow } from '../libm/pow';

/** Per-type QEq parameters, 1-based (chi, eta, gamma). */
export interface QeqParams { chi: Float64Array; eta: Float64Array; gamma: Float64Array }

/**
 * The per-type parameter file of fix qeq/reaxff and fix qtpie/reaxff: one line per type with
 * "itype chi eta gamma" (blank lines skipped), types 1..ntypes.
 */
export const parsePertypeFile = (text: string, nt: number, label: string): QeqParams => {
  const chi = new Float64Array(nt + 1), eta = new Float64Array(nt + 1), gamma = new Float64Array(nt + 1);
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '');
  for (let i = 1; i <= nt; i++) {
    const line = lines[i - 1];
    if (line === undefined) throw new StyleError(`Fix ${label}: Invalid param file format`);
    const w = line.trim().split(/\s+/);
    if (w.length !== 4) throw new StyleError(`Fix ${label}: Incorrect format of param file`);
    const itype = Number(w[0]);
    if (!Number.isInteger(itype) || itype < 1 || itype > nt) throw new StyleError(`Fix ${label}: invalid atom type in param file ${w[0]}`);
    chi[itype] = Number(w[1]);
    eta[itype] = Number(w[2]);
    gamma[itype] = Number(w[3]);
  }
  return { chi, eta, gamma };
};

/** fix qeq/reaxff (FixQEqReaxFF): charge equilibration with the ReaxFF shielded Coulomb matrix. */
export class FixQEqReaxFF extends Fix {
  readonly style = 'qeq/reaxff';

  private swa: number;
  private swb: number;
  private tolerance: number;
  private imax = 200;
  private maxwarn = true;
  private params: QeqParams | null = null;
  private pertype: string;
  private Tap: Float64Array = new Float64Array(8);
  private shld = new Float64Array(0);
  private ntypes = 0;
  /** fictitious-charge histories keyed by atom id: [s_hist 0..3, t_hist 4..7] (most recent first). */
  private hist: HistoryMap = new Map();
  private efieldFix: FixEfield | null = null;
  private matvecs = 0;

  constructor(sys: System, id: string, group: string, args: string[], private pairRef: () => PairReaxFF | null) {
    super(sys, id, group, args);
    this.scalarFlag = true;
    this.extscalar = 0;
    // narg of LAMMPS is 8..12 with "fix ID group style" in front: 5..9 arguments here
    if (args.length < 5 || args.length > 9) throw new StyleError('Illegal fix qeq/reaxff command');
    this.pairRef()?.attachQeqFix();
    this.nevery = intArg(args[0], 'Nevery');
    if (this.nevery <= 0) throw new StyleError('Illegal fix qeq/reaxff command');
    this.swa = numArg(args[1], 'swa');
    this.swb = numArg(args[2], 'swb');
    this.tolerance = numArg(args[3], 'tolerance');
    this.pertype = args[4];
    let iarg = 5;
    while (iarg < args.length) {
      const k = args[iarg];
      if (k === 'nowarn') this.maxwarn = false;
      else if (k === 'maxiter') {
        if (iarg + 1 > args.length - 1) throw new StyleError('Illegal fix qeq/reaxff command: maxiter needs a value');
        this.imax = intArg(args[iarg + 1], 'maxiter');
        iarg++;
      } else if (k === 'dual') throw new StyleError('fix qeq/reaxff keyword dual is not ported (OPENMP variant only)');
      else if (k === 'matfree') throw new StyleError('fix qeq/reaxff keyword matfree is not ported (Kokkos variant only)');
      else throw new StyleError(`Illegal fix qeq/reaxff command: unknown keyword ${k}`);
      iarg++;
    }
  }

  override init(): void {
    const s = this.sys.state;
    const st = this.sys.state;
    let count = 0;
    for (let i = 0; i < st.n; i++) if (st.mask[i] & this.groupBit) count++;
    if (count === 0) throw new StyleError('Fix qeq/reaxff group has no atoms');
    if (this.swb < 0) throw new StyleError('Fix qeq/reaxff has negative upper Taper radius cutoff');
    this.efieldFix = findEfield(this.sys, 'qeq/reaxff');
    if (this.efieldFix) efieldForModel(this.sys, this.efieldFix, 'qeq/reaxff', { allGroup: false, periodicCheck: true });
    this.ntypes = s.ntypes;
    this.Tap = taperFix(this.swa, this.swb);
    this.pertypeParameters();
    const p = this.params!;
    const nt = this.ntypes;
    this.shld = new Float64Array((nt + 1) * (nt + 1));
    for (let i = 1; i <= nt; i++) {
      for (let j = 1; j <= nt; j++) this.shld[i * (nt + 1) + j] = libmPow(p.gamma[i] * p.gamma[j], -1.5);
    }
  }

  /** pertype_parameters: the chi, eta, gamma columns of the file, or of the pair style (keyword reaxff). */
  private pertypeParameters(): void {
    if (this.params) return;
    const nt = this.ntypes;
    if (this.pertype.startsWith('reaxff')) {
      const pair = this.pairRef();
      if (!pair) throw new StyleError('No reaxff pair style for fix qeq/reaxff');
      this.params = pair.qeqParameters(nt);
      return;
    }
    if (this.pertype.startsWith('reax/c')) throw new StyleError("Fix qeq/reaxff keyword 'reax/c' is obsolete; please use 'reaxff'");
    this.params = parsePertypeFile(this.sys.readFile(this.pertype), this.ntypes, 'qeq/reaxff');
  }

  /** setup_pre_force: one pre_force before the setup force evaluation. */
  override setupPreForce(): void { this.preForce(); }
  /** min_pre_force: the solve before every minimization evaluation (the same step rule). */
  override minPreForce(): void { this.preForce(); }

  override preForce(): void {
    if (this.sys.state.step % this.nevery !== 0) return;
    this.solve();
  }

  override computeScalar(): number { return this.matvecs / 2.0; }

  /**
   * pre_force: compute_H, the two CG solves (s with b = -chi, t with b = -1) and calculate_Q
   * (charge.ts solveFictitious, shared with fix qtpie/reaxff).
   */
  private solve(): void {
    if (!this.params) this.init();
    const s = this.sys.state;
    const nb = this.sys.nb;
    const p = this.params!;
    const chiEff = new Float64Array(nb.nlocal);
    for (let i = 0; i < nb.nlocal; i++) chiEff[i] = p.chi[nb.typeall[i]];
    // chi_eff = chi + chi_field (b_s = -chi, then b_s -= chi_field in the C++ init_matvec)
    if (this.efieldFix) {
      const chiField = qeqChiField(this.sys, this.efieldFix, fieldOf(this.sys, this.efieldFix));
      for (let i = 0; i < nb.nlocal; i++) chiEff[i] += chiField[i];
    }
    const r = solveFictitious({
      sys: this.sys, groupBit: this.groupBit, swb: this.swb, ntypes: this.ntypes, shld: this.shld, Tap: this.Tap,
      eta: p.eta, chiEff, hist: this.hist,
      opts: { tolerance: this.tolerance, imax: this.imax, maxwarn: this.maxwarn, warn: (t) => this.sys.warn(t), step: s.step, name: 'qeq/reaxff' },
    });
    this.matvecs = r.matvecsS + r.matvecsT;
  }

  /** The QEq parameters of the fix (for tests and the notebook). */
  qeqParams(): QeqParams | null { return this.params; }
}

const intArg = (w: string, what: string): number => {
  const v = Number(w);
  if (!Number.isInteger(v)) throw new StyleError(`Illegal fix qeq/reaxff command: expected an integer for ${what}, got '${w}'`);
  return v;
};
const numArg = (w: string, what: string): number => {
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Illegal fix qeq/reaxff command: expected a number for ${what}, got '${w}'`);
  return v;
};
