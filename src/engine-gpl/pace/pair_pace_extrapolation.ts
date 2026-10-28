/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ML-PACE/pair_pace_extrapolation.cpp
 * and pair_pace_extrapolation.h (pair_style pace/extrapolation: settings, coeff, init_style, init_one, compute, extract
 * and extract_peratom). Copyright 2022 Yury Lysogorskiy, Ralf Drautz (Ruhr-University Bochum) and the LAMMPS Sandia
 * copyright notice (GNU General Public License). The evaluators it calls are ported from the ML-PACE library (see
 * ./bbasis.ts, ./bevaluator.ts and ./evaluator.ts). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_pace.html. "pair_style pace/extrapolation" with "pair_coeff * * file.yaml file.asi elem1
 * ... elemN": the B-basis potential and its active set inverse. The extrapolation grade gamma is computed only on the
 * steps fix pair asks for it ("fix ID all pair N pace/extrapolation gamma 1"): those steps use the B-basis evaluator,
 * the others the recursive C-tilde evaluator of pair_style pace.
 */
import { Pair, StyleError, type AdaptAccess, type PairCompute, type StyleContext } from '../../engine/force/types';
import { scaleAccess, unitScaleTable } from '../../engine/force/pair/scaleTable';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { BBasis } from './bbasis';
import { ACEBEvaluator } from './bevaluator';
import { ACEPaceEvaluator } from './evaluator';
import { fileBytes } from './npz';
import { ELEMENTS_PACE } from './pair_pace';

export class PairPaceExtrapolation extends Pair {
  override readonly name: string = 'pace/extrapolation';
  manybody = true;
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;

  private map = new Int32Array(0);
  private bbasis: BBasis | null = null;
  private ace: ACEBEvaluator | null = null;
  private recAce: ACEPaceEvaluator | null = null;
  private readonly natOrder = new NativeFullOrder();
  private setflagAll = false;
  /** flag_compute_extrapolation_grade and flag_corerep_factor: the triggers fix pair sets (extract "gamma_flag", "corerep_flag") */
  flagComputeExtrapolationGrade = 0;
  flagCorerepFactor = 0;
  /** extrapolation_grade_gamma and corerep_factor by owned atom (null until a step computes them) */
  gamma: Float64Array | null = null;
  corerep: Float64Array | null = null;
  /** fix adapt's scale by type pair, as in pair_style pace */
  scale: Float64Array = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.scale = unitScaleTable(ntypes);
    this.map = new Int32Array(ntypes + 1).fill(-1);
    this.bbasis = null;
    this.ace = null;
    this.recAce = null;
    this.setflagAll = false;
  }

  /** PairPACEExtrapolation::settings: [chunksize N] [neigh auto|shared|global] (messages as native). */
  override settings(args: string[], ctx: StyleContext): void {
    if (args.length > 4) throw new StyleError('Illegal pair_style pace/extrapolation command: missing argument(s)');
    if (ctx.s && ctx.s.units.style !== 'metal') throw new StyleError("ACE potentials require 'metal' units");
    let iarg = 0;
    while (iarg < args.length) {
      const w = args[iarg];
      if (w === 'chunksize') {
        const v = args[iarg + 1];
        if (v === undefined || v === '') throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
        if (!/^[+-]?\d+$/.test(v)) throw new StyleError(`Expected integer parameter instead of '${v}' in input script or data file`);
        iarg += 2;
      } else if (w === 'neigh') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal pair_style pace/extrapolation neigh command: missing argument(s)');
        const v = args[iarg + 1];
        if (v !== 'auto' && v !== 'shared' && v !== 'global') throw new StyleError(`Unknown pair_style pace/extrapolation neigh keyword: ${v}`);
        iarg += 2;
      } else {
        throw new StyleError(`Unknown pair_style pace keyword: ${w}`);
      }
    }
  }

  /** PairPACEExtrapolation::coeff: "pair_coeff * * potential.yaml active_set.asi elem1 ... elemN". */
  override coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 5) throw new StyleError('Illegal pair_coeff command: missing argument(s)');
    const nt = this.ntypes;
    if (nt === 0) throw new StyleError('Pair_coeff command before simulation box is defined');
    // pair.cpp map_element2type(narg - 4, arg + 4)
    if (args.length - 4 !== nt) throw new StyleError('Number of element to type mappings does not match number of atom types');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Incorrect args for pair coefficients');
    const fileName = args[2], asiName = args[3];
    // a missing file: utils::get_potential_file_path gives '', which Input::parse_input and npz_load refuse
    const bbasis = BBasis.fromYaml(ctx.readFile(fileName, () => "Potential file  doesn't exists"), fileName);
    const ctilde = bbasis.toCTilde();
    const map = new Int32Array(nt + 1).fill(-1);
    for (let i = 1; i <= nt; i++) {
      const elem = args[3 + i];
      if (elem === 'NULL') continue;
      if (ELEMENTS_PACE.indexOf(elem) < 1) throw new StyleError(`'${elem}' is not a valid element`);
      const mu = bbasis.speciesIndexByName(elem);
      if (mu === -1) throw new StyleError(`Element ${elem} is not supported by ACE-potential from file ${fileName}`);
      map[i] = mu;
    }
    const ace = new ACEBEvaluator(bbasis, map);
    ace.loadActiveSet(fileBytes(ctx.readFile(asiName, () => 'npz_load: Error! Unable to open file !')), asiName);
    ctx.log(ace.isLinearExtrapolationGrade ? 'LINEAR ASI is loaded' : 'FULL ASI is loaded');
    this.bbasis = bbasis;
    this.map = map;
    this.ace = ace;
    this.recAce = new ACEPaceEvaluator(ctilde, true, map);
    this.setflagAll = true;
    this.scale.fill(1);
  }

  override adaptParam(name: string): AdaptAccess | null {
    return name === 'scale' ? scaleAccess(this) : null;
  }

  /** init_style: newton pair on (the engine's atoms always have IDs); a full neighbor list. */
  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError('Pair style pace/extrapolation requires newton pair on');
  }

  /** init_one: the radial cutoff of the species pair. */
  override initOne(i: number, j: number): number {
    if (!this.setflagAll || !this.bbasis) throw new StyleError('All pair coeffs are not set');
    const nt = this.ntypes + 1;
    this.scale[j * nt + i] = this.scale[i * nt + j];
    const mi = this.map[i], mj = this.map[j];
    if (mi < 0 || mj < 0) return 0;
    return this.bbasis.radial.cut[mi * this.bbasis.nelements + mj];
  }

  /** extract("gamma_flag" / "corerep_flag"): the scalar triggers fix pair sets. */
  hasTrigger(name: string): boolean {
    return name === 'gamma_flag' || name === 'corerep_flag';
  }

  setTrigger(name: string, value: number): boolean {
    if (name === 'gamma_flag') this.flagComputeExtrapolationGrade = value;
    else if (name === 'corerep_flag') this.flagCorerepFactor = value;
    else return false;
    return true;
  }

  /** extract_peratom("gamma" / "corerep"): a per-atom vector (null before the first step that computes it). */
  extractPeratom(name: string): { values: Float64Array | null; columns: number } | undefined {
    if (name === 'gamma') return { values: this.gamma, columns: 0 };
    if (name === 'corerep') return { values: this.corerep, columns: 0 };
    return undefined;
  }

  /** PairPACEExtrapolation::compute: the B evaluator on grade steps, the recursive one otherwise (full list). */
  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    if (!this.ace || !this.recAce) throw new StyleError('All pair coeffs are not set');
    const grade = this.flagComputeExtrapolationGrade !== 0;
    const nlocal = pc.nlocal;
    // the per-atom arrays grow (and restart from zero) when there are more owned atoms than before
    if (grade && nlocal > (this.gamma?.length ?? 0)) this.gamma = new Float64Array(nlocal);
    if (this.flagCorerepFactor && nlocal > (this.corerep?.length ?? 0)) this.corerep = new Float64Array(nlocal);
    const x = pc.x, f = pc.f, type = pc.type;
    const nbrs = this.natOrder.neighbors(pc.s, pc.nb, pc.geom, list);
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      const jl = new Int32Array(jnum);
      for (let jj = 0; jj < jnum; jj++) jl[jj] = nbrs[k0 + jj] & NEIGHMASK;
      let nf: Float64Array, eAtom: number, fcut: number;
      if (grade) {
        this.ace.computeProjections = true;
        this.ace.computeAtom(i, x, type, jnum, jl);
        this.gamma![i] = this.ace.maxGammaGrade;
        nf = this.ace.neighboursForces;
        eAtom = this.ace.eAtom;
        fcut = this.ace.aceFcut;
      } else {
        this.recAce.computeAtom(i, x, type, jnum, jl, 0);
        nf = this.recAce.neighboursForces;
        eAtom = this.recAce.eAtom;
        fcut = this.recAce.aceFcut;
      }
      if (this.flagCorerepFactor) this.corerep![i] = 1 - fcut;
      const nt1 = this.ntypes + 1;
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let jj = 0; jj < jnum; jj++) {
        const j = jl[jj];
        const sc = this.scale[type[i] * nt1 + type[j]];
        const fx = sc * nf[3 * jj], fy = sc * nf[3 * jj + 1], fz = sc * nf[3 * jj + 2];
        f[3 * i] += fx;
        f[3 * i + 1] += fy;
        f[3 * i + 2] += fz;
        f[3 * j] -= fx;
        f[3 * j + 1] -= fy;
        f[3 * j + 2] -= fz;
        if (pc.vatom) {
          const dx = -(x[3 * j] - xtmp), dy = -(x[3 * j + 1] - ytmp), dz = -(x[3 * j + 2] - ztmp);
          const v = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
          const va = pc.vatom;
          for (let c = 0; c < 6; c++) {
            va[6 * i + c] += 0.5 * v[c];
            va[6 * j + c] += 0.5 * v[c];
          }
        }
      }
      const evdwl = this.scale[type[i] * nt1 + type[i]] * eAtom;
      pc.acc.evdwl += evdwl;
      if (pc.eatom) pc.eatom[i] += evdwl;
    }
  }
}
