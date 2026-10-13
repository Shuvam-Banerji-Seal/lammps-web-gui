/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/pair_reaxff.cpp and src/REAXFF/pair_reaxff.h (PairReaxFF: settings, coeff,
 * init_style, init_one, setup, compute, write_reax_atoms, read_reax_forces), at commit b332f28.
 * Authors credited in the C++ file headers: Hasan Metin Aktulga (Michigan State University,
 * hma@cse.msu.edu); per-atom energy and virial by Ray Shan (Sandia); hybrid compatibility by
 * Ray Shan (Sandia). Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Refused in this port (StyleError): lgvdw yes, list/blocking, checkqeq yes until the QEq fix is
 * ported, compute stress/atom and compute pe/atom. The memory keywords safezone, mincap and
 * minhbonds only size LAMMPS's buffers and do not change results, so they are accepted.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import {
  allocBondList, buildFarList, computeBondOrders, computeTotalForce, ensureHBondList, initForces,
  newEnergies, newWorkspace, regionCounts, resetHydrogenIndex, resetWorkspace, taperCoefficients,
  type RxSystem,
} from './core';
import { bonds as bondsTerm, atomEnergy, hydrogenBonds, torsionAngles, valenceAngles } from './bonded';
import { readControlFile, readForceField, type ControlParams, type ReaxParam } from './ffield';
import { initLookupTables, type LookupTables } from './lookup';
import { tabulatedVdwCoulombEnergy, vdwCoulombEnergy } from './nonbonded';
import { nativeFarOrder, type NativeFarOrder } from './nativeFar';
import { nativeOrder } from '../../engine/atoms';

const REAX_MIN_CAP = 50;

/** pair_style reaxff (PairReaxFF). Many-body: builds its own far neighbor lists from the owned and ghost positions. */
export class PairReaxFF extends Pair {
  readonly name = 'reaxff';
  manybody = true;
  needsFull = false;
  needsHalf = false;
  virialFdotr = true;

  protected ctrl: ControlParams = {
    tabulate: 0, bondCut: 5.0, nonbCut: 0, nonbLow: 0, hbondCut: 7.5, bgCut: 0.3, boCut: 0,
    thbCut: 0.001, thbCutsq: 0.00001, lgflag: 0, enobondsflag: 1,
  };
  protected qeqflag = 1;
  /** set by fix qeq/reaxff (attachQeqFix): the charges of checkqeq yes come from it. */
  protected qeqAttached = false;
  /** set by fix acks2/reaxff (attachAcks2): its solution s feeds the ACKS2 bond-softness terms. */
  protected acks2Source: (() => Float64Array) | null = null;
  protected fileRead = false;
  protected param: ReaxParam | null = null;
  /** element index (into param.sbp) per LAMMPS atom type; -1 for NULL. */
  protected map = new Int32Array(0);
  protected Tap: Float64Array<ArrayBufferLike> = new Float64Array(8);
  protected lookup: LookupTables | null = null;
  protected lookupKey = '';
  protected cutmax = 0;
  /** LAMMPS pvector: the 14 energy terms of the last evaluation (compute pair reaxff). */
  protected pvector = new Float64Array(14);
  /** Set by fix reaxff/bonds and fix reaxff/species: keep the bond list and the native order of the last evaluation. */
  nativeWanted = false;
  /** The bond list of the last evaluation (with the state the output fixes read); null before the first one. */
  lastBonds: RxSystem | null = null;
  /** Native far-list order of the current neighbor build (computed when nativeWanted). */
  nativeInfo: NativeFarOrder | null = null;
  private nativeBuild = -1;
  private nativeOrderRef: Int32Array | null = null;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.map = new Int32Array(ntypes + 1).fill(-1);
  }

  override settings(args: string[], ctx: StyleContext): void {
    if (args.length < 1) throw new StyleError('Illegal pair_style reaxff command');
    const ctrl = this.ctrl;
    if (args[0] === 'NULL') {
      ctrl.tabulate = 0;
      ctrl.bondCut = 5.0;
      ctrl.hbondCut = 7.5;
      ctrl.thbCut = 0.001;
      ctrl.thbCutsq = 0.00001;
      ctrl.bgCut = 0.3;
    } else {
      const parsed = readControlFile(ctx.readFile(args[0]), args[0]);
      ctrl.tabulate = 0;
      ctrl.bondCut = 5.0;
      ctrl.bgCut = 0.3;
      ctrl.thbCut = 0.001;
      ctrl.thbCutsq = 0.00001;
      ctrl.hbondCut = 7.5;
      Object.assign(ctrl, parsed);
    }
    this.qeqflag = 1;
    ctrl.lgflag = 0;
    ctrl.enobondsflag = 1;
    let iarg = 1;
    const word = (w: string | undefined, what: string): string => {
      if (w === undefined) throw new StyleError(`Illegal pair_style reaxff command: missing value for ${what}`);
      return w;
    };
    const logical = (w: string): boolean => {
      if (w === 'yes') return true;
      if (w === 'no') return false;
      throw new StyleError(`Expected boolean parameter instead of '${w}' in pair_style reaxff command`);
    };
    while (iarg < args.length) {
      const key = args[iarg];
      const val = args[iarg + 1];
      switch (key) {
        case 'checkqeq': this.qeqflag = logical(word(val, key)) ? 1 : 0; iarg += 2; break;
        case 'enobonds': ctrl.enobondsflag = logical(word(val, key)) ? 1 : 0; iarg += 2; break;
        case 'lgvdw':
          if (logical(word(val, key))) throw new StyleError('pair_style reaxff keyword lgvdw yes is not ported (lgvdw is not supported)');
          iarg += 2;
          break;
        case 'safezone': {
          const sz = parseNum(word(val, key), 'safezone');
          if (sz < 0) throw new StyleError('Illegal pair_style reaxff safezone command');
          iarg += 2;
          break;
        }
        case 'mincap': {
          const mc = Math.trunc(parseNum(word(val, key), 'mincap'));
          if (mc < 0) throw new StyleError('Illegal pair_style reaxff mincap command');
          void REAX_MIN_CAP;
          iarg += 2;
          break;
        }
        case 'minhbonds': {
          const mh = Math.trunc(parseNum(word(val, key), 'minhbonds'));
          if (mh < 0) throw new StyleError('Illegal pair_style reaxff minhbonds command');
          iarg += 2;
          break;
        }
        case 'list/blocking':
          throw new StyleError('pair_style reaxff keyword list/blocking is not ported');
        case 'tabulate': {
          const t = Math.trunc(parseNum(word(val, key), 'tabulate'));
          if (t < 0) throw new StyleError('Illegal pair_style reaxff tabulate command');
          ctrl.tabulate = t;
          iarg += 2;
          break;
        }
        default:
          throw new StyleError(`Illegal pair_style reaxff command: keyword ${key} is not ported (acks2, qtpie, qeq/rel and other fixes are not supported)`);
      }
    }
    this.version++;
    this.lookup = null;
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 3 + this.ntypes) throw new StyleError('Incorrect args for pair coefficients');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style reaxff must be * *');
    const { param, control } = readForceField(ctx.readFile(args[2]), args[2], this.ctrl.lgflag);
    this.param = param;
    Object.assign(this.ctrl, control);
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    let matched = 0;
    for (let i = 3; i < args.length; i++) {
      const w = args[i];
      if (w === 'NULL') {
        map[i - 2] = -1;
        matched++;
        continue;
      }
      const lw = w.toLowerCase();
      const k = param.sbp.findIndex((s) => s.name.toLowerCase() === lw);
      if (k >= 0) {
        map[i - 2] = k;
        matched++;
      }
    }
    if (matched !== this.ntypes) throw new StyleError('Non-existent ReaxFF type');
    this.map = map;
    this.fileRead = true;
    this.version++;
    this.lookup = null;
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.fileRead) throw new StyleError('pair_style reaxff needs a pair_coeff command with a ReaxFF force field file');
    const c = this.ctrl;
    this.cutmax = Math.max(c.nonbCut, c.hbondCut, c.bondCut);
    if (this.param) this.Tap = taperCoefficients(c.nonbLow, c.nonbCut);
  }

  override initOne(_i: number, _j: number): number {
    return this.cutmax;
  }

  override compute(pc: PairCompute): void {
    if (!this.fileRead || !this.param) throw new StyleError('pair_style reaxff has no force field');
    if (pc.vatom || pc.eatom) throw new StyleError('compute pe/atom and stress/atom are not supported by pair style reaxff');
    // checkqeq yes: the charges come from fix qeq/reaxff (pair_reaxff.cpp init_style requires one)
    if (this.qeqflag && !this.qeqAttached) {
      throw new StyleError('pair_style reaxff requires use of exactly one of the fix qeq/reaxff or fix qeq/shielded or fix acks2/reaxff or fix qtpie/reaxff or fix qeq/rel/reaxff commands (or use checkqeq no)');
    }
    const param = this.param;
    const ctrl = this.ctrl;
    const n = pc.nlocal, N = pc.nall;
    const type = new Int32Array(N);
    for (let a = 0; a < N; a++) type[a] = this.map[pc.type[a]] ?? -1;
    const orig = new Int32Array(N);
    const owner = pc.nb.owner;
    for (let a = 0; a < N; a++) orig[a] = pc.s.id[a < n ? a : owner[a]];
    const x = pc.x;
    const q = pc.q;
    this.Tap = taperCoefficients(ctrl.nonbLow, ctrl.nonbCut);
    const far = buildFarList(ctrl, n, N, x, type);
    const { Hindex, numH } = resetHydrogenIndex(type, n, param, ctrl.hbondCut);
    const counts = regionCounts(far, N, n, type, Hindex, numH, param, ctrl.hbondCut);
    const sys: RxSystem = {
      ctrl, param, n, N, type, orig, x, q, Hindex, numH, Tap: this.Tap, far,
      acks2s: this.acks2Source ? this.acks2Source() : null,
      bonds: allocBondList(N, far, counts.bondCount),
      ws: newWorkspace(N),
      en: newEnergies(),
      thb: { start: new Int32Array(0), end: new Int32Array(0), thb: new Int32Array(0), pthb: new Int32Array(0), theta: new Float64Array(0), dcos_di: new Float64Array(0), dcos_dj: new Float64Array(0), dcos_dk: new Float64Array(0), count: 0 },
      hb: ensureHBondList(counts.hbCount, numH),
    };
    resetWorkspace(sys.ws);
    initForces(sys);
    computeBondOrders(sys);
    bondsTerm(sys);
    atomEnergy(sys);
    valenceAngles(sys);
    torsionAngles(sys);
    if (ctrl.hbondCut > 0) hydrogenBonds(sys);
    if (ctrl.tabulate === 0) {
      vdwCoulombEnergy(sys);
    } else {
      tabulatedVdwCoulombEnergy(sys, this.tables(type, n));
    }
    computeTotalForce(sys);
    this.lastBonds = sys;
    if (this.nativeWanted && (this.nativeInfo === null || this.nativeBuild !== pc.nb.nbuild || this.nativeOrderRef !== nativeOrder(pc.s))) {
      if (pc.s.dimension === 2) throw new StyleError('fix reaxff/bonds and fix reaxff/species are not supported in 2d');
      this.nativeInfo = nativeFarOrder(pc.s, pc.nb, pc.geom, x);
      this.nativeBuild = pc.nb.nbuild;
      this.nativeOrderRef = nativeOrder(pc.s);
    }
    // read_reax_forces: LAMMPS adds -f to the forces of owned and ghost atoms (ghost forces are reverse-communicated)
    const f = pc.f;
    for (let a = 0; a < N; a++) {
      f[3 * a] += -sys.ws.f[3 * a];
      f[3 * a + 1] += -sys.ws.f[3 * a + 1];
      f[3 * a + 2] += -sys.ws.f[3 * a + 2];
    }
    const e = sys.en;
    const pv = this.pvector;
    pv[0] = e.e_bond;
    pv[1] = e.e_ov + e.e_un;
    pv[2] = e.e_lp;
    pv[3] = 0.0;
    pv[4] = e.e_ang;
    pv[5] = e.e_pen;
    pv[6] = e.e_coa;
    pv[7] = e.e_hb;
    pv[8] = e.e_tor;
    pv[9] = e.e_con;
    pv[10] = e.e_vdW;
    pv[11] = e.e_ele;
    pv[12] = 0.0;
    pv[13] = e.e_pol;
    pc.acc.evdwl += e.e_bond + e.e_ov + e.e_un + e.e_lp + e.e_ang + e.e_pen + e.e_coa + e.e_hb + e.e_tor + e.e_con + e.e_vdW;
    pc.acc.ecoul += e.e_ele + e.e_pol;
  }

  /** Called by fix reaxff/bonds and fix reaxff/species (init): the next evaluations keep the native order. */
  wantNative(): void { this.nativeWanted = true; }

  /** The element name of each LAMMPS type 1..ntypes (index t-1), as pair_reaxff.cpp eletype: "NULL" for NULL types. */
  elementNames(ntypes: number): string[] {
    if (!this.param) throw new StyleError('pair_style reaxff has no force field');
    const names: string[] = [];
    for (let t = 1; t <= ntypes; t++) {
      const k = this.map[t];
      names.push(k >= 0 ? this.param.sbp[k].name : 'NULL');
    }
    return names;
  }

  /** bond_graph_cutoff of the control file (control->bg_cut): the bond order cutoff of fix reaxff/bonds. */
  bondGraphCutoff(): number { return this.ctrl.bgCut; }

  /** extract('pvector') for compute pair (the 14 terms of pair_reaxff.cpp compute). */
  override extract(name: string): unknown {
    return name === 'pvector' ? this.pvector : undefined;
  }

  /** Called by fix acks2/reaxff: the solution s of the fix (read at each evaluation). */
  attachAcks2(source: () => Float64Array): void { this.acks2Source = source; this.qeqAttached = true; }

  /** pair_style reaxff extract() of fix acks2/reaxff: bcut_acks2 per type (1-based) and gp.l[34] (bond softness). */
  acks2Parameters(ntypes: number): { bcut: Float64Array; bondSoftness: number } {
    if (!this.param) throw new StyleError('pair_style reaxff has no force field');
    const bcut = new Float64Array(ntypes + 1);
    for (let t = 1; t <= ntypes; t++) {
      const k = this.map[t];
      if (k >= 0) bcut[t] = this.param.sbp[k].bcut_acks2;
    }
    if (this.param.gpL.length <= 34) throw new StyleError('pair_style reaxff: the force field has no bond softness parameter (gp.l[34])');
    return { bcut, bondSoftness: this.param.gpL[34] };
  }

  /** Called by fix qeq/reaxff when it is defined or initialised. */
  attachQeqFix(): void { this.qeqAttached = true; }

  /** pair_style reaxff extract(): chi, eta, gamma of each atom type (1-based), as fix qeq/reaxff reads them. */
  qeqParameters(ntypes: number): { chi: Float64Array; eta: Float64Array; gamma: Float64Array } {
    if (!this.param) throw new StyleError('pair_style reaxff has no force field');
    const chi = new Float64Array(ntypes + 1), eta = new Float64Array(ntypes + 1), gamma = new Float64Array(ntypes + 1);
    for (let t = 1; t <= ntypes; t++) {
      const k = this.map[t];
      if (k >= 0) {
        chi[t] = this.param.sbp[k].chi;
        eta[t] = this.param.sbp[k].eta;
        gamma[t] = this.param.sbp[k].gamma;
      }
    }
    return { chi, eta, gamma };
  }

  /** Lookup tables for the present types (Init_Lookup_Tables), built once per setting. */
  protected tables(type: Int32Array, n: number): LookupTables {
    const param = this.param!;
    const present = new Uint8Array(param.numAtomTypes);
    for (let a = 0; a < n; a++) if (type[a] >= 0) present[type[a]] = 1;
    const key = `${this.version}:${Array.from(present).join('')}`;
    if (!this.lookup || this.lookupKey !== key) {
      this.lookup = initLookupTables(param, this.ctrl, this.Tap, present);
      this.lookupKey = key;
    }
    return this.lookup;
  }
}
