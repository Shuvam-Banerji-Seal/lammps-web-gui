/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/fix_tgnh_drude.cpp and
 * src/DRUDE/fix_tgnh_drude.h (class FixTGNHDrude, the temperature-grouped Nose-Hoover base of fix tgnvt/drude and
 * fix tgnpt/drude). Copyright (2003) Sandia Corporation; GPL-2.0 (src/engine-gpl/LICENSE). DRUDE package authors:
 * Alain Dequidt, Julien Devemy, Agilio Padua; contributors Kateryna Goloviznina and Zheng Gong.
 *
 * Nose-Hoover chains on three groups of degrees of freedom of a Drude model (docs.lammps.org/fix_tgnh_drude.html):
 * the centres of mass of the molecules, the internal motion of the atoms (core and non-polarizable atoms relative to
 * their molecule, the core-Drude pair centre relative to the molecule), and the relative motion of each core-Drude
 * pair. Syntax: fix ID group tgnvt/drude temp Tstart Tstop Tperiod Tdrude Tdrude_period, and for fix tgnpt/drude the
 * barostat keywords of fix nh (iso, aniso, tri, x, y, z, xy, yz, xz, couple, tchain, pchain, mtk, tloop, ploop, nreset,
 * scalexy, scalexz, scaleyz, flip, fixedpoint). The deviatoric terms of the x ... xz keywords are the ones of fix nh
 * (compute_sigma, compute_deviatoric and compute_strain_energy are the same code; ../fix/nh_deviatoric.ts). With flip
 * yes, a tilt past 0.6 of its box length flips the box on a neighbor rebuild step (pre_exchange, through sys.flipBox).
 */
import { Fix } from '../../engine/fix/fix';
import { ownCompute, ramp, removeCompute } from '../../engine/fix/util';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { inumeric, logical, missingArgs, numeric } from '../../engine/force/util';
import { massOf } from '../../engine/atoms';
import { CORE_TYPE, DRUDE_TYPE, NOPOL_TYPE, drudeModelOf, type DrudeModel } from './drude_model';
import { exp as libmExp } from '../libm/exp';
import { boxHInv, computeDeviatoric, computeSigma, computeStrainEnergy } from '../fix/nh_deviatoric';

/** fix_tgnh_drude.cpp reads its numbers with utils::numeric and utils::inumeric (`what` documents the call site). */
const num = (w: string | undefined, _what: string): number => numeric(w);
const int = (w: string | undefined, _what: string): number => inumeric(w);

/** fix_tgnh_drude.cpp: a flip is triggered when a tilt exceeds 0.5 + DELTAFLIP of its box length; TILTMAX per step. */
const DELTAFLIP = 0.1;
const TILTMAX = 1.5;

export class FixTGNHDrude extends Fix {
  readonly style: string;
  private readonly tstat: boolean;
  private readonly pstat: boolean;
  private tStart = 0; private tStop = 0;
  private tdrudeTarget = 0;
  private tTarget = 0; private tFreq = 0; private tdrudeFreq = 0;
  private mtchain = 3; private mpchain = 3; private ncTchain = 1; private ncPchain = 1;
  private mtkFlag = true;
  // barostat: the six components x y z yz xz xy (Voigt order) of fix nh
  private pStart = new Float64Array(6); private pStop = new Float64Array(6);
  private pFlag = [false, false, false, false, false, false];
  private pTarget = new Float64Array(6); private pCurrent = new Float64Array(6); private pFreq = new Float64Array(6);
  private pHydro = 0; private pFreqMax = 0;
  private pcouple: 'none' | 'xyz' | 'xy' | 'yz' | 'xz' = 'none';
  private pstyle: 'iso' | 'aniso' | 'tri' = 'iso';
  private pdim = 0;
  private deviatoric = false;
  private nresetH0 = 0;
  private scalexy = false; private scalexz = false; private scaleyz = false;
  /** flip yes with a tilt barostat or a tilted box (native pre_exchange_flag): the box flips when a tilt needs it */
  private flipCheck = false;
  /** the reference volume and cell (vol0, h0_inv), taken at the first init or by nreset */
  private vol0 = 0;
  private h0Inv: number[] = [0, 0, 0, 0, 0, 0];
  private sigma: number[] = [0, 0, 0, 0, 0, 0];
  private readonly fixedpoint: number[];
  private omegaDot = new Float64Array(6);
  private omegaMass = new Float64Array(6);
  private mtkTerm1 = 0; private mtkTerm2 = 0;
  // chains: index 0..M-1 used, M is the dummy eta_dot of the chain end
  private etamol: Float64Array; private etamolDot: Float64Array; private etamolDotdot: Float64Array; private etamolMass: Float64Array;
  private etaint: Float64Array; private etaintDot: Float64Array; private etaintDotdot: Float64Array; private etaintMass: Float64Array;
  private etadrude: Float64Array; private etadrudeDot: Float64Array; private etadrudeDotdot: Float64Array; private etadrudeMass: Float64Array;
  private etap: Float64Array; private etapDot: Float64Array; private etapDotdot: Float64Array; private etapMass: Float64Array;
  // degrees of freedom and temperatures of the three groups
  private nMol = 0;
  private massMol = new Float64Array(0);
  private vMol = new Float64Array(0);
  private vMolTmp = new Float64Array(0);
  private dofMol = 0; private dofInt = 0; private dofDrude = 0; private tdof = 0;
  private tMol = 0; private tInt = 0; private tDrude = 0;
  private ke2mol = 0; private ke2int = 0; private ke2drude = 0;
  private ke2molTarget = 0; private ke2intTarget = 0; private ke2drudeTarget = 0;
  private factorMol = 1; private factorInt = 1; private factorDrude = 1;
  private tempComputedEnd = false;
  private tCurrent = 0;
  // time steps
  private dtv = 0; private dtf = 0; private dthalf = 0; private dt4 = 0; private dt8 = 0;
  private tempId: string;
  private pressId: string | null = null;
  private tempCompute!: Compute;
  private pressCompute: Compute | null = null;
  private model!: DrudeModel;

  constructor(sys: System, id: string, group: string, args: string[], style: 'tgnvt/drude' | 'tgnpt/drude') {
    super(sys, id, group, args);
    this.style = style;
    this.timeIntegrate = true;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.fixedpoint = [0, 1, 2].map((d) => 0.5 * (sys.state.box.lo[d] + sys.state.box.hi[d]));
    const dimension = sys.dimension;
    const box0 = sys.state.box, g0 = sys.geom;
    // turn on tilt factor scaling, whenever applicable
    this.scalexy = box0.periodic[1] && g0.xy !== 0;
    if (box0.periodic[2] && dimension === 3) {
      this.scaleyz = g0.yz !== 0;
      this.scalexz = g0.xz !== 0;
    }
    if (args.length < 1) throw new StyleError(`Illegal fix ${style} command`);
    let tdrudeT = 0;
    let tstat = false;
    let tPeriodSet = 0, tdrudePeriodSet = 0;
    let flipflag = true;
    const pPeriod = new Float64Array(6);
    const setP = (d: number, k: number): void => {
      if (k + 4 > args.length) throw missingArgs(`fix ${style} ${args[k]}`);
      this.pStart[d] = num(args[k + 1], 'Pstart');
      this.pStop[d] = num(args[k + 2], 'Pstop');
      pPeriod[d] = num(args[k + 3], 'Pdamp');
      this.pFlag[d] = true;
    };
    const unset = (d: number): void => {
      this.pStart[d] = this.pStop[d] = pPeriod[d] = 0;
      this.pFlag[d] = false;
    };
    for (let k = 0; k < args.length;) {
      const key = args[k];
      if (key === 'temp') {
        if (k + 6 > args.length) throw missingArgs(`fix ${style} temp`);
        tstat = true;
        this.tStart = num(args[k + 1], 'Tstart');
        this.tTarget = this.tStart;
        this.tStop = num(args[k + 2], 'Tstop');
        tPeriodSet = num(args[k + 3], 'Tperiod');
        if (this.tStart <= 0) throw new StyleError(`Target temperature for fix ${style} cannot be <= 0.0`);
        if (this.tStop <= 0) throw new StyleError(`Target temperature for fix ${style} cannot be <= 0.0`);
        tdrudeT = num(args[k + 4], 'Tdrude');
        tdrudePeriodSet = num(args[k + 5], 'Tdrude period');
        if (tdrudeT <= 0) throw new StyleError(`Target Drude temperature for fix ${style} cannot be <= 0.0`);
        k += 6;
      } else if (key === 'iso' || key === 'aniso' || key === 'tri') {
        this.pcouple = key === 'iso' ? 'xyz' : 'none';
        if (key === 'tri') this.scalexy = this.scalexz = this.scaleyz = false;
        for (let d = 0; d < 3; d++) setP(d, k);
        if (key === 'tri') {
          for (let d = 3; d < 6; d++) {
            this.pStart[d] = this.pStop[d] = 0;
            pPeriod[d] = pPeriod[0];
            this.pFlag[d] = true;
          }
        }
        if (dimension === 2) {
          unset(2);
          if (key === 'tri') { unset(3); unset(4); }
        }
        k += 4;
      } else if (key === 'x' || key === 'y' || key === 'z' || key === 'yz' || key === 'xz' || key === 'xy') {
        const d = ['x', 'y', 'z', 'yz', 'xz', 'xy'].indexOf(key);
        setP(d, k);
        this.deviatoric = true;
        if (key === 'yz') this.scaleyz = false;
        if (key === 'xz') this.scalexz = false;
        if (key === 'xy') this.scalexy = false;
        k += 4;
        if (dimension === 2 && (key === 'z' || key === 'yz' || key === 'xz')) throw new StyleError(`Invalid fix ${style} command for a 2d simulation`);
      } else if (key === 'couple') {
        const c = args[k + 1];
        if (c !== 'xyz' && c !== 'xy' && c !== 'yz' && c !== 'xz' && c !== 'none') throw new StyleError(`Illegal fix ${style} couple option: ${c ?? ''}`);
        this.pcouple = c;
        k += 2;
      } else if (key === 'tchain' || key === 'pchain' || key === 'tloop' || key === 'ploop' || key === 'nreset') {
        const n = int(args[k + 1], key);
        if (key === 'tchain' ? n < 1 : n < 0) throw new StyleError(`Invalid fix ${style} ${key} argument: ${n}`);
        if (key === 'tchain') this.mtchain = n;
        else if (key === 'pchain') this.mpchain = n;
        else if (key === 'tloop') this.ncTchain = n;
        else if (key === 'ploop') this.ncPchain = n;
        else this.nresetH0 = n;
        k += 2;
      } else if (key === 'mtk') {
        this.mtkFlag = logical(args[k + 1]);
        k += 2;
      } else if (key === 'scalexy' || key === 'scalexz' || key === 'scaleyz') {
        const v = logical(args[k + 1]);
        if (key === 'scalexy') this.scalexy = v;
        else if (key === 'scalexz') this.scalexz = v;
        else this.scaleyz = v;
        k += 2;
      } else if (key === 'flip') {
        flipflag = logical(args[k + 1]);
        k += 2;
      } else if (key === 'fixedpoint') {
        if (k + 4 > args.length) throw missingArgs(`fix ${style} fixedpoint`);
        for (let d = 0; d < 3; d++) this.fixedpoint[d] = num(args[k + 1 + d], 'fixedpoint');
        k += 4;
      } else {
        throw new StyleError(`Unknown fix ${style} keyword: ${key}`);
      }
    }
    // error checks (fix_tgnh_drude.cpp constructor, in its order)
    const f = this.pFlag, per = box0.periodic, c = this.pcouple;
    const bad = (msg: string): never => { throw new StyleError(msg); };
    if (dimension === 2 && (f[2] || f[3] || f[4])) bad(`Invalid fix ${style} command for a 2d simulation`);
    if (dimension === 2 && (c === 'yz' || c === 'xz')) bad(`Invalid fix ${style} command for a 2d simulation`);
    if (dimension === 2 && (this.scalexz || this.scaleyz)) bad(`Invalid fix ${style} command for a 2d simulation`);
    if (c === 'xyz' && (!f[0] || !f[1])) bad(`Invalid fix ${style} command pressure settings`);
    if (c === 'xyz' && dimension === 3 && !f[2]) bad(`Invalid fix ${style} command pressure settings`);
    if (c === 'xy' && (!f[0] || !f[1])) bad(`Invalid fix ${style} command pressure settings`);
    if (c === 'yz' && (!f[1] || !f[2])) bad(`Invalid fix ${style} command pressure settings`);
    if (c === 'xz' && (!f[0] || !f[2])) bad(`Invalid fix ${style} command pressure settings`);
    if (f[0] && !per[0]) bad(`Cannot use fix ${style} on a non-periodic x dimension`);
    if (f[1] && !per[1]) bad(`Cannot use fix ${style} on a non-periodic y dimension`);
    if (f[2] && !per[2]) bad(`Cannot use fix ${style} on a non-periodic z dimension`);
    if ((f[3] || f[4]) && !per[2]) bad(`Cannot use fix ${style} on a 2nd non-periodic dimension`);
    if (f[5] && !per[1]) bad(`Cannot use fix ${style} on a 2nd non-periodic dimension`);
    if (this.scaleyz && !per[2]) bad(`Cannot use fix ${style} with yz scaling when z is non-periodic dimension`);
    if (this.scalexz && !per[2]) bad(`Cannot use fix ${style} with xz scaling when z is non-periodic dimension`);
    if (this.scalexy && !per[1]) bad(`Cannot use fix ${style} with xy scaling when y is non-periodic dimension`);
    if (f[3] && this.scaleyz) bad(`Cannot use fix ${style} with both yz dynamics and yz scaling`);
    if (f[4] && this.scalexz) bad(`Cannot use fix ${style} with both xz dynamics and xz scaling`);
    if (f[5] && this.scalexy) bad(`Cannot use fix ${style} with both xy dynamics and xy scaling`);
    if (!box0.triclinic && (f[3] || f[4] || f[5])) bad(`Can not specify Pxy/Pxz/Pyz in fix ${style} with non-triclinic box`);
    const same = (a: number, b: number): boolean => this.pStart[a] === this.pStart[b] && this.pStop[a] === this.pStop[b] && pPeriod[a] === pPeriod[b];
    if (c === 'xyz' && dimension === 3 && (!same(0, 1) || !same(0, 2))) bad(`Invalid fix ${style} pressure settings`);
    if (c === 'xyz' && dimension === 2 && !same(0, 1)) bad(`Invalid fix ${style} pressure settings`);
    if (c === 'xy' && !same(0, 1)) bad(`Invalid fix ${style} pressure settings`);
    if (c === 'yz' && !same(1, 2)) bad(`Invalid fix ${style} pressure settings`);
    if (c === 'xz' && !same(0, 2)) bad(`Invalid fix ${style} pressure settings`);
    if ((tstat && tPeriodSet <= 0) || (tstat && tdrudePeriodSet <= 0) || [0, 1, 2, 3, 4, 5].some((d) => f[d] && pPeriod[d] <= 0)) {
      bad(`Fix ${style} damping parameters must be > 0.0`);
    }
    const pstat = f.some((b) => b);
    if (pstat) {
      // pstyle = TRICLINIC if any off-diagonal term is controlled, else ISO with xyz coupling (xy in 2d), else ANISO
      if (f[3] || f[4] || f[5]) this.pstyle = 'tri';
      else if (c === 'xyz' || (dimension === 2 && c === 'xy')) this.pstyle = 'iso';
      else this.pstyle = 'aniso';
      if (flipflag && (f[3] || f[4] || f[5])) this.flipCheck = true;
      if (flipflag && (g0.yz !== 0 || g0.xz !== 0 || g0.xy !== 0)) this.flipCheck = true;
    }
    // pre_exchange only when flips can occur (a fix without it keeps the runs that take no pre_exchange)
    if (this.flipCheck) this.preExchange = () => this.flipIfNeeded();
    this.tstat = tstat;
    this.pstat = pstat;
    // fix_tgnvt_drude.cpp and fix_tgnpt_drude.cpp
    if (!tstat) throw new StyleError(`Temperature control must be used with fix ${style}`);
    if (style === 'tgnvt/drude' && pstat) throw new StyleError(`Pressure control can not be used with fix ${style}`);
    if (style === 'tgnpt/drude' && !pstat) throw new StyleError(`Pressure control must be used with fix ${style}`);
    this.tdrudeTarget = tdrudeT;
    this.tFreq = 1 / tPeriodSet;
    this.tdrudeFreq = 1 / tdrudePeriodSet;
    for (let d = 0; d < 6; d++) if (f[d]) this.pFreq[d] = 1 / pPeriod[d];
    const mt = this.mtchain, mp = this.mpchain;
    this.etamol = new Float64Array(mt); this.etamolDot = new Float64Array(mt + 1); this.etamolDotdot = new Float64Array(mt); this.etamolMass = new Float64Array(mt);
    this.etaint = new Float64Array(mt); this.etaintDot = new Float64Array(mt + 1); this.etaintDotdot = new Float64Array(mt); this.etaintMass = new Float64Array(mt);
    this.etadrude = new Float64Array(mt); this.etadrudeDot = new Float64Array(mt + 1); this.etadrudeDotdot = new Float64Array(mt); this.etadrudeMass = new Float64Array(mt);
    this.etap = new Float64Array(Math.max(mp, 1)); this.etapDot = new Float64Array(Math.max(mp, 1) + 1);
    this.etapDotdot = new Float64Array(Math.max(mp, 1)); this.etapMass = new Float64Array(Math.max(mp, 1));
    this.tempId = `${id}_temp`;
    this.tempCompute = ownCompute(sys, this.tempId, pstat ? 'all' : group, 'temp', []);
    if (pstat) {
      this.pressId = `${id}_press`;
      this.pressCompute = ownCompute(sys, this.pressId, 'all', 'pressure', [this.tempId]);
    }
    if (sys.dimension === 2) throw new StyleError(`Fix ${style} is only implemented for 3d systems`);
    if (!sys.fixes.some((f) => f.style === 'drude')) throw new StyleError(`Fix ${style} requires fix drude`);
  }

  /** The Drude model of fix drude (checked here so that a missing fix is named at definition). */
  override init(): void {
    const s = this.sys.state;
    this.model = drudeModelOf(s, `fix ${this.style}`);
    const dt = s.dt;
    this.dtv = dt;
    this.dtf = 0.5 * dt * s.units.ftm2v;
    this.dthalf = 0.5 * dt;
    this.dt4 = 0.25 * dt;
    this.dt8 = 0.125 * dt;
    this.pFreqMax = 0;
    if (this.pstat) {
      const pf = this.pFreq;
      this.pFreqMax = Math.max(pf[0], pf[1]);
      this.pFreqMax = Math.max(this.pFreqMax, pf[2]);
      if (this.pstyle === 'tri') {
        this.pFreqMax = Math.max(this.pFreqMax, pf[3]);
        this.pFreqMax = Math.max(this.pFreqMax, pf[4]);
        this.pFreqMax = Math.max(this.pFreqMax, pf[5]);
      }
      // tally the number of dimensions that are barostatted; set the reference volume and cell, if not already done
      this.pdim = (this.pFlag[0] ? 1 : 0) + (this.pFlag[1] ? 1 : 0) + (this.pFlag[2] ? 1 : 0);
      if (this.vol0 === 0) this.setReferenceCell();
    }
  }

  /** The box h of LAMMPS' domain: xprd, yprd, zprd, yz, xz, xy. */
  private boxH(): number[] {
    const g = this.sys.geom;
    return [g.lx, g.ly, g.lz, g.yz, g.xz, g.xy];
  }

  /** vol0 and h0_inv from the current box (init, and nreset in compute_sigma). */
  private setReferenceCell(): void {
    const g = this.sys.geom;
    this.vol0 = this.sys.state.dimension === 3 ? g.lx * g.ly * g.lz : g.lx * g.ly;
    this.h0Inv = boxHInv(this.boxH(), this.sys.state.box.triclinic);
  }

  private volume(): number {
    const g = this.sys.geom;
    return this.sys.state.dimension === 3 ? g.lx * g.ly * g.lz : g.lx * g.ly;
  }

  /** The Drude model of fix drude, read once (the type words are fixed when the fix is defined). */
  private drude(): DrudeModel {
    if (!this.model) this.model = drudeModelOf(this.sys.state, `fix ${this.style}`);
    return this.model;
  }

  destroy(): void {
    removeCompute(this.sys, this.tempId);
    if (this.pressId) removeCompute(this.sys, this.pressId);
  }

  private tempScalar(): number {
    this.sys.refreshComputes();
    return this.tempCompute.scalarValue();
  }

  /** Molecule masses, degrees of freedom of the molecules, atoms and dipoles (setup_mol_mass_dof). */
  private setupMolMassDof(): void {
    const s = this.sys.state;
    const { mask, molecule, type } = s;
    const gb = this.groupBit;
    let maxMol = 0;
    for (let i = 0; i < s.n; i++) if (molecule[i] > maxMol) maxMol = molecule[i];
    this.nMol = maxMol;
    const inGroup = new Uint8Array(maxMol + 1);
    let nDrude = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & gb)) continue;
      inGroup[molecule[i]] = 1;
      if (this.drude().kind[type[i]] === DRUDE_TYPE) nDrude++;
    }
    let nMolInGroup = 0;
    for (let m = 1; m <= maxMol; m++) if (inGroup[m]) nMolInGroup++;
    this.massMol = new Float64Array(maxMol + 1);
    for (let i = 0; i < s.n; i++) this.massMol[molecule[i]] += massOf(s, i);
    this.vMol = new Float64Array(3 * (maxMol + 1));
    this.vMolTmp = new Float64Array(3 * (maxMol + 1));
    this.tCurrent = this.tempScalar();
    this.tdof = this.tempCompute.dof;
    this.dofMol = 3 * nMolInGroup - (3 * nMolInGroup) / maxMol;
    this.dofDrude = 3 * nDrude;
    this.dofInt = this.tdof - this.dofMol - this.dofDrude;
    if (this.dofMol <= 0 || this.dofInt <= 0 || this.dofDrude <= 0) {
      throw new StyleError('TGNHC thermostat requires DOFs of molecules, atoms and dipoles larger than 0');
    }
  }

  /** Target temperatures and kinetic energies (2 KE) at the current step of the run (compute_temp_target). */
  private computeTempTarget(): void {
    this.tTarget = ramp(this.sys, this.tStart, this.tStop);
    const kB = this.sys.state.units.boltz;
    this.ke2molTarget = this.dofMol * kB * this.tTarget;
    this.ke2intTarget = this.dofInt * kB * this.tTarget;
    this.ke2drudeTarget = this.dofDrude * kB * this.tdrudeTarget;
  }

  /** compute_press_target: the ramped targets, p_hydro over the barostatted diagonal, and sigma when deviatoric. */
  private computePressTarget(): void {
    this.pHydro = 0;
    for (let i = 0; i < 3; i++) {
      if (this.pFlag[i]) {
        this.pTarget[i] = ramp(this.sys, this.pStart[i], this.pStop[i]);
        this.pHydro += this.pTarget[i];
      }
    }
    if (this.pdim > 0) this.pHydro /= this.pdim;
    if (this.pstyle === 'tri') for (let i = 3; i < 6; i++) this.pTarget[i] = ramp(this.sys, this.pStart[i], this.pStop[i]);
    if (this.deviatoric) {
      // compute_sigma: with nreset > 0 the reference cell is taken again every nreset steps of the run
      if (this.nresetH0 > 0 && (this.sys.state.step - this.sys.run.beginStep) % this.nresetH0 === 0) this.setReferenceCell();
      this.sigma = computeSigma(this.vol0, this.h0Inv, this.pTarget, this.pHydro);
    }
  }

  /**
   * couple: the current pressure, the compute's scalar for an iso barostat, else its tensor (temperature tensor
   * fresh), averaged over the coupled dimensions; the tilt components for a triclinic barostat.
   */
  private couple(): void {
    this.sys.refreshComputes();
    const pc = this.pressCompute!;
    const p = this.pCurrent;
    if (this.pstyle === 'iso') {
      const sc = pc.scalarValue();
      p[0] = p[1] = p[2] = sc;
    } else {
      const tensor = pc.vectorValues();
      if (this.pcouple === 'xyz') {
        const ave = 1.0 / 3.0 * (tensor[0] + tensor[1] + tensor[2]);
        p[0] = p[1] = p[2] = ave;
      } else if (this.pcouple === 'xy') {
        const ave = 0.5 * (tensor[0] + tensor[1]);
        p[0] = p[1] = ave;
        p[2] = tensor[2];
      } else if (this.pcouple === 'yz') {
        const ave = 0.5 * (tensor[1] + tensor[2]);
        p[1] = p[2] = ave;
        p[0] = tensor[0];
      } else if (this.pcouple === 'xz') {
        const ave = 0.5 * (tensor[0] + tensor[2]);
        p[0] = p[2] = ave;
        p[1] = tensor[1];
      } else {
        p[0] = tensor[0]; p[1] = tensor[1]; p[2] = tensor[2];
      }
      // switch order from xy-xz-yz to Voigt
      if (this.pstyle === 'tri') { p[3] = tensor[5]; p[4] = tensor[4]; p[5] = tensor[3]; }
    }
    for (let d = 0; d < (this.pstyle === 'tri' ? 6 : 3); d++) {
      if (!Number.isFinite(p[d])) throw new StyleError(`Non-numeric pressure - simulation unstable (fix ${this.style})`);
    }
  }

  override setup(): void {
    this.setupMolMassDof();
    const s = this.sys.state;
    const kB = s.units.boltz;
    this.computeTempTarget();
    if (this.pstat) {
      this.computePressTarget();
      this.couple();
    }
    // masses and initial forces on the thermostat variables
    if (this.tstat) {
      const mt = this.mtchain;
      this.etaintMass[0] = this.ke2intTarget / (this.tFreq * this.tFreq);
      this.etamolMass[0] = this.ke2molTarget / (this.tFreq * this.tFreq);
      this.etadrudeMass[0] = this.ke2drudeTarget / (this.tdrudeFreq * this.tdrudeFreq);
      for (let ich = 1; ich < mt; ich++) {
        this.etaintMass[ich] = kB * this.tTarget / (this.tFreq * this.tFreq);
        this.etamolMass[ich] = kB * this.tTarget / (this.tFreq * this.tFreq);
        this.etadrudeMass[ich] = kB * this.tdrudeTarget / (this.tdrudeFreq * this.tdrudeFreq);
        this.etaintDotdot[ich] = (this.etaintMass[ich - 1] * this.etaintDot[ich - 1] * this.etaintDot[ich - 1] - kB * this.tTarget) / this.etaintMass[ich];
        this.etamolDotdot[ich] = (this.etamolMass[ich - 1] * this.etamolDot[ich - 1] * this.etamolDot[ich - 1] - kB * this.tTarget) / this.etamolMass[ich];
        this.etadrudeDotdot[ich] = (this.etadrudeMass[ich - 1] * this.etadrudeDot[ich - 1] * this.etadrudeDot[ich - 1] - kB * this.tdrudeTarget) / this.etadrudeMass[ich];
      }
    }
    // masses and initial forces on the barostat variables
    if (this.pstat) {
      const kt = kB * this.tTarget;
      const nkt = (s.n + 1) * kt;
      for (let i = 0; i < 3; i++) if (this.pFlag[i]) this.omegaMass[i] = nkt / (this.pFreq[i] * this.pFreq[i]);
      if (this.pstyle === 'tri') for (let i = 3; i < 6; i++) if (this.pFlag[i]) this.omegaMass[i] = nkt / (this.pFreq[i] * this.pFreq[i]);
      if (this.mpchain) {
        this.etapMass[0] = kB * this.tTarget / (this.pFreqMax * this.pFreqMax);
        for (let ich = 1; ich < this.mpchain; ich++) this.etapMass[ich] = kB * this.tTarget / (this.pFreqMax * this.pFreqMax);
        for (let ich = 1; ich < this.mpchain; ich++) {
          this.etapDotdot[ich] = (this.etapMass[ich - 1] * this.etapDot[ich - 1] * this.etapDot[ich - 1] - kB * this.tTarget) / this.etapMass[ich];
        }
      }
    }
  }

  override initialIntegrate(): void {
    if (this.pstat && this.mpchain) this.nhcPress();
    if (this.tstat) {
      this.computeTempTarget();
      this.nhcTemp();
    }
    if (this.pstat) {
      // t_current is not refreshed here (native: only the pressure is recomputed): the barostat's MTK term uses the
      // value of the previous final step
      this.couple();
      this.computePressTarget();
      this.nhOmegaDot();
      this.nhVPress();
    }
    this.nveV();
    if (this.pstat) this.remap();
    this.nveX();
    if (this.pstat) this.remap();
  }

  override finalIntegrate(): void {
    this.nveV();
    if (this.pstat) this.nhVPress();
    this.tCurrent = this.tempScalar();
    this.tdof = this.tempCompute.dof;
    if (this.pstat) this.couple();
    if (this.pstat) this.nhOmegaDot();
    if (this.tstat) this.nhcTemp();
    if (this.pstat && this.mpchain) this.nhcPress();
  }

  private nveV(): void {
    const s = this.sys.state;
    const { v, f, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const c = this.dtf / massOf(s, i);
      v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
    }
  }

  private nveX(): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      x[3 * i] += this.dtv * v[3 * i]; x[3 * i + 1] += this.dtv * v[3 * i + 1]; x[3 * i + 2] += this.dtv * v[3 * i + 2];
    }
  }

  /**
   * Temperatures of the molecules' centres of mass, of the internal motion and of the Drude relative motion, and
   * their kinetic energies (2 KE, energy units); compute_temp_mol_int_drude. A core and its Drude particle are
   * handled with the core, and the pair's centre of mass is taken as one internal degree of freedom.
   */
  private tempMolIntDrude(endOfStep: boolean): void {
    const s = this.sys.state;
    const { v, mask, type, molecule } = s;
    const gb = this.groupBit;
    const kind = this.drude().kind;
    const partner = this.drude().partnerIndex(s);
    const mvv2e = s.units.mvv2e, kB = s.units.boltz;
    this.vMolTmp.fill(0);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & gb)) continue;
      const m = massOf(s, i), im = 3 * molecule[i];
      for (let k = 0; k < 3; k++) this.vMolTmp[im + k] += v[3 * i + k] * m;
    }
    this.vMol.set(this.vMolTmp);
    this.ke2mol = 0;
    for (let i = 1; i <= this.nMol; i++) {
      const mm = this.massMol[i];
      for (let k = 0; k < 3; k++) {
        if (mm > 0) this.vMol[3 * i + k] /= mm;
        this.ke2mol += mm * this.vMol[3 * i + k] * this.vMol[3 * i + k];
      }
    }
    this.ke2mol *= mvv2e;
    this.tMol = this.ke2mol / this.dofMol / kB;
    let ke2int = 0, ke2drude = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & gb)) continue;
      const im = 3 * molecule[i];
      const ki = kind[type[i]];
      if (ki === NOPOL_TYPE) {
        const m = massOf(s, i);
        for (let k = 0; k < 3; k++) {
          const vint = v[3 * i + k] - this.vMol[im + k];
          ke2int += m * vint * vint;
        }
      } else if (ki === CORE_TYPE) {
        const di = partner[i];
        if (di < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[i]} is not defined`);
        const mc = massOf(s, i), md = massOf(s, di);
        const mcom = mc + md, mred = mc * md / mcom;
        for (let k = 0; k < 3; k++) {
          const vcom = (mc * v[3 * i + k] + md * v[3 * di + k]) / mcom;
          const vint = vcom - this.vMol[im + k];
          ke2int += mcom * vint * vint;
          const vrel = v[3 * di + k] - v[3 * i + k];
          ke2drude += mred * vrel * vrel;
        }
      }
    }
    this.ke2int = ke2int * mvv2e;
    this.ke2drude = ke2drude * mvv2e;
    this.tInt = this.ke2int / this.dofInt / kB;
    this.tDrude = this.ke2drude / this.dofDrude / kB;
    this.tempComputedEnd = endOfStep;
  }

  /** One thermostat chain over dt/2 (propagate); returns the velocity factor of the group it controls. */
  private propagate(eta: Float64Array, etaDot: Float64Array, etaDotdot: Float64Array, etaMass: Float64Array,
    ke2: number, ke2Target: number, tt: number): number {
    const mt = this.mtchain;
    const kB = this.sys.state.units.boltz;
    const ncfac = 1 / this.ncTchain;
    let factorEta = 1;
    etaDotdot[0] = (ke2 - ke2Target) / etaMass[0];
    for (let iloop = 0; iloop < this.ncTchain; iloop++) {
      for (let ich = mt - 1; ich > 0; ich--) {
        let expfac = libmExp(-ncfac * this.dt8 * etaDot[ich + 1]);
        etaDot[ich] *= expfac;
        etaDot[ich] += etaDotdot[ich] * ncfac * this.dt4;
        etaDot[ich] *= expfac;
      }
      let expfac = libmExp(-ncfac * this.dt8 * etaDot[1]);
      etaDot[0] *= expfac;
      etaDot[0] += etaDotdot[0] * ncfac * this.dt4;
      etaDot[0] *= expfac;
      factorEta *= libmExp(-ncfac * this.dthalf * etaDot[0]);
      for (let ich = 0; ich < mt; ich++) eta[ich] += ncfac * this.dthalf * etaDot[ich];
      etaDotdot[0] = (ke2 * factorEta * factorEta - ke2Target) / etaMass[0];
      etaDot[0] *= expfac;
      etaDot[0] += etaDotdot[0] * ncfac * this.dt4;
      etaDot[0] *= expfac;
      for (let ich = 1; ich < mt; ich++) {
        expfac = libmExp(-ncfac * this.dt8 * etaDot[ich + 1]);
        etaDot[ich] *= expfac;
        etaDotdot[ich] = (etaMass[ich - 1] * etaDot[ich - 1] * etaDot[ich - 1] - kB * tt) / etaMass[ich];
        etaDot[ich] += etaDotdot[ich] * ncfac * this.dt4;
        etaDot[ich] *= expfac;
      }
    }
    return factorEta;
  }

  private nhcTemp(): void {
    this.tempMolIntDrude(false);
    const kB = this.sys.state.units.boltz;
    this.etamolMass[0] = this.ke2molTarget / (this.tFreq * this.tFreq);
    this.etaintMass[0] = this.ke2intTarget / (this.tFreq * this.tFreq);
    for (let ich = 1; ich < this.mtchain; ich++) {
      this.etamolMass[ich] = kB * this.tTarget / (this.tFreq * this.tFreq);
      this.etaintMass[ich] = kB * this.tTarget / (this.tFreq * this.tFreq);
    }
    this.factorMol = this.propagate(this.etamol, this.etamolDot, this.etamolDotdot, this.etamolMass, this.ke2mol, this.ke2molTarget, this.tTarget);
    this.factorInt = this.propagate(this.etaint, this.etaintDot, this.etaintDotdot, this.etaintMass, this.ke2int, this.ke2intTarget, this.tTarget);
    this.factorDrude = this.propagate(this.etadrude, this.etadrudeDot, this.etadrudeDotdot, this.etadrudeMass, this.ke2drude, this.ke2drudeTarget, this.tdrudeTarget);
    this.nhVTemp();
  }

  /** Velocity scaling of the molecules' centres of mass, the internal motion and the Drude relative motion (nh_v_temp). */
  private nhVTemp(): void {
    const s = this.sys.state;
    const { v, mask, type, molecule } = s;
    const kind = this.drude().kind;
    const partner = this.drude().partnerIndex(s);
    const fm = this.factorMol, fi = this.factorInt, fd = this.factorDrude;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const im = 3 * molecule[i];
      const ki = kind[type[i]];
      if (ki === NOPOL_TYPE) {
        for (let k = 0; k < 3; k++) {
          let vint = v[3 * i + k] - this.vMol[im + k];
          vint *= fi;
          v[3 * i + k] = this.vMol[im + k] * fm + vint;
        }
      } else if (ki === CORE_TYPE) {
        const di = partner[i];
        if (di < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[i]} is not defined`);
        const mc = massOf(s, i), md = massOf(s, di);
        const mcom = mc + md;
        for (let k = 0; k < 3; k++) {
          const vcom = (mc * v[3 * i + k] + md * v[3 * di + k]) / mcom;
          let vint = vcom - this.vMol[im + k];
          let vrel = v[3 * di + k] - v[3 * i + k];
          vint *= fi;
          vrel *= fd;
          v[3 * i + k] = this.vMol[im + k] * fm + vint - vrel * md / mcom;
          v[3 * di + k] = this.vMol[im + k] * fm + vint + vrel * mc / mcom;
        }
      }
    }
  }

  /** Thermostat chain of the barostat over dt/2 (nhc_press_integrate). */
  private nhcPress(): void {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const kt = kB * this.tTarget;
    // update masses, to preserve initial freq, if t_target changed
    const nkt = (s.n + 1) * kt;
    const tri = this.pstyle === 'tri';
    const dims = tri ? 6 : 3;
    for (let i = 0; i < dims; i++) if (this.pFlag[i]) this.omegaMass[i] = nkt / (this.pFreq[i] * this.pFreq[i]);
    const mp = this.mpchain;
    if (mp) {
      this.etapMass[0] = kt / (this.pFreqMax * this.pFreqMax);
      for (let ich = 1; ich < mp; ich++) this.etapMass[ich] = kt / (this.pFreqMax * this.pFreqMax);
      for (let ich = 1; ich < mp; ich++) this.etapDotdot[ich] = (this.etapMass[ich - 1] * this.etapDot[ich - 1] * this.etapDot[ich - 1] - kt) / this.etapMass[ich];
    }
    const keBaro = (): number => {
      let ke = 0;
      for (let i = 0; i < dims; i++) if (this.pFlag[i]) ke += this.omegaMass[i] * this.omegaDot[i] * this.omegaDot[i];
      return ke;
    };
    let pdof = 0;
    for (let i = 0; i < dims; i++) if (this.pFlag[i]) pdof++;
    let kecurrent = keBaro();
    const lktPress = this.pstyle === 'iso' ? kt : pdof * kt;
    this.etapDotdot[0] = (kecurrent - lktPress) / this.etapMass[0];
    const ncfac = 1 / this.ncPchain;
    for (let iloop = 0; iloop < this.ncPchain; iloop++) {
      for (let ich = mp - 1; ich > 0; ich--) {
        const expfac = libmExp(-ncfac * this.dt8 * this.etapDot[ich + 1]);
        this.etapDot[ich] *= expfac;
        this.etapDot[ich] += this.etapDotdot[ich] * ncfac * this.dt4;
        this.etapDot[ich] *= expfac;
      }
      let expfac = libmExp(-ncfac * this.dt8 * this.etapDot[1]);
      this.etapDot[0] *= expfac;
      this.etapDot[0] += this.etapDotdot[0] * ncfac * this.dt4;
      this.etapDot[0] *= expfac;
      for (let ich = 0; ich < mp; ich++) this.etap[ich] += ncfac * this.dthalf * this.etapDot[ich];
      const factorEtap = libmExp(-ncfac * this.dthalf * this.etapDot[0]);
      for (let i = 0; i < dims; i++) if (this.pFlag[i]) this.omegaDot[i] *= factorEtap;
      kecurrent = keBaro();
      this.etapDotdot[0] = (kecurrent - lktPress) / this.etapMass[0];
      this.etapDot[0] *= expfac;
      this.etapDot[0] += this.etapDotdot[0] * ncfac * this.dt4;
      this.etapDot[0] *= expfac;
      for (let ich = 1; ich < mp; ich++) {
        expfac = libmExp(-ncfac * this.dt8 * this.etapDot[ich + 1]);
        this.etapDot[ich] *= expfac;
        this.etapDotdot[ich] = (this.etapMass[ich - 1] * this.etapDot[ich - 1] * this.etapDot[ich - 1] - kt) / this.etapMass[ich];
        this.etapDot[ich] += this.etapDotdot[ich] * ncfac * this.dt4;
        this.etapDot[ich] *= expfac;
      }
    }
  }

  /** Barostat velocity scaling by dt/4 on each side of the tilt coupling (nh_v_press). */
  private nhVPress(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const w = this.omegaDot;
    const f0 = libmExp(-this.dt4 * (w[0] + this.mtkTerm2));
    const f1 = libmExp(-this.dt4 * (w[1] + this.mtkTerm2));
    const f2 = libmExp(-this.dt4 * (w[2] + this.mtkTerm2));
    const tri = this.pstyle === 'tri';
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const k = 3 * i;
      v[k] *= f0; v[k + 1] *= f1; v[k + 2] *= f2;
      if (tri) {
        v[k] += -this.dthalf * (v[k + 1] * w[5] + v[k + 2] * w[4]);
        v[k + 1] += -this.dthalf * v[k + 2] * w[3];
      }
      v[k] *= f0; v[k + 1] *= f1; v[k + 2] *= f2;
    }
  }

  /** Barostat velocity update over dt/2 (nh_omega_dot). */
  private nhOmegaDot(): void {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const nktv2p = s.units.nktv2p;
    const volume = this.volume();
    const fdev = this.deviatoric ? computeDeviatoric(this.boxH(), this.sigma) : null;
    this.mtkTerm1 = 0;
    if (this.mtkFlag) {
      if (this.pstyle === 'iso') {
        this.mtkTerm1 = this.tdof * kB * this.tCurrent;
        this.mtkTerm1 /= this.pdim * s.n;
      } else {
        // the temperature's kinetic tensor, computed with the pressure in couple()
        const mvv = this.tempCompute.vectorValues();
        for (let i = 0; i < 3; i++) if (this.pFlag[i]) this.mtkTerm1 += mvv[i];
        this.mtkTerm1 /= this.pdim * s.n;
      }
    }
    for (let i = 0; i < 3; i++) {
      if (!this.pFlag[i]) continue;
      let fOmega = (this.pCurrent[i] - this.pHydro) * volume / (this.omegaMass[i] * nktv2p) + this.mtkTerm1 / this.omegaMass[i];
      if (fdev) fOmega -= fdev[i] / (this.omegaMass[i] * nktv2p);
      this.omegaDot[i] += fOmega * this.dthalf;
    }
    this.mtkTerm2 = 0;
    if (this.mtkFlag) {
      for (let i = 0; i < 3; i++) if (this.pFlag[i]) this.mtkTerm2 += this.omegaDot[i];
      if (this.pdim > 0) this.mtkTerm2 /= this.pdim * s.n;
    }
    if (this.pstyle === 'tri') {
      for (let i = 3; i < 6; i++) {
        if (!this.pFlag[i]) continue;
        let fOmega = this.pCurrent[i] * volume / (this.omegaMass[i] * nktv2p);
        if (fdev) fOmega -= fdev[i] / (this.omegaMass[i] * nktv2p);
        this.omegaDot[i] += fOmega * this.dthalf;
      }
    }
  }

  /**
   * remap: the box over dt/2 (h_dot = omega_dot h, the tilts swept before and after the diagonal scaling about the
   * fixed point, with the tilt scale flags) and all atoms in fractional coordinates. A tilt past TILTMAX of its box
   * length in one step is an error.
   */
  private remap(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const b = s.box;
    const dto = this.dthalf;
    const lam = new Float64Array(3 * s.n);
    const tmp = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      lam[3 * i] = tmp[0]; lam[3 * i + 1] = tmp[1]; lam[3 * i + 2] = tmp[2];
    }
    const w = this.omegaDot;
    const f = this.pFlag;
    const h = this.boxH();
    const xprd = h[0], yprd = h[1];
    const dto2 = dto / 2.0, dto4 = dto / 4.0, dto8 = dto / 8.0;
    const sweep = (): void => {
      if (this.pstyle !== 'tri') return;
      if (f[4]) {
        const e = libmExp(dto8 * w[0]);
        h[4] *= e; h[4] += dto4 * (w[5] * h[3] + w[4] * h[2]); h[4] *= e;
      }
      if (f[3]) {
        const e = libmExp(dto4 * w[1]);
        h[3] *= e; h[3] += dto2 * (w[3] * h[2]); h[3] *= e;
      }
      if (f[5]) {
        const e = libmExp(dto4 * w[0]);
        h[5] *= e; h[5] += dto2 * (w[5] * h[1]); h[5] *= e;
      }
      if (f[4]) {
        const e = libmExp(dto8 * w[0]);
        h[4] *= e; h[4] += dto4 * (w[5] * h[3] + w[4] * h[2]); h[4] *= e;
      }
    };
    sweep();
    for (let d = 0; d < 3; d++) {
      if (!f[d]) continue;
      const expfac = libmExp(dto * w[d]);
      const fp = this.fixedpoint[d];
      b.lo[d] = (b.lo[d] - fp) * expfac + fp;
      b.hi[d] = (b.hi[d] - fp) * expfac + fp;
      if (d === 1 && this.scalexy) h[5] *= expfac;
      if (d === 2) {
        if (this.scalexz) h[4] *= expfac;
        if (this.scaleyz) h[3] *= expfac;
      }
    }
    sweep();
    // tilt factor to cell length ratio can not exceed TILTMAX in one step (the lengths before this remap)
    if (h[3] < -TILTMAX * yprd || h[3] > TILTMAX * yprd || h[4] < -TILTMAX * xprd || h[4] > TILTMAX * xprd ||
      h[5] < -TILTMAX * xprd || h[5] > TILTMAX * xprd) {
      throw new StyleError(`Fix ${this.style} has tilted box too far in one step - periodic cell is too far from equilibrium state`);
    }
    if (b.triclinic) { b.tilt[0] = h[5]; b.tilt[1] = h[4]; b.tilt[2] = h[3]; }
    b.minLo = [...b.lo]; b.minHi = [...b.hi];
    g.update();
    for (let i = 0; i < s.n; i++) {
      g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
    }
    for (const fx of sys.fixes) if (fx !== this) fx.boxChanged?.();
  }

  /**
   * pre_exchange of fix_tgnh_drude.cpp, on the steps the neighbor lists are rebuilt: a tilt past (0.5 + DELTAFLIP) of
   * its box length flips the box ("flip is only triggered when tilt exceeds 0.5 by DELTAFLIP"), yz first by one y
   * length (which moves xz by xy), then xz and xy by one x length, along the periodic dimensions only; the image flags
   * and the atoms follow (Domain::image_flip, remap).
   */
  private flipIfNeeded(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const xprd = g.lx, yprd = g.ly;
    const xtiltmax = (0.5 + DELTAFLIP) * xprd, ytiltmax = (0.5 + DELTAFLIP) * yprd;
    let xy = g.xy, xz = g.xz, yz = g.yz;
    let flipxy = 0, flipxz = 0, flipyz = 0;
    if (s.box.periodic[1]) {
      if (yz < -ytiltmax) { yz += yprd; xz += xy; flipyz = 1; }
      else if (yz >= ytiltmax) { yz -= yprd; xz -= xy; flipyz = -1; }
    }
    if (s.box.periodic[0]) {
      if (xz < -xtiltmax) { xz += xprd; flipxz = 1; }
      else if (xz >= xtiltmax) { xz -= xprd; flipxz = -1; }
      if (xy < -xtiltmax) { xy += xprd; flipxy = 1; }
      else if (xy >= xtiltmax) { xy -= xprd; flipxy = -1; }
    }
    if (flipxy || flipxz || flipyz) this.sys.flipBox([xy, xz, yz], flipxy, flipxz, flipyz, this);
  }

  override ecouple(): number { return this.energyChain(); }

  override computeScalar(): number { return this.energyChain(); }

  /** Thermostat and barostat chain energy (compute_scalar, Martyna et al. Mol Phys 87, 1117). */
  private energyChain(): number {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const kt = kB * this.tTarget;
    const ktDrude = kB * this.tdrudeTarget;
    let energy = 0;
    if (this.tstat) {
      energy += this.ke2molTarget * this.etamol[0] + 0.5 * this.etamolMass[0] * this.etamolDot[0] * this.etamolDot[0];
      energy += this.ke2intTarget * this.etaint[0] + 0.5 * this.etaintMass[0] * this.etaintDot[0] * this.etaintDot[0];
      energy += this.ke2drudeTarget * this.etadrude[0] + 0.5 * this.etadrudeMass[0] * this.etadrudeDot[0] * this.etadrudeDot[0];
      for (let ich = 1; ich < this.mtchain; ich++) {
        energy += kt * this.etamol[ich] + 0.5 * this.etamolMass[ich] * this.etamolDot[ich] * this.etamolDot[ich];
        energy += kt * this.etaint[ich] + 0.5 * this.etaintMass[ich] * this.etaintDot[ich] * this.etaintDot[ich];
        energy += ktDrude * this.etadrude[ich] + 0.5 * this.etadrudeMass[ich] * this.etadrudeDot[ich] * this.etadrudeDot[ich];
      }
    }
    if (this.pstat) {
      const nktv2p = s.units.nktv2p;
      const volume = this.volume();
      let lktPress = 0;
      for (let i = 0; i < 3; i++) {
        if (!this.pFlag[i]) continue;
        energy += 0.5 * this.omegaDot[i] * this.omegaDot[i] * this.omegaMass[i] + this.pHydro * (volume - this.vol0) / (this.pdim * nktv2p);
        lktPress += kt;
      }
      if (this.pstyle === 'tri') {
        for (let i = 3; i < 6; i++) {
          if (!this.pFlag[i]) continue;
          energy += 0.5 * this.omegaDot[i] * this.omegaDot[i] * this.omegaMass[i];
          lktPress += kt;
        }
      }
      if (this.mpchain) {
        energy += lktPress * this.etap[0] + 0.5 * this.etapMass[0] * this.etapDot[0] * this.etapDot[0];
        for (let ich = 1; ich < this.mpchain; ich++) energy += kt * this.etap[ich] + 0.5 * this.etapMass[ich] * this.etapDot[ich] * this.etapDot[ich];
      }
      if (this.deviatoric) energy += computeStrainEnergy(this.boxH(), this.sigma, nktv2p);
    }
    return energy;
  }

  /** Molecule, internal and Drude temperatures (f_ID[1..3]). */
  override computeVector(i: number): number {
    if (!this.tempComputedEnd) this.tempMolIntDrude(true);
    switch (i) {
      case 0: return this.tMol;
      case 1: return this.tInt;
      case 2: return this.tDrude;
      default: return 0;
    }
  }
}
