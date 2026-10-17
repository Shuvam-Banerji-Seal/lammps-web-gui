/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/fix_tgnh_drude.cpp and
 * src/DRUDE/fix_tgnh_drude.h (class FixTGNHDrude, the temperature-grouped Nose-Hoover base of fix tgnvt/drude and
 * fix tgnpt/drude). Copyright (2003) Sandia Corporation; GPL-2.0 (src/engine-gpl/LICENSE). DRUDE package authors:
 * Alain Dequidt, Julien Devemy, Agilio Padua; contributors Kateryna Goloviznina and Zheng Gong.
 *
 * Nose-Hoover chains on three groups of degrees of freedom of a Drude model (docs.lammps.org/fix_tgnh_drude.html):
 * the centres of mass of the molecules, the internal motion of the atoms (core and non-polarizable atoms relative to
 * their molecule, the core-Drude pair centre relative to the molecule), and the relative motion of each core-Drude
 * pair. Syntax: fix ID group tgnvt/drude temp Tstart Tstop Tperiod Tdrude Tdrude_period [iso P Pstart Pstop Pperiod]
 * [tchain N] [pchain N] [mtk yes|no] [tloop N] [ploop N]. Fix tgnpt/drude takes the iso barostat; the other
 * barostat keywords (aniso, tri, x, y, z, xy, yz, xz, couple other than xyz, the box flip and fixedpoint keywords)
 * are refused by name.
 */
import { Fix } from '../../engine/fix/fix';
import { ownCompute, ramp, removeCompute } from '../../engine/fix/util';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { massOf } from '../../engine/atoms';
import { CORE_TYPE, DRUDE_TYPE, NOPOL_TYPE, drudeModelOf, type DrudeModel } from './drude_model';
import { exp as libmExp } from '../libm/exp';

const num = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || w === '' || !Number.isFinite(v)) throw new StyleError(`Illegal fix tgnh/drude ${what}: '${w ?? ''}'`);
  return v;
};

const int = (w: string | undefined, what: string): number => {
  const v = num(w, what);
  if (!Number.isInteger(v)) throw new StyleError(`Illegal fix tgnh/drude ${what}: '${w ?? ''}'`);
  return v;
};

/** Keywords the port does not implement (the barostat shapes and the box flip): named in the refusal. */
const REFUSED = new Set(['aniso', 'tri', 'x', 'y', 'z', 'xy', 'yz', 'xz', 'scalexy', 'scalexz', 'scaleyz', 'flip', 'fixedpoint', 'nreset']);

export class FixTGNHDrude extends Fix {
  readonly style: string;
  private readonly tstat: boolean;
  private readonly pstat: boolean;
  private tStart = 0; private tStop = 0;
  private tdrudeTarget = 0;
  private tTarget = 0; private tFreq = 0; private tdrudeFreq = 0;
  private mtchain = 3; private mpchain = 3; private ncTchain = 1; private ncPchain = 1;
  private mtkFlag = true;
  // barostat (iso only)
  private pStart = 0; private pStop = 0; private pPeriod = 0;
  private pTarget = 0; private pHydro = 0; private pCurrent = 0;
  private pFreq = 0; private pFreqMax = 0;
  private readonly pdim: number;
  private vol0 = 0;
  private readonly fixedpoint: number[];
  private omegaDot = new Float64Array(3);
  private omegaMass = new Float64Array(3);
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
    this.pdim = 3;
    if (args.length < 2) throw new StyleError(`Illegal fix ${style} command`);
    let tdrudeT = 0;
    let havePress = false;
    let tstat = false;
    let pstat = false;
    let tPeriodSet = 0, tdrudePeriodSet = 0;
    for (let k = 0; k < args.length;) {
      const key = args[k];
      if (key === 'temp') {
        if (k + 6 > args.length) throw new StyleError(`Illegal fix ${style} temp: needs Tstart Tstop Tperiod Tdrude Tdrude_period`);
        tstat = true;
        this.tStart = num(args[k + 1], 'Tstart');
        this.tTarget = this.tStart;
        this.tStop = num(args[k + 2], 'Tstop');
        tPeriodSet = num(args[k + 3], 'Tperiod');
        tdrudeT = num(args[k + 4], 'Tdrude');
        tdrudePeriodSet = num(args[k + 5], 'Tdrude period');
        if (this.tStart <= 0) throw new StyleError(`Target temperature for fix ${style} cannot be <= 0.0`);
        if (this.tStop <= 0) throw new StyleError(`Target temperature for fix ${style} cannot be <= 0.0`);
        if (tdrudeT <= 0) throw new StyleError(`Target Drude temperature for fix ${style} cannot be <= 0.0`);
        k += 6;
      } else if (key === 'iso') {
        if (k + 4 > args.length) throw new StyleError(`Illegal fix ${style} iso: needs Pstart Pstop Pperiod`);
        pstat = true; havePress = true;
        this.pStart = num(args[k + 1], 'Pstart');
        this.pStop = num(args[k + 2], 'Pstop');
        this.pPeriod = num(args[k + 3], 'Pperiod');
        k += 4;
      } else if (key === 'couple') {
        if (args[k + 1] !== 'xyz') throw new StyleError(`fix ${style} supports only 'couple xyz' (iso), got '${args[k + 1] ?? ''}'`);
        k += 2;
      } else if (key === 'tchain' || key === 'pchain' || key === 'tloop' || key === 'ploop') {
        const n = int(args[k + 1], key);
        if (n < 0 || ((key === 'tchain' || key === 'tloop') && n < 1)) throw new StyleError(`Invalid fix ${style} ${key} argument: ${n}`);
        if (key === 'tchain') this.mtchain = n;
        else if (key === 'pchain') this.mpchain = n;
        else if (key === 'tloop') this.ncTchain = n;
        else this.ncPchain = n;
        k += 2;
      } else if (key === 'mtk') {
        const v = args[k + 1];
        if (v !== 'yes' && v !== 'no') throw new StyleError(`Expected boolean parameter instead of '${v ?? ''}' in input script`);
        this.mtkFlag = v === 'yes';
        k += 2;
      } else if (REFUSED.has(key)) {
        throw new StyleError(`fix ${style} keyword '${key}' is not supported by the browser engine (iso barostat only)`);
      } else {
        throw new StyleError(`Unknown fix ${style} keyword: ${key}`);
      }
    }
    this.tstat = tstat;
    this.pstat = pstat;
    if (style === 'tgnvt/drude') {
      if (!tstat) throw new StyleError(`Temperature control must be used with fix ${style}`);
      if (havePress) throw new StyleError(`Pressure control can not be used with fix ${style}`);
    } else if (!pstat) throw new StyleError(`Pressure control must be used with fix ${style}`);
    if (tstat && (tPeriodSet <= 0 || tdrudePeriodSet <= 0)) throw new StyleError(`Fix ${style} damping parameters must be > 0.0`);
    if (pstat && this.pPeriod <= 0) throw new StyleError(`Fix ${style} damping parameters must be > 0.0`);
    this.tdrudeTarget = tdrudeT;
    if (tstat) {
      this.tFreq = 1 / tPeriodSet;
      this.tdrudeFreq = 1 / tdrudePeriodSet;
    }
    if (pstat) {
      this.pFreq = 1 / this.pPeriod;
      this.pFreqMax = this.pFreq;
      this.pTarget = this.pStart;
      this.pCurrent = 0;
    }
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
    if (this.pstat && this.vol0 === 0) this.vol0 = this.sys.geom.volume(s.dimension);
  }

  private volume(): number { return this.sys.geom.volume(this.sys.state.dimension); }

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

  private pressScalar(): number {
    this.sys.refreshComputes();
    return this.pressCompute!.scalarValue();
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

  private computePressTarget(): void {
    this.pTarget = ramp(this.sys, this.pStart, this.pStop);
    this.pHydro = this.pTarget;
  }

  /** Iso coupling: the current pressure is the scalar of compute pressure (couple). */
  private couple(): void {
    this.pCurrent = this.pressScalar();
    if (!Number.isFinite(this.pCurrent)) throw new StyleError(`Non-numeric pressure - simulation unstable (fix ${this.style})`);
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
      for (let i = 0; i < 3; i++) this.omegaMass[i] = nkt / (this.pFreq * this.pFreq);
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

  /** Thermostat chain of the barostat over dt/2 (nhc_press_integrate, iso). */
  private nhcPress(): void {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const kt = kB * this.tTarget;
    const nkt = (s.n + 1) * kt;
    for (let i = 0; i < 3; i++) this.omegaMass[i] = nkt / (this.pFreq * this.pFreq);
    const mp = this.mpchain;
    if (mp) {
      this.etapMass[0] = kt / (this.pFreqMax * this.pFreqMax);
      for (let ich = 1; ich < mp; ich++) this.etapMass[ich] = kt / (this.pFreqMax * this.pFreqMax);
      for (let ich = 1; ich < mp; ich++) this.etapDotdot[ich] = (this.etapMass[ich - 1] * this.etapDot[ich - 1] * this.etapDot[ich - 1] - kt) / this.etapMass[ich];
    }
    let kecurrent = 0;
    for (let i = 0; i < 3; i++) kecurrent += this.omegaMass[i] * this.omegaDot[i] * this.omegaDot[i];
    const lktPress = kt;
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
      for (let i = 0; i < 3; i++) this.omegaDot[i] *= factorEtap;
      kecurrent = 0;
      for (let i = 0; i < 3; i++) kecurrent += this.omegaMass[i] * this.omegaDot[i] * this.omegaDot[i];
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

  /** Barostat velocity scaling by dt/4 on each side of the kick (nh_v_press). */
  private nhVPress(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const factor = libmExp(-this.dt4 * (this.omegaDot[0] + this.mtkTerm2));
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      v[3 * i] *= factor; v[3 * i + 1] *= factor; v[3 * i + 2] *= factor;
      v[3 * i] *= factor; v[3 * i + 1] *= factor; v[3 * i + 2] *= factor;
    }
  }

  /** Barostat velocity update over dt/2 (nh_omega_dot, iso). */
  private nhOmegaDot(): void {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const nktv2p = s.units.nktv2p;
    const volume = this.volume();
    this.mtkTerm1 = 0;
    if (this.mtkFlag) this.mtkTerm1 = (this.tdof * kB * this.tCurrent) / (this.pdim * s.n);
    for (let i = 0; i < 3; i++) {
      const fOmega = (this.pCurrent - this.pHydro) * volume / (this.omegaMass[i] * nktv2p) + this.mtkTerm1 / this.omegaMass[i];
      this.omegaDot[i] += fOmega * this.dthalf;
    }
    this.mtkTerm2 = 0;
    if (this.mtkFlag) {
      for (let i = 0; i < 3; i++) this.mtkTerm2 += this.omegaDot[i];
      this.mtkTerm2 /= this.pdim * s.n;
    }
  }

  /** Box and all atoms over dt/2 (remap, iso): the box is dilated about the fixed point (box centre at definition). */
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
    for (let d = 0; d < 3; d++) {
      const expfac = libmExp(dto * this.omegaDot[d]);
      const fp = this.fixedpoint[d];
      b.lo[d] = (b.lo[d] - fp) * expfac + fp;
      b.hi[d] = (b.hi[d] - fp) * expfac + fp;
    }
    b.minLo = [...b.lo]; b.minHi = [...b.hi];
    g.update();
    for (let i = 0; i < s.n; i++) {
      g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
    }
    for (const f of sys.fixes) if (f !== this) f.boxChanged?.();
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
        energy += 0.5 * this.omegaDot[i] * this.omegaDot[i] * this.omegaMass[i] + this.pHydro * (volume - this.vol0) / (this.pdim * nktv2p);
        lktPress += kt;
      }
      if (this.mpchain) {
        energy += lktPress * this.etap[0] + 0.5 * this.etapMass[0] * this.etapDot[0] * this.etapDot[0];
        for (let ich = 1; ich < this.mpchain; ich++) energy += kt * this.etap[ich] + 0.5 * this.etapMass[ich] * this.etapDot[ich] * this.etapDot[ich];
      }
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
