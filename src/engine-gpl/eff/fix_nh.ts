/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/fix_nh.cpp and fix_nh.h (class FixNH),
 * the parts that the electron force field fixes use: the Nose-Hoover chain of the temperature (fix nvt: temp, tchain,
 * tloop, drag) and the isotropic barostat with its chain (fix npt and fix nph: iso, pchain, ploop, fixedpoint). The
 * anisotropic and triclinic barostats, the other pressure keywords, ptemp and the rigid and deform couplings are not
 * ported; they throw a StyleError naming them. Contributing authors (from the source): Mark Stevens (SNL), Aidan
 * Thompson (SNL). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The velocity and position updates and the thermostat scaling are methods, so the eFF subclasses (fix_nh_eff.ts)
 * extend them for the electron radial velocities, as fix_nh_eff.cpp overrides nve_v, nve_x and nh_v_temp. It is the
 * base of fix nvt/sllod/eff (../rigeff/fix_nvt_sllod_eff.ts), which takes no pressure keyword; fix nvt/eff, npt/eff
 * and nph/eff are built on the core FixNH, which has every barostat keyword.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import { massOf } from '../../engine/atoms';
import { ramp, temperatureOf } from '../../engine/fix/util';
import { exp as libmExp } from '../libm/exp';

/** fix_nh.cpp constructor: mtchain = mpchain = 3 by default, nc_tchain = nc_pchain = 1 (tloop, ploop). */
const DEFAULT_CHAIN = 3;

/** The keywords of native FixNH (fix_nh.cpp constructor at b332f28); the ones not ported here are refused by name. */
const FIX_NH_KEYWORDS: ReadonlySet<string> = new Set([
  'aniso', 'couple', 'dilate', 'disc', 'drag', 'erate', 'ext', 'fixedpoint', 'flip', 'integrator', 'iso', 'isochoric', 'kick',
  'mtk', 'nreset', 'pchain', 'peculiar', 'ploop', 'psllod', 'ptemp', 'scalexy', 'scalexz', 'scaleyz', 'strain', 'tchain',
  'temp', 'tloop', 'tri', 'update', 'x', 'xy', 'xz', 'y', 'yz', 'z',
]);

export class FixNHBase extends Fix {
  protected tstat = false;
  protected tStart = 0;
  protected tStop = 0;
  protected tPeriod = 0;
  protected tTarget = 0;
  protected keTarget = 0;
  protected tCurrent = 0;
  protected tdof = 0;
  protected tFreq = 0;
  protected mtchain = DEFAULT_CHAIN;
  protected ncTchain = 1;
  protected drag = 0;
  protected tdragFactor = 1;
  protected eta = new Float64Array(0);
  protected etaDot = new Float64Array(0);
  protected etaDotdot = new Float64Array(0);
  protected etaMass = new Float64Array(0);
  protected factorEta = 1;
  protected boltz = 0;
  protected nktv2p = 1;
  protected dtv = 0;
  protected dtf = 0;
  protected dthalf = 0;
  protected dt4 = 0;
  protected dt8 = 0;

  // barostat (isotropic)
  protected pstat = false;
  protected pStart = 0;
  protected pStop = 0;
  protected pPeriod = 0;
  protected pFreq = 0;
  protected pFreqMax = 0;
  protected mpchain = DEFAULT_CHAIN;
  protected ncPchain = 1;
  protected pdragFactor = 1;
  protected pdim = 3;
  protected pTarget = 0;
  protected pHydro = 0;
  protected pCurrent = 0;
  protected omegaDot = 0;
  protected omegaMass = 0;
  protected omega = 0;
  protected mtkTerm1 = 0;
  protected mtkTerm2 = 0;
  protected etap = new Float64Array(0);
  protected etapDot = new Float64Array(0);
  protected etapDotdot = new Float64Array(0);
  protected etapMass = new Float64Array(0);
  protected vol0 = 0;
  protected t0 = 0;
  protected fixedpoint: [number, number, number];
  /** the temperature compute the thermostat and the pressure use (id_temp) */
  protected tempId: string;
  protected temperature: Compute | null = null;
  /** the pressure compute of the barostat (id_press), when pstat */
  protected pressId: string | null = null;
  protected pressure: Compute | null = null;

  constructor(sys: System, id: string, group: string, args: string[], tempId: string, readonly style: string, pressId: string | null = null) {
    super(sys, id, group, args);
    this.tempId = tempId;
    this.pressId = pressId;
    this.timeIntegrate = true;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.ecouple = () => this.computeScalar();
    const box = sys.state.box;
    this.fixedpoint = [0.5 * (box.lo[0] + box.hi[0]), 0.5 * (box.lo[1] + box.hi[1]), 0.5 * (box.lo[2] + box.hi[2])];
    let i = 0;
    while (i < args.length) {
      const w = args[i];
      if (w === 'temp') {
        // As native LAMMPS does (src/fix_nh.cpp FixNH::FixNH at b332f28): if (iarg + 4 > narg) missing_cmd_args("fix {} temp")
        if (i + 4 > args.length) throw new StyleError(`Illegal fix ${style} temp command: missing argument(s)`);
        this.tstat = true;
        // As native LAMMPS does (src/fix_nh.cpp at b332f28): each value is read with utils::numeric and checked before the next is read
        this.tStart = numeric(args[i + 1]);
        if (!(this.tStart > 0)) throw new StyleError(`Target temperature for fix ${style} cannot be <= 0.0`);
        this.tStop = numeric(args[i + 2]);
        if (!(this.tStop > 0)) throw new StyleError(`Target temperature for fix ${style} cannot be <= 0.0`);
        this.tPeriod = numeric(args[i + 3]);
        if (!(this.tPeriod > 0)) throw new StyleError(`Temperature damping for fix ${style} cannot be <= 0.0`);
        i += 4;
      } else if (w === 'iso') {
        // As native LAMMPS does (src/fix_nh.cpp FixNH::FixNH at b332f28): if (iarg + 4 > narg) missing_cmd_args("fix {} iso")
        if (i + 4 > args.length) throw new StyleError(`Illegal fix ${style} iso command: missing argument(s)`);
        this.pstat = true;
        this.pStart = Number(args[i + 1]);
        this.pStop = Number(args[i + 2]);
        this.pPeriod = Number(args[i + 3]);
        if (!(this.pPeriod > 0)) throw new StyleError(`Pressure damping for fix ${style} cannot be <= 0.0`);
        i += 4;
      } else if (w === 'tchain' || w === 'pchain') {
        const n = Number(args[i + 1]);
        if (!Number.isInteger(n) || n < 1) throw new StyleError(`Invalid fix ${style} ${w} argument: ${args[i + 1]}`);
        if (w === 'tchain') this.mtchain = n; else this.mpchain = n;
        i += 2;
      } else if (w === 'tloop' || w === 'ploop') {
        const n = Number(args[i + 1]);
        if (!Number.isInteger(n) || n < 0) throw new StyleError(`Invalid fix ${style} ${w} argument: ${args[i + 1]}`);
        if (w === 'tloop') this.ncTchain = n; else this.ncPchain = n;
        i += 2;
      } else if (w === 'drag') {
        this.drag = Number(args[i + 1]);
        i += 2;
      } else if (w === 'fixedpoint') {
        this.fixedpoint = [Number(args[i + 1]), Number(args[i + 2]), Number(args[i + 3])];
        i += 4;
      } else if (FIX_NH_KEYWORDS.has(w)) {
        throw new StyleError(`fix ${style} keyword '${w}' is not supported by the browser engine (ported: temp, iso, tchain, pchain, tloop, ploop, drag, fixedpoint)`);
      } else {
        // fix_nh.cpp: a word that is no fix nh keyword
        throw new StyleError(`Unknown fix ${style} keyword: ${w}`);
      }
    }
    if (this.pstat && sys.dimension === 2) throw new StyleError(`fix ${style} iso in a 2d simulation is not supported by the browser engine`);
    this.tFreq = this.tstat ? 1 / this.tPeriod : 0;
    this.pFreq = this.pstat ? 1 / this.pPeriod : 0;
    this.pFreqMax = this.pFreq;
    this.eta = new Float64Array(this.mtchain);
    this.etaDot = new Float64Array(this.mtchain + 1);
    this.etaDotdot = new Float64Array(this.mtchain);
    this.etaMass = new Float64Array(this.mtchain);
    this.etap = new Float64Array(this.mpchain);
    this.etapDot = new Float64Array(this.mpchain + 1);
    this.etapDotdot = new Float64Array(this.mpchain);
    this.etapMass = new Float64Array(this.mpchain);
  }

  /** the temperature compute, looked up once the fix is initialized */
  protected tempCompute(): Compute {
    if (!this.temperature) throw new StyleError(`Temperature ID ${this.tempId} for fix ${this.style} does not exist`);
    return this.temperature;
  }

  protected pressCompute(): Compute {
    if (!this.pressure) throw new StyleError(`Pressure ID ${this.pressId} for fix ${this.style} does not exist`);
    return this.pressure;
  }

  init(): void {
    this.temperature = this.sys.compute(this.tempId);
    const t = this.tempCompute();
    if (!t.tempFlag) throw new StyleError(`Compute ID ${this.tempId} for fix ${this.style} does not compute a temperature`);
    if (this.pstat) {
      this.pressure = this.sys.compute(this.pressId!);
      if (!this.pressure.pressFlag) throw new StyleError(`Pressure ID ${this.pressId} for fix ${this.style} does not compute a pressure`);
    }
    const s = this.sys.state;
    this.boltz = s.units.boltz;
    this.nktv2p = s.units.nktv2p;
    if (this.pstat && this.vol0 === 0.0) this.vol0 = this.sys.geom.volume(this.sys.dimension);
    this.resetDt();
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    this.dthalf = 0.5 * s.dt;
    this.dt4 = 0.25 * s.dt;
    this.dt8 = 0.125 * s.dt;
    this.tdragFactor = 1.0 - (s.dt * this.tFreq * this.drag / this.ncTchain);
    this.pdragFactor = 1.0 - (s.dt * this.pFreqMax * this.drag / this.ncPchain);
  }

  /** fix_nh.cpp compute_temp_target: the ramp of the target and ke_target = tdof kB T. */
  protected computeTempTarget(): void {
    this.tTarget = ramp(this.sys, this.tStart, this.tStop);
    this.keTarget = this.tdof * this.boltz * this.tTarget;
  }

  /** fix_nh.cpp compute_press_target (iso): the ramp of the pressure target and its hydrostatic part. */
  protected computePressTarget(): void {
    this.pTarget = ramp(this.sys, this.pStart, this.pStop);
    this.pHydro = this.pTarget;
  }

  /** the pressure of the isotropic barostat (fix_nh.cpp couple, ISO): the scalar of the pressure compute. */
  protected couple(): void {
    this.pCurrent = temperatureOf(this.sys, this.pressCompute());
  }

  /** fix_nh.cpp setup: temperatures, the chain masses, the barostat masses and the first chain accelerations. */
  setup(): void {
    this.tCurrent = temperatureOf(this.sys, this.tempCompute());
    this.tdof = this.tempCompute().dof;
    if (this.tstat) {
      this.computeTempTarget();
    } else if (this.pstat) {
      if (this.t0 === 0.0) this.t0 = temperatureOf(this.sys, this.tempCompute());
      this.tTarget = this.t0;
    }
    if (this.pstat) {
      this.computePressTarget();
      this.couple();
    }
    if (this.tstat) {
      this.etaMass[0] = this.tdof * this.boltz * this.tTarget / (this.tFreq * this.tFreq);
      for (let ich = 1; ich < this.mtchain; ich++) this.etaMass[ich] = this.boltz * this.tTarget / (this.tFreq * this.tFreq);
      for (let ich = 1; ich < this.mtchain; ich++) {
        this.etaDotdot[ich] = (this.etaMass[ich - 1] * this.etaDot[ich - 1] * this.etaDot[ich - 1] - this.boltz * this.tTarget) / this.etaMass[ich];
      }
    }
    if (this.pstat) {
      const kt = this.boltz * this.tTarget;
      const nkt = (this.sys.state.n + 1) * kt;
      this.omegaMass = nkt / (this.pFreq * this.pFreq);
      if (this.mpchain) {
        this.etapMass[0] = this.boltz * this.tTarget / (this.pFreqMax * this.pFreqMax);
        for (let ich = 1; ich < this.mpchain; ich++) this.etapMass[ich] = this.boltz * this.tTarget / (this.pFreqMax * this.pFreqMax);
        for (let ich = 1; ich < this.mpchain; ich++) {
          this.etapDotdot[ich] = (this.etapMass[ich - 1] * this.etapDot[ich - 1] * this.etapDot[ich - 1] - this.boltz * this.tTarget) / this.etapMass[ich];
        }
      }
    }
  }

  /** fix_nh.cpp initial_integrate: barostat chain, thermostat chain, barostat half step, kick, box remap, drift, box remap. */
  initialIntegrate(): void {
    if (this.pstat && this.mpchain) this.nhcPressIntegrate();
    if (this.tstat) {
      this.computeTempTarget();
      this.nhcTempIntegrate();
    }
    if (this.pstat) {
      temperatureOf(this.sys, this.tempCompute());
      this.couple();
    }
    if (this.pstat) {
      this.computePressTarget();
      this.nhOmegaDot();
      this.nhVPress();
    }
    this.nveV();
    if (this.pstat) this.remap();
    this.nveX();
    if (this.pstat) this.remap();
  }

  /** fix_nh.cpp final_integrate: kick, barostat half step, temperature, pressure, chains. */
  finalIntegrate(): void {
    this.nveV();
    this.tCurrent = temperatureOf(this.sys, this.tempCompute());
    if (this.pstat) this.nhVPress();
    this.tCurrent = temperatureOf(this.sys, this.tempCompute());
    this.tdof = this.tempCompute().dof;
    if (this.pstat) {
      this.couple();
      this.nhOmegaDot();
    }
    if (this.tstat) this.nhcTempIntegrate();
    if (this.pstat && this.mpchain) this.nhcPressIntegrate();
  }

  /** fix_nh.cpp nhc_temp_integrate: the Martyna-Tuckerman-Klein chain with tloop sub-steps (eta_mass_flag is set). */
  protected nhcTempIntegrate(): void {
    const { boltz } = this;
    let kecurrent = this.tdof * boltz * this.tCurrent;
    this.etaMass[0] = this.tdof * boltz * this.tTarget / (this.tFreq * this.tFreq);
    for (let ich = 1; ich < this.mtchain; ich++) this.etaMass[ich] = boltz * this.tTarget / (this.tFreq * this.tFreq);
    const eta = this.eta, etaDot = this.etaDot, etaDotdot = this.etaDotdot, etaMass = this.etaMass;
    const { dt8, dt4, dthalf } = this;
    if (etaMass[0] > 0.0) etaDotdot[0] = (kecurrent - this.keTarget) / etaMass[0];
    else etaDotdot[0] = 0.0;
    const ncfac = 1.0 / this.ncTchain;
    for (let iloop = 0; iloop < this.ncTchain; iloop++) {
      for (let ich = this.mtchain - 1; ich > 0; ich--) {
        let expfac = libmExp(-ncfac * dt8 * etaDot[ich + 1]);
        etaDot[ich] *= expfac;
        etaDot[ich] += etaDotdot[ich] * ncfac * dt4;
        etaDot[ich] *= this.tdragFactor;
        etaDot[ich] *= expfac;
      }
      let expfac = libmExp(-ncfac * dt8 * etaDot[1]);
      etaDot[0] *= expfac;
      etaDot[0] += etaDotdot[0] * ncfac * dt4;
      etaDot[0] *= this.tdragFactor;
      etaDot[0] *= expfac;

      this.factorEta = libmExp(-ncfac * dthalf * etaDot[0]);
      this.nhVTemp();
      this.tCurrent *= this.factorEta * this.factorEta;
      kecurrent = this.tdof * boltz * this.tCurrent;
      if (etaMass[0] > 0.0) etaDotdot[0] = (kecurrent - this.keTarget) / etaMass[0];
      else etaDotdot[0] = 0.0;
      for (let ich = 0; ich < this.mtchain; ich++) eta[ich] += ncfac * dthalf * etaDot[ich];
      etaDot[0] *= expfac;
      etaDot[0] += etaDotdot[0] * ncfac * dt4;
      etaDot[0] *= expfac;
      for (let ich = 1; ich < this.mtchain; ich++) {
        expfac = libmExp(-ncfac * dt8 * etaDot[ich + 1]);
        etaDot[ich] *= expfac;
        etaDotdot[ich] = (etaMass[ich - 1] * etaDot[ich - 1] * etaDot[ich - 1] - boltz * this.tTarget) / etaMass[ich];
        etaDot[ich] += etaDotdot[ich] * ncfac * dt4;
        etaDot[ich] *= expfac;
      }
    }
  }

  /** fix_nh.cpp nhc_press_integrate (iso, omega_mass_flag and etap_mass_flag not set): the barostat chain. */
  protected nhcPressIntegrate(): void {
    const kt = this.boltz * this.tTarget;
    const etap = this.etap, etapDot = this.etapDot, etapDotdot = this.etapDotdot, etapMass = this.etapMass;
    const { dt8, dt4, dthalf } = this;
    // kecurrent sums omega_mass omega_dot^2 over the barostatted dimensions
    let kecurrent = 0.0;
    for (let d = 0; d < this.pdim; d++) kecurrent += this.omegaMass * this.omegaDot * this.omegaDot;
    const lktPress = kt;
    etapDotdot[0] = (kecurrent - lktPress) / etapMass[0];
    const ncfac = 1.0 / this.ncPchain;
    for (let iloop = 0; iloop < this.ncPchain; iloop++) {
      for (let ich = this.mpchain - 1; ich > 0; ich--) {
        let expfac = libmExp(-ncfac * dt8 * etapDot[ich + 1]);
        etapDot[ich] *= expfac;
        etapDot[ich] += etapDotdot[ich] * ncfac * dt4;
        etapDot[ich] *= this.pdragFactor;
        etapDot[ich] *= expfac;
      }
      let expfac = libmExp(-ncfac * dt8 * etapDot[1]);
      etapDot[0] *= expfac;
      etapDot[0] += etapDotdot[0] * ncfac * dt4;
      etapDot[0] *= this.pdragFactor;
      etapDot[0] *= expfac;
      for (let ich = 0; ich < this.mpchain; ich++) etap[ich] += ncfac * dthalf * etapDot[ich];
      const factorEtap = libmExp(-ncfac * dthalf * etapDot[0]);
      this.omegaDot *= factorEtap;
      kecurrent = 0.0;
      for (let d = 0; d < this.pdim; d++) kecurrent += this.omegaMass * this.omegaDot * this.omegaDot;
      etapDotdot[0] = (kecurrent - lktPress) / etapMass[0];
      etapDot[0] *= expfac;
      etapDot[0] += etapDotdot[0] * ncfac * dt4;
      etapDot[0] *= expfac;
      for (let ich = 1; ich < this.mpchain; ich++) {
        expfac = libmExp(-ncfac * dt8 * etapDot[ich + 1]);
        etapDot[ich] *= expfac;
        etapDotdot[ich] = (etapMass[ich - 1] * etapDot[ich - 1] * etapDot[ich - 1] - this.boltz * this.tTarget) / etapMass[ich];
        etapDot[ich] += etapDotdot[ich] * ncfac * dt4;
        etapDot[ich] *= expfac;
      }
    }
  }

  /** fix_nh.cpp nh_omega_dot (iso, mtk): the barostat acceleration and its damping. */
  protected nhOmegaDot(): void {
    const volume = this.sys.geom.volume(this.sys.dimension);
    this.mtkTerm1 = this.tdof * this.boltz * this.tCurrent;
    this.mtkTerm1 /= this.pdim * this.sys.state.n;
    const fOmega = (this.pCurrent - this.pHydro) * volume / (this.omegaMass * this.nktv2p) + this.mtkTerm1 / this.omegaMass;
    this.omegaDot += fOmega * this.dthalf;
    this.omegaDot *= this.pdragFactor;
    // the sum over the barostatted dimensions (each carries omega_dot)
    this.mtkTerm2 = 0.0;
    for (let d = 0; d < this.pdim; d++) this.mtkTerm2 += this.omegaDot;
    this.mtkTerm2 /= this.pdim * this.sys.state.n;
  }

  /** fix_nh.cpp nh_v_press (NOBIAS): the velocities are scaled by the barostat half step, twice (dt/4 each). */
  protected nhVPress(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const factor = libmExp(-this.dt4 * (this.omegaDot + this.mtkTerm2));
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      v[3 * i] *= factor; v[3 * i + 1] *= factor; v[3 * i + 2] *= factor;
      v[3 * i] *= factor; v[3 * i + 1] *= factor; v[3 * i + 2] *= factor;
    }
  }

  /**
   * fix_nh.cpp remap (orthogonal box, iso, allremap): omega advances by dto omega_dot, the box lengths scale about the
   * fixed point, and every atom keeps its fractional coordinates. dto is dt/2 (one half step).
   */
  protected remap(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const b = s.box;
    this.omega += this.dthalf * this.omegaDot;
    const lam = new Float64Array(3 * s.n);
    const tmp = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      lam[3 * i] = tmp[0]; lam[3 * i + 1] = tmp[1]; lam[3 * i + 2] = tmp[2];
    }
    for (let d = 0; d < 3; d++) {
      const e = libmExp(this.dthalf * this.omegaDot);
      b.lo[d] = (b.lo[d] - this.fixedpoint[d]) * e + this.fixedpoint[d];
      b.hi[d] = (b.hi[d] - this.fixedpoint[d]) * e + this.fixedpoint[d];
    }
    b.minLo = [...b.lo]; b.minHi = [...b.hi];
    g.update();
    for (let i = 0; i < s.n; i++) {
      g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
    }
    for (const f of sys.fixes) if (f !== this) f.boxChanged?.();
  }

  /** fix_nh.cpp nve_v: v += dtf F / m over the group. */
  protected nveV(): void {
    const s = this.sys.state;
    const { v, f, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const dtfm = this.dtf / massOf(s, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
    }
  }

  /** fix_nh.cpp nve_x: x += dtv v over the group. */
  protected nveX(): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      x[3 * i] += this.dtv * v[3 * i];
      x[3 * i + 1] += this.dtv * v[3 * i + 1];
      x[3 * i + 2] += this.dtv * v[3 * i + 2];
    }
  }

  /** fix_nh.cpp nh_v_temp (NOBIAS): v *= factor_eta over the group. */
  protected nhVTemp(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      v[3 * i] *= this.factorEta;
      v[3 * i + 1] *= this.factorEta;
      v[3 * i + 2] *= this.factorEta;
    }
  }

  hasTemp(): boolean { return this.tstat; }
  hasPress(): boolean { return this.pstat; }

  /** fix_nh.cpp compute_scalar: the chain energies of the thermostat and of the barostat (iso). */
  computeScalar(): number {
    const kt = this.boltz * this.tTarget;
    let energy = 0.0;
    if (this.tstat) {
      energy += this.keTarget * this.eta[0] + 0.5 * this.etaMass[0] * this.etaDot[0] * this.etaDot[0];
      for (let ich = 1; ich < this.mtchain; ich++) {
        energy += kt * this.eta[ich] + 0.5 * this.etaMass[ich] * this.etaDot[ich] * this.etaDot[ich];
      }
    }
    if (this.pstat) {
      const volume = this.sys.geom.volume(this.sys.dimension);
      // the barostat energy and its kT count are summed over the barostatted dimensions, as in fix_nh.cpp compute_scalar
      let lktPress = 0.0;
      for (let d = 0; d < this.pdim; d++) {
        energy += 0.5 * this.omegaDot * this.omegaDot * this.omegaMass + this.pHydro * (volume - this.vol0) / (this.pdim * this.nktv2p);
        lktPress += kt;
      }
      if (this.mpchain) {
        energy += lktPress * this.etap[0] + 0.5 * this.etapMass[0] * this.etapDot[0] * this.etapDot[0];
        for (let ich = 1; ich < this.mpchain; ich++) {
          energy += kt * this.etap[ich] + 0.5 * this.etapMass[ich] * this.etapDot[ich] * this.etapDot[ich];
        }
      }
    }
    return energy;
  }
}
