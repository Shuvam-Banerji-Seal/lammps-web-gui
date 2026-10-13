/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/QTB/fix_qbmsst.cpp and fix_qbmsst.h
 * (class FixQBMSST, fix qbmsst: the multi-scale shock method with quantum nuclear effects, the cell's dilation
 * driven by the shock Rayleigh and Hugoniot conditions and the colored thermostat of fix qtb with its temperature
 * updated from the shock energy). Contributing authors: Shen, Yuan, Qi, Tingting, and Reed, Evan.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_qbmsst.html): the keywords are q, mu, p0, v0, e0, tscale, damp, seed,
 * f_max, N_f, eta, beta and T_init; "fix qbmsst ... computes 5 global vector values: dhug, dray, lgr_vel, lgr_pos
 * and T_qm"; "The fix creates the computes ID_temp, ID_press and ID_pe" and "fix_modify temp and press" replace them.
 */
import { Fix } from '../../engine/fix/fix';
import { ownCompute, removeCompute } from '../../engine/fix/util';
import { StyleError } from '../../engine/force/types';
import { parseInt_, parseNum } from '../../engine/force/util';
import { nativeOrder } from '../../engine/atoms';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import { RanMars } from '../random_mars';
import { colorFilter } from './fix_qtb';

export class FixQBMSST extends Fix {
  readonly style = 'qbmsst';
  private readonly direction: number;
  private readonly velocity: number;
  private qmass = 1.0e1;
  private mu = 0.0;
  private e0 = 0.0;
  private v0 = 1.0;
  private p0 = 0.0;
  private p0Set = false;
  private v0Set = false;
  private e0Set = false;
  private tscale = 0.01;
  private tPeriod = 1.0;
  private fricCoef = 1.0;
  private seed = 880302;
  private fMax = 200.0;
  private nf = 100;
  private eta = 1.0;
  private beta = 100;
  private tInit = 300.0;
  private qtbSet = false;
  private counterL = 0;
  private counterMu = 0;
  private tCurrent = 0;
  private hTimestep = 0;
  private alpha = 1;
  private readonly random: RanMars;
  private gfactor: Float64Array;
  private omegaH: Float64Array;
  private timeH: Float64Array;
  private ra: [Float64Array, Float64Array, Float64Array] = [new Float64Array(0), new Float64Array(0), new Float64Array(0)];
  private fran = new Float64Array(0);
  private oldEavg = 0;
  private dtv = 0;
  private dthalf = 0;
  private ntotal = 0;
  private boltz = 1;
  private nktv2p = 1;
  private mvv2e = 1;
  private ftm2v = 1;
  private totalMass = 0;
  private readonly dilation = [1, 1, 1];
  private readonly omega = [0, 0, 0];
  private readonly pCurrent = [0, 0, 0];
  private velocitySum = 0;
  private lagrangianPosition = 0;
  private readonly idTemp: string;
  private readonly idPress: string;
  private readonly idPe: string;
  private temperature!: Compute;
  private pressure!: Compute;
  private pe!: Compute;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal fix qbmsst command: usage fix ID group qbmsst x|y|z velocity keywords');
    const dir = args[0];
    if (dir === 'x') this.direction = 0;
    else if (dir === 'y') this.direction = 1;
    else if (dir === 'z') this.direction = 2;
    else throw new StyleError(`Unknown fix qbmsst direction keyword: ${dir}`);
    this.velocity = parseNum(args[1], 'fix qbmsst velocity');
    if (this.velocity < 0) throw new StyleError(`Illegal fix qbmsst velocity value ${this.velocity}`);
    this.timeIntegrate = true;
    this.energyGlobal = true;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 5;
    this.extscalar = 1;
    this.extvector = 0;
    this.nevery = 1;
    for (let k = 2; k < args.length;) {
      const key = args[k];
      const val = args[k + 1];
      switch (key) {
        case 'q': this.qmass = parseNum(val, 'fix qbmsst q'); if (this.qmass < 0) throw new StyleError('Fix qbmsst qmass must be >= 0.0'); break;
        case 'mu': this.mu = parseNum(val, 'fix qbmsst mu'); if (this.mu < 0) throw new StyleError('Fix qbmsst mu must be >= 0.0'); break;
        case 'p0': this.p0 = parseNum(val, 'fix qbmsst p0'); if (this.p0 < 0) throw new StyleError('Fix qbmsst p0 must be >= 0.0'); this.p0Set = true; break;
        case 'v0': this.v0 = parseNum(val, 'fix qbmsst v0'); if (this.v0 < 0) throw new StyleError('Fix qbmsst v0 must be >= 0.0'); this.v0Set = true; break;
        case 'e0': this.e0 = parseNum(val, 'fix qbmsst e0'); this.e0Set = true; break;
        case 'tscale':
          this.tscale = parseNum(val, 'fix qbmsst tscale');
          if (this.tscale < 0.0 || this.tscale > 1.0) throw new StyleError('Fix qbmsst tscale must satisfy 0 <= tscale < 1');
          break;
        case 'damp':
          this.tPeriod = parseNum(val, 'fix qbmsst damp');
          if (this.tPeriod <= 0.0) throw new StyleError('Fix qbmsst damp must be > 0.0');
          this.fricCoef = 1 / this.tPeriod;
          break;
        case 'seed': this.seed = parseInt_(val, 'fix qbmsst seed'); if (this.seed <= 0) throw new StyleError('Fix qbmsst seed must be a positive integer'); break;
        case 'f_max': this.fMax = parseNum(val, 'fix qbmsst f_max'); if (this.fMax <= 0) throw new StyleError('Fix qbmsst f_max must be > 0.0'); break;
        case 'N_f': this.nf = parseInt_(val, 'fix qbmsst N_f'); if (this.nf <= 0) throw new StyleError('Fix qbmsst N_f must be a positive integer'); break;
        case 'eta': this.eta = parseNum(val, 'fix qbmsst eta'); if (this.eta <= 0) throw new StyleError('Fix qbmsst eta must be >= 0.0'); break;
        case 'beta': this.beta = parseInt_(val, 'fix qbmsst beta'); if (this.beta <= 0) throw new StyleError('Fix qbmsst beta must be a positive integer'); break;
        case 'T_init': this.tInit = parseNum(val, 'fix qbmsst T_init'); if (this.tInit <= 0) throw new StyleError('Fix qbmsst T_init must be >= 0.0'); break;
        default: throw new StyleError(`Illegal fix qbmsst command: unknown keyword '${key}'`);
      }
      if (val === undefined) throw new StyleError('Illegal fix qbmsst command');
      k += 2;
    }
    if (!sys.geom.periodic.every((p) => p)) throw new StyleError('Fix qbmsst requires a periodic box');
    this.random = new RanMars(this.seed);
    const nt = sys.state.ntypes;
    this.gfactor = new Float64Array(nt + 1);
    this.omegaH = new Float64Array(2 * this.nf);
    this.timeH = new Float64Array(2 * this.nf);
    this.growArrays(sys.state.n);
    // the computes the fix creates: temperature of all atoms, pressure with that temperature, potential energy
    this.idTemp = `${id}_temp`;
    this.idPress = `${id}_press`;
    this.idPe = `${id}_pe`;
    ownCompute(sys, this.idTemp, 'all', 'temp', []);
    ownCompute(sys, this.idPress, 'all', 'pressure', [this.idTemp]);
    ownCompute(sys, this.idPe, 'all', 'pe', []);
    this.bindComputes();
  }

  private bindComputes(): void {
    this.temperature = this.sys.compute(this.idTemp);
    this.pressure = this.sys.compute(this.idPress);
    this.pe = this.sys.compute(this.idPe);
  }

  private growArrays(n: number): void {
    const m = 2 * this.nf;
    if (this.fran.length === 3 * n && this.ra[0].length === m * n) return;
    this.ra = [new Float64Array(m * n), new Float64Array(m * n), new Float64Array(m * n)];
    this.fran = new Float64Array(3 * n);
  }

  destroy(): void {
    removeCompute(this.sys, this.idTemp);
    removeCompute(this.sys, this.idPress);
    removeCompute(this.sys, this.idPe);
  }

  modify(key: string, values: string[]): number {
    if (key === 'temp') {
      const c = this.sys.compute(values[0] ?? '');
      if (!c.tempFlag) throw new StyleError(`Fix_modify temperature ID ${values[0]} does not compute temperature`);
      if (values[0] !== this.idTemp) removeCompute(this.sys, this.idTemp);
      this.temperature = c;
      return 1;
    }
    if (key === 'press') {
      const c = this.sys.compute(values[0] ?? '');
      if (!c.pressFlag) throw new StyleError(`Fix_modify compute pressure ID ${values[0]} does not compute pressure`);
      if (values[0] !== this.idPress) removeCompute(this.sys, this.idPress);
      this.pressure = c;
      return 1;
    }
    return super.modify(key, values);
  }

  init(): void {
    const s = this.sys.state;
    this.bindComputes();
    if (this.sys.fixes.some((f) => f !== this && f.style.startsWith('rigid'))) {
      throw new StyleError('fix qbmsst with rigid bodies is not supported by the browser engine yet');
    }
    const u = s.units;
    this.dtv = s.dt;
    this.dthalf = 0.5 * s.dt;
    this.ntotal = s.n;
    this.boltz = u.boltz;
    this.nktv2p = u.nktv2p;
    this.mvv2e = u.mvv2e;
    this.ftm2v = u.ftm2v;
    this.growArrays(s.n);
    this.counterL = 0;
    this.counterMu = 0;
    if (!this.qtbSet) {
      this.tCurrent = this.tInit;
      this.qtbSet = true;
    }
    this.oldEavg = this.e0;
    if (Math.trunc(1.0 / (2 * this.fMax * this.dtv)) === 0) {
      this.sys.warn('Either f_max is too high or the time step is too big, setting f_max to be 1/timestep!');
      this.hTimestep = this.dtv;
      this.alpha = 1;
    } else {
      this.alpha = Math.trunc(1.0 / (2 * this.fMax * this.dtv));
      this.hTimestep = this.alpha * this.dtv;
    }
    for (let i = 1; i <= s.ntypes; i++) {
      this.gfactor[i] = Math.sqrt(2 * this.fricCoef * s.massByType[i]) * Math.sqrt(u.mvv2e) * Math.sqrt(12 / this.hTimestep);
    }
    // random number arrays with zero mean and variance 1/12, atom by atom in native order
    const m2 = 2 * this.nf;
    for (const i of nativeOrder(s)) {
      for (let m = 0; m < m2; m++) {
        this.ra[0][i * m2 + m] = this.random.uniform() - 0.5;
        this.ra[1][i * m2 + m] = this.random.uniform() - 0.5;
        this.ra[2][i * m2 + m] = this.random.uniform() - 0.5;
      }
    }
    this.dilation[0] = this.dilation[1] = this.dilation[2] = 1.0;
    this.omega[0] = this.omega[1] = this.omega[2] = 0.0;
    let mass = 0.0;
    for (let i = 0; i < s.n; i++) mass += s.massByType[s.type[i]];
    this.totalMass = mass;
  }

  /** The volume of the cell: xprd yprd zprd in 3d, xprd yprd in 2d (fix_qbmsst.cpp compute_vol). */
  private computeVol(): number {
    const g = this.sys.geom;
    return this.sys.state.dimension === 3 ? g.lx * g.ly * g.lz : g.lx * g.ly;
  }

  private freshComputes(): void {
    this.sys.refreshComputes();
  }

  /** couple(): the pressure tensor diagonal of the pressure compute. */
  private couple(): void {
    const t = this.pressure.vectorValues();
    this.pCurrent[0] = t[0];
    this.pCurrent[1] = t[1];
    this.pCurrent[2] = t[2];
  }

  private computeVsum(): number {
    const s = this.sys.state;
    let t = 0.0;
    for (const i of nativeOrder(s)) {
      if (s.mask[i] & this.groupBit) t += s.v[3 * i] * s.v[3 * i] + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2];
    }
    return t;
  }

  /** Kinetic plus potential energy of the atoms (compute_etotal). */
  private computeEtotal(): number {
    const epot = this.pe.scalarValue();
    const ekin = this.temperature.scalarValue() * 0.5 * this.temperature.dof * this.boltz;
    return epot + ekin;
  }

  /** Atomic energy plus the cell's external (shock) term (compute_egrand). */
  private computeEgrand(): number {
    const epot = this.pe.scalarValue();
    const ekin = this.temperature.scalarValue() * 0.5 * this.temperature.dof * this.boltz;
    return epot + ekin + this.computeScalarValue();
  }

  setup(): void {
    this.lagrangianPosition = 0.0;
    this.freshComputes();
    this.temperature.vectorValues();
    this.pressure.vectorValues();
    this.couple();
    this.velocitySum = this.computeVsum();
    const d = this.direction;
    if (!this.v0Set) {
      this.v0 = this.computeVol();
      this.v0Set = true;
      this.sys.log(`Fix QBMSST v0 = ${this.v0.toExponential(5)}`);
    }
    if (!this.p0Set) {
      this.p0 = this.pCurrent[d];
      this.p0Set = true;
      this.sys.log(`Fix QBMSST p0 = ${this.p0.toExponential(5)}`);
    }
    if (!this.e0Set) {
      this.e0 = this.computeEtotal();
      this.e0Set = true;
      this.oldEavg = this.e0;
      this.sys.log(`Fix QBMSST e0 = to be ${this.e0.toExponential(5)}`);
    }
    const kt = this.temperature.vectorValues();
    const keTemp = kt[0] + kt[1] + kt[2];
    if (keTemp > 0.0 && this.tscale > 0.0) {
      const s = this.sys.state;
      // transfer energy from atom velocities to cell volume motion to bias the initial compression
      const scale = Math.sqrt(1.0 - this.tscale);
      const fac1 = (this.tscale * this.totalMass) / this.qmass * keTemp / this.mvv2e;
      this.omega[d] = -1 * Math.sqrt(fac1);
      for (let i = 0; i < s.n; i++) {
        if (s.mask[i] & this.groupBit) for (let k = 0; k < 3; k++) s.v[3 * i + k] *= scale;
      }
    }
  }

  /** Colored-noise filter rebuilt at the current qtb temperature (initial_integrate). */
  private rebuildFilter(): void {
    colorFilter(this.nf, this.alpha, this.hTimestep, this.boltz, this.sys.state.units.hplanck, this.tCurrent, this.omegaH, this.timeH);
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const sd = this.direction;
    const n = s.n;
    const m2 = 2 * this.nf;
    this.freshComputes();
    this.temperature.vectorValues();
    this.pressure.vectorValues();
    this.couple();
    const vol = this.computeVol();
    // decide if the qtb temperature needs to be updated
    if (this.counterL === 0) {
      this.tCurrent -= this.dtv * this.fricCoef * this.eta * this.beta * (this.oldEavg - this.e0) / (3 * this.ntotal * this.boltz);
      if (this.tCurrent > 0.0) {
        this.oldEavg = 0;
        this.rebuildFilter();
      }
    }
    const order = nativeOrder(s);
    const [ra0, ra1, ra2] = this.ra;
    const fran = this.fran;
    const f = s.f, v = s.v, type = s.type, mask = s.mask;
    // update the colored random force every alpha MD steps
    if (this.counterMu === 0) {
      for (const j of order) {
        for (let m = 0; m < m2 - 1; m++) {
          ra0[j * m2 + m] = ra0[j * m2 + m + 1];
          ra1[j * m2 + m] = ra1[j * m2 + m + 1];
          ra2[j * m2 + m] = ra2[j * m2 + m + 1];
        }
        ra0[j * m2 + m2 - 1] = this.random.uniform() - 0.5;
        ra1[j * m2 + m2 - 1] = this.random.uniform() - 0.5;
        ra2[j * m2 + m2 - 1] = this.random.uniform() - 0.5;
        fran[3 * j] = 0.0;
        fran[3 * j + 1] = 0.0;
        fran[3 * j + 2] = 0.0;
        if (mask[j] & this.groupBit) {
          const gamma3 = this.gfactor[type[j]];
          for (let m = 0; m < m2; m++) {
            fran[3 * j] += this.timeH[m] * ra0[j * m2 + m2 - m - 1];
            fran[3 * j + 1] += this.timeH[m] * ra1[j * m2 + m2 - m - 1];
            fran[3 * j + 2] += this.timeH[m] * ra2[j * m2 + m2 - m - 1];
          }
          fran[3 * j] = fran[3 * j] * gamma3;
          fran[3 * j + 1] = fran[3 * j + 1] * gamma3;
          fran[3 * j + 2] = fran[3 * j + 2] * gamma3;
        }
      }
    }
    // estimate the old energy average in this step
    this.oldEavg = this.oldEavg + this.computeEgrand() / this.beta;
    this.counterL = (this.counterL + 1) % this.beta;
    this.counterMu = (this.counterMu + 1) % this.alpha;
    // propagate the time derivative of the volume 1/2 step at fixed vol, r, rdot
    const pQbmsst = this.nktv2p * this.mvv2e * this.velocity * this.velocity * this.totalMass * (this.v0 - vol) / (this.v0 * this.v0);
    let A = this.totalMass * (this.pCurrent[sd] - this.p0 - pQbmsst) / (this.qmass * this.nktv2p * this.mvv2e);
    const B = this.totalMass * this.mu / (this.qmass * vol);
    // prevent blow-up of the volume
    if (vol > this.v0 && A > 0.0) A = -A;
    // Taylor expansion to avoid the singularity at B = 0
    if (B * this.dthalf > 1.0e-06) {
      this.omega[sd] = (this.omega[sd] + A * (Math.exp(B * this.dthalf) - 1.0) / B) * Math.exp(-B * this.dthalf);
    } else {
      this.omega[sd] = this.omega[sd] + (A - B * this.omega[sd]) * this.dthalf +
        0.5 * (B * B * this.omega[sd] - A * B) * this.dthalf * this.dthalf;
    }
    // propagate the velocity sum 1/2 step by temporarily propagating the velocities
    this.velocitySum = this.computeVsum();
    const oldV = new Float64Array(3 * n);
    const omegaD = this.omega[sd];
    const halfKick = (keepOld: boolean): void => {
      for (let i = 0; i < n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        const m = s.massByType[type[i]];
        for (let k = 0; k < 3; k++) {
          const C = (f[3 * i + k] + fran[3 * i + k]) * this.ftm2v / m;
          let D = this.mu * omegaD * omegaD / (this.velocitySum * m * vol) - this.fricCoef;
          if (keepOld) oldV[3 * i + k] = v[3 * i + k];
          if (k === sd) D = D - 2.0 * omegaD / vol;
          if (Math.abs(this.dthalf * D) > 1.0e-06) {
            const expd = Math.exp(D * this.dthalf);
            v[3 * i + k] = expd * (C + D * v[3 * i + k] - C / expd) / D;
          } else {
            v[3 * i + k] = v[3 * i + k] + (C + D * v[3 * i + k]) * this.dthalf +
              0.5 * (D * D * v[3 * i + k] + C * D) * this.dthalf * this.dthalf;
          }
        }
      }
    };
    halfKick(true);
    this.velocitySum = this.computeVsum();
    // reset the velocities
    for (let i = 0; i < n; i++) {
      if (mask[i] & this.groupBit) for (let k = 0; k < 3; k++) v[3 * i + k] = oldV[3 * i + k];
    }
    // propagate velocities 1/2 step using the new velocity sum
    halfKick(false);
    // propagate the volume 1/2 step, rescale positions and change the box size
    const vol1 = vol + this.omega[sd] * this.dthalf;
    this.dilation[sd] = vol1 / vol;
    this.remap();
    // propagate particle positions 1 time step
    const x = s.x;
    for (let i = 0; i < n; i++) {
      if (mask[i] & this.groupBit) {
        x[3 * i] += this.dtv * v[3 * i];
        x[3 * i + 1] += this.dtv * v[3 * i + 1];
        x[3 * i + 2] += this.dtv * v[3 * i + 2];
      }
    }
    // propagate the volume 1/2 step
    const vol2 = vol1 + this.omega[sd] * this.dthalf;
    this.dilation[sd] = vol2 / vol1;
    this.remap();
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const sd = this.direction;
    const n = s.n;
    const f = s.f, v = s.v, type = s.type, mask = s.mask;
    let vol = this.computeVol();
    // propagate particle velocities 1/2 step
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = s.massByType[type[i]];
      for (let k = 0; k < 3; k++) {
        const C = (f[3 * i + k] + this.fran[3 * i + k]) * this.ftm2v / m;
        let D = this.mu * this.omega[sd] * this.omega[sd] / (this.velocitySum * m * vol) - this.fricCoef;
        if (k === sd) D = D - 2.0 * this.omega[sd] / vol;
        if (Math.abs(this.dthalf * D) > 1.0e-06) {
          const expd = Math.exp(D * this.dthalf);
          v[3 * i + k] = expd * (C + D * v[3 * i + k] - C / expd) / D;
        } else {
          v[3 * i + k] = v[3 * i + k] + (C + D * v[3 * i + k]) * this.dthalf +
            0.5 * (D * D * v[3 * i + k] + C * D) * this.dthalf * this.dthalf;
        }
      }
    }
    // compute new pressure and volume
    this.freshComputes();
    this.temperature.vectorValues();
    this.pressure.vectorValues();
    this.couple();
    this.velocitySum = this.computeVsum();
    vol = this.computeVol();
    // propagate the time derivative of the volume 1/2 step at fixed V, r, rdot
    const pQbmsst = this.nktv2p * this.mvv2e * this.velocity * this.velocity * this.totalMass * (this.v0 - vol) / (this.v0 * this.v0);
    let A = this.totalMass * (this.pCurrent[sd] - this.p0 - pQbmsst) / (this.qmass * this.nktv2p * this.mvv2e);
    const B = this.totalMass * this.mu / (this.qmass * vol);
    if (vol > this.v0 && A > 0.0) A = -A;
    if (B * this.dthalf > 1.0e-06) {
      this.omega[sd] = (this.omega[sd] + A * (Math.exp(B * this.dthalf) - 1.0) / B) * Math.exp(-B * this.dthalf);
    } else {
      this.omega[sd] = this.omega[sd] + (A - B * this.omega[sd]) * this.dthalf +
        0.5 * (B * B * this.omega[sd] - A * B) * this.dthalf * this.dthalf;
    }
    // Lagrangian position of the computational cell
    this.lagrangianPosition -= this.velocity * vol / this.v0 * this.sys.state.dt;
  }

  /**
   * Dilates the box along the shock direction about its centre and rescales the atoms (all of them, as native does)
   * and the velocity component along the shock (remap in fix_qbmsst.cpp).
   */
  private remap(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const b = s.box;
    const d = this.direction;
    const n = s.n;
    const lam = new Float64Array(3 * n);
    const tmp = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      lam[3 * i] = tmp[0]; lam[3 * i + 1] = tmp[1]; lam[3 * i + 2] = tmp[2];
    }
    const oldlo = b.lo[d], oldhi = b.hi[d];
    const ctr = 0.5 * (oldlo + oldhi);
    b.lo[d] = (oldlo - ctr) * this.dilation[d] + ctr;
    b.hi[d] = (oldhi - ctr) * this.dilation[d] + ctr;
    b.minLo = [...b.lo];
    b.minHi = [...b.hi];
    g.update();
    for (let i = 0; i < n; i++) {
      g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
      s.v[3 * i + d] = s.v[3 * i + d] * this.dilation[d];
    }
    for (const f of sys.fixes) if (f !== this) f.boxChanged?.();
  }

  /** The fix's own energy term (compute_scalar), from fresh temperature, pressure and volume. */
  private computeScalarValue(): number {
    const vol = this.computeVol();
    const i = this.direction;
    let energy = (this.qmass * this.omega[i] * this.omega[i]) / (2.0 * this.totalMass) * this.mvv2e;
    energy -= 0.5 * this.totalMass * this.velocity * this.velocity * (1.0 - vol / this.v0) * (1.0 - vol / this.v0) * this.mvv2e;
    energy -= this.p0 * (this.v0 - vol) / this.nktv2p;
    return energy;
  }

  computeScalar(): number {
    this.freshComputes();
    this.temperature.vectorValues();
    this.pressure.vectorValues();
    this.couple();
    return this.computeScalarValue();
  }

  /** f_ID[n]: dhug, dray, lgr_vel, lgr_pos, T_qm (0-based here). */
  computeVector(n: number): number {
    if (n === 0) return this.computeHugoniot();
    if (n === 1) return this.computeRayleigh();
    if (n === 2) return this.velocity * (1.0 - this.computeVol() / this.v0);
    if (n === 3) return this.lagrangianPosition;
    if (n === 4) return this.tCurrent;
    return 0.0;
  }

  /** The deviation of the current point from the Hugoniot, in kelvin (compute_hugoniot). */
  private computeHugoniot(): number {
    const e = this.computeEtotal();
    this.freshComputes();
    this.temperature.vectorValues();
    const pt = this.pressure.vectorValues();
    const p = pt[this.direction];
    const v = this.computeVol();
    let dhugo = (0.5 * (p + this.p0) * (this.v0 - v)) / this.nktv2p + this.e0 - e;
    dhugo /= this.temperature.dof * this.boltz;
    return dhugo;
  }

  /** The deviation of the current point from the Rayleigh line, in pressure units (compute_rayleigh). */
  private computeRayleigh(): number {
    this.freshComputes();
    this.temperature.vectorValues();
    const pt = this.pressure.vectorValues();
    const p = pt[this.direction];
    const v = this.computeVol();
    return p - this.p0 - this.totalMass * this.velocity * this.velocity * this.mvv2e * (1.0 - v / this.v0) * this.nktv2p / this.v0;
  }

  /** Extends the fix's energy to the potential energy (fix_modify energy yes). */
  energy(): number { return this.computeScalar(); }

  /** Energy the cell exchanges with the shock (thermo econserve). */
  ecouple(): number { return this.computeScalar(); }
}
