/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SHOCK/fix_msst.cpp and fix_msst.h
 * (class FixMSST, fix msst: the multi-scale shock technique of Reed, Fried and Joannopoulos, Phys Rev Lett 90,
 * 235503 (2003), with the cell's dilation driven by the Rayleigh and Hugoniot conditions). Contributing authors:
 * Laurence Fried (LLNL) and Evan Reed (LLNL, Stanford). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_msst.html): "fix ID group-ID msst dir shockvel keyword value ..." with
 * dir x, y or z; the keywords q, mu, p0, v0, e0, tscale, dftb and beta. "Fix msst requires a periodic box" and
 * "fix msst ... computes 4 global vector values". The fix creates the computes ID_temp, ID_press and ID_pe, and
 * fix_modify temp and press replace them. "dftb yes" needs a fix external that feeds the DFTB+ electronic entropy;
 * the browser engine has no fix external, so that keyword is refused by name.
 */
import { Fix } from '../../engine/fix/fix';
import { ownCompute, removeCompute } from '../../engine/fix/util';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';

const usage = 'Illegal fix msst command: usage fix ID group msst x|y|z shockvel [q v] [mu v] [p0 v] [v0 v] [e0 v] [tscale v] [dftb yes|no] [beta v]';

/** fix_msst.cpp: "Logical" keywords accept yes or no. */
const logicalWord = (w: string | undefined): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in fix msst command`);
};

export class FixMSST extends Fix {
  readonly style = 'msst';
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
  private beta = 0.0;
  private lagrangianPosition = 0.0;
  private totalMass = 0;
  private boltz = 1;
  private nktv2p = 1;
  private mvv2e = 1;
  private ftm2v = 1;
  private dthalf = 0;
  private dtv = 0;
  private velocitySum = 0;
  private oldV = new Float64Array(0);
  private readonly omega = [0, 0, 0];
  private readonly dilation = [1, 1, 1];
  private readonly pCurrent = [0, 0, 0];
  private readonly idTemp: string;
  private readonly idPress: string;
  private readonly idPe: string;
  private temperature!: Compute;
  private pressure!: Compute;
  private pe!: Compute;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError(usage);
    const dir = args[0];
    if (dir === 'x') this.direction = 0;
    else if (dir === 'y') this.direction = 1;
    else if (dir === 'z') this.direction = 2;
    else throw new StyleError(usage);
    this.velocity = parseNum(args[1], 'fix msst shockvel');
    if (this.velocity < 0) throw new StyleError(usage);
    this.timeIntegrate = true;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 4;
    this.extscalar = 1;
    this.extvector = 0;
    this.nevery = 1;
    for (let k = 2; k < args.length;) {
      const key = args[k];
      const val = args[k + 1];
      if (val === undefined) throw new StyleError(usage);
      switch (key) {
        case 'q': this.qmass = parseNum(val, 'fix msst q'); break;
        case 'mu': this.mu = parseNum(val, 'fix msst mu'); break;
        case 'p0': this.p0 = parseNum(val, 'fix msst p0'); this.p0Set = true; break;
        case 'v0': this.v0 = parseNum(val, 'fix msst v0'); this.v0Set = true; break;
        case 'e0': this.e0 = parseNum(val, 'fix msst e0'); this.e0Set = true; break;
        case 'tscale':
          this.tscale = parseNum(val, 'fix msst tscale');
          if (this.tscale < 0.0 || this.tscale > 1.0) throw new StyleError('Fix msst tscale must satisfy 0 <= tscale < 1');
          break;
        case 'dftb':
          if (logicalWord(val)) {
            throw new StyleError('fix msst dftb yes needs fix external (the DFTB+ electronic entropy), which the browser engine does not support');
          }
          break;
        case 'beta':
          this.beta = parseNum(val, 'fix msst beta');
          if (this.beta < 0.0 || this.beta > 1.0) throw new StyleError(usage);
          break;
        default: throw new StyleError(usage);
      }
      k += 2;
    }
    if (!sys.geom.periodic.every((p) => p)) throw new StyleError('Fix msst requires a periodic box');
    // "id = fix-ID + MSST_temp" (compute group all), "fix-ID + MSST_press" (pressure of that temperature), "fix-ID + MSST_pe"
    this.idTemp = `${id}MSST_temp`;
    this.idPress = `${id}MSST_press`;
    this.idPe = `${id}MSST_pe`;
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
      if (!c.pressFlag) throw new StyleError(`Fix_modify pressure ID ${values[0]} does not compute pressure`);
      if (values[0] !== this.idPress) removeCompute(this.sys, this.idPress);
      this.pressure = c;
      return 1;
    }
    return super.modify(key, values);
  }

  init(): void {
    const s = this.sys.state;
    this.bindComputes();
    const u = s.units;
    this.dtv = s.dt;
    this.dthalf = 0.5 * s.dt;
    this.boltz = u.boltz;
    this.nktv2p = u.nktv2p;
    this.mvv2e = u.mvv2e;
    this.ftm2v = u.ftm2v;
    let mass = 0.0;
    for (let i = 0; i < s.n; i++) mass += s.massByType[s.type[i]];
    this.totalMass = mass;
  }

  /** The volume of the cell: xprd yprd zprd in 3d, xprd yprd in 2d (fix_msst.cpp compute_vol). */
  private computeVol(): number {
    const g = this.sys.geom;
    return this.sys.state.dimension === 3 ? g.lx * g.ly * g.lz : g.lx * g.ly;
  }

  /** The pressure tensor diagonal of the pressure compute (couple()). */
  private couple(): void {
    const t = this.pressure.vectorValues();
    this.pCurrent[0] = t[0];
    this.pCurrent[1] = t[1];
    this.pCurrent[2] = t[2];
  }

  private computeVsum(): number {
    const s = this.sys.state;
    let t = 0.0;
    for (let i = 0; i < s.n; i++) {
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

  /** Recomputes temperature, pressure tensor (couple) for the current state. */
  private fresh(): void {
    this.sys.refreshComputes();
    this.temperature.vectorValues();
    this.pressure.vectorValues();
    this.couple();
  }

  setup(): void {
    this.lagrangianPosition = 0.0;
    this.fresh();
    this.velocitySum = this.computeVsum();
    const d = this.direction;
    if (!this.v0Set) {
      this.v0 = this.computeVol();
      this.v0Set = true;
      this.sys.log(`Fix MSST v0 = ${this.v0.toExponential(8)}`);
    }
    if (!this.p0Set) {
      this.p0 = this.pCurrent[d];
      this.p0Set = true;
      this.sys.log(`Fix MSST p0 = ${this.p0.toExponential(8)}`);
    }
    if (!this.e0Set) {
      this.e0 = this.computeEtotal();
      this.e0Set = true;
      this.sys.log(`Fix MSST e0 = ${this.e0.toExponential(8)}`);
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

  /** One velocity update of the cell's dilation: the Taylor-safe exact step of the omega equation (dthalf). */
  private propagateOmega(vol: number): void {
    const sd = this.direction;
    const pMsst = this.nktv2p * this.mvv2e * this.velocity * this.velocity * this.totalMass * (this.v0 - vol) / (this.v0 * this.v0);
    let A = this.totalMass * (this.pCurrent[sd] - this.p0 - pMsst) / (this.qmass * this.nktv2p * this.mvv2e);
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
  }

  /** The velocity half step of the atoms in the group, with the dilation damping term (fix_msst.cpp, non-dftb branch). */
  private kickVelocities(vol: number, keepOld: boolean): void {
    const s = this.sys.state;
    const sd = this.direction;
    const f = s.f, v = s.v, type = s.type, mask = s.mask;
    const omegaD = this.omega[sd];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = s.massByType[type[i]];
      for (let k = 0; k < 3; k++) {
        const C = f[3 * i + k] * this.ftm2v / m;
        let D = this.mu * omegaD * omegaD / (this.velocitySum * m * vol);
        if (keepOld) this.oldV[3 * i + k] = v[3 * i + k];
        if (k === sd) D -= 2.0 * omegaD / vol;
        if (Math.abs(this.dthalf * D) > 1.0e-06) {
          const expd = Math.exp(D * this.dthalf);
          v[3 * i + k] = expd * (C + D * v[3 * i + k] - C / expd) / D;
        } else {
          v[3 * i + k] = v[3 * i + k] + (C + D * v[3 * i + k]) * this.dthalf +
            0.5 * (D * D * v[3 * i + k] + C * D) * this.dthalf * this.dthalf;
        }
      }
    }
  }

  /** 1st half of the Verlet update (fix_msst.cpp initial_integrate, non-dftb). */
  initialIntegrate(): void {
    const s = this.sys.state;
    const sd = this.direction;
    const n = s.n;
    const v = s.v, x = s.x, mask = s.mask;
    if (this.oldV.length !== 3 * n) this.oldV = new Float64Array(3 * n);
    this.fresh();
    const vol = this.computeVol();
    this.propagateOmega(vol);
    // propagate velocity sum 1/2 step by temporarily propagating the velocities
    this.velocitySum = this.computeVsum();
    this.kickVelocities(vol, true);
    this.velocitySum = this.computeVsum();
    // reset the velocities
    for (let i = 0; i < n; i++) {
      if (mask[i] & this.groupBit) {
        v[3 * i] = this.oldV[3 * i];
        v[3 * i + 1] = this.oldV[3 * i + 1];
        v[3 * i + 2] = this.oldV[3 * i + 2];
      }
    }
    // propagate velocities 1/2 step using the new velocity sum
    this.kickVelocities(vol, false);
    // propagate the volume 1/2 step, rescale positions and change box size
    const vol1 = vol + this.omega[sd] * this.dthalf;
    this.dilation[sd] = vol1 / vol;
    this.remap();
    // propagate particle positions 1 time step
    for (let i = 0; i < n; i++) {
      if (mask[i] & this.groupBit) {
        x[3 * i] += this.dtv * v[3 * i];
        x[3 * i + 1] += this.dtv * v[3 * i + 1];
        x[3 * i + 2] += this.dtv * v[3 * i + 2];
      }
    }
    // propagate the volume 1/2 step, rescale positions and change box size
    const vol2 = vol1 + this.omega[sd] * this.dthalf;
    this.dilation[sd] = vol2 / vol1;
    this.remap();
  }

  /** 2nd half of the Verlet update (fix_msst.cpp final_integrate, non-dftb). */
  finalIntegrate(): void {
    let vol = this.computeVol();
    // propagate particle velocities 1/2 step
    this.kickVelocities(vol, false);
    // compute new pressure and volume
    this.fresh();
    this.velocitySum = this.computeVsum();
    vol = this.computeVol();
    this.propagateOmega(vol);
    // calculate Lagrangian position of computational cell
    this.lagrangianPosition -= this.velocity * vol / this.v0 * this.sys.state.dt;
  }

  /**
   * Dilates the box along the shock direction about its centre and rescales all atoms and the velocity component along
   * the shock (fix_msst.cpp remap, flag 0: owned atoms, which is all atoms of a serial run).
   */
  private remap(): void {
    const sys = this.sys;
    const s = sys.state;
    const d = this.direction;
    const b = s.box;
    const oldlo = b.lo[d], oldhi = b.hi[d];
    const ctr = 0.5 * (oldlo + oldhi);
    const lo = [b.lo[0], b.lo[1], b.lo[2]];
    const hi = [b.hi[0], b.hi[1], b.hi[2]];
    lo[d] = (oldlo - ctr) * this.dilation[d] + ctr;
    hi[d] = (oldhi - ctr) * this.dilation[d] + ctr;
    const allBit = sys.groups.bit('all');
    sys.setBox(lo, hi, [b.tilt[0], b.tilt[1], b.tilt[2]], allBit, this);
    for (let i = 0; i < s.n; i++) s.v[3 * i + d] *= this.dilation[d];
  }

  /** The fix's energy term (compute_scalar), from fresh temperature, pressure and volume. */
  private computeScalarValue(): number {
    const vol = this.computeVol();
    const i = this.direction;
    let energy = (this.qmass * this.omega[i] * this.omega[i]) / (2.0 * this.totalMass) * this.mvv2e;
    energy -= 0.5 * this.totalMass * this.velocity * this.velocity * (1.0 - vol / this.v0) * (1.0 - vol / this.v0) * this.mvv2e;
    energy -= this.p0 * (this.v0 - vol) / this.nktv2p;
    return energy;
  }

  computeScalar(): number {
    this.fresh();
    return this.computeScalarValue();
  }

  /** f_ID[n] (0-based): dhugoniot, drayleigh, lagrangian_speed, lagrangian_position (fix_msst.cpp compute_vector). */
  computeVector(n: number): number {
    if (n === 0) return this.computeHugoniot();
    if (n === 1) return this.computeRayleigh();
    if (n === 2) return this.velocity * (1.0 - this.computeVol() / this.v0);
    if (n === 3) return this.lagrangianPosition;
    return 0.0;
  }

  /** The deviation of the current point from the Hugoniot, in kelvin (compute_hugoniot). */
  private computeHugoniot(): number {
    const e = this.computeEtotal();
    this.fresh();
    const p = this.pressure.vectorValues()[this.direction];
    const v = this.computeVol();
    let dhugo = (0.5 * (p + this.p0) * (this.v0 - v)) / this.nktv2p + this.e0 - e;
    dhugo /= this.temperature.dof * this.boltz;
    return dhugo;
  }

  /** The deviation of the current point from the Rayleigh line, in pressure units (compute_rayleigh). */
  private computeRayleigh(): number {
    this.fresh();
    const p = this.pressure.vectorValues()[this.direction];
    const v = this.computeVol();
    return p - this.p0 - this.totalMass * this.velocity * this.velocity * this.mvv2e * (1.0 - v / this.v0) * this.nktv2p / this.v0;
  }

  /** Energy the cell exchanges with the shock (thermo econserve, ecouple_flag = 1). */
  ecouple(): number { return this.computeScalar(); }
}
