/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/RHEO/fix_rheo_viscosity.cpp (FixRHEOViscosity: fix rheo/viscosity) and src/RHEO/fix_rheo_pressure.cpp
 * (FixRHEOPressure: fix rheo/pressure), at commit b332f28. RHEO package authors: Joel Clemmer (SNL), Thomas O'Connor
 * (CMU), Eric Palermo (CMU). Copyright Sandia Corporation (GNU General Public License); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_rheo_viscosity.html): "fix ID group-ID rheo/viscosity type1 pstyle1 args1 ... typeN pstyleN
 * argsN" with "constant args = eta" (the viscosity of each type) and "power args = eta, gd0, K, n" (the shear-rate
 * dependent viscosity from the velocity gradient). Docs (docs.lammps.org/fix_rheo_pressure.html): "fix ID group-ID
 * rheo/pressure type1 pstyle1 args1 ... typeN pstyleN argsN" with the styles linear, cubic, tait/water,
 * tait/general, ideal/gas and the background pressure, in the forms of fix_rheo_pressure.cpp calc_pressure and
 * calc_csq.
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { StyleError, typeBounds } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { isRheoThermalStyle, RHEO_ESPH, RHEO_PRESSURE, RHEO_RHO, RHEO_VISC } from '../../engine/atoms';
import { pow as libmPow } from '../libm/pow';
import { rheoModelOf, rheoProp, type RheoModel } from './model';

/** fix_rheo_pressure.cpp SEVENTH */
const SEVENTH = 1.0 / 7.0;

/** The shared state of fix rheo, which must be defined before an accessory fix (fix rheo comes first in the input). */
const needModel = (sys: System, style: string): RheoModel => {
  const m = rheoModelOf(sys.state);
  if (!m) throw new StyleError(`Need to define fix rheo to use fix ${style}`);
  return m;
};

/** fix rheo/viscosity: the viscosity of each type (constant), set on every atom. */
export class FixRHEOViscosity extends Fix {
  readonly style: string = 'rheo/viscosity';
  private readonly eta: Float64Array;
  /** per type: 1 constant, 2 power (fix_rheo_viscosity.cpp viscosity_style) */
  private readonly vstyle: Uint8Array;
  private readonly gd0: Float64Array;
  private readonly K: Float64Array;
  private readonly npow: Float64Array;
  private readonly tau0: Float64Array;
  private evolve = false;
  private readonly model: RheoModel;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError('fix rheo/viscosity command requires group all');
    if (sys.fixes.some((f) => f.style === 'rheo/viscosity')) throw new StyleError('Can only specify one instance of fix rheo/viscosity');
    const n = sys.state.ntypes;
    this.eta = new Float64Array(n + 1);
    this.vstyle = new Uint8Array(n + 1);
    this.gd0 = new Float64Array(n + 1);
    this.K = new Float64Array(n + 1);
    this.npow = new Float64Array(n + 1);
    this.tau0 = new Float64Array(n + 1);
    let iarg = 0;
    while (iarg < args.length) {
      const [lo, hi] = typeBounds(args[iarg], n);
      if (iarg + 1 >= args.length) throw new StyleError('Illegal fix rheo/viscosity command: missing style');
      if (args[iarg + 1] === 'constant') {
        if (iarg + 2 >= args.length) throw new StyleError('Illegal fix rheo/viscosity constant: missing viscosity');
        const eta = parseNum(args[iarg + 2], 'viscosity');
        if (eta < 0.0) throw new StyleError('The viscosity must be positive');
        for (let t = lo; t <= hi; t++) { this.eta[t] = eta; this.vstyle[t] = 1; }
        iarg += 3;
      } else if (args[iarg + 1] === 'power') {
        if (iarg + 5 >= args.length) throw new StyleError('Illegal fix rheo/viscosity power: missing values');
        const eta = parseNum(args[iarg + 2], 'viscosity');
        const gd0 = parseNum(args[iarg + 3], 'gd0'), K = parseNum(args[iarg + 4], 'K'), np = parseNum(args[iarg + 5], 'n');
        if (eta < 0.0) throw new StyleError('The viscosity must be positive');
        this.evolve = true;
        for (let t = lo; t <= hi; t++) {
          this.vstyle[t] = 2;
          this.eta[t] = eta; this.gd0[t] = gd0; this.K[t] = K; this.npow[t] = np;
          // tau0 = eta gd0 - K gd0^n: the viscosity is continuous at the critical shear rate
          this.tau0[t] = eta * gd0 - K * libmPow(gd0, np);
        }
        iarg += 6;
      } else {
        throw new StyleError(`Illegal fix command, ${args[iarg]}`);
      }
    }
    for (let t = 1; t <= n; t++) {
      if (!this.vstyle[t]) throw new StyleError(`Must specify viscosity for atom type ${t} in fix/rheo/viscosity`);
    }
    this.model = needModel(sys, this.style);
  }

  setupPreForce(): void { this.preForce(); }

  /**
   * Viscosity of each atom from its type; the power types from the shear rate gdot = sqrt(sum_{a<=b} ((du_a/dx_b +
   * du_b/dx_a)^2, halved on the diagonal)) of the velocity gradient of this step (fix rheo's pre_force): eta below gd0,
   * else K gdot^(n-1) + tau0 / gdot (fix_rheo_viscosity.cpp pre_force).
   */
  preForce(): void {
    const s = this.sys.state;
    const visc = rheoProp(s, RHEO_VISC);
    for (let i = 0; i < s.n; i++) visc[i] = this.eta[s.type[i]];
    if (!this.evolve) return;
    const dim = this.model.dim, dd = dim * dim;
    const gradv = this.model.gradv;
    for (let i = 0; i < s.n; i++) {
      const t = s.type[i];
      if (this.vstyle[t] !== 2) continue;
      let gdot = 0.0;
      for (let a = 0; a < dim; a++) {
        for (let b = a; b < dim; b++) {
          let tmp = gradv[i * dd + a * dim + b] + gradv[i * dd + b * dim + a];
          tmp = tmp * tmp;
          if (a === b) tmp *= 0.5;
          gdot += tmp;
        }
      }
      gdot = Math.sqrt(gdot);
      if (gdot <= this.gd0[t]) visc[i] = this.eta[t];
      else visc[i] = this.K[t] * libmPow(gdot, this.npow[t] - 1) + this.tau0[t] / gdot;
    }
  }
}

/** fix rheo/pressure: the equation of state of each type (fix_rheo_pressure.cpp). */
const LINEAR = 1, CUBIC = 2, TAITWATER = 3, TAITGENERAL = 4, IDEAL = 5;

export class FixRHEOPressure extends Fix {
  readonly style: string = 'rheo/pressure';
  private readonly model: RheoModel;
  private readonly pstyle: Uint8Array;
  private readonly cCubic: Float64Array;
  private readonly tpower: Float64Array;
  private readonly gamma: Float64Array;
  private readonly pbackground: Float64Array;
  private backgroundFlag = false;
  private variableCsq = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError('fix rheo/pressure command requires group all');
    if (sys.fixes.some((f) => f.style === 'rheo/pressure')) throw new StyleError('Can only specify one instance of fix rheo/pressure');
    const n = sys.state.ntypes;
    this.pstyle = new Uint8Array(n + 1);
    this.cCubic = new Float64Array(n + 1);
    this.tpower = new Float64Array(n + 1);
    this.gamma = new Float64Array(n + 1);
    this.pbackground = new Float64Array(n + 1);
    let iarg = 0;
    const value = (w: string | undefined, what: string): number => {
      if (w === undefined) throw new StyleError(`Illegal fix rheo/pressure ${what}: missing value`);
      return parseNum(w, what);
    };
    while (iarg < args.length) {
      const [lo, hi] = typeBounds(args[iarg], n);
      if (iarg + 1 >= args.length) throw new StyleError('Illegal fix rheo/pressure command: missing style');
      const st = args[iarg + 1];
      if (st === 'linear') {
        for (let t = lo; t <= hi; t++) this.pstyle[t] = LINEAR;
      } else if (st === 'tait/water') {
        for (let t = lo; t <= hi; t++) this.pstyle[t] = TAITWATER;
      } else if (st === 'tait/general' || st === 'cubic' || st === 'ideal/gas' || st === 'background') {
        const v = value(args[iarg + 2], st);
        iarg += 1;
        for (let t = lo; t <= hi; t++) {
          if (st === 'tait/general') { this.pstyle[t] = TAITGENERAL; this.tpower[t] = v; }
          else if (st === 'cubic') { this.pstyle[t] = CUBIC; this.cCubic[t] = v; }
          else if (st === 'ideal/gas') { this.pstyle[t] = IDEAL; this.gamma[t] = v; }
          else this.pbackground[t] = v;
        }
        if (st === 'background') this.backgroundFlag = true;
        if (st === 'ideal/gas') {
          this.variableCsq = true;
          if (!isRheoThermalStyle(sys.state.atomStyle)) throw new StyleError('fix rheo/pressure ideal gas equation of state requires atom property esph');
        }
      } else {
        throw new StyleError(`Illegal fix command, ${st}`);
      }
      iarg += 2;
    }
    for (let t = 1; t <= n; t++) {
      if (!this.pstyle[t]) throw new StyleError(`Must specify pressure for atom type ${t} in fix/rheo/pressure`);
    }
    this.model = needModel(sys, this.style);
  }

  init(): void {
    // pair rheo reads calc_csq of each end for a state-dependent sound speed (pair_rheo.cpp variable_csq)
    this.model.variableCsq = this.variableCsq ? (i: number) => this.calcCsq(i) : null;
  }

  destroy(): void {
    if (this.model.variableCsq) this.model.variableCsq = null;
  }

  setupPreForce(): void { this.preForce(); }

  /** fix_rheo_pressure.cpp calc_pressure: the pressure of atom i at density rho. */
  calcPressure(rho: number, i: number): number {
    const s = this.sys.state;
    const m = this.model;
    const t = s.type[i];
    let p = 0.0;
    if (this.pstyle[t] === LINEAR) {
      p = m.csq[t] * (rho - m.rho0[t]);
    } else if (this.pstyle[t] === CUBIC) {
      const dr = rho - m.rho0[t];
      p = m.csq[t] * (dr + this.cCubic[t] * dr * dr * dr);
    } else if (this.pstyle[t] === TAITWATER) {
      const rhoRatio = rho * (1.0 / m.rho0[t]);
      const rr3 = rhoRatio * rhoRatio * rhoRatio;
      p = m.csq[t] * m.rho0[t] * SEVENTH * (rr3 * rr3 * rhoRatio - 1.0);
    } else if (this.pstyle[t] === TAITGENERAL) {
      const rhoRatio = rho * (1.0 / m.rho0[t]);
      p = m.csq[t] * m.rho0[t] * (libmPow(rhoRatio, this.tpower[t]) - 1.0) / this.tpower[t];
    } else if (this.pstyle[t] === IDEAL) {
      const imass = s.rmass ? s.rmass[i] : s.massByType[t];
      p = (this.gamma[t] - 1.0) * rho * rheoProp(s, RHEO_ESPH)[i] / imass;
    }
    if (this.backgroundFlag) p += this.pbackground[t];
    return p;
  }

  /** fix_rheo_pressure.cpp calc_csq: the per-type csq, or (gamma - 1) esph / mass[type] for ideal/gas. */
  calcCsq(i: number): number {
    const s = this.sys.state;
    const t = s.type[i];
    if (this.pstyle[t] === IDEAL) return (this.gamma[t] - 1.0) * rheoProp(s, RHEO_ESPH)[i] / s.massByType[t];
    return this.model.csq[t];
  }

  /** Pressure of each group atom from its density (pre_force). */
  preForce(): void {
    const s = this.sys.state;
    const rho = rheoProp(s, RHEO_RHO), pressure = rheoProp(s, RHEO_PRESSURE);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      pressure[i] = this.calcPressure(rho[i], i);
    }
  }
}
