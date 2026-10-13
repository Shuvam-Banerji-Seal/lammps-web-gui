/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/QTB/fix_qtb.cpp and fix_qtb.h
 * (class FixQTB, fix qtb: the colored thermostat for quantum nuclear effects, with its random forces filtered
 * by the spectral filter H built in init()). Contributing authors: Shen, Yuan, Qi, Tingting, and Reed, Evan.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_qtb.html): "Fix qtb applies a colored thermostat ... Its random
 * forces follow a power spectrum that depends on the quantum (Bose-Einstein) occupation of each frequency".
 * The keywords are temp, damp, seed, f_max and N_f; the defaults are temp 300, damp 1.0, seed 880302, f_max 200
 * and N_f 100. The fix only modifies forces (it is not a time integrator).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseInt_, parseNum } from '../../engine/force/util';
import { nativeOrder } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { RanMars } from '../random_mars';

/** fix_qtb.cpp: "MY_PI" from math_const.h. */
const MY_PI = 3.14159265358979323846;

/**
 * The spectrum omega_H (the square root of the quantum power spectrum at temperature tt, with the sinc correction
 * for the alpha-step hold) and the filter time_H (its cosine transform) of fix_qtb.cpp, which fix qbmsst rebuilds
 * for each temperature update. docs.lammps.org/fix_qtb.html: the random forces follow the quantum-corrected spectrum
 * at the target temperature.
 */
export const colorFilter = (
  nf: number, alpha: number, h: number, boltz: number, hplanck: number, tt: number,
  omegaH: Float64Array, timeH: Float64Array,
): void => {
  for (let k = 0; k < 2 * nf; k++) {
    const fk = (k - nf) / (2 * nf * h);
    if (k === nf) {
      omegaH[k] = Math.sqrt(boltz * tt);
    } else {
      const energyK = hplanck * Math.abs(fk);
      omegaH[k] = Math.sqrt(energyK * (0.5 + 1.0 / (Math.exp(energyK / (boltz * tt)) - 1.0)));
      omegaH[k] *= alpha * Math.sin(((k - nf) * MY_PI) / (2 * alpha * nf)) / Math.sin(((k - nf) * MY_PI) / (2 * nf));
    }
  }
  for (let n = 0; n < 2 * nf; n++) {
    timeH[n] = 0;
    const tn = n - nf;
    for (let k = 0; k < 2 * nf; k++) {
      const omegaK = ((k - nf) * MY_PI) / nf;
      timeH[n] += omegaH[k] * Math.cos(omegaK * tn);
    }
    timeH[n] /= 2.0 * nf;
  }
};

export class FixQTB extends Fix {
  readonly style = 'qtb';
  private tTarget = 300.0;
  private tPeriod = 1.0;
  private fricCoef = 1 / 1.0;
  private seed = 880302;
  private fMax = 200.0;
  private nf = 100;
  private counterMu = 0;
  private hTimestep = 0;
  private alpha = 1;
  private readonly random: RanMars;
  private gfactor1: Float64Array;
  private gfactor3: Float64Array;
  private omegaH: Float64Array;
  private timeH: Float64Array;
  /** random_array_0/1/2: per atom (engine index), 2*N_f values each. */
  private ra: [Float64Array, Float64Array, Float64Array] = [new Float64Array(0), new Float64Array(0), new Float64Array(0)];
  /** fran: per atom, 3 components. */
  private fran = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    for (let k = 0; k < args.length;) {
      const key = args[k];
      const val = args[k + 1];
      switch (key) {
        case 'temp':
          this.tTarget = parseNum(val, 'fix qtb temp');
          if (this.tTarget < 0.0) throw new StyleError('Fix qtb temp must be >= 0.0');
          break;
        case 'damp':
          this.tPeriod = parseNum(val, 'fix qtb damp');
          if (this.tPeriod <= 0.0) throw new StyleError('Fix qtb damp must be > 0.0');
          this.fricCoef = 1 / this.tPeriod;
          break;
        case 'seed':
          this.seed = parseInt_(val, 'fix qtb seed');
          if (this.seed <= 0) throw new StyleError('Illegal fix qtb command');
          break;
        case 'f_max':
          this.fMax = parseNum(val, 'fix qtb f_max');
          if (this.fMax <= 0) throw new StyleError('Illegal fix qtb command');
          break;
        case 'N_f':
          this.nf = parseInt_(val, 'fix qtb N_f');
          if (this.nf <= 0) throw new StyleError('Illegal fix qtb command');
          break;
        default:
          throw new StyleError(`Illegal fix qtb command: unknown keyword '${key}'`);
      }
      if (val === undefined) throw new StyleError('Illegal fix qtb command');
      k += 2;
    }
    this.random = new RanMars(this.seed);
    const nt = sys.state.ntypes;
    this.gfactor1 = new Float64Array(nt + 1);
    this.gfactor3 = new Float64Array(nt + 1);
    this.omegaH = new Float64Array(2 * this.nf);
    this.timeH = new Float64Array(2 * this.nf);
    this.growArrays(sys.state.n);
  }

  /** Per-atom arrays follow the atom count (the engine never permutes them: atoms keep their index). */
  private growArrays(n: number): void {
    const m = 2 * this.nf;
    if (this.fran.length === 3 * n && this.ra[0].length === m * n) return;
    this.ra = [new Float64Array(m * n), new Float64Array(m * n), new Float64Array(m * n)];
    this.fran = new Float64Array(3 * n);
  }

  modify(key: string, values: string[]): number {
    if (key === 'temp') {
      // the temperature compute is only checked: fix qtb's forces do not use it (native stores it for the run)
      const c = this.sys.compute(values[0] ?? '');
      if (!c.tempFlag) throw new StyleError(`Fix_modify temperature ID ${values[0]} does not compute temperature`);
      return 1;
    }
    return super.modify(key, values);
  }

  init(): void {
    const s = this.sys.state;
    const dtv = s.dt;
    this.growArrays(s.n);
    if (!s.massByType) throw new StyleError('Cannot use fix qtb without per-type mass defined');
    // initiate the counter mu
    this.counterMu = 0;
    // set up the h time step for updating the random force: delta h = pi / omega_max
    if (Math.trunc(1.0 / (2 * this.fMax * dtv)) === 0) {
      this.sys.warn('Either f_max is too high or the time step is too big, setting f_max to be 1/timestep!');
      this.hTimestep = dtv;
      this.alpha = 1;
    } else {
      this.alpha = Math.trunc(1.0 / (2 * this.fMax * dtv));
      this.hTimestep = this.alpha * dtv;
    }
    const u = s.units;
    // force prefactors per type: gfactor1 is the friction m gamma / ftm2v; gfactor3 the random amplitude
    for (let i = 1; i <= s.ntypes; i++) {
      this.gfactor1[i] = (s.massByType[i] * this.fricCoef) / u.ftm2v;
      this.gfactor3[i] = Math.sqrt(2 * this.fricCoef * s.massByType[i]) * Math.sqrt(u.mvv2e) * Math.sqrt(12 / this.hTimestep);
    }
    // random number arrays with zero mean and variance 1/12, drawn atom by atom in native order
    const m2 = 2 * this.nf;
    for (const i of nativeOrder(s)) {
      for (let m = 0; m < m2; m++) {
        this.ra[0][i * m2 + m] = this.random.uniform() - 0.5;
        this.ra[1][i * m2 + m] = this.random.uniform() - 0.5;
        this.ra[2][i * m2 + m] = this.random.uniform() - 0.5;
      }
    }
    this.buildFilter(u.boltz, u.hplanck, this.tTarget);
  }

  private buildFilter(boltz: number, hplanck: number, tt: number): void {
    colorFilter(this.nf, this.alpha, this.hTimestep, boltz, hplanck, tt, this.omegaH, this.timeH);
  }

  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const { f, v, type, mask } = s;
    const n = s.n;
    const m2 = 2 * this.nf;
    const order = nativeOrder(s);
    const [ra0, ra1, ra2] = this.ra;
    const fran = this.fran;
    // update the colored random force every alpha MD steps
    if (this.counterMu === this.alpha) {
      for (const j of order) {
        // shift the random arrays one slot and draw the new last element (three draws per atom)
        for (let m = 0; m < m2 - 1; m++) {
          ra0[j * m2 + m] = ra0[j * m2 + m + 1];
          ra1[j * m2 + m] = ra1[j * m2 + m + 1];
          ra2[j * m2 + m] = ra2[j * m2 + m + 1];
        }
        ra0[j * m2 + m2 - 1] = this.random.uniform() - 0.5;
        ra1[j * m2 + m2 - 1] = this.random.uniform() - 0.5;
        ra2[j * m2 + m2 - 1] = this.random.uniform() - 0.5;
      }
      this.counterMu = 0;
    }
    if (this.counterMu === 0) {
      for (let j = 0; j < n; j++) {
        fran[3 * j] = 0.0;
        fran[3 * j + 1] = 0.0;
        fran[3 * j + 2] = 0.0;
        if (mask[j] & this.groupBit) {
          const gamma3 = this.gfactor3[type[j]];
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
    // the force sums over the group, in native order
    const fsum = [0, 0, 0];
    for (const j of order) {
      if (mask[j] & this.groupBit) {
        const gamma1 = this.gfactor1[type[j]];
        fsum[0] += fran[3 * j] - gamma1 * v[3 * j];
        fsum[1] += fran[3 * j + 1] - gamma1 * v[3 * j + 1];
        fsum[2] += fran[3 * j + 2] - gamma1 * v[3 * j + 2];
      }
    }
    // no net force on the system: subtract the mean of the sums over all atoms (fsumall / ntotal)
    for (const j of order) {
      f[3 * j] -= fsum[0] / n;
      f[3 * j + 1] -= fsum[1] / n;
      f[3 * j + 2] -= fsum[2] / n;
      if (mask[j] & this.groupBit) {
        const gamma1 = this.gfactor1[type[j]];
        f[3 * j] += fran[3 * j] - gamma1 * v[3 * j];
        f[3 * j + 1] += fran[3 * j + 1] - gamma1 * v[3 * j + 1];
        f[3 * j + 2] += fran[3 * j + 2] - gamma1 * v[3 * j + 2];
      }
    }
    this.counterMu++;
  }
}
