/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_baoab.cpp and fix_baoab.h (FixBAOAB: constructor, init, compute_target, initial_integrate,
 * final_integrate, reset_dt, compute_scalar) at commit b332f28. Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * BAOAB splitting of Langevin dynamics (Leimkuhler and Matthews): B (half kick), A (half drift), O (exact
 * Ornstein-Uhlenbeck step), A (half drift), then the B kick after the forces (docs.lammps.org/fix_baoab.html). The
 * random draws are RanMars gaussians in native atom order, one (x, y, z) triple per atom, so a seed gives the same
 * trajectory as native LAMMPS.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { missingArgs } from '../../engine/force/util';
import { massOf, nativeOrder } from '../../engine/atoms';
import { ramp } from '../../engine/fix/util';
import { takeFixRecord } from '../../engine/fix/restartRecord';
import type { System } from '../../engine/system';
import { RanMars } from '../random_mars';
import { exp as libmExp } from '../libm/exp';

export class FixBAOAB extends Fix {
  readonly style = 'baoab';
  private readonly tStart: number;
  private readonly tStop: number;
  private readonly gamma: number;
  private readonly seed: number;
  private readonly zeroFlag: boolean;
  private readonly rng: RanMars;
  /** Cumulative kinetic energy exchanged with the reservoir (compute_scalar, thermo ecouple). */
  private energyAcc = 0;
  timeIntegrate = true;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('Illegal fix baoab command: usage fix ID group baoab Tstart Tstop damp seed [zero yes/no]');
    this.tStart = Number(args[0]);
    this.tStop = Number(args[1]);
    const damp = Number(args[2]);
    this.seed = Number(args[3]);
    if (this.tStart < 0.0 || this.tStop < 0.0) throw new StyleError('Fix baoab temperatures must be >= 0');
    if (!(damp > 0.0)) throw new StyleError('Fix baoab damp must be > 0');
    if (!Number.isInteger(this.seed) || this.seed <= 0) throw new StyleError('Fix baoab seed must be positive integer');
    // gamma = 1/damp (damp is the relaxation time, like fix langevin)
    this.gamma = 1.0 / damp;
    let zero = false;
    for (let k = 4; k < args.length;) {
      if (args[k] !== 'zero') throw new StyleError(`Unknown fix baoab keyword ${args[k]}`);
      if (k + 1 >= args.length) throw missingArgs('fix baoab zero');
      if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError(`Illegal fix baoab zero value '${args[k + 1]}'`);
      zero = args[k + 1] === 'yes';
      k += 2;
    }
    this.zeroFlag = zero;
    this.energyGlobal = false;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.rng = new RanMars(this.seed);
    // fix_baoab.cpp write_restart stores only the energy (rank 0's global value); the generator is seeded afresh
    const saved = takeFixRecord(sys, id, this.style)?.state as { energy: number } | undefined;
    if (saved) this.energyAcc = saved.energy;
  }

  /** fix_baoab.cpp write_restart: the cumulative energy exchanged with the reservoir (the stream is not stored). */
  restartState(): { energy: number } { return { energy: this.energyAcc }; }

  init(): void {
    // "fix baoab and fix shake/rattle/ilves cannot be used together" on a shared group
    for (const f of this.sys.fixes) {
      if (/^(shake|rattle|ilves)/.test(f.style) && (f.groupBit & this.groupBit)) {
        throw new StyleError(`Fix baoab is not compatible with fix ${f.style} on group ${this.group}`);
      }
    }
  }

  /** Target temperature at this step (the Tstart..Tstop ramp over the run). */
  private targetT(): number {
    return ramp(this.sys, this.tStart, this.tStop);
  }

  /**
   * B A O A: half kick and half drift from the previous forces, the exact Ornstein-Uhlenbeck velocity update
   * v = c1 v + c2 R with c1 = exp(-gamma dt) and c2 = sqrt(kT/m (1 - c1^2)), then the second half drift.
   * Coefficients use the current dt (fix_baoab.cpp recompute_dt_coeffs at init and reset_dt).
   */
  initialIntegrate(): void {
    const s = this.sys.state;
    const u = s.units;
    const dt = s.dt;
    const dtf = 0.5 * dt * u.ftm2v;
    const dtby2 = 0.5 * dt;
    const c1 = libmExp(-this.gamma * dt);
    const oneMinusC1sq = -Math.expm1(-2.0 * this.gamma * dt);
    const tTarget = this.targetT();
    // kT in velocity-squared * mass units (kB / mvv2e)
    const kT = u.boltz * tTarget / u.mvv2e;
    const { x, v, f } = s;
    let energyOneStep = 0;
    const fran = [0, 0, 0];
    let massTotal = 0;
    const order = nativeOrder(s);
    for (const i of order) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const mi = massOf(s, i);
      const invmass = 1.0 / mi;
      // B: half velocity kick from the conservative forces, dv = dtf F / m
      const dtfm = dtf * invmass;
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      // A: half position drift
      x[3 * i] += dtby2 * v[3 * i];
      x[3 * i + 1] += dtby2 * v[3 * i + 1];
      x[3 * i + 2] += dtby2 * v[3 * i + 2];
      // O: exact thermostat over the full dt
      const c2 = Math.sqrt(kT * invmass * oneMinusC1sq);
      const keBefore = 0.5 * u.mvv2e * mi * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
      const r0 = this.rng.gaussian();
      const r1 = this.rng.gaussian();
      const r2 = this.rng.gaussian();
      v[3 * i] = c1 * v[3 * i] + c2 * r0;
      v[3 * i + 1] = c1 * v[3 * i + 1] + c2 * r1;
      v[3 * i + 2] = c1 * v[3 * i + 2] + c2 * r2;
      if (this.zeroFlag) {
        fran[0] += mi * c2 * r0;
        fran[1] += mi * c2 * r1;
        fran[2] += mi * c2 * r2;
        massTotal += mi;
      }
      const keAfter = 0.5 * u.mvv2e * mi * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
      energyOneStep += keBefore - keAfter;
      // A: second half position drift
      x[3 * i] += dtby2 * v[3 * i];
      x[3 * i + 1] += dtby2 * v[3 * i + 1];
      x[3 * i + 2] += dtby2 * v[3 * i + 2];
    }
    // zero net random momentum: remove the mean random velocity and the matching position drift
    if (this.zeroFlag && massTotal > 0.0) {
      const invM = 1.0 / massTotal;
      const vcorr = [fran[0] * invM, fran[1] * invM, fran[2] * invM];
      for (const i of order) {
        if (!(s.mask[i] & this.groupBit)) continue;
        v[3 * i] -= vcorr[0];
        v[3 * i + 1] -= vcorr[1];
        v[3 * i + 2] -= vcorr[2];
        // the second A step already used the uncorrected velocities
        x[3 * i] -= dtby2 * vcorr[0];
        x[3 * i + 1] -= dtby2 * vcorr[1];
        x[3 * i + 2] -= dtby2 * vcorr[2];
      }
    }
    this.energyAcc += energyOneStep;
  }

  /** B: the second half kick from the new forces, dv = (dtf / m) F. */
  finalIntegrate(): void {
    const s = this.sys.state;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const { v, f } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const dtfm = dtf / massOf(s, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
    }
  }

  ecouple(): number { return this.energyAcc; }
  computeScalar(): number { return this.energyAcc; }
}
