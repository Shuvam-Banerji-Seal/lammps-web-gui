/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EFF/fix_langevin_eff.cpp and
 * fix_langevin_eff.h (class FixLangevinEff, fix langevin/eff) with the base fix langevin of src/fix_langevin.cpp it
 * inherits (constructor, init prefactors gfactor1 and gfactor2, compute_target, the keywords scale and tally).
 * The electron radial friction and random force follow the nuclear ones, with mefactor = dimension / 4. Contributing
 * author (from the source): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Refused with a StyleError naming the option: the omega and angmom keywords (the base class's torque thermostats),
 * the zero keyword (native LAMMPS reads an uninitialized value, see the constructor) and the gjf keyword (removed from LAMMPS).
 *
 * The random stream is RanMars(seed + comm->me) with comm->me = 0. Draws follow native local atom order (nativeOrder):
 * three per atom, then one more for an electron (fix_langevin_eff.cpp post_force_no_tally).
 */
import { Fix } from '../../engine/fix/fix';
import type { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { inumeric, logical, missingArgs, numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { nativeOrder, ELECTRON_RFORCE, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';
import { parseNumOrVar, valueOf, type NumOrVar } from '../../engine/fix/util';
import { RanMars } from '../../engine/rng';

export class FixLangevinEff extends Fix {
  readonly style = 'langevin/eff';
  private readonly tStart: NumOrVar;
  private readonly tStop: number;
  private readonly tPeriod: number;
  private readonly rng: RanMars;
  private readonly ratio: Float64Array;
  private readonly gfactor1: Float64Array;
  private readonly gfactor2: Float64Array;
  private readonly tally: boolean;
  /** fix_modify temp: the temperature compute whose bias is removed (tbiasflag BIAS when it has one) */
  private tempCompute: Compute | null = null;
  private tsqrt = 0;
  /** tally: the friction and random force per atom (flangevin) and the electron radial force (erforcelangevin) */
  private fl = new Float64Array(0);
  private erfl = new Float64Array(0);
  private energyTally = 0;
  private energyOnestep = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const s = sys.state;
    // the base constructor FixLangevin::FixLangevin (src/fix_langevin.cpp at b332f28) reads the arguments: Tstart (a
    // number or v_name), Tstop and Tperiod (utils::numeric), seed (utils::inumeric), then the keywords
    if (args.length < 4) throw missingArgs('fix langevin');
    this.tStart = parseNumOrVar(args[0], 'Tstart');
    this.tStop = numeric(args[1]);
    this.tPeriod = numeric(args[2]);
    const seed = inumeric(args[3]);
    if (this.tPeriod <= 0.0) throw new StyleError('Fix langevin period must be > 0.0');
    if (seed <= 0) throw new StyleError('Fix langevin seed value must be > 0');
    this.rng = new RanMars(seed);
    // Measured with native LAMMPS (black box, as in fix langevin): the stream starts one draw in, so the first draw is taken here.
    this.rng.uniform();
    this.ratio = new Float64Array(s.ntypes + 1).fill(1);
    this.gfactor1 = new Float64Array(s.ntypes + 1);
    this.gfactor2 = new Float64Array(s.ntypes + 1);
    let tally = false;
    for (let k = 4; k < args.length;) {
      const key = args[k];
      switch (key) {
        case 'scale': {
          // fix_langevin.cpp: if (iarg + 3 > narg) utils::missing_cmd_args(FLERR, "fix langevin scale", error);
          if (k + 3 > args.length) throw new StyleError('Illegal fix langevin scale command: missing argument(s)');
          const t = inumeric(args[k + 1]), r = numeric(args[k + 2]);
          if (t < 1 || t > s.ntypes) throw new StyleError(`Invalid fix langevin scale type. Must be 1 < type <= ${s.ntypes}`);
          this.ratio[t] = r;
          k += 3;
          break;
        }
        case 'tally': {
          // fix_langevin.cpp: if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix langevin tally", error);
          if (k + 2 > args.length) throw new StyleError('Illegal fix langevin tally command: missing argument(s)');
          tally = logical(args[k + 1]);
          k += 2;
          break;
        }
        case 'zero':
          // fix_langevin_eff.cpp: MPI_Allreduce(fsum, fsumall, 3, ...) reduces three of the four sums, so fsumall[3] (the
          // electron radial mean) is read uninitialized; native runs differ from one another, so the keyword is refused.
          throw new StyleError('fix langevin/eff zero yes is not supported by the browser engine: native LAMMPS reads an uninitialized electron radial sum (fix_langevin_eff.cpp MPI_Allreduce(fsum,fsumall,3,...))');
        case 'angmom': case 'omega':
          throw new StyleError(`fix langevin/eff ${key} is not supported by the browser engine yet`);
        case 'gjf':
          throw new StyleError('The GJF integrator formulation in fix langevin/eff has been removed. Please use fix gjf instead: https://docs.lammps.org/fix_gjf.html');
        default:
          throw new StyleError(`Unknown fix langevin keyword ${key}`);
      }
    }
    // native has no atom style check (it reads the electron attributes); the engine requires them here
    if (!isElectronStyle(s.atomStyle)) throw new StyleError('Fix langevin/eff requires atom style electron');
    this.tally = tally;
    this.scalarFlag = tally;
    this.extscalar = 1;
    this.ecouple = () => this.energyTally;
  }

  /** fix_modify temp: the compute whose velocity bias is removed (fix_langevin.cpp modify_param). */
  override modify(key: string, values: string[]): number {
    if (key === 'temp') {
      const id = values[0] ?? '';
      const c = this.sys.computes.find((x) => x.id === id);
      if (!c) throw new StyleError(`Temperature compute ID ${id} for fix langevin/eff does not exist`);
      if (!c.tempFlag) throw new StyleError(`Compute ID ${id} for fix langevin/eff does not compute temperature`);
      this.tempCompute = c;
      return 1;
    }
    return super.modify(key, values);
  }

  /** fix_langevin.cpp init: gfactor1 = -m/(t_period ftm2v), gfactor2 = sqrt(m)/ftm2v sqrt(24 kB/(t_period dt mvv2e)), scaled by the ratio. */
  override init(): void {
    const s = this.sys.state;
    const u = s.units;
    for (let t = 1; t <= s.ntypes; t++) {
      const m = s.massByType[t];
      this.gfactor1[t] = -m / this.tPeriod / u.ftm2v;
      this.gfactor2[t] = Math.sqrt(m) / u.ftm2v;
      this.gfactor2[t] *= Math.sqrt(24.0 * u.boltz / this.tPeriod / s.dt / u.mvv2e);
      this.gfactor1[t] *= 1.0 / this.ratio[t];
      this.gfactor2[t] *= 1.0 / Math.sqrt(this.ratio[t]);
    }
  }

  /** fix_langevin_eff.cpp compute_target: the constant or equal-style target and its square root. */
  private computeTarget(): void {
    // "delta /= update->endstep - update->beginstep" without the guard of fix langevin: a run 0 (endstep = beginstep)
    // divides 0 by 0, so the target, its square root and the random forces are NaN
    const r = this.sys.run;
    const delta = (this.sys.state.step - r.beginStep) / (r.endStep - r.beginStep);
    const tt = typeof this.tStart === 'number' ? this.tStart + delta * (this.tStop - this.tStart) : valueOf(this.sys, this.tStart);
    // only a variable's negative value is an error; a negative constant target gives NaN random forces (sqrt), which
    // native runs with and the engine refuses once steps are run (a run 0 ends normally, as natively)
    if (tt < 0.0 && typeof this.tStart !== 'number') throw new StyleError('Fix langevin/eff variable returned negative temperature');
    if (tt < 0.0 && this.sys.run.lastStep > this.sys.run.firstStep) throw new StyleError(`fix ${this.id} langevin/eff: the negative target temperature ${tt} gives NaN forces in native LAMMPS; the engine does not run it`);
    this.tsqrt = Math.sqrt(tt);
  }

  /** gfactor3 = (dof + electrons) / (dof - dimension electrons), with dof = dim N - (dim + fix dof). */
  private gfactor3(nelectrons: number, particles: number): number {
    const dim = this.sys.dimension;
    let dof = dim * particles;
    dof -= dim + this.sys.dofRemoved(this.groupBit);
    const dofelectrons = dim * nelectrons;
    const dofnuclei = dof - dofelectrons;
    return (dof + nelectrons) / dofnuclei;
  }

  /** fix_langevin_eff.cpp post_force: friction and random forces on the nuclei and the electron radial forces. */
  postForce(): void {
    this.computeTarget();
    const s = this.sys.state;
    const { f, v, type, mask } = s;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const erforce = s.custom.get(ELECTRON_RFORCE)!.data;
    const mefactor = this.sys.dimension / 4.0;
    const sqrtmefactor = Math.sqrt(mefactor);
    let particles = 0, nelectrons = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      particles++;
      if (Math.abs(spin[i]) === 1) nelectrons++;
    }
    const gfactor3 = this.gfactor3(nelectrons, particles);
    const tc = this.tempCompute;
    const bias = tc !== null && tc.hasBias();
    // fix_langevin.cpp: the bias compute evaluates its temperature before the atoms are visited
    if (bias) { tc!.invalidate(); tc!.scalarValue(); }
    const tsqrt = this.tsqrt;
    if (this.tally) {
      if (this.fl.length !== 3 * s.n) { this.fl = new Float64Array(3 * s.n); this.erfl = new Float64Array(s.n); }
      for (const i of nativeOrder(s)) {
        if (!(mask[i] & this.groupBit)) continue;
        const gamma1 = this.gfactor1[type[i]] * gfactor3;
        const gamma2 = this.gfactor2[type[i]] * tsqrt;
        if (bias) tc!.removeBias(i);
        // post_force_tally: flangevin = gamma1 v + gamma2 (u - 0.5); with a bias, a zero velocity component gets no force
        const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
        const fx = gamma1 * vx + gamma2 * (this.rng.uniform() - 0.5);
        const fy = gamma1 * vy + gamma2 * (this.rng.uniform() - 0.5);
        const fz = gamma1 * vz + gamma2 * (this.rng.uniform() - 0.5);
        this.fl[3 * i] = fx; this.fl[3 * i + 1] = fy; this.fl[3 * i + 2] = fz;
        if (!bias || vx !== 0.0) f[3 * i] += fx; else this.fl[3 * i] = 0.0;
        if (!bias || vy !== 0.0) f[3 * i + 1] += fy; else this.fl[3 * i + 1] = 0.0;
        if (!bias || vz !== 0.0) f[3 * i + 2] += fz; else this.fl[3 * i + 2] = 0.0;
        if (Math.abs(spin[i]) === 1) {
          const e = mefactor * gamma1 * ervel[i] + sqrtmefactor * gamma2 * (this.rng.uniform() - 0.5);
          if (!bias || ervel[i] !== 0.0) { this.erfl[i] = e; erforce[i] += e; } else this.erfl[i] = 0.0;
        }
        if (bias) tc!.restoreBias(i);
      }
      return;
    }
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      const gamma1 = this.gfactor1[type[i]] * gfactor3;
      const gamma2 = this.gfactor2[type[i]] * tsqrt;
      if (bias) tc!.removeBias(i);
      const fr0 = gamma2 * (this.rng.uniform() - 0.5);
      const fr1 = gamma2 * (this.rng.uniform() - 0.5);
      const fr2 = gamma2 * (this.rng.uniform() - 0.5);
      if (!bias || v[3 * i] !== 0.0) f[3 * i] += gamma1 * v[3 * i] + fr0;
      if (!bias || v[3 * i + 1] !== 0.0) f[3 * i + 1] += gamma1 * v[3 * i + 1] + fr1;
      if (!bias || v[3 * i + 2] !== 0.0) f[3 * i + 2] += gamma1 * v[3 * i + 2] + fr2;
      if (Math.abs(spin[i]) === 1) {
        const fr3 = sqrtmefactor * gamma2 * (this.rng.uniform() - 0.5);
        if (!bias || ervel[i] !== 0.0) erforce[i] += mefactor * gamma1 * ervel[i] + fr3;
      }
      if (bias) tc!.restoreBias(i);
    }
  }

  /** fix_langevin_eff.cpp end_of_step: the energy transferred to the reservoir (tally yes). */
  private stepEnergy(): number {
    const s = this.sys.state;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    let e = 0.0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      e += this.fl[3 * i] * s.v[3 * i] + this.fl[3 * i + 1] * s.v[3 * i + 1] + this.fl[3 * i + 2] * s.v[3 * i + 2];
      if (Math.abs(spin[i]) === 1) e += this.erfl[i];
    }
    return e;
  }

  endOfStep(): void {
    if (!this.tally) return;
    this.energyOnestep = this.stepEnergy();
    this.energyTally += this.energyOnestep * this.sys.state.dt;
  }

  /** fix_langevin_eff.cpp compute_scalar: minus the energy the atoms gained (the first step uses the half step energy). */
  computeScalar(): number {
    if (!this.tally || this.fl.length === 0) return 0.0;
    const s = this.sys.state;
    if (s.step === this.sys.run.beginStep) {
      this.energyOnestep = this.stepEnergy();
      this.energyTally = 0.5 * this.energyOnestep * s.dt;
    }
    const energyMe = this.energyTally - 0.5 * this.energyOnestep * s.dt;
    return -energyMe;
  }
}
