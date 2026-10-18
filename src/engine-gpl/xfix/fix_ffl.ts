/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_ffl.cpp and src/EXTRA-FIX/fix_ffl.h at commit b332f28 (class FixFFL, fix ffl: the
 * constructor, init, init_ffl, ffl_integrate, initial_integrate, final_integrate, compute_scalar, reset_target and
 * reset_dt). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Fast Fourier-type Langevin (FFL) thermostat of the fix. docs.lammps.org/fix_ffl.html: "The fix ffl applies a
 * thermostat ... The keyword flip ... rescale, hard, soft, no_flip". The random stream is RanMars(seed) with
 * Gaussian draws in the order of the group atoms (native storage order), one set of 3N draws per call.
 * The scalar (f_ID) is the cumulative energy exchanged with the reservoir, and the thermo ecouple keyword sums it.
 * Respa is not supported (refused by the core integrator, as fixes that are not respa-ready).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { exp as libmExp } from '../libm/exp';

type Flip = 'no_flip' | 'rescale' | 'hard' | 'soft';

function numeric(w: string | undefined): number {
  const x = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(x)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return x;
}

function inumeric(w: string | undefined): number {
  const x = Number(w);
  if (w === undefined || !/^[+-]?\d+$/.test(w.trim())) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return x;
}

export class FixFFL extends Fix {
  readonly style = 'ffl';
  scalarFlag = true;
  extscalar = 1;
  private readonly gamma: number;
  private readonly tStart: number;
  private readonly tStop: number;
  private tTarget: number;
  private readonly flip: Flip;
  private readonly rng: RanMars;
  private readonly ffl_every = 1;
  private fflStep = 0;
  private energyCum = 0;
  private c1 = 0;
  private c2 = 0;
  private dtv = 0;
  private dtf = 0;
  private doffl = 1;
  private sqrtM = new Float64Array(0);
  private tmp1 = new Float64Array(0);
  private tmp2 = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // native narg counts "fix ID group ffl": narg < 7 is fewer than four arguments
    if (args.length < 4) {
      throw new StyleError('Illegal fix ffl command. Expecting: fix <fix-ID> <group-ID> ffl <tau> <Tstart> <Tstop> <seed>  ');
    }
    this.timeIntegrate = true;
    let gamma = numeric(args[0]);
    if (gamma <= 0.0) throw new StyleError('Illegal fix ffl tau value, should be greater than 0');
    this.gamma = 1.0 / gamma;
    this.tStart = numeric(args[1]);
    this.tStop = numeric(args[2]);
    const seed = inumeric(args[3]);
    // flip type: rescale when none is given; native reads the flip word only for narg == 8
    if (args.length === 5) {
      const w = args[4];
      if (w === 'no_flip' || w === 'rescale' || w === 'hard' || w === 'soft') this.flip = w;
      else throw new StyleError('Illegal fix ffl flip type, only accepts : rescale - hard - soft - no_flip');
    } else {
      this.flip = 'rescale';
    }
    this.tTarget = this.tStart;
    if (seed <= 0) throw new StyleError('Illegal fix ffl command');
    this.rng = new RanMars(seed);
  }

  /** compute_scalar: cumulative energy exchanged with the reservoir (also the thermo ecouple term). */
  computeScalar(): number { return this.energyCum; }

  ecouple(): number { return this.energyCum; }

  init(): void {
    this.doffl = 1;
    this.resetDt();
    const s = this.sys.state;
    // set force prefactors (per-type masses; per-atom masses are read in the loops)
    if (!s.rmass) {
      this.sqrtM = new Float64Array(s.ntypes + 1);
      for (let i = 1; i <= s.ntypes; i++) this.sqrtM[i] = Math.sqrt(s.massByType[i]);
    }
    this.initFFL();
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    this.initFFL();
  }

  /** init_ffl: the friction and noise constants for the current target temperature and step. */
  private initFFL(): void {
    const u = this.sys.state.units;
    const kT = this.tTarget * u.boltz / u.mvv2e;
    this.c1 = libmExp(-this.gamma * 0.5 * this.dtv);
    this.c2 = Math.sqrt((1.0 - this.c1 * this.c1) * kT);
  }

  private grow(n3: number): void {
    if (this.tmp1.length < n3) {
      this.tmp1 = new Float64Array(n3);
      this.tmp2 = new Float64Array(n3);
    }
  }

  /** ffl_integrate: the Ornstein-Uhlenbeck update of the mass-scaled velocities, then the flip. */
  private fflIntegrate(): void {
    const s = this.sys.state;
    const { v, mask, type } = s;
    const bit = this.groupBit;
    const order = nativeOrder(s);
    this.grow(3 * s.n);
    const tmp1 = this.tmp1;
    const tmp2 = this.tmp2;
    const sqrtM = this.sqrtM;
    const rmass = s.rmass;
    // loads momentum data (mass-scaled) into the temporary vector and accumulates the conserved quantity
    let nk = 0;
    let deltae = 0.0;
    for (const i of order) {
      if (!(mask[i] & bit)) continue;
      const smi = rmass ? Math.sqrt(rmass[i]) : sqrtM[type[i]];
      for (let k = 0; k < 3; k++) {
        tmp2[nk] = v[3 * i + k] * smi;
        deltae += tmp2[nk] * tmp2[nk];
        nk++;
      }
    }
    // Gaussian deviates, one per component of the group
    for (let i = 0; i < nk; i++) tmp1[i] = this.rng.gaussian();
    // unloads the mass-scaled momentum with the new velocities
    nk = 0;
    const hard = this.flip === 'hard';
    for (const i of order) {
      if (!(mask[i] & bit)) continue;
      const ismi = rmass ? 1.0 / Math.sqrt(rmass[i]) : 1.0 / sqrtM[type[i]];
      for (let k = 0; k < 3; k++) {
        v[3 * i + k] = this.c1 * v[3 * i + k] + this.c2 * tmp1[nk] * ismi;
        deltae -= v[3 * i + k] * v[3 * i + k] / ismi / ismi;
        // hard flip: the sign of the momentum component is flipped when it reverses
        if (hard && v[3 * i + k] * tmp2[nk] < 0.0) v[3 * i + k] = -v[3 * i + k];
        nk++;
      }
    }
    // rescale flip: keep the old direction, the new magnitude
    if (this.flip === 'rescale') {
      nk = 0;
      for (const i of order) {
        if (!(mask[i] & bit)) continue;
        const factor = Math.sqrt((v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]) /
          (tmp2[nk] * tmp2[nk] + tmp2[nk + 1] * tmp2[nk + 1] + tmp2[nk + 2] * tmp2[nk + 2]));
        for (let k = 0; k < 3; k++) {
          v[3 * i + k] = factor * tmp2[nk];
          nk++;
        }
      }
    }
    // soft flip: reflect the component along the old direction only when it reverses
    if (this.flip === 'soft') {
      nk = 0;
      for (const i of order) {
        if (!(mask[i] & bit)) continue;
        let factor = v[3 * i] * tmp2[nk] + v[3 * i + 1] * tmp2[nk + 1] + v[3 * i + 2] * tmp2[nk + 2];
        if (factor < 0) {
          factor = factor / (tmp2[nk] * tmp2[nk] + tmp2[nk + 1] * tmp2[nk + 1] + tmp2[nk + 2] * tmp2[nk + 2]);
          for (let k = 0; k < 3; k++) {
            v[3 * i + k] -= 2.0 * factor * tmp2[nk];
            nk++;
          }
        } else {
          nk += 3;
        }
      }
    }
    this.energyCum += deltae * 0.5 * s.units.mvv2e;
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    this.fflStep--;
    if (this.doffl && this.fflStep < 1) this.fflIntegrate();
    const dtf = this.dtf, dtv = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(s, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      x[3 * i] += dtv * v[3 * i];
      x[3 * i + 1] += dtv * v[3 * i + 1];
      x[3 * i + 2] += dtv * v[3 * i + 2];
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const { v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(s, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
    }
    if (this.doffl && this.fflStep < 1) {
      this.fflIntegrate();
      this.fflStep = this.ffl_every;
    }
    // the target temperature of the next step: a linear ramp over the run
    const delta = (s.step - this.sys.run.beginStep) / (this.sys.run.endStep - this.sys.run.beginStep);
    this.tTarget = this.tStart + delta * (this.tStop - this.tStart);
    // only updates if it is really necessary
    if (this.tStop !== this.tStart) this.initFFL();
  }
}
