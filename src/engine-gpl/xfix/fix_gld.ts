/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_gld.cpp and src/EXTRA-FIX/fix_gld.h at commit b332f28 (class FixGLD, fix gld: the
 * constructor, init, init_s_gld, initial_integrate, final_integrate, reset_target, reset_dt; contributing authors
 * Stephen Bond (SNL) and Dave Hillier). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Generalized Langevin dynamics with a positive Prony series (series type pprony). The native build selects the
 * uniform random distribution (#define GLD_UNIFORM_DISTRO): every auxiliary variable draws 3 uniform deviates, scaled
 * to unit variance by sqrt(12). The per-atom auxiliary variables live in the atom custom arrays (one row of
 * 3*prony_terms values per atom), so they follow the atoms when the engine reorders them.
 * Keywords: zero yes|no (remove the mean random force of the group), frozen yes|no (zero the auxiliary variables).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { exp as libmExp } from '../libm/exp';

function numeric(w: string | undefined): number {
  const x = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(x)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return x;
}

function inumeric(w: string | undefined): number {
  if (w === undefined || !/^[+-]?\d+$/.test(w.trim())) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return Number(w);
}

function logical(w: string | undefined): boolean {
  if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
  if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in input script or data file`);
}

export class FixGLD extends Fix {
  readonly style = 'gld';
  private readonly tStart: number;
  private readonly tStop: number;
  private readonly terms: number;
  private readonly pronyC: Float64Array;
  private readonly pronyTau: Float64Array;
  private readonly rng: RanMars;
  private readonly key: string;
  private tTarget: number;
  private zeroFlag = false;
  private dtv = 0;
  private dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // native narg counts "fix ID group gld": narg < 8 is fewer than five arguments
    if (args.length < 5) throw new StyleError('Illegal fix gld command');
    this.timeIntegrate = true;
    this.tStart = numeric(args[0]);
    this.tStop = numeric(args[1]);
    this.terms = inumeric(args[2]);
    const seed = inumeric(args[3]);
    if (args[4] !== 'pprony') throw new StyleError('Fix gld series type must be pprony for now');
    if (seed <= 0) throw new StyleError('Illegal fix gld command');
    if (this.terms <= 0) throw new StyleError('Fix gld prony terms must be > 0');
    if (this.tStart < 0) throw new StyleError('Fix gld start temperature must be >= 0');
    if (this.tStop < 0) throw new StyleError('Fix gld stop temperature must be >= 0');
    if (args.length - 5 < 2 * this.terms) throw new StyleError('Fix gld needs more prony series coefficients');
    this.pronyC = new Float64Array(this.terms);
    this.pronyTau = new Float64Array(this.terms);
    this.key = `#gld:${id}`;

    let iarg = 5;
    let icoeff = 0;
    while (iarg < args.length && icoeff < this.terms) {
      const pc = numeric(args[iarg]);
      const ptau = numeric(args[iarg + 1]);
      if (pc < 0) throw new StyleError('Fix gld c coefficients must be >= 0');
      if (ptau <= 0) throw new StyleError('Fix gld tau coefficients must be > 0');
      this.pronyC[icoeff] = pc;
      this.pronyTau[icoeff] = ptau;
      icoeff += 1;
      iarg += 2;
    }
    this.rng = new RanMars(seed);
    this.initSGld();
    while (iarg < args.length) {
      if (args[iarg] === 'zero') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal fix gld command');
        this.zeroFlag = logical(args[iarg + 1]);
        iarg += 2;
      } else if (args[iarg] === 'frozen') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal fix gld command');
        if (logical(args[iarg + 1])) {
          const st = sys.state;
          const sg = this.sArr();
          const cols = 3 * this.terms;
          for (let i = 0; i < st.n; i++) {
            if (st.mask[i] & this.groupBit) for (let k = 0; k < cols; k++) sg[i * cols + k] = 0.0;
          }
        }
        iarg += 2;
      } else {
        throw new StyleError('Illegal fix gld command');
      }
    }
    this.tTarget = this.tStart;
  }

  /** The per-atom auxiliary variables (3 * prony_terms per atom), created at the fix's construction. */
  private sArr(): Float64Array {
    const st = this.sys.state;
    const cols = 3 * this.terms;
    let c = st.custom.get(this.key);
    if (!c) {
      c = { int: false, cols, data: new Float64Array(st.n * cols) };
      st.custom.set(this.key, c);
    }
    return c.data;
  }

  /** init_s_gld: the initial auxiliary variables drawn from the equilibrium distribution at Tstart (native order). */
  private initSGld(): void {
    const st = this.sys.state;
    const sg = this.sArr();
    const cols = 3 * this.terms;
    const u = st.units;
    // set kT to the temperature in mvvv units
    const kT = (u.boltz * this.tStart) / u.mvv2e;
    const scale = Math.sqrt(12.0 * kT) / u.ftm2v;
    for (const i of nativeOrder(st)) {
      if (st.mask[i] & this.groupBit) {
        let icoeff = 0;
        for (let k = 0; k < cols; k += 3) {
          const eqSdev = scale * Math.sqrt(this.pronyC[icoeff] / this.pronyTau[icoeff]);
          sg[i * cols + k] = eqSdev * (this.rng.uniform() - 0.5);
          sg[i * cols + k + 1] = eqSdev * (this.rng.uniform() - 0.5);
          sg[i * cols + k + 2] = eqSdev * (this.rng.uniform() - 0.5);
          icoeff += 1;
        }
      } else {
        // zero s_gld for atoms outside the group so it is never left uninitialised
        for (let k = 0; k < cols; k++) sg[i * cols + k] = 0.0;
      }
    }
  }

  init(): void { this.resetDt(); }

  resetDt(): void {
    const st = this.sys.state;
    this.dtv = st.dt;
    this.dtf = 0.5 * st.dt * st.units.ftm2v;
  }

  initialIntegrate(): void {
    const st = this.sys.state;
    const { x, v, f, mask } = st;
    const bit = this.groupBit;
    const sg = this.sArr();
    const cols = 3 * this.terms;
    const u = st.units;
    const ftm2v = u.ftm2v;
    const dtv = this.dtv, dtf = this.dtf;
    const kT = (u.boltz * this.tTarget) / u.mvv2e;
    const fsum = [0.0, 0.0, 0.0];
    const sqrt12 = Math.sqrt(12.0);
    const order = nativeOrder(st);
    for (const i of order) {
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(st, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      for (let k = 0; k < cols; k += 3) {
        v[3 * i] += dtfm * sg[i * cols + k];
        v[3 * i + 1] += dtfm * sg[i * cols + k + 1];
        v[3 * i + 2] += dtfm * sg[i * cols + k + 2];
      }
      x[3 * i] += dtv * v[3 * i];
      x[3 * i + 1] += dtv * v[3 * i + 1];
      x[3 * i + 2] += dtv * v[3 * i + 2];
      let icoeff = 0;
      for (let k = 0; k < cols; k += 3) {
        const theta = libmExp(-dtv / this.pronyTau[icoeff]);
        const ck = this.pronyC[icoeff];
        const vmult = ((theta - 1.0) * ck) / ftm2v;
        let rmult = (Math.sqrt((2.0 * kT * ck) / dtv) * (1.0 - theta)) / ftm2v;
        // the uniform distribution: rescaled to the variance of the Gaussian (sqrt(12) factor)
        rmult *= sqrt12;
        const fran0 = rmult * (this.rng.uniform() - 0.5);
        const fran1 = rmult * (this.rng.uniform() - 0.5);
        const fran2 = rmult * (this.rng.uniform() - 0.5);
        fsum[0] += fran0;
        fsum[1] += fran1;
        fsum[2] += fran2;
        const b = i * cols + k;
        sg[b] *= theta;
        sg[b + 1] *= theta;
        sg[b + 2] *= theta;
        sg[b] += vmult * v[3 * i];
        sg[b + 1] += vmult * v[3 * i + 1];
        sg[b + 2] += vmult * v[3 * i + 2];
        sg[b] += fran0;
        sg[b + 1] += fran1;
        sg[b + 2] += fran2;
        icoeff += 1;
      }
    }
    if (this.zeroFlag) {
      let count = 0;
      for (let i = 0; i < st.n; i++) if (mask[i] & bit) count++;
      if (count === 0) throw new StyleError('Cannot zero gld force for zero atoms');
      const denom = count * this.terms;
      const fsumall = [fsum[0] / denom, fsum[1] / denom, fsum[2] / denom];
      for (const i of order) {
        if (!(mask[i] & bit)) continue;
        for (let k = 0; k < cols; k += 3) {
          sg[i * cols + k] -= fsumall[0];
          sg[i * cols + k + 1] -= fsumall[1];
          sg[i * cols + k + 2] -= fsumall[2];
        }
      }
    }
  }

  finalIntegrate(): void {
    const st = this.sys.state;
    const { v, f, mask } = st;
    const bit = this.groupBit;
    const sg = this.sArr();
    const cols = 3 * this.terms;
    const dtf = this.dtf;
    for (const i of nativeOrder(st)) {
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(st, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      for (let k = 0; k < cols; k += 3) {
        v[3 * i] += dtfm * sg[i * cols + k];
        v[3 * i + 1] += dtfm * sg[i * cols + k + 1];
        v[3 * i + 2] += dtfm * sg[i * cols + k + 2];
      }
    }
    const delta = (st.step - this.sys.run.beginStep) / (this.sys.run.endStep - this.sys.run.beginStep);
    this.tTarget = this.tStart + delta * (this.tStop - this.tStart);
  }
}
