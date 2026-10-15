/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BROWNIAN/fix_brownian_base.cpp and
 * fix_brownian_base.h (class FixBrownianBase: the keyword parsing, init() and reset_dt() of the Brownian
 * integrators). Originally modified from CG-DNA/fix_nve_dotc_langevin.cpp. Contributing authors: Sam Cameron
 * (University of Bristol) and Arthur Straube (Zuse Institute Berlin), and the LAMMPS developers.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The random stream is RanMars seeded with the fix's seed (one processor: seed + comm->me, me = 0).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseInt_, parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { RanMars } from '../random_mars';
import { takeFixRecord, type MarsRestart } from '../../engine/fix/restartRecord';

/** A numeric argument; utils::numeric (strtod) accepts "inf", which the eigen-damping keywords use. */
export const brownNum = (w: string | undefined, what: string): number => (w === 'inf' ? Infinity : parseNum(w, what));

/** Arguments after the keyword must be present: utils::missing_cmd_args. */
const needArgs = (args: string[], k: number, count: number, what: string): void => {
  if (args.length < k + count) throw new StyleError(`Illegal ${what} command: missing arguments`);
};

export type RotStyle = 'projection' | 'geometric';
export type NoiseMode = 'none' | 'uniform' | 'gaussian';

export abstract class FixBrownianBase extends Fix {
  timeIntegrate = true;
  protected readonly seed: number;
  protected readonly temp: number;
  protected rotTemp = 0;
  protected dt = 0;
  protected sqrtdt = 0;
  protected gammaTFlag = false;
  protected gammaRFlag = false;
  protected gammaTEigenFlag = false;
  protected gammaREigenFlag = false;
  protected rotTempFlag = false;
  protected planarRotFlag = false;
  protected rotStyle: RotStyle;
  protected gammaT = 0;
  protected gammaR = 0;
  protected gammaTInv: number[] = [0, 0, 0];
  protected gammaRInv: number[] = [0, 0, 0];
  protected gammaTInvsqrt: number[] = [0, 0, 0];
  protected gammaRInvsqrt: number[] = [0, 0, 0];
  protected dipoleFlag = false;
  protected dipoleBody: number[] = [0, 0, 0];
  protected noiseMode: NoiseMode = 'uniform';
  protected g1 = 0;
  protected g2 = 0;
  protected readonly rng: RanMars;

  constructor(sys: System, id: string, group: string, args: string[], readonly style: string) {
    super(sys, id, group, args);
    this.rotStyle = style.startsWith('brownian/sphere') ? 'projection' : 'geometric';
    if (args.length < 2) throw new StyleError(`Illegal fix ${style} command: missing arguments`);
    this.temp = brownNum(args[0], 'temp');
    if (this.temp <= 0) throw new StyleError(`Fix ${style} temp must be > 0.0`);
    this.seed = parseInt_(args[1], 'seed');
    if (this.seed <= 0) throw new StyleError(`Fix ${style} seed must be > 0`);
    const dim = sys.dimension;
    let k = 2;
    while (k < args.length) {
      const kw = args[k];
      if (kw === 'rng') {
        needArgs(args, k, 2, `fix ${style} rng`);
        const mode = args[k + 1];
        if (mode === 'uniform') this.noiseMode = 'uniform';
        else if (mode === 'gaussian') this.noiseMode = 'gaussian';
        else if (mode === 'none') this.noiseMode = 'none';
        else throw new StyleError(`Unknown fix ${style} rng keyword ${mode}`);
        k += 2;
      } else if (kw === 'dipole') {
        needArgs(args, k, 4, `fix ${style} dipole`);
        this.dipoleFlag = true;
        this.dipoleBody = [brownNum(args[k + 1], 'dipole'), brownNum(args[k + 2], 'dipole'), brownNum(args[k + 3], 'dipole')];
        k += 4;
      } else if (kw === 'gamma_t_eigen') {
        needArgs(args, k, 4, `fix ${style} gamma_t_eigen`);
        const t = [brownNum(args[k + 1], 'gamma_t_eigen'), brownNum(args[k + 2], 'gamma_t_eigen'), 0];
        if (dim === 2) {
          if (args[k + 3] !== 'inf') throw new StyleError(`Fix ${style} gamma_t_eigen third value must be inf for 2D system`);
          t[2] = 1.0;
        } else t[2] = brownNum(args[k + 3], 'gamma_t_eigen');
        if (t[0] <= 0.0 || t[1] <= 0.0 || t[2] <= 0.0) throw new StyleError(`Fix ${style} gamma_t_eigen values must be > 0`);
        this.gammaTEigenFlag = true;
        this.gammaTInv = [1.0 / t[0], 1.0 / t[1], 1.0 / t[2]];
        if (dim === 2) this.gammaTInv[2] = 0.0;
        this.gammaTInvsqrt = [Math.sqrt(this.gammaTInv[0]), Math.sqrt(this.gammaTInv[1]), Math.sqrt(this.gammaTInv[2])];
        k += 4;
      } else if (kw === 'gamma_r_eigen') {
        needArgs(args, k, 4, `fix ${style} gamma_r_eigen`);
        const r = [0, 0, 0];
        if (dim === 2) {
          if (args[k + 1] !== 'inf') throw new StyleError(`Fix ${style} gamma_r_eigen first value must be inf for 2D system`);
          r[0] = 1.0;
          if (args[k + 2] !== 'inf') throw new StyleError(`Fix ${style} gamma_r_eigen second value must be inf for 2D system`);
          r[1] = 1.0;
        } else {
          r[0] = brownNum(args[k + 1], 'gamma_r_eigen');
          r[1] = brownNum(args[k + 2], 'gamma_r_eigen');
        }
        r[2] = brownNum(args[k + 3], 'gamma_r_eigen');
        if (r[0] <= 0.0 || r[1] <= 0 || r[2] <= 0) throw new StyleError(`Fix ${style} gamma_r_eigen values must be > 0`);
        this.gammaREigenFlag = true;
        this.gammaRInv = [1.0 / r[0], 1.0 / r[1], 1.0 / r[2]];
        if (dim === 2) {
          this.gammaRInv[0] = 0.0;
          this.gammaRInv[1] = 0.0;
        }
        this.gammaRInvsqrt = [Math.sqrt(this.gammaRInv[0]), Math.sqrt(this.gammaRInv[1]), Math.sqrt(this.gammaRInv[2])];
        k += 4;
      } else if (kw === 'gamma_t') {
        needArgs(args, k, 2, `fix ${style} gamma_t`);
        this.gammaTFlag = true;
        this.gammaT = brownNum(args[k + 1], 'gamma_t');
        if (this.gammaT <= 0.0) throw new StyleError(`Fix ${style} gamma_t value must be > 0`);
        k += 2;
      } else if (kw === 'gamma_r') {
        needArgs(args, k, 2, `fix ${style} gamma_r`);
        this.gammaRFlag = true;
        this.gammaR = brownNum(args[k + 1], 'gamma_r');
        if (this.gammaR <= 0) throw new StyleError(`Fix ${style} gamma_r value must be > 0`);
        k += 2;
      } else if (kw === 'rotation_temp') {
        needArgs(args, k, 2, `fix ${style} rotation_temp`);
        this.rotTempFlag = true;
        this.rotTemp = brownNum(args[k + 1], 'rotation_temp');
        if (this.rotTemp <= 0) throw new StyleError(`Fix ${style} rotation_temp value must be > 0`);
        k += 2;
      } else if (kw === 'planar_rotation') {
        this.planarRotFlag = true;
        if (dim === 2) throw new StyleError('The planar_rotation keyword is not allowed for 2D simulations');
        k += 1;
      } else if (kw === 'rotation_style') {
        if (!style.startsWith('brownian/sphere')) throw new StyleError('Keyword rotation_style is only supported for fix brownian/sphere');
        needArgs(args, k, 2, 'fix brownian rotation_style');
        if (args[k + 1] === 'projection') this.rotStyle = 'projection';
        else if (args[k + 1] === 'geometric') this.rotStyle = 'geometric';
        else throw new StyleError(`Unknown fix ${style} rotation_style ${args[k + 1]}`);
        k += 2;
      } else {
        throw new StyleError(`Unknown fix ${style} keyword ${kw}`);
      }
    }
    if (!this.rotTempFlag) this.rotTemp = this.temp;
    this.rng = new RanMars(this.seed);
    const saved = takeFixRecord(sys, id, style)?.state as MarsRestart | undefined;
    if (saved) this.rng.setState(saved.rng);
  }

  /** fix_brownian_base.cpp write_restart: the generator state of the one processor (RanMars::get_state). */
  restartState(): MarsRestart { return { rng: this.rng.getState() }; }

  /** Whether the noise terms are drawn (rng keyword other than none). */
  protected get noisy(): boolean {
    return this.noiseMode !== 'none';
  }

  /**
   * One noise factor of the current rng keyword: uniform gives rng->uniform() - 0.5, gaussian gives rng->gaussian().
   * Only called when noisy.
   */
  protected noise(): number {
    return this.noiseMode === 'gaussian' ? this.rng.gaussian() : this.rng.uniform() - 0.5;
  }

  /** FixBrownianBase::init(): time step, ftm2v and the noise prefactor g2. */
  override init(): void {
    const u = this.sys.units;
    this.dt = this.sys.state.dt;
    this.sqrtdt = Math.sqrt(this.dt);
    this.g1 = u.ftm2v;
    if (this.noiseMode === 'none') this.g2 = 0.0;
    else if (this.noiseMode === 'gaussian') this.g2 = Math.sqrt((2 * u.boltz) / this.dt / u.mvv2e);
    else this.g2 = Math.sqrt((24 * u.boltz) / this.dt / u.mvv2e);
  }

  /** FixBrownianBase::reset_dt(): the noise prefactor scales with 1/sqrt(dt). */
  override resetDt(): void {
    const sqrtdtOld = this.sqrtdt;
    this.dt = this.sys.state.dt;
    this.sqrtdt = Math.sqrt(this.dt);
    this.g2 *= sqrtdtOld / this.sqrtdt;
  }
}
