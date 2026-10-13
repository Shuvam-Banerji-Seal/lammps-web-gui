/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/fix_grem.cpp and fix_grem.h (FixGrem: constructor, init, setup, post_force, compute_scalar,
 * extract) at commit b332f28. Copyright Sandia Corporation (GNU General Public License). Contributing authors:
 * Edyta Malolepsza (Broad Institute), David Stelter and Tom Keyes (Boston University). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Generalized replica exchange method, single replica (docs.lammps.org/fix_grem.html): the forces of the group are
 * scaled each step by tbath / (lambda + eta (H - h0)), with H the enthalpy pe + P V / nktv2p. The thermostat or
 * barostat is the fix nvt / npt named last (its target temperature must not ramp; a barostat must be isotropic).
 * Refused: triclinic boxes, 2d, the temper/grem replica exchange (it needs partitions) and run_style respa.
 */
import { Fix } from '../../engine/fix/fix';
import { FixNH } from '../../engine/fix/nh';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { ownCompute, removeCompute } from '../../engine/fix/util';
import { ComputePressureGrem } from './compute_pressure_grem';

export class FixGrem extends Fix {
  readonly style = 'grem';
  private readonly lambda: number;
  private readonly eta: number;
  private readonly h0: number;
  private readonly idNh: string;
  private readonly idTemp: string;
  private readonly idPress: string;
  private readonly idKe: string;
  private readonly idPe: string;
  private scaleGrem = 1.0;
  private tbath = 0;
  private pressRef = 0;
  private pressFlag = false;
  /** The fix nvt/npt, looked up by ID (the thermostat or barostat of the gREM). */
  private nhKinetic(): Float64Array | null {
    const nh = this.sys.fixes.find((f) => f.id === this.idNh);
    return nh instanceof FixNH ? nh.initialKinetic : null;
  }

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4) throw new StyleError('Illegal fix grem command. Expecting: fix ID group grem lambda eta h0 fix-ID-nvt-or-npt');
    this.scalarFlag = true;
    this.extscalar = 0;
    this.lambda = Number(args[0]);
    this.eta = Number(args[1]);
    this.h0 = Number(args[2]);
    if (!Number.isFinite(this.lambda) || !Number.isFinite(this.eta) || !Number.isFinite(this.h0)) throw new StyleError('Illegal fix grem command');
    this.idNh = args[3];
    if (sys.dimension !== 3) throw new StyleError('fix grem is only supported in 3d');
    // the computes the source creates: temperature (all), pressure/grem, ke and pe (all)
    this.idTemp = `${id}_temp`;
    this.idPress = `${id}_press`;
    this.idKe = `${id}_ke`;
    this.idPe = `${id}_pe`;
    ownCompute(sys, this.idTemp, 'all', 'temp', []);
    removeCompute(sys, this.idPress);
    sys.computes.push(new ComputePressureGrem(sys, this.idPress, this.idTemp, () => this.scaleGrem, () => this.nhKinetic()));
    ownCompute(sys, this.idKe, 'all', 'ke', []);
    ownCompute(sys, this.idPe, 'all', 'pe', []);
    const nh = sys.fixes.find((f) => f.id === this.idNh);
    if (!nh) throw new StyleError('Fix id for nvt or npt fix does not exist');
    if (!(nh instanceof FixNH)) throw new StyleError(`Fix ID ${this.idNh} is not a compatible Nose-Hoover fix for fix grem`);
    // a barostat with all three dimensions takes the gREM pressure (the source's modify_param press)
    const pf = nh.extract('p_flag');
    if (Array.isArray(pf) && pf[0] && pf[1] && pf[2]) {
      this.pressFlag = true;
      nh.modify('press', [this.idPress]);
    }
  }

  override destroy(): void {
    // the source deletes its computes with the fix (unfix)
    for (const c of [this.idTemp, this.idPress, this.idKe, this.idPe]) removeCompute(this.sys, c);
  }

  init(): void {
    if (this.sys.state.box.triclinic) throw new StyleError('Triclinic cells are not supported');
    const nh = this.sys.fixes.find((f) => f.id === this.idNh);
    if (!(nh instanceof FixNH)) throw new StyleError(`Fix ID ${this.idNh} is not a compatible Nose-Hoover fix for fix grem`);
    const tStart = nh.extract('t_start');
    const tStop = nh.extract('t_stop');
    if (typeof tStart !== 'number' || typeof tStop !== 'number') throw new StyleError(`Problem extracting target temperature from fix ${nh.style}`);
    this.tbath = tStart;
    if (tStart !== tStop) throw new StyleError(`Temperature ramp in fix ${nh.style} ID ${this.idNh} not allowed`);
    this.pressRef = 0.0;
    if (this.pressFlag) {
      const pFlag = nh.extract('p_flag');
      const pStart = nh.extract('p_start');
      const pStop = nh.extract('p_stop');
      if (!Array.isArray(pFlag) || !Array.isArray(pStart) || !Array.isArray(pStop)) {
        throw new StyleError(`Problem extracting target pressure from fix ${nh.style}`);
      }
      this.pressRef = pStart[0];
      // the source's checks: equal start and stop in every dimension, all dimensions barostatted, one value, no tilt
      let bad = 0;
      if (pStart[0] !== pStop[0] || pFlag[0] !== 1) bad++;
      if (pStart[1] !== pStop[1] || pFlag[0] !== 1) bad++;
      if (pStart[2] !== pStop[2] || pFlag[0] !== 1) bad++;
      if (pStart[0] !== pStart[1] || pStart[1] !== pStart[2]) bad++;
      if (pFlag[3] !== 0 || pFlag[4] !== 0 || pFlag[5] !== 0) bad++;
      if (bad > 0) throw new StyleError(`Unsupported pressure settings in fix ${nh.style} ID ${this.idNh}`);
    }
  }

  /** The source's setup: post_force for verlet; respa is refused. */
  setup(): void { this.postForce(); }

  minSetup(): void { this.postForce(); }

  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const vol = sys.geom.volume(3);
    sys.refreshComputes();
    const pe = sys.compute(this.idPe).scalarValue();
    // enthalpy H = pe + P V / nktv2p
    const enthalpy = pe + (this.pressRef * vol) / s.units.nktv2p;
    const teffective = this.lambda + this.eta * (enthalpy - this.h0);
    this.scaleGrem = this.tbath / teffective;
    const { f, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      f[3 * i] *= this.scaleGrem;
      f[3 * i + 1] *= this.scaleGrem;
      f[3 * i + 2] *= this.scaleGrem;
    }
  }

  computeScalar(): number { return this.tbath / this.scaleGrem; }

  /** extract("scale_grem") of the source, read by PRESSURE/GREM. */
  get scale(): number { return this.scaleGrem; }
}
