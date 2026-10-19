/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANIFOLD/fix_nvt_manifold_rattle.cpp
 * and fix_nvt_manifold_rattle.h at commit b332f28 (fix nvt/manifold/rattle: nve/manifold/rattle with a Nose-Hoover
 * chain thermostat). MANIFOLD package by Stefan Paquay (Eindhoven University of Technology), GNU General Public
 * License; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/fix_nvt_manifold_rattle.html. Native behaviour kept as it is:
 * - the timestep sizes (dtv, dtf, dthalf, dt4, dt8 and the drag factor) are set once in the constructor; init does
 *   not reset them (FixNVTManifoldRattle::init replaces FixNVEManifoldRattle::init), so only a timestep command
 *   after the first run changes them (Input::timestep calls reset_dt only once update->first_update is set);
 * - the second half of the chain update multiplies eta_dot by eta_dotdot dt4 (eta_dot[ich] *= eta_dotdot[ich] * dt4);
 * - the keyword every of nve/manifold/rattle is an unknown keyword here.
 */
import { StyleError } from '../../engine/force/types';
import { libm } from '../../engine/libm';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { ownCompute, removeCompute } from '../../engine/fix/util';
import { FixNVEManifoldRattle, inumeric, numeric } from './fix_nve_manifold_rattle';

export class FixNVTManifoldRattle extends FixNVEManifoldRattle {
  override readonly style: string = 'nvt/manifold/rattle';
  private dthalf = 0;
  private dt4 = 0;
  private dt8 = 0;
  private tStart = 0;
  private tStop = 0;
  private tPeriod = 0;
  private tCurrent = 0;
  private tTarget = 0;
  private keTarget = 0;
  private tFreq = 0;
  private readonly drag = 0;
  private tdragFactor = 0;
  private tdof = 0;
  private mtchain = 3;
  private factorEta = 0.0;
  private readonly tempId: string;
  private temp: Compute;
  /** Set at the end of the constructor: later reset_dt calls come from a timestep command. */
  private constructed = false;
  private eta: Float64Array;
  private etaDot: Float64Array;
  private etaDotdot: Float64Array;
  private etaMass: Float64Array;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, false);
    if (args.length < 3) throw new StyleError('Illegal fix nvt/manifold/rattle command');
    let gotTemp = false;
    let argi = 3 + this.man.nparams;
    while (argi < args.length) {
      if (args[argi] === 'temp') {
        // native: argi+3 >= narg, with narg counting "ID group style"
        if (argi + 3 + 3 >= args.length + 3) throw new StyleError("Keyword 'temp' needs 3 arguments");
        this.tStart = numeric(args[argi + 1]);
        this.tStop = numeric(args[argi + 2]);
        this.tPeriod = numeric(args[argi + 3]);
        this.tTarget = this.tStart;
        gotTemp = true;
        argi += 4;
      } else if (args[argi] === 'tchain') {
        if (argi + 3 + 1 >= args.length + 3) throw new StyleError("Keyword 'tchain' needs 1 argument");
        this.mtchain = inumeric(args[argi + 1]);
        argi += 2;
      } else {
        throw new StyleError(`Error parsing arg "${args[argi]}".`);
      }
    }
    // reset_dt with t_freq still 0 (it is set below), as native
    this.resetDt();
    if (!gotTemp) throw new StyleError("Fix nvt/manifold/rattle needs 'temp'!");
    if (this.tPeriod < 0.0) throw new StyleError('Fix nvt/manifold/rattle damping parameter must be > 0.0');
    this.tempId = `${id}_temp`;
    this.temp = ownCompute(sys, this.tempId, group, 'temp', []);
    this.tFreq = 1.0 / this.tPeriod;
    const m = this.mtchain;
    this.eta = new Float64Array(m);
    this.etaDot = new Float64Array(m + 1);
    this.etaDotdot = new Float64Array(m);
    this.etaMass = new Float64Array(m);
    this.constructed = true;
  }

  /** init: the manifold parameters and the temperature compute; the timestep sizes stay those of the constructor. */
  override init(): void {
    this.updateVarParams();
    const c = this.sys.computes.find((x) => x.id === this.tempId);
    if (!c) throw new StyleError(`Temperature compute ID ${this.tempId} for fix ${this.style} does not exist`);
    if (!c.tempFlag) throw new StyleError(`Compute ID ${this.tempId} for fix ${this.style} does not compute a temperature`);
    this.temp = c;
  }

  override resetDt(): void {
    // a timestep command before the first run does not reach the fixes (input.cpp: if (update->first_update == 0) return)
    if (this.constructed && !this.sys.run.ranOnce) return;
    super.resetDt();
    const dt = this.sys.state.dt;
    this.dthalf = 0.5 * dt;
    this.dt4 = 0.25 * dt;
    this.dt8 = 0.125 * dt;
    this.tdragFactor = 1.0 - (dt * this.tFreq * this.drag);
  }

  override destroy(): void {
    removeCompute(this.sys, this.tempId);
  }

  private currentTemp(): number {
    this.sys.refreshComputes();
    return this.temp.scalarValue();
  }

  override setup(): void {
    this.computeTempTarget();
    this.tCurrent = this.currentTemp();
    this.tdof = this.temp.dof;
    const boltz = this.sys.state.units.boltz;
    const invTFreq2 = 1.0 / (this.tFreq * this.tFreq);
    this.etaMass[0] = this.tdof * boltz * this.tTarget * invTFreq2;
    for (let ich = 1; ich < this.mtchain; ++ich) this.etaMass[ich] = boltz * this.tTarget * invTFreq2;
    for (let ich = 1; ich < this.mtchain; ++ich) {
      this.etaDotdot[ich] = (this.etaMass[ich - 1] * this.etaDot[ich - 1] * this.etaDot[ich - 1] - boltz * this.tTarget) / this.etaMass[ich];
    }
  }

  private computeTempTarget(): void {
    this.tCurrent = this.currentTemp();
    this.tdof = this.temp.dof;
    const run = this.sys.run;
    let delta = this.sys.state.step - run.beginStep;
    if (delta !== 0.0) delta /= run.endStep - run.beginStep;
    this.tdof = this.temp.dof;
    this.tTarget = this.tStart + delta * (this.tStop - this.tStart);
    this.keTarget = this.tdof * this.sys.state.units.boltz * this.tTarget;
  }

  private nhcTempIntegrate(): void {
    this.computeTempTarget();
    const boltz = this.sys.state.units.boltz;
    const m = this.mtchain;
    const etaDot = this.etaDot, etaDotdot = this.etaDotdot, etaMass = this.etaMass;
    let kecurrent = this.tdof * boltz * this.tCurrent;
    const invTFreq2 = 1.0 / (this.tFreq * this.tFreq);
    etaMass[0] = this.tdof * boltz * this.tTarget * invTFreq2;
    for (let ich = 1; ich < m; ++ich) etaMass[ich] = boltz * this.tTarget * invTFreq2;
    etaDotdot[0] = etaMass[0] > 0.0 ? (kecurrent - this.keTarget) / etaMass[0] : 0;
    let expfac: number;
    for (let ich = m - 1; ich > 0; --ich) {
      expfac = libm.exp(-this.dt8 * etaDot[ich + 1]);
      etaDot[ich] *= expfac;
      etaDot[ich] += etaDotdot[ich] * this.dt4;
      etaDot[ich] *= this.tdragFactor * expfac;
    }
    expfac = libm.exp(-this.dt8 * etaDot[1]);
    etaDot[0] *= expfac;
    etaDot[0] += etaDotdot[0] * this.dt4;
    etaDot[0] *= this.tdragFactor * expfac;
    this.factorEta = libm.exp(-this.dthalf * etaDot[0]);
    if (this.factorEta === 0) throw new StyleError(`factor_eta is 0! dthalf = ${this.dthalf}, eta_dot[0] = ${etaDot[0]}`);
    this.nhVTemp();
    this.tCurrent *= this.factorEta * this.factorEta;
    kecurrent = this.tdof * boltz * this.tCurrent;
    etaDotdot[0] = etaMass[0] > 0.0 ? (kecurrent - this.keTarget) / etaMass[0] : 0.0;
    for (let ich = 1; ich < m; ++ich) this.eta[ich] += this.dthalf * etaDot[ich];
    etaDot[0] *= expfac;
    etaDot[0] += etaDotdot[0] * this.dt4;
    etaDot[0] *= expfac;
    for (let ich = 1; ich < m; ++ich) {
      expfac = libm.exp(-this.dt8 * etaDot[ich + 1]);
      etaDot[ich] *= expfac;
      etaDotdot[ich] = (etaMass[ich - 1] * etaDot[ich - 1] * etaDot[ich - 1] - boltz * this.tTarget) / etaMass[ich];
      etaDot[ich] *= etaDotdot[ich] * this.dt4;
      etaDot[ich] *= expfac;
    }
  }

  /** nh_v_temp: scale the group's velocities by factor_eta (with the compute's bias removed, if it has one). */
  private nhVTemp(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const bit = this.groupBit;
    const f = this.factorEta;
    const bias = this.temp.hasBias();
    if (bias) this.temp.computeBias();
    for (let i = 0; i < s.n; ++i) {
      if (!(mask[i] & bit)) continue;
      if (bias) this.temp.removeBias(i);
      v[3 * i] *= f;
      v[3 * i + 1] *= f;
      v[3 * i + 2] *= f;
      if (bias) this.temp.restoreBias(i);
    }
  }

  override initialIntegrate(): void {
    this.updateVarParams();
    this.computeTempTarget();
    this.nhcTempIntegrate();
    this.nveXRattle();
  }

  override finalIntegrate(): void {
    this.nveVRattle();
    this.nhcTempIntegrate();
  }
}
