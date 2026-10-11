/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/RIGID/fix_rigid_nh.cpp and
 * src/RIGID/fix_rigid_nh.h (class FixRigidNH: init, setup, initial_integrate, final_integrate, nhc_temp_integrate,
 * nhc_press_integrate, nh_epsilon_dot, couple, remap, compute_temp_target, compute_press_target, compute_scalar),
 * src/RIGID/fix_rigid_nvt.cpp, src/RIGID/fix_rigid_npt.cpp and src/RIGID/fix_rigid_nph.cpp (the constructor checks
 * and the computes each style creates), and the /small compute_scalar of src/RIGID/fix_rigid_nh_small.cpp.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Contributing author of the source: Tony Sheh (U Michigan), Trung Dac Nguyen (U Michigan); the /small energy of
 * compute_scalar is from src/RIGID/fix_rigid_nh_small.cpp (Trung Dac Nguyen, ORNL).
 * references: Kamberaj et al., J. Chem. Phys. 122, 224114 (2005); Miller et al., J Chem Phys. 116, 8649-8659 (2002).
 *
 * Supported: fix rigid/nvt (keyword temp, tparam), rigid/npt (temp, tparam, iso, aniso, x, y, z, couple, dilate,
 * pchain) and rigid/nph (iso, aniso, x, y, z, couple, dilate, pchain), and the /small variants (molecule bodies only,
 * no force and torque flags), in 3d with point particles, bodystyle single, molecule and group, and the force and
 * torque flags for the plain styles. Not ported (StyleError): 2d, infile, mol, langevin, gravity, bodystyle custom,
 * fix_modify bodyforces early.
 */
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { ramp, rampFraction, ownCompute, removeCompute } from '../../engine/fix/util';
import { UNIT_SYSTEMS } from '../../engine/units';
import { FixRigidBase, EPSILON } from './fix_rigid';
import { angmomToOmega, invquatvec, matvec, noSquishRotate, qToExyz, quatvec, transposeMatvec, type Q4, type V3 } from './math_extra';

/** FixRigidNH::maclaurin_series: 1 + x^2/6 + x^4/120 + x^6/5040 + x^8/362880 */
const maclaurinSeries = (x: number): number => {
  const x2 = x * x;
  const x4 = x2 * x2;
  return 1.0 + (1.0 / 6.0) * x2 + (1.0 / 120.0) * x4 + (1.0 / 5040.0) * x2 * x4 + (1.0 / 362880.0) * x4 * x4;
};

type Pcouple = 'none' | 'xyz' | 'xy' | 'yz' | 'xz';

export class FixRigidNH extends FixRigidBase {
  protected boltz = 0;
  protected mvv2e = 0;
  protected nktv2p = 0;
  protected nfT = 0;
  protected nfR = 0;
  protected gF = 0;
  protected w: number[] = [];
  protected wdti1: number[] = [];
  protected wdti2: number[] = [];
  protected wdti4: number[] = [];
  protected qT: number[] = [];
  protected qR: number[] = [];
  protected etaT: number[] = [];
  protected etaR: number[] = [];
  protected etaDotT: number[] = [];
  protected etaDotR: number[] = [];
  protected fEtaT: number[] = [];
  protected fEtaR: number[] = [];
  protected akinT = 0;
  protected akinR = 0;
  protected tTarget = 0;
  protected tFreq = 0;
  /** barostat: frequencies, masses, positions, velocities and forces of the barostat variables */
  protected pFreq: number[] = [0, 0, 0];
  protected pFreqMax = 0;
  protected pdim = 0;
  protected pstyleIso = false;
  protected epsilonMass: number[] = [0, 0, 0];
  protected epsilon: number[] = [0, 0, 0];
  protected epsilonDot: number[] = [0, 0, 0];
  protected qB: number[] = [];
  protected etaB: number[] = [];
  protected etaDotB: number[] = [];
  protected fEtaB: number[] = [];
  protected pCurrent: number[] = [0, 0, 0];
  protected pTarget: number[] = [0, 0, 0];
  protected pHydro = 0;
  protected mtkTerm1 = 0;
  protected mtkTerm2 = 0;
  protected vol0 = 0;
  protected t0 = 1.0;
  protected dilateBit = 0;
  /** the temperature and pressure computes this fix reads (fix_modify temp / press may name others) */
  protected tempId = '';
  protected pressId = '';
  /** true when the computes were created by this fix (rigid/npt, rigid/nph): unfix deletes them */
  protected computesOwned = false;
  protected tempCompute: Compute | null = null;
  protected pressCompute: Compute | null = null;

  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args, style);
    const th = this.thermo;
    const pr = this.press;
    const dim = sys.dimension;
    if (dim !== 3) throw new StyleError(`fix ${style} is only supported in 3d by the browser engine`);
    this.dilateBit = pr.allremap ? sys.groups.bit('all') : sys.groups.bit(pr.dilate ?? 'all');
    // error checks of FixRigidNH (fix_rigid_nh.cpp)
    for (let i = 0; i < 3; i++) {
      if (pr.pFlag[i] && pr.pPeriod[i] <= 0.0) throw new StyleError(`Fix ${style} period must be > 0.0`);
    }
    const pc = pr.pcouple;
    const pf = pr.pFlag;
    if (pc === 'xyz' && (pf[0] === 0 || pf[1] === 0 || pf[2] === 0)) throw new StyleError(`Invalid fix ${style} command pressure settings`);
    if (pc === 'xy' && (pf[0] === 0 || pf[1] === 0)) throw new StyleError(`Invalid fix ${style} command pressure settings`);
    if (pc === 'yz' && (pf[1] === 0 || pf[2] === 0)) throw new StyleError(`Invalid fix ${style} command pressure settings`);
    if (pc === 'xz' && (pf[0] === 0 || pf[2] === 0)) throw new StyleError(`Invalid fix ${style} command pressure settings`);
    const box = sys.state.box;
    for (let i = 0; i < 3; i++) if (pf[i] && !box.periodic[i]) throw new StyleError(`Cannot use fix ${style} on a non-periodic dimension`);
    const same = (a: number[]) => a[0] === a[1] && a[0] === a[2];
    if (pc === 'xyz' && (!same(pr.pStart) || !same(pr.pStop) || !same(pr.pPeriod))) throw new StyleError(`Invalid fix ${style} command pressure settings`);
    if (pc === 'xy' && (pr.pStart[0] !== pr.pStart[1] || pr.pStop[0] !== pr.pStop[1] || pr.pPeriod[0] !== pr.pPeriod[1])) throw new StyleError(`Invalid fix ${style} command pressure settings`);
    if (pc === 'yz' && (pr.pStart[1] !== pr.pStart[2] || pr.pStop[1] !== pr.pStop[2] || pr.pPeriod[1] !== pr.pPeriod[2])) throw new StyleError(`Invalid fix ${style} command pressure settings`);
    if (pc === 'xz' && (pr.pStart[0] !== pr.pStart[2] || pr.pStop[0] !== pr.pStop[2] || pr.pPeriod[0] !== pr.pPeriod[2])) throw new StyleError(`Invalid fix ${style} command pressure settings`);
    const pDamped = (pf[0] && pr.pPeriod[0] <= 0.0) || (pf[1] && pr.pPeriod[1] <= 0.0) || (pf[2] && pr.pPeriod[2] <= 0.0);
    if ((th.tstat && th.tPeriod <= 0.0) || (pr.pstat && pDamped)) {
      throw new StyleError(`Fix ${style} damping parameters must be > 0.0`);
    }
    this.pstyleIso = pc === 'xyz';
    this.pdim = pf[0] + pf[1] + pf[2];
    for (let i = 0; i < 3; i++) this.pFreq[i] = pf[i] ? 1.0 / pr.pPeriod[i] : 0.0;
    const T = th.tChain, P = Math.max(1, pr.pChain);
    this.qT = new Array(T).fill(0);
    this.qR = new Array(T).fill(0);
    this.etaT = new Array(T).fill(0);
    this.etaR = new Array(T).fill(0);
    this.etaDotT = new Array(T).fill(0);
    this.etaDotR = new Array(T).fill(0);
    this.fEtaT = new Array(T).fill(0);
    this.fEtaR = new Array(T).fill(0);
    this.w = new Array(th.tOrder).fill(0);
    this.wdti1 = new Array(th.tOrder).fill(0);
    this.wdti2 = new Array(th.tOrder).fill(0);
    this.wdti4 = new Array(th.tOrder).fill(0);
    this.qB = new Array(P).fill(0);
    this.etaB = new Array(P).fill(0);
    this.etaDotB = new Array(P).fill(0);
    this.fEtaB = new Array(P).fill(0);
    if (th.tstat) {
      this.tFreq = 1.0 / th.tPeriod;
    }
    this.extscalar = 1;
  }

  /** The thermostat settings of the style. */
  protected get tstart(): number { return this.thermo.tStart; }
  protected get tstop(): number { return this.thermo.tStop; }

  /** Pressure-coupled style: barostat settings (may be empty for rigid/nvt). */
  protected get pstat(): boolean { return this.press.pstat; }

  /** FixRigidNH::init: the step factors, the degrees of freedom, the weights and the pressure computes. */
  init(): void {
    super.init();
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    this.dtq = 0.5 * s.dt;
    this.boltz = s.units.boltz;
    this.nktv2p = s.units.nktv2p;
    this.mvv2e = s.units.mvv2e;
    const dimension = this.sys.dimension;
    this.nfT = dimension * this.nbody;
    this.nfR = dimension * this.nbody;
    for (const b of this.bodies) for (let k = 0; k < dimension; k++) if (Math.abs(b.inertia[k]) < EPSILON) this.nfR--;
    this.gF = this.nfT + this.nfR;
    // Table 1 in Kamberaj et al
    const order = this.thermo.tOrder;
    if (order === 3) {
      this.w[0] = 1.0 / (2.0 - Math.pow(2.0, 1.0 / 3.0));
      this.w[1] = 1.0 - 2.0 * this.w[0];
      this.w[2] = this.w[0];
    } else if (order === 5) {
      this.w[0] = 1.0 / (4.0 - Math.pow(4.0, 1.0 / 3.0));
      this.w[1] = this.w[0];
      this.w[2] = 1.0 - 4.0 * this.w[0];
      this.w[3] = this.w[0];
      this.w[4] = this.w[0];
    }
    if (this.pstat) {
      this.pFreqMax = Math.max(this.pFreq[0], this.pFreq[1]);
      this.pFreqMax = Math.max(this.pFreqMax, this.pFreq[2]);
      if (this.vol0 === 0.0) this.vol0 = this.sys.geom.volume(3);
      this.tempCompute = this.sys.compute(this.tempId);
      this.pressCompute = this.sys.compute(this.pressId);
    }
  }

  /** FixRigidNH::compute_temp_target: linear ramp of the target temperature over the run. */
  protected computeTempTarget(): void {
    this.tTarget = ramp(this.sys, this.tstart, this.tstop);
  }

  /** FixRigidNH::compute_press_target: ramped target pressures and their mean over the barostatted dimensions. */
  protected computePressTarget(): void {
    const delta = rampFraction(this.sys);
    this.pHydro = 0.0;
    for (let i = 0; i < 3; i++) {
      if (this.press.pFlag[i]) {
        this.pTarget[i] = this.press.pStart[i] + delta * (this.press.pStop[i] - this.press.pStart[i]);
        this.pHydro += this.pTarget[i];
      }
    }
    this.pHydro /= this.pdim;
  }

  /** FixRigidNH::couple: the current pressure per dimension from the pressure compute, with the coupling applied. */
  protected couple(): void {
    this.sys.refreshComputes();
    const pc = this.pressCompute!;
    const t = this.pstyleIso ? [0, 0, 0] : Array.from(pc.vectorValues().slice(0, 6));
    const pcouple: Pcouple = this.press.pcouple;
    if (this.pstyleIso) {
      const sc = pc.scalarValue();
      this.pCurrent[0] = this.pCurrent[1] = this.pCurrent[2] = sc;
    } else if (pcouple === 'xyz') {
      const ave = (1.0 / 3.0) * (t[0] + t[1] + t[2]);
      this.pCurrent[0] = this.pCurrent[1] = this.pCurrent[2] = ave;
    } else if (pcouple === 'xy') {
      const ave = 0.5 * (t[0] + t[1]);
      this.pCurrent[0] = this.pCurrent[1] = ave;
      this.pCurrent[2] = t[2];
    } else if (pcouple === 'yz') {
      const ave = 0.5 * (t[1] + t[2]);
      this.pCurrent[1] = this.pCurrent[2] = ave;
      this.pCurrent[0] = t[0];
    } else if (pcouple === 'xz') {
      const ave = 0.5 * (t[0] + t[2]);
      this.pCurrent[0] = this.pCurrent[2] = ave;
      this.pCurrent[1] = t[1];
    } else {
      this.pCurrent[0] = t[0];
      this.pCurrent[1] = t[1];
      this.pCurrent[2] = t[2];
    }
  }

  /** Temperature and pressure computes of this step (ISO reads the scalar, the others the tensor). */
  private evaluatePressure(): void {
    this.sys.refreshComputes();
    if (this.pstyleIso) {
      this.tempCompute!.scalarValue();
      this.pressCompute!.scalarValue();
    } else {
      this.tempCompute!.vectorValues();
      this.pressCompute!.vectorValues();
    }
    this.couple();
  }

  /** FixRigidNH::setup: conjugate quaternion momenta, the thermostat and barostat masses and the step coefficients. */
  setup(): void {
    super.setup();
    this.akinT = 0.0;
    this.akinR = 0.0;
    const mbody: V3 = [0, 0, 0];
    for (const b of this.bodies) {
      transposeMatvec(b.ex, b.ey, b.ez, b.angmom, mbody);
      const q: Q4 = [0, 0, 0, 0];
      quatvec(b.quat, mbody, q);
      b.conjqm = q.map((v) => v * 2.0);
      if (this.thermo.tstat || this.pstat) {
        this.akinT += b.mass * (b.vcm[0] * b.vcm[0] + b.vcm[1] * b.vcm[1] + b.vcm[2] * b.vcm[2]);
        this.akinR += b.angmom[0] * b.omega[0] + b.angmom[1] * b.omega[1] + b.angmom[2] * b.omega[2];
      }
    }
    if (this.thermo.tstat) this.computeTempTarget();
    else if (this.pstat) {
      this.sys.refreshComputes();
      this.t0 = this.tempCompute!.scalarValue();
      if (this.t0 === 0.0) this.t0 = this.sys.state.units === UNIT_SYSTEMS.lj ? 1.0 : 300.0;
      this.tTarget = this.t0;
    }
    if (this.pstat) this.evaluatePressure();
    const kt = this.boltz * this.tTarget;
    if (this.thermo.tstat) {
      const tMass = kt / (this.tFreq * this.tFreq);
      this.qT[0] = this.nfT * tMass;
      this.qR[0] = this.nfR * tMass;
      const T = this.thermo.tChain;
      for (let i = 1; i < T; i++) this.qT[i] = this.qR[i] = tMass;
      for (let i = 1; i < T; i++) {
        this.fEtaT[i] = (this.qT[i - 1] * this.etaDotT[i - 1] * this.etaDotT[i - 1] - kt) / this.qT[i];
        this.fEtaR[i] = (this.qR[i - 1] * this.etaDotR[i - 1] * this.etaDotR[i - 1] - kt) / this.qR[i];
      }
    }
    if (this.pstat) {
      const dimension = 3;
      for (let i = 0; i < 3; i++) {
        if (this.press.pFlag[i]) {
          this.epsilonMass[i] = ((this.gF + dimension) * kt) / (this.pFreq[i] * this.pFreq[i]);
          this.epsilon[i] = Math.log(this.vol0) / dimension;
        }
      }
      const tbMass = kt / (this.pFreqMax * this.pFreqMax);
      this.qB[0] = dimension * dimension * tbMass;
      for (let i = 1; i < this.press.pChain; i++) {
        this.qB[i] = tbMass;
        this.fEtaB[i] = (this.qB[i] * this.etaDotB[i - 1] * this.etaDotB[i - 1] - kt) / this.qB[i];
      }
    }
    this.updateStepCoefficients();
    if (this.pstat) {
      this.computePressTarget();
      this.nhEpsilonDot();
    }
  }

  /** wdti1 = w dt / t_iter and its halves and quarters. */
  protected updateStepCoefficients(): void {
    const order = this.thermo.tOrder;
    for (let i = 0; i < order; i++) {
      this.wdti1[i] = (this.w[i] * this.dtv) / this.thermo.tIter;
      this.wdti2[i] = this.wdti1[i] / 2.0;
      this.wdti4[i] = this.wdti1[i] / 4.0;
    }
  }

  /** FixRigidNH::initial_integrate: half kicks and the no_squish rotation of each body, the chains, the box and the positions. */
  initialIntegrate(): void {
    const dtv = this.dtv, dtf = this.dtf, dtq = this.dtq;
    const dtf2 = dtf * 2.0;
    const tstat = this.thermo.tstat, pstat = this.pstat;
    const scaleT = [1.0, 1.0, 1.0];
    const scaleV = [1.0, 1.0, 1.0];
    let scaleR = 1.0;
    if (tstat) {
      this.akinT = 0.0;
      this.akinR = 0.0;
      const tmp = Math.exp(-dtq * this.etaDotT[0]);
      scaleT[0] = scaleT[1] = scaleT[2] = tmp;
      scaleR = Math.exp(-dtq * this.etaDotR[0]);
    }
    if (pstat) {
      this.akinT = 0.0;
      this.akinR = 0.0;
      for (let i = 0; i < 3; i++) scaleT[i] *= Math.exp(-dtq * (this.epsilonDot[i] + this.mtkTerm2));
      scaleR *= Math.exp(-dtq * (this.pdim * this.mtkTerm2));
      for (let i = 0; i < 3; i++) {
        const tmp = dtq * this.epsilonDot[i];
        scaleV[i] = dtv * Math.exp(tmp) * maclaurinSeries(tmp);
      }
    }
    const mbody: V3 = [0, 0, 0], tbody: V3 = [0, 0, 0];
    for (const b of this.bodies) {
      // step 1.1: vcm by a half step, then the scaling
      const dtfm = dtf / b.mass;
      for (let d = 0; d < 3; d++) b.vcm[d] += dtfm * b.fcm[d] * b.fflag[d];
      if (tstat || pstat) for (let d = 0; d < 3; d++) b.vcm[d] *= scaleT[d];
      if (tstat || pstat) this.akinT += b.mass * (b.vcm[0] * b.vcm[0] + b.vcm[1] * b.vcm[1] + b.vcm[2] * b.vcm[2]);
      // step 1.2: xcm by a full step
      for (let d = 0; d < 3; d++) b.xcm[d] += pstat ? scaleV[d] * b.vcm[d] : dtv * b.vcm[d];
      // step 1.3: torque (body frame) to the quaternion momentum
      for (let d = 0; d < 3; d++) b.torque[d] *= b.tflag[d];
      transposeMatvec(b.ex, b.ey, b.ez, b.torque, tbody);
      const fquat: Q4 = [0, 0, 0, 0];
      quatvec(b.quat, tbody, fquat);
      for (let k = 0; k < 4; k++) b.conjqm[k] += dtf2 * fquat[k];
      if (tstat || pstat) for (let k = 0; k < 4; k++) b.conjqm[k] *= scaleR;
      // steps 1.4 to 1.13: no_squish rotations
      noSquishRotate(3, b.conjqm, b.quat, b.inertia, dtq);
      noSquishRotate(2, b.conjqm, b.quat, b.inertia, dtq);
      noSquishRotate(1, b.conjqm, b.quat, b.inertia, dtv);
      noSquishRotate(2, b.conjqm, b.quat, b.inertia, dtq);
      noSquishRotate(3, b.conjqm, b.quat, b.inertia, dtq);
      const e = qToExyz(b.quat);
      b.ex = e.ex;
      b.ey = e.ey;
      b.ez = e.ez;
      invquatvec(b.quat, b.conjqm, mbody);
      matvec(b.ex, b.ey, b.ez, mbody, b.angmom);
      for (let d = 0; d < 3; d++) b.angmom[d] *= 0.5;
      angmomToOmega(b.angmom, b.ex, b.ey, b.ez, b.inertia, b.omega);
      if (tstat || pstat) this.akinR += b.angmom[0] * b.omega[0] + b.angmom[1] * b.omega[1] + b.angmom[2] * b.omega[2];
    }
    if (tstat) {
      this.computeTempTarget();
      this.nhcTempIntegrate();
    }
    if (pstat) this.nhcPressIntegrate();
    this.virial.fill(0);
    if (pstat) this.remap();
    this.setXV(true, 0.5);
    if (pstat) this.remap();
  }

  /** FixRigidNH::final_integrate: the second half kicks, the momenta and the velocities of the atoms, then the barostat. */
  finalIntegrate(): void {
    const dtf = this.dtf, dtq = this.dtq;
    const dtf2 = dtf * 2.0;
    const tstat = this.thermo.tstat, pstat = this.pstat;
    const scaleT = [1.0, 1.0, 1.0];
    let scaleR = 1.0;
    if (tstat) {
      const tmp = Math.exp(-1.0 * dtq * this.etaDotT[0]);
      scaleT[0] = scaleT[1] = scaleT[2] = tmp;
      scaleR = Math.exp(-1.0 * dtq * this.etaDotR[0]);
    }
    if (pstat) {
      for (let i = 0; i < 3; i++) scaleT[i] *= Math.exp(-dtq * (this.epsilonDot[i] + this.mtkTerm2));
      scaleR *= Math.exp(-dtq * (this.pdim * this.mtkTerm2));
      this.akinT = 0.0;
      this.akinR = 0.0;
    }
    if (!this.bodyForcesEarly) this.computeForcesAndTorques();
    const mbody: V3 = [0, 0, 0], tbody: V3 = [0, 0, 0];
    for (const b of this.bodies) {
      const dtfm = dtf / b.mass;
      if (tstat || pstat) for (let d = 0; d < 3; d++) b.vcm[d] *= scaleT[d];
      for (let d = 0; d < 3; d++) b.vcm[d] += dtfm * b.fcm[d] * b.fflag[d];
      if (pstat) this.akinT += b.mass * (b.vcm[0] * b.vcm[0] + b.vcm[1] * b.vcm[1] + b.vcm[2] * b.vcm[2]);
      for (let d = 0; d < 3; d++) b.torque[d] *= b.tflag[d];
      transposeMatvec(b.ex, b.ey, b.ez, b.torque, tbody);
      const fquat: Q4 = [0, 0, 0, 0];
      quatvec(b.quat, tbody, fquat);
      if (tstat || pstat) for (let k = 0; k < 4; k++) b.conjqm[k] = scaleR * b.conjqm[k] + dtf2 * fquat[k];
      else for (let k = 0; k < 4; k++) b.conjqm[k] += dtf2 * fquat[k];
      invquatvec(b.quat, b.conjqm, mbody);
      matvec(b.ex, b.ey, b.ez, mbody, b.angmom);
      for (let d = 0; d < 3; d++) b.angmom[d] *= 0.5;
      angmomToOmega(b.angmom, b.ex, b.ey, b.ez, b.inertia, b.omega);
      if (pstat) this.akinR += b.angmom[0] * b.omega[0] + b.angmom[1] * b.omega[1] + b.angmom[2] * b.omega[2];
    }
    this.setXV(false, 0.5);
    if (pstat) {
      this.evaluatePressure();
      this.computePressTarget();
      this.nhEpsilonDot();
    }
  }

  /** Bodies with the early bodyforces setting sum their forces in post_force; not ported, so always late here. */
  protected bodyForcesEarly = false;

  /**
   * FixRigidNH::remap: the box and the dilated atoms and bodies over half a step (the source's remap()). The box
   * edges scale about their centre by exp(dtq epsilon_dot); atoms and centres of mass are mapped through fractional
   * coordinates before and after the change.
   */
  protected remap(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const b = s.box;
    for (let i = 0; i < 3; i++) this.epsilon[i] += this.dtq * this.epsilonDot[i];
    const tmp = [0, 0, 0];
    const lamAtoms = new Map<number, number[]>();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.dilateBit)) continue;
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      lamAtoms.set(i, [tmp[0], tmp[1], tmp[2]]);
    }
    for (const body of this.bodies) {
      g.toLamda(body.xcm[0], body.xcm[1], body.xcm[2], tmp);
      body.xcm = [tmp[0], tmp[1], tmp[2]];
    }
    for (let i = 0; i < 3; i++) {
      if (!this.press.pFlag[i]) continue;
      const oldlo = b.lo[i], oldhi = b.hi[i];
      const ctr = 0.5 * (oldlo + oldhi);
      const expfac = Math.exp(this.dtq * this.epsilonDot[i]);
      b.lo[i] = (oldlo - ctr) * expfac + ctr;
      b.hi[i] = (oldhi - ctr) * expfac + ctr;
    }
    b.minLo = [...b.lo];
    b.minHi = [...b.hi];
    g.update();
    for (const [i, l] of lamAtoms) {
      g.fromLamda(l[0], l[1], l[2], tmp);
      s.x[3 * i] = tmp[0];
      s.x[3 * i + 1] = tmp[1];
      s.x[3 * i + 2] = tmp[2];
    }
    for (const body of this.bodies) {
      g.fromLamda(body.xcm[0], body.xcm[1], body.xcm[2], tmp);
      body.xcm = [tmp[0], tmp[1], tmp[2]];
    }
  }

  /**
   * FixRigidNH::nhc_temp_integrate: the Yoshida-Suzuki update of the translational and rotational chains
   * (Kamberaj et al., update_nhcp). Mirrors the source's loop order.
   */
  protected nhcTempIntegrate(): void {
    if (this.gF === 0) return;
    const T = this.thermo.tChain;
    const order = this.thermo.tOrder;
    const kt = this.boltz * this.tTarget;
    const gfktT = this.nfT * kt;
    const gfktR = this.nfR * kt;
    const tMass = (this.boltz * this.tTarget) / (this.tFreq * this.tFreq);
    this.qT[0] = this.nfT * tMass;
    this.qR[0] = this.nfR * tMass;
    for (let i = 1; i < T; i++) this.qT[i] = this.qR[i] = tMass;
    this.fEtaT[0] = (this.akinT * this.mvv2e - gfktT) / this.qT[0];
    this.fEtaR[0] = (this.akinR * this.mvv2e - gfktR) / this.qR[0];
    const eT = this.etaDotT, eR = this.etaDotR, fT = this.fEtaT, fR = this.fEtaR;
    const qT = this.qT, qR = this.qR;
    for (let i = 0; i < this.thermo.tIter; i++) {
      for (let j = 0; j < order; j++) {
        eT[T - 1] += this.wdti2[j] * fT[T - 1];
        eR[T - 1] += this.wdti2[j] * fR[T - 1];
        for (let k = 1; k < T; k++) {
          let tmp = this.wdti4[j] * eT[T - k];
          let ms = maclaurinSeries(tmp);
          let sc = Math.exp(-1.0 * tmp);
          let s2 = sc * sc;
          eT[T - k - 1] = eT[T - k - 1] * s2 + this.wdti2[j] * fT[T - k - 1] * sc * ms;
          tmp = this.wdti4[j] * eR[T - k];
          ms = maclaurinSeries(tmp);
          sc = Math.exp(-1.0 * tmp);
          s2 = sc * sc;
          eR[T - k - 1] = eR[T - k - 1] * s2 + this.wdti2[j] * fR[T - k - 1] * sc * ms;
        }
        for (let k = 0; k < T; k++) {
          this.etaT[k] += this.wdti1[j] * eT[k];
          this.etaR[k] += this.wdti1[j] * eR[k];
        }
        for (let k = 1; k < T; k++) {
          fT[k] = qT[k - 1] * eT[k - 1] * eT[k - 1] - kt;
          fT[k] /= qT[k];
          fR[k] = qR[k - 1] * eR[k - 1] * eR[k - 1] - kt;
          fR[k] /= qR[k];
        }
        for (let k = 0; k < T - 1; k++) {
          let tmp = this.wdti4[j] * eT[k + 1];
          let ms = maclaurinSeries(tmp);
          let sc = Math.exp(-1.0 * tmp);
          let s2 = sc * sc;
          eT[k] = eT[k] * s2 + this.wdti2[j] * fT[k] * sc * ms;
          tmp = qT[k] * eT[k] * eT[k] - kt;
          fT[k + 1] = tmp / qT[k + 1];
          tmp = this.wdti4[j] * eR[k + 1];
          ms = maclaurinSeries(tmp);
          sc = Math.exp(-1.0 * tmp);
          s2 = sc * sc;
          eR[k] = eR[k] * s2 + this.wdti2[j] * fR[k] * sc * ms;
          tmp = qR[k] * eR[k] * eR[k] - kt;
          fR[k + 1] = tmp / qR[k + 1];
        }
        eT[T - 1] += this.wdti2[j] * fT[T - 1];
        eR[T - 1] += this.wdti2[j] * fR[T - 1];
      }
    }
  }

  /** FixRigidNH::nhc_press_integrate: the barostat chain (update_nhcb in Kamberaj et al.). */
  protected nhcPressIntegrate(): void {
    const P = this.press.pChain;
    const order = this.thermo.tOrder;
    const kt = this.boltz * this.tTarget;
    const lktPress = kt;
    const dimension = 3;
    const tbMass = kt / (this.pFreqMax * this.pFreqMax);
    this.qB[0] = dimension * dimension * tbMass;
    for (let i = 1; i < P; i++) {
      this.qB[i] = tbMass;
      this.fEtaB[i] = this.qB[i - 1] * this.etaDotB[i - 1] * this.etaDotB[i - 1] - kt;
      this.fEtaB[i] /= this.qB[i];
    }
    let kecurrent = 0.0;
    for (let i = 0; i < 3; i++) {
      if (this.press.pFlag[i]) {
        this.epsilonMass[i] = ((this.gF + dimension) * kt) / (this.pFreq[i] * this.pFreq[i]);
        kecurrent += this.epsilonMass[i] * this.epsilonDot[i] * this.epsilonDot[i];
      }
    }
    kecurrent /= this.pdim;
    this.fEtaB[0] = (kecurrent - lktPress) / this.qB[0];
    const eB = this.etaDotB, fB = this.fEtaB, qB = this.qB;
    for (let i = 0; i < this.thermo.tIter; i++) {
      for (let j = 0; j < order; j++) {
        eB[P - 1] += this.wdti2[j] * fB[P - 1];
        for (let k = 1; k < P; k++) {
          const tmp = this.wdti4[j] * eB[P - k];
          const ms = maclaurinSeries(tmp);
          const sc = Math.exp(-0.5 * tmp);
          const s2 = sc * sc;
          eB[P - k - 1] = eB[P - k - 1] * s2 + this.wdti2[j] * fB[P - k - 1] * sc * ms;
        }
        for (let k = 0; k < P; k++) this.etaB[k] += this.wdti1[j] * eB[k];
        for (let k = 1; k < P; k++) {
          fB[k] = qB[k - 1] * eB[k - 1] * eB[k - 1] - kt;
          fB[k] /= qB[k];
        }
        for (let k = 0; k < P - 1; k++) {
          const tmp = this.wdti4[j] * eB[k + 1];
          const ms = maclaurinSeries(tmp);
          const sc = Math.exp(-0.5 * tmp);
          const s2 = sc * sc;
          eB[k] = eB[k] * s2 + this.wdti2[j] * fB[k] * sc * ms;
          const t2 = qB[k] * eB[k] * eB[k] - kt;
          fB[k + 1] = t2 / qB[k + 1];
        }
        eB[P - 1] += this.wdti2[j] * fB[P - 1];
      }
    }
  }

  /** FixRigidNH::nh_epsilon_dot: the barostat velocities over half a step, then the mean kinetic term. */
  protected nhEpsilonDot(): void {
    if (this.gF === 0) return;
    const volume = this.sys.geom.volume(3);
    this.mtkTerm1 = ((this.akinT + this.akinR) * this.mvv2e) / this.gF;
    const scale = Math.exp(-1.0 * this.dtq * this.etaDotB[0]);
    for (let i = 0; i < 3; i++) {
      if (!this.press.pFlag[i]) continue;
      let fEpsilon = ((this.pCurrent[i] - this.pHydro) * volume) / this.nktv2p + this.mtkTerm1;
      fEpsilon /= this.epsilonMass[i];
      this.epsilonDot[i] += this.dtq * fEpsilon;
      this.epsilonDot[i] *= scale;
    }
    this.mtkTerm2 = 0.0;
    for (let i = 0; i < 3; i++) if (this.press.pFlag[i]) this.mtkTerm2 += this.epsilonDot[i];
    this.mtkTerm2 /= this.gF;
  }

  /**
   * FixRigidNH::compute_scalar: the rigid-body temperature plus the energy of the extended variables (thermostat
   * chains, barostat, and the box work p0 V).
   */
  computeScalar(): number {
    const kt = this.boltz * this.tTarget;
    let energy = this.rigidEnergy();
    if (this.thermo.tstat) {
      const T = this.thermo.tChain;
      energy += kt * (this.nfT * this.etaT[0] + this.nfR * this.etaR[0]);
      for (let i = 1; i < T; i++) energy += kt * (this.etaT[i] + this.etaR[i]);
      for (let i = 0; i < T; i++) {
        energy += 0.5 * this.qT[i] * (this.etaDotT[i] * this.etaDotT[i]);
        energy += 0.5 * this.qR[i] * (this.etaDotR[i] * this.etaDotR[i]);
      }
    }
    if (this.pstat) {
      let e = 0.0;
      for (let i = 0; i < 3; i++) if (this.press.pFlag[i]) e += this.epsilonMass[i] * this.epsilonDot[i] * this.epsilonDot[i];
      energy += e * (0.5 / this.pdim);
      const vol = this.sys.geom.volume(3);
      const p0 = (this.pTarget[0] + this.pTarget[1] + this.pTarget[2]) / 3.0;
      energy += (p0 * vol) / this.nktv2p;
      for (let i = 0; i < this.press.pChain; i++) {
        energy += kt * this.etaB[i];
        energy += 0.5 * this.qB[i] * (this.etaDotB[i] * this.etaDotB[i]);
      }
    }
    return energy;
  }

  /**
   * The rigid-body part of compute_scalar. Plain styles: FixRigid::compute_scalar, the temperature of the bodies. The
   * /small styles (fix_rigid_nh_small.cpp, compute_scalar, Kamberaj et al. eq. for H_NVE): the translational energy
   * sum m v_cm^2 and the rotational sum (conjqm . P_k q)^2 / (8 I_k) (zero for |I_k| < 1e-6), times mvv2e.
   */
  protected rigidEnergy(): number {
    if (!this.small) return super.computeScalar();
    let keT = 0.0, keQ = 0.0;
    for (const b of this.bodies) {
      keT += b.mass * (b.vcm[0] * b.vcm[0] + b.vcm[1] * b.vcm[1] + b.vcm[2] * b.vcm[2]);
      const q = b.quat, c = b.conjqm;
      for (let k = 1; k <= 3; k++) {
        let P: number[];
        if (k === 1) P = [-q[1], q[0], q[3], -q[2]];
        else if (k === 2) P = [-q[2], -q[3], q[0], q[1]];
        else P = [-q[3], q[2], -q[1], q[0]];
        let tmp = c[0] * P[0] + c[1] * P[1] + c[2] * P[2] + c[3] * P[3];
        tmp *= tmp;
        if (Math.abs(b.inertia[k - 1]) < 1e-6) tmp = 0.0;
        else tmp /= 8.0 * b.inertia[k - 1];
        keQ += tmp;
      }
    }
    return (keT + keQ) * this.mvv2e;
  }

  /** The conserved-energy contribution of the reservoirs (thermo ecouple), as modify->energy_couple sums compute_scalar. */
  ecouple(): number {
    return this.computeScalar();
  }

  /** Names of the temperature and pressure computes (rigid/npt and rigid/nph create them as ID_temp and ID_press). */
  setComputes(tempId: string, pressId: string): void {
    this.tempId = tempId;
    this.pressId = pressId;
  }

  /** unfix: deletes the computes this fix created (not those fix_modify pointed it to). */
  destroy(): void {
    if (this.computesOwned) {
      removeCompute(this.sys, this.tempId);
      removeCompute(this.sys, this.pressId);
    }
  }
}

/** Creates the temperature and pressure computes of rigid/npt and rigid/nph (fix ID_temp and ID_press, group all). */
const createComputes = (f: FixRigidNH, sys: System): void => {
  f.setComputes(`${f.id}_temp`, `${f.id}_press`);
  ownCompute(sys, `${f.id}_temp`, 'all', 'temp', []);
  ownCompute(sys, `${f.id}_press`, 'all', 'pressure', [`${f.id}_temp`]);
};

/** FixRigidNVT (fix_rigid_nvt.cpp): the Nose-Hoover chain on the translational and rotational kinetic energies. */
export class FixRigidNVT extends FixRigidNH {
  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args, style);
    if (!this.thermo.tstat) throw new StyleError(`Did not set temperature for fix ${style}`);
    if (this.press.pstat) throw new StyleError(`fix ${style}: the pressure keywords (iso, aniso, x, y, z) are not ported; use rigid/npt`);
    if (this.thermo.tStart < 0.0 || this.thermo.tStop <= 0.0) throw new StyleError(`Target temperature for fix ${style} cannot be 0.0`);
    if (this.thermo.tPeriod <= 0.0) throw new StyleError(`Fix ${style} period must be > 0.0`);
    if (this.thermo.tChain < 1) throw new StyleError(`Illegal fix ${style} command`);
    if (this.thermo.tIter < 1) throw new StyleError(`Illegal fix ${style} command`);
    if (this.thermo.tOrder !== 3 && this.thermo.tOrder !== 5) throw new StyleError(`Fix ${style} temperature order must be 3 or 5`);
  }
}

/** FixRigidNPT (fix_rigid_npt.cpp): temperature and pressure both coupled; creates ID_temp and ID_press. */
export class FixRigidNPT extends FixRigidNH {
  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args, style);
    if (!this.thermo.tstat || !this.press.pstat) throw new StyleError(`Did not set temperature or pressure for fix ${style}`);
    if (this.thermo.tStart <= 0.0 || this.thermo.tStop <= 0.0) throw new StyleError(`Target temperature for fix ${style} cannot be 0.0`);
    if (this.thermo.tPeriod <= 0.0) throw new StyleError(`Fix ${style} period must be > 0.0`);
    if (this.thermo.tChain < 1) throw new StyleError(`Illegal fix ${style} command`);
    if (this.thermo.tIter < 1) throw new StyleError(`Illegal fix ${style} command`);
    if (this.thermo.tOrder !== 3 && this.thermo.tOrder !== 5) throw new StyleError(`Fix ${style} temperature order must be 3 or 5`);
    this.computesOwned = true;
    createComputes(this, sys);
  }
}

/** FixRigidNPH (fix_rigid_nph.cpp): the barostat only; the temperature is the one of the start. */
export class FixRigidNPH extends FixRigidNH {
  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args, style);
    if (!this.press.pstat) throw new StyleError(`Did not set pressure for fix ${style}`);
    if (this.thermo.tstat) throw new StyleError(`Cannot set temperature for fix ${style}`);
    this.computesOwned = true;
    createComputes(this, sys);
  }
}
