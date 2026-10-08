import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Compute } from '../compute/compute';
import { ownCompute, ramp, removeCompute } from './util';
import { massOf } from '../atoms';

/*
 * fix nvt / npt / nph — docs.lammps.org/fix_nh.html.
 *
 * "These commands perform time integration on Nose-Hoover style
 * non-Hamiltonian equations of motion ... The equations of motion used are
 * those of Shinoda et al in :ref:`(Shinoda) <nh-Shinoda>`, which combine the
 * hydrostatic equations of Martyna, Tobias and Klein in :ref:`(Martyna)
 * <nh-Martyna>` with the strain energy proposed by Parrinello and Rahman in
 * :ref:`(Parrinello) <nh-Parrinello>`. The time integration schemes closely
 * follow the time-reversible measure-preserving Verlet and rRESPA integrators
 * derived by Tuckerman et al in :ref:`(Tuckerman) <nh-Tuckerman>`." Written
 * here from those papers (MTK J Chem Phys 101,
 * 4177 (1994); Martyna et al. Mol Phys 87, 1117 (1996); Tuckerman et al. J
 * Phys A 39, 5629 (2006)), not from LAMMPS code. Per step:
 *   initial: thermostat chain (dt/2); barostat chain (dt/2); barostat
 *            velocities (dt/2); velocity scaling by the strain rate (dt/4);
 *            half kick; velocity scaling (dt/4); box and positions scaled
 *            (dt/2); drift; box (dt/2)
 *   final:   velocity scaling (dt/4); half kick; velocity scaling (dt/4);
 *            barostat velocities (dt/2); barostat chain; thermostat chain.
 * Masses: thermostat Q_1 = N_f k_B T Tdamp^2, Q_k = k_B T Tdamp^2 (k > 1);
 * barostat "W = (N + 1) k_B T_\mathrm{target} P_\mathrm{damp}^2";
 * barostat-thermostat chain
 * Q_1 = n_dims k_B T Pdamp^2, Q_k = k_B T Pdamp^2.
 * Keywords: temp, iso, aniso, tri, x, y, z, xy, yz, xz, couple, tchain,
 * pchain, mtk, tloop, ploop, nreset, drag, ptemp, dilate, scalexy/yz/xz,
 * flip, fixedpoint. Defaults "tchain = 3, pchain = 3, mtk = yes, tloop = 1,
 * ploop = 1, nreset = 0, drag = 0.0, dilate = all, couple = none, flip =
 * yes". "iso ... couple xyz"; "aniso ... couple none"; Using "tri Pstart
 * Pstop Pdamp" is the same as specifying these 7 keywords, one per line in the
 * docs: "x Pstart Pstop Pdamp", "y Pstart Pstop Pdamp", "z Pstart Pstop
 * Pdamp", "xy 0.0 0.0 Pdamp", "yz 0.0 0.0 Pdamp", "xz 0.0 0.0 Pdamp",
 * "couple none".
 * For fix nvt, the docs give the code line "compute fix-ID_temp group-ID
 * temp"; for fix npt and fix nph, the code lines "compute fix-ID_temp all
 * temp" and "compute fix-ID_press all pressure fix-ID_temp". "If a thermostat
 * is not defined, :math:`T_\mathrm{target}` is set to the current temperature
 * of the system when the barostat is initialized."
 * Global scalar: "The scalar is the same cumulative energy change due to this
 * fix described in the previous paragraph", reported by thermo ecouple.
 * Not implemented (StyleError): dilate with a partial group's strain-energy
 * reference cell (nreset > 0 is accepted; the reference cell is the current
 * one), box flips (flip yes stops with an error when a tilt passes 0.6 of the
 * box length), update dipole.
 */

type Dim = 0 | 1 | 2 | 3 | 4 | 5;   // x y z yz xz xy (Voigt order as LAMMPS documents for tensors)
const KEYS = ['x', 'y', 'z', 'yz', 'xz', 'xy'];

export class FixNH extends Fix {
  readonly style: string;
  // thermostat
  private tstat = false;
  private tStart = 0; private tStop = 0; private tPeriod = 0;
  private tchain = 3; private tloop = 1;
  private eta: Float64Array; private etaDot: Float64Array;
  // barostat
  private pstat = false;
  private pStart = new Float64Array(6); private pStop = new Float64Array(6); private pPeriod = new Float64Array(6);
  private pFlag = [false, false, false, false, false, false];
  private couple: 'none' | 'xyz' | 'xy' | 'yz' | 'xz' = 'none';
  private pchain = 3; private ploop = 1;
  private mtk = true;
  private drag = 0;
  private ptemp: number | null = null;
  private dilateBit: number;
  private fixedpoint: number[] | null = null;
  private omegaDot = new Float64Array(6);
  private omegaMass = new Float64Array(6);
  private etap: Float64Array; private etapDot: Float64Array;
  private pTarget = new Float64Array(6);
  private pCurrent = new Float64Array(6);
  private tTarget = 0;
  private tempCompute!: Compute;
  private pressCompute: Compute | null = null;
  private tempId: string;
  private pressId: string | null = null;
  private dthalf = 0; private dt4 = 0; private dt8 = 0; private dtf = 0; private dtv = 0;
  private vol0 = 0;
  private ecoupleCum = 0;
  private scaleTilt = [true, true, true];   // yz xz xy with their second dimension

  constructor(sys: System, id: string, group: string, args: string[], style: 'nvt' | 'npt' | 'nph') {
    super(sys, id, group, args);
    this.style = style;
    this.timeIntegrate = true;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.dilateBit = sys.groupBit('all');
    for (let k = 0; k < args.length;) {
      const key = args[k];
      const n = (j: number, what: string) => {
        const v = Number(args[k + j]);
        if (args[k + j] === undefined || !Number.isFinite(v)) throw new StyleError(`fix ${style} ${key}: expected a number for ${what}`);
        return v;
      };
      switch (key) {
        case 'temp':
          this.tstat = true;
          this.tStart = n(1, 'Tstart'); this.tStop = n(2, 'Tstop'); this.tPeriod = n(3, 'Tdamp');
          if (!(this.tPeriod > 0)) throw new StyleError(`fix ${style}: Tdamp must be > 0`);
          k += 4;
          break;
        case 'iso': case 'aniso': case 'tri': {
          const ps = n(1, 'Pstart'), pe = n(2, 'Pstop'), pd = n(3, 'Pdamp');
          this.couple = key === 'iso' ? 'xyz' : 'none';
          const dims = sys.dimension === 2 ? [0, 1] : [0, 1, 2];
          for (const d of dims) this.setP(d as Dim, ps, pe, pd);
          if (key === 'tri') {
            for (const d of (sys.dimension === 2 ? [5] : [3, 4, 5])) this.setP(d as Dim, 0, 0, pd);
          }
          k += 4;
          break;
        }
        case 'x': case 'y': case 'z': case 'xy': case 'yz': case 'xz': {
          const d = KEYS.indexOf(key) as Dim;
          if (sys.dimension === 2 && (key === 'z' || key === 'yz' || key === 'xz')) throw new StyleError(`fix ${style} ${key} is invalid for a 2d simulation`);
          this.setP(d, n(1, 'Pstart'), n(2, 'Pstop'), n(3, 'Pdamp'));
          k += 4;
          break;
        }
        case 'couple':
          if (!['none', 'xyz', 'xy', 'yz', 'xz'].includes(args[k + 1])) throw new StyleError(`fix ${style} couple must be none, xyz, xy, yz or xz`);
          this.couple = args[k + 1] as FixNH['couple'];
          k += 2;
          break;
        case 'tchain': this.tchain = Math.trunc(n(1, 'tchain')); if (this.tchain < 1) throw new StyleError('tchain must be >= 1'); k += 2; break;
        case 'pchain': this.pchain = Math.trunc(n(1, 'pchain')); if (this.pchain < 0) throw new StyleError('pchain must be >= 0'); k += 2; break;
        case 'tloop': this.tloop = Math.trunc(n(1, 'tloop')); k += 2; break;
        case 'ploop': this.ploop = Math.trunc(n(1, 'ploop')); k += 2; break;
        case 'mtk':
          if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError('mtk must be yes or no');
          this.mtk = args[k + 1] === 'yes';
          k += 2;
          break;
        case 'nreset': n(1, 'nreset'); k += 2; break;
        case 'drag': this.drag = n(1, 'drag'); if (this.drag < 0) throw new StyleError('drag must be >= 0'); k += 2; break;
        case 'ptemp': this.ptemp = n(1, 'ptemp'); k += 2; break;
        case 'dilate': this.dilateBit = sys.groupBit(args[k + 1] ?? ''); k += 2; break;
        case 'scalexy': case 'scaleyz': case 'scalexz': {
          if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError(`${key} must be yes or no`);
          const t = ['scaleyz', 'scalexz', 'scalexy'].indexOf(key);
          this.scaleTilt[t] = args[k + 1] === 'yes';
          k += 2;
          break;
        }
        case 'flip':
          if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError('flip must be yes or no');
          k += 2;
          break;
        case 'fixedpoint': this.fixedpoint = [n(1, 'x'), n(2, 'y'), n(3, 'z')]; k += 4; break;
        case 'update': throw new StyleError(`fix ${style} update ${args[k + 1] ?? ''} needs dipoles, which the browser engine does not support`);
        case 'disc': throw new StyleError(`fix ${style} disc needs finite-size particles`);
        default: throw new StyleError(`unknown fix ${style} keyword '${key}'`);
      }
    }
    if (style === 'nvt' && (!this.tstat || this.pstat)) throw new StyleError('fix nvt needs the temp keyword and no pressure keywords');
    if (style === 'npt' && (!this.tstat || !this.pstat)) throw new StyleError('fix npt needs both temp and pressure (iso, aniso, tri, x, ...) keywords');
    if (style === 'nph' && (this.tstat || !this.pstat)) throw new StyleError('fix nph needs pressure keywords and no temp keyword');
    if (this.pstat) {
      const s = sys.state;
      for (const d of [0, 1, 2]) if (this.pFlag[d] && !s.box.periodic[d]) throw new StyleError(`fix ${style}: cannot barostat a non-periodic dimension (${KEYS[d]})`);
      if ((this.pFlag[3] || this.pFlag[4] || this.pFlag[5]) && !s.box.triclinic) throw new StyleError(`fix ${style}: tilt keywords need a triclinic box`);
      // "Pstart, Pstop, Pdamp parameters for any coupled dimensions must be identical"
      const coupled = this.coupledDims();
      for (const d of coupled.slice(1)) {
        if (this.pStart[d] !== this.pStart[coupled[0]] || this.pStop[d] !== this.pStop[coupled[0]] || this.pPeriod[d] !== this.pPeriod[coupled[0]]) {
          throw new StyleError(`fix ${style}: coupled dimensions must have identical Pstart, Pstop and Pdamp`);
        }
      }
      // tilt scaling default: "yes if periodic in second dimension and not coupled to barostat"
      if (this.pFlag[3]) this.scaleTilt[0] = false;
      if (this.pFlag[4]) this.scaleTilt[1] = false;
      if (this.pFlag[5]) this.scaleTilt[2] = false;
    }
    this.eta = new Float64Array(this.tchain);
    this.etaDot = new Float64Array(this.tchain + 1);
    this.etap = new Float64Array(Math.max(1, this.pchain));
    this.etapDot = new Float64Array(Math.max(1, this.pchain) + 1);
    this.tempId = `${id}_temp`;
    this.tempCompute = ownCompute(sys, this.tempId, this.pstat ? 'all' : group, 'temp', []);
    if (this.pstat) {
      this.pressId = `${id}_press`;
      this.pressCompute = ownCompute(sys, this.pressId, 'all', 'pressure', [this.tempId]);
      this.virialGlobal = false;
    }
  }

  private setP(d: Dim, ps: number, pe: number, pd: number): void {
    if (!(pd > 0)) throw new StyleError(`fix ${this.style}: Pdamp must be > 0`);
    this.pstat = true;
    this.pFlag[d] = true;
    this.pStart[d] = ps; this.pStop[d] = pe; this.pPeriod[d] = pd;
  }

  private coupledDims(): number[] {
    switch (this.couple) {
      case 'xyz': return this.sys.dimension === 2 ? [0, 1] : [0, 1, 2];
      case 'xy': return [0, 1];
      case 'yz': return [1, 2];
      case 'xz': return [0, 2];
      default: return [];
    }
  }

  destroy(): void {
    removeCompute(this.sys, this.tempId);
    if (this.pressId) removeCompute(this.sys, this.pressId);
  }

  modify(key: string, values: string[]): number {
    if (key === 'temp') {
      const c = this.sys.compute(values[0] ?? '');
      if (!c.tempFlag) throw new StyleError(`fix_modify temp: compute ${values[0]} does not compute a temperature`);
      this.tempId = values[0];
      this.tempCompute = c;
      // the fix's pressure compute uses the new temperature too
      if (this.pressCompute) (this.pressCompute as unknown as { tempId: string }).tempId = values[0];
      return 1;
    }
    if (key === 'press') {
      if (!this.pstat) throw new StyleError(`fix_modify press: fix ${this.id} has no barostat`);
      const c = this.sys.compute(values[0] ?? '');
      if (!c.pressFlag) throw new StyleError(`fix_modify press: compute ${values[0]} does not compute a pressure`);
      this.pressId = values[0];
      this.pressCompute = c;
      return 1;
    }
    return super.modify(key, values);
  }

  init(): void {
    const s = this.sys.state;
    this.tempCompute = this.sys.compute(this.tempId);
    if (this.pressId) this.pressCompute = this.sys.compute(this.pressId);
    this.resetDt();
    this.vol0 = this.sys.geom.volume(s.dimension);
    // barostat target temperature
    this.tTarget = this.tstat ? ramp(this.sys, this.tStart, this.tStop) : this.currentTemp();
    if (this.pstat) {
      let tb = this.tstat ? this.tTarget : this.ptemp ?? this.currentTemp();
      if (!this.tstat && this.ptemp === null && !(tb > 0)) {
        throw new StyleError(`fix ${this.style}: the system temperature is 0; set ptemp to give the barostat a target temperature`);
      }
      if (!(tb > 0)) tb = this.ptemp ?? 1;
      const kt = s.units.boltz * tb;
      const nAtoms = s.n;
      for (let d = 0; d < 6; d++) {
        if (this.pFlag[d]) this.omegaMass[d] = (nAtoms + 1) * kt * this.pPeriod[d] * this.pPeriod[d];
      }
    }
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    this.dthalf = 0.5 * s.dt;
    this.dt4 = 0.25 * s.dt;
    this.dt8 = 0.125 * s.dt;
  }

  private currentTemp(): number {
    this.sys.refreshComputes();
    return this.tempCompute.scalarValue();
  }

  /** ke-energy-like 2*KE of the thermostatted dof: dof k_B T_current. */
  private dofKT(): { dof: number; kT: number } {
    const t = this.currentTemp();
    return { dof: this.tempCompute.dof, kT: this.sys.state.units.boltz * t };
  }

  // ------------------------------------------------------------------ hooks

  setup(): void {
    if (this.tstat) this.tTarget = ramp(this.sys, this.tStart, this.tStop);
    if (this.pstat) {
      this.computePressTarget();
      this.computePressCurrent();
    }
  }

  initialIntegrate(): void {
    if (this.tstat) {
      this.tTarget = ramp(this.sys, this.tStart, this.tStop);
      this.nhcTemp();
    }
    if (this.pstat) {
      this.computePressTarget();
      if (this.pchain > 0) this.nhcPress();
      this.computePressCurrent();
      this.omegaUpdate();
      this.velocityPress();
    }
    this.kick();
    if (this.pstat) {
      this.velocityPress();
      this.remap();
    }
    this.drift();
    if (this.pstat) this.remap();
  }

  finalIntegrate(): void {
    if (this.pstat) this.velocityPress();
    this.kick();
    if (this.pstat) {
      this.velocityPress();
      this.computePressCurrent();
      this.omegaUpdate();
      if (this.pchain > 0) this.nhcPress();
    }
    if (this.tstat) this.nhcTemp();
  }

  // ------------------------------------------------------------------ pieces

  private kick(): void {
    const s = this.sys.state;
    const { v, f, mask, type } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const c = this.dtf / massOf(s, i);
      v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
    }
  }

  private drift(): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      x[3 * i] += this.dtv * v[3 * i]; x[3 * i + 1] += this.dtv * v[3 * i + 1]; x[3 * i + 2] += this.dtv * v[3 * i + 2];
    }
  }

  /** Scales group velocities by `factor`, honouring a temperature bias. */
  private scaleVelocities(factor: number | number[]): void {
    const s = this.sys.state;
    const c = this.tempCompute;
    const bias = c.hasBias();
    if (bias) { c.computeBias(); c.removeBiasAll(); }
    const fx = typeof factor === 'number' ? factor : factor[0];
    const fy = typeof factor === 'number' ? factor : factor[1];
    const fz = typeof factor === 'number' ? factor : factor[2];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      s.v[3 * i] *= fx; s.v[3 * i + 1] *= fy; s.v[3 * i + 2] *= fz;
    }
    if (bias) c.restoreBiasAll();
  }

  /**
   * Thermostat chain over dt/2 (Martyna-Tuckerman-Klein / Frenkel-Smit E.2),
   * in tloop sub-steps: sweep the chain velocities top-down, scale particle
   * velocities by exp(-etaDot_1 dt), advance eta, sweep back up.
   */
  private nhcTemp(): void {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const M = this.tchain;
    const { dof, kT: kTcur } = this.dofKT();
    if (dof <= 0) return;
    const kt = kB * this.tTarget;
    if (!(kt > 0)) return;
    const tfreq = 1 / this.tPeriod;
    const q = new Float64Array(M);
    q[0] = dof * kt / (tfreq * tfreq);
    for (let k = 1; k < M; k++) q[k] = kt / (tfreq * tfreq);
    const nc = this.tloop;
    const dthalf = this.dthalf / nc, dt4 = this.dt4 / nc, dt8 = this.dt8 / nc;
    let ke2 = dof * kTcur;   // 2 * KE of the thermostatted dof
    const ed = this.etaDot;
    const drag = this.drag > 0 ? 1 - this.dtv * tfreq * this.drag / nc : 1;
    for (let loop = 0; loop < nc; loop++) {
      // top of chain down
      const g = (k: number) => (k === 0 ? (ke2 - dof * kt) / q[0] : (q[k - 1] * ed[k - 1] * ed[k - 1] - kt) / q[k]);
      ed[M - 1] += g(M - 1) * dt4;
      ed[M - 1] *= drag;
      for (let k = M - 2; k >= 0; k--) {
        const ex = Math.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex;
        ed[k] += g(k) * dt4;
        ed[k] *= drag;
        ed[k] *= ex;
      }
      // scale velocities
      const factor = Math.exp(-dthalf * ed[0]);
      this.scaleVelocities(factor);
      ke2 *= factor * factor;
      for (let k = 0; k < M; k++) this.eta[k] += dthalf * ed[k];
      // back up the chain
      for (let k = 0; k < M - 1; k++) {
        const ex = Math.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex;
        ed[k] += (k === 0 ? (ke2 - dof * kt) / q[0] : (q[k - 1] * ed[k - 1] * ed[k - 1] - kt) / q[k]) * dt4;
        ed[k] *= ex;
      }
      ed[M - 1] += (M === 1 ? (ke2 - dof * kt) / q[0] : (q[M - 2] * ed[M - 2] * ed[M - 2] - kt) / q[M - 1]) * dt4;
    }
    this.sys.refreshComputes();
  }

  /** Barostat thermostat chain over dt/2. */
  private nhcPress(): void {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const M = this.pchain;
    const kt = kB * (this.tstat ? this.tTarget : (this.ptemp ?? this.currentTemp()));
    if (!(kt > 0)) return;
    // kinetic energy of the barostat
    let kecurrent = 0;
    let ndims = 0;
    let pfreqMax = 0;
    for (let d = 0; d < 6; d++) {
      if (!this.pFlag[d]) continue;
      kecurrent += this.omegaMass[d] * this.omegaDot[d] * this.omegaDot[d];
      ndims++;
      pfreqMax = Math.max(pfreqMax, 1 / this.pPeriod[d]);
    }
    const q = new Float64Array(M);
    q[0] = ndims * kt / (pfreqMax * pfreqMax);
    for (let k = 1; k < M; k++) q[k] = kt / (pfreqMax * pfreqMax);
    const nc = this.ploop;
    const dthalf = this.dthalf / nc, dt4 = this.dt4 / nc, dt8 = this.dt8 / nc;
    const ed = this.etapDot;
    for (let loop = 0; loop < nc; loop++) {
      const g = (k: number) => (k === 0 ? (kecurrent - ndims * kt) / q[0] : (q[k - 1] * ed[k - 1] * ed[k - 1] - kt) / q[k]);
      ed[M - 1] += g(M - 1) * dt4;
      for (let k = M - 2; k >= 0; k--) {
        const ex = Math.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex; ed[k] += g(k) * dt4; ed[k] *= ex;
      }
      const factor = Math.exp(-dthalf * ed[0]);
      for (let d = 0; d < 6; d++) if (this.pFlag[d]) this.omegaDot[d] *= factor;
      kecurrent *= factor * factor;
      for (let k = 0; k < M; k++) this.etap[k] += dthalf * ed[k];
      for (let k = 0; k < M - 1; k++) {
        const ex = Math.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex; ed[k] += g(k) * dt4; ed[k] *= ex;
      }
      ed[M - 1] += g(M - 1) * dt4;
    }
  }

  private computePressTarget(): void {
    for (let d = 0; d < 6; d++) {
      if (this.pFlag[d]) this.pTarget[d] = ramp(this.sys, this.pStart[d], this.pStop[d]);
    }
  }

  /** Current pressure (tensor), averaged over coupled dimensions. */
  private computePressCurrent(): void {
    const pc = this.pressCompute!;
    this.sys.refreshComputes();
    const v = pc.vectorValues();
    // tensor Voigt order of compute pressure: xx yy zz xy xz yz -> ours x y z yz xz xy
    const cur = [v[0], v[1], v[2], v[5], v[4], v[3]];
    const coupled = this.coupledDims();
    if (coupled.length) {
      let avg = 0;
      for (const d of coupled) avg += cur[d];
      avg /= coupled.length;
      for (const d of coupled) cur[d] = avg;
    }
    for (let d = 0; d < 6; d++) this.pCurrent[d] = cur[d];
  }

  /**
   * Barostat velocities over dt/2: W domega/dt = V (P - P_target) / nktv2p
   * + (MTK) k_B T_current-like term 2 KE / N_f on the diagonal.
   */
  private omegaUpdate(): void {
    const s = this.sys.state;
    const vol = this.sys.geom.volume(s.dimension);
    let mtkTerm1 = 0;
    if (this.mtk) {
      const { dof, kT } = this.dofKT();
      if (dof > 0) mtkTerm1 = kT;   // 2 KE / N_f = k_B T_current
      // energy per dimension (MTK): (1/N_f) sum p^2/m
    }
    const drag = this.drag > 0 ? 1 - this.dtv / this.pPeriod.reduce((a, b) => Math.max(a, b), 0) * this.drag : 1;
    for (let d = 0; d < 6; d++) {
      if (!this.pFlag[d]) continue;
      let force = (this.pCurrent[d] - this.pTarget[d]) * vol / s.units.nktv2p;
      if (d < 3) force += mtkTerm1;
      this.omegaDot[d] += force / this.omegaMass[d] * this.dthalf;
      this.omegaDot[d] *= drag;
    }
  }

  /** Velocity scaling by the strain rate over dt/4 (with the MTK trace term). */
  private velocityPress(): void {
    const s = this.sys.state;
    let mtkTerm2 = 0;
    if (this.mtk) {
      const dof = this.tempCompute.dof;
      if (dof > 0) {
        let tr = 0;
        for (let d = 0; d < (s.dimension === 2 ? 2 : 3); d++) if (this.pFlag[d]) tr += this.omegaDot[d];
        mtkTerm2 = tr / dof;
      }
    }
    // each call advances the strain-rate scaling by dt/4: diagonal factors split
    // symmetrically around the off-diagonal (tilt) coupling, x then y then z
    const f8 = [0, 1, 2].map((d) => (s.dimension === 2 && d === 2 ? 1 : Math.exp(-this.dt8 * (this.omegaDot[d] + mtkTerm2))));
    const c = this.tempCompute;
    const bias = c.hasBias();
    if (bias) { c.computeBias(); c.removeBiasAll(); }
    const wyz = this.omegaDot[3], wxz = this.omegaDot[4], wxy = this.omegaDot[5];
    const v = s.v;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const k = 3 * i;
      v[k] *= f8[0];
      v[k] -= this.dt4 * (v[k + 1] * wxy + v[k + 2] * wxz);
      v[k] *= f8[0];
      v[k + 1] *= f8[1];
      v[k + 1] -= this.dt4 * v[k + 2] * wyz;
      v[k + 1] *= f8[1];
      v[k + 2] *= f8[2] * f8[2];
    }
    if (bias) c.restoreBiasAll();
  }

  /**
   * Box and dilated atoms over dt/2: h <- exp(omega dt/2) h about the fixed
   * point (box centre by default), atoms mapped in fractional coordinates.
   */
  private remap(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const b = s.box;
    // fractional coordinates of dilated atoms
    const lam = new Float64Array(3 * s.n);
    const tmp = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.dilateBit)) continue;
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      lam[3 * i] = tmp[0]; lam[3 * i + 1] = tmp[1]; lam[3 * i + 2] = tmp[2];
    }
    const fp = this.fixedpoint ?? [0.5 * (b.lo[0] + b.hi[0]), 0.5 * (b.lo[1] + b.hi[1]), 0.5 * (b.lo[2] + b.hi[2])];
    const old = [g.lx, g.ly, g.lz];
    // tilts driven by the off-diagonal strain rates, then the diagonal ones
    // h_dot = omega h for an upper-triangular strain rate (yz, xz, xy), first order over dt/2
    if (this.pFlag[3]) b.tilt[2] += this.dthalf * this.omegaDot[3] * g.lz;
    if (this.pFlag[4]) b.tilt[1] += this.dthalf * (this.omegaDot[4] * g.lz + this.omegaDot[5] * g.yz);
    if (this.pFlag[5]) b.tilt[0] += this.dthalf * this.omegaDot[5] * g.ly;
    for (let d = 0; d < 3; d++) {
      if (!this.pFlag[d]) continue;
      const e = Math.exp(this.dthalf * this.omegaDot[d]);
      b.lo[d] = fp[d] + (b.lo[d] - fp[d]) * e;
      b.hi[d] = fp[d] + (b.hi[d] - fp[d]) * e;
    }
    // tilt factors follow their second dimension when not barostatted ("scale xy with ly" etc.)
    const nl = [b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]];
    if (this.scaleTilt[2] && !this.pFlag[5] && b.triclinic) b.tilt[0] *= nl[1] / old[1];
    if (this.scaleTilt[1] && !this.pFlag[4] && b.triclinic) b.tilt[1] *= nl[2] / old[2];
    if (this.scaleTilt[0] && !this.pFlag[3] && b.triclinic) b.tilt[2] *= nl[2] / old[2];
    // pressure-driven tilts also scale with the diagonal stretch
    if (this.pFlag[5]) b.tilt[0] *= Math.exp(this.dthalf * this.omegaDot[0]);
    if (this.pFlag[4]) b.tilt[1] *= Math.exp(this.dthalf * this.omegaDot[0]);
    if (this.pFlag[3]) b.tilt[2] *= Math.exp(this.dthalf * this.omegaDot[1]);
    for (let t = 0; t < 3; t++) {
      const len = t === 0 ? nl[0] : t === 1 ? nl[0] : nl[1];
      if (Math.abs(b.tilt[t]) > 0.6 * len && b.triclinic) {
        throw new StyleError(`fix ${this.style}: a tilt factor exceeded 0.6 of the box length; box flips are not supported by the browser engine`);
      }
    }
    b.minLo = [...b.lo]; b.minHi = [...b.hi];
    g.update();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.dilateBit)) continue;
      g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
    }
    for (const f of sys.fixes) if (f !== this) f.boxChanged?.();
  }

  // ------------------------------------------------------------------ output

  /** Energy exchanged with the reservoirs (extended-system terms). */
  ecouple(): number {
    const s = this.sys.state;
    const kB = s.units.boltz;
    let e = 0;
    if (this.tstat) {
      const kt = kB * this.tTarget;
      const tfreq = 1 / this.tPeriod;
      const dof = this.tempCompute.dof;
      const q0 = dof * kt / (tfreq * tfreq), qk = kt / (tfreq * tfreq);
      e += dof * kt * this.eta[0] + 0.5 * q0 * this.etaDot[0] * this.etaDot[0];
      for (let k = 1; k < this.tchain; k++) e += kt * this.eta[k] + 0.5 * qk * this.etaDot[k] * this.etaDot[k];
    }
    if (this.pstat) {
      const vol = this.sys.geom.volume(s.dimension);
      let ndims = 0, pfreqMax = 0;
      let pHydro = 0, nhyd = 0;
      for (let d = 0; d < 6; d++) {
        if (!this.pFlag[d]) continue;
        e += 0.5 * this.omegaDot[d] * this.omegaDot[d] * this.omegaMass[d];
        ndims++;
        pfreqMax = Math.max(pfreqMax, 1 / this.pPeriod[d]);
        if (d < 3) { pHydro += this.pTarget[d]; nhyd++; }
      }
      if (nhyd) e += (pHydro / nhyd) * (vol - this.vol0) / s.units.nktv2p;
      if (this.pchain > 0) {
        const kt = kB * (this.tstat ? this.tTarget : (this.ptemp ?? 0));
        const q0 = ndims * kt / (pfreqMax * pfreqMax), qk = kt / (pfreqMax * pfreqMax);
        e += ndims * kt * this.etap[0] + 0.5 * q0 * this.etapDot[0] * this.etapDot[0];
        for (let k = 1; k < this.pchain; k++) e += kt * this.etap[k] + 0.5 * qk * this.etapDot[k] * this.etapDot[k];
      }
    }
    return e + this.ecoupleCum;
  }

  computeScalar(): number {
    return this.ecouple();
  }
}
