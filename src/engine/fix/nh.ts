import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Compute } from '../compute/compute';
import { ownCompute, ramp, removeCompute } from './util';
import { massOf } from '../atoms';
import type { ComputePressure } from '../compute/pressure';

/*
 * fix nvt / npt / nph — docs.lammps.org/fix_nh.html.
 *
 * "The equations of motion used are those of Shinoda et al in :ref:`(Shinoda)
 * <nh-Shinoda>`, which combine the hydrostatic equations of Martyna, Tobias and
 * Klein in :ref:`(Martyna) <nh-Martyna>` with the strain energy proposed by
 * Parrinello and Rahman in :ref:`(Parrinello) <nh-Parrinello>`. The time
 * integration schemes closely follow the time-reversible measure-preserving
 * Verlet and rRESPA integrators derived by Tuckerman et al in :ref:`(Tuckerman)
 * <nh-Tuckerman>`." The equations are written from those papers (Martyna,
 * Tobias, Klein, J Chem Phys 101, 4177 (1994); Shinoda, Shiga, Mikami, Phys
 * Rev B 69, 134103 (2004); Tuckerman et al, J Phys A 39, 5629 (2006)).
 *
 * Step order (per timestep, the factorisation is the one checked against
 * native LAMMPS, see the measurements below):
 *   initial: thermostat chain; barostat chain; pressure; barostat velocity
 *            update (dt/2); velocity scaling by the strain rate (dt/4) twice;
 *            half kick; box remap (dt/2); drift; box remap (dt/2)
 *   final:   half kick; velocity scaling (dt/4) twice; pressure; barostat
 *            velocity update (dt/2); barostat chain; thermostat chain.
 * Masses: thermostat Q_1 = N_f k_B T Tdamp^2, Q_k = k_B T Tdamp^2 (k > 1);
 * barostat "W = (N + 1) k_B T_\mathrm{target} P_\mathrm{damp}^2"; the
 * barostat chain has Q_1 = k_B T Pdamp^2 and Q_k = k_B T Pdamp^2 (k > 1), with
 * the number of barostatted dimensions entering the chain force.
 * "If a thermostat is not defined, :math:`T_\mathrm{target}` is set to the
 * current temperature of the system when the barostat is initialized." "The
 * *mtk* keyword controls whether or not the correction terms due to Martyna,
 * Tuckerman, and Klein are included in the equations of motion". The MTK terms are measured below.
 * Keywords: temp, iso, aniso, tri, x, y, z, xy, yz, xz, couple, tchain,
 * pchain, mtk, tloop, ploop, nreset, drag, ptemp, dilate, scalexy/yz/xz,
 * flip, fixedpoint. Defaults "tchain = 3, pchain = 3, mtk = yes, tloop = 1,
 * ploop = 1, nreset = 0, drag = 0.0, dilate = all, couple = none, flip =
 * yes". "iso ... couple xyz"; "aniso ... couple none"; "tri Pstart Pstop Pdamp"
 * is the same as x, y and z with Pstart Pstop Pdamp, xy, yz and xz with
 * 0.0 0.0 Pdamp, and couple none (the doc lists the 7 keywords).
 * "The optional *drag* keyword will damp these oscillations, although it alters
 * the Nose/Hoover equations." The drag factor is applied to the particle
 * thermostat, the barostat velocities and the barostat chain.
 * For fix nvt, the docs give the code line "compute fix-ID_temp group-ID
 * temp"; for fix npt and fix nph, the code lines "compute fix-ID_temp all
 * temp" and "compute fix-ID_press all pressure fix-ID_temp".
 * Global scalar: "The scalar is the same cumulative energy change due to this
 * fix described in the previous paragraph", reported by thermo ecouple.
 *
 * Measured with native LAMMPS (black box, tests/oracle/w7nh_*.in and w42nhm_*.in):
 *   - the global vector (f_ID[k]) layout: eta, eta_dot (tchain), omega, omega_dot (ndof), etap, etap_dot
 *     (pchain), PE_eta, KE_eta_dot, PE_omega, KE_omega_dot, PE_etap, KE_etap_dot, and PE_strain[1] for the
 *     diagonal couplings (couple none; 25 entries for nph with pchain 3, 13 with pchain 0; npt adds the tchain
 *     blocks). Iso has no PE_strain entry (12 entries for nph with pchain 3).
 *   - PE_etap[k] = k_B T_p etap[k] for every k (T_p the barostat temperature); no count of barostatted dims.
 *   - PE_omega per barostatted diagonal dimension = Pbar (V - V0) / (nbaro nktv2p), with Pbar the mean
 *     target over the barostatted dimensions and free dimensions 0. Native for x only: P_x (V - V0) (one entry).
 *   - the scalar is the sum of the KE_omega_dot, PE_omega, PE_strain, and barostat-chain entries (equal
 *     to ecouple with the thermo normalisation: lj units divide an extensive scalar by the atom count).
 *   - PE_strain = [ -(3 - nbaro)/2 * Pbar * V0 + sum_d (P_d - Pbar) V0 (r_d^2 - 1)/2 ] / nktv2p, with
 *     r_d = L_d / L_d(0) over the barostatted diagonal dimensions. The first term is the step-0 reference
 *     (x only: -P V0; x and y: -P V0/2). The second term was fitted to native (least squares over the 100 steps
 *     of nph x 1 y 3 z 2 and npt x 2 y 3, z free) with residual 1e-12. Under a ramp the current targets are used.
 *   - the barostat velocity update, the MTK terms and the thermostat chain for the cases with equal
 *     targets (or one barostatted dimension) match native to rel 1e-9 over 100 steps (w42nhm_*).
 *   - the MTK force term (per barostatted dimension) is the sum over the barostatted diagonal
 *     dimensions of sum m v^2, divided by (number of barostatted dimensions) * N. The iso
 *     barostat (couple xyz) uses the temperature compute's dof * k_B T instead (rotational dof
 *     included, the sphere/asphere cases). The MTK trace term in the velocity update is the sum
 *     of omega_dot over the barostatted dimensions, divided by (number of barostatted dimensions) * N.
 *   - barostat velocity scaling: both scalings of a half step come before the kick in the initial step and
 *     after it in the final step.
 *   - the barostat chain force counts the barostatted dimensions (couple none).
 *   - the step-0 scalar is -(3 - nbaro)/2 * P_mean * V0 / nktv2p (nbaro = barostatted diagonal
 *     dimensions): x only -P V0, x and y -P V0/2, iso and xyz 0. The 3 is used in 2d too (2d x only
 *     -P V0, 2d x and y -P V0/2), so the 2d form is this same rule.
 *   - two Nose-Hoover fixes in one run, npt on a group that is a subset of the temperature group (npt's
 *     temperature compute is group all) with nvt on the other atoms (w74nhsub_*): the first thermostat update of
 *     a step and the iso barostat's MTK term use the kinetic energy tracked through the previous thermostat update
 *     of the same fix, not the current kinetic energy. The tracked value multiplies 2 KE by the square of every
 *     velocity factor the chain applies, although only the fix group is scaled, so it differs from the current
 *     value only when atoms outside the fix group have kinetic energy (zero-velocity drudes: exact; npt on all
 *     atoms: exact). Measured residual with this rule: 5e-9 relative in the barostat velocity at step 1 (iso),
 *     3e-9 in the pressure at step 30; aniso (no MTK iso term) 1e-13. The tracked value is not used after setup.
 *     fix nph (no thermostat) with nve on the other atoms: its MTK term uses the all-atom kinetic energy at the end
 *     of the previous step (w74nhsub_nph_drudes, same 1e-8 tolerance); the current value differs by 4e-7 at step 2.
 *   - tilts: the xy-only barostat is exact (xy += ly * omega_xy); with the
 *     diagonal and tilt couplings together, the box is within ~3e-8 of the
 *     exact flow at dt = 0.02 (not reproduced; see the w7nh_npt_tri notes).
 * Not resolved (unequal diagonal targets, e.g. x 1, y 3, z 2): the barostat velocities differ from native
 * by a small term proportional to (P_d - Pbar) (about 1e-5 relative after one step, 9 % in the box after
 * 100 steps for nph x 1, y 3, z 2 with pchain 0; measured with w42nhm cases generated and checked, not
 * committed). The term is linear in (P_d - Pbar) and its coefficient depends on the state (it grows as the
 * kinetic energy falls). Its form has not been identified. The strain energy and the vector formulas above
 * were fitted to native values from the same state, so they hold; the scalar matches native only for the
 * cases with equal targets.
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
  private omegaPos = new Float64Array(6);
  private omegaMass = new Float64Array(6);
  private etap: Float64Array; private etapDot: Float64Array;
  private pTarget = new Float64Array(6);
  private pCurrent = new Float64Array(6);
  private tTarget = 0;
  private pTempFixed = 0;
  private tempCompute!: Compute;
  private pressCompute: Compute | null = null;
  /** kinetic tensor of the pressure's temperature compute at the initial step's pressure (see computePressCurrent). */
  private keInitial: Float64Array | null = null;
  private tempId: string;
  private pressId: string | null = null;
  private dthalf = 0; private dt4 = 0; private dt8 = 0; private dtf = 0; private dtv = 0;
  private vol0 = 0;
  private len0: number[] = [1, 1, 1];   // box lengths at the start of the run (strain reference)
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
    this.vectorFlag = true;
    this.sizeVector = this.vectorLayout().reduce((a, b) => a + b.n, 0);
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

  /**
   * unfix deletes the computes this fix created (ID_temp, ID_press), not computes that fix_modify
   * temp/press pointed it to: those belong to the input script.
   */
  destroy(): void {
    removeCompute(this.sys, `${this.id}_temp`);
    removeCompute(this.sys, `${this.id}_press`);
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
    this.len0 = [this.sys.geom.lx, this.sys.geom.ly, this.sys.geom.lz];
    // barostat target temperature
    this.tTarget = this.tstat ? ramp(this.sys, this.tStart, this.tStop) : this.currentTemp();
    if (this.pstat) {
      let tb = this.tstat ? this.tTarget : this.ptemp ?? this.currentTemp();
      if (!this.tstat && this.ptemp === null && !(tb > 0)) {
        throw new StyleError(`fix ${this.style}: the system temperature is 0; set ptemp to give the barostat a target temperature`);
      }
      if (!(tb > 0)) tb = this.ptemp ?? 1;
      this.pTempFixed = tb;
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

  /** Temperature of the barostat chain: the thermostat target, or for nph the fixed ptemp / initial temperature. */
  private baroTemp(): number {
    return this.tstat ? this.tTarget : this.pTempFixed;
  }

  /**
   * Temperature at this fix's own setup. The first thermostat evaluation after setup uses it and then clears it; every
   * later evaluation uses the current temperature (subclasses such as nvt/sllod override the hooks, so the clearing
   * is done at the use, not in finalIntegrate).
   * Measured with native LAMMPS (black box; dimer of w50il_dimer.data, temp 300 300 100, lj/cut pair, 10 steps, f_1[1..4]
   * of the fix): with fix nvt defined before fix ilves, the setup projection of ilves is not seen by the first
   * thermostat update (f_1[1] at step 1 is 0.000284299502599865, the value of the setup-time temperature; the current
   * temperature gives 0.000250618774807639); with fix ilves defined before fix nvt, nvt's setup already sees the
   * projected velocities and the current temperature is the right one (step 10: f_1[1] 0.0247325814526837). Without
   * ilves both orders are the current temperature. The end-of-step alternatives (caching the temperature after the
   * final kick or after its thermostat scaling) do not reproduce the 10-step values, so later updates stay current.
   */
  private tSetup: number | null = null;
  /**
   * Kinetic energy 2 KE (energy units, = dof k_B T) tracked through the chain's last update of the previous step
   * (null after setup). The chain multiplies it by the square of each velocity factor it applies, as its own
   * update does, even though only the fix group is scaled (see the measurement in the header comment). For fix nph
   * (no chain) it is the all-atom kinetic energy at the end of the final step.
   */
  private trackKE: number | null = null;
  /** Product of the squared velocity factors of the current initial-step chain update (1 outside it). */
  private nhcScale2 = 1;

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
    this.tSetup = this.currentTemp();
    this.trackKE = null;
    if (this.tstat) this.tTarget = ramp(this.sys, this.tStart, this.tStop);
    if (this.pstat) {
      this.computePressTarget();
      this.computePressCurrent();
    }
  }

  initialIntegrate(): void {
    if (this.tstat) {
      this.tTarget = ramp(this.sys, this.tStart, this.tStop);
      this.nhcTemp(true);
    }
    if (this.pstat) {
      this.computePressTarget();
      if (this.pchain > 0) this.nhcPress();
      this.computePressCurrent('init');
      this.omegaUpdate(true);
      // barostat velocity scaling (dt/4 each) on both sides of the half kick is applied before it
      this.velocityPress();
      this.velocityPress();
    }
    this.kick();
    if (this.pstat) this.remap();
    this.drift();
    if (this.pstat) this.remap();
  }

  finalIntegrate(): void {
    this.kick();
    if (this.pstat) {
      this.velocityPress();
      this.velocityPress();
      this.computePressCurrent('final');
      this.omegaUpdate();
      if (this.pchain > 0) this.nhcPress();
    }
    if (this.tstat) this.nhcTemp();
    else if (this.pstat) {
      // fix nph has no thermostat chain to track the kinetic energy: the next initial MTK term uses the value at the
      // end of this step (see trackKE and the measurement in the header comment)
      const d = this.dofKT();
      this.trackKE = d.dof * d.kT;
    }
  }

  // ------------------------------------------------------------------ run_style respa (run/respa.ts)

  /**
   * Only the plain thermostat (fix nvt, no barostat, not the sphere/asphere variants) has a respa
   * factorization: the thermostat half steps sit at the outermost level (fix_nh.rst: "the barostat
   * is only updated at the outermost rRESPA level"; the barostat is not implemented under respa).
   */
  get respaOK(): boolean { return this.constructor === FixNH && this.tstat && !this.pstat; }

  /** Outer step start: thermostat dt/2 (exp(i L_Tpart dt/2) at the outermost level). */
  respaBegin(): void {
    this.tTarget = ramp(this.sys, this.tStart, this.tStop);
    this.nhcTemp();
  }

  /** Outer step end: the matching thermostat dt/2. */
  respaEnd(): void {
    this.nhcTemp();
  }

  /** Level kick v += h F/m over the group, with that level's force array f. */
  respaKick(f: Float64Array, h: number): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const ftm2v = h * s.units.ftm2v;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const c = ftm2v / massOf(s, i);
      v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
    }
  }

  /** Innermost drift x += dt v over the group. */
  respaDrift(dt: number): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      x[3 * i] += dt * v[3 * i]; x[3 * i + 1] += dt * v[3 * i + 1]; x[3 * i + 2] += dt * v[3 * i + 2];
    }
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
  private nhcTemp(initial = false): void {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const M = this.tchain;
    const { dof, kT: kTfresh } = this.dofKT();
    // native: the first thermostat update uses the setup-time temperature (before later fixes' setup changes velocities)
    const tSetup = this.tSetup;
    this.tSetup = null;
    // the first update of a step uses the tracked value (see trackKE), not the current kinetic energy
    if (initial) this.nhcScale2 = 1;
    const kTcur = tSetup !== null ? kB * tSetup : initial && this.trackKE !== null ? this.trackKE / dof : kTfresh;
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
      this.nhcScale2 *= factor * factor;
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
    if (!initial) this.trackKE = ke2;
    this.sys.refreshComputes();
  }

  /** Barostat thermostat chain over dt/2. */
  private nhcPress(): void {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const M = this.pchain;
    const kt = kB * this.baroTemp();
    if (!(kt > 0)) return;
    // kinetic energy of the barostat
    let kecurrent = 0;
    let pfreqMax = 0;
    for (let d = 0; d < 6; d++) {
      if (!this.pFlag[d]) continue;
      kecurrent += this.omegaMass[d] * this.omegaDot[d] * this.omegaDot[d];
      pfreqMax = Math.max(pfreqMax, 1 / this.pPeriod[d]);
    }
    const ndims = this.chainDims();
    const q = new Float64Array(M);
    q[0] = kt / (pfreqMax * pfreqMax);
    for (let k = 1; k < M; k++) q[k] = kt / (pfreqMax * pfreqMax);
    const nc = this.ploop;
    const dthalf = this.dthalf / nc, dt4 = this.dt4 / nc, dt8 = this.dt8 / nc;
    const ed = this.etapDot;
    const pdrag = this.drag > 0 ? 1 - this.dtv * pfreqMax * this.drag / nc : 1;
    for (let loop = 0; loop < nc; loop++) {
      const g = (k: number) => (k === 0 ? (kecurrent - ndims * kt) / q[0] : (q[k - 1] * ed[k - 1] * ed[k - 1] - kt) / q[k]);
      ed[M - 1] += g(M - 1) * dt4;
      ed[M - 1] *= pdrag;
      for (let k = M - 2; k >= 0; k--) {
        const ex = Math.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex; ed[k] += g(k) * dt4; ed[k] *= pdrag; ed[k] *= ex;
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

  /**
   * Current pressure (tensor), averaged over coupled dimensions.
   *
   * Kinetic part, measured with native LAMMPS (black box, tests/oracle/w81nhpress_*). When the barostat's
   * pressure compute takes a temperature compute other than the fix's own (fix_modify press thermo_press, or
   * a pressure compute on another temperature), the final-step pressure uses the kinetic tensor of the initial
   * step's pressure evaluation (before the final half kick and the velocity scalings), not the current one.
   * The fix's own pressure compute (the fix's temperature) uses the current tensor, as do the initial step and
   * the thermo output for couple xyz. For couple none the thermo output keeps the initial-step tensor too
   * (diagonal components press_xx, press_yy, press_zz; the scalar stays current). The mechanism is not identified.
   * Measured with this rule (w81nhpress_iso_* and w81nhpress_aniso_*, 30 steps): relative error below 1e-12 in
   * the thermo press, pxx, pyy, pzz and the ecouple scalar (f_1), and in the box lengths.
   */
  private computePressCurrent(stage: 'setup' | 'init' | 'final' = 'setup'): void {
    const pc = this.pressCompute! as ComputePressure;
    pc.keOverride = null;
    this.sys.refreshComputes();
    const pressTemp = pc.tempId;
    const kt = pressTemp !== null && pressTemp !== this.tempId ? this.sys.compute(pressTemp) : null;
    if (kt && kt.vectorFlag) {
      if (stage === 'init') this.keInitial = Float64Array.from(kt.vectorValues());
      else if (stage === 'final' && this.keInitial) {
        pc.keOverride = this.keInitial;
        this.sys.refreshComputes();
      }
    }
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
    if (stage === 'final' && this.couple !== 'none') {
      pc.keOverride = null;
      this.sys.refreshComputes();
    }
  }

  /**
   * Barostat velocities over dt/2: W domega/dt = V (P - P_target) / nktv2p
   * + (MTK) k_B T_current-like term 2 KE / N_f on the diagonal.
   */
  private omegaUpdate(initial = false): void {
    const s = this.sys.state;
    const vol = this.sys.geom.volume(s.dimension);
    let mtkTerm1 = 0;
    if (this.mtk) {
      // MTK term per barostat dimension: sum over the barostatted dimensions of sum m v^2, divided by
      // (number of barostatted dimensions) * N (measured, see the header comment)
      const nb = this.nbaroDiag();
      if (nb > 0) {
        if (this.couple === 'xyz') {
          // isotropic barostat: the temperature compute's full dof * kT (rotational dof included)
          const { dof, kT } = this.dofKT();
          // the first update of a step: the tracked kinetic energy, after the chain's own scaling
          const ke2 = initial && this.trackKE !== null ? this.trackKE * this.nhcScale2 : dof * kT;
          if (dof > 0) mtkTerm1 = ke2 / (nb * s.n);
        } else {
          mtkTerm1 = this.keBaro() / (nb * s.n);
        }
      }
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
  /**
   * 2 KE of the barostatted diagonal dimensions in energy units: the diagonal of the temperature compute's
   * kinetic energy tensor (its group and velocity bias apply), summed over the barostatted dimensions. With
   * every dimension barostatted it equals dof * kT, the iso form. Checked against native LAMMPS (black box)
   * with fix_modify temp of a compute temp/partial (case w41nptp_bias) and in metal units (w41nptp_metal_*).
   */
  private keBaro(): number {
    const tc = this.tempCompute;
    if (!tc.vectorFlag) return this.sys.state.units.mvv2e * this.mvvBaro();
    const ke = tc.vectorValues();
    let acc = 0;
    for (let d = 0; d < 3; d++) if (this.pFlag[d]) acc += ke[d];
    return acc;
  }

  /** Sum over the barostatted diagonal dimensions of sum_i m_i v_i^2 over all atoms (a temperature compute without a vector). */
  private mvvBaro(): number {
    const s = this.sys.state;
    let acc = 0;
    for (let i = 0; i < s.n; i++) {
      const m = massOf(s, i);
      for (let d = 0; d < 3; d++) if (this.pFlag[d]) acc += m * s.v[3 * i + d] * s.v[3 * i + d];
    }
    return acc;
  }

  /**
   * Degrees of freedom in the barostat chain force. Measured for independent diagonal dimensions
   * (couple none): the barostatted dimensions. Other couplings and tilts keep the ndof of the vector.
   */
  private chainDims(): number {
    if (this.couple === 'none') {
      let n = 0;
      for (let d = 0; d < 6; d++) if (this.pFlag[d]) n++;
      return n;
    }
    return this.ndofDims().length;
  }

  /** Number of diagonal dimensions that are barostatted (the MTK terms divide by this times N). */
  private nbaroDiag(): number {
    return (this.pFlag[0] ? 1 : 0) + (this.pFlag[1] ? 1 : 0) + (this.pFlag[2] ? 1 : 0);
  }

  private velocityPress(): void {
    const s = this.sys.state;
    let mtkTerm2 = 0;
    if (this.mtk) {
      const dof = this.tempCompute.dof;
      if (dof > 0) {
        let tr = 0;
        for (let d = 0; d < (s.dimension === 2 ? 2 : 3); d++) if (this.pFlag[d]) tr += this.omegaDot[d];
        mtkTerm2 = tr / (this.nbaroDiag() * s.n);
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
      // the coupling terms act on the partner velocity while it decays over dt/4 (its own factor, f8)
      v[k] *= f8[0];
      v[k] -= this.dt4 * (v[k + 1] * f8[1] * wxy + v[k + 2] * f8[2] * wxz);
      v[k] *= f8[0];
      v[k + 1] *= f8[1];
      v[k + 1] -= this.dt4 * v[k + 2] * f8[2] * wyz;
      v[k + 1] *= f8[1];
      v[k + 2] *= f8[2] * f8[2];
    }
    if (bias) c.restoreBiasAll();
  }

  /**
   * Tilt factors [xy, xz, yz] after dt/2 of the barostat flow of the barostatted tilts. With
   * h_dot = omega h (upper-triangular h; omega 0..2 the diagonal strain rates, 3 = yz, 4 = xz, 5 = xy), the
   * diagonal lengths are L_d(s) = L_d exp(omega_d s) and the tilts obey
   *   xy' = w0 xy + w5 ly(s),   yz' = w1 yz + w3 lz(s),   xz' = w0 xz + w5 yz(s) + w4 lz(s).
   * The decay of each tilt is applied in full over dt/2; the sources are integrated over the half step with
   * the partner's time dependence, and the source of xy and xz carries half the rate difference
   * exp(dt (w_tilt_row - w_source_row) / 2). yz has the closed form for yz(s). Measured against native LAMMPS
   * (black box, tests/oracle/w97tri_npt_xy, w97tri_nph_xz, w97tri_npt_tri, 20 steps): xy and yz agree to about
   * 1e-11 relative; xz (the all-tilt case) to about 2e-7 relative at step 20, the one residual not identified.
   * Unbarostatted tilts keep their input value here (they are scaled after the box update).
   */
  private tiltFlow(ly: number, lz: number, yz: number, tilt: [number, number, number]): [number, number, number] {
    const out: [number, number, number] = [tilt[0], tilt[1], tilt[2]];
    const tau = this.dthalf;
    const w = this.omegaDot;
    const expo = (a: number) => Math.exp(a);
    const lyAt = (s: number) => ly * expo(w[1] * s);
    const lzAt = (s: number) => lz * expo(w[2] * s);
    // yz(s) = yz exp(w1 s) + w3 * integral of lz(u) exp(w1 (s - u)) du, in closed form
    const yzAt = (s: number): number => {
      if (!this.pFlag[3]) return yz * expo(w[2] * s);
      const d = w[2] - w[1];
      const cross = Math.abs(d) < 1e-12 ? w[3] * lz * s * expo(w[1] * s) : w[3] * lz * (expo(w[2] * s) - expo(w[1] * s)) / d;
      return yz * expo(w[1] * s) + cross;
    };
    // Simpson's rule over [0, tau]
    const simpson = (f: (s: number) => number): number => {
      const n = 8, h = tau / n;
      let acc = f(0) + f(tau);
      for (let k = 1; k < n; k++) acc += (k % 2 ? 4 : 2) * f(k * h);
      return acc * h / 3;
    };
    if (this.pFlag[5]) {
      out[0] = expo(tau * w[0]) * tilt[0] + simpson((s) => w[5] * lyAt(s)) * expo(tau * (w[0] - w[1]) / 2);
    }
    if (this.pFlag[4]) {
      out[1] = expo(tau * w[0]) * tilt[1]
        + simpson((s) => w[5] * yzAt(s)) * expo(tau * (w[0] - w[1]) / 2)
        + simpson((s) => w[4] * lzAt(s)) * expo(tau * (w[0] - w[2]) / 2);
    }
    if (this.pFlag[3]) {
      out[2] = expo(tau * w[1]) * yz + simpson((s) => w[3] * lzAt(s)) * expo(tau * (w[1] - w[2]) / 2);
    }
    return out;
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
    for (let d = 0; d < 6; d++) if (this.pFlag[d]) this.omegaPos[d] += this.dthalf * this.omegaDot[d];
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
    // barostatted tilts: the exact flow of h_dot = omega h over dt/2 (see tiltFlow)
    const tiltNew = this.tiltFlow(g.ly, g.lz, g.yz, b.tilt);
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
    if (this.pFlag[5]) b.tilt[0] = tiltNew[0];
    if (this.pFlag[4]) b.tilt[1] = tiltNew[1];
    if (this.pFlag[3]) b.tilt[2] = tiltNew[2];
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

  /** Mean target pressure over the barostatted diagonal dimensions (0 when none). */
  private pbarDiag(): number {
    const nb = this.nbaroDiag();
    if (nb === 0) return 0;
    let acc = 0;
    for (let d = 0; d < 3; d++) if (this.pFlag[d]) acc += this.pTarget[d];
    return acc / nb;
  }

  /**
   * Strain energy of the diagonal barostat (the scalar's PE_strain entry), in energy units:
   *   [ -(3 - nbaro)/2 * Pbar * V0 + sum_d (P_d - Pbar) * V0 * (r_d^2 - 1)/2 ] / nktv2p,
   * over the barostatted diagonal dimensions d, with r_d = L_d / L_d(0) and Pbar the mean target
   * (see the header comment for the measurements behind it).
   */
  private strainEnergy(): number {
    const s = this.sys.state;
    const nb = this.nbaroDiag();
    if (nb === 0) return 0;
    const pbar = this.pbarDiag();
    const g = this.sys.geom;
    const len = [g.lx, g.ly, g.lz];
    let e = -((3 - nb) / 2) * pbar * this.vol0;
    for (let d = 0; d < 3; d++) {
      if (!this.pFlag[d]) continue;
      const r = len[d] / this.len0[d];
      e += (this.pTarget[d] - pbar) * this.vol0 * (r * r - 1) / 2;
    }
    return e / s.units.nktv2p;
  }

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
      let pfreqMax = 0;
      let pHydro = 0, nhyd = 0;
      for (let d = 0; d < 6; d++) {
        if (!this.pFlag[d]) continue;
        e += 0.5 * this.omegaDot[d] * this.omegaDot[d] * this.omegaMass[d];
        pfreqMax = Math.max(pfreqMax, 1 / this.pPeriod[d]);
        if (d < 3) { pHydro += this.pTarget[d]; nhyd++; }
      }
      let nflag = 0;
      for (let d = 0; d < 6; d++) if (this.pFlag[d]) nflag++;
      if (nhyd) {
        const pMean = pHydro / nhyd;
        e += pMean * (vol - this.vol0) / s.units.nktv2p;
        e += this.strainEnergy();
      }
      if (this.pchain > 0) {
        const kt = kB * this.baroTemp();
        const q0 = kt / (pfreqMax * pfreqMax), qk = kt / (pfreqMax * pfreqMax);
        // the scalar's potential term of the first barostat chain element counts the barostatted dimensions
        e += nflag * kt * this.etap[0] + 0.5 * q0 * this.etapDot[0] * this.etapDot[0];
        for (let k = 1; k < this.pchain; k++) e += kt * this.etap[k] + 0.5 * qk * this.etapDot[k] * this.etapDot[k];
      }
    }
    return e + this.ecoupleCum;
  }

  computeScalar(): number {
    return this.ecouple();
  }

  /**
   * Global vector, in the documented order: "eta[tchain] ... eta_dot[tchain]
   * ... omega[ndof] ... omega_dot[ndof] ... etap[pchain] ... etap_dot[pchain]
   * ... PE_eta[tchain] ... KE_eta_dot[tchain] ... PE_omega[ndof] ...
   * KE_omega_dot[ndof] ... PE_etap[pchain] ... KE_etap_dot[pchain]". The
   * "ndof" barostat entries are the coupled dimensions (one entry for couple
   * xyz; six for tilt keywords; otherwise three).
   */
  computeVector(i: number): number {
    for (const seg of this.vectorLayout()) {
      if (i < seg.n) return seg.value(i);
      i -= seg.n;
    }
    return 0;
  }

  /**
   * thermo_modify colname auto names (fix_nh.rst, versionadded 10Dec2025): the scalar is ecouple, and each vector
   * entry is its documented keyword with an index, for example eta[1] or PE_omega[2]. The order follows vectorLayout.
   */
  autoColumnNames(): { scalar: string; vector: string[] } {
    const keywords = ['eta', 'eta_dot', 'omega', 'omega_dot', 'etap', 'etap_dot', 'PE_eta', 'KE_eta_dot', 'PE_omega', 'KE_omega_dot', 'PE_etap', 'KE_etap_dot', 'PE_strain'];
    const vector: string[] = [];
    this.vectorLayout().forEach((seg, s) => {
      for (let j = 1; j <= seg.n; j++) vector.push(`${keywords[s]}[${j}]`);
    });
    return { scalar: 'ecouple', vector };
  }

  /** Barostat dimensions reported in the vector (ndof of the docs). */
  private ndofDims(): number[] {
    if (!this.pstat) return [];
    if (this.pFlag[3] || this.pFlag[4] || this.pFlag[5]) return [0, 1, 2, 3, 4, 5];
    if (this.couple === 'xyz' || (this.couple === 'xy' && this.sys.dimension === 2)) return [0];
    return [0, 1, 2];
  }

  private vectorLayout(): { n: number; value: (i: number) => number }[] {
    const s = this.sys.state;
    const kB = s.units.boltz;
    const nT = this.tstat ? this.tchain : 0;
    const nP = this.pstat ? this.pchain : 0;
    const dims = this.ndofDims();
    const tilt = this.pFlag[3] || this.pFlag[4] || this.pFlag[5];
    const nb = this.nbaroDiag();
    const dV = this.sys.geom.volume(s.dimension) - this.vol0;
    // barostat potential energy: a single entry (iso, 2d xy) holds Pbar dV; otherwise each barostatted
    // dimension holds an equal share of Pbar dV (tilt keywords keep the earlier group form)
    const peOmega = (j: number): number => {
      if (tilt) return this.omegaGroupSum(j) * dV / (3 * s.units.nktv2p);
      if (dims.length === 1) return this.pbarDiag() * dV / s.units.nktv2p;
      return this.pFlag[dims[j]] ? this.pbarDiag() * dV / (nb * s.units.nktv2p) : 0;
    };
    const kt = kB * this.tTarget;
    const ktP = kB * this.baroTemp();
    const tfreq = this.tstat ? 1 / this.tPeriod : 0;
    const dof = this.tstat ? this.tempCompute.dof : 0;
    const qT = (k: number) => (k === 0 ? dof * kt / (tfreq * tfreq) : kt / (tfreq * tfreq));
    let pfreq = 0;
    for (const d of dims) if (this.pFlag[d]) pfreq = Math.max(pfreq, 1 / this.pPeriod[d]);
    const qP = (k: number) => ktP / (pfreq * pfreq);
    return [
      { n: nT, value: (k) => this.eta[k] },
      { n: nT, value: (k) => this.etaDot[k] },
      { n: dims.length, value: (j) => this.omegaPos[dims[j]] },
      { n: dims.length, value: (j) => this.omegaDot[dims[j]] },
      { n: nP, value: (k) => this.etap[k] },
      { n: nP, value: (k) => this.etapDot[k] },
      { n: nT, value: (k) => (k === 0 ? dof * kt : kt) * this.eta[k] },
      { n: nT, value: (k) => 0.5 * qT(k) * this.etaDot[k] * this.etaDot[k] },
      { n: dims.length, value: (j) => peOmega(j) },
      { n: dims.length, value: (j) => (dims.length === 1 ? [0, 1, 2] : [dims[j]]).reduce((a, d) => a + 0.5 * this.omegaMass[d] * this.omegaDot[d] * this.omegaDot[d], 0) },
      { n: nP, value: (k) => ktP * this.etap[k] },
      { n: nP, value: (k) => 0.5 * qP(k) * this.etapDot[k] * this.etapDot[k] },
      // "PE_strain[1] = scalar strain energy (energy units)" (fix_nh.rst): the diagonal barostat's strain energy
      // (diagonal couplings only; the measured iso layout has no such entry)
      { n: this.pstat && !tilt && this.couple === 'none' ? 1 : 0, value: () => this.strainEnergy() },
    ];
  }

  /** Sum of target pressures over the group of a tilt-keyword vector entry (the earlier form, kept for tilts). */
  private omegaGroupSum(j: number): number {
    const dims = this.ndofDims();
    const group = dims.length === 1 ? [0, 1, 2] : [dims[j]];
    return group.reduce((acc, d) => acc + this.pTarget[d], 0);
  }

}
