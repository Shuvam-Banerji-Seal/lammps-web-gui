import { takeFixRecord } from './restartRecord';
import { Fix, fixModifyCompute } from './fix';
import { StyleError } from '../force/types';
import { missingArgs } from '../force/util';
import type { System } from '../system';
import type { Compute } from '../compute/compute';
import { ownCompute, ramp, removeCompute } from './util';
import { massOf } from '../atoms';
import { ComputePressure } from '../compute/pressure';
import { libm } from '../libm';
import { gplHook } from '../gplHooks';

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
 * flip, isochoric, fixedpoint. "The *isochoric* keyword allows to maintain constant volume when barostatting up to two
 * dimensions with this fix. ... The selected dimensions are scaled to compensate the strain induced by the barostat and
 * keep the system at a constant volume (or area in 2d)." Defaults "tchain = 3, pchain = 3, mtk = yes, tloop = 1,
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
 * Reference cell and deviatoric stress, as native LAMMPS does (src/fix_nh.cpp at b332f28, checked with
 * tests/oracle/w358nh_*-w362nh_*): the reference volume vol0 and cell h0_inv are taken at the fix's first init and
 * kept for later runs (nreset N takes them again every N steps of a run, counted from its first step). The x, y,
 * z, xy, yz and xz keywords set deviatoric_flag: the barostat force then subtracts fdev = h sigma h^t, with
 * sigma = vol0 h0_inv (P_target - p_hydro) h0_inv^t, the scalar adds the strain energy 0.5 Tr(sigma h h^t), and the
 * vector ends with that PE_strain entry. The diagonal barostat force is (P - p_hydro) V (p_hydro the mean target
 * over the barostatted diagonal dimensions), the tilt force P V. These terms are the port in
 * src/engine-gpl/fix/nh_deviatoric.ts, loaded with the keywords that set the flag (lazy HOOKS x, y, z, xy, yz, xz).
 * The global vector follows fix_nh.cpp compute_vector: eta, eta_dot (tchain), omega, omega_dot (1 for iso, 3, or 6
 * with tilts), etap, etap_dot (pchain), PE_eta, KE_eta_dot, PE_omega (p_hydro (V - vol0), divided among the
 * barostatted diagonal dimensions), KE_omega_dot, PE_etap, KE_etap_dot, and PE_strain[1] with deviatoric_flag.
 * Without a thermostat the barostat temperature t0 is set at the first run and kept.
 *
 * Measured with native LAMMPS (black box, tests/oracle/w7nh_*.in and w42nhm_*.in):
 *   - PE_etap[k] = k_B T_p etap[k] for every k (T_p the barostat temperature); no count of barostatted dims.
 *   - the scalar is the sum of the KE_omega_dot, PE_omega, PE_strain, and barostat-chain entries (equal
 *     to ecouple with the thermo normalisation: lj units divide an extensive scalar by the atom count).
 *   - the barostat velocity update, the MTK terms and the thermostat chain match native to rel 1e-9 over
 *     100 steps (w42nhm_*).
 *   - the MTK force term (per barostatted dimension) is the sum over the barostatted diagonal
 *     dimensions of sum m v^2, divided by (number of barostatted dimensions) * N. The iso
 *     barostat (couple xyz) uses the temperature compute's dof * k_B T instead (rotational dof
 *     included, the sphere/asphere cases). The MTK trace term in the velocity update is the sum
 *     of omega_dot over the barostatted dimensions, divided by (number of barostatted dimensions) * N.
 *   - barostat velocity scaling: both scalings of a half step come before the kick in the initial step and
 *     after it in the final step.
 *   - the barostat chain force counts the barostatted dimensions (couple none).
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
 * Box flips: with flip yes, a tilt past 0.6 of its box length is flipped on a neighbor rebuild step (preExchange,
 * System.flipBox). update dipole is for the /sphere variants (nh_sphere.ts).
 */


type Dim = 0 | 1 | 2 | 3 | 4 | 5;   // x y z yz xz xy (Voigt order as LAMMPS documents for tensors)

const KEYS = ['x', 'y', 'z', 'yz', 'xz', 'xy'];

/**
 * The deviatoric barostat terms (src/engine-gpl/fix/nh_deviatoric.ts, a port of fix_nh.cpp compute_sigma,
 * compute_deviatoric and compute_strain_energy), loaded lazily under the HOOKS words x, y, z, xy, yz, xz: the keywords
 * that set native's deviatoric_flag. Tensors are Voigt (xx yy zz yz xz xy); h is (xprd, yprd, zprd, yz, xz, xy).
 */
export interface NHDeviatoricHook {
  boxHInv(h: ArrayLike<number>, triclinic: boolean): number[];
  computeSigma(vol0: number, h0Inv: ArrayLike<number>, pTarget: ArrayLike<number>, pHydro: number): number[];
  computeDeviatoric(h: ArrayLike<number>, sigma: ArrayLike<number>): number[];
  computeStrainEnergy(h: ArrayLike<number>, sigma: ArrayLike<number>, nktv2p: number): number;
}

/**
 * The thermostat and barostat variables of fix nvt/npt/nph that a binary restart file holds. fix_nh.html: "These
 * fixes writes the state of all the thermostat and barostat variables to binary restart files. See the read_restart
 * command for info on how to re-specify a fix in an input script that reads a restart file, so that the operation of
 * the fix continues in an uninterrupted fashion."
 */
/** utils::numeric: a floating-point number as LAMMPS reads one. */
const numeric = (w: string | undefined): number => {
  if (w === undefined || !/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(w)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return Number(w);
};

/** utils::inumeric. */
const inumeric = (w: string | undefined): number => {
  if (w === undefined || !/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  return Number(w);
};

/** utils::logical: yes, no, true, false, on, off, 1, 0. */
const logical = (w: string | undefined): boolean => {
  if (w === 'yes' || w === 'true' || w === 'on' || w === '1') return true;
  if (w === 'no' || w === 'false' || w === 'off' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in input script or data file`);
};

export interface NHRestart {
  eta: number[];
  etaDot: number[];
  /** the subclass's own variables (restartExtra), when it has any */
  extra?: Record<string, number>;
  etap: number[];
  etapDot: number[];
  omegaDot: number[];
  omegaPos: number[];
  ecoupleCum: number;
  /** the reference volume (fix_nh.cpp vol0) of the fix that wrote the file */
  vol0: number;
  /** the reference cell h0_inv (fix_nh.cpp, stored when deviatoric_flag is set) */
  h0Inv?: number[];
  /** the box lengths of the reference cell (files written before h0Inv; not used) */
  len0?: number[];
  /** the barostat reference temperature (fix_nh.cpp t0, the temperature of the masses of an nph run), when there is a barostat */
  t0?: number;
  /** isochoric: the compensating dimensions (fix_nh.cpp p_isoch) and the volume of the box at the fix's creation (vol_start) */
  pIsoch?: number[];
  volStart?: number;
}

export class FixNH extends Fix {
  readonly style: string;
  /** The fix style as written (nvt/eff, npt/sphere ...), for native's messages. */
  private readonly styleName: string;

  override get invokedStyle(): string {
    return this.styleName;
  }
  // thermostat
  private tstat = false;
  protected tStart = 0; protected tStop = 0; private tPeriod = 0;
  private tchain = 3; private tloop = 1;
  private eta: Float64Array; private etaDot: Float64Array;
  /** eta_dotdot of the chain (k >= 1 persists across calls, see nhcTemp). */
  private etaDotdot: Float64Array;
  // barostat
  private pstat = false;
  protected pStart = new Float64Array(6); protected pStop = new Float64Array(6); protected pPeriod = new Float64Array(6);
  protected pFlag = [false, false, false, false, false, false];
  private couple: 'none' | 'xyz' | 'xy' | 'yz' | 'xz' = 'none';
  private pchain = 3; private ploop = 1;
  private mtk = true;
  private drag = 0;
  private ptemp: number | null = null;
  protected dilateBit: number;
  /** fix_nh.cpp fixedpoint: the box centre when the fix is created, or the fixedpoint keyword */
  protected fixedpoint: number[] = [0, 0, 0];
  /**
   * fix_nh.html isochoric: "maintain constant volume when barostatting up to two dimensions"; the dimensions that
   * compensate (fix_nh.cpp p_isoch) and the volume (area in 2d) of the box when the fix was created (vol_start)
   */
  private isochoric = false;
  private pIsoch = [0, 0, 0];
  private volStart = 0;
  protected omegaDot = new Float64Array(6);
  protected omegaPos = new Float64Array(6);
  protected omegaMass = new Float64Array(6);
  private etap: Float64Array; private etapDot: Float64Array;
  /** Accelerations of the barostat chain elements k >= 1, stored between updates (used with chainMassTemp only). */
  private etapDotdot: Float64Array;
  protected pTarget = new Float64Array(6);
  /** fix_nh.cpp p_hydro: mean of the barostatted diagonal targets (the CauchyStat hook may change it). */
  protected pHydro = 0;
  private pCurrent = new Float64Array(6);
  protected tTarget = 0;
  private pTempFixed = 0;
  protected tempCompute!: Compute;
  protected pressCompute: Compute | null = null;
  /** kinetic tensor of the pressure's temperature compute at the initial step's pressure (see computePressCurrent). */
  private keInitial: Float64Array | null = null;
  private tempId: string;
  private pressId: string | null = null;
  protected dthalf = 0; protected dt4 = 0; private dt8 = 0; private dtf = 0; private dtv = 0;
  /**
   * The reference volume and cell (fix_nh.cpp vol0 and h0_inv): as native LAMMPS does (src/fix_nh.cpp init at b332f28),
   * they are taken at the first init of the fix ("if (vol0 == 0.0)") and kept for the later runs, unless nreset
   * resets them (computePressTarget) or a restart file gives them.
   */
  protected vol0 = 0;
  private h0Inv: number[] | null = null;
  /** fix_nh.cpp nreset_h0: the reference cell is reset every this many steps of a run (0: never) */
  private nresetH0 = 0;
  /** fix_nh.cpp sigma: the deviatoric target stress in the reference cell (set by computePressTarget) */
  private sigma: number[] = [0, 0, 0, 0, 0, 0];
  /**
   * fix_nh.cpp t0: the barostat temperature without a thermostat. As native LAMMPS does (src/fix_nh.cpp setup at
   * b332f28), it is set once ("if (t0 == 0.0)"), from ptemp or the current temperature, and kept for later runs; a
   * restart file gives it ("if it was read in from a restart file, leave it be").
   */
  private baroT0 = 0;
  private ecoupleCum = 0;
  protected scaleTilt = [true, true, true];   // yz xz xy with their second dimension

  /**
   * style is the family whose rules apply (nvt, npt, nph, nphug, npt/cauchy); name is the fix style the messages name
   * (nvt/sphere, npt/eff, ...), the family by default.
   */
  constructor(sys: System, id: string, group: string, args: string[], style: 'nvt' | 'npt' | 'nph' | 'nphug' | 'npt/cauchy', name: string = style) {
    super(sys, id, group, args);
    this.style = style;
    this.styleName = name;
    this.timeIntegrate = true;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.dilateBit = sys.groupBit('all');
    // As native LAMMPS does (src/fix_nh.cpp FixNH::FixNH at b332f28): the keywords in order, each read by utils::numeric,
    // inumeric or logical; then the checks of the whole command; the style's own checks (fix_nvt.cpp, fix_npt.cpp,
    // fix_nph.cpp, ...) come after them
    if (args.length < 1) throw new StyleError(`Illegal fix ${name} command: missing argument(s)`);
    const dimension = sys.dimension;
    const box = sys.state.box;
    const geom = sys.geom;
    // "turn on tilt factor scaling, whenever applicable": a periodic second dimension and a tilt that is not 0
    let scaleyz = false, scalexz = false, scalexy = false;
    if (box.periodic[1] && geom.xy !== 0.0) scalexy = true;
    if (box.periodic[2] && dimension === 3) {
      if (geom.yz !== 0.0) scaleyz = true;
      if (geom.xz !== 0.0) scalexz = true;
    }
    // "set fixed-point to default = center of cell"
    this.fixedpoint = [0.5 * (box.lo[0] + box.hi[0]), 0.5 * (box.lo[1] + box.hi[1]), 0.5 * (box.lo[2] + box.hi[2])];
    let flipflag = true;
    let dipole = false;
    let pTempFlag = false;
    // fix npt/cauchy (src/EXTRA-FIX/fix_npt_cauchy.cpp at b332f28) keeps the older wording: most bad words are
    // "Illegal fix npt/cauchy command"
    const cauchy = name === 'npt/cauchy';
    const old = (msg: string): StyleError => new StyleError(cauchy ? 'Illegal fix npt/cauchy command' : msg);
    const missing = (k: number, need: number, key: string): void => {
      if (k + need > args.length) throw old(`Illegal fix ${name} ${key} command: missing argument(s)`);
    };
    const P = (k: number): [number, number, number] => [numeric(args[k + 1]), numeric(args[k + 2]), numeric(args[k + 3])];
    for (let k = 0; k < args.length;) {
      const key = args[k];
      switch (key) {
        case 'temp': {
          missing(k, 4, key);
          this.tstat = true;
          this.tStart = numeric(args[k + 1]);
          // fix_nh.cpp constructor: "t_target = t_start" (what extract("t_target") gives before the first setup)
          this.tTarget = this.tStart;
          this.tStop = numeric(args[k + 2]);
          this.tPeriod = numeric(args[k + 3]);
          if (this.tStart <= 0.0 || this.tStop <= 0.0) throw new StyleError(`Target temperature for fix ${name} cannot be ${cauchy ? '0.0' : '<= 0.0'}`);
          if (this.tPeriod <= 0.0 && !cauchy) throw new StyleError(`Temperature damping for fix ${name} cannot be <= 0.0`);
          k += 4;
          break;
        }
        case 'iso': case 'aniso': case 'tri': {
          missing(k, 4, key);
          const [ps, pe, pd] = P(k);
          this.couple = key === 'iso' ? 'xyz' : 'none';
          if (key === 'tri') scalexy = scalexz = scaleyz = false;
          for (const d of dimension === 2 ? [0, 1] : [0, 1, 2]) this.setP(d as Dim, ps, pe, pd);
          if (dimension === 2) this.clearP(2);
          if (key === 'tri') {
            for (const d of [3, 4, 5]) this.setP(d as Dim, 0, 0, pd);
            if (dimension === 2) { this.clearP(3); this.clearP(4); }
          }
          k += 4;
          break;
        }
        case 'x': case 'y': case 'z': case 'xy': case 'yz': case 'xz': {
          missing(k, 4, key);
          const d = KEYS.indexOf(key) as Dim;
          const [ps, pe, pd] = P(k);
          this.setP(d, ps, pe, pd);
          // As native LAMMPS does (src/fix_nh.cpp at b332f28): each of these keywords sets deviatoric_flag, and a tilt
          // keyword turns off the scaling of its own tilt
          this.deviatoric = true;
          if (key === 'yz') scaleyz = false;
          if (key === 'xz') scalexz = false;
          if (key === 'xy') scalexy = false;
          k += 4;
          if (dimension === 2 && (key === 'z' || key === 'yz' || key === 'xz')) throw new StyleError(`Invalid fix ${name} command for a 2d simulation`);
          break;
        }
        case 'couple': {
          missing(k, 2, key);
          const c = args[k + 1];
          if (c !== 'xyz' && c !== 'xy' && c !== 'yz' && c !== 'xz' && c !== 'none') throw old(`Illegal fix ${name} couple option: ${c}`);
          this.couple = c;
          k += 2;
          break;
        }
        case 'drag':
          missing(k, 2, key);
          this.drag = numeric(args[k + 1]);
          if (this.drag < 0.0) throw old(`Invalid fix ${name} drag argument: ${this.drag}`);
          k += 2;
          break;
        case 'ptemp':
          missing(k, 2, key);
          this.ptemp = numeric(args[k + 1]);
          pTempFlag = true;
          if (this.ptemp <= 0.0) throw new StyleError(`Invalid fix ${name} ptemp argument: ${this.ptemp}`);
          k += 2;
          break;
        case 'dilate':
          missing(k, 2, key);
          if (args[k + 1] === 'all') this.dilateBit = sys.groupBit('all');
          else {
            if (sys.groups.find(args[k + 1]) < 0) throw new StyleError(cauchy ? 'Fix npt/cauchy dilate group ID does not exist' : `Fix ${name} dilate group ID ${args[k + 1]} does not exist`);
            this.dilateBit = sys.groupBit(args[k + 1]);
          }
          k += 2;
          break;
        case 'tchain':
          missing(k, 2, key);
          this.tchain = inumeric(args[k + 1]);
          if (this.tchain < 1) throw old(`Invalid fix ${name} tchain argument: ${this.tchain}`);
          k += 2;
          break;
        case 'pchain':
          missing(k, 2, key);
          this.pchain = inumeric(args[k + 1]);
          if (this.pchain < 0) throw old(`Invalid fix ${name} pchain argument: ${this.pchain}`);
          k += 2;
          break;
        case 'mtk': missing(k, 2, key); this.mtk = logical(args[k + 1]); k += 2; break;
        case 'tloop':
          missing(k, 2, key);
          this.tloop = inumeric(args[k + 1]);
          if (this.tloop < 0) throw old(`Invalid fix ${name} tloop argument: ${this.tloop}`);
          k += 2;
          break;
        case 'ploop':
          missing(k, 2, key);
          this.ploop = inumeric(args[k + 1]);
          if (this.ploop < 0) throw old(`Invalid fix ${name} ploop argument: ${this.ploop}`);
          k += 2;
          break;
        case 'nreset':
          missing(k, 2, key);
          this.nresetH0 = inumeric(args[k + 1]);
          if (this.nresetH0 < 0) throw old(`Invalid fix ${name} nreset argument: ${this.nresetH0}`);
          k += 2;
          break;
        case 'scalexy': missing(k, 2, key); scalexy = logical(args[k + 1]); k += 2; break;
        case 'scalexz': missing(k, 2, key); scalexz = logical(args[k + 1]); k += 2; break;
        case 'scaleyz': missing(k, 2, key); scaleyz = logical(args[k + 1]); k += 2; break;
        case 'flip': missing(k, 2, key); flipflag = logical(args[k + 1]); k += 2; break;
        case 'update':
          // the dipole update is parsed by the /sphere variants (nh_sphere.ts strips the keyword before it reaches here)
          missing(k, 2, key);
          if (args[k + 1] === 'dipole' || args[k + 1] === 'dipole/dlm') dipole = true;
          else throw new StyleError(`Invalid fix ${name} update argument: ${args[k + 1]}`);
          k += 2;
          break;
        case 'isochoric': {
          missing(k, 2, key);
          const w = args[k + 1];
          const sel: Record<string, number[]> = { x: [0], y: [1], z: [2], xy: [0, 1], yz: [1, 2], xz: [0, 2] };
          if (!sel[w]) throw new StyleError(`Illegal fix ${name} isochoric option: ${w}`);
          for (const d of sel[w]) this.pIsoch[d] = 1;
          this.isochoric = true;
          k += 2;
          break;
        }
        case 'fixedpoint':
          missing(k, 4, key);
          this.fixedpoint = [numeric(args[k + 1]), numeric(args[k + 2]), numeric(args[k + 3])];
          k += 4;
          break;
        // "disc keyword is also parsed in fix/nh/sphere", "keywords erate, strain, and ext are also parsed in
        // fix/nh/uef", "keywords psllod, peculiar, kick and integrator are parsed in fix/nvt/sllod": skipped here
        case 'disc': k += 1; break;
        case 'erate': case 'strain': k += 3; break;
        case 'ext': case 'psllod': case 'peculiar': case 'kick': case 'integrator': k += 2; break;
        default: throw old(`Unknown fix ${name} keyword: ${key}`);
      }
    }
    const pf = this.pFlag, per = box.periodic;
    const bad2d = `Invalid fix ${name} command for a 2d simulation`;
    if (dimension === 2 && (pf[2] || pf[3] || pf[4])) throw new StyleError(bad2d);
    if (dimension === 2 && (this.couple === 'yz' || this.couple === 'xz')) throw new StyleError(bad2d);
    if (dimension === 2 && (scalexz || scaleyz)) throw new StyleError(bad2d);
    const badP = `Invalid fix ${name} command pressure settings`;
    if (this.couple === 'xyz' && (!pf[0] || !pf[1])) throw new StyleError(badP);
    if (this.couple === 'xyz' && dimension === 3 && !pf[2]) throw new StyleError(badP);
    if (this.couple === 'xy' && (!pf[0] || !pf[1])) throw new StyleError(badP);
    if (this.couple === 'yz' && (!pf[1] || !pf[2])) throw new StyleError(badP);
    if (this.couple === 'xz' && (!pf[0] || !pf[2])) throw new StyleError(badP);
    // "require periodicity in tensile dimension"
    for (const d of [0, 1, 2]) if (pf[d] && !per[d]) throw new StyleError(cauchy ? 'Cannot use fix npt/cauchy on a non-periodic dimension' : `Cannot use fix ${name} on a non-periodic ${KEYS[d]} dimension`);
    // "require periodicity in 2nd dim of off-diagonal tilt component"
    if ((pf[3] && !per[2]) || (pf[4] && !per[2]) || (pf[5] && !per[1])) throw new StyleError(`Cannot use fix ${name} on a 2nd non-periodic dimension`);
    if (scaleyz && !per[2]) throw new StyleError(`Cannot use fix ${name} with yz scaling when z is non-periodic dimension`);
    if (scalexz && !per[2]) throw new StyleError(`Cannot use fix ${name} with xz scaling when z is non-periodic dimension`);
    if (scalexy && !per[1]) throw new StyleError(`Cannot use fix ${name} with xy scaling when y is non-periodic dimension`);
    if (pf[3] && scaleyz) throw new StyleError(`Cannot use fix ${name} with both yz dynamics and yz scaling`);
    if (pf[4] && scalexz) throw new StyleError(`Cannot use fix ${name} with both xz dynamics and xz scaling`);
    if (pf[5] && scalexy) throw new StyleError(`Cannot use fix ${name} with both xy dynamics and xy scaling`);
    if (!box.triclinic && (pf[3] || pf[4] || pf[5])) throw new StyleError(`Can not specify Pxy/Pxz/Pyz in fix ${name} with non-triclinic box`);
    // "Pstart, Pstop, Pdamp parameters for any coupled dimensions must be identical"
    const coupled = this.coupledDims();
    for (const d of coupled.slice(1)) {
      if (this.pStart[d] !== this.pStart[coupled[0]] || this.pStop[d] !== this.pStop[coupled[0]] || this.pPeriod[d] !== this.pPeriod[coupled[0]]) {
        throw new StyleError(`Invalid fix ${name} pressure settings`);
      }
    }
    if (dipole) throw new StyleError("Must use a '/sphere' Nose-Hoover fix style for updating dipoles");
    if (this.isochoric) {
      for (const d of [0, 1, 2]) if (pf[d] && this.pIsoch[d]) throw new StyleError('Cannot use barostated dimension as isochoric dimension.');
      const nbaro = (pf[0] ? 1 : 0) + (pf[1] ? 1 : 0) + (pf[2] ? 1 : 0);
      if (dimension === 3 && nbaro > 2) throw new StyleError('Cannot perform isochoric NPT with all dimensions barostated.');
      else if (dimension === 2 && (pf[0] ? 1 : 0) + (pf[1] ? 1 : 0) > 1) throw new StyleError('Cannot perform 2d isochoric NPT with all dimensions barostated.');
      else if (nbaro === 0) throw new StyleError('Isochoric simulation requires at least 1 diagonal dimension to be barostated.');
      const bad = 'Isochoric NPT requires periodic boundary conditions along isochoric dimensions.';
      if (dimension === 3) {
        if ((per[0] ? this.pIsoch[0] : 0) + (per[1] ? this.pIsoch[1] : 0) + (per[2] ? this.pIsoch[2] : 0) === 0) throw new StyleError(bad);
      } else {
        // native tests the x flag twice here ("domain->xperiodic * p_isoch[0] + domain->yperiodic * p_isoch[0]")
        if ((per[0] ? this.pIsoch[0] : 0) + (per[1] ? this.pIsoch[0] : 0) === 0) throw new StyleError(bad);
        if (this.pIsoch[2]) throw new StyleError('2d Isochoric NPT cannot use z dimension.');
      }
    }
    if ((this.tstat && this.tPeriod <= 0.0) || [0, 1, 2, 3, 4, 5].some((d) => pf[d] && this.pPeriod[d] <= 0.0)) {
      throw new StyleError(`Fix ${name} damping parameters must be > 0.0`);
    }
    // "check that ptemp is not defined with a thermostat"
    if (this.tstat && pTempFlag) throw new StyleError(`Thermostat in fix ${name} is incompatible with ptemp command`);
    this.scaleTilt = [scaleyz, scalexz, scalexy];
    // "In the isochoric case, dimensions can change size while not being barostated to maintain volume/area"
    if (this.pstat && this.isochoric) this.volStart = dimension === 3 ? geom.lx * geom.ly * geom.lz : geom.lx * geom.ly;
    // As native LAMMPS does (src/fix_nvt.cpp, fix_npt.cpp, fix_nph.cpp and their variants at b332f28)
    if ((style === 'nvt' || style === 'npt') && !this.tstat) throw new StyleError(`Temperature control must be used with fix ${name}`);
    if (style === 'nph' && this.tstat) throw new StyleError(`Temperature control can not be used with fix ${name}`);
    if (style === 'nvt' && this.pstat) throw new StyleError(`Pressure control can not be used with fix ${name}`);
    if ((style === 'npt' || style === 'nph') && !this.pstat) throw new StyleError(`Pressure control must be used with fix ${name}`);
    // fix_nphug.cpp: "Temperature control must be used with fix nphug" and "Pressure control must be used with fix nphug"
    if (style === 'nphug' && !this.tstat) throw new StyleError('Temperature control must be used with fix nphug');
    if (style === 'nphug' && !this.pstat) throw new StyleError('Pressure control must be used with fix nphug');
    // fix_npt_cauchy.cpp (EXTRA-FIX) constructor: "Temperature control must be used with fix npt/cauchy" and "Pressure control must be used with fix npt/cauchy"
    if (style === 'npt/cauchy' && !this.tstat) throw new StyleError('Temperature control must be used with fix npt/cauchy');
    if (style === 'npt/cauchy' && !this.pstat) throw new StyleError('Pressure control must be used with fix npt/cauchy');
    this.eta = new Float64Array(this.tchain);
    this.etaDot = new Float64Array(this.tchain + 1);
    this.etaDotdot = new Float64Array(this.tchain);
    this.etap = new Float64Array(Math.max(1, this.pchain));
    this.etapDotdot = new Float64Array(Math.max(1, this.pchain));
    this.etapDot = new Float64Array(Math.max(1, this.pchain) + 1);
    // a restart file holding this fix's state is applied when the fix is re-specified (restart.ts)
    // the variant styles (nvt/sphere, npt/asphere, ...) are set by their subclass after this constructor: a record
    // of such a variant belongs to this family
    const rec = sys.pendingFixData.get(id);
    const ofFamily = !!rec && (rec.style === undefined || rec.style === style || rec.style.startsWith(`${style}/`));
    const saved = ofFamily ? takeFixRecord(sys, id, rec.style ?? style)?.nh : undefined;
    if (saved) this.restoreState(saved);
    this.tempId = `${id}_temp`;
    this.tempCompute = ownCompute(sys, this.tempId, this.pstat ? 'all' : group, 'temp', []);
    this.vectorFlag = true;
    this.sizeVector = this.vectorLayout().reduce((a, b) => a + b.n, 0);
    if (this.pstat) {
      this.pressId = `${id}_press`;
      this.pressCompute = ownCompute(sys, this.pressId, 'all', 'pressure', [this.tempId]);
      this.virialGlobal = false;
      // As native LAMMPS does (src/fix_nh.cpp constructor at b332f28): "pre_exchange only required if flips can occur
      // due to shape changes", a tilt barostat or a tilted box with flip yes
      const g = sys.geom;
      // the hook is set only then, so that a fix without flips keeps the runs that take no pre_exchange (run/accel.ts);
      // a subclass with its own pre_exchange (fix nvt/npt/uef) keeps it, as native's overrides FixNH's
      if (flipflag && typeof this.preExchange !== 'function' && (this.pFlag[3] || this.pFlag[4] || this.pFlag[5] || g.yz !== 0 || g.xz !== 0 || g.xy !== 0)) {
        this.preExchange = () => this.flipIfNeeded();
      }
    }
  }

  /**
   * As native LAMMPS does (src/fix_nh.cpp pre_exchange at b332f28), on the steps the neighbor lists are rebuilt: a
   * tilt past (0.5 + 0.1) of its box length flips the box ("flip is only triggered when tilt exceeds 0.5 by
   * DELTAFLIP"): yz first, by one y length (which moves xz by xy), then xz and xy by one x length, along the periodic
   * dimensions only. fix_nh.html: "the tilt factors xy, xz, yz ... the box is flipped to an equivalent shape".
   */
  private flipIfNeeded(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const xprd = g.lx, yprd = g.ly;
    const xtiltmax = (0.5 + 0.1) * xprd, ytiltmax = (0.5 + 0.1) * yprd;
    let xy = g.xy, xz = g.xz, yz = g.yz;
    let m = 0, n = 0, p = 0;
    if (s.box.periodic[1]) {
      if (yz < -ytiltmax) { yz += yprd; xz += xy; p = 1; }
      else if (yz >= ytiltmax) { yz -= yprd; xz -= xy; p = -1; }
    }
    if (s.box.periodic[0]) {
      if (xz < -xtiltmax) { xz += xprd; n = 1; }
      else if (xz >= xtiltmax) { xz -= xprd; n = -1; }
      if (xy < -xtiltmax) { xy += xprd; m = 1; }
      else if (xy >= xtiltmax) { xy -= xprd; m = -1; }
    }
    if (m || n || p) this.sys.flipBox([xy, xz, yz], m, n, p, this);
  }

  private setP(d: Dim, ps: number, pe: number, pd: number): void {
    this.pstat = true;
    this.pFlag[d] = true;
    this.pStart[d] = ps; this.pStop[d] = pe; this.pPeriod[d] = pd;
  }

  /** A dimension iso, aniso or tri leaves unset in 2d ("p_start[2] = p_stop[2] = p_period[2] = 0.0; p_flag[2] = 0;"). */
  private clearP(d: Dim): void {
    this.pFlag[d] = false;
    this.pStart[d] = 0; this.pStop[d] = 0; this.pPeriod[d] = 0;
    this.pstat = this.pFlag.some((f) => f);
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

  /**
   * Kinetic tensor of the pressure's temperature compute taken at the initial step's pressure evaluation (null before the
   * first evaluation). Used by fix grem's PRESSURE/GREM compute, which reads the same stored value for its scalar.
   */
  get initialKinetic(): Float64Array | null { return this.keInitial; }

  /**
   * reset_target (temper, temper/npt: src/fix_nh.cpp FixNH::reset_target at b332f28): "t_target = t_start = t_stop = t_new".
   * The tempering commands call it after a swap of replicas (docs.lammps.org/temper.html: the temperature of the fix is
   * set to the one of the swapped replica).
   */
  resetTarget(tNew: number): void {
    // the barostat chain keeps its setup-time masses (native etap_mass_flag = 0), see baroMassTemp
    this.tStart = tNew; this.tStop = tNew; this.tTarget = tNew;
  }

  /**
   * extract (used by fix grem, docs.lammps.org/fix_nh.html's extract): "t_start" and "t_stop" are the thermostat
   * targets (null without a temp keyword); "p_start", "p_stop" and "p_flag" are the six barostat entries (null
   * without a barostat). Other names are null.
   */
  extract(name: string): number | number[] | null {
    if (name === 't_start' || name === 't_stop') return this.tstat ? (name === 't_start' ? this.tStart : this.tStop) : null;
    // fix_nh.cpp extract: "if (tstat_flag && strcmp(str,"t_target") == 0)" (compute hma reads it)
    if (name === 't_target') return this.tstat ? this.tTarget : null;
    if (name === 'p_start') return this.pstat ? Array.from(this.pStart) : null;
    if (name === 'p_stop') return this.pstat ? Array.from(this.pStop) : null;
    if (name === 'p_flag') return this.pstat ? this.pFlag.map((b) => (b ? 1 : 0)) : null;
    return null;
  }

  modify(key: string, values: string[]): number {
    // the messages of native's FixNH::modify_param (src/fix_nh.cpp at b332f28)
    if (key === 'temp') {
      const c = fixModifyCompute(this.sys, values[0], 'temp', 'Could not find fix_modify temperature ID {}', 'Fix_modify temperature ID {} does not compute temperature');
      this.tempId = values[0];
      this.tempCompute = c;
      // the fix's pressure compute uses the new temperature too
      if (this.pressCompute) (this.pressCompute as unknown as { tempId: string }).tempId = values[0];
      return 1;
    }
    if (key === 'press') {
      if (values.length < 1) throw missingArgs('fix_modify press');
      if (!this.pstat) throw new StyleError('Fix_modify press command without a barostat');
      const c = fixModifyCompute(this.sys, values[0], 'press', 'Could not find fix_modify pressure ID {}', 'Fix_modify pressure ID {} does not compute pressure');
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
    // As native LAMMPS does (src/fix_nh.cpp init at b332f28): "set initial volume and reference cell, if not already done"
    if (this.pstat && this.vol0 === 0) this.setReferenceCell();
    else if (this.pstat && this.deviatoric && this.h0Inv === null) this.h0Inv = this.deviatoricHook().boxHInv(this.boxH(), s.box.triclinic);
    // barostat target temperature
    this.tTarget = this.tstat ? ramp(this.sys, this.tStart, this.tStop) : this.currentTemp();
    if (this.pstat) {
      if (!this.tstat && this.baroT0 === 0) {
        if (this.ptemp !== null) this.baroT0 = this.ptemp;
        else {
          this.baroT0 = this.currentTemp();
          if (this.baroT0 < 1.0e-6) throw new StyleError('Current temperature too close to zero, consider using ptemp keyword');
        }
      }
      const tb = this.tstat ? this.tTarget : this.baroT0;
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
  protected tSetup: number | null = null;
  /**
   * fix_nh.cpp deviatoric_flag: set by the x, y, z, xy, yz and xz keywords (fix nphug turns it off). With it the
   * barostat force has the deviatoric term, the scalar the strain energy, and the vector its PE_strain entry.
   */
  protected deviatoric = false;
  /**
   * Kinetic energy 2 KE (energy units, = dof k_B T) tracked through the chain's last update of the previous step
   * (null after setup). The chain multiplies it by the square of each velocity factor it applies, as its own
   * update does, even though only the fix group is scaled (see the measurement in the header comment). For fix nph
   * (no chain) it is the all-atom kinetic energy at the end of the final step.
   */
  private trackKE: number | null = null;
  /**
   * Thermostat dof recorded at the initial update of the current step. A dynamic group is reassigned after the forces,
   * so the final update of that step must still use the dof that the initial update saw. Measured with native LAMMPS
   * (black box; 4x4x4 fcc lj/cut 2.5 at T 1.44, fix nvt temp 1 1 0.5 on group region half every 2 with 28 atoms
   * leaving the group at step 2): the members' velocities after step 2 differ from the engine's by a uniform factor
   * 1.00000226266, which is the change of the target term when it uses the dof of the step start (381 = 3*128-3) instead
   * of the dof after the reassignment (297 = 3*100-3); the 84 difference over q0 and dt gives the factor exactly.
   * The chain's inner elements (eta[1..], eta_dot[1..]) follow the same rule: as native LAMMPS does
   * (src/fix_nh.cpp nhc_temp_integrate at b332f28), the top sweep reads eta_dotdot[k] for k >= 1 as left by the previous
   * back sweep (or setup), computed with the eta_mass[0] of that call; only eta_dotdot[0] is recomputed from the
   * current kinetic energy. With the dof recomputed per call the inner elements drifted (temp 6.7e-5 relative at step
   * 200 of tests/oracle/w114cf_nylon_stab_long.in); with the stored values they match to about 3e-8 (see the cases).
   */
  private tdofStep: number | null = null;
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

  /**
   * Thermostat target temperature at setup (fix_nh.cpp setup(): compute_temp_target, which is the ramp). Fix nphug
   * overrides it: its masses use the initial temperature (fix_nphug.cpp setup()).
   */
  protected setupTargetTemp(): number { return ramp(this.sys, this.tStart, this.tStop); }

  /**
   * Thermostat target temperature for a step's first thermostat update (fix_nh.cpp initial_integrate's
   * compute_temp_target, the ramp). Fix nphug overrides it with the Hugoniot target.
   */
  protected stepTargetTemp(): number { return ramp(this.sys, this.tStart, this.tStop); }

  /**
   * Temperature the thermostat masses are built from, when a subclass fixes them at setup (native eta_mass_flag = 0,
   * fix_nphug.cpp); null: the masses follow the target temperature (fix nvt).
   */
  protected chainMassTemp: number | null = null;

  /**
   * Temperature the barostat chain's masses were built from at setup. As native LAMMPS does (src/fix_nh.cpp at
   * b332f28): etap_mass is set in setup() from the setup target and kept (etap_mass_flag = 0), while the thermostat
   * masses follow the target (eta_mass_flag = 1); the chain's forces use the current target (kt and lkt_press of
   * nhc_press_integrate), and its inner accelerations are the stored etap_dotdot. With a ramped thermostat target
   * (temp Tstart Tstop) the two temperatures differ. null without a barostat chain.
   */
  private baroMassTemp: number | null = null;
  /**
   * dof k_B T of the temperature compute at setup. As native LAMMPS does (src/fix_nh.cpp setup and nh_omega_dot at
   * b332f28): the first step's iso MTK term is tdof boltz t_current with the setup values (t_current scaled by the
   * thermostat's own factors), not a temperature computed again after the other fixes' initial integration.
   */
  private keSetup: number | null = null;
  /** The barostat temperature at setup (the masses' temperature, see baroMassTemp). */
  private baroSetupTemp = 0;

  /**
   * The current temperature as native fix_nh.cpp's t_current: the setup-time value before the first step, then the
   * kinetic temperature the thermostat chain left at the end of the previous step.
   */
  protected kineticTemp(): number {
    const kB = this.sys.state.units.boltz;
    if (this.tSetup !== null) return this.tSetup;
    if (this.trackKE !== null) return this.trackKE / (this.tempCompute.dof * kB);
    return this.currentTemp();
  }

  setup(): void {
    this.tSetup = this.currentTemp();
    this.keSetup = this.tempCompute.dof * this.sys.state.units.boltz * this.tSetup;
    this.trackKE = null;
    if (this.tstat) {
      this.tTarget = this.setupTargetTemp();
      // As native LAMMPS does (src/fix_nh.cpp setup() at b332f28): the chain's inner accelerations are set from the
      // setup-time masses, eta_dotdot[k] = (eta_mass[k-1] eta_dot[k-1]^2 - k_B T) / eta_mass[k], with
      // eta_mass[0] = dof k_B T / t_freq^2 (dof at setup).
      const kt = this.sys.state.units.boltz * this.tTarget;
      const tf2 = (1 / this.tPeriod) ** 2;
      const dof = this.tempCompute.dof;
      const mass = (k: number) => (k === 0 ? dof * kt / tf2 : kt / tf2);
      for (let k = 1; k < this.tchain; k++) {
        this.etaDotdot[k] = mass(k) > 0 ? (mass(k - 1) * this.etaDot[k - 1] * this.etaDot[k - 1] - kt) / mass(k) : 0;
      }
    }
    if (this.pstat) {
      this.computePressTarget();
      this.computePressCurrent();
      // As native LAMMPS does (src/fix_nh.cpp setup() at b332f28): the barostat chain's inner accelerations at setup,
      // etap_dotdot[k] = (etap_mass[k-1] etap_dot[k-1]^2 - k_B T) / etap_mass[k], masses from the setup temperature
      this.baroSetupTemp = this.baroTemp();
      this.baroMassTemp = this.pchain > 0 ? this.baroSetupTemp : null;
      if (this.pchain > 0) {
        const kt = this.sys.state.units.boltz * this.baroTemp();
        const pf = this.pfreqMax();
        const qk = kt / (pf * pf);
        for (let k = 1; k < this.pchain; k++) this.etapDotdot[k] = qk > 0 ? (qk * this.etapDot[k - 1] * this.etapDot[k - 1] - kt) / qk : 0;
      }
    }
  }

  /** Largest barostat frequency 1/Pdamp over the barostatted dimensions (nhc_press_integrate's p_freq_max). */
  private pfreqMax(): number {
    let f = 0;
    for (let d = 0; d < 6; d++) if (this.pFlag[d]) f = Math.max(f, 1 / this.pPeriod[d]);
    return f;
  }

  initialIntegrate(): void {
    // As native LAMMPS does (src/fix_nh.cpp initial_integrate at b332f28): the barostat chain runs before the thermostat
    // target is updated, so it sees the previous step's target (a ramped target, or fix nphug's Hugoniot target).
    const chainFirst = this.barostatChainFirst() && this.pstat && this.pchain > 0;
    if (chainFirst) this.nhcPress();
    if (this.tstat) {
      this.tTarget = this.stepTargetTemp();
      this.nhcTemp(true);
    }
    if (this.pstat) {
      this.computePressTarget();
      if (this.pchain > 0 && !chainFirst) this.nhcPress();
      this.computePressCurrent('init');
      this.omegaUpdate(true);
      // barostat velocity scaling (dt/4 each) on both sides of the half kick is applied before it
      this.barostatVelocityFlow();
    }
    this.kick();
    if (this.pstat) this.remap();
    this.drift();
    if (this.pstat) this.remap();
  }

  finalIntegrate(): void {
    this.kick();
    if (this.pstat) {
      this.barostatVelocityFlow();
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
    const { v, f, mask } = s;
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
    // the target term (dofT * kT) uses the dof of the membership at the start of the step: a dynamic group is
    // reassigned after the forces, so the final update still sees the old count (see tdofStep)
    if (initial) this.tdofStep = dof;
    const dofT = !initial && this.tdofStep !== null ? this.tdofStep : dof;
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
    const qT = this.chainMassTemp !== null ? kB * this.chainMassTemp : kt;
    const q = new Float64Array(M);
    q[0] = dof * qT / (tfreq * tfreq);
    for (let k = 1; k < M; k++) q[k] = qT / (tfreq * tfreq);
    const nc = this.tloop;
    const dthalf = this.dthalf / nc, dt4 = this.dt4 / nc, dt8 = this.dt8 / nc;
    let ke2 = dof * kTcur;   // 2 * KE of the thermostatted dof
    const ed = this.etaDot;
    const drag = this.drag > 0 ? 1 - this.dtv * tfreq * this.drag / nc : 1;
    for (let loop = 0; loop < nc; loop++) {
      // top of chain down
      // the inner accelerations (k >= 1) are the stored ones: native reads eta_dotdot[k] as left by the previous back
      // sweep, computed with the eta_mass[0] of that call (src/fix_nh.cpp nhc_temp_integrate at b332f28)
      const g0 = () => (ke2 - dofT * kt) / q[0];
      ed[M - 1] += (M === 1 ? g0() : this.etaDotdot[M - 1]) * dt4;
      ed[M - 1] *= drag;
      for (let k = M - 2; k >= 0; k--) {
        const ex = libm.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex;
        ed[k] += (k === 0 ? g0() : this.etaDotdot[k]) * dt4;
        ed[k] *= drag;
        ed[k] *= ex;
      }
      // scale velocities
      const factor = libm.exp(-dthalf * ed[0]);
      this.scaleVelocities(factor);
      ke2 *= factor * factor;
      this.nhcScale2 *= factor * factor;
      for (let k = 0; k < M; k++) this.eta[k] += dthalf * ed[k];
      // back up the chain
      for (let k = 0; k < M - 1; k++) {
        const ex = libm.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex;
        if (k > 0) this.etaDotdot[k] = (q[k - 1] * ed[k - 1] * ed[k - 1] - kt) / q[k];
        ed[k] += (k === 0 ? g0() : this.etaDotdot[k]) * dt4;
        ed[k] *= ex;
      }
      if (M > 1) this.etaDotdot[M - 1] = (q[M - 2] * ed[M - 2] * ed[M - 2] - kt) / q[M - 1];
      ed[M - 1] += (M === 1 ? g0() : this.etaDotdot[M - 1]) * dt4;
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
    for (let d = 0; d < 6; d++) {
      if (!this.pFlag[d]) continue;
      kecurrent += this.omegaMass[d] * this.omegaDot[d] * this.omegaDot[d];
    }
    const pfreqMax = this.pfreqMax();
    const ndims = this.chainDims();
    // masses fixed at setup (native etap_mass_flag = 0; fix nphug's chainMassTemp, fix_nphug.cpp) and inner
    // accelerations stored between updates (etap_dotdot)
    const massT = this.chainMassTemp ?? this.baroMassTemp;
    const stored = massT !== null;
    const qT = stored ? kB * massT! : kt;
    const q = new Float64Array(M);
    q[0] = qT / (pfreqMax * pfreqMax);
    for (let k = 1; k < M; k++) q[k] = qT / (pfreqMax * pfreqMax);
    const nc = this.ploop;
    const dthalf = this.dthalf / nc, dt4 = this.dt4 / nc, dt8 = this.dt8 / nc;
    const ed = this.etapDot;
    const pdrag = this.drag > 0 ? 1 - this.dtv * pfreqMax * this.drag / nc : 1;
    const g = (k: number) => (k === 0 ? (kecurrent - ndims * kt) / q[0] : (q[k - 1] * ed[k - 1] * ed[k - 1] - kt) / q[k]);
    // the top sweep reads the stored accelerations of the inner elements (k >= 1), as the back sweep left them
    const gTop = (k: number) => (stored && k > 0 ? this.etapDotdot[k] : g(k));
    for (let loop = 0; loop < nc; loop++) {
      ed[M - 1] += gTop(M - 1) * dt4;
      ed[M - 1] *= pdrag;
      for (let k = M - 2; k >= 0; k--) {
        const ex = libm.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex; ed[k] += gTop(k) * dt4; ed[k] *= pdrag; ed[k] *= ex;
      }
      const factor = libm.exp(-dthalf * ed[0]);
      for (let d = 0; d < 6; d++) if (this.pFlag[d]) this.omegaDot[d] *= factor;
      kecurrent *= factor * factor;
      for (let k = 0; k < M; k++) this.etap[k] += dthalf * ed[k];
      for (let k = 0; k < M - 1; k++) {
        const ex = libm.exp(-dt8 * ed[k + 1]);
        ed[k] *= ex;
        // the stored inner accelerations are tracked in every case (they are read only once the masses are fixed)
        if (k > 0) this.etapDotdot[k] = g(k);
        ed[k] += g(k) * dt4; ed[k] *= ex;
      }
      if (M > 1) this.etapDotdot[M - 1] = g(M - 1);
      ed[M - 1] += g(M - 1) * dt4;
    }
  }

  /**
   * fix_nh.cpp compute_press_target: the ramped targets, then p_hydro as their mean over the barostatted diagonal
   * dimensions, then the subclass hook (pressTargetSet).
   */
  /**
   * The pressure tensor (Voigt xx yy zz xy xz yz) the barostat reads for the current step. The default is the
   * compute's own tensor; a subclass whose pressure carries a correction to the diagonal (fix bocs,
   * src/BOCS/fix_bocs.cpp couple(): the ISO pressure is the scalar, which includes the correction) overrides it.
   */
  protected pressureTensor(pc: ComputePressure): ArrayLike<number> {
    // fix_nh.cpp couple(): an ISO barostat (couple xyz, or xy in 2d) reads compute_scalar(), allowed with MSM's
    // kspace_modify pressure/scalar yes; the diagonal mean of the tensor is the same scalar
    // (fix_modify press can name another pressure compute, e.g. pressure/alchemy, which has no such check)
    const iso = this.couple === 'xyz' || (this.couple === 'xy' && this.sys.dimension === 2);
    if (iso && pc instanceof ComputePressure) return pc.isoVectorValues();
    return pc.vectorValues();
  }

  /** The iso barostat's pressure: the compute's scalar (fix bocs reads its corrected tensor instead). */
  protected isoPressure(pc: ComputePressure): number {
    return pc.scalarValue();
  }

  protected computePressTarget(): void {
    for (let d = 0; d < 6; d++) {
      if (this.pFlag[d]) this.pTarget[d] = ramp(this.sys, this.pStart[d], this.pStop[d]);
    }
    let acc = 0;
    for (let d = 0; d < 3; d++) if (this.pFlag[d]) acc += this.pTarget[d];
    const nb = this.nbaroDiag();
    this.pHydro = nb > 0 ? acc / nb : 0;
    this.pressTargetSet();
    // "if deviatoric, recompute sigma each time p_target changes" (src/fix_nh.cpp compute_press_target at b332f28)
    if (this.deviatoric) this.computeSigma();
  }

  /** Hook after the targets of a step are set (fix npt/cauchy changes them here). */
  protected pressTargetSet(): void {}

  /** The deviatoric helpers, loaded with the keyword that set the flag (gpl_nh_deviatoric). */
  private deviatoricHook(): NHDeviatoricHook {
    return gplHook<NHDeviatoricHook>('x');
  }

  /** The box h of LAMMPS' domain: xprd, yprd, zprd, yz, xz, xy. */
  private boxH(): number[] {
    const g = this.sys.geom;
    return [g.lx, g.ly, g.lz, g.yz, g.xz, g.xy];
  }

  /** fix_nh.cpp vol0 and h0_inv from the current box (xprd yprd zprd, or xprd yprd in 2d). */
  private setReferenceCell(): void {
    const g = this.sys.geom;
    this.vol0 = this.sys.state.dimension === 3 ? g.lx * g.ly * g.lz : g.lx * g.ly;
    if (this.deviatoric) this.h0Inv = this.deviatoricHook().boxHInv(this.boxH(), this.sys.state.box.triclinic);
  }

  /**
   * fix_nh.cpp compute_sigma: with nreset > 0 the reference cell is taken again on every nreset-th step of the run
   * (counted from the run's begin step), then sigma from the reference cell and the current targets.
   */
  private computeSigma(): void {
    if (this.nresetH0 > 0 && (this.sys.state.step - this.sys.run.beginStep) % this.nresetH0 === 0) this.setReferenceCell();
    this.sigma = this.deviatoricHook().computeSigma(this.vol0, this.h0Inv!, this.pTarget, this.pHydro);
  }

  /**
   * Hook: whether the kinetic tensor of the temperature compute is the initial step's one at the final half step. Native
   * fix npt/cauchy (src/EXTRA-FIX/fix_npt_cauchy.cpp final_integrate at b332f28) calls pressure->compute_vector() but
   * not temperature->compute_vector() there, so LAMMPS' per-timestep cache of the temperature compute keeps the
   * vector of the initial step; fix nh (src/fix_nh.cpp final_integrate) refreshes it. Default: refreshed.
   */
  protected staleKineticFinal(): boolean { return false; }

  /** Hook: whether the barostat chain runs before the thermostat target update of the step (fix_nh.cpp order). */
  protected barostatChainFirst(): boolean { return this.chainMassTemp !== null || this.baroMassTemp !== null; }

  /** fix_nh.cpp compute_strain_energy: 0.5 Tr(sigma h h^t) / nktv2p (deviatoric_flag only). */
  protected strainEnergyValue(): number {
    return this.deviatoricHook().computeStrainEnergy(this.boxH(), this.sigma, this.sys.state.units.nktv2p);
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
    // a pressure compute without a temperature compute (compute pressure/alchemy, fix alchemy's mixed tensor, which
    // already holds the kinetic term of both partitions: compute_pressure_alchemy.cpp at b332f28) has no tempId
    const pressTemp = pc.tempId ?? null;
    // A subclass whose kinetic tensor is not refreshed between the initial and the final half step (see
    // staleKineticFinal) reads the initial step's tensor for the final pressure as well.
    const stale = this.staleKineticFinal();
    const kt = pressTemp !== null && pressTemp !== this.tempId ? this.sys.compute(pressTemp) : stale ? this.tempCompute : null;
    if (kt && kt.vectorFlag) {
      if (stage === 'init') this.keInitial = Float64Array.from(kt.vectorValues());
      else if (stage === 'final' && this.keInitial) {
        pc.keOverride = this.keInitial;
        this.sys.refreshComputes();
      }
    }
    const v = this.pressureTensor(pc);
    // tensor Voigt order of compute pressure: xx yy zz xy xz yz -> ours x y z yz xz xy
    const cur = [v[0], v[1], v[2], v[5], v[4], v[3]];
    const coupled = this.coupledDims();
    if (this.ndofDims().length === 1 && !kt) {
      // As native LAMMPS does (src/fix_nh.cpp couple at b332f28): the iso barostat (pstyle ISO) reads the pressure
      // compute's scalar, whose kinetic term is dof k_B T of the temperature compute. It differs from the mean of the
      // tensor's diagonal when the temperature's tensor does not split its scalar evenly (compute temp/eff adds the
      // electron radial energy to every diagonal entry).
      const p = this.isoPressure(pc);
      for (const d of coupled) cur[d] = p;
    } else if (coupled.length) {
      let avg = 0;
      for (const d of coupled) avg += cur[d];
      avg /= coupled.length;
      for (const d of coupled) cur[d] = avg;
    }
    for (let d = 0; d < 6; d++) this.pCurrent[d] = cur[d];
    // As native LAMMPS does (src/fix_nh.cpp couple at b332f28): a pressure that is not finite stops the run
    const unstable = 'Non-numeric pressure - simulation unstable\nFor more information see https://docs.lammps.org/err0006';
    if (!Number.isFinite(cur[0]) || !Number.isFinite(cur[1]) || !Number.isFinite(cur[2])) throw new StyleError(unstable);
    if ((this.pFlag[3] || this.pFlag[4] || this.pFlag[5]) && (!Number.isFinite(cur[3]) || !Number.isFinite(cur[4]) || !Number.isFinite(cur[5]))) {
      throw new StyleError(unstable);
    }
    if (stage === 'final' && this.couple !== 'none' && !stale) {
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
          // the first update of a step: the tracked kinetic energy (the setup value in the first step), after the
          // chain's own scaling
          const tracked = this.trackKE ?? this.keSetup;
          const ke2 = initial && tracked !== null ? tracked * this.nhcScale2 : dof * kT;
          if (dof > 0) mtkTerm1 = ke2 / (nb * s.n);
        } else {
          mtkTerm1 = this.keBaro(!initial && this.staleKineticFinal()) / (nb * s.n);
        }
      }
    }
    // As native LAMMPS does (src/fix_nh.cpp init at b332f28): "pdrag_factor = 1.0 - (update->dt * p_freq_max * drag /
    // nc_pchain)", p_freq_max the largest 1/Pdamp; 0/0 (no drag, ploop 0) makes it NaN, which reaches the pressure
    const drag = 1.0 - (this.dtv * this.pfreqMax() * this.drag / this.ploop);
    // As native LAMMPS does (src/fix_nh.cpp nh_omega_dot at b332f28): the diagonal force uses the hydrostatic target
    // p_hydro, the tilt force the current tilt stress alone, and with deviatoric_flag the deviatoric force
    // fdev = h sigma h^t of the current box enters both.
    const fdev = this.deviatoric ? this.deviatoricHook().computeDeviatoric(this.boxH(), this.sigma) : null;
    for (let d = 0; d < 6; d++) {
      if (!this.pFlag[d]) continue;
      let force = (d < 3 ? this.pCurrent[d] - this.pHydro : this.pCurrent[d]) * vol / s.units.nktv2p;
      if (d < 3) force += mtkTerm1;
      if (fdev) force -= fdev[d] / s.units.nktv2p;
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
  private keBaro(stale = false): number {
    const tc = this.tempCompute;
    if (!tc.vectorFlag) return this.sys.state.units.mvv2e * this.mvvBaro();
    const ke = stale && this.keInitial ? this.keInitial : tc.vectorValues();
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

  /**
   * Hook: the barostat velocity flow of a half step. As native LAMMPS does (src/fix_nh.cpp at b332f28, FixNH::initial_integrate
   * and final_integrate call nh_v_press() once per half step), a subclass whose velocity sweep differs overrides it.
   */
  protected barostatVelocityFlow(): void {
    this.velocityPress();
  }

  /** The MTK velocity term: trace of the barostat velocities over the barostatted diagonal dimensions (0 without mtk). */
  protected mtkVelocityTerm(): number {
    const s = this.sys.state;
    if (!this.mtk) return 0;
    const dof = this.tempCompute.dof;
    if (!(dof > 0)) return 0;
    let tr = 0;
    for (let d = 0; d < (s.dimension === 2 ? 2 : 3); d++) if (this.pFlag[d]) tr += this.omegaDot[d];
    // As native LAMMPS does (src/fix_nh.cpp nh_omega_dot at b332f28): "if (pdim > 0) mtk_term2 /= pdim * atom->natoms;"
    // (a barostat on tilts alone has no diagonal dimension and a zero term)
    const pdim = this.nbaroDiag();
    return pdim > 0 ? tr / (pdim * s.n) : tr;
  }

  private velocityPress(): void {
    const s = this.sys.state;
    const mtkTerm2 = this.mtkVelocityTerm();
    // As native LAMMPS does (src/fix_nh.cpp at b332f28, FixNH::nh_v_press): factor[d] = exp(-dt/4 (omega_dot[d] +
    // mtk_term2)) scales the diagonal velocities before and after the tilt coupling, and the coupling of the
    // triclinic case uses the full half step dt/2: vx += -dt/2 (vy w5 + vz w4), vy += -dt/2 vz w3 (vz has no partner).
    const f = [0, 1, 2].map((d) => (s.dimension === 2 && d === 2 ? 1 : libm.exp(-this.dt4 * (this.omegaDot[d] + mtkTerm2))));
    const c = this.tempCompute;
    const bias = c.hasBias();
    if (bias) { c.computeBias(); c.removeBiasAll(); }
    const tri = this.pFlag[3] || this.pFlag[4] || this.pFlag[5];
    const wyz = this.omegaDot[3], wxz = this.omegaDot[4], wxy = this.omegaDot[5];
    const dthalf = this.dthalf;
    const v = s.v;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const k = 3 * i;
      v[k] *= f[0];
      v[k + 1] *= f[1];
      v[k + 2] *= f[2];
      if (tri) {
        v[k] += -dthalf * (v[k + 1] * wxy + v[k + 2] * wxz);
        v[k + 1] += -dthalf * v[k + 2] * wyz;
      }
      v[k] *= f[0];
      v[k + 1] *= f[1];
      v[k + 2] *= f[2];
    }
    if (bias) c.restoreBiasAll();
  }

  /**
   * Box and dilated atoms over dt/2: h <- exp(omega dt/2) h about the fixed
   * point (box centre by default), atoms mapped in fractional coordinates.
   */
  protected remap(): void {
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
    const fp = this.fixedpoint;
    // As native LAMMPS does (src/fix_nh.cpp at b332f28, FixNH::remap): the tilts are swept in place in the
    // upper-triangular h (h[1], h[2] the lengths before the remap, as native refreshes them only in
    // set_global_box). The first half is the sweep xz, yz, xy, xz with factors dto/8, dto/4, dto/4, dto/8 and
    // increments dto/4, dto/2, dto/2, dto/4; then the diagonal lengths scale about the fixed point with
    // exp(dto omega_dot) and the non-barostatted tilts follow their scale flags; the same sweep then runs again.
    const dto = this.dthalf, dto2 = dto / 2, dto4 = dto / 4, dto8 = dto / 8;
    const w = this.omegaDot;
    const h = [g.lx, g.ly, g.lz, b.tilt[2], b.tilt[1], b.tilt[0]];
    const sweep = (): void => {
      if (this.pFlag[4]) {
        const e = libm.exp(dto8 * w[0]);
        h[4] *= e; h[4] += dto4 * (w[5] * h[3] + w[4] * h[2]); h[4] *= e;
      }
      if (this.pFlag[3]) {
        const e = libm.exp(dto4 * w[1]);
        h[3] *= e; h[3] += dto2 * (w[3] * h[2]); h[3] *= e;
      }
      if (this.pFlag[5]) {
        const e = libm.exp(dto4 * w[0]);
        h[5] *= e; h[5] += dto2 * (w[5] * h[1]); h[5] *= e;
      }
      if (this.pFlag[4]) {
        const e = libm.exp(dto8 * w[0]);
        h[4] *= e; h[4] += dto4 * (w[5] * h[3] + w[4] * h[2]); h[4] *= e;
      }
    };
    const expo = [0, 0, 0];
    for (let d = 0; d < 3; d++) expo[d] = this.pFlag[d] ? libm.exp(dto * w[d]) : 1;
    sweep();
    for (let d = 0; d < 3; d++) {
      if (!this.pFlag[d]) continue;
      b.lo[d] = fp[d] + (b.lo[d] - fp[d]) * expo[d];
      b.hi[d] = fp[d] + (b.hi[d] - fp[d]) * expo[d];
    }
    if (this.pFlag[1] && this.scaleTilt[2]) h[5] *= expo[1];
    if (this.pFlag[2]) {
      if (this.scaleTilt[1]) h[4] *= expo[2];
      if (this.scaleTilt[0]) h[3] *= expo[2];
    }
    if (this.isochoric) {
      // As native LAMMPS does (src/fix_nh.cpp remap at b332f28): isofac = vol_start over the barostatted lengths (after
      // their scaling) and over the current lengths of every other dimension, its square root when two dimensions
      // compensate; the compensating dimensions scale by it about the fixed point, with the tilts of their scale flags
      let isofac = this.volStart;
      for (let d = 0; d < 3; d++) if (this.pFlag[d]) isofac /= b.hi[d] - b.lo[d];
      for (let d = 0; d < 3; d++) if (this.pIsoch[d] || !this.pFlag[d]) isofac /= b.hi[d] - b.lo[d];
      if (this.pIsoch[0] + this.pIsoch[1] + this.pIsoch[2] === 2) isofac = Math.sqrt(isofac);
      for (let d = 0; d < 3; d++) {
        if (!this.pIsoch[d]) continue;
        b.lo[d] = (b.lo[d] - fp[d]) * isofac + fp[d];
        b.hi[d] = (b.hi[d] - fp[d]) * isofac + fp[d];
      }
      if (this.pIsoch[1] && this.scaleTilt[2]) h[5] *= isofac;
      if (this.pIsoch[2]) {
        if (this.scaleTilt[1]) h[4] *= isofac;
        if (this.scaleTilt[0]) h[3] *= isofac;
      }
    }
    sweep();
    b.tilt[0] = h[5];
    b.tilt[1] = h[4];
    b.tilt[2] = h[3];
    // As native LAMMPS does (src/fix_nh.cpp remap at b332f28): "tilt factor to cell length ratio can not exceed TILTMAX
    // in one step" (TILTMAX 1.5, against the lengths before this remap); larger tilts are flipped in preExchange
    const TILTMAX = 1.5;
    if (h[3] < -TILTMAX * h[1] || h[3] > TILTMAX * h[1] || h[4] < -TILTMAX * h[0] || h[4] > TILTMAX * h[0] ||
      h[5] < -TILTMAX * h[0] || h[5] > TILTMAX * h[0]) {
      throw new StyleError(`Fix ${this.styleName} has tilted box too far in one step - periodic cell is too far from equilibrium state`);
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

  /** The volume of LAMMPS' compute_scalar and compute_vector: xprd yprd zprd, or xprd yprd in 2d. */
  private boxVolume(): number {
    const g = this.sys.geom;
    return this.sys.state.dimension === 3 ? g.lx * g.ly * g.lz : g.lx * g.ly;
  }

  /** Energy exchanged with the reservoirs (extended-system terms). */
  /** The chain and barostat variables, as the restart file stores them. */
  restartState(): NHRestart {
    const r: NHRestart = {
      eta: Array.from(this.eta), etaDot: Array.from(this.etaDot), etap: Array.from(this.etap), etapDot: Array.from(this.etapDot),
      omegaDot: Array.from(this.omegaDot), omegaPos: Array.from(this.omegaPos), ecoupleCum: this.ecoupleCum,
      vol0: this.vol0,
    };
    if (this.deviatoric && this.h0Inv) r.h0Inv = this.h0Inv.slice();
    // fix_nh.cpp pack_restart_data writes t0, which only a barostat without a thermostat sets (0 for npt)
    if (this.pstat) r.t0 = this.baroT0;
    if (this.pstat && this.isochoric) { r.pIsoch = this.pIsoch.slice(); r.volStart = this.volStart; }
    const extra = this.restartExtra();
    if (extra) r.extra = extra;
    return r;
  }

  /** The variables a subclass keeps beyond the NH ones (fix_nphug.cpp pack_restart_data); none for the base class. */
  protected restartExtra(): Record<string, number> | undefined { return undefined; }

  /** The subclass variables read from a restart file; the subclass applies them (applyRestoredExtra) before its setup. */
  protected restoredExtra: Record<string, number> | undefined;

  private restoreState(r: NHRestart): void {
    const same = (a: Float64Array, b: number[]) => a.length === b.length;
    // the barostat chain is stored only by a file written with a barostat (fix_nh.cpp pack_restart_data: pstat_flag)
    const barostat = this.pstat && r.etap.length > 0;
    // the thermostat chain is restored only for a fix with a thermostat (an nph run has none, so its chain is not used)
    const chain = this.tstat;
    if ((chain && (!same(this.eta, r.eta) || !same(this.etaDot, r.etaDot))) || (barostat && (!same(this.etap, r.etap) || !same(this.etapDot, r.etapDot)))) {
      throw new StyleError(`fix ${this.id}: the restart file was written with other tchain or pchain settings`);
    }
    if (chain) { this.eta.set(r.eta); this.etaDot.set(r.etaDot); }
    if (barostat) { this.etap.set(r.etap); this.etapDot.set(r.etapDot); }
    if (this.pstat && r.t0 !== undefined) this.baroT0 = r.t0;
    this.omegaDot.set(r.omegaDot); this.omegaPos.set(r.omegaPos);
    this.ecoupleCum = r.ecoupleCum;
    this.vol0 = r.vol0;
    if (r.h0Inv) this.h0Inv = r.h0Inv.slice();
    // fix_nh.cpp restart: a file with the isochoric flag gives p_isoch and vol_start (they act only with the keyword)
    if (r.pIsoch && r.volStart !== undefined) { this.pIsoch = r.pIsoch.slice(); this.volStart = r.volStart; }
    this.restoredExtra = r.extra;
  }

  ecouple(): number {
    const s = this.sys.state;
    const kB = s.units.boltz;
    let e = 0;
    if (this.tstat) {
      const kt = kB * this.tTarget;
      const tfreq = 1 / this.tPeriod;
      const dof = this.tempCompute.dof;
      // the masses of the stored chain (chainMassTemp: native eta_mass_flag = 0) or those of the target
      const kM = this.chainMassTemp !== null ? kB * this.chainMassTemp : kt;
      const q0 = dof * kM / (tfreq * tfreq), qk = kM / (tfreq * tfreq);
      e += dof * kt * this.eta[0] + 0.5 * q0 * this.etaDot[0] * this.etaDot[0];
      for (let k = 1; k < this.tchain; k++) e += kt * this.eta[k] + 0.5 * qk * this.etaDot[k] * this.etaDot[k];
    }
    if (this.pstat) {
      const vol = this.boxVolume();
      let pfreqMax = 0;
      // As native LAMMPS does (src/fix_nh.cpp compute_scalar at b332f28): each barostatted diagonal dimension adds
      // p_hydro (V - vol0) / pdim, with vol0 the fix's reference volume
      const pdim = this.nbaroDiag();
      for (let d = 0; d < 6; d++) {
        if (!this.pFlag[d]) continue;
        e += 0.5 * this.omegaDot[d] * this.omegaDot[d] * this.omegaMass[d];
        if (d < 3) e += this.pHydro * (vol - this.vol0) / (pdim * s.units.nktv2p);
        pfreqMax = Math.max(pfreqMax, 1 / this.pPeriod[d]);
      }
      let nflag = 0;
      for (let d = 0; d < 6; d++) if (this.pFlag[d]) nflag++;
      // fix_nh.cpp compute_scalar: "if (deviatoric_flag) energy += compute_strain_energy()" (inside pstat_flag)
      if (this.deviatoric) e += this.strainEnergyValue();
      if (this.pchain > 0) {
        const kt = kB * this.baroTemp();
        // etap_mass: fixed at setup (baroMassTemp, or fix nphug's chainMassTemp)
        const massT = this.chainMassTemp ?? this.baroMassTemp;
        const kM = massT !== null ? kB * massT : kt;
        const q0 = kM / (pfreqMax * pfreqMax), qk = kM / (pfreqMax * pfreqMax);
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
    const pdim = this.nbaroDiag();
    const dV = this.boxVolume() - this.vol0;
    const nktv2p = s.units.nktv2p;
    // As native LAMMPS does (src/fix_nh.cpp compute_vector at b332f28): the iso entry is p_hydro (V - vol0); the
    // anisotropic and triclinic entries are p_hydro (V - vol0) / pdim for each barostatted diagonal dimension and 0
    // otherwise (0 for the tilts); the iso KE_omega_dot entry is pdim times the first dimension's energy.
    const iso = dims.length === 1;
    const peOmega = (j: number): number => {
      if (iso) return this.pHydro * dV / nktv2p;
      return j < 3 && this.pFlag[j] ? this.pHydro * dV / (pdim * nktv2p) : 0;
    };
    const keOmega = (j: number): number => {
      if (iso) return pdim * 0.5 * this.omegaDot[0] * this.omegaDot[0] * this.omegaMass[0];
      return this.pFlag[j] ? 0.5 * this.omegaDot[j] * this.omegaDot[j] * this.omegaMass[j] : 0;
    };
    const kt = kB * this.tTarget;
    const ktP = kB * this.baroTemp();
    const tfreq = this.tstat ? 1 / this.tPeriod : 0;
    const dof = this.tstat ? this.tempCompute.dof : 0;
    const qT = (k: number) => (k === 0 ? dof * kt / (tfreq * tfreq) : kt / (tfreq * tfreq));
    let pfreq = 0;
    for (const d of dims) if (this.pFlag[d]) pfreq = Math.max(pfreq, 1 / this.pPeriod[d]);
    const massTP = this.chainMassTemp ?? this.baroMassTemp;
    const qP = () => (massTP !== null ? kB * massTP : ktP) / (pfreq * pfreq);
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
      { n: dims.length, value: (j) => keOmega(j) },
      { n: nP, value: (k) => ktP * this.etap[k] },
      { n: nP, value: (k) => 0.5 * qP() * this.etapDot[k] * this.etapDot[k] },
      // "PE_strain[1] = scalar strain energy (energy units)" (fix_nh.rst), with deviatoric_flag only (fix_nh.cpp)
      { n: this.pstat && this.deviatoric ? 1 : 0, value: () => this.strainEnergyValue() },
    ];
  }

}
