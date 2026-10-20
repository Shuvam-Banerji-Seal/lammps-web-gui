/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_press_langevin.cpp and src/fix_press_langevin.h at commit b332f28 (class FixPressLangevin: the
 * constructor, init, setup, initial_integrate, post_integrate (remap), post_force, end_of_step, couple_pressure,
 * couple_kinetic, couple_beta, reset_dt and modify_param; contributing author Germain Clavier (TUe)). Copyright
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Langevin barostat of docs.lammps.org/fix_press_langevin.html, with the equations of motion of Gronbech-Jensen and
 * Farago, J. Chem. Phys. 141, 194108 (2014). The fix moves only the box and the atoms' fractional coordinates; the
 * atoms are integrated by a separate time integration fix, as the doc requires. Supported keywords: iso, aniso, tri, x,
 * y, z, yz, xz, xy, couple, friction, dilate, temp, flip and fix_modify press. The tilt pistons keep native's indices:
 * piston 3 (the yz keyword) is driven by Pxy and moves xy, piston 4 (xz) by Pxz and moves xz, piston 5 (xy) by Pyz and
 * moves yz (couple_pressure takes the pressure tensor's xy, xz, yz in that order, and remap adds dilation[3] to xy and
 * dilation[5] to yz). With flip yes, a tilt past 0.6 of its box length flips the box on a neighbor rebuild step
 * (pre_exchange). The restart state (write_restart/restart) is not written by the browser engine.
 */
import { Fix } from '../../engine/fix/fix';
import { isDeformFix, type FixDeform } from '../../engine/fix/deform';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import { ownCompute, removeCompute, rampFraction } from '../../engine/fix/util';
import { RanMars } from '../random_mars';

type Couple = 'none' | 'xyz' | 'xy' | 'yz' | 'xz';

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

const INVALID_SETTINGS = 'Invalid fix press/langevin pressure settings';
const NON_PERIODIC = 'Cannot use fix press/langevin on a non-periodic dimension';
/** fix_press_langevin.cpp: a tilt over TILTMAX of its box length in one step is an error; a flip needs 0.5 + DELTAFLIP. */
const TILTMAX = 1.5;
const DELTAFLIP = 0.1;

export class FixPressLangevin extends Fix {
  readonly style = 'press/langevin';
  private readonly dimension: 2 | 3;
  private pcouple: Couple = 'none';
  private pstyle: 'iso' | 'aniso' | 'tri' = 'aniso';
  private allremap = true;
  /** 1 if the pressure is controlled in this dimension (x, y, z, yz, xz, xy). */
  private readonly pFlag: boolean[] = [false, false, false, false, false, false];
  private readonly tStart: number;
  private readonly tStop: number;
  private tTarget: number;
  private readonly pLtime: number;
  private readonly pFric = [0, 0, 0, 0, 0, 0];
  private readonly pAlpha = [0, 0, 0, 0, 0, 0];
  private readonly pStart = [0, 0, 0, 0, 0, 0];
  private readonly pStop = [0, 0, 0, 0, 0, 0];
  private readonly pPeriod = [0, 0, 0, 0, 0, 0];
  private readonly pMass = [0, 0, 0, 0, 0, 0];
  private readonly pTarget = [0, 0, 0, 0, 0, 0];
  private readonly pCurrent = [0, 0, 0, 0, 0, 0];
  private readonly pDeriv = [0, 0, 0, 0, 0, 0];
  private readonly dilation = [0, 0, 0, 0, 0, 0];
  private readonly fPiston = [0, 0, 0, 0, 0, 0];
  private readonly fOldPiston = [0, 0, 0, 0, 0, 0];
  private readonly gjfa = [0, 0, 0, 0, 0, 0];
  private readonly gjfb = [0, 0, 0, 0, 0, 0];
  private readonly fran = [0, 0, 0, 0, 0, 0];
  private readonly rng: RanMars;
  private readonly ownPressId: string;
  private pressId: string;
  private pressCompute!: Compute;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // native narg < 5 is fewer than one argument after "fix ID group press/langevin"
    if (args.length < 1) throw new StyleError('Illegal fix press/langevin command: missing argument(s)');
    this.nevery = 1;
    this.dimension = sys.dimension as 2 | 3;
    const dim = this.dimension;
    const pStart = this.pStart, pStop = this.pStop, pPeriod = this.pPeriod, pFlag = this.pFlag;
    let tStart = 0, tStop = 0, seed = 111111, pLtime = 0;
    let flipflag = true;

    // keyword values: iso/aniso/x/y/z take Pstart Pstop Pdamp; the native check is "iarg + 4 > narg"
    const need = (k: number, key: string): void => {
      if (k + 4 > args.length) throw new StyleError(`Illegal fix press/langevin ${key} command: missing argument(s)`);
    };
    const setDim = (d: number, k: number): void => {
      pStart[d] = numeric(args[k + 1]);
      pStop[d] = numeric(args[k + 2]);
      pPeriod[d] = numeric(args[k + 3]);
      pFlag[d] = true;
    };

    let k = 0;
    while (k < args.length) {
      const key = args[k];
      if (key === 'iso' || key === 'aniso') {
        need(k, key);
        this.pcouple = key === 'iso' ? 'xyz' : 'none';
        for (const d of [0, 1, 2]) setDim(d, k);
        if (dim === 2) { pStart[2] = pStop[2] = pPeriod[2] = 0.0; pFlag[2] = false; }
        k += 4;
      } else if (key === 'tri') {
        need(k, key);
        this.pcouple = 'none';
        for (const d of [0, 1, 2]) setDim(d, k);
        for (const d of [3, 4, 5]) {
          pStart[d] = pStop[d] = 0.0;
          pPeriod[d] = numeric(args[k + 3]);
          pFlag[d] = true;
        }
        if (dim === 2) {
          for (const d of [2, 3, 4]) { pStart[d] = pStop[d] = pPeriod[d] = 0.0; pFlag[d] = false; }
        }
        k += 4;
      } else if (key === 'x' || key === 'y' || key === 'z') {
        need(k, key);
        const d = key === 'x' ? 0 : key === 'y' ? 1 : 2;
        setDim(d, k);
        k += 4;
        if (key === 'z' && dim === 2) throw new StyleError('Fix press/langevin z option not allowed for a 2d simulation');
      } else if (key === 'yz' || key === 'xz' || key === 'xy') {
        need(k, key);
        setDim(key === 'yz' ? 3 : key === 'xz' ? 4 : 5, k);
        k += 4;
        // native's message for xz says "zz"
        if (dim === 2 && key !== 'xy') throw new StyleError(`Fix press/langevin ${key === 'yz' ? 'yz' : 'zz'} option not allowed for a 2d simulation`);
      } else if (key === 'flip') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix press/langevin flip command: missing argument(s)');
        const w = args[k + 1];
        if (['yes', 'on', 'true', '1'].includes(w)) flipflag = true;
        else if (['no', 'off', 'false', '0'].includes(w)) flipflag = false;
        else throw new StyleError(`Expected boolean parameter instead of '${w}' in input script or data file`);
        k += 2;
      } else if (key === 'couple') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix press/langevin couple command: missing argument(s)');
        const w = args[k + 1];
        if (w === 'xyz' || w === 'xy' || w === 'yz' || w === 'xz' || w === 'none') this.pcouple = w;
        else throw new StyleError(`Unknown fix press/langevin couple option: ${w}`);
        k += 2;
      } else if (key === 'friction') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix press/langevin friction command: missing argument(s)');
        pLtime = numeric(args[k + 1]);
        if (pLtime <= 0.0) throw new StyleError('Fix press/langevin friction value must be > 0');
        k += 2;
      } else if (key === 'dilate') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix press/langevin dilate command: missing argument(s)');
        const w = args[k + 1];
        if (w === 'all') this.allremap = true;
        else if (w === 'partial') this.allremap = false;
        else throw new StyleError(`Unknown fix press/langevin dilate option: ${w}`);
        k += 2;
      } else if (key === 'temp') {
        if (k + 4 > args.length) throw new StyleError('Illegal fix press/langevin temp command: missing argument(s)');
        tStart = numeric(args[k + 1]);
        tStop = numeric(args[k + 2]);
        seed = inumeric(args[k + 3]);
        if (seed <= 0) throw new StyleError('Fix press/langevin temp seed must be > 0');
        k += 4;
      } else {
        throw new StyleError(`Unknown fix press/langevin keyword: ${key}`);
      }
    }

    this.rng = new RanMars(seed);
    this.tStart = tStart;
    this.tStop = tStop;
    this.tTarget = tStart;
    this.pLtime = pLtime;

    // error checks (fix_press_langevin.cpp constructor, same order)
    const box = sys.state.box;
    if (dim === 2 && pFlag[2]) throw new StyleError('Invalid fix press/langevin command for a 2d simulation');
    if (dim === 2 && (this.pcouple === 'yz' || this.pcouple === 'xz')) {
      throw new StyleError('Invalid fix press/langevin command for a 2d simulation');
    }
    const pc = this.pcouple;
    if (pc === 'xyz' && (!pFlag[0] || !pFlag[1])) throw new StyleError(INVALID_SETTINGS);
    if (pc === 'xyz' && dim === 3 && !pFlag[2]) throw new StyleError(INVALID_SETTINGS);
    if (pc === 'xy' && (!pFlag[0] || !pFlag[1])) throw new StyleError(INVALID_SETTINGS);
    if (pc === 'yz' && (!pFlag[1] || !pFlag[2])) throw new StyleError(INVALID_SETTINGS);
    if (pc === 'xz' && (!pFlag[0] || !pFlag[2])) throw new StyleError(INVALID_SETTINGS);
    for (let d = 0; d < 3; d++) {
      if (pFlag[d] && !box.periodic[d]) throw new StyleError(NON_PERIODIC);
    }
    // require periodicity in 2nd dim of off-diagonal tilt component
    if (pFlag[3] && !box.periodic[2]) throw new StyleError('Cannot use fix press/langevin on a 2nd non-periodic dimension');
    if (pFlag[4] && !box.periodic[2]) throw new StyleError('Cannot use fix press/langevin on a 2nd non-periodic dimension');
    if (pFlag[5] && !box.periodic[1]) throw new StyleError('Cannot use fix press/langevin on a 2nd non-periodic dimension');
    if (!box.triclinic && (pFlag[3] || pFlag[4] || pFlag[5])) {
      throw new StyleError('Can not specify Pxy/Pxz/Pyz in fix press/langevin with non-triclinic box');
    }
    if (pc === 'xyz' && dim === 3 &&
      (pStart[0] !== pStart[1] || pStart[0] !== pStart[2] || pStop[0] !== pStop[1] || pStop[0] !== pStop[2] ||
        pPeriod[0] !== pPeriod[1] || pPeriod[0] !== pPeriod[2])) throw new StyleError(INVALID_SETTINGS);
    if (pc === 'xyz' && dim === 2 &&
      (pStart[0] !== pStart[1] || pStop[0] !== pStop[1] || pPeriod[0] !== pPeriod[1])) throw new StyleError(INVALID_SETTINGS);
    if (pc === 'xy' && (pStart[0] !== pStart[1] || pStop[0] !== pStop[1] || pPeriod[0] !== pPeriod[1])) {
      throw new StyleError(INVALID_SETTINGS);
    }
    if (pc === 'yz' && (pStart[1] !== pStart[2] || pStop[1] !== pStop[2] || pPeriod[1] !== pPeriod[2])) {
      throw new StyleError(INVALID_SETTINGS);
    }
    if (pc === 'xz' && (pStart[0] !== pStart[2] || pStop[0] !== pStop[2] || pPeriod[0] !== pPeriod[2])) {
      throw new StyleError(INVALID_SETTINGS);
    }
    if (tStart < 0.0 || tStop < 0.0) throw new StyleError('Fix press/langevin temperature parameters must be >= 0.0');
    for (let d = 0; d < 6; d++) {
      if (pFlag[d] && pPeriod[d] <= 0.0) throw new StyleError('Fix press/langevin damping parameters must be > 0.0');
    }

    // pstyle = TRICLINIC with a tilt piston, else ISO if XYZ coupling or XY coupling in 2d (one degree of freedom),
    // else ANISO
    if (pFlag[3] || pFlag[4] || pFlag[5]) this.pstyle = 'tri';
    else this.pstyle = pc === 'xyz' || (dim === 2 && pc === 'xy') ? 'iso' : 'aniso';

    // pre_exchange only when flips can occur due to shape changes (a fix without it keeps the runs that take none)
    const g0 = sys.geom;
    if (flipflag && (pFlag[3] || pFlag[4] || pFlag[5] || g0.yz !== 0 || g0.xz !== 0 || g0.xy !== 0)) {
      this.preExchange = () => this.flipIfNeeded();
    }

    // the pressure compute: virial only, the kinetic term is added by couple_kinetic (native "pressure NULL virial")
    this.ownPressId = `${id}_press`;
    this.pressId = this.ownPressId;
    this.pressCompute = ownCompute(sys, this.pressId, 'all', 'pressure', ['NULL', 'virial']);

    // GJF piston coefficients from the current dt (reset in setup) and the atom count at construction
    const u = sys.state.units;
    const kt = u.boltz * tStart;
    const nkt = (sys.state.n + 1) * kt;
    for (let i = 0; i < 6; i++) this.pFric[i] = pLtime > 0.0 ? pLtime : pPeriod[i];
    for (let i = 0; i < 6; i++) {
      this.pMass[i] = nkt * pPeriod[i] * pPeriod[i];
      this.pAlpha[i] = this.pMass[i] * this.pFric[i];
    }
    this.resetDt();
  }

  /** Piston coefficients of the GJF scheme for the current dt (fix_press_langevin.cpp reset_dt). */
  resetDt(): void {
    const dt = this.sys.state.dt;
    for (let i = 0; i < 6; i++) {
      this.gjfa[i] = (1.0 - this.pAlpha[i] * dt / 2.0 / this.pMass[i]) / (1.0 + this.pAlpha[i] * dt / 2.0 / this.pMass[i]);
      this.gjfb[i] = 1.0 / (1.0 + this.pAlpha[i] * dt / 2.0 / this.pMass[i]);
    }
  }

  init(): void {
    const sys = this.sys;
    // ensure no conflict with fix deform (the same component of the stress tensor)
    for (const f of sys.fixes) {
      if (!isDeformFix(f)) continue;
      const df = f as FixDeform;
      // p_flag and dimflag in Voigt order: x, y, z, then yz, xz, xy (the deform fix's tilts 2, 1, 0)
      for (let d = 0; d < 6; d++) {
        if (this.pFlag[d] && (d < 3 ? df.controlsLength(d) : df.controlsTilt(5 - d))) {
          throw new StyleError('Cannot use fix press/langevin and fix deform on same component of stress tensor');
        }
      }
    }
    const p = sys.compute(this.pressId);
    if (!p.pressFlag) throw new StyleError(`Compute ID ${this.pressId} for fix ${this.style} does not compute pressure`);
    this.pressCompute = p;
    // the piston derivatives and dilations start from zero in every run
    for (let i = 0; i < 6; i++) { this.pDeriv[i] = 0.0; this.dilation[i] = 0.0; }
  }

  /** setup (fix_press_langevin.cpp setup): only the dt-dependent coefficients; no pressure is computed here. */
  setup(): void {
    this.resetDt();
  }

  /** initial_integrate: the new target temperature, the random piston force and the dilation of this step. */
  initialIntegrate(): void {
    const s = this.sys.state;
    const dt = s.dt;
    // native delta = ntimestep - beginstep over the run length; ntimestep is already incremented here (verlet.ts)
    const delta = rampFraction(this.sys);
    this.tTarget = this.tStart + delta * (this.tStop - this.tStart);
    this.coupleBeta();
    const box = s.box;
    for (let i = 0; i < 6; i++) {
      if (!this.pFlag[i]) continue;
      // see equation 13 of Gronbech-Jensen and Farago (2014)
      let displacement = dt * this.pDeriv[i] * this.gjfb[i];
      displacement += 0.5 * dt * dt * this.fPiston[i] * this.gjfb[i] / this.pMass[i];
      displacement += 0.5 * dt * this.fran[i] * this.gjfb[i] / this.pMass[i];
      if (i < 3) {
        const dl = box.hi[i] - box.lo[i];
        this.dilation[i] = (dl + displacement) / dl;
      } else {
        this.dilation[i] = displacement;
      }
    }
  }

  /** post_integrate: remap the box and the (dilated) atoms, after every initial_integrate of the step. */
  postIntegrate(): void {
    this.remap();
  }

  /** remap (fix_press_langevin.cpp): atoms to lamda coordinates, the new box about its centre, and back. */
  private remap(): void {
    const sys = this.sys;
    const b = sys.state.box;
    const lo = [b.lo[0], b.lo[1], b.lo[2]];
    const hi = [b.hi[0], b.hi[1], b.hi[2]];
    for (let i = 0; i < 3; i++) {
      if (!this.pFlag[i]) continue;
      const oldlo = b.lo[i];
      const oldhi = b.hi[i];
      const ctr = 0.5 * (oldlo + oldhi);
      lo[i] = (oldlo - ctr) * this.dilation[i] + ctr;
      hi[i] = (oldhi - ctr) * this.dilation[i] + ctr;
    }
    // the tilt pistons as native's remap adds them: dilation[3] to xy, [4] to xz, [5] to yz
    const tilt = [b.tilt[0], b.tilt[1], b.tilt[2]];
    if (this.pFlag[3]) tilt[0] += this.dilation[3];
    if (this.pFlag[4]) tilt[1] += this.dilation[4];
    if (this.pFlag[5]) tilt[2] += this.dilation[5];
    // against the box lengths before this remap (native's domain->xprd and yprd are reset after the check)
    const xprd = b.hi[0] - b.lo[0], yprd = b.hi[1] - b.lo[1];
    if (tilt[2] < -TILTMAX * yprd || tilt[2] > TILTMAX * yprd || tilt[1] < -TILTMAX * xprd || tilt[1] > TILTMAX * xprd ||
      tilt[0] < -TILTMAX * xprd || tilt[0] > TILTMAX * xprd) {
      throw new StyleError('Fix press/langevin has tilted box too far in one step - periodic cell is too far from equilibrium state');
    }
    const remapBit = this.allremap ? sys.groupBit('all') : this.groupBit;
    sys.setBox(lo, hi, tilt, remapBit, this);
  }

  /**
   * pre_exchange (fix_press_langevin.cpp), on the steps the neighbor lists are rebuilt: a tilt past (0.5 + DELTAFLIP)
   * of its box length flips the box ("flip is only triggered when tilt exceeds 0.5 by DELTAFLIP"), yz first by one y
   * length (which moves xz by xy), then xz and xy by one x length, along the periodic dimensions only; the image flags
   * and the atoms follow (Domain::image_flip, remap).
   */
  private flipIfNeeded(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const xprd = g.lx, yprd = g.ly;
    const xtiltmax = (0.5 + DELTAFLIP) * xprd, ytiltmax = (0.5 + DELTAFLIP) * yprd;
    let xy = g.xy, xz = g.xz, yz = g.yz;
    let flipxy = 0, flipxz = 0, flipyz = 0;
    if (s.box.periodic[1]) {
      if (yz < -ytiltmax) { yz += yprd; xz += xy; flipyz = 1; }
      else if (yz >= ytiltmax) { yz -= yprd; xz -= xy; flipyz = -1; }
    }
    if (s.box.periodic[0]) {
      if (xz < -xtiltmax) { xz += xprd; flipxz = 1; }
      else if (xz >= xtiltmax) { xz -= xprd; flipxz = -1; }
      if (xy < -xtiltmax) { xy += xprd; flipxy = 1; }
      else if (xy >= xtiltmax) { xy -= xprd; flipxy = -1; }
    }
    if (flipxy || flipxz || flipyz) this.sys.flipBox([xy, xz, yz], flipxy, flipxz, flipyz, this);
  }

  /** post_force: the instantaneous pressure (virial plus kinetic term) and the piston force. */
  postForce(): void {
    const sys = this.sys;
    const delta = rampFraction(sys);
    sys.refreshComputes();
    const pc = this.pressCompute;
    const tensor = this.pstyle === 'iso' ? null : pc.vectorValues();
    const scalar = this.pstyle === 'iso' ? pc.scalarValue() : 0;
    this.couplePressure(scalar, tensor);
    this.coupleKinetic();
    for (let i = 0; i < 6; i++) {
      if (!this.pFlag[i]) continue;
      this.fOldPiston[i] = this.fPiston[i];
      this.pTarget[i] = this.pStart[i] + delta * (this.pStop[i] - this.pStart[i]);
      this.fPiston[i] = this.pCurrent[i] - this.pTarget[i];
    }
  }

  /** end_of_step: the piston velocities from the forces of this step and the random force. */
  endOfStep(): void {
    const dt = this.sys.state.dt;
    for (let i = 0; i < 6; i++) {
      if (!this.pFlag[i]) continue;
      this.pDeriv[i] *= this.gjfa[i];
      this.pDeriv[i] += 0.5 * dt * (this.gjfa[i] * this.fOldPiston[i] + this.fPiston[i]) / this.pMass[i];
      this.pDeriv[i] += this.fran[i] * this.gjfb[i] / this.pMass[i];
    }
  }

  /** couple_pressure: the virial pressure per barostatted component, coupled as the couple keyword says. */
  private couplePressure(scalar: number, t: Float64Array | null): void {
    const pc = this.pCurrent;
    if (this.pstyle === 'iso') {
      pc[0] = pc[1] = pc[2] = scalar;
    } else if (t === null) {
      throw new StyleError('fix press/langevin: no pressure tensor');
    } else if (this.pcouple === 'xyz') {
      const ave = 1.0 / 3.0 * (t[0] + t[1] + t[2]);
      pc[0] = pc[1] = pc[2] = ave;
    } else if (this.pcouple === 'xy') {
      const ave = 0.5 * (t[0] + t[1]);
      pc[0] = pc[1] = ave;
      pc[2] = t[2];
    } else if (this.pcouple === 'yz') {
      const ave = 0.5 * (t[1] + t[2]);
      pc[1] = pc[2] = ave;
      pc[0] = t[0];
    } else if (this.pcouple === 'xz') {
      const ave = 0.5 * (t[0] + t[2]);
      pc[0] = pc[2] = ave;
      pc[1] = t[1];
    } else {
      pc[0] = t[0];
      pc[1] = t[1];
      pc[2] = t[2];
    }
    if (t !== null) {
      pc[3] = t[3];
      pc[4] = t[4];
      pc[5] = t[5];
    } else {
      // the scalar (iso) case: the tensor's shear components are not used (no tilt)
      pc[3] = pc[4] = pc[5] = 0;
    }
  }

  /** couple_kinetic: the ideal-gas term N k T_target / V (natoms, the target temperature, not a measured one). */
  private coupleKinetic(): void {
    const sys = this.sys;
    const s = sys.state;
    const volume = sys.geom.volume(this.dimension);
    let pk = s.n * s.units.boltz * this.tTarget / volume;
    pk *= s.units.nktv2p;
    this.pCurrent[0] += pk;
    this.pCurrent[1] += pk;
    if (this.dimension === 3) this.pCurrent[2] += pk;
  }

  /** couple_beta: the random piston forces, drawn in the native order for the coupling style (one per call). */
  private coupleBeta(): void {
    const s = this.sys.state;
    const boltz = s.units.boltz;
    const dt = s.dt;
    const gamma = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 6; i++) gamma[i] = Math.sqrt(2.0 * this.pFric[i] * boltz * dt * this.tTarget);
    const fr = this.fran;
    fr.fill(0);
    const g = () => this.rng.gaussian();
    if (this.pstyle === 'iso' || this.pcouple === 'xyz') {
      fr[0] = fr[1] = fr[2] = gamma[0] * g();
    } else if (this.pcouple === 'xy') {
      fr[0] = fr[1] = gamma[0] * g();
      fr[2] = gamma[2] * g();
    } else if (this.pcouple === 'yz') {
      fr[1] = fr[2] = gamma[1] * g();
      fr[0] = gamma[0] * g();
    } else if (this.pcouple === 'xz') {
      fr[0] = fr[2] = gamma[0] * g();
      fr[1] = gamma[1] * g();
    } else {
      fr[0] = gamma[0] * g();
      fr[1] = gamma[1] * g();
      fr[2] = gamma[2] * g();
    }
    fr[3] = gamma[3] * g();
    fr[4] = gamma[4] * g();
    fr[5] = gamma[5] * g();
  }

  /** fix_modify press ID (fix_press_langevin.cpp modify_param): the pressure compute used by the barostat. */
  modify(key: string, values: string[]): number {
    if (key !== 'press') return super.modify(key, values);
    const id = values[0];
    if (id === undefined) throw new StyleError('Illegal fix_modify command: missing argument(s)');
    const c = this.sys.computes.find((x) => x.id === id);
    if (!c) throw new StyleError(`Could not find fix_modify pressure compute ID: ${id}`);
    if (!c.pressFlag) throw new StyleError(`Fix_modify pressure compute ${id} does not compute pressure`);
    if (this.pressId === this.ownPressId) removeCompute(this.sys, this.pressId);
    this.pressId = id;
    this.pressCompute = c;
    return 1;
  }

  destroy(): void {
    if (this.pressId === this.ownPressId) removeCompute(this.sys, this.pressId);
  }
}
