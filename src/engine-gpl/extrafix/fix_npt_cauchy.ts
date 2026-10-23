/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_npt_cauchy.cpp and src/EXTRA-FIX/fix_npt_cauchy.h at commit b332f28 (class FixNPTCauchy,
 * fix npt/cauchy): the CauchyStat routines (CauchyStat_init, CauchyStat, CauchyStat_Step) and the keyword side effects
 * of the constructor. The Nose-Hoover machinery is the core FixNH (src/engine/fix/nh.ts), reached through its
 * protected hooks; its deviatoric terms (compute_sigma, compute_deviatoric, compute_strain_energy, which
 * fix_npt_cauchy.cpp repeats from fix_nh.cpp) are ../fix/nh_deviatoric.ts, called by the core. Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_npt_cauchy.html): the keywords of fix npt; "cauchystat" is not a keyword
 * here: "alpha" sets the integration parameter of the CauchyStat algorithm, and "continue yes" restarts it from the
 * stored target stress of the previous run. CauchyStat adjusts the target pressure tensor so that the Cauchy stress
 * converges to the requested values. The stored state lives per system (keyed by the fix ID plus "_FIX_NH_STORE",
 * as the STORE/GLOBAL fix of native LAMMPS does), so it survives re-specifying the fix.
 *
 * Refused by name (StyleError): ptemp, drag (other than 0), and 2d boxes.
 */
import { FixNH } from '../../engine/fix/nh';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { ComputePressure } from '../../engine/compute/pressure';
import { exp as libmExp } from '../libm/exp';

type Mat3 = number[][];

/** Stored target stress per system and fix ID (the STORE/GLOBAL fix of fix_npt_cauchy.cpp CauchyStat_init). */
const stores = new WeakMap<System, Map<string, number[]>>();

/** Keywords of fix npt/cauchy's constructor that the browser engine has no implementation for. */
const REFUSED = ['ptemp', 'update', 'disc', 'erate', 'strain', 'ext', 'psllod', 'peculiar', 'kick', 'integrator', 'isochoric'];

/** Upper-triangular Voigt h (lx ly lz yz xz xy) to a 3x3 matrix with the h[5], h[4], h[3] off-diagonal terms. */
function upperFromVoigt(h: number[]): Mat3 {
  return [[h[0], h[5], h[4]], [0, h[1], h[3]], [0, 0, h[2]]];
}

/** Domain h_inv of a box with lengths and tilts (LAMMPS Domain::set_global_box, Voigt order). */
function voigtInverse(lx: number, ly: number, lz: number, xy: number, xz: number, yz: number): number[] {
  return [
    1 / lx,
    1 / ly,
    1 / lz,
    -yz / (ly * lz),
    (yz * xy - ly * xz) / (lx * ly * lz),
    -xy / (lx * ly),
  ];
}

function times3(a: Mat3, b: Mat3): Mat3 {
  const c: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  }
  return c;
}

function det3(m: Mat3): number {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

function invert3(m: Mat3): Mat3 {
  const d = det3(m);
  const r: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
  r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d;
  r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
  r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
  r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
  r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
  r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
  r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d;
  r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
  return r;
}

/** (row, col) pairs of the six Voigt entries, 0-based: uv[ii] = (i, j) (fix_npt_cauchy.cpp CauchyStat_Step). */
const UV: [number, number][] = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]];

/**
 * CauchyStat_Step (fix_npt_cauchy.cpp): one step of the PK stress update from the Cauchy stress error. Updates setPK in
 * place. Inputs: Fi (inverse deformation gradient), Fdot (its rate), cauchy and setcauchy (current and target Cauchy
 * stress), volume and volume0 (current and reference cell volume), deltat (time step), alpha (integration parameter).
 */
function cauchyStatStep(
  Fi: Mat3, Fdot: Mat3, cauchy: Mat3, setcauchy: Mat3, setPK: Mat3, volume: number, volume0: number,
  deltat: number, alpha: number,
): void {
  const deltastress = [0, 0, 0, 0, 0, 0];
  const fdotvec = [0, 0, 0, 0, 0, 0];
  const dsdf: number[][] = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]];
  const dsds: number[][] = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]];
  const deltaF = [0, 0, 0, 0, 0, 0];
  const deltaPK = [0, 0, 0, 0, 0, 0];

  for (let ii = 0; ii < 6; ii++) {
    const [i, j] = UV[ii];
    deltastress[ii] = setcauchy[i][j] - cauchy[i][j];
    if (ii > 2) deltastress[ii] = deltastress[ii] * 2.0;
    fdotvec[ii] = Fdot[i][j] * deltat;
  }

  for (let ii = 0; ii < 6; ii++) {
    const [i, j] = UV[ii];
    for (let jj = 0; jj < 6; jj++) {
      const [m, n] = UV[jj];
      dsds[ii][jj] = Fi[i][m] * Fi[j][n] + Fi[i][n] * Fi[j][m] + Fi[j][m] * Fi[i][n] + Fi[j][n] * Fi[i][m];
      for (let l = 0; l < 3; l++) {
        for (let k = 0; k < 3; k++) {
          dsdf[ii][jj] = dsdf[ii][jj] + cauchy[k][l] *
            (Fi[i][k] * Fi[j][l] * Fi[n][m] - Fi[i][m] * Fi[j][l] * Fi[n][k] - Fi[i][k] * Fi[j][m] * Fi[n][l]);
        }
      }
    }
  }

  const jac = volume / volume0;
  for (let ii = 0; ii < 6; ii++) {
    for (let jj = 0; jj < 6; jj++) {
      dsds[ii][jj] = dsds[ii][jj] * jac / 4.0;
      dsdf[ii][jj] = dsdf[ii][jj] * jac;
    }
  }

  for (let ii = 0; ii < 6; ii++) {
    for (let jj = 0; jj < 6; jj++) deltaF[ii] = deltaF[ii] + dsdf[ii][jj] * fdotvec[jj];
  }

  for (let ii = 0; ii < 6; ii++) {
    for (let jj = 0; jj < 6; jj++) deltaPK[ii] = deltaPK[ii] + alpha * dsds[ii][jj] * deltastress[jj];
    deltaPK[ii] = deltaPK[ii] + alpha * deltaF[ii];
  }

  setPK[0][0] = setPK[0][0] + deltaPK[0];
  setPK[1][1] = setPK[1][1] + deltaPK[1];
  setPK[2][2] = setPK[2][2] + deltaPK[2];
  setPK[1][2] = setPK[1][2] + deltaPK[3];
  setPK[2][1] = setPK[2][1] + deltaPK[3];
  setPK[0][2] = setPK[0][2] + deltaPK[4];
  setPK[2][0] = setPK[2][0] + deltaPK[4];
  setPK[0][1] = setPK[0][1] + deltaPK[5];
  setPK[1][0] = setPK[1][0] + deltaPK[5];
}

export class FixNPTCauchy extends FixNH {
  private readonly alpha: number;
  private readonly restartPK: boolean;
  private readonly idStore: string;
  private astore!: number[];
  private readonly invH0: Mat3;
  private readonly csVol0: number;
  private hOld: number[] = [0, 0, 0, 0, 0, 0];
  private setPK: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  private initPK = true;
  private initRUN = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    // the keywords this class reads (alpha, continue) are removed before the core parse
    const scan = FixNPTCauchy.scanArgs(sys, args);
    super(sys, id, group, scan.coreArgs, 'npt/cauchy');
    this.alpha = scan.alpha;
    this.restartPK = scan.restartPK;
    this.scaleTilt = [scan.scaleyz, scan.scalexz, scan.scalexy];
    this.idStore = `${id}_FIX_NH_STORE`;

    // CauchyStat_init (called from post_constructor in native LAMMPS)
    let all = stores.get(sys);
    if (!all) { all = new Map(); stores.set(sys, all); }
    let store = all.get(this.idStore);
    if (this.restartPK && !store) {
      throw new StyleError('Illegal fix npt/cauchy command.  Continuation run must follow a previously equilibrated npt/cauchy run');
    }
    if (!(this.alpha > 0)) throw new StyleError('Illegal fix npt/cauchy command: Alpha cannot be zero or negative.');
    if (!store) {
      store = [0, 0, 0, 0, 0, 0];
      all.set(this.idStore, store);
    }
    this.astore = store;
    this.initRUN = false;
    this.initPK = true;

    // H0 and invH0: the current box shape, the reference of the Cauchy stress (CauchyStat_init)
    const g = sys.geom;
    const h0 = upperFromVoigt([g.lx, g.ly, g.lz, g.yz, g.xz, g.xy]);
    const hi = voigtInverse(g.lx, g.ly, g.lz, g.xy, g.xz, g.yz);
    this.invH0 = [[hi[0], hi[5], hi[4]], [0, hi[1], hi[3]], [0, 0, hi[2]]];
    this.csVol0 = Math.abs(det3(h0));
  }

  /**
   * Reads the keywords of the fix that the core FixNH does not take (alpha, continue), refuses the ones this engine
   * does not implement, and works out the constructor's side effects of fix_npt_cauchy.cpp: the scalexy, scaleyz and
   * scalexz defaults (on when the periodic second dimension has a tilt), and the keywords that clear them (x, xy, xz,
   * yz and tri set them to 0). deviatoric_flag (set by the x, y, z, xy, yz and xz keywords) is set by the core parse.
   */
  private static scanArgs(sys: System, args: string[]): {
    coreArgs: string[]; alpha: number; restartPK: boolean;
    scalexy: boolean; scalexz: boolean; scaleyz: boolean;
  } {
    const s = sys.state;
    if (s.dimension === 2) throw new StyleError('fix npt/cauchy in a 2d simulation is not supported by the browser engine yet');
    const g = sys.geom;
    let scalexy = s.box.periodic[1] && g.xy !== 0;
    let scalexz = s.box.periodic[2] && g.xz !== 0;
    let scaleyz = s.box.periodic[2] && g.yz !== 0;
    let alpha = 0.001;
    let restartPK = false;
    const coreArgs: string[] = [];
    const num = (k: number, key: string): number => {
      const v = Number(args[k]);
      if (args[k] === undefined || !Number.isFinite(v)) throw new StyleError(`fix npt/cauchy ${key}: expected a number`);
      return v;
    };
    const logical = (k: number, key: string): boolean => {
      // fix_npt_cauchy.cpp: if (iarg+2 > narg) error->all(FLERR,"Illegal fix npt/cauchy command") before the value is read
      if (args[k] === undefined) throw new StyleError('Illegal fix npt/cauchy command');
      if (args[k] !== 'yes' && args[k] !== 'no') throw new StyleError(`fix npt/cauchy ${key} must be yes or no`);
      return args[k] === 'yes';
    };
    for (let k = 0; k < args.length;) {
      const key = args[k];
      if (REFUSED.includes(key)) throw new StyleError(`fix npt/cauchy: '${key}' is not a fix npt/cauchy keyword`);
      switch (key) {
        case 'alpha': alpha = num(k + 1, key); k += 2; break;
        case 'continue': restartPK = logical(k + 1, key); k += 2; break;
        case 'x': case 'y': case 'z':
          if (key === 'x') scalexy = false;
          coreArgs.push(...args.slice(k, k + 4)); k += 4; break;
        case 'xy': scalexy = false; coreArgs.push(...args.slice(k, k + 4)); k += 4; break;
        case 'yz': scaleyz = false; coreArgs.push(...args.slice(k, k + 4)); k += 4; break;
        case 'xz': scalexz = false; coreArgs.push(...args.slice(k, k + 4)); k += 4; break;
        case 'tri': scalexy = false; scalexz = false; scaleyz = false; coreArgs.push(...args.slice(k, k + 4)); k += 4; break;
        case 'scalexy': scalexy = logical(k + 1, key); coreArgs.push(...args.slice(k, k + 2)); k += 2; break;
        case 'scalexz': scalexz = logical(k + 1, key); coreArgs.push(...args.slice(k, k + 2)); k += 2; break;
        case 'scaleyz': scaleyz = logical(k + 1, key); coreArgs.push(...args.slice(k, k + 2)); k += 2; break;
        case 'drag':
          if (num(k + 1, key) !== 0) throw new StyleError('fix npt/cauchy drag is not supported by the browser engine yet (src/EXTRA-FIX/fix_npt_cauchy.cpp pdrag_factor)');
          k += 2; break;
        case 'iso': case 'aniso': case 'temp': case 'couple': case 'tchain': case 'pchain': case 'tloop': case 'ploop':
        case 'mtk': case 'dilate': case 'flip': case 'fixedpoint': case 'nreset':
          // the arity of the core keywords: 4 for the ones with three numbers, 2 for one value
          {
            const n = key === 'iso' || key === 'aniso' || key === 'temp' || key === 'fixedpoint' ? 4 : 2;
            coreArgs.push(...args.slice(k, k + n));
            k += n;
          }
          break;
        default:
          // anything else goes to the core parse, which names unknown keywords
          coreArgs.push(key);
          k += 1;
          break;
      }
    }
    return { coreArgs, alpha, restartPK, scalexy, scalexz, scaleyz };
  }

  /** Voigt h of the domain: lx ly lz yz xz xy (LAMMPS domain->h). */
  private hVoigt(): number[] {
    const g = this.sys.geom;
    return [g.lx, g.ly, g.lz, g.yz, g.xz, g.xy];
  }

  /** Whether the box has tilts (LAMMPS domain->triclinic, the pstyle TRICLINIC branch of the CauchyStat matrices). */
  private isTriclinic(): boolean {
    return this.sys.state.box.triclinic;
  }

  /** Pressure tensor of the barostat's pressure compute at the current state (LAMMPS pressure->vector, xx..yz). */
  private currentPressure(): number[] {
    const pc = this.pressCompute as ComputePressure;
    pc.keOverride = null;
    this.sys.refreshComputes();
    const v = pc.vectorValues();
    return [v[0], v[1], v[2], v[3], v[4], v[5]];
  }

  /**
   * fix_npt_cauchy.cpp compute_press_target tail: CauchyStat once the pressure has been evaluated (initRUN), the
   * reference box at the first call; the core computes sigma after this hook when the deviatoric keywords are used.
   */
  protected pressTargetSet(): void {
    if (this.initRUN) this.cauchyStat();
    else this.hOld = this.hVoigt();
    this.initRUN = true;
  }

  /** CauchyStat (fix_npt_cauchy.cpp): the PK stress and the target pressure of the next step. */
  private cauchyStat(): void {
    const dt = this.sys.state.dt;
    const h = this.hVoigt();
    const hRate = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 6; i++) {
      hRate[i] = (h[i] - this.hOld[i]) / dt;
      this.hOld[i] = h[i];
    }
    const H = upperFromVoigt(h);
    const Hdot = upperFromVoigt(hRate);
    const F = times3(H, this.invH0);
    const Fdot = times3(Hdot, this.invH0);
    const Fi = invert3(F);
    const J = det3(F);
    const vol = this.csVol0 * J;
    const deltat = dt;

    // current Cauchy stress: minus the pressure tensor (tensor order xx yy zz xy xz yz)
    const tensor = this.currentPressure();
    const tri = this.isTriclinic();
    const cauchy: Mat3 = [[-tensor[0], 0, 0], [0, -tensor[1], 0], [0, 0, -tensor[2]]];
    if (tri) {
      cauchy[0][1] = -tensor[3]; cauchy[0][2] = -tensor[4];
      cauchy[1][0] = -tensor[3]; cauchy[1][2] = -tensor[5];
      cauchy[2][0] = -tensor[4]; cauchy[2][1] = -tensor[5];
    }

    // target Cauchy stress: minus the target pressure tensor (target order x y z yz xz xy)
    const pt = this.pTarget;
    const setcauchy: Mat3 = [[-pt[0], 0, 0], [0, -pt[1], 0], [0, 0, -pt[2]]];
    if (tri) {
      setcauchy[0][1] = -pt[5]; setcauchy[0][2] = -pt[4];
      setcauchy[1][0] = -pt[5]; setcauchy[1][2] = -pt[3];
      setcauchy[2][0] = -pt[4]; setcauchy[2][1] = -pt[3];
    }

    if (this.initPK) {
      if (this.restartPK) {
        const a = this.astore;
        this.setPK = [[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]];
      } else {
        this.setPK = [[cauchy[0][0], cauchy[0][1], cauchy[0][2]], [cauchy[1][0], cauchy[1][1], cauchy[1][2]], [cauchy[2][0], cauchy[2][1], cauchy[2][2]]];
      }
      this.initPK = false;
    }

    cauchyStatStep(Fi, Fdot, cauchy, setcauchy, this.setPK, vol, this.csVol0, deltat, this.alpha);

    // use the current PK stress as the new target (p_target of x y z, and the tilts when triclinic)
    this.pTarget[0] = -this.setPK[0][0];
    this.pTarget[1] = -this.setPK[1][1];
    this.pTarget[2] = -this.setPK[2][2];
    if (this.pFlag[3] || this.pFlag[4] || this.pFlag[5]) {
      this.pTarget[3] = -this.setPK[1][2];
      this.pTarget[4] = -this.setPK[0][2];
      this.pTarget[5] = -this.setPK[0][1];
    }
    const nb = [0, 1, 2].filter((d) => this.pFlag[d]).length;
    let acc = 0;
    for (let i = 0; i < 3; i++) if (this.pFlag[i]) acc += this.pTarget[i];
    this.pHydro = nb > 0 ? acc / nb : 0;

    // save for the continuation run (the target Cauchy stress, as native LAMMPS stores setcauchy)
    const a = this.astore;
    a[0] = setcauchy[0][0]; a[1] = setcauchy[0][1]; a[2] = setcauchy[0][2];
    a[3] = setcauchy[1][1]; a[4] = setcauchy[1][2]; a[5] = setcauchy[2][2];
  }

  /** fix_npt_cauchy.cpp initial_integrate: the barostat chain runs before the thermostat target update. */
  protected barostatChainFirst(): boolean { return true; }

  /**
   * fix_npt_cauchy.cpp nh_v_press (b332f28): one velocity sweep per half step. The diagonal factors
   * exp(-dt/4 (omega_dot + mtk_term2)) bracket one tilt coupling of dt/2 (v_x -= dt/2 (v_y omega_xy + v_z omega_xz),
   * v_y -= dt/2 v_z omega_yz) when the box is triclinic (pstyle TRICLINIC).
   */
  protected barostatVelocityFlow(): void {
    const s = this.sys.state;
    const mtkTerm2 = this.mtkVelocityTerm();
    const w = this.omegaDot;
    const f = [0, 1, 2].map((d) => libmExp(-this.dt4 * (w[d] + mtkTerm2)));
    const tri = this.pFlag[3] || this.pFlag[4] || this.pFlag[5];
    const dthalf = this.dthalf;
    const v = s.v;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const k = 3 * i;
      v[k] *= f[0]; v[k + 1] *= f[1]; v[k + 2] *= f[2];
      if (tri) {
        v[k] += -dthalf * (v[k + 1] * w[5] + v[k + 2] * w[4]);
        v[k + 1] += -dthalf * v[k + 2] * w[3];
      }
      v[k] *= f[0]; v[k + 1] *= f[1]; v[k + 2] *= f[2];
    }
  }

  /**
   * fix_npt_cauchy.cpp remap (b332f28): the tilt factors take a sweep before and after the diagonal stretch about the
   * fixed point, the xy, xz and yz follow-ups of scalexy, scalexz and scaleyz are applied with the stretch, and the
   * dilated atoms map with the box. The tilts must stay within 1.5 box lengths in one step.
   */
  protected remap(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const b = s.box;
    const dto = this.dthalf;
    const w = this.omegaDot;
    const p = this.pFlag;
    for (let d = 0; d < 6; d++) this.omegaPos[d] += dto * w[d];
    // fractional coordinates of the dilated atoms in the old box
    const lam = new Float64Array(3 * s.n);
    const tmp = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.dilateBit)) continue;
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      lam[3 * i] = tmp[0]; lam[3 * i + 1] = tmp[1]; lam[3 * i + 2] = tmp[2];
    }
    // h in Voigt order: lx ly lz yz xz xy (the old lengths; domain->h is set at the end)
    const h = [g.lx, g.ly, g.lz, g.yz, g.xz, g.xy];
    const old = [g.lx, g.ly, g.lz];
    const dto2 = dto / 2.0, dto4 = dto / 4.0, dto8 = dto / 8.0;
    const tri = p[3] || p[4] || p[5];
    const tiltSweep = (): void => {
      if (p[4]) {
        const e = libmExp(dto8 * w[0]);
        h[4] *= e; h[4] += dto4 * (w[5] * h[3] + w[4] * h[2]); h[4] *= e;
      }
      if (p[3]) {
        const e = libmExp(dto4 * w[1]);
        h[3] *= e; h[3] += dto2 * (w[3] * h[2]); h[3] *= e;
      }
      if (p[5]) {
        const e = libmExp(dto4 * w[0]);
        h[5] *= e; h[5] += dto2 * (w[5] * h[1]); h[5] *= e;
      }
      if (p[4]) {
        const e = libmExp(dto8 * w[0]);
        h[4] *= e; h[4] += dto4 * (w[5] * h[3] + w[4] * h[2]); h[4] *= e;
      }
    };
    if (tri) tiltSweep();
    const fp = this.fixedpoint ?? [0.5 * (b.lo[0] + b.hi[0]), 0.5 * (b.lo[1] + b.hi[1]), 0.5 * (b.lo[2] + b.hi[2])];
    const stretch = (d: number): number => {
      const e = libmExp(dto * w[d]);
      b.lo[d] = (b.lo[d] - fp[d]) * e + fp[d];
      b.hi[d] = (b.hi[d] - fp[d]) * e + fp[d];
      return e;
    };
    if (p[0]) stretch(0);
    if (p[1]) {
      const e = stretch(1);
      if (this.scaleTilt[2]) h[5] *= e;
    }
    if (p[2]) {
      const e = stretch(2);
      if (this.scaleTilt[1]) h[4] *= e;
      if (this.scaleTilt[0]) h[3] *= e;
    }
    if (tri) tiltSweep();
    if (Math.abs(h[3]) > 1.5 * old[1] || Math.abs(h[4]) > 1.5 * old[0] || Math.abs(h[5]) > 1.5 * old[0]) {
      throw new StyleError('Fix npt/cauchy has tilted box too far in one step - periodic cell is too far from equilibrium state');
    }
    b.tilt[0] = h[5];
    b.tilt[1] = h[4];
    b.tilt[2] = h[3];
    b.minLo = [...b.lo];
    b.minHi = [...b.hi];
    g.update();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.dilateBit)) continue;
      g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
    }
    for (const f of sys.fixes) if (f !== this) f.boxChanged?.();
  }

  /** fix_npt_cauchy.cpp final_integrate: no temperature->compute_vector() before the pressure (see the core hook). */
  protected staleKineticFinal(): boolean { return true; }
}
