/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/COLLOID/pair_brownian.cpp and
 * pair_brownian.h (Amit Kumar and Michael Bybee (UIUC)). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style brownian mu flaglog flagfld cut_inner cut T seed [flagHI flagVF] (docs.lammps.org/pair_brownian.html):
 * "Style brownian computes the Brownian force and torque on finite-size particles ... the force and torque from the
 * random and drag terms are added to the force and torque of the pair." The random stream is RanMars(seed + me) created
 * when pair_style is read (native PairBrownian::settings); me = 0 on one processor. Draws happen per owned atom in
 * native local order (the isotropic FLD terms of each atom, then the pairs of that atom in native pair order), with
 * one draw per a_sq term, two per a_sh term and two per a_pu term, as pair_brownian.cpp compute() does.
 *
 * Native's compute() enumerates pairs with the half list (newton on, the default); newton off gives the same list
 * enumeration with each pair's forces only on owned atoms (native warns). The fix deform and fix wall volume-fraction
 * corrections follow init_style and compute.
 */
import { Pair, PairParams, StyleError, mixDistance, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum, parseInt_ } from '../../engine/force/util';
import { nativeOrder } from '../../engine/atoms';
import type { Geometry } from '../../engine/domain';
import type { SimState } from '../../engine/types';
import type { Neighbor } from '../../engine/neighbor';
import type { Fix } from '../../engine/fix/fix';
import { FixWallBase } from '../../engine/fix/walls';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { RanMars } from '../random_mars';
import { log as libmLog } from '../libm/log';

/**
 * Native force->vxmu2f per unit style (update.cpp set_units at b332f28: lj 1.0, real 1.4393264316e4, metal 0.6241509647,
 * si 1.0, cgs 1.0, electron 3.39893149e1, micro 1.0, nano 1.0).
 */
export const VXMU2F: Record<string, number> = {
  lj: 1.0, real: 1.4393264316e4, metal: 0.6241509647, si: 1.0, cgs: 1.0, electron: 3.39893149e1, micro: 1.0, nano: 1.0,
};

/** math_special.h cube(x) = x * x * x. */
export const cube = (x: number): number => x * x * x;

/** The fix wall styles that native "^wall" can match (fix styles of the wall family). */
export const WALL_STYLES = [
  'wall/lj93', 'wall/lj126', 'wall/lj1043', 'wall/harmonic', 'wall/harmonic/outside', 'wall/morse', 'wall/colloid',
  'wall/srd', 'wall/gran', 'wall/gran/region', 'wall/region', 'wall/reflect', 'wall/table', 'wall/lepton', 'wall/ees',
];

/** The wall description of a FixWallBase (its protected walls and coords, read through the same object). */
export interface WallView {
  readonly walls: readonly { dim: number; lo: boolean; coord: { edge: boolean; v?: unknown } }[];
  coords(): Float64Array;
}

/** The wall fix of the system (native modify->get_fix_by_style("^wall")), first match; null when none. */
export const findWall = (ctx: StyleContext): Fix | null => {
  for (const st of WALL_STYLES) {
    const f = ctx.fixOf?.(st);
    if (f) return f;
  }
  return null;
};

/** True when a wall position is a variable (native FixWall xflag: moving walls). */
export const wallMoves = (view: WallView): boolean =>
  view.walls.some((w) => !w.coord.edge && typeof w.coord.v === 'object' && w.coord.v !== null && 'variable' in w.coord.v);

/**
 * Native PairBrownian::set_3_orthogonal_vectors (pair_brownian.cpp at b332f28): p2 orthogonal to p1 with the largest
 * component of p1 set aside, normalized, and p3 = p2 x p1.
 */
export const orthogonalVectors = (p1: Float64Array, p2: Float64Array, p3: Float64Array): void => {
  let ix: number, iy: number, iz: number;
  if (Math.abs(p1[0]) > Math.abs(p1[1])) { iz = 0; ix = 1; iy = 2; } else { iz = 1; ix = 2; iy = 0; }
  if (iz === 0) {
    if (Math.abs(p1[0]) < Math.abs(p1[2])) { iz = 2; ix = 0; iy = 1; }
  } else if (Math.abs(p1[1]) < Math.abs(p1[2])) { iz = 2; ix = 0; iy = 1; }
  p2[ix] = 1.0;
  p2[iy] = 1.0;
  p2[iz] = -(p1[ix] * p2[ix] + p1[iy] * p2[iy]) / p1[iz];
  const norm = Math.sqrt(p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2]);
  p2[0] = p2[0] / norm;
  p2[1] = p2[1] / norm;
  p2[2] = p2[2] / norm;
  p3[0] = p1[1] * p2[2] - p1[2] * p2[1];
  p3[1] = p1[2] * p2[0] - p1[0] * p2[2];
  p3[2] = p1[0] * p2[1] - p1[1] * p2[0];
};

/** Native Atom::radius_consistency: [consistent, radius of the type's first atom (-1 when the type has no atoms)]. */
const radiusConsistency = (s: SimState, itype: number): [boolean, number] => {
  let value = -1.0;
  let flag = false;
  const radius = s.radius!;
  for (let i = 0; i < s.n; i++) {
    if (s.type[i] !== itype) continue;
    if (value < 0.0) value = radius[i];
    else if (value !== radius[i]) flag = true;
  }
  return flag ? [false, value] : [true, value];
};

/** Native ev_tally_xyz for the vatom part (newton on: both atoms take half; newton off: i always, j when owned). */
export const tallyVatom = (
  pc: PairCompute, i: number, j: number, jOwned: boolean, fx: number, fy: number, fz: number,
  dx: number, dy: number, dz: number,
): void => {
  const va = pc.vatom;
  if (!va) return;
  const w = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
  for (let c = 0; c < 6; c++) va[6 * i + c] += 0.5 * w[c];
  if (jOwned) for (let c = 0; c < 6; c++) va[6 * j + c] += 0.5 * w[c];
};

export class PairBrownian extends Pair {
  readonly name: string = 'brownian';
  needsFull = false;
  needsHalf = true;
  /**
   * pair brownian: the global virial is the fdotr sum with newton on (native no_virial_fdotr_compute stays 0); with newton
   * off native's Verlet tallies it pair by pair (src/verlet.cpp init: "virial_style = VIRIAL_PAIR"), see initStyle.
   */
  virialFdotr = true;
  mu = 0;
  flaglog = 0;
  flagfld = 0;
  cutInnerGlobal = 0;
  cutGlobal = 0;
  tTarget = 0;
  seed = 0;
  flagHI = 1;
  flagVF = 1;
  /** mono-disperse radius (native rad) and the total particle volume (native vol_P). */
  protected rad = 0;
  protected volP = 0;
  protected R0 = 0;
  protected RT0 = 0;
  protected flagdeform = 0;
  protected flagwall = 0;
  protected wall: WallView | null = null;
  protected newtonOn = true;
  protected logFn: (text: string) => void = () => {};
  private rng: RanMars | null = null;
  private p: PairParams | null = null;
  private order: NativePairOrder | null = null;
  private warnedTriclinic = false;

  settings(args: string[], ctx: StyleContext): void {
    if (args.length !== 7 && args.length !== 9) throw new StyleError('Illegal pair_style command');
    this.mu = parseNum(args[0], 'mu');
    this.flaglog = parseInt_(args[1], 'flaglog');
    this.flagfld = parseInt_(args[2], 'flagfld');
    this.cutInnerGlobal = parseNum(args[3], 'cut_inner');
    this.cutGlobal = parseNum(args[4], 'cut');
    this.tTarget = parseNum(args[5], 'T');
    this.seed = parseInt_(args[6], 'seed');
    this.flagHI = 1;
    this.flagVF = 1;
    if (args.length === 9) {
      this.flagHI = parseInt_(args[7], 'flagHI');
      this.flagVF = parseInt_(args[8], 'flagVF');
    }
    if (this.flaglog === 1 && this.flagHI === 0) {
      ctx.log('WARNING: Cannot include log terms without 1/r terms; setting flagHI to 1');
      this.flagHI = 1;
    }
    // native: "initialize Marsaglia RNG with processor-unique seed": RanMars(lmp, seed + comm->me), me = 0 here
    this.rng = new RanMars(this.seed);
    // reset cutoffs that have been explicitly set
    if (this.p) {
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) {
          if (this.p.isSet(i, j)) this.p.set(i, j, [this.cutInnerGlobal, this.cutGlobal]);
        }
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['cut_inner', 'cut']);
  }

  protected params(): PairParams {
    if (!this.p) throw new StyleError(`pair_coeff for ${this.name} before the atom types are defined`);
    return this.p;
  }

  coeff(args: string[]): void {
    if (args.length !== 2 && args.length !== 4) throw new StyleError('Incorrect args for pair coefficients');
    const p = this.params();
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    let cutInnerOne = this.cutInnerGlobal;
    let cutOne = this.cutGlobal;
    if (args.length === 4) {
      cutInnerOne = parseNum(args[2], 'cut_inner');
      cutOne = parseNum(args[3], 'cut');
    }
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        p.set(i, j, [cutInnerOne, cutOne]);
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /**
   * Native PairBrownian::init_style (pair_brownian.cpp at b332f28): atom attribute radius, the newton warning, extended
   * particles, monodisperse radii, fix deform and fix wall volume corrections, and the isotropic constants R0 and RT0.
   */
  initStyle(ctx: StyleContext): void {
    this.logFn = ctx.log;
    this.newtonOn = ctx.newtonPair !== false;
    // newton off: native's ev_tally_xyz adds half of each pair's virial for each owned atom of the pair (pair.cpp), and
    // the forces of ghost partners are dropped, so the fdotr sum would not be the native virial
    this.virialFdotr = this.newtonOn;
    this.explicitVirial = !this.newtonOn;
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    const s = ctx.s;
    if (!s) return;
    if (!s.radius) throw new StyleError('Pair brownian requires atom attribute radius');
    if (!this.newtonOn) ctx.log('WARNING: Pair brownian needs newton pair on for momentum conservation');
    for (let i = 0; i < s.n; i++) {
      if (s.radius[i] === 0.0) throw new StyleError('Pair brownian requires extended particles');
    }
    // require monodisperse system with same radii for all types
    let radtype = 0;
    for (let i = 1; i <= this.ntypes; i++) {
      const [ok, rt] = radiusConsistency(s, i);
      radtype = rt;
      if (!ok) throw new StyleError('Pair brownian requires monodisperse particles');
      if (i > 1 && radtype !== this.rad) throw new StyleError('Pair brownian requires monodisperse particles');
      this.rad = radtype;
    }
    this.flagdeform = ctx.fixOf?.('deform') ? 1 : 0;
    this.wall = this.findWallView(ctx, 'pair brownian');
    this.flagwall = this.wall ? (wallMoves(this.wall) ? 2 : 1) : 0;
    const prd = [s.box.hi[0] - s.box.lo[0], s.box.hi[1] - s.box.lo[1], s.box.hi[2] - s.box.lo[2]];
    const volT = this.initVolume(prd);
    // vol_P = volume of particles, assuming mono-dispersity
    this.volP = s.n * (4.0 / 3.0) * Math.PI * cube(this.rad);
    const volF = this.flagVF ? this.volP / volT : 0;
    [this.R0, this.RT0] = this.constants(volF);
  }

  /** Wall lookup shared by the brownian styles: at most one wall fix, and only FixWall-family styles. */
  protected findWallView(ctx: StyleContext, who: string): WallView | null {
    const f = findWall(ctx);
    if (!f) return null;
    if (!(f instanceof FixWallBase)) throw new StyleError(`Fix ${f.style} is not compatible with pair ${who}`);
    return f as unknown as WallView;
  }

  /** Volume of the region the particles occupy at init: the box, or the box between the walls (native init_style). */
  protected initVolume(prd: number[]): number {
    if (!this.wall) return prd[0] * prd[1] * prd[2];
    const { lo, hi } = this.wallBounds(prd);
    return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
  }

  /** The walls' lo and hi positions per dimension (native wallhi / walllo; a dimension without wall uses the box). */
  protected wallBounds(prd: number[]): { lo: number[]; hi: number[] } {
    const lo = [0, 0, 0];
    const hi = [prd[0], prd[1], prd[2]];
    const view = this.wall!;
    const coords = view.coords();
    for (let k = 0; k < view.walls.length; k++) {
      const w = view.walls[k];
      if (w.lo) lo[w.dim] = coords[k];
      else hi[w.dim] = coords[k];
    }
    return { lo, hi };
  }

  /** R0 and RT0 for a volume fraction (native isotropic constants of init_style and compute). */
  protected constants(volF: number): [number, number] {
    const mu = this.mu, rad = this.rad;
    if (this.flaglog === 0) {
      return [6 * Math.PI * mu * rad * (1.0 + 2.16 * volF), 8 * Math.PI * mu * cube(rad)];
    }
    return [
      6 * Math.PI * mu * rad * (1.0 + 2.725 * volF - 6.583 * volF * volF),
      8 * Math.PI * mu * cube(rad) * (1.0 + 0.749 * volF - 2.469 * volF * volF),
    ];
  }

  /** Native compute(): the volume-fraction correction of R0 and RT0 when the box or a wall moves. */
  protected volumeCorrection(g: Geometry): void {
    if (!(this.flagVF && (this.flagdeform || this.flagwall === 2))) return;
    let dims: number[];
    if (this.flagdeform && !this.flagwall) dims = [g.lx, g.ly, g.lz];
    else {
      const { lo, hi } = this.wallBounds([g.lx, g.ly, g.lz]);
      dims = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    }
    const volT = dims[0] * dims[1] * dims[2];
    const volF = this.volP / volT;
    [this.R0, this.RT0] = this.constants(volF);
  }

  /** scale factor for Brownian moments (native compute, with update->dt and the unit conversions). */
  protected prethermostat(s: SimState): number {
    const vx = VXMU2F[s.units.style];
    if (vx === undefined) throw new StyleError(`pair ${this.name}: unit style ${s.units.style} has no vxmu2f`);
    let pre = Math.sqrt(24.0 * s.units.boltz * this.tTarget / s.dt);
    pre *= Math.sqrt(vx / s.units.ftm2v / s.units.mvv2e);
    return pre;
  }

  initOne(i: number, j: number): number {
    const p = this.params();
    if (!p.isSet(i, j)) {
      // native init_one: unset pairs take the mixed cutoffs (cut_inner mixes like sigma)
      p.setMixed(i, j, 'cut_inner', mixDistance(this.mix, p.get('cut_inner', i, i), p.get('cut_inner', j, j)));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    return p.get('cut', i, j);
  }

  /** The native pair enumeration for the random draws (half list, newton on or off). */
  protected pairOrder(): NativePairOrder {
    if (!this.order) this.order = new NativePairOrder(this.newtonOn);
    return this.order;
  }

  /** Per-atom wall-free state the subclasses share: the scratch torque and the warning for triclinic boxes. */
  protected triclinicWarning(g: Geometry, name: string): void {
    if (g.triclinic && !this.warnedTriclinic) {
      this.warnedTriclinic = true;
      this.logFn(`WARNING: pair_style ${name}: triclinic box: random forces are drawn in the browser engine's pair order (native's enumeration is not ported for triclinic boxes); the run matches native LAMMPS only statistically`);
    }
  }

  protected rng0(): RanMars {
    if (!this.rng) throw new StyleError(`pair ${this.name} used before pair_style`);
    return this.rng;
  }

  compute(pc: PairCompute): void {
    const s = pc.s, nb: Neighbor = pc.nb, g = pc.geom;
    const rng = this.rng0();
    this.triclinicWarning(g, this.name);
    this.volumeCorrection(g);
    const pre = this.prethermostat(s);
    const radius = s.radius!;
    const order = nativeOrder(s);
    const nlocal = pc.nlocal, nall = pc.nall;
    const x = pc.x, f = pc.f, type = pc.type;
    const cutsq = this.cutsq;
    const nt = this.ntypes + 1;
    const cutInner = this.params().p('cut_inner');
    const { R0, RT0, flaglog, flagfld, flagHI, mu } = this;
    const vxmu = VXMU2F[s.units.style];
    const newton = this.newtonOn;
    // torque scratch over owned and ghost atoms, summed into the owners at the end (native reverse comm of torque)
    const tq = flaglog && s.torque ? new Float64Array(3 * nall) : null;
    const pairs = flagHI ? this.pairOrder().pairs(s, nb, g, pc.half) : null;
    const p1 = new Float64Array(3), p2 = new Float64Array(3), p3 = new Float64Array(3);
    const xl = new Float64Array(3);
    let q = 0;
    for (let k = 0; k < nlocal; k++) {
      const i = order[k];
      const radi = radius[i];
      if (flagfld) {
        f[3 * i] += pre * Math.sqrt(R0) * (rng.uniform() - 0.5);
        f[3 * i + 1] += pre * Math.sqrt(R0) * (rng.uniform() - 0.5);
        f[3 * i + 2] += pre * Math.sqrt(R0) * (rng.uniform() - 0.5);
        if (flaglog) {
          tq![3 * i] += pre * Math.sqrt(RT0) * (rng.uniform() - 0.5);
          tq![3 * i + 1] += pre * Math.sqrt(RT0) * (rng.uniform() - 0.5);
          tq![3 * i + 2] += pre * Math.sqrt(RT0) * (rng.uniform() - 0.5);
        }
      }
      if (!flagHI) continue;
      while (q < pairs!.length && pairs![q] === i) {
        const jEnc = pairs![q + 1];
        q += 2;
        const j = jEnc & NEIGHMASK;
        const jOwned = j < nlocal;
        const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = type[i] * nt + type[j];
        if (!(rsq < cutsq[t])) continue;
        const r = Math.sqrt(rsq);
        // scalar resistances a_sq and a_sh
        let hSep = r - 2.0 * radi;
        // if less than minimum gap, use minimum gap instead
        if (r < cutInner[t]) hSep = cutInner[t] - 2.0 * radi;
        // scale h_sep by radi
        hSep = hSep / radi;
        let aSq: number, aSh = 0, aPu = 0;
        if (flaglog) {
          aSq = 6.0 * Math.PI * mu * radi * (1.0 / 4.0 / hSep + 9.0 / 40.0 * libmLog(1.0 / hSep));
          aSh = 6.0 * Math.PI * mu * radi * (1.0 / 6.0 * libmLog(1.0 / hSep));
          aPu = 8.0 * Math.PI * mu * cube(radi) * (3.0 / 160.0 * libmLog(1.0 / hSep));
        } else {
          aSq = 6.0 * Math.PI * mu * radi * (1.0 / 4.0 / hSep);
        }
        // generate the Pairwise Brownian Force: a_sq
        let Fbmag = pre * Math.sqrt(aSq);
        let randr = rng.uniform() - 0.5;
        // contribution due to Brownian motion
        let fx = Fbmag * randr * delx / r;
        let fy = Fbmag * randr * dely / r;
        let fz = Fbmag * randr * delz / r;
        // add terms due to a_sh
        if (flaglog) {
          // generate two orthogonal vectors to the line of centers
          p1[0] = delx / r;
          p1[1] = dely / r;
          p1[2] = delz / r;
          orthogonalVectors(p1, p2, p3);
          Fbmag = pre * Math.sqrt(aSh);
          // force in each of the two directions
          randr = rng.uniform() - 0.5;
          fx += Fbmag * randr * p2[0];
          fy += Fbmag * randr * p2[1];
          fz += Fbmag * randr * p2[2];
          randr = rng.uniform() - 0.5;
          fx += Fbmag * randr * p3[0];
          fy += Fbmag * randr * p3[1];
          fz += Fbmag * randr * p3[2];
        }
        // scale forces to appropriate units
        fx = vxmu * fx;
        fy = vxmu * fy;
        fz = vxmu * fz;
        // sum to total force
        f[3 * i] -= fx;
        f[3 * i + 1] -= fy;
        f[3 * i + 2] -= fz;
        if (newton || jOwned) {
          f[3 * j] += fx;
          f[3 * j + 1] += fy;
          f[3 * j + 2] += fz;
        }
        // torque due to the Brownian Force
        if (flaglog) {
          // location of the point of closest approach on I from its center
          xl[0] = -delx / r * radi;
          xl[1] = -dely / r * radi;
          xl[2] = -delz / r * radi;
          // torque = xl_cross_F
          let tx = xl[1] * fz - xl[2] * fy;
          let ty = xl[2] * fx - xl[0] * fz;
          let tz = xl[0] * fy - xl[1] * fx;
          // torque is same on both particles
          tq![3 * i] -= tx;
          tq![3 * i + 1] -= ty;
          tq![3 * i + 2] -= tz;
          if (newton || jOwned) {
            tq![3 * j] -= tx;
            tq![3 * j + 1] -= ty;
            tq![3 * j + 2] -= tz;
          }
          // torque due to a_pu
          Fbmag = pre * Math.sqrt(aPu);
          randr = rng.uniform() - 0.5;
          tx = Fbmag * randr * p2[0];
          ty = Fbmag * randr * p2[1];
          tz = Fbmag * randr * p2[2];
          randr = rng.uniform() - 0.5;
          tx += Fbmag * randr * p3[0];
          ty += Fbmag * randr * p3[1];
          tz += Fbmag * randr * p3[2];
          // torque has opposite sign on two particles
          tq![3 * i] -= tx;
          tq![3 * i + 1] -= ty;
          tq![3 * i + 2] -= tz;
          if (newton || jOwned) {
            tq![3 * j] += tx;
            tq![3 * j + 1] += ty;
            tq![3 * j + 2] += tz;
          }
        }
        if (!newton) {
          // ev_tally_xyz(i, j, nlocal, 0, 0, 0, -fx, -fy, -fz, del): half for the owned i, half for j when owned
          const h = jOwned ? 1 : 0.5, v = pc.acc.virial;
          v[0] -= h * delx * fx; v[1] -= h * dely * fy; v[2] -= h * delz * fz;
          v[3] -= h * delx * fy; v[4] -= h * delx * fz; v[5] -= h * dely * fz;
        }
        if (pc.vatom) tallyVatom(pc, i, j, newton || jOwned, -fx, -fy, -fz, delx, dely, delz);
      }
    }
    if (tq) nb.reverseSum(tq, 3, s.torque!);
  }
}
