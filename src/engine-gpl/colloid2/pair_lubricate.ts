/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/COLLOID/pair_lubricate.cpp and
 * pair_lubricate.h (Randy Schunk (SNL), Amit Kumar and Michael Bybee (UIUC)), and src/COLLOID/pair_lubricate_poly.cpp
 * and pair_lubricate_poly.h (the same authors, with Dave Heine (Corning) for the polydisperse terms). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lubricate and lubricate/poly (docs.lammps.org/pair_lubricate.html): "pair_style style mu flaglog flagfld
 * cutinner cutoff flagHI flagVF"; "mu = dynamic viscosity", "flaglog = 0/1 to exclude/include log terms in the
 * lubrication approximation", "flagfld = 0/1 to exclude/include Fast Lubrication Dynamics (FLD) effects", "cutinner =
 * inner cutoff distance", "cutoff = outer cutoff for interactions", "flagHI (optional) = 0/1 to exclude/include 1/r
 * hydrodynamic interactions", "flagVF (optional) = 0/1 to exclude/include volume fraction corrections in the long-range
 * isotropic terms". Restrictions in the same page: "Only spherical monodisperse particles are allowed for pair_style
 * lubricate." and "Only spherical particles are allowed for pair_style lubricate/poly."
 *
 * Forces and torques are the native loops, operation for operation. lubricate uses the half list (each pair once, the
 * force and torque of the pair applied to both atoms, native newton on); lubricate/poly uses the full list and applies
 * the pair terms to atom i only (native newton off). Ghost velocities and omegas are copies of their owners (native
 * comm->forward_comm with ghost_velocity); torques on ghosts are summed into their owners as native reverse comm does.
 *
 * Fix deform (native domain->h_rate, h_ratelo, x2lamda and the streaming velocity) comes from FixDeform.streamingRates
 * (orthogonal boxes with a constant rate; other styles are refused naming them). Fix wall positions (native FixWall
 * coord0 and the variable walls) come from the wall fix's coords(). The volume fraction terms use the box or the walls
 * as native does at init_style and at each compute with moving boxes or walls.
 */
import { Pair, StyleError, mixDistance, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import { FixWallBase } from '../../engine/fix/walls';
import { FixDeform } from '../../engine/fix/deform';
import { valueOf, type NumOrVar } from '../../engine/fix/util';
import type { System } from '../../engine/system';
import type { Fix } from '../../engine/fix/fix';
import { log as libmLog } from '../libm/log';
import { pow as libmPow } from '../libm/pow';

/** Fix styles the native "^wall" lookup can find in this engine (fix_wall.cpp and the other wall fixes). */
const WALL_STYLES = [
  'wall/body/polygon', 'wall/body/polyhedron', 'wall/colloid', 'wall/ees', 'wall/flow', 'wall/gran', 'wall/gran/region',
  'wall/harmonic', 'wall/harmonic/outside', 'wall/lepton', 'wall/lj1043', 'wall/lj126', 'wall/lj93', 'wall/morse',
  'wall/piston', 'wall/reflect', 'wall/reflect/stochastic', 'wall/region', 'wall/region/ees', 'wall/srd', 'wall/table',
];

/** force->vxmu2f per unit style (native update.cpp unit settings at b332f28). */
const VXMU2F: Record<string, number> = {
  lj: 1.0, real: 1.4393264316e4, metal: 0.6241509647, si: 1.0, cgs: 1.0, electron: 3.39893149e1, micro: 1.0, nano: 1.0,
};

const intArg = (w: string | undefined, what: string): number => {
  if (w === undefined || !/^[+-]?\d+$/.test(w)) throw new StyleError(`expected an integer for ${what}, got '${w ?? ''}'`);
  return Number(w);
};

/** The fix wall members the native code reads (walls keeps the parsed faces; coords() the resolved positions). */
interface WallView {
  walls: { dim: number; lo: boolean; coord: { edge: boolean; v?: unknown } }[];
}
interface CoordsView {
  coords(): Float64Array;
}

/** Native compute state of one step (the streaming velocity removed from the owned atoms). */
interface Shear {
  h: number[];
  hl: number[];
}

export class PairLubricate extends Pair {
  readonly name: string = 'lubricate';
  /** Native lubricate: the half list with newton on. */
  needsHalf = true;
  needsFull = false;
  /** Native no_virial_fdotr_compute = 1: the virial is tallied pair by pair (ev_tally_xyz). */
  explicitVirial = true;

  mu = 0;
  flaglog = 0;
  flagfld = 0;
  cutInnerGlobal = 0;
  cutGlobal = 0;
  flagHI = 1;
  flagVF = 1;
  /** Monodisperse radius (lubricate) and the particle volume of the system (vol_P). */
  protected rad = 0;
  protected volP = 0;
  protected R0 = 0;
  protected RT0 = 0;
  protected RS0 = 0;
  protected shearing = false;
  protected flagdeform = false;
  /** 0 no fix wall, 1 fix wall with constant positions, 2 with moving (variable) walls. */
  protected flagwall = 0;
  protected wallFix: FixWallBase | null = null;
  protected deformFix: FixDeform | null = null;
  /** Ef[3*a+b]: the strain rate tensor of a deforming box (native Ef[a][b]). */
  protected Ef = new Float64Array(9);
  protected setflag = new Uint8Array(0);
  protected cutIn = new Float64Array(0);
  protected cutOut = new Float64Array(0);
  private vx = 1;

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.setflag = new Uint8Array(nt * nt);
    this.cutIn = new Float64Array(nt * nt);
    this.cutOut = new Float64Array(nt * nt);
  }

  settings(args: string[], ctx: StyleContext): void {
    // docs.lammps.org/pair_lubricate.html: "pair_style style mu flaglog flagfld cutinner cutoff flagHI flagVF"
    if (args.length !== 5 && args.length !== 7) throw new StyleError('Illegal pair_style command');
    this.mu = parseNum(args[0], 'mu');
    this.flaglog = intArg(args[1], 'flaglog');
    this.flagfld = intArg(args[2], 'flagfld');
    this.cutInnerGlobal = parseNum(args[3], 'cutinner');
    this.cutGlobal = parseNum(args[4], 'cutoff');
    this.flagHI = 1;
    this.flagVF = 1;
    if (args.length === 7) {
      this.flagHI = intArg(args[5], 'flagHI');
      this.flagVF = intArg(args[6], 'flagVF');
    }
    if (this.flaglog === 1 && this.flagHI === 0) {
      ctx.log('WARNING: Cannot include log terms without 1/r terms; setting flagHI to 1');
      this.flagHI = 1;
    }
    // reset the cutoffs that were explicitly set (native settings(): "if (setflag[i][j])")
    if (this.setflag.length) {
      const nt = this.ntypes + 1;
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) {
          if (this.setflag[i * nt + j]) {
            this.cutIn[i * nt + j] = this.cutInnerGlobal;
            this.cutOut[i * nt + j] = this.cutGlobal;
          }
        }
      }
    }
  }

  coeff(args: string[], _ctx: StyleContext): void {
    // docs.lammps.org/pair_lubricate.html: "pair_coeff * * [cutinner cutoff]" (narg 2 or 4)
    if (args.length !== 2 && args.length !== 4) throw new StyleError('Incorrect args for pair coefficients');
    if (this.ntypes < 1) throw new StyleError('Incorrect args for pair coefficients');
    if (this.setflag.length !== (this.ntypes + 1) ** 2) this.allocate(this.ntypes);
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    let cin = this.cutInnerGlobal, cut = this.cutGlobal;
    if (args.length === 4) {
      cin = parseNum(args[2], 'cutinner');
      cut = parseNum(args[3], 'cutoff');
    }
    const nt = this.ntypes + 1;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.cutIn[i * nt + j] = cin;
        this.cutOut[i * nt + j] = cut;
        this.setflag[i * nt + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** Both orders of a type pair get the cutoffs; an unset pair mixes them (native init_one). */
  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (i === j && !this.setflag[i * nt + i]) throw new StyleError('All pair coeffs are not set');
    if (!this.setflag[i * nt + j]) {
      this.cutIn[i * nt + j] = mixDistance(this.mix, this.cutIn[i * nt + i], this.cutIn[j * nt + j]);
      this.cutOut[i * nt + j] = mixDistance(this.mix, this.cutOut[i * nt + i], this.cutOut[j * nt + j]);
    }
    this.cutIn[j * nt + i] = this.cutIn[i * nt + j];
    return this.cutOut[i * nt + j];
  }

  /** The native checks shared by both styles (fix deform and fix wall), in the native order. */
  protected checkDeformAndWalls(ctx: StyleContext): void {
    const df = ctx.fixOf?.('deform') ?? null;
    this.shearing = false;
    this.flagdeform = false;
    if (df) {
      this.shearing = true;
      this.flagdeform = true;
      if (df instanceof FixDeform) {
        this.deformFix = df;
        if (df.remapFlag() !== 'v') throw new StyleError(`Using pair ${this.name} with inconsistent fix deform remap option`);
      }
    }
    this.wallFix = null;
    const found: Fix[] = [];
    for (const st of WALL_STYLES) {
      const f = ctx.fixOf?.(st);
      if (f) found.push(f);
    }
    if (found.length > 1) throw new StyleError(`Cannot use multiple fix wall commands with pair ${this.name}`);
    this.flagwall = 0;
    if (found.length === 1) {
      const f = found[0];
      if (!(f instanceof FixWallBase)) throw new StyleError(`Fix ${f.style} is not compatible with pair ${this.name}`);
      this.wallFix = f;
      this.flagwall = this.wallsMove(f) ? 2 : 1;
    }
  }

  /** native wallfix->xflag: any wall position is a variable ("Moving walls exist"). */
  protected wallsMove(f: FixWallBase): boolean {
    return (f as unknown as WallView).walls.some((w) => !w.coord.edge && typeof w.coord.v === 'object');
  }

  /**
   * The wall positions the native pair uses, one per wall. A constant or EDGE wall is its scaled coord0 (fix wall
   * coords()). A variable wall is the raw equal-style value: pair_lubricate.cpp calls compute_equal(xindex) with no
   * lattice scale (fix_wall.cpp multiplies a variable position by xscale for its own forces, so the two differ when the
   * lattice spacing is not 1). Measured with native LAMMPS (w231co_lubricate_walls_moving).
   */
  protected wallCoords(): Float64Array {
    const c = Float64Array.from((this.wallFix as unknown as CoordsView).coords());
    const f = this.wallFix as unknown as { sys: System; walls: WallView['walls'] };
    f.walls.forEach((w, m) => {
      if (!w.coord.edge && typeof w.coord.v === 'object') c[m] = valueOf(f.sys, w.coord.v as NumOrVar);
    });
    return c;
  }

  /**
   * native: wallhi = prd, walllo = 0, then each wall replaces the lo or hi bound of its dimension. Returns the
   * available lengths (vol_T = product of them).
   */
  protected wallDims(lx: number, ly: number, lz: number): [number, number, number] {
    const hiv = [lx, ly, lz], lov = [0, 0, 0];
    const walls = (this.wallFix as unknown as WallView).walls;
    const c = this.wallCoords();
    for (let m = 0; m < walls.length; m++) {
      const dim = walls[m].dim, lo = walls[m].lo;
      if (lo) lov[dim] = c[m];
      else hiv[dim] = c[m];
    }
    return [hiv[0] - lov[0], hiv[1] - lov[1], hiv[2] - lov[2]];
  }

  /** The isotropic FLD constants at volume fraction vol_f (native init_style and the compute-time update). */
  protected setIsotropic(volF: number): void {
    const mu = this.mu, rad = this.rad;
    if (this.flaglog === 0) {
      this.R0 = 6 * Math.PI * mu * rad * (1.0 + 2.16 * volF);
      this.RT0 = 8 * Math.PI * mu * libmPow(rad, 3.0);
      this.RS0 = 20.0 / 3.0 * Math.PI * mu * libmPow(rad, 3.0) * (1.0 + 3.33 * volF + 2.80 * volF * volF);
    } else {
      this.R0 = 6 * Math.PI * mu * rad * (1.0 + 2.725 * volF - 6.583 * volF * volF);
      this.RT0 = 8 * Math.PI * mu * libmPow(rad, 3.0) * (1.0 + 0.749 * volF - 2.469 * volF * volF);
      this.RS0 = 20.0 / 3.0 * Math.PI * mu * libmPow(rad, 3.0) * (1.0 + 3.64 * volF - 6.95 * volF * volF);
    }
  }

  /** vol_T of init_style: the box, or the region between the walls. */
  protected initVolumeT(s: { box: { lo: number[]; hi: number[] } }): number {
    const lx = s.box.hi[0] - s.box.lo[0], ly = s.box.hi[1] - s.box.lo[1], lz = s.box.hi[2] - s.box.lo[2];
    if (!this.wallFix) return lx * ly * lz;
    const d = this.wallDims(lx, ly, lz);
    return d[0] * d[1] * d[2];
  }

  protected checkCommon(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) throw new StyleError(`pair_style ${this.name} needs the atoms (define the box and atoms first)`);
    if (!s.omega) throw new StyleError(`Pair ${this.name} requires atom attribute omega`);
    if (!s.radius) throw new StyleError(`Pair ${this.name} requires atom attribute radius`);
    if (!ctx.ghostVelocity) throw new StyleError(`Pair ${this.name} requires ghost atoms store velocity`);
  }

  initStyle(ctx: StyleContext): void {
    // the half list of this port is the newton on list (native default); newton off would need native's j < nlocal tallies
    if (ctx.newtonPair === false) throw new StyleError('pair lubricate with newton pair off is not supported (use newton on)');
    this.checkCommon(ctx);
    const s = ctx.s!;
    // native: radius_consistency per type and the single radius of all types (-1 for a type without atoms)
    for (let t = 1; t <= this.ntypes; t++) {
      let value = -1;
      let flag = false;
      for (let k = 0; k < s.n; k++) {
        if (s.type[k] !== t) continue;
        if (value < 0) value = s.radius![k];
        else if (value !== s.radius![k]) flag = true;
      }
      if (flag) throw new StyleError(`Pair ${this.name} requires monodisperse particles`);
      if (t > 1 && value !== this.rad) throw new StyleError(`Pair ${this.name} requires monodisperse particles`);
      this.rad = value;
    }
    this.checkDeformAndWalls(ctx);
    const volT = this.initVolumeT(s);
    // native: vol_P = atom->natoms*(4.0/3.0)*MY_PI*pow(rad,3.0)
    this.volP = s.n * (4.0 / 3.0) * Math.PI * libmPow(this.rad, 3.0);
    let volF = this.volP / volT;
    if (!this.flagVF) volF = 0;
    this.setIsotropic(volF);
    this.Ef.fill(0);
  }

  /** Native compute-time volume fraction update (flagVF with a moving box or moving walls). */
  protected updateVolumeFraction(pc: PairCompute): void {
    if (!this.flagVF) return;
    if (!(this.flagdeform || this.flagwall === 2)) return;
    const g = pc.geom;
    let dims: [number, number, number];
    if (this.flagdeform && !this.flagwall) dims = [g.lx, g.ly, g.lz];
    else if (this.flagwall === 2 || (this.flagdeform && this.flagwall === 1)) dims = this.wallDims(g.lx, g.ly, g.lz);
    else return;
    const volT = dims[0] * dims[1] * dims[2];
    this.setIsotropic(this.volP / volT);
  }

  /**
   * Native compute start: the streaming velocity of a fix deform is subtracted from the owned atoms' v and omega, and the
   * strain rate Ef set (native "subtract streaming component of velocity, omega, angmom"). Returns null when not shearing.
   */
  protected shearStart(pc: PairCompute): Shear | null {
    if (!this.shearing) return null;
    const s = pc.s;
    if (!this.deformFix) throw new StyleError(`pair ${this.name} with a fix deform that is not a fix deform cannot shear`);
    let rates: { rate: number[]; ratelo: number[] };
    try {
      rates = this.deformFix.streamingRates();
    } catch (e) {
      // the fix's own refusal names fix srd; this style needs the same rates, so say which style refuses them
      if (e instanceof StyleError) throw new StyleError(`pair ${this.name} with a fix deform whose box rate changes during the run is not supported (${e.message})`);
      throw e;
    }
    const { rate, ratelo } = rates;
    const h = rate, hl = ratelo;
    const g = pc.geom;
    const lam = [0, 0, 0];
    const v = s.v, omega = s.omega!;
    for (let i = 0; i < pc.nlocal; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      const vs0 = h[0] * lam[0] + h[5] * lam[1] + h[4] * lam[2] + hl[0];
      const vs1 = h[1] * lam[1] + h[3] * lam[2] + hl[1];
      const vs2 = h[2] * lam[2] + hl[2];
      v[3 * i] -= vs0;
      v[3 * i + 1] -= vs1;
      v[3 * i + 2] -= vs2;
      omega[3 * i] += 0.5 * h[3];
      omega[3 * i + 1] -= 0.5 * h[4];
      omega[3 * i + 2] += 0.5 * h[5];
    }
    const E = this.Ef;
    E[0] = h[0] / g.lx;
    E[4] = h[1] / g.ly;
    E[8] = h[2] / g.lz;
    E[1] = E[3] = 0.5 * h[5] / g.ly;
    E[2] = E[6] = 0.5 * h[4] / g.lz;
    E[5] = E[7] = 0.5 * h[3] / g.lz;
    return { h, hl };
  }

  /** Native compute end: the streaming velocity is added back ("restore streaming component"). */
  protected shearEnd(pc: PairCompute, sh: Shear): void {
    const s = pc.s;
    const { h, hl } = sh;
    const g = pc.geom;
    const lam = [0, 0, 0];
    const v = s.v, omega = s.omega!;
    for (let i = 0; i < pc.nlocal; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      const vs0 = h[0] * lam[0] + h[5] * lam[1] + h[4] * lam[2] + hl[0];
      const vs1 = h[1] * lam[1] + h[3] * lam[2] + hl[1];
      const vs2 = h[2] * lam[2] + hl[2];
      v[3 * i] += vs0;
      v[3 * i + 1] += vs1;
      v[3 * i + 2] += vs2;
      omega[3 * i] -= 0.5 * h[3];
      omega[3 * i + 1] += 0.5 * h[4];
      omega[3 * i + 2] -= 0.5 * h[5];
    }
  }

  /** Velocities and omegas of owned and ghost atoms (native forward comm: a ghost copies its owner). */
  protected ghostState(pc: PairCompute): { vAll: Float64Array; wAll: Float64Array } {
    const s = pc.s, owner = pc.nb.owner, nall = pc.nall;
    const vAll = new Float64Array(3 * nall), wAll = new Float64Array(3 * nall);
    const omega = s.omega!;
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      vAll[3 * k] = s.v[3 * o];
      vAll[3 * k + 1] = s.v[3 * o + 1];
      vAll[3 * k + 2] = s.v[3 * o + 2];
      wAll[3 * k] = omega[3 * o];
      wAll[3 * k + 1] = omega[3 * o + 1];
      wAll[3 * k + 2] = omega[3 * o + 2];
    }
    return { vAll, wAll };
  }

  protected vxmu2fOf(pc: PairCompute): number {
    const u = VXMU2F[pc.s.units.style];
    if (u === undefined) throw new StyleError(`pair ${this.name}: no force->vxmu2f for units ${pc.s.units.style}`);
    return u;
  }

  /** Virial of one pair as ev_tally_xyz tallies it (newton on: the full value to the global virial). */
  protected tallyVirial(pc: PairCompute, i: number, j: number, delx: number, dely: number, delz: number, fx: number, fy: number, fz: number, newtonHalf: boolean): void {
    const v = pc.acc.virial;
    const k = newtonHalf ? 1 : 0.5;
    const v0 = delx * fx, v1 = dely * fy, v2 = delz * fz, v3 = delx * fy, v4 = delx * fz, v5 = dely * fz;
    v[0] += k * v0; v[1] += k * v1; v[2] += k * v2; v[3] += k * v3; v[4] += k * v4; v[5] += k * v5;
    if (pc.vatom) {
      const va = pc.vatom;
      const vs = [v0, v1, v2, v3, v4, v5];
      for (let c = 0; c < 6; c++) va[6 * i + c] += 0.5 * vs[c];
      if (j >= 0) for (let c = 0; c < 6; c++) va[6 * j + c] += 0.5 * vs[c];
    }
  }

  compute(pc: PairCompute): void {
    const list = pc.half;
    if (!list) throw new StyleError(`pair ${this.name} needs a half neighbor list`);
    const s = pc.s;
    const vx = this.vxmu2fOf(pc);
    const nall = pc.nall;
    const nt = this.ntypes + 1;
    const mu = this.mu, flaglog = this.flaglog, flagfld = this.flagfld, flagHI = this.flagHI;
    const x = pc.x, f = pc.f, typ = pc.type;
    const radius = s.radius!, omega = s.omega!, v = s.v;
    const torqueAll = new Float64Array(3 * nall);
    const E = this.Ef;
    const sh = this.shearStart(pc);
    this.updateVolumeFraction(pc);
    const { vAll, wAll } = this.ghostState(pc);
    const cutsq = this.cutsq, cutIn = this.cutIn;
    const acc = pc.acc;
    for (let i = 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = typ[i];
      const radi = radius[i];
      const wi0 = omega[3 * i], wi1 = omega[3 * i + 1], wi2 = omega[3 * i + 2];
      if (flagfld) {
        f[3 * i] -= vx * this.R0 * v[3 * i];
        f[3 * i + 1] -= vx * this.R0 * v[3 * i + 1];
        f[3 * i + 2] -= vx * this.R0 * v[3 * i + 2];
        torqueAll[3 * i] -= vx * this.RT0 * wi0;
        torqueAll[3 * i + 1] -= vx * this.RT0 * wi1;
        torqueAll[3 * i + 2] -= vx * this.RT0 * wi2;
        if (this.shearing) {
          const vRS0 = -vx * this.RS0;
          acc.virial[0] += vRS0 * E[0];
          acc.virial[1] += vRS0 * E[4];
          acc.virial[2] += vRS0 * E[8];
          acc.virial[3] += vRS0 * E[1];
          acc.virial[4] += vRS0 * E[2];
          acc.virial[5] += vRS0 * E[5];
          if (pc.vatom) {
            const q = [vRS0 * E[0], vRS0 * E[4], vRS0 * E[8], vRS0 * E[1], vRS0 * E[2], vRS0 * E[5]];
            for (let c = 0; c < 6; c++) pc.vatom[6 * i + c] += 0.5 * q[c] + 0.5 * q[c];
          }
        }
      }
      if (!flagHI) continue;
      const jStart = list.firstneigh[i], jEnd = jStart + list.numneigh[i];
      for (let kk = jStart; kk < jEnd; kk++) {
        const j = list.neighbors[kk] & NEIGHMASK;
        const delx = xtmp - x[3 * j];
        const dely = ytmp - x[3 * j + 1];
        const delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = typ[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const r = Math.sqrt(rsq);
        const wj0 = wAll[3 * j], wj1 = wAll[3 * j + 1], wj2 = wAll[3 * j + 2];
        // xl: point of closest approach on particle i from its center
        const xl0 = -delx / r * radi, xl1 = -dely / r * radi, xl2 = -delz / r * radi;
        // velocities at the point of closest approach: v = v + omega x xl - Ef.xl
        const vi0 = v[3 * i] + (wi1 * xl2 - wi2 * xl1) - (E[0] * xl0 + E[1] * xl1 + E[2] * xl2);
        const vi1 = v[3 * i + 1] + (wi2 * xl0 - wi0 * xl2) - (E[3] * xl0 + E[4] * xl1 + E[5] * xl2);
        const vi2 = v[3 * i + 2] + (wi0 * xl1 - wi1 * xl0) - (E[6] * xl0 + E[7] * xl1 + E[8] * xl2);
        const vj0 = vAll[3 * j] - (wj1 * xl2 - wj2 * xl1) + (E[0] * xl0 + E[1] * xl1 + E[2] * xl2);
        const vj1 = vAll[3 * j + 1] - (wj2 * xl0 - wj0 * xl2) + (E[3] * xl0 + E[4] * xl1 + E[5] * xl2);
        const vj2 = vAll[3 * j + 2] - (wj0 * xl1 - wj1 * xl0) + (E[6] * xl0 + E[7] * xl1 + E[8] * xl2);
        // scalar resistances
        let hsep = r - 2.0 * radi;
        if (r < cutIn[itype * nt + jtype]) hsep = cutIn[itype * nt + jtype] - 2.0 * radi;
        hsep = hsep / radi;
        let asq: number, ash = 0, apu = 0;
        if (flaglog) {
          asq = 6.0 * Math.PI * mu * radi * (1.0 / 4.0 / hsep + 9.0 / 40.0 * libmLog(1.0 / hsep));
          ash = 6.0 * Math.PI * mu * radi * (1.0 / 6.0 * libmLog(1.0 / hsep));
          apu = 8.0 * Math.PI * mu * libmPow(radi, 3.0) * (3.0 / 160.0 * libmLog(1.0 / hsep));
        } else {
          asq = 6.0 * Math.PI * mu * radi * (1.0 / 4.0 / hsep);
        }
        // relative velocity, normal and tangential components
        const vr1 = vi0 - vj0, vr2 = vi1 - vj1, vr3 = vi2 - vj2;
        const vnnr = (vr1 * delx + vr2 * dely + vr3 * delz) / r;
        const vn1 = vnnr * delx / r, vn2 = vnnr * dely / r, vn3 = vnnr * delz / r;
        const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
        let fx = asq * vn1, fy = asq * vn2, fz = asq * vn3;
        if (flaglog) {
          fx = fx + ash * vt1;
          fy = fy + ash * vt2;
          fz = fz + ash * vt3;
        }
        fx *= vx;
        fy *= vx;
        fz *= vx;
        f[3 * i] -= fx;
        f[3 * i + 1] -= fy;
        f[3 * i + 2] -= fz;
        f[3 * j] += fx;
        f[3 * j + 1] += fy;
        f[3 * j + 2] += fz;
        if (flaglog) {
          let tx = xl1 * fz - xl2 * fy;
          let ty = xl2 * fx - xl0 * fz;
          let tz = xl0 * fy - xl1 * fx;
          torqueAll[3 * i] -= vx * tx;
          torqueAll[3 * i + 1] -= vx * ty;
          torqueAll[3 * i + 2] -= vx * tz;
          torqueAll[3 * j] -= vx * tx;
          torqueAll[3 * j + 1] -= vx * ty;
          torqueAll[3 * j + 2] -= vx * tz;
          // torque due to a_pu
          const wdotn = ((wi0 - wj0) * delx + (wi1 - wj1) * dely + (wi2 - wj2) * delz) / r;
          const wt1 = (wi0 - wj0) - wdotn * delx / r;
          const wt2 = (wi1 - wj1) - wdotn * dely / r;
          const wt3 = (wi2 - wj2) - wdotn * delz / r;
          tx = apu * wt1;
          ty = apu * wt2;
          tz = apu * wt3;
          torqueAll[3 * i] -= vx * tx;
          torqueAll[3 * i + 1] -= vx * ty;
          torqueAll[3 * i + 2] -= vx * tz;
          torqueAll[3 * j] += vx * tx;
          torqueAll[3 * j + 1] += vx * ty;
          torqueAll[3 * j + 2] += vx * tz;
        }
        this.tallyVirial(pc, i, j, delx, dely, delz, -fx, -fy, -fz, true);
      }
    }
    if (sh) this.shearEnd(pc, sh);
    pc.nb.reverseSum(torqueAll, 3, s.torque!);
  }
}

/**
 * pair_style lubricate/poly: the same terms with the polydisperse resistances, the full list, and the pair terms applied
 * to atom i only (native PairLubricatePoly::compute; native init_style requires newton pair off and extended particles).
 */
export class PairLubricatePoly extends PairLubricate {
  override readonly name: string = 'lubricate/poly';
  override needsHalf = false;
  override needsFull = true;

  override initStyle(ctx: StyleContext): void {
    // native order: newton pair, ghost velocities, omega, radius, then the extended-particle check
    if (ctx.newtonPair !== false) throw new StyleError('Pair lubricate/poly requires newton pair off');
    if (!ctx.ghostVelocity) throw new StyleError('Pair lubricate/poly requires ghost atoms store velocity');
    if (!ctx.s) throw new StyleError('pair_style lubricate/poly needs the atoms (define the box and atoms first)');
    this.checkCommon(ctx);
    const s = ctx.s;
    for (let k = 0; k < s.n; k++) {
      if (s.radius![k] === 0.0) throw new StyleError('Pair lubricate/poly requires only extended particles');
    }
    this.checkDeformAndWalls(ctx);
    const volT = this.initVolumeT(s);
    let volP = 0.0;
    for (let k = 0; k < s.n; k++) volP += (4.0 / 3.0) * Math.PI * libmPow(s.radius![k], 3.0);
    this.volP = volP;
    let volF = this.volP / volT;
    if (!this.flagVF) volF = 0;
    this.setIsotropic(volF);
    this.Ef.fill(0);
  }

  /** Polydisperse isotropic constants: no radius factor in R0 and RT0 (native pair_lubricate_poly.cpp). */
  protected override setIsotropic(volF: number): void {
    const mu = this.mu;
    if (this.flaglog === 0) {
      this.R0 = 6 * Math.PI * mu * (1.0 + 2.16 * volF);
      this.RT0 = 8 * Math.PI * mu;
      this.RS0 = 20.0 / 3.0 * Math.PI * mu * (1.0 + 3.33 * volF + 2.80 * volF * volF);
    } else {
      this.R0 = 6 * Math.PI * mu * (1.0 + 2.725 * volF - 6.583 * volF * volF);
      this.RT0 = 8 * Math.PI * mu * (1.0 + 0.749 * volF - 2.469 * volF * volF);
      this.RS0 = 20.0 / 3.0 * Math.PI * mu * (1.0 + 3.64 * volF - 6.95 * volF * volF);
    }
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new StyleError(`pair ${this.name} needs a full neighbor list`);
    const s = pc.s;
    const vx = this.vxmu2fOf(pc);
    const nall = pc.nall;
    const nt = this.ntypes + 1;
    const mu = this.mu, flaglog = this.flaglog, flagfld = this.flagfld, flagHI = this.flagHI;
    const x = pc.x, f = pc.f, typ = pc.type, owner = pc.nb.owner;
    const radius = s.radius!, omega = s.omega!, v = s.v;
    const torqueAll = new Float64Array(3 * nall);
    const E = this.Ef;
    const sh = this.shearStart(pc);
    this.updateVolumeFraction(pc);
    const { vAll, wAll } = this.ghostState(pc);
    const cutsq = this.cutsq, cutIn = this.cutIn;
    const acc = pc.acc;
    for (let i = 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = typ[i];
      const radi = radius[i];
      const wi0 = omega[3 * i], wi1 = omega[3 * i + 1], wi2 = omega[3 * i + 2];
      const radi3 = radi * radi * radi;
      if (flagfld) {
        f[3 * i] -= vx * this.R0 * radi * v[3 * i];
        f[3 * i + 1] -= vx * this.R0 * radi * v[3 * i + 1];
        f[3 * i + 2] -= vx * this.R0 * radi * v[3 * i + 2];
        torqueAll[3 * i] -= vx * this.RT0 * radi3 * wi0;
        torqueAll[3 * i + 1] -= vx * this.RT0 * radi3 * wi1;
        torqueAll[3 * i + 2] -= vx * this.RT0 * radi3 * wi2;
        if (this.shearing) {
          const vRS0 = -vx * this.RS0 * radi3;
          acc.virial[0] += vRS0 * E[0];
          acc.virial[1] += vRS0 * E[4];
          acc.virial[2] += vRS0 * E[8];
          acc.virial[3] += vRS0 * E[1];
          acc.virial[4] += vRS0 * E[2];
          acc.virial[5] += vRS0 * E[5];
          if (pc.vatom) {
            const q = [vRS0 * E[0], vRS0 * E[4], vRS0 * E[8], vRS0 * E[1], vRS0 * E[2], vRS0 * E[5]];
            for (let c = 0; c < 6; c++) pc.vatom[6 * i + c] += 0.5 * q[c] + 0.5 * q[c];
          }
        }
      }
      if (!flagHI) continue;
      const jStart = list.firstneigh[i], jEnd = jStart + list.numneigh[i];
      for (let kk = jStart; kk < jEnd; kk++) {
        const j = list.neighbors[kk] & NEIGHMASK;
        const delx = xtmp - x[3 * j];
        const dely = ytmp - x[3 * j + 1];
        const delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = typ[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const radj = radius[owner[j]];
        const r = Math.sqrt(rsq);
        const wj0 = wAll[3 * j], wj1 = wAll[3 * j + 1], wj2 = wAll[3 * j + 2];
        const xl0 = -delx / r * radi, xl1 = -dely / r * radi, xl2 = -delz / r * radi;
        const jl0 = -delx / r * radj, jl1 = -dely / r * radj, jl2 = -delz / r * radj;
        const vi0 = v[3 * i] + (wi1 * xl2 - wi2 * xl1) - (E[0] * xl0 + E[1] * xl1 + E[2] * xl2);
        const vi1 = v[3 * i + 1] + (wi2 * xl0 - wi0 * xl2) - (E[3] * xl0 + E[4] * xl1 + E[5] * xl2);
        const vi2 = v[3 * i + 2] + (wi0 * xl1 - wi1 * xl0) - (E[6] * xl0 + E[7] * xl1 + E[8] * xl2);
        const vj0 = vAll[3 * j] - (wj1 * jl2 - wj2 * jl1) + (E[0] * jl0 + E[1] * jl1 + E[2] * jl2);
        const vj1 = vAll[3 * j + 1] - (wj2 * jl0 - wj0 * jl2) + (E[3] * jl0 + E[4] * jl1 + E[5] * jl2);
        const vj2 = vAll[3 * j + 2] - (wj0 * jl1 - wj1 * jl0) + (E[6] * jl0 + E[7] * jl1 + E[8] * jl2);
        let hsep = r - radi - radj;
        if (r < cutIn[itype * nt + jtype]) hsep = cutIn[itype * nt + jtype] - radi - radj;
        hsep = hsep / radi;
        const beta0 = radj / radi;
        const beta1 = 1.0 + beta0;
        let asq: number, ash = 0, apu = 0;
        if (flaglog) {
          asq = beta0 * beta0 / beta1 / beta1 / hsep +
            (1.0 + 7.0 * beta0 + beta0 * beta0) / 5.0 / libmPow(beta1, 3.0) * libmLog(1.0 / hsep);
          asq += (1.0 + 18.0 * beta0 - 29.0 * beta0 * beta0 + 18.0 *
                  libmPow(beta0, 3.0) + libmPow(beta0, 4.0)) / 21.0 / libmPow(beta1, 4.0) *
            hsep * libmLog(1.0 / hsep);
          asq *= 6.0 * Math.PI * mu * radi;
          ash = 4.0 * beta0 * (2.0 + beta0 + 2.0 * beta0 * beta0) / 15.0 / libmPow(beta1, 3.0) *
            libmLog(1.0 / hsep);
          ash += 4.0 * (16.0 - 45.0 * beta0 + 58.0 * beta0 * beta0 - 45.0 * libmPow(beta0, 3.0) +
                        16.0 * libmPow(beta0, 4.0)) / 375.0 / libmPow(beta1, 4.0) *
            hsep * libmLog(1.0 / hsep);
          ash *= 6.0 * Math.PI * mu * radi;
          apu = 2.0 * beta0 / 5.0 / beta1 * libmLog(1.0 / hsep);
          apu += 2.0 * (8.0 + 6.0 * beta0 + 33.0 * beta0 * beta0) / 125.0 / beta1 / beta1 *
            hsep * libmLog(1.0 / hsep);
          apu *= 8.0 * Math.PI * mu * libmPow(radi, 3.0);
        } else {
          asq = 6.0 * Math.PI * mu * radi * (beta0 * beta0 / beta1 / beta1 / hsep);
        }
        const vr1 = vi0 - vj0, vr2 = vi1 - vj1, vr3 = vi2 - vj2;
        const vnnr = (vr1 * delx + vr2 * dely + vr3 * delz) / r;
        const vn1 = vnnr * delx / r, vn2 = vnnr * dely / r, vn3 = vnnr * delz / r;
        const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
        let fx = asq * vn1, fy = asq * vn2, fz = asq * vn3;
        if (flaglog) {
          fx = fx + ash * vt1;
          fy = fy + ash * vt2;
          fz = fz + ash * vt3;
        }
        fx *= vx;
        fy *= vx;
        fz *= vx;
        f[3 * i] -= fx;
        f[3 * i + 1] -= fy;
        f[3 * i + 2] -= fz;
        if (flaglog) {
          let tx = xl1 * fz - xl2 * fy;
          let ty = xl2 * fx - xl0 * fz;
          let tz = xl0 * fy - xl1 * fx;
          torqueAll[3 * i] -= vx * tx;
          torqueAll[3 * i + 1] -= vx * ty;
          torqueAll[3 * i + 2] -= vx * tz;
          const wdotn = ((wi0 - wj0) * delx + (wi1 - wj1) * dely + (wi2 - wj2) * delz) / r;
          const wt1 = (wi0 - wj0) - wdotn * delx / r;
          const wt2 = (wi1 - wj1) - wdotn * dely / r;
          const wt3 = (wi2 - wj2) - wdotn * delz / r;
          tx = apu * wt1;
          ty = apu * wt2;
          tz = apu * wt3;
          torqueAll[3 * i] -= vx * tx;
          torqueAll[3 * i + 1] -= vx * ty;
          torqueAll[3 * i + 2] -= vx * tz;
        }
        // native ev_tally_xyz(i, nlocal, nlocal, newton off): half of the pair virial is tallied for atom i
        this.tallyVirial(pc, i, -1, delx, dely, delz, -fx, -fy, -fz, false);
      }
    }
    if (sh) this.shearEnd(pc, sh);
    pc.nb.reverseSum(torqueAll, 3, s.torque!);
  }
}
