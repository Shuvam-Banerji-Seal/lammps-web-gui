/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/msm.cpp and src/KSPACE/msm.h at commit b332f28 (Paul Crozier, Stan Moore and Stephen Bond, SNL),
 * src/KSPACE/msm_cg.cpp (Axel Kohlmeyer, Temple U) and the KSpace helpers they use from src/kspace.cpp (qsum_qsq,
 * two_charge, estimate_table_accuracy, x2lamdaT, lamda2xT, lamda2xvector) and src/math_extra.cpp (tribbox). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * kspace_style msm and msm/cg: the multilevel summation method (Hardy). The charges are interpolated to the finest
 * grid with the order-p polynomial phi (compute_phi), restricted to coarser and coarser grids (a factor 2 per level),
 * and on each level the direct sum with the weights g_n(r) = gamma(r/(2^n a))/(2^n a) - gamma(r/(2^(n+1) a))/(2^(n+1) a)
 * is done over the hemisphere of grid points within 2a/h; the top level of a non-periodic system uses the full
 * gamma(r/(2^n a))/(2^n a). The potentials are prolongated back to the finest grid and interpolated to the atoms.
 *
 * Native structure kept: the grid choice (set_grid_global: Hardy's cost estimate with cutoff/adjust, else doubling the
 * grid until the error estimate is met), the cutoff adjustment written back to the pair style, the extra grid points of
 * non-periodic dimensions (alpha, beta), the hemisphere loops and their order of accumulation, and the fast scalar
 * pressure (kspace_modify pressure/scalar yes, the default: the virial diagonal is E/3). The ghost grid points of
 * periodic dimensions are folded into (reverse) or copied from (forward) their periodic images one dimension at a
 * time, the brick order of native Grid3d (z, y, x in reverse, x, y, z in forward) on one process. A triclinic box runs
 * in lamda coordinates, and the owned positions go through x2lamda and lamda2x as natively (the round trip can change
 * the last bit of a coordinate).
 */
import { KSpaceBase, ANGSTROM, QELECTRON } from '../../engine/force/kspace/ewald';
import { Pair, StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { hasChargeStyle } from '../../engine/atoms';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import { pow } from '../libm/pow';
import { log } from '../libm/log';
import { msmGamma, msmDgamma } from './tables';

const MAX_LEVELS = 10;
const OFFSET = 16384;
/** kspace.cpp SMALL: the net charge above which a system is not neutral. */
const SMALL = 0.00001;
/** msm_cg.cpp SMALLQ: the default charge below which msm/cg skips an atom. */
const SMALLQ = 0.00001;

/** A pair style that works with MSM (native Pair::msmflag): the coul/msm family. */
export interface MsmPair {
  readonly msmflag: true;
  /** The pair style's Coulomb cutoff (pair extract cut_coul). */
  cutCoul: number;
}
export const isMsmPair = (p: Pair | null | undefined): p is Pair & MsmPair => !!p && (p as Partial<MsmPair>).msmflag === true;

/** One MSM grid level: global size, owned (in) and owned+ghost (out) bounds, and its grids. */
interface Level {
  nx: number; ny: number; nz: number;
  xlo: number; xhi: number; ylo: number; yhi: number; zlo: number; zhi: number;
  oxlo: number; oxhi: number; oylo: number; oyhi: number; ozlo: number; ozhi: number;
  sx: number; sy: number; sz: number;
  q: Float64Array;
  e: Float64Array;
  /** Per-atom virial grids v0..v5 (allocate_peratom), allocated when first needed. */
  v: Float64Array[] | null;
}

const numeric = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file (${what})`);
  return v;
};
const inumeric = (w: string | undefined, what: string): number => {
  if (w === undefined || !/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file (${what})`);
  return Number(w);
};
const logical = (w: string | undefined, key: string): boolean => {
  if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
  if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in input script or data file (kspace_modify ${key})`);
};

/** C++ static_cast<int> of a double (truncation toward zero). */
const toInt = (x: number): number => Math.trunc(x);

export class KSpaceMSM extends KSpaceBase {
  readonly name: string = 'msm';
  /** kspace_modify order (MSM default 10; must be 4, 6, 8 or 10). */
  order = 10;
  /** kspace_modify cutoff/adjust (default yes). */
  adjustCutoff = true;
  /** kspace_modify mesh: the user's grid (gridflag) or null. */
  meshUser: [number, number, number] | null = null;
  /** The Coulomb cutoff a (the pair style's cut_coul, after cutoff/adjust). */
  cutoff = 0;
  protected qsum2 = 0;
  /** q2 = qsqsum * qqrd2e (KSpace::qsum_qsq). */
  protected q2 = 0;
  protected natomsOriginal = -1;
  protected accuracyForce = 0;
  protected twoChargeForce = 0;
  protected cP = 1;
  protected hx = 0; protected hy = 0; protected hz = 0;
  protected levels = 0;
  protected lv: Level[] = [];
  protected alpha: number[] = [];
  protected betax: number[] = []; protected betay: number[] = []; protected betaz: number[] = [];
  protected delxinv: number[] = []; protected delyinv: number[] = []; protected delzinv: number[] = [];
  protected nxhiDirect = 0; protected nyhiDirect = 0; protected nzhiDirect = 0;
  protected gDirect: Float64Array[] = [];
  protected vDirect: Float64Array[][] = [];
  protected gDirectTop = new Float64Array(0);
  protected vDirectTop: Float64Array[] = [];
  protected nlower = 0; protected nupper = 0;
  protected periodic: [boolean, boolean, boolean] = [true, true, true];
  protected nonperiodic = false;
  protected triclinic = false;
  /** domain->h and h_inv of the last setup (triclinic). */
  protected h = new Float64Array(6);
  protected hInv = new Float64Array(6);
  /** prd and tilt of the last setup(); a change re-runs setup (fix nh, deform and shrink-wrap call kspace->setup()). */
  private setupKey = '';
  /** The pair style (or its MSM sub-style) whose cut_coul cutoff/adjust sets (pair extract "cut_coul"). */
  private coulPair: (Pair & MsmPair) | null = null;
  /** preparePair ran the full init for the current ForceField.init. */
  private prepared = false;
  // per compute: energy and virial accumulators
  protected energy = 0;
  protected virial = new Float64Array(6);
  protected eflagAtom = false;
  protected vflagAtom = false;
  protected vflagEither = true;
  /** part2grid of each owned atom, and the positions the grid sees (lamda for triclinic). */
  protected part2grid = new Int32Array(0);
  protected xg = new Float64Array(0);
  protected phi1d = [new Float64Array(0), new Float64Array(0), new Float64Array(0)];
  protected dphi1d = [new Float64Array(0), new Float64Array(0), new Float64Array(0)];

  constructor() {
    super();
    // msm.cpp constructor: "scalar_pressure_flag = 1"; kspace_modify.html default "pressure/scalar = yes (MSM)"
    this.scalarPressure = true;
  }

  /** MSM::settings: kspace_style msm accuracy (the relative accuracy, absolute value taken, at most 1). */
  settings(args: string[]): void {
    if (args.length < 1) throw new StyleError(`Illegal kspace_style ${this.name} command: missing argument(s)`);
    this.accuracy = Math.abs(numeric(args[0], 'accuracy'));
    if (this.accuracy > 1.0) throw new StyleError(`Invalid relative accuracy ${this.accuracy} for kspace_style ${this.name}`);
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'mesh': {
        if (values.length < 3) throw new StyleError('Illegal kspace_modify mesh command: missing argument(s)');
        const m = [0, 1, 2].map((d) => inumeric(values[d], 'kspace_modify mesh')) as [number, number, number];
        if (m.every((v) => v === 0)) this.meshUser = null;
        else if (m.some((v) => v <= 0)) throw new StyleError('Kspace_modify mesh parameters must be all zero or all positive');
        else this.meshUser = m;
        return 3;
      }
      case 'order': this.order = inumeric(values[0], 'kspace_modify order'); return 1;
      case 'force': this.forceAbs = numeric(values[0], 'kspace_modify force'); return 1;
      case 'cutoff/adjust': this.adjustCutoff = logical(values[0], key); return 1;
      case 'pressure/scalar': this.scalarPressure = logical(values[0], key); return 1;
      // kspace.cpp modify_params checks splittol (and its alias eigtol) for every solver; msm.cpp never reads it
      case 'splittol':
      case 'eigtol': {
        const v = numeric(values[0], `kspace_modify ${key}`);
        if (v >= 1.0) throw new StyleError('Kspace_modify splittol value must be smaller than one');
        return 1;
      }
      // kspace.cpp modify_params sets g_ewald (g_ewald_6) for every solver; MSM has no use for either
      case 'gewald':
      case 'gewald/disp': numeric(values[0], `kspace_modify ${key}`); return 1;
      // slab (warned about and ignored at init), minorder, overlap and the FFT keywords: KSpaceBase
      default: return super.modify(key, values);
    }
  }

  /** KSpace::qsum_qsq with warn_nonneutral = 0 (MSM stops on a non-neutral system). */
  protected qsumQsq(s: SimState, qqrd2e: number, ctx: StyleContext | null): void {
    let qs = 0, q2 = 0;
    for (let i = 0; i < s.n; i++) { qs += s.q[i]; q2 += s.q[i] * s.q[i]; }
    this.qsum = qs;
    this.qsqsum = q2;
    if (q2 === 0 && ctx) ctx.log('WARNING: Using kspace solver on system with no charge');
    this.q2 = q2 * qqrd2e;
    if (Math.abs(qs) > SMALL) throw new StyleError(`System is not charge neutral, net charge = ${qs.toPrecision(8)}`);
  }

  updateChargeSums(s: SimState): void {
    this.qsumQsq(s, this.qqrd2eRun, null);
    this.natomsOriginal = s.n;
  }

  /** force->qqrd2e of the last init (qsum_qsq, and single() of the coul/msm styles). */
  qqrd2eRun = 1;

  /**
   * MSM::init, run before the pair style's init (native Force::init: "kspace must come before pair"), so that the
   * cutoff/adjust value reaches the pair cutoffs and the neighbor list.
   */
  preparePair(pair: Pair | null, s: SimState, geom: Geometry, qqrd2e: number, ctx: StyleContext): void {
    this.prepared = false;
    ctx.log('MSM initialization ...');
    geom = this.initBox(s, geom, ctx);
    // triclinic_check: msm supports triclinic boxes (triclinic_support stays 1), msm/cg does not
    if (s.box.triclinic && !this.triclinicSupport) throw new StyleError(`KSpace style ${this.name} does not yet support triclinic geometries`);
    if (s.dimension === 2) throw new StyleError('Cannot (yet) use MSM with 2d simulation');
    if (!hasChargeStyle(s.atomStyle)) throw new StyleError('Kspace style requires atom attribute q');
    if (this.slab > 1 || this.slabAuto) {
      ctx.log('WARNING: Slab correction not needed for MSM and will be ignored');
      this.slab = 1;
      this.slabAuto = false;
    }
    if (this.order < 4 || this.order > 10 || this.order % 2 !== 0) throw new StyleError('MSM order must be 4, 6, 8, or 10');
    const u = s.units;
    const ang = ANGSTROM[u.style], qe = QELECTRON[u.style];
    this.twoChargeForce = (u.qqr2e * (qe * qe)) / (ang * ang);
    this.triclinic = s.box.triclinic;
    // pair_check: the pair style must be an MSM style (msmflag), and cut_coul comes from it
    if (!pair) throw new StyleError(`KSpace style ${this.name} requires a pair style`);
    const subs = pair instanceof PairHybrid ? pair.subs.map((x) => x.style) : [pair];
    if (!subs.some(isMsmPair)) throw new StyleError(`KSpace style ${this.name} is incompatible with Pair style ${pair.name}`);
    // pair_coul_msm.cpp adds force->pair->eng_coul/3 to the virial in scalar-pressure mode: under pair hybrid that is
    // the hybrid's sum of the sub-styles evaluated before, not the style's own Coulomb energy
    if (pair instanceof PairHybrid && this.scalarPressure) {
      throw new StyleError(`kspace_modify pressure/scalar yes with pair_style ${pair.name} is not supported by the browser engine (native LAMMPS adds the hybrid style's partial Coulomb energy to the virial); use kspace_modify pressure/scalar no`);
    }
    const cut = pair.extract('cut_coul');
    if (typeof cut !== 'number') throw new StyleError('KSpace style is incompatible with Pair style');
    // PairHybrid::extract: the cut_coul pointer is that of the last sub-style that has one
    let last: Pair | null = null;
    for (const p of subs) if (typeof p.extract('cut_coul') === 'number') last = p;
    this.coulPair = isMsmPair(last) ? last : null;
    this.cutoff = cut;
    this.qqrd2eRun = qqrd2e;
    this.qsumQsq(s, qqrd2e, ctx);
    this.natomsOriginal = s.n;
    this.accuracyForce = this.forceAbs >= 0 ? this.forceAbs : this.accuracy * this.twoChargeForce;
    this.periodic = [...geom.periodic] as [boolean, boolean, boolean];
    this.nonperiodic = !this.periodic[0] || !this.periodic[1] || !this.periodic[2];
    this.setGridGlobal(s, geom, ctx);
    this.setup(geom, s);
    const err = this.estimateTotalError(s, geom, pair, ctx);
    const L0 = this.lv[0];
    const ngridMax = L0.sx * L0.sy * L0.sz;
    ctx.log(`  3d grid size/proc = ${ngridMax}\n  estimated absolute RMS force accuracy = ${err.toPrecision(8)}\n`
      + `  estimated relative force accuracy = ${(err / this.twoChargeForce).toPrecision(8)}\n`
      + `  grid = ${L0.nx} ${L0.ny} ${L0.nz}\n  order = ${this.order}`);
    this.prepared = true;
  }

  /**
   * The box MSM::init sees: before the first reset after create_box, native's shrink-wrapped faces are still moved out
   * by SMALL = 1e-4 of the created length (src/domain.cpp set_initial_box), which the engine keeps aside
   * (StyleContext.initialBoxExpansion). The grid choice and the adjusted cutoff use those lengths; the set-up is re-run
   * with the shrink-wrapped box at the first compute.
   */
  private initBox(s: SimState, geom: Geometry, ctx: StyleContext): Geometry {
    const small = ctx.initialBoxExpansion;
    if (!small) return geom;
    const lo = [...geom.lo] as [number, number, number], hi = [...geom.hi] as [number, number, number];
    let changed = false;
    for (let d = 0; d < 3; d++) {
      if (s.box.boundary[d][0] === 's') { lo[d] -= small[d]; changed = true; }
      if (s.box.boundary[d][1] === 's') { hi[d] += small[d]; changed = true; }
    }
    if (!changed) return geom;
    const g = Object.assign(Object.create(Object.getPrototypeOf(geom) as object) as Geometry, geom);
    g.lo = lo; g.hi = hi;
    g.lx = hi[0] - lo[0]; g.ly = hi[1] - lo[1]; g.lz = hi[2] - lo[2];
    return g;
  }

  /** msm supports triclinic boxes; msm/cg sets triclinic_support = 0. */
  protected triclinicSupport = true;

  /** The work of MSM::init is done by preparePair, which the force field runs before the pair style's init. */
  init(_s: SimState, _geom: Geometry, _cutCoul: number, _qqrd2e: number, _ctx: StyleContext): void {
    if (!this.prepared) throw new StyleError(`kspace_style ${this.name}: the force field did not run the MSM set-up before the pair style`);
    this.prepared = false;
  }

  /** MSM::estimate_1d_error (Hardy's thesis: Mp from table 5.1, cprime from eq. 4.17, eq. 3.197). */
  protected estimate1dError(h: number, prd: number, natoms: number): number {
    const a = this.cutoff;
    const p = this.order - 1;
    let Mp = 1, cprime = 1, errorScaling = 1;
    if (p === 3) { Mp = 9; cprime = 1.0 / 6.0; errorScaling = 0.39189561; }
    else if (p === 5) { Mp = 825; cprime = 1.0 / 30.0; errorScaling = 0.150829428; }
    else if (p === 7) { Mp = 130095; cprime = 1.0 / 140.0; errorScaling = 0.049632967; }
    else if (p === 9) { Mp = 34096545; cprime = 1.0 / 630.0; errorScaling = 0.013520855; }
    else throw new StyleError('MSM order must be 4, 6, 8, or 10');
    this.cP = (4.0 * cprime * Mp) / 3.0;
    this.cP *= errorScaling;
    let error1d = (this.cP * pow(h, p - 1)) / pow(a, p + 1);
    error1d *= (this.q2 * a) / (prd * Math.sqrt(natoms));
    return error1d;
  }

  /** MSM::estimate_total_error (the grid error with estimate_table_accuracy; for the log). */
  protected estimateTotalError(s: SimState, geom: Geometry, pair: Pair, ctx: StyleContext): number {
    const natoms = s.n;
    const ex = this.estimate1dError(this.hx, geom.lx, natoms);
    const ey = this.estimate1dError(this.hy, geom.ly, natoms);
    const ez = this.estimate1dError(this.hz, geom.lz, natoms);
    const grid = Math.sqrt(ex * ex + ey * ey + ez * ez) / Math.sqrt(3.0);
    const q2OverSqrt = this.q2 / Math.sqrt(natoms * this.cutoff * geom.lx * geom.ly * geom.lz);
    const nctb = pair.table;
    let table = 0;
    if (nctb) {
      ctx.log(`  using ${nctb}-bit tables for long-range coulomb`);
      const prec: Record<number, number> = { 6: 6.99e-3, 7: 1.78e-3, 8: 4.72e-4, 9: 1.17e-4, 10: 2.95e-5, 11: 7.41e-6, 12: 1.76e-6, 13: 9.28e-7, 14: 7.46e-7, 15: 7.32e-7, 16: 7.30e-7 };
      table = (nctb <= 6 ? prec[6] : nctb <= 16 ? prec[nctb] : prec[16]) * q2OverSqrt;
      if (table > 0) ctx.log("WARNING: For better accuracy use 'pair_modify table 0'");
    } else ctx.log('  using polynomial approximation for long-range coulomb');
    return Math.sqrt(grid * grid + table * table);
  }

  /** MSM::factorable with factors = {2}: true when n is a power of 2; nlevels = log2(n) + 1. */
  private static factorable(n: number): { ok: boolean; nlevels: number } {
    let nlevels = 1;
    while (n > 1) {
      if (n % 2 === 0) { n /= 2; nlevels++; }
      else return { ok: false, nlevels };
    }
    return { ok: true, nlevels };
  }

  /** MSM::set_grid_global: grid size, cutoff/adjust, number of levels and the non-periodic extra points. */
  protected setGridGlobal(s: SimState, geom: Geometry, ctx: StyleContext): void {
    if (this.accuracy <= 0.0) throw new StyleError('KSpace accuracy must be > 0');
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    const natoms = s.n;
    const gridflag = this.meshUser !== null;
    let nxMax: number, nyMax: number, nzMax: number;
    const accuracy = this.accuracyForce;
    if (this.adjustCutoff && !gridflag) {
      // Hardy's cost estimate for the optimal Coulombic cutoff and number of grid levels
      const p = this.order - 1;
      let hmin = (3072.0 * (p + 1)) / (p - 1) / (448.0 * Math.PI + (56.0 * Math.PI * this.order) / 2 + 1701.0);
      hmin = pow(hmin, 1.0 / 6.0) * pow((xprd * yprd * zprd) / natoms, 1.0 / 3.0);
      nxMax = Math.max(toInt(xprd / hmin), 2);
      nyMax = Math.max(toInt(yprd / hmin), 2);
      nzMax = Math.max(toInt(zprd / hmin), 2);
    } else if (!gridflag) {
      nxMax = nyMax = nzMax = 2;
      let xError = 2.0 * accuracy, yError = 2.0 * accuracy, zError = 2.0 * accuracy;
      const LIMIT = 1 << 30;
      while (xError > accuracy) {
        nxMax *= 2;
        if (nxMax > LIMIT) throw new StyleError('MSM grid is too large');
        xError = this.estimate1dError(xprd / nxMax, xprd, natoms);
      }
      while (yError > accuracy) {
        nyMax *= 2;
        if (nyMax > LIMIT) throw new StyleError('MSM grid is too large');
        yError = this.estimate1dError(yprd / nyMax, yprd, natoms);
      }
      while (zError > accuracy) {
        nzMax *= 2;
        if (nzMax > LIMIT) throw new StyleError('MSM grid is too large');
        zError = this.estimate1dError(zprd / nzMax, zprd, natoms);
      }
    } else {
      [nxMax, nyMax, nzMax] = this.meshUser!;
    }
    // scale the grid for the triclinic skew (lamda2xT uses |yz|, |xz|, |xy|)
    if (this.triclinic && !gridflag) {
      this.setH(geom);
      const h = this.h;
      const t0 = nxMax / xprd, t1 = nyMax / yprd, t2 = nzMax / zprd;
      const v0 = h[0] * t0;
      const v1 = Math.abs(h[5]) * t0 + h[1] * t1;
      const v2 = Math.abs(h[4]) * t0 + Math.abs(h[3]) * t1 + h[2] * t2;
      nxMax = toInt(v0); nyMax = toInt(v1); nzMax = toInt(v2);
    }
    // boost the grid size until it is a power of 2, rounding up or down, whichever is closer
    let flag = false;
    const pow2 = (n0: number): { n: number; levels: number } => {
      let n = n0;
      for (;;) {
        const f = KSpaceMSM.factorable(n);
        if (f.ok) return { n, levels: f.nlevels };
        flag = true;
        const k = log(n) / log(2.0);
        const r = k - Math.floor(k);
        if (r > 0.5) n++;
        else n--;
      }
    };
    const fx = pow2(nxMax), fy = pow2(nyMax), fz = pow2(nzMax);
    nxMax = fx.n; nyMax = fy.n; nzMax = fz.n;
    let xlevels = fx.levels, ylevels = fy.levels, zlevels = fz.levels;
    if (flag && gridflag) ctx.log('WARNING: Number of MSM mesh points changed to be a multiple of 2');
    if (this.adjustCutoff) {
      const hx = xprd / nxMax, hy = yprd / nyMax, hz = zprd / nzMax;
      const p = this.order - 1;
      const hx2pm2 = pow(hx, 2.0 * p - 2.0), hy2pm2 = pow(hy, 2.0 * p - 2.0), hz2pm2 = pow(hz, 2.0 * p - 2.0);
      this.estimate1dError(1.0, 1.0, natoms); // make sure that C_p is defined
      const k = (this.q2 * this.cP) / accuracy / Math.sqrt(natoms);
      const sum = hx2pm2 / (xprd * xprd) + hy2pm2 / (yprd * yprd) + hz2pm2 / (zprd * zprd);
      this.cutoff = pow((k * k * sum) / 3.0, 1.0 / (2.0 * p));
      if (!this.coulPair) throw new StyleError(`kspace_modify cutoff/adjust yes: the browser engine cannot set the Coulomb cutoff of this pair style (use kspace_modify cutoff/adjust no)`);
      this.coulPair.cutCoul = this.cutoff;
      ctx.log(`WARNING: Adjusting Coulombic cutoff for MSM, new cutoff = ${this.cutoff.toPrecision(8)}`);
    }
    if (!this.triclinic) {
      this.hx = xprd / nxMax; this.hy = yprd / nyMax; this.hz = zprd / nzMax;
    } else {
      // x2lamdaT of (nx, ny, nz)
      this.setH(geom);
      const hi = this.hInv;
      const l0 = hi[0] * nxMax;
      const l1 = hi[5] * nxMax + hi[1] * nyMax;
      const l2 = hi[4] * nxMax + hi[3] * nyMax + hi[2] * nzMax;
      this.hx = 1.0 / l0; this.hy = 1.0 / l1; this.hz = 1.0 / l2;
    }
    let levels = Math.max(xlevels, ylevels, zlevels);
    if (levels > MAX_LEVELS) throw new StyleError('Too many MSM grid levels');
    // at least 2 MSM levels for periodic systems
    if (levels <= 1) {
      levels = xlevels = ylevels = zlevels = 2;
      nxMax = nyMax = nzMax = 2;
      if (gridflag) ctx.log('WARNING: MSM mesh too small, increasing to 2 points in each direction');
    }
    // the top grid level is omitted for periodic systems
    if (!this.nonperiodic) levels -= 1;
    this.levels = levels;
    const nx: number[] = [], ny: number[] = [], nz: number[] = [];
    for (let n = 0; n < levels; n++) {
      nx[n] = xlevels - n - 1 > 0 ? toInt(2 ** (xlevels - n - 1)) : 1;
      ny[n] = ylevels - n - 1 > 0 ? toInt(2 ** (ylevels - n - 1)) : 1;
      nz[n] = zlevels - n - 1 > 0 ? toInt(2 ** (zlevels - n - 1)) : 1;
    }
    if (nx[0] >= OFFSET || ny[0] >= OFFSET || nz[0] >= OFFSET) throw new StyleError('MSM grid is too large');
    // extra grid points for non-periodic boundaries
    const half = (this.order >> 1) - 1;
    this.alpha = [-half];
    this.betax = [nx[0] + half]; this.betay = [ny[0] + half]; this.betaz = [nz[0] + half];
    for (let n = 1; n < levels; n++) {
      this.alpha[n] = -Math.trunc((-this.alpha[n - 1] + 1) / 2) - half;
      this.betax[n] = Math.trunc((this.betax[n - 1] + 1) / 2) + half;
      this.betay[n] = Math.trunc((this.betay[n - 1] + 1) / 2) + half;
      this.betaz[n] = Math.trunc((this.betaz[n - 1] + 1) / 2) + half;
    }
    this.lv = [];
    for (let n = 0; n < levels; n++) {
      this.lv.push({
        nx: nx[n], ny: ny[n], nz: nz[n], xlo: 0, xhi: 0, ylo: 0, yhi: 0, zlo: 0, zhi: 0,
        oxlo: 0, oxhi: 0, oylo: 0, oyhi: 0, ozlo: 0, ozhi: 0, sx: 0, sy: 0, sz: 0,
        q: new Float64Array(0), e: new Float64Array(0), v: null,
      });
    }
    this.setupKey = '';
  }

  /** domain->h = (xprd, yprd, zprd, yz, xz, xy) and its inverse h_inv (domain.cpp set_global_box). */
  protected setH(geom: Geometry): void {
    const h = this.h, hi = this.hInv;
    h[0] = geom.lx; h[1] = geom.ly; h[2] = geom.lz;
    h[3] = geom.yz; h[4] = geom.xz; h[5] = geom.xy;
    hi[0] = 1.0 / h[0]; hi[1] = 1.0 / h[1]; hi[2] = 1.0 / h[2];
    hi[3] = -h[3] / (h[1] * h[2]);
    hi[4] = (h[3] * h[5] - h[1] * h[4]) / (h[0] * h[1] * h[2]);
    hi[5] = -h[5] / (h[0] * h[1]);
  }

  /** MathExtra::tribbox: half-extent in lamda coordinates of a sphere of the given radius. */
  protected tribbox(radius: number): [number, number, number] {
    const [lx, ly, lz, yz, xz, xy] = this.h;
    return [
      (radius * Math.sqrt(ly * ly * lz * lz + ly * ly * xz * xz - 2.0 * ly * xy * xz * yz + lz * lz * xy * xy + xy * xy * yz * yz)) / (lx * ly * lz),
      (radius * Math.sqrt(lz * lz + yz * yz)) / (ly * lz),
      radius / lz,
    ];
  }

  private static key(geom: Geometry): string {
    return `${geom.lx} ${geom.ly} ${geom.lz} ${geom.xy} ${geom.xz} ${geom.yz}`;
  }

  /** MSM::setup: grid spacings, direct-sum limits and weights, local grids. */
  protected setup(geom: Geometry, s: SimState): void {
    const a = this.cutoff;
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    this.volume = xprd * yprd * zprd;
    if (this.triclinic) this.setH(geom);
    for (let n = 0; n < this.levels; n++) {
      const L = this.lv[n];
      if (!this.triclinic) {
        this.delxinv[n] = L.nx / xprd;
        this.delyinv[n] = L.ny / yprd;
        this.delzinv[n] = L.nz / zprd;
      } else {
        this.delxinv[n] = L.nx;
        this.delyinv[n] = L.ny;
        this.delzinv[n] = L.nz;
      }
    }
    let ax = a, ay = a, az = a;
    if (this.triclinic) [ax, ay, az] = this.tribbox(a);
    this.nxhiDirect = toInt(2.0 * ax * this.delxinv[0]);
    this.nyhiDirect = toInt(2.0 * ay * this.delyinv[0]);
    this.nzhiDirect = toInt(2.0 * az * this.delzinv[0]);
    this.getGDirect();
    this.getVirialDirect();
    if (this.nonperiodic) {
      this.getGDirectTop(this.levels - 1);
      this.getVirialDirectTop(this.levels - 1);
    }
    this.setGridLocal(geom, s);
    this.allocate();
    this.setupKey = KSpaceMSM.key(geom);
  }

  /** lamda2xvector: v = h lamda. */
  private lamda2xvector(l0: number, l1: number, l2: number, out: number[]): void {
    const h = this.h;
    out[0] = h[0] * l0 + h[5] * l1 + h[4] * l2;
    out[1] = h[1] * l1 + h[3] * l2;
    out[2] = h[2] * l2;
  }

  /** MSM::get_g_direct: the energy weights of the intermediate levels. */
  private getGDirect(): void {
    const a = this.cutoff;
    const nxhi = this.nxhiDirect, nyhi = this.nyhiDirect, nzhi = this.nzhiDirect;
    const nx = 2 * nxhi + 1, ny = 2 * nyhi + 1, nz = 2 * nzhi + 1;
    const tmp = [0, 0, 0];
    let twoN = 1.0;
    this.gDirect = [];
    for (let n = 0; n < this.levels; n++) {
      const g = new Float64Array(nx * ny * nz);
      for (let iz = -nzhi; iz <= nzhi; iz++) {
        const zdiff = iz / this.delzinv[n];
        const zk = (iz + nzhi) * ny;
        for (let iy = -nyhi; iy <= nyhi; iy++) {
          const ydiff = iy / this.delyinv[n];
          const zyk = (zk + iy + nyhi) * nx;
          for (let ix = -nxhi; ix <= nxhi; ix++) {
            const xdiff = ix / this.delxinv[n];
            let dx = xdiff, dy = ydiff, dz = zdiff;
            if (this.triclinic) { this.lamda2xvector(xdiff, ydiff, zdiff, tmp); [dx, dy, dz] = tmp; }
            const rsq = dx * dx + dy * dy + dz * dz;
            const rho = Math.sqrt(rsq) / (twoN * a);
            g[zyk + ix + nxhi] = msmGamma(this.order, rho) / (twoN * a) - msmGamma(this.order, rho / 2.0) / (2.0 * twoN * a);
          }
        }
      }
      this.gDirect.push(g);
      twoN *= 2.0;
    }
  }

  /** MSM::get_virial_direct: the virial weights of the intermediate levels. */
  private getVirialDirect(): void {
    const a = this.cutoff, aSq = this.cutoff * this.cutoff;
    const nxhi = this.nxhiDirect, nyhi = this.nyhiDirect, nzhi = this.nzhiDirect;
    const nx = 2 * nxhi + 1, ny = 2 * nyhi + 1, nz = 2 * nzhi + 1;
    const tmp = [0, 0, 0];
    let twoN = 1.0;
    this.vDirect = [];
    for (let n = 0; n < this.levels; n++) {
      const twoNsq = twoN * twoN;
      const v = Array.from({ length: 6 }, () => new Float64Array(nx * ny * nz));
      for (let iz = -nzhi; iz <= nzhi; iz++) {
        const zdiff = iz / this.delzinv[n];
        const zk = (iz + nzhi) * ny;
        for (let iy = -nyhi; iy <= nyhi; iy++) {
          const ydiff = iy / this.delyinv[n];
          const zyk = (zk + iy + nyhi) * nx;
          for (let ix = -nxhi; ix <= nxhi; ix++) {
            const xdiff = ix / this.delxinv[n];
            let dx = xdiff, dy = ydiff, dz = zdiff;
            if (this.triclinic) { this.lamda2xvector(xdiff, ydiff, zdiff, tmp); [dx, dy, dz] = tmp; }
            const rsq = dx * dx + dy * dy + dz * dz;
            const k = zyk + ix + nxhi;
            const r = Math.sqrt(rsq);
            if (r === 0) continue;
            const rho = r / (twoN * a);
            const dg = -(msmDgamma(this.order, rho) / (twoNsq * aSq) - msmDgamma(this.order, rho / 2.0) / (4.0 * twoNsq * aSq)) / r;
            v[0][k] = dg * dx * dx; v[1][k] = dg * dy * dy; v[2][k] = dg * dz * dz;
            v[3][k] = dg * dx * dy; v[4][k] = dg * dx * dz; v[5][k] = dg * dy * dz;
          }
        }
      }
      this.vDirect.push(v);
      twoN *= 2.0;
    }
  }

  /** MSM::get_g_direct_top: the full gamma weights of the top level (non-periodic systems). */
  private getGDirectTop(n: number): void {
    const nxTop = this.betax[n] - this.alpha[n], nyTop = this.betay[n] - this.alpha[n], nzTop = this.betaz[n] - this.alpha[n];
    const nx = 2 * nxTop + 1, ny = 2 * nyTop + 1, nz = 2 * nzTop + 1;
    const a = this.cutoff;
    const twoN = 2.0 ** n;
    const g = new Float64Array(nx * ny * nz);
    const tmp = [0, 0, 0];
    for (let iz = -nzTop; iz <= nzTop; iz++) {
      const zdiff = iz / this.delzinv[n];
      const zk = (iz + nzTop) * ny;
      for (let iy = -nyTop; iy <= nyTop; iy++) {
        const ydiff = iy / this.delyinv[n];
        const zyk = (zk + iy + nyTop) * nx;
        for (let ix = -nxTop; ix <= nxTop; ix++) {
          const xdiff = ix / this.delxinv[n];
          let dx = xdiff, dy = ydiff, dz = zdiff;
          if (this.triclinic) { this.lamda2xvector(xdiff, ydiff, zdiff, tmp); [dx, dy, dz] = tmp; }
          const rsq = dx * dx + dy * dy + dz * dz;
          const rho = Math.sqrt(rsq) / (twoN * a);
          g[zyk + ix + nxTop] = msmGamma(this.order, rho) / (twoN * a);
        }
      }
    }
    this.gDirectTop = g;
  }

  /** MSM::get_virial_direct_top. */
  private getVirialDirectTop(n: number): void {
    const nxTop = this.betax[n] - this.alpha[n], nyTop = this.betay[n] - this.alpha[n], nzTop = this.betaz[n] - this.alpha[n];
    const nx = 2 * nxTop + 1, ny = 2 * nyTop + 1, nz = 2 * nzTop + 1;
    const a = this.cutoff, aSq = a * a;
    const twoN = 2.0 ** n, twoNsq = twoN * twoN;
    const v = Array.from({ length: 6 }, () => new Float64Array(nx * ny * nz));
    const tmp = [0, 0, 0];
    for (let iz = -nzTop; iz <= nzTop; iz++) {
      const zdiff = iz / this.delzinv[n];
      const zk = (iz + nzTop) * ny;
      for (let iy = -nyTop; iy <= nyTop; iy++) {
        const ydiff = iy / this.delyinv[n];
        const zyk = (zk + iy + nyTop) * nx;
        for (let ix = -nxTop; ix <= nxTop; ix++) {
          const xdiff = ix / this.delxinv[n];
          let dx = xdiff, dy = ydiff, dz = zdiff;
          if (this.triclinic) { this.lamda2xvector(xdiff, ydiff, zdiff, tmp); [dx, dy, dz] = tmp; }
          const rsq = dx * dx + dy * dy + dz * dz;
          const k = zyk + ix + nxTop;
          const r = Math.sqrt(rsq);
          if (r === 0) continue;
          const rho = r / (twoN * a);
          const dg = -(msmDgamma(this.order, rho) / (twoNsq * aSq)) / r;
          v[0][k] = dg * dx * dx; v[1][k] = dg * dy * dy; v[2][k] = dg * dz * dz;
          v[3][k] = dg * dx * dy; v[4][k] = dg * dx * dz; v[5][k] = dg * dy * dz;
        }
      }
    }
    this.vDirectTop = v;
  }

  /**
   * set_grid_local sizes the ghost region of the finest grid for atoms within skin/2 of the box. The value only sets how
   * far ghost points reach (the out-of-range check); the grids beyond also hold the direct-sum ghosts, which are wider.
   */
  skin = 2.0;

  /** MSM::set_grid_local on one process: the owned grid is the whole grid (plus alpha..beta in non-periodic dims). */
  private setGridLocal(geom: Geometry, _s: SimState): void {
    this.nlower = -Math.trunc((this.order - 1) / 2);
    this.nupper = this.order >> 1;
    const prd = this.triclinic ? [1, 1, 1] : [geom.lx, geom.ly, geom.lz];
    const sublo = this.triclinic ? [0, 0, 0] : geom.lo;
    const subhi = this.triclinic ? [1, 1, 1] : geom.hi;
    const boxlo = this.triclinic ? [0, 0, 0] : geom.lo;
    const order = this.order;
    const direct = [this.nxhiDirect, this.nyhiDirect, this.nzhiDirect];
    for (let n = 0; n < this.levels; n++) {
      const L = this.lv[n];
      const N = [L.nx, L.ny, L.nz];
      const cuthalf = n === 0 ? 0.5 * this.skin : 0.0;
      let dist = [cuthalf, cuthalf, cuthalf];
      if (this.triclinic) dist = this.tribbox(cuthalf);
      const lo = [0, 0, 0], hi = [0, 0, 0], olo = [0, 0, 0], ohi = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        lo[d] = 0;
        hi[d] = N[d] - 1;
        const nlo = toInt(((sublo[d] - dist[d] - boxlo[d]) * N[d]) / prd[d] + OFFSET) - OFFSET;
        const nhi = toInt(((subhi[d] + dist[d] - boxlo[d]) * N[d]) / prd[d] + OFFSET) - OFFSET;
        // the hemisphere of the direct sum needs no ghosts in -z beyond the stencil
        olo[d] = d === 2 ? nlo - order : nlo + Math.min(-order, -direct[d]);
        ohi[d] = nhi + Math.max(order, direct[d]);
        if (this.nonperiodic && !this.periodic[d]) {
          const beta = d === 0 ? this.betax[n] : d === 1 ? this.betay[n] : this.betaz[n];
          if (lo[d] === 0) lo[d] = this.alpha[n];
          olo[d] = Math.max(olo[d], this.alpha[n]);
          if (hi[d] === N[d] - 1) hi[d] = beta;
          ohi[d] = Math.min(ohi[d], beta);
        }
      }
      L.xlo = lo[0]; L.xhi = hi[0]; L.ylo = lo[1]; L.yhi = hi[1]; L.zlo = lo[2]; L.zhi = hi[2];
      L.oxlo = olo[0]; L.oxhi = ohi[0]; L.oylo = olo[1]; L.oyhi = ohi[1]; L.ozlo = olo[2]; L.ozhi = ohi[2];
      L.sx = ohi[0] - olo[0] + 1; L.sy = ohi[1] - olo[1] + 1; L.sz = ohi[2] - olo[2] + 1;
    }
  }

  /** MSM::allocate: the charge and potential grids of each level (per-atom virial grids are made when needed). */
  private allocate(): void {
    for (const L of this.lv) {
      const n = L.sx * L.sy * L.sz;
      if (L.q.length !== n) { L.q = new Float64Array(n); L.e = new Float64Array(n); L.v = null; }
    }
    const o = this.order;
    for (let d = 0; d < 3; d++) {
      // phi1d[d][-order..order], offset by order
      this.phi1d[d] = new Float64Array(2 * o + 1);
      this.dphi1d[d] = new Float64Array(2 * o + 1);
    }
  }

  private allocatePeratom(): void {
    for (const L of this.lv) if (!L.v) L.v = Array.from({ length: 6 }, () => new Float64Array(L.sx * L.sy * L.sz));
  }

  /** Flat index of grid point (x, y, z) of a level. */
  private static at(L: Level, x: number, y: number, z: number): number {
    return ((z - L.ozlo) * L.sy + (y - L.oylo)) * L.sx + (x - L.oxlo);
  }

  /** Reverse grid communication: every ghost value of a periodic dimension is added to its periodic image (z, y, x). */
  protected fold(L: Level, a: Float64Array): void {
    const { sx, sy } = L;
    const wrap = (i: number, N: number): number => ((i % N) + N) % N;
    if (this.periodic[2]) {
      for (let z = L.ozlo; z <= L.ozhi; z++) {
        if (z >= L.zlo && z <= L.zhi) continue;
        const zw = wrap(z, L.nz);
        const src = (z - L.ozlo) * sy * sx, dst = (zw - L.ozlo) * sy * sx;
        for (let k = 0; k < sy * sx; k++) a[dst + k] += a[src + k];
      }
    }
    if (this.periodic[1]) {
      for (let z = L.zlo; z <= L.zhi; z++) {
        for (let y = L.oylo; y <= L.oyhi; y++) {
          if (y >= L.ylo && y <= L.yhi) continue;
          const yw = wrap(y, L.ny);
          const src = ((z - L.ozlo) * sy + (y - L.oylo)) * sx, dst = ((z - L.ozlo) * sy + (yw - L.oylo)) * sx;
          for (let k = 0; k < sx; k++) a[dst + k] += a[src + k];
        }
      }
    }
    if (this.periodic[0]) {
      for (let z = L.zlo; z <= L.zhi; z++) {
        for (let y = L.ylo; y <= L.yhi; y++) {
          const row = ((z - L.ozlo) * sy + (y - L.oylo)) * sx - L.oxlo;
          for (let x = L.oxlo; x <= L.oxhi; x++) {
            if (x >= L.xlo && x <= L.xhi) continue;
            a[row + wrap(x, L.nx)] += a[row + x];
          }
        }
      }
    }
  }

  /** Forward grid communication: every ghost point of a periodic dimension takes its owned image's value (x, y, z). */
  protected fill(L: Level, a: Float64Array): void {
    const { sx, sy } = L;
    const wrap = (i: number, N: number): number => ((i % N) + N) % N;
    if (this.periodic[0]) {
      for (let z = L.zlo; z <= L.zhi; z++) {
        for (let y = L.ylo; y <= L.yhi; y++) {
          const row = ((z - L.ozlo) * sy + (y - L.oylo)) * sx - L.oxlo;
          for (let x = L.oxlo; x <= L.oxhi; x++) {
            if (x >= L.xlo && x <= L.xhi) continue;
            a[row + x] = a[row + wrap(x, L.nx)];
          }
        }
      }
    }
    if (this.periodic[1]) {
      for (let z = L.zlo; z <= L.zhi; z++) {
        for (let y = L.oylo; y <= L.oyhi; y++) {
          if (y >= L.ylo && y <= L.yhi) continue;
          const yw = wrap(y, L.ny);
          const dst = ((z - L.ozlo) * sy + (y - L.oylo)) * sx, src = ((z - L.ozlo) * sy + (yw - L.oylo)) * sx;
          for (let k = 0; k < sx; k++) a[dst + k] = a[src + k];
        }
      }
    }
    if (this.periodic[2]) {
      for (let z = L.ozlo; z <= L.ozhi; z++) {
        if (z >= L.zlo && z <= L.zhi) continue;
        const zw = wrap(z, L.nz);
        const dst = (z - L.ozlo) * sy * sx, src = (zw - L.ozlo) * sy * sx;
        for (let k = 0; k < sy * sx; k++) a[dst + k] = a[src + k];
      }
    }
  }

  /** The atoms the grid sees: all owned atoms for msm; msm/cg keeps only the charged ones. */
  protected atomList(s: SimState): Int32Array | null { void s; return null; }

  /** MSM::compute. */
  compute(kc: KSpaceCompute): void {
    const s = kc.s, geom = kc.geom;
    this.eflagAtom = kc.eatom !== null;
    this.vflagAtom = kc.vatom !== null;
    this.beforeCompute(s);
    if (this.scalarPressure && this.vflagAtom) {
      throw new StyleError("Must use 'kspace_modify pressure/scalar no' to obtain per-atom virial with kspace_style MSM");
    }
    if (s.n !== this.natomsOriginal) {
      this.qsumQsq(s, this.qqrd2eRun, null);
      this.natomsOriginal = s.n;
    }
    if (this.qsqsumZeroReturns && this.qsqsum === 0.0) return;
    if (KSpaceMSM.key(geom) !== this.setupKey) this.setup(geom, s);
    if (this.vflagAtom) this.allocatePeratom();
    this.energy = 0;
    this.virial.fill(0);
    this.curLo = this.triclinic ? [0, 0, 0] : geom.lo;
    const list = this.atomList(s);
    this.positions(s, geom, list);
    this.particleMap(s, list);
    this.makeRho(s, list);
    const levels = this.levels;
    this.fold(this.lv[0], this.lv[0].q);
    for (let n = 0; n <= levels - 2; n++) {
      this.fill(this.lv[n], this.lv[n].q);
      this.direct(n);
      this.restriction(n);
    }
    const top = this.lv[levels - 1];
    if (this.nonperiodic) {
      this.fill(top, top.q);
      this.directTop(levels - 1);
      this.fold(top, top.e);
    } else {
      // grid_swap_forward / grid_swap_reverse
      this.fill(top, top.q);
      this.direct(levels - 1);
      this.fold(top, top.e);
    }
    if (this.vflagAtom) for (const v of top.v!) this.fold(top, v);
    for (let n = levels - 2; n >= 0; n--) {
      this.prolongation(n);
      this.fold(this.lv[n], this.lv[n].e);
      if (this.vflagAtom) for (const v of this.lv[n].v!) this.fold(this.lv[n], v);
    }
    const L0 = this.lv[0];
    this.fill(L0, L0.e);
    if (this.vflagAtom) for (const v of L0.v!) this.fill(L0, v);
    this.fieldforce(s, kc.f, kc.qqrd2e, list);
    if (this.eflagAtom || this.vflagAtom) this.fieldforcePeratom(s, list, kc.eatom, kc.vatom);
    this.afterFieldforce(s);
    const qscale = kc.qqrd2e;
    const g0 = msmGamma(this.order, 0.0);
    let energy = this.energy;
    const eSelf = (this.qsqsum * g0) / this.cutoff;
    energy -= eSelf;
    energy *= 0.5 * qscale;
    const vir = this.virial;
    if (!this.scalarPressure) for (let i = 0; i < 6; i++) vir[i] = 0.5 * qscale * vir[i];
    else { for (let i = 0; i < 3; i++) vir[i] = energy / 3.0; vir[3] = vir[4] = vir[5] = 0; }
    kc.acc.elong += energy;
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += vir[c];
    if (kc.eatom || kc.vatom) this.peratomFinish(s, list, kc.eatom, kc.vatom, qscale, g0);
    this.restorePositions(s, geom, list);
  }

  /** msm/cg updates its charged list here; plain msm has nothing to do. */
  protected beforeCompute(_s: SimState): void {}
  protected afterFieldforce(_s: SimState): void {}
  /** MSM::compute returns early when no atom is charged (msm/cg does not). */
  protected qsqsumZeroReturns = true;

  /** eatom/vatom of the owned atoms: self energy and the 0.5 qqrd2e factor (end of MSM::compute). */
  protected peratomFinish(s: SimState, list: Int32Array | null, eatom: Float64Array | null, vatom: Float64Array | null, qscale: number, g0: number): void {
    const n = list ? list.length : s.n;
    for (let k = 0; k < n; k++) {
      const i = list ? list[k] : k;
      if (eatom) {
        eatom[i] -= (s.q[i] * s.q[i] * g0) / this.cutoff;
        eatom[i] *= 0.5 * qscale;
      }
      if (vatom) for (let j = 0; j < 6; j++) vatom[6 * i + j] *= 0.5 * qscale;
    }
  }

  /** The grid coordinates of the owned atoms: x itself, or x2lamda (domain.cpp) for a triclinic box. */
  private positions(s: SimState, geom: Geometry, _list: Int32Array | null): void {
    const n = s.n;
    if (this.xg.length < 3 * n) this.xg = new Float64Array(3 * n);
    const x = s.x, xg = this.xg;
    if (!this.triclinic) {
      xg.set(x.subarray(0, 3 * n));
      return;
    }
    const hi = this.hInv, lo = geom.lo;
    for (let i = 0; i < n; i++) {
      const d0 = x[3 * i] - lo[0], d1 = x[3 * i + 1] - lo[1], d2 = x[3 * i + 2] - lo[2];
      xg[3 * i] = hi[0] * d0 + hi[5] * d1 + hi[4] * d2;
      xg[3 * i + 1] = hi[1] * d1 + hi[3] * d2;
      xg[3 * i + 2] = hi[2] * d2;
    }
  }

  /** domain->lamda2x after the triclinic compute: the owned positions go back from their lamda coordinates. */
  private restorePositions(s: SimState, geom: Geometry, _list: Int32Array | null): void {
    if (!this.triclinic) return;
    const h = this.h, lo = geom.lo, x = s.x, xg = this.xg;
    for (let i = 0; i < s.n; i++) {
      const l0 = xg[3 * i], l1 = xg[3 * i + 1], l2 = xg[3 * i + 2];
      x[3 * i] = h[0] * l0 + h[5] * l1 + h[4] * l2 + lo[0];
      x[3 * i + 1] = h[1] * l1 + h[3] * l2 + lo[1];
      x[3 * i + 2] = h[2] * l2 + lo[2];
    }
  }

  /** MSM::particle_map: the grid point to the lower left of each atom; the stencil must fit in the owned+ghost grid. */
  private particleMap(s: SimState, list: Int32Array | null): void {
    const n = list ? list.length : s.n;
    if (this.part2grid.length < 3 * s.n) this.part2grid = new Int32Array(3 * s.n);
    const lo = this.curLo;
    if (!Number.isFinite(lo[0]) || !Number.isFinite(lo[1]) || !Number.isFinite(lo[2])) throw new StyleError('Non-numeric box dimensions - simulation unstable');
    const L = this.lv[0];
    const xg = this.xg, p2g = this.part2grid;
    let flag = false;
    for (let k = 0; k < n; k++) {
      const i = list ? list[k] : k;
      const nx = toInt((xg[3 * i] - lo[0]) * this.delxinv[0] + OFFSET) - OFFSET;
      const ny = toInt((xg[3 * i + 1] - lo[1]) * this.delyinv[0] + OFFSET) - OFFSET;
      const nz = toInt((xg[3 * i + 2] - lo[2]) * this.delzinv[0] + OFFSET) - OFFSET;
      p2g[3 * i] = nx; p2g[3 * i + 1] = ny; p2g[3 * i + 2] = nz;
      if (nx + this.nlower < L.oxlo || nx + this.nupper > L.oxhi || ny + this.nlower < L.oylo || ny + this.nupper > L.oyhi
        || nz + this.nlower < L.ozlo || nz + this.nupper > L.ozhi) flag = true;
    }
    if (flag) throw new StyleError('Out of range atoms - cannot compute MSM');
  }

  /** boxlo of the current compute (domain->boxlo, or boxlo_lamda = 0 for a triclinic box). */
  private curLo: readonly number[] = [0, 0, 0];

  /** MSM::compute_phi: the interpolating polynomial (Parallel Computing 35 (2009) 164-177, eq. 7, and Hardy's thesis). */
  protected computePhi(xi: number): number {
    let phi = 0.0;
    const absXi = Math.abs(xi);
    const xi2 = xi * xi;
    const order = this.order;
    if (order === 4) {
      if (absXi <= 1) phi = (1.0 - absXi) * (1.0 + absXi - 1.5 * xi2);
      else if (absXi <= 2) phi = -0.5 * (absXi - 1.0) * (2.0 - absXi) * (2.0 - absXi);
    } else if (order === 6) {
      if (absXi <= 1) phi = ((1.0 - xi2) * (2.0 - absXi) * (6.0 + 3.0 * absXi - 5.0 * xi2)) / 12.0;
      else if (absXi <= 2) phi = (-(absXi - 1.0) * (2.0 - absXi) * (3.0 - absXi) * (4.0 + 9.0 * absXi - 5.0 * xi2)) / 24.0;
      else if (absXi <= 3) phi = ((absXi - 1.0) * (absXi - 2.0) * (3.0 - absXi) * (3.0 - absXi) * (4.0 - absXi)) / 24.0;
    } else if (order === 8) {
      if (absXi <= 1) phi = ((1.0 - xi2) * (4.0 - xi2) * (3.0 - absXi) * (12.0 + 4.0 * absXi - 7.0 * xi2)) / 144.0;
      else if (absXi <= 2) phi = (-(xi2 - 1.0) * (2.0 - absXi) * (3.0 - absXi) * (4.0 - absXi) * (10.0 + 12.0 * absXi - 7.0 * xi2)) / 240.0;
      else if (absXi <= 3) phi = ((absXi - 1.0) * (absXi - 2.0) * (3.0 - absXi) * (4.0 - absXi) * (5.0 - absXi) * (6.0 + 20.0 * absXi - 7.0 * xi2)) / 720.0;
      else if (absXi <= 4) phi = (-(absXi - 1.0) * (absXi - 2.0) * (absXi - 3.0) * (4.0 - absXi) * (4.0 - absXi) * (5.0 - absXi) * (6.0 - absXi)) / 720.0;
    } else if (order === 10) {
      if (absXi <= 1) phi = ((1.0 - xi2) * (4.0 - xi2) * (9.0 - xi2) * (4.0 - absXi) * (20.0 + 5.0 * absXi - 9.0 * xi2)) / 2880.0;
      else if (absXi <= 2) phi = (-(xi2 - 1.0) * (4.0 - xi2) * (3.0 - absXi) * (4.0 - absXi) * (5.0 - absXi) * (6.0 + 5.0 * absXi - 3.0 * xi2)) / 1440.0;
      else if (absXi <= 3) phi = ((xi2 - 1.0) * (absXi - 2.0) * (3.0 - absXi) * (4.0 - absXi) * (5.0 - absXi) * (6.0 - absXi) * (14.0 + 25.0 * absXi - 9.0 * xi2)) / 10080.0;
      else if (absXi <= 4) phi = (-(absXi - 1.0) * (absXi - 2.0) * (absXi - 3.0) * (4.0 - absXi) * (5.0 - absXi) * (6.0 - absXi) * (7.0 - absXi) * (8.0 + 35.0 * absXi - 9.0 * xi2)) / 40320.0;
      else if (absXi <= 5) phi = ((absXi - 1.0) * (absXi - 2.0) * (absXi - 3.0) * (absXi - 4.0) * (5.0 - absXi) * (5.0 - absXi) * (6.0 - absXi) * (7.0 - absXi) * (8.0 - absXi)) / 40320.0;
    }
    return phi;
  }

  /** MSM::compute_dphi: the derivative of phi. */
  protected computeDphi(xi: number): number {
    let dphi = 0.0;
    const absXi = Math.abs(xi);
    const order = this.order;
    if (absXi === 0.0) return 0.0;
    if (order === 4) {
      const xi2 = xi * xi;
      const absXi2 = absXi * absXi;
      if (absXi <= 1) dphi = (xi * (3 * xi2 + 6 * absXi2 - 10 * absXi)) / 2.0 / absXi;
      else if (absXi <= 2) dphi = (xi * (2 - absXi) * (3 * absXi - 4)) / 2.0 / absXi;
    } else if (order === 6) {
      const xi2 = xi * xi, xi4 = xi2 * xi2;
      const absXi2 = absXi * absXi, absXi3 = absXi2 * absXi, absXi4 = absXi2 * absXi2;
      if (absXi <= 1) {
        dphi = (xi * (46 * xi2 * absXi - 20 * xi2 * absXi2 - 5 * xi4 + 5 * xi2 + 6 * absXi3 + 10 * absXi2 - 50 * absXi)) / 12.0 / absXi;
      } else if (absXi <= 2) {
        dphi = (xi * (15 * xi2 * absXi2 - 60 * xi2 * absXi + 55 * xi2 + 10 * absXi4 - 96 * absXi3 + 260 * absXi2 - 210 * absXi + 10)) / 24.0 / absXi;
      } else if (absXi <= 3) {
        dphi = (-xi * (absXi - 3) * (5 * absXi3 - 37 * absXi2 + 84 * absXi - 58)) / 24.0 / absXi;
      }
    } else if (order === 8) {
      const xi2 = xi * xi, xi4 = xi2 * xi2, xi6 = xi4 * xi2;
      const absXi3 = xi2 * absXi, absXi5 = xi4 * absXi;
      if (absXi <= 1) {
        dphi = (xi * (49 * xi6 - 175 * xi4 + 84 * xi2 - 150 * absXi5 + 644 * absXi3 - 560 * absXi)) / 144.0 / absXi;
      } else if (absXi <= 2) {
        dphi = (xi * (-49 * xi6 - 1365 * xi4 + 756 * xi2 + 450 * absXi5 + 1260 * absXi3 - 1260 * absXi + 28)) / 240.0 / absXi;
      } else if (absXi <= 3) {
        dphi = (xi * (49 * xi6 + 4445 * xi4 + 17724 * xi2 - 750 * absXi5 - 12740 * absXi3 - 9940 * absXi + 756)) / 720.0 / absXi;
      } else if (absXi <= 4) {
        dphi = (-xi * (absXi - 4) * (7 * absXi5 - 122 * xi4 + 807 * absXi3 - 2512 * xi2 + 3644 * absXi - 1944)) / 720.0 / absXi;
      }
    } else if (order === 10) {
      const xi2 = xi * xi, xi4 = xi2 * xi2, xi6 = xi4 * xi2, xi8 = xi6 * xi2;
      const absXi2 = absXi * absXi, absXi3 = absXi2 * absXi, absXi4 = absXi2 * absXi2, absXi5 = absXi4 * absXi;
      const absXi6 = absXi5 * absXi, absXi7 = absXi6 * absXi, absXi8 = absXi7 * absXi;
      if (absXi <= 1) {
        dphi = (xi * (298 * xi6 * absXi - 72 * xi6 * absXi2 - 9 * xi8 + 126 * xi6 + 30 * xi4 * absXi3 + 756 * xi4 * absXi2 - 3644 * xi4 * absXi
          - 441 * xi4 - 280 * xi2 * absXi3 - 1764 * xi2 * absXi2 + 12026 * xi2 * absXi + 324 * xi2 + 490 * absXi3 + 648 * absXi2 - 10792 * absXi)) / 2880.0 / absXi;
      } else if (absXi <= 2) {
        dphi = (xi * (9 * xi6 * absXi2 - 72 * xi6 * absXi + 141 * xi6 + 18 * xi4 * absXi4 - 236 * xi4 * absXi3 + 963 * xi4 * absXi2
          - 1046 * xi4 * absXi - 687 * xi4 - 20 * xi2 * absXi5 + 156 * xi2 * absXi4 + 168 * xi2 * absXi3 - 3522 * xi2 * absXi2
          + 6382 * xi2 * absXi + 474 * xi2 + 50 * absXi5 - 516 * absXi4 + 1262 * absXi3 + 1596 * absXi2 - 6344 * absXi + 72)) / 1440.0 / absXi;
      } else if (absXi <= 3) {
        dphi = (xi * (720 * xi4 * absXi3 - 45 * xi4 * absXi4 - 4185 * xi4 * absXi2 + 10440 * xi4 * absXi - 9396 * xi4 - 36 * xi2 * absXi6
          + 870 * xi2 * absXi5 - 7965 * xi2 * absXi4 + 34540 * xi2 * absXi3 - 70389 * xi2 * absXi2 + 51440 * xi2 * absXi + 6012 * xi2
          + 50 * absXi7 - 954 * absXi6 + 6680 * absXi5 - 19440 * absXi4 + 11140 * absXi3 + 49014 * absXi2 - 69080 * absXi + 3384)) / 10080.0 / absXi;
      } else if (absXi <= 4) {
        dphi = (xi * (63 * xi2 * absXi6 - 1512 * xi2 * absXi5 + 14490 * xi2 * absXi4 - 70560 * xi2 * absXi3 + 182763 * xi2 * absXi2
          - 236376 * xi2 * absXi + 117612 * xi2 + 18 * absXi8 - 784 * absXi7 + 12600 * absXi6 - 101556 * absXi5 + 451962 * absXi4
          - 1121316 * absXi3 + 1451628 * absXi2 - 795368 * absXi + 71856)) / 40320.0 / absXi;
      } else if (absXi <= 5) {
        dphi = (-xi * (absXi - 5) * (9 * absXi7 - 283 * absXi6 + 3667 * absXi5 - 25261 * absXi4 + 99340 * absXi3 - 221416 * absXi2
          + 256552 * absXi - 117648)) / 40320.0 / absXi;
      }
    }
    return dphi;
  }

  /** MSM::compute_phis (and compute_phis_and_dphis when dphi). */
  private computePhis(dx: number, dy: number, dz: number, dphi: boolean): void {
    const o = this.order;
    const [px, py, pz] = this.phi1d;
    const [qx, qy, qz] = this.dphi1d;
    for (let nu = this.nlower; nu <= this.nupper; nu++) {
      const delx = dx + nu, dely = dy + nu, delz = dz + nu;
      px[nu + o] = this.computePhi(delx);
      py[nu + o] = this.computePhi(dely);
      pz[nu + o] = this.computePhi(delz);
      if (dphi) {
        qx[nu + o] = this.computeDphi(delx);
        qy[nu + o] = this.computeDphi(dely);
        qz[nu + o] = this.computeDphi(delz);
      }
    }
  }

  /** MSM::make_rho: the charges interpolated to the finest grid. */
  private makeRho(s: SimState, list: Int32Array | null): void {
    const L = this.lv[0];
    const q0 = L.q;
    q0.fill(0);
    const n = list ? list.length : s.n;
    const lo = this.curLo, xg = this.xg, p2g = this.part2grid, o = this.order;
    const [px, py, pz] = this.phi1d;
    for (let k = 0; k < n; k++) {
      const i = list ? list[k] : k;
      const nx = p2g[3 * i], ny = p2g[3 * i + 1], nz = p2g[3 * i + 2];
      const dx = nx - (xg[3 * i] - lo[0]) * this.delxinv[0];
      const dy = ny - (xg[3 * i + 1] - lo[1]) * this.delyinv[0];
      const dz = nz - (xg[3 * i + 2] - lo[2]) * this.delzinv[0];
      this.computePhis(dx, dy, dz, false);
      const z0 = s.q[i];
      for (let nn = this.nlower; nn <= this.nupper; nn++) {
        const mz = nn + nz;
        const y0 = z0 * pz[nn + o];
        for (let m = this.nlower; m <= this.nupper; m++) {
          const my = m + ny;
          const x0 = y0 * py[m + o];
          const row = ((mz - L.ozlo) * L.sy + (my - L.oylo)) * L.sx - L.oxlo;
          for (let l = this.nlower; l <= this.nupper; l++) q0[row + l + nx] += x0 * px[l + o];
        }
      }
    }
  }

  /** MSM::direct (with direct_peratom): the direct sum of an intermediate level (and the periodic top level). */
  private direct(n: number): void {
    const L = this.lv[n];
    const q = L.q, e = L.e;
    e.fill(0);
    const vgrid = this.vflagAtom ? L.v! : null;
    if (vgrid) for (const v of vgrid) v.fill(0);
    const g = this.gDirect[n];
    const vd = this.vDirect[n];
    const tensor = this.vflagEither && !this.scalarPressure;
    const nxhi = this.nxhiDirect, nyhi = this.nyhiDirect, nzhi = this.nzhiDirect;
    const nxlo = -nxhi, nylo = -nyhi;
    const nx = 2 * nxhi + 1, ny = 2 * nyhi + 1;
    const [xper, yper, zper] = this.periodic;
    const alpha = this.alpha[n], betax = this.betax[n], betay = this.betay[n], betaz = this.betaz[n];
    const sx = L.sx, sy = L.sy;
    const vir = this.virial;
    for (let icz = L.zlo; icz <= L.zhi; icz++) {
      const kmax = zper ? nzhi : Math.min(nzhi, betaz - icz);
      for (let icy = L.ylo; icy <= L.yhi; icy++) {
        const jmin = yper ? nylo : Math.max(nylo, alpha - icy);
        const jmax = yper ? nyhi : Math.min(nyhi, betay - icy);
        for (let icx = L.xlo; icx <= L.xhi; icx++) {
          const imin = xper ? nxlo : Math.max(nxlo, alpha - icx);
          const imax = xper ? nxhi : Math.min(nxhi, betax - icx);
          const c = KSpaceMSM.at(L, icx, icy, icz);
          let qtmp = q[c];
          let esum = 0.0;
          let v0sum = 0.0, v1sum = 0.0, v2sum = 0.0, v3sum = 0.0, v4sum = 0.0, v5sum = 0.0;
          // the hemisphere: no computations in the -z direction
          for (let iz = 1; iz <= kmax; iz++) {
            const kk = icz + iz;
            const zk = (iz + nzhi) * ny;
            for (let iy = jmin; iy <= jmax; iy++) {
              const jj = icy + iy;
              const row = ((kk - L.ozlo) * sy + (jj - L.oylo)) * sx - L.oxlo + icx;
              const zyk = (zk + iy + nyhi) * nx + nxhi;
              for (let ix = imin; ix <= imax; ix++) {
                const qtmp2 = q[row + ix];
                const k = zyk + ix;
                const gtmp = g[k];
                esum += gtmp * qtmp2;
                e[row + ix] += gtmp * qtmp;
                if (tensor) {
                  v0sum += vd[0][k] * qtmp2; v1sum += vd[1][k] * qtmp2; v2sum += vd[2][k] * qtmp2;
                  v3sum += vd[3][k] * qtmp2; v4sum += vd[4][k] * qtmp2; v5sum += vd[5][k] * qtmp2;
                }
              }
            }
          }
          // iz = 0
          {
            const zk = nzhi * ny;
            for (let iy = 1; iy <= jmax; iy++) {
              const jj = icy + iy;
              const row = ((icz - L.ozlo) * sy + (jj - L.oylo)) * sx - L.oxlo + icx;
              const zyk = (zk + iy + nyhi) * nx + nxhi;
              for (let ix = imin; ix <= imax; ix++) {
                const qtmp2 = q[row + ix];
                const k = zyk + ix;
                const gtmp = g[k];
                esum += gtmp * qtmp2;
                e[row + ix] += gtmp * qtmp;
                if (tensor) {
                  v0sum += vd[0][k] * qtmp2; v1sum += vd[1][k] * qtmp2; v2sum += vd[2][k] * qtmp2;
                  v3sum += vd[3][k] * qtmp2; v4sum += vd[4][k] * qtmp2; v5sum += vd[5][k] * qtmp2;
                }
              }
            }
            // iz = 0, iy = 0
            const row = ((icz - L.ozlo) * sy + (icy - L.oylo)) * sx - L.oxlo + icx;
            const zyk = (zk + nyhi) * nx + nxhi;
            for (let ix = 1; ix <= imax; ix++) {
              const qtmp2 = q[row + ix];
              const k = zyk + ix;
              const gtmp = g[k];
              esum += gtmp * qtmp2;
              e[row + ix] += gtmp * qtmp;
              if (tensor) {
                v0sum += vd[0][k] * qtmp2; v1sum += vd[1][k] * qtmp2; v2sum += vd[2][k] * qtmp2;
                v3sum += vd[3][k] * qtmp2; v4sum += vd[4][k] * qtmp2; v5sum += vd[5][k] * qtmp2;
              }
            }
            // iz = 0, iy = 0, ix = 0 (the virial is zero there)
            const gtmp = g[zyk];
            esum += 0.5 * gtmp * qtmp;
            e[c] += 0.5 * gtmp * qtmp;
          }
          e[c] += esum;
          if (vgrid && !this.scalarPressure) {
            vgrid[0][c] += v0sum; vgrid[1][c] += v1sum; vgrid[2][c] += v2sum;
            vgrid[3][c] += v3sum; vgrid[4][c] += v4sum; vgrid[5][c] += v5sum;
          }
          qtmp = q[c];
          this.energy += 2.0 * esum * qtmp;
          if (tensor) {
            vir[0] += 2.0 * v0sum * qtmp; vir[1] += 2.0 * v1sum * qtmp; vir[2] += 2.0 * v2sum * qtmp;
            vir[3] += 2.0 * v3sum * qtmp; vir[4] += 2.0 * v4sum * qtmp; vir[5] += 2.0 * v5sum * qtmp;
          }
        }
      }
    }
    if (vgrid) this.directPeratom(L, vd, n, nxhi, nyhi, nzhi);
  }

  /** MSM::direct_peratom: the partner side of the per-atom virial of the hemisphere. */
  private directPeratom(L: Level, vd: Float64Array[], n: number, nxhi: number, nyhi: number, nzhi: number): void {
    const q = L.q, vg = L.v!;
    const nx = 2 * nxhi + 1, ny = 2 * nyhi + 1;
    const [xper, yper, zper] = this.periodic;
    const alpha = this.alpha[n], betax = this.betax[n], betay = this.betay[n], betaz = this.betaz[n];
    const sx = L.sx, sy = L.sy;
    const add = (row: number, ix: number, k: number, qtmp: number): void => {
      for (let c = 0; c < 6; c++) vg[c][row + ix] += vd[c][k] * qtmp;
    };
    for (let icz = L.zlo; icz <= L.zhi; icz++) {
      const kmax = zper ? nzhi : Math.min(nzhi, betaz - icz);
      for (let icy = L.ylo; icy <= L.yhi; icy++) {
        const jmin = yper ? -nyhi : Math.max(-nyhi, alpha - icy);
        const jmax = yper ? nyhi : Math.min(nyhi, betay - icy);
        for (let icx = L.xlo; icx <= L.xhi; icx++) {
          const imin = xper ? -nxhi : Math.max(-nxhi, alpha - icx);
          const imax = xper ? nxhi : Math.min(nxhi, betax - icx);
          const qtmp = q[KSpaceMSM.at(L, icx, icy, icz)];
          for (let iz = 1; iz <= kmax; iz++) {
            const zk = (iz + nzhi) * ny;
            for (let iy = jmin; iy <= jmax; iy++) {
              const row = ((icz + iz - L.ozlo) * sy + (icy + iy - L.oylo)) * sx - L.oxlo + icx;
              const zyk = (zk + iy + nyhi) * nx + nxhi;
              for (let ix = imin; ix <= imax; ix++) add(row, ix, zyk + ix, qtmp);
            }
          }
          const zk = nzhi * ny;
          for (let iy = 1; iy <= jmax; iy++) {
            const row = ((icz - L.ozlo) * sy + (icy + iy - L.oylo)) * sx - L.oxlo + icx;
            const zyk = (zk + iy + nyhi) * nx + nxhi;
            for (let ix = imin; ix <= imax; ix++) add(row, ix, zyk + ix, qtmp);
          }
          const row = ((icz - L.ozlo) * sy + (icy - L.oylo)) * sx - L.oxlo + icx;
          const zyk = (zk + nyhi) * nx + nxhi;
          for (let ix = 1; ix <= imax; ix++) add(row, ix, zyk + ix, qtmp);
        }
      }
    }
  }

  /** MSM::direct_top (with direct_peratom_top): the full direct sum of the top level of a non-periodic system. */
  private directTop(n: number): void {
    const L = this.lv[n];
    const q = L.q, e = L.e;
    e.fill(0);
    const vgrid = this.vflagAtom ? L.v! : null;
    if (vgrid) for (const v of vgrid) v.fill(0);
    const g = this.gDirectTop, vd = this.vDirectTop;
    const tensor = this.vflagEither && !this.scalarPressure;
    const alpha = this.alpha[n], betax = this.betax[n], betay = this.betay[n], betaz = this.betaz[n];
    const nxTop = betax - alpha, nyTop = betay - alpha, nzTop = betaz - alpha;
    const nx = 2 * nxTop + 1, ny = 2 * nyTop + 1;
    const [xper, yper, zper] = this.periodic;
    const sx = L.sx, sy = L.sy;
    const vir = this.virial;
    for (let icz = L.zlo; icz <= L.zhi; icz++) {
      const kmax = zper ? L.nz - 1 : betaz - icz;
      for (let icy = L.ylo; icy <= L.yhi; icy++) {
        const jmin = yper ? 0 : alpha - icy;
        const jmax = yper ? L.ny - 1 : betay - icy;
        for (let icx = L.xlo; icx <= L.xhi; icx++) {
          const imin = xper ? 0 : alpha - icx;
          const imax = xper ? L.nx - 1 : betax - icx;
          const c = KSpaceMSM.at(L, icx, icy, icz);
          let qtmp = q[c];
          let esum = 0.0;
          let v0sum = 0.0, v1sum = 0.0, v2sum = 0.0, v3sum = 0.0, v4sum = 0.0, v5sum = 0.0;
          for (let iz = 1; iz <= kmax; iz++) {
            const kk = icz + iz;
            const zk = (iz + nzTop) * ny;
            for (let iy = jmin; iy <= jmax; iy++) {
              const jj = icy + iy;
              const row = ((kk - L.ozlo) * sy + (jj - L.oylo)) * sx - L.oxlo + icx;
              const zyk = (zk + iy + nyTop) * nx + nxTop;
              for (let ix = imin; ix <= imax; ix++) {
                const qtmp2 = q[row + ix];
                const k = zyk + ix;
                const gtmp = g[k];
                esum += gtmp * qtmp2;
                e[row + ix] += gtmp * qtmp;
                if (tensor) {
                  v0sum += vd[0][k] * qtmp2; v1sum += vd[1][k] * qtmp2; v2sum += vd[2][k] * qtmp2;
                  v3sum += vd[3][k] * qtmp2; v4sum += vd[4][k] * qtmp2; v5sum += vd[5][k] * qtmp2;
                }
              }
            }
          }
          const zk = nzTop * ny;
          for (let iy = 1; iy <= jmax; iy++) {
            const jj = icy + iy;
            const row = ((icz - L.ozlo) * sy + (jj - L.oylo)) * sx - L.oxlo + icx;
            const zyk = (zk + iy + nyTop) * nx + nxTop;
            for (let ix = imin; ix <= imax; ix++) {
              const qtmp2 = q[row + ix];
              const k = zyk + ix;
              const gtmp = g[k];
              esum += gtmp * qtmp2;
              e[row + ix] += gtmp * qtmp;
              if (tensor) {
                v0sum += vd[0][k] * qtmp2; v1sum += vd[1][k] * qtmp2; v2sum += vd[2][k] * qtmp2;
                v3sum += vd[3][k] * qtmp2; v4sum += vd[4][k] * qtmp2; v5sum += vd[5][k] * qtmp2;
              }
            }
          }
          const row = ((icz - L.ozlo) * sy + (icy - L.oylo)) * sx - L.oxlo + icx;
          const zyk = (zk + nyTop) * nx + nxTop;
          for (let ix = 1; ix <= imax; ix++) {
            const qtmp2 = q[row + ix];
            const k = zyk + ix;
            const gtmp = g[k];
            esum += gtmp * qtmp2;
            e[row + ix] += gtmp * qtmp;
            if (tensor) {
              v0sum += vd[0][k] * qtmp2; v1sum += vd[1][k] * qtmp2; v2sum += vd[2][k] * qtmp2;
              v3sum += vd[3][k] * qtmp2; v4sum += vd[4][k] * qtmp2; v5sum += vd[5][k] * qtmp2;
            }
          }
          const gtmp = g[zyk];
          esum += 0.5 * gtmp * qtmp;
          e[c] += 0.5 * gtmp * qtmp;
          if (tensor) {
            v0sum += vd[0][zyk] * qtmp; v1sum += vd[1][zyk] * qtmp; v2sum += vd[2][zyk] * qtmp;
            v3sum += vd[3][zyk] * qtmp; v4sum += vd[4][zyk] * qtmp; v5sum += vd[5][zyk] * qtmp;
          }
          e[c] += esum;
          if (vgrid && !this.scalarPressure) {
            vgrid[0][c] += v0sum; vgrid[1][c] += v1sum; vgrid[2][c] += v2sum;
            vgrid[3][c] += v3sum; vgrid[4][c] += v4sum; vgrid[5][c] += v5sum;
          }
          qtmp = q[c];
          this.energy += 2.0 * esum * qtmp;
          if (tensor) {
            vir[0] += 2.0 * v0sum * qtmp; vir[1] += 2.0 * v1sum * qtmp; vir[2] += 2.0 * v2sum * qtmp;
            vir[3] += 2.0 * v3sum * qtmp; vir[4] += 2.0 * v4sum * qtmp; vir[5] += 2.0 * v5sum * qtmp;
          }
        }
      }
    }
    if (vgrid) {
      // direct_peratom_top
      const vg = vgrid;
      for (let icz = L.zlo; icz <= L.zhi; icz++) {
        const kmax = zper ? L.nz - 1 : betaz - icz;
        for (let icy = L.ylo; icy <= L.yhi; icy++) {
          const jmin = yper ? 0 : alpha - icy;
          const jmax = yper ? L.ny - 1 : betay - icy;
          for (let icx = L.xlo; icx <= L.xhi; icx++) {
            const imin = xper ? 0 : alpha - icx;
            const imax = xper ? L.nx - 1 : betax - icx;
            const qtmp = q[KSpaceMSM.at(L, icx, icy, icz)];
            const add = (row: number, ix: number, k: number): void => {
              for (let cc = 0; cc < 6; cc++) vg[cc][row + ix] += vd[cc][k] * qtmp;
            };
            for (let iz = 1; iz <= kmax; iz++) {
              const zk = (iz + nzTop) * ny;
              for (let iy = jmin; iy <= jmax; iy++) {
                const row = ((icz + iz - L.ozlo) * sy + (icy + iy - L.oylo)) * sx - L.oxlo + icx;
                const zyk = (zk + iy + nyTop) * nx + nxTop;
                for (let ix = imin; ix <= imax; ix++) add(row, ix, zyk + ix);
              }
            }
            const zk = nzTop * ny;
            for (let iy = 1; iy <= jmax; iy++) {
              const row = ((icz - L.ozlo) * sy + (icy + iy - L.oylo)) * sx - L.oxlo + icx;
              const zyk = (zk + iy + nyTop) * nx + nxTop;
              for (let ix = imin; ix <= imax; ix++) add(row, ix, zyk + ix);
            }
            const row = ((icz - L.ozlo) * sy + (icy - L.oylo)) * sx - L.oxlo + icx;
            const zyk = (zk + nyTop) * nx + nxTop;
            for (let ix = 1; ix <= imax; ix++) add(row, ix, zyk + ix);
          }
        }
      }
    }
  }

  /** The restriction/prolongation stencil of level n: phi at nu * h_n/h_{n+1} for nu = -p..p, odd or 0. */
  private stencil(n: number): { index: number[]; px: number[]; py: number[]; pz: number[] } {
    const p = this.order - 1;
    const index: number[] = [], px: number[] = [], py: number[] = [], pz: number[] = [];
    for (let nu = -p; nu <= p; nu++) {
      if (nu % 2 === 0 && nu !== 0) continue;
      px.push(this.computePhi((nu * this.delxinv[n + 1]) / this.delxinv[n]));
      py.push(this.computePhi((nu * this.delyinv[n + 1]) / this.delyinv[n]));
      pz.push(this.computePhi((nu * this.delzinv[n + 1]) / this.delzinv[n]));
      index.push(nu);
    }
    return { index, px, py, pz };
  }

  /** MSM::restriction: the charges of level n interpolated to level n+1. */
  private restriction(n: number): void {
    const p = this.order - 1;
    const L1 = this.lv[n], L2 = this.lv[n + 1];
    const q1 = L1.q, q2 = L2.q;
    const { index, px, py, pz } = this.stencil(n);
    q2.fill(0);
    const [xper, yper, zper] = this.periodic;
    const alpha = this.alpha[n], betax = this.betax[n], betay = this.betay[n], betaz = this.betaz[n];
    const rx = toInt(this.delxinv[n] / this.delxinv[n + 1]);
    const ry = toInt(this.delyinv[n] / this.delyinv[n + 1]);
    const rz = toInt(this.delzinv[n] / this.delzinv[n + 1]);
    for (let kp = L2.zlo; kp <= L2.zhi; kp++) {
      for (let jp = L2.ylo; jp <= L2.yhi; jp++) {
        for (let ip = L2.xlo; ip <= L2.xhi; ip++) {
          const ic = ip * rx, jc = jp * ry, kc = kp * rz;
          let q2sum = 0.0;
          for (let k = 0; k <= p + 1; k++) {
            const kk = kc + index[k];
            if (!zper) {
              if (kk < alpha) continue;
              if (kk > betaz) break;
            }
            const phiz = pz[k];
            for (let j = 0; j <= p + 1; j++) {
              const jj = jc + index[j];
              if (!yper) {
                if (jj < alpha) continue;
                if (jj > betay) break;
              }
              const phizy = py[j] * phiz;
              const row = ((kk - L1.ozlo) * L1.sy + (jj - L1.oylo)) * L1.sx - L1.oxlo;
              for (let i = 0; i <= p + 1; i++) {
                const ii = ic + index[i];
                if (!xper) {
                  if (ii < alpha) continue;
                  if (ii > betax) break;
                }
                q2sum += q1[row + ii] * px[i] * phizy;
              }
            }
          }
          q2[KSpaceMSM.at(L2, ip, jp, kp)] += q2sum;
        }
      }
    }
  }

  /** MSM::prolongation: the potentials (and per-atom virials) of level n+1 interpolated back to level n. */
  private prolongation(n: number): void {
    const p = this.order - 1;
    const L1 = this.lv[n], L2 = this.lv[n + 1];
    const e1 = L1.e, e2 = L2.e;
    const v1 = this.vflagAtom ? L1.v! : null, v2 = this.vflagAtom ? L2.v! : null;
    const { index, px, py, pz } = this.stencil(n);
    const [xper, yper, zper] = this.periodic;
    const alpha = this.alpha[n], betax = this.betax[n], betay = this.betay[n], betaz = this.betaz[n];
    const rx = toInt(this.delxinv[n] / this.delxinv[n + 1]);
    const ry = toInt(this.delyinv[n] / this.delyinv[n + 1]);
    const rz = toInt(this.delzinv[n] / this.delzinv[n + 1]);
    const vt = [0, 0, 0, 0, 0, 0];
    for (let kp = L2.zlo; kp <= L2.zhi; kp++) {
      for (let jp = L2.ylo; jp <= L2.yhi; jp++) {
        for (let ip = L2.xlo; ip <= L2.xhi; ip++) {
          const ic = ip * rx, jc = jp * ry, kc = kp * rz;
          const c2 = KSpaceMSM.at(L2, ip, jp, kp);
          const etmp2 = e2[c2];
          if (v2) for (let c = 0; c < 6; c++) vt[c] = v2[c][c2];
          for (let k = 0; k <= p + 1; k++) {
            const kk = kc + index[k];
            if (!zper) {
              if (kk < alpha) continue;
              if (kk > betaz) break;
            }
            const phiz = pz[k];
            for (let j = 0; j <= p + 1; j++) {
              const jj = jc + index[j];
              if (!yper) {
                if (jj < alpha) continue;
                if (jj > betay) break;
              }
              const phizy = py[j] * phiz;
              const row = ((kk - L1.ozlo) * L1.sy + (jj - L1.oylo)) * L1.sx - L1.oxlo;
              for (let i = 0; i <= p + 1; i++) {
                const ii = ic + index[i];
                if (!xper) {
                  if (ii < alpha) continue;
                  if (ii > betax) break;
                }
                const phi3d = px[i] * phizy;
                e1[row + ii] += etmp2 * phi3d;
                if (v1) for (let c = 0; c < 6; c++) v1[c][row + ii] += vt[c] * phi3d;
              }
            }
          }
        }
      }
    }
  }

  /** MSM::fieldforce: the field at each atom from the finest-grid potential, times qqrd2e q_i. */
  private fieldforce(s: SimState, f: Float64Array, qqrd2e: number, list: Int32Array | null): void {
    const L = this.lv[0];
    const e = L.e;
    const n = list ? list.length : s.n;
    const lo = this.curLo, xg = this.xg, p2g = this.part2grid, o = this.order;
    const [px, py, pz] = this.phi1d;
    const [qx, qy, qz] = this.dphi1d;
    for (let k = 0; k < n; k++) {
      const i = list ? list[k] : k;
      const nx = p2g[3 * i], ny = p2g[3 * i + 1], nz = p2g[3 * i + 2];
      const dx = nx - (xg[3 * i] - lo[0]) * this.delxinv[0];
      const dy = ny - (xg[3 * i + 1] - lo[1]) * this.delyinv[0];
      const dz = nz - (xg[3 * i + 2] - lo[2]) * this.delzinv[0];
      this.computePhis(dx, dy, dz, true);
      let ekx = 0.0, eky = 0.0, ekz = 0.0;
      for (let nn = this.nlower; nn <= this.nupper; nn++) {
        const mz = nn + nz;
        const phiZ = pz[nn + o], dphiZ = qz[nn + o];
        for (let m = this.nlower; m <= this.nupper; m++) {
          const my = m + ny;
          const phiY = py[m + o], dphiY = qy[m + o];
          const row = ((mz - L.ozlo) * L.sy + (my - L.oylo)) * L.sx - L.oxlo + nx;
          for (let l = this.nlower; l <= this.nupper; l++) {
            const phiX = px[l + o], dphiX = qx[l + o];
            const etmp = e[row + l];
            ekx += dphiX * phiY * phiZ * etmp;
            eky += phiX * dphiY * phiZ * etmp;
            ekz += phiX * phiY * dphiZ * etmp;
          }
        }
      }
      ekx *= this.delxinv[0];
      eky *= this.delyinv[0];
      ekz *= this.delzinv[0];
      if (this.triclinic) {
        // x2lamdaT
        const hi = this.hInv;
        const t0 = hi[0] * ekx;
        const t1 = hi[5] * ekx + hi[1] * eky;
        const t2 = hi[4] * ekx + hi[3] * eky + hi[2] * ekz;
        ekx = t0; eky = t1; ekz = t2;
      }
      const qfactor = qqrd2e * s.q[i];
      f[3 * i] += qfactor * ekx;
      f[3 * i + 1] += qfactor * eky;
      f[3 * i + 2] += qfactor * ekz;
    }
  }

  /** MSM::fieldforce_peratom: the potential and per-atom virial at each atom (times q_i). */
  private fieldforcePeratom(s: SimState, list: Int32Array | null, eatom: Float64Array | null, vatom: Float64Array | null): void {
    const L = this.lv[0];
    const e = L.e, vg = L.v;
    const n = list ? list.length : s.n;
    const lo = this.curLo, xg = this.xg, p2g = this.part2grid, o = this.order;
    const [px, py, pz] = this.phi1d;
    for (let k = 0; k < n; k++) {
      const i = list ? list[k] : k;
      const nx = p2g[3 * i], ny = p2g[3 * i + 1], nz = p2g[3 * i + 2];
      const dx = nx - (xg[3 * i] - lo[0]) * this.delxinv[0];
      const dy = ny - (xg[3 * i + 1] - lo[1]) * this.delyinv[0];
      const dz = nz - (xg[3 * i + 2] - lo[2]) * this.delzinv[0];
      this.computePhis(dx, dy, dz, true);
      let u = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;
      for (let nn = this.nlower; nn <= this.nupper; nn++) {
        const mz = nn + nz;
        const z0 = pz[nn + o];
        for (let m = this.nlower; m <= this.nupper; m++) {
          const my = m + ny;
          const y0 = z0 * py[m + o];
          const row = ((mz - L.ozlo) * L.sy + (my - L.oylo)) * L.sx - L.oxlo + nx;
          for (let l = this.nlower; l <= this.nupper; l++) {
            const x0 = y0 * px[l + o];
            const c = row + l;
            if (eatom) u += x0 * e[c];
            if (vatom && vg) {
              v0 += x0 * vg[0][c]; v1 += x0 * vg[1][c]; v2 += x0 * vg[2][c];
              v3 += x0 * vg[3][c]; v4 += x0 * vg[4][c]; v5 += x0 * vg[5][c];
            }
          }
        }
      }
      const qi = s.q[i];
      if (eatom) eatom[i] += qi * u;
      if (vatom) {
        vatom[6 * i] += qi * v0; vatom[6 * i + 1] += qi * v1; vatom[6 * i + 2] += qi * v2;
        vatom[6 * i + 3] += qi * v3; vatom[6 * i + 4] += qi * v4; vatom[6 * i + 5] += qi * v5;
      }
    }
  }
}

/**
 * kspace_style msm/cg accuracy (smallq): MSM over the atoms with |q| > smallq only (msm_cg.cpp). The charged list is
 * rebuilt at each compute (see beforeCompute); triclinic boxes and pressure/scalar yes are refused as
 * natively ("Must use 'kspace_modify pressure/scalar no' with kspace_style msm/cg").
 */
export class KSpaceMSMCG extends KSpaceMSM {
  override readonly name: string = 'msm/cg';
  smallq = SMALLQ;
  private isCharged = new Int32Array(0);

  constructor() {
    super();
    this.triclinicSupport = false;
    this.qsqsumZeroReturns = false;
  }

  override settings(args: string[]): void {
    if (args.length < 1 || args.length > 2) throw new StyleError(`Illegal kspace_style ${this.name} command`);
    super.settings(args.slice(0, 1));
    this.smallq = args.length === 2 ? Math.abs(numeric(args[1], 'smallq')) : SMALLQ;
  }

  /**
   * The charged atoms, in index order. Native msm/cg rebuilds the list only when the neighbor list was rebuilt
   * (neighbor->ago == 0); atoms keep their indices between rebuilds, so rebuilding at every compute gives the same list
   * unless a fix changes charges between two neighbor list builds.
   */
  protected override beforeCompute(s: SimState): void {
    if (this.scalarPressure) throw new StyleError("Must use 'kspace_modify pressure/scalar no' with kspace_style msm/cg");
    let m = 0;
    if (this.isCharged.length < s.n) this.isCharged = new Int32Array(s.n);
    for (let i = 0; i < s.n; i++) if (Math.abs(s.q[i]) > this.smallq) this.isCharged[m++] = i;
    this.charged = this.isCharged.subarray(0, m);
  }

  private charged = new Int32Array(0);

  protected override atomList(_s: SimState): Int32Array | null {
    return this.charged;
  }
}
