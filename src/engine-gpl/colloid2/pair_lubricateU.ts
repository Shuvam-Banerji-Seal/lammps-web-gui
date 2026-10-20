/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/COLLOID/pair_lubricateU.cpp and
 * pair_lubricateU.h (Amit Kumar and Michael Bybee (UIUC)). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lubricateU (docs.lammps.org/pair_lubricateU.html): "Styles lubricateU and lubricateU/poly compute velocities
 * and angular velocities for finite-size spherical particles such that the hydrodynamic interaction balances the force
 * and torque due to all other types of interactions." Monodisperse spheres. The velocities and angular velocities of the
 * atoms are set by this style (the integrator must not reset them: fix nve/noforce), and the net force and torque are
 * zero after it. The solver is a conjugate gradient over the full 6N system with a midpoint (two stage) time scheme.
 */
import { Pair, StyleError, mixDistance, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, type NeighList } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { FixWallBase } from '../../engine/fix/walls';
import type { Fix } from '../../engine/fix/fix';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { log } from '../libm/log';
import { pow } from '../libm/pow';

/** force->vxmu2f per unit style (native update.cpp at b332f28: lj 1.0, real 1.4393264316e4, metal 0.6241509647, ...). */
export const VXMU2F: Record<string, number> = {
  lj: 1.0, real: 1.4393264316e4, metal: 0.6241509647, si: 1.0, cgs: 1.0, electron: 3.39893149e1, micro: 1.0, nano: 1.0,
};

/** Fix styles that native "^wall" can match; the engine has no list of fixes, so the candidates are looked up by style. */
export const WALL_STYLES = [
  'wall/lj93', 'wall/lj126', 'wall/lj1043', 'wall/harmonic', 'wall/harmonic/outside', 'wall/morse', 'wall/colloid',
  'wall/srd', 'wall/reflect', 'wall/reflect/stochastic', 'wall/region', 'wall/gran', 'wall/gran/region', 'wall/ees',
  'wall/region/ees', 'wall/lepton', 'wall/body/polygon', 'wall/body/polyhedron',
];

/** The parts of a FixWallBase (src/engine/fix/walls.ts) that the lubricate styles read (native FixWall fields). */
export interface WallView {
  readonly walls: { readonly dim: 0 | 1 | 2; readonly lo: boolean; readonly coord: { readonly edge: boolean; readonly v?: unknown } }[];
  coords(): Float64Array;
}

export const intArg = (w: string | undefined, what: string): number => {
  const v = parseNum(w, what);
  if (!Number.isInteger(v)) throw new StyleError(`expected an integer for ${what}, got '${w}'`);
  return v;
};

/** Native Pair::ev_tally_xyz (pair.cpp at b332f28) for the virial and per-atom virial: the arguments are native's. */
export const evTallyXYZ = (
  pc: PairCompute, i: number, j: number, nlocal: number, newton: boolean,
  fx: number, fy: number, fz: number, delx: number, dely: number, delz: number,
): void => {
  const v0 = delx * fx, v1 = dely * fy, v2 = delz * fz, v3 = delx * fy, v4 = delx * fz, v5 = dely * fz;
  const vir = pc.acc.virial;
  if (newton) {
    vir[0] += v0; vir[1] += v1; vir[2] += v2; vir[3] += v3; vir[4] += v4; vir[5] += v5;
  } else {
    if (i < nlocal) { vir[0] += 0.5 * v0; vir[1] += 0.5 * v1; vir[2] += 0.5 * v2; vir[3] += 0.5 * v3; vir[4] += 0.5 * v4; vir[5] += 0.5 * v5; }
    if (j < nlocal) { vir[0] += 0.5 * v0; vir[1] += 0.5 * v1; vir[2] += 0.5 * v2; vir[3] += 0.5 * v3; vir[4] += 0.5 * v4; vir[5] += 0.5 * v5; }
  }
  const va = pc.vatom;
  if (va) {
    if (newton || i < nlocal) {
      va[6 * i] += 0.5 * v0; va[6 * i + 1] += 0.5 * v1; va[6 * i + 2] += 0.5 * v2;
      va[6 * i + 3] += 0.5 * v3; va[6 * i + 4] += 0.5 * v4; va[6 * i + 5] += 0.5 * v5;
    }
    if (newton || j < nlocal) {
      va[6 * j] += 0.5 * v0; va[6 * j + 1] += 0.5 * v1; va[6 * j + 2] += 0.5 * v2;
      va[6 * j + 3] += 0.5 * v3; va[6 * j + 4] += 0.5 * v4; va[6 * j + 5] += 0.5 * v5;
    }
  }
};

/** The wall fixes of the system that native "^wall" would find, in the engine's fix list order. */
export const findWallFixes = (ctx: StyleContext): Fix[] => {
  const out: Fix[] = [];
  for (const name of WALL_STYLES) {
    const f = ctx.fixOf?.(name);
    if (f) out.push(f);
  }
  return out;
};

export class PairLubricateU extends Pair {
  readonly name: string = 'lubricateU';
  mu = 0;
  flaglog = 0;
  flagHI = 1;
  flagVF = 1;
  gdot = 0;
  cutInnerGlobal = 0;
  cutGlobal = 0;
  /** Monodisperse radius (native rad). */
  protected rad = 0;
  protected flagdeform = 0;
  protected flagwall = 0;
  protected wallFix: (Fix & WallView) | null = null;
  protected volP = 0;
  protected R0 = 0;
  protected RT0 = 0;
  protected RS0 = 0;
  /** Rate of strain tensor, row-major 3x3 (native Ef). */
  protected readonly Ef = new Float64Array(9);
  /** Conjugate gradient tolerance (native TOL). */
  protected tol = 1e-4;
  /** native "if (update->setupflag) return;" in compute(): the monodisperse style skips setup. */
  protected skipAtSetup = true;
  /** native evflag is taken as always set: compute_Fh runs at every force evaluation (see the report). */
  protected readonly evflag = true;
  /** Per type pair coefficients, row-major (ntypes+1)^2 (native cut, cut_inner, setflag). */
  protected cutCoeff = new Float64Array(0);
  protected cutInnerCoeff = new Float64Array(0);
  protected setflag = new Uint8Array(0);

  /* per-evaluation state (set by compute) */
  protected pc!: PairCompute;
  protected nlocal = 0;
  protected nall = 0;
  protected inum = 0;
  protected newton = true;
  protected vxmu2f = 1;
  protected radAll = new Float64Array(0);
  protected V = new Float64Array(0);
  protected W = new Float64Array(0);
  protected F = new Float64Array(0);
  protected T = new Float64Array(0);
  protected bcg = new Float64Array(0);
  protected xcg = new Float64Array(0);
  protected rcg = new Float64Array(0);
  protected rcg1 = new Float64Array(0);
  protected pcg = new Float64Array(0);
  protected RU = new Float64Array(0);

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.cutCoeff = new Float64Array(nt * nt);
    this.cutInnerCoeff = new Float64Array(nt * nt);
    this.setflag = new Uint8Array(nt * nt);
  }

  /** The neighbor list the style walks (native lubricateU: a half list with newton on). */
  protected list(): NeighList {
    const l = this.pc.half;
    if (!l) throw new StyleError(`pair ${this.name} needs a half neighbor list`);
    return l;
  }

  /** Pair order of the native half/newton list (NativePairOrder) for the owned atoms; poly walks its full list instead. */
  private readonly pairOrder = new NativePairOrder(true);

  /**
   * Neighbors of each owned atom in native's order, as a CSR (start[i]..start[i+1]-1 index j). Native lubricateU keeps one
   * copy of each pair (half list, newton on, npair_bin.cpp): which copy decides the geometry, because a ghost partner's
   * intermediate position uses its owner's velocity before the fluid term. nativePairs reproduces that choice; the
   * engine's list supplies the membership and the cutoff test is done by the caller.
   */
  private csr: { start: Int32Array; j: Int32Array } | null = null;
  private csrBuild = -1;

  /** The CSR of the current neighbor build: native fixes the copy of each pair when the list is built (see buildCsr). */
  protected neighborCsr(pc: PairCompute = this.pc): { start: Int32Array; j: Int32Array } {
    if (this.csr && this.csrBuild === pc.nb.nbuild) return this.csr;
    this.csr = this.buildCsr(pc);
    this.csrBuild = pc.nb.nbuild;
    return this.csr;
  }

  protected buildCsr(pc: PairCompute): { start: Int32Array; j: Int32Array } {
    const l = this.useHalfOrientation() ? pc.half : pc.full;
    if (!l) throw new StyleError(`pair ${this.name} needs a ${this.useHalfOrientation() ? 'half' : 'full'} neighbor list`);
    const inum = l.inum;
    let flat: Int32Array;
    if (this.useHalfOrientation()) {
      flat = this.pairOrder.pairs(pc.s, pc.nb, pc.geom, l);
    } else {
      const out: number[] = [];
      for (let i = 0; i < inum; i++) for (let k = l.firstneigh[i], k1 = k + l.numneigh[i]; k < k1; k++) out.push(i, l.neighbors[k]);
      flat = Int32Array.from(out);
    }
    const start = new Int32Array(inum + 1);
    const n = flat.length / 2;
    for (let p = 0; p < n; p++) start[flat[2 * p] + 1]++;
    for (let i = 0; i < inum; i++) start[i + 1] += start[i];
    const fill = Int32Array.from(start);
    const j = new Int32Array(n);
    for (let p = 0; p < n; p++) {
      const i = flat[2 * p];
      j[fill[i]++] = flat[2 * p + 1] & NEIGHMASK;
    }
    return { start, j };
  }

  /** Native half/newton selection: true for the half list (the poly class walks its full list, every pair). */
  protected useHalfOrientation(): boolean { return true; }

  settings(args: string[], ctx: StyleContext): void {
    if (args.length !== 5 && args.length !== 7) throw new StyleError('Illegal pair_style command');
    this.mu = parseNum(args[0], 'mu');
    this.flaglog = intArg(args[1], 'flaglog');
    this.cutInnerGlobal = parseNum(args[2], 'cutinner');
    this.cutGlobal = parseNum(args[3], 'cutoff');
    this.gdot = parseNum(args[4], 'gdot');
    this.flagHI = 1;
    this.flagVF = 1;
    if (args.length === 7) {
      this.flagHI = intArg(args[5], 'flagHI');
      this.flagVF = intArg(args[6], 'flagVF');
    }
    if (this.flaglog === 1 && this.flagHI === 0) {
      ctx.log('WARNING: Cannot include log terms without 1/r terms; setting flagHI to 1.');
      this.flagHI = 1;
    }
    // reset cutoffs that have been explicitly set
    const nt = this.ntypes + 1;
    if (this.setflag.length === nt * nt) {
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) {
          if (this.setflag[i * nt + j]) {
            this.cutInnerCoeff[i * nt + j] = this.cutInnerGlobal;
            this.cutCoeff[i * nt + j] = this.cutGlobal;
          }
        }
      }
    }
    // store the rate of strain tensor (shear flow: Ef[0][1] = Ef[1][0] = gdot/2)
    this.Ef.fill(0);
    this.Ef[1] = 0.5 * this.gdot;
    this.Ef[3] = 0.5 * this.gdot;
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length !== 2 && args.length !== 4) throw new StyleError('Incorrect args for pair coefficients');
    if (this.ntypes < 1) throw new StyleError('pair coefficients need the number of atom types (create_box or read_data first)');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    let cutInnerOne = this.cutInnerGlobal;
    let cutOne = this.cutGlobal;
    if (args.length === 4) {
      cutInnerOne = parseNum(args[2], 'cutinner');
      cutOne = parseNum(args[3], 'cutoff');
    }
    const nt = this.ntypes + 1;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.cutInnerCoeff[i * nt + j] = cutInnerOne;
        this.cutCoeff[i * nt + j] = cutOne;
        this.setflag[i * nt + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** Native init_one: mixed cutoffs from the diagonal pairs (mix_distance), the cutoff is returned. */
  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const ij = i * nt + j;
    if (!this.setflag[ij]) {
      this.cutInnerCoeff[ij] = mixDistance(this.mix, this.cutInnerCoeff[i * nt + i], this.cutInnerCoeff[j * nt + j]);
      this.cutCoeff[ij] = mixDistance(this.mix, this.cutCoeff[i * nt + i], this.cutCoeff[j * nt + j]);
    }
    this.cutInnerCoeff[j * nt + i] = this.cutInnerCoeff[ij];
    return this.cutCoeff[ij];
  }

  /** Lengths of the box in x, y and z (native domain->prd). */
  protected prdOf(lo: readonly number[], hi: readonly number[]): number[] {
    return [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  }

  /**
   * The box lengths with each wall's face moved to its position (native init_style and compute_*: walllo/wallhi per
   * dimension, starting from walllo = 0 and wallhi = prd): the dimensions a wall does not set keep prd.
   */
  protected wallDims(prd: number[]): number[] {
    const wf = this.wallFix!;
    const hi = [prd[0], prd[1], prd[2]];
    const lo = [0, 0, 0];
    const c = wf.coords();
    for (let m = 0; m < wf.walls.length; m++) {
      const w = wf.walls[m];
      if (w.lo) lo[w.dim] = c[m];
      else hi[w.dim] = c[m];
    }
    return [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  }

  /** The total volume used by the volume fraction corrections in compute (null when native skips the block). */
  protected volTCompute(pc: PairCompute): number | null {
    if (!this.flagVF) return null;
    if (!(this.flagdeform || this.flagwall === 2)) return null;
    const g = pc.geom;
    const prd = this.prdOf(g.lo, g.hi);
    const dims = this.flagdeform && !this.flagwall ? prd : this.wallDims(prd);
    return dims[0] * dims[1] * dims[2];
  }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) throw new StyleError(`pair ${this.name} needs the atoms (run setup first)`);
    if (!s.omega) throw new StyleError(`Pair ${this.name} requires atom attribute omega`);
    if (!s.radius) throw new StyleError(`Pair ${this.name} requires atom attribute radius`);
    if (!ctx.ghostVelocity) throw new StyleError(`Pair ${this.name} requires ghost atoms store velocity`);
    if (ctx.newtonPair === false) throw new StyleError(`pair ${this.name} with newton pair off is not supported by the engine`);
    // native: atom->radius_consistency(i, radtype) per type (value -1 for a type with no atoms)
    let radtype = 0;
    let rad = 0;
    for (let i = 1; i <= this.ntypes; i++) {
      let value = -1.0;
      let flag = false;
      for (let k = 0; k < s.n; k++) {
        if (s.type[k] !== i) continue;
        if (value < 0.0) value = s.radius[k];
        else if (value !== s.radius[k]) flag = true;
      }
      if (flag) throw new StyleError(`Pair ${this.name} requires monodisperse particles`);
      radtype = value;
      if (i > 1 && radtype !== rad) throw new StyleError(`Pair ${this.name} requires monodisperse particles`);
      rad = radtype;
    }

    // fix deform must exist for the volume fraction corrections (no remap check in this style)
    this.flagdeform = (ctx.fixOf?.('deform') ?? ctx.fixOf?.('deform/pressure')) ? 1 : 0;
    const walls = findWallFixes(ctx);
    this.flagwall = 0;
    this.wallFix = null;
    if (walls.length > 1) throw new StyleError(`Cannot use multiple fix wall commands with pair ${this.name}`);
    if (walls.length === 1) {
      const f = walls[0];
      if (!(f instanceof FixWallBase)) throw new StyleError(`Fix ${f.style} is not compatible with pair ${this.name}`);
      this.wallFix = f as unknown as Fix & WallView;
      this.flagwall = 1;
      if (this.wallFix.walls.some((w) => !w.coord.edge && typeof w.coord.v === 'object')) this.flagwall = 2; // moving walls
    }

    // vol_T = total volume (box, or the space between the walls)
    const prd = this.prdOf(s.box.lo, s.box.hi);
    const volT = !this.flagwall ? prd[0] * prd[1] * prd[2] : (() => {
      const d = this.wallDims(prd);
      return d[0] * d[1] * d[2];
    })();

    // rad = radius of atom 0 (native: MPI_MAX of atom->radius[0]); vol_P = natoms * 4/3 pi rad^3
    this.rad = s.n > 0 ? s.radius[0] : 0;
    this.volP = s.n * (4.0 / 3.0) * Math.PI * pow(this.rad, 3.0);
    let volf = this.volP / volT;
    if (!this.flagVF) volf = 0;
    this.setIsotropic(volf);
  }

  /** The isotropic resistance constants R0, RT0, RS0 at volume fraction volf (native init_style). */
  protected setIsotropic(volf: number): void {
    const mu = this.mu, rad = this.rad;
    if (this.flaglog === 0) {
      this.R0 = 6 * Math.PI * mu * rad * (1.0 + 2.16 * volf);
      this.RT0 = 8 * Math.PI * mu * pow(rad, 3.0);
      this.RS0 = 20.0 / 3.0 * Math.PI * mu * pow(rad, 3.0) * (1.0 + 3.33 * volf + 2.80 * volf * volf);
    } else {
      this.R0 = 6 * Math.PI * mu * rad * (1.0 + 2.725 * volf - 6.583 * volf * volf);
      this.RT0 = 8 * Math.PI * mu * pow(rad, 3.0) * (1.0 + 0.749 * volf - 2.469 * volf * volf);
      this.RS0 = 20.0 / 3.0 * Math.PI * mu * pow(rad, 3.0) * (1.0 + 3.64 * volf - 6.95 * volf * volf);
    }
  }

  /** Per-evaluation arrays: owned and ghost velocities and angular velocities, forces and torques (3 per atom). */
  protected setup(pc: PairCompute): void {
    const s = pc.s;
    const nb = pc.nb;
    this.pc = pc;
    this.nlocal = pc.nlocal;
    this.nall = pc.nall;
    this.inum = this.list().inum;
    this.vxmu2f = VXMU2F[s.units.style] ?? 1.0;
    const nall = this.nall;
    this.radAll = new Float64Array(nall);
    for (let k = 0; k < nall; k++) this.radAll[k] = s.radius![nb.owner[k]];
    this.V = new Float64Array(3 * nall);
    this.W = new Float64Array(3 * nall);
    for (let k = 0; k < 3 * this.nlocal; k++) {
      this.V[k] = s.v[k];
      this.W[k] = s.omega![k];
    }
    this.F = new Float64Array(3 * nall);
    this.T = new Float64Array(3 * nall);
    const n6 = 6 * this.inum;
    this.bcg = new Float64Array(n6);
    this.xcg = new Float64Array(n6);
    this.rcg = new Float64Array(n6);
    this.rcg1 = new Float64Array(n6);
    this.pcg = new Float64Array(n6);
    this.RU = new Float64Array(n6);
  }

  /** Native reverse_comm: ghost forces and torques are added to their owners. */
  protected reverseForces(): void {
    const o = this.pc.nb.owner, F = this.F, T = this.T;
    for (let k = this.nlocal; k < this.nall; k++) {
      const i = o[k];
      F[3 * i] += F[3 * k]; F[3 * i + 1] += F[3 * k + 1]; F[3 * i + 2] += F[3 * k + 2];
      T[3 * i] += T[3 * k]; T[3 * i + 1] += T[3 * k + 1]; T[3 * i + 2] += T[3 * k + 2];
    }
  }

  /** Native forward_comm (pack v and omega): ghost copies of the owners. */
  protected forward(): void {
    this.pc.nb.forwardCopy(this.V, 3);
    this.pc.nb.forwardCopy(this.W, 3);
  }

  /** Native copy_vec_uo: the solver vector (6 per owned atom) to v and omega of the owned atoms. */
  protected copyVecUO(xc: Float64Array): void {
    for (let ii = 0; ii < this.inum; ii++) {
      for (let j = 0; j < 3; j++) {
        this.V[3 * ii + j] = xc[6 * ii + j];
        this.W[3 * ii + j] = xc[6 * ii + j + 3];
      }
    }
  }

  /** Native copy_uo_vec: the owned forces and torques to the solver vector RU. */
  protected copyUoVec(out: Float64Array): void {
    for (let ii = 0; ii < this.inum; ii++) {
      for (let j = 0; j < 3; j++) {
        out[6 * ii + j] = this.F[3 * ii + j];
        out[6 * ii + j + 3] = this.T[3 * ii + j];
      }
    }
  }

  /** Native dot_vec_vec (a sequential sum). */
  protected dot(n: number, x: Float64Array, y: Float64Array): number {
    let d = 0.0;
    for (let i = 0; i < n; i++) d += x[i] * y[i];
    return d;
  }

  /** Native compute(): the midpoint scheme (stage one at the positions, stage two at the half step positions). */
  compute(pc: PairCompute): void {
    // native picks the copy of each pair when the list is built: take the CSR of this build before the setup skip
    this.neighborCsr(pc);
    if (this.skipAtSetup && !pc.historyUpdate) return;
    this.setup(pc);
    const nall = this.nall, nlocal = this.nlocal;
    const x = pc.x;
    const xl = new Float64Array(3 * nall);
    this.F.fill(0);
    this.T.fill(0);
    this.iterate(x, 1, this.tol);
    // intermediates: positions at half the timestep (owned and ghost velocities as they stand)
    const dtv = pc.s.dt;
    for (let k = 0; k < 3 * nall; k++) xl[k] = x[k] + 0.5 * dtv * this.V[k];
    this.F.fill(0);
    this.T.fill(0);
    this.iterate(xl, 2, this.tol);
    const s = pc.s;
    for (let k = 0; k < 3 * nlocal; k++) {
      s.v[k] = this.V[k];
      s.omega![k] = this.W[k];
    }
  }

  /**
   * Native stage_one (stage 1) and stage_two (stage 2) with the same conjugate gradient: the right hand side is minus the
   * owned forces and torques, the unknowns the relative velocities (v and omega) with the fluid's shear removed.
   */
  protected iterate(x: Float64Array, stage: 1 | 2, tol: number): void {
    const inum = this.inum, n6 = 6 * inum;
    const { bcg, xcg, rcg, rcg1, pcg, RU } = this;
    this.computeRE(x);
    if (this.newton) this.reverseForces();
    for (let ii = 0; ii < inum; ii++) {
      for (let j = 0; j < 3; j++) {
        bcg[6 * ii + j] = -this.F[3 * ii + j];
        bcg[6 * ii + j + 3] = -this.T[3 * ii + j];
      }
    }
    xcg.fill(0, 0, n6);
    this.copyVecUO(xcg);
    this.forward();
    this.computeRU(x);
    if (this.newton) this.reverseForces();
    this.copyUoVec(RU);
    for (let i = 0; i < n6; i++) rcg[i] = bcg[i] - RU[i];
    for (let i = 0; i < n6; i++) pcg[i] = rcg[i];
    const normi = this.dot(n6, bcg, bcg);
    const normig = normi;
    let error: number;
    do {
      this.copyVecUO(pcg);
      this.forward();
      this.computeRU(x);
      if (this.newton) this.reverseForces();
      this.copyUoVec(RU);
      const send0 = this.dot(n6, rcg, rcg);
      const send1 = this.dot(n6, RU, pcg);
      const alpha = send0 / send1;
      const rcgDotRcg = send0;
      for (let i = 0; i < n6; i++) xcg[i] = xcg[i] + alpha * pcg[i];
      for (let i = 0; i < n6; i++) rcg1[i] = rcg[i] - alpha * RU[i];
      const r1 = this.dot(n6, rcg1, rcg1);
      const beta = r1 / rcgDotRcg;
      for (let i = 0; i < n6; i++) pcg[i] = rcg1[i] + beta * pcg[i];
      for (let i = 0; i < n6; i++) rcg[i] = rcg1[i];
      error = Math.sqrt(r1 / normig);
    } while (error > tol);
    this.copyVecUO(xcg);
    this.forward();
    if (stage === 2 && this.evflag) this.computeFh(x);
    // the fluid's velocity (gdot y, along x) and its rotation (-gdot/2 about z) are added to the owned atoms
    for (let ii = 0; ii < inum; ii++) {
      this.V[3 * ii] = this.V[3 * ii] + this.gdot * x[3 * ii + 1];
      this.W[3 * ii + 2] = this.W[3 * ii + 2] - this.gdot / 2.0;
    }
  }

  /** Native compute_RE: the strain-rate part R_FE E of the hydrodynamic force (no isotropic terms). */
  protected computeRE(x: Float64Array): void {
    if (!this.flagHI) return;
    const pc = this.pc, list = this.list();
    const csr = this.neighborCsr();
    const type = pc.type, cutsq = this.cutsq, nt = this.ntypes + 1;
    const cutInner = this.cutInnerCoeff, Ef = this.Ef, F = this.F, T = this.T;
    const mu = this.mu, flaglog = this.flaglog, vxmu2f = this.vxmu2f, nlocal = this.nlocal, newton = this.newton;
    for (let i = 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      const radi = this.radAll[i];
      for (let k = csr.start[i], k1 = csr.start[i + 1]; k < k1; k++) {
        const j = csr.j[k];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const r = Math.sqrt(rsq);
        // loc of the point of closest approach on particle i from its center
        const xl0 = -delx / r * radi, xl1 = -dely / r * radi, xl2 = -delz / r * radi;
        let hSep = r - 2.0 * radi;
        if (r < cutInner[itype * nt + jtype]) hSep = cutInner[itype * nt + jtype] - 2.0 * radi;
        hSep = hSep / radi;
        let aSq: number, aSh = 0;
        if (flaglog) {
          aSq = 6 * Math.PI * mu * radi * (1.0 / 4.0 / hSep + 9.0 / 40.0 * log(1.0 / hSep));
          aSh = 6 * Math.PI * mu * radi * (1.0 / 6.0 * log(1.0 / hSep));
        } else {
          aSq = 6 * Math.PI * mu * radi * (1.0 / 4.0 / hSep);
        }
        // relative velocity at the point of closest approach due to Ef only
        const vr1 = -2.0 * (Ef[0] * xl0 + Ef[1] * xl1 + Ef[2] * xl2);
        const vr2 = -2.0 * (Ef[3] * xl0 + Ef[4] * xl1 + Ef[5] * xl2);
        const vr3 = -2.0 * (Ef[6] * xl0 + Ef[7] * xl1 + Ef[8] * xl2);
        const vnnr = (vr1 * delx + vr2 * dely + vr3 * delz) / r;
        const vn1 = vnnr * delx / r, vn2 = vnnr * dely / r, vn3 = vnnr * delz / r;
        const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
        let fx = aSq * vn1, fy = aSq * vn2, fz = aSq * vn3;
        if (flaglog) {
          fx = fx + aSh * vt1;
          fy = fy + aSh * vt2;
          fz = fz + aSh * vt3;
        }
        fx = vxmu2f * fx; fy = vxmu2f * fy; fz = vxmu2f * fz;
        F[3 * i] -= fx; F[3 * i + 1] -= fy; F[3 * i + 2] -= fz;
        if (newton || j < nlocal) {
          F[3 * j] += fx; F[3 * j + 1] += fy; F[3 * j + 2] += fz;
        }
        if (flaglog) {
          const tx = xl1 * fz - xl2 * fy, ty = xl2 * fx - xl0 * fz, tz = xl0 * fy - xl1 * fx;
          T[3 * i] -= vxmu2f * tx; T[3 * i + 1] -= vxmu2f * ty; T[3 * i + 2] -= vxmu2f * tz;
          if (newton || j < nlocal) {
            T[3 * j] -= vxmu2f * tx; T[3 * j + 1] -= vxmu2f * ty; T[3 * j + 2] -= vxmu2f * tz;
          }
        }
      }
    }
  }

  /** Native compute_RU: R_FU U, with the isotropic terms of the owned atoms. */
  protected computeRU(x: Float64Array): void {
    const pc = this.pc, list = this.list();
    const csr = this.neighborCsr();
    const type = pc.type, cutsq = this.cutsq, nt = this.ntypes + 1;
    const cutInner = this.cutInnerCoeff, F = this.F, T = this.T, V = this.V, W = this.W;
    const mu = this.mu, flaglog = this.flaglog, vxmu2f = this.vxmu2f, nlocal = this.nlocal, newton = this.newton;
    const vT = this.volTCompute(pc);
    if (vT !== null) {
      const volf = this.volP / vT;
      if (flaglog === 0) {
        this.R0 = 6 * Math.PI * mu * this.rad * (1.0 + 2.16 * volf);
        this.RT0 = 8 * Math.PI * mu * pow(this.rad, 3.0);
      } else {
        this.R0 = 6 * Math.PI * mu * this.rad * (1.0 + 2.725 * volf - 6.583 * volf * volf);
        this.RT0 = 8 * Math.PI * mu * pow(this.rad, 3.0) * (1.0 + 0.749 * volf - 2.469 * volf * volf);
      }
    }
    const R0 = this.R0, RT0 = this.RT0;
    F.fill(0);
    T.fill(0);
    for (let i = 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      const radi = this.radAll[i];
      const wi0 = W[3 * i], wi1 = W[3 * i + 1], wi2 = W[3 * i + 2];
      // contribution due to the isotropic terms
      F[3 * i] += -vxmu2f * R0 * V[3 * i];
      F[3 * i + 1] += -vxmu2f * R0 * V[3 * i + 1];
      F[3 * i + 2] += -vxmu2f * R0 * V[3 * i + 2];
      T[3 * i] += -vxmu2f * RT0 * wi0;
      T[3 * i + 1] += -vxmu2f * RT0 * wi1;
      T[3 * i + 2] += -vxmu2f * RT0 * wi2;
      if (!this.flagHI) continue;
      for (let k = csr.start[i], k1 = csr.start[i + 1]; k < k1; k++) {
        const j = csr.j[k];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const r = Math.sqrt(rsq);
        const wj0 = W[3 * j], wj1 = W[3 * j + 1], wj2 = W[3 * j + 2];
        const xl0 = -delx / r * radi, xl1 = -dely / r * radi, xl2 = -delz / r * radi;
        // velocity at the point of closest approach on both particles, v = v + omega x xl
        const vi0 = V[3 * i] + (wi1 * xl2 - wi2 * xl1);
        const vi1 = V[3 * i + 1] + (wi2 * xl0 - wi0 * xl2);
        const vi2 = V[3 * i + 2] + (wi0 * xl1 - wi1 * xl0);
        const vj0 = V[3 * j] - (wj1 * xl2 - wj2 * xl1);
        const vj1 = V[3 * j + 1] - (wj2 * xl0 - wj0 * xl2);
        const vj2 = V[3 * j + 2] - (wj0 * xl1 - wj1 * xl0);
        // scalar resistances
        let hSep = r - 2.0 * radi;
        if (r < cutInner[itype * nt + jtype]) hSep = cutInner[itype * nt + jtype] - 2.0 * radi;
        hSep = hSep / radi;
        let aSq: number, aSh = 0, aPu = 0;
        if (flaglog) {
          aSq = 6.0 * Math.PI * mu * radi * (1.0 / 4.0 / hSep + 9.0 / 40.0 * log(1.0 / hSep));
          aSh = 6.0 * Math.PI * mu * radi * (1.0 / 6.0 * log(1.0 / hSep));
          aPu = 8.0 * Math.PI * mu * pow(radi, 3.0) * (3.0 / 160.0 * log(1.0 / hSep));
        } else {
          aSq = 6.0 * Math.PI * mu * radi * (1.0 / 4.0 / hSep);
        }
        const vr1 = vi0 - vj0, vr2 = vi1 - vj1, vr3 = vi2 - vj2;
        const vnnr = (vr1 * delx + vr2 * dely + vr3 * delz) / r;
        const vn1 = vnnr * delx / r, vn2 = vnnr * dely / r, vn3 = vnnr * delz / r;
        const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
        let fx = aSq * vn1, fy = aSq * vn2, fz = aSq * vn3;
        if (flaglog) {
          fx = fx + aSh * vt1;
          fy = fy + aSh * vt2;
          fz = fz + aSh * vt3;
        }
        fx = vxmu2f * fx; fy = vxmu2f * fy; fz = vxmu2f * fz;
        F[3 * i] -= fx; F[3 * i + 1] -= fy; F[3 * i + 2] -= fz;
        if (newton || j < nlocal) {
          F[3 * j] += fx; F[3 * j + 1] += fy; F[3 * j + 2] += fz;
        }
        if (flaglog) {
          let tx = xl1 * fz - xl2 * fy, ty = xl2 * fx - xl0 * fz, tz = xl0 * fy - xl1 * fx;
          T[3 * i] -= vxmu2f * tx; T[3 * i + 1] -= vxmu2f * ty; T[3 * i + 2] -= vxmu2f * tz;
          if (newton || j < nlocal) {
            T[3 * j] -= vxmu2f * tx; T[3 * j + 1] -= vxmu2f * ty; T[3 * j + 2] -= vxmu2f * tz;
          }
          // torque due to a_pu
          const wdotn = ((wi0 - wj0) * delx + (wi1 - wj1) * dely + (wi2 - wj2) * delz) / r;
          const wt1 = (wi0 - wj0) - wdotn * delx / r;
          const wt2 = (wi1 - wj1) - wdotn * dely / r;
          const wt3 = (wi2 - wj2) - wdotn * delz / r;
          tx = aPu * wt1;
          ty = aPu * wt2;
          tz = aPu * wt3;
          T[3 * i] -= vxmu2f * tx; T[3 * i + 1] -= vxmu2f * ty; T[3 * i + 2] -= vxmu2f * tz;
          if (newton || j < nlocal) {
            T[3 * j] += vxmu2f * tx; T[3 * j + 1] += vxmu2f * ty; T[3 * j + 2] += vxmu2f * tz;
          }
        }
      }
    }
  }

  /**
   * Native compute_Fh: the final hydrodynamic force is zero (f and torque are cleared); only the stresslet virial is
   * tallied (the isotropic RS0 term of each atom and the pair terms, with native's ev_tally_xyz arguments).
   */
  protected computeFh(x: Float64Array): void {
    const pc = this.pc, list = this.list();
    const csr = this.neighborCsr();
    const type = pc.type, cutsq = this.cutsq, nt = this.ntypes + 1;
    const cutInner = this.cutInnerCoeff, Ef = this.Ef, V = this.V, W = this.W;
    const mu = this.mu, flaglog = this.flaglog, vxmu2f = this.vxmu2f, nlocal = this.nlocal, newton = this.newton;
    const gdot = this.gdot;
    const vT = this.volTCompute(pc);
    if (vT !== null) {
      const volf = this.volP / vT;
      if (flaglog === 0) {
        this.RS0 = 20.0 / 3.0 * Math.PI * mu * pow(this.rad, 3.0) * (1.0 + 3.33 * volf + 2.80 * volf * volf);
      } else {
        this.RS0 = 20.0 / 3.0 * Math.PI * mu * pow(this.rad, 3.0) * (1.0 + 3.64 * volf - 6.95 * volf * volf);
      }
    }
    const RS0 = this.RS0;
    this.F.fill(0);
    this.T.fill(0);
    for (let i = 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      const radi = this.radAll[i];
      // isotropic RS0 stress: a pseudo force gives the required contribution (delx and fy only)
      const fyS = vxmu2f * RS0 * gdot / 2.0 / radi;
      if (this.evflag) evTallyXYZ(pc, i, i, nlocal, newton, -0.0, -fyS, -0.0, radi, 0.0, 0.0);
      const wi0 = W[3 * i], wi1 = W[3 * i + 1], wi2 = W[3 * i + 2];
      if (!this.flagHI) continue;
      for (let k = csr.start[i], k1 = csr.start[i + 1]; k < k1; k++) {
        const j = csr.j[k];
        const dx = xtmp - x[3 * j], dy = ytmp - x[3 * j + 1], dz = ztmp - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const r = Math.sqrt(rsq);
        const wj0 = W[3 * j], wj1 = W[3 * j + 1], wj2 = W[3 * j + 2];
        const xl0 = -dx / r * radi, xl1 = -dy / r * radi, xl2 = -dz / r * radi;
        const vi0 = V[3 * i] + (wi1 * xl2 - wi2 * xl1);
        const vi1 = V[3 * i + 1] + (wi2 * xl0 - wi0 * xl2);
        const vi2 = V[3 * i + 2] + (wi0 * xl1 - wi1 * xl0);
        const vj0 = V[3 * j] - (wj1 * xl2 - wj2 * xl1);
        const vj1 = V[3 * j + 1] - (wj2 * xl0 - wj0 * xl2);
        const vj2 = V[3 * j + 2] - (wj0 * xl1 - wj1 * xl0);
        // relative velocity at the point of closest approach, including the Ef of the fluid
        const vr1 = vi0 - vj0 - 2.0 * (Ef[0] * xl0 + Ef[1] * xl1 + Ef[2] * xl2);
        const vr2 = vi1 - vj1 - 2.0 * (Ef[3] * xl0 + Ef[4] * xl1 + Ef[5] * xl2);
        const vr3 = vi2 - vj2 - 2.0 * (Ef[6] * xl0 + Ef[7] * xl1 + Ef[8] * xl2);
        const vnnr = (vr1 * dx + vr2 * dy + vr3 * dz) / r;
        const vn1 = vnnr * dx / r, vn2 = vnnr * dy / r, vn3 = vnnr * dz / r;
        const vt1 = vr1 - vn1, vt2 = vr2 - vn2, vt3 = vr3 - vn3;
        let hSep = r - 2.0 * radi;
        if (r < cutInner[itype * nt + jtype]) hSep = cutInner[itype * nt + jtype] - 2.0 * radi;
        hSep = hSep / radi;
        let aSq: number, aSh = 0;
        if (flaglog) {
          aSq = 6.0 * Math.PI * mu * radi * (1.0 / 4.0 / hSep + 9.0 / 40.0 * log(1.0 / hSep));
          aSh = 6.0 * Math.PI * mu * radi * (1.0 / 6.0 * log(1.0 / hSep));
        } else {
          aSq = 6.0 * Math.PI * mu * radi * (1.0 / 4.0 / hSep);
        }
        let fx = aSq * vn1, fy = aSq * vn2, fz = aSq * vn3;
        if (flaglog) {
          fx = fx + aSh * vt1;
          fy = fy + aSh * vt2;
          fz = fz + aSh * vt3;
        }
        fx = vxmu2f * fx; fy = vxmu2f * fy; fz = vxmu2f * fz;
        if (this.evflag) evTallyXYZ(pc, i, j, nlocal, newton, -fx, -fy, -fz, dx, dy, dz);
      }
    }
  }
}
