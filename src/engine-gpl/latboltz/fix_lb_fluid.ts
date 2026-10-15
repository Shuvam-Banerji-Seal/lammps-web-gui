/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/LATBOLTZ/fix_lb_fluid.cpp and src/LATBOLTZ/fix_lb_fluid.h at commit b332f28 (fix lb/fluid: a D3Q15
 * lattice-Boltzmann fluid that exchanges momentum with the particles of its group through the trilinear stencil
 * (stencil 2) or the 3-point IBM stencil (stencil 3); the flat z walls with wall velocities, the x pressure
 * boundary, the Gaussian noise of the fluctuating hydrodynamics, scaleGamma, dof, linearInit and the scalar and
 * vector outputs). Contributing authors: Frances Mackay, Santtu Ollila, Tyson Whitehead, Colin Denniston (UWO).
 * Copyright (2003) Sandia Corporation, GNU General Public License. This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_lb_fluid.html: "The lb/fluid fix ... implements a lattice-Boltzmann fluid ... The fluid
 * is coupled to the particles in the group through a force that is interpolated to and from the fluid grid."
 *
 * Browser-engine notes. Serial only (one block: the periodic ghost layers that native fills from its
 * neighbours are copied from the opposite side of the same block, in native's order). The per-atom hydroF
 * and massp arrays are kept by atom ID, so they follow the atoms through the engine's sorting as native's
 * copy_arrays does. Pit geometry (npits, wp, sw) is ported in pit_geometry.ts. Not ported and refused by name: D3Q19,
 * stencil 4 (Keys), dumpxdmf, read_restart and write_restart.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { BBL, BBL_WALL_BOTTOM, BBL_WALL_TOP, E15, MG15, NG15, OD, W15 } from './lb_tables';
import { flatGeometry, LatticeGeometry, pitGeometry, PitSpec, wholeIndex } from './pit_geometry';

const NV = 15;
/** W15[l] * MG15[ii][l] (the first two factors of the equilibrium sum, in native's order), indexed ii * NV + l. */
const WM15 = Float64Array.from({ length: NV * NV }, (_, n) => W15[n % NV] * MG15[n]);
const NG15F = Float64Array.from(NG15);
const FLT_EPSILON = 1.1920928955078125e-7;

/** C round(): halves away from zero. */
const cRound = (x: number): number => (x < 0 ? -Math.round(-x) : Math.round(x));
/** C lround(): halves away from zero (the same rounding as cRound). */
const lround = cRound;

const intArg = (w: string | undefined, what: string): number => {
  const n = Number(w);
  if (w === undefined || !Number.isInteger(n)) throw new StyleError(`Illegal fix lb/fluid ${what} ${w ?? ''}`);
  return n;
};
const numArg = (w: string | undefined, what: string): number => {
  const n = Number(w);
  if (w === undefined || !Number.isFinite(n)) throw new StyleError(`Illegal fix lb/fluid ${what} ${w ?? ''}`);
  return n;
};

/** Refused keywords (not ported). */
const REFUSED: Record<string, string> = {
  D3Q19: 'the D3Q19 lattice is not ported to the browser engine (use the default D3Q15)',
  dumpxdmf: 'dumpxdmf (fluid output files) is not supported by the browser engine',
  read_restart: 'read_restart is not supported by the browser engine',
  write_restart: 'write_restart is not supported by the browser engine',
};

export class FixLbFluid extends Fix {
  readonly style = 'lb/fluid';

  // ---- parameters (fix_lb_fluid.h names kept where it helps the port)
  viscosity: number;
  densityinitReal: number;
  aoReal = 0;
  T = 300.0;
  setdx = 1;
  seta0 = 1;
  setdof = 0;
  dmLb = -1.0;
  dxLb = 0;
  dtLb = 0;
  noisestress = 0;
  seed = 0;
  nStencil = 2;
  pressure = 0;
  rhofactor = 0;
  rhoH = 0;
  rhoL = 0;
  bodyforcex = 0;
  bodyforcey = 0;
  bodyforcez = 0;
  vwtp = 0;
  vwbt = 0;
  linInit = 0;
  Gamma: Float64Array;
  densityinit = 0;
  a0 = 0;
  tau = 0;
  K0 = 0;
  kB = 0;
  namp = 0;
  noisefactor = 1;
  dofLb = 0;
  fixviscouslb = 0;
  random: RanMars | null = null;

  // ---- grid
  Nbx = 0;
  Nby = 0;
  Nbz = 0;
  /** subNb* = Nb* + 2 (serial): the grid including one ghost layer on each side. */
  subNbx = 0;
  subNby = 0;
  subNbz = 0;
  /** Sizes of the arrays on the density / force grid (subNb + 3). */
  QX = 0;
  QY = 0;
  QZ = 0;

  // ---- fields (flat; f-grid index fi = (i*subNby + j)*subNbz + k, times NV for the populations)
  private fLb!: Float64Array;
  private fnew!: Float64Array;
  private feq!: Float64Array;
  private density!: Float64Array;
  private u!: Float64Array;
  private Ff!: Float64Array;
  private Wf!: Float64Array;
  /** pit geometry (null without npits), its sidewall flag and the height of the slit. */
  pit: PitSpec | null = null;
  pitHs = 0;
  /** sublattice type (0 bulk, 1 bounce-back, 2 outside, 3 flat wall) and orientation of each f-grid node. */
  private stype!: Int8Array;
  private sori!: Int8Array;

  /** Per-atom hydrodynamic force and mass, keyed by atom ID (follows atoms through sorting). */
  hydroF = new Float64Array(0);
  massp = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal fix lb/fluid command: fix ID group lb/fluid nevery viscosity density ...');
    const state = sys.state;
    if (state.dimension !== 3) throw new StyleError('fix lb/fluid requires a 3d simulation');
    const box = state.box;
    const nevery = intArg(args[0], 'nevery');
    if (nevery <= 0) throw new StyleError('Illegal fix lb/fluid nevery value');
    this.nevery = nevery;
    this.viscosity = numArg(args[1], 'viscosity');
    this.densityinitReal = numArg(args[2], 'density');

    const ntypes = state.ntypes;
    this.Gamma = new Float64Array(ntypes + 1).fill(1.0);
    this.vectorFlag = true;
    this.sizeVector = 5;
    this.scalarFlag = true;

    let dxGiven = false;
    let a0Given = false;
    let pit: PitSpec | null = null;
    let sidewalls = 0;
    let wpArg = 0;
    for (let iarg = 3; iarg < args.length; ) {
      const key = args[iarg];
      if (Object.prototype.hasOwnProperty.call(REFUSED, key)) throw new StyleError(`fix lb/fluid ${key}: ${REFUSED[key]}`);
      const need = (n: number) => {
        if (iarg + n > args.length) throw new StyleError(`Illegal fix lb/fluid command: ${key}`);
      };
      if (key === 'scaleGamma') {
        need(3);
        const itype = args[iarg + 1] === '*' ? 0 : intArg(args[iarg + 1], 'scaleGamma type');
        const scale = numArg(args[iarg + 2], 'scaleGamma factor');
        if (itype < 0 || itype > ntypes) throw new StyleError(`Illegal fix lb/fluid command: ${key}`);
        if (itype) this.Gamma[itype] = scale;
        else for (let it = 1; it <= ntypes; it++) this.Gamma[it] = scale;
        iarg += 3;
      } else if (key === 'dx') {
        need(2);
        this.dxLb = numArg(args[iarg + 1], 'dx');
        dxGiven = true;
        this.setdx = 0;
        iarg += 2;
      } else if (key === 'dm') {
        need(2);
        this.dmLb = numArg(args[iarg + 1], 'dm');
        iarg += 2;
      } else if (key === 'a0') {
        need(2);
        this.aoReal = numArg(args[iarg + 1], 'a0');
        this.seta0 = 0;
        a0Given = true;
        iarg += 2;
      } else if (key === 'noise') {
        need(3);
        this.noisestress = 1;
        this.T = numArg(args[iarg + 1], 'noise temperature');
        this.seed = intArg(args[iarg + 2], 'noise seed');
        iarg += 3;
      } else if (key === 'stencil') {
        need(2);
        this.nStencil = intArg(args[iarg + 1], 'stencil');
        if (this.nStencil === 4) throw new StyleError('fix lb/fluid stencil 4 (Keys interpolation) is not ported to the browser engine');
        iarg += 2;
      } else if (key === 'zwall_velocity') {
        need(3);
        if (box.periodic[2]) throw new StyleError('setting a z wall velocity without implementing fixed BCs in z');
        this.vwbt = numArg(args[iarg + 1], 'zwall_velocity');
        this.vwtp = numArg(args[iarg + 2], 'zwall_velocity');
        iarg += 3;
      } else if (key === 'pressurebcx') {
        need(2);
        this.pressure = 1;
        this.rhofactor = numArg(args[iarg + 1], 'pressurebcx');
        iarg += 2;
      } else if (key === 'bodyforce') {
        need(4);
        this.bodyforcex = numArg(args[iarg + 1], 'bodyforce');
        this.bodyforcey = numArg(args[iarg + 2], 'bodyforce');
        this.bodyforcez = numArg(args[iarg + 3], 'bodyforce');
        iarg += 4;
      } else if (key === 'npits') {
        // native: npits N h_p l_p l_pp l_e (fix_lb_fluid.cpp constructor)
        need(6);
        pit = {
          npits: intArg(args[iarg + 1], 'npits'),
          hp: intArg(args[iarg + 2], 'npits h_p'),
          lp: intArg(args[iarg + 3], 'npits l_p'),
          lpp: intArg(args[iarg + 4], 'npits l_pp'),
          le: intArg(args[iarg + 5], 'npits l_e'),
          wp: 0,
          sw: 0,
        };
        iarg += 6;
      } else if (key === 'sw') {
        // native: "sw" takes no value (y-sidewalls on)
        sidewalls = 1;
        iarg += 1;
      } else if (key === 'wp') {
        need(2);
        wpArg = intArg(args[iarg + 1], 'wp');
        iarg += 2;
      } else if (key === 'linearInit') {
        this.linInit = 1;
        iarg += 1;
      } else if (key === 'dof') {
        need(2);
        this.setdof = intArg(args[iarg + 1], 'dof');
        iarg += 2;
      } else {
        throw new StyleError(`Illegal fix lb/fluid command: ${key}`);
      }
    }

    // ---- timestep and lattice spacing (fix_lb_fluid.cpp constructor; procgrid = 1 in serial)
    this.dtLb = nevery * state.dt;
    const lo = box.lo, hi = box.hi;
    const xprd = hi[0] - lo[0], yprd = hi[1] - lo[1], zprd = hi[2] - lo[2];
    if (!dxGiven) {
      const dx1 = Math.sqrt((3.0 * this.viscosity * this.dtLb) / this.densityinitReal);
      const mindomain = Math.min(Math.min(xprd, yprd), zprd);
      this.dxLb = mindomain / Math.floor(mindomain / dx1);
    }
    if (!a0Given) this.aoReal = 0.333333333333333 * this.dxLb * this.dxLb / this.dtLb / this.dtLb;
    this.rhofactor = (this.rhofactor / 1000) * xprd / this.densityinitReal / this.aoReal;
    if (Math.abs(this.rhofactor) > 2) throw new StyleError('Illegal pressure jump');
    if (Math.abs(this.rhofactor) > 0.15) sys.warn('Huge pressure jump requested.');

    // ---- grid
    const dx = this.dxLb;
    const Nx = lround(xprd / dx), Ny = lround(yprd / dx), NzOrig = lround(zprd / dx);
    const walls = !box.periodic[2];
    if (!box.periodic[0]) throw new StyleError('the x-direction must be periodic');
    if (box.boundary[1][0] === 's' || box.boundary[1][0] === 'm') throw new StyleError('the y-direction can not have shrink-wrap boundary conditions');
    if (box.boundary[2][0] === 's' || box.boundary[2][0] === 'm') throw new StyleError('the z-direction can not have shrink-wrap boundary conditions');
    this.Nbx = Nx;
    this.Nby = Ny;
    this.Nbz = NzOrig;
    if (NzOrig + 2 < 7 || Nx + 2 < 7 || Ny + 2 < 7) throw new StyleError('Need at least 5 grid points in each direction per processor');
    if (walls) this.Nbz += 1;
    // grid checks for pits (fix_lb_fluid.cpp constructor, serial)
    if (pit !== null) {
      if (!walls) throw new StyleError('the z-direction cannot have periodic boundary conditions when there are pits.');
      const p = pit as PitSpec;
      p.wp = wpArg === 0 ? Ny : wpArg;
      p.sw = sidewalls;
      const hs = this.Nbz - 1 - p.hp;
      if (hs <= 1) throw new StyleError('hp too big for system in z-direction');
      if (p.wp > Ny) throw new StyleError('wp bigger than system in y-direction');
      if (2 * p.le + p.npits * p.lp + (p.npits - 1) * p.lpp < Nx)
        throw new StyleError('length of pits and end segments too small to fill system in x-direction');
      if (this.vwtp !== 0.0 || this.vwbt !== 0.0) throw new StyleError('Moving walls not compatible with pit geometry options');
      this.pit = p;
      this.pitHs = hs;
    } else if (sidewalls) {
      throw new StyleError('Sidewalls require pit geometry (you can enable by setting npits to 0 or higher)');
    }
    this.subNbx = Nx + 2;
    this.subNby = Ny + 2;
    this.subNbz = this.Nbz + 2;
    this.QX = this.subNbx + 3;
    this.QY = this.subNby + 3;
    this.QZ = this.subNbz + 3;

    // ---- arrays
    const nf = this.subNbx * this.subNby * this.subNbz;
    const nq = this.QX * this.QY * this.QZ;
    this.fLb = new Float64Array(nf * NV);
    this.fnew = new Float64Array(nf * NV);
    this.feq = new Float64Array(nf * NV);
    this.density = new Float64Array(nq);
    this.u = new Float64Array(nq * 3);
    this.Ff = new Float64Array(nq * 3);
    this.Wf = new Float64Array(nq);
    this.stype = new Int8Array(nf);
    this.sori = new Int8Array(nf);

    if (this.noisestress === 1) {
      this.random = new RanMars(this.seed);
    }

    this.rescale(state.units.boltz, state.units.mvv2e);
    this.initializeGeometry(walls);
    this.initializeLB();
    this.parametercalcFull();
  }

  // ---------------------------------------------------------------- indices
  /** f-grid node index. */
  private fi(i: number, j: number, k: number): number {
    return (i * this.subNby + j) * this.subNbz + k;
  }
  /** density / force / velocity grid node index. */
  private qi(i: number, j: number, k: number): number {
    return (i * this.QY + j) * this.QZ + k;
  }

  // ---------------------------------------------------------------- setup of the lattice
  /** rescale(): lattice units with dx = dt = dm = 1 (fix_lb_fluid.cpp rescale()). */
  private rescale(boltz: number, mvv2e: number): void {
    const dx = this.dxLb, dt = this.dtLb;
    this.densityinit = this.densityinitReal * dx * dx * dx;
    if (this.dmLb < 0) {
      this.dmLb = this.densityinit;
      this.densityinit = 1.0;
    } else {
      this.densityinit /= this.dmLb;
    }
    this.vwtp = (this.vwtp * dt) / dx;
    this.vwbt = (this.vwbt * dt) / dx;
    this.bodyforcex = (this.bodyforcex * dt * dt) / dx;
    this.bodyforcey = (this.bodyforcey * dt * dt) / dx;
    this.bodyforcez = (this.bodyforcez * dt * dt) / dx;
    if (this.pressure) {
      this.rhoH = (1.0 + this.rhofactor * 0.5) * this.densityinitReal;
      this.rhoL = (1.0 - this.rhofactor * 0.5) * this.densityinitReal;
      this.rhoH = (this.rhoH * dx * dx * dx) / this.dmLb;
      this.rhoL = (this.rhoL * dx * dx * dx) / this.dmLb;
    } else {
      this.rhoH = this.rhoL = this.densityinit;
    }
    this.tau = ((3.0 * this.viscosity) / this.densityinitReal) * dt * dt / dx / dx;
    this.tau /= dt;
    this.tau = this.tau + 0.5;
    this.a0 = (this.aoReal * dt * dt) / (dx * dx);
    if (this.noisestress === 1 && this.a0 > 0.5555555) {
      throw new StyleError('Fix lb/fluid ERROR: the Lattice Boltzmann dx and dt need to be chosen such that the scaled a_0 < 5/9');
    }
    if (this.a0 >= 1.0) {
      throw new StyleError('Fix lb/fluid ERROR: the lattice Boltzmann dx and dt do not satisfy the Courant condition.');
    }
    this.kB = ((boltz / mvv2e) * dt * dt) / dx / dx / this.dmLb;
    this.namp = (2.0 * this.kB * this.T * (this.tau - 0.5)) / 3.0;
    this.noisefactor = 1.0;
    if (this.a0 <= 0.333333333333333) this.K0 = 5.17 * (0.333333333333333 - this.a0);
    else this.K0 = 2.57 * (this.a0 - 0.333333333333333);
  }

  /**
   * initializeGlobalGeometry() (flat walls or pit geometry) and initializeGeometry() for the one block: the sublattice
   * takes the whole-lattice site of each node, wrapped periodically, and the two z ghost layers are corrected where the
   * wrap crossed a fixed boundary that bounds no fluid on that side (fix_lb_fluid.cpp, initializeGeometry).
   */
  private initializeGeometry(walls: boolean): void {
    const Nbx = this.Nbx, Nby = this.Nby, Nbz = this.Nbz;
    const whole: LatticeGeometry = this.pit
      ? pitGeometry(Nbx, Nby, Nbz, this.pitHs, this.pit)
      : flatGeometry(Nbx, Nby, Nbz, walls);
    const wt = whole.type, wo = whole.ori;
    // inlet and outlet must have the same fluid sites (native's openingsites check)
    let inlet = 0, outlet = 0;
    for (let j = 0; j < Nby; j++)
      for (let k = 0; k < Nbz; k++) {
        if (wt[wholeIndex(whole, 0, j, k)] !== 2) inlet++;
        if (wt[wholeIndex(whole, Nbx - 1, j, k)] !== 2) outlet++;
      }
    if (inlet !== outlet) throw new StyleError('inlet and outlet geometry do not match in x-direction');

    for (let i = 0; i < this.subNbx; i++)
      for (let j = 0; j < this.subNby; j++)
        for (let k = 0; k < this.subNbz; k++) {
          const wi = (i - 1 + Nbx) % Nbx, wj = (j - 1 + Nby) % Nby, wk = (k - 1 + Nbz) % Nbz;
          const n = wholeIndex(whole, wi, wj, wk);
          let type = wt[n], ori = wo[n];
          if (k === 0) {
            // true boundary ghost rather than neighbouring processor ghost
            const just = wt[wholeIndex(whole, wi, wj, (k + Nbz) % Nbz)];
            if (type === 1 && just === 2) {
              type = 2;
              ori = 0;
            }
          } else if (k === this.subNbz - 1) {
            const just = wt[wholeIndex(whole, wi, wj, (k - 2 + Nbz) % Nbz)];
            if (type === 1 && just === 2) {
              type = 2;
              ori = 0;
            }
          }
          this.stype[this.fi(i, j, k)] = type;
          this.sori[this.fi(i, j, k)] = ori;
        }
  }

  /** initializeLB(): uniform (or linearly interpolated) density and velocity, equilibrium-free populations at u. */
  private initializeLB(): void {
    const dx = this.dxLb;
    const sublo = this.sys.state.box.lo;
    const boxlo = this.sys.state.box.lo;
    const Nbx = this.Nbx, Nbz = this.Nbz;
    for (let i = 0; i < this.subNbx + 3; i++) {
      const ix = i === this.subNbx + 2 ? cRound((sublo[0] - boxlo[0]) / dx) - 2 : cRound((sublo[0] - boxlo[0]) / dx) + (i - 1);
      for (let j = 0; j < this.subNby + 3; j++)
        for (let k = 0; k < this.subNbz + 3; k++) {
          const q = this.qi(i, j, k);
          this.density[q] = this.linInit
            ? this.rhoH + ((this.rhoL - this.rhoH) * ix) / (Nbx - 1)
            : this.densityinit;
          const iz = k === this.subNbz + 2 ? cRound((sublo[2] - boxlo[2]) / dx) - 2 : cRound((sublo[2] - boxlo[2]) / dx) + (k - 1);
          this.u[3 * q] = 0.0;
          this.u[3 * q + 1] = this.linInit ? this.vwbt + ((this.vwtp - this.vwbt) * iz) / (Nbz - 1) : 0.0;
          this.u[3 * q + 2] = 0.0;
        }
    }
    for (let i = 0; i < this.subNbx; i++)
      for (let j = 0; j < this.subNby; j++)
        for (let k = 0; k < this.subNbz; k++) {
          const q = this.qi(i, j, k);
          const f = this.fi(i, j, k) * NV;
          for (let m = 0; m < NV; m++) {
            this.fLb[f + m] =
              this.density[q] * W15[m] *
              (1 + (this.u[3 * q] * E15[m][0] + this.u[3 * q + 1] * E15[m][1] + this.u[3 * q + 2] * E15[m][2]) / this.a0);
          }
        }
  }

  // ---------------------------------------------------------------- periodic ghost layers (serial)
  /**
   * Copies, for every node of the given (j, k) or (i, k) or (i, j) range, the value of a source plane onto a
   * destination plane. Used for the ghost exchanges native makes with MPI between a block and itself.
   * `arr` has `comp` components per node on a grid of sizes (nx, ny, nz).
   */
  private copyPlanes(
    arr: Float64Array, ny: number, nz: number, comp: number,
    dim: 0 | 1 | 2, pairs: readonly (readonly [number, number])[],
    a0: number, a1: number, b0: number, b1: number,
  ): void {
    for (const [dst, src] of pairs) {
      if (dim === 0) {
        // plane i = dst from i = src; ranges over j in [a0, a1), k in [b0, b1)
        for (let j = a0; j < a1; j++)
          for (let k = b0; k < b1; k++) {
            const d = ((dst * ny + j) * nz + k) * comp, s = ((src * ny + j) * nz + k) * comp;
            for (let c = 0; c < comp; c++) arr[d + c] = arr[s + c];
          }
      } else if (dim === 1) {
        for (let i = a0; i < a1; i++)
          for (let k = b0; k < b1; k++) {
            const d = ((i * ny + dst) * nz + k) * comp, s = ((i * ny + src) * nz + k) * comp;
            for (let c = 0; c < comp; c++) arr[d + c] = arr[s + c];
          }
      } else {
        for (let i = a0; i < a1; i++)
          for (let j = b0; j < b1; j++) {
            const d = ((i * ny + j) * nz + dst) * comp, s = ((i * ny + j) * nz + src) * comp;
            for (let c = 0; c < comp; c++) arr[d + c] = arr[s + c];
          }
      }
    }
  }

  /** Pairs (dst, src) of a serial block exchange of the lower and upper ghost layers (n = interior count). */
  private lowerUpperPairs(n: number): [number, number][] {
    return [[0, n], [n + 1, 1]];
  }
  /** Pairs for the density and velocity exchange of the outer ghost layers (native: subNb, +1, +2 from 2, 3, subNb-3). */
  private outerPairs(n: number): [number, number][] {
    return [[n + 2, 2], [n + 3, 3], [n + 4, n - 1]];
  }
  /** Pairs for the Ff / Wf / u exchange (native: wrap of all outer ghosts, including the ones before the block). */
  private wrapPairs(n: number): [number, number][] {
    return [[n + 1, 1], [n + 2, 2], [n + 3, 3], [n + 4, n - 1], [0, n]];
  }

  /** Adds the ghost-layer values of a grid into the interior nodes they stand for (serial calc_fluidforce folds). */
  private foldPairs(n: number): [number, number][] {
    // dst (interior) += src (ghost): native adds Fftemp[0] to subNb-2, [1] to subNb-3, [2] to 1, [3] to 2, [4] to 3
    return [[n, 0], [n - 1, n + 4], [1, n + 1], [2, n + 2], [3, n + 3]];
  }

  /** Adds ghost-layer values of an array with `comp` components into the interior nodes (sequence x, y, z). */
  private foldGrid(arr: Float64Array, comp: number): void {
    const QX = this.QX, QY = this.QY, QZ = this.QZ;
    const nx = this.subNbx - 2, ny = this.subNby - 2, nz = this.subNbz - 2;
    const pairs = this.foldPairs(nx);
    // x: for all j, k of the padded grid
    for (const [dst, src] of pairs) {
      for (let j = 0; j < QY; j++)
        for (let k = 0; k < QZ; k++) {
          const d = ((dst * QY + j) * QZ + k) * comp, s = ((src * QY + j) * QZ + k) * comp;
          for (let c = 0; c < comp; c++) arr[d + c] += arr[s + c];
        }
    }
    const pairsY = this.foldPairs(ny);
    for (const [dst, src] of pairsY) {
      for (let i = 0; i < QX; i++)
        for (let k = 0; k < QZ; k++) {
          const d = ((i * QY + dst) * QZ + k) * comp, s = ((i * QY + src) * QZ + k) * comp;
          for (let c = 0; c < comp; c++) arr[d + c] += arr[s + c];
        }
    }
    const pairsZ = this.foldPairs(nz);
    for (const [dst, src] of pairsZ) {
      for (let i = 0; i < QX; i++)
        for (let j = 0; j < QY; j++) {
          const d = ((i * QY + j) * QZ + dst) * comp, s = ((i * QY + j) * QZ + src) * comp;
          for (let c = 0; c < comp; c++) arr[d + c] += arr[s + c];
        }
    }
  }

  /** Copies the completed weights back to the ghost layers (native pass-back of Wf; z only when periodic). */
  private passBackWf(): void {
    const QX = this.QX, QY = this.QY, QZ = this.QZ;
    const nx = this.subNbx - 2, ny = this.subNby - 2, nz = this.subNbz - 2;
    this.copyPlanes(this.Wf, QY, QZ, 1, 0, this.wrapPairs(nx), 0, QY, 0, QZ);
    this.copyPlanes(this.Wf, QY, QZ, 1, 1, this.wrapPairs(ny), 0, QX, 0, QZ);
    if (!this.sys.state.box.periodic[2]) return;
    this.copyPlanes(this.Wf, QY, QZ, 1, 2, this.wrapPairs(nz), 0, QX, 0, QY);
  }

  // ---------------------------------------------------------------- hooks
  init(): void {
    const fluids = this.sys.fixes.filter((f) => f.style === 'lb/fluid');
    if (fluids.length > 1) throw new StyleError('Only one fix lb/fluid at a time is supported');
    const dtNow = this.nevery * this.sys.state.dt;
    if (Math.abs(dtNow - this.dtLb) > 1.0e-12) this.sys.warn('Timestep changed between runs.  Must re-issue fix lb/fluid command.');
    const box = this.sys.state.box;
    const wallsNow = !box.periodic[2];
    const NbxNow = lround((box.hi[0] - box.lo[0]) / this.dxLb);
    const NbyNow = lround((box.hi[1] - box.lo[1]) / this.dxLb);
    const NbzNow = lround((box.hi[2] - box.lo[2]) / this.dxLb) + (wallsNow ? 1 : 0);
    if (NbxNow !== this.Nbx || NbyNow !== this.Nby || NbzNow !== this.Nbz) {
      throw new StyleError('Simulation domain can not change shape between runs with the same lb/fluid');
    }
    if (!box.periodic[0]) throw new StyleError('the x-direction must be periodic');
    if (box.boundary[1][0] === 's' || box.boundary[1][0] === 'm') throw new StyleError('the y-direction cannot have shrink-wrap boundary conditions');
    if (box.boundary[2][0] === 's' || box.boundary[2][0] === 'm') throw new StyleError('the z-direction cannot have shrink-wrap boundary conditions');
    const visc = this.sys.fixes.filter((f) => f.style === 'lb/viscous');
    if (visc.length > 1) throw new StyleError('More than one fix lb/viscous at a time is not supported');
    this.fixviscouslb = visc.length ? 1 : 0;
  }

  /** native setup() is empty: the step-0 forces of the particles are not changed by this fix. */
  setup(): void {}

  initialIntegrate(): void {
    this.equilibriumDist();
    this.updateFull();
    const tmp = this.fLb;
    this.fLb = this.fnew;
    this.fnew = tmp;
    this.parametercalcFull();
  }

  postForce(): void {
    if (this.fixviscouslb === 1) {
      this.calcFluidForceWeight();
      this.calcFluidForceI();
    }
  }

  finalIntegrate(): void {
    this.correctUFull();
  }

  // ---------------------------------------------------------------- equilibrium and collision
  /** equilibriumdist15 over the interior nodes (fix_lb_fluid.cpp), with the noise of the fluctuating hydrodynamics. */
  private equilibriumDist(): void {
    const a0 = this.a0, tauR = this.tau - 0.5;
    const bfx = this.bodyforcex, bfy = this.bodyforcey, bfz = this.bodyforcez;
    const etacov = new Float64Array(15);
    const S0 = new Float64Array(3), S1 = new Float64Array(3);
    const feq = this.feq;
    for (let i = 1; i < this.subNbx - 1; i++)
      for (let j = 1; j < this.subNby - 1; j++)
        for (let k = 1; k < this.subNbz - 1; k++) {
          if (this.stype[this.fi(i, j, k)] === 2) continue; // outside the domain: feq stays zero, no noise drawn
          const q = this.qi(i, j, k);
          const fidx = this.fi(i, j, k) * NV;
          const rho = this.density[q];
          const p0 = rho * a0;
          const u0 = this.u[3 * q], u1 = this.u[3 * q + 1], u2 = this.u[3 * q + 2];
          const Fx = this.Ff[3 * q], Fy = this.Ff[3 * q + 1], Fz = this.Ff[3 * q + 2];
          etacov[0] = rho;
          etacov[1] = rho * u0 + (Fx + rho * bfx) * tauR;
          etacov[2] = rho * u1 + (Fy + rho * bfy) * tauR;
          etacov[3] = rho * u2 + (Fz + rho * bfz) * tauR;
          etacov[4] = p0 + rho * u0 * u0 - rho / 3.0 + tauR * (2.0 * u0 * (Fx + rho * bfx));
          etacov[5] = p0 + rho * u1 * u1 - rho / 3.0 + tauR * (2.0 * u1 * (Fy + rho * bfy));
          etacov[6] = p0 + rho * u2 * u2 - rho / 3.0 + tauR * (2.0 * u2 * (Fz + rho * bfz));
          etacov[7] = rho * u0 * u1 + tauR * (u0 * (Fy + rho * bfy) + (Fx + rho * bfx) * u1);
          etacov[8] = rho * u1 * u2 + tauR * (u1 * (Fz + rho * bfz) + (Fy + rho * bfy) * u2);
          etacov[9] = rho * u0 * u2 + tauR * (u0 * (Fz + rho * bfz) + (Fx + rho * bfx) * u2);
          etacov[10] = 0.0;
          etacov[11] = 0.0;
          etacov[12] = 0.0;
          etacov[13] = rho * u0 * u1 * u2;
          etacov[14] = this.K0 * rho * (1.0 - 3.0 * a0);
          for (let l = 0; l < NV; l++) {
            let acc = 0.0;
            for (let ii = 0; ii < NV; ii++) acc += WM15[ii * NV + l] * etacov[ii] * NG15F[ii];
            feq[fidx + l] = acc;
          }
          if (this.noisestress === 1 && this.random) {
            const rnd = this.random;
            const stdv = Math.sqrt(this.namp * rho);
            for (let jj = 0; jj < 3; jj++) S0[jj] = stdv * rnd.gaussian();
            for (let jj = 0; jj < 3; jj++) S1[jj] = stdv * rnd.gaussian();
            etacov[4] = S0[0] * Math.sqrt(3.0 - 3.0 * a0);
            etacov[5] = ((1.0 - 3.0 * a0) * S0[0]) / Math.sqrt(3.0 - 3.0 * a0) + Math.sqrt((8.0 - 12.0 * a0) / (3.0 - 3.0 * a0)) * S0[1];
            etacov[6] =
              ((1.0 - 3.0 * a0) * S0[0]) / Math.sqrt(3.0 - 3.0 * a0) +
              ((2.0 - 6.0 * a0) * S0[1]) / Math.sqrt((8.0 - 12.0 * a0) * (3.0 - 3.0 * a0)) +
              Math.sqrt((5.0 - 9.0 * a0) / (2.0 - 3.0 * a0)) * S0[2];
            etacov[7] = S1[0];
            etacov[8] = S1[1];
            etacov[9] = S1[2];
            for (let l = 10; l < 15; l++) etacov[l] = Math.sqrt((9.0 * this.namp * rho) / NG15[l]) * rnd.gaussian();
            etacov[14] += -this.K0 * (etacov[4] + etacov[5] + etacov[6]);
            for (let l = 0; l < NV; l++) {
              const ghostnoise =
                W15[l] * (MG15[4 * NV + l] * etacov[4] * NG15[4] + MG15[5 * NV + l] * etacov[5] * NG15[5] +
                  MG15[6 * NV + l] * etacov[6] * NG15[6] + MG15[7 * NV + l] * etacov[7] * NG15[7] +
                  MG15[8 * NV + l] * etacov[8] * NG15[8] + MG15[9 * NV + l] * etacov[9] * NG15[9] +
                  MG15[10 * NV + l] * etacov[10] * NG15[10] + MG15[11 * NV + l] * etacov[11] * NG15[11] +
                  MG15[12 * NV + l] * etacov[12] * NG15[12] + MG15[13 * NV + l] * etacov[13] * NG15[13] +
                  MG15[14 * NV + l] * etacov[14] * NG15[14]);
              feq[fidx + l] += ghostnoise * this.noisefactor;
            }
          }
        }
  }

  /** update_full15(): exchange feq ghosts, stream and bounce back, pressure boundary. Result in fnew. */
  private updateFull(): void {
    const nx = this.subNbx - 2, ny = this.subNby - 2, nz = this.subNbz - 2;
    const sx = this.subNbx, sy = this.subNby, sz = this.subNbz;
    // feq exchange (f-grid, NV components): x, then y (interior k), then z (all i, j)
    this.copyPlanesF(this.feq, 0, this.lowerUpperPairs(nx), 1, sy - 1, 1, sz - 1);
    this.copyPlanesF(this.feq, 1, this.lowerUpperPairs(ny), 0, sx, 1, sz - 1);
    this.copyPlanesF(this.feq, 2, this.lowerUpperPairs(nz), 0, sx, 0, sy);
    this.updatePeriodic(1, sx - 1, 1, sy - 1, 1, sz - 1);
    if (this.pressure === 1) this.applyPressureBC();
  }

  /** The feq / f exchange on the f-grid (NV components per node). */
  private copyPlanesF(arr: Float64Array, dim: 0 | 1 | 2, pairs: readonly (readonly [number, number])[], a0: number, a1: number, b0: number, b1: number): void {
    this.copyPlanes(arr, this.subNby, this.subNbz, NV, dim, pairs, a0, a1, b0, b1);
  }

  /** update_periodic() for types 0 (bulk) and 3 (flat wall) with D3Q15. */
  private updatePeriodic(i0: number, i1: number, j0: number, j1: number, k0: number, k1: number): void {
    const tau = this.tau;
    const f = this.fLb, fn = this.fnew, feq = this.feq;
    for (let i = i0; i < i1; i++)
      for (let j = j0; j < j1; j++)
        for (let k = k0; k < k1; k++) {
          const node = this.fi(i, j, k);
          const type = this.stype[node];
          const ori = this.sori[node];
          const base = node * NV;
          if (type === 0) {
            for (let m = 0; m < NV; m++) {
              const im = i - E15[m][0], jm = j - E15[m][1], km = k - E15[m][2];
              const src = this.fi(im, jm, km) * NV + m;
              fn[base + m] = f[src] + (feq[src] - f[src]) / tau;
            }
          } else if (type === 1) {
            // pit geometry boundary fluid nodes (update_periodic, native): bounce back, then normal propagation
            const row = BBL[ori];
            const nb = row[0];
            fn[base] = f[base] - (f[base] - feq[base]) / tau;
            for (let nbb = 1; nbb <= nb; nbb++) {
              const ll = row[nbb];
              const o = base + OD[ll];
              fn[base + ll] = f[o] - (f[o] - feq[o]) / tau;
            }
            for (let nbb = nb + 1; nbb <= 15; nbb++) {
              const ll = row[nbb];
              const im = i - E15[ll][0], jm = j - E15[ll][1], km = k - E15[ll][2];
              const src = this.fi(im, jm, km) * NV + ll;
              fn[base + ll] = f[src] - (f[src] - feq[src]) / tau;
            }
          } else if (type === 3) {
            // normal propagation directions (includes on-wall which will be modified)
            fn[base + 0] = f[base + 0] - (f[base + 0] - feq[base + 0]) / tau;
            const bbl = ori === 3 ? BBL_WALL_BOTTOM : BBL_WALL_TOP;
            for (let nbb = bbl[0] + 1; nbb <= 15; nbb++) {
              const ll = bbl[nbb];
              const im = i - E15[ll][0], jm = j - E15[ll][1], km = k - E15[ll][2];
              const src = this.fi(im, jm, km) * NV + ll;
              fn[base + ll] = f[src] - (f[src] - feq[src]) / tau;
            }
            if (ori === 3) this.bottomWall(base, tau, i, j, k);
            else this.topWall(base, tau, i, j, k);
          }
        }
  }

  /** Bottom flat wall (orientation 3), modified return of the populations. */
  private bottomWall(base: number, tau: number, i: number, j: number, k: number): void {
    const f = this.fLb, fn = this.fnew, feq = this.feq;
    const q = this.qi(i, j, k);
    const rho = this.density[q];
    const vw = this.vwbt;
    const F = (m: number) => fn[base + m];
    const rb = F(0) + F(1) + F(2) + F(3) + F(4) + F(6) + F(11) + F(12) + F(13) + F(14) +
      f[base + 6] + (feq[base + 6] - f[base + 6]) / tau + f[base + 11] +
      (feq[base + 11] - f[base + 11]) / tau + f[base + 12] +
      (feq[base + 12] - f[base + 12]) / tau + f[base + 13] +
      (feq[base + 13] - f[base + 13]) / tau + f[base + 14] +
      (feq[base + 14] - f[base + 14]) / tau;
    const rbvw = rb * vw;
    const rvw2 = rbvw * vw;
    const seqyy = rb * this.a0 + rvw2;
    const r = rb / rho;
    fn[base + 2] = r * feq[base + 2];
    fn[base + 4] = r * feq[base + 4];
    fn[base + 0] = rb - 2.0 * (F(6) + F(11) + F(12) + F(13) + F(14) + F(2) + F(4)) + rvw2;
    fn[base + 1] = 0.5 * (F(2) + F(4) - rvw2);
    fn[base + 3] = fn[base + 1];
    fn[base + 5] = F(6) + 2.0 * (F(11) + F(12) + F(13) + F(14)) + F(2) + F(4) - seqyy;
    fn[base + 7] = 0.25 * (rbvw + seqyy - 2.0 * F(2)) - F(11);
    fn[base + 8] = 0.25 * (rbvw + seqyy - 2.0 * F(2)) - F(12);
    fn[base + 9] = 0.25 * (-rbvw + seqyy - 2.0 * F(4)) - F(13);
    fn[base + 10] = 0.25 * (-rbvw + seqyy - 2.0 * F(4)) - F(14);
  }

  /** Top flat wall (orientation 16), modified return of the populations. */
  private topWall(base: number, tau: number, i: number, j: number, k: number): void {
    const f = this.fLb, fn = this.fnew, feq = this.feq;
    const q = this.qi(i, j, k);
    const rho = this.density[q];
    const vw = this.vwtp;
    const F = (m: number) => fn[base + m];
    const rb = F(0) + F(1) + F(2) + F(3) + F(4) + F(5) + F(7) + F(8) + F(9) + F(10) +
      f[base + 5] + (feq[base + 5] - f[base + 5]) / tau + f[base + 7] +
      (feq[base + 7] - f[base + 7]) / tau + f[base + 8] +
      (feq[base + 8] - f[base + 8]) / tau + f[base + 9] +
      (feq[base + 9] - f[base + 9]) / tau + f[base + 10] +
      (feq[base + 10] - f[base + 10]) / tau;
    const rbvw = rb * vw;
    const rvw2 = rbvw * vw;
    const seqyy = rb * this.a0 + rvw2;
    const r = rb / rho;
    fn[base + 2] = r * feq[base + 2];
    fn[base + 4] = r * feq[base + 4];
    fn[base + 0] = rb - 2.0 * (F(5) + F(7) + F(8) + F(9) + F(10) + F(2) + F(4)) + rvw2;
    fn[base + 1] = 0.5 * (F(2) + F(4) - rvw2);
    fn[base + 3] = fn[base + 1];
    fn[base + 6] = F(5) + 2.0 * (F(7) + F(8) + F(9) + F(10)) + F(2) + F(4) - seqyy;
    fn[base + 11] = 0.25 * (rbvw + seqyy - 2.0 * F(2)) - F(7);
    fn[base + 12] = 0.25 * (rbvw + seqyy - 2.0 * F(2)) - F(8);
    fn[base + 13] = 0.25 * (-rbvw + seqyy - 2.0 * F(4)) - F(9);
    fn[base + 14] = 0.25 * (-rbvw + seqyy - 2.0 * F(4)) - F(10);
  }

  /** Pressure boundary in x (update_full15): the populations leaving the x ends are rescaled. */
  private applyPressureBC(): void {
    const fn = this.fnew;
    const N = this.subNbx - 2;
    const sy = this.subNby, sz = this.subNbz;
    const rd = (i: number, j: number, k: number) => this.density[this.qi(i, j, k)];
    for (let j = 1; j < sy - 1; j++) {
      const jup = j + 1, jdwn = j - 1;
      for (let k = 1; k < sz - 1; k++) {
        const kup = k + 1, kdwn = k - 1;
        if (this.stype[this.fi(1, j, k)] === 0) {
          const b = this.fi(1, j, k) * NV;
          fn[b + 1] = (fn[b + 1] * this.rhoH) / rd(0, j, k);
          fn[b + 7] = (fn[b + 7] * this.rhoH) / rd(0, jdwn, kdwn);
          fn[b + 10] = (fn[b + 10] * this.rhoH) / rd(0, jup, kdwn);
          fn[b + 11] = (fn[b + 11] * this.rhoH) / rd(0, jdwn, kup);
          fn[b + 14] = (fn[b + 14] * this.rhoH) / rd(0, jup, kup);
        }
      }
    }
    for (let j = 1; j < sy - 1; j++) {
      const jup = j + 1, jdwn = j - 1;
      for (let k = 1; k < sz - 1; k++) {
        const kup = k + 1, kdwn = k - 1;
        if (this.stype[this.fi(N, j, k)] === 0) {
          const b = this.fi(N, j, k) * NV;
          fn[b + 3] = (fn[b + 3] * this.rhoL) / rd(N + 1, j, k);
          fn[b + 8] = (fn[b + 8] * this.rhoL) / rd(N + 1, jdwn, kdwn);
          fn[b + 9] = (fn[b + 9] * this.rhoL) / rd(N + 1, jup, kdwn);
          fn[b + 12] = (fn[b + 12] * this.rhoL) / rd(N + 1, jdwn, kup);
          fn[b + 13] = (fn[b + 13] * this.rhoL) / rd(N + 1, jup, kup);
        }
      }
    }
  }

  // ---------------------------------------------------------------- moments
  /** parametercalc_full(): exchange f ghosts, then density and velocity on the whole grid, then the u/rho ghosts. */
  private parametercalcFull(): void {
    const nx = this.subNbx - 2, ny = this.subNby - 2, nz = this.subNbz - 2;
    const sx = this.subNbx, sy = this.subNby, sz = this.subNbz;
    const QX = this.QX, QY = this.QY, QZ = this.QZ;
    this.copyPlanesF(this.fLb, 0, this.lowerUpperPairs(nx), 1, sy - 1, 1, sz - 1);
    this.copyPlanesF(this.fLb, 1, this.lowerUpperPairs(ny), 0, sx, 1, sz - 1);
    this.copyPlanesF(this.fLb, 2, this.lowerUpperPairs(nz), 0, sx, 0, sy);
    for (let i = 0; i < sx; i++)
      for (let j = 0; j < sy; j++)
        for (let k = 0; k < sz; k++) this.parametercalcNode(i, j, k);
    // density and velocity ghosts that native copies from the opposite side of the block
    this.copyPlanes(this.density, QY, QZ, 1, 0, this.outerPairs(nx), 0, QY, 0, QZ);
    this.copyPlanes(this.density, QY, QZ, 1, 1, this.outerPairs(ny), 0, QX, 0, QZ);
    this.copyPlanes(this.u, QY, QZ, 3, 0, this.outerPairs(nx), 0, QY, 0, QZ);
    this.copyPlanes(this.u, QY, QZ, 3, 1, this.outerPairs(ny), 0, QX, 0, QZ);
    if (this.sys.state.box.periodic[2]) {
      this.copyPlanes(this.density, QY, QZ, 1, 2, this.outerPairs(nz), 0, QX, 0, QY);
      this.copyPlanes(this.u, QY, QZ, 3, 2, this.outerPairs(nz), 0, QX, 0, QY);
    }
  }

  /** parametercalc_part() for one node: density and mass-weighted velocity from the populations. */
  private parametercalcNode(i: number, j: number, k: number): void {
    const q = this.qi(i, j, k);
    if (this.stype[this.fi(i, j, k)] === 2) {
      // type 2 is outside the domain (native parametercalc_part)
      this.density[q] = 0.0;
      this.u[3 * q] = 0.0;
      this.u[3 * q + 1] = 0.0;
      this.u[3 * q + 2] = 0.0;
      return;
    }
    const f = this.fi(i, j, k) * NV;
    let rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
    for (let m = 0; m < NV; m++) {
      rho += this.fLb[f + m];
      ux += this.fLb[f + m] * E15[m][0];
      uy += this.fLb[f + m] * E15[m][1];
      uz += this.fLb[f + m] * E15[m][2];
    }
    this.density[q] = rho;
    this.u[3 * q] = ux / rho;
    this.u[3 * q + 1] = uy / rho;
    this.u[3 * q + 2] = uz / rho;
  }

  // ---------------------------------------------------------------- atom coupling
  /** Native order of the atoms in this fix's group. */
  private groupAtoms(): number[] {
    const s = this.sys.state;
    const order = nativeOrder(s);
    const out: number[] = [];
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (s.mask[i] & this.groupBit) out.push(i);
    }
    return out;
  }

  /** Per-atom storage by ID (grown on demand; lb/viscous reads hydroF through this as well). */
  slot(id: number): void {
    if (id < this.massp.length) return;
    const n = Math.max(id + 1, this.massp.length * 2);
    const m = new Float64Array(n);
    m.set(this.massp);
    this.massp = m;
    const h = new Float64Array(3 * n);
    h.set(this.hydroF);
    this.hydroF = h;
  }

  /** calc_fluidforceweight(): the interpolation weights of the group's atoms, folded and passed back. */
  private calcFluidForceWeight(): void {
    const s = this.sys.state;
    this.Wf.fill(0);
    for (const i of this.groupAtoms()) this.interpolationWeight(s.x, i);
    this.foldGrid(this.Wf, 1);
    this.passBackWf();
  }

  /** calc_fluidforceI(): spread the particle forces onto the fluid, then fold the ghost layers. */
  private calcFluidForceI(): void {
    this.Ff.fill(0);
    this.dofLb = 0.0;
    for (const i of this.groupAtoms()) this.interpolate(i);
    this.foldGrid(this.Ff, 3);
  }

  /** Stencil of an atom: the lower-left node and the fractional offsets, as native computes them. */
  private stencilBase(x: Float64Array, i: number): { ix: number; iy: number; iz: number; dx1: number; dy1: number; dz1: number } {
    const lo = this.sys.state.box.lo;
    const dx = this.dxLb;
    const ix = Math.ceil((x[3 * i] - lo[0]) / dx);
    const iy = Math.ceil((x[3 * i + 1] - lo[1]) / dx);
    const iz = Math.ceil((x[3 * i + 2] - lo[2]) / dx);
    let dx1 = x[3 * i] - (lo[0] + (ix - 1) * dx);
    let dy1 = x[3 * i + 1] - (lo[1] + (iy - 1) * dx);
    let dz1 = x[3 * i + 2] - (lo[2] + (iz - 1) * dx);
    dx1 = dx1 / dx;
    dy1 = dy1 / dx;
    dz1 = dz1 / dx;
    return { ix, iy, iz, dx1, dy1, dz1 };
  }

  /** Trilinear / IBM3 weight of one atom, added to Wf (interpolationweight). */
  private interpolationWeight(x: Float64Array, i: number): void {
    const { ix, iy, iz, dx1, dy1, dz1 } = this.stencilBase(x, i);
    if (this.nStencil === 3) {
      this.ibm3Weight(ix, iy, iz, dx1, dy1, dz1);
      return;
    }
    this.checkInside(ix, iy, iz, 'trilinear');
    const FfP = [
      (1.0 - dx1) * (1.0 - dy1) * (1.0 - dz1), (1.0 - dx1) * (1.0 - dy1) * dz1,
      (1.0 - dx1) * dy1 * (1.0 - dz1), (1.0 - dx1) * dy1 * dz1,
      dx1 * (1.0 - dy1) * (1.0 - dz1), dx1 * (1.0 - dy1) * dz1,
      dx1 * dy1 * (1.0 - dz1), dx1 * dy1 * dz1,
    ];
    const ixp = ix + 1, iyp = iy + 1, izp = iz + 1;
    this.Wf[this.qi(ix, iy, iz)] += FfP[0];
    this.Wf[this.qi(ix, iy, izp)] += FfP[1];
    this.Wf[this.qi(ix, iyp, iz)] += FfP[2];
    this.Wf[this.qi(ix, iyp, izp)] += FfP[3];
    this.Wf[this.qi(ixp, iy, iz)] += FfP[4];
    this.Wf[this.qi(ixp, iy, izp)] += FfP[5];
    this.Wf[this.qi(ixp, iyp, iz)] += FfP[6];
    this.Wf[this.qi(ixp, iyp, izp)] += FfP[7];
  }

  /** The atom must lie within one grid point outside the block (native error message). */
  private checkInside(ix: number, iy: number, iz: number, kind: string): void {
    const bad = kind === 'trilinear'
      ? ix < 0 || ix + 1 > this.subNbx + 1 || iy < 0 || iy + 1 > this.subNby + 1 || iz < 0 || iz + 1 > this.subNbz + 1
      : ix < 0 || ix > this.subNbx - 1 || iy < 0 || iy > this.subNby - 1 || iz < 0 || iz > this.subNbz - 1;
    if (bad) {
      throw new StyleError(
        kind === 'trilinear'
          ? 'Atom outside local processor simulation domain.  Either unstable fluid pararmeters, or require more frequent neighborlist rebuilds'
          : 'Atom outside local processor simulation domain!  Either unstable fluid pararmeters or require more frequent neighborlist rebuilds',
      );
    }
  }

  /** IBM 3-point weights along one direction (fix_lb_fluid.cpp, IBM3_interpolationweight). */
  private static ibmWeight(d: number): number {
    const r = Math.abs(d);
    if (r >= 1.5) return 0.0;
    if (r > 0.5) return (5.0 - 3.0 * r - Math.sqrt(1.0 - 3.0 * (1.0 - r) * (1.0 - r))) / 6.0;
    return (1.0 + Math.sqrt(1.0 - 3.0 * r * r)) / 3;
  }

  private ibm3Weight(ix: number, iy: number, iz: number, dx1: number, dy1: number, dz1: number): void {
    if (ix < 0 || ix > this.subNbx - 1 || iy < 0 || iy > this.subNby - 1 || iz < 0 || iz > this.subNbz - 1) {
      throw new StyleError('Atom outside local processor simulation domain!  Either unstable fluid pararmeters or require more frequent neighborlist rebuilds');
    }
    const imin = dx1 < 0.5 ? -1 : 0, jmin = dy1 < 0.5 ? -1 : 0, kmin = dz1 < 0.5 ? -1 : 0;
    for (let ii = imin; ii < 3 + imin; ii++) {
      const wx = FixLbFluid.ibmWeight(-dx1 + ii);
      for (let jj = jmin; jj < 3 + jmin; jj++) {
        const wy = FixLbFluid.ibmWeight(-dy1 + jj);
        for (let kk = kmin; kk < 3 + kmin; kk++) {
          const wz = FixLbFluid.ibmWeight(-dz1 + kk);
          let ixp = ix + ii, iyp = iy + jj, izp = iz + kk;
          if (ixp === -1) ixp = this.subNbx + 2;
          if (iyp === -1) iyp = this.subNby + 2;
          if (izp === -1) izp = this.subNbz + 2;
          this.Wf[this.qi(ixp, iyp, izp)] += wx * wy * wz;
        }
      }
    }
  }

  /** Force spreading and the atom's hydrodynamic force for one atom (trilinear or IBM3). */
  private interpolate(i: number): void {
    const s = this.sys.state;
    const x = s.x, v = s.v, f = s.f;
    const { ix, iy, iz, dx1, dy1, dz1 } = this.stencilBase(x, i);
    const dt = this.dtLb, dx = this.dxLb, dm = this.dmLb;
    const type = s.type[i];
    const gamma = this.Gamma[type];
    const massone0 = massOf(s, i);
    this.slot(s.id[i]);
    const id = s.id[i];
    const hyd = this.hydroF;
    const mp = this.massp;

    if (this.nStencil === 3) {
      this.checkInside(ix, iy, iz, 'ibm');
      const imin = dx1 < 0.5 ? -1 : 0, jmin = dy1 < 0.5 ? -1 : 0, kmin = dz1 < 0.5 ? -1 : 0;
      const nten = 27;
      const FfP = new Float64Array(nten), TfP = new Float64Array(nten);
      const qidx = new Int32Array(nten);
      const unew = [0.0, 0.0, 0.0];
      let mnode = 0.0, myweight = 0.0, myweightsq = 0.0;
      let isten = 0;
      for (let ii = imin; ii < 3 + imin; ii++) {
        const wx = FixLbFluid.ibmWeight(-dx1 + ii);
        for (let jj = jmin; jj < 3 + jmin; jj++) {
          const wy = FixLbFluid.ibmWeight(-dy1 + jj);
          for (let kk = kmin; kk < 3 + kmin; kk++) {
            const wz = FixLbFluid.ibmWeight(-dz1 + kk);
            let ixp = ix + ii, iyp = iy + jj, izp = iz + kk;
            if (ixp === -1) ixp = this.subNbx + 2;
            if (iyp === -1) iyp = this.subNby + 2;
            if (izp === -1) izp = this.subNbz + 2;
            const q = this.qi(ixp, iyp, izp);
            qidx[isten] = q;
            FfP[isten] = wx * wy * wz;
            if (Math.abs(FfP[isten]) > 1e-14) TfP[isten] = (FfP[isten] * FfP[isten]) / this.Wf[q];
            else TfP[isten] = 0.0;
            for (let k = 0; k < 3; k++) unew[k] += this.density[q] * this.u[3 * q + k] * FfP[isten];
            mnode += this.density[q] * FfP[isten];
            myweight += TfP[isten];
            myweightsq += TfP[isten] * FfP[isten];
            isten++;
          }
        }
      }
      for (let k = 0; k < 3; k++) unew[k] /= mnode;
      this.dofLb += myweight;
      mnode = (mnode * myweight * myweight) / myweightsq;
      mp[id] = mnode * dm;
      const massone = massone0 / dm;
      const hF = [0, 0, 0], Ffull = [0, 0, 0];
      if (gamma < 0) {
        const gammavalue = Math.abs(gamma) * 2.0 * mnode;
        for (let k = 0; k < 3; k++) {
          hF[k] = gammavalue * ((v[3 * i + k] * dt) / dx - unew[k]);
          hyd[3 * id + k] = 0.0;
          Ffull[k] = hF[k] / myweight;
        }
      } else {
        const gammavalue = gamma * 2.0 * ((mnode * massone) / (mnode + massone));
        for (let k = 0; k < 3; k++) {
          hF[k] = gammavalue * ((v[3 * i + k] * dt) / dx - unew[k]);
          hyd[3 * id + k] = (-hF[k] * dm * dx) / dt / dt;
          Ffull[k] = (hF[k] + ((f[3 * i + k] * dt * dt) / dm / dx) * mnode / (mnode + massone)) / myweight;
        }
      }
      for (let n = 0; n < nten; n++)
        for (let k = 0; k < 3; k++) this.Ff[3 * qidx[n] + k] += Ffull[k] * TfP[n];
      return;
    }

    // trilinear (stencil 2 and any other value, as native's default branch)
    this.checkInside(ix, iy, iz, 'trilinear');
    const FfP = [
      (1.0 - dx1) * (1.0 - dy1) * (1.0 - dz1), (1.0 - dx1) * (1.0 - dy1) * dz1,
      (1.0 - dx1) * dy1 * (1.0 - dz1), (1.0 - dx1) * dy1 * dz1,
      dx1 * (1.0 - dy1) * (1.0 - dz1), dx1 * (1.0 - dy1) * dz1,
      dx1 * dy1 * (1.0 - dz1), dx1 * dy1 * dz1,
    ];
    const ixp = ix + 1, iyp = iy + 1, izp = iz + 1;
    const qn = [
      this.qi(ix, iy, iz), this.qi(ix, iy, izp), this.qi(ix, iyp, iz), this.qi(ix, iyp, izp),
      this.qi(ixp, iy, iz), this.qi(ixp, iy, izp), this.qi(ixp, iyp, iz), this.qi(ixp, iyp, izp),
    ];
    const d = this.density, u = this.u;
    let mnode =
      d[qn[0]] * FfP[0] + d[qn[1]] * FfP[1] + d[qn[2]] * FfP[2] + d[qn[3]] * FfP[3] +
      d[qn[4]] * FfP[4] + d[qn[5]] * FfP[5] + d[qn[6]] * FfP[6] + d[qn[7]] * FfP[7];
    const unew = [0.0, 0.0, 0.0];
    for (let k = 0; k < 3; k++) {
      unew[k] =
        (d[qn[0]] * u[3 * qn[0] + k] * FfP[0] + d[qn[1]] * u[3 * qn[1] + k] * FfP[1] +
          d[qn[2]] * u[3 * qn[2] + k] * FfP[2] + d[qn[3]] * u[3 * qn[3] + k] * FfP[3] +
          d[qn[4]] * u[3 * qn[4] + k] * FfP[4] + d[qn[5]] * u[3 * qn[5] + k] * FfP[5] +
          d[qn[6]] * u[3 * qn[6] + k] * FfP[6] + d[qn[7]] * u[3 * qn[7] + k] * FfP[7]) / mnode;
    }
    const TfP = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let n = 0; n < 8; n++) if (FfP[n] > 0) TfP[n] = (FfP[n] * FfP[n]) / this.Wf[qn[n]];
    const myweight = TfP[0] + TfP[1] + TfP[2] + TfP[3] + TfP[4] + TfP[5] + TfP[6] + TfP[7];
    const myweightsq = TfP[0] * FfP[0] + TfP[1] * FfP[1] + TfP[2] * FfP[2] + TfP[3] * FfP[3] +
      TfP[4] * FfP[4] + TfP[5] * FfP[5] + TfP[6] * FfP[6] + TfP[7] * FfP[7];
    this.dofLb += myweight;
    mnode = (mnode * myweight * myweight) / myweightsq;
    mp[id] = mnode * dm;
    const massone = massone0 / dm;
    for (let k = 0; k < 3; k++) {
      const hFk = (gamma < 0 ? Math.abs(gamma) * 2.0 * mnode : gamma * 2.0 * ((mnode * massone) / (mnode + massone))) *
        ((v[3 * i + k] * dt) / dx - unew[k]);
      let Fp = 0;
      if (gamma < 0) {
        hyd[3 * id + k] = 0.0;
      } else {
        hyd[3 * id + k] = (-hFk * dm * dx) / dt / dt;
        Fp = ((f[3 * i + k] * dt * dt) / dm / dx) * mnode / (mnode + massone);
      }
      const Ffull = (hFk + Fp) / myweight;
      this.Ff[3 * qn[0] + k] += Ffull * TfP[0];
      this.Ff[3 * qn[1] + k] += Ffull * TfP[1];
      this.Ff[3 * qn[2] + k] += Ffull * TfP[2];
      this.Ff[3 * qn[3] + k] += Ffull * TfP[3];
      this.Ff[3 * qn[4] + k] += Ffull * TfP[4];
      this.Ff[3 * qn[5] + k] += Ffull * TfP[5];
      this.Ff[3 * qn[6] + k] += Ffull * TfP[6];
      this.Ff[3 * qn[7] + k] += Ffull * TfP[7];
    }
  }

  // ---------------------------------------------------------------- velocity correction
  /** correctu_full(): add half the coupling force to the velocity of the bulk nodes, then exchange ghosts. */
  private correctUFull(): void {
    const sx = this.subNbx, sy = this.subNby, sz = this.subNbz;
    const nx = sx - 2, ny = sy - 2, nz = sz - 2;
    for (let i = 1; i < sx - 1; i++)
      for (let j = 1; j < sy - 1; j++)
        for (let k = 1; k < sz - 1; k++) {
          if (this.stype[this.fi(i, j, k)] !== 0) continue;
          const q = this.qi(i, j, k);
          const rho = this.density[q];
          this.u[3 * q] += 0.5 * (this.Ff[3 * q] / rho + this.bodyforcex);
          this.u[3 * q + 1] += 0.5 * (this.Ff[3 * q + 1] / rho + this.bodyforcey);
          this.u[3 * q + 2] += 0.5 * (this.Ff[3 * q + 2] / rho + this.bodyforcez);
        }
    const QX = this.QX, QY = this.QY, QZ = this.QZ;
    this.copyPlanes(this.u, QY, QZ, 3, 0, this.wrapPairs(nx), 0, QY, 0, QZ);
    this.copyPlanes(this.u, QY, QZ, 3, 1, this.wrapPairs(ny), 0, QX, 0, QZ);
    if (this.sys.state.box.periodic[2]) this.copyPlanes(this.u, QY, QZ, 3, 2, this.wrapPairs(nz), 0, QX, 0, QY);
  }

  // ---------------------------------------------------------------- outputs and fields for the companion fixes
  /** Density on the density grid (index (i * QY + j) * QZ + k). */
  densityField(): Float64Array {
    return this.density;
  }
  /** Mass-weighted velocity on the density grid, 3 per node. */
  velocityField(): Float64Array {
    return this.u;
  }
  /** Populations on the f-grid, NV per node (the current ones, after the last swap). */
  populationField(): Float64Array {
    return this.fLb;
  }

  /** calc_mass_momentum(): total fluid mass and momentum in LAMMPS units (used by fix lb/momentum). */
  massMomentum(): { mass: number; momentum: [number, number, number] } {
    let m = 0.0;
    const p = [0.0, 0.0, 0.0];
    for (let i = 1; i < this.subNbx - 1; i++)
      for (let j = 1; j < this.subNby - 1; j++)
        for (let k = 1; k < this.subNbz - 1; k++) {
          const q = this.qi(i, j, k);
          m += this.density[q];
          p[0] += this.density[q] * this.u[3 * q];
          p[1] += this.density[q] * this.u[3 * q + 1];
          p[2] += this.density[q] * this.u[3 * q + 2];
        }
    const conv = (this.dmLb * this.dxLb) / this.dtLb;
    return { mass: m * this.dmLb, momentum: [p[0] * conv, p[1] * conv, p[2] * conv] };
  }

  /** calc_MPT(): fluid mass, momentum and the temperature of the fluid (f_ID[1..4] and the vector output). */
  private calcMPT(): { mass: number; momentum: [number, number, number]; Tave: number } {
    let m = 0.0;
    const p = [0.0, 0.0, 0.0];
    let tave = 0.0;
    for (let i = 1; i < this.subNbx - 1; i++)
      for (let j = 1; j < this.subNby - 1; j++)
        for (let k = 1; k < this.subNbz - 1; k++) {
          const q = this.qi(i, j, k);
          m += this.density[q];
          p[0] += this.density[q] * this.u[3 * q];
          p[1] += this.density[q] * this.u[3 * q + 1];
          p[2] += this.density[q] * this.u[3 * q + 2];
          tave += (this.u[3 * q] * this.u[3 * q] + this.u[3 * q + 1] * this.u[3 * q + 1] + this.u[3 * q + 2] * this.u[3 * q + 2]) * this.density[q];
        }
    const conv = (this.dmLb * this.dxLb) / this.dtLb;
    const { boltz, mvv2e } = this.sys.state.units;
    const kB = ((boltz / mvv2e) * this.dtLb * this.dtLb) / this.dxLb / this.dxLb / this.dmLb;
    tave = tave / (3.0 * this.Nbx * this.Nby * this.Nbz * kB);
    return { mass: m * this.dmLb, momentum: [p[0] * conv, p[1] * conv, p[2] * conv], Tave: tave };
  }

  /** dof_compute(): the temperature factor of the particles (fix_lb_fluid.cpp). */
  private dofCompute(): number {
    let dof: number;
    if (this.setdof) dof = this.setdof;
    else dof = 3.0 * this.dofLb;
    const { boltz, mvv2e } = this.sys.state.units;
    return dof > FLT_EPSILON ? mvv2e / (dof * boltz) : 0.0;
  }

  /** compute_scalar(): kinetic temperature of the particles with their fluid mass (f_ID). */
  computeScalar(): number {
    const s = this.sys.state;
    let t = 0.0;
    for (const i of this.groupAtoms()) {
      const vv = s.v[3 * i] * s.v[3 * i] + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2];
      t += vv * (massOf(s, i) + this.massp[s.id[i]]);
    }
    return t * this.dofCompute();
  }

  private vectorCache: { step: number; mass: number; momentum: [number, number, number]; Tave: number } | null = null;

  /** compute_vector(): f_ID[1] = Tave, [2] total mass, [3..5] total momentum (particles + fluid). */
  computeVector(n: number): number {
    const step = this.sys.state.step;
    if (!this.vectorCache || this.vectorCache.step !== step) {
      const fl = this.calcMPT();
      const s = this.sys.state;
      let pm = 0.0;
      const pv = [0.0, 0.0, 0.0];
      for (let i = 0; i < s.n; i++) if (s.mask[i] & 1) pm += massOf(s, i);
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & 1)) continue;
        const mi = massOf(s, i);
        pv[0] += s.v[3 * i] * mi;
        pv[1] += s.v[3 * i + 1] * mi;
        pv[2] += s.v[3 * i + 2] * mi;
      }
      const pcm = pm > 0 ? [pv[0] / pm, pv[1] / pm, pv[2] / pm] : [0, 0, 0];
      this.vectorCache = {
        step,
        mass: fl.mass + pm,
        momentum: [pm * pcm[0] + fl.momentum[0], pm * pcm[1] + fl.momentum[1], pm * pcm[2] + fl.momentum[2]],
        Tave: fl.Tave,
      };
    }
    const c = this.vectorCache;
    switch (n) {
      case 0: return c.Tave;
      case 1: return c.mass;
      case 2: return c.momentum[0];
      case 3: return c.momentum[1];
      case 4: return c.momentum[2];
      default: throw new StyleError('No fix lb/fluid variable available for that index.');
    }
  }
}
