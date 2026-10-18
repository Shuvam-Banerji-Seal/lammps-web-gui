/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MC/pair_dsmc.cpp and pair_dsmc.h
 * (the LAMMPS developers; Copyright (2003) Sandia Corporation, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * pair_style dsmc (docs.lammps.org/pair_dsmc.html): direct simulation Monte Carlo collisions between the atoms of each
 * cell of a uniform grid, with the variable hard sphere cross section and random scattering of the relative velocity.
 * "pair_style dsmc max_cell_size seed weighting Tref Nrecompute Nsample". The collisions change the velocities inside
 * the force evaluation, as native does; the random draws (RanMars, seed) follow the native order of the cells, the
 * types and the atoms (native local order, nativeOrder). The style has no energy and no virial.
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { nativeOrder } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { RanMars } from '../random_mars';
import { pow } from '../libm/pow';
import { cos, sin } from '../libm/sincos';

/** convert_double_to_equivalent_int: INT_MAX (limits.h). */
const INT_MAX = 2147483647;

export class PairDSMC extends Pair {
  readonly name = 'dsmc';
  /** No neighbor list: the collisions are drawn per cell. */
  needsHalf = false;
  private cutGlobal = 0;
  private maxCellSize = 0;
  private seed = 0;
  private weighting = 0;
  private Tref = 0;
  private kTref = 0;
  private recomputeStride = 0;
  private vsigmaSamples = 0;
  private random: RanMars | null = null;
  private setflag = new Uint8Array(0);
  private sigma = new Float64Array(0);
  private cutCoef = new Float64Array(0);
  private vsMax = new Float64Array(0);
  private ncells: [number, number, number] = [1, 1, 1];
  private cellSize: [number, number, number] = [0, 0, 0];
  private totalCells = 0;
  private vol = 0;
  private dt = 0;
  private twoPi = 0;
  /** Units boltz from the last settings()/init() (native force->boltz). */
  private boltz = 1;
  /** Log sink of the warnings (set by settings and init). */
  private logSink: ((text: string) => void) | null = null;

  // the state of the (itype, jtype) pair being collided (native members itype, jtype, reduced_mass, ...)
  private itype = 0;
  private jtype = 0;
  private reducedMass = 0;
  private imassTmass = 0;
  private jmassTmass = 0;
  private imass = 0;
  private jmass = 0;
  private totalNumberOfCollisions = 0;
  /** Native-order atom positions: the engine index of the native atom k (SimState.order). */
  private order: Int32Array = new Int32Array(0);
  private vel: Float64Array = new Float64Array(0);
  private lists: number[][] = [];
  private nA = 0;
  private nB = 0;

  /** pair_dsmc.cpp settings(): six arguments, the RanMars stream is seeded from the seed. */
  settings(args: string[], ctx: StyleContext): void {
    if (args.length !== 6) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = 0.0;
    this.maxCellSize = parseNum(args[0], 'max_cell_size');
    this.seed = parseNum(args[1], 'seed');
    this.weighting = parseNum(args[2], 'weighting');
    this.Tref = parseNum(args[3], 'Tref');
    this.recomputeStride = parseNum(args[4], 'Nrecompute');
    this.vsigmaSamples = parseNum(args[5], 'Nsample');
    if (this.maxCellSize <= 0.0) throw new StyleError('Illegal pair_style command');
    if (this.seed <= 0) throw new StyleError('Illegal pair_style command');
    this.random = new RanMars(this.seed);
    this.logSink = (text: string) => ctx.log(text);
    this.boltz = ctx.s?.units.boltz ?? 1;
    this.kTref = this.boltz * this.Tref;
    if (this.setflag.length) {
      const nt = this.ntypes + 1;
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) if (this.setflag[i * nt + j]) this.cutCoef[i * nt + j] = this.cutGlobal;
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = (ntypes + 1) * (ntypes + 1);
    this.setflag = new Uint8Array(n);
    this.sigma = new Float64Array(n);
    this.cutCoef = new Float64Array(n);
    this.vsMax = new Float64Array(n);
  }

  /** pair_dsmc.cpp coeff(): "I J sigma [cutoff]". */
  coeff(args: string[]): void {
    if (args.length < 3 || args.length > 4) throw new StyleError('Incorrect args for pair coefficients');
    const iR = typeBounds(args[0], this.ntypes), jR = typeBounds(args[1], this.ntypes);
    const sigmaOne = parseNum(args[2], 'sigma');
    const cutOne = args.length === 4 ? parseNum(args[3], 'cutoff') : this.cutGlobal;
    const nt = this.ntypes + 1;
    let count = 0;
    for (let i = iR[0]; i <= iR[1]; i++) {
      for (let j = Math.max(jR[0], i); j <= jR[1]; j++) {
        this.sigma[i * nt + j] = sigmaOne;
        this.cutCoef[i * nt + j] = cutOne;
        this.setflag[i * nt + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** pair_dsmc.cpp init_style(): the cell grid (cells no wider than max_cell_size) and the cell volume. */
  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) throw new StyleError('Pair style dsmc requires a simulation box');
    this.logSink = (text: string) => ctx.log(text);
    for (let t = 1; t <= this.ntypes; t++) {
      if (Number.isNaN(s.massByType[t])) throw new StyleError('Pair style dsmc requires per atom type masses');
    }
    this.boltz = s.units.boltz;
    this.kTref = this.boltz * this.Tref;
    const lo = s.box.lo, hi = s.box.hi;
    const nc: [number, number, number] = [1, 1, 1];
    for (let d = 0; d < 3; d++) {
      while ((hi[d] - lo[d]) / nc[d] > this.maxCellSize) nc[d]++;
    }
    this.ncells = nc;
    this.cellSize = [(hi[0] - lo[0]) / nc[0], (hi[1] - lo[1]) / nc[1], (hi[2] - lo[2]) / nc[2]];
    this.totalCells = nc[0] * nc[1] * nc[2];
    this.vol = this.cellSize[0] * this.cellSize[1] * this.cellSize[2];
    this.dt = s.dt;
    this.vsMax.fill(0.0);
    this.twoPi = 8.0 * Math.atan(1.0);
    ctx.log(`DSMC cell size = ${this.cellSize[0]} x ${this.cellSize[1]} x ${this.cellSize[2]}`);
  }

  /** pair_dsmc.cpp init_one(): the cutoff of an unset pair is 0. */
  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (this.setflag[i * nt + j] === 0) this.cutCoef[i * nt + j] = 0.0;
    return this.cutCoef[i * nt + j];
  }

  /** VHS cross section times the relative speed (pair_dsmc.cpp V_sigma). */
  private vSigma(i: number, j: number): number {
    const ei = this.order[i], ej = this.order[j];
    const v = this.vel;
    const dx = v[3 * ei] - v[3 * ej], dy = v[3 * ei + 1] - v[3 * ej + 1], dz = v[3 * ei + 2] - v[3 * ej + 2];
    const rvsq = dx * dx + dy * dy + dz * dz;
    const rv = Math.sqrt(rvsq);
    let psig: number;
    if (rvsq !== 0.0) {
      const nt = this.ntypes + 1;
      psig = this.sigma[this.itype * nt + this.jtype] * pow(this.kTref / (0.5 * this.reducedMass * rvsq), 0.17) * 1.06418029298371;
    } else psig = 0.0;
    return rv * psig;
  }

  /** pair_dsmc.cpp recompute_V_sigma_max(): the largest V_sigma of sampled pairs of the current types. */
  private recomputeVSigmaMax(): void {
    const nt = this.ntypes + 1;
    let vmax = 0;
    if (this.nA && this.nB) {
      for (let k = 0; k < this.vsigmaSamples; k++) {
        const i = this.lists[this.itype][Math.trunc(this.random!.uniform() * this.nA)];
        const j = this.lists[this.jtype][Math.trunc(this.random!.uniform() * this.nB)];
        if (i === j) continue;
        const vs = this.vSigma(i, j);
        vmax = vmax > vs ? vmax : vs;
      }
    }
    this.vsMax[this.itype * nt + this.jtype] = vmax;
  }

  /** pair_dsmc.cpp scatter_random(): new velocities for the collided pair from a random direction. */
  private scatterRandom(i: number, j: number): void {
    const ei = this.order[i], ej = this.order[j];
    const v = this.vel;
    const dx = v[3 * ei] - v[3 * ej], dy = v[3 * ei + 1] - v[3 * ej + 1], dz = v[3 * ei + 2] - v[3 * ej + 2];
    const dot = dx * dx + dy * dy + dz * dz;
    const same = this.itype === this.jtype;
    const magDelv = same ? Math.sqrt(dot) * 0.5 : Math.sqrt(dot);
    const cosPhi = 1.0 - (2.0 * this.random!.uniform());
    const cosSquared = Math.min(1.0, cosPhi * cosPhi);
    const r = Math.sqrt(1.0 - cosSquared);
    const d0 = cosPhi * magDelv;
    const theta = this.twoPi * this.random!.uniform();
    const d1 = r * magDelv * cos(theta);
    const d2 = r * magDelv * sin(theta);
    if (same) {
      const c0 = (v[3 * ei] + v[3 * ej]) * 0.5;
      const c1 = (v[3 * ei + 1] + v[3 * ej + 1]) * 0.5;
      const c2 = (v[3 * ei + 2] + v[3 * ej + 2]) * 0.5;
      v[3 * ei] = c0 + d0; v[3 * ei + 1] = c1 + d1; v[3 * ei + 2] = c2 + d2;
      v[3 * ej] = c0 - d0; v[3 * ej + 1] = c1 - d1; v[3 * ej + 2] = c2 - d2;
    } else {
      const c0 = v[3 * ei] * this.imassTmass + v[3 * ej] * this.jmassTmass;
      const c1 = v[3 * ei + 1] * this.imassTmass + v[3 * ej + 1] * this.jmassTmass;
      const c2 = v[3 * ei + 2] * this.imassTmass + v[3 * ej + 2] * this.jmassTmass;
      v[3 * ei] = c0 + d0 * this.jmassTmass;
      v[3 * ei + 1] = c1 + d1 * this.jmassTmass;
      v[3 * ei + 2] = c2 + d2 * this.jmassTmass;
      v[3 * ej] = c0 - d0 * this.imassTmass;
      v[3 * ej + 1] = c1 - d1 * this.imassTmass;
      v[3 * ej + 2] = c2 - d2 * this.imassTmass;
    }
    this.totalNumberOfCollisions++;
  }

  /** pair_dsmc.cpp convert_double_to_equivalent_int(): the integer part after a uniform draw is added. */
  private convertToInt(input: number): number {
    if (input > INT_MAX) throw new StyleError('Tried to convert a double to int, but input_double > INT_MAX');
    return Math.trunc(input + this.random!.uniform());
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const nlocal = pc.nlocal;
    const ntypes = this.ntypes;
    const nt = ntypes + 1;
    const order = nativeOrder(s);
    this.order = order;
    this.vel = s.v;
    const x = s.x, type = s.type, mass = s.massByType;
    const lo = s.box.lo;
    const [ncx, ncy, ncz] = this.ncells;
    const [cellx, celly, cellz] = this.cellSize;
    const total = this.totalCells;
    this.dt = s.dt;
    const first = new Int32Array(nt * total).fill(-1);
    const number = new Int32Array(nt * total);
    const nextParticle = new Int32Array(nlocal);
    // bin the atoms in native order; a list is built by prepending (native first/next_particle)
    for (let k = 0; k < nlocal; k++) {
      const e = order[k];
      const xcell = Math.trunc((x[3 * e] - lo[0]) / cellx);
      const ycell = Math.trunc((x[3 * e + 1] - lo[1]) / celly);
      const zcell = Math.trunc((x[3 * e + 2] - lo[2]) / cellz);
      if (xcell < 0 || xcell > ncx - 1 || ycell < 0 || ycell > ncy - 1 || zcell < 0 || zcell > ncz - 1) continue;
      const icell = xcell + ycell * ncx + zcell * ncx * ncy;
      const itype = type[e];
      nextParticle[k] = first[itype * total + icell];
      first[itype * total + icell] = k;
      number[itype * total + icell]++;
    }
    const lists: number[][] = [];
    for (let t = 0; t <= ntypes; t++) lists.push([]);
    this.lists = lists;
    const stride = this.recomputeStride;
    const step = s.step;
    for (let icell = 0; icell < total; icell++) {
      for (let itype = 1; itype <= ntypes; itype++) {
        const n = number[itype * total + icell];
        const list = new Array<number>(n);
        let m = first[itype * total + icell];
        for (let k = 0; k < n; k++) {
          list[k] = m;
          m = nextParticle[m];
        }
        lists[itype] = list;
      }
      for (let itype = 1; itype <= ntypes; itype++) {
        this.itype = itype;
        this.imass = mass[itype];
        const nA = number[itype * total + icell];
        this.nA = nA;
        for (let jtype = itype; jtype <= ntypes; jtype++) {
          this.jtype = jtype;
          this.jmass = mass[jtype];
          const nB = number[jtype * total + icell];
          this.nB = nB;
          const imass = this.imass, jmass = this.jmass;
          this.reducedMass = imass * jmass / (imass + jmass);
          const totalMass = imass + jmass;
          this.jmassTmass = jmass / totalMass;
          this.imassTmass = imass / totalMass;
          if (stride && step % stride === 0) this.recomputeVSigmaMax();
          const vsMaxPair = this.vsMax[itype * nt + jtype];
          let numDouble = nA * nB * this.weighting * vsMaxPair * this.dt / this.vol;
          if (itype === jtype && nB) numDouble *= 0.5 * (nB - 1) / nB;
          const numCollisions = this.convertToInt(numDouble);
          if (numCollisions > nA) this.warn('Pair dsmc: num_of_collisions > number_of_A');
          if (numCollisions > nB) this.warn('Pair dsmc: num_of_collisions > number_of_B');
          for (let k = 0; k < numCollisions; k++) {
            if (nA < 1 || nB < 1) break;
            if (itype === jtype && nA < 2) break;
            const ithA = Math.trunc(this.random!.uniform() * nA);
            const jthB = Math.trunc(this.random!.uniform() * nB);
            const i = lists[itype][ithA];
            const j = lists[jtype][jthB];
            if (i === j) {
              k--;
              continue;
            }
            const probability = this.vSigma(i, j) / vsMaxPair;
            if (probability > this.random!.uniform()) this.scatterRandom(i, j);
          }
        }
      }
    }
  }

  private warn(text: string): void {
    this.logSink?.(`WARNING: ${text}`);
  }
}
