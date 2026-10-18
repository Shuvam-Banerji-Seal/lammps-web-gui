/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/SRD/fix_wall_srd.cpp and
 * src/SRD/fix_wall_srd.h (Jeremy Lechman (SNL), Steve Plimpton (SNL): the SRD package). Copyright (2003) Sandia
 * Corporation. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix wall/srd (docs.lammps.org/fix_wall_srd.html): "Each specified wall is a flat plane ... SRD particles are
 * reflected from it by fix srd; the wall position may be constant, EDGE or an equal-style variable, and its velocity
 * is the difference of its positions over one timestep." The fix has no forces of its own: fix srd reads its wall
 * positions (xwall), the reneighbor copies (xwallhold), the wall velocities (vwall) and accumulates the force on each
 * wall (fwall, the f_ID array).
 *
 * Syntax: fix ID group-ID wall/srd face args ... [units box|lattice]; face = xlo, xhi, ylo, yhi, zlo or zhi; args =
 * EDGE, a constant coordinate, or v_name for an equal-style variable.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { valueOf } from '../../engine/fix/util';
import type { SrdWallFix } from '../srd/fix_srd';

/** enum { XLO, XHI, YLO, YHI, ZLO, ZHI } (fix_wall_srd.cpp) */
const WALL_NAMES = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'];
/** enum { NONE, EDGE, CONSTANT, VARIABLE } */
const EDGE = 1;
const CONSTANT = 2;
const VARIABLE = 3;

export class FixWallSRD extends Fix implements SrdWallFix {
  readonly style = 'wall/srd';
  nwall = 0;
  varflag = 0;
  overlap = 0;
  wallwhich: number[] = [];
  xwall = new Float64Array(6);
  xwallhold = new Float64Array(6);
  vwall = new Float64Array(6);
  /** force on each wall, fwall[3 m + k] (native fwall[nwall][3]) */
  fwall = new Float64Array(18);

  private readonly wallstyle: number[] = [];
  private readonly coord0 = new Float64Array(6);
  private readonly varstr: string[] = [];
  private readonly xwalllast = new Float64Array(6);
  private laststep = -1;
  /** timestep (update->dt), set by init() */
  private dt = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal fix wall/srd command: missing argument(s)');
    const s = sys.state;
    const box = s.box;
    let scaleflag = true;
    let iarg = 0;
    while (iarg < args.length) {
      const w = args[iarg];
      const idx = WALL_NAMES.indexOf(w);
      if (idx >= 0) {
        if (iarg + 2 > args.length) throw new StyleError(`Illegal fix wall/srd ${w} command: missing argument(s)`);
        for (let m = 0; m < this.nwall; m++) {
          if (this.wallwhich[m] === idx) throw new StyleError(`Wall ${w} defined twice in fix wall/srd command`);
        }
        this.wallwhich.push(idx);
        const val = args[iarg + 1];
        const dim = idx >> 1, side = idx & 1;
        if (val === 'EDGE') {
          this.wallstyle.push(EDGE);
          this.coord0[this.nwall] = side === 0 ? box.lo[dim] : box.hi[dim];
          this.varstr.push('');
        } else if (val.startsWith('v_')) {
          this.wallstyle.push(VARIABLE);
          this.varstr.push(val.slice(2));
        } else {
          const c = Number(val);
          if (val === '' || !Number.isFinite(c)) throw new StyleError(`Expected floating point parameter instead of '${val}' in input script or data file`);
          this.wallstyle.push(CONSTANT);
          this.coord0[this.nwall] = c;
          this.varstr.push('');
        }
        this.nwall++;
        iarg += 2;
      } else if (w === 'units') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal fix wall/srd units command: missing argument(s)');
        const u = args[iarg + 1];
        if (u === 'box') scaleflag = false;
        else if (u === 'lattice') scaleflag = true;
        else throw new StyleError(`Unknown fix wall/srd units setting ${u}`);
        iarg += 2;
      } else {
        throw new StyleError(`Unknown fix wall/srd keyword ${w}`);
      }
    }
    if (this.nwall === 0) throw new StyleError('Illegal fix wall command');
    for (let m = 0; m < this.nwall; m++) {
      const dim = this.wallwhich[m] >> 1;
      if (dim === 0 && box.periodic[0]) throw new StyleError('Cannot use fix wall/srd xlo or xhi with periodic x dimension');
      if (dim === 1 && box.periodic[1]) throw new StyleError('Cannot use fix wall/srd ylo or yhi with periodic y dimension');
      if (dim === 2 && box.periodic[2]) throw new StyleError('Cannot use fix wall/srd zlo or zhi with periodic z dimension');
    }
    for (let m = 0; m < this.nwall; m++) {
      if ((this.wallwhich[m] >> 1) === 2 && s.dimension === 2) throw new StyleError('Cannot use fix wall/srd zlo/zhi for a 2d simulation');
    }
    // array_flag: f_ID[row][col] is the force on wall row, column = x, y, z
    this.arrayFlag = true;
    this.sizeArrayRows = this.nwall;
    this.sizeArrayCols = 3;
    this.fwall = new Float64Array(3 * this.nwall);

    // scale coord for CONSTANT walls (lattice spacing of the wall's dimension, or 1 for units box)
    for (let m = 0; m < this.nwall; m++) {
      if (this.wallstyle[m] !== CONSTANT) continue;
      const dim = this.wallwhich[m] >> 1;
      const scale = scaleflag ? sys.lattice?.spacing[dim] ?? 1 : 1;
      this.coord0[m] *= scale;
    }
    // overlap: walls in more than one dimension
    const dimflag = [0, 0, 0];
    for (let m = 0; m < this.nwall; m++) dimflag[this.wallwhich[m] >> 1] = 1;
    this.overlap = dimflag[0] + dimflag[1] + dimflag[2] > 1 ? 1 : 0;
    // varflag: some wall position is an equal-style variable
    this.varflag = this.wallstyle.some((t) => t === VARIABLE) ? 1 : 0;
    this.laststep = -1;
  }

  /** Native FixWallSRD::init(): fix srd must be defined, the variables must exist, dt = update->dt. */
  init(): void {
    if (!this.sys.fixes.some((f) => f.style === 'srd')) throw new StyleError('Cannot use fix wall/srd without fix srd');
    for (let m = 0; m < this.nwall; m++) {
      if (this.wallstyle[m] !== VARIABLE) continue;
      if (!this.sys.vars.has(this.varstr[m])) {
        throw new StyleError(`Variable ${this.varstr[m]} for fix wall/srd does not exist`);
      }
    }
    this.dt = this.sys.state.dt;
  }

  /**
   * Native FixWallSRD::wall_params(flag): sets each wall's position for this timestep and its velocity (the difference
   * of positions over dt), zeroes the wall forces, and with flag (called at a reneighbor) archives the positions that
   * the reneighbor trigger compares against.
   */
  wallParams(flag: 0 | 1): void {
    const ntimestep = this.sys.state.step;
    for (let m = 0; m < this.nwall; m++) {
      let xnew: number;
      if (this.wallstyle[m] === VARIABLE) xnew = valueOf(this.sys, { variable: this.varstr[m] });
      else xnew = this.coord0[m];
      if (this.laststep < 0) {
        this.xwall[m] = this.xwalllast[m] = xnew;
        this.vwall[m] = 0.0;
      } else if (this.laststep < ntimestep) {
        this.xwalllast[m] = this.xwall[m];
        this.xwall[m] = xnew;
        this.vwall[m] = (this.xwall[m] - this.xwalllast[m]) / this.dt;
      }
      this.fwall[3 * m] = this.fwall[3 * m + 1] = this.fwall[3 * m + 2] = 0.0;
    }
    this.laststep = ntimestep;
    if (flag) for (let m = 0; m < this.nwall; m++) this.xwallhold[m] = this.xwall[m];
  }

  /** Native compute_array(i, j): the force on wall i in direction j (summed over processors: one here). */
  computeArray(i: number, j: number): number {
    return this.fwall[3 * i + j];
  }
}
