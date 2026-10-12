/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SRD/fix_srd.cpp and src/SRD/fix_srd.h at commit b332f28 (fix srd, the SRD package: streaming of
 * the stochastic rotation dynamics solvent, the rotation of SRD velocities in bins with one of six random axes,
 * and collisions of SRD particles with spherical big particles). Contributing authors: Jeremy Lechman (SNL),
 * Pieter in 't Veld (BASF). Copyright (2003) Sandia Corporation, GNU General Public License. This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_srd.html: "Perform stochastic rotation dynamics (SRD) on particles in the group, in order to
 * simulate a solvent ... Fix srd also enables ... collisions between SRD and big particles ... each SRD particle is
 * advected ... the SRD velocities are rotated ... in each bin".
 *
 * Supported here: lj or other unit styles through the unit table, 2d and 3d boxes, point SRD particles of one mass,
 * big particles that are spheres, collision slip, exact and inexact (exact yes/no), bounce N, inside
 * error/warn/ignore, lamda, radius, search, cubic, rescale, and the default shift no, tstat no, overlap no.
 * Refused with a StyleError naming the keyword: collision noslip, shift yes/possible, tstat yes, unbiased yes,
 * overlap yes, ellipsoid/line/tri big particles, and fix wall/srd.
 *
 * Browser-engine difference: native processes SRD atoms in its own local order; the engine keeps that order in
 * SimState.order (nativeOrder), and every loop over SRD particles goes through it, so the random draws happen in
 * the same sequence as the single-processor native run. Ghost images of big particles are emulated: the periodic
 * copies that overlap the search region are registered in the search bins exactly as native registers its ghosts
 * (same bins, same stencil, same counts), and forces on a copy are added to its owner.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isLineAtom, isTriAtom, massOf, nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';

/** enum { INSIDE_ERROR, INSIDE_WARN, INSIDE_IGNORE } */
const INSIDE_ERROR = 0;
const INSIDE_WARN = 1;
const INSIDE_IGNORE = 2;
/** enum { CUBIC_ERROR, CUBIC_WARN } */
const CUBIC_ERROR = 0;
const CUBIC_WARN = 1;
/** enum { BIG_MOVE, SRD_MOVE, SRD_ROTATE } */
const BIG_MOVE = 0;
const SRD_MOVE = 1;
const SRD_ROTATE = 2;

/** static const int ATOMPERBIN = 30 (native error "too many big particles in bin"). */
const ATOMPERBIN = 30;
const BIG = 1.0e20;
const EMPTY: number[] = [];

const ZERO3 = [0, 0, 0];

export class FixSRD extends Fix {
  readonly style: string = 'srd';

  private readonly bigBit: number;
  private readonly bigExist: boolean;
  private temperatureSrd: number;
  private readonly gridsrd: number;
  private readonly random: RanMars;
  private lamdaflag = false;
  private lamda = 0;
  private radfactor = 1;
  private exactflag = true;
  private insideflag = INSIDE_ERROR;
  private maxbounceallow = 0;
  private gridsearch: number;
  private cubicflag = CUBIC_ERROR;
  private cubictol = 0.01;
  private rescaleRotate = true;
  private rescaleCollide = true;

  private initflag = false;
  private dtBig = 0;
  private dtSrd = 0;
  private massSrd = 0;
  private sigma = 0;
  private dmax = 0;
  private vmax = 0;
  private vmaxsq = 0;
  private maxbigdiam = 0;
  private minbigdiam = 0;

  /** velocity bins (reset_velocities), orthogonal box: nbin1 per dimension (z = 1 in 2d) */
  private readonly nbin1 = [1, 1, 1];
  private readonly binsize1 = [0, 0, 0];
  private readonly bininv1 = [0, 0, 0];
  private nbins1 = 1;

  private distGhost = 0;
  private distSrd = 0;
  private distSrdReneigh = 0;
  private readonly srdlo = [0, 0, 0];
  private readonly srdhi = [0, 0, 0];
  private readonly srdloRe = [0, 0, 0];
  private readonly srdhiRe = [0, 0, 0];

  /** search bins for big particles and their stencil (setup_search_bins / setup_search_stencil) */
  private readonly nbin2 = [1, 1, 1];
  private readonly binsize2 = [0, 0, 0];
  private readonly bininv2 = [0, 0, 0];
  private readonly blo2 = [0, 0, 0];
  private nbins2 = 0;
  private stencil: number[][] = [];
  /** big copies (owned particles and periodic images) registered in each search bin */
  private binbig: number[][] = [];
  private binsrd = new Int32Array(0);

  /** big copies: owner index, periodic shift (3 per copy), radius, squared bin cutoff, position */
  private copyOwner = new Int32Array(0);
  private copyShift = new Int32Array(0);
  private copyRadius = new Float64Array(0);
  private copyCutbinsq = new Float64Array(0);
  private copyPos = new Float64Array(0);

  private reneighflag = BIG_MOVE;
  private nextReneighbor = -1;

  /** per-timestep and per-run statistics (compute_vector indices 0..11) */
  private ncheck = 0;
  private ncollide = 0;
  private nbounce = 0;
  private ninside = 0;
  private nrescale = 0;
  private reneighcount = 0;
  private bouncemaxnum = 0;
  private bouncemax = 0;
  private srdBinCount = 0;
  private srdBinTemp = 0;
  private statsFlag = 1;
  private readonly statsAll = new Float64Array(12);

  /** reset_velocities scratch, one entry per velocity bin */
  private binhead = new Int32Array(0);
  private binnext = new Int32Array(0);
  private binVsum = new Float64Array(0);
  private binCount = new Int32Array(0);
  private binRandom = new Float64Array(0);
  private binValue = new Float64Array(0);

  /** collision scratch vectors */
  private readonly xs = new Float64Array(3);
  private readonly vs = new Float64Array(3);
  private readonly xb = new Float64Array(3);
  private readonly xscoll = new Float64Array(3);
  private readonly xbcoll = new Float64Array(3);
  private readonly norm = new Float64Array(3);
  private readonly vsnew = new Float64Array(3);
  private readonly tangent = new Float64Array(3);
  private readonly vsurf = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_srd.cpp: "if (narg < 8) utils::missing_cmd_args(FLERR, "fix srd", error);"
    if (args.length < 5) throw new StyleError('Illegal fix srd command: fix ID group srd N big-group-ID T grid seed ...');
    this.vectorFlag = true;
    this.sizeVector = 12;
    this.extvector = 0;

    const nevery = intArg(args[0], 'nevery');
    if (nevery <= 0) throw new StyleError('Fix srd nevery value must be > 0');
    this.nevery = nevery;

    this.bigExist = args[1] !== 'NULL';
    if (this.bigExist) {
      if (sys.groups.find(args[1]) < 0) throw new StyleError(`Could not find fix srd big particle group ${args[1]}`);
      this.bigBit = sys.groups.bit(args[1]);
    } else {
      this.bigBit = 0;
    }

    this.temperatureSrd = numArg(args[2], 'temperature');
    if (this.temperatureSrd <= 0.0) throw new StyleError(`Illegal fix srd temperature ${args[2]}`);
    this.gridsrd = numArg(args[3], 'grid spacing');
    if (this.gridsrd <= 0.0) throw new StyleError(`Illegal fix srd grid spacing ${args[3]}`);
    const seed = intArg(args[4], 'random seed');
    if (seed <= 0) throw new StyleError(`Illegal fix srd random seed ${args[4]}`);
    this.gridsearch = this.gridsrd;

    let collide = 'noslip';
    let collideSet = false;
    let shift = 'no';
    let tstat = false;
    let unbias = false;
    let overlap = false;
    let iarg = 5;
    const need = (k: number, key: string) => {
      if (iarg + k > args.length) throw new StyleError(`Illegal fix srd ${key} keyword: missing argument`);
    };
    while (iarg < args.length) {
      const key = args[iarg];
      if (key === 'lamda') {
        need(2, key);
        this.lamda = numArg(args[iarg + 1], 'lamda');
        if (this.lamda <= 0.0) throw new StyleError('Illegal fix srd lamda value');
        this.lamdaflag = true;
        iarg += 2;
      } else if (key === 'collision') {
        need(2, key);
        collide = args[iarg + 1];
        if (collide !== 'slip' && collide !== 'noslip') throw new StyleError(`Unknown fix srd collision argument ${collide}`);
        collideSet = true;
        iarg += 2;
      } else if (key === 'overlap') {
        need(2, key);
        overlap = logicalArg(args[iarg + 1], 'overlap');
        iarg += 2;
      } else if (key === 'inside') {
        need(2, key);
        const w = args[iarg + 1];
        if (w === 'error') this.insideflag = INSIDE_ERROR;
        else if (w === 'warn') this.insideflag = INSIDE_WARN;
        else if (w === 'ignore') this.insideflag = INSIDE_IGNORE;
        else throw new StyleError(`Unknown fix srd inside argument ${w}`);
        iarg += 2;
      } else if (key === 'exact') {
        need(2, key);
        this.exactflag = logicalArg(args[iarg + 1], 'exact');
        iarg += 2;
      } else if (key === 'radius') {
        need(2, key);
        this.radfactor = numArg(args[iarg + 1], 'radius');
        if (this.radfactor <= 0.0) throw new StyleError('Illegal fix srd radius value');
        iarg += 2;
      } else if (key === 'bounce') {
        need(2, key);
        this.maxbounceallow = intArg(args[iarg + 1], 'bounce');
        if (this.maxbounceallow < 0) throw new StyleError('Illegal fix srd bounce value');
        iarg += 2;
      } else if (key === 'search') {
        need(2, key);
        this.gridsearch = numArg(args[iarg + 1], 'search');
        if (this.gridsearch <= 0.0) throw new StyleError('Illegal fix srd search value');
        iarg += 2;
      } else if (key === 'cubic') {
        need(3, key);
        const w = args[iarg + 1];
        if (w === 'error') this.cubicflag = CUBIC_ERROR;
        else if (w === 'warn') this.cubicflag = CUBIC_WARN;
        else throw new StyleError(`Unknown fix srd cubic argument ${w}`);
        this.cubictol = numArg(args[iarg + 2], 'cubic');
        if (this.cubictol < 0.0 || this.cubictol > 1.0) throw new StyleError('Illegal fix srd cubic value');
        iarg += 3;
      } else if (key === 'shift') {
        need(3, key);
        shift = args[iarg + 1];
        if (shift !== 'no' && shift !== 'yes' && shift !== 'possible') throw new StyleError(`Unknown fix srd shift argument ${shift}`);
        intArg(args[iarg + 2], 'shift seed');
        iarg += 3;
      } else if (key === 'tstat') {
        need(2, key);
        tstat = logicalArg(args[iarg + 1], 'tstat');
        iarg += 2;
      } else if (key === 'unbiased') {
        need(2, key);
        unbias = logicalArg(args[iarg + 1], 'unbiased');
        if (unbias && !tstat) throw new StyleError('Must use SRD thermostat when enabling unbiased thermostatting');
        iarg += 2;
      } else if (key === 'rescale') {
        need(2, key);
        const w = args[iarg + 1];
        if (w === 'no') { this.rescaleRotate = false; this.rescaleCollide = false; }
        else if (w === 'yes') { this.rescaleRotate = true; this.rescaleCollide = true; }
        else if (w === 'rotate') { this.rescaleRotate = true; this.rescaleCollide = false; }
        else if (w === 'collide') { this.rescaleRotate = false; this.rescaleCollide = true; }
        else throw new StyleError(`Unknown fix srd rescale argument ${w}`);
        iarg += 2;
      } else {
        throw new StyleError(`Unknown fix srd keyword ${key}`);
      }
    }

    // the refusals: keywords whose physics is not ported yet (named, never ignored)
    if (collideSet && collide === 'noslip') throw new StyleError('fix srd collision noslip is not supported by this engine yet (only collision slip)');
    if (shift !== 'no') throw new StyleError(`fix srd shift ${shift} is not supported by this engine yet (only shift no)`);
    if (tstat) throw new StyleError('fix srd tstat yes is not supported by this engine yet');
    if (unbias) throw new StyleError('fix srd unbiased yes is not supported by this engine yet');
    if (overlap) throw new StyleError('fix srd overlap yes is not supported by this engine yet (only overlap no)');

    // initialize Marsaglia RNG with processor-unique seed (one processor: seed + 0)
    this.random = new RanMars(seed);
  }

  /** fix_srd.cpp setmask(): PRE_NEIGHBOR and POST_FORCE (the engine calls preNeighbor and postForce). */

  /** Forced reneighboring of the SRD step (Neighbor::decide: fix next_reneighbor == ntimestep). */
  reneighborDue(step: number): boolean {
    return step === this.nextReneighbor;
  }

  init(): void {
    const s = this.sys.state;
    if (s.box.triclinic) throw new StyleError('Fix srd does not support triclinic boxes');
    if (this.initflag && s.dt !== this.dtBig) throw new StyleError('Cannot change timestep once fix srd is set up');
    if (this.bigExist) this.checkBig();

    this.parameterize();

    // limit initial SRD velocities if necessary
    const { v, mask } = s;
    const order = nativeOrder(s);
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(mask[i] & this.groupBit)) continue;
      const vsq = v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2];
      if (vsq > this.vmaxsq) {
        this.nrescale++;
        scale3(v, 3 * i, this.vmax / Math.sqrt(vsq));
      }
    }

    // zero per-run stats
    this.nrescale = 0;
    this.bouncemaxnum = 0;
    this.bouncemax = 0;
    this.reneighcount = 0;
    this.initflag = true;
    this.nextReneighbor = -1;
  }

  setup(): void {
    this.setupBounds();
    if (this.bigExist) {
      this.setupSearchBins();
      this.setupSearchStencil();
    } else {
      this.nbins2 = 0;
    }
    // perform first binning of SRD and big particles; no SRD rotation in setup
    this.reneighflag = BIG_MOVE;
    this.preNeighbor();
  }

  /** Big particles must be spheres here (ellipsoid, line and tri big particles are refused). */
  private checkBig(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.bigBit)) continue;
      if (s.shape && (s.shape[3 * i] > 0 || s.shape[3 * i + 1] > 0 || s.shape[3 * i + 2] > 0)) {
        throw new StyleError('fix srd big particles that are ellipsoids are not supported by this engine yet (spheres only)');
      }
      if (isLineAtom(s, i) || isTriAtom(s, i)) {
        throw new StyleError('fix srd big particles that are lines or triangles are not supported by this engine yet (spheres only)');
      }
    }
  }

  /** fix_srd.cpp parameterize(): sizes, masses, velocity limits and the bin grid. */
  private parameterize(): void {
    const s = this.sys.state;
    this.dtBig = s.dt;
    this.dtSrd = this.nevery * s.dt;
    const u = s.units;

    // maxbigdiam, minbigdiam = max/min diameter of any big particle; radfactor applies at the end
    let maxd = 0;
    let mind = BIG;
    if (this.bigExist) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.bigBit)) continue;
        if (!(s.radius && s.radius[i] > 0.0)) throw new StyleError('Big particle in fix srd cannot be point particle');
        maxd = Math.max(maxd, 2.0 * s.radius[i]);
        mind = Math.min(mind, 2.0 * s.radius[i]);
      }
    }
    this.maxbigdiam = maxd * this.radfactor;
    this.minbigdiam = mind * this.radfactor;

    // mass of SRD particles, require monodispersity
    let massSrd = 0;
    let flag = false;
    const order = nativeOrder(s);
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      if (massSrd === 0.0) massSrd = m;
      else if (m !== massSrd) flag = true;
    }
    if (flag) throw new StyleError('Fix srd requires SRD particles all have same mass');
    this.massSrd = massSrd;

    // set temperature and lamda of SRD particles from each other
    if (!this.lamdaflag) {
      this.lamda = this.dtSrd * Math.sqrt(u.boltz * this.temperatureSrd / this.massSrd / u.mvv2e);
    } else {
      this.temperatureSrd = u.mvv2e * (this.lamda / this.dtSrd) * (this.lamda / this.dtSrd) * this.massSrd / u.boltz;
    }

    // vmax = maximum velocity of an SRD particle; dmax = maximum distance an SRD can move = 4*lamda
    this.sigma = this.lamda / this.dtSrd;
    this.dmax = 4.0 * this.lamda;
    this.vmax = this.dmax / this.dtSrd;
    this.vmaxsq = this.vmax * this.vmax;

    this.setupVelocityBins();

    // SRD bins must be cubic within tolerance, and close to the requested size
    const b3 = this.binsize1;
    const dim = s.dimension;
    let tolflag = false;
    if (b3[1] / b3[0] > 1.0 + this.cubictol || b3[0] / b3[1] > 1.0 + this.cubictol) tolflag = true;
    if (dim === 3 && (b3[2] / b3[0] > 1.0 + this.cubictol || b3[0] / b3[2] > 1.0 + this.cubictol)) tolflag = true;
    if (tolflag) this.cubicCheck('SRD bins for fix srd are not cubic enough');
    tolflag = false;
    const g = this.gridsrd;
    if (b3[0] / g > 1.0 + this.cubictol || g / b3[0] > 1.0 + this.cubictol) tolflag = true;
    if (b3[1] / g > 1.0 + this.cubictol || g / b3[1] > 1.0 + this.cubictol) tolflag = true;
    if (dim === 3 && (b3[2] / g > 1.0 + this.cubictol || g / b3[2] > 1.0 + this.cubictol)) tolflag = true;
    if (tolflag) this.cubicCheck('SRD bin size for fix srd differs from user request');

    // lamda must be at least 0.6 of the SRD grid size (shifting is refused, so no fallback to it)
    let maxgridsrd = Math.max(b3[0], b3[1]);
    if (dim === 3) maxgridsrd = Math.max(maxgridsrd, b3[2]);
    if (this.lamda < 0.6 * maxgridsrd) throw new StyleError('Fix srd lamda must be >= 0.6 of SRD grid size');
  }

  private cubicCheck(msg: string): void {
    if (this.cubicflag === CUBIC_ERROR) throw new StyleError(msg);
    this.sys.warn(msg);
  }

  /** fix_srd.cpp setup_velocity_bins() for one processor and no shift (orthogonal box). */
  private setupVelocityBins(): void {
    const s = this.sys.state;
    const len = [s.box.hi[0] - s.box.lo[0], s.box.hi[1] - s.box.lo[1], s.box.hi[2] - s.box.lo[2]];
    this.nbin1[0] = Math.round(len[0] / this.gridsrd);
    this.nbin1[1] = Math.round(len[1] / this.gridsrd);
    this.nbin1[2] = Math.round(len[2] / this.gridsrd);
    if (s.dimension === 2) this.nbin1[2] = 1;
    for (let d = 0; d < 3; d++) if (this.nbin1[d] === 0) this.nbin1[d] = 1;
    for (let d = 0; d < 3; d++) {
      this.binsize1[d] = len[d] / this.nbin1[d];
      this.bininv1[d] = 1.0 / this.binsize1[d];
    }
    this.nbins1 = this.nbin1[0] * this.nbin1[1] * this.nbin1[2];
    const nb = this.nbins1;
    if (this.binhead.length !== nb) {
      this.binhead = new Int32Array(nb);
      this.binVsum = new Float64Array(3 * nb);
      this.binCount = new Int32Array(nb);
      this.binRandom = new Float64Array(nb);
      this.binValue = new Float64Array(nb);
    }
  }

  /** fix_srd.cpp setup_bounds(): bounds for SRD movement and reneighboring (one processor, box = subdomain). */
  private setupBounds(): void {
    const s = this.sys.state;
    const nb = this.sys.nb;
    const cut = Math.max(nb.cutneighmax, nb.commCutoff);
    const onemove = this.dtBig * this.vmax;
    if (this.bigExist) {
      this.distGhost = cut + 0.5 * nb.skin;
      this.distSrd = cut - 0.5 * nb.skin - 0.5 * this.maxbigdiam;
      this.distSrdReneigh = this.distSrd - onemove;
    } else {
      this.distGhost = 0;
      this.distSrd = 0;
      let subsize = s.box.hi[0] - s.box.lo[0];
      subsize = Math.min(subsize, s.box.hi[1] - s.box.lo[1]);
      if (s.dimension === 3) subsize = Math.min(subsize, s.box.hi[2] - s.box.lo[2]);
      this.distSrdReneigh = subsize - onemove;
    }
    for (let d = 0; d < 3; d++) {
      this.srdlo[d] = s.box.lo[d] - this.distSrd;
      this.srdhi[d] = s.box.hi[d] + this.distSrd;
      this.srdloRe[d] = s.box.lo[d] - this.distSrdReneigh;
      this.srdhiRe[d] = s.box.hi[d] + this.distSrdReneigh;
    }
  }

  /** fix_srd.cpp setup_search_bins(): bins for big particles (and their copies), one processor. */
  private setupSearchBins(): void {
    const s = this.sys.state;
    const subboxlo = [0, 0, 0];
    const subboxhi = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      subboxlo[d] = s.box.lo[d] - this.distGhost;
      subboxhi[d] = s.box.hi[d] + this.distGhost;
    }
    // require integer # of bins for that volume
    for (let d = 0; d < 3; d++) this.nbin2[d] = Math.trunc((subboxhi[d] - subboxlo[d]) / this.gridsearch);
    if (s.dimension === 2) this.nbin2[2] = 1;
    for (let d = 0; d < 3; d++) if (this.nbin2[d] === 0) this.nbin2[d] = 1;
    for (let d = 0; d < 3; d++) {
      this.binsize2[d] = (subboxhi[d] - subboxlo[d]) / this.nbin2[d];
      this.bininv2[d] = 1.0 / this.binsize2[d];
    }
    // add bins on either end due to extent of big particles (radmax includes the skin movement)
    const radmax = 0.5 * this.maxbigdiam + 0.5 * this.sys.nb.skin;
    const nx = Math.trunc(radmax / this.binsize2[0]) + 1;
    const ny = Math.trunc(radmax / this.binsize2[1]) + 1;
    let nz = Math.trunc(radmax / this.binsize2[2]) + 1;
    if (s.dimension === 2) nz = 0;
    this.nbin2[0] += 2 * nx;
    this.nbin2[1] += 2 * ny;
    this.nbin2[2] += 2 * nz;
    this.blo2[0] = subboxlo[0] - nx * this.binsize2[0];
    this.blo2[1] = subboxlo[1] - ny * this.binsize2[1];
    this.blo2[2] = subboxlo[2] - nz * this.binsize2[2];
    if (s.dimension === 2) this.blo2[2] = s.box.lo[2];

    this.nbins2 = this.nbin2[0] * this.nbin2[1] * this.nbin2[2];
    if (this.binbig.length !== this.nbins2) this.binbig = Array.from({ length: this.nbins2 }, () => []);
  }

  /** fix_srd.cpp setup_search_stencil(): bin offsets a big particle can overlap. */
  private setupSearchStencil(): void {
    const s = this.sys.state;
    const radmax = 0.5 * this.maxbigdiam + 0.5 * this.sys.nb.skin;
    const radsq = radmax * radmax;
    const nx = Math.trunc(radmax / this.binsize2[0]) + 1;
    const ny = Math.trunc(radmax / this.binsize2[1]) + 1;
    let nz = Math.trunc(radmax / this.binsize2[2]) + 1;
    if (s.dimension === 2) nz = 0;
    this.stencil = [];
    for (let k = -nz; k <= nz; k++)
      for (let j = -ny; j <= ny; j++)
        for (let i = -nx; i <= nx; i++)
          if (this.binBinDistance(i, j, k) < radsq) {
            this.stencil.push([i, j, k, k * this.nbin2[1] * this.nbin2[0] + j * this.nbin2[0] + i]);
          }
  }

  /** closest squared distance between a point and bin (i, j, k) */
  private pointBinDistance(x: ArrayLike<number>, bi: number, bj: number, bk: number): number {
    const lo = [this.blo2[0] + bi * this.binsize2[0], this.blo2[1] + bj * this.binsize2[1], this.blo2[2] + bk * this.binsize2[2]];
    let rsq = 0;
    for (let d = 0; d < 3; d++) {
      const hi = lo[d] + this.binsize2[d];
      let del = 0;
      if (x[d] < lo[d]) del = lo[d] - x[d];
      else if (x[d] > hi) del = x[d] - hi;
      rsq += del * del;
    }
    return rsq;
  }

  /** closest squared distance between the central bin (0,0,0) and bin (i, j, k) */
  private binBinDistance(i: number, j: number, k: number): number {
    const off = (n: number, b: number) => (n > 0 ? (n - 1) * b : n === 0 ? 0 : (n + 1) * b);
    const dx = off(i, this.binsize2[0]);
    const dy = off(j, this.binsize2[1]);
    const dz = off(k, this.binsize2[2]);
    return dx * dx + dy * dy + dz * dz;
  }

  /**
   * fix_srd.cpp pre_neighbor(): big particles are assigned to the search bins they overlap (copies of
   * periodic images included), then SRD velocities are rotated on an SRD step.
   */
  preNeighbor(): void {
    const s = this.sys.state;
    if (this.binsrd.length < s.n) this.binsrd = new Int32Array(s.n);
    if (this.bigExist) {
      this.buildCopies();
      this.registerCopies();
    }
    if (this.reneighflag === SRD_ROTATE) this.resetVelocities();
    if (this.reneighflag === SRD_MOVE) this.reneighcount++;
    this.reneighflag = BIG_MOVE;
  }

  /**
   * big_static() and the copies: each owned big particle, plus its periodic images within the ghost cutoff
   * (native ghosts). Copies are refreshed from their owners every step (forward communication).
   */
  private buildCopies(): void {
    const s = this.sys.state;
    const order = nativeOrder(s);
    const cutghost = Math.max(this.sys.nb.cutneighmax, this.sys.nb.commCutoff);
    const skinhalf = 0.5 * this.sys.nb.skin;
    const dim = s.dimension;
    const len = [s.box.hi[0] - s.box.lo[0], s.box.hi[1] - s.box.lo[1], s.box.hi[2] - s.box.lo[2]];
    const kmax = [0, 0, 0];
    for (let d = 0; d < dim; d++) {
      kmax[d] = s.box.periodic[d] && cutghost > 0 ? Math.ceil(cutghost / len[d]) : 0;
    }
    const owner: number[] = [];
    const shift: number[] = [];
    const radius: number[] = [];
    const cutbinsq: number[] = [];
    for (let k = 0; k < s.n; k++) {
      const j = order[k];
      if (!(s.mask[j] & this.bigBit)) continue;
      const rad = this.radfactor * s.radius![j];
      const cb = (rad + skinhalf) * (rad + skinhalf);
      for (let sz = -kmax[2]; sz <= kmax[2]; sz++)
        for (let sy = -kmax[1]; sy <= kmax[1]; sy++)
          for (let sx = -kmax[0]; sx <= kmax[0]; sx++) {
            const sh = [sx, sy, sz];
            if (sh[0] !== 0 || sh[1] !== 0 || sh[2] !== 0) {
              // a periodic image is a ghost only if it lies within the ghost cutoff of the box
              let ghost = true;
              for (let d = 0; d < dim && ghost; d++) {
                if (sh[d] === 0) continue;
                const p = s.x[3 * j + d] + sh[d] * len[d];
                if (p < s.box.lo[d] - cutghost || p > s.box.hi[d] + cutghost) ghost = false;
              }
              if (!ghost) continue;
            }
            owner.push(j);
            shift.push(sx, sy, sz);
            radius.push(rad);
            cutbinsq.push(cb);
          }
    }
    const nc = owner.length;
    this.copyOwner = Int32Array.from(owner);
    this.copyShift = Int32Array.from(shift);
    this.copyRadius = Float64Array.from(radius);
    this.copyCutbinsq = Float64Array.from(cutbinsq);
    this.copyPos = new Float64Array(3 * nc);
    this.updateCopyPositions();
  }

  /** copy positions = owner position + periodic shift (forward communication of ghost positions) */
  private updateCopyPositions(): void {
    const s = this.sys.state;
    const len = [s.box.hi[0] - s.box.lo[0], s.box.hi[1] - s.box.lo[1], s.box.hi[2] - s.box.lo[2]];
    for (let c = 0; c < this.copyOwner.length; c++) {
      const j = this.copyOwner[c];
      for (let d = 0; d < 3; d++) {
        this.copyPos[3 * c + d] = s.x[3 * j + d] + this.copyShift[3 * c + d] * len[d];
      }
    }
  }

  /** register each copy in the search bins it overlaps (native pre_neighbor loop, with the stencil) */
  private registerCopies(): void {
    for (const b of this.binbig) b.length = 0;
    const nc = this.copyOwner.length;
    for (let c = 0; c < nc; c++) {
      const x = this.copyPos.subarray(3 * c, 3 * c + 3);
      const ix = Math.trunc((x[0] - this.blo2[0]) * this.bininv2[0]);
      const iy = Math.trunc((x[1] - this.blo2[1]) * this.bininv2[1]);
      const iz = Math.trunc((x[2] - this.blo2[2]) * this.bininv2[2]);
      const [nx, ny, nz] = this.nbin2;
      if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
        throw new StyleError('Fix SRD: bad search bin assignment');
      }
      const ibin = iz * ny * nx + iy * nx + ix;
      for (const [si, sj, sk, off] of this.stencil) {
        const jx = ix + si;
        const jy = iy + sj;
        const jz = iz + sk;
        if (jx < 0 || jx >= nx || jy < 0 || jy >= ny || jz < 0 || jz >= nz) {
          throw new StyleError('Fix SRD: bad stencil bin for big particle');
        }
        const rsq = this.pointBinDistance(x, jx, jy, jz);
        if (rsq < this.copyCutbinsq[c]) {
          const jbin = ibin + off;
          if (this.binbig[jbin].length === ATOMPERBIN) throw new StyleError('Fix SRD: too many big particles in bin');
          this.binbig[jbin].push(c);
        }
      }
    }
  }

  /**
   * fix_srd.cpp reset_velocities(): SRD velocities are rotated about one of six random axes inside each bin,
   * relative to the bin's average velocity; too-large velocities are rescaled. Draws one uniform per bin.
   */
  private resetVelocities(): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    const order = nativeOrder(s);
    const dim = s.dimension;
    const [nbx, nby, nbz] = this.nbin1;
    const nbins = this.nbins1;
    this.binhead.fill(-1);
    this.binnext = new Int32Array(s.n);

    // binhead = first SRD particle in each bin, binnext = next particle in the same bin
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(mask[i] & this.groupBit)) continue;
      const ix = clampInt(Math.trunc((x[3 * i] - s.box.lo[0]) * this.bininv1[0]), 0, nbx - 1);
      const iy = clampInt(Math.trunc((x[3 * i + 1] - s.box.lo[1]) * this.bininv1[1]), 0, nby - 1);
      const iz = clampInt(Math.trunc((x[3 * i + 2] - s.box.lo[2]) * this.bininv1[2]), 0, nbz - 1);
      const ibin = iz * nby * nbx + iy * nbx + ix;
      this.binnext[i] = this.binhead[ibin];
      this.binhead[ibin] = i;
    }

    // per bin: summed velocity and count; one random number per bin (every bin is owned on one processor)
    for (let b = 0; b < nbins; b++) {
      let n = 0;
      let s0 = 0;
      let s1 = 0;
      let s2 = 0;
      for (let j = this.binhead[b]; j >= 0; j = this.binnext[j]) {
        s0 += v[3 * j];
        s1 += v[3 * j + 1];
        s2 += v[3 * j + 2];
        n++;
      }
      this.binVsum[3 * b] = s0;
      this.binVsum[3 * b + 1] = s1;
      this.binVsum[3 * b + 2] = s2;
      this.binCount[b] = n;
      this.binRandom[b] = this.random.uniform();
    }

    // rotate the thermal velocity of each particle around one of six random axes, then add the average back
    const dofTemp = 1;
    this.srdBinTemp = 0.0;
    this.srdBinCount = 0;
    let axis = 2;
    for (let b = 0; b < nbins; b++) {
      this.binValue[b] = 0.0;
      const n = this.binCount[b];
      if (n === 0) continue;
      const vx = this.binVsum[3 * b] / n;
      const vy = this.binVsum[3 * b + 1] / n;
      const vz = this.binVsum[3 * b + 2] / n;
      const irandom = Math.trunc(6.0 * this.binRandom[b]);
      const sign = irandom % 2;
      if (dim === 3) axis = Math.trunc(irandom / 2);
      let vsq = 0.0;
      for (let j = this.binhead[b]; j >= 0; j = this.binnext[j]) {
        let u0 = 0;
        let u1 = 0;
        let u2 = 0;
        const dx = v[3 * j] - vx;
        const dy = v[3 * j + 1] - vy;
        const dz = v[3 * j + 2] - vz;
        if (axis === 0) {
          u0 = dx;
          u1 = sign ? dz : -dz;
          u2 = sign ? -dy : dy;
        } else if (axis === 1) {
          u1 = dy;
          u0 = sign ? dz : -dz;
          u2 = sign ? -dx : dx;
        } else {
          u2 = dz;
          u1 = sign ? dx : -dx;
          u0 = sign ? -dy : dy;
        }
        vsq += u0 * u0 + u1 * u1 + u2 * u2;
        v[3 * j] = u0 + vx;
        v[3 * j + 1] = u1 + vy;
        v[3 * j + 2] = u2 + vz;
      }
      if (n > 1) this.binValue[b] = vsq;
    }

    // SRD temperature from the thermal velocities of the bins with more than one particle
    const tfactor = s.units.mvv2e * this.massSrd / (dim * s.units.boltz);
    for (let b = 0; b < nbins; b++) {
      if (this.binCount[b] > 1) {
        this.srdBinTemp += this.binValue[b] / (this.binCount[b] - dofTemp);
        this.srdBinCount++;
      }
    }
    this.srdBinTemp *= tfactor;

    // rescale any too-large velocities
    if (this.rescaleRotate) {
      for (let k = 0; k < s.n; k++) {
        const i = order[k];
        if (!(mask[i] & this.groupBit)) continue;
        const vsq = v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2];
        if (vsq > this.vmaxsq) {
          this.nrescale++;
          scale3(v, 3 * i, this.vmax / Math.sqrt(vsq));
        }
      }
    }
  }

  /**
   * fix_srd.cpp post_force(): SRDs advect, collide with big particles (collisions_single), and
   * set the next reneighboring step (SRD moved too far, or the next step is an SRD step).
   */
  postForce(): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    const order = nativeOrder(s);
    this.statsFlag = 0;
    this.ncheck = 0;
    this.ncollide = 0;
    this.nbounce = 0;
    this.ninside = 0;

    // advect SRD particles; assign them to search bins when big particles exist
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(mask[i] & this.groupBit)) continue;
      x[3 * i] += this.dtBig * v[3 * i];
      x[3 * i + 1] += this.dtBig * v[3 * i + 1];
      x[3 * i + 2] += this.dtBig * v[3 * i + 2];
      if (this.bigExist) {
        const ix = Math.trunc((x[3 * i] - this.blo2[0]) * this.bininv2[0]);
        const iy = Math.trunc((x[3 * i + 1] - this.blo2[1]) * this.bininv2[1]);
        const iz = Math.trunc((x[3 * i + 2] - this.blo2[2]) * this.bininv2[2]);
        const [nx, ny, nz] = this.nbin2;
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
          throw new StyleError(`Fix SRD: bad bin assignment for SRD advection (SRD particle ${s.id[i]} on step ${s.step})`);
        }
        this.binsrd[i] = iz * ny * nx + iy * nx + ix;
      }
    }

    // detect collision of SRDs with big particles
    if (this.bigExist) {
      this.updateCopyPositions();
      this.collisionsSingle();
    }

    // if any SRD particle has moved too far, trigger reneighboring on the next step
    let flag = false;
    for (let k = 0; k < s.n && !flag; k++) {
      const i = order[k];
      if (!(mask[i] & this.groupBit)) continue;
      for (let d = 0; d < 3; d++) {
        if (x[3 * i + d] < this.srdloRe[d] || x[3 * i + d] > this.srdhiRe[d]) { flag = true; break; }
      }
    }
    if (flag) {
      this.nextReneighbor = s.step + 1;
      this.reneighflag = SRD_MOVE;
    }

    // if next timestep is SRD timestep, trigger reneighboring and the rotation
    if ((s.step + 1) % this.nevery === 0) {
      this.nextReneighbor = s.step + 1;
      this.reneighflag = SRD_ROTATE;
    }
  }

  /**
   * fix_srd.cpp collisions_single(): each SRD particle is checked against the big copies in its bin; on
   * overlap the collision is found in time (exact) or pushed to the surface (inexact), the SRD is slipped
   * off the sphere with a Gaussian normal velocity, and the big particle receives the momentum change.
   */
  private collisionsSingle(): void {
    const s = this.sys.state;
    const { x, v, f } = s;
    const order = nativeOrder(s);
    const { xs, vs, xb, xscoll, xbcoll, norm, vsnew } = this;

    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & this.groupBit)) continue;

      let ibin = this.binsrd[i];
      if ((this.binbig[ibin] ?? EMPTY).length === 0) continue;

      let ibounce = 0;
      let collideFlag = true;
      let dt = this.dtBig;

      while (collideFlag) {
        const cands = this.binbig[ibin] ?? EMPTY;
        const nbig = cands.length;
        if (ibounce === 0) this.ncheck += nbig;

        collideFlag = false;
        for (let m = 0; m < nbig; m++) {
          const c = cands[m];
          const j = this.copyOwner[c];
          for (let d = 0; d < 3; d++) {
            xs[d] = x[3 * i + d];
            vs[d] = v[3 * i + d];
            xb[d] = this.copyPos[3 * c + d];
          }
          const rad = this.copyRadius[c];
          const inside = insideSphere(xs, xb, rad * rad);
          if (!inside) continue;

          // the big particle's velocity and angular velocity (sphere: vb = v of the owner)
          const vb = this.bigVelocity(j);
          let tRemain: number;
          if (this.exactflag) {
            tRemain = collisionSphereExact(xs, xb, vs, vb, rad * rad, xscoll, xbcoll, norm);
          } else {
            tRemain = 0.5 * dt;
            collisionSphereInexact(xs, xb, rad, xscoll, xbcoll, norm);
          }

          if (tRemain > dt) {
            this.ninside++;
            if (this.insideflag !== INSIDE_IGNORE) {
              const msg = `SRD particle ${s.id[i]} started inside big particle ${s.id[j]} on step ${s.step} bounce ${ibounce + 1}`;
              if (this.insideflag === INSIDE_ERROR) throw new StyleError(msg);
              this.sys.warn(msg);
            }
            break;
          }

          // slip collision (fix_srd.cpp slip())
          this.slip(vs, vb, xb, this.bigOmega(j), xscoll, norm, vsnew);
          vsnew[2] = s.dimension === 2 ? 0.0 : vsnew[2];

          // check on rescaling of vsnew
          if (this.rescaleCollide) {
            const vsq = vsnew[0] * vsnew[0] + vsnew[1] * vsnew[1] + vsnew[2] * vsnew[2];
            if (vsq > this.vmaxsq) {
              this.nrescale++;
              const sc = this.vmax / Math.sqrt(vsq);
              vsnew[0] *= sc;
              vsnew[1] *= sc;
              vsnew[2] *= sc;
            }
          }

          // big particle gets the momentum change (sphere with slip: no torque)
          const factor = this.massSrd / this.dtBig / s.units.ftm2v;
          for (let d = 0; d < 3; d++) {
            const dpdt = factor * (vsnew[d] - v[3 * i + d]);
            f[3 * j + d] -= dpdt;
          }

          // update_srd(): SRD velocity, position from the collision point, and its bin
          for (let d = 0; d < 3; d++) {
            v[3 * i + d] = vsnew[d];
            x[3 * i + d] = xscoll[d] + tRemain * vsnew[d];
          }
          this.checkSrdDomain(i);
          const ix = Math.trunc((x[3 * i] - this.blo2[0]) * this.bininv2[0]);
          const iy = Math.trunc((x[3 * i + 1] - this.blo2[1]) * this.bininv2[1]);
          const iz = Math.trunc((x[3 * i + 2] - this.blo2[2]) * this.bininv2[2]);
          const [nx, ny] = this.nbin2;
          ibin = this.binsrd[i] = iz * ny * nx + iy * nx + ix;

          if (ibounce === 0) this.ncollide++;
          ibounce++;
          if (ibounce < this.maxbounceallow || this.maxbounceallow === 0) collideFlag = true;
          dt = tRemain;
          break;
        }
      }

      this.nbounce += ibounce;
      if (this.maxbounceallow && ibounce >= this.maxbounceallow) this.bouncemaxnum++;
      if (ibounce > this.bouncemax) this.bouncemax = ibounce;
    }
  }

  /** warns once per escape from the SRD domain (update_srd: "Fix srd particle moved outside valid domain") */
  private checkSrdDomain(i: number): void {
    const x = this.sys.state.x;
    for (let d = 0; d < 3; d++) {
      if (x[3 * i + d] < this.srdlo[d] || x[3 * i + d] > this.srdhi[d]) {
        this.sys.warn(`Fix srd particle moved outside valid domain: particle ${this.sys.state.id[i]} at timestep ${this.sys.state.step}`);
        return;
      }
    }
  }

  private bigVelocity(j: number): Float64Array {
    const v = this.sys.state.v;
    return v.subarray(3 * j, 3 * j + 3);
  }

  private bigOmega(j: number): Float64Array {
    const om = this.sys.state.omega;
    return om ? om.subarray(3 * j, 3 * j + 3) : new Float64Array(ZERO3);
  }

  /** fix_srd.cpp slip(): new SRD velocity from a Gaussian normal component and the surface velocity */
  private slip(
    vs: Float64Array, vb: Float64Array, xb: Float64Array, omega: Float64Array, xsurf: Float64Array,
    norm: Float64Array, vsnew: Float64Array,
  ): void {
    let r1 = 0;
    let r2 = 0;
    let vnmag = 0;
    for (;;) {
      r1 = this.sigma * this.random.gaussian();
      r2 = this.sigma * this.random.gaussian();
      vnmag = Math.sqrt(r1 * r1 + r2 * r2);
      if (vnmag * vnmag <= this.vmaxsq) break;
    }
    const vsDotN = vs[0] * norm[0] + vs[1] * norm[1] + vs[2] * norm[2];
    const tangent = this.tangent;
    tangent[0] = vs[0] - vsDotN * norm[0];
    tangent[1] = vs[1] - vsDotN * norm[1];
    tangent[2] = vs[2] - vsDotN * norm[2];

    // vsurf = velocity of collision pt = translation/rotation of BIG particle
    const vsurf = this.vsurf;
    vsurf[0] = vb[0] + omega[1] * (xsurf[2] - xb[2]) - omega[2] * (xsurf[1] - xb[1]);
    vsurf[1] = vb[1] + omega[2] * (xsurf[0] - xb[0]) - omega[0] * (xsurf[2] - xb[2]);
    vsurf[2] = vb[2] + omega[0] * (xsurf[1] - xb[1]) - omega[1] * (xsurf[0] - xb[0]);
    const vsurfDotN = vsurf[0] * norm[0] + vsurf[1] * norm[1] + vsurf[2] * norm[2];

    vsnew[0] = (vnmag + vsurfDotN) * norm[0] + tangent[0];
    vsnew[1] = (vnmag + vsurfDotN) * norm[1] + tangent[1];
    vsnew[2] = (vnmag + vsurfDotN) * norm[2] + tangent[2];
  }

  /** compute_vector(n): the 12 statistics of fix srd (f_ID[n+1] in thermo output). */
  computeVector(n: number): number {
    if (this.statsFlag === 0) {
      const stats = [
        this.ncheck, this.ncollide, this.nbounce, this.ninside, this.nrescale, this.nbins2, this.nbins1,
        this.srdBinCount, this.srdBinTemp, this.bouncemaxnum, this.bouncemax, this.reneighcount,
      ];
      // native: index 11 is never reduced into stats_all and stays 0
      for (let k = 0; k < 10; k++) this.statsAll[k] = stats[k];
      this.statsAll[10] = stats[10];
      this.statsAll[11] = 0;
      if (this.statsAll[7] !== 0.0) this.statsAll[8] /= this.statsAll[7];
      this.statsFlag = 1;
    }
    return this.statsAll[n];
  }
}

/* ---------------------------------------------------------------------- */

/** inside_sphere(): SRD point inside (inclusive) the sphere of squared radius radsq */
const insideSphere = (xs: Float64Array, xb: Float64Array, radsq: number): boolean => {
  const dx = xs[0] - xb[0];
  const dy = xs[1] - xb[1];
  const dz = xs[2] - xb[2];
  return dx * dx + dy * dy + dz * dz <= radsq;
};

/**
 * collision_sphere_exact(): time before now at which the SRD meets the sphere surface; writes the collision
 * points of the SRD and the big particle and the unit normal of the surface. Returns the time dt.
 */
const collisionSphereExact = (
  xs: Float64Array, xb: Float64Array, vs: Float64Array, vb: Float64Array, radsq: number,
  xscoll: Float64Array, xbcoll: Float64Array, norm: Float64Array,
): number => {
  const vsDotVs = vs[0] * vs[0] + vs[1] * vs[1] + vs[2] * vs[2];
  const vbDotVb = vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2];
  const vsDotVb = vs[0] * vb[0] + vs[1] * vb[1] + vs[2] * vb[2];
  const vsDotXb = vs[0] * xb[0] + vs[1] * xb[1] + vs[2] * xb[2];
  const vbDotXs = vb[0] * xs[0] + vb[1] * xs[1] + vb[2] * xs[2];
  const vsDotXs = vs[0] * xs[0] + vs[1] * xs[1] + vs[2] * xs[2];
  const vbDotXb = vb[0] * xb[0] + vb[1] * xb[1] + vb[2] * xb[2];
  const xsDotXs = xs[0] * xs[0] + xs[1] * xs[1] + xs[2] * xs[2];
  const xbDotXb = xb[0] * xb[0] + xb[1] * xb[1] + xb[2] * xb[2];
  const xsDotXb = xs[0] * xb[0] + xs[1] * xb[1] + xs[2] * xb[2];

  const a = vsDotVs + vbDotVb - 2.0 * vsDotVb;
  const b = 2.0 * (vsDotXb + vbDotXs - vsDotXs - vbDotXb);
  const c = xsDotXs + xbDotXb - 2.0 * xsDotXb - radsq;

  const dt = (-b + Math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a);

  for (let d = 0; d < 3; d++) {
    xscoll[d] = xs[d] - dt * vs[d];
    xbcoll[d] = xb[d] - dt * vb[d];
    norm[d] = xscoll[d] - xbcoll[d];
  }
  const scale = 1.0 / Math.sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
  norm[0] *= scale;
  norm[1] *= scale;
  norm[2] *= scale;
  return dt;
};

/** collision_sphere_inexact(): push the SRD to the sphere surface at the end of the step */
const collisionSphereInexact = (
  xs: Float64Array, xb: Float64Array, rad: number, xscoll: Float64Array, xbcoll: Float64Array, norm: Float64Array,
): void => {
  for (let d = 0; d < 3; d++) norm[d] = xs[d] - xb[d];
  const scale = 1.0 / Math.sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
  norm[0] *= scale;
  norm[1] *= scale;
  norm[2] *= scale;
  for (let d = 0; d < 3; d++) {
    xscoll[d] = xb[d] + rad * norm[d];
    xbcoll[d] = xb[d];
  }
};

const scale3 = (v: Float64Array, off: number, s: number): void => {
  v[off] *= s;
  v[off + 1] *= s;
  v[off + 2] *= s;
};

const clampInt = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);

const intArg = (w: string | undefined, what: string): number => {
  const n = Number(w);
  if (w === undefined || !Number.isInteger(n)) throw new StyleError(`Expected integer parameter for fix srd ${what}, got ${w ?? ''}`);
  return n;
};

const numArg = (w: string | undefined, what: string): number => {
  const n = Number(w);
  if (w === undefined || !Number.isFinite(n)) throw new StyleError(`Expected floating point parameter for fix srd ${what}, got ${w ?? ''}`);
  return n;
};

const logicalArg = (w: string | undefined, what: string): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`Expected boolean parameter for fix srd ${what}, got ${w ?? ''}`);
};
