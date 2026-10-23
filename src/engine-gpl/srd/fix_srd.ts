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
 * big particles that are spheres, ellipsoids (a sphere or line of a rigid body included), 2d lines (overlap yes
 * only: inside_line, collision_line_exact, newton_raphson, lineside) or 3d triangles (overlap yes only: inside_tri,
 * collision_tri_exact), collision slip or noslip, exact and inexact
 * (exact yes/no), overlap yes (collisions_multi), bounce N, inside error/warn/ignore, lamda, radius,
 * search, cubic, rescale, shift no/yes/possible (RanPark bin shifts, periodic boxes), tstat yes without box
 * deformation, and a fix deform that changes orthogonal box lengths with a constant rate (final, delta, scale, vel,
 * erate): the SRD streaming velocity h_rate / h_ratelo of reset_velocities, the last-bin correction and its undo,
 * the bins that follow the box (change_size), and unbiased yes with tstat yes (the bins keep their average velocity).
 * Triangle big particles (3d, overlap yes; inside_tri, collision_tri_exact, the triangle normal and omega of
 * big_dynamic) are supported for free triangles (fix nve/tri) and for triangles held in a fix rigid body (the body's
 * angmom and quaternion of set_xv/set_v; matches native: tests/oracle/w254srd_tri_rigid).
 * Triclinic boxes (domain->triclinic, fix_srd.cpp at b332f28): the velocity bins are in lamda units (binsize1 = 1/nbin1,
 * bininv1 = nbin1; binsize3 = the bin in box units for the cubic and lamda checks), the bounds and the reneighbor check
 * are in lamda units, the search bins are the real-space bounding box of the lamda sub-domain (Domain::bbox), the
 * periodic images of big particles are lattice shifts (copies at x + sx lx + sy xy + sz xz, ...), and the streaming
 * velocity of a fix deform includes the tilt rates (h_rate[3..5]). A tilted box needs fix deform remap v (native
 * "inconsistent fix deform remap option" otherwise). Verified against native: tests/oracle/w177px_srd_triclinic.
 *
 * Refused here with a StyleError naming the keyword: tstat yes with a fix deform and no unbiased yes (the biased
 * streaming bins do not match native yet), a fix deform with a time-varying rate (trate, wiggle, variable, volume).
 *
 * Walls (fix wall/srd, src/engine-gpl/colloid2/fix_wall_srd.ts; native src/SRD/fix_wall_srd.cpp) are supported with
 * or without big particles: wall_params(1) at each reneighbor and wall_params(0) at each SRD step (the wall positions,
 * velocities and force sums), the wall search bins (pre_neighbor, the walls that cover each bin up to the walltrigger),
 * the wall setup_bounds (dist_ghost = dist_srd = 4 onemove), inside_wall, collision_wall_exact and _inexact, the slip and
 * noslip collisions with a wall (slip_wall and noslip with WALL: the wall velocity in the wall normal direction is added),
 * force_wall, and the reneighbor trigger when a wall moves more than the walltrigger (half the neighbor skin).
 *
 * Browser-engine difference: native processes SRD atoms in its own local order; the engine keeps that order in
 * SimState.order (nativeOrder), and every loop over SRD particles goes through it, so the random draws happen in
 * the same sequence as the single-processor native run. Ghost images of big particles are emulated: the periodic
 * copies that overlap the search region are registered in the search bins exactly as native registers its ghosts
 * (same bins, same stencil, same counts), and forces on a copy are added to its owner.
 */
import { Fix } from '../../engine/fix/fix';
import { FixDeform } from '../../engine/fix/deform';
import { StyleError } from '../../engine/force/types';
import { inumeric, missingArgs, numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { isEllipsoid, isLineAtom, isTriAtom, LINE_BONUS, massOf, nativeOrder, TRI_BONUS, TRI_BONUS_COLS } from '../../engine/atoms';
import { bodyMotion, quatToMatrix, rotateVec } from '../../engine/compute/asphere';
import { RanMars } from '../random_mars';
import { RanPark } from '../random_park';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

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
/** big particle kinds (big_static): sphere, ellipsoid and line (fix_srd.cpp enum { SPHERE, ELLIPSOID, LINE, TRIANGLE, WALL }) */
const KIND_SPHERE = 0;
const KIND_ELLIPSOID = 1;
const KIND_LINE = 2;
const KIND_TRI = 3;
/** a wall entry of the search bins (native WALL kind); its index in binbig is copies + wall number */
const KIND_WALL = 4;

/**
 * The fix wall/srd of the system (native FixWallSRD, read by FixSRD as its wallfix): wall positions xwall, the archived
 * positions xwallhold, velocities vwall and forces fwall (3 per wall) and the wall_params() that sets them.
 */
export interface SrdWallFix extends Fix {
  readonly nwall: number;
  readonly varflag: number;
  readonly overlap: number;
  readonly wallwhich: readonly number[];
  readonly xwall: Float64Array;
  readonly xwallhold: Float64Array;
  readonly vwall: Float64Array;
  readonly fwall: Float64Array;
  wallParams(flag: 0 | 1): void;
}
const EMPTY: number[] = [];
/** fix_srd.cpp: static constexpr double TOLERANCE = 0.00001; static constexpr int MAXITER = 20; */
const LINE_TOLERANCE = 0.00001;
const LINE_MAXITER = 20;


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
  private shiftuser = 'no';

  private initflag = false;
  /** the fix wall/srd of the system (native wallfix), or null; walltrigger = half the neighbor skin (native) */
  private wallFix: SrdWallFix | null = null;
  private walltrigger = 0;
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
  /** fix deform (native deformflag): its box rates give the SRD streaming velocity (h_rate, h_ratelo) */
  private deformFix: FixDeform | null = null;
  /** native change_size: a fix with BOX_CHANGE_SIZE, so the velocity bins are rebuilt at every pre-neighbor */
  private changeSize = false;
  /** native change_shape: a fix deform with a tilt (BOX_CHANGE_SHAPE), so the search bins follow the tilted box */
  private changeShape = false;
  /** domain->triclinic: bins, bounds and the reneighbor check are in lamda units (fix_srd.cpp) */
  private triclinic = false;
  /** binsize3 = binsize1 in box units (fix_srd.cpp parameterize; the lamda binsize times the box length for triclinic) */
  private readonly binsize3 = [0, 0, 0];

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
  /** shift no/yes/possible: shiftflag = bins move with a random corner (shifts[1] in fix_srd.cpp) */
  private shiftflag = false;
  private readonly randomShift: RanPark | null;
  /** collision noslip (true) or slip (false) with big spheres */
  private readonly noslipflag: boolean;
  private readonly tstat: boolean;
  /** unbiased yes: with a box deformation the thermostat keeps the bin average (no streaming velocity) */
  private readonly unbiasflag: boolean;

  /** overlap yes: every overlapping big particle is considered, the first collision applied (collisions_multi) */
  private readonly overlapflag: boolean;
  private readonly savedXscoll = new Float64Array(3);
  private readonly savedXbcoll = new Float64Array(3);
  private readonly savedNorm = new Float64Array(3);
  /** collision scratch vectors */
  private readonly xs = new Float64Array(3);
  private readonly vs = new Float64Array(3);
  private readonly xb = new Float64Array(3);
  private readonly xscoll = new Float64Array(3);
  private readonly xbcoll = new Float64Array(3);
  private readonly norm = new Float64Array(3);
  private readonly vsnew = new Float64Array(3);
  /** big_static()/big_dynamic() per atom: kind (0 sphere, 1 ellipsoid), 1/(semi-axis)^2 (3), body axes ex ey ez (9), omega (3) */
  private bigKind = new Uint8Array(0);
  private bigInv = new Float64Array(0);
  private bigAxes = new Float64Array(0);
  private bigOmegaAll = new Float64Array(0);
  /** line big particles: segment length (big_static) and orientation theta (big_dynamic, from the line bonus) */
  private bigLength = new Float64Array(0);
  private bigTheta = new Float64Array(0);
  /** triangle big particles: body-frame corners c1 c2 c3 (9 per atom, TRI_BONUS), body normal (3), space normal (3) */
  private bigCorners = new Float64Array(0);
  private bigNormBody = new Float64Array(0);
  private bigNorm = new Float64Array(0);
  /** inside_tri() state: collision point, big particle point and normal at the collision time (xsc, xbc, nbc) */
  private readonly tsc = new Float64Array(3);
  private readonly tbc = new Float64Array(3);
  private readonly tnbc = new Float64Array(3);
  /** inside_line() state (fix_srd.cpp members xs0, xs1, xb0, xb1, theta0, theta1, tfraction, xsc, xbc, nbc) */
  private readonly lxs0 = new Float64Array(2);
  private readonly lxs1 = new Float64Array(2);
  private readonly lxb0 = new Float64Array(2);
  private readonly lxb1 = new Float64Array(2);
  private ltheta0 = 0;
  private ltheta1 = 0;
  private ltfraction = 0;
  private readonly lxsc = new Float64Array(2);
  private readonly lxbc = new Float64Array(2);
  private readonly lnbc = new Float64Array(2);
  private readonly tangent = new Float64Array(3);
  private readonly tangent1 = new Float64Array(3);
  private readonly tangent2 = new Float64Array(3);
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
    let shiftSeed = 0;
    let tstatUser = false;
    let unbias = false;
    let overlap = false;
    let iarg = 5;
    const need = (k: number, key: string) => {
      // As native LAMMPS does (src/SRD/fix_srd.cpp FixSRD::FixSRD at b332f28): "Illegal fix srd {keyword} command: missing argument(s)"
      if (iarg + k > args.length) throw missingArgs(`fix srd ${key}`);
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
        this.shiftuser = args[iarg + 1];
        if (this.shiftuser !== 'no' && this.shiftuser !== 'yes' && this.shiftuser !== 'possible') {
          throw new StyleError(`Unknown fix srd shift argument ${this.shiftuser}`);
        }
        shiftSeed = intArg(args[iarg + 2], 'shift seed');
        iarg += 3;
      } else if (key === 'tstat') {
        need(2, key);
        tstatUser = logicalArg(args[iarg + 1], 'tstat');
        iarg += 2;
      } else if (key === 'unbiased') {
        need(2, key);
        unbias = logicalArg(args[iarg + 1], 'unbiased');
        if (unbias && !tstatUser) throw new StyleError('Must use SRD thermostat when enabling unbiased thermostatting');
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

    this.noslipflag = collide === 'noslip';
    this.overlapflag = overlap;
    this.tstat = tstatUser;
    this.unbiasflag = unbias;
    // initialize Marsaglia RNG with processor-unique seed (one processor: seed + 0)
    this.random = new RanMars(seed);
    // shift yes/possible: RanPark with the shift seed (fix_srd.cpp: "randomshift = new RanPark(lmp, shiftseed)")
    if (this.shiftuser !== 'no') {
      if (shiftSeed <= 0) throw new StyleError('Invalid seed for Park random # generator');
      this.randomShift = new RanPark(shiftSeed);
    } else {
      this.randomShift = null;
    }
  }

  /** fix_srd.cpp setmask(): PRE_NEIGHBOR and POST_FORCE (the engine calls preNeighbor and postForce). */

  /** Forced reneighboring of the SRD step (Neighbor::decide: fix next_reneighbor == ntimestep). */
  reneighborDue(step: number): boolean {
    return step === this.nextReneighbor;
  }

  init(): void {
    const s = this.sys.state;
    this.triclinic = s.box.triclinic;
    if (this.initflag && s.dt !== this.dtBig) throw new StyleError('Cannot change timestep once fix srd is set up');
    if (this.bigExist) {
      this.checkBig();
      this.bigStatic();
    }
    // fix_srd.cpp init(): "Fix srd no-slip requires atom attribute torque"
    if (this.bigExist && this.noslipflag && !s.torque) throw new StyleError('Fix srd no-slip requires atom attribute torque');
    // fix_srd.cpp init(): wallexist = 1 if an SRD wall is defined (get_fix_by_style("^wall/srd"), one at most)
    const wallFixes = this.sys.fixes.filter((f) => f.style === 'wall/srd');
    if (wallFixes.length > 1) throw new StyleError('Cannot use fix wall/srd more than once');
    this.wallFix = wallFixes.length === 1 ? (wallFixes[0] as SrdWallFix) : null;
    if (this.wallFix) {
      this.walltrigger = 0.5 * this.sys.nb.skin;
      if (this.wallFix.overlap && !this.overlapflag) this.sys.warn('Fix SRD walls overlap but fix srd overlap not set');
    }
    // fix_srd.cpp init(): "deformflag = 1" for a fix deform, and "change_size" for a fix that changes a box length
    // (the velocity bins then follow the box, see preNeighbor). The streaming correction itself is in resetVelocities.
    this.deformFix = null;
    this.changeSize = false;
    this.changeShape = false;
    for (const f of this.sys.fixes) {
      if (!(f instanceof FixDeform)) continue;
      if (this.deformFix) throw new StyleError('More than one fix deform');
      this.deformFix = f;
      // fix_srd.cpp init(): a shape (tilt) change with a remap other than v is an error
      const tilted = f.hasTilt();
      if (tilted && f.remapFlag() !== 'v') throw new StyleError('Using fix srd with inconsistent fix deform remap option');
      if (f.hasLengthChange()) this.changeSize = true;
      if (tilted) this.changeShape = true;
    }
    if (this.deformFix) this.deformFix.streamingRates();
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
    if (this.bigExist || this.wallFix) {
      this.setupSearchBins();
      this.setupSearchStencil();
    } else {
      this.nbins2 = 0;
    }
    // perform first binning of SRD and big particles; no SRD rotation in setup
    this.reneighflag = BIG_MOVE;
    this.preNeighbor();
  }

  /**
   * Big particles must be spheres, ellipsoids, lines or triangles (free or in a fix rigid body). fix_srd.cpp init():
   * "Cannot use lines with fix srd unless overlap is set"; lines and tris need the atom torque.
   */
  private checkBig(): void {
    const s = this.sys.state;
    let anyLines = false;
    let anyTris = false;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.bigBit)) continue;
      if (isTriAtom(s, i)) anyTris = true;
      if (isLineAtom(s, i)) anyLines = true;
    }
    if (anyLines && !this.overlapflag) throw new StyleError('Cannot use lines with fix srd unless overlap is set');
    if (anyLines && !s.torque) throw new StyleError('Fix srd big particles that are lines require atom attribute torque');
    // fix_srd.cpp init(): "Cannot use tris with fix srd unless overlap is set"
    if (anyTris && !this.overlapflag) throw new StyleError('Cannot use tris with fix srd unless overlap is set');
    if (anyTris && !s.torque) throw new StyleError('Fix srd big particles that are tris require atom attribute torque');
  }

  /**
   * fix_srd.cpp big_static(): the kind of each big particle and its inverse squared semi-axes, arad = radfactor
   * times the semi-axis (the shape of atom_style ellipsoid is stored as semi-axes).
   */
  private bigStatic(): void {
    const s = this.sys.state;
    const n = s.n;
    if (this.bigKind.length !== n) {
      this.bigKind = new Uint8Array(n);
      this.bigInv = new Float64Array(3 * n);
      this.bigAxes = new Float64Array(9 * n);
      this.bigOmegaAll = new Float64Array(3 * n);
      this.bigLength = new Float64Array(n);
      this.bigTheta = new Float64Array(n);
      this.bigCorners = new Float64Array(9 * n);
      this.bigNormBody = new Float64Array(3 * n);
      this.bigNorm = new Float64Array(3 * n);
    }
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.bigBit)) continue;
      if (isLineAtom(s, i)) {
        // fix_srd.cpp big_static(), LINE: "set length and cutoff based on 1/2 length" (no radfactor)
        this.bigKind[i] = KIND_LINE;
        this.bigLength[i] = s.custom.get(LINE_BONUS)!.data[2 * i];
      } else if (isEllipsoid(s, i)) {
        this.bigKind[i] = KIND_ELLIPSOID;
        const sh = s.shape!;
        for (let d = 0; d < 3; d++) {
          const arad = this.radfactor * sh[3 * i + d];
          this.bigInv[3 * i + d] = 1.0 / (arad * arad);
        }
      } else if (isTriAtom(s, i)) {
        // fix_srd.cpp big_static(), TRIANGLE: normbody = (c2 - c1) x (c3 - c1) from the body-frame corners
        this.bigKind[i] = KIND_TRI;
        const tb = s.custom.get(TRI_BONUS)!.data;
        const c = this.bigCorners;
        for (let d = 0; d < 9; d++) c[9 * i + d] = tb[TRI_BONUS_COLS * i + d];
        const e1 = [c[9 * i + 3] - c[9 * i], c[9 * i + 4] - c[9 * i + 1], c[9 * i + 5] - c[9 * i + 2]];
        const e2 = [c[9 * i + 6] - c[9 * i], c[9 * i + 7] - c[9 * i + 1], c[9 * i + 8] - c[9 * i + 2]];
        this.bigNormBody[3 * i] = e1[1] * e2[2] - e1[2] * e2[1];
        this.bigNormBody[3 * i + 1] = e1[2] * e2[0] - e1[0] * e2[2];
        this.bigNormBody[3 * i + 2] = e1[0] * e2[1] - e1[1] * e2[0];
      } else {
        this.bigKind[i] = KIND_SPHERE;
      }
    }
  }

  /** big_dynamic(): body axes and angular velocity of each big particle, from its current quaternion and angmom */
  private bigDynamic(): void {
    const s = this.sys.state;
    const omega = s.omega;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.bigBit)) continue;
      if (this.bigKind[i] === KIND_ELLIPSOID) {
        const q = s.quat!;
        const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
        const ax = this.bigAxes;
        const o = 9 * i;
        // ex, ey, ez: columns of R
        ax[o] = R[0]; ax[o + 1] = R[3]; ax[o + 2] = R[6];
        ax[o + 3] = R[1]; ax[o + 4] = R[4]; ax[o + 5] = R[7];
        ax[o + 6] = R[2]; ax[o + 7] = R[5]; ax[o + 8] = R[8];
        const { wb } = bodyMotion(s, i);
        const w = rotateVec(R, wb[0], wb[1], wb[2]);
        this.bigOmegaAll[3 * i] = w[0];
        this.bigOmegaAll[3 * i + 1] = w[1];
        this.bigOmegaAll[3 * i + 2] = w[2];
      } else {
        // fix_srd.cpp big_dynamic(), LINE: "set omega from atom->omega directly" and theta from the line bonus
        if (this.bigKind[i] === KIND_LINE) this.bigTheta[i] = this.sys.state.custom.get(LINE_BONUS)!.data[2 * i + 1];
        if (this.bigKind[i] === KIND_TRI) this.triDynamic(i);
        else if (omega) for (let d = 0; d < 3; d++) this.bigOmegaAll[3 * i + d] = omega[3 * i + d];
      }
    }
  }

  /**
   * fix_srd.cpp big_dynamic(), TRIANGLE: ex, ey, ez from the quaternion; omega from the angular momentum and the
   * principal moments (MathExtra::angmom_to_omega); the space-frame unit normal from the body normal (matvec, norm3).
   */
  private triDynamic(i: number): void {
    const s = this.sys.state;
    const q = s.quat!;
    const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
    const ax = this.bigAxes;
    const o = 9 * i;
    ax[o] = R[0]; ax[o + 1] = R[3]; ax[o + 2] = R[6];
    ax[o + 3] = R[1]; ax[o + 4] = R[4]; ax[o + 5] = R[7];
    ax[o + 6] = R[2]; ax[o + 7] = R[5]; ax[o + 8] = R[8];
    const tb = s.custom.get(TRI_BONUS)!.data;
    const I = [tb[TRI_BONUS_COLS * i + 9], tb[TRI_BONUS_COLS * i + 10], tb[TRI_BONUS_COLS * i + 11]];
    const L = s.angmom!;
    const wbody = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      // the axis e_k is stored as three consecutive values (ex = ax[0..2], ey = ax[3..5], ez = ax[6..8])
      const dot = L[3 * i] * ax[o + 3 * k] + L[3 * i + 1] * ax[o + 3 * k + 1] + L[3 * i + 2] * ax[o + 3 * k + 2];
      wbody[k] = I[k] === 0.0 ? 0.0 : dot / I[k];
    }
    for (let d = 0; d < 3; d++) {
      this.bigOmegaAll[3 * i + d] = wbody[0] * ax[o + d] + wbody[1] * ax[o + 3 + d] + wbody[2] * ax[o + 6 + d];
    }
    const nb = this.bigNormBody;
    const n = [0, 0, 0];
    for (let d = 0; d < 3; d++) n[d] = ax[o + d] * nb[3 * i] + ax[o + 3 + d] * nb[3 * i + 1] + ax[o + 6 + d] * nb[3 * i + 2];
    const len = Math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (let d = 0; d < 3; d++) this.bigNorm[3 * i + d] = n[d] / len;
  }

  /** big radius for the search bins and the ghost cutoff: radfactor times the radius, or the longest semi-axis */
  private bigRadius(j: number): number {
    const s = this.sys.state;
    // fix_srd.cpp big_static(), LINE: rad = 0.5 * length (no radfactor)
    if (this.bigKind[j] === KIND_LINE) return 0.5 * this.bigLength[j];
    // fix_srd.cpp big_static(), TRIANGLE: rad = the longest body-frame corner (no radfactor)
    if (this.bigKind[j] === KIND_TRI) {
      const c = this.bigCorners;
      let rad = 0;
      for (let k = 0; k < 3; k++) rad = Math.max(rad, Math.hypot(c[9 * j + 3 * k], c[9 * j + 3 * k + 1], c[9 * j + 3 * k + 2]));
      return rad;
    }
    if (this.bigKind[j] === KIND_ELLIPSOID) {
      const sh = s.shape!;
      return this.radfactor * Math.max(sh[3 * j], sh[3 * j + 1], sh[3 * j + 2]);
    }
    return this.radfactor * s.radius![j];
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
        if (isLineAtom(s, i)) {
          // fix_srd.cpp parameterize(): a line contributes its length (not 2*radius)
          const len = s.custom.get(LINE_BONUS)!.data[2 * i];
          maxd = Math.max(maxd, len);
          mind = Math.min(mind, len);
          continue;
        }
        if (isEllipsoid(s, i)) {
          // fix_srd.cpp parameterize(): the diameters 2*shape of an ellipsoid
          for (let d = 0; d < 3; d++) {
            maxd = Math.max(maxd, 2.0 * s.shape![3 * i + d]);
            mind = Math.min(mind, 2.0 * s.shape![3 * i + d]);
          }
          continue;
        }
        if (isTriAtom(s, i)) {
          // fix_srd.cpp parameterize(): the longest corner length of a triangle (not 2*radius)
          const tb = s.custom.get(TRI_BONUS)!.data;
          let len = 0;
          for (let k = 0; k < 3; k++) {
            const b = TRI_BONUS_COLS * i + 3 * k;
            len = Math.max(len, Math.hypot(tb[b], tb[b + 1], tb[b + 2]));
          }
          maxd = Math.max(maxd, len);
          mind = Math.min(mind, len);
          continue;
        }
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

    // SRD bins must be cubic within tolerance, and close to the requested size (binsize3: in box units)
    const b3 = this.binsize3;
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

    // lamda must be at least 0.6 of the SRD grid size; shift possible turns on bin shifting in that case
    let maxgridsrd = Math.max(b3[0], b3[1]);
    if (dim === 3) maxgridsrd = Math.max(maxgridsrd, b3[2]);
    this.shiftflag = false;
    if (this.lamda < 0.6 * maxgridsrd && this.shiftuser === 'no') {
      throw new StyleError('Fix srd lamda must be >= 0.6 of SRD grid size');
    } else if (this.lamda < 0.6 * maxgridsrd && this.shiftuser === 'possible') {
      this.shiftflag = true;
      this.sys.warn('SRD bin shifting turned on due to small lamda');
    } else if (this.shiftuser === 'yes') {
      this.shiftflag = true;
    }
  }

  private cubicCheck(msg: string): void {
    if (this.cubicflag === CUBIC_ERROR) throw new StyleError(msg);
    this.sys.warn(msg);
  }

  /**
   * fix_srd.cpp setup_velocity_bins() for one processor and no shift. Orthogonal: bins of the box lengths. Triclinic: the
   * bins are in lamda units (binsize1 = 1/nbin1, bininv1 = nbin1) and binsize3 is the bin in box units.
   */
  private setupVelocityBins(): void {
    const s = this.sys.state;
    const len = [s.box.hi[0] - s.box.lo[0], s.box.hi[1] - s.box.lo[1], s.box.hi[2] - s.box.lo[2]];
    this.nbin1[0] = Math.round(len[0] / this.gridsrd);
    this.nbin1[1] = Math.round(len[1] / this.gridsrd);
    this.nbin1[2] = Math.round(len[2] / this.gridsrd);
    if (s.dimension === 2) this.nbin1[2] = 1;
    for (let d = 0; d < 3; d++) if (this.nbin1[d] === 0) this.nbin1[d] = 1;
    for (let d = 0; d < 3; d++) {
      if (this.triclinic) {
        this.binsize1[d] = 1.0 / this.nbin1[d];
        this.bininv1[d] = this.nbin1[d];
        this.binsize3[d] = this.binsize1[d] * len[d];
      } else {
        this.binsize1[d] = len[d] / this.nbin1[d];
        this.bininv1[d] = 1.0 / this.binsize1[d];
        this.binsize3[d] = this.binsize1[d];
      }
    }
    this.nbins1 = this.nbin1[0] * this.nbin1[1] * this.nbin1[2];
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
    } else if (this.wallFix) {
      // fix_srd.cpp setup_bounds(): walls and no big particles
      this.distGhost = 4 * onemove;
      this.distSrd = 4 * onemove;
      this.distSrdReneigh = 4 * onemove - onemove;
    } else {
      this.distGhost = 0;
      this.distSrd = 0;
      // fix_srd.cpp setup_bounds(): the perpendicular distance between the sub-domain faces (lamda units scaled back)
      let subsize: number;
      if (this.triclinic) {
        const { len } = boxMatrix(s.box.lo, s.box.hi, s.box.tilt);
        subsize = 1.0 / len[0];
        subsize = Math.min(subsize, 1.0 / len[1]);
        if (s.dimension === 3) subsize = Math.min(subsize, 1.0 / len[2]);
      } else {
        subsize = s.box.hi[0] - s.box.lo[0];
        subsize = Math.min(subsize, s.box.hi[1] - s.box.lo[1]);
        if (s.dimension === 3) subsize = Math.min(subsize, s.box.hi[2] - s.box.lo[2]);
      }
      this.distSrdReneigh = subsize - onemove;
    }
    if (this.triclinic) {
      // fix_srd.cpp setup_bounds(): for triclinic boxes the bounds are in lamda units (sublo_lamda 0, subhi_lamda 1)
      const { len } = boxMatrix(s.box.lo, s.box.hi, s.box.tilt);
      for (let d = 0; d < 3; d++) {
        this.srdlo[d] = 0 - this.distSrd * len[d];
        this.srdhi[d] = 1 + this.distSrd * len[d];
        this.srdloRe[d] = 0 - this.distSrdReneigh * len[d];
        this.srdhiRe[d] = 1 + this.distSrdReneigh * len[d];
      }
      return;
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
    if (this.triclinic) {
      // fix_srd.cpp setup_search_bins(): the lamda sub-domain grown by dist_ghost, then Domain::bbox (the real-space box
      // around its eight corners, lamda2x at each corner as native)
      const { h, len } = boxMatrix(s.box.lo, s.box.hi, s.box.tilt);
      const lo = [0 - this.distGhost * len[0], 0 - this.distGhost * len[1], 0 - this.distGhost * len[2]];
      const hi = [1 + this.distGhost * len[0], 1 + this.distGhost * len[1], 1 + this.distGhost * len[2]];
      const bl = [BIG, BIG, BIG], bh = [-BIG, -BIG, -BIG];
      const c = new Float64Array(3);
      for (let k = 0; k < 8; k++) {
        c[0] = k & 1 ? hi[0] : lo[0];
        c[1] = k & 2 ? hi[1] : lo[1];
        c[2] = k & 4 ? hi[2] : lo[2];
        lamda2xAll(c, 1, s.box.lo, h);
        for (let d = 0; d < 3; d++) {
          bl[d] = Math.min(bl[d], c[d]);
          bh[d] = Math.max(bh[d], c[d]);
        }
      }
      for (let d = 0; d < 3; d++) {
        subboxlo[d] = bl[d];
        subboxhi[d] = bh[d];
      }
    } else {
      for (let d = 0; d < 3; d++) {
        subboxlo[d] = s.box.lo[d] - this.distGhost;
        subboxhi[d] = s.box.hi[d] + this.distGhost;
      }
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
    if (this.bigExist) this.bigStatic();
    // fix_srd.cpp pre_neighbor(): wall_params(1) sets the wall positions and archives them for the reneighbor trigger
    if (this.wallFix) this.wallFix.wallParams(1);
    // fix_srd.cpp pre_neighbor(): "if simulation box size changes, reset velocity bins" (change_size, set by a fix
    // deform that changes a length); the search bins follow the box as well when big particles exist
    // fix_srd.cpp pre_neighbor(): setup_bounds and the velocity bins on change_size, the search bins on change_size
    // or change_shape (a tilted triclinic box moves the bins every reneighbor)
    if (this.changeSize) {
      this.setupBounds();
      this.setupVelocityBins();
    }
    if ((this.changeSize || this.changeShape) && (this.bigExist || this.wallFix)) {
      this.setupSearchBins();
      this.setupSearchStencil();
    }
    // native: the search bins of big particles and walls are emptied before they are filled
    if (this.bigExist || this.wallFix) for (const b of this.binbig) b.length = 0;
    if (this.bigExist) {
      this.buildCopies();
      this.registerCopies();
    }
    if (this.wallFix) this.registerWalls();
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
    // triclinic: the periodic images are lattice shifts, i.e. integer lamda shifts; the ghost cutoff is in lamda units
    const geom = this.triclinic ? boxMatrix(s.box.lo, s.box.hi, s.box.tilt) : null;
    const cutlam = [0, 0, 0];
    const kmax = [0, 0, 0];
    for (let d = 0; d < dim; d++) {
      cutlam[d] = geom ? cutghost * geom.len[d] : cutghost;
      const span = geom ? 1 : len[d];
      kmax[d] = s.box.periodic[d] && cutghost > 0 ? Math.ceil(cutlam[d] / span) : 0;
    }
    const lam = new Float64Array(3);
    const owner: number[] = [];
    const shift: number[] = [];
    const radius: number[] = [];
    const cutbinsq: number[] = [];
    for (let k = 0; k < s.n; k++) {
      const j = order[k];
      if (!(s.mask[j] & this.bigBit)) continue;
      const rad = this.bigRadius(j);
      const cb = (rad + skinhalf) * (rad + skinhalf);
      // the owner's lamda position (triclinic ghost test); the lamda shift of an image is its integer shift
      if (geom) {
        lam.set(s.x.subarray(3 * j, 3 * j + 3));
        x2lamdaAll(lam, 1, s.box.lo, geom.hinv);
      }
      for (let sz = -kmax[2]; sz <= kmax[2]; sz++)
        for (let sy = -kmax[1]; sy <= kmax[1]; sy++)
          for (let sx = -kmax[0]; sx <= kmax[0]; sx++) {
            const sh = [sx, sy, sz];
            if (sh[0] !== 0 || sh[1] !== 0 || sh[2] !== 0) {
              // a periodic image is a ghost only if it lies within the ghost cutoff of the box
              let ghost = true;
              for (let d = 0; d < dim && ghost; d++) {
                if (sh[d] === 0) continue;
                if (geom) {
                  const p = lam[d] + sh[d];
                  if (p < 0 - cutlam[d] || p > 1 + cutlam[d]) ghost = false;
                  continue;
                }
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
    // the image of shift (sx, sy, sz): x += sx lx + sy xy + sz xz, y += sy ly + sz yz, z += sz lz (Comm::borders, triclinic)
    const { h } = boxMatrix(s.box.lo, s.box.hi, s.box.tilt);
    for (let c = 0; c < this.copyOwner.length; c++) {
      const j = this.copyOwner[c];
      const sx = this.copyShift[3 * c], sy = this.copyShift[3 * c + 1], sz = this.copyShift[3 * c + 2];
      this.copyPos[3 * c] = s.x[3 * j] + (sx * h[0] + sy * h[5] + sz * h[4]);
      this.copyPos[3 * c + 1] = s.x[3 * j + 1] + (sy * h[1] + sz * h[3]);
      this.copyPos[3 * c + 2] = s.x[3 * j + 2] + sz * h[2];
    }
  }

  /** register each copy in the search bins it overlaps (native pre_neighbor loop, with the stencil) */
  private registerCopies(): void {
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
   * relative to the bin's average velocity; too-large velocities are rescaled. With shift yes or possible, the
   * bin grid is first moved by a random corner (RanPark, shift seed), and the bins that duplicate a periodic
   * image are merged into their partner bin (the upper ghost bins are not owners and draw no random number).
   * Draws one uniform per owner bin, in bin order.
   */
  private resetVelocities(): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    const order = nativeOrder(s);
    const dim = s.dimension;
    const per = s.box.periodic;

    // the shifted grid corner (fix_srd.cpp: shifts[1].corner, drawn x, y, then z in 3d)
    // triclinic: the origin is boxlo_lamda (0), and the bins are in lamda units (fix_srd.cpp reset_velocities)
    const tri = this.triclinic;
    const geomR = tri ? boxMatrix(s.box.lo, s.box.hi, s.box.tilt) : null;
    const corner = tri ? [0, 0, 0] : [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
    if (this.shiftflag && this.randomShift) {
      corner[0] = corner[0] - this.binsize1[0] * this.randomShift.uniform();
      corner[1] = corner[1] - this.binsize1[1] * this.randomShift.uniform();
      if (dim === 3) corner[2] = corner[2] - this.binsize1[2] * this.randomShift.uniform();
    }
    // bin range per dimension: [0, nb-1]; the shifted grid runs to the bin holding the upper box face
    const nb = [1, 1, 1];
    for (let d = 0; d < 3; d++) {
      let hi = this.nbin1[d] - 1;
      if (this.shiftflag) hi = Math.trunc(((tri ? 1 : s.box.hi[d]) - corner[d]) * this.bininv1[d]);
      if (d === 2 && dim === 2) hi = 0;
      nb[d] = hi + 1;
    }
    const [nbx, nby, nbz] = nb;
    const nbins = nbx * nby * nbz;
    const strides = [1, nbx, nbx * nby];
    // triclinic: the positions are taken to lamda units for the binning (native x2lamda(nlocal)) and restored after it
    // (native lamda2x(nlocal)), so x keeps the round-off of the round trip as native does
    const lamB = tri && geomR ? Float64Array.from(x.subarray(0, 3 * s.n)) : null;
    if (lamB && geomR) x2lamdaAll(lamB, s.n, s.box.lo, geomR.hinv);
    let pos: Float64Array = lamB ?? x;
    const cellOf = (i: number): [number, number, number] => [
      clampInt(Math.trunc((pos[3 * i] - corner[0]) * this.bininv1[0]), 0, nbx - 1),
      clampInt(Math.trunc((pos[3 * i + 1] - corner[1]) * this.bininv1[1]), 0, nby - 1),
      clampInt(Math.trunc((pos[3 * i + 2] - corner[2]) * this.bininv1[2]), 0, nbz - 1),
    ];
    // fix_srd.cpp reset_velocities(): h_rate = [lx, ly, lz, yz, xz, xy] rates and h_ratelo of fix deform (the tilt
    // rates are zero in an orthogonal box)
    const deformRates = this.deformFix ? this.deformFix.streamingRates() : null;
    const hRate = deformRates ? deformRates.rate : [0, 0, 0, 0, 0, 0];
    const hRateLo = deformRates ? deformRates.ratelo : [0, 0, 0];

    // binhead = first SRD particle in each bin, binnext = next particle in the same bin
    const binhead = new Int32Array(nbins).fill(-1);
    const binnext = new Int32Array(s.n);
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(mask[i] & this.groupBit)) continue;
      const [ix, iy, iz] = cellOf(i);
      // native "shift velocities in last bins": a particle in the extra bin (ix == nbin1x) of a periodic dimension
      // loses the box rate of that dimension
      if (deformRates) {
        if (per[0] && ix === this.nbin1[0]) v[3 * i] -= hRate[0];
        if (per[1] && iy === this.nbin1[1]) {
          v[3 * i] -= hRate[5];
          v[3 * i + 1] -= hRate[1];
        }
        if (per[2] && iz === this.nbin1[2]) {
          v[3 * i] -= hRate[4];
          v[3 * i + 1] -= hRate[3];
          v[3 * i + 2] -= hRate[2];
        }
      }
      const ibin = iz * nby * nbx + iy * nbx + ix;
      binnext[i] = binhead[ibin];
      binhead[ibin] = i;
    }
    if (lamB && geomR) {
      lamda2xAll(lamB, s.n, s.box.lo, geomR.h);
      x.set(lamB, 0);
    }

    // owners: with shifting, the last bin of a periodic dimension duplicates the first one (not an owner)
    const owner = new Uint8Array(nbins).fill(1);
    if (this.shiftflag) {
      for (let b = 0; b < nbins; b++) {
        const ix = b % nbx;
        const iy = Math.trunc(b / nbx) % nby;
        const iz = Math.trunc(b / (nbx * nby));
        if ((per[0] && ix === nbx - 1) || (per[1] && iy === nby - 1) || (dim === 3 && per[2] && iz === nbz - 1)) owner[b] = 0;
      }
    }

    // per bin: count, summed velocity and one random number for owners (bin order)
    const cnt = new Int32Array(nbins);
    const vsum = new Float64Array(3 * nbins);
    const rnd = new Float64Array(nbins);
    const value = new Float64Array(nbins);
    for (let b = 0; b < nbins; b++) {
      let n = 0;
      let s0 = 0;
      let s1 = 0;
      let s2 = 0;
      for (let j = binhead[b]; j >= 0; j = binnext[j]) {
        s0 += v[3 * j];
        s1 += v[3 * j + 1];
        s2 += v[3 * j + 2];
        n++;
      }
      vsum[3 * b] = s0;
      vsum[3 * b + 1] = s1;
      vsum[3 * b + 2] = s2;
      cnt[b] = n;
      if (owner[b]) rnd[b] = this.random.uniform();
    }

    // sum the bins that are periodic images of each other (fix_srd.cpp vbin_comm / xbin_comm, one processor)
    const combine = (arrs: Array<{ a: Float64Array | Int32Array; w: number }>): void => {
      for (let d = 0; d < dim; d++) {
        if (!this.shiftflag || !per[d]) continue;
        for (let b = 0; b < nbins; b++) {
          const coord = Math.trunc(b / strides[d]) % nb[d];
          if (coord !== 0) continue;
          const p = b + (nb[d] - 1) * strides[d];
          for (const { a, w } of arrs) {
            for (let c = 0; c < w; c++) {
              const t = a[w * b + c];
              a[w * b + c] = t + a[w * p + c];
              a[w * p + c] = a[w * p + c] + t;
            }
          }
        }
      }
    };
    combine([{ a: cnt, w: 1 }, { a: vsum, w: 3 }, { a: rnd, w: 1 }]);

    // rotate the thermal velocity of each particle around one of six random axes, then add the average back
    const tfactor = s.units.mvv2e * this.massSrd / (dim * s.units.boltz);
    // fix_srd.cpp reset_velocities(): "no deformation, no tstat: dof_temp = 1; yes deformation, yes tstat:
    // dof_temp = dof_tstat = 0" (unbiased yes: the biased deformation case does not apply, dof 1)
    const dofTemp = this.tstat && deformRates && !this.unbiasflag ? 0 : 1;
    const dofTstat = dofTemp;
    this.srdBinTemp = 0.0;
    this.srdBinCount = 0;
    let axis = 2;
    for (let b = 0; b < nbins; b++) {
      const n = cnt[b];
      if (n === 0) continue;
      const vx = vsum[3 * b] / n;
      const vy = vsum[3 * b + 1] / n;
      const vz = vsum[3 * b + 2] / n;
      vsum[3 * b] = vx;
      vsum[3 * b + 1] = vy;
      vsum[3 * b + 2] = vz;
      const irandom = Math.trunc(6.0 * rnd[b]);
      const sign = irandom % 2;
      if (dim === 3) axis = Math.trunc(irandom / 2);
      let vsq = 0.0;
      for (let j = binhead[b]; j >= 0; j = binnext[j]) {
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
      if (n > 1) value[b] = vsq;
    }
    combine([{ a: value, w: 1 }]);

    // thermostat: rescale the thermal velocities of each bin to the SRD temperature. With a box deformation the
    // bin streams with the box: vstream = h_rate * xctr + h_ratelo, where xctr is the bin centre as
    // setup_velocity_shift() sets it (corner + (i + 0.5) / nbin1 per dimension, "in lamda units")
    if (this.tstat) {
      for (let b = 0; b < nbins; b++) {
        const n = cnt[b];
        if (n <= 1) continue;
        let sx = vsum[3 * b];
        let sy = vsum[3 * b + 1];
        let sz = vsum[3 * b + 2];
        if (deformRates && !this.unbiasflag) {
          const bx = b % nbx;
          const by = Math.trunc(b / nbx) % nby;
          const bz = Math.trunc(b / (nbx * nby));
          const xc0 = corner[0] + (bx + 0.5) / this.nbin1[0];
          const xc1 = corner[1] + (by + 0.5) / this.nbin1[1];
          const xc2 = corner[2] + (bz + 0.5) / this.nbin1[2];
          // As native LAMMPS does (src/SRD/fix_srd.cpp reset_velocities at b332f28): vstream[0] = h_rate[0]*x0 + h_rate[5]*x1
          // + h_rate[4]*x2 + h_ratelo[0], vstream[1] = h_rate[1]*x1 + h_rate[3]*x2 + h_ratelo[1], vstream[2] = h_rate[2]*x2 + h_ratelo[2]
          sx = hRate[0] * xc0 + hRate[5] * xc1 + hRate[4] * xc2 + hRateLo[0];
          sy = hRate[1] * xc1 + hRate[3] * xc2 + hRateLo[1];
          sz = hRate[2] * xc2 + hRateLo[2];
        }
        const tbin = (value[b] / (n - dofTstat)) * tfactor;
        const scale = Math.sqrt(this.temperatureSrd / tbin);
        let vsq = 0.0;
        for (let j = binhead[b]; j >= 0; j = binnext[j]) {
          const u0 = (v[3 * j] - vsum[3 * b]) * scale;
          const u1 = (v[3 * j + 1] - vsum[3 * b + 1]) * scale;
          const u2 = (v[3 * j + 2] - vsum[3 * b + 2]) * scale;
          vsq += u0 * u0 + u1 * u1 + u2 * u2;
          v[3 * j] = u0 + sx;
          v[3 * j + 1] = u1 + sy;
          v[3 * j + 2] = u2 + sz;
        }
        value[b] = vsq;
      }
      combine([{ a: value, w: 1 }]);
    }

    // SRD temperature from the thermal velocities of the owner bins with more than one particle
    for (let b = 0; b < nbins; b++) {
      if (owner[b] && cnt[b] > 1) {
        this.srdBinTemp += value[b] / (cnt[b] - dofTemp);
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

    // "undo velocity remap (only if using PUT or tstat no)": the box-rate correction of the last bins is added back
    if (deformRates && (this.unbiasflag || !this.tstat)) {
      if (lamB && geomR) {
        const lam2 = Float64Array.from(x.subarray(0, 3 * s.n));
        x2lamdaAll(lam2, s.n, s.box.lo, geomR.hinv);
        pos = lam2;
        for (let k = 0; k < s.n; k++) {
          const i = order[k];
          if (!(mask[i] & this.groupBit)) continue;
          const [ix, iy, iz] = cellOf(i);
          if (per[0] && ix === this.nbin1[0]) v[3 * i] += hRate[0];
          if (per[1] && iy === this.nbin1[1]) {
            v[3 * i] += hRate[5];
            v[3 * i + 1] += hRate[1];
          }
          if (per[2] && iz === this.nbin1[2]) {
            v[3 * i] += hRate[4];
            v[3 * i + 1] += hRate[3];
            v[3 * i + 2] += hRate[2];
          }
        }
        lamda2xAll(lam2, s.n, s.box.lo, geomR.h);
        x.set(lam2, 0);
      } else {
        // As native LAMMPS does (src/SRD/fix_srd.cpp reset_velocities at b332f28): the undo pass calls domain->x2lamda(nlocal)
        // for every box, not only a triclinic one, so its bins are taken from lamda positions against the corner in box
        // units (orthogonal: boxlo, or the shifted corner). A particle of the last bin of a shifted grid therefore
        // gets no add-back in an orthogonal box, which is the native result (measured with native LAMMPS: w179rg_srd_line_deform).
        const geomO = boxMatrix(s.box.lo, s.box.hi, s.box.tilt);
        const lam2 = Float64Array.from(x.subarray(0, 3 * s.n));
        x2lamdaAll(lam2, s.n, s.box.lo, geomO.hinv);
        pos = lam2;
        for (let k = 0; k < s.n; k++) {
          const i = order[k];
          if (!(mask[i] & this.groupBit)) continue;
          const [ix, iy, iz] = cellOf(i);
          if (per[0] && ix === this.nbin1[0]) v[3 * i] += hRate[0];
          if (per[1] && iy === this.nbin1[1]) v[3 * i + 1] += hRate[1];
          if (per[2] && iz === this.nbin1[2]) v[3 * i + 2] += hRate[2];
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
      if (this.bigExist || this.wallFix) {
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
      this.bigDynamic();
      this.updateCopyPositions();
    }
    // native post_force(): wall_params(0) sets the wall positions for this step and zeroes the wall forces
    if (this.wallFix) this.wallFix.wallParams(0);
    if (this.bigExist || this.wallFix) {
      if (this.overlapflag) this.collisionsMulti();
      else this.collisionsSingle();
    }

    // if any SRD particle has moved too far, trigger reneighboring on the next step (lamda units for triclinic:
    // fix_srd.cpp post_force(), x2lamda(nlocal) and lamda2x(nlocal) around the check)
    let flag = false;
    const geomP = this.triclinic ? boxMatrix(s.box.lo, s.box.hi, s.box.tilt) : null;
    if (geomP) x2lamdaAll(x, s.n, s.box.lo, geomP.hinv);
    for (let k = 0; k < s.n && !flag; k++) {
      const i = order[k];
      if (!(mask[i] & this.groupBit)) continue;
      for (let d = 0; d < 3; d++) {
        if (x[3 * i + d] < this.srdloRe[d] || x[3 * i + d] > this.srdhiRe[d]) { flag = true; break; }
      }
    }
    if (geomP) lamda2xAll(x, s.n, s.box.lo, geomP.h);
    if (flag) {
      this.nextReneighbor = s.step + 1;
      this.reneighflag = SRD_MOVE;
    }
    // a wall that moved more than the walltrigger since the last reneighbor triggers one on the next step (native)
    if (this.wallFix) {
      for (let m = 0; m < this.wallFix.nwall; m++) {
        if (Math.abs(this.wallFix.xwall[m] - this.wallFix.xwallhold[m]) > this.walltrigger) this.nextReneighbor = s.step + 1;
      }
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
   * off the surface with a Gaussian normal velocity, and the big particle receives the momentum change.
   */
  private collisionsSingle(): void {
    const s = this.sys.state;
    const order = nativeOrder(s);
    const { xs, vs, xb, xscoll, xbcoll, norm } = this;

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
          if (c >= this.copyOwner.length) {
            // a wall (native collisions_single, type WALL: inside_wall, collision_wall_exact or _inexact)
            const w = c - this.copyOwner.length;
            this.loadPair(i, c, xs, vs, xb);
            if (!this.insideWall(xs, w)) continue;
            let tw: number;
            if (this.exactflag) {
              tw = this.collisionWallExact(xs, vs, w, xscoll, xbcoll, norm);
            } else {
              tw = 0.5 * dt;
              this.collisionWallInexact(xs, w, xscoll, xbcoll, norm);
            }
            if (tw > dt) {
              this.startedInside(i, w, ibounce, true);
              break;
            }
            ibin = this.applyCollisionAt(i, w, c, tw);
            if (ibounce === 0) this.ncollide++;
            ibounce++;
            if (ibounce < this.maxbounceallow || this.maxbounceallow === 0) collideFlag = true;
            dt = tw;
            break;
          }
          const j = this.copyOwner[c];
          this.loadPair(i, c, xs, vs, xb);
          const rad = this.copyRadius[c];
          const ell = this.bigKind[j] === KIND_ELLIPSOID;
          const ax = this.bigAxes.subarray(9 * j, 9 * j + 9);
          const inv = this.bigInv.subarray(3 * j, 3 * j + 3);
          const inside = ell ? insideEllipsoid(xs, xb, ax, inv) : insideSphere(xs, xb, rad * rad);
          if (!inside) continue;

          // the big particle's velocity and angular velocity (the owner's v and omega)
          const vb = this.bigVelocity(j);
          let tRemain: number;
          if (this.exactflag) {
            tRemain = ell
              ? collisionEllipsoidExact(xs, xb, vs, vb, this.bigOmega(j), ax, inv, xscoll, xbcoll, norm)
              : collisionSphereExact(xs, xb, vs, vb, rad * rad, xscoll, xbcoll, norm);
          } else {
            tRemain = 0.5 * dt;
            if (ell) collisionEllipsoidInexact(xs, xb, ax, inv, xscoll, xbcoll, norm);
            else collisionSphereInexact(xs, xb, rad, xscoll, xbcoll, norm);
          }

          if (tRemain > dt) {
            this.startedInside(i, j, ibounce);
            break;
          }
          ibin = this.applyCollisionAt(i, j, c, tRemain);
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

  /**
   * fix_srd.cpp collisions_multi() (neigh_modify ... overlap yes): the exact collision with every big copy the SRD
   * overlaps is found, and the one that happened first (largest time before now, not the last collided big
   * particle) is applied; the last big particle is skipped once, and the search repeats from the new position.
   * Collisions are always exact here.
   */
  private collisionsMulti(): void {
    const s = this.sys.state;
    const order = nativeOrder(s);
    const { xs, vs, xb, xscoll, xbcoll, norm } = this;
    const xscF = this.savedXscoll, xbcF = this.savedXbcoll, normF = this.savedNorm;

    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & this.groupBit)) continue;

      let ibin = this.binsrd[i];
      if ((this.binbig[ibin] ?? EMPTY).length === 0) continue;

      let ibounce = 0;
      let jlast = -1;
      let typefirst = -1;
      let dt = this.dtBig;

      for (;;) {
        const cands = this.binbig[ibin] ?? EMPTY;
        const nbig = cands.length;
        if (ibounce === 0) this.ncheck += nbig;

        let tFirst = 0;
        let jfirst = -1;
        let kfirst = -1;
        for (let m = 0; m < nbig; m++) {
          const c = cands[m];
          const isWall = c >= this.copyOwner.length;
          const j = isWall ? c - this.copyOwner.length : this.copyOwner[c];
          const kind = isWall ? KIND_WALL : this.bigKind[j];
          if (j === jlast && kind === typefirst) continue;
          this.loadPair(i, c, xs, vs, xb);
          if (isWall) {
            // native collisions_multi, type WALL: inside_wall, then collision_wall_exact (always exact here)
            if (!this.insideWall(xs, j)) continue;
            const tw = this.collisionWallExact(xs, vs, j, xscoll, xbcoll, norm);
            if (tw > dt || tw < 0.0) {
              this.startedInside(i, j, ibounce, true);
              tFirst = 0;
              break;
            }
            if (tw > tFirst) {
              tFirst = tw;
              jfirst = j;
              kfirst = c;
              typefirst = kind;
              xscF.set(xscoll);
              xbcF.set(xbcoll);
              normF.set(norm);
            }
            continue;
          }
          const rad = this.copyRadius[c];
          const ell = kind === KIND_ELLIPSOID;
          const ax = this.bigAxes.subarray(9 * j, 9 * j + 9);
          const inv = this.bigInv.subarray(3 * j, 3 * j + 3);
          const vb = this.bigVelocity(j);
          let tRemain: number;
          if (kind === KIND_LINE) {
            // fix_srd.cpp collisions_multi(): inside_line(x[i], x[j], v[i], v[j], big, dt), then collision_line_exact
            if (!this.insideLine(xs, xb, vs, vb, this.bigTheta[j], this.bigOmega(j)[2], this.bigLength[j], dt)) continue;
            tRemain = this.collisionLineExact(dt, xscoll, xbcoll, norm);
          } else if (kind === KIND_TRI) {
            // fix_srd.cpp collisions_multi(): inside_tri(x[i], x[j], v[i], v[j], big, dt), then collision_tri_exact
            if (!this.insideTri(xs, xb, vs, vb, j, dt)) continue;
            tRemain = this.collisionTriExact(dt, xscoll, xbcoll, norm);
          } else {
            const inside = ell ? insideEllipsoid(xs, xb, ax, inv) : insideSphere(xs, xb, rad * rad);
            if (!inside) continue;
            tRemain = ell
              ? collisionEllipsoidExact(xs, xb, vs, vb, this.bigOmega(j), ax, inv, xscoll, xbcoll, norm)
              : collisionSphereExact(xs, xb, vs, vb, rad * rad, xscoll, xbcoll, norm);
          }

          if (tRemain > dt || tRemain < 0.0) {
            this.startedInside(i, j, ibounce);
            tFirst = 0;
            break;
          }
          if (tRemain > tFirst) {
            tFirst = tRemain;
            jfirst = j;
            kfirst = c;
            typefirst = kind;
            xscF.set(xscoll);
            xbcF.set(xbcoll);
            normF.set(norm);
          }
        }
        if (tFirst === 0) break;

        // apply the first collision with the saved collision points of that big particle
        jlast = jfirst;
        xscoll.set(xscF);
        xbcoll.set(xbcF);
        norm.set(normF);
        ibin = this.applyCollisionAt(i, jfirst, kfirst, tFirst);
        if (ibounce === 0) this.ncollide++;
        ibounce++;
        if (ibounce === this.maxbounceallow) break;
        dt = tFirst;
      }

      this.nbounce += ibounce;
      if (this.maxbounceallow && ibounce >= this.maxbounceallow) this.bouncemaxnum++;
      if (ibounce > this.bouncemax) this.bouncemax = ibounce;
    }
  }

  /**
   * fix_srd.cpp inside_line(): is the SRD particle inside the line big particle during the step? The SRD starts on the
   * positive side of the (moving, rotating) infinite line and ends on the negative side; the crossing time is found by
   * newtonRaphson() and the collision point must lie within half the length of the line. Sets the state used by
   * collisionLineExact(). xs, xb: positions at the end of the step; vs, vb: velocities; theta: orientation at the end of
   * the step; omegaZ: its angular velocity; dtStep: the time before now.
   */
  private insideLine(
    xs: Float64Array, xb: Float64Array, vs: Float64Array, vb: Float64Array, theta: number, omegaZ: number,
    length: number, dtStep: number,
  ): boolean {
    const xs0 = this.lxs0, xs1 = this.lxs1, xb0 = this.lxb0, xb1 = this.lxb1;
    for (let d = 0; d < 2; d++) {
      xs1[d] = xs[d];
      xb1[d] = xb[d];
      xs0[d] = xs1[d] - dtStep * vs[d];
      xb0[d] = xb1[d] - dtStep * vb[d];
    }
    this.ltheta1 = theta;
    this.ltheta0 = theta - dtStep * omegaZ;
    const th0 = this.ltheta0, th1 = this.ltheta1;
    // n = normal to the line = [sin(theta), -cos(theta)]; side = (P - C) . n
    const side0 = (xs0[0] - xb0[0]) * libmSin(th0) + (xs0[1] - xb0[1]) * -libmCos(th0);
    const side1 = (xs1[0] - xb1[0]) * libmSin(th1) + (xs1[1] - xb1[1]) * -libmCos(th1);
    if (side0 <= 0.0 || side1 >= 0.0) return false;
    // Newton-Raphson solve of the full non-linear parametric equation for the crossing time (0 to 1)
    const tf = this.newtonRaphson(0.0, 1.0);
    this.ltfraction = tf;
    for (let d = 0; d < 2; d++) {
      this.lxsc[d] = xs0[d] + tf * (xs1[d] - xs0[d]);
      this.lxbc[d] = xb0[d] + tf * (xb1[d] - xb0[d]);
    }
    const delx = this.lxsc[0] - this.lxbc[0];
    const dely = this.lxsc[1] - this.lxbc[1];
    const rsq = delx * delx + dely * dely;
    if (rsq > 0.25 * length * length) return false;
    const thc = th0 + tf * (th1 - th0);
    this.lnbc[0] = libmSin(thc);
    this.lnbc[1] = -libmCos(thc);
    return true;
  }

  /**
   * fix_srd.cpp inside_tri(): is the SRD particle inside the triangle big particle j during the step? The SRD starts on
   * the positive side of the (moving, rotating) plane of the triangle and ends on the negative side; the crossing time
   * is a root of the quadratic approximation of the plane distance, and the collision point (at that time) must lie
   * inside the triangle. Sets tfraction (ltfraction), tsc, tbc and tnbc for collisionTriExact(). xs, xb: positions at
   * the end of the step; vs, vb: velocities; j: the big particle (its omega, space normal and axes are from bigDynamic).
   */
  private insideTri(xs: Float64Array, xb: Float64Array, vs: Float64Array, vb: Float64Array, j: number, dtStep: number): boolean {
    const omega = this.bigOmega(j);
    const n1 = this.bigNorm.subarray(3 * j, 3 * j + 3);
    // n0 = n1 - dt (omega x n1)
    const n0 = [
      n1[0] - dtStep * (omega[1] * n1[2] - omega[2] * n1[1]),
      n1[1] - dtStep * (omega[2] * n1[0] - omega[0] * n1[2]),
      n1[2] - dtStep * (omega[0] * n1[1] - omega[1] * n1[0]),
    ];
    const pmc0 = [0, 1, 2].map((d) => xs[d] - dtStep * vs[d] - xb[d] + dtStep * vb[d]);
    const pmc1 = [0, 1, 2].map((d) => xs[d] - xb[d]);
    const dot = (a: number[] | Float64Array, b: number[] | Float64Array) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const side0 = dot(pmc0, n0);
    const side1 = dot(pmc1, n1);
    if (side0 <= 0.0 || side1 >= 0.0) return false;

    // quadratic equation solve of the approximate parametric equation for the crossing time (0 to 1)
    const n1n0 = [0, 1, 2].map((d) => n1[d] - n0[d]);
    const pmc1pmc0 = [0, 1, 2].map((d) => pmc1[d] - pmc0[d]);
    const a = dot(pmc1pmc0, n1n0);
    const b = dot(pmc1pmc0, n0) + dot(n1n0, pmc0);
    const c = dot(pmc0, n0);
    let tf: number;
    if (a === 0.0) {
      const dot0 = dot(pmc0, n0);
      const dot1 = dot(pmc1, n0);
      tf = -dot0 / (dot1 - dot0);
    } else {
      const term = Math.sqrt(b * b - 4.0 * a * c);
      const root1 = (-b + term) / (2.0 * a);
      const root2 = (-b - term) / (2.0 * a);
      if (0.0 <= root1 && root1 <= 1.0) tf = root1;
      else if (0.0 <= root2 && root2 <= 1.0) tf = root2;
      else throw new StyleError('Bad quadratic solve for particle/tri collision');
    }
    this.ltfraction = tf;

    // the position and orientation of S and B at the collision time (dt before now)
    const dtc = (1.0 - tf) * dtStep;
    const ax = this.bigAxes.subarray(9 * j, 9 * j + 9);
    const cb = this.bigCorners.subarray(9 * j, 9 * j + 9);
    const ex = [ax[0], ax[1], ax[2]], ey = [ax[3], ax[4], ax[5]], ez = [ax[6], ax[7], ax[8]];
    const xsc = this.tsc, xbc = this.tbc;
    for (let d = 0; d < 3; d++) {
      xsc[d] = xs[d] - dtc * vs[d];
      xbc[d] = xb[d] - dtc * vb[d];
    }
    // orientation axes at the collision time: e - dt (omega x e)
    const rot = (e: number[]) => [
      e[0] - dtc * (omega[1] * e[2] - omega[2] * e[1]),
      e[1] - dtc * (omega[2] * e[0] - omega[0] * e[2]),
      e[2] - dtc * (omega[0] * e[1] - omega[1] * e[0]),
    ];
    const exc = rot(ex), eyc = rot(ey), ezc = rot(ez);
    // corners c1, c2, c3 at the collision time: xbc + (exc, eyc, ezc) . body corner
    const corner = (k: number): number[] => {
      const cb0 = cb[3 * k], cb1 = cb[3 * k + 1], cb2 = cb[3 * k + 2];
      return [0, 1, 2].map((d) => xbc[d] + exc[d] * cb0 + eyc[d] * cb1 + ezc[d] * cb2);
    };
    const c1 = corner(0), c2 = corner(1), c3 = corner(2);
    const c2mc1 = [0, 1, 2].map((d) => c2[d] - c1[d]);
    const c3mc2 = [0, 1, 2].map((d) => c3[d] - c2[d]);
    const c1mc3 = [0, 1, 2].map((d) => c1[d] - c3[d]);
    const nbc = [
      c2mc1[1] * c3mc2[2] - c2mc1[2] * c3mc2[1],
      c2mc1[2] * c3mc2[0] - c2mc1[0] * c3mc2[2],
      c2mc1[0] * c3mc2[1] - c2mc1[1] * c3mc2[0],
    ];
    const nlen = Math.sqrt(nbc[0] * nbc[0] + nbc[1] * nbc[1] + nbc[2] * nbc[2]);
    for (let d = 0; d < 3; d++) nbc[d] /= nlen;
    this.tnbc.set(nbc);

    // the collision point must be inside the triangle: on the inner side of each of the three edges
    const cross = (p: number[], q: number[]) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
    const edges: [number[], number[], number[]] = [c2mc1, c3mc2, c1mc3];
    const verts = [c1, c2, c3];
    for (let k = 0; k < 3; k++) {
      const pvec = [0, 1, 2].map((d) => xsc[d] - verts[k][d]);
      if (dot(cross(edges[k], pvec), nbc) < 0.0) return false;
    }
    return true;
  }

  /** fix_srd.cpp collision_tri_exact(): the collision points and normal of insideTri(); returns the time before now. */
  private collisionTriExact(dtStep: number, xscoll: Float64Array, xbcoll: Float64Array, norm: Float64Array): number {
    xscoll.set(this.tsc);
    xbcoll.set(this.tbc);
    norm.set(this.tnbc);
    return (1.0 - this.ltfraction) * dtStep;
  }

  /** fix_srd.cpp collision_line_exact(): the collision points and normal of inside_line(); returns the time before now. */
  private collisionLineExact(dtStep: number, xscoll: Float64Array, xbcoll: Float64Array, norm: Float64Array): number {
    xscoll[0] = this.lxsc[0];
    xscoll[1] = this.lxsc[1];
    xscoll[2] = 0.0;
    xbcoll[0] = this.lxbc[0];
    xbcoll[1] = this.lxbc[1];
    xbcoll[2] = 0.0;
    norm[0] = this.lnbc[0];
    norm[1] = this.lnbc[1];
    norm[2] = 0.0;
    return (1.0 - this.ltfraction) * dtStep;
  }

  /** fix_srd.cpp newton_raphson(): root of lineside() in [t1, t2] (safeguarded Newton-bisection, as native). */
  private newtonRaphson(t1: number, t2: number): number {
    const first = this.lineSide(t1);
    let tlo: number, thi: number;
    if (first.f < 0.0) {
      tlo = t1;
      thi = t2;
    } else {
      thi = t1;
      tlo = t2;
    }
    let t = 0.5 * (t1 + t2);
    let dtold = Math.abs(t2 - t1);
    let dt = dtold;
    let side = this.lineSide(t);
    for (let it = 0; it < LINE_MAXITER; it++) {
      const f = side.f;
      const df = side.df;
      if ((((t - thi) * df - f) * ((t - tlo) * df - f) > 0.0) || Math.abs(2.0 * f) > Math.abs(dtold * df)) {
        dtold = dt;
        dt = 0.5 * (thi - tlo);
        t = tlo + dt;
        if (tlo === t) return t;
      } else {
        dtold = dt;
        dt = f / df;
        const temp = t;
        t -= dt;
        if (temp === t) return t;
      }
      if (Math.abs(dt) < LINE_TOLERANCE) return t;
      side = this.lineSide(t);
      if (side.f < 0.0) tlo = t;
      else thi = t;
    }
    return t;
  }

  /** fix_srd.cpp lineside(): the signed distance f of the SRD from the rotating line at t, and its derivative df. */
  private lineSide(t: number): { f: number; df: number } {
    const xs0 = this.lxs0, xs1 = this.lxs1, xb0 = this.lxb0, xb1 = this.lxb1;
    const p0 = xs0[0] + (xs1[0] - xs0[0]) * t;
    const p1 = xs0[1] + (xs1[1] - xs0[1]) * t;
    const c0 = xb0[0] + (xb1[0] - xb0[0]) * t;
    const c1 = xb0[1] + (xb1[1] - xb0[1]) * t;
    const dtheta = this.ltheta1 - this.ltheta0;
    const theta = this.ltheta0 + dtheta * t;
    const cosT = libmCos(theta);
    const sinT = libmSin(theta);
    const f = (p1 - c1) * cosT - (p0 - c0) * sinT;
    const df = ((xs1[1] - xs0[1]) - (xb1[1] - xb0[1])) * cosT - (p1 - c1) * sinT * dtheta -
      ((xs1[0] - xs0[0]) - (xb1[0] - xb0[0])) * sinT - (p0 - c0) * cosT * dtheta;
    return { f, df };
  }

  /** the SRD position and velocity of candidate c's pair: xs = x[i], vs = v[i], xb = the big copy's position */
  private loadPair(i: number, c: number, xs: Float64Array, vs: Float64Array, xb: Float64Array): void {
    const { x, v } = this.sys.state;
    for (let d = 0; d < 3; d++) {
      xs[d] = x[3 * i + d];
      vs[d] = v[3 * i + d];
      if (c < this.copyOwner.length) xb[d] = this.copyPos[3 * c + d];
    }
  }

  /** the inside-a-big-particle case (collisions_single and collisions_multi): an error, a warning, or nothing */
  private startedInside(i: number, j: number, ibounce: number, wall = false): void {
    const s = this.sys.state;
    this.ninside++;
    if (this.insideflag === INSIDE_IGNORE) return;
    const msg = wall
      ? `SRD particle ${s.id[i]} started inside wall ${j} on step ${s.step} bounce ${ibounce + 1}`
      : `SRD particle ${s.id[i]} started inside big particle ${s.id[j]} on step ${s.step} bounce ${ibounce + 1}`;
    if (this.insideflag === INSIDE_ERROR) throw new StyleError(msg);
    this.sys.warn(msg);
  }

  /**
   * the collision update of collisions_single and collisions_multi, for the collision points in the scratch
   * arrays (xscoll, xbcoll, norm) at time tRemain: the slip or noslip velocity vsnew, the rescale check, the
   * momentum (and no-slip torque) given to the big particle, and update_srd(): the SRD moves from the collision
   * point with its new velocity for tRemain and is binned again. Returns the new search bin.
   */
  private applyCollisionAt(i: number, j: number, c: number, tRemain: number): number {
    if (c >= this.copyOwner.length) return this.applyWallCollisionAt(i, c - this.copyOwner.length, tRemain);
    const s = this.sys.state;
    const { x, v } = s;
    const { vs, xb, xscoll, xbcoll, norm, vsnew } = this;
    for (let d = 0; d < 3; d++) {
      vs[d] = v[3 * i + d];
      xb[d] = this.copyPos[3 * c + d];
    }
    // noslip/slip take the big particle's position at the time of the step (x[j]), not its collision point
    const vb = this.bigVelocity(j);
    const omega = this.bigOmega(j);
    if (this.noslipflag) this.noslip(vs, vb, xb, omega, xscoll, norm, vsnew);
    else this.slip(vs, vb, xb, omega, xscoll, norm, vsnew);
    if (s.dimension === 2) vsnew[2] = 0.0;

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

    // fix_srd.cpp: a sphere with slip collisions gets no torque, every other big particle does
    this.forceTorque(v, i, vsnew, xscoll, xbcoll, j, this.noslipflag || this.bigKind[j] !== KIND_SPHERE);

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
    return (this.binsrd[i] = iz * ny * nx + iy * nx + ix);
  }

  /**
   * fix_srd.cpp pre_neighbor(), walls: each wall is registered in every search bin it covers, from its position (plus
   * the walltrigger for a moving wall) up to the non-periodic boundary. The entry of wall m is copies + m.
   */
  private registerWalls(): void {
    const wf = this.wallFix!;
    const nc = this.copyOwner.length;
    const delta = wf.varflag ? this.walltrigger : 0;
    const [nx, ny, nz] = this.nbin2;
    for (let m = 0; m < wf.nwall; m++) {
      const dim = wf.wallwhich[m] >> 1;
      const side = wf.wallwhich[m] & 1;
      const nd = this.nbin2[dim];
      const origin = this.blo2[dim];
      const inv = this.bininv2[dim];
      let lo: number;
      let hi: number;
      if (side === 0) {
        hi = Math.trunc((wf.xwall[m] + delta - origin) * inv);
        if (hi < 0) continue;
        if (hi >= nd) throw new StyleError('Fix SRD: bad search bin assignment');
        lo = 0;
      } else {
        lo = Math.trunc((wf.xwall[m] - delta - origin) * inv);
        if (lo >= nd) continue;
        if (lo < 0) throw new StyleError('Fix SRD: bad search bin assignment');
        hi = nd - 1;
      }
      for (let iz = 0; iz < nz; iz++) {
        for (let iy = 0; iy < ny; iy++) {
          for (let ix = 0; ix < nx; ix++) {
            const idx = dim === 0 ? ix : dim === 1 ? iy : iz;
            if (idx < lo || idx > hi) continue;
            const ibin = iz * ny * nx + iy * nx + ix;
            if (this.binbig[ibin].length === ATOMPERBIN) throw new StyleError('Fix SRD: too many walls in bin');
            this.binbig[ibin].push(nc + m);
          }
        }
      }
    }
  }

  /** fix_srd.cpp inside_wall(): is the SRD particle on the far side of wall m (the side it has passed)? */
  private insideWall(xs: Float64Array, m: number): boolean {
    const wf = this.wallFix!;
    const dim = wf.wallwhich[m] >> 1;
    const side = wf.wallwhich[m] & 1;
    if (side === 0 && xs[dim] < wf.xwall[m]) return true;
    if (side !== 0 && xs[dim] > wf.xwall[m]) return true;
    return false;
  }

  /**
   * fix_srd.cpp collision_wall_exact(): the collision time dt before now (xs - xwall) / (vs - vwall) along the wall
   * normal, the SRD position at that time (xscoll), the wall position (xbcoll) and the unit normal of the wall.
   */
  private collisionWallExact(xs: Float64Array, vs: Float64Array, m: number, xscoll: Float64Array, xbcoll: Float64Array, norm: Float64Array): number {
    const wf = this.wallFix!;
    const dim = wf.wallwhich[m] >> 1;
    const side = wf.wallwhich[m] & 1;
    const dt = (xs[dim] - wf.xwall[m]) / (vs[dim] - wf.vwall[m]);
    xscoll[0] = xs[0] - dt * vs[0];
    xscoll[1] = xs[1] - dt * vs[1];
    xscoll[2] = xs[2] - dt * vs[2];
    xbcoll[0] = xbcoll[1] = xbcoll[2] = 0.0;
    xbcoll[dim] = wf.xwall[m] - dt * wf.vwall[m];
    norm[0] = norm[1] = norm[2] = 0.0;
    norm[dim] = side === 0 ? 1.0 : -1.0;
    return dt;
  }

  /** fix_srd.cpp collision_wall_inexact(): the SRD is pushed to the wall at the end of the step (time = end of step). */
  private collisionWallInexact(xs: Float64Array, m: number, xscoll: Float64Array, xbcoll: Float64Array, norm: Float64Array): void {
    const wf = this.wallFix!;
    const dim = wf.wallwhich[m] >> 1;
    const side = wf.wallwhich[m] & 1;
    xscoll[0] = xs[0];
    xscoll[1] = xs[1];
    xscoll[2] = xs[2];
    xscoll[dim] = wf.xwall[m];
    xbcoll[0] = xbcoll[1] = xbcoll[2] = 0.0;
    xbcoll[dim] = wf.xwall[m];
    norm[0] = norm[1] = norm[2] = 0.0;
    norm[dim] = side === 0 ? 1.0 : -1.0;
  }

  /**
   * fix_srd.cpp slip_wall() and noslip() with a WALL: the same draws as noslip (a Gaussian normal and two tangential
   * velocities, rejected above vmax); the wall's velocity along its normal is added (the wall has no tangential motion).
   */
  private noslipWall(vs: Float64Array, m: number, norm: Float64Array, vsnew: Float64Array): void {
    const wf = this.wallFix!;
    const zero = new Float64Array(3);
    this.noslip(vs, zero, zero, zero, zero, norm, vsnew);
    const dim = wf.wallwhich[m] >> 1;
    vsnew[dim] += wf.vwall[m];
  }

  /** fix_srd.cpp force_wall(): the wall gets -dp/dt of the SRD particle (the factor of force_torque). */
  private forceWall(v: Float64Array, i: number, vsnew: Float64Array, m: number): void {
    const wf = this.wallFix!;
    const factor = this.massSrd / this.dtBig / this.sys.state.units.ftm2v;
    for (let d = 0; d < 3; d++) wf.fwall[3 * m + d] -= factor * (vsnew[d] - v[3 * i + d]);
  }

  /**
   * The collision update of applyCollisionAt for a wall: the SRD velocity from noslipWall, the rescale check, the wall
   * force, and update_srd() (the SRD moves from the collision point for tRemain and is binned again).
   */
  private applyWallCollisionAt(i: number, m: number, tRemain: number): number {
    const s = this.sys.state;
    const { x, v } = s;
    const { vs, xscoll, norm, vsnew } = this;
    for (let d = 0; d < 3; d++) vs[d] = v[3 * i + d];
    this.noslipWall(vs, m, norm, vsnew);
    if (s.dimension === 2) vsnew[2] = 0.0;
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
    this.forceWall(v, i, vsnew, m);
    for (let d = 0; d < 3; d++) {
      v[3 * i + d] = vsnew[d];
      x[3 * i + d] = xscoll[d] + tRemain * vsnew[d];
    }
    this.checkSrdDomain(i);
    const ix = Math.trunc((x[3 * i] - this.blo2[0]) * this.bininv2[0]);
    const iy = Math.trunc((x[3 * i + 1] - this.blo2[1]) * this.bininv2[1]);
    const iz = Math.trunc((x[3 * i + 2] - this.blo2[2]) * this.bininv2[2]);
    const [nx, ny] = this.nbin2;
    return (this.binsrd[i] = iz * ny * nx + iy * nx + ix);
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

  /**
   * fix_srd.cpp noslip(): the SRD leaves the surface with a Gaussian normal velocity and two Gaussian tangential
   * velocities (four draws per trial, rejected above vmax), plus the surface velocity of the big particle.
   */
  private noslip(
    vs: Float64Array, vb: Float64Array, xb: Float64Array, omega: Float64Array, xsurf: Float64Array,
    norm: Float64Array, vsnew: Float64Array,
  ): void {
    const vsDotN = vs[0] * norm[0] + vs[1] * norm[1] + vs[2] * norm[2];
    const t1 = this.tangent1;
    t1[0] = vs[0] - vsDotN * norm[0];
    t1[1] = vs[1] - vsDotN * norm[1];
    t1[2] = vs[2] - vsDotN * norm[2];
    const scale = 1.0 / Math.sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    t1[0] *= scale;
    t1[1] *= scale;
    t1[2] *= scale;
    const t2 = this.tangent2;
    t2[0] = norm[1] * t1[2] - norm[2] * t1[1];
    t2[1] = norm[2] * t1[0] - norm[0] * t1[2];
    t2[2] = norm[0] * t1[1] - norm[1] * t1[0];
    let vnmag = 0;
    let vtmag1 = 0;
    let vtmag2 = 0;
    for (;;) {
      const r1 = this.sigma * this.random.gaussian();
      const r2 = this.sigma * this.random.gaussian();
      vnmag = Math.sqrt(r1 * r1 + r2 * r2);
      vtmag1 = this.sigma * this.random.gaussian();
      vtmag2 = this.sigma * this.random.gaussian();
      if (vnmag * vnmag + vtmag1 * vtmag1 + vtmag2 * vtmag2 <= this.vmaxsq) break;
    }
    for (let d = 0; d < 3; d++) vsnew[d] = vnmag * norm[d] + vtmag1 * t1[d] + vtmag2 * t2[d];
    vsnew[0] += vb[0] + omega[1] * (xsurf[2] - xb[2]) - omega[2] * (xsurf[1] - xb[1]);
    vsnew[1] += vb[1] + omega[2] * (xsurf[0] - xb[0]) - omega[0] * (xsurf[2] - xb[2]);
    vsnew[2] += vb[2] + omega[0] * (xsurf[1] - xb[1]) - omega[1] * (xsurf[0] - xb[0]);
  }

  /**
   * fix_srd.cpp force_torque(): the big particle gets -dp/dt of the SRD as a force, and as a torque about its
   * centre (xs - xb cross -dp/dt) when the collision is no-slip.
   */
  private forceTorque(
    v: Float64Array, i: number, vsnew: Float64Array, xs: Float64Array, xb: Float64Array, j: number, applyTorque: boolean,
  ): void {
    const f = this.sys.state.f;
    const factor = this.massSrd / this.dtBig / this.sys.state.units.ftm2v;
    const dp0 = factor * (vsnew[0] - v[3 * i]);
    const dp1 = factor * (vsnew[1] - v[3 * i + 1]);
    const dp2 = factor * (vsnew[2] - v[3 * i + 2]);
    f[3 * j] -= dp0;
    f[3 * j + 1] -= dp1;
    f[3 * j + 2] -= dp2;
    if (!applyTorque) return;
    const tq = this.sys.state.torque!;
    const r0 = xs[0] - xb[0];
    const r1 = xs[1] - xb[1];
    const r2 = xs[2] - xb[2];
    tq[3 * j] -= r1 * dp2 - r2 * dp1;
    tq[3 * j + 1] -= r2 * dp0 - r0 * dp2;
    tq[3 * j + 2] -= r0 * dp1 - r1 * dp0;
  }

  private bigVelocity(j: number): Float64Array {
    const v = this.sys.state.v;
    return v.subarray(3 * j, 3 * j + 3);
  }

  /** omega of big particle j (bigDynamic(): its atom omega, or the angular velocity of an ellipsoid) */
  private bigOmega(j: number): Float64Array {
    return this.bigOmegaAll.subarray(3 * j, 3 * j + 3);
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

/**
 * inside_ellipsoid(): SRD point inside the ellipsoid. ex/ey/ez are the body axes (9 numbers: ex, ey, ez), inv the
 * inverse squared semi-axes (aradsqinv, bradsqinv, cradsqinv).
 */
const insideEllipsoid = (xs: Float64Array, xb: Float64Array, ax: Float64Array, inv: Float64Array): boolean => {
  const d0 = xs[0] - xb[0];
  const d1 = xs[1] - xb[1];
  const d2 = xs[2] - xb[2];
  const x = d0 * ax[0] + d1 * ax[1] + d2 * ax[2];
  const y = d0 * ax[3] + d1 * ax[4] + d2 * ax[5];
  const z = d0 * ax[6] + d1 * ax[7] + d2 * ax[8];
  return x * x * inv[0] + y * y * inv[1] + z * z * inv[2] <= 1.0;
};

/**
 * collision_ellipsoid_exact(): time before now at which the SRD meets the ellipsoid surface (the surface moves with
 * the ellipsoid's translation and rotation). Writes the collision points and the unit normal; returns dt.
 */
const collisionEllipsoidExact = (
  xs: Float64Array, xb: Float64Array, vs: Float64Array, vb: Float64Array, omega: Float64Array, ax: Float64Array,
  inv: Float64Array, xscoll: Float64Array, xbcoll: Float64Array, norm: Float64Array,
): number => {
  const ox = omega[0], oy = omega[1], oz = omega[2];
  const vsvb0 = vs[0] - vb[0], vsvb1 = vs[1] - vb[1], vsvb2 = vs[2] - vb[2];
  const xsxb0 = xs[0] - xb[0], xsxb1 = xs[1] - xb[1], xsxb2 = xs[2] - xb[2];
  // omega x e (for each body axis e)
  const wex = [oy * ax[2] - oz * ax[1], oz * ax[0] - ox * ax[2], ox * ax[1] - oy * ax[0]];
  const wey = [oy * ax[5] - oz * ax[4], oz * ax[3] - ox * ax[5], ox * ax[4] - oy * ax[3]];
  const wez = [oy * ax[8] - oz * ax[7], oz * ax[6] - ox * ax[8], ox * ax[7] - oy * ax[6]];
  const ex = [ax[0], ax[1], ax[2]], ey = [ax[3], ax[4], ax[5]], ez = [ax[6], ax[7], ax[8]];
  const dot = (p: number[], a0: number, a1: number, a2: number): number => p[0] * a0 + p[1] * a1 + p[2] * a2;
  const ax_ = dot(wex, vsvb0, vsvb1, vsvb2);
  let bx = -dot(ex, vsvb0, vsvb1, vsvb2);
  bx -= dot(wex, xsxb0, xsxb1, xsxb2);
  const cx = dot(ex, xsxb0, xsxb1, xsxb2);
  const ay = dot(wey, vsvb0, vsvb1, vsvb2);
  let by = -dot(ey, vsvb0, vsvb1, vsvb2);
  by -= dot(wey, xsxb0, xsxb1, xsxb2);
  const cy = dot(ey, xsxb0, xsxb1, xsxb2);
  const az = dot(wez, vsvb0, vsvb1, vsvb2);
  let bz = -dot(ez, vsvb0, vsvb1, vsvb2);
  bz -= dot(wez, xsxb0, xsxb1, xsxb2);
  const cz = dot(ez, xsxb0, xsxb1, xsxb2);
  const a = (bx * bx + 2.0 * ax_ * cx) * inv[0] + (by * by + 2.0 * ay * cy) * inv[1] + (bz * bz + 2.0 * az * cz) * inv[2];
  const b = 2.0 * (bx * cx * inv[0] + by * cy * inv[1] + bz * cz * inv[2]);
  const c = cx * cx * inv[0] + cy * cy * inv[1] + cz * cz * inv[2] - 1.0;
  const dt = (-b + Math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a);

  for (let d = 0; d < 3; d++) {
    xscoll[d] = xs[d] - dt * vs[d];
    xbcoll[d] = xb[d] - dt * vb[d];
  }
  // body axes at the collision time: e - dt (omega x e)
  const exc = [0, 0, 0], eyc = [0, 0, 0], ezc = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    exc[d] = ex[d] - dt * (wex[d]);
    eyc[d] = ey[d] - dt * (wey[d]);
    ezc[d] = ez[d] - dt * (wez[d]);
  }
  const delta0 = xscoll[0] - xbcoll[0], delta1 = xscoll[1] - xbcoll[1], delta2 = xscoll[2] - xbcoll[2];
  // xbody = (exc, eyc, ezc)^T delta; nbody scales by the inverse squared axes; norm = (exc, eyc, ezc) nbody
  const xb0 = dot(exc, delta0, delta1, delta2);
  const xb1 = dot(eyc, delta0, delta1, delta2);
  const xb2 = dot(ezc, delta0, delta1, delta2);
  const n0 = xb0 * inv[0], n1 = xb1 * inv[1], n2 = xb2 * inv[2];
  for (let d = 0; d < 3; d++) norm[d] = exc[d] * n0 + eyc[d] * n1 + ezc[d] * n2;
  const scale = 1.0 / Math.sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
  norm[0] *= scale;
  norm[1] *= scale;
  norm[2] *= scale;
  return dt;
};

/** collision_ellipsoid_inexact(): push the SRD to the ellipsoid surface at the end of the step */
const collisionEllipsoidInexact = (
  xs: Float64Array, xb: Float64Array, ax: Float64Array, inv: Float64Array, xscoll: Float64Array,
  xbcoll: Float64Array, norm: Float64Array,
): void => {
  const d0 = xs[0] - xb[0], d1 = xs[1] - xb[1], d2 = xs[2] - xb[2];
  let x = d0 * ax[0] + d1 * ax[1] + d2 * ax[2];
  let y = d0 * ax[3] + d1 * ax[4] + d2 * ax[5];
  let z = d0 * ax[6] + d1 * ax[7] + d2 * ax[8];
  const scale = 1.0 / Math.sqrt(x * x * inv[0] + y * y * inv[1] + z * z * inv[2]);
  x *= scale;
  y *= scale;
  z *= scale;
  for (let d = 0; d < 3; d++) {
    xscoll[d] = x * ax[d] + y * ax[3 + d] + z * ax[6 + d] + xb[d];
    xbcoll[d] = xb[d];
  }
  const delta0 = xscoll[0] - xbcoll[0], delta1 = xscoll[1] - xbcoll[1], delta2 = xscoll[2] - xbcoll[2];
  const xb0 = ax[0] * delta0 + ax[1] * delta1 + ax[2] * delta2;
  const xb1 = ax[3] * delta0 + ax[4] * delta1 + ax[5] * delta2;
  const xb2 = ax[6] * delta0 + ax[7] * delta1 + ax[8] * delta2;
  const n0 = xb0 * inv[0], n1 = xb1 * inv[1], n2 = xb2 * inv[2];
  for (let d = 0; d < 3; d++) norm[d] = ax[d] * n0 + ax[3 + d] * n1 + ax[6 + d] * n2;
  const nn = 1.0 / Math.sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
  norm[0] *= nn;
  norm[1] *= nn;
  norm[2] *= nn;
};

const scale3 = (v: Float64Array, off: number, s: number): void => {
  v[off] *= s;
  v[off + 1] *= s;
  v[off + 2] *= s;
};

const clampInt = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);

/**
 * The box matrix of domain.cpp at b332f28 (set_global_box): h = [lx, ly, lz, yz, xz, xy] (h[3..5] are the tilts) and
 * its inverse hinv, with the same operations as native, so the lamda round trips agree bit for bit.
 */
interface BoxMatrix { h: number[]; hinv: number[]; len: number[] }
const boxMatrix = (lo: number[], hi: number[], tilt: number[]): BoxMatrix => {
  const h = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], tilt[2], tilt[1], tilt[0]];
  const hinv = [1 / h[0], 1 / h[1], 1 / h[2], -h[3] / (h[1] * h[2]), (h[3] * h[5] - h[1] * h[4]) / (h[0] * h[1] * h[2]), -h[5] / (h[0] * h[1])];
  // fix_srd.cpp: the perpendicular-distance scale factors length0..2 of a lamda distance
  const len = [Math.sqrt(hinv[0] * hinv[0] + hinv[5] * hinv[5] + hinv[4] * hinv[4]), Math.sqrt(hinv[1] * hinv[1] + hinv[3] * hinv[3]), hinv[2]];
  return { h, hinv, len };
};

/** Domain::x2lamda(n) in place for atoms 0..n-1 (domain.cpp at b332f28). */
const x2lamdaAll = (x: Float64Array, n: number, lo: number[], hinv: number[]): void => {
  for (let i = 0; i < n; i++) {
    const d0 = x[3 * i] - lo[0], d1 = x[3 * i + 1] - lo[1], d2 = x[3 * i + 2] - lo[2];
    x[3 * i] = hinv[0] * d0 + hinv[5] * d1 + hinv[4] * d2;
    x[3 * i + 1] = hinv[1] * d1 + hinv[3] * d2;
    x[3 * i + 2] = hinv[2] * d2;
  }
};

/** Domain::lamda2x(n) in place for atoms 0..n-1 (domain.cpp at b332f28: the boxlo is added last). */
const lamda2xAll = (x: Float64Array, n: number, lo: number[], h: number[]): void => {
  for (let i = 0; i < n; i++) {
    const l0 = x[3 * i], l1 = x[3 * i + 1], l2 = x[3 * i + 2];
    x[3 * i] = h[0] * l0 + h[5] * l1 + h[4] * l2 + lo[0];
    x[3 * i + 1] = h[1] * l1 + h[3] * l2 + lo[1];
    x[3 * i + 2] = h[2] * l2 + lo[2];
  }
};

// As native LAMMPS does (src/SRD/fix_srd.cpp FixSRD::FixSRD at b332f28): utils::inumeric reads the integers and
// utils::numeric the real values, with the input-parser messages.
const intArg = (w: string | undefined, _what: string): number => inumeric(w);

const numArg = (w: string | undefined, _what: string): number => numeric(w);

const logicalArg = (w: string | undefined, what: string): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`Expected boolean parameter for fix srd ${what}, got ${w ?? ''}`);
};
