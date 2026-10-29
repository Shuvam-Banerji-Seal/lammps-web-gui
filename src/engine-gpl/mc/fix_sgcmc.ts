/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/MC/fix_sgcmc.cpp and src/MC/fix_sgcmc.h at commit b332f28 (fix sgcmc: parallel Monte Carlo in the
 * semi-grand canonical (SGC) and the variance-constrained semi-grand canonical (VC-SGC) ensembles of Sadigh et al.,
 * Phys. Rev. B 85, 184203 (2012); code author Alexander Stukowski, integration into LAMMPS by Aidan Thompson and Axel
 * Kohlmeyer), and src/MANYBODY/pair_eam.cpp and pair_eam.h at the same commit (PairEAM::interpolate, the spline
 * evaluation, embedding_index and compute_atomic_energy, which the fix uses). Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Native structure kept here, for one processor. post_force runs doMC on the steps that are multiples of N (no MC at
 * setup: the fix has no setup()). doMC takes window_moves sampling windows; each window sits at the next corner of the
 * box (samplingWindowPosition starts at 5; bit d of it shifts dimension d by the margin), lists the window atoms of the
 * fix group in neighbor-list order (native local order), twice per dimension for atoms within the margin of a face,
 * and makes int(swap_fraction * numFixAtoms / window_moves) trials. Each trial draws from the local stream (seed + the
 * processor rank 0): the dice (always passed with one processor), the window index, the new species when there are
 * three or more types, and the acceptance draw when deltaB < 0; the variance-constrained second test draws from the
 * global stream (seed). An accepted flip rescales the atom's velocity by sqrt(m_old / m_new).
 *
 * The energy change, as native computes it: with an EAM pair style (eam, eam/alloy, eam/fs) from the electron
 * densities, which every flip updates incrementally (with native's ghost bookkeeping: changedAtoms, then a reverse and
 * a forward communication at the end of each window, through the ghosts' 'from' chain); with 'atomic/energy yes' from
 * the per-atom energies of the flipped atom and its full-list neighbours, ghosts included (compute_atomic_energy);
 * with any other pair style from the total potential energy after and before the trial, then setup_minimal(0).
 *
 * The spline tables of native (interpolate: 7 coefficients per grid point) are built here from the engine pair's
 * tabulated values, the file values that eam/alloy and eam/fs copy (file2array_setfl/fs). Not native bit for bit, by
 * design: the densities at the start of an event are summed over the full list (native sums the half list and folds
 * the ghosts) and the lists are in the engine's order (native: bin order); both change sums in the last bit only,
 * and a trial compares deltaB with the log of a uniform, so the same moves are accepted. pair eam (funcfl) uses each
 * file on its own grid, as the engine's pair eam does (native interpolates the files to a common grid).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairEAMBase, type EAMTable } from '../../engine/force/pair/eam';
import { atomSubStyles, nativeOrder } from '../../engine/atoms';
import { nativeGhosts } from '../neighbor/nativeOrder';
import { RanPark } from '../random_park';
import { log as libmLog } from '../libm/log';

/** Atom styles with per-type masses (native AtomVec mass_type = PER_TYPE); every other style has per-atom masses. */
const PER_TYPE_MASS = new Set([
  'amoeba', 'angle', 'apip', 'atomic', 'bond', 'charge', 'dielectric', 'dipole', 'dpd', 'edpd', 'electron', 'full',
  'mdpd', 'molecular', 'oxdna', 'rheo', 'rheo/thermal', 'smd', 'sph', 'spin', 'tdpd', 'template',
]);

/** utils::numeric. */
const numeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return v;
};
/** utils::inumeric. */
const inumeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isInteger(v)) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return v;
};
/** utils::logical. */
const logical = (w: string | undefined): boolean => {
  const t = (w ?? '').toLowerCase();
  if (t === 'yes' || t === 'on' || t === 'true' || t === '1') return true;
  if (t === 'no' || t === 'off' || t === 'false' || t === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in input script or data file`);
};
const missing = (what: string): StyleError => new StyleError(`Illegal ${what} command: missing argument(s)`);

/** PairEAM::interpolate: 7 coefficients per grid point m = 1 .. n (index 7 m + k), from f[1 .. n] = y[0 .. n - 1]. */
interface NativeSpline { c: Float64Array; n: number; rdelta: number }
const interpolate = (t: EAMTable): NativeSpline => {
  const y = t.y;
  const n = y.length;
  const delta = t.dx;
  const s = new Float64Array((n + 1) * 7);
  for (let m = 1; m <= n; m++) s[7 * m + 6] = y[m - 1];
  s[7 + 5] = s[14 + 6] - s[7 + 6];
  s[14 + 5] = 0.5 * (s[21 + 6] - s[7 + 6]);
  s[7 * (n - 1) + 5] = 0.5 * (s[7 * n + 6] - s[7 * (n - 2) + 6]);
  s[7 * n + 5] = s[7 * n + 6] - s[7 * (n - 1) + 6];
  for (let m = 3; m <= n - 2; m++) {
    s[7 * m + 5] = ((s[7 * (m - 2) + 6] - s[7 * (m + 2) + 6]) + 8.0 * (s[7 * (m + 1) + 6] - s[7 * (m - 1) + 6])) / 12.0;
  }
  for (let m = 1; m <= n - 1; m++) {
    s[7 * m + 4] = 3.0 * (s[7 * (m + 1) + 6] - s[7 * m + 6]) - 2.0 * s[7 * m + 5] - s[7 * (m + 1) + 5];
    s[7 * m + 3] = s[7 * m + 5] + s[7 * (m + 1) + 5] - 2.0 * (s[7 * (m + 1) + 6] - s[7 * m + 6]);
  }
  s[7 * n + 4] = 0.0;
  s[7 * n + 3] = 0.0;
  for (let m = 1; m <= n; m++) {
    s[7 * m + 2] = s[7 * m + 5] / delta;
    s[7 * m + 1] = 2.0 * s[7 * m + 4] / delta;
    s[7 * m + 0] = 3.0 * s[7 * m + 3] / delta;
  }
  return { c: s, n, rdelta: 1.0 / delta };
};

/** ((coeff[3] p + coeff[4]) p + coeff[5]) p + coeff[6] at grid point m. */
const value = (sp: NativeSpline, m: number, p: number): number => {
  const c = sp.c, o = 7 * m;
  return ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
};

/** Index and fraction of an embedding argument: MAX(1, MIN(m, nrho - 1)) and MIN(p, 1) (embedding_index, not eam/he). */
const embedAt = (sp: NativeSpline, rho: number, out: { m: number; p: number }): void => {
  let p = rho * sp.rdelta + 1.0;
  let m = Math.trunc(p);
  m = Math.max(1, Math.min(m, sp.n - 1));
  p -= m;
  out.p = Math.min(p, 1.0);
  out.m = m;
};

export class FixSgcmc extends Fix {
  readonly style = 'sgcmc';
  private readonly neveryMdsteps: number;
  private readonly swapFraction: number;
  private readonly beta: number;
  private readonly deltamu: number[];
  private numSamplingWindowMoves = 8;
  private seed = 324234;
  private atomicEnergyFlag = false;
  /**
   * fix_sgcmc.cpp init: a perpetual full list, "NeighConst::REQ_FULL | NeighConst::REQ_GHOST" with atomic/energy (a ghost
   * list, which no half list is derived from) and "NeighConst::REQ_FULL" otherwise (Fix.nativeFullList).
   */
  nativeFullList = true;
  private samplingWindowUserSize = 0;
  private samplingWindowPosition = 5;
  private kappa = 0;
  private serialMode = false;
  private targetConcentration: number[] = [];
  private random: RanPark;
  private localRandom: RanPark;
  private nAcceptedSwaps = 0;
  private nRejectedSwaps = 0;
  private speciesCounts: number[] = [];
  private interactionRadius = 0;
  private cutforcesq = 0;
  /** the EAM pair style (native pairEAM), null for the atomic/energy and generic methods */
  private pairEAM: PairEAMBase | null = null;
  private samplingWindowAtoms: number[] = [];
  private numFixAtomsLocal = 0;

  // native spline tables, by the engine pair's table object
  private splines = new Map<EAMTable, NativeSpline>();
  private rhoTab: readonly (EAMTable | null)[] = [];
  private z2rTab: readonly (EAMTable | null)[] = [];
  private frhoTab: readonly EAMTable[] = [];
  private nr = 0;
  private rdr = 0;

  // native layout of the last neighbor build: local order, ghosts, full lists (owned atoms; ghosts too for atomic/energy)
  private builtAt = -1;
  private builtOrder: Int32Array | null = null;
  private order: Int32Array = new Int32Array(0);
  private n = 0;
  private nall = 0;
  private img: Int32Array = new Int32Array(0);
  private from: Int32Array = new Int32Array(0);
  private root: Int32Array = new Int32Array(0);
  private listStart = new Int32Array(0);
  private listNbr = new Int32Array(0);

  // per-event state over native atoms (owned, then ghosts)
  private X = new Float64Array(0);
  private typeN = new Int32Array(0);
  private rho = new Float64Array(0);
  private changed = new Uint8Array(0);
  private readonly at = { m: 0, p: 0 };

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const s = sys.state;
    const ntypes = s.ntypes;
    this.vectorFlag = true;
    this.sizeVector = 2 + ntypes;
    this.extvector = 0;
    this.globalFreq = 1;
    if (sys.geom.triclinic) throw new StyleError('Fix sgcmc does not support non-orthogonal simulation boxes.');
    if (args.length < 3) throw missing('fix sgcmc');
    this.neveryMdsteps = inumeric(args[0]);
    if (this.neveryMdsteps <= 0) throw new StyleError(`Invalid number of MD timesteps ${this.neveryMdsteps}`);
    this.swapFraction = numeric(args[1]);
    if (this.swapFraction < 0.0 || this.swapFraction > 1.0) throw new StyleError(`Invalid fraction ${this.swapFraction} of swap atoms`);
    const temperature = numeric(args[2]);
    if (temperature <= 0) throw new StyleError(`Temperature ${temperature} invalid`);
    this.beta = 1.0 / (sys.units.boltz * temperature);
    let iarg = 3;
    this.deltamu = new Array<number>(ntypes + 1).fill(0);
    if (ntypes < 2) throw new StyleError('Fix sgcmc can only be used in simulations with at least two atom types.');
    for (let i = 2; i <= ntypes; i++, iarg++) {
      if (iarg >= args.length) throw new StyleError('Too few chemical potentials specified');
      this.deltamu[i] = numeric(args[iarg]);
    }
    while (iarg < args.length) {
      const key = args[iarg];
      if (key === 'randseed') {
        if (iarg + 2 > args.length) throw missing('fix sgcmc randseed');
        this.seed = inumeric(args[iarg + 1]);
        if (this.seed <= 0) throw new StyleError(`Random number seed ${this.seed} must be positive`);
        iarg += 2;
      } else if (key === 'window_moves') {
        if (iarg + 2 > args.length) throw missing('fix sgcmc window_moves');
        this.numSamplingWindowMoves = inumeric(args[iarg + 1]);
        if (this.numSamplingWindowMoves <= 0) throw new StyleError(`Invalid number ${this.numSamplingWindowMoves} of sampling window moves`);
        iarg += 2;
      } else if (key === 'window_size') {
        // native names the window_moves keyword in this message
        if (iarg + 2 > args.length) throw missing('fix sgcmc window_moves');
        this.samplingWindowUserSize = numeric(args[iarg + 1]);
        if (this.samplingWindowUserSize < 0.5 || this.samplingWindowUserSize > 1.0) {
          throw new StyleError(`Sampling window size ${this.samplingWindowUserSize} is out of range.`);
        }
        iarg += 2;
      } else if (key === 'variance') {
        if (iarg + 1 + ntypes > args.length) throw missing('fix sgcmc variance');
        iarg++;
        this.kappa = numeric(args[iarg]);
        if (this.kappa < 0) throw new StyleError('Variance constraint parameter must not be negative.');
        iarg++;
        this.targetConcentration = new Array<number>(ntypes + 1).fill(0);
        this.targetConcentration[0] = 1.0;
        this.targetConcentration[1] = 1.0;
        for (let i = 2; i <= ntypes; i++, iarg++) {
          this.targetConcentration[i] = numeric(args[iarg]);
          this.targetConcentration[1] -= this.targetConcentration[i];
        }
        for (let i = 1; i <= ntypes; i++) {
          const c = this.targetConcentration[i];
          if (c < 0.0 || c > 1.0) throw new StyleError(`Target concentration ${c} for species ${i} is out of range`);
        }
      } else if (key === 'serial') {
        this.serialMode = true;
        iarg++;
      } else if (key === 'atomic/energy') {
        if (iarg + 2 > args.length) throw missing('fix sgcmc atomic/energy');
        this.atomicEnergyFlag = logical(args[iarg + 1]);
        this.nativeFullList = !this.atomicEnergyFlag;
        iarg += 2;
      } else {
        throw new StyleError(`Unknown fix sgcmc keyword: ${key}`);
      }
    }
    this.random = new RanPark(this.seed);
    this.localRandom = new RanPark(this.seed + 0);
  }

  /** FixSemiGrandCanonicalMC::init (the pair checks; the cutoff and the tables follow the force-field init). */
  init(): void {
    const sys = this.sys;
    const s = sys.state;
    const subs = atomSubStyles(sys.atomStyle);
    if (!subs.some((st) => PER_TYPE_MASS.has(st))) throw new StyleError('Fix sgcmc requires per atom type masses');
    if (s.rmass) sys.warn('Fix sgcmc will use per atom type masses for velocity initialization');
    if (sys.fixes.filter((f) => f.style === 'sgcmc').length > 1) throw new StyleError('More than one fix sgcmc defined.');
    const pair = sys.ff.pair;
    this.pairEAM = null;
    if (this.atomicEnergyFlag) {
      if (!(pair instanceof PairEAMBase)) throw new StyleError('SGC - Pair style does not support atomic energy method');
    } else if (pair instanceof PairEAMBase) {
      this.pairEAM = pair;
    }
    // count the atoms of each species in the group
    this.speciesCounts = new Array<number>(s.ntypes + 1).fill(0);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.speciesCounts[s.type[i]]++;
  }

  /** After the force-field init and the first neighbor build of a run: cutforce, the EAM tables, the native layout. */
  afterForceInit(): void {
    const sys = this.sys;
    const pair = sys.ff.pair;
    if (!pair) throw new StyleError('Fix sgcmc requires a pair style');
    const nt = sys.state.ntypes + 1;
    let cutforce = 0;
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) cutforce = Math.max(cutforce, pair.cut[i * nt + j]);
    this.interactionRadius = cutforce;
    this.cutforcesq = cutforce * cutforce;
    if (pair instanceof PairEAMBase && (this.pairEAM || this.atomicEnergyFlag)) {
      const t = pair.tables();
      this.rhoTab = t.rho;
      this.z2rTab = t.z2r;
      this.frhoTab = t.frho;
      this.splines.clear();
      const r11 = this.rhoTab[nt + 1];
      if (!r11) throw new StyleError('fix sgcmc: the EAM tables of type 1 are not set');
      this.nr = r11.y.length;
      this.rdr = 1.0 / r11.dx;
    }
    this.buildLayout();
  }

  /** Native has no setup(): no Monte Carlo at the setup of a run. */
  override setup(): void {}

  postNeighbor(): void {
    this.buildLayout();
  }

  postForce(): void {
    if (this.sys.state.step % this.neveryMdsteps === 0) this.doMC();
  }

  /** compute_vector: accepted and rejected swaps of the last event, then the concentration of each species. */
  computeVector(index: number): number {
    if (index === 0) return this.nAcceptedSwaps;
    if (index === 1) return this.nRejectedSwaps;
    index -= 1;
    let totalAtoms = 0;
    for (let i = 0; i < this.speciesCounts.length; i++) totalAtoms += this.speciesCounts[i];
    if (index <= this.sys.state.ntypes) return this.speciesCounts[index] / (totalAtoms > 0 ? totalAtoms : 1);
    return 0.0;
  }

  private spline(t: EAMTable | null): NativeSpline {
    if (!t) throw new StyleError('fix sgcmc: an EAM table is not set for this pair of types');
    let sp = this.splines.get(t);
    if (!sp) { sp = interpolate(t); this.splines.set(t, sp); }
    return sp;
  }
  /** rhor_spline[type2rhor[a][b]]: the density at a type-b site from a type-a neighbour. */
  private rhor(a: number, b: number): NativeSpline {
    return this.spline(this.rhoTab[b * (this.sys.state.ntypes + 1) + a]);
  }
  private z2r(a: number, b: number): NativeSpline {
    return this.spline(this.z2rTab[a * (this.sys.state.ntypes + 1) + b]);
  }
  private frho(t: number): NativeSpline {
    return this.spline(this.frhoTab[t]);
  }

  /**
   * The native atoms of the last neighbor build: local order, ghosts (CommBrick::borders), and the full lists by the
   * build-time positions within the neighbor cutoff (rsq <= cutneighsq), for the owned atoms (EAM) or for every atom
   * (atomic/energy: the full list with ghosts).
   */
  private buildLayout(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    if (this.builtAt === nb.nbuild && this.builtOrder === s.order) return;
    const order = nativeOrder(s);
    const n = s.n;
    const xn = new Float64Array(3 * n);
    for (let k = 0; k < n; k++) {
      const e = order[k];
      xn[3 * k] = s.x[3 * e]; xn[3 * k + 1] = s.x[3 * e + 1]; xn[3 * k + 2] = s.x[3 * e + 2];
    }
    const gh = nativeGhosts(xn, n, sys.geom, nb.cutghost, s.dimension === 2);
    this.order = order;
    this.n = n;
    this.nall = gh.nall;
    this.img = gh.img;
    this.from = gh.from;
    this.root = gh.root;
    this.builtAt = nb.nbuild;
    this.builtOrder = s.order;
    if (!this.pairEAM && !this.atomicEnergyFlag) return;
    const X = gh.x;
    const nall = gh.nall;
    const rows = this.atomicEnergyFlag ? nall : n;
    const cut = nb.cutneighmax;
    const cutsq = cut * cut;
    // cells of size cut over the bounding box of the native atoms
    let lo0 = Infinity, lo1 = Infinity, lo2 = Infinity, hi0 = -Infinity, hi1 = -Infinity, hi2 = -Infinity;
    for (let k = 0; k < nall; k++) {
      const x = X[3 * k], y = X[3 * k + 1], z = X[3 * k + 2];
      if (x < lo0) lo0 = x; if (x > hi0) hi0 = x;
      if (y < lo1) lo1 = y; if (y > hi1) hi1 = y;
      if (z < lo2) lo2 = z; if (z > hi2) hi2 = z;
    }
    const nc0 = Math.max(1, Math.floor((hi0 - lo0) / cut) + 1);
    const nc1 = Math.max(1, Math.floor((hi1 - lo1) / cut) + 1);
    const nc2 = Math.max(1, Math.floor((hi2 - lo2) / cut) + 1);
    const cellOf = new Int32Array(nall);
    const count = new Int32Array(nc0 * nc1 * nc2 + 1);
    for (let k = 0; k < nall; k++) {
      const c0 = Math.min(nc0 - 1, Math.floor((X[3 * k] - lo0) / cut));
      const c1 = Math.min(nc1 - 1, Math.floor((X[3 * k + 1] - lo1) / cut));
      const c2 = Math.min(nc2 - 1, Math.floor((X[3 * k + 2] - lo2) / cut));
      const c = (c2 * nc1 + c1) * nc0 + c0;
      cellOf[k] = c;
      count[c + 1]++;
    }
    for (let c = 0; c < nc0 * nc1 * nc2; c++) count[c + 1] += count[c];
    const fill = count.slice(0, nc0 * nc1 * nc2);
    const members = new Int32Array(nall);
    for (let k = 0; k < nall; k++) members[fill[cellOf[k]]++] = k;
    const start = new Int32Array(rows + 1);
    let buf = new Int32Array(Math.max(64, rows * 32));
    let len = 0;
    for (let i = 0; i < rows; i++) {
      const xi = X[3 * i], yi = X[3 * i + 1], zi = X[3 * i + 2];
      const c = cellOf[i];
      const c0 = c % nc0, c1 = Math.floor(c / nc0) % nc1, c2 = Math.floor(c / (nc0 * nc1));
      for (let d2 = Math.max(0, c2 - 1); d2 <= Math.min(nc2 - 1, c2 + 1); d2++) {
        for (let d1 = Math.max(0, c1 - 1); d1 <= Math.min(nc1 - 1, c1 + 1); d1++) {
          for (let d0 = Math.max(0, c0 - 1); d0 <= Math.min(nc0 - 1, c0 + 1); d0++) {
            const cc = (d2 * nc1 + d1) * nc0 + d0;
            for (let q = count[cc]; q < count[cc + 1]; q++) {
              const j = members[q];
              if (j === i) continue;
              const dx = xi - X[3 * j], dy = yi - X[3 * j + 1], dz = zi - X[3 * j + 2];
              if (dx * dx + dy * dy + dz * dz <= cutsq) {
                if (len === buf.length) { const b2 = new Int32Array(2 * len); b2.set(buf); buf = b2; }
                buf[len++] = j;
              }
            }
          }
        }
      }
      start[i + 1] = len;
    }
    this.listStart = start;
    this.listNbr = buf.slice(0, len);
  }

  /** The positions, types and (EAM) densities of the native atoms at the start of an event. */
  private prepare(): void {
    this.buildLayout();
    const sys = this.sys;
    const s = sys.state;
    const { n, nall, order, from, img } = this;
    const g = sys.geom;
    const prd = [g.hi[0] - g.lo[0], g.hi[1] - g.lo[1], g.hi[2] - g.lo[2]];
    if (this.X.length < 3 * nall) {
      this.X = new Float64Array(3 * nall);
      this.typeN = new Int32Array(nall);
      this.rho = new Float64Array(nall);
      this.changed = new Uint8Array(nall);
    }
    const X = this.X, typeN = this.typeN;
    for (let k = 0; k < n; k++) {
      const e = order[k];
      X[3 * k] = s.x[3 * e]; X[3 * k + 1] = s.x[3 * e + 1]; X[3 * k + 2] = s.x[3 * e + 2];
      typeN[k] = s.type[e];
    }
    // ghosts: the forward communication of the step (x of the atom sent plus the swap's box length) and the types of
    // their owners (the last forward communication of types)
    for (let k = n; k < nall; k++) {
      const f = from[k];
      for (let d = 0; d < 3; d++) {
        const shift = img[3 * k + d] - img[3 * f + d];
        X[3 * k + d] = shift === 0 ? X[3 * f + d] : X[3 * f + d] + shift * prd[d];
      }
      typeN[k] = typeN[this.root[k]];
    }
    if (!this.pairEAM) return;
    // the densities of the force evaluation (PairEAM::compute), then the forward communication to the ghosts
    const rho = this.rho;
    const at = this.at;
    for (let i = 0; i < n; i++) {
      const xi = X[3 * i], yi = X[3 * i + 1], zi = X[3 * i + 2];
      const itype = typeN[i];
      let sum = 0.0;
      for (let q = this.listStart[i]; q < this.listStart[i + 1]; q++) {
        const j = this.listNbr[q];
        const dx = xi - X[3 * j], dy = yi - X[3 * j + 1], dz = zi - X[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= this.cutforcesq) continue;
        this.rIndex(Math.sqrt(rsq), at);
        sum += value(this.rhor(typeN[j], itype), at.m, at.p);
      }
      rho[i] = sum;
    }
    for (let k = n; k < nall; k++) rho[k] = rho[from[k]];
  }

  /** p = r rdr + 1, m = MIN(int(p), nr - 1), p -= m, p = MIN(p, 1). */
  private rIndex(r: number, out: { m: number; p: number }): void {
    let p = r * this.rdr + 1.0;
    let m = Math.trunc(p);
    m = Math.min(m, this.nr - 1);
    p -= m;
    out.p = Math.min(p, 1.0);
    out.m = m;
  }

  /** FixSemiGrandCanonicalMC::doMC. */
  private doMC(): void {
    const sys = this.sys;
    const s = sys.state;
    const ntypes = s.ntypes;
    const natoms = s.n;
    this.prepare();
    let nAcceptedSwapsLocal = 0;
    let nRejectedSwapsLocal = 0;
    let oldSpecies = 0, newSpecies = 0;
    const deltaN = new Array<number>(ntypes + 1).fill(0);
    const deltaNGlobal = new Array<number>(ntypes + 1).fill(0);
    for (let w = 0; w < this.numSamplingWindowMoves; w++) {
      this.changed.fill(0);
      const oversizeWindow = this.placeSamplingWindow();
      const nDice = Math.trunc(this.swapFraction * this.numFixAtomsLocal / this.numSamplingWindowMoves);
      const largestnDice = nDice;
      const diceProbability = nDice / largestnDice;
      for (let j = 0; j < largestnDice; j++) {
        let deltaE = 0;
        deltaN.fill(0);
        let selectedAtom = -1;
        if (this.localRandom.uniform() <= diceProbability) {
          if (this.samplingWindowAtoms.length === 0) throw new StyleError('fix sgcmc: the sampling window holds no atoms');
          const index = Math.trunc(this.localRandom.uniform() * this.samplingWindowAtoms.length);
          // a full list without ghosts: ilist[ii] = ii, the native local index
          selectedAtom = this.samplingWindowAtoms[index];
          oldSpecies = this.typeN[selectedAtom];
          if (ntypes > 2) {
            newSpecies = Math.trunc(this.localRandom.uniform() * (ntypes - 1)) + 1;
            if (newSpecies >= oldSpecies) newSpecies++;
          } else {
            newSpecies = oldSpecies === 1 ? 2 : 1;
          }
          deltaN[oldSpecies] = -1;
          deltaN[newSpecies] = +1;
          if (this.atomicEnergyFlag) deltaE = this.energyChangeEatom(selectedAtom, oldSpecies, newSpecies);
          else if (this.pairEAM) deltaE = this.energyChangeEAM(selectedAtom, oldSpecies, newSpecies);
          else deltaE = this.energyChangeGeneric(selectedAtom, oldSpecies, newSpecies);
          let dm = 0.0;
          if (this.serialMode && this.kappa !== 0.0) {
            for (let i = 2; i <= ntypes; i++) {
              dm += (this.deltamu[i] + this.kappa / natoms * (2.0 * this.speciesCounts[i] + deltaN[i])) * deltaN[i];
            }
          } else {
            for (let i = 2; i <= ntypes; i++) dm += this.deltamu[i] * deltaN[i];
          }
          const deltaB = -(deltaE + dm) * this.beta;
          if (deltaB < 0.0) {
            if (deltaB < libmLog(this.localRandom.uniform())) {
              deltaN.fill(0);
              selectedAtom = -1;
              deltaE = 0;
            }
          }
        }
        if (this.kappa !== 0.0 && !this.serialMode) {
          // the second rejection test of the variance-constrained ensemble (one processor: the global sum is deltaN)
          for (let i = 0; i <= ntypes; i++) deltaNGlobal[i] = deltaN[i];
          let A = 0.0;
          for (let i = 1; i <= ntypes; i++) {
            A += deltaNGlobal[i] * deltaNGlobal[i];
            A += 2.0 * deltaNGlobal[i] * (this.speciesCounts[i] - Math.trunc(this.targetConcentration[i] * natoms));
          }
          const deltaB = -this.beta * (this.kappa / natoms) * A;
          if (deltaB < 0.0) {
            if (deltaB < libmLog(this.random.uniform())) {
              deltaN.fill(0);
              deltaNGlobal.fill(0);
              selectedAtom = -1;
            }
          }
          for (let i = 1; i <= ntypes; i++) this.speciesCounts[i] += deltaNGlobal[i];
        } else if (this.serialMode) {
          for (let i = 1; i <= ntypes; i++) this.speciesCounts[i] += deltaN[i];
        }
        if (selectedAtom >= 0) {
          if (this.pairEAM && !this.atomicEnergyFlag) this.flipAtomEAM(selectedAtom, oldSpecies, newSpecies);
          else this.flipAtom(selectedAtom, oldSpecies, newSpecies);
          nAcceptedSwapsLocal++;
        } else {
          nRejectedSwapsLocal++;
        }
        if (oversizeWindow) this.communicateRhoAndTypes();
      }
      if (!oversizeWindow) this.communicateRhoAndTypes();
    }
    this.nAcceptedSwaps = nAcceptedSwapsLocal;
    this.nRejectedSwaps = nRejectedSwapsLocal;
    if (this.kappa === 0.0 && !this.serialMode) {
      const counts = new Array<number>(ntypes + 1).fill(0);
      for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) counts[s.type[i]]++;
      this.speciesCounts = counts;
    }
    // the engine's copies of the types (its ghosts take their owners' types, as the last forward communication)
    this.forwardTypesEngine();
  }

  /** Every engine atom (owned and ghost) takes the type of its owner. */
  private forwardTypesEngine(): void {
    const s = this.sys.state;
    const nb = this.sys.nb;
    for (let k = 0; k < nb.nall; k++) nb.typeall[k] = s.type[nb.owner[k]];
  }

  /** placeSamplingWindow: the window at the next corner and its atoms; true when the window is oversized. */
  private placeSamplingWindow(): boolean {
    const g = this.sys.geom;
    const s = this.sys.state;
    let oversizeWindow = false;
    const lo = [0, 0, 0], hi = [0, 0, 0], margin = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      margin[i] = this.interactionRadius * 2.0;
      if (this.samplingWindowUserSize > 0.0) {
        margin[i] = (g.hi[i] - g.lo[i]) * (1.0 - this.samplingWindowUserSize);
        if (margin[i] < this.interactionRadius * 2.0) oversizeWindow = true;
      }
      const shift = ((this.samplingWindowPosition >> i) & 1) * margin[i];
      lo[i] = g.lo[i] + shift;
      hi[i] = g.hi[i] + shift - margin[i];
      if (hi[i] - lo[i] + 1e-6 < (g.hi[i] - g.lo[i]) * 0.5) {
        throw new StyleError('Per-node simulation cell is too small for fix sgcmc. Processor cell size must be at least 4 times cutoff radius.');
      }
    }
    this.samplingWindowPosition += 1;
    const list: number[] = [];
    this.numFixAtomsLocal = 0;
    const X = this.X;
    for (let ii = 0; ii < this.n; ii++) {
      if ((s.mask[this.order[ii]] & this.groupBit) === 0) continue;
      this.numFixAtomsLocal++;
      const x = X[3 * ii], y = X[3 * ii + 1], z = X[3 * ii + 2];
      if (x >= lo[0] && x < hi[0] && y >= lo[1] && y < hi[1] && z >= lo[2] && z < hi[2]) {
        let multiplicity = 1;
        for (let k = 0; k < 3; k++) {
          const xk = X[3 * ii + k];
          if (xk < g.lo[k] + margin[k] || xk > g.hi[k] - margin[k]) multiplicity *= 2;
        }
        for (let m = 0; m < multiplicity; m++) list.push(ii);
      }
    }
    this.samplingWindowAtoms = list;
    return oversizeWindow;
  }

  /**
   * communicateRhoAndTypes: the reverse communication of the changed densities of the ghosts (each ghost's value goes
   * to the atom it was sent from unless that atom changed itself; swaps from the last to the first), then the forward
   * communication of the types and the densities of the unchanged atoms (swaps from the first to the last).
   */
  private communicateRhoAndTypes(): void {
    const { n, nall, from, changed, rho, typeN } = this;
    if (this.pairEAM && !this.atomicEnergyFlag) {
      for (let k = nall - 1; k >= n; k--) {
        const t = from[k];
        if (!changed[t]) rho[t] = rho[k];
      }
      for (let k = n; k < nall; k++) {
        typeN[k] = typeN[from[k]];
        if (!changed[k]) rho[k] = rho[from[k]];
      }
    } else {
      for (let k = n; k < nall; k++) typeN[k] = typeN[from[k]];
    }
  }

  /** computeEnergyChangeEAM: the change of the pair terms and the embedding energies when flipAtom changes species. */
  private energyChangeEAM(flipAtom: number, oldSpecies: number, newSpecies: number): number {
    const X = this.X, typeN = this.typeN, rho = this.rho;
    const at = this.at;
    let newTotalRhoI = 0.0;
    let deltaE = 0.0;
    const xi = X[3 * flipAtom], yi = X[3 * flipAtom + 1], zi = X[3 * flipAtom + 2];
    for (let q = this.listStart[flipAtom]; q < this.listStart[flipAtom + 1]; q++) {
      const j = this.listNbr[q];
      const delx = xi - X[3 * j], dely = yi - X[3 * j + 1], delz = zi - X[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= this.cutforcesq) continue;
      const jtype = typeN[j];
      const r = Math.sqrt(rsq);
      this.rIndex(r, at);
      const m = at.m, p = at.p;
      const oldz2 = value(this.z2r(oldSpecies, jtype), m, p);
      const newz2 = value(this.z2r(newSpecies, jtype), m, p);
      deltaE += (newz2 - oldz2) / r;
      const oldrhoContr = value(this.rhor(oldSpecies, jtype), m, p);
      const newrhoContr = value(this.rhor(newSpecies, jtype), m, p);
      const deltaRho = newrhoContr - oldrhoContr;
      newTotalRhoI += value(this.rhor(jtype, newSpecies), m, p);
      const fj = this.frho(jtype);
      embedAt(fj, rho[j], at);
      const oldF = value(fj, at.m, at.p);
      embedAt(fj, rho[j] + deltaRho, at);
      const newF = value(fj, at.m, at.p);
      deltaE += newF - oldF;
    }
    embedAt(this.frho(oldSpecies), rho[flipAtom], at);
    const oldF = value(this.frho(oldSpecies), at.m, at.p);
    embedAt(this.frho(newSpecies), newTotalRhoI, at);
    const newF = value(this.frho(newSpecies), at.m, at.p);
    deltaE += newF - oldF;
    return deltaE;
  }

  /** flipAtomEAM: the new species, the velocity rescaled, the densities of the neighbours and of the flipped atom. */
  private flipAtomEAM(flipAtom: number, oldSpecies: number, newSpecies: number): void {
    const X = this.X, typeN = this.typeN, rho = this.rho, changed = this.changed;
    const at = this.at;
    let newTotalRhoI = 0.0;
    this.setSpecies(flipAtom, oldSpecies, newSpecies);
    changed[flipAtom] = 1;
    const xi = X[3 * flipAtom], yi = X[3 * flipAtom + 1], zi = X[3 * flipAtom + 2];
    for (let q = this.listStart[flipAtom]; q < this.listStart[flipAtom + 1]; q++) {
      const j = this.listNbr[q];
      const delx = xi - X[3 * j], dely = yi - X[3 * j + 1], delz = zi - X[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= this.cutforcesq) continue;
      const jtype = typeN[j];
      this.rIndex(Math.sqrt(rsq), at);
      const m = at.m, p = at.p;
      const oldrhoContr = value(this.rhor(oldSpecies, jtype), m, p);
      const newrhoContr = value(this.rhor(newSpecies, jtype), m, p);
      rho[j] += newrhoContr - oldrhoContr;
      newTotalRhoI += value(this.rhor(jtype, newSpecies), m, p);
      changed[j] = 1;
    }
    rho[flipAtom] = newTotalRhoI;
  }

  /** flipAtomGeneric and flipAtomEatom: the new species and the velocity rescaled. */
  private flipAtom(flipAtom: number, oldSpecies: number, newSpecies: number): void {
    this.setSpecies(flipAtom, oldSpecies, newSpecies);
    this.changed[flipAtom] = 1;
  }

  /** atom->type of the owned atom and the velocity rescaled by sqrt(mass[old] / mass[new]) to keep its kinetic energy. */
  private setSpecies(k: number, oldSpecies: number, newSpecies: number): void {
    const s = this.sys.state;
    const e = this.order[k];
    this.typeN[k] = newSpecies;
    s.type[e] = newSpecies;
    const vScaleFactor = Math.sqrt(s.massByType[oldSpecies] / s.massByType[newSpecies]);
    s.v[3 * e] *= vScaleFactor;
    s.v[3 * e + 1] *= vScaleFactor;
    s.v[3 * e + 2] *= vScaleFactor;
  }

  /** PairEAM::compute_atomic_energy over the full list with ghosts. */
  private atomicEnergy(i: number): number {
    const X = this.X, typeN = this.typeN;
    const at = this.at;
    let Ei = 0.0;
    let rhoi = 0.0;
    const xi = X[3 * i], yi = X[3 * i + 1], zi = X[3 * i + 2];
    const itype = typeN[i];
    for (let q = this.listStart[i]; q < this.listStart[i + 1]; q++) {
      const j = this.listNbr[q];
      const delx = xi - X[3 * j], dely = yi - X[3 * j + 1], delz = zi - X[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= this.cutforcesq) continue;
      const jtype = typeN[j];
      const r = Math.sqrt(rsq);
      this.rIndex(r, at);
      const z2 = value(this.z2r(jtype, itype), at.m, at.p);
      Ei += 0.5 * z2 / r;
      rhoi += value(this.rhor(jtype, itype), at.m, at.p);
    }
    const f = this.frho(itype);
    embedAt(f, rhoi, at);
    Ei += value(f, at.m, at.p);
    return Ei;
  }

  /** computeEnergyChangeEatom: the per-atom energies of flipAtom and of its list neighbours, after minus before. */
  private energyChangeEatom(flipAtom: number, oldSpecies: number, newSpecies: number): number {
    const k0 = this.listStart[flipAtom], k1 = this.listStart[flipAtom + 1];
    let Eold = this.atomicEnergy(flipAtom);
    for (let q = k0; q < k1; q++) Eold += this.atomicEnergy(this.listNbr[q]);
    this.typeN[flipAtom] = newSpecies;
    let Enew = this.atomicEnergy(flipAtom);
    for (let q = k0; q < k1; q++) Enew += this.atomicEnergy(this.listNbr[q]);
    this.typeN[flipAtom] = oldSpecies;
    return Enew - Eold;
  }

  /**
   * computeEnergyChangeGeneric: the total energy with the trial species and with the old one (the types sent to the
   * ghosts before each), then setup_minimal(0), which recomputes the forces and runs the fixes' setup.
   */
  private energyChangeGeneric(flipAtom: number, oldSpecies: number, newSpecies: number): number {
    const s = this.sys.state;
    const e = this.order[flipAtom];
    if (flipAtom >= 0) s.type[e] = newSpecies;
    this.forwardTypesEngine();
    const newEnergy = this.computeTotalEnergy();
    if (flipAtom >= 0) s.type[e] = oldSpecies;
    this.forwardTypesEngine();
    const oldEnergy = this.computeTotalEnergy();
    this.setupMinimal();
    return newEnergy - oldEnergy;
  }

  /** computeTotalEnergy: the force styles' energies, read as compute thermo_pe sums them. */
  private computeTotalEnergy(): number {
    const sys = this.sys;
    const a = sys.computeForcesInRun(sys.computeFlags());
    let e = 0;
    e += a.evdwl + a.ecoul;
    e += a.ebond;
    e += a.eangle;
    e += a.edihed;
    e += a.eimp;
    e += a.elong;
    e += sys.fixEnergy();
    return e;
  }

  /** Verlet::setup_minimal(0): no reneighboring; setup_pre_force, the forces, setup_pre_reverse, the fixes' setup. */
  private setupMinimal(): void {
    const sys = this.sys;
    for (const f of sys.fixes) f.setupPreForce?.();
    sys.computeForcesInRun(sys.computeFlags());
    sys.forcesCurrent();
    for (const f of sys.fixes) f.setupPreReverse?.();
    for (const f of sys.fixes) f.setup();
  }
}
