import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext, pairCoeffArgs, potentialFile } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { eamDensities, eamForces, eamGrid, type EAMKernelTables } from './eamKernel';
import { parseInt_, parseNum } from '../util';

/*
 * pair_style eam, eam/alloy, eam/fs — docs.lammps.org/pair_eam.html
 * (source: plans/lammps-docs/pair_eam.rst)
 *
 * Syntax (verbatim):
 *    pair_style style
 * * style = *eam* or *eam/alloy* or *eam/cd* or *eam/cd/old* or *eam/fs* or *eam/he*
 *
 * Style eam formula (verbatim):
 *    E_i = F_\alpha \left(\sum_{j \neq i}\ \rho_\beta (r_{ij})\right) +
 *          \frac{1}{2} \sum_{j \neq i} \phi_{\alpha\beta} (r_{ij})
 * "where F is the embedding energy which is a function of the atomic
 * electron density rho, phi is a pair potential interaction, and alpha
 * and beta are the element types of atoms I and J." "Both summations in
 * the formula are over all neighbors J of atom I within the cutoff
 * distance."
 *
 * Style eam/fs formula (verbatim):
 *    E_i = F_\alpha \left(\sum_{j \neq i}\
 *    \rho_{\alpha\beta} (r_{ij})\right) +
 *    \frac{1}{2} \sum_{j \neq i} \phi_{\alpha\beta} (r_{ij})
 * "where :math:`\rho_{\alpha\beta}` refers to the density contributed by a
 * neighbor atom J of element :math:`\beta` at the site of atom I of element
 * :math:`\alpha`."
 *
 * funcfl pair term (verbatim):
 *    r \cdot \phi = 27.2 \cdot 0.529 \cdot Z_i \cdot Z_j
 * "where 1 Hartree = 27.2 eV and 1 Bohr = 0.529 Angstroms."
 *
 * funcfl file format (verbatim):
 * "line 1: comment (ignored)", "line 2: atomic number, mass, lattice
 * constant, lattice type (e.g. FCC)", "line 3: Nrho, drho, Nr, dr, cutoff";
 * arrays "embedding function F(rho) (Nrho values)", "effective charge
 * function Z(r) (Nr values)", "density function rho(r) (Nr values)"; "The
 * values for each array can be listed as multiple values per line, so long
 * as each array starts on a new line." On line 2 "all values but the mass
 * are ignored by LAMMPS"; "the specified cutoff becomes the pairwise cutoff
 * used by LAMMPS for the potential".
 *
 * setfl file format (verbatim): "lines 1,2,3 = comments (ignored)",
 * "line 4: Nelements Element1 Element2 ... ElementN",
 * "line 5: Nrho, drho, Nr, dr, cutoff"; per element "line 1 = atomic
 * number, mass, lattice constant, lattice type (e.g. FCC)", "embedding
 * function F(rho) (Nrho values)", "density function rho(r) (Nr values)";
 * "Note that the cutoff (in Angstroms) is a global value, valid for all
 * pairwise interactions for all element pairings." Then "only phi arrays
 * with i >= j are listed, in the following order: i,j = (1,1), (2,1),
 * (2,2), (3,1), (3,2), (3,3), (4,1), ..., (Nelements, Nelements)" and
 * "the tabulated values for each phi function are listed in setfl files
 * directly as r\*phi (in units of eV-Angstroms)".
 *
 * FS file format: header "identical to an FS EAM file" (= setfl); per
 * element section: "line 1 = atomic number, mass, lattice constant, lattice
 * type (e.g. FCC)", "embedding function F(rho) (Nrho values)", "density
 * function :math:`\rho_{1\beta} (r)` for element :math:`\beta` at element 1
 * (Nr values)", ... up to rho_{Nβ}
 * so "there are Nelements^2 of them listed in the file" and "the rho(r)
 * arrays in Finnis/Sinclair can be asymmetric"; "the phi(r) arrays are
 * still symmetric, so only phi arrays for i >= j are listed" (as r*phi).
 *
 * Mapping (eam/alloy and eam/fs pair_coeff): "filename" then "N element
 * names = mapping of setfl elements to atom types"; "The first 2 arguments
 * must be \* \* so as to span all LAMMPS atom types."; "If a mapping value is
 * specified as NULL, the mapping is not performed." For style eam: "a
 * potential file must be assigned to each I,I pair of atom types by using
 * one or more pair_coeff commands, each with a single argument: filename";
 * wildcard ranges assign the file to the I,I pairs they cover ("type pairs
 * 1,2 and 2,1 are ignored").
 *
 * funcfl mixing for I != J: "For the alloy case LAMMPS mixes the
 * single-element potentials to produce alloy potentials, the same way that
 * DYNAMO does." — the mixed pair term is built from the product of the two
 * effective charges via the r·phi formula above (so phi_AB(r) =
 * 27.2·0.529·Z_A(r)·Z_B(r)/r); the density at an atom of element A from a
 * neighbor of element B is rho_B(r), B's own tabulated density function;
 * each element keeps its own embedding function F_A. The two files must
 * share the same Nr and dr to be mixed; the mixed pair cutoff is the
 * average of the two file cutoffs.
 *
 * Beyond the last tabulated density (verbatim): "LAMMPS will assume a
 * linearly increasing embedding energy for electron densities beyond the
 * maximum tabulated value."
 *
 * "This pair style does not support the pair_modify shift, table, and tail
 * options."
 *
 * Default (verbatim): "none"
 *
 * Units metadata (pair_coeff.rst): "The first line of potential files may
 * contain metadata with upper case tags followed their value. ... The
 * UNITS: tag indicates the units setting required for this particular
 * potential file. If the potential file was created for a different sets
 * of units, LAMMPS will terminate with an error. If the potential file
 * does not contain the tag, no check will be made".
 *
 * Masses (mass.rst): "pair_style eam and pair_style bop commands define
 * the masses of atom types in their respective potential files, in which
 * case the mass command is normally not used." A mass set with the mass
 * command before pair_coeff is not replaced.
 *
 * Interpolation: the doc pages do not state how the tables are evaluated
 * between tabulated points; the measured scheme (cubic Hermite with
 * finite-difference slopes) is recorded next to the Spline class below.
 */

const HARTREE_EV = 27.2;
const BOHR_ANG = 0.529;

/**
 * Piecewise cubic Hermite interpolation over a uniform grid x_k = k*dx,
 * k = 0..n-1, with node slopes (per grid step) from finite differences:
 *   s_0 = y_1 - y_0,  s_1 = (y_2 - y_0)/2,  s_{n-2} = (y_{n-1} - y_{n-3})/2,
 *   s_{n-1} = y_{n-1} - y_{n-2},
 *   s_k = ((y_{k-2} - y_{k+2}) + 8 (y_{k+1} - y_{k-1})) / 12 otherwise
 * (the five-point central difference), and on [x_k, x_{k+1}]
 *   S(p) = ((c3 p + c2) p + s_k) p + y_k,  p = x/dx - k clamped to [0, 1],
 *   c2 = 3 (y_{k+1} - y_k) - 2 s_k - s_{k+1},  c3 = s_k + s_{k+1} - 2 (y_{k+1} - y_k).
 * The docs do not say how the tables are interpolated. Measured with native
 * LAMMPS (black box): a two-atom eam/alloy probe on a deliberately coarse
 * 20-point table (rough values, distances between nodes) matches this
 * scheme to 1e-15 in energy and force, while a natural cubic spline is off
 * by up to 2%.
 */
class Spline {
  readonly n: number;
  readonly dx: number;
  readonly y: Float64Array;
  private readonly s: Float64Array;
  private readonly c2: Float64Array;
  private readonly c3: Float64Array;
  /**
   * The seven coefficients of each node k (7 k .. 7 k + 6), as native LAMMPS lays out its spline arrays (src/MANYBODY/
   * pair_eam.cpp interpolate at b332f28): the derivative's 3 c3/dx, 2 c2/dx, s/dx, then c3, c2, s and y. The last node
   * has c2 = c3 = 0 (it starts no interval).
   */
  readonly coef: Float64Array;
  /** 1/dx: the table index of x is x * rdx (native's rdr and rdrho). */
  readonly rdx: number;

  constructor(y: Float64Array, dx: number) {
    const n = y.length;
    if (n < 2 || !(dx > 0)) throw new StyleError('EAM table needs at least 2 values and a positive spacing');
    this.n = n;
    this.dx = dx;
    this.y = y;
    const s = new Float64Array(n);
    if (n === 2) s[0] = s[1] = y[1] - y[0];
    else {
      s[0] = y[1] - y[0];
      s[n - 1] = y[n - 1] - y[n - 2];
      s[1] = 0.5 * (y[2] - y[0]);
      s[n - 2] = 0.5 * (y[n - 1] - y[n - 3]);
      for (let k = 2; k < n - 2; k++) s[k] = ((y[k - 2] - y[k + 2]) + 8 * (y[k + 1] - y[k - 1])) / 12;
    }
    this.s = s;
    this.c2 = new Float64Array(n - 1);
    this.c3 = new Float64Array(n - 1);
    for (let k = 0; k < n - 1; k++) {
      const dy = y[k + 1] - y[k];
      this.c2[k] = 3 * dy - 2 * s[k] - s[k + 1];
      this.c3[k] = s[k] + s[k + 1] - 2 * dy;
    }
    const coef = new Float64Array(7 * n);
    for (let k = 0; k < n; k++) {
      const c3 = k < n - 1 ? this.c3[k] : 0, c2 = k < n - 1 ? this.c2[k] : 0;
      coef[7 * k] = (3.0 * c3) / dx;
      coef[7 * k + 1] = (2.0 * c2) / dx;
      coef[7 * k + 2] = s[k] / dx;
      coef[7 * k + 3] = c3;
      coef[7 * k + 4] = c2;
      coef[7 * k + 5] = s[k];
      coef[7 * k + 6] = y[k];
    }
    this.coef = coef;
    this.rdx = 1.0 / dx;
  }

  // the interval index k and the local coordinate p in [0, 1], written inline in eval and deriv (no array per call)
  eval(x: number): number {
    let p = x / this.dx;
    let k = Math.floor(p);
    if (k < 0) k = 0;
    else if (k > this.n - 2) k = this.n - 2;
    p -= k;
    if (p > 1) p = 1;
    return ((this.c3[k] * p + this.c2[k]) * p + this.s[k]) * p + this.y[k];
  }

  deriv(x: number): number {
    let p = x / this.dx;
    let k = Math.floor(p);
    if (k < 0) k = 0;
    else if (k > this.n - 2) k = this.n - 2;
    p -= k;
    if (p > 1) p = 1;
    return ((3 * this.c3[k] * p + 2 * this.c2[k]) * p + this.s[k]) / this.dx;
  }

  /** Slope at the last tabulated point (linear extrapolation of F), per grid step. */
  endSlope(): number {
    return this.s[this.n - 1];
  }
}

/** A tabulated function of an EAM file: y[k] at x = k * dx, k = 0 .. y.length - 1. */
export interface EAMTable {
  readonly y: Float64Array;
  readonly dx: number;
}

/** Embedding energy F(rho): spline in [0, rhoMax], linear above it (doc quote in the header). */
class Embed {
  private constructor(readonly sp: Spline, readonly rhoMax: number, private readonly slope: number) {}

  /** The tabulated values of F(rho) and their spacing drho. */
  get table(): EAMTable {
    return this.sp;
  }

  static of(y: Float64Array, drho: number): Embed {
    const sp = new Spline(y, drho);
    return new Embed(sp, (y.length - 1) * drho, sp.endSlope() / drho);
  }

  eval(rho: number): number {
    if (rho > this.rhoMax) return this.sp.eval(this.rhoMax) + this.slope * (rho - this.rhoMax);
    return this.sp.eval(rho < 0 ? 0 : rho);
  }

  deriv(rho: number): number {
    if (rho > this.rhoMax) return this.slope;
    return this.sp.deriv(rho < 0 ? 0 : rho);
  }
}

const ZERO_EMBED = Embed.of(new Float64Array([0, 0]), 1);

interface EAMFile {
  name: string;
  nrho: number;
  drho: number;
  nr: number;
  dr: number;
  cutoff: number;
  /** Element names (funcfl: none). */
  names: string[];
  /** Mass per element. */
  masses: number[];
  /** Embedding function per element. */
  F: Embed[];
  /** funcfl: raw effective charge Z(r) table. */
  z?: Float64Array;
  /** funcfl and eam/alloy: one density function per element; eam/fs: [neighbour beta][site alpha]. */
  rho?: Spline[];
  fsRho?: Spline[][];
  /** setfl/fs: r*phi splines, [i][j] with i >= j (0-based element indices). */
  phi?: Spline[][];
}

/** Reads numeric tables that span lines ("each array starts on a new line"). */
class Tables {
  private readonly lines: string[];
  private pos = 0;
  private res: number[] = [];

  constructor(text: string) {
    this.lines = text.split(/\r?\n/);
  }

  line(what: string): string {
    if (this.pos >= this.lines.length) throw new StyleError(`EAM potential file: unexpected end of file while reading ${what}`);
    return this.lines[this.pos++];
  }

  take(count: number, what: string): Float64Array {
    const out = new Float64Array(count);
    let k = 0;
    while (k < count) {
      if (this.res.length) {
        out[k++] = this.res.shift() as number;
        continue;
      }
      const line = this.line(what).trim();
      if (line === '') continue;
      for (const t of line.split(/\s+/)) {
        const v = Number(t);
        if (!Number.isFinite(v)) throw new StyleError(`EAM potential file: expected numbers for ${what}, found '${t}'`);
        this.res.push(v);
      }
    }
    return out;
  }
}

/** Line 2 / element line: atomic number, mass, lattice constant, lattice type; only the mass is used. */
const massOfElementLine = (line: string, name: string): number => {
  const t = line.trim().split(/\s+/);
  if (t.length < 2) throw new StyleError(`EAM potential file ${name}: element line needs an atomic number and a mass`);
  const mass = parseNum(t[1], 'mass');
  if (mass <= 0) throw new StyleError(`EAM potential file ${name}: mass must be > 0 (got ${t[1]})`);
  return mass;
};

/** pair_coeff.rst: the UNITS: metadata tag on line 1, when present, must match the unit style. */
const checkUnitsTag = (line: string, ctx: StyleContext, name: string): void => {
  const m = /UNITS:\s*(\S+)/.exec(line);
  if (m && ctx.s && m[1] !== ctx.s.units.style) {
    throw new StyleError(`EAM potential file ${name} requires ${m[1]} units but the simulation uses ${ctx.s.units.style} units`);
  }
};

const readGrid = (t: Tables, name: string): { nrho: number; drho: number; nr: number; dr: number; cutoff: number } => {
  const g = t.take(5, 'the grid line (Nrho, drho, Nr, dr, cutoff)');
  const nrho = Math.trunc(g[0]);
  const nr = Math.trunc(g[2]);
  const drho = g[1], dr = g[3], cutoff = g[4];
  if (nrho < 2 || nr < 2) throw new StyleError(`EAM potential file ${name}: Nrho and Nr must be at least 2`);
  if (!(drho > 0) || !(dr > 0) || !(cutoff > 0)) throw new StyleError(`EAM potential file ${name}: drho, dr and cutoff must be > 0`);
  // measured with native LAMMPS (black box): a cutoff beyond the last table point is accepted;
  // between the two, the tables hold the value of the last interval at p = 1 (Spline clamps it)
  return { nrho, drho, nr, dr, cutoff };
};

/** setfl/fs phi array order: (1,1), (2,1), (2,2), (3,1), ... */
const rphiPairs = (nelem: number): [number, number][] => {
  const out: [number, number][] = [];
  for (let i = 0; i < nelem; i++) for (let j = 0; j <= i; j++) out.push([i, j]);
  return out;
};

/** Parses a DYNAMO funcfl file (style eam). */
const parseFuncfl = (text: string, name: string, ctx: StyleContext): EAMFile => {
  const t = new Tables(text);
  checkUnitsTag(t.line('the comment line'), ctx, name);
  const mass = massOfElementLine(t.line('the element line'), name);
  const g = readGrid(t, name);
  const F = Embed.of(t.take(g.nrho, 'the embedding function F(rho)'), g.drho);
  const z = t.take(g.nr, 'the effective charge function Z(r)');
  const rho = new Spline(t.take(g.nr, 'the density function rho(r)'), g.dr);
  return { name, ...g, names: [], masses: [mass], F: [F], z, rho: [rho] };
};

/** Parses a setfl (fs = false) or Finnis-Sinclair (fs = true) file. */
const parseSetfl = (text: string, name: string, ctx: StyleContext, fs: boolean): EAMFile => {
  const t = new Tables(text);
  checkUnitsTag(t.line('comment line 1'), ctx, name);
  t.line('comment line 2');
  t.line('comment line 3');
  const head = t.line('the element list line').trim().split(/\s+/);
  const nelem = parseInt_(head[0], 'Nelements');
  if (nelem < 1) throw new StyleError(`EAM potential file ${name}: Nelements must be >= 1`);
  if (head.length < 1 + nelem) {
    throw new StyleError(`EAM potential file ${name}: line 4 must list ${nelem} element names after Nelements`);
  }
  const names = head.slice(1, 1 + nelem);
  const g = readGrid(t, name);
  const masses: number[] = [];
  const F: Embed[] = [];
  const rho: Spline[] = [];
  const fsRho: Spline[][] = [];
  for (let e = 0; e < nelem; e++) {
    masses.push(massOfElementLine(t.line(`the element ${names[e]} line`), name));
    F.push(Embed.of(t.take(g.nrho, `the embedding function of element ${names[e]}`), g.drho));
    if (!fs) {
      rho.push(new Spline(t.take(g.nr, `the density function of element ${names[e]}`), g.dr));
    } else {
      const arr: Spline[] = [];
      for (let a = 0; a < nelem; a++) {
        arr.push(new Spline(t.take(g.nr, `the density function rho_${a + 1}${e + 1}(r)`), g.dr));
      }
      fsRho.push(arr);
    }
  }
  const phi: Spline[][] = [];
  for (const [i, j] of rphiPairs(nelem)) {
    const y = t.take(g.nr, `the pair potential r*phi for elements ${names[j]}-${names[i]}`);
    phi[i] = phi[i] ?? [];
    phi[i][j] = new Spline(y, g.dr);
  }
  return { name, ...g, names, masses, F, rho: fs ? undefined : rho, fsRho: fs ? fsRho : undefined, phi };
};

/**
 * Shared compute for all three styles. The densities rho_i are summed over
 * the FULL neighbor list (every neighbor of every owned atom, owned or
 * ghost) and copied to the ghosts through owner; F'(rho_i) likewise. The
 * force loop then runs over the HALF list, where every physical pair
 * appears exactly once (owned-owned with j > i, or an owned atom with the
 * ghost image that lies above it), so each atom receives its force once
 * and the force-field virial sum x.f over owned and ghost atoms is the
 * pairwise sum r_ij.F_ij (virialFdotr). The embedding part of the pair
 * force, -F'_i(rho_i) rho'_{ij}(r)/r on atom i and -F'_j(rho_j)
 * rho'_{ji}(r)/r on atom j, follows from dE/dx of the documented energy.
 */
export abstract class PairEAMBase extends Pair {
  manybody = true;
  // the full list serves only tables on different grids (computeGeneral, see init); the half list serves both
  needsFull = true;
  needsHalf = true;
  virialFdotr = true;

  protected fileCache = new Map<string, EAMFile>();
  protected rhoTab: (Spline | null)[] = [];
  protected gTab: (Spline | null)[] = [];
  protected fTab: Embed[] = [];
  private rhoBuf = new Float64Array(0);
  private fpBuf = new Float64Array(0);

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError(`usage: pair_style ${this.name} (no arguments)`);
  }

  /** The full list only for tables on different grids (computeGeneral): set after the tables are built. */
  override init(ctx: StyleContext): void {
    super.init(ctx);
    this.needsFull = this.sharedGrid() === null;
  }

  override initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.table !== 12) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify table option`);
    }
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.rhoTab = new Array(nt * nt).fill(null);
    this.gTab = new Array(nt * nt).fill(null);
    this.fTab = new Array(nt).fill(ZERO_EMBED);
  }

  /**
   * The per-type tables after init, read-only, for code that evaluates EAM energies itself (fix sgcmc): with
   * nt = ntypes + 1, rho[i * nt + j] is the density at a type-i site from a type-j neighbour, z2r[i * nt + j] the
   * pair's r*phi(r), and frho[t] the embedding function F of type t.
   */
  tables(): { rho: readonly (EAMTable | null)[]; z2r: readonly (EAMTable | null)[]; frho: readonly EAMTable[] } {
    return { rho: this.rhoTab, z2r: this.gTab, frho: this.fTab.map((e) => e.table) };
  }

  /** Sets the per-type tables for both type orders; called from initOne (i <= j). */
  protected setPair(i: number, j: number, rhoIJ: Spline | null, rhoJI: Spline | null, g: Spline | null, fI: Embed, fJ: Embed): void {
    const nt = this.ntypes + 1;
    this.rhoTab[i * nt + j] = rhoIJ;
    this.rhoTab[j * nt + i] = rhoJI;
    this.gTab[i * nt + j] = g;
    this.gTab[j * nt + i] = g;
    this.fTab[i] = fI;
    this.fTab[j] = fJ;
  }

  /** The density and r*phi tables as the two loops read them (eamKernel.ts). */
  private kernelTables(): EAMKernelTables {
    return { ntypes: this.ntypes, cutsq: this.cutsq, rhoTab: this.rhoTab, gTab: this.gTab };
  }

  /**
   * The table grid every density and r*phi table shares (one file, or files on the same grid), or null. On a shared grid
   * the compute evaluates the tables as native LAMMPS does (eamKernel.ts), the densities summed over the half list (both
   * atoms of a pair, ghosts reverse-summed to their owners) as native does with newton on.
   */
  private sharedGrid(): { nr: number; rdr: number } | null {
    return eamGrid(this.kernelTables());
  }

  compute(pc: PairCompute): void {
    if (this.sharedGrid()) this.computeShared(pc);
    else this.computeGeneral(pc);
  }

  /** compute() on a shared table grid (sharedGrid), in native's order of operations: densities, embedding, forces. */
  private computeShared(pc: PairCompute): void {
    const nall = pc.nall;
    if (this.rhoBuf.length !== nall) {
      this.rhoBuf = new Float64Array(nall);
      this.fpBuf = new Float64Array(nall);
    }
    this.rhoBuf.fill(0);
    this.densities(pc, this.rhoBuf);
    this.embed(pc, this.rhoBuf, this.fpBuf);
    this.forces(pc, this.fpBuf);
  }

  /**
   * Whether the style can run its two passes on the shared-memory threads (cpu/threadedPairs.ts TwoPassPair): its
   * tables share one grid (sharedGrid).
   */
  twoPassReady(): boolean {
    return this.sharedGrid() !== null;
  }

  /** Native's first loop over the half-list rows [ilo, inum) (eamKernel.ts eamDensities). */
  densities(pc: PairCompute, rho: Float64Array): void {
    eamDensities(this.kernelTables(), this.name, pc, rho);
  }

  /**
   * Between the two loops (engine thread): the ghosts' densities summed to their owners (native's reverse
   * communication), F'(rho) and the embedding energy of each owned atom (compute_embedding), and F' copied to the
   * ghosts (forward communication).
   */
  embed(pc: PairCompute, rho: Float64Array, fp: Float64Array): void {
    const { type } = pc;
    const nlocal = pc.nlocal;
    pc.nb.reverseAddGhosts(rho, 1);
    const eatom = pc.eatom;
    let evdwl = 0;
    for (let i = 0; i < nlocal; i++) {
      // embedding_index: p = rho rdrho + 1, m clamped to [1, nrho - 1], p -= m clamped to 1 from above only
      const emb = this.fTab[type[i]];
      const sp = emb.sp;
      let p = rho[i] * sp.rdx + 1.0;
      let m = Math.trunc(p);
      if (m > sp.n - 1) m = sp.n - 1;
      if (m < 1) m = 1;
      p -= m;
      if (p > 1.0) p = 1.0;
      const c = sp.coef, o = 7 * (m - 1);
      fp[i] = (c[o] * p + c[o + 1]) * p + c[o + 2];
      let phi = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
      if (rho[i] > emb.rhoMax) phi += fp[i] * (rho[i] - emb.rhoMax);
      evdwl += phi;
      if (eatom) eatom[i] += phi;
    }
    pc.nb.forwardCopy(fp, 1);
    pc.acc.evdwl += evdwl;
  }

  /** Native's second loop over the half-list rows [ilo, inum) (eamKernel.ts eamForces). */
  forces(pc: PairCompute, fp: Float64Array): void {
    eamForces(this.kernelTables(), this.name, pc, fp);
  }

  /** compute() for tables on different grids (several funcfl files): each table evaluated on its own grid. */
  private computeGeneral(pc: PairCompute): void {
    const full = pc.full;
    const half = pc.half;
    if (!full || !half) throw new Error(`pair style ${this.name} needs full and half neighbor lists`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nall = pc.nall;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    if (this.rhoBuf.length !== nall) {
      this.rhoBuf = new Float64Array(nall);
      this.fpBuf = new Float64Array(nall);
    }
    const rho = this.rhoBuf;
    const fp = this.fpBuf;
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let s = 0;
      const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = full.neighbors[k] & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti + type[j]]) continue;
        s += (this.rhoTab[ti + type[j]] as Spline).eval(Math.sqrt(rsq));
      }
      rho[i] = s;
    }
    pc.nb.forwardCopy(rho, 1);
    const eatom = pc.eatom;
    let evdwl = 0;
    for (let i = 0; i < nlocal; i++) {
      const ft = this.fTab[type[i]];
      fp[i] = ft.deriv(rho[i]);
      const fe = ft.eval(rho[i]);
      evdwl += fe;
      if (eatom) eatom[i] += fe;
    }
    pc.nb.forwardCopy(fp, 1);
    const vatom = pc.vatom;
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const tii = type[i];
      const ti = tii * nt;
      const fpi = fp[i];
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const tj = type[j];
        if (rsq >= cutsq[ti + tj]) continue;
        const r = Math.sqrt(rsq);
        const gt = this.gTab[ti + tj] as Spline;
        const g = gt.eval(r);
        const gp = gt.deriv(r);
        // phi = g/r with g = r*phi(r): -phi'(r)/r = -g'/r^2 + g/r^3; the embedding
        // terms -F'_i rho'_{ij}(r)/r and -F'_j rho'_{ji}(r)/r follow from dE/dx_i
        const A = -gp / (r * r) + g / (r * r * r)
          - (fpi * (this.rhoTab[ti + tj] as Spline).deriv(r)) / r
          - (fp[j] * (this.rhoTab[tj * nt + tii] as Spline).deriv(r)) / r;
        const fx = A * dx, fy = A * dy, fz = A * dz;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = g / r;
        evdwl += e;
        if (eatom) { eatom[i] += 0.5 * e; eatom[j] += 0.5 * e; }
        if (vatom) {
          const h = 0.5 * A;
          const v0 = h * dx * dx, v1 = h * dy * dy, v2 = h * dz * dz, v3 = h * dx * dy, v4 = h * dx * dz, v5 = h * dy * dz;
          vatom[6 * i] += v0; vatom[6 * i + 1] += v1; vatom[6 * i + 2] += v2;
          vatom[6 * i + 3] += v3; vatom[6 * i + 4] += v4; vatom[6 * i + 5] += v5;
          vatom[6 * j] += v0; vatom[6 * j + 1] += v1; vatom[6 * j + 2] += v2;
          vatom[6 * j + 3] += v3; vatom[6 * j + 4] += v4; vatom[6 * j + 5] += v5;
        }
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  /** Pair term only (the embedding part depends on rho, not on r alone). */
  override single(_i: number, _j: number, itype: number, jtype: number, rsq: number) {
    const nt = this.ntypes + 1;
    if (rsq >= this.cutsq[itype * nt + jtype]) return { eng: 0, fforce: 0 };
    const gt = this.gTab[itype * nt + jtype];
    if (!gt) return { eng: 0, fforce: 0 };
    const r = Math.sqrt(rsq);
    const g = gt.eval(r);
    return { eng: g / r, fforce: -gt.deriv(r) / (r * r) + g / (r * r * r) };
  }
}

/**
 * The potential file defines the type masses (mass.rst: the eam styles
 * "define the masses of atom types in their respective potential files, in
 * which case the mass command is normally not used"). Measured against the
 * native-LAMMPS oracle fixtures: a mass set by an earlier mass command is
 * REPLACED by the file value (the w2eam_funcfl thermo temp at step 0 only
 * matches with the file masses 63.55/58.69, not the earlier mass * 60.0).
 * A dummy file mass of 0.0 keeps the previous value (pair_eam.rst: "unless
 * the potential file uses a dummy value (e.g. 0.0). LAMMPS will print a
 * warning, if this is the case.").
 */
const applyFileMass = (ctx: StyleContext, type: number, mass: number, name: string): void => {
  const s = ctx.s;
  if (!s || s.rmass) return;
  if (mass <= 0) {
    ctx.log(`WARNING: EAM potential file ${name} uses a dummy mass of 0.0 for type ${type}`);
    if (Number.isNaN(s.massByType[type])) s.massByType[type] = 1.0;
    return;
  }
  s.massByType[type] = mass;
};

/**
 * pair_style eam — DYNAMO funcfl files, one per I,I type pair; mixed pair
 * terms for I != J from the product of the effective charges (header).
 */
export class PairEAM extends PairEAMBase {
  readonly name = 'eam';
  private files: (EAMFile | null)[] = [];

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.files = new Array(ntypes + 1).fill(null);
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 3) throw pairCoeffArgs('usage: pair_coeff I J filename (style eam reads one funcfl file per I,I pair)');
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const file = this.readFuncfl(args[2], ctx);
    for (let i = ilo; i <= ihi; i++) {
      for (let j = jlo; j <= jhi; j++) {
        if (i === j) {
          this.files[i] = file;
          applyFileMass(ctx, i, file.masses[0], file.name);
        }
      }
    }
  }

  private readFuncfl(filename: string, ctx: StyleContext): EAMFile {
    const cached = this.fileCache.get(filename);
    if (cached) return cached;
    const file = parseFuncfl(ctx.readFile(filename, potentialFile('eam')), filename, ctx);
    this.fileCache.set(filename, file);
    return file;
  }

  override initOne(i: number, j: number): number {
    const fi = this.files[i];
    const fj = this.files[j];
    if (!fi || !fj) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    if (i === j) {
      this.setPair(i, j, fi.rho![0], fi.rho![0], this.chargeProduct(fi, fi), fi.F[0], fi.F[0]);
      return fi.cutoff;
    }
    if (fi.nr !== fj.nr || fi.dr !== fj.dr) {
      throw new StyleError(`funcfl files ${fi.name} and ${fj.name} do not share the same Nr and dr; cannot mix them`);
    }
    this.setPair(i, j, fj.rho![0], fi.rho![0], this.chargeProduct(fi, fj), fi.F[0], fj.F[0]);
    return 0.5 * (fi.cutoff + fj.cutoff);
  }

  /** g(r) = r*phi(r) from the effective charges, exact at the table points. */
  private chargeProduct(fi: EAMFile, fj: EAMFile): Spline {
    const zA = fi.z as Float64Array;
    const zB = fj.z as Float64Array;
    const n = fi.nr;
    const y = new Float64Array(n);
    for (let k = 0; k < n; k++) y[k] = HARTREE_EV * BOHR_ANG * zA[k] * zB[k];
    return new Spline(y, fi.dr);
  }
}

/**
 * pair_style eam/alloy — one DYNAMO setfl file, elements mapped to LAMMPS
 * types ("The first 2 arguments must be \* \* so as to span all LAMMPS
 * atom types.").
 */
export class PairEAMAlloy extends PairEAMBase {
  readonly name = 'eam/alloy';
  private setfl: EAMFile | null = null;
  private elemOf: Int32Array = new Int32Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.elemOf = new Int32Array(ntypes + 1).fill(-1);
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args[0] !== '*' || args[1] !== '*') {
      throw new StyleError('the first 2 arguments of pair_coeff for style eam/alloy must be * *');
    }
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    this.setfl = this.readSetfl(args[2], ctx, false);
    const file = this.setfl;
    for (let t = 1; t <= this.ntypes; t++) {
      if (elems[t - 1] === 'NULL') {
        this.elemOf[t] = -1;
        continue;
      }
      const idx = file.names.indexOf(elems[t - 1]);
      if (idx < 0) {
        throw new StyleError(`element '${elems[t - 1]}' is not in EAM potential file ${file.name} (${file.names.join(' ')})`);
      }
      this.elemOf[t] = idx;
      applyFileMass(ctx, t, file.masses[idx], file.name);
    }
  }

  private readSetfl(filename: string, ctx: StyleContext, fs: boolean): EAMFile {
    const cached = this.fileCache.get(filename);
    if (cached) return cached;
    const file = parseSetfl(ctx.readFile(filename, potentialFile(fs ? 'eam/fs' : 'eam/alloy')), filename, ctx, fs);
    this.fileCache.set(filename, file);
    return file;
  }

  override initOne(i: number, j: number): number {
    const ei = this.elemOf[i], ej = this.elemOf[j];
    if (ei < 0 || ej < 0) return 0;
    const f = this.setfl as EAMFile;
    this.setPair(i, j, f.rho![ej], f.rho![ei], f.phi![Math.max(ei, ej)][Math.min(ei, ej)], f.F[ei], f.F[ej]);
    return f.cutoff;
  }
}

/**
 * pair_style eam/fs — Finnis-Sinclair file: asymmetric density functions
 * rho_{alpha beta}(r) for every element pair, phi arrays as in setfl.
 */
export class PairEAMFS extends PairEAMBase {
  readonly name = 'eam/fs';
  private fs: EAMFile | null = null;
  private elemOf: Int32Array = new Int32Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.elemOf = new Int32Array(ntypes + 1).fill(-1);
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args[0] !== '*' || args[1] !== '*') {
      throw new StyleError('the first 2 arguments of pair_coeff for style eam/fs must be * *');
    }
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    this.fs = this.readSetfl(args[2], ctx);
    const file = this.fs;
    for (let t = 1; t <= this.ntypes; t++) {
      if (elems[t - 1] === 'NULL') {
        this.elemOf[t] = -1;
        continue;
      }
      const idx = file.names.indexOf(elems[t - 1]);
      if (idx < 0) {
        throw new StyleError(`element '${elems[t - 1]}' is not in EAM potential file ${file.name} (${file.names.join(' ')})`);
      }
      this.elemOf[t] = idx;
      applyFileMass(ctx, t, file.masses[idx], file.name);
    }
  }

  private readSetfl(filename: string, ctx: StyleContext): EAMFile {
    const cached = this.fileCache.get(filename);
    if (cached) return cached;
    const file = parseSetfl(ctx.readFile(filename, potentialFile('eam/fs')), filename, ctx, true);
    this.fileCache.set(filename, file);
    return file;
  }

  override initOne(i: number, j: number): number {
    const ei = this.elemOf[i], ej = this.elemOf[j];
    if (ei < 0 || ej < 0) return 0;
    const f = this.fs as EAMFile;
    this.setPair(i, j, f.fsRho![ej][ei], f.fsRho![ei][ej], f.phi![Math.max(ei, ej)][Math.min(ei, ej)], f.F[ei], f.F[ej]);
    return f.cutoff;
  }
}
