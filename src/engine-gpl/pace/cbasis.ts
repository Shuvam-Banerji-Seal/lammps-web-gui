/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28) and the ML-PACE library (ACE evaluator), GPL-2.0 use: ML-PACE/ace-evaluator/ace_c_basis.cpp (the .ace loader
 * ACECTildeBasisSet::load, fread_c_tilde_b_basis_func, _load_radial_ACERadial) and ace_abstract_basis.cpp
 * (ACEAbstractBasisSet::inner_cutoff, FS_values_and_derivatives, FexpShiftedScaled) of lammps-user-pace v.2025.12.4.p1
 * (checked against LAMMPS's CMake SHA256). The LAMMPS package that reads it is src/ML-PACE/pair_pace.cpp at commit
 * b332f28. Library authors (from its headers): Yury Lysogorskiy, Cas van der Oord, Anton Bochkarev, Sarath Menon, Matteo
 * Rinaldi, Thomas Hammerschmidt, Matous Mrovec, Aidan Thompson, Gabor Csanyi, Christoph Ortner, Ralf Drautz (ICAMS,
 * Ruhr-University Bochum and co-authors). Used here under GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * The C-tilde basis of a .ace file: the embedding (cluster functional) parameters, the radial functions, and the rank-1 and
 * higher-rank basis functions with their m-combinations and coefficients. The file is read token by token with the same
 * scanf-like steps as the C++ loader (literal matches, then one value each).
 */
import { StyleError } from '../../engine/force/types';
import { ACERadialFunctions, cutoffFuncPoly } from './radial';
import { CartSphericalHarmonics } from './spherical';
import { pow as libmPow } from '../libm/pow';
import { exp as libmExp } from '../libm/exp';

/** One C-tilde basis function (ACECTildeBasisFunction): rank, species, (n, l) per particle and the m-combinations. */
export interface CTildeFunc {
  rank: number;
  ndensity: number;
  mu0: number;
  mus: Int32Array;
  ns: Int32Array;
  ls: Int32Array;
  numMsCombs: number;
  /** ms_combs[m_ind * rank + r] */
  msCombs: Int32Array;
  /** ctildes[m_ind * ndensity + p] */
  ctildes: Float64Array;
}

/** ACEEmbeddingSpecification: the cluster functional and the core-energy cutoff of one species. */
export interface EmbeddingSpec {
  npoti: string;
  FS: Float64Array;
  ndensity: number;
  rhoCoreCutoff: number;
  drhoCoreCutoff: number;
}

/** scanf-style cursor over the text of a .ace file: whitespace skipping, literal matches and single tokens. */
class Cursor {
  pos = 0;
  constructor(private readonly text: string, private readonly filename: string) {}

  ws(): void {
    const t = this.text;
    while (this.pos < t.length && /\s/.test(t[this.pos])) this.pos++;
  }

  /** Skips whitespace, then matches the literal (returns false, consuming nothing more, when it does not match). */
  tryLit(lit: string): boolean {
    this.ws();
    if (this.text.startsWith(lit, this.pos)) {
      this.pos += lit.length;
      return true;
    }
    return false;
  }

  lit(lit: string, what: string): void {
    if (!this.tryLit(lit)) throw new StyleError(`Error reading ${what} from ACE potential file ${this.filename}`);
  }

  /** %s: the next whitespace-delimited token. */
  tok(what: string): string {
    this.ws();
    const start = this.pos;
    while (this.pos < this.text.length && !/\s/.test(this.text[this.pos])) this.pos++;
    if (this.pos === start) throw new StyleError(`Error reading ${what} from ACE potential file ${this.filename}`);
    return this.text.slice(start, this.pos);
  }

  int(what: string): number {
    const s = this.tok(what);
    const v = parseInt(s, 10);
    if (!Number.isFinite(v)) throw new StyleError(`Error reading ${what} (${s}) from ACE potential file ${this.filename}`);
    return v;
  }

  num(what: string): number {
    const s = this.tok(what);
    const v = Number(s);
    if (!Number.isFinite(v)) throw new StyleError(`Error reading ${what} (${s}) from ACE potential file ${this.filename}`);
    return v;
  }
}

/** The C-tilde basis set of one .ace file (ACECTildeBasisSet, single species use as in the loader). */
export class CTildeBasis {
  nelements = 0;
  elementsName: string[] = [];
  lmax = 0;
  nradbase = 0;
  nradmax = 0;
  cutoffmax = 0;
  deltaSplineBins = 0;
  rankmax = 0;
  ndensitymax = 0;
  numCtildeMax = 0;
  numMsCombinationsMax = 0;
  E0vals = new Float64Array(0);
  embedding: EmbeddingSpec[] = [];
  radial!: ACERadialFunctions;
  sph!: CartSphericalHarmonics;
  totalBasisSizeRank1: number[] = [];
  basisRank1: CTildeFunc[][] = [];
  totalBasisSize: number[] = [];
  basis: CTildeFunc[][] = [];

  /** get_species_index_by_name: the species of an element symbol, or -1. */
  speciesIndexByName(name: string): number {
    for (let t = 0; t < this.nelements; t++) if (this.elementsName[t] === name) return t;
    return -1;
  }

  /** ACECTildeBasisSet::load for a .ace text (the file name only labels messages). */
  static fromText(text: string, filename: string): CTildeBasis {
    const b = new CTildeBasis();
    const c = new Cursor(text, filename);
    // number of elements: "nelements=N", searched line by line as the loader does (comment lines may come first)
    const at = text.search(/^[ \t]*nelements=/m);
    if (at < 0) throw new StyleError(`Error reading nelements from ACE potential file ${filename}`);
    c.pos = at;
    c.lit('nelements=', 'nelements');
    b.nelements = parseInt(c.tok('nelements'), 10);
    if (b.nelements !== 1) {
      throw new StyleError(`pair_style pace: only single-element ACE potentials are supported (file ${filename} has ${b.nelements} elements)`);
    }
    c.lit('elements:', 'elements');
    b.elementsName = [];
    for (let mu = 0; mu < b.nelements; mu++) b.elementsName.push(c.tok('elements'));
    // angular basis
    c.lit('lmax=', 'lmax');
    b.lmax = parseInt(c.tok('lmax'), 10);
    b.sph = new CartSphericalHarmonics(b.lmax);
    // embedding function (optional line) and FS parameters
    let npoti = 'FinnisSinclair';
    let specified = false;
    if (c.tryLit('embedding-function:')) {
      npoti = c.tok('embedding-function');
      specified = true;
    }
    const nfs = c.int('FS parameters size');
    c.lit('FS', 'FS parameters');
    c.lit('parameters:', 'FS parameters');
    const FS = new Float64Array(nfs);
    for (let i = 0; i < nfs; i++) FS[i] = c.num('FS parameter');
    if (!specified) {
      for (let j = 1; j < nfs; j += 2) {
        if (FS[j] !== 1.0) {
          throw new StyleError(`pair_style pace: ${filename} is non-linear; specify embedding-function: FinnisSinclair or FinnisSinclairShiftedScaled`);
        }
      }
    }
    // core energy cutoffs
    c.lit('core', 'core energy-cutoff parameters');
    c.lit('energy-cutoff', 'core energy-cutoff parameters');
    c.lit('parameters:', 'core energy-cutoff parameters');
    b.embedding = [];
    for (let mu = 0; mu < b.nelements; mu++) {
      const rho = c.num('rho core cutoff');
      const drho = c.num('drho core cutoff');
      b.embedding.push({ npoti: 'FinnisSinclair', FS: new Float64Array(0), ndensity: 0, rhoCoreCutoff: rho, drhoCoreCutoff: drho });
    }
    b.embedding[0].npoti = npoti;
    b.embedding[0].FS = FS;
    // atom energy shifts E0
    b.E0vals = new Float64Array(b.nelements);
    if (c.tryLit('E0:')) {
      b.E0vals[0] = parseFloat(c.tok('E0'));
      for (let mu = 1; mu < b.nelements; mu++) b.E0vals[mu] = c.num('E0');
    }
    // radial basis
    c.lit('radbasename=', 'radbasename');
    const radbasename = c.tok('radbasename');
    if (radbasename === 'ACE.jl.Basic') {
      throw new StyleError(`pair_style pace: radbasename=ACE.jl.Basic (SHIPs radial basis) is not supported yet in ${filename}`);
    }
    if (radbasename !== 'ChebExpCos' && radbasename !== 'ChebPow' && radbasename !== 'ChebLinear') {
      throw new StyleError(`pair_style pace: unknown radbasename ${radbasename} in ${filename}`);
    }
    b.loadRadial(c, radbasename, filename);
    c.lit('rankmax=', 'rankmax');
    b.rankmax = parseInt(c.tok('rankmax'), 10);
    c.lit('ndensitymax=', 'ndensitymax');
    b.ndensitymax = parseInt(c.tok('ndensitymax'), 10);
    b.embedding[0].ndensity = b.ndensitymax;
    c.lit('num_c_tilde_max=', 'num_c_tilde_max');
    b.numCtildeMax = parseInt(c.tok('num_c_tilde_max'), 10);
    c.lit('num_ms_combinations_max=', 'num_ms_combinations_max');
    b.numMsCombinationsMax = parseInt(c.tok('num_ms_combinations_max'), 10);
    // rank-1 functions
    c.lit('total_basis_size_rank1:', 'total_basis_size_rank1');
    b.totalBasisSizeRank1 = [];
    for (let mu = 0; mu < b.nelements; mu++) b.totalBasisSizeRank1.push(parseInt(c.tok('total_basis_size_rank1'), 10));
    b.basisRank1 = [];
    for (let mu = 0; mu < b.nelements; mu++) {
      const arr: CTildeFunc[] = [];
      for (let k = 0; k < b.totalBasisSizeRank1[mu]; k++) arr.push(readFunc(c, b.ndensitymax, filename));
      b.basisRank1.push(arr);
    }
    // higher-rank functions
    c.lit('total_basis_size:', 'total_basis_size');
    b.totalBasisSize = [];
    for (let mu = 0; mu < b.nelements; mu++) b.totalBasisSize.push(parseInt(c.tok('total_basis_size'), 10));
    b.basis = [];
    for (let mu = 0; mu < b.nelements; mu++) {
      const arr: CTildeFunc[] = [];
      for (let k = 0; k < b.totalBasisSize[mu]; k++) arr.push(readFunc(c, b.ndensitymax, filename));
      b.basis.push(arr);
    }
    b.radial.elements = b.elementsName.slice();
    b.radial.setuplookupRadspline();
    return b;
  }

  /** _load_radial_ACERadial: the radial parameters, core repulsion, lambda, cutoffs and the crad coefficients. */
  private loadRadial(c: Cursor, radbasename: string, filename: string): void {
    const nel = this.nelements;
    c.lit('nradbase=', 'nradbase');
    this.nradbase = parseInt(c.tok('nradbase'), 10);
    c.lit('nradmax=', 'nradmax');
    this.nradmax = parseInt(c.tok('nradmax'), 10);
    c.lit('cutoffmax=', 'cutoffmax');
    this.cutoffmax = parseFloat(c.tok('cutoffmax'));
    c.lit('deltaSplineBins=', 'deltaSplineBins');
    this.deltaSplineBins = parseFloat(c.tok('deltaSplineBins'));
    this.radial = new ACERadialFunctions(this.nradbase, this.lmax, this.nradmax, this.deltaSplineBins, nel, radbasename);
    const rf = this.radial;
    c.lit('core', 'core repulsion parameters');
    c.lit('repulsion', 'core repulsion parameters');
    c.lit('parameters:', 'core repulsion parameters');
    for (let mi = 0; mi < nel; mi++) {
      for (let mj = 0; mj < nel; mj++) {
        rf.prehc[mi * nel + mj] = c.num('core repulsion parameters');
        rf.lambdahc[mi * nel + mj] = c.num('core repulsion parameters');
      }
    }
    c.lit('radparameter=', 'radparameter');
    for (let mi = 0; mi < nel; mi++) for (let mj = 0; mj < nel; mj++) rf.lambda[mi * nel + mj] = c.num('radparameter');
    c.lit('cutoff=', 'cutoff');
    for (let mi = 0; mi < nel; mi++) for (let mj = 0; mj < nel; mj++) rf.cut[mi * nel + mj] = c.num('cutoff');
    c.lit('dcut=', 'dcut');
    for (let mi = 0; mi < nel; mi++) for (let mj = 0; mj < nel; mj++) rf.dcut[mi * nel + mj] = c.num('dcut');
    c.lit('crad=', 'crad');
    for (let mi = 0; mi < nel; mi++) {
      for (let mj = 0; mj < nel; mj++) {
        const pair = mi * nel + mj;
        for (let k = 0; k < this.nradbase; k++) {
          for (let n = 0; n < this.nradmax; n++) {
            for (let l = 0; l <= this.lmax; l++) {
              rf.crad[rf.cradIndex(pair, n, l, k)] = c.num('crad');
            }
          }
        }
      }
    }
    void filename;
  }

}

/** fread_c_tilde_b_basis_func: one C-tilde basis function, with its m-combinations and coefficients. */
const readFunc = (c: Cursor, ndensity: number, filename: string): CTildeFunc => {
  c.lit('ctilde_basis_func:', 'ctilde_basis_func');
  c.lit('rank=', 'rank');
  const rank = parseInt(c.tok('rank'), 10);
  c.lit('ndens=', 'ndens');
  const nd = parseInt(c.tok('ndens'), 10);
  c.lit('mu0=', 'mu0');
  const mu0 = parseInt(c.tok('mu0'), 10);
  const mus = new Int32Array(rank), ns = new Int32Array(rank), ls = new Int32Array(rank);
  c.lit('mu=(', 'mu');
  for (let r = 0; r < rank; r++) mus[r] = parseInt(c.tok('mu'), 10);
  c.lit(')', 'mu');
  c.lit('n=(', 'n');
  for (let r = 0; r < rank; r++) ns[r] = parseInt(c.tok('n'), 10);
  c.lit(')', 'n');
  c.lit('l=(', 'l');
  for (let r = 0; r < rank; r++) ls[r] = parseInt(c.tok('l'), 10);
  c.lit(')', 'l');
  c.lit('num_ms=', 'num_ms');
  const numMs = parseInt(c.tok('num_ms'), 10);
  const msCombs = new Int32Array(rank * numMs);
  const ctildes = new Float64Array(nd * numMs);
  for (let m = 0; m < numMs; m++) {
    c.lit('<', 'ms combination');
    for (let r = 0; r < rank; r++) msCombs[m * rank + r] = parseInt(c.tok('ms'), 10);
    c.lit('>:', 'ms combination');
    for (let p = 0; p < nd; p++) ctildes[m * nd + p] = Number(c.tok('ctilde'));
  }
  void ndensity; void filename;
  return { rank, ndensity: nd, mu0, mus, ns, ls, numMsCombs: numMs, msCombs, ctildes };
};

/** Exp-based FS embedding of the FinnisSinclairShiftedScaled type (ace_abstract_basis.cpp FexpShiftedScaled). */
export const fexpShiftedScaled = (rho: number, mexp: number): [number, number] => {
  const eps = 1e-10;
  if (Math.abs(mexp - 1.0) < eps) return [rho, 1];
  const a = Math.abs(rho);
  const exprho = libmExp(-a);
  const nx = 1. / mexp;
  const xoff = libmPow(nx, (nx / (1.0 - nx))) * exprho;
  const yoff = libmPow(nx, (1 / (1.0 - nx))) * exprho;
  const sign = (rho < 0 || Object.is(rho, -0)) ? -1 : 1;
  const F = sign * (libmPow(xoff + a, mexp) - yoff);
  const DF = yoff + mexp * (-xoff + 1.0) * libmPow(xoff + a, mexp - 1.);
  return [F, DF];
};

/** inner_cutoff of ACEAbstractBasisSet (ace_abstract_basis.cpp). */
export const innerCutoff = (rhoCore: number, rhoCut: number, drhoCut: number): [number, number] => {
  const rhoLow = rhoCut - drhoCut;
  if (rhoCore >= rhoCut) return [0, 0];
  if (rhoCore <= rhoLow) return [1, 0];
  return cutoffFuncPoly(rhoCore, rhoCut, drhoCut);
};
