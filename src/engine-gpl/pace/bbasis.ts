/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the ML-PACE library v.2025.12.4.p1 it builds
 * for src/ML-PACE, GPL-2.0 use: ace/ace_yaml_input.cpp (Input::parse_input, read_core_repulsion, read_core_rho_drho_cut,
 * read_radcoefficients, copy/check of the radial basis and core repulsion of dual blocks), ace/ace_b_basis.cpp
 * (ACEBBasisSet::initialize_basis, order_and_compress_b_basis_function, compress_basis_functions, flatten_basis,
 * to_ACECTildeBasisSet with group_basis_functions_by_index and summation_over_LS,
 * BBasisFunctionsSpecificationBlock::update_params, validate_individual_functions, validate_radcoefficients),
 * ace/ace_b_basisfunction.cpp (BBasisFunctionSpecification::validate, to_string), ace/ace_couplings.cpp (validate_ls_LS,
 * expand_ls_LS, generate_basis_function_n_body) and ace-evaluator/ace_abstract_basis.cpp (compare). Library authors (from
 * its headers): Yury Lysogorskiy, Cas van der Oord, Anton Bochkarev, Sarath Menon, Matteo Rinaldi, Thomas Hammerschmidt,
 * Matous Mrovec, Aidan Thompson, Gabor Csanyi, Christoph Ortner, Ralf Drautz. This file is GPL-2.0
 * (src/engine-gpl/LICENSE).
 *
 * The B-basis potential file (.yaml, the pacemaker format): species blocks with the embedding, the radial basis and the
 * "nbody" functions. ACEBBasisSet builds every function's m-combinations with the generalized Clebsch-Gordan
 * coefficients (./couplings.ts), and to_ACECTildeBasisSet sums the functions that share (mu0, mus, ns, ls) into the
 * C-tilde basis of pair_style pace.
 */
import { StyleError } from '../../engine/force/types';
import { std_sort } from '../ptm/libstdcxx_sort';
import { CTildeBasis, type CTildeFunc, type EmbeddingSpec } from './cbasis';
import { generateMsCgList } from './couplings';
import { ACERadialFunctions } from './radial';
import { CartSphericalHarmonics } from './spherical';
import { asMap, asNum, asInt, asSeq, asStr, need, nums, ints, parseYaml, YamlMap, type YamlValue } from './yace';

/** BBasisFunctionSpecification: one "nbody" entry of a species block. */
interface FuncSpec {
  rank: number;
  elements: string[];
  ns: number[];
  ls: number[];
  LS: number[];
  coeffs: number[];
  skip: boolean;
}

/** BBasisFunctionsSpecificationBlock: one species block. */
interface SpecBlock {
  blockName: string;
  rankmax: number;
  numberOfSpecies: number;
  elementsVec: string[];
  mu0: string;
  isRadialBasisDefined: boolean;
  lmaxi: number;
  nradmaxi: number;
  ndensityi: number;
  npoti: string;
  fsParameters: number[];
  coreRepParameters: number[];
  isCoreRepulsionDefined: boolean;
  rhoCut: number;
  drhoCut: number;
  rcutij: number;
  dcutij: number;
  rIn: number;
  deltaIn: number;
  innerCutoffType: string;
  nradbaseij: number;
  radbase: string;
  radparameters: number[];
  radcoefficients: number[][][];
  funcspecs: FuncSpec[];
}

/** One B-basis function (ACEBBasisFunction). */
export interface BFunc {
  rank: number;
  rankL: number;
  ndensity: number;
  mu0: number;
  mus: Int32Array;
  ns: Int32Array;
  ls: Int32Array;
  LS: Int32Array;
  numMsCombs: number;
  /** ms_combs[m_ind * rank + r] */
  msCombs: Int32Array;
  /** gen_cgs[m_ind] */
  genCgs: Float64Array;
  /** coeff[p] */
  coeff: Float64Array;
}

/** max_element of a short (rank-sized) array. */
const maxOf = (xs: number[]): number => {
  let m = xs[0];
  for (const x of xs) if (x > m) m = x;
  return m;
};
const splitKey = (s: string): string[] => s.split(/\s+/).filter((w) => w.length > 0);
const join = (xs: ArrayLike<number | string>): string => Array.from(xs).join(',');
const sameNums = (a: number[], b: number[]): boolean => a.length === b.length && a.every((x, i) => x === b[i]);
const sameRad = (a: number[][][], b: number[][][]): boolean =>
  a.length === b.length && a.every((x, i) => x.length === b[i].length && x.every((y, j) => sameNums(y, b[i][j])));

/** BBasisFunctionSpecification::to_string. */
const specToString = (s: FuncSpec): string => {
  let out = `BBasisFunctionSpecification(elements=[${join(s.elements)}],  ns=[${join(s.ns)}],  ls=[${join(s.ls)}],  `;
  if (s.LS.length > 0) out += `LS=[${join(s.LS)}],  `;
  out += `coeffs=[${join(s.coeffs)}]`;
  if (s.skip) out += ' skipped';
  return `${out})`;
};

/** validate_ls_LS: the coupling rules of ls and LS (the intermediate L) by rank. */
const validateLsLS = (ls: number[], LS: number[]): void => {
  const rank = ls.length, rankL = LS.length;
  if (rank <= 2) {
    if (rankL !== 0) throw new StyleError(`len of LS should be 0, but ${rankL} is found`);
  } else if (rankL !== rank - 2) {
    throw new StyleError(`len of LS should be ${rank - 2}, but ${rankL} is found`);
  }
  const list = (xs: number[]) => xs.map((x) => ` ${x} `).join('');
  if (rank === 1) {
    if (ls[0] !== 0) throw new StyleError(`ls(${list(ls)}) should be (0)`);
  } else if (rank === 2) {
    if (ls[0] !== ls[1]) throw new StyleError(`All elements of ls (${list(ls)}) should be equal`);
  } else if (rank === 3 || rank === 5) {
    if (LS[LS.length - 1] !== ls[ls.length - 1]) {
      throw new StyleError(`Last element of LS (${list(LS)}) != last element of ls (${list(ls)})`);
    }
  } else if (rank >= 4) {
    if (LS[LS.length - 1] !== LS[LS.length - 2]) throw new StyleError(`Last element of LS (${list(LS)}) != to its next-to-last element`);
  }
  let sum = 0;
  for (const l of ls) sum += l;
  if (sum % 2 !== 0) throw new StyleError(`Sum of ls is not even: ls = (${ls.map((l) => `${l} `).join('')})`);
};

/** expand_ls_LS: complete ls and LS from the symmetry relations of each rank. */
const expandLsLS = (rank: number, ls: number[], LS: number[]): void => {
  if (rank === 1) {
    if (ls.length === 0) ls.push(0);
  } else if (rank === 2) {
    if (ls.length === 1) ls.push(ls[0]);
  } else if (rank === 3) {
    if (LS.length === 0 && ls.length === rank) LS.push(ls[2]);
  } else if (rank === 4) {
    if (LS.length === 1) LS.push(LS[0]);
  } else if (rank === 5) {
    if (LS.length === rank - 3 && ls.length === rank) LS.push(ls[4]);
  } else if (rank === 6) {
    if (LS.length === rank - 3 && ls.length === rank) LS.push(LS[LS.length - 1]);
  }
};

/** BBasisFunctionSpecification::validate. */
const validateSpec = (s: FuncSpec): void => {
  s.rank = s.elements.length - 1;
  expandLsLS(s.rank, s.ls, s.LS);
  if (s.ns.length !== s.ls.length) throw new StyleError("'ls' should have the same length as 'ns'");
  let nsMin = s.ns[0];
  for (const n of s.ns) if (n < nsMin) nsMin = n;
  if (nsMin < 1) throw new StyleError(`${specToString(s)}:\nminimum value of 'ns'(${nsMin}) should be not less than 1`);
  let lsMin = s.ls[0];
  for (const l of s.ls) if (l < lsMin) lsMin = l;
  if (lsMin < 0) throw new StyleError(`${specToString(s)}:\nminimum value of 'ls'(${lsMin}) should be not less than 0`);
  if (s.rank !== s.ns.length) {
    throw new StyleError(`${specToString(s)}:\nsize of 'ns'(${s.ns.length}) is inconsistent with the rank(${s.rank}) and size of 'elements' - 1 (${s.elements.length - 1})`);
  }
  if (s.ls.length !== s.rank) {
    throw new StyleError(`${specToString(s)}:\nsize of 'ls'(${s.ls.length}) is inconsistent with the rank(${s.rank}) and size of 'elements' - 1 (${s.elements.length - 1})`);
  }
  validateLsLS(s.ls, s.LS);
};

/** validate_radcoefficients: crad must have the shape [nradmaxi][lmaxi+1][nradbaseij] at least. */
const validateRadcoefficients = (b: SpecBlock): void => {
  if (b.radcoefficients.length < b.nradmaxi) {
    throw new StyleError(`Error: species block ${b.blockName} has insufficient number of radcoefficients (shape=(${b.radcoefficients.length},...)) whereas nradmaxi = ${b.nradmaxi}`);
  }
  for (let n = 0; n < b.nradmaxi; n++) {
    if (b.radcoefficients[n].length < b.lmaxi + 1) {
      throw new StyleError(`Error: species block ${b.blockName} has insufficient number in radcoefficients[${n + 1}] (shape=(${b.radcoefficients[n].length},...)) whereas lmaxi+1 = ${b.lmaxi + 1}`);
    }
    for (let l = 0; l <= b.lmaxi; l++) {
      if (b.radcoefficients[n][l].length < b.nradbaseij) {
        throw new StyleError(`Error: species block ${b.blockName} has insufficient number radcoefficients[${n + 1}][${l}].size=${b.radcoefficients[n][l].length} whereas it should be nradbase = ${b.nradbaseij}`);
      }
    }
  }
};

/** yaml-cpp's bool conversion. */
const asBool = (v: YamlValue, what: string): boolean => {
  const s = asStr(v, what);
  if (/^(y|Y|yes|Yes|YES|true|True|TRUE|on|On|ON)$/.test(s)) return true;
  if (/^(n|N|no|No|NO|false|False|FALSE|off|Off|OFF)$/.test(s)) return false;
  throw new StyleError(`${what}: expected a boolean, found ${s}`);
};

/** read_radcoefficients: the 3-d [n][l][k] array, or the deprecated flat one in k, n, l order. */
const readRadcoefficients = (y: YamlMap, b: SpecBlock): void => {
  const v = need(y, 'radcoefficients', b.blockName);
  const seq = asSeq(v, 'radcoefficients');
  if (seq.every((x) => Array.isArray(x))) {
    b.radcoefficients = seq.map((x) => asSeq(x, 'radcoefficients').map((z) => nums(z, 'radcoefficients')));
    return;
  }
  const flat = nums(v, 'radcoefficients');
  b.radcoefficients = Array.from({ length: b.nradmaxi }, () => Array.from({ length: b.lmaxi + 1 }, () => new Array<number>(b.nradbaseij).fill(0)));
  let j = 0;
  for (let k = 0; k < b.nradbaseij; k++) {
    for (let n = 0; n < b.nradmaxi; n++) {
      for (let l = 0; l <= b.lmaxi; l++, j++) {
        if (j >= flat.length) throw new StyleError(`${b.blockName}: radcoefficients are too short`);
        b.radcoefficients[n][l][k] = flat[j];
      }
    }
  }
};

/** Input::parse_input: the global spline step and the species blocks of a B-basis .yaml text. */
const parseInput = (text: string, filename: string): { deltaSplineBins: number; blocks: SpecBlock[] } => {
  const root = asMap(parseYaml(text, filename), filename);
  const deltaSplineBins = asNum(need(need(root, 'global', filename), 'DeltaSplineBins', 'global'), 'DeltaSplineBins');
  const species = asSeq(need(root, 'species', filename), 'species');
  const ndensityMap = new Map<string, number>();
  const blocks: SpecBlock[] = [];
  for (const sv of species) {
    const y = asMap(sv, 'species');
    const blockName = asStr(need(y, 'speciesblock', 'species'), 'speciesblock');
    const elementsVec = splitKey(blockName);
    const mu0 = elementsVec[0];
    const elementsSet = new Set(elementsVec);
    const b: SpecBlock = {
      blockName, rankmax: 0, numberOfSpecies: elementsSet.size, elementsVec, mu0, isRadialBasisDefined: false,
      lmaxi: 0, nradmaxi: 0, ndensityi: 0, npoti: '', fsParameters: [], coreRepParameters: [0, 0],
      isCoreRepulsionDefined: false, rhoCut: 100000.0, drhoCut: 250.0, rcutij: 0, dcutij: 0, rIn: 0, deltaIn: 0,
      innerCutoffType: 'density', nradbaseij: 0, radbase: '', radparameters: [], radcoefficients: [], funcspecs: [],
    };
    if (b.numberOfSpecies === 1 && elementsVec.length === 1) elementsVec.push(mu0);
    if (b.numberOfSpecies === 1) {
      b.ndensityi = asInt(need(y, 'ndensityi', blockName), 'ndensityi');
      if (ndensityMap.has(mu0)) throw new StyleError(`ndensity for ${mu0} = ${b.ndensityi} is already defined\n`);
      ndensityMap.set(mu0, b.ndensityi);
      b.npoti = asStr(need(y, 'npoti', blockName), 'npoti');
      b.fsParameters = nums(need(y, 'parameters', blockName), 'parameters');
      // read_core_rho_drho_cut
      if (y.has('rho_core_cut')) b.rhoCut = asNum(y.get('rho_core_cut')!, 'rho_core_cut');
      if (y.has('drho_core_cut')) b.drhoCut = asNum(y.get('drho_core_cut')!, 'drho_core_cut');
    } else {
      b.ndensityi = ndensityMap.get(mu0) ?? 0;
    }
    if (y.has('lmaxi')) {
      b.lmaxi = asInt(y.get('lmaxi')!, 'lmaxi');
      b.nradmaxi = asInt(need(y, 'nradmaxi', blockName), 'nradmaxi');
      b.nradbaseij = asInt(need(y, 'nradbaseij', blockName), 'nradbaseij');
      b.radbase = asStr(need(y, 'radbase', blockName), 'radbase');
      b.radparameters = nums(need(y, 'radparameters', blockName), 'radparameters');
      readRadcoefficients(y, b);
      validateRadcoefficients(b);
      b.rcutij = asNum(need(y, 'rcutij', blockName), 'rcutij');
      b.dcutij = asNum(need(y, 'dcutij', blockName), 'dcutij');
      b.isRadialBasisDefined = true;
    } else if (y.has('nradmaxi') || y.has('nradbaseij') || y.has('radbase') || y.has('radparameters') || y.has('radcoefficients')) {
      throw new StyleError(`Radial basis functions are partially defined for  ${blockName}`);
    }
    // read_core_repulsion
    if (y.has('core-repulsion')) {
      b.coreRepParameters = nums(y.get('core-repulsion')!, 'core-repulsion');
      b.isCoreRepulsionDefined = true;
    }
    if (y.has('inner_cutoff_type')) b.innerCutoffType = asStr(y.get('inner_cutoff_type')!, 'inner_cutoff_type');
    if (y.has('r_in')) b.rIn = asNum(y.get('r_in')!, 'r_in');
    if (y.has('delta_in')) b.deltaIn = asNum(y.get('delta_in')!, 'delta_in');

    const nbody = y.has('nbody') && Array.isArray(y.get('nbody')) ? (y.get('nbody') as YamlValue[]) : [];
    for (const fv of nbody) {
      const f = asMap(fv, 'nbody');
      const typeStr = asStr(need(f, 'type', 'nbody'), 'type');
      const elements = splitKey(typeStr);
      if (elements[0] !== mu0) {
        throw new StyleError(`Function specification with type: '${typeStr}' is inconsistent with block specification '${blockName}'`);
      }
      for (let ii = 1; ii < elements.length; ii++) {
        if (!elementsSet.has(elements[ii])) {
          throw new StyleError(`Function specification with type: '${typeStr}' is inconsistent with block specification '${blockName}': element ${elements[ii]} is not in block's expected elements`);
        }
      }
      const spec: FuncSpec = { rank: elements.length - 1, elements, ns: [], ls: [], LS: [], coeffs: [], skip: false };
      if (spec.rank > 0) {
        spec.ns = ints(need(f, 'nr', 'nbody'), 'nr');
        spec.ls = ints(need(f, 'nl', 'nbody'), 'nl');
        const c = need(f, 'c', 'nbody');
        spec.coeffs = Array.isArray(c) ? nums(c, 'c') : [asNum(c, 'c')];
        const expect = ndensityMap.get(elements[0]) ?? 0;
        if (spec.coeffs.length !== expect) {
          throw new StyleError(`'n-body' function has ${spec.coeffs.length} coefficients, but the embedding function for ${elements[0]} expects ${expect}`);
        }
      }
      if (spec.rank > 2) spec.LS = ints(need(f, 'lint', 'nbody'), 'lint');
      if (f.has('skip')) spec.skip = asBool(f.get('skip')!, 'skip');
      validateLsLS(spec.ls, spec.LS);
      b.funcspecs.push(spec);
    }
    blocks.push(b);
  }

  for (const b of blocks) {
    if (b.numberOfSpecies === 2) {
      // the radial basis and the core repulsion of a two-species block may be given in its dual block only
      const dual = [b.elementsVec[1], b.elementsVec[0]];
      for (const d of blocks) {
        if (!(d.elementsVec.length === 2 && d.elementsVec[0] === dual[0] && d.elementsVec[1] === dual[1])) continue;
        const copyRadial = (from: SpecBlock, to: SpecBlock) => {
          to.lmaxi = from.lmaxi; to.nradmaxi = from.nradmaxi; to.nradbaseij = from.nradbaseij; to.radbase = from.radbase;
          to.radparameters = from.radparameters; to.radcoefficients = from.radcoefficients; to.rcutij = from.rcutij; to.dcutij = from.dcutij;
        };
        if (!b.isRadialBasisDefined) {
          if (d.isRadialBasisDefined) copyRadial(d, b);
          else throw new StyleError(`Radial basis is not defined neither in '${b.blockName}' nor in '${d.blockName}' 2-species blocks`);
        } else if (!d.isRadialBasisDefined) {
          copyRadial(b, d);
        } else if (b.lmaxi !== d.lmaxi || b.nradmaxi !== d.nradmaxi || b.nradbaseij !== d.nradbaseij || b.radbase !== d.radbase ||
          !sameNums(b.radparameters, d.radparameters) || !sameRad(b.radcoefficients, d.radcoefficients) || b.rcutij !== d.rcutij ||
          b.dcutij !== d.dcutij || b.rIn !== d.rIn || b.deltaIn !== d.deltaIn || b.innerCutoffType !== d.innerCutoffType) {
          throw new StyleError(`Radial basis in blocks '${b.blockName}' and '${d.blockName}' is not consistent`);
        }
        const copyCore = (from: SpecBlock, to: SpecBlock) => {
          to.coreRepParameters = from.coreRepParameters; to.innerCutoffType = from.innerCutoffType; to.rIn = from.rIn; to.deltaIn = from.deltaIn;
        };
        if (!b.isCoreRepulsionDefined && d.isCoreRepulsionDefined) copyCore(d, b);
        else if (b.isCoreRepulsionDefined && !d.isCoreRepulsionDefined) copyCore(b, d);
        else if (b.isCoreRepulsionDefined && d.isCoreRepulsionDefined) {
          if (!sameNums(b.coreRepParameters, d.coreRepParameters)) throw new StyleError(`Core-repulsion in blocks '${b.blockName}' and '${d.blockName}' is not consistent`);
          if (b.rIn !== d.rIn || b.deltaIn !== d.deltaIn) throw new StyleError(`Inner cutoff (r_in, delta_in) in blocks '${b.blockName}' and '${d.blockName}' is not consistent`);
        }
      }
    }
    if (b.numberOfSpecies <= 2) {
      let maxNradbase = 0, maxNrad = 0, maxL = 0, maxLS = 0, maxRank = 0;
      for (const s of b.funcspecs) {
        if (s.rank > maxRank) maxRank = s.rank;
        const fns = maxOf(s.ns), fls = maxOf(s.ls);
        if (fls > maxL) maxL = fls;
        if (s.rank === 1) {
          if (fns > maxNradbase) maxNradbase = fns;
          if (b.nradbaseij < fns) throw new StyleError(`Given nradbaseij = ${b.nradbaseij} is less than the max(nr) = ${fns} for rank=1 ${specToString(s)}`);
          if (fls > 0) throw new StyleError(`Given current max(nl) = ${fls} is not equal to 0 for rank = 1 function ${specToString(s)}`);
        } else {
          if (fns > maxNrad) maxNrad = fns;
          if (fns > b.nradmaxi) throw new StyleError(`Given nradmaxi = ${b.nradmaxi} is less than the max(nr) = ${fns} for rank>1 ${specToString(s)}`);
          if (fls > b.lmaxi) throw new StyleError(`Given current max(nl) = ${fls} is larger than block lmaxi = ${b.lmaxi} for function ${specToString(s)}`);
          if (s.rank > 2) {
            const fLS = maxOf(s.LS);
            if (fLS > maxLS) maxLS = fLS;
            if (2 * b.lmaxi < maxLS) throw new StyleError(`Given 2*lmaxi = ${2 * b.lmaxi} is less than the max(lint) = ${maxLS} in function ${specToString(s)}`);
          }
        }
      }
      if (maxRank > b.rankmax) b.rankmax = maxRank;
      if (maxL > b.lmaxi) b.lmaxi = maxL;
      if (maxNrad > b.nradmaxi) b.nradmaxi = maxNrad;
      if (maxNradbase > b.nradbaseij) b.nradbaseij = maxNradbase;
    }
  }
  return { deltaSplineBins, blocks };
};

/** compare (ace_abstract_basis.cpp): rank, mu0, mus, ns, ls, the number of m-combinations, then the combinations. */
const compareFuncs = (b1: { rank: number; mu0: number; mus: Int32Array; ns: Int32Array; ls: Int32Array; numMsCombs: number; msCombs: Int32Array },
  b2: typeof b1): boolean => {
  if (b1.rank !== b2.rank) return b1.rank < b2.rank;
  if (b1.mu0 !== b2.mu0) return b1.mu0 < b2.mu0;
  for (let r = 0; r < b1.rank; r++) if (b1.mus[r] !== b2.mus[r]) return b1.mus[r] < b2.mus[r];
  for (let r = 0; r < b1.rank; r++) if (b1.ns[r] !== b2.ns[r]) return b1.ns[r] < b2.ns[r];
  for (let r = 0; r < b1.rank; r++) if (b1.ls[r] !== b2.ls[r]) return b1.ls[r] < b2.ls[r];
  if (b1.numMsCombs !== b2.numMsCombs) return b1.numMsCombs < b2.numMsCombs;
  for (let i = 0; i < b1.numMsCombs * b1.rank; i++) if (b1.msCombs[i] !== b2.msCombs[i]) return b1.msCombs[i] < b2.msCombs[i];
  return false;
};

/** std::vector's lexicographic operator<, as a three-way comparison. */
const cmpVec = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  return a.length - b.length;
};

/**
 * order_and_compress_b_basis_function: sort the particles of every m-combination by (mu, n, l, m), merge the
 * combinations that become equal, drop those whose coefficient sums below 1e-15, and keep the result only if it is
 * smaller.
 */
const orderAndCompress = (func: BFunc): void => {
  const rank = func.rank;
  const msMap = new Map<string, { ms: number[]; c: number }>();
  let newXS: number[] = [], newNS: number[] = [], newLS: number[] = [];
  for (let k = 0; k < func.numMsCombs; k++) {
    const v: [number, number, number, number, number][] = [];
    for (let r = 0; r < rank; r++) v.push([func.mus[r], func.ns[r], func.ls[r], func.msCombs[k * rank + r], r]);
    v.sort((a, b) => cmpVec(a, b));
    if (k === 0) {
      newXS = v.map((t) => t[0]);
      newNS = v.map((t) => t[1]);
      newLS = v.map((t) => t[2]);
    }
    for (let r = 0; r < rank; r++) {
      if (newXS[r] !== v[r][0] || newNS[r] !== v[r][1] || newLS[r] !== v[r][2]) {
        throw new StyleError(`INCONSISTENT SORTED BLOCK!\n->>sorted XS-ns-ls-ms combinations: {\n${v.map((t) => `(${t[0]}, ${t[1]}, ${t[2]}, ${t[3]})\n`).join('')}}`);
      }
    }
    const ms = v.map((t) => t[3]);
    const key = ms.join(',');
    const e = msMap.get(key);
    if (e) e.c += func.genCgs[k];
    else msMap.set(key, { ms, c: 0 + func.genCgs[k] });
  }
  // std::map order (lexicographic in ms) without the near-zero coefficients
  const kept = [...msMap.values()].filter((e) => !(Math.abs(e.c) < 1e-15)).sort((a, b) => cmpVec(a.ms, b.ms));
  if (func.numMsCombs - kept.length > 0) {
    func.mus = Int32Array.from(newXS);
    func.ns = Int32Array.from(newNS);
    func.ls = Int32Array.from(newLS);
    func.numMsCombs = kept.length;
    func.msCombs = new Int32Array(kept.length * rank);
    func.genCgs = new Float64Array(kept.length);
    kept.forEach((e, i) => {
      for (let r = 0; r < rank; r++) func.msCombs[i * rank + r] = e.ms[r];
      func.genCgs[i] = e.c;
    });
  }
};

/** ACEBBasisSet: the B-basis of a .yaml potential, ready for the B evaluator and for the C-tilde conversion. */
export class BBasis {
  nelements = 0;
  elementsName: string[] = [];
  elementsToIndex = new Map<string, number>();
  lmax = 0;
  nradbase = 0;
  nradmax = 0;
  cutoffmax = 0;
  deltaSplineBins = 0;
  rankmax = 0;
  ndensitymax = 0;
  numMsCombinationsMax = 0;
  E0vals = new Float64Array(0);
  embedding: EmbeddingSpec[] = [];
  radial!: ACERadialFunctions;
  sph!: CartSphericalHarmonics;
  /** mu0_bbasis_vector: every function of each central species, sorted and compressed */
  mu0Basis: BFunc[][] = [];
  totalBasisSizeRank1: number[] = [];
  basisRank1: BFunc[][] = [];
  totalBasisSize: number[] = [];
  basis: BFunc[][] = [];

  speciesIndexByName(name: string): number {
    for (let t = 0; t < this.nelements; t++) if (this.elementsName[t] === name) return t;
    return -1;
  }

  /** ACEBBasisSet::load: BBasisConfiguration::load (Input::parse_input) then initialize_basis. */
  static fromYaml(text: string, filename: string): BBasis {
    const cfg = parseInput(text, filename);
    const b = new BBasis();
    b.initialize(cfg.deltaSplineBins, cfg.blocks, filename);
    return b;
  }

  private initialize(deltaSplineBins: number, blocks: SpecBlock[], filename: string): void {
    const elementsSet = new Set<string>();
    for (const blk of blocks) {
      // update_params
      let nd = blk.ndensityi > 0 ? blk.ndensityi : 1;
      if (blk.funcspecs.length > 0) nd = blk.funcspecs[0].coeffs.length;
      blk.ndensityi = nd;
      let rk = 0;
      for (const s of blk.funcspecs) if (s.rank > rk) rk = s.rank;
      blk.elementsVec = splitKey(blk.blockName);
      if (blk.elementsVec.length === 1) blk.elementsVec.push(blk.elementsVec[0]);
      blk.rankmax = rk;
      blk.numberOfSpecies = new Set(blk.elementsVec).size;
      blk.mu0 = blk.elementsVec[0];
      // validate_individual_functions
      let fnd = 1, bNradmax = 0, bNradbase = 0;
      if (blk.funcspecs.length > 0) fnd = blk.funcspecs[0].coeffs.length;
      for (const s of blk.funcspecs) {
        validateSpec(s);
        if (fnd !== s.coeffs.length) {
          throw new StyleError(`${specToString(s)}:\nNumber of function 'coeffs'(${s.coeffs.length}) is inconsistent with the expected density(${fnd})`);
        }
        if (s.rank === 1) { if (bNradbase < s.ns[0]) bNradbase = s.ns[0]; } else {
          const m = maxOf(s.ns);
          if (bNradmax < m) bNradmax = m;
        }
      }
      if (blk.numberOfSpecies <= 2) {
        if (blk.nradbaseij < bNradbase) throw new StyleError("Function specifications max 'ns' is larger than block nradbaseij");
        if (blk.nradmaxi < bNradmax) throw new StyleError("Function specifications max 'ns' is larger than block nradmaxi");
      }
      validateRadcoefficients(blk);
      if (blk.rankmax > this.rankmax) this.rankmax = blk.rankmax;
      if (blk.ndensityi > this.ndensitymax) this.ndensitymax = blk.ndensityi;
      if (blk.lmaxi > this.lmax) this.lmax = blk.lmaxi;
      if (blk.nradbaseij > this.nradbase) this.nradbase = blk.nradbaseij;
      if (blk.nradmaxi > this.nradmax) this.nradmax = blk.nradmaxi;
      if (blk.rcutij > this.cutoffmax) this.cutoffmax = blk.rcutij;
      for (const el of blk.elementsVec) elementsSet.add(el);
    }
    const names = [...elementsSet].sort();
    for (const el of names) if (!this.elementsToIndex.has(el)) this.elementsToIndex.set(el, this.elementsToIndex.size);
    const nel = names.length;
    const idx = (el: string): number => this.elementsToIndex.get(el)!;

    const radbasename = new Array<string>(nel * nel).fill('ChebExpCos');
    let innerFunc = '';
    const rIn = new Float64Array(nel * nel), deltaIn = new Float64Array(nel * nel);
    for (const blk of blocks) {
      const ei = idx(blk.elementsVec[0]);
      const ej = blk.numberOfSpecies === 2 ? idx(blk.elementsVec[1]) : ei;
      if (blk.numberOfSpecies <= 2) {
        radbasename[ei * nel + ej] = blk.radbase;
        if (innerFunc === '') innerFunc = blk.innerCutoffType;
        else if (innerFunc !== blk.innerCutoffType) {
          throw new StyleError(`\`inner_cutoff_type\` should be identical for all one- and two-species blocks, but \`${blk.innerCutoffType}\` for ${blk.blockName} is different from \`${innerFunc}\``);
        }
        rIn[ei * nel + ej] = blk.rIn;
        deltaIn[ei * nel + ej] = blk.deltaIn;
      }
    }
    if (deltaSplineBins > 0) this.deltaSplineBins = deltaSplineBins;
    else throw new StyleError('ACEBBasisSet:deltaSplineBins should be positive');
    this.nelements = nel;
    for (const name of radbasename) {
      if (name !== 'ChebExpCos' && name !== 'ChebPow' && name !== 'ChebLinear') {
        throw new StyleError(`pair_style pace: radial basis ${name} in ${filename} is not supported`);
      }
    }
    if (innerFunc !== 'density' && innerFunc !== 'distance') {
      throw new StyleError(`pair_style pace: inner_cutoff_type ${innerFunc} in ${filename} is not supported (density and distance are)`);
    }
    const rf = new ACERadialFunctions(this.nradbase, this.lmax, this.nradmax, deltaSplineBins, nel, radbasename);
    this.radial = rf;
    rf.innerCutoffType = innerFunc as 'density' | 'distance';
    for (let p = 0; p < nel * nel; p++) {
      rf.cutIn[p] = rIn[p];
      rf.dcutIn[p] = deltaIn[p];
    }
    this.E0vals = new Float64Array(nel);
    this.embedding = new Array<EmbeddingSpec>(nel);
    const bonds = new Map<number, SpecBlock>();
    for (const blk of blocks) {
      if (blk.numberOfSpecies > 2) continue;
      const ei = idx(blk.elementsVec[0]);
      const ej = blk.elementsVec.length === 1 ? ei : idx(blk.elementsVec[1]);
      const pair = ei * nel + ej;
      bonds.set(pair, blk);
      const sym = bonds.get(ej * nel + ei);
      if (sym && sym !== blk && (sym.lmaxi !== blk.lmaxi || sym.nradbaseij !== blk.nradbaseij || sym.nradmaxi !== blk.nradmaxi ||
        sym.radbase !== blk.radbase || !sameNums(sym.radparameters, blk.radparameters) || !sameRad(sym.radcoefficients, blk.radcoefficients) ||
        sym.coreRepParameters[0] !== blk.coreRepParameters[0] || sym.coreRepParameters[1] !== blk.coreRepParameters[1] ||
        sym.rcutij !== blk.rcutij || sym.dcutij !== blk.dcutij || sym.innerCutoffType !== blk.innerCutoffType ||
        sym.rIn !== blk.rIn || sym.deltaIn !== blk.deltaIn)) {
        throw new StyleError(`Bonds specifications for pair (${ei},${ej}) are inconsistent`);
      }
      rf.lambda[pair] = blk.radparameters[0];
      rf.cut[pair] = blk.rcutij;
      rf.dcut[pair] = blk.dcutij;
      for (let n = 0; n < blk.nradmaxi; n++) {
        for (let l = 0; l <= blk.lmaxi; l++) {
          for (let k = 0; k < blk.nradbaseij; k++) rf.crad[rf.cradIndex(pair, n, l, k)] = blk.radcoefficients[n][l][k];
        }
      }
      rf.prehc[pair] = blk.coreRepParameters[0];
      rf.lambdahc[pair] = blk.coreRepParameters[1];
      if (blk.numberOfSpecies === 1) {
        if (blk.fsParameters.length !== 2 * blk.ndensityi) {
          throw new StyleError(`Number of fs_parameters = ${blk.fsParameters.length} is not equal to ndensity*2 = ${2 * blk.ndensityi} for ele_i = ${ei}`);
        }
        this.embedding[ei] = {
          npoti: blk.npoti, FS: Float64Array.from(blk.fsParameters), ndensity: blk.ndensityi,
          rhoCoreCutoff: blk.rhoCut, drhoCoreCutoff: blk.drhoCut,
        };
      }
    }
    for (let mu = 0; mu < nel; mu++) {
      if (!this.embedding[mu]) throw new StyleError(`${filename}: no embedding (single-species block) for ${names[mu]}`);
    }
    rf.elements = names.slice();
    rf.setuplookupRadspline();
    this.sph = new CartSphericalHarmonics(this.lmax);
    this.elementsName = new Array<string>(nel);
    for (const [el, i] of this.elementsToIndex) this.elementsName[i] = el;

    this.mu0Basis = Array.from({ length: nel }, () => []);
    this.numMsCombinationsMax = 0;
    for (const blk of blocks) {
      const mu0 = idx(blk.mu0);
      if (blk.funcspecs.length === 0) continue;
      const basis = this.mu0Basis[mu0];
      for (const s of blk.funcspecs) {
        if (s.skip) continue;
        const rank = s.rank;
        let pairs;
        try {
          pairs = generateMsCgList(rank, s.ls, rank > 2 ? s.LS : [], true);
        } catch (e) {
          throw new StyleError(`${specToString(s)} could not be constructed: \n${(e as Error).message}`);
        }
        const func: BFunc = {
          rank, rankL: rank > 2 ? rank - 2 : 0, ndensity: blk.ndensityi, mu0: idx(s.elements[0]),
          mus: new Int32Array(rank), ns: Int32Array.from(s.ns), ls: Int32Array.from(s.ls),
          LS: Int32Array.from(rank > 2 ? s.LS.slice(0, rank - 2) : []),
          numMsCombs: pairs.length, msCombs: new Int32Array(rank * pairs.length), genCgs: new Float64Array(pairs.length),
          coeff: Float64Array.from(s.coeffs.slice(0, blk.ndensityi)),
        };
        pairs.forEach((p, i) => {
          for (let r = 0; r < rank; r++) func.msCombs[i * rank + r] = p.ms[r];
          func.genCgs[i] = p.c;
        });
        for (let r = 1; r <= rank; r++) func.mus[r - 1] = idx(s.elements[r]);
        if (this.numMsCombinationsMax < func.numMsCombs) this.numMsCombinationsMax = func.numMsCombs;
        basis.push(func);
        for (let r = 0; r < rank; r++) {
          const bond = bonds.get(func.mu0 * nel + func.mus[r]);
          if (!bond) throw new StyleError(`${filename}: no bond specification for the pair (${func.mu0}, ${func.mus[r]}) of ${specToString(s)}`);
          const bondId = `(${func.mu0}, ${func.mus[r]})`;
          if (rank === 1) {
            if (func.ns[r] > bond.nradbaseij) {
              throw new StyleError(`BASIS INCONSISTENCY: B-function \n${specToString(s)}\n has larger ns = ${func.ns[r]} than the nradbasemax=${bond.nradbaseij} for corresponding bond ${bondId}`);
            }
          } else if (func.ns[r] > bond.nradmaxi) {
            throw new StyleError(`BASIS INCONSISTENCY: B-function \n${specToString(s)}\n has larger ns = ${func.ns[r]} than the nradmax=${bond.nradmaxi} for corresponding bond ${bondId}`);
          }
          if (func.ls[r] > bond.lmaxi) {
            throw new StyleError(`BASIS INCONSISTENCY: B-function \n${specToString(s)}\n has larger ls = ${func.ls[r]} than the lmax=${bond.lmaxi} for corresponding bond ${bondId}`);
          }
        }
      }
      std_sort(basis, 0, basis.length, compareFuncs);
    }
    // compress_basis_functions
    for (let mu = 0; mu < nel; mu++) {
      const sub = this.mu0Basis[mu];
      std_sort(sub, 0, sub.length, compareFuncs);
      for (const func of sub) orderAndCompress(func);
    }
    // flatten_basis
    this.totalBasisSizeRank1 = [];
    this.totalBasisSize = [];
    this.basisRank1 = [];
    this.basis = [];
    for (let mu = 0; mu < nel; mu++) {
      this.basisRank1.push(this.mu0Basis[mu].filter((f) => f.rank === 1));
      this.basis.push(this.mu0Basis[mu].filter((f) => f.rank !== 1));
      this.totalBasisSizeRank1.push(this.basisRank1[mu].length);
      this.totalBasisSize.push(this.basis[mu].length);
    }
  }

  /** to_ACECTildeBasisSet: the functions with equal (mu0, mus, ns, ls) summed over LS into C-tilde functions. */
  toCTilde(): CTildeBasis {
    const c = new CTildeBasis();
    c.nelements = this.nelements;
    c.elementsName = this.elementsName.slice();
    c.lmax = this.lmax;
    c.nradbase = this.nradbase;
    c.nradmax = this.nradmax;
    c.cutoffmax = this.cutoffmax;
    c.deltaSplineBins = this.deltaSplineBins;
    c.rankmax = this.rankmax;
    c.ndensitymax = this.ndensitymax;
    c.numMsCombinationsMax = this.numMsCombinationsMax;
    c.E0vals = this.E0vals.slice();
    c.embedding = this.embedding.slice();
    c.radial = this.radial;
    c.sph = this.sph;
    c.numCtildeMax = 0;
    c.totalBasisSizeRank1 = [];
    c.totalBasisSize = [];
    c.basisRank1 = [];
    c.basis = [];
    for (let mu0 = 0; mu0 < this.nelements; mu0++) {
      // group_basis_functions_by_index: std::map keyed by (X0, Xs, ns, ls), lists in basis order
      const groups = new Map<string, { X0: number; Xs: Int32Array; ns: Int32Array; ls: Int32Array; list: BFunc[] }>();
      for (const f of this.mu0Basis[mu0]) {
        const key = `${f.mu0}|${Array.from(f.mus)}|${Array.from(f.ns)}|${Array.from(f.ls)}`;
        const g = groups.get(key);
        if (g) g.list.push(f);
        else groups.set(key, { X0: f.mu0, Xs: f.mus, ns: f.ns, ls: f.ls, list: [f] });
      }
      const ordered = [...groups.values()].sort((a, b) =>
        a.X0 !== b.X0 ? a.X0 - b.X0 : cmpVec(a.Xs, b.Xs) || cmpVec(a.ns, b.ns) || cmpVec(a.ls, b.ls));
      // summation_over_LS: per m-combination (std::map order) the sum of coeff * gen_cg over the group
      const funcs: CTildeFunc[] = ordered.map((g) => {
        const rank = g.ns.length;
        const nd = g.list[0].ndensity;
        const msMap = new Map<string, { ms: Int32Array; c: Float64Array }>();
        for (const bf of g.list) {
          for (let k = 0; k < bf.numMsCombs; k++) {
            const ms = bf.msCombs.slice(k * rank, (k + 1) * rank);
            const key = Array.from(ms).join(',');
            let e = msMap.get(key);
            if (!e) {
              e = { ms, c: new Float64Array(nd) };
              msMap.set(key, e);
            }
            for (let p = 0; p < nd; p++) e.c[p] += bf.coeff[p] * bf.genCgs[k];
          }
        }
        const combos = [...msMap.values()].sort((a, b) => cmpVec(a.ms, b.ms));
        const msCombs = new Int32Array(rank * combos.length);
        const ctildes = new Float64Array(nd * combos.length);
        combos.forEach((e, i) => {
          msCombs.set(e.ms, i * rank);
          ctildes.set(e.c, i * nd);
        });
        return {
          rank, ndensity: nd, mu0: g.X0, mus: Int32Array.from(g.Xs), ns: Int32Array.from(g.ns), ls: Int32Array.from(g.ls),
          numMsCombs: combos.length, msCombs, ctildes,
        };
      });
      std_sort(funcs, 0, funcs.length, compareFuncs);
      if (c.numCtildeMax < funcs.length) c.numCtildeMax = funcs.length;
      c.basisRank1.push(funcs.filter((f) => f.rank === 1));
      c.basis.push(funcs.filter((f) => f.rank !== 1));
      c.totalBasisSizeRank1.push(c.basisRank1[mu0].length);
      c.totalBasisSize.push(c.basis[mu0].length);
    }
    return c;
  }
}
