/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28) and the ML-PACE library (ACE evaluator), GPL-2.0 use:
 * ML-PACE/ace-evaluator/ace_c_basis.cpp (ACECTildeBasisSet::load_yaml) and ace_abstract_basis.h
 * (ACEBondSpecification::from_YAML) of lammps-user-pace v.2025.12.4.p1 (checked against LAMMPS's CMake SHA256). The LAMMPS
 * package that reads it is src/ML-PACE/pair_pace.cpp at commit b332f28 (a .yace file is the C-tilde basis in YAML).
 * Library authors (from its headers): Yury Lysogorskiy, Cas van der Oord, Anton Bochkarev, Sarath Menon, Matteo Rinaldi,
 * Thomas Hammerschmidt, Matous Mrovec, Aidan Thompson, Gabor Csanyi, Christoph Ortner, Ralf Drautz (ICAMS,
 * Ruhr-University Bochum and co-authors). Used here under GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * The C++ reads the file with yaml-cpp; here a small reader covers the YAML that the .yace files use (written by
 * pacemaker with yaml-cpp's emitter): block mappings and block sequences by indentation, flow mappings and sequences
 * (which may span lines), plain and quoted scalars, comments. Keys are kept as text: a flow-sequence key such as the
 * bond [0, 0] becomes "0,0".
 */
import { StyleError } from '../../engine/force/types';
import { ACERadialFunctions } from './radial';
import { CartSphericalHarmonics } from './spherical';
import { CTildeBasis, type CTildeFunc } from './cbasis';

/** A YAML value of the subset: scalars stay text (converted when read), mappings keep their key order. */
export type YamlValue = string | null | YamlValue[] | YamlMap;
export class YamlMap extends Map<string, YamlValue> {}

interface Line { indent: number; text: string; no: number }

/** Strips a comment (# at the start or after whitespace, outside quotes). */
const stripComment = (s: string): string => {
  let q: string | null = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === q) q = null; continue; }
    if (c === '"' || c === "'") q = c;
    else if (c === '#' && (i === 0 || /\s/.test(s[i - 1]))) return s.slice(0, i);
  }
  return s;
};

/** Flow-style parser over one string ({...}, [...], scalars); pos advances through it. */
class Flow {
  pos = 0;
  constructor(private readonly s: string, private readonly where: string) {}

  private ws(): void {
    while (this.pos < this.s.length && /\s/.test(this.s[this.pos])) this.pos++;
  }

  private fail(msg: string): never {
    throw new StyleError(`${this.where}: ${msg}`);
  }

  value(): YamlValue {
    this.ws();
    const c = this.s[this.pos];
    if (c === '{') return this.map();
    if (c === '[') return this.seq();
    return this.scalar(true);
  }

  /** A scalar: quoted, or plain up to , ] } (and up to ": " when it is a key). */
  scalar(inFlow: boolean, key = false): string | null {
    this.ws();
    const c = this.s[this.pos];
    if (c === '"' || c === "'") {
      let out = '';
      this.pos++;
      while (this.pos < this.s.length) {
        const d = this.s[this.pos];
        if (d === c) {
          if (c === "'" && this.s[this.pos + 1] === "'") { out += "'"; this.pos += 2; continue; }
          this.pos++;
          return out;
        }
        if (c === '"' && d === '\\') { out += this.s[this.pos + 1] ?? ''; this.pos += 2; continue; }
        out += d;
        this.pos++;
      }
      this.fail('unterminated quoted string');
    }
    const start = this.pos;
    while (this.pos < this.s.length) {
      const d = this.s[this.pos];
      if (inFlow && (d === ',' || d === ']' || d === '}')) break;
      if (key && d === ':' && (this.pos + 1 >= this.s.length || /\s/.test(this.s[this.pos + 1]))) break;
      this.pos++;
    }
    const t = this.s.slice(start, this.pos).trim();
    return t === '' || t === '~' || t === 'null' ? null : t;
  }

  private seq(): YamlValue[] {
    this.pos++;
    const out: YamlValue[] = [];
    this.ws();
    if (this.s[this.pos] === ']') { this.pos++; return out; }
    for (;;) {
      out.push(this.value());
      this.ws();
      const c = this.s[this.pos++];
      if (c === ']') return out;
      if (c !== ',') this.fail(`expected , or ] in a flow sequence, found '${c ?? 'end'}'`);
    }
  }

  /** A mapping key: a flow sequence (as "a,b") or a scalar. */
  key(): string {
    this.ws();
    if (this.s[this.pos] === '[') return keyText(this.seq());
    const k = this.scalar(true, true);
    return k ?? '';
  }

  private map(): YamlMap {
    this.pos++;
    const out = new YamlMap();
    this.ws();
    if (this.s[this.pos] === '}') { this.pos++; return out; }
    for (;;) {
      const k = this.key();
      this.ws();
      if (this.s[this.pos] !== ':') this.fail(`expected : after key '${k}'`);
      this.pos++;
      out.set(k, this.value());
      this.ws();
      const c = this.s[this.pos++];
      if (c === '}') return out;
      if (c !== ',') this.fail(`expected , or } in a flow mapping, found '${c ?? 'end'}'`);
    }
  }

  atEnd(): boolean {
    this.ws();
    return this.pos >= this.s.length;
  }

  rest(): string {
    return this.s.slice(this.pos);
  }
}

const keyText = (v: YamlValue[]): string => v.map((x) => (Array.isArray(x) || x instanceof YamlMap ? '?' : x ?? '')).join(',');

/** True while the brackets of a flow value are not closed (outside quotes). */
const openBrackets = (s: string): number => {
  let depth = 0;
  let q: string | null = null;
  for (const c of s) {
    if (q) { if (c === q) q = null; continue; }
    if (c === '"' || c === "'") q = c;
    else if (c === '[' || c === '{') depth++;
    else if (c === ']' || c === '}') depth--;
  }
  return depth;
};

/** Reads the YAML subset of a .yace file into maps, sequences and scalar texts. */
export const parseYaml = (text: string, filename: string): YamlValue => {
  const lines: Line[] = [];
  const raw = text.split(/\r?\n/);
  for (let i = 0; i < raw.length; i++) {
    const t = stripComment(raw[i]).replace(/\s+$/, '');
    if (t.trim() === '' || t.trim() === '---' || t.trim() === '...') continue;
    lines.push({ indent: t.length - t.trimStart().length, text: t.trimStart(), no: i + 1 });
  }
  let li = 0;
  const where = (l: Line) => `${filename} line ${l.no}`;

  /**
   * A value after "key:" or "- ": a flow collection (continued on the next lines until its brackets close), or a scalar
   * that runs to the end of the line (block context: commas belong to it).
   */
  const flowValue = (first: string, l: Line): YamlValue => {
    if (first[0] !== '[' && first[0] !== '{') {
      if (first[0] === '"' || first[0] === "'") return new Flow(first, where(l)).scalar(false);
      const t = first.trim();
      return t === '~' || t === 'null' ? null : t;
    }
    let s = first;
    while (openBrackets(s) > 0 && li < lines.length) s += ' ' + lines[li++].text;
    const f = new Flow(s, where(l));
    const v = f.value();
    if (!f.atEnd()) throw new StyleError(`${where(l)}: unexpected text after a value: ${f.rest().trim()}`);
    return v;
  };

  const block = (indent: number): YamlValue => {
    if (li >= lines.length || lines[li].indent < indent) return null;
    const ind = lines[li].indent;
    if (lines[li].text.startsWith('- ') || lines[li].text === '-') {
      const out: YamlValue[] = [];
      while (li < lines.length && lines[li].indent === ind && (lines[li].text.startsWith('- ') || lines[li].text === '-')) {
        const l = lines[li++];
        const rest = l.text.slice(1).trim();
        if (rest === '') { out.push(block(ind + 1)); continue; }
        if (rest[0] === '{' || rest[0] === '[' || !/:(\s|$)/.test(rest)) { out.push(flowValue(rest, l)); continue; }
        // a compact mapping "- key: value" continued by the deeper lines
        const sub = ind + (l.text.length - rest.length);
        li--;
        lines[li] = { indent: sub, text: rest, no: l.no };
        out.push(mapping(sub));
      }
      return out;
    }
    return mapping(ind);
  };

  const mapping = (ind: number): YamlMap => {
    const out = new YamlMap();
    while (li < lines.length && lines[li].indent === ind) {
      const l = lines[li++];
      const f = new Flow(l.text, where(l));
      const k = f.key();
      const after = f.rest();
      const m = /^\s*:(\s+|$)/.exec(after);
      if (!m) throw new StyleError(`${where(l)}: expected 'key: value'`);
      const rest = after.slice(m[0].length).trim();
      if (rest === '') out.set(k, block(ind + 1));
      else out.set(k, flowValue(rest, l));
    }
    if (li < lines.length && lines[li].indent > ind) throw new StyleError(`${where(lines[li])}: unexpected indentation`);
    return out;
  };

  const v = block(0);
  if (li < lines.length) throw new StyleError(`${where(lines[li])}: unexpected text`);
  return v;
};

// --- typed access in the manner of yaml-cpp's as<T>()

const need = (m: YamlValue, key: string, what: string): YamlValue => {
  if (!(m instanceof YamlMap) || !m.has(key)) throw new StyleError(`${what}: missing key '${key}'`);
  return m.get(key)!;
};
const asNum = (v: YamlValue, what: string): number => {
  const x = typeof v === 'string' ? Number(v) : NaN;
  if (typeof v !== 'string' || v.trim() === '' || Number.isNaN(x)) throw new StyleError(`${what}: expected a number, found ${JSON.stringify(v)}`);
  return x;
};
const asInt = (v: YamlValue, what: string): number => {
  const x = asNum(v, what);
  if (!Number.isInteger(x)) throw new StyleError(`${what}: expected an integer, found ${v}`);
  return x;
};
const asStr = (v: YamlValue, what: string): string => {
  if (typeof v !== 'string') throw new StyleError(`${what}: expected a string`);
  return v;
};
const asSeq = (v: YamlValue, what: string): YamlValue[] => {
  if (!Array.isArray(v)) throw new StyleError(`${what}: expected a sequence`);
  return v;
};
const asMap = (v: YamlValue, what: string): YamlMap => {
  if (!(v instanceof YamlMap)) throw new StyleError(`${what}: expected a mapping`);
  return v;
};
const nums = (v: YamlValue, what: string): number[] => asSeq(v, what).map((x) => asNum(x, what));
const ints = (v: YamlValue, what: string): number[] => asSeq(v, what).map((x) => asInt(x, what));

/** ACEBondSpecification (ace_abstract_basis.h) as from_YAML reads it. */
interface BondSpec {
  nradmax: number; lmax: number; nradbasemax: number; radbasename: string;
  radparameters: number[]; radcoefficients: number[][][];
  prehc: number; lambdahc: number; rcut: number; dcut: number;
  rcutIn: number; dcutIn: number; innerCutoffType: string;
}

const bondFromYaml = (b: YamlMap, what: string): BondSpec => {
  const radbasename = asStr(need(b, 'radbasename', what), `${what}::radbasename`);
  const opt = (k: string, dflt: number): number => (b.has(k) ? asNum(b.get(k)!, `${what}::${k}`) : dflt);
  return {
    radbasename,
    nradmax: asInt(need(b, 'nradmax', what), `${what}::nradmax`),
    lmax: asInt(need(b, 'lmax', what), `${what}::lmax`),
    nradbasemax: asInt(need(b, 'nradbasemax', what), `${what}::nradbasemax`),
    radparameters: nums(need(b, 'radparameters', what), `${what}::radparameters`),
    radcoefficients: asSeq(need(b, 'radcoefficients', what), `${what}::radcoefficients`)
      .map((n) => asSeq(n, `${what}::radcoefficients`).map((l) => nums(l, `${what}::radcoefficients`))),
    prehc: asNum(need(b, 'prehc', what), `${what}::prehc`),
    lambdahc: asNum(need(b, 'lambdahc', what), `${what}::lambdahc`),
    rcut: asNum(need(b, 'rcut', what), `${what}::rcut`),
    dcut: asNum(need(b, 'dcut', what), `${what}::dcut`),
    rcutIn: opt('rcut_in', 0),
    dcutIn: opt('dcut_in', 0),
    // from_YAML: "default value to read for backward compatibility"
    innerCutoffType: b.has('inner_cutoff_type') ? asStr(b.get('inner_cutoff_type')!, `${what}::inner_cutoff_type`) : 'density',
  };
};

/** ACECTildeBasisSet::load_yaml for the text of a .yace file. */
export const loadYace = (text: string, filename: string): CTildeBasis => {
  const root = asMap(parseYaml(text, filename), filename);
  const b = new CTildeBasis();
  b.elementsName = asSeq(need(root, 'elements', filename), 'elements').map((e) => asStr(e, 'elements'));
  b.nelements = b.elementsName.length;
  const nel = b.nelements;
  b.E0vals = new Float64Array(nel);
  nums(need(root, 'E0', filename), 'E0').forEach((x, i) => { if (i < nel) b.E0vals[i] = x; });

  // embeddings
  b.embedding = [];
  b.ndensitymax = 0;
  for (const [k, v] of asMap(need(root, 'embeddings', filename), 'embeddings')) {
    const mu = asInt(k, 'embeddings key');
    if (mu > nel - 1) throw new StyleError('yace::embeddings has species type key larger than nelements');
    const e = asMap(v, 'embeddings');
    const ndensity = asInt(need(e, 'ndensity', 'embedding'), 'embedding::ndensity');
    b.embedding[mu] = {
      ndensity,
      FS: Float64Array.from(nums(need(e, 'FS_parameters', 'embedding'), 'embedding::FS_parameters')),
      npoti: asStr(need(e, 'npoti', 'embedding'), 'embedding::npoti'),
      rhoCoreCutoff: asNum(need(e, 'rho_core_cutoff', 'embedding'), 'embedding::rho_core_cutoff'),
      drhoCoreCutoff: asNum(need(e, 'drho_core_cutoff', 'embedding'), 'embedding::drho_core_cutoff'),
    };
    if (ndensity > b.ndensitymax) b.ndensitymax = ndensity;
  }
  for (let mu = 0; mu < nel; mu++) if (!b.embedding[mu]) throw new StyleError(`${filename}: no embedding for species ${mu}`);

  // bonds
  const bonds = new Map<string, BondSpec>();
  let jl = false, paceRad = false;
  for (const [k, v] of asMap(need(root, 'bonds', filename), 'bonds')) {
    const spec = bondFromYaml(asMap(v, 'bonds'), `bonds[${k}]`);
    if (spec.radbasename.startsWith('ACE.jl')) jl = true; else paceRad = true;
    const pair = k.split(',').map((x) => asInt(x.trim(), 'bonds key'));
    if (pair.length !== 2) throw new StyleError(`${filename}: a bonds key must be a pair [mu_i, mu_j]`);
    if (pair[0] > nel - 1 || pair[1] > nel - 1) throw new StyleError('yace::bonds has species type key larger than nelements');
    bonds.set(`${pair[0]},${pair[1]}`, spec);
  }
  if (jl && paceRad) throw new StyleError("Only ACE.jl.* or PACE's radial basis are possible, but both types are used simultaneously.");
  if (jl) throw new StyleError(`pair_style pace: the ACE.jl radial basis (SHIPs) in ${filename} is not supported yet`);
  b.lmax = 0; b.nradmax = 0; b.nradbase = 0; b.cutoffmax = 0;
  // radbasename_ij of load_yaml: a pair without a bond specification keeps an empty name (and zero cutoffs)
  const radbasename = new Array<string>(nel * nel).fill('');
  let innerType = '';
  for (const [key, spec] of bonds) {
    if (spec.nradmax > b.nradmax) b.nradmax = spec.nradmax;
    if (spec.lmax > b.lmax) b.lmax = spec.lmax;
    if (spec.nradbasemax > b.nradbase) b.nradbase = spec.nradbasemax;
    if (spec.rcut > b.cutoffmax) b.cutoffmax = spec.rcut;
    if (spec.radbasename !== 'ChebExpCos' && spec.radbasename !== 'ChebPow' && spec.radbasename !== 'ChebLinear') {
      throw new StyleError(`pair_style pace: radial basis ${spec.radbasename} in ${filename} is not supported`);
    }
    const [bi, bj] = key.split(',').map(Number);
    radbasename[bi * nel + bj] = spec.radbasename;
    innerType = spec.innerCutoffType;
  }
  if (innerType !== 'density' && innerType !== 'distance') {
    throw new StyleError(`pair_style pace: inner_cutoff_type ${innerType} in ${filename} is not supported (density and distance are)`);
  }
  b.deltaSplineBins = asNum(need(root, 'deltaSplineBins', filename), 'deltaSplineBins');
  const rf = new ACERadialFunctions(b.nradbase, b.lmax, b.nradmax, b.deltaSplineBins, nel, radbasename);
  b.radial = rf;
  for (let mi = 0; mi < nel; mi++) {
    for (let mj = 0; mj < nel; mj++) {
      const spec = bonds.get(`${mi},${mj}`);
      const pair = mi * nel + mj;
      // a pair without a specification: the default ACEBondSpecification of the C++ map (zero cutoffs, no coefficients)
      if (!spec) { rf.cut[pair] = 0; rf.dcut[pair] = 0; rf.prehc[pair] = 0; rf.lambdahc[pair] = 0; rf.lambda[pair] = 0; continue; }
      rf.cut[pair] = spec.rcut;
      rf.dcut[pair] = spec.dcut;
      rf.prehc[pair] = spec.prehc;
      rf.lambdahc[pair] = spec.lambdahc;
      rf.lambda[pair] = spec.radparameters[0];
      rf.cutIn[pair] = spec.rcutIn;
      rf.dcutIn[pair] = spec.dcutIn;
      rf.innerCutoffType = spec.innerCutoffType as 'density' | 'distance';
      for (let n = 0; n < spec.nradmax; n++) {
        for (let l = 0; l <= spec.lmax; l++) {
          for (let k = 0; k < spec.nradbasemax; k++) {
            const c = spec.radcoefficients[n]?.[l]?.[k];
            if (c === undefined) throw new StyleError(`${filename}: radcoefficients of [${mi}, ${mj}] are too short`);
            rf.crad[rf.cradIndex(pair, n, l, k)] = c;
          }
        }
      }
    }
  }
  b.sph = new CartSphericalHarmonics(b.lmax);
  rf.elements = b.elementsName.slice();
  rf.setuplookupRadspline();

  // basis functions, rank 1 and higher rank, in file order
  b.totalBasisSizeRank1 = new Array(nel).fill(0);
  b.totalBasisSize = new Array(nel).fill(0);
  b.basisRank1 = Array.from({ length: nel }, () => []);
  b.basis = Array.from({ length: nel }, () => []);
  b.rankmax = 0;
  for (const [k, v] of asMap(need(root, 'functions', filename), 'functions')) {
    const mu = asInt(k, 'functions key');
    if (mu > nel - 1) throw new StyleError('yace::functions has species type key larger than nelements');
    for (const fy of asSeq(v, 'functions')) {
      const f = asMap(fy, 'functions');
      const rank = asInt(need(f, 'rank', 'function'), 'rank');
      const ndensity = asInt(need(f, 'ndensity', 'function'), 'ndensity');
      const numMs = asInt(need(f, 'num_ms_combs', 'function'), 'num_ms_combs');
      const vec = (key: string, len: number): number[] => {
        const x = ints(need(f, key, 'function'), key);
        if (x.length !== len) throw new StyleError(`${key}:: not sufficient number of values`);
        return x;
      };
      const ctildes = nums(need(f, 'ctildes', 'function'), 'ctildes');
      if (ctildes.length !== ndensity * numMs) throw new StyleError('ctildes:: not sufficient number of values');
      const func: CTildeFunc = {
        rank, ndensity, mu0: asInt(need(f, 'mu0', 'function'), 'mu0'),
        mus: Int32Array.from(vec('mus', rank)), ns: Int32Array.from(vec('ns', rank)), ls: Int32Array.from(vec('ls', rank)),
        numMsCombs: numMs, msCombs: Int32Array.from(vec('ms_combs', rank * numMs)), ctildes: Float64Array.from(ctildes),
      };
      if (rank === 1) { b.basisRank1[mu].push(func); b.totalBasisSizeRank1[mu]++; } else { b.basis[mu].push(func); b.totalBasisSize[mu]++; }
      if (b.rankmax < rank) b.rankmax = rank;
    }
  }
  return b;
};
