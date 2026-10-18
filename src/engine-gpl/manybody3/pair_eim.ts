/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANYBODY/pair_eim.cpp and
 * src/MANYBODY/pair_eim.h (PairEIM, EIMPotentialFileReader: the potential file parser, file2array, array2spline,
 * interpolate, funccutoff, funcphi, funcsigma, funccoul, compute). Contributing author: Xiaowang Zhou (SNL). Copyright
 * (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The tables are built as native does: the setfl arrays on nr = 5000 points out to the largest cutoff, then the cubic
 * splines of interpolate(). The half neighbour list is summed in native order: rho and then fp are reverse-communicated
 * and forward-communicated between the passes, and the forces are applied pair by pair.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { erfc } from '../libm/erf';
import { exp } from '../libm/exp';
import { pow } from '../libm/pow';
import { METAL2REAL_ENERGY, REAL2METAL_ENERGY, nextDouble, tallyPair } from '../manybody2/common';
import { forwardCopy, reverseAdd } from '../manyeam/pair_base';

/** setfl->nr (pair_eim.cpp read_file). */
const NR = 5000;
/** Seven spline coefficients per table row (spline[m][0..6]). */
const W = 7;

export interface EimElement {
  name: string;
  ielement: number;
  mass: number;
  negativity: number;
  ra: number;
  ri: number;
  Ec: number;
  q0: number;
}

export interface EimPair {
  rcutphiA: number;
  rcutphiR: number;
  Eb: number;
  r0: number;
  alpha: number;
  beta: number;
  rcutq: number;
  Asigma: number;
  rq: number;
  rcutsigma: number;
  Ac: number;
  zeta: number;
  rs: number;
  tp: number;
}

export interface EimFile {
  division: number;
  rbig: number;
  rsmall: number;
  /** Keyed by the element name. */
  elements: Map<string, EimElement>;
  /** Keyed by the two names in sorted order ("a b", a <= b), as EIMPotentialFileReader::get_pair. */
  pairs: Map<string, EimPair>;
}

/** EIMPotentialFileReader::get_pair: the key of an unordered element pair. */
const pairKey = (a: string, b: string): string => (a < b ? `${a} ${b}` : `${b} ${a}`);

/** The packed index of an element pair (pair_eim.cpp read_file and funcphi use the same formula). */
const pairIndex = (i: number, j: number, n: number): number => {
  if (i === j) return i;
  if (i < j) return n * (i + 1) - ((i + 1) * (i + 2)) / 2 + j;
  return n * (j + 1) - ((j + 1) * (j + 2)) / 2 + i;
};

/** Count of ValueTokenizer words (Tokenizer::count). */
const countWords = (s: string): number => {
  const t = s.trim();
  return t === '' ? 0 : t.split(/\s+/).length;
};

/**
 * EIMPotentialFileReader::next_line: a line is read with its comment ('#' to the end) removed; a line ending in '&'
 * is joined with the next one; blank lines are skipped. At the end of the file a blank line already read is returned
 * (native returns the buffer), and the caller then finds no words.
 */
class EimLines {
  private pos = 0;
  constructor(private readonly raw: string[]) {}

  private fgets(): string | null {
    return this.pos < this.raw.length ? this.raw[this.pos++] : null;
  }

  next(): string | null {
    let raw = this.fgets();
    if (raw === null) return null;
    let line = raw;
    let h = line.indexOf('#');
    if (h >= 0) line = line.slice(0, h);
    let a = line.lastIndexOf('&');
    let concat = false;
    if (a >= 0) {
      concat = true;
      line = line.slice(0, a);
    }
    let nwords = countWords(line);
    let n = nwords > 0 ? line.length : 0;
    while (n === 0 || concat) {
      raw = this.fgets();
      if (raw === null) return line;
      line = line.slice(0, n) + raw;
      h = line.indexOf('#');
      if (h >= 0) line = line.slice(0, h);
      a = line.lastIndexOf('&');
      if (a >= 0) {
        concat = true;
        line = line.slice(0, a);
      } else {
        concat = false;
      }
      nwords += countWords(line.slice(n));
      if (nwords > 0) n = line.length;
    }
    return line;
  }
}

/** ValueTokenizer::next_int. */
const nextInt = (w: string | undefined): number => {
  if (w === undefined) throw new StyleError('No more tokens');
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Not a valid integer number: '${w}'`);
  return Number(w);
};

/**
 * EIMPotentialFileReader (constructor and parse). conversion is the energy factor of the unit conversion (Eb and Ac
 * are multiplied by it, as in native). Lines other than global:, element: and pair: are ignored.
 */
export const parseEimFile = (text: string, conversion = 1): EimFile => {
  const raw = text === '' ? [] : text.split(/(?<=\n)/);
  const reader = new EimLines(raw);
  let found = false;
  let division = 0, rbig = 0, rsmall = 0;
  const elements = new Map<string, EimElement>();
  const pairs = new Map<string, EimPair>();
  for (let line = reader.next(); line !== null; line = reader.next()) {
    const w = line.split(/\s+/).filter((s) => s !== '');
    if (w.length === 0) throw new StyleError('No more tokens');
    const type = w[0];
    if (type === 'global:') {
      if (w.length !== 4) throw new StyleError('Invalid global line in EIM potential file');
      division = nextDouble(w[1]);
      rbig = nextDouble(w[2]);
      rsmall = nextDouble(w[3]);
      found = true;
    } else if (type === 'element:') {
      if (w.length !== 9) throw new StyleError('Invalid element line in EIM potential file');
      const name = w[1];
      const data: EimElement = {
        name,
        ielement: nextInt(w[2]),
        mass: nextDouble(w[3]),
        negativity: nextDouble(w[4]),
        ra: nextDouble(w[5]),
        ri: nextDouble(w[6]),
        Ec: nextDouble(w[7]),
        q0: nextDouble(w[8]),
      };
      if (elements.has(name)) throw new StyleError('Duplicate pair line in EIM potential file');
      elements.set(name, data);
    } else if (type === 'pair:') {
      if (w.length !== 17) throw new StyleError('Invalid element line in EIM potential file');
      const elementA = w[1], elementB = w[2];
      const data: EimPair = {
        rcutphiA: nextDouble(w[3]),
        rcutphiR: nextDouble(w[4]),
        Eb: nextDouble(w[5]) * conversion,
        r0: nextDouble(w[6]),
        alpha: nextDouble(w[7]),
        beta: nextDouble(w[8]),
        rcutq: nextDouble(w[9]),
        Asigma: nextDouble(w[10]),
        rq: nextDouble(w[11]),
        rcutsigma: nextDouble(w[12]),
        Ac: nextDouble(w[13]) * conversion,
        zeta: nextDouble(w[14]),
        rs: nextDouble(w[15]),
        // native: (int)values.next_double(), for the 1.0000e+00 format of the existing files
        tp: Math.trunc(nextDouble(w[16])),
      };
      const key = pairKey(elementA, elementB);
      if (pairs.has(key)) throw new StyleError('Duplicate pair line in EIM potential file');
      pairs.set(key, data);
    }
  }
  if (!found) throw new StyleError('Missing global line in EIM potential file');
  return { division, rbig, rsmall, elements, pairs };
};

/**
 * The energy factor of the potential file's UNITS tag (utils::open_potential with the auto-convert mask of EIM:
 * metal and real convert, any other mismatch is an error naming the file).
 */
export const eimUnitFactor = (text: string, fileName: string, ctx: StyleContext): number => {
  const first = (text.split('\n')[0] ?? '').split(/\s+/).filter((s) => s !== '');
  const k = first.indexOf('UNITS:');
  const units = k >= 0 && k + 1 < first.length ? first[k + 1] : '';
  const style = ctx.s?.units.style ?? 'lj';
  if (units === '' || units === style) return 1;
  if (units === 'metal' && style === 'real') {
    ctx.log(`WARNING: Converting potential file in ${units} units to ${style} units`);
    return METAL2REAL_ENERGY;
  }
  if (units === 'real' && style === 'metal') {
    ctx.log(`WARNING: Converting potential file in ${units} units to ${style} units`);
    return REAL2METAL_ENERGY;
  }
  throw new StyleError(`Potential file ${fileName} requires ${units} units but ${style} units are in use`);
};

/**
 * The cutoff function (pair_eim.cpp funccutoff): erfc of a line from rbig at rp to rsmall at rc, normalised between
 * erfc(rbig) (value 1) and erfc(rsmall) (value 0).
 */
const funccutoff = (rbig: number, rsmall: number, rp: number, rc: number, r: number): number => {
  let a = ((rsmall - rbig) / (rc - rp)) * (r - rp) + rbig;
  a = erfc(a);
  const b = erfc(rbig);
  const c = erfc(rsmall);
  return (a - c) / (b - c);
};

/** funcphi (pair_eim.cpp): the pair term of the element pair, with r clamped to 0.2 from below. */
const funcphi = (pr: EimPair, rbig: number, rsmall: number, rin: number): number => {
  let r = rin;
  let value = 0.0;
  if (r < 0.2) r = 0.2;
  if (pr.tp === 1) {
    const a = (pr.Eb * pr.alpha) / (pr.beta - pr.alpha);
    const b = (pr.Eb * pr.beta) / (pr.beta - pr.alpha);
    if (r < pr.rcutphiA) {
      value -= a * exp(-pr.beta * (r / pr.r0 - 1.0)) * funccutoff(rbig, rsmall, pr.r0, pr.rcutphiA, r);
    }
    if (r < pr.rcutphiR) {
      value += b * exp(-pr.alpha * (r / pr.r0 - 1.0)) * funccutoff(rbig, rsmall, pr.r0, pr.rcutphiR, r);
    }
  } else if (pr.tp === 2) {
    const a = (pr.Eb * pr.alpha * pow(pr.r0, pr.beta)) / (pr.beta - pr.alpha);
    const b = ((a * pr.beta) / pr.alpha) * pow(pr.r0, pr.alpha - pr.beta);
    if (r < pr.rcutphiA) {
      value -= (a / pow(r, pr.beta)) * funccutoff(rbig, rsmall, pr.r0, pr.rcutphiA, r);
    }
    if (r < pr.rcutphiR) {
      value += (b / pow(r, pr.alpha)) * funccutoff(rbig, rsmall, pr.r0, pr.rcutphiR, r);
    }
  }
  return value;
};

/** funcsigma: the charge transfer eta_ij, A (chi_j - chi_i) fc (negativity of the element of index i and j). */
const funcsigma = (pr: EimPair, chiI: number, chiJ: number, rbig: number, rsmall: number, rin: number): number => {
  let r = rin;
  let value = 0.0;
  if (r < 0.2) r = 0.2;
  if (r < pr.rcutq) {
    value = (pr.Asigma * (chiJ - chiI)) * funccutoff(rbig, rsmall, pr.rq, pr.rcutq, r);
  }
  return value;
};

/** funccoul: the charge-charge function psi_ij, A exp(-zeta r) fc. */
const funccoul = (pr: EimPair, rbig: number, rsmall: number, rin: number): number => {
  let r = rin;
  let value = 0.0;
  if (r < 0.2) r = 0.2;
  if (r < pr.rcutsigma) {
    value = (pr.Ac * exp(-pr.zeta * r)) * funccutoff(rbig, rsmall, pr.rs, pr.rcutsigma, r);
  }
  return value;
};

/** interpolate() of pair_eim.cpp (its end row differs from the EAM one: spline[n][5] = 0). */
export const interpolateEim = (n: number, delta: number, f: Float64Array): Float64Array => {
  const s = new Float64Array((n + 1) * W);
  const at = (m: number, k: number): number => m * W + k;
  for (let m = 1; m <= n; m++) s[at(m, 6)] = f[m];
  s[at(1, 5)] = s[at(2, 6)] - s[at(1, 6)];
  s[at(2, 5)] = 0.5 * (s[at(3, 6)] - s[at(1, 6)]);
  s[at(n - 1, 5)] = 0.5 * (s[at(n, 6)] - s[at(n - 2, 6)]);
  s[at(n, 5)] = 0.0;
  for (let m = 3; m <= n - 2; m++) {
    s[at(m, 5)] = ((s[at(m - 2, 6)] - s[at(m + 2, 6)]) + 8.0 * (s[at(m + 1, 6)] - s[at(m - 1, 6)])) / 12.0;
  }
  for (let m = 1; m <= n - 1; m++) {
    s[at(m, 4)] = 3.0 * (s[at(m + 1, 6)] - s[at(m, 6)]) - 2.0 * s[at(m, 5)] - s[at(m + 1, 5)];
    s[at(m, 3)] = s[at(m, 5)] + s[at(m + 1, 5)] - 2.0 * (s[at(m + 1, 6)] - s[at(m, 6)]);
  }
  s[at(n, 4)] = 0.0;
  s[at(n, 3)] = 0.0;
  for (let m = 1; m <= n; m++) {
    s[at(m, 2)] = s[at(m, 5)] / delta;
    s[at(m, 1)] = (2.0 * s[at(m, 4)]) / delta;
    s[at(m, 0)] = (3.0 * s[at(m, 3)]) / delta;
  }
  return s;
};

/** The setfl arrays of read_file for the names of the mapped elements (in order of first appearance). */
interface EimSetfl {
  nel: number;
  elems: EimElement[];
  pairs: EimPair[];
  rbig: number;
  rsmall: number;
  dr: number;
  /** setfl->cut: the largest cutoff of all element pairs. */
  cut: number;
  /** setfl->cuts[i][j] at i * nel + j. */
  cuts: Float64Array;
  /** Raw tables Fij, Gij, phiij per element pair (index i * nel + j), rows 1..NR (row k+1 is r = k dr). */
  Fij: Float64Array[];
  Gij: Float64Array[];
  phiij: Float64Array[];
}

const buildSetfl = (file: EimFile, names: string[]): EimSetfl => {
  const nel = names.length;
  const npair = (nel * (nel + 1)) / 2;
  const elems: EimElement[] = [];
  for (let i = 0; i < nel; i++) {
    const e = file.elements.get(names[i]);
    if (!e) throw new StyleError(`Element ${names[i]} not defined in EIM potential file`);
    elems.push(e);
  }
  const pairs: EimPair[] = new Array(npair);
  for (let i = 0; i < nel; i++) {
    for (let j = i; j < nel; j++) {
      const p = file.pairs.get(pairKey(names[i], names[j]));
      if (!p) throw new StyleError(`Element pair (${names[i]}, ${names[j]}) is not defined in EIM potential file`);
      pairs[pairIndex(i, j, nel)] = p;
    }
  }
  const { rbig, rsmall } = file;
  let cut = 0.0;
  for (let ij = 0; ij < npair; ij++) {
    const p = pairs[ij];
    if (cut < p.rcutphiA) cut = p.rcutphiA;
    if (cut < p.rcutphiR) cut = p.rcutphiR;
    if (cut < p.rcutq) cut = p.rcutq;
    if (cut < p.rcutsigma) cut = p.rcutsigma;
  }
  const dr = cut / (NR - 1.0);
  const cuts = new Float64Array(nel * nel);
  for (let i = 0; i < nel; i++) {
    for (let j = 0; j < nel; j++) {
      if (i > j) {
        cuts[i * nel + j] = cuts[j * nel + i];
      } else {
        const p = pairs[pairIndex(i, j, nel)];
        let c = p.rcutphiA;
        if (c < p.rcutphiR) c = p.rcutphiR;
        if (c < p.rcutq) c = p.rcutq;
        if (c < p.rcutsigma) c = p.rcutsigma;
        cuts[i * nel + j] = c;
      }
    }
  }
  const Fij: Float64Array[] = [], Gij: Float64Array[] = [], phiij: Float64Array[] = [];
  for (let i = 0; i < nel * nel; i++) {
    Fij.push(new Float64Array(NR + 1));
    Gij.push(new Float64Array(NR + 1));
    phiij.push(new Float64Array(NR + 1));
  }
  for (let i = 0; i < nel; i++) {
    for (let j = 0; j < nel; j++) {
      const ij = pairIndex(i, j, nel);
      for (let k = 0; k < NR; k++) {
        if (i > j) {
          phiij[i * nel + j][k + 1] = phiij[j * nel + i][k + 1];
        } else {
          phiij[i * nel + j][k + 1] = funcphi(pairs[ij], rbig, rsmall, k * dr);
        }
      }
    }
  }
  for (let i = 0; i < nel; i++) {
    for (let j = 0; j < nel; j++) {
      const ij = pairIndex(i, j, nel);
      for (let k = 0; k < NR; k++) {
        Fij[i * nel + j][k + 1] = funcsigma(pairs[ij], elems[i].negativity, elems[j].negativity, rbig, rsmall, k * dr);
      }
    }
  }
  for (let i = 0; i < nel; i++) {
    for (let j = 0; j < nel; j++) {
      const ij = pairIndex(i, j, nel);
      for (let k = 0; k < NR; k++) {
        if (i > j) {
          Gij[i * nel + j][k + 1] = Gij[j * nel + i][k + 1];
        } else {
          Gij[i * nel + j][k + 1] = funccoul(pairs[ij], rbig, rsmall, k * dr);
        }
      }
    }
  }
  return { nel, elems, pairs, rbig, rsmall, dr, cut, cuts, Fij, Gij, phiij };
};

/** The spline tables of array2spline (per element pair) and the reciprocal spacing. */
interface EimSplines {
  rdr: number;
  Fij: Float64Array[];
  Gij: Float64Array[];
  phiij: Float64Array[];
}

/** The value polynomial of a spline row m (p in [0, 1]): ((c3 p + c4) p + c5) p + c6. */
const polyVal = (c: Float64Array, o: number, p: number): number => ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
/** The derivative polynomial of a spline row m: (c0 p + c1) p + c2. */
const polyDer = (c: Float64Array, o: number, p: number): number => (c[o] * p + c[o + 1]) * p + c[o + 2];

/**
 * pair_style eim. pair_coeff * * El1 ... ElK file Map1 ... MapN: K is the number of distinct element names of the mapping,
 * and the file follows the K names. Forces are summed over the half list in native order.
 */
export class PairEIM extends Pair {
  readonly name = 'eim';
  manybody = true;
  needsHalf = true;
  needsFull = false;
  virialFdotr = true;

  private setfl: EimSetfl | null = null;
  private splines: EimSplines | null = null;
  /** Element index of each atom type (-1 for NULL). */
  private elemOf = new Int32Array(0);
  /** cutforcesq per type pair (i * nt + j). */
  private cutforcesq = new Float64Array(0);
  private rho = new Float64Array(0);
  private fp = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.elemOf = new Int32Array(ntypes + 1).fill(-1);
    this.setfl = null;
    this.splines = null;
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 5) throw new StyleError('Incorrect args for pair coefficients');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Incorrect args for pair coefficients');
    // map_element2type(ntypes, arg + (narg - ntypes)): the element names of the types, in order of first appearance
    const mapping = args.slice(args.length - this.ntypes);
    const names: string[] = [];
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const entry = mapping[t - 1];
      if (entry === 'NULL') continue;
      let e = names.indexOf(entry);
      if (e < 0) {
        e = names.length;
        names.push(entry);
      }
      map[t] = e;
    }
    // map_element2type: no mapped type is an error (pair.cpp setflag count)
    if (names.length === 0) throw new StyleError('Incorrect args for pair coefficients');
    // read_file(arg[2 + nelements]): the file follows the distinct names
    const fileName = args[2 + names.length];
    if (fileName === undefined) throw new StyleError('Incorrect args for pair coefficients');
    const text = ctx.readFile(fileName);
    const conversion = eimUnitFactor(text, fileName, ctx);
    const file = parseEimFile(text, conversion);
    const setfl = buildSetfl(file, names);
    this.setfl = setfl;
    this.elemOf = map;
    this.splines = null;
    // atom->set_mass for the mapped I,I types
    for (let i = 1; i <= this.ntypes; i++) {
      if (map[i] >= 0) this.setMass(ctx, i, setfl.elems[map[i]].mass);
    }
  }

  /** atom->set_mass (atom.cpp): a value <= 0 is ignored with a warning. */
  private setMass(ctx: StyleContext, type: number, value: number): void {
    const s = ctx.s;
    if (!s) return;
    if (s.rmass) throw new StyleError('Cannot set per-type atom mass for this atom style');
    if (!(value > 0)) {
      ctx.log(`WARNING: Ignoring invalid mass value ${value} for atom type ${type}`);
      return;
    }
    s.massByType[type] = value;
  }

  /** file2array and array2spline (native init_style). */
  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.table !== 12) throw new StyleError(`pair_modify table is not supported for pair style ${this.name}`);
    const sf = this.setfl;
    if (!sf) throw new StyleError('All pair coeffs are not set');
    // Pair::init (pair.cpp): map_element2type sets I,J only when both types are mapped, so a NULL type of a plain
    // eim pair stops here with native's status lines ("{:6d} {:6d}: is set" / "is not set")
    if (Array.from({ length: this.ntypes }, (_, k) => this.elemOf[k + 1]).some((e) => e < 0)) {
      const lines: string[] = [];
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) {
          const set = this.elemOf[i] >= 0 && this.elemOf[j] >= 0;
          lines.push(`${String(i).padStart(6)} ${String(j).padStart(6)}: is ${set ? '' : 'not '}set`);
        }
      }
      throw new StyleError(`All pair coeffs are not set. Status:\n${lines.join('\n')}`);
    }
    const nt = this.ntypes + 1;
    this.cutforcesq = new Float64Array(nt * nt);
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = 1; j <= this.ntypes; j++) {
        let c: number;
        if (this.elemOf[i] < 0 || this.elemOf[j] < 0) {
          c = sf.cut;
        } else {
          c = sf.cuts[this.elemOf[i] * sf.nel + this.elemOf[j]];
        }
        this.cutforcesq[i * nt + j] = c * c;
      }
    }
    this.splines = {
      rdr: 1.0 / sf.dr,
      Fij: sf.Fij.map((t) => interpolateEim(NR, sf.dr, t)),
      Gij: sf.Gij.map((t) => interpolateEim(NR, sf.dr, t)),
      phiij: sf.phiij.map((t) => interpolateEim(NR, sf.dr, t)),
    };
  }

  /** init_one: sqrt of cutforcesq (native cutmax). */
  initOne(i: number, j: number): number {
    if (this.cutforcesq.length === 0) throw new StyleError('All pair coeffs are not set');
    return Math.sqrt(this.cutforcesq[i * (this.ntypes + 1) + j]);
  }

  compute(pc: PairCompute): void {
    const half = pc.half;
    if (!half) throw new Error(`pair style ${this.name} needs a half neighbor list`);
    const sf = this.setfl, sp = this.splines;
    if (!sf || !sp) throw new StyleError('All pair coeffs are not set');
    const { x, f, type, nlocal, nall } = pc;
    const owner = pc.nb.owner;
    const nel = sf.nel;
    const nt = this.ntypes + 1;
    const el = this.elemOf;
    const cf = this.cutforcesq;
    const rdr = sp.rdr;
    if (this.rho.length !== nall) {
      this.rho = new Float64Array(nall);
      this.fp = new Float64Array(nall);
    }
    const rho = this.rho, fp = this.fp;
    rho.fill(0);
    fp.fill(0);
    const eatom = pc.eatom;

    // pass 1: rho_i = sum_j eta (both directions of each half-list pair, newton on)
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i], ei = el[itype];
      const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const jtype = type[j], ej = el[jtype];
        if (ei < 0 || ej < 0) continue;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cf[itype * nt + jtype]) {
          let p = Math.sqrt(rsq) * rdr + 1.0;
          let m = Math.trunc(p);
          m = Math.min(m, NR - 1);
          p -= m;
          p = Math.min(p, 1.0);
          const o = m * W;
          rho[i] += polyVal(sp.Fij[ei * nel + ej], o, p);
          rho[j] += polyVal(sp.Fij[ej * nel + ei], o, p);
        }
      }
    }
    reverseAdd(owner, nlocal, nall, rho, 1);
    forwardCopy(owner, nlocal, nall, rho, 1);

    // pass 2: fp_i = sum_j rho_j G_ij (both directions)
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i], ei = el[itype];
      const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const jtype = type[j], ej = el[jtype];
        if (ei < 0 || ej < 0) continue;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cf[itype * nt + jtype]) {
          let p = Math.sqrt(rsq) * rdr + 1.0;
          let m = Math.trunc(p);
          m = Math.min(m, NR - 1);
          p -= m;
          p = Math.min(p, 1.0);
          const g = polyVal(sp.Gij[ei * nel + ej], m * W, p);
          fp[i] += rho[j] * g;
          fp[j] += rho[i] * g;
        }
      }
    }
    reverseAdd(owner, nlocal, nall, fp, 1);
    forwardCopy(owner, nlocal, nall, fp, 1);

    // embedding energies 1/2 rho_i fp_i (native: eng_vdwl += phi, eatom[i] += phi)
    for (let i = 0; i < nlocal; i++) {
      const phi = 0.5 * rho[i] * fp[i];
      pc.acc.evdwl += phi;
      if (eatom) eatom[i] += phi;
    }

    // pass 3: pair term and the chain-rule terms of rho and fp
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i], ei = el[itype];
      const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const jtype = type[j], ej = el[jtype];
        if (ei < 0 || ej < 0) continue;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cf[itype * nt + jtype]) {
          const r = Math.sqrt(rsq);
          let p = r * rdr + 1.0;
          let m = Math.trunc(p);
          m = Math.min(m, NR - 1);
          p -= m;
          p = Math.min(p, 1.0);
          const o = m * W;
          const rhoip = polyDer(sp.Fij[ej * nel + ei], o, p);
          const rhojp = polyDer(sp.Fij[ei * nel + ej], o, p);
          const phip = polyDer(sp.phiij[ei * nel + ej], o, p);
          const phi = polyVal(sp.phiij[ei * nel + ej], o, p);
          const coul = polyVal(sp.Gij[ei * nel + ej], o, p);
          const coulp = polyDer(sp.Gij[ei * nel + ej], o, p);
          const q0i = sf.elems[ei].q0, q0j = sf.elems[ej].q0;
          const psip = phip + (rho[i] * rho[j] - q0i * q0j) * coulp + fp[i] * rhojp + fp[j] * rhoip;
          const recip = 1.0 / r;
          const fpair = -psip * recip;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          const evdwl = phi - q0i * q0j * coul;
          tallyPair(pc, i, j, evdwl, fpair, delx, dely, delz);
        }
      }
    }
  }
}
