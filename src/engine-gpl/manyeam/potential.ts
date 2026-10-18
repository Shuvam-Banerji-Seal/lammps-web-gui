/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANYBODY/pair_eam.cpp
 * (read_setfl, read_fs, file2array_setfl, file2array_fs, array2spline, interpolate), src/MANYBODY/pair_adp.cpp
 * (read_file, file2array, array2spline, interpolate) and the file reader rules of src/text_file_reader.cpp and
 * src/potential_file_reader.cpp. Contributing authors named in those sources: Stephen Foiles (SNL), Murray Daw (SNL),
 * Tim Lau (MIT), Xiaowang Zhou (Sandia), Ray Shan (Sandia, adp). Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * File reading, spline tables and the type-to-table maps for the setfl (eam/cd), Finnis-Sinclair/he (eam/he) and
 * ADP potential files. Arrays are 1-based (index 0 unused) to keep the C++ index arithmetic exact.
 */
import { StyleError, type StyleContext } from '../../engine/force/types';

/** Spline table of one function: row m (0..n) holds seven coefficients at m*7 + k (native spline[m][k]). */
export type Spline = Float64Array;

/** One potential file (setfl, FS/he or ADP), arrays 1-based. */
export interface EamFile {
  readonly name: string;
  readonly nelements: number;
  readonly names: string[];
  readonly mass: number[];
  readonly nrho: number;
  readonly drho: number;
  readonly nr: number;
  readonly dr: number;
  readonly cut: number;
  /** eam/he only: the rhomax of the grid line (sixth value). */
  readonly rhomax: number;
  /** frho[i]: embedding function of element i, length nrho+1. */
  readonly frho: Float64Array[];
  /** setfl: rhor[i] (length nr+1). FS: rhor[i * nelements + j] with the file's rhor[i][j]. */
  readonly rhor: Float64Array[];
  /** z2r[i][j] for j <= i (length nr+1). */
  readonly z2r: Float64Array[][];
  /** ADP only: u2r[i][j] and w2r[i][j] for j <= i. */
  readonly u2r?: Float64Array[][];
  readonly w2r?: Float64Array[][];
}

/**
 * Tables after array2spline(): the splines plus the per-type maps. Native type2frho[i], type2rhor[i][j],
 * type2z2r[i][j] (flattened with nt = ntypes + 1).
 */
export interface EamTables {
  readonly nrho: number;
  readonly nr: number;
  readonly drho: number;
  readonly dr: number;
  readonly rdr: number;
  readonly rdrho: number;
  readonly frho: Spline[];
  readonly rhor: Spline[];
  readonly z2r: Spline[];
  readonly u2r: Spline[];
  readonly w2r: Spline[];
  readonly type2frho: Int32Array;
  readonly type2rhor: Int32Array;
  readonly type2z2r: Int32Array;
  readonly type2u2r: Int32Array;
  readonly type2w2r: Int32Array;
}

/** Native PotentialFileReader/TextFileReader rules: '#' starts a comment, blank lines are skipped, and each array begins on a new line. */
export class PotentialText {
  private readonly lines: string[];
  private pos = 0;

  constructor(text: string, private readonly filetype: string, private readonly filename: string) {
    this.lines = text.split('\n');
  }

  private eof(): StyleError {
    return new StyleError(`Incorrect format in ${this.filetype} file ${this.filename}: unexpected end of file`);
  }

  /** TextFileReader::skip_line(): one raw line, comments kept. */
  skipLine(): void {
    if (this.pos >= this.lines.length) throw this.eof();
    this.pos++;
  }

  /** TextFileReader::next_line(nparams) with comment stripping: concatenate lines until nparams words. */
  nextValues(nparams: number): string[] {
    const words: string[] = [];
    while (words.length < nparams || words.length === 0) {
      if (this.pos >= this.lines.length) {
        if (words.length === 0) throw this.eof();
        throw new StyleError(`Incorrect format in ${this.filetype} file ${this.filename}: ${words.length}/${nparams} parameters`);
      }
      const raw = this.lines[this.pos++];
      const hash = raw.indexOf('#');
      const line = hash >= 0 ? raw.slice(0, hash) : raw;
      for (const w of line.split(/\s+/)) if (w !== '') words.push(w);
    }
    return words;
  }

  /** TextFileReader::next_dvector(list, n): values from successive non-blank lines; the rest of a line is dropped. */
  nextDvector(n: number, out: Float64Array, offset: number): void {
    let i = 0;
    while (i < n) {
      if (this.pos >= this.lines.length) {
        if (i === 0) throw this.eof();
        throw new StyleError(`Incorrect format in ${this.filetype} file ${this.filename}! ${i}/${n} values`);
      }
      const raw = this.lines[this.pos++];
      const hash = raw.indexOf('#');
      const line = hash >= 0 ? raw.slice(0, hash) : raw;
      for (const w of line.split(/\s+/)) {
        if (w === '' || i >= n) continue;
        out[offset + i] = parseReal(w, this.filetype, this.filename);
        i++;
      }
    }
  }
}

/** ValueTokenizer::next_double(): a floating-point number or an error naming the word. */
export const parseReal = (w: string, filetype: string, filename: string): number => {
  const v = Number(w);
  if (w === '' || !Number.isFinite(v)) {
    throw new StyleError(`Expected floating-point number but found '${w}' in ${filetype} file ${filename}`);
  }
  return v;
};

/** ValueTokenizer::next_int(): an integer or an error naming the word. */
const parseInteger = (w: string, filetype: string, filename: string): number => {
  const v = Number(w);
  if (!/^[+-]?\d+$/.test(w) || !Number.isFinite(v)) {
    throw new StyleError(`Expected integer number but found '${w}' in ${filetype} file ${filename}`);
  }
  return v;
};

export { parseInteger };

/**
 * PotentialFileReader constructor: the UNITS: tag of line 1 must match the unit style (no unit conversion here; native
 * converts metal and real files, this engine refuses a mismatch naming both).
 */
export const openPotential = (text: string, filetype: string, filename: string, ctx: StyleContext): PotentialText => {
  const first = text.split('\n')[0] ?? '';
  const words = first.split(/\s+/).filter((w) => w !== '');
  const k = words.indexOf('UNITS:');
  const units = k >= 0 && k + 1 < words.length ? words[k + 1] : '';
  const style = ctx.s?.units.style;
  if (units !== '' && style !== undefined && units !== style) {
    throw new StyleError(`${filetype} file ${filename} requires ${units} units but ${style} units are in use (unit conversion is not supported)`);
  }
  return new PotentialText(text, filetype, filename);
};

const readGrid = (t: PotentialText, filename: string, he: boolean): { nrho: number; drho: number; nr: number; dr: number; cut: number; rhomax: number } => {
  const v = t.nextValues(he ? 6 : 5);
  const nrho = parseInteger(v[0], 'eam', filename);
  const drho = parseReal(v[1], 'eam', filename);
  const nr = parseInteger(v[2], 'eam', filename);
  const dr = parseReal(v[3], 'eam', filename);
  const cut = parseReal(v[4], 'eam', filename);
  const rhomax = he ? parseReal(v[5], 'eam', filename) : 0;
  if (nrho <= 0 || nr <= 0 || !(dr > 0)) throw new StyleError('Invalid EAM potential file');
  return { nrho, drho, nr, dr, cut, rhomax };
};

const readNames = (t: PotentialText, filename: string, filetype: string): { nelements: number; names: string[] } => {
  const v = t.nextValues(1);
  const nelements = parseInteger(v[0], filetype, filename);
  if (v.length !== nelements + 1) throw new StyleError(`Incorrect element names in ${filetype} potential file`);
  return { nelements, names: v.slice(1) };
};

/** Element line: "I mass ..." (ignored int, then the mass). */
const readMass = (t: PotentialText, filename: string): number => {
  const v = t.nextValues(2);
  parseInteger(v[0], 'eam', filename);
  return parseReal(v[1], 'eam', filename);
};

const array = (n: number): Float64Array => new Float64Array(n + 1);

/** read_setfl(): DYNAMO multi-element file (eam/alloy layout, used by eam/cd). */
export const readSetfl = (text: string, filename: string, ctx: StyleContext): EamFile => {
  const t = openPotential(text, 'eam/alloy', filename, ctx);
  t.skipLine();
  t.skipLine();
  t.skipLine();
  const { nelements, names } = readNames(t, filename, 'EAM');
  const g = readGrid(t, filename, false);
  const mass: number[] = [];
  const frho: Float64Array[] = [];
  const rhor: Float64Array[] = [];
  for (let i = 0; i < nelements; i++) {
    mass.push(readMass(t, filename));
    const f = array(g.nrho);
    t.nextDvector(g.nrho, f, 1);
    frho.push(f);
    const r = array(g.nr);
    t.nextDvector(g.nr, r, 1);
    rhor.push(r);
  }
  const z2r: Float64Array[][] = [];
  for (let i = 0; i < nelements; i++) {
    z2r.push([]);
    for (let j = 0; j <= i; j++) {
      const z = array(g.nr);
      t.nextDvector(g.nr, z, 1);
      z2r[i].push(z);
    }
  }
  return { name: filename, nelements, names, mass, ...g, frho, rhor, z2r };
};

/** read_fs(): Finnis-Sinclair layout with a density per element pair; eam/he adds rhomax to the grid line. */
export const readFs = (text: string, filename: string, ctx: StyleContext, he: boolean): EamFile => {
  const filetype = he ? 'eam/he' : 'eam/fs';
  const t = openPotential(text, filetype, filename, ctx);
  t.skipLine();
  t.skipLine();
  t.skipLine();
  const { nelements, names } = readNames(t, filename, 'EAM');
  const g = readGrid(t, filename, he);
  const mass: number[] = [];
  const frho: Float64Array[] = [];
  const rhor: Float64Array[] = [];
  for (let i = 0; i < nelements; i++) {
    mass.push(readMass(t, filename));
    const f = array(g.nrho);
    t.nextDvector(g.nrho, f, 1);
    frho.push(f);
    for (let j = 0; j < nelements; j++) {
      const r = array(g.nr);
      t.nextDvector(g.nr, r, 1);
      rhor.push(r);
    }
  }
  const z2r: Float64Array[][] = [];
  for (let i = 0; i < nelements; i++) {
    z2r.push([]);
    for (let j = 0; j <= i; j++) {
      const z = array(g.nr);
      t.nextDvector(g.nr, z, 1);
      z2r[i].push(z);
    }
  }
  return { name: filename, nelements, names, mass, ...g, frho, rhor, z2r };
};

/** read_file() of pair_adp.cpp: setfl layout with u2r and w2r arrays after the z2r block. */
export const readAdp = (text: string, filename: string, ctx: StyleContext): EamFile => {
  const t = openPotential(text, 'adp', filename, ctx);
  t.skipLine();
  t.skipLine();
  t.skipLine();
  const { nelements, names } = readNames(t, filename, 'ADP');
  const g = readGrid(t, filename, false);
  const mass: number[] = [];
  const frho: Float64Array[] = [];
  const rhor: Float64Array[] = [];
  for (let i = 0; i < nelements; i++) {
    mass.push(readMass(t, filename));
    const f = array(g.nrho);
    t.nextDvector(g.nrho, f, 1);
    frho.push(f);
    const r = array(g.nr);
    t.nextDvector(g.nr, r, 1);
    rhor.push(r);
  }
  const block = (): Float64Array[][] => {
    const out: Float64Array[][] = [];
    for (let i = 0; i < nelements; i++) {
      out.push([]);
      for (let j = 0; j <= i; j++) {
        const a = array(g.nr);
        t.nextDvector(g.nr, a, 1);
        out[i].push(a);
      }
    }
    return out;
  };
  const z2r = block();
  const u2r = block();
  const w2r = block();
  return { name: filename, nelements, names, mass, ...g, frho, rhor, z2r, u2r, w2r };
};

/**
 * interpolate() of pair_eam.cpp (identical in pair_adp.cpp): the cubic spline coefficients of a table f[1..n] with
 * spacing delta, seven per row, rows 1..n (row 0 unused).
 */
export const interpolate = (n: number, delta: number, f: Float64Array): Spline => {
  const s = new Float64Array((n + 1) * 7);
  const at = (m: number, k: number): number => m * 7 + k;
  for (let m = 1; m <= n; m++) s[at(m, 6)] = f[m];
  s[at(1, 5)] = s[at(2, 6)] - s[at(1, 6)];
  s[at(2, 5)] = 0.5 * (s[at(3, 6)] - s[at(1, 6)]);
  s[at(n - 1, 5)] = 0.5 * (s[at(n, 6)] - s[at(n - 2, 6)]);
  s[at(n, 5)] = s[at(n, 6)] - s[at(n - 1, 6)];
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
    s[at(m, 1)] = 2.0 * s[at(m, 4)] / delta;
    s[at(m, 0)] = 3.0 * s[at(m, 3)] / delta;
  }
  return s;
};

/** Index of the (i, j) pair in the packed j <= i order of setfl z2r arrays, with the type-map rule of file2array_setfl. */
const packedIndex = (irow: number, icol: number): number => {
  let a = irow, b = icol;
  if (a < b) {
    a = icol;
    b = irow;
  }
  let n = 0;
  for (let m = 0; m < a; m++) n += m + 1;
  return n + b;
};

/**
 * file2array_setfl / file2array_fs / file2array (ADP) followed by array2spline(): the spline tables and the per-type
 * maps. map[t] is the element of type t (1-based types, -1 for NULL). fs selects the Finnis-Sinclair density layout.
 */
export const buildTables = (file: EamFile, map: Int32Array, ntypes: number, fs: boolean): EamTables => {
  const nel = file.nelements;
  const nt = ntypes + 1;
  const nrho = file.nrho, nr = file.nr, drho = file.drho, dr = file.dr;
  const nfrho = nel + 1;
  const frhoArr: Float64Array[] = [];
  for (let i = 0; i < nel; i++) frhoArr.push(file.frho[i]);
  frhoArr.push(array(nrho));
  const type2frho = new Int32Array(nt);
  for (let i = 1; i <= ntypes; i++) type2frho[i] = map[i] >= 0 ? map[i] : nfrho - 1;

  const nrhor = fs ? nel * nel : nel;
  const rhorArr: Float64Array[] = [];
  for (let n = 0; n < nrhor; n++) rhorArr.push(file.rhor[n]);
  const type2rhor = new Int32Array(nt * nt);
  for (let i = 1; i <= ntypes; i++) {
    for (let j = 1; j <= ntypes; j++) {
      type2rhor[i * nt + j] = fs ? map[i] * nel + map[j] : map[i];
    }
  }

  const z2rArr: Float64Array[] = [];
  const u2rArr: Float64Array[] = [];
  const w2rArr: Float64Array[] = [];
  for (let i = 0; i < nel; i++) {
    for (let j = 0; j <= i; j++) {
      z2rArr.push(file.z2r[i][j]);
      if (file.u2r) u2rArr.push(file.u2r[i][j]);
      if (file.w2r) w2rArr.push(file.w2r[i][j]);
    }
  }
  const type2z2r = new Int32Array(nt * nt);
  const type2u2r = new Int32Array(nt * nt);
  const type2w2r = new Int32Array(nt * nt);
  for (let i = 1; i <= ntypes; i++) {
    for (let j = 1; j <= ntypes; j++) {
      const irow = map[i], icol = map[j];
      if (irow === -1 || icol === -1) continue;
      const n = packedIndex(irow, icol);
      type2z2r[i * nt + j] = n;
      type2u2r[i * nt + j] = n;
      type2w2r[i * nt + j] = n;
    }
  }

  return {
    nrho, nr, drho, dr,
    rdr: 1.0 / dr,
    rdrho: 1.0 / drho,
    frho: frhoArr.map((f) => interpolate(nrho, drho, f)),
    rhor: rhorArr.map((f) => interpolate(nr, dr, f)),
    z2r: z2rArr.map((f) => interpolate(nr, dr, f)),
    u2r: u2rArr.map((f) => interpolate(nr, dr, f)),
    w2r: w2rArr.map((f) => interpolate(nr, dr, f)),
    type2frho, type2rhor, type2z2r, type2u2r, type2w2r,
  };
};

/**
 * Reads the h(x) polynomial of eam/cd from the LAST line of the file (pair_eam_cd.cpp read_h_coeff): the last
 * MAXLINE = 1024 bytes are searched, the last non-empty line is "degree c0 c1 ... c_degree".
 */
export const readHCoeff = (text: string, filename: string): Float64Array => {
  const MAXLINE = 1024;
  if (text.length < MAXLINE) throw new StyleError(`Failure to seek to end-of-file for reading h(x) coeffs of ${filename}`);
  const tail = text.slice(text.length - MAXLINE);
  const parts = tail.split('\n').filter((p) => p.trim() !== '');
  const lastline = parts.length ? parts[parts.length - 1] : '';
  const words = lastline.split(/\s+/).filter((w) => w !== '');
  const degree = words.length ? parseInteger(words[0], 'EAMCD', filename) : -1;
  const nhcoeff = degree + 1;
  if (words.length !== nhcoeff + 1 || nhcoeff < 1) {
    throw new StyleError('Failed to read h(x) function coefficients in EAM file.');
  }
  const h = new Float64Array(nhcoeff);
  for (let i = 0; i < nhcoeff; i++) h[i] = parseReal(words[i + 1], 'EAMCD', filename);
  return h;
};
