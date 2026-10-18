/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the shared input and bookkeeping of
 * the MANYBODY pair styles gw, gw/zbl, tersoff/table and polymorphic (src/pair.cpp map_element2type, Pair::ev_tally,
 * Pair::v_tally3; src/potential_file_reader.cpp and src/text_file_reader.cpp next_line; src/tokenizer.cpp messages;
 * src/utils.cpp get_potential_units and get_conversion_factor). Contributing authors of those sources: Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';

/** utils::get_conversion_factor(ENERGY, METAL2REAL) at b332f28 (src/utils.cpp). */
export const METAL2REAL_ENERGY = 23.060549;
export const REAL2METAL_ENERGY = 1.0 / 23.060549;

/** Words of a number as ValueTokenizer::next_double accepts them (utils::is_double). */
const NUM_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/** ValueTokenizer::next_double: a token that is not a number raises the native InvalidFloatException text. */
export const nextDouble = (word: string | undefined): number => {
  if (word === undefined) throw new StyleError('No more tokens');
  if (!NUM_RE.test(word)) throw new StyleError(`Not a valid floating-point number: '${word}'`);
  return Number(word);
};

/**
 * PotentialFileReader::next_line(nparams) over the text of a potential file: comment-stripped lines are joined until
 * nparams words are read (blank lines skipped); null at the end of the file. A last entry with too few words is an
 * error "Incorrect format in <filetype> file! n/N parameters" (src/text_file_reader.cpp).
 */
export class PotentialEntryReader {
  private pos = 0;
  constructor(private readonly lines: readonly string[], private readonly filetype: string) {}

  next(nparams: number): string[] | null {
    let words: string[] = [];
    while (words.length === 0 || words.length < nparams) {
      if (this.pos >= this.lines.length) {
        if (words.length > 0 && words.length < nparams) {
          throw new StyleError(`Incorrect format in ${this.filetype} file! ${words.length}/${nparams} parameters`);
        }
        return null;
      }
      const raw = this.lines[this.pos++];
      const h = raw.indexOf('#');
      const t = (h >= 0 ? raw.slice(0, h) : raw).trim();
      if (t) words = words.concat(t.split(/\s+/));
    }
    return words;
  }
}

/**
 * The units tag of a potential file (utils::get_potential_units reads "UNITS: x" on the first line) and the
 * conversion the file needs in this unit style (PotentialFileReader::open_potential). Returns the factor applied to
 * the energy parameters (1 when none). Only metal and real convert, as the native styles allow; with convert false (a
 * reader without unit_convert, as polymorphic) any mismatch is an error.
 */
export const potentialUnitFactor = (
  text: string, filetype: string, fileName: string, ctx: StyleContext, convert = true,
): number => {
  const first = text.split(/\r?\n/)[0] ?? '';
  const words = first.split(/\s+/);
  let units: string | null = null;
  for (let k = 0; k < words.length; k++) {
    if (words[k] === 'UNITS:' && k + 1 < words.length) { units = words[k + 1]; break; }
  }
  const style = ctx.s?.units.style ?? 'lj';
  if (units === null || units === style) return 1;
  if (convert && units === 'metal' && style === 'real') {
    ctx.log(`WARNING: Converting ${filetype} in ${units} units to ${style} units`);
    return METAL2REAL_ENERGY;
  }
  if (convert && units === 'real' && style === 'metal') {
    ctx.log(`WARNING: Converting ${filetype} in ${units} units to ${style} units`);
    return REAL2METAL_ENERGY;
  }
  throw new StyleError(`${filetype} file ${fileName} requires ${units} units but ${style} units are in use`);
};

/** Pair::map_element2type: the element of each atom type (-1 for NULL) and the unique element names in order. */
export interface ElementMap {
  elements: string[];
  /** map[t] for t = 1..ntypes (map[0] unused, -1). */
  map: Int32Array;
}

export const mapElementsToTypes = (names: readonly string[], ntypes: number): ElementMap => {
  if (names.length !== ntypes) {
    throw new StyleError('Number of element to type mappings does not match number of atom types');
  }
  const elements: string[] = [];
  const map = new Int32Array(ntypes + 1).fill(-1);
  for (let t = 1; t <= ntypes; t++) {
    const entry = names[t - 1];
    if (entry === 'NULL') { map[t] = -1; continue; }
    let j = elements.indexOf(entry);
    if (j < 0) { j = elements.length; elements.push(entry); }
    map[t] = j;
  }
  return { elements, map };
};

/** The common pair_coeff checks: "* * file elem1 ... elemN" for a many-body style (one_coeff). */
export const checkOneCoeffArgs = (args: readonly string[], ntypes: number, styleName: string): void => {
  if (ntypes === 0) throw new StyleError('Pair_coeff command before simulation box is defined');
  if (args.length < 3) throw new StyleError('Incorrect args for pair coefficients');
  if (args[0] !== '*' || args[1] !== '*') {
    throw new StyleError(`Pair_coeff must start with * * for pair style ${styleName}`);
  }
};

/**
 * Owner atom ID of an owned or ghost atom: the image of an atom carries the atom's own tag (the tag comparisons of
 * the native half-list rule use it).
 */
export const tagOf = (pc: PairCompute, j: number): number => pc.s.id[pc.nb.owner[j]];

/**
 * Pair::ev_tally for the energy and the per-atom parts (newton on): the global energy is summed; the per-atom energy
 * and virial halves go to i and j. The global virial is the force-field dot product (virialFdotr).
 */
export const tallyPair = (
  pc: PairCompute, i: number, j: number, evdwl: number, fpair: number, dx: number, dy: number, dz: number,
): void => {
  pc.acc.evdwl += evdwl;
  if (pc.eatom) {
    pc.eatom[i] += 0.5 * evdwl;
    pc.eatom[j] += 0.5 * evdwl;
  }
  if (pc.vatom) {
    const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
    const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
    const va = pc.vatom;
    for (const k of [i, j]) {
      va[6 * k] += 0.5 * v0; va[6 * k + 1] += 0.5 * v1; va[6 * k + 2] += 0.5 * v2;
      va[6 * k + 3] += 0.5 * v3; va[6 * k + 4] += 0.5 * v4; va[6 * k + 5] += 0.5 * v5;
    }
  }
};

/** Pair::v_tally3 per-atom part: the three-body virial split equally over i, j and k (THIRD). */
export const tallyTriplet = (
  pc: PairCompute, i: number, j: number, k: number,
  fi: Float64Array, fj: Float64Array, drik: Float64Array, drjk: Float64Array,
): void => {
  if (!pc.vatom) return;
  const v = [
    drik[0] * fi[0] + drjk[0] * fj[0],
    drik[1] * fi[1] + drjk[1] * fj[1],
    drik[2] * fi[2] + drjk[2] * fj[2],
    drik[0] * fi[1] + drjk[0] * fj[1],
    drik[0] * fi[2] + drjk[0] * fj[2],
    drik[1] * fi[2] + drjk[1] * fj[2],
  ];
  const va = pc.vatom;
  for (const q of [i, j, k]) {
    for (let c = 0; c < 6; c++) va[6 * q + c] += v[c] * (1 / 3);
  }
};
