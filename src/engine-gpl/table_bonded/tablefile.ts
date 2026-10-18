/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/table_file_reader.cpp
 * (TableFileReader::find_section_start), src/potential_file_reader.cpp (PotentialFileReader, error texts),
 * src/text_file_reader.cpp (TextFileReader::next_line, skip_line), src/tokenizer.cpp and tokenizer.h
 * (ValueTokenizer::next_int, next_double and the exception texts), at commit b332f28. Copyright (2003)
 * Sandia Corporation (GNU General Public License). Contributing author of the reader: Richard Berger (Temple U).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Table files of bond_style table, angle_style table and dihedral_style table(/cut) are read the way the native
 * reader reads them: a section starts at a line whose first word is the keyword; next_line() skips blank and
 * comment-only lines and cuts every line at its first '#'; skip_line() skips one physical line whatever it holds.
 * docs.lammps.org/bond_table.html: "The first line of a section is a keyword, followed by a line with the table
 * parameters, followed by N lines of data."
 */
import { StyleError } from '../../engine/force/types';
import type { StyleContext } from '../../engine/force/types';

export type TableKind = 'bond' | 'angle' | 'dihedral';

/** A ValueTokenizer exception: its text is the TokenizerException::what() of the native code. */
export class TokenError extends Error {}

/** Words of a line, split at the default separators " \t\r\n\f" (native Tokenizer). */
export const splitWords = (line: string): string[] => line.split(/[ \t\r\n\f]+/).filter((w) => w !== '');

const FLOAT_RE = /^[+-]?((\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)$/;
const INT_RE = /^[+-]?\d+$/;

/** A sequence of words read with the ValueTokenizer conversions (next_string, next_int, next_double). */
export class WordStream {
  private k = 0;
  constructor(private readonly words: string[]) {}
  hasNext(): boolean { return this.k < this.words.length; }
  nextString(): string {
    if (!this.hasNext()) throw new TokenError('No more tokens');
    return this.words[this.k++];
  }
  nextInt(): number {
    const t = this.nextString();
    if (!INT_RE.test(t)) throw new TokenError(`Not a valid integer number: '${t}'`);
    return Number(t);
  }
  nextDouble(): number {
    const t = this.nextString();
    if (FLOAT_RE.test(t)) return Number(t);
    // stod accepts "inf" and "nan" spellings, and a value out of the double range is an error
    if (/^[+-]?(inf|infinity)$/i.test(t)) return t.startsWith('-') ? -Infinity : Infinity;
    if (/^[+-]?nan$/i.test(t)) return Number.NaN;
    throw new TokenError(`Not a valid floating-point number: '${t}'`);
  }
}

/** Reader over the physical lines of a table file (TextFileReader with ignore_comments set). */
export class TableFileReader {
  private readonly lines: string[];
  private pos = 0;

  constructor(text: string, private readonly filename: string, private readonly kind: TableKind) {
    const parts = text.split('\n');
    // a final newline ends the last line; it does not start another one
    if (parts.length > 0 && parts[parts.length - 1] === '') parts.pop();
    this.lines = parts;
  }

  /** The line whose first word is the keyword, or null at the end of the file (find_section_start). */
  findSectionStart(keyword: string): string | null {
    let line: string | null;
    while ((line = this.nextLine()) !== null) {
      const word = splitWords(line)[0];
      if (word === keyword) return line;
    }
    return null;
  }

  /** The next line with at least one word, cut at its first '#' (TextFileReader::next_line with 0 parameters). */
  nextLine(): string | null {
    while (this.pos < this.lines.length) {
      let line = this.lines[this.pos++];
      const hash = line.indexOf('#');
      if (hash >= 0) line = line.slice(0, hash);
      if (splitWords(line).length > 0) return line;
    }
    return null;
  }

  /** Skips one physical line, blank or not (TextFileReader::skip_line). */
  skipLine(): void {
    if (this.pos >= this.lines.length) throw new StyleError(`Missing line in ${this.kind} table potential file!`);
    this.pos++;
  }
}

/** The file text the notebook has for a table file, with the native open error (PotentialFileReader). */
export const readTableText = (ctx: StyleContext, file: string, kind: TableKind): string => {
  try {
    return ctx.readFile(file);
  } catch {
    throw new StyleError(`cannot open ${kind} table potential file ${file}: No such file or directory`);
  }
};

/** fmt's "{:.8}" (general format, as %.8g) for a double, used in the native error texts. */
export const fmtG = (x: number, precision = 8): string => {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x < 0 ? '-inf' : 'inf';
  if (x === 0) return Object.is(x, -0) ? '-0' : '0';
  const p = precision;
  const sci = x.toExponential(p - 1);
  const ePos = sci.indexOf('e');
  const exp = Number(sci.slice(ePos + 1));
  if (exp < -4 || exp >= p) {
    let mant = sci.slice(0, ePos);
    if (mant.includes('.')) mant = mant.replace(/0+$/, '').replace(/\.$/, '');
    const ae = Math.abs(exp);
    return `${mant}e${exp < 0 ? '-' : '+'}${ae < 10 ? '0' : ''}${ae}`;
  }
  let fixed = x.toFixed(Math.max(0, p - 1 - exp));
  if (fixed.includes('.')) fixed = fixed.replace(/0+$/, '').replace(/\.$/, '');
  return fixed;
};
