/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/rx_table_file_reader.cpp and
 * rx_table_file_reader.h (RxTableFileReader: the keyword section of a table file read by fix eos/table/rx and pair
 * table/rx), and the line reading they share with the potential files (src/text_file_reader.cpp: TextFileReader::next_line
 * concatenates lines until nparams words are read; src/table_file_reader.cpp: find_section_start matches the first word
 * of a line). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author (from rx_table_file_reader.cpp):
 * James Larentzos (U.S. Army Research Laboratory).
 *
 * Input rules (rx_table_file_reader.cpp): the keyword line starts a section; the next line is the parameter line, in which
 * "N" (the first one) is followed by the number of table entries; the other words are kept for the caller. The entries
 * follow one per line, and the caller reads the words it needs from each.
 */
import { StyleError } from '../../engine/force/types';

/** Lines of a text file as TextFileReader reads them: '#' starts a comment, blank lines are skipped. */
export class TextLines {
  private pos = 0;
  private readonly raw: string[];

  constructor(text: string, private readonly fileName: string) {
    this.raw = text.split('\n');
  }

  /**
   * TextFileReader::next_line(nparams): the words of the next entry. With nparams > 0 lines are concatenated until
   * nparams words are read; an entry that ends the file short of them is an error. Null at the end of the file.
   */
  next(nparams = 0): string[] | null {
    let words: string[] = [];
    for (;;) {
      if (this.pos >= this.raw.length) {
        if (words.length > 0 && words.length < nparams) {
          throw new StyleError(`Incorrect format in ${this.fileName} file! ${words.length}/${nparams} parameters`);
        }
        return words.length > 0 ? words : null;
      }
      const line = this.raw[this.pos++];
      const h = line.indexOf('#');
      const t = (h >= 0 ? line.slice(0, h) : line).trim();
      if (t) words = words.concat(t.split(/\s+/));
      if (words.length > 0 && words.length >= nparams) return words;
    }
  }

  /** TableFileReader::find_section_start: the first line whose first word is keyword (its words), or null. */
  findSection(keyword: string): string[] | null {
    for (let w = this.next(); w; w = this.next()) {
      if (w[0] === keyword) return w;
    }
    return null;
  }
}

/** The words of one line, read in order as ValueTokenizer reads them (a number must parse whole). */
export class Tokens {
  private k = 0;

  constructor(private readonly words: string[], private readonly line: string) {}

  hasNext(): boolean {
    return this.k < this.words.length;
  }

  next(): string {
    if (this.k >= this.words.length) throw new StyleError(`Missing value in line '${this.line}'`);
    return this.words[this.k++];
  }

  double(): number {
    const w = this.next();
    const v = Number(w);
    if (w === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating-point number instead of '${w}' in line '${this.line}'`);
    return v;
  }

  int(): number {
    const w = this.next();
    const v = Number(w);
    if (!/^[-+]?\d+$/.test(w)) throw new StyleError(`Expected integer number instead of '${w}' in line '${this.line}'`);
    return v;
  }
}

/** A keyword section of a table file: N entries, the parameter words after N, then the entries. */
export class RxTableReader {
  /** Number of table entries (the number after the first N on the parameter line). */
  readonly N: number;
  /** The parameter words other than N and its value, in order (rx_table_file_reader.cpp param_tokens). */
  readonly params: string[] = [];
  private readonly lines: TextLines;
  private readonly fileName: string;
  private read = false;

  constructor(text: string, keyword: string, fileName: string, type: string) {
    this.fileName = fileName;
    this.lines = new TextLines(text, fileName);
    if (!this.lines.findSection(keyword)) throw new StyleError(`Did not find keyword '${keyword}' in '${fileName}'`);
    const words = this.lines.next();
    if (!words) throw new StyleError(`Reading of table parameter line failed. File: ${fileName} (${type})`);
    const line = words.join(' ');
    let n = NaN;
    for (let k = 0; k < words.length; k++) {
      if (words[k] === 'N' && Number.isNaN(n)) {
        const v = words[k + 1];
        if (v === undefined || !/^[-+]?\d+$/.test(v)) throw new StyleError(`Expected integer number instead of '${v ?? ''}' in line '${line}'`);
        n = Number(v);
        k++;
      } else {
        this.params.push(words[k]);
      }
    }
    if (Number.isNaN(n)) throw new StyleError(`Failed to read value for N from line in table parameter file ${fileName}. Full line: ${line}`);
    this.N = n;
  }

  /** Calls fn(i, tokens) for each of the N entry lines, in order (rx_table_file_reader.h read_in_table_data). */
  readTable(fn: (i: number, values: Tokens) => void): void {
    if (this.read) throw new StyleError('Attempted to read table more than once.');
    for (let i = 0; i < this.N; i++) {
      const words = this.lines.next();
      if (!words) throw new StyleError(`Unexpected end of table in '${this.fileName}': ${this.N} entries expected`);
      const tok = new Tokens(words, words.join(' '));
      fn(i, tok);
      if (tok.hasNext()) throw new StyleError(`Misplaced characters at end of line in file ${this.fileName}. Full line: ${words.join(' ')}`);
    }
    this.read = true;
  }
}
