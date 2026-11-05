/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the Colvars library it bundles (lib/colvars,
 * Colvars version 2026-07-27): lib/colvars/colvarparse.cpp (read_config_line, key_lookup, get_keyval, check_keywords)
 * and colvarvalue.cpp / colvartypes.cpp (reading values).
 *
 * Licence: Colvars (https://github.com/Colvars/colvars; Giacomo Fiorin, Jerome Henin and the Colvars developers) is
 * distributed under the GNU Lesser General Public License, version 3 (LGPL-3.0). This file translates library code and
 * is LGPL-3.0 too (the licence texts are colvars/LICENSE.lgpl-3.0 and colvars/LICENSE.gpl-3.0; see
 * src/engine-gpl/README.md). The rest of this directory, LAMMPS ported code, is GPL-2.0.
 *
 * A Colvars configuration is a list of "keyword value" lines; a value that opens a brace runs to the matching brace,
 * over several lines, and is a block of its own keywords. Keywords are matched without regard to case; "#" starts a
 * comment that runs to the end of the line. Every keyword a block has must be read by the object it configures, or the
 * configuration stops with Colvars's "keyword ... is not supported, or not recognized in this context" error.
 */
import { StyleError } from '../../engine/force/types';

/** A Colvars error, with the library's message. */
export const colvarsError = (msg: string): StyleError => new StyleError(msg.endsWith('\n') ? msg.slice(0, -1) : msg);

interface Entry {
  key: string;
  lower: string;
  value: string;
  block: boolean;
  used: boolean;
}

/** colvarparse::read_config_line: a line keeps the text before its first "#". */
export const stripComments = (text: string): string =>
  text.split('\n').map((l) => { const k = l.indexOf('#'); return k < 0 ? l : l.slice(0, k); }).join('\n');

/** One configuration block: its keywords in order. */
export class Conf {
  private readonly entries: Entry[] = [];

  constructor(text: string) {
    const lines = stripComments(text).split('\n');
    for (let i = 0; i < lines.length; i++) {
      const t = lines[i].trim();
      if (t === '') continue;
      const m = /^([^\s{}]+)\s*([\s\S]*)$/.exec(t);
      if (!m) throw colvarsError(`Error: parse error in the configuration near "${t}".`);
      const key = m[1];
      let rest = m[2];
      const open = rest.indexOf('{');
      if (open < 0) {
        if (rest.includes('}')) throw colvarsError(`Error: parse error: unmatched closing brace after keyword "${key}".`);
        this.entries.push({ key, lower: key.toLowerCase(), value: rest.trim(), block: false, used: false });
        continue;
      }
      // a brace block: read to the matching closing brace, over further lines when needed
      let depth = 0;
      let inner = '';
      let done = false;
      let k = open;
      for (;;) {
        for (; k < rest.length; k++) {
          const c = rest[k];
          if (c === '{') { depth++; if (depth === 1) continue; }
          else if (c === '}') { depth--; if (depth === 0) { done = true; break; } }
          inner += c;
        }
        if (done) break;
        i++;
        if (i >= lines.length) {
          throw colvarsError(`Parse error: reached the end while looking for closing brace; until now the following was parsed: "\n${key} ${rest}\n"`);
        }
        inner += '\n';
        rest = lines[i];
        k = 0;
      }
      this.entries.push({ key, lower: key.toLowerCase(), value: inner, block: true, used: false });
    }
  }

  /** All values of a keyword, in order (key_lookup with a saved position); marks them read. */
  all(key: string): string[] {
    const lower = key.toLowerCase();
    const out: string[] = [];
    for (const e of this.entries) if (e.lower === lower) { e.used = true; out.push(e.value); }
    return out;
  }

  /** The value of a keyword given once, or undefined (get_keyval). */
  get(key: string): string | undefined {
    const v = this.all(key);
    if (v.length > 1) throw colvarsError(`Error: keyword "${key}" must not be used more than once in the same block.`);
    return v[0];
  }

  /** The keywords of the block that are blocks, in order (for the component lookup). */
  blockKeys(): string[] {
    return this.entries.filter((e) => e.block).map((e) => e.key);
  }

  /** colvarparse::check_keywords: every keyword must have been read. */
  check(context: string): void {
    for (const e of this.entries) {
      if (!e.used) throw colvarsError(`Error: keyword "${e.key}" is not supported, or not recognized in this context (${context}).`);
    }
  }
}

/** A real value (colvarparse reads it with operator>>). */
export const parseReal = (s: string, key: string): number => {
  const t = s.trim().split(/\s+/);
  const v = Number(t[0]);
  if (t.length !== 1 || t[0] === '' || !Number.isFinite(v)) throw colvarsError(`Error: in parsing "${key}": could not read a real number from "${s.trim()}".`);
  return v;
};

/** An integer value. */
export const parseInt10 = (s: string, key: string): number => {
  const t = s.trim();
  if (!/^[-+]?\d+$/.test(t)) throw colvarsError(`Error: in parsing "${key}": could not read an integer from "${t}".`);
  return Number(t);
};

/** colvarparse::_get_keyval_scalar_value_ for bool: on/yes/true and off/no/false (any case); no value means on. */
export const parseBool = (s: string, key: string): boolean => {
  const t = s.trim().toLowerCase();
  if (t === '' || t === 'on' || t === 'yes' || t === 'true') return true;
  if (t === 'off' || t === 'no' || t === 'false') return false;
  throw colvarsError(`Error: boolean values only allowed for "${key}": "${s.trim()}".`);
};

/**
 * Splits a list of colvar values: plain numbers separated by spaces, or parenthesized vectors "( a , b , c )" (the
 * stream form of cvm::rvector and cvm::vector1d).
 */
export const splitValues = (s: string): (number | number[])[] => {
  const out: (number | number[])[] = [];
  let k = 0;
  while (k < s.length) {
    const c = s[k];
    if (/\s/.test(c)) { k++; continue; }
    if (c === '(') {
      const end = s.indexOf(')', k);
      if (end < 0) throw colvarsError(`Error: unmatched parenthesis in the value "${s.trim()}".`);
      const parts = s.slice(k + 1, end).split(',').map((p) => p.trim());
      const vals = parts.map((p) => Number(p));
      if (parts.some((p) => p === '') || vals.some((v) => !Number.isFinite(v))) throw colvarsError(`Error: could not read the vector "${s.slice(k, end + 1)}".`);
      out.push(vals);
      k = end + 1;
      continue;
    }
    let e = k;
    while (e < s.length && !/\s/.test(s[e]) && s[e] !== '(') e++;
    const tok = s.slice(k, e);
    const v = Number(tok);
    if (!Number.isFinite(v)) throw colvarsError(`Error: could not read a number from "${tok}".`);
    out.push(v);
    k = e;
  }
  return out;
};
