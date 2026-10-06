/*
 * Splitting LAMMPS input text into commands and words, following
 * docs.lammps.org/Commands_parse.html:
 *
 *  1. "If the last printable character on the line is a “&” character, the
 *     command is assumed to continue on the next line. The next line is
 *     concatenated to the previous line by removing the “&” character and
 *     line break."
 *  2. "All characters from the first “#” character onward are treated as
 *     comment and discarded. ... a comment after a trailing “&” character
 *     will prevent the command from continuing on the next line. Also note
 *     that for multi-line commands a single leading “#” will comment out the
 *     entire command."
 *  3. "$" substitution: "${myTemp} and $x refer to variables named “myTemp”
 *     and “x”"; "$(...)" is an immediate equal-style formula, optionally
 *     followed by ":" and a C-style format ("If a format string is not
 *     specified, a high-precision %.20g is used as the default format").
 *     "neither the curly-bracket or immediate form of variables can contain
 *     nested $ characters".
 *  4/5. Words are separated by white-space; the first word is the command.
 *  6. Single, double or triple quotes make one argument; "the single,
 *     double, or triple quotes are removed"; triple quotes keep line breaks
 *     and "“&” characters are not needed and do not function as line
 *     continuation character"; "A “#” or “$” character that is between
 *     quotes will not be treated as a comment indicator in 2 or substituted
 *     for as a variable in 3."
 */

export interface RawCommand {
  /** 1-based line where the command starts. */
  line: number;
  /** Command text with comments removed and continuations joined. */
  text: string;
}

const restOfLineBlank = (s: string, from: number): boolean => {
  for (let k = from; k < s.length && s[k] !== '\n'; k++) {
    if (s[k] !== ' ' && s[k] !== '\t' && s[k] !== '\r') return false;
  }
  return true;
};

const skipToNextLine = (s: string, from: number): number => {
  let k = from;
  while (k < s.length && s[k] !== '\n') k++;
  return k;   // index of '\n' (or end)
};

/** Splits input text into commands (quotes kept; comments and '&' handled). */
export const splitCommands = (text: string, firstLine = 1): RawCommand[] => {
  const out: RawCommand[] = [];
  let buf = '';
  let line = firstLine;
  let start = firstLine;
  let mode: 'normal' | 'sq' | 'dq' | 'tq' | 'comment' = 'normal';
  const flush = () => {
    if (buf.trim()) out.push({ line: start, text: buf.trim() });
    buf = '';
  };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '\n') {
      line++;
      if (mode === 'tq') { buf += c; continue; }
      mode = 'normal';
      flush();
      start = line;
      continue;
    }
    if (mode === 'tq') {
      if (text.startsWith('"""', i)) { buf += '"""'; i += 2; mode = 'normal'; } else buf += c;
      continue;
    }
    if (c === '&' && restOfLineBlank(text, i + 1)) {
      // line continuation, also inside a comment or a single/double quote
      i = skipToNextLine(text, i + 1);
      if (i < text.length) line++;
      continue;
    }
    if (mode === 'comment') continue;
    if (mode === 'sq' || mode === 'dq') {
      buf += c;
      if ((mode === 'sq' && c === "'") || (mode === 'dq' && c === '"')) mode = 'normal';
      continue;
    }
    // normal
    if (c === '#') { mode = 'comment'; continue; }
    if (text.startsWith('"""', i)) { buf += '"""'; i += 2; mode = 'tq'; continue; }
    if (c === '"') { buf += c; mode = 'dq'; continue; }
    if (c === "'") { buf += c; mode = 'sq'; continue; }
    buf += c;
  }
  flush();
  return out;
};

/** Calls `fn` on each unquoted stretch of `text`, leaving quoted parts unchanged. */
export const mapUnquoted = (text: string, fn: (part: string) => string): string => {
  let out = '';
  let i = 0;
  while (i < text.length) {
    let q: string | null = null;
    let j = i;
    for (; j < text.length; j++) {
      if (text.startsWith('"""', j)) { q = '"""'; break; }
      if (text[j] === '"' || text[j] === "'") { q = text[j]; break; }
    }
    out += fn(text.slice(i, j));
    if (q === null) break;
    const end = text.indexOf(q, j + q.length);
    const stop = end < 0 ? text.length : end + q.length;
    out += text.slice(j, stop);
    i = stop;
  }
  return out;
};

/**
 * Replaces $x, ${name} and $(formula[:fmt]) in `text`. `lookup` returns the
 * text of a named variable; `immediate` evaluates a formula. Throws on an
 * unknown variable or an unterminated reference.
 */
export const substituteVariables = (
  text: string,
  lookup: (name: string) => string,
  immediate: (formula: string, format?: string) => string,
): string => {
  let out = '';
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c !== '$') { out += c; continue; }
    const next = text[i + 1];
    if (next === undefined) throw new Error("a '$' at the end of the line has no variable name");
    if (next === '{') {
      const end = text.indexOf('}', i + 2);
      if (end < 0) throw new Error("unterminated '${' variable reference");
      const name = text.slice(i + 2, end);
      if (name.includes('$')) throw new Error('variable references cannot be nested');
      out += lookup(name);
      i = end;
    } else if (next === '(') {
      let depth = 0;
      let end = -1;
      for (let k = i + 1; k < text.length; k++) {
        if (text[k] === '(') depth++;
        else if (text[k] === ')') { depth--; if (depth === 0) { end = k; break; } }
      }
      if (end < 0) throw new Error("unterminated '$(' immediate variable");
      const body = text.slice(i + 2, end);
      if (body.includes('$')) throw new Error('immediate variables cannot contain $ references');
      // optional trailing :%fmt — a colon followed by a % format
      const colon = body.lastIndexOf(':');
      if (colon >= 0 && /^%/.test(body.slice(colon + 1).trim())) {
        out += immediate(body.slice(0, colon), body.slice(colon + 1).trim());
      } else {
        out += immediate(body);
      }
      i = end;
    } else {
      out += lookup(next);
      i += 1;
    }
  }
  return out;
};

/** Splits a substituted command into words, removing the quotes of quoted words. */
export const tokenize = (text: string): string[] => {
  const words: string[] = [];
  let i = 0;
  while (i < text.length) {
    while (i < text.length && /\s/.test(text[i])) i++;
    if (i >= text.length) break;
    let word = '';
    let quoted = false;
    while (i < text.length && !/\s/.test(text[i])) {
      const q = text.startsWith('"""', i) ? '"""' : text[i] === '"' || text[i] === "'" ? text[i] : null;
      if (q) {
        const end = text.indexOf(q, i + q.length);
        if (end < 0) throw new Error(`unbalanced ${q} quote`);
        word += text.slice(i + q.length, end);
        i = end + q.length;
        quoted = true;
      } else {
        word += text[i++];
      }
    }
    if (word || quoted) words.push(word);
  }
  return words;
};

/**
 * C-style number formatting for "$(expr:%fmt)" and print: %[flags][width]
 * [.precision](f|e|g|d|i). Anything else is an error — a format must
 * consume one floating-point value (Commands_parse.html).
 */
export const formatNumber = (value: number, format: string): string => {
  const m = /^%([-+ 0#]*)(\d*)(?:\.(\d+))?([feEgGdi])$/.exec(format.trim());
  if (!m) throw new Error(`unsupported number format '${format}'`);
  const [, flags, width, precStr, conv] = m;
  const prec = precStr === undefined ? 6 : Number(precStr);
  let s: string;
  if (!Number.isFinite(value)) s = Number.isNaN(value) ? 'nan' : value > 0 ? 'inf' : '-inf';
  else if (conv === 'f') s = value.toFixed(prec);
  else if (conv === 'e' || conv === 'E') s = cExp(value, prec, conv === 'E');
  else if (conv === 'd' || conv === 'i') s = Math.trunc(value).toString();
  else s = cGeneral(value, prec === 0 ? 1 : prec, conv === 'G', flags.includes('#'));
  if (flags.includes('+') && value >= 0) s = '+' + s;
  else if (flags.includes(' ') && value >= 0) s = ' ' + s;
  const w = Number(width || 0);
  if (s.length < w) {
    if (flags.includes('-')) s = s.padEnd(w);
    else if (flags.includes('0') && Number.isFinite(value)) {
      const sign = /^[+\- ]/.test(s) ? s[0] : '';
      s = sign + s.slice(sign.length).padStart(w - sign.length, '0');
    } else s = s.padStart(w);
  }
  return s;
};

/** C's %e: mantissa with `prec` decimals and an at-least-two-digit exponent. */
const cExp = (v: number, prec: number, upper: boolean): string => {
  const [mant, exp] = v.toExponential(prec).split('e');
  const e = Number(exp);
  const s = `${mant}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
  return upper ? s.toUpperCase() : s;
};

/** C's %g: shortest of %e/%f at `prec` significant digits, trailing zeros removed. */
const cGeneral = (v: number, prec: number, upper: boolean, keepZeros: boolean): string => {
  if (v === 0) return '0';
  const exp = Math.floor(Math.log10(Math.abs(Number(v.toPrecision(prec)))));
  let s: string;
  if (exp < -4 || exp >= prec) {
    s = cExp(v, prec - 1, upper);
    if (!keepZeros) s = s.replace(/\.?0+(e|E)/, '$1');
  } else {
    s = v.toFixed(Math.max(0, prec - 1 - exp));
    if (!keepZeros && s.includes('.')) s = s.replace(/\.?0+$/, '');
  }
  return s;
};
