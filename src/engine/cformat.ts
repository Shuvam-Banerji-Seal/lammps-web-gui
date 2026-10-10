import { StyleError } from './force/types';

/*
 * C-style number formats for print / immediate $(expr:format), variable
 * format, thermo_modify format and dump_modify format.
 *
 * Per variable.rst (style format, versionchanged 2Sep2026), a conversion in the
 * format must match a floating-point value, and the string may contain literal
 * text and any C flag or modifier (for example +12.6e or <%.4f>).
 * Per thermo_modify.rst and dump_modify.rst (versionchanged 2Sep2026), format
 * strings are checked when they are set up and a format that does not match the
 * kind of value it is applied to is an error (for example %d on a floating-point
 * value).
 *
 * Accepted here: literal text, the percent-percent escape, flags (- + space # 0),
 * a width and a precision (no star and no dollar positions), length modifiers
 * h, l and ll (C ignores them for double and the integer types), and the
 * conversions d and i (integer values) and f, F, e, E, g, G (floating-point
 * values). A format that holds a single conversion formats one value; zero
 * conversions is literal text. Measured with native LAMMPS (black box): a
 * conversion of the wrong kind is an error, and so is a format with two
 * conversions for one value, while a format with no conversion is accepted and
 * printed as is. Rounding of the last digit follows C: an exact tie rounds to
 * the even digit (measured: %.0f of 2.5 prints 2, %.1e of 1.25 prints 1.2e+00).
 *
 * The C printf conversions are specified by the C library, not by LAMMPS; this
 * file contains no LAMMPS source code.
 */

export type ValueKind = 'int' | 'float';

export interface CSpec {
  flags: string;
  width: number;
  prec: number | null;
  conv: string;
  text: string;
}

export type CPiece = { text: string } | { spec: CSpec };

const INT_CONV = new Set(['d', 'i']);
const FLOAT_CONV = new Set(['f', 'F', 'e', 'E', 'g', 'G']);

/** Splits a C format string into literal text and conversion specs. Unsupported syntax throws StyleError. */
export const parseCFormat = (format: string, what = 'Invalid format'): CPiece[] => {
  const out: CPiece[] = [];
  let lit = '';
  let i = 0;
  while (i < format.length) {
    const c = format[i];
    if (c !== '%') { lit += c; i++; continue; }
    if (format[i + 1] === '%') { lit += '%'; i += 2; continue; }
    const start = i;
    i++;
    let flags = '';
    while (i < format.length && '-+ #0'.includes(format[i])) flags += format[i++];
    if (format[i] === '*') throw new StyleError(`${what}: variable field width '*' is not supported in '${format}'`);
    let w = '';
    while (i < format.length && /\d/.test(format[i])) w += format[i++];
    let prec: number | null = null;
    if (format[i] === '.') {
      i++;
      if (format[i] === '*') throw new StyleError(`${what}: variable precision '*' is not supported in '${format}'`);
      let p = '';
      while (i < format.length && /\d/.test(format[i])) p += format[i++];
      prec = p === '' ? 0 : Number(p);
    }
    if (format[i] === 'L') throw new StyleError(`${what}: long double conversions (%L) are not supported in '${format}'`);
    if (format[i] === 'h' || format[i] === 'l') {
      i++;
      if (format[i] === format[i - 1]) i++;
    }
    if (i >= format.length) throw new StyleError(`${what}: incomplete conversion '${format.slice(start)}' in '${format}'`);
    const conv = format[i++];
    const text = format.slice(start, i);
    if (!INT_CONV.has(conv) && !FLOAT_CONV.has(conv)) {
      throw new StyleError(`${what}: unsupported conversion '${text}' in '${format}'`);
    }
    if (lit) { out.push({ text: lit }); lit = ''; }
    out.push({ spec: { flags, width: w === '' ? 0 : Number(w), prec, conv, text } });
  }
  if (lit) out.push({ text: lit });
  return out;
};

/**
 * Checks that a format is a valid format for one value of the given kind:
 * no conversion, or exactly one conversion of the matching family.
 */
export const checkCFormat = (format: string, kind: ValueKind, what = 'Invalid format'): CSpec | null => {
  const specs = parseCFormat(format, what).flatMap((p) => ('spec' in p ? [p.spec] : []));
  if (specs.length > 1) {
    throw new StyleError(`${what}: '${format}' has ${specs.length} conversion(s) but only 1 value(s) are provided`);
  }
  const spec = specs[0] ?? null;
  if (spec) {
    const isInt = INT_CONV.has(spec.conv);
    if (kind === 'float' && isInt) {
      throw new StyleError(`${what}: conversion 1 of '${spec.text}' formats integer values, but floating-point values are provided`);
    }
    if (kind === 'int' && !isInt) {
      throw new StyleError(`${what}: conversion 1 of '${spec.text}' formats floating-point values, but integer values are provided`);
    }
  }
  return spec;
};

// ---------------------------------------------------------------- exact decimal expansion

const view = new DataView(new ArrayBuffer(8));

/**
 * A finite non-negative double as N / 10^kd with N an integer (exact). For a
 * value with fraction digits, N = m * 5^kd where m is odd, so the value has
 * exactly kd decimals; integers have kd = 0.
 */
const binParts = (abs: number): { m: number; e2: number } => {
  view.setFloat64(0, abs);
  const hi = view.getUint32(0), lo = view.getUint32(4);
  const expField = (hi >>> 20) & 0x7ff;
  let m = (hi & 0xfffff) * 4294967296 + lo;
  let e2: number;
  if (expField === 0) e2 = -1074;
  else { m += 4503599627370496; e2 = expField - 1075; }
  while (m > 0 && m % 2 === 0) { m /= 2; e2++; }
  return { m, e2 };
};

/** Number of decimals of abs (exact): 0 for integers. Cheap, no BigInt. */
const decimalsOf = (abs: number): number => {
  const { e2 } = binParts(abs);
  return e2 < 0 ? -e2 : 0;
};

const exactParts = (abs: number): { N: bigint; kd: number } => {
  const { m, e2 } = binParts(abs);
  if (e2 >= 0) return { N: BigInt(m) << BigInt(e2), kd: 0 };
  const kd = -e2;
  return { N: BigInt(m) * 5n ** BigInt(kd), kd };
};

/** N / D rounded to the nearest integer, ties to even (C's rounding of exact halves). */
const roundHalfEven = (N: bigint, D: bigint): bigint => {
  const q = N / D;
  const r2 = 2n * (N - q * D);
  if (r2 > D || (r2 === D && (q & 1n) === 1n)) return q + 1n;
  return q;
};

/** abs with `prec` decimals, exact. */
const exactFixed = (abs: number, prec: number): string => {
  const { N, kd } = exactParts(abs);
  const Q = kd <= prec ? N * 10n ** BigInt(prec - kd) : roundHalfEven(N, 10n ** BigInt(kd - prec));
  let s = Q.toString().padStart(prec + 1, '0');
  if (prec > 0) s = `${s.slice(0, s.length - prec)}.${s.slice(s.length - prec)}`;
  return s;
};

/** Exponent of the leading digit of abs, exact (no rounding). */
const exactLeadingExp = (abs: number): number => {
  const { N, kd } = exactParts(abs);
  return N.toString().length - 1 - kd;
};

/** abs rounded to P significant digits, exact: digits (length P) and the decimal exponent of the first digit. */
const exactSci = (abs: number, P: number): { digits: string; e: number } => {
  const { N, kd } = exactParts(abs);
  const len = N.toString().length;
  let E0 = len - 1 - kd;
  const s = P - 1 - E0 - kd;
  let Q = s >= 0 ? N * 10n ** BigInt(s) : roundHalfEven(N, 10n ** BigInt(-s));
  if (Q.toString().length > P) { Q /= 10n; E0++; }
  return { digits: Q.toString(), e: E0 };
};

/**
 * A tie needs the exact expansion to end exactly at the rounding digit with a 5.
 * Fixed: the value has kd decimals, so a tie at `prec` needs kd == prec + 1.
 * Scientific: the value has at least 0.69 * kd significant digits, so a tie at P
 * digits needs kd <= (P + 1) / 0.69; those values take the exact path.
 */
const sciMayTie = (kd: number, P: number): boolean => kd <= Math.ceil((P + 1) / 0.69) + 1;

/** abs with `prec` decimals (no sign). */
const fixedDigits = (abs: number, prec: number): string => {
  if (abs === 0) return prec > 0 ? `0.${'0'.repeat(prec)}` : '0';
  if (abs >= 1e21 || prec > 100 || decimalsOf(abs) === prec + 1) return exactFixed(abs, prec);
  return abs.toFixed(prec);
};

/** abs rounded to P significant digits (no sign): digits of length P and the exponent of the first digit. */
const sciDigits = (abs: number, P: number): { digits: string; e: number } => {
  if (abs === 0) return { digits: '0'.repeat(P), e: 0 };
  if (P > 100 || sciMayTie(decimalsOf(abs), P)) return exactSci(abs, P);
  const s = abs.toExponential(P - 1);
  const k = s.indexOf('e');
  return { digits: s.slice(0, k).replace('.', ''), e: Number(s.slice(k + 1)) };
};

// ---------------------------------------------------------------- formatting one value

const signOf = (neg: boolean, flags: string): string => (neg ? '-' : flags.includes('+') ? '+' : flags.includes(' ') ? ' ' : '');

/** Pads body (after its sign) to the width with the flags' rules: '-' left-justify, '0' zeros after the sign. */
const pad = (sign: string, body: string, spec: CSpec, zeroOk: boolean): string => {
  const len = sign.length + body.length;
  if (len >= spec.width) return sign + body;
  if (spec.flags.includes('-')) return (sign + body).padEnd(spec.width);
  if (spec.flags.includes('0') && zeroOk) return sign + body.padStart(spec.width - sign.length, '0');
  return (sign + body).padStart(spec.width);
};

const formatFloatValue = (value: number, spec: CSpec): string => {
  const upper = spec.conv === 'E' || spec.conv === 'G' || spec.conv === 'F';
  const alt = spec.flags.includes('#');
  if (!Number.isFinite(value)) {
    const body = Number.isNaN(value) ? 'nan' : 'inf';
    const sign = value < 0 ? '-' : signOf(false, spec.flags);
    return pad(sign, upper ? body.toUpperCase() : body, spec, false);
  }
  const neg = value < 0 || Object.is(value, -0);
  const abs = Math.abs(value);
  const prec = spec.prec ?? 6;
  let body: string;
  if (spec.conv === 'f' || spec.conv === 'F') {
    body = fixedDigits(abs, prec);
    if (prec === 0 && alt) body += '.';
  } else if (spec.conv === 'e' || spec.conv === 'E') {
    const { digits, e } = sciDigits(abs, prec + 1);
    body = expForm(digits, e, prec, alt, upper);
  } else {
    const P = prec === 0 ? 1 : prec;
    const { digits, e } = sciDigits(abs, P);
    if (P > e && e >= -4) {
      body = fixedDigits(abs, P - 1 - e);
      if (!alt) body = stripZeros(body);
      if (alt && P - 1 - e === 0) body += '.';
    } else {
      // Measured with native LAMMPS (black box): with the # flag, a rounding carry that lands exactly on
      // 10^P (for example %#g of 999999.5 prints 1.e+06, and %#.2g of 99.5 prints 1.e+02) keeps P-1-X decimals,
      // X being the exponent before rounding; other sci-branch values keep P-1 decimals.
      let dec = P - 1;
      if (alt && e === P) {
        const x0 = exactLeadingExp(abs);
        if (x0 !== e) dec = Math.max(0, P - 1 - x0);
      }
      body = expForm(digits, e, dec, alt, upper);
      if (!alt) body = stripZeros(body);
    }
  }
  return pad(signOf(neg, spec.flags), body, spec, true);
};

/** Mantissa with `prec` decimals and an exponent of at least two digits. */
const expForm = (digits: string, e: number, prec: number, alt: boolean, upper: boolean): string => {
  const mant = prec > 0 || alt ? `${digits[0]}.${digits.slice(1, prec + 1)}` : digits[0];
  const ex = `${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
  return `${mant}${upper ? 'E' : 'e'}${ex}`;
};

/** Removes trailing zeros of a fraction (and the point when nothing remains); an exponent suffix is kept. */
const stripZeros = (s: string): string => {
  const k = s.search(/[eE]/);
  const mant = k < 0 ? s : s.slice(0, k);
  const rest = k < 0 ? '' : s.slice(k);
  if (!mant.includes('.')) return s;
  return mant.replace(/0+$/, '').replace(/\.$/, '') + rest;
};

const formatIntValue = (value: number, spec: CSpec): string => {
  const v = Math.trunc(value);
  const neg = v < 0;
  let digits = Number.isSafeInteger(Math.abs(v)) ? String(Math.abs(v)) : BigInt(Math.abs(v)).toString();
  if (spec.prec !== null) digits = spec.prec === 0 && v === 0 ? '' : digits.padStart(spec.prec, '0');
  return pad(signOf(neg, spec.flags), digits, spec, spec.prec === null);
};

// ---------------------------------------------------------------- public API

const cache = new Map<string, (v: number) => string>();

/**
 * A formatter for one value of the given kind. The format is checked once
 * (checkCFormat) and compiled into a closure. Throws StyleError for an
 * unsupported or mismatched format.
 */
export const compileCFormat = (format: string, kind: ValueKind, what = 'Invalid format'): ((v: number) => string) => {
  const key = `${kind}\u0000${format}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const spec = checkCFormat(format, kind, what);
  const pieces = parseCFormat(format, what);
  let fn: (v: number) => string;
  if (!spec) {
    const literal = pieces.map((p) => ('text' in p ? p.text : '')).join('');
    fn = () => literal;
  } else {
    const at = pieces.findIndex((p) => 'spec' in p);
    const before = pieces.slice(0, at).map((p) => ('text' in p ? p.text : '')).join('');
    const after = pieces.slice(at + 1).map((p) => ('text' in p ? p.text : '')).join('');
    const one = kind === 'int' ? (v: number) => formatIntValue(v, spec) : (v: number) => formatFloatValue(v, spec);
    fn = (v) => before + one(v) + after;
  }
  if (cache.size > 512) cache.clear();
  cache.set(key, fn);
  return fn;
};

/** A floating-point value in a C format ("%.4f", "<%g>", "%+12.6e"). */
export const formatNumber = (value: number, format: string): string =>
  compileCFormat(format, 'float', 'Invalid format string for format style variable')(value);

/** An integer value in a C format ("%d", "%5i"). */
export const formatInteger = (value: number, format: string): string =>
  compileCFormat(format, 'int', 'Invalid format')(value);
