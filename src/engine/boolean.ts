import { StyleError } from './force/types';

/*
 * Boolean expressions of the if command — docs.lammps.org/if.html:
 * "An expression is built out of numbers (which start with a digit or period
 * or minus sign) or strings (which start with a letter and can contain
 * alphanumeric characters, underscores, or forward slashes)" and the
 * operators "A == B, A != B, A < B, A <= B, A > B, A >= B, A && B, A || B,
 * A |^ B, !A". "The Boolean operators == and != can operate on a pair or
 * strings or numbers. They cannot compare a number to a string. All the other
 * Boolean operations can only operate on numbers." Precedence: "the unary
 * logical NOT operator ! has the highest precedence, the 4 relational
 * operators <, <=, >, and >= are next; the two remaining relational operators
 * == and != are next; then the logical AND operator &&; and finally the
 * logical OR operator || and logical XOR (exclusive or) operator |^ have the
 * lowest precedence." "If the Boolean expression is a single string, an error
 * message will be issued." Variables are substituted before evaluation.
 */

type V = number | string;
type Tok = { k: 'num'; v: number } | { k: 'str'; v: string } | { k: 'op'; v: string };

const OPS = ['&&', '||', '|^', '==', '!=', '<=', '>=', '<', '>', '!', '(', ')'];

const lex = (src: string): Tok[] => {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i++; continue; }
    const op = OPS.find((o) => src.startsWith(o, i));
    // a minus sign starts a number only where a value is expected
    const valueExpected = out.length === 0 || out[out.length - 1].k === 'op' && (out[out.length - 1] as { v: string }).v !== ')';
    const num = /^[-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(src.slice(i));
    if (num && (src[i] !== '-' || valueExpected)) { out.push({ k: 'num', v: Number(num[0]) }); i += num[0].length; continue; }
    if (op) { out.push({ k: 'op', v: op }); i += op.length; continue; }
    const str = /^[A-Za-z][A-Za-z0-9_/]*/.exec(src.slice(i));
    if (str) { out.push({ k: 'str', v: str[0] }); i += str[0].length; continue; }
    throw new StyleError(`invalid character '${c}' in if Boolean "${src}"`);
  }
  return out;
};

export const evaluateBoolean = (src: string): boolean => {
  const toks = lex(src);
  if (toks.length === 1 && toks[0].k === 'str') throw new StyleError(`if Boolean "${src}" is a single string`);
  let p = 0;
  const isOp = (v: string) => toks[p]?.k === 'op' && (toks[p] as { v: string }).v === v;
  const num = (x: V, op: string): number => {
    if (typeof x !== 'number') throw new StyleError(`if: operator ${op} needs numbers, got '${x}'`);
    return x;
  };
  const b = (t: boolean) => (t ? 1 : 0);
  const LEVELS: string[][] = [['||', '|^'], ['&&'], ['==', '!='], ['<', '<=', '>', '>=']];
  const level = (l: number): V => {
    if (l === LEVELS.length) return unary();
    let a = level(l + 1);
    for (;;) {
      const t = toks[p];
      if (!t || t.k !== 'op' || !LEVELS[l].includes(t.v)) return a;
      p++;
      const c = level(l + 1);
      switch (t.v) {
        case '||': a = b(num(a, t.v) !== 0 || num(c, t.v) !== 0); break;
        case '|^': a = b((num(a, t.v) !== 0) !== (num(c, t.v) !== 0)); break;
        case '&&': a = b(num(a, t.v) !== 0 && num(c, t.v) !== 0); break;
        case '==': case '!=': {
          if (typeof a !== typeof c) throw new StyleError(`if: cannot compare a number to a string in "${src}"`);
          a = b(t.v === '==' ? a === c : a !== c);
          break;
        }
        case '<': a = b(num(a, t.v) < num(c, t.v)); break;
        case '<=': a = b(num(a, t.v) <= num(c, t.v)); break;
        case '>': a = b(num(a, t.v) > num(c, t.v)); break;
        case '>=': a = b(num(a, t.v) >= num(c, t.v)); break;
      }
    }
  };
  const unary = (): V => {
    if (isOp('!')) { p++; return b(num(unary(), '!') === 0); }
    if (isOp('(')) {
      p++;
      const v = level(0);
      if (!isOp(')')) throw new StyleError(`if: missing ')' in "${src}"`);
      p++;
      return v;
    }
    const t = toks[p++];
    if (!t || t.k === 'op') throw new StyleError(`if: invalid Boolean "${src}"`);
    return t.v;
  };
  const v = level(0);
  if (p !== toks.length) throw new StyleError(`if: invalid Boolean "${src}"`);
  if (typeof v !== 'number') throw new StyleError(`if Boolean "${src}" is a single string`);
  return v !== 0;
};
