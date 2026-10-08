import { describe, expect, it } from 'vitest';
import { substituteVariables } from '../src/engine/script';

/*
 * $ substitution (docs.lammps.org/Commands_parse.html). Measured with native
 * LAMMPS (black box): the text a variable substitutes is scanned again, so
 * string variables defined in quotes (quotes keep the $ at definition) expand
 * fully when used, as the SNAP examples' option strings rely on.
 */

const vars: Record<string, string> = { a: '2', s: '${a}', s2: '${a}0', p: '${s2} 7', loop: '${loop}' };
const sub = (t: string) => substituteVariables(t, (n) => {
  if (!(n in vars)) throw new Error(`unknown variable ${n}`);
  return vars[n];
}, (f) => String(eval(f)));

describe('variable substitution', () => {
  it('rescans substituted text', () => {
    expect(sub('variable c equal ${s}+1')).toBe('variable c equal 2+1');
    expect(sub('x ${s2} y')).toBe('x 20 y');
    expect(sub('variable q index ${p}')).toBe('variable q index 20 7');
    expect(sub('$a$a')).toBe('22');
  });
  it('does not rescan immediate values and keeps formats', () => {
    expect(sub('$(1+2) ${a}')).toBe('3 2');
  });
  it('rejects nested references and self-reference loops', () => {
    expect(() => sub('${${a}}')).toThrow(/nested/);
    expect(() => sub('${loop}')).toThrow(/does not terminate/);
  });
});
