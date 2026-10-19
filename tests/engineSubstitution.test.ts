import { describe, expect, it } from 'vitest';
import { substituteVariables } from '../src/engine/script';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * $ substitution (docs.lammps.org/Commands_parse.html). As native LAMMPS does (src/input.cpp Input::substitute and
 * src/variable.cpp Variable::set at b332f28): the substituted text is not scanned again, a backslash before $ keeps the
 * $, and a string variable is substituted when it is defined (variable.html: "string performs variable substitution
 * even if the string parameter is quoted"). Measured with native LAMMPS: the prints of the session test below.
 */

const vars: Record<string, string> = { a: '2', s: '${a}', s2: '${a}0', p: '${s2} 7', loop: '${loop}' };
const sub = (t: string) => substituteVariables(t, (n) => {
  if (!(n in vars)) throw new Error(`unknown variable ${n}`);
  return vars[n];
}, (f) => String(eval(f)));

describe('variable substitution', () => {
  it('does not rescan substituted text', () => {
    expect(sub('variable c equal ${s}+1')).toBe('variable c equal ${a}+1');
    expect(sub('x ${s2} y')).toBe('x ${a}0 y');
    expect(sub('variable q index ${p}')).toBe('variable q index ${s2} 7');
    expect(sub('$a$a')).toBe('22');
    expect(sub('${loop}')).toBe('${loop}');
  });
  it('keeps a $ after a backslash', () => {
    expect(sub('\\$a and $a')).toBe('$a and 2');
  });
  it('does not rescan immediate values and keeps formats', () => {
    expect(sub('$(1+2) ${a}')).toBe('3 2');
  });
  it('rejects nested references', () => {
    expect(() => sub('${${a}}')).toThrow(/nested/);
  });

  it('substitutes a string variable when it is defined (native prints)', async () => {
    const prints: string[] = [];
    const session = new Session({
      emit: (e: EngineEvent) => { if (e.kind === 'log' && /^[SIBTF]2?=/.test(e.text)) prints.push(e.text); },
      writeFile: () => {},
    });
    await session.execute([
      'variable a equal 2', 'variable s string "${a}0"', 'print "S=${s}"', 'variable a equal 3', 'print "S2=${s}"',
      'variable i index \\${a}0', 'print "I=${i}"', 'variable b index "x${a}"', 'print "B=${b}"',
      'variable t string \\${a}9', 'print "T=${t}"',
      'variable f string ""', 'variable k loop 3', 'label l', 'variable g string "${f} a$k"', 'variable f string "${g}"',
      'next k', 'jump SELF l', 'print "F=${f}"',
    ].join('\n'));
    expect(prints).toEqual(['S=20', 'S2=20', 'I=${a}0', 'B=x${a}', 'T=39', 'F= a1 a2 a3']);
  });
});
