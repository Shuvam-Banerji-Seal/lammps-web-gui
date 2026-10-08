import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Fixes found by running the LAMMPS examples through the engine (wave 19). Expected values were measured
 * with native LAMMPS (black box).
 */

const newSession = () => {
  const files = new Map<string, string>();
  const logs: string[] = [];
  const session = new Session({ emit: (e) => { if (e.kind === 'log') logs.push(e.text); }, writeFile: (n, t) => files.set(n, t) });
  return { session, files, logs };
};

describe('variable formulas with unquoted spaces (examples/granular/in.restitution)', () => {
  it('joins the words of an equal-style formula', async () => {
    const { session, logs } = newSession();
    await session.execute('variable q equal 7\nvariable a equal 2 +    3\nvariable c equal (2   + 3)*v_q\nprint "a=${a} c=${c}"\n');
    // measured: a=5 c=35
    expect(logs.some((l) => l.includes('a=5 c=35'))).toBe(true);
  });

  it('still refuses extra words for string, format and internal', async () => {
    for (const line of ['variable z string hello world', 'variable z format a %.3f extra', 'variable z internal 2 3']) {
      const { session } = newSession();
      await expect(session.execute(`variable a equal 1\n${line}\n`)).rejects.toThrow();
    }
  });
});

describe('dump custom element column (examples/VISCOSITY/in.cos.1000SPCE)', () => {
  const input = `units lj
region b block 0 2 0 1 0 1
create_box 2 b
create_atoms 1 single 0.5 0.5 0.5
create_atoms 2 single 1.5 0.5 0.5
mass * 1
dump d all custom 1 e1.dump id element type x
dump_modify d sort id
dump d2 all custom 1 e2.dump id element type
dump_modify d2 sort id element O H
run 0
`;
  it('writes C for every type without dump_modify element, and the names with it', async () => {
    const { session, files } = newSession();
    await session.execute(input);
    expect(files.get('e1.dump')!.trim().split('\n').slice(-2)).toEqual(['1 C 1 0.5', '2 C 2 1.5']);
    expect(files.get('e2.dump')!.trim().split('\n').slice(-2)).toEqual(['1 O 1', '2 H 2']);
  });
});
