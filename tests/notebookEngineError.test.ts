import { describe, expect, it } from 'vitest';
import { closestName, editDistance, explainEngineError } from '../src/components/workbench/engineError';
import { Session } from '../src/engine/interpreter';

const errorOf = async (script: string): Promise<string> => {
  const s = new Session({ emit: () => {} });
  try { await s.execute(script); } catch (e) { return (e as Error).message; }
  throw new Error('no error');
};

describe('notebook engine-error display', () => {
  it('edit distance counts a transposition as one edit', () => {
    expect(editDistance('pair_coef', 'pair_coeff')).toBe(1);
    expect(editDistance('fxi', 'fix')).toBe(1);
    expect(closestName('pair_coef', ['pair_style', 'pair_coeff', 'pair_modify'])).toBe('pair_coeff');
    expect(closestName('zzzzzz', ['fix', 'run'])).toBeNull();
  });

  it('an unknown command from the engine: head without the list, a suggestion, the list kept', async () => {
    const v = explainEngineError(await errorOf('pair_coef 1 1 1.0 1.0\n'));
    expect(v.head).toMatch(/'pair_coef' is not supported by the in-browser engine \(it is not LAMMPS\)\.$/);
    expect(v.head).not.toMatch(/Supported commands/);
    expect(v.suggestion).toBe('pair_coeff');
    expect(v.supported).toContain('pair_coeff');
    expect(v.supported.length).toBeGreaterThan(50);
  });

  it('an unknown fix style and an unknown keyword', async () => {
    const box = 'units lj\nlattice fcc 0.8442\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\ncreate_atoms 1 box\nmass 1 1.0\n';
    const fix = explainEngineError(await errorOf(`${box}fix 9 all nvz temp 1 1 0.1\n`));
    expect(fix.suggestion).toBe('nve');
    expect(fix.head).not.toMatch(/supported:/);
    const kw = explainEngineError("line 2: velocity: unknown keyword 'bais' (supported: dist, sum, mom, rot, bias, loop)");
    expect(kw).toEqual({ head: "line 2: velocity: unknown keyword 'bais'", suggestion: 'bias', supported: ['dist', 'sum', 'mom', 'rot', 'bias', 'loop'] });
  });

  it('other messages pass through unchanged', () => {
    expect(explainEngineError('line 4: Lost atoms: original 500 current 499')).toEqual({ head: 'line 4: Lost atoms: original 500 current 499', suggestion: null, supported: [] });
  });
});
