import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

const PARAMS = readFileSync(join(__dirname, 'oracle', 'w32qeq_point.params'), 'utf8');
const pairOf = (fix: string, extra = '') => `units metal
atom_style charge
boundary f f f
region box block 0 30 0 30 0 30
create_box 2 box
create_atoms 1 single 5 15 15
create_atoms 2 single 8 15 15
mass * 1.0
pair_style coul/cut 20.0
pair_coeff * *
${fix}
${extra}run 0
`;
const session = () => { const s = new Session({ emit: () => {} }); s.addFile('w32qeq_point.params', PARAMS); return s; };

describe('fix qeq/point and qeq/shielded', () => {
  it('two point charges: the closed-form QEq solution, neutral, as native (oracle w32qeq_point2)', async () => {
    const s = session();
    await s.execute(pairOf('fix q all qeq/point 1 20 1.0e-14 200 w32qeq_point.params'));
    const q = (s as unknown as { sys: { state: { q: Float64Array } } }).sys.state.q;
    // minimising chi.q + eta q^2/2 + q1 q2 / r with q1 = -q2: q1 = (chi2 - chi1) / (eta1 + eta2 - 2/r)
    const expected = (11.26882 - 0) / (7.25028 + 15.3792 - 2 / 3);
    expect(q[0]).toBeCloseTo(expected, 9);
    expect(q[0] + q[1]).toBeCloseTo(0, 12);
  });

  it('the other qeq styles are refused by name', async () => {
    for (const style of ['qeq/dynamic', 'qeq/fire', 'qeq/slater']) {
      await expect(session().execute(pairOf(`fix q all ${style} 1 20 1.0e-6 200 w32qeq_point.params`))).rejects.toThrow(style);
    }
  });

  it('reports bad input: an unknown keyword and a type without parameters', async () => {
    await expect(session().execute(pairOf('fix q all qeq/point 1 20 1.0e-6 200 w32qeq_point.params bogus 1'))).rejects.toThrow(/bogus/);
    const s = new Session({ emit: () => {} });
    s.addFile('one.params', '1 0.0 7.25028 0.2 0.772871 0.0\n');
    await expect(s.execute(pairOf('fix q all qeq/point 1 20 1.0e-6 200 one.params'))).rejects.toThrow(/no parameters for atom type 2/);
  });
});
