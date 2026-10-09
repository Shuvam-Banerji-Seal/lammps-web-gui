import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

const PARAMS = readFileSync(join(__dirname, 'oracle', 'w32qeq_point.params'), 'utf8');
const PARAM36 = readFileSync(join(__dirname, 'oracle', 'w36qeq_param.qeq2'), 'utf8');

const twoAtom = (fix: string) => `units metal
atom_style charge
boundary f f f
region box block 0 30 0 30 0 30
create_box 2 box
create_atoms 1 single 5 15 15
create_atoms 2 single 8 15 15
mass 1 1.0
mass 2 1.0
set atom 1 charge 0.0
set atom 2 charge 0.0
pair_style coul/cut 20.0
pair_coeff * *
${fix}
run 0
`;
const fourAtom = (fix: string) => `units metal
atom_style charge
boundary f f f
region box block 0 30 0 30 0 30
create_box 2 box
create_atoms 1 single 5 15 15
create_atoms 2 single 7 15 15
create_atoms 1 single 9 15 15
create_atoms 2 single 11 15 15
mass 1 1.0
mass 2 1.0
set atom * charge 0.0
pair_style coul/cut 20.0
pair_coeff * *
${fix}
run 0
`;
const session = (params = PARAMS) => { const s = new Session({ emit: () => {} }); s.addFile('params.qeq', params); return s; };
const qOf = (s: Session) => (s as unknown as { sys: { state: { q: Float64Array } } }).sys.state.q;

describe('fix qeq/dynamic and qeq/fire', () => {
  it('two point charges: converged charges equal qeq/point and native LAMMPS', async () => {
    // native LAMMPS (black box), metal, coul/cut: dynamic at tol 1e-20 gives q = ±0.513086362342165, pe = -1.26360540091214
    const ref = 0.513086362342165;
    for (const style of ['qeq/point', 'qeq/dynamic', 'qeq/fire']) {
      const s = session();
      await s.execute(twoAtom(`fix q all ${style} 1 20 1.0e-18 200000 params.qeq`));
      const q = qOf(s);
      expect(q[0], style).toBeCloseTo(ref, 12);
      expect(q[0] + q[1], style).toBeCloseTo(0, 12);
    }
  });

  it('four point charges: dynamic and fire agree with qeq/point and native LAMMPS', async () => {
    // native LAMMPS (black box), metal, coul/cut, 4 atoms
    const ref = [0.5128099311325, -0.530828426934142, 0.537548463880667, -0.519529968079025];
    for (const style of ['qeq/point', 'qeq/dynamic', 'qeq/fire']) {
      const s = session();
      await s.execute(fourAtom(`fix q all ${style} 1 20 1.0e-18 200000 params.qeq`));
      const q = qOf(s);
      for (let i = 0; i < 4; i++) expect(q[i], `${style} atom ${i}`).toBeCloseTo(ref[i], 11);
    }
  });

  it('qeq/dynamic reproduces the native default iterates and iteration count exactly', async () => {
    // native LAMMPS (black box), qdamp 0.1 / qstep 0.02, tolerance 1e-3:
    // 73 iterations and q = 0.513041007887819 (the charge is off the converged value).
    const s = session();
    await s.execute(twoAtom('fix q all qeq/dynamic 1 20 1.0e-3 200 params.qeq'));
    const q = qOf(s);
    expect(q[0]).toBeCloseTo(0.513041007887819, 12);
    const fix = (s as unknown as { sys: { fixes: { computeScalar(): number }[] } }).sys.fixes[0];
    expect(fix.computeScalar()).toBe(73);
  });

  it('qeq/dynamic keyword qdamp/qstep change the trajectory; defaults are qdamp 0.1 / qstep 0.02', async () => {
    const s = session();
    await s.execute(twoAtom('fix q all qeq/dynamic 1 20 1.0e-3 200 params.qeq'));
    const def = qOf(s)[0];
    const s2 = session();
    await s2.execute(twoAtom('fix q all qeq/dynamic 1 20 1.0e-3 200 params.qeq qdamp 0.5 qstep 0.05'));
    const changed = qOf(s2)[0];
    expect(def).not.toBeCloseTo(changed, 6);
  });

  it('qeq/fire ignores qdamp, uses qstep, and refuses malformed keywords', async () => {
    const a = session(); await a.execute(twoAtom('fix q all qeq/fire 1 20 1.0e-3 200 params.qeq qdamp 0.1 qstep 0.1'));
    const b = session(); await b.execute(twoAtom('fix q all qeq/fire 1 20 1.0e-3 200 params.qeq qdamp 0.5 qstep 0.1'));
    // measured with native LAMMPS (black box): qeq/fire ignores qdamp
    expect(qOf(a)[0]).toBeCloseTo(qOf(b)[0], 12);
    // qdamp/qstep are documented for qeq/dynamic and qeq/fire only
    await expect(session().execute(twoAtom('fix q all qeq/point 1 20 1.0e-6 200 params.qeq qstep 0.1'))).rejects.toThrow(/qstep/);
    await expect(session().execute(twoAtom('fix q all qeq/dynamic 1 20 1.0e-6 200 params.qeq bogus 1'))).rejects.toThrow(/bogus/);
  });

  it('the example param.qeq2 loads and drives a two-charge system', async () => {
    const s = session(PARAM36);
    await s.execute(twoAtom('fix q all qeq/dynamic 1 8 1.0e-18 200000 params.qeq'));
    expect(qOf(s)[0]).toBeCloseTo(0.513086362342165, 12);
  });
});

