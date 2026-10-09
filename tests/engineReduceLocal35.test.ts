import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute reduce with "inputs local" (docs.lammps.org/compute_reduce.html): the local vectors and
 * local array columns of the engine's local computes are reduced with the same modes as per-atom
 * inputs. Measured with native LAMMPS (black box): the reduce group does not filter local rows (a
 * subgroup of the input compute's group gives the same value), the count used by ave is the number
 * of local rows, an empty local list gives 0 for sum/sumsq/sumabs/ave/avesq/aveabs, 1e20 for
 * min/minabs, -1e20 for max and 0 for maxabs, and mixing per-atom and local inputs is an error.
 */

const close = (got: number, want: number, rel = 1e-12) => Math.abs(got - want) <= rel * Math.max(1, Math.abs(got), Math.abs(want));

const DATA = `LAMMPS data

3 atoms
1 atom types
2 bonds
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # molecular

1 1 1 1.0 1.0 1.0
2 1 1 2.0 1.0 1.0
3 1 1 2.0 3.0 1.0

Bonds

1 1 1 2
2 1 2 3
`;

/** Bond 1 (atoms 1-2) is 1 long, bond 2 (atoms 2-3) is 2 long. */
const molecule = async (extra: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  session.addFile('mol3.data', DATA);
  await session.execute(`
    units lj
    atom_style molecular
    read_data mol3.data
    bond_style harmonic
    bond_coeff 1 100.0 1.0
    pair_style lj/cut 2.5
    pair_coeff 1 1 1.0 1.0
    ${extra}
    run 0
  `);
  return session;
};

describe('compute reduce inputs local: values', () => {
  it('reduces a local vector with every mode', async () => {
    const s = await molecule(`
      compute b all bond/local dist
      compute sum all reduce sum c_b inputs local
      compute mn all reduce min c_b inputs local
      compute mx all reduce max c_b inputs local
      compute ave all reduce ave c_b inputs local
      compute sq all reduce sumsq c_b inputs local
      compute mnabs all reduce minabs c_b inputs local
      compute mxabs all reduce maxabs c_b inputs local
      compute sabs all reduce sumabs c_b inputs local
      compute asq all reduce avesq c_b inputs local
      compute aabs all reduce aveabs c_b inputs local
    `);
    const sys = s.sys;
    expect(sys.compute('sum').scalarValue()).toBeCloseTo(3, 12);
    expect(sys.compute('mn').scalarValue()).toBeCloseTo(1, 12);
    expect(sys.compute('mx').scalarValue()).toBeCloseTo(2, 12);
    expect(sys.compute('ave').scalarValue()).toBeCloseTo(1.5, 12);
    expect(sys.compute('sq').scalarValue()).toBeCloseTo(5, 12);
    expect(sys.compute('mnabs').scalarValue()).toBeCloseTo(1, 12);
    expect(sys.compute('mxabs').scalarValue()).toBeCloseTo(2, 12);
    expect(sys.compute('sabs').scalarValue()).toBeCloseTo(3, 12);
    expect(sys.compute('asq').scalarValue()).toBeCloseTo(2.5, 12);
    expect(sys.compute('aabs').scalarValue()).toBeCloseTo(1.5, 12);
    // sum/sumsq/sumabs on local vectors are extensive (compute_reduce.html)
    expect(sys.compute('sum').extscalar).toBe(1);
    expect(sys.compute('sq').extscalar).toBe(1);
    expect(sys.compute('ave').extscalar).toBe(0);
  });

  it('reduces a local array column and expands a local wildcard', async () => {
    const s = await molecule(`
      compute b all bond/local dist engpot
      compute one all reduce sum c_b[1] inputs local
      compute wild all reduce sum c_b[*] inputs local
    `);
    expect(s.sys.compute('one').scalarValue()).toBeCloseTo(3, 12);
    const wild = s.sys.compute('wild');
    expect(wild.vectorFlag).toBe(true);
    expect(wild.sizeVector).toBe(2);
    expect(wild.vectorValues()[0]).toBeCloseTo(3, 12); // dist
    expect(wild.vectorValues()[1]).toBeCloseTo(100, 12); // engpot = K*(r-r0)^2, r0 = 1: bond2 (r = 2) only
  });

  it('does not filter local rows by the group of compute reduce', async () => {
    const s = await molecule(`
      group sub id 1
      compute b all bond/local dist
      compute rall all reduce sum c_b inputs local
      compute rsub sub reduce sum c_b inputs local
      compute rsub2 sub reduce ave c_b inputs local
    `);
    // the rows come from compute b (group all); the reduce group does not add a filter
    expect(s.sys.compute('rsub').scalarValue()).toBeCloseTo(3, 12);
    expect(s.sys.compute('rsub2').scalarValue()).toBeCloseTo(1.5, 12);
    expect(s.sys.compute('rall').scalarValue()).toBeCloseTo(s.sys.compute('rsub').scalarValue(), 12);
  });

  it('filters rows by the group of the input compute itself', async () => {
    const s = await molecule(`
      group sub id 1
      compute bsub sub bond/local dist
      compute r all reduce sum c_bsub inputs local
    `);
    expect(s.sys.compute('bsub').localRows).toBe(0);
    expect(s.sys.compute('r').scalarValue()).toBe(0);
  });

  it('uses the measured native empty-local-list sentinels', async () => {
    const s = await molecule(`
      group sub id 1
      compute bsub sub bond/local dist engpot
      compute sum all reduce sum c_bsub[1] inputs local
      compute ave all reduce ave c_bsub[1] inputs local
      compute sq all reduce sumsq c_bsub[1] inputs local
      compute mn all reduce min c_bsub[1] inputs local
      compute mx all reduce max c_bsub[1] inputs local
      compute mnabs all reduce minabs c_bsub[1] inputs local
      compute mxabs all reduce maxabs c_bsub[1] inputs local
      compute sabs all reduce sumabs c_bsub[1] inputs local
      compute asq all reduce avesq c_bsub[1] inputs local
      compute aabs all reduce aveabs c_bsub[1] inputs local
    `);
    const sys = s.sys;
    expect(sys.compute('sum').scalarValue()).toBe(0);
    expect(sys.compute('ave').scalarValue()).toBe(0);
    expect(sys.compute('sq').scalarValue()).toBe(0);
    expect(sys.compute('sabs').scalarValue()).toBe(0);
    expect(sys.compute('asq').scalarValue()).toBe(0);
    expect(sys.compute('aabs').scalarValue()).toBe(0);
    expect(sys.compute('mn').scalarValue()).toBe(1e20);
    expect(sys.compute('mnabs').scalarValue()).toBe(1e20);
    expect(sys.compute('mx').scalarValue()).toBe(-1e20);
    expect(sys.compute('mxabs').scalarValue()).toBe(0);
  });

  it('replace selects from vec1 at the min/max row of vec2 (local)', async () => {
    const s = await molecule(`
      compute p all property/local batom1 batom2
      compute b all bond/local dist
      compute r all reduce max c_p[1] c_p[2] c_b replace 1 3 replace 2 3 inputs local
    `);
    const vals = s.sys.compute('r').vectorValues();
    // the longest bond is bond 2 (atoms 2-3, distance 2)
    expect(close(vals[0], 2)).toBe(true);
    expect(close(vals[1], 3)).toBe(true);
    expect(close(vals[2], 2)).toBe(true);
  });
});

describe('compute reduce inputs local: errors', () => {
  it('rejects per-atom inputs under inputs local and local inputs without the keyword', async () => {
    await expect(molecule('compute b all bond/local dist engpot\ncompute r all reduce sum c_b[1]')).rejects.toThrow('does not calculate per-atom values');
    await expect(molecule('compute r all reduce sum x inputs local')).rejects.toThrow('inputs local is not supported for per-atom inputs');
    await expect(molecule('compute p all property/atom x\ncompute r all reduce sum c_p inputs local')).rejects.toThrow('does not calculate local values');
    await expect(molecule('fix 1 all nve\ncompute r all reduce sum f_1 inputs local')).rejects.toThrow('does not calculate local values');
  });

  it('rejects local vector/array index mistakes', async () => {
    await expect(molecule('compute b all bond/local dist\ncompute r all reduce sum c_b[1] inputs local')).rejects.toThrow('local vector, which has no columns');
    await expect(molecule('compute b all bond/local dist engpot\ncompute r all reduce sum c_b inputs local')).rejects.toThrow('local array; give a column');
    await expect(molecule('compute b all bond/local dist engpot\ncompute r all reduce sum c_b[3] inputs local')).rejects.toThrow('column out of range 1..2');
    await expect(molecule('compute b all bond/local dist\ncompute r all reduce sum c_b[*] inputs local')).rejects.toThrow('no local columns');
  });

  it('rejects local inputs in reduce/region and a mixing of kinds', async () => {
    await expect(molecule('region rr block 0 5 0 5 0 5\ncompute b all bond/local dist engpot\ncompute r all reduce/region rr sum c_b[1] inputs local')).rejects.toThrow('cannot use local data as input');
    await expect(molecule('compute b all bond/local dist engpot\ncompute r all reduce sum c_b[1] x inputs local')).rejects.toThrow('inputs local is not supported for per-atom inputs');
  });
});
