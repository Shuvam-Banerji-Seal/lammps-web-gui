import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { expandWildcards, globalScalar, globalVector, parseRef, peratomValues } from '../src/engine/refs';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/* c_ID / f_ID / v_name references shared by fix ave/* and compute reduce (src/engine/refs.ts). */

const setup = async () => {
  const rows: ThermoRow[] = [];
  const session = new Session({ emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); }, writeFile: () => {} });
  await session.execute(`
units           lj
lattice         fcc 0.8442
region          box block 0 3 0 3 0 3
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.5 4928459 loop geom
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0
variable        twice equal 2*temp
variable        vx2 atom 2*vx
variable        vec vector [1,4,9]
thermo_style    custom step temp press pxx pyy pzz pxy
run             0
`);
  return { sys: session.sys, row: rows[rows.length - 1] };
};

describe('refs: parsing', () => {
  it('accepts c_/f_/v_ with optional [I] and attributes only where allowed', () => {
    expect(parseRef('c_myT')).toMatchObject({ kind: 'c', id: 'myT', index: null });
    expect(parseRef('f_ave[3]')).toMatchObject({ kind: 'f', id: 'ave', index: 3 });
    expect(parseRef('vx', true)).toMatchObject({ kind: 'attr', id: 'vx' });
    expect(() => parseRef('vx')).toThrow(/invalid input value 'vx'/);
    expect(() => parseRef('c_x[0]')).toThrow(/indices start at 1/);
    expect(() => parseRef('temp')).toThrow(/expected c_ID/);
  });
});

describe('refs: values', () => {
  it('global scalars and vectors agree with thermo output', async () => {
    const { sys, row } = await setup();
    expect(globalScalar(sys, parseRef('c_thermo_temp'))).toBeCloseTo(row.temp, 12);
    expect(globalScalar(sys, parseRef('c_thermo_press[1]'))).toBeCloseTo(row.pxx, 10);
    expect(globalScalar(sys, parseRef('v_twice'))).toBeCloseTo(2 * row.temp, 12);
    expect(globalScalar(sys, parseRef('v_vec[3]'))).toBe(9);
    const p = globalVector(sys, parseRef('c_thermo_press'));
    expect(Array.from(p.slice(0, 4)).map((x) => +x.toFixed(9))).toEqual([row.pxx, row.pyy, row.pzz, row.pxy].map((x) => +x.toFixed(9)));
    expect(Array.from(globalVector(sys, parseRef('v_vec')))).toEqual([1, 4, 9]);
    expect(() => globalScalar(sys, parseRef('c_thermo_press[7]'))).toThrow(/out of range 1..6/);
    expect(() => globalScalar(sys, parseRef('c_nope'))).toThrow(/compute ID 'nope' does not exist/);
  });

  it('per-atom values: attributes and atom-style variables', async () => {
    const { sys } = await setup();
    const s = sys.state;
    const vx = peratomValues(sys, parseRef('vx', true));
    const vx2 = peratomValues(sys, parseRef('v_vx2'));
    expect(vx.length).toBe(s.n);
    for (let i = 0; i < s.n; i++) {
      expect(vx[i]).toBe(s.v[3 * i]);
      expect(vx2[i]).toBeCloseTo(2 * s.v[3 * i], 14);
    }
    expect(() => peratomValues(sys, parseRef('c_thermo_temp'))).toThrow(/does not calculate per-atom values/);
  });
});

describe('refs: wildcards', () => {
  it('expands *, *n, m* and m*n over the vector length', async () => {
    const { sys } = await setup();
    expect(expandWildcards(sys, ['c_thermo_press[*]'], 'scalar')).toHaveLength(6);
    expect(expandWildcards(sys, ['c_thermo_press[*2]'], 'scalar')).toEqual(['c_thermo_press[1]', 'c_thermo_press[2]']);
    expect(expandWildcards(sys, ['c_thermo_press[5*]'], 'scalar')).toEqual(['c_thermo_press[5]', 'c_thermo_press[6]']);
    expect(expandWildcards(sys, ['v_twice', 'c_thermo_press[2*3]'], 'scalar')).toEqual(['v_twice', 'c_thermo_press[2]', 'c_thermo_press[3]']);
    expect(expandWildcards(sys, ['v_vec[*]'], 'scalar')).toEqual(['v_vec[1]', 'v_vec[2]', 'v_vec[3]']);
  });
});
