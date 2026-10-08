import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/session';
import { ComputePropertyGrid } from '../src/engine/compute/property_grid';
import { ComputeBornMatrix } from '../src/engine/compute/born_matrix';

/*
 * Wave-26 grid computes: compute property/grid (docs.lammps.org/compute_property_grid.html)
 * and compute born/matrix (docs.lammps.org/compute_born_matrix.html). Native parity
 * is in tests/oracle/w26cgrid_propgrid*.in and w26cgrid_born.in; here the doc rules
 * and the StyleError paths are checked directly. The grid cell values are compared
 * against a native LAMMPS dump grid probe (plans/scratch/w26cgrid/probe_grid.in).
 */

const io = { emit: () => {}, writeFile: () => {} };

const run = async (text: string): Promise<Session> => {
  const s = new Session(io);
  await s.execute(`${text}\n`);
  return s;
};

const grid = (id: string, args: string) => `units lj
atom_style atomic
region box block 0 2 0 4 0 6
create_box 1 box
create_atoms 1 single 1.0 2.0 3.0
mass 1 1.0
compute ${id} all property/grid ${args}
run 0
`;

describe('compute property/grid', () => {
  it('reproduces native cell ids, indices, corners, centers and scaled coords (3d)', async () => {
    const s = await run(grid('pg', '2 2 3 id proc ix iy iz x y z xs ys zs xc yc zc xsc ysc zsc'));
    const pg = s.sys.compute('pg') as ComputePropertyGrid;
    expect(pg.nx).toBe(2);
    expect(pg.ny).toBe(2);
    expect(pg.nz).toBe(3);
    expect(pg.ncell).toBe(12);
    expect(pg.ncols).toBe(17);
    // column order is the input order
    const col = (name: string) => pg.attrs.indexOf(name as never);
    const val = (cell: number, name: string) => pg.gridValue(cell, col(name));
    // native dump grid row 1: 1 1 1 1 0 0 0 0.5 1 1 0 0 0 0.25 0.25 0.1666667 0
    expect(val(0, 'id')).toBe(1);
    expect(val(0, 'ix')).toBe(1);
    expect(val(0, 'iy')).toBe(1);
    expect(val(0, 'iz')).toBe(1);
    expect(val(0, 'x')).toBe(0);
    expect(val(0, 'y')).toBe(0);
    expect(val(0, 'z')).toBe(0);
    expect(val(0, 'xc')).toBeCloseTo(0.5, 12);
    expect(val(0, 'yc')).toBeCloseTo(1, 12);
    expect(val(0, 'zc')).toBeCloseTo(1, 12);
    expect(val(0, 'xsc')).toBeCloseTo(0.25, 12);
    expect(val(0, 'ysc')).toBeCloseTo(0.25, 12);
    expect(val(0, 'zsc')).toBeCloseTo(1 / 6, 12);
    // row 2 (x fastest): id 2, ix 2, corner (1,0,0), center (1.5,1,1)
    expect(val(1, 'id')).toBe(2);
    expect(val(1, 'ix')).toBe(2);
    expect(val(1, 'x')).toBe(1);
    expect(val(1, 'xc')).toBeCloseTo(1.5, 12);
    // row 5 (z slowest): id 5, iz 2, corner (0,0,2)
    expect(val(4, 'id')).toBe(5);
    expect(val(4, 'iz')).toBe(2);
    expect(val(4, 'z')).toBe(2);
    expect(val(4, 'zc')).toBeCloseTo(3, 12);
    expect(val(4, 'zs')).toBeCloseTo(1 / 3, 12);
    // proc is 0 in the single-process engine
    for (let c = 0; c < 12; c++) expect(val(c, 'proc')).toBe(0);
  });

  it('2d grid uses Nz = 1 and the x-fastest id order', async () => {
    const s = await run(`units lj
atom_style atomic
dimension 2
region box block 0 2 0 2 -0.5 0.5
create_box 1 box
create_atoms 1 single 1.0 1.0 0.0
mass 1 1.0
compute pg all property/grid 2 3 1 id ix iy xc yc
run 0
`);
    const pg = s.sys.compute('pg') as ComputePropertyGrid;
    expect(pg.ncell).toBe(6);
    expect(pg.gridValue(0, 0)).toBe(1);
    expect(pg.gridValue(1, 0)).toBe(2);
    expect(pg.gridValue(2, 0)).toBe(3);
    expect(pg.gridValue(2, 1)).toBe(1);
    expect(pg.gridValue(2, 2)).toBe(2);
    expect(pg.gridValue(1, 3)).toBeCloseTo(1.5, 12);
    expect(pg.gridValue(1, 4)).toBeCloseTo(1 / 3, 12);
  });

  it('rejects unknown attributes and Z attributes in 2d', async () => {
    await expect(run(grid('pg', '2 2 3 bogus'))).rejects.toThrow(/unknown attribute 'bogus'/);
    await expect(run(grid('pg', '2 2 3 id nope'))).rejects.toThrow(/unknown attribute/);
    await expect(run(`units lj
atom_style atomic
dimension 2
region box block 0 2 0 2 -0.5 0.5
create_box 1 box
compute pg all property/grid 2 2 1 z
run 0
`)).rejects.toThrow(/refers to the Z dimension/);
  });

  it('rejects a 2d grid with Nz != 1 and non-integer sizes', async () => {
    await expect(run(`units lj
atom_style atomic
dimension 2
region box block 0 2 0 2 -0.5 0.5
create_box 1 box
compute pg all property/grid 2 2 2 id
run 0
`)).rejects.toThrow(/Nz must be 1/);
    await expect(run(grid('pg', '2.5 2 3 id'))).rejects.toThrow(/positive integer/);
  });
});

describe('compute born/matrix', () => {
  const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'oracle', 'w26cgrid_born.json'), 'utf8')) as {
    thermo: Record<string, number>[];
  };
  const native = fx.thermo[0];
  const INPUT = readFileSync(join(__dirname, 'oracle', 'w26cgrid_born.in'), 'utf8');

  it('analytic lj/cut matches native LAMMPS', async () => {
    const s = await run(INPUT);
    const ba = s.sys.compute('ba') as ComputeBornMatrix;
    const v = ba.vectorValues();
    expect(v.length).toBe(21);
    for (let i = 0; i < 21; i++) {
      expect(v[i], `c_ba[${i + 1}]`).toBeCloseTo(native[`c_ba[${i + 1}]`], 10);
    }
  });

  it('numdiff finite differences match the analytic Born matrix', async () => {
    const s = await run(INPUT);
    const bm = s.sys.compute('bm') as ComputeBornMatrix;
    const ba = s.sys.compute('ba') as ComputeBornMatrix;
    const vm = bm.vectorValues();
    const va = ba.vectorValues();
    for (let i = 0; i < 21; i++) {
      const scale = Math.max(1, Math.abs(va[i]));
      expect(Math.abs(vm[i] - va[i]) / scale, `c_bm[${i + 1}]`).toBeLessThan(1e-7);
    }
    // and native's own numdiff fixture
    for (let i = 0; i < 21; i++) {
      expect(vm[i], `c_bm[${i + 1}]`).toBeCloseTo(native[`c_bm[${i + 1}]`], 8);
    }
  });

  it('rejects unknown keywords, numdiff combinations and non-lj/cut analytic styles', async () => {
    await expect(run(`${INPUT.replace('compute         ba all born/matrix', 'compute ba all born/matrix bogus')}`)).rejects.toThrow(/unknown keyword 'bogus'/);
    await expect(run(`${INPUT.replace('compute         ba all born/matrix', 'compute ba all born/matrix numdiff 1.0e-6 myvirial bond')}`)).rejects.toThrow(/numdiff needs exactly two arguments/);
    await expect(run(`${INPUT.replace('compute         ba all born/matrix', 'compute ba all born/matrix bond')}`)).rejects.toThrow(/'bond' term is not supported/);
    await expect(run(`units lj
atom_style atomic
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 single 1 1 1
mass 1 1.0
pair_style lj/cut/coul/cut 2.5 2.5
pair_coeff 1 1 1.0 1.0 2.5
compute ba all born/matrix
thermo_style custom step c_ba[1]
run 0
`)).rejects.toThrow(/only implemented for pair style lj\/cut/);
  });

  it('rejects a numdiff virial-ID that is not a pressure compute', async () => {
    await expect(run(`${INPUT.replace('compute         myvirial all pressure NULL virial', 'compute myvirial all pe')}`)).rejects.toThrow(/is not a pressure compute/);
  });
});
