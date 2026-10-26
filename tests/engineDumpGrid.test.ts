import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/session';

/*
 * dump grid and dump grid/vtk (GPL port, src/engine-gpl/dump/dump_grid.ts), the per-grid references of src/engine/grid.ts
 * and the per-grid inputs of fix ave/grid: native's messages for wrong references, sizes and settings. Native parity of
 * the files is in the oracle cases w420dump_grid, w421dump_grid_ttm, w422dump_grid_tri, w423dump_grid_2d and
 * w424avegrid_pergrid.
 */

const files = new Map<string, string>();
const run = async (text: string): Promise<Session> => {
  const s = new Session({ emit: () => {}, writeFile: (name, body) => { files.set(name, body); } });
  await s.execute(`${text}\n`);
  return s;
};

const BASE = `units lj
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 4127 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 all nve
compute pg all property/grid 2 2 2 id xc yc
compute pv all property/grid 2 2 2 id
compute p3 all property/grid 3 2 2 id
fix ag all ave/grid 5 2 10 2 2 2 vx
`;

describe('dump grid references (grid.ts, utils.cpp check_grid_reference)', () => {
  it('names the wrong part of a reference', async () => {
    const bad = async (dump: string) => run(`${BASE}${dump}\nrun 10`);
    await expect(bad('dump 1 all grid 10 g.dump c_nope:grid:data')).rejects.toThrow('Dump grid compute ID nope not found');
    await expect(bad('dump 1 all grid 10 g.dump c_pg:mesh:data[1]')).rejects.toThrow('Dump grid compute pg does not recognize grid name mesh');
    await expect(bad('dump 1 all grid 10 g.dump c_pg:grid:vals[1]')).rejects.toThrow('Dump grid compute pg does not recognize data name vals');
    await expect(bad('dump 1 all grid 10 g.dump c_pg:grid:data')).rejects.toThrow('Dump grid compute pg data data is not per-grid vector');
    await expect(bad('dump 1 all grid 10 g.dump c_pv:grid:data[1]')).rejects.toThrow('Dump grid compute pv data data is not per-grid array');
    await expect(bad('dump 1 all grid 10 g.dump c_pg:grid:data[4]')).rejects.toThrow('Dump grid compute pg array data is accessed out-of-range');
    await expect(bad('dump 1 all grid 10 g.dump c_pg:grid')).rejects.toThrow("Grid ID pg:grid must be 3 strings separated by 2 ':'characters");
    await expect(bad('dump 1 all grid 10 g.dump c_pv:grid:data xyz')).rejects.toThrow('Invalid attribute xyz in dump grid command');
    await expect(bad('dump 1 all grid 5 g.dump f_ag:grid:data')).rejects.toThrow('Dump grid fix ag not computed at compatible time');
    await expect(bad('dump 1 all grid 10 g.dump c_pv:grid:data c_p3:grid:data')).rejects.toThrow('Dump grid field grid sizes do not match');
    await expect(bad('compute ke all ke/atom\ndump 1 all grid 10 g.dump c_ke:grid:data')).rejects.toThrow('Dump grid compute ke does not compute per-grid info');
    await expect(bad('dump 1 all grid 10 g.dump')).rejects.toThrow('No dump grid arguments specified');
    await expect(bad('dump 1 all grid 10 g.dump c_pv:grid:data\ndump_modify 1 thresh c_pv:grid:data > 1')).rejects.toThrow('Unknown dump_modify keyword: thresh');
  });

  it('writes one line per cell after the grid header, wildcards expanded, sort by a column', async () => {
    files.clear();
    await run(`${BASE}dump 1 all grid 10 g.dump c_pg:grid:data[*]\ndump_modify 1 sort -1\nrun 0`);
    const text = files.get('g.dump')!;
    const lines = text.trim().split('\n');
    expect(lines.slice(0, 2)).toEqual(['ITEM: TIMESTEP', '0']);
    expect(lines).toContain('ITEM: GRID SIZE nx ny nz');
    expect(lines).toContain('ITEM: GRID CELLS c_pg:grid:data[1] c_pg:grid:data[2] c_pg:grid:data[3]');
    const cells = lines.slice(lines.indexOf('ITEM: GRID CELLS c_pg:grid:data[1] c_pg:grid:data[2] c_pg:grid:data[3]') + 1);
    expect(cells.map((l) => Number(l.split(' ')[0]))).toEqual([8, 7, 6, 5, 4, 3, 2, 1]);
  });
});

describe('dump grid/vtk settings (dump_grid_vtk.cpp)', () => {
  it('needs one or three fields, a "*" file name, sorting on IDs and an orthogonal box', async () => {
    await expect(run(`${BASE}dump 1 all grid/vtk 10 g.*.vtr c_pg:grid:data[1] c_pg:grid:data[2]\nrun 0`)).rejects.toThrow('Dump grid/vtk requires one or three fields');
    await expect(run(`${BASE}dump 1 all grid/vtk 10 g.vtr c_pv:grid:data\nrun 0`)).rejects.toThrow('Dump grid/vtk requires one snapshot per file');
    await expect(run(`${BASE}dump 1 all grid/vtk 10 g.*.vtr c_pv:grid:data\ndump_modify 1 sort off\nrun 0`)).rejects.toThrow('Dump grid/vtk requires sorting on IDs');
    await expect(run(`${BASE.replace('fix ag all ave/grid 5 2 10 2 2 2 vx\n', '')}change_box all triclinic\ndump 1 all grid/vtk 10 g.*.vtr c_pv:grid:data\nrun 0`)).rejects.toThrow('Dump grid/vtk does not support triclinic simulation boxes');
    await expect(run(`${BASE}dump 1 all grid/vtk 10 g.*.vtr c_pv:grid:data\ndump_modify 1 region all`)).rejects.toThrow('Unknown dump_modify keyword: region');
  });
});

describe('fix ave/grid per-grid inputs (fix_ave_grid.cpp)', () => {
  it('stops on mixed per-atom and per-grid values, other grid sizes and the count of a per-grid average', async () => {
    await expect(run(`${BASE}fix a2 all ave/grid 1 1 1 2 2 2 vx c_pv:grid:data`)).rejects.toThrow('Fix ave/grid cannot operate on per-atom and per-grid values together');
    await expect(run(`${BASE}fix a2 all ave/grid 1 1 1 2 2 2 c_p3:grid:data`)).rejects.toThrow('Fix ave/grid value grid sizes do not match');
    await expect(run(`${BASE}fix a2 all ave/grid 2 1 10 2 2 2 f_ag:grid:data`)).rejects.toThrow('Fix ave/grid fix ag not computed at compatible time');
    await expect(run(`${BASE}fix a2 all ave/grid 10 1 10 2 2 2 f_ag:grid:data\ndump 1 all grid 10 g.dump f_a2:grid:count\nrun 0`)).rejects.toThrow('Dump grid fix a2 does not recognize data name count');
  });

  it('averages a per-grid compute over the samples', async () => {
    const s = await run(`${BASE}fix a2 all ave/grid 1 4 4 2 2 2 c_pg:grid:data[2]\nrun 4`);
    const f = s.sys.fix('a2');
    const data = f.griddata(0);
    const pg = s.sys.compute('pg');
    pg.pergridValues();
    const xc = pg.griddata(0);
    for (let c = 0; c < 8; c++) expect(data[c]).toBeCloseTo(xc[c * 3 + 1], 12);
  });
});
