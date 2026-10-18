import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { BlockRegion } from '../src/engine/region';

/*
 * region INF bounds: docs.lammps.org/region.html says "INF means a large negative or positive number (1.0e20)", but the
 * source uses 1.0e200 (src/region_block.cpp at b332f28: "static constexpr double BIG = 1.0e200;", the same in
 * region_cylinder.cpp, region_cone.cpp, region_prism.cpp and region_ellipsoid.cpp). The engine follows the source.
 */
describe('region INF bounds', () => {
  it('are +-1e200 as native region_block.cpp sets them', async () => {
    const s = new Session({ emit: () => {}, writeFile: () => {} });
    await s.execute('units lj\nregion box block 0 4 0 4 0 4\ncreate_box 1 box\nregion half block INF 2 INF INF 1 INF units box\n');
    const r = (s as unknown as { sys: { region(id: string): BlockRegion } }).sys.region('half');
    expect(r.bounds()).toEqual([-1e200, 2, -1e200, 1e200, 1, 1e200]);
    expect(r.match(-3, 1, 2)).toBe(true);
    expect(r.match(3, 1, 2)).toBe(false);
  });

  it('need an existing box, with the native message', async () => {
    let err = '';
    const s = new Session({ emit: (e) => { if (e.kind === 'error') err ||= e.message; }, writeFile: () => {} });
    try { await s.execute('units lj\nregion box block 0 4 INF INF 0 4\ncreate_box 1 box\n'); } catch (e) { err ||= String(e); }
    expect(err).toMatch(/Cannot use region INF or EDGE when box does not exist/);
  });
});
