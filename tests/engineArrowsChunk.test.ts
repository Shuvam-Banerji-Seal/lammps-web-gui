import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { ImageObject } from '../src/engine/output/image_objects';

/*
 * fix graphics/arrows chunk (src/engine-gpl/graphics/fix_graphics_arrows.ts; native src/GRAPHICS/fix_graphics_arrows.cpp,
 * FixGraphicsArrows::end_of_step, the CHUNK branch): one ARROW per chunk, centred on the per-chunk position array and
 * pointing along the per-chunk vector array, type 1, the length is |vector| * scale (or the autoscale value over the
 * mean vector length), shaft diameter 2 * radius, tip ratio 0.2. compute property/chunk coordN gives the bin centres of a
 * bin/3d compute chunk/atom (docs.lammps.org/compute_property_chunk.html: "coordN = coordinates for spatial bins").
 */

const SCRIPT = `units lj
atom_style molecular
boundary p p p
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 1 1 1
create_atoms 1 single 2 1 1
create_atoms 1 single 6 6 6
create_atoms 1 single 7 6 6
mass * 1.0
set atom 1 mol 1
set atom 2 mol 1
set atom 3 mol 2
set atom 4 mol 2
compute molchunk all chunk/atom molecule
compute cpos all com/chunk molchunk
compute cvec all com/chunk molchunk
fix vec all graphics/arrows 1 chunk molchunk cpos cvec 2 0.1
fix auto all graphics/arrows 1 chunk molchunk cpos cvec 2 0.1 autoscale 3.0
compute binchunk all chunk/atom bin/3d x lower 5.0 y lower 5.0 z lower 5.0
compute bpos all property/chunk binchunk coord1 coord2 coord3
`;

const newSession = async (extra = '') => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  await session.execute(SCRIPT + extra);
  return session;
};

const arrows = (objs: ImageObject[]) => objs.filter((o): o is Extract<ImageObject, { kind: 'arrow' }> => o.kind === 'arrow');

describe('fix graphics/arrows chunk mode', () => {
  it('draws one arrow per chunk: centre = per-chunk position, direction = per-chunk vector', async () => {
    const s = await newSession();
    const objs = arrows((s.sys.fix('vec').imageObjects?.() ?? []));
    expect(objs).toHaveLength(2);
    // the centre of mass of molecule 1 is (1.5, 1, 1) and of molecule 2 (6.5, 6, 6); the vectors are the same
    const c1 = objs.find((o) => o.center[0] < 3)!;
    const c2 = objs.find((o) => o.center[0] > 3)!;
    expect(c1.center).toEqual([1.5, 1, 1]);
    expect(c2.center).toEqual([6.5, 6, 6]);
    // scale 2: length = |vector| * 2; the direction is the vector itself
    expect(c1.dir).toEqual([1.5, 1, 1]);
    expect(c1.length).toBeCloseTo(Math.hypot(1.5, 1, 1) * 2, 12);
    expect(c2.length).toBeCloseTo(Math.hypot(6.5, 6, 6) * 2, 12);
    for (const o of objs) {
      expect(o.type).toBe(1);
      expect(o.diameter).toBe(0.2);
      expect(o.ratio).toBe(0.2);
    }
  });

  it('autoscale sets the mean arrow length to the autoscale value', async () => {
    const s = await newSession();
    const objs = arrows((s.sys.fix('auto').imageObjects?.() ?? []));
    expect(objs).toHaveLength(2);
    const mean = (objs[0].length + objs[1].length) / 2;
    expect(mean).toBeCloseTo(3.0, 12);
  });

  it('refuses a per-chunk compute that is missing', async () => {
    await expect(newSession('fix bad all graphics/arrows 1 chunk molchunk nope cvec 2 0.1\n')).rejects.toThrow(/nope/);
  });
});

describe('compute property/chunk coordN (bin centres of bin/3d)', () => {
  it('gives every bin centre once, in the bins of compute chunk/atom bin/3d', async () => {
    const s = await newSession();
    const pos = s.sys.compute('bpos').arrayValues();
    // a 10 x 10 x 10 box in bins of 5: 2 x 2 x 2 bins whose centres are 2.5 or 7.5 in each dimension
    expect(s.sys.compute('bpos').sizeArrayCols).toBe(3);
    const seen = new Set<string>();
    for (let c = 0; c < pos.length / 3; c++) {
      const key = [pos[3 * c], pos[3 * c + 1], pos[3 * c + 2]].join(',');
      seen.add(key);
      for (let d = 0; d < 3; d++) expect([2.5, 7.5]).toContain(pos[3 * c + d]);
    }
    expect(seen.size).toBe(8);
  });

  it('refuses coord3 for a 2d binning (native: "stores no coord3")', async () => {
    const s = new Session({ emit: () => {}, writeFile: () => {} });
    await expect(s.execute(`units lj
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
compute b2 all chunk/atom bin/2d x lower 5.0 y lower 5.0
compute p2 all property/chunk b2 coord1 coord3
`)).rejects.toThrow(/coord3/);
  });
});
