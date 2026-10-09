import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix wall/gran/region with side-out regions and compound regions that have a
 * side-out member (docs.lammps.org/fix_wall_gran_region.html,
 * docs.lammps.org/region.html).  A side-out primitive contributes the nearest
 * point of the solid ("The distance between a particle and the region boundary
 * is the distance to the nearest point on the region surface", the docs quote);
 * a compound region filters sub-region contacts (region.html: "LAMMPS discards
 * points that are part of multiple sub-regions when calculating wall/particle
 * interactions, to avoid double-counting the interaction").
 *
 * Exact parity with native LAMMPS is in tests/oracle/w36wgr_*.in; the numbers
 * below were measured with native LAMMPS (black box).
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  return { error, files };
};

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Single-sphere input; the force on atom 1 after run 0 (fix setup force). */
const sphereInput = (fix: string, region: string, x: number, y: number, z: number, diameter = 1.0): string => `units lj
atom_style sphere
atom_modify map array
comm_modify vel yes
boundary f f f
region box block -8 8 -8 8 -8 8 units box
create_box 1 box
${region}
create_atoms 1 single ${x} ${y} ${z} units box
set type 1 diameter ${diameter} density 1.0
pair_style none
fix 1 all nve/sphere
${fix}
run 0
`;

const forceOn = async (text: string): Promise<number[]> => {
  const { error, files } = await runScript(`${text}\nwrite_dump all custom wg.dump id fx fy fz modify format float %.17g sort id\n`);
  if (error) throw new Error(message(error));
  const lines = files.get('wg.dump')!.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return lines[k + 1].trim().split(/\s+/).slice(1).map(Number);
};

const HOOKE = 'fix 2 all wall/gran/region hooke 100.0 50.0 0.0 0.0 0.0 1 region reg';
const HERTZ = 'fix 2 all wall/gran/region hertz/history 100.0 50.0 0.0 0.0 0.0 1 region reg';

describe('fix wall/gran/region on a side-out block', () => {
  const REG = 'region reg block 0 2 0 2 0 2 side out units box';
  // hooke: F = Kn * delta along (particle - nearest point of the solid); the
  // nearest point is a face point (20,0,0), an edge point (15.3553390593274
  // each of x,y) or a corner point (8.86751345948127 each of x,y,z).
  it('hooke: face, edge and corner contacts of the outer surface', async () => {
    const face = await forceOn(sphereInput(HOOKE, REG, 2.3, 1.0, 1.0));
    expect(face[0]).toBeCloseTo(20, 9);
    expect(Math.abs(face[1])).toBeLessThan(1e-9);
    expect(Math.abs(face[2])).toBeLessThan(1e-9);
    const edge = await forceOn(sphereInput(HOOKE, REG, 2.2, 2.2, 1.0));
    expect(edge[0]).toBeCloseTo(15.3553390593274, 9);
    expect(edge[1]).toBeCloseTo(15.3553390593274, 9);
    expect(Math.abs(edge[2])).toBeLessThan(1e-9);
    const corner = await forceOn(sphereInput(HOOKE, REG, 2.2, 2.2, 2.2));
    for (const c of corner) expect(c).toBeCloseTo(8.86751345948127, 9);
  });

  it('hertz/history: block faces, edges and corners are flat (R_eff = R)', async () => {
    // delta = 0.2, R = 0.5: F = Kn delta sqrt(delta R) = 100 * 0.2 * sqrt(0.1)
    const face = await forceOn(sphereInput(HERTZ, REG, 2.3, 1.0, 1.0));
    expect(face[0]).toBeCloseTo(6.324555320336759, 9);
    const edge = await forceOn(sphereInput(HERTZ, REG, 2.2, 2.2, 1.0));
    expect(edge[0]).toBeCloseTo(5.05977979907876, 9);
    const corner = await forceOn(sphereInput(HERTZ, REG, 2.2, 2.2, 2.2));
    expect(corner[0]).toBeCloseTo(2.457356127702879, 9);
  });

  it('no contact for a particle inside the solid (outside the region)', async () => {
    const c = await forceOn(sphereInput(HOOKE, REG, 1.0, 1.0, 1.0));
    expect(c).toEqual([0, 0, 0]);
  });
});

describe('fix wall/gran/region on a side-out sphere', () => {
  const REG = 'region reg sphere 0 0 0 1 side out units box';
  it('hooke: the outer surface, normal along the radius', async () => {
    const f = await forceOn(sphereInput(HOOKE, REG, 1.3, 0.0, 0.0));
    expect(f[0]).toBeCloseTo(20, 9);
    const g = await forceOn(sphereInput(HOOKE, REG, 0.0, 0.0, -1.4));
    expect(g[2]).toBeCloseTo(-10, 9);
  });
  it('hertz/history: R_eff = R Rw / (R + Rw) with Rw = +R (convex outer surface)', async () => {
    const f = await forceOn(sphereInput(HERTZ, REG, 1.3, 0.0, 0.0));
    expect(f[0]).toBeCloseTo(5.163977794943222, 9);
  });
});

describe('fix wall/gran/region on a side-out cylinder', () => {
  const REG = 'region reg cylinder z 0 0 1.0 -1 1 side out units box';
  it('hertz/history: Rw = +2 Rc at the lateral surface, 0 at the cap', async () => {
    const lat = await forceOn(sphereInput(HERTZ, REG, 1.3, 0.0, 0.0));
    expect(lat[0]).toBeCloseTo(5.656854249492381, 9);
    const cap = await forceOn(sphereInput(HERTZ, REG, 0.0, 0.0, 1.3));
    expect(cap[2]).toBeCloseTo(6.324555320336759, 9);
  });
});

describe('fix wall/gran/region on compound regions with a side-out member', () => {
  it('intersect of a side-in cylinder and a side-out block: the inner obstacle face', async () => {
    // block [2.5,4]x[-0.5,0.5]x[-0.5,0.5] side out inside a cylinder R = 3;
    // the particle at (2.8,0.6,0) feels the block edge (+7.94733192202057,
    // +2.64911064067352) and the cylinder wall (-25.7713723398736,
    // -5.523... -> total measured with native LAMMPS)
    const text = sphereInput(
      'fix 2 all wall/gran/region hooke 100.0 50.0 0.0 0.0 0.0 1 region I',
      'region cyl cylinder z 0 0 3.0 -4 4 side in units box\nregion blk block 2.5 4 -0.5 0.5 -0.5 0.5 side out units box\nregion I intersect 2 cyl blk',
      2.8, 0.6, 0.0, 0.8,
    );
    const f = await forceOn(text);
    expect(f[0]).toBeCloseTo(-25.7713723398736, 8);
    expect(f[1]).toBeCloseTo(24.4775630700271, 8);
  });

  it('intersect of a side-in cylinder and a side-out block: only the block at the centre', async () => {
    const text = sphereInput(
      'fix 2 all wall/gran/region hooke 100.0 50.0 0.0 0.0 0.0 1 region I',
      'region cyl cylinder z 0 0 3.0 -4 4 side in units box\nregion blk block -0.5 0.5 -0.5 0.5 -0.5 0.5 side out units box\nregion I intersect 2 cyl blk',
      0.8, 0.0, 0.0, 0.8,
    );
    const f = await forceOn(text);
    expect(f[0]).toBeCloseTo(10, 8);
    expect(Math.abs(f[1])).toBeLessThan(1e-9);
    expect(Math.abs(f[2])).toBeLessThan(1e-9);
  });

  it('union keeps a side-out member face that no other member covers', async () => {
    const text = sphereInput(
      'fix 2 all wall/gran/region hooke 100.0 50.0 0.0 0.0 0.0 1 region U',
      'region blk block 0 2 0 2 0 2 side out units box\nregion far block 20 22 20 22 0 2 side in units box\nregion U union 2 blk far',
      2.3, 1.0, 1.0,
    );
    const f = await forceOn(text);
    expect(f[0]).toBeCloseTo(20, 8);
  });
});

describe('fix wall/gran/region side-out restrictions', () => {
  it('a side-out plane region is a StyleError naming the style', async () => {
    const text = sphereInput(
      'fix 2 all wall/gran/region hooke 100.0 50.0 0.0 0.0 0.0 1 region reg',
      'region reg plane 1.0 1.0 1.0 1.0 1.0 1.0 side out units box',
      1.6, 1.5, 1.7,
    );
    const { error } = await runScript(text);
    expect(message(error)).toMatch(/side-out wall/);
  });
});
