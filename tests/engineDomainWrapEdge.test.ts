import { describe, expect, it } from 'vitest';
import { Geometry, makeBox } from '../src/engine/domain';

/*
 * Periodic wrap of an atom just below the lower box face (domain.cpp Domain::pbc() at b332f28: "if (x[i][0] < lo[0])
 * x[i][0] += period[0]"). Measured with native LAMMPS (black box, 14 x 14 lattice 0.1 in the box -0.7..0.7, write_data
 * after run 0): the atom at x = -0.7000000000000001 is written at 0.6999999999999998 with image -1.
 */
describe('periodic wrap at the box edge', () => {
  it('wraps a coordinate just below lo and keeps it wrapped', () => {
    const g = new Geometry(makeBox({ lo: [-0.7, -0.7, -0.5], hi: [0.7, 0.7, 0.5] }));
    const x = Float64Array.from([-0.7000000000000001, -0.7000000000000001, 0]);
    const image = new Int32Array(3);
    g.remap(x, image, 0);
    expect(x[0]).toBe(0.6999999999999998);
    expect(x[1]).toBe(0.6999999999999998);
    expect(Array.from(image)).toEqual([-1, -1, 0]);
  });

  it('moves a coordinate on the upper face to the lower face', () => {
    const g = new Geometry(makeBox({ lo: [-0.7, -0.7, -0.5], hi: [0.7, 0.7, 0.5] }));
    const x = Float64Array.from([0.7, 0, 0]);
    const image = new Int32Array(3);
    g.remap(x, image, 0);
    expect(x[0]).toBe(-0.7);
    expect(image[0]).toBe(1);
  });
});
