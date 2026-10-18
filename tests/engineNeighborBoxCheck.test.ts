import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * The neighbor-list distance check while the box changes (src/engine/neighbor.ts movedTooFar), as native LAMMPS
 * does it (src/neighbor.cpp at b332f28, check_distance): the threshold 0.5 (skin - (delta1 + delta2)) shrinks with
 * the displacement of the lower and upper box corners (orthogonal) or of the two corners the corner loop keeps
 * (triclinic). fix deform remap v kicks on the rebuild steps, so it asks the same check.
 *
 * Measured with native LAMMPS: 256-atom LJ fcc, neighbor 0.3 bin, neigh_modify every 1 delay 0 check yes, 200 steps;
 * the log's "Neighbor list builds" is 29 (npt iso), 20 (deform x y z erate, remap x), 46 (triclinic deform xy and xz
 * erate, remap v), 33 (deform x delta 0 1.5, only the upper face moves) and 21 (nvt, a fixed box). The engine's
 * nbuild also counts the build of the run setup, one more.
 */

const HEAD = (region: string) => `units lj
atom_style atomic
lattice fcc 0.8442
region box ${region}
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.5 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
`;
const ORTHO = HEAD('block 0 4 0 4 0 4');
const TRI = HEAD('prism 0 4 0 4 0 4 0.5 0.3 0.2');

const builds = async (script: string): Promise<number> => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  await session.execute(script);
  return (session as unknown as { sys: { nb: { nbuild: number } } }).sys.nb.nbuild - 1;
};

describe('neighbor rebuilds while the box changes (native check_distance)', () => {
  it.each([
    ['npt iso', `${ORTHO}fix 1 all npt temp 1.5 1.5 0.5 iso 6.0 6.0 1.0\nrun 200\n`, 29],
    ['deform x y z erate, remap x', `${ORTHO}fix 1 all nve\nfix 2 all deform 1 x erate 0.05 y erate 0.05 z erate 0.05 remap x\nrun 200\n`, 20],
    ['triclinic deform xy xz erate, remap v', `${TRI}fix 1 all nve\nfix 2 all deform 1 xy erate 0.2 xz erate 0.1 remap v\nrun 200\n`, 46],
    ['deform x delta 0 1.5 (upper face only)', `${ORTHO}fix 1 all nve\nfix 2 all deform 1 x delta 0 1.5 remap x\nrun 200\n`, 33],
    ['nvt, fixed box', `${ORTHO}fix 1 all nvt temp 1.5 1.5 0.5\nrun 200\n`, 21],
  ])('%s: native build count', async (_name, script, native) => {
    expect(await builds(script)).toBe(native);
  }, 120000);
});
